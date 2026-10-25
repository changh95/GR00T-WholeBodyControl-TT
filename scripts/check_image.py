"""T-3.2: static checks of the X280 monitor image before it may go near a hart (PLAN.md WP-3.2, §2 M3).

    make -C $RT/x280/fw x280 && simpy $RT/scripts/check_image.py $RT/x280/fw/build/monitor.elf

Checks (each prints PASS/FAIL; exit 1 on any FAIL):
  image_size      loadable bytes of monitor.elf (alloc sections with contents) <= 48 KiB; monitor.bin agrees
  entry           e_entry == _start == 0x0800_0000 (the reset handler address, §5.2 boot step 5)
  sections_lim    every SHF_ALLOC section inside LIM [0x0800_0000, 0x081E_0000); the monitor's below the
                  mailbox 0x0801_0000; payloads at or above 0x0801_1000
  undefined       no undefined symbols in any of the ELFs
  park_stub       park_stub.elf: park_stub at 0x2001_0100, code <= 48 B, park_spin at +48, all in 64 B scratch
  stage0          stage0.elf: stage0_start at 0x2001_0100, <= 64 B
  csr_smrnmi      no CSR instruction touches 0x740-0x744 (standard Smrnmi numbers; the X280 uses
                  0x350-0x353, tt-isa-documentation BlackholeA0/L2CPUTile/RNMIs.md:6)
  mnret           every mnret-pattern word is exactly 0x70200073 (RNMIs.md:31) and no source file uses
                  the `mnret` mnemonic (assemblers disagree on it)
  rnmi_rearm      the X280 image contains csrsi 0x353, 8 (0x35346073) and csrwi 0x7c1, 0 (0x7c105073)
  hart0_fence_i   crt0 runs fence.i (0x0000100F) between _start and hart 0's first call (mon_entry): after E1F hart 0's
                  L1I still holds stage 0's GP-scratch line until its own fence.i (Caches.md:18-22; crt0.S)
  priv_insns      only ecall/ebreak/wfi/mret/mnret among SYSTEM funct3=0 words
  crc32_tab       mon_crc32's slicing-by-8 tables (mon_crc32_tab, x280/fw/crc32_tab.h): in .rodata, 64 B aligned,
                  8 KiB, every entry equal to its zlib definition (INFO when the image was built -DMON_CRC_BITWISE)
  weight_blob     the real packed blob (x280/tools/pack_weights.py, fp16, both policies) ends <= 0x081E_0000
                  when placed at 0x0801_1000; fp32 single-policy blobs (DP4) are reported too
  amo             no A-extension instruction (major opcode 0x2F: LR, SC, AMO*, vector AMOs) in any of the ELFs: LIM is
                  not data-cacheable and the A extension is implemented at L1D (Caches.md:26-34)
  mh_layout       harts 1-3 (mh.h): mh_sec_rnmi_exc = MH_SEC_RNMI_EXC 0x0800_0200 holding `wfi; j .`; the NOLOAD .mh
                  region exactly [0x0800_A000, 0x0800_C000), after .persist and below the stack (SPIKE_HOST: INFO)
  mh_bench        the 4-hart bench payload: entry 0x081D_F400, code below its data lines 0x081D_FE40, image ending at
                  0x081D_FF80 (the E1F entry-word line)
The disassembly is this script's own RV64GC length decoder (no pyelftools in the sim venv); CSR numbers are
read from the instruction word, so a `.word`-encoded instruction is seen like any other.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

RT = Path(__file__).resolve().parents[1]
FW = RT / "x280" / "fw"
TOOLS = RT / "x280" / "tools"

LIM_BASE, LIM_END = 0x0800_0000, 0x081E_0000      # MemoryMap.md:84; Caches.md:73, 92
MH_SEC_RNMI_EXC, MH_BASE, MH_END = 0x0800_0200, 0x0800_A000, 0x0800_C000   # x280/fw/mh.h, asm.h, lim.ld
MHB_BASE, MHB_DATA, MHB_END = 0x081D_F400, 0x081D_FE40, 0x081D_FF80       # x280/fw/mh.h, mh_bench.ld
WFI, J_SELF = 0x10500073, 0xFFDFF06F                                       # wfi; jal x0, -4
CODE_END, PAYLOAD_BASE = 0x0801_0000, 0x0801_1000  # mailbox.h (§1.4.4)
IMAGE_MAX = 48 * 1024
GP_SCRATCH, GP_SCRATCH_LEN = 0x2001_0100, 64       # MemoryMap.md:113
SMRNMI_CSRS = range(0x740, 0x745)
MNRET = 0x70200073
REARM = 0x35346073                                 # csrrsi x0, 0x353, 8
CHICKEN = 0x7C105073                               # csrrwi x0, 0x7c1, 0
FENCE_I = 0x0000100F                               # fence.i
PRIV_OK = {0x00000073: "ecall", 0x00100073: "ebreak", 0x10500073: "wfi", 0x30200073: "mret", MNRET: "mnret"}

SHF_ALLOC, SHF_EXECINSTR = 0x2, 0x4
SHT_SYMTAB, SHT_NOBITS = 2, 8


class Elf:
    """Just enough ELF64-LE: sections, symbols, program headers."""

    def __init__(self, path: Path):
        self.path = Path(path)
        d = self.data = self.path.read_bytes()
        if d[:4] != b"\x7fELF" or d[4] != 2 or d[5] != 1:
            raise ValueError(f"{path}: not an ELF64 little-endian file")
        (self.e_type, self.e_machine, _, self.e_entry, e_phoff, e_shoff, _, _, e_phentsize, e_phnum,
         e_shentsize, e_shnum, e_shstrndx) = struct.unpack_from("<HHIQQQIHHHHHH", d, 16)
        self.sections = []
        for i in range(e_shnum):
            name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from(
                "<IIQQQQIIQQ", d, e_shoff + i * e_shentsize)
            self.sections.append(dict(idx=i, name_off=name, type=typ, flags=flags, addr=addr, off=off,
                                      size=size, link=link))
        shstr = self.sections[e_shstrndx]
        for s in self.sections:
            s["name"] = self._str(shstr["off"] + s["name_off"])
        self.phdrs = []
        for i in range(e_phnum):
            p_type, p_flags, p_off, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack_from(
                "<IIQQQQQQ", d, e_phoff + i * e_phentsize)
            self.phdrs.append(dict(type=p_type, paddr=p_paddr, vaddr=p_vaddr, filesz=p_filesz, memsz=p_memsz))
        self.symbols = []
        for s in self.sections:
            if s["type"] != SHT_SYMTAB:
                continue
            strtab = self.sections[s["link"]]
            for j in range(1, s["size"] // 24):
                st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
                    "<IBBHQQ", d, s["off"] + j * 24)
                self.symbols.append(dict(name=self._str(strtab["off"] + st_name), shndx=st_shndx,
                                         value=st_value, size=st_size, bind=st_info >> 4, type=st_info & 15))

    def _str(self, off: int) -> str:
        end = self.data.index(b"\0", off)
        return self.data[off:end].decode()

    def sym(self, name: str):
        for s in self.symbols:
            if s["name"] == name and s["shndx"] != 0:
                return s["value"]
        return None

    def alloc_sections(self):
        return [s for s in self.sections if s["flags"] & SHF_ALLOC and s["size"]]

    def exec_sections(self):
        return [s for s in self.sections if s["flags"] & SHF_EXECINSTR and s["type"] != SHT_NOBITS and s["size"]]

    def undefined(self):
        return sorted({s["name"] for s in self.symbols if s["shndx"] == 0 and s["name"]})

    def instructions(self):
        """(address, word, length) for every instruction of every executable section (RV64GC lengths)."""
        for s in self.exec_sections():
            raw = self.data[s["off"]:s["off"] + s["size"]]
            i = 0
            while i + 2 <= len(raw):
                h = raw[i] | (raw[i + 1] << 8)
                if h & 3 == 3:
                    if i + 4 > len(raw):
                        break
                    w = struct.unpack_from("<I", raw, i)[0]
                    yield s["addr"] + i, w, 4
                    i += 4
                else:
                    yield s["addr"] + i, h, 2
                    i += 2

    def load_span(self):
        """[lo, hi) of the bytes that objcopy -O binary emits: alloc sections that have file contents."""
        secs = [s for s in self.alloc_sections() if s["type"] != SHT_NOBITS]
        if not secs:
            return 0, 0
        return min(s["addr"] for s in secs), max(s["addr"] + s["size"] for s in secs)


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name: str, ok: bool, detail: str):
        self.rows.append(dict(name=name, ok=bool(ok), detail=detail))
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")

    def info(self, name: str, detail: str):
        self.rows.append(dict(name=name, ok=None, detail=detail))
        print(f"INFO {name}: {detail}")

    @property
    def failed(self):
        return [r for r in self.rows if r["ok"] is False]


def csr_accesses(elf: Elf):
    """{csr: [addresses]} for every Zicsr instruction (SYSTEM, funct3 != 0, != 4)."""
    out = {}
    for addr, w, n in elf.instructions():
        if n == 4 and (w & 0x7F) == 0x73 and ((w >> 12) & 7) not in (0, 4):
            out.setdefault(w >> 20, []).append(addr)
    return out


def priv_words(elf: Elf):
    return [(addr, w) for addr, w, n in elf.instructions()
            if n == 4 and (w & 0x7F) == 0x73 and ((w >> 12) & 7) == 0]


def words(elf: Elf):
    return {w for _, w, n in elf.instructions() if n == 4}


def amo_words(elf: Elf):
    """(address, word) of every A-extension instruction: major opcode 0x2F (LR, SC, AMO*, and the vector AMOs)."""
    return [(addr, w) for addr, w, n in elf.instructions() if n == 4 and (w & 0x7F) == 0x2F]


def fence_i_before_first_call(elf: Elf, start: int):
    """(a fence.i came before the first call, the address of that call) walking the instructions from `start`; a call is
    jal / jalr with rd = ra, or c.jalr."""
    seen = False
    for addr, w, n in elf.instructions():
        if addr < start:
            continue
        if n == 4 and w == FENCE_I:
            seen = True
        if (n == 4 and (w & 0x7F) in (0x6F, 0x67) and (w >> 7) & 0x1F == 1) or \
                (n == 2 and (w & 0xF07F) == 0x9002 and (w >> 7) & 0x1F):
            return seen, addr
    return seen, None


def word_at(elf: Elf, addr: int):
    for s in elf.exec_sections():
        if s["addr"] <= addr and addr + 4 <= s["addr"] + s["size"]:
            return struct.unpack_from("<I", elf.data, s["off"] + addr - s["addr"])[0]
    return None


MNEMONIC_RE = re.compile(r'(^\s*(\w+:\s*)?mnret\b)|("[^"\n]*\bmnret\b)')


def source_mnret_uses(fw_dir: Path):
    hits = []
    for p in sorted(fw_dir.glob("*")):
        if p.suffix not in (".S", ".s", ".c", ".h"):
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            code = line.split("//")[0]
            if p.suffix in (".S", ".s"):
                code = code.split("#")[0] if not code.lstrip().startswith("#") else ""
            if "/*" in code or code.lstrip().startswith("*"):
                continue
            if MNEMONIC_RE.search(code):
                hits.append(f"{p.name}:{i}")
    return hits


def crc32_tab_problem(elf: Elf):
    """None if mon_crc32_tab is absent (bitwise build); '' if it is right; else what is wrong. Entry [k][i] must be
    zlib's CRC register after byte i and k zero bytes: zlib.crc32(bytes([i]) + bytes(k), ~0) ^ ~0."""
    sym = next((s for s in elf.symbols if s["name"] == "mon_crc32_tab" and s["shndx"] != 0), None)
    if sym is None:
        return None
    sec = elf.sections[sym["shndx"]]
    if sec["name"] != ".rodata" or sec["type"] == SHT_NOBITS or sym["size"] != 8192 or sym["value"] % 64:
        return f"in {sec['name']}, {sym['size']} B at {sym['value']:#x} (want .rodata, 8192 B, 64 B aligned)"
    raw = elf.data[sec["off"] + sym["value"] - sec["addr"]:][:8192]
    got = struct.unpack("<2048I", raw)
    bad = [(k, i) for k in range(8) for i in range(256)
           if got[k * 256 + i] != zlib.crc32(bytes([i]) + bytes(k), 0xFFFFFFFF) ^ 0xFFFFFFFF]
    return f"{len(bad)} entries differ from zlib, first {bad[:4]}" if bad else ""


def pack_blob(dtype: str, policies: str | None, out: Path) -> tuple[int, str]:
    pw = TOOLS / "pack_weights.py"
    cmd = [sys.executable, str(pw), "--dtype", dtype, "--out", str(out)]
    if policies:
        cmd += ["--policies", policies]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(RT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(RT))
    if r.returncode != 0 or not out.exists():
        raise RuntimeError((r.stdout + r.stderr).strip()[-600:])
    return out.stat().st_size, r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("monitor_elf", type=Path)
    ap.add_argument("--park", type=Path, help="park_stub.elf (default: next to monitor.elf)")
    ap.add_argument("--stage0", type=Path, help="stage0.elf (default: next to monitor.elf)")
    ap.add_argument("--payloads", type=Path, help="directory of payload ELFs (default: <build>/payloads)")
    ap.add_argument("--blob", type=Path, help="check this packed blob instead of running pack_weights.py")
    ap.add_argument("--skip-blob", action="store_true", help="do not check the weight blob (reported as FAIL-free INFO)")
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    args = ap.parse_args(argv)

    rep = Report()
    mon = Elf(args.monitor_elf)
    build = args.monitor_elf.parent
    park = Elf(args.park or build / "park_stub.elf")
    stage0 = Elf(args.stage0 or build / "stage0.elf")
    pay_dir = args.payloads or build / "payloads"
    payloads = [Elf(p) for p in sorted(pay_dir.glob("*.elf"))] if pay_dir.is_dir() else []
    is_spike = mon.sym("sh_run") is not None
    rep.check("elf", mon.e_machine == 243 and mon.e_type == 2,
              f"{args.monitor_elf.name}: RISC-V ET_EXEC{' (SPIKE_HOST build)' if is_spike else ''}")

    # image size
    lo, hi = mon.load_span()
    size = hi - lo
    binp = args.monitor_elf.with_suffix(".bin")
    bin_note = ""
    if binp.exists():
        bsz = binp.stat().st_size
        bin_note = f"; {binp.name} {bsz} B"
        if bsz != size:
            rep.check("image_bin", False, f"{binp.name} is {bsz} B but the ELF's loadable span is {size} B (stale?)")
    rep.check("image_size", size <= IMAGE_MAX and lo == LIM_BASE,
              f"{size} B loadable from {lo:#010x} (limit {IMAGE_MAX} B){bin_note}")

    start = mon.sym("_start")
    rep.check("entry", mon.e_entry == LIM_BASE and start == LIM_BASE,
              f"e_entry {mon.e_entry:#010x}, _start {start if start is None else hex(start)}")

    # sections
    bad = []
    top = 0
    for s in mon.alloc_sections():
        end = s["addr"] + s["size"]
        top = max(top, end)
        if not (LIM_BASE <= s["addr"] and end <= CODE_END):
            bad.append(f"{s['name']} [{s['addr']:#x},{end:#x})")
    stack_top = mon.sym("__stack_top")
    rep.check("sections_lim", not bad and (stack_top is None or stack_top <= CODE_END),
              ("all alloc sections in [0x08000000, 0x08010000), last byte "
               f"{top:#010x}, stack top {stack_top:#010x}" if not bad and stack_top else f"outside: {bad}"))
    for p in payloads:
        pbad = [f"{s['name']} [{s['addr']:#x},{s['addr'] + s['size']:#x})" for s in p.alloc_sections()
                if not (PAYLOAD_BASE <= s["addr"] and s["addr"] + s["size"] <= LIM_END)]
        rep.check(f"sections_payload[{p.path.stem}]", not pbad,
                  f"in [0x08011000, 0x081E0000), entry {p.e_entry:#010x}" if not pbad else f"outside: {pbad}")

    # undefined symbols
    for e in [mon, park, stage0] + payloads:
        u = e.undefined()
        rep.check(f"undefined[{e.path.name}]", not u, "none" if not u else ", ".join(u))

    # park stub and stage 0
    ps, pe, pspin = park.sym("park_stub"), park.sym("park_stub_end"), park.sym("park_spin")
    psec = [s for s in park.alloc_sections()]
    pspan = (min(s["addr"] for s in psec), max(s["addr"] + s["size"] for s in psec)) if psec else (0, 0)
    ok = (ps == GP_SCRATCH and pe is not None and pe - ps <= 48 and pspin == GP_SCRATCH + 48 and
          pspan[0] >= GP_SCRATCH and pspan[1] <= GP_SCRATCH + GP_SCRATCH_LEN)
    rep.check("park_stub", ok, f"park_stub {ps and hex(ps)}, code {pe - ps if pe and ps else '?'} B (<= 48), "
              f"park_spin at +{pspin - ps if pspin and ps else '?'}, blob [{pspan[0]:#x},{pspan[1]:#x})")
    s0, s0e = stage0.sym("stage0_start"), stage0.sym("stage0_end")
    s0sec = stage0.alloc_sections()
    s0span = (min(s["addr"] for s in s0sec), max(s["addr"] + s["size"] for s in s0sec)) if s0sec else (0, 0)
    ok = (s0 == GP_SCRATCH and s0e is not None and s0e - s0 <= 64 and s0span[1] <= GP_SCRATCH + GP_SCRATCH_LEN)
    rep.check("stage0", ok, f"stage0_start {s0 and hex(s0)}, {s0e - s0 if s0 and s0e else '?'} B (<= 64)")

    # CSRs, privileged words, mnret
    all_elfs = [mon, park, stage0] + payloads
    csr_union = {}
    for e in all_elfs:
        acc = csr_accesses(e)
        smr = {hex(c): [hex(a) for a in v] for c, v in acc.items() if c in SMRNMI_CSRS}
        rep.check(f"csr_smrnmi[{e.path.name}]", not smr,
                  "no access to 0x740-0x744" if not smr else f"accesses {smr}")
        for c, v in acc.items():
            csr_union.setdefault(c, 0)
            csr_union[c] += len(v)
    rep.info("csrs_used", ", ".join(f"{c:#05x}x{n}" for c, n in sorted(csr_union.items())))

    for e in all_elfs:
        pw = priv_words(e)
        odd = [(hex(a), hex(w)) for a, w in pw if w not in PRIV_OK]
        mn_like = [(hex(a), hex(w)) for a, w in pw if (w & 0xFFF0707F) == MNRET and w != MNRET]
        n_mnret = sum(1 for _, w in pw if w == MNRET)
        rep.check(f"priv_insns[{e.path.name}]", not odd,
                  ("only " + ", ".join(sorted({PRIV_OK[w] for _, w in pw}))) if pw and not odd else
                  "none" if not odd else f"unexpected {odd}")
        rep.check(f"mnret[{e.path.name}]", not mn_like,
                  f"{n_mnret} x 0x70200073, no other mnret encoding" if not mn_like else f"bad encodings {mn_like}")
    hits = source_mnret_uses(FW)
    rep.check("mnret_source", not hits, "no source uses the mnret mnemonic" if not hits else f"used at {hits}")

    mw = words(mon)
    if is_spike:
        rep.info("rnmi_rearm", "SPIKE_HOST build: 0x353/0x7c1 are emulated (not required)")
    else:
        rep.check("rnmi_rearm", REARM in mw and CHICKEN in mw,
                  f"csrsi 0x353, 8 {'present' if REARM in mw else 'MISSING'}; "
                  f"csrwi 0x7c1, 0 {'present' if CHICKEN in mw else 'MISSING'}")

    fenced, call = fence_i_before_first_call(mon, start if start is not None else LIM_BASE)
    rep.check("hart0_fence_i", fenced and call is not None,
              f"fence.i before hart 0's first call at {call and hex(call)}" if fenced and call is not None else
              f"no fence.i between _start and hart 0's first call ({call and hex(call)}): after E1F the first RNMI "
              "would run stage 0 from hart 0's stale L1I line")

    for e in all_elfs:
        am = amo_words(e)
        rep.check(f"amo[{e.path.name}]", not am, "no LR/SC/AMO instruction" if not am else
                  f"{len(am)} A-extension words, first {[(hex(a), hex(w)) for a, w in am[:3]]}")

    exc, mh0, mh1 = mon.sym("mh_sec_rnmi_exc"), mon.sym("__mh_start"), mon.sym("__mh_end")
    pend, sbot = mon.sym("__persist_end"), mon.sym("__stack_bottom")
    mhsec = next((s for s in mon.sections if s["name"] == ".mh"), None)
    spin = [word_at(mon, exc), word_at(mon, exc + 4)] if exc is not None else []
    if is_spike:
        rep.info("mh_layout", f"SPIKE_HOST build: compact .mh [{mh0 and hex(mh0)}, {mh1 and hex(mh1)})")
    else:
        ok = (exc == MH_SEC_RNMI_EXC and spin == [WFI, J_SELF] and (mh0, mh1) == (MH_BASE, MH_END) and mhsec is not None
              and mhsec["type"] == SHT_NOBITS and pend is not None and pend <= mh0 and sbot is not None and mh1 <= sbot)
        rep.check("mh_layout", ok, f"mh_sec_rnmi_exc {exc and hex(exc)} = {[w and hex(w) for w in spin]}, .mh "
                  f"[{mh0 and hex(mh0)}, {mh1 and hex(mh1)}) NOLOAD, .persist ends {pend and hex(pend)}, stack bottom "
                  f"{sbot and hex(sbot)}")
    mhb = next((p for p in payloads if p.path.stem == "mh_bench"), None)
    if mhb is not None:
        ce, pe = mhb.sym("__code_end"), mhb.sym("__payload_end")
        ok = mhb.e_entry == MHB_BASE and ce is not None and ce <= MHB_DATA and pe == MHB_END
        rep.check("mh_bench", ok, f"entry {mhb.e_entry:#x}, code ends {ce and hex(ce)} (<= {MHB_DATA:#x}), image ends "
                  f"{pe and hex(pe)} (= {MHB_END:#x})")

    tab = crc32_tab_problem(mon)
    if tab is None:
        rep.info("crc32_tab", "no mon_crc32_tab: mon_crc32 built bitwise (-DMON_CRC_BITWISE)")
    else:
        rep.check("crc32_tab", tab == "", tab or f"mon_crc32_tab at {mon.sym('mon_crc32_tab'):#010x} in .rodata, "
                  "8 x 256 entries equal to zlib")

    if mon.sym("wbc_infer") is None:
        rep.check("kernel", False, "wbc_infer not linked")
    elif any(s["name"] == "wbc_stub.c" for s in mon.symbols):
        rep.info("kernel", "WARNING: wbc_stub.c linked (no Makefile.kernel): every INFER fails with MB_ERR_KERNEL")
    else:
        rep.info("kernel", "wbc_infer linked from the kernel lane's wbc_mlp.c")

    # weight blob
    if args.skip_blob:
        rep.info("weight_blob", "skipped (--skip-blob)")
    elif args.blob:
        n = args.blob.stat().st_size
        end = PAYLOAD_BASE + n
        rep.check("weight_blob", end <= LIM_END, f"{args.blob.name}: {n} B, ends {end:#010x} (<= {LIM_END:#010x})")
    elif not (TOOLS / "pack_weights.py").exists():
        rep.check("weight_blob", False, "x280/tools/pack_weights.py not found (use --blob FILE)")
    else:
        with tempfile.TemporaryDirectory() as td:
            for dtype, pol, gate in (("fp16", None, True), ("fp32", "balance", False), ("fp32", "walk", False)):
                out = Path(td) / f"{dtype}_{pol or 'both'}.blob"
                try:
                    n, line = pack_blob(dtype, pol, out)
                except RuntimeError as e:
                    rep.check(f"weight_blob[{dtype},{pol or 'both'}]", False, f"pack_weights.py failed: {e}")
                    continue
                end = PAYLOAD_BASE + n
                msg = f"{n} B real packed blob, ends {end:#010x} (LIM end {LIM_END:#010x}, spare {LIM_END - end} B)"
                if gate:
                    rep.check("weight_blob", end <= LIM_END, msg + " [deployed: fp16, Balance+Walk]")
                else:
                    (rep.info if end <= LIM_END else rep.check)(
                        f"weight_blob[{dtype},{pol}]", *((msg,) if end <= LIM_END else (False, msg)))

    print(f"== {len(rep.failed)} FAIL, {sum(1 for r in rep.rows if r['ok'])} PASS")
    if args.json:
        args.json.write_text(json.dumps(dict(elf=str(args.monitor_elf), rows=rep.rows), indent=1))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
