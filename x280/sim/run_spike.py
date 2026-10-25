#!/usr/bin/env python3
"""Run a bare-metal RISC-V ELF on spike set up like one X280 hart of L2CPU3 (PLAN.md WP-3.2b).

The point is to execute the one-shot firmware image on a RISC-V model before it ever runs on silicon (R37, X-5).
Stdlib only, so it runs under simpy, xpy or a bare python3. Use as a library (`run(...)` -> SpikeResult) or as
a CLI (`run_spike.py ELF [options] [-- extra spike args]`, `--explain` prints what is and is not modelled).

Configuration (spike built by build_spike.sh at the SHA in SHAS.txt; spike paths below are relative to
build/riscv-isa-sim at that SHA, tt-isa-documentation paths to third_party_refs/ BlackholeA0/L2CPUTile/):
- ISA `rv64gcv_zfh_zba_zbb` + VLEN 512 / ELEN 64. Spike at this SHA has no `--varch` (removed upstream in
  riscv-isa-sim 3b6a8fa0, 2024-06-21); VLEN is set by `Zvl<N>b` in the ISA string (disasm/isa_parser.cc:348)
  and ELEN 64 is implied by V (V -> Zve64d + Zvl128b, isa_parser.cc:93). `--varch vlen:512,elen:64` is
  therefore translated to the suffix `_zvl512b`.
- Memory: LIM 0x0800_0000 + 0x1E_0000 (tt-isa-documentation BlackholeA0/L2CPUTile/MemoryMap.md:84; 15 x 128 KiB
  at reset). With x280_regions (default) also, as plain RAM: the external-peripheral page 0x2001_0000 + 4 KiB
  (reset handler addresses +0x000, MemoryMap.md:111; GP scratch +0x100, :113; RNMI trigger bits +0x414, :118;
  RNMI handler addresses +0x418, :119) and the four hart Bus-Error Units 0x0170_0000 + 16 KiB (:69-72).
  Optional: the Zero Device 0x0A00_0000 + 2 MiB (:86), as plain RAM that starts zeroed.
- CLINT: spike's own CLINT sits at 0x0200_0000 (riscv/platform.h:9-10, 48 KiB), where the X280 has its CLINT too
  (MemoryMap.md:74), so mtime at 0x0200_BFF8 is readable; spike advances it once per 100 instructions
  (riscv/sim.h:79, sim.cc:380), not 16 core cycles per tick as on the 800 MHz / 50 MHz X280.
- HTIF: the ELF must define `tohost`/`fromhost`. Exit code = tohost >> 1 (fesvr/htif.cc:343-345); spike prints
  "*** FAILED *** (tohost = N)" on stderr for N != 0 (fesvr/syscall.cc:213-215), which is parsed so codes > 255
  survive. The console is the HTIF bcd device (device 1, command 1 = putchar; fesvr/device.cc:56-70) on stdout.
- dtc: spike execs `dtc` from PATH to build its DTB (riscv/dts.cc:154; configure.ac:63-65); spike-install/bin, where
  build_spike.sh installs dtc, is put first on PATH. `dtb=` passes a prebuilt DTB instead.
- Instruction counts come from spike's `-l` execution log streamed through a FIFO (nothing hits the disk); it
  slows spike down several-fold, so it is opt-in. `audit_stores` adds `--log-commits` and records, per 64 B line of
  LIM, which harts committed a store there, plus every committed A-extension instruction (LR/SC/AMO, major opcode
  0x2F): the multi-hart one-writer audit (x280/fw/mh.h). Spike has no timing model: one instruction per cycle,
  mcycle == minstret, so "cycles" are not reported separately. Speed on this host without the log
  (2026-10-24): ~75 M scalar instructions/s, ~1.9 M e32m4 vector instructions/s at VLEN 512.
- Spike runs at nice 19 on CPUs 0,1,8-11,16,17,24-27 by default (off the real-time cores and their SMT siblings);
  RT_SPIKE_CPUS="a,b-c" overrides, pin=False disables.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
SPIKE_BIN = SIM_DIR / "spike-install" / "bin" / "spike"

DEFAULT_ISA = "rv64gcv_zfh_zba_zbb"
DEFAULT_VARCH = "vlen:512,elen:64"
LIM = (0x0800_0000, 0x1E_0000)          # MemoryMap.md:84; 0x1E0000 = 15 x 128 KiB (PLAN.md WP-3.2 lim.ld)
X280_REGIONS = {
    "gpx": (0x2001_0000, 0x1000),     # MemoryMap.md:111-119 (reset vectors .. RNMI handler addresses)
    "beu": (0x0170_0000, 0x4000),     # MemoryMap.md:69-72 (hart 0-3 Bus-Error Units)
}
ZERO_DEVICE = (0x0A00_0000, 0x20_0000)  # MemoryMap.md:86
GP_SCRATCH = (0x2001_0100, 64)          # MemoryMap.md:113
BOOT_ROM = (0x1000, 0x1000)   # spike's reset vector page (riscv/platform.h:6 DEFAULT_RSTVEC); excluded from insns_image
DEFAULT_CPUS = "0,1,8-11,16,17,24-27"  # off the real-time cores 2,3,4,6,7,12-15 and their SMT siblings
TIMEOUT_EXIT, SPIKE_ERROR_EXIT = 124, 125

NOT_MODELLED = (
    "RNMI delivery: spike has no RNMI source (riscv/processor.cc:522-524: 'we don't model RNMI sources'), so "
    "setting a bit in 0x2001_0414 does nothing and the handler addresses at 0x2001_0418 are never used; the park "
    "stub can only be entered by a jump (SPIKE_HOST).",
    "X280 RNMI CSRs 0x350-0x353 (mnscratch/mnepc/mncause/mnstatus, RNMIs.md:6,63-70), 0x7c1 (feature-disable "
    "chicken bits, README.md:49) and mnret (0x70200073, RNMIs.md:31) trap as illegal instructions: spike's Smrnmi "
    "uses the standard numbers 0x740-0x744 (and is off in this ISA string); with Smcsrind added, 0x350 would "
    "silently become miselect (0x351-0x353 still trap with miselect = 0). Stub them behind SPIKE_HOST.",
    "Bus-Error Unit semantics: 0x0170_0000.. is plain RAM (reads 0 until written); no accrual, no local interrupt, "
    "no RNMI cause 3.",
    "LIM ECC and the partial-write-to-uninitialised-line fault (R26): spike RAM has no ECC.",
    "Reset: spike starts every hart in its boot ROM at 0x1000, which jumps to the ELF entry (or --pc / --pcs); "
    "the reset handler addresses at 0x2001_0000 + 8h are plain RAM; no once-only reset release (R24), no "
    "L2CPU_RESET, PLL4, CCACHE (WAYENABLE) or L2 prefetcher registers (0x0203_0000 is unmapped and faults).",
    "Timing: no caches, no uncached-LIM latency, 1 instruction per cycle; mtime ticks every 100 instructions.",
    "The host side: nothing writes the LIM mailbox while spike runs; the host must be scripted inside the "
    "image (SPIKE_HOST) and results reported through the HTIF console / exit code.",
    "Zvfh/Zvfhmin are not in the ISA string (the X280's support is unproven; PLAN.md WP-3.2 v2 kernel).",
)

_FAILED_RE = re.compile(rb"\*\*\* FAILED \*\*\* \(tohost = (-?\d+)\)")


@dataclasses.dataclass
class SpikeResult:
    """Outcome of one spike run. exit_code is the HTIF exit code, None when the guest never exited."""
    cmd: list[str]
    returncode: int
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    wall_s: float
    insns: int | None = None             # every retired instruction, all harts (count_insns)
    insns_image: int | None = None       # same, minus spike's boot ROM
    insns_by_hart: dict[int, int] | None = None
    traps: list[dict] | None = None      # spike's own record: hart, name, epc, tval (count_insns or trace_traps)
    stores_by_line: dict[int, list[int]] | None = None   # audit_stores: LIM line address -> harts that stored there
    store_pcs: dict[tuple[int, int], int] | None = None  # audit_stores: (hart, line) -> pc of its first store there
    amos: list[dict] | None = None       # audit_stores: committed A-extension instructions (hart, pc, insn)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ElfInfo:
    entry: int
    machine: int
    segments: list[tuple[int, int]]      # PT_LOAD (paddr, memsz), memsz > 0
    symbols: dict[str, int]


def elf_info(path: str | os.PathLike) -> ElfInfo:
    """Entry, loadable segments and symbols of an ELF64 little-endian file (enough to sanity-check a spike run)."""
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError(f"{path}: not an ELF64 little-endian file")
    (_, machine, _, entry, phoff, shoff, _, _, phentsize, phnum, shentsize, shnum, _) = struct.unpack_from(
        "<HHIQQQIHHHHHH", data, 16)
    segments = []
    for i in range(phnum):
        p_type, _, _, _, paddr, _, memsz, _ = struct.unpack_from("<IIQQQQQQ", data, phoff + i * phentsize)
        if p_type == 1 and memsz:
            segments.append((paddr, memsz))
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    symbols: dict[str, int] = {}
    for sh in sections:
        if sh[1] != 2:   # SHT_SYMTAB
            continue
        strtab = sections[sh[6]]
        str_off = strtab[4]
        for off in range(sh[4], sh[4] + sh[5], 24):
            st_name, _, _, _, st_value, _ = struct.unpack_from("<IBBHQQ", data, off)
            if st_name:
                end = data.index(b"\0", str_off + st_name)
                symbols[data[str_off + st_name:end].decode()] = st_value
    return ElfInfo(entry, machine, segments, symbols)


def isa_string(isa: str = DEFAULT_ISA, varch: str | None = DEFAULT_VARCH) -> str:
    """The ISA string for spike: `isa` plus `_zvl<vlen>b` from a legacy `vlen:N,elen:M` varch string."""
    if not varch:
        return isa
    fields = dict(kv.split(":", 1) for kv in varch.split(","))
    unknown = set(fields) - {"vlen", "elen"}
    if unknown:
        raise ValueError(f"varch: unknown fields {sorted(unknown)}")
    base = isa.lower().split("_")[0]
    has_v = "v" in base[4:].replace("g", "")
    elen = int(fields.get("elen", 64))
    if has_v and elen != 64:
        raise ValueError("varch: V implies Zve64d, so ELEN must be 64")
    if "vlen" in fields:
        vlen = int(fields["vlen"])
        if vlen < 32 or vlen & (vlen - 1):
            raise ValueError(f"varch: vlen {vlen} is not a power of two >= 32")
        if re.search(r"_zvl\d+b", isa.lower()):
            raise ValueError("varch: vlen given twice (varch and a Zvl*b in the ISA string)")
        isa = f"{isa}_zvl{vlen}b"
    return isa


def memory_regions(x280_regions: bool = True, zero_device: bool = False,
                   extra: tuple[tuple[int, int], ...] = ()) -> list[tuple[int, int]]:
    regions = [LIM]
    if x280_regions:
        regions += list(X280_REGIONS.values())
    if zero_device:
        regions.append(ZERO_DEVICE)
    return regions + list(extra)


def mem_arg(regions: list[tuple[int, int]]) -> str:
    return "-m" + ",".join(f"{base:#x}:{size:#x}" for base, size in regions)


def _parse_cpus(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _pin_prefix(pin: bool) -> list[str]:
    """`nice -n 19 taskset -c <cpus>` (task rule 4), as exec wrappers rather than a preexec_fn: the log
    reader thread may be running and preexec_fn is not fork-safe with threads."""
    if not pin:
        return []
    cpus = _parse_cpus(os.environ.get("RT_SPIKE_CPUS", DEFAULT_CPUS)) & os.sched_getaffinity(0)
    prefix = ["nice", "-n", "19"]
    if cpus:
        prefix += ["taskset", "-c", ",".join(str(c) for c in sorted(cpus))]
    return prefix


class _LogReader(threading.Thread):
    """Counts instructions and collects exceptions from spike's -l log, read from a FIFO while spike runs."""
    MAX_TRAPS = 1000

    MEM_RE = re.compile(rb"mem (0x[0-9a-f]+) (0x[0-9a-f]+)")
    COMMIT_RE = re.compile(rb"^\d (0x[0-9a-f]+) \((0x[0-9a-f]+)\)")

    def __init__(self, fifo: str, proc_done: threading.Event, audit: bool = False):
        super().__init__(daemon=True)
        self.fifo, self.proc_done, self.audit = fifo, proc_done, audit
        self.by_hart: dict[int, int] = {}
        self.rom = 0
        self.traps: list[dict] = []
        self.stores: dict[int, set] = {}
        self.store_pcs: dict[tuple[int, int], int] = {}
        self.amos: list[dict] = []
        self._tail = b""

    def run(self) -> None:
        fd = os.open(self.fifo, os.O_RDONLY | os.O_NONBLOCK)
        try:
            while True:
                try:
                    chunk = os.read(fd, 1 << 20)
                except BlockingIOError:   # writer connected, no data yet
                    time.sleep(0.002)
                    continue
                if chunk:
                    self._feed(chunk)
                elif self.proc_done.is_set():   # no writer and spike gone: done
                    break
                else:
                    time.sleep(0.005)           # writer not connected yet (or between opens)
        finally:
            os.close(fd)
            if self._tail:
                self._line(self._tail)

    def _feed(self, chunk: bytes) -> None:
        lines = (self._tail + chunk).split(b"\n")
        self._tail = lines.pop()
        for line in lines:
            self._line(line)

    def _line(self, line: bytes) -> None:
        # "core   0: 0x0000000008000000 (0x00000297) auipc ..."   one retired instruction
        # "core   0: exception trap_illegal_instruction, epc 0x..."  then optionally "core   0:  tval 0x..."
        if not line.startswith(b"core"):
            return
        colon = line.find(b":", 4)
        rest = line[colon + 2:]
        if rest.startswith(b"0x"):
            hart = int(line[4:colon])
            self.by_hart[hart] = self.by_hart.get(hart, 0) + 1
            pc = int(rest[2:18], 16)
            if BOOT_ROM[0] <= pc < BOOT_ROM[0] + BOOT_ROM[1]:
                self.rom += 1
        elif rest.startswith(b"exception ") or rest.startswith(b"interrupt "):
            if len(self.traps) < self.MAX_TRAPS:
                kind, _, tail = rest.partition(b" ")
                name, _, epc = tail.partition(b", epc ")
                self.traps.append({"hart": int(line[4:colon]), "kind": kind.decode(), "name": name.decode(),
                                   "epc": int(epc, 16) if epc else None, "tval": None})
        elif rest.lstrip().startswith(b"tval ") and self.traps:
            self.traps[-1]["tval"] = int(rest.split()[1], 16)
        elif self.audit and rest[:1].isdigit() and rest[1:4] == b" 0x":
            # commit line: "core   1: 3 0x<pc> (0x<insn>) [x5 0x..] [mem 0x<addr> [0x<stored value>]]..."
            c = self.COMMIT_RE.match(rest)
            if c is None:
                return
            hart, pc, insn = int(line[4:colon]), int(c.group(1), 16), int(c.group(2), 16)
            if insn & 3 == 3 and insn & 0x7F == 0x2F and len(self.amos) < self.MAX_TRAPS:
                self.amos.append({"hart": hart, "pc": pc, "insn": insn})
            for m in self.MEM_RE.finditer(rest):
                a = int(m.group(1), 16)
                if LIM[0] <= a < LIM[0] + LIM[1]:
                    ln = a & ~63
                    self.stores.setdefault(ln, set()).add(hart)
                    self.store_pcs.setdefault((hart, ln), pc)


def run(elf: str | os.PathLike, *, isa: str = DEFAULT_ISA, varch: str | None = DEFAULT_VARCH,
        x280_regions: bool = True, zero_device: bool = False, extra_regions: tuple[tuple[int, int], ...] = (),
        harts: int = 1, timeout_s: float = 60.0, count_insns: bool = False, trace_traps: bool = False,
        audit_stores: bool = False, pin: bool = True, dtb: str | None = None, extra_args: tuple[str, ...] = (),
        spike: str | os.PathLike = SPIKE_BIN, allow_no_htif: bool = False,
        check_segments: bool = True) -> SpikeResult:
    """Run `elf` on spike and return its HTIF exit code, console and (optionally) instruction counts."""
    spike = Path(spike)
    if not spike.exists():
        raise FileNotFoundError(f"{spike} missing; run {SIM_DIR / 'build_spike.sh'}")
    info = elf_info(elf)
    if info.machine != 243:   # EM_RISCV
        raise ValueError(f"{elf}: e_machine {info.machine} is not RISC-V")
    if not allow_no_htif and not {"tohost", "fromhost"} <= info.symbols.keys():
        raise ValueError(f"{elf}: no tohost/fromhost symbols; spike could neither print nor exit")
    regions = memory_regions(x280_regions, zero_device, extra_regions)
    if check_segments:
        for base, size in info.segments:
            if not any(lo <= base and base + size <= lo + n for lo, n in regions):
                raise ValueError(f"{elf}: segment [{base:#x}, {base + size:#x}) is outside every memory region")

    cmd = [str(spike), f"--isa={isa_string(isa, varch)}", mem_arg(regions), f"-p{harts}"]
    if dtb:
        cmd.append(f"--dtb={dtb}")
    env = dict(os.environ, PATH=f"{spike.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    with tempfile.TemporaryDirectory(prefix="run_spike_") as tmp:
        reader = None
        proc_done = threading.Event()
        if count_insns or trace_traps or audit_stores:
            fifo = os.path.join(tmp, "log.fifo")
            os.mkfifo(fifo)
            cmd += ["-l", f"--log={fifo}"] + (["--log-commits"] if audit_stores else [])
            reader = _LogReader(fifo, proc_done, audit=audit_stores)
            reader.start()   # spike's open of the FIFO for writing blocks until this side has it open
        cmd += [*extra_args, str(elf)]
        t0 = time.monotonic()
        proc = subprocess.Popen(_pin_prefix(pin) + cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, cwd=tmp, start_new_session=True)
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
        wall = time.monotonic() - t0
        proc_done.set()
        if reader is not None:
            reader.join(timeout=30)

    m = _FAILED_RE.search(err)
    if timed_out:
        exit_code = None
    elif m:
        exit_code = int(m.group(1))
    elif proc.returncode == 0:
        exit_code = 0
    else:
        exit_code = None   # spike itself failed (bad args, bad address, abort)
    res = SpikeResult(cmd=cmd, returncode=proc.returncode, exit_code=exit_code, timed_out=timed_out,
                      stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"), wall_s=wall)
    if reader is not None:
        if count_insns:
            res.insns = sum(reader.by_hart.values())
            res.insns_image = res.insns - reader.rom
            res.insns_by_hart = dict(sorted(reader.by_hart.items()))
        res.traps = reader.traps
        if audit_stores:
            res.stores_by_line = {a: sorted(h) for a, h in sorted(reader.stores.items())}
            res.store_pcs = dict(reader.store_pcs)
            res.amos = reader.amos
    return res


def explain() -> str:
    lines = [f"spike: {SPIKE_BIN}", f"isa: {isa_string()}", f"memory: {mem_arg(memory_regions())}",
             "not modelled:"]
    lines += [f"  - {item}" for item in NOT_MODELLED]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("elf", nargs="?")
    ap.add_argument("--isa", default=DEFAULT_ISA)
    ap.add_argument("--varch", default=DEFAULT_VARCH, help="legacy vlen:N,elen:M (mapped to _zvlNb)")
    ap.add_argument("--no-x280-regions", action="store_true", help="LIM only (no 0x2001_0000 / BEU pages)")
    ap.add_argument("--zero-device", action="store_true", help="add 0x0A00_0000 + 2 MiB as RAM")
    ap.add_argument("--harts", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=60.0, help="wall-clock seconds before SIGKILL")
    ap.add_argument("--count-insns", action="store_true", help="count retired instructions (slow)")
    ap.add_argument("--trace-traps", action="store_true", help="collect spike's exception records (slow)")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--dtb")
    ap.add_argument("--console", help="also write the console to this file")
    ap.add_argument("--json", help="write the result as JSON to this file ('-' = stdout, after the console)")
    ap.add_argument("--explain", action="store_true", help="print the configuration and what is not modelled")
    args = ap.parse_args(argv)
    if args.explain:
        print(explain())
        return 0
    if not args.elf:
        ap.error("ELF required")
    res = run(args.elf, isa=args.isa, varch=args.varch, x280_regions=not args.no_x280_regions,
              zero_device=args.zero_device, harts=args.harts, timeout_s=args.timeout,
              count_insns=args.count_insns, trace_traps=args.trace_traps, pin=not args.no_pin,
              dtb=args.dtb, extra_args=tuple(extra))
    sys.stdout.write(res.stdout)
    sys.stdout.flush()
    if args.console:
        Path(args.console).write_text(res.stdout)
    if args.json:
        text = json.dumps(res.to_json(), indent=1)
        if args.json == "-":
            print(text)
        else:
            Path(args.json).write_text(text + "\n")
    print(f"run_spike: exit={res.exit_code} timed_out={res.timed_out} returncode={res.returncode} "
          f"insns={res.insns} wall={res.wall_s:.2f}s", file=sys.stderr)
    if res.exit_code is None and not res.timed_out:
        sys.stderr.write(res.stderr)
    if res.timed_out:
        return TIMEOUT_EXIT
    if res.exit_code is None:
        return SPIKE_ERROR_EXIT
    return 0 if res.exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
