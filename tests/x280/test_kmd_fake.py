"""WP-3.3 host access layer against a fake device (PLAN.md T-3.4 prerequisites; no hardware is touched).

Run: source $RT/env.sh && nice -n 19 taskset -c 0,1,8-11,16,17,24-27 xpy -m pytest tests/x280/test_kmd_fake.py -q

The fake (kmd.FakeDevice) is one memfd per 2 MiB NoC region plus the KMD's ioctl/mmap/release rules; every
ioctl, mapping and access is recorded. Struct layouts are checked against tt-kmd 2.11's ioctl.h by a C
program compiled with clang-20; libwcio's store loop is checked by disassembly and by its transaction log.
"""

import ctypes
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys

import pytest

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT))

from x280 import regs  # noqa: E402
from x280 import ledger as L  # noqa: E402

HOST = RT / "x280" / "host"
KMD_HDR = RT / "third_party_refs" / "tt-kmd-2.11.0" / "ioctl.h"
# Off the real-time cores 2,3,4,6,7,12-15 and their SMT siblings (task rule; override with X280_CPUS).
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
NICE = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]
ARC, T3 = regs.ARC_XY, regs.L2CPU_XY[3]
CTXS = ("", "lim_init", "image", "stage0_entry", "pll4_step", "l2cpu_reset_rmw", "boot_vectors", "gp_scratch", "park",
        "l2pf", "xp_read")


def _build_libwcio():
    so, src = HOST / "libwcio.so", HOST / "wcio.c"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(NICE + ["make", "-s", "-C", str(HOST)], check=True)
    return so


_build_libwcio()
from x280.host import kmd as K  # noqa: E402

# Invented CMFW telemetry of a fake chip (tag -> raw u32 in the CMFW encoding), for the decoding tests only: kmd and
# x280d's sim hold no such values (verify:all D4). AICLK 800 MHz, 45.5 C (signed 16.16), 750 mV, 31 W, 41 A, 62 W.
FAKE_TELEMETRY = {regs.TAG_AICLK: 800, regs.TAG_ASIC_TEMPERATURE: (45 << 16) | 0x8000, regs.TAG_VCORE: 750,
                  regs.TAG_TDP: 31, regs.TAG_TDC: 41, regs.TAG_INPUT_POWER: 62}


@pytest.fixture
def dev():
    return K.FakeDevice.default_chip()


@pytest.fixture
def k(dev):
    s = K.Kmd(dev).open()
    dev.advance(1.5)  # past the keeper settle time
    s.read_l2cpu_reset()
    yield s
    if s.fd is not None:
        s.close()


@pytest.fixture
def ledger(tmp_path):
    led = L.Ledger(tmp_path / "x280_ledger.json")
    led.start_hold("test-hold", 1000)
    return led


def _writes(s):
    return [o for o in s.oplog if o.kind == "W32"]


STUB = bytes(range(1, 49)) + b"\x6f\x00\x00\x00" + bytes(12)  # a 64 B stand-in for park_stub.bin
IMG = bytes((i * 7 + 3) & 0xFF for i in range(4096))  # a stand-in monitor image


def _artefacts(s, tile=3, image=IMG, stub=STUB) -> int:
    """Boot steps 2-5 on FakeDevice, what Session.boot does before release(); returns the read-back image CRC."""
    with s.l2cpu_batch(tile):
        s.lim_init(tile=tile)
        crc = s.write_image(tile, image)
        s.write_gp_scratch(tile, stub)
        s.write_boot_vectors(tile)
    return crc


# --- ABI ---------------------------------------------------------------------------------------

STRUCTS = {
    "AllocateTlbIn": "tenstorrent_allocate_tlb_in", "AllocateTlbOut": "tenstorrent_allocate_tlb_out",
    "AllocateTlb": "tenstorrent_allocate_tlb", "FreeTlbIn": "tenstorrent_free_tlb_in",
    "FreeTlbOut": "tenstorrent_free_tlb_out", "FreeTlb": "tenstorrent_free_tlb",
    "NocTlbConfig": "tenstorrent_noc_tlb_config", "ConfigureTlbIn": "tenstorrent_configure_tlb_in",
    "ConfigureTlbOut": "tenstorrent_configure_tlb_out", "ConfigureTlb": "tenstorrent_configure_tlb",
    "SetNocCleanup": "tenstorrent_set_noc_cleanup",
}
IOCTLS = {"IOCTL_ALLOCATE_TLB": "TENSTORRENT_IOCTL_ALLOCATE_TLB", "IOCTL_FREE_TLB": "TENSTORRENT_IOCTL_FREE_TLB",
          "IOCTL_CONFIGURE_TLB": "TENSTORRENT_IOCTL_CONFIGURE_TLB",
          "IOCTL_SET_NOC_CLEANUP": "TENSTORRENT_IOCTL_SET_NOC_CLEANUP"}


def _leaves(ct, path="", base=0):
    """(C member path, offset, size) for every field of a ctypes Structure, recursively."""
    for name, typ in ct._fields_:
        cname = "in" if name == "in_" else name
        off = base + getattr(ct, name).offset
        p = f"{path}.{cname}" if path else cname
        yield p, off, ctypes.sizeof(typ)
        if isinstance(typ, type) and issubclass(typ, ctypes.Structure):
            yield from _leaves(typ, p, off)


def test_structs_and_ioctl_numbers_match_kmd_header(tmp_path):
    lines, expect = [], {}
    for py, c in STRUCTS.items():
        ct = getattr(K, py)
        lines.append(f'printf("sizeof {c}\\t%zu\\n", sizeof(struct {c}));')
        expect[f"sizeof {c}"] = str(ctypes.sizeof(ct))
        for p, off, size in _leaves(ct):
            lines.append(f'printf("off {c}.{p}\\t%zu %zu\\n", offsetof(struct {c}, {p}), sizeof(((struct {c} *)0)->{p}));')
            expect[f"off {c}.{p}"] = f"{off} {size}"
    for py, c in IOCTLS.items():
        lines.append(f'printf("ioctl {c}\\t%lu\\n", (unsigned long)({c}));')
        expect[f"ioctl {c}"] = str(getattr(K, py))
    prog = tmp_path / "abi.c"
    prog.write_text("#include <stdio.h>\n#include <stddef.h>\n#include \"ioctl.h\"\nint main(void) {\n"
                    + "\n".join(lines) + "\nreturn 0;\n}\n")
    exe = tmp_path / "abi"
    subprocess.run(NICE + ["clang-20", "-std=gnu11", "-Wall", "-Werror", f"-I{KMD_HDR.parent}", "-o", str(exe),
                           str(prog)], check=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    got = dict(ln.split("\t") for ln in out.splitlines())
    assert got == expect
    assert got["sizeof tenstorrent_set_noc_cleanup"] == "32"  # argsz (chardev.c:515)
    assert len(got) > 40  # every leaf field of the 11 structs was compared


def test_only_tlb_and_cleanup_ioctls_exist():
    names = {n for n in dir(K) if n.startswith("IOCTL_")}
    assert names == set(IOCTLS) | {"IOCTL_NAMES", "IOCTL_TYPES"}
    src = (HOST / "kmd.py").read_text()
    assert re.findall(r"= _io\((\d+)\)", src) == ["11", "12", "13", "14"]  # ioctl.h:25-28 only
    for banned in ("SMC_MSG", "SET_POWER_STATE", "RESET_DEVICE", "import pyluwen", "import ttnn", "from ttnn"):
        assert banned not in src


def test_set_noc_cleanup_struct_bytes(k):
    with k.l2cpu_batch(3):
        k.lim_init(regs.MB_HOST_GONE, regs.MB_HOST_GONE + 64)
    s = k.set_noc_cleanup(3)
    assert bytes(s) == struct.pack("<IIBBBBIQQ", 32, 0, 1, 8, 7, 0, 0, 0x0801_0980, 0x474F_4E45)
    fd, name, fields, rc = k.backend.ioctls[-1]
    assert (name, rc, fields["argsz"], fields["x"], fields["y"], fields["addr"], fields["data"]) == \
        ("SET_NOC_CLEANUP", 0, 32, 8, 7, regs.MB_HOST_GONE, regs.HOST_GONE_VALUE)


@pytest.mark.parametrize("field,value", [("argsz", 31), ("flags", 1), ("enabled", 2), ("addr", 0x0801_0982),
                                         ("noc", 2), ("x", 65)])
def test_fake_validates_cleanup_like_kmd(dev, field, value):
    fd = dev.open(power_aware=True)
    s = K.SetNocCleanup(argsz=32, enabled=1, x=8, y=7, addr=regs.MB_HOST_GONE, data=regs.HOST_GONE_VALUE)
    setattr(s, field, value)
    with pytest.raises(OSError) as e:
        dev.ioctl(fd, K.IOCTL_SET_NOC_CLEANUP, s)
    assert e.value.errno == 22  # EINVAL (chardev.c:515-536)


# --- windows -----------------------------------------------------------------------------------


def test_windows_configured_once_and_mapped_by_policy(dev):
    s = K.Kmd(dev, cross_read=True).open()
    specs = regs.window_specs((3,), cross_read=True)
    assert dev.names() == ["ALLOCATE_TLB", "CONFIGURE_TLB"] * len(specs)
    assert sorted(dev.configure_count.values()) == [1] * len(specs)
    cfgs = [f for _, n, f, _ in dev.ioctls if n == "CONFIGURE_TLB"]
    for spec, c in zip(specs, cfgs, strict=True):
        assert (c["addr"], (c["x_end"], c["y_end"])) == (spec.base, spec.xy)
        assert (c["noc"], c["ordering"], c["mcast"], c["linked"], c["static_vc"], c["x_start"], c["y_start"]) == \
            (0, 1, 0, 0, 0, 0, 0)
    maps = {(c[2], c[3]): c[5] for c in dev.calls if c[0] == "mmap"}
    ids = {w.spec.name: w.tlb_id for w in s.windows.values()}
    rw, ro = K.PROT_READ | K.PROT_WRITE, K.PROT_READ
    assert maps == {(ids["arc"], "uc"): ro, (ids["csm"], "uc"): ro, (ids["lim3"], "uc"): rw, (ids["lim3"], "wc"): rw,
                    (ids["ccache3"], "uc"): ro, (ids["beu3"], "uc"): ro, (ids["ccache3x"], "uc"): ro}
    assert ids["xp3"] not in {i for i, _ in maps}  # external peripherals: transient only
    s.close()
    assert dev.names().count("FREE_TLB") == len(specs) and not dev.tlb_owner and not dev.live
    assert [c[0] for c in dev.calls[-2:]] == ["close", "release"]


def test_fake_kmd_mmap_and_free_rules(dev):
    fd = dev.open(power_aware=False)
    a = K.AllocateTlb()
    a.in_.size = regs.TLB_2M
    dev.ioctl(fd, K.IOCTL_ALLOCATE_TLB, a)
    c = K.ConfigureTlb()
    c.in_.id, c.in_.config.addr = a.out.id, 0x0800_0001
    with pytest.raises(OSError):  # 2 MiB alignment (blackhole.c:181-182)
        dev.ioctl(fd, K.IOCTL_CONFIGURE_TLB, c)
    c.in_.config.addr, c.in_.config.x_end, c.in_.config.y_end = regs.LIM_BASE, 8, 7
    dev.ioctl(fd, K.IOCTL_CONFIGURE_TLB, c)
    with pytest.raises(OSError):  # only from the window's start (memory.c:1582-1593)
        dev.mmap(fd, 4096, K.PROT_READ, a.out.mmap_offset_uc + 4096)
    va = dev.mmap(fd, 4096, K.PROT_READ, a.out.mmap_offset_uc)
    f = K.FreeTlb()
    f.in_.id = a.out.id
    with pytest.raises(OSError) as e:  # EBUSY while mapped (memory.c:1017-1024)
        dev.ioctl(fd, K.IOCTL_FREE_TLB, f)
    assert e.value.errno == 16
    dev.munmap(va, 4096)
    dev.ioctl(fd, K.IOCTL_FREE_TLB, f)


def test_dram_or_foreign_window_refused(k):
    n = len(k.backend.ioctls)
    for xy, base in (((17, 12), 0), ((0, 0), 0), (T3, 0x4000_3000_0000), (T3, 0x3000_0000)):
        spec = regs.WindowSpec("bad", xy, base, 4096, "ro", False, "test")
        with pytest.raises(K.PolicyError, match="not a regs window"):
            k._setup_window(spec)
        assert regs.classify_read(xy, base) != ""  # not even readable
    assert len(k.backend.ioctls) == n


# --- UC / WC access, libwcio -------------------------------------------------------------------


def test_uc_wc_access_and_per_tick_order(k, dev):
    with k.l2cpu_batch(3):
        k.lim_init(regs.MAILBOX_BASE, regs.MAILBOX_BASE + 64 * 64)
        obs = bytearray(os.urandom(regs.MB_OBS_LINES * 64))
        first = k.wcio.log_total()
        k.wc_write_lines(regs.MB_OBS, obs)
        k.uc_write32(regs.MB_SEQ_REQ, 42)
        assert k.read_lim32(regs.MB_SEQ_REQ) == 42
        assert k.read_lim32(regs.MB_OBS + 64) == int.from_bytes(obs[64:68], "little")
    assert dev.peek(T3, regs.MB_OBS, len(obs)) == bytes(obs)
    tx = [t for t in k.txn_log() if t["seq"] >= first]
    ops = [t["op"] for t in tx]
    assert ops[:34] == ["WC64"] * 33 + ["SFENCE"] and ops[34] == "UC_W32"
    assert [t["noc"] for t in tx[:33]] == [regs.MB_OBS + 64 * i for i in range(33)]
    assert all(t["kind"] == "wc" and t["size"] == 64 and t["align64"] for t in tx[:33])
    assert tx[34]["noc"] == regs.MB_SEQ_REQ and tx[34]["kind"] == "uc" and tx[34]["size"] == 4


def test_full_lim_init_is_only_aligned_64B_stores_with_sfence(k, dev):
    first = k.wcio.log_total()
    with k.l2cpu_batch(3):
        assert k.lim_init(pattern=lambda a: a.to_bytes(8, "little") * 8) == regs.LIM_LINES
        for i in range(100):  # E1a: partial 4 B writes on initialised lines
            a = regs.WEIGHTS_BASE + 4096 * i + 4 * (i % 16)
            k.uc_write32(a, 0xA5A5_0000 | i)
            assert k.read_lim32(a) == 0xA5A5_0000 | i
    tx = [t for t in k.txn_log() if t["seq"] >= first]
    wc = [i for i, t in enumerate(tx) if t["op"] == "WC64"]
    assert len(wc) == regs.LIM_LINES
    assert all(tx[i]["size"] == 64 and tx[i]["align64"] and tx[i]["kind"] == "wc" and tx[i + 1]["op"] == "SFENCE"
               for i in wc)
    assert sorted(tx[i]["noc"] for i in wc) == list(range(regs.LIM_BASE, regs.LIM_END, 64))
    assert not [t for t in tx if t["op"] == "REFUSED"]
    inited, partial_ok = set(), True
    for t in tx:  # no partial write precedes the init of its line
        if t["op"] == "WC64":
            inited.add(t["noc"])
        elif t["op"] == "UC_W32" and t["xy"] == T3:
            partial_ok &= (t["noc"] & ~63) in inited
    assert partial_ok
    assert dev.peek(T3, regs.LIM_END - 64, 8) == (regs.LIM_END - 64).to_bytes(8, "little")


def test_partial_write_before_init_refused(k):
    with k.l2cpu_batch(3):
        first = k.wcio.log_total()
        with pytest.raises(K.PolicyError, match="uninitialised"):
            k.uc_write32(regs.MB_SEQ_REQ, 1)
        with pytest.raises(K.PolicyError, match="uninitialised"):
            k.read_lim32(regs.MB_SEQ_REQ)
    assert not [t for t in k.txn_log() if t["seq"] >= first and t["op"] in ("UC_W32", "UC_R32")]


def test_wc_store_loop_disassembly_only_nt_zmm_stores():
    out = subprocess.run(["llvm-objdump-20", "-d", "--no-show-raw-insn", "--disassemble-symbols=wcio__stream_lines",
                          str(HOST / "libwcio.so")], check=True, capture_output=True, text=True).stdout
    insns = [ln.split(":", 1)[1].strip() for ln in out.splitlines() if re.match(r"\s+[0-9a-f]+:", ln)]
    assert insns, out
    def operands(s):  # AT&T operands, split at commas outside parentheses
        out, depth, cur = [], 0, ""
        for ch in s:
            depth += (ch == "(") - (ch == ")")
            if ch == "," and depth == 0:
                out.append(cur.strip())
                cur = ""
            else:
                cur += ch
        return out + [cur.strip()] if cur.strip() else out

    stores = []
    for ins in insns:
        mnem, _, ops = ins.partition("\t")
        ops = operands(ops)
        if len(ops) >= 2 and "(" in ops[-1] and not mnem.startswith(("nop", "cmp", "test", "prefetch")):
            stores.append(ins)  # last AT&T operand is the destination: a memory store
    assert stores and all(re.match(r"vmovnt(ps|pd|dq)\s+%zmm\d+, ", s) for s in stores), stores
    assert any(i.startswith("sfence") for i in insns)


def test_cpu_has_avx512_and_refusal_writes_nothing(k, dev):
    flags = set(next(ln for ln in Path("/proc/cpuinfo").read_text().splitlines() if ln.startswith("flags")).split())
    assert "avx512f" in flags and k.wcio.avx512_ok()
    with k.l2cpu_batch(3):
        k.lim_init(regs.MB_OBS, regs.MB_OBS + 64 * 33)
        before = dev.peek(T3, regs.MB_OBS, 64 * 33)
        first = k.wcio.log_total()
        k.wcio.force_no_avx512(True)
        try:
            with pytest.raises(K.WcioError, match="nothing written"):
                k.wc_write_lines(regs.MB_OBS, bytearray(b"\xff" * 64 * 33))
        finally:
            k.wcio.force_no_avx512(False)
    assert dev.peek(T3, regs.MB_OBS, 64 * 33) == before
    ops = [t["op"] for t in k.txn_log() if t["seq"] >= first]
    assert ops == ["REFUSED"]


def test_misaligned_or_partial_wc_refused(k):
    with k.l2cpu_batch(3):
        for addr, n in ((regs.MB_OBS + 4, 64), (regs.MB_OBS, 60), (regs.LIM_END - 64, 128)):
            with pytest.raises(K.PolicyError):
                k.wc_write_lines(addr, bytearray(n))


# --- the policy: WAYENABLE, GDDR, PLL, L2CPU_RESET ---------------------------------------------


def test_wayenable_write_impossible(k, dev):
    for xy in list(regs.L2CPU_XY.values()) + list(regs.L2CPU_XY_TRANSLATED.values()):
        for a in range(regs.CCACHE0_WAYENABLE, regs.CCACHE0_WAYENABLE + 8, 4):
            for size in (4, 64):
                assert regs.classify_write(xy, a, size)[0] is None
    with k.l2cpu_batch(3):
        for ctx in CTXS:
            with pytest.raises(K.PolicyError):
                k._policy(T3, regs.CCACHE0_WAYENABLE, 4, 0xF, ctx)
        with k._transient("ccache3", "l2pf") as (w32, _):
            with pytest.raises(K.PolicyError, match="DENY CCACHE0_WAYENABLE"):
                w32(regs.CCACHE0_WAYENABLE, 0xF)
        snap = dev.peek(T3, 0x0200_0000, 0x3_8000)
        k.l2pf_setup(3)
    after = dev.peek(T3, 0x0200_0000, 0x3_8000)
    changed = {0x0200_0000 + i for i in range(0, len(snap), 4) if snap[i:i + 4] != after[i:i + 4]}
    assert changed == {a for a, _ in regs.L2PF_WRITES}
    assert dev.peek32(T3, regs.CCACHE0_WAYENABLE) == 0
    cid = k.windows["ccache3"].tlb_id
    rw = [c for c in dev.calls if c[0] == "mmap" and c[2] == cid and c[5] & K.PROT_WRITE]
    assert len(rw) == 2  # the explicit transient above + the one inside l2pf_setup, both unmapped again
    assert len([v for v in dev.live.values() if v[1] == cid]) == 1  # only the persistent PROT_READ mapping


RAW_STORE = """
import ctypes, sys; sys.path.insert(0, {rt!r})
assert ctypes.CDLL(None).prctl(4, 0, 0, 0, 0) == 0  # PR_SET_DUMPABLE 0: no core dump, so apport is never invoked
from x280 import regs
from x280.host import kmd as K
dev = K.FakeDevice.default_chip(); k = K.Kmd(dev).open()
w = k.windows[{win!r}]
print("mapped", flush=True)
k.wcio.uc_write32(w.uc + {addr} - w.spec.base, 0xF)
print("WROTE", flush=True)
"""


@pytest.mark.parametrize("win,addr", [("ccache3", regs.CCACHE0_WAYENABLE), ("arc", regs.DDR_RESET),
                                      ("arc", regs.L2CPU_RESET), ("arc", regs.PLL4_CNTL_1)])
def test_raw_store_to_readonly_window_faults(win, addr):
    """Even code that bypasses the policy cannot store there: the persistent mapping is PROT_READ."""
    r = subprocess.run([sys.executable, "-c", RAW_STORE.format(rt=str(RT), win=win, addr=addr)], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == -signal.SIGSEGV and "mapped" in r.stdout and "WROTE" not in r.stdout, (r.returncode, r.stdout)


GDDR_ADDRS = (0x3000_0000, 0x1_2FFF_FFFC, 0x4000_3000_0000, 0x0004_3000_0000, 0x0804_3000_0000,
              0x4804_3000_0000, 0x0000_8000_3000_0000, 0xFFFF_F7FE_FFE0_0000, 0x2000_0000, 0xFFFF_F7FE_FFF0_0000,
              0x2008_0000)


def test_gddr_unreachable(k):
    for a in GDDR_ADDRS:
        for xy in (T3, regs.L2CPU_XY[2]):
            allow, denies, why = regs.classify_write(xy, a, 4)
            assert allow is None and denies, hex(a)
            for ctx in CTXS:
                with pytest.raises((K.PolicyError, K.InterlockError)):
                    k._policy(xy, a, 4, 0, ctx)
    assert regs.classify_write((17, 12), 0, 4)[0] is None and regs.classify_write((9, 0), 0, 4)[0] is None
    assert regs.classify_write(ARC, regs.DDR_RESET, 4)[0] is None
    gddr = [d for d in regs.DENY if regs.GDDR_REACHING in d.why]
    for w in k.windows.values():  # no persistent mapping covers GDDR or an alias of it
        if w.uc or w.wc:
            lo, hi = w.spec.base, w.spec.base + w.spec.map_len - 1
            assert not [d for d in gddr if d.tiles == "l2cpu" and regs.tile_of(w.spec.xy) is not None
                        and d.lo <= hi and lo <= d.hi], w.spec.name
    with k.l2cpu_batch(3), k._transient("xp3", "park") as (w32, r32):  # the transient XP prefix's alias MiB
        for a in (0xFFFF_F7FE_FFE0_0000, 0xFFFF_F7FE_FFF0_0000):
            with pytest.raises(K.PolicyError):
                w32(a, 0)
        with pytest.raises(K.PolicyError):
            r32(0xFFFF_F7FE_FFE0_0000)
        assert r32(0xFFFF_F7FE_FFF0_0000) == 0  # small TLB slot 0: readable since S5a (regs X280_TLB_SMALL_CFG)


def test_pll_registers_denied_except_the_clock_py_walk(k, dev):
    for a in (regs.PLL4_CNTL_0, regs.PLL4_USE_POSTDIV, regs.PLL_LOCK, 0x8002_0104):  # incl. PLL0 CNTL_1
        allow, denies, _ = regs.classify_write(ARC, a, 4)
        assert allow is None and [d.name for d in denies] == ["PLL"]
    with pytest.raises(K.PolicyError, match="DENY PLL"):
        k._policy(ARC, regs.PLL4_CNTL_1, 4, regs.PLL4_200_CNTL_1, "")
    with k._transient("arc", "pll4_step") as (w32, _):  # nothing armed: even a valid value is refused
        with pytest.raises(K.PolicyError, match="out of the clock.py sequence"):
            w32(regs.PLL4_CNTL_5, 0x0101_0102)
        with pytest.raises(K.PolicyError):
            w32(regs.PLL4_USE_POSTDIV, 0)
    with pytest.raises(K.InterlockError):  # 800 walk needs the 200 MHz start state
        k.pll4_step_to(800)
    n = len(k.oplog)
    k.pll4_step_to(200)
    walk = [(o.addr, o.value) for o in k.oplog[n:] if o.kind == "W32"]
    assert walk == list(regs.PLL4_STEP[200][2]) and len(walk) == 120
    # refdiv 2 and the silicon's postdiv byte 1 are kept (S1c): 200 MHz = fbdiv 128, 800 MHz = 0x0040_0102 again
    assert (dev.peek32(ARC, regs.PLL4_CNTL_1), dev.peek32(ARC, regs.PLL4_CNTL_5)) == (0x0080_0102, 0x0F0F_0F0F)
    with pytest.raises(K.InterlockError):  # no L2CPU access at 200 MHz
        with k.l2cpu_batch(3):
            pass
    k.pll4_step_to(800)
    assert (dev.peek32(ARC, regs.PLL4_CNTL_1), dev.peek32(ARC, regs.PLL4_CNTL_5)) == (0x0040_0102, 0x0101_0101)
    with pytest.raises(K.PolicyError, match="one PLL4 excursion"):
        k.pll4_step_to(200)


def _vco_limits() -> tuple:
    pll = (RT / "third_party_refs/fw-19.12.0/lib/tenstorrent/bh_arc/pll.c").read_text()
    return tuple(int(re.search(rf"#define {n}\s+(\d+)", pll)[1]) for n in ("VCO_MIN_FREQ", "VCO_MAX_FREQ"))


def test_pll4_walks_keep_refdiv_and_postdiv_vco_in_range_l2cpuclk_never_above_800(k, dev, ledger):
    """Both walks as kmd runs them on the S1c chip state, read back from the device after every write: only fbdiv
    (+-1 per CNTL_1 write, refdiv and the postdiv byte as read at the walk's start, BHL clock.py:40-45, 58-60) and
    one CNTL_5 postdiv (+-1 per write; increasing ones before any fbdiv write, decreasing ones after, clock.py:62-71)
    move; the VCO stays in pll.c's VCO_MIN_FREQ..VCO_MAX_FREQ and no L2CPUCLK exceeds 800 MHz at any step."""
    vmin, vmax = _vco_limits()
    assert (vmin, vmax) == (1600, 5000)
    crc = _artefacts(k)
    for mhz in (200, 800):
        c1, c5 = dev.peek32(ARC, regs.PLL4_CNTL_1), dev.peek32(ARC, regs.PLL4_CNTL_5)
        keep, states, n0 = c1 & 0xFFFF, [], len(_writes(k))
        k.step_sleep = lambda: states.append((dev.peek32(ARC, regs.PLL4_CNTL_1), dev.peek32(ARC, regs.PLL4_CNTL_5)))
        k.pll4_step_to(mhz)
        assert len(states) == len(_writes(k)) - n0 == 120
        phase = 0  # 0: postdivs up, 1: fbdiv, 2: postdivs down
        for n1, n5 in states:
            if n1 != c1:
                assert n5 == c5 and n1 & 0xFFFF == keep and abs((n1 >> 16) - (c1 >> 16)) == 1 and phase <= 1
                phase = 1
            else:
                d = [(i, b - a) for i, (a, b) in enumerate(zip(c5.to_bytes(4, "little"), n5.to_bytes(4, "little")))
                     if a != b]
                assert len(d) == 1 and abs(d[0][1]) == 1
                if d[0][1] > 0:  # an increasing postdiv: nothing else has moved yet
                    assert phase == 0
                phase = 0 if d[0][1] > 0 else 2
            c1, c5 = n1, n5
            refdiv, fbdiv = c1 & 0xFF, c1 >> 16
            vco = 50 * fbdiv / refdiv  # FW pll.c:490-493 (REFCLK_F_MHZ 50, timer.h:12)
            outs = [vco / (p + 1) for p in c5.to_bytes(4, "little")]  # pll.c:462-475 external postdivs only
            assert vmin <= vco <= vmax and max(outs) <= 800.0, (mhz, hex(c1), hex(c5), vco, outs)
        assert c1 & 0xFFFF == 0x0102  # refdiv 2 / postdiv 1 at both walk ends
        if mhz == 200:  # refdiv 2 / postdiv 1 / fbdiv 128
            assert c1 == 0x0080_0102 and k.release((3,), ledger, crc, gp_code=STUB) == (0x0F, 0x8F, 0x8F)
    assert c1 == regs.PLL4_CNTL_1_DEFAULT == 0x0040_0102 and c5 == regs.PLL4_CNTL_5_DEFAULT


def test_pll4_at_the_source_value_is_dirty_and_never_walked(dev):
    """pll.c:226's CNTL_1 0x0040_0002 (postdiv 0) was never read on silicon (S1c: 0x0040_0102): no L2CPU batch, no
    PLL4 write, and the refusal says why."""
    dev.poke32(ARC, regs.PLL4_CNTL_1, regs.PLL4_CNTL_1_SOURCE)
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    assert not s.pll4_is_default(s.read_pll4()) and "unexplained" in s.pll4_default_why(s.read_pll4())
    with pytest.raises(K.InterlockError, match="PLL4 not at CMFW defaults .*0x00400002 is the bh_arc source value"):
        with s.l2cpu_batch(3):
            pass
    with pytest.raises(K.InterlockError, match="sequence start for 200 MHz .*unexplained, not walked"):
        s.pll4_step_to(200)
    assert not _writes(s) and dev.peek32(ARC, regs.PLL4_CNTL_1) == regs.PLL4_CNTL_1_SOURCE
    s.close()


def test_l2cpu_reset_single_rmw(k, dev, ledger):
    crc = _artefacts(k)
    with pytest.raises(K.InterlockError, match="200 MHz"):
        k.release((3,), ledger, crc, gp_code=STUB)
    with k._transient("arc", "l2cpu_reset_rmw") as (w32, _):  # not armed by release()
        with pytest.raises(K.PolicyError, match="only by release"):
            w32(regs.L2CPU_RESET, 0x8F)
    k.pll4_step_to(200)
    n = len(k.oplog)
    assert k.release((3,), ledger, crc, gp_code=STUB) == (0x0F, 0x8F, 0x8F)
    seq = [o for o in k.oplog[n:] if o.addr == regs.L2CPU_RESET and o.kind in ("R32", "W32")]
    assert [(o.kind, o.value) for o in seq] == [("R32", 0x0F), ("W32", 0x8F), ("R32", 0x8F)]
    assert ledger.state(3) == "released" and ledger.read()["tiles"]["3"]["image_crc"] == crc
    with pytest.raises(K.PolicyError, match="already written once"):
        k.release((3,), ledger, crc, gp_code=STUB)
    k.pll4_step_to(800)
    assert len([o for o in _writes(k) if o.addr == regs.L2CPU_RESET]) == 1
    assert dev.peek32(ARC, regs.L2CPU_RESET) == 0x8F


def test_release_refused_when_already_released_or_ledger_spent(tmp_path):
    led = L.Ledger(tmp_path / "l.json")
    led.start_hold("h", 1)
    for chip, burn in ((K.FakeDevice.default_chip(l2cpu_reset=0x8F), False), (K.FakeDevice.default_chip(), True)):
        if burn:
            led.mark_burned(3)
        s = K.Kmd(chip).open()
        chip.advance(2)
        s.read_l2cpu_reset()
        crc = _artefacts(s) if not s.released[3] else 1
        s.pll4_step_to(200)
        with pytest.raises((K.InterlockError, K.PolicyError)):
            s.release((3,), led, crc, gp_code=STUB)
        assert not [o for o in _writes(s) if o.addr == regs.L2CPU_RESET] and chip.peek32(ARC, regs.L2CPU_RESET) in (0x0F, 0x8F)
        s.close()


def test_dp4_both_tiles_one_rmw_and_tile0_refused(tmp_path):
    with pytest.raises(K.PolicyError):
        K.Kmd(K.FakeDevice.default_chip(), tiles=(0,))
    led = L.Ledger(tmp_path / "l.json")
    led.start_hold("h", 1)
    chip = K.FakeDevice.default_chip()
    s = K.Kmd(chip, tiles=(2, 3)).open()
    chip.advance(2)
    s.read_l2cpu_reset()
    crc = _artefacts(s, 3)
    assert _artefacts(s, 2) == crc
    s.pll4_step_to(200)
    assert s.release((2, 3), led, crc, gp_code=STUB) == (0x0F, 0xCF, 0xCF)
    assert led.state(2) == led.state(3) == "released"
    s.close()


# --- interlocks ---------------------------------------------------------------------------------


def test_interlocks(dev):
    s = K.Kmd(dev).open()
    with pytest.raises(K.InterlockError, match="opened"):
        with s.l2cpu_batch(3):
            pass
    dev.advance(1.0)
    with pytest.raises(K.InterlockError, match="outside l2cpu_batch"):
        s.read_wayenable()
    dev.poke32(ARC, regs.PLL4_CNTL_5, 0x0101_0102)  # CMFW relock mid-flight / someone else's clock
    with pytest.raises(K.InterlockError, match="PLL4"):
        with s.l2cpu_batch(3):
            pass
    dev.poke32(ARC, regs.PLL4_CNTL_5, regs.PLL4_CNTL_5_DEFAULT)
    with s.l2cpu_batch(3):
        assert s.read_wayenable() == 0
    s.close()
    chip = K.FakeDevice.default_chip(enabled_l2cpu=0x7)
    s = K.Kmd(chip).open()
    chip.advance(2)
    with pytest.raises(K.InterlockError, match="ENABLED_L2CPU"):
        with s.l2cpu_batch(3):
            pass
    s.close()


def test_boot_vectors_values_and_pre_release_only(dev, ledger):
    s = K.Kmd(dev).open()
    dev.advance(2)
    with s.l2cpu_batch(3):
        with pytest.raises(K.PolicyError, match="read_l2cpu_reset"):
            s.write_boot_vectors(3)  # release state unknown
    s.read_l2cpu_reset()
    with s.l2cpu_batch(3):
        with pytest.raises(K.PolicyError):
            s.write_boot_vectors(3, reset_pc=0)
        with pytest.raises(K.PolicyError):
            s.write_boot_vectors(3, rnmi=0x0800_0000)
        s.write_boot_vectors(3)
        s.write_gp_scratch(3, STUB)
    for h in range(4):
        assert dev.peek32(T3, regs.reset_vector(h)) == regs.LIM_BASE and dev.peek32(T3, regs.reset_vector(h) + 4) == 0
        assert dev.peek32(T3, regs.rnmi_handler(h)) == 0x2001_0100 and dev.peek32(T3, regs.rnmi_exc_handler(h)) == 0x2001_0130
    assert dev.peek(T3, regs.GP_SCRATCH, 64) == STUB
    crc = _artefacts(s)
    s.pll4_step_to(200)
    s.release((3,), ledger, crc, gp_code=STUB)
    s.pll4_step_to(800)
    with s.l2cpu_batch(3):
        with pytest.raises(K.PolicyError, match="before release"):
            s.write_boot_vectors(3)
        with pytest.raises(K.PolicyError, match="only before release"):
            s.write_gp_scratch(3, STUB)  # the RNMI handler code is never rewritten under a live hart
        assert s.rnmi_set(3, 0b0001) == 1 and s.rnmi_set(3, 0) == 0  # PARK trigger set / clear
        with pytest.raises(K.PolicyError, match="reserved"):
            s.rnmi_set(3, 0x10)
        assert s.read_hart_status(3) == 0
    ro = [c for c in dev.calls if c[0] == "mmap" and c[2] == s.windows["xp3"].tlb_id and not c[5] & K.PROT_WRITE]
    assert len(ro) == 1  # read_hart_status maps the XP prefix read-only
    s.close()


def test_mailbox_line_ownership_after_release(dev, ledger):
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    crc = _artefacts(s)
    with s.l2cpu_batch(3):
        s.uc_write32(regs.MB_STATUS, 5)  # before release one writer is moot (E1a patterns)
    s.pll4_step_to(200)
    s.release((3,), ledger, crc, gp_code=STUB)
    s.pll4_step_to(800)
    with s.l2cpu_batch(3):
        for a in (regs.MB_STATUS, regs.MB_ACTION, regs.MB_IDENT, regs.MB_ESCRATCH):
            with pytest.raises(K.PolicyError, match="X280-owned"):
                s.uc_write32(a, 1)
        with pytest.raises(K.PolicyError, match="not host-owned"):
            s.wc_write_lines(regs.MB_ACTION, bytearray(64))
        with pytest.raises(K.PolicyError, match="L38"):
            s.uc_write32(regs.MB_HOST_GONE, 1)
        s.uc_write32(regs.MB_HOST_GONE, 0)  # the attach clear
        s.wc_write_lines(regs.MB_OBS, bytearray(64 * 33))
        s.uc_write32(regs.MB_SEQ_REQ, 1)
    s.close()


# --- SET_NOC_CLEANUP ----------------------------------------------------------------------------


def test_noc_cleanup_requires_initialised_line_and_fires_on_close(k, dev):
    n = len(dev.ioctls)
    with pytest.raises(K.PolicyError, match="before L38 is initialised"):
        k.set_noc_cleanup(3)
    assert len(dev.ioctls) == n
    with k.l2cpu_batch(3):
        k.lim_init(regs.MAILBOX_BASE, regs.MAILBOX_BASE + 64 * 64)
    fd = k.open_test_fd()  # E1a check on a separate, power-aware fd
    assert dev.fds[fd]["power_aware"] is True and dev.fds[k.fd]["power_aware"] is False
    k.set_noc_cleanup(3, fd=fd)
    assert dev.peek32(T3, regs.MB_HOST_GONE) == 0
    k.close_test_fd(fd)
    assert dev.peek32(T3, regs.MB_HOST_GONE) == regs.HOST_GONE_VALUE
    with k.l2cpu_batch(3):
        assert k.read_lim32(regs.MB_HOST_GONE) == 0x474F_4E45
        k.uc_write32(regs.MB_HOST_GONE, 0)
    k.set_noc_cleanup(3)
    k.close()  # keeper fd: the KMD writes GONE after the last mapping and the fd are gone
    assert dev.peek32(T3, regs.MB_HOST_GONE) == regs.HOST_GONE_VALUE
    assert [c[0] for c in dev.calls[-3:]] == ["close", "cleanup", "release"]


def test_noc_cleanup_cleared_does_not_fire(k, dev):
    with k.l2cpu_batch(3):
        k.lim_init(regs.MB_HOST_GONE, regs.MB_HOST_GONE + 64)
    k.set_noc_cleanup(3)
    k.set_noc_cleanup(3, enabled=False)
    k.close()
    assert dev.peek32(T3, regs.MB_HOST_GONE) == 0 and "cleanup" not in [c[0] for c in dev.calls]


# --- telemetry, cross-read, real backend ---------------------------------------------------------


def test_telemetry_like_kmd(k, dev):
    assert k.read_telemetry() == {regs.TAG_TIMER_HEARTBEAT: 123_456, regs.TAG_ENABLED_L2CPU: 0xF}
    dev.set_heartbeat(123_457)
    assert k.timer_heartbeat() == 123_457
    dev.poke32(ARC, regs.TELEMETRY_TABLE_PTR, 0x2000_0000)  # outside CSM
    with pytest.raises(K.KmdError, match="telemetry not available"):
        k.read_telemetry()


def test_translated_cross_read(dev):
    s = K.Kmd(dev, cross_read=True).open()
    dev.advance(2)
    with s.l2cpu_batch(3):
        assert s.read_ccache_config(3, translated=True) == s.read_ccache_config(3) == 0x0609_1004
    assert "read-only" in regs.classify_write(regs.L2CPU_XY_TRANSLATED[3], regs.LIM_BASE, 4)[2]
    s.close()


def test_real_backend_refuses_outside_devx(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("os.open reached: the device would have been opened")

    monkeypatch.setattr(K.os, "open", boom)
    monkeypatch.delenv("WITH_DEVICE_RESET_AFTER", raising=False)
    with pytest.raises(K.DeviceAccessRefused, match="DEVX"):
        K.RealBackend().open(power_aware=False)
    monkeypatch.setenv("WITH_DEVICE_RESET_AFTER", "1")
    with pytest.raises(K.DeviceAccessRefused, match="_device_guard"):
        K.RealBackend().open(power_aware=True)


# --- dry-run review shape (what T-3.4 checks on x280d's op list) ---------------------------------


def test_dry_run_boot_shape(tmp_path):
    led = L.Ledger(tmp_path / "l.json")
    s = K.Kmd.dry_run()
    dev = s.backend
    s.open()
    s.wait_settled()
    led.start_hold("dry", s.timer_heartbeat())
    assert s.read_l2cpu_reset() == 0x0F
    with s.l2cpu_batch(3):
        assert s.read_wayenable() == 0
        s.read_ccache_ecc(3)
        s.lim_init()
        crc = s.write_image(3, os.urandom(4096))
        s.write_gp_scratch(3, STUB)
        s.write_boot_vectors(3)
    s.pll4_step_to(200)
    s.release((3,), led, crc, gp_code=STUB)
    s.pll4_step_to(800)
    with s.l2cpu_batch(3):
        s.l2pf_setup(3)
        s.set_noc_cleanup(3)
    s.close()
    text = s.oplog_text()
    assert "REFUSED" not in text and "W32" in text
    w = _writes(s)
    for o in w:  # every write target is in the allowlist
        allow, denies, why = regs.classify_write(o.xy, o.addr, o.size)
        assert allow is not None and all(d.exception == allow.ctx for d in denies), (str(o), why)
    gddr = [d for d in regs.DENY if regs.GDDR_REACHING in d.why or d.name == "CCACHE0_WAYENABLE"]
    for o in [o for o in s.oplog if o.kind in ("W32", "WC64")]:
        assert not [d for d in gddr if d.lo <= o.addr <= d.hi and regs._kind(o.xy) == d.tiles], str(o)
    rst = [i for i, o in enumerate(s.oplog) if o.kind == "W32" and o.addr == regs.L2CPU_RESET]
    assert len(rst) == 1
    i = rst[0]
    prev = [o for o in s.oplog[:i] if o.kind == "R32" and o.addr == regs.L2CPU_RESET][-1]
    nxt = next(o for o in s.oplog[i + 1:] if o.kind in ("R32", "W32"))
    v = s.oplog[i].value
    assert v == prev.value | 0x80 and v & 0xF == 0xF and (nxt.kind, nxt.addr, nxt.value) == ("R32", regs.L2CPU_RESET, v)
    assert dev.names().count("SET_NOC_CLEANUP") == 1 and set(dev.names()) <= set(K.IOCTL_NAMES.values())
    assert [c for c in dev.calls if c[0] == "cleanup"]
    summ = s.txn_summary()
    assert summ["wc64"] == regs.LIM_LINES + 64 and summ["wc64_all_aligned_64B"] and summ["wc64_all_fenced"]
    assert summ["lim_partial_before_init"] == 0 and summ["refused"] == 0 and not summ["log_wrapped"]


# --- review:x280 fixes (F2-F6) and the checked line read -------------------------------------------------------


@pytest.fixture
def live(dev, ledger):
    """A session past boot step 8 (L2CPU3 released and running at 800 MHz)."""
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    crc = _artefacts(s)
    s.pll4_step_to(200)
    s.release((3,), ledger, crc, gp_code=STUB)
    s.pll4_step_to(800)
    yield s
    if s.fd is not None:
        s.close()


@pytest.mark.parametrize("skip", ["lim", "image", "gp", "vectors", "gp_other", "image_crc", "image_overwritten"])
def test_release_refuses_without_each_boot_artefact(dev, ledger, skip):
    """F2: release() itself checks the §5.2 E2 preconditions, each read back in this session; nothing is written
    and the ledger tile stays fresh when one is missing."""
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    crc = 0
    with s.l2cpu_batch(3):
        if skip != "lim":
            s.lim_init()
        else:
            s.lim_init(regs.LIM_BASE, regs.WEIGHTS_BASE)  # the image region only: LIM not fully initialised
        if skip != "image":
            crc = s.write_image(3, IMG)
        if skip == "image_overwritten":
            s.wc_write_lines(regs.IMAGE_BASE + 64, bytearray(64))  # after the read-back: no longer verified
        if skip != "gp":
            s.write_gp_scratch(3, STUB if skip != "gp_other" else bytes(range(64)))
        if skip != "vectors":
            s.write_boot_vectors(3)
    s.pll4_step_to(200)
    with pytest.raises(K.PolicyError, match="release refused"):
        s.release((3,), ledger, crc ^ (skip == "image_crc"), gp_code=STUB)
    assert dev.peek32(ARC, regs.L2CPU_RESET) == 0x0F and not [o for o in _writes(s) if o.addr == regs.L2CPU_RESET]
    assert ledger.state(3) == "fresh" and not s.reset_rmw_done
    s.close()


def test_stage0_release_needs_approval_and_its_own_vectors(dev, ledger, monkeypatch):
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    with s.l2cpu_batch(3):
        s.write_gp_scratch(3, STUB)
        s.write_boot_vectors(3, reset_pc=regs.STAGE0_ENTRY)
    s.pll4_step_to(200)
    monkeypatch.delenv("X280_E1F_APPROVED", raising=False)
    with pytest.raises(K.PolicyError, match="DP3"):
        s.release((3,), ledger, None, gp_code=STUB, stage0=True)
    with pytest.raises(K.PolicyError, match="vectors"):  # stage-0 vectors with a normal release
        s.release((3,), ledger, None, gp_code=STUB)
    assert dev.peek32(ARC, regs.L2CPU_RESET) == 0x0F
    s.close()


def test_e1f_lim_states_after_a_stage0_release(dev, ledger, monkeypatch):
    """F1/F4 on E1F: no host LIM write while stage 0 zero-fills; then the image only via write_image and the entry
    word only via stage0_enter; afterwards the monitor region is the hart's; the GP scratch is swapped once."""
    monkeypatch.setenv("X280_E1F_APPROVED", "1")
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    with s.l2cpu_batch(3):
        s.write_gp_scratch(3, STUB)
        s.write_boot_vectors(3, reset_pc=regs.STAGE0_ENTRY)
    s.pll4_step_to(200)
    s.release((3,), ledger, None, gp_code=STUB, stage0=True)
    s.pll4_step_to(800)
    with s.l2cpu_batch(3):
        for a in (regs.WEIGHTS_BASE, regs.MB_OBS, regs.IMAGE_BASE):
            with pytest.raises(K.PolicyError, match="zero-filling"):
                s.wc_write_lines(a, bytearray(64))
        with pytest.raises(K.PolicyError, match="E1F state zeroing"):
            s.write_image(3, IMG)
        s.assume_lim_initialised(3, regs.LIM_BASE, regs.LIM_END, "test: stage-0 done word seen")
        with pytest.raises(K.PolicyError, match="not host-owned"):
            s.wc_write_lines(regs.IMAGE_BASE, bytearray(64))  # plain writes: still the hart's region
        with pytest.raises(K.PolicyError, match="entry word"):
            s.uc_write32(regs.STAGE0_ENTRY_WORD, regs.RESET_PC)
        with pytest.raises(K.PolicyError, match="stage0_enter"):
            s.stage0_enter(3)  # no image yet
        s.write_image(3, IMG)
        with pytest.raises(K.PolicyError, match="only before release"):
            s.write_gp_scratch(3, STUB)  # not before the monitor runs
        s.stage0_enter(3)
        assert dev.peek32(T3, regs.STAGE0_ENTRY_WORD) == regs.RESET_PC
        with pytest.raises(K.PolicyError):
            s.write_image(3, IMG)  # crt0 runs now: the monitor region is X280-owned
        s.write_gp_scratch(3, STUB)  # the one swap stage 0 -> park stub
        with pytest.raises(K.PolicyError, match="only before release"):
            s.write_gp_scratch(3, STUB)
        s.wc_write_lines(regs.WEIGHTS_BASE, bytearray(64))  # host lines stay host lines
    s.close()


def test_lim_init_refused_after_or_without_a_known_release_state(dev, live):
    """F3: LIM init is pre-release only, also through wc_write_lines(..., _ctx='lim_init')."""
    with live.l2cpu_batch(3):
        with pytest.raises(K.PolicyError, match="lim_init"):
            live.lim_init()
        with pytest.raises(K.PolicyError, match="after release"):
            live.wc_write_lines(regs.MB_STATUS, bytearray(64), _ctx="lim_init")
        with pytest.raises(K.PolicyError, match="after release"):
            live.wc_write_lines(regs.WEIGHTS_BASE, bytearray(64), _ctx="lim_init")
    s = K.Kmd(K.FakeDevice.default_chip()).open()  # never read L2CPU_RESET: released unknown -> refused
    s.backend.advance(2)
    with s.l2cpu_batch(3):
        with pytest.raises(K.PolicyError, match="released=None"):
            s.lim_init()
    s.close()


def test_lim_init_check_is_method_a_s_guard_without_writes(k, dev):
    """verify:all D12: method B's pre-write check (server_g1 writes, not kmd) refuses what lim_init would - a range
    outside LIM or not line-aligned, any access outside an l2cpu_batch - and writes or marks nothing itself."""
    n0 = len(dev.calls)
    with pytest.raises(K.InterlockError, match="outside l2cpu_batch"):
        k.lim_init_check(regs.WEIGHTS_BASE, regs.WEIGHTS_BASE + 4096, 3)
    with k.l2cpu_batch(3):
        assert k.lim_init_check(regs.LIM_BASE, regs.LIM_END, 3) == regs.LIM_LINES
        for lo, hi in ((regs.LIM_BASE - 64, regs.LIM_BASE + 64), (regs.WEIGHTS_BASE + 4, regs.WEIGHTS_BASE + 68),
                       (regs.LIM_END - 64, regs.LIM_END + 64)):
            with pytest.raises(K.PolicyError, match="lim_init range"):
                k.lim_init_check(lo, hi, 3)
    assert not [o for o in k.oplog if o.kind in ("W32", "WC64")] and k.lim[3].count() == 0
    assert len(dev.calls) == n0 and dev.peek(T3, regs.WEIGHTS_BASE, 64) == bytes(64)
    assert [o.detail for o in k.oplog if o.kind == "CHECK"] == ["lim_init (method B, written by server_g1)"]


def test_lim_init_check_refuses_a_released_or_unknown_tile(live):
    """verify:all D12: the per-call release check of lim_init (review F3) applies to method B too."""
    with live.l2cpu_batch(3):  # a released tile: refused like lim_init, for every range
        for lo, hi in ((regs.LIM_BASE, regs.LIM_END), (regs.WEIGHTS_BASE, regs.WEIGHTS_BASE + 64)):
            with pytest.raises(K.PolicyError, match="released=True"):
                live.lim_init_check(lo, hi, 3)
    s = K.Kmd(K.FakeDevice.default_chip()).open()  # L2CPU_RESET never read: unknown -> refused
    s.backend.advance(2)
    with s.l2cpu_batch(3):
        with pytest.raises(K.PolicyError, match="released=None"):
            s.lim_init_check(regs.WEIGHTS_BASE, regs.WEIGHTS_BASE + 64, 3)
    s.close()


def test_monitor_region_is_x280_owned_after_release(dev, live):
    """F4: .text, .bss, .persist and the stack (x280/fw/lim.ld) refuse host stores once the harts run."""
    with live.l2cpu_batch(3):
        for a in (0x0800_0000, 0x0800_3B00, 0x0800_59C0, 0x0800_FFC0):
            with pytest.raises(K.PolicyError, match="monitor region"):
                live.wc_write_lines(a, bytearray(64))
            with pytest.raises(K.PolicyError, match="X280-owned"):
                live.uc_write32(a + 8, 0)
        with pytest.raises(K.PolicyError, match="monitor region"):  # a range straddling the region's end
            live.wc_write_lines(0x0800_FF80, bytearray(64 * 4))
        before = dev.peek(T3, regs.IMAGE_BASE, 64)
        live.wc_write_lines(regs.WEIGHTS_BASE, bytearray(64 * 8))  # payload / weights region: still the host's
        assert dev.peek(T3, regs.IMAGE_BASE, 64) == before
    assert regs.lim_line_owner(0x0800_1000) == "host" and regs.lim_line_owner(0x0800_1000, released=True) == "x280"


def _relock_at(dev, n_at: int, seen: list):
    n = [0]

    def hook():
        n[0] += 1
        if n[0] == n_at:  # CMFW re-programs PLL4 from its cached settings (fbdiv 64, postdivs 1)
            dev.poke32(ARC, regs.PLL4_CNTL_1, regs.PLL4_CNTL_1_DEFAULT)
            dev.poke32(ARC, regs.PLL4_CNTL_5, regs.PLL4_CNTL_5_DEFAULT)
        seen.append(max(regs.pll4_mhz(dev.peek32(ARC, regs.PLL4_CNTL_1), dev.peek32(ARC, regs.PLL4_CNTL_5),
                                          dev.peek32(ARC, regs.PLL4_USE_POSTDIV))[1]))
    return hook


@pytest.mark.parametrize("mhz,at", [(800, 10), (800, 1), (800, 119), (200, 60)])
def test_pll4_walk_stops_at_a_concurrent_relock(dev, ledger, mhz, at):
    """F5: every step re-reads PLL4; on the first mismatch nothing more is written, so L2CPUCLK never exceeds
    800 MHz (the reviewer's run reached 1462 MHz with the harts running)."""
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    if mhz == 800:
        crc = _artefacts(s)
        s.pll4_step_to(200)
        s.release((3,), ledger, crc, gp_code=STUB)
    seen = []
    s.step_sleep = _relock_at(dev, at, seen)
    n0 = len(_writes(s))
    with pytest.raises(K.InterlockError, match="PLL4 changed under the walk"):
        s.pll4_step_to(mhz)
    assert max(seen) <= 800.0 and len(_writes(s)) - n0 == at  # stopped right after the relock
    assert "CMFW relock" in s.pll_interlock and s.pll_mhz is None
    assert (dev.peek32(ARC, regs.PLL4_CNTL_1), dev.peek32(ARC, regs.PLL4_CNTL_5)) == (
        regs.PLL4_CNTL_1_DEFAULT, regs.PLL4_CNTL_5_DEFAULT)  # left to CMFW
    s.close()


@pytest.mark.parametrize("reg,value", [(regs.PLL4_CNTL_0, 0x02), (regs.PLL4_CNTL_0, 0x00), (regs.PLL_LOCK, 0x0F)])
def test_pll4_walk_stops_while_cmfw_is_relocking(dev, reg, value):
    """F5: a relock in flight (bypass on, powered down, or PLL4 unlocked) stops the walk before the next write."""
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    n = [0]

    def hook():
        n[0] += 1
        if n[0] == 5:
            dev.poke32(ARC, reg, value)
    s.step_sleep = hook
    with pytest.raises(K.InterlockError, match="before write 6/120"):
        s.pll4_step_to(200)
    assert len(_writes(s)) == 5
    s.close()


def test_translated_window_only_with_noc_translation():
    """F6: TAG_NOC_TRANSLATION (FW telemetry.h:173) = 0 -> the (8,29) window is never allocated, reads refused."""
    dev = K.FakeDevice.default_chip(noc_translation=0)
    s = K.Kmd(dev, cross_read=True).open()
    dev.advance(2)
    assert s.noc_translation == 0 and "ccache3x" not in s.windows
    assert not [i for i in dev.ioctls if i[1] == "CONFIGURE_TLB" and (i[2]["x_end"], i[2]["y_end"]) == (8, 29)]
    with s.l2cpu_batch(3):
        with pytest.raises(K.InterlockError, match="NoC translation off"):
            s.read_ccache_config(3, translated=True)
        assert s.read_ccache_config(3) == 0x0609_1004
    s.close()
    s = K.Kmd(K.FakeDevice.default_chip(), cross_read=False).open()  # no cross read: translation never even read
    assert s.noc_translation is None
    s.close()


def test_checked_word_and_line_reads(k, dev):
    """The 64 B line read: one libwcio call, every word still a logged 4 B UC read, policy checked once."""
    with k.l2cpu_batch(3):
        k.lim_init(regs.MAILBOX_BASE, regs.MAILBOX_BASE + 64 * 64)
        dev.poke(T3, regs.MB_ACTION, bytes(range(64)))
        first = k.wcio.log_total()
        assert k.read_lim_line(regs.MB_ACTION) == bytes(range(64))
        tx = [t for t in k.txn_log() if t["seq"] >= first]
        assert [t["op"] for t in tx] == ["UC_R32"] * 16
        assert [t["noc"] for t in tx] == [regs.MB_ACTION + 4 * i for i in range(16)]
        assert k.read_lim_words(regs.MB_STATUS + 8, 2) == [0, 0]
        for addr, n in ((regs.MB_ACTION + 4, 16), (regs.MB_ACTION + 60, 2), (regs.MB_ACTION + 2, 1), (regs.MB_ACTION, 17)):
            with pytest.raises(K.PolicyError, match="one 64 B line"):
                k.read_lim_words(addr, n)
        with pytest.raises(K.PolicyError, match="uninitialised"):
            k.read_lim_line(regs.WEIGHTS_BASE)
    with pytest.raises(K.WcioError):
        k.wcio.uc_read_words(k.windows["lim3"].uc + 60, 2)  # the C side refuses a line crossing too


def test_status_telemetry_decoded_from_the_cmfw_table():
    """The HUD chip-panel telemetry: CMFW encodings (FW telemetry.c:464-541), table walked once then cached. The
    values are this test's own (FAKE_TELEMETRY): kmd's FakeDevice holds none (verify:all D4)."""
    dev = K.FakeDevice.default_chip(telemetry=FAKE_TELEMETRY)
    k = K.Kmd(dev).open()
    dev.advance(1.5)
    k.read_l2cpu_reset()
    t = k.read_status_telemetry()
    assert t == {"aiclk_mhz": 800, "l2cpuclk_mhz": [800] * 4, "asic_temp_c": 45.5, "vcore_mv": 750, "tdp_w": 31,
                 "tdc_a": 41, "input_power_w": 62, "timer_heartbeat": 123_456}
    n = len(k.oplog)
    data = dev.peek32(ARC, regs.TELEMETRY_DATA_PTR)
    dev.poke32(ARC, data + 4 * regs.TAG_AICLK, (1350 << 16) | 1000)  # upper 16 bits: "MAX (not available)"
    dev.poke32(ARC, data + 4 * regs.TAG_ASIC_TEMPERATURE, regs.TELEMETRY_Q16_ERROR)
    t2 = k.read_status_telemetry()
    assert t2["aiclk_mhz"] == 1000 and t2["asic_temp_c"] is None
    assert len(k.oplog) - n == 2 + len(regs.STATUS_TELEMETRY_TAGS)  # 2 pointers + one read per tag
    k.pll4_step_to(200)
    dev.update_l2cpuclk_telemetry()
    assert k.read_status_telemetry()["l2cpuclk_mhz"] == [200] * 4
    assert not [o for o in k.oplog if o.kind in ("W32", "WC64") and o.xy == ARC and o.addr >= regs.ARC_CSM_BASE
                and o.addr < regs.ARC_CSM_BASE + regs.ARC_CSM_SIZE]
    k.close()
    assert not hasattr(K.FakeDevice, "FAKE_TELEMETRY")  # no invented chip values outside the tests
    bare = K.FakeDevice.default_chip()
    data = bare.peek32(ARC, regs.TELEMETRY_DATA_PTR)
    assert all(bare.peek32(ARC, data + 4 * tag) == 0 for tag in FAKE_TELEMETRY)


def test_beu_read_only_once_the_harts_run(k, live):
    """F7: the per-hart BEU page is never read while the harts are in reset."""
    with k.l2cpu_batch(3):
        with pytest.raises(K.InterlockError, match="BEU read"):
            k.read_beu(3, 0)
    with live.l2cpu_batch(3):
        assert live.read_beu(3, 0)["accrued"] == 0


def test_rnmi_trigger_read_is_transient_and_read_only(live, dev):
    with live.l2cpu_batch(3):
        assert live.read_rnmi_trigger(3) == 0
        live.rnmi_set(3, 1)
        assert live.read_rnmi_trigger(3) == 1
        live.rnmi_set(3, 0)
    xp = live.windows["xp3"].tlb_id
    ro = [c for c in dev.calls if c[0] == "mmap" and c[2] == xp and not c[5] & K.PROT_WRITE]
    assert len(ro) == 2 and live.windows["xp3"].uc == 0  # no persistent XP mapping


# --- ledger -------------------------------------------------------------------------------------


def test_ledger_schema_and_transitions(tmp_path):
    p = tmp_path / "x280_ledger.json"
    led = L.Ledger(p)
    out = led.start_hold("hold-1", 5000, now=1_700_000_000.0)
    assert out["epoch"] == "first_hold" and out["previous"] is None
    rec = json.loads(p.read_text())
    assert sorted(rec) == sorted(L.KEYS) and sorted(rec["tiles"]) == ["0", "1", "2", "3"]
    assert all(sorted(v) == sorted(L.TILE_KEYS) and v["state"] == "fresh" for v in rec["tiles"].values())
    led.mark_released(3, 0xDEADBEEF)
    led.mark_parked(3)
    led.mark_resumed(3)
    with pytest.raises(L.LedgerError):
        led.transition(3, "fresh")  # a spent release never comes back
    led.mark_burned(3)
    with pytest.raises(L.LedgerError):
        led.mark_resumed(3)
    rec = led.read()
    assert rec["tiles"]["3"]["state"] == "burned" and rec["tiles"]["3"]["image_crc"] == 0xDEADBEEF
    assert rec["tiles"]["3"]["released_at"] and led.burned_count() == 1
    with pytest.raises(L.LedgerError):
        L.validate({**rec, "extra": 1})
    out = led.start_hold("hold-2", 100, now=1_700_000_100.0)  # chip reset in between: counter restarted
    assert out["epoch"] == "reset" and out["previous"]["hold_id"] == "hold-1"
    assert led.state(3) == "fresh" and len((tmp_path / "x280_ledger.json.history.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("hb,verdict", [(100 + 1000, "no_reset"), (60, "reset"), (100 + 5000, "inconsistent")])
def test_ledger_epoch_verdict(hb, verdict):
    prev = L.new_record("p", 100, "2026-10-24T00:00:00.000Z")
    t_now = L._parse_time(prev["started_at"]) + 100.0  # 100 s later = 1000 heartbeat ticks at 10 Hz
    assert L.epoch_verdict(prev, hb, t_now) == verdict


# --- S1d store widths (jobs/s1d.sh): libwcio loops, the allowlist, kmd.lim_init's store variants -----------------------


def _disasm(sym: str) -> list:
    out = subprocess.run(["llvm-objdump-20", "-d", "--no-show-raw-insn", f"--disassemble-symbols={sym}",
                          str(HOST / "libwcio.so")], check=True, capture_output=True, text=True).stdout
    return [ln.split(":", 1)[1].strip() for ln in out.splitlines() if re.match(r"\s+[0-9a-f]+:", ln)]


def _mem_stores(insns: list) -> list:
    """Instructions whose last AT&T operand is a memory operand (a store), cmp/test/prefetch/nop excluded."""
    out = []
    for ins in insns:
        mnem, _, ops = ins.partition("\t")
        last = ops.rsplit(", ", 1)[-1] if ", " in ops else ""
        if "(" in last and not mnem.startswith(("nop", "cmp", "test", "prefetch")):
            out.append(ins)
    return out


def test_s1d_store_loops_disassembly():
    """wcio__stream_lines_2x32: exactly two regular 32 B ymm stores per line (low half first), no zmm, no NT store,
    no memcpy call; wcio__store_words64: only 64-bit GPR stores (one aligned 8 B UC store each)."""
    ins = _disasm("wcio__stream_lines_2x32")
    st = _mem_stores(ins)
    assert len(st) == 2 and all(re.match(r"vmovdqu\s+%ymm\d+, ", s) for s in st), st
    assert re.search(r", \(%\w+\)$", st[0]) and re.search(r", 0x20\(%\w+\)$", st[1]), st  # +0 then +32
    assert not [i for i in ins if "zmm" in i or "vmovnt" in i or i.startswith("call")] and any(i == "sfence" for i in ins)
    ins = _disasm("wcio__store_words64")
    st = _mem_stores(ins)
    assert st and all(re.match(r"movq\s+%r(?:[a-d]x|[sd]i|bp|\d+), ", s) for s in st), st
    assert not [i for i in ins if i.startswith("call") or "mm" in i]


STORE_TXN = {"a64": ["WC64", "SFENCE"] * 4, "a64nf": ["WC64"] * 4 + ["SFENCE"],
             "wc2x32": ["WC32"] * 8 + ["SFENCE"], "uc8": ["UC_W64"] * 32}


@pytest.mark.parametrize("store", list(K.LIM_INIT_STORES))
def test_lim_init_store_variants_write_whole_lines_with_their_widths(k, dev, store):
    lo = regs.S1D_BASE
    pat = lambda a: bytes((a + i * 13) & 0xFF for i in range(64))  # noqa: E731
    with k.l2cpu_batch(3):
        first = k.wcio.log_total()  # after the batch's own interlock reads
        assert k.lim_init(lo, lo + 4 * 64, store=store, pattern=pat) == 4
        assert k.lim[3].is_init(lo, 4 * 64)
        k.uc_write32(lo + 8, 0x1234_5678)  # the lines count as initialised for R26's partial-write rule
    assert dev.peek(T3, lo + 64, 3 * 64) == b"".join(pat(lo + 64 * i) for i in range(1, 4))
    tx = [t for t in k.txn_log() if t["seq"] >= first]
    assert [t["op"] for t in tx][:-1] == STORE_TXN[store] and tx[-1]["op"] == "UC_W32"
    if store == "wc2x32":
        assert [t["noc"] for t in tx[:8]] == [lo + 32 * i for i in range(8)] and all(t["kind"] == "wc" for t in tx[:8])
        assert all(t["size"] == 32 for t in tx[:8])
    if store == "uc8":
        assert [t["noc"] for t in tx[:32]] == [lo + 8 * i for i in range(32)]
        assert all(t["kind"] == "uc" and t["size"] == 8 for t in tx[:32])
    summ = k.txn_summary()
    assert summ["lim_partial_before_init"] == 0 and summ["refused"] == 0


def test_s1d_store_widths_are_lim_init_only(k, dev, live):
    """LIM_INIT_WC32 / LIM_INIT_UC8: LIM only, ctx lim_init only (so never a mailbox or post-release write), whole
    aligned lines; every refusal writes nothing."""
    for size, name in ((8, "LIM_INIT_UC8"), (32, "LIM_INIT_WC32")):
        assert regs.classify_write(T3, regs.LIM_BASE, size)[0].name == name
        assert regs.classify_write(T3, regs.LIM_END - 8, size)[0] is None or size == 8
        assert regs.classify_write(T3, regs.CCACHE0_WAYENABLE, size)[0] is None
        assert regs.classify_write(ARC, regs.PLL4_CNTL_1, size)[0] is None
        assert regs.classify_write(regs.L2CPU_XY[0], regs.LIM_BASE, size)[0] is None  # tile 0 is read-only
    lo = regs.S1D_BASE
    before = dev.peek(T3, lo, 128)
    with k.l2cpu_batch(3):
        with pytest.raises(K.PolicyError, match="LIM_INIT_WC32 is writable only via lim_init"):
            k.wc_write_lines(lo, bytearray(b"\x11" * 64), _store="2x32")  # ctx "": e.g. a mailbox write
        with pytest.raises(K.PolicyError, match="LIM_INIT_UC8 is writable only via lim_init"):
            k._uc8_write_lines(lo, bytearray(b"\x11" * 64), 3, "")
        with pytest.raises(K.PolicyError, match="store"):
            k.lim_init(lo, lo + 64, store="nt128")
        for a, b in ((lo + 8, lo + 72), (lo, lo + 60), (regs.LIM_END - 64, regs.LIM_END + 64)):
            with pytest.raises(K.PolicyError):
                k.lim_init(a, b, store="uc8")
        with pytest.raises(K.WcioError, match="misaligned"):
            k.wcio.uc_store64(k.windows["lim3"].uc + 4, ctypes.addressof(ctypes.create_string_buffer(8)), 1)
        k.wcio.force_no_avx512(True)  # the test hook turns AVX off too: the 2x32 loop refuses, writes nothing
        try:
            with pytest.raises(K.WcioError, match="nothing written"):
                k.lim_init(lo, lo + 64, store="wc2x32")
        finally:
            k.wcio.force_no_avx512(False)
    assert dev.peek(T3, lo, 128) == before and not k.lim[3].is_init(lo)
    with live.l2cpu_batch(3):  # after the release: no LIM init store of any width
        for store in K.LIM_INIT_STORES:
            with pytest.raises(K.PolicyError, match="released"):
                live.lim_init(lo, lo + 64, store=store)


# --- FakeEcc: the CCACHE LIM ECC model behind --sim-ecc (S1d's verdict logic is tested both ways) --------------------


def test_fake_has_no_ecc_model_unless_asked(k, dev):
    assert dev.ecc is None and not hasattr(dev, "ecc_access") and k._ecc_hook is None
    with k.l2cpu_batch(3):
        k.lim_init(regs.S1D_BASE, regs.S1D_BASE + 64 * 64)
        assert k.read_ccache_ecc_counts(3) == {"fix": 0, "fail": 0}


def test_fake_ecc_poison_reproduces_s1c_e1a():
    """'poison' (the S1c hypothesis) on a whole-LIM method A init: S1c's per-access event rates, the last events at
    the last line + 0x20 (two 32 B accesses per line), 0.935 fail events per 4 B re-read, 0 fixes, the bytes intact;
    a partial write and the KMD's L38 cleanup write count as accesses (S1c: dat_fail_low = L38 after E1a)."""
    dev = K.FakeDevice.default_chip(ecc="poison")
    k = K.Kmd(dev).open()
    dev.advance(1.5)
    k.read_l2cpu_reset()
    try:
        with k.l2cpu_batch(3):
            e0 = k.read_ccache_ecc(3)
            assert (e0["dat_fail_count"], e0["dat_fail_low"]) == (0, 0xA7D6_F189)  # S1c E0: garbage until a count moves
            k.lim_init(pattern=lambda a: a.to_bytes(8, "little") * 8)
            e1 = k.read_ccache_ecc(3)
            per = {kk: e1[f"dat_{kk}_count"] / regs.LIM_LINES for kk in ("fix", "fail")}
            assert per["fix"] == pytest.approx(57_681 / 30_720, rel=0.02) and per["fail"] == pytest.approx(
                57_447 / 30_720, rel=0.02)
            assert e1["dat_fix_low"] == e1["dat_fail_low"] == regs.LIM_END - 64 + 0x20 and e1["dat_fail_high"] == 0
            for a in range(regs.LIM_BASE, regs.LIM_END, 64):
                assert k.read_lim_words(a, 2) == [a & 0xFFFF_FFFF, a >> 32]
            e2 = k.read_ccache_ecc(3)
            assert e2["dat_fix_count"] == e1["dat_fix_count"]
            assert (e2["dat_fail_count"] - e1["dat_fail_count"]) / (2 * regs.LIM_LINES) == pytest.approx(0.935, abs=0.01)
            n = dev.ecc.counts(3)
            assert n["fresh"] == 0 and n["poison"] / (n["poison"] + n["ok"]) == pytest.approx(0.935, abs=0.01)
            k.set_noc_cleanup(3)
            g = (regs.MB_HOST_GONE - regs.LIM_BASE) // K.FakeEcc.GRANULE
            l38_poisoned = dev.ecc.tiles[3]["state"][g] == K.FakeEcc.POISON
        k.close()  # the KMD's cleanup: a 4 B write into L38, a read-modify-write like any other
        e3 = {kk: dev.peek32(T3, regs.CCACHE0_ECC[kk]) for kk in ("dat_fail_count", "dat_fail_low")}
        if l38_poisoned:
            assert e3 == {"dat_fail_count": e2["dat_fail_count"] + 1, "dat_fail_low": regs.MB_HOST_GONE}
        else:
            assert e3 == {"dat_fail_count": e2["dat_fail_count"], "dat_fail_low": e2["dat_fail_low"]}
        assert dev.peek32(T3, regs.MB_HOST_GONE) == regs.HOST_GONE_VALUE  # the bytes land either way
    finally:
        if k.fd is not None:
            k.close()


@pytest.mark.parametrize("mode,clean_stores", [("clean", {"a64", "a64nf", "wc2x32", "uc8"}), ("granule", {"uc8"}),
                                               ("poison", set())])
def test_fake_ecc_modes(mode, clean_stores):
    dev = K.FakeDevice.default_chip(ecc=mode)
    k = K.Kmd(dev).open()
    dev.advance(1.5)
    k.read_l2cpu_reset()
    try:
        with k.l2cpu_batch(3):
            seen = set()
            for i, store in enumerate(K.LIM_INIT_STORES):
                lo = regs.S1D_BASE + i * 256 * 64
                k.lim_init(lo, lo + 256 * 64, store=store)
                c0 = k.read_ccache_ecc_counts(3)
                for a in range(lo, lo + 256 * 64, 64):
                    k.read_lim_words(a, 16)
                if k.read_ccache_ecc_counts(3) == c0:
                    seen.add(store)
            assert seen == clean_stores
            # a sub-granule (4 B) write onto never-written SRAM is a read-modify-write in every mode (R26's hazard)
            c0 = k.read_ccache_ecc_counts(3)
            dev.ecc.access(dev, 3, "uc_w32", regs.LIM_END - 4096, 64)
            assert k.read_ccache_ecc_counts(3)["fail"] > c0["fail"]
    finally:
        k.close()


def test_fake_ecc_refuses_unknown_modes_and_accesses_outside_lim():
    with pytest.raises(ValueError, match="mode"):
        K.FakeDevice.default_chip(ecc="sometimes")
    dev = K.FakeDevice.default_chip(ecc="poison")
    with pytest.raises(ValueError, match="leaves LIM"):
        dev.ecc.access(dev, 3, "uc_w64", regs.LIM_END - 8, 2)
    dev._ecc_access(T3, "uc_w32", regs.CCACHE0_WAYENABLE, 1)  # not LIM: ignored
    assert dev.ecc.accesses == {"read": 0, "write": 0}


# --- PLL4 walk hardening (verify B1, B2) and regs.pll4_mhz (B4) -------------------------------------------------------


def test_pll4_walk_policy_first_then_reads_cntl0_and_lock_last(k, dev):
    """B1: for every walk write the policy runs before the interlock reads, and the reads end with CNTL_0 and
    PLL_LOCK immediately before the store; a write the policy refuses issues no read at all."""
    n = len(k.oplog)
    k.pll4_step_to(200)
    ops = k.oplog[n:]
    ws = [i for i, o in enumerate(ops) if o.kind == "W32"]
    assert len(ws) == 120
    for i in ws:
        assert [(o.kind, o.addr) for o in ops[i - 5:i]] == [("R32", a) for a in (
            regs.PLL4_CNTL_1, regs.PLL4_CNTL_5, regs.PLL4_USE_POSTDIV, regs.PLL4_CNTL_0, regs.PLL_LOCK)]
    calls = []
    with k._transient("arc", "pll4_step") as (w32, _):  # nothing armed: the policy refuses before pre_store runs
        m = len(k.oplog)
        with pytest.raises(K.PolicyError, match="out of the clock.py sequence"):
            w32(regs.PLL4_CNTL_5, 0x0101_0102, pre_store=lambda: calls.append(1))
    assert not calls and not [o for o in k.oplog[m:] if o.kind == "R32"]


@pytest.mark.parametrize("reg,value", [(regs.PLL4_CNTL_0, 0x02), (regs.PLL_LOCK, 0x0F)])
def test_pll4_walk_sees_a_relock_starting_between_its_reads(k, dev, reg, value):
    """B1 race: CMFW starts a relock (bypass, then unlocked) right after the walk read CNTL_1/5/USE_POSTDIV for write
    7. CNTL_0 and PLL_LOCK are read after that, so the walk stops before write 7 (read first, it would not)."""
    orig, n = k._tr32, {"pd": 0}

    def tr32(va, w, addr):
        v = orig(va, w, addr)
        if addr == regs.PLL4_USE_POSTDIV:
            n["pd"] += 1
            if n["pd"] == 7:
                dev.poke32(ARC, reg, value)
        return v
    k._tr32 = tr32
    with pytest.raises(K.InterlockError, match="before write 7/120"):
        k.pll4_step_to(200)
    assert len(_writes(k)) == 6


def test_pll4_walk_stops_when_use_postdiv_changes(k, dev):
    """B2: USE_POSTDIV is re-read before every write: a cleared bit (that L2CPUCLK at the VCO, FW pll.c:438-452)
    stops the walk before the next write."""
    n = [0]

    def hook():
        n[0] += 1
        if n[0] == 5:
            dev.poke32(ARC, regs.PLL4_USE_POSTDIV, 0xE)
    k.step_sleep = hook
    with pytest.raises(K.InterlockError, match=r"before write 6/120.*USE_POSTDIV 0xe"):
        k.pll4_step_to(200)
    assert len(_writes(k)) == 5 and "USE_POSTDIV" in k.pll_interlock


def test_pll4_mhz_is_cmfw_s_formula():
    """B4: regs.pll4_mhz = CMFW's CalculateFreqFromPllRegs with GetExtPostdiv (FW pll.c:413-475): USE_POSTDIV bit
    clear -> the VCO frequency, postdiv 0 -> disabled (0 MHz), 1..16 -> /(p+1), above 16 -> /((p+1)*2)."""
    pll = (RT / "third_party_refs/fw-19.12.0/lib/tenstorrent/bh_arc/pll.c").read_text()
    body = pll[pll.index("uint32_t GetExtPostdiv("):pll.index("/* What we don't support: */")]
    assert "if (postdiv_value == 0) {\n\t\t\teff_postdiv = 0;" in body
    assert "} else if (postdiv_value <= 16) {\n\t\t\teff_postdiv = postdiv_value + 1;" in body
    assert "eff_postdiv = (postdiv_value + 1) * 2;" in body and "} else {\n\t\treturn 1;" in body
    c1 = regs.PLL4_CNTL_1_DEFAULT  # VCO 1600 MHz

    def cmfw(c5, up):  # the C, transcribed (integer division as in pll.c:474)
        out = []
        for i in range(4):
            v, en = (c5 >> 8 * i) & 0xFF, (up >> i) & 1
            eff = 1 if not en else 0 if v == 0 else v + 1 if v <= 16 else (v + 1) * 2
            out.append(0 if eff == 0 else (50 * (c1 >> 16)) // ((c1 & 0xFF) * eff))
        return out
    for c5, up in ((0x0101_0101, 0xF), (0x0F0F_0F0F, 0xF), (0x1100_0110, 0xF), (0x0101_0101, 0x0), (0x2001_0001, 0x5),
                   (0x0010_1011, 0xA)):
        vco, outs = regs.pll4_mhz(c1, c5, up)
        assert vco == 1600.0 and [int(x) for x in outs] == cmfw(c5, up), (hex(c5), up, outs)
    assert regs.pll4_mhz(c1, 0x0101_0101, 0xE)[1][0] == 1600.0  # the reason a walk needs USE_POSTDIV & 0xF == 0xF
    dev = K.FakeDevice.default_chip()  # CMFW's telemetry writer in the fake follows the same formula
    dev.poke32(ARC, regs.PLL4_USE_POSTDIV, 0xE)
    dev.update_l2cpuclk_telemetry()
    data = dev.peek32(ARC, regs.TELEMETRY_DATA_PTR)
    assert [dev.peek32(ARC, data + 4 * t) for t in regs.TAG_L2CPUCLK] == [1600, 800, 800, 800]


def test_oplog_is_bounded_and_drops_one_record_per_append():
    """S3 2026-10-25: the list op log was halved in one `del` when full (65,536 records freed at once), a 2-7 ms stall
    every ~1,070 W-WBC calls on the device path. The deque drops its oldest record per append; slices still work."""
    k = K.Kmd.dry_run(oplog_cap=64)
    for i in range(200):
        k._op("NOTE", detail=str(i))
    assert len(k.oplog) == 64 and k.oplog_dropped == 136
    assert [o.detail for o in k.oplog[-3:]] == ["197", "198", "199"] and k.oplog[0].detail == "136"
    n = len(k.oplog) - 2
    assert [o.detail for o in k.oplog[n:]] == ["198", "199"]
    assert "NOTE" in k.oplog_text()


# --- harts 1-3 (the 4-hart bench): secondary vectors and RNMI bits 1-3 ------------------------------------------------


def _vector_writes(s):
    return [(o.addr, o.value) for o in s.oplog if o.kind == "W32" and o.detail == "boot_vectors"]


def test_secondary_vectors_need_the_flag_and_the_approval(dev, monkeypatch):
    """write_boot_vectors(harts=4): only with X280_MH_APPROVED=1 and only before release; hart 0 keeps PARK_STUB /
    RNMI_EXC_STUB, harts 1-3 get RESET_PC / MH_SEC_RNMI_EXC; the same addresses in the same order as the default,
    which stays byte-identical; hart 0 can never get RESET_PC (test_boot_vectors_values_and_pre_release_only)."""
    monkeypatch.delenv("X280_MH_APPROVED", raising=False)
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    with s.l2cpu_batch(3):
        s.write_boot_vectors(3, reset_pc=regs.STAGE0_ENTRY)
        default = _vector_writes(s)
        n = len(s.oplog)
        with pytest.raises(K.PolicyError, match="X280_MH_APPROVED"):
            s.write_boot_vectors(3, reset_pc=regs.STAGE0_ENTRY, harts=4)
        assert not [o for o in s.oplog[n:] if o.kind == "W32"]          # refused before any store
        with pytest.raises(K.PolicyError, match="harts=2"):
            s.write_boot_vectors(3, harts=2)
        monkeypatch.setenv("X280_MH_APPROVED", "1")
        with pytest.raises(K.PolicyError):
            s.write_boot_vectors(3, rnmi=regs.RESET_PC, harts=4)          # hart 0 is never pointed at crt0
        n = len(s.oplog)
        s.write_boot_vectors(3, reset_pc=regs.STAGE0_ENTRY, harts=4)
        mh = [(o.addr, o.value) for o in s.oplog[n:] if o.kind == "W32" and o.detail == "boot_vectors"]
    assert [a for a, _ in mh] == [a for a, _ in default] and len(mh) == len(default) == 24
    diff = {a: v for (a, v), (_, d) in zip(mh, default, strict=True) if v != d}
    want = {}
    for h in (1, 2, 3):
        want[regs.rnmi_handler(h)] = regs.RESET_PC
        want[regs.rnmi_exc_handler(h)] = regs.MH_SEC_RNMI_EXC
    assert diff == want
    assert dev.peek32(T3, regs.rnmi_handler(0)) == regs.PARK_STUB and dev.peek32(T3, regs.rnmi_exc_handler(0)) == regs.RNMI_EXC_STUB
    for h in (1, 2, 3):
        assert dev.peek32(T3, regs.rnmi_handler(h)) == regs.RESET_PC
        assert dev.peek32(T3, regs.rnmi_exc_handler(h)) == regs.MH_SEC_RNMI_EXC
    assert s._boot[3]["mh_vectors"] and s._boot[3]["by_hart"][0] == (regs.STAGE0_ENTRY, regs.PARK_STUB, regs.RNMI_EXC_STUB)
    s.close()


def test_secondary_vector_values_only_inside_write_boot_vectors(dev, monkeypatch):
    """The policy itself: RESET_PC / MH_SEC_RNMI_EXC at a hart 1-3 handler address is refused unless
    write_boot_vectors(harts=4) is running (its arm flag), and always at hart 0's."""
    monkeypatch.setenv("X280_MH_APPROVED", "1")
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    with s.l2cpu_batch(3):
        with s._transient("xp3", "boot_vectors") as (w32, _):
            for addr, v in ((regs.rnmi_handler(1), regs.RESET_PC), (regs.rnmi_exc_handler(2), regs.MH_SEC_RNMI_EXC),
                            (regs.rnmi_handler(0), regs.RESET_PC)):
                with pytest.raises(K.PolicyError, match="not permitted"):
                    w32(addr, v)
        s._mh_arm[3] = True
        with s._transient("xp3", "boot_vectors") as (w32, _):
            w32(regs.rnmi_handler(1), regs.RESET_PC)
            with pytest.raises(K.PolicyError, match="not permitted"):
                w32(regs.rnmi_handler(0), regs.RESET_PC)
            with pytest.raises(K.PolicyError, match="not permitted"):
                w32(regs.rnmi_handler(2), regs.MH_SEC_RNMI_EXC)              # handler / exception slots not mixed
        s._mh_arm[3] = False
    s.close()


def _stage0_session(dev, ledger, monkeypatch, harts: int):
    monkeypatch.setenv("X280_E1F_APPROVED", "1")
    monkeypatch.setenv("X280_MH_APPROVED", "1")
    s = K.Kmd(dev).open()
    dev.advance(2)
    s.read_l2cpu_reset()
    with s.l2cpu_batch(3):
        s.write_gp_scratch(3, STUB)
        s.write_boot_vectors(3, reset_pc=regs.STAGE0_ENTRY, harts=harts)
    return s


def test_boot_artefacts_per_hart_and_mh_vectors_only_with_stage0(dev, ledger, monkeypatch):
    s = _stage0_session(dev, ledger, monkeypatch, 4)
    assert s.boot_artefacts_missing(3, None, STUB, stage0=True) == ""
    assert "stage-0" in s.boot_artefacts_missing(3, None, STUB, stage0=False) or "vectors" in \
        s.boot_artefacts_missing(3, None, STUB, stage0=False)
    s._boot[3]["by_hart"][2] = (regs.STAGE0_ENTRY, regs.PARK_STUB, regs.RNMI_EXC_STUB)   # one hart as read back differs
    assert "per-hart vectors" in s.boot_artefacts_missing(3, None, STUB, stage0=True)
    s._boot[3]["by_hart"][2] = (regs.STAGE0_ENTRY, regs.RESET_PC, regs.MH_SEC_RNMI_EXC)
    monkeypatch.delenv("X280_MH_APPROVED")
    assert "X280_MH_APPROVED" in s.boot_artefacts_missing(3, None, STUB, stage0=True)
    s.pll4_step_to(200)
    with pytest.raises(K.PolicyError, match="X280_MH_APPROVED"):
        s.release((3,), ledger, None, gp_code=STUB, stage0=True)
    assert dev.peek32(ARC, regs.L2CPU_RESET) == 0x0F
    s.close()


def _to_stub(s, dev, ledger) -> None:
    """E1F as x280d does it: release, done word, image, entry word, park stub swapped in ('stub')."""
    s.pll4_step_to(200)
    s.release((3,), ledger, None, gp_code=STUB, stage0=True)
    s.pll4_step_to(800)
    with s.l2cpu_batch(3):
        s.assume_lim_initialised(3, regs.LIM_BASE, regs.LIM_END, "test: stage-0 done word seen")
        s.write_image(3, IMG)
        s.stage0_enter(3)


def test_rnmi_bits_1_3_gated(dev, ledger, monkeypatch):
    """rnmi_set with bits 1-3: refused unless this session wrote the secondary vectors, the E1F state is 'stub' and the
    monitor's L36.mh has the multi-hart bit; never together with bit 0; reserved bits stay refused; 0 always works.
    A refusal writes nothing (and maps nothing)."""
    s = _stage0_session(dev, ledger, monkeypatch, 4)
    _to_stub(s, dev, ledger)

    def trig_writes():
        return [o for o in s.oplog if o.kind == "W32" and o.addr == regs.RNMI_TRIGGER]
    with s.l2cpu_batch(3):
        with pytest.raises(K.PolicyError, match="'stub'"):
            s.rnmi_set(3, 0b1110)                                   # the monitor runs, stage 0's stub not yet swapped
        s.write_gp_scratch(3, STUB)                                   # E1F: entered -> stub
        with pytest.raises(K.PolicyError, match="multi-hart support"):
            s.rnmi_set(3, 0b1110)                                   # L36.mh still 0
        dev.poke32(T3, regs.MB_IDENT_MH, regs.MH_CAP_BIT | 1)           # what the monitor writes at entry
        with pytest.raises(K.PolicyError, match="bit 0"):
            s.rnmi_set(3, 0b0011)
        with pytest.raises(K.PolicyError, match="reserved"):
            s.rnmi_set(3, 0b10010)
        assert not trig_writes()
        assert s.rnmi_set(3, 0b1110) == 0b1110 and s.rnmi_set(3, 0) == 0
        assert s.rnmi_set(3, 0b0001) == 1 and s.rnmi_set(3, 0) == 0     # PARK unchanged
        assert [o.value for o in trig_writes()] == [0b1110, 0, 1, 0]
        for h in (1, 2, 3):
            assert s.read_beu(3, h)["accrued"] == 0                    # harts 1-3's BEU pages, after release
    s.close()


def test_rnmi_bits_1_3_refused_with_the_default_vectors(dev, ledger, monkeypatch, live):
    """A standard boot (method A here, the stage-0 one below) never RNMIs harts 1-3: their handler is the GP-scratch
    park stub, which after E1F still sits as stage 0 in their L1I and strands them."""
    with live.l2cpu_batch(3):
        dev.poke32(T3, regs.MB_IDENT_MH, regs.MH_CAP_BIT | 1)
        for m in (0b0010, 0b0100, 0b1000, 0b1110):
            with pytest.raises(K.PolicyError, match="secondary RNMI vectors"):
                live.rnmi_set(3, m)
    dev2 = K.FakeDevice.default_chip()
    s = _stage0_session(dev2, L.Ledger(ledger.path.parent / "l2.json"), monkeypatch, 1)
    led2 = L.Ledger(ledger.path.parent / "l2.json")
    led2.start_hold("h2", 1)
    _to_stub(s, dev2, led2)
    with s.l2cpu_batch(3):
        s.write_gp_scratch(3, STUB)
        dev2.poke32(T3, regs.MB_IDENT_MH, regs.MH_CAP_BIT | 1)
        with pytest.raises(K.PolicyError, match="secondary RNMI vectors"):
            s.rnmi_set(3, 0b1110)
    s.close()


def test_beu_of_harts_1_3_only_after_release(k, live):
    with k.l2cpu_batch(3):
        for h in (1, 2, 3):
            with pytest.raises(K.InterlockError, match="BEU read"):
                k.read_beu(3, h)
    with live.l2cpu_batch(3):
        assert [live.read_beu(3, h)["accrued"] for h in range(4)] == [0, 0, 0, 0]
