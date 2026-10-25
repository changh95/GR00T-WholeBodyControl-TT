"""The monitor's zlib CRC-32 (x280/fw/monitor.c mon_crc32: slicing-by-8 over .rodata tables, R33 follow-up).

Every CRC the hart computes (L0 body, obs, BENCH act_crc32, L37 descriptor, LOAD data) goes through mon_crc32 or
its 8 B step, so it must stay bit-identical to zlib for every input. Checked here:
  - x280/fw/crc32_tab.h is exactly what x280/tools/gen_crc32_tab.py generates, and every entry, in the header, in
    the x86 build and in the linked monitor.elf (.rodata), equals its zlib definition;
  - the x86 build (tables) and the -DMON_CRC_BITWISE x86 build (build/x86-crcbit) equal zlib.crc32 on random
    buffers of every length 0..4096 from every start alignment 0..7, chained from non-zero CRCs, and on patterns;
  - the mailbox users in both builds: an INFER with the host's zlib obs_crc32 runs, one off by one or a wrong body
    CRC faults with MB_ERR_CRC (mtval 2 / 1), LOAD's data and descriptor CRCs, BENCH act_crc32 = zlib of L35.
The rv64 build is checked on spike (test_spike.py: `SH crc` lines, the bitwise vs table build).

    simpy -m pytest $RT/tests/x280/test_crc32.py -q        (a few seconds; builds `make x86` and `make x280`)
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

RT = Path(__file__).resolve().parents[2]
FW = RT / "x280" / "fw"
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
PIN = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]
LIBS = {"table": FW / "build" / "x86" / "libmonitor_x86.so",
        "bitwise": FW / "build" / "x86-crcbit" / "libmonitor_x86.so"}


def _load(name: str, path: Path):
    old = sys.modules.get(name)
    if old is not None and Path(getattr(old, "__file__", "")).resolve() == path.resolve():
        return old
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load("gen_crc32_tab", RT / "x280" / "tools" / "gen_crc32_tab.py")
ci = _load("check_image", RT / "scripts" / "check_image.py")
mb = _load("x280_mailbox_x86_helpers", Path(__file__).resolve().parent / "test_mailbox_x86.py")
ml = mb.ml


def _ref_tables() -> list[list[int]]:
    return [[zlib.crc32(bytes([i]) + bytes(k), 0xFFFFFFFF) ^ 0xFFFFFFFF for i in range(256)] for k in range(8)]


@pytest.fixture(scope="module")
def libs():
    r = subprocess.run(PIN + ["make", "-C", str(FW), "x86"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = {}
    for name, path in LIBS.items():
        lib = mb._load(path)
        lib.mon_x86_crc_slices.restype, lib.mon_x86_crc_slices.argtypes = ctypes.c_uint32, []
        lib.mon_x86_crc32_tab.restype, lib.mon_x86_crc32_tab.argtypes = ctypes.c_void_p, []
        out[name] = lib
    return out


@pytest.fixture(scope="module")
def image():
    r = subprocess.run(PIN + ["make", "-C", str(FW), "x280"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return ci.Elf(FW / "build" / "monitor.elf")


# ---- the tables ---------------------------------------------------------------------------------------------

def test_header_is_the_generator_output():
    t = gen.tables()
    gen.check_against_zlib(t)
    assert (FW / "crc32_tab.h").read_text() == gen.render(t), "regenerate: python3 x280/tools/gen_crc32_tab.py"
    assert t == _ref_tables()
    assert gen.main(["--check"]) == 0


def test_x86_build_tables_equal_zlib(libs):
    assert libs["table"].mon_x86_crc_slices() == 8 and libs["bitwise"].mon_x86_crc_slices() == 0
    assert libs["bitwise"].mon_x86_crc32_tab() is None
    p = libs["table"].mon_x86_crc32_tab()
    got = np.ctypeslib.as_array((ctypes.c_uint32 * 2048).from_address(p)).reshape(8, 256)
    assert got.tolist() == _ref_tables()


def test_linked_image_table_equals_zlib(image):
    """What the hart reads: mon_crc32_tab in monitor.elf's .rodata, 64 B aligned, 8 KiB, every entry zlib's;
    the loadable image stays <= 48 KiB (check_image.py checks the same as T-3.2 'crc32_tab')."""
    sym = next(s for s in image.symbols if s["name"] == "mon_crc32_tab" and s["shndx"] != 0)
    sec = image.sections[sym["shndx"]]
    assert sec["name"] == ".rodata" and sym["value"] % 64 == 0 and sym["size"] == 8192, (sec["name"], sym)
    raw = image.data[sec["off"] + sym["value"] - sec["addr"]:][:8192]
    assert list(struct.unpack("<2048I", raw)) == [v for row in _ref_tables() for v in row]
    lo, hi = image.load_span()
    assert lo == ml.LIM_BASE and hi - lo <= 48 * 1024
    assert (FW / "build" / "monitor.bin").stat().st_size == hi - lo


# ---- mon_crc32 on x86, both builds --------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ["table", "bitwise"])
def test_crc32_every_length_and_alignment_is_zlib(libs, impl):
    lib = libs[impl]
    rng = np.random.default_rng(20261025)
    data = rng.integers(0, 256, 4096 + 8, dtype=np.uint8).tobytes()
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    base = ctypes.addressof(buf)
    assert base % 8 == 0
    bad = [(off, n) for off in range(8) for n in range(4097)
           if lib.mon_x86_crc32(0, base + off, n) != zlib.crc32(data[off:off + n])]
    assert not bad, (impl, len(bad), bad[:8])


@pytest.mark.parametrize("impl", ["table", "bitwise"])
def test_crc32_chained_and_patterns_is_zlib(libs, impl):
    lib = libs[impl]
    rng = np.random.default_rng(7)
    for trial in range(200):   # zlib.crc32(b, zlib.crc32(a)) == zlib.crc32(a + b), from any start value
        n = int(rng.integers(0, 3000))
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        off = int(rng.integers(0, 8))
        buf = (ctypes.c_uint8 * (n + 8))()
        ctypes.memmove(ctypes.addressof(buf) + off, data, n)
        start = int(rng.integers(0, 2**32))
        cut = int(rng.integers(0, n + 1))
        c = lib.mon_x86_crc32(start, ctypes.addressof(buf) + off, cut)
        c = lib.mon_x86_crc32(c, ctypes.addressof(buf) + off + cut, n - cut)
        assert c == zlib.crc32(data, start), (impl, trial, n, off, cut)
    for pat in (b"\x00", b"\xff", b"\x80", b"\x01", bytes(range(256))):
        data = (pat * (4100 // len(pat) + 1))[:4100]
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        for n in (0, 1, 3, 4, 7, 8, 9, 44, 60, 64, 2112, 4096, 4100):
            assert lib.mon_x86_crc32(0, ctypes.addressof(buf), n) == zlib.crc32(data[:n]), (impl, pat[:2], n)
            assert lib.mon_x86_crc32(0xFFFFFFFF, ctypes.addressof(buf), n) == zlib.crc32(data[:n], 0xFFFFFFFF)


# ---- the mailbox users, both builds ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ["table", "bitwise"])
def test_mailbox_crc_users(libs, impl):
    """L0 body CRC, obs_crc32, LOAD data CRC, L37 desc_crc32 and BENCH act_crc32: the host's zlib values are
    accepted, a wrong one is refused with the same error as before, in both builds."""
    S, E, A = mb.S, mb.E, mb.A
    h = mb.Hart(libs[impl])
    h.lib.mon_x86_set_clock(0, 25, 16)        # short CRC retry window (50 us of fake mtime)
    rng = np.random.default_rng(3)
    data = rng.integers(0, 256, 4000, dtype=np.uint8).tobytes()   # not a multiple of 8 or 64
    h.load(mb.PAY + 0x4000, data, ml.LOAD_KIND["DATA"])
    l1 = h.load(mb.PAY + 0x4000, data, ml.LOAD_KIND["DATA"], expect_state=S["FAULT"], crc_override=1)
    assert l1["mcause"] == E["LOAD_CRC"] and l1["mtval"] == zlib.crc32(data)   # mtval = the hart's CRC
    l1 = h.load(mb.PAY + 0x4000, data, ml.LOAD_KIND["DATA"], expect_state=S["FAULT"], desc_crc_override=0)
    assert l1["mcause"] == E["DESC"]
    h.fake_blob()
    for i in range(20):
        obs = mb.rand_obs(rng)
        h.cmd(ml.CMD["INFER"], obs=obs, policy=i % 2, flags=ml.variant_flags(0))
        np.testing.assert_array_equal(np.float32(h.l35()["action"]), mb.fake_act(obs, i % 2, 0))
    obs = mb.rand_obs(rng)
    l1 = h.cmd(ml.CMD["INFER"], expect_state=S["FAULT"], obs=obs, obs_crc_delta=1)
    assert l1["mcause"] == E["CRC"] and l1["mtval"] == 2
    l1 = h.cmd(ml.CMD["NOP"], expect_state=S["FAULT"], crc_override=ml.crc32(bytes(44)))
    assert l1["mcause"] == E["CRC"] and l1["mtval"] == 1
    h.cmd(ml.CMD["BENCH"], obs=obs, arg_len=3)
    b = h.l39()
    assert b["kind"] == ml.L39_KIND["BENCH"] and b["iters"] == 3
    assert b["act_crc32"] == zlib.crc32(h.read(A["L35"], 60))
    l36 = h.l36()
    assert (l36["n_runs"], l36["n_crc_fail"], l36["n_crc_retry"]) == (20, 2, 2)
