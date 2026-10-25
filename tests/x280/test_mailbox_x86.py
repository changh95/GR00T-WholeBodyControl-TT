"""x86 build of the X280 monitor against a fake LIM buffer (PLAN.md WP-3.2, mailbox layout v2 §1.4.4).

The monitor sources (monitor.c, trap.c) are compiled unchanged with -DMON_X86 into
x280/fw/build/x86/libmonitor_x86.so; plat_x86.c fakes the CSRs, clock, BEU, RNMI and payloads. This
test plays the host: it writes host lines exactly as §1.4.4 orders them (obs lines, L0 bytes 0-59, then
seq_req as its own 4 B write) and checks the hart side: protocol, CRC retry and fault path, seq
ordering, every command, trap recovery, warm re-entry, single-writer discipline.

    simpy -m pytest $RT/tests/x280/test_mailbox_x86.py -q
"""
from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

RT = Path(__file__).resolve().parents[2]
FW = RT / "x280" / "fw"
sys.path.insert(0, str(FW))
import mailbox_layout as ml  # noqa: E402

LIB = FW / "build" / "x86" / "libmonitor_x86.so"
LIB_STUB = FW / "build" / "x86-stub" / "libmonitor_x86.so"
# Off the real-time cores 2,3,4,6,7,12-15 and their SMT siblings (task rule; override with X280_CPUS).
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
PIN = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]
A = ml.ADDR
S = ml.STATE
E = ml.ERR
PAY = ml.PAYLOAD_BASE

KIND_CHECKSUM, KIND_ILLEGAL, KIND_HANG, KIND_LIM_BENCH, KIND_CB = 1, 2, 3, 4, 5
X86_RUNNING, X86_PARKED, X86_HUNG = 0, 1, 2
PAYLOAD_CB = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)
CLOCK_HOOK = ctypes.CFUNCTYPE(None, ctypes.c_uint64)


class X86Stats(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in
                ("fence_i", "rearm", "mtime_reads", "calls", "parks", "hangs", "traps_emulated", "restarts")]


def _load(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    u32, u64, i32, vp = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p
    sig = {
        "mon_x86_attach": (i32, [vp, u64]), "mon_x86_boot": (i32, []), "mon_x86_reenter": (i32, [u64, u64]),
        "mon_x86_step": (i32, [u32]), "mon_x86_run_until": (i32, [u32, u32]),
        "mon_x86_run_thread": (i32, [vp, u64]), "mon_x86_set_clock": (None, [u64, u64, u64]),
        "mon_x86_set_clock_hook": (None, [vp]), "mon_x86_set_csrs": (None, [u64, u32, u32]),
        "mon_x86_set_beu": (None, [u32, u64, u32]), "mon_x86_stats": (None, [vp]),
        "mon_x86_mnstatus": (u64, []), "mon_x86_use_fake_kernel": (None, [i32]),
        "mon_x86_register_payload": (i32, [u64, i32, vp]), "mon_x86_clear_payloads": (None, []),
        "mon_x86_corrupt_persist": (None, []), "mon_x86_build_id": (u32, []),
        "mon_x86_kernel_is_stub": (u32, []), "mon_x86_variants_built": (u32, []),
        "mon_x86_crc32": (u32, [u32, vp, u32]), "mon_x86_layout_n": (i32, []),
        "mon_x86_layout_name": (ctypes.c_char_p, [i32]), "mon_x86_layout_value": (u64, [i32]),
        "mon_x86_mtime": (u64, []),
        "mon_x86_secondary_start": (i32, [u32, u64, u64]), "mon_x86_secondary_stop": (None, [u32]),
        "mon_x86_secondary_stop_all": (None, []), "mon_x86_secondary_running": (i32, [u32]),
        "mon_x86_secondary_mnstatus": (u64, [u32]), "mon_x86_secondary_rearms": (u32, [u32]),
        "mon_x86_secondary_hang": (None, [u32, i32]), "mon_x86_set_beu_hart": (None, [u32, u32]),
        "mon_x86_msip": (u32, [u32]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    return lib


@pytest.fixture(scope="module")
def libs():
    r = subprocess.run(PIN + ["make", "-C", str(FW), "x86"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return {"main": _load(LIB), "stub": _load(LIB_STUB)}


class Hart:
    """The fake LIM plus host-side helpers; one instance per test (cold boot)."""

    def __init__(self, lib: ctypes.CDLL, fake_kernel: bool = True, boot: bool = True):
        self.lib = lib
        self._raw = (ctypes.c_uint8 * (ml.LIM_SIZE + 64))()
        base = ctypes.addressof(self._raw)
        self._pad = (-base) % 64
        self.ptr = base + self._pad
        self.mem = np.frombuffer(self._raw, dtype=np.uint8)[self._pad:self._pad + ml.LIM_SIZE]
        rng = np.random.default_rng(1)
        self.mem[:] = rng.integers(0, 256, ml.LIM_SIZE, dtype=np.uint8)   # "initialised" but not zero
        for name in ("L0", "L1", "L35", "L36", "L37", "L38", "L39"):
            self.write(A[name], bytes(64))
        assert lib.mon_x86_attach(self.ptr, ml.LIM_SIZE) == 0
        lib.mon_x86_clear_payloads()
        lib.mon_x86_set_clock_hook(None)
        lib.mon_x86_use_fake_kernel(1 if fake_kernel else 0)
        self._keep = []
        if boot:
            assert lib.mon_x86_boot() == X86_RUNNING

    # -- raw access ---------------------------------------------------------------------------------
    def off(self, addr: int) -> int:
        assert ml.LIM_BASE <= addr < ml.LIM_END
        return addr - ml.LIM_BASE

    def write(self, addr: int, data: bytes) -> None:
        o = self.off(addr)
        self.mem[o:o + len(data)] = np.frombuffer(bytes(data), dtype=np.uint8)

    def read(self, addr: int, n: int) -> bytes:
        o = self.off(addr)
        return self.mem[o:o + n].tobytes()

    def w32(self, addr: int, v: int) -> None:
        ctypes.c_uint32.from_address(self.ptr + self.off(addr)).value = v & 0xFFFFFFFF

    def r32(self, addr: int) -> int:
        return ctypes.c_uint32.from_address(self.ptr + self.off(addr)).value

    def l1(self) -> dict:
        return ml.unpack_l1(self.read(A["L1"], 64))

    def l35(self) -> dict:
        return ml.unpack_l35(self.read(A["L35"], 64))

    def l36(self) -> dict:
        return ml.unpack_l36(self.read(A["L36"], 64))

    def l39(self) -> dict:
        return ml.unpack_l39(self.read(A["L39"], 64))

    def stats(self) -> X86Stats:
        s = X86Stats()
        self.lib.mon_x86_stats(ctypes.byref(s))
        return s

    # -- host protocol (§1.4.4 steps 1-2) -------------------------------------------------------------
    def issue(self, cmd: int, seq: int | None = None, obs=None, obs_crc_delta: int = 0, **kw) -> int:
        cur = self.r32(A["L0"] + 60)
        seq = (self.l1()["seq_done"] + 1) & 0xFFFFFFFF if seq is None else seq
        obs_crc = 0
        if obs is not None:
            ob = ml.pack_obs(np.asarray(obs, dtype=np.float32))
            self.write(A["L2"], ob)
            obs_crc = (ml.crc32(ob) + obs_crc_delta) & 0xFFFFFFFF
        l0 = ml.pack_l0(cmd, seq_req=cur, obs_crc32=obs_crc, **kw)
        self.write(A["L0"], l0[:60])     # body as one line (seq_req bytes untouched)
        self.w32(A["L0"] + 60, seq)       # then seq_req as its own 4 B write
        return seq

    def run(self, seq: int, max_iters: int = 200_000) -> int:
        return self.lib.mon_x86_run_until(seq, max_iters)

    def cmd(self, cmd: int, expect_state: int = S["IDLE"], **kw) -> dict:
        seq = self.issue(cmd, **kw)
        assert self.run(seq) == X86_RUNNING, f"cmd {cmd} did not complete"
        l1 = self.l1()
        assert l1["seq_done"] == seq
        assert l1["state"] == expect_state, l1
        return l1

    def load(self, dst: int, data: bytes, kind: int, expect_state: int = S["IDLE"], seq: int | None = None,
             desc_crc_override: int | None = None, load_seq: int | None = None, crc_override: int | None = None):
        padded = data + bytes((-len(data)) % 64)
        self.write(dst, padded)
        seq = (self.l1()["seq_done"] + 1) & 0xFFFFFFFF if seq is None else seq
        crc = ml.crc32(data) if crc_override is None else crc_override
        self.write(A["L37"], ml.pack_l37(dst, len(data), crc, kind, seq if load_seq is None else load_seq,
                                         desc_crc_override))
        return self.cmd(ml.CMD["LOAD"], expect_state=expect_state, seq=seq)

    def fake_blob(self, dst: int = PAY) -> dict:
        return self.load(dst, struct.pack("<I", 0x454B4146) + bytes(60), ml.LOAD_KIND["BLOB"])

    def register(self, addr: int, kind: int, cb=None) -> None:
        if cb is not None:
            self._keep.append(cb)
        assert self.lib.mon_x86_register_payload(addr, kind, ctypes.cast(cb, ctypes.c_void_p) if cb else None) == 0

    def code(self, dst: int, nbytes: int = 256) -> None:
        """LOAD a code region (the x86 'code' is the registered payload table) and FENCE_I it."""
        self.load(dst, bytes(range(256)) * (nbytes // 256), ml.LOAD_KIND["CODE"])
        self.cmd(ml.CMD["FENCE_I"])


def fake_act(obs: np.ndarray, policy: int, variant: int) -> np.ndarray:
    """numpy replica of plat_x86.c fake_infer, float32 throughout, same evaluation order."""
    o = np.asarray(obs, dtype=np.float32)
    k = np.float32(policy * 100 + variant * 10) + np.arange(15, dtype=np.float32)
    return (o[:15] * np.float32(0.5) + o[515]) + k


def rand_obs(rng) -> np.ndarray:
    return rng.standard_normal(516).astype(np.float32)


@pytest.fixture
def hart(libs):
    return Hart(libs["main"])


# ---- layout ------------------------------------------------------------------------------------------

def test_layout_matches_c(libs):
    lib = libs["main"]
    seen = {}
    for i in range(lib.mon_x86_layout_n()):
        seen[lib.mon_x86_layout_name(i).decode()] = lib.mon_x86_layout_value(i)
    expected = {**ml.OFFSETS, **ml.CONSTANTS}
    missing = sorted(set(seen) - set(expected))
    assert not missing, f"mailbox_layout.py lacks {missing}"
    bad = {k: (seen[k], expected[k]) for k in seen if seen[k] != expected[k]}
    assert not bad, bad


def test_crc32_is_zlib(libs):
    lib = libs["main"]
    rng = np.random.default_rng(7)
    for n in (0, 1, 3, 4, 5, 44, 60, 2112, 4097):
        buf = (ctypes.c_uint8 * (n + 3))()
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        for skew in (0, 1, 3):   # unaligned starts
            ctypes.memmove(ctypes.addressof(buf) + skew, data, n)
            assert lib.mon_x86_crc32(0, ctypes.addressof(buf) + skew, n) == zlib.crc32(data)


# ---- boot / ident ----------------------------------------------------------------------------------------

def test_cold_boot_publishes_status_and_ident(libs):
    h = Hart(libs["main"], boot=False)
    h.w32(A["L0"] + 60, 1234)            # a stale seq_req left in L0 before release
    assert h.lib.mon_x86_boot() == X86_RUNNING
    l1, l36, l39 = h.l1(), h.l36(), h.l39()
    assert l1["fw_magic"] == ml.MAGIC and l1["fw_ver"] == ml.FW_VER
    assert l1["state"] == S["IDLE"] and l1["seq_done"] == 1234   # adopted, not executed
    assert l1["vlenb"] == 64 and l1["mhartid"] == 0 and l1["mcause"] == 0
    assert l36["mtime_hz"] == 50_000_000 and l36["build_id"] == h.lib.mon_x86_build_id()
    assert l36["misa"] & (1 << 21) and l36["n_reentry"] == 0 and l36["entry_flags"] & 1 == 0
    assert l36["variants_built"] == h.lib.mon_x86_variants_built()
    assert l39["kind"] == ml.L39_KIND["ENTRY"] and l39["entry_kind"] == 0 and l39["mnstatus"] & 8
    assert h.lib.mon_x86_mnstatus() & 8 and h.stats().rearm == 1
    hb0 = l1["heartbeat"]
    h.lib.mon_x86_set_clock(h.lib.mon_x86_mtime(), 50, 16)     # the fake mtime moves 50 ticks per read
    assert h.lib.mon_x86_step(5000) == X86_RUNNING
    assert h.l1()["heartbeat"] > hb0                                # idle loop keeps the heartbeat going
    assert h.l36()["n_runs"] == 0


def test_heartbeat_rate_and_hi_lo_hi(hart):
    hart.lib.mon_x86_set_clock((1 << 32) - 5000, 7, 16)            # crosses the 32-bit boundary
    beats = []
    for _ in range(50):
        hart.lib.mon_x86_step(64)
        hi1 = hart.r32(A["L1"] + 4)
        lo = hart.r32(A["L1"])
        hi2 = hart.r32(A["L1"] + 4)
        assert hi1 == hi2
        beats.append((hi1 << 32) | lo)
    assert all(b2 >= b1 for b1, b2 in zip(beats, beats[1:])) and beats[-1] > beats[0]


# ---- protocol --------------------------------------------------------------------------------------------

def test_nop_and_seq_ordering(hart):
    for _ in range(5):
        hart.cmd(ml.CMD["NOP"])
    s = hart.l1()["seq_done"]
    # unchanged seq_req: nothing is re-executed
    hart.lib.mon_x86_step(1000)
    assert hart.l1()["seq_done"] == s
    # the host may skip values or wrap: any change is a new command
    hart.cmd(ml.CMD["NOP"], seq=s + 10)
    hart.cmd(ml.CMD["NOP"], seq=0xFFFFFFFF)
    hart.cmd(ml.CMD["NOP"], seq=0)
    # two writes before the hart polls: only the newest is executed
    hart.fake_blob()
    rng = np.random.default_rng(3)
    o1, o2 = rand_obs(rng), rand_obs(rng)
    hart.issue(ml.CMD["INFER"], seq=100, obs=o1)
    seq = hart.issue(ml.CMD["INFER"], seq=101, obs=o2)
    assert hart.run(seq) == X86_RUNNING
    assert hart.l35()["seq_echo"] == 101
    np.testing.assert_array_equal(np.float32(hart.l35()["action"]), fake_act(o2, 0, 0))
    assert hart.l36()["n_runs"] == 1


def test_infer_fake_kernel_bit_exact(hart):
    hart.fake_blob()
    rng = np.random.default_rng(11)
    for i in range(40):
        obs = rand_obs(rng)
        policy, variant = i % 2, (i // 2) % 4
        flags = ml.variant_flags(variant) | ml.FLAG_HOST_ALIVE
        l1 = hart.cmd(ml.CMD["INFER"], obs=obs, policy=policy, flags=flags, tick=1000 + i)
        a = hart.l35()
        assert a["seq_echo"] == l1["seq_done"]
        np.testing.assert_array_equal(np.float32(a["action"]), fake_act(obs, policy, variant))
        assert l1["cycles_last"] > 0 and l1["cycles_max"] >= l1["cycles_last"]
        assert hart.l36()["kernel_variant"] == variant
    assert hart.l36()["n_runs"] == 40
    # default variant when VARIANT_SET is clear
    obs = rand_obs(rng)
    hart.cmd(ml.CMD["INFER"], obs=obs, policy=1)
    np.testing.assert_array_equal(np.float32(hart.l35()["action"]), fake_act(obs, 1, 0))


def test_default_variant_follows_the_blob_weight_type(hart):
    """Without the variant bits an fp16 blob gets V2 (V2S if V2 is not built), not V0's WBC_EDTYPE (kernel lane
    open issue); an fp32 blob keeps the V0 default. wdtype is the blob header's u32 at offset 20 (wbc_mlp.h)."""
    rng = np.random.default_rng(21)
    hart.load(PAY, struct.pack("<I", 0x454B4146) + bytes(16) + struct.pack("<I", 1) + bytes(40), ml.LOAD_KIND["BLOB"])
    built = hart.lib.mon_x86_variants_built()
    want = 2 if built >> 2 & 1 else 3
    obs = rand_obs(rng)
    hart.cmd(ml.CMD["INFER"], obs=obs, policy=0)
    np.testing.assert_array_equal(np.float32(hart.l35()["action"]), fake_act(obs, 0, want))
    assert hart.l36()["kernel_variant"] == want
    hart.fake_blob()  # wdtype 0 (fp32)
    obs = rand_obs(rng)
    hart.cmd(ml.CMD["INFER"], obs=obs, policy=1)
    np.testing.assert_array_equal(np.float32(hart.l35()["action"]), fake_act(obs, 1, 0))


def test_infer_errors(hart):
    obs = np.zeros(516, np.float32)
    l1 = hart.cmd(ml.CMD["INFER"], expect_state=S["FAULT"], obs=obs)            # no blob yet
    assert l1["mcause"] == E["BLOB"]
    hart.fake_blob()
    l1 = hart.cmd(ml.CMD["INFER"], expect_state=S["FAULT"], obs=obs, policy=2)
    assert l1["mcause"] == E["ARG"] and l1["mtval"] == ml.ARG["POLICY"]
    seq_echo = hart.l35()["seq_echo"]
    l1 = hart.cmd(ml.CMD["INFER"], expect_state=S["FAULT"], obs=obs, flags=ml.variant_flags(15))
    assert l1["mcause"] == E["KERNEL"] and l1["mtval"] == (-8 & (2**64 - 1)) and l1["mepc"] == 15
    assert hart.l35()["seq_echo"] == seq_echo                                    # L35 untouched on error
    hart.cmd(ml.CMD["INFER"], obs=obs)                                           # and it recovers
    l1 = hart.cmd(9, expect_state=S["FAULT"])
    assert l1["mcause"] == E["CMD"] and l1["mtval"] == 9
    l1 = hart.cmd(ml.CMD["NOP"], expect_state=S["FAULT"], magic=0x12345678)
    assert l1["mcause"] == E["MAGIC"] and l1["mtval"] == 0x12345678
    hart.cmd(ml.CMD["NOP"])
    assert hart.l1()["mcause"] == 0


def test_crc_mismatch_is_retried_then_faults(hart):
    hart.fake_blob()
    hart.lib.mon_x86_set_clock(0, 25, 16)        # 2,500 ticks = 50 us at 50 MHz -> ~100 re-reads
    obs = rand_obs(np.random.default_rng(5))
    t0 = hart.lib.mon_x86_mtime()
    l1 = hart.cmd(ml.CMD["INFER"], expect_state=S["FAULT"], obs=obs, obs_crc_delta=1)
    assert l1["mcause"] == E["CRC"] and l1["mtval"] == 2                         # b1: obs CRC
    assert hart.lib.mon_x86_mtime() - t0 >= 2500                                 # retried for >= 50 us
    l1 = hart.cmd(ml.CMD["NOP"], expect_state=S["FAULT"], crc_override=0xDEADBEEF)
    assert l1["mcause"] == E["CRC"] and l1["mtval"] == 1                         # b0: body CRC
    l36 = hart.l36()
    assert l36["n_crc_fail"] == 2 and l36["n_crc_retry"] == 2
    hart.cmd(ml.CMD["INFER"], obs=obs)                                           # a good one passes
    np.testing.assert_array_equal(np.float32(hart.l35()["action"]), fake_act(obs, 0, 0))


def test_crc_mismatch_fixed_within_retry_window(hart):
    """The host's obs lines land after seq_req (WC/UC reordering): the re-read inside 50 us succeeds."""
    hart.fake_blob()
    obs = rand_obs(np.random.default_rng(6))
    good = ml.pack_obs(obs)
    calls = {"n": 0}

    def hook(n):
        calls["n"] += 1
        if calls["n"] == 40:                     # some re-reads later, the right bytes arrive
            hart.write(A["L2"], good)
    cb = CLOCK_HOOK(hook)
    seq = hart.issue(ml.CMD["INFER"], obs=obs)
    hart.write(A["L2"] + 8, b"\xff\xff\xff\xff")       # torn obs: CRC in L0 is for `good`
    hart.lib.mon_x86_set_clock_hook(cb)
    assert hart.run(seq) == X86_RUNNING
    hart.lib.mon_x86_set_clock_hook(None)
    assert hart.l1()["state"] == S["IDLE"]
    np.testing.assert_array_equal(np.float32(hart.l35()["action"]), fake_act(obs, 0, 0))
    assert hart.l36()["n_crc_retry"] == 1 and hart.l36()["n_crc_fail"] == 0


def test_crc_retry_abandoned_when_seq_moves_on(hart):
    hart.fake_blob()
    obs = rand_obs(np.random.default_rng(8))
    state = {"n": 0}

    def hook(n):
        state["n"] += 1
        if state["n"] == 30:                     # the host times out and issues the next command
            hart.issue(ml.CMD["NOP"], seq=77)
    cb = CLOCK_HOOK(hook)
    hart.issue(ml.CMD["INFER"], seq=76, obs=obs, obs_crc_delta=5)
    hart.lib.mon_x86_set_clock_hook(cb)
    assert hart.run(77) == X86_RUNNING
    hart.lib.mon_x86_set_clock_hook(None)
    l1 = hart.l1()
    assert l1["seq_done"] == 77 and l1["state"] == S["IDLE"]
    assert hart.l36()["n_crc_fail"] == 0          # 76 was abandoned, not failed


def test_overrun_counted_and_next_command_runs(hart):
    hart.fake_blob()
    obs = rand_obs(np.random.default_rng(9))
    st = {"n": 0}

    def hook(n):
        st["n"] += 1
        if st["n"] == 20:
            hart.issue(ml.CMD["NOP"], seq=501)    # host moved on while BENCH runs
    cb = CLOCK_HOOK(hook)
    hart.issue(ml.CMD["BENCH"], seq=500, obs=obs, arg_len=50)
    hart.lib.mon_x86_set_clock_hook(cb)
    assert hart.run(501) == X86_RUNNING
    hart.lib.mon_x86_set_clock_hook(None)
    assert hart.l1()["seq_done"] == 501
    assert hart.l36()["n_overrun"] == 1
    assert hart.l39()["seq"] == 500 and hart.l39()["iters"] == 50


# ---- LOAD / FENCE_I / CALL ---------------------------------------------------------------------------

def test_load_checks(hart):
    data = bytes(range(256)) * 8
    hart.load(PAY, data, ml.LOAD_KIND["DATA"])
    for dst, n, code in ((PAY + 4, 64, ml.ARG["ALIGN"]), (ml.MB_BASE, 64, ml.ARG["RANGE"]),
                         (ml.LIM_BASE, 64, ml.ARG["RANGE"]), (ml.LIM_END - 1024, 1025, ml.ARG["RANGE"]),
                         (PAY, 0, ml.ARG["LEN"])):
        seq = hart.l1()["seq_done"] + 1
        hart.write(A["L37"], ml.pack_l37(dst, n, 0, ml.LOAD_KIND["DATA"], seq))
        l1 = hart.cmd(ml.CMD["LOAD"], expect_state=S["FAULT"], seq=seq)
        assert l1["mcause"] == E["ARG"] and l1["mtval"] == code, (hex(dst), l1)
    l1 = hart.load(PAY, data, ml.LOAD_KIND["DATA"], expect_state=S["FAULT"], crc_override=1)
    assert l1["mcause"] == E["LOAD_CRC"] and l1["mepc"] == PAY and l1["mtval"] == zlib.crc32(data)
    l1 = hart.load(PAY, data, ml.LOAD_KIND["DATA"], expect_state=S["FAULT"], desc_crc_override=0)
    assert l1["mcause"] == E["DESC"]
    l1 = hart.load(PAY, data, ml.LOAD_KIND["DATA"], expect_state=S["FAULT"], load_seq=1)   # stale descriptor
    assert l1["mcause"] == E["DESC"]
    l1 = hart.load(PAY, data, 7, expect_state=S["FAULT"])
    assert l1["mcause"] == E["ARG"] and l1["mtval"] == ml.ARG["KIND"]
    l1 = hart.load(PAY, b"not a blob" + bytes(54), ml.LOAD_KIND["BLOB"], expect_state=S["FAULT"])
    assert l1["mcause"] == E["BLOB"] and l1["mtval"] == (-2 & (2**64 - 1))
    hart.load(ml.LIM_END - 128, bytes(range(128)), ml.LOAD_KIND["DATA"])                 # last two lines ok


def test_load_overwriting_blob_unregisters_it(hart):
    hart.fake_blob(PAY + 0x1000)
    obs = np.ones(516, np.float32)
    hart.cmd(ml.CMD["INFER"], obs=obs)
    hart.load(PAY + 0x1000 - 64, bytes(128), ml.LOAD_KIND["DATA"])     # overlaps the blob
    l1 = hart.cmd(ml.CMD["INFER"], expect_state=S["FAULT"], obs=obs)
    assert l1["mcause"] == E["BLOB"]


def test_call_checksum_10_of_10(hart):
    entry = PAY
    hart.code(entry)
    hart.register(entry, KIND_CHECKSUM)
    region = ml.LIM_BASE + 0x8000
    for i in range(10):
        n = 1000 + 97 * i
        expect = zlib.crc32(hart.read(region, n))
        hart.cmd(ml.CMD["CALL"], arg_entry=entry, arg_addr=region, arg_len=n, policy=i)
        r = hart.l39()
        assert r["kind"] == ml.L39_KIND["CALL"] and r["ret"] == expect and r["seq"] == hart.l1()["seq_done"]
        assert struct.unpack("<II", r["data"][:8]) == (expect, n)
    assert hart.stats().calls == 10


def test_call_requires_loaded_fenced_code(hart):
    entry = PAY + 0x100
    hart.register(entry, KIND_CHECKSUM)
    l1 = hart.cmd(ml.CMD["CALL"], expect_state=S["FAULT"], arg_entry=entry)
    assert l1["mcause"] == E["NOCODE"] and l1["mepc"] == entry and l1["mtval"] == 0
    hart.load(PAY, bytes(1024), ml.LOAD_KIND["CODE"])
    l1 = hart.cmd(ml.CMD["CALL"], expect_state=S["FAULT"], arg_entry=entry)
    assert l1["mcause"] == E["NOCODE"] and l1["mtval"] == 1                      # loaded, not fenced
    f0 = hart.stats().fence_i
    hart.cmd(ml.CMD["FENCE_I"])
    assert hart.stats().fence_i == f0 + 1
    hart.cmd(ml.CMD["CALL"], arg_entry=entry, arg_addr=ml.LIM_BASE, arg_len=64)
    for bad in (PAY + 1024, PAY - 2, entry + 1):
        l1 = hart.cmd(ml.CMD["CALL"], expect_state=S["FAULT"], arg_entry=bad)
        assert l1["mcause"] == E["NOCODE"], hex(bad)
    hart.load(PAY + 512, bytes(64), ml.LOAD_KIND["DATA"])                       # rewrites part of the code
    l1 = hart.cmd(ml.CMD["CALL"], expect_state=S["FAULT"], arg_entry=entry)
    assert l1["mcause"] == E["NOCODE"]


def test_call_python_payload_args_and_result(hart):
    seen = {}

    def payload(a_ptr, api_ptr):
        a = struct.unpack("<IIqQQII", ctypes.string_at(a_ptr, 40))
        seen.update(policy=a[0], flags=a[1], tick=a[2], arg_addr=a[3], arg_entry=a[4], arg_len=a[5], seq=a[6])
        api = struct.unpack("<IIQ", ctypes.string_at(api_ptr, 16))
        seen["api_version"], seen["api_size"] = api[0], api[1]
        result = ctypes.c_void_p.from_address(api_ptr + 48).value
        ctypes.memmove(result, b"R" * 40, 40)
        return 0x1122334455667788
    cb = PAYLOAD_CB(payload)
    hart.code(PAY)
    hart.register(PAY + 64, KIND_CB, cb)
    hart.cmd(ml.CMD["CALL"], arg_entry=PAY + 64, arg_addr=0xABC, arg_len=7, policy=3, flags=5, tick=-9)
    assert seen == dict(policy=3, flags=5, tick=-9, arg_addr=0xABC, arg_entry=PAY + 64, arg_len=7,
                        seq=hart.l1()["seq_done"], api_version=2, api_size=96)   # mon_api v2: + the mh services
    r = hart.l39()
    assert r["ret"] == 0x1122334455667788 and r["data"] == b"R" * 40


def test_trap_in_payload_is_dumped_and_monitor_survives(hart):
    hart.code(PAY)
    hart.register(PAY, KIND_ILLEGAL)
    for _ in range(3):
        l1 = hart.cmd(ml.CMD["CALL"], expect_state=S["FAULT"], arg_entry=PAY)
        assert (l1["mcause"], l1["mepc"], l1["mtval"]) == (2, PAY, 0xC0001073)
        hart.cmd(ml.CMD["NOP"])
    assert hart.l36()["n_traps"] == 3
    # an unregistered entry inside the code region: instruction access fault at the entry
    l1 = hart.cmd(ml.CMD["CALL"], expect_state=S["FAULT"], arg_entry=PAY + 128)
    assert (l1["mcause"], l1["mepc"], l1["mtval"]) == (1, PAY + 128, PAY + 128)


def test_hang_park_reentry_rearm_10_of_10(hart):
    """T-3.E2's RNMI re-arm sequence, host-side: hung payload -> RNMI park -> warm entry -> CALL again."""
    hart.code(PAY)
    hart.register(PAY, KIND_HANG)
    hart.fake_blob(PAY + 0x1000)
    for i in range(1, 11):
        seq = hart.issue(ml.CMD["CALL"], arg_entry=PAY)
        assert hart.run(seq) == X86_HUNG
        assert hart.l1()["state"] == S["BUSY"]
        hb = hart.l1()["heartbeat"]
        assert hart.lib.mon_x86_step(100) == X86_HUNG and hart.l1()["heartbeat"] == hb   # heartbeat silence
        rearm0 = hart.stats().rearm
        mncause, mnepc = (1 << 63) | 2, PAY
        assert hart.lib.mon_x86_reenter(mncause, mnepc) == X86_RUNNING
        assert hart.stats().rearm == rearm0 + 1 and hart.lib.mon_x86_mnstatus() & 8
        l1, l36, l39 = hart.l1(), hart.l36(), hart.l39()
        assert l1["state"] == S["REENTERED"] and l1["seq_done"] == seq        # the hung CALL is not re-run
        assert (l1["mcause"], l1["mepc"]) == (mncause, mnepc)
        assert l36["n_reentry"] == i and l36["entry_flags"] & ml.ENTRY_FLAG["WARM"]
        assert not l36["entry_flags"] & ml.ENTRY_FLAG["RELOADED"]
        assert l39["kind"] == ml.L39_KIND["ENTRY"] and l39["entry_kind"] == 1 and l39["mnepc"] == PAY
    # registrations survive warm entries: INFER and CALL work without a new LOAD
    hart.cmd(ml.CMD["INFER"], obs=np.ones(516, np.float32))
    hart.register(PAY, KIND_CHECKSUM)
    hart.cmd(ml.CMD["CALL"], arg_entry=PAY, arg_addr=ml.LIM_BASE, arg_len=64)


def test_park_command(hart):
    seq = hart.issue(ml.CMD["PARK"])
    assert hart.run(seq) == X86_PARKED
    l1 = hart.l1()
    assert l1["state"] == S["PARKED"] and l1["seq_done"] == seq
    hb = l1["heartbeat"]
    assert hart.lib.mon_x86_step(1000) == X86_PARKED and hart.l1()["heartbeat"] == hb
    assert hart.lib.mon_x86_reenter((1 << 63) | 2, 0x08000123) == X86_RUNNING
    assert hart.l1()["state"] == S["REENTERED"] and hart.l36()["n_reentry"] == 1
    hart.cmd(ml.CMD["NOP"])


def test_reload_detected_on_warm_entry(hart):
    seq = hart.issue(ml.CMD["PARK"])
    assert hart.run(seq) == X86_PARKED
    hart.lib.mon_x86_corrupt_persist()            # a different image was written while parked
    assert hart.lib.mon_x86_reenter((1 << 63) | 2, 0) == X86_RUNNING
    l36 = hart.l36()
    assert l36["entry_flags"] & ml.ENTRY_FLAG["RELOADED"] and l36["n_reentry"] == 1
    assert hart.l39()["entry_kind"] == 2


# ---- SELFTEST / BENCH -----------------------------------------------------------------------------------

def test_selftest_reads_every_line(hart):
    hart.lib.mon_x86_set_beu(0, 0, 0)
    l1 = hart.cmd(ml.CMD["SELFTEST"])
    r = hart.l39()
    assert r["kind"] == ml.L39_KIND["SELFTEST"] and r["lines"] == ml.LIM_SIZE // 64
    assert r["seq"] == l1["seq_done"]
    assert r["beu_accrued_after"] == 0 and hart.l36()["beu_accrued"] == 0
    assert r["misa_v"] == 1 and r["vlenb"] == 64
    assert r["mtime_ticks"] > 0 and r["cyc_per_mtime_x1000"] == r["cycles"] * 1000 // r["mtime_ticks"]
    assert l1["heartbeat"] > 0
    # a sub-range: exact xor of every word; the BEU values are reported
    hart.lib.mon_x86_set_beu(0x5, 0x0801_2340, 0x40)
    hart.cmd(ml.CMD["SELFTEST"], arg_addr=PAY, arg_len=4096)
    r = hart.l39()
    seg = hart.mem[PAY - ml.LIM_BASE:PAY - ml.LIM_BASE + 4096].view(np.uint64)
    assert r["lines"] == 64 and r["ret"] == int(np.bitwise_xor.reduce(seg))
    assert (r["beu_cause"], r["beu_value"], r["beu_accrued_before"], r["beu_accrued_after"]) == \
        (5, 0x08012340, 0x40, 0x40)
    assert hart.l36()["beu_accrued"] == 0x40
    for addr, n in ((PAY + 8, 64), (ml.LIM_END - 64, 128)):
        l1 = hart.cmd(ml.CMD["SELFTEST"], expect_state=S["FAULT"], arg_addr=addr, arg_len=n)
        assert l1["mcause"] == E["ARG"]


def test_selftest_xor_excluding_hart_lines(hart):
    """Exact check of the scan: everything except the lines the hart itself writes (L1, L39)."""
    lo, hi = A["L1"], A["L39"]
    hart.cmd(ml.CMD["SELFTEST"], arg_addr=ml.LIM_BASE, arg_len=lo - ml.LIM_BASE)
    words = hart.mem[:lo - ml.LIM_BASE].view(np.uint64)
    assert hart.l39()["ret"] == int(np.bitwise_xor.reduce(words))
    start = hi + 64
    hart.cmd(ml.CMD["SELFTEST"], arg_addr=start, arg_len=ml.LIM_END - start)
    words = hart.mem[start - ml.LIM_BASE:].view(np.uint64)
    assert hart.l39()["ret"] == int(np.bitwise_xor.reduce(words))


def test_bench(hart):
    hart.fake_blob()
    obs = rand_obs(np.random.default_rng(12))
    out = PAY + 0x2000
    hart.cmd(ml.CMD["BENCH"], obs=obs, policy=1, flags=ml.variant_flags(2), arg_len=50, arg_addr=out)
    r = hart.l39()
    exp = fake_act(obs, 1, 2)
    assert r["kind"] == ml.L39_KIND["BENCH"] and r["iters"] == 50 and r["ret"] == 50 and r["nondet"] == 0
    assert r["rc"] == 0 and r["variant"] == 2 and r["policy"] == 1
    cyc = np.frombuffer(hart.read(out, 200), dtype=np.uint32)
    assert (cyc > 0).all() and r["cyc_min"] == cyc.min() and r["cyc_max"] == cyc.max()
    assert r["cyc_sum"] == int(cyc.sum()) == r["cycles"]
    assert r["act_crc32"] == zlib.crc32(exp.astype("<f4").tobytes())
    np.testing.assert_array_equal(np.float32(hart.l35()["action"]), exp)
    l1 = hart.cmd(ml.CMD["BENCH"], expect_state=S["FAULT"], obs=obs, arg_len=4, arg_addr=PAY)   # over the blob
    assert l1["mcause"] == E["ARG"] and l1["mtval"] == ml.ARG["OVERLAP"]
    l1 = hart.cmd(ml.CMD["BENCH"], expect_state=S["FAULT"], obs=obs, arg_len=4, flags=ml.variant_flags(15))
    assert l1["mcause"] == E["KERNEL"] and hart.l39()["iters"] == 0


# ---- host gone -------------------------------------------------------------------------------------------

def test_host_gone_stops_commands_until_cleared(hart):
    hart.fake_blob()
    hart.w32(A["L38"], ml.HOST_GONE)                           # KMD cleanup write after x280d died
    hart.lib.mon_x86_step(20000)
    assert hart.l1()["state"] == S["HOST_GONE"] and hart.l36()["n_host_gone"] == 1
    runs = hart.l36()["n_runs"]
    seq = hart.issue(ml.CMD["INFER"], obs=np.ones(516, np.float32))
    assert hart.run(seq) == X86_RUNNING
    assert hart.l1()["state"] == S["HOST_GONE"] and hart.l36()["n_runs"] == runs   # adopted, not run
    hart.w32(A["L38"], 0)                                      # new x280d attaches and clears L38
    hart.lib.mon_x86_step(20000)
    assert hart.l1()["state"] == S["IDLE"]
    hart.cmd(ml.CMD["INFER"], obs=np.ones(516, np.float32))
    assert hart.l36()["n_runs"] == runs + 1 and hart.l36()["n_host_gone"] == 1


# ---- single writer -------------------------------------------------------------------------------------

def test_hart_never_writes_host_lines(hart):
    """After a run of every command, host-owned lines hold exactly what the host wrote."""
    hart.fake_blob()
    hart.code(PAY + 0x1000)
    hart.register(PAY + 0x1000, KIND_CHECKSUM)
    rng = np.random.default_rng(13)
    host_lines = [("L37", 64), ("L38", 64), ("L40", 64 * 24)]
    marks = {}
    for name, n in host_lines[1:]:
        pat = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        if name == "L38":
            pat = b"\0\0\0\0" + pat[4:]   # host_gone clear, the rest of the line arbitrary
        hart.write(A[name], pat)
    for cmd, kw in ((ml.CMD["NOP"], {}), (ml.CMD["INFER"], {"obs": rand_obs(rng)}),
                    (ml.CMD["BENCH"], {"obs": rand_obs(rng), "arg_len": 3}), (ml.CMD["SELFTEST"], {}),
                    (ml.CMD["FENCE_I"], {}),
                    (ml.CMD["CALL"], {"arg_entry": PAY + 0x1000, "arg_addr": ml.LIM_BASE, "arg_len": 64})):
        for name, n in host_lines:
            marks[name] = hart.read(A[name], n)
        seq = hart.issue(cmd, **kw)
        l0_after_host = hart.read(A["L0"], 64)
        obs_after_host = hart.read(A["L2"], 33 * 64)
        assert hart.run(seq) == X86_RUNNING and hart.l1()["state"] == S["IDLE"]
        assert hart.read(A["L0"], 64) == l0_after_host, cmd
        assert hart.read(A["L2"], 33 * 64) == obs_after_host, cmd
        for name, n in host_lines:
            assert hart.read(A[name], n) == marks[name], (cmd, name)


def test_hart_writes_only_its_lines_and_bss(hart):
    """Snapshot all of LIM, run commands, diff: only L1, L35, L36, L39 and the BENCH array change."""
    hart.fake_blob()
    before = hart.mem.copy()
    rng = np.random.default_rng(14)
    out = PAY + 0x3000
    obs = rand_obs(rng)
    # host writes, tracked so they can be excluded from the diff
    host = np.zeros(ml.LIM_SIZE, dtype=bool)

    def mark(addr, n):
        host[addr - ml.LIM_BASE:addr - ml.LIM_BASE + n] = True
    for cmd, kw in ((ml.CMD["INFER"], {"obs": obs}), (ml.CMD["BENCH"], {"obs": obs, "arg_len": 8, "arg_addr": out}),
                    (ml.CMD["SELFTEST"], {}), (ml.CMD["NOP"], {})):
        mark(A["L0"], 64)
        mark(A["L2"], 33 * 64)
        hart.cmd(cmd, **kw)
    changed = (hart.mem != before) & ~host
    lines = sorted(set((np.nonzero(changed)[0] // 64).tolist()))
    allowed = {(A[n] - ml.LIM_BASE) // 64 for n in ("L1", "L35", "L36", "L39")}
    allowed |= set(range((out - ml.LIM_BASE) // 64, (out - ml.LIM_BASE + 32 + 63) // 64))
    assert set(lines) <= allowed, [hex(ml.LIM_BASE + 64 * x) for x in sorted(set(lines) - allowed)]


# ---- lim_bench on x86 ------------------------------------------------------------------------------------

def test_lim_bench_payload_scalar_tests(hart):
    hart.code(PAY)
    hart.register(PAY, KIND_LIM_BENCH)
    ws = PAY + 0x10000
    for test, kw in ((1, {"arg_addr": ws, "arg_len": 64 * 1024, "tick": 5000}),
                     (2, {"arg_addr": ws, "arg_len": 32 * 1024, "tick": 2}),
                     (6, {"tick": 100}), (8, {})):
        hart.cmd(ml.CMD["CALL"], arg_entry=PAY, policy=test, **kw)
        r = ml.unpack_lb_result(hart.l39()["data"])
        assert r["test"] == test and r["status"] == 0, r
        assert r["cycles"] > 0 and r["iters"] > 0
    r = ml.unpack_lb_result(hart.l39()["data"])                           # mtime_ratio: cycles*1000/ticks
    assert r["bytes"] == r["cycles"] * 1000 // r["iters"] and r["iters"] >= 50_000
    for test, status in ((3, 3), (7, 4), (99, 1)):                         # no V on x86, zero-dev denied, unknown
        hart.cmd(ml.CMD["CALL"], arg_entry=PAY, policy=test, flags=1)
        assert ml.unpack_lb_result(hart.l39()["data"])["status"] == status
    hart.cmd(ml.CMD["CALL"], arg_entry=PAY, policy=1, arg_addr=ml.MB_BASE, arg_len=4096)
    assert ml.unpack_lb_result(hart.l39()["data"])["status"] == 2           # refuses the mailbox


# ---- concurrency -----------------------------------------------------------------------------------------

def test_threaded_host_and_hart(libs):
    """The hart loop runs in a thread (no GIL) while the host issues 300 INFERs back to back."""
    h = Hart(libs["main"])
    h.fake_blob()
    stop = ctypes.c_uint32(0)
    res = {}
    t = threading.Thread(target=lambda: res.update(r=h.lib.mon_x86_run_thread(ctypes.addressof(stop), 10**10)))
    t.start()
    rng = np.random.default_rng(15)
    try:
        for i in range(300):
            obs = rand_obs(rng)
            seq = h.issue(ml.CMD["INFER"], obs=obs, policy=i & 1)
            deadline = time.monotonic() + 5.0
            while h.r32(A["L1"] + 8) != seq:
                assert time.monotonic() < deadline, f"seq {seq} not done"
            assert h.r32(A["L1"] + 12) == S["IDLE"]
            a = h.l35()
            assert a["seq_echo"] == seq
            np.testing.assert_array_equal(np.float32(a["action"]), fake_act(obs, i & 1, 0))
    finally:
        stop.value = 1
        t.join(10)
    assert not t.is_alive() and res.get("r") == X86_RUNNING
    assert h.l36()["n_runs"] == 300 and h.l36()["n_crc_fail"] == 0


# ---- real kernel and stub --------------------------------------------------------------------------------

def test_stub_kernel_reports_itself(libs):
    h = Hart(libs["stub"], fake_kernel=False)
    assert h.lib.mon_x86_kernel_is_stub() == 1
    l36 = h.l36()
    assert l36["variants_built"] == 0 and l36["entry_flags"] & ml.ENTRY_FLAG["STUB_KERNEL"]
    l1 = h.load(PAY, b"x" * 400, ml.LOAD_KIND["BLOB"], expect_state=S["FAULT"])
    assert l1["mcause"] == E["BLOB"] and l1["mtval"] == 2**64 - 1


def _pack_blob(dtype: str, policies: str | None):
    sys.path.insert(0, str(RT / "x280" / "tools"))
    sys.path.insert(0, str(RT))
    try:
        import pack_weights as pw
    except Exception as e:   # pragma: no cover - depends on the kernel lane
        pytest.skip(f"x280/tools/pack_weights.py not importable: {e}")
    return pw.pack(pw.ONNX_DIR, dtype, pw.parse_policies(policies))


@pytest.mark.parametrize("dtype,policies,variant,tol", [("fp16", None, 2, 1e-2), ("fp32", "walk", 0, 5e-5)])
def test_real_kernel_through_the_mailbox(libs, dtype, policies, variant, tol):
    """INFER through the monitor == a direct wbc_infer call, bit for bit; and close to onnxruntime."""
    lib = libs["main"]
    if lib.mon_x86_kernel_is_stub():
        pytest.skip("x86 library built with the stub kernel")
    blob = _pack_blob(dtype, policies)
    h = Hart(lib, fake_kernel=False)
    assert ml.PAYLOAD_BASE + len(blob) <= ml.LIM_END
    h.load(ml.PAYLOAD_BASE, blob, ml.LOAD_KIND["BLOB"])
    lib.wbc_infer.restype = ctypes.c_int
    lib.wbc_infer.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p]
    import onnxruntime as ort
    onnx_dir = Path(os.environ.get("ONNX") or
                    Path.home() / "experiments/gr00t/Isaac-GR00T-n1.6/external_dependencies/"
                    "GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy")
    names = ("GR00T-WholeBodyControl-Balance", "GR00T-WholeBodyControl-Walk")
    pols = [0, 1] if policies is None else [1]
    sess = {p: ort.InferenceSession(str(onnx_dir / f"{names[p]}.onnx"), providers=["CPUExecutionProvider"])
            for p in pols}
    rng = np.random.default_rng(16)
    scratch = (ctypes.c_uint8 * 16384)()
    blob_ptr = h.ptr + (ml.PAYLOAD_BASE - ml.LIM_BASE)
    for i in range(12):
        policy = pols[i % len(pols)]
        obs = rng.standard_normal(516).astype(np.float32)
        h.cmd(ml.CMD["INFER"], obs=obs, policy=policy, flags=ml.variant_flags(variant))
        got = np.float32(h.l35()["action"])
        ref = np.zeros(15, np.float32)
        obs_c = np.ascontiguousarray(obs)
        assert lib.wbc_infer(blob_ptr, policy, variant, obs_c.ctypes.data, ref.ctypes.data,
                             ctypes.addressof(scratch)) == 0
        np.testing.assert_array_equal(got, ref)
        s = sess[policy]
        ort_out = s.run(None, {s.get_inputs()[0].name: obs[None, :]})[0][0]
        assert np.max(np.abs(got - ort_out)) <= tol


# ---- harts 1-3 (mh.c on x86: secondary threads against the same fake LIM; mh.h layout) ---------------------------

KIND_MH_BENCH = 6
RNMI_CAUSE = (1 << 63) | 2
STAGE0_SEC_WFI = 0x2001_013A


@pytest.fixture
def mhart(libs):
    h = Hart(libs["main"])
    yield h
    h.lib.mon_x86_secondary_stop_all()   # no secondary thread may outlive this test's LIM buffer


def _mh(h: Hart) -> dict:
    return {"ctl": ml.unpack_mh_ctl(h.read(ml.MH_CTL, 64)),
            "stat": {i: ml.unpack_mh_stat(h.read(ml.mh_stat_addr(i), 64)) for i in (1, 2, 3)},
            "aux": {i: ml.unpack_mh_aux(h.read(ml.mh_aux_addr(i), 64)) for i in (1, 2, 3)}}


def _adopt(h: Hart, harts=(1, 2, 3), timeout_s: float = 5.0) -> dict:
    """The host's RNMI with the --harts 4 vectors + crt0's secondary path, then wait for the stat lines (IDLE, this
    epoch, heartbeat moving)."""
    for i in harts:
        assert h.lib.mon_x86_secondary_start(i, RNMI_CAUSE, STAGE0_SEC_WFI) == 0
    t0 = time.monotonic()
    first = {}
    while time.monotonic() - t0 < timeout_s:
        m = _mh(h)
        ok = True
        for i in harts:
            st = m["stat"][i]
            if not (st["magic"] == ml.MH_STAT_MAGIC and st["state"] == ml.MH_STATE["IDLE"]
                    and st["epoch"] == m["ctl"]["epoch"]):
                ok = False
                continue
            first.setdefault(i, st["heartbeat"])
            ok &= st["heartbeat"] != first[i]
        if ok:
            return m
        h.lib.mon_x86_step(64)            # hart 0 idles: the fake clock moves
        time.sleep(0.001)
    raise AssertionError(f"secondaries {harts} not up: {_mh(h)['stat']}")


def _ping(h: Hart, mask: int = 0xE, us: int = 2000) -> dict:
    h.cmd(ml.CMD["MH"], policy=ml.MH_OP["PING"], flags=mask, arg_len=us)
    l39 = h.l39()
    assert l39["kind"] == ml.L39_KIND["MH"]
    return l39


def test_mh_ping_before_any_adoption(hart):
    """Cold boot: hart 0 owns a fresh ctl line (magic, epoch); a ping finds nobody ready and returns at once."""
    m = _mh(hart)
    assert m["ctl"]["magic"] == ml.MH_CTL_MAGIC and m["ctl"]["epoch"] & 1 and m["ctl"]["gen"] == 0
    assert hart.l36()["mh"] == ml.MH_CAP | 1
    p = _ping(hart)
    assert (p["done"], p["mask"], p["ready"], p["trapped"]) == (0, 0xE, 1, 0)
    assert p["wake_ticks"] == [0xFFFFFFFF] * 3 and _mh(hart)["ctl"]["gen"] == 0   # nothing dispatched


def test_mh_adopt_three_harts_then_ping(mhart):
    m = _adopt(mhart)
    for i in (1, 2, 3):
        st = m["stat"][i]
        assert (st["hartid"], st["entries"], st["mncause"], st["mnepc"]) == (i, 1, RNMI_CAUSE, STAGE0_SEC_WFI)
        assert st["mode"] == ml.MH_MODE["WFI"] and st["mip"] & ml.MIP_MSIP and st["traps"] == 0
        assert mhart.lib.mon_x86_secondary_mnstatus(i) & 8 and mhart.lib.mon_x86_secondary_rearms(i) == 1  # re-armed
    p = _ping(mhart)
    assert (p["done"], p["ready"], p["gen"]) == (0xE, 0xF, 2) and all(w != 0xFFFFFFFF for w in p["wake_ticks"])
    assert mhart.l36()["mh"] == ml.MH_CAP | 0xF and mhart.l1()["state"] == S["IDLE"]
    after = _mh(mhart)
    assert all(after["stat"][i]["done_gen"] == 2 and after["aux"][i]["result"] == 0x9 ^ i for i in (1, 2, 3))
    assert after["ctl"]["n_dispatch"] == 1 and after["ctl"]["mask"] == 0xE


def test_mh_missing_and_hung_harts_are_reported(mhart):
    """mh_run's waits are bounded: a hart never adopted is not dispatched; a hung one (idle state, but its loop stopped)
    is dispatched and times out; both are missing from the done mask, nothing else is held up."""
    _adopt(mhart, (1, 3))
    p = _ping(mhart)
    assert (p["done"], p["ready"]) == (0xA, 0xB) and p["wake_ticks"][1] == 0xFFFFFFFF
    mhart.lib.mon_x86_secondary_hang(3, 1)
    time.sleep(0.01)
    p = _ping(mhart, mask=0xE, us=300)
    assert p["done"] == 0x2 and p["total_ticks"] >= 300 * 50            # hart 3 timed out after 300 us of mtime
    assert mhart.l1()["state"] == S["IDLE"] and mhart.l36()["n_traps"] == 0
    mhart.lib.mon_x86_secondary_hang(3, 0)
    assert _ping(mhart)["done"] == 0xA


def _mhb(h: Hart, kernel: int, mask: int, pattern: int = 0, kbit: int = 0, mode: int = 0, ws=0x0802_0000,
         ws_len=0x1_0000, tick: int = 3000, kernel0: int = 0) -> dict:
    h.cmd(ml.CMD["CALL"], arg_entry=ml.MHB_BASE, policy=kernel | kernel0 << 8,
          flags=ml.mhb_flags(mask, pattern, kbit, mode), tick=tick, arg_addr=ws, arg_len=ws_len)
    sm = ml.unpack_mhb_summary(h.l39()["data"])
    job = ml.unpack_mhb_job(h.read(ml.MHB_DATA, 64))
    res = {i: ml.unpack_mhb_res(h.read(ml.mhb_res_addr(i), 64)) for i in range(4)}
    return {"sm": sm, "job": job, "res": res}


@pytest.fixture
def mhb(mhart):
    """Harts 1-3 adopted, mh_bench.bin LOADed at MHB_BASE (the x86 stand-in registered there), a 64 KiB random working
    set at 0x0802_0000."""
    blob = (FW / "build" / "payloads" / "mh_bench.bin").read_bytes()
    assert len(blob) == ml.MHB_MAX_BYTES
    _adopt(mhart)
    mhart.load(ml.MHB_BASE, blob, ml.LOAD_KIND["CODE"])
    mhart.cmd(ml.CMD["FENCE_I"])
    mhart.register(ml.MHB_BASE, KIND_MH_BENCH)
    ws = np.random.default_rng(5).integers(0, 256, 0x1_0000, dtype=np.uint8).tobytes()
    mhart.write(0x0802_0000, ws)
    mhart.ws = ws
    return mhart


@pytest.mark.parametrize("kernel,pattern,kbit", [(1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 3, 0), (5, 0, 0), (6, 2, 0),
                                                 (8, 1, 0), (5, 4, 7), (9, 0, 0), (10, 3, 0), (11, 0, 0), (12, 0, 0)])
def test_mh_bench_results_on_x86(mhb, kernel, pattern, kbit):
    """Four harts, one kernel: every result line current, accesses a multiple of the chunk, bytes = accesses x bytes per
    access, the sink equal to mailbox_layout.mhb_sink over the working set (the riscv definition, also K5-K8)."""
    r = _mhb(mhb, kernel, 0xF, pattern, kbit)
    sm, job = r["sm"], r["job"]
    assert (sm["status"], sm["done"], sm["mask"]) == (0, 0xF, 0xF) and sm["tag"] == job["tag"] == mhb.l1()["seq_done"]
    for i in range(4):
        x = r["res"][i]
        assert (x["hart"], x["tag"], x["kernel"], x["status"]) == (i, sm["tag"], kernel, 0)
        assert x["accesses"] and x["accesses"] % ml.MHB_CHUNK[kernel] == 0
        assert x["bytes"] == x["accesses"] * ml.MHB_BPA[kernel]
        want = ml.mhb_sink(kernel, pattern, kbit, i, 0x0802_0000, 0x1_0000, x["accesses"],
                           lambda a, n: mhb.ws[a - 0x0802_0000:a - 0x0802_0000 + n])
        assert x["sink"] == want, (i, x["sink"], want)
    assert sm["bytes"] == sum(r["res"][i]["bytes"] for i in range(4))


def test_mh_bench_refusals_on_x86(mhb):
    S_ = ml.MHB_STATUS
    for kw, want in (({"ws": 0x0801_0000}, "ERANGE"), ({"ws": 0x081D_0000}, "ERANGE"), ({"ws": 0x0802_0040,
                     "ws_len": 0x8000}, "ERANGE"), ({"ws": 0x081D_0000 + 0x1_0000, "ws_len": 0x2_0000}, "ERANGE"),
                     ({"pattern": 4, "kbit": 5}, "EPATTERN"), ({"tick": 50_000_001}, "EWINDOW")):
        r = _mhb(mhb, 1, 0xF, **{k: v for k, v in kw.items() if k in ("ws", "ws_len", "pattern", "kbit", "tick")})
        assert (r["sm"]["status"], r["sm"]["done"]) == (S_[want], 0), (kw, r["sm"])
    assert _mhb(mhb, 13, 0xF)["sm"]["status"] == S_["EKERNEL"]
    coll = _mhb(mhb, 1, 0, mode=ml.MHB_MODE["COLLECT"])
    assert coll["sm"]["status"] == S_["ENOJOB"]


def test_mh_bench_background_and_collect_on_x86(mhb):
    start = _mhb(mhb, 6, 0xF, mode=ml.MHB_MODE["START"], tick=20_000)
    assert (start["sm"]["status"], start["sm"]["done"], start["job"]["dispatched"]) == (0, 0xE, 0xE)  # bit 0 dropped
    mhb.cmd(ml.CMD["NOP"])                                             # hart 0 serves while they run
    coll = _mhb(mhb, 0, 0, mode=ml.MHB_MODE["COLLECT"])
    assert (coll["sm"]["status"], coll["sm"]["done"], coll["sm"]["tag"]) == (0, 0xE, start["sm"]["tag"])
    assert _mhb(mhb, 0, 0, mode=ml.MHB_MODE["COLLECT"])["sm"]["status"] == ml.MHB_STATUS["ENOJOB"]   # collected once


def test_mh_secondary_trap_on_x86(mhb):
    r = _mhb(mhb, ml.MHB_K["TRAP"], 0xF, kbit=2)
    assert (r["sm"]["status"], r["sm"]["done"]) == (0, 0xB)
    m = _mh(mhb)
    assert m["stat"][2]["traps"] == 1 and m["stat"][2]["trap_gen"] == m["ctl"]["gen"]
    assert m["aux"][2]["mcause"] == 2 and m["aux"][2]["mtval"] == 0xC0001073
    assert mhb.l36()["n_traps"] == 0 and mhb.l1()["state"] == S["IDLE"]    # hart 0's lines untouched by it
    p = _ping(mhb)
    assert (p["done"], p["trapped"]) == (0xE, 1 << 2)


def test_mh_harts_write_only_their_lines_on_x86(mhb):
    """A ping and two mh_bench runs: the only LIM lines that change are hart 0's (L1, L36, L39, ctl, job line, result
    line 0) and each secondary's own stat / aux / private block / result line; never a host line, the mailbox or the
    working set."""
    before = mhb.mem.copy()
    host = np.zeros(ml.LIM_SIZE, dtype=bool)
    for _ in range(2):
        host[ml.ADDR["L0"] - ml.LIM_BASE:ml.ADDR["L0"] - ml.LIM_BASE + 64] = True
        _ping(mhb)
        _mhb(mhb, 1, 0xF)
    lines = set((np.nonzero((mhb.mem != before) & ~host)[0] // 64).tolist())
    ln = (lambda a: (a - ml.LIM_BASE) // 64)
    allowed = {ln(ml.ADDR[n]) for n in ("L1", "L36", "L39")} | {ln(ml.MH_CTL), ln(ml.MHB_DATA), ln(ml.mhb_res_addr(0))}
    for i in (1, 2, 3):
        allowed |= {ln(ml.mh_stat_addr(i)), ln(ml.mh_aux_addr(i)), ln(ml.mhb_res_addr(i))}
        allowed |= set(range(ln(ml.MH_PRIV[i]), ln(ml.MH_PRIV[i]) + 4))   # the job guard (jmp_buf)
    assert lines <= allowed, sorted(hex(ml.LIM_BASE + 64 * x) for x in lines - allowed)


def test_mh_survives_a_hart0_warm_entry_and_a_cold_boot_starts_a_new_epoch(mhart):
    m0 = _adopt(mhart)
    mhart.lib.mon_x86_step(10)
    seq = mhart.issue(ml.CMD["PARK"])
    assert mhart.run(seq) == X86_PARKED
    assert mhart.lib.mon_x86_reenter(RNMI_CAUSE, 0x0801_2000) == X86_RUNNING   # hart 0's warm entry: .bss cleared
    m1 = _mh(mhart)
    assert m1["ctl"]["epoch"] == m0["ctl"]["epoch"] and mhart.l36()["mh"] == ml.MH_CAP | 0xF
    assert _ping(mhart)["done"] == 0xE
    mhart.lib.mon_x86_secondary_stop_all()
    mhart.lib.mon_x86_set_clock(mhart.lib.mon_x86_mtime() + 1000, 1, 16)
    assert mhart.lib.mon_x86_boot() == X86_RUNNING                              # cold: a new epoch
    m2 = _mh(mhart)
    assert m2["ctl"]["epoch"] != m0["ctl"]["epoch"] and mhart.l36()["mh"] == ml.MH_CAP | 1
    assert _ping(mhart)["done"] == 0
