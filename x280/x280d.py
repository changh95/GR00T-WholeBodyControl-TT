#!/usr/bin/env python3
"""x280d: host daemon of the gear_wbc MLP on the Blackhole X280, L2CPU3 hart 0 (PLAN.md WP-3.4, D12, D13).

It owns the L2CPU tile for one device hold: the keeper fd and the fixed TLB windows (x280/host/kmd.py, the
only device path), the §5.2 boot sequence (the session's one L2CPU_RESET RMW), the LIM mailbox protocol of
§1.4.4 (x280/fw/mailbox_layout.py, the Python mirror of mailbox.h) and the ledger (x280/ledger.py). It serves
W-WBC (§1.4.3) on REP 127.0.0.1:5558, so rt_sim switches to the X280 with one URL (`--wbc tcp://...:5558`)
and falls back to ORT on any error reply. Every address and expected value comes from x280/regs.py.

    xpy $RT/x280/x280d.py probe --json                       # chip_session step 1 (L2CPU health check)
    xpy $RT/x280/x280d.py probe --tile 3,2,1,0 --via tcp://127.0.0.1:5558  # S5a (jobs/s5a_probe.sh): read-only, the
                                                             # daemon runs `serve --no-release --probe-tiles 3,2,1,0`
    X280_E1F_APPROVED=1 xpy $RT/x280/x280d.py serve --port 5558 --tile 3 --stage0  # chip_session step 3, boot:3
                                                             # (probe: --no-release, attach: --attach)
    xpy $RT/x280/x280d.py serve --backend sim --port 5558    # T-3.3: no device (x280/mailbox_sim.py)
    xpy $RT/x280/x280d.py boot --tile 3 --dry-run            # T-3.4: every NoC op and libwcio transaction
    xpy $RT/x280/x280d.py selftest --via tcp://127.0.0.1:5558   # jobs: ask the serving daemon
    xpy $RT/x280/x280d.py dump --via tcp://127.0.0.1:5558 --out DIR  # jobs: its tick ring for analyze_run (T-3.7)
    xpy $RT/x280/x280d.py serve ... --status-pub             # + HUD chip-panel status PUB on :5573 (PLAN §4.3 item 7)
    xpy $RT/x280/x280d.py lim-ecc --via tcp://127.0.0.1:5558   # S1d (jobs/s1d.sh): LIM init stores vs CCACHE ECC
    X280_E1F_APPROVED=1 X280_MH_APPROVED=1 xpy $RT/x280/x280d.py serve --port 5558 --tile 3 --stage0 --harts 4
    xpy $RT/x280/x280d.py harts adopt --via tcp://127.0.0.1:5558   # S3b (jobs/s3b_mh.sh): adopt harts 1-3 (mh.c)
    xpy $RT/x280/x280d.py bench --mh --via tcp://127.0.0.1:5558    # S3b: the 4-hart LIM bench (mh_bench payload)

With CHIP_SAFE_CPUS set (a session confined to the non-real-time cores), a --cpus or --sim-cpus outside it is
refused (exit 2) before anything starts, as server_g1 refuses its --cpus. Both take taskset lists ('5', '5,21',
'8-9'); --sim-cpus is one CPU per hart in tile order.

Backends: `device` (/dev/tenstorrent/0, only inside a DEVX hold: kmd.RealBackend refuses otherwise),
`sim` (kmd.FakeDevice + the monitor's x86 build as the hart, x280/mailbox_sim.py) and `--dry-run` (= sim,
plus the op and transaction logs on stdout). `--sim-ecc poison|clean|granule` gives the sim chip kmd.FakeEcc, a model
of the CCACHE's LIM data ECC (S1d's and S1c's ECC checks under the S1c hypothesis and its negation; refused on the
device). Inside a chip_session the serving daemon holds the only fd
(§5.2 invariant 3), so job steps go to it with `--via URL` (endpoint `x280d_admin`); without --via a
subcommand runs in this process against --backend.

Boot: on the device the standard boot is `--stage0` (E1F, PLAN §5.2; X280_E1F_APPROVED=1): the hart zero-fills LIM
itself. `boot` / `serve --boot` (method A: the host initialises every LIM line with 64 B stores, §5.2 step 2) poisons
the LIM ECC (S1c/S1d) and is refused on the device unless X280_ALLOW_METHOD_A=1; the sim and --dry-run keep it.

Harts 1-3 (the 4-hart bench, S3b): `--harts 4` (with --stage0 and X280_MH_APPROVED=1, the user's approval) writes their
RNMI handler / exception handler as crt0 _start / MH_SEC_RNMI_EXC before the release (hart 0 keeps the park stub, so a
default boot is byte-identical without the flag). `harts adopt` pulses their trigger bits once (kmd refuses bits 1-3
without those vectors, the E1F 'stub' state and the monitor's L36 multi-hart bit), waits for their stat lines in the MH
region (mh.h) and pings them; `bench --mh` runs the mh_bench matrix with BEU/ECC reads before, between and after.

Exit codes: 0 ok, 1 failed, 2 usage or missing build artefact (or a refused boot mode), 3 X280 state dirty
(chip_session tears down and DEVX resets the chip), 4 probe error / tile unusable (a reset would not fix it).
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import threading
import time
import zlib
from typing import Optional

import numpy as np
import zmq

T_START = time.monotonic()  # process start, for the status stream's uptime_s
RT = Path(__file__).resolve().parents[1]
if str(RT) not in sys.path:
    sys.path.insert(0, str(RT))

from rtloop import wire  # noqa: E402
from x280 import ledger as ledger_mod  # noqa: E402
from x280 import regs  # noqa: E402
from x280.fw import mailbox_layout as ml  # noqa: E402
from x280.host import kmd as K  # noqa: E402

# --- artefacts and constants ---------------------------------------------------------------------------------

FW_BUILD = RT / "x280" / "fw" / "build"
IMAGE_BIN = FW_BUILD / "monitor.bin"  # make -C $RT/x280/fw x280 (WP-3.2)
PARK_STUB_BIN = FW_BUILD / "park_stub.bin"
STAGE0_BIN = FW_BUILD / "stage0.bin"  # E1F only (DP3(d)): never loaded without X280_E1F_APPROVED=1
# Method A on the device (verify finding): its host LIM init leaves LIM ECC poisoned (S1c E1a; S1d: no host store
# clean), so a device `boot` / `serve --boot` needs this explicit override; --stage0 is the standard boot.
METHOD_A_ENV = "X280_ALLOW_METHOD_A"
METHOD_A_REFUSED = ("method-A boot refused on the device: its host LIM init (64 B stores to every LIM line, PLAN §5.2 "
                    "step 2) poisons the LIM ECC (S1c/S1d). Boot with --stage0 (E1F, the standard boot since "
                    "2026-10-25; needs X280_E1F_APPROVED=1), or set X280_ALLOW_METHOD_A=1 to run method A anyway")


def method_a_allowed() -> bool:
    return os.environ.get(METHOD_A_ENV) == "1"
PAYLOAD_DIR = FW_BUILD / "payloads"
KERNEL_BUILD = FW_BUILD / "kernel"  # make -C $RT/x280/fw -f Makefile.kernel kernel-all (WP-3.1)
LIBWBC_X86 = KERNEL_BUILD / "libwbc_mlp.so"
BLOB_FP16 = KERNEL_BUILD / "wbc_fp16.blob"
BLOB_FP32 = {0: KERNEL_BUILD / "wbc_fp32_balance.blob", 1: KERNEL_BUILD / "wbc_fp32_walk.blob"}

POLICIES = ("GR00T-WholeBodyControl-Balance", "GR00T-WholeBodyControl-Walk")  # index = WBC_POLICY_* (wbc_mlp.h)
VARIANTS = {"v0": 0, "v1": 1, "v2": 2, "v2s": 3, "v3": 4}  # WBC_VARIANT_* (wbc_mlp.h)
VARIANT_NAME = {v: k for k, v in VARIANTS.items()}
VARIANT_DTYPE = {0: "fp32", 1: "fp32", 2: "fp16", 3: "fp16", 4: "fp16"}  # WBC_EDTYPE otherwise (wbc_mlp.h)
X86_FAMILY = {0: 0, 1: 0, 2: 2, 3: 2, 4: 2}  # bit-identical x86 reference variant (wbc_mlp.h arithmetic contract)
X86_EXACT = frozenset({0, 2, 3, 4})  # must equal X86_FAMILY's x86 build bit for bit; v1 within X86_TOL
WDTYPE = {"fp32": 0, "fp16": 1}  # WBC_WDTYPE_* (wbc_mlp.h), blob header offset 20
PARITY_TOL = {"fp32": 5e-5, "fp16": 1e-2}  # T-3.1 / T-3.E5a / T-3.6 limits vs ORT
X86_TOL = 1e-5  # T-3.E5a: v1/v2 vs the x86 build of the same blob (v0 must be bit-identical)

POLL_FIRST_US = 100.0  # §1.4.4 step 4: first poll after 100 us ...
DEADLINE_MS = 4.0  # ... deadline 4 ms
DEADLINE_SLOW_MS = 12.0  # ... 12 ms under m3_rt_slow (DP11)
HB_STALL_MS = 5.0  # WP-3.4 watchdog: heartbeat stall > 5 ms -> error reply
HEALTH_S = 1.0  # §1.3: 1 Hz direct PLL4/heartbeat reads
SLOW_HEALTH_S = 10.0  # §4.1: PLL4 + heartbeat + counters to the status log every 10 s
BOOT_MONITOR_S = 2.0  # magic + heartbeat after the release (§5.2 step 10)
# blob LOAD: 86.7 M spike instructions (the monitor's slicing-by-8 CRC ~10 M + wbc_blob_check's bitwise one ~77 M,
# T-3.2b; 184.2 M with the bitwise monitor CRC) + ~2.6 M uncached loads (236 K data words and 1.89 M table entries
# for the monitor CRC, 472 K words for wbc_blob_check): ~0.11 s at IPC 1 plus 25-195 ms at 7.8-60 cycles per access
# (x280/R33_ESTIMATE.md), so 5 s is >= 10x even at IPC 0.5
LOAD_DEADLINE_S = 5.0
CMD_DEADLINE_S = 2.0
PARK_SILENCE_S = 0.001  # §5.2 PARK protocol: >= 1 ms of heartbeat silence before clearing the bit
PARK_SAMPLE_S = 0.0001  # heartbeat sampling period while waiting for that silence
SIM_LEDGER_ROOT = RT / "runs" / "x280sim"
STAGE0_ENTRY_WORD = regs.STAGE0_ENTRY_WORD  # stage0.S: the hart spins on this word (host-written line) ...
STAGE0_DONE_WORD = regs.STAGE0_DONE_WORD  # ... after writing 0x081E_0000 (= LIM end) here, the last LIM line
# Never read the done-word line before stage 0 has zeroed it (R26): 30,720 uncached 64 B stores take ~2.3 ms at
# 800 MHz and 60 cycles each; even at 1000 cycles per line and 200 MHz (the release clock) it is 154 ms, so 1 s
# leaves >= 6x on that pessimistic bound.
STAGE0_FIRST_POLL_S = 1.0
HOST_BENCH_BASE = 0x081D_E400  # 35 spare lines after the fp16 blob end 0x081D_E340 (kernel lane), below E1F's words
# BENCH (review F9): a 10-iteration pilot measures the per-iteration wall time, the real run's deadline is BENCH_MARGIN
# x that, and the scalar parity references (v0/v2s: 13-45 ms per call, x280/R33_ESTIMATE.md) run at most 100 times
BENCH_PILOT_ITERS = 10
BENCH_PILOT_DEADLINE_S = 30.0  # 10 x v2s at 10x the R33 worst case (45 ms) is 4.5 s
BENCH_MARGIN = 3.0
BENCH_SCALAR_ITERS_MAX = 100
QUIESCE_S = 2.0  # before a LOAD: wait this long for a running command to finish, then RNMI-park the hart
RING = 1 << 16  # tick-ring rows: 65,536 W-WBC calls = 21.8 min at 50 Hz
RING_FILE = "x280d_ticks.npy"  # scripts/analyze_run.py X280_RING
# S1d (jobs/s1d.sh, `lim-ecc`): each plan item gets its own never-written region; "x+y" overwrites one region with x,
# then y (does y heal what x left?). Default: the boot's store, the same without per-line fences, UMD's 2 x 32 B
# pattern, aligned 8 B UC stores, and 8 B stores over method A's result.
S1D_PLAN = "a64,a64nf,wc2x32,uc8,a64+uc8"
S1D_LINES = 1024  # 64 KiB per region: 16,384 word reads per re-read; five regions use 320 KiB of the weights area
S1D_PROBE_LINES = 4  # lines written, then read, one access at a time with the counters read after each
S1D_PARTIAL = 16  # 4 B UC writes (+ read-back) on the region after its first re-read, as a mailbox seq_req write is
S1D_SEED = 11
ECC_COUNT_KEYS = ("dat_fix_count", "dat_fail_count", "dir_fix_count", "dir_fail_count")
MH_APPROVED_ENV = "X280_MH_APPROVED"  # the user's approval of the 4-hart bench (2026-10-25): --harts 4 needs it
MH_BENCH_BIN = PAYLOAD_DIR / "mh_bench.bin"
MH_ADOPT_PULSE_S = 0.001  # trigger bits 1-3 held this long (the secondaries spin on them in crt0, then go on)
MH_ADOPT_WAIT_S = 0.050  # stat lines IDLE, mncause 2^63|2, heartbeat moving within 50 ms of the clear
MH_ADOPT_WAIT_SIM_S = 2.0  # the sim's secondaries are threads that sleep in wfi; give them a wall-clock margin
MH_PING_TIMEOUT_US = 2000
MH_BUSY_WAIT_S = 1.5  # quiesce: a background mh_bench job ends by its hard t_end <= 1 s
MH_CHECK_EVERY = 16  # bench --mh: BEU/ECC re-read after this many CALLs
MH_CPT = (14.4, 17.6)  # mcycle per mtime tick on silicon: 800 MHz / 50 MHz = 16, +-10 %
MH_SKEW_FRAC = 0.01  # start / end skew must stay < 1 % of the window
MH_SPLIT_S4 = 2.0  # build the MLP split only if S_4(K9) >= 2.0 (or K9 ~ K10: compute-bound)
MH_V2_INFER_CYCLES = 307_877  # v2-fp16 INFER on silicon (S3), the T_N model's base
MH_RNMI_CAUSE = (1 << 63) | 2  # mncause of the trigger-bit RNMI (ISA RNMIs.md:76-77)
GC_IDLE_S = 1.0  # serve runs with the cyclic GC off (kmd keeps ~1e5 op records); collect only after this much idle
GC_EVERY_S = 30.0

TICK_LOG_DTYPE = np.dtype([
    ("t_recv_ns", "<i8"), ("tick", "<i8"), ("seq", "<u4"), ("mb_seq", "<u4"), ("tile", "u1"), ("policy", "u1"),
    ("status", "u1"), ("early", "u1"), ("x280d_us", "<f4"), ("mb_us", "<f4"), ("compute_us", "<f4"),
    ("cycles", "<u4"), ("hb_age_us", "<f4"),
])  # §4.1: x280d per tick in a ring (recv->send us, cycles, status, heartbeat, early-dispatch hit/miss)
ST_OK, ST_TIMEOUT, ST_FAULT, ST_STALL, ST_INTERLOCK, ST_BAD, ST_DEGRADED = range(7)
EARLY_NONE, EARLY_HIT, EARLY_MISS = 0, 1, 2


def log(msg: str) -> None:
    print(f"x280d[{os.getpid()}] {time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


class X280Error(RuntimeError):
    pass


class Dirty(X280Error):
    """The tile or chip is not in the state this session needs: exit 3, DEVX resets after the hold."""


class Unusable(X280Error):
    """A reset would not help (tile harvested / disabled): exit 4."""


class MailboxTimeout(X280Error):
    pass


class MailboxFault(X280Error):
    def __init__(self, msg: str, l1: dict):
        super().__init__(msg)
        self.l1 = l1


def _pad64(b: bytes) -> bytes:
    return bytes(b) + bytes(-len(b) % regs.LINE)


def _read_artefact(p: Path) -> bytes:
    if not p.exists():
        raise FileNotFoundError(f"{p} missing (build: make -C {RT}/x280/fw x280; kernel: "
                                f"make -C {RT}/x280/fw -f Makefile.kernel kernel-all)")
    return p.read_bytes()


def blob_header(blob: bytes) -> dict:
    """Host-side check of a weight blob (format in x280/fw/wbc_mlp.h): magic, version, CRC-32, extent."""
    magic, version, hdr, total, crc, wdtype, layout, mask = struct.unpack_from("<8I", blob, 0)
    if magic != 0x42434257 or version != 1 or hdr != 320:  # WBC_BLOB_MAGIC "WBCB", version 1, 320 B header
        raise X280Error(f"not a v1 weight blob (magic 0x{magic:08x} version {version} hdr {hdr})")
    if total > len(blob) or zlib.crc32(blob[20:total]) != crc:  # CRC over [20, total_len) (WBC_BLOB_CRC_FROM)
        raise X280Error("weight blob CRC/length mismatch")
    if regs.WEIGHTS_BASE + total > regs.LIM_END:
        raise X280Error(f"blob of {total} B does not fit LIM at 0x{regs.WEIGHTS_BASE:x}")
    return {"len": total, "crc32": crc, "wdtype": wdtype, "dtype": {0: "fp32", 1: "fp16"}.get(wdtype, "?"),
            "policy_mask": mask, "end": regs.WEIGHTS_BASE + total}


def ecc_delta(a: dict, b: dict) -> dict:
    """CCACHE ECC counter increments from register snapshot a to b (32-bit wrap) and, for each count that moved, the
    event address b holds (HIGH << 32 | LOW: the address of the last access that counted one, S1c). An address register
    means nothing until its count moves: S1c's E0 read garbage there (s1c_e0_keeper.json)."""
    d = {k: (b[k] - a[k]) & 0xFFFF_FFFF for k in ECC_COUNT_KEYS}
    for kind in ("dat_fix", "dat_fail", "dir_fix", "dir_fail"):
        d[f"{kind}_addr"] = (b[f"{kind}_high"] << 32 | b[f"{kind}_low"]) if d[f"{kind}_count"] else None
    return d


def lim_addr_info(addr: Optional[int], lo: Optional[int] = None, hi: Optional[int] = None) -> Optional[dict]:
    """Where an ECC event address lies: the range [lo, hi), a mailbox line (L38 = the KMD cleanup word), or elsewhere."""
    if addr is None:
        return None
    out = {"addr": hex(addr), "line": hex(addr & ~63), "off_in_line": addr & 63}
    if lo is not None and lo <= addr < hi:
        out["where"] = "inside the range"
    elif regs.MAILBOX_BASE <= addr < regs.MAILBOX_BASE + 64 * regs.LINE:
        n = (addr - regs.MAILBOX_BASE) // regs.LINE
        out["where"] = f"mailbox L{n}" + (" (MB_HOST_GONE: the SET_NOC_CLEANUP word)" if addr & ~63 == regs.MB_HOST_GONE
                                          else "")
    elif regs.LIM_BASE <= addr < regs.LIM_END:
        out["where"] = "LIM, outside the range"
    else:
        out["where"] = "outside LIM"
    return out


def _ecc_view(d: dict, lo: Optional[int] = None, hi: Optional[int] = None) -> dict:
    """ecc_delta for a JSON report: counts, and the event addresses decoded by lim_addr_info."""
    return {**{k: d[k] for k in ECC_COUNT_KEYS},
            **{f"{kind}_at": lim_addr_info(d[f"{kind}_addr"], lo, hi) for kind in ("dat_fix", "dat_fail")}}


def parse_lim_ecc_plan(plan) -> list:
    """'a64,uc8,a64+uc8' (or a list of such items) -> [['a64'], ['uc8'], ['a64', 'uc8']]: one fresh region per item,
    '+' = the next store overwrites the same region. Stores are kmd.LIM_INIT_STORES."""
    items = plan.split(",") if isinstance(plan, str) else [str(x) for x in plan]
    out = []
    for it in items:
        steps = [x.strip() for x in it.split("+") if x.strip()]
        bad = [x for x in steps if x not in K.LIM_INIT_STORES]
        if not steps or bad:
            raise ValueError(f"lim-ecc plan item {it!r}: stores {', '.join(K.LIM_INIT_STORES)}, joined by '+'")
        out.append(steps)
    if not out:
        raise ValueError("lim-ecc: empty plan")
    return out


def lim_pattern(seed: int):
    """Deterministic per-line content for E1a/E1b: word i of the line at addr = addr + 4i XOR a seed mix."""
    mix = (seed * 0x9E37_79B9) & 0xFFFF_FFFF

    def line(addr: int) -> bytes:
        return (np.arange(addr, addr + 64, 4, dtype=np.uint64).astype(np.uint32) ^ np.uint32(mix)).tobytes()
    return line


# --- S5a: the `probe --tile LIST` report (Session.tiles_probe) -----------------------------------------------


S5A_SCHEMA = 1  # `probe --tile LIST` report (Session.tiles_probe); x280/tools/m3_jobs.py s5a-summary reads it


def _tlb_summary(slots: dict) -> dict:
    """The raw slot words plus each distinct value with how many slots hold it and the first one (S5a's baseline)."""
    out = dict(slots)
    for kind in ("small", "large"):
        groups: dict = {}
        for i, words in enumerate(slots[kind]):
            groups.setdefault(tuple(words), []).append(i)
        out[f"{kind}_distinct"] = [{"words": list(w), "slots": len(ix), "first": ix[0]} for w, ix in groups.items()]
        out[f"{kind}_nonzero"] = sum(1 for words in slots[kind] if any(words))
    return out


def _tile_probe_warnings(t: int, tr: dict, noc_translation: Optional[int]) -> list:
    """What a probed tile's recorded registers show that the documents would not lead one to expect (never gating)."""
    out = []
    if tr.get("cross_read_equal") is False:
        out.append(f"L2CPU{t}: CCACHE0_CONFIG 0x{tr['ccache_config']:08x} raw != 0x{tr['ccache_config_translated']:08x} "
                   "translated (R31)")
    ecc = tr.get("ecc") or {}
    counts = {key: ecc[key] for key in ECC_COUNT_KEYS if ecc.get(key)}
    if counts:
        out.append(f"L2CPU{t}: CCACHE ECC counts at the start {counts} (baseline recorded)")
    niu = tr.get("niu0")
    if niu:
        node, ep = niu["node"], niu["endpoint"]
        if [node["x"], node["y"]] != tr["xy"] or (node["width"], node["height"]) != regs.NOC_GRID or not node["x_first"]:
            out.append(f"L2CPU{t}: NOC_NODE_ID 0x{niu['noc_node_id']:08x} {node} is not NoC #0 {tuple(tr['xy'])} on the "
                       f"{regs.NOC_GRID[0]} x {regs.NOC_GRID[1]} X-first grid")
        if ep["type"] != regs.NOC_ENDPOINT_TYPE_L2CPU or ep["noc"] != 0:
            out.append(f"L2CPU{t}: NOC_ENDPOINT_ID 0x{niu['noc_endpoint_id']:08x} {ep} is not an L2CPU on NoC #0")
        if niu["translation_enable"] != (noc_translation == 1):
            out.append(f"L2CPU{t}: NIU_CFG_0 bit 14 = {int(niu['translation_enable'])}, TAG_NOC_TRANSLATION = "
                       f"{noc_translation}")
        if niu["tile_clock_disable"]:
            out.append(f"L2CPU{t}: NIU_CFG_0 bit 12 (tile clock disable) is set")
    return out


def tiles_probe_rc(rep: dict) -> int:
    """Exit code of `probe --tile LIST`: 0 clean; 4 when only ENABLED_L2CPU != 0xF (harvested: a reset would not help);
    3 for any other failed check (X280 dirty); 1 when there is no report (an error reply)."""
    c = rep.get("checks") if isinstance(rep, dict) else None
    if not isinstance(c, dict):
        return 1
    if not all(v for key, v in c.items() if key != "enabled_l2cpu"):
        return 3
    return 0 if c.get("enabled_l2cpu") else 4


# --- x86 reference kernel (the same blob, ctypes) ------------------------------------------------------------


class X86Ref:
    """The kernel lane's x86 build (libwbc_mlp.so) on a 64 B-aligned copy of the blob: the bit-parity oracle."""

    def __init__(self, blob: bytes, path: Path = LIBWBC_X86):
        self.lib = ctypes.CDLL(str(path)) if Path(path).exists() else None
        if self.lib is None:
            return
        vp, u32 = ctypes.c_void_p, ctypes.c_uint32
        self.lib.wbc_infer.argtypes = [vp, u32, u32, vp, vp, vp]
        self.lib.wbc_infer.restype = ctypes.c_int
        buf = np.zeros(len(blob) + 64, np.uint8)
        off = (-buf.ctypes.data) % 64
        self.blob = buf[off:off + len(blob)]
        self.blob[:] = np.frombuffer(blob, np.uint8)
        self._buf = buf
        self.scratch = np.zeros(1024, np.float32)
        self.act = np.zeros(16, np.float32)

    def __call__(self, policy: int, variant: int, obs: np.ndarray) -> Optional[np.ndarray]:
        if self.lib is None:
            return None
        o = np.ascontiguousarray(obs, np.float32).reshape(-1)
        rc = self.lib.wbc_infer(self.blob.ctypes.data, policy, X86_FAMILY[variant], o.ctypes.data,
                                self.act.ctypes.data, self.scratch.ctypes.data)
        if rc != 0:
            raise X280Error(f"x86 wbc_infer rc={rc}")
        return self.act[:15].copy()


# --- host half of the §1.4.4 protocol ------------------------------------------------------------------------


class HostMailbox:
    """One tile's mailbox, written and read only through kmd's checked accessors (WC lines, UC words)."""

    def __init__(self, k: K.Kmd, tile: int, poll_sleep_s: float = 0.0):
        self.k, self.tile, self.poll_sleep_s = k, tile, poll_sleep_s
        self.seq = 0  # the seq_req value currently in L0 (written by us or adopted at boot/attach)

    def r32(self, addr: int) -> int:
        return self.k.read_lim32(addr, self.tile)

    def line(self, n: int) -> bytes:
        """Mailbox line Ln: 16 checked UC words in one libwcio call (kmd.read_lim_line)."""
        return self.k.read_lim_line(ml.line_addr(n), self.tile)

    def l1(self) -> dict:
        return ml.unpack_l1(self.line(1))

    def l36(self) -> dict:
        return ml.unpack_l36(self.line(36))

    def l39(self) -> dict:
        d = ml.unpack_l39(self.line(39))
        d["data"] = d["data"].hex()
        return d

    def heartbeat(self) -> int:
        return self.k.read_lim64_hlh(ml.ADDR["L1"], self.tile)

    def adopt(self) -> int:
        """The monitor adopts whatever seq_req it finds at entry (monitor.c mon_start): continue after it."""
        self.seq = self.r32(regs.MB_SEQ_REQ)
        return self.seq

    def issue(self, cmd: int, *, policy: int = 0, flags: int = 0, tick: int = 0, arg_addr: int = 0,
              arg_entry: int = 0, arg_len: int = 0, obs=None, obs_crc_delta: int = 0) -> int:
        """§1.4.4 steps 1-2: obs lines, L0 as one full line (seq_req bytes unchanged), sfence; then seq_req."""
        obs_crc = 0
        if obs is not None:
            ob = ml.pack_obs(np.asarray(obs, np.float32).reshape(-1))
            self.k.wc_write_lines(ml.ADDR["L2"], bytearray(ob), self.tile)
            obs_crc = (ml.crc32(ob) + obs_crc_delta) & 0xFFFF_FFFF
        seq = (self.seq + 1) & 0xFFFF_FFFF
        l0 = ml.pack_l0(cmd, seq_req=self.seq, policy=policy, flags=flags, tick=tick, arg_addr=arg_addr,
                        arg_entry=arg_entry, arg_len=arg_len, obs_crc32=obs_crc)
        self.k.wc_write_lines(ml.ADDR["L0"], bytearray(l0), self.tile)
        self.k.uc_write32(regs.MB_SEQ_REQ, seq, self.tile)
        self.seq = seq
        return seq

    def wait(self, seq: int, deadline_s: float, first_poll_s: float = POLL_FIRST_US * 1e-6) -> int:
        """§1.4.4 step 4: UC polls of L1.seq_done; returns L1.state (written before seq_done, monitor.c complete).
        Each poll reads seq_done and state as two adjacent words in one call, seq_done first (ascending address),
        so the state that comes with a matching seq_done is the one the monitor wrote before it."""
        t0 = time.perf_counter()
        if self.poll_sleep_s:
            time.sleep(first_poll_s)
        while time.perf_counter() - t0 < first_poll_s:
            pass
        sd = ml.ADDR["L1"] + ml.OFFSETS["l1.seq_done"]
        assert ml.OFFSETS["l1.state"] == ml.OFFSETS["l1.seq_done"] + 4
        while True:
            done, st = self.k.read_lim_words(sd, 2, self.tile)
            if done == seq:
                return st
            if time.perf_counter() - t0 > deadline_s:
                raise MailboxTimeout(f"L2CPU{self.tile}: seq {seq} not done after {deadline_s * 1e3:.1f} ms")
            if self.poll_sleep_s:
                time.sleep(self.poll_sleep_s)  # sim hart on this CPU: let it run (a spin would wait for preemption)

    def idle(self) -> bool:
        """The last command we issued (or adopted) is finished: L1.seq_done == our seq_req."""
        return self.r32(ml.ADDR["L1"] + ml.OFFSETS["l1.seq_done"]) == self.seq

    def action(self, seq: int) -> tuple:
        """L35 as one line read: (action f32[15], cycles_last); seq_echo must be `seq`."""
        words = self.k.read_lim_words(ml.ADDR["L35"], 16, self.tile)
        if words[15] != seq:
            raise X280Error(f"L35 seq_echo {words[15]} != {seq}")
        return (np.array(words[:15], np.uint32).view(np.float32),
                self.r32(ml.ADDR["L1"] + ml.OFFSETS["l1.cycles_last"]))

    def command(self, cmd: int, deadline_s: float = CMD_DEADLINE_S, ok_states=(ml.STATE["IDLE"],), **kw) -> dict:
        seq = self.issue(cmd, **kw)
        st = self.wait(seq, deadline_s)
        if st not in ok_states:
            l1 = self.l1()
            raise MailboxFault(f"L2CPU{self.tile}: cmd {cmd} seq {seq} -> state {st} mcause 0x{l1['mcause']:x} "
                               f"mepc 0x{l1['mepc']:x} mtval 0x{l1['mtval']:x}", l1)
        return {"seq": seq, "state": st}

    def infer(self, policy: int, variant: int, obs, tick: int, deadline_s: float, first_poll_s: float,
              obs_crc_delta: int = 0) -> tuple:
        """One INFER: (action f32[15], mailbox seq, cycles_last). Raises MailboxTimeout / MailboxFault."""
        seq = self.issue(ml.CMD["INFER"], policy=policy, flags=ml.variant_flags(variant), tick=tick, obs=obs,
                         obs_crc_delta=obs_crc_delta)
        st = self.wait(seq, deadline_s, first_poll_s)
        if st != ml.STATE["IDLE"]:
            l1 = self.l1()
            raise MailboxFault(f"L2CPU{self.tile}: INFER seq {seq} -> state {st} mcause 0x{l1['mcause']:x} "
                               f"mtval 0x{l1['mtval']:x}", l1)
        act, cyc = self.action(seq)
        return act, seq, cyc

    def load(self, dst: int, data: bytes, kind: int, deadline_s: float = LOAD_DEADLINE_S) -> dict:
        """LOAD (firmware order): data as full lines, then L37 (load_seq = this LOAD's seq), then L0, then seq_req.
        Nothing is written unless the range passes monitor.c cmd_load's own checks (review F4: the monitor refuses
        a bad dst only after the host has already written the data lines) and the previous command has finished
        (review F9: never overwrite weights a running BENCH/INFER still reads)."""
        end = dst + len(_pad64(data))
        if not data or dst % regs.LINE or dst < regs.LOAD_BASE or end > regs.LIM_END:
            raise X280Error(f"LOAD 0x{dst:x}+{len(data)} B refused before any write: dst must be 64 B aligned and "
                            f"[dst, dst+len) inside [0x{regs.LOAD_BASE:x}, 0x{regs.LIM_END:x}) (monitor.c cmd_load)")
        if not self.idle():
            raise X280Error(f"L2CPU{self.tile}: seq {self.seq} still running; LOAD refused before any write "
                            "(Session.quiesce first)")
        self.k.wc_write_lines(dst, bytearray(_pad64(data)), self.tile)
        seq = (self.seq + 1) & 0xFFFF_FFFF
        self.k.wc_write_lines(ml.ADDR["L37"], bytearray(ml.pack_l37(dst, len(data), ml.crc32(data), kind, seq)),
                              self.tile)
        return self.command(ml.CMD["LOAD"], deadline_s)


# --- one session -----------------------------------------------------------------------------------------------


def _hold_id() -> str:
    return os.environ.get("CHIP_SESSION_ID") or f"x280d-{time.strftime('%Y%m%d_%H%M%S')}-{os.getpid()}"


class Session:
    """The keeper fd, the windows, the ledger and one HostMailbox per tile, for one x280d process."""

    def __init__(self, backend: str = "sim", tiles: tuple = (regs.DEFAULT_TILE,), *, dtype: str = "fp16",
                 variant: Optional[str] = None, ledger_path: Optional[Path] = None, cross_read: bool = False,
                 sim_cpus: Optional[list] = None, hart: bool = True, blob_paths: Optional[dict] = None,
                 first_poll_us: float = POLL_FIRST_US, chip=None, sim_ecc: Optional[str] = None, harts: int = 1,
                 probe_tiles: tuple = ()):
        if backend not in ("device", "sim"):
            raise ValueError(f"backend {backend!r}")
        if probe_tiles and hart:
            raise ValueError("probe tiles (S5a) go only with a session that releases no hart (probe, serve --no-release)")
        if harts not in (1, regs.MH_HARTS):
            raise ValueError(f"--harts {harts}: 1 or {regs.MH_HARTS}")
        if harts == regs.MH_HARTS and tuple(tiles) != (regs.DEFAULT_TILE,):
            raise ValueError(f"--harts 4 is for L2CPU{regs.DEFAULT_TILE} alone (tiles 0-2 are not part of the 4-hart "
                             "bench approval)")
        if sim_ecc and backend != "sim":
            raise ValueError("--sim-ecc models the CCACHE ECC of the sim chip only; the device has its own")
        if sim_ecc and hart and chip is None:
            raise ValueError("--sim-ecc needs a session without a sim hart (no-release, e0, lim-init, lim-check, "
                             "lim-ecc): the model does not see the hart's own LIM accesses")
        if dtype == "fp16" and len(tiles) != 1:
            raise ValueError("fp16 holds both policies on one tile (DP4 default): give one --tile")
        if len(tiles) > 2 or len(set(tiles)) != len(tiles):
            raise ValueError(f"tiles {tiles}: one tile, or two for DP4 fp32 (Balance on the first, Walk on the second)")
        self.backend, self.tiles, self.dtype = backend, tuple(tiles), dtype
        self.sim = backend == "sim"
        # fp16 default v3 since S3v3 2026-10-25: bit-identical to v2 on silicon (E5a, 10,080 obs) and 0.72x its cycles
        v = variant or ("v3" if dtype == "fp16" else ("v0" if self.sim else "v1"))
        if VARIANT_DTYPE[VARIANTS[v]] != dtype:
            raise ValueError(f"variant {v} needs {VARIANT_DTYPE[VARIANTS[v]]} weights, not {dtype} (WBC_EDTYPE)")
        self.variant = VARIANTS[v]
        # policy index -> tile; fp32 = one policy per tile (DP4: L2CPU3 Balance, L2CPU2 Walk)
        self.policy_tile = {0: tiles[0], 1: tiles[0]} if dtype == "fp16" else {i: t for i, t in enumerate(tiles)}
        self.image = _pad64(_read_artefact(IMAGE_BIN))
        self.image_crc = zlib.crc32(self.image)
        self.park_stub = _read_artefact(PARK_STUB_BIN)
        if len(self.park_stub) != regs.GP_SCRATCH_SIZE:
            raise X280Error(f"park_stub.bin is {len(self.park_stub)} B, GP scratch is {regs.GP_SCRATCH_SIZE} B")
        self.blobs = {}
        for i, t in enumerate(self.tiles):
            path = (blob_paths or {}).get(t) or (BLOB_FP16 if dtype == "fp16" else BLOB_FP32[i])
            data = _read_artefact(Path(path))
            hdr = blob_header(data)
            if hdr["dtype"] != dtype:
                raise X280Error(f"{path}: {hdr['dtype']} weights, session is {dtype}")
            self.blobs[t] = {"path": str(path), "data": data, **hdr}
        self.ref = {t: X86Ref(b["data"]) for t, b in self.blobs.items()}
        self.chip = None
        if self.sim and sim_cpus and chip is None:
            # keep this thread (and every thread it creates: the sim watcher, zmq's I/O thread) off the CPUs the
            # spinning hart threads take: a woken thread queued behind an equal-priority spinner waits milliseconds
            rest = os.sched_getaffinity(0) - set(sim_cpus)
            if rest:
                os.sched_setaffinity(0, rest)
        if self.sim:
            from x280 import mailbox_sim
            # chip=: an existing simulated chip, i.e. a restarted x280d attaching inside the same "hold" (tests)
            self.chip = chip or mailbox_sim.SimChip.create(self.tiles if hart else (), image=self.image,
                                                           park_stub=self.park_stub, cpus=sim_cpus,
                                                           **({"ecc": sim_ecc} if sim_ecc else {}))
            be = self.chip
        else:
            be = K.RealBackend()
        self.k = K.Kmd(be, self.tiles, cross_read=cross_read, probe_tiles=tuple(probe_tiles))
        # sim: a spinning hart thread without its own CPU, and too few CPUs for it next to this thread -> poll by sleeping
        unpinned = len(self.tiles) - len(sim_cpus or []) if hart else 0
        free = len(os.sched_getaffinity(0) - set(sim_cpus or []))
        self.sim_shared_cpu = bool(self.sim and unpinned > 0 and free < 1 + unpinned)
        self.mb = {t: HostMailbox(self.k, t, poll_sleep_s=50e-6 if self.sim_shared_cpu else 0.0) for t in self.tiles}
        self.first_poll_s = first_poll_us * 1e-6
        self.hold_id = _hold_id()
        # sim: never the real ledger; a per-session file under runs/x280sim, removed again by close()
        self._own_ledger_dir = None
        if ledger_path is None and self.sim:
            self._own_ledger_dir = SIM_LEDGER_ROOT / f"{self.hold_id}-{os.getpid()}-{id(self):x}"
            self._own_ledger_dir.mkdir(parents=True, exist_ok=True)
            ledger_path = self._own_ledger_dir / "ledger.json"
        self.ledger = ledger_mod.Ledger(Path(ledger_path) if ledger_path else ledger_mod.LEDGER_PATH)
        self.steps: list = []  # (label, first op index, first libwcio txn seq)
        self.extra_fds: dict = {}  # tile -> O_APPEND fd holding that tile's SET_NOC_CLEANUP (one per fd, chardev.c:539)
        self.clock_mhz: Optional[float] = None
        self.mode = "closed"
        self.boot_parity: dict = {}
        self.monitor: dict = {}
        self.harts = harts
        self.mh: dict = {t: None for t in self.tiles}  # harts_adopt() result: adopted / lost masks
        self._mh_payload: dict = {t: False for t in self.tiles}  # mh_bench LOADed at MHB_BASE: BENCH arrays end below it

    # -- bookkeeping --
    def mark(self, label: str) -> None:
        self.steps.append((label, len(self.k.oplog), self.k.wcio.log_total()))
        log(label)

    def batch(self, tile: int):
        return self.k.l2cpu_batch(tile)

    def all_batches(self) -> ExitStack:
        st = ExitStack()
        try:
            for t in self.tiles:
                st.enter_context(self.batch(t))
        except BaseException:
            st.close()
            raise
        return st

    # -- step 1 --
    def open(self) -> None:
        self.mark("step 1: keeper fd (legacy O_RDWR), windows, settle >= 1 s")
        self.k.open()
        self.k.wait_settled()
        self.mode = "open"

    def interlocks(self, *, wayenable: bool = True, ecc: bool = True, released_ok: bool = False) -> dict:
        """§5.2 step 1 / WP-3.4 interlocks, read directly (ARC first, then the L2CPU tiles). A released tile of this
        session is dirty unless `released_ok` (attach, probe): a no-release session (S1c, lim-init, lim-check,
        host-only bench) must never run next to a live hart (review F3), and from a second process it would also be
        a second keeper fd inside the hold (§5.2 invariant 3). PLL4 not walk-ready (CNTL_0 & 0x1F != 0x12 or PLL_LOCK
        bit 4 clear: what kmd's per-write walk interlock refuses) is dirty too, so a boot stops before its first
        store instead of at its first PLL4 write."""
        k = self.k
        rst = k.read_l2cpu_reset()
        pll = k.read_pll4()
        tele = k.read_telemetry()
        hb = tele.get(regs.TAG_TIMER_HEARTBEAT)
        en = tele.get(regs.TAG_ENABLED_L2CPU, 0)
        vco, outs = regs.pll4_mhz(pll["cntl_1"], pll["cntl_5"], pll["use_postdiv"])
        lock = k.read32(regs.ARC_XY, regs.PLL_LOCK)
        r = {"l2cpu_reset": rst, "pll4": pll, "pll4_default": k.pll4_is_default(pll), "pll4_mhz": outs,
             "pll_lock": lock,  # kmd's per-step walk check needs CNTL_0 running (0x12) and PLL_LOCK bit 4 (review F5)
             "pll4_walk_ready": bool(pll["cntl_0"] & regs.PLL_CNTL_0_MASK == regs.PLL_CNTL_0_RUNNING
                                     and lock & regs.pll_lock_bit(4)),
             "timer_heartbeat": hb, "enabled_l2cpu": en, "tiles": {}, "dirty": [], "unusable": [],
             "released_tiles": [t for t in self.tiles if rst & regs.release_bit(t)]}
        if rst & regs.L2CPU_RESET_TILE_RESETS != regs.L2CPU_RESET_TILE_RESETS:
            r["dirty"].append(f"L2CPU_RESET[3:0] = 0x{rst & 0xF:x}, expected 0xF")
        if not r["pll4_default"]:  # the reason names an unexplained CNTL_1 (e.g. pll.c:226's 0x0040_0002) as such
            r["dirty"].append(f"PLL4 not at the CMFW defaults: {k.pll4_default_why(pll)}")
        if not r["pll4_walk_ready"]:  # verify: boot refuses here, before any LIM/GP/vector store, not at walk write 1
            r["dirty"].append(f"PLL4 not walk-ready: CNTL_0 0x{pll['cntl_0']:x} (& 0x{regs.PLL_CNTL_0_MASK:x} must be "
                              f"0x{regs.PLL_CNTL_0_RUNNING:x}: running, not bypassed), PLL_LOCK 0x{lock:x} (bit 4 must "
                              "be set); the boot's PLL4 walks would stop at their first write (CMFW relocking?)")
        if self.k.cross_read:
            r["noc_translation"] = k.noc_translation
        for t in self.tiles:
            tr = {"released": bool(rst & regs.release_bit(t)), "enabled": bool(en & regs.enabled_l2cpu_bit(t))}
            r["tiles"][str(t)] = tr
            if tr["released"] and not released_ok:
                r["dirty"].append(f"L2CPU{t} is released (a live hart): only attach or the serving x280d (--via) "
                                  "may touch it")
            if not tr["enabled"]:
                r["unusable"].append(f"ENABLED_L2CPU=0x{en:x}: L2CPU{t} disabled/harvested")
                continue
            if not r["pll4_default"]:
                continue  # invariant 4: no L2CPU-tile access unless PLL4 is at its defaults
            with self.batch(t):
                if wayenable:
                    tr["wayenable"] = k.read_wayenable(t)
                    if tr["wayenable"] != regs.WAYENABLE_EXPECTED:
                        r["dirty"].append(f"L2CPU{t} CCACHE0_WAYENABLE = {tr['wayenable']} (one-way: needs a chip reset)")
                tr["ccache_config"] = k.read_ccache_config(t)
                if self.k.cross_read and k.noc_translation == 1:
                    tr["ccache_config_translated"] = k.read_ccache_config(t, translated=True)
                elif self.k.cross_read:  # review F6: never a translated access with translation off
                    tr["cross_read"] = f"skipped: NoC translation off (TAG_NOC_TRANSLATION={k.noc_translation})"
                if ecc:
                    tr["ecc"] = k.read_ccache_ecc(t)
        self.clock_mhz = outs[self.tiles[0]] if r["pll4_default"] else None
        return r

    def ledger_hold(self, hb: int) -> dict:
        """Start this hold's ledger record once (chip_session's id); a restarted x280d keeps it (attach)."""
        cur = None
        try:
            cur = self.ledger.read()
        except ledger_mod.LedgerError as e:
            log(f"ledger unreadable ({e}): starting a new hold record")
        if cur is not None and cur["hold_id"] == self.hold_id:
            return {"epoch": "same_hold", "record": cur}
        out = self.ledger.start_hold(self.hold_id, int(hb or 0))
        log(f"ledger: hold {self.hold_id} started (epoch {out['epoch']})")
        return out

    # -- boot (§5.2 steps 1-10) --
    def boot(self) -> dict:
        """Method A (§5.2 steps 1-10, host LIM init). On the device only with X280_ALLOW_METHOD_A=1 (METHOD_A_REFUSED),
        checked before anything is opened; the sim and --dry-run run it as always."""
        if not self.sim and not method_a_allowed():
            raise X280Error(METHOD_A_REFUSED)
        if self.harts != 1:
            raise X280Error("--harts 4 goes only with the stage-0 boot (--stage0): kmd adopts harts 1-3 only there")
        if self.k.fd is None:
            self.open()
        inter = self.interlocks()
        if inter["unusable"]:
            raise Unusable("; ".join(inter["unusable"]))
        rst = inter["l2cpu_reset"]
        if rst != regs.L2CPU_RESET_AT_SESSION_START:
            inter["dirty"].append(f"L2CPU_RESET = 0x{rst:02x} at session start (a tile was released without a reset)")
        if inter["dirty"]:
            raise Dirty("; ".join(inter["dirty"]))
        hold = self.ledger_hold(inter["timer_heartbeat"])
        for t in self.tiles:
            if self.ledger.state(t) != "fresh":
                raise Dirty(f"ledger: L2CPU{t} is {self.ledger.state(t)} in hold {self.hold_id}")
        self.mark("step 2: LIM init, method A (every line: one 64 B NT store + sfence)")
        for t in self.tiles:
            with self.batch(t):
                self.k.lim_init(tile=t)
        self.mark("step 3: monitor image (64 B lines) and its read-back CRC")
        for t in self.tiles:
            with self.batch(t):
                try:
                    crc = self.k.write_image(t, self.image)
                except K.KmdError as e:
                    raise Dirty(f"L2CPU{t}: {e}") from e
            if crc != self.image_crc:
                raise Dirty(f"L2CPU{t}: image read-back CRC 0x{crc:08x} != 0x{self.image_crc:08x}")
        self.mark("steps 4-5: GP-scratch park stub + spin stub, RNMI handlers (+16h, +8), reset vectors; read back")
        for t in self.tiles:
            with self.batch(t):
                self.k.write_gp_scratch(t, self.park_stub)
                self.k.write_boot_vectors(t)
        self.mark("step 6: PLL4 800 -> 200 MHz, one field step at a time (clock.py order)")
        self._pll_walk(200)
        self.mark("step 7: the session's one L2CPU_RESET RMW (read | mask, [3:0] == 0xF), read back")
        prev, value, rb = self.k.release(self.tiles, self.ledger, self.image_crc, gp_code=self.park_stub)
        self.mark("step 8: PLL4 200 -> 800 MHz, defaults asserted")
        self._pll_walk(800, released=True)
        pll = self.k.read_pll4()
        if not self.k.pll4_is_default(pll):
            raise Dirty(f"PLL4 not at defaults after boot: {pll}")
        self.mark("step 9: L2PF control, SET_NOC_CLEANUP ('GONE' into L38 when our fd closes)")
        self._cleanup_and_l2pf()
        self.mark("step 10: monitor magic, fw_ver, heartbeat; SELFTEST")
        out = {"l2cpu_reset": {"before": prev, "written": value, "read_back": rb}, "epoch": hold.get("epoch"),
               "image_crc32": self.image_crc, "monitor": {}, "selftest": {}}
        for t in self.tiles:
            try:
                with self.batch(t):
                    out["monitor"][str(t)] = self.wait_monitor(t)
                    out["selftest"][str(t)] = self.selftest(t)
            except (X280Error, K.KmdError) as e:
                self.ledger.mark_burned(t)
                raise Dirty(f"L2CPU{t} released but the monitor did not come up: {e}") from e
        self.mode = "booted"
        return out

    def _pll_walk(self, mhz: int, released: bool = False) -> None:
        """A PLL4 walk; kmd stops at the first sign of a concurrent CMFW relock (review F5). Exit 3 either way;
        after the release the harts already ran on whatever clock that produced, so their tiles are burned."""
        try:
            self.k.pll4_step_to(mhz)
        except K.InterlockError as e:
            if released:
                for t in self.tiles:
                    try:
                        self.ledger.mark_burned(t)
                    except ledger_mod.LedgerError as le:
                        log(f"ledger: {le}")
            raise Dirty(f"PLL4 walk to {mhz} MHz: {e}") from e

    def boot_stage0(self) -> dict:
        """E1F (PLAN §5.2, DP3(d)): the hart initialises LIM itself. The standard boot on the device since 2026-10-25
        (S1c/S1d: host-written LIM init poisons the LIM ECC, so method A's boot() is refused there). Like every boot it
        spends the release, and it runs only with X280_E1F_APPROVED=1 (the user's DP3(d) approval).
        Order: interlocks; GP scratch = stage0.bin; reset vectors -> the stub, RNMI handler -> the stub entry (a
        re-run only re-zeroes LIM); PLL4 200 -> release -> 800; wait for the done word; only then any LIM write:
        image, entry word; monitor up; GP scratch = the park stub; FENCE_I (hart 0's L1I drops stage 0); SELFTEST."""
        if os.environ.get("X280_E1F_APPROVED") != "1":
            raise X280Error("E1F stage-0 needs the user's explicit DP3(d) approval: X280_E1F_APPROVED=1")
        if self.harts == regs.MH_HARTS and os.environ.get(MH_APPROVED_ENV) != "1":
            raise X280Error(f"--harts 4 needs the user's approval of the 4-hart bench: {MH_APPROVED_ENV}=1")
        stage0 = _read_artefact(STAGE0_BIN)
        if len(stage0) != regs.GP_SCRATCH_SIZE:
            raise X280Error(f"stage0.bin is {len(stage0)} B, expected {regs.GP_SCRATCH_SIZE}")
        if self.k.fd is None:
            self.open()
        inter = self.interlocks()
        if inter["unusable"]:
            raise Unusable("; ".join(inter["unusable"]))
        if inter["l2cpu_reset"] != regs.L2CPU_RESET_AT_SESSION_START:
            inter["dirty"].append(f"L2CPU_RESET = 0x{inter['l2cpu_reset']:02x} at session start")
        if inter["dirty"]:
            raise Dirty("; ".join(inter["dirty"]))
        self.ledger_hold(inter["timer_heartbeat"])
        for t in self.tiles:
            if self.ledger.state(t) != "fresh":
                raise Dirty(f"ledger: L2CPU{t} is {self.ledger.state(t)}")
        label = "E1F step A: GP scratch = stage0.bin; reset vectors -> 0x2001_0100; RNMI handler -> stub entry"
        mh = {"harts": regs.MH_HARTS} if self.harts == regs.MH_HARTS else {}
        if mh:
            label += "; harts 1-3 RNMI handler -> crt0 0x0800_0000, exception -> 0x0800_0200 (--harts 4)"
        self.mark(label)
        for t in self.tiles:
            with self.batch(t):
                self.k.write_gp_scratch(t, stage0)
                self.k.write_boot_vectors(t, reset_pc=regs.STAGE0_ENTRY, rnmi=regs.PARK_STUB, **mh)
        self.mark("E1F step B: PLL4 -> 200 MHz, the one L2CPU_RESET RMW, PLL4 -> 800 MHz")
        self._pll_walk(200)
        prev, value, rb = self.k.release(self.tiles, self.ledger, self.image_crc, gp_code=stage0, stage0=True)
        self._pll_walk(800, released=True)
        self.mark("E1F step C: wait for the stage-0 done word, then LIM counts as initialised")
        time.sleep(STAGE0_FIRST_POLL_S)
        out = {"l2cpu_reset": {"before": prev, "written": value, "read_back": rb}, "tiles": {}}
        for t in self.tiles:
            with self.batch(t):
                t0 = time.monotonic()
                while self.k.read_lim32(STAGE0_DONE_WORD, t, allow_uninit=True) != regs.LIM_END:
                    if time.monotonic() - t0 > 2.0:
                        self.ledger.mark_burned(t)
                        raise Dirty(f"L2CPU{t}: no stage-0 done word after 2 s")
                    time.sleep(0.005)
                self.k.assume_lim_initialised(t, regs.LIM_BASE, regs.LIM_END,
                                              "E1F: stage-0 on the hart zero-filled every LIM line (done word seen)")
        self.mark("E1F step D: image, read-back CRC, L2PF, SET_NOC_CLEANUP, entry word -> crt0")
        for t in self.tiles:
            with self.batch(t):
                try:
                    crc = self.k.write_image(t, self.image)
                except K.KmdError as e:
                    raise Dirty(f"L2CPU{t}: image after stage 0: {e}") from e
                if crc != self.image_crc:
                    raise Dirty(f"L2CPU{t}: image read-back CRC mismatch after stage 0")
        self._cleanup_and_l2pf()
        for t in self.tiles:
            with self.batch(t):
                self.k.stage0_enter(t, regs.RESET_PC)
        # The reset vectors keep pointing at 0x2001_0100: they are consumed only at reset (ISA README.md:46), and the
        # park stub jumps to crt0 at the fixed 0x0800_0000 (park_stub.S, lim.ld:63), not through them (review F1).
        # The swap rewrites code hart 0 has run: stage 0's line stays in its L1I (stage0.S refetches it after its own
        # fence.i) until the hart itself runs fence.i, which no host write can replace (Caches.md:22). crt0 runs one on
        # entry, and the FENCE_I after the swap orders one after it, as after every code LOAD; without either the first
        # RNMI ran stage 0 again on silicon (review F10, T-3.8 s4).
        self.mark("E1F step E: monitor up; GP scratch = the park stub; FENCE_I; SELFTEST")
        for t in self.tiles:
            try:
                with self.batch(t):
                    out["tiles"][str(t)] = {"monitor": self.wait_monitor(t)}
                    self.k.write_gp_scratch(t, self.park_stub)
                    self.mb[t].command(ml.CMD["FENCE_I"])
                    out["tiles"][str(t)]["selftest"] = self.selftest(t)
            except (X280Error, K.KmdError) as e:
                self.ledger.mark_burned(t)
                raise Dirty(f"L2CPU{t}: monitor did not come up after stage 0: {e}") from e
        self.mode = "booted"
        return out

    def _cleanup_and_l2pf(self, l2pf: bool = True) -> None:
        for i, t in enumerate(self.tiles):
            with self.batch(t):
                if l2pf:
                    self.k.l2pf_setup(t)
                fd = None
                if i:  # one cleanup per fd (KMD chardev.c:539): extra tiles get an O_APPEND fd (no power message)
                    fd = self.extra_fds.get(t) or self.k.open_test_fd()
                    self.extra_fds[t] = fd
                self.k.set_noc_cleanup(t, fd=fd)

    def wait_monitor(self, t: int, timeout_s: float = BOOT_MONITOR_S) -> dict:
        """§5.2 step 10 and the firmware's boot contract: fw_magic 'WBC2', fw_ver, a moving heartbeat, the real
        kernel linked (L36.entry_flags bit 2 clear) and the chosen variant built; then adopt seq_req."""
        mb = self.mb[t]
        t0 = time.monotonic()
        hb0, l1 = None, {}
        while time.monotonic() - t0 < timeout_s:
            l1 = mb.l1()
            if l1["fw_magic"] == ml.MAGIC and l1["fw_ver"] == ml.FW_VER:
                hb = mb.heartbeat()
                if hb0 is not None and hb != hb0:
                    break
                hb0 = hb
            time.sleep(0.002)
        else:
            raise X280Error(f"no monitor on L2CPU{t} after {timeout_s} s (L1 {l1})")
        l36 = mb.l36()
        if l36["entry_flags"] & ml.ENTRY_FLAG["STUB_KERNEL"]:
            raise X280Error("image links wbc_stub.c (every INFER would fail)")
        if not (l36["variants_built"] >> self.variant) & 1:
            raise X280Error(f"variant {VARIANT_NAME[self.variant]} not built (variants_built 0x{l36['variants_built']:x})")
        mb.adopt()
        self.monitor[t] = {"l1": l1, "l36": l36}
        return {"fw_magic": l1["fw_magic"], "fw_ver": l1["fw_ver"], "state": l1["state"], "vlenb": l1["vlenb"],
                "mhartid": l1["mhartid"], "build_id": l36["build_id"], "variants_built": l36["variants_built"],
                "mtime_hz": l36["mtime_hz"], "n_reentry": l36["n_reentry"], "entry_flags": l36["entry_flags"],
                "seq": mb.seq, "secs": round(time.monotonic() - t0, 4)}

    def selftest(self, t: int, arg_addr: int = 0, arg_len: int = 0) -> dict:
        mb = self.mb[t]
        r = mb.command(ml.CMD["SELFTEST"], CMD_DEADLINE_S, arg_addr=arg_addr, arg_len=arg_len)
        st, l36 = mb.l39(), mb.l36()
        st.update(seq=r["seq"], lines_expected=(arg_len or regs.LIM_SIZE) // regs.LINE, misa=l36["misa"],
                  beu_accrued_l36=l36["beu_accrued"])
        st["pass"] = bool(st["lines"] == st["lines_expected"] and st["misa_v"] and st["beu_accrued_after"] == 0)
        return st

    # -- attach (T-3.8: a restarted x280d inside the hold, no release) --
    def attach(self) -> dict:
        if self.k.fd is None:
            self.open()
        inter = self.interlocks(ecc=False, released_ok=True)
        if inter["unusable"]:
            raise Unusable("; ".join(inter["unusable"]))
        if inter["dirty"]:
            raise Dirty("; ".join(inter["dirty"]))
        rec = self.ledger.read()
        out = {"tiles": {}}
        for t in self.tiles:
            if not inter["tiles"][str(t)]["released"]:
                raise Dirty(f"attach: L2CPU{t} is not released (nothing to attach to)")
            if rec is None or rec["hold_id"] != self.hold_id or rec["tiles"][str(t)]["state"] not in ("released", "parked"):
                raise Dirty(f"attach: ledger does not show L2CPU{t} released in hold {self.hold_id}")
            self.k.assume_lim_initialised(t, regs.LIM_BASE, regs.LIM_END,
                                          f"attach: L2CPU{t} released and LIM initialised earlier in hold {self.hold_id}")
            with self.batch(t):
                resumed = self._resume_park(t)
                mon = self.wait_monitor(t)
                self.k.uc_write32(regs.MB_HOST_GONE, 0, t)  # the attach clear (kmd: L38 accepts only 0)
            out["tiles"][str(t)] = {**mon, **({"resumed_park": resumed} if resumed else {})}
        self._cleanup_and_l2pf(l2pf=False)
        self.mode = "attached"
        return out

    def _resume_park(self, t: int, timeout_s: float = 1.0) -> Optional[dict]:
        """Review F8: an x280d that died inside park() (between rnmi_set(1) and rnmi_set(0), e.g. `park --hold-s`)
        leaves hart 0 spinning in the park stub with the level-triggered bit set (park_stub.S; ISA RNMIs.md:15).
        Finish the PARK protocol for it: clear the bit, wait for the warm entry (n_reentry + 1), adopt seq_req and
        mark the ledger resumed. None if the bit was clear."""
        trig = self.k.read_rnmi_trigger(t)
        if not trig & 1:
            return None
        mb = self.mb[t]
        n0 = mb.l36()["n_reentry"]
        hb0, t0 = mb.heartbeat(), time.perf_counter()
        while time.perf_counter() - t0 < PARK_SILENCE_S * 5:
            time.sleep(PARK_SAMPLE_S)
        live = mb.heartbeat() != hb0
        self.k.rnmi_set(t, 0)
        log(f"attach: L2CPU{t} RNMI trigger 0x{trig:x} left set ({'heartbeat live' if live else 'hart parked'}): cleared")
        if live:  # RNMI not taken (NMIE = 0): the bit only had to go, nothing re-entered
            return {"trigger": trig, "heartbeat_live": True}
        t1 = time.perf_counter()
        n = n0
        while time.perf_counter() - t1 < timeout_s:
            n = mb.l36()["n_reentry"]
            if n == n0 + 1:
                break
            time.sleep(0.001)
        if n != n0 + 1:
            raise X280Error(f"attach: L2CPU{t} did not re-enter after clearing the RNMI bit (n_reentry {n0} -> {n})")
        mb.adopt()
        try:
            if self.ledger.state(t) == "parked":
                self.ledger.mark_resumed(t)
        except ledger_mod.LedgerError as e:
            log(f"ledger: {e}")
        return {"trigger": trig, "heartbeat_live": False, "n_reentry": [n0, n], "resume_s": round(time.perf_counter() - t1, 4)}

    # -- weights --
    def load_blob(self, t: int) -> dict:
        """The weight blob LOAD; a command still running (e.g. one a dead x280d left behind before this attach) is
        finished or RNMI-parked first, never overwritten (review F9)."""
        b = self.blobs[t]
        mb = self.mb[t]
        self.quiesce(t)
        t0 = time.perf_counter()
        r = mb.load(regs.WEIGHTS_BASE, b["data"], ml.LOAD_KIND["BLOB"], LOAD_DEADLINE_S)
        return {"tile": t, "len": b["len"], "crc32": b["crc32"], "end": b["end"], "seq": r["seq"],
                "secs": round(time.perf_counter() - t0, 4)}

    def use_blob(self, t: int, which: str) -> dict:
        """E4/E5a: LOAD another blob on tile t ('fp16', 'fp32_balance', 'fp32_walk' or 'session'); the monitor drops
        the old registration on overlap (monitor.c cmd_load), and the x86 reference follows."""
        if which == "session":
            path = self.blobs[t].get("session_path", self.blobs[t]["path"])
        else:
            path = {"fp16": BLOB_FP16, "fp32_balance": BLOB_FP32[0], "fp32_walk": BLOB_FP32[1]}[which]
        data = _read_artefact(Path(path))
        hdr = blob_header(data)
        session_path = self.blobs[t].get("session_path", self.blobs[t]["path"])
        with self.batch(t):
            self.quiesce(t)  # before the x86 reference changes too
            self.blobs[t] = {"path": str(path), "data": data, "session_path": session_path, **hdr}
            self.ref[t] = X86Ref(data)
            return self.load_blob(t)

    def quiesce(self, t: int, wait_s: float = QUIESCE_S) -> dict:
        """Before a LOAD (review F9): the hart must not be running a command that reads what the LOAD overwrites
        (a BENCH or v0 INFER past its host deadline). Wait for seq_done; if it never comes, RNMI-park the hart
        (the warm entry adopts seq_req, so the stale command is dropped, not re-run)."""
        mb = self.mb[t]
        t0 = time.perf_counter()
        while not mb.idle():
            if time.perf_counter() - t0 > wait_s:
                p = self.park(t)
                if not p["pass"]:
                    raise X280Error(f"L2CPU{t}: seq {mb.seq} still running after {wait_s} s and PARK failed: {p}")
                return {"waited_s": round(time.perf_counter() - t0, 4), "parked": True, **self._mh_quiesce(t)}
            time.sleep(0.001)
        return {"waited_s": round(time.perf_counter() - t0, 4), "parked": False, **self._mh_quiesce(t)}

    # -- harts 1-3 (mh.c; the 4-hart bench) --
    def _mh_quiesce(self, t: int) -> dict:
        """Adopted secondaries must be idle before a LOAD: a background mh_bench job executes payload code from the LOAD
        area (its hard t_end is <= 1 s). Nothing to wait for (and no key) on a session without adopted harts."""
        adopted = (self.mh.get(t) or {}).get("adopted", 0)
        if not adopted:
            return {}
        t0 = time.perf_counter()
        while True:
            busy = [h for h, d in self.mh_lines(t)["harts"].items() if (adopted >> int(h)) & 1 and d["busy"]]
            if not busy:
                return {"mh_waited_s": round(time.perf_counter() - t0, 4)}
            if time.perf_counter() - t0 > MH_BUSY_WAIT_S:
                raise X280Error(f"L2CPU{t}: harts {busy} still busy after {MH_BUSY_WAIT_S} s: no LOAD under a running job")
            time.sleep(0.001)

    def mh_lines(self, t: int) -> dict:
        """The MH region as the host reads it (LIM reads only): the ctl line and the stat/aux lines of harts 1-3, each
        with `valid` (magic, hart id and hart 0's epoch), `ready` (valid and idle) and `busy`."""
        rd = (lambda a: self.k.read_lim_line(a, t))
        ctl = ml.unpack_mh_ctl(rd(ml.MH_CTL))
        out = {"ctl": ctl, "harts": {}}
        for h in regs.MH_SEC_HARTS:
            st, ax = ml.unpack_mh_stat(rd(regs.mh_stat(h))), ml.unpack_mh_aux(rd(regs.mh_aux(h)))
            valid = bool(ctl["magic"] == ml.MH_CTL_MAGIC and st["magic"] == ml.MH_STAT_MAGIC and st["hartid"] == h
                         and st["epoch"] == ctl["epoch"])
            out["harts"][str(h)] = {**st, **{f"aux_{k}": v for k, v in ax.items()}, "valid": valid,
                                    "ready": valid and st["state"] == ml.MH_STATE["IDLE"],
                                    "busy": valid and st["state"] == ml.MH_STATE["BUSY"]}
        return out

    def mh_ping(self, t: int, mask: int = regs.MH_SEC_MASK, timeout_us: int = MH_PING_TIMEOUT_US) -> dict:
        """MB_CMD_MH ping: a NOP job on the ready harts of mask; done mask and wake latency (mtime ticks) from L39."""
        mb = self.mb[t]
        r = mb.command(ml.CMD["MH"], CMD_DEADLINE_S + timeout_us * 1e-6, policy=ml.MH_OP["PING"], flags=mask & 0xF,
                       arg_len=timeout_us)
        l39 = ml.unpack_l39(self.k.read_lim_line(ml.ADDR["L39"], t))
        return {"seq": r["seq"], "done": l39["done"], "mask": l39["mask"], "gen": l39["gen"],
                "wake_ticks": l39["wake_ticks"], "total_ticks": l39["total_ticks"], "ready": l39["ready"],
                "trapped": l39["trapped"], "l36_mh": mb.l36()["mh"]}

    def harts_adopt(self, t: int, mask: int = regs.MH_SEC_MASK) -> dict:
        """The 4-hart bench design A.3: preconditions (--harts 4 vectors of this session, E1F 'stub', monitor up with
        L36.mh's multi-hart bit: kmd.mh_rnmi_refusal; harts not marked lost), trigger bits 1-3 for 1 ms, cleared, then
        up to 50 ms for their stat lines (IDLE, mncause 2^63|2, heartbeat moving, own BEU accrued 0), then a ping. A
        hart that does not come up is marked lost: never RNMI'd again in this session, S3 goes on with hart 0."""
        mask &= regs.MH_SEC_MASK
        k = self.k
        out = {"tile": t, "mask": mask, "pass": False}
        prev = self.mh.get(t) or {}
        why = []
        if self.harts != regs.MH_HARTS:
            why.append("the session was not started with --harts 4")
        if self.mode != "booted":
            why.append(f"mode {self.mode}: only the session that booted the tile wrote the secondary vectors")
        if mask & prev.get("lost", 0):
            why.append(f"harts {mask & prev['lost']:#x} were lost earlier in this session")
        if not mask:
            why.append("empty hart mask")
        with self.batch(t):
            if not why:
                r = k.mh_rnmi_refusal(t, mask)
                if r:
                    why.append(r)
            out["hart_status_before"] = k.read_hart_status(t)
            before = self.mh_lines(t)
            out["before"] = {h: {"valid": d["valid"], "state": d["state"], "entries": d["entries"]}
                             for h, d in before["harts"].items()}
            if why:
                out["why"] = "; ".join(why)
                return out
            entries0 = {h: (d["entries"] if d["valid"] else 0) for h, d in before["harts"].items()}
            k.rnmi_set(t, mask)
            time.sleep(MH_ADOPT_PULSE_S)
            k.rnmi_set(t, 0)
            t0 = time.perf_counter()
            limit = MH_ADOPT_WAIT_SIM_S if self.sim else MH_ADOPT_WAIT_S
            hb0: dict = {}
            want = {str(h) for h in regs.MH_SEC_HARTS if (mask >> h) & 1}
            ok: set = set()
            lines = before
            while time.perf_counter() - t0 < limit:
                lines = self.mh_lines(t)
                for h in want - ok:
                    d = lines["harts"][h]
                    if not (d["ready"] and d["entries"] == entries0[h] + 1 and d["mncause"] == MH_RNMI_CAUSE):
                        continue
                    if h not in hb0:
                        hb0[h] = d["heartbeat"]
                    elif d["heartbeat"] != hb0[h]:
                        ok.add(h)
                if ok == want:
                    break
                time.sleep(0.0002)
            out["secs"] = round(time.perf_counter() - t0, 4)
            adopted = sum(1 << int(h) for h in ok)
            harts = {}
            for h in want:
                d = lines["harts"][h]
                harts[h] = {k_: d[k_] for k_ in ("valid", "state", "entries", "mncause", "mnepc", "mode", "mip",
                                                 "traps", "aux_wfi_ticks", "aux_beu_accrued")}
                harts[h]["heartbeat_moving"] = h in ok
                if h in ok:
                    harts[h]["beu"] = k.read_beu(t, hart=int(h))
            out["harts"] = harts
            beu_bad = [h for h in ok if harts[h]["beu"]["accrued"]]
            out["ping"] = self.mh_ping(t, adopted) if adopted else None
            out["hart_status_after"] = k.read_hart_status(t)
        lost = mask & ~adopted
        self.mh[t] = {"adopted": (prev.get("adopted", 0) & ~mask) | adopted, "lost": prev.get("lost", 0) | lost,
                      "when": time.time()}
        out.update(adopted=adopted, lost=lost, beu_accrued_harts=beu_bad)
        out["pass"] = bool(adopted == mask and not beu_bad and out["ping"] and out["ping"]["done"] == mask)
        return out

    # -- the 4-hart bench (x280/fw/mh_bench.c; the 4-hart bench design C-E) --
    MH_KERNEL_NAMES = {1: "ld8x8", 2: "ld8", 3: "chain", 4: "lw16", 5: "vle8m1", 6: "vle8m2", 7: "vle8m4", 8: "vle8m8",
                       9: "mlp", 10: "mlp_alu", 11: "alu", 12: "mlp_ld"}

    @staticmethod
    def mh_plan(plan: str = "full") -> list:
        """The bench --mh matrix (design C): masks x K1-K8 x P0-P3, the P4 bank-bit sweep (K5, 0xF, k = 6..12), K3 on
        hart 0 under K6 on harts 1-3 (loaded latency), K9 / K10 / K12 on 0x1, 0x3, 0xF. 'quick' is a small subset for the
        sim. Each entry: name, kernel, kernel0, mask, pattern, kbit."""
        out = []

        def add(k, mask, p=0, kb=0, k0=0, name=None):
            out.append({"name": name or f"K{k}/P{p}{f'k{kb}' if p == 4 else ''}/{mask:#x}", "kernel": k, "kernel0": k0,
                        "mask": mask, "pattern": p, "kbit": kb})
        if plan == "quick":
            for k in (1, 3, 5, 9, 10, 12):
                for m in (0x1, 0xF):
                    add(k, m)
            add(1, 0xF, 2)
            add(5, 0xF, 4, 6)
            add(6, 0xF, k0=3, name="K3@h0+K6/P0/0xf")
            return out
        for k in range(1, 9):
            for p in range(4):
                for m in (0x1, 0x2, 0x3, 0x5, 0xF, 0xE):
                    add(k, m, p)
        for kb in range(6, 13):
            add(5, 0xF, 4, kb)
        add(6, 0xF, k0=3, name="K3@h0+K6/P0/0xf")
        add(3, 0x1, name="K3/P0/0x1 (unloaded)")
        for k in (9, 10, 12):
            for m in (0x1, 0x3, 0xF):
                add(k, m)
        return out

    def _mh_events(self, t: int) -> dict:
        """BEU of harts 0-3 (accrued, cause) and the CCACHE ECC counts: any accrued event or new ECC fail aborts bench --mh."""
        k = self.k
        beu = {str(h): k.read_beu(t, hart=h) for h in range(regs.HARTS_PER_TILE)}
        return {"beu_accrued": {h: v["accrued"] for h, v in beu.items()}, "beu_cause": {h: v["cause"] for h, v in beu.items()},
                "ecc": k.read_ccache_ecc_counts(t)}

    @staticmethod
    def _mh_event_why(e0: dict, e: dict) -> str:
        bad = [f"hart {h} BEU accrued 0x{v:x}" for h, v in e["beu_accrued"].items() if v]
        if e["ecc"]["fail"] != e0["ecc"]["fail"]:
            bad.append(f"CCACHE ECC fail count {e0['ecc']['fail']} -> {e['ecc']['fail']}")
        return "; ".join(bad)

    def _mh_call(self, t: int, c: dict, *, window: int, hz: int, base: int, length: int, mode: int = 0) -> dict:
        """One mh_bench CALL; the summary (L39) and the result lines of the asked harts (LIM reads after seq_done)."""
        lead = hz * ml.MHB_LEAD_US // 1_000_000
        deadline = (lead + window) / hz + 0.05 + CMD_DEADLINE_S
        r = self.call(t, ml.MHB_BASE, policy=c["kernel"] | c.get("kernel0", 0) << 8,
                      flags=ml.mhb_flags(c["mask"], c.get("pattern", 0), c.get("kbit", 0), mode),
                      arg_addr=base, arg_len=length, tick=window, deadline_s=deadline)
        if r["timeout"]:
            q = self.quiesce(t)  # hart 0 finishes the CALL or is RNMI-parked before anything else is sent (review F9)
            raise X280Error(f"mh_bench {c.get('name')} mode {mode}: CALL seq {r['seq']} timed out (quiesced: {q})")
        if r["state"] != ml.STATE["IDLE"]:
            raise X280Error(f"mh_bench {c.get('name')} mode {mode}: CALL seq {r['seq']} "
                            f"{'timed out' if r['timeout'] else 'state %s mcause 0x%x' % (r['state'], r['l1']['mcause'])}")
        sm = ml.unpack_mhb_summary(bytes.fromhex(r["l39"]["data"]))
        res = {str(h): ml.unpack_mhb_res(self.k.read_lim_line(ml.mhb_res_addr(h), t)) for h in range(regs.HARTS_PER_TILE)
               if (sm["mask"] >> h) & 1}
        return {"seq": r["seq"], "summary": sm, "res": res}

    def _mh_metrics(self, call: dict, window: int, hz: int) -> dict:
        """Per hart: bytes / cycle, cycles / access, cycles / mtime tick (16 at 800/50 MHz); aggregate accesses and bytes
        per mtime tick over the span (max t1 - min t0, mtime is global), GB/s, and the run's flags."""
        sm = call["summary"]
        cur = {h: r for h, r in call["res"].items() if r["tag"] == sm["tag"] and r["status"] == 0}
        per, flags = {}, []
        for h, r in cur.items():
            cyc, acc, dt = r["cycles"], r["accesses"], r["dt"]
            per[h] = {"bytes": r["bytes"], "accesses": acc, "cycles": cyc, "dt": dt,
                      "bytes_per_cycle": r["bytes"] / cyc if cyc else None, "cycles_per_access": cyc / acc if acc else None,
                      "cycles_per_tick": cyc / dt if dt else None, "beu_accrued": r["beu_accrued"]}
            if r["beu_accrued"]:
                flags.append(f"hart {h} BEU accrued")
            if not self.sim and dt and not MH_CPT[0] <= cyc / dt <= MH_CPT[1]:
                flags.append(f"hart {h} cycles/tick {cyc / dt:.2f}")
        span = sm["span"]
        if sm["done"] != sm["mask"]:
            flags.append(f"done {sm['done']:#x} != mask {sm['mask']:#x}")
        if max(sm["start_skew"], sm["end_skew"]) > MH_SKEW_FRAC * window:
            flags.append(f"skew {sm['start_skew']}/{sm['end_skew']} ticks > 1 % of {window}")
        acc = sum(p["accesses"] for p in per.values())
        agg_bytes = sm["bytes"] / span if span else None
        return {"done": sm["done"], "status": sm["status"], "span": span, "start_skew": sm["start_skew"],
                "end_skew": sm["end_skew"], "agg_access_per_tick": acc / span if span else None,
                "agg_bytes_per_tick": agg_bytes, "gbps": agg_bytes * hz / 1e9 if agg_bytes is not None else None,
                "harts": per, "flags": flags}

    @staticmethod
    def _median(xs: list):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else None

    def bench_mh(self, t: int, plan: str = "full", repeats: int = 3, window_us: int = ml.MHB_WINDOW_US,
                 bench_iters: int = 200, ws_base: int = ml.MHB_WS_DEFAULT_BASE, ws_len: int = ml.MHB_WS_DEFAULT_LEN) -> dict:
        """bench --mh (design C-E): BEU/ECC/hart status before; LOAD + FENCE_I of mh_bench.bin at MHB_BASE; the matrix,
        `repeats` CALLs per entry (median), BEU/ECC re-read every MH_CHECK_EVERY CALLs (any event aborts); hart-0 BENCH v2
        x bench_iters with harts 1-3 idle, then under a background K6/P0 job; BEU/ECC/hart status after, the
        secondaries' heartbeats, an INFER parity spot check. Derived: S_N per kernel/pattern and the split verdict."""
        adopted = (self.mh.get(t) or {}).get("adopted", 0)
        if not adopted:
            raise X280Error("bench --mh: no secondary adopted (run `harts adopt` first)")
        if not (regs.WEIGHTS_BASE <= ws_base and ws_base + ws_len <= min(b["end"] for b in self.blobs.values())):
            raise X280Error(f"bench --mh: working set 0x{ws_base:x}+{ws_len} must lie inside the registered blob "
                            "(read-only weights: never the BENCH arrays or the payload)")
        hz = int(self.monitor.get(t, {}).get("l36", {}).get("mtime_hz") or 50_000_000)
        window = hz * window_us // 1_000_000
        out = {"tile": t, "plan": plan, "repeats": repeats, "window_ticks": window, "mtime_hz": hz,
               "ws": [hex(ws_base), ws_len], "adopted": adopted, "sim": self.sim, "configs": [], "abort": None}
        t_start = time.perf_counter()
        with self.batch(t):
            out["hart_status_before"] = self.k.read_hart_status(t)
            e0 = self._mh_events(t)
            out["events_before"] = e0
            n = 0
            try:
                why = self._mh_event_why(e0, e0)               # an accrued BEU event already: nothing is run
                if why:
                    raise X280Error(f"event before the bench: {why}")
                out["load"] = self.load_code(t, _read_artefact(MH_BENCH_BIN), dst=ml.MHB_BASE)
                self._mh_payload[t] = True
                for c in self.mh_plan(plan):
                    if c["mask"] & ~(adopted | 1):
                        out["configs"].append({**c, "skipped": f"harts {c['mask'] & ~(adopted | 1):#x} not adopted"})
                        continue
                    runs = []
                    for _ in range(repeats):
                        call = self._mh_call(t, c, window=window, hz=hz, base=ws_base, length=ws_len)
                        runs.append({"seq": call["seq"], **self._mh_metrics(call, window, hz)})
                        n += 1
                        if n % MH_CHECK_EVERY == 0:
                            why = self._mh_event_why(e0, self._mh_events(t))
                            if why:
                                raise X280Error(f"event after {n} CALLs: {why}")
                    med = self._median([r["agg_access_per_tick"] for r in runs])
                    h0 = [r["harts"].get("0", {}).get("cycles_per_access") for r in runs]
                    out["configs"].append({**c, "runs": runs, "agg_access_per_tick": med,
                                           "agg_bytes_per_tick": self._median([r["agg_bytes_per_tick"] for r in runs]),
                                           "gbps": self._median([r["gbps"] for r in runs]),
                                           "h0_cycles_per_access": self._median(h0),
                                           "flags": sorted({f for r in runs for f in r["flags"]})})
                why = self._mh_event_why(e0, self._mh_events(t))
                if why:
                    raise X280Error(f"event after the matrix: {why}")
                out["interference"] = self._mh_interference(t, adopted, bench_iters, hz, ws_base, ws_len)
            except (X280Error, K.KmdError) as e:
                out["abort"] = str(e)
            out["events_after"] = self._mh_events(t)
            out["hart_status_after"] = self.k.read_hart_status(t)
            hb0 = {h: d["heartbeat"] for h, d in self.mh_lines(t)["harts"].items()}
            time.sleep(0.002 if not self.sim else 0.05)
            lines = self.mh_lines(t)
            out["secondaries_after"] = {h: {"ready": d["ready"], "state": d["state"], "traps": d["traps"],
                                            "heartbeat_moving": d["heartbeat"] != hb0[h]}
                                        for h, d in lines["harts"].items() if (adopted >> int(h)) & 1}
        out["parity"] = self.check_parity(n=4)
        out["secs"] = round(time.perf_counter() - t_start, 3)
        out["derived"] = self.mh_derived(out)
        why = self._mh_event_why(e0, out["events_after"])
        out["pass"] = bool(not out["abort"] and not why and out["parity"]["pass"]
                           and all(v["heartbeat_moving"] for v in out["secondaries_after"].values()))
        if why and not out["abort"]:
            out["abort"] = f"event: {why}"
        return out

    def _mh_interference(self, t: int, adopted: int, iters: int, hz: int, base: int, length: int) -> dict:
        """Hart-0 BENCH v2 x iters with harts 1-3 idle, then with them in a background K6/P0 job (mode START, window =
        2x the idle BENCH's wall time + 20 ms, at most 1 s; then COLLECT): the slowdown is the direct cost of their LIM
        traffic on the INFER."""
        t0 = time.perf_counter()
        idle = self.bench(t, 0, iters, out_addr=HOST_BENCH_BASE)
        wall = time.perf_counter() - t0
        bg_mask = adopted & regs.MH_SEC_MASK
        window = min(hz, int((2 * wall + 0.02) * hz))
        c = {"name": "background K6/P0", "kernel": 6, "mask": bg_mask, "pattern": 0}
        start = self._mh_call(t, c, window=window, hz=hz, base=base, length=length, mode=ml.MHB_MODE["START"])
        busy = self.bench(t, 0, iters, out_addr=HOST_BENCH_BASE)
        coll = self._mh_call(t, {"name": "collect", "kernel": 6, "mask": 0}, window=window, hz=hz, base=base,
                             length=length, mode=ml.MHB_MODE["COLLECT"])
        keys = ("iters", "cyc_min", "cyc_max", "cyc_sum", "nondet", "us_mean", "us_p50", "us_p99")
        a, b = ({k: r.get(k) for k in keys} for r in (idle, busy))
        mean = (lambda r: r["cyc_sum"] / max(1, r["iters"]))
        return {"bench_idle": a, "bench_loaded": b, "slowdown_pct": (mean(busy) / mean(idle) - 1) * 100 if mean(idle) else None,
                "background": {"window_ticks": window, "dispatched": start["summary"]["done"],
                               "collected": self._mh_metrics(coll, window, hz)},
                "nondet": (idle.get("nondet"), busy.get("nondet"))}

    @staticmethod
    def mh_derived(out: dict) -> dict:
        """S_N = agg(mask) / agg(0x1) per kernel and pattern (accesses per mtime tick over the span), bank and sweep
        readings, K9 vs K10 vs K12 per-access cycles, the T_N model and the verdict: build the MLP split only if
        S_4(K9) >= 2.0, or if K9 is compute-bound (K9 within 15 % of K10 and >= 1.5x K12)."""
        agg = {}
        h0cpa = {}
        for c in out.get("configs", []):
            if "runs" not in c or c.get("kernel0"):
                continue
            agg[(c["kernel"], c["pattern"], c["kbit"], c["mask"])] = c["agg_access_per_tick"]
            if c["mask"] == 1:
                h0cpa[(c["kernel"], c["pattern"])] = c["h0_cycles_per_access"]
        scaling = {}
        for (k, p, kb, m), v in agg.items():
            base = agg.get((k, p, kb, 1))
            if m != 1 and v is not None and base:
                scaling.setdefault(f"K{k}/P{p}" + (f"k{kb}" if p == 4 else ""), {})[f"{m:#x}"] = round(v / base, 3)
        s4 = (lambda k, p=0: scaling.get(f"K{k}/P{p}", {}).get("0xf"))
        c9, c10, c12 = h0cpa.get((9, 0)), h0cpa.get((10, 0)), h0cpa.get((12, 0))
        compute_bound = bool(c9 and c10 and c12 and abs(c9 - c10) / c10 <= 0.15 and c9 >= 1.5 * c12)
        s4k9 = s4(9)
        sweep = {c["kbit"]: c["agg_access_per_tick"] for c in out.get("configs", [])
                 if c.get("pattern") == 4 and "runs" in c}
        bank = {}
        for k in (1, 5):
            p1, p2 = s4(k, 1), s4(k, 2)
            if p1 and p2:
                bank[f"K{k}"] = {"S4_P1_same_bank": p1, "S4_P2_own_bank": p2, "ratio": round(p2 / p1, 3)}

        def t_n(n, s):
            return MH_V2_INFER_CYCLES * (0.027 + 0.973 * 1.15 / min(n, s)) + 6 * 200 + 1200 if s else None
        a1, a2 = agg.get((1, 0, 0, 1)), agg.get((1, 0, 0, 2))
        anchors = {  # design C "sanity anchors on silicon": SELFTEST 62 cycles/line, INFER ~21 cycles per 64 B
            "k1_h0_cycles_per_line": h0cpa.get((1, 0)), "selftest_cycles_per_line": 62,
            "k9_h0_cycles_per_64B": c9 / 2 if c9 else None, "infer_v2_cycles_per_64B": 21,
            "hart1_alone_vs_hart0_alone_k1": round(a2 / a1, 3) if a1 and a2 else None}
        worth = bool((s4k9 is not None and s4k9 >= MH_SPLIT_S4) or compute_bound)
        return {"scaling": scaling, "S4_K9": s4k9, "S2_K9": scaling.get("K9/P0", {}).get("0x3"),
                "k9_k10_k12_cycles_per_row": [c9, c10, c12], "compute_bound": compute_bound,
                "bank": bank, "p4_sweep_agg": sweep,
                "p4_peak_k": max(sweep, key=lambda kb: sweep[kb] or 0) if sweep else None,
                "chain_cycles": {"unloaded": h0cpa.get((3, 0)),
                                 "under_k6": next((c["runs"][0]["harts"].get("0", {}).get("cycles_per_access")
                                                   for c in out.get("configs", []) if c.get("kernel0") == 3 and "runs" in c),
                                                  None)},
                "t4_estimate_cycles": t_n(4, s4k9), "v2_infer_cycles": MH_V2_INFER_CYCLES, "anchors": anchors,
                "split_worth_it": worth,
                "verdict": ("split worth it" if worth else "split not worth it") + (
                    f": S_4(K9) = {s4k9}" if s4k9 is not None else ": no S_4(K9)") + (
                    ", compute-bound" if compute_bound else ""),
                "rule": f"build the MLP split only if S_4(K9) >= {MH_SPLIT_S4} or K9 ~ K10 (compute-bound)",
                "meaningful": not out.get("sim")}

    def mh_status(self, t: int, hart_status: bool = True) -> dict:
        """Harts 1-3 as the host sees them: adoption record, the MH lines, L36.mh and (unless hart_status=False, as in the
        serving daemon's periodic status) the raw hart-status word 0x2001_0400 (its bit layout is undocumented)."""
        with self.batch(t):
            return {"harts": self.harts, "mh": self.mh.get(t), "lines": self.mh_lines(t),
                    "l36_mh": self.mb[t].l36()["mh"],
                    "hart_status": self.k.read_hart_status(t) if hart_status else None}

    def variant_for(self, t: int, variant: Optional[str]) -> int:
        v = self.variant if variant is None else VARIANTS[variant]
        if VARIANT_DTYPE[v] != self.blobs[t]["dtype"]:
            raise X280Error(f"variant {VARIANT_NAME[v]} needs {VARIANT_DTYPE[v]} weights; L2CPU{t} holds "
                            f"{self.blobs[t]['dtype']} (use_blob first)")
        return v

    def served_policies(self, t: int) -> list:
        mask = self.blobs[t]["policy_mask"]
        return [p for p, tt in self.policy_tile.items() if tt == t and (mask >> p) & 1]

    def check_parity(self, n: int = 4, seed: int = 0) -> dict:
        """Boot self-check: INFER on seeded N(0,1) obs vs the x86 build of the same blob (cf. wbc_gpu_server)."""
        rng = np.random.default_rng(seed)
        out = {}
        for t in self.tiles:
            for p in self.served_policies(t):
                worst = 0.0
                for _ in range(n):
                    obs = rng.standard_normal(ml.OBS_DIM).astype(np.float32)
                    with self.batch(t):
                        act, _, _ = self.mb[t].infer(p, self.variant, obs, 0, CMD_DEADLINE_S, self.first_poll_s)
                    ref = self.ref[t](p, self.variant, obs)
                    if ref is not None:
                        worst = max(worst, float(np.max(np.abs(act.astype(np.float64) - ref))))
                out[POLICIES[p]] = worst
        ok = all(v <= (0.0 if self.variant in X86_EXACT else X86_TOL) for v in out.values())
        self.boot_parity = {"max_abs_vs_x86": out, "pass": ok, "n": n}
        return self.boot_parity

    # -- LIM tests (S1c: E1a / E1b) --
    def lim_init(self, tile: int, method: str = "A", lo: int = regs.LIM_BASE, hi: int = regs.LIM_END,
                 seed: Optional[int] = None, vla_url: Optional[str] = None, store: str = "a64") -> dict:
        """T-3.E1a/b LIM init. `store` (method A only): kmd.LIM_INIT_STORES, 'a64' = the boot's method A. The CCACHE
        ECC counters are read before and after (reporting only: S1c saw the init itself count events)."""
        if method != "A" and store != "a64":
            raise ValueError(f"--store {store} applies to method A (host stores) only")
        t0 = time.perf_counter()
        pat = lim_pattern(seed) if seed is not None else None
        if method == "A":
            with self.batch(tile):
                e0 = self.k.read_ccache_ecc(tile)
                n = self.k.lim_init(lo, hi, tile, pattern=pat, store=store)
                e1 = self.k.read_ccache_ecc(tile)
        elif method == "B":
            with self.batch(tile):
                e0 = self.k.read_ccache_ecc(tile)
            n = self._lim_init_method_b(tile, lo, hi, pat, vla_url)
            with self.batch(tile):
                e1 = self.k.read_ccache_ecc(tile)
        else:
            raise ValueError("method must be A or B (C = E1F stage-0, see jobs/s1f_stage0.sh)")
        d = ecc_delta(e0, e1)
        return {"tile": tile, "method": method, "store": store if method == "A" else "server_g1 x280_lim_write",
                "lo": lo, "hi": hi, "lines": n, "seed": seed, "secs": round(time.perf_counter() - t0, 3),
                "ecc_before": e0, "ecc_after": e1, "ecc_delta": _ecc_view(d, lo, hi),
                "ecc_events_per_line": {"fix": round(d["dat_fix_count"] / max(n, 1), 4),
                                        "fail": round(d["dat_fail_count"] / max(n, 1), 4)}}

    def _lim_init_method_b(self, tile: int, lo: int, hi: int, pat, vla_url: Optional[str]) -> int:
        """Method B (§1.4.4): ttnn.cluster.write_to_core in 64 B-aligned 64 B chunks, through server_g1's
        `x280_lim_write(x, y, addr, data, chunk)` debug endpoint (rttt/server_g1.py, --debug-endpoints). The bytes
        travel on server_g1's fd, not through kmd, so x280d applies method A's guards itself (verify:all D12): the
        whole range passes kmd.lim_init_check before the first request (nothing is sent if any line is refused), and
        every 4 KiB request runs inside its own l2cpu_batch (PLL4 defaults, ENABLED_L2CPU) after the same per-call
        check (release state, per-line policy and ownership)."""
        url = vla_url or os.environ.get("CHIP_VLA_URL")
        if not url:
            raise X280Error("method B needs server_g1 (--vla-url or CHIP_VLA_URL)")
        x, y = regs.L2CPU_XY[tile]
        with self.batch(tile):
            self.k.lim_init_check(lo, hi, tile)
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 10_000)
        s.connect(url)
        try:
            for a in range(lo, hi, 4096):
                n = min(4096, hi - a)
                data = b"".join(pat(a + i) for i in range(0, n, 64)) if pat else bytes(n)
                with self.batch(tile):
                    self.k.lim_init_check(a, a + n, tile)
                    s.send(wire.request("x280_lim_write", {"x": x, "y": y, "addr": a, "data": data, "chunk": 64}))
                    rep = wire.loads(s.recv())
                if not isinstance(rep, dict) or "error" in rep:
                    raise X280Error(f"server_g1 x280_lim_write at 0x{a:x}: {rep}")
        finally:
            s.close(0)
        self.k.assume_lim_initialised(tile, lo, hi, f"method B via server_g1 x280_lim_write {url}")
        return (hi - lo) // regs.LINE

    def lim_check(self, tile: int, lo: int = regs.LIM_BASE, hi: int = regs.LIM_END, seed: Optional[int] = None,
                  stride: int = 1, partial: int = 0, cleanup_check: bool = False) -> dict:
        """T-3.E1a/E1b: pattern read-back, N partial 4 B UC writes on initialised lines, ECC and reset/WAYENABLE
        unchanged, and the SET_NOC_CLEANUP check on a separate O_APPEND test fd."""
        k = self.k
        pat = lim_pattern(seed) if seed is not None else (lambda a: bytes(64))
        t0 = time.perf_counter()
        r = {"tile": tile, "lo": lo, "hi": hi, "seed": seed, "stride": stride}
        rst0 = k.read_l2cpu_reset()
        with self.batch(tile):
            ecc0, way0 = k.read_ccache_ecc(tile), k.read_wayenable(tile)
            bad, words = [], 0
            for a in range(lo, hi, 64):
                want = pat(a)
                for i in range(0, 16, stride):
                    v = k.read_lim32(a + 4 * i, tile)
                    words += 1
                    if v != int.from_bytes(want[4 * i:4 * i + 4], "little"):
                        bad.append(a + 4 * i)
            r.update(words_read=words, mismatches=len(bad), first_bad=[hex(x) for x in bad[:8]])
            if partial:
                rng = np.random.default_rng(seed or 0)
                lines = [a for a in range(lo, hi, 64) if regs.lim_line_owner(a) == "host"
                         and not (regs.MAILBOX_BASE <= a < regs.MAILBOX_BASE + 64 * regs.LINE)]
                pick = rng.choice(len(lines), size=min(partial, len(lines)), replace=False)
                pbad = 0
                for j, li in enumerate(pick):
                    a = lines[int(li)] + 4 * int(rng.integers(0, 16))
                    v = int(rng.integers(0, 1 << 32))
                    k.uc_write32(a, v, tile)
                    pbad += k.read_lim32(a, tile) != v
                r.update(partial_writes=len(pick), partial_mismatches=int(pbad))
            if cleanup_check:
                fd = k.open_test_fd()
                k.set_noc_cleanup(tile, fd=fd)
                k.close_test_fd(fd)  # the KMD performs the cleanup write at release (chardev.c:1101-1115)
                v = k.read_lim32(regs.MB_HOST_GONE, tile)
                k.uc_write32(regs.MB_HOST_GONE, 0, tile)
                r.update(cleanup_value=v, cleanup_ok=v == regs.HOST_GONE_VALUE)
            ecc1, way1 = k.read_ccache_ecc(tile), k.read_wayenable(tile)
        rst1 = k.read_l2cpu_reset()
        fails = {n: (ecc0[n], ecc1[n]) for n in ecc0 if "fail" in n and ecc0[n] != ecc1[n]}
        d = ecc_delta(ecc0, ecc1)
        r.update(ecc_before=ecc0, ecc_after=ecc1, ecc_fail_delta=fails, l2cpu_reset=(rst0, rst1),
                 wayenable=(way0, way1), secs=round(time.perf_counter() - t0, 3),
                 ecc_delta=_ecc_view(d, lo, hi),
                 ecc_events_per_word_read={"fix": round(d["dat_fix_count"] / max(words, 1), 4),
                                           "fail": round(d["dat_fail_count"] / max(words, 1), 4)},
                 # the LIM accesses between ecc_before and ecc_after (each counts at most one fix and one fail, S1c)
                 lim_accesses={"word_reads": words, "partial_writes": r.get("partial_writes", 0),
                               "partial_read_backs": r.get("partial_writes", 0),
                               "cleanup": ("KMD 4 B write, read, 4 B clear" if cleanup_check else None)})
        # The gate is unchanged (T-3.E1a/b); `checks` names each part of it, `failed_checks` the ones that failed.
        r["checks"] = {"data_read_back": not bad, "partial_writes_read_back": not r.get("partial_mismatches"),
                       "ecc_fail_counts_unchanged": not fails, "l2cpu_reset_unchanged": rst0 == rst1,
                       "wayenable_unchanged_0": way0 == way1 == regs.WAYENABLE_EXPECTED,
                       "cleanup_wrote_gone": r.get("cleanup_ok", True)}
        r["failed_checks"] = [n for n, ok in r["checks"].items() if not ok]
        r["pass"] = bool(not bad and not r.get("partial_mismatches") and not fails and rst0 == rst1
                         and way0 == way1 == regs.WAYENABLE_EXPECTED and r.get("cleanup_ok", True))
        assert r["pass"] == all(r["checks"].values())
        return r

    # -- S1d: which host LIM write leaves the CCACHE's data ECC valid? --
    def lim_ecc(self, tile: int, plan=S1D_PLAN, lines: int = S1D_LINES, base: int = regs.S1D_BASE,
                probe_lines: int = S1D_PROBE_LINES, partial: int = S1D_PARTIAL, seed: int = S1D_SEED) -> dict:
        """S1d (jobs/s1d.sh). For each plan item, on its own never-written region of `lines` lines from `base` on:
        write it with each store of the item (kmd.lim_init: method A's checks, pre-release only), then read every word
        back with 4 B UC reads and the CCACHE ECC counters before and after each phase. The first `probe_lines` lines
        of each region are written, then read, one access at a time with the counters read after each (accesses per
        line, the event address, which words of a line fail). After the last store: `partial` 4 B UC writes with
        read-back (a mailbox seq_req write), then a second full re-read. A store is 'clean' when no fix and no fail
        event follows its write: re-read 1, the partial writes and re-read 2 count nothing and every word reads back
        ('reread_fail_free': only the re-reads' fail counts are 0, the literal "dat_fail_count unchanged").
        pass = the measurement is valid (L2CPU_RESET and WAYENABLE unchanged); the verdict is the per-variant `clean`
        (x280/tools/m3_jobs.py s1d-summary turns it into a lim-init recommendation)."""
        k = self.k
        variants = parse_lim_ecc_plan(plan)
        span = lines * regs.LINE
        end = base + len(variants) * span
        if lines < 1 or not 0 <= probe_lines <= lines or partial < 0:
            raise X280Error(f"lim-ecc: lines {lines}, probe-lines {probe_lines}, partial {partial}")
        if base % regs.LINE or base < regs.S1D_BASE or end > regs.LIM_END:
            raise X280Error(f"lim-ecc regions 0x{base:x}..0x{end:x} must be 64 B aligned inside the weights area "
                            f"[0x{regs.S1D_BASE:x}, 0x{regs.LIM_END:x})")
        i0, i1 = (base - regs.LIM_BASE) // regs.LINE, (end - regs.LIM_BASE) // regs.LINE
        if any(k.lim[tile].bits[i0:i1]):
            raise X280Error(f"lim-ecc: lines in 0x{base:x}..0x{end:x} were already written in this session; S1d "
                            "measures never-written lines (pick another --base)")
        t0 = time.perf_counter()
        rst0 = k.read_l2cpu_reset()
        with self.batch(tile):
            way0, e_start = k.read_wayenable(tile), k.read_ccache_ecc(tile)
        out = {"tile": tile, "plan": ",".join("+".join(v) for v in variants), "lines": lines, "base": hex(base),
               "probe_lines": probe_lines, "partial": partial, "seed": seed, "ecc_start": e_start, "variants": []}
        for vi, steps in enumerate(variants):
            lo = base + vi * span
            v = {"name": "+".join(steps), "lo": hex(lo), "hi": hex(lo + span), "lines": lines, "steps": []}
            for si, store in enumerate(steps):
                last = si == len(steps) - 1
                v["steps"].append(self._ecc_step(tile, store, lo, lo + span, seed + 100 * vi + si, probe_lines,
                                                 partial if last else 0, last))
            v["clean"] = v["steps"][-1]["clean"]
            v["reread_fail_free"] = v["steps"][-1]["reread_fail_free"]
            ws = v["steps"][-1].get("bulk_write_s_per_line")
            v["full_lim_write_s"] = round(ws * regs.LIM_LINES, 3) if ws is not None else None
            out["variants"].append(v)
        rst1 = k.read_l2cpu_reset()
        with self.batch(tile):
            way1, e_end = k.read_wayenable(tile), k.read_ccache_ecc(tile)
        out.update(ecc_end=e_end, l2cpu_reset=[rst0, rst1], wayenable=[way0, way1],
                   clean_variants=[v["name"] for v in out["variants"] if v["clean"]],
                   reread_fail_free_variants=[v["name"] for v in out["variants"] if v["reread_fail_free"]],
                   mismatches=sum(st["mismatches"] + st.get("partial_mismatches", 0) for v in out["variants"]
                                  for st in v["steps"]),
                   secs=round(time.perf_counter() - t0, 3))
        out["integrity"] = bool(rst0 == rst1 == regs.L2CPU_RESET_AT_SESSION_START
                                and way0 == way1 == regs.WAYENABLE_EXPECTED)
        out["pass"] = out["integrity"]
        return out

    def _ecc_step(self, tile: int, store: str, lo: int, hi: int, seed: int, probe_lines: int, partial: int,
                  last: bool) -> dict:
        """One S1d store over [lo, hi): probe-line writes, the bulk write, re-read 1 (probe lines word by word), and
        for the item's last store the partial writes and re-read 2. Counters are read only between accesses."""
        k = self.k
        pat = lim_pattern(seed)
        n = (hi - lo) // regs.LINE
        p = min(probe_lines, n)

        def words(a):
            return np.frombuffer(pat(a), "<u4")
        snap = {}
        with self.batch(tile):
            snap["before"] = prev = k.read_ccache_ecc(tile)
            pw = []
            for a in range(lo, lo + p * regs.LINE, regs.LINE):
                k.lim_init(a, a + regs.LINE, tile, pattern=pat, store=store)
                c1 = k.read_ccache_ecc(tile)
                c2 = k.read_ccache_ecc(tile)  # again: a count that moves now landed after the first read
                d = ecc_delta(prev, c2)
                pw.append({"line": hex(a), "fix": d["dat_fix_count"], "fail": d["dat_fail_count"],
                           "fix_at": None if d["dat_fix_addr"] is None else d["dat_fix_addr"] - a,
                           "fail_at": None if d["dat_fail_addr"] is None else d["dat_fail_addr"] - a,
                           "settled": c1 == c2})
                prev = c2
            tw = time.perf_counter()
            if p < n:
                k.lim_init(lo + p * regs.LINE, hi, tile, pattern=pat, store=store)
            tw = time.perf_counter() - tw
            snap["written"] = prev = k.read_ccache_ecc(tile)
            bad, pr = [], []
            for a in range(lo, lo + p * regs.LINE, regs.LINE):
                want = words(a)
                fail_mask = fix_mask = most = 0
                at_read = True
                for wi in range(16):
                    if k.read_lim32(a + 4 * wi, tile) != int(want[wi]):
                        bad.append(a + 4 * wi)
                    c = k.read_ccache_ecc(tile)
                    d = ecc_delta(prev, c)
                    prev = c
                    fail_mask |= bool(d["dat_fail_count"]) << wi
                    fix_mask |= bool(d["dat_fix_count"]) << wi
                    most = max(most, d["dat_fail_count"], d["dat_fix_count"])
                    at_read &= all(d[f"{kd}_addr"] in (None, a + 4 * wi) for kd in ("dat_fix", "dat_fail"))
                pr.append({"line": hex(a), "fail_words": f"0x{fail_mask:04x}", "fix_words": f"0x{fix_mask:04x}",
                           "max_events_per_read": most, "event_addr_is_the_read": bool(at_read)})
            for a in range(lo + p * regs.LINE, hi, regs.LINE):
                got = k.read_lim_words(a, 16, tile)
                bad += [a + 4 * i for i, (g, w) in enumerate(zip(got, words(a))) if g != int(w)]
            snap["read1"] = k.read_ccache_ecc(tile)
            r = {"store": store, "how": K.LIM_INIT_STORES[store][3], "lines": n, "probe_write": pw, "probe_read": pr,
                 "bulk_write_s": round(tw, 6), "bulk_write_s_per_line": (tw / (n - p)) if n > p else None,
                 "mismatches": len(bad), "first_bad": [hex(x) for x in bad[:8]]}
            phases = [("write", "before", "written"), ("read1", "written", "read1")]
            if last:
                # each partial write and its read-back with the counters read after each: the events a 4 B RMW and a
                # 4 B read count on what this store left (S1c's aggregate gave 1.76-1.87 per pair on poisoned lines)
                over, pbad, pp = {}, 0, []
                rng = np.random.default_rng(seed)
                prev = snap["read1"]
                for li in rng.choice(n, size=min(partial, n), replace=False):
                    a = lo + regs.LINE * int(li) + 4 * int(rng.integers(0, 16))
                    val = int(rng.integers(0, 1 << 32))
                    k.uc_write32(a, val, tile)
                    cw = k.read_ccache_ecc(tile)
                    back = k.read_lim32(a, tile)
                    cr = k.read_ccache_ecc(tile)
                    dw, dr = ecc_delta(prev, cw), ecc_delta(cw, cr)
                    pp.append({"addr": hex(a), "write": [dw["dat_fix_count"], dw["dat_fail_count"]],
                               "read_back": [dr["dat_fix_count"], dr["dat_fail_count"]], "ok": back == val})
                    prev = cr
                    over[a] = val
                    pbad += back != val
                snap["partial"] = prev
                r["probe_partial"] = pp  # [fix, fail] events of each write and of its read-back
                bad2 = []
                for a in range(lo, hi, regs.LINE):
                    got = k.read_lim_words(a, 16, tile)
                    for i, (g, w) in enumerate(zip(got, words(a))):
                        if g != over.get(a + 4 * i, int(w)):
                            bad2.append(a + 4 * i)
                snap["read2"] = k.read_ccache_ecc(tile)
                r.update(partial_writes=len(over), partial_mismatches=int(pbad), mismatches_read2=len(bad2))
                r["mismatches"] += len(bad2)
                phases += [("partial", "read1", "partial"), ("read2", "partial", "read2")]
        r["snapshots"] = snap
        r["phases"] = {ph: _ecc_view(ecc_delta(snap[a], snap[b]), lo, hi) for ph, a, b in phases}
        words_n = 16 * n
        w, r1 = r["phases"]["write"], r["phases"]["read1"]
        r["derived"] = {
            "write_events_per_line": {"fix": round(w["dat_fix_count"] / n, 4), "fail": round(w["dat_fail_count"] / n, 4)},
            "accesses_per_line_min": max([max(x["fix"], x["fail"]) for x in pw], default=None),
            "probe_event_offsets": sorted({x[kk] for x in pw for kk in ("fix_at", "fail_at") if x[kk] is not None}),
            "read1_events_per_word": {"fix": round(r1["dat_fix_count"] / words_n, 4),
                                      "fail": round(r1["dat_fail_count"] / words_n, 4)}}
        if last:
            r2 = r["phases"]["read2"]
            r["derived"]["read2_events_per_word"] = {"fix": round(r2["dat_fix_count"] / words_n, 4),
                                                     "fail": round(r2["dat_fail_count"] / words_n, 4)}
            pp = r.get("probe_partial") or []
            r["derived"]["partial_fail_events"] = {"writes": sum(x["write"][1] for x in pp),
                                                   "read_backs": sum(x["read_back"][1] for x in pp), "pairs": len(pp)}
        quiet = [ph for ph in r["phases"] if ph != "write"]
        rereads = [ph for ph in ("read1", "read2") if ph in r["phases"]]
        # the question S1d answers: no uncorrectable event on any full re-read of what this store wrote ...
        r["reread_fail_free"] = bool(r["mismatches"] == 0 and all(
            r["phases"][ph][c] == 0 for ph in rereads for c in ("dat_fail_count", "dir_fail_count")))
        # ... and the stricter verdict lim-init needs: nothing counted at all after the write (re-reads, the partial
        # writes the mailbox protocol makes every tick, their read-backs), every word as written
        r["clean"] = bool(r["mismatches"] == 0 and not r.get("partial_mismatches")
                          and all(r["phases"][ph][c] == 0 for ph in quiet for c in ECC_COUNT_KEYS))
        return r

    def e0(self, sample_s: float = 0.15) -> dict:
        """T-3.E0 register set (read-only, PLAN T-3.E0's list), telemetry heartbeat read twice >= 100 ms apart. No
        per-hart device is read while the harts are in reset (review F7: nothing documents that a hart's BEU
        answers then; E2's SELFTEST reads the hart-0 BEU from the hart side, monitor.c). With cross_read, the
        translated (8,29) read needs CMFW's NoC translation on (review F6); off, E0 fails with that reason."""
        r = self.interlocks()
        time.sleep(sample_s)
        r["timer_heartbeat_2"] = self.k.timer_heartbeat()
        r["heartbeat_increasing"] = r["timer_heartbeat_2"] > (r["timer_heartbeat"] or 0)
        for t in self.tiles:
            tr = r["tiles"][str(t)]
            if "ccache_config_translated" in tr:
                tr["cross_read_equal"] = tr["ccache_config"] == tr["ccache_config_translated"]
            elif self.k.cross_read and "ccache_config" in tr:
                tr["cross_read_equal"] = False  # not checkable: translation off (tr["cross_read"] says why)
        r["pass"] = bool(r["l2cpu_reset"] == regs.L2CPU_RESET_AT_SESSION_START and r["pll4_default"]
                         and all(tr.get("wayenable") == 0 and tr["enabled"] and tr.get("cross_read_equal", True)
                                 for tr in r["tiles"].values()) and r["heartbeat_increasing"])
        return r

    # -- S5a (docs/plan/vla-wbc-x280/research/four_tile_design.md step_plan): several tiles, read only --
    def tiles_probe(self, tiles: tuple) -> dict:
        """The read-only probe of L2CPU tiles `tiles`, in that order, on this session's keeper fd; nothing is written and
        no hart is released. ARC first: L2CPU_RESET, PLL4, PLL_LOCK, the telemetry (ENABLED_L2CPU, TAG_NOC_TRANSLATION,
        TIMER_HEARTBEAT, L2CPUCLK0-3). A tile that is not enabled, whose L2CPUCLK derives to 0 MHz from PLL4, that is
        released, or any tile while PLL4 is off its defaults (§5.2 invariant 4, R28) is reported and skipped, never
        read. Then per tile, inside its l2cpu_batch (PLL4 re-read): WAYENABLE, CCACHE0_CONFIG raw and (translation on)
        translated, the CCACHE ECC registers, the hart-status word, NIU #0's NOC_NODE_ID / NOC_ENDPOINT_ID /
        NIU_CFG_0, all 256 TLB window slots; L2CPU_RESET and PLL4 again at the end. The BEUs are not read: no source
        says a hart's BEU answers while the harts are in reset (review F7, kmd.read_beu). Only L2CPU_RESET 0x0F, PLL4
        at its defaults, ENABLED_L2CPU 0xF and WAYENABLE 0 (and a complete probe) decide the verdict; the rest is a
        recorded baseline, with `warnings` for what looks odd."""
        k = self.k
        tiles = tuple(int(t) for t in tiles)
        if not tiles or len(set(tiles)) != len(tiles) or any(t not in k.probe_tiles for t in tiles):
            raise X280Error(f"tiles probe {tiles}: this session's probe tiles are {k.probe_tiles} (standalone "
                            "`probe --tile LIST`, or `serve --no-release --probe-tiles LIST` for `probe --via`)")
        n0 = k.wcio.log_total()
        rst, pll = k.read_l2cpu_reset(), k.read_pll4()
        lock = k.read32(regs.ARC_XY, regs.PLL_LOCK)
        # CMFW's cable-fault mode clock-gates every tile outside column 15 through NIU_CFG_0, which only a read through
        # the tile itself would show (R28): ERROR_STATUS0 = SCRATCH_RAM[4] bit INIT_STAGE_CABLE_FAULT (FW reset.c:51-59,
        # 341-348; init.h:113; status_reg.h:37)
        err0 = k.read32(regs.ARC_XY, regs.scratch_ram(4))
        cable_fault = bool(err0 & 1 << 1)
        tele = k.read_telemetry((regs.TAG_TIMER_HEARTBEAT, regs.TAG_ENABLED_L2CPU, regs.TAG_NOC_TRANSLATION,
                                 *regs.TAG_L2CPUCLK))
        en, xlat = tele.get(regs.TAG_ENABLED_L2CPU, 0), tele.get(regs.TAG_NOC_TRANSLATION, 0)
        why_pll = k.pll4_default_why(pll)
        _, mhz = regs.pll4_mhz(pll["cntl_1"], pll["cntl_5"], pll["use_postdiv"])
        checks = {"l2cpu_reset": rst == regs.L2CPU_RESET_AT_SESSION_START, "pll4_default": not why_pll,
                  "enabled_l2cpu": en == 0xF, "wayenable": True, "complete": True, "no_cable_fault": not cable_fault}
        reasons, warnings = [], []
        if cable_fault:
            reasons.append(f"CMFW cable-fault mode (ERROR_STATUS0 0x{err0:x} bit 1): every L2CPU tile is clock-gated")
        if not checks["l2cpu_reset"]:
            reasons.append(f"L2CPU_RESET = 0x{rst:02x}, expected 0x{regs.L2CPU_RESET_AT_SESSION_START:02x}"
                           + (": a tile was released without a chip reset" if rst & 0xF0 else ""))
        if why_pll:
            reasons.append(f"PLL4 not at the CMFW defaults: {why_pll}")
        if not checks["enabled_l2cpu"]:
            off = [t for t in regs.L2CPU_XY if not en & regs.enabled_l2cpu_bit(t)]
            reasons.append(f"ENABLED_L2CPU = 0x{en:x}, expected 0xf: L2CPU{off} harvested/disabled (a "
                           f"{4 - len(off)}-rank plan)")
        walk_ready = bool(pll["cntl_0"] & regs.PLL_CNTL_0_MASK == regs.PLL_CNTL_0_RUNNING and lock & regs.pll_lock_bit(4))
        if not walk_ready:
            warnings.append(f"PLL4 not walk-ready (CNTL_0 0x{pll['cntl_0']:x}, PLL_LOCK 0x{lock:x}): S5b's walk would stop")
        if k.noc_translation != 1:
            warnings.append(f"NoC translation off (TAG_NOC_TRANSLATION={k.noc_translation}): no translated cross-read")
        rep = {"probe": "s5a", "schema": S5A_SCHEMA, "backend": "sim" if self.sim else "device", "tiles_order": list(tiles),
               "l2cpu_reset": rst, "pll4": pll, "pll4_default": not why_pll, "pll4_why": why_pll, "pll_lock": lock,
               "pll4_walk_ready": walk_ready, "l2cpuclk_mhz": mhz, "error_status0": err0, "cable_fault": cable_fault,
               "l2cpuclk_telemetry_mhz": [None if tele.get(g) is None else tele[g] & 0xFFFF for g in regs.TAG_L2CPUCLK],
               "timer_heartbeat": tele.get(regs.TAG_TIMER_HEARTBEAT), "enabled_l2cpu": en, "noc_translation": xlat,
               "beu": "not read: the harts are in reset and no source says a hart's BEU answers then (review F7, "
                      "kmd.read_beu); the BEU baseline comes from the hart side after the release (S5b SELFTEST)",
               "tiles": {}}
        for t in tiles:
            tr = {"xy": list(regs.L2CPU_XY[t]), "enabled": bool(en & regs.enabled_l2cpu_bit(t)),
                  "released": bool(rst & regs.release_bit(t)), "l2cpuclk_mhz": mhz[t], "probed": False, "reads": 0}
            rep["tiles"][str(t)] = tr
            skip = ("CMFW cable-fault mode: tile clock-gated (ERROR_STATUS0 bit 1; R28)" if cable_fault
                    else f"not enabled (ENABLED_L2CPU 0x{en:x}, bit {t} clear: harvested or disabled)" if not tr["enabled"]
                    else f"held in reset (L2CPU_RESET 0x{rst:02x}, bit {t} clear; R28)" if not rst & 1 << t
                    else f"clock off: L2CPUCLK{t} derives to 0 MHz from PLL4 (CNTL_5 0x{pll['cntl_5']:08x}, USE_POSTDIV "
                         f"0x{pll['use_postdiv']:x}; R28)" if not mhz[t]
                    else "PLL4 not at its CMFW defaults (§5.2 invariant 4)" if why_pll
                    else "released: a live hart, and S5a releases none" if tr["released"] else "")
            if skip:
                tr["skipped"] = f"{skip}: never read"
                continue
            t0 = k.wcio.log_total()
            try:
                with self.batch(t):
                    tr["wayenable"] = k.read_wayenable(t)
                    tr["ccache_config"] = k.read_ccache_config(t)
                    if k.noc_translation == 1:
                        tr["ccache_config_translated"] = k.read_ccache_config(t, translated=True)
                        tr["cross_read_equal"] = tr["ccache_config"] == tr["ccache_config_translated"]
                    else:  # review F6: never a translated access with translation off
                        tr["cross_read"] = f"skipped: NoC translation off (TAG_NOC_TRANSLATION={k.noc_translation})"
                    tr["ecc"] = k.read_ccache_ecc(t)
                    tr["hart_status"] = k.read_hart_status(t)
                    niu = k.read_niu0(t)
                    tr["niu0"] = {**niu, "node": regs.decode_noc_node_id(niu["noc_node_id"]),
                                  "endpoint": regs.decode_noc_endpoint_id(niu["noc_endpoint_id"]),
                                  "translation_enable": bool(niu["niu_cfg_0"] & regs.NIU_CFG_0_TRANSLATION),
                                  "tile_clock_disable": bool(niu["niu_cfg_0"] & regs.NIU_CFG_0_TILE_CLOCK_DISABLE)}
                    tr["tlb"] = _tlb_summary(k.read_tlb_slots(t))
                tr["probed"] = True
            except (K.KmdError, OSError) as e:
                tr["error"] = f"{type(e).__name__}: {e}"
                checks["complete"] = False
                reasons.append(f"L2CPU{t}: probe stopped: {tr['error']}")
            tr["reads"] = k.wcio.log_total() - t0
            if "wayenable" in tr and tr["wayenable"] != regs.WAYENABLE_EXPECTED:
                checks["wayenable"] = False
                reasons.append(f"L2CPU{t} CCACHE0_WAYENABLE = {tr['wayenable']}, expected 0 (one-way: needs a chip reset)")
            warnings += _tile_probe_warnings(t, tr, k.noc_translation)
        rep["l2cpu_reset_after"], rep["pll4_after"] = k.read_l2cpu_reset(), k.read_pll4()
        if rep["l2cpu_reset_after"] != rst:
            checks["l2cpu_reset"] = False
            reasons.append(f"L2CPU_RESET changed during the probe: 0x{rst:02x} -> 0x{rep['l2cpu_reset_after']:02x}")
        if rep["pll4_after"] != pll:
            why = f"PLL4 changed during the probe (CMFW relock?): {K._hexd(pll)} -> {K._hexd(rep['pll4_after'])}"
            if k.pll4_is_default(rep["pll4_after"]):
                warnings.append(why)
            else:
                checks["pll4_default"] = False
                reasons.append(why)
        rep.update(checks=checks, reasons=reasons, warnings=warnings, dirty=bool(reasons),
                   verdict="dirty" if reasons else "clean", attachable=False, reads=k.wcio.log_total() - n0,
                   harvested=[t for t in regs.L2CPU_XY if not en & regs.enabled_l2cpu_bit(t)])
        return rep

    # -- payloads (E2, E2m) --
    def load_code(self, t: int, data: bytes, dst: int = regs.WEIGHTS_BASE, fence: bool = True) -> dict:
        mb = self.mb[t]
        self.quiesce(t)
        r = mb.load(dst, data, ml.LOAD_KIND["CODE"], CMD_DEADLINE_S)
        if self.chip is not None:
            r["sim_payload"] = self.chip.note_code_load(t, dst, data)
        if fence:
            mb.command(ml.CMD["FENCE_I"])
        r.update(dst=dst, len=len(data))
        return r

    def call(self, t: int, entry: int, *, policy: int = 0, flags: int = 0, arg_addr: int = 0, arg_len: int = 0,
             tick: int = 0, deadline_s: float = CMD_DEADLINE_S) -> dict:
        mb = self.mb[t]
        seq = mb.issue(ml.CMD["CALL"], policy=policy, flags=flags, arg_addr=arg_addr, arg_entry=entry,
                       arg_len=arg_len, tick=tick)
        try:
            st = mb.wait(seq, deadline_s)
        except MailboxTimeout:
            return {"seq": seq, "timeout": True}
        r = {"seq": seq, "timeout": False, "state": st, "l1": mb.l1()}
        if st == ml.STATE["IDLE"]:
            r["l39"] = mb.l39()
        return r

    def park(self, t: int, hold_s: float = 0.0, timeout_s: float = 1.0) -> dict:
        """§5.2 PARK protocol: set bit h of 0x2001_0414 -> wait >= 1 ms of heartbeat silence -> (hold) -> clear ->
        n_reentry + 1 -> the monitor adopted seq_req."""
        mb, k = self.mb[t], self.k
        with self.batch(t):
            n0 = mb.l36()["n_reentry"]
            try:
                self.ledger.mark_parked(t)
            except ledger_mod.LedgerError as e:
                log(f"ledger: {e}")
            k.rnmi_set(t, 1)  # hart 0
            t0 = time.perf_counter()
            hb, t_change = mb.heartbeat(), time.perf_counter()
            silent = False
            while time.perf_counter() - t0 < timeout_s:
                time.sleep(PARK_SAMPLE_S)
                v = mb.heartbeat()
                if v != hb:
                    hb, t_change = v, time.perf_counter()
                elif time.perf_counter() - t_change >= PARK_SILENCE_S:
                    silent = True
                    break
            if hold_s:
                time.sleep(hold_s)
            k.rnmi_set(t, 0)
            if not silent:
                return {"pass": False, "why": "heartbeat never went silent (RNMI not taken: NMIE not re-armed?)"}
            t1 = time.perf_counter()
            l36 = mb.l36()
            while l36["n_reentry"] != n0 + 1 and time.perf_counter() - t1 < timeout_s:
                time.sleep(0.001)
                l36 = mb.l36()
            l1 = mb.l1()
            seq_req = mb.adopt()
            # warm entry: L1 shows REENTERED (or HOST_GONE), mcause = mncause, and seq_req was adopted, not re-run
            ok = (l36["n_reentry"] == n0 + 1 and l1["fw_magic"] == ml.MAGIC and l1["seq_done"] == seq_req
                  and l1["state"] in (ml.STATE["REENTERED"], ml.STATE["HOST_GONE"]))
            r = {"pass": bool(ok), "n_reentry": (n0, l36["n_reentry"]), "state": l1["state"],
                 "mncause": l1["mcause"], "mnepc": l1["mepc"], "silence_after_s": round(t_change - t0, 6),
                 "resume_s": round(time.perf_counter() - t1, 6), "seq": mb.seq}
            if not ok:  # stage 0 run by the RNMI zero-fills LIM, rewrites its done word and spins on the zeroed entry word
                done, entry = k.read_lim32(STAGE0_DONE_WORD, t), k.read_lim32(STAGE0_ENTRY_WORD, t)
                r["stage0_words"] = {"done": done, "entry": entry}
                if l1["fw_magic"] == 0 and done == regs.LIM_END and entry == 0:
                    r["why"] = ("the RNMI ran stage 0: LIM zero-filled (L1/L36 all 0), its done word written again, the "
                                "hart spinning on the zeroed entry word; hart 0's L1I still held the stage-0 line at "
                                "0x2001_0100 (no fence.i on hart 0 since E1F: crt0.S, boot_stage0 step E); the tile is "
                                "lost until a chip reset")
            try:
                self.ledger.mark_resumed(t)
            except ledger_mod.LedgerError as e:
                log(f"ledger: {e}")
        return r

    def e2(self, t: int, n: int = 10) -> dict:
        """T-3.E2 host part: LOAD+CALL of the checksum payload n times, then the RNMI re-arm loop n times."""
        out = {"checksum": [], "rearm": []}
        with self.batch(t):
            out["load_checksum"] = self.load_code(t, _read_artefact(PAYLOAD_DIR / "checksum.bin"))
            for i in range(n):
                ln = 1000 + 97 * i
                r = self.call(t, regs.WEIGHTS_BASE, policy=i, arg_addr=regs.IMAGE_BASE, arg_len=ln)
                want = zlib.crc32(self.image[:ln])
                out["checksum"].append(bool(not r["timeout"] and r.get("l39", {}).get("ret") == want))
            out["load_hang"] = self.load_code(t, _read_artefact(PAYLOAD_DIR / "hang.bin"))
        for i in range(n):
            with self.batch(t):
                r = self.call(t, regs.WEIGHTS_BASE, deadline_s=0.05)
            p = self.park(t)
            out["rearm"].append({"hung": r["timeout"], "park": p["pass"], "n_reentry": p["n_reentry"][1]})
        out["pass"] = bool(all(out["checksum"]) and all(x["hung"] and x["park"] for x in out["rearm"])
                           and [x["n_reentry"] for x in out["rearm"]] == sorted(x["n_reentry"] for x in out["rearm"]))
        return out

    def lim_bench(self, t: int, tests=(1, 2, 3, 4, 5, 6, 8), arg_len: int = 0, iters: int = 0, flags: int = 0) -> dict:
        """T-3.E2m: lim_bench.bin by LOAD/FENCE_I/CALL with policy = test id (lim_bench.c)."""
        out = {}
        with self.batch(t):
            out["load"] = self.load_code(t, _read_artefact(PAYLOAD_DIR / "lim_bench.bin"))
            for test in tests:
                r = self.call(t, regs.WEIGHTS_BASE, policy=test, flags=flags, arg_len=arg_len, tick=iters,
                              deadline_s=10.0)
                if r.get("l39"):
                    r["result"] = ml.unpack_lb_result(bytes.fromhex(r["l39"]["data"]))
                out[str(test)] = r
        return out

    def bench(self, t: int, policy: int = 0, iters: int = 1000, out_addr: int = 0, seed: int = 0,
              variant: Optional[str] = None) -> dict:
        """T-3.E4: BENCH (repeated INFER timed by mcycle on the hart; optional per-iteration array in LIM)."""
        mb = self.mb[t]
        v = self.variant_for(t, variant)
        if out_addr and not (out_addr >= max(b["end"] for b in self.blobs.values())
                             and out_addr + 4 * iters <= STAGE0_ENTRY_WORD):
            raise X280Error(f"BENCH array 0x{out_addr:x}+{4 * iters} must sit between the blob end and 0x081D_FF80")
        if out_addr and self._mh_payload.get(t) and out_addr + 4 * iters > ml.MHB_BASE:
            raise X280Error(f"BENCH array 0x{out_addr:x}+{4 * iters} runs into the mh_bench payload at 0x{ml.MHB_BASE:x}")
        obs = np.random.default_rng(seed).standard_normal(ml.OBS_DIM).astype(np.float32)
        with self.batch(t):
            # review F9: the deadline comes from a measured pilot, not from a guessed 50 ms per iteration
            pilot = None
            deadline = BENCH_PILOT_DEADLINE_S
            if iters > BENCH_PILOT_ITERS:
                t0 = time.perf_counter()
                self._bench_cmd(t, BENCH_PILOT_DEADLINE_S, policy=policy, flags=ml.variant_flags(v),
                                arg_len=BENCH_PILOT_ITERS, arg_addr=0, obs=obs)
                per_iter = (time.perf_counter() - t0) / BENCH_PILOT_ITERS
                deadline = max(10.0, BENCH_MARGIN * per_iter * iters + 1.0)
                pilot = {"iters": BENCH_PILOT_ITERS, "per_iter_ms": round(per_iter * 1e3, 4),
                         "deadline_s": round(deadline, 3)}
            r = self._bench_cmd(t, deadline, policy=policy, flags=ml.variant_flags(v), arg_len=iters,
                                arg_addr=out_addr, obs=obs)
            b = mb.l39()
            cyc = None
            if out_addr:
                cyc = [mb.r32(out_addr + 4 * i) for i in range(iters)]
        mhz = self.clock_mhz
        res = {**b, "seq": r["seq"], "variant": VARIANT_NAME[v], "dtype": self.blobs[t]["dtype"],
               "policy": POLICIES[policy], "pilot": pilot}  # names over the L39 record's numeric variant/policy
        if mhz:
            res.update(us_min=b["cyc_min"] / mhz, us_max=b["cyc_max"] / mhz, us_mean=b["cyc_sum"] / max(1, b["iters"]) / mhz)
        if cyc:
            a = np.array(cyc, np.float64) / (mhz or 1.0)
            res.update(us_p50=float(np.percentile(a, 50)), us_p99=float(np.percentile(a, 99)), unit="us" if mhz else "cycles")
        if self.sim:
            res["note"] = "sim: the x86 fake mcycle advances 16 per read, cycle numbers are meaningless"
        return res

    def _bench_cmd(self, t: int, deadline_s: float, **kw) -> dict:
        """One BENCH; on a host timeout the hart is quiesced (finish or RNMI park) before anything else is sent."""
        try:
            return self.mb[t].command(ml.CMD["BENCH"], deadline_s, **kw)
        except MailboxTimeout as e:
            q = self.quiesce(t, wait_s=deadline_s)
            raise X280Error(f"{e} (quiesced: {q})") from e

    def bench_matrix(self, t: int, iters: int = 1000, policies=(0, 1)) -> dict:
        """T-3.E4 matrix: every built variant on its weight dtype (v0/v1 fp32, v2/v2s/v3 fp16), per policy, with the
        per-iteration mcycle array in the spare lines after the blob; the session blob is LOADed again at the end.
        The scalar parity references v0/v2s run at most BENCH_SCALAR_ITERS_MAX iterations (13-45 ms each, R33)."""
        built = self.monitor.get(t, {}).get("l36", {}).get("variants_built", 0x1F)
        out = {"iters": iters, "iters_scalar": min(iters, BENCH_SCALAR_ITERS_MAX), "runs": []}
        plan = [("fp16", (2, 3, 4)), ("fp32_balance", (0, 1)), ("fp32_walk", (0, 1))]
        try:
            for which, variants in plan:
                load = self.use_blob(t, which)
                for v in variants:
                    if not (built >> v) & 1:
                        continue
                    for p in policies:
                        if not (self.blobs[t]["policy_mask"] >> p) & 1:
                            continue
                        n = min(iters, BENCH_SCALAR_ITERS_MAX) if v in (0, 3) else iters
                        r = self.bench(t, p, n, out_addr=HOST_BENCH_BASE, variant=VARIANT_NAME[v])
                        out["runs"].append({"blob": which, "load_s": load["secs"], **{k: r.get(k) for k in (
                            "variant", "dtype", "policy", "iters", "cyc_min", "cyc_max", "cyc_sum", "nondet", "rc",
                            "us_min", "us_max", "us_mean", "us_p50", "us_p99", "pilot")}})
        finally:
            self.use_blob(t, "session")
        return out

    def bench_host_only(self, t: int, n: int = 2000, polls: int = 3) -> dict:
        """S1c `bench --host-only`: the per-tick host path (33 WC obs lines + one WC L0-sized line + sfence + one UC
        write, then UC polls) on initialised spare lines after the blob, with no command reaching any hart."""
        k = self.k
        base = HOST_BENCH_BASE
        assert base >= max(b["end"] for b in self.blobs.values()) and base + 35 * 64 < 0x081D_FF80
        obs = np.random.default_rng(1).standard_normal(ml.OBS_DIM).astype(np.float32)
        ob = bytearray(ml.pack_obs(obs))
        l0 = bytearray(ml.pack_l0(ml.CMD["NOP"], 0))
        wr, rd = np.zeros(n), np.zeros(n)
        with self.batch(t):
            if not k.lim[t].is_init(base, 35 * 64):
                k.wc_write_lines(base, bytearray(35 * 64), t)
            for i in range(n):
                t0 = time.perf_counter_ns()
                k.wc_write_lines(base, ob, t)
                k.wc_write_lines(base + 33 * 64, l0, t)
                k.uc_write32(base + 34 * 64 + 60, i, t)
                t1 = time.perf_counter_ns()
                for _ in range(polls):
                    k.read_lim32(base + 34 * 64 + 60, t)
                t2 = time.perf_counter_ns()
                wr[i], rd[i] = (t1 - t0) / 1e3, (t2 - t1) / 1e3
        tot = wr + rd

        def q(a):
            return {"p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}
        return {"n": n, "polls": polls, "base": hex(base), "write_us": q(wr), "poll_us": q(rd), "total_us": q(tot),
                "per_read_us_p50": float(np.percentile(rd, 50)) / polls}

    def parity(self, obs: np.ndarray, policies: np.ndarray, ort: Optional[np.ndarray] = None,
               variant: Optional[str] = None) -> dict:
        """T-3.E5a: every obs through the hart vs the x86 build of the same blob (bit-identical for X86_EXACT) and vs the
        logged ORT action when given."""
        n = len(obs)
        v = {t: self.variant_for(t, variant) for t in self.tiles}
        dev = np.full((n, 15), np.nan, np.float32)
        x86 = np.full((n, 15), np.nan, np.float32)
        served = np.zeros(n, bool)
        ok = {t: set(self.served_policies(t)) for t in self.tiles}
        with self.all_batches():  # one interlock read for the whole run, not one per row
            for i in range(n):
                p = int(policies[i])
                t = self.policy_tile.get(p)
                if t is None or p not in ok[t]:
                    continue
                dev[i], _, _ = self.mb[t].infer(p, v[t], obs[i], i, CMD_DEADLINE_S, self.first_poll_s)
                ref = self.ref[t](p, v[t], obs[i])
                if ref is not None:
                    x86[i] = ref
                served[i] = True
        dt = {self.blobs[t]["dtype"] for t in self.tiles}.pop()
        vv = {v[t] for t in self.tiles}.pop()
        out = {"n": int(n), "n_served": int(served.sum()), "variant": VARIANT_NAME[vv], "dtype": dt, "per_policy": {}}
        for p in (0, 1):
            m = served & (policies == p)
            if not m.any():
                continue
            d = {"n": int(m.sum()),
                 "max_abs_vs_x86": float(np.nanmax(np.abs(dev[m].astype(np.float64) - x86[m]))),
                 "bits_differ_vs_x86": int((dev[m].view(np.uint32) != x86[m].view(np.uint32)).sum())}
            if ort is not None:
                e = np.max(np.abs(dev[m].astype(np.float64) - ort[m]), axis=1)
                d.update(max_abs_vs_ort=float(e.max()), p999_abs_vs_ort=float(np.percentile(e, 99.9)))
            out["per_policy"][POLICIES[p]] = d
        exact = vv in X86_EXACT
        out["pass"] = bool(out["per_policy"] and all(
            (x["bits_differ_vs_x86"] == 0 if exact else x["max_abs_vs_x86"] <= X86_TOL)
            and x.get("max_abs_vs_ort", 0.0) <= PARITY_TOL[dt] for x in out["per_policy"].values()))
        out["_arrays"] = {"act_dev": dev, "act_x86": x86, "served": served}
        return out

    def heartbeat_rate(self, t: int, secs: float = 0.2, sample_us: float = 50.0) -> dict:
        """T-3.E2 'heartbeat advances at >= 1 kHz', and mtime against wall time (L36.mtime_hz, 50 MHz expected)."""
        mb = self.mb[t]
        with self.batch(t):
            v0 = last = mb.heartbeat()
            t0 = time.perf_counter()
            changes = 0
            while time.perf_counter() - t0 < secs:
                v = mb.heartbeat()
                changes += v != last
                last = v
                t_next = time.perf_counter() + sample_us * 1e-6
                while time.perf_counter() < t_next:
                    pass
            dt = time.perf_counter() - t0
            l36 = mb.l36()
        rate = changes / dt
        mtime_hz = (last - v0) / dt
        return {"changes_per_s": round(rate, 1), "mtime_ticks_per_s": round(mtime_hz), "l36_mtime_hz": l36["mtime_hz"],
                "sample_us": sample_us, "pass": bool(rate >= 1000.0 and (self.sim or abs(mtime_hz / l36["mtime_hz"] - 1) < 0.05)),
                "note": "sim: the fake mtime advances per read, not with wall time" if self.sim else ""}

    def powertest(self, t: int, fd_churn: int = 10, watch_s: float = 0.0, gap_s: float = 0.5,
                  sample_us: float = 200.0) -> dict:
        """T-3.E3: heartbeat continuity across legacy fd open/close (each sends a power message, chardev.c:954-1017)
        and/or a passive watch (server_g1 start/stop runs meanwhile), with PLL4 read after each event and at the end
        of the watch (jobs/s2_e2_e3.sh repeats the watch across server_g1 start -> stop -> start, so every server_g1
        event is followed by a PLL4 read within one watch window)."""
        mb, k = self.mb[t], self.k
        hart = self.mode in ("booted", "attached")  # S1c (no release): fd churn and PLL4 only
        events, gaps, pll_after = [], [], []
        t_end_watch = time.perf_counter() + watch_s
        # Hart silence is measured only across consecutive samples that saw the SAME heartbeat value (first to last
        # sight), so a stall of this sampling thread is never counted as a hart gap; such stalls are reported apart.
        st = {"hb": None, "first": 0.0, "last": 0.0, "max_gap": 0.0, "prev": None, "max_interval": 0.0, "stalls": 0}
        with self.batch(t):
            def sample(until: float):
                while time.perf_counter() < until:
                    v = mb.heartbeat() if hart else 0
                    now = time.perf_counter()
                    if st["prev"] is not None:
                        iv = now - st["prev"]
                        st["max_interval"] = max(st["max_interval"], iv)
                        st["stalls"] += iv > 0.005
                    st["prev"] = now
                    if v != st["hb"]:
                        st["max_gap"] = max(st["max_gap"], st["last"] - st["first"])
                        st["hb"], st["first"] = v, now
                    st["last"] = now
                    c0 = k.read32(regs.ARC_XY, regs.PLL4_CNTL_0)
                    if c0 != 0x12 and (not events or events[-1][1] != c0):
                        events.append((round(now, 6), c0))
                    t_next = now + sample_us * 1e-6
                    while time.perf_counter() < t_next:
                        pass
            for i in range(fd_churn):
                sample(time.perf_counter() + gap_s / 2)
                fd = k.backend.open(power_aware=False)  # legacy: power message, PLL4 relock (R27)
                sample(time.perf_counter() + gap_s / 2)
                k.backend.close(fd)
                gaps.append(round(max(st["max_gap"], st["last"] - st["first"]) * 1e3, 3))
                pll_after.append(k.pll4_is_default(k.read_pll4()))
            if watch_s:
                sample(t_end_watch)
                pll_after.append(k.pll4_is_default(k.read_pll4()))
        max_gap = max(st["max_gap"], st["last"] - st["first"])
        return {"fd_churn": fd_churn, "watch_s": watch_s, "hart": hart,
                "max_heartbeat_gap_ms": round(max_gap * 1e3, 3) if hart else None,
                "gap_ms_after_each_event": gaps if hart else None, "pll4_default_after_each": pll_after,
                "max_sample_interval_ms": round(st["max_interval"] * 1e3, 3), "sampler_stalls_over_5ms": st["stalls"],
                "pll4_cntl0_changes": events[:64],
                "pass": bool((max_gap < 0.010 or not hart) and all(pll_after)),
                "inconclusive": bool(hart and st["max_interval"] > 0.010)}

    # -- teardown --
    def close(self, keep_chip: bool = False) -> None:
        """Extra cleanup fds, then the keeper fd (the KMD then writes 'GONE' into L38). keep_chip: leave the sim
        chip running, as a real tile keeps running after x280d dies (T-3.8)."""
        for fd in list(self.extra_fds.values()):
            try:
                self.k.close_test_fd(fd)
            except OSError:
                pass
        self.extra_fds.clear()
        try:
            self.k.close()
        finally:
            if self.chip is not None and not keep_chip:
                time.sleep(0.003)  # let the sim hart see 'GONE' before it is stopped (cosmetic)
                self.chip.shutdown()
            if self._own_ledger_dir is not None:
                shutil.rmtree(self._own_ledger_dir, ignore_errors=True)
        self.mode = "closed"

    def _marks(self, which: int) -> dict:
        out: dict = {}
        for st in self.steps:
            out.setdefault(st[which], []).append(st[0])
        return out

    def oplog_lines(self) -> list:
        marks = self._marks(1)
        out = []
        for i, o in enumerate(self.k.oplog):
            out += [f"STEP {lbl}" for lbl in marks.pop(i, [])]
            out.append(f"{i:06d} {o}")
        return out + [f"STEP {lbl}" for v in marks.values() for lbl in v]

    def txn_lines(self) -> list:
        marks = self._marks(2)
        out = []
        for t in self.k.txn_log():
            out += [f"STEP {lbl}" for lbl in marks.pop(t["seq"], [])]
            noc = f"0x{t['noc']:016x}" if t["noc"] is not None else "-"
            xy = f"({t['xy'][0]},{t['xy'][1]})" if t["xy"] else "-"
            out.append(f"{t['seq']:07d} {t['op']:<7} {xy} {noc} {t['size']}B a64={int(t['align64'])} "
                       f"a4={int(t['align4'])} {t.get('kind') or '-'}" + (f" = 0x{t['value']:08x}" if t["op"] in
                                                                         ("UC_W32", "UC_R32") else ""))
        return out + [f"STEP {lbl}" for v in marks.values() for lbl in v]


# --- out-of-band status for the HUD chip panel (PLAN §4.3 item 7) ---------------------------------------------

STATUS_PUB_URL = "tcp://127.0.0.1:5573"  # x280d's slot in the shared contract (server_g1: :5574)
STATUS_HZ = 5.0  # contract: 2-10 Hz
STATUS_ECC_S = 1.0  # CCACHE ECC counters at most once a second (a COUNT read also clears its ECC interrupt signal)
STATE_NAME = {v: k.lower() for k, v in ml.STATE.items()}
# The status stream's telemetry_source (verify:all D4): the device's CMFW table (read-only CSM windows), or nothing at
# all on the sim, whose FakeDevice has no chip behind it: its stream carries telemetry None, never invented values.
TELEMETRY_SOURCE_DEVICE = "cmfw"
TELEMETRY_SOURCE_SIM = "none (sim)"
TELEMETRY_SOURCE_TEST = "cmfw (FakeDevice table, unit test)"


class StatusPub:
    """zmq PUB of [b"status", wire.dumps(msg)] at STATUS_HZ, SNDHWM 10, NOBLOCK sends: a slow or absent subscriber
    loses messages, nothing ever blocks. The publishing thread never touches kmd, the mailbox or any request-path
    state: the serving thread hands it a finished snapshot by one reference swap under `_lock`, and the publisher
    holds that lock only for the reference copy. url '0' binds a random port (tests); `endpoint` is the bound URL."""

    def __init__(self, ctx: zmq.Context, url: str, hz: float = STATUS_HZ):
        self.hz = min(10.0, max(2.0, float(hz)))
        self.sock = ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind("tcp://127.0.0.1:*" if str(url) == "0" else url)  # a taken port fails here, before any release
        self.endpoint = self.sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self.seq = self.sent = self.dropped = 0
        self._snap: Optional[dict] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="x280d-status-pub", daemon=True)
        self._thread.start()

    def update(self, snap: dict) -> None:
        with self._lock:
            self._snap = snap

    def _run(self) -> None:
        period = 1.0 / self.hz
        while not self._stop.wait(period):
            with self._lock:
                snap = self._snap
            if snap is None:
                continue
            msg = dict(snap)
            now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            msg.update(seq=self.seq, t_ns=now_ns, uptime_s=round(time.monotonic() - T_START, 3),
                       snapshot_age_ms=round((now_ns - snap["_t_snap_ns"]) / 1e6, 3))
            msg.pop("_t_snap_ns", None)
            self.seq += 1
            try:
                self.sock.send_multipart([b"status", wire.dumps(msg)], zmq.NOBLOCK)
                self.sent += 1
            except zmq.Again:
                self.dropped += 1

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2.0)
        self.sock.close(0)


# --- W-WBC server ----------------------------------------------------------------------------------------------


class Server:
    """REP 127.0.0.1:PORT: ping, info, gear_wbc_action (+ gear_wbc_dispatch/collect with --early) and x280d_admin.
    Single-threaded: kmd is touched only here, between requests (health) or inside one."""

    def __init__(self, sess: Session, port: int, *, early: bool = False, slow: bool = False,
                 deadline_ms: Optional[float] = None, status_pub: Optional[str] = None, status_hz: float = STATUS_HZ,
                 log_dir: Optional[Path] = None, admin: bool = True, fake_telemetry: bool = False, spin: bool = False):
        """fake_telemetry: unit tests only (no CLI flag sets it): a sim session reads its FakeDevice's table as if it
        were CMFW's, with values the test poked itself; otherwise the sim publishes no telemetry.
        spin: busy-poll the REP socket (yielding on every miss) instead of blocking in poll, so a request is picked up
        without the sleeping thread's wake-up (T-3.5 2026-10-25: x280d p99 0.63 ms but client RTT p99 3.25 ms)."""
        self.s, self.port, self.early, self.admin_on = sess, port, early, admin
        self.spin = spin
        self.telemetry_source = (TELEMETRY_SOURCE_DEVICE if not sess.sim else
                                 TELEMETRY_SOURCE_TEST if fake_telemetry else TELEMETRY_SOURCE_SIM)
        self.deadline_s = (deadline_ms or (DEADLINE_SLOW_MS if slow else DEADLINE_MS)) / 1e3
        self.log_dir = log_dir
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://127.0.0.1:{port}")
        self.pub = StatusPub(self.ctx, status_pub, status_hz) if status_pub else None
        self.t_status = 0.0
        self.t_ecc, self.ecc_last = 0.0, None
        self.stop = False
        self.degraded: Optional[str] = None
        self.batches: Optional[ExitStack] = None
        self.hb = {t: (None, time.perf_counter()) for t in sess.tiles}
        self.counts = {k: 0 for k in ("calls", "ok", "timeouts", "faults", "stalls", "interlock", "bad", "degraded",
                                      "early_hit", "early_miss", "admin")}
        self.ring = np.zeros(RING, TICK_LOG_DTYPE)
        self.n_ring = 0
        self.pending = None  # early dispatch: (tile, policy, mb_seq, tick, seq, t_issue)
        self.inject = {"crc": 0}
        self.t_health = self.t_slow = 0.0
        self.t_last_req = self.t_last_gc = time.perf_counter()
        self.n_gc = 0
        self.status_log = (log_dir / "x280d_status.jsonl") if log_dir else None

    # -- interlock batches held while serving; re-entered every HEALTH_S (re-reads PLL4) --
    def _enter_batches(self) -> None:
        if self.batches is not None:
            self.batches.close()
            self.batches = None
        try:
            self.batches = self.s.all_batches()
            if self.degraded and self.degraded.startswith("interlock"):
                log(f"interlocks OK again: leaving degraded ({self.degraded})")
                self.degraded = None
        except (K.InterlockError, K.KmdError) as e:
            self.degraded = f"interlock: {e}"
            log(f"DEGRADED: {self.degraded}")

    def health(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.t_health < HEALTH_S:
            return
        self.t_health = now
        if not gc.isenabled() and now - self.t_last_req > GC_IDLE_S and now - self.t_last_gc > GC_EVERY_S:
            gc.collect()  # a full pass walks every live op record: only while no client is waiting (e.g. a reset)
            self.t_last_gc, self.n_gc = time.perf_counter(), self.n_gc + 1
        self._enter_batches()
        if self.s.mode not in ("booted", "attached") or self.degraded:
            return
        for t in self.s.tiles:
            try:
                self._hb_check(t)
            except K.KmdError as e:
                self.degraded = f"interlock: {e}"
        if now - self.t_slow >= SLOW_HEALTH_S or force:
            self.t_slow = now
            st = self.status()
            if self.status_log:
                with open(self.status_log, "a") as f:
                    f.write(json.dumps(st, default=_jsonable) + "\n")

    def status_tick(self, force: bool = False) -> None:
        """Refresh the chip-panel snapshot for StatusPub (serving thread, between requests: it runs right after a
        reply went out or on an idle poll, ~50 UC reads at STATUS_HZ, never inside a W-WBC call)."""
        if self.pub is None:
            return
        now = time.perf_counter()
        if not force and now - self.t_status < 1.0 / self.pub.hz:
            return
        self.t_status = now
        try:
            snap = self.status_snapshot()
        except Exception as e:  # noqa: BLE001 - the HUD stream must never take the daemon down
            snap = self._status_common()
            snap["error"] = f"{type(e).__name__}: {e}"
        self.pub.update(snap)

    def _status_common(self) -> dict:
        s = self.s
        return {"src": "x280d", "pid": os.getpid(), "backend": self.backend_stamp(), "_t_snap_ns":
                time.clock_gettime_ns(time.CLOCK_MONOTONIC), "mode": s.mode, "degraded": self.degraded,
                "telemetry_source": self.telemetry_source}

    def status_snapshot(self) -> dict:
        """The x280d half of the HUD chip-panel contract (PLAN §4.3 item 7). Unknown -> None, never invented:
        monitor fields need a live hart and an open interlock batch, telemetry needs the CMFW table of a real chip
        (telemetry_source says which; the sim's is 'none (sim)' and its telemetry always None)."""
        s = self.s
        t = s.tiles[0]
        out = self._status_common()
        out.update({"tile": t, "tile_state": None, "monitor_state": None, "heartbeat_age_us": None, "seq_done": None,
                    "cycles_last": None, "cycles_max": None, "compute_us_last": None, "duty": None, "n_runs": None,
                    "n_overrun": None, "n_reentry": None, "beu_accrued": None,
                    "ccache_ecc": {"fix": None, "fail": None}, "variant": VARIANT_NAME[s.variant], "dtype": s.dtype,
                    "fw_ver": None, "build_id": None, "telemetry": None})
        try:
            out["tile_state"] = s.ledger.state(t)
        except (ledger_mod.LedgerError, OSError, ValueError):
            pass
        ok = self._recent_ok(50)
        if len(ok):
            out["compute_us_last"] = float(ok["compute_us"][-1])
            out["duty"] = float(np.mean(ok["compute_us"]) / 20_000.0)  # compute us per 20 ms tick, last 50 calls
        if s.k.fd is None:
            return out
        if self.telemetry_source != TELEMETRY_SOURCE_SIM:
            try:
                out["telemetry"] = s.k.read_status_telemetry()
            except K.KmdError as e:
                out["telemetry_error"] = str(e)
        if s.mode not in ("booted", "attached") or self.batches is None or self.degraded:
            return out
        mb = s.mb[t]
        l1, l36 = mb.l1(), mb.l36()
        out.update(monitor_state=STATE_NAME.get(l1["state"], l1["state"]), seq_done=l1["seq_done"],
                   cycles_last=l1["cycles_last"], cycles_max=l1["cycles_max"], fw_ver=l1["fw_ver"],
                   n_runs=l36["n_runs"], n_overrun=l36["n_overrun"], n_reentry=l36["n_reentry"],
                   beu_accrued=l36["beu_accrued"], build_id=l36["build_id"],
                   heartbeat_age_us=round(self._hb_check(t) * 1e3, 1))
        now = time.perf_counter()
        if self.ecc_last is None or now - self.t_ecc >= STATUS_ECC_S:
            e = s.k.read_ccache_ecc_counts(t)
            self.ecc_last, self.t_ecc = e, now
        out["ccache_ecc"] = dict(self.ecc_last)
        if len(s.tiles) > 1:  # DP4: the second tile's monitor too (the top level is tiles[0])
            out["tiles"] = {}
            for u in s.tiles[1:]:
                a, b = s.mb[u].l1(), s.mb[u].l36()
                out["tiles"][str(u)] = {"monitor_state": STATE_NAME.get(a["state"], a["state"]), "seq_done": a["seq_done"],
                                        "cycles_last": a["cycles_last"], "cycles_max": a["cycles_max"],
                                        "n_runs": b["n_runs"], "n_overrun": b["n_overrun"], "n_reentry": b["n_reentry"],
                                        "beu_accrued": b["beu_accrued"]}
        return out

    def _recent_ok(self, n: int) -> np.ndarray:
        """The last n ST_OK rows of the tick ring (field copies of the serving thread's own ring)."""
        k = min(self.n_ring, RING)
        if not k:
            return self.ring[:0]
        idx = (np.arange(self.n_ring - k, self.n_ring) % RING)
        rows = self.ring[idx]
        return rows[rows["status"] == ST_OK][-n:]

    def _hb_check(self, t: int) -> float:
        """Heartbeat age in ms: time since the heartbeat value last changed, as seen by this host."""
        v = self.s.mb[t].heartbeat()
        last, t_last = self.hb[t]
        now = time.perf_counter()
        if v != last:
            self.hb[t] = (v, now)
            return 0.0
        return (now - t_last) * 1e3

    def status(self) -> dict:
        s = self.s
        out = {"t": time.time(), "mode": s.mode, "degraded": self.degraded, "counts": dict(self.counts),
               "gc_runs": self.n_gc,
               "backend": self.backend_stamp(), "tiles": {}}
        if self.batches is None or s.mode not in ("booted", "attached"):
            return out
        for t in s.tiles:
            mb = s.mb[t]
            try:
                l1, l36 = mb.l1(), mb.l36()
                out["tiles"][str(t)] = {"state": l1["state"], "heartbeat": l1["heartbeat"],
                                        "cycles_last": l1["cycles_last"], "cycles_max": l1["cycles_max"],
                                        **{k: l36[k] for k in ("n_runs", "n_overrun", "n_reentry", "beu_accrued",
                                                               "n_traps", "n_crc_fail", "n_crc_retry", "n_host_gone",
                                                               "kernel_variant", "build_id")},
                                        "ledger": s.ledger.state(t), "l1": l1, "l36": l36}
                out["tiles"][str(t)]["pll4_default"] = s.k.pll4_is_default(s.k.read_pll4())
                if s.harts == regs.MH_HARTS:  # --harts 4 only: harts 1-3's stat lines (a default status is unchanged)
                    out["tiles"][str(t)]["mh"] = s.mh_status(t, hart_status=False)
            except (K.KmdError, ledger_mod.LedgerError, X280Error) as e:
                out["tiles"][str(t)] = {"error": str(e)}
        if s.chip is not None:
            out["sim"] = s.chip.status()
        return out

    def backend_stamp(self) -> str:
        s = self.s
        v = f"{VARIANT_NAME[s.variant]}-{s.dtype}"
        if s.sim:
            return f"x280-sim:{v}"
        tiles = "+".join(str(t) for t in s.tiles)
        return f"x280:l2cpu{tiles}:h0@{int(round(s.clock_mhz or 800))}MHz:{v}"

    def info(self) -> dict:
        s = self.s
        served = sorted({POLICIES[p] for t in s.tiles for p in s.served_policies(t)}) if s.mode in ("booted", "attached") else []
        return {"backend": self.backend_stamp(), "proto": 2, "policies": served, "mode": s.mode,
                "tiles": list(s.tiles), "policy_tile": {POLICIES[p]: t for p, t in s.policy_tile.items()},
                "dtype": s.dtype, "variant": VARIANT_NAME[s.variant], "parity_tol": PARITY_TOL[s.dtype],
                "blobs": {str(t): {k: b[k] for k in ("path", "len", "crc32", "end")} for t, b in s.blobs.items()},
                "image": {"path": str(IMAGE_BIN), "len": len(s.image), "crc32": s.image_crc},
                "monitor": {str(t): {"build_id": m["l36"]["build_id"], "fw_ver": m["l1"]["fw_ver"],
                                     "vlenb": m["l1"]["vlenb"]} for t, m in s.monitor.items()},
                "clock_mhz": s.clock_mhz, "compute_us_src": "host_turnaround" if s.sim else "mcycle",
                "deadline_ms": self.deadline_s * 1e3, "early": self.early, "spin": self.spin, "boot_parity": s.boot_parity,
                "degraded": self.degraded, "hold_id": s.hold_id, "pid": os.getpid(),
                "sim_shared_cpu": s.sim_shared_cpu if s.sim else None,
                "status_pub": self.pub.endpoint if self.pub is not None else None,
                "telemetry_source": self.telemetry_source}

    # -- W-WBC --
    def _validate(self, policy, obs) -> tuple:
        if not (isinstance(obs, np.ndarray) and obs.shape == (1, ml.OBS_DIM) and obs.dtype == np.float32):
            raise ValueError(f"obs must be float32 (1, {ml.OBS_DIM}), got {getattr(obs, 'dtype', type(obs))} "
                             f"{getattr(obs, 'shape', None)}")
        if policy not in POLICIES:
            raise KeyError(f"unknown policy {policy!r}")
        p = POLICIES.index(policy)
        t = self.s.policy_tile.get(p)
        if t is None or p not in self.s.served_policies(t):
            raise KeyError(f"policy {policy!r} is not served by this session")
        return p, t

    def _ready_or_error(self) -> Optional[str]:
        if self.s.mode not in ("booted", "attached"):
            return f"no hart serving (mode {self.s.mode})"
        if self.degraded:
            return f"degraded: {self.degraded}"
        if self.batches is None:
            return "interlock batch not open"
        return None

    def _issue(self, t: int, p: int, obs: np.ndarray, tick: int) -> int:
        age = self._hb_check(t)
        if age > HB_STALL_MS:
            st = self.s.mb[t].r32(ml.ADDR["L1"] + ml.OFFSETS["l1.state"])
            if st != ml.STATE["BUSY"]:
                raise _Stall(f"x280 heartbeat stalled {age:.1f} ms (state {st})", age)
        delta = 0
        if self.inject["crc"]:
            self.inject["crc"] -= 1
            delta = 1
        return self.s.mb[t].issue(ml.CMD["INFER"], policy=p, flags=ml.variant_flags(self.s.variant), tick=tick,
                                  obs=obs, obs_crc_delta=delta)

    def _complete(self, t: int, mb_seq: int, t_issue: float) -> tuple:
        mb = self.s.mb[t]
        st = mb.wait(mb_seq, self.deadline_s, self.s.first_poll_s)
        t_done = time.perf_counter()
        if st != ml.STATE["IDLE"]:
            l1 = mb.l1()
            raise MailboxFault(f"INFER seq {mb_seq} -> state {st} mcause 0x{l1['mcause']:x} mtval 0x{l1['mtval']:x}", l1)
        act, cyc = mb.action(mb_seq)
        act = act.reshape(1, 15).copy()
        turnaround_us = (t_done - t_issue) * 1e6
        compute_us = turnaround_us if self.s.sim or not self.s.clock_mhz else cyc / self.s.clock_mhz
        return act, cyc, compute_us

    def _record(self, t_recv: int, tick, seq, mb_seq, t, p, status, early, cyc=0, compute_us=np.nan, mb_us=np.nan,
                hb_age=np.nan) -> float:
        x_us = (time.perf_counter_ns() - t_recv) / 1e3
        r = self.ring[self.n_ring % RING]
        r["t_recv_ns"], r["tick"], r["seq"], r["mb_seq"] = t_recv, tick or 0, seq or 0, mb_seq or 0
        r["tile"], r["policy"], r["status"], r["early"] = t, p, status, early
        r["x280d_us"], r["mb_us"], r["compute_us"], r["cycles"], r["hb_age_us"] = x_us, mb_us, compute_us, cyc, hb_age
        self.n_ring += 1
        return x_us

    def gear_wbc_action(self, policy, obs, tick=None, seq=None, **_) -> dict:
        t_recv = time.perf_counter_ns()
        self.counts["calls"] += 1
        p, t = self._validate(policy, obs)
        early = EARLY_NONE
        if self.pending is not None:  # a dispatch nobody collected: the client chose another path (miss)
            self.counts["early_miss"] += 1
            early = EARLY_MISS
            self.pending = None
        why = self._ready_or_error()
        if why:
            self.counts["degraded"] += 1
            self._record(t_recv, tick, seq, 0, t, p, ST_DEGRADED, early)
            return {"error": why}
        t_issue = time.perf_counter()
        try:
            mb_seq = self._issue(t, p, obs, int(tick or 0))
            act, cyc, compute_us = self._complete(t, mb_seq, t_issue)
        except _Stall as e:
            self.counts["stalls"] += 1
            self._record(t_recv, tick, seq, 0, t, p, ST_STALL, early, hb_age=e.age * 1e3)
            return {"error": str(e)}
        except MailboxTimeout as e:
            self.counts["timeouts"] += 1
            self._record(t_recv, tick, seq, self.s.mb[t].seq, t, p, ST_TIMEOUT, early)
            return {"error": f"x280 timeout: {e}"}
        except MailboxFault as e:
            self.counts["faults"] += 1
            self._record(t_recv, tick, seq, self.s.mb[t].seq, t, p, ST_FAULT, early)
            return {"error": f"x280 fault: {e}"}
        except K.KmdError as e:
            self.counts["interlock"] += 1
            self._record(t_recv, tick, seq, 0, t, p, ST_INTERLOCK, early)
            return {"error": f"x280d: {type(e).__name__}: {e}"}
        mb_us = (time.perf_counter() - t_issue) * 1e6
        self.counts["ok"] += 1
        x_us = self._record(t_recv, tick, seq, mb_seq, t, p, ST_OK, early, cyc, compute_us, mb_us)
        return {"action": act, "tick": tick, "seq": seq, "compute_us": float(compute_us), "x280d_us": x_us,
                "mailbox_us": mb_us, "cycles": int(cyc), "mb_seq": mb_seq}

    def gear_wbc_dispatch(self, policy, obs, tick=None, seq=None, **_) -> dict:
        """--early (WP-3.8b): issue the INFER now, reply at once; gear_wbc_collect returns the W-WBC reply."""
        if not self.early:
            return {"error": "early dispatch disabled (serve --early)"}
        p, t = self._validate(policy, obs)
        why = self._ready_or_error()
        if why:
            return {"error": why}
        if self.pending is not None:
            self.counts["early_miss"] += 1
        t_issue = time.perf_counter()
        try:
            mb_seq = self._issue(t, p, obs, int(tick or 0))
        except (_Stall, K.KmdError) as e:
            self.pending = None
            return {"error": str(e)}
        self.pending = (t, p, mb_seq, tick, seq, t_issue, time.perf_counter_ns())
        return {"dispatched": True, "tick": tick, "seq": seq, "mb_seq": mb_seq}

    def gear_wbc_collect(self, tick=None, seq=None, **_) -> dict:
        if self.pending is None or self.pending[3] != tick or self.pending[4] != seq:
            return {"error": f"no dispatch pending for tick={tick} seq={seq}"}
        t, p, mb_seq, _, _, t_issue, t_recv = self.pending
        self.pending = None
        try:
            act, cyc, compute_us = self._complete(t, mb_seq, t_issue)
        except (MailboxTimeout, MailboxFault, X280Error, K.KmdError) as e:
            self._record(t_recv, tick, seq, mb_seq, t, p, ST_TIMEOUT if isinstance(e, MailboxTimeout) else ST_FAULT,
                         EARLY_HIT)
            return {"error": f"x280: {e}"}
        self.counts["early_hit"] += 1
        self.counts["ok"] += 1
        x_us = self._record(t_recv, tick, seq, mb_seq, t, p, ST_OK, EARLY_HIT, cyc, compute_us,
                            (time.perf_counter() - t_issue) * 1e6)
        return {"action": act, "tick": tick, "seq": seq, "compute_us": float(compute_us), "x280d_us": x_us,
                "cycles": int(cyc), "mb_seq": mb_seq, "early": True}

    # -- admin (job steps inside a chip_session; outside the frozen W-WBC contract) --
    def admin(self, cmd: str, **kw) -> dict:
        self.counts["admin"] += 1
        if cmd == "status":
            return self.status()
        if cmd == "info":
            return self.info()
        if cmd == "park":
            self.pending = None
        if cmd == "inject" and kw.get("what") == "crc":
            self.inject["crc"] += int(kw.get("n", 1))
            return {"armed": "crc", "n": self.inject["crc"]}
        if cmd == "dump":
            return self.dump_ring(kw.get("out")) or {"ring": None, "error": "no --log-dir and no out: nowhere to write"}
        return run_admin(self.s, cmd, **kw)

    def dump_ring(self, out: Optional[str] = None) -> Optional[dict]:
        """The tick ring, oldest row first, as x280d_ticks.npy in `out` (a directory or a .npy path) or the log dir:
        what analyze_run's x280_heartbeat_continuity reads (T-3.7). The ring keeps the last RING calls (~22 min at
        50 Hz); `wrapped` says older calls are gone."""
        p = Path(out) if out else (self.log_dir / RING_FILE if self.log_dir else None)
        if p is None:
            return None
        if p.is_dir():
            p = p / RING_FILE
        elif p.suffix != ".npy":  # np.save would append .npy and the reply would name the wrong file
            raise X280Error(f"dump: {p} is neither a directory nor a .npy path")
        n = min(self.n_ring, RING)
        rows = self.ring[:n] if self.n_ring <= RING else np.roll(self.ring, -(self.n_ring % RING))
        np.save(p, rows)
        return {"ring": str(p), "rows": int(n), "calls": int(self.n_ring), "wrapped": self.n_ring > RING,
                "t_first_ns": int(rows["t_recv_ns"][0]) if n else None,
                "t_last_ns": int(rows["t_recv_ns"][-1]) if n else None}

    def handle(self, msg: bytes) -> bytes:
        try:
            req = wire.loads(msg)
            ep = req.get("endpoint")
            data = req.get("data") or {}
            if ep == "gear_wbc_action":
                out = self.gear_wbc_action(**data)
            elif ep == "ping":
                out = {"status": "ok", "message": "x280d is running", "mode": self.s.mode}
            elif ep == "info":
                out = self.info()
            elif ep == "gear_wbc_dispatch":
                out = self.gear_wbc_dispatch(**data)
            elif ep == "gear_wbc_collect":
                out = self.gear_wbc_collect(**data)
            elif ep == "x280d_admin" and self.admin_on:
                out = self.admin(**data)
                out = json.loads(json.dumps(out, default=_jsonable))
            else:
                out = {"error": f"Unknown endpoint: {ep}"}
        except Exception as e:  # noqa: BLE001 - every failure is an error reply; the client falls back to ORT
            out = {"error": f"{type(e).__name__}: {e}"}
        return wire.dumps(out)

    def run(self, ready=None) -> None:
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        self.health(force=True)
        log(f"READY on tcp://127.0.0.1:{self.port}: {self.backend_stamp()} mode={self.s.mode} "
            f"policies={self.info()['policies']}")
        if ready:
            ready()
        gc.collect()
        gc.freeze()  # boot-time objects never need scanning again
        gc.disable()  # a gen-2 pass over kmd's op log stalls a reply by milliseconds (same reason rt_sim disables it)
        self.status_tick(force=True)
        try:
            while not self.stop:
                if self.spin:
                    msg = self._recv_spin(0.1 if self.pub is None else 0.5 / self.pub.hz)
                    if msg is not None:
                        self.sock.send(self.handle(msg))
                        self.t_last_req = time.perf_counter()
                elif poller.poll(100 if self.pub is None else int(500 / self.pub.hz)):
                    self.sock.send(self.handle(self.sock.recv()))
                    self.t_last_req = time.perf_counter()
                self.health()
                self.status_tick()  # after the reply went out: the next request is ~20 ms away
        finally:
            gc.enable()

    def _recv_spin(self, budget_s: float) -> Optional[bytes]:
        """Busy-poll for a request for up to budget_s, yielding the CPU on every miss: --cpus pins zmq's I/O thread
        to the same core, and a poll that never yields holds it off for a scheduler slice (as wbc_gpu_server --spin
        found in T-1.4: overhead p99 0.92 ms without the yield, 0.37 ms with it). None when the budget ran out."""
        t_end = time.perf_counter() + budget_s
        while not self.stop:
            try:
                return self.sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                if time.perf_counter() >= t_end:
                    return None
                os.sched_yield()
        return None

    def close(self) -> None:
        try:
            self.dump_ring()
        except OSError:
            pass
        if self.batches is not None:
            self.batches.close()
        self.sock.close(0)
        if self.pub is not None:
            self.pub.close()
        self.ctx.term()


class _Stall(X280Error):
    def __init__(self, msg: str, age_ms: float):
        super().__init__(msg)
        self.age = age_ms / 1e3


LIM_WRITERS = ("lim_init", "lim_ecc")  # rewrite LIM: only while no hart runs (S1c, S1d), never under a live monitor


def run_admin(s: Session, cmd: str, **kw) -> dict:
    """The job-step commands, shared by the serving daemon (x280d_admin) and standalone subcommands."""
    tile = int(kw.pop("tile", s.tiles[0]))
    if tile not in s.tiles:
        raise X280Error(f"tile {tile} is not in this session {s.tiles}")
    live = s.mode in ("booted", "attached")
    if cmd == "e0":
        return s.e0()
    if cmd == "tiles_probe":  # S5a: `probe --tile LIST --via URL` on a `serve --no-release --probe-tiles` daemon
        if live:
            raise X280Error(f"tiles probe (S5a): a no-release session only (mode {s.mode})")
        return s.tiles_probe(tuple(kw.get("tiles") or s.k.probe_tiles))
    if cmd in LIM_WRITERS or (cmd == "lim_check" and (kw.get("partial") or kw.get("cleanup_check"))):
        if live:
            raise X280Error(f"{cmd} writes LIM lines and L38: only in a --no-release session (mode {s.mode})")
    if cmd == "lim_init":
        return s.lim_init(tile, **kw)
    if cmd == "lim_check":
        return s.lim_check(tile, **kw)
    if cmd == "lim_ecc":
        return s.lim_ecc(tile, **kw)
    if cmd == "bench_host_only":
        return s.bench_host_only(tile, **kw)
    if cmd == "powertest" and not live:
        return s.powertest(tile, **kw)
    if not live:
        raise X280Error(f"{cmd}: no hart in this session (mode {s.mode})")
    if cmd == "status":
        out = {"mode": s.mode, "tiles": {str(t): {"l1": s.mb[t].l1(), "l36": s.mb[t].l36()} for t in s.tiles},
               "sim": s.chip.status() if s.chip is not None else None}
        if s.harts == regs.MH_HARTS:  # --harts 4 only: a default session's status is unchanged
            for t in s.tiles:
                out["tiles"][str(t)]["mh"] = s.mh_status(t)
        return out
    if cmd == "harts_adopt":
        return s.harts_adopt(tile, **kw)
    if cmd == "mh_status":
        return s.mh_status(tile)
    if cmd == "mh_ping":
        with s.batch(tile):
            return s.mh_ping(tile, **kw)
    if cmd == "bench_mh":
        return s.bench_mh(tile, **kw)
    if cmd == "selftest":
        with s.batch(tile):
            return s.selftest(tile, **kw)
    if cmd == "load_blob":
        with s.batch(tile):
            return s.load_blob(tile)
    if cmd == "load":
        data = kw.pop("data", None)
        if data is None:
            data = Path(kw.pop("file")).read_bytes()
        with s.batch(tile):
            return s.load_code(tile, bytes(data), **kw)
    if cmd == "call":
        with s.batch(tile):
            return s.call(tile, **kw)
    if cmd == "park":
        return s.park(tile, **kw)
    if cmd == "park_cooperative":
        with s.batch(tile):
            return s.mb[tile].command(ml.CMD["PARK"], ok_states=(ml.STATE["PARKED"],))
    if cmd == "e2":
        return s.e2(tile, **kw)
    if cmd == "lim_bench":
        return s.lim_bench(tile, **kw)
    if cmd == "bench":
        blob = kw.pop("blob", None)
        if blob:
            s.use_blob(tile, blob)
        try:
            return s.bench(tile, **kw)
        finally:
            if blob:
                s.use_blob(tile, "session")
    if cmd == "bench_matrix":
        return s.bench_matrix(tile, **kw)
    if cmd == "heartbeat_rate":
        return s.heartbeat_rate(tile, **kw)
    if cmd == "use_blob":
        return s.use_blob(tile, kw["which"])
    if cmd == "parity":
        d = np.load(kw.pop("obs"))
        pol_all = (np.array([POLICIES.index(str(x)) for x in d["policy"]]) if "policy" in d.files
                   else np.zeros(len(d["obs"]), int))
        blob = kw.pop("blob", None)
        if blob:
            s.use_blob(tile, blob)
        try:
            served = sorted({p for t in s.tiles for p in s.served_policies(t)})
            rows = np.where(np.isin(pol_all, served))[0]  # rows the loaded blob(s) can serve, then the first n
            n = int(kw.pop("n", 0) or 0)
            rows = rows[:n] if n else rows
            r = s.parity(d["obs"][rows], pol_all[rows], d["action"][rows] if "action" in d.files else None,
                         variant=kw.pop("variant", None))
        finally:
            if blob:
                s.use_blob(tile, "session")
        arrays = r.pop("_arrays")
        r["rows_from"] = "first n rows of the served policies"
        out = kw.pop("out", None)
        if out:
            np.savez(out, obs=d["obs"][rows], policy=pol_all[rows], rows=rows, **arrays)
            r["out"] = out
        return r
    if cmd == "powertest":
        return s.powertest(tile, **kw)
    if cmd == "inject":
        if kw.get("what") == "hang" and s.chip is not None:
            s.chip.hart(tile).inject_hang()
            return {"injected": "hang", "tile": tile}
        if kw.get("what") == "mh_hang" and s.chip is not None:  # sim only: a wedged secondary (hart `n`)
            s.chip.hart(tile).inject_secondary_hang(int(kw.get("n", 1)))
            return {"injected": "mh_hang", "tile": tile, "hart": int(kw.get("n", 1))}
        raise X280Error(f"inject {kw.get('what')!r}: crc (serving daemon) or hang (sim only)")
    raise X280Error(f"unknown admin command {cmd!r}")


def peek_cross_check(e0: dict, vla_url: str, tile: int = regs.DEFAULT_TILE, timeout_ms: int = 10_000) -> dict:
    """T-3.E0 'x280_peek agrees with kmd.py on every register': the same addresses read through server_g1's debug
    endpoint (ttnn.cluster.read_reg; rttt/server_g1.py x280_peek), compared with x280d's own reads in `e0`."""
    regs_kmd = {("arc", regs.L2CPU_RESET): e0["l2cpu_reset"], ("arc", regs.PLL4_CNTL_1): e0["pll4"]["cntl_1"],
                ("arc", regs.PLL4_CNTL_5): e0["pll4"]["cntl_5"], ("arc", regs.PLL4_USE_POSTDIV): e0["pll4"]["use_postdiv"]}
    tr = e0["tiles"][str(tile)]
    if "wayenable" in tr:
        regs_kmd[("l2cpu", regs.CCACHE0_WAYENABLE)] = tr["wayenable"]
        regs_kmd[("l2cpu", regs.CCACHE0_CONFIG)] = tr["ccache_config"]
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(vla_url)
    rows = []
    try:
        for (kind, addr), want in regs_kmd.items():
            x, y = regs.ARC_XY if kind == "arc" else regs.L2CPU_XY[tile]
            sock.send(wire.request("x280_peek", {"x": x, "y": y, "addr": addr, "n": 1}))
            if not sock.poll(timeout_ms):
                raise X280Error(f"{vla_url}: x280_peek timeout")
            rep = wire.loads(sock.recv())
            got = rep.get("values", [None])[0] if isinstance(rep, dict) and "error" not in rep else None
            rows.append({"xy": [x, y], "addr": hex(addr), "kmd": want, "peek": got, "agree": got == want,
                         **({"error": rep.get("error")} if isinstance(rep, dict) and "error" in rep else {})})
    finally:
        sock.close(0)
    return {"rows": rows, "agree": all(r["agree"] for r in rows)}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --- CLI -------------------------------------------------------------------------------------------------------


def _tiles(spec) -> tuple:
    out = []
    for s in spec or [str(regs.DEFAULT_TILE)]:
        out += [int(x) for x in str(s).split(",") if x != ""]
    return tuple(out)


def _cpu_seq(spec: str) -> list:
    """'5' or '5,21' or '0,1,8-11' (taskset -c syntax) -> [CPU ids] in the order written, each once: --sim-cpus
    gives one CPU per hart in tile order, and the CHIP_SAFE_CPUS check and the sim must read the same list (N4)."""
    out: list = []
    for part in str(spec).split(","):
        lo, _, hi = part.strip().partition("-")
        a, b = int(lo), int(hi or lo)
        if b < a:  # taskset refuses '9-8' too; an empty set would pass the safe-set check and fail at sched_setaffinity
            raise ValueError(f"empty CPU range {part!r}")
        out += [c for c in range(a, b + 1) if c not in out]
    return out


def _cpu_list(spec: str) -> set:
    """'5' or '5,21' or '0,1,8-11' (taskset -c syntax) -> {CPU ids}."""
    return set(_cpu_seq(spec))


def _outside_safe_cpus(args) -> Optional[str]:
    """With CHIP_SAFE_CPUS set (chip_lib.sh: a session confined to the non-real-time cores), --cpus (this process) and
    --sim-cpus (the spinning sim harts) must stay inside it, as rttt/server_g1.py's --cpus must (verify:all D9). A
    --cpus / --sim-cpus that is not a taskset list is refused with or without it (a clean exit 2, never a traceback
    from the parse in cmd_serve / _session)."""
    flags = {}
    for flag, spec in (("--cpus", getattr(args, "cpus", None)), ("--sim-cpus", getattr(args, "sim_cpus", None))):
        if not spec:
            continue
        try:
            flags[flag] = (spec, _cpu_list(spec))
        except ValueError:
            return f"{flag} {spec!r} is not a CPU list"
    safe = os.environ.get("CHIP_SAFE_CPUS")
    if not safe:
        return None
    try:
        allowed = _cpu_list(safe)
    except ValueError:
        return f"CHIP_SAFE_CPUS={safe!r} is not a CPU list"
    for flag, (spec, cpus) in flags.items():
        if not cpus <= allowed:
            return f"{flag} {spec} is outside CHIP_SAFE_CPUS={safe}"
    return None


def _session(args, **kw) -> Session:
    backend = "sim" if args.dry_run else args.backend
    cpus = _cpu_seq(args.sim_cpus) if getattr(args, "sim_cpus", None) else None
    return Session(backend, _tiles(args.tile), dtype=args.dtype, variant=args.variant, ledger_path=args.ledger,
                   sim_cpus=cpus, first_poll_us=args.first_poll_us, sim_ecc=getattr(args, "sim_ecc", None),
                   **({"harts": args.harts} if getattr(args, "harts", 1) != 1 else {}), **kw)


def _admin_call(url: str, cmd: str, timeout_s: float, **kw) -> dict:
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(url)
    try:
        s.send(wire.request("x280d_admin", {"cmd": cmd, **kw}))
        if not s.poll(int(timeout_s * 1000)):
            raise X280Error(f"{url}: no reply to {cmd} within {timeout_s} s")
        return wire.loads(s.recv())
    finally:
        s.close(0)


def _print(obj, as_json: bool = True) -> None:
    print(json.dumps(obj, default=_jsonable, sort_keys=True) if as_json else obj, flush=True)


def _dry_run_dump(sess: Session) -> None:
    print("## OPS (kmd op log: every ioctl, mmap, read, write, refusal)")
    for line in sess.oplog_lines():
        print(line)
    print("## TXN (libwcio transaction log: every WC line store, sfence, UC access)")
    for line in sess.txn_lines():
        print(line)
    summ = sess.k.txn_summary()
    print("## SUMMARY " + json.dumps({"txn": summ, "steps": [s[0] for s in sess.steps],
                                       "ops": len(sess.k.oplog), "fake_ioctls": sess.k.backend.names(),
                                       "fake_calls": [c[0] for c in sess.k.backend.calls]}, default=_jsonable))


def cmd_probe_tiles(args, tiles: tuple) -> int:
    """S5a: `probe --tile 3,2,1,0` (Session.tiles_probe), standalone on its own keeper fd like chip_session's step 1,
    or `--via URL` on the session's serving `serve --no-release --probe-tiles` daemon (§5.2 invariant 3: inside a
    session that daemon holds the only fd). One JSON object (S5A_SCHEMA) with 'dirty', 'verdict', 'reasons' and a
    section per tile; exit tiles_probe_rc (0 clean, 3 dirty, 4 harvested only, 1 no report)."""
    if len(set(tiles)) != len(tiles) or any(t not in regs.L2CPU_XY for t in tiles):
        log(f"probe: --tile {','.join(map(str, tiles))}: each of the L2CPU tiles {tuple(regs.L2CPU_XY)} at most once")
        return 2
    if args.via:
        try:
            rep = _admin_call(args.via, "tiles_probe", 120.0, tiles=list(tiles))
        except X280Error as e:
            rep = {"error": str(e)}
        _print(rep)
        return tiles_probe_rc(rep)
    sess = Session("sim" if args.dry_run else args.backend, (regs.DEFAULT_TILE,), dtype=args.dtype,
                   variant=args.variant, ledger_path=args.ledger, cross_read=True, hart=False, probe_tiles=tiles,
                   first_poll_us=args.first_poll_us, sim_ecc=getattr(args, "sim_ecc", None))
    try:
        sess.open()
        rep = sess.tiles_probe(tiles)
        rc = tiles_probe_rc(rep)
    except K.DeviceAccessRefused as e:
        rep, rc = {"error": f"device refused: {e}"}, 2
    except (X280Error, K.KmdError, OSError) as e:
        rep, rc = {"error": f"{type(e).__name__}: {e}"}, 1
    finally:
        sess.close()
    if args.dry_run:
        _dry_run_dump(sess)
    _print(rep)
    return rc


def cmd_probe(args) -> int:
    """chip_session step 1: one JSON object with a boolean 'dirty' (+ reasons, attachable); exit 0 / 3 / 4. With more
    than one --tile, a tile outside regs.WRITABLE_TILES or --via: the S5a probe (cmd_probe_tiles)."""
    tiles = _tiles(args.tile)
    if args.via or len(tiles) > 1 or tiles[0] not in regs.WRITABLE_TILES:
        return cmd_probe_tiles(args, tiles)
    sess = _session(args, cross_read=True, hart=False)
    rep = {"dirty": False, "reasons": [], "attachable": False, "backend": "sim" if sess.sim else "device"}
    rc = 0
    try:
        sess.open()
        inter = sess.interlocks(released_ok=True)  # probe states release itself (reasons below) and checks attach
        rep.update({k: inter[k] for k in ("l2cpu_reset", "pll4", "pll4_default", "pll_lock", "pll4_walk_ready",
                                          "timer_heartbeat", "enabled_l2cpu", "tiles")})
        if "noc_translation" in inter:
            rep["noc_translation"] = inter["noc_translation"]
        reasons = list(inter["dirty"])
        released = [t for t in sess.tiles if inter["tiles"][str(t)]["released"]]
        if inter["l2cpu_reset"] & 0xF0:
            reasons.append(f"L2CPU_RESET = 0x{inter['l2cpu_reset']:02x}: a tile was released without a chip reset")
        try:
            prev = sess.ledger.read()
        except ledger_mod.LedgerError as e:
            prev = None
            rep["ledger_error"] = str(e)
        if prev is not None:
            rep["ledger"] = {"hold_id": prev["hold_id"], "epoch": ledger_mod.epoch_verdict(
                prev, int(inter["timer_heartbeat"] or 0), time.time()),
                "tiles": {t: v["state"] for t, v in prev["tiles"].items()}}
        if released and prev is not None and prev["hold_id"] == sess.hold_id and inter["pll4_default"]:
            ok = True
            for t in released:  # attach (T-3.8): a live monitor released earlier in THIS hold
                sess.k.assume_lim_initialised(t, regs.LIM_BASE, regs.LIM_END, "probe: released in this hold")
                with sess.batch(t):
                    mb = sess.mb[t]
                    l1 = mb.l1()
                    hb0 = mb.heartbeat()
                    time.sleep(0.005)
                    live = mb.heartbeat() != hb0
                    trig = sess.k.read_rnmi_trigger(t)  # review F8: a hart left in the park stub can be resumed
                tr = rep["tiles"][str(t)]
                tr.update(rnmi_trigger=trig, heartbeat_live=bool(live), attach_resumes_park=bool(trig & 1 and not live))
                ok &= l1["fw_magic"] == ml.MAGIC and (live or bool(trig & 1))
            rep["attachable"] = bool(ok)
        rep["reasons"] = reasons
        rep["dirty"] = bool(reasons)
        if inter["unusable"]:
            rep["error"] = "; ".join(inter["unusable"])
            rc = 4
        elif rep["dirty"]:
            rc = 3
    except K.DeviceAccessRefused as e:
        rep.update(error=f"device refused: {e}")
        rc = 2
    except (K.KmdError, OSError) as e:
        rep.update(error=f"{type(e).__name__}: {e}")
        rc = 4
    finally:
        sess.close()
    if args.dry_run:
        _dry_run_dump(sess)
    _print(rep)
    return rc


def _standalone_up(sess: Session, args) -> dict:
    """Bring a standalone session to a serving hart: sim/dry-run boots; the device needs --boot or --attach."""
    if getattr(args, "attach", False):
        r = sess.attach()
    elif sess.sim and sess.harts == regs.MH_HARTS:
        r = sess.boot_stage0()  # --harts 4 goes with the stage-0 boot only (X280_E1F_APPROVED / X280_MH_APPROVED)
    elif sess.sim or getattr(args, "boot", False):
        r = sess.boot()  # refused on the device without X280_ALLOW_METHOD_A=1 (METHOD_A_REFUSED)
    else:
        raise X280Error("standalone on the device: pass --attach, or use --via (a standalone --boot is method A: "
                        "refused without X280_ALLOW_METHOD_A=1; boot with `serve --stage0`)")
    for t in sess.tiles:
        with sess.batch(t):
            r.setdefault("blob", {})[str(t)] = sess.load_blob(t)
    r["boot_parity"] = sess.check_parity()
    return r


def _harts_refusal(args) -> str:
    """'' unless --harts 4 is asked without the stage-0 boot or without the user's X280_MH_APPROVED=1."""
    if getattr(args, "harts", 1) != regs.MH_HARTS:
        return ""
    if not getattr(args, "stage0", False):
        return "--harts 4 goes only with --stage0 (kmd adopts harts 1-3 only on the E1F boot)"
    if os.environ.get(MH_APPROVED_ENV) != "1":
        return f"--harts 4 needs the user's approval of the 4-hart bench: {MH_APPROVED_ENV}=1"
    return ""


def cmd_boot(args) -> int:
    why = _harts_refusal(args)
    if why:
        log(f"boot: {why}")
        _print({"boot": {"error": why}, "steps": []})
        return 2
    if not args.stage0 and args.backend == "device" and not args.dry_run and not method_a_allowed():
        log(f"boot: {METHOD_A_REFUSED}")
        _print({"boot": {"error": METHOD_A_REFUSED}, "steps": []})
        return 2  # before any Session: nothing opened
    sess = _session(args)
    rc, out = 0, {}
    try:
        out = sess.boot_stage0() if args.stage0 else sess.boot()
        if args.load_blob:
            for t in sess.tiles:
                with sess.batch(t):
                    out.setdefault("blob", {})[str(t)] = sess.load_blob(t)
    except Dirty as e:
        out, rc = {"error": str(e), "dirty": True}, 3
    except Unusable as e:
        out, rc = {"error": str(e)}, 4
    except (X280Error, K.KmdError) as e:
        out, rc = {"error": f"{type(e).__name__}: {e}"}, 1
    finally:
        sess.close()
    if args.dry_run:
        print(f"# x280d boot --dry-run: tiles {sess.tiles}, kmd.FakeDevice + the monitor's x86 build as the hart "
              f"(x280/mailbox_sim.py); /dev/tenstorrent was not opened")
        print(f"# image {IMAGE_BIN} {len(sess.image)} B crc32 0x{sess.image_crc:08x}; park stub {PARK_STUB_BIN} "
              f"{len(sess.park_stub)} B; ledger {sess.ledger.path}")
        _dry_run_dump(sess)
    _print({"boot": out, "steps": [s[0] for s in sess.steps]})
    return rc


def cmd_serve(args) -> int:
    modes = [m for m in ("no_release", "boot", "attach", "stage0") if getattr(args, m)]
    if len(modes) > 1:
        log("serve: give one of --no-release, --boot, --attach, --stage0")
        return 2
    if "stage0" in modes and os.environ.get("X280_E1F_APPROVED") != "1":
        log("serve --stage0 (E1F) refused: it spends the release on the stage-0 initializer and needs the user's "
            "explicit DP3(d) approval (X280_E1F_APPROVED=1)")
        return 2
    why = _harts_refusal(args)
    if why:
        log(f"serve: {why}")
        return 2
    sim = args.backend == "sim" or args.dry_run
    mode = modes[0] if modes else ("boot" if sim else None)
    if mode is None:
        log("serve on the device needs --stage0 (the standard boot, X280_E1F_APPROVED=1), --attach or --no-release "
            "(chip_session passes one); --boot is method A, refused without X280_ALLOW_METHOD_A=1")
        return 2
    if mode == "boot" and not sim and not method_a_allowed():
        log(f"serve --boot: {METHOD_A_REFUSED}")
        return 2
    probe_tiles = _tiles([args.probe_tiles]) if args.probe_tiles else ()
    if probe_tiles and (mode != "no_release" or any(t not in regs.L2CPU_XY for t in probe_tiles)
                        or len(set(probe_tiles)) != len(probe_tiles)):
        log(f"serve --probe-tiles {args.probe_tiles} (S5a): L2CPU tiles {tuple(regs.L2CPU_XY)}, each once, and only "
            "with --no-release")
        return 2
    if args.cpus:
        os.sched_setaffinity(0, _cpu_list(args.cpus))  # before any thread or socket exists
    if sim and not args.sim_cpus and mode != "no_release":
        # A sim hart is a spinning thread: on x280d's own CPU it would make every reply wait for preemption, so
        # x280d gives it CPUs of its own mask when there are enough, and says so when there are not.
        aff, n = sorted(os.sched_getaffinity(0)), len(_tiles(args.tile))
        if len(aff) >= n + 1:
            args.sim_cpus = ",".join(str(c) for c in aff[-n:])
            log(f"sim: hart spin thread(s) on CPU {args.sim_cpus}; x280d keeps CPU(s) {aff[:-n]}")
        else:
            log(f"sim: CPU(s) {aff} for x280d and {n} spinning sim hart(s): they share x280d's CPU and x280d polls "
                f"by sleeping (higher RTT); give it one more CPU, e.g. taskset -c 5,21 ... --sim-cpus 21")
    log_dir = Path(args.log_dir) if args.log_dir else (Path(os.environ["CHIP_SESSION_DIR"]) if os.environ.get(
        "CHIP_SESSION_DIR") else None)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    try:
        sess = _session(args, cross_read=mode == "no_release", hart=mode != "no_release",
                        **({"probe_tiles": probe_tiles} if probe_tiles else {}))
    except (X280Error, FileNotFoundError, ValueError) as e:
        log(f"serve: {e}")
        return 2
    srv = None
    stop = {"flag": False}

    def on_sig(*_):
        stop["flag"] = True
        if srv is not None:
            srv.stop = True
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    rc = 0
    try:
        srv = Server(sess, args.port, early=args.early, slow=args.slow, deadline_ms=args.deadline_ms,
                     status_pub=args.status_pub, status_hz=args.status_hz,
                     log_dir=log_dir, spin=args.spin)  # bind first: a taken port fails before any release
        if srv.pub is not None:
            log(f"status PUB on {srv.pub.endpoint} at {srv.pub.hz:g} Hz")
        if mode == "no_release":
            sess.open()
            inter = sess.interlocks()
            if inter["unusable"]:
                raise Unusable("; ".join(inter["unusable"]))
            if inter["dirty"]:
                raise Dirty("; ".join(inter["dirty"]))
            sess.ledger_hold(inter["timer_heartbeat"])
            sess.mode = "no_release"
            log(f"no-release: keeper fd and windows up, L2CPU_RESET 0x{inter['l2cpu_reset']:02x}; no hart will run")
        else:
            r = {"boot": sess.boot, "attach": sess.attach, "stage0": sess.boot_stage0}[mode]()
            log(f"{mode}: {json.dumps(r.get('monitor') or r.get('tiles'), default=_jsonable)}")
            for t in sess.tiles:
                with sess.batch(t):
                    if not args.no_blob:
                        b = sess.load_blob(t)
                        log(f"blob on L2CPU{t}: {b['len']} B crc32 0x{b['crc32']:08x} in {b['secs']} s")
            if not args.no_blob:
                bp = sess.check_parity()
                log(f"boot parity vs x86 build: {bp}")
                if not bp["pass"]:
                    srv.degraded = f"boot parity failed: {bp['max_abs_vs_x86']}"
        if not stop["flag"]:
            srv.run()
    except Dirty as e:
        log(f"DIRTY: {e}")
        rc = 3
    except Unusable as e:
        log(f"UNUSABLE: {e}")
        rc = 4
    except zmq.ZMQError as e:
        log(f"serve: cannot bind :{args.port}: {e}")
        rc = 2
    except (X280Error, K.KmdError, OSError) as e:
        log(f"serve failed: {type(e).__name__}: {e}")
        rc = 1
    finally:
        if srv is not None:
            log(f"stopping: counts {srv.counts}")
            srv.close()
        sess.close()
        if args.dry_run:
            _dry_run_dump(sess)
    return rc


def _via_or_local(args, cmd: str, timeout_s: float = 120.0, **kw) -> int:
    """Run an admin command on the serving daemon (--via) or in a standalone session."""
    if args.via:
        rep = _admin_call(args.via, cmd, timeout_s, tile=_tiles(args.tile)[0], **kw)
        _print(rep)
        return 1 if "error" in rep or rep.get("pass") is False else 0
    no_hart = cmd in ("e0", "lim_init", "lim_check", "lim_ecc", "bench_host_only")
    sess = _session(args, cross_read=cmd == "e0", hart=not no_hart)
    try:
        if no_hart:  # S1c-style: keeper fd and windows, no release
            sess.open()
            inter = sess.interlocks()
            if (inter["dirty"] and cmd != "e0") or inter["released_tiles"]:  # e0 reports dirt, but never on a live hart
                raise Dirty("; ".join(inter["dirty"]))
            sess.mode = "no_release"  # LIM init state is per process (kmd LimTracker), as on the device
        else:
            _standalone_up(sess, args)
        rep = run_admin(sess, cmd, tile=_tiles(args.tile)[0], **kw)
        rc = 1 if rep.get("pass") is False else 0
    except Dirty as e:
        rep, rc = {"error": str(e), "dirty": True}, 3
    except Unusable as e:
        rep, rc = {"error": str(e)}, 4
    except (X280Error, K.KmdError, OSError, ValueError) as e:
        rep, rc = {"error": f"{type(e).__name__}: {e}"}, 1
    finally:
        sess.close()
    if args.dry_run:
        _dry_run_dump(sess)
    _print(rep)
    return rc


def cmd_e0(args) -> int:
    if not args.peek:
        return _via_or_local(args, "e0")
    if not args.via:
        log("e0 --peek needs --via: server_g1 and x280d share the device inside one session")
        return 2
    rep = _admin_call(args.via, "e0", 60.0, tile=_tiles(args.tile)[0])
    if "error" in rep:
        _print(rep)
        return 1
    rep["peek"] = peek_cross_check(rep, args.peek, _tiles(args.tile)[0])
    rep["pass"] = bool(rep.get("pass") and rep["peek"]["agree"])
    _print(rep)
    return 0 if rep["pass"] else 1


def cmd_dump(args) -> int:
    """Jobs (T-3.7): the serving daemon writes its tick ring where analyze_run looks for it (x280d_ticks.npy for
    x280_heartbeat_continuity). The ring lives in that daemon, so this needs --via."""
    if not args.via:
        log("dump needs --via URL: the tick ring belongs to the serving x280d")
        return 2
    rep = _admin_call(args.via, "dump", 60.0, **({"out": str(Path(args.out).resolve())} if args.out else {}))
    _print(rep)
    return 0 if rep.get("ring") and "error" not in rep else 1


def cmd_ping(args) -> int:
    url = args.via or os.environ.get("CHIP_X280_URL") or os.environ.get("WBC_X280", "tcp://127.0.0.1:5558")
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(url)
    s.send(wire.request("ping"))
    if not s.poll(int(args.timeout_s * 1000)):
        _print({"url": url, "ok": False, "error": "timeout"})
        return 1
    rep = wire.loads(s.recv())
    s.send(wire.request("info"))
    info = wire.loads(s.recv()) if s.poll(int(args.timeout_s * 1000)) else {}
    s.close(0)
    _print({"url": url, "ok": "error" not in rep, "ping": rep, "info": info})
    return 0 if "error" not in rep else 1


def _reply_ready(s, timeout_ms: float, spin: bool) -> bool:
    """A reply is waiting on REQ socket s within timeout_ms: zmq poll (blocks in the kernel, as rt_sim's ZmqWbc
    waits on the socket FD), or with spin a busy re-check of EVENTS (no sleep, so no wake-up latency in the RTT)."""
    if not spin:
        return bool(s.poll(int(timeout_ms)))
    t_end = time.perf_counter() + timeout_ms / 1e3
    while time.perf_counter() < t_end:
        if s.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            return True
    return bool(s.getsockopt(zmq.EVENTS) & zmq.POLLIN)


def cmd_bench_rtt(args) -> int:
    """T-3.5 client: n W-WBC calls at a fixed period against the daemon; client RTT vs the daemon's own split.

    --cpus LIST places the client as rt_sim places its tick thread (PLAN §1.1, rtloop/affinity.py): the calling
    thread on the lowest CPU of LIST and the zmq I/O thread on the highest (2,3 = rt_sim's tick and helper CPUs), so
    the client RTT is a loop's, not a floating process's on whatever core is free. --spin busy-polls each reply
    instead of blocking in poll (a lower bound without the wake-up). `pass` is PLAN T-3.5's verdict, x280d side only
    (`checks`); the client RTT is reported apart (`reported`), never part of it."""
    url = args.via or os.environ.get("CHIP_X280_URL") or "tcp://127.0.0.1:5558"
    obs, pols = None, None
    if args.obs:
        d = np.load(args.obs)
        obs, pols = d["obs"], [str(x) for x in d["policy"]] if "policy" in d.files else None
    rng = np.random.default_rng(0)
    client = {"cpus": args.cpus, "spin": bool(args.spin), "thread_cpus": None, "io_cpu": None}
    if args.cpus:
        cpus = sorted(_cpu_list(args.cpus))
        os.sched_setaffinity(0, {cpus[-1]})  # the I/O thread inherits the mask of the thread that starts it ...
        ctx = zmq.Context(io_threads=1)
        ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, cpus[-1])  # ... and is pinned explicitly before the first socket
        client["io_cpu"] = cpus[-1]
    else:
        ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(url)
    if args.cpus:
        os.sched_setaffinity(0, {cpus[0]})  # the calling thread: rt_sim's tick CPU
        client["thread_cpus"] = sorted(os.sched_getaffinity(0))
    s.send(wire.request("info"))
    if not s.poll(5000):
        _print({"url": url, "error": "no reply to info", "pass": False})
        return 1
    backend = str(wire.loads(s.recv()).get("backend", "?"))
    rtt, comp, xd, mbx, errs, to = [], [], [], [], {}, 0
    period = args.period_ms / 1e3
    t_next = time.perf_counter()
    for i in range(args.n):
        o = (obs[i % len(obs)] if obs is not None else rng.standard_normal(ml.OBS_DIM).astype(np.float32))
        pol = pols[i % len(pols)] if pols else POLICIES[i & 1 if args.alternate else 0]
        t0 = time.perf_counter()
        s.send(wire.request("gear_wbc_action", {"policy": pol, "obs": np.asarray(o, np.float32).reshape(1, -1),
                                                "tick": i, "seq": i}))
        if not _reply_ready(s, args.timeout_ms, args.spin):
            to += 1
            s.close(0)
            s = ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(url)
        else:
            rep = wire.loads(s.recv())
            dt = (time.perf_counter() - t0) * 1e6
            if "error" in rep:
                k = str(rep["error"])[:60]
                errs[k] = errs.get(k, 0) + 1
            else:
                rtt.append(dt)
                comp.append(rep.get("compute_us", np.nan))
                xd.append(rep.get("x280d_us", np.nan))
                mbx.append(rep.get("mailbox_us", np.nan))
        t_next += period
        while time.perf_counter() < t_next:
            pass
    s.close(0)
    if args.cpus:
        ctx.term()  # this call's own context (Context.instance() is shared and left alone)

    def q(a):
        a = np.asarray(a, np.float64)
        a = a[np.isfinite(a)]
        return {"p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)), "max": float(a.max())} if a.size else None
    host = np.asarray(xd, np.float64) - np.asarray(comp, np.float64)
    rep = {"url": url, "n": args.n, "ok": len(rtt), "timeouts": to, "errors": errs, "rtt_us": q(rtt),
           "compute_us": q(comp), "x280d_us": q(xd), "mailbox_us": q(mbx), "host_overhead_us": q(host)}
    r, c, h, x = rep["rtt_us"], rep["compute_us"], rep["host_overhead_us"], rep["x280d_us"]
    rep["backend"] = backend
    limit = 9000.0 if args.slow else 2000.0
    # The PLAN T-3.5 gates, all x280d side, and the only ones `pass` reads: x280d end-to-end (recv -> send in x280d)
    # p99 <= compute p99 + 0.5 ms and <= 2 ms (9 ms DP11); host mailbox overhead (x280d - compute) p99 <= 0.3 ms,
    # device only (in the sim compute_us is the turnaround, so the row does not apply there); 0 timeouts and errors.
    # A row without data fails.
    rep["gates"] = "PLAN T-3.5, x280d side: checks"
    rep["checks"] = {"x280d_p99_le_compute_p99_plus_500us": bool(x and c and x["p99"] <= c["p99"] + 500.0),
                     "x280d_p99_le_limit": bool(x and x["p99"] <= limit),
                     "host_overhead_p99_le_300us": backend.startswith("x280-sim") or bool(h and h["p99"] <= 300.0),
                     "no_timeouts_or_errors": bool(args.n) and not to and not errs}
    # Reported, not gated: the client RTT adds ZMQ, msgpack and this client's own scheduling (where it ran: `client`).
    # The loop's WBC RTT is m3_rt's wbc_rtt_p99_ms, gated by analyze_run in T-3.7 on rt_sim's own calls.
    rep["reported"] = {"client_rtt_p99_le_limit": bool(r and r["p99"] <= limit), "client_rtt_limit_us": limit,
                       "client": client}
    rep["pass"] = all(rep["checks"].values())
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))
    _print(rep)
    return 0 if rep["pass"] else 1


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", choices=("device", "sim"), default="device")
    common.add_argument("--dry-run", action="store_true", help="sim backend + print every NoC op and libwcio txn")
    common.add_argument("--tile", action="append", help="L2CPU tile(s), e.g. 3 or 3,2 (DP4 fp32)")
    common.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    common.add_argument("--variant", choices=tuple(VARIANTS), default=None)
    common.add_argument("--ledger", default=None, help="default: ~/experiments/gr00t/.x280_ledger.json (device), "
                                                       "a temp file (sim)")
    common.add_argument("--sim-cpus", default=None, help="sim: one CPU per hart for its spin thread")
    common.add_argument("--first-poll-us", type=float, default=POLL_FIRST_US)
    common.add_argument("--via", default=None, help="send the command to the serving x280d at this URL")
    common.add_argument("--boot", action="store_true",
                        help="method-A boot (host LIM init; spends the release). Device: refused unless "
                             "X280_ALLOW_METHOD_A=1, use --stage0")
    common.add_argument("--attach", action="store_true", help="standalone: attach to a released tile")
    common.add_argument("--sim-ecc", choices=K.FakeEcc.MODES, default=None,
                        help="sim only: model the CCACHE's LIM data ECC (kmd.FakeEcc; poison = the S1c hypothesis)")
    common.add_argument("--harts", type=int, choices=(1, regs.MH_HARTS), default=1,
                        help="4: the 4-hart bench (S3b): harts 1-3's RNMI vectors -> crt0 / MH_SEC_RNMI_EXC before the "
                             f"release; --stage0 and {MH_APPROVED_ENV}=1 only; default 1 = the standard boot")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", parents=[common], help="L2CPU health check (chip_session step 1); --tile 3,2,1,0 "
                                                       "(or --via): the S5a read-only probe of several tiles")
    p.add_argument("--json", action="store_true", help="(always JSON on the last stdout line)")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("boot", parents=[common], help="§5.2 boot sequence (use --dry-run for T-3.4); device: --stage0 "
                                                      "(method A needs X280_ALLOW_METHOD_A=1)")
    p.add_argument("--load-blob", action="store_true")
    p.add_argument("--stage0", action="store_true", help="E1F, the standard boot: hart-run LIM init (needs "
                                                         "X280_E1F_APPROVED=1, DP3(d))")
    p.set_defaults(fn=cmd_boot)

    p = sub.add_parser("serve", parents=[common], help="W-WBC on REP 127.0.0.1:PORT")
    p.add_argument("--port", type=int, default=5558)
    p.add_argument("--no-release", action="store_true", help="S1c: keeper fd and windows, never release")
    p.add_argument("--probe-tiles", default=None, help="S5a, with --no-release only: read-only probe windows on these "
                                                       "L2CPU tiles (e.g. 3,2,1,0) for `probe --tile LIST --via`")
    p.add_argument("--stage0", action="store_true", help="E1F boot, the standard one (X280_E1F_APPROVED=1 only, DP3(d))")
    p.add_argument("--early", action="store_true", help="enable gear_wbc_dispatch/collect (WP-3.8b)")
    p.add_argument("--slow", action="store_true", help="m3_rt_slow deadline (12 ms, DP11)")
    p.add_argument("--deadline-ms", type=float, default=None)
    p.add_argument("--no-blob", action="store_true", help="do not LOAD the weight blob (S2 payload work)")
    p.add_argument("--status-pub", nargs="?", const=STATUS_PUB_URL, default=None,
                   help=f"HUD chip-panel status PUB (PLAN §4.3 item 7): bare flag = {STATUS_PUB_URL}, "
                        "'0' = a random port (tests); default off")
    p.add_argument("--status-hz", type=float, default=STATUS_HZ, help="status rate, clamped to 2-10 Hz")
    p.add_argument("--log-dir", default=None, help="default $CHIP_SESSION_DIR")
    p.add_argument("--cpus", default=None, help="CPU affinity of this process, e.g. 5 (PLAN §1.1 x280d row); "
                                                "refused outside CHIP_SAFE_CPUS when that is set")
    p.add_argument("--spin", action="store_true", help="busy-poll the REP socket, yielding per miss (a dedicated "
                                                       "--cpus core; its SMT sibling idle)")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("attach", parents=[common], help="attach to a released tile, report, exit")
    p.set_defaults(fn=lambda a: _via_or_local(_force(a, attach=True), "status"))

    p = sub.add_parser("ping", parents=[common])
    p.add_argument("--timeout-s", type=float, default=2.0)
    p.set_defaults(fn=cmd_ping)

    p = sub.add_parser("e0", parents=[common], help="T-3.E0 register set (--peek: cross-check via server_g1)")
    p.add_argument("--peek", default=None, help="server_g1 URL with --debug-endpoints (x280_peek)")
    p.set_defaults(fn=cmd_e0)

    p = sub.add_parser("lim-init", parents=[common], help="T-3.E1a/b LIM init")
    p.add_argument("--method", choices=("A", "B"), default="A")
    p.add_argument("--store", choices=tuple(K.LIM_INIT_STORES), default="a64",
                   help="method A's host stores (kmd.LIM_INIT_STORES; a64 = the boot's); S1d compares them")
    p.add_argument("--lo", type=lambda x: int(x, 0), default=regs.LIM_BASE)
    p.add_argument("--hi", type=lambda x: int(x, 0), default=regs.LIM_END)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--vla-url", default=None)
    p.set_defaults(fn=lambda a: _via_or_local(a, "lim_init", 600.0, method=a.method, lo=a.lo, hi=a.hi, seed=a.seed,
                                              vla_url=a.vla_url, store=a.store))

    p = sub.add_parser("lim-check", parents=[common], help="T-3.E1a/b read-back + partial writes + cleanup check")
    p.add_argument("--lo", type=lambda x: int(x, 0), default=regs.LIM_BASE)
    p.add_argument("--hi", type=lambda x: int(x, 0), default=regs.LIM_END)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--partial", type=int, default=0)
    p.add_argument("--cleanup-check", action="store_true")
    p.set_defaults(fn=lambda a: _via_or_local(a, "lim_check", 600.0, lo=a.lo, hi=a.hi, seed=a.seed, stride=a.stride,
                                              partial=a.partial, cleanup_check=a.cleanup_check))

    p = sub.add_parser("lim-ecc", parents=[common],
                       help="S1d: LIM init store variants on fresh regions vs the CCACHE ECC counters (jobs/s1d.sh)")
    p.add_argument("--plan", default=S1D_PLAN, help=f"comma list of stores {'/'.join(K.LIM_INIT_STORES)}; "
                                                    f"'x+y' overwrites one region with x then y (default {S1D_PLAN})")
    p.add_argument("--lines", type=int, default=S1D_LINES, help="lines per region")
    p.add_argument("--base", type=lambda x: int(x, 0), default=regs.S1D_BASE)
    p.add_argument("--probe-lines", type=int, default=S1D_PROBE_LINES)
    p.add_argument("--partial", type=int, default=S1D_PARTIAL)
    p.add_argument("--seed", type=int, default=S1D_SEED)
    p.set_defaults(fn=lambda a: _via_or_local(a, "lim_ecc", 600.0, plan=a.plan, lines=a.lines, base=a.base,
                                              probe_lines=a.probe_lines, partial=a.partial, seed=a.seed))

    p = sub.add_parser("selftest", parents=[common])
    p.set_defaults(fn=lambda a: _via_or_local(a, "selftest"))

    p = sub.add_parser("load", parents=[common], help="LOAD a file (code: also FENCE_I)")
    p.add_argument("--file", required=False)
    p.add_argument("--blob", action="store_true", help="re-LOAD the session's weight blob")
    p.add_argument("--dst", type=lambda x: int(x, 0), default=regs.WEIGHTS_BASE)
    p.set_defaults(fn=lambda a: _via_or_local(a, "load_blob") if a.blob else
                   _via_or_local(a, "load", file=str(Path(a.file).resolve()), dst=a.dst))

    p = sub.add_parser("call", parents=[common])
    p.add_argument("--entry", type=lambda x: int(x, 0), default=regs.WEIGHTS_BASE)
    p.add_argument("--policy", type=int, default=0)
    p.add_argument("--flags", type=int, default=0)
    p.add_argument("--arg-addr", type=lambda x: int(x, 0), default=0)
    p.add_argument("--arg-len", type=int, default=0)
    p.add_argument("--tick", type=int, default=0)
    p.add_argument("--deadline-s", type=float, default=CMD_DEADLINE_S)
    p.set_defaults(fn=lambda a: _via_or_local(a, "call", entry=a.entry, policy=a.policy, flags=a.flags,
                                              arg_addr=a.arg_addr, arg_len=a.arg_len, tick=a.tick,
                                              deadline_s=a.deadline_s))

    p = sub.add_parser("park", parents=[common], help="RNMI PARK protocol (§5.2); --cooperative = mailbox PARK")
    p.add_argument("--hold-s", type=float, default=0.0, help="keep the RNMI bit set this long (fault injection)")
    p.add_argument("--cooperative", action="store_true")
    p.set_defaults(fn=lambda a: _via_or_local(a, "park_cooperative") if a.cooperative else
                   _via_or_local(a, "park", 30.0 + a.hold_s, hold_s=a.hold_s))

    p = sub.add_parser("e2", parents=[common], help="T-3.E2 host part: checksum LOAD+CALL and RNMI re-arm, n times")
    p.add_argument("--n", type=int, default=10)
    p.set_defaults(fn=lambda a: _via_or_local(a, "e2", 300.0, n=a.n))

    p = sub.add_parser("lim-bench", parents=[common], help="T-3.E2m LIM microbench payload")
    p.add_argument("--tests", default="1,2,3,4,5,6,8")
    p.add_argument("--arg-len", type=int, default=0)
    p.add_argument("--iters", type=int, default=0)
    p.add_argument("--zero-device", action="store_true", help="also allow test 7 (Zero Device, flags bit 0)")
    p.set_defaults(fn=lambda a: _via_or_local(a, "lim_bench", 600.0, tests=[int(x) for x in a.tests.split(",")],
                                              arg_len=a.arg_len, iters=a.iters, flags=int(a.zero_device)))

    p = sub.add_parser("parity", parents=[common], help="T-3.E5a: obs516.npz through the hart vs x86 and ORT")
    p.add_argument("--obs", required=True)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--blob", choices=("fp16", "fp32_balance", "fp32_walk"), default=None,
                   help="LOAD this blob for the run (the session blob is LOADed again after)")
    p.add_argument("--kernel", dest="kvariant", choices=tuple(VARIANTS), default=None, help="variant for this run")
    p.set_defaults(fn=lambda a: _via_or_local(a, "parity", 3600.0, obs=str(Path(a.obs).resolve()), n=a.n,
                                              out=str(Path(a.out).resolve()) if a.out else None, blob=a.blob,
                                              variant=a.kvariant))

    p = sub.add_parser("harts", parents=[common], help="harts 1-3 (S3b): adopt (RNMI bits 1-3 once), status, ping")
    p.add_argument("action", choices=("adopt", "status", "ping"))
    p.add_argument("--mask", type=lambda x: int(x, 0), default=regs.MH_SEC_MASK, help="harts (bits 1-3), default 0xe")
    p.set_defaults(fn=lambda a: _via_or_local(a, "harts_adopt", 60.0, mask=a.mask) if a.action == "adopt" else
                   _via_or_local(a, "mh_status") if a.action == "status" else
                   _via_or_local(a, "mh_ping", 30.0, mask=a.mask))

    p = sub.add_parser("bench", parents=[common], help="T-3.E4 BENCH / --host-only (S1c) / --rtt (T-3.5) / --mh (S3b)")
    p.add_argument("--host-only", action="store_true")
    p.add_argument("--mh", action="store_true", help="the 4-hart LIM bench (mh_bench payload; needs `harts adopt`)")
    p.add_argument("--mh-plan", choices=("full", "quick"), default="full")
    p.add_argument("--mh-repeats", type=int, default=3)
    p.add_argument("--mh-window-us", type=int, default=ml.MHB_WINDOW_US)
    p.add_argument("--mh-bench-iters", type=int, default=200, help="--mh: hart-0 BENCH iterations of the interference row")
    p.add_argument("--rtt", action="store_true")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--policy", type=int, default=0)
    p.add_argument("--out-addr", type=lambda x: int(x, 0), default=0)
    p.add_argument("--period-ms", type=float, default=20.0)
    p.add_argument("--timeout-ms", type=float, default=50.0)
    p.add_argument("--obs", default=None)
    p.add_argument("--alternate", action="store_true", help="--rtt: alternate Balance/Walk")
    p.add_argument("--slow", action="store_true", help="--rtt: DP11 limit (9 ms)")
    p.add_argument("--cpus", default=None, help="--rtt: the client's CPUs, as rt_sim's (PLAN §1.1): the calling "
                                                "thread on the lowest, the zmq I/O thread on the highest (2,3); "
                                                "refused outside CHIP_SAFE_CPUS when that is set")
    p.add_argument("--spin", action="store_true", help="--rtt: busy-poll each reply instead of blocking in poll")
    p.add_argument("--out", default=None)
    p.add_argument("--matrix", action="store_true", help="T-3.E4: every built variant x weight dtype x policy")
    p.add_argument("--blob", choices=("fp16", "fp32_balance", "fp32_walk"), default=None)
    p.add_argument("--kernel", dest="kvariant", choices=tuple(VARIANTS), default=None)
    p.set_defaults(fn=lambda a: cmd_bench_rtt(a) if a.rtt else (
        _via_or_local(a, "bench_mh", 900.0, plan=a.mh_plan, repeats=a.mh_repeats, window_us=a.mh_window_us,
                      bench_iters=a.mh_bench_iters) if a.mh else
        _via_or_local(a, "bench_host_only", 600.0, n=a.n) if a.host_only else
        _via_or_local(a, "bench_matrix", 3600.0, iters=a.iters) if a.matrix else
        _via_or_local(a, "bench", 600.0, policy=a.policy, iters=a.iters, out_addr=a.out_addr, blob=a.blob,
                      variant=a.kvariant)))

    p = sub.add_parser("hbrate", parents=[common], help="T-3.E2 heartbeat rate and mtime vs wall time")
    p.add_argument("--secs", type=float, default=0.2)
    p.set_defaults(fn=lambda a: _via_or_local(a, "heartbeat_rate", 30.0, secs=a.secs))

    p = sub.add_parser("powertest", parents=[common], help="T-3.E3 heartbeat across legacy fd churn / a watch window")
    p.add_argument("--fd-churn", type=int, default=10)
    p.add_argument("--watch-s", type=float, default=0.0)
    p.set_defaults(fn=lambda a: _via_or_local(a, "powertest", 120.0 + a.watch_s, fd_churn=a.fd_churn,
                                              watch_s=a.watch_s))

    p = sub.add_parser("inject", parents=[common], help="fault injection on the serving daemon (T-3.8)")
    p.add_argument("what", choices=("crc", "hang", "mh_hang"))
    p.add_argument("--n", type=int, default=1)
    p.set_defaults(fn=lambda a: _via_or_local(a, "inject", what=a.what, n=a.n))

    p = sub.add_parser("status", parents=[common])
    p.set_defaults(fn=lambda a: _via_or_local(a, "status"))

    p = sub.add_parser("dump", parents=[common], help=f"the serving x280d (--via) writes its tick ring ({RING_FILE})")
    p.add_argument("--out", default=None, help="a directory or .npy path (default: the daemon's --log-dir)")
    p.set_defaults(fn=cmd_dump)

    args = ap.parse_args(argv)
    if args.cmd == "bench" and not args.rtt and (args.cpus or args.spin):
        log("bench: --cpus and --spin place the --rtt client; without --rtt they would be ignored")
        return 2
    if getattr(args, "sim_ecc", None) and not (args.dry_run or args.backend == "sim"):
        log(f"{args.cmd}: --sim-ecc models the sim chip's CCACHE ECC; with --backend device it is refused")
        return 2
    why = _outside_safe_cpus(args)
    if why:
        log(f"{args.cmd}: {why}; refusing to start")
        return 2
    try:
        return args.fn(args)
    except FileNotFoundError as e:
        log(str(e))
        return 2
    except ValueError as e:  # a Session argument combination it refuses (e.g. --sim-ecc with a sim hart): usage
        log(f"{args.cmd}: {e}")
        return 2
    except Dirty as e:
        log(f"DIRTY: {e}")
        return 3


def _force(a, **kw):
    for k, v in kw.items():
        setattr(a, k, v)
    return a


if __name__ == "__main__":
    sys.exit(main())
