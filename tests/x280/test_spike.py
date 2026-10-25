"""T-3.2b (PLAN.md §2 M3, WP-3.2b): the X280 monitor image runs on spike before E2 may run on silicon.

The firmware is built with -DSPIKE_HOST (contract: x280/fw/SPIKE_HOST.md): the crt0/monitor/trap/kernel
sources of build/monitor.elf plus a host scripted into the image. The X280-only RNMI CSRs 0x350-0x353, the
chicken-bit CSR 0x7c1 and mnret are emulated there, because this spike traps all of them
(x280/sim/selftest/t_csr.c). x280/sim/run_spike.py runs the image on spike (rv64gcv_zfh_zba_zbb, VLEN 512).
What T-3.2b asks for, and where it is checked:
  - SELFTEST; LOAD/FENCE_I/CALL 10/10; the illegal-instruction trap dump, also against spike's own
    exception record; the LIM microbench payload, every test: the default script, run LIM-only (the plan's
    -m0x8000000:0x1e0000) and again with the real park stub in the GP scratch. The same runs cover PARK,
    hang -> RNMI -> warm entry 10/10, the obs-CRC fault, host gone and an unknown command.
  - INFER on 1,000 logged obs516 (500 Balance + 500 Walk) per variant: v0 is bit-identical to the x86
    build's v0, and v1 (against x86 v0) and v2 (against x86 v2) are within 1e-6. v2s is checked on 64 obs.
    v3 (the single-hart kernel: v2's arithmetic, one vector op fewer per weight) is bit-identical to x86 v2 and to
    spike's own v2 run on the same obs, and again on crafted obs whose rows take v3's v2 fallback (|x| >= 2^16,
    inf, NaN) or sit on its edge; those rows show the fallback in their instruction counts.
  - The stage-0 stub, the bytes of build/stage0.bin unchanged, zero-fills LIM and jumps to the entry word.
  - mon_crc32 as built for the hart (slicing-by-8 tables, R33 follow-up) against zlib: every run's `SH crc` lines
    (all lengths 0..520 plus 2,112 and 4,096 from each 8 B alignment, and a chained CRC); and a -DMON_CRC_BITWISE
    build (SPIKE_CRC_BITWISE=1) whose INFER outputs, L1/L35 lines and BENCH act_crc32 equal the table build's.
Instruction counts per INFER (spike: mcycle == minstret) are printed and, with $T32B_REPORT=<path>, written
as JSON; the kernel lane's R33 budget table rests on them. $T32B_CRC_REPORT=<path> writes the bitwise vs table
monitor overhead per INFER.

  simpy -m pytest $RT/tests/x280/test_spike.py -q        (about 35 s; no device, no GPU)

Needs x280/sim/build_spike.sh output, clang-20/ld.lld-20/llvm-objcopy-20, and the logged obs516 that
test_kernel_x86.py uses ($WBC_OBS516_NPZ; the INFER tests skip without it). Every build and spike run is
pinned to nice 19 on CPUs 0,1,8-11,16,17,24-27.
"""
from __future__ import annotations

import concurrent.futures
import ctypes
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

RT = Path(__file__).resolve().parents[2]
FW = RT / "x280" / "fw"
SIM = RT / "x280" / "sim"
# Off the real-time cores 2,3,4,6,7,12-15 and their SMT siblings (task rule; override with X280_CPUS).
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
PIN = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]   # never the tick thread's CPUs 2, 3, 18, 19
OBS516_NPZ = os.environ.get(
    "WBC_OBS516_NPZ", str(RT / "runs/x280/20261024_211935_obs516_kernel/obs516.npz"))
N_PER_POLICY = 500          # x 2 policies = the plan's 1k obs per variant
N_V2S = 32                  # per policy; v2s is ~3.6 M instructions per INFER
TOL_V1_V2 = 1e-6            # PLAN.md T-3.2b
V0, V1, V2, V2S, V3 = 0, 1, 2, 3, 4
VNAME = {V0: "v0", V1: "v1", V2: "v2", V2S: "v2s", V3: "v3"}
DEFAULT_TIMEOUT_S = 120.0  # wall clock per spike run (a default-script run takes < 1 s, a parity run < 30 s)
PARITY_TIMEOUT_S = 600.0
# Kernel instructions per INFER that the kernel lane measured on spike (VLEN 512) and that its R33
# budget table uses; a drift beyond KERNEL_DRIFT means that table has to be re-derived.
KERNEL_LANE_INSNS = {V0: 1_712_077, V1: 40_042, V2: 78_287, V2S: 3_588_086, V3: 54_145}
KERNEL_DRIFT = 0.15


def _load(name: str, path: Path):
    """Import a file by path once per session (the other x280 test files load the same tools)."""
    old = sys.modules.get(name)
    if old is not None and Path(getattr(old, "__file__", "")).resolve() == path.resolve():
        return old
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # dataclasses resolves annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


rs = _load("run_spike", SIM / "run_spike.py")
ml = _load("mailbox_layout", FW / "mailbox_layout.py")
ci = _load("check_image", RT / "scripts" / "check_image.py")

TOOLS = ("clang-20", "ld.lld-20", "llvm-objcopy-20")
pytestmark = pytest.mark.skipif(
    not rs.SPIKE_BIN.exists() or not all(shutil.which(t) for t in TOOLS),
    reason="needs x280/sim/build_spike.sh output and clang-20/ld.lld-20/llvm-objcopy-20")

ST = ml.STATE
PAY = ml.PAYLOAD_BASE
GP_SCRATCH = rs.GP_SCRATCH[0]                            # 0x2001_0100, MemoryMap.md:113
X280_ONLY_CSRS = {0x350, 0x351, 0x352, 0x353, 0x7C1}   # RNMIs.md:63-70; README.md:49
SMRNMI_CSRS = set(range(0x740, 0x745))
RNMI_CAUSE = (1 << 63) | 2                               # what spike_host.c reports for the host's RNMI
SPIKE_CYC_PER_MTIME = 100                                # riscv-isa-sim riscv/sim.h:79-80 (SPIKE_HOST.md)

STEP_RE = re.compile(
    r"^SH step=(?P<step>\d+) op=(?P<op>\S+) sub=(?P<sub>\d+) seq=(?P<seq>\d+) res=(?P<res>\S+) "
    r"cyc=(?P<cyc>\d+) cyc_sum=(?P<cyc_sum>\d+) L1=(?P<L1>[0-9a-f]{128}) L35=(?P<L35>[0-9a-f]{128}) "
    r"L36=(?P<L36>[0-9a-f]{128}) L39=(?P<L39>[0-9a-f]{128})$", re.M)
BOOT_RE = re.compile(r"^SH boot L1=(?P<L1>[0-9a-f]{128}) L36=(?P<L36>[0-9a-f]{128}) L39=(?P<L39>[0-9a-f]{128})$",
                     re.M)
END_RE = re.compile(r"^SH end steps=(\d+) timeouts=(\d+) entries=(\d+)$", re.M)
CRC_RE = re.compile(r"^SH crc off=(\d+) nmax=(\d+) slices=(\d+) digest=(\d+) chained=(\d+)$", re.M)
N_CRC_CMP = 32              # logged Balance obs for the bitwise-CRC build
MON_BITWISE_OVERHEAD = 124_305   # monitor instructions per INFER with the bitwise CRC (T-3.2b, R33_ESTIMATE.md)


# --- helpers ----------------------------------------------------------------------------------------------

def _make(*args: str) -> None:
    r = subprocess.run([*PIN, "make", "-s", "-C", str(FW), *args], capture_output=True, text=True)
    assert r.returncode == 0, f"make {' '.join(args)}\n{r.stdout}\n{r.stderr}"


def _build_spike(out: Path, **cfg) -> Path:
    """`make spike` into its own SPIKE_BUILD dir (SPIKE_HOST.md). Builds share build/gen and build/payloads,
    so they run one at a time; only the spike runs go in parallel."""
    _make("spike", f"SPIKE_BUILD={out}", *(f"{k}={v}" for k, v in cfg.items()))
    return out


class Run:
    """A parsed SPIKE_HOST run: the boot record, one dict per step line, the end line."""

    def __init__(self, res):
        self.res = res
        out = res.stdout
        b = BOOT_RE.search(out)
        self.boot = None if b is None else {
            "l1": ml.unpack_l1(bytes.fromhex(b["L1"])), "l36": ml.unpack_l36(bytes.fromhex(b["L36"])),
            "l39": ml.unpack_l39(bytes.fromhex(b["L39"]))}
        self.steps = []
        for m in STEP_RE.finditer(out):
            d = {k: (int(m[k]) if k in ("step", "sub", "seq", "cyc", "cyc_sum") else m[k])
                 for k in ("step", "op", "sub", "seq", "res", "cyc", "cyc_sum")}
            d["l1"] = ml.unpack_l1(bytes.fromhex(m["L1"]))
            d["l35"] = ml.unpack_l35(bytes.fromhex(m["L35"]))
            d["l36"] = ml.unpack_l36(bytes.fromhex(m["L36"]))
            d["l39"] = ml.unpack_l39(bytes.fromhex(m["L39"]))
            d["l35_raw"] = bytes.fromhex(m["L35"])
            self.steps.append(d)
        e = END_RE.search(out)
        self.end = None if e is None else tuple(int(x) for x in e.groups())

    def op(self, name: str) -> list[dict]:
        return [s for s in self.steps if s["op"] == name]

    def one(self, name: str) -> dict:
        found = self.op(name)
        assert len(found) == 1, (name, len(found))
        return found[0]

    def describe(self) -> str:
        return f"exit={self.res.exit_code} timed_out={self.res.timed_out}\n{self.res.stdout[-3000:]}\n" \
               f"{self.res.stderr[-2000:]}"


def _spike(elf: Path, **kw) -> Run:
    kw.setdefault("timeout_s", DEFAULT_TIMEOUT_S)
    return Run(rs.run(elf, **kw))


def _bytes_at(e, lo: int, hi: int) -> bytes:
    sec = next(s for s in e.sections if s["addr"] <= lo and hi <= s["addr"] + s["size"] and s["type"] != 8)
    return e.data[sec["off"] + lo - sec["addr"]:sec["off"] + hi - sec["addr"]]


def _const_obs(seed: int) -> np.ndarray:
    """SH_INFER_CONST obs (SPIKE_HOST.md): xorshift32 from (seed << 1) | 1, obs[i] = (x >> 8) * 2^-23 - 1."""
    x = ((seed << 1) | 1) & 0xFFFFFFFF
    out = np.empty(516, np.float64)
    for i in range(516):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = (x >> 8) * 2.0 ** -23 - 1.0
    return out.astype(np.float32)


def _aligned(nbytes: int, align: int = 64) -> np.ndarray:
    raw = np.zeros(nbytes + align, np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + nbytes]


class X86Kernel:
    """The kernel lane's x86 build (Makefile.kernel kernel-x86): the bit-parity reference."""

    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        vp, u32 = ctypes.c_void_p, ctypes.c_uint32
        self.lib.wbc_infer.argtypes = [vp, u32, u32, vp, vp, vp]
        self.lib.wbc_infer.restype = ctypes.c_int
        self.scratch = _aligned(4096)

    def infer(self, blob: bytes, policy: int, variant: int, obs: np.ndarray) -> np.ndarray:
        buf = _aligned(len(blob))
        buf[:] = np.frombuffer(blob, np.uint8)
        obs = np.ascontiguousarray(obs, np.float32).reshape(-1, 516)
        act = np.full((len(obs), 15), np.nan, np.float32)
        for n in range(len(obs)):
            rc = self.lib.wbc_infer(buf.ctypes.data, policy, variant, obs[n].ctypes.data, act[n].ctypes.data,
                                    self.scratch.ctypes.data)
            assert rc == 0, (policy, variant, n, rc)
        return act


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def _fnv32(h: int, v: int) -> int:
    for i in range(4):
        h = ((h ^ ((v >> (8 * i)) & 0xFF)) * 16777619) & 0xFFFFFFFF
    return h


def _crc_expect(image: bytes, off: int, nmax: int) -> tuple[int, int]:
    """spike_host.c crc_report() recomputed with zlib: (digest, chained) for start alignment `off`."""
    src = image[off:off + 4096]
    assert len(src) == 4096
    h = 2166136261
    for n in [*range(nmax + 1), 2112, 4096]:
        h = _fnv32(h, zlib.crc32(src[:n]))
    c, pos, k = 0, 0, 1
    while pos < 4096:
        n = min(k, 4096 - pos)
        c = zlib.crc32(src[pos:pos + n], c)
        pos += n
        k = k % 13 + 1
    return h, c


def _check_crc_lines(run: "Run", image: bytes, slices: int) -> None:
    got = [tuple(int(x) for x in m.groups()) for m in CRC_RE.finditer(run.res.stdout)]
    assert [g[0] for g in got] == list(range(8)), (got, run.describe())
    for off, nmax, sl, digest, chained in got:
        assert nmax == 520 and sl == slices, (off, nmax, sl)
        assert (digest, chained) == _crc_expect(image, off, nmax), off


# --- fixtures ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fw() -> Path:
    """The x280 build (monitor, park stub, stage 0, payloads): the spike builds embed its payloads."""
    _make("x280")
    return FW / "build"


@pytest.fixture(scope="module")
def default_images(fw, tmp_path_factory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("spike_default")
    return {"lim_only": _build_spike(tmp / "lim_only"),
            "x280_regions": _build_spike(tmp / "x280_regions", SPIKE_X280_REGIONS=1),
            # review F1: hart 0's reset handler address left at the E1F stage-0 entry 0x2001_0100; the park stub
            # must reach crt0 anyway (it jumps to the fixed 0x0800_0000)
            "x280_regions_e1f_vectors": _build_spike(tmp / "x280_regions_e1f", SPIKE_X280_REGIONS=1,
                                                     SPIKE_E1F_VECTORS=1)}


@pytest.fixture(scope="module")
def default_runs(default_images) -> dict[str, Run]:
    """lim_only: LIM alone, the plan's memory string. x280_regions: also the 0x2001_0000 page (reset
    handler addresses, GP scratch) and the BEU page, so every warm entry goes through the real park stub."""
    cfgs = {"lim_only": dict(x280_regions=False), "x280_regions": dict(x280_regions=True),
            "x280_regions_e1f_vectors": dict(x280_regions=True)}
    with concurrent.futures.ThreadPoolExecutor(len(cfgs)) as ex:
        futs = {k: ex.submit(_spike, default_images[k] / "monitor_spike.elf", **kw) for k, kw in cfgs.items()}
        return {k: f.result() for k, f in futs.items()}


@pytest.fixture(scope="module")
def x86(tmp_path_factory) -> X86Kernel:
    out = tmp_path_factory.mktemp("kernel_x86")
    _make("-f", "Makefile.kernel", f"WBCK_BUILD={out}", "kernel-x86")
    return X86Kernel(out / "libwbc_mlp.so")


@pytest.fixture(scope="module")
def blobs(tmp_path_factory) -> dict[str, Path]:
    pw = _load("pack_weights", RT / "x280" / "tools" / "pack_weights.py")
    tmp = tmp_path_factory.mktemp("blobs")
    out = {}
    for name, dtype, pols in (("fp16", "fp16", (0, 1)), ("fp32_0", "fp32", (0,)), ("fp32_1", "fp32", (1,))):
        p = tmp / f"{name}.blob"
        p.write_bytes(pw.pack(dtype=dtype, policies=pols))
        out[name] = p
    return out


@pytest.fixture(scope="module")
def logged_obs() -> dict[int, np.ndarray]:
    """N_PER_POLICY logged obs per policy, evenly spread over the run (index = WBC_POLICY_*)."""
    if not Path(OBS516_NPZ).is_file():
        pytest.skip(f"no logged obs516 at {OBS516_NPZ} (set $WBC_OBS516_NPZ; recipe in test_kernel_x86.py)")
    d = np.load(OBS516_NPZ)
    names = ("GR00T-WholeBodyControl-Balance", "GR00T-WholeBodyControl-Walk")
    out = {}
    for p, name in enumerate(names):
        idx = np.flatnonzero(d["policy"] == name)
        assert len(idx) >= N_PER_POLICY, (name, len(idx))
        pick = idx[np.unique(np.linspace(0, len(idx) - 1, N_PER_POLICY).round().astype(int))]
        assert len(pick) == N_PER_POLICY
        out[p] = np.ascontiguousarray(d["obs"][pick], np.float32)
    return out


def _infer_records(d: Path) -> list[bytes]:
    raw = (d / "out.bin").read_bytes() if (d / "out.bin").exists() else b""
    return [raw[i:i + 128] for i in range(0, len(raw) - 127, 128)]


@pytest.fixture(scope="module")
def parity(fw, blobs, logged_obs, tmp_path_factory) -> dict[tuple[int, int], dict]:
    """One spike run per (variant, policy): the default script, then SPIKE_BLOB/SPIKE_OBS/SPIKE_OUT
    (load_blob, infer_const x4, bench x3, infer_file over the logged obs). 10 runs in parallel."""
    tmp = tmp_path_factory.mktemp("spike_parity")
    obs_files = {}
    for p, o in logged_obs.items():
        obs_files[p] = tmp / f"obs_p{p}.bin"
        o.astype("<f4").tofile(obs_files[p])
    cfgs = []
    for v in (V0, V1, V2, V2S, V3):
        for p in (0, 1):
            blob = blobs["fp16"] if v in (V2, V2S, V3) else blobs[f"fp32_{p}"]
            n = N_V2S if v == V2S else N_PER_POLICY
            cfgs.append((v, p, blob, n))
    jobs = {}
    for v, p, blob, n in cfgs:
        d = tmp / f"{VNAME[v]}_p{p}"
        _build_spike(d, SPIKE_BLOB=blob, SPIKE_OBS=obs_files[p], SPIKE_OUT=d / "out.bin", SPIKE_N=n,
                     SPIKE_POLICY=p, SPIKE_VARIANT=v)
        jobs[(v, p)] = (d, blob, n)
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(_spike, d / "monitor_spike.elf", x280_regions=False, timeout_s=PARITY_TIMEOUT_S)
                for k, (d, _, _) in jobs.items()}
        runs = {k: f.result() for k, f in futs.items()}
    out = {}
    for k, (d, blob, n) in jobs.items():
        run = runs[k]
        recs = _infer_records(d)
        out[k] = {"run": run, "blob": blob.read_bytes(), "n": n, "obs": logged_obs[k[1]][:n], "recs": recs,
                  "image": (d / "monitor_spike.bin").read_bytes(),
                  "act": np.array([ml.unpack_l35(r[:64])["action"] for r in recs], np.float32).reshape(-1, 15),
                  "seq_echo": [ml.unpack_l35(r[:64])["seq_echo"] for r in recs],
                  "l1": [ml.unpack_l1(r[64:]) for r in recs]}
    return out


@pytest.fixture(scope="module")
def crc_bitwise(fw, blobs, logged_obs, tmp_path_factory) -> dict:
    """One SPIKE_CRC_BITWISE=1 build (-DMON_CRC_BITWISE: the bitwise mon_crc32 that T-3.2b measured before the
    tables), v2 fp16 Balance on the first N_CRC_CMP obs of parity's (v2, 0) run."""
    tmp = tmp_path_factory.mktemp("spike_crcbit")
    obs = tmp / "obs_p0.bin"
    logged_obs[0][:N_CRC_CMP].astype("<f4").tofile(obs)
    d = tmp / "v2_p0_bitwise"
    _build_spike(d, SPIKE_BLOB=blobs["fp16"], SPIKE_OBS=obs, SPIKE_OUT=d / "out.bin", SPIKE_N=N_CRC_CMP,
                 SPIKE_POLICY=0, SPIKE_VARIANT=V2, SPIKE_CRC_BITWISE=1)
    run = _spike(d / "monitor_spike.elf", x280_regions=False, timeout_s=PARITY_TIMEOUT_S)
    return {"run": run, "recs": _infer_records(d), "image": (d / "monitor_spike.bin").read_bytes()}


# --- static: the spike build is the X280 build minus the X280-only CSRs -----------------------------------

def test_spike_image_has_no_x280_only_csrs(fw, default_images):
    """Spike traps 0x350-0x353, 0x7c1 and mnret (spike lane, t_csr.c), so SPIKE_HOST must emulate every one
    of them; the X280 image must still contain the real re-arm and chicken-bit writes."""
    for name, d in default_images.items():
        e = ci.Elf(d / "monitor_spike.elf")
        used = set(ci.csr_accesses(e))
        assert not used & (X280_ONLY_CSRS | SMRNMI_CSRS), (name, sorted(hex(c) for c in used))
        assert ci.MNRET not in ci.words(e), name
        # HTIF words and everything the image allocates stay below the mailbox (spike lane note)
        assert ml.LIM_BASE <= e.sym("tohost") < e.sym("fromhost") < ml.MB_BASE, name
        assert all(ml.LIM_BASE <= s["addr"] and s["addr"] + s["size"] <= ml.MB_BASE
                   for s in e.alloc_sections()), name
        assert not e.undefined(), (name, e.undefined())
        # the park stub the runner installs (x280_regions) is build/park_stub.bin, byte for byte
        park = _bytes_at(e, e.sym("sh_bin_park"), e.sym("sh_bin_park_end"))
        assert park == (fw / "park_stub.bin").read_bytes(), name
    x = ci.Elf(fw / "monitor.elf")
    assert {ci.REARM, ci.CHICKEN} <= ci.words(x)
    assert 0x353 in ci.csr_accesses(x) and 0x7C1 in ci.csr_accesses(x)


# --- the default script (SPIKE_HOST.md "Default script and the expected results") ---------------------------

def _check_default(run: Run, image_bin: bytes) -> None:
    assert run.res.ok, run.describe()
    assert run.end == (32, 0, 12), run.describe()   # 32 steps, 0 timeouts, 1 cold + 11 warm entries
    by = run.op
    # boot: identity lines (device_session_notes: what x280d waits for after the release)
    b = run.boot
    assert b is not None, run.describe()
    assert b["l1"]["fw_magic"] == ml.MAGIC and b["l1"]["fw_ver"] == ml.FW_VER
    assert b["l1"]["state"] == ST["IDLE"] and b["l1"]["mhartid"] == 0 and b["l1"]["vlenb"] == 64
    assert b["l36"]["entry_flags"] & ml.ENTRY_FLAG["STUB_KERNEL"] == 0, "stub kernel linked"
    assert b["l36"]["variants_built"] == 0b11111, hex(b["l36"]["variants_built"])
    assert b["l36"]["mtime_hz"] == 10_000_000 and b["l36"]["misa"] >> 21 & 1
    assert b["l39"]["kind"] == ml.L39_KIND["ENTRY"] and b["l39"]["entry_kind"] == 0   # cold
    # every command step completed under its own seq, in order; L38 pokes issue no command
    seqs = [s["seq"] for s in run.steps if s["op"] not in ("kmd_gone", "attach_clear")]
    assert seqs == list(range(1, len(seqs) + 1)), seqs
    for s in run.steps:
        assert s["res"] in ("done", "reentered"), (s["op"], s["res"])
        assert s["l1"]["seq_done"] == s["seq"], (s["op"], s["l1"]["seq_done"], s["seq"])

    assert by("nop")[0]["l1"]["state"] == ST["IDLE"]
    # SELFTEST: every LIM line read, V present, VLEN 512, BEU quiet
    st = run.one("selftest")
    l39 = st["l39"]
    assert st["l1"]["state"] == ST["IDLE"]
    assert (l39["kind"], l39["seq"]) == (ml.L39_KIND["SELFTEST"], st["seq"])
    assert l39["lines"] == ml.LIM_SIZE // ml.LINE == 30720
    assert l39["misa_v"] == 1 and l39["vlenb"] == 64
    assert l39["beu_cause"] == 0 and l39["beu_accrued_before"] == 0 and l39["beu_accrued_after"] == 0
    assert st["l36"]["beu_accrued"] == 0
    assert abs(l39["cyc_per_mtime_x1000"] - SPIKE_CYC_PER_MTIME * 1000) <= 2000, l39["cyc_per_mtime_x1000"]
    # LOAD / FENCE_I / CALL: a CALL before FENCE_I is refused; then the checksum payload 10/10
    assert run.one("load_checksum")["l1"]["state"] == ST["IDLE"]
    cb = run.one("call_before_fence")["l1"]
    assert (cb["state"], cb["mcause"], cb["mtval"]) == (ST["FAULT"], ml.ERR["NOCODE"], 1)
    calls = by("call_checksum")
    want = zlib.crc32(image_bin[:4096])
    assert [c["sub"] for c in calls] == list(range(10))
    for c in calls:
        assert c["l1"]["state"] == ST["IDLE"], c["l1"]
        assert (c["l39"]["kind"], c["l39"]["seq"], c["l39"]["ret"]) == (ml.L39_KIND["CALL"], c["seq"], want)
    # illegal instruction in a payload: the dump, then the monitor keeps serving commands
    ill = run.one("call_illegal")
    l1 = ill["l1"]
    assert (l1["state"], l1["mcause"], l1["mepc"], l1["mtval"]) == (ST["FAULT"], 2, PAY + 0x1000, 0xC0001073), l1
    before = run.steps[run.steps.index(ill) - 1]["l36"]["n_traps"]
    assert ill["l36"]["n_traps"] == before + 1
    after = run.one("nop_after_trap")
    assert after["l1"]["state"] == ST["IDLE"] and after["l1"]["mcause"] == 0
    # LIM microbench: every test runs to completion
    lb_expect = {  # op: (test, status, iters, bytes)
        "lb_ptr_chase": (1, 0, 2000, 16000), "lb_stream": (2, 0, 2048, 16384),
        "lb_vle32_m1": (3, 0, 16384 // 64, 16384), "lb_vle32_m4": (4, 0, 16384 // 256, 16384),
        "lb_vle32_m8": (5, 0, 16384 // 512, 16384), "lb_exec_loop": (6, 0, 200, 200 * 256),
        "lb_zero_dev_denied": (7, 4, 0, 0)}
    for op, (test, status, iters, nbytes) in lb_expect.items():
        s = run.one(op)
        assert s["l1"]["state"] == ST["IDLE"] and s["l39"]["kind"] == ml.L39_KIND["CALL"], (op, s["l1"])
        r = ml.unpack_lb_result(s["l39"]["data"])
        assert (r["test"], r["status"], r["iters"], r["bytes"]) == (test, status, iters, nbytes), (op, r)
        if status == 0:
            assert r["cycles"] > 0 and s["l39"]["ret"] == r["cycles"], (op, r)
    ex = ml.unpack_lb_result(run.one("lb_exec_loop")["l39"]["data"])
    assert 1.0 <= ex["cycles"] / (ex["iters"] * 64) < 1.2, ex        # spike: one instruction per cycle
    mr = ml.unpack_lb_result(run.one("lb_mtime_ratio")["l39"]["data"])
    assert (mr["test"], mr["status"]) == (8, 0) and mr["iters"] >= 10_000, mr
    assert abs(mr["bytes"] - SPIKE_CYC_PER_MTIME * 1000) <= 2000, mr   # 16000 expected on silicon only
    # PARK: cooperative, then the (emulated) host RNMI -> park path -> crt0 -> warm entry
    pk = run.one("park")
    assert pk["res"] == "reentered" and pk["cyc"] == 0
    assert (pk["l1"]["state"], pk["l1"]["mcause"]) == (ST["REENTERED"], RNMI_CAUSE), pk["l1"]
    assert pk["l36"]["n_reentry"] == 1 and pk["l36"]["entry_flags"] & ml.ENTRY_FLAG["WARM"]
    ent = pk["l39"]
    assert ent["kind"] == ml.L39_KIND["ENTRY"] and ent["entry_kind"] == 1 and ent["mncause"] == RNMI_CAUSE
    assert ent["mnstatus"] & 8 == 0                                  # NMIE cleared by the RNMI
    assert run.one("nop_after_park")["l1"]["state"] == ST["IDLE"]
    # hung payload -> RNMI -> warm entry -> CALL again, 10/10; NMIE re-armed every time (else exit 0x7F)
    hangs = by("call_hang")
    assert [h["sub"] for h in hangs] == list(range(10))
    for i, h in enumerate(hangs):
        assert h["res"] == "reentered"
        assert (h["l1"]["state"], h["l1"]["mcause"], h["l1"]["mepc"]) == (ST["REENTERED"], RNMI_CAUSE, PAY + 0x2000)
        assert h["l36"]["n_reentry"] == 2 + i
    assert run.one("nop_after_hang")["l1"]["state"] == ST["IDLE"]
    # obs CRC fault, host gone (seq adopted, not run), unknown command
    bad = run.one("infer_bad_obs_crc")
    assert (bad["l1"]["state"], bad["l1"]["mcause"], bad["l1"]["mtval"]) == (ST["FAULT"], ml.ERR["CRC"], 2)
    assert bad["l36"]["n_crc_fail"] == 1 and bad["l36"]["n_crc_retry"] >= 1
    assert run.one("kmd_gone")["l1"]["state"] == ST["HOST_GONE"]
    gone = run.one("nop_while_gone")
    assert gone["l1"]["state"] == ST["HOST_GONE"] and gone["l36"]["n_host_gone"] == 1
    assert run.one("attach_clear")["l1"]["state"] == ST["IDLE"]
    assert run.one("nop_after_attach")["l1"]["state"] == ST["IDLE"]
    unk = run.one("unknown_cmd")["l1"]
    assert (unk["state"], unk["mcause"], unk["mtval"]) == (ST["FAULT"], ml.ERR["CMD"], 9)
    final = run.steps[-1]["l36"]
    assert final["n_traps"] == 1 and final["n_reentry"] == 11 and final["n_overrun"] == 0


@pytest.mark.parametrize("cfg", ["lim_only", "x280_regions", "x280_regions_e1f_vectors"])
def test_default_script(cfg, default_images, default_runs):
    image = (default_images[cfg] / "monitor_spike.bin").read_bytes()
    _check_default(default_runs[cfg], image)


@pytest.mark.parametrize("cfg", ["lim_only", "x280_regions", "x280_regions_e1f_vectors"])
def test_crc32_on_spike_is_zlib(cfg, default_images, default_runs):
    """The hart's own build of mon_crc32 (slicing-by-8 over the .rodata tables) equals zlib for every length
    0..520, 2,112 and 4,096 from each of the 8 start alignments, and chained over 1..13 B pieces."""
    run = default_runs[cfg]
    assert run.res.ok, run.describe()
    _check_crc_lines(run, (default_images[cfg] / "monitor_spike.bin").read_bytes(), slices=8)


def test_trap_dump_matches_spike(default_images):
    """Spike's own exception record agrees with the monitor's dump, and it is the only exception of the
    whole default script: an X280-only CSR or mnret reaching spike would be another illegal instruction."""
    run = _spike(default_images["lim_only"] / "monitor_spike.elf", x280_regions=False, trace_traps=True)
    assert run.res.ok and run.end == (32, 0, 12), run.describe()
    exc = [t for t in run.res.traps if not t["name"].startswith("interrupt")]
    assert exc == [{"hart": 0, "kind": "exception", "name": "trap_illegal_instruction", "epc": PAY + 0x1000,
                    "tval": 0xC0001073}], exc
    l1 = run.one("call_illegal")["l1"]
    assert (l1["mcause"], l1["mepc"], l1["mtval"]) == (2, exc[0]["epc"], exc[0]["tval"])
    timer = [t for t in run.res.traps if t["name"] == "interrupt #7"]   # the 10 emulated host RNMIs
    assert len(timer) == 10 and {t["epc"] for t in timer} == {PAY + 0x2000}, timer


# --- stage 0 (E1F contingency; built and checked, never run on silicon without DP3 approval) ---------------

def test_stage0_stub_runs_to_completion(fw, default_images):
    elf = default_images["lim_only"] / "stage0_spike.elf"
    e = ci.Elf(elf)
    sec = next(s for s in e.sections if s["name"] == ".stage0")
    assert sec["addr"] == GP_SCRATCH and e.sym("stage0_start") == GP_SCRATCH
    assert e.data[sec["off"]:sec["off"] + sec["size"]] == (fw / "stage0.bin").read_bytes(), "stage 0 bytes differ"
    res = rs.run(elf, x280_regions=False, extra_regions=((0x2001_0000, 0x1000),), timeout_s=120)
    assert res.ok, (res.exit_code, res.stdout, res.stderr[-2000:])   # 1 = non-zero LIM word, 2 = bad done word


# --- INFER parity on the logged obs ----------------------------------------------------------------------

def _check_infer_run(k, r) -> None:
    run = r["run"]
    assert run.res.ok, (k, run.describe())
    assert run.end == (36, 0, 12), (k, run.end)
    assert run.one("load_blob")["l1"]["state"] == ST["IDLE"], (k, run.one("load_blob")["l1"])
    f = run.one("infer_file")
    assert f["res"] == "done" and f["sub"] == r["n"], (k, f["res"], f["sub"])
    assert len(r["l1"]) == r["n"] and r["act"].shape == (r["n"], 15)
    seq0 = run.one("bench")["seq"] + 1
    for i, (l1, echo) in enumerate(zip(r["l1"], r["seq_echo"])):
        assert l1["state"] == ST["IDLE"] and l1["mcause"] == 0, (k, i, l1)
        assert l1["seq_done"] == echo == seq0 + i, (k, i, l1["seq_done"], echo)
        assert l1["cycles_last"] > 0


def _ref(x86: X86Kernel, r: dict, v: int, p: int, obs: np.ndarray) -> np.ndarray:
    return x86.infer(r["blob"], p, V0 if v in (V0, V1) else V2, obs)   # x86 has no v1; v2s == v2 there


@pytest.mark.parametrize("variant", [V0, V1, V2, V2S, V3], ids=lambda v: VNAME[v])
def test_infer_parity(variant, parity, x86):
    """v0, v2s and v3 bit-identical to x86 (v0 / v2 / v2); v1 and v2 within 1e-6 (PLAN.md T-3.2b); v3 also equal to
    spike's v2 on the same obs, bit for bit."""
    total, identical, worst = 0, 0, 0.0
    for p in (0, 1):
        r = parity[(variant, p)]
        _check_infer_run((VNAME[variant], p), r)
        ref = _ref(x86, r, variant, p, r["obs"])
        got = r["act"]
        assert np.isfinite(got).all()
        same = (_bits(got) == _bits(ref)).all(axis=1)
        identical += int(same.sum())
        total += len(got)
        worst = max(worst, float(np.abs(got.astype(np.float64) - ref).max()))
        if variant in (V0, V2S, V3):
            assert same.all(), (VNAME[variant], p, int((~same).sum()), np.flatnonzero(~same)[:10])
        else:
            assert worst <= TOL_V1_V2, (VNAME[variant], p, worst)
        if variant == V3:
            assert np.array_equal(_bits(got), _bits(parity[(V2, p)]["act"][:len(got)])), p
    print(f"\nT-3.2b INFER {VNAME[variant]}: {total} obs, {identical} bit-identical to x86, max|d| {worst:.3g}")
    assert total == 2 * (N_V2S if variant == V2S else N_PER_POLICY)


@pytest.mark.parametrize("variant", [V0, V1, V2, V2S, V3], ids=lambda v: VNAME[v])
def test_infer_const_and_bench(variant, parity, x86):
    """The default script's infer_const x4 (xorshift obs) and BENCH x3 on the same image."""
    for p in (0, 1):
        r = parity[(variant, p)]
        run = r["run"]
        consts = run.op("infer_const")
        assert [c["sub"] for c in consts] == [0, 1, 2, 3]
        obs = np.stack([_const_obs(1 + c["sub"]) for c in consts])
        ref = _ref(x86, r, variant, p, obs)
        for c, want in zip(consts, ref):
            got = np.array(c["l35"]["action"], np.float32)
            assert c["l1"]["state"] == ST["IDLE"] and c["l35"]["seq_echo"] == c["seq"]
            if variant in (V0, V2S, V3):
                assert (_bits(got) == _bits(want)).all(), (VNAME[variant], p, c["sub"])
            else:
                assert np.abs(got.astype(np.float64) - want).max() <= TOL_V1_V2
        bench = run.one("bench")
        b = bench["l39"]
        assert bench["l1"]["state"] == ST["IDLE"] and b["kind"] == ml.L39_KIND["BENCH"], bench["l1"]
        assert (b["iters"], b["nondet"], b["rc"], b["variant"], b["policy"]) == (3, 0, 0, variant, p), b
        assert b["act_crc32"] == ml.crc32(bench["l35_raw"][:60])
        assert bench["l35"]["action"] == consts[-1]["l35"]["action"]   # BENCH reuses the last obs lines
        assert b["cyc_min"] <= b["cyc_max"] and b["cyc_sum"] >= 3 * b["cyc_min"]


def test_instruction_counts_per_infer(parity):
    """Spike instructions per INFER (mcycle == minstret): the kernel call alone (L1.cycles_last) and the
    whole command on the hart (SPIKE_HOST cyc: obs copy + CRC, kernel, L35/L1 writes). Printed, and with
    $T32B_REPORT written as JSON; the kernel count must stay within KERNEL_DRIFT of the R33 table's."""
    report = {}
    lines = [f"{'variant':8} {'policy':6} {'n':>4} {'kernel min':>11} {'median':>11} {'max':>11} "
             f"{'command mean':>13} {'overhead':>9} {'load_blob':>11}"]
    for (v, p), r in sorted(parity.items()):
        run = r["run"]
        if not run.res.ok:
            pytest.fail(f"{VNAME[v]} p{p}: spike run failed\n{run.describe()}")
        k = np.array([l1["cycles_last"] for l1 in r["l1"]], np.int64)
        f = run.one("infer_file")
        cmd = f["cyc_sum"] / f["sub"]
        row = {"variant": VNAME[v], "policy": p, "n": int(len(k)), "kernel_min": int(k.min()),
               "kernel_median": float(np.median(k)), "kernel_max": int(k.max()), "kernel_mean": float(k.mean()),
               "command_mean": float(cmd), "overhead_mean": float(cmd - k.mean()),
               "load_blob": int(run.one("load_blob")["cyc"]), "bench_cyc_min": int(run.one("bench")["l39"]["cyc_min"])}
        report[f"{VNAME[v]}_p{p}"] = row
        lines.append(f"{VNAME[v]:8} {p:6} {row['n']:4} {row['kernel_min']:11,} {row['kernel_median']:11,.0f} "
                     f"{row['kernel_max']:11,} {row['command_mean']:13,.0f} {row['overhead_mean']:9,.0f} "
                     f"{row['load_blob']:11,}")
        ref = KERNEL_LANE_INSNS[v]
        assert abs(row["kernel_median"] - ref) <= KERNEL_DRIFT * ref, (VNAME[v], p, row["kernel_median"], ref)
        assert 0 < row["overhead_mean"] < 40_000, row   # obs copy + table CRC (bitwise: 124,305); see the report
    print("\nT-3.2b spike instructions per INFER (VLEN 512; spike cycles are instructions, not X280 cycles)")
    print("\n".join(lines))
    for p in (0, 1):
        if f"v3_p{p}" in report and f"v2_p{p}" in report:
            print(f"v3/v2 kernel instructions, policy {p}: "
                  f"{report[f'v3_p{p}']['kernel_median'] / report[f'v2_p{p}']['kernel_median']:.3f}")
    if os.environ.get("T32B_REPORT"):
        Path(os.environ["T32B_REPORT"]).write_text(json.dumps(report, indent=1) + "\n")


def test_crc_bitwise_build_matches_table_build(parity, crc_bitwise):
    """-DMON_CRC_BITWISE and the slicing-by-8 default are the same function: the bitwise build's CRC lines equal
    zlib, and on the same blob and obs its INFER records (L35 bytes; L1 state, mcause, seq_done, kernel
    cycles_last), its BENCH act_crc32 and its load_blob state equal the table build's. Its monitor overhead per
    INFER is the 124,305 instructions measured before the tables; the table build's is printed beside it
    ($T32B_CRC_REPORT writes both)."""
    tab, bit = parity[(V2, 0)], crc_bitwise
    run = bit["run"]
    assert run.res.ok and run.end == (36, 0, 12), run.describe()
    _check_crc_lines(run, bit["image"], slices=0)
    _check_crc_lines(tab["run"], tab["image"], slices=8)
    assert len(bit["recs"]) == N_CRC_CMP <= len(tab["recs"])
    for i, (rb, rt) in enumerate(zip(bit["recs"], tab["recs"])):
        assert rb[:64] == rt[:64], i                                   # L35: action bits and seq_echo
        lb, lt = ml.unpack_l1(rb[64:]), ml.unpack_l1(rt[64:])
        assert {k: lb[k] for k in ("state", "mcause", "mtval", "seq_done", "cycles_last")} == \
               {k: lt[k] for k in ("state", "mcause", "mtval", "seq_done", "cycles_last")}, i
    assert run.one("load_blob")["l1"]["state"] == tab["run"].one("load_blob")["l1"]["state"] == ST["IDLE"]
    bb, bt = run.one("bench"), tab["run"].one("bench")
    assert bb["l39"]["act_crc32"] == bt["l39"]["act_crc32"] == ml.crc32(bb["l35_raw"][:60])
    for a, b in zip(run.op("infer_const"), tab["run"].op("infer_const")):
        assert a["l35_raw"] == b["l35_raw"]
    f_bit, f_tab = run.one("infer_file"), tab["run"].one("infer_file")
    k_bit = np.mean([ml.unpack_l1(r[64:])["cycles_last"] for r in bit["recs"]])
    k_tab = np.mean([ml.unpack_l1(r[64:])["cycles_last"] for r in tab["recs"]])
    rep = {"n_bitwise": N_CRC_CMP, "n_table": tab["n"],
           "overhead_bitwise": float(f_bit["cyc_sum"] / f_bit["sub"] - k_bit),
           "overhead_table": float(f_tab["cyc_sum"] / f_tab["sub"] - k_tab),
           "load_blob_bitwise": int(run.one("load_blob")["cyc"]),
           "load_blob_table": int(tab["run"].one("load_blob")["cyc"])}
    print(f"\nT-3.2b monitor instructions per INFER (v2 p0): bitwise CRC {rep['overhead_bitwise']:,.0f}, "
          f"slicing-by-8 {rep['overhead_table']:,.0f}; load_blob {rep['load_blob_bitwise']:,} -> "
          f"{rep['load_blob_table']:,}")
    assert abs(rep["overhead_bitwise"] - MON_BITWISE_OVERHEAD) <= 0.01 * MON_BITWISE_OVERHEAD, rep
    assert rep["overhead_table"] < rep["overhead_bitwise"] / 4, rep
    if os.environ.get("T32B_CRC_REPORT"):
        Path(os.environ["T32B_CRC_REPORT"]).write_text(json.dumps(rep, indent=1) + "\n")


def _edge_obs(n_base: int = 8) -> np.ndarray:
    """Crafted obs for v3's guard: rows whose layer inputs reach |x| >= 2^16 (v3 then computes that layer again with
    v2's ops), inf / NaN, values just below 2^16, subnormals and signed zeros; then n_base plain N(0,1) rows."""
    f32 = np.float32
    rng = np.random.default_rng(7)
    o = rng.standard_normal((16 + n_base, 516)).astype(f32)
    edge = [f32(65536.0), np.nextafter(f32(65536.0), f32(0)), f32(-1e5), f32(np.inf), f32(np.nan), f32(1e-45)]
    for k, v in enumerate(edge):
        o[k, (53 * k) % 516] = v
    o[6] *= 3e4                     # later layers' inputs past 2^16 only
    o[7] = 1e-40
    o[8] = -0.0
    o[9, 430:] = 7e4                # the newest frame feeds the actor's layer 3 directly
    o[10] *= 1e6
    o[11, :] = np.nextafter(f32(65536.0), f32(0))
    o[12, 100] = f32(-3.0e38)
    o[13] *= 1e-30
    o[14, 431] = f32(np.inf)
    o[15, 515] = f32(-65536.0)
    return o


@pytest.fixture(scope="module")
def adversarial(fw, blobs, tmp_path_factory) -> dict[tuple[int, int], dict]:
    """v2 and v3, both policies, on _edge_obs: 4 runs in parallel."""
    tmp = tmp_path_factory.mktemp("spike_edge")
    obs = _edge_obs()
    f = tmp / "edge_obs.bin"
    obs.astype("<f4").tofile(f)
    jobs = {}
    for v in (V2, V3):
        for p in (0, 1):
            d = tmp / f"{VNAME[v]}_p{p}"
            _build_spike(d, SPIKE_BLOB=blobs["fp16"], SPIKE_OBS=f, SPIKE_OUT=d / "out.bin", SPIKE_N=len(obs),
                         SPIKE_POLICY=p, SPIKE_VARIANT=v)
            jobs[(v, p)] = d
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(_spike, d / "monitor_spike.elf", x280_regions=False, timeout_s=PARITY_TIMEOUT_S)
                for k, d in jobs.items()}
        runs = {k: f.result() for k, f in futs.items()}
    out = {}
    for k, d in jobs.items():
        recs = _infer_records(d)
        out[k] = {"run": runs[k], "obs": obs, "blob": blobs["fp16"].read_bytes(),
                  "act": np.array([ml.unpack_l35(r[:64])["action"] for r in recs], np.float32).reshape(-1, 15),
                  "l1": [ml.unpack_l1(r[64:]) for r in recs]}
    return out


def test_v3_edge_inputs_on_spike(adversarial, x86):
    """On rows that take v3's fallback or sit on its edge, v3 on the hart equals v2 on the hart and x86 v2 bit for
    bit (RISC-V NaNs are canonical, so NaN rows compare too); the fallback rows cost more instructions than plain
    rows, rows that stay below 2^16 (one input at 65535.996, subnormals, -0) do not."""
    for p in (0, 1):
        v2, v3 = adversarial[(V2, p)], adversarial[(V3, p)]
        for r in (v2, v3):
            assert r["run"].res.ok and r["run"].one("infer_file")["sub"] == len(r["obs"]), r["run"].describe()
            assert all(l1["state"] == ST["IDLE"] and l1["mcause"] == 0 for l1 in r["l1"])
        assert np.array_equal(_bits(v3["act"]), _bits(v2["act"])), p
        ref = x86.infer(v3["blob"], p, V2, v3["obs"])
        fin = np.isfinite(ref).all(axis=1)
        assert np.array_equal(_bits(v3["act"][fin]), _bits(ref[fin])), p
        assert (~fin).any() and np.isnan(v3["act"][~fin]).any()
        k = np.array([l1["cycles_last"] for l1 in v3["l1"]], np.int64)
        plain = int(np.median(k[16:]))
        assert (k[16:] == plain).all() and (k[[1, 5, 7, 8, 13]] == plain).all()   # |x| < 2^16 everywhere
        assert (k[[0, 2, 3, 4, 6, 9, 10, 15]] > plain).all(), k                 # fallback in one or more layers
        print(f"\nv3 edge rows, policy {p}: plain {plain:,} instructions, fallback rows {sorted(set(k[k > plain]))}")


# --- harts 1-3 on spike -p4 (x280/fw/mh.h, mh.c, mh_bench.c; SPIKE_HOST.md "Multi-hart script") ------------------------

MH_RE = re.compile(r"^SH mh step=(\d+) sub=(\d+) C=([0-9a-f]{128}) S1=([0-9a-f]{128}) A1=([0-9a-f]{128}) "
                   r"S2=([0-9a-f]{128}) A2=([0-9a-f]{128}) S3=([0-9a-f]{128}) A3=([0-9a-f]{128})$", re.M)
MHB_RE = re.compile(r"^SH mhb step=(\d+) sub=(\d+) J=([0-9a-f]{128}) R0=([0-9a-f]{128}) R1=([0-9a-f]{128}) "
                    r"R2=([0-9a-f]{128}) R3=([0-9a-f]{128})$", re.M)
ADOPT_RE = re.compile(r"^SH mhadopt step=(\d+) mask=(\d+) pre_ready=(\d+) ready=(\d+) polls=(\d+)$", re.M)
MH_WS, MH_WS_LEN, MH_WIN = 0x0802_0000, 0x1_0000, 100      # spike_script_mh.c
SPIKE_ROUND_TICKS = 50   # every hart runs 5000 instructions, then mtime + 50 (riscv-isa-sim sim.cc:370-382)


def _mh_records(run: Run) -> dict:
    """{(step, sub): {"mh": {ctl, stat{h}, aux{h}}, "mhb": {job, res{h}}}} from the extra SH mh / SH mhb lines."""
    out: dict = {}
    for m in MH_RE.finditer(run.res.stdout):
        key = (int(m[1]), int(m[2]))
        out.setdefault(key, {})["mh"] = {"ctl": ml.unpack_mh_ctl(bytes.fromhex(m[3])),
                                         "stat": {h: ml.unpack_mh_stat(bytes.fromhex(m[2 + 2 * h])) for h in (1, 2, 3)},
                                         "aux": {h: ml.unpack_mh_aux(bytes.fromhex(m[3 + 2 * h])) for h in (1, 2, 3)}}
    for m in MHB_RE.finditer(run.res.stdout):
        key = (int(m[1]), int(m[2]))
        out.setdefault(key, {})["mhb"] = {"job": ml.unpack_mhb_job(bytes.fromhex(m[3])),
                                          "res": {h: ml.unpack_mhb_res(bytes.fromhex(m[4 + h])) for h in range(4)}}
    return out


def _xorshift_ws(seed: int, n: int) -> bytes:
    """SH_LOAD_PATTERN's working set: xorshift64 (13, 7, 17) from seed | 1, one u64 per step."""
    x, words = (seed | 1) & 0xFFFF_FFFF_FFFF_FFFF, []
    for _ in range(n // 8):
        x ^= (x << 13) & 0xFFFF_FFFF_FFFF_FFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFF_FFFF_FFFF_FFFF
        words.append(x)
    return struct.pack(f"<{len(words)}Q", *words)


@pytest.fixture(scope="module")
def mh_runs(fw, tmp_path_factory) -> dict:
    """The multi-hart script on spike -p4 (X280 regions), and its SPIKE_MH_AUDIT subset under `-l --log-commits`."""
    tmp = tmp_path_factory.mktemp("spike_mh")
    full = _build_spike(tmp / "mh", SPIKE_X280_REGIONS=1, SPIKE_MH=1, SPIKE_SCRIPT="spike_script_mh.c")
    audit = _build_spike(tmp / "mh_audit", SPIKE_X280_REGIONS=1, SPIKE_MH=1, SPIKE_MH_AUDIT=1,
                         SPIKE_SCRIPT="spike_script_mh.c")
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        f1 = ex.submit(_spike, full / "monitor_spike.elf", x280_regions=True, harts=4, timeout_s=PARITY_TIMEOUT_S)
        f2 = ex.submit(_spike, audit / "monitor_spike.elf", x280_regions=True, harts=4, timeout_s=PARITY_TIMEOUT_S,
                       audit_stores=True)
        return {"full": f1.result(), "audit": f2.result(), "full_elf": full / "monitor_spike.elf",
                "audit_elf": audit / "monitor_spike.elf"}


@pytest.mark.parametrize("cfg", ["lim_only", "x280_regions"])
def test_default_script_on_four_harts_is_bit_identical(cfg, default_images, default_runs):
    """The flag-off regression: with -p4 the default script's hart-0 records (every SH line, cycle counts included) are
    those of -p1; harts 1-3 sit in crt0's cold wfi (lim_only: mie 0; regions: the emulated RNMI's wfi loop) and
    nothing wakes them."""
    one = default_runs[cfg]
    four = _spike(default_images[cfg] / "monitor_spike.elf", x280_regions=cfg != "lim_only", harts=4)
    assert four.res.ok and four.end == one.end == (32, 0, 12), four.describe()
    assert four.res.stdout == one.res.stdout


def test_mh_adoption_and_ping_on_spike(mh_runs):
    run = mh_runs["full"]
    assert run.res.ok and run.end is not None and run.end[1] == 0, run.describe()
    rec = _mh_records(run)
    pre = run.one("mh_ping_unadopted")
    l39 = ml.unpack_l39_mh(pre["l39"]["data"])
    assert pre["l39"]["kind"] == ml.L39_KIND["MH"] and (l39["done"], l39["ready"], l39["mask"]) == (0, 1, 0xE)
    assert pre["l36"]["mh"] == ml.MH_CAP | 1
    assert all(st["magic"] == 0 for st in rec[(pre["step"], 0)]["mh"]["stat"].values())   # no hart came up
    ad = ADOPT_RE.search(run.res.stdout)
    assert ad and (int(ad[2]), int(ad[3]), int(ad[4])) == (0xE, 1, 0xF), ad   # spinning on their bits, then ready
    mh = rec[(int(ad[1]), 0)]["mh"]
    e = ci.Elf(mh_runs["full_elf"])
    for h in (1, 2, 3):
        st = mh["stat"][h]
        assert (st["magic"], st["hartid"], st["state"], st["entries"]) == (ml.MH_STAT_MAGIC, h, ml.MH_STATE["IDLE"], 1)
        assert st["epoch"] == mh["ctl"]["epoch"] and mh["ctl"]["magic"] == ml.MH_CTL_MAGIC
        assert st["mncause"] == RNMI_CAUSE and e.sym("mh_sec_rnmi_exc") is not None
        assert st["mode"] == ml.MH_MODE["WFI"] and st["mip"] & ml.MIP_MSIP   # the self-IPI showed up and woke wfi
        assert st["traps"] == 0
    ping = run.one("mh_ping")
    p = ml.unpack_l39_mh(ping["l39"]["data"])
    assert (p["done"], p["ready"], p["trapped"]) == (0xE, 0xF, 0) and p["gen"] == 2
    assert all(w <= 2 * SPIKE_ROUND_TICKS for w in p["wake_ticks"]), p   # picked up within two spike rounds
    assert ping["l36"]["mh"] == ml.MH_CAP | 0xF and ping["l1"]["state"] == ST["IDLE"]
    after = rec[(ping["step"], 0)]["mh"]
    assert all(after["stat"][h]["done_gen"] == 2 and after["aux"][h]["gen_seen"] == 2 for h in (1, 2, 3))


def _check_mhb_call(step: dict, recs: dict, image: bytes) -> dict:
    sm = ml.unpack_mhb_summary(step["l39"]["data"])
    assert step["res"] == "done" and step["l1"]["state"] == ST["IDLE"], step["l1"]
    assert sm["status"] == 0 and sm["tag"] == step["seq"], sm
    mhb = recs[(step["step"], step["sub"])]["mhb"]
    job = mhb["job"]
    assert sm["done"] == sm["mask"] == job["mask"], (sm, job)
    for h in range(4):
        if not (sm["mask"] >> h) & 1:
            continue
        r = mhb["res"][h]
        k = job["kernel0"] if h == 0 else job["kernel"]
        assert (r["hart"], r["tag"], r["status"], r["kernel"]) == (h, sm["tag"], 0, k), r
        assert r["accesses"] > 0 and r["accesses"] % ml.MHB_CHUNK[k] == 0, r
        assert r["bytes"] == r["accesses"] * ml.MHB_BPA[k], r
        assert r["t0"] >= job["t_start"] and r["t0"] + r["dt"] >= job["t_end"], (r, job)
        want = ml.mhb_sink(k, job["pattern"], job["kbit"], h, job["ws_base"], job["ws_len"], r["accesses"],
                           lambda a, n: image[a - MH_WS:a - MH_WS + n])
        assert r["sink"] == want, (k, job["pattern"], h, r["sink"], want)
    assert sm["start_skew"] <= 2 * SPIKE_ROUND_TICKS and sm["bytes"] == sum(
        mhb["res"][h]["bytes"] for h in range(4) if (sm["mask"] >> h) & 1)
    return sm


def test_mh_bench_kernels_patterns_masks_on_spike(mh_runs):
    """Every kernel x pattern x mask {0x1, 0x3, 0xF, 0xE}, the P4 sweep and K3 under K6: status OK, done = mask, each
    result line current (tag) with accesses a multiple of its chunk, bytes = accesses x bytes per access, the sink
    equal to the host's recomputation over the xorshift working set, every hart started at or after t_start."""
    run = mh_runs["full"]
    assert run.res.ok, run.describe()
    recs = _mh_records(run)
    ws = _xorshift_ws(0x5EED, MH_WS_LEN)
    sweep = run.op("mhb_sweep")
    assert len(sweep) == 12 * 4 * 4
    seen = set()
    for s in sweep:
        sm = _check_mhb_call(s, recs, ws)
        seen.add((sm["kernel"] & 0xFF, recs[(s["step"], s["sub"])]["mhb"]["job"]["pattern"], sm["mask"]))
    assert seen == {(k, p, m) for k in range(1, 13) for p in range(4) for m in (0x1, 0x3, 0xF, 0xE)}
    p4 = run.op("mhb_p4")
    assert [recs[(s["step"], s["sub"])]["mhb"]["job"]["kbit"] for s in p4] == list(range(6, 13))
    for s in p4:
        _check_mhb_call(s, recs, ws)
    ll = run.one("mhb_loaded_latency")
    sm = _check_mhb_call(ll, recs, ws)
    job = recs[(ll["step"], 0)]["mhb"]["job"]
    assert (job["kernel"], job["kernel0"]) == (ml.MHB_K["VLE8_M2"], ml.MHB_K["CHAIN"]) and sm["done"] == 0xF
    bad = ml.unpack_mhb_summary(run.one("mhb_bad_range")["l39"]["data"])
    assert (bad["status"], bad["done"]) == (ml.MHB_STATUS["ERANGE"], 0)


def test_mh_background_job_and_collect_on_spike(mh_runs):
    run = mh_runs["full"]
    recs = _mh_records(run)
    start = ml.unpack_mhb_summary(run.one("mhb_bg_start")["l39"]["data"])
    assert (start["status"], start["done"]) == (0, 0xE)          # dispatched to harts 1-3, still running
    mid = recs[(run.one("nop_during_bg")["step"], 0)]["mh"]
    assert {mid["stat"][h]["state"] for h in (1, 2, 3)} == {ml.MH_STATE["BUSY"]}   # hart 0 served a NOP meanwhile
    coll = run.one("mhb_bg_collect")
    sm = ml.unpack_mhb_summary(coll["l39"]["data"])
    job = recs[(coll["step"], 0)]["mhb"]["job"]
    assert (sm["status"], sm["done"], sm["tag"]) == (0, 0xE, run.one("mhb_bg_start")["seq"])
    assert job["mode"] == ml.MHB_MODE["RUN"] and sm["span"] >= 5000   # collected; the whole window ran


def test_mh_secondary_trap_on_spike(mh_runs):
    """An illegal instruction in a worker on hart 2: its aux line holds the dump (mcause 2, mtval 0xC0001073), it is back
    in its idle loop, the CALL completes without bit 2, hart 0's L1/L36 are untouched by it, and the next ping and
    bench see all three harts again."""
    run = mh_runs["full"]
    recs = _mh_records(run)
    t = run.one("mhb_trap_h2")
    sm = ml.unpack_mhb_summary(t["l39"]["data"])
    assert t["l1"]["state"] == ST["IDLE"] and t["l1"]["mcause"] == 0 and t["l36"]["n_traps"] == 0
    assert (sm["status"], sm["mask"], sm["done"]) == (0, 0xF, 0xB)
    mh = recs[(t["step"], 0)]["mh"]
    st2, ax2 = mh["stat"][2], mh["aux"][2]
    assert st2["traps"] == 1 and st2["trap_gen"] == mh["ctl"]["gen"] and st2["state"] == ml.MH_STATE["IDLE"]
    assert (ax2["mcause"], ax2["mtval"]) == (2, 0xC0001073) and ml.MHB_BASE <= ax2["mepc"] < ml.MHB_DATA
    assert all(mh["stat"][h]["traps"] == 0 for h in (1, 3))
    p = ml.unpack_l39_mh(run.one("mh_ping_after_trap")["l39"]["data"])
    assert (p["done"], p["ready"], p["trapped"]) == (0xE, 0xF, 1 << 2)
    after = ml.unpack_mhb_summary(run.one("mhb_after_trap")["l39"]["data"])
    assert (after["status"], after["done"]) == (0, 0xF)
    end = run.one("nop_end")
    assert end["l36"]["n_traps"] == 0 and end["l36"]["n_reentry"] == 0 and run.end[2] == 1   # hart 0: no trap, no entry


def test_mh_one_writer_audit_on_spike(mh_runs):
    """spike -l --log-commits on the audit subset: every committed store's 64 B line and the hart that made it. Harts
    1-3 store only into their own lines (sh_emu[h], stat/aux, trap stack, private block, stack, mh_bench result line);
    hart 0 stores into the MH region only its ctl line; the one line shared by two harts is a result line that hart 0
    wrote as the scripted host's LOAD of mh_bench (the declared host-to-hart handover); no AMO, LR or SC commits."""
    run = mh_runs["audit"]
    assert run.res.ok and run.end[1] == 0, run.describe()
    assert run.res.amos == [], run.res.amos[:3]
    e = ci.Elf(mh_runs["audit_elf"])
    mh0, mh1, emu = e.sym("__mh_start"), e.sym("__mh_end"), e.sym("sh_emu")
    funcs = sorted((s["value"], s["value"] + s["size"], s["name"]) for s in e.symbols if s["type"] == 2 and s["shndx"])
    host_fns = {"sh_run", "host_line_write", "host_write", "run_step"}

    def fn(pc):
        return next((n for lo, hi, n in funcs if lo <= pc < hi), None)
    lines = run.res.stores_by_line
    assert lines and mh1 - mh0 == 0x1000   # the compact SPIKE_HOST geometry (asm.h)
    for h in (1, 2, 3):
        mine = {emu + 64 * h, mh0 + 0x80 * h, mh0 + 0x80 * h + 0x40, ml.mhb_res_addr(h)}
        mine |= set(range(mh0 + 0x200 + 0x80 * (h - 1), mh0 + 0x200 + 0x80 * h, 64))      # trap stack
        mine |= set(range(mh0 + 0x400 + 0x400 * (h - 1), mh0 + 0x400 + 0x400 * h, 64))    # private block + stack
        got = {a for a, hs in lines.items() if h in hs}
        assert got and got <= mine, (h, sorted(hex(a) for a in got - mine))
        assert not [a for a in got if ml.MB_BASE <= a < ml.PAYLOAD_BASE]       # never the mailbox
    h0 = {a for a, hs in lines.items() if 0 in hs}
    assert {a for a in h0 if mh0 <= a < mh1} == {mh0}                          # the ctl line only
    shared = {a: hs for a, hs in lines.items() if len(hs) > 1}
    assert set(shared) <= {ml.mhb_res_addr(h) for h in (1, 2, 3)}, {hex(a): v for a, v in shared.items()}
    for a in shared:
        assert fn(run.res.store_pcs[(0, a)]) in host_fns, (hex(a), fn(run.res.store_pcs[(0, a)]))


def test_mh_images_have_no_amo_and_keep_the_csr_rules(fw, mh_runs):
    """Static: the X280 image and every payload have no A-extension instruction; the SPIKE_MH image still has none of the
    X280-only CSRs (spike traps them) and its hart-1-3 path reads the emulated mnstatus instead."""
    x = ci.Elf(fw / "monitor.elf")
    assert ci.amo_words(x) == [] and x.sym("mh_sec_rnmi_exc") == ml.MH_SEC_RNMI_EXC
    for p in sorted((fw / "payloads").glob("*.elf")):
        assert ci.amo_words(ci.Elf(p)) == [], p.name
    s = ci.Elf(mh_runs["full_elf"])
    assert not set(ci.csr_accesses(s)) & (X280_ONLY_CSRS | SMRNMI_CSRS) and ci.amo_words(s) == []
    assert ml.LIM_BASE <= s.sym("__mh_start") < s.sym("__mh_end") <= s.sym("__stack_bottom")
