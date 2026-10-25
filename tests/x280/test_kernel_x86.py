"""T-3.1 (PLAN.md M3, WP-3.1): the freestanding gear_wbc kernel, x86 build, against onnxruntime.

The x280/fw kernel is built with Makefile.kernel into a temp dir and loaded with ctypes; the blobs
come from x280/tools/pack_weights.py (fp16 with both policies, fp32 one policy each as under DP4).
Balance and Walk both run on every logged obs516 of an M1 VL run (10,080 ticks, 5,600 Balance and
4,480 Walk; $WBC_OBS516_NPZ overrides) plus 256 N(0,1) inputs, against per-row ORT on the CPU
provider with one thread (the loop's OrtCpu "single"):
  v0  fp32 weights   max|da| <= 5e-5
  v2  fp16 weights   max|da| <= 1e-2   (v2s must equal v2 bit for bit)
Also: the exact graph, the blob format and every error code, CRC-32 vs zlib, the fp16 widening on
all 65,536 patterns, expf/ELU accuracy, and that the riscv objects have no undefined symbols and
no Zvfh instructions.

  simpy -m pytest $RT/tests/x280/test_kernel_x86.py -q

The logged obs come from (about 2 min; seeds 0-6 all run 1440 ticks, detfn walks every 200 ticks):
  simpy -m rtserve.mock_vla_server --port 0 --mode detfn         # prints READY ... url=<url>
  simpy -m rtloop.loop --mode vl --arm v1 --d-inj 4 --vla <url> --wbc ort --seeds 0-6 \
      --dump-obs516 --tag obs516_kernel --runs-root $RT/runs/x280   # -> <run>/obs516.npz
(both under `nice -n 19 taskset -c 0,1,8-11,16,17,24-27`). Without the file the logged-obs tests skip.
"""

import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import zlib

import numpy as np
import pytest

RT = Path(__file__).resolve().parents[2]
FW = RT / "x280" / "fw"
OBS516_NPZ = os.environ.get(
    "WBC_OBS516_NPZ", str(RT / "runs/x280/20261024_211935_obs516_kernel/obs516.npz"))
N_RANDOM = 256
TOL_FP32 = 5e-5
TOL_FP16 = 1e-2
SCRATCH_BYTES = 4096
V0, V1, V2, V2S = 0, 1, 2, 3
EINVAL, EMAGIC, EVERSION, ESIZE, ECRC, ELAYOUT, EPOLICY, EVARIANT, EDTYPE = range(-1, -10, -1)
# Off the real-time cores 2,3,4,6,7,12-15 and their SMT siblings (task rule; override with X280_CPUS).
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
PIN = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]  # never the tick thread's CPUs 2, 3, 18, 19


def _load_pack_weights():
    spec = importlib.util.spec_from_file_location("pack_weights", RT / "x280/tools/pack_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pw = _load_pack_weights()
POLICIES = pw.POLICIES


def _aligned(nbytes: int, align: int = 64) -> np.ndarray:
    raw = np.zeros(nbytes + align, np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + nbytes]


# --- fixtures -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    out = tmp_path_factory.mktemp("kernel")
    r = subprocess.run([*PIN, "make", "-s", "-C", str(FW), "-f", "Makefile.kernel", f"WBCK_BUILD={out}",
                        "kernel-x86", "kernel-rv64"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "warning" not in (r.stdout + r.stderr).lower(), r.stdout + r.stderr
    return out


@pytest.fixture(scope="module")
def lib(build):
    so = ctypes.CDLL(str(build / "libwbc_mlp.so"))
    vp, u32 = ctypes.c_void_p, ctypes.c_uint32
    so.wbc_infer.argtypes = [vp, u32, u32, vp, vp, vp]
    so.wbc_infer.restype = ctypes.c_int
    so.wbc_blob_check.argtypes = [vp, u32]
    so.wbc_blob_check.restype = ctypes.c_int
    so.wbc_crc32.argtypes = [u32, vp, u32]
    so.wbc_crc32.restype = u32
    so.wbc_expf.argtypes = [ctypes.c_float]
    so.wbc_expf.restype = ctypes.c_float
    so.wbc_elu.argtypes = [vp, u32]
    so.wbc_elu.restype = None
    so.wbc_f16_to_f32.argtypes = [ctypes.c_uint16]
    so.wbc_f16_to_f32.restype = ctypes.c_float
    so.wbc_variants_built.restype = u32
    return so


class Blob:
    """A packed blob in a 64 B-aligned buffer, plus a one-call-per-row runner."""

    def __init__(self, lib, data: bytes):
        self.lib, self.data = lib, data
        self.buf = _aligned(len(data))
        self.buf[:] = np.frombuffer(data, np.uint8)
        self.scratch = _aligned(SCRATCH_BYTES)

    @property
    def ptr(self) -> int:
        return self.buf.ctypes.data

    def infer(self, policy: int, variant: int, obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs, np.float32)
        act = np.full((len(obs), 15), np.nan, np.float32)
        o0, a0, s = obs.ctypes.data, act.ctypes.data, self.scratch.ctypes.data
        for n in range(len(obs)):
            rc = self.lib.wbc_infer(self.ptr, policy, variant, o0 + n * 516 * 4, a0 + n * 15 * 4, s)
            assert rc == 0, (policy, variant, n, rc)
        return act


@pytest.fixture(scope="module")
def blobs(lib):
    return {
        "fp16": Blob(lib, pw.pack(dtype="fp16", policies=(0, 1))),
        ("fp32", 0): Blob(lib, pw.pack(dtype="fp32", policies=(0,))),
        ("fp32", 1): Blob(lib, pw.pack(dtype="fp32", policies=(1,))),
    }


@pytest.fixture(scope="module")
def logged():
    if not Path(OBS516_NPZ).is_file():
        pytest.skip(f"no logged obs516 at {OBS516_NPZ} (set $WBC_OBS516_NPZ)")
    return np.load(OBS516_NPZ)


@pytest.fixture(scope="module")
def inputs(logged):
    rnd = np.random.default_rng(0).standard_normal((N_RANDOM, 516)).astype(np.float32)
    return np.concatenate([logged["obs"].astype(np.float32), rnd])


@pytest.fixture(scope="module")
def ort_sessions():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return [ort.InferenceSession(str(Path(pw.ONNX_DIR) / f"{name}.onnx"), so, providers=["CPUExecutionProvider"])
            for name in POLICIES]


@pytest.fixture(scope="module")
def ort_ref(ort_sessions, inputs):
    """Per-row ORT outputs (batch 1, as the loop calls it) for both policies on every input."""
    out = []
    for sess in ort_sessions:
        name = sess.get_inputs()[0].name
        out.append(np.concatenate([sess.run(None, {name: inputs[n:n + 1]})[0] for n in range(len(inputs))]))
    return out


# --- graph and data -----------------------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_graph_is_the_kernel_architecture(policy):
    path = Path(pw.ONNX_DIR) / f"{policy}.onnx"
    pw.check_graph(str(path))
    lines = pw.describe(str(path))
    assert len(lines) == 22
    assert [l.split("(")[0] for l in lines] == list(pw.wbc_numpy_ref.EXPECTED_OPS)


def test_logged_obs_cover_both_policies(logged, ort_sessions):
    pol, obs, act = logged["policy"], logged["obs"], logged["action"]
    assert len(obs) >= 10_000 and obs.shape[1] == 516 and np.isfinite(obs).all()
    counts = {p: int((pol == p).sum()) for p in POLICIES}
    assert all(c >= 1000 for c in counts.values()), counts
    # The labels are right: ORT on each logged row reproduces the logged action of its policy.
    for p, sess in zip(POLICIES, ort_sessions):
        idx = np.flatnonzero(pol == p)[::50]
        name = sess.get_inputs()[0].name
        got = np.concatenate([sess.run(None, {name: obs[i:i + 1]})[0] for i in idx])
        assert np.max(np.abs(got - act[idx])) <= 1e-5, p


# --- parity against ORT -------------------------------------------------------------------------


@pytest.mark.parametrize("policy", [0, 1], ids=POLICIES)
def test_fp32_v0_matches_ort(blobs, inputs, ort_ref, policy):
    got = blobs[("fp32", policy)].infer(policy, V0, inputs)
    err = np.abs(got - ort_ref[policy])
    print(f"\n{POLICIES[policy]} v0 fp32: max|da| {err.max():.3g} (logged {err[:-N_RANDOM].max():.3g}, "
          f"random {err[-N_RANDOM:].max():.3g}) over {len(inputs)} inputs")
    assert np.isfinite(got).all()
    assert err.max() <= TOL_FP32, err.max()


@pytest.mark.parametrize("policy", [0, 1], ids=POLICIES)
def test_fp16_v2_matches_ort(blobs, inputs, ort_ref, policy):
    got = blobs["fp16"].infer(policy, V2, inputs)
    err = np.abs(got - ort_ref[policy])
    print(f"\n{POLICIES[policy]} v2 fp16: max|da| {err.max():.3g} (logged {err[:-N_RANDOM].max():.3g}, "
          f"random {err[-N_RANDOM:].max():.3g}) over {len(inputs)} inputs")
    assert np.isfinite(got).all()
    assert err.max() <= TOL_FP16, err.max()
    # v2s is the same scalar path on x86, and v2 on riscv must equal it bit for bit (wbc_mlp.h).
    sub = inputs[::40]
    assert np.array_equal(blobs["fp16"].infer(policy, V2S, sub).view(np.uint32),
                          got[::40].view(np.uint32))


def _f64_forward(layers, obs: np.ndarray) -> np.ndarray:
    """float64 forward pass over the blob's own (dequantised) weights; ELU via expm1."""
    def mlp(x, ls):
        for j, (wt, b, *_rest) in enumerate(ls):
            x = x @ wt.astype(np.float64) + b
            if j < len(ls) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return x
    x = obs.astype(np.float64)
    est = mlp(x, layers[:3])
    lat = est[:, 3:] / np.maximum(np.linalg.norm(est[:, 3:], axis=1, keepdims=True), 1e-12)
    return mlp(np.concatenate([x[:, -86:], est[:, :3], lat], axis=1), layers[3:])


@pytest.mark.parametrize("policy", [0, 1], ids=POLICIES)
def test_fp16_kernel_is_exact_on_its_weights(blobs, inputs, policy):
    """Against float64 math on the same fp16-rounded weights the v2 error is fp32 rounding only."""
    layers = pw.parse(blobs["fp16"].data)["layers"][POLICIES[policy]]
    sub = inputs[::20]
    err = np.abs(blobs["fp16"].infer(policy, V2, sub) - _f64_forward(layers, sub))
    assert err.max() <= 2e-5, err.max()


def test_fp16_widening_all_patterns(lib):
    h = np.arange(65536, dtype=np.uint16)
    finite = np.isfinite(h.view(np.float16))
    got = np.array([lib.wbc_f16_to_f32(int(v)) for v in h[finite]], np.float32)
    want = h[finite].view(np.float16).astype(np.float32)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))  # incl. subnormals and -0


def test_results_do_not_depend_on_scratch(blobs, inputs):
    b = blobs["fp16"]
    sub = inputs[:16]
    first = b.infer(1, V2, sub)
    b.scratch[:] = np.frombuffer(np.full(SCRATCH_BYTES // 4, np.nan, np.float32).tobytes(), np.uint8)
    assert np.array_equal(b.infer(1, V2, sub).view(np.uint32), first.view(np.uint32))


# --- blob format and error codes ----------------------------------------------------------------


def test_blob_fits_lim_and_cli_asserts_it(capsys):
    fp16 = pw.pack(dtype="fp16")
    assert pw.end_address(len(fp16)) <= pw.LIM_END
    assert pw.end_address(len(pw.pack(dtype="fp32", policies=(0,)))) <= pw.LIM_END
    assert pw.end_address(len(pw.pack(dtype="fp32", policies=(0, 1)))) > pw.LIM_END
    assert pw.main(["--dtype", "fp16"]) == 0
    assert "ends 0x081de340" in capsys.readouterr().out
    assert pw.main(["--dtype", "fp32"]) == 2            # both fp32 policies do not fit
    assert pw.main(["--dtype", "fp32", "--host-only"]) == 0


def test_parse_roundtrip():
    info = pw.parse(pw.pack(dtype="fp16"))
    assert info["policy_mask"] == 3 and info["dtype"] == "float16" and info["tag"] == "pack_weights v1 fp16"
    for name in POLICIES:
        layers = info["layers"][name]
        assert [(wt.shape, b.shape) for wt, b, *_ in layers] == [((i, o), (o,)) for i, o in pw.DIMS]
        assert all(w_off % 64 == 0 and b_off % 64 == 0 for *_, w_off, b_off in layers)


def test_blob_check_accepts_good_blobs(lib, blobs):
    for b in blobs.values():
        assert lib.wbc_blob_check(b.ptr, len(b.data)) == 0
        assert lib.wbc_blob_check(b.ptr, len(b.data) + 64) == 0  # a padded LOAD length is fine


def _with(data: bytes, off: int, value: int, recrc: bool = False) -> bytes:
    raw = bytearray(data)
    raw[off:off + 4] = int(value).to_bytes(4, "little")
    if recrc:
        raw[16:20] = zlib.crc32(bytes(raw[20:])).to_bytes(4, "little")
    return bytes(raw)


def test_blob_check_rejects(lib, blobs):
    good = blobs["fp16"].data

    def check(data, n=None):
        b = Blob(lib, data)
        return lib.wbc_blob_check(b.ptr, len(data) if n is None else n)

    flipped = bytearray(good)
    flipped[len(good) // 2] ^= 0x01
    assert check(bytes(flipped)) == ECRC
    assert check(_with(good, 0, 0x12345678)) == EMAGIC
    assert check(_with(good, 4, 2)) == EVERSION
    assert check(good, 100) == ESIZE                       # shorter than the header
    assert check(good, len(good) - 4) == ESIZE             # total_len > len
    assert check(_with(good, 64, 515, recrc=True)) == ELAYOUT   # layer 0 in = 515, valid CRC
    assert check(_with(good, 24, 0, recrc=True)) == ELAYOUT     # layout
    assert check(_with(good, 28, 0, recrc=True)) == ELAYOUT     # no policy present
    assert check(_with(good, 64 + 8, 0x40, recrc=True)) == ELAYOUT  # w_off inside the header
    assert check(_with(good, 64 + 12, 0x7FFFFFC0, recrc=True)) == ESIZE  # b_off past the end
    assert lib.wbc_blob_check(None, 1000) == EINVAL
    assert lib.wbc_blob_check(blobs["fp16"].ptr + 2, len(good)) == EINVAL


def test_infer_error_codes(lib, blobs):
    b16, b32 = blobs["fp16"], blobs[("fp32", 0)]
    obs = _aligned(516 * 4 + 64)
    act = np.full(16, 7.0, np.float32)
    s = b16.scratch.ctypes.data

    def run(blob, policy, variant, o=None, a=None, sc=s):
        return lib.wbc_infer(blob.ptr, policy, variant, obs.ctypes.data if o is None else o,
                             act.ctypes.data if a is None else a, sc)

    assert lib.wbc_variants_built() == 0b1101            # x86: v0, v2, v2s; no RVV
    assert run(b32, 0, V1) == EVARIANT
    assert run(b32, 0, 9) == EVARIANT
    assert run(b16, 0, V0) == EDTYPE
    assert run(b32, 0, V2) == EDTYPE
    assert run(b32, 1, V0) == EPOLICY                    # fp32 Balance-only blob (DP4)
    assert run(b16, 2, V2) == EPOLICY
    assert run(b16, 0, V2, o=obs.ctypes.data + 2) == EINVAL
    assert run(b16, 0, V2, sc=None) == EINVAL
    assert lib.wbc_infer(None, 0, V2, obs.ctypes.data, act.ctypes.data, s) == EINVAL
    assert lib.wbc_infer(blobs[("fp32", 1)].ptr, 0, V0, obs.ctypes.data, act.ctypes.data, s) == EPOLICY
    bad = Blob(lib, _with(b16.data, 0, 0))
    assert run(bad, 0, V2) == EMAGIC
    assert np.all(act == 7.0)                            # nothing written on an error
    assert run(b16, 0, V2) == 0 and np.isfinite(act[:15]).all() and act[15] == 7.0


def test_crc32_matches_zlib(lib):
    rng = np.random.default_rng(1)
    buf = _aligned(4096)
    buf[:] = rng.integers(0, 256, 4096, dtype=np.uint8)
    for start in (0, 1, 2, 3, 5):
        for n in (0, 1, 3, 4, 7, 64, 1000, 4091 - start):
            want = zlib.crc32(buf[start:start + n].tobytes())
            assert lib.wbc_crc32(0, buf.ctypes.data + start, n) == want, (start, n)
    head = lib.wbc_crc32(0, buf.ctypes.data, 1001)
    assert lib.wbc_crc32(head, buf.ctypes.data + 1001, 999) == zlib.crc32(buf[:2000].tobytes())


# --- expf / ELU ---------------------------------------------------------------------------------


def test_expf_accuracy(lib):
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.uniform(-87, 88, 100_000), rng.uniform(-1, 1, 50_000), np.logspace(-40, 1.9, 2000),
                        -np.logspace(-40, 1.93, 2000), [-87.0, -0.0, 0.0, 88.0]]).astype(np.float32)
    got = np.array([lib.wbc_expf(float(v)) for v in x], np.float64)
    ref = np.exp(x.astype(np.float64))
    ulp = np.abs(got - ref) / np.ldexp(1.0, np.frexp(ref)[1] - 24)
    assert ulp.max() <= 1.0, (ulp.max(), x[np.argmax(ulp)])   # exhaustive: 0.990 (elu.c)
    assert lib.wbc_expf(0.0) == 1.0 and lib.wbc_expf(-200.0) == lib.wbc_expf(-87.0) > 0.0  # clamped


def test_elu_accuracy_and_specials(lib):
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.uniform(-30, 30, 200_000), rng.uniform(-1e-3, 1e-3, 20_000),
                        [-87.5, -1e30, 0.0, -0.0, 1e30, np.inf, -np.inf, np.nan]]).astype(np.float32)
    v = x.copy()
    lib.wbc_elu(v.ctypes.data, len(v))
    fin = np.isfinite(x)
    ref = np.where(x[fin] > 0, x[fin].astype(np.float64), np.expm1(np.minimum(x[fin], 0).astype(np.float64)))
    assert np.max(np.abs(v[fin] - ref)) <= 1e-7               # exhaustive: 4.82e-8 (elu.c)
    assert v[-3] == np.inf and v[-2] == -1.0 and np.isnan(v[-1])


# --- riscv objects ------------------------------------------------------------------------------


def test_rv64_objects_are_freestanding(build):
    kobj = build / "rv64" / "wbc_kernel.o"
    nm = shutil.which("llvm-nm-20") or "llvm-nm"
    assert subprocess.run([nm, "-u", str(kobj)], capture_output=True, text=True, check=True).stdout.strip() == ""
    assert subprocess.run([nm, "-u", str(build / "libwbc_mlp.so")], capture_output=True, text=True,
                          check=True).stdout.strip() == ""
    dis = subprocess.run([shutil.which("llvm-objdump-20") or "llvm-objdump", "-d", "--no-show-raw-insn", str(kobj)],
                         capture_output=True, text=True, check=True).stdout
    mnem = [ln.split()[1] for ln in dis.splitlines() if ln.startswith(" ") and ":" in ln and len(ln.split()) > 1]
    assert not [m for m in mnem if m.startswith(("vfwcvt", "vfncvt", "fcvt.s.h", "fcvt.h.s"))]  # no Zvfh/Zvfhmin
    assert not [m for m in mnem if m.startswith(("fmadd", "fmsub", "fnmadd", "fnmsub", "vfmacc", "vfmadd"))]
    assert "vle16.v" in mnem and "vwmul.vx" in mnem and "vfadd.vv" in mnem
    size = subprocess.run([shutil.which("llvm-size-20") or "llvm-size", str(kobj)], capture_output=True, text=True,
                          check=True).stdout.splitlines()[1].split()
    text, data, bss = map(int, size[:3])
    assert text <= 16 * 1024 and data == 0 and bss == 0, size
