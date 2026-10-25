"""rttt/bench_g1.py and tests/tt/test_g1_fidelity.py against a CPU fake of server_g1 (no device; PLAN.md WP-2.4).

    simpy -m pytest $RT/tests/tt/test_bench_g1_fake.py -q

The fake (tests/tt/fake_g1_server.py) has a known "model", so a fake golden in the WP-2.2 layout (rttt/golden_io.py)
can be built that it matches; its faults (perturbation, stale state, non-determinism, unseeded repeats, disabled
debug endpoints, slow calls) must each be caught by the check that exists for it.
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import zmq

from rtloop import schema, telemetry
from rttt import bench_g1 as B
from rttt import golden_io as gio

from . import fake_golden

HERE = Path(__file__).resolve().parent
RT = HERE.parents[1]
PY = sys.executable
FIDELITY = HERE / "test_g1_fidelity.py"


def free_port() -> int:
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        if not 5555 <= port <= 5572:
            return port


class Server:
    def __init__(self, *flags: str):
        self.port = free_port()
        self.url = f"tcp://127.0.0.1:{self.port}"
        self.p = subprocess.Popen([PY, str(HERE / "fake_g1_server.py"), "--port", str(self.port), *flags],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        c = B.G1Client(self.url, 2000)
        try:
            for _ in range(100):
                if c.ping(timeout_ms=200):
                    return
                if self.p.poll() is not None:
                    raise RuntimeError(self.p.stderr.read())
            raise RuntimeError("fake server did not start")
        finally:
            c.close()

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()


@pytest.fixture
def server(request):
    servers = []

    def start(*flags):
        s = Server(*flags)
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.stop()


@pytest.fixture(scope="module")
def golden(tmp_path_factory) -> Path:
    """A 20-sample golden in the WP-2.2 layout whose fp32 run is the fake model and bf16 run its bf16 rounding."""
    return fake_golden.build(tmp_path_factory.mktemp("g") / "golden")


def run_bench(tmp_path, url, *args) -> tuple[int, dict]:
    out = tmp_path / f"bench_{time.monotonic_ns()}.json"
    rc = B.main(["--endpoint", url, "--out", str(out), *map(str, args)])
    return rc, json.loads(out.read_text()) if out.exists() else {}


def checks(res: dict) -> dict:
    return {c["name"]: c for c in res["checks"]}


# --- latency ---------------------------------------------------------------------------------------------------------


def test_latency_ok(tmp_path, server, golden):
    s = server("--device-ms", "2")
    rc, res = run_bench(tmp_path, s.url, "--obs", golden, "--n", 30, "--period-ms", 15, "--warmup", 3,
                        "--keep-calls")
    assert rc == 0, res["checks"]
    lat = res["latency"]
    assert lat["n_ok"] == 30 and lat["n_errors"] == 0 and not lat["slow_calls"]
    assert lat["device_ms"]["p50"] >= 1.9 and lat["server_total_ms"]["p50"] <= lat["rtt_ms"]["p50"]
    assert lat["d_est"]["p99"] == 1 and lat["wire_in_ms"]["p50"] >= 0 and lat["wire_out_ms"]["p50"] >= 0
    assert len(lat["calls"]) == 30 and res["info"]["backend"] == "fake:tt"
    gaps = np.diff([c["i"] for c in lat["calls"]])
    assert (gaps == 1).all()


def test_latency_slow_calls_attributed_and_gated(tmp_path, server, golden):
    s = server("--device-ms", "1", "--slow-every", "4", "--slow-ms", "110")
    rc, res = run_bench(tmp_path, s.url, "--obs", golden, "--n", 12, "--period-ms", 5, "--warmup", 0)
    lat = res["latency"]
    assert len(lat["slow_calls"]) == 3 and lat["slow_causes"] == {"device": 3}
    assert all(c["device_ms"] > 100 for c in lat["slow_calls"])
    assert rc == 1 and not checks(res)["rtt_p99_ms"]["pass"] and checks(res)["rtt_p50_ms"]["pass"]


def test_attribute_rules():
    base = {"rtt_ms": 150.0, "encode_ms": 2.0, "device_ms": 40.0, "decode_ms": 1.0, "server_total_ms": 44.0,
            "wire_in_ms": 100.0, "wire_out_ms": 6.0}
    assert B.attribute(base) == "wire_in"
    assert B.attribute(dict(base, wire_in_ms=None, wire_out_ms=None)) == "wire"
    assert B.attribute(dict(base, encode_ms=130.0, server_total_ms=175.0, wire_in_ms=0.1)) == "encode"
    assert B.attribute({"rtt_ms": 120.0}) == "unreported"
    assert B.d_est(39.4, 0.5) == 2 and B.d_est(40.0, 0.5) == 3 and B.d_est(1.0, 0.5) == 1


# --- ABAB and determinism --------------------------------------------------------------------------------------------


def test_abab_and_determinism_pass(tmp_path, server, golden):
    s = server("--debug-endpoints")
    rc, res = run_bench(tmp_path, s.url, "--obs", golden, "--abab", 20, "--determinism")
    assert rc == 0, res["checks"]
    assert res["abab"]["n_mismatches"] == 0 and res["abab"]["a_b_distinct"]
    det = res["determinism"]
    assert det["replays_identical"] and det["unseeded_differ"] and det["same_seed_identical"]


def test_abab_catches_stale_state(tmp_path, server, golden):
    s = server("--debug-endpoints", "--stale")
    rc, res = run_bench(tmp_path, s.url, "--obs", golden, "--abab", 10)
    assert rc == 1 and res["abab"]["n_mismatches"] == 10


@pytest.mark.parametrize("fault,check", [("--nondeterministic", "fixed_noise_replays_identical"),
                                         ("--unseeded-repeat", "unseeded_calls_differ")])
def test_determinism_faults(tmp_path, server, golden, fault, check):
    s = server("--debug-endpoints", fault)
    rc, res = run_bench(tmp_path, s.url, "--obs", golden, "--determinism")
    assert rc == 1 and not checks(res)[check]["pass"]


def test_debug_endpoints_disabled_is_a_failure(tmp_path, server, golden):
    s = server()
    rc, res = run_bench(tmp_path, s.url, "--obs", golden, "--abab", 4)
    assert rc == 1 and "debug endpoints disabled" in res["fatal"]


def test_unreachable_server_exits_2(tmp_path):
    rc, _ = run_bench(tmp_path, f"tcp://127.0.0.1:{free_port()}", "--n", 1)
    assert rc == 2


def test_abab_pair_differs_in_state_and_image(golden):
    """T-2.4's pair must differ in state AND image (review:tt issue 1): on the real n16_g1 golden s00-s09 are resets
    with bit-identical state, so B is the first mid-episode sample, s10; with the fake golden every obs differs."""
    if (gio.GOLDEN_DIR / "obs_index.json").exists():
        real = B.load_obs_set([str(gio.GOLDEN_DIR)])
        (na, a), (nb, b) = B.abab_pair(real)
        assert (na, nb) == ("s00", "s10") and B.inputs_differ(a, b) == {"state": True, "image": True}
        assert not B.inputs_differ(real[0][1], real[1][1])["state"]  # s00/s01: the pair that hid the bug
    (na, _), (nb, _) = B.abab_pair(B.load_obs_set([str(golden)]))
    assert (na, nb) == ("s00", "s01")
    obs = B.synthetic_obs(3)
    same_state = dict(obs, **{schema.VIDEO_KEY: B.synthetic_obs(4)[schema.VIDEO_KEY]})
    assert B.abab_pair([("a", obs), ("b", same_state)]) is None
    assert B.abab_pair([("a", obs)]) is None and B.abab_pair([]) is None


def test_abab_refuses_a_pair_with_equal_state(tmp_path, server):
    s = server("--debug-endpoints")
    a, b = B.synthetic_obs(5), B.synthetic_obs(6)
    b = dict(a, **{schema.VIDEO_KEY: b[schema.VIDEO_KEY]})  # image differs, state does not
    gio.save_obs(tmp_path / "a.npz", a)
    gio.save_obs(tmp_path / "b.npz", b)
    rc, res = run_bench(tmp_path, s.url, "--obs", tmp_path / "a.npz", "--obs", tmp_path / "b.npz", "--abab", 4)
    assert rc == 2 and res == {}


def test_abab_records_what_differs(tmp_path, server, golden):
    s = server("--debug-endpoints")
    rc, res = run_bench(tmp_path, s.url, "--obs", golden, "--abab", 4)
    assert rc == 0 and res["abab"]["inputs_differ"] == {"state": True, "image": True}


# --- status stream ---------------------------------------------------------------------------------------------------


def test_status_sub_checks_the_stream(tmp_path, server, golden):
    port = free_port()
    s = server("--status-pub", f"tcp://127.0.0.1:{port}")
    rc, res = run_bench(tmp_path, s.url, "--obs", golden, "--n", 10, "--period-ms", 50, "--warmup", 2,
                        "--status-sub", f"tcp://127.0.0.1:{port}", "--rtt-p99-max", 1000)
    assert rc == 0, res["checks"]
    st = res["status_stream"]
    assert st["n"] >= 3 and st["src"] == "server_g1" and st["calls_delta"] > 0 and st["seq_gaps"] == 0
    assert {"status_msgs", "status_src", "status_rate_hz", "status_backend", "status_calls_advance"} <= set(checks(res))


def test_status_sub_without_a_publisher_fails(tmp_path, server, golden):
    s = server()
    rc, res = run_bench(tmp_path, s.url, "--obs", golden, "--n", 3, "--period-ms", 5, "--warmup", 0,
                        "--status-sub", f"tcp://127.0.0.1:{free_port()}")
    assert rc == 1 and res["status_stream"]["n"] == 0 and not checks(res)["status_msgs"]["pass"]


# --- observation sources ---------------------------------------------------------------------------------------------


def test_obs_sources(tmp_path, golden):
    g = B.load_obs_set([str(golden)])
    assert [n for n, _ in g][:2] == ["s00", "s01"] and len(g) == 20
    run = tmp_path / "run"
    run.mkdir()
    for seq, (_, obs) in enumerate(g[:3]):
        telemetry.save_vla_obs(run, seq, obs)  # the loop's own dump format ('.' -> '__')
    from_run = B.load_obs_set([str(run)])
    assert len(from_run) == 3
    for (_, a), (_, b) in zip(from_run, g[:3], strict=True):
        assert all(np.array_equal(a[k], b[k]) for k in schema.VLA_OBS_KEYS[:-1])
        assert a[schema.LANGUAGE_KEY] == b[schema.LANGUAGE_KEY]
    one = B.load_obs_set([str(run / "vla_obs" / "000001.npz"), str(golden / "obs" / "s05.npz")])
    assert len(one) == 2
    with pytest.raises(ValueError):
        B.load_obs_set([str(tmp_path / "nothing.txt")])


def test_compare_is_bitwise():
    a = {"x": np.array([0.0, 1.0], np.float32)}
    assert B.compare(a, {"x": np.array([0.0, 1.0], np.float32)}) == (True, 0.0)
    assert B.compare(a, {"x": np.array([-0.0, 1.0], np.float32)})[0] is False  # -0.0 differs in bits
    assert B.compare(a, {"x": np.array([0.0, 1.0], np.float64)})[0] is False
    assert B.compare(a, {"y": np.array([0.0, 1.0], np.float32)})[0] is False


# --- the T-2.2 fidelity test against the fake --------------------------------------------------------------------------


def run_fidelity(golden: Path, url: str | None, report: Path, use_env: bool = False):
    env = {k: v for k, v in os.environ.items() if not k.startswith("G1_")}
    args = [PY, "-m", "pytest", str(FIDELITY), "-q", "-p", "no:cacheprovider", "-rfs"]
    if use_env:
        env.update(G1_GOLDEN=str(golden), G1_FIDELITY_REPORT=str(report), **({"G1_ENDPOINT": url} if url else {}))
    else:
        args += ["-p", "conftest_fidelity", "--golden", str(golden), "--fidelity-report", str(report)]
        if url:
            args += ["--endpoint", url]
    env["PYTHONPATH"] = os.pathsep.join([str(RT), str(HERE), env.get("PYTHONPATH", "")])
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=300, cwd=str(RT))


def test_fidelity_passes_on_small_error(tmp_path, server, golden):
    s = server("--debug-endpoints", "--perturb", "1e-4")
    rep = tmp_path / "t22.json"
    r = run_fidelity(golden, s.url, rep)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "5 passed" in r.stdout
    v = json.loads(rep.read_text())
    assert v["n_samples"] == 20 and v["kinds"] == {"mid": 10, "reset": 10}
    assert v["verdicts"]["primary"]["min"] > 0.999
    g = v["verdicts"]["groups"]["groups"]
    assert not g["left_hand"]["gated"] and not g["base_height_command"]["gated"] and g["right_hand"]["gated"]
    assert all(abs(g[n]["floor"] - g[n]["manifest_floor"]) < 1e-12 for n in g)
    assert set(v["verdicts"]["replays"]["samples"]) == {"s00", "s10"}


def failed(r) -> set[str]:
    """Test names in the FAILED lines of pytest's short summary."""
    return {ln.split("::", 1)[1].split()[0] for ln in r.stdout.splitlines() if ln.startswith("FAILED ") and "::" in ln}


def test_fidelity_fails_on_large_error(tmp_path, server, golden):
    s = server("--debug-endpoints", "--perturb", "0.05")
    r = run_fidelity(golden, s.url, tmp_path / "t22.json", use_env=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert failed(r) == {"test_primary_pcc_vs_bf16", "test_secondary_vs_fp32", "test_group_abs_floors"}, r.stdout


def test_fidelity_catches_nondeterministic_replays(tmp_path, server, golden):
    s = server("--debug-endpoints", "--nondeterministic")
    r = run_fidelity(golden, s.url, tmp_path / "t22.json")
    assert r.returncode == 1
    assert "test_fixed_noise_replays_bit_identical" in failed(r), r.stdout


def test_fidelity_without_debug_endpoints_fails(tmp_path, server, golden):
    s = server()
    r = run_fidelity(golden, s.url, tmp_path / "t22.json")
    assert r.returncode == 1 and "debug endpoints disabled" in r.stdout


def test_fidelity_skips_without_endpoint(tmp_path, golden):
    r = run_fidelity(golden, None, tmp_path / "t22.json", use_env=True)
    assert r.returncode == 0 and "5 skipped" in r.stdout, r.stdout


def test_fidelity_metrics_unit(golden):
    """sample_metrics/verdicts on the golden itself: TT == bf16 passes every gate; a shifted group fails it."""
    from . import test_g1_fidelity as T

    per = []
    for s in gio.load_index(golden):
        smp = gio.load_sample(golden, s["sid"])
        tt = {"action_pred_norm": smp["bf16.action_pred_norm"], **gio.actions(smp, "bf16")}
        per.append(T.sample_metrics(s["sid"], s["kind"], smp, tt))
    v = T.verdicts(per, {"s00": {"identical": True}})
    assert all(v[k]["pass"] for k in ("primary", "secondary", "groups", "replays"))
    smp = gio.load_sample(golden, "s03")
    tt = {"action_pred_norm": smp["bf16.action_pred_norm"], **gio.actions(smp, "bf16")}
    tt["action.waist"] = tt["action.waist"] + 0.03  # over the 0.02 floor
    tt["action.left_hand"] = tt["action.left_hand"] + 1.0  # excluded group: must not fail
    per[3] = T.sample_metrics("s03", "reset", smp, tt)
    v = T.verdicts(per, {"s00": {"identical": True}})
    assert v["groups"]["failures"] == ["waist"]
    assert v["primary"]["pass"] and v["secondary"]["pass"]


def test_fidelity_records_tt_vs_fp32_per_group_without_gating_it(golden):
    """s1:device next step: per group the TT-vs-fp32 max|d| sits next to the gated TT-vs-bf16 one; the gates are
    unchanged (TT == fp32 everywhere would still fail a group whose TT-vs-bf16 is over its floor, and vice versa)."""
    from . import test_g1_fidelity as T

    def run(src: str) -> dict:
        per = []
        for s in gio.load_index(golden):
            smp = gio.load_sample(golden, s["sid"])
            tt = {"action_pred_norm": smp[f"{src}.action_pred_norm"], **gio.actions(smp, src)}
            per.append(T.sample_metrics(s["sid"], s["kind"], smp, tt))
        return {"per": per, "v": T.verdicts(per, {"s00": {"identical": True}})}

    bf16, fp32 = run("bf16"), run("fp32")
    for name, g in bf16["v"]["groups"]["groups"].items():
        assert g["tt_bf16_max"] == 0 and g["tt_fp32_max"] == pytest.approx(g["bf16_fp32_max"])
        assert g["argmax_fp32"] in {m["sid"] for m in bf16["per"]}
        assert all(m["groups"][name]["tt_fp32"] == m["groups"][name]["bf16_fp32"] for m in bf16["per"])
        assert fp32["v"]["groups"]["groups"][name]["tt_fp32_max"] == 0  # TT == fp32
    assert bf16["v"]["groups"]["pass"] and fp32["v"]["groups"]["pass"] == all(
        g["bf16_fp32_max"] <= g["floor"] or not g["gated"] for g in fp32["v"]["groups"]["groups"].values())


def test_zmq_ctx_clean():
    # bench_g1 uses zmq.Context.instance(); make sure nothing above left it unusable
    s = zmq.Context.instance().socket(zmq.REQ)
    s.close(0)
