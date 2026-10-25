"""The M3 session jobs' verdict logic and wiring (verify:all D1, D8, D9; fix:x280 open issue F5 -> S1c's S2 go/no-go).

x280/tools/m3_jobs.py is tested as functions and as the CLI the jobs call; jobs/m3_eval.sh and jobs/m3_shadow.sh run
end to end in --mode sync (the jobs' smoke-test switch) against a mock VLA and `x280d serve --backend sim` on random
ports; jobs/demo_m3.sh's CHIP_SAFE_CPUS refusal runs as is. No device, no --mode rt, no port in 5555-5574.

    source $RT/env.sh
    nice -n 19 taskset -c 0,1,8-11,16,17,24-27 bash -c 'source $RT/env.sh && simpy -m pytest $RT/tests/x280/test_m3_jobs.py -q'
"""

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT))

from x280 import regs  # noqa: E402
from x280.tools import m3_jobs as M  # noqa: E402

X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
NICE = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]
XPY = RT / ".venv-x280" / "bin" / "python"
PY = str(XPY) if XPY.exists() else sys.executable
M3J = RT / "x280" / "tools" / "m3_jobs.py"
JOBS = RT / "jobs"
S1_BENCH = RT / "runs/sessions/20261025_004501_1113962_s1/t23_bench.json"  # S1, default arena (span p50 44.12 ms)


def _cpu_set(spec: str) -> set:
    """taskset's list format ('0,1,8-11') -> {0, 1, 8, 9, 10, 11}."""
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
    assert not 5555 <= p <= 5574
    return p


def _analyze_run():
    spec = importlib.util.spec_from_file_location("analyze_run_m3_jobs", RT / "scripts" / "analyze_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bench(tmp_path, span=44.1184495, total=43.54998850612901, name="t23_bench.json") -> Path:
    p = tmp_path / name
    lat = {"server_total_ms": {"p50": total}}
    if span is not None:
        lat["server_span_ms"] = {"p50": span}
    p.write_text(json.dumps({"latency": lat, "pass": True}))
    return p


# --- m3-gates (D1) ---------------------------------------------------------------------------------------------------


def test_m3_gates_copy_gates_yaml_and_derive_from_the_span(tmp_path):
    """T-3.7's session sets are gates.yaml's m3_rt / m3_rt_slow row for row (starvation, TT errors, X280 share and
    heartbeat included), with vla_p50_ms_max = 1.05 x the T-2.3 SPAN p50 (what analyze_run gates), not the total."""
    yaml = pytest.importorskip("yaml")
    src = yaml.safe_load((RT / "gates.yaml").read_text())
    dst = tmp_path / "gates_m3.yaml"
    r = M.m3_gates(RT / "gates.yaml", dst, _bench(tmp_path), "fp16")
    got = yaml.safe_load(dst.read_text())
    assert set(got) == {"m3_rt", "m3_rt_slow"} and r["t23_span_p50_ms"] == pytest.approx(44.1184495)
    for name in got:
        assert set(got[name]) == set(src[name])
        for k in ("starved_frac_max", "starved_run_max", "vla_errors_max", "wbc_x280_frac_min",
                  "x280_heartbeat_continuity_min"):
            assert got[name][k] == src[name][k]
        assert got[name]["vla_p50_ms_max"] == round(1.05 * 44.1184495, 3) == 46.324  # not 1.05 x 43.55 = 45.727
        assert {k: v for k, v in got[name].items() if k != "vla_p50_ms_max"} == \
            {k: v for k, v in src[name].items() if k != "vla_p50_ms_max"}
    ar = _analyze_run()
    for name in got:  # every key is one analyze_run knows
        assert ar.load_gates(dst, name) == got[name]
    M.m3_gates(RT / "gates.yaml", dst, None, "fp32")
    got = yaml.safe_load(dst.read_text())
    assert got["m3_rt"]["shadow_da_p99_max"] == 5e-5 and got["m3_rt"]["vla_p50_ms_max"] == src["m3_rt"]["vla_p50_ms_max"]
    with pytest.raises(ValueError, match="server_span_ms"):
        M.m3_gates(RT / "gates.yaml", dst, _bench(tmp_path, span=None, name="old.json"), "fp16")
    if S1_BENCH.exists():
        assert M.t23_span_p50(S1_BENCH) == pytest.approx(44.1184495)


def test_m3_eval_never_reads_the_superseded_proposal():
    text = (JOBS / "m3_eval.sh").read_text()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "gates_m3.proposed" not in code and '"$RT/gates.yaml"' in code and "--x280-ticks" in code
    assert (RT / "configs" / "gates_m3.proposed.yaml").read_text().startswith("# SUPERSEDED")
    assert "RING_CALLS=65536" in text and M.RING_CALLS == 65536 and "65,536" in text


def test_constants_match_x280d():
    import inspect
    pytest.importorskip("zmq")
    from x280 import x280d as D
    assert M.SHADOW_TOL == D.PARITY_TOL and M.RING_CALLS == D.RING
    src = inspect.getsource(D.Server.__init__)  # the Server.counts keys t37-summary differences
    assert all(f'"{k}"' in src for k in (*M.X280D_FAULTS, "calls", "ok"))


# --- S1c summary and the S2 go/no-go (fix:x280 open issue F5, task item 8) -------------------------------------------


def _e0(ready=True, lock=0x1F, cntl_0=0x12, ok=True) -> dict:
    return {"pass": ok, "pll4_walk_ready": ready, "pll_lock": lock, "pll4_default": True,
            "pll4": {"cntl_0": cntl_0, "cntl_1": regs.PLL4_CNTL_1_DEFAULT, "cntl_5": regs.PLL4_CNTL_5_DEFAULT,
                     "use_postdiv": 0xF}}


def _s1c_dir(tmp_path, e0k=None, e0t=None) -> Path:
    d = tmp_path / "s1c"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (d / f"s1c_tt_digest_{i}.json").write_text(json.dumps({"s00": "ab", "s10": "cd"}))
    for n, p50 in (("s1c_tt_lat_0.json", 44.8), ("s1c_tt_lat_1.json", 45.0)):
        (d / n).write_text(json.dumps({"latency": {"rtt_ms": {"p50": p50, "max": 60.0}}}))
    for n, e in (("s1c_e0_keeper.json", e0k), ("s1c_e0_tt.json", e0t)):
        if e is not None:
            (d / n).write_text(json.dumps(e, sort_keys=True) + "\n")
    return d


RCS = {k: "0" for k in ("e0_keeper", "e1a_init", "e1a_check", "host_only", "fd_no_tt", "e0_tt", "fd_tt",
                        "e1b_init", "e1b_check")}


def test_s1c_go_needs_e0_e1a_and_the_pll4_walk_values(tmp_path):
    s = M.s1c_summary(_s1c_dir(tmp_path, _e0(), _e0()), True, "A", RCS)
    assert s["pass"] and s["s2"]["go"] and not s["s2"]["why"] and not s["s2"]["notes"]
    assert s["s2"]["e0_keeper"] == {"pll4_walk_ready": True, "pll_lock": "0x1f", "pll4_cntl_0": "0x12",
                                    "pll4_default": True, "cntl_0_running": True, "pll4_locked": True}
    # silicon reads something else than the CMFW sources say: kmd's walk would refuse at write 1 -> NO-GO, with values
    s = M.s1c_summary(_s1c_dir(tmp_path, _e0(ready=False, lock=0x0F, cntl_0=0x13), _e0()), True, "A", RCS)
    assert not s["s2"]["go"] and len(s["s2"]["why"]) == 1
    why = s["s2"]["why"][0]
    assert "pll4_walk_ready=False" in why and "PLL_LOCK=0xf" in why and "CNTL_0=0x13" in why
    assert s["s2"]["e0_keeper"]["pll4_locked"] is False and s["s2"]["e0_keeper"]["cntl_0_running"] is False
    s = M.s1c_summary(_s1c_dir(tmp_path, _e0(), _e0()), False, "B", {**RCS, "e1a_init": "1"})
    assert not s["s2"]["go"] and "E1a" in s["s2"]["why"][0] and not s["pass"]
    s = M.s1c_summary(_s1c_dir(tmp_path / "missing", None, None), True, "A", RCS)
    assert not s["s2"]["go"] and "no result" in s["s2"]["why"][0] and "no result" in s["s2"]["notes"][0]
    e0_old = _e0()
    del e0_old["pll4_walk_ready"], e0_old["pll_lock"]  # an x280d from before review F5 records neither
    s = M.s1c_summary(_s1c_dir(tmp_path / "old", e0_old, _e0()), True, "A", RCS)
    assert not s["s2"]["go"] and "pll4_walk_ready=None" in s["s2"]["why"][0]
    s = M.s1c_summary(_s1c_dir(tmp_path / "tt", _e0(), _e0(ready=False, lock=0x0F)), True, "A", RCS)
    assert s["s2"]["go"] and "D13" in s["s2"]["notes"][0]  # S2 boots before server_g1: a note, not a NO-GO


def test_s1c_summary_reads_what_x280d_e0_records(tmp_path, monkeypatch):
    """The key names the summary reads (pll_lock, pll4_walk_ready, pll4.cntl_0) are what `x280d e0` prints."""
    pytest.importorskip("zmq")
    from x280 import x280d as D
    if not D.IMAGE_BIN.exists():
        pytest.skip("firmware build missing (make -C $RT/x280/fw x280)")
    monkeypatch.setenv("CHIP_SESSION_ID", f"test-m3-jobs-{os.getpid()}")
    s = D.Session("sim", ledger_path=tmp_path / "ledger.json", hart=False, cross_read=True)
    try:
        s.open()
        s.mode = "no_release"
        e0 = json.loads(json.dumps(D.run_admin(s, "e0"), default=D._jsonable))
    finally:
        s.close()
    v = M.pll_view(e0)
    assert v["pll4_walk_ready"] is True and v["pll_lock"] == "0x1f" and v["pll4_cntl_0"] == "0x12"
    assert v["cntl_0_running"] and v["pll4_locked"]


def test_s1c_summary_cli_is_stdlib_only(tmp_path):
    d = _s1c_dir(tmp_path, _e0(), _e0(ready=False))
    rcs = [f"{k}={v}" for k, v in RCS.items()]
    r = subprocess.run(["python3", "-I", str(M3J), "s1c-summary", str(d), "1", "A", *rcs], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.loads((d / "s1c_summary.json").read_text())
    assert s == json.loads(r.stdout.strip().splitlines()[-1]) and s["s2"]["go"] and s["s2"]["notes"]


def test_s1c_job_logs_the_go_no_go_line():
    text = (JOBS / "s1c.sh").read_text()
    assert "m3_jobs.py\" s1c-summary" in text and "S1c -> S2: GO" in text and "S1c -> S2: NO-GO" in text
    assert "pll4_walk_ready" in text and "pll_lock" in text and "not implemented yet" not in text


# --- T-3.6 / T-3.7 summaries -----------------------------------------------------------------------------------------


def _verdict(tmp_path, name="verdict.json", ok=True, **metrics) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"pass": ok, "metrics": metrics}))
    return p


def test_t36_gates_and_summary(tmp_path):
    pytest.importorskip("yaml")  # analyze_run's load_gates (simpy)
    ar = _analyze_run()
    for dtype, tol in (("fp16", 1e-2), ("fp32", 5e-5)):
        g = tmp_path / f"gates_t36_{dtype}.yaml"
        M.t36_gates(g, dtype)
        assert ar.load_gates(g, "t36") == {"shadow_da_p999_max": tol}
    with pytest.raises(ValueError):
        M.t36_gates(tmp_path / "x.yaml", "bf16")
    ok = dict(n_ticks=28_000, shadow_da_p999=7.1e-3, shadow_da_p99=6e-3, shadow_da_max=8e-3, shadow_miss_frac=0.0)
    assert M.t36_summary(_verdict(tmp_path, **ok), "fp16", 0)["pass"]
    assert not M.t36_summary(_verdict(tmp_path, **{**ok, "shadow_miss_frac": 0.002}), "fp16", 0)["pass"]
    assert not M.t36_summary(_verdict(tmp_path, ok=False, **ok), "fp16", 0)["pass"]
    assert not M.t36_summary(_verdict(tmp_path, **{**ok, "shadow_miss_frac": None}), "fp16", 0)["pass"]
    assert not M.t36_summary(_verdict(tmp_path, **ok), "fp16", 1)["pass"]


def test_t37_summary_x280d_record_and_the_t23_band(tmp_path):
    def counts(p, **c):
        base = {"calls": 0, "ok": 0, "timeouts": 0, "faults": 0, "stalls": 0, "interlock": 0, "degraded": 0}
        p.write_text("x280d log line\n" + json.dumps({"counts": {**base, **c}}) + "\n")
        return p
    before = counts(tmp_path / "before.json", calls=10, ok=10)
    after = counts(tmp_path / "after.json", calls=28_810, ok=28_810)
    dump = tmp_path / "ring.json"
    dump.write_text(json.dumps({"ring": "/x/x280d_ticks.npy", "rows": 28_810, "calls": 28_810, "wrapped": False}))
    bench = _bench(tmp_path)
    v = _verdict(tmp_path, vla_p50_ms=44.5, x280_heartbeat_continuity=1.0)
    s = M.t37_summary(v, before, after, dump, bench)
    assert s["pass"] and s["x280d_ok"] and s["x280d_during_soak"]["calls"] == 28_800 and s["ring"]["rows"] == 28_810
    assert s["vla_p50_vs_t23"]["pass"] and s["vla_p50_vs_t23"]["lo_ms"] == round(0.95 * 44.1184495, 3)
    assert not M.t37_summary(_verdict(tmp_path, vla_p50_ms=41.0), before, after, dump, bench)["pass"]  # < 0.95 x
    assert M.t37_summary(_verdict(tmp_path, vla_p50_ms=41.0), before, after, dump, None)["t23_row"].startswith("not")
    stalled = counts(tmp_path / "stalled.json", calls=28_810, ok=28_809, stalls=1)
    assert not M.t37_summary(v, before, stalled, dump, bench)["x280d_ok"]
    assert not M.t37_summary(v, before, before, dump, bench)["x280d_ok"]  # no call reached x280d


# --- the jobs ----------------------------------------------------------------------------------------------------------


def _job_env(tmp_path, **kw) -> dict:
    sd = tmp_path / "session"
    sd.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "CHIP_SAFE_CPUS", "M3_T23_BENCH")}
    env.update(CHIP_SESSION_DIR=str(sd), CHIP_X280_MODE="boot:3", CHIP_X280_URL="tcp://127.0.0.1:9",
               CHIP_VLA_URL="tcp://127.0.0.1:9", CHIP_SESSION_ID=f"test-m3-jobs-{os.getpid()}")
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _job(name, env, timeout=60):
    return subprocess.run(NICE + ["bash", str(JOBS / name)], env=env, capture_output=True, text=True, timeout=timeout)


def test_demo_m3_refuses_under_chip_safe_cpus(tmp_path):
    """verify:all D9: as demo_m2.sh, the M3 demo (rt_sim on CPU 2, x280d on CPU 5) never runs in a session confined
    to the non-real-time cores; it stops before exec'ing demo.sh."""
    env = {**os.environ, "CHIP_SAFE_CPUS": "0,1", "DEMO_ARGS": "--definitely-not-a-demo-flag"}
    for job in ("demo_m3.sh", "demo_m2.sh"):
        r = subprocess.run(["bash", str(JOBS / job)], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 2 and "CHIP_SAFE_CPUS=0,1 is set" in r.stderr and "needs the real-time cores" in r.stderr


def test_m3_eval_refuses_a_soak_the_tick_ring_cannot_hold(tmp_path):
    r = _job("m3_eval.sh", _job_env(tmp_path, M3_EVAL_EPISODES=50, M3_T23_BENCH="none"))
    assert r.returncode == 2 and "65536-call tick ring" in r.stderr, r.stderr[-2000:]
    r = _job("m3_eval.sh", _job_env(tmp_path))
    assert r.returncode == 2 and "set M3_T23_BENCH" in r.stderr, r.stderr[-2000:]
    r = _job("m3_eval.sh", _job_env(tmp_path, M3_T23_BENCH=tmp_path / "nope.json"))
    assert r.returncode == 2 and "no such file" in r.stderr


# CHIP_SAFE_CPUS (verify2 N1): rtloop --mode rt pins itself onto CPUs 2/3 whatever the job's taskset mask, and under
# CHIP_SAFE_CPUS chip_session leaves x280d off CPU 5, so x280d's own refusal never fires: the jobs must refuse.


def _safe_env(tmp_path, **kw) -> dict:
    """A job env confined to X280_CPUS, with HOME on an empty dir: env.sh's simpy / xpy cannot start there (their cd
    into $R / $RT fails), so a job that did not refuse still could not reach `rtloop.loop --mode rt`."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return _job_env(tmp_path, CHIP_SAFE_CPUS=X280_CPUS, HOME=home, **kw)


@pytest.mark.parametrize("job,kw", [
    ("m3_eval.sh", {}),                        # T-3.7: the rt soak
    ("m3_eval.sh", {"M3_EVAL_MODE": "sync"}),  # the screen after the sync smoke is still eval_paired --mode rt
    ("m3_faults.sh", {}),                      # T-3.8
    ("m3_shadow.sh", {}),                      # T-3.6
    ("m3_soak.sh", {}),                        # T-3.9
])
def test_m3_rt_jobs_refuse_chip_safe_cpus(tmp_path, job, kw):
    env = _safe_env(tmp_path, **kw)
    r = _job(job, env)
    assert r.returncode == 2, (r.returncode, r.stderr[-2000:])
    assert f"CHIP_SAFE_CPUS={X280_CPUS} is set" in r.stderr and "on the real-time cores" in r.stderr, r.stderr[-2000:]
    assert {p.name for p in Path(env["CHIP_SESSION_DIR"]).iterdir()} <= {"session.log", "events.jsonl"}  # nothing ran


def test_m3_sync_smokes_still_run_under_chip_safe_cpus(tmp_path):
    """The CPU smoke switches (no rt loop, no screen) get past the refusal and stop at their next check."""
    r = _job("m3_eval.sh", _safe_env(tmp_path, M3_EVAL_MODE="sync", M3_SKIP_SCREEN=1))
    assert r.returncode == 2 and "real-time" not in r.stderr and "set M3_T23_BENCH" in r.stderr, r.stderr[-2000:]
    r = _job("m3_shadow.sh", _safe_env(tmp_path, M3_SHADOW_MODE="sync"))
    assert r.returncode == 2 and "real-time" not in r.stderr and "does not answer ping" in r.stderr, r.stderr[-2000:]


# x280d started again inside a job (verify2 N3): T-3.8's re-attach after kill -9 and S2's E3(d) attach.


def _x280_harness(tmp_path, body: str, **env) -> subprocess.CompletedProcess:
    """Source jobs/_common.sh and jobs/_x280.sh with chip_start / chip_wait_ready replaced by recorders, then run
    ``body``: the command line a job would start, with nothing started."""
    sd = tmp_path / "sd"
    sd.mkdir(exist_ok=True)
    calls = tmp_path / "calls.txt"
    script = tmp_path / "harness.sh"
    script.write_text(f"""source {JOBS / "_common.sh"}
source {JOBS / "_x280.sh"}
chip_start() {{ echo "START $1 $2 :: ${{*:3}}" >>{calls}; CHIP_PID=$$; }}
chip_wait_ready() {{ echo "WAIT $1 $2 $4" >>{calls}; return 0; }}
{body}
""")
    base = {k: v for k, v in os.environ.items() if not k.startswith("CHIP_")}
    base.update(CHIP_SESSION_DIR=str(sd), CHIP_X280_URL="tcp://127.0.0.1:9", CHIP_X280_PORT="9", **env)
    return subprocess.run(NICE + ["bash", str(script)], env=base, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("env,tail", [
    ({}, "--cpus 5"),                                               # chip_session's default: no status stream
    ({"CHIP_X280_STATUS_PUB": "tcp://127.0.0.1:40125"}, "--cpus 5 --status-pub tcp://127.0.0.1:40125"),
    ({"CHIP_SAFE_CPUS": X280_CPUS}, ""),                            # a confined session keeps x280d off CPU 5
])
def test_x280_reattach_places_x280d_as_chip_session_does(tmp_path, env, tail):
    if "CHIP_SAFE_CPUS" in env and 5 in _cpu_set(X280_CPUS):
        pytest.skip("X280_CPUS includes CPU 5")
    r = _x280_harness(tmp_path, 'x280_reattach; echo "rc=$?"', **env)
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    start, wait = (tmp_path / "calls.txt").read_text().splitlines()
    sd = tmp_path / "sd"
    cmd = start.split(" :: ", 1)[1]
    assert start.startswith(f"START x280d {sd / 'x280d.attach.log'} :: ") and "x280/x280d.py serve" in cmd
    assert cmd.endswith(f"serve --port 9 --tile 3 --attach --log-dir {sd / 'x280d_attach'}" + (f" {tail}" if tail else ""))
    assert cmd.count("--cpus") == (1 if tail else 0) and "--release" not in cmd and "--boot" not in cmd
    assert wait == "WAIT x280d tcp://127.0.0.1:9 60"
    if not tail:
        assert "excludes CPU 5" in r.stderr


def test_jobs_start_x280d_again_only_through_x280_reattach():
    """chip_session (start_x280d) and every job start x280d through chip_lib.sh x280d_start, the one place that adds
    --cpus 5 and --status-pub."""
    def code(p: Path) -> str:
        return "\n".join(ln for ln in p.read_text().splitlines() if not ln.lstrip().startswith("#"))
    for job in ("m3_faults.sh", "s2_e2_e3.sh"):
        assert "x280_reattach" in code(JOBS / job), job
    for p in JOBS.glob("*.sh"):
        assert "chip_start x280d" not in code(p) and "chip_cmd X280D" not in code(p), p
    session = code(RT / "scripts" / "chip_session.sh")
    assert "chip_start x280d" not in session and "x280d_start " in session


# End-to-end smokes: a mock VLA and x280d --backend sim on random ports, the job's loop in --mode sync.


@pytest.fixture
def servers(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("yaml")
    from rtserve.mock_vla_server import spawn
    vla, vla_url = spawn("hold", python=sys.executable)
    port = _free_port()
    cpus = sorted(os.sched_getaffinity(0))
    log = open(tmp_path / "x280d.log", "w")
    # --deadline-ms 50 (and SMOKE_LOOP's timeouts): these smokes check the jobs' wiring, not timing, on shared
    # non-real-time cores where a sim reply can miss the 4 ms deadline and turn into a fallback / shadow miss
    x280d = subprocess.Popen(NICE + [PY, str(RT / "x280" / "x280d.py"), "serve", "--backend", "sim", "--port",
                                     str(port), "--sim-cpus", str(cpus[-1]), "--deadline-ms", "50",
                                     "--log-dir", str(tmp_path / "x280d")],
                             cwd=RT, env={**os.environ, "PYTHONPATH": str(RT)}, stdout=log, stderr=subprocess.STDOUT)
    url = f"tcp://127.0.0.1:{port}"
    t0 = time.monotonic()
    while subprocess.run([PY, str(RT / "x280" / "x280d.py"), "ping", "--via", url, "--timeout-s", "1"], cwd=RT,
                         env={**os.environ, "PYTHONPATH": str(RT)}, capture_output=True).returncode != 0:
        assert time.monotonic() - t0 < 90 and x280d.poll() is None, (tmp_path / "x280d.log").read_text()[-2000:]
    try:
        yield {"vla": vla_url, "x280": url}
    finally:
        for p in (x280d, vla):
            p.terminate()
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(5)
        log.close()
        for f in (vla.stdout, vla.stderr):
            f.close()


def test_m3_eval_job_end_to_end_in_sync_mode(tmp_path, servers):
    """D1 wiring: session gates from gates.yaml with the span-derived VLA limit, the loop through x280d, x280d's
    tick ring dumped into the session dir through --via, analyze_run's heartbeat gate computed from it (data, not
    'no data'), and t37_summary.json. The sync run fails m3_rt's timing rows (expected: T-3.7 is the rt run)."""
    env = _job_env(tmp_path, CHIP_X280_URL=servers["x280"], CHIP_VLA_URL=servers["vla"], M3_EVAL_MODE="sync",
                   M3_EVAL_EPISODES=1, M3_EVAL_MAX_TICKS=60, M3_SKIP_SCREEN=1, M3_EVAL_LOOP_ARGS="--wbc-timeout-ms 100",
                   M3_T23_BENCH=_bench(tmp_path))
    r = _job("m3_eval.sh", env, timeout=900)
    sd = Path(env["CHIP_SESSION_DIR"])
    assert r.returncode in (0, 1), (r.returncode, r.stderr[-4000:])
    import yaml
    g = yaml.safe_load((sd / "gates_m3.yaml").read_text())
    assert g["m3_rt"]["vla_p50_ms_max"] == 46.324 and g["m3_rt"]["x280_heartbeat_continuity_min"] == 1.0
    ring = np.load(sd / "x280d_ticks.npy")
    dump = json.loads((sd / "t37_ring.json").read_text())
    assert dump["ring"] == str(sd / "x280d_ticks.npy") and dump["rows"] == len(ring) >= 60 and not dump["wrapped"]
    run = next((sd / "runs_t37").glob("*_t37"))
    v = json.loads((run / "verdict.json").read_text())
    m = v["metrics"]
    assert v["gates"] == "m3_rt" and m["x280_ring"] == str(sd / "x280d_ticks.npy")
    assert m["x280_heartbeat_continuity"] == 1.0 and m["x280_heartbeat"]["rows"] >= 60, m["x280_heartbeat"]
    assert m["wbc_x280_frac"] == 1.0
    checks = {c["name"]: c for c in v["checks"]}
    assert checks["x280_heartbeat_continuity_min"]["pass"] and checks["wbc_x280_frac_min"]["pass"]
    s = json.loads((sd / "t37_summary.json").read_text())
    assert s["x280d_ok"] and s["x280d_during_soak"]["calls"] >= 60 and s["vla_p50_vs_t23"]["t23_span_p50_ms"] > 44


def test_m3_shadow_job_end_to_end_in_sync_mode(tmp_path, servers):
    """D8: T-3.6 live: ORT drives, x280d (sim, fp16) shadows every tick; analyze_run's t36 gate on shadow_da p99.9
    and the miss check pass, and the job exits 0."""
    env = _job_env(tmp_path, CHIP_X280_URL=servers["x280"], CHIP_VLA_URL=servers["vla"], M3_SHADOW_MODE="sync",
                   M3_SHADOW_EPISODES=1, M3_SHADOW_MAX_TICKS=60, M3_SHADOW_LOOP_ARGS="--shadow-timeout-ms 100")
    r = _job("m3_shadow.sh", env, timeout=900)
    sd = Path(env["CHIP_SESSION_DIR"])
    assert r.returncode == 0, (r.returncode, r.stderr[-4000:])
    s = json.loads((sd / "t36_summary.json").read_text())
    assert s["pass"] and s["dtype"] == "fp16" and s["tol"] == 1e-2 and s["shadow_miss_frac"] == 0.0
    assert 0 < s["shadow_da_p999"] <= 1e-2 and s["n_ticks"] >= 60
    run = next((sd / "runs_t36").glob("*_t36"))
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["args"]["wbc"] == "ort" and cfg["args"]["shadow"] == servers["x280"]
    assert (run / "obs516.npz").exists()


# --- S1c's ECC record (verify D) and S1d (jobs/s1d.sh) ---------------------------------------------------------------

S1C_SILICON = RT / "runs/sessions/20261025_053543_2185222_s1c"  # 2026-10-25: E1a / E1b failed on the ECC delta


def test_s1c_summary_reports_each_step_s_ecc_deltas_from_the_silicon_session(tmp_path):
    """The S1c session as recorded (old step JSONs without ecc_delta / failed_checks): the summary's `ecc` block gives
    the init and check deltas, per-read rates, the last fail address and the one failed part of each gate. The gate
    and the S2 verdict are unchanged (E1a failed -> NO-GO)."""
    if not (S1C_SILICON / "s1c_e1a_check.json").exists():
        pytest.skip(f"{S1C_SILICON} is not recorded in this checkout")
    d = tmp_path / "s1c"
    d.mkdir()
    for p in S1C_SILICON.glob("s1c_*.json"):
        if p.name != "s1c_summary.json":
            (d / p.name).write_bytes(p.read_bytes())
    rcs = {**RCS, "e1a_check": "1", "e1b_check": "1"}
    s = M.s1c_summary(d, False, "B", rcs)
    assert not s["pass"] and not s["E1a"] and not s["s2"]["go"] and "E1a" in s["s2"]["why"][0]
    e = s["ecc"]
    assert e["e1a"]["init"]["dat_fix_count"] == 57_681 and e["e1a"]["init"]["dat_fail_count"] == 57_447
    assert e["e1a"]["init_events_per_line"] == {"fix": round(57_681 / 30_720, 4), "fail": round(57_447 / 30_720, 4)}
    assert e["e1a"]["check"]["dat_fail_count"] == 459_753 and e["e1a"]["check"]["dat_fix_count"] == 0
    assert e["e1a"]["check"]["last_fail"] == hex(regs.MB_HOST_GONE) and e["e1a"]["check_fail_per_word_read"] == 0.9354
    assert e["e1a"]["check_failed"] == ["ecc_fail_counts_unchanged"] and e["e1a"]["mismatches"] == 0
    assert e["e1b"]["init"]["dat_fail_count"] == 8_165 and e["e1b"]["init_method"] == "B"
    assert e["e1b"]["check"]["dat_fail_count"] == 130_816 and e["e1b"]["check"]["last_fail"] == "0x811c790"
    assert e["e1a_to_e0_tt"] == {k: 0 for k in M.ECC_COUNTS}  # nothing touched LIM between the steps
    assert any("459753" in ln and "failed on ecc_fail_counts_unchanged" in ln for ln in e["lines"])


def test_s1c_job_logs_the_ecc_lines_and_takes_s1d_s_store():
    text = (JOBS / "s1c.sh").read_text()
    assert "d['ecc']['lines']" in text and 'S1c ECC: $line' in text
    assert 'lim-init --method A --store "${S1C_E1A_STORE:-a64}"' in text


def _sim_lim_ecc(tmp_path, ecc: str, monkeypatch, **kw) -> dict:
    """`x280d lim-ecc` and the two E0s as jobs/s1d.sh records them, on a sim chip with the FakeEcc model `ecc`."""
    pytest.importorskip("zmq")
    from x280 import x280d as D
    if not D.IMAGE_BIN.exists():
        pytest.skip("firmware build missing (make -C $RT/x280/fw x280)")
    monkeypatch.setenv("CHIP_SESSION_ID", f"test-m3-jobs-s1d-{os.getpid()}")
    s = D.Session("sim", ledger_path=tmp_path / "ledger.json", hart=False, cross_read=True, sim_ecc=ecc)
    try:
        s.open()
        s.mode = "no_release"
        out = {"s1d_e0": D.run_admin(s, "e0"), "s1d_lim_ecc": D.run_admin(s, "lim_ecc", **{"lines": 96, **kw}),
               "s1d_e0_after": D.run_admin(s, "e0")}
    finally:
        s.close()
    d = tmp_path / f"s1d_{ecc}"
    d.mkdir(exist_ok=True)
    for name, r in out.items():
        (d / f"{name}.json").write_text(json.dumps(r, default=D._jsonable, sort_keys=True) + "\n")
    return d


S1D_RCS = {"e0": "0", "lim_ecc": "0", "e0_after": "0"}


@pytest.mark.parametrize("ecc,verdict,store", [("poison", "no_host_init", None), ("clean", "host_init", "a64"),
                                               ("granule", "host_init", "uc8")])
def test_s1d_summary_verdict_both_ways(tmp_path, monkeypatch, ecc, verdict, store):
    """The S1c hypothesis true ('poison'): no store is clean -> S2 stays NO-GO with a host-initialised LIM, E1F needs
    the user's DP3(d) approval. False ('clean'): method A stays. Only the 8 B stores initialise ('granule'): lim-init
    store uc8, verified with S1c's E1a gate first."""
    d = _sim_lim_ecc(tmp_path, ecc, monkeypatch)
    s = M.s1d_summary(d, S1D_RCS)
    assert s["pass"] and s["E0"] and s["E0_after"] and s["integrity"] and s["fresh_at_start"]
    rec = s["recommendation"]
    assert (rec["verdict"], rec["lim_init"]) == (verdict, store)
    names = [r["name"] for r in s["variants"]]
    assert names == ["a64", "a64nf", "wc2x32", "uc8", "a64+uc8"]
    assert s["lines"][-3] == ("S1d stores whose full re-reads leave dat_fail_count unchanged: "
                              + (", ".join(s["reread_fail_free_variants"]) or "none"))
    assert s["lines"][-2] == ("S1d clean variants (nothing counted after the write: re-reads, partial writes): "
                              + (", ".join(s["clean_variants"]) or "none"))
    assert s["lines"][-1] == f"S1d -> lim-init: {rec['text']}"
    assert s["reread_fail_free_variants"] == s["clean_variants"]  # the model has no correctable-only state
    if ecc == "poison":
        assert s["clean_variants"] == [] and "E1F" in rec["text"] and "DP3(d)" in rec["text"] and "NO-GO" in rec["text"]
        assert s["hypothesis"] and "poisoning" in s["hypothesis"]
        a64 = s["variants"][0]["write"][0]
        assert a64["accesses_per_line_min"] == 2 and a64["fail"] == pytest.approx(1.87, abs=0.15)
        assert all("ECC NOT VALID" in ln for ln in s["lines"][:5])
    elif ecc == "clean":
        assert s["clean_variants"] == names and "keep method A" in rec["text"] and s["hypothesis"] is None
    else:
        assert s["clean_variants"] == ["uc8", "a64+uc8"] and "S1C_E1A_STORE=uc8" in rec["text"]
    r = subprocess.run(["python3", "-I", str(M3J), "s1d-summary", str(d), *[f"{k}={v}" for k, v in S1D_RCS.items()]],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((d / "s1d_summary.json").read_text())["recommendation"] == rec


def test_s1d_summary_is_inconclusive_without_a_valid_measurement(tmp_path, monkeypatch):
    d = _sim_lim_ecc(tmp_path, "granule", monkeypatch)
    for broken, why in (("s1d_e0.json", "E0 before"), ("s1d_lim_ecc.json", "lim-ecc did not run"),
                        ("s1d_e0_after.json", "E0 after")):
        good = (d / broken).read_text()
        (d / broken).write_text(json.dumps({"error": "X280Error: test", "pass": False}) + "\n")
        s = M.s1d_summary(d, {**S1D_RCS, "lim_ecc": "1"})
        assert not s["pass"] and s["recommendation"]["verdict"] == "inconclusive" and why in s["recommendation"]["text"]
        assert s["lines"][-1].startswith("S1d -> lim-init: no recommendation, the measurement is not valid: ")
        assert s["recommendation"]["lim_init"] is None
        (d / broken).write_text(good)
    res = json.loads((d / "s1d_lim_ecc.json").read_text())
    res["integrity"] = False  # WAYENABLE or L2CPU_RESET moved
    (d / "s1d_lim_ecc.json").write_text(json.dumps(res))
    assert M.s1d_summary(d, S1D_RCS)["recommendation"]["verdict"] == "inconclusive"
    res["integrity"] = True
    res["variants"][3]["steps"][0]["mismatches"] = 3  # uc8 clean but wrong bytes: not chosen, and flagged
    res["variants"][3]["clean"] = False
    (d / "s1d_lim_ecc.json").write_text(json.dumps(res))
    s = M.s1d_summary(d, S1D_RCS)
    assert s["mismatch_variants"] == ["uc8"] and "DATA MISMATCH" in s["lines"][3]
    assert s["recommendation"]["verdict"] == "two_pass" and "uc8" in s["recommendation"]["text"]


@pytest.fixture
def sim_x280d(tmp_path):
    """`x280d serve --backend sim --no-release --sim-ecc MODE` on a random port (what chip_session's --x280 probe
    starts, on the FakeEcc chip)."""
    procs = []

    def start(ecc):
        port = _free_port()
        log = open(tmp_path / f"x280d_{ecc}.log", "w")
        p = subprocess.Popen(NICE + [PY, str(RT / "x280" / "x280d.py"), "serve", "--backend", "sim", "--no-release",
                                     "--sim-ecc", ecc, "--port", str(port), "--log-dir", str(tmp_path / f"x280d_{ecc}")],
                             cwd=RT, env={**os.environ, "PYTHONPATH": str(RT)}, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        url = f"tcp://127.0.0.1:{port}"
        t0 = time.monotonic()
        while subprocess.run([PY, str(RT / "x280" / "x280d.py"), "ping", "--via", url, "--timeout-s", "1"], cwd=RT,
                             env={**os.environ, "PYTHONPATH": str(RT)}, capture_output=True).returncode != 0:
            assert time.monotonic() - t0 < 60 and p.poll() is None, (tmp_path / f"x280d_{ecc}.log").read_text()[-2000:]
        return url
    yield start
    for p in procs:
        p.terminate()
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(5)


@pytest.mark.parametrize("ecc,last", [("poison", "S1d -> lim-init: no host store tried"),
                                      ("granule", "S1d -> lim-init: store uc8")])
def test_s1d_job_end_to_end_on_the_sim(tmp_path, sim_x280d, ecc, last):
    """jobs/s1d.sh as chip_session runs it (--x280 probe), against a no-release sim x280d with the ECC model: E0,
    lim-ecc, E0 again, the summary, and the job log ends with the clean variants and the recommendation."""
    pytest.importorskip("zmq")
    url = sim_x280d(ecc)
    env = _job_env(tmp_path, CHIP_X280_MODE="probe", CHIP_X280_URL=url, S1D_LINES=64, S1D_PARTIAL=4)
    r = _job("s1d.sh", env, timeout=300)
    sd = Path(env["CHIP_SESSION_DIR"])
    assert r.returncode == 0, r.stderr[-3000:]
    for n in ("s1d_e0.json", "s1d_lim_ecc.json", "s1d_e0_after.json", "s1d_summary.json"):
        assert (sd / n).exists(), n
    s = json.loads((sd / "s1d_summary.json").read_text())
    assert s["pass"] and s["recommendation"]["verdict"] == ("no_host_init" if ecc == "poison" else "host_init")
    log = [ln.split("] ", 1)[-1] for ln in r.stderr.splitlines() if "[s1d]" in ln]
    assert any(ln.startswith("S1d stores whose full re-reads leave dat_fail_count unchanged: ") for ln in log)
    assert any(ln.startswith("S1d clean variants (nothing counted after the write") for ln in log)
    assert any(ln.startswith(last) for ln in log), log[-6:]
    assert json.loads((sd / "s1d_lim_ecc.json").read_text())["lines"] == 64


def test_s1d_job_refuses_other_x280_modes_and_writes_nothing_when_e0_fails(tmp_path, sim_x280d):
    r = _job("s1d.sh", _job_env(tmp_path, CHIP_X280_MODE="boot:3"))
    assert r.returncode == 2 and "needs --x280 probe" in r.stderr
    url = sim_x280d("poison")
    env = _job_env(tmp_path, CHIP_X280_MODE="probe", CHIP_X280_URL=url, S1D_PLAN="a64,bogus")
    r = _job("s1d.sh", env, timeout=300)  # a bad plan: lim-ecc refuses before any write, the job says so
    sd = Path(env["CHIP_SESSION_DIR"])
    assert r.returncode == 1, r.stderr[-2000:]
    s = json.loads((sd / "s1d_summary.json").read_text())
    assert s["recommendation"]["verdict"] == "inconclusive" and "plan item" in s["lim_ecc_error"]


def test_s1d_job_header_and_wiring():
    text = (JOBS / "s1d.sh").read_text()
    assert "$DEVX $RT/scripts/chip_session.sh --x280 probe --vla none --job $RT/jobs/s1d.sh --tag s1d" in text
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "need_x280 probe" in code and "m3_jobs.py\" s1d-summary" in code and "x280_step s1d_lim_ecc lim-ecc" in code
    assert "$DEV " not in code and "tt-smi" not in code and "need_debug_endpoints" not in code


# --- x280_reattach from the session's CHIP_X280D_SERVE_ARGS (verify V2) -----------------------------------------------


@pytest.mark.parametrize("serve_args,want", [
    ("serve --port 9 --tile 3 --boot --slow", "serve --port 9 --tile 3 --slow --attach"),
    ("serve --port 9 --tile 3,2 --dtype fp32 --boot --cpus 6 --status-pub tcp://127.0.0.1:40125",
     "serve --port 9 --tile 3,2 --dtype fp32 --cpus 6 --status-pub tcp://127.0.0.1:40125 --attach"),
    ("serve --port 9 --tile 3 --no-release --status-pub", "serve --port 9 --tile 3 --status-pub --attach"),
    ("serve --port=9 --tile 3 --boot --log-dir /elsewhere --early --deadline-ms 6 --variant v2s --no-blob",
     "serve --port=9 --tile 3 --early --deadline-ms 6 --variant v2s --no-blob --attach"),
    ("serve --port 9 --tile 3 --attach", "serve --port 9 --tile 3 --attach"),
    # the E1F stage-0 boot, every M3 session's since 2026-10-25: a boot like --boot, so the restart attaches
    ("serve --port 9 --tile 3 --stage0", "serve --port 9 --tile 3 --attach"),
    ("serve --port 9 --tile 3 --stage0 --cpus 5 --status-pub tcp://127.0.0.1:40125 --slow",
     "serve --port 9 --tile 3 --cpus 5 --status-pub tcp://127.0.0.1:40125 --slow --attach"),
    # serve --spin (busy-poll the REP socket) is a serving option the restart keeps
    ("serve --port 9 --tile 3 --stage0 --spin", "serve --port 9 --tile 3 --spin --attach"),
])
def test_x280_reattach_keeps_the_session_s_serve_args(tmp_path, serve_args, want):
    r = _x280_harness(tmp_path, 'x280_reattach; echo "rc=$?"', CHIP_X280D_SERVE_ARGS=serve_args)
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    start, wait = (tmp_path / "calls.txt").read_text().splitlines()
    cmd = start.split(" :: ", 1)[1]
    tail = "" if "--cpus" in serve_args else " --cpus 5"
    assert cmd.endswith(f"x280/x280d.py {want} --log-dir {tmp_path / 'sd' / 'x280d_attach'}{tail}"), cmd
    assert "--boot" not in cmd and "--no-release" not in cmd and "--stage0" not in cmd and cmd.count("--attach") == 1
    assert cmd.count("--cpus") == 1 and cmd.count("--status-pub") <= 1 and "/elsewhere" not in cmd


@pytest.mark.parametrize("serve_args,why", [
    ("serve --port 9 --tile 3", "exactly one of --stage0, --boot, --no-release, --attach"),
    ("serve --port 9 --tile 3 --stage0 --boot", "exactly one of"),
    ("serve --port 9 --tile 3 --boot --sim-cpus 5", "cannot carry '--sim-cpus'"),
    ("serve --port 9 --tile 3 --boot --backend sim", "cannot carry '--backend'"),
    ("serve --port 9 --boot --attach", "exactly one of"),
    ("serve --port 9 --tile", "--tile without a value"),
    ("boot --tile 3", "is not an x280d serve command"),
])
def test_x280_reattach_refuses_what_it_cannot_carry(tmp_path, serve_args, why):
    r = _x280_harness(tmp_path, 'x280_reattach; echo "rc=$?"', CHIP_X280D_SERVE_ARGS=serve_args)
    assert "rc=1" in r.stdout and why in r.stderr, r.stdout + r.stderr
    assert not (tmp_path / "calls.txt").exists()  # nothing started


def test_s1c_ecc_block_reads_the_new_step_json(tmp_path, monkeypatch):
    """The ecc block on step JSONs as x280d writes them now (lim-init with its own ecc_before/after, lim-check with
    failed_checks), produced on the sim with the 'poison' model: E1a's init and check rows, one failed gate part."""
    pytest.importorskip("zmq")
    from x280 import x280d as D
    if not D.IMAGE_BIN.exists():
        pytest.skip("firmware build missing (make -C $RT/x280/fw x280)")
    monkeypatch.setenv("CHIP_SESSION_ID", f"test-m3-jobs-s1c-ecc-{os.getpid()}")
    d = tmp_path / "s1c"
    d.mkdir()
    s = D.Session("sim", ledger_path=tmp_path / "ledger.json", hart=False, cross_read=True, sim_ecc="poison")
    lo, hi = regs.MAILBOX_BASE, regs.MAILBOX_BASE + 256 * 64
    try:
        s.open()
        s.mode = "no_release"
        steps = {"s1c_e0_keeper": D.run_admin(s, "e0"),
                 "s1c_e1a_init": D.run_admin(s, "lim_init", method="A", lo=lo, hi=hi, seed=1),
                 "s1c_e1a_check": D.run_admin(s, "lim_check", lo=lo, hi=hi, seed=1, partial=10, cleanup_check=True)}
    finally:
        s.close()
    for name, r in steps.items():
        (d / f"{name}.json").write_text(json.dumps(r, default=D._jsonable) + "\n")
    e = M.s1c_ecc(d)
    init, chk = steps["s1c_e1a_init"]["ecc_delta"], steps["s1c_e1a_check"]["ecc_delta"]
    assert e["e1a"]["init"]["dat_fail_count"] == init["dat_fail_count"] > 0 and e["e1a"]["init_store"] == "a64"
    assert e["e1a"]["check"]["dat_fail_count"] == chk["dat_fail_count"] > 0
    assert e["e1a"]["check_failed"] == ["ecc_fail_counts_unchanged"] and e["e1a"]["check_pass"] is False
    assert e["e1a"]["init_events_per_line"]["fail"] == pytest.approx(1.87, abs=0.15)
    assert e["lines"][0].startswith("E1a init (A/a64, 256 lines): ECC fix +")
    assert e["e1b"]["init"] is None and e["e1b"]["check"] is None  # E1b did not run here: no row, no line
    assert not [ln for ln in e["lines"] if ln.startswith("E1b")]


# --- the session's boot flag and the bench --rtt client (stage-0 sessions, T-3.5) ------------------------------------


@pytest.mark.parametrize("env,want", [
    ({"CHIP_X280D_SERVE_ARGS": "serve --port 5558 --tile 3 --stage0 --cpus 5"}, "stage0"),
    ({"CHIP_X280D_SERVE_ARGS": "serve --port 5558 --tile 3 --boot"}, "boot"),
    ({"CHIP_X280_MODE": "boot:3"}, "stage0"),  # chip_session's own default for boot:3: the stage-0 boot
    ({"CHIP_X280_MODE": "attach"}, "attach"),
    ({"CHIP_X280D_SERVE_ARGS": "serve --port 5558"}, "?"),
])
def test_x280_boot_flag(tmp_path, env, want):
    r = _x280_harness(tmp_path, 'echo "flag=$(x280_boot_flag)"', **env)
    assert f"flag={want}" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("env,want,logged", [
    ({}, "--cpus 2,3", False),                                    # rt_sim's tick + helper CPUs (no rt_sim in T-3.5)
    ({"X280_RTT_CLIENT_CPUS": ""}, "", False),                    # unpinned
    ({"X280_RTT_CLIENT_CPUS": "16,17", "X280_RTT_SPIN": "1"}, "--cpus 16,17 --spin", False),
    ({"CHIP_SAFE_CPUS": X280_CPUS}, "", True),                    # 2,3 are outside a confined session: dropped
])
def test_x280_rtt_client_args(tmp_path, env, want, logged):
    r = _x280_harness(tmp_path, 'mapfile -t a < <(x280_rtt_client_args); echo "args=[${a[*]}]"', **env)
    assert f"args=[{want}]" in r.stdout, r.stdout + r.stderr
    assert ("outside CHIP_SAFE_CPUS" in r.stderr) == logged, r.stderr


def test_m3_jobs_take_stage0_sessions_and_never_assume_a_host_lim_init():
    """Every remaining M3 job's header names the stage-0 session command (E1F, the standard boot since 2026-10-25);
    no job checks for --boot; s1f_stage0.sh alone requires --stage0 (it is the E1F job); none runs lim-init."""
    def code(p: Path) -> str:
        return "\n".join(ln for ln in p.read_text().splitlines() if not ln.lstrip().startswith("#"))
    for job in ("s2_e2_e3.sh", "e4_bench.sh", "e5_parity.sh", "m3_rtt.sh", "m3_shadow.sh", "m3_eval.sh",
                "m3_faults.sh", "m3_soak.sh", "demo_m3.sh"):
        text = (JOBS / job).read_text()
        assert "--stage0" in text and "X280_E1F_APPROVED=1" in text, job
        assert "--boot" not in code(JOBS / job) and "lim-init" not in code(JOBS / job), job
    assert '*" --stage0 "*' in code(JOBS / "s1f_stage0.sh")


# --- T-3.9 (jobs/m3_soak.sh) ------------------------------------------------------------------------------------------


def _ticks(tmp_path, be, shadow, fb=None, ep=None, name="ticks.npy") -> Path:
    from rtloop.schema import TICK_DTYPE
    n = len(be)
    t = np.zeros(n, TICK_DTYPE)
    t["ep"] = ep if ep is not None else 0
    t["k"] = np.arange(n)
    t["wbc_be"], t["shadow_da"], t["wbc_rtt_us"] = be, shadow, 700.0
    t["wbc_fb"] = fb if fb is not None else 0
    p = tmp_path / name
    np.save(p, t)
    return p


def test_t39_gates_and_swap_da(tmp_path):
    pytest.importorskip("yaml")
    ar = _analyze_run()
    M.t39_gates(tmp_path / "g.yaml")
    assert ar.load_gates(tmp_path / "g.yaml", "t39") == {"period_p99_ms": 21.0, "slips_max": 0,
                                                       "wbc_fallback_frac_max": 0.0}
    nan = float("nan")
    # X280 -> ORT -> GPU -> X280: the remote side's ORT shadow, and the sum of both through ORT for GPU -> X280
    be = [2, 2, 0, 0, 1, 1, 2, 2]
    sd = [7e-3, 6e-3, nan, nan, 1e-5, 2e-5, 5e-3, 5e-3]
    s = M.swap_da(_ticks(tmp_path, be, sd))
    assert s["swaps"] == 3 and s["no_data"] == 0 and s["pass"]
    assert s["pairs"] == {"2->0": 1, "0->1": 1, "1->2": 1} and s["max_da"] == pytest.approx(6e-3)
    assert M.swap_da(_ticks(tmp_path, be, [7e-3, 2e-2, nan, nan, 1e-5, 2e-5, 5e-3, 5e-3]))["pass"] is False
    s = M.swap_da(_ticks(tmp_path, be, [7e-3, nan, nan, nan, 1e-5, 2e-5, 5e-3, 5e-3]))  # no shadow at the swap
    assert s["no_data"] == 1 and not s["pass"]
    # a fallback (ORT served a failed call) and an episode boundary are not swaps; no swap at all is no data
    s = M.swap_da(_ticks(tmp_path, [2, 0, 2, 2], [7e-3, nan, 7e-3, 7e-3], fb=[0, 1, 0, 0]))
    assert s["swaps"] == 0 and not s["pass"]
    assert M.swap_da(_ticks(tmp_path, [2, 2, 0, 0], [7e-3] * 2 + [nan] * 2, ep=[0, 0, 1, 1]))["swaps"] == 0
    p = _ticks(tmp_path, be, sd)
    (tmp_path / "ticks.len").write_text("3\n")  # a run cut after 3 committed ticks: only the X280 -> ORT swap
    assert M.swap_da(p)["swaps"] == 1


def test_t39_summary_rows(tmp_path):
    nan = float("nan")
    ticks = _ticks(tmp_path, [2, 2, 0, 0, 2], [7e-3, 7e-3, nan, nan, 7e-3])

    def status(p, traps=0, calls=0, **c):
        base = {"calls": calls, "ok": calls, "timeouts": 0, "faults": 0, "stalls": 0, "interlock": 0, "degraded": 0}
        p.write_text("x280d log\n" + json.dumps({"counts": {**base, **c}, "tiles": {"3": {"n_traps": traps}}}) + "\n")
        return p

    def log(p, rows):
        p.write_text("".join(json.dumps({"t": t, "tiles": {"3": {"pll4_default": ok, "n_traps": tr}}}) + "\n"
                             for t, ok, tr in rows))
        return p
    before, after = status(tmp_path / "b.json", calls=4), status(tmp_path / "a.json", calls=104)
    v = _verdict(tmp_path, ok=True)
    good = log(tmp_path / "s.jsonl", [(100.0 + 10 * i, True, 0) for i in range(7)])
    s = M.t39_summary(v, ticks, before, after, [good, tmp_path / "missing.jsonl"], 95.0, 165.0, 0)
    assert s["pass"] and s["pll4"]["reads"] == 7 and s["pll4"]["max_gap_s"] == 10.0 and s["swap_da"]["swaps"] == 2
    assert s["x280d_during_soak"]["calls"] == 100 and s["traps"]["pass"]
    assert not M.t39_summary(v, ticks, before, after, [good], 95.0, 165.0, 1)["pass"]  # loop rc
    assert not M.t39_summary(_verdict(tmp_path, "v_fail.json", ok=False), ticks, before, after, [good], 95.0, 165.0,
                         0)["pass"]
    bad = log(tmp_path / "p.jsonl", [(100.0, True, 0), (110.0, False, 0), (120.0, True, 0)])
    s = M.t39_summary(v, ticks, before, after, [bad], 95.0, 125.0, 0)
    assert not s["pass"] and s["pll4"]["not_default_at"] == [110.0]
    s = M.t39_summary(v, ticks, before, after, [good], 95.0, 200.0, 0)  # 40 s without a read at the end
    assert not s["pll4"]["pass"] and s["pll4"]["max_gap_s"] == 40.0
    assert not M.t39_summary(v, ticks, before, after, [tmp_path / "missing.jsonl"], 95.0, 165.0, 0)["pll4"]["pass"]
    trap = status(tmp_path / "t.json", traps=1, calls=104)
    assert not M.t39_summary(v, ticks, before, trap, [good], 95.0, 165.0, 0)["traps"]["pass"]
    stalled = status(tmp_path / "st.json", calls=104, stalls=1)
    assert not M.t39_summary(v, ticks, before, stalled, [good], 95.0, 165.0, 0)["x280d_ok"]
    assert not M.t39_summary(v, ticks, before, before, [good], 95.0, 165.0, 0)["x280d_ok"]  # no call reached x280d
    out = tmp_path / "t39.json"
    r = subprocess.run([sys.executable, str(M3J), "t39-summary", str(v), str(ticks), str(before), str(after), "95",
                        "165", "0", str(out), str(good)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    # X280-only soak (the default since 2026-10-25, no GPU WBC in M3): no swap row to gate, even with no swap in ticks
    flat = _ticks(tmp_path, [2, 2, 2, 2, 2], [7e-3] * 5)
    assert not M.t39_summary(v, flat, before, after, [good], 95.0, 165.0, 0)["pass"]  # swaps expected, none found
    s = M.t39_summary(v, flat, before, after, [good], 95.0, 165.0, 0, swaps=False)
    assert s["pass"] and s["swap_da"]["applicable"] is False
    r = subprocess.run([sys.executable, str(M3J), "t39-summary", "--no-swaps", str(v), str(flat), str(before),
                        str(after), "95", "165", "0", str(out), str(good)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and json.loads(out.read_text())["swap_da"]["applicable"] is False, r.stdout + r.stderr
    assert json.loads(out.read_text())["pass"]
