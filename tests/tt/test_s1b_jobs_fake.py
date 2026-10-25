"""The S1b closed-loop jobs in chip_session's fake mode (PLAN.md WP-2.6: T-2.5 t2_sync.sh, T-2.6 t2_soak.sh through
demo.sh m2, T-2.7 t2_primary.sh, T-2.8 t2_soak_2h.sh), on random ports, without the device or the RT cores.

    simpy -m pytest $RT/tests/tt/test_s1b_jobs_fake.py -q

T-2.5 runs for real at smoke size: eval_paired.py drives two short --mode sync loops (one seed, 80 ticks, the
in-process ORT WBC) against the real rttt/server_g1.py --fake-model as the session's TT server and a fake GPU VLA
stand-in, so the matrix rewrite, the noise-pairing check, CRN and the error check run end to end. Nothing here runs
--mode rt: T-2.6 / T-2.8 run demo.sh m2 --in-session with tests/tt/soak_stub_loop.py in place of rt_sim (a
synthetic RT run on TT), T-2.7 is checked through eval_paired --dry-run (T2_DRY_RUN=1), and the default T-2.6 plan
(HUD, recording, viewer, the RT loop arguments) is asserted from demo.sh --dry-run. No GPU server, HUD window,
recorder or port 5555-5574 is started or messaged.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import time

import pytest
import yaml

from .test_chip_session_fake import (JOBS, PY, RT, Fake, _allowed_cpus, _server_g1_fake_model_cmd,
                                     assert_no_session_procs, free_port, MK_ENV)

FAKE_G1 = Path(__file__).resolve().parent / "fake_g1_server.py"
STUB_LOOP = Path(__file__).resolve().parent / "soak_stub_loop.py"
SRC = {"model": "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate",
       "revision": "2a5b61cc7660bdef4cf17ad18dc5dd97f13e4ce5"}


class S1bFake(Fake):
    """Fake with earlier sessions under the root (the T-2.3 benches): session_dir is the one chip_session made."""

    def session_dir(self) -> Path:
        dirs = [d for d in self.root.iterdir() if (d / "session.log").exists()]
        assert len(dirs) == 1, dirs
        return dirs[0]


@pytest.fixture
def fk(tmp_path):
    f = S1bFake(tmp_path)
    yield f
    assert_no_session_procs(f)


@pytest.fixture
def gpu_standin():
    """start(noise) -> URL of a fake GPU VLA (tests/tt/fake_g1_server.py) whose INFO says DP1@SRC_SHA, that noise
    mode and a cuda backend; every one started is killed at the end."""
    procs = []

    def start(noise: str | None = "cpu") -> str:
        port = free_port()
        info = dict(SRC, backend="cuda:fake-standin:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc",
                    **({"noise": noise} if noise else {}))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("FAKE_", "CHIP_"))}
        p = subprocess.Popen([PY, str(FAKE_G1), "--port", str(port), "--info-json", json.dumps(info)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(p)
        assert "READY" in p.stdout.readline()
        return f"tcp://127.0.0.1:{port}"

    yield start
    for p in procs:
        p.kill()
        p.wait(10)


def bench_session(root: Path, name: str, passed: bool, arena: str, span: float, mtime: float) -> Path:
    """An earlier session's T-2.3 artefacts under the session root (what rttt/m2_checks.py bench reads)."""
    d = root / name
    d.mkdir(parents=True)
    (d / "t23_bench.json").write_text(json.dumps({"pass": passed, "latency": {"server_span_ms": {"p50": span}}}))
    (d / "server_g1_info.json").write_text(json.dumps({"stamp": {"mk_arena_dtype": arena}}))
    os.utime(d / "t23_bench.json", (mtime, mtime))
    return d / "t23_bench.json"


def s1_benches(f: Fake) -> Path:
    """S1's two benches: the default-arena one (44.12 ms, pass) and the newer WP-2.7 bf16 one (47.06 ms, fail)."""
    s1 = bench_session(f.root, "20261025_004501_1113962_s1", True, "bfp8_b", 44.12, time.time() - 200)
    bench_session(f.root, "20261025_004849_1127496_s1_wp27_bf16", False, "bf16", 47.06, time.time() - 100)
    return s1


def tiny_sync_matrix(tmp: Path) -> Path:
    """eval_m2_sync.yaml at smoke size: one seed per arm, 80 ticks; the D15 URLs, as the job's rewrite expects."""
    mx = yaml.safe_load((RT / "configs" / "eval_m2_sync.yaml").read_text())
    mx.update(seeds="1000-1000", block=1, max_ticks=80,
              common_args=mx["common_args"].replace("--max-ticks 1440", "--max-ticks 80"))
    p = tmp / "eval_m2_sync_smoke.yaml"
    p.write_text(yaml.safe_dump(mx, sort_keys=False))
    return p


def loop_lines(text: str) -> list[str]:
    """eval_paired --dry-run's loop lines: '<arm>_<A>-<B>: RTLOOP_RUNS_ROOT=... <loop command>'."""
    return [ln for ln in text.splitlines() if re.match(r"^C\d\w+_\d+-\d+: ", ln)]


# --- T-2.5 (t2_sync.sh) ----------------------------------------------------------------------------------------------


@pytest.mark.skipif(not MK_ENV.exists(), reason="no MK env (publish/worktree-mk-env.sh) for server_g1 --fake-model")
def test_t25_sync_smoke_end_to_end(fk, gpu_standin):
    """Two short sync episodes (C1tt on server_g1 --fake-model, C1gpu on the stand-in), paired by seed and noise_seed,
    through the session copy of the matrix; the error check passes and the report has both arms."""
    gpu = gpu_standin("cpu")
    env = {"CHIP_VLA_CMD": _server_g1_fake_model_cmd(fk.tmp), "CHIP_VLA_READY_TIMEOUT_S": "120",
           "CHIP_VLA_EXTRA_ARGS": "--cpus " + ",".join(map(str, _allowed_cpus()[:2])),
           "T2_SYNC_MATRIX": str(tiny_sync_matrix(fk.tmp)), "T2_GPU_VLA": gpu, "T2_WBC": "ort", "T2_SKIP_ANCHOR": "1"}
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_sync.sh", env=env, timeout=900)
    sd = fk.session_dir()
    assert r.returncode in (0, 1), r.stdout[-4000:] + r.stderr[-2000:]
    tt = f"tcp://127.0.0.1:{fk.vla_port}"
    mx = yaml.safe_load((sd / "m2_sync.yaml").read_text())
    assert {a["name"]: a["args"].split()[-1] for a in mx["arms"]} == {"C1tt": tt, "C1gpu": gpu}
    assert "--wbc ort" in mx["common_args"] and "--crn" in mx["common_args"]
    rep = json.loads((sd / "m2_sync" / "report.json").read_text())
    arms = {a["arm"]: a for a in rep["arms"]}
    assert arms["C1tt"]["n"] == 1 and arms["C1gpu"]["n"] == 1
    err = json.loads((sd / "m2_sync_errors.json").read_text())
    assert err["pass"] and all(a["requests"] > 0 for a in err["per_arm"].values()), err
    from rtloop.telemetry import load_requests

    seeds = {}
    for arm, url, backend in (("C1tt", tt, "mock:server_g1-fake"), ("C1gpu", gpu, "cuda:fake-standin")):
        (cfg,) = list((sd / "m2_sync" / "runs" / arm).rglob("config.json"))
        c = json.loads(cfg.read_text())
        assert c["args"]["vla"] == url and c["args"]["mode"] == "sync" and c["args"]["crn"] is True
        assert c["vla_info"]["backend"].startswith(backend)
        reqs = load_requests(cfg.parent)
        seeds[arm] = [int(x) for x in reqs["noise_seed"]]
    n = min(map(len, seeds.values()))  # an arm's episode may end earlier (a fall): compare the common requests
    assert n >= 1 and seeds["C1tt"][:n] == seeds["C1gpu"][:n] and all(seeds["C1tt"]), seeds  # CRN per request
    assert json.loads((sd / "gpu_vla_info.json").read_text())["noise"] == "cpu"


@pytest.mark.parametrize("noise", ["cuda", None])
def test_t25_refuses_an_unpaired_gpu_vla(fk, gpu_standin, noise):
    env = {"T2_GPU_VLA": gpu_standin(noise), "T2_WBC": "ort", "T2_SKIP_ANCHOR": "1"}
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_sync.sh", env=env, timeout=120)
    assert r.returncode == 2 and "does not draw server_g1's noise" in r.stdout, r.stdout[-3000:]
    assert not (fk.session_dir() / "m2_sync").exists()  # refused before any episode


def test_t25_dry_run_prints_the_paired_loops_and_the_anchor(fk):
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_sync.sh", env={"T2_DRY_RUN": "1"}, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:]
    lines = loop_lines(r.stdout)
    tt = f"tcp://127.0.0.1:{fk.vla_port}"
    assert len(lines) == 10 and all("--mode sync" in ln and "--crn" in ln for ln in lines), r.stdout[-3000:]
    assert {ln.split("--vla ")[-1].split()[0] for ln in lines if ln.startswith("C1tt_")} == {tt}
    assert {ln.split("--vla ")[-1].split()[0] for ln in lines if ln.startswith("C1gpu_")} == {"tcp://127.0.0.1:5565"}
    anchor = next(ln for ln in r.stdout.splitlines() if "rollout_policy.py" in ln and "--n_envs" in ln)
    assert f"--policy_client_port {fk.vla_port}" in anchor and "--n_episodes 50" in anchor
    assert "--max_episode_steps 1440 --n_action_steps 20" in anchor


# --- T-2.6 / T-2.8 (t2_soak.sh through demo.sh m2, t2_soak_2h.sh) ---------------------------------------------------


def test_t26_dry_run_is_demo_m2_with_hud_recording_and_the_rt_loop(fk):
    """PLAN.md T-2.6 = demo.sh m2 as a session job, timing gates with the HUD and recording on (§4.3)."""
    s1_benches(fk)
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_soak.sh", env={"T2_DRY_RUN": "1"}, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:]
    out, sd, tt = r.stdout, fk.session_dir(), f"tcp://127.0.0.1:{fk.vla_port}"
    assert "# demo m2 (inside the device hold)" in out and f"logs in {sd}/demo_logs/" in out
    assert "scripts/hud.py" in out and "record_demo.sh" in out  # HUD and recording on
    assert "rtserve.wbc_gpu_server" in out and "gpu_vla_server" not in out  # GPU WBC; no GPU VLA swap target
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    for part in ("--mode rt --arm v1", f"--vla {tt} --wbc tcp://127.0.0.1:5556", "--viewer", "--seeds 3000-",
                 "--tag t26", "--duration 1800", "--s 16 --B 0 --d-init 3 --crn --warmup 20",
                 f"--runs-root {sd}/runs_t26"):
        assert part in loop, (part, loop)
    assert f"--gates m2_rt --gates-file {sd}/gates_m2.yaml" in out and "_soak.json" in out
    assert "--gate-drift" not in out  # T-2.6 reports the drift, T-2.8 gates it
    g = yaml.safe_load((sd / "gates_m2.yaml").read_text())["m2_rt"]
    assert g["vla_p50_ms_max"] == pytest.approx(48.532) and "viewer_k0_ms_max" in g  # the S1 bench, not WP-2.7's
    assert "20261025_004501_1113962_s1/t23_bench.json" in out and "pass False" in out
    assert "CHIP_VLA_STATUS_PUB is not set" in out  # the HUD chip panel would have no Tensix column


def test_t28_dry_run_defaults(fk):
    s1_benches(fk)
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_soak_2h.sh", env={"T2_DRY_RUN": "1"}, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:]
    loop = next(ln for ln in r.stdout.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--duration 7200" in loop and "--tag t28" in loop and "runs_t28" in loop
    sd = fk.session_dir()
    assert f"m2_checks.py soak <run dir> 7200 {sd}/t28_soak.json --gate-drift" in r.stdout


def test_t26_needs_a_t23_baseline(fk):
    bench_session(fk.root, "old_wp27", False, "bf16", 47.0, time.time() - 100)
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_soak.sh", env={"T2_DRY_RUN": "1"}, timeout=120)
    assert r.returncode == 2 and "no passing T-2.3 bench" in r.stdout, r.stdout[-2000:]


def test_t26_explicit_or_projected_baseline(fk, tmp_path):
    mine = bench_session(tmp_path / "elsewhere", "x", True, "bfp8_b", 40.0, time.time())
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_soak.sh", env={"T2_DRY_RUN": "1", "T26_BENCH": str(mine)})
    assert r.returncode == 0, r.stdout[-2000:]
    assert yaml.safe_load((fk.session_dir() / "gates_m2.yaml").read_text())["m2_rt"]["vla_p50_ms_max"] == 44.0


def test_t26_projected_baseline(fk):
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_soak.sh", env={"T2_DRY_RUN": "1", "T26_BENCH": "projected"})
    assert r.returncode == 0, r.stdout[-2000:]
    assert yaml.safe_load((fk.session_dir() / "gates_m2.yaml").read_text())["m2_rt"]["vla_p50_ms_max"] == 49.5


def soak_env(f: Fake, **kw) -> dict:
    """demo.sh without HUD, recorder or viewer (no window) and with its PUB / control on random ports."""
    demo = (f"--no-viewer --no-alt --pub-url tcp://127.0.0.1:{free_port()} "
            f"--control-url tcp://127.0.0.1:{free_port()}")
    return {"T2_SOAK_DEMO_ARGS": demo, "T2_WBC": "ort", "DEMO_LOOP_MODULE": str(STUB_LOOP), **kw}


def test_t26_live_with_a_stub_loop_passes_the_session_gates(fk):
    s1_benches(fk)
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_soak.sh", env=soak_env(fk, T2_SOAK_DURATION="30"), timeout=300)
    assert r.returncode == 0, r.stdout[-4000:]
    sd, tt = fk.session_dir(), f"tcp://127.0.0.1:{fk.vla_port}"
    (run,) = list((sd / "runs_t26").glob("*_t26"))
    argv = json.loads((run / "argv.json").read_text())
    assert argv[argv.index("--mode") + 1] == "rt" and argv[argv.index("--vla") + 1] == tt and "--crn" in argv
    assert argv[argv.index("--d-init") + 1] == "3" and argv[argv.index("--runs-root") + 1] == str(sd / "runs_t26")
    verdict = json.loads((run / "verdict.json").read_text())
    assert verdict["pass"] and verdict["gates"] == "m2_rt", [c for c in verdict["checks"] if not c["pass"]]
    soak = json.loads((sd / "t26_soak.json").read_text())
    assert soak["pass"] and soak["vla_placement"]["device"] == "tensix"
    assert (run / "demo_logs").is_symlink() and (run / "demo_logs" / "loop.log").exists()
    g = yaml.safe_load((sd / "gates_m2.yaml").read_text())["m2_rt"]
    assert "viewer_k0_ms_max" not in g  # --no-viewer: the viewer rows are dropped, with a note


def test_t28_live_gates_the_drift(fk, tmp_path):
    """A 21 min stub run whose server time ramps 45 -> 52 ms: T-2.8 fails on the drift only (gates pass under a 55 ms
    VLA limit); the same run as T-2.6 reports the drift and passes."""
    bench = bench_session(tmp_path / "b", "x", True, "bfp8_b", 50.0, time.time())
    env = soak_env(fk, T2_SOAK_DURATION="1260", SOAK_STUB_RTT="45,52", T26_BENCH=str(bench))
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_soak_2h.sh", env=env, timeout=600)
    assert r.returncode == 1, r.stdout[-4000:]
    sd = fk.session_dir()
    soak = json.loads((sd / "t28_soak.json").read_text())
    drift = next(c for c in soak["checks"] if c["name"] == "vla_p50_drift")
    assert drift["gated"] and not drift["pass"] and not soak["pass"]
    (run,) = list((sd / "runs_t28").glob("*_t28"))
    assert json.loads((run / "verdict.json").read_text())["pass"]  # every m2_rt gate passed


# --- T-2.7 (t2_primary.sh) -------------------------------------------------------------------------------------------


def test_t27_dry_run_runs_both_arms_fresh(fk):
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_primary.sh", env={"T2_DRY_RUN": "1"}, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:]
    lines, tt = loop_lines(r.stdout), f"tcp://127.0.0.1:{fk.vla_port}"
    assert len(lines) == 10 and all("--mode rt --arm v1 --s 16 --B 0" in ln and "--crn" in ln for ln in lines)
    c4tt = [ln for ln in lines if ln.startswith("C4tt_")]
    c4gpu = [ln for ln in lines if ln.startswith("C4gpu_")]
    assert len(c4tt) == len(c4gpu) == 5
    assert all(f"--vla {tt} --d-init 3" in ln for ln in c4tt)
    assert all("--vla tcp://127.0.0.1:5565 --d-init 4" in ln for ln in c4gpu)
    assert sorted(ln.split("--seeds ")[1].split()[0] for ln in c4tt)[0] == "2000-2009"


@pytest.mark.parametrize("force", ["0", "1"])
def test_t27_refuses_t19s_gpu_arm_before_the_tt_arm(fk, tmp_path, force):
    """T-1.9's GPU runs (no info.noise: CUDA-seeded) are never a stand-in for T-2.7's GPU arm, forced or not."""
    d = tmp_path / "t19" / "20261024_120000_m1_primary_C4"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"vla_info": dict(SRC, backend="cuda:rtx5090:x@2a5b61cc")}))
    env = {"T2_GPU_ARM_DIRS": str(d), "T2_FORCE_REUSE": force, "T2_WBC": "ort"}
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_primary.sh", env=env, timeout=120)
    assert r.returncode == 2 and "not noise-paired" in r.stdout, r.stdout[-3000:]
    out = fk.session_dir() / "m2_primary"
    assert not (out / "tt_arm").exists()
    assert json.loads((out / "reuse_noise_check.json").read_text())["gpu_noise"] == ["None"]


def test_t27_refuses_an_unpaired_gpu_vla(fk, gpu_standin):
    r = fk.run("--vla", "tt", "--job", JOBS / "t2_primary.sh", env={"T2_GPU_VLA": gpu_standin("cuda"), "T2_WBC": "ort"},
               timeout=120)
    assert r.returncode == 2 and "does not draw server_g1's noise" in r.stdout, r.stdout[-3000:]
    assert not (fk.session_dir() / "m2_primary" / "runs").exists()


# --- every S1b job: the real-time cores, the session's port ----------------------------------------------------------


@pytest.mark.parametrize("job", ["t2_sync.sh", "t2_soak.sh", "t2_soak_2h.sh", "t2_primary.sh"])
def test_s1b_jobs_refuse_chip_safe_cpus(fk, job):
    cpus = _allowed_cpus()
    env = {"CHIP_SAFE_CPUS": ",".join(map(str, cpus)), "CHIP_VLA_EXTRA_ARGS": f"--cpus {cpus[0]}", "T2_DRY_RUN": "1"}
    r = fk.run("--vla", "tt", "--job", JOBS / job, env=env, timeout=120)
    assert r.returncode == 2 and "CHIP_SAFE_CPUS" in r.stdout and "real-time" in r.stdout, r.stdout[-2000:]


def test_demo_m2_job_uses_the_session_vla(fk):
    r = fk.run("--vla", "tt", "--job", JOBS / "demo_m2.sh", env={"DEMO_ARGS": "--dry-run --no-alt --tag x"})
    assert r.returncode == 0, r.stdout[-2000:]
    loop = next(ln for ln in r.stdout.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert f"--vla tcp://127.0.0.1:{fk.vla_port}" in loop


def test_nothing_in_the_s1b_path_assumes_s1s_port():
    """S1 served on :5577 to stay clear of the RT campaign; S1b uses the default :5557 (CHIP_VLA_PORT)."""
    for d in ("jobs", "scripts", "rttt", "configs"):
        for p in (RT / d).iterdir():
            if p.suffix in (".sh", ".py", ".yaml"):
                assert "5577" not in p.read_text(), p
    assert 'CHIP_VLA_PORT:=5557' in (RT / "scripts" / "chip_lib.sh").read_text()
