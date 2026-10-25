"""The M3 session jobs end to end through chip_session.sh's fake mode, on a real x280d booted with --stage0 on the sim.

Since E1F (2026-10-25) every M3 session boots L2CPU3 with the stage-0 path (`serve --stage0`, X280_E1F_APPROVED=1),
so each device job must run on such a session exactly as on a --boot one. Here chip_session runs as under $DEVX
(probe, x280d serve, server_g1, the jobs in order, teardown) with every device command stubbed (CHIP_SESSION_FAKE=1):
x280d is x280/x280d.py with `--backend sim` put after its subcommand (a wrapper written per test; the chip is a
kmd.FakeDevice, so X280_E1F_APPROVED=1 approves nothing real), server_g1 is tests/tt/fake_g1_server.py or the mock
VLA, tt-smi / tt_probe / fuser are tests/tt/chip_fakes.py. The closed-loop jobs run as their --mode sync smokes
(m3_shadow, m3_eval) or with tests/x280/m3_stub_loop.py in place of rt_sim (m3_faults, m3_soak, demo_m3): never a
--mode rt loop. The sim chip lives in its x280d process, so an x280d started again (E3(d), T-3.8 case 1) finds no
released tile and refuses to attach: these tests check that it was started as `serve --attach` from the session's
--stage0 arguments and that the jobs report the refusal, not that the attach works (tests/x280/test_x280d_sim.py
attaches after a stage-0 boot in one process). Ports are random (never 5555-5574); everything runs on X280_CPUS at
nice 19, x280d's --cpus and its sim hart included (never CPU 5).

    source $RT/env.sh
    nice -n 19 taskset -c 0,1,8-11,16,17,24-27 bash -c 'source $RT/env.sh && simpy -m pytest $RT/tests/x280/test_m3_sessions_fake.py -q'
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT))

SESSION = RT / "scripts" / "chip_session.sh"
JOBS = RT / "jobs"
TT = RT / "tests" / "tt"
STUB_LOOP = Path(__file__).resolve().parent / "m3_stub_loop.py"
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
NICE = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]
XPY = RT / ".venv-x280" / "bin" / "python"
XPY = str(XPY) if XPY.exists() else sys.executable
PY = sys.executable
OBS516 = Path(os.environ.get("WBC_OBS516_NPZ", RT / "runs/x280/20261024_211935_obs516_kernel/obs516.npz"))
NEED = [RT / "x280" / "fw" / "build" / p for p in ("monitor.bin", "park_stub.bin", "stage0.bin",
                                                   "x86/libmonitor_x86.so", "kernel/wbc_fp16.blob",
                                                   "kernel/libwbc_mlp.so", "payloads/checksum.bin",
                                                   "payloads/hang.bin", "payloads/lim_bench.bin")]
if [p for p in NEED if not p.exists()]:
    pytest.skip(f"firmware/kernel build outputs missing: {[str(p) for p in NEED if not p.exists()]}",
                allow_module_level=True)
pytest.importorskip("zmq")
pytest.importorskip("msgpack")


def _cpu_set(spec: str) -> list:
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


CPUS = sorted(set(os.sched_getaffinity(0)) & set(_cpu_set(X280_CPUS))) or _cpu_set(X280_CPUS)
X280D_CPU, HART_CPU = CPUS[-2], CPUS[-1]  # x280d's --cpus and its sim hart: never CPU 5 (the device run's)
# bench --rtt's client (the device run pins rt_sim's 2,3); with fewer than 4 CPUs (X280_CPUS=0,1) it shares theirs
CLIENT_CPUS = f"{CPUS[-4]},{CPUS[-3]}" if len(CPUS) >= 4 else f"{CPUS[0]},{CPUS[-1]}"
assert 5 not in (X280D_CPU, HART_CPU)


def free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        if not 5555 <= p <= 5574:
            return p


def procs_with(marker: str) -> list:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if marker.encode() in (d / "environ").read_bytes().split(b"\0"):
                if (d / "stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X"):
                    out.append(int(d.name))
        except OSError:
            pass
    return out


class Session:
    """One fake chip_session environment: a sim x280d booted with --stage0 behind the x280d command."""

    def __init__(self, tmp: Path, vla: str = "g1"):
        self.tmp = tmp
        tmp.mkdir(parents=True)
        self.fake = tmp / "fake"
        self.fake.mkdir()
        self.root = tmp / "sessions"
        self.vla_port, self.x280_port = free_port(), free_port()
        wrap = tmp / "x280d_sim.sh"
        wrap.write_text(f"""#!/usr/bin/env bash
# x280d on the sim for chip_session's fake mode: --backend sim after the subcommand; a serve gets the sim hart's CPU
sub=$1
shift
sim=()
[ "$sub" = serve ] && sim=(--sim-cpus {HART_CPU})
exec env PYTHONPATH={RT} {XPY} {RT / "x280" / "x280d.py"} "$sub" --backend sim "${{sim[@]}}" "$@"
""")
        if vla == "g1":
            vla_cmd = f"{PY} {TT / 'fake_g1_server.py'}"
        else:  # the loops' W-VLA: rtserve.mock_vla_server, which takes only --port of server_g1's arguments
            mock = tmp / "mock_vla.sh"
            mock.write_text(f"""#!/usr/bin/env bash
port=""
while [ $# -gt 0 ]; do case $1 in --port) port=$2; shift 2 ;; *) shift ;; esac; done
exec env PYTHONPATH={RT} {PY} -m rtserve.mock_vla_server --mode hold --port "$port"
""")
            vla_cmd = f"bash {mock}"
        self.serve_args = (f"serve --port {self.x280_port} --tile 3 --stage0 --cpus {X280D_CPU} --deadline-ms 50")
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("CHIP_", "FAKE_", "WITH_DEVICE_", "TT_METAL_WATCHER", "TT_METAL_TRACE",
                                         "X280_", "M3_", "E4_", "T2_", "DEMO_"))
                    and k not in ("DISPLAY",)}
        self.env.update({
            "CHIP_SESSION_FAKE": "1", "FAKE_DIR": str(self.fake), "CHIP_SESSION_ROOT": str(self.root),
            "CHIP_VLA_PORT": str(self.vla_port), "CHIP_X280_PORT": str(self.x280_port),
            "CHIP_DEV_NODE": str(self.fake / "devnode"), "CHIP_X280D_CMD": f"bash {wrap}", "CHIP_VLA_CMD": vla_cmd,
            "CHIP_TT_PROBE_CMD": f"{PY} {TT / 'chip_fakes.py'} tt-probe",
            "CHIP_TTSMI_CMD": f"{PY} {TT / 'chip_fakes.py'} tt-smi", "CHIP_FUSER_CMD": f"{PY} {TT / 'chip_fakes.py'} fuser",
            "CHIP_PING_PY": PY, "CHIP_PY3": PY, "CHIP_TEARDOWN_GRACE_S": "5", "CHIP_X280_PROBE_TIMEOUT_S": "60",
            "CHIP_X280_READY_TIMEOUT_S": "120", "CHIP_VLA_READY_TIMEOUT_S": "60", "WITH_DEVICE_RESET_AFTER": "1",
            "X280_E1F_APPROVED": "1", "CHIP_X280D_SERVE_ARGS": self.serve_args,
            "X280_RTT_CLIENT_CPUS": CLIENT_CPUS,
            # the fake server_g1 on this run's CPUs: chip_lib.sh vla_start pins server_g1 to its --cpus, default 8-11
            "CHIP_VLA_EXTRA_ARGS": f"--cpus {','.join(map(str, CPUS))}",
        })

    def run(self, *args, timeout=900, **env) -> subprocess.CompletedProcess:
        e = {**self.env, **{k: str(v) for k, v in env.items() if v is not None}}
        e = {k: v for k, v in e.items() if v != "__unset__"}
        return subprocess.run(NICE + ["bash", str(SESSION), *map(str, args)], env=e, capture_output=True, text=True,
                              timeout=timeout)

    def dir(self) -> Path:
        dirs = [d for d in self.root.iterdir() if d.is_dir()]
        assert len(dirs) == 1, dirs
        return dirs[0]

    def events(self) -> list:
        return [json.loads(ln) for ln in (self.dir() / "events.jsonl").read_text().splitlines()]

    def json(self, name: str) -> dict:
        return json.loads((self.dir() / name).read_text())


@pytest.fixture
def session(tmp_path):
    made = []

    def make(vla: str = "g1") -> Session:
        s = Session(tmp_path / f"s{len(made)}", vla)
        made.append(s)
        return s
    yield make
    for s in made:
        if s.root.exists():
            for d in s.root.iterdir():
                if d.is_dir():
                    sid = "_".join(d.name.split("_")[:3])
                    assert not procs_with(f"CHIP_SESSION_ID={sid}"), f"session {sid} left processes behind"


def _fail(r) -> str:
    return f"rc={r.returncode}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"


def _started(s: Session, name: str) -> list:
    return [e["cmd"] for e in s.events() if e["event"] == "started" and e["name"] == name]


# --- the boot every M3 session uses ------------------------------------------------------------------------------


def test_stage0_session_boots_serves_and_needs_the_approval(session, tmp_path):
    """A --x280 boot:3 session with CHIP_X280D_SERVE_ARGS --stage0: x280d runs E1F's steps (no host LIM init), answers
    the jobs in mode 'booted' with SELFTEST passed; without X280_E1F_APPROVED=1 x280d refuses before READY."""
    s = session()
    job = tmp_path / "probe_job.sh"
    job.write_text(f"""source {JOBS / "_common.sh"}
source {JOBS / "_x280.sh"}
need_x280 boot:3
echo "BOOT=$(x280_boot_flag)"
x280_step st status && x280_step sel selftest
""")
    r = s.run("--x280", "boot:3", "--vla", "none", "--job", job, timeout=300)
    assert r.returncode == 0, _fail(r)
    assert "BOOT=stage0" in r.stdout + r.stderr
    st, sel = s.json("st.json"), s.json("sel.json")
    assert st["mode"] == "booted" and st["backend"] == "x280-sim:v3-fp16" and sel["pass"] and sel["lines"] == 30720
    log = (s.dir() / "x280d.log").read_text()
    assert "E1F step A" in log and "E1F step E" in log and "step 2: LIM init" not in log and "READY" in log
    assert _started(s, "x280d")[0].endswith(s.serve_args)  # the session's args as given: --cpus kept, no --cpus 5
    s2 = session()
    r = s2.run("--x280", "boot:3", "--vla", "none", "--job", job, timeout=300, X280_E1F_APPROVED="__unset__")
    assert r.returncode == 5, _fail(r)
    assert "DP3(d)" in (s2.dir() / "x280d.log").read_text()
    assert not (s2.dir() / "st.json").exists()  # the job never ran


# --- S2: T-3.E2, E2m, E3, E3(d) -----------------------------------------------------------------------------------


def test_s2_jobs_on_a_stage0_session(session):
    """jobs/s2_e2_e3.sh on a stage-0 session: E2 (status, heartbeat, SELFTEST without an RNMI, checksum and RNMI
    re-arm 10/10), E2m, E3 (fd churn, server_g1 start -> stop -> start under the heartbeat watch whose windows end
    with a PLL4 read), then E3(d): the new x280d is `serve --attach` from the session's --stage0 arguments (the sim
    chip died with the first x280d, so this attach is refused: reported as such). The sim cannot meet S2's device
    timing rows: its hart advances mtime per mailbox poll, not with time (x280/mailbox_sim.py MTIME_STEP), so
    mcycle/mtime reads 0 and the heartbeat moves in ~24 ms steps under the sampler; they are recorded, and gated only
    on silicon."""
    s = session()
    r = s.run("--x280", "boot:3", "--vla", "none", "--job", JOBS / "s2_e2_e3.sh", "--tag", "s2", timeout=900)
    assert r.returncode == 1, _fail(r)  # the sim's timing rows and E3(d)'s attach
    sm = s.json("s2_summary.json")
    assert sm["boot"] == "stage0" and sm["vlenb"] == 64 and sm["misa_V"], sm
    for k in ("hbrate_pass", "selftest_pass", "selftest_no_rnmi", "e2_pass", "e2m_recorded", "e3_vla_cycle_ok",
              "e3_watch_pll4_default"):
        assert sm[k] is True, (k, sm)
    assert sm["mcycle_per_mtime"] is not None and sm["e3_watch_max_gap_ms"] is not None  # recorded (sim values)
    assert sm["e3_watch_chunks"] >= 1 and sm["e3_relock_events"] == 0 and sm["e3_conclusive"] is False  # sim: none
    assert sm["e3d"] == "attach failed (see x280d.attach.log)" and not sm["pass"]
    fd, fd_tt = s.json("s2_e3_fd.json"), s.json("s2_e3_fd_tt.json")
    assert fd["pll4_default_after_each"] == [True] * 10  # after each legacy fd event
    assert fd_tt["pll4_default_after_each"] == [True] * 11  # 10 events and the end of its 1 s watch
    e2 = s.json("s2_e2.json")
    assert all(e2["checksum"]) and [x["n_reentry"] for x in e2["rearm"]] == list(range(1, 11))
    starts = _started(s, "x280d")
    assert len(starts) == 2 and "--stage0" in starts[0]
    again = starts[1].split("x280d_sim.sh ", 1)[1]
    assert again == (f"serve --port {s.x280_port} --tile 3 --cpus {X280D_CPU} --deadline-ms 50 --attach --log-dir "
                     f"{s.dir() / 'x280d_attach'}"), again
    assert "attach: L2CPU3 is not released" in (s.dir() / "x280d.attach.log").read_text()
    assert len(_started(s, "server_g1")) == 2  # E3's start -> stop -> start


# --- S3 + T-3.5 in one hold: E4, E5a, m3_rtt -------------------------------------------------------------------------


def test_s3_and_t35_jobs_in_one_stage0_hold(session, tmp_path):
    """jobs/e4_bench.sh, e5_parity.sh and m3_rtt.sh as one --vla tt hold on a stage-0 session (the grouping of the
    remaining sessions): E4's matrix and choice, the RTT client placed as asked (X280_RTT_CLIENT_CPUS), E5a vs x86 and
    the logged ORT action, T-3.5's PLAN gates (x280d side) with the client RTT reported apart."""
    import numpy as np

    if OBS516.exists():
        obs = OBS516
    else:  # a logged-shape file without ORT actions (x86 parity only)
        obs = tmp_path / "obs516.npz"
        rng = np.random.default_rng(3)
        np.savez(obs, tick=np.arange(64), obs=rng.standard_normal((64, 516)).astype(np.float32),
                 policy=np.array(["GR00T-WholeBodyControl-Balance", "GR00T-WholeBodyControl-Walk"] * 32))
    s = session()
    r = s.run("--x280", "boot:3", "--vla", "tt", "--job", JOBS / "e4_bench.sh", "--job", JOBS / "e5_parity.sh",
              "--job", JOBS / "m3_rtt.sh", "--tag", "s3", timeout=900, E4_ITERS=20, E4_RTT_N=150, E4_TT_N=10,
              X280_PARITY_OBS=obs, X280_PARITY_N=160, X280_PARITY_N_V0=40, M3_RTT_N=300)
    assert r.returncode in (0, 1), _fail(r)
    ch = s.json("e4_choice.json")
    assert ch["recommended"] in ("v2-fp16", "v2s-fp16", "v3-fp16", "v1-fp32"), ch
    assert not ch["tier"].startswith("shadow"), ch
    assert set(ch["v3_vs_v2"]) == {"GR00T-WholeBodyControl-Balance", "GR00T-WholeBodyControl-Walk"}, ch["v3_vs_v2"]
    want = sorted(int(c) for c in CLIENT_CPUS.split(","))
    assert ch["rtt_client"]["thread_cpus"] == [want[0]] and ch["rtt_client"]["io_cpu"] == want[-1], ch["rtt_client"]
    assert set(ch["t35_gates_busy"]) == {"x280d_p99_le_compute_p99_plus_500us", "x280d_p99_le_limit",
                                         "host_overhead_p99_le_300us", "no_timeouts_or_errors"}
    e5 = s.json("e5_summary.json")
    assert e5["pass"] and e5["boot"] == "stage0" and e5["obs"] == str(obs), e5
    assert e5["runs"]["session"]["rows"] == 160 and e5["runs"]["v0_fp32_balance"]["rows"] == 40
    assert e5["runs"]["v3_fp16"]["pass"] and e5["runs"]["v3_fp16"]["variant"] == "v3"
    t35 = s.json("t35_summary.json")
    assert t35["gates"].startswith("PLAN T-3.5") and set(t35["checks"]) == set(ch["t35_gates_busy"])
    # every call answered or counted (the sim's timing is not the X280's: its rows may fail here, never vacuously)
    assert t35["n"] == 300 and t35["ok"] + t35["timeouts"] + sum(t35["errors"].values()) == 300, t35
    assert t35["reported"]["client"]["thread_cpus"] == [want[0]] and "client_rtt_p99_le_limit" in t35["reported"]
    assert t35["pass"] == all(t35["checks"].values())
    full = s.json("m3_rtt_full.json")
    assert "client_rtt_p99_le_limit" not in full["checks"]  # reported, never a T-3.5 gate
    jobs = [e for e in s.events() if e["event"] == "job_end"]
    assert [e["rc"] for e in jobs][:2] == ["0", "0"], jobs  # E4 recorded a choice, E5a passed


# --- S4 + T-3.8 in one hold: m3_shadow, m3_eval (sync smokes), m3_faults (stub loop) ------------------------------


def test_s4_and_t38_jobs_in_one_stage0_hold(session, tmp_path):
    """T-3.6 and T-3.7 as their sync smokes, then T-3.8 with the stub loop, in one --vla tt hold on a stage-0 session:
    the shadow and the X280-primary loop through the stage-0 x280d, then case 1 kills x280d and restarts it as
    `serve --attach` from the --stage0 arguments (refused on the sim, so the job ends X280-dirty, exit 3)."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("yaml")
    bench = tmp_path / "t23_bench.json"
    bench.write_text(json.dumps({"latency": {"server_span_ms": {"p50": 43.9609265}}, "pass": True}))
    argv = tmp_path / "stub_argv.json"
    s = session(vla="mock")
    r = s.run("--x280", "boot:3", "--vla", "tt", "--job", JOBS / "m3_shadow.sh", "--job", JOBS / "m3_eval.sh",
              "--job", JOBS / "m3_faults.sh", "--tag", "s4", timeout=1200,
              M3_SHADOW_MODE="sync", M3_SHADOW_EPISODES=1, M3_SHADOW_MAX_TICKS=60,
              M3_SHADOW_LOOP_ARGS="--shadow-timeout-ms 100", M3_EVAL_MODE="sync", M3_EVAL_EPISODES=1,
              M3_EVAL_MAX_TICKS=60, M3_SKIP_SCREEN=1, M3_EVAL_LOOP_ARGS="--wbc-timeout-ms 100", M3_T23_BENCH=bench,
              M3_LOOP_MODULE=STUB_LOOP, M3_FAULT_GAP_S=3, M3_FAULT_DURATION=30, M3_STUB_SLEEP_S=30)
    assert r.returncode == 3, _fail(r)
    t36 = s.json("t36_summary.json")
    assert t36["pass"] and t36["dtype"] == "fp16" and t36["shadow_miss_frac"] == 0.0, t36
    t37 = s.json("t37_summary.json")
    assert t37["x280d_ok"] and t37["vla_p50_vs_t23"]["t23_span_p50_ms"] == pytest.approx(43.9609265), t37
    import yaml
    assert yaml.safe_load((s.dir() / "gates_m3.yaml").read_text())["m3_rt"]["vla_p50_ms_max"] == 46.159
    run = next((s.dir() / "runs_t38").glob("*_t38"))
    a = json.loads((run / "argv.json").read_text())
    assert a[:2] == ["--mode", "rt"] and "--wbc-no-latch" in a and a[a.index("--wbc") + 1] == f"tcp://127.0.0.1:{s.x280_port}"
    ev = [json.loads(x)["event"] for x in (s.dir() / "t38_events.jsonl").read_text().splitlines()]
    assert ev == ["kill9_x280d", "attach_start", "attach_failed"], ev
    starts = _started(s, "x280d")
    assert len(starts) == 2 and "--stage0" in starts[0]
    assert starts[1].endswith(f"--tile 3 --cpus {X280D_CPU} --deadline-ms 50 --attach --log-dir "
                              f"{s.dir() / 'x280d_attach'}") and "--stage0" not in starts[1]
    assert "attach: L2CPU3 is not released" in (s.dir() / "x280d.attach.log").read_text()
    assert any(e["event"] == "x280_dirty" and "could not attach" in e["why"] for e in s.events())


# --- T-3.9: the soak job and the demo job -------------------------------------------------------------------------


def test_t39_soak_job_on_a_stage0_session(session):
    """jobs/m3_soak.sh: demo.sh m3 --in-session with the stub loop on the session's x280d (stage 0) and VLA, WBC
    swapped every 2 s (x280 sim <-> ORT, --no-alt), then T-3.9's rows: the t39 gates and the x280d / swap / PLL4
    summary from x280d's own status log."""
    pytest.importorskip("yaml")
    s = session()
    r = s.run("--x280", "boot:3", "--vla", "tt", "--job", JOBS / "m3_soak.sh", "--tag", "t39", timeout=600,
              DEMO_LOOP_MODULE=STUB_LOOP, M3_SOAK_DURATION=14, M3_SOAK_SWAP_S=2,
              M3_SOAK_DEMO_ARGS="--no-viewer --no-alt")
    assert r.returncode == 0, _fail(r)
    sm = s.json("t39_summary.json")
    assert sm["pass"] and sm["gates_pass"] and sm["x280d_ok"] and sm["x280d_during_soak"]["calls"] > 0, sm
    assert sm["swap_da"]["swaps"] >= 6 and sm["swap_da"]["max_da"] == pytest.approx(7e-3)
    assert sm["pll4"]["reads"] >= 1 and sm["pll4"]["pass"] and sm["traps"]["pass"], sm
    run = next((s.dir() / "runs_t39").glob("*_t39"))
    a = json.loads((run / "argv.json").read_text())
    for part in ("--swap-wbc-every", "--vla", "--wbc", "--duration"):
        assert part in a, (part, a)
    assert a[a.index("--wbc") + 1] == f"tcp://127.0.0.1:{s.x280_port}" and a[a.index("--swap-wbc-every") + 1] == "2"
    v = json.loads((run / "verdict.json").read_text())
    assert v["gates"] == "t39" and {c["name"] for c in v["checks"]} >= {"period_p99_ms", "slips_max",
                                                                        "wbc_fallback_frac_max"}


def test_t39_soak_job_fails_a_fallback_and_a_large_swap_jump(session):
    pytest.importorskip("yaml")
    s = session()
    r = s.run("--x280", "boot:3", "--vla", "tt", "--job", JOBS / "m3_soak.sh", "--tag", "t39", timeout=600,
              DEMO_LOOP_MODULE=STUB_LOOP, M3_SOAK_DURATION=12, M3_SOAK_SWAP_S=2,
              M3_SOAK_DEMO_ARGS="--no-viewer --no-alt", M3_STUB_FALLBACKS=3, M3_STUB_SHADOW_DA=0.02)
    assert r.returncode == 1, _fail(r)
    sm = s.json("t39_summary.json")
    assert not sm["pass"] and not sm["gates_pass"] and not sm["swap_da"]["pass"] and sm["pll4"]["pass"], sm


def test_demo_m3_job_on_a_stage0_session(session):
    """jobs/demo_m3.sh (what `demo.sh m3` runs in its DEVX hold) on a stage-0 session: demo.sh m3 --in-session with
    the stub loop, the session's VLA and x280d."""
    s = session()
    url_v, url_x = f"tcp://127.0.0.1:{s.vla_port}", f"tcp://127.0.0.1:{s.x280_port}"
    demo_args = f"--no-viewer --no-alt --duration 3 --vla {url_v} --wbc {url_x} --tag d3 -- --runs-root {s.tmp / 'runs'}"
    r = s.run("--x280", "boot:3", "--vla", "tt", "--job", JOBS / "demo_m3.sh", timeout=300,
              DEMO_ARGS=demo_args, DEMO_LOOP_MODULE=STUB_LOOP, DEMO_LOG_ROOT=s.tmp / "demo_logs")
    assert r.returncode == 0, _fail(r)
    run = next((s.tmp / "runs").glob("*_d3"))
    a = json.loads((run / "argv.json").read_text())
    assert a[a.index("--vla") + 1] == url_v and a[a.index("--wbc") + 1] == url_x and "--mode" in a
    assert "--stage0" in _started(s, "x280d")[0]


# --- S3b: the 4-hart bench on its own session (jobs/s3b_mh.sh) -------------------------------------------------------


def test_s3b_four_hart_bench_session(session):
    """jobs/s3b_mh.sh on a stage-0 session with --harts 4 (X280_MH_APPROVED=1; FakeDevice only here): x280d writes harts
    1-3's vectors before the release, the job adopts them once, runs bench --mh (the quick plan on the sim) and writes
    s3b_mh.json with the scaling and the split verdict; the tile keeps serving on hart 0 throughout."""
    s = session()
    r = s.run("--x280", "boot:3", "--vla", "none", "--job", JOBS / "s3b_mh.sh", "--tag", "s3b", timeout=900,
              CHIP_X280D_SERVE_ARGS=s.serve_args + " --harts 4", X280_MH_APPROVED="1", S3B_PLAN="quick",
              S3B_REPEATS="1", S3B_BENCH_ITERS="20")
    assert r.returncode == 0, _fail(r)
    out = s.json("s3b_mh.json")
    assert out["adoption"]["pass"] and out["adoption"]["adopted"] == 0xE and out["bench_pass"] and not out["abort"]
    assert isinstance(out["split_worth_it"], bool) and out["verdict"].startswith("split ") and out["S4_K9"] is not None
    assert out["sim"] is True and out["configs"] and out["parity"]["pass"]
    assert {h: v["mncause"] for h, v in out["adoption"]["harts"].items()} == {h: (1 << 63) | 2 for h in ("1", "2", "3")}
    log = (s.dir() / "x280d.log").read_text()
    assert "harts 1-3 RNMI handler -> crt0 0x0800_0000" in log and "E1F step E" in log
    before, after = s.json("s3b_status_before.json"), s.json("s3b_status_after.json")
    assert before["tiles"]["3"]["mh"]["mh"] is None and after["tiles"]["3"]["mh"]["mh"]["adopted"] == 0xE


def test_s3b_refuses_without_the_secondary_vectors_or_the_approval(session):
    """Without --harts 4 in the serve arguments the job stops before any RNMI (exit 2: the boot's park-stub vectors would
    strand harts 1-3); with --harts 4 but no X280_MH_APPROVED=1, x280d refuses to serve before READY."""
    s = session()
    r = s.run("--x280", "boot:3", "--vla", "none", "--job", JOBS / "s3b_mh.sh", "--tag", "s3b", timeout=600,
              X280_MH_APPROVED="1")
    assert r.returncode == 2, _fail(r)
    assert not (s.dir() / "s3b_adopt.json").exists()
    s2 = session()
    r = s2.run("--x280", "boot:3", "--vla", "none", "--job", JOBS / "s3b_mh.sh", "--tag", "s3b", timeout=600,
               CHIP_X280D_SERVE_ARGS=s2.serve_args + " --harts 4", X280_MH_APPROVED="__unset__")
    assert r.returncode != 0 and "X280_MH_APPROVED" in (s2.dir() / "x280d.log").read_text(), _fail(r)
    assert not (s2.dir() / "s3b_adopt.json").exists()


# --- S5a: the read-only probe of L2CPU0-3 on a no-release session (jobs/s5a_probe.sh) ---------------------------------


def test_s5a_probe_session(session):
    """jobs/s5a_probe.sh as its header launches it: --x280 probe with the daemon's --probe-tiles 3,2,1,0, --vla none
    --debug-endpoints; the job probes all four tiles through that daemon, starts server_g1 (the fake) as the
    session's child, probes again while it serves and compares. No release (L2CPU_RESET 0x0F before and after both
    probes), both probes clean and equal field by field, TT digests identical; the TT p50 row is the fake's timing,
    so only its consistency with the verdict is checked."""
    s = session()
    args = f"serve --port {s.x280_port} --tile 3 --no-release --probe-tiles 3,2,1,0 --cpus {X280D_CPU}"
    r = s.run("--x280", "probe", "--vla", "none", "--debug-endpoints", "--job", JOBS / "s5a_probe.sh", "--tag", "s5a",
              timeout=900, CHIP_X280D_SERVE_ARGS=args, S5A_TT_N=10)
    assert r.returncode in (0, 1), _fail(r)
    assert _started(s, "x280d")[0].endswith(args) and len(_started(s, "server_g1")) == 1
    sm = s.json("s5a_summary.json")
    for name in ("probe_keeper", "probe_tt"):
        assert sm[name]["verdict"] == "clean" and sm[name]["rc"] == "0" and not sm[name]["reasons"], sm[name]
    assert sm["probes_equal"] and sm["probe_diffs"] == [] and sm["tt_bit_identical"] and sm["harvested"] == []
    assert sm["tt_p50_ms"][0] and sm["tt_p50_ms"][1] and sm["tt_max_ms_during_probe"] is not None
    assert sm["pass"] == sm["tt_p50_within_2pct"] and r.returncode == (0 if sm["pass"] else 1), sm["s5b"]
    for name in ("s5a_probe_keeper.json", "s5a_probe_tt.json"):
        p = s.json(name)
        assert p["probe"] == "s5a" and p["backend"] == "sim" and p["tiles_order"] == [3, 2, 1, 0]
        assert p["l2cpu_reset"] == p["l2cpu_reset_after"] == 0x0F and all(p["tiles"][t]["probed"] for t in "0123")
    log = r.stdout + r.stderr
    assert "S5a L2CPU0 (8,3): WAYENABLE 0" in log and ("S5a -> S5b: GO" in log or "S5a -> S5b: NO-GO: TT p50" in log)
    assert "no-release: keeper fd and windows up" in (s.dir() / "x280d.log").read_text()
    assert [e["rc"] for e in s.events() if e["event"] == "job_end"] == [str(r.returncode)]
