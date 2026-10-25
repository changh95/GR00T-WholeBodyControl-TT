"""scripts/demo.sh, scripts/record_demo.sh, jobs/demo_m{2,3}.sh (PLAN.md WP-1.7, §0.2, T-1.4, T-1.5, T-2.6,
T-3.3, T-3.9): syntax, the --dry-run plans of every milestone, and two live runs of demo.sh with a stub loop
(tests/hud/stub_loop.py) and the headless HUD on random ports. Nothing here starts a GPU server, opens the
Tenstorrent device, opens a window or touches ports 5555-5572."""

import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RT = HERE.parents[1]
DEMO = RT / "scripts" / "demo.sh"
RECORD = RT / "scripts" / "record_demo.sh"
JOBS = [RT / "jobs" / "demo_m2.sh", RT / "jobs" / "demo_m3.sh"]
# Never a real-time CPU (2-4, 6, 7, 12-15) or its SMT sibling (18-20, 22, 23, 28-31): PLAN.md §1.1.
BG_CPUS = os.environ.get("RTLOOP_TEST_CPUS", "0,1,8-11,16,17,24-27")
AUX_CPUS = os.environ.get("RTLOOP_TEST_AUX_CPUS", "26,27")  # the demo's HUD and helpers in the live tests
LOW_PRIO = ["nice", "-n", "19", "taskset", "-c", BG_CPUS]


def sh(*args, env=None, timeout=60, check=True):
    e = dict(os.environ)
    e.pop("DISPLAY", None)
    e.update(env or {})
    p = subprocess.run(["bash", *map(str, args)], env=e, capture_output=True, text=True, timeout=timeout)
    if check:
        assert p.returncode == 0, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    return p


def dry(*args, **kw):
    return sh(DEMO, *args, "--dry-run", **kw).stdout


def cmd_lines(out):
    return [ln for ln in out.splitlines() if ln.strip() and not ln.lstrip("# ").startswith(("NOTE", "require"))]


@pytest.mark.parametrize("path", [DEMO, RECORD, *JOBS])
def test_syntax_and_executable(path):
    sh("-n", path)
    assert os.access(path, os.X_OK)


# --- dry runs -------------------------------------------------------------------------------------


def gpu_vla_line(out):
    return next(ln for ln in out.splitlines() if "rtserve.gpu_vla_server" in ln and not ln.startswith("#"))


def test_m1_dry_run_plan():
    out = dry("m1", "--seeds", "1000-1019", "--hud", "--record", "--on-top", "--tag", "t14", "--gates", "m1_rt")
    assert re.search(r"^taskset -c 4 srvpy -m rtserve\.wbc_gpu_server --port 5556 .*--cpus 4", out, re.M)
    assert re.search(r"^HF_HUB_OFFLINE=1 taskset -c 12-15 srvpy -m rtserve\.gpu_vla_server --port 5565 --model "
                     r"\S+/models--cloudwalk-research--GR00T-N1\.6-G1-PnPAppleToPlate/snapshots/", out, re.M)
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    for part in ("--mode rt", "--arm v1", "--vla tcp://127.0.0.1:5565", "--wbc tcp://127.0.0.1:5556",
                 "--pub tcp://127.0.0.1:5570", "--viewer", "--seeds 1000-1019", "--tag t14"):
        assert part in loop, part
    assert re.search(r"^taskset -c 6,7 simpy \S+/scripts/hud\.py --sub tcp://127\.0\.0\.1:5570 "
                     r"--control tcp://127\.0\.0\.1:5571", out, re.M)
    assert re.search(r"^taskset -c 6,7 simpy \S+/scripts/keep_viewer_on_top\.py", out, re.M)
    assert re.search(r"^\S+/scripts/record_demo\.sh --out \S+/demo\.mp4 --display :1", out, re.M)
    assert "analyze_run.py <run dir> --gates m1_rt" in out
    assert "XAUTHORITY=/run/user/1000/gdm/Xauthority" in out
    assert "5555" not in out


def test_m1_dry_run_never_manages_5555():
    out = dry("m1", "--vla", "tcp://127.0.0.1:5555", "--wbc", "tcp://127.0.0.1:5555", "--wbc-timeout-ms", "200",
              "--wbc-no-latch", "--seeds", "1000-1001", "--tag", "t15")
    assert "srvpy" not in out and "kill" not in out.lower().replace("sigkill after 10 s", "")
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--vla tcp://127.0.0.1:5555 --wbc tcp://127.0.0.1:5555" in loop
    assert loop.split(" 2>&1")[0].rstrip().endswith("--wbc-timeout-ms 200 --wbc-no-latch")  # verbatim


def test_m1_no_viewer_and_ort():
    out = dry("m1", "--no-viewer", "--wbc", "ort", "--duration", "120")
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--viewer" not in loop and "--wbc ort" in loop and "--duration 120" in loop
    assert "wbc_gpu_server" not in out and "gpu_vla_server" in out


def test_m2_dry_run_is_one_dev_hold():
    out = dry("m2", "--duration", "1800", "--tag", "t26", "--hud")
    first = next(ln for ln in out.splitlines() if "chip_session.sh" in ln and not ln.startswith("#"))
    assert first.startswith("DEMO_ARGS=--duration\\ 1800\\ --tag\\ t26\\ --hud\\ ")
    assert "/bin/with-device.sh " in first and "WITH_DEVICE_RESET_AFTER" not in first
    assert first.rstrip().endswith("chip_session.sh --x280 none --vla tt --job " + str(RT / "jobs/demo_m2.sh"))
    inner = "\n".join(ln[4:] for ln in out.splitlines() if ln.startswith("#   "))
    assert "(inside the device hold)" in inner and "require: tcp://127.0.0.1:5557" in inner
    loop = next(ln for ln in inner.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--vla tcp://127.0.0.1:5557 --wbc tcp://127.0.0.1:5556" in loop and "--duration 1800" in loop
    cmds = [ln for ln in inner.splitlines() if not ln.startswith("#")]
    assert not any("chip_session" in ln or "with-device" in ln for ln in cmds)  # DEV is never nested (D13)


def test_m3_dry_run_is_one_devx_hold_with_boot3():
    out = dry("m3", "--duration", "7200", "--swap-wbc-every", "10", "--allow-reset", env={"X280_E1F_APPROVED": "1"})
    first = next(ln for ln in out.splitlines() if "chip_session.sh" in ln and not ln.startswith("#"))
    assert "env WITH_DEVICE_RESET_AFTER=1 " in first and "/bin/with-device.sh " in first
    assert "chip_session.sh --x280 boot:3 --vla tt --allow-reset --job " + str(RT / "jobs/demo_m3.sh") in first
    # the stage-0 boot (E1F, every M3 session since 2026-10-25), with the approval handed to the session
    assert "X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS=serve\\ --port\\ 5558\\ --tile\\ 3\\ --stage0\\ " in first
    assert "--boot" not in first and "REFUSED" not in out
    inner = "\n".join(ln[4:] for ln in out.splitlines() if ln.startswith("#   "))
    assert "require: tcp://127.0.0.1:5557" in inner and "require: tcp://127.0.0.1:5558" in inner
    loop = next(ln for ln in inner.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--vla tcp://127.0.0.1:5557 --wbc tcp://127.0.0.1:5558" in loop


def test_sessions_enable_the_status_streams_with_the_hud():
    """HUD chip panel: with --hud, server_g1 and x280d publish their status (chip_session's CHIP_VLA_STATUS_PUB and
    CHIP_X280D_SERVE_ARGS); m3 always pins x280d to CPU 5 (§1.1) with chip_session's own boot:3 arguments."""
    first = next(ln for ln in dry("m3", "--hud").splitlines() if "chip_session.sh" in ln and not ln.startswith("#"))
    assert "CHIP_VLA_STATUS_PUB=tcp://127.0.0.1:5574 " in first
    assert "CHIP_X280D_SERVE_ARGS=serve\\ --port\\ 5558\\ --tile\\ 3\\ --stage0\\ --cpus\\ 5\\ --status-pub\\ " \
           "tcp://127.0.0.1:5573 " in first
    first = next(ln for ln in dry("m3").splitlines() if "chip_session.sh" in ln and not ln.startswith("#"))
    assert "STATUS_PUB" not in first and "--status-pub" not in first and "--cpus\\ 5" in first
    first = next(ln for ln in dry("m2", "--hud").splitlines() if "chip_session.sh" in ln and not ln.startswith("#"))
    assert "CHIP_VLA_STATUS_PUB=tcp://127.0.0.1:5574 " in first and "CHIP_X280D" not in first
    first = next(ln for ln in dry("m2").splitlines() if "chip_session.sh" in ln and not ln.startswith("#"))
    assert "CHIP_" not in first.split("chip_session.sh")[0] and " '' " not in first  # no empty env word
    out = dry("m3", "--x280-sim", "--hud")
    assert re.search(r"x280d\.py serve --backend sim --port 5558 --sim-cpus 21 --status-pub tcp://127\.0\.0\.1:5573 ",
                     out)


def test_m3_x280_sim_needs_no_device():
    out = dry("m3", "--x280-sim", "--duration", "120", "--tag", "t33")
    assert "chip_session" not in out and "with-device" not in out
    # x280d on 5, the sim hart's spin thread on 21 (§1.1), never sharing a CPU (integrate:x280d attribution)
    assert re.search(r"^taskset -c 5,21 xpy \S+/x280/x280d\.py serve --backend sim --port 5558 --sim-cpus 21 ",
                     out, re.M)
    assert "gpu_vla_server --port 5565" in out
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--vla tcp://127.0.0.1:5565 --wbc tcp://127.0.0.1:5558" in loop


def test_swap_targets_follow_the_loop_cli():
    """With a loop that has --control/--vla-alt/--wbc-alt the HUD keys are live and the alternates start."""
    env = {"DEMO_LOOP_MODULE": str(HERE / "stub_loop.py")}
    out = dry("m3", "--x280-sim", "--hud", env=env)
    loop = next(ln for ln in out.splitlines() if "stub_loop.py" in ln and not ln.startswith("#"))
    assert "--control tcp://127.0.0.1:5571" in loop and "--wbc-alt gpu=tcp://127.0.0.1:5556" in loop
    assert "wbc_gpu_server --port 5556" in out and "--no-control" not in out
    out = dry("m2", "--in-session", env=env)
    loop = next(ln for ln in out.splitlines() if "stub_loop.py" in ln and not ln.startswith("#"))
    assert "--vla-alt gpu=tcp://127.0.0.1:5565" in loop and "gpu_vla_server --port 5565" in out
    out = dry("m2", "--in-session", "--no-alt", env=env)
    assert "--vla-alt" not in out and "gpu_vla_server" not in out


def test_gpu_vla_loads_dp1_offline_and_pairs_noise_outside_m1():
    """D9 / review:tt #7: the demo's GPU VLA loads DP1 from its local snapshot at SRC_SHA with HF_HUB_OFFLINE=1, as
    jobs/_common.sh does. D3 / review:tt #8: in m2 and m3 runs, which may be compared with TT, it draws server_g1's
    CPU noise (--noise cpu); m1 keeps the CUDA stream of the T-1.x baselines."""
    from rttt.convert_bf16 import DEFAULT_SRC

    m1 = gpu_vla_line(dry("m1"))
    assert m1.startswith("HF_HUB_OFFLINE=1 taskset -c 12-15 srvpy -m rtserve.gpu_vla_server ")
    assert f"--model {DEFAULT_SRC} " in m1 and "--noise" not in m1
    for args in (("m2", "--in-session"), ("m3", "--x280-sim"), ("m3", "--x280-sim", "--hud")):
        line = gpu_vla_line(dry(*args))
        assert line.startswith("HF_HUB_OFFLINE=1 ") and f"--model {DEFAULT_SRC} " in line, args
        assert " --noise cpu " in line, args
    out = dry("m2")  # the device hold's inner plan is the --in-session one
    inner = "\n".join(ln[4:] for ln in out.splitlines() if ln.startswith("#   "))
    assert " --noise cpu " in gpu_vla_line(inner)
    if DEFAULT_SRC.is_dir():
        assert "no local" not in dry("m1")
    out = dry("m1", "--vla", "tcp://127.0.0.1:5555")  # nothing started: no snapshot lookup, no server line
    assert "gpu_vla_server" not in out and "no local" not in out


def test_wbc_spin_is_the_default():
    """T-1.4 passed only with the busy-polling WBC server; --no-wbc-spin opts out."""
    assert re.search(r"^taskset -c 4 srvpy -m rtserve\.wbc_gpu_server --port 5556 .*--cpus 4 --spin", dry("m1"), re.M)
    assert "--spin" not in dry("m1", "--no-wbc-spin")
    out = dry("m1", "--wbc-spin")
    assert re.search(r"^taskset -c 4 srvpy -m rtserve\.wbc_gpu_server --port 5556 .*--cpus 4 --spin ", out, re.M)
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--wbc-spin" not in loop and "--spin" not in loop  # a server option, never passed to the loop


def test_shadow_runs_x280d_like_a_primary():
    """T-3.6 through the demo: --wbc ort --shadow $WBC_X280 starts the x280d sim (or, on the device, requires it)."""
    out = dry("m3", "--x280-sim", "--wbc", "ort", "--shadow", "tcp://127.0.0.1:5558")
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--wbc ort" in loop and "--shadow tcp://127.0.0.1:5558" in loop
    assert re.search(r"^taskset -c 5,21 xpy \S+/x280d\.py serve --backend sim --port 5558", out, re.M)
    out = dry("m3", "--in-session", "--wbc", "ort", "--shadow", "tcp://127.0.0.1:5558")
    assert "require: tcp://127.0.0.1:5558" in out and "x280d.py serve" not in out


def _fake_guard_home(tmp_path) -> Path:
    """A HOME whose experiments/gr00t/bin/with-device.sh (env.sh's $DEV, inside $DEVX) only records what it was given:
    demo.sh's device branch then reaches no real guard, flock or device whatever happens."""
    home = tmp_path / "home"
    guard = home / "experiments" / "gr00t" / "bin" / "with-device.sh"
    guard.parent.mkdir(parents=True)
    guard.write_text("#!/usr/bin/env bash\n"
                     "{ printf 'ARG %s\\n' \"$@\"; echo \"ENV X280_E1F_APPROVED=${X280_E1F_APPROVED:-}\";"
                     " echo \"ENV CHIP_X280D_SERVE_ARGS=${CHIP_X280D_SERVE_ARGS:-}\";"
                     " echo \"ENV WITH_DEVICE_RESET_AFTER=${WITH_DEVICE_RESET_AFTER:-}\"; } >\"$(dirname \"$0\")/called.txt\"\n")
    guard.chmod(0o755)
    return home


def test_m3_device_run_needs_the_e1f_approval_and_the_stage0_boot(tmp_path):
    """demo.sh m3 (device) boots with --stage0, so it refuses before the hold without X280_E1F_APPROVED=1 in the
    caller's environment, and refuses a CHIP_X280D_SERVE_ARGS without --stage0 (the method-A --boot is not used on
    silicon); approved, it hands chip_session that approval and the stage-0 serve arguments. --x280-sim needs
    neither (no device)."""
    home = _fake_guard_home(tmp_path)
    called = home / "experiments" / "gr00t" / "bin" / "called.txt"
    base = {"HOME": str(home), "X280_E1F_APPROVED": "", "CHIP_X280D_SERVE_ARGS": ""}
    p = sh(DEMO, "m3", "--hud", env=base, check=False)
    assert p.returncode == 2 and "REFUSED" in p.stderr and "X280_E1F_APPROVED=1" in p.stderr and "DP3(d)" in p.stderr
    assert not called.exists()
    out = sh(DEMO, "m3", "--dry-run", env=base).stdout  # the plan still prints, with the refusal noted
    assert "# NOTE: REFUSED without --dry-run: X280_E1F_APPROVED=1 is not set" in out
    p = sh(DEMO, "m3", env={**base, "X280_E1F_APPROVED": "1",
                            "CHIP_X280D_SERVE_ARGS": "serve --port 5558 --tile 3 --boot --cpus 5"}, check=False)
    assert p.returncode == 2 and "has no --stage0" in p.stderr and not called.exists()
    p = sh(DEMO, "m3", "--hud", "--duration", "60", env={**base, "X280_E1F_APPROVED": "1"}, check=False)
    assert p.returncode == 0, p.stderr[-2000:]
    rec = called.read_text().splitlines()
    assert "ENV X280_E1F_APPROVED=1" in rec and "ENV WITH_DEVICE_RESET_AFTER=1" in rec
    assert "ENV CHIP_X280D_SERVE_ARGS=serve --port 5558 --tile 3 --stage0 --cpus 5 --status-pub tcp://127.0.0.1:5573" in rec
    args = [ln[4:] for ln in rec if ln.startswith("ARG ")]
    assert args[:5] == [str(RT / "scripts" / "chip_session.sh"), "--x280", "boot:3", "--vla", "tt"]
    assert args[-2:] == ["--job", str(RT / "jobs" / "demo_m3.sh")]
    out = dry("m3", "--x280-sim", env=base)
    assert "REFUSED" not in out and "chip_session" not in out


def test_bad_milestone():
    p = sh(DEMO, "m9", check=False)
    assert p.returncode == 2 and "m1, m2 or m3" in p.stderr


@pytest.mark.parametrize("job,ms", [(JOBS[0], "m2"), (JOBS[1], "m3")])
def test_job_scripts_reenter_demo_in_session(job, ms):
    out = sh(job, env={"DEMO_ARGS": "--dry-run --tag job\\ test --hud"}).stdout
    assert f"# demo {ms} (inside the device hold)" in out and "_job test" in out


def test_record_dry_run_and_report():
    out = sh(RECORD, "--dry-run", "--out", "/tmp/x.mp4", "--display", ":1").stdout
    assert out.startswith("taskset -c 6\\,7 nice -n 10 ") or out.startswith("# NOTE: no ffmpeg")
    for part in ("-f x11grab", "-framerate 30", "-i :1+0\\,0", "-c:v libx264", "-threads 2", "/tmp/x.mp4",
                 "-g 60", "-movflags +frag_keyframe+empty_moov+default_base_moof"):  # plays even if ffmpeg is killed
        assert part in out, part
    assert not Path("/tmp/.X11-unix/X97").exists()
    p = sh(RECORD, "--out", "/tmp/never.mp4", "--display", ":97", check=False)  # no such X display
    assert p.returncode == 3 and "not recording" in p.stderr
    assert not Path("/tmp/never.mp4").exists()


# --- live: demo.sh with the stub loop and the headless HUD ---------------------------------------------


def free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if not 5555 <= port <= 5572:
            return port


class PingServer(threading.Thread):
    """Answers W-VLA/W-WBC `ping` on a random port (demo.sh checks every endpoint before starting)."""

    def __init__(self):
        import zmq

        super().__init__(daemon=True)
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.url = f"tcp://127.0.0.1:{self.port}"
        self.n = 0
        self._halt = threading.Event()

    def run(self):
        from rtloop import wire

        while not self._halt.is_set():
            if self.sock.poll(50):
                self.sock.recv()
                self.sock.send(wire.dumps({"status": "ok"}))
                self.n += 1

    def stop(self):
        self._halt.set()
        self.join(timeout=2)
        self.sock.close(linger=0)
        self.ctx.term()


@pytest.fixture
def live(tmp_path):
    ping = PingServer()
    ping.start()
    ports = {"pub": free_port(), "ctl": free_port()}
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    env.update(DEMO_LOG_ROOT=str(tmp_path / "demo_logs"), DEMO_AUX_CPUS=AUX_CPUS,
               DEMO_LOOP_MODULE=str(HERE / "stub_loop.py"), RTLOOP_RUNS_ROOT=str(tmp_path / "runs"),
               SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    marker = f"127.0.0.1:{ports['ctl']}"

    def start(duration, tag):
        cmd = [*LOW_PRIO, "bash", str(DEMO), "m1", "--vla", ping.url, "--wbc", "ort", "--no-viewer", "--no-alt",
               "--hud", "--pub-url", f"tcp://127.0.0.1:{ports['pub']}", "--control-url", f"tcp://{marker}",
               "--duration", str(duration), "--tag", tag]
        return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)

    yield start, ports, marker, tmp_path
    ping.stop()
    left = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.split()
    for pid in left:  # never leave anything behind, even if an assertion failed
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_for(pred, timeout):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def logdir(tmp_path, tag):
    dirs = sorted((tmp_path / "demo_logs").glob(f"*_{tag}"))
    return dirs[-1] if dirs else None


def run_dir(tmp_path, tag):
    d = logdir(tmp_path, tag)
    log = d / "loop.log" if d else None
    if not log or not log.exists():
        return None
    m = re.search(r"run dir (\S+)", log.read_text())
    return Path(m.group(1)) if m else None


def test_live_demo_runs_hud_applies_control_and_cleans_up(live):
    from rtloop import control as cp

    start, ports, marker, tmp = live
    p = start(8, "livetest")
    assert wait_for(lambda: run_dir(tmp, "livetest") is not None, 40), p.stdout and p.stdout.read()
    push = cp.ControlPush(f"tcp://{marker}")
    assert wait_for(lambda: push.send("swap_wbc", "next"), 10)
    time.sleep(0.3)
    for c in (("pause", None), ("pause", None), ("swap_vla", "next"), ("swap_wbc", "nope")):
        assert push.send(*c)
        time.sleep(0.2)
    push.close()
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out[-3000:]
    run = run_dir(tmp, "livetest")
    acts = [json.loads(ln) for ln in (run / "control.jsonl").read_text().splitlines()]
    kinds = [(a.get("cmd"), a.get("action"), a.get("arg")) for a in acts if "action" in a]
    assert ("swap_wbc", "swap_wbc", "ort") in kinds
    assert ("pause", "pause", None) in kinds and ("pause", "resume", None) in kinds
    assert ("swap_vla", "noop", "vla already primary") in kinds
    assert any(a[0] == "swap_wbc" and a[1] == "noop" and "unknown target" in a[2] for a in kinds)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["wbc"] == "ort" and summary["ticks"] > 100
    hud = json.loads((logdir(tmp, "livetest") / "hud_summary.json").read_text())
    assert hud["rx"]["tick"] > 100 and hud["bad"] == 0
    assert (run / "demo_logs").resolve() == logdir(tmp, "livetest").resolve()
    assert subprocess.run(["pgrep", "-f", marker], capture_output=True).returncode == 1  # nothing left


def test_live_demo_ctrl_c_tears_down(live):
    start, ports, marker, tmp = live
    p = start(120, "ctrlc")
    assert wait_for(lambda: (logdir(tmp, "ctrlc") or tmp / "x").joinpath("hud.log").exists()
                    and run_dir(tmp, "ctrlc") is not None, 40)
    time.sleep(2.0)
    os.killpg(p.pid, signal.SIGINT)  # what Ctrl-C in the terminal delivers to the foreground group
    t0 = time.monotonic()
    out, _ = p.communicate(timeout=60)
    assert time.monotonic() - t0 < 30
    assert p.returncode == 130, out[-3000:]
    assert "stopping rt_sim after this tick" in out
    run = run_dir(tmp, "ctrlc")
    assert json.loads((run / "summary.json").read_text())["ticks"] > 50  # the loop closed its logs
    assert (logdir(tmp, "ctrlc") / "hud_summary.json").exists()  # the HUD exited cleanly on SIGTERM
    assert subprocess.run(["pgrep", "-f", marker], capture_output=True).returncode == 1
