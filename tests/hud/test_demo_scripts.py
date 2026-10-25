"""scripts/demo.sh, scripts/record_demo.sh, jobs/demo_m3.sh (PLAN.md WP-1.7, §0.2, T-1.5, T-3.3, T-3.9): syntax,
the --dry-run plans (device hold, --in-session, --x280-sim), and two live runs of demo.sh with a stub loop
(tests/hud/stub_loop.py) and the headless HUD on random ports. Nothing here opens the Tenstorrent device, opens a
window or touches ports 5555-5572."""

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
JOBS = [RT / "jobs" / "demo_m3.sh"]
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


def test_dry_run_never_manages_5555():
    out = dry("m3", "--x280-sim", "--vla", "tcp://127.0.0.1:5555", "--wbc", "tcp://127.0.0.1:5555",
              "--wbc-timeout-ms", "200", "--wbc-no-latch", "--seeds", "1000-1001", "--tag", "t15")
    assert "xpy" not in out and "require" not in out and "kill" not in out.lower().replace("sigkill after 10 s", "")
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--vla tcp://127.0.0.1:5555 --wbc tcp://127.0.0.1:5555" in loop
    assert loop.split(" 2>&1")[0].rstrip().endswith("--wbc-timeout-ms 200 --wbc-no-latch")  # verbatim


def test_no_viewer_and_ort():
    out = dry("m3", "--x280-sim", "--no-viewer", "--wbc", "ort", "--duration", "120")
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--viewer" not in loop and "--wbc ort" in loop and "--duration 120" in loop
    assert "x280d.py serve" not in out  # --wbc ort: no x280d sim to start


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
    assert " '' " not in first  # no empty env word
    out = dry("m3", "--x280-sim", "--hud")
    assert re.search(r"x280d\.py serve --backend sim --port 5558 --sim-cpus 21 --status-pub tcp://127\.0\.0\.1:5573 ",
                     out)


def test_m3_x280_sim_needs_no_device():
    out = dry("m3", "--x280-sim", "--duration", "120", "--tag", "t33")
    assert "chip_session" not in out and "with-device" not in out
    # x280d on 5, the sim hart's spin thread on 21 (§1.1), never sharing a CPU (integrate:x280d attribution)
    assert re.search(r"^taskset -c 5,21 xpy \S+/x280/x280d\.py serve --backend sim --port 5558 --sim-cpus 21 ",
                     out, re.M)
    # the VLA is the TT one, which must already answer (never started here), as inside the device hold
    assert "# require: tcp://127.0.0.1:5557 answers ping" in out and "srvpy" not in out
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--vla tcp://127.0.0.1:5557 --wbc tcp://127.0.0.1:5558" in loop
    out = dry("m3", "--x280-sim", "--vla", "tcp://127.0.0.1:41000")  # --vla still overrides it
    loop = next(ln for ln in out.splitlines() if "rtloop.loop" in ln and not ln.startswith("#"))
    assert "--vla tcp://127.0.0.1:41000 --wbc tcp://127.0.0.1:5558" in loop and "require" not in out


def test_swap_targets_follow_the_loop_cli():
    """With a loop that has --control the HUD keys are live; the swap targets are the loop's own (the primary and
    ORT), so demo.sh registers none."""
    env = {"DEMO_LOOP_MODULE": str(HERE / "stub_loop.py")}
    out = dry("m3", "--x280-sim", "--hud", env=env)
    loop = next(ln for ln in out.splitlines() if "stub_loop.py" in ln and not ln.startswith("#"))
    assert "--control tcp://127.0.0.1:5571" in loop and "--no-control" not in out
    assert "--wbc-alt" not in loop and "--vla-alt" not in loop


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
    for ms in ("m9", "m1", "m2"):  # m1 and m2 (the GPU baselines) are gone
        p = sh(DEMO, ms, "--dry-run", check=False)
        assert p.returncode == 2 and "first argument must be m3" in p.stderr, ms


@pytest.mark.parametrize("job,ms", [(JOBS[0], "m3")])
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
        cmd = [*LOW_PRIO, "bash", str(DEMO), "m3", "--x280-sim", "--vla", ping.url, "--wbc", "ort", "--no-viewer",
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
