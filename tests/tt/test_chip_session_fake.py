"""chip_session.sh / start_vla.sh / tt_probe.py without the device (PLAN.md WP-2.4): every branch in the fake mode.

    simpy -m pytest $RT/tests/tt/test_chip_session_fake.py -q

The fake mode (CHIP_SESSION_FAKE=1) refuses to start unless every device command is stubbed and the device node is a
temp file, so these runs cannot reach /dev/tenstorrent; test_never_touches_device_or_lock proves it with strace.
Stubs: tests/tt/chip_fakes.py (x280d, tt-smi, tt_probe, fuser) and tests/tt/fake_g1_server.py (server_g1). Ports
are random free ones, never 5555-5572.
"""

import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest

RT = Path(__file__).resolve().parents[2]
SESSION = RT / "scripts" / "chip_session.sh"
START_VLA = RT / "scripts" / "start_vla.sh"
TT_PROBE = RT / "scripts" / "tt_probe.py"
FAKES = Path(__file__).resolve().parent / "chip_fakes.py"
FAKE_G1 = Path(__file__).resolve().parent / "fake_g1_server.py"
PY = sys.executable
REAL_LOCK = "/home/deepgadget/experiments/gr00t/.p150a.lock"


def free_port() -> int:
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        if not 5555 <= port <= 5572:
            return port


class Fake:
    """One fake session environment under tmp_path."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.dir = tmp / "fake"
        self.dir.mkdir()
        self.root = tmp / "sessions"
        self.vla_port, self.x280_port = free_port(), free_port()
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("CHIP_", "FAKE_", "WITH_DEVICE_", "TT_METAL_WATCHER", "TT_METAL_TRACE"))}
        self.env.update({
            "CHIP_SESSION_FAKE": "1", "FAKE_DIR": str(self.dir), "FAKE_EVENTS": str(self.dir / "events.log"),
            "CHIP_SESSION_ROOT": str(self.root), "CHIP_VLA_PORT": str(self.vla_port),
            "CHIP_X280_PORT": str(self.x280_port), "CHIP_DEV_NODE": str(self.dir / "devnode"),
            "CHIP_X280D_CMD": f"{PY} {FAKES} x280d", "CHIP_VLA_CMD": f"{PY} {FAKE_G1}",
            "CHIP_TT_PROBE_CMD": f"{PY} {FAKES} tt-probe", "CHIP_TTSMI_CMD": f"{PY} {FAKES} tt-smi",
            "CHIP_FUSER_CMD": f"{PY} {FAKES} fuser", "CHIP_PING_PY": PY, "CHIP_PY3": PY,
            "CHIP_RESET_WAIT_S": "0.1", "CHIP_X280_PROBE_TIMEOUT_S": "2", "CHIP_TEARDOWN_GRACE_S": "1",
            "CHIP_VLA_READY_TIMEOUT_S": "20", "CHIP_X280_READY_TIMEOUT_S": "20", "WITH_DEVICE_RESET_AFTER": "1",
        })

    def set(self, **files) -> "Fake":
        for name, value in files.items():
            (self.dir / name).write_text(str(value))
        return self

    def job(self, body: str, name: str = "job.sh") -> Path:
        p = self.tmp / name
        p.write_text("set -u\n" + body + "\n")
        return p

    def cmd(self, *args) -> list[str]:
        return ["bash", str(SESSION), *map(str, args)]

    def run(self, *args, env: dict | None = None, timeout: float = 60, prefix: list[str] | None = None):
        e = dict(self.env, **(env or {}))
        e = {k: v for k, v in e.items() if v is not None}
        return subprocess.run([*(prefix or []), *self.cmd(*args)], env=e, capture_output=True, text=True,
                              timeout=timeout)

    def events(self) -> list[str]:
        p = self.dir / "events.log"
        return [ln.split(" ", 1)[1] for ln in p.read_text().splitlines()] if p.exists() else []

    def session_dir(self) -> Path:
        dirs = [d for d in self.root.iterdir() if d.is_dir()]
        assert len(dirs) == 1, dirs
        return dirs[0]

    def session_events(self) -> list[dict]:
        return [json.loads(ln) for ln in (self.session_dir() / "events.jsonl").read_text().splitlines()]

    def session_log(self) -> str:
        return (self.session_dir() / "session.log").read_text()


@pytest.fixture
def fk(tmp_path):
    f = Fake(tmp_path)
    yield f
    assert_no_session_procs(f)


def session_ids(f: Fake) -> list[str]:
    if not f.root.exists():
        return []
    return ["_".join(d.name.split("_")[:3]) for d in f.root.iterdir() if d.is_dir()]


def procs_with(env_marker: str) -> list[int]:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if env_marker.encode() in (d / "environ").read_bytes().split(b"\0"):
                st = (d / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if st not in ("Z", "X"):
                    out.append(int(d.name))
        except OSError:
            pass
    return out


def assert_no_session_procs(f: Fake) -> None:
    for sid in session_ids(f):
        left = procs_with(f"CHIP_SESSION_ID={sid}")
        assert not left, f"session {sid} left processes behind: {left}"


def ok_job(f: Fake) -> Path:
    return f.job('echo "job ran"; exit 0')


def ping_job(f: Fake) -> Path:
    """Records the job's environment and pings every exported server."""
    return f.job(f"""
env | grep -E '^(CHIP_|DEVICE_LOCK_TIMEOUT|TT_METAL_)' | sort > "$CHIP_SESSION_DIR/job_env.txt"
for url in ${{CHIP_VLA_URL:-}} ${{CHIP_X280_URL:-}}; do
  {PY} - "$url" <<'EOF' || exit 9
import sys, zmq, msgpack
s = zmq.Context().socket(zmq.REQ); s.setsockopt(zmq.LINGER, 0); s.connect(sys.argv[1])
s.send(msgpack.packb({{"endpoint": "ping"}}))
sys.exit(0 if s.poll(3000) and "error" not in msgpack.unpackb(s.recv()) else 1)
EOF
done
exit 0""")


def job_env(f: Fake) -> dict:
    lines = (f.session_dir() / "job_env.txt").read_text().splitlines()
    return dict(ln.split("=", 1) for ln in lines if "=" in ln)


def index_of(evs: list[str], prefix: str) -> int:
    hits = [i for i, e in enumerate(evs) if e.startswith(prefix)]
    assert hits, f"no event {prefix!r} in {evs}"
    return hits[0]


# --- happy paths ----------------------------------------------------------------------------------------------------


def test_no_x280_no_vla_runs_job_only(fk):
    r = fk.run("--vla", "none", "--job", ping_job(fk))
    assert r.returncode == 0, r.stderr
    assert fk.events() == ["fuser " + fk.env["CHIP_DEV_NODE"]]  # only the teardown's device-user check
    env = job_env(fk)
    assert env["CHIP_VLA_URL"] == "" and env["CHIP_X280_URL"] == "" and env["DEVICE_LOCK_TIMEOUT"] == "1"
    idx = [json.loads(ln) for ln in (fk.root / "index.jsonl").read_text().splitlines()]
    assert idx[-1]["rc"] == 0 and idx[-1]["vla"] == "none"


def test_full_order_probe_x280d_vla_job_teardown(fk):
    fk.set(x280_state="clean")
    r = fk.run("--x280", "probe", "--vla", "tt", "--debug-endpoints", "--job", ping_job(fk),
               env={"TT_METAL_WATCHER": "10", "TT_METAL_TRACE_ALLOC_TRACKING": "1"})
    assert r.returncode == 0, r.stderr
    evs = fk.events()
    order = [index_of(evs, p) for p in ("x280d probe clean", "x280d serve start", "x280d serve ready",
                                        "server_g1 start", "server_g1 ready", "server_g1 stop", "x280d serve stop")]
    assert order == sorted(order), evs
    assert "--tile 3 --no-release" in evs[index_of(evs, "x280d serve start")]
    assert "--debug-endpoints" in evs[index_of(evs, "server_g1 start")]
    assert not any(e.startswith(("tt-smi", "tt-probe")) for e in evs)
    env = job_env(fk)
    assert env["CHIP_VLA_URL"] == f"tcp://127.0.0.1:{fk.vla_port}"
    assert env["CHIP_X280_URL"] == f"tcp://127.0.0.1:{fk.x280_port}"
    assert env["CHIP_X280_MODE"] == "probe" and env["CHIP_DEBUG_ENDPOINTS"] == "1"
    assert "TT_METAL_WATCHER" not in env and "TT_METAL_TRACE_ALLOC_TRACKING" not in env
    names = [e["event"] for e in fk.session_events()]  # every line is valid JSON
    assert names[0] == "session_start" and names[-1] == "session_end"


E1F = {"X280_E1F_APPROVED": "1"}  # the user's DP3(d) approval, which --x280 boot:3's default --stage0 needs


@pytest.mark.parametrize("mode,flag", [("boot:3", "--stage0"), ("attach", "--attach")])
def test_x280d_serve_args_per_mode(fk, mode, flag):
    """boot:3's default is the stage-0 boot (E1F, the standard boot since 2026-10-25), not method A's --boot."""
    fk.set(x280_state="clean")
    r = fk.run("--x280", mode, "--vla", "none", "--job", ok_job(fk), env=E1F)
    assert r.returncode == 0, r.stderr
    evs = fk.events()
    line = evs[index_of(evs, "x280d serve start")]
    assert line.endswith(f"--port {fk.x280_port} --tile 3 {flag} --cpus 5") and "--boot" not in line, line


@pytest.mark.parametrize("env", [{"X280_E1F_APPROVED": None}, {"X280_E1F_APPROVED": "0"}])
def test_boot3_default_refused_without_e1f_approval(fk, env):
    """Without CHIP_X280D_SERVE_ARGS, --x280 boot:3 (x280d serve --stage0) needs X280_E1F_APPROVED=1: exit 2 with a
    message naming it, before the probe, x280d, server_g1 or the job ran."""
    fk.set(x280_state="clean")
    r = fk.run("--x280", "boot:3", "--vla", "tt", "--job", ok_job(fk), env=env)
    assert r.returncode == 2, r.stderr
    assert "X280_E1F_APPROVED=1" in r.stderr and "--stage0" in r.stderr and "boot:3" in r.stderr, r.stderr
    assert fk.events() == [] and not fk.root.exists()
    # an explicit CHIP_X280D_SERVE_ARGS is passed through as given (x280d itself checks --stage0's approval)
    r = fk.run("--x280", "boot:3", "--vla", "none", "--job", ok_job(fk),
               env={**env, "CHIP_X280D_SERVE_ARGS": f"serve --port {fk.x280_port} --tile 3 --attach"})
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("env,tail", [
    (E1F, "--tile 3 --stage0 --cpus 5"),                                          # default: no status stream
    ({**E1F, "CHIP_X280_STATUS_PUB": ""}, "--tile 3 --stage0 --cpus 5"),           # empty: no status stream
    ({**E1F, "CHIP_X280_STATUS_PUB": "tcp://127.0.0.1:40123"},
     "--stage0 --cpus 5 --status-pub tcp://127.0.0.1:40123"),
    ({"CHIP_X280D_SERVE_ARGS": "serve --port {port} --tile 3 --boot --cpus 7"},   # an explicit --cpus is kept
     "--tile 3 --boot --cpus 7"),
    ({"CHIP_X280D_SERVE_ARGS": "serve --port {port} --tile 3 --boot --status-pub tcp://127.0.0.1:40124"},
     "--boot --status-pub tcp://127.0.0.1:40124 --cpus 5"),                       # and an explicit --status-pub
    # E1F's stage-0 boot, every M3 session's since 2026-10-25: passed through as given, placed like --boot
    ({"CHIP_X280D_SERVE_ARGS": "serve --port {port} --tile 3 --stage0", "X280_E1F_APPROVED": "1"},
     "--tile 3 --stage0 --cpus 5"),
    ({"CHIP_X280D_SERVE_ARGS": "serve --port {port} --tile 3 --stage0 --status-pub tcp://127.0.0.1:40125",
      "X280_E1F_APPROVED": "1"}, "--stage0 --status-pub tcp://127.0.0.1:40125 --cpus 5"),
])
def test_x280d_serve_cpus_and_status_pub(fk, env, tail):
    """x280d runs on CPU 5 (PLAN.md §1.1) and publishes the HUD's X280 status (§4.3 item 7) only when asked;
    the fake x280d binds neither (it only reads --port)."""
    fk.set(x280_state="clean")
    env = {k: v.format(port=fk.x280_port) for k, v in env.items()}
    r = fk.run("--x280", "boot:3", "--vla", "none", "--job", ping_job(fk), env=env)
    assert r.returncode == 0, r.stderr
    evs = fk.events()
    line = evs[index_of(evs, "x280d serve start")]
    assert line.endswith(tail) and line.count("--cpus") == 1 and line.count("--status-pub") <= 1, line
    assert job_env(fk)["CHIP_X280_STATUS_PUB"] == env.get("CHIP_X280_STATUS_PUB", "")
    if "CHIP_X280D_SERVE_ARGS" in env:  # the jobs read the session's serve flags (jobs/_x280.sh x280_boot_flag)
        assert job_env(fk)["CHIP_X280D_SERVE_ARGS"] == env["CHIP_X280D_SERVE_ARGS"]


def test_x280d_stays_on_chip_safe_cpus(fk):
    """A session confined to the non-real-time cores does not move x280d onto CPU 5."""
    fk.set(x280_state="clean")
    cpus = [c for c in _allowed_cpus() if c != 5][:2]
    r = fk.run("--x280", "probe", "--vla", "none", "--job", ok_job(fk),
               env={"CHIP_SAFE_CPUS": ",".join(map(str, cpus))})
    assert r.returncode == 0, r.stderr
    evs = fk.events()
    assert "--cpus" not in evs[index_of(evs, "x280d serve start")] and "excludes CPU 5" in fk.session_log()


def test_attach_accepts_attachable_probe_mode_does_not(fk):
    fk.set(x280_state="attachable")
    assert fk.run("--x280", "attach", "--vla", "none", "--job", ok_job(fk)).returncode == 0
    r = fk.run("--x280", "probe", "--vla", "none", "--job", ok_job(fk))
    assert r.returncode == 3, r.stderr


def test_multiple_jobs_and_job_rc(fk):
    a = fk.job('echo a >> "$CHIP_SESSION_DIR/ran"; exit 7', "a.sh")
    b = fk.job('echo b >> "$CHIP_SESSION_DIR/ran"; exit 0', "b.sh")
    r = fk.run("--vla", "tt", "--job", a, "--job", b)
    assert r.returncode == 7, r.stderr
    assert (fk.session_dir() / "ran").read_text().split() == ["a", "b"]
    assert (fk.session_dir() / "job.1.log").exists() and (fk.session_dir() / "job.2.log").exists()


# --- dirty X280 and the reset path ----------------------------------------------------------------------------------


def test_dirty_without_allow_reset_exits_3_before_any_child(fk):
    fk.set(x280_state="dirty")
    r = fk.run("--x280", "boot:3", "--vla", "tt", "--job", ok_job(fk), env=E1F)
    assert r.returncode == 3, r.stderr
    evs = fk.events()
    assert evs[0] == "x280d probe dirty"
    assert not any(e.startswith(("tt-smi", "x280d serve", "server_g1", "tt-probe")) for e in evs), evs
    assert not (fk.session_dir() / "job.1.log").exists()


def test_dirty_allow_reset_resets_reprobes_and_continues(fk):
    fk.set(x280_state="dirty_until_reset")
    r = fk.run("--x280", "probe", "--vla", "tt", "--allow-reset", "--job", ok_job(fk))
    assert r.returncode == 0, r.stderr
    evs = fk.events()
    order = [index_of(evs, p) for p in ("x280d probe dirty", "fuser", "tt-smi -r 0", "tt-probe", "x280d probe clean",
                                        "x280d serve start", "server_g1 start", "server_g1 stop", "x280d serve stop")]
    assert order == sorted(order), evs


def test_reset_refused_when_device_held(fk):
    fk.set(x280_state="dirty", fuser_pids="4242 4343")
    r = fk.run("--x280", "probe", "--vla", "none", "--allow-reset", "--job", ok_job(fk))
    assert r.returncode == 3, r.stderr
    assert not any(e.startswith("tt-smi") for e in fk.events())
    assert "not resetting" in fk.session_log()


def test_still_dirty_after_reset_exits_3(fk):
    fk.set(x280_state="dirty")
    r = fk.run("--x280", "probe", "--vla", "none", "--allow-reset", "--job", ok_job(fk))
    assert r.returncode == 3, r.stderr
    evs = fk.events()
    assert [e for e in evs if e.startswith("x280d probe")] == ["x280d probe dirty"] * 2
    assert evs.count("tt-smi -r 0") == 1  # one reset per session, never a loop


@pytest.mark.parametrize("files", [{"ttsmi_rc": 1}, {"tt_probe_rc": 1}])
def test_reset_or_tt_probe_failure_exits_4(fk, files):
    fk.set(x280_state="dirty_until_reset", **files)
    r = fk.run("--x280", "probe", "--vla", "none", "--allow-reset", "--job", ok_job(fk))
    assert r.returncode == 4, r.stderr
    assert not any(e.startswith("x280d serve") for e in fk.events())


@pytest.mark.parametrize("state,rc", [("hang", 3), ("exit3", 3), ("error", 4)])
def test_probe_hang_exit3_error(fk, state, rc):
    fk.set(x280_state=state)
    t0 = time.monotonic()
    r = fk.run("--x280", "probe", "--vla", "none", "--job", ok_job(fk))
    assert r.returncode == rc, r.stderr
    assert time.monotonic() - t0 < 15
    if state == "hang":
        assert "hung" in fk.session_log()


def test_hung_probe_then_reset_recovers(fk):
    fk.set(x280_state="hang_until_reset")  # hangs (killed at the 2 s timeout = dirty) until tt-smi ran
    r = fk.run("--x280", "probe", "--vla", "none", "--allow-reset", "--job", ok_job(fk))
    assert r.returncode == 0, r.stderr
    evs = fk.events()
    assert [e for e in evs if e.startswith("x280d probe")] == ["x280d probe hang", "x280d probe clean"], evs
    assert "tt-smi -r 0" in evs


# --- children failing ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("serve,rc", [("exit3", 3), ("die", 5)])
def test_x280d_serve_failure(fk, serve, rc):
    fk.set(x280_state="clean", x280_serve=serve)
    r = fk.run("--x280", "boot:3", "--vla", "tt", "--job", ok_job(fk), env=E1F)
    assert r.returncode == rc, r.stderr
    assert not any(e.startswith("server_g1") for e in fk.events())


def test_server_g1_dies_before_ready_stops_x280d(fk):
    fk.set(x280_state="clean")
    r = fk.run("--x280", "probe", "--vla", "tt", "--job", ok_job(fk), env={"FAKE_G1_DIE_BEFORE_READY": "1"})
    assert r.returncode == 5, r.stderr
    evs = fk.events()
    assert "server_g1 died" in evs and "x280d serve stop" in evs
    assert not (fk.session_dir() / "job.1.log").exists()


def test_server_g1_ready_timeout(fk):
    r = fk.run("--vla", "tt", "--job", ok_job(fk), env={"FAKE_G1_BOOT_S": "30", "CHIP_VLA_READY_TIMEOUT_S": "1"})
    assert r.returncode == 5, r.stderr
    assert "not READY" in fk.session_log()


def test_vla_port_busy(fk):
    s = socket.socket()
    s.bind(("127.0.0.1", fk.vla_port))
    s.listen(1)
    try:
        r = fk.run("--vla", "tt", "--job", ok_job(fk))
    finally:
        s.close()
    assert r.returncode == 5, r.stderr
    assert "already in use" in fk.session_log()


def test_job_exit_3_tears_down_and_propagates(fk):
    fk.set(x280_state="clean")
    second = fk.job('touch "$CHIP_SESSION_DIR/second_ran"', "second.sh")
    r = fk.run("--x280", "probe", "--vla", "tt", "--job", fk.job("exit 3"), "--job", second)
    assert r.returncode == 3, r.stderr
    evs = fk.events()
    assert "server_g1 stop" in evs and "x280d serve stop" in evs
    assert not (fk.session_dir() / "second_ran").exists()


def test_sigkill_after_grace_for_term_ignoring_children(fk):
    fk.set(x280_state="clean", x280_serve="ignore_term")
    r = fk.run("--x280", "probe", "--vla", "tt", "--job", ok_job(fk), env={"FAKE_G1_IGNORE_TERM": "1"})
    assert r.returncode == 0, r.stderr
    log = fk.session_log()
    assert "server_g1 still alive after 1s: SIGKILL" in log and "x280d still alive after 1s: SIGKILL" in log
    assert log.index("server_g1 still alive") < log.index("x280d still alive")  # reverse order


def test_server_died_mid_session_skips_next_job(fk):
    fk.set(x280_state="clean")
    kill = fk.job('read -r pid _ < "$CHIP_SESSION_DIR/x280d.pid"; kill -KILL "$pid"; sleep 0.5', "kill.sh")
    second = fk.job('touch "$CHIP_SESSION_DIR/second_ran"', "second.sh")
    r = fk.run("--x280", "probe", "--vla", "none", "--job", kill, "--job", second)
    assert r.returncode == 5, r.stderr
    assert not (fk.session_dir() / "second_ran").exists()


def test_orphan_left_by_job_is_killed(fk):
    body = "for i in 1 2; do setsid sleep 300 >/dev/null 2>&1 < /dev/null & done; disown -a; exit 0"
    r = fk.run("--vla", "none", "--job", fk.job(body))
    assert r.returncode == 0, r.stderr
    log = fk.session_log()
    assert "orphans of this session" in log and log.count("stopping orphan") == 2  # the fixture checks none is left


# --- start_vla.sh inside a job (S1c / E3 restarts) ------------------------------------------------------------------


def test_job_restarts_server_g1_via_start_vla(fk):
    job = fk.job("""
"$CHIP_START_VLA" status > "$CHIP_SESSION_DIR/status1" || exit 11
"$CHIP_START_VLA" stop || exit 12
if "$CHIP_START_VLA" status; then exit 13; fi
"$CHIP_START_VLA" tt || exit 14
"$CHIP_START_VLA" status > "$CHIP_SESSION_DIR/status2" || exit 15
exit 0""")
    r = fk.run("--vla", "tt", "--debug-endpoints", "--job", job)
    assert r.returncode == 0, r.stderr + r.stdout
    evs = [e for e in fk.events() if e.startswith("server_g1")]
    kinds = [e.split()[1] for e in evs]
    assert kinds == ["start", "ready", "stop", "start", "ready", "stop"], evs
    assert all("--debug-endpoints" in e for e in evs if e.startswith("server_g1 start"))
    sd = fk.session_dir()
    assert (sd / "status1").read_text() != (sd / "status2").read_text()
    assert (sd / "server_g1.1.log").exists() and (sd / "server_g1.2.log").exists()


def test_job_starts_server_g1_when_session_did_not(fk):
    r = fk.run("--vla", "none", "--job", fk.job('"$CHIP_START_VLA" tt || exit 14; exit 0'))
    assert r.returncode == 0, r.stderr
    assert [e.split()[1] for e in fk.events() if e.startswith("server_g1")] == ["start", "ready", "stop"]


def test_start_vla_refuses_outside_a_session(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHIP_")}
    r = subprocess.run(["bash", str(START_VLA), "tt"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "not inside a chip_session" in r.stderr


# --- signals --------------------------------------------------------------------------------------------------------


def test_sigterm_during_job_tears_everything_down(fk):
    fk.set(x280_state="clean")
    job = fk.job('touch "$CHIP_SESSION_DIR/job_started"; sleep 300')
    e = {k: v for k, v in fk.env.items()}
    p = subprocess.Popen(fk.cmd("--x280", "probe", "--vla", "tt", "--job", job), env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if fk.root.exists() and any((d / "job_started").exists() for d in fk.root.iterdir() if d.is_dir()):
                break
            time.sleep(0.1)
        else:
            pytest.fail("job never started")
        p.send_signal(signal.SIGTERM)
        out, err = p.communicate(timeout=30)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 128 + signal.SIGTERM, err
    evs = fk.events()
    assert "server_g1 stop" in evs and "x280d serve stop" in evs
    assert any(ev["event"] == "signal" for ev in fk.session_events())


# --- preconditions --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("env,msg", [({"WITH_DEVICE_RESET_AFTER": None}, "needs $DEVX"),
                                     ({"WITH_DEVICE_NO_PROBE": "1"}, "WITH_DEVICE_NO_PROBE")])
def test_x280_needs_devx_with_probe(fk, env, msg):
    r = fk.run("--x280", "probe", "--vla", "none", "--job", ok_job(fk), env=env)
    assert r.returncode == 2 and msg in r.stderr
    assert fk.events() == []


@pytest.mark.parametrize("drop", ["CHIP_X280D_CMD", "CHIP_VLA_CMD", "CHIP_TT_PROBE_CMD", "CHIP_TTSMI_CMD",
                                  "CHIP_FUSER_CMD"])
def test_fake_mode_requires_every_stub(fk, drop):
    r = fk.run("--vla", "none", "--job", ok_job(fk), env={drop: None})
    assert r.returncode == 2 and drop in r.stderr


def test_fake_mode_refuses_real_device_node(fk):
    r = fk.run("--vla", "none", "--job", ok_job(fk), env={"CHIP_DEV_NODE": "/dev/tenstorrent/0"})
    assert r.returncode == 2 and "outside /dev/tenstorrent" in r.stderr


@pytest.mark.parametrize("args,msg", [(["--x280", "boot:2"], "--x280 must be"), (["--vla", "gpu"], "--vla must be"),
                                      ([], "--job <script> is required"), (["--bogus"], "unknown argument")])
def test_usage_errors(fk, args, msg):
    extra = [] if not args or args == ["--bogus"] else ["--job", str(ok_job(fk))]
    r = fk.run(*args, *extra)
    assert r.returncode == 2 and msg in r.stderr


def test_refuses_tt_smi_in_arguments(fk):
    job = fk.job("exit 0", "check-tt-smi.sh")  # the guard would exec this command line without probe or reset
    r = fk.run("--vla", "none", "--job", job)
    assert r.returncode == 2 and "_device_guard.sh:76-78" in r.stderr


def test_real_mode_outside_the_guard_refuses(fk):
    # Real mode (no CHIP_SESSION_FAKE) still has every command stubbed and a temp node: belt and braces.
    r = fk.run("--vla", "none", "--job", ok_job(fk), env={"CHIP_SESSION_FAKE": None})
    assert r.returncode == 2 and "not inside a device hold" in r.stderr
    assert not fk.root.exists() or not any(fk.root.iterdir())


def _fake_guard_chain(tmp: Path) -> list[str]:
    """flock <tmp>/.p150a.lock bash <tmp>/_device_guard.sh ...: the process shape with-device.sh creates."""
    guard = tmp / "_device_guard.sh"
    guard.write_text('#!/usr/bin/env bash\n"$@" &\nchild=$!\ntrap \'kill -TERM $child\' TERM\nwait $child\n')
    return ["flock", "-w", "5", str(tmp / ".p150a.lock"), "bash", str(guard)]


def test_guard_check_accepts_the_guard_chain(fk):
    env = {"CHIP_GUARD_CHECK": "1"}
    r = fk.run("--vla", "none", "--job", ok_job(fk), env=env)
    assert r.returncode == 2 and "not inside a device hold" in r.stderr
    r = fk.run("--vla", "none", "--job", ok_job(fk), env=env, prefix=_fake_guard_chain(fk.tmp))
    assert r.returncode == 0, r.stderr
    assert "guard=" in fk.session_log() and "guard=none" not in fk.session_log()


def test_tt_probe_guard_check(tmp_path):
    # torch/ttnn shadowed by modules that exit 99: even a broken guard check could never reach a real device here
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    for mod in ("torch", "ttnn"):
        (shadow / f"{mod}.py").write_text("raise SystemExit(99)\n")
    env = dict(os.environ, PYTHONPATH=str(shadow))
    r = subprocess.run([PY, str(TT_PROBE), "--guard-check-only"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "refused" in r.stderr
    r = subprocess.run([*_fake_guard_chain(tmp_path), PY, str(TT_PROBE), "--guard-check-only"], env=env,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "guard ancestor pid" in r.stdout, r.stderr
    r = subprocess.run([PY, str(TT_PROBE)], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 2  # refused before the imports (99 would mean it got past the check)
    r = subprocess.run([*_fake_guard_chain(tmp_path), PY, str(TT_PROBE)], env=env, capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 99  # inside a (fake) hold it proceeds to the probe's imports


@pytest.mark.skipif(shutil.which("strace") is None, reason="strace not installed")
def test_never_touches_device_or_lock(fk):
    """A full fake session (reset path, both servers, a job that restarts server_g1, teardown) under strace -f."""
    fk.set(x280_state="dirty_until_reset")
    job = fk.job('"$CHIP_START_VLA" restart || exit 14; exit 0')
    trace = fk.tmp / "strace.txt"
    r = fk.run("--x280", "probe", "--vla", "tt", "--allow-reset", "--debug-endpoints", "--job", job, timeout=120,
               prefix=["strace", "-f", "-qq", "-e", "trace=%file", "-o", str(trace)])  # every syscall taking a path
    if r.returncode != 0 and "ptrace" in r.stderr:
        pytest.skip(f"strace cannot trace here: {r.stderr[-200:]}")
    assert r.returncode == 0, r.stderr[-2000:]
    text = trace.read_text()
    assert len(text) > 10000  # the trace really covered the run
    assert "/dev/tenstorrent" not in text
    assert REAL_LOCK not in text and ".p150a.lock" not in text


# --- the S1 jobs end to end on the fake (jobs/t2_fidelity.sh, t2_bench.sh, t2_abab.sh) -------------------------------

JOBS = RT / "jobs"


@pytest.fixture(scope="module")
def fake_gold(tmp_path_factory):
    from . import fake_golden

    return fake_golden.build(tmp_path_factory.mktemp("g") / "golden")


def s1_env(gold: Path, perturb: str) -> dict:
    return {"T2_GOLDEN": str(gold), "T2_BENCH_N": "20", "T2_BENCH_PERIOD_MS": "5", "T2_ABAB_N": "6",
            "FAKE_G1_PERTURB": perturb,
            "T2_CHECK_GOLDEN": "0"}  # the fake golden has no full manifest (rttt/check_golden.py needs one)


def test_s1_jobs_pass_on_a_faithful_fake(fk, fake_gold):
    r = fk.run("--vla", "tt", "--debug-endpoints", "--job", JOBS / "t2_fidelity.sh", "--job", JOBS / "t2_bench.sh",
               "--job", JOBS / "t2_abab.sh", env=s1_env(fake_gold, "1e-4"), timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    sd = fk.session_dir()
    fid = json.loads((sd / "t22_fidelity.json").read_text())
    assert fid["n_samples"] == 20 and all(v["pass"] for v in fid["verdicts"].values())
    for name in ("t23_bench.json", "t23_determinism.json", "t24_abab.json"):
        assert json.loads((sd / name).read_text())["pass"], name
    assert json.loads((sd / "t23_bench.json").read_text())["latency"]["n_ok"] == 20
    assert "5 passed" in (sd / "t22_pytest.txt").read_text()


def test_s1_fidelity_failure_does_not_stop_the_other_jobs(fk, fake_gold):
    r = fk.run("--vla", "tt", "--debug-endpoints", "--job", JOBS / "t2_fidelity.sh", "--job", JOBS / "t2_abab.sh",
               env=s1_env(fake_gold, "0.05"), timeout=300)
    assert r.returncode == 1, r.stderr[-3000:]
    sd = fk.session_dir()
    assert not json.loads((sd / "t22_fidelity.json").read_text())["verdicts"]["primary"]["pass"]
    assert json.loads((sd / "t24_abab.json").read_text())["pass"]


@pytest.mark.parametrize("job", ["t2_fidelity.sh", "t2_bench.sh", "t2_abab.sh"])
def test_s1_jobs_need_debug_endpoints(fk, fake_gold, job):
    r = fk.run("--vla", "tt", "--job", JOBS / job, env=s1_env(fake_gold, "0"))
    assert r.returncode == 2 and "--debug-endpoints" in r.stdout  # job output reaches the console via its tee


@pytest.mark.parametrize("job", ["t2_fidelity.sh", "t2_bench.sh", "t2_abab.sh"])
def test_jobs_need_a_session_and_a_vla(fk, job):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHIP_")}
    r = subprocess.run(["bash", str(JOBS / job)], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "CHIP_SESSION_DIR" in r.stderr
    r = fk.run("--vla", "none", "--job", JOBS / job)
    assert r.returncode == 2 and "--vla tt" in r.stdout
    assert "--vla tt" in (fk.session_dir() / "job.1.log").read_text()


def test_s1_fidelity_validates_the_golden_first(fk, fake_gold):
    """T2_CHECK_GOLDEN=1 (the default) runs rttt/check_golden.py before T-2.2: a golden that fails its own manifest
    (the fake one has none) is a misconfiguration, exit 2, and T-2.2 never runs."""
    r = fk.run("--vla", "tt", "--debug-endpoints", "--job", JOBS / "t2_fidelity.sh",
               env=dict(s1_env(fake_gold, "1e-4"), T2_CHECK_GOLDEN="1"), timeout=120)
    assert r.returncode == 2 and "fails its manifest" in r.stdout
    sd = fk.session_dir()
    assert (sd / "check_golden.txt").exists() and not (sd / "t22_fidelity.json").exists()


def test_s1_fidelity_records_golden_hash_and_server_info(fk, fake_gold):
    r = fk.run("--vla", "tt", "--debug-endpoints", "--job", JOBS / "t2_fidelity.sh", env=s1_env(fake_gold, "1e-4"),
               timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    sd = fk.session_dir()
    assert "manifest.json" in (sd / "golden.sha256").read_text()
    assert json.loads((sd / "server_g1_info.json").read_text())["backend"] == "fake:tt"
    # s1:device next step: TT vs fp32 per group in the report, every TT output in the session dir (S1's layout)
    import numpy as np

    rep = json.loads((sd / "t22_fidelity.json").read_text())
    assert all("tt_fp32_max" in g and "argmax_fp32" in g for g in rep["verdicts"]["groups"]["groups"].values())
    assert all("tt_fp32" in g for m in rep["per_sample"] for g in m["groups"].values())
    with np.load(sd / "t22_tt_outputs.npz") as z:
        assert len(z.files) == 20 * 8 and z["s00/action_pred_norm"].shape == (1, 50, 128)
        assert z["s19/action.navigate_command"].shape == (1, 30, 3)
    assert "TT-fp32" in (sd / "job.1.log").read_text()  # the per-group summary line


# --- review:tt fixes: SIGTERM during the READY wait, CHIP_SAFE_CPUS, the status PUB --------------------------------


def test_sigterm_during_server_g1_ready_wait_is_prompt(fk):
    """chip_wait_ready's ping helper runs in the background and is reaped with `wait`, so the TERM trap runs at once
    instead of after the helper's (up to CHIP_VLA_READY_TIMEOUT_S) poll; teardown stops the helper and the server."""
    env = dict(fk.env, FAKE_G1_BOOT_S="40", CHIP_VLA_READY_TIMEOUT_S="120")
    p = subprocess.Popen(fk.cmd("--vla", "tt", "--job", ok_job(fk)), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        t_end = time.monotonic() + 30
        while time.monotonic() < t_end:
            if fk.root.exists() and any((d / "server_g1.pid").exists() for d in fk.root.iterdir() if d.is_dir()):
                break
            time.sleep(0.1)
        time.sleep(1.0)
        t0 = time.monotonic()
        p.send_signal(signal.SIGTERM)
        p.communicate(timeout=30)
        took = time.monotonic() - t0
    finally:
        if p.poll() is None:
            p.kill()
            p.communicate()
    assert p.returncode == 128 + signal.SIGTERM and took < 5, (p.returncode, took)
    assert "signal TERM" in fk.session_log()


def _allowed_cpus() -> list[int]:
    return sorted(os.sched_getaffinity(0))


def _cpu_set(text: str) -> set[int]:
    """taskset's list format ('0,1,8-11') -> {0, 1, 8, 9, 10, 11}."""
    out: set[int] = set()
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def test_chip_safe_cpus_pins_session_and_jobs(fk):
    cpus = _allowed_cpus()[:2]
    job = fk.job('taskset -pc $$ | sed "s/.*: //" > "$CHIP_SESSION_DIR/job_cpus.txt"; exit 0')
    r = fk.run("--vla", "none", "--job", job, env={"CHIP_SAFE_CPUS": ",".join(map(str, cpus))})
    assert r.returncode == 0, r.stderr
    assert _cpu_set((fk.session_dir() / "job_cpus.txt").read_text()) == set(cpus)
    assert "(CHIP_SAFE_CPUS)" in fk.session_log()


def test_chip_safe_cpus_job_pins_itself_through_common(fk, tmp_path):
    """A job sourcing jobs/_common.sh pins itself even when started outside a pinned session (e.g. by a test)."""
    cpus = _allowed_cpus()[:1]
    sd = tmp_path / "sd"
    sd.mkdir()
    script = tmp_path / "j.sh"
    script.write_text(f'source {JOBS / "_common.sh"}\ntaskset -pc $$ | sed "s/.*: //"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHIP_")}
    r = subprocess.run(["bash", str(script)], env=dict(env, CHIP_SESSION_DIR=str(sd), CHIP_SAFE_CPUS=str(cpus[0])),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and r.stdout.strip() == str(cpus[0]), r.stdout + r.stderr
    r = subprocess.run(["bash", str(script)], env=dict(env, CHIP_SESSION_DIR=str(sd), CHIP_SAFE_CPUS="not-a-list"),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "cannot pin" in r.stderr


def test_chip_safe_cpus_invalid_refuses_session(fk):
    r = fk.run("--vla", "none", "--job", ok_job(fk), env={"CHIP_SAFE_CPUS": "x-y"})
    assert r.returncode == 2 and "CHIP_SAFE_CPUS" in r.stderr and not fk.root.exists()


def test_status_pub_reaches_server_g1_and_t2_bench(fk, fake_gold):
    url = f"tcp://127.0.0.1:{free_port()}"
    env = dict(s1_env(fake_gold, "1e-4"), CHIP_VLA_STATUS_PUB=url, T2_BENCH_PERIOD_MS="60")
    r = fk.run("--vla", "tt", "--debug-endpoints", "--job", JOBS / "t2_bench.sh", env=env, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:]
    assert any(f"--status-pub {url}" in e for e in fk.events())
    bench = json.loads((fk.session_dir() / "t23_bench.json").read_text())
    assert bench["status_stream"]["n"] >= 3
    assert all(c["pass"] for c in bench["checks"] if c["name"].startswith("status_"))


def test_nothing_in_the_s1b_path_assumes_s1s_port():
    """S1 served on :5577 to stay clear of the RT campaign; S1b uses the default :5557 (CHIP_VLA_PORT)."""
    for d in ("jobs", "scripts", "rttt", "configs"):
        for p in (RT / d).iterdir():
            if p.suffix in (".sh", ".py", ".yaml"):
                assert "5577" not in p.read_text(), p
    assert 'CHIP_VLA_PORT:=5557' in (RT / "scripts" / "chip_lib.sh").read_text()


# --- D6: server_g1 starts inside its --cpus with one-thread library pools -------------------------------------------


def _proc_status(pid: int, key: str) -> str:
    for ln in Path(f"/proc/{pid}/status").read_text().splitlines():
        if ln.startswith(key + ":"):
            return ln.split(":", 1)[1].strip()
    return ""


def _pid_job(f: Fake) -> Path:
    """Records server_g1's CPU list and thread-pool environment, as the session started it."""
    return f.job("""
pid=$(cut -d' ' -f1 "$CHIP_SESSION_DIR/server_g1.pid")
grep Cpus_allowed_list /proc/$pid/status | sed 's/.*:[[:space:]]*//' > "$CHIP_SESSION_DIR/vla_cpus.txt"
tr '\\0' '\\n' </proc/$pid/environ | grep -E '^(OPENBLAS|OMP|MKL)_NUM_THREADS=' | sort > "$CHIP_SESSION_DIR/vla_env.txt"
exit 0""")


def test_vla_start_pins_server_g1_to_its_cpus_with_one_thread_pools(fk):
    cpus = _allowed_cpus()[:2]
    r = fk.run("--vla", "tt", "--job", _pid_job(fk), env={"CHIP_VLA_EXTRA_ARGS": f"--cpus {cpus[0]},{cpus[1]}"})
    assert r.returncode == 0, r.stderr
    sd = fk.session_dir()
    assert _cpu_set((sd / "vla_cpus.txt").read_text()) == set(cpus)
    assert (sd / "vla_env.txt").read_text().split() == ["MKL_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                                                        "OPENBLAS_NUM_THREADS=1"]
    assert f"taskset -c {cpus[0]},{cpus[1]}" in fk.session_log()


def test_vla_start_keeps_the_callers_pools_and_inherits_with_empty_cpus(fk):
    r = fk.run("--vla", "tt", "--job", _pid_job(fk), env={"CHIP_VLA_EXTRA_ARGS": "--cpus=", "OMP_NUM_THREADS": "3"})
    assert r.returncode == 0, r.stderr
    sd = fk.session_dir()
    assert _cpu_set((sd / "vla_cpus.txt").read_text()) == set(_allowed_cpus())  # '' = inherit the session's
    assert "OMP_NUM_THREADS=3" in (sd / "vla_env.txt").read_text().split()
    assert "taskset" not in next(e["cmd"] for e in fk.session_events() if e["event"] == "started")


@pytest.mark.parametrize("extra,msg", [(None, "outside CHIP_SAFE_CPUS"), ("--cpus x-y", "is not a CPU list")])
def test_vla_start_refuses_bad_cpus_before_starting(fk, extra, msg):
    cpus = _allowed_cpus()
    env = {"CHIP_SAFE_CPUS": str(cpus[0]), "CHIP_VLA_EXTRA_ARGS": extra or f"--cpus {cpus[1]}"}
    r = fk.run("--vla", "tt", "--job", ok_job(fk), env=env)
    assert r.returncode == 5 and msg in r.stderr, r.stderr
    assert not any(e.startswith("server_g1 start") for e in fk.events())  # never exec'd, not even briefly


MK_ENV = Path.home() / "experiments" / "gr00t" / "publish" / "worktree-mk-env.sh"


def _server_g1_fake_model_cmd(tmp: Path) -> str:
    """CHIP_VLA_CMD for the real rttt/server_g1.py --fake-model under the MK env (chip_lib's real command plus the
    flag): the CPU-only server with the real imports, so its thread pools are the real ones."""
    w = tmp / "server_g1_fake.sh"
    w.write_text(f"""source {MK_ENV} >/dev/null && cd {RT} && export PYTHONPATH="${{PYTHONPATH:+$PYTHONPATH:}}{RT}" &&
exec python {RT / "rttt" / "server_g1.py"} --fake-model "$@"
""")
    return f"bash {w}"


def _info_job(f: Fake) -> Path:
    return f.job(f"""{PY} - "$CHIP_VLA_URL" "$CHIP_SESSION_DIR/info.json" <<'EOF'
import json, sys
sys.path.insert(0, "{RT}")
from rttt.bench_g1 import G1Client
c = G1Client(sys.argv[1], 10000)
open(sys.argv[2], "w").write(json.dumps(c.info(), default=str))
c.close()
EOF""")


@pytest.mark.skipif(not MK_ENV.exists(), reason="no MK env (publish/worktree-mk-env.sh)")
def test_server_g1_started_by_the_session_repins_no_thread(fk):
    """verify:all D6 / S1: affinity_check.repinned is [] for the real server_g1 (fake model) started by chip_session,
    while the same server started with the session's whole mask and default pools does re-pin its OpenBLAS pool."""
    cpus = _allowed_cpus()[:2]
    spec = f"{cpus[0]},{cpus[1]}"
    env = {"CHIP_VLA_CMD": _server_g1_fake_model_cmd(fk.tmp), "CHIP_VLA_EXTRA_ARGS": f"--cpus {spec}",
           "CHIP_VLA_READY_TIMEOUT_S": "120"}
    r = fk.run("--vla", "tt", "--job", _info_job(fk), env=env, timeout=180)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    info = json.loads((fk.session_dir() / "info.json").read_text())
    assert info["fake_model"] is True and info["affinity"] == cpus
    assert info["affinity_check"]["repinned"] == [] and info["affinity_check"]["cpus"] == cpus
    assert "re-pinned" not in (fk.session_dir() / "server_g1.1.log").read_text()
    # The control: no taskset, no pool caps (what chip_lib did before): the pool threads carry the whole mask.
    port = free_port()
    base = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}
    p = subprocess.Popen(["bash", str(fk.tmp / "server_g1_fake.sh"), "--port", str(port), "--log", "none",
                          "--cpus", spec, "--max-idle-s", "20"], env=base, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    try:
        t_end = time.monotonic() + 120
        out = ""
        while time.monotonic() < t_end and "READY" not in out:
            out += p.stdout.readline()
        assert "READY" in out, out
        if len(_allowed_cpus()) > 2:
            assert "re-pinned threads outside --cpus" in out, out
    finally:
        p.terminate()
        p.wait(30)
