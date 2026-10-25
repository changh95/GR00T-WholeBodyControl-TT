"""Adversarial review of scripts/chip_session.sh in its fake mode (no device: every device command is a stub).

    simpy -m pytest -p no:cacheprovider $RT/tests/tt/review_m2_session.py -q

Uses the chip lane's stubs (tests/tt/chip_fakes.py, tests/tt/fake_g1_server.py). Ports are random, never 5555-5572.
"""

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

RT = Path(__file__).resolve().parents[2]
SESSION = RT / "scripts" / "chip_session.sh"
HERE = Path(__file__).resolve().parent
PY = sys.executable


def free_port() -> int:
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        if not 5555 <= port <= 5572:
            return port


def fake_env(tmp: Path) -> dict:
    d = tmp / "fake"
    d.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CHIP_", "FAKE_", "WITH_DEVICE_", "TT_METAL_WATCHER", "TT_METAL_TRACE"))}
    env.update({
        "CHIP_SESSION_FAKE": "1", "FAKE_DIR": str(d), "CHIP_SESSION_ROOT": str(tmp / "sessions"),
        "CHIP_VLA_PORT": str(free_port()), "CHIP_X280_PORT": str(free_port()), "CHIP_DEV_NODE": str(d / "devnode"),
        "CHIP_X280D_CMD": f"{PY} {HERE / 'chip_fakes.py'} x280d", "CHIP_VLA_CMD": f"{PY} {HERE / 'fake_g1_server.py'}",
        "CHIP_TT_PROBE_CMD": f"{PY} {HERE / 'chip_fakes.py'} tt-probe",
        "CHIP_TTSMI_CMD": f"{PY} {HERE / 'chip_fakes.py'} tt-smi", "CHIP_FUSER_CMD": f"{PY} {HERE / 'chip_fakes.py'} fuser",
        "CHIP_PING_PY": PY, "CHIP_PY3": PY, "CHIP_TEARDOWN_GRACE_S": "1", "CHIP_VLA_READY_TIMEOUT_S": "120",
        "FAKE_G1_BOOT_S": "40",  # server_g1 still building (bound, not serving), as in a cold S1 build
    })
    return env


def test_sigterm_while_waiting_for_server_g1_ready_tears_down_promptly(tmp_path):
    """The guard forwards SIGTERM to chip_session only (bin/_device_guard.sh:118). While chip_session waits for
    server_g1's READY (chip_wait_ready runs a foreground python ping loop for up to CHIP_VLA_READY_TIMEOUT_S, 3600 s
    by default), bash defers the TERM trap until that child exits, so the session keeps the device held and the
    build running instead of tearing down within the 10 s grace."""
    env = fake_env(tmp_path)
    job = tmp_path / "job.sh"
    job.write_text("exit 0\n")
    p = subprocess.Popen(["bash", str(SESSION), "--vla", "tt", "--job", str(job)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        root = Path(env["CHIP_SESSION_ROOT"])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if root.exists() and any((d / "server_g1.pid").exists() for d in root.iterdir() if d.is_dir()):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("server_g1 never started")
        time.sleep(1.0)
        t0 = time.monotonic()
        p.send_signal(signal.SIGTERM)
        try:
            p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        took = time.monotonic() - t0
    finally:
        if p.poll() is None:
            p.kill()
            p.communicate()
        # the fake server_g1 is a setsid child: kill it if chip_session did not
        for d in Path(env["CHIP_SESSION_ROOT"]).iterdir():
            pf = d / "server_g1.pid"
            if pf.exists():
                try:
                    os.killpg(int(pf.read_text().split()[0]), signal.SIGKILL)
                except (ProcessLookupError, ValueError, PermissionError):
                    pass
    assert took < 10, f"chip_session needed {took:.1f} s to act on SIGTERM during the READY wait (rc {p.returncode})"
