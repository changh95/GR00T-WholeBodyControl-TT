"""Helpers for the fault, CUDA-guard and request-schema tests (T-1.6, T-1.7, PLAN.md §1.4.1).

Servers bind 127.0.0.1 on a random free port (never 5555-5572) and are started and killed only
through the Popen objects made here, so a test can never signal a process it did not start (the
:5555 server in particular). Loop runs go through ``ProbeRun`` (tests/guard_probe.py: the loop CLI
in-process plus passive recorders).
"""

import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import time

import numpy as np

from loop_utils import R, RT, load_run

RESERVED_PORTS = range(5555, 5573)
SRV_PYTHON = os.environ.get("SRV_PYTHON", str(R / ".venv/bin/python"))
SIM_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID")


def free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in RESERVED_PORTS:
            return port


def _read_ready(p: subprocess.Popen, timeout_s: float, what: str) -> str:
    """The url= of the child's READY line (the mocks print one once bound)."""
    t_end = time.monotonic() + timeout_s
    while (rem := t_end - time.monotonic()) > 0:
        if not select.select([p.stdout], [], [], rem)[0]:
            break
        line = p.stdout.readline()
        if line.startswith("READY"):
            return dict(kv.split("=", 1) for kv in line.split()[1:])["url"]
        if not line:
            break
    p.kill()
    out, err = p.communicate(timeout=5)
    raise RuntimeError(f"{what} did not start: {err or out}")


def spawn_mock(mode: str, *args: str, port: int = 0, timeout_s: float = 10.0):
    """(process, url) of rtserve.mock_vla_server on `port` (0 = a free one)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RT)
    cmd = [sys.executable, "-m", "rtserve.mock_vla_server", "--port", str(port), "--mode", mode, *args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=RT,
                         env=env)
    return p, _read_ready(p, timeout_s, "mock_vla_server")


def spawn_fault_mock(mode: str, *args: str, port: int = 0, timeout_s: float = 10.0):
    """(process, url) of tests/fault_mock.py (jump / capture behaviours)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RT)
    cmd = [sys.executable, str(RT / "tests/fault_mock.py"), "--port", str(port), "--mode", mode, *args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=RT,
                         env=env)
    return p, _read_ready(p, timeout_s, "fault_mock")


def kill9(proc: subprocess.Popen, timeout_s: float = 10) -> None:
    """SIGKILL one of our own children and reap it (no cleanup handlers run in it)."""
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout_s)
    for f in (proc.stdout, proc.stderr):
        if f is not None:
            f.close()


class WbcServer:
    """rtserve.wbc_gpu_server under the server venv (what `srvpy -m ...` runs) on a given port."""

    def __init__(self, port: int, log_path: Path):
        self.port = port
        self.url = f"tcp://127.0.0.1:{port}"
        env = {k: v for k, v in os.environ.items() if k not in SIM_ENV_KEYS}
        env["PYTHONPATH"] = f"{RT}:{R}"
        self.log_path = Path(log_path)
        self.log = open(self.log_path, "a")
        self.proc = subprocess.Popen([SRV_PYTHON, "-m", "rtserve.wbc_gpu_server", "--port", str(port)],
                                     cwd=R, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout_s: float = 120) -> float:
        import zmq

        from rtloop import wire

        t0 = time.monotonic()
        ctx = zmq.Context.instance()
        while time.monotonic() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"wbc_gpu_server exited {self.proc.returncode}:\n{self.tail()}")
            s = ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, 500)
            s.connect(self.url)
            try:
                s.send(wire.request("ping"))
                if wire.loads(s.recv()).get("status") == "ok":
                    return time.monotonic() - t0
            except zmq.Again:
                pass
            finally:
                s.close(linger=0)
        raise TimeoutError(f"wbc_gpu_server not ready after {timeout_s} s:\n{self.tail()}")

    def tail(self, n: int = 30) -> str:
        self.log.flush()
        return "".join(self.log_path.read_text(errors="replace").splitlines(True)[-n:])

    def kill9(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(20)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        if not self.log.closed:
            self.log.close()


def display_usable(timeout_s: float = 20.0) -> bool:
    """True if $DISPLAY is set and GLFW can open it (what rtloop.viewer_proc needs for its window)."""
    if not os.environ.get("DISPLAY"):
        return False
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_EGL_DEVICE_ID"}
    env["MUJOCO_GL"] = "glfw"
    try:
        p = subprocess.run([sys.executable, "-c", "import glfw, sys; sys.exit(0 if glfw.init() else 1)"],
                           env=env, capture_output=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0


def run_dir_of(root: Path) -> Path | None:
    """The one run directory a LoopRun writes under its runs root (None until it exists)."""
    root = Path(root)
    if not root.is_dir():
        return None
    dirs = [d for d in root.iterdir() if d.is_dir() and not d.is_symlink()]
    return dirs[0] if len(dirs) == 1 else None


def committed_ticks(root: Path) -> int:
    """Ticks published so far by a live run (ticks.len is rewritten every 50 ticks and at episode ends)."""
    d = run_dir_of(root)
    try:
        return int((d / "ticks.len").read_text().split()[0]) if d is not None else 0
    except (FileNotFoundError, IndexError, ValueError):
        return 0


def load_request_extras(run_dir: Path) -> dict[int, dict]:
    """requests.jsonl, last line per seq (the jsonl-only keys: error, status, drain, bytes, ...)."""
    out: dict[int, dict] = {}
    for line in (Path(run_dir) / "requests.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[int(rec["seq"])] = rec
    return out


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, length)] of the runs of True in a 1-D bool array."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]



def spawn_stub_server(port: int, capture_dir: Path, timeout_s: float = 60.0):
    """(process, url) of tests/stub_policy_server.py: the stock PolicyServer + Gr00tSimPolicyWrapper
    (strict) around a model-free Gr00tPolicy. Same interpreter (simpy: CUDA hidden)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{RT}:{R}"
    cmd = [sys.executable, str(RT / "tests/stub_policy_server.py"), "--port", str(port),
           "--capture-dir", str(capture_dir)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=R,
                         env=env)
    return p, _read_ready(p, timeout_s, "stub_policy_server")


class ProbeRun:
    """One tests/guard_probe.py subprocess: ``rtloop.loop`` in-process with its own runs root and log.

    ``probe_args`` go to the probe, ``loop_args`` to the loop CLI (``--runs-root`` and ``--tag`` are
    added); ``env`` overrides the child's environment (T-1.7's CUDA-visible arm).
    """

    def __init__(self, tmp_path: Path, name: str, probe_args, loop_args, env: dict | None = None):
        self.name = name
        self.root = Path(tmp_path) / name
        self.out = Path(tmp_path) / f"{name}_probe"
        self.log_path = Path(tmp_path) / f"{name}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "w")
        e = dict(os.environ)
        e["PYTHONPATH"] = f"{RT}:{R}"
        e.pop("GR00T_WBC_SERVER", None)
        e.update(env or {})
        cmd = [sys.executable, str(RT / "tests/guard_probe.py"), "--out", str(self.out), *probe_args,
               "--", *loop_args, "--runs-root", str(self.root), "--tag", name]
        self.t_start_ns = time.monotonic_ns()
        self.proc = subprocess.Popen(cmd, cwd=R, env=e, stdout=self.log, stderr=subprocess.STDOUT)
        self.run_dir: Path | None = None
        self.rc: int | None = None

    def tail(self, n: int = 40) -> str:
        if not self.log.closed:
            self.log.flush()
        return "".join(self.log_path.read_text(errors="replace").splitlines(True)[-n:])

    def ticks(self) -> int:
        return committed_ticks(self.root)

    def finish(self, timeout_s: float, want_rc: int | None = 0) -> Path:
        """Wait for the exit (rc checked unless want_rc is None); returns the run dir."""
        try:
            self.rc = self.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self.kill()
            raise AssertionError(f"{self.name}: no exit after {timeout_s} s (hang)\n{self.tail()}") from None
        finally:
            if not self.log.closed:
                self.log.close()
        if want_rc is not None:
            assert self.rc == want_rc, f"{self.name}: exit {self.rc}\n{self.tail()}"
        self.run_dir = run_dir_of(self.root)
        assert self.run_dir is not None, f"{self.name}: no run dir under {self.root}"
        return self.run_dir

    def load(self):
        """(ticks, requests, episodes) of the finished run."""
        return load_run(self.run_dir)

    def probe(self) -> dict:
        return json.loads((self.out / "probe.json").read_text())

    def obs(self) -> dict:
        with np.load(self.out / "obs.npz") as z:
            return {k: z[k] for k in z.files}

    def splicer_trace(self) -> np.ndarray:
        """(n, 6) int64 rows (ep, k, row, starved, starve_n, starve_fault), one per select()."""
        with np.load(self.out / "splicer.npz") as z:
            return z["trace"]

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)
        if not self.log.closed:
            self.log.close()

