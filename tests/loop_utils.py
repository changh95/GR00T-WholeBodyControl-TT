"""Helpers for the rt_sim integration tests: a mock VLA in a subprocess and loop runs in subprocesses.

Every server binds a random free port (49152-65535, never 5555-5572); every run writes under the
test's tmp_path. The loop runs as `python -m rtloop.loop` with the same interpreter as the test
(simpy: CUDA hidden, EGL on GPU 0), so a run is exactly what the CLI does.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from rtloop.schema import Q43_GROUP_INDICES, S43, STATE_GROUPS, STATE_SLICES
from rtloop.telemetry import load_episodes, load_requests, load_ticks

RT = Path(__file__).resolve().parents[1]
R = Path(os.environ.get("R") or Path.home() / "experiments/gr00t/Isaac-GR00T-n1.6")


def start_mock(mode: str, *args: str):
    """(process, url) of a mock_vla_server CLI on a free port."""
    from rtserve.mock_vla_server import spawn

    return spawn(mode, *args, python=sys.executable)


def stop(proc: subprocess.Popen, timeout_s: float = 10) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout_s)
    for f in (proc.stdout, proc.stderr):
        if f is not None:
            f.close()


class LoopRun:
    """One `python -m rtloop.loop` subprocess with its own runs root and log file."""

    def __init__(self, tmp_path: Path, name: str, *args: str):
        self.name = name
        self.root = Path(tmp_path) / name
        self.log_path = Path(tmp_path) / f"{name}.log"
        self.log = open(self.log_path, "w")
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{RT}:{R}"
        env.pop("GR00T_WBC_SERVER", None)
        cmd = [sys.executable, "-m", "rtloop.loop", *args, "--runs-root", str(self.root), "--tag", name]
        self.proc = subprocess.Popen(cmd, cwd=R, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def tail(self, n: int = 40) -> str:
        self.log.flush()
        return "".join(self.log_path.read_text(errors="replace").splitlines(True)[-n:])

    def finish(self, timeout_s: float) -> Path:
        """Wait for exit 0 and return the run dir (the only one under this run's root)."""
        try:
            rc = self.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(10)
            raise AssertionError(f"{self.name}: no exit after {timeout_s} s\n{self.tail()}")
        finally:
            self.log.close()
        assert rc == 0, f"{self.name}: exit {rc}\n{self.tail()}"
        dirs = [d for d in self.root.iterdir() if d.is_dir() and not d.is_symlink()]
        assert len(dirs) == 1, dirs
        return dirs[0]

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)
        if not self.log.closed:
            self.log.close()


def load_run(run_dir: Path):
    return load_ticks(run_dir), load_requests(run_dir), load_episodes(run_dir)


def q43_to_state43(q43: np.ndarray) -> np.ndarray:
    """q43 rows (env joint order) -> state43 rows (VLA observation order), same dtype."""
    q43 = np.asarray(q43)
    out = np.empty(q43.shape[:-1] + (S43,), dtype=q43.dtype)
    for g, _ in STATE_GROUPS:
        out[..., STATE_SLICES[g]] = q43[..., Q43_GROUP_INDICES[g]]
    return out
