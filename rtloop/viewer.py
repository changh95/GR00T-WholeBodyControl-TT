"""Loop side of the out-of-process MuJoCo viewer (PLAN.md §1.1, §1.2, R9, R10).

The onscreen viewer used to run inside env.step (robosuite's mjviewer; deviations (f) onscreen=True and (g)
relaunch in the reset window). In T-1.4 (run 20261024_223015_t14) its viewer.sync cost 6.8 ms p50 per tick
and the sim step went from 5.6 to 12.0 ms, while a VL A/B showed it changes neither the physics nor the VLA
input. A robot's visualisation is never on its control path, so the viewer is now its own process
(rtloop.viewer_proc, on the aux CPUs it shares with the HUD) and the env is built as stock (onscreen=False):

- per tick, ``ViewerLink.publish`` copies the scene state (time, qpos, qvel, act, mocap pose, geom and site
  rgba) into a POSIX shared-memory block under a seqlock. The tick thread is the only writer and never
  waits; the reader retries a torn copy. The publish time is the tick's ``viewer_us``.
- robosuite builds a new MjModel at every hard reset, so ``handoff`` (reset window) saves it (mj_saveModel,
  MJB, about 140 MB with the textures) into a fresh shared-memory segment and bumps a generation counter;
  the viewer loads the model of a new generation and drops the old one.

Nothing else in this env's model changes during an episode: every MjModel array was compared before and after
300 ticks (hold, then walk; seeds 0 and 1) and none changed, and the mocap bodies (eef targets) did not move.
geom_rgba and site_rgba, the arrays robosuite's visualize() writes, are copied every tick anyway (2.5 KB).

The seqlock relies on x86-64 ordering (stores are seen in program order, loads are not reordered with other
loads); numpy's separate C calls keep the compiler from merging them. A weakly ordered CPU needs fences.
The resource tracker of this process unlinks the segments if the loop dies without close().
"""

import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import numpy as np

from rtloop import affinity

RT = Path(__file__).resolve().parents[1]

# Header: int64 words. 0-7 are written by the loop only, 8-15 by the viewer only (its own cache line).
HDR = 16
SEQ, GEN, K, T_NS, NBYTES, LOOP_PID = range(6)
V_PID, V_FRAMES, V_SEQ, V_GEN, V_LOADS = range(8, 13)
PAYLOAD = HDR * 8  # byte offset of the scene state
STATE_BYTES = 1 << 20  # this env's state is 3.5 KB
POLL_EVERY = 50  # publishes between checks that the viewer process is still alive (1 s)
MJB_DIR = Path("/dev/shm")  # where POSIX shared memory lives on Linux (the viewer loads the MJB by path)
OWNED: set[str] = set()  # segments this process created (viewer_proc.attach leaves their tracking alone)
PER_TICK = ("time", "qpos", "qvel", "act", "mocap_pos", "mocap_quat", "geom_rgba", "site_rgba")


def scene_fields(m) -> tuple:
    """(source, name, shape, dtype) of what is copied every tick, in payload order (after the time)."""
    f8, f4 = np.float64, np.float32
    return (("data", "qpos", (m.nq,), f8), ("data", "qvel", (m.nv,), f8), ("data", "act", (m.na,), f8),
            ("data", "mocap_pos", (m.nmocap, 3), f8), ("data", "mocap_quat", (m.nmocap, 4), f8),
            ("model", "geom_rgba", (m.ngeom, 4), f4), ("model", "site_rgba", (m.nsite, 4), f4))


def scene_views(buf, m) -> tuple[np.ndarray, list, int]:
    """Views of the payload in `buf` laid out for model `m`: (time (1,) f8, [(source, name, view)], bytes)."""
    off = PAYLOAD
    t = np.ndarray((1,), np.float64, buf, off)
    off += 8
    views = []
    for src, name, shape, dt in scene_fields(m):
        views.append((src, name, np.ndarray(shape, dt, buf, off)))
        off += -(-math.prod(shape) * np.dtype(dt).itemsize // 8) * 8
    if off > len(buf):
        raise ValueError(f"scene state of {off - PAYLOAD} B does not fit in {len(buf) - PAYLOAD} B")
    return t, views, off - PAYLOAD


def model_segment(prefix: str, gen: int) -> str:
    return f"{prefix}_m{gen}"


class ViewerLink:
    """The shared memory and the viewer process of one loop run (--viewer); see the module docstring.

    ``__init__`` only creates the state block (its name goes into config.json), ``spawn`` starts the
    process once the run dir exists, ``handoff`` runs in every reset window, ``publish`` in every tick,
    ``close`` at loop exit. Nothing here ever waits for the viewer, and a dead viewer is logged once."""

    def __init__(self, cpus: list[int] | None = None):
        from multiprocessing import shared_memory

        self._shared_memory = shared_memory
        self.prefix = f"rtv{os.getpid()}_{secrets.token_hex(3)}"
        self.state = shared_memory.SharedMemory(name=f"{self.prefix}_s", create=True, size=STATE_BYTES)
        OWNED.add(self.state.name)
        self.hdr = np.ndarray((HDR,), np.int64, self.state.buf)
        self.hdr[:] = 0
        self.hdr[LOOP_PID] = os.getpid()
        self.cpus = list(cpus or [])
        self.seq = 0  # even: consistent; odd: a write is in progress
        self.gen = 0  # model generation (one per handoff)
        self.n = 0  # publishes
        self._model_seg = None
        self._data = None
        self._t = None
        self._copies: list = []
        self._nbytes = 0
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self._log = None
        self.exited: dict | None = None  # {rc, k, published} once the viewer process is seen dead
        self._closed = False

    def plan(self) -> dict:
        """The config.json record before the process exists."""
        return {"process": "rtloop.viewer_proc", "cpus": self.cpus or None, "shm": self.prefix,
                "state_bytes": STATE_BYTES, "env_onscreen": False,
                "per_tick": list(PER_TICK),
                "model_handoff": f"MJB per reset in {MJB_DIR}/{self.prefix}_m<gen>"}

    def spawn(self, log_path: Path) -> dict:
        """Start rtloop.viewer_proc (same interpreter, CUDA hidden, GLFW on $DISPLAY); returns the config record."""
        env = dict(os.environ)
        env.update(MUJOCO_GL="glfw", CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        for key in ("MUJOCO_EGL_DEVICE_ID", "PYOPENGL_PLATFORM", "GR00T_WBC_SERVER"):
            env.pop(key, None)
        path = env.get("PYTHONPATH", "")
        if str(RT) not in path.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join(p for p in (str(RT), path) if p)
        cmd = [sys.executable, "-m", "rtloop.viewer_proc", "--shm", self.prefix, "--parent", str(os.getpid())]
        if self.cpus:
            cmd += ["--cpus", ",".join(map(str, self.cpus))]
        self.log_path = Path(log_path)
        self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=self._log,
                                     stderr=subprocess.STDOUT, close_fds=True)
        return {**self.plan(), "pid": self.proc.pid, "log": str(self.log_path),
                "display": os.environ.get("DISPLAY")}

    def handoff(self, model, data) -> dict:
        """Reset window: publish the env's new MjModel as the next generation and the reset state with it."""
        import mujoco

        t0 = time.perf_counter_ns()
        self.check(-1)
        gen = self.gen + 1
        size = mujoco.mj_sizeModel(model)
        seg = self._shared_memory.SharedMemory(name=model_segment(self.prefix, gen), create=True, size=size)
        OWNED.add(seg.name)
        # Created (and tracked) as shared memory, written through its file: writing 140 MB through the
        # mapping page-faults it in this process (about 2x slower). The name lives on until unlink, and
        # this process keeps no mapping of it (not in its RSS).
        seg.close()
        mujoco.mj_saveModel(model, str(MJB_DIR / seg.name), None)
        self._t, views, self._nbytes = scene_views(self.state.buf, model)
        self._copies = [(v, getattr(data if src == "data" else model, name)) for src, name, v in views]
        self._data = data
        self.gen = gen
        old, self._model_seg = self._model_seg, seg
        self.publish(-1)
        if old is not None:
            old.unlink()  # a viewer still reading it keeps its open file; a late one skips to the new gen
            OWNED.discard(old.name)
        return {"gen": gen, "mjb_mb": round(size / 2**20, 1),
                "ms": round((time.perf_counter_ns() - t0) / 1e6, 1)}

    def publish(self, k: int) -> float:
        """Tick path: copy the scene state under the seqlock; returns its duration in µs (viewer_us)."""
        t0 = time.perf_counter_ns()
        h = self.hdr
        self.seq += 1
        h[SEQ] = self.seq
        self._t[0] = self._data.time
        for dst, src in self._copies:
            np.copyto(dst, src)
        h[GEN] = self.gen
        h[K] = k
        h[NBYTES] = self._nbytes
        h[T_NS] = t0
        self.seq += 1
        h[SEQ] = self.seq
        self.n += 1
        if self.n % POLL_EVERY == 0 and self.exited is None:
            self.check(k)
        return (time.perf_counter_ns() - t0) / 1e3

    def check(self, k: int) -> bool:
        """True while the viewer process runs; the first time it is found dead, say so once (the loop goes on)."""
        if self.proc is None or self.exited is not None:
            return False
        rc = self.proc.poll()
        if rc is None:
            return True
        self.exited = {"rc": rc, "k": k, "published": self.n, "gen": self.gen}
        print(f"rt_sim: viewer process {self.proc.pid} exited (rc {rc}) at k={k}, gen {self.gen}; "
              f"the loop runs on without it (log {self.log_path})", file=sys.stderr, flush=True)
        return False

    def stats(self) -> dict:
        h = self.hdr
        return {"published": self.n, "gen": self.gen, "viewer_pid": int(h[V_PID]), "frames": int(h[V_FRAMES]),
                "models_loaded": int(h[V_LOADS]), "viewer_gen": int(h[V_GEN]), "exited": self.exited}

    def close(self, grace_s: float = 3.0) -> dict:
        """Stop the viewer process (SIGTERM, SIGKILL after grace_s) and remove the shared memory; returns stats."""
        if self._closed:
            return {}
        self._closed = True
        out = self.stats()
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    out["threads"] = affinity.thread_report(self.proc.pid)
                except OSError:
                    pass
                self.proc.terminate()
                try:
                    self.proc.wait(grace_s)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(5)
                out["terminated"] = True
            out["rc"] = self.proc.returncode
            self._log.close()
        self._copies, self._t, self._data, self.hdr = [], None, None, None  # release the buffer exports
        self.state.close()
        for seg in (self.state, self._model_seg):
            if seg is not None:
                try:
                    seg.unlink()
                except FileNotFoundError:
                    pass
                OWNED.discard(seg.name)
        return out
