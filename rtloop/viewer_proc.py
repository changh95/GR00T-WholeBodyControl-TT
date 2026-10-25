"""The onscreen MuJoCo viewer of rt_sim, in its own process, fed by the loop through shared memory (PLAN.md §1.1, R9).

    python -m rtloop.viewer_proc --shm PREFIX --parent PID [--cpus 6,7] [--hz 60]

Spawned by rtloop.loop under --viewer (rtloop.viewer.ViewerLink), with CUDA hidden and MUJOCO_GL=glfw for a
window on $DISPLAY. It owns its own MjModel + MjData, loaded from the MJB the loop saves at every reset, and a
mujoco passive viewer with the free tracking camera of the day-1 prototype live_viewer.py (git history at
c1d5b90; the camera follows the pelvis and the mouse orbits and zooms; collision geoms hidden as in robosuite's
mjviewer). About 60 times a second it takes the latest consistent state snapshot (seqlock reader: it retries a
torn copy), runs mj_forward and syncs the window. A new model generation is loaded into the same window. Nothing
it does is ever waited for by the loop; it writes only its own counters (pid, frames, last seq, generation, loads)
into the header.

It exits when the window is closed, on SIGINT/SIGTERM, or when the loop dies (PR_SET_PDEATHSIG kills it with
SIGKILL, and every frame checks the parent pid). numpy and mujoco are imported after --cpus is applied, so all
of its threads (GLFW render thread, driver threads) inherit the aux CPUs.
"""

import argparse
import ctypes
import os
import signal
import sys
import threading
import time

FREE_CAM: dict = {}  # the free camera's angles, kept when a window has to be relaunched
START_CAMERA = "robot0_rs_tppview"  # the first window starts where the robot's third-person camera is
DEFAULT_CAM = {"azimuth": 180.0, "elevation": -20.0, "distance": 2.5}  # models without START_CAMERA


def die_with_parent(parent: int) -> None:
    """Have the kernel SIGKILL this process when the loop dies, however it dies (PR_SET_PDEATHSIG)."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, ctypes.c_ulong(signal.SIGKILL), 0, 0, 0)
    except (OSError, AttributeError):
        pass
    if parent and os.getppid() != parent:  # it died before the prctl
        sys.exit(0)


def attach(name: str):
    """Attach a segment of the loop. Before Python 3.13 attaching registers it with this process's resource
    tracker, which would unlink the loop's segment when the viewer exits (bpo-38119), so that is undone,
    except for a segment this process created itself (the in-process handoff test)."""
    from multiprocessing import resource_tracker, shared_memory

    from rtloop import viewer as vl

    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(name=name)
        if name not in vl.OWNED:
            resource_tracker.unregister(shm._name, "shared_memory")
        return shm


class StateReader:
    """Seqlock reader of the loop's scene state; also used in-process by the handoff test."""

    def __init__(self, prefix: str):
        import numpy as np

        from rtloop import viewer as vl

        self.vl = vl
        self.prefix = prefix
        self.shm = attach(f"{prefix}_s")
        self.hdr = np.ndarray((vl.HDR,), np.int64, self.shm.buf)
        self.raw = np.ndarray((vl.STATE_BYTES,), np.uint8, self.shm.buf)
        self.local = np.zeros(vl.STATE_BYTES, np.uint8)
        self.torn = 0  # reads that saw a write in progress and retried
        self._views: tuple | None = None
        self._views_of = None

    def read(self, tries: int = 1000) -> tuple[int, int, int] | None:
        """(seq, gen, k) of a consistent snapshot now in self.local; None before the first publish."""
        vl, h = self.vl, self.hdr
        for _ in range(tries):
            s1 = int(h[vl.SEQ])
            if s1 == 0:
                return None
            if s1 & 1:
                self.torn += 1
                continue
            gen, k = int(h[vl.GEN]), int(h[vl.K])
            n = min(max(int(h[vl.NBYTES]), 0), vl.STATE_BYTES - vl.PAYLOAD)
            self.local[vl.PAYLOAD:vl.PAYLOAD + n] = self.raw[vl.PAYLOAD:vl.PAYLOAD + n]
            if int(h[vl.SEQ]) == s1:
                return s1, gen, k
            self.torn += 1
        return None

    def apply(self, model, data) -> None:
        """Copy the snapshot in self.local into (model, data) of the snapshot's generation."""
        import numpy as np

        if self._views_of != (id(model), id(data)):
            t, views, _ = self.vl.scene_views(self.local, model)
            self._views = (t, [(getattr(data if s == "data" else model, name), v) for s, name, v in views])
            self._views_of = (id(model), id(data))
        t, pairs = self._views
        for dst, src in pairs:
            np.copyto(dst, src)
        data.time = float(t[0])

    def load_model(self, gen: int):
        """The MjModel of generation `gen`, or None if the loop has superseded it meanwhile."""
        import mujoco

        try:
            return mujoco.MjModel.from_binary_path(str(self.vl.MJB_DIR / self.vl.model_segment(self.prefix, gen)))
        except (FileNotFoundError, ValueError):  # unlinked before we opened it: a newer one is published
            return None

    def close(self) -> None:
        self.hdr = self.raw = None
        self._views = self._views_of = None
        self.shm.close()


def camera_angles(model, data, camera, body) -> dict:
    """The free-camera angles that look along `camera` at `body` (the day-1 prototype live_viewer.py, c1d5b90)."""
    import numpy as np

    cam = model.camera(camera).id
    forward = -data.cam_xmat[cam].reshape(3, 3)[:, 2]
    return {
        "azimuth": float(np.degrees(np.arctan2(forward[1], forward[0]))),
        "elevation": float(np.degrees(np.arcsin(forward[2]))),
        # The tracking camera aims at the pelvis rather than the upper body, so back off a bit
        # to keep the head and the table in frame.
        "distance": 1.5 * float(np.linalg.norm(data.cam_xpos[cam] - data.xpos[body])),
    }


def pelvis_id(model) -> int:
    return next((i for i in range(model.nbody) if model.body(i).name.endswith("pelvis")), 1 if model.nbody > 1 else 0)


def track_pelvis(handle, model, data, keep_angles: bool) -> None:
    """Free camera that tracks the pelvis, so the mouse can orbit / zoom while the robot walks."""
    import mujoco

    pelvis = pelvis_id(model)
    if not FREE_CAM:
        try:
            FREE_CAM.update(camera_angles(model, data, START_CAMERA, pelvis))
        except KeyError:
            FREE_CAM.update(DEFAULT_CAM)
    with handle.lock():
        cam = handle.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = pelvis
        if not keep_angles:
            cam.azimuth, cam.elevation, cam.distance = (FREE_CAM["azimuth"], FREE_CAM["elevation"],
                                                        FREE_CAM["distance"])
        handle.opt.geomgroup[0] = 0  # collision geoms off, as robosuite's MjviewerRenderer


def launch(model, data):
    import mujoco.viewer

    before = set(threading.enumerate())
    handle = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    handle.gui_threads = [t for t in threading.enumerate() if t not in before]  # joined by close_window
    track_pelvis(handle, model, data, keep_angles=False)
    return handle


def close_window(handle, timeout_s: float = 3.0) -> bool:
    """Close the window and wait for mujoco's GUI thread to finish tearing it down. That thread is a daemon:
    exiting while it is still in Simulate.destroy (and racing the atexit glfw.terminate) segfaulted (rc -11).
    False if it did not finish in time."""
    handle.close()
    for t in getattr(handle, "gui_threads", []):
        t.join(timeout_s)
    return not any(t.is_alive() for t in getattr(handle, "gui_threads", []))


def reload(handle, model, data):
    """Show a new model generation: in the same window if the simulate object takes it, else a new window."""
    sim = handle._get_sim()  # mujoco 3.2 has no public reload for a passive handle
    if sim is not None:
        try:
            sim.load(model, data, "")
            track_pelvis(handle, model, data, keep_angles=True)
            return handle
        except Exception as e:  # noqa: BLE001 - fall back to a fresh window
            print(f"viewer_proc: in-window reload failed ({type(e).__name__}: {e}); relaunching", flush=True)
    cam = handle.cam
    FREE_CAM.update(azimuth=cam.azimuth, elevation=cam.elevation, distance=cam.distance)
    close_window(handle)
    return launch(model, data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rtloop.viewer_proc", description=__doc__.split("\n")[0])
    ap.add_argument("--shm", required=True, help="segment prefix (rtloop.viewer.ViewerLink.prefix)")
    ap.add_argument("--parent", type=int, default=0, help="the loop's pid (exit when it is gone)")
    ap.add_argument("--cpus", default="", help="CPU affinity, e.g. 6,7 (PLAN.md §1.1 aux cores)")
    ap.add_argument("--hz", type=float, default=60.0)
    a = ap.parse_args(argv)
    if a.cpus:
        os.sched_setaffinity(0, {int(c) for c in a.cpus.split(",") if c.strip()})
    die_with_parent(a.parent)
    stop: list[int] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, _frame: stop.append(signum))

    import mujoco

    from rtloop import viewer as vl

    reader = StateReader(a.shm)
    h = reader.hdr
    h[vl.V_PID] = os.getpid()
    print(f"viewer_proc: pid {os.getpid()} cpus {sorted(os.sched_getaffinity(0))} shm {a.shm} "
          f"DISPLAY={os.environ.get('DISPLAY')} MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
    import glfw

    # Fail now, not at the first model: launch_passive's GUI thread would die on this and leave the
    # main thread waiting for a handle forever.
    if not glfw.init():
        print("viewer_proc: GLFW cannot open a window (no usable X display); exiting", flush=True)
        reader.close()
        return 2
    handle = model = data = None
    cur_gen = last_seq = 0
    period = 1.0 / a.hz
    t_next = time.monotonic()
    why = "signal"
    try:
        while not stop:
            if a.parent and os.getppid() != a.parent:
                why = "the loop is gone"
                break
            snap = reader.read()
            if snap is not None:
                seq, gen, _k = snap
                if gen != cur_gen:
                    m2 = reader.load_model(gen)
                    if m2 is not None:
                        d2 = mujoco.MjData(m2)
                        reader.apply(m2, d2)
                        mujoco.mj_forward(m2, d2)
                        t0 = time.monotonic()
                        handle = launch(m2, d2) if handle is None else reload(handle, m2, d2)
                        model, data, cur_gen, last_seq = m2, d2, gen, seq
                        h[vl.V_GEN] = gen
                        h[vl.V_LOADS] += 1
                        print(f"viewer_proc: gen {gen} shown ({model.nbody} bodies) in "
                              f"{(time.monotonic() - t0) * 1e3:.0f} ms", flush=True)
                elif seq != last_seq and handle is not None:
                    with handle.lock():
                        reader.apply(model, data)
                        mujoco.mj_forward(model, data)
                    last_seq = seq
            if handle is not None:
                if not handle.is_running():
                    why = "window closed"
                    break
                handle.sync()
                h[vl.V_FRAMES] += 1
                h[vl.V_SEQ] = last_seq
            t_next += period
            dt = t_next - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                t_next = time.monotonic()
    finally:
        print(f"viewer_proc: exit ({why}), {int(h[vl.V_FRAMES])} frames, {int(h[vl.V_LOADS])} models, "
              f"{reader.torn} torn reads retried", flush=True)
        clean = handle is None or close_window(handle)
        reader.close()
    if not clean:  # the GUI thread is stuck: skip the interpreter's teardown (atexit glfw.terminate) entirely
        print("viewer_proc: GUI thread did not finish; leaving without teardown", flush=True)
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
