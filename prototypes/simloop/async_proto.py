"""Wall-clock 50 Hz async loop prototype with a MOCK VLA thread (no server, no CUDA).

Control thread: every 20 ms tick -> pick action for current sim time from the chunk buffer ->
WholeBodyControlWrapper.step (WBC on onnxruntime CPU + robosuite physics).
VLA thread: takes the latest snapshot (on-demand ego render + state + language), serializes it with
the real MsgSerializer, sleeps `latency` (stand-in for the server), returns a 30-step hold chunk
stamped with the snapshot's sim time. Measures tick jitter / deadline misses / chunk staleness.
"""
import os
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("GR00T_WBC_SERVER", None)
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402

from gr00t.policy.server_client import MsgSerializer  # noqa: E402
from sim_timing import hold_action, make_env  # noqa: E402

DT = 0.02
HORIZON = 30
latency = float(sys.argv[1]) if len(sys.argv) > 1 else 0.045
duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
request_every = int(sys.argv[3]) if len(sys.argv) > 3 else 10  # ticks between requests (if idle)
STATE_KEYS = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]

env = make_env(offscreen=True)
obs, _ = env.reset()
rs = env.env.unwrapped.env.env
for k in list(rs._observables):  # keep image keys alive but sample them rarely
    if k.endswith("_image"):
        rs.modify_observable(k, "sampling_rate", 1.0)

lock = threading.Lock()
snap_ready = threading.Event()
state = {"snap": None, "chunk": None, "chunk_t0": None, "busy": False, "stop": False}
log = {"vla_lat": [], "stale_ticks": []}


def vla_thread():
    while not state["stop"]:
        if not snap_ready.wait(timeout=0.1):
            continue
        snap_ready.clear()
        with lock:
            snap = state["snap"]
        t_req = time.perf_counter()
        payload = MsgSerializer.to_bytes({"endpoint": "get_action", "data": {"observation": snap["obs"]}})
        time.sleep(latency)  # stand-in for the server round trip
        chunk = {f"action.{k}": np.repeat(snap["obs"][f"state.{k}"][0], HORIZON, 0) for k in
                 ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]}
        chunk["action.navigate_command"] = np.zeros((HORIZON, 3), np.float32)
        chunk["action.base_height_command"] = np.full((HORIZON, 1), 0.74, np.float32)
        chunk = MsgSerializer.from_bytes(MsgSerializer.to_bytes(chunk))  # response decode cost
        with lock:
            state["chunk"], state["chunk_t0"], state["busy"] = chunk, snap["t_obs"], False
        log["vla_lat"].append(time.perf_counter() - t_req)


th = threading.Thread(target=vla_thread, daemon=True)
th.start()

n_ticks = int(duration_s / DT)
starts, computes = [], []
held = hold_action(obs)
t_next = time.perf_counter()
last_req_tick = -10**9
dry = 0
for i in range(n_ticks):
    now = time.perf_counter()
    if now < t_next:
        time.sleep(t_next - now)
    t0 = time.perf_counter()
    starts.append(t0)
    t_sim = i * DT
    with lock:
        chunk, c0 = state["chunk"], state["chunk_t0"]
    if chunk is None:
        act = held
    else:
        k = int(round((t_sim - c0) / DT))  # index aligned to the snapshot time
        if k >= HORIZON:
            dry += 1
            k = HORIZON - 1
        act = {kk: v[k] for kk, v in chunk.items()}
        if i % 25 == 0:
            log["stale_ticks"].append(k)
    obs, *_ = env.step(act)
    with lock:
        busy = state["busy"]
    if not busy and i - last_req_tick >= request_every:
        img = rs.sim.render(camera_name="robot0_oak_egoview", width=640, height=480)[::-1]
        snap = {"video.ego_view": img[None, None].copy()}
        for k in STATE_KEYS:
            snap[f"state.{k}"] = np.asarray(obs[f"state.{k}"], np.float32)[None, None]
        snap["annotation.human.task_description"] = (obs["annotation.human.task_description"],)
        with lock:
            state["snap"], state["busy"] = {"obs": snap, "t_obs": t_sim}, True
        snap_ready.set()
        last_req_tick = i
    computes.append(time.perf_counter() - t0)
    t_next += DT
state["stop"] = True
th.join(1)
starts = np.array(starts)
period = np.diff(starts) * 1e3
comp = np.array(computes) * 1e3
print(f"latency={latency*1e3:.0f} ms, request_every={request_every} ticks, {n_ticks} ticks")
print(f"tick period ms: mean {period.mean():.2f} std {period.std():.2f} p99 {np.percentile(period,99):.2f} max {period.max():.2f}")
print(f"tick compute ms: mean {comp.mean():.2f} p99 {np.percentile(comp,99):.2f} max {comp.max():.2f}; deadline misses(>20ms) {int((comp>20).sum())}")
print(f"VLA mock round trip ms: mean {np.mean(log['vla_lat'])*1e3:.1f} n={len(log['vla_lat'])}; chunk index at sample ticks {log['stale_ticks'][:12]}; dry ticks {dry}")
print(f"wall {starts[-1]-starts[0]:.2f}s for sim {(n_ticks-1)*DT:.2f}s; pelvis z {float(rs.sim.data.qpos[2]):.3f}")
