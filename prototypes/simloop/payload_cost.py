"""Client-side cost of serializing a VLA request (no server call): full rollout obs vs minimal keys."""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("GR00T_WBC_SERVER", None)
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402

from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper  # noqa: E402
from gr00t.policy.server_client import MsgSerializer  # noqa: E402
from sim_timing import hold_action, make_env  # noqa: E402

env = make_env(offscreen=True)
env = MultiStepWrapper(env, video_delta_indices=np.array([0]), state_delta_indices=np.array([0]),
                       n_action_steps=20, max_episode_steps=1440, terminate_on_success=True)
obs, _ = env.reset()
# batch dim like SyncVectorEnv(n_envs=1)
full = {k: (np.asarray(v)[None] if not isinstance(v, str) else (v,)) for k, v in obs.items()}
need = ["video.ego_view"] + [f"state.{k}" for k in ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]]
mini = {k: full[k].astype(np.float32) if k.startswith("state.") else full[k] for k in need}
mini["annotation.human.task_description"] = full["annotation.human.task_description"]
for name, o in [("full rollout obs", full), ("minimal VLA obs", mini)]:
    req = {"endpoint": "get_action", "data": {"observation": o, "options": None}}
    for _ in range(3):
        MsgSerializer.to_bytes(req)
    n = 30
    t = time.perf_counter()
    for _ in range(n):
        b = MsgSerializer.to_bytes(req)
    dt = (time.perf_counter() - t) / n
    t = time.perf_counter()
    for _ in range(n):
        MsgSerializer.from_bytes(b)
    dt2 = (time.perf_counter() - t) / n
    print(f"{name}: keys={len(o)} bytes={len(b)/1e6:.2f} MB  to_bytes={dt*1e3:.2f} ms  from_bytes={dt2*1e3:.2f} ms")
print("full keys:", sorted(full.keys()))
# response size: 30-step chunk of 7 action groups
resp = ({f"action.{k}": np.zeros((1, 30, d), np.float32) for k, d in
         [("left_arm", 7), ("right_arm", 7), ("left_hand", 7), ("right_hand", 7), ("waist", 3), ("base_height_command", 1), ("navigate_command", 3)]}, {})
b = MsgSerializer.to_bytes(resp)
t = time.perf_counter()
for _ in range(200):
    MsgSerializer.from_bytes(b)
print(f"response chunk bytes={len(b)} from_bytes={(time.perf_counter()-t)/200*1e3:.3f} ms")
# MultiStepWrapper per-chunk overhead with hold action
act = {k: np.repeat(np.asarray(v).reshape(1, -1), 20, 0) for k, v in hold_action({kk: (np.asarray(vv)[-1] if not isinstance(vv, str) else vv) for kk, vv in obs.items()}).items()}
t = time.perf_counter()
for _ in range(5):
    obs, *_ = env.step(act)
print(f"MultiStepWrapper.step (20 ticks): {(time.perf_counter()-t)/5*1e3:.1f} ms")
