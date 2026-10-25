"""Check when robosuite samples observables inside a 10-substep env step."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("GR00T_WBC_SERVER", None)
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402

from sim_timing import hold_action, make_env  # noqa: E402

env = make_env(offscreen=False)
obs, _ = env.reset()
rs = env.env.unwrapped.env.env
sim = rs.sim
robot = rs.robots[0]
o = rs._observables["robot0_joint_pos"]
print("sampling_dt", o._sampling_timestep, "delay", o._current_delay, "robot_joints", len(robot.robot_joints))
print("joint names", robot.robot_joints[:6])
# record qpos at every substep
orig_step2 = type(sim).step2 if hasattr(type(sim), "step2") else None
hist = []


def step2(self, *a, **k):
    out = orig_step2(self, *a, **k)
    hist.append(np.array(sim.data.qpos[robot._ref_joint_pos_indexes]))
    return out


type(sim).step2 = step2
for i in range(60):
    hist.clear()
    obs, *_ = env.step(hold_action(obs))
    ob = np.array(o.obs)
    d = [float(np.abs(ob - h).max()) for h in hist]
    if i % 10 == 0 or i > 55:
        print(i, "nsub", len(hist), "argmin substep", int(np.argmin(d)), "min", round(min(d), 6), "end", round(d[-1], 5),
              "pelvis z", round(float(sim.data.qpos[2]), 3))
# raw observable vs SyncEnv q
print("obs time", obs.get("time", None), "sim time", sim.data.time, "cur_time", rs.cur_time)
