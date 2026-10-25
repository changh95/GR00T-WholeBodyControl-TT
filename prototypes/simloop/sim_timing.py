"""Per-tick cost breakdown of the G1 gear_wbc sim (CPU only, WBC on onnxruntime, no VLA).

Run with the client venv, MUJOCO_GL=egl, GR00T_WBC_SERVER unset. No CUDA is touched here.
"""

import collections
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("GR00T_WBC_SERVER", None)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

T = collections.defaultdict(list)


def timed(obj, name, key):
    fn = getattr(obj, name)

    def wrapper(*a, **k):
        t = time.perf_counter()
        out = fn(*a, **k)
        T[key].append(time.perf_counter() - t)
        return out

    setattr(obj, name, wrapper)


def make_env(offscreen=True):
    from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401
    from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
    from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
    import robocasa  # noqa: F401

    env = gym.make(
        "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
        onscreen=False,
        offscreen=offscreen,
        enable_waist=True,
        randomize_cameras=False,
        camera_names=["robot0_oak_egoview", "robot0_rs_tppview"],
    )
    wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
    return WholeBodyControlWrapper(env, wbc_config)


def hold_action(obs):
    a = {
        "action.navigate_command": np.zeros(3, np.float32),
        "action.base_height_command": np.array([0.74], np.float32),
    }
    for k in ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]:
        a[f"action.{k}"] = np.asarray(obs[f"state.{k}"], np.float32).copy()
    return a


def summarize(prefix, n_skip=10):
    out = {}
    for k, v in T.items():
        v = np.array(v[n_skip:]) * 1e3
        if len(v) == 0:
            continue
        out[k] = dict(n=len(v), mean_ms=round(float(v.mean()), 3), p50=round(float(np.median(v)), 3),
                      p99=round(float(np.percentile(v, 99)), 3), max=round(float(v.max()), 3))
    print(f"==== {prefix}")
    for k in sorted(out):
        print(f"  {k:40s} {out[k]}")
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    results = {"mode": mode}

    t = time.perf_counter()
    env = make_env(offscreen=(mode != "nocam"))
    results["make_env_s"] = time.perf_counter() - t

    sync_env = env.env.unwrapped  # SyncEnv subclass
    loco = sync_env.env  # Gr00tLocomanipRoboCasaEnv
    rs = loco.env  # robosuite env
    sim = rs.sim
    results["model_timestep"] = float(sim.model.opt.timestep)
    results["control_timestep"] = float(rs.control_timestep)
    results["substeps_per_env_step"] = int(rs.control_timestep / rs.model_timestep)
    results["camera_names"] = list(rs.camera_names)
    results["camera_hw"] = [list(x) for x in zip(rs.camera_heights, rs.camera_widths)]
    results["use_camera_obs"] = bool(rs.use_camera_obs)
    cam_obs = {k: (float(o._sampling_timestep), o.is_enabled()) for k, o in rs._observables.items() if k.endswith("_image")}
    results["camera_observables(sampling_dt,enabled)"] = cam_obs
    results["n_observables"] = len(rs._observables)
    results["nq_nv_nu"] = [int(sim.model.nq), int(sim.model.nv), int(sim.model.nu)]
    results["ngeom_nbody"] = [int(sim.model.ngeom), int(sim.model.nbody)]
    results["action_space"] = {k: list(v.shape) for k, v in env.action_space.items()}

    # instrumentation
    import mujoco

    orig_render = type(sim).render

    def render_wrapper(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_render(self, *a, **k)
        T["mjsim.render(camera)"].append(time.perf_counter() - t0)
        return out

    type(sim).render = render_wrapper
    orig_step = type(sim).step

    def step_wrapper(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_step(self, *a, **k)
        T["mjsim.step(physics substep)"].append(time.perf_counter() - t0)
        return out

    type(sim).step = step_wrapper
    orig_fwd = type(sim).forward

    def fwd_wrapper(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_fwd(self, *a, **k)
        T["mjsim.forward(substep)"].append(time.perf_counter() - t0)
        return out

    type(sim).forward = fwd_wrapper
    for nm in ["step1", "step2"]:
        if hasattr(type(sim), nm):
            orig = getattr(type(sim), nm)

            def mk(orig, nm):
                def w(self, *a, **k):
                    t0 = time.perf_counter()
                    out = orig(self, *a, **k)
                    T[f"mjsim.{nm}(physics substep)"].append(time.perf_counter() - t0)
                    return out
                return w

            setattr(type(sim), nm, mk(orig, nm))
    timed(rs, "_update_observables", "robosuite._update_observables(per substep)")

    timed(rs, "step", "robosuite.env.step(total, 10 substeps+obs)")
    timed(rs, "_pre_action", "robosuite._pre_action(PD/controller)")
    timed(rs, "_post_action", "robosuite._post_action(reward/success)")
    timed(rs, "_get_observations", "robosuite._get_observations")
    timed(loco, "get_basic_observation", "Gr00tLocomanip.get_basic_observation(img flip/copy)")
    timed(loco, "get_gr00t_observation", "Gr00tLocomanip.get_gr00t_observation")
    timed(sync_env, "observe", "SyncEnv.observe(joint reorder+FK eef)")
    timed(sync_env.safety_monitor, "handle_violations", "G1SyncEnv.safety_monitor")
    timed(sync_env, "step", "SyncEnv.step(total)")

    sim_id_before = id(rs.sim)
    t = time.perf_counter()
    obs, info = env.reset()
    results["reset1_s"] = time.perf_counter() - t
    results["mjsim_recreated_on_reset"] = id(rs.sim) != sim_id_before

    def instrument_wbc():
        wp = env.wbc_policy
        timed(wp, "get_action", "WBC.decoupled.get_action(total)")
        timed(wp, "set_observation", "WBC.set_observation(obs+history)")
        timed(wp, "set_goal", "WBC.set_goal")
        lb = wp.lower_body_policy
        timed(lb, "get_action", "WBC.lower.get_action(MLP+scale)")
        # wrap the ONNX callables
        for attr, key in [("policy_1", "WBC.MLP Balance (ort)"), ("policy_2", "WBC.MLP Walk (ort)")]:
            timed(lb, attr, key)
        timed(wp.robot_model, "cache_forward_kinematics", "pinocchio.cache_forward_kinematics")

    instrument_wbc()

    results["obs_keys"] = {k: (list(np.shape(v)), str(np.asarray(v).dtype)) for k, v in obs.items() if not isinstance(v, str)}
    results["language"] = obs.get("annotation.human.task_description")

    if mode == "nocam_runtime":
        # Disable camera observables at runtime, render on demand only.
        # Disabling the camera observables breaks robocasa_env.get_basic_observation (KeyError on
        # the image key), so instead sample them at 2.5 Hz (once per 20-tick VLA chunk).
        for k in list(rs._observables):
            if k.endswith("_image"):
                rs.modify_observable(k, "sampling_rate", 2.5)

    # check obs staleness: observable joint pos vs sim qpos at the end of the step
    stale = []
    t_loop = time.perf_counter()
    walk_ticks = 0
    for i in range(nsteps):
        a = hold_action(obs)
        t0 = time.perf_counter()
        obs, r, term, trunc, info = env.step(a)
        T["WholeBodyControlWrapper.step(total tick)"].append(time.perf_counter() - t0)
        if mode == "nocam_runtime" and i % 20 == 0:
            t1 = time.perf_counter()
            for cam in ["robot0_oak_egoview", "robot0_rs_tppview"]:
                img = rs.sim.render(camera_name=cam, width=640, height=480)
            T["on-demand render 2 cams 640x480"].append(time.perf_counter() - t1)
        robot = rs.robots[0]
        sim = rs.sim  # robosuite hard_reset re-creates MjSim on every reset
        qpos_now = sim.data.qpos[robot._ref_joint_pos_indexes]
        stale.append(float(np.abs(rs._observables["robot0_joint_pos"].obs - qpos_now).max()))
        if np.linalg.norm(env.wbc_policy.lower_body_policy.cmd) >= 0.05:
            walk_ticks += 1
    results["loop_wall_s"] = time.perf_counter() - t_loop
    results["sim_time_s"] = nsteps * rs.control_timestep
    results["realtime_factor"] = results["sim_time_s"] / results["loop_wall_s"]
    results["max|obs.joint_pos - qpos_end_of_step| (rad)"] = [float(np.median(stale)), float(np.max(stale))]
    results["walk_policy_ticks"] = walk_ticks
    results["pelvis_height_end"] = float(sim.data.qpos[2])

    t = time.perf_counter()
    obs, info = env.reset()
    results["reset2_s"] = time.perf_counter() - t

    results["timing"] = summarize(mode)
    for k, v in results.items():
        if k != "timing":
            print(f"{k}: {v}")
    with open(os.path.join(os.path.dirname(__file__), f"sim_timing_{mode}.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main()
