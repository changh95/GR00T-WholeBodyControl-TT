"""The G1 apple-to-plate env exactly as the stock rollout builds it, plus the handles rt_sim reads.

``make_env`` mirrors ``R/gr00t/eval/rollout_policy.py:get_groot_locomanip_env_fn`` (same gym id
and kwargs, then ``WholeBodyControlWrapper(BaseConfig(wbc_version="gear_wbc", enable_waist=True))``)
without importing gr00t (torch + transformers, about 1.6 s) and without SyncVectorEnv /
MultiStepWrapper: the loop steps the per-tick env itself. The loop always builds it with
``onscreen=False``, as stock, also under ``--viewer``: the onscreen viewer is its own process
(rtloop.viewer_proc), so PLAN.md §1.1 deviation (f) no longer applies.
"""

import math
import os

import numpy as np

from rtloop.schema import FALL_PELVIS_Z, Q43_GROUP_INDICES

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
CAMERA_NAMES = ["robot0_oak_egoview", "robot0_rs_tppview"]
ARM_Q43 = np.r_[Q43_GROUP_INDICES["left_arm"], Q43_GROUP_INDICES["right_arm"]]
LOWER_BODY = slice(0, 15)


def assert_cuda_free(where: str) -> None:
    """U5 / R2: the sim process must never see a GPU through CUDA (it corrupts the EGL renders)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd != "":
        raise RuntimeError(f"{where}: CUDA_VISIBLE_DEVICES={cvd!r}, must be '' in rt_sim (use simpy)")
    if "GR00T_WBC_SERVER" in os.environ:
        raise RuntimeError(f"{where}: GR00T_WBC_SERVER is set; rt_sim routes the WBC itself (D5)")
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError(f"{where}: torch initialised CUDA in rt_sim")


def patch_render_context_teardown() -> bool:
    """Make robosuite's offscreen render context safe to garbage-collect at any time (EGL only).

    robosuite hard-resets on every env.reset(): a new MjSim and a new EGL render context are made
    current, and the old pair lives on in a sim <-> context reference cycle until the cyclic GC runs.
    MjRenderContext.__del__ then frees the old MjrContext's GL objects *in whatever context is
    current* (the new one: same object names) and EGLGLContext.free() calls eglReleaseThread(), which
    unbinds the new context. Every later mjr_readPixels silently leaves its np.empty buffer as it
    was, so the "ego image" is stale memory for the rest of the episode while the physics is fine
    (found here: gc.collect() after a reset changed every later frame). The replacement frees the
    old context while it is current and then restores the context that was current before.
    Returns True if the patch is (now) installed. Like the original it never raises: at interpreter
    exit robosuite's atexit hook has already called eglTerminate, so a context collected then has
    nothing left to free and every EGL call fails with EGL_NOT_INITIALIZED.
    """
    if os.environ.get("MUJOCO_GL") != "egl":
        return False
    from robosuite.utils import binding_utils as bu

    cls = bu.MjRenderContext
    if getattr(cls.__del__, "rt_safe", False):
        return True
    from OpenGL import EGL
    from robosuite.renderers.context import egl_context

    def __del__(self):
        con, gl = getattr(self, "con", None), getattr(self, "gl_ctx", None)
        mine = getattr(gl, "_context", None)
        try:
            prev = EGL.eglGetCurrentContext()
        except Exception:  # noqa: BLE001 - EGL already torn down
            prev = None
        was_mine = bool(prev) and bool(mine) and prev.address == mine.address
        try:
            if mine:
                gl.make_current()
            if con is not None:
                con.free()
        except Exception:  # noqa: BLE001 - EGL terminated (interpreter exit): nothing left to free
            pass
        finally:
            if gl is not None:
                try:
                    gl.free()  # it is current, so free() unbinds it before destroying it
                except Exception:  # noqa: BLE001 - as the original: never raise from __del__
                    pass
            if prev and not was_mine:
                try:
                    EGL.eglMakeCurrent(egl_context.EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE,
                                       prev)
                except Exception:  # noqa: BLE001
                    pass
            self.con = None

    __del__.rt_safe = True
    cls.__del__ = __del__
    return True


def make_env(onscreen: bool = False, env_name: str = ENV_NAME):
    """gym.make(...) + WholeBodyControlWrapper, as get_groot_locomanip_env_fn's env_fn does."""
    import gymnasium as gym
    from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401 (registers the env ids)
    from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
    from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
    import robocasa  # noqa: F401

    gym_env = gym.make(
        env_name,
        onscreen=onscreen,
        offscreen=True,
        enable_waist=True,
        randomize_cameras=False,
        camera_names=list(CAMERA_NAMES),
    )
    wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
    return WholeBodyControlWrapper(gym_env, wbc_config)


def skip_first_step_check(env) -> int:
    """Mark gymnasium's PassiveEnvChecker step check as done; returns how many were marked.

    gym.make wraps the env in a checker whose first step() rebuilds the observation space and
    validates it (about 20 ms, warnings only, the step itself is unchanged). That one-off cost would
    otherwise land in tick 0 of the first episode.
    """
    from gymnasium.wrappers.env_checker import PassiveEnvChecker

    n, e = 0, env
    while e is not None:
        if isinstance(e, PassiveEnvChecker):
            e.checked_step = True
            n += 1
        e = getattr(e, "env", None)
    return n


class SimHandles:
    """The objects behind the wrappers that the loop reads every tick, resolved once per env.

    ``floating_base_pose`` is the pelvis free joint only if it is the model's first joint
    (robocasa_env.get_gr00t_observation checks the same thing); that is asserted here, so pelvis z
    is read from the observation instead of guessing a qpos index.
    """

    def __init__(self, env):
        import mujoco

        self.env = env
        self.sync_env = env.unwrapped           # G1SyncEnv subclass
        self.robosuite = self.sync_env.base_env  # robosuite RobotEnv (has .sim, .viewer)
        model = self.robosuite.sim.model
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("the first MuJoCo joint is not the pelvis free joint")
        rm = env.robot_model
        for g, idx in Q43_GROUP_INDICES.items():
            got = list(rm.get_joint_group_indices(g))
            if got != idx:
                raise RuntimeError(f"robot_model {g} indices {got} != schema {idx}")
        if list(rm.get_joint_group_indices("lower_body")) != list(range(15)):
            raise RuntimeError("lower_body is not q43[0:15]")

    @property
    def wbc(self):
        """G1DecoupledWholeBodyPolicy (re-created by every reset)."""
        return self.env.wbc_policy

    @property
    def lower(self):
        """G1GearWbcPolicy (``.cmd`` is the nav command it used this tick)."""
        return self.env.wbc_policy.lower_body_policy

    @property
    def safety_monitor(self):
        return self.sync_env.safety_monitor


def pelvis_state(obs: dict) -> tuple[float, float, float]:
    """(z, roll, pitch) of the pelvis from floating_base_pose [x, y, z, qw, qx, qy, qz]."""
    p = obs["floating_base_pose"]
    w, x, y, z = float(p[3]), float(p[4]), float(p[5]), float(p[6])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return float(p[2]), roll, pitch


def fell(pelvis_z: float) -> bool:
    return pelvis_z < FALL_PELVIS_Z
