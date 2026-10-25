"""G1 apple-to-plate rollout with the MuJoCo viewer open, paced to real time.

Same pipeline as gr00t/eval/rollout_policy.py (one env, VLA + lower-body WBC both served by
run_gr00t_server.py), with the env's onscreen MuJoCo viewer turned on.
"""

import argparse
import time

from gr00t.eval import rollout_policy as rp
import gymnasium as gym
import mujoco
import numpy as np
from robosuite.renderers.viewer.mjviewer_renderer import MjviewerRenderer


# robosuite relaunches the viewer on every episode reset (it reloads the scene), so the free
# camera's angle is kept here and carried over to the next window. The first window starts
# where the robot's third-person camera is.
FREE_CAM = {}
START_CAMERA = "robot0_rs_tppview"


def camera_angles(model, data, camera, body):
    cam = model.camera(camera).id
    forward = -data.cam_xmat[cam].reshape(3, 3)[:, 2]
    return {
        "azimuth": float(np.degrees(np.arctan2(forward[1], forward[0]))),
        "elevation": float(np.degrees(np.arcsin(forward[2]))),
        # The tracking camera aims at the pelvis rather than the upper body, so back off a bit
        # to keep the head and the table in frame.
        "distance": 1.5 * float(np.linalg.norm(data.cam_xpos[cam] - data.xpos[body])),
    }


def patch_viewer_camera():
    """Free camera that tracks the pelvis, so the mouse can orbit / zoom while the robot walks."""
    launch, close = MjviewerRenderer.update, MjviewerRenderer.close

    def update(self):
        launching = self.viewer is None
        launch(self)
        if launching and self.camera_id is None:
            model, data = self.env.sim.model._model, self.env.sim.data._data
            pelvis = next(
                (i for i in range(model.nbody) if model.body(i).name.endswith("pelvis")), 1
            )
            if not FREE_CAM:
                FREE_CAM.update(camera_angles(model, data, START_CAMERA, pelvis))
            with self.viewer.lock():
                cam = self.viewer.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = pelvis
                cam.azimuth = FREE_CAM["azimuth"]
                cam.elevation = FREE_CAM["elevation"]
                cam.distance = FREE_CAM["distance"]

    def save_and_close(self):
        if self.viewer is not None and self.camera_id is None:
            cam = self.viewer.cam
            FREE_CAM.update(azimuth=cam.azimuth, elevation=cam.elevation, distance=cam.distance)
        close(self)

    MjviewerRenderer.update = update
    MjviewerRenderer.close = save_and_close


class RealtimeWrapper(gym.Wrapper):
    """Sleeps so sim time advances no faster than wall time."""

    def reset(self, **kwargs):
        self._anchor = None
        return self.env.reset(**kwargs)

    def step(self, action):
        out = self.env.step(action)
        sim_t = self.env.unwrapped.base_env.cur_time
        now = time.perf_counter()
        if self._anchor is None:
            self._anchor = (now, sim_t)
        else:
            wait = self._anchor[0] + (sim_t - self._anchor[1]) - now
            if wait > 0:
                time.sleep(wait)
            elif wait < -0.25:
                # Fell behind (e.g. waiting on a VLA chunk): carry on from here instead of
                # fast-forwarding to catch up.
                self._anchor = (now, sim_t)
        return out


def make_env_fn(camera: str):
    def get_env_fn(env_name):
        def env_fn():
            from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401
            from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
            from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
            import robocasa  # noqa: F401

            gym_env = gym.make(
                env_name,
                onscreen=True,
                offscreen=True,
                enable_waist=True,
                randomize_cameras=False,
                camera_names=["robot0_oak_egoview", "robot0_rs_tppview"],
                render_camera=[camera] if camera else None,
            )
            wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
            return RealtimeWrapper(WholeBodyControlWrapper(gym_env, wbc_config))

        return env_fn

    return get_env_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=1440)
    parser.add_argument(
        "--camera", default="", help="fixed camera name, e.g. robot0_rs_tppview ('' = free camera)"
    )
    parser.add_argument(
        "--env_name", default="gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
    )
    args = parser.parse_args()

    patch_viewer_camera()
    rp.get_groot_locomanip_env_fn = make_env_fn(args.camera)
    results = rp.run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        policy_client_host=args.host,
        policy_client_port=args.port,
        n_envs=1,
        n_action_steps=20,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
