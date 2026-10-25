"""T-1.3 official arm: one seed through the stock rollout stack, recording every env step.

Run by tests/test_official_equivalence.py in its own simpy subprocess (one per seed, so no class
patch, ORT session or GC history leaks between arms; PLAN.md C-14):

    python tests/official_arm.py --seed S --port P --out D/official_seedS.npz

The env, wrappers, vector env, client and rollout loop are the stock ones, imported from
``gr00t/eval/rollout_policy.py``: ``create_eval_env`` (no video; ``MultiStepWrapper(n_action_steps=20,
max_episode_steps=1440, terminate_on_success=True)``), ``SyncVectorEnv``, ``create_gr00t_sim_policy``
(a ``PolicyClient``) and ``run_rollout_gymnasium_policy`` itself. The WBC runs on the official
``G1GearWbcPolicy`` ORT sessions (``ort.InferenceSession(path)``: default SessionOptions). Two
things are added, neither of which changes a value the stack computes:

- ``StepRecorder``, a gym.Wrapper between MultiStepWrapper and the WholeBodyControlWrapper env
  (inserted by wrapping ``rollout_policy.get_gym_env``), records q, the per-step action, the WBC
  joint command and success for every env step of the first episode. Because the stock loop resets
  unseeded, it supplies ``seed=S`` to that first reset (the T-1.3 "seeded reset").
- ``PolicyTap`` passes every call through to the client unchanged and notes the tick of each
  request and the keys/dtypes of the first observation (the stock request has extra keys).

The VLA at ``--port`` is a replay mock serving the chunks the loop recorded, so a request whose
state differs by any bit (or whose image sha1 differs) is answered with an error; PolicyClient
raises on it, and the arm saves what it recorded up to there and exits 2.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist", "base_height_command",
          "navigate_command")  # A35 order (rtloop.schema.ACTION_GROUPS), duplicated: no rtloop import


def _recorder_class():
    import gymnasium as gym

    class StepRecorder(gym.Wrapper):
        """Pass-through wrapper that records each step of the first (seeded) episode."""

        def __init__(self, env, seed: int):
            super().__init__(env)
            self.first_seed = int(seed)
            self.n_resets = 0
            self.recording = False
            self.reset_kwargs = None
            self.q0 = None
            self.rows = {k: [] for k in ("q", "dq", "cmd35", "q_cmd", "nav_wbc", "base_pose",
                                         "base_vel", "success", "terminated", "truncated")}
            self.action_dtypes = None

        def reset(self, **kwargs):
            self.n_resets += 1
            if self.n_resets == 1 and kwargs.get("seed") is None:
                kwargs["seed"] = self.first_seed
            if self.n_resets == 1:
                self.reset_kwargs = {k: repr(v) for k, v in kwargs.items()}
            obs, info = self.env.reset(**kwargs)
            self.recording = self.n_resets == 1
            if self.recording:
                self.q0 = np.array(obs["q"], dtype=np.float64, copy=True)
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.recording:
                r = self.rows
                if self.action_dtypes is None:
                    self.action_dtypes = {k: [str(np.asarray(v).dtype), list(np.shape(v))]
                                          for k, v in action.items()}
                r["cmd35"].append(np.concatenate(
                    [np.asarray(action[f"action.{g}"], dtype=np.float32).reshape(-1) for g in GROUPS]))
                r["q"].append(np.array(obs["q"], dtype=np.float64, copy=True))
                r["dq"].append(np.array(obs["dq"], dtype=np.float64, copy=True))
                r["base_pose"].append(np.array(obs["floating_base_pose"], dtype=np.float64, copy=True))
                r["base_vel"].append(np.array(obs["floating_base_vel"], dtype=np.float64, copy=True))
                wbc = self.env.wbc_policy  # the WholeBodyControlWrapper's (re-created per reset)
                r["q_cmd"].append(np.array(wbc.last_action["q"], dtype=np.float64, copy=True))
                r["nav_wbc"].append(np.array(wbc.lower_body_policy.cmd, dtype=np.float64, copy=True))
                r["success"].append(bool(info.get("success", False)))
                r["terminated"].append(bool(terminated))
                r["truncated"].append(bool(truncated))
            return obs, reward, terminated, truncated, info

        @property
        def n_steps(self) -> int:
            return len(self.rows["q"])

    return StepRecorder


class PolicyTap:
    """Forwards get_action/reset to the stock client unchanged; notes request ticks and keys."""

    def __init__(self, inner, recorder_ref: list):
        self.inner = inner
        self.recorder_ref = recorder_ref
        self.request_ticks: list[int] = []
        self.first_obs = None
        self.n_reset = 0

    def get_action(self, observation, options=None):
        rec = self.recorder_ref[0] if self.recorder_ref else None
        self.request_ticks.append(rec.n_steps if rec is not None and rec.recording else -1)
        if self.first_obs is None:
            self.first_obs = {k: ([str(v.dtype), list(v.shape)] if isinstance(v, np.ndarray)
                                  else [type(v).__name__, repr(v)[:120]])
                              for k, v in observation.items()}
        if options is None:
            return self.inner.get_action(observation)
        return self.inner.get_action(observation, options)

    def reset(self, options=None):
        self.n_reset += 1
        return self.inner.reset() if options is None else self.inner.reset(options)


def wbc_session_report(env) -> dict:
    """Providers and thread options of the official WBC's ORT sessions (closure cells of
    run_inference, read only), so a mismatch can be traced to SessionOptions first."""
    out = {}
    lower = env.wbc_policy.lower_body_policy
    for name in ("policy_1", "policy_2"):
        fn = getattr(lower, name)
        sess = None
        for cell in fn.__closure__ or ():
            v = cell.cell_contents
            if type(v).__name__ == "InferenceSession":
                sess = v
        if sess is None:
            out[name] = None
            continue
        so = sess.get_session_options()
        out[name] = {"model": Path(sess._model_path).name if getattr(sess, "_model_path", None) else None,
                     "providers": sess.get_providers(),
                     "intra_op_num_threads": so.intra_op_num_threads,
                     "inter_op_num_threads": so.inter_op_num_threads,
                     "graph_optimization_level": str(so.graph_optimization_level),
                     "execution_mode": str(so.execution_mode)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True, help="replay mock (never 5555)")
    ap.add_argument("--out", required=True, help="output .npz (a .json with the same stem is written too)")
    ap.add_argument("--n-action-steps", type=int, default=20)
    ap.add_argument("--max-episode-steps", type=int, default=1440)
    ap.add_argument("--render-patch", action="store_true",
                    help="apply rtloop.sim_env.patch_render_context_teardown (diagnosis only)")
    a = ap.parse_args(argv)
    if a.port == 5555:
        ap.error("5555 belongs to the existing GR00T server")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or "GR00T_WBC_SERVER" in os.environ:
        raise SystemExit("run under simpy: CUDA hidden and GR00T_WBC_SERVER unset")

    t0 = time.monotonic()
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval import rollout_policy as rp

    render_patch = False
    if a.render_patch:
        from rtloop.sim_env import patch_render_context_teardown

        render_patch = patch_render_context_teardown()

    StepRecorder = _recorder_class()
    recorders: list = []
    stock_get_gym_env = rp.get_gym_env

    def get_gym_env(env_name, env_idx, total_n_envs):
        rec = StepRecorder(stock_get_gym_env(env_name, env_idx, total_n_envs), a.seed)
        recorders.append(rec)
        return rec

    rp.get_gym_env = get_gym_env  # create_eval_env looks it up at call time
    wrapper_configs = rp.WrapperConfigs(
        video=rp.VideoConfig(video_dir=None, max_episode_steps=a.max_episode_steps),
        multistep=rp.MultiStepConfig(n_action_steps=a.n_action_steps,
                                     max_episode_steps=a.max_episode_steps,
                                     terminate_on_success=True),
    )
    client = rp.create_gr00t_sim_policy("", EmbodimentTag.UNITREE_G1, a.host, a.port)
    tap = PolicyTap(client, recorders)
    t_import = time.monotonic() - t0

    full_gcs: list[list[int]] = []  # [env resets so far, recorded steps] at each gen-2 collection

    def on_gc(phase, info):
        if phase == "start" and info.get("generation") == 2:
            r = recorders[0] if recorders else None
            full_gcs.append([r.n_resets if r else 0, r.n_steps if r is not None and r.recording else -1])

    gc.callbacks.append(on_gc)
    error, result, sessions, ident = None, None, None, None
    gc0 = [s["collections"] for s in gc.get_stats()]
    t1 = time.monotonic()
    try:
        result = rp.run_rollout_gymnasium_policy(env_name=ENV_NAME, policy=tap,
                                                 wrapper_configs=wrapper_configs, n_episodes=1,
                                                 n_envs=1)
    except Exception as e:  # noqa: BLE001 - a replay mismatch arrives as RuntimeError
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    t_run = time.monotonic() - t1
    gc1 = [s["collections"] for s in gc.get_stats()]

    rec = recorders[0] if recorders else None
    if rec is not None:
        try:
            sessions = wbc_session_report(rec.env)
        except Exception as e:  # noqa: BLE001 - diagnostics only
            sessions = {"error": f"{type(e).__name__}: {e}"}
        from loop_tap import env_identity  # same function on both arms (tests/ is sys.path[0])

        ident = env_identity(rec.env)
    arrays = {}
    if rec is not None:
        for k, v in rec.rows.items():
            if v:
                arrays[k] = np.asarray(v)
        if rec.q0 is not None:
            arrays["q0"] = rec.q0
    arrays["request_ticks"] = np.asarray(tap.request_ticks, dtype=np.int64)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    meta = {
        "seed": a.seed, "port": a.port, "error": error,
        "n_steps": rec.n_steps if rec is not None else 0,
        "n_requests": len(tap.request_ticks), "n_policy_reset": tap.n_reset,
        "n_env_resets": rec.n_resets if rec is not None else 0,
        "first_reset_kwargs": rec.reset_kwargs if rec is not None else None,
        "rollout_success": (bool(result[1][0]) if result is not None and result[1] else None),
        "first_obs": tap.first_obs, "action_dtypes": rec.action_dtypes if rec is not None else None,
        "wbc_sessions": sessions, "env": ident, "gc_collections_during_rollout": [b - a_ for a_, b in zip(gc0, gc1)],
        "full_gc_at": full_gcs,
        "render_patch": render_patch, "t_import_s": t_import, "t_rollout_s": t_run,
        "n_action_steps": a.n_action_steps, "max_episode_steps": a.max_episode_steps,
        "python": sys.executable,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"official_arm: seed {a.seed}: {meta['n_steps']} steps, {meta['n_requests']} requests, "
          f"success {meta['rollout_success']}, error {error}", flush=True)
    return 0 if error is None else 2


if __name__ == "__main__":
    sys.exit(main())
