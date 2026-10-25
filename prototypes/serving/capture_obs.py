"""Reset the G1 apple-to-plate env once (CPU WBC, EGL render) and save the batched VLA observation."""
import os, pickle, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("GR00T_WBC_SERVER", None)
from functools import partial
import gymnasium as gym
from gr00t.eval import rollout_policy as rp

env_name = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
cfg = rp.WrapperConfigs(multistep=rp.MultiStepConfig(n_action_steps=20, max_episode_steps=1440))
t0 = time.time()
env = gym.vector.SyncVectorEnv([partial(rp.create_eval_env, env_idx=0, env_name=env_name, total_n_envs=1, wrapper_configs=cfg)])
obs, _ = env.reset(seed=0)
print("env up+reset in %.1fs" % (time.time() - t0))
for k, v in obs.items():
    print(k, type(v).__name__, getattr(v, "shape", None), getattr(v, "dtype", None), (v if not hasattr(v, "shape") else ""))
pickle.dump(obs, open(sys.argv[1], "wb"))
# also time one env.step of a zero-ish 20-step chunk is skipped (sim timing is another task)
env.close()
