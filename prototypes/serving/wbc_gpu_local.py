"""WBC CUDA graph in its OWN process/CUDA context (server venv), 50 Hz paced; logs latency.
argv: <rate_hz> <dur_s> <log> [priority]  (priority only affects streams inside this context)"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/deepgadget/experiments/gr00t/Isaac-GR00T-n1.6")
from gr00t.policy.gear_wbc_policy import GearWbcPolicies
now = time.monotonic_ns
rate, dur, logp = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
pol = GearWbcPolicies("/home/deepgadget/experiments/gr00t/Isaac-GR00T-n1.6/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy", "cuda")
obs = np.random.default_rng(1).standard_normal((1, 516)).astype(np.float32)
for _ in range(50): pol.get_action("GR00T-WholeBodyControl-Balance", obs)
log = open(logp, "w"); dt = int(1e9 / rate); t_end = now() + int(dur * 1e9); nxt = now() + dt
while now() < t_end:
    s = (nxt - now()) / 1e9
    if s > 0: time.sleep(s)
    t0 = now(); pol.get_action("GR00T-WholeBodyControl-Balance", obs); t1 = now()
    log.write(json.dumps({"sched": nxt, "t0": t0, "t1": t1}) + "\n"); nxt += dt
log.close()
