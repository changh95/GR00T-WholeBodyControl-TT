"""Concurrency probe against the running server. Modes:
  vla <obs.pkl> <period_s|0> <dur_s> <log>   : trimmed VLA get_action, back-to-back (0) or periodic
  wbc <endpoint> <rate_hz> <dur_s> <log>      : gear_wbc_action or ping at a fixed rate
Logs one JSON line per call with CLOCK_MONOTONIC ns (shared across processes)."""
import json, pickle, sys, time
import numpy as np
from gr00t.policy.server_client import PolicyClient

now = time.monotonic_ns
mode = sys.argv[1]
import os
c = PolicyClient(host="127.0.0.1", port=int(os.environ.get("PORT", "5555")), timeout_ms=15000)
if mode == "vla":
    obs = pickle.load(open(sys.argv[2], "rb")); period = float(sys.argv[3]); dur = float(sys.argv[4]); log = open(sys.argv[5], "w")
    mc = c.get_modality_config()
    keep = [f"video.{k}" for k in mc["video"].modality_keys] + [f"state.{k}" for k in mc["state"].modality_keys] + list(mc["language"].modality_keys)
    o = {k: obs[k] for k in keep}
    c.get_action(o)
    t_end = now() + int(dur * 1e9); nxt = now()
    while now() < t_end:
        if period > 0:
            s = (nxt - now()) / 1e9
            if s > 0: time.sleep(s)
            nxt += int(period * 1e9)
        t0 = now(); c.get_action(o); t1 = now()
        log.write(json.dumps({"t0": t0, "t1": t1}) + "\n")
else:
    ep = sys.argv[2]; rate = float(sys.argv[3]); dur = float(sys.argv[4]); log = open(sys.argv[5], "w")
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((1, 516)).astype(np.float32)
    call = (lambda: c.call_endpoint("ping", requires_input=False)) if ep == "ping" else \
           (lambda: c.call_endpoint("gear_wbc_action", {"policy": "GR00T-WholeBodyControl-Balance", "obs": obs}))
    call()
    dt = int(1e9 / rate); t_end = now() + int(dur * 1e9); nxt = now() + dt
    while now() < t_end:
        s = (nxt - now()) / 1e9
        if s > 0: time.sleep(s)
        t0 = now(); call(); t1 = now()
        log.write(json.dumps({"sched": nxt, "t0": t0, "t1": t1}) + "\n")
        nxt += dt
log.close()
