"""VLA get_action latency on the running server with a real reset observation (full vs trimmed obs)."""
import json, pickle, sys, time
import numpy as np
from gr00t.policy.server_client import PolicyClient, MsgSerializer

obs = pickle.load(open(sys.argv[1], "rb"))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
c = PolicyClient(host="127.0.0.1", port=5555, timeout_ms=15000)
mc = c.get_modality_config()
for k, v in mc.items():
    print("modality", k, v.modality_keys, "delta", list(v.delta_indices)[:3], "...len", len(v.delta_indices))
keep = [f"video.{k}" for k in mc["video"].modality_keys] + [f"state.{k}" for k in mc["state"].modality_keys]
lang = [k for k in obs if k.startswith("annotation") or k == "task"]
trim = {k: obs[k] for k in keep + lang}

def st(name, xs):
    a = np.asarray(xs) * 1e3
    print(f"{name:40s} n={len(a):3d} mean={a.mean():7.2f} p50={np.percentile(a,50):7.2f} p90={np.percentile(a,90):7.2f} max={a.max():7.2f} min={a.min():7.2f} ms")
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)), max=float(a.max()), min=float(a.min()))

res = {}
for name, o in [("full", obs), ("trim", trim)]:
    req = {"endpoint": "get_action", "data": {"observation": o, "options": None}}
    b = MsgSerializer.to_bytes(req)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); b = MsgSerializer.to_bytes(req); ts.append(time.perf_counter() - t0)
    td = []
    for _ in range(5):
        t0 = time.perf_counter(); MsgSerializer.from_bytes(b); td.append(time.perf_counter() - t0)
    print(f"[{name}] request bytes={len(b)/1e6:.2f} MB  serialize p50={np.median(ts)*1e3:.2f} ms  deserialize p50={np.median(td)*1e3:.2f} ms")
    res[name + "_bytes"] = len(b); res[name + "_ser_ms"] = float(np.median(ts) * 1e3); res[name + "_deser_ms"] = float(np.median(td) * 1e3)
    for _ in range(3):
        a, info = c.get_action(o)
    lat = []
    for _ in range(N):
        t0 = time.perf_counter(); a, info = c.get_action(o); lat.append(time.perf_counter() - t0)
    res[name] = st(f"VLA get_action [{name}]", lat)
for k, v in a.items():
    print("action", k, v.shape, v.dtype)
json.dump(res, open(sys.argv[3], "w"), indent=1) if len(sys.argv) > 3 else None
