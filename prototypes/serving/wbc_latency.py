"""Idle-server latency of ping and gear_wbc_action on the running GR00T server (no VLA in flight)."""
import json, sys, time
import numpy as np
from gr00t.policy.server_client import PolicyClient, MsgSerializer

HOST, PORT = "127.0.0.1", 5555
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = sys.argv[2] if len(sys.argv) > 2 else None

def stats(name, xs):
    a = np.asarray(xs) * 1e3
    d = dict(name=name, n=len(a), mean=a.mean(), p50=np.percentile(a, 50), p90=np.percentile(a, 90),
             p99=np.percentile(a, 99), p999=np.percentile(a, 99.9), max=a.max(), min=a.min())
    print(f"{name:34s} n={d['n']:5d} mean={d['mean']:.3f} p50={d['p50']:.3f} p90={d['p90']:.3f} "
          f"p99={d['p99']:.3f} p99.9={d['p999']:.3f} max={d['max']:.3f} min={d['min']:.3f} ms")
    return d

c = PolicyClient(host=HOST, port=PORT, timeout_ms=15000)
assert c.ping()
rng = np.random.default_rng(0)
res = {}
# warmup
for _ in range(50):
    c.call_endpoint("gear_wbc_action", {"policy": "GR00T-WholeBodyControl-Balance",
                                         "obs": rng.standard_normal((1, 516)).astype(np.float32)})
t_ping = []
for _ in range(N):
    t0 = time.perf_counter(); c.call_endpoint("ping", requires_input=False); t_ping.append(time.perf_counter() - t0)
res["ping"] = stats("ping (REQ/REP + msgpack, no GPU)", t_ping)
for pol in ["GR00T-WholeBodyControl-Balance", "GR00T-WholeBodyControl-Walk"]:
    ts = []
    for _ in range(N):
        obs = rng.standard_normal((1, 516)).astype(np.float32)
        t0 = time.perf_counter()
        r = c.call_endpoint("gear_wbc_action", {"policy": pol, "obs": obs})
        ts.append(time.perf_counter() - t0)
    assert r["action"].shape == (1, 15), r["action"].shape
    res[pol] = stats(f"gear_wbc_action {pol.split('-')[-1]}", ts)
# 50 Hz paced (like the sim tick), to see whether idle gaps change latency (GPU clock down-scaling)
ts = []
nxt = time.perf_counter()
for i in range(500):
    nxt += 0.02
    obs = rng.standard_normal((1, 516)).astype(np.float32)
    t0 = time.perf_counter()
    c.call_endpoint("gear_wbc_action", {"policy": "GR00T-WholeBodyControl-Balance", "obs": obs})
    ts.append(time.perf_counter() - t0)
    s = nxt - time.perf_counter()
    if s > 0: time.sleep(s)
res["paced50"] = stats("gear_wbc_action Balance @50Hz paced", ts)
# client-side serialization cost only
obs = rng.standard_normal((1, 516)).astype(np.float32)
req = {"endpoint": "gear_wbc_action", "data": {"policy": "GR00T-WholeBodyControl-Balance", "obs": obs}}
ts = []
for _ in range(N):
    t0 = time.perf_counter(); b = MsgSerializer.to_bytes(req); MsgSerializer.from_bytes(b); ts.append(time.perf_counter() - t0)
res["ser"] = stats("msgpack+np.save roundtrip (516 f32)", ts)
print("request bytes:", len(b))
if OUT:
    json.dump({k: {kk: float(vv) if not isinstance(vv, str) else vv for kk, vv in v.items()} for k, v in res.items()}, open(OUT, "w"), indent=1)
