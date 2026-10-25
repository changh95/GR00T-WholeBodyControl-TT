"""analyze.py <wbc_log> [<vla_log>] : latency stats split by overlap with in-flight VLA calls."""
import json, sys
import numpy as np
W = [json.loads(l) for l in open(sys.argv[1])]
V = [json.loads(l) for l in open(sys.argv[2])] if len(sys.argv) > 2 and sys.argv[2] != "-" else []
lat = np.array([(w["t1"] - w["t0"]) / 1e6 for w in W])
late = np.array([(w["t1"] - w["sched"]) / 1e6 for w in W])
iv = np.array([(v["t0"], v["t1"]) for v in V]) if V else np.zeros((0, 2))
def overl(w):
    return bool(((iv[:, 0] < w["t1"]) & (iv[:, 1] > w["t0"])).any()) if len(iv) else False
ov = np.array([overl(w) for w in W])
def st(a):
    if len(a) == 0: return "n=0"
    return (f"n={len(a):4d} p50={np.percentile(a,50):7.3f} p90={np.percentile(a,90):7.3f} p99={np.percentile(a,99):7.3f} "
            f"max={a.max():7.3f} mean={a.mean():7.3f} ms")
print("ALL       ", st(lat))
print("overlapVLA", st(lat[ov]))
print("noVLA     ", st(lat[~ov]))
print(f"frac calls overlapping VLA: {ov.mean():.3f};  latency>20ms (missed 50Hz tick): {(lat>20).mean():.3f};  "
      f"completion later than sched+20ms: {(late>20).mean():.3f}")
if V:
    vl = np.array([(v["t1"] - v["t0"]) / 1e6 for v in V])
    print("VLA       ", st(vl), f" duty={vl.sum()/((V[-1]['t1']-V[0]['t0'])/1e6):.2f}")
    # blocked WBC: does latency ~= remaining VLA time when it arrived?
    rem = []
    for w, l in zip(W, lat):
        hit = iv[(iv[:, 0] <= w["t0"]) & (iv[:, 1] > w["t0"])]
        if len(hit): rem.append((l, (hit[0, 1] - w["t0"]) / 1e6))
    if rem:
        r = np.array(rem); print(f"arrived mid-VLA: n={len(r)}  mean(lat - remainingVLA)={np.mean(r[:,0]-r[:,1]):.3f} ms  corr={np.corrcoef(r[:,0], r[:,1])[0,1]:.3f}")
