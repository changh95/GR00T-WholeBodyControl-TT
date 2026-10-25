"""Same-process VLA + WBC on one GPU (a SECOND copy of the VLA, not the running server).
VLA thread: back-to-back Gr00tSimPolicyWrapper.get_action on the legacy default stream.
WBC thread: 50 Hz CUDA-graph replay, in several stream / GIL-switch-interval variants."""
import json, pickle, sys, threading, time
sys.path.insert(0, "/home/deepgadget/experiments/gr00t/Isaac-GR00T-n1.6")
import numpy as np, torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.gear_wbc_policy import GearWbcMLP, load_onnx_state_dict

ONNX = "/home/deepgadget/experiments/gr00t/Isaac-GR00T-n1.6/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx"
OUT = sys.argv[2]; DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
now = time.perf_counter
print("stream priority range (least, greatest):", torch.cuda.Stream.priority_range())

class GraphWbc:
    """CudaGraphPolicy logic, but replayed + synchronized on a caller-chosen stream."""
    def __init__(self, stream):
        self.m = GearWbcMLP(load_onnx_state_dict(ONNX)).cuda().eval()
        self.si = torch.zeros(1, 516, device="cuda"); self.hi = torch.zeros(1, 516).pin_memory()
        with torch.no_grad():
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self.m(self.si)
            torch.cuda.current_stream().wait_stream(s)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g): self.so = self.m(self.si)
        self.ho = torch.zeros(self.so.shape).pin_memory(); self.stream = stream
    def __call__(self, x):
        with torch.cuda.stream(self.stream):
            self.hi.copy_(x); self.si.copy_(self.hi, non_blocking=True); self.g.replay()
            self.ho.copy_(self.so, non_blocking=True)
        self.stream.synchronize()
        return self.ho.clone()

pol = Gr00tSimPolicyWrapper(Gr00tPolicy(embodiment_tag=EmbodimentTag.UNITREE_G1,
        model_path="cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate", device="cuda"))
obs = pickle.load(open(sys.argv[1], "rb"))
mc = pol.get_modality_config()
keep = [f"video.{k}" for k in mc["video"].modality_keys] + [f"state.{k}" for k in mc["state"].modality_keys] + list(mc["language"].modality_keys)
o = {k: obs[k] for k in keep}
for _ in range(3): pol.get_action(o)
x = torch.from_numpy(np.random.default_rng(1).standard_normal((1, 516)).astype(np.float32))

variants = [
    ("idle_hiprio", torch.cuda.Stream(priority=-100), 0.005, False),
    ("legacy_default_stream", torch.cuda.default_stream(), 0.005, True),
    ("side_stream_prio0", torch.cuda.Stream(priority=0), 0.005, True),
    ("side_stream_hiprio", torch.cuda.Stream(priority=-100), 0.005, True),
    ("side_stream_hiprio_switch0.5ms", torch.cuda.Stream(priority=-100), 0.0005, True),
    ("pure_python_tick_no_gpu", None, 0.005, True),
    ("pure_python_tick_no_gpu_switch0.5ms", None, 0.0005, True),
]
res = {}
for name, stream, sw, vla_on in variants:
    sys.setswitchinterval(sw)
    wbc = GraphWbc(stream) if stream is not None else None
    if wbc: [wbc(x) for _ in range(20)]
    stop = threading.Event(); vl = []
    def vla_loop():
        while not stop.is_set():
            t0 = now(); pol.get_action(o); vl.append(now() - t0)
    th = threading.Thread(target=vla_loop, daemon=True)
    if vla_on: th.start(); time.sleep(1.0)
    lat, late = [], []
    nxt = now() + 0.02; t_end = now() + DUR
    while now() < t_end:
        s = nxt - now()
        if s > 0: time.sleep(s)
        t0 = now()
        if wbc: wbc(x)
        else: _ = (x.numpy() * 2.0).sum()
        t1 = now(); lat.append(t1 - t0); late.append(t1 - nxt); nxt += 0.02
        if t1 > nxt: nxt = t1 + 0.02  # do not accumulate drift
    stop.set(); th.join() if vla_on else None
    a = np.array(lat) * 1e3; w = (np.array(late) * 1e3)
    r = dict(n=len(a), p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
             max=float(a.max()), wake_late_p50=float(np.percentile(w - a, 50)), wake_late_p99=float(np.percentile(w - a, 99)),
             tick_complete_late_p99=float(np.percentile(w, 99)), frac_over_20ms=float((w > 20).mean()),
             vla_p50=float(np.median(vl) * 1e3) if vl else None, vla_n=len(vl))
    res[name] = r
    print(f"{name:38s} wbc p50={r['p50']:.3f} p90={r['p90']:.3f} p99={r['p99']:.3f} max={r['max']:.3f} ms | "
          f"wake-up late p50={r['wake_late_p50']:.3f} p99={r['wake_late_p99']:.3f} | tick done late p99={r['tick_complete_late_p99']:.3f} "
          f">20ms={r['frac_over_20ms']:.3f} | VLA p50={r['vla_p50']} n={r['vla_n']}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
