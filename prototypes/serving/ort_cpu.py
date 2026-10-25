"""onnxruntime CPU latency of the WBC (the in-sim fallback path, g1_gear_wbc_policy.py:58,68-71)."""
import time, numpy as np, onnxruntime as ort
P = "/home/deepgadget/experiments/gr00t/Isaac-GR00T-n1.6/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx"
x = np.random.default_rng(0).standard_normal((1, 516)).astype(np.float32)
for name, threads in [("default opts", None), ("intra_op=1", 1)]:
    so = ort.SessionOptions()
    if threads: so.intra_op_num_threads = threads
    s = ort.InferenceSession(P, so, providers=["CPUExecutionProvider"]); n = s.get_inputs()[0].name
    for _ in range(200): s.run(None, {n: x})
    for mode in ["back-to-back", "50Hz paced"]:
        ts = []
        for i in range(2000 if mode == "back-to-back" else 300):
            if mode != "back-to-back": time.sleep(0.02)
            t0 = time.perf_counter(); s.run(None, {n: x}); ts.append(time.perf_counter() - t0)
        a = np.array(ts) * 1e3
        print(f"ORT CPU {name:12s} {mode:12s} p50={np.percentile(a,50):.3f} p99={np.percentile(a,99):.3f} max={a.max():.3f} ms")
print("ort", ort.__version__)
