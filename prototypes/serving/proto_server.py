"""PROTOTYPE (scratch only): VLA on a REP socket in the main thread (port A, unchanged PolicyServer),
WBC on its own REP socket + thread + highest-priority CUDA stream (port B). Second VLA copy; runs DUR s."""
import sys, threading, time
sys.path.insert(0, "/home/deepgadget/experiments/gr00t/Isaac-GR00T-n1.6")
import numpy as np, torch, zmq
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer, MsgSerializer
from gr00t.policy.gear_wbc_policy import GearWbcMLP, load_onnx_state_dict
from pathlib import Path

VLA_PORT, WBC_PORT, DUR = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
ONNX_DIR = "/home/deepgadget/experiments/gr00t/Isaac-GR00T-n1.6/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy"

class StreamGraphPolicy:
    """CudaGraphPolicy with replay + sync pinned to one explicit stream."""
    def __init__(self, module, stream):
        self.m = module.cuda().eval(); n = self.m.estimator[0].in_features
        self.si = torch.zeros(1, n, device="cuda"); self.hi = torch.zeros(1, n).pin_memory()
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

hp = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])  # greatest priority
wbc = {p.stem: StreamGraphPolicy(GearWbcMLP(load_onnx_state_dict(str(p))), hp) for p in sorted(Path(ONNX_DIR).glob("*.onnx"))}
policy = Gr00tSimPolicyWrapper(Gr00tPolicy(embodiment_tag=EmbodimentTag.UNITREE_G1,
          model_path="cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate", device="cuda"))
server = PolicyServer(policy, host="127.0.0.1", port=VLA_PORT)   # main-thread REP, VLA only

def wbc_loop():
    sock = server.context.socket(zmq.REP); sock.bind(f"tcp://127.0.0.1:{WBC_PORT}")  # socket owned by this thread only
    while True:
        req = MsgSerializer.from_bytes(sock.recv())
        try:
            if req.get("endpoint") == "ping": out = {"status": "ok"}
            else:
                d = req["data"]; out = {"action": wbc[d["policy"]](torch.from_numpy(d["obs"])).numpy()}
        except Exception as e:
            out = {"error": str(e)}
        sock.send(MsgSerializer.to_bytes(out))
threading.Thread(target=wbc_loop, daemon=True, name="wbc-rt").start()
threading.Timer(DUR, lambda: (setattr(server, "running", False))).start()
print("proto server ready", VLA_PORT, WBC_PORT, flush=True)
server.run()   # exits on the first message after the timer fires
