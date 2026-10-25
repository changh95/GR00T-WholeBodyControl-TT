import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
from transformers import AutoProcessor
import gr00t.model  # noqa: F401  (registers the processor)
from gr00t.data.types import VLAStepData, MessageType
from gr00t.data.embodiment_tags import EmbodimentTag
CW = os.path.expanduser("~/.cache/huggingface/hub/models--cloudwalk-research--GR00T-N1.6-G1-PnPAppleToPlate/snapshots/2a5b61cc7660bdef4cf17ad18dc5dd97f13e4ce5")
z = np.load(sys.argv[1]); out = {}
p = AutoProcessor.from_pretrained(CW); p.eval()
state = {k[6:]: z[k] for k in z.files if k.startswith("state.")}
instr = "pick up the apple, walk left and place the apple on the plate."
for name in ("img_noise", "img_grad"):
    step = VLAStepData(images={"ego_view": z[name]}, states=state, actions={}, text=instr, embodiment=EmbodimentTag.UNITREE_G1)
    t0 = time.perf_counter()
    proc = p([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    coll = p.collator([proc])
    dt = time.perf_counter() - t0
    b = coll["inputs"]
    print(name, "keys", sorted(b.keys()), "host ms %.1f" % (dt * 1e3))
    for k, v in b.items():
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
            v = torch.stack(list(v))
        if isinstance(v, torch.Tensor):
            out[f"{name}.{k}"] = v.float().numpy() if v.is_floating_point() else v.numpy()
            print("  ", k, tuple(v.shape), v.dtype)
bs = {k: v[None] for k, v in state.items()}
acts = p.decode_action(z["pred"], EmbodimentTag.UNITREE_G1, bs)
for k, v in acts.items():
    out["act." + k] = np.asarray(v, np.float32); print("  act", k, np.asarray(v).shape)
np.savez(sys.argv[2], **out)
