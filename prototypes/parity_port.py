import os, sys, time
from pathlib import Path
import numpy as np, torch
D = Path(sys.argv[1]).parent
os.environ["HF_HUB_CACHE"] = str(Path(__file__).parent / "g1_host_probe" / "hubshim")  # pinned sha -> cloudwalk snapshot
from models.experimental.gr00t.common.preprocessing import Observation, encode
from models.experimental.gr00t.common import normalization as Nz
z = np.load(sys.argv[1]); ref = np.load(sys.argv[2])
state = {k[6:]: z[k] for k in z.files if k.startswith("state.")}
instr = "pick up the apple, walk left and place the apple on the plate."
for name in ("img_noise", "img_grad"):
    obs = Observation(images={"ego_view": z[name]}, state=state, instruction=instr, embodiment_tag="unitree_g1")
    mi = encode("n16", obs, embodiment="unitree_g1", layout_name="gr1")
    ids_ok = np.array_equal(mi.input_ids.numpy(), ref[f"{name}.input_ids"])
    pv_p = mi.pixel_values.numpy().reshape(-1); pv_r = ref[f"{name}.pixel_values"].reshape(-1)
    st_p = mi.state.numpy(); st_r = ref[f"{name}.state"]
    print(f"{name}: input_ids equal={ids_ok} (port {tuple(mi.input_ids.shape)} ref {ref[name+'.input_ids'].shape}); "
          f"pixel max|d|={np.abs(pv_p - pv_r).max():.3g} equal={np.array_equal(pv_p, pv_r)} n_diff={(pv_p != pv_r).sum()}; "
          f"state max|d|={np.abs(st_p - st_r).max():.3g} equal={np.array_equal(st_p, st_r)}; emb_id port {int(mi.embodiment_id[0])} ref {int(ref[name+'.embodiment_id'][0])}")
acts = Nz.decode("n16", torch.from_numpy(z["pred"]), obs, embodiment="unitree_g1")
for k, v in acts.items():
    r = ref["act." + k]; v = v.numpy()
    print(f"  act {k}: shape {v.shape} vs {r.shape} max|d|={np.abs(v - r).max():.3g} equal={np.array_equal(v, r)}")
