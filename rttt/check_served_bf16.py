"""Is the bf16 golden what Gr00tPolicy serves without the determinism flags? (WP-2.2, optional GPU check, srvpy.)

make_g1_golden.py runs with torch's deterministic-algorithm flags and a fixed cuBLAS workspace;
a policy server runs without them. This loads Gr00tPolicy exactly as a server does, in a fresh
process with no flags, feeds each golden observation with its saved noise through
``Gr00tSimPolicyWrapper.get_action`` and compares the normalised prediction and the decoded actions
with ``samples/<sid>.npz`` bit for bit. Exit 0 when all 20 are identical.

    srvpy -m rttt.check_served_bf16 [--golden DIR]     # needs >= 10 GB free VRAM
"""

import argparse
import os
from pathlib import Path

import torch

from rttt import golden_io as gio
from rttt.make_g1_golden import REPO, REVISION, official, snapshot_dir, wait_for_vram


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--golden", default=str(gio.GOLDEN_DIR))
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--wait-min", type=float, default=60.0)
    args = ap.parse_args(argv)
    golden = Path(args.golden).expanduser()
    wait_for_vram(args.min_free_gb, args.wait_min)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.UNITREE_G1, model_path=str(snapshot_dir(REPO, REVISION)),
                         device="cuda", strict=True)
    wrapper = Gr00tSimPolicyWrapper(policy)
    print(f"deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}, "
          f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)
    n_same = 0
    index = gio.load_index(golden)
    for s in index:
        smp = gio.load_sample(golden, s["sid"])
        pred, acts = official(wrapper, smp["obs"], torch.from_numpy(smp["noise"]))
        dp = gio.max_abs(pred, smp["bf16.action_pred_norm"])
        da = max(gio.max_abs(acts[k], smp[f"bf16.{k}"]) for k in gio.ACTION_KEYS)
        n_same += dp == 0 and da == 0
        print(f"{s['sid']}: pred max|d| {dp:.3g} actions max|d| {da:.3g}", flush=True)
    print(f"served path bit-identical to the bf16 golden: {n_same}/{len(index)}", flush=True)
    return 0 if n_same == len(index) else 1


if __name__ == "__main__":
    raise SystemExit(main())
