"""Isaac host reference for T-2.1 (WP-2.2): processor encode and decode on the 20 golden obs.

python_env (ttpy) cannot import the Isaac processor (no albumentations), so this srvpy script
writes what the TT host pipeline must reproduce bit for bit (max|d| = 0) into isaac_ref.npz:

- encode: Gr00tN1d6Processor + collator on each observation, as Gr00tPolicy._get_action runs them
  before its bf16 cast (gr00t_policy.py:326-338): input_ids, attention_mask, pixel_values, the
  normalised state (1, 1, 128) and embodiment_id, plus the chat-template text;
- decode: processor.decode_action (the call at gr00t_policy.py:347-352) of fixed prediction arrays
  (seeded uniform in [-1.1, 1.1], so the [-1, 1] clip is exercised) with the observation's state,
  and of the bf16 golden's action_pred_norm when samples/ exists (realistic values; asserted equal
  to the golden's decoded bf16 actions).

Arrays are stacked over the samples in obs_index.json order (key ``sid``); a per-sample array
keeps its own leading batch axis, so sample i of ``in.state`` is ``z["in.state"][i]`` (1, 1, 128).
CPU only (CUDA hidden). Sidecar isaac_ref.json: keys, shapes, dtypes, seeds, sha256.

    srvpy -m rttt.make_isaac_ref [--golden ~/experiments/gr00t/golden/n16_g1]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the processor is CPU code; never touch the GPU

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from rttt import golden_io as gio
from rttt.make_g1_golden import REPO, REVISION, sha256_file, snapshot_dir

PRED_LO, PRED_HI = -1.1, 1.1


def to_numpy(v):
    """A collated value -> ndarray (a list of per-image tensors is stacked; ints keep int64)."""
    import torch

    if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
        v = torch.stack(list(v))
    if isinstance(v, torch.Tensor):
        return v.float().numpy() if v.is_floating_point() else v.numpy()
    return np.asarray(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--golden", default=str(gio.GOLDEN_DIR))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--pred-seed", type=int, default=1000, help="sample i uses seed + i")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor

    import gr00t.model  # noqa: F401  (registers the N1.6 processor)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData

    t_start = time.time()
    golden = Path(args.golden).expanduser()
    index = gio.load_index(golden)
    snap = snapshot_dir(args.repo, args.revision)
    tag = EmbodimentTag.UNITREE_G1
    proc = AutoProcessor.from_pretrained(snap)
    proc.eval()
    mc = proc.get_modality_configs()[tag.value]
    lang_key = mc["language"].modality_keys[0]
    have_golden = all((golden / "samples" / f"{s['sid']}.npz").exists() for s in index)

    per = {}  # key -> list over samples
    pred_types = {}

    def put(key, value):
        per.setdefault(key, []).append(value)

    for i, s in enumerate(index):
        sid = s["sid"]
        obs = gio.load_obs(golden / "obs" / f"{sid}.npz")
        # Gr00tSimPolicyWrapper + Gr00tPolicy._unbatch_observation/_to_vla_step_data for B = 1.
        images = {k: obs[f"video.{k}"][0] for k in mc["video"].modality_keys}
        states = {k: obs[f"state.{k}"][0] for k in mc["state"].modality_keys}
        step = VLAStepData(images=images, states=states, actions={}, text=str(obs[lang_key][0]),
                           embodiment=tag)
        processed = proc([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        inputs = proc.collator([processed])["inputs"]
        put("sid", np.array(sid))
        put("text", np.array(processed["vlm_content"]["text"]))
        for k, v in sorted(inputs.items()):  # the collator builds its keys from a set
            if k not in pred_types:
                pred_types[k] = type(v).__name__
            put(f"in.{k}", to_numpy(v))

        batched = {k: v[None] for k, v in states.items()}  # (B=1, T, D) as gr00t_policy.py:347-349
        rng = np.random.default_rng(args.pred_seed + i)
        preds = {"uniform": rng.uniform(PRED_LO, PRED_HI, gio.NOISE_SHAPE).astype(np.float32)}
        if have_golden:
            with np.load(golden / "samples" / f"{sid}.npz") as z:
                preds["golden_bf16"] = z["bf16.action_pred_norm"]
                golden_acts = {k: z[f"bf16.{k}"] for k in gio.ACTION_KEYS}
        for name, pred in preds.items():
            acts = proc.decode_action(pred, tag, batched)
            put(f"pred.{name}", pred)
            for k, v in acts.items():
                put(f"dec.{name}.action.{k}", np.asarray(v, np.float32))
            if name == "golden_bf16":
                for k in gio.ACTION_KEYS:
                    got = np.asarray(acts[k.removeprefix("action.")], np.float32)
                    assert np.array_equal(got, golden_acts[k]), f"{sid} {k}: decode != golden"
        print(f"[isaac_ref] {sid}: input_ids {tuple(inputs['input_ids'].shape)} "
              f"pixel_values {to_numpy(inputs['pixel_values']).shape} state {tuple(inputs['state'].shape)}",
              flush=True)

    shapes = {k: {tuple(np.shape(v)) for v in vs} for k, vs in per.items()}
    ragged = {k: sorted(s) for k, s in shapes.items() if len(s) > 1}
    assert not ragged, f"per-sample shapes differ, cannot stack: {ragged}"
    # C order explicitly: stacking views of torch tensors can come out F-ordered, which changes
    # the .npy header (not the values) from run to run.
    arrays = {k: np.asarray(np.stack(vs), order="C") for k, vs in per.items()}
    out = golden / "isaac_ref.npz"
    np.savez(out, **arrays)

    here = Path(__file__).resolve()
    side = {
        "what": "Isaac Gr00tN1d6Processor encode/decode reference for T-2.1 (PLAN.md WP-2.2)",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wallclock_s": round(time.time() - t_start, 1),
        "command": " ".join([sys.executable, *sys.argv]),
        "checkpoint": f"{args.repo}@{args.revision}", "snapshot": str(snap),
        "file": out.name, "sha256": sha256_file(out),
        "n_samples": len(index), "sids": [s["sid"] for s in index],
        "keys": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "collated_types": pred_types,
        "encode": "proc([EPISODE_STEP]) -> proc.collator([...])['inputs'], fp32 (before Gr00tPolicy's bf16 cast)",
        "decode": "proc.decode_action(pred, unitree_g1, {k: state[None]}) -> (1, 30, D) float32",
        "preds": {"uniform": f"np.random.default_rng({args.pred_seed} + i).uniform({PRED_LO}, {PRED_HI}, (1,50,128)).astype(float32)",
                  "golden_bf16": ("samples/<sid>.npz bf16.action_pred_norm; decode asserted equal to the golden's bf16 actions"
                                  if have_golden else "absent (no samples/ yet)")},
        "obs_index_sha256": sha256_file(golden / "obs_index.json"),
        "script": str(here), "script_sha256": sha256_file(here),
        "isaac_git_head": os.popen(f"git -C {Path(gr00t.model.__file__).resolve().parents[2]} rev-parse HEAD").read().strip(),
        "torch": torch.__version__,
    }
    (golden / "isaac_ref.json").write_text(json.dumps(side, indent=1))
    print(f"[isaac_ref] wrote {out} ({len(arrays)} keys, {out.stat().st_size / 1e6:.1f} MB) and isaac_ref.json",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
