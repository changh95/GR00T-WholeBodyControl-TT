"""Offline validation of the G1 golden directory (WP-2.2); numpy only, runs under simpy, srvpy or ttpy.

Re-derives every number the manifest states from the files themselves: file hashes, keys/shapes,
the obs selection (10 reset with distinct seeds + 10 mid), bf16 rounding of the noise and of the
bf16 prediction, the bf16-vs-fp32 statistics and T-2.2 floors, and the isaac_ref.npz cross-check
(its decode of the golden bf16 prediction must equal the golden's decoded actions). If torch is
importable, it also reports whether this torch regenerates the noise from its seed bit-exactly
(informational: consumers must use the saved ``noise``, never the seed).

    simpy -m rttt.check_golden [--golden DIR]      # exit 0 = every check passed
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rtloop import schema
from rttt import golden_io as gio


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bf16_round(x: np.ndarray) -> np.ndarray:
    """float32 -> nearest bfloat16 (round to nearest even), returned as float32."""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return b.astype(np.uint32).view(np.float32)


def is_bf16(x: np.ndarray) -> bool:
    return not np.any(np.ascontiguousarray(x, np.float32).view(np.uint32) & 0xFFFF)


class Checks:
    def __init__(self):
        self.passed, self.failed = [], []

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        (self.passed if ok else self.failed).append(name)
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--golden", default=str(gio.GOLDEN_DIR))
    ap.add_argument("--tol", type=float, default=1e-9, help="stat recomputation tolerance")
    args = ap.parse_args(argv)
    g = Path(args.golden).expanduser()
    c = Checks()
    man = json.loads((g / "manifest.json").read_text())
    index = gio.load_index(g)
    by_sid = {s["sid"]: s for s in man["samples"]}

    # --- observations -------------------------------------------------------------------------
    kinds = [s["kind"] for s in index]
    c("index: 10 reset + 10 mid", kinds.count("reset") == 10 and kinds.count("mid") == 10, str(kinds.count("reset")))
    reset_seeds = [s["seed"] for s in index if s["kind"] == "reset"]
    c("index: reset samples from 10 distinct seeds", len(set(reset_seeds)) == 10, str(reset_seeds))
    c("index: reset samples at k_obs 0", all(s["k_obs"] == 0 for s in index if s["kind"] == "reset"))
    mids = [s for s in index if s["kind"] == "mid"]
    c("index: mid samples moving (walk or manip)", all(s["picked_as"] in ("walk", "manip") for s in mids),
      ", ".join(f"{s['sid']}:{s['picked_as']}@{s['k_obs']}" for s in mids))
    c("manifest: obs_index sha256", sha256(g / "obs_index.json") == man["obs_source"]["index_sha256"])
    obs_bad = []
    for s in index:
        obs = gio.load_obs(g / "obs" / f"{s['sid']}.npz")
        if not (schema.obs_sha1(obs).tobytes().hex() == s["obs_sha1"]
                and sha256(g / "obs" / f"{s['sid']}.npz") == by_sid[s["sid"]]["obs_sha256"]
                and np.array_equal(schema.state43(obs), np.asarray(s["state43"], np.float32))):
            obs_bad.append(s["sid"])
    c("obs: every sha1, sha256 and state43 matches the index and manifest", not obs_bad, ", ".join(obs_bad))

    # --- samples ------------------------------------------------------------------------------
    keys = man["sample_keys"]
    pccs, worst = [], {name: 0.0 for name, _ in schema.ACTION_GROUPS}
    bad = []
    for s in index:
        sid = s["sid"]
        path = g / "samples" / f"{sid}.npz"
        if sha256(path) != by_sid[sid]["sha256"]:
            bad.append(f"{sid}: sha256")
        z = gio.load_sample(g, sid)
        for k, spec in keys.items():
            if k not in z or list(z[k].shape) != spec["shape"] or str(z[k].dtype) != spec["dtype"]:
                bad.append(f"{sid}: key {k}")
        if int(z["noise_seed"]) != by_sid[sid]["noise_seed"]:
            bad.append(f"{sid}: noise_seed")
        if not np.array_equal(z["bf16.noise_in"], bf16_round(z["noise"])):
            bad.append(f"{sid}: bf16.noise_in != bf16(noise)")
        if not is_bf16(z["bf16.action_pred_norm"]):
            bad.append(f"{sid}: bf16 prediction not bf16-representable")
        for p in gio.PRECISIONS:
            a = gio.actions(z, p)
            if not all(np.isfinite(v).all() for v in a.values()):
                bad.append(f"{sid}: {p} non-finite action")
            if not (np.all(a["action.left_hand"] == 0) and np.all(a["action.base_height_command"] == np.float32(0.74))):
                bad.append(f"{sid}: {p} constant groups")
        pcc = gio.pcc(gio.valid(z["bf16.action_pred_norm"]), gio.valid(z["fp32.action_pred_norm"]))
        if abs(pcc - by_sid[sid]["bf16_vs_fp32"]["pcc_valid"]) > args.tol:
            bad.append(f"{sid}: pcc_valid")
        pccs.append(pcc)
        for name, _ in schema.ACTION_GROUPS:
            d = gio.max_abs(z[f"bf16.action.{name}"], z[f"fp32.action.{name}"])
            worst[name] = max(worst[name], d)
    c("samples: sha256, keys/shapes/dtypes, seeds, bf16 rounding, finite, constant groups, pcc", not bad,
      "; ".join(bad[:8]))
    summ = man["bf16_vs_fp32"]
    c("summary: pcc_valid min/mean", abs(min(pccs) - summ["pcc_valid"]["min"]) <= args.tol
      and abs(float(np.mean(pccs)) - summ["pcc_valid"]["mean"]) <= args.tol,
      f"min {min(pccs):.6f} mean {np.mean(pccs):.6f}")
    floors_ok = all(abs(worst[n] - v["max_abs"]) <= args.tol and abs(v["floor"] - max(2 * worst[n], 0.02)) <= args.tol
                    for n, v in summ["per_group"].items())
    c("summary: per-group max|d| and floors", floors_ok,
      ", ".join(f"{n} {v['floor']:.4f}" for n, v in summ["per_group"].items()))

    # --- isaac_ref ----------------------------------------------------------------------------
    side = json.loads((g / "isaac_ref.json").read_text())
    c("isaac_ref: sha256", sha256(g / "isaac_ref.npz") == side["sha256"])
    with np.load(g / "isaac_ref.npz") as z:
        ref = {k: z[k] for k in z.files}
    c("isaac_ref: sample order", list(ref["sid"]) == [s["sid"] for s in index])
    c("isaac_ref: embodiment_id 8", bool(np.all(ref["in.embodiment_id"] == 8)))
    c("isaac_ref: one prompt (input_ids equal across samples)",
      bool(np.all(ref["in.input_ids"] == ref["in.input_ids"][:1])), str(ref["in.input_ids"].shape))
    c("isaac_ref: state and pixels in [-1, 1]",
      bool(np.abs(ref["in.state"]).max() <= 1 and np.abs(ref["in.pixel_values"]).max() <= 1))
    same = True
    for i, s in enumerate(index):
        z = gio.load_sample(g, s["sid"])
        same &= np.array_equal(ref["pred.golden_bf16"][i], z["bf16.action_pred_norm"])
        for k in gio.ACTION_KEYS:
            same &= np.array_equal(ref[f"dec.golden_bf16.{k}"][i], z[f"bf16.{k}"])
    c("isaac_ref: decode of the golden bf16 prediction == golden bf16 actions", bool(same))
    clipped = np.abs(ref["pred.uniform"][..., :30, :35]) > 1
    c("isaac_ref: uniform preds exercise the [-1, 1] clip", bool(clipped.any()), f"{clipped.mean():.3f} of values")

    # --- informational: noise from seed under this torch -------------------------------------
    try:
        import torch

        regen = 0
        for s in index:
            gen = torch.Generator(device="cpu").manual_seed(by_sid[s["sid"]]["noise_seed"])
            n = torch.randn(gio.NOISE_SHAPE, generator=gen, dtype=torch.float32).numpy()
            regen += np.array_equal(n, gio.load_sample(g, s["sid"])["noise"])
        print(f"INFO torch {torch.__version__}: noise regenerated bit-exactly from its seed for {regen}/20 samples",
              flush=True)
    except ImportError:
        print("INFO torch not importable here; noise-from-seed not checked", flush=True)

    print(f"{len(c.passed)} passed, {len(c.failed)} failed", flush=True)
    return 1 if c.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
