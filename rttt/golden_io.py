"""Layout of the G1 goldens (WP-2.2) and numpy-only readers for them.

Importable from every interpreter (simpy, srvpy, ttpy): only numpy, the stdlib and rtloop.schema,
so the CPU parity test (T-2.1, python_env) and the fidelity test (T-2.2, simpy) read the same
files the same way. Layout of ``~/experiments/gr00t/golden/n16_g1`` (GOLDEN.md):

    obs_index.json     the 20 samples: sid, kind (reset|mid), source run/seq/seed/k_obs, sha1
    obs/<sid>.npz      the raw W-VLA observation, keys exactly as sent (OBS_KEYS)
    samples/<sid>.npz  noise + per precision action_pred_norm and decoded actions (SAMPLE_KEYS)
    isaac_ref.npz      host encode/decode reference of the Isaac processor (make_isaac_ref.py)
    manifest.json      checkpoint, dtypes, keys/shapes, noise seeds, sha256s, measured floors
"""

import json
from pathlib import Path

import numpy as np

from rtloop import schema

GOLDEN_DIR = Path.home() / "experiments/gr00t/golden/n16_g1"
OBS_KEYS = schema.VLA_OBS_KEYS  # video, 7 states, language (W-VLA request, PLAN.md §1.4.1)
ACTION_KEYS = tuple(f"action.{name}" for name, _ in schema.ACTION_GROUPS)
NOISE_SHAPE = (1, 50, 128)  # (B, action_horizon, max_action_dim) of the N1.6 head
VALID_ROWS, VALID_DIMS = schema.H, schema.A35  # action_pred_valid = pred[0, :30, :35]
PRECISIONS = ("bf16", "fp32")


def dump_key(key: str) -> str:
    """Key of a W-VLA obs field in the loop's own dump (telemetry.save_vla_obs replaces '.')."""
    return key.replace(".", "__")


def load_loop_dump(path) -> dict:
    """A loop ``vla_obs/<seq>.npz`` -> the W-VLA observation dict exactly as it was sent."""
    with np.load(path) as z:
        return _obs_from(z, lambda k: dump_key(k))


def save_obs(path, obs: dict) -> None:
    """Write a W-VLA observation with its own keys (language as a 0-d str array)."""
    np.savez(path, **obs_arrays(obs))


def obs_arrays(obs: dict, prefix: str = "") -> dict:
    out = {}
    for k in OBS_KEYS:
        v = obs[k]
        out[prefix + k] = np.array(str(v[0])) if k == schema.LANGUAGE_KEY else np.asarray(v)
    return out


def load_obs(path, prefix: str = "") -> dict:
    with np.load(path) as z:
        return _obs_from(z, lambda k: prefix + k)


def _obs_from(z, name) -> dict:
    obs = {}
    for k in OBS_KEYS:
        v = z[name(k)]
        obs[k] = (str(v[()]),) if k == schema.LANGUAGE_KEY else np.array(v)
    assert obs[schema.VIDEO_KEY].dtype == np.uint8 and obs[schema.VIDEO_KEY].shape == (1, 1, *schema.IMAGE_SHAPE)
    for (name_, n), key in zip(schema.STATE_GROUPS, schema.STATE_KEYS):
        assert obs[key].dtype == np.float32 and obs[key].shape == (1, 1, n), (key, obs[key].shape)
    return obs


def load_index(golden_dir=GOLDEN_DIR) -> list[dict]:
    return json.loads((Path(golden_dir) / "obs_index.json").read_text())["samples"]


def load_sample(golden_dir, sid: str) -> dict:
    """samples/<sid>.npz as a dict of arrays plus ``obs`` (the W-VLA request of obs/<sid>.npz)."""
    golden_dir = Path(golden_dir)
    with np.load(golden_dir / "samples" / f"{sid}.npz") as z:
        out = {k: z[k] for k in z.files}
    out["obs"] = load_obs(golden_dir / "obs" / f"{sid}.npz")
    return out


def actions(sample: dict, prec: str) -> dict:
    """The decoded W-VLA reply of one precision: {'action.<group>': f32 (1, 30, D)}."""
    return {k: sample[f"{prec}.{k}"] for k in ACTION_KEYS}


def valid(pred: np.ndarray) -> np.ndarray:
    """action_pred_valid: the trained rows and the G1 action dims of a (1, 50, 128) prediction."""
    pred = np.asarray(pred)
    assert pred.shape == NOISE_SHAPE, pred.shape
    return pred[0, :VALID_ROWS, :VALID_DIMS]


def pcc(a, b) -> float:
    """Pearson correlation of two arrays (flattened, float64), as the TT goldens define it."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float((a * b).sum() / den)


def max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def group_floors(manifest: dict) -> dict:
    """T-2.2 per-group absolute floors (decoded max|d| vs the bf16 golden) from manifest.json."""
    return {g: v["floor"] for g, v in manifest["bf16_vs_fp32"]["per_group"].items()}
