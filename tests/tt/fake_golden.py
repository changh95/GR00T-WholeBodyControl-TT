"""A 20-sample G1 golden in the WP-2.2 layout (rttt/golden_io.py) that tests/tt/fake_g1_server.py reproduces (tests).

The fp32 run is the fake model itself and the bf16 run its bf16 rounding, so a fake server with a small
--perturb passes T-2.2 and a large one fails it.
"""

import json
from pathlib import Path

import numpy as np

from rtloop import schema
from rttt import bench_g1 as B
from rttt import golden_io as gio

from . import fake_g1_server as F


def bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 (round to nearest even) -> float32."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return r.view(np.float32)


def build(d: Path, n: int = 20) -> Path:
    d = Path(d)
    (d / "obs").mkdir(parents=True)
    (d / "samples").mkdir()
    index, per_group = [], {name: 0.0 for name, _ in schema.ACTION_GROUPS}
    for i in range(n):
        sid, kind = f"s{i:02d}", ("reset" if i < n // 2 else "mid")
        obs, noise = B.synthetic_obs(100 + i), B.fixed_noise(1000 + i)
        fp = F.fake_pred(obs, noise)
        arrays = {"noise": noise, "noise_seed": np.array(1000 + i, np.int64), "bf16.noise_in": bf16(noise)}
        dec = {}
        for p, pred in (("fp32", fp), ("bf16", bf16(fp))):
            arrays[f"{p}.action_pred_norm"] = pred
            dec[p] = F.decode(pred)
            arrays.update({f"{p}.{k}": v for k, v in dec[p].items()})
        for name, _ in schema.ACTION_GROUPS:
            k = f"action.{name}"
            per_group[name] = max(per_group[name], gio.max_abs(dec["bf16"][k], dec["fp32"][k]))
        np.savez(d / "samples" / f"{sid}.npz", **arrays)
        gio.save_obs(d / "obs" / f"{sid}.npz", obs)
        index.append({"sid": sid, "kind": kind, "seed": 100 + i, "k_obs": 0})
    (d / "obs_index.json").write_text(json.dumps({"samples": index}))
    floors = {k: {"max_abs": v, "floor": max(2 * v, 0.02)} for k, v in per_group.items()}
    (d / "manifest.json").write_text(json.dumps({"bf16_vs_fp32": {"per_group": floors}}))
    return d
