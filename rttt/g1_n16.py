"""GR00T N1.6 on the Unitree G1 (unitree_g1, slot 8) for the TT port: registry patch, host pipeline, model loading.

Port of ``TT/libero/n16.py`` (``install`` at :176-229, host pipeline at :232-313, loading/stamp at :319-384) to the
DP1 checkpoint ``cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc`` (PLAN.md D11, WP-2.1). Nothing in the
port is edited: :func:`install` patches its registries in this process only, exactly as the LIBERO install does.

What :func:`install` does, and why:

1. ``HF_HUB_CACHE`` -> ``~/.cache/gr00t-tt-g1/hub`` (process-local) and ``configs.HF_REPOS["n16"]`` /
   ``HF_SNAPSHOT_SHAS["n16"]`` -> the BF16 copy written by :mod:`rttt.convert_bf16` (WP-2.1b). The port resolves
   every checkpoint, ``processor_config.json`` and ``statistics.json`` read through ``snapshot_dir("n16")``
   (``MK/.../common/configs.py:54-70``), so this one re-point moves all of them. The tokenizer is vendored
   (``common/prompts.py:51-58``), so the hub redirect cannot affect it.
2. The copy is checked before anything uses it: 1106 tensors, every dtype BF16, total 6,573,217,664 B and the
   per-shard framing (``common/checkpoint.py:252-261``), because the port reads tensors as stored.
3. ``configs._CONFIGS["n16"]`` gets the copy's repo/sha and one more static layout, ``g1`` = ``gr1`` with the G1
   embodiment (``common/configs.py:978-1007``): the G1 prompt fits gr1's geometry (1 camera ``ego_view``, 81 tokens,
   13 text tokens <= L_max 24, S = 117 -> S_pad 128), so every device shape is gr1's. No layer-table rows are added
   (the LIBERO ``_extend_layer_tables`` is not ported): MK already has the S_pad-128 rows, see
   :func:`layer_table_rows`. ``valid_slice`` and ``action_groups`` are golden metadata only (U:tt "UNITREE_G1
   support: layout gap"); ``action_groups`` is set to the G1 groups and ``s_golden`` cleared (no gr1 golden applies).
4. The normalization ``lru_cache``s are cleared (``common/normalization.py:146-177, 414``) and the resolved spec is
   asserted against the checkpoint's own ``processor_config.json`` / ``statistics.json``: slot 8, ``ego_view``,
   state 43 in :data:`rtloop.schema.STATE_GROUPS` order (all clipped min-max), action 35 in A35 order, horizon 30,
   both arms RELATIVE with (30, 7) ``relative_action`` bounds, the rest ABSOLUTE.

Weights cache: a dedicated ``cache_root`` ``~/.cache/gr00t-tt-g1/<repo>@<sha8>/`` (U:tt "Weight-cache key
collision": ``HostPlanCache`` keys by version / sha / slot, ``common/weights.py:1942-1962``).

Use::

    from rttt import g1_n16 as G
    G.install()
    mi, obs = G.encode_obs(one)                 # one un-batched element of a W-VLA observation
    model = G.load_tt_model()                   # Gr00tTT, megakernel DiT (MK TTPolicy() default); opens the device
    pred = model.predict_normalized(mi, noise)  # (1, 50, 128)
    acts = G.decode_actions(pred, obs)          # {"action.<group>": (1, 30, D) float32}
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from rtloop import schema
from rttt import convert_bf16 as CB

VERSION = "n16"
SRC_REPO, SRC_SHA = CB.SRC_REPO, CB.SRC_SHA  # DP1: the F32 checkpoint the 5090 serves
REPO, SHA = CB.BF16_REPO, CB.BF16_SHA  # what the port reads (WP-2.1b copy)
LAYOUT_NAME = "g1"
BASE_LAYOUT = "gr1"
EMBODIMENT_TAG = "unitree_g1"
EMBODIMENT_ID = 8  # common/configs.py:806-816 and the checkpoint's embodiment_id.json
VIDEO_KEYS: Tuple[str, ...] = ("ego_view",)
STATE_KEYS: Tuple[str, ...] = tuple(n for n, _ in schema.STATE_GROUPS)
ACTION_KEYS: Tuple[str, ...] = tuple(n for n, _ in schema.ACTION_GROUPS)
RELATIVE_KEYS: Tuple[str, ...] = ("left_arm", "right_arm")
LANG_KEY = schema.LANGUAGE_KEY
MODALITY_KEYS: Tuple[str, ...] = schema.VLA_OBS_KEYS  # the 9 keys server_g1 keeps (PLAN.md §1.4.1)
STATE_DIM, ACTION_DIM = schema.S43, schema.A35
ACTION_HORIZON = schema.H  # 30 trained rows
MODEL_HORIZON, ACTION_DIM_PAD = 50, 128
DENOISING_STEPS = 4  # config.json num_inference_timesteps
S_PAD, L_MAX, TOKENS_PER_IMAGE = 128, 24, 81
N_TENSORS, N_BYTES = CB.EXPECT_N_TENSORS, CB.EXPECT_BF16_BYTES
HUB_ENV = "GR00T_TT_G1_HUB"
CACHE_ENV = "GR00T_TT_G1_CACHE"
DEFAULT_HUB = CB.DEFAULT_OUT_HUB
G1_ROOT = Path.home() / ".cache/gr00t-tt-g1"

_LOCK = threading.Lock()
_INSTALLED = False
INSTALL_FACTS: Dict[str, Any] = {}


def hub_cache() -> Path:
    return Path(os.environ.get(HUB_ENV, str(DEFAULT_HUB))).expanduser()


def snapshot_path(hub: Optional[Path] = None) -> Path:
    return CB.snapshot_path(Path(hub) if hub is not None else hub_cache())


def default_cache_root() -> Path:
    """``~/.cache/gr00t-tt-g1/<repo>@<sha8>/``, so a G1 plan can never land in the base checkpoint's cache."""
    if CACHE_ENV in os.environ:
        return Path(os.environ[CACHE_ENV]).expanduser()
    return G1_ROOT / f"{REPO.replace('/', '--')}@{SHA[:8]}"


# --------------------------------------------------------------------------------------------------------------------
# install (process-local registry patches; module docstring)
# --------------------------------------------------------------------------------------------------------------------
def g1_layout(gr1: Any) -> Any:
    """gr1 with the G1 embodiment; every geometric field (and so every device shape) is gr1's."""
    return dataclasses.replace(
        gr1,
        name=LAYOUT_NAME,
        embodiment_tag=EMBODIMENT_TAG,
        embodiment_id=EMBODIMENT_ID,
        valid_slice=(ACTION_HORIZON, ACTION_DIM),
        action_groups=ACTION_KEYS,
        s_golden=None,
        s_typical="117 (13-token apple instruction)",
        notes="G1 (unitree_g1, slot 8) copy of gr1, injected at runtime by rtloop rttt/g1_n16.py.",
    )


def straddling_unit(weight_map: Mapping[str, str]) -> Optional[str]:
    """The LLM layer / DiT block whose tensors sit in both shards (the copy keeps the source's map, so it is not
    base N1.6's block 18; only MK's checkpoint tests read this field)."""
    import re

    units: Dict[str, set] = {}
    for k, shard in weight_map.items():
        m = re.match(r"(.*?(?:transformer_blocks|layers)\.\d+\.)", k)
        if m:
            units.setdefault(m.group(1), set()).add(shard)
    split = sorted(u for u, s in units.items() if len(s) > 1)
    return split[0] if len(split) == 1 else None


def check_checkpoint(snap: Path) -> Dict[str, Any]:
    """Header-only checks of the BF16 copy (no tensor is read): the port's bf16 invariant holds."""
    from models.experimental.gr00t.common.checkpoint import LazyCheckpoint

    ck = LazyCheckpoint(snap)
    dtypes = sorted({ck.info(k).dtype for k in ck.keys()})
    if len(ck) != N_TENSORS or dtypes != ["BF16"]:
        raise AssertionError(f"{snap}: {len(ck)} tensors with dtypes {dtypes}; expected {N_TENSORS} BF16")
    per_shard = ck.verify_headers()
    if ck.total_bytes != N_BYTES or len(per_shard) != 2:
        raise AssertionError(f"{snap}: {ck.total_bytes} B in {len(per_shard)} shards; expected {N_BYTES} in 2")
    return {"n_tensors": len(ck), "dtypes": dtypes, "total_bytes": ck.total_bytes,
            "straddling": straddling_unit(ck.weight_map)}


def checkpoint_metadata(snap: Path) -> Dict[str, Any]:
    """The G1 facts read straight from the checkpoint's own files, independently of the port's parser."""
    kw = json.loads((snap / "processor_config.json").read_text())["processor_kwargs"]
    mc = kw["modality_configs"][EMBODIMENT_TAG]
    st = json.loads((snap / "statistics.json").read_text())[EMBODIMENT_TAG]
    ids = json.loads((snap / "embodiment_id.json").read_text())
    cfg = json.loads((snap / "config.json").read_text())
    acfg = mc["action"]["action_configs"]
    return {
        "embodiment_id": int(ids[EMBODIMENT_TAG]),
        "video_keys": tuple(mc["video"]["modality_keys"]),
        "state_keys": tuple(mc["state"]["modality_keys"]),
        "state_dims": {k: len(st["state"][k]["min"]) for k in mc["state"]["modality_keys"]},
        "action_keys": tuple(mc["action"]["modality_keys"]),
        "action_dims": {k: len(st["action"][k]["min"]) for k in mc["action"]["modality_keys"]},
        "horizon": len(mc["action"]["delta_indices"]),
        "reps": {k: c["rep"] for k, c in zip(mc["action"]["modality_keys"], acfg)},
        "types": {k: c["type"] for k, c in zip(mc["action"]["modality_keys"], acfg)},
        "relative_shapes": {k: tuple(np.asarray(v["min"]).shape) for k, v in st.get("relative_action", {}).items()},
        "language_keys": tuple(mc["language"]["modality_keys"]),
        "sin_cos_keys": mc["state"].get("sin_cos_embedding_keys"),
        "use_relative_action": bool(kw["use_relative_action"]),
        "use_percentiles": bool(kw["use_percentiles"]),
        "clip_outliers": bool(kw["clip_outliers"]),
        "num_inference_timesteps": int(cfg["num_inference_timesteps"]),
    }


def assert_spec(spec: Any, meta: Mapping[str, Any]) -> None:
    """The resolved port spec equals the checkpoint's metadata and the rtloop schema (WP-2.1 spec asserts)."""
    want_state = tuple((n, d) for n, d in schema.STATE_GROUPS)
    want_action = tuple((n, d) for n, d in schema.ACTION_GROUPS)
    checks = [
        ("embodiment_id", spec.embodiment_id, EMBODIMENT_ID),
        ("meta embodiment_id", meta["embodiment_id"], EMBODIMENT_ID),
        ("video_keys", spec.video_keys, VIDEO_KEYS),
        ("meta video_keys", meta["video_keys"], VIDEO_KEYS),
        ("frames_per_camera", spec.frames_per_camera, 1),
        ("state groups", tuple((g.key, g.dim) for g in spec.state_groups), want_state),
        ("meta state groups", tuple(meta["state_dims"].items()), want_state),
        ("state_dim", spec.state_dim, STATE_DIM),
        ("action groups", tuple((g.key, g.dim) for g in spec.action_groups), want_action),
        ("meta action groups", tuple(meta["action_dims"].items()), want_action),
        ("action_dim", spec.action_dim, ACTION_DIM),
        ("action_horizon", spec.action_horizon, ACTION_HORIZON),
        ("meta horizon", meta["horizon"], ACTION_HORIZON),
        ("language", meta["language_keys"], (LANG_KEY,)),
        ("denoising steps", meta["num_inference_timesteps"], DENOISING_STEPS),
        ("max dims", (spec.max_state_dim, spec.max_action_dim), (ACTION_DIM_PAD, ACTION_DIM_PAD)),
    ]
    for name, got, want in checks:
        if got != want:
            raise AssertionError(f"unitree_g1 spec: {name} = {got!r}, expected {want!r}")
    for g in spec.state_groups:
        if g.mode != "min_max" or not g.clip:
            raise AssertionError(f"state.{g.key}: {g.mode} clip={g.clip}; G1 states are clipped min-max")
    for g in spec.action_groups:
        rel = g.key in RELATIVE_KEYS
        if g.rep != ("RELATIVE" if rel else "ABSOLUTE") or meta["reps"][g.key] != g.rep:
            raise AssertionError(f"action.{g.key}: rep {g.rep} (checkpoint {meta['reps'][g.key]})")
        if g.type != "NON_EEF" or meta["types"][g.key] != "NON_EEF":
            raise AssertionError(f"action.{g.key}: type {g.type}")
        if rel and (g.stats_source != "relative_action.min/max" or g.lo.shape != (ACTION_HORIZON, g.dim)
                    or meta["relative_shapes"].get(g.key) != (ACTION_HORIZON, g.dim)):
            raise AssertionError(f"action.{g.key}: {g.stats_source} {g.lo.shape}; expected per-step (30, {g.dim})")
        if not rel and (g.stats_source != "action.min/max" or g.lo.shape != (g.dim,)):
            raise AssertionError(f"action.{g.key}: {g.stats_source} {g.lo.shape}")
    if not meta["use_relative_action"] or meta["use_percentiles"] or not meta["clip_outliers"] or meta["sin_cos_keys"]:
        raise AssertionError(f"processor flags changed: {meta}")


def install(hub: Optional[Path] = None) -> Dict[str, Any]:
    """Point the port's N1.6 registries at the BF16 G1 copy (idempotent, process-local; module docstring)."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return INSTALL_FACTS
        hub = Path(hub).expanduser() if hub is not None else hub_cache()
        snap = snapshot_path(hub)
        if not (snap / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"BF16 G1 snapshot missing: {snap} (run `ttpy -m rttt.convert_bf16`)")
        prev_hub = os.environ.get("HF_HUB_CACHE")
        os.environ["HF_HUB_CACHE"] = str(hub)
        from models.experimental.gr00t.common import configs as C
        from models.experimental.gr00t.common import normalization as N

        ck_facts = check_checkpoint(snap)
        C.HF_REPOS["n16"] = REPO  # type: ignore[index]
        C.HF_SNAPSHOT_SHAS["n16"] = SHA  # type: ignore[index]
        if C.snapshot_dir("n16") != snap:
            raise AssertionError(f"snapshot_dir(n16) -> {C.snapshot_dir('n16')} != {snap}")

        base = C._CONFIGS["n16"]
        if any(l.name == LAYOUT_NAME for l in base.layouts):
            raise AssertionError("n16 already has a 'g1' layout")
        ck = dataclasses.replace(base.checkpoint, hf_repo=REPO, hf_sha=SHA,
                                 straddling_prefix=ck_facts["straddling"] or base.checkpoint.straddling_prefix)
        cfg = dataclasses.replace(base, checkpoint=ck, layouts=tuple(base.layouts) + (g1_layout(base.layout(BASE_LAYOUT)),))
        C._CONFIGS["n16"] = cfg  # type: ignore[index]
        if C.get_config("n16").embodiment_id(EMBODIMENT_TAG) != EMBODIMENT_ID:
            raise AssertionError("unitree_g1 is not slot 8")

        for fn in (N.load_processor_kwargs, N.load_statistics, N.load_embodiment_id_json, N.get_embodiment_spec):
            fn.cache_clear()
        meta = checkpoint_metadata(snap)
        spec = N.get_embodiment_spec(VERSION, EMBODIMENT_TAG)
        assert_spec(spec, meta)
        INSTALL_FACTS.update(
            snapshot_dir=str(snap),
            repo=REPO,
            sha=SHA,
            source=f"{SRC_REPO}@{SRC_SHA}",
            hub=str(hub),
            hf_hub_cache_before=prev_hub,
            checkpoint=ck_facts,
            layout=LAYOUT_NAME,
            s_pad=S_PAD,
            l_max=L_MAX,
            embodiment=f"{EMBODIMENT_TAG} ({EMBODIMENT_ID})",
            added_table_rows={"sdpa": [], "matmul": []},
        )
        _INSTALLED = True
        return INSTALL_FACTS


def layer_table_rows() -> Dict[str, List[Tuple]]:
    """The tt.layers rows the gr1 / g1 S_pad-128 geometry looks up, each resolved in MK's tables (``KeyError`` if one
    is missing). Shapes are gr1's: SigLIP 352 patch rows, projector 81 tokens, LLM S_pad 128, cross K/V hoist over
    the text (64) and image (96) key sets, DiT 51 -> 64 rows; the enumeration mirrors MK
    ``tests/tt/test_layers_cpu.py:58-121`` restricted to n16 / gr1."""
    from models.experimental.gr00t.tt import layers as TL

    both = ("bf16", "bfp8_b")
    mm = []
    for m in (51, 64):
        mm += [(m, 1536, 6144, d) for d in both] + [(m, 1536, 4608, d) for d in both]
        mm += [(m, 1536, 2048, d) for d in both] + [(m, 2048, 1536, d) for d in both]
        mm += [(m, 1536, 1536, d) for d in both] + [(m, 6144, 1536, d) for d in both]
        mm += [(m, 1536, 1024, "bf16"), (m, 1024, 1024, "bf16")]
    mm += [(64, 1024, 128, "bf16"), (64, 128, 1536, "bf16"), (64, 1536, 1536, "bf16"),
           (1, 128, 1024, "bf16"), (1, 1024, 1536, "bf16")]
    mm += [(352, k, n, d) for (k, n) in ((608, 1152), (1152, 4608), (1536, 1152), (1152, 4304), (4304, 1152))
           for d in both]
    mm += [(81, 4608, 2048, d) for d in both] + [(81, 2048, 2048, d) for d in both]
    mm += [(128, k, n, d) for (k, n) in ((2048, 4096), (2048, 2048), (2048, 12288), (6144, 2048)) for d in both]
    mm += [(l, 2048, 4096, d) for l in (64, 96) for d in both]
    for row in mm:
        TL.matmul_spec(*row)
    sdpa = [(128, 128, 128), (64, 64, 64), (64, 96, 64), (352, 352, 96)]  # llm, dit self/text, dit image, siglip
    for row in sdpa:
        if not TL.sdpa_chunks_allowed(*row):
            raise KeyError(f"no SDPA chunks for {row}")
    return {"matmul": mm, "sdpa": sdpa, "forbidden_sdpa": sorted(TL.FORBIDDEN_SDPA_CHUNKS)}


# --------------------------------------------------------------------------------------------------------------------
# host pipeline: W-VLA flat obs -> ModelInputs ; normalised chunk -> action dict
# --------------------------------------------------------------------------------------------------------------------
def filter_obs(flat: Mapping[str, Any]) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    """Keep only the 9 modality keys (before ``split_batch``: the stock rollout's 13 extra keys include two f32
    480x640 camera copies that ``split_batch`` would turn into Python lists, ``TT/libero/n16.py:259-266``)."""
    if not isinstance(flat, Mapping):
        raise TypeError(f"observation must be a dict, got {type(flat).__name__}")
    missing = [k for k in MODALITY_KEYS if k not in flat]
    if missing:
        raise KeyError(f"observation lacks {missing}")
    kept = {k: flat[k] for k in MODALITY_KEYS}
    return kept, tuple(sorted(k for k in flat if k not in kept))


def check_obs(flat: Mapping[str, Any]) -> int:
    """The checks ``Gr00tSimPolicyWrapper.check_observation`` (strict) applies (``R/gr00t/policy/gr00t_policy.py:
    452-560``), on the filtered keys; returns the batch size."""
    v = flat[schema.VIDEO_KEY]
    if not isinstance(v, np.ndarray) or v.dtype != np.uint8 or v.ndim != 5 or v.shape[1] != 1 or v.shape[-1] != 3:
        raise ValueError(f"{schema.VIDEO_KEY} must be uint8 (B, 1, H, W, 3), got "
                         f"{getattr(v, 'dtype', type(v).__name__)} {getattr(v, 'shape', '')}")
    b = int(v.shape[0])
    for (name, d), key in zip(schema.STATE_GROUPS, schema.STATE_KEYS):
        a = flat[key]
        if not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.shape != (b, 1, d):
            raise ValueError(f"{key} must be float32 ({b}, 1, {d}), got "
                             f"{getattr(a, 'dtype', type(a).__name__)} {getattr(a, 'shape', '')}")
    lang = flat[LANG_KEY]
    if isinstance(lang, np.ndarray):
        lang = lang.reshape(-1).tolist()
    if not isinstance(lang, (list, tuple)) or len(lang) != b or not all(isinstance(s, str) for s in lang):
        raise ValueError(f"{LANG_KEY} must be a tuple of {b} str, got {lang!r}")
    return b


def instruction_of(lang: Any) -> str:
    if isinstance(lang, np.ndarray):
        lang = lang.reshape(-1).tolist()
    if isinstance(lang, (list, tuple)):
        if len(lang) != 1:
            raise ValueError(f"expected one instruction, got {len(lang)}")
        lang = lang[0]
    if isinstance(lang, (list, tuple)):  # nested [[str]]
        lang = lang[0]
    if isinstance(lang, bytes):
        lang = lang.decode()
    return str(lang)


def split_batch(flat: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Batched flat obs (video (B,T,H,W,3), state (B,T,D), language (B,)) -> B single-element dicts."""
    b = None
    for k, v in flat.items():
        if k.startswith("video.") or k.startswith("state."):
            n = int(np.asarray(v).shape[0])
            if b is None:
                b = n
            elif b != n:
                raise ValueError(f"inconsistent batch sizes ({k}: {n} vs {b})")
    if b is None:
        raise ValueError("observation has no video./state. keys")
    out = []
    for j in range(b):
        d: Dict[str, Any] = {}
        for k, v in flat.items():
            if k.startswith("video.") or k.startswith("state."):
                d[k] = np.asarray(v)[j]
            else:
                seq = v.reshape(-1).tolist() if isinstance(v, np.ndarray) else list(v)
                d[k] = seq[j] if len(seq) == b else seq[0]
        out.append(d)
    return out


def to_observation(one: Mapping[str, Any]):
    """One un-batched element (video (1,H,W,3) uint8, state (1,D) float32, language str) -> ``Observation``."""
    from models.experimental.gr00t.common.preprocessing import Observation

    images = {}
    for k in VIDEO_KEYS:
        v = np.asarray(one[f"video.{k}"])
        if v.ndim == 5 and v.shape[0] == 1:
            v = v[0]
        if v.ndim == 3:
            v = v[None]
        if v.dtype != np.uint8 or v.ndim != 4 or v.shape[0] != 1 or v.shape[-1] != 3:
            raise ValueError(f"video.{k}: expected (1, H, W, 3) uint8, got {v.shape} {v.dtype}")
        images[k] = np.ascontiguousarray(v)
    state = {}
    for k in STATE_KEYS:
        v = np.asarray(one[f"state.{k}"])
        if v.ndim == 3 and v.shape[0] == 1:
            v = v[0]
        if v.ndim == 1:
            v = v[None]
        if v.ndim != 2 or v.shape[0] != 1 or v.dtype.kind != "f":
            raise ValueError(f"state.{k}: expected (1, D) float, got {v.shape} {v.dtype}")
        state[k] = np.ascontiguousarray(v)  # dtype as sent (float32 on the wire)
    return Observation(images=images, state=state, instruction=instruction_of(one[LANG_KEY]),
                       embodiment_tag=EMBODIMENT_TAG)


def encode_obs(one: Mapping[str, Any]):
    """One un-batched element -> ``(ModelInputs for layout 'g1', Observation)``."""
    install()
    from models.experimental.gr00t.common.preprocessing import encode

    obs = to_observation(one)
    return encode(VERSION, obs, embodiment=EMBODIMENT_TAG, layout_name=LAYOUT_NAME), obs


def decode_actions(pred: torch.Tensor, obs: Any) -> Dict[str, np.ndarray]:
    """``(1, 50, 128)`` normalised chunk -> ``{"action.<key>": (1, 30, D) float32}`` in A35 group order (official
    ``decode_action``: first 30 rows, clip + min-max inverse in float64, arms + state[-1], cast to fp32)."""
    from models.experimental.gr00t.common.normalization import decode

    acts = decode(VERSION, pred.detach().to("cpu", torch.float32), obs, embodiment=EMBODIMENT_TAG)
    return {f"action.{k}": np.ascontiguousarray(acts[k].numpy(), dtype=np.float32) for k in ACTION_KEYS}


def stack_actions(per: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Per-element ``(1, 30, D)`` dicts -> the batched reply ``(B, 30, D)`` float32 (Gr00tSimPolicyWrapper shape)."""
    return {k: np.concatenate([p[k] for p in per], 0).astype(np.float32, copy=False) for k in per[0]}


# --------------------------------------------------------------------------------------------------------------------
# model loading and stamps
# --------------------------------------------------------------------------------------------------------------------
RUNTIME_ROOT_ENV = "TT_METAL_RUNTIME_ROOT"


def runtime_root() -> Path:
    """The tree whose ``models.experimental.gr00t`` is imported (MK: ``publish/worktree-mk``), as imported, symlinks
    not resolved: the root tt-metal must compile this tree's device kernels from."""
    import models.experimental.gr00t as pkg_mod

    return Path(os.path.abspath(pkg_mod.__file__)).parent.parents[2]


def pin_runtime_root() -> Dict[str, Any]:
    """Set ``TT_METAL_RUNTIME_ROOT`` to :func:`runtime_root` before ttnn is imported; refuse a different root.

    tt-metal compiles a kernel from a path relative to its root dir (MK ``tt/megakernel/descriptors.py:60``,
    ``KERNEL_SOURCE = "models/experimental/gr00t/tt/megakernel/kernels/dit_kernel.cpp"``). The root is
    ``TT_METAL_RUNTIME_ROOT`` or, without it, what ``ttnn/__init__.py:654-664`` derives from
    ``Path(ttnn.__file__).resolve()``; TT_METAL_HOME is not read (``tt_metal/llrt/rtoptions.cpp:330-345``). MK's
    ``ttnn`` is a symlink into the live clone, so the root became the live tree and S1's first cold build
    (2026-10-25) compiled the live, Stage-3 ``dit_kernel.cpp`` against MK's host descriptors: named compile-time args
    ``k4_brisc_gather`` / ``cb_mbox`` missing, build failed. Pinned, the kernels are MK's own, so ``info.tree`` is
    true of the device code too; ``tt_metal``, ``ttnn`` and ``runtime`` under MK are links to the same built clone."""
    root = runtime_root()
    for sub in ("tt_metal", "ttnn", "runtime"):
        if not (root / sub).exists():
            raise FileNotFoundError(f"{root}/{sub} missing: {root} cannot serve as tt-metal's runtime root")
    cur = os.environ.get(RUNTIME_ROOT_ENV)
    if cur is not None and os.path.normpath(cur) != str(root):
        raise RuntimeError(f"{RUNTIME_ROOT_ENV}={cur}, but the imported tree is {root}: kernels would come from "
                           "another tree than the Python")
    if cur is None and "ttnn" in sys.modules:
        raise RuntimeError(f"ttnn was imported before {RUNTIME_ROOT_ENV} was set: its root is not {root}")
    os.environ[RUNTIME_ROOT_ENV] = str(root)
    return {"runtime_root": str(root), "preset": cur is not None}


def default_policy():
    """MK's shipped ``TTPolicy()`` (megakernel DiT, bfp8_b arena, ttnn backbone, mixed_dit)."""
    from models.experimental.gr00t.tt.policy import TTPolicy

    return TTPolicy()


def load_tt_model(policy: Any = None, device: Any = None, cache_root: Optional[Path] = None, **kw):
    """``Gr00tTT.from_pretrained("n16", "unitree_g1", "g1", ...)`` on the BF16 copy (install() first). With
    ``device=None`` this opens the device (2 CQs, the megakernel worker-L1 cut): call it under ``$DEV`` only."""
    install()
    pin_runtime_root()  # before the import below pulls in ttnn
    from models.experimental.gr00t.tt.model import Gr00tTT

    root = Path(cache_root) if cache_root is not None else default_cache_root()
    kw.setdefault("num_command_queues", 2)
    return Gr00tTT.from_pretrained(
        VERSION,
        embodiment=EMBODIMENT_TAG,
        layout_name=LAYOUT_NAME,
        policy=policy if policy is not None else default_policy(),
        device=device,
        cache_root=root,
        **kw,
    )


def backend_stamp(model: Any) -> Dict[str, Any]:
    """What was actually constructed, read from fields that exist on MK's ``TTPolicy`` / ``Gr00tTT`` /
    ``ActionHeadTT`` (MK has no ``backbone_backend`` / ``fused_stages``: ``tt/policy.py:170-182``)."""
    pol = getattr(model, "policy", None)
    head = getattr(model, "head", None)
    stamp: Dict[str, Any] = {
        "policy": pol.to_dict() if hasattr(pol, "to_dict") else None,
        "dit_backend": getattr(pol, "dit_backend", None),
        "dtype_policy": getattr(pol, "dtype_policy", None),
        "trace_layout": getattr(pol, "trace_layout", None),
        "rope_layout": getattr(pol, "rope_layout", None),
        "mk_arena_dtype": getattr(head, "mk_arena_dtype", None),  # resolved at build (action_head.py:449)
        "megakernel_built": getattr(head, "arena", None) is not None,
        "head_class": type(head).__name__ if head is not None else None,
        "embed_gather": getattr(model, "embed_gather", None),
        "adapter_mem": getattr(model, "adapter_mem", None),
        "layout": getattr(model, "layout_name", None),
        "s_pad": int(getattr(getattr(model, "dlayout", None), "s_pad", 0) or 0),
        "embodiment_id": getattr(model, "embodiment_id", None),
        "checkpoint": f"{REPO}@{SHA[:8]}",
        "source_checkpoint": f"{SRC_REPO}@{SRC_SHA[:8]}",
        "checkpoint_dir": str(getattr(getattr(model, "ck", None), "dir", "")),
        "weights_skipped": {k: v for k, v in (getattr(model, "weights_skipped", None) or {}).items()
                            if k in ("n_tensors", "n_kept", "n_full", "device_bytes")},
        "denoising_steps": DENOISING_STEPS,
    }
    stamp["label"] = (f"tt-p150-{stamp['dit_backend']}-dit-{stamp['mk_arena_dtype'] or 'na'}-{stamp['dtype_policy']}"
                      f"-{stamp['trace_layout']}-denoise{DENOISING_STEPS}")
    return stamp


def megakernel_weight_counts(policy: Any = None) -> Tuple[int, int]:
    """(kept, skipped) plan tensors under the megakernel backend for layout g1 (CPU only; WP-2.1 expects 503 / 534):
    the arena replaces the DiT / precompute / encoder tensors except those ``TTWeights.constants`` reads
    (``tt/model.py:472-506``)."""
    install()
    from models.experimental.gr00t.tt import model as M

    names, facts = M.megakernel_weight_names(VERSION, EMBODIMENT_ID, policy or default_policy(), LAYOUT_NAME)
    return len(names), int(facts["n_tensors"])


def make_noise(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((1, MODEL_HORIZON, ACTION_DIM_PAD), generator=g, dtype=torch.float32)


def modality_config_json() -> Dict[str, Any]:
    """``processor_config.json`` modality config of unitree_g1 (the fields of the official ``ModalityConfig``)."""
    from models.experimental.gr00t.common.normalization import load_processor_kwargs

    install()
    return dict(load_processor_kwargs(VERSION)["modality_configs"][EMBODIMENT_TAG])


_TREE_SUFFIXES = (".py", ".cpp", ".hpp", ".h")


def _git(root: Path, *args: str, stdin: Optional[str] = None) -> Optional[str]:
    try:
        p = subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def tree_files(pkg: Path) -> List[Path]:
    return sorted(p for sub in ("common", "tt") for p in (pkg / sub).rglob("*")
                  if p.is_file() and p.suffix in _TREE_SUFFIXES and "__pycache__" not in p.parts)


def content_hash(pkg: Path) -> str:
    """sha256 over (relative path, contents) of ``{common,tt}/**/*.{py,cpp,hpp,h}``, sorted: the live tree is
    untracked, so a git hash cannot identify it (PLAN.md §1.4.1)."""
    h = hashlib.sha256()
    for p in tree_files(pkg):
        h.update(str(p.relative_to(pkg)).encode() + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def tree_stamp() -> str:
    """``MK@<HEAD12>`` when the imported ``models.experimental.gr00t`` is a git tree whose ``{common,tt}`` files all
    equal their HEAD blobs (MK is an archive extract with an empty index, so ``git status`` cannot be used; blob
    hashes are compared instead); else ``live@<content_hash[:12]>``; a modified MK reads ``MK@<HEAD12>+dirty@...``."""
    import models.experimental.gr00t as pkg_mod

    pkg = Path(pkg_mod.__file__).resolve().parent
    root = pkg.parents[2]
    rel = pkg.relative_to(root)
    head = (_git(root, "rev-parse", "HEAD") or "").strip()
    listing = _git(root, "ls-tree", "-r", "HEAD", "--", f"{rel}/common", f"{rel}/tt") if head else None
    blobs = {}
    for line in (listing or "").splitlines():
        meta, path = line.split("\t", 1)
        if Path(path).suffix in _TREE_SUFFIXES:
            blobs[root / path] = meta.split()[2]
    if not blobs:
        return f"live@{content_hash(pkg)[:12]}"
    files = tree_files(pkg)
    got = _git(root, "hash-object", "--stdin-paths", stdin="\n".join(str(p) for p in files) + "\n")
    hashes = dict(zip(files, (got or "").split()))
    if set(files) == set(blobs) and all(hashes.get(p) == b for p, b in blobs.items()):
        return f"MK@{head[:12]}"
    return f"MK@{head[:12]}+dirty@{content_hash(pkg)[:12]}"
