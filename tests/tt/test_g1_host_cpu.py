"""T-2.1 (CPU half): the G1 port glue of rttt.g1_n16 and the BF16 checkpoint copy, without ttnn or the device.

    ttpy -m pytest $RT/tests/tt/test_g1_host_cpu.py -q

Covers PLAN.md WP-2.1 / WP-2.1b: the BF16 copy (1106 BF16 tensors, 6,573,217,664 B, the source's 2-shard map), the
spec asserts (slot 8, ego_view, state 43, action 35, horizon 30, arms RELATIVE) against the checkpoint's own files,
the g1 layout = gr1 geometry (assert_call, per-call input specs, MK's layer tables already hold its rows), the
megakernel weight split (503 kept / 534 skipped), the dedicated cache root, the backend stamp fields, and host
encode / decode against ``isaac_ref.npz`` (max|d| = 0; skipped until ``rttt/make_isaac_ref.py`` has written it).
"""

import dataclasses
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from rtloop import schema
from rttt import convert_bf16 as CB
from rttt import golden_io as gio

# LayoutConfig fields g1 may differ from gr1 in: identity and golden metadata only (U:tt "layout gap")
G1_METADATA_FIELDS = {"name", "embodiment_tag", "embodiment_id", "valid_slice", "action_groups", "s_golden",
                      "s_typical", "notes"}


# --- the BF16 copy (WP-2.1b) -----------------------------------------------------------------------------------------


def test_bf16_snapshot(g1):
    from models.experimental.gr00t.common.checkpoint import LazyCheckpoint

    snap = g1.snapshot_path()
    assert snap.is_dir() and snap.name == CB.BF16_SHA == g1.SHA
    assert Path.home() / ".cache/huggingface" not in snap.resolve().parents
    ck = LazyCheckpoint(snap)
    assert len(ck) == 1106
    assert {ck.info(k).dtype for k in ck.keys()} == {"BF16"}
    assert ck.total_bytes == 6_573_217_664
    assert json.loads((snap / "model.safetensors.index.json").read_text())["metadata"]["total_size"] == 6_573_217_664
    assert len(ck.verify_headers()) == 2
    by_version = LazyCheckpoint("n16")  # what every port loader opens after install()
    assert by_version.dir == snap
    assert by_version.expected_counts() == (1106, 2, 6_573_217_664)


def test_bf16_copy_keeps_source_map_and_files(g1):
    src = CB.DEFAULT_SRC
    if not (src / CB.INDEX).is_file():
        pytest.skip(f"source snapshot {src} not present")
    snap = g1.snapshot_path()
    si = json.loads((src / CB.INDEX).read_text())
    di = json.loads((snap / CB.INDEX).read_text())
    assert di["weight_map"] == si["weight_map"]  # 2-shard map kept
    for f in CB.NON_WEIGHT_FILES:
        assert (snap / f).read_bytes() == (src / f).read_bytes(), f
    rec = json.loads((CB.repo_dir(g1.hub_cache()) / "conversion.json").read_text())
    assert rec["checks"]["values_equal_src_to_bf16"] == 1106 and rec["checks"]["all_bf16"]
    assert rec["sha256"]["model-00001-of-00002.safetensors"] == CB.sha256_file(snap / "model-00001-of-00002.safetensors")


def test_synthetic_sha_is_stable():
    assert CB.synthetic_sha() == CB.BF16_SHA and len(CB.BF16_SHA) == 40
    assert CB.synthetic_sha(src_sha="0" * 40) != CB.BF16_SHA


# --- install and spec (WP-2.1) ---------------------------------------------------------------------------------------


def test_install_repoints_registries(g1):
    from models.experimental.gr00t.common import configs as C

    f = g1.install()
    assert C.HF_REPOS["n16"] == "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate-bf16"
    assert C.HF_SNAPSHOT_SHAS["n16"] == CB.BF16_SHA
    assert C.snapshot_dir("n16") == g1.snapshot_path()
    cfg = C.get_config("n16")
    assert cfg.checkpoint.hf_repo == g1.REPO and cfg.checkpoint.hf_sha == g1.SHA
    assert [l.name for l in cfg.layouts].count("g1") == 1
    assert cfg.embodiment_id("unitree_g1") == 8
    assert f["added_table_rows"] == {"sdpa": [], "matmul": []}  # no _extend_layer_tables
    assert g1.install() is f  # idempotent


def test_spec(g1):
    from models.experimental.gr00t.common import normalization as N

    spec = N.get_embodiment_spec("n16", "unitree_g1")
    assert spec.embodiment_id == 8
    assert spec.video_keys == ("ego_view",) and spec.frames_per_camera == 1
    assert spec.state_keys == tuple(n for n, _ in schema.STATE_GROUPS) and spec.state_dim == 43
    assert spec.action_keys == tuple(n for n, _ in schema.ACTION_GROUPS) and spec.action_dim == 35
    assert spec.action_horizon == 30
    reps = {g.key: g.rep for g in spec.action_groups}
    assert reps == {"left_arm": "RELATIVE", "right_arm": "RELATIVE", "left_hand": "ABSOLUTE",
                    "right_hand": "ABSOLUTE", "waist": "ABSOLUTE", "base_height_command": "ABSOLUTE",
                    "navigate_command": "ABSOLUTE"}
    for g in spec.action_groups:
        if g.rep == "RELATIVE":
            assert g.stats_source == "relative_action.min/max" and g.lo.shape == (30, g.dim)
    assert all(g.mode == "min_max" and g.clip for g in spec.state_groups)
    g1.assert_spec(spec, g1.checkpoint_metadata(g1.snapshot_path()))


def test_spec_from_checkpoint_files_alone(g1):
    """The same facts read straight from processor_config.json / statistics.json, no port parser involved."""
    m = g1.checkpoint_metadata(g1.snapshot_path())
    assert m["embodiment_id"] == 8 and m["video_keys"] == ("ego_view",)
    assert tuple(m["state_dims"].items()) == schema.STATE_GROUPS
    assert tuple(m["action_dims"].items()) == schema.ACTION_GROUPS
    assert m["horizon"] == 30 and m["language_keys"] == (schema.LANGUAGE_KEY,)
    assert m["reps"]["left_arm"] == m["reps"]["right_arm"] == "RELATIVE"
    assert m["relative_shapes"] == {"left_arm": (30, 7), "right_arm": (30, 7)}
    assert m["num_inference_timesteps"] == 4


def test_normalization_caches_follow_the_copy(g1):
    from models.experimental.gr00t.common import normalization as N

    stats = N.load_statistics("n16")
    raw = json.loads((g1.snapshot_path() / "statistics.json").read_text())
    assert "unitree_g1" in stats and stats["unitree_g1"]["state"]["waist"] == raw["unitree_g1"]["state"]["waist"]


# --- layout g1 = gr1 geometry ----------------------------------------------------------------------------------------


def test_g1_layout_is_gr1_geometry(g1):
    from models.experimental.gr00t.common import configs as C

    cfg = C.get_config("n16")
    gr1, lay = cfg.layout("gr1"), cfg.layout("g1")
    for f in dataclasses.fields(lay):
        if f.name not in G1_METADATA_FIELDS:
            assert getattr(lay, f.name) == getattr(gr1, f.name), f.name
    assert (lay.embodiment_tag, lay.embodiment_id, lay.valid_slice) == ("unitree_g1", 8, (30, 35))
    assert lay.s_pad_llm == 128 and lay.l_max == 24 and lay.tokens_per_image == 81


def test_assert_call_ok_for_g1(g1, obs_factory):
    from models.experimental.gr00t.common import configs as C
    from models.experimental.gr00t.tt import model as M
    from models.experimental.gr00t.tt.layout import DeviceLayout, StaticShapeError
    from models.experimental.gr00t.tt.policy import TTPolicy

    cfg = C.get_config("n16")
    dl = DeviceLayout.from_layout(cfg, cfg.layout("g1"))
    dl_gr1 = DeviceLayout.from_layout(cfg, cfg.layout("gr1"))
    assert dl.s_pad == 128 and tuple(dl.noise_shape) == (1, 50, 128) and dl.embodiment_id == 8
    kept, _ = g1.filter_obs(obs_factory(1))
    mi, _ = g1.encode_obs(g1.split_batch(kept)[0])
    assert (mi.seq_len, mi.n_text, int(mi.embodiment_id[0])) == (117, 13, 8)
    dl.assert_call(mi, g1.make_noise(0), loaded_embodiment_ids=(8,))
    with pytest.raises(StaticShapeError):
        dl.assert_call(mi, g1.make_noise(0), loaded_embodiment_ids=(20,))
    pol = TTPolicy()
    for gather in ("device", "host"):
        a = {n: (tuple(s.shape), s.dtype, s.per_call) for n, s in M.model_input_specs(cfg, dl, pol, gather).items()}
        b = {n: (tuple(s.shape), s.dtype, s.per_call) for n, s in M.model_input_specs(cfg, dl_gr1, pol, gather).items()}
        assert a == b, gather


def test_mk_layer_tables_hold_the_s_pad_128_rows(g1):
    from models.experimental.gr00t.tt import layers as TL

    rows = g1.layer_table_rows()  # raises KeyError on a missing row
    assert (128, 2048, 12288, "bf16") in rows["matmul"] and (352, 1152, 4608, "bfp8_b") in rows["matmul"]
    for key in (("llm_causal", 128, 128, 128), ("dit_self", 64, 64, 64), ("dit_cross_text", 64, 64, 64),
                ("dit_cross_image_n16", 64, 96, 64), ("vit_siglip", 352, 352, 96)):
        assert key in TL.SDPA_CHUNKS, key
    assert TL.FORBIDDEN_SDPA_CHUNKS == frozenset({(64, 320), (64, 448)})
    for key in ((128, 2048, 4096, "bf16"), (128, 2048, 2048, "bf16"), (128, 6144, 2048, "bf16"),
                (128, 4608, 2048, "bf16")):
        assert key in TL.MATMUL_CONFIGS, key


def test_megakernel_weight_split_503_534(g1):
    assert g1.megakernel_weight_counts() == (503, 534)


# --- cache root and stamps -------------------------------------------------------------------------------------------


def test_dedicated_cache_root(g1, monkeypatch):
    from models.experimental.gr00t.common.weights import HostPlanCache

    monkeypatch.delenv(g1.CACHE_ENV, raising=False)
    root = g1.default_cache_root()
    assert root == Path.home() / ".cache/gr00t-tt-g1" / f"cloudwalk-research--GR00T-N1.6-G1-PnPAppleToPlate-bf16@{CB.BF16_SHA[:8]}"
    d = HostPlanCache(root).directory("n16", 8, "mixed_dit")
    assert root in d.parents and f"/n16/{CB.BF16_SHA[:12]}/emb8/" in str(d)
    assert "d0814e7ecb19" not in str(d)


def test_backend_stamp_uses_mk_fields(g1):
    from models.experimental.gr00t.tt.policy import TTPolicy

    names = {f.name for f in dataclasses.fields(TTPolicy)}
    assert {"dit_backend", "dtype_policy", "trace_layout", "rope_layout", "mk_arena_dtype"} <= names
    assert "backbone_backend" not in names  # MK has none (the live tree does)

    class Stub:  # the attributes Gr00tTT / ActionHeadTT expose after a build
        policy = TTPolicy()
        head = type("Head", (), {"mk_arena_dtype": "bfp8_b", "arena": object()})()
        embed_gather, adapter_mem, layout_name, embodiment_id = "device", "DRAM", "g1", 8
        weights_skipped = {"n_tensors": 534, "n_kept": 503, "n_full": 1037, "device_bytes": 1.0, "categories": {}}

    s = g1.backend_stamp(Stub())
    assert s["dit_backend"] == "megakernel" and s["mk_arena_dtype"] == "bfp8_b" and s["megakernel_built"]
    assert s["checkpoint"] == f"{g1.REPO}@{g1.SHA[:8]}" and s["layout"] == "g1"
    assert s["weights_skipped"]["n_kept"] == 503 and "categories" not in s["weights_skipped"]
    assert s["label"].startswith("tt-p150-megakernel-dit-bfp8_b-mixed_dit")
    json.dumps(s)


def test_tree_stamp_is_mk(g1):
    import models.experimental.gr00t as pkg

    assert "worktree-mk" in pkg.__file__
    assert g1.tree_stamp() == "MK@009de892c895"


# --- the tt-metal runtime root (S1: the device kernels came from the live tree; verify:all D7) -----------------------

MK_TREE = Path.home() / "experiments" / "gr00t" / "publish" / "worktree-mk"


@pytest.fixture
def no_root(monkeypatch):
    """TT_METAL_RUNTIME_ROOT unset for the test and restored after it (pin_runtime_root writes os.environ)."""
    monkeypatch.delenv("TT_METAL_RUNTIME_ROOT", raising=False)
    return monkeypatch


def test_runtime_root_is_the_mk_tree(g1, no_root):
    """The root is the tree the Python is imported from, as imported: MK's ttnn is a symlink into the live clone,
    and the root ttnn derives from its resolved path is the live tree (S1's failed cold build)."""
    assert g1.runtime_root() == MK_TREE == Path(os.environ["TT_METAL_HOME"])
    assert (MK_TREE / "ttnn").is_symlink() and (MK_TREE / "ttnn").resolve().parent != MK_TREE
    assert g1.pin_runtime_root() == {"runtime_root": str(MK_TREE), "preset": False}
    assert os.environ["TT_METAL_RUNTIME_ROOT"] == str(MK_TREE)
    assert "ttnn" not in sys.modules  # pinning never imports ttnn (the conftest blocks it)


def test_runtime_root_refuses_another_preset_root(g1, no_root):
    live = str(Path.home() / "experiments" / "gr00t" / "tt-metal")
    no_root.setenv("TT_METAL_RUNTIME_ROOT", live)
    with pytest.raises(RuntimeError, match="kernels would come from another tree"):
        g1.pin_runtime_root()
    assert os.environ["TT_METAL_RUNTIME_ROOT"] == live  # left as it was


@pytest.mark.parametrize("suffix", ["", "/"])  # "": what env.sh's ttpy now presets ($MK)
def test_runtime_root_accepts_the_mk_tree_preset(g1, no_root, suffix):
    no_root.setenv("TT_METAL_RUNTIME_ROOT", str(MK_TREE) + suffix)
    assert g1.pin_runtime_root() == {"runtime_root": str(MK_TREE), "preset": True}
    assert os.environ["TT_METAL_RUNTIME_ROOT"] == str(MK_TREE)


def test_runtime_root_refuses_after_ttnn_was_imported(g1, no_root):
    """A ttnn imported before the pin has derived its root already. The real import stays blocked (conftest), so a
    stand-in module plays the imported ttnn."""
    no_root.setitem(sys.modules, "ttnn", types.ModuleType("ttnn"))
    with pytest.raises(RuntimeError, match="ttnn was imported before TT_METAL_RUNTIME_ROOT"):
        g1.pin_runtime_root()
    assert "TT_METAL_RUNTIME_ROOT" not in os.environ


# --- host pipeline ---------------------------------------------------------------------------------------------------


def test_filter_and_check(g1, obs_factory):
    obs = obs_factory(2)
    obs["video.tpp_view"] = obs[schema.VIDEO_KEY].copy()
    obs["q"] = np.zeros((1, 1, 43), np.float32)
    kept, dropped = g1.filter_obs(obs)
    assert tuple(kept) == schema.VLA_OBS_KEYS and dropped == ("q", "video.tpp_view")
    assert g1.check_obs(kept) == 1
    bad = dict(kept, **{"state.waist": kept["state.waist"].astype(np.float64)})
    with pytest.raises(ValueError, match="float32"):
        g1.check_obs(bad)
    with pytest.raises(KeyError):
        g1.filter_obs({k: v for k, v in kept.items() if k != "state.left_hand"})


def test_decode_shapes_and_order(g1, obs_factory):
    kept, _ = g1.filter_obs(obs_factory(3))
    one = g1.split_batch(kept)[0]
    mi, obs = g1.encode_obs(one)
    acts = g1.decode_actions(torch.zeros(1, 50, 128), obs)
    assert tuple(acts) == tuple(f"action.{n}" for n, _ in schema.ACTION_GROUPS)
    for (n, d) in schema.ACTION_GROUPS:
        assert acts[f"action.{n}"].shape == (1, 30, d) and acts[f"action.{n}"].dtype == np.float32
    assert schema.actions_to_rows(acts).shape == (30, 35)


def test_arms_relative_rest_absolute(g1, obs_factory):
    """Same prediction, two states: only the arms move, by exactly the state difference (+ state[-1])."""
    pred = torch.empty(1, 50, 128).uniform_(-1, 1, generator=torch.Generator().manual_seed(0))
    out = []
    for seed in (4, 5):
        kept, _ = g1.filter_obs(obs_factory(seed))
        _, obs = g1.encode_obs(g1.split_batch(kept)[0])
        out.append((g1.decode_actions(pred, obs), obs))
    (a, oa), (b, ob) = out
    for n, _ in schema.ACTION_GROUPS:
        if n in ("left_arm", "right_arm"):
            d = b[f"action.{n}"].astype(np.float64) - a[f"action.{n}"].astype(np.float64)
            want = (ob.state[n][-1].astype(np.float64) - oa.state[n][-1].astype(np.float64))
            assert np.allclose(d, want[None, None, :], atol=1e-6), n
        else:
            assert np.array_equal(a[f"action.{n}"], b[f"action.{n}"]), n


def test_batch_split_and_stack(g1, obs_factory):
    obs = obs_factory(6, b=2)
    kept, _ = g1.filter_obs(obs)
    assert g1.check_obs(kept) == 2
    parts = g1.split_batch(kept)
    assert len(parts) == 2 and parts[1]["video.ego_view"].shape == (1, 480, 640, 3)
    per = [g1.decode_actions(torch.zeros(1, 50, 128), g1.encode_obs(p)[1]) for p in parts]
    st = g1.stack_actions(per)
    assert st["action.left_arm"].shape == (2, 30, 7)


# --- parity with the Isaac processor (isaac_ref.npz from rttt/make_isaac_ref.py) -------------------------------------


def _isaac_ref(golden_dir: Path):
    path = golden_dir / "isaac_ref.npz"
    if not path.is_file():
        pytest.skip(f"{path} not written yet (WP-2.2: srvpy -m rttt.make_isaac_ref after the goldens' obs exist)")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def test_host_encode_equals_isaac(g1, golden_dir):
    """input_ids, pixel_values, normalised state and embodiment_id bit-identical (max|d| = 0) on the 20 obs."""
    ref = _isaac_ref(golden_dir)
    sids = [str(s) for s in ref["sid"]]
    assert len(sids) == 20, sids
    for i, sid in enumerate(sids):
        obs = gio.load_obs(golden_dir / "obs" / f"{sid}.npz")
        kept, _ = g1.filter_obs(obs)
        mi, _ = g1.encode_obs(g1.split_batch(kept)[0])
        ids = mi.input_ids.numpy()
        assert ids.shape == ref["in.input_ids"][i].shape and np.array_equal(ids, ref["in.input_ids"][i]), sid
        pv, pv_ref = mi.pixel_values.numpy().reshape(-1), ref["in.pixel_values"][i].reshape(-1)
        assert pv.shape == pv_ref.shape and float(np.abs(pv - pv_ref).max()) == 0.0, sid
        st = mi.state.numpy()
        assert st.shape == ref["in.state"][i].shape and float(np.abs(st - ref["in.state"][i]).max()) == 0.0, sid
        assert int(mi.embodiment_id.reshape(-1)[0]) == int(ref["in.embodiment_id"][i].reshape(-1)[0]) == 8


def test_host_decode_equals_isaac(g1, golden_dir):
    """Decoded actions of the reference's fixed predictions bit-identical on the 20 obs (every pred set present)."""
    ref = _isaac_ref(golden_dir)
    sids = [str(s) for s in ref["sid"]]
    names = sorted(k[len("pred."):] for k in ref if k.startswith("pred."))
    assert "uniform" in names, names
    for i, sid in enumerate(sids):
        obs = gio.load_obs(golden_dir / "obs" / f"{sid}.npz")
        kept, _ = g1.filter_obs(obs)
        _, o = g1.encode_obs(g1.split_batch(kept)[0])
        for name in names:
            acts = g1.decode_actions(torch.from_numpy(np.ascontiguousarray(ref[f"pred.{name}"][i])), o)
            for k in gio.ACTION_KEYS:
                want = ref[f"dec.{name}.{k}"][i]
                assert acts[k].shape == want.shape and float(np.abs(acts[k] - want).max()) == 0.0, (sid, name, k)


@pytest.mark.skipif(os.environ.get("RTTT_CPU_REF") != "1",
                    reason="opt-in (RTTT_CPU_REF=1): the fp32 CPU reference needs ~13 GB RAM and 12 threads for ~20 s")
def test_cpu_reference_matches_g1_goldens(g1, golden_dir):
    """The port's fp32 CPU reference (MK reference/model.py) on the BF16 copy, layout g1, slot 8, the golden noise:
    the boot samples' action_pred_norm[0, :30, :35] vs the 5090 goldens. Measured 2026-10-24: PCC 0.99998 vs bf16,
    0.999999 vs fp32, decoded max|d| <= 0.0069 vs bf16. Checks weights + layout + host pipeline before the device."""
    if not (golden_dir / "samples").is_dir():
        pytest.skip(f"G1 goldens not written yet under {golden_dir}")
    from models.experimental.gr00t.reference.model import Gr00tReference
    from rttt import server_g1 as S

    torch.set_num_threads(12)
    ref = Gr00tReference("n16", embodiment="unitree_g1")
    for sid in S.boot_samples(golden_dir):
        s = gio.load_sample(golden_dir, sid)
        kept, _ = g1.filter_obs(s["obs"])
        mi, obs = g1.encode_obs(g1.split_batch(kept)[0])
        pred = ref.forward_from_inputs(mi, noise=torch.from_numpy(np.ascontiguousarray(s["noise"]))).action_pred_normalized
        p = pred.detach().float().numpy()
        assert gio.pcc(gio.valid(p), gio.valid(s["bf16.action_pred_norm"])) >= 0.9995, sid
        assert gio.pcc(gio.valid(p), gio.valid(s["fp32.action_pred_norm"])) >= 0.9999, sid
        acts = g1.decode_actions(pred, obs)
        for k in gio.ACTION_KEYS:
            assert gio.max_abs(acts[k], s[f"bf16.{k}"]) <= 0.02, (sid, k)


def test_no_ttnn():
    import sys

    assert "ttnn" not in sys.modules
