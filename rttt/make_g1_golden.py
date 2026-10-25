"""G1 goldens on the RTX 5090 (WP-2.2): bf16 and fp32 runs of the Isaac Gr00tPolicy with saved noise.

A variant of ``R/tt_golden/dump_golden_n16.py`` for the DP1 checkpoint and the 20 recorded G1
observations of ``select_g1_obs.py``. Per sample it saves the initial flow-matching noise, the
normalised prediction ``action_pred_norm`` (1, 50, 128) and the decoded W-VLA actions of two runs:

- ``bf16``, the deployed numerics: Gr00tPolicy exactly as gpu_vla_server builds it (weights cast to
  bf16 at load, bf16 inputs, flash_attention_2, default SDPA backend). Each sample is checked
  bit-identical against ``Gr00tSimPolicyWrapper.get_action`` with the same noise.
- ``fp32``, the cleaner reference: the checkpoint at its stored precision (the 517 F32 tensors kept,
  the BF16 ones upcast exactly; checked tensor by tensor against the safetensors), fp32 inputs, torch
  SDPA with the math backend (flash-attn has no fp32), as dump_golden_n16.py's ``fp32_math`` run.

The noise is drawn once per sample on the CPU (``torch.Generator().manual_seed(base + i)``, fp32)
and injected by replacing ``torch.randn`` for the one call in
``gr00t_n1d6.py:310-314``; the bf16 run therefore starts from bf16(noise), which is saved too.

    srvpy -m rttt.make_g1_golden [--golden ~/experiments/gr00t/golden/n16_g1]

Needs >= 20 GB free VRAM (PLAN.md R17); waits up to --wait-min for it. Writes samples/<sid>.npz and
manifest.json (checkpoint, dtypes, keys/shapes, noise seeds, sha256s, measured T-2.2 floors).
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

from rtloop import schema
from rttt import golden_io as gio

REPO = "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate"  # DP1
REVISION = "2a5b61cc7660bdef4cf17ad18dc5dd97f13e4ce5"
HF_HUB = Path.home() / ".cache/huggingface/hub"
NAV_GROUP = "navigate_command"
FLOOR_MIN = 0.02  # T-2.2: 0.02 rad, or 0.02 m/s / rad/s for nav
T22_EXCLUDED = ("left_hand", "base_height_command")  # T-2.2 per-group gate excludes these


def snapshot_dir(repo: str, revision: str) -> Path:
    d = HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    assert (d / "config.json").exists(), f"checkpoint snapshot missing: {d}"
    return d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def vram_free_gb() -> tuple[float, float, float]:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
    return used / 1024, total / 1024, (total - used) / 1024


def wait_for_vram(min_free_gb: float, wait_min: float) -> dict:
    """PLAN.md WP-2.2/R17: only start when >= min_free_gb is free; poll every 60 s."""
    t0 = time.monotonic()
    while True:
        used, total, free = vram_free_gb()
        print(f"[golden] VRAM used {used:.1f} / {total:.1f} GB, free {free:.1f} GB", flush=True)
        if free >= min_free_gb:
            return {"used_gb": round(used, 2), "total_gb": round(total, 2), "free_gb": round(free, 2),
                    "waited_s": round(time.monotonic() - t0, 1), "min_free_gb": min_free_gb}
        if time.monotonic() - t0 > wait_min * 60:
            raise SystemExit(f"[golden] < {min_free_gb} GB free VRAM after {wait_min} min; giving up")
        time.sleep(60)


def set_determinism(seed: int) -> None:
    """dump_golden_n16.py:46-55: same recipe, so a re-run reproduces the files bit for bit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_noise(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(gio.NOISE_SHAPE, generator=g, dtype=torch.float32)


class FixedNoise:
    """While active, torch.randn returns the saved noise in the dtype/device the model asks for."""

    def __init__(self, noise: torch.Tensor):
        self.noise, self.calls = noise, 0

    def __enter__(self):
        self._orig = torch.randn

        def fixed_randn(*size, **kw):
            if "size" in kw:
                size = tuple(kw.pop("size"))
            shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
            assert shape == tuple(self.noise.shape), f"unexpected randn shape {shape}"
            self.calls += 1
            return self.noise.to(device=kw.get("device", "cpu"), dtype=kw.get("dtype", torch.float32))

        torch.randn = fixed_randn
        return self

    def __exit__(self, *exc):
        torch.randn = self._orig
        return False


# --- attention for the fp32 run (dump_golden_n16.py:102-164) -----------------------------------


def _block_diag_mask(seq_len_list, device):
    n = int(sum(seq_len_list))
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    s = 0
    for L in seq_len_list:
        mask[s:s + L, s:s + L] = True
        s += L
    return mask


def make_packed_attention():
    """Torch SDPA honouring Siglip2's ``seq_len_list`` packing (the block-diagonal mask that
    flash_attn_varlen_func applies implicitly through cu_seqlens)."""
    import torch.nn.functional as F

    def packed_attn(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None,
                    is_causal=False, seq_len_list=None, **kwargs):
        assert attention_mask is None and seq_len_list is not None
        B, H, S, D = query.shape
        assert B == 1 and sum(seq_len_list) == S, (B, S, seq_len_list)
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        mask = _block_diag_mask(seq_len_list, query.device)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0,
                                             is_causal=False, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    return packed_attn


def use_sdpa_math(model) -> dict:
    """Switch Qwen3 + Siglip2 to torch SDPA after construction and leave only the math backend."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    eagle = model.backbone.model
    ALL_ATTENTION_FUNCTIONS["sdpa_packed"] = make_packed_attention()
    eagle.vision_model.config._attn_implementation = "sdpa_packed"
    eagle.vision_model.vision_model.config._attn_implementation = "sdpa_packed"
    for layer in eagle.vision_model.vision_model.encoder.layers:
        layer.self_attn.config._attn_implementation = "sdpa_packed"
    eagle.language_model.config._attn_implementation = "sdpa"
    eagle.language_model.model.config._attn_implementation = "sdpa"
    for layer in eagle.language_model.model.layers:
        layer.self_attn.config._attn_implementation = "sdpa"
    eagle.config._attn_implementation = "sdpa"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return {"llm_attn": "sdpa", "vision_attn": "sdpa_packed", "sdpa_backend": "math"}


def check_fp32_weights(model, snap: Path) -> dict:
    """Every model tensor equals its stored safetensors value (F32 kept, BF16 upcast exactly)."""
    from safetensors import safe_open

    sd = model.state_dict()
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    n_eq, n_missing, dtypes, bad = 0, 0, {}, []
    for shard in sorted(set(index.values())):
        with safe_open(str(snap / shard), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
                p = sd.get(key)
                if p is None:
                    n_missing += 1
                    continue
                if p.dtype == torch.float32 and torch.equal(p.cpu(), t.float()):
                    n_eq += 1
                else:
                    bad.append(key)
    unused = [k for k in sd if k not in index]
    res = {"stored_dtypes": dtypes, "equal": n_eq, "missing_in_model": n_missing,
           "mismatch": bad[:10], "n_mismatch": len(bad), "model_only_tensors": unused[:10],
           "n_model_only": len(unused)}
    assert not bad and n_missing == 0, res
    return res


# --- one inference ------------------------------------------------------------------------------


def nest(policy, obs: dict) -> dict:
    """Gr00tSimPolicyWrapper._get_action's flat -> nested mapping (gr00t_policy.py:586-608)."""
    mc = policy.modality_configs
    out = {"video": {}, "state": {}, "language": {}}
    for k in mc["video"].modality_keys:
        out["video"][k] = obs[f"video.{k}"]
    for k in mc["state"].modality_keys:
        out["state"][k] = obs[f"state.{k}"]
    for k in mc["language"].modality_keys:
        out["language"][k] = [[str(x)] for x in obs[k]]
    return out


def infer(policy, obs: dict, noise: torch.Tensor, dtype: torch.dtype) -> tuple[np.ndarray, dict]:
    """Gr00tPolicy._get_action (gr00t_policy.py:306-358) with the input dtype as a parameter.

    Returns action_pred_norm f32 (1, 50, 128) and the decoded {'action.<group>': f32 (1, 30, D)}.
    """
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    nested = nest(policy, obs)
    policy.check_observation(nested)
    states, processed = [], []
    for o in policy._unbatch_observation(nested):
        step = policy._to_vla_step_data(o)
        states.append(step.states)
        processed.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
    collated = _rec_to_dtype(policy.collate_fn(processed), dtype=dtype)
    with FixedNoise(noise) as fx, torch.inference_mode():
        pred = policy.model.get_action(**collated)["action_pred"]
    assert fx.calls == 1, f"torch.randn called {fx.calls} times"
    assert pred.dtype == dtype, (pred.dtype, dtype)
    pred = pred.float().cpu().numpy()
    batched = {k: np.stack([s[k] for s in states], axis=0) for k in policy.modality_configs["state"].modality_keys}
    acts = policy.processor.decode_action(pred, policy.embodiment_tag, batched)
    return pred, {f"action.{k}": np.asarray(v, np.float32) for k, v in acts.items()}


def official(wrapper, obs: dict, noise: torch.Tensor) -> tuple[np.ndarray, dict]:
    """The served path (Gr00tSimPolicyWrapper.get_action) with the noise injected and the
    normalised prediction captured on its way out of model.get_action."""
    model = wrapper.policy.model
    seen = {}
    orig = model.get_action

    def capture(*a, **kw):
        out = orig(*a, **kw)
        seen["pred"] = out["action_pred"].float().cpu().numpy()
        return out

    model.get_action = capture
    try:
        with FixedNoise(noise):
            acts, _ = wrapper.get_action(obs)
    finally:
        del model.get_action
    return seen["pred"], acts


# --- main ----------------------------------------------------------------------------------------


def stats_pair(a: dict, b: dict, pa: np.ndarray, pb: np.ndarray) -> dict:
    out = {"pcc_valid": gio.pcc(gio.valid(pa), gio.valid(pb)),
           "max_abs_valid": gio.max_abs(gio.valid(pa), gio.valid(pb)),
           "pcc_full": gio.pcc(pa, pb),
           "pcc_actions": gio.pcc(np.concatenate([a[k].ravel() for k in gio.ACTION_KEYS]),
                                  np.concatenate([b[k].ravel() for k in gio.ACTION_KEYS])),
           "per_group": {}}
    for name, _ in schema.ACTION_GROUPS:
        k = f"action.{name}"
        sl = schema.ACTION_SLICES[name]
        out["per_group"][name] = {"max_abs": gio.max_abs(a[k], b[k]), "pcc": gio.pcc(a[k], b[k]),
                                  "max_abs_norm": gio.max_abs(gio.valid(pa)[:, sl], gio.valid(pb)[:, sl])}
    return out


def constant_groups(policy) -> list[str]:
    """Action groups whose statistics have min == max (G1: left_hand 0, base_height 0.74)."""
    params = policy.processor.state_action_processor.norm_params[policy.embodiment_tag.value]["action"]
    out = []
    for name, _ in schema.ACTION_GROUPS:
        p = params[name]
        if "min" in p and "max" in p and np.array_equal(np.asarray(p["min"]), np.asarray(p["max"])):
            out.append(name)
    return out


def env_info() -> dict:
    import diffusers
    import transformers

    try:
        import flash_attn
        fa = flash_attn.__version__
    except Exception as e:  # noqa: BLE001
        fa = f"not importable: {e}"
    drv = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                  text=True).strip()
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__, "diffusers": diffusers.__version__, "flash_attn": fa,
            "gpu": torch.cuda.get_device_name(0), "driver": drv,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def git(repo: Path, *a: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR {e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--golden", default=str(gio.GOLDEN_DIR))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--noise-seed", type=int, default=42, help="sample i uses seed + i")
    ap.add_argument("--precisions", default="bf16,fp32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--wait-min", type=float, default=60.0)
    ap.add_argument("--no-weight-check", action="store_true")
    args = ap.parse_args(argv)
    # Before the first cuBLAS handle: deterministic GEMM workspaces (dump_golden_n16.py:55). Set
    # here, not at import, so importers (check_served_bf16.py) keep the server's defaults.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    precs = [p for p in args.precisions.split(",") if p]
    assert all(p in gio.PRECISIONS for p in precs), precs

    golden = Path(args.golden).expanduser()
    index = gio.load_index(golden)
    snap = snapshot_dir(args.repo, args.revision)
    vram = wait_for_vram(args.min_free_gb, args.wait_min)
    set_determinism(args.noise_seed)
    t_start = time.time()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    obs = {s["sid"]: gio.load_obs(golden / "obs" / f"{s['sid']}.npz") for s in index}
    seeds = {s["sid"]: args.noise_seed + i for i, s in enumerate(index)}
    noise = {sid: make_noise(seed) for sid, seed in seeds.items()}
    res = {p: {} for p in precs}  # prec -> sid -> (pred, acts)
    runs, checks = {}, {p: {} for p in precs}

    t0 = time.time()
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.UNITREE_G1, model_path=str(snap),
                         device=args.device, strict=True)
    wrapper = Gr00tSimPolicyWrapper(policy)
    load_bf16_s = time.time() - t0
    if "bf16" in precs:
        # Warm-up on the first sample so the timed/compared calls do not pay kernel setup.
        infer(policy, obs[index[0]["sid"]], noise[index[0]["sid"]], torch.bfloat16)
        for s in index:
            sid = s["sid"]
            t1 = time.perf_counter()
            pred, acts = infer(policy, obs[sid], noise[sid], torch.bfloat16)
            ms = (time.perf_counter() - t1) * 1e3
            pred_o, acts_o = official(wrapper, obs[sid], noise[sid])
            pred_r, _ = infer(policy, obs[sid], noise[sid], torch.bfloat16)
            checks["bf16"][sid] = {
                "official_vs_replicated_pred_max_abs": gio.max_abs(pred_o, pred),
                "official_vs_replicated_actions_max_abs": max(gio.max_abs(acts_o[k], acts[k]) for k in gio.ACTION_KEYS),
                "rerun_pred_max_abs": gio.max_abs(pred_r, pred), "infer_ms": round(ms, 2)}
            res["bf16"][sid] = (pred, acts)
            print(f"[golden] bf16 {sid}: {checks['bf16'][sid]}", flush=True)
        runs["bf16"] = {"weights": "bf16 (Gr00tPolicy: model.to(bfloat16), gr00t_policy.py:82-84)",
                        "inputs": "bf16 (_rec_to_dtype, gr00t_policy.py:339)",
                        "noise_in": "bf16(noise): torch.randn(dtype=vl_embeds.dtype)",
                        "attention": {"llm_attn": "flash_attention_2", "vision_attn": "flash_attention_2 (packed)",
                                      "dit_attn": "diffusers SDPA, default backend"},
                        "path": "rttt.make_g1_golden.infer == Gr00tSimPolicyWrapper.get_action (checked per sample)",
                        "load_s": round(load_bf16_s, 1)}
    proc_policy = policy
    if "fp32" in precs:
        from transformers import AutoModel

        policy.model = None
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        model = AutoModel.from_pretrained(snap)
        model.eval()
        # The Eagle backbone comes up partly bf16 (frozen layers); .float() upcasts those exactly.
        loaded = {}
        for prm in model.parameters():
            loaded[str(prm.dtype)] = loaded.get(str(prm.dtype), 0) + 1
        model = model.float()
        weights = {"skipped": True} if args.no_weight_check else check_fp32_weights(model, snap)
        weights["param_dtypes_at_load"] = loaded
        print(f"[golden] fp32 weights: {weights}", flush=True)
        model = model.to(device=args.device)
        attn = use_sdpa_math(model)
        proc_policy.model = model
        load_fp32_s = time.time() - t0
        infer(proc_policy, obs[index[0]["sid"]], noise[index[0]["sid"]], torch.float32)
        for s in index:
            sid = s["sid"]
            t1 = time.perf_counter()
            pred, acts = infer(proc_policy, obs[sid], noise[sid], torch.float32)
            ms = (time.perf_counter() - t1) * 1e3
            pred_r, _ = infer(proc_policy, obs[sid], noise[sid], torch.float32)
            checks["fp32"][sid] = {"rerun_pred_max_abs": gio.max_abs(pred_r, pred), "infer_ms": round(ms, 2)}
            res["fp32"][sid] = (pred, acts)
            print(f"[golden] fp32 {sid}: {checks['fp32'][sid]}", flush=True)
        runs["fp32"] = {"weights": "stored precision: F32 tensors kept, BF16 upcast exactly (AutoModel.from_pretrained + .float())",
                        "weight_check": weights, "inputs": "fp32", "noise_in": "noise (fp32)",
                        "attention": {**attn, "dit_attn": "diffusers SDPA, math backend"},
                        "path": "rttt.make_g1_golden.infer (Gr00tPolicy._get_action with fp32 inputs)",
                        "load_s": round(load_fp32_s, 1)}
    vram_peak_gb = torch.cuda.max_memory_allocated() / 2**30

    # --- write samples ------------------------------------------------------------------------
    sdir = golden / "samples"
    sdir.mkdir(exist_ok=True)
    keys, samples = {}, []
    per_group = {name: {"max_abs": 0.0, "argmax": None} for name, _ in schema.ACTION_GROUPS}
    pccs, worst = [], {name: [] for name, _ in schema.ACTION_GROUPS}
    for s in index:
        sid = s["sid"]
        arrays = {"noise": noise[sid].numpy(), "noise_seed": np.array(seeds[sid], np.int64),
                  "bf16.noise_in": noise[sid].to(torch.bfloat16).float().numpy()}
        for p in precs:
            pred, acts = res[p][sid]
            arrays[f"{p}.action_pred_norm"] = pred
            arrays.update({f"{p}.{k}": v for k, v in acts.items()})
        path = sdir / f"{sid}.npz"
        np.savez(path, **{k: np.asarray(v, order="C") for k, v in arrays.items()})  # keeps 0-d
        if not keys:
            keys = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}
        entry = {"sid": sid, "kind": s["kind"], "seed": s["seed"], "k_obs": s["k_obs"],
                 "file": str(path.relative_to(golden)), "sha256": sha256_file(path),
                 "obs_file": f"obs/{sid}.npz", "obs_sha256": sha256_file(golden / "obs" / f"{sid}.npz"),
                 "noise_seed": seeds[sid], "checks": {p: checks[p][sid] for p in precs}}
        if set(gio.PRECISIONS) <= set(precs):
            (pb, ab), (pf, af) = res["bf16"][sid], res["fp32"][sid]
            st = stats_pair(ab, af, pb, pf)
            entry["bf16_vs_fp32"] = st
            pccs.append(st["pcc_valid"])
            for name, g in st["per_group"].items():
                worst[name].append((g["max_abs"], sid))
                if g["max_abs"] > per_group[name]["max_abs"]:
                    per_group[name] = {"max_abs": g["max_abs"], "argmax": sid}
        samples.append(entry)

    summary = None
    if pccs:
        const = constant_groups(proc_policy)
        for name, g in per_group.items():
            g["top3"] = [[sid, d] for d, sid in sorted(worst[name], reverse=True)[:3]]
            g["constant_in_statistics"] = name in const  # min == max: decode ignores the prediction
            g["floor"] = max(2 * g["max_abs"], FLOOR_MIN)
            g["units"] = "m/s (vx, vy), rad/s (wz)" if name == NAV_GROUP else ("m" if name == "base_height_command" else "rad")
            g["t22_gated"] = name not in T22_EXCLUDED
        summary = {"pcc_valid": {"min": min(pccs), "mean": float(np.mean(pccs)), "max": max(pccs)},
                   "per_group": per_group,
                   "floor_rule": "floor = max(2 x largest GPU-bf16-vs-fp32 decoded max|d| over the 20 samples, 0.02)"}
        print(f"[golden] bf16 vs fp32: pcc_valid min {min(pccs):.6f} mean {np.mean(pccs):.6f}", flush=True)
        for name, g in per_group.items():
            print(f"[golden]   {name:20s} max|d| {g['max_abs']:.5f} ({g['argmax']}) floor {g['floor']:.4f}"
                  f"{'' if g['t22_gated'] else '  (not gated)'}", flush=True)

    blobs = {f.name: os.readlink(f).split("/")[-1] for f in sorted(snap.iterdir()) if f.is_symlink()}
    here = Path(__file__).resolve()
    manifest = {
        "what": "GR00T N1.6 G1 goldens (PLAN.md WP-2.2) for T-2.2 fidelity and the server_g1 boot check",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wallclock_s": round(time.time() - t_start, 1),
        "command": " ".join([sys.executable, *sys.argv]),
        "checkpoint": {"repo": args.repo, "revision": args.revision, "id": f"{args.repo}@{args.revision[:8]}",
                       "snapshot": str(snap), "blobs (sha256 for LFS files, git sha1 otherwise)": blobs},
        "embodiment": {"tag": "unitree_g1", "embodiment_id": 8, "action_horizon_model": 50, "rows_used": gio.VALID_ROWS,
                       "max_action_dim": 128, "action_dims_used": gio.VALID_DIMS,
                       "action_groups": [list(g) for g in schema.ACTION_GROUPS]},
        "runs": runs,
        "noise": {"shape": list(gio.NOISE_SHAPE), "dtype": "float32",
                  "rule": "torch.randn((1,50,128), generator=torch.Generator('cpu').manual_seed(noise_seed), dtype=float32)",
                  "noise_seed": f"{args.noise_seed} + sample index (s00 -> {args.noise_seed})",
                  "injection": "torch.randn replaced for the one call in gr00t_n1d6.py:310-314 (asserted once per inference)"},
        "obs_source": {"index": "obs_index.json", "index_sha256": sha256_file(golden / "obs_index.json"),
                       "run": json.loads((golden / "obs_index.json").read_text())["source_run"],
                       "format": "obs/<sid>.npz: the 9 W-VLA keys exactly as sent; language a 0-d str"},
        "sample_keys": keys,
        "action_pred_valid": "action_pred_norm[0, :30, :35]",
        "bf16_vs_fp32": summary,
        "samples": samples,
        "vram": {"preflight": vram, "peak_allocated_gb_this_process": round(vram_peak_gb, 2)},
        "env": env_info(),
        "code": {"isaac_repo": str(Path(sys.modules["gr00t"].__file__).resolve().parents[1]),
                 "isaac_git_head": git(Path(sys.modules["gr00t"].__file__).resolve().parents[1], "rev-parse", "HEAD"),
                 "isaac_git_dirty": git(Path(sys.modules["gr00t"].__file__).resolve().parents[1], "status", "--porcelain",
                                        "--untracked-files=no"),
                 "script": str(here), "script_sha256": sha256_file(here),
                 "golden_io_sha256": sha256_file(here.with_name("golden_io.py"))},
    }
    (golden / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"[golden] wrote {len(samples)} samples to {sdir}; manifest {golden / 'manifest.json'} "
          f"({manifest['wallclock_s']} s, peak {vram_peak_gb:.1f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
