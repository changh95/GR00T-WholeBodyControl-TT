"""T-2.2 TT G1 fidelity (PLAN.md §2 M2): server_g1's predict_debug against the WP-2.2 G1 goldens.

    $DEV $RT/scripts/chip_session.sh --vla tt --debug-endpoints --job $RT/jobs/t2_fidelity.sh
      -> simpy -m pytest $RT/tests/tt/test_g1_fidelity.py   with --endpoint/--golden (conftest_fidelity.py) or
         G1_ENDPOINT / G1_GOLDEN / G1_FIDELITY_REPORT in the environment

Gates, as T-2.2 states them, on action_pred_valid = action_pred_norm[0, :30, :35] (rttt/golden_io.py valid()):
- primary, vs the bf16 golden: PCC mean >= 0.999 and min >= 0.998 over the 20 samples;
- secondary, vs fp32: per sample, PCC(TT, fp32) >= PCC(GPU-bf16, fp32) - 0.001;
- per group, absolute, vs the bf16 golden: decoded max|d| <= max(2 x the largest GPU-bf16-vs-fp32 decoded max|d| of
  that group over all 20 samples, 0.02) (rad; m/s and rad/s for nav); left_hand and base_height excluded;
- 5 replays with fixed noise are bit-identical (action_pred_norm and every decoded action, compared as bytes).
Recorded next to the gates, never gated: TT vs fp32 decoded max|d| per group (a group miss can then be told apart
from the bf16 golden's own error, S1's navigate_command case), and with --fidelity-outputs every TT output as
``<sid>/<key>`` in one npz, for offline analysis without a second device hold.
Each sample is sent with its saved fp32 noise (samples/<sid>.npz "noise"), as the LIBERO boot check does; the bf16
golden started from bf16(noise) inside the model. Floors are recomputed from the samples (the manifest's are
recorded next to them). Everything is computed once per module; the report (--fidelity-report) holds every number.

Skipped without an endpoint (no device session). An endpoint without a complete golden is a failure.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rtloop import schema
from rttt import golden_io as gio
from rttt.bench_g1 import G1Client, ServerError, compare, outputs

PRIMARY_MEAN, PRIMARY_MIN = 0.999, 0.998
SECONDARY_SLACK = 0.001
FLOOR_MIN = 0.02  # rad; m/s and rad/s for nav
EXCLUDED = ("left_hand", "base_height_command")
GATED_GROUPS = tuple(name for name, _ in schema.ACTION_GROUPS if name not in EXCLUDED)
N_SAMPLES = 20
N_REPLAYS = 5
ENV = {"--endpoint": "G1_ENDPOINT", "--golden": "G1_GOLDEN", "--fidelity-report": "G1_FIDELITY_REPORT",
       "--fidelity-timeout-ms": "G1_FIDELITY_TIMEOUT_MS", "--fidelity-outputs": "G1_FIDELITY_OUTPUTS"}


def _opt(config, name: str):
    v = config.getoption(name, default=None)
    return v if v not in (None, "") else (os.environ.get(ENV[name]) or None)


def _rows(a) -> np.ndarray:
    """Decoded action (1, T, D) or (T, D) -> float64 (H, D)."""
    a = np.asarray(a, np.float64)
    return (a[0] if a.ndim == 3 else a)[: schema.H]


# --- metrics (pure; also used by tests/tt/test_bench_g1_fake.py through this module) --------------------------------


def sample_metrics(sid: str, kind: str, golden: dict, tt: dict) -> dict:
    """One sample: PCCs on action_pred_valid and per-group decoded max|d| (TT vs bf16 and bf16 vs fp32, gated; TT vs
    fp32, recorded)."""
    tv = gio.valid(tt["action_pred_norm"])
    bv, fv = gio.valid(golden["bf16.action_pred_norm"]), gio.valid(golden["fp32.action_pred_norm"])
    out = {"sid": sid, "kind": kind, "pcc_tt_bf16": gio.pcc(tv, bv), "pcc_tt_fp32": gio.pcc(tv, fv),
           "pcc_bf16_fp32": gio.pcc(bv, fv), "max_abs_valid_tt_bf16": gio.max_abs(tv, bv), "groups": {}}
    for name, _ in schema.ACTION_GROUPS:
        k = f"action.{name}"
        if k not in tt:
            raise KeyError(f"{sid}: predict_debug reply has no {k}")
        t, b, f = _rows(tt[k]), _rows(golden[f"bf16.{k}"]), _rows(golden[f"fp32.{k}"])
        if t.shape != b.shape:
            raise ValueError(f"{sid}: {k} TT {t.shape} vs golden {b.shape}")
        out["groups"][name] = {"tt_bf16": gio.max_abs(t, b), "bf16_fp32": gio.max_abs(b, f),
                               "tt_fp32": gio.max_abs(t, f)}
    return out


def verdicts(per: list[dict], replays: dict, manifest_floors: dict | None = None) -> dict:
    """The four T-2.2 gates over the per-sample metrics; per group also the TT-vs-fp32 maximum (not gated)."""
    p = np.array([m["pcc_tt_bf16"] for m in per])
    primary = {"mean": float(p.mean()), "min": float(p.min()), "argmin": per[int(p.argmin())]["sid"],
               "want": {"mean": PRIMARY_MEAN, "min": PRIMARY_MIN}}
    primary["pass"] = bool(primary["mean"] >= PRIMARY_MEAN and primary["min"] >= PRIMARY_MIN)
    sec_fail = [{"sid": m["sid"], "pcc_tt_fp32": m["pcc_tt_fp32"], "pcc_bf16_fp32": m["pcc_bf16_fp32"]}
                for m in per if m["pcc_tt_fp32"] < m["pcc_bf16_fp32"] - SECONDARY_SLACK]
    secondary = {"slack": SECONDARY_SLACK, "failures": sec_fail, "pass": not sec_fail,
                 "min_margin": float(min(m["pcc_tt_fp32"] - m["pcc_bf16_fp32"] for m in per))}
    groups = {}
    for name, _ in schema.ACTION_GROUPS:
        ref = max(m["groups"][name]["bf16_fp32"] for m in per)
        floor = max(2 * ref, FLOOR_MIN)
        worst = max(per, key=lambda m: m["groups"][name]["tt_bf16"])
        worst32 = max(per, key=lambda m: m["groups"][name]["tt_fp32"])
        g = {"floor": floor, "bf16_fp32_max": ref, "tt_bf16_max": worst["groups"][name]["tt_bf16"],
             "argmax": worst["sid"], "tt_fp32_max": worst32["groups"][name]["tt_fp32"], "argmax_fp32": worst32["sid"],
             "gated": name in GATED_GROUPS}
        if manifest_floors and name in manifest_floors:
            g["manifest_floor"] = manifest_floors[name]
        g["pass"] = (not g["gated"]) or g["tt_bf16_max"] <= floor
        groups[name] = g
    group = {"groups": groups, "pass": all(g["pass"] for g in groups.values()),
             "failures": [n for n, g in groups.items() if not g["pass"]]}
    rep = {"pass": bool(replays) and all(r["identical"] for r in replays.values()), "samples": replays}
    return {"primary": primary, "secondary": secondary, "groups": group, "replays": rep}


def save_outputs(path, firsts: dict) -> None:
    """Every TT output of the first pass as ``<sid>/<key>`` arrays in one npz (S1's tt_outputs.npz layout)."""
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"{sid}/{k}": np.asarray(v) for sid, tt in firsts.items() for k, v in tt.items()})


# --- the one pass over the server -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fid(request):
    endpoint = _opt(request.config, "--endpoint")
    if not endpoint:
        pytest.skip("T-2.2 needs a live server_g1 with --debug-endpoints: give --endpoint or G1_ENDPOINT")
    golden = Path(_opt(request.config, "--golden") or gio.GOLDEN_DIR).expanduser()
    if not (golden / "obs_index.json").exists():
        pytest.fail(f"golden {golden}: no obs_index.json (WP-2.2 make_g1_golden.py writes it)")
    timeout_ms = int(_opt(request.config, "--fidelity-timeout-ms") or 60000)
    client = G1Client(endpoint, timeout_ms)
    if not client.ping(timeout_ms=10000):
        pytest.fail(f"{endpoint} does not answer ping")
    info = client.info()
    index = gio.load_index(golden)
    manifest_path = golden / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    try:
        manifest_floors = gio.group_floors(manifest)
    except (KeyError, TypeError):
        manifest_floors = None
    per, firsts, loaded = [], {}, {}
    try:
        for s in index:
            sid = s["sid"]
            smp = gio.load_sample(golden, sid)
            missing = [k for k in ("noise", "bf16.action_pred_norm", "fp32.action_pred_norm") if k not in smp]
            if missing:
                pytest.fail(f"golden sample {sid} lacks {missing}")
            reply, _ = client.predict_debug(smp["obs"], smp["noise"])
            tt = outputs(reply)
            firsts[sid], loaded[sid] = tt, smp
            per.append(sample_metrics(sid, s.get("kind", "?"), smp, tt))
        save_outputs(_opt(request.config, "--fidelity-outputs"), firsts)
        replays = {}
        for kind in ("reset", "mid"):  # one sample of each kind, N_REPLAYS more calls each
            sid = next((s["sid"] for s in index if s.get("kind") == kind), None)
            if sid is None:
                continue
            diffs = [compare(firsts[sid], outputs(client.predict_debug(loaded[sid]["obs"], loaded[sid]["noise"])[0]))
                     for _ in range(N_REPLAYS)]
            replays[sid] = {"identical": all(same for same, _ in diffs), "max_abs": max(d for _, d in diffs),
                            "n": N_REPLAYS}
        if not replays and index:
            sid = index[0]["sid"]
            diffs = [compare(firsts[sid], outputs(client.predict_debug(loaded[sid]["obs"], loaded[sid]["noise"])[0]))
                     for _ in range(N_REPLAYS)]
            replays[sid] = {"identical": all(same for same, _ in diffs), "max_abs": max(d for _, d in diffs),
                            "n": N_REPLAYS}
    except (TimeoutError, ServerError) as e:
        pytest.fail(f"predict_debug failed: {e} (server_g1 needs --debug-endpoints)")
    finally:
        client.close()
    res = {"endpoint": endpoint, "golden": str(golden), "info": info, "n_samples": len(per),
           "kinds": {k: sum(m["kind"] == k for m in per) for k in sorted({m["kind"] for m in per})},
           "noise_key": "noise", "per_sample": per, "verdicts": verdicts(per, replays, manifest_floors)}
    yield res
    report = _opt(request.config, "--fidelity-report")
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(res, indent=1, default=str) + "\n")


def _table(per: list[dict], key: str, ref: str | None = None) -> str:
    return "\n".join(f"  {m['sid']:>6} {m['kind']:>5} {key} {m[key]:.6f}" + (f"  {ref} {m[ref]:.6f}" if ref else "")
                     for m in per)


def test_golden_complete(fid):
    assert fid["n_samples"] == N_SAMPLES, f"golden has {fid['n_samples']} samples, T-2.2 wants {N_SAMPLES}"


def test_primary_pcc_vs_bf16(fid):
    v = fid["verdicts"]["primary"]
    assert v["pass"], (f"PCC(TT, bf16) mean {v['mean']:.6f} (>= {PRIMARY_MEAN}), min {v['min']:.6f} at {v['argmin']} "
                       f"(>= {PRIMARY_MIN})\n" + _table(fid["per_sample"], "pcc_tt_bf16"))


def test_secondary_vs_fp32(fid):
    v = fid["verdicts"]["secondary"]
    assert v["pass"], (f"PCC(TT, fp32) < PCC(GPU-bf16, fp32) - {SECONDARY_SLACK} on {len(v['failures'])} samples: "
                       f"{v['failures']}")


def test_group_abs_floors(fid):
    v = fid["verdicts"]["groups"]
    lines = [f"  {n:<20} TT-bf16 max|d| {g['tt_bf16_max']:.5f} ({g['argmax']})  floor {g['floor']:.5f}"
             f"{'' if g['gated'] else '  (not gated)'}  [TT-fp32 {g['tt_fp32_max']:.5f} ({g['argmax_fp32']})]"
             for n, g in v["groups"].items()]
    assert v["pass"], f"groups over their floor: {v['failures']}\n" + "\n".join(lines)


def test_fixed_noise_replays_bit_identical(fid):
    v = fid["verdicts"]["replays"]
    assert v["samples"], "no sample to replay"
    assert v["pass"], f"fixed-noise replays differ: {v['samples']}"
