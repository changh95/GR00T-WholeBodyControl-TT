#!/usr/bin/env python
"""The checks of the S1b jobs (jobs/t2_sync.sh, t2_soak.sh, t2_primary.sh) that analyze_run.py / eval_paired.py lack.

    simpy $RT/rttt/m2_checks.py gates SRC DST [BENCH_JSON] [--no-viewer]   m2_rt, vla_p50 = 1.1 x the T-2.3 server p50
    simpy $RT/rttt/m2_checks.py bench SESSIONS_ROOT [ARENA]      the T-2.3 baseline (prints its path)
    simpy $RT/rttt/m2_checks.py soak RUN_DIR DURATION_S OUT [--gate-drift]   T-2.6/T-2.8 checks with no analyze_run key
    simpy $RT/rttt/m2_checks.py errors RUNS_ROOT OUT             T-2.5 / T-2.7: 0 server errors or hangs per arm
    simpy $RT/rttt/m2_checks.py matrix SRC DST TT_URL GPU_URL WBC  a matrix on this session's endpoints
    simpy $RT/rttt/m2_checks.py tt-only MATRIX DST               the T-2.7 matrix with only the C4tt arm
    simpy $RT/rttt/m2_checks.py reuse TT_RUNS|- GPU_DIRS OUT [--force]   may earlier GPU C4 runs stand in for T-2.7's?

Exit 0 pass, 1 fail, 2 bad input. The m2_rt gate set itself (timing rows, starvation <= 0.1% with runs <= 10 ticks,
0 TT error replies, 0 hangs) is gates.yaml's, applied by analyze_run.py; ``gates`` only re-derives its VLA p50 limit
for the session. What is left here: T-2.6's "HUD shows vla=tensix from info" (scripts/hud.py classify on the run's
VLA stamp) and T-2.8's VLA p50 drift < 5% between the first and the last 10 minutes (gated with --gate-drift, else
reported). The drift is taken on the server's service time, the per-request value analyze_run.py gates as
vla_p50_ms (srv_t_reply_ns - srv_t_recv_ns, else srv_infer_ms): the client RTT is stamped at the tick poll that finds
the reply, so it moves in 20 ms steps and would hide a 10% slowdown or invent a 33% one. A GPU arm stands in for
T-2.7's only "when checkpoint and code hashes match" and, since the TT and GPU arms are compared with --crn, only if
it drew server_g1's noise (gpu_vla_server --noise cpu); T-1.9's GPU runs used CUDA-seeded noise, so they never do.
"""

import functools
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

import numpy as np

RT = Path(__file__).resolve().parents[1]
if str(RT) not in sys.path:
    sys.path.insert(0, str(RT))


def _script(name: str):
    """scripts/<name>.py as a module (scripts/ is not a package): analyze_run's vla_server_ms, hud's classify."""
    spec = importlib.util.spec_from_file_location(f"{name}_m2_checks", RT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_analyze_run = functools.lru_cache(maxsize=1)(lambda: _script("analyze_run"))
_hud = functools.lru_cache(maxsize=1)(lambda: _script("hud"))

DRIFT_MAX, DRIFT_WINDOW_S = 0.05, 600.0  # T-2.8
CKPT_ID = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{8}")
DEPLOYED_ARENA = "bfp8_b"  # MK TTPolicy's n16 arena (server_g1 --mk-arena-dtype auto); WP-2.7's bf16 is the fallback
PAIRED_NOISE = "cpu"  # server_g1's draw: torch.Generator("cpu").manual_seed(noise_seed), fp32 (rttt/server_g1.py)
D15 = {"tt": "tcp://127.0.0.1:5557", "gpu": "tcp://127.0.0.1:5565", "wbc": "tcp://127.0.0.1:5556"}


VIEWER_KEYS = ("viewer_k0_ms_max", "viewer_p99_ms_max")


def _p50(bench: dict) -> tuple[str, float] | tuple[None, None]:
    """The T-2.3 server p50 the VLA gate compares with: the span (t_reply - t_recv, what analyze_run gates), then
    total_ms."""
    lat = bench.get("latency") or {}
    for key in ("server_span_ms", "server_total_ms"):
        p50 = (lat.get(key) or {}).get("p50")
        if p50 is not None:
            return key, float(p50)
    return None, None


def derive_gates(src: Path, dst: Path, bench: Path | None, viewer: bool = True) -> dict:
    """The session's m2_rt: gates.yaml's set with vla_p50_ms_max = 1.1 x the T-2.3 server p50 (T-2.6). analyze_run.py
    gates vla_p50_ms as the server's service time (srv_t_reply_ns - srv_t_recv_ns), so the bench's same span is used
    (then its total_ms); the client RTT is quantised by the loop's tick polls and is not comparable. Without the viewer
    the viewer rows are dropped (analyze_run fails them as "no data"), and the note says so. DST holds only m2_rt."""
    import yaml

    gate = dict(yaml.safe_load(Path(src).read_text())["m2_rt"])
    notes = [f"vla_p50_ms_max projected ({gate['vla_p50_ms_max']} ms): no T-2.3 bench JSON"]
    if bench:
        key, p50 = _p50(json.loads(Path(bench).read_text()))
        if p50 is not None:
            gate["vla_p50_ms_max"] = round(1.1 * p50, 3)
            notes = [f"vla_p50_ms_max = 1.1 x T-2.3 {key} p50 {p50:.3f} ms ({bench})"]
        else:
            notes = [f"vla_p50_ms_max kept: {bench} has no server timing"]
    if not viewer:
        for k in VIEWER_KEYS:
            gate.pop(k, None)
        notes.append("viewer rows dropped (run without --viewer; PLAN.md §4.3 wants HUD + viewer for T-2.6)")
    head = f"# m2_rt of one session (rttt/m2_checks.py gates) from {src}: " + "; ".join(notes) + "\n"
    Path(dst).write_text(head + yaml.safe_dump({"m2_rt": gate}, sort_keys=False))
    return {"vla_p50_ms_max": gate["vla_p50_ms_max"], "note": "; ".join(notes)}


def _arena(info) -> str | None:
    return ((info or {}).get("stamp") or {}).get("mk_arena_dtype") if isinstance(info, dict) else None


def pick_bench(root: Path, arena: str = DEPLOYED_ARENA) -> dict:
    """T-2.6's T-2.3 baseline: the newest ``<session>/t23_bench.json`` (by mtime) that passed and whose server ran the
    deployed arena, read from the session's server_g1_info.json (t2_fidelity.sh / t2_bench.sh save it), else from
    the bench's own INFO. The newest file alone was the WP-2.7 bf16 run after S1 (pass false, span p50 47.06 ms
    against 44.12 ms), which would have set the gate at 51.8 ms instead of 48.5 ms."""
    skipped = []
    for p in sorted(Path(root).glob("*/t23_bench.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            bench = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            skipped.append(f"{p.parent.name}: unreadable ({type(e).__name__})")
            continue
        info, src = None, "server_g1_info.json"
        try:
            info = json.loads((p.parent / src).read_text())
        except (OSError, ValueError):
            pass
        if _arena(info) is None:
            info, src = bench.get("info"), "t23_bench.json info"
        if bench.get("pass") is not True:
            skipped.append(f"{p.parent.name}: pass {bench.get('pass')}")
        elif _arena(info) != arena:
            skipped.append(f"{p.parent.name}: arena {_arena(info)}")
        else:
            key, p50 = _p50(bench)
            return {"bench": str(p), "arena": arena, "arena_from": src, "p50_key": key, "p50_ms": p50,
                    "skipped": skipped}
    return {"bench": None, "arena": arena, "skipped": skipped}


def _check(name, value, threshold, ok, gated: bool = True) -> dict:
    return {"name": name, "value": value, "threshold": threshold, "pass": bool(ok), "gated": bool(gated)}


def server_p50_drift(run: Path, window_s: float = DRIFT_WINDOW_S) -> dict:
    """T-2.8: |p50(last window) / p50(first window) - 1| of the server's service time per request (module doc), by
    send time, over the requests analyze_run.py counts for vla_p50_ms (a reply came back, not an episode's pre-roll
    chunk at k_obs 0). drift is None when either window has no server timing (a stock server reports none)."""
    from rtloop.telemetry import load_request_extras, load_requests

    reqs = load_requests(Path(run))
    srv = _analyze_run().vla_server_ms(reqs, load_request_extras(Path(run)))
    ok = (reqs["t_send_ns"] > 0) & (reqs["t_recv_ns"] > reqs["t_send_ns"]) & (reqs["k_obs"] != 0) & np.isfinite(srv)
    t, srv = reqs["t_send_ns"][ok].astype(np.float64), srv[ok]
    out = {"metric": "server service time (analyze_run.vla_server_ms)", "window_s": window_s, "n": int(t.size),
           "p50_first_ms": None, "p50_last_ms": None, "n_first": 0, "n_last": 0, "drift": None}
    if t.size:
        first, last = srv[t < t.min() + window_s * 1e9], srv[t > t.max() - window_s * 1e9]
        out.update(n_first=int(first.size), n_last=int(last.size))
        if first.size and last.size:
            p0, p1 = float(np.median(first)), float(np.median(last))
            out.update(p50_first_ms=round(p0, 3), p50_last_ms=round(p1, 3), drift=float(abs(p1 / p0 - 1)))
    return out


def vla_placement(run: Path) -> dict:
    """Where the HUD draws this run's VLA: scripts/hud.py classify on the backend stamp the loop publishes as
    processor.vla (config.json vla_info, 'policy-server' without one) and the VLA URL. T-2.6: 'vla=tensix'."""
    cfg = json.loads((Path(run) / "config.json").read_text())
    backend = (cfg.get("vla_info") or {}).get("backend", "policy-server")
    return {"backend": backend, **_hud().classify("vla", backend, (cfg.get("args") or {}).get("vla"))}


def soak_checks(run: Path, duration_s: float, gate_drift: bool = False) -> dict:
    """T-2.6 / T-2.8 checks beyond gates.yaml's m2_rt: the VLA placement, and the drift once the run spans two
    windows (gated for T-2.8, reported for T-2.6, whose PLAN row does not ask for it)."""
    where = vla_placement(Path(run))
    checks = [_check("vla_on_tensix", where["device"], "tensix", where["device"] == "tensix")]
    res = {"run": str(run), "duration_s": duration_s, "vla_placement": where}
    if duration_s >= 2 * DRIFT_WINDOW_S:
        res["drift"] = d = server_p50_drift(Path(run))
        checks.append(_check("vla_p50_drift", d["drift"], DRIFT_MAX, d["drift"] is not None and d["drift"] < DRIFT_MAX,
                             gated=gate_drift))
    res.update(checks=checks, **{"pass": all(c["pass"] for c in checks if c["gated"])})
    return res


def server_errors(runs: Path) -> dict:
    """T-2.5's "0 server errors or hangs" (eval_paired.py counts episodes only): request outcomes ERROR (an error
    reply) and TIMEOUT (no reply within the give-up time) per arm, over every run dir under eval_paired's
    ``<out>/runs/<arm>/<block>/``."""
    from rtloop.schema import Outcome
    from rtloop.telemetry import load_requests

    per: dict = {}
    for cfg in sorted(Path(runs).rglob("config.json")):
        arm = cfg.relative_to(runs).parts[0]
        reqs = load_requests(cfg.parent)
        a = per.setdefault(arm, {"runs": 0, "requests": 0, "ERROR": 0, "TIMEOUT": 0})
        a["runs"] += 1
        a["requests"] += int(reqs.size)
        for o in ("ERROR", "TIMEOUT"):
            a[o] += int((reqs["outcome"] == Outcome[o]).sum())
    checks = [_check(f"{arm}_{o.lower()}s", a[o], 0, a[o] == 0) for arm, a in per.items() for o in ("ERROR", "TIMEOUT")]
    if not per:
        checks.append(_check("runs", 0, ">= 1", False))
    return {"runs_root": str(runs), "per_arm": per, "checks": checks, "pass": all(c["pass"] for c in checks)}


def session_matrix(src: Path, dst: Path, tt: str, gpu: str, wbc: str) -> dict:
    """SRC with its D15 endpoints (--vla :5557 TT, --vla :5565 GPU, --wbc :5556) replaced by this session's, so the
    same matrix runs on a session at another port (or a test's); any other tcp endpoint in it is refused."""
    import yaml

    mx = yaml.safe_load(Path(src).read_text())
    swap = {"--vla": {D15["tt"]: tt, D15["gpu"]: gpu}, "--wbc": {D15["wbc"]: wbc}}

    def fix(args: str) -> str:
        tok = shlex.split(args or "")
        for i in range(len(tok) - 1):
            if tok[i] in swap and tok[i + 1].startswith("tcp://"):
                if tok[i + 1] not in swap[tok[i]]:
                    raise ValueError(f"{src}: {tok[i]} {tok[i + 1]} is not a PLAN.md D15 endpoint")
                tok[i + 1] = swap[tok[i]][tok[i + 1]]
        return shlex.join(tok)

    mx["common_args"] = fix(mx.get("common_args", ""))
    for a in mx["arms"]:
        a["args"] = fix(a.get("args", ""))
    head = f"# {src} on this session's endpoints (rttt/m2_checks.py matrix): TT {tt}, GPU {gpu}, WBC {wbc}\n"
    Path(dst).write_text(head + yaml.safe_dump(mx, sort_keys=False))
    return mx


def tt_only_matrix(src: Path, dst: Path) -> dict:
    import yaml

    mx = yaml.safe_load(Path(src).read_text())
    mx["arms"] = [a for a in mx["arms"] if a["name"] == "C4tt"]
    if not mx["arms"]:
        raise ValueError(f"{src}: no C4tt arm")
    mx["compare"] = []
    Path(dst).write_text(yaml.safe_dump(mx, sort_keys=False))
    return mx


def checkpoint_ids(info) -> set[str]:
    """Every '<org>/<repo>@<sha8>' an INFO dict names (model+revision pairs and stamps); '-bf16' copies fold onto
    their source (WP-2.1b writes the BF16 conversion as '<repo>-bf16')."""
    out: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("model"), str) and isinstance(x.get("revision"), str):
                out.add(f"{x['model']}@{x['revision'][:8]}")
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            out.update(CKPT_ID.findall(x))

    walk(info)
    return {re.sub(r"-bf16@", "@", c) for c in out}


def code_hashes(cfg: dict) -> dict:
    """sha256 of every RT rtloop/ source the run imported (config.json imported_dirty: RT is untracked)."""
    return {e["path"]: e["sha256"] for e in cfg.get("imported_dirty") or []
            if e.get("tree") == "RT" and str(e.get("path", "")).startswith("rtloop/")}


def gpu_noise(gpu: list[dict]) -> dict:
    """The noise modes the GPU runs' VLA drew (config.json vla_info.noise; None = a server older than --noise)."""
    noise = sorted({str((c.get("vla_info") or {}).get("noise")) for c in gpu})
    return {"gpu_noise": noise, "noise_paired": noise == [PAIRED_NOISE]}


def reuse_check(tt_runs: Path | None, gpu_dirs: list[Path], force: bool = False) -> dict:
    """``force`` (T2_FORCE_REUSE=1) accepts a checkpoint or code mismatch, never unpaired noise: a GPU arm that did
    not draw server_g1's CPU noise is not a --crn partner of the TT arm (gpu_vla_server info.noise, 'cuda' or absent
    on T-1.9's runs). ``tt_runs`` None checks the GPU side only (the pre-flight before the TT arm runs)."""
    gpu = [json.loads(p.read_text()) for d in gpu_dirs for p in sorted(Path(d).rglob("config.json"))]
    if tt_runs is None:
        rep = {"gpu_runs": len(gpu), **gpu_noise(gpu)}
        rep["accepted"] = rep["reusable"] = bool(gpu) and rep["noise_paired"]
        rep["why"] = ("ok" if rep["accepted"] else "no config.json found" if not gpu
                      else f"GPU arm not noise-paired (info.noise {rep['gpu_noise']}, want ['{PAIRED_NOISE}'])")
        return rep
    tt = [json.loads(p.read_text()) for p in sorted(Path(tt_runs).rglob("config.json"))]
    rep = {"tt_runs": len(tt), "gpu_runs": len(gpu), "forced": bool(force)}
    if not tt or not gpu:
        return dict(rep, reusable=False, accepted=False, why="no config.json found")
    tt_ck = set().union(*(checkpoint_ids(c.get("vla_info") or {}) for c in tt))
    gpu_ck = set().union(*(checkpoint_ids(c.get("vla_info") or {}) for c in gpu))
    noise = gpu_noise(gpu)["gpu_noise"]
    hashes = [code_hashes(c) for c in tt + gpu]
    ref = hashes[0]
    diff = sorted({k for h in hashes for k in set(h) | set(ref) if h.get(k) != ref.get(k)})
    rep.update(tt_checkpoints=sorted(tt_ck), gpu_checkpoints=sorted(gpu_ck), checkpoint_match=bool(tt_ck & gpu_ck),
               code_files=len(ref), code_match=not diff and bool(ref), code_diff=diff, gpu_noise=noise,
               noise_paired=noise == [PAIRED_NOISE])
    rep["reusable"] = rep["checkpoint_match"] and rep["code_match"] and rep["noise_paired"]
    rep["accepted"] = rep["reusable"] or (rep["forced"] and rep["noise_paired"])
    unpaired = f"GPU arm not noise-paired (info.noise {noise}, want ['{PAIRED_NOISE}'])"
    rep["why"] = ("ok" if rep["reusable"] else unpaired if not rep["noise_paired"]
                  else "checkpoint differs" if not rep["checkpoint_match"]
                  else "no rtloop/ hashes recorded" if not ref else f"code differs: {diff[:10]}")
    return rep


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, a = argv[0], argv[1:]
    flags = {x for x in a if x.startswith("--")}
    a = [x for x in a if not x.startswith("--")]
    try:
        if cmd == "gates":
            res = derive_gates(Path(a[0]), Path(a[1]), Path(a[2]) if len(a) > 2 and a[2] else None,
                               "--no-viewer" not in flags)
            print(f"m2_rt vla_p50_ms_max = {res['vla_p50_ms_max']} ({res['note']})")
            return 0
        if cmd == "bench":
            res = pick_bench(Path(a[0]), a[1] if len(a) > 1 else DEPLOYED_ARENA)
            for s in res["skipped"]:
                print(f"T-2.3 bench skipped: {s}", file=sys.stderr)
            if not res["bench"]:
                print(f"no passing T-2.3 bench with arena {res['arena']} under {a[0]}", file=sys.stderr)
                return 1
            print(f"T-2.3 baseline: {res['bench']} (pass, arena {res['arena']} from {res['arena_from']}, "
                  f"{res['p50_key']} p50 {res['p50_ms']} ms)", file=sys.stderr)
            print(res["bench"])
            return 0
        if cmd == "matrix":
            session_matrix(Path(a[0]), Path(a[1]), a[2], a[3], a[4])
            return 0
        if cmd == "tt-only":
            tt_only_matrix(Path(a[0]), Path(a[1]))
            return 0
        if cmd == "soak":
            res, out = soak_checks(Path(a[0]), float(a[1]), "--gate-drift" in flags), Path(a[2])
        elif cmd == "errors":
            res, out = server_errors(Path(a[0])), Path(a[1])
        elif cmd == "reuse":
            res = reuse_check(None if a[0] == "-" else Path(a[0]), [Path(d) for d in a[1].split(",") if d],
                              "--force" in flags)
            out = Path(a[2])
            print(f"GPU-arm reuse: {res['reusable']}{' (forced)' if res['accepted'] and not res['reusable'] else ''}"
                  f" ({res['why']})")
        else:
            raise ValueError(f"unknown command {cmd!r}")
    except (IndexError, KeyError, ValueError, OSError) as e:
        print(f"m2_checks {cmd}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    for c in res.get("checks", []):
        print(f"  {c['name']:<20} {c['value']!s:>12}  (want {c['threshold']})  {'PASS' if c['pass'] else 'FAIL'}"
              f"{'' if c['gated'] else ' (reported, not gated)'}")
    return 0 if res.get("pass", res.get("accepted")) else 1


if __name__ == "__main__":
    sys.exit(main())
