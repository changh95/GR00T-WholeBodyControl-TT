#!/usr/bin/env python
"""Paired success evaluation of loop arms over common seeds (PLAN.md §4.4; DP10: 50 paired episodes per arm,
gate only gross drops).

    simpy $RT/scripts/eval_paired.py --matrix MATRIX.yaml [--out DIR] [--procs N]
        [--block N] [--seeds 2000-2049] [--resume] [--dry-run] [--loop-cmd "CMD"]
    simpy $RT/scripts/eval_paired.py --report-only C1=DIR[,DIR...] C4=DIR [--compare C4,C1] [--matrix M] --out DIR

Matrix YAML (only `name` and `arms` are required):

    name: m3_primary          # run tags are <name>_<arm>
    seeds: "2000-2049"        # "A-B", "A-B,C,D-E" or a YAML list of ints; default 2000-2049
    common_args: "--vla tcp://127.0.0.1:5557 --wbc tcp://127.0.0.1:5558 --crn"
    procs: 1                  # loop processes at once (VL matrices may use more; RT timing needs 1)
    block: 10                 # seeds per loop invocation; blocks interleave across arms (0 = all seeds at once)
    max_ticks: 1440           # time-to-success censoring
    gross_drop: -0.25         # lower 80% bound of (new - ref) below this = "gross drop"
    arms:
      - {name: C1, args: "--mode sync"}
      - {name: C4, args: "--mode rt --arm v1 --s 16 --B 0"}
    compare:                  # [new, ref] pairs are gated; {new: X, ref: Y, gate: false} is reported only
      - [C4, C1]

Run mode: for each seed block, for each arm, `simpy -m rtloop.loop <common_args> <arm args> --seeds A-B --tag
<name>_<arm>`, with RTLOOP_RUNS_ROOT=<out>/runs/<arm>/<A>-<B> so the job's run dir is the only one there (if the
loop ignores it, the job's dir is looked up in $RT/runs by tag, start time and seeds). Logs go to <out>/logs/,
one completion marker per job to <out>/jobs/ (--resume skips jobs whose marker says exit 0).

Statistics. Per arm: success rate with a Wilson 95% CI. Paired, over the seeds both arms finished: b = new-only
successes, c = ref-only successes, delta = (b - c)/n, SE = sqrt(b + c - (b - c)^2/n)/n, two-sided 95% Wald CI
delta +- 1.96 SE, lower one-sided 80% bound delta - 0.8416 SE, exact two-sided McNemar binomial p (scipy's
binomtest when importable, else the exact sum). Secondary: time to success censored at max_ticks, as the paired
restricted-mean difference with a 2000-resample seed bootstrap CI and a seed-stratified log-rank test.

Writes <out>/report.json and <out>/report.md. Exit 0; 1 if a gated comparison is a gross drop; 3 if a loop job
failed, seeds are missing, an arm has no episodes or a gated comparison has no paired seeds; 2 on bad input.

Each loop job runs in its own session (process group): SIGINT or SIGTERM to eval_paired sends SIGTERM to the
whole group (bash wrappers and rt_sim alike; rt_sim stops after the tick and flushes its logs), waits for the
group to empty, and SIGKILLs it after STOP_TIMEOUT_S, so no loop outlives eval_paired.
"""

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RT = Path(__file__).resolve().parents[1]
if str(RT) not in sys.path:
    sys.path.insert(0, str(RT))

from rtloop.schema import MAX_EPISODE_TICKS, EpisodeEnd  # noqa: E402
from rtloop.telemetry import RUNS, RUNS_ENV, jsonable, load_episodes  # noqa: E402

Z95 = 1.959963984540054  # two-sided 95%
Z95_ONE = 1.6448536269514722  # one-sided 95%
Z80_ONE = 0.8416212335729143  # one-sided 80%
GROSS_DROP = -0.25
DEFAULT_SEEDS = "2000-2049"
DEFAULT_LOOP_CMD = ["bash", "-c", 'source "$0" >/dev/null && simpy -m rtloop.loop "$@"', str(RT / "env.sh")]
STOP_TIMEOUT_S = 30.0


# --- Statistics ---------------------------------------------------------------------------------


def wilson(x: int, n: int, z: float = Z95) -> tuple[float, float] | tuple[None, None]:
    """Wilson score interval for x successes in n trials."""
    if n == 0:
        return None, None
    p = x / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def binom_two_sided_half(k: int, n: int) -> float:
    """Exact two-sided binomial p for k = min(b, c) of n discordant pairs at p = 1/2."""
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def mcnemar_exact(b: int, c: int) -> tuple[float, str]:
    n = b + c
    if n == 0:
        return 1.0, "no discordant pairs"
    try:
        from scipy.stats import binomtest
    except ImportError:
        return binom_two_sided_half(min(b, c), n), "exact binomial"
    return float(binomtest(min(b, c), n, 0.5).pvalue), "scipy.stats.binomtest"


def paired_success(sn: np.ndarray, sr: np.ndarray, gross_drop: float = GROSS_DROP) -> dict:
    """Paired difference of two success vectors over the same seeds (new, ref)."""
    sn, sr = np.asarray(sn, bool), np.asarray(sr, bool)
    n = int(len(sn))
    b, c = int((sn & ~sr).sum()), int((~sn & sr).sum())
    out = {"n": n, "b": b, "c": c, "both": int((sn & sr).sum()), "neither": int((~sn & ~sr).sum())}
    if n == 0:
        out.update(delta=None, se=None, ci95=[None, None], lower80=None, lower95_one_sided=None,
                   p_mcnemar=None, p_method=None, verdict="no data")
        return out
    delta = (b - c) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    p, method = mcnemar_exact(b, c)
    lower80 = delta - Z80_ONE * se
    out.update(delta=delta, se=se, ci95=[delta - Z95 * se, delta + Z95 * se], lower80=lower80,
               lower95_one_sided=delta - Z95_ONE * se, p_mcnemar=p, p_method=method,
               verdict="gross drop" if lower80 < gross_drop else "no gross drop")
    return out


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def logrank_stratified(tn, en, tr, er) -> dict:
    """Log-rank test of new vs ref, stratified by seed (each stratum = one seed's pair of episodes)."""
    o_e = var = 0.0
    for t1, e1, t2, e2 in zip(tn, en, tr, er, strict=True):
        for t in sorted({t for t, e in ((t1, e1), (t2, e2)) if e}):
            at1, at2 = t1 >= t, t2 >= t
            n_risk = at1 + at2
            d1, d2 = bool(e1 and t1 == t), bool(e2 and t2 == t)
            d = d1 + d2
            o_e += d1 - d * at1 / n_risk
            if n_risk > 1:
                var += d * (at1 / n_risk) * (1 - at1 / n_risk) * (n_risk - d) / (n_risk - 1)
    if var <= 0:
        return {"o_minus_e": o_e, "var": var, "z": None, "p": None}
    z = o_e / math.sqrt(var)
    return {"o_minus_e": o_e, "var": var, "z": z, "p": 2 * _norm_sf(abs(z))}


def time_to_success(new: list[dict], ref: list[dict], max_ticks: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired restricted-mean time to success (ticks; failures censored at max_ticks), new - ref."""

    def tt(rec):
        fst = rec.get("first_success_tick")
        if is_success(rec):
            return (min(int(fst), max_ticks) if fst is not None else max_ticks), True
        return max_ticks, False

    if not new:
        return {"n": 0}
    tn, en = zip(*(tt(r) for r in new))
    tr, er = zip(*(tt(r) for r in ref))
    diff = np.asarray(tn, np.float64) - np.asarray(tr, np.float64)
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    return {"n": len(diff), "max_ticks": max_ticks, "rmst_new": float(np.mean(tn)), "rmst_ref": float(np.mean(tr)),
            "rmst_diff": float(diff.mean()),
            "rmst_diff_ci95_boot": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "logrank": logrank_stratified(tn, en, tr, er),
            "success_missing_tick": sum(is_success(r) and r.get("first_success_tick") is None for r in new + ref)}


# --- Episodes -----------------------------------------------------------------------------------


def end_name(v) -> str:
    if isinstance(v, str):
        return v.upper()
    try:
        return EpisodeEnd(int(v)).name
    except (TypeError, ValueError):
        return str(v)


def is_success(rec: dict) -> bool:
    if "success" in rec and rec["success"] is not None:
        return bool(rec["success"])
    return end_name(rec.get("end")) == "SUCCESS"


def load_arm(dirs: list[Path]) -> dict:
    """{seed: episode record} over the arm's run dirs (a repeated seed keeps the last record)."""
    per_seed, dups, unseeded = {}, 0, 0
    for d in dirs:
        for rec in load_episodes(d):
            s = rec.get("seed")
            if s is None:
                unseeded += 1
                continue
            dups += int(s) in per_seed
            per_seed[int(s)] = rec
    return {"per_seed": per_seed, "dup_seeds": dups, "unseeded": unseeded, "run_dirs": [str(d) for d in dirs]}


def arm_summary(name: str, arm: dict, expected: list[int] | None) -> dict:
    recs = list(arm["per_seed"].values())
    x, n = sum(is_success(r) for r in recs), len(recs)
    lo, hi = wilson(x, n)
    ends = [end_name(r.get("end")) for r in recs]
    fst = [r["first_success_tick"] for r in recs if is_success(r) and r.get("first_success_tick") is not None]
    return {"arm": name, "n": n, "successes": x, "rate": x / n if n else None, "wilson95": [lo, hi],
            "falls": ends.count("FALL"), "safety_faults": ends.count("SAFETY_FAULT"), "truncs": ends.count("TRUNC"),
            "starve_faults": sum(bool(r.get("starve_fault")) for r in recs),
            "first_success_tick_mean": float(np.mean(fst)) if fst else None,
            "missing_seeds": sorted(set(expected) - set(arm["per_seed"])) if expected else [],
            "dup_seeds": arm["dup_seeds"], "unseeded": arm["unseeded"], "run_dirs": arm["run_dirs"]}


def compare(new_name: str, ref_name: str, arms: dict, gate: bool, gross_drop: float, max_ticks: int) -> dict:
    new, ref = arms[new_name]["per_seed"], arms[ref_name]["per_seed"]
    seeds = sorted(set(new) & set(ref))
    sn = np.array([is_success(new[s]) for s in seeds], bool)
    sr = np.array([is_success(ref[s]) for s in seeds], bool)
    out = {"new": new_name, "ref": ref_name, "gate": gate, "seeds": [seeds[0], seeds[-1]] if seeds else None}
    out.update(paired_success(sn, sr, gross_drop))
    falls_new = sum(end_name(new[s].get("end")) == "FALL" for s in seeds)
    falls_ref = sum(end_name(ref[s].get("end")) == "FALL" for s in seeds)
    out.update(falls_new=falls_new, falls_ref=falls_ref, falls_ok=falls_new <= falls_ref + 2,
               time_to_success=time_to_success([new[s] for s in seeds], [ref[s] for s in seeds], max_ticks))
    return out


# --- Matrix and jobs ----------------------------------------------------------------------------


def parse_seeds(spec) -> list[int]:
    if isinstance(spec, bool):
        raise ValueError(f"bad seeds {spec!r}")
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return [s for item in spec for s in parse_seeds(item)]
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if int(b) < int(a):
                raise ValueError(f"bad seed range {part!r}")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def seed_blocks(seeds: list[int], size: int) -> list[tuple[int, int]]:
    """Contiguous (first, last) runs of at most `size` seeds (size 0: split only at gaps)."""
    out, cur = [], []
    for s in sorted(set(seeds)):
        if cur and (s != cur[-1] + 1 or (size and len(cur) >= size)):
            out.append((cur[0], cur[-1]))
            cur = []
        cur.append(s)
    if cur:
        out.append((cur[0], cur[-1]))
    return out


def load_matrix(path: Path) -> dict:
    import yaml

    try:
        mx = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: {e}") from e
    if not isinstance(mx, dict) or not mx.get("name") or not mx.get("arms"):
        raise ValueError(f"{path}: a matrix needs `name` and `arms`")
    names = [a["name"] for a in mx["arms"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate arm names")
    for a in mx["arms"]:
        a.setdefault("args", "")
    mx["compare"] = [normalize_compare(c, names) for c in mx.get("compare") or []]
    mx.setdefault("seeds", DEFAULT_SEEDS)
    return mx


def normalize_compare(c, names: list[str]) -> dict:
    if isinstance(c, str):
        c = c.split(",")
    if isinstance(c, (list, tuple)) and len(c) == 2:
        c = {"new": c[0], "ref": c[1], "gate": True}
    if not isinstance(c, dict) or not {"new", "ref"} <= set(c):
        raise ValueError(f"bad compare entry {c!r}")
    for k in ("new", "ref"):
        if c[k] not in names:
            raise ValueError(f"compare names unknown arm {c[k]!r} (arms: {', '.join(names)})")
    return {"new": str(c["new"]), "ref": str(c["ref"]), "gate": bool(c.get("gate", True))}


def plan_jobs(mx: dict, out: Path, loop_cmd: list[str], seeds: list[int], block: int) -> list[dict]:
    jobs = []
    for a, b in seed_blocks(seeds, block):
        for arm in mx["arms"]:
            name = arm["name"]
            jid = f"{name}_{a}-{b}"
            cmd = [*loop_cmd, *shlex.split(mx.get("common_args") or ""), *shlex.split(arm["args"]),
                   "--seeds", f"{a}-{b}", "--tag", f"{mx['name']}_{name}"]
            jobs.append({"id": jid, "arm": name, "seeds": [a, b], "tag": f"{mx['name']}_{name}", "cmd": cmd,
                         "root": str(out / "runs" / name / f"{a}-{b}"), "log": str(out / "logs" / f"{jid}.log"),
                         "marker": str(out / "jobs" / f"{jid}.json")})
    return jobs


def _episode_seeds(d: Path) -> set[int]:
    return {int(r["seed"]) for r in load_episodes(d) if r.get("seed") is not None}


def find_run_dirs(job: dict, t_start: float) -> list[str]:
    root = Path(job["root"])
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.is_symlink()) if root.exists() else []
    if dirs:
        return [str(d) for d in dirs]
    a, b = job["seeds"]  # the loop ignored RTLOOP_RUNS_ROOT: look in the default root
    cands = sorted(RUNS.glob(f"*_{job['tag']}")) if RUNS.exists() else []
    return [str(d) for d in cands if d.is_dir() and not d.is_symlink() and d.stat().st_mtime >= t_start - 1
            and _episode_seeds(d) and _episode_seeds(d) <= set(range(a, b + 1))]


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_group(p: subprocess.Popen, timeout_s: float = STOP_TIMEOUT_S) -> None:
    """SIGTERM the job's whole process group, wait until it is empty, then SIGKILL what is left.

    p.terminate() would reach only the outer `bash -c`; the simpy subshell and rt_sim would live on."""
    pgid = p.pid  # start_new_session=True: the job leads its own group
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        p.poll()  # reap the leader so it does not linger as a zombie group member
        if not _group_alive(pgid):
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _raise_interrupt(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


def run_jobs(jobs: list[dict], procs: int, resume: bool) -> list[dict]:
    """Run the loop jobs in order, `procs` at a time; returns one marker dict per job."""
    done, pending, running = {}, list(jobs), []
    for job in jobs:
        Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job["marker"]).parent.mkdir(parents=True, exist_ok=True)
    try:
        while pending or running:
            while pending and len(running) < max(1, procs):
                job = pending.pop(0)
                marker = Path(job["marker"])
                if resume and marker.exists():
                    prev = json.loads(marker.read_text())
                    if prev.get("returncode") == 0 and prev.get("run_dirs"):
                        print(f"[skip] {job['id']} (done)", flush=True)
                        done[job["id"]] = prev
                        continue
                Path(job["root"]).mkdir(parents=True, exist_ok=True)
                log = open(job["log"], "w")
                env = dict(os.environ, **{RUNS_ENV: job["root"]})
                t0 = time.time()
                p = subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
                running.append((job, p, log, t0))
                print(f"[start] {job['id']}: {shlex.join(job['cmd'])}", flush=True)
            for item in list(running):
                job, p, log, t0 = item
                rc = p.poll()
                if rc is None:
                    continue
                log.close()
                running.remove(item)
                rec = {**{k: job[k] for k in ("id", "arm", "seeds", "tag", "cmd", "log")}, "returncode": rc,
                       "run_dirs": find_run_dirs(job, t0), "t_start": t0, "t_end": time.time()}
                Path(job["marker"]).write_text(json.dumps(rec, indent=1) + "\n")
                done[job["id"]] = rec
                print(f"[done] {job['id']} rc={rc} {time.time() - t0:.0f} s -> {rec['run_dirs']}", flush=True)
            if running:
                time.sleep(0.2)
    except KeyboardInterrupt:
        for _, p, log, _ in running:  # our own loop children only (each in its own process group)
            stop_group(p)
            log.close()
        raise
    return [done[j["id"]] for j in jobs]


# --- Report -------------------------------------------------------------------------------------


def _pct(v, signed: bool = False) -> str:
    if v is None:
        return "-"
    return f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"


def report_md(rep: dict) -> str:
    lines = [f"# Paired evaluation: {rep['name']}", "",
             f"Seeds {rep['seeds_spec']} ({rep['n_seeds_expected'] or 'as found'}), gross drop = lower 80% bound "
             f"< {100 * rep['gross_drop']:+.0f} pts (DP10). Generated {rep['generated']}.", "",
             "## Arms", "",
             "| arm | n | successes | rate % | Wilson 95% | falls | safety faults | starve faults | trunc | missing |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for a in rep["arms"]:
        lo, hi = a["wilson95"]
        lines.append(f"| {a['arm']} | {a['n']} | {a['successes']} | {_pct(a['rate'])} | [{_pct(lo)}, {_pct(hi)}] | "
                     f"{a['falls']} | {a['safety_faults']} | {a['starve_faults']} | {a['truncs']} | "
                     f"{len(a['missing_seeds'])} |")
    lines += ["", "## Paired comparisons (new - ref over common seeds, pts)", "",
              "| new - ref | gated | pairs | b new-only | c ref-only | delta | 95% CI | lower 80% | McNemar p | "
              "falls new/ref | verdict |",
              "|---|---|---:|---:|---:|---:|---|---:|---:|---|---|"]
    for c in rep["comparisons"]:
        lo, hi = c["ci95"]
        p = "-" if c["p_mcnemar"] is None else f"{c['p_mcnemar']:.3g}"
        lines.append(f"| {c['new']} - {c['ref']} | {'yes' if c['gate'] else 'no'} | {c['n']} | {c['b']} | {c['c']} | "
                     f"{_pct(c['delta'], True)} | [{_pct(lo, True)}, {_pct(hi, True)}] | {_pct(c['lower80'], True)} | "
                     f"{p} | {c['falls_new']}/{c['falls_ref']}{'' if c['falls_ok'] else ' (> +2)'} | "
                     f"**{c['verdict']}** |")
    tts = [c for c in rep["comparisons"] if c["time_to_success"].get("n")]
    if tts:
        lines += ["", "## Time to success (ticks, failures censored at max_ticks; secondary)", "",
                  "| new - ref | RMST new | RMST ref | diff | 95% bootstrap CI | log-rank p |",
                  "|---|---:|---:|---:|---|---:|"]
        for c in tts:
            t = c["time_to_success"]
            lo, hi = t["rmst_diff_ci95_boot"]
            p = t["logrank"]["p"]
            lines.append(f"| {c['new']} - {c['ref']} | {t['rmst_new']:.0f} | {t['rmst_ref']:.0f} | "
                         f"{t['rmst_diff']:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {'-' if p is None else f'{p:.3g}'} |")
    if rep.get("job_failures"):
        lines += ["", "## Job failures", ""] + [f"- {j['id']}: rc={j['returncode']}, log {j['log']}"
                                                for j in rep["job_failures"]]
    lines += ["", f"Verdict: **{rep['verdict']}**", ""]
    return "\n".join(lines)


def build_report(name: str, arm_dirs: dict, comparisons: list[dict], expected: list[int] | None, seeds_spec: str,
                 gross_drop: float, max_ticks: int, jobs: list[dict] | None = None) -> dict:
    arms = {a: load_arm([Path(d) for d in dirs]) for a, dirs in arm_dirs.items()}
    rep = {"name": name, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seeds_spec": seeds_spec,
           "n_seeds_expected": len(expected) if expected else None, "gross_drop": gross_drop, "max_ticks": max_ticks,
           "arms": [arm_summary(a, arms[a], expected) for a in arm_dirs],
           "comparisons": [compare(c["new"], c["ref"], arms, c["gate"], gross_drop, max_ticks) for c in comparisons]}
    failures = [j for j in jobs or [] if j["returncode"] != 0 or not j["run_dirs"]]
    rep["job_failures"] = failures
    empty = [a["arm"] for a in rep["arms"] if a["n"] == 0]  # missing even when no seed list was given
    missing = any(a["missing_seeds"] for a in rep["arms"]) or bool(empty)
    nodata = [c for c in rep["comparisons"] if c["gate"] and not c["n"]]
    gross = [c for c in rep["comparisons"] if c["gate"] and c["verdict"] == "gross drop"]
    if gross:
        rep["verdict"] = "FAIL: gross drop in " + ", ".join(f"{c['new']} - {c['ref']}" for c in gross)
    elif failures or missing or nodata:
        why = []
        if failures:
            why.append("loop job failures")
        if missing:
            why.append("missing seeds" + (f" (no episodes: {', '.join(empty)})" if empty else ""))
        if nodata:
            why.append("no paired seeds in " + ", ".join(f"{c['new']} - {c['ref']}" for c in nodata))
        rep["verdict"] = "INCOMPLETE: " + "; ".join(why)
    else:
        rep["verdict"] = "PASS: no gross drop" if any(c["gate"] for c in rep["comparisons"]) else "REPORTED"
    rep["exit_code"] = 1 if gross else 3 if (failures or missing or nodata) else 0
    return rep


def write_report(out: Path, rep: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(jsonable(rep), indent=1, allow_nan=False) + "\n")
    (out / "report.md").write_text(report_md(rep))


def parse_report_only(items: list[str]) -> dict:
    arm_dirs: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--report-only wants ARM=DIR[,DIR...], got {item!r}")
        arm, dirs = item.split("=", 1)
        for d in dirs.split(","):
            if not Path(d).is_dir():
                raise ValueError(f"{arm}: {d} is not a directory")
            arm_dirs.setdefault(arm, []).append(d)
    return arm_dirs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--matrix", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report-only", nargs="+", metavar="ARM=DIR[,DIR...]")
    ap.add_argument("--compare", action="append", metavar="NEW,REF", help="repeatable; overrides the matrix's")
    ap.add_argument("--seeds", help="override the matrix's seeds")
    ap.add_argument("--procs", type=int)
    ap.add_argument("--block", type=int)
    ap.add_argument("--gross-drop", type=float)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the loop commands and exit")
    ap.add_argument("--loop-cmd", help="replaces `simpy -m rtloop.loop` (tests, other interpreters)")
    args = ap.parse_args(argv)
    if __name__ == "__main__":  # the CLI: SIGTERM takes the same cleanup path as Ctrl-C
        signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        mx = load_matrix(args.matrix) if args.matrix else None
        if mx is None and not args.report_only:
            raise ValueError("--matrix is required unless --report-only is given")
        seeds_spec = args.seeds or (mx["seeds"] if mx else None)
        expected = parse_seeds(seeds_spec) if seeds_spec is not None else None
        gross_drop = args.gross_drop if args.gross_drop is not None else float((mx or {}).get("gross_drop", GROSS_DROP))
        max_ticks = int((mx or {}).get("max_ticks", MAX_EPISODE_TICKS))
        name = mx["name"] if mx else "report"
        out = args.out or RUNS / f"eval_{name}{'' if mx else time.strftime('_%Y%m%d_%H%M%S')}"
        jobs = None
        if args.report_only:
            arm_dirs = parse_report_only(args.report_only)
        else:
            loop_cmd = shlex.split(args.loop_cmd) if args.loop_cmd else DEFAULT_LOOP_CMD
            block = args.block if args.block is not None else int(mx.get("block", 10))
            plan = plan_jobs(mx, out.resolve(), loop_cmd, expected, block)
            if args.dry_run:
                for j in plan:
                    print(f"{j['id']}: {RUNS_ENV}={j['root']} {shlex.join(j['cmd'])}")
                return 0
            procs = args.procs or int(mx.get("procs", 1))
            if procs > 1 and any("--mode rt" in f"{mx.get('common_args', '')} {a['args']}" for a in mx["arms"]):
                print(f"eval_paired: warning: RT arms with procs={procs}; RT timing gates assume one loop at a time",
                      file=sys.stderr)
            jobs = run_jobs(plan, procs, args.resume)
            arm_dirs = {a["name"]: [d for j in jobs if j["arm"] == a["name"] for d in j["run_dirs"]]
                        for a in mx["arms"]}
        names = list(arm_dirs)
        if args.compare:
            comparisons = [normalize_compare(c, names) for c in args.compare]
        elif mx and mx["compare"]:
            comparisons = [c for c in mx["compare"] if c["new"] in arm_dirs and c["ref"] in arm_dirs]
        else:
            comparisons = [{"new": a, "ref": names[0], "gate": True} for a in names[1:]]
    except (ValueError, OSError) as e:
        print(f"eval_paired: error: {e}", file=sys.stderr)
        return 2
    rep = build_report(name, arm_dirs, comparisons, expected, str(seeds_spec or "as found"),
                       gross_drop, max_ticks, jobs)
    write_report(out, rep)
    print(report_md(rep))
    print(f"report: {out / 'report.md'}")
    return rep["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
