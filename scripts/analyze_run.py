#!/usr/bin/env python
"""Apply one gates.yaml gate set to a run dir and plot its timing (PLAN.md §4.1, §4.2).

    simpy $RT/scripts/analyze_run.py RUN_DIR [--gates m1_rt] [--gates-file $RT/gates.yaml] [--no-plots]
                                     [--x280-ticks PATH]

Reads ticks.npy, requests.npz (or requests.jsonl), the jsonl-only request fields (server stamps, drain),
episodes.jsonl, config.json (args.viewer, args.wbc) and summary.json (interrupted) through rtloop.telemetry,
computes every metric a gate can name, writes RUN_DIR/verdict.json ({"gates", "pass", "checks": [{name, value,
threshold, pass, status}], "metrics", ...}) and RUN_DIR/plots/*.png, prints a compact table, and exits 0 on PASS,
1 on FAIL, 2 on bad input (missing run, unknown gate set, or a gate key this script does not know: an unknown key
must never pass silently).

A check is "pass", "fail" or "n/a" (pass = null: the metric does not apply to this run, e.g. the ORT shadow of an
in-process --wbc ort run). Missing data is never a pass: a gated metric without data fails ("no data"), and every
verdict carries a `run_data_complete` check that fails when episodes.jsonl or the RequestLog is empty or has fewer
episodes than the TickLog (one fewer is allowed when summary.json says the run was interrupted).

Measurement rules (fixed here so every milestone reads the same numbers):
- The TickLog holds in-episode ticks only, so everything is outside the reset windows; tick 0 is included.
- Period = difference of consecutive t_start_ns within an episode (the interval that starts at tick 0 counts),
  except the interval that contains an operator pause (HUD key p): the loop logs period_us = NaN on the first tick
  after a pause, and that interval is left out (counted as operator_pauses). A pause is never a slip or overrun.
- Overrun = compute_us > 20 ms. Slips = sum of the slip flags.
- viewer_k0 = the worst viewer_us over the k = 0 ticks; viewer_p99 over all ticks. Only for runs with the viewer on
  (config.json args.viewer); without it viewer_us is 0 by construction, so the viewer gates fail with "no data".
- WBC RTT stats use wbc_rtt_us of the non-ORT rows (ORT rows are the fallback, not the remote path), or of all
  rows when the run is ORT-only. Fallback fraction is over all ticks. "over20_or_timeout" counts remote calls
  with RTT > 20 ms plus fallback ticks, over the ticks that attempted a remote call.
- Shadow (D16): n/a when the run had no remote WBC (config args.wbc == "ort", or, without a config, no remote or
  fallback tick); with a remote WBC and no finite shadow_da the shadow gate fails ("no data").
- d stats: APPLIED requests except each episode's pre-roll chunk (k_obs == 0); integer percentiles use the
  "higher" order statistic, so a reported p99 is an observed d.
- In-flight max: the most requests whose [t_send_ns, end) intervals overlap, end = t_recv_ns, except that a request
  the client abandoned (TIMEOUT, or DRAINED with a socket re-create) stays in flight while the server kept the next
  request it answered waiting: a single-threaded REP server cannot take request N+1 before it is done with N, so
  when N+1 waited at the server (srv_t_recv_ns - t_send_ns) more than QUEUE_TOL_MS beyond the run's minimum wait
  (the transfer time), N is held in flight until N+1 was taken (srv_t_recv_ns minus that transfer time). None
  (fails) when no request has both stamps. A request sent while the server was down and taken when it came back
  is counted the same way (conservative).
- VLA latency (gated, vla_p50_ms): the server's service time of the non-pre-roll requests, srv_t_reply_ns -
  srv_t_recv_ns (both CLOCK_MONOTONIC on the server), else srv_infer_ms. The client-seen RTT (t_recv_ns -
  t_send_ns, vla_rtt_*) is reported but not gated: replies are polled once per tick, so it is quantised to 20 ms
  (a 60 ms server reads about 80 ms). vla_queue_* = srv_t_recv_ns - t_send_ns (time before the server took it).
- SAFETY_FAULT count: episodes that end SAFETY_FAULT in episodes.jsonl, or that have a TickLog row with `safety`
  set, whichever is larger (a run killed after the fault has no episode record).
- An operator reset (HUD key r) ends an episode without an EpisodeLog record; run_data_complete allows as many
  missing records as summary.json control.operator_resets says.
- Starvation (m2/m3): starved_frac = starved ticks / ticks; starved_run_max = the longest run of consecutive starved
  ticks within one episode. vla_errors = requests with outcome ERROR (server error replies).
- Shadow p99.9 (T-3.6): over the finite shadow_da of all ticks; n/a like the p99. A remote shadow (--shadow URL)
  counts as a shadow even with --wbc ort; its misses are NaN and are counted in shadow_miss_frac.
- wbc_x280_frac: ticks whose WBC action came from the X280 or the X280 sim (wbc_be 2 or 3), over all ticks.
- wbc_visible_wait_p99_ms (DP11 with early dispatch, WP-3.8b): p99 of wbc_rtt_us over the ticks with wbc_early = 1;
  the early-dispatch client logs the wait it could not hide there. No such ticks = no data.
- x280_heartbeat_continuity (T-3.7): 1 - (x280d ring rows with status 3 = heartbeat stall) / (ring rows), over the
  rows whose t_recv_ns falls inside the run's [first t_start_ns, last t_end_ns] (both CLOCK_MONOTONIC on this host).
  The ring is x280d's x280d_ticks.npy, from --x280-ticks, else RUN_DIR/x280d_ticks.npy, else
  $CHIP_SESSION_DIR/x280d_ticks.npy (as recorded in config.json, then as set now). A ring that does not cover the
  run (it wrapped: 65,536 calls, about 22 min at 50 Hz; or it was dumped before the run ended) is no data, never a
  partial pass.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RT = Path(__file__).resolve().parents[1]
if str(RT) not in sys.path:
    sys.path.insert(0, str(RT))

from rtloop.schema import TICK_NS, EpisodeEnd, Outcome, WbcBackend  # noqa: E402
from rtloop.telemetry import (jsonable, load_episodes, load_request_extras, load_requests,  # noqa: E402
                              load_ticks)

GATES_FILE = RT / "gates.yaml"
OVERRUN_US = TICK_NS / 1e3  # compute > one tick
QUEUE_TOL_MS = 5.0  # server-side wait beyond the transfer time that means "queued behind another request"
X280_ST_STALL = 3  # x280/x280d.py ST_STALL: a reply refused because the hart's heartbeat stalled > 5 ms
X280_RING = "x280d_ticks.npy"

# gate key -> (metric, comparison). "le": value <= threshold; "ge": value >= threshold; "in": lo <= value <= hi.
GATE_KEYS = {
    "period_p99_ms": ("period_p99_ms", "le"),
    "period_max_ms": ("period_max_ms", "le"),
    "overrun_frac_max": ("overrun_frac", "le"),
    "slips_max": ("slips", "le"),
    "viewer_k0_ms_max": ("viewer_k0_ms", "le"),
    "viewer_p99_ms_max": ("viewer_p99_ms", "le"),
    "wbc_rtt_p50_ms": ("wbc_rtt_p50_ms", "le"),
    "wbc_rtt_p99_ms": ("wbc_rtt_p99_ms", "le"),
    "wbc_rtt_over20_frac_max": ("wbc_rtt_over20_frac", "le"),
    "wbc_rtt_over20_or_timeout_frac_min": ("wbc_rtt_over20_or_timeout_frac", "ge"),
    "wbc_fallback_frac_max": ("wbc_fallback_frac", "le"),
    "shadow_da_p99_max": ("shadow_da_p99", "le"),
    "shadow_da_p999_max": ("shadow_da_p999", "le"),
    "d_p50_in": ("d_p50", "in"),
    "d_p99_max": ("d_p99", "le"),
    "inflight_max": ("inflight_max", "le"),
    "starved_ticks_max": ("starved_ticks", "le"),
    "starved_frac_max": ("starved_frac", "le"),
    "starved_run_max": ("starved_run_max", "le"),
    "safety_faults_max": ("safety_faults", "le"),
    "vla_p50_ms_max": ("vla_p50_ms", "le"),
    "vla_queue_p99_ms_max": ("vla_queue_p99_ms", "le"),
    "vla_timeouts_max": ("vla_timeouts", "le"),
    "vla_errors_max": ("vla_errors", "le"),
    "wbc_x280_frac_min": ("wbc_x280_frac", "ge"),
    "wbc_visible_wait_p99_ms": ("wbc_visible_wait_p99_ms", "le"),
    "x280_heartbeat_continuity_min": ("x280_heartbeat_continuity", "ge"),
}
OPS = {"le": "<=", "ge": ">=", "in": "in"}


def _pct(x, q, method: str = "linear"):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q, method=method)) if x.size else None


def _max(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.max()) if x.size else None


def _frac(mask):
    mask = np.asarray(mask, dtype=bool)
    return float(mask.mean()) if mask.size else None


def _ms(v):
    return None if v is None else v / 1e3


def _counts(enum_cls, values) -> dict:
    out = {}
    for v, n in zip(*np.unique(values, return_counts=True)):
        try:
            name = enum_cls(int(v)).name
        except ValueError:
            name = str(int(v))
        out[name] = int(n)
    return out


def end_name(v) -> str:
    """Episode `end` as an EpisodeEnd name, whether it was logged as a name or an int."""
    if isinstance(v, str):
        return v.upper()
    try:
        return EpisodeEnd(int(v)).name
    except (TypeError, ValueError):
        return str(v)


def episode_groups(ticks: np.ndarray):
    """(ep, rows sorted by k) per episode."""
    for ep in np.unique(ticks["ep"]):
        t = ticks[ticks["ep"] == ep]
        yield int(ep), t[np.argsort(t["k"], kind="stable")]


def pause_marks(t: np.ndarray) -> np.ndarray:
    """Per row: the first tick after an operator pause (the loop logs period_us = NaN there, k > 0)."""
    return (t["k"] > 0) & np.isnan(t["period_us"].astype(np.float64))


def periods_ms(ticks: np.ndarray) -> np.ndarray:
    out = []
    for _, t in episode_groups(ticks):
        dk = np.diff(t["k"])
        dt = np.diff(t["t_start_ns"]).astype(np.float64) / 1e6
        out.append(dt[(dk == 1) & ~pause_marks(t)[1:]])
    return np.concatenate(out) if out else np.zeros(0)


def _run_lengths(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _srv_stamp(extras: dict, seq: int, key: str):
    v = (extras.get(int(seq)) or {}).get(key)
    return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def _abandoned(q, extras: dict) -> bool:
    """The client gave up on it (or drained it by re-creating the socket): no reply ever came back."""
    if int(q["outcome"]) == Outcome.TIMEOUT:
        return True
    return int(q["outcome"]) == Outcome.DRAINED and (extras.get(int(q["seq"])) or {}).get("drain") == "recreate"


def inflight_max(reqs: np.ndarray, extras: dict | None = None) -> int | None:
    """Most requests in flight at once (see the module docstring); None without data."""
    extras = extras or {}
    rows = reqs[np.argsort(reqs["t_send_ns"], kind="stable")]
    srv_recv = [_srv_stamp(extras, q["seq"], "srv_t_recv_ns") for q in rows]
    waits = [r - int(q["t_send_ns"]) for q, r in zip(rows, srv_recv) if r is not None and q["t_send_ns"] > 0]
    transfer = min(waits) if waits else 0
    starts, ends = [], []
    for j, q in enumerate(rows):
        t_send, t_recv = int(q["t_send_ns"]), int(q["t_recv_ns"])
        if t_send <= 0:
            continue
        end = t_recv if t_recv > 0 else None
        if _abandoned(q, extras):
            nxt = next((i for i in range(j + 1, len(rows)) if srv_recv[i] is not None), None)
            if nxt is not None and srv_recv[nxt] - int(rows[nxt]["t_send_ns"]) > transfer + QUEUE_TOL_MS * 1e6:
                held = srv_recv[nxt] - transfer  # the server was still busy with this one (or its successors)
                end = held if end is None else max(end, held)
        if end is None:
            continue
        starts.append(t_send)
        ends.append(end)
    if not starts:
        return None
    ev = np.concatenate([np.stack([np.asarray(starts, np.int64), np.ones(len(starts), np.int64)], 1),
                         np.stack([np.asarray(ends, np.int64), -np.ones(len(ends), np.int64)], 1)])
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]  # by time; a reply at t closes before a send at t opens
    return int(np.cumsum(ev[:, 1]).max())


def vla_server_ms(reqs: np.ndarray, extras: dict | None = None) -> np.ndarray:
    """Per request: the server's service time (srv_t_reply_ns - srv_t_recv_ns), else srv_infer_ms, else NaN."""
    extras = extras or {}
    out = np.full(len(reqs), np.nan)
    for i, q in enumerate(reqs):
        a, b = _srv_stamp(extras, q["seq"], "srv_t_recv_ns"), _srv_stamp(extras, q["seq"], "srv_t_reply_ns")
        if a is not None and b is not None and b >= a:
            out[i] = (b - a) / 1e6
        elif float(q["srv_infer_ms"]) > 0:
            out[i] = float(q["srv_infer_ms"])
    return out


def vla_queue_ms(reqs: np.ndarray, extras: dict | None = None) -> np.ndarray:
    """Per request: srv_t_recv_ns - t_send_ns (how long it waited before the server took it), else NaN."""
    extras = extras or {}
    out = np.full(len(reqs), np.nan)
    for i, q in enumerate(reqs):
        a = _srv_stamp(extras, q["seq"], "srv_t_recv_ns")
        if a is not None and int(q["t_send_ns"]) > 0:
            out[i] = (a - int(q["t_send_ns"])) / 1e6
    return out


def load_x280_ring(run: Path, info: dict, path: Path | None = None) -> tuple[np.ndarray | None, str]:
    """(x280d tick ring, where it came from) or (None, why not); see the module docstring for the search order."""
    cands = [Path(path)] if path else [run / X280_RING]
    if not path:
        for d in (info.get("chip_session_dir"), os.environ.get("CHIP_SESSION_DIR")):
            if d:
                cands.append(Path(d) / X280_RING)
    for c in cands:
        if c.is_file():
            try:
                ring = np.load(c, allow_pickle=False)
            except (OSError, ValueError) as e:
                return None, f"{c}: {e}"
            if ring.dtype.names is None or not {"t_recv_ns", "status"} <= set(ring.dtype.names):
                return None, f"{c}: not an x280d tick ring (fields {ring.dtype.names})"
            return ring, str(c)
    return None, "no x280d ring (" + ", ".join(map(str, cands)) + ")"


def heartbeat_continuity(ticks: np.ndarray, ring: np.ndarray | None) -> tuple[float | None, dict]:
    """1 - stalls / rows of the x280d ring inside the run's time window; None (with the reason) without data."""
    if ring is None or not len(ticks):
        return None, {"why": "no x280d ring"}
    t0, t1 = int(ticks["t_start_ns"].min()), int(ticks["t_end_ns"].max())
    tr = ring["t_recv_ns"].astype(np.int64)
    if not len(tr) or tr.min() > t0 + 1_000_000_000:
        return None, {"why": "the x280d ring does not reach back to the run's start (it wrapped)",
                      "ring_rows": int(len(tr))}
    if tr.max() < t1 - 1_000_000_000:
        return None, {"why": "the x280d ring ends before the run did (dumped before the run ended)",
                      "ring_rows": int(len(tr))}
    inside = (tr >= t0) & (tr <= t1)
    n = int(inside.sum())
    if not n:
        return None, {"why": "no x280d ring rows inside the run's time window"}
    stalls = int((ring["status"][inside] == X280_ST_STALL).sum())
    return 1.0 - stalls / n, {"rows": n, "stalls": stalls}


def compute_metrics(ticks: np.ndarray, reqs: np.ndarray, eps: list[dict], extras: dict | None = None,
                    run_info: dict | None = None, x280_ring: np.ndarray | None = None) -> dict:
    """Every gateable metric. `extras` = load_request_extras(run); `run_info` = {viewer, wbc, shadow, interrupted,
    operator_resets} from config.json / summary.json (None = unknown); `x280_ring` = load_x280_ring(...)[0]. A metric
    that does not apply goes to m["na"] with the reason; a metric left None for a known reason gets it in
    m["no_data"]."""
    extras, run_info = extras or {}, run_info or {}
    m: dict = {"n_ticks": int(len(ticks)), "n_episodes_ticklog": int(len(np.unique(ticks["ep"]))),
               "n_episodes": len(eps), "n_requests": int(len(reqs)), "na": {}, "no_data": {}}

    per = periods_ms(ticks)
    m["operator_pauses"] = sum(int(pause_marks(t).sum()) for _, t in episode_groups(ticks))
    m.update(period_p50_ms=_pct(per, 50), period_p99_ms=_pct(per, 99), period_max_ms=_max(per),
             period_std_ms=float(per.std()) if per.size else None)
    comp = ticks["compute_us"].astype(np.float64)
    m.update(compute_p50_ms=_ms(_pct(comp, 50)), compute_p99_ms=_ms(_pct(comp, 99)), compute_max_ms=_ms(_max(comp)),
             overrun_n=int((comp > OVERRUN_US).sum()), overrun_frac=_frac(comp > OVERRUN_US),
             late_p99_ms=_ms(_pct(ticks["late_us"], 99)), late_max_ms=_ms(_max(ticks["late_us"])),
             slips=int(ticks["slip"].sum()),
             sim_p50_ms=_ms(_pct(ticks["sim_us"], 50)), sim_p99_ms=_ms(_pct(ticks["sim_us"], 99)))
    k0 = ticks["k"] == 0
    m["viewer_on"] = run_info.get("viewer") is True
    if m["viewer_on"]:
        m.update(viewer_k0_ms=_ms(_max(ticks["viewer_us"][k0])) if k0.any() else None,
                 viewer_p99_ms=_ms(_pct(ticks["viewer_us"], 99)))
    else:  # viewer_us is 0 by construction without --viewer: a 0 ms "pass" would be vacuous
        m.update(viewer_k0_ms=None, viewer_p99_ms=None)
        for key in ("viewer_k0_ms", "viewer_p99_ms"):
            m["no_data"][key] = "viewer off (config.json args.viewer)"

    be, fb, rtt = ticks["wbc_be"], ticks["wbc_fb"].astype(bool), ticks["wbc_rtt_us"].astype(np.float64)
    remote = be != WbcBackend.ORT
    sel = remote if remote.any() else np.ones(len(ticks), bool)
    attempted = remote | fb if (remote | fb).any() else np.ones(len(ticks), bool)
    m.update(wbc_ort_only=bool(not remote.any()),
             wbc_backend_counts=_counts(WbcBackend, be),
             wbc_rtt_p50_ms=_ms(_pct(rtt[sel], 50)), wbc_rtt_p99_ms=_ms(_pct(rtt[sel], 99)),
             wbc_rtt_max_ms=_ms(_max(rtt[sel])),
             wbc_rtt_over20_frac=_frac(rtt[sel] > OVERRUN_US),
             wbc_rtt_over20_or_timeout_frac=_frac((((rtt > OVERRUN_US) & remote) | fb)[attempted]),
             wbc_fallback_frac=_frac(fb), wbc_fallback_n=int(fb.sum()),
             wbc_dev_p50_ms=_ms(_pct(ticks["wbc_dev_us"][sel], 50)),
             wbc_x280_frac=_frac(np.isin(be, (WbcBackend.X280, WbcBackend.X280_SIM))),
             shadow_da_p99=_pct(ticks["shadow_da"], 99), shadow_da_p999=_pct(ticks["shadow_da"], 99.9),
             shadow_da_max=_max(ticks["shadow_da"]))
    early = ticks["wbc_early"].astype(bool)
    m["wbc_visible_wait_p99_ms"] = _ms(_pct(rtt[early], 99)) if early.any() else None
    if not early.any():
        m["no_data"]["wbc_visible_wait_p99_ms"] = "no early-dispatch ticks (wbc_early = 1; WP-3.8b)"
    wbc_arg = run_info.get("wbc")
    m["wbc_remote"] = (wbc_arg != "ort") if wbc_arg is not None else bool((remote | fb).any())
    m["shadow_remote"] = bool(run_info.get("shadow"))  # --shadow URL behind the ORT primary (T-3.6)
    shadow_keys = ("shadow_da_p99", "shadow_da_p999", "shadow_da_max")
    if not (m["wbc_remote"] or m["shadow_remote"]):  # in-process ORT is its own reference: nothing to shadow
        m.update(dict.fromkeys(shadow_keys))
        for key in shadow_keys:
            m["na"][key] = "no shadow (--wbc ort)"
    elif m["shadow_da_p99"] is None:
        for key in shadow_keys:
            m["no_data"][key] = "a shadow was configured but no shadow_da was logged"
    m["shadow_miss_frac"] = _frac(np.isnan(ticks["shadow_da"].astype(np.float64))) if m["shadow_remote"] else None

    starved = ticks["starved"].astype(bool)
    flips, runs = 0, 0
    for _, t in episode_groups(ticks):
        flips += int((np.diff(t["policy"].astype(np.int16)) != 0).sum())
        runs = max(runs, _run_lengths(t["starved"].astype(bool)))
    with_row = ticks["row"] >= 0
    splice = ticks["splice"].astype(bool)
    m.update(starved_ticks=int(starved.sum()), starved_frac=_frac(starved), starved_run_max=runs,
             policy_flips=flips, chunk_age_mean=float(ticks["age"][with_row].mean()) if with_row.any() else None,
             boundary_jump_p99=_pct(ticks["jump_rad"][splice], 99), n_splices=int(splice.sum()))

    preroll = reqs["k_obs"] == 0
    applied = reqs["outcome"] == Outcome.APPLIED
    d = reqs["d"][applied & ~preroll].astype(np.float64)
    m.update(n_applied=int(applied.sum()), d_p50=_pct(d, 50, "higher"), d_p99=_pct(d, 99, "higher"),
             d_max=_max(d), d_hat_max=int(reqs["d_hat_max"].max()) if len(reqs) else None,
             inflight_max=inflight_max(reqs, extras),
             outcome_counts=_counts(Outcome, reqs["outcome"]),
             vla_timeouts=int((reqs["outcome"] == Outcome.TIMEOUT).sum()),
             vla_errors=int((reqs["outcome"] == Outcome.ERROR).sum()))
    if m["inflight_max"] is None:
        m["no_data"]["inflight_max"] = "no request with send and receive stamps"
    has_reply = (reqs["t_send_ns"] > 0) & (reqs["t_recv_ns"] > reqs["t_send_ns"]) & ~preroll
    vla_rtt = (reqs["t_recv_ns"][has_reply] - reqs["t_send_ns"][has_reply]).astype(np.float64) / 1e6
    srv = vla_server_ms(reqs, extras)[has_reply]
    queue = vla_queue_ms(reqs, extras)[~preroll]
    m.update(vla_p50_ms=_pct(srv, 50), vla_p99_ms=_pct(srv, 99), vla_max_ms=_max(srv),
             vla_rtt_p50_ms=_pct(vla_rtt, 50), vla_rtt_p99_ms=_pct(vla_rtt, 99), vla_rtt_max_ms=_max(vla_rtt),
             vla_srv_infer_p50_ms=_pct(reqs["srv_infer_ms"][has_reply], 50),
             vla_queue_p50_ms=_pct(queue, 50), vla_queue_p99_ms=_pct(queue, 99), vla_queue_max_ms=_max(queue))
    if m["vla_p50_ms"] is None and vla_rtt.size:
        m["no_data"]["vla_p50_ms"] = "the VLA server reports no timings (stock server?)"

    ends = [end_name(e.get("end")) for e in eps]
    tick_faults = sum(int(t["safety"].any()) for _, t in episode_groups(ticks))
    m.update(safety_faults=max(ends.count("SAFETY_FAULT"), tick_faults), falls=ends.count("FALL"),
             successes=ends.count("SUCCESS"), truncs=ends.count("TRUNC"),
             starve_fault_episodes=sum(bool(e.get("starve_fault")) for e in eps))
    m["x280_heartbeat_continuity"], m["x280_heartbeat"] = heartbeat_continuity(ticks, x280_ring)
    if m["x280_heartbeat_continuity"] is None:
        m["no_data"]["x280_heartbeat_continuity"] = m["x280_heartbeat"]["why"]
    m["data_problems"] = data_problems(m, run_info)
    return m


def data_problems(m: dict, run_info: dict) -> list[str]:
    """Why the run's logs cannot support a verdict (empty = complete)."""
    out = []
    if m["n_ticks"] == 0:
        out.append("no ticks")
    if m["n_episodes"] == 0:
        out.append("no episodes in episodes.jsonl")
    if m["n_requests"] == 0:
        out.append("no requests in the RequestLog")
    missing = m["n_episodes_ticklog"] - m["n_episodes"]
    allowed = (1 if run_info.get("interrupted") is True else 0) + int(run_info.get("operator_resets") or 0)
    if missing > allowed:
        out.append(f"{m['n_episodes_ticklog']} episodes in the TickLog but {m['n_episodes']} in episodes.jsonl")
    return out


def load_gates(path: Path, name: str) -> dict:
    import yaml

    try:
        sets = yaml.safe_load(Path(path).read_text()) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: {e}") from e
    if name not in sets:
        raise ValueError(f"gate set {name!r} not in {path} (have: {', '.join(sets)})")
    gates = sets[name]
    if not isinstance(gates, dict) or not gates:
        raise ValueError(f"gate set {name!r} in {path} has no checks")
    unknown = [k for k in gates if k not in GATE_KEYS]
    if unknown:
        raise ValueError(f"unknown gate key(s) in {name!r}: {', '.join(unknown)} (known: {', '.join(GATE_KEYS)})")
    for key, thr in gates.items():
        if GATE_KEYS[key][1] == "in":
            if not (isinstance(thr, (list, tuple)) and len(thr) == 2):
                raise ValueError(f"gate {key!r} needs [lo, hi], got {thr!r}")
        elif not isinstance(thr, (int, float)) or isinstance(thr, bool):
            raise ValueError(f"gate {key!r} needs a number, got {thr!r}")
    return gates


def apply_gates(metrics: dict, gates: dict) -> list[dict]:
    """One check per gate key: pass, fail, or n/a (pass None) when metrics["na"] says the metric does not apply.
    A metric with no data (None) fails."""
    checks = []
    na, why = metrics.get("na") or {}, metrics.get("no_data") or {}
    for key, thr in gates.items():
        if key not in GATE_KEYS:
            raise ValueError(f"unknown gate key {key!r}")
        metric, op = GATE_KEYS[key]
        v = metrics[metric]
        check = {"name": key, "metric": metric, "op": OPS[op], "value": v, "threshold": thr}
        if metric in na:
            check.update(value=None, threshold=thr, status="n/a", note=na[metric])
            check["pass"] = None
            checks.append(check)
            continue
        if v is None:
            ok = False
            check["note"] = why.get(metric, "no data")
        elif op == "le":
            ok = v <= thr
        elif op == "ge":
            ok = v >= thr
        else:
            ok = thr[0] <= v <= thr[1]
        check.update(status="pass" if ok else "fail")
        check["pass"] = bool(ok)
        checks.append(check)
    return checks


def data_check(metrics: dict) -> dict:
    """The check every verdict carries: the logs are complete enough to judge the run at all."""
    probs = metrics["data_problems"]
    return {"name": "run_data_complete", "metric": "data_problems", "op": "==", "value": probs or "ok",
            "threshold": "ok", "pass": not probs, "status": "fail" if probs else "pass",
            **({"note": "; ".join(probs)} if probs else {})}


def verdict_pass(checks: list[dict]) -> bool:
    """No failed check, and at least one check that actually passed (n/a alone is not a pass)."""
    return not any(c["pass"] is False for c in checks) and any(c["pass"] is True for c in checks)


def _fmt(v) -> str:
    if v is None:
        return "no data"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-2 and v != 0 else f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def table(run: Path, name: str, metrics: dict, checks: list[dict]) -> str:
    lines = [f"run {run}  gates {name}  ticks {metrics['n_ticks']}  episodes {metrics['n_episodes']}  "
             f"requests {metrics['n_requests']}",
             f"{'check':<36} {'value':>10}   {'threshold':<12} result"]
    for c in checks:
        thr = f"{c['op']} {c['threshold']}"
        value = "n/a" if c["pass"] is None else ("no data" if c["value"] is None else
                                                  "problems" if isinstance(c["value"], list) else _fmt(c["value"]))
        result = {True: "PASS", False: "FAIL", None: "n/a"}[c["pass"]]
        note = f"  ({c['note']})" if c.get("note") else ""
        lines.append(f"{c['name']:<36} {value:>10}   {thr:<12} {result}{note}")
    n_ok = sum(c["pass"] is True for c in checks)
    n_na = sum(c["pass"] is None for c in checks)
    info = ("period_p50_ms", "compute_p99_ms", "wbc_rtt_p99_ms", "vla_p50_ms", "vla_rtt_p50_ms", "vla_queue_p99_ms",
            "d_p50", "chunk_age_mean", "policy_flips", "successes", "falls")
    lines.append("info: " + "  ".join(f"{k}={_fmt(metrics.get(k))}" for k in info))
    na = f", {n_na} n/a" if n_na else ""
    lines.append(f"VERDICT: {'PASS' if verdict_pass(checks) else 'FAIL'} ({n_ok}/{len(checks) - n_na} checks{na})")
    return "\n".join(lines)


# --- Plots --------------------------------------------------------------------------------------

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
CRITICAL = "#d03b3b"
OUTCOME_COLOR = {Outcome.APPLIED: SERIES[0], Outcome.LATE_REJECT: SERIES[1], Outcome.PENDING: MUTED}


def _style(ax, title: str, xlabel: str, ylabel: str = "ticks") -> None:
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def _limit(ax, x: float, label: str) -> None:
    ax.axvline(x, color=CRITICAL, linewidth=1.2, linestyle="--")
    ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"), xytext=(3, -12), textcoords="offset points",
                color=INK2, fontsize=8)


def _hist(ax, x, title, xlabel, bins=60, limits=(), integer=False) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size:
        title += f"\np50 {np.percentile(x, 50):.3g}, p99 {np.percentile(x, 99):.3g}, max {x.max():.3g} {xlabel}"
    _style(ax, title, xlabel)
    if not x.size:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=MUTED)
        return
    if integer:
        bins = np.arange(x.min() - 0.5, x.max() + 1.5, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # no bar edges: a surface-coloured edge erases bars that a far-away gate line makes narrow
    ax.hist(x, bins=bins, color=SERIES[0], linewidth=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for v, label in limits:
        if v is not None:
            _limit(ax, v, label)


def make_plots(run: Path, ticks, reqs, gates: dict) -> list[str]:
    """histograms.png, gantt_ep<first>.png, boundary_jump_cdf.png; ImportError when matplotlib is missing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    out_dir = run / "plots"
    out_dir.mkdir(exist_ok=True)
    paths = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    preroll = reqs["k_obs"] == 0
    applied = (reqs["outcome"] == Outcome.APPLIED) & ~preroll
    remote = ticks["wbc_be"] != WbcBackend.ORT
    rtt = ticks["wbc_rtt_us"][remote if remote.any() else slice(None)] / 1e3
    _hist(axes[0, 0], periods_ms(ticks), "Tick period", "ms",
          limits=((gates.get("period_p99_ms"), "p99 gate"), (gates.get("period_max_ms"), "max gate")))
    _hist(axes[0, 1], ticks["late_us"] / 1e3, "Wake-up lateness", "ms")
    _hist(axes[0, 2], ticks["compute_us"] / 1e3, "Tick compute", "ms", limits=((OVERRUN_US / 1e3, "overrun"),))
    _hist(axes[1, 0], reqs["d"][applied], "Chunk delay d (applied, no pre-roll)", "ticks", integer=True,
          limits=((gates.get("d_p99_max"), "p99 gate"),))
    _hist(axes[1, 1], ticks["age"][ticks["row"] >= 0], "Chunk age at execution", "ticks", integer=True)
    _hist(axes[1, 2], rtt, "WBC round trip" + (" (remote rows)" if remote.any() else " (ORT)"), "ms",
          limits=((gates.get("wbc_rtt_p99_ms"), "p99 gate"),))
    fig.tight_layout()
    p = out_dir / "histograms.png"
    fig.savefig(p, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    paths.append(str(p))

    if len(ticks):
        ep, t = next(episode_groups(ticks))
        r = reqs[reqs["ep"] == ep]
        fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(15, 7.5), sharex=True, facecolor=SURFACE,
                                         gridspec_kw={"height_ratios": [1, 1.4, 1.4]})
        _style(a0, f"Episode {ep}: VLA requests, k_obs to k_arr (overlapping bars = more than one in flight)",
               "", "lane")
        for j, q in enumerate(r):
            end = q["k_arr"] if q["k_arr"] >= q["k_obs"] else q["k_obs"] + 1
            color = OUTCOME_COLOR.get(int(q["outcome"]), CRITICAL)
            a0.barh(j % 2, max(end - q["k_obs"], 0.6), left=q["k_obs"], height=0.7, color=color, edgecolor=SURFACE)
        a0.set_yticks([0, 1])
        a0.set_ylim(-0.6, 1.6)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (SERIES[0], SERIES[1], CRITICAL)]
        a0.legend(handles, ["applied", "late reject", "timeout / error / dropped"], fontsize=8, frameon=False,
                  loc="upper right", ncol=3)
        _style(a1, "Tick compute (red = overrun > 20 ms)", "", "ms")
        comp = t["compute_us"] / 1e3
        a1.bar(t["k"], comp, width=1.0, color=np.where(comp > OVERRUN_US / 1e3, CRITICAL, SERIES[0]))
        a1.axhline(OVERRUN_US / 1e3, color=CRITICAL, linewidth=1, linestyle="--")
        _style(a2, "Executed row (vertical lines = splices; red = starved)", "tick k", "row")
        a2.yaxis.set_major_locator(MaxNLocator(integer=True))
        a2.step(t["k"], t["row"], where="post", color=SERIES[0], linewidth=1.5)
        for k in t["k"][t["splice"].astype(bool)]:
            a2.axvline(k, color=MUTED, linewidth=0.6)
        st = t["starved"].astype(bool)
        if st.any():
            a2.scatter(t["k"][st], t["row"][st], s=10, color=CRITICAL, zorder=3)
        fig.tight_layout()
        p = out_dir / f"gantt_ep{ep}.png"
        fig.savefig(p, dpi=110, facecolor=SURFACE)
        plt.close(fig)
        paths.append(str(p))

    jump = np.sort(ticks["jump_rad"][ticks["splice"].astype(bool)].astype(np.float64))
    if jump.size:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
        _style(ax, f"Boundary jump at splices (L2, arms + waist)\np50 {np.percentile(jump, 50):.3g}, "
                   f"p99 {np.percentile(jump, 99):.3g} rad, n {jump.size}", "rad", "fraction of splices")
        ax.step(jump, np.arange(1, jump.size + 1) / jump.size, where="post", color=SERIES[0], linewidth=2)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        p = out_dir / "boundary_jump_cdf.png"
        fig.savefig(p, dpi=110, facecolor=SURFACE)
        plt.close(fig)
        paths.append(str(p))
    return paths


def _read_json(path: Path) -> dict:
    try:
        out = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def run_info(run: Path) -> dict:
    """What the gates need to know about how the run was made: {viewer, wbc, shadow, interrupted, operator_resets,
    chip_session_dir} (None = unknown)."""
    cfg = _read_json(run / "config.json")
    args = cfg.get("args") or {}
    summary = _read_json(run / "summary.json")
    return {"viewer": args.get("viewer"), "wbc": args.get("wbc"), "shadow": args.get("shadow"),
            "interrupted": summary.get("interrupted"),
            "operator_resets": (summary.get("control") or {}).get("operator_resets"),
            "chip_session_dir": (cfg.get("env") or {}).get("CHIP_SESSION_DIR")}


def analyze(run: Path, gates_name: str = "m1_rt", gates_file: Path = GATES_FILE, plots: bool = True,
            x280_ticks: Path | None = None) -> dict:
    run = Path(run).resolve()
    if not (run / "ticks.npy").exists():
        raise ValueError(f"{run}: no ticks.npy")
    gates = load_gates(gates_file, gates_name)
    ticks, reqs, eps = load_ticks(run), load_requests(run), load_episodes(run)
    info = run_info(run)
    ring, ring_src = load_x280_ring(run, info, x280_ticks)
    metrics = compute_metrics(ticks, reqs, eps, load_request_extras(run), info, ring)
    metrics["x280_ring"] = ring_src
    checks = [data_check(metrics), *apply_gates(metrics, gates)]
    verdict = {"gates": gates_name, "pass": verdict_pass(checks), "checks": checks,
               "run": str(run), "gates_file": str(gates_file), "run_info": info, "metrics": metrics}
    if plots:
        try:
            verdict["plots"] = make_plots(run, ticks, reqs, gates)
        except ImportError as e:
            verdict["plots"] = []
            verdict["plots_note"] = f"skipped: {e}"
    else:
        verdict["plots"] = []
        verdict["plots_note"] = "skipped: --no-plots"
    (run / "verdict.json").write_text(json.dumps(jsonable(verdict), indent=1, allow_nan=False) + "\n")
    verdict["table"] = table(run, gates_name, metrics, checks)
    return verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--gates", default="m1_rt", help="gate set name in the gates file")
    ap.add_argument("--gates-file", type=Path, default=GATES_FILE)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--x280-ticks", type=Path, default=None,
                    help=f"x280d's {X280_RING} (default: RUN_DIR, then $CHIP_SESSION_DIR)")
    args = ap.parse_args(argv)
    try:
        verdict = analyze(args.run_dir, args.gates, args.gates_file, plots=not args.no_plots,
                          x280_ticks=args.x280_ticks)
    except (ValueError, OSError) as e:
        print(f"analyze_run: error: {e}", file=sys.stderr)
        return 2
    print(verdict["table"])
    if verdict.get("plots_note"):
        print(f"plots {verdict['plots_note']}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
