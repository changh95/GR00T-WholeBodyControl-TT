#!/usr/bin/env python3
"""The verdict logic of the M3 device-session jobs, kept out of their shell heredocs so a CPU test can run it
(PLAN.md WP-3.5, WP-3.8, WP-3.9; jobs/s1c.sh, jobs/s1d.sh, jobs/s5a_probe.sh, jobs/m3_shadow.sh, jobs/m3_eval.sh,
jobs/m3_faults.sh, jobs/m3_soak.sh; tests/x280/test_m3_jobs.py).

    python3 $RT/x280/tools/m3_jobs.py s1c-summary SESSION_DIR E1A_OK METHOD [k=rc ...]   # S1c + S2 go/no-go
    python3 $RT/x280/tools/m3_jobs.py s1d-summary SESSION_DIR [k=rc ...]                 # S1d: lim-init store
    python3 $RT/x280/tools/m3_jobs.py s5a-summary SESSION_DIR [k=rc ...]                 # S5a: L2CPU0-3 probe
    simpy $RT/x280/tools/m3_jobs.py m3-gates GATES_YAML OUT_YAML BENCH|none DTYPE         # T-3.7 session gate sets
    python3 $RT/x280/tools/m3_jobs.py t37-summary VERDICT BEFORE AFTER DUMP BENCH|none OUT
    python3 $RT/x280/tools/m3_jobs.py t36-gates OUT_YAML DTYPE
    python3 $RT/x280/tools/m3_jobs.py t36-summary VERDICT DTYPE LOOP_RC OUT
    simpy $RT/x280/tools/m3_jobs.py t38-summary RUN EVENTS ATTACH_LOG PARK_JSON LOOP_RC OUT  # T-3.8 (m3_faults.sh)
    python3 $RT/x280/tools/m3_jobs.py t39-gates OUT_YAML                                  # T-3.9 (jobs/m3_soak.sh)
    simpy $RT/x280/tools/m3_jobs.py t39-summary VERDICT TICKS BEFORE AFTER T0 T1 LOOP_RC OUT STATUS_JSONL...

Each prints its JSON result as the last stdout line and exits 0 (pass / written), 1 (a check failed), 2 (bad input).
Stdlib only, except m3-gates (PyYAML), t38-summary and t39-summary (numpy, for ticks.npy): run those with simpy.
Absolute paths: simpy runs from $R.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

RT = Path(__file__).resolve().parents[2]
if str(RT) not in sys.path:
    sys.path.insert(0, str(RT))

from x280 import regs  # noqa: E402

M3_SETS = ("m3_rt", "m3_rt_slow")  # gates.yaml (the merged sets; configs/gates_m3.proposed.yaml is superseded)
T23_BAND = 0.05  # T-3.7: TT VLA p50 within +-5 % of T-2.3
SHADOW_TOL = {"fp16": 1e-2, "fp32": 5e-5}  # T-3.1 / T-3.6 limits vs ORT (x280d.PARITY_TOL)
SHADOW_MISS_MAX = 0.001  # T-3.6 is a p99.9 bound: a missed shadow reply is a failed tick, so <= 0.1 % may miss
RING_CALLS = 1 << 16  # x280d.RING: the tick ring keeps the last 65,536 W-WBC calls (21.8 min at 50 Hz)
X280D_FAULTS = ("timeouts", "faults", "stalls", "interlock", "degraded")  # x280d Server.counts that fail T-3.7


def _json(path, default=None):
    """A JSON file, or the last line of a JSONL / tool stdout capture; `default` when missing or unreadable."""
    try:
        text = Path(path).read_text().strip()
    except (OSError, TypeError):
        return default
    for cand in (text, text.splitlines()[-1] if text else ""):
        try:
            return json.loads(cand)
        except ValueError:
            continue
    return default


def _hex(v) -> Optional[str]:
    return f"0x{v:x}" if isinstance(v, int) else None


# --- S1c (T-3.E0, E1a, E1b) and the go/no-go for S2 ---------------------------------------------------------------


def pll_view(e0: Optional[dict]) -> dict:
    """What boot steps 6/8 depend on, from one E0 result: kmd.pll4_step_to re-reads CNTL_0 and PLL_LOCK before every
    walk write and stops unless CNTL_0 & 0x1F == 0x12 (running, not bypassed) and PLL_LOCK bit 4 is set (review F5;
    taken from the CMFW sources, unverified on silicon until S1c records it)."""
    e0 = e0 or {}
    pll = e0.get("pll4") or {}
    c0, lock = pll.get("cntl_0"), e0.get("pll_lock")
    return {"pll4_walk_ready": e0.get("pll4_walk_ready"), "pll_lock": _hex(lock), "pll4_cntl_0": _hex(c0),
            "pll4_default": e0.get("pll4_default"),
            "cntl_0_running": (c0 & regs.PLL_CNTL_0_MASK == regs.PLL_CNTL_0_RUNNING) if isinstance(c0, int) else None,
            "pll4_locked": bool(lock & regs.pll_lock_bit(4)) if isinstance(lock, int) else None}


def s2_verdict(e0_keeper: Optional[dict], e0_tt: Optional[dict], e1a_ok: bool) -> dict:
    """GO for S2 (PLAN §5.2 E2 preconditions, the boot as x280d implements it) needs: E0 with the keeper fd passed,
    e0.pll4_walk_ready True on that read (S2 boots before any TT build, so this is the clock state the walks will
    meet), and E1a passed (boot step 2 is LIM init method A). The read with TT serving is recorded; a difference
    there is a note, since every M3 session starts x280d before server_g1 (D13)."""
    why, notes = [], []
    k, t = pll_view(e0_keeper), pll_view(e0_tt)
    if e0_keeper is None:
        why.append("E0 (keeper fd) has no result")
    else:
        if e0_keeper.get("pass") is not True:
            why.append("E0 (keeper fd) failed")
        if k["pll4_walk_ready"] is not True:
            why.append(f"e0.pll4_walk_ready={k['pll4_walk_ready']} (PLL4 CNTL_0={k['pll4_cntl_0']}, need & 0x1F == "
                       f"0x12; PLL_LOCK={k['pll_lock']}, need bit 4): the boot's PLL4 walks would stop at their first "
                       "write (exit 3, before the release: nothing spent, but no boot). Re-derive kmd's walk interlock "
                       "(regs.PLL_CNTL_0_RUNNING, pll_lock_bit) from these silicon values before S2")
    if not e1a_ok:
        why.append("E1a (LIM init method A, no TT) failed: boot step 2 is method A, so a normal S2 boot cannot run; "
                   "if E1b's method B passed, a method-B boot is not implemented; E1F needs the user's DP3(d) approval")
    if e0_tt is None:
        notes.append("E0 with TT serving has no result")
    elif t["pll4_walk_ready"] is not True:
        notes.append(f"with server_g1 serving pll4_walk_ready={t['pll4_walk_ready']} (CNTL_0={t['pll4_cntl_0']}, "
                     f"PLL_LOCK={t['pll_lock']}): boot x280d before server_g1 (D13), as chip_session does")
    return {"go": not why, "why": why, "notes": notes, "e0_keeper": k, "e0_tt": t}


ECC_COUNTS = ("dat_fix_count", "dat_fail_count", "dir_fix_count", "dir_fail_count")


def _tile_ecc(e0: Optional[dict]) -> Optional[dict]:
    """The CCACHE ECC registers of an `x280d e0` result (its first tile)."""
    tiles = (e0 or {}).get("tiles") or {}
    return next((t.get("ecc") for t in tiles.values() if isinstance(t, dict) and t.get("ecc")), None)


def _ecc_d(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    if not (isinstance(a, dict) and isinstance(b, dict)) or any(k not in a or k not in b for k in ECC_COUNTS):
        return None
    out = {k: (b[k] - a[k]) & 0xFFFF_FFFF for k in ECC_COUNTS}
    if out["dat_fail_count"]:
        out["last_fail"] = _hex(b.get("dat_fail_high", 0) << 32 | b.get("dat_fail_low", 0))
    if out["dat_fix_count"]:
        out["last_fix"] = _hex(b.get("dat_fix_high", 0) << 32 | b.get("dat_fix_low", 0))
    return out


def _check_failures(c: dict) -> list:
    """The parts of x280d lim_check's (unchanged) gate that failed: its `failed_checks`, or, for a result from before
    that key existed (S1c 2026-10-25), the same derived from the fields the gate reads."""
    if isinstance(c.get("failed_checks"), list):
        return c["failed_checks"]
    rst, way = c.get("l2cpu_reset") or [None, None], c.get("wayenable") or [None, None]
    checks = {"data_read_back": not c.get("mismatches"), "partial_writes_read_back": not c.get("partial_mismatches"),
              "ecc_fail_counts_unchanged": not c.get("ecc_fail_delta"), "l2cpu_reset_unchanged": rst[0] == rst[1],
              "wayenable_unchanged_0": way[0] == way[1] == 0, "cleanup_wrote_gone": c.get("cleanup_ok", True)}
    return [k for k, ok in checks.items() if not ok]


def s1c_ecc(d: Path) -> dict:
    """S1c's CCACHE data-ECC record, step by step from the step JSONs (S1c 2026-10-25 failed E1a/E1b on it alone):
    the counter deltas of each LIM init and check, events per line / per word read, the last event address, and per
    check which parts of its gate failed. Reporting only: the E1a/E1b verdicts are lim_check's own `pass`."""
    d = Path(d)
    e0k, e0t = _tile_ecc(_json(d / "s1c_e0_keeper.json")), _tile_ecc(_json(d / "s1c_e0_tt.json"))
    out, lines = {}, []
    prev = e0k
    for tag, label, nxt in (("e1a", "E1a", e0t), ("e1b", "E1b", None)):
        init, chk = _json(d / f"s1c_{tag}_init.json") or {}, _json(d / f"s1c_{tag}_check.json") or {}
        before = chk.get("ecc_before")
        ini = _ecc_d(init.get("ecc_before"), init.get("ecc_after")) or _ecc_d(prev, before)
        row = {"init": ini, "init_lines": init.get("lines"), "init_method": init.get("method"),
               "init_store": init.get("store", "a64" if init.get("method") == "A" else None),
               "check": _ecc_d(before, chk.get("ecc_after")), "words_read": chk.get("words_read"),
               "mismatches": chk.get("mismatches"), "check_pass": chk.get("pass"),
               "check_failed": _check_failures(chk) if chk else None}
        if ini and row["init_lines"]:
            row["init_events_per_line"] = {"fix": round(ini["dat_fix_count"] / row["init_lines"], 4),
                                           "fail": round(ini["dat_fail_count"] / row["init_lines"], 4)}
        if row["check"] and row["words_read"]:
            row["check_fail_per_word_read"] = round(row["check"]["dat_fail_count"] / row["words_read"], 4)
        out[tag] = row
        if ini:
            lines.append(f"{label} init ({row['init_method']}{'/' + row['init_store'] if row['init_store'] else ''}, "
                         f"{row['init_lines']} lines): ECC fix +{ini['dat_fix_count']} fail +{ini['dat_fail_count']}")
        if row["check"]:
            ok = "passed" if row["check_pass"] else f"failed on {', '.join(row['check_failed']) or '?'}"
            lines.append(f"{label} check: ECC fail +{row['check']['dat_fail_count']} fix +{row['check']['dat_fix_count']} "
                         f"over {row['words_read']} word reads ({row.get('check_fail_per_word_read')} per read), "
                         f"{row['mismatches']} data mismatches, last fail at {row['check'].get('last_fail')}; {ok}")
        prev = chk.get("ecc_after") or prev
        if nxt is not None and tag == "e1a":
            gap = _ecc_d(prev, nxt)
            out["e1a_to_e0_tt"] = gap
            if gap and any(gap[k] for k in ECC_COUNTS):
                lines.append(f"between E1a and E0 (TT): ECC moved {gap}")
            prev = nxt
    out["lines"] = lines
    return out


def s1c_summary(d: Path, e1a_ok: bool, method: str, rc: dict) -> dict:
    """jobs/s1c.sh's s1c_summary.json: the T-3.E0/E1a/E1b checks from the step results in the session dir, plus the
    PLL4 walk readiness of both E0 reads and the S2 go/no-go."""
    d = Path(d)
    dig = [_json(d / f"s1c_tt_digest_{i}.json") for i in range(3)]
    lat0, lat1 = _json(d / "s1c_tt_lat_0.json"), _json(d / "s1c_tt_lat_1.json")

    def p50(b):
        return ((b or {}).get("latency") or {}).get("rtt_ms", {}).get("p50")
    p0, p1 = p50(lat0), p50(lat1)
    fd = _json(d / "s1c_tt_lat_fd.json") or {}
    e0k, e0t = _json(d / "s1c_e0_keeper.json"), _json(d / "s1c_e0_tt.json")
    s = {"E0_keeper": rc.get("e0_keeper") == "0", "E0_tt_and_peek": rc.get("e0_tt") == "0",
         "E1a": e1a_ok, "E1b": rc.get("e1b_init") == "0" and rc.get("e1b_check") == "0", "E1b_method": method,
         "tt_bit_identical": bool(dig[0]) and dig[0] == dig[1] == dig[2],
         "tt_p50_ms": [p0, p1], "tt_p50_change": (p1 / p0 - 1) if p0 and p1 else None,
         "tt_max_ms_during_fd_events": (fd.get("latency") or {}).get("rtt_ms", {}).get("max"),
         "host_only": (_json(d / "s1c_host_only.json") or {}).get("total_us"),
         "pll4_after_fd_events": [(_json(d / n) or {}).get("pll4_default_after_each") for n in
                                  ("s1c_fd_events_no_tt.json", "s1c_fd_events_tt.json")]}
    s["tt_p50_within_2pct"] = s["tt_p50_change"] is not None and abs(s["tt_p50_change"]) <= 0.02
    s["pass"] = all(s[k] for k in ("E0_keeper", "E0_tt_and_peek", "E1a", "E1b", "tt_bit_identical",
                                   "tt_p50_within_2pct"))
    if not s["E1a"] and not s["E1b"]:
        s["next"] = "E1F stage-0 (jobs/s1f_stage0.sh) - ask the user for DP3(d) approval first"
    s["s2"] = s2_verdict(e0k if isinstance(e0k, dict) else None, e0t if isinstance(e0t, dict) else None, e1a_ok)
    s["ecc"] = s1c_ecc(d)
    return s


# --- S5a (jobs/s5a_probe.sh): the read-only probe of L2CPU0-3, then again with server_g1 serving ----------------------

S5A_TT_P50_BAND = 0.02  # S5a step_plan / T-3.E0: TT p50 within +-2 % of the baseline taken before the second probe


def _probe_view(p) -> Optional[dict]:
    """The fields of an `x280d probe --tile LIST` report (x280d.S5A_SCHEMA) two probes of one unchanged chip must
    repeat: everything read except TIMER_HEARTBEAT, the telemetry clocks and the read counts."""
    if not isinstance(p, dict) or not isinstance(p.get("checks"), dict):
        return None
    tiles = {}
    for t, tr in sorted((p.get("tiles") or {}).items()):
        v = {k: tr.get(k) for k in ("enabled", "released", "probed", "skipped", "error", "wayenable", "ccache_config",
                                    "ccache_config_translated", "ecc", "hart_status")}
        niu, tlb = tr.get("niu0") or {}, tr.get("tlb") or {}
        v["niu0"] = {k: niu.get(k) for k in ("noc_node_id", "noc_endpoint_id", "niu_cfg_0")} if niu else None
        v["tlb"] = {k: tlb.get(k) for k in ("small", "large")} if tlb else None
        tiles[t] = v
    return {**{k: p.get(k) for k in ("l2cpu_reset", "pll4", "pll_lock", "enabled_l2cpu", "noc_translation",
                                     "tiles_order")}, "tiles": tiles}


def _diffs(a, b, path: str = "", out: Optional[list] = None, limit: int = 20) -> list:
    """'path: a -> b' for each leaf where two JSON values differ (at most `limit`)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            _diffs(a.get(k), b.get(k), f"{path}.{k}" if path else str(k), out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diffs(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {a!r} -> {b!r}")
    return out


def _probe_brief(p, rc) -> dict:
    p = p if isinstance(p, dict) else {}
    return {"rc": rc, "verdict": p.get("verdict") or "no report", "reasons": p.get("reasons"),
            "warnings": p.get("warnings"), "error": p.get("error"), "reads": p.get("reads")}


def _s5a_tile_line(t: str, tr: dict) -> str:
    head = f"S5a L2CPU{t} ({tr.get('xy', ['?', '?'])[0]},{tr.get('xy', ['?', '?'])[1]}): "
    if tr.get("skipped"):
        return head + f"skipped, {tr['skipped']}"
    if tr.get("error"):
        return head + f"probe stopped: {tr['error']}"
    ecc, niu, tlb = tr.get("ecc") or {}, tr.get("niu0") or {}, tr.get("tlb") or {}
    cross = tr.get("cross_read_equal")
    return head + (f"WAYENABLE {tr.get('wayenable')}, CCACHE0_CONFIG {_hex(tr.get('ccache_config'))} (translated "
                   f"{'equal' if cross else 'DIFFERS' if cross is False else 'not read'}), ECC fix/fail dir "
                   f"{ecc.get('dir_fix_count')}/{ecc.get('dir_fail_count')} dat {ecc.get('dat_fix_count')}/"
                   f"{ecc.get('dat_fail_count')}, hart status {_hex(tr.get('hart_status'))}, NOC_NODE_ID "
                   f"{_hex(niu.get('noc_node_id'))}, NOC_ENDPOINT_ID {_hex(niu.get('noc_endpoint_id'))}, NIU_CFG_0 "
                   f"{_hex(niu.get('niu_cfg_0'))}, TLB slots nonzero {tlb.get('small_nonzero')}/{regs.TLB_SMALL_N} small "
                   f"({len(tlb.get('small_distinct') or [])} distinct), {tlb.get('large_nonzero')}/{regs.TLB_LARGE_N} "
                   f"large ({len(tlb.get('large_distinct') or [])} distinct)")


def s5a_summary(d: Path, rc: dict) -> dict:
    """jobs/s5a_probe.sh's s5a_summary.json: both probe verdicts, the per-tile baselines, the two probes compared field
    by field (no interference from server_g1), TT bit-identity and p50, and the S5a -> S5b go/no-go."""
    d = Path(d)
    pk, pt = _json(d / "s5a_probe_keeper.json"), _json(d / "s5a_probe_tt.json")
    dig = [_json(d / f"s5a_tt_digest_{i}.json") for i in range(2)]

    def p50(b):
        return ((b or {}).get("latency") or {}).get("rtt_ms", {}).get("p50")
    p0, p1 = p50(_json(d / "s5a_tt_lat_0.json")), p50(_json(d / "s5a_tt_lat_1.json"))
    vk, vt = _probe_view(pk), _probe_view(pt)
    diffs = _diffs(vk, vt) if vk and vt else None
    s = {"probe_keeper": _probe_brief(pk, rc.get("probe_keeper")), "probe_tt": _probe_brief(pt, rc.get("probe_tt")),
         "probes_equal": diffs == [], "probe_diffs": diffs,
         "tt_bit_identical": bool(dig[0]) and dig[0] == dig[1], "tt_p50_ms": [p0, p1],
         "tt_p50_change": (p1 / p0 - 1) if p0 and p1 else None,
         "tt_max_ms_during_probe": ((_json(d / "s5a_tt_lat_probe.json") or {}).get("latency") or {}).get(
             "rtt_ms", {}).get("max"),
         "harvested": (pk or {}).get("harvested") if isinstance(pk, dict) else None}
    s["tt_p50_within_2pct"] = s["tt_p50_change"] is not None and abs(s["tt_p50_change"]) <= S5A_TT_P50_BAND
    why, notes = [], []
    for name, label in (("probe_keeper", "the probe with the keeper fd"), ("probe_tt", "the probe with server_g1 serving")):
        b = s[name]
        if b["verdict"] != "clean":
            why.append(f"{label}: {b['verdict']}: " + "; ".join(b["reasons"] or [b["error"] or "no JSON report"]))
        notes += [f"{label}: {w}" for w in b["warnings"] or []]
    if s["harvested"]:
        notes.append(f"L2CPU{s['harvested']} harvested/disabled: S5b-S5e as planned need four ranks; re-plan with "
                     f"{4 - len(s['harvested'])}")
    if diffs:
        why.append(f"the two probes differ ({len(diffs)}{'+' if len(diffs) >= 20 else ''}): " + "; ".join(diffs[:5]))
    elif diffs is None and not why:
        why.append("the two probes cannot be compared (a report is missing)")
    if not s["tt_bit_identical"]:
        why.append(f"TT predict_debug digests differ or are missing: {dig}")
    if not s["tt_p50_within_2pct"]:
        why.append(f"TT p50 {p0} -> {p1} ms (change {s['tt_p50_change']}): not within +-{S5A_TT_P50_BAND:.0%}")
    s["tiles"] = {t: tr for t, tr in ((pk or {}).get("tiles") or {}).items()} if isinstance(pk, dict) else {}
    s["pass"] = not why
    s["s5b"] = {"go": not why, "why": why, "notes": notes}
    lines = [_s5a_tile_line(t, s["tiles"][str(t)]) for t in ((pk or {}).get("tiles_order") or [])
             if str(t) in s["tiles"]]
    lines.append(f"S5a TT: bit-identical {s['tt_bit_identical']}, p50 {p0} -> {p1} ms, max during the probe "
                 f"{s['tt_max_ms_during_probe']} ms; probes equal {s['probes_equal']}")
    lines.append("S5a -> S5b: GO" if not why else "S5a -> S5b: NO-GO: " + "; ".join(why))
    s["lines"] = lines
    for tr in s["tiles"].values():  # the raw slot words stay in the probe reports; the summary keeps their digest
        if isinstance(tr.get("tlb"), dict):
            tr["tlb"] = {k: v for k, v in tr["tlb"].items() if k not in ("small", "large")}
    return s


# --- S1d (jobs/s1d.sh): which host LIM write leaves the CCACHE's ECC valid --------------------------------------------

S1D_ORDER = ("a64", "wc2x32", "a64nf", "uc8")  # among clean stores: the boot's, UMD's pattern, then the rest
E1F_NEXT = ("E1F (the hart zero-fills LIM itself: jobs/s1f_stage0.sh, PLAN §5.2) is the remaining path; it spends a "
            "release and needs the user's DP3(d) approval before it is planned")


def _s1d_row(v: dict) -> dict:
    """One plan item of `x280d lim-ecc`, reduced to what the verdict reads."""
    steps = v.get("steps") or []
    last = steps[-1] if steps else {}
    ph = last.get("phases") or {}
    der = [st.get("derived") or {} for st in steps]
    row = {"name": v.get("name"), "lo": v.get("lo"), "clean": v.get("clean") is True,
           "reread_fail_free": v.get("reread_fail_free") is True,
           "full_lim_write_s": v.get("full_lim_write_s"),
           "write": [{"store": st.get("store"), **(dv.get("write_events_per_line") or {}),
                      "accesses_per_line_min": dv.get("accesses_per_line_min"),
                      "event_offsets": dv.get("probe_event_offsets")} for st, dv in zip(steps, der)],
           "read1_per_word": der[-1].get("read1_events_per_word") if der else None,
           "read2_per_word": der[-1].get("read2_events_per_word") if der else None,
           "partial": {k: (ph.get("partial") or {}).get(k) for k in ("dat_fix_count", "dat_fail_count")},
           "probe_unsettled": sum(not x.get("settled", True) for st in steps for x in st.get("probe_write") or []),
           "mismatches": sum(st.get("mismatches", 0) + st.get("partial_mismatches", 0) for st in steps)}
    return row


def _s1d_line(r: dict) -> str:
    w = "; ".join(f"{x['store']} write fix/fail per line {x.get('fix')}/{x.get('fail')} (>= {x['accesses_per_line_min']} "
                  f"accesses, events at +{x['event_offsets']})" for x in r["write"])
    rd = r["read1_per_word"] or {}
    verdict = ("CLEAN" if r["clean"] else "DATA MISMATCH" if r["mismatches"] else
               "RE-READS FAIL-FREE, ECC STILL COUNTS EVENTS" if r["reread_fail_free"] else "ECC NOT VALID")
    return (f"S1d {r['name']}: {w}; re-read fail/fix per word {rd.get('fail')}/{rd.get('fix')}, mismatches "
            f"{r['mismatches']} -> {verdict}")


def s1d_summary(d: Path, rc: dict) -> dict:
    """jobs/s1d.sh's s1d_summary.json: per plan item whether the CCACHE counts nothing on a full re-read of what that
    store wrote (`clean`), and a lim-init recommendation. Valid only if E0 passed before and after (WAYENABLE 0,
    L2CPU_RESET 0x0F, PLL4) and lim-ecc ran with its integrity check."""
    d = Path(d)
    e0, e0b, res = _json(d / "s1d_e0.json"), _json(d / "s1d_e0_after.json"), _json(d / "s1d_lim_ecc.json")
    ran = isinstance(res, dict) and isinstance(res.get("variants"), list) and "error" not in res
    s = {"E0": isinstance(e0, dict) and e0.get("pass") is True, "E0_after": isinstance(e0b, dict) and e0b.get("pass") is True,
         "lim_ecc_ran": ran, "integrity": bool(ran and res.get("integrity") is True), "rc": rc,
         "ecc_at_start": _tile_ecc(e0), "variants": [_s1d_row(v) for v in (res or {}).get("variants", [])] if ran else []}
    if isinstance(res, dict) and "error" in res:
        s["lim_ecc_error"] = res["error"]
    rows = s["variants"]
    clean = [r for r in rows if r["clean"]]
    s["clean_variants"] = [r["name"] for r in clean]
    s["reread_fail_free_variants"] = [r["name"] for r in rows if r["reread_fail_free"]]
    s["mismatch_variants"] = [r["name"] for r in rows if r["mismatches"]]
    s["fresh_at_start"] = bool(s["ecc_at_start"]) and all(s["ecc_at_start"].get(k) == 0 for k in ECC_COUNTS)
    why = [w for ok, w in ((s["E0"], "E0 before the writes failed"), (s["E0_after"], "E0 after the writes failed"),
                           (ran, "lim-ecc did not run"), (s["integrity"], "L2CPU_RESET or WAYENABLE changed"))
           if not ok]
    singles = [r for r in clean if "+" not in r["name"]]
    if why:
        rec = {"verdict": "inconclusive", "lim_init": None,
               "text": "no recommendation, the measurement is not valid: " + "; ".join(why)}
    elif singles:
        def key(r):  # the boot's own store first (no code change), then the others; time only breaks a tie
            return (S1D_ORDER.index(r["name"]) if r["name"] in S1D_ORDER else len(S1D_ORDER),
                    r["full_lim_write_s"] if r["full_lim_write_s"] is not None else float("inf"))
        best = min(singles, key=key)
        t = f" ({best['full_lim_write_s']} s for all 30,720 lines here)" if best["full_lim_write_s"] is not None else ""
        if best["name"] == "a64":
            text = (f"keep method A (store a64){t}: its lines re-read clean here, unlike S1c's E1a; re-run S1c "
                    "(jobs/s1c.sh, gate unchanged) before S2")
        else:
            text = (f"store {best['name']}{t}. Verify it with S1c's E1a gate unchanged "
                    f"(S1C_E1A_STORE={best['name']} jobs/s1c.sh), then boot step 2 must use the same store "
                    "(Session.boot calls kmd.lim_init with a64 today) before S2")
        rec = {"verdict": "host_init", "lim_init": best["name"], "text": text}
    elif clean:
        text = (f"only an overwrite leaves the ECC valid ({', '.join(r['name'] for r in clean)}): lim-init would need "
                f"{clean[0]['name'].replace('+', ' then ')}; verify with S1c's E1a gate before S2")
        rec = {"verdict": "two_pass", "lim_init": clean[0]["name"], "text": text}
    else:
        tried = ", ".join(r["name"] for r in rows)
        text = (f"no host store tried ({tried}) leaves the LIM ECC valid: the CCACHE counts uncorrectable errors on "
                "re-reads of lines that read back right, so S2 stays NO-GO with a host-initialised LIM. " + E1F_NEXT)
        rec = {"verdict": "no_host_init", "lim_init": None, "text": text}
    if s["mismatch_variants"]:
        rec["text"] += f" (data mismatches in {', '.join(s['mismatch_variants'])}: those stores are unreliable)"
    s["recommendation"] = rec
    poisoned = [r for r in rows if not r["clean"] and not r["mismatches"]]
    s["hypothesis"] = ("every non-clean store counted events while writing, then fail-only events on re-read with the "
                       "data intact (write-time read-modify-write poisoning)" if poisoned and all(
                           any((x.get("fail") or 0) > 0 for x in r["write"]) and (r["read1_per_word"] or {}).get("fail")
                           and not (r["read1_per_word"] or {}).get("fix") for r in poisoned) else None)
    s["notes"] = []
    if s["ecc_at_start"] and not s["fresh_at_start"]:
        s["notes"].append("the ECC counts were not 0 at E0 (a host wrote LIM since the chip reset?): the deltas still "
                          "hold, the regions were never written in this session")
    late = [r["name"] for r in rows if r["probe_unsettled"]]
    if late:
        s["notes"].append(f"probe-line counters moved between two reads after a write ({', '.join(late)}): an event "
                          "landed late, so the per-line probe attribution may be off by one line; the phase deltas hold")
    s["lines"] = [_s1d_line(r) for r in rows] + [f"S1d note: {n}" for n in s["notes"]] + [
        "S1d stores whose full re-reads leave dat_fail_count unchanged: "
        f"{', '.join(s['reread_fail_free_variants']) or 'none'}",
        f"S1d clean variants (nothing counted after the write: re-reads, partial writes): "
        f"{', '.join(s['clean_variants']) or 'none'}",
        f"S1d -> lim-init: {rec['text']}"]
    s["pass"] = not why and all(v == "0" for v in rc.values())
    return s


# --- T-3.7 (jobs/m3_eval.sh) --------------------------------------------------------------------------------------


def t23_span_p50(bench) -> Optional[float]:
    """The T-2.3 server span p50 (srv t_reply - t_recv, what analyze_run gates as vla_p50_ms), never the server
    total; ValueError when the bench has none (an older bench_g1: re-run T-2.3)."""
    if not bench:
        return None
    b = _json(bench)
    if not isinstance(b, dict):
        raise ValueError(f"{bench}: not a T-2.3 bench JSON")
    p50 = ((b.get("latency") or {}).get("server_span_ms") or {}).get("p50")
    if p50 is None:
        raise ValueError(f"{bench}: no latency.server_span_ms (analyze_run gates the span, not server_total_ms)")
    return float(p50)


def m3_gates(src: Path, dst: Path, bench, dtype: str) -> dict:
    """The session copy of gates.yaml's m3_rt and m3_rt_slow (verify:all D1: every row, starvation, error and X280
    rows included; never configs/gates_m3.proposed.yaml). With a T-2.3 bench, vla_p50_ms_max = (1 + 5 %) x its server
    span p50 in both sets; DP4 fp32 tightens the shadow bound to 5e-5."""
    import yaml

    sets = yaml.safe_load(Path(src).read_text()) or {}
    missing = [n for n in M3_SETS if not isinstance(sets.get(n), dict)]
    if missing:
        raise ValueError(f"{src}: no gate set {', '.join(missing)}")
    if dtype not in SHADOW_TOL:
        raise ValueError(f"dtype {dtype!r} (fp16 or fp32)")
    out = {n: dict(sets[n]) for n in M3_SETS}
    span = t23_span_p50(bench)
    for g in out.values():
        if dtype == "fp32":
            g["shadow_da_p99_max"] = SHADOW_TOL["fp32"]
        if span is not None:
            g["vla_p50_ms_max"] = round((1 + T23_BAND) * span, 3)
    note = (f"vla_p50_ms_max = 1.05 x T-2.3 server span p50 {span:.3f} ms ({bench})" if span is not None else
            f"vla_p50_ms_max kept at gates.yaml's projection {out['m3_rt']['vla_p50_ms_max']} ms: no T-2.3 bench, so "
            "T-3.7's +-5 % row is not applied")
    head = (f"# T-3.7 session gate sets: gates.yaml {', '.join(M3_SETS)} ({src}), dtype {dtype}.\n# {note}\n")
    Path(dst).write_text(head + yaml.safe_dump(out, sort_keys=False))
    return {"gates_file": str(dst), "from": str(src), "sets": list(M3_SETS), "dtype": dtype,
            "vla_p50_ms_max": out["m3_rt"]["vla_p50_ms_max"], "t23_span_p50_ms": span,
            "shadow_da_p99_max": out["m3_rt"]["shadow_da_p99_max"], "note": note}


def t37_summary(verdict, before, after, dump, bench) -> dict:
    """T-3.7's rows that analyze_run's gate set does not hold: x280d's own counts across the soak (0 timeouts,
    faults, stalls, interlock or degraded replies; calls > 0), TT VLA p50 >= 0.95 x the T-2.3 span (the gate set has
    the <= 1.05 x half), and where the heartbeat gate's ring came from (x280d dump)."""
    v = _json(verdict) or {}
    m = v.get("metrics") or {}
    a, b = ((_json(p) or {}).get("counts") or {} for p in (before, after))
    delta = {k: b.get(k, 0) - a.get(k, 0) for k in ("calls", "ok", *X280D_FAULTS)}
    x280d_ok = bool(b) and delta["calls"] > 0 and not any(delta[k] for k in X280D_FAULTS)
    span = t23_span_p50(bench)
    p50 = m.get("vla_p50_ms")
    band = None
    if span is not None:
        band = {"t23_span_p50_ms": span, "vla_p50_ms": p50, "lo_ms": round((1 - T23_BAND) * span, 3),
                "hi_ms": round((1 + T23_BAND) * span, 3),
                "pass": p50 is not None and (1 - T23_BAND) * span <= p50 <= (1 + T23_BAND) * span}
    ring = _json(dump) or {}
    s = {"x280d_during_soak": delta, "x280d_ok": x280d_ok, "vla_p50_vs_t23": band,
         "t23_row": "applied" if band else "not applied (no T-2.3 bench)",
         "ring": {k: ring.get(k) for k in ("ring", "rows", "calls", "wrapped")}, "ring_limit_calls": RING_CALLS,
         "heartbeat_continuity": m.get("x280_heartbeat_continuity"), "heartbeat": m.get("x280_heartbeat"),
         "gates_pass": v.get("pass")}
    s["pass"] = bool(x280d_ok and (band is None or band["pass"]))
    return s


# --- T-3.6 (jobs/m3_shadow.sh) ------------------------------------------------------------------------------------


def t36_gates(dst: Path, dtype: str) -> dict:
    """The T-3.6 session gate set: per-tick |a_x280 - a_ort| p99.9 <= 1e-2 (fp16) / 5e-5 (fp32), analyze_run's
    shadow_da_p999_max. Plain text, no PyYAML here: the float is written with a '.' and a signed exponent, the only
    form YAML 1.1 (PyYAML) reads as a number (JSON's 5e-05 would come back as a string)."""
    if dtype not in SHADOW_TOL:
        raise ValueError(f"dtype {dtype!r} (fp16 or fp32)")
    tol = SHADOW_TOL[dtype]
    Path(dst).write_text(f"# T-3.6 session gate set ({dtype}, x280/tools/m3_jobs.py t36-gates)\n"
                         f"t36:\n  shadow_da_p999_max: {tol:.4e}\n")
    return {"gates_file": str(dst), "set": "t36", "dtype": dtype, "shadow_da_p999_max": tol}


def t36_summary(verdict, dtype: str, loop_rc: int) -> dict:
    """T-3.6 pass: analyze_run's t36 gates (p99.9 of the live shadow's |a_x280 - a_ort|) and at most 0.1 % missed
    shadow replies (a miss is NaN in shadow_da, outside the p99.9: counted as a failed tick here), loop rc 0."""
    v = _json(verdict) or {}
    m = v.get("metrics") or {}
    miss = m.get("shadow_miss_frac")
    s = {"dtype": dtype, "tol": SHADOW_TOL.get(dtype), "n_ticks": m.get("n_ticks"),
         "shadow_da_p999": m.get("shadow_da_p999"), "shadow_da_p99": m.get("shadow_da_p99"),
         "shadow_da_max": m.get("shadow_da_max"), "shadow_miss_frac": miss, "miss_max": SHADOW_MISS_MAX,
         "gates_pass": v.get("pass"), "loop_rc": int(loop_rc)}
    s["pass"] = bool(v.get("pass") is True and miss is not None and miss <= SHADOW_MISS_MAX and int(loop_rc) == 0)
    return s


# --- T-3.8 (jobs/m3_faults.sh) -------------------------------------------------------------------------------------

T38_CASES = ("kill9_x280d", "park_start", "inject_crc")  # m3_faults.sh's events, one per case
T38_WINDOW_NS = 20_000_000_000  # a case's ticks: the 20 s after its event
T38_PERIOD_MAX_MS = 30.0  # tick to tick inside one episode after a case: no tick missed while ORT takes over


def _load_ticks(ticks):
    """TickLog ticks.npy up to its committed length, as rtloop.telemetry.load_ticks (min of the header and the
    ticks.len sidecar); numpy, so the summaries that read ticks run under simpy."""
    import numpy as np

    t = np.load(ticks)
    try:
        t = t[:min(len(t), int((Path(ticks).parent / "ticks.len").read_text().split()[0]))]
    except (OSError, IndexError, ValueError):
        pass
    return t


def t38_summary(run, events, attach_log, park, loop_rc: int) -> dict:
    """T-3.8: per case, the ticks of the 20 s after its event (m3_faults.sh's t38_events.jsonl, monotonic ns like
    TickLog t_start_ns): rt_sim fell back (wbc_fb: ORT served the failed call on the same tick), the tick period
    stayed under 30 ms between consecutive ticks of one episode (an env reset between episodes, ~4.3 s on the G1, is
    not a WBC period: S4 2026-10-25 failed every case on it), and the X280 served again after the first fallback
    (a fault must not cost the tile: S4's PARK ran a stale stage 0 and every later call stalled). Also: the PARK
    step passed (warm entry, n_reentry + 1: t38_park.json), no SAFETY_FAULT, no fall, the restarted x280d attached
    and its log has no release step, loop rc 0."""
    import numpy as np

    t = _load_ticks(Path(run) / "ticks.npy")
    ev = [json.loads(x) for x in Path(events).read_text().splitlines() if x.strip()]
    eps = [json.loads(x) for x in (Path(run) / "episodes.jsonl").read_text().splitlines() if x.strip()]
    fb, ts, ep = t["wbc_fb"].astype(bool), t["t_start_ns"], t["ep"]
    res = {"cases": {}, "fallback_ticks": int(fb.sum()), "episodes": len(eps), "loop_rc": int(loop_rc),
           "safety_faults": int(sum(e.get("end") == "SAFETY_FAULT" for e in eps)),
           "falls": int(sum(e.get("end") == "FALL" for e in eps))}
    for name in T38_CASES:
        e = next((x for x in ev if x["event"] == name), None)
        if e is None:
            res["cases"][name] = {"ran": False}
            continue
        idx = np.where((ts >= e["mono_ns"]) & (ts < e["mono_ns"] + T38_WINDOW_NS))[0]
        f = fb[idx]
        first = int(np.argmax(f)) if f.any() else None
        same_ep = ep[idx][1:] == ep[idx][:-1]
        gaps = np.diff(ts[idx])[same_ep]
        res["cases"][name] = {
            "ran": True, "ticks": int(idx.size), "fallback_ticks": int(f.sum()),
            "first_fallback_tick_after_event": first,
            "served_after_fallback": int((~f[first + 1:]).sum()) if first is not None else 0,
            "max_period_ms_after": float(gaps.max() / 1e6) if gaps.size else None,
            "episode_ends_in_window": int((~same_ep).sum())}
    pk = _json(park) or {}
    res["park"] = {k: pk.get(k) for k in ("pass", "n_reentry", "state", "resume_s", "why")}
    try:
        log = Path(attach_log).read_text() if attach_log else ""
    except OSError:
        log = ""
    res["attach_released"] = "L2CPU_RESET RMW" in log or "step 7:" in log
    res["attach_ok"] = "READY" in log and "attach:" in log
    ok_cases = all(c.get("ran") and c["fallback_ticks"] > 0 and c["served_after_fallback"] > 0
                   and (c["max_period_ms_after"] if c["max_period_ms_after"] is not None else 99) < T38_PERIOD_MAX_MS
                   for c in res["cases"].values())
    res["pass"] = bool(ok_cases and res["park"]["pass"] is True and not res["safety_faults"] and not res["falls"]
                       and not res["attach_released"] and res["attach_ok"] and res["loop_rc"] == 0)
    return res


# --- T-3.9 (jobs/m3_soak.sh) --------------------------------------------------------------------------------------

T39_GATES = {"period_p99_ms": 21.0, "slips_max": 0, "wbc_fallback_frac_max": 0.0}  # PLAN T-3.9's loop rows
T39_SWAP_DA_MAX = 1e-2  # PLAN T-3.9: |da| at a swap <= 1e-2
T39_STATUS_GAP_S = 30.0  # x280d logs PLL4 + counters every 10 s (x280d.SLOW_HEALTH_S): > 3 periods = reads missed
WBC_ORT = 0  # rtloop.schema.WbcBackend.ORT (TickLog wbc_be)


def t39_gates(dst: Path) -> dict:
    """The T-3.9 session gate set for analyze_run: tick period p99 <= 21 ms, 0 slips, 0 unplanned fallbacks
    (wbc_fb marks a W-WBC call that failed and was served by ORT on the same call; a planned swap to ORT is not
    one). Plain text, as t36-gates."""
    rows = "".join(f"  {k}: {v}\n" for k, v in T39_GATES.items())
    Path(dst).write_text(f"# T-3.9 session gate set (x280/tools/m3_jobs.py t39-gates): PLAN T-3.9's loop rows\nt39:\n{rows}")
    return {"gates_file": str(dst), "set": "t39", **T39_GATES}


def swap_da(ticks) -> dict:
    """|da| at every WBC swap of a run (TickLog ticks.npy; numpy, so run t39-summary with simpy). A swap is a change
    of wbc_be between consecutive ticks of one episode, both served without a fallback. Every remote backend is
    shadowed by ORT (D16: shadow_da = |a_remote - a_ort| on that tick), so a swap to or from ORT is the shadow_da of
    its remote side, and a remote <-> remote swap (x280 <-> gpu) is bounded by the sum of both sides' (through ORT).
    A swap whose needed shadow_da is missing (NaN) counts as no data, which fails."""
    import numpy as np

    t = _load_ticks(ticks)
    be, sd, ep, k = t["wbc_be"].astype(int), t["shadow_da"].astype(np.float64), t["ep"], t["k"]
    served = np.isfinite(t["wbc_rtt_us"]) & ~t["wbc_fb"].astype(bool)
    idx = np.where((be[1:] != be[:-1]) & (ep[1:] == ep[:-1]) & (k[1:] == k[:-1] + 1) & served[1:] & served[:-1])[0]
    das, no_data, pairs = [], 0, {}
    for j in idx:
        a, b = be[j], be[j + 1]
        d = sd[j + 1] if a == WBC_ORT else sd[j] if b == WBC_ORT else sd[j] + sd[j + 1]
        pairs[f"{a}->{b}"] = pairs.get(f"{a}->{b}", 0) + 1
        if np.isfinite(d):
            das.append(float(d))
        else:
            no_data += 1
    mx = max(das) if das else None
    return {"swaps": int(idx.size), "no_data": no_data, "max_da": mx, "tol": T39_SWAP_DA_MAX, "pairs": pairs,
            "wbc_be": "rtloop.schema.WbcBackend (0 ORT, 1 GPU, 2 X280, 3 X280_SIM)",
            "pass": bool(idx.size and not no_data and mx is not None and mx <= T39_SWAP_DA_MAX)}


def _tile_traps(status: Optional[dict]) -> dict:
    return {t: (v or {}).get("n_traps") for t, v in ((status or {}).get("tiles") or {}).items()}


def t39_summary(verdict, ticks, before, after, status_logs, t0: float, t1: float, loop_rc: int,
                swaps: bool = True) -> dict:
    """T-3.9's rows besides analyze_run's t39 set: 0 traps (every tile's L36 n_traps unchanged from x280d's status
    before the soak to after it and in every status read between) and x280d's own record (calls > 0; 0 timeouts,
    faults, stalls, interlock or degraded replies: no unplanned fallback on its side either); |da| at every swap
    <= 1e-2 (swap_da); PLL4 at its default values in every one of x280d's 10 s direct reads inside [t0, t1]
    (x280d_status.jsonl, the files in status_logs that exist; no tt-smi), with no gap over 30 s between reads; and
    loop rc 0."""
    v = _json(verdict) or {}
    st0, st1 = _json(before) or {}, _json(after) or {}
    a, b = st0.get("counts") or {}, st1.get("counts") or {}
    delta = {k: b.get(k, 0) - a.get(k, 0) for k in ("calls", "ok", *X280D_FAULTS)}
    x280d_ok = bool(b) and delta["calls"] > 0 and not any(delta[k] for k in X280D_FAULTS)
    rows = []
    for p in status_logs:
        try:
            lines = Path(p).read_text().splitlines()
        except OSError:
            continue
        for ln in lines:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if isinstance(r, dict) and isinstance(r.get("t"), (int, float)) and t0 <= r["t"] <= t1:
                rows.append(r)
    rows.sort(key=lambda r: r["t"])
    ts = [t0] + [r["t"] for r in rows] + [t1]
    gap = max(y - x for x, y in zip(ts, ts[1:]))
    pll_bad = [r["t"] for r in rows if not all((tile or {}).get("pll4_default") is True
                                               for tile in (r.get("tiles") or {}).values()) or not r.get("tiles")]
    tr0, tr1 = _tile_traps(st0), _tile_traps(st1)
    traps_between = sorted({str(n) for r in rows for n in _tile_traps(r).values()})
    traps = {"before": tr0, "after": tr1, "in_status_reads": traps_between,
             "pass": bool(tr0) and tr0 == tr1 and all(x == 0 for x in tr1.values())
             and set(traps_between) <= {"0"}}
    pll = {"reads": len(rows), "not_default_at": pll_bad[:20], "max_gap_s": round(gap, 3),
           "gap_max_s": T39_STATUS_GAP_S, "pass": bool(rows) and not pll_bad and gap <= T39_STATUS_GAP_S}
    if not swaps:  # X280-only soak (the default since 2026-10-25): no planned swap, so no |da| row to gate
        sw = {"swaps": 0, "pass": True, "applicable": False, "why": "X280-only soak (--no-swaps)"}
    else:
        sw = swap_da(ticks) if ticks and Path(ticks).exists() else {"swaps": 0, "pass": False, "why": "no ticks.npy"}
    s = {"gates_pass": v.get("pass"), "x280d_during_soak": delta, "x280d_ok": x280d_ok, "traps": traps,
         "pll4": pll, "swap_da": sw, "loop_rc": int(loop_rc), "window": [t0, t1]}
    s["pass"] = bool(v.get("pass") is True and x280d_ok and traps["pass"] and pll["pass"] and sw["pass"]
                     and int(loop_rc) == 0)
    return s


# --- CLI ----------------------------------------------------------------------------------------------------------


def _opt(p: str):
    return None if p in ("", "none") else p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("s1c-summary")
    p.add_argument("dir", type=Path)
    p.add_argument("e1a_ok", choices=("0", "1"))
    p.add_argument("method", choices=("A", "B"))
    p.add_argument("rc", nargs="*", help="step=rc pairs")
    p = sub.add_parser("s1d-summary")
    p.add_argument("dir", type=Path)
    p.add_argument("rc", nargs="*", help="step=rc pairs")
    p = sub.add_parser("s5a-summary")
    p.add_argument("dir", type=Path)
    p.add_argument("rc", nargs="*", help="step=rc pairs")
    p = sub.add_parser("m3-gates")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("bench")
    p.add_argument("dtype")
    p = sub.add_parser("t37-summary")
    for a in ("verdict", "before", "after", "dump", "bench", "out"):
        p.add_argument(a)
    p = sub.add_parser("t36-gates")
    p.add_argument("dst", type=Path)
    p.add_argument("dtype")
    p = sub.add_parser("t36-summary")
    for a in ("verdict", "dtype", "loop_rc", "out"):
        p.add_argument(a)
    p = sub.add_parser("t38-summary")
    for a in ("run", "events", "attach_log", "park"):
        p.add_argument(a)
    p.add_argument("loop_rc", type=int)
    p.add_argument("out")
    p = sub.add_parser("t39-gates")
    p.add_argument("dst", type=Path)
    p = sub.add_parser("t39-summary")
    p.add_argument("--no-swaps", action="store_true", help="X280-only soak: the swap |da| row does not apply")
    for a in ("verdict", "ticks", "before", "after"):
        p.add_argument(a)
    p.add_argument("t0", type=float)
    p.add_argument("t1", type=float)
    p.add_argument("loop_rc", type=int)
    p.add_argument("out")
    p.add_argument("status_logs", nargs="+", help="x280d_status.jsonl files (missing ones are skipped)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "s1c-summary":
            rc = dict(kv.split("=", 1) for kv in a.rc)
            r = s1c_summary(a.dir, a.e1a_ok == "1", a.method, rc)
            (a.dir / "s1c_summary.json").write_text(json.dumps(r, indent=1))
            ok = r["pass"]
        elif a.cmd == "s1d-summary":
            r = s1d_summary(a.dir, dict(kv.split("=", 1) for kv in a.rc))
            (a.dir / "s1d_summary.json").write_text(json.dumps(r, indent=1))
            ok = r["pass"]
        elif a.cmd == "s5a-summary":
            r = s5a_summary(a.dir, dict(kv.split("=", 1) for kv in a.rc))
            (a.dir / "s5a_summary.json").write_text(json.dumps(r, indent=1))
            ok = r["pass"]
        elif a.cmd == "m3-gates":
            r, ok = m3_gates(a.src, a.dst, _opt(a.bench), a.dtype), True
        elif a.cmd == "t37-summary":
            r = t37_summary(a.verdict, a.before, a.after, a.dump, _opt(a.bench))
            Path(a.out).write_text(json.dumps(r, indent=1))
            ok = r["pass"]
        elif a.cmd == "t36-gates":
            r, ok = t36_gates(a.dst, a.dtype), True
        elif a.cmd == "t38-summary":
            r = t38_summary(a.run, a.events, _opt(a.attach_log), a.park, a.loop_rc)
            Path(a.out).write_text(json.dumps(r, indent=1))
            ok = r["pass"]
        elif a.cmd == "t39-gates":
            r, ok = t39_gates(a.dst), True
        elif a.cmd == "t39-summary":
            r = t39_summary(a.verdict, a.ticks, a.before, a.after, a.status_logs, a.t0, a.t1, a.loop_rc,
                            swaps=not a.no_swaps)
            Path(a.out).write_text(json.dumps(r, indent=1))
            ok = r["pass"]
        else:
            r = t36_summary(a.verdict, a.dtype, int(a.loop_rc))
            Path(a.out).write_text(json.dumps(r, indent=1))
            ok = r["pass"]
    except (ValueError, OSError, KeyError) as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 2
    print(json.dumps(r))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
