#!/usr/bin/env bash
# Session S2 (PLAN.md WP-3.6, DP3 approved): T-3.E2 release + monitor, T-3.E2m LIM microbench, T-3.E3 power survival,
# and E3(d) last (DP3(c) approved: "yes, last in S2"; X280_E3D=0 skips it).
#
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla none --job $RT/jobs/s2_e2_e3.sh --tag s2
#
# chip_session has probed, then started `x280d serve --stage0` (the E1F boot, standard since 2026-10-25: the hart
# zero-fills LIM itself, then the monitor; host-written LIM is unusable, S1c/S1d) or `--boot` (method A, §5.2 steps
# 1-10; kept in the code, not used on silicon): either way the session's one L2CPU3 release, before any TT build, the
# blob LOAD and the boot parity check vs the x86 build, and x280d's mode 'booted'. Nothing below assumes a
# host-initialised LIM. This job then runs, in order:
#   E2   status (fw_magic, fw_ver, vlenb recorded - 64 expected, misa V); heartbeat >= 1 kHz and mtime ~ 50 MHz
#        (x280d hbrate); SELFTEST (every LIM line read by the hart, BEU accrued 0) and no RNMI taken across it
#        (n_reentry and n_traps unchanged in the status after it); mcycle/mtime = 16 +- 5 % (lim_bench test 8);
#        LOAD+CALL checksum 10/10 and the RNMI re-arm loop hang -> PARK -> CALL ... 10/10 with n_reentry 1..10
#        (x280d e2).
#   E2m  lim_bench tests 1-6, 8 (+7 Zero Device with X280_E2M_ZERO_DEVICE=1) -> s2_e2m.json; §1.3 and the WP-3.7
#        gates are re-derived from it before S3. The payloads overwrite the blob region; the blob is LOADed again.
#   E3   10 legacy fd open/close (power messages) with no heartbeat gap > 10 ms and PLL4 at its defaults after each;
#        server_g1 start -> stop -> start (start_vla.sh, children of this session) under a heartbeat watch whose
#        20 s windows each end with a PLL4 read (so every server_g1 event is followed by one); then, with server_g1
#        serving, 10 more fd events while bench_g1 records the TT latency. Relock evidence (PLL4 CNTL_0 read away
#        from 0x12 by the 200 us sampler) is reported as e3_relock_events: without any, E3 shows no relock was
#        survived (e3_conclusive false; reported, not gated).
#   E3(d) all fds closed for 10 s (x280d stopped: its keeper fd close writes 'GONE'; server_g1 stopped), then a new
#        `x280d serve --attach` (no release; from a --stage0 session too, jobs/_x280.sh x280_reattach) must see the
#        heartbeat resume.
# Writes s2_*.json and s2_summary.json (with "boot": the serve mode). Exit 0 all pass, 1 a check failed,
# 2 misconfigured, 3 X280 dirty (e.g. PARK did not bring the hart back).

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3
declare -A RCS=()  # not R: env.sh's R is the Isaac-GR00T repo that simpy cd's into
BOOT=$(x280_boot_flag)
job_log "S2 on a session x280d booted with --$BOOT"

job_log "E2: monitor status, heartbeat, SELFTEST, mcycle/mtime, checksum CALL 10/10, RNMI re-arm 10/10"
x280_step s2_status status; RCS[status]=$?
x280_step s2_hbrate hbrate --secs 0.5; RCS[hbrate]=$?
x280_step s2_selftest selftest; RCS[selftest]=$?
x280_step s2_status_after_selftest status  # no RNMI across SELFTEST: n_reentry / n_traps as in s2_status
x280_step s2_mtime_ratio lim-bench --tests 8; RCS[ratio]=$?
x280_step s2_e2 e2 --n 10; RCS[e2]=$?
if [ "${RCS[e2]}" != 0 ] && [ "$(jget "$CHIP_SESSION_DIR/s2_e2.json" "all(x['park'] for x in d.get('rearm', []))")" != True ]; then
    x280_dirty "E2: PARK did not bring the hart back (RNMI re-arm failed): the ledger marks it, DEVX resets"
fi

job_log "E2m: LIM microbench"
zd=()
[ "${X280_E2M_ZERO_DEVICE:-0}" = 1 ] && zd=(--zero-device)
x280_step s2_e2m lim-bench --tests "1,2,3,4,5,6,8${zd:+,7}" "${zd[@]}"; RCS[e2m]=$?
x280_step s2_blob_reload load --blob; RCS[blob]=$?

job_log "E3: 10 legacy fd open/close with the heartbeat sampled"
x280_step s2_e3_fd powertest --fd-churn 10; RCS[e3_fd]=$?

job_log "E3: server_g1 start -> stop -> start under a heartbeat watch"
stopf=$CHIP_SESSION_DIR/.e3_watch_stop
rm -f "$stopf"
( while [ ! -f "$stopf" ]; do
      x280 powertest --fd-churn 0 --watch-s 20 >>"$CHIP_SESSION_DIR/s2_e3_watch.jsonl" 2>/dev/null
  done ) &
watch_pid=$!
"$CHIP_START_VLA" tt; RCS[vla_start1]=$?
"$CHIP_START_VLA" stop; RCS[vla_stop]=$?
"$CHIP_START_VLA" tt; RCS[vla_start2]=$?
touch "$stopf"
wait "$watch_pid"
CHIP_VLA_URL=tcp://127.0.0.1:${CHIP_VLA_PORT:-5557}
export CHIP_VLA_URL
job_log "E3: 10 fd events with server_g1 serving (TT latency recorded)"
tt_latency "$CHIP_SESSION_DIR/s2_e3_tt_lat.json" 200 50 &
lat_pid=$!
sleep 2
x280_step s2_e3_fd_tt powertest --fd-churn 10 --watch-s 1; RCS[e3_fd_tt]=$?
wait "$lat_pid"
"$CHIP_START_VLA" stop

E3D=skipped
if [ "${X280_E3D:-1}" = 1 ]; then
    job_log "E3(d) (DP3(c) approved, last): every fd closed for 10 s, then attach"
    chip_stop_named x280d || x280_dirty "E3(d): x280d did not stop"
    "$CHIP_START_VLA" stop >/dev/null 2>&1
    sleep 10
    if x280_reattach; then  # placed as the session's x280d (CPU 5, status stream if the session has one)
        x280_step s2_e3d_hbrate hbrate --secs 0.5; RCS[e3d]=$?
        x280_step s2_e3d_status status
        E3D=$([ "${RCS[e3d]}" = 0 ] && echo pass || echo fail)
    else
        RCS[e3d]=1
        E3D="attach failed (see x280d.attach.log)"
    fi
fi

python3 - "$CHIP_SESSION_DIR" "$E3D" "$(for k in "${!RCS[@]}"; do echo "$k=${RCS[$k]}"; done)" "$BOOT" <<'EOF'
import json, sys
from pathlib import Path
d, e3d, rcs, boot = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
rc = dict(kv.split("=") for kv in rcs.split())
def j(name):
    try:
        return json.loads((d / name).read_text())
    except (OSError, ValueError):
        return None
st = j("s2_status.json") or {}
t3 = (st.get("tiles") or {}).get("3", {})
l1, l36 = t3.get("l1", {}), t3.get("l36", {})
l36_after = (((j("s2_status_after_selftest.json") or {}).get("tiles") or {}).get("3") or {}).get("l36") or {}
ratio = (((j("s2_mtime_ratio.json") or {}).get("8") or {}).get("result") or {}).get("bytes")  # mcycle/mtime x1000
watch = [json.loads(x) for x in (d / "s2_e3_watch.jsonl").read_text().splitlines() if x.strip()] \
    if (d / "s2_e3_watch.jsonl").exists() else []
gaps = [w.get("max_heartbeat_gap_ms") for w in watch if w.get("max_heartbeat_gap_ms") is not None]
inconclusive = sum(bool(w.get("inconclusive")) for w in watch)  # x280d's own sampler stalled > 10 ms in a chunk
fd_runs = [r for r in (j("s2_e3_fd.json"), j("s2_e3_fd_tt.json")) if r]
relock = sum(len(r.get("pll4_cntl0_changes") or []) for r in fd_runs + watch)  # CNTL_0 != 0x12 (bypass) sightings
s = {"boot": boot, "vlenb": l1.get("vlenb"), "misa_V": bool((l36.get("misa") or 0) >> 21 & 1),
     "fw_ver": l1.get("fw_ver"),
     "heartbeat": (j("s2_hbrate.json") or {}).get("changes_per_s"), "hbrate_pass": rc.get("hbrate") == "0",
     "selftest_pass": rc.get("selftest") == "0",
     # T-3.E2 "no RNMI cause 3": an RNMI during SELFTEST would re-enter crt0 (n_reentry + 1) or trap
     "selftest_no_rnmi": bool(l36_after) and all(l36_after.get(k) == l36.get(k) for k in ("n_reentry", "n_traps"))
     and l36_after.get("n_traps") == 0,
     "mcycle_per_mtime": ratio / 1000 if ratio is not None else None,
     "e2_pass": rc.get("e2") == "0", "e2m_recorded": rc.get("e2m") == "0",
     "e3_fd_pass": rc.get("e3_fd") == "0" and rc.get("e3_fd_tt") == "0",
     "e3_vla_cycle_ok": all(rc.get(k) == "0" for k in ("vla_start1", "vla_stop", "vla_start2")),
     "e3_watch_max_gap_ms": max(gaps) if gaps else None, "e3_watch_chunks": len(watch),
     "e3_watch_inconclusive_chunks": inconclusive,
     # PLL4 at its defaults after each server_g1 event: every watch window ends with a PLL4 read
     "e3_watch_pll4_default": bool(watch) and all(w.get("pll4_default_after_each") and
                                                  all(w["pll4_default_after_each"]) for w in watch),
     "e3_relock_events": relock, "e3_conclusive": relock > 0,
     "e3_tt_max_ms_during_fd": ((j("s2_e3_tt_lat.json") or {}).get("latency", {}).get("rtt_ms") or {}).get("max"),
     "e3d": e3d}
s["ratio_ok"] = s["mcycle_per_mtime"] is not None and abs(s["mcycle_per_mtime"] / 16.0 - 1) <= 0.05
s["e3_watch_ok"] = bool(gaps) and max(gaps) < 10.0
s["pass"] = all(s[k] for k in ("misa_V", "hbrate_pass", "selftest_pass", "selftest_no_rnmi", "ratio_ok", "e2_pass",
                               "e2m_recorded", "e3_fd_pass", "e3_vla_cycle_ok", "e3_watch_ok",
                               "e3_watch_pll4_default")) and e3d in ("pass", "skipped")
(d / "s2_summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
sys.exit(0 if s["pass"] else 1)
EOF
job_exit $?
