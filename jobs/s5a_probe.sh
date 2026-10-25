#!/usr/bin/env bash
# Session S5a (docs/plan/vla-wbc-x280/research/four_tile_design.md step_plan S5a): read-only probe of all four L2CPU
# tiles L2CPU0-3, with the keeper fd only, then again with server_g1 serving (no interference). No hart is released
# and nothing is written to any L2CPU tile.
#
#   CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --no-release --probe-tiles 3,2,1,0" \
#       $DEVX $RT/scripts/chip_session.sh --x280 probe --vla none --debug-endpoints --job $RT/jobs/s5a_probe.sh --tag s5a
#
# Why --x280 probe: it is chip_session's only no-release mode (x280d serve --no-release: the keeper fd and windows,
# never a release, as in S1c/S1d); boot:3 releases L2CPU3 and attach needs a released tile. --x280 != none also gives
# the DEVX reset after the hold and chip_session's own step-1 L2CPU3 health check (`x280d probe --json`, 10 s
# timeout). CHIP_X280D_SERVE_ARGS adds --probe-tiles 3,2,1,0: the serving daemon then also opens read-only probe
# windows on all four tiles (CCACHE0, the translated cross-read, the external-peripheral prefix up to NIU #0), and
# since it holds the session's only fd (§5.2 invariant 3) the probes below go to it with --via. --vla none and
# --debug-endpoints: server_g1 is started here, after the first probe, and its predict_debug gives the TT digest
# (S1c's E0 pattern).
#   1  s5a_probe_keeper   x280d probe --tile 3,2,1,0 --via (Session.tiles_probe): ARC first (L2CPU_RESET, PLL4,
#                         PLL_LOCK, ENABLED_L2CPU, TAG_NOC_TRANSLATION, L2CPUCLK0-3); a tile that is not enabled, whose
#                         L2CPUCLK derives to 0 MHz, that is released, or any tile while PLL4 is off its defaults is
#                         reported and skipped, never read. Then per tile in the order 3,2,1,0, inside its PLL4
#                         interlock batch: WAYENABLE, CCACHE0_CONFIG raw and translated, the CCACHE ECC registers, the
#                         hart-status word 0x2001_0400, NIU #0 NOC_NODE_ID / NOC_ENDPOINT_ID / NIU_CFG_0, all 256 TLB
#                         window slots. Exit 3 (dirty: L2CPU_RESET != 0x0F, PLL4 off its defaults, WAYENABLE != 0, an
#                         incomplete probe) stops the session before server_g1 starts; exit 4 (only ENABLED_L2CPU !=
#                         0xF: harvested) goes on with the enabled tiles; no report and a daemon that no longer
#                         answers ping (a hung tile read, R28) is X280 dirty too.
#   2  start_vla.sh tt; the TT fixed-noise digest 0 and latency baseline 0 (S5A_TT_N calls at 320 ms)
#   3  s5a_probe_tt       the same probe with server_g1 serving, with a TT latency run beside it (its max is recorded,
#                         not gated); then digest 1 and latency 1
#   4  s5a_summary.json (x280/tools/m3_jobs.py s5a-summary): both verdicts, the per-tile baselines, the two probes
#      compared field by field (TIMER_HEARTBEAT and read counts aside), TT digests bit-identical, TT p50 within +-2 %,
#      and the S5a -> S5b GO / NO-GO line.
# Only L2CPU_RESET 0x0F, PLL4 at its defaults, ENABLED_L2CPU 0xF and WAYENABLE 0 decide a probe's verdict; the ECC
# counts, hart status, NIU words and slot values are recorded baselines (S5b compares its slot read-back to them). The
# BEUs are not read: no source says a hart's BEU answers while the harts are in reset (review F7, kmd.read_beu), so
# S5b takes that baseline from the hart side (SELFTEST).
# Env: S5A_TILES (3,2,1,0), S5A_TT_N (100). Writes s5a_*.json, s5a_summary.json. Exit 0 all pass, 1 a check failed,
# 2 misconfigured, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 probe
need_debug_endpoints
TILES=${S5A_TILES:-3,2,1,0}
case " ${CHIP_X280D_SERVE_ARGS:-} " in
    *" --probe-tiles "*) ;;
    *) job_log "S5a needs the daemon's probe windows: CHIP_X280D_SERVE_ARGS must hold --probe-tiles $TILES" \
           "(it is '${CHIP_X280D_SERVE_ARGS:-}')"
       exit 2 ;;
esac
declare -A RCS=()  # not R: env.sh's R is the Isaac-GR00T repo that simpy cd's into
REASONS="'; '.join(d.get('reasons') or [])"  # jget expression: a probe report's reasons

probe_hung() {  # probe_hung WHICH: a probe without a report and a daemon that no longer answers ping hung on a tile
    # access (R28; chip_session counts a step-1 probe killed at its timeout as dirty too): exit 3
    ping_url "$CHIP_X280_URL" 10 ||
        x280_dirty "S5a probe ($1): no report and x280d does not answer ping (a hung tile read?)"
}

s5a_summary() {  # s5a_summary: s5a_summary.json from the step results, its lines into the job log; sets SUM_RC
    local sum=$CHIP_SESSION_DIR/s5a_summary.json line
    # shellcheck disable=SC2046  # one step=rc word per step
    python3 "$RT/x280/tools/m3_jobs.py" s5a-summary "$CHIP_SESSION_DIR" \
        $(for k in "${!RCS[@]}"; do echo "$k=${RCS[$k]}"; done) >/dev/null
    SUM_RC=$?
    [ -f "$sum" ] || return 0
    while IFS= read -r line; do
        [ -n "$line" ] && job_log "$line"
    done < <(jget "$sum" "'\n'.join(d['lines'] + ['S5a note: ' + n for n in d['s5b']['notes']])")
}

job_log "S5a step 1: probe of L2CPU $TILES with the keeper fd only (no TT)"
x280_step s5a_probe_keeper probe --tile "$TILES"; RCS[probe_keeper]=$?
case ${RCS[probe_keeper]} in
    0) ;;
    4) job_log "S5a: ENABLED_L2CPU != 0xF: harvested tile(s) reported and skipped; going on with the enabled ones" ;;
    3) s5a_summary
       x280_dirty "S5a probe with the keeper fd: $(jget "$CHIP_SESSION_DIR/s5a_probe_keeper.json" "$REASONS")" ;;
    *) s5a_summary
       probe_hung keeper
       job_log "S5a probe with the keeper fd gave no report (rc ${RCS[probe_keeper]}): see s5a_probe_keeper.json"
       job_exit 1 ;;
esac

job_log "S5a step 2: server_g1 as a child, TT digest and latency baseline"
"$CHIP_START_VLA" tt || { job_log "server_g1 did not start"; job_exit 2; }
CHIP_VLA_URL=tcp://127.0.0.1:${CHIP_VLA_PORT:-5557}
export CHIP_VLA_URL
tt_digest "$CHIP_SESSION_DIR/s5a_tt_digest_0.json" >/dev/null
tt_latency "$CHIP_SESSION_DIR/s5a_tt_lat_0.json" "${S5A_TT_N:-100}" 320

job_log "S5a step 3: the probe again with server_g1 serving"
tt_latency "$CHIP_SESSION_DIR/s5a_tt_lat_probe.json" 60 50 &
lat_pid=$!
sleep 2
x280_step s5a_probe_tt probe --tile "$TILES"; RCS[probe_tt]=$?
wait "$lat_pid"
tt_digest "$CHIP_SESSION_DIR/s5a_tt_digest_1.json" >/dev/null
tt_latency "$CHIP_SESSION_DIR/s5a_tt_lat_1.json" "${S5A_TT_N:-100}" 320

job_log "S5a step 4: summary"
s5a_summary
case ${RCS[probe_tt]} in 0 | 3 | 4) ;; *) probe_hung tt ;; esac
[ "${RCS[probe_tt]}" = 3 ] &&
    x280_dirty "S5a probe with server_g1 serving: $(jget "$CHIP_SESSION_DIR/s5a_probe_tt.json" "$REASONS")"
job_exit "$SUM_RC"  # 0 GO, 1 a check failed, 2 the summary had bad input
