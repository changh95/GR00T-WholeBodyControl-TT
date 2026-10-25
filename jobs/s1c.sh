#!/usr/bin/env bash
# Session S1c (PLAN.md WP-3.5; T-3.E0, T-3.E1a, T-3.E1b): read-only probe, LIM init by the host, TT coexistence.
# No hart is released in this session.
#
#   $DEVX $RT/scripts/chip_session.sh --x280 probe --vla none --debug-endpoints --job $RT/jobs/s1c.sh
#
# chip_session has run `x280d probe --json` (L2CPU health check) and started `x280d serve --no-release` (keeper fd and
# windows, never a release) before this job; server_g1 is started HERE as a child (step 3), so E0/E1a run without TT.
#   1  E0 with the keeper fd only: L2CPU_RESET 0x0F, WAYENABLE 0, PLL4 defaults, ENABLED_L2CPU bit 3, (8,7) = (8,29)
#      CCACHE0_CONFIG, ECC counters readable, TIMER_HEARTBEAT increasing (x280d e0). It also records e0.pll_lock and
#      e0.pll4_walk_ready: the boot's PLL4 walks (§5.2 steps 6/8) re-read CNTL_0 and PLL_LOCK before every write and
#      stop unless CNTL_0 & 0x1F == 0x12 and PLL_LOCK bit 4 is set (review F5), values taken from the CMFW sources and
#      first seen on silicon here.
#   2  E1a, no TT: method A over all 1,966,080 B (pattern seed 1), read back 100 %, 100 partial 4 B UC writes on
#      initialised lines, CCACHE ECC fail Δ = 0, L2CPU_RESET and WAYENABLE unchanged, SET_NOC_CLEANUP check on a test
#      fd; then the host-only mailbox bench (§1.4.4: S1c measures the host overhead) and 3 legacy fd events (PLL4).
#   3  start_vla.sh tt; TT fixed-noise digest + latency baseline; E0 again with server_g1 serving, cross-checked through
#      server_g1's x280_peek; TT latency while x280d opens/closes 3 legacy fds; digest again (bit-identical).
#   4  E1b: the same pattern checks (seed 2) on a second region 0x0810_0000-0x0818_0000 with TT serving; method B there
#      if method A failed in E1a: server_g1's x280_lim_write debug endpoint (ttnn write_to_core in 64 B chunks), with
#      x280d checking the release state, the per-line policy and the PLL4/ENABLED_L2CPU batch interlock before every
#      4 KiB request (verify:all D12); TT digest identical, TT p50 within ±2 %.
#   5  s1c_summary.json (x280/tools/m3_jobs.py s1c-summary) and a GO / NO-GO line for S2: GO needs E0 (keeper fd)
#      passed, e0.pll4_walk_ready true on that read (S2 boots before any TT build) and E1a passed (boot step 2 is
#      method A). A NO-GO names the reason; for a PLL4 one, re-derive kmd's walk interlock from the recorded
#      pll_lock / CNTL_0 values before S2 (the walk would otherwise stop at its first write: exit 3, nothing spent).
# If methods A and B both fail, E1F (jobs/s1f_stage0.sh) is next, and it needs the user's DP3(d) approval first.
# Every LIM init and check records the CCACHE ECC counters before and after; the summary's `ecc` block and the job log
# give each step's fix/fail deltas, events per line / per word read, the last event address and which part of the
# check's gate failed (S1c 2026-10-25 failed E1a and E1b on the ECC fail delta alone). The gates are unchanged.
# Env: S1C_REGION_LO/HI (E1b region), S1C_HOST_ONLY_N (5000), S1C_TT_N (100 latency calls per sample),
# S1C_E1A_STORE (method A's host store, default a64 = the boot's; S1d's s1d_summary.json may recommend another).
# Writes s1c_*.json, s1c_summary.json. Exit 0 all pass, 1 a check failed, 2 misconfigured, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 probe
need_debug_endpoints
LO=${S1C_REGION_LO:-0x08100000}
HI=${S1C_REGION_HI:-0x08180000}
declare -A RCS=()  # not R: env.sh's R is the Isaac-GR00T repo that simpy cd's into

job_log "S1c step 1: E0 with the keeper fd only"
x280_step s1c_e0_keeper e0; RCS[e0_keeper]=$?

job_log "S1c step 2: E1a (method A, whole LIM, no TT)"
x280_step s1c_e1a_init lim-init --method A --store "${S1C_E1A_STORE:-a64}" --seed 1; RCS[e1a_init]=$?
x280_step s1c_e1a_check lim-check --seed 1 --partial 100 --cleanup-check; RCS[e1a_check]=$?
E1A_OK=$(( RCS[e1a_init] == 0 && RCS[e1a_check] == 0 ))
if [ "$E1A_OK" = 1 ]; then
    x280_step s1c_host_only bench --host-only --n "${S1C_HOST_ONLY_N:-5000}"; RCS[host_only]=$?
fi
x280_step s1c_fd_events_no_tt powertest --fd-churn 3; RCS[fd_no_tt]=$?

job_log "S1c step 3: server_g1 as a child, then E0 with TT serving"
"$CHIP_START_VLA" tt || { job_log "server_g1 did not start"; job_exit 2; }
CHIP_VLA_URL=tcp://127.0.0.1:${CHIP_VLA_PORT:-5557}
export CHIP_VLA_URL
tt_digest "$CHIP_SESSION_DIR/s1c_tt_digest_0.json" >/dev/null
tt_latency "$CHIP_SESSION_DIR/s1c_tt_lat_0.json" "${S1C_TT_N:-100}" 320
x280_step s1c_e0_tt e0 --peek "$CHIP_VLA_URL"; RCS[e0_tt]=$?
tt_latency "$CHIP_SESSION_DIR/s1c_tt_lat_fd.json" 60 50 &
lat_pid=$!
sleep 2
x280_step s1c_fd_events_tt powertest --fd-churn 3; RCS[fd_tt]=$?
wait "$lat_pid"
tt_digest "$CHIP_SESSION_DIR/s1c_tt_digest_1.json" >/dev/null

job_log "S1c step 4: E1b on $LO-$HI with TT serving"
METHOD=A
[ "$E1A_OK" = 1 ] || METHOD=B
x280_step s1c_e1b_init lim-init --method "$METHOD" --lo "$LO" --hi "$HI" --seed 2 --vla-url "$CHIP_VLA_URL"; RCS[e1b_init]=$?
x280_step s1c_e1b_check lim-check --lo "$LO" --hi "$HI" --seed 2 --partial 100; RCS[e1b_check]=$?
tt_latency "$CHIP_SESSION_DIR/s1c_tt_lat_1.json" "${S1C_TT_N:-100}" 320
tt_digest "$CHIP_SESSION_DIR/s1c_tt_digest_2.json" >/dev/null

# shellcheck disable=SC2046  # one step=rc word per check
python3 "$RT/x280/tools/m3_jobs.py" s1c-summary "$CHIP_SESSION_DIR" "$E1A_OK" "$METHOD" \
    $(for k in "${!RCS[@]}"; do echo "$k=${RCS[$k]}"; done)
rc=$?
[ "$rc" = 2 ] && job_exit 2
sum=$CHIP_SESSION_DIR/s1c_summary.json
job_log "S1c E0 PLL4: keeper fd pll4_walk_ready=$(jget "$sum" "d['s2']['e0_keeper']['pll4_walk_ready']")" \
    "pll_lock=$(jget "$sum" "d['s2']['e0_keeper']['pll_lock']") cntl_0=$(jget "$sum" "d['s2']['e0_keeper']['pll4_cntl_0']");" \
    "with TT pll4_walk_ready=$(jget "$sum" "d['s2']['e0_tt']['pll4_walk_ready']")" \
    "pll_lock=$(jget "$sum" "d['s2']['e0_tt']['pll_lock']")"
if [ "$(jget "$sum" "d['s2']['go']")" = True ]; then
    job_log "S1c -> S2: GO (E0, E1a and the PLL4 walk interlock values hold; $sum)"
else
    job_log "S1c -> S2: NO-GO: $(jget "$sum" "'; '.join(d['s2']['why'])")"
fi
while IFS= read -r line; do
    [ -n "$line" ] && job_log "S1c ECC: $line"
done < <(jget "$sum" "'\n'.join(d['ecc']['lines'])")
notes=$(jget "$sum" "'; '.join(d['s2']['notes'])")
[ -n "$notes" ] && job_log "S1c -> S2 note: $notes"
job_exit $rc
