#!/usr/bin/env bash
# Session S4 (a), T-3.6 (PLAN.md WP-3.8): the X280 in the shadow of ORT, 20 episodes; the X280 never drives the legs.
#
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla tt --job $RT/jobs/m3_shadow.sh
#   X280_SLOW=1 ...   DP11 (x280d --slow): the shadow's reply timeout is 12 ms
#
# The session x280d booted with --stage0 (the standard E1F boot since 2026-10-25) or --boot; the job reads neither.
# PLAN's form, live: `rt_sim --wbc ort --shadow $CHIP_X280_URL`. ORT drives every tick; the same obs goes to x280d as
# the shadow (rtloop.loop --shadow: no fallback, latched like a primary), and shadow_da = per-tick
# max|a_x280 - a_ort| is logged. A missed shadow reply is NaN, never a 0 that reads as parity (shadow_miss_frac).
# Then analyze_run.py --gates t36 with a session set holding the one T-3.6 row, shadow_da_p999_max (x280/tools/
# m3_jobs.py t36-gates: 1e-2 fp16, 5e-5 fp32, from x280d's info.dtype), and m3_jobs.py t36-summary: pass = that gate,
# at most 0.1 % missed replies (a miss counts as a failed tick of the p99.9) and loop rc 0. --dump-obs516 still
# writes the run's obs516.npz (inputs + ORT actions) for T-3.5 and T-3.E5a (jobs/_x280.sh newest_obs516).
# M3_SHADOW_MODE=sync is for CPU smoke tests of this job only (tests/x280/test_m3_jobs.py); T-3.6 is the rt run,
# which exits 2 under CHIP_SAFE_CPUS (need_rt_cores: rt_sim --mode rt moves itself onto CPUs 2/3).
# Env: M3_SHADOW_SEEDS (2000-2019), M3_SHADOW_EPISODES (20), M3_SHADOW_MAX_TICKS (1440), M3_SHADOW_LOOP_ARGS.
# Writes runs_t36/, gates_t36.yaml, t36_summary.json. Exit 0 pass, 1 fail, 2 misconfigured, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3 attach
need_vla
M3J=$RT/x280/tools/m3_jobs.py
mode=${M3_SHADOW_MODE:-rt}
case $mode in rt | sync) ;; *) job_log "M3_SHADOW_MODE=$mode (rt, or sync for smoke tests)"; job_exit 2 ;; esac
[ "$mode" = rt ] && need_rt_cores "rt_sim (CPUs 2/3)"
shadow_to=()
[ "${X280_SLOW:-0}" = 1 ] && shadow_to=(--shadow-timeout-ms 12)
dtype=$(x280 ping | python3 -c "import json,sys; print(json.load(sys.stdin)['info'].get('dtype','fp16'))") ||
    { job_log "x280d $CHIP_X280_URL does not answer ping"; job_exit 2; }
gates=$CHIP_SESSION_DIR/gates_t36.yaml
python3 "$M3J" t36-gates "$gates" "$dtype" || job_exit 2
runs=$CHIP_SESSION_DIR/runs_t36
# shellcheck disable=SC2206
extra=(${M3_SHADOW_LOOP_ARGS-})
simpy -m rtloop.loop --mode "$mode" --arm v1 --s 16 --B 0 --d-init 3 --vla "$CHIP_VLA_URL" --wbc ort \
    --shadow "$CHIP_X280_URL" "${shadow_to[@]}" --crn --seeds "${M3_SHADOW_SEEDS:-2000-2019}" \
    --n-episodes "${M3_SHADOW_EPISODES:-20}" --max-ticks "${M3_SHADOW_MAX_TICKS:-1440}" --dump-obs516 --tag t36 \
    --runs-root "$runs" "${extra[@]}"
loop_rc=$?
run=$(find "$runs" -mindepth 1 -maxdepth 1 -type d -name "*_t36" 2>/dev/null | sort | tail -n 1)
[ -n "$run" ] || { job_log "the loop wrote no run dir (rc=$loop_rc)"; job_exit 2; }
simpy "$RT/scripts/analyze_run.py" "$run" --gates t36 --gates-file "$gates"
gate_rc=$?
[ "$gate_rc" = 2 ] && job_exit 2
python3 "$M3J" t36-summary "$run/verdict.json" "$dtype" "$loop_rc" "$CHIP_SESSION_DIR/t36_summary.json"
rc=$?
job_log "T-3.6: loop rc=$loop_rc, t36 gate rc=$gate_rc, summary rc=$rc ($CHIP_SESSION_DIR/t36_summary.json)"
job_exit $rc
