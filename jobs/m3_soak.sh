#!/usr/bin/env bash
# Session S5, T-3.9 (PLAN.md WP-3.9, §0.2 M3): the 2 h soak of the final demo, gated: VLA on Tensix (server_g1), WBC
# MLP on the X280 (x280d), the WBC swapped every 10 s, the HUD with both status streams and the recording on.
#
#   X280_E1F_APPROVED=1 CHIP_VLA_STATUS_PUB=$TT_STATUS \
#   CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0 --cpus 5 --status-pub $X280_STATUS" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla tt --job $RT/jobs/m3_soak.sh --tag t39
#
# PLAN.md T-3.9 is `demo.sh m3 --duration 7200 --swap-wbc-every 10`; as a session job (the shape of jobs/t2_soak.sh
# for T-2.6 / T-2.8) that is scripts/demo.sh m3 --in-session --hud --record on the session's server_g1 and x280d
# (booted with --stage0, the standard E1F boot since 2026-10-25, or --boot), with the GPU WBC :5556 as the second
# swap target (demo.sh starts it unless it answers). --swap-wbc-every 10 cycles the loop's WBC x280 -> ort -> gpu ->
# x280 every 10 s of ticks (rtloop.control NEXT; stateless, D2). Then PLAN T-3.9's rows:
#   analyze_run.py --gates t39 (the session set of x280/tools/m3_jobs.py t39-gates): tick period p99 <= 21 ms,
#     0 slips, 0 unplanned fallbacks (wbc_fb: a W-WBC call that failed and was served by ORT; a swap is planned);
#   m3_jobs.py t39-summary: 0 traps (every tile's n_traps unchanged in x280d's status before, during and after) and
#     x280d's own counts clean (0 timeouts, faults, stalls, interlock, degraded); |da| at every swap <= 1e-2 (the ORT
#     shadow of each swap's remote side, TickLog shadow_da); PLL4 at its default values in every 10 s direct read of
#     x280d during the soak (x280d_status.jsonl; no tt-smi), with no read gap over 30 s; loop rc 0.
# The HUD keys (w: swap now, v, r, p) stay live. The swap demo's "proof switch" (RNMI-park the hart: the HUD shows
# the ORT fallback within 1 tick, then the reload) is a planned fallback, so it runs after this job (demo_m3.sh, or
# `xpy $RT/x280/x280d.py park --via tcp://127.0.0.1:5558 --hold-s 2` during an ungated demo), never during the soak.
# x280d's tick ring (65,536 calls) does not hold a 2 h soak: T-3.9 has no heartbeat-continuity row (T-3.7 has).
# Env: M3_SOAK_DURATION (7200 s), M3_SOAK_SWAP_S (10), M3_SOAK_TAG (t39), M3_SOAK_SEEDS (3000-), M3_SOAK_DEMO_ARGS
# (demo.sh options, default "--hud --record"), M3_SOAK_LOOP_ARGS (more rtloop.loop arguments), T2_DRY_RUN=1 (demo.sh
# --dry-run: print the plan, run nothing). Refused under CHIP_SAFE_CPUS (need_rt_cores).
# Writes demo_logs/, runs_<tag>/, gates_t39.yaml, t39_x280d_before.json / _after.json, t39_summary.json.
# Exit 0 pass, 1 fail, 2 misconfigured, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3 attach
need_vla
need_rt_cores "rt_sim (CPUs 2/3) and the HUD (CPUs 6-7)"
M3J=$RT/x280/tools/m3_jobs.py
dur=${M3_SOAK_DURATION:-7200}
# The user (2026-10-25): the GPU WBC has no place in M3, so T-3.9 is an X280-only soak by default (no swaps; the
# HUD's fault/fallback demo uses the host-CPU ORT only). M3_SOAK_SWAP_S=N restores the PLAN's x280 -> ort -> gpu cycle.
swap=${M3_SOAK_SWAP_S:-0}
tag=${M3_SOAK_TAG:-t39}
runs=$CHIP_SESSION_DIR/runs_$tag
gates=$CHIP_SESSION_DIR/gates_$tag.yaml
# shellcheck disable=SC2206  # option lists, split on whitespace
demo_opts=(${M3_SOAK_DEMO_ARGS---hud --record --no-alt})
# shellcheck disable=SC2206
loop_extra=(${M3_SOAK_LOOP_ARGS:-})
case " ${demo_opts[*]} " in
    *" --hud "*)
        [ -n "${CHIP_VLA_STATUS_PUB:-}" ] || job_log "WARNING: CHIP_VLA_STATUS_PUB is not set on the session, so the" \
            "HUD chip panel has no Tensix status (set CHIP_VLA_STATUS_PUB=$TT_STATUS)"
        case " ${CHIP_X280D_SERVE_ARGS:-} ${CHIP_X280_STATUS_PUB:-} " in
            *" --status-pub"* | *tcp://*) ;;
            *) job_log "WARNING: the session x280d has no --status-pub, so the HUD chip panel has no X280 status" \
                   "(CHIP_X280D_SERVE_ARGS=\"... --status-pub $X280_STATUS\")" ;;
        esac ;;
esac
python3 "$M3J" t39-gates "$gates" >/dev/null || job_exit 2

dry=()
job_dry && dry=(--dry-run)
demo=(env DEMO_LOG_ROOT="$CHIP_SESSION_DIR/demo_logs" "$RT/scripts/demo.sh" m3 --in-session "${demo_opts[@]}"
      "${dry[@]}" --vla "$CHIP_VLA_URL" --wbc "$CHIP_X280_URL" --seeds "${M3_SOAK_SEEDS:-3000-}" --duration "$dur"
      --tag "$tag")
[ "$swap" != 0 ] && demo+=(--swap-wbc-every "$swap")
demo+=(-- --s 16 --B 0 --d-init 3 --crn --warmup 20 --runs-root "$runs" "${loop_extra[@]}")
swap_flag=()
[ "$swap" = 0 ] && swap_flag=(--no-swaps)
job_log "T-3.9 soak: ${dur}s RT, VLA $CHIP_VLA_URL, WBC $CHIP_X280_URL (x280d --$(x280_boot_flag)), swap every" \
    "${swap}s (0 = X280 only), tag $tag: ${demo[*]}"
if job_dry; then
    "${demo[@]}" || job_exit 2
    echo "simpy $RT/scripts/analyze_run.py <run dir in $runs> --gates t39 --gates-file $gates"
    echo "simpy $M3J t39-summary <run>/verdict.json <run>/ticks.npy $CHIP_SESSION_DIR/t39_x280d_before.json" \
        "$CHIP_SESSION_DIR/t39_x280d_after.json T0 T1 LOOP_RC $CHIP_SESSION_DIR/t39_summary.json" \
        "$CHIP_SESSION_DIR/x280d_status.jsonl"
    job_exit 0
fi
before=$CHIP_SESSION_DIR/t39_x280d_before.json
after=$CHIP_SESSION_DIR/t39_x280d_after.json
x280 status >"$before" || { job_log "x280d $CHIP_X280_URL gave no status"; job_exit 2; }
t0=$(date +%s.%N)
"${demo[@]}"
loop_rc=$?
t1=$(date +%s.%N)
x280 status >"$after" || job_log "x280d gave no status after the soak: the trap and count rows will fail"
run=$(find "$runs" -mindepth 1 -maxdepth 1 -type d -name "*_$tag" 2>/dev/null | sort | tail -n 1)
[ -n "$run" ] || { job_log "the loop wrote no run dir (demo.sh rc=$loop_rc, logs in $CHIP_SESSION_DIR/demo_logs)"; job_exit 2; }
simpy "$RT/scripts/analyze_run.py" "$run" --gates t39 --gates-file "$gates"
gate_rc=$?
[ "$gate_rc" = 2 ] && job_exit 2
simpy "$M3J" t39-summary "${swap_flag[@]}" "$run/verdict.json" "$run/ticks.npy" "$before" "$after" "$t0" "$t1" "$loop_rc" \
    "$CHIP_SESSION_DIR/t39_summary.json" "$CHIP_SESSION_DIR/x280d_status.jsonl" \
    "$CHIP_SESSION_DIR/x280d_attach/x280d_status.jsonl"
sum_rc=$?
[ "$sum_rc" = 2 ] && job_exit 2
job_log "T-3.9: demo.sh rc=$loop_rc, t39 gates rc=$gate_rc, x280d / swap / PLL4 rows rc=$sum_rc" \
    "($CHIP_SESSION_DIR/t39_summary.json, run $run)"
[ "$loop_rc" = 0 ] && [ "$gate_rc" = 0 ] && [ "$sum_rc" = 0 ] && job_exit 0
job_exit 1
