#!/usr/bin/env bash
# T-2.6 async RT soak (30 min) on TT, and T-2.8 (2 h) through jobs/t2_soak_2h.sh (PLAN.md §2 M2, session S1b).
#
#   CHIP_VLA_STATUS_PUB=$TT_STATUS $DEV $RT/scripts/chip_session.sh --vla tt --job $RT/jobs/t2_soak.sh --tag t26
#
# PLAN.md T-2.6 is `demo.sh m2 --duration 1800 --tag t26` as a session job plus analyze_run.py --gates m2_rt, and §4.3
# counts the timing gates only with the HUD and recording on. So the job runs scripts/demo.sh m2 --in-session with
# --hud --record (viewer on; demo.sh starts the GPU WBC :5556 unless it answers; --no-alt: no GPU VLA swap target is
# started next to the TT run), passes the soak's loop arguments after `--`, and then gates the run. The HUD chip
# panel's Tensix column needs server_g1's status stream: CHIP_VLA_STATUS_PUB on the session (warned if unset).
# Gates: gates.yaml's m2_rt in a session copy (gates_m2.yaml) whose vla_p50_ms_max is 1.1 x the server p50 of the
# T-2.3 baseline: T26_BENCH=<t23_bench.json>, else the newest passing bench of the deployed bfp8_b arena under the
# session root (rttt/m2_checks.py bench, printed); none = exit 2 (T26_BENCH=projected keeps gates.yaml's 49.5 ms).
# Then rttt/m2_checks.py soak: the HUD's placement of the run's VLA (vla=tensix) and the p50 drift on the server
# time, gated with T2_SOAK_GATE_DRIFT=1 (T-2.8) and reported otherwise. Refused under CHIP_SAFE_CPUS.
# Env: T2_SOAK_DURATION (1800 s), T2_SOAK_TAG (t26), T2_SEEDS (3000-), T2_WBC (the GPU WBC; 'ort' allowed),
# T2_SOAK_DEMO_ARGS (demo.sh options, default "--hud --record --no-alt"; with --no-viewer the viewer rows are dropped),
# T2_SOAK_LOOP_ARGS (more rtloop.loop arguments), T2_DRY_RUN=1 (demo.sh --dry-run: print the plan, run nothing).
# Writes gates_m2.yaml, demo_logs/ (loop, HUD, recording), runs_<tag>/, <tag>_soak.json. Exit 0 pass, 1 fail, 2 misconfigured.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_vla
need_rt_cores
dur=${T2_SOAK_DURATION:-1800}
tag=${T2_SOAK_TAG:-t26}
runs=$CHIP_SESSION_DIR/runs_$tag
gates=$CHIP_SESSION_DIR/gates_m2.yaml
# shellcheck disable=SC2206  # option lists, split on whitespace
demo_opts=(${T2_SOAK_DEMO_ARGS---hud --record --no-alt})
# shellcheck disable=SC2206
loop_extra=(${T2_SOAK_LOOP_ARGS:-})
viewer=()
for a in "${demo_opts[@]}"; do [ "$a" = --no-viewer ] && viewer=(--no-viewer); done
case " ${demo_opts[*]} " in
    *" --hud "*) [ -n "${CHIP_VLA_STATUS_PUB:-}" ] || job_log "WARNING: CHIP_VLA_STATUS_PUB is not set on the" \
        "session, so the HUD chip panel has no Tensix status (set CHIP_VLA_STATUS_PUB=$TT_STATUS)" ;;
esac

case ${T26_BENCH:-} in
    projected) bench="" ;;
    "") bench=$(simpy "$RT/rttt/m2_checks.py" bench "$(dirname "$CHIP_SESSION_DIR")") || {
            job_log "no passing T-2.3 bench of the deployed arena under $(dirname "$CHIP_SESSION_DIR"): run" \
                "jobs/t2_bench.sh first, or set T26_BENCH=<t23_bench.json> (or T26_BENCH=projected)"
            job_exit 2
        } ;;
    *) bench=$T26_BENCH
       [ -f "$bench" ] || { job_log "T26_BENCH=$bench is not a file"; job_exit 2; } ;;
esac
job_log "T-2.3 baseline of the VLA p50 gate: ${bench:-none, the projected limit (T26_BENCH=projected)}"
simpy "$RT/rttt/m2_checks.py" gates "$RT/gates.yaml" "$gates" "$bench" "${viewer[@]}" || job_exit 2

dry=()
job_dry && dry=(--dry-run)
demo=(env DEMO_LOG_ROOT="$CHIP_SESSION_DIR/demo_logs" "$RT/scripts/demo.sh" m2 --in-session "${demo_opts[@]}"
      "${dry[@]}" --vla "$CHIP_VLA_URL" --wbc "$GPU_WBC_URL" --seeds "${T2_SEEDS:-3000-}" --duration "$dur"
      --tag "$tag" -- --s 16 --B 0 --d-init 3 --crn --warmup 20 --runs-root "$runs" "${loop_extra[@]}")
drift=()
[ "${T2_SOAK_GATE_DRIFT:-0}" = 1 ] && drift=(--gate-drift)
job_log "soak: ${dur}s RT on $CHIP_VLA_URL, WBC $GPU_WBC_URL, tag $tag: ${demo[*]}"
if job_dry; then
    "${demo[@]}" || job_exit 2
    echo "simpy $RT/scripts/analyze_run.py <run dir in $runs> --gates m2_rt --gates-file $gates"
    echo "simpy $RT/rttt/m2_checks.py soak <run dir> $dur $CHIP_SESSION_DIR/${tag}_soak.json ${drift[*]}"
    job_exit 0
fi
"${demo[@]}"
loop_rc=$?
run=$(find "$runs" -mindepth 1 -maxdepth 1 -type d -name "*_$tag" 2>/dev/null | sort | tail -n 1)
[ -n "$run" ] || { job_log "the loop wrote no run dir (demo.sh rc=$loop_rc, logs in $CHIP_SESSION_DIR/demo_logs)"; job_exit 2; }

simpy "$RT/scripts/analyze_run.py" "$run" --gates m2_rt --gates-file "$gates"
gate_rc=$?
[ "$gate_rc" = 2 ] && job_exit 2
simpy "$RT/rttt/m2_checks.py" soak "$run" "$dur" "$CHIP_SESSION_DIR/${tag}_soak.json" "${drift[@]}"
extra_rc=$?
job_log "soak: demo.sh rc=$loop_rc, m2_rt gates rc=$gate_rc, soak checks rc=$extra_rc (run $run)"
[ "$extra_rc" = 2 ] && job_exit 2
[ "$loop_rc" = 0 ] && [ "$gate_rc" = 0 ] && [ "$extra_rc" = 0 ] && job_exit 0
job_exit 1
