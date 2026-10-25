#!/usr/bin/env bash
# Session S4 (b), T-3.7 (PLAN.md WP-3.8, §4.4 M3 screen): closed loop with the X280 driving the legs.
#
#   M3_T23_BENCH=$RT/runs/sessions/20261025_031915_1803710_s1b_t23/t23_bench.json \
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla tt --job $RT/jobs/m3_eval.sh
#   X280_SLOW=1 ...   DP11: gates m3_rt_slow, rt_sim --wbc-timeout-ms 12 (x280d must run --slow:
#                     CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0 --slow")
#
# The session x280d booted with --stage0 (the standard E1F boot since 2026-10-25) or --boot; the job reads neither.
# m3_rt, not m3_rt_slow, is T-3.7's set: E1F measured the v2-fp16 kernel at 0.385 ms compute, x280d p99 0.69 ms
# (DP11's fallback is for a compute p99 > 2 ms). The S1b T-2.3 bench above is the clean one (server span p50 of the
# deployed checkpoint and arena); viewer rows are gated, so run with DISPLAY set (rt_sim --viewer, the aux CPUs).
#   1  Session gate sets: gates.yaml's m3_rt and m3_rt_slow copied whole (x280/tools/m3_jobs.py m3-gates;
#      configs/gates_m3.proposed.yaml is superseded: it lacks the starvation, TT-error, X280-share and heartbeat rows,
#      verify:all D1). vla_p50_ms_max = 1.05 x the T-2.3 server SPAN p50 (srv t_reply - t_recv, what analyze_run
#      gates; never server_total) of M3_T23_BENCH, which must be named (D5: the newest bench may be the wrong arena);
#      M3_T23_BENCH=none keeps gates.yaml's projection and leaves T-3.7's +-5 % row unapplied. fp32 (DP4): shadow
#      p99 5e-5.
#   2  RT soak: M3_EVAL_EPISODES (20) x <= M3_EVAL_MAX_TICKS (1440), VLA on TT, WBC on x280d, ORT shadow (default),
#      --viewer when DISPLAY is set.
#   3  x280d dump through --via: the serving daemon writes its tick ring as $CHIP_SESSION_DIR/x280d_ticks.npy, the
#      data of analyze_run's x280_heartbeat_continuity_min. LIMIT: the ring keeps the last 65,536 W-WBC calls
#      (21.8 min at 50 Hz). This soak is <= 20 x 1440 = 28,800 calls; a run that does not fit is refused here
#      (exit 2), and one that wrapped anyway fails the gate "no data", never a partial pass.
#   4  analyze_run.py --gates m3_rt (m3_rt_slow) --gates-file <session copy> --x280-ticks <the dump>.
#   5  t37_summary.json (m3_jobs.py t37-summary): x280d's own counts across the soak (0 stalls, timeouts, faults,
#      interlock or degraded replies) and TT p50 >= 0.95 x the T-2.3 span (the gate set holds the <= 1.05 x half).
#   6  Screen (50 paired, seeds 2000-2049, vs the M2 configuration TT VLA + GPU WBC): eval_paired.py with a session
#      matrix; gross-drop gate lower 80 % bound >= -25 pts (DP10). M3_SKIP_SCREEN=1 skips it; the GPU WBC must answer
#      (or T2_START_GPU=1, see jobs/_common.sh).
# M3_EVAL_MODE=sync is for CPU smoke tests of this job only (tests/x280/test_m3_jobs.py: mock VLA, x280d --backend
# sim): T-3.7 is the rt run. Under CHIP_SAFE_CPUS the job exits 2 (need_rt_cores) unless it is that sync smoke with
# M3_SKIP_SCREEN=1: rt_sim --mode rt moves itself onto CPUs 2/3 whatever the session's mask.
# Exit 0 all pass, 1 a gate failed, 2 misconfigured, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3 attach
need_vla
M3J=$RT/x280/tools/m3_jobs.py
RING_CALLS=65536  # x280/x280d.py RING
slow=0
[ "${X280_SLOW:-0}" = 1 ] && slow=1
GATE=$([ $slow = 1 ] && echo m3_rt_slow || echo m3_rt)
mode=${M3_EVAL_MODE:-rt}
case $mode in rt | sync) ;; *) job_log "M3_EVAL_MODE=$mode (rt, or sync for smoke tests)"; job_exit 2 ;; esac
# The soak (rt) and the screen (eval_paired --mode rt, the GPU WBC on CPU 4) run on the real-time cores; only the
# sync smoke without the screen may run in a session confined to CHIP_SAFE_CPUS.
if [ "${M3_SKIP_SCREEN:-0}" != 1 ]; then
    need_rt_cores "rt_sim (CPUs 2/3) and, for the screen, the GPU WBC (CPU 4)"
elif [ "$mode" = rt ]; then
    need_rt_cores "rt_sim (CPUs 2/3)"
fi
episodes=${M3_EVAL_EPISODES:-20}
max_ticks=${M3_EVAL_MAX_TICKS:-1440}
if (( episodes * max_ticks >= RING_CALLS )); then
    job_log "$episodes episodes x $max_ticks ticks can exceed x280d's $RING_CALLS-call tick ring (~22 min): the" \
        "heartbeat gate would have no data; run shorter soaks"
    job_exit 2
fi
bench=${M3_T23_BENCH-}
if [ -z "$bench" ]; then
    job_log "set M3_T23_BENCH to the T-2.3 bench of this checkpoint and arena (runs/sessions/<S1>/t23_bench.json)," \
        "or M3_T23_BENCH=none to keep the projection (T-3.7's +-5 % row then not applied)"
    job_exit 2
fi
[ "$bench" = none ] || [ -f "$bench" ] || { job_log "M3_T23_BENCH=$bench: no such file"; job_exit 2; }
gates=$CHIP_SESSION_DIR/gates_m3.yaml
dtype=$(x280 ping | python3 -c "import json,sys; print(json.load(sys.stdin)['info'].get('dtype','fp16'))") ||
    { job_log "x280d $CHIP_X280_URL does not answer ping"; job_exit 2; }
simpy "$M3J" m3-gates "$RT/gates.yaml" "$gates" "$bench" "$dtype" || job_exit 2
# shellcheck disable=SC2206
extra=(${M3_EVAL_LOOP_ARGS-${DISPLAY:+--viewer}})
wbc_to=()
[ $slow = 1 ] && wbc_to=(--wbc-timeout-ms 12)
runs=$CHIP_SESSION_DIR/runs_t37
before=$CHIP_SESSION_DIR/t37_x280d_before.json
after=$CHIP_SESSION_DIR/t37_x280d_after.json
x280 status >"$before"
simpy -m rtloop.loop --mode "$mode" --arm v1 --s 16 --B 0 --d-init 3 --vla "$CHIP_VLA_URL" --wbc "$CHIP_X280_URL" \
    "${wbc_to[@]}" --crn --seeds "${M3_EVAL_SEEDS:-2000-2019}" --n-episodes "$episodes" --max-ticks "$max_ticks" \
    --warmup 20 --tag t37 --runs-root "$runs" "${extra[@]}"
loop_rc=$?
x280 status >"$after"
x280_step t37_ring dump --out "$CHIP_SESSION_DIR" || job_log "x280d dump failed: the heartbeat gate will have no data"
run=$(find "$runs" -mindepth 1 -maxdepth 1 -type d -name "*_t37" 2>/dev/null | sort | tail -n 1)
[ -n "$run" ] || { job_log "the loop wrote no run dir (rc=$loop_rc)"; job_exit 2; }
simpy "$RT/scripts/analyze_run.py" "$run" --gates "$GATE" --gates-file "$gates" \
    --x280-ticks "$CHIP_SESSION_DIR/x280d_ticks.npy"
gate_rc=$?
[ "$gate_rc" = 2 ] && job_exit 2
python3 "$M3J" t37-summary "$run/verdict.json" "$before" "$after" "$CHIP_SESSION_DIR/t37_ring.json" "$bench" \
    "$CHIP_SESSION_DIR/t37_summary.json"
sum_rc=$?
[ "$sum_rc" = 2 ] && job_exit 2
screen_rc=0
if [ "${M3_SKIP_SCREEN:-0}" != 1 ]; then
    need_gpu_servers wbc
    matrix=$CHIP_SESSION_DIR/eval_m3_screen.yaml
    cat >"$matrix" <<EOF
# T-3.7 screen (PLAN.md §4.4): RT C4 with the WBC on the X280 vs RT C4 with the GPU WBC (the M2 configuration).
name: m3_screen
seeds: "2000-2049"
procs: 1
block: 10
max_ticks: 1440
gross_drop: -0.25
common_args: "--mode rt --arm v1 --s 16 --B 0 --vla $CHIP_VLA_URL --d-init 3 --crn --max-ticks 1440 ${wbc_to[*]}"
arms:
  - {name: C4x280, args: "--wbc $CHIP_X280_URL"}
  - {name: C4gpu,  args: "--wbc $WBC_GPU"}
compare:
  - [C4x280, C4gpu]
EOF
    simpy "$RT/scripts/eval_paired.py" --matrix "$matrix" --out "$CHIP_SESSION_DIR/m3_screen"
    screen_rc=$?
fi
job_log "T-3.7: loop rc=$loop_rc, $GATE gates rc=$gate_rc, x280d record + T-2.3 band rc=$sum_rc, screen rc=$screen_rc"
[ "$loop_rc" = 0 ] && [ "$gate_rc" = 0 ] && [ "$sum_rc" = 0 ] && [ "$screen_rc" = 0 ] && job_exit 0
job_exit 1
