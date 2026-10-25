#!/usr/bin/env bash
# T-2.5 sync on TT (PLAN.md §2 M2, session S1b): C1 (--mode sync) on server_g1 vs C1 on the RT-owned GPU VLA :5565,
# 50 paired seeds (configs/eval_m2_sync.yaml), then the stock-rollout anchor against server_g1.
#
#   $DEV $RT/scripts/chip_session.sh --vla tt --job $RT/jobs/t2_sync.sh --tag t25
#
# Needs the GPU WBC :5556 and a GPU VLA :5565 that draws server_g1's noise (gpu_vla_server --noise cpu), so the two
# arms' --crn noise_seeds give the same initial noise; any other GPU VLA is refused. T2_START_GPU=1 starts the missing
# servers as this job's children (with --noise cpu and the §1.1 pins) and stops them at its end. The matrix runs on
# this session's endpoints (session copy m2_sync.yaml), so a session on any port works. Refused under CHIP_SAFE_CPUS.
# Gate (triage): lower 80% bound of (TT - GPU) >= -25 pts, and 0 server errors or hangs (request outcomes ERROR /
# TIMEOUT per arm, rttt/m2_checks.py errors); the anchor is reported against SP-7's 95% CI by hand.
# Env: T2_SYNC_MATRIX (configs/eval_m2_sync.yaml), T2_SYNC_OUT, T2_RESUME=1 (eval_paired --resume into the same
# T2_SYNC_OUT), T2_SKIP_ANCHOR=1, T2_ANCHOR_EPISODES (50), T2_DRY_RUN=1 (print the loop and anchor commands, run
# nothing), T2_GPU_VLA / T2_WBC (other GPU endpoints; T2_WBC=ort: no WBC server).
# Writes m2_sync.yaml, m2_sync/ (report.json, report.md, runs/), m2_sync_errors.json and anchor_rollout.log.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_vla
need_rt_cores
job_dry || need_gpu_servers pair

matrix=$CHIP_SESSION_DIR/m2_sync.yaml
session_matrix "${T2_SYNC_MATRIX:-$RT/configs/eval_m2_sync.yaml}" "$matrix"
out=${T2_SYNC_OUT:-$CHIP_SESSION_DIR/m2_sync}
port=${CHIP_VLA_URL##*:}
anchor=(simpy gr00t/eval/rollout_policy.py --policy_client_host 127.0.0.1 --policy_client_port "$port" --n_envs 1
        --n_episodes "${T2_ANCHOR_EPISODES:-50}" --max_episode_steps 1440 --n_action_steps 20
        --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc)
if job_dry; then
    job_log "T2_DRY_RUN: the T-2.5 loop commands, then the anchor"
    simpy "$RT/scripts/eval_paired.py" --matrix "$matrix" --out "$out" --dry-run || job_exit 2
    [ "${T2_SKIP_ANCHOR:-0}" = 1 ] || echo "${anchor[*]}"
    job_exit 0
fi

resume=()
[ "${T2_RESUME:-0}" = 1 ] && resume=(--resume)
job_log "T-2.5 matrix $matrix -> $out"
simpy "$RT/scripts/eval_paired.py" --matrix "$matrix" --out "$out" "${resume[@]}"
rc=$?
job_log "eval_paired rc=$rc (0 no gross drop, 1 gross drop, 3 incomplete)"
simpy "$RT/rttt/m2_checks.py" errors "$out/runs" "$CHIP_SESSION_DIR/m2_sync_errors.json"
erc=$?
job_log "server errors / hangs per arm: rc=$erc"
[ "$erc" = 2 ] && job_exit 2
[ "$erc" != 0 ] && [ "$rc" = 0 ] && rc=1

if [ "${T2_SKIP_ANCHOR:-0}" != 1 ]; then
    job_log "T-2.5 anchor: stock rollout_policy.py against :$port (${T2_ANCHOR_EPISODES:-50} episodes)"
    "${anchor[@]}" >"$CHIP_SESSION_DIR/anchor_rollout.log" 2>&1
    arc=$?
    job_log "anchor rc=$arc; its summary lines:"
    grep -iE 'success|episodes took' "$CHIP_SESSION_DIR/anchor_rollout.log" | tail -n 5 >&2
    [ "$arc" != 0 ] && [ "$rc" = 0 ] && rc=1
fi
job_exit "$rc"
