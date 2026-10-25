#!/usr/bin/env bash
# T-2.7 comparison (PLAN.md §2 M2, session S1b): RT C4 on TT vs RT C4 on the GPU VLA, n = 50 paired (seeds
# 2000-2049, configs/eval_m2_primary.yaml). Gross-drop gate: lower 80% bound of (TT - GPU) >= -25 pts; delta, CI and
# McNemar p reported; 0 server errors or hangs per arm (rttt/m2_checks.py errors), as for T-2.5.
#
#   $DEV $RT/scripts/chip_session.sh --vla tt --job $RT/jobs/t2_primary.sh --tag t27
#
# Both arms run fresh, interleaved by seed blocks of 10, against a GPU VLA that draws server_g1's noise
# (gpu_vla_server --noise cpu; any other GPU VLA is refused, T2_START_GPU=1 starts one so): the arms are compared with
# --crn, and T-1.9's GPU C4 arm drew CUDA-seeded noise, so it is not noise-paired with TT and is not reused.
# T2_GPU_ARM_DIRS=<GPU C4 run dirs, comma-separated> runs only the TT arm and reports it against those runs, if
# rttt/m2_checks.py reuse accepts them: the same checkpoint, the same rtloop/ code hashes, and GPU runs that drew
# --noise cpu. T2_FORCE_REUSE=1 overrides a checkpoint or code mismatch, never unpaired noise (reuse_check.json).
# The matrix runs on this session's endpoints (session copy m2_primary.yaml). Refused under CHIP_SAFE_CPUS.
# Env: T2_PRIMARY_MATRIX, T2_PRIMARY_OUT, T2_RESUME=1, T2_DRY_RUN=1 (print the loop commands, run nothing),
# T2_GPU_VLA / T2_WBC. Writes m2_primary.yaml, m2_primary/ (report.json, report.md, runs/), m2_primary_errors.json.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_vla
need_rt_cores
matrix=$CHIP_SESSION_DIR/m2_primary.yaml
session_matrix "${T2_PRIMARY_MATRIX:-$RT/configs/eval_m2_primary.yaml}" "$matrix"
out=${T2_PRIMARY_OUT:-$CHIP_SESSION_DIR/m2_primary}
resume=()
[ "${T2_RESUME:-0}" = 1 ] && resume=(--resume)
dry=()
job_dry && dry=(--dry-run)

gate_errors() {  # gate_errors RUNS_ROOT: rc becomes 1 if an arm had server errors or hangs (exit 2 on bad input)
    simpy "$RT/rttt/m2_checks.py" errors "$1" "$CHIP_SESSION_DIR/m2_primary_errors.json"
    case $? in 0) ;; 2) job_exit 2 ;; *) [ "$rc" = 0 ] && rc=1 ;; esac
}

if [ -z "${T2_GPU_ARM_DIRS:-}" ]; then
    job_dry || need_gpu_servers pair
    job_log "T-2.7 matrix $matrix -> $out"
    simpy "$RT/scripts/eval_paired.py" --matrix "$matrix" --out "$out" "${resume[@]}" "${dry[@]}"
    rc=$?
    job_dry && job_exit "$rc"
    job_log "eval_paired rc=$rc (0 no gross drop, 1 gross drop, 3 incomplete)"
    gate_errors "$out/runs"
    job_exit "$rc"
fi

mkdir -p "$out"
force=()
[ "${T2_FORCE_REUSE:-0}" = 1 ] && force=(--force)
# Before an hour of TT episodes: the GPU runs must have drawn server_g1's noise, whatever T2_FORCE_REUSE says.
simpy "$RT/rttt/m2_checks.py" reuse - "$T2_GPU_ARM_DIRS" "$out/reuse_noise_check.json" || {
    job_log "T2_GPU_ARM_DIRS are not noise-paired with TT (see $out/reuse_noise_check.json; T-1.9's arm never is):" \
        "rerun without T2_GPU_ARM_DIRS so a --noise cpu GPU arm runs fresh"
    job_exit 2
}
tt_matrix=$out/eval_m2_primary_tt_only.yaml
simpy "$RT/rttt/m2_checks.py" tt-only "$matrix" "$tt_matrix" || job_exit 2
job_dry || need_gpu_servers wbc
simpy "$RT/scripts/eval_paired.py" --matrix "$tt_matrix" --out "$out/tt_arm" "${resume[@]}" "${dry[@]}"
rc=$?
job_dry && job_exit "$rc"
[ "$rc" = 0 ] || { job_log "TT arm incomplete (eval_paired rc=$rc)"; job_exit "$rc"; }

simpy "$RT/rttt/m2_checks.py" reuse "$out/tt_arm/runs" "$T2_GPU_ARM_DIRS" "$out/reuse_check.json" "${force[@]}" || {
    job_log "the GPU runs are not reusable (see $out/reuse_check.json): rerun without T2_GPU_ARM_DIRS, or" \
        "T2_FORCE_REUSE=1 for a checkpoint or code mismatch"
    job_exit 2
}
tt_dirs=$(find "$out/tt_arm/runs" -mindepth 3 -maxdepth 3 -type d -name '*_m2_primary_C4tt' | sort | paste -sd, -)
[ -n "$tt_dirs" ] || { job_log "no TT run dirs under $out/tt_arm/runs"; job_exit 2; }
simpy "$RT/scripts/eval_paired.py" --report-only "C4tt=$tt_dirs" "C4gpu=$T2_GPU_ARM_DIRS" --compare C4tt,C4gpu \
    --matrix "$matrix" --out "$out"
rc=$?
gate_errors "$out/tt_arm/runs"
job_exit "$rc"
