#!/usr/bin/env bash
# T-2.3 TT latency through ZMQ (PLAN.md §2 M2, session S1): bench_g1.py --n 500 --period-ms 320, then the
# determinism part of T-2.3 (two unseeded calls differ; fixed-noise replays and equal noise_seed are identical).
#
#   $DEV $RT/scripts/chip_session.sh --vla tt --debug-endpoints --job $RT/jobs/t2_bench.sh
#
# The thresholds are projections: a latency FAIL with d p99 <= 4 means "re-baseline the latency rows in gates.yaml
# with a recorded rationale", a human decision, so the job reports and exits 1.
# With CHIP_VLA_STATUS_PUB set (server_g1 --status-pub), the latency run also checks that status stream (§4.3 item 7).
# Env: T2_BENCH_N (500), T2_BENCH_PERIOD_MS (320), T2_GOLDEN. Writes t23_bench.json, t23_determinism.json and,
# unless an earlier job of the session did, server_g1_info.json.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_vla
need_debug_endpoints
# server_g1's INFO (stamp, arena): the configuration this bench measured.
[ -f "$CHIP_SESSION_DIR/server_g1_info.json" ] || save_server_info
obs=()
[ -f "$GOLDEN/obs_index.json" ] && obs=(--obs "$GOLDEN")
status=()
[ -n "${CHIP_VLA_STATUS_PUB:-}" ] && status=(--status-sub "$CHIP_VLA_STATUS_PUB")

simpy "$RT/rttt/bench_g1.py" --endpoint "$CHIP_VLA_URL" "${obs[@]}" --n "${T2_BENCH_N:-500}" \
    --period-ms "${T2_BENCH_PERIOD_MS:-320}" --keep-calls "${status[@]}" --out "$CHIP_SESSION_DIR/t23_bench.json"
rc_lat=$?
simpy "$RT/rttt/bench_g1.py" --endpoint "$CHIP_VLA_URL" "${obs[@]}" --determinism \
    --out "$CHIP_SESSION_DIR/t23_determinism.json"
rc_det=$?
job_log "T-2.3 latency rc=$rc_lat determinism rc=$rc_det (reports in $CHIP_SESSION_DIR)"
for rc in $rc_lat $rc_det; do [ "$rc" = 2 ] && job_exit 2; done
[ "$rc_lat" = 0 ] && [ "$rc_det" = 0 ] && job_exit 0
job_exit 1
