#!/usr/bin/env bash
# T-2.4 stale state (PLAN.md §2 M2, session S1): alternating A/B/A/B obs with fixed noise through predict_debug
# equal the single-call outputs bit-exactly (bench_g1.py --abab 200), plus the 5-replay determinism check.
#
#   $DEV $RT/scripts/chip_session.sh --vla tt --debug-endpoints --job $RT/jobs/t2_abab.sh
#
# A is the first golden observation and B the first later one whose state AND image both differ from A's
# (bench_g1.abab_pair): on n16_g1 that is s00 (reset) and s10 (mid-episode), because s00-s09 are resets with
# bit-identical state and would hide a stale-state bug. T2_ABAB_OBS lists .npz files to use instead; a list with no
# such pair exits 2. Env: T2_ABAB_N (200), T2_GOLDEN. Writes t24_abab.json. Exit 0 pass, 1 fail, 2 misconfigured.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_vla
need_debug_endpoints
obs=()
if [ -n "${T2_ABAB_OBS:-}" ]; then
    for f in $T2_ABAB_OBS; do obs+=(--obs "$f"); done
elif [ -f "$GOLDEN/obs_index.json" ]; then
    obs=(--obs "$GOLDEN")
else
    job_log "no golden obs at $GOLDEN and no T2_ABAB_OBS: synthetic obs would make T-2.4 meaningless"
    exit 2
fi
simpy "$RT/rttt/bench_g1.py" --endpoint "$CHIP_VLA_URL" "${obs[@]}" --abab "${T2_ABAB_N:-200}" --determinism \
    --out "$CHIP_SESSION_DIR/t24_abab.json"
job_exit $?
