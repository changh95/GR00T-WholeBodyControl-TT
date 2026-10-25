#!/usr/bin/env bash
# Job of `demo.sh m3` (PLAN.md §0.2 M3, WP-3.9, T-3.9): runs INSIDE the $DEVX hold, as a child of
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0 --cpus 5 [--status-pub URL]" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla tt --job $RT/jobs/demo_m3.sh
# (what `demo.sh m3` runs). chip_session has probed the L2CPU tiles, started x280d (boot:3 with the stage-0 boot,
# E1F, standard since 2026-10-25: the session's one L2CPU3 release, DP3; --boot would serve the same) on :5558 and
# server_g1 on :5557 before this runs; the guard resets the chip after the hold (DEVX). The demo
# (rt_sim with --wbc x280d, ORT as the HUD-w swap target, HUD, recorder) is started and torn down by
# demo.sh --in-session. Nothing here touches the device, x280d's windows or $DEV (D13).
# Exit 3 is reserved for "X280 state dirty" (chip_session tears down and DEVX resets); demo.sh never
# returns 3 for an X280 reason, so its status is passed through unchanged.
# DEMO_ARGS carries the outer demo.sh options, serialised with printf %q.
# The demo IS the real-time workload (rt_sim on CPU 2, x280d on CPU 5, PLAN.md §1.1), so a session confined to the
# non-real-time cores (CHIP_SAFE_CPUS) cannot run it: exit 2 instead of a mis-pinned demo (x280d itself also refuses
# a --cpus outside CHIP_SAFE_CPUS).
set -u
if [ -n "${CHIP_SAFE_CPUS:-}" ]; then
    echo "demo_m3: CHIP_SAFE_CPUS=$CHIP_SAFE_CPUS is set; the M3 demo needs the real-time cores (PLAN.md §1.1)" >&2
    exit 2
fi
RT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
eval "set -- ${DEMO_ARGS:-}"
exec "$RT_DIR/scripts/demo.sh" m3 --in-session "$@"
