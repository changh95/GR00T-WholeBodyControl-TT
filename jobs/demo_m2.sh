#!/usr/bin/env bash
# Job of `demo.sh m2` (PLAN.md §0.2 M2, T-2.6): runs INSIDE the $DEV hold, as a child of
#   $DEV $RT/scripts/chip_session.sh --x280 none --vla tt --job $RT/jobs/demo_m2.sh
# chip_session has started server_g1 (CHIP_VLA_URL, default :5557) and waited for READY. The demo itself
# (WBC GPU server :5556, the GPU VLA :5565 as the HUD-v swap target, rt_sim, HUD, recorder) is started and
# torn down by demo.sh --in-session; nothing here opens the device or nests $DEV (D13).
# DEMO_ARGS carries the outer demo.sh options, serialised with printf %q. The VLA is the session's server_g1
# (--vla $CHIP_VLA_URL first, so a --vla in DEMO_ARGS still wins), whatever port the session serves it on.
# T-2.6 / T-2.8 with their gates (the T-2.3-derived VLA p50 limit, the HUD placement, the drift) run through
# jobs/t2_soak.sh / t2_soak_2h.sh, which call demo.sh m2 --in-session the same way.
# The demo IS the real-time workload (rt_sim on CPU 2, GPU servers on 4 and 12-15, PLAN.md §1.1), so a session
# confined to the non-real-time cores (CHIP_SAFE_CPUS) cannot run it: exit 2 instead of a mis-pinned demo.
set -u
if [ -n "${CHIP_SAFE_CPUS:-}" ]; then
    echo "demo_m2: CHIP_SAFE_CPUS=$CHIP_SAFE_CPUS is set; the M2 demo needs the real-time cores (PLAN.md §1.1)" >&2
    exit 2
fi
RT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
eval "set -- ${DEMO_ARGS:-}"
exec "$RT_DIR/scripts/demo.sh" m2 --in-session ${CHIP_VLA_URL:+--vla "$CHIP_VLA_URL"} "$@"
