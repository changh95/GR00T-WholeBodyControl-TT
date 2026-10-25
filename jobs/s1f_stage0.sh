#!/usr/bin/env bash
# Session S2', E1F (PLAN.md WP-3.E1F, §5.2 "E1F stage-0 initializer", DP3(d)): CONTINGENCY ONLY.
#
# Ran 2026-10-25 with the user's DP3(d) approval (runs/sessions/20261025_092909_2787811_e1f): stage 0 zero-filled LIM,
# SELFTEST passed on all 30,720 lines, BEU 0, 0 traps. Since then the stage-0 boot is the standard boot of every M3
# session (jobs/_x280.sh header); this job stays the E1F record and check.
#
# Runs only after T-3.E1a AND T-3.E1b failed with host methods A and B, and only with the user's explicit approval at
# that time (DP3(d): "ask when it arises"). It spends L2CPU3's one release on the hart-run LIM initialiser
# (x280/fw/stage0.S). Nothing runs it automatically; this job refuses unless X280_E1F_APPROVED=1, and so does
# x280d (`serve --stage0` / `boot --stage0` check the same variable before touching anything).
#
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla none --job $RT/jobs/s1f_stage0.sh
#
# x280d's stage-0 boot: interlocks (as §5.2 step 1); GP scratch = stage0.bin, reset vectors -> 0x2001_0100, RNMI
# handler -> the stub entry (a re-run only re-zeroes LIM); PLL4 -> 200 MHz, the one L2CPU_RESET RMW, PLL4 -> 800 MHz;
# no host LIM access until the done word 0x081E_0000 appears at 0x081D_FFC0; then the image (read-back CRC), L2PF,
# SET_NOC_CLEANUP, the entry word 0x081D_FF80 = 0x0800_0000; monitor magic + heartbeat; GP scratch = the park stub;
# SELFTEST; blob LOAD. The reset vectors stay at 0x2001_0100 (used only at reset): the park stub jumps to crt0 at
# the fixed 0x0800_0000, so PARK works after E1F too (review F1; kmd allows only this one GP-scratch swap). This job then checks what E1a could not: host partial 4 B UC writes on hart-initialised spare
# lines (bench --host-only writes and reads back a word there) and the monitor's view (SELFTEST every line, BEU
# accrued 0), plus a heartbeat and mailbox round-trip check.
# Exit 0 pass, 1 fail, 2 not approved / misconfigured, 3 X280 dirty.

set -uo pipefail
if [ "${X280_E1F_APPROVED:-0}" != 1 ]; then
    echo "s1f_stage0.sh: REFUSED. E1F spends the release on the stage-0 initializer and needs the user's explicit" >&2
    echo "  DP3(d) approval, asked when E1a and E1b have both failed. Set X280_E1F_APPROVED=1 only after that." >&2
    exit 2
fi
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3
case " ${CHIP_X280D_SERVE_ARGS:-} " in
    *" --stage0 "*) ;;
    *) job_log "the session's x280d was not started with --stage0 (CHIP_X280D_SERVE_ARGS)"; job_exit 2 ;;
esac
chip_event e1f_approved "by=X280_E1F_APPROVED"
declare -A RCS=()  # not R: env.sh's R is the Isaac-GR00T repo that simpy cd's into
x280_step s1f_status status; RCS[status]=$?
x280_step s1f_selftest selftest; RCS[selftest]=$?
x280_step s1f_hbrate hbrate --secs 0.5; RCS[hbrate]=$?
x280_step s1f_host_partial bench --host-only --n 200; RCS[host_partial]=$?
x280_step s1f_rtt bench --rtt --n 500 --period-ms 20 --alternate; RCS[rtt]=$?
fails=0
for k in "${!RCS[@]}"; do [ "${RCS[$k]}" = 0 ] || fails=$((fails + 1)); done
job_log "E1F: ${#RCS[@]} checks, $fails failed (results in $CHIP_SESSION_DIR/s1f_*.json)"
[ "$fails" = 0 ] && job_exit 0
job_exit 1
