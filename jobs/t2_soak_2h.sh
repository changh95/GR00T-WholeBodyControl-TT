#!/usr/bin/env bash
# T-2.8 soak (PLAN.md §2 M2, session S1b): 2 h RT on TT; 0 error replies, 0 device hangs (gates.yaml m2_rt) and a VLA
# p50 drift < 5% between the first and the last 10 minutes (on the server's service time, rttt/m2_checks.py soak).
#
#   CHIP_VLA_STATUS_PUB=$TT_STATUS $DEV $RT/scripts/chip_session.sh --vla tt --job $RT/jobs/t2_soak_2h.sh --tag t28
#
# jobs/t2_soak.sh with T2_SOAK_DURATION=7200, T2_SOAK_TAG=t28 and T2_SOAK_GATE_DRIFT=1 (each one kept if already set);
# every other T2_SOAK_* / T26_BENCH setting works as there. Writes runs_t28/, t28_soak.json, gates_m2.yaml, demo_logs/.
export T2_SOAK_DURATION=${T2_SOAK_DURATION:-7200} T2_SOAK_TAG=${T2_SOAK_TAG:-t28} T2_SOAK_GATE_DRIFT=${T2_SOAK_GATE_DRIFT:-1}
exec "$(dirname "${BASH_SOURCE[0]}")/t2_soak.sh"
