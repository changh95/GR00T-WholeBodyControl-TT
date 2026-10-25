#!/usr/bin/env bash
# Session S1d (after S1c 2026-10-25, runs/sessions/20261025_053543_2185222_s1c): which host LIM write, if any, leaves
# the L2CPU3 CCACHE's data ECC valid? No hart is released, no TT server runs.
#
#   $DEVX $RT/scripts/chip_session.sh --x280 probe --vla none --job $RT/jobs/s1d.sh --tag s1d
#
# Why: S1c's E1a wrote all of LIM with method A (one 64 B NT store + sfence per line); every line read back right, but
# the init itself counted ~1.9 fix and ~1.9 fail events per line and the re-read 0.935 uncorrectable events per 4 B
# read (E1b's method B, UMD's 2 x 32 B stores over method A's lines, 0.998). Working hypothesis: a host store reaches
# the CCACHE as a read-modify-write of never-initialised SRAM and an uncorrectable pre-read poisons what it writes, so
# a hart load from LIM could fault. S1d writes fresh regions with each store width x280d has and re-reads them.
#
# chip_session has run `x280d probe --json` and started `x280d serve --no-release` (keeper fd and windows, never a
# release) before this job; every step goes to that daemon (--via).
#   1  E0 (x280d e0): L2CPU_RESET 0x0F, WAYENABLE 0, PLL4 defaults and walk-ready, the ECC counters at the start (0
#      on a chip no host has written since its reset). If E0 fails nothing is written.
#   2  x280d lim-ecc (Session.lim_ecc): one never-written region of S1D_LINES lines per plan item, from 0x0801_1000
#      (regs.S1D_BASE) on. Each store goes through kmd.lim_init (method A's checks: whole lines, pre-release only,
#      the l2cpu_batch interlock; the 32 B WC and 8 B UC widths are regs LIM_INIT_WC32 / LIM_INIT_UC8, ctx lim_init
#      only). Per store: the first S1D_PROBE_LINES lines written, then read, one access at a time with the counters
#      read after each; the bulk write; a full 4 B re-read; for an item's last store S1D_PARTIAL 4 B UC writes with
#      read-back, then a second full re-read. The CCACHE ECC registers (fix/fail counts and addresses) are read
#      between phases, never during one.
#   3  E0 again (WAYENABLE and L2CPU_RESET unchanged, the counters at the end).
#   4  s1d_summary.json (x280/tools/m3_jobs.py s1d-summary): per item ECC NOT VALID / CLEAN / DATA MISMATCH, which
#      store (if any) leaves dat_fail_count unchanged across a full re-read, and the lim-init recommendation.
# Plan items (S1D_PLAN, default x280d.S1D_PLAN = a64,a64nf,wc2x32,uc8,a64+uc8): a64 = the boot's method A; a64nf = the
# same stores with one sfence per call; wc2x32 = two 32 B ymm stores per line (UMD memcpy_to_device); uc8 = aligned
# 8 B UC stores; x+y = x, then y over the same lines (does y heal what x left?).
# Env: S1D_PLAN, S1D_LINES (1024), S1D_PROBE_LINES (4), S1D_PARTIAL (16), S1D_SEED (11), S1D_BASE (regs.S1D_BASE).
# Writes s1d_*.json, s1d_summary.json. Exit 0 the measurement is complete (the verdict is in the summary, also when no
# store is clean), 1 a step failed or the measurement is not valid, 2 misconfigured, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 probe
declare -A RCS=()  # not R: env.sh's R is the Isaac-GR00T repo that simpy cd's into
ECC_ARGS=(--lines "${S1D_LINES:-1024}" --probe-lines "${S1D_PROBE_LINES:-4}" --partial "${S1D_PARTIAL:-16}"
          --seed "${S1D_SEED:-11}")
[ -n "${S1D_PLAN:-}" ] && ECC_ARGS+=(--plan "$S1D_PLAN")
[ -n "${S1D_BASE:-}" ] && ECC_ARGS+=(--base "$S1D_BASE")

job_log "S1d step 1: E0 with the keeper fd only (no LIM write yet)"
x280_step s1d_e0 e0; RCS[e0]=$?
if [ "${RCS[e0]}" = 0 ]; then
    job_log "S1d step 2: lim-ecc ${ECC_ARGS[*]}"
    x280_step s1d_lim_ecc lim-ecc "${ECC_ARGS[@]}"; RCS[lim_ecc]=$?
    [ "${RCS[lim_ecc]}" = 3 ] && x280_dirty "lim-ecc: x280d reports the X280 state dirty"
else
    job_log "S1d: E0 failed, so no LIM write in this session"
fi
job_log "S1d step 3: E0 after the writes"
x280_step s1d_e0_after e0; RCS[e0_after]=$?

# shellcheck disable=SC2046  # one step=rc word per step
python3 "$RT/x280/tools/m3_jobs.py" s1d-summary "$CHIP_SESSION_DIR" \
    $(for k in "${!RCS[@]}"; do echo "$k=${RCS[$k]}"; done) >/dev/null
rc=$?
sum=$CHIP_SESSION_DIR/s1d_summary.json
[ -f "$sum" ] || { job_log "no $sum (m3_jobs.py s1d-summary rc=$rc)"; job_exit 2; }
[ "$rc" = 2 ] && job_exit 2
while IFS= read -r line; do
    job_log "$line"
done < <(jget "$sum" "'\n'.join(d['lines'])")
job_log "S1d summary: $sum (verdict $(jget "$sum" "d['recommendation']['verdict']"))"
job_exit $rc
