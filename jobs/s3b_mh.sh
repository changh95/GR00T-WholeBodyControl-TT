#!/usr/bin/env bash
# Session S3b: the 4-hart LIM bench on L2CPU3 (the user's approval, 2026-10-25: a 4-hart bench added to S3 as its own
# session; tiles 0-2 not allowed; a 16-hart version only after the user has seen this result). It decides whether the
# MLP split across harts 0-3 (phase 2) is worth building.
#
#   X280_E1F_APPROVED=1 X280_MH_APPROVED=1 \
#   CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0 --harts 4" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla none --job $RT/jobs/s3b_mh.sh --tag s3b
#
# One DEVX hold. chip_session boots L2CPU3 with the standard stage-0 (E1F) boot, plus --harts 4: before the release
# x280d writes harts 1-3's RNMI handler = crt0 0x0800_0000 and RNMI exception handler = 0x0800_0200 (the fixed spin;
# hart 0 keeps the park stub), so this session's x280d may RNMI them once (kmd refuses bits 1-3 otherwise). Then:
#   1 s3b_status_before   status: L36.mh (multi-hart bit), the MH region (no stat line yet), the hart-status word
#   2 s3b_adopt           harts adopt: trigger bits 1-3 for 1 ms, cleared; three stat lines IDLE within 50 ms with
#                         mncause 2^63|2, mnepc = stage 0's wfi 0x2001_013A, a moving heartbeat and own BEU accrued 0;
#                         a ping (done mask, wake latency in mtime ticks). A failed adoption marks the harts lost: the
#                         bench is skipped and the tile still serves on hart 0
#   3 s3b_bench_mh        bench --mh: LOAD mh_bench at 0x081D_F400; masks {1,2,3,5,F,E} x K1-K8 x P0-P3, the P4 bank-bit
#                         sweep (K5, 0xF, k 6..12), K3 on hart 0 under K6 on harts 1-3, K9/K10/K12 on {1,3,F}, 3 repeats
#                         each (median); BEU of harts 0-3 and the CCACHE ECC counts before, every 16 CALLs and after (any
#                         accrued event or new ECC fail aborts); hart-0 BENCH v2 x200 idle and under a background K6/P0;
#                         the secondaries' heartbeats after; an INFER parity spot check (x86 build, bit-exact)
#   4 s3b_status_after    status again
#   5 s3b_mh.json         S_N per kernel/pattern, the K9 numbers and the verdict: the split is worth building if
#                         S_4(K9) >= 2.0 or K9 is compute-bound (K9 ~ K10, well above K12); adoption, events, interference
# The working set is read-only weights inside the registered blob; nothing outside L2CPU3's LIM, its CLINT (the harts'
# own IPIs and timers) and hart-status / BEU / ECC reads is touched; no WAYENABLE, WAYMASK, Zero Device or GDDR.
# Exit 0 recorded (a verdict exists), 1 the adoption or the bench failed / aborted, 2 misconfigured, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3

serve_args=" ${CHIP_X280D_SERVE_ARGS:-} "
case $serve_args in
    *" --stage0 "*) ;;
    *) job_log "S3b needs the stage-0 boot: CHIP_X280D_SERVE_ARGS must hold --stage0 (it is '${CHIP_X280D_SERVE_ARGS:-}')"
       exit 2 ;;
esac
case $serve_args in
    *" --harts 4 "*) ;;
    *) job_log "S3b needs harts 1-3's vectors from the boot: CHIP_X280D_SERVE_ARGS must hold --harts 4" \
           "(it is '${CHIP_X280D_SERVE_ARGS:-}'); without them an RNMI would strand the harts until a chip reset"
       exit 2 ;;
esac
if [ "${X280_MH_APPROVED:-0}" != 1 ]; then
    job_log "S3b needs the user's approval of the 4-hart bench in the environment: X280_MH_APPROVED=1"
    exit 2
fi

x280_step s3b_status_before status || job_log "status before adoption failed"
adopted=0
if x280_step s3b_adopt harts adopt; then
    adopted=1
else
    job_log "adoption failed: harts 1-3 marked lost, no bench (the tile serves on hart 0)"
fi
if [ "$adopted" = 1 ]; then
    x280_step s3b_bench_mh bench --mh --mh-plan "${S3B_PLAN:-full}" --mh-repeats "${S3B_REPEATS:-3}" \
        --mh-window-us "${S3B_WINDOW_US:-500}" --mh-bench-iters "${S3B_BENCH_ITERS:-200}" ||
        job_log "bench --mh reported a failure or an abort"
fi
x280_step s3b_status_after status || job_log "status after the bench failed"

python3 - "$CHIP_SESSION_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])


def load(name):
    try:
        return json.loads((d / f"{name}.json").read_text())
    except (OSError, ValueError):
        return {}


adopt, bench = load("s3b_adopt"), load("s3b_bench_mh")
derived = bench.get("derived") or {}
out = {
    "adoption": {k: adopt.get(k) for k in ("pass", "why", "mask", "adopted", "lost", "secs", "harts", "ping",
                                            "hart_status_before", "hart_status_after", "beu_accrued_harts")},
    "bench_pass": bench.get("pass"), "abort": bench.get("abort"), "sim": bench.get("sim"),
    "scaling": derived.get("scaling"), "S4_K9": derived.get("S4_K9"), "S2_K9": derived.get("S2_K9"),
    "k9_k10_k12_cycles_per_row": derived.get("k9_k10_k12_cycles_per_row"),
    "compute_bound": derived.get("compute_bound"), "bank": derived.get("bank"),
    "p4_sweep_agg": derived.get("p4_sweep_agg"), "p4_peak_k": derived.get("p4_peak_k"),
    "chain_cycles": derived.get("chain_cycles"), "t4_estimate_cycles": derived.get("t4_estimate_cycles"),
    "anchors": derived.get("anchors"),
    "split_worth_it": derived.get("split_worth_it"), "verdict": derived.get("verdict") or "no verdict (no bench)",
    "rule": derived.get("rule"), "interference": bench.get("interference"),
    "events_before": bench.get("events_before"), "events_after": bench.get("events_after"),
    "secondaries_after": bench.get("secondaries_after"), "parity": bench.get("parity"),
    "configs": [{k: c.get(k) for k in ("name", "kernel", "kernel0", "mask", "pattern", "kbit", "agg_access_per_tick",
                                        "agg_bytes_per_tick", "gbps", "h0_cycles_per_access", "flags", "skipped")}
                for c in bench.get("configs", [])],
    "note": "phase 2 (the MLP split) only if split_worth_it; a 16-hart version only after the user has seen this result",
}
(d / "s3b_mh.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("verdict", "split_worth_it", "S4_K9", "compute_bound", "bench_pass", "abort")}))
sys.exit(0 if adopt.get("pass") and bench.get("pass") and derived.get("verdict") else 1)
EOF
job_exit $?
