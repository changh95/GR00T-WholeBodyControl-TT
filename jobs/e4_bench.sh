#!/usr/bin/env bash
# Session S3, T-3.E4 (PLAN.md WP-3.7): kernel bench and mailbox RTT; recommends a kernel (the choice is a human one:
# DP4 precision, DP11 timing fallback).
#
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla tt --job $RT/jobs/e4_bench.sh
#
# The session x280d booted with --stage0 (the standard E1F boot since 2026-10-25) or --boot; the job reads neither.
#
#   1  BENCH matrix: every built variant on its weight dtype (v0/v1 fp32 one policy per blob, v2/v2s fp16 both),
#      E4_ITERS (1000) INFERs each, the scalar parity references v0/v2s at most 100 (15-45 ms per call), mcycle per
#      iteration into the spare LIM lines after the blob (x280d bench --matrix). Each BENCH's deadline is 3x a
#      10-iteration pilot, and a timed-out BENCH is finished or RNMI-parked before the next LOAD (review F9).
#      Compare the cycles with the spike instruction counts per inference (kernel lane: v0 1.712M, v1 40.0K,
#      v2 78.3K, v2s 3.588M at VLEN 512) to separate stalls from issue, and re-derive the budget with E2m's
#      measured uncached latency c in x280/R33_ESTIMATE.md (formulas and break-even c per gate there).
#      The -DWBC_RVV_FMA build is a timing data point only (it fails the E5a limit).
#   2  Mailbox RTT through x280d with the VLA idle, then with bench_g1 keeping server_g1 busy (x280d bench --rtt,
#      its client placed as rt_sim's tick thread: jobs/_x280.sh x280_rtt_client_args, X280_RTT_CLIENT_CPUS=2,3).
#      Their `pass` is PLAN T-3.5's x280d-side verdict (recorded here, gated in jobs/m3_rtt.sh).
#   3  e4_choice.json: per variant the worst p99 over policies; targets compute p99 <= 1.0 ms (v1/v2), gate <= 2.0 ms
#      (m3_rt), <= 8.0 ms (m3_rt_slow, DP11); v0 is a parity reference (recorded, not gated).
# Not implemented: option E4c (stack + activations in a Zero-Device pinned L3 way via CCACHE0_WAYMASK0) - it needs a
# firmware build with a different stack placement.
# Exit 0 recorded (a choice exists), 1 no variant meets the m3_rt_slow bound, 2 misconfigured, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3
need_vla

mapfile -t client < <(x280_rtt_client_args)
x280_step e4_matrix bench --matrix --iters "${E4_ITERS:-1000}" || job_log "bench matrix reported a failure"
x280_step e4_rtt_idle bench --rtt --n "${E4_RTT_N:-3000}" --period-ms 20 --alternate "${client[@]}"
tt_latency "$CHIP_SESSION_DIR/e4_tt_busy.json" "${E4_TT_N:-400}" 160 &
lat_pid=$!
sleep 3
x280_step e4_rtt_busy bench --rtt --n "${E4_RTT_N:-3000}" --period-ms 20 --alternate "${client[@]}"
wait "$lat_pid"

python3 - "$CHIP_SESSION_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
m = json.loads((d / "e4_matrix.json").read_text())
worst = {}
for r in m.get("runs", []):
    k = f"{r['variant']}-{r['dtype']}"
    if r.get("us_p99") is not None:
        worst[k] = max(worst.get(k, 0.0), r["us_p99"])
def rtt(n):
    try:
        return json.loads((d / n).read_text())
    except (OSError, ValueError):
        return {}
idle, busy = rtt("e4_rtt_idle.json"), rtt("e4_rtt_busy.json")
cands = {k: v for k, v in worst.items() if not k.startswith("v0")}
best = min(cands, key=cands.get) if cands else None
tier = None
if best is not None:
    p = cands[best]
    tier = "target (<= 1.0 ms)" if p <= 1000 else "m3_rt (<= 2.0 ms)" if p <= 2000 else \
        "m3_rt_slow (<= 8.0 ms, DP11 + early dispatch)" if p <= 8000 else "shadow only (> 8 ms, DP11)"
out = {"compute_p99_us_by_variant": worst, "recommended": best, "tier": tier,
       "rtt_idle_us": idle.get("rtt_us"), "rtt_busy_us": busy.get("rtt_us"),
       "host_overhead_idle_us": idle.get("host_overhead_us"), "host_overhead_busy_us": busy.get("host_overhead_us"),
       "x280d_idle_us": idle.get("x280d_us"), "x280d_busy_us": busy.get("x280d_us"),
       "t35_gates_idle": idle.get("checks"), "t35_gates_busy": busy.get("checks"),
       "rtt_client": (busy.get("reported") or {}).get("client"),
       "note": "recommendation only: DP4 (precision) and DP11 (timing fallback) are the user's decisions"}
(d / "e4_choice.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
sys.exit(0 if tier and not tier.startswith("shadow") else 1)
EOF
job_exit $?
