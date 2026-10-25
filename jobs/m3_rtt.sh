#!/usr/bin/env bash
# Session S4, T-3.5 (PLAN.md WP-3.8): x280d end-to-end timing, 10 min at 50 Hz with the VLA running.
#
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla tt --job $RT/jobs/m3_rtt.sh
#   X280_SLOW=1 ...   DP11 bounds (x280d must then run with --slow: CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3
#                     --stage0 --slow")
#
# The session x280d booted with --stage0 (the standard E1F boot since 2026-10-25) or --boot; the job reads neither.
# `x280d bench --rtt --n 30000 --period-ms 20 --alternate` (Balance/Walk alternating) as a W-WBC client of the serving
# x280d while bench_g1 keeps server_g1 busy at the loop's request rate (one call per 320 ms). Per call: client RTT,
# compute_us (mcycle on the hart), x280d recv->send and the mailbox time; host overhead = x280d time - compute.
# The client runs where rt_sim's tick thread runs (CPU 2, its zmq I/O thread on 3, PLAN.md §1.1; no rt_sim runs in
# this job), so its RTT is a loop's: jobs/_x280.sh x280_rtt_client_args (X280_RTT_CLIENT_CPUS='' = unpinned,
# X280_RTT_SPIN=1 = busy-poll the reply).
# Pass = PLAN T-3.5, x280d side only (bench --rtt `checks`): end-to-end p99 <= compute p99 + 0.5 ms and <= 2.0 ms
# (v1/v2); host mailbox overhead p99 <= 0.3 ms; 0 timeouts (and 0 errors). Under DP11: <= compute p99 + 0.5 ms and
# <= 9 ms (or, with early dispatch, the visible wait <= 2.0 ms - that needs the WP-3.8b client, not written yet).
# Reported, not gated: the client RTT (`reported`: ZMQ, msgpack and the client's scheduling on top; m3_rt's
# wbc_rtt_p99_ms gates the loop's own RTT in T-3.7) and the TT latency next to it (m3_rtt_tt.json).
# E1F (2026-10-25, 500 calls, client unpinned): x280d p50 0.66 / p99 0.69 ms on compute 0.385 ms, but host overhead
# p99 301 us: the 0.3 ms gate missed by 1 us (a finding; the gate stands); client RTT p99 2.8 ms.
# Writes m3_rtt.json (bench_g1's last line), m3_rtt_full.json, m3_rtt_tt.json, t35_summary.json. Exit 0 pass,
# 1 fail, 2 misconfigured, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3 attach
need_vla
N=${M3_RTT_N:-30000}
slow=()
[ "${X280_SLOW:-0}" = 1 ] && slow=(--slow)
obs=()
o=$(newest_obs516)
[ -n "$o" ] && obs=(--obs "$o")
mapfile -t client < <(x280_rtt_client_args)
job_log "T-3.5: $N calls at 50 Hz on the session x280d (--$(x280_boot_flag)), client ${client[*]:-unpinned}," \
    "obs ${o:-random}"
tt_latency "$CHIP_SESSION_DIR/m3_rtt_tt.json" $(( N * 20 / 320 + 10 )) 320 &
lat_pid=$!
sleep 5
X280_TIMEOUT_S=$(( N / 50 + 300 )) x280_step m3_rtt bench --rtt --n "$N" --period-ms 20 --alternate \
    --out "$CHIP_SESSION_DIR/m3_rtt_full.json" "${obs[@]}" "${slow[@]}" "${client[@]}"
rc=$?
wait "$lat_pid"
python3 - "$CHIP_SESSION_DIR" "$rc" <<'EOF'
import json, sys
from pathlib import Path
d, rc = Path(sys.argv[1]), int(sys.argv[2])
try:
    r = json.loads((d / "m3_rtt.json").read_text())
except (OSError, ValueError):
    r = {}
try:
    tt = (json.loads((d / "m3_rtt_tt.json").read_text()).get("latency") or {}).get("rtt_ms")
except (OSError, ValueError):
    tt = None
s = {"gates": r.get("gates"), "checks": r.get("checks"), "pass": r.get("pass") is True and rc == 0,
     "x280d_us": r.get("x280d_us"), "compute_us": r.get("compute_us"), "host_overhead_us": r.get("host_overhead_us"),
     "timeouts": r.get("timeouts"), "errors": r.get("errors"), "n": r.get("n"), "ok": r.get("ok"),
     "backend": r.get("backend"), "reported": {**(r.get("reported") or {}), "client_rtt_us": r.get("rtt_us"),
                                               "tt_rtt_ms_during": tt}}
(d / "t35_summary.json").write_text(json.dumps(s, indent=1))
failed = [k for k, v in (s["checks"] or {}).items() if not v]
print(f"T-3.5 (PLAN gates, x280d side): {'PASS' if s['pass'] else 'FAIL'}"
      + (f" on {', '.join(failed)}" if failed else "")
      + f"; reported, not gated: client RTT p99 {((s['reported'].get('client_rtt_us') or {}).get('p99'))} us",
      file=sys.stderr)
print(json.dumps(s))
EOF
job_exit $rc
