#!/usr/bin/env bash
# Session S4, T-3.8 (PLAN.md WP-3.8): fault injection while rt_sim runs with the X280 driving the legs.
#
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0" \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla tt --job $RT/jobs/m3_faults.sh
#
# The session x280d booted with --stage0 (the standard E1F boot since 2026-10-25) or --boot: either way the restarted
# x280d is `serve --attach` with the session's other serve flags (jobs/_x280.sh x280_reattach), never a second boot.
# rt_sim runs RT (--wbc-no-latch, so every tick retries x280d) for M3_FAULT_DURATION (150 s); meanwhile:
#   t0+30 s  case 1: kill -9 x280d. Its keeper fd closes, the KMD writes 'GONE' into L38 and the monitor idles
#            (state 5); rt_sim falls back to ORT on the same tick. A new `x280d serve --attach` (x280_reattach: a
#            child of this job, recorded in x280d.pid so chip_session's teardown stops it, on CPU 5 as the session's)
#            attaches WITHOUT a release - server_g1's fd keeps the L2CPU clock on - clears L38, re-registers
#            SET_NOC_CLEANUP and serves again (after a stage-0 boot too: LIM was zero-filled by the hart, and the
#            reset vectors that still point at the stage-0 stub are consumed only at reset).
#   t0+60 s  case 2: `x280d park --hold-s 2` (RNMI PARK protocol, the hart held in the park stub for 2 s, then
#            released: warm entry, n_reentry + 1). Should x280d die inside that hold, `serve --attach` finds the
#            trigger bit still set and finishes the PARK protocol itself (review F8).
#   t0+90 s  case 3: `x280d inject crc`: the next INFER carries a wrong obs CRC; the monitor retries 50 us, reports
#            FAULT 0xC0C, x280d answers with an error.
# Pass (T-3.8): for each case rt_sim falls back within 1 tick (ORT on the same call), no SystemExit (SAFETY_FAULT),
# no fall, the episode continues, and the restarted x280d did not write L2CPU_RESET (its log has no release step).
# Env: M3_FAULT_DURATION (150 s), M3_FAULT_GAP_S (30 s between the cases; >= 3, and the run must outlast case 3),
# M3_FAULT_SEEDS (2100-); M3_LOOP_MODULE (default rtloop.loop; a .py path runs as a script) is a CPU-test hook only:
# tests/x280 put a stand-in there that records its command line, so no --mode rt loop ever runs in a test.
# Writes runs_t38/, t38_events.json, t38_summary.json. Exit 0 pass, 1 fail, 2 misconfigured (also: CHIP_SAFE_CPUS
# is set, need_rt_cores), 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3 attach
need_vla
need_rt_cores "rt_sim (CPUs 2/3)"
DUR=${M3_FAULT_DURATION:-150}
GAP=${M3_FAULT_GAP_S:-30}
if ! [[ $DUR =~ ^[0-9]+$ && $GAP =~ ^[0-9]+$ ]] || [ "$GAP" -lt 3 ] || [ "$DUR" -le $((3 * GAP + 10)) ]; then
    job_log "M3_FAULT_DURATION=$DUR / M3_FAULT_GAP_S=$GAP: whole seconds, gap >= 3, duration > 3 x gap + 10"
    job_exit 2
fi
loop_mod=${M3_LOOP_MODULE:-rtloop.loop}
case $loop_mod in *.py) loop=(simpy "$loop_mod") ;; *) loop=(simpy -m "$loop_mod") ;; esac
job_log "T-3.8 on the session x280d (--$(x280_boot_flag)): cases at +${GAP}s, +$((2 * GAP))s, +$((3 * GAP))s of ${DUR}s"
runs=$CHIP_SESSION_DIR/runs_t38
ev=$CHIP_SESSION_DIR/t38_events.jsonl
: >"$ev"
note() { echo "{\"t_ns\": $(date +%s%N), \"mono_ns\": $(python3 -c 'import time; print(time.monotonic_ns())'), \"event\": \"$1\"}" >>"$ev"; }

"${loop[@]}" --mode rt --arm v1 --s 16 --B 0 --d-init 3 --vla "$CHIP_VLA_URL" --wbc "$CHIP_X280_URL" \
    --wbc-no-latch --crn --seeds "${M3_FAULT_SEEDS:-2100-}" --duration "$DUR" --max-ticks 1440 --tag t38 \
    --runs-root "$runs" >"$CHIP_SESSION_DIR/t38_loop.log" 2>&1 &
loop_pid=$!
sleep "$GAP"
pid=$(chip_pidfile_pid x280d) || { job_log "no live x280d before case 1"; kill "$loop_pid"; job_exit 2; }
note kill9_x280d
kill -KILL "$pid"
sleep 2
note attach_start
x280_reattach || { note attach_failed; kill "$loop_pid"; wait "$loop_pid"
                   x280_dirty "T-3.8: x280d could not attach after kill -9"; }
note attach_ready
sleep $((GAP - 2))
note park_start
x280_step t38_park park --hold-s 2
note park_done
sleep $((GAP - 2))
note inject_crc
x280_step t38_inject inject crc
wait "$loop_pid"
loop_rc=$?
run=$(find "$runs" -mindepth 1 -maxdepth 1 -type d -name "*_t38" 2>/dev/null | sort | tail -n 1)
[ -n "$run" ] || { job_log "the loop wrote no run dir (rc=$loop_rc)"; job_exit 2; }
simpy - "$run" "$ev" "$CHIP_SESSION_DIR/x280d.attach.log" "$CHIP_SESSION_DIR/t38_summary.json" "$loop_rc" <<'EOF'
import json, sys
import numpy as np
run, evf, attach_log, out, loop_rc = sys.argv[1:6]
t = np.load(f"{run}/ticks.npy")
ev = [json.loads(x) for x in open(evf) if x.strip()]
eps = [json.loads(x) for x in open(f"{run}/episodes.jsonl") if x.strip()]
fb = t["wbc_fb"].astype(bool)
res = {"cases": {}, "fallback_ticks": int(fb.sum()), "safety_faults": int(sum(e.get("end") == "SAFETY_FAULT" for e in eps)),
       "falls": int(sum(e.get("end") == "FALL" for e in eps)), "episodes": len(eps), "loop_rc": int(loop_rc)}
for name in ("kill9_x280d", "park_start", "inject_crc"):
    e = next((x for x in ev if x["event"] == name), None)
    if e is None:
        res["cases"][name] = {"ran": False}
        continue
    after = (t["t_start_ns"] >= e["mono_ns"]) & (t["t_start_ns"] < e["mono_ns"] + 20_000_000_000)
    i = np.where(after & fb)[0]
    first = int(t["k"][i[0]] - t["k"][after][0]) if i.size else None
    # "within 1 tick": the first tick whose WBC call failed is served by ORT on that same call (fallback flag set),
    # so the leg command never misses a tick; the gap between the event and that tick is how long the call took.
    res["cases"][name] = {"ran": True, "fallback_ticks": int(i.size), "first_fallback_tick_after_event": first,
                          "max_period_ms_after": float(np.max(np.diff(t["t_start_ns"][after])) / 1e6) if after.sum() > 1 else None}
log = open(attach_log).read() if attach_log else ""
res["attach_released"] = "L2CPU_RESET RMW" in log or "step 7:" in log
res["attach_ok"] = "READY" in log and "attach:" in log
ok_cases = all(c.get("ran") and c["fallback_ticks"] > 0 and (c["max_period_ms_after"] or 99) < 30 for c in res["cases"].values())
res["pass"] = bool(ok_cases and not res["safety_faults"] and not res["falls"] and not res["attach_released"]
                   and res["attach_ok"] and res["loop_rc"] == 0)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
sys.exit(0 if res["pass"] else 1)
EOF
job_exit $?
