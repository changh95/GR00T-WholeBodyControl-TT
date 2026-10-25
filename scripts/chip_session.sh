#!/usr/bin/env bash
# chip_session.sh: one device session on the p150a, run INSIDE the guard's hold (PLAN.md WP-2.4, D13, §5.2).
#
#   $DEV  $RT/scripts/chip_session.sh [--vla none|tt] [--debug-endpoints] --job <script> [--job <script> ...]
#   $DEVX $RT/scripts/chip_session.sh --x280 probe|boot:3|attach [--vla none|tt] [--debug-endpoints] [--allow-reset]
#         --job <script> [--tag T]
#
# bin/with-device.sh takes the flock and runs bin/_device_guard.sh, which probes Tensix and runs this script as its
# child (with-device.sh:8-10, _device_guard.sh:113-119); this script holds that one flock for the whole session.
# Order, exactly as WP-2.4:
#   1. --x280 != none: L2CPU health check `x280d.py probe --json` (10 s timeout), which the guard's Tensix-only
#      probe cannot do (_device_guard.sh:31-44).
#   2. Dirty and --allow-reset: fuser on the node must be empty -> tt-smi -r 0 -> wait 5 s -> scripts/tt_probe.py
#      (the guard's probe snippet) -> step 1 again. Dirty without --allow-reset: exit 3.
#   3. --x280 != none: x280d serve (it holds the keeper fd for the whole session, §5.2 invariant 3).
#   4. --vla tt: server_g1 (scripts/start_vla.sh), wait until it answers ping (= past its boot check).
#   5. The job(s), with CHIP_VLA_URL / CHIP_X280_URL / CHIP_SESSION_DIR exported (env.sh's VLA_TT/WBC_X280 are
#      re-exported by every `source env.sh`, so jobs read the CHIP_* names).
#   6. Teardown in reverse (job, server_g1, x280d, then anything else carrying this CHIP_SESSION_ID): SIGTERM,
#      SIGKILL after 10 s, so the guard's post-hold reset finds no device users (_device_guard.sh:122-130).
# Every (re)start is a child of this session; $DEV is never called in here: a nested hold blocks on the flock (R29),
# so jobs get DEVICE_LOCK_TIMEOUT=1 and a stray nested $DEV fails after 1 s instead of hanging for 7200 s.
#
# Preconditions (exit 2): an ancestor is the guard under flock (unless the fake mode); --x280 != none needs DEVX
# (WITH_DEVICE_RESET_AFTER=1, D13) and a probed hold (WITH_DEVICE_NO_PROBE=1 skips the guard's post-job reset,
# _device_guard.sh:79-86). TT_METAL_WATCHER and TT_METAL_TRACE_ALLOC_TRACKING are unset (§5.2 invariant 5, R32).
#
# x280d probe contract (stdout): one JSON object with a boolean "dirty" ("reasons": [str] optional; with
# --x280 attach, "attachable": true accepts a released tile with a live monitor). Exit 0 = report, exit 3 = dirty;
# a probe killed at the timeout counts as dirty (it hung on a tile access); any other exit, or no "dirty", is a
# probe error (exit 4: a reset would be a guess). chip_session never interprets registers itself: x280/regs.py is
# the only source of addresses and expected values (PLAN.md WP-3.3).
#
# x280d serve contract: `serve --port P --tile 3 --no-release|--stage0|--attach` for --x280 probe|boot:3|attach
# (override with CHIP_X280D_SERVE_ARGS, passed through as given). boot:3 is the stage-0 boot (E1F, the standard boot
# since 2026-10-25: the hart initialises LIM, since host-written LIM init poisons the LIM ECC, S1c/S1d), so without
# CHIP_X280D_SERVE_ARGS it needs the user's DP3(d) approval X280_E1F_APPROVED=1 in the environment and is
# refused (exit 2, before anything runs) without it; x280d itself refuses a device --boot (method A) unless
# X280_ALLOW_METHOD_A=1. Plus `--cpus 5` (PLAN.md §1.1's x280d row) unless the arguments already
# set --cpus (or CHIP_SAFE_CPUS excludes CPU 5), and `--status-pub $CHIP_X280_STATUS_PUB` when that is set
# (e.g. $X280_STATUS, the HUD chip panel's X280 source, §4.3 item 7) unless they already set --status-pub. It is
# off by default: the stream adds a status snapshot after every reply, which gated timing runs must not pay for
# unless they show the HUD (demo.sh --hud passes it itself). chip_lib.sh x280d_start adds both, also for the jobs
# that start x280d again (jobs/_x280.sh x280_reattach: T-3.8, S2's E3(d)). It answers W-WBC `ping` once its keeper
# fd and windows are up (and, for --boot or --stage0, once the monitor heartbeat is live). Exit 3 before READY = X280
# dirty.
#
# Exit codes: 0 all jobs passed; the first failing job's code otherwise (3 wins); 2 usage or precondition;
# 3 X280 dirty (probe, reset impossible or ineffective, x280d refused, or a job exited 3); 4 probe/reset/tt_probe
# failure; 5 x280d or server_g1 did not start, or died before a job; 6 a child survived SIGKILL; 128+N on signal N.
#
# Environment: CHIP_SESSION_ROOT (default $RT/runs/sessions), CHIP_{VLA,X280}_PORT, CHIP_DEV_NODE, CHIP_*_CMD
# (X280D, VLA, TT_PROBE, TTSMI, FUSER), CHIP_X280D_SERVE_ARGS, CHIP_VLA_EXTRA_ARGS, CHIP_VLA_STATUS_PUB (server_g1
# --status-pub URL, exported to jobs), CHIP_X280_STATUS_PUB (x280d --status-pub URL, exported to jobs),
# CHIP_SAFE_CPUS (pin the session, hence every child, to a CPU list), timeouts
# CHIP_X280_PROBE_TIMEOUT_S (10), CHIP_RESET_TIMEOUT_S (300, as the guard), CHIP_RESET_WAIT_S (5),
# CHIP_TT_PROBE_TIMEOUT_S (60, as the guard), CHIP_X280_READY_TIMEOUT_S (120), CHIP_VLA_READY_TIMEOUT_S (3600),
# CHIP_TEARDOWN_GRACE_S (10). A SIGTERM is acted on within about a second at every step, also while server_g1 builds.
# Fake mode CHIP_SESSION_FAKE=1 (tests/tt/test_chip_session_fake.py): every device command and CHIP_DEV_NODE must be
# overridden (and the node must not be under /dev/tenstorrent), or the script exits 2; the guard-ancestor check is
# skipped unless CHIP_GUARD_CHECK=1.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RT_DIR=$(dirname "$HERE")
export CHIP_LOG_NAME=chip_session
# shellcheck source=scripts/chip_lib.sh
source "$HERE/chip_lib.sh"

: "${CHIP_X280_PROBE_TIMEOUT_S:=10}"
: "${CHIP_RESET_TIMEOUT_S:=300}"
: "${CHIP_RESET_WAIT_S:=5}"
: "${CHIP_TT_PROBE_TIMEOUT_S:=${DEVICE_PROBE_TIMEOUT:-60}}"
: "${CHIP_X280_STATUS_PUB=}"  # off unless set (e.g. to env.sh's X280_STATUS); see the serve contract above

usage() { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() {  # die RC MESSAGE (before the session dir exists)
    echo "chip_session: $2" >&2
    exit "$1"
}

case " $* " in  # the guard execs any command line containing tt-smi without its probe or post-hold reset
    *tt-smi*) die 2 "an argument contains 'tt-smi': bin/_device_guard.sh:76-78 would skip its probe and the DEVX reset" ;;
esac
X280_MODE=none VLA_MODE=none DEBUG_EP=0 ALLOW_RESET=0 TAG="" JOBS=()
while [ $# -gt 0 ]; do
    case $1 in
        --x280) X280_MODE=${2:-}; shift 2 || die 2 "--x280 needs a value" ;;
        --x280=*) X280_MODE=${1#*=}; shift ;;
        --vla) VLA_MODE=${2:-}; shift 2 || die 2 "--vla needs a value" ;;
        --vla=*) VLA_MODE=${1#*=}; shift ;;
        --debug-endpoints) DEBUG_EP=1; shift ;;
        --allow-reset) ALLOW_RESET=1; shift ;;
        --job) [ $# -ge 2 ] || die 2 "--job needs a script"; JOBS+=("$2"); shift 2 ;;
        --job=*) JOBS+=("${1#*=}"); shift ;;
        --tag) TAG=${2:-}; shift 2 || die 2 "--tag needs a value" ;;
        -h | --help) usage; exit 0 ;;
        *) die 2 "unknown argument: $1 (see --help)" ;;
    esac
done

case $X280_MODE in none | probe | boot:3 | attach) ;; *) die 2 "--x280 must be none|probe|boot:3|attach, got '$X280_MODE'" ;; esac
if [ "$X280_MODE" = boot:3 ] && [ -z "${CHIP_X280D_SERVE_ARGS:-}" ] && [ "${X280_E1F_APPROVED:-0}" != 1 ]; then
    die 2 "--x280 boot:3 boots L2CPU3 with the stage-0 initializer (x280d serve --stage0, E1F: the standard boot since \
2026-10-25; host-written LIM init poisons the LIM ECC, S1c/S1d), which needs the user's DP3(d) approval: \
X280_E1F_APPROVED=1 is not set (or give CHIP_X280D_SERVE_ARGS explicitly)"
fi
case $VLA_MODE in none | tt) ;; *) die 2 "--vla must be none|tt, got '$VLA_MODE'" ;; esac
[ ${#JOBS[@]} -gt 0 ] || die 2 "--job <script> is required"
for j in "${JOBS[@]}"; do [ -f "$j" ] && [ -r "$j" ] || die 2 "job $j is not a readable file"; done

# --- preconditions (nothing has touched the device yet) ------------------------------------------------------------

FAKE=${CHIP_SESSION_FAKE:-0}
if [ "$FAKE" = 1 ]; then
    for v in CHIP_X280D_CMD CHIP_VLA_CMD CHIP_TT_PROBE_CMD CHIP_TTSMI_CMD CHIP_FUSER_CMD; do
        [ -n "${!v:-}" ] || die 2 "fake mode needs $v (every device command stubbed)"
    done
    [ "$CHIP_DEV_NODE" != /dev/tenstorrent/0 ] && [ "${CHIP_DEV_NODE#/dev/tenstorrent}" = "$CHIP_DEV_NODE" ] ||
        die 2 "fake mode needs CHIP_DEV_NODE outside /dev/tenstorrent"
fi

ppid_of() { sed 's/.*) //' /proc/"$1"/stat 2>/dev/null | awk '{print $2}'; }  # field 4, after "(comm)"

guard_ancestor() {  # pid of the ancestor running _device_guard.sh whose parent is `flock ... .p150a.lock ...`
    # with-device.sh execs flock, which forks the guard, which runs us as "$@" & (with-device.sh:10,
    # _device_guard.sh:116), so the chain is flock -> bash _device_guard.sh -> this script. Reads /proc only;
    # never opens the lock file.
    local p=$PPID q cmd pcmd arg
    while [ "$p" -gt 1 ] 2>/dev/null; do
        cmd=$(tr '\0' ' ' </proc/"$p"/cmdline 2>/dev/null) || return 1
        for arg in $cmd; do
            if [ "${arg##*/}" = _device_guard.sh ]; then
                q=$(ppid_of "$p")
                pcmd=$(tr '\0' ' ' </proc/"$q"/cmdline 2>/dev/null)
                case " $pcmd " in *flock\ *.p150a.lock\ *) echo "$p"; return 0 ;; esac
            fi
        done
        p=$(ppid_of "$p")
    done
    return 1
}

GUARD_PID=""
if [ "$FAKE" != 1 ] || [ "${CHIP_GUARD_CHECK:-0}" = 1 ]; then
    GUARD_PID=$(guard_ancestor) || die 2 "not inside a device hold: run as \$DEV $0 ... (or \$DEVX when --x280 != none)"
fi
if [ "$X280_MODE" != none ]; then
    [ "${WITH_DEVICE_RESET_AFTER:-0}" = 1 ] || die 2 "--x280 $X280_MODE needs \$DEVX (WITH_DEVICE_RESET_AFTER=1, PLAN.md D13)"
    [ "${WITH_DEVICE_NO_PROBE:-0}" != 1 ] || die 2 "--x280 needs the guard's probe and post-job reset; unset WITH_DEVICE_NO_PROBE"
fi

for v in TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING; do
    if [ -n "${!v+x}" ]; then echo "chip_session: unsetting $v=${!v} (PLAN.md §5.2 invariant 5)" >&2; unset "$v"; fi
done
export DEVICE_LOCK_TIMEOUT=1  # a nested $DEV inside the hold fails fast instead of deadlocking (R29)
chip_pin_safe || die 2 "CHIP_SAFE_CPUS='${CHIP_SAFE_CPUS:-}' is not a CPU list taskset accepts"

# --- session dir ---------------------------------------------------------------------------------------------------

CHIP_SESSION_ID="$(date +%Y%m%d_%H%M%S)_$$"
[ -n "$TAG" ] || TAG=$(basename "${JOBS[0]}" .sh)
CHIP_SESSION_DIR=${CHIP_SESSION_ROOT:-$RT_DIR/runs/sessions}/${CHIP_SESSION_ID}_${TAG}
mkdir -p "$CHIP_SESSION_DIR" || die 2 "cannot create $CHIP_SESSION_DIR"
CHIP_X280_MODE=$X280_MODE CHIP_VLA_MODE=$VLA_MODE CHIP_DEBUG_ENDPOINTS=$DEBUG_EP
CHIP_VLA_URL="" CHIP_X280_URL=""
[ "$VLA_MODE" = tt ] && CHIP_VLA_URL=tcp://127.0.0.1:$CHIP_VLA_PORT
[ "$X280_MODE" != none ] && CHIP_X280_URL=tcp://127.0.0.1:$CHIP_X280_PORT
CHIP_START_VLA=$HERE/start_vla.sh
export CHIP_SESSION_ID CHIP_SESSION_DIR CHIP_X280_MODE CHIP_VLA_MODE CHIP_DEBUG_ENDPOINTS CHIP_VLA_URL CHIP_X280_URL \
    CHIP_START_VLA CHIP_VLA_PORT CHIP_X280_PORT CHIP_DEV_NODE CHIP_VLA_STATUS_PUB CHIP_X280_STATUS_PUB \
    CHIP_SESSION_FAKE=$FAKE

SESSION_CPUS=$(taskset -pc $$ 2>/dev/null | sed 's/.*: //')
chip_log "session $CHIP_SESSION_ID: x280=$X280_MODE vla=$VLA_MODE debug_endpoints=$DEBUG_EP allow_reset=$ALLOW_RESET" \
    "guard=${GUARD_PID:-none} fake=$FAKE cpus=$SESSION_CPUS${CHIP_SAFE_CPUS:+ (CHIP_SAFE_CPUS)} jobs=${JOBS[*]}" \
    "dir=$CHIP_SESSION_DIR"
chip_event session_start "id=$CHIP_SESSION_ID" "x280=$X280_MODE" "vla=$VLA_MODE" "debug_endpoints=$DEBUG_EP" \
    "allow_reset=$ALLOW_RESET" "guard_pid=$GUARD_PID" "fake=$FAKE" "cpus=$SESSION_CPUS" \
    "safe_cpus=${CHIP_SAFE_CPUS:-}" "jobs=${JOBS[*]}" "argv=$0 $*"

T_START=$SECONDS
FINAL_RC=0
TORN_DOWN=0
JOB_PID=""
TEE_PID=""

device_users() {  # pids holding the device node (as the guard counts them, _device_guard.sh:23-25)
    chip_cmd FUSER || return 0
    "${CHIP_CMD[@]}" "$CHIP_DEV_NODE" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | grep -vx "$$" || true
}

has_session_env() { grep -qz "^CHIP_SESSION_ID=$CHIP_SESSION_ID\$" /proc/"$1"/environ 2>/dev/null; }

session_procs() {  # live processes (other than this shell) that inherited this CHIP_SESSION_ID
    local f p
    # One grep over every readable environ (other users' are skipped by -s); run with the variable unexported.
    grep -lsz "^CHIP_SESSION_ID=$CHIP_SESSION_ID\$" /proc/[0-9]*/environ 2>/dev/null | while read -r f; do
        p=${f#/proc/}
        p=${p%/environ}
        [ "$p" = $$ ] && continue
        chip_alive "$p" && echo "$p"
    done
}

teardown() {  # reverse order; idempotent; sets FINAL_RC to 6 if something survived SIGKILL
    [ "$TORN_DOWN" = 1 ] && return 0
    TORN_DOWN=1
    local left users p failed=0
    chip_log "teardown (reverse order)"
    chip_wait_abort  # a READY wait interrupted by a signal leaves its ping helper behind
    if [ -n "$JOB_PID" ] && chip_alive "$JOB_PID"; then chip_kill_tree job "$JOB_PID" || failed=1; fi
    vla_stop || failed=1
    chip_stop_named x280d || failed=1
    if [ -n "$TEE_PID" ]; then kill "$TEE_PID" 2>/dev/null; wait "$TEE_PID" 2>/dev/null; fi
    left=$(export -n CHIP_SESSION_ID; session_procs | tr '\n' ' ')
    if [ -n "${left// /}" ]; then
        chip_log "orphans of this session: $left"
        for p in $left; do chip_alive "$p" && { chip_kill_tree "orphan" "$p" || failed=1; }; done
    fi
    users=$(device_users | tr '\n' ' ')
    for p in $users; do
        if has_session_env "$p"; then
            chip_log "ERROR: session process $p still holds $CHIP_DEV_NODE"
            failed=1
        else
            chip_log "WARNING: $CHIP_DEV_NODE is held by a process outside this session: $(ps -o pid=,cmd= -p "$p" 2>/dev/null)"
        fi
    done
    [ $failed = 1 ] && [ "$FINAL_RC" = 0 ] && FINAL_RC=6
    chip_event teardown_done "failed=$failed"
}

finish() {  # finish RC: teardown, record, exit
    trap '' TERM INT HUP
    [ "$FINAL_RC" = 0 ] && FINAL_RC=$1
    teardown
    chip_log "session end rc=$FINAL_RC after $((SECONDS - T_START))s"
    chip_event session_end "rc=$FINAL_RC" "secs=$((SECONDS - T_START))"
    echo "{\"id\":\"$CHIP_SESSION_ID\",\"dir\":\"$(_chip_jesc "$CHIP_SESSION_DIR")\",\"x280\":\"$X280_MODE\",\"vla\":\"$VLA_MODE\",\"rc\":$FINAL_RC,\"secs\":$((SECONDS - T_START))}" \
        >>"$(dirname "$CHIP_SESSION_DIR")/index.jsonl"
    trap - EXIT
    exit "$FINAL_RC"
}

# shellcheck disable=SC2329  # invoked by the traps below
on_signal() {
    chip_log "signal $1: tearing down"
    chip_event signal "sig=$1"
    FINAL_RC=$((128 + $(kill -l "$1")))
    finish "$FINAL_RC"
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
trap 'chip_log "unexpected exit"; finish 1' EXIT

run_timed() {  # run_timed SECS OUTFILE CMD...: own group, KILL at SECS; RUN_RC (124 = killed at the timeout, 125 = survived KILL)
    local secs=$1 out=$2 pid i
    shift 2
    (cd "$RT_DIR" && exec setsid "$@") >"$out" 2>>"$out.err" </dev/null &
    pid=$!
    for ((i = 0; i < secs * 10; i++)); do
        chip_alive "$pid" || break
        sleep 0.1
    done
    if chip_alive "$pid"; then
        chip_log "$(basename "$out"): still running after ${secs}s: SIGKILL"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
        for ((i = 0; i < 50; i++)); do chip_alive "$pid" || break; sleep 0.1; done
        if chip_alive "$pid"; then RUN_RC=125; return; fi
        wait "$pid" 2>/dev/null
        RUN_RC=124
        return
    fi
    wait "$pid"
    RUN_RC=$?
}

PROBE_N=0
x280_probe() {  # step 1: sets PROBE_VERDICT = clean | dirty | error, PROBE_WHY
    PROBE_N=$((PROBE_N + 1))
    local out=$CHIP_SESSION_DIR/x280_probe.$PROBE_N.json line
    chip_cmd X280D || { PROBE_VERDICT=error PROBE_WHY="no x280d command"; return; }
    chip_log "step 1: x280d probe --json (#$PROBE_N, ${CHIP_X280_PROBE_TIMEOUT_S}s timeout)"
    run_timed "$CHIP_X280_PROBE_TIMEOUT_S" "$out" "${CHIP_CMD[@]}" probe --json
    case $RUN_RC in
        124) PROBE_VERDICT=dirty PROBE_WHY="probe hung for ${CHIP_X280_PROBE_TIMEOUT_S}s (killed)" ;;
        125) PROBE_VERDICT=error PROBE_WHY="probe survived SIGKILL (stuck in the kernel on a tile access)" ;;
        *)
            line=$("$CHIP_PY3" - "$out" "$X280_MODE" "$RUN_RC" <<'EOF'
import json, sys
path, mode, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = open(path, errors="replace").read()
obj = None
for cand in [ln.strip() for ln in reversed(text.splitlines())] + [text]:
    if cand.startswith("{"):
        try:
            obj = json.loads(cand)
            break
        except ValueError:
            pass
if rc not in (0, 3):
    print(f"error probe exited {rc}")
elif not isinstance(obj, dict) or not isinstance(obj.get("dirty"), bool):
    print("dirty probe exited 3 without a report" if rc == 3 else "error no JSON object with a boolean 'dirty'")
else:
    reasons = "; ".join(map(str, obj.get("reasons") or [])) or "-"
    dirty = obj["dirty"] or rc == 3
    if dirty and mode == "attach" and obj.get("attachable") is True:
        print(f"clean attachable ({reasons})")
    else:
        print(("dirty " if dirty else "clean ") + reasons)
EOF
            ) || line="error cannot parse the probe output"
            PROBE_VERDICT=${line%% *} PROBE_WHY=${line#* }
            ;;
    esac
    chip_log "x280 probe #$PROBE_N: $PROBE_VERDICT ($PROBE_WHY), rc=$RUN_RC, report $out"
    chip_event x280_probe "n=$PROBE_N" "verdict=$PROBE_VERDICT" "why=$PROBE_WHY" "rc=$RUN_RC"
}

reset_and_reprobe() {  # step 2: returns 0 clean, 3 still dirty / cannot reset, 4 reset or tt_probe failed
    local users
    users=$(device_users | tr '\n' ' ')
    if [ -n "${users// /}" ]; then
        chip_log "dirty, but $CHIP_DEV_NODE is held by: $(ps -o pid=,cmd= -p "${users// /,}" 2>/dev/null | tr '\n' ';'); not resetting"
        chip_event reset_refused "users=$users"
        return 3
    fi
    chip_cmd TTSMI || return 4
    chip_log "step 2: RESET ${CHIP_CMD[*]} -r 0 (fuser empty; DP3(b))"
    chip_event reset_start
    run_timed "$CHIP_RESET_TIMEOUT_S" "$CHIP_SESSION_DIR/tt_smi_reset.log" "${CHIP_CMD[@]}" -r 0
    chip_event reset_end "rc=$RUN_RC"
    if [ "$RUN_RC" != 0 ]; then chip_log "tt-smi -r 0 failed rc=$RUN_RC"; return 4; fi
    sleep "$CHIP_RESET_WAIT_S"
    chip_cmd TT_PROBE || return 4
    chip_log "step 2: tt_probe.py (${CHIP_TT_PROBE_TIMEOUT_S}s timeout)"
    run_timed "$CHIP_TT_PROBE_TIMEOUT_S" "$CHIP_SESSION_DIR/tt_probe.log" "${CHIP_CMD[@]}"
    chip_event tt_probe "rc=$RUN_RC"
    if [ "$RUN_RC" != 0 ]; then
        chip_log "tt_probe failed after the reset rc=$RUN_RC: $(tail -n 3 "$CHIP_SESSION_DIR/tt_probe.log"* 2>/dev/null | tr '\n' ' ')"
        return 4
    fi
    x280_probe
    case $PROBE_VERDICT in clean) return 0 ;; dirty) return 3 ;; *) return 4 ;; esac
}

start_x280d() {  # step 3: 0 READY, 3 x280d exited 3 (dirty), 5 otherwise
    local args=() pid rc
    case $X280_MODE in
        probe) args=(--no-release) ;;
        boot:3) args=(--stage0) ;;  # X280_E1F_APPROVED=1 checked above; x280d checks it again
        attach) args=(--attach) ;;
    esac
    if [ -n "${CHIP_X280D_SERVE_ARGS:-}" ]; then
        read -r -a args <<<"$CHIP_X280D_SERVE_ARGS"
    else
        args=(serve --port "$CHIP_X280_PORT" --tile 3 "${args[@]}")
    fi
    x280d_start "$CHIP_SESSION_DIR/x280d.log" "${args[@]}" || return 5  # + --cpus 5 / --status-pub (chip_lib.sh)
    pid=$CHIP_PID
    chip_wait_ready x280d "tcp://127.0.0.1:$CHIP_X280_PORT" "$pid" "$CHIP_X280_READY_TIMEOUT_S"
    rc=$?
    [ $rc = 0 ] && return 0
    if [ $rc = 2 ]; then
        wait "$pid"
        rc=$?
        chip_log "x280d exited rc=$rc before READY: $(tail -n 5 "$CHIP_SESSION_DIR/x280d.log" | tr '\n' ' ')"
        [ $rc = 3 ] && return 3
        return 5
    fi
    chip_stop_named x280d
    return 5
}

servers_alive() {  # before each job: a server this session started must still run unless a job stopped it
    local name
    for name in x280d server_g1; do
        if [ -f "$CHIP_SESSION_DIR/$name.pid" ] && ! chip_pidfile_pid "$name" >/dev/null; then
            chip_log "$name died during the session (see $CHIP_SESSION_DIR/$name*.log)"
            chip_event died "name=$name"
            return 1
        fi
    done
    return 0
}

run_job() {  # step 5: one job in its own group, output to the console and job.N.log; sets JOB_RC
    local job=$1 n=$2 i
    local fifo=$CHIP_SESSION_DIR/.job.$n.fifo
    mkfifo "$fifo"
    tee -a "$CHIP_SESSION_DIR/job.$n.log" <"$fifo" &
    TEE_PID=$!
    chip_log "step 5: job $n: $job"
    chip_event job_start "n=$n" "job=$job"
    if [ -x "$job" ]; then
        (cd "$RT_DIR" && exec setsid "$job") >"$fifo" 2>&1 </dev/null &
    else
        (cd "$RT_DIR" && exec setsid bash "$job") >"$fifo" 2>&1 </dev/null &
    fi
    JOB_PID=$!
    wait "$JOB_PID"
    JOB_RC=$?
    JOB_PID=""
    for ((i = 0; i < 50; i++)); do chip_alive "$TEE_PID" || break; sleep 0.1; done  # a daemon may keep the fifo open
    kill "$TEE_PID" 2>/dev/null
    wait "$TEE_PID" 2>/dev/null
    TEE_PID=""
    rm -f "$fifo"
    chip_log "job $n exited rc=$JOB_RC"
    chip_event job_end "n=$n" "rc=$JOB_RC"
}

# --- 1-2: L2CPU health check and the optional reset -----------------------------------------------------------------

if [ "$X280_MODE" != none ]; then
    x280_probe
    case $PROBE_VERDICT in
        clean) ;;
        error) chip_log "x280 probe error: $PROBE_WHY"; finish 4 ;;
        dirty)
            if [ "$ALLOW_RESET" != 1 ]; then
                chip_log "X280 dirty ($PROBE_WHY) and no --allow-reset: exit 3 (DEVX resets after the hold)"
                finish 3
            fi
            reset_and_reprobe
            rc=$?
            [ $rc = 0 ] || { chip_log "X280 not clean after the reset path (rc=$rc)"; finish $rc; }
            ;;
    esac
fi

# --- 3-4: x280d before server_g1 (D13) -------------------------------------------------------------------------------

if [ "$X280_MODE" != none ]; then
    chip_log "step 3: x280d serve ($X280_MODE)"
    start_x280d
    rc=$?
    [ $rc = 0 ] || finish $rc
fi
if [ "$VLA_MODE" = tt ]; then
    chip_log "step 4: server_g1"
    vla_start || finish 5
fi

# --- 5: jobs ---------------------------------------------------------------------------------------------------------

n=0
for job in "${JOBS[@]}"; do
    n=$((n + 1))
    servers_alive || { FINAL_RC=5; break; }
    run_job "$job" $n
    if [ "$JOB_RC" = 3 ]; then
        chip_log "job $n reported X280 dirty (exit 3): stopping"
        FINAL_RC=3
        break
    fi
    [ "$JOB_RC" != 0 ] && [ "$FINAL_RC" = 0 ] && FINAL_RC=$JOB_RC
done

# --- 6: teardown -----------------------------------------------------------------------------------------------------

finish "$FINAL_RC"
