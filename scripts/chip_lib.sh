# shellcheck shell=bash
# chip_lib.sh: helpers shared by chip_session.sh, start_vla.sh and the session jobs (sourced, never executed).
#
# Why a library: the session (chip_session.sh) and a job that restarts server_g1 mid-session (start_vla.sh, used by
# S1c and E3, PLAN.md §2 conventions) must start, find and kill the same processes the same way. Every child is
# started in its own process group (setsid) and recorded in $CHIP_SESSION_DIR/<name>.pid as "<pid> <starttime>",
# so teardown kills whole trees and never a recycled pid. Every child also inherits CHIP_SESSION_ID, which the
# final sweep uses to find anything a job left behind, so the guard's post-hold reset sees no device users
# (bin/_device_guard.sh:122-130 resets only when fuser is empty).
#
# Every device-touching command is a variable (CHIP_*_CMD below), so a fake mode can stub all of them.

CHIP_LIB_RT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# --- configuration (environment overrides; defaults are the real device session) ---------------------------------

: "${CHIP_VLA_PORT:=5557}"   # server_g1 (PLAN.md D15)
: "${CHIP_X280_PORT:=5558}"  # x280d (PLAN.md D15)
: "${CHIP_DEV_NODE:=/dev/tenstorrent/0}"  # the node the guard watches (bin/_device_guard.sh:19)
: "${CHIP_TEARDOWN_GRACE_S:=10}"          # SIGTERM -> SIGKILL (PLAN.md WP-2.4 step 6)
: "${CHIP_VLA_READY_TIMEOUT_S:=3600}"     # a cold G1 build (S1) can take most of an hour; set 7200 if needed
: "${CHIP_X280_READY_TIMEOUT_S:=120}"
: "${CHIP_PY3:=python3}"                  # stdlib-only JSON parsing
: "${CHIP_PING_PY:=${SIM_PYTHON:-$HOME/experiments/gr00t/Isaac-GR00T-n1.6/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python}}"
: "${CHIP_MK_ENV:=$HOME/experiments/gr00t/publish/worktree-mk-env.sh}"  # what env.sh's ttpy sources
: "${CHIP_VLA_STATUS_PUB:=}"              # server_g1 --status-pub URL (PLAN.md §4.3 item 7: :5574); empty = off
# CHIP_SAFE_CPUS (unset = no pin): a CPU list (taskset -c syntax) the session and every job pin themselves to, for
# device sessions on a machine whose real-time cores are busy (chip_pin_safe). vla_start (and server_g1 itself)
# refuses a server_g1 --cpus outside it. server_g1 runs under `taskset -c <its --cpus>` with OPENBLAS_NUM_THREADS,
# OMP_NUM_THREADS and MKL_NUM_THREADS 1 unless the caller set them (so --torch-threads 0, "library default", also
# needs OMP_NUM_THREADS set to the wanted count).

# CHIP_VLA_CMD / CHIP_X280D_CMD / CHIP_TT_PROBE_CMD / CHIP_TTSMI_CMD / CHIP_FUSER_CMD: whitespace-split command
# prefixes; unset = the real commands built by chip_cmd below.

# --- logging -------------------------------------------------------------------------------------------------------

chip_log() {  # one line to stderr and, inside a session, to session.log
    local line
    line="$(date +%F_%T) [${CHIP_LOG_NAME:-chip}] $*"
    echo "$line" >&2
    if [ -n "${CHIP_SESSION_DIR:-}" ] && [ -d "$CHIP_SESSION_DIR" ]; then echo "$line" >>"$CHIP_SESSION_DIR/session.log"; fi
}

_chip_jesc() {
    local s=$1
    s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/\\n}; s=${s//$'\t'/\\t}; s=${s//$'\r'/\\r}
    printf '%s' "$s"
}

chip_pin_safe() {  # CHIP_SAFE_CPUS set: pin this shell (so every child it starts) to that list; 1 = cannot pin
    [ -n "${CHIP_SAFE_CPUS:-}" ] || return 0
    if ! taskset -pc "$CHIP_SAFE_CPUS" $$ >/dev/null 2>&1; then
        chip_log "cannot pin to CHIP_SAFE_CPUS=$CHIP_SAFE_CPUS (taskset -pc failed)"
        return 1
    fi
    chip_log "pinned to CPUs $(taskset -pc $$ 2>/dev/null | sed 's/.*: //') (CHIP_SAFE_CPUS)"
}

_chip_cpu_list() {  # _chip_cpu_list LIST: taskset's "0,1,8-11" -> "0 1 8 9 10 11"; 1 if LIST is not such a list
    local part lo hi i out=() parts=()
    IFS=, read -r -a parts <<<"$1"
    [ ${#parts[@]} -gt 0 ] || return 1
    for part in "${parts[@]}"; do
        [[ $part =~ ^([0-9]+)(-([0-9]+))?$ ]] || return 1
        lo=$((10#${BASH_REMATCH[1]})) hi=$((10#${BASH_REMATCH[3]:-${BASH_REMATCH[1]}}))
        for ((i = lo; i <= hi; i++)); do out+=("$i"); done
    done
    echo "${out[*]}"
}

chip_cpus_within() {  # chip_cpus_within LIST OUTER: 0 if LIST is a CPU list and every CPU of it is in OUTER
    local inner outer c
    inner=$(_chip_cpu_list "$1") && outer=$(_chip_cpu_list "$2") || return 1
    for c in $inner; do [[ " $outer " == *" $c "* ]] || return 1; done
}

chip_event() {  # chip_event NAME key=value ...: one JSON line in events.jsonl (string values)
    [ -n "${CHIP_SESSION_DIR:-}" ] && [ -d "$CHIP_SESSION_DIR" ] || return 0
    local name=$1 kv out
    shift
    out="{\"t\":\"$(date +%FT%T.%N)\",\"src\":\"$(_chip_jesc "${CHIP_LOG_NAME:-chip}")\",\"event\":\"$(_chip_jesc "$name")\""
    for kv in "$@"; do out+=",\"$(_chip_jesc "${kv%%=*}")\":\"$(_chip_jesc "${kv#*=}")\""; done
    echo "$out}" >>"$CHIP_SESSION_DIR/events.jsonl"
}

# --- commands ------------------------------------------------------------------------------------------------------

chip_cmd() {  # chip_cmd NAME -> CHIP_CMD=(...): the override from CHIP_<NAME>_CMD or the real default
    local var="CHIP_${1}_CMD" ttm=$HOME/experiments/gr00t/tt-metal
    if [ -n "${!var:-}" ]; then
        read -r -a CHIP_CMD <<<"${!var}"
        return 0
    fi
    case $1 in
        X280D) CHIP_CMD=(env "PYTHONPATH=$CHIP_LIB_RT" "$CHIP_LIB_RT/.venv-x280/bin/python" "$CHIP_LIB_RT/x280/x280d.py") ;;
        VLA)  # ttpy (env.sh), but exec'd so the recorded pid is python's and SIGTERM reaches it
            # shellcheck disable=SC2016  # expanded by the inner bash
            CHIP_CMD=(bash -c 'source "$1" >/dev/null && cd "$2" && export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$2" && shift 2 && exec python "$@"'
                      chip-vla "$CHIP_MK_ENV" "$CHIP_LIB_RT" "$CHIP_LIB_RT/rttt/server_g1.py") ;;
        TT_PROBE)  # the guard's probe environment (bin/_device_guard.sh:15-16, 33-35)
            CHIP_CMD=(env "TT_METAL_HOME=$ttm" "PYTHONPATH=$ttm:$ttm/ttnn"
                      "TT_METAL_CACHE=${TT_METAL_CACHE:-$HOME/.cache/tt-metal-cache-gr00t}" TT_METAL_LOGGER_LEVEL=FATAL
                      "$ttm/python_env/bin/python" "$CHIP_LIB_RT/scripts/tt_probe.py") ;;
        TTSMI) CHIP_CMD=("$HOME/.tenstorrent-venv/bin/tt-smi") ;;  # bin/_device_guard.sh:17
        FUSER) CHIP_CMD=(fuser) ;;
        *) chip_log "chip_cmd: unknown command $1"; return 1 ;;
    esac
}

# --- processes -----------------------------------------------------------------------------------------------------

chip_alive() {  # running (not a zombie) process
    local st rest
    read -r st 2>/dev/null </proc/"$1"/stat || return 1
    rest=${st##*) }
    case ${rest%% *} in Z | X | x) return 1 ;; esac
    return 0
}

chip_starttime() {  # /proc/PID/stat field 22 (boot-relative start time): identifies a pid across reuse
    local st rest
    read -r st 2>/dev/null </proc/"$1"/stat || return 1
    rest=${st##*) }
    # shellcheck disable=SC2086
    set -- $rest
    echo "${20}"
}

chip_pgid() {  # /proc/PID/stat field 5
    local st rest
    read -r st 2>/dev/null </proc/"$1"/stat || return 1
    rest=${st##*) }
    # shellcheck disable=SC2086
    set -- $rest
    echo "$3"
}

_chip_signal_tree() {  # _chip_signal_tree SIG LEADER PIDS...: the leader's group once, then members of other groups
    local sig=$1 leader=$2 p
    shift 2
    kill "-$sig" -- "-$leader" 2>/dev/null || kill "-$sig" "$leader" 2>/dev/null
    for p in "$@"; do
        [ "$p" = "$leader" ] && continue
        [ "$(chip_pgid "$p" 2>/dev/null)" = "$leader" ] && continue
        kill "-$sig" "$p" 2>/dev/null
    done
}

chip_descendants() {  # all descendants of PID (breadth first)
    local -A kids=()
    local pid ppid queue=("$1") cur found=()
    while read -r pid ppid; do kids[$ppid]+="$pid "; done < <(ps -e -o pid=,ppid= 2>/dev/null)
    while [ ${#queue[@]} -gt 0 ]; do
        cur=${queue[0]}
        queue=("${queue[@]:1}")
        for pid in ${kids[$cur]:-}; do found+=("$pid"); queue+=("$pid"); done
    done
    echo "${found[*]}"
}

_chip_any_alive() {
    local p
    for p in "$@"; do chip_alive "$p" && return 0; done
    return 1
}

chip_kill_tree() {  # chip_kill_tree NAME PID [GRACE_S]: TERM group + descendants, KILL after GRACE_S; 0 = all gone
    local name=$1 pid=$2 grace=${3:-$CHIP_TEARDOWN_GRACE_S} pids i
    if ! chip_alive "$pid"; then
        wait "$pid" 2>/dev/null
        return 0
    fi
    pids="$pid $(chip_descendants "$pid")"
    chip_log "stopping $name (pid $pid, tree: $pids): SIGTERM"
    # shellcheck disable=SC2086
    _chip_signal_tree TERM "$pid" $pids
    for ((i = 0; i < grace * 10; i++)); do
        # shellcheck disable=SC2086
        _chip_any_alive $pids || break
        sleep 0.1
    done
    # shellcheck disable=SC2086
    if _chip_any_alive $pids; then
        pids="$pids $(chip_descendants "$pid")"
        chip_log "$name still alive after ${grace}s: SIGKILL"
        chip_event sigkill "name=$name" "pid=$pid"
        kill -KILL -- "-$pid" 2>/dev/null
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null  # everything this time, whatever its group
        for ((i = 0; i < 50; i++)); do
            # shellcheck disable=SC2086
            _chip_any_alive $pids || break
            sleep 0.1
        done
    fi
    wait "$pid" 2>/dev/null
    # shellcheck disable=SC2086
    if _chip_any_alive $pids; then
        chip_log "ERROR: $name survived SIGKILL (D state on the device?): $(ps -o pid=,stat=,cmd= -p "${pids// /,}" 2>/dev/null | tr '\n' ';')"
        chip_event kill_failed "name=$name" "pid=$pid"
        return 1
    fi
    chip_event stopped "name=$name" "pid=$pid"
    return 0
}

chip_start() {  # chip_start NAME LOGFILE CMD...: own process group, cwd $RT, stdin /dev/null; sets CHIP_PID
    local name=$1 logf=$2
    shift 2
    (cd "$CHIP_LIB_RT" && exec setsid "$@") >>"$logf" 2>&1 </dev/null &
    CHIP_PID=$!
    chip_pidfile_write "$name" "$CHIP_PID"
    chip_event started "name=$name" "pid=$CHIP_PID" "cmd=$*"
    chip_log "started $name (pid $CHIP_PID): $* (log $logf)"
}

chip_pidfile_write() {
    [ -n "${CHIP_SESSION_DIR:-}" ] || return 0
    echo "$2 $(chip_starttime "$2" || echo 0)" >"$CHIP_SESSION_DIR/$1.pid"
}

chip_pidfile_pid() {  # print the pid recorded for NAME if that very process is still alive; else return 1
    local f=${CHIP_SESSION_DIR:-}/$1.pid pid st
    [ -n "${CHIP_SESSION_DIR:-}" ] && [ -f "$f" ] || return 1
    read -r pid st <"$f" || return 1
    chip_alive "$pid" || return 1
    [ "$st" = 0 ] || [ "$(chip_starttime "$pid")" = "$st" ] || return 1
    echo "$pid"
}

chip_stop_named() {  # chip_stop_named NAME: stop the process of NAME.pid (if any) and remove the file; 0 = gone
    local pid rc=0
    if pid=$(chip_pidfile_pid "$1"); then
        chip_kill_tree "$1" "$pid" || rc=1
    fi
    [ $rc -eq 0 ] && rm -f "${CHIP_SESSION_DIR:-/nonexistent}/$1.pid"
    return $rc
}

CHIP_WAIT_PID=""  # the ping helper chip_wait_ready is waiting on; chip_wait_abort stops it

chip_wait_ready() {  # chip_wait_ready NAME URL PID TIMEOUT_S: 0 = answered ping, 1 = timeout, 2 = process died
    local name=$1 url=$2 pid=$3 tmo=$4 rc t0=$SECONDS
    chip_log "waiting for $name to answer ping on $url (pid $pid, up to ${tmo}s)"
    # One REQ and one queued ping: answered as soon as the server's poll loop starts (server_g1 binds before it
    # opens the device and serves only after its boot check, TT/libero/server_n16.py:358-376).
    # The helper runs in the background and is reaped with `wait`: bash defers a trap until a foreground child
    # exits, but a trapped signal interrupts `wait` at once, so chip_session's SIGTERM teardown is not held up for
    # the rest of a cold build (up to CHIP_VLA_READY_TIMEOUT_S); the trap's teardown calls chip_wait_abort.
    "$CHIP_PING_PY" - "$url" "$pid" "$tmo" <<'EOF' &
import sys, time
import msgpack, zmq
url, pid, tmo = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
def alive(p):
    try:
        with open(f"/proc/{p}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state not in ("Z", "X", "x")
ctx = zmq.Context()
s = ctx.socket(zmq.REQ)
s.setsockopt(zmq.LINGER, 0)
s.connect(url)
s.send(msgpack.packb({"endpoint": "ping"}))
deadline = time.monotonic() + tmo
rc = 1
while time.monotonic() < deadline:
    if s.poll(500, zmq.POLLIN):
        rep = msgpack.unpackb(s.recv(), raw=False, strict_map_key=False)
        rc = 3 if isinstance(rep, dict) and "error" in rep else 0
        break
    if pid > 0 and not alive(pid):
        rc = 2
        break
s.close(0)
ctx.term()
sys.exit(rc)
EOF
    CHIP_WAIT_PID=$!
    wait "$CHIP_WAIT_PID"
    rc=$?
    CHIP_WAIT_PID=""
    case $rc in
        0) chip_log "$name READY after $((SECONDS - t0))s"; chip_event ready "name=$name" "secs=$((SECONDS - t0))" ;;
        2) chip_log "$name died before READY"; chip_event died "name=$name" ;;
        3) chip_log "$name answered ping with an error"; chip_event ping_error "name=$name"; rc=1 ;;
        *) chip_log "$name not READY after ${tmo}s"; chip_event ready_timeout "name=$name"; rc=1 ;;
    esac
    return $rc
}

chip_wait_abort() {  # stop a pending chip_wait_ready helper (signal teardown); no-op when none is pending
    [ -n "${CHIP_WAIT_PID:-}" ] || return 0
    kill "$CHIP_WAIT_PID" 2>/dev/null
    wait "$CHIP_WAIT_PID" 2>/dev/null
    CHIP_WAIT_PID=""
}

# --- x280d serve (chip_session.sh step 3, and the jobs that start it again: T-3.8, S2's E3(d)) -----------------------

CHIP_X280_CPUS=5  # PLAN.md §1.1: x280d on CPU 5 (its sibling 21 idle)

x280d_start() {  # x280d_start LOGFILE ARGS...: chip_start x280d `x280d.py ARGS` as every session x280d; 1 = no cmd
    # One place for the placement, so an x280d a job re-attaches runs where the session's first one ran (verify2 N3):
    # `--cpus 5` unless ARGS set --cpus or CHIP_SAFE_CPUS excludes CPU 5 (a session confined to the non-real-time
    # cores keeps x280d on them too), and `--status-pub $CHIP_X280_STATUS_PUB` when that is set (chip_session: off by
    # default; demo.sh --hud passes --status-pub itself) unless ARGS already have one.
    local logf=$1 args=("${@:2}")
    case " ${args[*]} " in
        *" --cpus "* | *" --cpus="*) ;;
        *) if [ -z "${CHIP_SAFE_CPUS:-}" ] || chip_cpus_within "$CHIP_X280_CPUS" "$CHIP_SAFE_CPUS"; then
               args+=(--cpus "$CHIP_X280_CPUS")
           else
               chip_log "x280d stays on the session's CPUs: CHIP_SAFE_CPUS=$CHIP_SAFE_CPUS excludes CPU $CHIP_X280_CPUS"
           fi ;;
    esac
    if [ -n "${CHIP_X280_STATUS_PUB:-}" ]; then
        case " ${args[*]} " in *" --status-pub"*) ;; *) args+=(--status-pub "$CHIP_X280_STATUS_PUB") ;; esac
    fi
    chip_cmd X280D || return 1
    chip_start x280d "$logf" "${CHIP_CMD[@]}" "${args[@]}"
}

# --- server_g1 (start_vla.sh and chip_session.sh step 4) -----------------------------------------------------------

vla_cpus() {  # vla_cpus ARGS...: the CPU list server_g1 pins itself to (its --cpus; the last one wins, as argparse)
    local cpus=8,9,10,11 i  # rttt/server_g1.py DEFAULT_CPUS, PLAN.md §1.1
    for ((i = 1; i <= $#; i++)); do
        case ${!i} in
            --cpus) i=$((i + 1)); cpus=${!i:-} ;;
            --cpus=*) cpus=${!i#--cpus=} ;;
        esac
    done
    echo "$cpus"
}

vla_start() {  # 0 = READY; 5 = failed to start or not READY (the process is stopped again)
    local pid logf n cpus args=() pin=()
    if pid=$(chip_pidfile_pid server_g1); then
        chip_log "server_g1 already running (pid $pid)"
        return 0
    fi
    if ss -Hltn "sport = :$CHIP_VLA_PORT" 2>/dev/null | grep -q .; then
        chip_log "port $CHIP_VLA_PORT is already in use by another process; not starting server_g1"
        return 5
    fi
    n=$(($(cat "$CHIP_SESSION_DIR/server_g1.starts" 2>/dev/null || echo 0) + 1))
    echo "$n" >"$CHIP_SESSION_DIR/server_g1.starts"
    logf=$CHIP_SESSION_DIR/server_g1.$n.log
    args=(--port "$CHIP_VLA_PORT" --log "$CHIP_SESSION_DIR/server_g1_calls.$n.jsonl")  # rttt/server_g1.py CLI
    [ "${CHIP_DEBUG_ENDPOINTS:-0}" = 1 ] && args+=(--debug-endpoints)
    [ -n "$CHIP_VLA_STATUS_PUB" ] && args+=(--status-pub "$CHIP_VLA_STATUS_PUB")
    # shellcheck disable=SC2206
    [ -n "${CHIP_VLA_EXTRA_ARGS:-}" ] && args+=($CHIP_VLA_EXTRA_ARGS)
    chip_cmd VLA || return 5
    # Library thread pools are created when python imports numpy / torch, long before server_g1's main() pins its
    # --cpus (S1: OpenBLAS's 11 pool threads carried the session's 12-CPU mask until enforce_affinity re-pinned them
    # after the build). So server_g1 starts inside its --cpus already (taskset; '' = inherit) with one-thread pools
    # (OPENBLAS / OMP / MKL_NUM_THREADS=1 unless the caller set them), and affinity_check.repinned stays [].
    cpus=$(vla_cpus "${args[@]}")
    if [ -n "$cpus" ]; then
        if ! _chip_cpu_list "$cpus" >/dev/null; then
            chip_log "server_g1 --cpus '$cpus' is not a CPU list; not starting it"
            return 5
        fi
        if [ -n "${CHIP_SAFE_CPUS:-}" ] && ! chip_cpus_within "$cpus" "$CHIP_SAFE_CPUS"; then
            chip_log "server_g1 --cpus $cpus is outside CHIP_SAFE_CPUS=$CHIP_SAFE_CPUS; not starting it"
            return 5
        fi
        pin=(taskset -c "$cpus")
    fi
    chip_start server_g1 "$logf" env OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" "${pin[@]}" \
        "${CHIP_CMD[@]}" "${args[@]}"
    pid=$CHIP_PID
    if ! chip_wait_ready server_g1 "tcp://127.0.0.1:$CHIP_VLA_PORT" "$pid" "$CHIP_VLA_READY_TIMEOUT_S"; then
        tail -n 20 "$logf" 2>/dev/null | sed 's/^/    | /' >&2
        chip_stop_named server_g1
        return 5
    fi
    return 0
}

vla_stop() {  # 0 = stopped (or was not running); 6 = survived SIGKILL
    chip_stop_named server_g1 && return 0
    return 6
}
