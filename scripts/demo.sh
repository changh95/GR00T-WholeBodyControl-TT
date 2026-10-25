#!/usr/bin/env bash
# demo.sh: the live demo (PLAN.md §0.2 demo ladder M3, WP-1.7, T-3.3, T-3.9).
#
#   demo.sh m3 [--seeds A-B] [--duration S] [--hud] [--record] [--tag T] [--on-top] [--dry-run]
#              [--vla URL] [--wbc URL|ort] [--no-viewer] [--gates NAME] [--allow-reset]
#              [--x280-sim] [--shadow URL] [--in-session] [extra loop args ...] [-- ...]
#
# m3: runs jobs/demo_m3.sh under $DEVX via chip_session.sh (--x280 boot:3 --vla tt; DP3: one L2CPU3 release
#     per session, chip reset after the hold): VLA Tensix (server_g1 :5557), WBC-MLP on the X280 via x280d :5558,
#     the host's ORT as the live swap alternative (key w). The job re-enters this script with --in-session, which runs
#     rt_sim in rt mode with --viewer --pub, the HUD (CPUs 6-7), the recorder and the keep-on-top helper, and tears
#     everything it started down on exit or Ctrl-C. x280d boots with --stage0 (E1F, the standard boot since
#     2026-10-25: host-written LIM is unusable, S1c/S1d), which needs the user's DP3(d) approval as
#     X280_E1F_APPROVED=1 in the caller's environment: without it the device run is refused (exit 2), and so is a
#     CHIP_X280D_SERVE_ARGS without --stage0.
#     --x280-sim runs m3 with no X280 (T-3.3): x280d --backend sim on CPUs 5,21 (its own threads on 5, the sim
#     hart's spin thread on 21, PLAN.md §1.1), started unless it already answers ping; the VLA is still the TT one
#     ($VLA_TT), which must answer ping (this script never starts it; --vla URL picks another). T-3.9's gated 2 h
#     soak is jobs/m3_soak.sh (this script with --in-session inside the hold, then the T-3.9 checks).
# --shadow URL (T-3.6, with --wbc ort): that W-WBC shadows the ORT primary; x280d (sim or device) is started or
#     required for it like a primary. With --hud, x280d and server_g1 publish their status for the HUD chip panel
#     ($X280_STATUS, $TT_STATUS; chip_session's CHIP_VLA_STATUS_PUB / CHIP_X280D_SERVE_ARGS, unless already set).
#
# Unknown options are passed to rtloop.loop verbatim (e.g. --wbc-timeout-ms 200 --wbc-no-latch for T-1.5).
# Test hooks: --pub-url / --control-url (default $TELEMETRY_PUB / $CONTROL_PULL), and the environment
# variables DEMO_LOG_ROOT (default $RT/runs/demo_logs), DEMO_AUX_CPUS (HUD and helpers, default 6,7) and
# DEMO_LOOP_MODULE (default rtloop.loop).
# :5555 (the existing server) is never started, stopped, pinged or signalled by this script; only processes
# this script started are ever killed (by their own process group), never anything found by port.
set -u -o pipefail

RT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../env.sh
source "$RT_DIR/env.sh"
RT=$RT_DIR

usage() { sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

MS=${1:-}
case "$MS" in
  m3) shift ;;
  -h|--help) usage; exit 0 ;;
  *) echo "demo.sh: first argument must be m3" >&2; usage >&2; exit 2 ;;
esac

ORIG_ARGS=("$@")
SEEDS="1000-1019"; DURATION=""; HUD=0; RECORD=0; TAG=""; ON_TOP=0; DRY=0; VIEWER=1
VLA_URL=""; WBC_URL=""; GATES=""; ALLOW_RESET=0; X280_SIM=0; IN_SESSION=0; SWAP_EVERY=""; SHADOW=""
PUB_URL=$TELEMETRY_PUB; CTL_URL=$CONTROL_PULL
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS=$2; shift 2 ;;
    --duration) DURATION=$2; shift 2 ;;
    --hud) HUD=1; shift ;;
    --record) RECORD=1; shift ;;
    --tag) TAG=$2; shift 2 ;;
    --on-top) ON_TOP=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --no-viewer) VIEWER=0; shift ;;
    --vla) VLA_URL=$2; shift 2 ;;
    --wbc) WBC_URL=$2; shift 2 ;;
    --gates) GATES=$2; shift 2 ;;
    --allow-reset) ALLOW_RESET=1; shift ;;
    --x280-sim) X280_SIM=1; shift ;;
    --shadow) SHADOW=$2; shift 2 ;;
    --in-session) IN_SESSION=1; shift ;;
    --swap-wbc-every) SWAP_EVERY=$2; shift 2 ;;
    --pub-url) PUB_URL=$2; shift 2 ;;
    --control-url) CTL_URL=$2; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
TAG=${TAG:-demo_$MS}
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="${DEMO_LOG_ROOT:-$RT/runs/demo_logs}/${STAMP}_${TAG}"
AUX=${DEMO_AUX_CPUS:-6,7}
LOOP_MODULE=${DEMO_LOOP_MODULE:-rtloop.loop}
export DISPLAY=${DISPLAY:-:1}
export XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}

say() { echo "demo.sh[$MS]: $*" >&2; }
show() { printf '%q ' "$@"; }  # a command as a copy-pasteable line
port_of() { local u=${1##*:}; [[ $u =~ ^[0-9]+$ ]] && echo "$u" || echo ""; }
# Does the loop's CLI define option $1? (grep of its source; the loop is a module name or a .py path)
loop_src() { case "$LOOP_MODULE" in *.py) echo "$LOOP_MODULE" ;; *) echo "$RT/${LOOP_MODULE//.//}.py" ;; esac; }
loop_has() { grep -q -- "\"$1\"" "$(loop_src)" 2>/dev/null; }

# --- the device session: hand over to chip_session.sh under DEVX ------------------------------------------
if [ "$IN_SESSION" = 0 ] && [ "$X280_SIM" = 0 ]; then
  read -r -a GUARD <<<"$DEVX"; JOB="$RT/jobs/demo_m3.sh"
  SESSION=("${GUARD[@]}" "$RT/scripts/chip_session.sh" --x280 boot:3 --vla tt)
  [ "$ALLOW_RESET" = 1 ] && SESSION+=(--allow-reset)
  SESSION+=(--job "$JOB")
  # chip_session's own overrides (chip_session.sh header): the status streams for the HUD chip panel, and x280d on
  # CPU 5 (§1.1) with the same serve arguments chip_session would build for --x280 boot:3.
  SESS_ENV=()
  if [ "$HUD" = 1 ]; then SESS_ENV+=("CHIP_VLA_STATUS_PUB=${CHIP_VLA_STATUS_PUB:-$TT_STATUS}"); fi
  # The stage-0 boot (E1F, PLAN.md §5.2 "E1F stage-0 initializer"): every M3 session since 2026-10-25. x280d itself
  # refuses --stage0 without X280_E1F_APPROVED=1; checking here refuses before the hold (no reset spent).
  X280D_ARGS="serve --port $(port_of "$WBC_X280") --tile 3 --stage0 --cpus 5"
  [ "$HUD" = 1 ] && X280D_ARGS+=" --status-pub $X280_STATUS"
  X280D_ARGS=${CHIP_X280D_SERVE_ARGS:-$X280D_ARGS}
  E1F_REFUSED=""
  if [ "${X280_E1F_APPROVED:-0}" != 1 ]; then
    E1F_REFUSED="X280_E1F_APPROVED=1 is not set: the M3 device demo boots L2CPU3 with the stage-0 initializer (E1F),"
    E1F_REFUSED+=" which needs the user's DP3(d) approval in the caller's environment (X280_E1F_APPROVED=1 demo.sh m3 ...)"
  fi
  case " $X280D_ARGS " in
    *" --stage0 "*) ;;
    *) E1F_REFUSED="CHIP_X280D_SERVE_ARGS='$X280D_ARGS' has no --stage0: since E1F (2026-10-25) every M3 session boots"
       E1F_REFUSED+=" with the stage-0 path (host-written LIM is unusable, S1c/S1d); the method-A --boot is not used on silicon" ;;
  esac
  SESS_ENV+=("X280_E1F_APPROVED=${X280_E1F_APPROVED:-0}" "CHIP_X280D_SERVE_ARGS=$X280D_ARGS")
  # The job re-enters this script inside the hold with the same options (chip_session runs it as a child,
  # and the device guard passes the environment through).
  JOB_ARGS=()
  for a in "${ORIG_ARGS[@]}"; do [ "$a" = --dry-run ] || JOB_ARGS+=("$a"); done
  DEMO_ARGS=""
  [ ${#JOB_ARGS[@]} -gt 0 ] && DEMO_ARGS=$(printf '%q ' "${JOB_ARGS[@]}")
  if [ "$DRY" = 1 ]; then
    echo "# $MS: one device hold; chip_session starts x280d (boot:3, stage-0 boot) and server_g1, then the job"
    [ -n "${E1F_REFUSED:-}" ] && echo "# NOTE: REFUSED without --dry-run: $E1F_REFUSED"
    echo "DEMO_ARGS=$(printf '%q' "$DEMO_ARGS") ${SESS_ENV[*]:+$(show "${SESS_ENV[@]}")}$(show "${SESSION[@]}")"
    [ -x "$RT/scripts/chip_session.sh" ] || echo "# NOTE: $RT/scripts/chip_session.sh does not exist yet (WP-2.4 lane)"
    echo "# inside the hold, $(basename "$JOB") runs:"
    "$RT/scripts/demo.sh" "$MS" --in-session "${ORIG_ARGS[@]}" | sed 's/^/#   /'
    exit "${PIPESTATUS[0]}"
  fi
  if [ -n "${E1F_REFUSED:-}" ]; then say "REFUSED: $E1F_REFUSED"; exit 2; fi
  if [ ! -x "$RT/scripts/chip_session.sh" ]; then
    say "$RT/scripts/chip_session.sh is missing (WP-2.4); cannot open a device session"; exit 2
  fi
  say "device session: $(show "${SESSION[@]}")"
  DEMO_ARGS=$DEMO_ARGS exec env "${SESS_ENV[@]}" "${SESSION[@]}"
fi

# --- endpoints and servers of this run ------------------------------------------------------------
# The only server this script starts is the x280d sim (--x280-sim), and only if nothing answers its ping.
START_X280_SIM=0; NEED=()
VLA=${VLA_URL:-$VLA_TT}; WBC=${WBC_URL:-$WBC_X280}
[ "$X280_SIM" = 1 ] && { [ "$WBC" = "$WBC_X280" ] || [ "$SHADOW" = "$WBC_X280" ]; } && START_X280_SIM=1
# Endpoints that must already be served by someone else (chip_session, the user): checked, never started.
SERVED_BY=$([ "$IN_SESSION" = 1 ] && echo "started by chip_session" || echo "never started by demo.sh")
[ "$VLA" = "$VLA_TT" ] && NEED+=("$VLA_TT")
{ [ "$WBC" = "$WBC_X280" ] || [ "$SHADOW" = "$WBC_X280" ]; } && [ "$X280_SIM" = 0 ] && NEED+=("$WBC_X280")
for u in "$VLA" "$WBC"; do
  if [ "$(port_of "$u")" = 5555 ]; then say "note: $u is the existing :5555 server; used as a client only"; fi
done

case "$LOOP_MODULE" in *.py) LOOP=(simpy "$LOOP_MODULE") ;; *) LOOP=(simpy -m "$LOOP_MODULE") ;; esac
LOOP+=(--mode rt --arm v1 --vla "$VLA" --wbc "$WBC" --pub "$PUB_URL")
[ "$VIEWER" = 1 ] && LOOP+=(--viewer)
LOOP+=(--seeds "$SEEDS" --tag "$TAG")
[ -n "$DURATION" ] && LOOP+=(--duration "$DURATION")
[ -n "$SHADOW" ] && LOOP+=(--shadow "$SHADOW")
NOTES=()
if loop_has --control; then LOOP+=(--control "$CTL_URL"); CONTROL_OK=1
else CONTROL_OK=0; NOTES+=("rtloop.loop has no --control yet: HUD keys v/w/r/p are inert (WP-1.7 requested change)"); fi
if [ -n "$SWAP_EVERY" ]; then
  if loop_has --swap-wbc-every; then LOOP+=(--swap-wbc-every "$SWAP_EVERY"); else NOTES+=("no --swap-wbc-every in rtloop.loop: ignored"); fi
fi
LOOP+=("${EXTRA[@]}")

# The sim hart spins on its own CPU (21, the idle sibling of x280d's CPU 5, §1.1); sharing CPU 5 with x280d made it
# fall back to sleep-polling (T-3.3 attribution, integrate:x280d).
X280_CPUS=5,21; X280_HART_CPU=21
X280_SRV=(xpy "$RT/x280/x280d.py" serve --backend sim --port "$(port_of "$WBC_X280")" --sim-cpus "$X280_HART_CPU")
[ "$HUD" = 1 ] && X280_SRV+=(--status-pub "$X280_STATUS")  # the sim's status for the HUD chip panel
HUD_CMD=(simpy "$RT/scripts/hud.py" --sub "$PUB_URL" --control "$CTL_URL" --cpus "$AUX"
         --summary "$LOGDIR/hud_summary.json")
[ "$CONTROL_OK" = 1 ] || HUD_CMD+=(--no-control)
ONTOP_CMD=(simpy "$RT/scripts/keep_viewer_on_top.py")
REC_CMD=("$RT/scripts/record_demo.sh" --out "$LOGDIR/demo.mp4" --display "$DISPLAY")

# --- dry run: print the plan, touch nothing --------------------------------------------------------
if [ "$DRY" = 1 ]; then
  echo "# demo $MS$([ "$IN_SESSION" = 1 ] && echo ' (inside the device hold)')$([ "$X280_SIM" = 1 ] && echo ' (x280 sim, no device)'): logs in $LOGDIR"
  echo "export DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY"
  for u in "${NEED[@]}"; do echo "# require: $u answers ping ($SERVED_BY)"; done
  [ "$START_X280_SIM" = 1 ] && echo "# unless $WBC_X280 answers ping:" && echo "taskset -c $X280_CPUS $(show "${X280_SRV[@]}") > $LOGDIR/x280d_sim.log 2>&1 &"
  echo "# wait until every endpoint answers ping$([ "$START_X280_SIM" = 1 ] && echo ' (the x280d sim up to 60 s)')"
  [ "$HUD" = 1 ] && echo "taskset -c $AUX $(show "${HUD_CMD[@]}") > $LOGDIR/hud.log 2>&1 &"
  [ "$ON_TOP" = 1 ] && echo "taskset -c $AUX $(show "${ONTOP_CMD[@]}") > $LOGDIR/on_top.log 2>&1 &"
  [ "$RECORD" = 1 ] && echo "$(show "${REC_CMD[@]}") > $LOGDIR/record.log 2>&1 &"
  echo "$(show "${LOOP[@]}") 2>&1 | tee $LOGDIR/loop.log"
  [ -n "$GATES" ] && echo "simpy $RT/scripts/analyze_run.py <run dir> --gates $GATES"
  echo "# on exit or Ctrl-C: SIGINT the loop (it finishes the tick and closes its logs), then stop the HUD,"
  echo "# helpers and recorder, then SIGTERM (SIGKILL after 10 s) only the servers started above"
  for n in "${NOTES[@]}"; do echo "# NOTE: $n"; done
  exit 0
fi

# --- live run ----------------------------------------------------------------------------------------
mkdir -p "$LOGDIR"
for n in "${NOTES[@]}"; do say "NOTE: $n"; done
declare -A PIDS=()   # name -> pid (= process group id, via setsid)
ORDER=()

launch() {  # launch NAME CPUS|- LOG cmd... : own session/process group, so teardown kills exactly it
  local name=$1 cpus=$2 log=$3; shift 3
  if [ "$cpus" = - ]; then
    setsid bash -c 'source "$RT/env.sh"; "$@"' _ "$@" >"$log" 2>&1 </dev/null &
  else
    setsid taskset -c "$cpus" bash -c 'source "$RT/env.sh"; "$@"' _ "$@" >"$log" 2>&1 </dev/null &
  fi
  PIDS[$name]=$!; ORDER+=("$name")
  say "started $name (pgid ${PIDS[$name]}): $*"
}

zping() {  # zping URL TIMEOUT_MS -> 0 if the server answers the PolicyServer/W-WBC ping
  [ "$(port_of "$1")" = 5555 ] && return 0  # never message :5555 from here (the loop checks it itself)
  simpy -c '
import sys, zmq
from rtloop import wire
ctx = zmq.Context(); s = ctx.socket(zmq.REQ); t = int(sys.argv[2])
s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, t); s.setsockopt(zmq.SNDTIMEO, t); s.connect(sys.argv[1])
try:
    s.send(wire.request("ping")); r = wire.loads(s.recv()); ok = isinstance(r, dict) and "error" not in r
except zmq.ZMQError:
    ok = False
s.close(); ctx.term(); sys.exit(0 if ok else 1)' "$1" "${2:-1000}" 2>/dev/null
}


wait_ready() {  # wait_ready URL SECONDS [NAME]: until ping answers; abort if NAME (our child) died
  local url=$1 limit=$2 name=${3:-} t0=$SECONDS
  while ! zping "$url" 1000; do
    if [ -n "$name" ] && ! kill -0 "${PIDS[$name]}" 2>/dev/null; then
      say "$name died before answering ping; last log lines:"; tail -n 20 "$LOGDIR/$name.log" >&2; return 1
    fi
    if [ $((SECONDS - t0)) -ge "$limit" ]; then say "$url did not answer ping within ${limit}s"; return 1; fi
    sleep 2
  done
  say "$url ready after $((SECONDS - t0))s"
}

stop_group() {  # stop_group NAME SIGNAL GRACE_S: signal the whole group, wait until every member is gone
  local name=$1 sig=$2 grace=$3 pid=${PIDS[$1]:-}
  [ -n "$pid" ] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -"$sig" -- "-$pid" 2>/dev/null
    local i=0
    while kill -0 -- "-$pid" 2>/dev/null && [ $i -lt $((grace * 10)) ]; do sleep 0.1; i=$((i + 1)); done
    if kill -0 -- "-$pid" 2>/dev/null; then
      say "$name ignored SIG$sig for ${grace}s: SIGKILL"; kill -KILL -- "-$pid" 2>/dev/null
    fi
  fi
  unset "PIDS[$name]"
}

stop_record() {  # ffmpeg 7 with x11grab keeps recording after one SIGINT (measured: > 60 s); a second signal ends it
  local pid=${PIDS[record]:-} i=0
  if [ -n "$pid" ] && kill -0 -- "-$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null
    while kill -0 -- "-$pid" 2>/dev/null && [ $i -lt 30 ]; do sleep 0.1; i=$((i + 1)); done
  fi
  stop_group record TERM 30  # the fragmented mp4 plays up to its last fragment even if this ends in SIGKILL
}

TORN=0
teardown() {
  [ "$TORN" = 1 ] && return; TORN=1
  trap '' INT TERM
  stop_group loop INT 30
  for n in hud on_top; do stop_group "$n" TERM 5; done
  stop_record
  stop_group x280d_sim TERM 10
  local run
  run=$(grep -m1 -o 'run dir .*' "$LOGDIR/loop.log" 2>/dev/null | sed 's/^run dir //')
  if [ -n "$run" ] && [ -d "$run" ]; then
    ln -sfn "$LOGDIR" "$run/demo_logs" 2>/dev/null
    say "run dir $run (demo logs linked as demo_logs/)"
  fi
  say "logs in $LOGDIR"
}
trap teardown EXIT
trap 'say "interrupted"; exit 130' INT TERM

for u in "${NEED[@]}"; do
  if ! zping "$u" 2000; then say "$u does not answer ping ($SERVED_BY)"; exit 2; fi
done
if [ "$START_X280_SIM" = 1 ] && ! zping "$WBC_X280" 1000; then
  [ -f "$RT/x280/x280d.py" ] || { say "$WBC_X280 is not served and $RT/x280/x280d.py does not exist"; exit 2; }
  launch x280d_sim "$X280_CPUS" "$LOGDIR/x280d_sim.log" "${X280_SRV[@]}"
fi
[ -n "${PIDS[x280d_sim]:-}" ] && { wait_ready "$WBC_X280" 60 x280d_sim || exit 1; }
for u in "$VLA" "$WBC" ${SHADOW:+"$SHADOW"}; do
  [ "$u" = ort ] && continue
  zping "$u" 2000 || { say "$u does not answer ping"; exit 2; }
done

[ "$HUD" = 1 ] && launch hud "$AUX" "$LOGDIR/hud.log" "${HUD_CMD[@]}"
[ "$ON_TOP" = 1 ] && launch on_top "$AUX" "$LOGDIR/on_top.log" "${ONTOP_CMD[@]}"
[ "$RECORD" = 1 ] && launch record - "$LOGDIR/record.log" "${REC_CMD[@]}"

# The loop pins itself (tick thread CPU 2, helpers CPU 3); no taskset here.
# tee ignores INT/TERM: a Ctrl-C must not break rt_sim's stdout while it finishes the tick and closes its logs.
setsid bash -c 'source "$RT/env.sh"; "$@"' _ "${LOOP[@]}" > >(trap '' INT TERM; exec tee "$LOGDIR/loop.log") 2>&1 </dev/null &
PIDS[loop]=$!; ORDER+=(loop)
say "rt_sim (pgid ${PIDS[loop]}): $(show "${LOOP[@]}")"
trap 'say "Ctrl-C: stopping rt_sim after this tick"; kill -INT -- "-${PIDS[loop]}" 2>/dev/null' INT TERM
rc=0
while kill -0 "${PIDS[loop]}" 2>/dev/null; do wait "${PIDS[loop]}"; rc=$?; done
say "rt_sim exited rc=$rc"
trap 'exit 130' INT TERM
if [ -n "$GATES" ]; then
  run=$(grep -m1 -o 'run dir .*' "$LOGDIR/loop.log" | sed 's/^run dir //')
  [ -n "$run" ] && simpy "$RT/scripts/analyze_run.py" "$run" --gates "$GATES"
  rc=$?
fi
exit "$rc"
