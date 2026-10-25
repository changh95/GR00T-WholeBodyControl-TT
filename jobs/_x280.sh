# shellcheck shell=bash
# jobs/_x280.sh: helpers of the M3 device-session jobs (PLAN.md §2 M3, §5.2), sourced after jobs/_common.sh.
#
# Inside a chip_session with --x280 != none the serving x280d holds the only keeper fd and windows (§5.2 invariant 3),
# so every X280 step goes to it: `x280 SUB ARGS` = `xpy x280/x280d.py SUB --via $CHIP_X280_URL ARGS` (the x280d_admin
# endpoint). No job opens /dev/tenstorrent itself, and none calls $DEV/$DEVX (a nested hold deadlocks, R29).
# Results are JSON files in $CHIP_SESSION_DIR. Exit 3 means "X280 state dirty": chip_session tears down, DEVX resets.
# Boot: since E1F (2026-10-25, S1c/S1d: host-written LIM is unusable) every M3 session runs `--x280 boot:3` with
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0 ..."
# (the hart zero-fills LIM, then the monitor; x280d's mode is 'booted' as after --boot). The jobs accept either boot.

X280_TIMEOUT_S=${X280_TIMEOUT_S:-900}

need_x280() {  # need_x280 MODE...: the session runs x280d in one of these --x280 modes
    [ -n "${CHIP_X280_URL:-}" ] || { job_log "needs x280d: run the session with --x280 ${1:-boot:3}"; exit 2; }
    local m
    for m in "$@"; do [ "${CHIP_X280_MODE:-}" = "$m" ] && return 0; done
    job_log "this job needs --x280 $* (the session has --x280 ${CHIP_X280_MODE:-?})"
    exit 2
}

x280() {  # x280 SUB ARGS...: one x280d subcommand on the serving daemon; prints its JSON line
    xpy "$RT/x280/x280d.py" "$1" --via "$CHIP_X280_URL" "${@:2}"
}

x280_step() {  # x280_step NAME SUB ARGS...: run, keep the JSON as $CHIP_SESSION_DIR/NAME.json, log PASS/FAIL; rc of x280d
    local name=$1 out rc
    shift
    out=$CHIP_SESSION_DIR/$name.json
    timeout "$X280_TIMEOUT_S" bash -c 'source "$RT/env.sh"; xpy "$RT/x280/x280d.py" "$1" --via "$CHIP_X280_URL" "${@:2}"' \
        _ "$@" >"$out.raw" 2>"$CHIP_SESSION_DIR/$name.stderr"
    rc=$?
    tail -n 1 "$out.raw" >"$out"
    rm -f "$out.raw"
    if [ $rc = 124 ]; then
        job_log "$name: x280d did not answer within ${X280_TIMEOUT_S}s"
        x280_dirty "$name: x280d unresponsive"
    fi
    job_log "$name: $([ $rc = 0 ] && echo PASS || echo "FAIL rc=$rc") ($out)"
    chip_event x280_step "name=$name" "rc=$rc"
    return $rc
}

jget() {  # jget FILE PYEXPR: evaluate PYEXPR on the JSON object `d` from FILE (stdlib only)
    python3 - "$1" "$2" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
v = eval(sys.argv[2], {"d": d})
print(json.dumps(v) if isinstance(v, (dict, list)) else v)
EOF
}

x280_boot_flag() {  # x280_boot_flag: the session x280d's serve mode (stage0 | boot | attach | no-release | ?) for logs
    # From CHIP_X280D_SERVE_ARGS, else chip_session's default flag for CHIP_X280_MODE (boot:3 -> --stage0). Since E1F
    # (2026-10-25) every M3 session boots with --stage0 (the hart zero-fills LIM, then the monitor; mode 'booted' as
    # after --boot); the jobs treat both alike, and nothing after the boot assumes a host-initialised LIM.
    local a args=()
    if [ -n "${CHIP_X280D_SERVE_ARGS:-}" ]; then
        read -r -a args <<<"$CHIP_X280D_SERVE_ARGS"
        for a in "${args[@]}"; do
            case $a in --stage0 | --boot | --attach | --no-release) echo "${a#--}"; return 0 ;; esac
        done
        echo "?"
        return 0
    fi
    case ${CHIP_X280_MODE:-} in boot:3) echo stage0 ;; attach) echo attach ;; probe) echo no-release ;; *) echo "?" ;; esac
}

x280_rtt_client_args() {  # x280_rtt_client_args: `x280d bench --rtt` client placement, one argument per line
    # X280_RTT_CLIENT_CPUS (default 2,3: rt_sim's tick and helper CPUs, PLAN.md §1.1, idle while no rt_sim runs, which
    # is the case in T-3.E4 and T-3.5) makes the client RTT a loop's; '' leaves the client on the job's mask (it then
    # floats, as in E1F's 2.8 ms client p99). With CHIP_SAFE_CPUS set, a pin outside it is dropped (logged): x280d
    # would refuse it. X280_RTT_SPIN=1 adds --spin (busy-poll the reply: a lower bound without the wake-up).
    local cpus=${X280_RTT_CLIENT_CPUS-2,3}
    if [ -n "$cpus" ] && [ -n "${CHIP_SAFE_CPUS:-}" ] && ! chip_cpus_within "$cpus" "$CHIP_SAFE_CPUS"; then
        job_log "bench --rtt client: CPUs $cpus are outside CHIP_SAFE_CPUS=$CHIP_SAFE_CPUS, so it runs unpinned" \
            "(its client RTT is not a loop's; the PLAN gates are x280d side and unaffected)"
        cpus=""
    fi
    [ -n "$cpus" ] && printf '%s\n' --cpus "$cpus"
    [ "${X280_RTT_SPIN:-0}" = 1 ] && echo --spin
    return 0
}

x280_reattach_args() {  # x280_reattach_args: the re-attach `serve` arguments, one per line; 1 (and nothing) = cannot
    # Built from the session's CHIP_X280D_SERVE_ARGS (verify V2), so the new daemon serves as the first one did: the
    # mode flag (--stage0, --boot or --no-release; --attach as is) becomes --attach, --log-dir becomes x280d_attach,
    # and every other flag x280d serve's first instance was given is kept with its value (--port, --tile, --dtype,
    # --variant, --slow, --early, --deadline-ms, --no-blob, --spin, --cpus, --status-pub [URL], --status-hz,
    # --first-poll-us).
    # --stage0 is a boot like --boot (the E1F boot, standard since 2026-10-25): the hart already runs the monitor, so
    # the new daemon attaches and never releases or re-runs stage 0. Anything else (--backend, --sim-cpus, an unknown
    # flag, no `serve`, no or two mode flags) is refused: guessing would start an x280d unlike the session's.
    # Unset: chip_session's default `serve --port P --tile 3 <mode>`.
    local src=() out=(serve) i n a modes=0
    if [ -n "${CHIP_X280D_SERVE_ARGS:-}" ]; then
        read -r -a src <<<"$CHIP_X280D_SERVE_ARGS"
    else
        src=(serve --port "${CHIP_X280_PORT:-5558}" --tile 3 --attach)
    fi
    if [ "${src[0]:-}" != serve ]; then
        job_log "x280_reattach: CHIP_X280D_SERVE_ARGS='${CHIP_X280D_SERVE_ARGS:-}' is not an x280d serve command"
        return 1
    fi
    n=${#src[@]}
    for ((i = 1; i < n; i++)); do
        a=${src[i]}
        case $a in
            --stage0 | --boot | --no-release | --attach) modes=$((modes + 1)) ;;
            --slow | --early | --no-blob | --spin) out+=("$a") ;;
            --port=* | --tile=* | --dtype=* | --variant=* | --deadline-ms=* | --cpus=* | --status-pub=* | \
                --status-hz=* | --first-poll-us=*) out+=("$a") ;;
            --port | --tile | --dtype | --variant | --deadline-ms | --cpus | --status-hz | --first-poll-us)
                if [ $((i + 1)) -ge "$n" ] || [ "${src[i + 1]#--}" != "${src[i + 1]}" ]; then
                    job_log "x280_reattach: $a without a value in CHIP_X280D_SERVE_ARGS"
                    return 1
                fi
                out+=("$a" "${src[i + 1]}")
                i=$((i + 1)) ;;
            --status-pub)  # optional value (x280d: nargs='?')
                out+=("$a")
                if [ $((i + 1)) -lt "$n" ] && [ "${src[i + 1]#--}" = "${src[i + 1]}" ]; then
                    out+=("${src[i + 1]}")
                    i=$((i + 1))
                fi ;;
            --log-dir) i=$((i + 1)) ;;  # replaced below
            --log-dir=*) ;;
            *) job_log "x280_reattach: cannot carry '$a' from CHIP_X280D_SERVE_ARGS into an --attach"; return 1 ;;
        esac
    done
    if [ "$modes" != 1 ]; then
        job_log "x280_reattach: CHIP_X280D_SERVE_ARGS needs exactly one of --stage0, --boot, --no-release, --attach" \
            "($modes)"
        return 1
    fi
    printf '%s\n' "${out[@]}" --attach --log-dir "$CHIP_SESSION_DIR/x280d_attach"
}

x280_reattach() {  # x280_reattach: a new `x280d serve --attach` (no release) as this job's child; 0 = it answers ping
    # Recorded in x280d.pid, so chip_session's teardown stops it; its log is x280d.attach.log. Its arguments are the
    # session's own (x280_reattach_args: --slow, --dtype, --tile, ... kept), and it is started through chip_lib.sh
    # x280d_start, so it runs where chip_session's first x280d ran (--cpus 5, --status-pub unless the arguments set
    # them), not on the job's whole CPU mask next to rt_sim (verify2 N3). 1 = refused or not started.
    local spec args=()
    spec=$(x280_reattach_args) || return 1
    mapfile -t args <<<"$spec"
    x280d_start "$CHIP_SESSION_DIR/x280d.attach.log" "${args[@]}" || return 1
    chip_wait_ready x280d "$CHIP_X280_URL" "$CHIP_PID" 60
}

x280_dirty() {  # x280_dirty WHY: the tile is in an unknown state; stop the session (DEVX resets after the hold)
    job_log "X280 state dirty: $*"
    chip_event x280_dirty "why=$*"
    exit 3
}

tt_digest() {  # tt_digest OUT: sha256 of server_g1's predict_debug outputs on the golden obs with fixed noise (T-3.E0/E1b)
    simpy - "$CHIP_VLA_URL" "$GOLDEN" "$1" <<'EOF'
import hashlib, json, sys
from rttt import bench_g1 as b
url, golden, out = sys.argv[1:4]
c = b.G1Client(url)
obs = b.load_obs_set([golden])[:2]
res = {}
for name, o in obs:
    rep, _ = c.predict_debug(o, b.fixed_noise(0))
    res[name] = hashlib.sha256(b"".join(v.tobytes() for _, v in sorted(b.outputs(rep).items()))).hexdigest()
c.close()
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
EOF
}

tt_latency() {  # tt_latency OUT N PERIOD_MS: bench_g1 latency through ZMQ (T-2.3 shape); p50/max in OUT
    local obs=()
    [ -f "$GOLDEN/obs_index.json" ] && obs=(--obs "$GOLDEN")
    simpy "$RT/rttt/bench_g1.py" --endpoint "$CHIP_VLA_URL" "${obs[@]}" --n "$2" --period-ms "$3" --warmup 5 \
        --keep-calls --out "$1" >"$1.log" 2>&1
}

newest_obs516() {  # the newest logged obs516.npz of a run (T-3.E5a / T-3.6 inputs)
    # shellcheck disable=SC2012
    ls -t "$RT"/runs/*/obs516.npz "$RT"/runs/*/*/obs516.npz "$CHIP_SESSION_DIR"/*/*/obs516.npz 2>/dev/null | head -n 1
}
