# shellcheck shell=bash
# jobs/_common.sh: sourced by the session jobs jobs/t2_*.sh (PLAN.md WP-2.4, S1 / S1b).
#
# A job runs as a child of chip_session.sh inside the device hold. It reads the session's servers from
# CHIP_VLA_URL / CHIP_X280_URL (env.sh re-exports VLA_TT etc. on every source, so those names are not trusted) and
# writes every artefact into CHIP_SESSION_DIR. Exit 3 is reserved: it tells chip_session "X280 state dirty"
# (PLAN.md §2 conventions), and eval_paired.py (incomplete) and pytest (internal error) also exit 3, so job_exit
# maps a tool's 3 to 4.
#
# CPU placement: with CHIP_SAFE_CPUS set (a device session on a machine whose real-time cores are busy), the job
# pins itself, and so every tool it runs, to that list before anything else (chip_pin_safe; exit 2 if it cannot).
# GPU servers a job starts (T2_START_GPU=1) get PLAN.md §1.1's pins, wbc_gpu_server CPU 4 and gpu_vla_server CPUs
# 12-15, off server_g1's 8-11, and the WBC server busy-polls (--spin; T2_WBC_SPIN=0 for blocking: T-1.4 misses its
# WBC RTT gates without it); those are real-time cores, so starting them under CHIP_SAFE_CPUS is refused, and the
# closed-loop jobs (need_rt_cores: S1b's t2_sync/t2_soak/t2_soak_2h/t2_primary, M3's m3_eval, m3_faults, m3_soak and
# m3_shadow in rt mode) refuse to run at all under it: rt_sim --mode rt pins itself to CPUs 2/3 (rtloop/affinity.py), which the
# job's taskset mask does not prevent.
#
# GPU endpoints: GPU_VLA_URL (T2_GPU_VLA, default $VLA_GPU) and GPU_WBC_URL (T2_WBC, default $WBC_GPU; 'ort' = the
# loop's in-process WBC, no server). A TT-vs-GPU comparison (T-2.5, T-2.7: need_gpu_servers pair) runs with --crn,
# which pairs the arms only if the GPU VLA draws server_g1's noise for a noise_seed: gpu_vla_server --noise cpu,
# which start_gpu_servers always passes and check_gpu_vla_revision reads back from info.noise. T2_DRY_RUN=1: the S1b
# jobs print the commands of their sim runs and run none (job_dry).

JOB_RT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JOB_NAME=$(basename "${BASH_SOURCE[1]:-job}" .sh)

if [ -z "${CHIP_SESSION_DIR:-}" ] || [ ! -d "${CHIP_SESSION_DIR:-/nonexistent}" ]; then
    echo "$JOB_NAME: run me as \$DEV $JOB_RT/scripts/chip_session.sh ... --job $0 (CHIP_SESSION_DIR is unset)" >&2
    exit 2
fi
# shellcheck source=env.sh
source "$JOB_RT/env.sh"
export CHIP_LOG_NAME=$JOB_NAME
# shellcheck source=scripts/chip_lib.sh
source "$JOB_RT/scripts/chip_lib.sh"
chip_pin_safe || exit 2

# shellcheck disable=SC2034  # used by the jobs that source this file
GOLDEN=${T2_GOLDEN:-$HOME/experiments/gr00t/golden/n16_g1}
GPU_WBC_CPUS=4 GPU_VLA_CPUS=12,13,14,15  # PLAN.md §1.1 process table (R21), as scripts/demo.sh starts them
GPU_VLA_URL=${T2_GPU_VLA:-$VLA_GPU}
GPU_WBC_URL=${T2_WBC:-$WBC_GPU}
PAIRED_NOISE=cpu  # server_g1's noise: torch.Generator("cpu").manual_seed(noise_seed), fp32 (rttt/m2_checks.py)

job_log() { chip_log "$*"; }

job_exit() {  # job_exit RC: never let a tool's exit 3 read as "X280 dirty"
    local rc=$1
    [ "$rc" = 3 ] && rc=4
    chip_event job_verdict "job=$JOB_NAME" "rc=$rc"
    exit "$rc"
}

job_dry() { [ "${T2_DRY_RUN:-0}" = 1 ]; }  # T2_DRY_RUN=1: print the sim commands, run none

need_vla() {
    [ -n "${CHIP_VLA_URL:-}" ] || { job_log "needs a TT VLA: run the session with --vla tt"; exit 2; }
}

need_debug_endpoints() {
    [ "${CHIP_DEBUG_ENDPOINTS:-0}" = 1 ] || { job_log "needs server_g1 --debug-endpoints: run the session with --debug-endpoints"; exit 2; }
}

need_rt_cores() {  # need_rt_cores [WHAT]: WHAT (default rt_sim, CPUs 2/3, and the GPU servers, 4 and 12-15) is the
    # real-time workload of PLAN.md §1.1, so the job exits 2 under CHIP_SAFE_CPUS before starting any of it
    [ -z "${CHIP_SAFE_CPUS:-}" ] && return 0
    job_log "CHIP_SAFE_CPUS=$CHIP_SAFE_CPUS is set, but this job runs ${1:-rt_sim and the GPU servers} on the" \
        "real-time cores (PLAN.md §1.1): run the session with CHIP_SAFE_CPUS unset, on a machine with no other" \
        "real-time run"
    exit 2
}

port_listening() { ss -Hltn "sport = :$1" 2>/dev/null | grep -q .; }

ping_url() {  # ping_url URL [TIMEOUT_S]: 0 if the W-VLA / W-WBC server answers ping
    chip_wait_ready "$1" "$1" 0 "${2:-5}" 2>/dev/null
}

gpu_vla_snapshot() {  # the local DP1 snapshot dir (rttt/convert_bf16.DEFAULT_SRC = .../snapshots/<SRC_SHA>)
    simpy -c 'from rttt.convert_bf16 import DEFAULT_SRC; print(DEFAULT_SRC)'
}

start_gpu_servers() {  # start_gpu_servers wbc|vla|pair: start the GPU WBC / VLA that are not up, as this job's children
    local snap
    if [ -n "${CHIP_SAFE_CPUS:-}" ]; then
        job_log "T2_START_GPU=1 with CHIP_SAFE_CPUS set: the GPU servers' §1.1 pins (CPU $GPU_WBC_CPUS, CPUs" \
            "$GPU_VLA_CPUS) are real-time cores; start them outside the session"
        return 1
    fi
    if [ "$GPU_WBC_URL" != ort ] && ! ping_url "$GPU_WBC_URL" 2; then
        local spin=()  # busy-poll by default (T2_WBC_SPIN=0: blocking); T-1.4 misses the WBC RTT gates without it
        [ "${T2_WBC_SPIN:-1}" = 1 ] && spin=(--spin)
        chip_start wbc_gpu "$CHIP_SESSION_DIR/wbc_gpu_server.log" env -C "$R" PYTHONPATH="$RT:$R" \
            "$SRV_PYTHON" -m rtserve.wbc_gpu_server --port "${GPU_WBC_URL##*:}" --onnx-dir "$ONNX" --cpus "$GPU_WBC_CPUS" \
            "${spin[@]}"
        chip_wait_ready wbc_gpu "$GPU_WBC_URL" "$CHIP_PID" 300 || return 1
    fi
    if [ "${1:-vla}" != wbc ] && ! ping_url "$GPU_VLA_URL" 2; then
        # The local snapshot of DP1@SRC_SHA, offline: the hub name would follow the repo's main ref, and the TT arm's
        # BF16 copy was made from SRC_SHA (gpu_vla_server.checkpoint_id stamps a snapshot dir with the same repo id).
        # --noise cpu: a noise_seed draws server_g1's CPU noise, so TT and GPU arms are --crn paired (D17).
        snap=$(gpu_vla_snapshot) && [ -d "$snap" ] || { job_log "no local DP1 snapshot (${snap:-?})"; return 1; }
        chip_start gpu_vla "$CHIP_SESSION_DIR/gpu_vla_server.log" env -C "$R" PYTHONPATH="$RT:$R" HF_HUB_OFFLINE=1 \
            "$SRV_PYTHON" -m rtserve.gpu_vla_server --port "${GPU_VLA_URL##*:}" --model "$snap" \
            --noise "$PAIRED_NOISE" --cpus "$GPU_VLA_CPUS" --log "$CHIP_SESSION_DIR/gpu_vla_server_calls.jsonl"
        chip_wait_ready gpu_vla "$GPU_VLA_URL" "$CHIP_PID" 900 || return 1
    fi
}

stop_gpu_servers() { chip_stop_named gpu_vla; chip_stop_named wbc_gpu; }

check_gpu_vla_revision() {  # check_gpu_vla_revision [NOISE]: 0 ok, 1 not DP1@SRC_SHA, 4 its noise is not NOISE
    # The GPU arm must serve DP1 at SRC_SHA, the revision the TT BF16 copy was made from. With NOISE (pair) it must
    # also draw that noise: info.noise (a server older than --noise reports none), and one seeded get_action whose
    # reply says noise NOISE and randn_calls 1, i.e. the action head's draw was the seeded one and nothing drew from
    # that generator before it (the on-GPU check the --noise cpu change left open). All in gpu_vla_info.json.
    simpy - "$GPU_VLA_URL" "$CHIP_SESSION_DIR/gpu_vla_info.json" "${1:-}" <<'EOF'
import json, sys, time
from rttt.bench_g1 import G1Client, ServerError
from rttt.convert_bf16 import SRC_REPO, SRC_SHA
from rtserve.gpu_vla_server import warmup_obs
url, out, want = sys.argv[1], sys.argv[2], sys.argv[3]
c = G1Client(url, 60000)
info = c.info()
model, rev = str(info.get("model") or ""), str(info.get("revision") or "")
ok = model == SRC_REPO and rev == SRC_SHA
print(f"GPU VLA {url} serves {model or '?'}@{rev or '?'}: {'ok' if ok else f'NOT {SRC_REPO}@{SRC_SHA}'}",
      file=sys.stderr)
rc = 0 if ok else 1
if ok and want:
    noise, probe = info.get("noise"), None
    if noise == want:
        rt = {"seq": 0, "gen": 0, "ep": -1, "k_obs": 0, "t_obs_ns": time.monotonic_ns(), "noise_seed": 20261025}
        try:
            probe = {k: v for k, v in c.get_action(warmup_obs(), rt)[1].items() if k in ("noise", "randn_calls")}
        except (TimeoutError, ServerError) as e:
            probe = {"error": str(e)}
        info["noise_probe"] = probe
    paired = noise == want and probe == {"noise": want, "randn_calls": 1}
    print(f"GPU VLA {url} noise {noise or 'not reported'} (seeded call: {probe}): " + (
        "ok, the same draw as server_g1 for a noise_seed" if paired else
        f"NOT '{want}' in one draw: its noise_seed draws are not server_g1's, so the TT and GPU arms would not "
        "be --crn paired"), file=sys.stderr)
    rc = 0 if paired else 4
c.close()
open(out, "w").write(json.dumps(info, default=str, indent=1) + "\n")
sys.exit(rc)
EOF
}

need_gpu_servers() {  # need_gpu_servers wbc|vla|pair: the GPU WBC (and VLA) must answer, or be started (T2_START_GPU=1)
    # vla: the GPU VLA must serve DP1@SRC_SHA; pair (a TT-vs-GPU comparison) also needs its noise to be server_g1's.
    local what=${1:-vla} rc
    if [ "${T2_START_GPU:-0}" = 1 ]; then
        trap stop_gpu_servers EXIT
        start_gpu_servers "$what" || { job_log "could not start the GPU servers (logs in $CHIP_SESSION_DIR)"; exit 2; }
    fi
    if [ "$GPU_WBC_URL" != ort ]; then
        ping_url "$GPU_WBC_URL" 5 || { job_log "GPU WBC $GPU_WBC_URL does not answer (start it, or T2_START_GPU=1)"; exit 2; }
    fi
    [ "$what" = wbc ] && return 0
    ping_url "$GPU_VLA_URL" 5 || { job_log "GPU VLA $GPU_VLA_URL does not answer (start it, or T2_START_GPU=1)"; exit 2; }
    check_gpu_vla_revision "$([ "$what" = pair ] && echo "$PAIRED_NOISE")"
    rc=$?
    case $rc in
        0) ;;
        4) job_log "GPU VLA $GPU_VLA_URL does not draw server_g1's noise (info in $CHIP_SESSION_DIR/gpu_vla_info.json):" \
               "a TT-vs-GPU comparison with --crn would not be noise-paired. Restart it with" \
               "'srvpy -m rtserve.gpu_vla_server --noise $PAIRED_NOISE ...', or stop it and set T2_START_GPU=1"
           exit 2 ;;
        *) job_log "GPU VLA $GPU_VLA_URL is not the DP1 revision the TT copy came from" \
               "(info in $CHIP_SESSION_DIR/gpu_vla_info.json); TT-vs-GPU arms would compare different checkpoints"
           exit 2 ;;
    esac
}

session_matrix() {  # session_matrix SRC DST: the matrix on this session's TT / GPU VLA / WBC endpoints
    simpy "$RT/rttt/m2_checks.py" matrix "$1" "$2" "$CHIP_VLA_URL" "$GPU_VLA_URL" "$GPU_WBC_URL" || job_exit 2
    job_log "matrix $1 on TT $CHIP_VLA_URL, GPU VLA $GPU_VLA_URL, WBC $GPU_WBC_URL: $2"
}

save_server_info() {  # server_g1's INFO -> $CHIP_SESSION_DIR/server_g1_info.json, with a one-line stamp summary
    simpy - "$CHIP_VLA_URL" "$CHIP_SESSION_DIR/server_g1_info.json" <<'EOF'
import json, sys
from rttt.bench_g1 import G1Client
c = G1Client(sys.argv[1], 10000)
info = c.info()
c.close()
open(sys.argv[2], "w").write(json.dumps(info, default=str, indent=1) + "\n")
st, boot = info.get("stamp") or {}, info.get("boot_check") or {}
print(f"server_g1: backend {info.get('backend')} tree {info.get('tree')} dit {st.get('dit_backend')} arena "
      f"{st.get('mk_arena_dtype')} megakernel {st.get('megakernel_built')} weights {st.get('weights_skipped')} "
      f"boot pcc {boot.get('pcc')} affinity {info.get('affinity')}", file=sys.stderr)
EOF
}
