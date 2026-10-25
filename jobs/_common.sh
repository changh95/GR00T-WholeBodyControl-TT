# shellcheck shell=bash
# jobs/_common.sh: sourced by the session jobs in jobs/ (PLAN.md WP-2.4: the S1 jobs t2_*.sh, the M3 jobs).
#
# A job runs as a child of chip_session.sh inside the device hold. It reads the session's servers from
# CHIP_VLA_URL / CHIP_X280_URL (env.sh re-exports VLA_TT etc. on every source, so those names are not trusted) and
# writes every artefact into CHIP_SESSION_DIR. Exit 3 is reserved: it tells chip_session "X280 state dirty"
# (PLAN.md §2 conventions), and pytest (internal error) also exits 3, so job_exit maps a tool's 3 to 4.
#
# CPU placement: with CHIP_SAFE_CPUS set (a device session on a machine whose real-time cores are busy), the job
# pins itself, and so every tool it runs, to that list before anything else (chip_pin_safe; exit 2 if it cannot).
# The closed-loop jobs (need_rt_cores: M3's m3_eval, m3_faults, m3_soak and m3_shadow in rt mode) refuse to run at
# all under it: rt_sim --mode rt pins itself to CPUs 2/3 (rtloop/affinity.py), which the job's taskset mask does not
# prevent. T2_DRY_RUN=1: a job that supports it prints the commands of its sim runs and runs none (job_dry).

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

need_rt_cores() {  # need_rt_cores [WHAT]: WHAT (default rt_sim, CPUs 2/3) is the real-time workload of PLAN.md
    # §1.1, so the job exits 2 under CHIP_SAFE_CPUS before starting any of it
    [ -z "${CHIP_SAFE_CPUS:-}" ] && return 0
    job_log "CHIP_SAFE_CPUS=$CHIP_SAFE_CPUS is set, but this job runs ${1:-rt_sim} on the" \
        "real-time cores (PLAN.md §1.1): run the session with CHIP_SAFE_CPUS unset, on a machine with no other" \
        "real-time run"
    exit 2
}

ping_url() {  # ping_url URL [TIMEOUT_S]: 0 if the W-VLA / W-WBC server answers ping
    chip_wait_ready "$1" "$1" 0 "${2:-5}" 2>/dev/null
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
