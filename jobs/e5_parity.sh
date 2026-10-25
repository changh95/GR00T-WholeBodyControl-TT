#!/usr/bin/env bash
# Session S3, T-3.E5a (PLAN.md WP-3.7): on-device parity of every kernel on logged obs.
#
#   X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0" \
#   X280_PARITY_OBS=$RT/runs/x280/20261024_211935_obs516_kernel/obs516.npz \
#       $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla none --job $RT/jobs/e5_parity.sh
#   (S3 runs it after jobs/e4_bench.sh in one --vla tt hold; the session's boot, --stage0 or --boot, does not matter)
#
# Input: X280_PARITY_OBS = an obs516.npz logged by rt_sim --dump-obs516 (tick, policy, obs, and the ORT action of the
# run when it used --wbc ort), default the newest under $RT/runs (jobs/_x280.sh newest_obs516); X280_PARITY_N rows
# (default all). The T-3.1 kernel set above (runs/x280/..._obs516_kernel, VL, ORT WBC, seeds 0-6) has 10,080 rows
# (5,600 Balance, 4,480 Walk) with the ORT action: PLAN's "10k obs".
# Runs, through the serving x280d (x280d parity):
#   - the session kernel (v2 on fp16, both policies, unless the session was started otherwise);
#   - v2s on fp16; v0 and v1 on the fp32 Balance blob, then on the fp32 Walk blob (one fp32 policy per blob, DP4).
# Pass (T-3.E5a): against the x86 build of the same blob v0/v2/v2s bit-identical, v1 <= 1e-5; against the logged ORT
# action <= 1e-2 (fp16) / <= 5e-5 (fp32). Expected from the kernel lane: fp32 <= 1.53e-5, fp16 <= 7.36e-3 (logged obs
# <= 8.5e-4). A difference on silicon points to frm, a flush mode or a hardware deviation (kernel lane notes).
# v0 runs ~15-45 ms per call on uncached LIM (§1.3): ~5k rows per fp32 blob take minutes, and a step that outlives
# X280_TIMEOUT_S (900 s) is an X280-dirty exit, so X280_PARITY_N_V0 (2000) caps v0's rows. v0 is the parity
# reference (bit-identical vs x86); the fp32 ORT bound on all rows is v1's. e5_summary.json gives every run's rows.
# Writes e5_<kernel>_<blob>.json/.npz and e5_summary.json. Exit 0 all pass, 1 a parity failed, 2 no obs, 3 X280 dirty.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# shellcheck source=jobs/_x280.sh
source "$(dirname "${BASH_SOURCE[0]}")/_x280.sh"
need_x280 boot:3 attach
OBS=${X280_PARITY_OBS:-$(newest_obs516)}
[ -n "$OBS" ] && [ -f "$OBS" ] || { job_log "no obs516.npz (set X280_PARITY_OBS)"; job_exit 2; }
N=${X280_PARITY_N:-0}
NV0=${X280_PARITY_N_V0:-2000}
job_log "E5a on $OBS"
rcs=()
run() {  # run TAG N [x280d parity args]
    local tag=$1 n=$2
    shift 2
    x280_step "e5_$tag" parity --obs "$OBS" --n "$n" --out "$CHIP_SESSION_DIR/e5_$tag.npz" "$@"
    rcs+=("$tag=$?")
}
# L36.variants_built (bit v = variant v): the riscv+V image builds 0b1111; skip what the image does not have
vb=$(x280 status | python3 -c "import json,sys; d=json.loads(sys.stdin.read().strip().splitlines()[-1]); print(d['tiles']['3']['l36']['variants_built'])") ||
    x280_dirty "E5a: no status from x280d"
built() { [ $(( vb >> $1 & 1 )) = 1 ]; }
run session "$N"
built 3 && run v2s_fp16 "$N" --kernel v2s
for b in fp32_balance fp32_walk; do
    built 0 && run "v0_$b" "$NV0" --blob "$b" --kernel v0
    if built 1; then run "v1_$b" "$N" --blob "$b" --kernel v1; else job_log "v1 not built in this image (0x$(printf %x "$vb")): skipped"; fi
done
python3 - "$CHIP_SESSION_DIR" "$OBS" "$(x280_boot_flag)" "${rcs[@]}" <<'EOF'
import json, sys
from pathlib import Path
d, obs, boot, rcs = Path(sys.argv[1]), sys.argv[2], sys.argv[3], dict(x.split("=") for x in sys.argv[4:])
out = {}
for tag, rc in rcs.items():
    try:
        r = json.loads((d / f"e5_{tag}.json").read_text())
    except (OSError, ValueError):
        r = {}
    out[tag] = {"rc": int(rc), "pass": r.get("pass"), "variant": r.get("variant"), "dtype": r.get("dtype"),
                "rows": r.get("n"), "rows_served": r.get("n_served"), "per_policy": r.get("per_policy")}
s = {"obs": obs, "boot": boot, "runs": out, "pass": bool(out) and all(v["pass"] for v in out.values())}
(d / "e5_summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
sys.exit(0 if s["pass"] else 1)
EOF
job_exit $?
