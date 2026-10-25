#!/usr/bin/env bash
# T-2.2 TT G1 fidelity (PLAN.md §2 M2, session S1): server_g1's predict_debug vs the WP-2.2 goldens.
#
#   $DEV $RT/scripts/chip_session.sh --vla tt --debug-endpoints --job $RT/jobs/t2_fidelity.sh
#   (S1 runs T-2.2..T-2.4 in one hold: --job t2_fidelity.sh --job t2_bench.sh --job t2_abab.sh)
#
# server_g1 answering ping means its boot sanity check (PCC >= 0.995 vs the bf16 golden) already passed.
# First the golden is re-validated against its own manifest (rttt/check_golden.py: file hashes, selection, floors;
# T2_CHECK_GOLDEN=0 skips it, for a test golden without a full manifest) and the manifest's sha256 is recorded in
# golden.sha256, so S1 and later sessions provably used the same files. Then server_g1's INFO is saved as
# server_g1_info.json. Writes $CHIP_SESSION_DIR/t22_fidelity.json (every per-sample number, with TT vs fp32 per group
# next to the gated TT vs bf16), t22_tt_outputs.npz (every TT output, '<sid>/<key>', so a group miss can be judged
# offline without a second hold) and t22_pytest.txt.
# T2_GOLDEN overrides the golden dir (default ~/experiments/gr00t/golden/n16_g1). Exit 0 pass, 1 a gate failed,
# 2 misconfigured (no golden, or one that fails its own manifest).

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_vla
need_debug_endpoints
[ -f "$GOLDEN/obs_index.json" ] || { job_log "no golden at $GOLDEN (WP-2.2 make_g1_golden.py)"; exit 2; }
if [ "${T2_CHECK_GOLDEN:-1}" = 1 ]; then
    simpy -m rttt.check_golden --golden "$GOLDEN" >"$CHIP_SESSION_DIR/check_golden.txt" 2>&1 ||
        { job_log "the golden at $GOLDEN fails its manifest (see check_golden.txt)"; exit 2; }
    job_log "golden $GOLDEN: $(grep -c '^PASS' "$CHIP_SESSION_DIR/check_golden.txt") checks passed"
fi
[ -f "$GOLDEN/manifest.json" ] && (cd "$GOLDEN" && sha256sum manifest.json obs_index.json) >"$CHIP_SESSION_DIR/golden.sha256"
save_server_info

export G1_ENDPOINT=$CHIP_VLA_URL G1_GOLDEN=$GOLDEN G1_FIDELITY_REPORT=$CHIP_SESSION_DIR/t22_fidelity.json
export G1_FIDELITY_OUTPUTS=$CHIP_SESSION_DIR/t22_tt_outputs.npz
job_log "T-2.2 against $G1_ENDPOINT, golden $G1_GOLDEN"
simpy -m pytest "$RT/tests/tt/test_g1_fidelity.py" -q -rfs -p no:cacheprovider 2>&1 | tee "$CHIP_SESSION_DIR/t22_pytest.txt"
rc=${PIPESTATUS[0]}
if [ -f "$G1_FIDELITY_REPORT" ]; then
    simpy - "$G1_FIDELITY_REPORT" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))["verdicts"]
p = v["primary"]
print(f"T-2.2 primary PCC mean {p['mean']:.6f} min {p['min']:.6f} ({'PASS' if p['pass'] else 'FAIL'}); "
      f"secondary {'PASS' if v['secondary']['pass'] else 'FAIL'}; groups {'PASS' if v['groups']['pass'] else 'FAIL'} "
      f"{v['groups']['failures']}; replays {'PASS' if v['replays']['pass'] else 'FAIL'}")
for n, g in v["groups"]["groups"].items():  # TT vs fp32 next to the gated TT vs bf16, to judge a group miss
    print(f"  {n:<20} TT-bf16 max|d| {g['tt_bf16_max']:.5f} ({g['argmax']}) floor {g['floor']:.5f}"
          f"{'' if g['pass'] else ' FAIL'}; TT-fp32 {g['tt_fp32_max']:.5f} ({g['argmax_fp32']}),"
          f" GPU-bf16-fp32 {g['bf16_fp32_max']:.5f}")
EOF
fi
[ "$rc" = 5 ] && { job_log "pytest collected no tests"; rc=2; }  # pytest 5 = nothing collected
job_exit "$rc"
