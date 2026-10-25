#!/usr/bin/env bash
# S1 step 1 (PLAN.md WP-2.5): the cold G1 build + boot sanity check, recorded.
#
#   GR00T_TT_FORCE_CONVERT=1 CHIP_VLA_READY_TIMEOUT_S=10800 $DEV $RT/scripts/chip_session.sh --vla tt \
#       --debug-endpoints --job $RT/jobs/s1_boot.sh --tag s1_boot
#
# chip_session runs a job only once server_g1 answers ping, and server_g1 answers only after its build and its boot
# sanity check (PCC >= 0.995 vs the bf16 golden), so this job just records what the build did, for the S1 report:
# server_g1_info.json (boot-check PCC, load / capture / warm times, stamp, affinity) and s1_boot.json with the READY
# time from events.jsonl, the expected-stamp checks and the caches the build wrote (the G1 weights cache root and
# TT_METAL_CACHE, as file count / bytes / newest mtime). Exit 0 if the stamp is the expected one and the boot check
# passed, 1 otherwise, 2 if server_g1 cannot be queried.

set -uo pipefail
# shellcheck source=jobs/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_vla
TTMC=$(bash -c 'source "$1" >/dev/null 2>&1; echo "$TT_METAL_CACHE"' _ "$CHIP_MK_ENV")  # the cache server_g1 used
simpy - "$CHIP_VLA_URL" "$CHIP_SESSION_DIR" "$TTMC" <<'EOF'
import json, os, sys, time
from pathlib import Path
from rttt.bench_g1 import G1Client

url, sdir, ttmc = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
try:
    c = G1Client(url, 10000)
    info = c.info()
    c.close()
except Exception as e:  # noqa: BLE001 - unreachable server = misconfigured session
    print(f"s1_boot: cannot query {url}: {e}", file=sys.stderr)
    sys.exit(2)
(sdir / "server_g1_info.json").write_text(json.dumps(info, default=str, indent=1) + "\n")


def tree(p) -> dict:
    """File count, bytes and newest mtime under p (the caches the cold build wrote); None = not reported."""
    if not p:
        return {"path": None, "exists": False}
    p, n, size = Path(p), 0, 0
    newest = 0.0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                st = os.stat(os.path.join(root, f))
            except OSError:
                continue
            n, size, newest = n + 1, size + st.st_size, max(newest, st.st_mtime)
    return {"path": str(p), "exists": p.exists(), "files": n, "bytes": size,
            "newest": time.strftime("%F %T", time.localtime(newest)) if n else None}


ready = [json.loads(ln) for ln in (sdir / "events.jsonl").read_text().splitlines() if '"event":"ready"' in ln]
st, boot = info.get("stamp") or {}, info.get("boot_check") or {}
ws = st.get("weights_skipped") or {}
aff = info.get("affinity_check") or {}
want = {
    "tree": ("MK@009de892c895", info.get("tree")),
    "backend": ("tt:p150a:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate-bf16@29c1f369", info.get("backend")),
    "dit_backend": ("megakernel", st.get("dit_backend")),
    "mk_arena_dtype": (os.environ.get("S1_EXPECT_ARENA", "bfp8_b"), st.get("mk_arena_dtype")),
    "megakernel_built": (True, st.get("megakernel_built")),
    "weights n_kept/n_tensors": ([503, 534], [ws.get("n_kept"), ws.get("n_tensors")]),
    "affinity": ([8, 9, 10, 11], info.get("affinity")),
    "affinity_check.repinned": ([], aff.get("repinned")),
}
checks = {k: {"want": w, "got": g, "pass": w == g} for k, (w, g) in want.items()}
pcc = boot.get("pcc") or []
checks["boot_check.pcc >= min_pcc"] = {"want": boot.get("min_pcc"), "got": pcc,
                                       "pass": bool(pcc) and boot.get("pass") is True}
out = {"ready_s": int(ready[-1]["secs"]) if ready else None, "load_s": info.get("load_s"),
       "capture_s": boot.get("capture_s"), "warm_latency_ms": boot.get("warm_latency_ms"),
       "boot_samples": boot.get("samples"), "boot_pcc": pcc, "status_pub": info.get("status_pub"),
       "caches": {"g1_weights": tree(info.get("cache_root")), "tt_metal_cache": tree(ttmc)},
       "checks": checks, "pass": all(v["pass"] for v in checks.values())}
(sdir / "s1_boot.json").write_text(json.dumps(out, default=str, indent=1) + "\n")
for k, v in checks.items():
    print(f"{'PASS' if v['pass'] else 'FAIL'} {k}: got {v['got']} want {v['want']}", file=sys.stderr)
print(f"s1_boot: READY after {out['ready_s']} s (load {out['load_s']} s, capture {out['capture_s']} s), boot pcc "
      f"{pcc}, warm ms {out['warm_latency_ms']}; caches {json.dumps(out['caches'])}", file=sys.stderr)
sys.exit(0 if out["pass"] else 1)
EOF
job_exit $?
