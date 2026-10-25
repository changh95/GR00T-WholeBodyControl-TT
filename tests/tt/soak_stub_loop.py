"""Stand-in for rtloop.loop behind demo.sh in the T-2.6 / T-2.8 job tests (tests/tt/test_s1b_jobs_fake.py).

No sim, no device, no RT pinning: it keeps the command line it was given (argv.json in the run dir), writes a
synthetic RT run on TT (tests/tt/synth_runs.py) under --runs-root with --tag, and prints the loop's "run dir" line,
which demo.sh links its logs from. SOAK_STUB_MINUTES (default --duration / 60) sets the run's length,
SOAK_STUB_RTT="45,52" ramps the server time from 45 to 52 ms (a drifting run), SOAK_STUB_RC its exit code.

    DEMO_LOOP_MODULE=$RT/tests/tt/soak_stub_loop.py $RT/scripts/demo.sh m2 --in-session ... -- --runs-root DIR
"""

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from synth_runs import synth_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="rt")
    ap.add_argument("--vla", default="")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--tag", default="stub")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--viewer", action="store_true")
    a, _unknown = ap.parse_known_args()
    minutes = float(os.environ.get("SOAK_STUB_MINUTES") or a.duration / 60)
    rtt = tuple(float(x) for x in os.environ.get("SOAK_STUB_RTT", "45,45").split(","))
    run = synth_run(Path(a.runs_root), minutes=minutes, rtt_ms=rtt, viewer=a.viewer, tag=a.tag, vla=a.vla)
    (run / "argv.json").write_text(json.dumps(sys.argv[1:]))
    print(f"rt_sim: run dir {run}", flush=True)
    return int(os.environ.get("SOAK_STUB_RC", "0"))


if __name__ == "__main__":
    sys.exit(main())
