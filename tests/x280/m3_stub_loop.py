"""Stand-in for rtloop.loop behind the M3 jobs in tests/x280/test_m3_sessions_fake.py: never a --mode rt loop.

jobs/m3_faults.sh runs it through M3_LOOP_MODULE and jobs/m3_soak.sh through demo.sh's DEMO_LOOP_MODULE. It accepts
the loop's CLI ("--mode", "--vla", "--wbc", "--duration", "--tag", "--runs-root", "--control", "--wbc-alt",
"--swap-wbc-every", anything else ignored), keeps its command line (argv.json in the run dir), writes a synthetic RT
run in the loop's on-disk format (rtloop.telemetry: TickLog, RequestLog, EpisodeLog), prints the loop's "run dir"
line and then waits until --duration has passed in wall time (M3_STUB_SLEEP_S caps the wait) or SIGTERM / SIGINT,
meanwhile sending one W-WBC gear_wbc_action to a tcp:// --wbc every 50 ms (so x280d counts real calls; a missed
reply re-creates the REQ socket, as rt_sim's client does).
The run: 20 ms ticks at 50 Hz, episodes of 1440 ticks, the WBC on the X280 sim (wbc_be X280_SIM, 0.7 ms) shadowed
by ORT with shadow_da M3_STUB_SHADOW_DA (default 7e-3, the fp16 kernel's size) and, with --swap-wbc-every S, on ORT
for every other S seconds of ticks (a swap per S seconds: X280_SIM -> ORT -> X280_SIM, as demo.sh m3
cycles). M3_STUB_RUN_S sets the run's length (default --duration), M3_STUB_FALLBACKS N marks N ticks wbc_fb.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT))

from rtloop import telemetry  # noqa: E402
from rtloop.schema import Outcome, WbcBackend  # noqa: E402


def synth(root: Path, tag: str, secs: float, swap_s, shadow_da: float, fallbacks: int, cfg: dict) -> Path:
    run = telemetry.RunDir(root, tag=tag, config=cfg)
    tl, rl, el = telemetry.TickLog(run), telemetry.RequestLog(run), telemetry.EpisodeLog(run)
    n, ep_len, t0, seq = max(int(secs * 50), 2), 1440, time.monotonic_ns(), 0
    swap_ticks = int(swap_s * 50) if swap_s else 0
    for i in range(n):
        ep, k = divmod(i, ep_len)
        r = tl.next()
        r["ep"], r["k"] = ep, k
        r["t_start_ns"] = t0 + i * 20_000_000 + ep * 3_000_000_000
        r["t_end_ns"] = r["t_start_ns"] + 6_000_000
        r["period_us"], r["compute_us"] = 20000.0, 6000.0
        remote = not swap_ticks or (i // swap_ticks) % 2 == 0
        fb = i < fallbacks
        r["wbc_be"] = WbcBackend.X280_SIM if remote and not fb else WbcBackend.ORT
        r["wbc_fb"] = 1 if fb else 0
        r["wbc_rtt_us"] = 700.0 if remote else 150.0
        r["shadow_da"] = shadow_da if remote and not fb else float("nan")
        r["row"], r["age"] = k % 16, k % 16
        tl.commit()
        if k % 16 == 0:
            seq += 1
            send = int(r["t_start_ns"]) + 300_000
            rl.add({"seq": seq, "ep": ep, "k_obs": k, "k_arr": k + 3, "d": 3, "t_send_ns": send,
                    "t_recv_ns": send + 47_000_000, "outcome": Outcome.APPLIED, "srv_t_recv_ns": send + 500_000,
                    "srv_t_reply_ns": send + 45_500_000})
        if k == ep_len - 1 or i == n - 1:
            el.add({"ep": ep, "seed": 3000 + ep, "end": "TRUNC"})
    tl.close()
    rl.close()
    el.close()
    return run.path


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)  # the loop's own --s 16 must not read as --swap-wbc-every 16
    ap.add_argument("--mode", default="rt")
    ap.add_argument("--vla", default="")
    ap.add_argument("--wbc", default="ort")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--tag", default="stub")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--control", default=None)
    ap.add_argument("--wbc-alt", default="")
    ap.add_argument("--swap-wbc-every", type=float, default=None)
    ap.add_argument("--viewer", action="store_true")
    a, _unknown = ap.parse_known_args()
    stop = {"flag": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    t_start = time.monotonic()
    secs = float(os.environ.get("M3_STUB_RUN_S") or a.duration)
    cfg = {"args": {"mode": a.mode, "viewer": a.viewer, "wbc": a.wbc, "vla": a.vla}, "vla_info": {"backend": "mock"}}
    run = synth(Path(a.runs_root), a.tag, secs, a.swap_wbc_every, float(os.environ.get("M3_STUB_SHADOW_DA", 7e-3)),
                int(os.environ.get("M3_STUB_FALLBACKS", 0)), cfg)
    (run / "argv.json").write_text(json.dumps(sys.argv[1:]))
    print(f"rt_sim: run dir {run}", flush=True)
    wait = min(a.duration, float(os.environ.get("M3_STUB_SLEEP_S", a.duration)))
    wbc = WbcCaller(a.wbc) if a.wbc.startswith("tcp://") else None
    i = 0
    while not stop["flag"] and time.monotonic() - t_start < wait:
        if wbc is not None:
            wbc.call(i)
            i += 1
        time.sleep(0.05)
    if wbc is not None:
        wbc.close()
    return 130 if stop["flag"] else 0


class WbcCaller:
    """W-WBC calls with random obs, alternating Balance / Walk; a reply missed within 1 s re-creates the socket."""

    def __init__(self, url: str):
        import numpy as np
        import zmq

        self.np, self.zmq, self.url = np, zmq, url
        self.ctx = zmq.Context()
        self.rng = np.random.default_rng(0)
        self._connect()

    def _connect(self):
        self.s = self.ctx.socket(self.zmq.REQ)
        self.s.setsockopt(self.zmq.LINGER, 0)
        self.s.connect(self.url)

    def call(self, i: int) -> None:
        from rtloop import wire

        pol = ("GR00T-WholeBodyControl-Balance", "GR00T-WholeBodyControl-Walk")[i & 1]
        obs = self.rng.standard_normal((1, 516)).astype(self.np.float32)
        self.s.send(wire.request("gear_wbc_action", {"policy": pol, "obs": obs, "tick": i, "seq": i}))
        if self.s.poll(1000):
            self.s.recv()
        else:
            self.s.close(0)
            self._connect()

    def close(self):
        self.s.close(0)
        self.ctx.term()


if __name__ == "__main__":
    sys.exit(main())
