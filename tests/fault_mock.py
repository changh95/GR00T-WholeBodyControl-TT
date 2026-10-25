"""Test-only arm-jump behaviour on top of rtserve.mock_vla_server, for the T-1.6 fault test.

It exists only to break the loop, so it lives under tests/ and leaves the mock itself unchanged:

  --jump-ep E --jump-k K --jump-rad A   get_action requests of episode E (options.rt.ep) with
                                         k_obs >= K answer arm rows = state + A on all 14 arm
                                         joints, every row: a step the arm PD turns into a joint
                                         speed above the safety monitor's 6 rad/s (SAFETY_FAULT)

Everything else (--mode hold|detfn|ramp, --latency-ms, --stall-after, ...) is the base mock's.

    python tests/fault_mock.py --port 0 --mode detfn --jump-ep 0 --jump-k 100 --jump-rad 1.5
"""

import argparse
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

RT = Path(__file__).resolve().parents[1]
if str(RT) not in sys.path:
    sys.path.insert(0, str(RT))

from rtloop import wire  # noqa: E402
from rtloop.schema import ARMS, rows_to_actions  # noqa: E402
from rtserve import mock_vla_server as mvs  # noqa: E402


class FaultMock(mvs.MockVlaServer):
    """The base mock plus the arm-jump episode."""

    def __init__(self, mode: str, port: int = 0, *, jump_ep: int = -1, jump_k: int = 0,
                 jump_rad: float = 0.0, **kw):
        super().__init__(mode, port, **kw)
        self.jump_ep, self.jump_k, self.jump_rad = int(jump_ep), int(jump_k), np.float32(jump_rad)
        self.n_jumped = 0

    def handle(self, raw: bytes) -> bytes | None:
        if self.jump_ep < 0:
            return super().handle(raw)
        t_recv = time.monotonic_ns()
        try:
            req = wire.loads(raw)
        except Exception:  # noqa: BLE001 - let the base mock produce its error reply
            return super().handle(raw)
        if not isinstance(req, dict) or req.get("endpoint", "get_action") != "get_action":
            return super().handle(raw)
        data = req.get("data") or {}
        rt = (data.get("options") or {}).get("rt") or {}
        if self.jump_ep >= 0 and rt.get("ep") == self.jump_ep and int(rt.get("k_obs", -1)) >= self.jump_k:
            return self._jump(data.get("observation"), rt, t_recv)
        return super().handle(raw)

    def _jump(self, obs, rt: dict, t_recv: int) -> bytes | None:
        idx = self.n_get
        self.n_get += 1
        deadline = t_recv + self._delay_ns(idx)
        mvs.check_obs(obs)
        rows = mvs.rows_hold(obs)
        rows[:, ARMS] += self.jump_rad
        self.n_jumped += 1
        if not self._sleep_until(deadline):
            return None
        t_reply = time.monotonic_ns()
        info = {"rt": rt, "t_recv_ns": t_recv, "t_reply_ns": t_reply,
                "infer_ms": (t_reply - t_recv) / 1e6, "backend": f"mock:{self.mode}+jump"}
        return wire.dumps([rows_to_actions(rows), info])

    def info(self) -> dict:
        return {**super().info(), "n_jumped": self.n_jumped}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--mode", choices=("hold", "detfn", "ramp"), default="detfn")
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--stall-after", type=int, default=-1)
    ap.add_argument("--stall-s", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jump-ep", type=int, default=-1)
    ap.add_argument("--jump-k", type=int, default=0)
    ap.add_argument("--jump-rad", type=float, default=1.5)
    a = ap.parse_args(argv)
    if 5555 <= a.port <= 5572:
        ap.error("5555-5572 are reserved (PLAN.md §2; the existing GR00T server owns 5555)")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    srv = FaultMock(a.mode, a.port, jump_ep=a.jump_ep, jump_k=a.jump_k, jump_rad=a.jump_rad,
                    latency_ms=a.latency_ms, jitter_ms=a.jitter_ms, stall_after=a.stall_after,
                    stall_s=a.stall_s, seed=a.seed)
    try:
        print(f"READY mode={srv.mode} url={srv.url} pid={os.getpid()}", flush=True)
        srv.serve_forever(stop)
    finally:
        srv.close()


if __name__ == "__main__":
    main()
