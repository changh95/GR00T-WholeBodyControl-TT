"""Synthetic device status streams for the HUD chip panel tests (PLAN.md §4.3 item 7; INTERFACES.md "Device status
streams"): x280d's and server_g1's out-of-band PUB, multipart [b"status", rtloop.wire msgpack], as the contract
describes them. The values are test data, not measurements.

    simpy tests/hud/synth_status.py --src x280d --url tcp://127.0.0.1:* --seconds 5
"""

import argparse
import os
import threading
import time

import zmq

from rtloop import wire

TELEMETRY = {"aiclk_mhz": 1350, "l2cpuclk_mhz": [800, 800, 800, 800], "asic_temp_c": 52.5, "vcore_mv": 750,
             "tdp_w": 48, "tdc_a": 62, "input_power_w": 55, "timer_heartbeat": 1000}
CORE_MAP_COUNTS = {"grid": [11, 10], "megakernel": {"readers": 8, "hub": 1, "compute": 96, "spare": 5}}


def x280d_msg(seq: int, pid: int = 4242, **over) -> dict:
    m = {"src": "x280d", "pid": pid, "t_ns": time.monotonic_ns(), "seq": seq,
         "backend": "x280:l2cpu3:h0@800MHz:v2-fp16", "uptime_s": 0.2 * seq,
         "tile": 3, "tile_state": "released", "monitor_state": 1, "heartbeat_age_us": 35.0, "seq_done": 50 * seq,
         "cycles_last": 328000, "cycles_max": 401000, "compute_us_last": 410.0, "duty": 0.0205,
         "n_runs": 50 * seq, "n_overrun": 0, "n_reentry": 1, "beu_accrued": 0, "ccache_ecc": {"fix": 0, "fail": 0},
         "variant": "v2-fp16", "fw_ver": 0x00020001, "build_id": 0x1A2B3C4D,
         "telemetry": {**TELEMETRY, "timer_heartbeat": 1000 + seq}}
    m.update(over)
    return m


def server_g1_msg(seq: int, pid: int = 4343, core_map: dict | None = None, **over) -> dict:
    m = {"src": "server_g1", "pid": pid, "t_ns": time.monotonic_ns(), "seq": seq,
         "backend": "tt:p150a:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate-bf16@29c1f369", "uptime_s": 0.2 * seq,
         "calls": 3 * seq, "errors": 0,
         "last_ms": {"encode": 2.6, "device": 39.4, "decode": 0.15, "total": 42.2},
         "p50_ms": {"encode": 2.5, "device": 39.3, "decode": 0.15, "total": 42.0},
         "device_duty": 0.123, "trace": True, "cq": 2, "core_map": core_map or CORE_MAP_COUNTS,
         "tree": "MK@009de892c895",
         # server_g1 has no telemetry read (contract: None when unknown, never invented)
         "telemetry": dict.fromkeys(TELEMETRY)}
    m.update(over)
    return m


MAKERS = {"x280d": x280d_msg, "server_g1": server_g1_msg}


class StatusPublisher(threading.Thread):
    """One device process's status PUB on a random port (SNDHWM 10, NOBLOCK, as the contract says).

    mode: "ok"; "stall" (publishes `stall_after` messages, then stays bound and silent); "garbage" (every other
    frame is undecodable, a wrong topic, a single part, a non-dict, a foreign src or wrongly typed fields);
    "restart" (the pid changes half way). `over` overrides message fields."""

    def __init__(self, src: str, rate_hz: float = 10.0, mode: str = "ok", stall_after: int = 10, **over):
        super().__init__(daemon=True)
        self.src, self.rate_hz, self.mode, self.stall_after, self.over = src, rate_hz, mode, stall_after, over
        self.ctx = zmq.Context(io_threads=1)
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        assert not 5555 <= self.port <= 5572
        self.url = f"tcp://127.0.0.1:{self.port}"
        self.sent = 0
        self._halt = threading.Event()
        self.stalled = threading.Event()

    def _send(self, parts) -> None:
        try:
            self.sock.send_multipart(parts, zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _garbage(self, i: int):
        bad = [
            [b"status", b"\xc1\xc1\xc1"],                                       # undecodable msgpack
            [b"weird", wire.dumps(MAKERS[self.src](i))],                       # wrong topic
            [b"status"],                                                       # one part
            [b"status", wire.dumps([1, 2, 3])],                                # not a map
            [b"status", wire.dumps({"src": "someone", "seq": i})],             # foreign src
            [b"status", wire.dumps({"src": self.src, "pid": "x", "seq": None, "tile": "three",
                                    "last_ms": [1, 2], "p50_ms": "fast", "core_map": {"grid": "11x10"},
                                    "telemetry": [1], "duty": "high", "ccache_ecc": 7, "calls": float("nan"),
                                    "monitor_state": {"a": 1}, "heartbeat_age_us": True})],  # every field wrong
        ]
        return bad[i % len(bad)]

    def run(self):
        period = 1.0 / self.rate_hz
        i = 0
        while not self._halt.is_set():
            i += 1
            if self.mode == "garbage" and i % 2 == 0:
                self._send(self._garbage(i // 2))
            else:
                pid = 4242 + (1000 if self.mode == "restart" and i > 10 else 0)
                self._send([b"status", wire.dumps(MAKERS[self.src](i, pid=pid, **self.over))])
                self.sent += 1
            if self.mode == "stall" and self.sent >= self.stall_after:
                self.stalled.set()
                self._halt.wait()
                return
            self._halt.wait(period)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        self.sock.close(linger=0)
        self.ctx.term()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=sorted(MAKERS), default="x280d")
    ap.add_argument("--mode", default="ok")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    p = StatusPublisher(args.src, mode=args.mode)
    print(f"URL {p.url} pid {os.getpid()}", flush=True)
    p.start()
    time.sleep(args.seconds)
    p.stop()


if __name__ == "__main__":
    main()
