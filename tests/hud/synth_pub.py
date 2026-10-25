"""Synthetic rt_sim telemetry for the HUD tests: real TICK_DTYPE rows through telemetry.tick_msg and the
real telemetry.Publisher, so the HUD is tested against exactly the bytes the loop sends.

The stream imitates M1 timing: 20 ms ticks, ~7 ms compute with an overrun every 97 ticks, a VLA request every
16 ticks answered 4 ticks later (d = 4), an ego thumbnail per request, meta at 1 Hz, Walk every other 200
ticks, a starved stretch in episode 1. Usable as a thread (tests) or a script (preview):

    simpy tests/hud/synth_pub.py --url tcp://127.0.0.1:* --seconds 5
"""

import argparse
import math
import threading
import time

import numpy as np

from rtloop import telemetry
from rtloop.schema import TICK_DTYPE

META = {
    "m1": {"sim": "host-cpu", "vla": "cuda:rtx5090:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc",
           "wbc": "cuda-graph:rtx5090"},
    "m2": {"sim": "host-cpu", "vla": "tt:p150a:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate-bf16@2a5b61cc",
           "wbc": "cuda-graph:rtx5090"},
    "m3": {"sim": "host-cpu", "vla": "tt:p150a:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate-bf16@2a5b61cc",
           "wbc": "x280:l2cpu3:h0@800MHz"},
}
# rt_sim's meta `targets` (the loop's swap targets: its own backends plus --vla-alt / --wbc-alt), as demo.sh configures
# them: m2 with the GPU VLA as the key-v alternative, m3 as `demo.sh m3 --no-alt` (nothing on the GPU).
TARGETS = {"m1": {"vla": ["gpu"], "wbc": ["gpu", "ort"]}, "m2": {"vla": ["tt", "gpu"], "wbc": ["gpu", "ort"]},
           "m3": {"vla": ["tt"], "wbc": ["x280", "ort"]}}


def ego_image(seq: int) -> np.ndarray:
    y, x = np.mgrid[0:120, 0:160]
    img = np.zeros((120, 160, 3), np.uint8)
    img[..., 0] = (x * 255 // 159).astype(np.uint8)
    img[..., 1] = (y * 255 // 119).astype(np.uint8)
    img[..., 2] = (seq * 37) % 256
    return img


class SynthStream:
    """Generates (topic, fields) in loop order for ep 0.. with `ticks_per_ep` ticks each."""

    def __init__(self, ticks_per_ep: int = 600, profile: str = "m1", s: int = 16, d: int = 4, seed0: int = 1000):
        self.ticks_per_ep, self.profile, self.s, self.d, self.seed0 = ticks_per_ep, profile, s, d, seed0
        self.row = np.zeros((), dtype=TICK_DTYPE)
        self.seq = 0

    def episode(self, ep: int):
        r, s, d = self.row, self.s, self.d
        act_seq, act_kobs = 0, 0
        inflight = None  # (seq, k_obs)
        t0 = time.monotonic_ns()
        for k in range(self.ticks_per_ep):
            r[...] = 0
            # request / arrival bookkeeping (as splicer v1: trigger every s ticks, applied at k_obs + d)
            if inflight is not None and k == inflight[1] + d:
                act_seq, act_kobs = inflight
                yield "req", {"seq": act_seq, "gen": ep + 1, "ep": ep, "k_obs": act_kobs, "k_arr": k, "d": d,
                              "d_hat": d, "s_eff": s, "rtt_ms": 62.0 + (act_seq % 5), "ser_ms": 0.12,
                              "srv": {"infer_ms": 58.8, "encode_ms": None, "device_ms": None, "decode_ms": None},
                              "outcome": "APPLIED"}
                inflight = None
            if inflight is None and k % s == 0:
                self.seq += 1
                inflight = (self.seq, k)
                yield "ego", (self.seq, k, ego_image(self.seq))
            starved = ep == 1 and 300 <= k < 312
            row_i = k - act_kobs if act_seq else -1
            r["ep"], r["k"] = ep, k
            r["t_start_ns"] = t0 + k * 20_000_000
            r["period_us"] = 0.0 if k == 0 else 20000.0 + 40.0 * math.sin(k * 0.7)
            r["compute_us"] = 22500.0 if k % 97 == 50 else 6800.0 + 300.0 * math.sin(k * 0.3)
            r["overrun"] = r["compute_us"] > 20000
            r["late_us"] = 3.0
            r["slack_us"] = 20000.0 - r["compute_us"]
            r["sim_us"] = 5300.0
            r["viewer_us"] = 900.0
            r["wbc_rtt_us"] = 610.0 + 50.0 * math.sin(k)
            r["wbc_dev_us"] = 410.0
            r["wbc_be"] = 2 if self.profile == "m3" else 1
            r["wbc_fb"] = 1 if k % 211 == 100 else 0
            if r["wbc_fb"]:
                r["wbc_be"] = 0
            walk = (k // 200) % 2 == 1
            r["policy"] = 1 if walk else 0
            r["shadow_da"] = 3.0e-6
            r["chunk_seq"] = act_seq
            r["row"] = 30 if starved else row_i
            r["age"] = max(row_i, 0)
            r["rows_left"] = 0 if starved else max(0, 29 - row_i)
            r["starved"] = starved
            r["in_flight"] = inflight is not None
            r["splice"] = row_i == d
            r["jump_rad"] = 0.01 if row_i == d else 0.0
            r["nav_wbc"] = (0.2 if walk else 0.0, 0.0, 0.1 * math.sin(k / 50))
            r["pelvis_z"] = 0.752
            r["success"] = ep == 0 and k == self.ticks_per_ep - 1
            yield "tick", telemetry.tick_msg(r)
            if k % 50 == 0:
                yield "meta", {"processor": META[self.profile], "ep": ep, "k": k, "mode": "rt",
                               "rt_factor": 0.999, "seed": self.seed0 + ep,
                               "current": {"vla": "gpu" if self.profile == "m1" else "tt",
                                           "wbc": {"m1": "gpu", "m2": "gpu", "m3": "x280"}[self.profile]},
                               "targets": TARGETS[self.profile],
                               "endpoints": {"vla": "tcp://127.0.0.1:5565", "wbc": "tcp://127.0.0.1:5556"}}


def publish(pub: telemetry.Publisher, item) -> None:
    topic, fields = item
    if topic == "tick":
        pub.tick(fields)
    elif topic == "req":
        pub.req(fields)
    elif topic == "meta":
        pub.meta(fields)
    elif topic == "ego":
        pub.ego(*fields)


class SynthPublisher(threading.Thread):
    """Publishes SynthStream at `rate_hz` ticks/s (0 = as fast as possible) until `n_ticks` or stop().

    With `stall_after` the thread publishes that many ticks, then keeps the socket bound but silent (a
    stalled loop) until stop()."""

    def __init__(self, url: str = "tcp://127.0.0.1:*", rate_hz: float = 50.0, n_ticks: int = 10**9,
                 stall_after: int | None = None, profile: str = "m1", ticks_per_ep: int = 600, hwm: int = 100):
        super().__init__(daemon=True)
        self.pub = telemetry.Publisher(url, hwm=hwm)
        self.url = self.pub.url
        self.rate_hz, self.n_ticks, self.stall_after = rate_hz, n_ticks, stall_after
        self.stream = SynthStream(ticks_per_ep=ticks_per_ep, profile=profile)
        self._halt = threading.Event()
        self.ticks_sent = 0
        self.stalled = threading.Event()

    def run(self):
        period = 1.0 / self.rate_hz if self.rate_hz else 0.0
        t_next = time.monotonic()
        ep = 0
        while not self._halt.is_set() and self.ticks_sent < self.n_ticks:
            for item in self.stream.episode(ep):
                if self._halt.is_set() or self.ticks_sent >= self.n_ticks:
                    break
                publish(self.pub, item)
                if item[0] != "tick":
                    continue
                self.ticks_sent += 1
                if self.stall_after is not None and self.ticks_sent >= self.stall_after:
                    self.stalled.set()
                    self._halt.wait()  # bound but silent: a stalled loop
                    return
                if period:
                    t_next += period
                    delay = t_next - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            ep += 1

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        self.pub.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="tcp://127.0.0.1:*")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--profile", default="m1", choices=sorted(META))
    args = ap.parse_args()
    p = SynthPublisher(args.url, profile=args.profile)
    print(f"URL {p.url}", flush=True)
    p.start()
    time.sleep(args.seconds)
    p.stop()
    print(f"sent {p.ticks_sent} ticks, dropped {p.pub.dropped}", flush=True)


if __name__ == "__main__":
    main()
