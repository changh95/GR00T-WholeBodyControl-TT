"""A CPU stand-in for rttt/server_g1.py: the W-VLA wire plus predict_debug, with switchable faults (tests only).

The "model" is a fixed function of the observation and the noise (fake_pred), and decode() maps
action_pred_norm[0, :30, :35] to the W-VLA action groups, so a test can build goldens that match it exactly.
Faults (flags, or FAKE_G1_* in the environment because chip_session.sh builds the argv):
  --perturb S          add S x a deterministic pseudo-noise to every prediction (TT numerics)
  --nondeterministic   predict_debug adds fresh random noise (replays differ)
  --stale              predict_debug uses the previous call's image (stale state: ABAB mismatches)
  --unseeded-repeat    get_action without noise_seed reuses one noise (calls do not differ)
  --device-ms D, --slow-every K --slow-ms S   simulated device time
  --boot-s T, --die-before-ready, --ignore-term (chip_session's SIGKILL escalation)
  --status-pub URL     a 5 Hz status PUB with server_g1's src/seq/calls/backend keys (bench_g1 --status-sub)
  --info-json JSON     extra INFO keys (FAKE_G1_INFO), e.g. model / revision
Never binds 5555-5572 (the tests pass a free port).

    python fake_g1_server.py --port P [--debug-endpoints] [--log FILE] [--events FILE] [faults]
"""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import zmq

RT = Path(__file__).resolve().parents[2]
if str(RT) not in sys.path:
    sys.path.insert(0, str(RT))

from rtloop import schema, wire  # noqa: E402

NOISE_SHAPE = (1, 50, 128)


def fake_pred(obs: dict, noise: np.ndarray) -> np.ndarray:
    """float32 (1, 50, 128): depends on every state value, the image and the noise."""
    s = schema.state43(obs).astype(np.float64)
    img = float(np.asarray(obs[schema.VIDEO_KEY])[0, 0, ::16, ::16].mean()) / 255.0
    j = np.arange(128, dtype=np.float64)
    base = 0.4 * np.sin(0.07 * j[None, :] * (1.0 + img) + np.arange(50)[:, None] * 0.05 + s.sum())
    base[:, :43] += 0.5 * s[None, :]
    out = np.tanh(0.6 * np.asarray(noise, np.float64)[0] + base)
    return out[None].astype(np.float32)


def decode(pred: np.ndarray) -> dict:
    """W-VLA action dict (1, 30, D) float32 from action_pred_norm, one affine map per A35 column."""
    v = np.asarray(pred, np.float64)[0, : schema.H, : schema.A35]
    return {f"action.{name}": (0.5 * v[:, schema.ACTION_SLICES[name]] + 0.01)[None].astype(np.float32)
            for name, _ in schema.ACTION_GROUPS}


def pseudo(obs: dict, noise: np.ndarray) -> np.ndarray:
    """Deterministic N(0, 1) of the request bytes (the TT error model)."""
    seed = int.from_bytes(schema.obs_sha1(obs)[:8].tobytes(), "little") ^ int(np.asarray(noise).view(np.uint32).sum())
    return np.random.default_rng(seed).standard_normal(NOISE_SHAPE)


class FakeG1:
    def __init__(self, a):
        self.a = a
        self.gen = np.random.default_rng(a.seed)
        self.prev_img = None
        self.fixed = np.random.default_rng(1).standard_normal(NOISE_SHAPE).astype(np.float32)
        self.n = 0
        self.log = open(a.log, "a") if a.log else None

    def _model(self, obs: dict, noise: np.ndarray, debug: bool) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        incoming = np.asarray(obs[schema.VIDEO_KEY]).copy()
        if debug and self.a.stale and self.prev_img is not None:
            obs = dict(obs, **{schema.VIDEO_KEY: self.prev_img})  # the previous request's image
        self.prev_img = incoming
        t1 = time.perf_counter()
        pred = fake_pred(obs, noise)
        if self.a.perturb:
            pred = (pred + self.a.perturb * pseudo(obs, noise)).astype(np.float32)
        if debug and self.a.nondeterministic:
            pred = (pred + 1e-3 * np.random.default_rng().standard_normal(NOISE_SHAPE)).astype(np.float32)
        self.n += 1
        dev = self.a.device_ms + (self.a.slow_ms if self.a.slow_every and self.n % self.a.slow_every == 0 else 0)
        if dev:
            time.sleep(dev / 1e3)
        t2 = time.perf_counter()
        acts = decode(pred)
        t3 = time.perf_counter()
        ms = {"encode_ms": (t1 - t0) * 1e3, "device_ms": (t2 - t1) * 1e3, "decode_ms": (t3 - t2) * 1e3}
        if self.log:
            self.log.write(json.dumps({"n": self.n, "debug": debug, **ms}) + "\n")
            self.log.flush()
        return pred, {**acts, "_ms": ms}

    def get_action(self, observation, options=None):
        t_recv = time.monotonic_ns()
        rt = (options or {}).get("rt") if isinstance(options, dict) else None
        seed = rt.get("noise_seed") if isinstance(rt, dict) else None
        if seed is not None:
            self.gen = np.random.default_rng(int(seed))
        noise = self.fixed if (self.a.unseeded_repeat and seed is None) else \
            self.gen.standard_normal(NOISE_SHAPE).astype(np.float32)
        _, acts = self._model(observation, noise, debug=False)
        ms = acts.pop("_ms")
        info = {"rt": rt, "t_recv_ns": t_recv, "t_reply_ns": time.monotonic_ns(), "noise_seed": seed,
                "backend": "fake:tt", "tree": "fake", **ms}
        info["infer_ms"] = (info["t_reply_ns"] - t_recv) / 1e6
        return [acts, info]

    def predict_debug(self, observation, noise, options=None):
        if not self.a.debug_endpoints:
            raise PermissionError("debug endpoints disabled")
        noise = np.asarray(noise)
        if noise.dtype != np.float32 or noise.shape != NOISE_SHAPE:
            raise ValueError(f"noise must be float32 {NOISE_SHAPE}")
        pred, acts = self._model(observation, noise, debug=True)
        ms = acts.pop("_ms")
        return {"action_pred_norm": pred, "actions": acts, **ms}


def _publish(ctx, url, srv, stop) -> None:
    """The fake's status stream: the real server_g1's envelope and a few of its keys, 5 Hz."""
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.setsockopt(zmq.SNDHWM, 10)
    pub.bind(url)
    seq = 0
    while not stop.wait(0.2):
        seq += 1
        msg = {"src": "server_g1", "pid": os.getpid(), "t_ns": time.monotonic_ns(), "seq": seq, "backend": "fake:tt",
               "ready": True, "calls": srv.n, "errors": 0, "telemetry": None}
        pub.send_multipart([b"status", wire.dumps(msg)], zmq.NOBLOCK)
    pub.close(0)


def _events(path, msg):
    if path:
        with open(path, "a") as f:
            f.write(f"{time.monotonic():.6f} {msg}\n")


def main(argv=None) -> int:
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--debug-endpoints", action="store_true")
    ap.add_argument("--log", default=None)
    ap.add_argument("--events", default=env("FAKE_EVENTS"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", type=float, default=float(env("FAKE_G1_PERTURB", 0)))
    ap.add_argument("--nondeterministic", action="store_true")
    ap.add_argument("--stale", action="store_true")
    ap.add_argument("--unseeded-repeat", action="store_true")
    ap.add_argument("--device-ms", type=float, default=float(env("FAKE_G1_DEVICE_MS", 0)))
    ap.add_argument("--slow-every", type=int, default=0)
    ap.add_argument("--slow-ms", type=float, default=0.0)
    ap.add_argument("--boot-s", type=float, default=float(env("FAKE_G1_BOOT_S", 0)))
    ap.add_argument("--die-before-ready", action="store_true", default=env("FAKE_G1_DIE_BEFORE_READY") == "1")
    ap.add_argument("--ignore-term", action="store_true", default=env("FAKE_G1_IGNORE_TERM") == "1")
    ap.add_argument("--status-pub", default=None)
    ap.add_argument("--info-json", default=env("FAKE_G1_INFO"))
    ap.add_argument("--cpus", default=None, help="accepted like server_g1's; chip_lib pins the process itself")
    a = ap.parse_args(argv)
    a.info = info = {"backend": "fake:tt", "tree": "fake", "proto": 2, **json.loads(a.info_json or "{}")}
    assert not 5555 <= a.port <= 5572, "tests never bind the reserved ports"
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{a.port}")  # bind first, like server_g1
    _events(a.events, f"server_g1 start pid={os.getpid()} argv={' '.join(sys.argv[1:])}")
    stop = {"flag": False}
    if a.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    if a.boot_s:
        time.sleep(a.boot_s)
    if a.die_before_ready:
        _events(a.events, "server_g1 died")
        return 1
    srv = FakeG1(a)
    handlers = {"ping": lambda: {"status": "ok"}, "info": lambda: dict(info),
                "reset": lambda **_: {}, "get_modality_config": lambda: {},
                "get_action": srv.get_action, "predict_debug": srv.predict_debug}
    pub_stop = threading.Event()
    if a.status_pub:
        threading.Thread(target=_publish, args=(ctx, a.status_pub, srv, pub_stop), daemon=True).start()
    print(f"server_g1 READY tcp://127.0.0.1:{a.port}", flush=True)
    _events(a.events, "server_g1 ready")
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    while not stop["flag"]:
        if not poller.poll(100):
            continue
        req = wire.loads(sock.recv())
        try:
            ep = req.get("endpoint", "get_action")
            fn = handlers[ep]
            data = req.get("data") or {}
            res = fn(**data) if ep in ("get_action", "predict_debug", "reset") else fn()
        except Exception as e:  # noqa: BLE001 - reply like the real server
            res = {"error": f"{type(e).__name__}: {e}"}
        sock.send(wire.dumps(res))
    pub_stop.set()
    _events(a.events, "server_g1 stop")
    sock.close(0)
    ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
