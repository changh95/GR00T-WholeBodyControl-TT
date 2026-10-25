"""W-WBC on the RTX 5090: the vendored CUDA-graph gear_wbc MLPs behind REP 127.0.0.1:5556.

Its own process and CUDA context, so the WBC is never queued behind a VLA call on a shared REP
socket (PLAN.md D3, R1). Requests are served from the main thread: CudaGraphPolicy synchronises
the stream that was current when it captured the graph, which is the main thread's, so serving
from any other thread would return before the replay finished. The vendored module is hashed
against vendor/SHA256SUMS before it is imported (the source in R is untracked), and every policy is
checked at boot against a float64 numpy forward pass of the ONNX initializers, because this venv
has no onnxruntime.

    srvpy -m rtserve.wbc_gpu_server --port 5556 --onnx-dir $ONNX
"""

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import zmq

from rtloop import wire
from rtloop.schema import WBC_ACT_DIM, WBC_OBS_DIM
from rtserve import wbc_numpy_ref

VENDOR = Path(__file__).resolve().parent / "vendor"
DEFAULT_ONNX = os.environ.get("ONNX") or str(
    Path.home() / "experiments/gr00t/Isaac-GR00T-n1.6/external_dependencies/GR00T-WholeBodyControl"
    / "gr00t_wbc/sim2mujoco/resources/robots/g1/policy"
)
SELF_CHECK_TOL = 1e-4


def check_vendor() -> dict[str, str]:
    """sha256 of every file listed in vendor/SHA256SUMS; exits if any differs."""
    sums = {}
    for line in (VENDOR / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            want, name = line.split()
            got = hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
            if got != want:
                sys.exit(f"vendor/{name}: sha256 {got} != SHA256SUMS {want}")
            sums[name] = got
    if "gear_wbc_policy.py" not in sums:
        sys.exit("vendor/SHA256SUMS does not list gear_wbc_policy.py")
    return sums


def load_vendor():
    """Import the exact file that was hashed (not whatever `gr00t.policy` resolves to)."""
    spec = importlib.util.spec_from_file_location("rt_vendor_gear_wbc_policy",
                                                  VENDOR / "gear_wbc_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gpu_tag(device: str) -> str:
    """'NVIDIA GeForce RTX 5090' -> 'rtx5090'."""
    import torch

    name = torch.cuda.get_device_name(torch.device(device)).lower()
    for prefix in ("nvidia ", "geforce "):
        name = name.replace(prefix, "")
    return name.replace(" ", "")


def self_check(policies, onnx_dir: str, n: int) -> dict[str, float]:
    """max |gpu - numpy float64| per policy on n random batch-1 inputs; exits above SELF_CHECK_TOL."""
    refs = wbc_numpy_ref.load_dir(onnx_dir)
    rng = np.random.default_rng(0)
    errs = {}
    for stem in policies.policies:
        x = rng.standard_normal((n, WBC_OBS_DIM)).astype(np.float32)
        got = np.concatenate([policies.get_action(stem, x[i:i + 1])["action"] for i in range(n)])
        errs[stem] = float(np.max(np.abs(got.astype(np.float64) - refs[stem](x))))
        if not errs[stem] < SELF_CHECK_TOL:
            sys.exit(f"self-check {stem}: max |err| = {errs[stem]:.3e} >= {SELF_CHECK_TOL}")
    return errs


class WbcHandler:
    """Endpoint handlers; every reply is a dict, errors are {"error": str}."""

    def __init__(self, policies, info: dict):
        self.policies, self._info = policies, info

    def ping(self) -> dict:
        return {"status": "ok", "message": "Server is running"}

    def info(self) -> dict:
        return self._info

    def gear_wbc_action(self, policy: str, obs: np.ndarray, tick: int | None = None,
                        seq: int | None = None) -> dict:
        if not (isinstance(obs, np.ndarray) and obs.shape == (1, WBC_OBS_DIM)
                and obs.dtype == np.float32):
            raise ValueError(f"obs must be float32 (1, {WBC_OBS_DIM}), got "
                             f"{getattr(obs, 'dtype', type(obs))} {getattr(obs, 'shape', None)}")
        if policy not in self.policies.policies:
            raise KeyError(f"unknown policy {policy!r}")
        t0 = time.perf_counter_ns()
        action = self.policies.get_action(policy, obs)["action"]
        compute_us = (time.perf_counter_ns() - t0) / 1e3
        assert action.shape == (1, WBC_ACT_DIM) and action.dtype == np.float32
        return {"action": action, "tick": tick, "seq": seq, "compute_us": compute_us}

    def handle(self, msg: bytes) -> bytes:
        try:
            req = wire.loads(msg)
            endpoint = req.get("endpoint")
            data = req.get("data") or {}
            if endpoint == "gear_wbc_action":
                out = self.gear_wbc_action(**data)
            elif endpoint == "ping":
                out = self.ping()
            elif endpoint == "info":
                out = self.info()
            else:
                out = {"error": f"Unknown endpoint: {endpoint}"}
        except Exception as e:  # the client falls back to ORT on any error reply
            out = {"error": f"{type(e).__name__}: {e}"}
        return wire.dumps(out)


def _recv_spin(sock: zmq.Socket) -> bytes:
    """Busy-poll for the next request. On a dedicated core this removes the sleeping process's
    wake-up latency from every WBC round trip (T-1.4 under full load: RTT - compute = 0.71 ms p50).
    Each miss yields the CPU: --cpus pins zmq's I/O thread to the same core, and a poll that never
    yields delays it by a scheduler slice (50 Hz bench: overhead p99 0.92 ms without, 0.37 ms with)."""
    while True:
        try:
            return sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            os.sched_yield()


def serve(handler: WbcHandler, port: int, spin: bool = False):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{port}")
    print(f"wbc_gpu_server ready on tcp://127.0.0.1:{port} {handler.info()}", flush=True)
    recv = (lambda: _recv_spin(sock)) if spin else sock.recv
    try:
        while True:
            sock.send(handler.handle(recv()))
    finally:
        sock.close(linger=0)
        ctx.term()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--onnx-dir", default=DEFAULT_ONNX)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self-check-n", type=int, default=64, help="random inputs per policy at boot")
    ap.add_argument("--cpus", default=None, help="CPU affinity, e.g. '4' (PLAN.md §1.1)")
    # Off by default: measured inconclusive in T-1.4 because the SMT sibling (20 for CPU 4) was busy.
    ap.add_argument("--spin", action="store_true",
                    help="busy-poll for requests (use only with a dedicated --cpus core and an idle SMT sibling)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if args.cpus:
        os.sched_setaffinity(0, {int(c) for c in args.cpus.split(",")})

    vendor = check_vendor()
    gear = load_vendor()
    policies = gear.GearWbcPolicies(args.onnx_dir, args.device)
    errs = self_check(policies, args.onnx_dir, args.self_check_n)
    print(f"  self-check vs numpy float64 (n={args.self_check_n}): "
          + ", ".join(f"{k} {v:.2e}" for k, v in errs.items()), flush=True)
    info = {
        "backend": f"cuda-graph:{gpu_tag(args.device)}",
        "proto": 2,
        "policies": sorted(policies.policies),
        "onnx_dir": str(args.onnx_dir),
        "vendor_sha256": vendor["gear_wbc_policy.py"],
        "self_check_max_err": errs,
        "pid": os.getpid(),
        "spin": args.spin,
    }
    try:
        serve(WbcHandler(policies, info), args.port, spin=args.spin)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
