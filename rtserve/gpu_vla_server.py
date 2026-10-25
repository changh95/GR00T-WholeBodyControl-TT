"""W-VLA on the RTX 5090 for M1: GR00T N1.6 behind the stock PolicyServer wire on 127.0.0.1:5565.

The policy is built exactly as ``R/gr00t/eval/run_gr00t_server.py`` builds it with
``--use-sim-policy-wrapper`` (Gr00tPolicy(strict=True) -> Gr00tSimPolicyWrapper), and the request
loop is the stock PolicyServer's (MsgSerializer, ``handler(**data)``, ``{"error": str}`` replies),
so stock clients work unchanged. What this RT-owned copy adds (PLAN.md §1.4.1, D17):

- ``options.rt.noise_seed`` seeds torch's CPU and CUDA generators before the call, so paired arms
  draw the same flow-matching noise for the same request (common random numbers). The noise is
  still fresh per call, so the policy's output distribution is unchanged.
- ``--noise cpu`` (opt-in; default ``cuda`` keeps the M1 stream): with a noise_seed, the action head's one
  ``torch.randn`` instead returns ``randn(shape, generator=torch.Generator("cpu").manual_seed(seed), float32)``
  cast to the dtype and device it asked for. That is exactly rttt/server_g1.py's draw, so a TT arm and a GPU
  arm with the same noise_seed start from the same noise (D17 across backends; the CUDA generator never can).
- ``get_action`` returns ``[action, info]`` with info = the rt echo, receive/reply stamps
  (CLOCK_MONOTONIC, comparable with the client's), infer_ms, the seed used and the backend stamp
  ``cuda:<gpu>:<repo>@<sha8>`` of the checkpoint revision transformers loaded.
- A request whose client has already given up on it (``options.rt`` t_send_ns + give_up_ms before
  the receive stamp, ``schema.request_abandoned``) gets ``{"error": "stale ..."}`` without inference:
  after an outage the requests abandoned meanwhile would otherwise each cost a full inference
  (about 60 ms) ahead of the live one.
- It binds 127.0.0.1 only, has no ``kill`` endpoint and no WBC endpoint (the WBC has its own
  process, D3).

    srvpy -m rtserve.gpu_vla_server --port 5565 --model cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate
"""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import zmq

from rtloop import schema

DEFAULT_MODEL = "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate"
NOISE_MODES = ("cuda", "cpu")


class SeededNoise:
    """While active, ``torch.randn`` draws from a CPU generator seeded with ``seed`` in float32 and casts the
    result to the dtype/device the caller asked for (the model asks for bf16 on cuda). ``calls`` counts the
    draws: the N1.6 action head makes exactly one per get_action (rttt/make_g1_golden.py asserts the same)."""

    def __init__(self, seed: int):
        import torch

        self.torch, self.calls = torch, 0
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def __enter__(self):
        torch = self.torch
        self._orig = orig = torch.randn

        def seeded_randn(*size, **kw):
            if "size" in kw:
                size = (tuple(kw.pop("size")),)
            shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
            self.calls += 1
            x = orig(tuple(int(s) for s in shape), generator=self.gen, dtype=torch.float32)
            return x.to(device=kw.get("device") or "cpu", dtype=kw.get("dtype") or torch.float32)

        torch.randn = seeded_randn
        return self

    def __exit__(self, *exc):
        self.torch.randn = self._orig
        return False


def checkpoint_id(model: str) -> tuple[str, str]:
    """(repo, revision) of a hub id or a local snapshot dir; revision 'local' if neither says."""
    p = Path(model).expanduser()
    if p.exists():
        parts = p.resolve().parts
        if "snapshots" in parts:
            i = parts.index("snapshots")
            repo = parts[i - 1].removeprefix("models--").replace("--", "/")
            return repo, parts[i + 1]
        return p.name, "local"
    from huggingface_hub import snapshot_download

    # After from_pretrained, refs/main names the revision that was actually loaded.
    return model, Path(snapshot_download(model, local_files_only=True)).name


def gpu_tag(device: str) -> str:
    """'NVIDIA GeForce RTX 5090' -> 'rtx5090'."""
    import torch

    name = torch.cuda.get_device_name(torch.device(device)).lower()
    for prefix in ("nvidia ", "geforce "):
        name = name.replace(prefix, "")
    return name.replace(" ", "")


def build_policy(model: str, device: str):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.UNITREE_G1, model_path=model,
                         device=device, strict=True)
    return Gr00tSimPolicyWrapper(policy)


def warmup_obs() -> dict:
    """A blank 9-key G1 observation: the first calls pay CUDA/cuDNN setup, not a client's request."""
    obs = {schema.VIDEO_KEY: np.zeros(schema.IMAGE_SHAPE, np.uint8), schema.LANGUAGE_KEY: "warm up"}
    for name, n in schema.STATE_GROUPS:
        obs[f"state.{name}"] = np.zeros(n, np.float32)
    return schema.build_vla_obs(obs)


class RtPolicyServer:
    """The stock PolicyServer request loop with a receive stamp taken before deserialisation."""

    def __init__(self, policy, port: int, backend: str, model: str, revision: str,
                 log_path: str | None = None, noise: str = "cuda"):
        from gr00t.policy.server_client import MsgSerializer

        self.ser = MsgSerializer
        self.policy, self.backend = policy, backend
        self.noise = noise
        self._info = {"backend": backend, "proto": 2, "model": model, "revision": revision,
                      "pid": os.getpid(), "noise": noise}
        self.endpoints = {  # name -> (handler, requires_input), as PolicyServer registers them
            "ping": (self.ping, False),
            "info": (self.info, False),
            "get_modality_config": (policy.get_modality_config, False),
            "reset": (policy.reset, True),
            "get_action": (self.get_action, True),
        }
        self.log = open(log_path, "a", buffering=1) if log_path else None
        self.t_recv_ns = 0
        self.n_stale = 0
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://127.0.0.1:{port}")
        self.port = port

    def ping(self) -> dict:
        return {"status": "ok", "message": "Server is running"}

    def info(self) -> dict:
        return {**self._info, "n_stale": self.n_stale}

    def get_action(self, observation: dict, options: dict | None = None) -> list:
        import torch

        t_recv = self.t_recv_ns
        rt = (options or {}).get("rt")
        if schema.request_abandoned(rt, t_recv):
            self.n_stale += 1
            if self.log is not None:
                self.log.write(json.dumps({**{k: rt.get(k) for k in ("seq", "gen", "ep", "k_obs")},
                                           "t_recv_ns": t_recv, "stale": True}) + "\n")
            return {"error": f"stale request seq {rt.get('seq')}: received "
                             f"{(t_recv - rt['t_send_ns']) / 1e6:.0f} ms after send, the client gave up "
                             f"after {rt['give_up_ms']} ms"}
        seed = rt.get("noise_seed") if isinstance(rt, dict) else None
        seeded = None
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if self.noise == "cpu":
                seeded = SeededNoise(seed)
        t0 = time.perf_counter_ns()
        if seeded is not None:
            with seeded:
                action, info = self.policy.get_action(observation, options)
        else:
            action, info = self.policy.get_action(observation, options)
        infer_ms = (time.perf_counter_ns() - t0) / 1e6
        info = dict(info)
        info.update(rt=rt, t_recv_ns=t_recv, t_reply_ns=time.monotonic_ns(), infer_ms=infer_ms,
                    noise_seed=seed, backend=self.backend)
        if seeded is not None:
            info.update(noise="cpu", randn_calls=seeded.calls)
        if self.log is not None:
            rec = {k: rt.get(k) for k in ("seq", "gen", "ep", "k_obs")} if isinstance(rt, dict) else {}
            rec.update(t_recv_ns=t_recv, t_reply_ns=info["t_reply_ns"], infer_ms=infer_ms,
                       noise_seed=seed)
            self.log.write(json.dumps(rec) + "\n")
        return [action, info]

    def run(self):
        print(f"gpu_vla_server ready on tcp://127.0.0.1:{self.port} {self._info}", flush=True)
        while True:
            msg = self.sock.recv()
            self.t_recv_ns = time.monotonic_ns()
            try:
                req = self.ser.from_bytes(msg)
                endpoint = req.get("endpoint", "get_action")
                if endpoint not in self.endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                handler, requires_input = self.endpoints[endpoint]
                out = handler(**(req.get("data") or {})) if requires_input else handler()
                reply = self.ser.to_bytes(out)
            except Exception as e:
                print(f"Error in server: {e}\n{traceback.format_exc()}", flush=True)
                reply = self.ser.to_bytes({"error": str(e)})
            self.sock.send(reply)

    def close(self):
        self.sock.close(linger=0)
        self.ctx.term()
        if self.log is not None:
            self.log.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=5565)
    ap.add_argument("--model", default=os.environ.get("VLA_CHECKPOINT", DEFAULT_MODEL))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=2, help="blank inference calls before serving")
    ap.add_argument("--log", default=None, help="per-call JSONL (PLAN.md §4.1 server-side log)")
    ap.add_argument("--cpus", default=None, help="CPU affinity, e.g. '12,13,14,15' (PLAN.md §1.1)")
    ap.add_argument("--noise", choices=NOISE_MODES, default="cuda",
                    help="noise_seed source: cuda = torch.cuda.manual_seed (M1 default); cpu = server_g1's CPU draw "
                         "(TT and GPU arms noise-paired, D17)")
    args = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if args.cpus:
        os.sched_setaffinity(0, {int(c) for c in args.cpus.split(",")})

    t0 = time.monotonic()
    policy = build_policy(args.model, args.device)
    repo, revision = checkpoint_id(args.model)
    backend = f"cuda:{gpu_tag(args.device)}:{repo}@{revision[:8]}"
    print(f"  loaded {repo}@{revision} on {args.device} in {time.monotonic() - t0:.1f} s", flush=True)
    for _ in range(args.warmup):
        policy.get_action(warmup_obs())
    server = RtPolicyServer(policy, args.port, backend, repo, revision, args.log, noise=args.noise)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
