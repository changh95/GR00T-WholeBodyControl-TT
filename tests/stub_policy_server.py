"""The stock gr00t PolicyServer around a real Gr00tSimPolicyWrapper(strict=True) with no model (§1.4.1).

The request test needs the *real* acceptance rules of the stock server (``run_gr00t_server.py
--use-sim-policy-wrapper``) without a GPU: the wrapper's flat ``check_observation`` and, behind it,
``Gr00tPolicy.check_observation`` on the nested observation and ``check_action`` on the reply. Only
the model is missing: ``Gr00tPolicy`` is created without ``__init__`` (which would load the
checkpoint onto a device), gets the checkpoint's own UNITREE_G1 modality configs from its
``processor_config.json`` (Hugging Face cache, no download), and its ``_get_action`` returns a
hold chunk built from the request's state. Every other line that touches the request is gr00t's:
``MsgSerializer`` decodes it, ``PolicyServer``'s endpoint table dispatches it, ``BasePolicy.get_action``
runs the strict checks.

    python tests/stub_policy_server.py --port P [--capture-dir D] [--capture-n 64]

The serve loop is PolicyServer.run's (decode, dispatch, reply ``{"error": str(e)}`` on any
exception) plus two audit outputs in D: the raw bytes of the first N get_action requests
(``<i:06d>.msgpack``) and ``log.jsonl`` with one line per get_action (ok / error, size, options.rt).
"""

import argparse
import glob
import json
import os
from pathlib import Path
import signal
import sys
import threading

import numpy as np

RT = Path(__file__).resolve().parents[1]
CHECKPOINT = "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate"
EMBODIMENT = "unitree_g1"


def processor_config_path(repo: str = CHECKPOINT) -> Path | None:
    """The checkpoint's processor_config.json in the local Hugging Face cache, or None."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or
                                                       Path.home() / ".cache/huggingface") / "hub")
    pattern = str(hub / f"models--{repo.replace('/', '--')}" / "snapshots" / "*" / "processor_config.json")
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def modality_configs(path: Path | None = None) -> dict:
    """{modality: gr00t ModalityConfig} for UNITREE_G1, exactly as the checkpoint's processor has them."""
    from gr00t.data.types import ModalityConfig

    path = path or processor_config_path()
    if path is None:
        raise FileNotFoundError(f"{CHECKPOINT} processor_config.json is not in the HF cache")
    cfg = json.loads(Path(path).read_text())["processor_kwargs"]["modality_configs"][EMBODIMENT]
    return {m: ModalityConfig(**c) for m, c in cfg.items()}


def hold_action(new_obs: dict, horizon: int) -> dict:
    """Nested Gr00tPolicy action (B=1, T=horizon, D) float32 that holds the request's upper body."""
    from rtloop.schema import ACTION_GROUPS, ACTION_SLICES, H, hold_row

    state = {f"state.{k}": np.asarray(v, dtype=np.float32).reshape(-1)
             for k, v in new_obs["state"].items()}
    row = hold_row(state)
    rows = np.repeat(row[None], max(horizon, H), axis=0)
    return {name: rows[None, :horizon, ACTION_SLICES[name]].copy() for name, _ in ACTION_GROUPS}


def make_wrapper(mc: dict | None = None, on_call=None):
    """Gr00tSimPolicyWrapper(strict=True) around a model-free Gr00tPolicy (see the module doc).

    ``on_call(new_obs, options)`` sees the nested observation that passed Gr00tPolicy's checks."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.policy import BasePolicy

    mc = mc or modality_configs()
    policy = Gr00tPolicy.__new__(Gr00tPolicy)  # no __init__: it would load the checkpoint on a device
    BasePolicy.__init__(policy, strict=True)
    policy.embodiment_tag = EmbodimentTag.UNITREE_G1
    policy.modality_configs = mc
    policy.language_key = mc["language"].modality_keys[0]
    horizon = len(mc["action"].delta_indices)

    def _get_action(observation, options=None):
        if on_call is not None:
            on_call(observation, options)
        return hold_action(observation, horizon), {}

    policy._get_action = _get_action  # instance attribute: BasePolicy.get_action calls it after the checks
    return Gr00tSimPolicyWrapper(policy, strict=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--capture-dir", default=None)
    ap.add_argument("--capture-n", type=int, default=64)
    a = ap.parse_args(argv)
    if 5555 <= a.port <= 5572:
        ap.error("5555-5572 are reserved (PLAN.md §2; the existing GR00T server owns 5555)")
    import zmq

    from gr00t.policy.server_client import MsgSerializer, PolicyServer

    cap = Path(a.capture_dir) if a.capture_dir else None
    if cap is not None:
        cap.mkdir(parents=True, exist_ok=True)
    log = open(cap / "log.jsonl", "a") if cap is not None else None
    server = PolicyServer(make_wrapper(), host="127.0.0.1", port=a.port)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    print(f"READY mode=stub-gr00t-sim-wrapper url=tcp://127.0.0.1:{a.port} pid={os.getpid()}", flush=True)
    sock, n_get = server.socket, 0
    while not stop.is_set():
        if not sock.poll(100, zmq.POLLIN):
            continue
        message = sock.recv()
        entry = None
        try:
            request = MsgSerializer.from_bytes(message)
            endpoint = request.get("endpoint", "get_action")
            if endpoint == "get_action":
                if cap is not None and n_get < a.capture_n:
                    (cap / f"{n_get:06d}.msgpack").write_bytes(message)
                opts = (request.get("data") or {}).get("options") or {}
                entry = {"i": n_get, "bytes": len(message), "rt": opts.get("rt"), "ok": False}
                n_get += 1
            if endpoint not in server._endpoints:
                raise ValueError(f"Unknown endpoint: {endpoint}")
            handler = server._endpoints[endpoint]
            result = handler.handler(**request.get("data", {})) if handler.requires_input else handler.handler()
            reply = MsgSerializer.to_bytes(result)
            if entry is not None:
                entry["ok"] = True
        except Exception as e:  # noqa: BLE001 - as PolicyServer.run: the error goes back to the client
            if entry is not None:
                entry["error"] = f"{type(e).__name__}: {e}"
            reply = MsgSerializer.to_bytes({"error": str(e)})
        sock.send(reply)
        if entry is not None and log is not None:
            log.write(json.dumps(entry) + "\n")
            log.flush()
    sock.close(linger=0)
    if log is not None:
        log.close()


if __name__ == "__main__":
    sys.path.insert(0, str(RT))
    main()
