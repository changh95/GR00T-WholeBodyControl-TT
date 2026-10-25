"""Mock W-VLA server speaking the stock PolicyServer wire (INTERFACES.md WP-1.3, PLAN.md §1.4.1).

It stands in for the TT VLA in unit, equivalence, alignment and fault tests. Only numpy, zmq,
msgpack and rtloop.schema/wire are imported (no gr00t, no torch), so it starts in well under 1 s
and runs under either venv.

Modes:
  hold          every row holds the request's arms/hands/waist, height 0.74, nav 0
  detfn         deterministic function of the request's state bytes and options.rt.k_obs only:
                arms = state + 0.05 sin(2 pi i/30 + phi(state bytes)); nav [0.2,0,0] on rows where
                ((k_obs + i) // 200) is odd (exercises Walk); hands/waist held
  ramp          arms row i = state + float32(i) * float32(1e-3), all in float32 (T-1.2b); nav 0
  record-proxy  forwards every request to --upstream unchanged and returns its reply unchanged;
                get_action requests are saved as <record-dir>/<i:06d>.npz + index.jsonl
  replay        replies with the recorded replies in order; a request whose state arrays differ
                from the recording by any bit (image: sha1 of its bytes) gets an error reply

Run: `simpy -m rtserve.mock_vla_server --port 5559 --mode detfn` (``--port 0`` picks a free port
and prints it on the READY line). Tests import ``start_thread`` / ``spawn``.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import zmq

from rtloop import wire
from rtloop.schema import (ARMS, H, LANGUAGE_KEY, NAV, STATE_KEYS, VIDEO_KEY, hold_row,
                           request_abandoned, rows_to_actions)

MODES = ("hold", "detfn", "ramp", "record-proxy", "replay")
PROTO = 2
RAMP_DELTA = np.float32(1e-3)  # rad per row, f32 so the alignment test can reproduce it bit-exactly
DETFN_AMP = 0.05  # rad
DETFN_NAV = 0.2  # m/s forward, above WALK_THRESHOLD so the WBC switches to Walk
DETFN_NAV_PERIOD = 200  # ticks
INDEX = "index.jsonl"
MODALITY_RAW = "modality_config.msgpack"


# --- synthetic chunks ---------------------------------------------------------------------------


def check_obs(obs) -> None:
    """The dtype/rank checks Gr00tSimPolicyWrapper applies (strict=True), so loop bugs that the
    real server would reject also fail against the mock."""
    if not isinstance(obs, dict):
        raise ValueError(f"observation must be a dict, got {type(obs).__name__}")
    for k in STATE_KEYS:
        a = obs.get(k)
        if not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.ndim != 3 \
                or a.shape[:2] != (1, 1):
            raise ValueError(f"{k} must be float32 (1, 1, D), got "
                             f"{getattr(a, 'dtype', type(a).__name__)} {getattr(a, 'shape', '')}")
    v = obs.get(VIDEO_KEY)
    if not isinstance(v, np.ndarray) or v.dtype != np.uint8 or v.ndim != 5:
        raise ValueError(f"{VIDEO_KEY} must be uint8 (1, 1, H, W, C), got "
                         f"{getattr(v, 'dtype', type(v).__name__)} {getattr(v, 'shape', '')}")
    lang = obs.get(LANGUAGE_KEY)
    if not isinstance(lang, (list, tuple)) or len(lang) != 1 or not isinstance(lang[0], str):
        raise ValueError(f"{LANGUAGE_KEY} must be a 1-list of str, got {lang!r}")


def _hold(obs: dict) -> np.ndarray:
    return hold_row({k: np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in STATE_KEYS})


def state_phase(obs: dict) -> float:
    """phi in [0, 2 pi) from the sha1 of the state bytes (STATE_KEYS order)."""
    h = hashlib.sha1()
    for k in STATE_KEYS:
        h.update(np.ascontiguousarray(obs[k]).tobytes())
    return int.from_bytes(h.digest()[:8], "little") / 2.0**64 * 2.0 * np.pi


def rows_hold(obs: dict, k_obs: int = 0) -> np.ndarray:
    return np.repeat(_hold(obs)[None], H, axis=0)


def rows_ramp(obs: dict, k_obs: int = 0) -> np.ndarray:
    hold = _hold(obs)
    rows = np.repeat(hold[None], H, axis=0)
    rows[:, ARMS] = hold[ARMS][None, :] + np.arange(H, dtype=np.float32)[:, None] * RAMP_DELTA
    return rows


def rows_detfn(obs: dict, k_obs: int = 0) -> np.ndarray:
    hold = _hold(obs)
    rows = np.repeat(hold[None], H, axis=0)
    i = np.arange(H)
    offs = (DETFN_AMP * np.sin(2.0 * np.pi * i / H + state_phase(obs))).astype(np.float32)
    rows[:, ARMS] = hold[ARMS][None, :] + offs[:, None]
    walk = ((int(k_obs) + i) // DETFN_NAV_PERIOD) % 2 == 1
    rows[walk, NAV[0]] = np.float32(DETFN_NAV)
    return rows


SYNTH = {"hold": rows_hold, "detfn": rows_detfn, "ramp": rows_ramp}


# --- helpers ------------------------------------------------------------------------------------


def _sha1(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _lang(obs: dict):
    lang = obs.get(LANGUAGE_KEY)
    return [str(x) for x in lang] if isinstance(lang, (list, tuple)) else lang


# --- server -------------------------------------------------------------------------------------


class MockVlaServer:
    """A REP socket plus per-mode state. Bound in __init__ so the port is known before serving;
    the socket may then be handed to one serving thread (thread start is a full barrier)."""

    def __init__(self, mode: str, port: int = 5559, host: str = "127.0.0.1", *,
                 latency_ms: float = 0.0, jitter_ms: float = 0.0, stall_after: int = -1,
                 stall_s: float = 0.0, seed: int = 0, upstream: str | None = None,
                 upstream_timeout_ms: int = 30000, record_dir: str | None = None,
                 replay_dir: str | None = None, ctx: zmq.Context | None = None,
                 verbose: bool = False):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        self.mode = mode
        self.latency_ms, self.jitter_ms = float(latency_ms), float(jitter_ms)
        self.stall_after, self.stall_s = int(stall_after), float(stall_s)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.n_get = 0  # get_action requests served (the stall counter, 0-based)
        self.n_stale = 0  # abandoned by the client before they were taken (answered at once, not served)
        self._stop: threading.Event | None = None
        self._own_ctx = ctx is None
        self.ctx = zmq.Context() if ctx is None else ctx
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        if port == 0:
            self.port = self.sock.bind_to_random_port(f"tcp://{host}")  # 49152-65535
        else:
            self.sock.bind(f"tcp://{host}:{port}")
            self.port = port
        self.url = f"tcp://{host}:{self.port}"

        self.upstream, self.up = upstream, None
        self.upstream_timeout_ms = upstream_timeout_ms
        if mode == "record-proxy":
            if not upstream or not record_dir:
                raise ValueError("record-proxy needs --upstream and --record-dir")
            self.record_dir = Path(record_dir)
            self.record_dir.mkdir(parents=True, exist_ok=True)
            if (self.record_dir / INDEX).exists():
                raise FileExistsError(f"{self.record_dir / INDEX} exists; use a fresh --record-dir")
            self._index = open(self.record_dir / INDEX, "a")
            self._connect_upstream()
        self.n_mismatch = 0
        if mode == "replay":
            if not replay_dir:
                raise ValueError("replay needs --replay-dir")
            self.replay_dir = Path(replay_dir)
            with open(self.replay_dir / INDEX) as f:
                self.entries = sorted((json.loads(line) for line in f if line.strip()),
                                      key=lambda e: e["i"])
            self.replay_pos = 0
            p = self.replay_dir / MODALITY_RAW
            self.modality_raw = p.read_bytes() if p.exists() else None

    # --- upstream (record-proxy) -------------------------------------------------------------

    def _connect_upstream(self) -> None:
        self.up = self.ctx.socket(zmq.REQ)
        self.up.setsockopt(zmq.LINGER, 0)
        self.up.connect(self.upstream)

    def _forward(self, raw: bytes) -> bytes:
        self.up.send(raw)
        if not self.up.poll(self.upstream_timeout_ms, zmq.POLLIN):
            self.up.close(linger=0)
            self._connect_upstream()
            raise TimeoutError(f"upstream {self.upstream}: "
                               f"no reply in {self.upstream_timeout_ms} ms")
        return self.up.recv()

    # --- timing ------------------------------------------------------------------------------

    def _delay_ns(self, idx: int) -> int:
        ms = self.latency_ms
        if self.jitter_ms:
            ms += float(self.rng.uniform(-self.jitter_ms, self.jitter_ms))
        if idx == self.stall_after:
            ms += self.stall_s * 1e3
        return max(0, int(ms * 1e6))

    def _sleep_until(self, t_ns: int) -> bool:
        """False if stop was requested meanwhile (the reply is then dropped)."""
        while True:
            rem = t_ns - time.monotonic_ns()
            if rem <= 0:
                return True
            if self._stop is not None and self._stop.is_set():
                return False
            time.sleep(min(rem / 1e9, 0.05))

    # --- endpoints ---------------------------------------------------------------------------

    def info(self) -> dict:
        """Backend stamp plus every setting that shapes the replies, so a run's config.json can
        reproduce a mock-backed run (latency, jitter, seed, stall)."""
        out = {"backend": f"mock:{self.mode}", "proto": PROTO, "n_get_action": self.n_get,
               "latency_ms": self.latency_ms, "jitter_ms": self.jitter_ms, "seed": self.seed,
               "stall_after": self.stall_after, "stall_s": self.stall_s, "n_stale": self.n_stale,
               "pid": os.getpid()}
        if self.mode == "record-proxy":
            out.update(upstream=self.upstream, record_dir=str(self.record_dir))
        if self.mode == "replay":
            out.update(replay_dir=str(self.replay_dir), n_recorded=len(self.entries),
                       replay_pos=self.replay_pos, n_mismatch=self.n_mismatch)
        return out

    def handle(self, raw: bytes) -> bytes | None:
        """One request -> reply bytes (after the configured latency); None if stopped meanwhile."""
        t_recv = time.monotonic_ns()
        try:
            req = wire.loads(raw)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}
            if endpoint == "get_action":
                return self._get_action(raw, data, t_recv)
            if endpoint == "ping":
                out = {"status": "ok", "message": "Server is running"}
            elif endpoint == "info":
                out = self.info()
            elif endpoint in ("get_modality_config", "reset") and self.mode == "record-proxy":
                up = self._forward(raw)
                if endpoint == "get_modality_config":
                    (self.record_dir / MODALITY_RAW).write_bytes(up)
                return up
            elif endpoint == "get_modality_config":
                if self.mode == "replay" and self.modality_raw is not None:
                    return self.modality_raw
                out = {}
            elif endpoint == "reset":
                out = {}
            else:
                out = {"error": f"Unknown endpoint: {endpoint}"}
        except Exception as e:  # noqa: BLE001 - the stock server replies {"error"} on anything
            out = {"error": f"{type(e).__name__}: {e}"}
        return wire.dumps(out)

    def _get_action(self, raw: bytes, data: dict, t_recv: int) -> bytes | None:
        obs = data.get("observation")
        options = data.get("options") or {}
        rt = options.get("rt") if isinstance(options, dict) else None
        if self.mode in SYNTH and request_abandoned(rt, t_recv):  # as server_g1: no service time
            self.n_stale += 1
            return wire.dumps({"error": f"stale request seq {rt.get('seq')} (abandoned by the client)"})
        idx = self.n_get
        self.n_get += 1
        deadline = t_recv + self._delay_ns(idx)
        try:
            if self.mode in SYNTH:
                check_obs(obs)
                k_obs = int(rt.get("k_obs", 0)) if isinstance(rt, dict) else 0
                action = rows_to_actions(SYNTH[self.mode](obs, k_obs))
                reply = None
            elif self.mode == "record-proxy":
                up = self._forward(raw)
                self._record(idx, obs, options, up, (time.monotonic_ns() - t_recv) / 1e6)
                return up if self._sleep_until(deadline) else None
            else:
                action, reply = self._replay(obs)
        except Exception as e:  # noqa: BLE001
            reply = {"error": f"{type(e).__name__}: {e}"}
        if not self._sleep_until(deadline):
            return None
        if reply is not None:
            return wire.dumps(reply)
        t_reply = time.monotonic_ns()
        info = {"rt": rt, "t_recv_ns": t_recv, "t_reply_ns": t_reply,
                "infer_ms": (t_reply - t_recv) / 1e6, "backend": f"mock:{self.mode}"}
        if self.mode == "replay":
            info["replay_index"] = self.replay_pos - 1
        return wire.dumps([action, info])

    # --- record / replay ---------------------------------------------------------------------

    def _record(self, i: int, obs: dict, options, up: bytes, upstream_ms: float) -> None:
        arrays, entry = {}, {"i": i, "file": f"{i:06d}.npz", "upstream_ms": upstream_ms}
        obs = obs if isinstance(obs, dict) else {}
        state_keys = sorted(k for k in obs if k.startswith("state."))
        for k in state_keys:
            arrays[k] = np.asarray(obs[k])
        img = obs.get(VIDEO_KEY)
        if isinstance(img, np.ndarray):
            entry.update(image_sha1=_sha1(img), image_shape=list(img.shape),
                         image_dtype=str(img.dtype))
        else:
            entry.update(image_sha1=None)
        entry.update(state_keys=state_keys, language=_lang(obs), options=options,
                     rt=options.get("rt") if isinstance(options, dict) else None,
                     reply_sha1=hashlib.sha1(up).hexdigest())
        rep = wire.loads(up)
        if isinstance(rep, (list, tuple)) and len(rep) == 2 and isinstance(rep[0], dict):
            for k, v in rep[0].items():
                arrays[k] = np.asarray(v)
            entry.update(status="ok", action_keys=sorted(rep[0]),
                         info=rep[1] if isinstance(rep[1], dict) else {})
        else:
            err = rep.get("error") if isinstance(rep, dict) else f"unexpected {type(rep).__name__}"
            entry.update(status="error", error=str(err), action_keys=[], info={})
        np.savez(self.record_dir / entry["file"], **arrays)
        self._index.write(json.dumps(entry, default=_jsonable) + "\n")
        self._index.flush()

    def _mismatch(self, obs, e: dict, z) -> str | None:
        if not isinstance(obs, dict):
            return "observation is not a dict"
        for k in e["state_keys"]:
            if k not in obs:
                return f"{k} missing"
            a, r = np.asarray(obs[k]), z[k]
            if a.dtype != r.dtype or a.shape != r.shape:
                return f"{k}: {a.dtype}{a.shape} vs recorded {r.dtype}{r.shape}"
            if a.tobytes() != r.tobytes():
                d = np.abs(a.astype(np.float64) - r.astype(np.float64)).max()
                return f"{k}: bits differ (max |d| {d:.3g})"
        if e.get("image_sha1") is not None:
            img = obs.get(VIDEO_KEY)
            if not isinstance(img, np.ndarray):
                return f"{VIDEO_KEY} missing"
            if list(img.shape) != e["image_shape"] or _sha1(img) != e["image_sha1"]:
                return f"{VIDEO_KEY}: sha1 differs"
        if _lang(obs) != e.get("language"):
            return f"{LANGUAGE_KEY}: {_lang(obs)!r} vs recorded {e.get('language')!r}"
        return None

    def _replay(self, obs) -> tuple[dict | None, dict | None]:
        if self.replay_pos >= len(self.entries):
            return None, {"error": f"replay exhausted: {len(self.entries)} recorded replies"}
        e = self.entries[self.replay_pos]
        with np.load(self.replay_dir / e["file"], allow_pickle=False) as z:
            why = self._mismatch(obs, e, z)
            if why is not None:
                self.n_mismatch += 1
                return None, {"error": f"replay state mismatch at request {self.replay_pos} "
                                       f"(recorded i={e['i']}): {why}"}
            self.replay_pos += 1
            if e["status"] != "ok":
                return None, {"error": e.get("error", "recorded error")}
            return {k: z[k] for k in e["action_keys"]}, None

    # --- loop --------------------------------------------------------------------------------

    def serve_forever(self, stop: threading.Event | None = None) -> None:
        self._stop = stop
        while stop is None or not stop.is_set():
            if not self.sock.poll(50, zmq.POLLIN):
                continue
            raw = self.sock.recv()
            reply = self.handle(raw)
            if reply is None:
                break
            self.sock.send(reply)
            if self.verbose:
                print(f"[mock:{self.mode}] {len(raw)} B -> {len(reply)} B", flush=True)

    def close(self) -> None:
        self.sock.close(linger=0)
        if self.up is not None:
            self.up.close(linger=0)
        if self.mode == "record-proxy":
            self._index.close()
        if self._own_ctx:
            self.ctx.term()


def serve(port: int = 5559, mode: str = "hold", *, stop_event: threading.Event | None = None,
          ready=None, **kw) -> None:
    """Run a mock until ``stop_event`` is set. ``ready(server)`` is called once bound."""
    srv = MockVlaServer(mode, port, **kw)
    try:
        if ready is not None:
            ready(srv)
        srv.serve_forever(stop_event)
    finally:
        srv.close()


class ThreadedMock:
    """A mock serving from a daemon thread on a random free port (tests)."""

    def __init__(self, mode: str = "hold", port: int = 0, **kw):
        self.server = MockVlaServer(mode, port, **kw)
        self.url, self.port = self.server.url, self.server.port
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"mock-vla-{mode}")
        self.thread.start()

    def _run(self) -> None:
        try:
            self.server.serve_forever(self.stop_event)
        finally:
            self.server.close()

    def stop(self, timeout_s: float = 5.0) -> None:
        self.stop_event.set()
        self.thread.join(timeout_s)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def start_thread(mode: str = "hold", port: int = 0, **kw) -> ThreadedMock:
    return ThreadedMock(mode, port, **kw)


def spawn(mode: str, *args: str, python: str | None = None,
          timeout_s: float = 10.0) -> tuple[subprocess.Popen, str]:
    """Start the CLI in a subprocess on a free port; returns (process, url) once it is READY.

    Only the READY line is read from stdout afterwards, so do not pass --verbose (a full pipe
    would block the server)."""
    rt_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = rt_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [python or sys.executable, "-m", "rtserve.mock_vla_server", "--port", "0",
           "--mode", mode, *args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         cwd=rt_root, env=env)
    t_end = time.monotonic() + timeout_s
    while (rem := t_end - time.monotonic()) > 0:
        if not select.select([p.stdout], [], [], rem)[0]:
            break
        line = p.stdout.readline()
        if line.startswith("READY"):
            url = dict(kv.split("=", 1) for kv in line.split()[1:])["url"]
            return p, url
        if not line:  # EOF: the process died before READY
            break
    p.kill()
    out, err = p.communicate(timeout=5)
    raise RuntimeError(f"mock_vla_server did not start: {err or out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=5559, help="0 = a free port (printed on READY)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mode", choices=MODES, default="hold")
    ap.add_argument("--latency-ms", type=float, default=0.0,
                    help="get_action service time, measured from request receipt")
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="uniform +-J added per request")
    ap.add_argument("--stall-after", type=int, default=-1,
                    help="get_action requests 0..N-1 are normal; request N (0-based) stalls")
    ap.add_argument("--stall-s", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="jitter RNG seed")
    ap.add_argument("--upstream", help="record-proxy: upstream URL, e.g. tcp://127.0.0.1:5557")
    ap.add_argument("--upstream-timeout-ms", type=int, default=30000)
    ap.add_argument("--record-dir")
    ap.add_argument("--replay-dir")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.port == 5555:
        ap.error("5555 belongs to the existing GR00T server")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def ready(srv):
        print(f"READY mode={srv.mode} url={srv.url} pid={os.getpid()}", flush=True)

    serve(a.port, a.mode, stop_event=stop, ready=ready, host=a.host, latency_ms=a.latency_ms,
          jitter_ms=a.jitter_ms, stall_after=a.stall_after, stall_s=a.stall_s, seed=a.seed,
          upstream=a.upstream, upstream_timeout_ms=a.upstream_timeout_ms,
          record_dir=a.record_dir, replay_dir=a.replay_dir, verbose=a.verbose)


if __name__ == "__main__":
    main()
