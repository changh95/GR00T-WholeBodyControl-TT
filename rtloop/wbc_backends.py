"""WBC backends behind the D5 seam: in-process ORT, the W-WBC REQ client, and the class patch.

The lower-body policy stays the verified gr00t_wbc Python (obs build, 6x86 history, Balance/Walk
switch); only its stateless 516 -> 15 MLP call is replaced (PLAN.md D2, D5). That call goes to a
process-wide ``WbcRuntime``: a primary backend (``OrtCpu`` or ``ZmqWbc``), an optional shadow
whose max |a - a_shadow| is logged every tick (D16: ORT behind a remote primary; T-3.6: a remote
W-WBC behind the ORT primary, ``--shadow URL``), and a per-call ``WbcCall`` record the loop reads
back after ``env.step``. ``ZmqWbc`` never blocks a tick for longer than its timeout: a missed reply
is replaced by ORT on the same call (stateless, so safe at any tick), the REQ socket is re-created
so a late reply can never be taken for a later request (lazy pirate), and three misses in a row
pin the rest of the episode to ORT. A shadow client has no fallback: a missed shadow reply is a
NaN shadow_da (counted), never an ORT-vs-ORT zero that would read as perfect parity.
"""

from dataclasses import dataclass
import hashlib
import itertools
import os
from pathlib import Path
import select
import time

import numpy as np
import zmq

from rtloop import wire
from rtloop.schema import WBC_ACT_DIM, WBC_OBS_DIM, WBC_POLICIES, WbcBackend

ONNX_DIR = os.environ.get("ONNX") or str(
    Path.home() / "experiments/gr00t/Isaac-GR00T-n1.6/external_dependencies/GR00T-WholeBodyControl"
    / "gr00t_wbc/sim2mujoco/resources/robots/g1/policy"
)
NAN = float("nan")


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- In-process onnxruntime ---------------------------------------------------------------------


class OrtCpu:
    """onnxruntime on the CPU provider, one session per policy stem: fallback, shadow, and --wbc ort.

    ``opts="default"`` builds each session exactly as g1_gear_wbc_policy.py does
    (``ort.InferenceSession(path)``), so equivalence tests can be bit-exact against the official
    stack. ``opts="single"`` uses one intra-op and one inter-op thread, so a call never wakes a
    thread pool that would compete with the tick thread.
    """

    backend = WbcBackend.ORT

    def __init__(self, onnx_dir: str = ONNX_DIR, opts: str = "single"):
        import onnxruntime as ort

        if onnx_dir in ("single", "default") and not Path(onnx_dir).is_dir():
            onnx_dir, opts = ONNX_DIR, onnx_dir  # OrtCpu("default") reads as the options name
        if opts not in ("single", "default"):
            raise ValueError(f"opts must be 'single' or 'default', got {opts!r}")
        self.onnx_dir, self.opts = str(onnx_dir), opts
        paths = sorted(Path(onnx_dir).glob("*.onnx"))
        if not paths:
            raise FileNotFoundError(f"No .onnx policies in {onnx_dir}")
        self.sessions, self._input, self.sha256 = {}, {}, {}
        for p in paths:
            if opts == "default":
                sess = ort.InferenceSession(str(p))
            else:
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                sess = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
            if sess.get_providers() != ["CPUExecutionProvider"]:
                raise RuntimeError(f"{p.name}: providers {sess.get_providers()}, want CPU only")
            self.sessions[p.stem] = sess
            self._input[p.stem] = sess.get_inputs()[0].name
            self.sha256[p.stem] = _sha256(str(p))
        self.policies = tuple(self.sessions)

    def __call__(self, policy: str, obs: np.ndarray) -> np.ndarray:
        """obs float32 (1, 516) -> raw action float32 (1, 15)."""
        return self.sessions[policy].run(None, {self._input[policy]: obs})[0]


# --- W-WBC REQ client ---------------------------------------------------------------------------


def _backend_of(info: dict) -> WbcBackend:
    name = str(info.get("backend", ""))
    if name.startswith("x280-sim"):
        return WbcBackend.X280_SIM
    if name.startswith("x280"):
        return WbcBackend.X280
    return WbcBackend.GPU  # cuda-graph:* and the stock PolicyServer's gear_wbc_action


class ZmqWbc:
    """W-WBC client (PLAN.md §1.4.3): REQ with an ORT fallback.

    The socket carries RCVTIMEO/SNDTIMEO = timeout_ms, but a tick call waits against a deadline of
    timeout_ms from its own start (see ``_recv_by``), so a miss costs timeout_ms plus the
    fallback, not up to a millisecond more. At construction it asks the server for ``info`` (a stock PolicyServer has none: proto 1, so
    tick/seq are never sent to it, as its handler takes only (policy, obs)) and checks the served
    policies against ORT on random inputs, so a backend that is merely reachable but wrong
    never gets to drive the legs.

    The parity tolerance is the server's own ``info.parity_tol`` when it reports one (x280d: 1e-2 for
    fp16 weights, 5e-5 for fp32; T-3.1), never looser than ``MAX_PARITY_TOL``, else ``PARITY_TOL``.
    ``fallback=None`` (a shadow, T-3.6) returns ``(None, meta)`` on a miss instead of an ORT action;
    ``reference`` is then the ORT used for the parity check.
    """

    CONNECT_TIMEOUT_MS = 2000   # info, parity check and ping; never on the tick path
    RESET_PING_MS = 200         # reset_episode's retry; the reset window is about 2.5 s
    MAX_FAILURES = 3            # consecutive misses before the episode is latched to ORT
    N_PARITY = 8
    PARITY_TOL = 1e-4           # a server that reports no parity_tol (GPU: fp32 weights, T-1.1 <= 1.6e-5)
    MAX_PARITY_TOL = 1e-2       # the loosest a server may claim: T-3.1's fp16 bound

    def __init__(self, url: str, fallback: OrtCpu | None, timeout_ms: int = 5, latch: bool = True,
                 proto: int | None = None, ctx: zmq.Context | None = None, reference: OrtCpu | None = None):
        self.url, self.fallback = url, fallback
        self.reference = reference if reference is not None else fallback
        if self.reference is None:
            raise ValueError("ZmqWbc needs an ORT fallback or a reference for its parity check")
        self.timeout_ms, self.latch = int(timeout_ms), bool(latch)
        self._own_ctx = ctx is None
        self.ctx = zmq.Context() if ctx is None else ctx
        self.sock = None
        self.latched = False
        self.consecutive = 0
        self.n_calls = self.n_fallbacks = self.n_timeouts = self.n_errors = 0
        self.n_latches = self.n_recreate = 0
        self.last_error: str | None = None
        self._connect()
        try:
            self.server_info = self._query_info()
            self.proto = int(proto if proto is not None else self.server_info.get("proto", 1))
            self.backend = _backend_of(self.server_info)
            self.policies = tuple(self.server_info.get("policies", WBC_POLICIES))
            missing = [p for p in WBC_POLICIES if p not in self.policies]
            if missing:
                raise RuntimeError(f"{url} does not serve {missing}")
            self.parity_tol = self._parity_tol(self.server_info)
            self.parity_max_err = self._parity_check()
        except BaseException:
            self.close()
            raise

    # -- socket handling

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(self.url)
        self._fd = self.sock.getsockopt(zmq.FD)
        # poll(), not select(): select() raises ValueError for fds >= FD_SETSIZE (1024), which a
        # process with many open files reaches (RLIMIT_NOFILE here is 1048576).
        self._poller = select.poll()
        self._poller.register(self._fd, select.POLLIN)

    def _recv_by(self, deadline_ns: int) -> bytes | None:
        """The reply, or None at the deadline.

        libzmq's RCVTIMEO runs on a cached clock with about 1 ms resolution (a 5 ms timeout
        returned after 4.2-5.9 ms here), too coarse for a 5 ms budget inside a 20 ms tick. Waiting
        on the socket's FD (edge-triggered: re-check EVENTS after every wake-up) bounds the whole
        call by the deadline instead: poll() sleeps whole milliseconds that end before the
        deadline, and the last sub-millisecond is a busy re-check of EVENTS (timeouts only).
        """
        while True:
            if self.sock.getsockopt(zmq.EVENTS) & zmq.POLLIN:
                return self.sock.recv(zmq.NOBLOCK)
            remaining = deadline_ns - time.perf_counter_ns()
            if remaining <= 0:
                return None
            ms = remaining // 1_000_000
            if ms:
                self._poller.poll(ms)

    def _recreate(self):
        """Lazy pirate: a REQ that missed its reply is unusable, and the reply may still arrive."""
        self.sock.close(linger=0)
        self._connect()
        self.n_recreate += 1

    def _rpc(self, endpoint: str, data: dict | None, timeout_ms: int):
        """Blocking call with a one-off timeout (setup paths only); TimeoutError re-creates the socket."""
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            self.sock.send(wire.request(endpoint, data))
            return wire.loads(self.sock.recv())
        except zmq.Again:
            self._recreate()
            raise TimeoutError(f"{self.url}: no reply to {endpoint!r} within {timeout_ms} ms")
        except zmq.ZMQError:
            self._recreate()
            raise
        finally:
            self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

    def _query_info(self) -> dict:
        rep = self._rpc("info", None, self.CONNECT_TIMEOUT_MS)
        if isinstance(rep, dict) and "error" not in rep:
            return dict(rep)
        # The stock PolicyServer answers an unknown endpoint with {"error": ...}.
        return {"backend": "policy-server", "proto": 1}

    @classmethod
    def _parity_tol(cls, info: dict) -> float:
        """The server's declared tolerance (finite, > 0), capped at MAX_PARITY_TOL; PARITY_TOL if none."""
        tol = info.get("parity_tol")
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not np.isfinite(tol) or tol <= 0:
            return cls.PARITY_TOL
        return float(min(tol, cls.MAX_PARITY_TOL))

    def _parity_check(self) -> float:
        rng = np.random.default_rng(0)
        worst = 0.0
        for policy in WBC_POLICIES:
            for _ in range(self.N_PARITY):
                obs = rng.standard_normal((1, WBC_OBS_DIM)).astype(np.float32)
                rep = self._rpc("gear_wbc_action", {"policy": policy, "obs": obs},
                                self.CONNECT_TIMEOUT_MS)
                if not isinstance(rep, dict) or "error" in rep:
                    raise RuntimeError(f"{self.url} {policy}: parity call failed: {rep}")
                got = np.asarray(rep["action"], dtype=np.float32)
                err = float(np.max(np.abs(got - self.reference(policy, obs))))
                worst = max(worst, err)
                if not err < self.parity_tol:
                    raise RuntimeError(f"{self.url} {policy}: served policy mismatches ORT, "
                                       f"max |err| = {err:.3e} (tol {self.parity_tol:g})")
        return worst

    # -- per tick

    def __call__(self, policy: str, obs: np.ndarray, tick: int, seq: int) -> tuple[np.ndarray, dict]:
        """One W-WBC call: (action float32 (1, 15), meta {rtt_us, dev_us, fallback, backend, error}).

        rtt_us is the whole call as the tick sees it (a timeout's wait included); dev_us is the
        server's compute_us (NaN when the server does not report it), or the ORT time on fallback.
        """
        t0 = time.perf_counter_ns()
        self.n_calls += 1
        if self.latched:
            return self._fallback(policy, obs, t0, "latched")
        data = {"policy": policy, "obs": obs}
        if self.proto >= 2:
            data["tick"], data["seq"] = int(tick), int(seq)
        error = msg = None
        try:
            # A connected REQ queues the request at once (the pipe exists even with no peer).
            self.sock.send(wire.request("gear_wbc_action", data), zmq.NOBLOCK)
            msg = self._recv_by(t0 + self.timeout_ms * 1_000_000)
            if msg is None:
                raise zmq.Again()
        except zmq.Again:
            error = "timeout"
            self.n_timeouts += 1
        except zmq.ZMQError as e:
            error = f"zmq: {e}"
        except Exception as e:  # noqa: BLE001 - an I/O failure must fall back, not stop the tick
            error = f"io: {type(e).__name__}: {e}"
        if error is None:
            try:
                rep = wire.loads(msg)
            except Exception as e:  # undecodable reply: never let it reach env.step
                error = f"decode: {type(e).__name__}: {e}"
        if error is None:
            if not isinstance(rep, dict) or "error" in rep:
                error = f"server: {rep.get('error') if isinstance(rep, dict) else type(rep).__name__}"
            else:
                action = rep.get("action")
                if not (isinstance(action, np.ndarray) and action.shape == (1, WBC_ACT_DIM)
                        and action.dtype == np.float32):
                    error = f"bad action {getattr(action, 'shape', None)} {getattr(action, 'dtype', None)}"
                elif self.proto >= 2 and (rep.get("tick") != data["tick"] or rep.get("seq") != data["seq"]):
                    error = f"stale reply tick={rep.get('tick')} seq={rep.get('seq')}"
        if error is None:
            self.consecutive = 0
            dev = rep.get("compute_us")
            return action, {"rtt_us": (time.perf_counter_ns() - t0) / 1e3,
                            "dev_us": NAN if dev is None else float(dev),
                            "fallback": False, "backend": self.backend, "error": None}
        if error != "timeout":
            self.n_errors += 1
        self._recreate()
        self.consecutive += 1
        if self.latch and self.consecutive >= self.MAX_FAILURES:
            self.latched = True
            self.n_latches += 1
        return self._fallback(policy, obs, t0, error)

    def _fallback(self, policy: str, obs: np.ndarray, t0: int, error: str) -> tuple[np.ndarray | None, dict]:
        t1 = time.perf_counter_ns()
        action = self.fallback(policy, obs) if self.fallback is not None else None  # shadow: no stand-in
        t2 = time.perf_counter_ns()
        self.n_fallbacks += 1
        self.last_error = error
        return action, {"rtt_us": (t2 - t0) / 1e3, "dev_us": (t2 - t1) / 1e3, "fallback": True,
                        "backend": WbcBackend.ORT, "error": error}

    # -- episode and lifecycle

    def ping(self, timeout_ms: int | None = None) -> bool:
        try:
            rep = self._rpc("ping", None, timeout_ms or self.CONNECT_TIMEOUT_MS)
        except (TimeoutError, zmq.ZMQError):
            return False
        return isinstance(rep, dict) and "error" not in rep

    def reset_episode(self) -> bool:
        """Clear the latch and the miss count and retry the server on a fresh socket.

        Returns whether the server answered a ping. Either way the next call goes to the server;
        if it is still down, three timeouts latch the new episode to ORT again.
        """
        self.latched = False
        self.consecutive = 0
        self._recreate()
        return self.ping(self.RESET_PING_MS)

    def info(self) -> dict:
        """The server's info as queried at construction (no I/O), plus the client's view of it."""
        return {**self.server_info, "url": self.url, "proto_used": self.proto,
                "timeout_ms": self.timeout_ms, "latch": self.latch,
                "parity_tol_used": self.parity_tol, "parity_max_err": self.parity_max_err,
                "fallback": self.fallback is not None}

    def close(self):
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None
        if self._own_ctx:
            self.ctx.term()
            self._own_ctx = False


# --- Runtime and the D5 class patch -------------------------------------------------------------


@dataclass
class WbcCall:
    """What the loop reads back after each env.step (one WBC call per tick)."""

    tick: int
    policy: str
    obs: np.ndarray       # float32 (1, 516), a copy (the policy reuses its buffer in place)
    action: np.ndarray    # float32 (1, 15), raw policy output
    backend: WbcBackend   # who produced `action` (ORT on fallback)
    fallback: bool
    rtt_us: float
    dev_us: float
    shadow_da: float      # max |a - a_shadow|; NaN without a shadow or when a remote shadow missed


_SEQ = itertools.count()  # per-process W-WBC sequence number


class WbcRuntime:
    """The process-wide backend the patched G1GearWbcPolicy calls, keyed by policy stem.

    ``shadow`` is an OrtCpu (D16) or a fallback-free ZmqWbc (T-3.6); a remote shadow is called after
    the primary with the same tick and seq, and its misses are counted in ``n_shadow_miss``."""

    def __init__(self, primary, shadow=None, dump: list | None = None):
        self.primary = primary
        self.shadow = shadow
        self.tick = 0           # set by the loop before each env.step
        self.last: WbcCall | None = None
        self.dump = dump        # if a list: (tick, policy, obs, action) per call
        self.n_calls = {p: 0 for p in WBC_POLICIES}
        self.n_shadow = self.n_shadow_miss = 0

    def call(self, policy: str, obs: np.ndarray) -> np.ndarray:
        tick, seq = self.tick, next(_SEQ)
        primary = self.primary
        if isinstance(primary, OrtCpu):
            t0 = time.perf_counter_ns()
            action = primary(policy, obs)
            us = (time.perf_counter_ns() - t0) / 1e3
            meta = {"rtt_us": us, "dev_us": us, "fallback": False, "backend": WbcBackend.ORT}
        else:
            action, meta = primary(policy, obs, tick, seq)
        action = np.ascontiguousarray(action, dtype=np.float32).reshape(1, WBC_ACT_DIM)
        shadow_da = NAN
        shadow = self.shadow
        if isinstance(shadow, OrtCpu):
            shadow_da = float(np.max(np.abs(action - shadow(policy, obs))))
        elif shadow is not None:  # remote shadow: never on the action path, a miss is NaN
            self.n_shadow += 1
            a_sh, sh_meta = shadow(policy, obs, tick, seq)
            if a_sh is None or sh_meta["fallback"]:
                self.n_shadow_miss += 1
            else:
                shadow_da = float(np.max(np.abs(action - a_sh)))
        self.last = WbcCall(tick, policy, obs, action, meta["backend"], bool(meta["fallback"]),
                            float(meta["rtt_us"]), float(meta["dev_us"]), shadow_da)
        self.n_calls[policy] = self.n_calls.get(policy, 0) + 1
        if isinstance(self.dump, list):
            self.dump.append((tick, policy, obs, action))
        return action


_runtime: WbcRuntime | None = None
_original = None          # G1GearWbcPolicy.load_onnx_policy before install()
_file_sha: dict = {}      # (path, size, mtime_ns) -> sha256, so resets do not re-hash


def _ort_backends(runtime: WbcRuntime) -> list:
    out = [b for b in (runtime.primary, runtime.shadow) if isinstance(b, OrtCpu)]
    fb = getattr(runtime.primary, "fallback", None)
    if isinstance(fb, OrtCpu):
        out.append(fb)
    return out


def _check_policy_file(runtime: WbcRuntime, stem: str, model_path: str):
    """The env's .onnx must be the file every in-process backend loaded (else bit-exact claims lie)."""
    served = getattr(runtime.primary, "policies", WBC_POLICIES)
    if stem not in served:
        raise KeyError(f"WBC policy {stem!r} is not served by the primary backend ({served})")
    st = os.stat(model_path)
    key = (str(model_path), st.st_size, st.st_mtime_ns)
    if key not in _file_sha:
        _file_sha[key] = _sha256(model_path)
    for ort_be in _ort_backends(runtime):
        if ort_be.sha256.get(stem) != _file_sha[key]:
            raise RuntimeError(f"{model_path} differs from {ort_be.onnx_dir}/{stem}.onnx")


def _patched_load_onnx_policy(self, model_path: str):
    """Replacement for G1GearWbcPolicy.load_onnx_policy: no ORT session, no GR00T_WBC_SERVER hook."""
    import torch

    runtime = _runtime
    stem = Path(model_path).stem
    _check_policy_file(runtime, stem, model_path)

    def run_inference(input_tensor: torch.Tensor) -> torch.Tensor:
        obs = np.array(input_tensor.detach().cpu().numpy(), dtype=np.float32, copy=True)
        if obs.shape != (1, WBC_OBS_DIM):
            raise ValueError(f"WBC obs shape {obs.shape}, want (1, {WBC_OBS_DIM})")
        return torch.from_numpy(runtime.call(stem, obs))

    run_inference.rt_stem = stem
    return run_inference


def _policy_class():
    from gr00t_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy

    return G1GearWbcPolicy


def install(runtime: WbcRuntime) -> None:
    """Route every G1GearWbcPolicy created from now on (including the one each env.reset() builds)
    through ``runtime``. Idempotent; a second call only swaps the runtime for new policies."""
    global _runtime, _original
    cls = _policy_class()
    _runtime = runtime
    if _original is None:
        _original = cls.__dict__["load_onnx_policy"]
        cls.load_onnx_policy = _patched_load_onnx_policy


def uninstall() -> None:
    """Restore the original loader; policies created while installed keep their runtime."""
    global _runtime, _original
    if _original is not None:
        _policy_class().load_onnx_policy = _original
    _runtime, _original = None, None


def installed() -> bool:
    return _original is not None
