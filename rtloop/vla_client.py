"""W-VLA client for the 50 Hz tick thread (PLAN.md §1.2 steps 2-3, §1.4.1, §3.1 "Timeouts").

The stock ``PolicyClient`` blocks in ``recv`` and never applies its timeout (R5), so a dead server
hangs the sim. This client keeps one REQ socket, sends with NOBLOCK and polls with a zero timeout
once per tick. A request still unanswered ``give_up_ms`` after its send is abandoned: the socket is
closed and re-created (lazy pirate), because a REQ socket cannot send again while a reply is
outstanding. Not thread-safe: one client per thread (the tick thread).

Only ``rtloop.schema`` / ``rtloop.wire`` are imported (no gr00t, no torch), so the sim process
stays light and CUDA-free.
"""

from dataclasses import dataclass
import io
import time

import msgpack
import numpy as np
from numpy.lib import format as npy_format
import zmq

from rtloop import wire
from rtloop.schema import ACTION_GROUPS, H, VLA_OBS_KEYS, ChunkStatus, actions_to_rows

RT_KEYS = ("seq", "gen", "ep", "k_obs", "t_obs_ns")  # required options.rt fields (PLAN.md §1.4.1)


def encode_ndarray(obj):
    """``wire._encode`` with the same bytes, minus np.save's intermediate copies (nditer buffer,
    tobytes, BytesIO): the npy 1.0 header plus one copy of a C-contiguous array's data. Anything
    else falls back to ``wire._encode``. Halves the p90 of packing the 0.92 MB request."""
    if isinstance(obj, np.ndarray) and obj.flags.c_contiguous and obj.ndim and \
            not obj.dtype.hasobject:
        h = io.BytesIO()
        npy_format.write_array_header_1_0(h, npy_format.header_data_from_array_1_0(obj))
        return {"__ndarray_class__": True, "as_npy": b"".join((h.getvalue(), obj.data))}
    return wire._encode(obj)


_NPY_HEADERS: dict[bytes, tuple | None] = {}  # npy header bytes -> (shape, dtype, count)


def decode_ndarray(obj):
    """``wire._decode`` with the same arrays (writable copies, same dtype/shape/bytes), minus
    np.load's per-array header parse: numpy's own parser runs once per distinct header and is
    cached. About 70 us vs 290 us for a 7-array reply. Anything unusual falls back to np.load."""
    if not (isinstance(obj, dict) and "__ndarray_class__" in obj):
        return obj
    b = obj["as_npy"]
    if isinstance(b, bytes) and b[:6] == b"\x93NUMPY" and b[6] == 1:
        end = 10 + int.from_bytes(b[8:10], "little")
        key = b[:end]
        meta = _NPY_HEADERS.get(key, False)
        if meta is False:
            f = io.BytesIO(key)
            npy_format.read_magic(f)
            shape, fortran, dtype = npy_format.read_array_header_1_0(f)
            meta = None if fortran or dtype.hasobject else (shape, dtype, int(np.prod(shape)))
            if len(_NPY_HEADERS) > 1024:
                _NPY_HEADERS.clear()
            _NPY_HEADERS[key] = meta
        if meta is not None and len(b) - end == meta[2] * meta[1].itemsize:
            shape, dtype, n = meta
            return np.frombuffer(b, dtype, count=n, offset=end).reshape(shape).copy()
    return wire._decode(obj)


def _modality(mc: dict, name: str) -> dict:
    cfg = mc[name]
    if isinstance(cfg, dict) and "__ModalityConfig_class__" in cfg:  # as rtloop.wire leaves it
        return cfg["as_json"]
    return cfg if isinstance(cfg, dict) else vars(cfg)  # a gr00t ModalityConfig object


def obs_keys_from_modality_config(mc: dict) -> tuple[str, ...] | None:
    """The request keys a server's get_modality_config implies (video, state, language order),
    or None when it publishes none (mocks reply ``{}``)."""
    if not mc:
        return None
    return (*(f"video.{k}" for k in _modality(mc, "video")["modality_keys"]),
            *(f"state.{k}" for k in _modality(mc, "state")["modality_keys"]),
            *_modality(mc, "language")["modality_keys"])


def check_modality_config(mc: dict) -> bool:
    """Assert a server's modality config matches the 9-key request and the A35 action layout
    (PLAN.md §1.4.1: keys derived at startup and asserted to equal the table). False if the
    server publishes none."""
    keys = obs_keys_from_modality_config(mc)
    if keys is None:
        return False
    if keys != VLA_OBS_KEYS:
        raise ValueError(f"server observation keys {keys} != {VLA_OBS_KEYS}")
    act = _modality(mc, "action")
    names = tuple(act["modality_keys"])
    if names != tuple(n for n, _ in ACTION_GROUPS) or len(act["delta_indices"]) < H:
        raise ValueError(f"server action keys {names} / horizon {len(act['delta_indices'])} "
                         f"do not match the A35 layout with H = {H}")
    return True


@dataclass
class VlaReply:
    """The outcome of one get_action request, whatever happened to it.

    ``ser_us`` covers serialisation (pack plus one copy). ``t_send_ns`` is stamped just before it
    and is also ``options.rt.t_send_ns`` on the wire; ``t_recv_ns`` just after the reply bytes are
    received (or at give-up), before decoding. ``deser_us`` covers decode plus shape validation.
    ``rows`` and ``rt`` are additions to INTERFACES.md: the validated float32 (H, 35) A35 array on
    OK (so the loop does not decode the chunk twice) and the options.rt that was sent.
    """

    seq: int
    gen: int
    status: ChunkStatus
    action: dict | None
    info: dict
    t_send_ns: int
    t_recv_ns: int
    ser_us: float
    deser_us: float
    error: str | None
    rows: np.ndarray | None = None
    rt: dict | None = None

    @property
    def rtt_ms(self) -> float:
        return (self.t_recv_ns - self.t_send_ns) / 1e6


class VlaClient:
    """One REQ socket to a PolicyServer-wire VLA server; at most one request in flight."""

    def __init__(self, url: str, give_up_ms: int = 500, ctx: zmq.Context | None = None):
        self.url = url
        self.give_up_ms = give_up_ms
        self._own_ctx = ctx is None
        self.ctx = zmq.Context() if ctx is None else ctx
        self.sock: zmq.Socket | None = None
        self._rt: dict | None = None  # options.rt of the request in flight
        self._t_send = 0
        self._ser_us = 0.0
        self._send_error: str | None = None
        # A persistent Packer keeps its ~1 MiB buffer between requests; msgpack.packb allocates
        # (and page-faults) a fresh one per call: about 0.3 ms vs 1.0 ms p50 for send() on a
        # 0.92 MB request. Same bytes as wire.dumps (tested).
        self._packer = msgpack.Packer(default=encode_ndarray, autoreset=False)
        self.last_request_bytes = 0
        self.last_request = b""
        self.drained: VlaReply | None = None  # the reply drain() discarded, for the RequestLog
        # Counters for config/episode logs (T-1.10: drain_recreate <= 1 % of episodes).
        self.stats = {"sent": 0, "ok": 0, "error": 0, "bad_shape": 0, "timeout": 0,
                      "recreate": 0, "drain_ok": 0, "drain_recreate": 0}
        self._connect()

    # --- socket lifecycle ---------------------------------------------------------------------

    def _connect(self) -> None:
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)  # close() must never block on an abandoned request
        s.connect(self.url)
        self.sock = s

    def _recreate(self) -> None:
        """Lazy pirate: the only way out of a REQ socket waiting for a reply that is not coming."""
        if self.sock is not None:
            self.sock.close(linger=0)
        self._connect()
        self.stats["recreate"] += 1

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None
        self._rt = None
        if self._own_ctx and not self.ctx.closed:
            self.ctx.term()

    # --- blocking calls (outside the tick loop) -----------------------------------------------

    def call(self, endpoint: str, data: dict | None = None, timeout_ms: int = 5000):
        """Blocking request/reply for setup endpoints (ping, info, get_modality_config, reset).

        Raises TimeoutError (after re-creating the socket), or RuntimeError on an ``{"error"}``
        reply.
        """
        if self._rt is not None:
            raise RuntimeError("VlaClient.call() while a get_action request is in flight")
        self.sock.send(wire.request(endpoint, data))
        if not self.sock.poll(timeout_ms, zmq.POLLIN):
            self._recreate()
            raise TimeoutError(f"{endpoint} on {self.url}: no reply in {timeout_ms} ms")
        msg = wire.loads(self.sock.recv())
        if isinstance(msg, dict) and "error" in msg:
            raise RuntimeError(f"{endpoint} on {self.url}: server error: {msg['error']}")
        return msg

    def ping(self, timeout_ms: int = 1000) -> bool:
        try:
            self.call("ping", timeout_ms=timeout_ms)
            return True
        except (TimeoutError, RuntimeError, zmq.ZMQError):
            return False

    def check_modality_config(self, timeout_ms: int = 5000) -> bool:
        """Startup check of the server's keys against the 9-key request (False for mocks)."""
        return check_modality_config(self.call("get_modality_config", timeout_ms=timeout_ms))

    def info(self, timeout_ms: int = 1000) -> dict:
        """Server INFO; ``{}`` from a stock PolicyServer, which has no ``info`` endpoint."""
        try:
            out = self.call("info", timeout_ms=timeout_ms)
        except RuntimeError:
            return {}
        return out if isinstance(out, dict) else {}

    # --- non-blocking get_action --------------------------------------------------------------

    @property
    def in_flight(self) -> bool:
        return self._rt is not None

    @property
    def inflight_rt(self) -> dict | None:
        return self._rt

    def send(self, vla_obs: dict, rt: dict) -> None:
        """Serialise and send one get_action request without blocking (PLAN.md §1.2 step 3).

        ``rt`` is ``options.rt`` (seq, gen, ep, k_obs, t_obs_ns[, noise_seed]); a copy is sent with
        ``t_send_ns`` and ``give_up_ms`` (this client's give-up for the request) stamped here, so an
        RT-owned server can skip a request the client has abandoned (schema.request_abandoned). The
        caller's dict is not modified.
        """
        if self._rt is not None:
            raise RuntimeError(f"VlaClient.send() while seq {self._rt.get('seq')} is in flight")
        missing = [k for k in RT_KEYS if k not in rt]
        if missing:
            raise ValueError(f"options.rt lacks {missing}")
        rt = dict(rt)
        t0 = time.monotonic_ns()
        rt["t_send_ns"] = t0
        rt["give_up_ms"] = int(self.give_up_ms)
        p = self._packer
        p.reset()
        p.pack({"endpoint": "get_action",
                "data": {"observation": vla_obs, "options": {"rt": rt}}})
        payload = p.bytes()
        t1 = time.monotonic_ns()
        self._rt, self._t_send, self._ser_us = rt, t0, (t1 - t0) / 1e3
        self.last_request_bytes = len(payload)
        self.last_request = payload  # kept for audits/tests; it is replaced on the next send
        self.stats["sent"] += 1
        self._send_error = None
        try:
            self.sock.send(payload, zmq.NOBLOCK, copy=False)
        except zmq.ZMQError as e:
            # Report the failure on the next poll as an ERROR reply, so the caller's bookkeeping
            # has a single path; a fresh socket rules out a REQ state-machine wedge.
            self._send_error = f"send failed: {e}"
            self._recreate()

    def _finish(self, status: ChunkStatus, t_recv: int, deser_us: float = 0.0,
                action: dict | None = None, info: dict | None = None, rows=None,
                error: str | None = None, count: bool = True) -> VlaReply:
        rt = self._rt
        self._rt = None
        if count:
            self.stats[status.name.lower()] += 1
        return VlaReply(seq=int(rt.get("seq", 0)), gen=int(rt.get("gen", 0)), status=status,
                        action=action, info={} if info is None else info, t_send_ns=self._t_send,
                        t_recv_ns=t_recv, ser_us=self._ser_us, deser_us=deser_us, error=error,
                        rows=rows, rt=rt)

    def _decode(self, frame, t_recv: int) -> VlaReply:
        """Reply bytes -> VlaReply: ``[action, info]`` -> OK / BAD_SHAPE, ``{"error"}`` -> ERROR."""
        t0 = time.monotonic_ns()
        try:
            msg = msgpack.unpackb(frame, object_hook=decode_ndarray)
        except Exception as e:  # noqa: BLE001 - any undecodable reply is a server fault
            return self._finish(ChunkStatus.ERROR, t_recv, (time.monotonic_ns() - t0) / 1e3,
                                error=f"undecodable reply: {e!r}")
        if isinstance(msg, dict) and "error" in msg:
            return self._finish(ChunkStatus.ERROR, t_recv, (time.monotonic_ns() - t0) / 1e3,
                                error=str(msg["error"]))
        if not (isinstance(msg, (list, tuple)) and len(msg) == 2 and isinstance(msg[0], dict)):
            return self._finish(ChunkStatus.BAD_SHAPE, t_recv, (time.monotonic_ns() - t0) / 1e3,
                                error=f"reply is not [action, info]: {type(msg).__name__}")
        action, info = msg[0], msg[1] if isinstance(msg[1], dict) else {}
        try:
            rows = actions_to_rows(action)
        except (KeyError, AssertionError, ValueError, TypeError) as e:
            return self._finish(ChunkStatus.BAD_SHAPE, t_recv, (time.monotonic_ns() - t0) / 1e3,
                                info=info, error=f"bad action: {type(e).__name__}: {e}")
        if not np.isfinite(rows).all():
            return self._finish(ChunkStatus.BAD_SHAPE, t_recv, (time.monotonic_ns() - t0) / 1e3,
                                info=info, error="bad action: non-finite values")
        echo = info.get("rt")
        if isinstance(echo, dict) and isinstance(echo.get("seq"), int) \
                and echo["seq"] != self._rt.get("seq"):
            return self._finish(ChunkStatus.ERROR, t_recv, (time.monotonic_ns() - t0) / 1e3,
                                info=info,
                                error=f"seq mismatch: sent {self._rt.get('seq')}, "
                                      f"got {echo['seq']}")
        return self._finish(ChunkStatus.OK, t_recv, (time.monotonic_ns() - t0) / 1e3,
                            action=action, info=info, rows=rows)

    def _give_up(self, now: int) -> VlaReply:
        self._recreate()
        return self._finish(ChunkStatus.TIMEOUT, now,
                            error=f"no reply {(now - self._t_send) / 1e6:.1f} ms after send "
                                  f"(give-up {self.give_up_ms} ms)")

    def poll(self) -> VlaReply | None:
        """Zero-timeout check, once per tick: a VlaReply if the request finished, else None."""
        if self._rt is None:
            return None
        if self._send_error is not None:
            return self._finish(ChunkStatus.ERROR, time.monotonic_ns(), error=self._send_error)
        if self.sock.poll(0, zmq.POLLIN):
            frame = self.sock.recv(copy=False)
            return self._decode(frame.buffer, time.monotonic_ns())
        now = time.monotonic_ns()
        if now - self._t_send > self.give_up_ms * 1_000_000:
            return self._give_up(now)
        return None

    def wait(self, timeout_ms: int) -> VlaReply | None:
        """Blocking poll (pre-roll, VL visibility, sync mode): a VlaReply, or None after
        ``timeout_ms`` with the request still in flight. The give-up deadline still applies, so
        this returns a TIMEOUT reply at ``t_send + give_up_ms`` if that comes first."""
        if self._rt is None:
            return None
        if self._send_error is not None:
            return self._finish(ChunkStatus.ERROR, time.monotonic_ns(), error=self._send_error)
        deadline = time.monotonic_ns() + int(timeout_ms * 1_000_000)
        give_up_at = self._t_send + self.give_up_ms * 1_000_000
        while True:
            now = time.monotonic_ns()
            until = min(deadline, give_up_at)
            wait_ms = max(0, -(-(until - now) // 1_000_000))  # ceil, so we never wake early
            if self.sock.poll(wait_ms, zmq.POLLIN):
                frame = self.sock.recv(copy=False)
                return self._decode(frame.buffer, time.monotonic_ns())
            now = time.monotonic_ns()
            if now >= give_up_at:
                return self._give_up(now)
            if now >= deadline:
                return None

    def drain(self, timeout_ms: int = 500) -> str:
        """Episode end (PLAN.md §1.2 lifecycle step 1): wait up to ``timeout_ms`` for the reply
        in flight and discard it (kept in ``self.drained``); if none comes, re-create the socket.

        Returns "none" (nothing in flight), "ok" (reply received) or "recreate".
        """
        self.drained = None
        if self._rt is None:
            return "none"
        if self._send_error is None and self.sock.poll(int(timeout_ms), zmq.POLLIN):
            frame = self.sock.recv(copy=False)
            self.drained = self._decode(frame.buffer, time.monotonic_ns())
            self.stats["drain_ok"] += 1
            return "ok"
        if self._send_error is None:  # (a failed send already re-created the socket)
            self._recreate()
        self.drained = self._finish(ChunkStatus.TIMEOUT, time.monotonic_ns(), count=False,
                                    error=self._send_error or "drained without reply")
        self.stats["drain_recreate"] += 1
        return "recreate"
