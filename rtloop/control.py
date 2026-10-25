"""Operator commands from the HUD to rt_sim: PUSH -> PULL tcp://127.0.0.1:5571 (PLAN.md §1.4.5, WP-1.7).

(The loop imports this as ``rtloop.control``; ``scripts/control_proto.py`` is a re-export shim for older callers.)

One msgpack dict per key press, ``{"cmd": swap_vla|swap_wbc|reset|pause, "to": name, "id": int, "t_ns": int}``,
applied by rt_sim at a tick boundary. Everything that decides *what* a command means is a pure function here
(``parse``, ``validate``, ``resolve``, ``apply``), so the loop side is one non-blocking recv per tick plus a
dispatch on the returned action. Semantics (PLAN.md §1.4.5, §0.2 demo ladder, WP-2.8, WP-3.9):

- ``swap_wbc``: the WBC call is stateless (D2), so the swap takes effect from the next tick.
- ``swap_vla``: queued and applied in the next reset window, where ``gen`` is bumped anyway, so a chunk from
  the old server can never be spliced (M2: "GPU<->TT swap between episodes"). ``reset`` applies it at once.
- ``reset``: ends the current episode at the boundary (no EpisodeLog outcome) and starts the next seed.
- ``pause``: freezes the sim between ticks (``to`` = on | off | None = toggle); the pacer re-anchors on resume.

``to`` is a target name registered by the loop (e.g. gpu, tt, ort, x280) or ``next`` (cycle in registration
order). Unknown keys, wrong types, oversized messages and commands older than ``MAX_AGE_NS`` are rejected,
so a stale key press queued while no loop was listening is never applied to a later run.

The two socket wrappers at the end are the only I/O: ``ControlPull`` (loop side, bind) and ``ControlPush``
(HUD side, connect with ZMQ_IMMEDIATE, so nothing queues while no loop is listening).
"""

from dataclasses import dataclass, replace
import itertools
import os
import re
import time
from collections.abc import Sequence

from rtloop import wire

CONTROL_PULL = os.environ.get("CONTROL_PULL", "tcp://127.0.0.1:5571")
CMDS = ("swap_vla", "swap_wbc", "reset", "pause")
SWAPS = ("swap_vla", "swap_wbc")
NEXT = "next"
PAUSE_TO = ("on", "off")
KEYS = frozenset({"cmd", "to", "id", "t_ns"})
MAX_BYTES = 512  # a command is ~60 B; anything larger is not ours
MAX_AGE_NS = 2_000_000_000  # sender and loop share CLOCK_MONOTONIC (same host)
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,31}$")
KEYMAP = {"v": ("swap_vla", NEXT), "w": ("swap_wbc", NEXT), "r": ("reset", None), "p": ("pause", None)}


class ControlError(ValueError):
    """A command that must not be applied (the loop logs it and carries on)."""


@dataclass(frozen=True)
class Command:
    cmd: str
    to: str | None = None
    id: int | None = None
    t_ns: int | None = None


# --- pure: encode / decode / validate ------------------------------------------------------------


def make(cmd: str, to: str | None = None, *, id: int | None = None, t_ns: int | None = None) -> dict:
    """The wire dict of one command (validated, so a sender cannot emit what the loop would reject)."""
    msg = {"cmd": cmd}
    if to is not None:
        msg["to"] = to
    if id is not None:
        msg["id"] = int(id)
    if t_ns is not None:
        msg["t_ns"] = int(t_ns)
    validate(msg)
    return msg


def encode(cmd: str, to: str | None = None, *, id: int | None = None, t_ns: int | None = None) -> bytes:
    return wire.dumps(make(cmd, to, id=id, t_ns=t_ns))


def decode(raw) -> dict:
    """bytes (msgpack via rtloop.wire) or an already-decoded dict -> dict; ControlError otherwise."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ControlError(f"command must be bytes or dict, got {type(raw).__name__}")
    raw = bytes(raw)
    if len(raw) > MAX_BYTES:
        raise ControlError(f"command is {len(raw)} B (max {MAX_BYTES})")
    try:
        msg = wire.loads(raw)
    except Exception as e:  # msgpack raises several unrelated types
        raise ControlError(f"undecodable command: {type(e).__name__}: {e}") from None
    if not isinstance(msg, dict):
        raise ControlError(f"command must be a map, got {type(msg).__name__}")
    return msg


def _int_field(msg: dict, key: str) -> int | None:
    v = msg.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise ControlError(f"{key!r} must be an int, got {type(v).__name__}")
    return v


def validate(msg: dict, now_ns: int | None = None, max_age_ns: int = MAX_AGE_NS) -> Command:
    """Check one decoded command; ``now_ns`` enables the staleness check on ``t_ns``."""
    if not isinstance(msg, dict):
        raise ControlError(f"command must be a map, got {type(msg).__name__}")
    extra = set(map(str, msg)) - KEYS
    if extra:
        raise ControlError(f"unknown keys {sorted(extra)}")
    cmd = msg.get("cmd")
    if cmd not in CMDS:
        raise ControlError(f"cmd must be one of {CMDS}, got {cmd!r}")
    to = msg.get("to")
    if to is not None and not isinstance(to, str):
        raise ControlError(f"'to' must be a string, got {type(to).__name__}")
    if cmd in SWAPS:
        if to is None:
            raise ControlError(f"{cmd} needs 'to' (a target name or {NEXT!r})")
        if not NAME_RE.match(to):
            raise ControlError(f"bad target name {to!r}")
    elif cmd == "pause":
        if to is not None and to not in PAUSE_TO:
            raise ControlError(f"pause 'to' must be one of {PAUSE_TO} or absent, got {to!r}")
    elif to is not None:
        raise ControlError(f"{cmd} takes no 'to'")
    cid, t_ns = _int_field(msg, "id"), _int_field(msg, "t_ns")
    if now_ns is not None and t_ns is not None and now_ns - t_ns > max_age_ns:
        raise ControlError(f"stale command ({(now_ns - t_ns) / 1e9:.1f} s old)")
    return Command(cmd, to, cid, t_ns)


def parse(raw, now_ns: int | None = None, max_age_ns: int = MAX_AGE_NS) -> Command:
    return validate(decode(raw), now_ns, max_age_ns)


# --- pure: targets and the loop-side state machine ---------------------------------------------


def parse_targets(spec: str | None) -> dict[str, str]:
    """'gpu=tcp://127.0.0.1:5556,x280=tcp://127.0.0.1:5558' -> {name: url}, in order (the CLI form)."""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, url = part.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not url:
            raise ControlError(f"target {part!r} is not NAME=URL")
        if not NAME_RE.match(name) or name == NEXT:
            raise ControlError(f"bad target name {name!r}")
        if name in out:
            raise ControlError(f"duplicate target {name!r}")
        out[name] = url
    return out


def resolve(to: str, names: Sequence[str], current: str | None) -> str:
    """The target a swap command names: ``next`` cycles through ``names`` from ``current``."""
    names = list(names)
    if not names:
        raise ControlError("no swap targets registered")
    if to == NEXT:
        if current not in names:
            return names[0]
        return names[(names.index(current) + 1) % len(names)]
    if to not in names:
        raise ControlError(f"unknown target {to!r} (have {names})")
    return to


@dataclass(frozen=True)
class ControlState:
    """What the operator has asked for so far; the loop owns one and replaces it on every command."""

    paused: bool = False
    pending_vla: str | None = None  # applied in the next reset window
    n_applied: int = 0
    n_rejected: int = 0
    last_error: str | None = None


@dataclass(frozen=True)
class Action:
    """What the loop must do now for one command.

    kind: swap_wbc (to) | queue_vla (to) | reset | pause | resume | noop (reason in ``to``)."""

    kind: str
    to: str | None = None


def apply(state: ControlState, cmd: Command, vla: Sequence[str], vla_cur: str | None,
          wbc: Sequence[str], wbc_cur: str | None) -> tuple[ControlState, Action]:
    """One command against the registered targets -> (new state, action). Never raises."""
    try:
        if cmd.cmd == "swap_wbc":
            to = resolve(cmd.to, wbc, wbc_cur)
            if to == wbc_cur:
                return replace(state, n_applied=state.n_applied + 1), Action("noop", f"wbc already {to}")
            return replace(state, n_applied=state.n_applied + 1), Action("swap_wbc", to)
        if cmd.cmd == "swap_vla":
            base = state.pending_vla or vla_cur  # 'next' twice before a reset moves two steps
            to = resolve(cmd.to, vla, base)
            pending = None if to == vla_cur else to
            return (replace(state, pending_vla=pending, n_applied=state.n_applied + 1),
                    Action("queue_vla", pending) if pending else Action("noop", f"vla already {to}"))
        if cmd.cmd == "reset":
            return replace(state, paused=False, n_applied=state.n_applied + 1), Action("reset")
        if cmd.cmd == "pause":
            want = {"on": True, "off": False}.get(cmd.to, not state.paused)
            if want == state.paused:
                return replace(state, n_applied=state.n_applied + 1), Action("noop", f"paused={want}")
            return replace(state, paused=want, n_applied=state.n_applied + 1), Action("pause" if want else "resume")
        raise ControlError(f"unhandled cmd {cmd.cmd!r}")
    except ControlError as e:
        return replace(state, n_rejected=state.n_rejected + 1, last_error=str(e)), Action("noop", str(e))


def reject(state: ControlState, err: ControlError) -> ControlState:
    """Count a message that failed parse/validate."""
    return replace(state, n_rejected=state.n_rejected + 1, last_error=str(err))


def vla_swapped(state: ControlState) -> ControlState:
    """The reset window applied ``pending_vla``."""
    return replace(state, pending_vla=None)


def auto_swap_due(n_ticks: int, every_s: float | None, dt: float = 0.02) -> bool:
    """T-3.9 ``--swap-wbc-every S``: a synthetic ``swap_wbc next`` every S seconds of ticks."""
    if not every_s or every_s <= 0 or n_ticks <= 0:
        return False
    period = max(1, round(every_s / dt))
    return n_ticks % period == 0


def key_command(key: str) -> tuple[str, str | None] | None:
    """HUD key -> (cmd, to): v swap_vla next, w swap_wbc next, r reset, p pause (toggle)."""
    return KEYMAP.get(key)


# --- I/O: thin socket wrappers -------------------------------------------------------------------


class ControlPull:
    """Loop side: PULL bound on 127.0.0.1, drained without blocking once per tick.

    ``poll`` reads ZMQ_EVENTS first (no exception, ~1 us when empty) and returns at most ``max_n``
    items, each a Command or the ControlError that rejected it. A port of ``*`` binds a free port (tests).
    """

    def __init__(self, url: str = CONTROL_PULL, ctx=None, hwm: int = 16):
        import zmq

        self._zmq = zmq
        self._own_ctx = ctx is None
        self.ctx = zmq.Context(io_threads=1) if ctx is None else ctx
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVHWM, hwm)
        self.sock.setsockopt(zmq.LINGER, 0)
        base, _, port = url.rpartition(":")
        if port in ("*", "0"):
            self.url = f"{base}:{self.sock.bind_to_random_port(base)}"
        else:
            self.sock.bind(url)
            self.url = url
        self.n_received = 0

    def poll(self, now_ns: int | None = None, max_n: int = 8) -> list:
        zmq, out = self._zmq, []
        while len(out) < max_n and self.sock.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            try:
                raw = self.sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            self.n_received += 1
            try:
                out.append(parse(raw, time.monotonic_ns() if now_ns is None else now_ns))
            except ControlError as e:
                out.append(e)
        return out

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None
            if self._own_ctx:
                self.ctx.term()


class ControlPush:
    """HUD side: PUSH with ZMQ_IMMEDIATE, so a key press with no loop listening fails at once
    (``send`` returns False) instead of being queued and applied to whatever loop starts next."""

    def __init__(self, url: str = CONTROL_PULL, ctx=None, hwm: int = 4):
        import zmq

        self._zmq = zmq
        self._own_ctx = ctx is None
        self.ctx = zmq.Context(io_threads=1) if ctx is None else ctx
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.IMMEDIATE, 1)
        self.sock.setsockopt(zmq.SNDHWM, hwm)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(url)
        self.url = url
        self._ids = itertools.count(1)
        self.n_sent = self.n_failed = 0

    def send(self, cmd: str, to: str | None = None) -> bool:
        raw = encode(cmd, to, id=next(self._ids), t_ns=time.monotonic_ns())
        try:
            self.sock.send(raw, self._zmq.NOBLOCK)
        except self._zmq.Again:
            self.n_failed += 1
            return False
        self.n_sent += 1
        return True

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None
            if self._own_ctx:
                self.ctx.term()
