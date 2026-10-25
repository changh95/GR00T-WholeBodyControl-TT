"""rtloop/control.py: command parsing, validation, target resolution, the loop-side state machine and
the PUSH/PULL wrappers (PLAN.md §1.4.5 control PULL, WP-1.7); scripts/control_proto.py is a shim of it.
Sockets bind 127.0.0.1 random ports only."""

import socket
import sys
import time
from pathlib import Path

import msgpack
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import control_proto  # noqa: E402  (scripts/ shim for older callers)
from rtloop import control as cp  # noqa: E402
from rtloop import wire  # noqa: E402

VLA = ["gpu", "tt"]
WBC = ["gpu", "ort", "x280"]


# --- encode / decode / validate ---------------------------------------------------------------------


@pytest.mark.parametrize("cmd,to", [("swap_vla", "tt"), ("swap_wbc", "next"), ("reset", None), ("pause", None),
                                    ("pause", "on"), ("pause", "off")])
def test_roundtrip(cmd, to):
    raw = cp.encode(cmd, to, id=7, t_ns=123)
    c = cp.parse(raw)
    assert c == cp.Command(cmd, to, 7, 123)
    assert len(raw) < 64


def test_parse_accepts_dict_and_minimal_message():
    assert cp.parse({"cmd": "reset"}) == cp.Command("reset")
    assert cp.parse(wire.dumps({"cmd": "swap_wbc", "to": "x280"})).to == "x280"


@pytest.mark.parametrize("msg,match", [
    ({"cmd": "explode"}, "cmd must be one of"),
    ({}, "cmd must be one of"),
    ({"cmd": "swap_vla"}, "needs 'to'"),
    ({"cmd": "swap_wbc", "to": ""}, "bad target name"),
    ({"cmd": "swap_wbc", "to": "GPU!"}, "bad target name"),
    ({"cmd": "swap_wbc", "to": 3}, "'to' must be a string"),
    ({"cmd": "reset", "to": "gpu"}, "takes no 'to'"),
    ({"cmd": "pause", "to": "maybe"}, "pause 'to'"),
    ({"cmd": "reset", "extra": 1}, "unknown keys"),
    ({"cmd": "reset", "id": "1"}, "'id' must be an int"),
    ({"cmd": "reset", "id": True}, "'id' must be an int"),
    ({"cmd": "reset", "t_ns": 1.5}, "'t_ns' must be an int"),
])
def test_validate_rejects(msg, match):
    with pytest.raises(cp.ControlError, match=match):
        cp.validate(msg)
    with pytest.raises(cp.ControlError):
        cp.parse(msgpack.packb(msg))


def test_decode_rejects_garbage_oversize_and_non_maps():
    with pytest.raises(cp.ControlError, match="undecodable"):
        cp.decode(b"\xc1\xc1\xc1")
    with pytest.raises(cp.ControlError, match="max"):
        cp.decode(b"\x00" * (cp.MAX_BYTES + 1))
    with pytest.raises(cp.ControlError, match="must be a map"):
        cp.decode(msgpack.packb([1, 2]))
    with pytest.raises(cp.ControlError, match="bytes or dict"):
        cp.decode(42)


def test_stale_commands_are_rejected_only_with_a_clock():
    now = 10_000_000_000
    fresh = cp.encode("reset", t_ns=now - 100_000_000)
    stale = cp.encode("reset", t_ns=now - cp.MAX_AGE_NS - 1)
    assert cp.parse(fresh, now_ns=now).cmd == "reset"
    with pytest.raises(cp.ControlError, match="stale"):
        cp.parse(stale, now_ns=now)
    assert cp.parse(stale).cmd == "reset"  # no clock given: no staleness check


def test_make_refuses_what_the_loop_would_reject():
    with pytest.raises(cp.ControlError):
        cp.make("swap_wbc")


# --- targets ---------------------------------------------------------------------------------------


def test_parse_targets():
    assert cp.parse_targets("gpu=tcp://127.0.0.1:5556, x280=tcp://127.0.0.1:5558") == {
        "gpu": "tcp://127.0.0.1:5556", "x280": "tcp://127.0.0.1:5558"}
    assert cp.parse_targets(None) == {} and cp.parse_targets("") == {}
    for bad in ("gpu", "gpu=", "=tcp://x:1", "next=tcp://x:1", "gpu=a,gpu=b", "Bad Name=tcp://x:1"):
        with pytest.raises(cp.ControlError):
            cp.parse_targets(bad)


def test_resolve():
    assert cp.resolve("next", WBC, "gpu") == "ort"
    assert cp.resolve("next", WBC, "x280") == "gpu"  # wraps
    assert cp.resolve("next", WBC, "unknown") == "gpu"
    assert cp.resolve("x280", WBC, "gpu") == "x280"
    with pytest.raises(cp.ControlError, match="unknown target"):
        cp.resolve("tt", WBC, "gpu")
    with pytest.raises(cp.ControlError, match="no swap targets"):
        cp.resolve("next", [], None)


# --- loop-side state machine -----------------------------------------------------------------------


def run(state, cmd, to=None, vla_cur="gpu", wbc_cur="gpu"):
    return cp.apply(state, cp.Command(cmd, to), VLA, vla_cur, WBC, wbc_cur)


def test_swap_wbc_is_immediate_and_noop_when_unchanged():
    st, act = run(cp.ControlState(), "swap_wbc", "next")
    assert act == cp.Action("swap_wbc", "ort") and st.n_applied == 1
    st, act = run(st, "swap_wbc", "gpu")
    assert act.kind == "noop" and "already" in act.to


def test_swap_vla_is_queued_for_the_reset_window():
    st, act = run(cp.ControlState(), "swap_vla", "next")
    assert act == cp.Action("queue_vla", "tt") and st.pending_vla == "tt"
    st2, act = run(st, "swap_vla", "next")  # 'next' again moves from the pending target, back to gpu
    assert st2.pending_vla is None and act.kind == "noop"
    st3 = cp.vla_swapped(st)
    assert st3.pending_vla is None


def test_reset_and_pause():
    st, act = run(cp.ControlState(), "pause")
    assert act.kind == "pause" and st.paused
    st, act = run(st, "pause", "on")
    assert act.kind == "noop" and st.paused
    st, act = run(st, "reset")
    assert act.kind == "reset" and not st.paused  # a reset also ends a pause
    st, act = run(st, "pause")
    st, act = run(st, "pause")
    assert act.kind == "resume" and not st.paused


def test_unknown_target_is_rejected_not_raised():
    st, act = run(cp.ControlState(), "swap_wbc", "tt")
    assert act.kind == "noop" and st.n_rejected == 1 and "unknown target" in st.last_error
    st = cp.reject(st, cp.ControlError("bad"))
    assert st.n_rejected == 2 and st.last_error == "bad"


def test_auto_swap_and_keys():
    assert [n for n in range(1, 2001) if cp.auto_swap_due(n, 10)] == [500, 1000, 1500, 2000]
    assert not cp.auto_swap_due(500, None) and not cp.auto_swap_due(0, 10) and not cp.auto_swap_due(5, -1)
    assert cp.key_command("v") == ("swap_vla", "next") and cp.key_command("w") == ("swap_wbc", "next")
    assert cp.key_command("r") == ("reset", None) and cp.key_command("p") == ("pause", None)
    assert cp.key_command("x") is None


# --- sockets -----------------------------------------------------------------------------------------


def free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if not 5555 <= port <= 5572:
            return port


def send_until(push, *a, timeout=2.0) -> bool:
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if push.send(*a):
            return True
        time.sleep(0.01)
    return False


def test_push_without_a_listener_fails_fast(zmq_ctx):
    push = cp.ControlPush(f"tcp://127.0.0.1:{free_port()}", ctx=zmq_ctx)
    t0 = time.monotonic()
    assert push.send("reset") is False
    assert time.monotonic() - t0 < 0.05 and push.n_failed == 1
    push.close()


def test_push_pull_roundtrip(zmq_ctx):
    pull = cp.ControlPull("tcp://127.0.0.1:*", ctx=zmq_ctx)
    assert not pull.url.endswith(":*") and not 5555 <= int(pull.url.rsplit(":", 1)[1]) <= 5572
    assert pull.poll() == []
    push = cp.ControlPush(pull.url, ctx=zmq_ctx)
    assert send_until(push, "swap_wbc", "next")
    for c in (("pause", None), ("reset", None)):
        assert push.send(*c)
    got, t_end = [], time.monotonic() + 2
    while len(got) < 3 and time.monotonic() < t_end:
        got += pull.poll()
        time.sleep(0.005)
    assert [(c.cmd, c.to) for c in got] == [("swap_wbc", "next"), ("pause", None), ("reset", None)]
    assert all(isinstance(c.id, int) and c.t_ns for c in got)
    push.close()
    pull.close()


def test_pull_bounds_per_poll_and_reports_bad_messages(zmq_ctx):
    import zmq

    pull = cp.ControlPull("tcp://127.0.0.1:*", ctx=zmq_ctx, hwm=64)
    raw = zmq_ctx.socket(zmq.PUSH)
    raw.setsockopt(zmq.LINGER, 0)
    raw.connect(pull.url)
    raw.send(b"\xc1garbage")
    for _ in range(11):
        raw.send(cp.encode("pause"))
    stale = cp.encode("reset", t_ns=time.monotonic_ns() - 10 * cp.MAX_AGE_NS)
    raw.send(stale)
    items, t_end = [], time.monotonic() + 2
    while len(items) < 13 and time.monotonic() < t_end:
        batch = pull.poll(max_n=4)
        assert len(batch) <= 4
        items += batch
        time.sleep(0.005)
    assert len(items) == 13
    assert isinstance(items[0], cp.ControlError) and isinstance(items[-1], cp.ControlError)
    assert "stale" in str(items[-1])
    assert all(isinstance(c, cp.Command) for c in items[1:-1])
    raw.close()
    pull.close()


def test_pull_poll_is_cheap_when_empty(zmq_ctx):
    pull = cp.ControlPull("tcp://127.0.0.1:*", ctx=zmq_ctx)
    t0 = time.perf_counter_ns()
    for _ in range(1000):
        pull.poll()
    per_us = (time.perf_counter_ns() - t0) / 1000 / 1e3
    pull.close()
    assert per_us < 50, per_us  # once per 20 ms tick; measured ~1-3 us


def test_scripts_shim_reexports_the_module():
    for name in ("parse", "encode", "apply", "resolve", "parse_targets", "auto_swap_due", "key_command",
                 "Command", "ControlError", "ControlState", "ControlPull", "ControlPush", "CONTROL_PULL", "NEXT"):
        assert getattr(control_proto, name) is getattr(cp, name), name
