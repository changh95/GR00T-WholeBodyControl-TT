"""VlaClient + mock_vla_server (WP-1.3, part of T-1.1): round trips, determinism, ramp exactness,
latency, give-up and lazy pirate, drain, bookkeeping, error/bad-shape, payload size, record/replay.

Servers bind 127.0.0.1 random free ports (49152-65535), never 5555-5572. Timing numbers are
printed with a ``[measure]`` prefix (run with ``-s`` to see them).
"""

import io
import json
import os
import signal
import socket
import threading
import time

import msgpack
import numpy as np
import pytest
import zmq

from rtloop import wire
from rtloop.schema import (ACTION_GROUPS, ARMS, H, HANDS, HEIGHT, LANGUAGE_KEY, NAV, STATE_GROUPS,
                           STATE_KEYS, VIDEO_KEY, WAIST, ChunkStatus, actions_to_rows,
                           build_vla_obs, hold_row, request_abandoned, rows_to_actions)
from rtloop.vla_client import (VlaClient, VlaReply, check_modality_config, decode_ndarray,
                               encode_ndarray, obs_keys_from_modality_config)
from rtserve import mock_vla_server as mock

GIVE_UP_SLACK_MS = 50


# --- fixtures and helpers -----------------------------------------------------------------------


def make_obs(seed: int = 0, img: bool = True) -> dict:
    """A real-sized 9-key VLA observation (states (1,1,D) f32, image (1,1,480,640,3) u8)."""
    rng = np.random.default_rng(seed)
    env_obs = {f"state.{n}": rng.uniform(-1, 1, d) for n, d in STATE_GROUPS}  # f64 like the env
    env_obs[VIDEO_KEY] = (rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) if img
                          else np.zeros((4, 4, 3), np.uint8))
    env_obs[LANGUAGE_KEY] = "pick up the apple and place it on the plate"
    return build_vla_obs(env_obs)


def make_rt(seq: int = 1, gen: int = 0, k_obs: int = 0, **kw) -> dict:
    return {"seq": seq, "gen": gen, "ep": 0, "k_obs": k_obs, "t_obs_ns": time.monotonic_ns(), **kw}


def state_hold(obs: dict) -> np.ndarray:
    return hold_row({k: obs[k].reshape(-1) for k in STATE_KEYS})


def free_port() -> int:
    """A currently unused TCP port above 49152 (nothing listens on it)."""
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        if port >= 49152:
            return port


def tick_poll(c: VlaClient, max_s: float = 5.0, tick_s: float = 0.02) -> VlaReply:
    """Poll at the 50 Hz tick cadence like the loop does, until the request finishes."""
    t_end = time.monotonic() + max_s
    while time.monotonic() < t_end:
        r = c.poll()
        if r is not None:
            return r
        time.sleep(tick_s)
    raise AssertionError("no reply/timeout within max_s")


def roundtrip(c: VlaClient, obs: dict, rt: dict, timeout_ms: int = 3000) -> VlaReply:
    c.send(obs, rt)
    r = c.wait(timeout_ms)
    assert r is not None
    return r


class Scripted:
    """A REP server that answers each request with the next scripted raw reply (bad replies)."""

    def __init__(self, replies: list):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.url = f"tcp://127.0.0.1:{self.port}"
        self.replies = list(replies)
        self.requests = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            if self.sock.poll(20):
                self.requests.append(wire.loads(self.sock.recv()))
                rep = self.replies.pop(0)
                self.sock.send(rep if isinstance(rep, bytes) else wire.dumps(rep))
        self.sock.close(linger=0)
        self.ctx.term()

    def stop(self):
        self.stop_event.set()
        self.thread.join(5)


@pytest.fixture
def server_factory():
    started = []

    def start(mode="hold", **kw):
        m = mock.start_thread(mode, **kw)
        started.append(m)
        return m

    yield start
    for m in started:
        m.stop()


@pytest.fixture
def procs():
    started = []
    yield started
    for p in started:
        if p.poll() is None:
            p.kill()
        p.communicate(timeout=5)


@pytest.fixture
def client_factory():
    made = []

    def make(url, **kw):
        c = VlaClient(url, **kw)
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


# --- round trips --------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["hold", "detfn", "ramp"])
def test_roundtrip_modes(mode, server_factory, client_factory):
    srv = server_factory(mode)
    c = client_factory(srv.url)
    obs = make_obs(1)
    rt = make_rt(seq=7, gen=3, k_obs=190)
    c.send(obs, rt)
    r = tick_poll(c)
    assert r.status == ChunkStatus.OK, r.error
    assert (r.seq, r.gen) == (7, 3)
    assert r.rows.dtype == np.float32 and r.rows.shape == (H, 35)
    assert set(r.action) == {f"action.{n}" for n, _ in ACTION_GROUPS}
    assert r.action["action.left_arm"].shape == (1, H, 7)
    np.testing.assert_array_equal(r.rows, actions_to_rows(r.action))
    # info echoes options.rt as sent, with the client's t_send_ns and give_up_ms stamps
    assert r.info["rt"] == {**rt, "t_send_ns": r.t_send_ns, "give_up_ms": c.give_up_ms}
    assert r.info["backend"] == f"mock:{mode}"
    assert r.t_send_ns < r.info["t_recv_ns"] <= r.info["t_reply_ns"] < r.t_recv_ns
    assert r.ser_us > 0 and r.deser_us > 0 and r.error is None

    hold = state_hold(obs)
    for idx in (HANDS, WAIST):
        np.testing.assert_array_equal(r.rows[:, idx], np.broadcast_to(hold[idx], (H, len(idx))))
    assert (r.rows[:, HEIGHT] == np.float32(0.74)).all()
    if mode == "hold":
        np.testing.assert_array_equal(r.rows, np.broadcast_to(hold, (H, 35)))
    if mode == "detfn":
        phi = mock.state_phase(obs)
        i = np.arange(H)
        offs = (0.05 * np.sin(2 * np.pi * i / H + phi)).astype(np.float32)
        np.testing.assert_array_equal(r.rows[:, ARMS], hold[ARMS] + offs[:, None])
        walk = ((190 + i) // 200) % 2 == 1  # rows 10.. cross tick 200
        nav = np.where(walk, 0.2, 0.0).astype(np.float32)
        np.testing.assert_array_equal(r.rows[:, NAV[0]], nav)
        assert (r.rows[:, NAV[1:]] == 0).all()
        assert walk.sum() == 20
    if mode in ("hold", "ramp"):
        assert (r.rows[:, NAV] == 0).all()


def test_detfn_nav_schedule_follows_k_obs(server_factory, client_factory):
    srv = server_factory("detfn")
    c = client_factory(srv.url)
    obs = make_obs(2, img=False)
    nav0 = roundtrip(c, obs, make_rt(1, k_obs=0)).rows[:, NAV[0]]
    nav1 = roundtrip(c, obs, make_rt(2, k_obs=200)).rows[:, NAV[0]]
    assert (nav0 == 0).all() and (nav1 == np.float32(0.2)).all()


def test_detfn_deterministic_across_processes(procs):
    """Identical request bytes -> bit-identical action payloads from separate processes (and,
    when env.sh's SRV_PYTHON is available, across the sim and server venvs)."""
    pythons = [None, None]
    srv_python = os.environ.get("SRV_PYTHON")
    if srv_python and os.path.exists(srv_python):
        pythons.append(srv_python)
    urls = []
    for py in pythons:
        p, url = mock.spawn("detfn", python=py)
        procs.append(p)
        urls.append(url)
    ctx = zmq.Context()
    try:
        raw_replies = []
        obs = make_obs(3)
        req = wire.dumps({"endpoint": "get_action",
                          "data": {"observation": obs, "options": {"rt": make_rt(5, k_obs=390)}}})
        for url in urls:
            s = ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(url)
            per = []
            for _ in range(2):  # repeatable within a process, too
                s.send(req)
                assert s.poll(5000)
                per.append(s.recv())
            s.close()
            raw_replies.append(per)
        actions = [wire.loads(b)[0] for per in raw_replies for b in per]
        ref = wire.dumps(actions[0])
        for a in actions[1:]:
            assert wire.dumps(a) == ref  # byte-identical serialised action dict
        # The action dict is the leading part of the reply bytes (info carries timestamps).
        for per in raw_replies:
            for b in per:
                assert ref in b
        # Same function in-process, too
        mine = rows_to_actions(mock.rows_detfn(obs, 390))
        assert wire.dumps(mine) == ref
        # Different state bits -> different phase
        obs2 = {**obs, "state.left_leg": obs["state.left_leg"].copy()}
        obs2["state.left_leg"][0, 0, 0] = np.nextafter(obs2["state.left_leg"][0, 0, 0],
                                                       np.float32(np.inf))
        assert mock.state_phase(obs2) != mock.state_phase(obs)
        # The image does not enter detfn
        obs3 = {**obs, VIDEO_KEY: np.zeros_like(obs[VIDEO_KEY])}
        assert wire.dumps(rows_to_actions(mock.rows_detfn(obs3, 390))) == ref
    finally:
        ctx.destroy(linger=0)


def test_ramp_float32_exact(server_factory, client_factory):
    srv = server_factory("ramp")
    c = client_factory(srv.url)
    for seed in range(5):
        obs = make_obs(10 + seed, img=False)
        r = roundtrip(c, obs, make_rt(seed + 1, k_obs=seed * 16))
        assert r.status == ChunkStatus.OK
        s43 = np.concatenate([obs[k].reshape(-1) for k in STATE_KEYS])  # state43 layout
        arms_state = np.concatenate([s43[15:22], s43[22:29]])  # state left_arm, right_arm
        for i in range(H):
            expect = arms_state + np.float32(i) * np.float32(1e-3)
            assert expect.dtype == np.float32
            assert r.rows[i, ARMS].view(np.uint32).tolist() == expect.view(np.uint32).tolist()
        assert r.action["action.left_arm"].dtype == np.float32
        assert (r.rows[:, NAV] == 0).all()


# --- latency ------------------------------------------------------------------------------------


def _p(x, q):
    return float(np.percentile(np.asarray(x), q))


def test_latency_honoured(procs, client_factory):
    """60 +- 5 ms mock (separate process) -> RTT p50 in [55, 70] ms; reports ser/deser."""
    p, url = mock.spawn("detfn", "--latency-ms", "60", "--jitter-ms", "5", "--seed", "1")
    procs.append(p)
    c = client_factory(url)
    obs = make_obs(4)
    for i in range(3):  # warm-up
        roundtrip(c, obs, make_rt(i))
    rtt, ser, deser, srv = [], [], [], []
    for i in range(40):
        r = roundtrip(c, obs, make_rt(100 + i, k_obs=i))
        assert r.status == ChunkStatus.OK
        rtt.append(r.rtt_ms)
        ser.append(r.ser_us)
        deser.append(r.deser_us)
        srv.append(r.info["infer_ms"])
    assert 55.0 <= _p(rtt, 50) <= 70.0, rtt
    # server service time is >= the drawn delay; sleep overshoot is load-dependent, so bound p50
    assert min(srv) >= 55.0 - 0.01 and _p(srv, 50) <= 65.0 + 1.0, (min(srv), _p(srv, 50))
    assert _p(np.subtract(rtt, srv), 90) < 5.0  # transport + ser overhead
    print(f"\n[measure] mock 60+-5 ms (subprocess), 0.92 MB request: RTT p50 {_p(rtt, 50):.2f} "
          f"p90 {_p(rtt, 90):.2f} max {max(rtt):.2f} ms; overhead (RTT - server) p50 "
          f"{_p(np.subtract(rtt, srv), 50):.3f} ms; ser p50 {_p(ser, 50):.0f} us p99 "
          f"{_p(ser, 99):.0f} us; deser p50 {_p(deser, 50):.0f} us p99 {_p(deser, 99):.0f} us; "
          f"request {c.last_request_bytes} B")


def test_rtt_no_latency(procs, client_factory):
    """Transport floor for the 0.92 MB request against a 0 ms mock (reported, loose bound)."""
    p, url = mock.spawn("hold")
    procs.append(p)
    c = client_factory(url)
    obs = make_obs(5)
    for i in range(5):
        roundtrip(c, obs, make_rt(i))
    rtt = [roundtrip(c, obs, make_rt(10 + i)).rtt_ms for i in range(50)]
    assert _p(rtt, 50) < 10.0
    print(f"\n[measure] mock 0 ms (subprocess), 0.92 MB request: RTT p50 {_p(rtt, 50):.3f} p90 "
          f"{_p(rtt, 90):.3f} max {max(rtt):.3f} ms")


def test_jitter_is_seeded():
    a = mock.MockVlaServer("hold", 0, latency_ms=60, jitter_ms=5, seed=3)
    b = mock.MockVlaServer("hold", 0, latency_ms=60, jitter_ms=5, seed=3)
    try:
        da = [a._delay_ns(i) for i in range(20)]
        db = [b._delay_ns(i) for i in range(20)]
        assert da == db
        assert all(55e6 <= d <= 65e6 for d in da) and len(set(da)) > 1
    finally:
        a.close()
        b.close()


# --- give-up and lazy pirate --------------------------------------------------------------------


def _assert_give_up(r: VlaReply, give_up_ms: int = 500):
    assert r.status == ChunkStatus.TIMEOUT and r.action is None and r.rows is None
    assert give_up_ms <= r.rtt_ms <= give_up_ms + GIVE_UP_SLACK_MS, r.rtt_ms
    return r.rtt_ms


def test_give_up_stalled_then_recover(server_factory, client_factory):
    srv = server_factory("hold", stall_after=0, stall_s=0.8)
    c = client_factory(srv.url)
    obs = make_obs(6, img=False)
    c.send(obs, make_rt(1))
    t0 = time.monotonic()
    r = tick_poll(c)
    ms = _assert_give_up(r)
    assert (time.monotonic() - t0) * 1e3 <= 500 + GIVE_UP_SLACK_MS
    assert r.seq == 1 and not c.in_flight and c.stats["recreate"] == 1
    assert c.stats["timeout"] == 1
    # Lazy pirate: the re-created socket can send at once; the server answers after its stall.
    r2 = roundtrip(c, obs, make_rt(2))
    assert r2.status == ChunkStatus.OK and r2.seq == 2
    print(f"\n[measure] give-up vs stalled server at {ms:.1f} ms (20 ms tick polls)")


def test_give_up_dead_server_then_server_appears(client_factory):
    port = free_port()
    url = f"tcp://127.0.0.1:{port}"
    c = client_factory(url)
    obs = make_obs(7, img=False)
    c.send(obs, make_rt(1))
    ms = _assert_give_up(tick_poll(c))
    assert c.stats["recreate"] == 1
    # The server comes up on that port; the re-created REQ socket reconnects by itself.
    m = mock.start_thread("hold", port=port)
    try:
        r = roundtrip(c, obs, make_rt(2))
        assert r.status == ChunkStatus.OK and r.seq == 2
    finally:
        m.stop()
    print(f"\n[measure] give-up vs dead server at {ms:.1f} ms")


def test_give_up_server_killed_in_flight(procs, client_factory):
    p, url = mock.spawn("hold", "--latency-ms", "300")
    procs.append(p)
    c = client_factory(url)
    assert c.ping()
    c.send(make_obs(8, img=False), make_rt(1))
    time.sleep(0.05)
    p.send_signal(signal.SIGKILL)
    _assert_give_up(tick_poll(c))
    assert not c.ping(timeout_ms=200)


def test_custom_give_up(server_factory, client_factory):
    srv = server_factory("hold", latency_ms=400)
    c = client_factory(srv.url, give_up_ms=200)
    c.send(make_obs(9, img=False), make_rt(1))
    _assert_give_up(tick_poll(c), give_up_ms=200)


def test_request_abandoned_rule():
    rt = {"t_send_ns": 10**9, "give_up_ms": 500}
    assert not request_abandoned(rt, 10**9 + 500_000_000)
    assert request_abandoned(rt, 10**9 + 500_000_001)
    assert not request_abandoned({"t_send_ns": 10**9}, 10**13)  # stock client: no give_up_ms
    assert not request_abandoned({"give_up_ms": 500}, 10**13)
    assert not request_abandoned(None, 10**13)


def test_server_skips_requests_the_client_abandoned(server_factory, client_factory):
    """After a give-up the abandoned request still sits in the server's REP queue. The mock (like
    gpu_vla_server) answers it at once with a stale error, so the live request waits only for the
    one being computed, not for every abandoned one as well."""
    srv = server_factory("hold", latency_ms=250)
    c = client_factory(srv.url, give_up_ms=100)
    obs = make_obs(10, img=False)
    c.send(obs, make_rt(1))  # taken at once, computed for 250 ms
    _assert_give_up(tick_poll(c, tick_s=0.005), give_up_ms=100)
    c.send(obs, make_rt(2))  # queued behind seq 1, abandoned at +100 ms
    _assert_give_up(tick_poll(c, tick_s=0.005), give_up_ms=100)
    c.give_up_ms = 2000
    c.send(obs, make_rt(3))  # live: taken as soon as seq 1 is done, seq 2 is skipped
    r = tick_poll(c, tick_s=0.005)
    assert r.status == ChunkStatus.OK and r.seq == 3
    info = srv.server.info()
    assert info["n_stale"] == 1 and info["n_get_action"] == 2
    assert r.rtt_ms < 450, r.rtt_ms  # ~290 ms; serving seq 2 too would add its 250 ms
    print(f"\n[measure] live request after 2 give-ups: {r.rtt_ms:.0f} ms (1 stale request skipped)")


def test_gpu_vla_server_rejects_abandoned_request_without_inference():
    from rtserve import gpu_vla_server as gvs

    class Policy:
        calls = 0

        def get_action(self, observation, options=None):
            Policy.calls += 1
            return {"action.x": np.zeros((1, H, 1), np.float32)}, {}

    srv = object.__new__(gvs.RtPolicyServer)  # the handler only; no socket, no gr00t import
    srv.policy, srv.backend, srv.log, srv.n_stale = Policy(), "cuda:test", None, 0
    srv.t_recv_ns = 10**9 + 600_000_000
    out = srv.get_action({}, {"rt": {"seq": 4, "t_send_ns": 10**9, "give_up_ms": 500}})
    assert isinstance(out, dict) and "stale" in out["error"] and Policy.calls == 0 and srv.n_stale == 1
    out = srv.get_action({}, {"rt": {"seq": 5, "t_send_ns": 10**9 + 550_000_000, "give_up_ms": 500}})
    assert isinstance(out, list) and Policy.calls == 1 and out[1]["rt"]["seq"] == 5


# --- drain --------------------------------------------------------------------------------------


def test_drain_none(server_factory, client_factory):
    c = client_factory(server_factory("hold").url)
    assert c.drain() == "none" and c.drained is None


def test_drain_ok(server_factory, client_factory):
    srv = server_factory("hold", latency_ms=100)
    c = client_factory(srv.url)
    obs = make_obs(10, img=False)
    c.send(obs, make_rt(1, gen=4))
    t0 = time.monotonic()
    assert c.drain(500) == "ok"
    dt = (time.monotonic() - t0) * 1e3
    assert dt < 200, dt
    assert not c.in_flight and c.drained.status == ChunkStatus.OK and c.drained.gen == 4
    assert c.stats["drain_ok"] == 1 and c.stats["recreate"] == 0
    assert roundtrip(c, obs, make_rt(2)).status == ChunkStatus.OK


def test_drain_recreate(server_factory, client_factory):
    srv = server_factory("hold", stall_after=0, stall_s=0.6)
    c = client_factory(srv.url)
    obs = make_obs(11, img=False)
    c.send(obs, make_rt(1))
    t0 = time.monotonic()
    assert c.drain(200) == "recreate"
    dt = (time.monotonic() - t0) * 1e3
    assert 195 <= dt <= 260, dt
    assert not c.in_flight and c.drained.status == ChunkStatus.TIMEOUT
    assert c.stats["drain_recreate"] == 1 and c.stats["recreate"] == 1 and c.stats["timeout"] == 0
    r = roundtrip(c, obs, make_rt(2))  # answered once the server's stall ends
    assert r.status == ChunkStatus.OK and r.seq == 2


# --- bookkeeping --------------------------------------------------------------------------------


def test_inflight_bookkeeping(server_factory, client_factory):
    srv = server_factory("hold", latency_ms=50)
    c = client_factory(srv.url)
    obs = make_obs(12, img=False)
    assert not c.in_flight and c.inflight_rt is None
    assert c.poll() is None and c.wait(10) is None
    rt = make_rt(3, gen=1, k_obs=42, noise_seed=2**64 - 1)
    rt_copy = dict(rt)
    c.send(obs, rt)
    assert rt == rt_copy  # caller's dict untouched
    assert c.in_flight and c.inflight_rt == {**rt, "t_send_ns": c.inflight_rt["t_send_ns"],
                                             "give_up_ms": c.give_up_ms}
    with pytest.raises(RuntimeError):
        c.send(obs, make_rt(4))
    with pytest.raises(RuntimeError):
        c.call("ping")
    r = c.wait(2000)
    assert r.status == ChunkStatus.OK and r.rt["noise_seed"] == 2**64 - 1
    assert r.info["rt"]["noise_seed"] == 2**64 - 1  # u64 survives msgpack
    assert not c.in_flight and c.inflight_rt is None and c.poll() is None
    assert c.stats["sent"] == 1 and c.stats["ok"] == 1


def test_rt_keys_required(server_factory, client_factory):
    c = client_factory(server_factory("hold").url)
    rt = make_rt(1)
    del rt["ep"]
    with pytest.raises(ValueError, match="ep"):
        c.send(make_obs(21, img=False), rt)
    assert not c.in_flight and c.stats["sent"] == 0


class _FailingSend:
    """Wraps a socket whose send raises, to exercise the send-failure path."""

    def __init__(self, real):
        self.real = real

    def send(self, *a, **kw):
        raise zmq.Again()

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_send_failure_is_an_error_reply(server_factory, client_factory):
    c = client_factory(server_factory("hold").url)
    obs = make_obs(22, img=False)
    c.sock = _FailingSend(c.sock)
    c.send(obs, make_rt(1))
    assert c.in_flight
    r = c.poll()
    assert r.status == ChunkStatus.ERROR and "send failed" in r.error and not c.in_flight
    assert c.stats["recreate"] == 1
    assert roundtrip(c, obs, make_rt(2)).status == ChunkStatus.OK  # the new socket works


def test_external_context(zmq_ctx, server_factory):
    srv = server_factory("hold")
    c = VlaClient(srv.url, ctx=zmq_ctx)
    assert c.ping()
    c.close()
    assert not zmq_ctx.closed  # a passed context is not terminated by the client


# --- error and bad-shape replies ----------------------------------------------------------------


def test_error_reply_from_mock(server_factory, client_factory):
    c = client_factory(server_factory("detfn").url)
    obs = make_obs(13, img=False)
    obs["state.left_arm"] = obs["state.left_arm"].astype(np.float64)  # the real server rejects it
    r = roundtrip(c, obs, make_rt(1))
    assert r.status == ChunkStatus.ERROR and "float32" in r.error and r.action is None
    assert c.stats["error"] == 1 and not c.in_flight
    with pytest.raises(RuntimeError, match="Unknown endpoint"):
        c.call("nope")


def _good_action(T=H):
    return rows_to_actions(np.zeros((T, 35), np.float32))


@pytest.mark.parametrize("reply, status, needle", [
    ({"error": "CUDA out of memory"}, ChunkStatus.ERROR, "CUDA out of memory"),
    (b"\xc1garbage", ChunkStatus.ERROR, "undecodable"),
    ({"action.left_arm": 1}, ChunkStatus.BAD_SHAPE, "not [action, info]"),
    ([{k: v for k, v in _good_action().items() if k != "action.waist"}, {}],
     ChunkStatus.BAD_SHAPE, "action.waist"),
    ([{**_good_action(), "action.left_arm": np.zeros((1, H, 6), np.float32)}, {}],
     ChunkStatus.BAD_SHAPE, "left_arm"),
    ([_good_action(T=20), {}], ChunkStatus.BAD_SHAPE, "AssertionError"),
    ([{**_good_action(), "action.waist": np.zeros((2, H, 3), np.float32)}, {}],
     ChunkStatus.BAD_SHAPE, "AssertionError"),
    ([{**_good_action(), "action.waist": np.full((1, H, 3), np.nan, np.float32)}, {}],
     ChunkStatus.BAD_SHAPE, "non-finite"),
    ([_good_action(), {"rt": {"seq": 999}}], ChunkStatus.ERROR, "seq mismatch"),
    ([_good_action(T=50), {}], ChunkStatus.OK, None),  # 50 model rows: first 30 kept
    ([{k: v[0] for k, v in _good_action().items()}, []], ChunkStatus.OK, None),  # (T, D) arrays
])
def test_bad_replies(reply, status, needle, client_factory):
    s = Scripted([reply])
    try:
        c = client_factory(s.url)
        r = roundtrip(c, make_obs(14, img=False), make_rt(1))
        assert r.status == status, r.error
        if needle is None:
            assert r.rows.shape == (H, 35) and r.error is None and r.info == {}
        else:
            assert needle in r.error and r.rows is None and r.action is None
        assert not c.in_flight
        assert s.requests[0]["data"]["options"]["rt"]["seq"] == 1
    finally:
        s.stop()


# --- payload ------------------------------------------------------------------------------------


def test_request_payload_size(server_factory, client_factory):
    """The 9-key trimmed request for a real 480x640x3 frame is <= 1.0 MB (R7)."""
    srv = server_factory("hold")
    c = client_factory(srv.url)
    obs = make_obs(15)
    assert set(obs) == {VIDEO_KEY, *STATE_KEYS, LANGUAGE_KEY} and len(obs) == 9
    assert obs[VIDEO_KEY].shape == (1, 1, 480, 640, 3) and obs[VIDEO_KEY].dtype == np.uint8
    assert all(obs[k].dtype == np.float32 and obs[k].ndim == 3 for k in STATE_KEYS)
    r = roundtrip(c, obs, make_rt(1, noise_seed=123))
    assert r.status == ChunkStatus.OK
    assert c.last_request_bytes <= 1_000_000, c.last_request_bytes
    # The client's reused Packer emits exactly the stock wire bytes.
    assert c.last_request == wire.dumps({"endpoint": "get_action",
                                         "data": {"observation": obs, "options": {"rt": r.rt}}})
    assert len(c.last_request) == c.last_request_bytes
    print(f"\n[measure] request payload {c.last_request_bytes} B "
          f"({c.last_request_bytes / 1e6:.3f} MB)")


def test_fast_encoder_is_byte_identical():
    """The client's ndarray encoder emits exactly what wire/np.save emit, for any array."""
    cases = [np.zeros((1, 1, 480, 640, 3), np.uint8), np.ones((1, 1, 7), np.float32),
             np.arange(5, dtype=">i4"), np.array(3.0), np.zeros((0, 3)), np.ones((2, 3), np.bool_),
             np.arange(6.0).reshape(2, 3)[:, ::2], np.asfortranarray(np.ones((3, 4))),
             np.array([1 + 2j]), np.zeros((1, 30, 7), np.float16), np.array(["ab", "c"]),
             np.zeros((1, 1, 7), np.float64), np.arange(2**20, dtype=np.int64)]
    for a in cases:
        assert encode_ndarray(a)["as_npy"] == wire._encode(a)["as_npy"], (a.dtype, a.shape)
        back = np.load(io.BytesIO(encode_ndarray(a)["as_npy"]), allow_pickle=False)
        np.testing.assert_array_equal(back, a)
    assert encode_ndarray(np.float32(2.0)) == 2.0 and encode_ndarray("x") == "x"


def test_fast_decoder_matches_wire():
    """The client's reply decoder gives the arrays wire.loads gives (writable, same bits)."""
    cases = [np.ones((1, 30, 7), np.float32), np.arange(5, dtype=">i4"), np.array(3.0),
             np.zeros((0, 3)), np.ones((2, 3), np.bool_), np.asfortranarray(np.ones((3, 4))),
             np.array([1 + 2j]), np.zeros((1, 30, 7), np.float16), np.array(["ab", "c"]),
             np.random.default_rng(0).standard_normal((1, 50, 3))]
    for _ in range(2):  # second pass hits the header cache
        for a in cases:
            msg = wire.dumps({"a": a, "nested": [a, {"x": 1}]})
            ref = wire.loads(msg)
            got = msgpack.unpackb(msg, object_hook=decode_ndarray)
            for r, g in ((ref["a"], got["a"]), (ref["nested"][0], got["nested"][0])):
                assert g.dtype == r.dtype and g.shape == r.shape and g.tobytes() == r.tobytes()
                assert g.flags.writeable and g.flags.c_contiguous == r.flags.c_contiguous
            assert got["nested"][1] == {"x": 1}
    mc = {"__ModalityConfig_class__": True, "as_json": {"modality_keys": ["a"]}}
    assert decode_ndarray(mc) is mc  # left as the dict wire leaves it


# --- wait() -------------------------------------------------------------------------------------


def test_wait_timeout_then_reply(server_factory, client_factory):
    srv = server_factory("hold", latency_ms=200)
    c = client_factory(srv.url)
    c.send(make_obs(16, img=False), make_rt(1))
    t0 = time.monotonic()
    assert c.wait(50) is None
    dt = (time.monotonic() - t0) * 1e3
    assert 49 <= dt <= 80, dt
    assert c.in_flight
    r = c.wait(1000)
    assert r.status == ChunkStatus.OK and 195 <= r.rtt_ms <= 260, r.rtt_ms


def test_wait_honours_give_up(server_factory, client_factory):
    srv = server_factory("hold", stall_after=0, stall_s=1.5)
    c = client_factory(srv.url)
    c.send(make_obs(17, img=False), make_rt(1))
    r = c.wait(3000)
    _assert_give_up(r)
    assert c.stats["recreate"] == 1


def test_stall_after_counts_get_action_only(server_factory, client_factory):
    srv = server_factory("hold", stall_after=2, stall_s=0.3)
    c = client_factory(srv.url, give_up_ms=2000)
    obs = make_obs(18, img=False)
    assert c.ping() and c.ping()  # pings do not advance the stall counter
    rtts = [roundtrip(c, obs, make_rt(i)).rtt_ms for i in range(4)]
    assert rtts[0] < 50 and rtts[1] < 50 and rtts[2] >= 300 and rtts[3] < 50, rtts


# --- info / ping / setup endpoints --------------------------------------------------------------


def test_info_ping_endpoints(server_factory, client_factory):
    srv = server_factory("ramp")
    c = client_factory(srv.url)
    assert c.ping()
    info = c.info()
    assert info["backend"] == "mock:ramp" and info["proto"] == 2
    assert c.call("get_modality_config") == {}
    assert c.call("reset", {"options": None}) == {}
    assert c.call("ping")["status"] == "ok"
    # no server at all: ping is False within its timeout, and the client stays usable
    dead = client_factory(f"tcp://127.0.0.1:{free_port()}")
    t0 = time.monotonic()
    assert dead.ping(timeout_ms=200) is False
    assert (time.monotonic() - t0) < 0.4
    with pytest.raises(TimeoutError):
        dead.call("info", timeout_ms=100)


def _mc(video=("ego_view",), state=tuple(n for n, _ in STATE_GROUPS),
        action=tuple(n for n, _ in ACTION_GROUPS), horizon=30):
    """A get_modality_config reply as rtloop.wire decodes a stock server's."""
    def m(keys, n=1):
        return {"__ModalityConfig_class__": True,
                "as_json": {"delta_indices": list(range(n)), "modality_keys": list(keys)}}
    return {"video": m(video), "state": m(state), "action": m(action, horizon),
            "language": m(["annotation.human.task_description"])}


def test_modality_config_check(server_factory, client_factory):
    assert obs_keys_from_modality_config({}) is None and check_modality_config({}) is False
    assert obs_keys_from_modality_config(_mc()) == (VIDEO_KEY, *STATE_KEYS, LANGUAGE_KEY)
    assert check_modality_config(_mc()) is True
    with pytest.raises(ValueError, match="observation keys"):
        check_modality_config(_mc(state=("right_leg", "left_leg", "waist", "left_arm",
                                         "right_arm", "left_hand", "right_hand")))
    with pytest.raises(ValueError, match="observation keys"):
        check_modality_config(_mc(video=("ego_view", "tpp_view")))
    with pytest.raises(ValueError, match="action keys"):
        check_modality_config(_mc(horizon=16))
    assert client_factory(server_factory("hold").url).check_modality_config() is False
    s = Scripted([_mc()])
    try:
        assert client_factory(s.url).check_modality_config() is True
    finally:
        s.stop()


def test_info_on_stock_server_is_empty(client_factory):
    s = Scripted([{"error": "Unknown endpoint: info"}])
    try:
        assert client_factory(s.url).info() == {}
    finally:
        s.stop()


def test_cli_starts_fast_and_exits_on_sigterm(procs):
    t0 = time.monotonic()
    p, url = mock.spawn("hold")
    procs.append(p)
    dt = time.monotonic() - t0
    assert dt < 1.0, dt
    c = VlaClient(url)
    try:
        assert c.ping()
    finally:
        c.close()
    p.send_signal(signal.SIGTERM)
    assert p.wait(5) == 0
    print(f"\n[measure] mock CLI start -> READY {dt * 1e3:.0f} ms")


# --- record-proxy -> replay ---------------------------------------------------------------------


def _record(server_factory, client_factory, rec_dir, n=5):
    up = server_factory("detfn", latency_ms=5)
    proxy = server_factory("record-proxy", upstream=up.url, record_dir=str(rec_dir))
    c = client_factory(proxy.url)
    assert c.call("get_modality_config") == {}
    obs_list, replies = [], []
    for i in range(n):
        obs = make_obs(100 + i)
        r = roundtrip(c, obs, make_rt(i + 1, k_obs=16 * i))
        assert r.status == ChunkStatus.OK, r.error
        assert r.info["backend"] == "mock:detfn"  # upstream reply passed through unchanged
        obs_list.append(obs)
        replies.append(r)
    return obs_list, replies


def test_record_proxy_then_replay(tmp_path, server_factory, client_factory):
    obs_list, recorded = _record(server_factory, client_factory, tmp_path)
    lines = [json.loads(x) for x in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert [e["i"] for e in lines] == list(range(5))
    assert all((tmp_path / e["file"]).exists() for e in lines)
    assert lines[2]["rt"]["seq"] == 3 and lines[2]["rt"]["k_obs"] == 32
    assert lines[0]["image_shape"] == [1, 1, 480, 640, 3] and lines[0]["status"] == "ok"
    with np.load(tmp_path / lines[1]["file"]) as z:
        for k in STATE_KEYS:
            assert z[k].tobytes() == obs_list[1][k].tobytes() and z[k].dtype == np.float32
    assert (tmp_path / "modality_config.msgpack").exists()

    rep = server_factory("replay", replay_dir=str(tmp_path))
    c = client_factory(rep.url)
    assert c.call("get_modality_config") == {}
    for i, obs in enumerate(obs_list):
        r = roundtrip(c, obs, make_rt(1000 + i))  # different seq numbering than the recording
        assert r.status == ChunkStatus.OK, r.error
        assert r.info["rt"]["seq"] == 1000 + i and r.info["replay_index"] == i
        for k, v in recorded[i].action.items():
            assert r.action[k].dtype == v.dtype and r.action[k].shape == v.shape
            assert r.action[k].tobytes() == v.tobytes()
    info = c.info()
    assert info["n_mismatch"] == 0 and info["replay_pos"] == 5 and info["n_recorded"] == 5
    r = roundtrip(c, obs_list[0], make_rt(2000))
    assert r.status == ChunkStatus.ERROR and "replay exhausted" in r.error


def test_replay_mismatch(tmp_path, server_factory, client_factory):
    obs_list, _ = _record(server_factory, client_factory, tmp_path, n=2)
    rep = server_factory("replay", replay_dir=str(tmp_path))
    c = client_factory(rep.url)
    bad = {**obs_list[0], "state.right_hand": obs_list[0]["state.right_hand"].copy()}
    v = bad["state.right_hand"][0, 0, 3]
    bad["state.right_hand"][0, 0, 3] = np.nextafter(v, np.float32(np.inf))  # one ulp
    r = roundtrip(c, bad, make_rt(1))
    assert r.status == ChunkStatus.ERROR and "replay state mismatch" in r.error
    assert "state.right_hand" in r.error
    img = {**obs_list[0], VIDEO_KEY: obs_list[0][VIDEO_KEY].copy()}
    img[VIDEO_KEY][0, 0, 100, 100, 0] ^= 1
    r = roundtrip(c, img, make_rt(2))
    assert r.status == ChunkStatus.ERROR and "replay state mismatch" in r.error
    assert "sha1" in r.error
    f64 = {**obs_list[0], "state.waist": obs_list[0]["state.waist"].astype(np.float64)}
    r = roundtrip(c, f64, make_rt(3))
    assert r.status == ChunkStatus.ERROR and "replay state mismatch" in r.error
    # mismatches do not consume the recording: the true request still replays
    r = roundtrip(c, obs_list[0], make_rt(4))
    assert r.status == ChunkStatus.OK
    assert c.info()["n_mismatch"] == 3


def test_replay_extra_keys_and_no_options(tmp_path, server_factory, client_factory):
    """The stock rollout sends extra keys and options=None; replay checks only recorded keys."""
    obs_list, recorded = _record(server_factory, client_factory, tmp_path, n=1)
    rep = server_factory("replay", replay_dir=str(tmp_path))
    ctx = zmq.Context()
    try:
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(rep.url)
        obs = {**obs_list[0], "video.tpp_view": np.zeros((1, 1, 8, 8, 3), np.uint8),
               "state.extra": np.zeros((1, 1, 2), np.float64)}
        s.send(wire.request("get_action", {"observation": obs, "options": None}))
        assert s.poll(3000)
        action, info = wire.loads(s.recv())
        assert info["rt"] is None
        for k, v in recorded[0].action.items():
            assert action[k].tobytes() == v.tobytes()
        s.close()
    finally:
        ctx.destroy(linger=0)


def test_record_proxy_upstream_dead(tmp_path, server_factory, client_factory):
    proxy = server_factory("record-proxy", upstream=f"tcp://127.0.0.1:{free_port()}",
                           record_dir=str(tmp_path), upstream_timeout_ms=200)
    c = client_factory(proxy.url)
    r = roundtrip(c, make_obs(19, img=False), make_rt(1))
    assert r.status == ChunkStatus.ERROR and "upstream" in r.error


def test_record_proxy_does_not_forward_kill(tmp_path, server_factory, client_factory):
    """Only get_action/get_modality_config/reset are forwarded, so a proxy in front of a stock
    server can never relay `kill` to it."""
    s = Scripted([{"status": "should not be reached"}])
    try:
        proxy = server_factory("record-proxy", upstream=s.url, record_dir=str(tmp_path))
        c = client_factory(proxy.url)
        with pytest.raises(RuntimeError, match="Unknown endpoint"):
            c.call("kill")
        assert s.requests == []
    finally:
        s.stop()


# --- stock wire compatibility -------------------------------------------------------------------


def test_stock_policy_client_talks_to_mock(server_factory):
    """gr00t's PolicyClient/MsgSerializer (what the official rollout uses) and our mock agree."""
    from gr00t.policy.server_client import MsgSerializer, PolicyClient

    srv = server_factory("detfn")
    obs = make_obs(20)
    req = {"endpoint": "get_action", "data": {"observation": obs, "options": {"rt": make_rt(1)}}}
    assert MsgSerializer.to_bytes(req) == wire.dumps(req)  # identical request bytes
    pc = PolicyClient(host="127.0.0.1", port=srv.port, timeout_ms=3000)
    try:
        assert pc.ping()
        action, info = pc._get_action(obs, None)
        np.testing.assert_array_equal(actions_to_rows(action), mock.rows_detfn(obs, 0))
        assert info["rt"] is None and info["backend"] == "mock:detfn"
    finally:
        pc.socket.close(linger=0)


def test_real_g1_modality_config_passes_check():
    """gr00t's own unitree_g1 config, through the stock encoder and rtloop.wire, matches."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.policy.server_client import MsgSerializer

    mc = MODALITY_CONFIGS["unitree_g1"]
    assert check_modality_config(mc) is True  # ModalityConfig objects
    assert check_modality_config(wire.loads(MsgSerializer.to_bytes(mc))) is True  # off the wire
