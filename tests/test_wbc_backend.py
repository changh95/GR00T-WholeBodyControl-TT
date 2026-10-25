"""WP-1.4: ORT backends, the numpy oracle, ZmqWbc failover and the D5 patch.

Unit tests run fake W-WBC servers on random 127.0.0.1 ports. ``sim`` tests build the real G1 env.
Measurements are printed as ``REPORT`` lines.
"""

import os
from pathlib import Path
import shutil
import socket
import threading
import time

import numpy as np
import pytest
import zmq

from rtloop import schema, wire
from rtloop import wbc_backends as wb
from rtloop.schema import WBC_BALANCE, WBC_OBS_DIM, WBC_POLICIES, WBC_WALK, WbcBackend

RT = Path(__file__).resolve().parents[1]
RESERVED_PORTS = range(5555, 5573)


def report(capsys, msg: str):
    with capsys.disabled():
        print(f"\nREPORT {msg}")


def unit_obs(rng, n: int) -> np.ndarray:
    """(n, 1, 516) float32, unit normal: harsher than the sim (|x| up to ~4 in every slot)."""
    return rng.standard_normal((n, 1, WBC_OBS_DIM)).astype(np.float32)


def sim_like_obs(rng, n: int) -> np.ndarray:
    """(n, 516) float32 at the magnitudes the sim produces, per the PLAN.md §1.4.3 frame layout."""
    f = np.zeros((n, schema.WBC_HISTORY, schema.WBC_FRAME))
    s = f.shape[:2]
    f[..., 0:3] = rng.uniform(-1, 1, s + (3,)) * [1.0, 1.0, 0.5]  # nav cmd x [2, 2, 0.5]
    f[..., 3] = rng.uniform(0.65, 0.8, s)                          # height cmd
    f[..., 4:7] = rng.normal(0, 0.1, s + (3,))                     # rpy cmd
    f[..., 7:10] = rng.normal(0, 0.2, s + (3,))                    # pelvis omega x 0.5
    g = rng.normal(0, 0.1, s + (3,)) + [0.0, 0.0, -1.0]
    f[..., 10:13] = g / np.linalg.norm(g, axis=-1, keepdims=True)  # gravity in the pelvis frame
    f[..., 13:42] = rng.normal(0, 0.3, s + (29,))                  # q29 - default
    f[..., 42:71] = rng.normal(0, 0.05, s + (29,))                 # dq29 x 0.05
    f[..., 71:86] = rng.normal(0, 1.0, s + (15,))                  # last raw action
    return f.reshape(n, WBC_OBS_DIM).astype(np.float32)


@pytest.fixture(scope="module")
def ort1():
    return wb.OrtCpu(opts="single")


# --- OrtCpu and the numpy oracle ----------------------------------------------------------------


@pytest.mark.parametrize("opts", ["single", "default"])
def test_ort_cpu_matches_direct_session(opts):
    import onnxruntime as ort

    be = wb.OrtCpu(opts=opts)
    assert be.policies == WBC_POLICIES and be.backend == WbcBackend.ORT
    x = unit_obs(np.random.default_rng(1), 64)
    for stem in WBC_POLICIES:
        path = f"{wb.ONNX_DIR}/{stem}.onnx"
        if opts == "default":
            ref = ort.InferenceSession(path)  # exactly g1_gear_wbc_policy.py:58
        else:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            ref = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        for xi in x:
            a = be(stem, xi)
            assert a.dtype == np.float32 and a.shape == (1, 15)
            assert np.array_equal(a, ref.run(None, {"input": xi})[0])


def test_ort_cpu_positional_opts_and_single_vs_default(capsys):
    single, default = wb.OrtCpu(), wb.OrtCpu("default")
    assert single.opts == "single" and default.opts == "default"
    assert default.onnx_dir == wb.ONNX_DIR
    x = unit_obs(np.random.default_rng(2), 500)
    worst = max(float(np.max(np.abs(single(p, xi) - default(p, xi)))) for p in WBC_POLICIES for xi in x)
    report(capsys, f"ORT single vs default, 1000 unit-normal inputs: max |da| = {worst:.3e}")
    assert worst < 1e-5


def test_numpy_reference_matches_ort(ort1, capsys):
    from rtserve import wbc_numpy_ref

    refs = wbc_numpy_ref.load_dir(wb.ONNX_DIR)
    assert set(refs) == set(WBC_POLICIES)
    rng = np.random.default_rng(3)
    xs, xu = sim_like_obs(rng, 1000), unit_obs(rng, 1000)[:, 0]
    for stem, ref in refs.items():
        assert ref.num_obs == WBC_OBS_DIM and ref.num_single_obs == schema.WBC_FRAME
        ort_s = np.concatenate([ort1(stem, x[None]) for x in xs])
        ort_u = np.concatenate([ort1(stem, x[None]) for x in xu])
        es = float(np.max(np.abs(ort_s - ref(xs))))
        eu = float(np.max(np.abs(ort_u - ref(xu))))
        report(capsys, f"numpy f64 vs ORT f32 {stem}: sim-like max {es:.2e}, unit-normal max {eu:.2e}")
        assert es < 1e-5            # the oracle reproduces the graph at sim magnitudes
        assert eu < 1e-4            # unit-normal inputs reach |a| ~ 20; f32 rounding ~1e-5 there


def test_numpy_reference_rejects_other_graphs(tmp_path):
    import onnx

    from rtserve import wbc_numpy_ref

    m = onnx.load(f"{wb.ONNX_DIR}/{WBC_BALANCE}.onnx")
    elu = next(n for n in m.graph.node if n.op_type == "Elu")
    elu.op_type = "Relu"
    del elu.attribute[:]
    onnx.save(m, str(tmp_path / "x.onnx"))
    with pytest.raises(ValueError, match="op sequence"):
        wbc_numpy_ref.NumpyWbc(str(tmp_path / "x.onnx"))


# --- ZmqWbc against fake servers ----------------------------------------------------------------


def _stock_gear_wbc_action(policy, obs):
    """The stock endpoint's signature (gear_wbc_policy.py:107): no tick/seq."""


class FakeWbcServer:
    """W-WBC stand-in on a ROUTER socket: unlike REP it can read a request and never answer."""

    def __init__(self, ctx: zmq.Context, ort: wb.OrtCpu, proto: int = 2, port: int | None = None,
                 info_extra: dict | None = None, offset: float = 1e-3):
        self.ort, self.proto = ort, proto
        self.mode = "ok"            # ok | silent | error | wrong | late
        self.late_s = 0.02
        self.info_extra = info_extra or {}  # e.g. x280d's parity_tol
        self.offset = np.float32(offset)    # what "wrong" adds to every action
        self.requests = []          # (endpoint, sorted data keys, data)
        self.sock = ctx.socket(zmq.ROUTER)
        self.sock.setsockopt(zmq.LINGER, 0)
        if port is None:
            self.port = self.sock.bind_to_random_port("tcp://127.0.0.1", 20000, 60000)
        else:
            deadline = time.monotonic() + 3.0
            while True:
                try:
                    self.sock.bind(f"tcp://127.0.0.1:{port}")
                    break
                except zmq.ZMQError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.02)
            self.port = port
        assert self.port not in RESERVED_PORTS
        self.url = f"tcp://127.0.0.1:{self.port}"
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def n_actions(self) -> int:
        return sum(1 for r in self.requests if r[0] == "gear_wbc_action")

    def _reply(self, endpoint: str, data: dict):
        if endpoint == "info":
            if self.proto < 2:
                return {"error": "Unknown endpoint: info"}
            return {"backend": "fake", "proto": self.proto, "policies": list(WBC_POLICIES), **self.info_extra}
        if endpoint == "ping":
            return {"status": "ok"}
        if endpoint != "gear_wbc_action":
            return {"error": f"Unknown endpoint: {endpoint}"}
        if self.mode == "silent":
            return None
        if self.mode == "error":
            return {"error": "boom"}
        if self.mode == "garbage":
            return b"\xc1 not msgpack"
        try:
            if self.proto < 2:
                _stock_gear_wbc_action(**data)
            action = self.ort(data["policy"], data["obs"])
        except Exception as e:
            return {"error": str(e)}
        if self.mode == "wrong":
            action = action + self.offset
        if self.mode == "late":
            time.sleep(self.late_s)
        out = {"action": action}
        if self.proto >= 2:
            out.update(tick=data.get("tick"), seq=data.get("seq"), compute_us=12.5)
        return out

    def _run(self):
        while not self._stop:
            if not self.sock.poll(10):
                continue
            ident, empty, msg = self.sock.recv_multipart()
            req = wire.loads(msg)
            data = req.get("data") or {}
            self.requests.append((req.get("endpoint"), tuple(sorted(data)), data))
            rep = self._reply(req.get("endpoint"), data)
            if rep is not None:
                body = rep if isinstance(rep, bytes) else wire.dumps(rep)
                self.sock.send_multipart([ident, empty, body])
        self.sock.close(linger=0)

    def stop(self):
        self._stop = True
        self.thread.join(2.0)


@pytest.fixture
def fake(zmq_ctx, ort1):
    servers = []

    def make(**kw):
        servers.append(FakeWbcServer(zmq_ctx, ort1, **kw))
        return servers[-1]

    yield make
    for s in servers:
        s.stop()


def test_zmq_normal_reply(fake, ort1):
    srv = fake()
    z = wb.ZmqWbc(srv.url, ort1)
    try:
        assert z.proto == 2 and z.backend == WbcBackend.OTHER and z.parity_max_err == 0.0
        assert z.info()["backend"] == "fake" and z.info()["proto_used"] == 2
        rng = np.random.default_rng(4)
        for k in range(20):
            policy = WBC_POLICIES[k % 2]
            x = unit_obs(rng, 1)[0]
            a, meta = z(policy, x, tick=k, seq=100 + k)
            assert np.array_equal(a, ort1(policy, x))
            assert not meta["fallback"] and meta["backend"] == WbcBackend.OTHER and meta["error"] is None
            assert meta["dev_us"] == 12.5 and 0 < meta["rtt_us"] < 50_000
            last = srv.requests[-1]
            assert last[1] == ("obs", "policy", "seq", "tick")
            assert last[2]["tick"] == k and last[2]["seq"] == 100 + k
        assert z.n_fallbacks == 0
    finally:
        z.close()


def test_zmq_silent_server_falls_back_on_same_call(fake, ort1, capsys):
    srv = fake()
    timeout_ms = 5
    z = wb.ZmqWbc(srv.url, ort1, timeout_ms=timeout_ms, latch=False)
    try:
        srv.mode = "silent"
        x = unit_obs(np.random.default_rng(5), 1)[0]
        walls = []
        for k in range(10):
            t0 = time.perf_counter_ns()
            a, meta = z(WBC_BALANCE, x, tick=k, seq=k)
            walls.append((time.perf_counter_ns() - t0) / 1e6)
            assert meta["fallback"] and meta["error"] == "timeout" and meta["backend"] == WbcBackend.ORT
            assert np.array_equal(a, ort1(WBC_BALANCE, x))
            assert meta["rtt_us"] / 1e3 <= walls[-1] + 1e-3
        report(capsys, f"silent-server fallback wall ms (timeout {timeout_ms}): "
                       + ", ".join(f"{w:.3f}" for w in walls))
        # The mechanism costs ~0.2 ms past the timeout (libzmq's own overshoot ~0.08 ms, socket
        # re-create ~0.02 ms, ORT ~0.03 ms); single calls can see host scheduling jitter on top, so
        # the 0.5 ms bound is on the median and every call gets a hard ceiling (no second wait).
        assert all(w >= timeout_ms for w in walls), walls
        assert np.median(walls) <= timeout_ms + 0.5 and max(walls) <= timeout_ms + 1.5, walls
        assert not z.latched and z.n_timeouts == 10 and z.n_recreate >= 10
    finally:
        z.close()


def test_zmq_latch_after_three_then_reset_episode(fake, ort1):
    srv = fake()
    z = wb.ZmqWbc(srv.url, ort1, timeout_ms=5)
    try:
        srv.mode = "silent"
        x = unit_obs(np.random.default_rng(6), 1)[0]
        for k in range(3):
            assert not z.latched
            a, meta = z(WBC_WALK, x, tick=k, seq=k)
            assert meta["fallback"] and meta["error"] == "timeout"
        assert z.latched and z.n_latches == 1
        sent = srv.n_actions()
        t0 = time.perf_counter_ns()
        a, meta = z(WBC_WALK, x, tick=3, seq=3)
        assert (time.perf_counter_ns() - t0) / 1e6 < 1.0      # no wait once latched
        assert meta["fallback"] and meta["error"] == "latched"
        assert np.array_equal(a, ort1(WBC_WALK, x))
        time.sleep(0.05)
        assert srv.n_actions() == sent                        # nothing sent while latched
        srv.mode = "ok"
        assert z.reset_episode() is True
        assert not z.latched and z.consecutive == 0
        a, meta = z(WBC_WALK, x, tick=0, seq=4)
        assert not meta["fallback"] and np.array_equal(a, ort1(WBC_WALK, x))
    finally:
        z.close()


def test_zmq_no_latch_and_error_replies(fake, ort1):
    srv = fake()
    z = wb.ZmqWbc(srv.url, ort1, timeout_ms=5, latch=False)
    try:
        x = unit_obs(np.random.default_rng(7), 1)[0]
        srv.mode = "silent"
        for k in range(5):
            assert z(WBC_BALANCE, x, k, k)[1]["fallback"]
        assert not z.latched and z.consecutive == 5
        srv.mode = "error"
        a, meta = z(WBC_BALANCE, x, 5, 5)
        assert meta["fallback"] and meta["error"] == "server: boom" and z.n_errors == 1
        srv.mode = "garbage"                                  # undecodable: fallback, never raise
        a, meta = z(WBC_BALANCE, x, 6, 6)
        assert meta["fallback"] and meta["error"].startswith("decode:") and z.n_errors == 2
        assert np.array_equal(a, ort1(WBC_BALANCE, x))
        srv.mode = "ok"
        assert not z(WBC_BALANCE, x, 7, 7)[1]["fallback"]
    finally:
        z.close()
    # Latching counts consecutive misses only: a good reply in between resets the count.
    srv.mode = "ok"
    z = wb.ZmqWbc(srv.url, ort1, timeout_ms=5)
    try:
        for mode in ("error", "error", "ok", "error", "error", "ok"):
            srv.mode = mode
            assert z(WBC_BALANCE, x, 0, 0)[1]["fallback"] == (mode != "ok")
        assert not z.latched
    finally:
        z.close()


def test_zmq_late_reply_never_reaches_a_later_call(fake, ort1):
    srv = fake()
    z = wb.ZmqWbc(srv.url, ort1, timeout_ms=5, latch=False)
    try:
        rng = np.random.default_rng(8)
        x1, x2 = unit_obs(rng, 2)
        srv.mode, srv.late_s = "late", 0.02
        assert z(WBC_BALANCE, x1, 1, 1)[1]["error"] == "timeout"
        time.sleep(0.03)                                      # the late reply goes out now
        srv.mode = "ok"
        a, meta = z(WBC_BALANCE, x2, 2, 2)
        assert not meta["fallback"] and np.array_equal(a, ort1(WBC_BALANCE, x2))
    finally:
        z.close()


def test_zmq_recovers_when_server_comes_back(fake, ort1):
    srv = fake()
    port = srv.port
    z = wb.ZmqWbc(srv.url, ort1, timeout_ms=5, latch=False)
    try:
        x = unit_obs(np.random.default_rng(9), 1)[0]
        assert not z(WBC_BALANCE, x, 0, 0)[1]["fallback"]
        srv.stop()                                            # server process "killed"
        for k in range(3):
            a, meta = z(WBC_BALANCE, x, k, k)
            assert meta["fallback"] and np.array_equal(a, ort1(WBC_BALANCE, x))
        fake(port=port)                                       # restarted on the same port
        results = [z(WBC_BALANCE, x, 10 + k, 10 + k)[1]["fallback"] for k in range(3)]
        assert results[-1] is False and results.index(False) <= 1, results
    finally:
        z.close()


def test_zmq_latched_episode_recovers_after_restart(fake, ort1):
    srv = fake()
    port = srv.port
    z = wb.ZmqWbc(srv.url, ort1, timeout_ms=5)
    try:
        x = unit_obs(np.random.default_rng(10), 1)[0]
        srv.stop()
        for k in range(3):
            z(WBC_WALK, x, k, k)
        assert z.latched
        assert z.reset_episode() is False and not z.latched   # still down: unlatched, retried
        for k in range(3):
            z(WBC_WALK, x, k, k)
        assert z.latched
        fake(port=port)
        assert z.reset_episode() is True
        a, meta = z(WBC_WALK, x, 0, 0)
        assert not meta["fallback"] and np.array_equal(a, ort1(WBC_WALK, x))
    finally:
        z.close()


def test_zmq_proto1_server_gets_only_policy_and_obs(fake, ort1):
    srv = fake(proto=1)
    z = wb.ZmqWbc(srv.url, ort1)
    try:
        assert z.proto == 1 and z.backend == WbcBackend.OTHER
        x = unit_obs(np.random.default_rng(11), 1)[0]
        for k in range(5):
            a, meta = z(WBC_BALANCE, x, tick=k, seq=k)
            assert not meta["fallback"] and np.isnan(meta["dev_us"])
            assert np.array_equal(a, ort1(WBC_BALANCE, x))
        keys = {r[1] for r in srv.requests if r[0] == "gear_wbc_action"}
        assert keys == {("obs", "policy")}
    finally:
        z.close()


def test_zmq_construction_checks(fake, ort1, monkeypatch):
    monkeypatch.setattr(wb.ZmqWbc, "CONNECT_TIMEOUT_MS", 200)
    srv = fake()
    srv.mode = "wrong"
    with pytest.raises(RuntimeError, match="mismatches ORT"):
        wb.ZmqWbc(srv.url, ort1)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        wb.ZmqWbc(f"tcp://127.0.0.1:{dead}", ort1)
    assert time.monotonic() - t0 < 1.0


def test_runtime_with_zmq_primary_and_shadow(fake, ort1):
    srv = fake()
    z = wb.ZmqWbc(srv.url, ort1)
    try:
        dump = []
        rt = wb.WbcRuntime(primary=z, shadow=ort1, dump=dump)
        rng = np.random.default_rng(12)
        seqs = []
        for k in range(4):
            rt.tick = 40 + k
            x = unit_obs(rng, 1)[0]
            a = rt.call(WBC_WALK, x)
            seqs.append(srv.requests[-1][2]["seq"])
            c = rt.last
            assert c.tick == 40 + k and c.policy == WBC_WALK and c.obs is x
            assert c.backend == WbcBackend.OTHER and not c.fallback and c.shadow_da == 0.0
            assert c.dev_us == 12.5 and c.rtt_us > 0 and np.array_equal(c.action, a)
        assert np.all(np.diff(seqs) == 1)
        assert [d[0] for d in dump] == [40, 41, 42, 43] and rt.n_calls[WBC_WALK] == 4
        srv.mode = "silent"
        rt.call(WBC_WALK, x)
        assert rt.last.fallback and rt.last.backend == WbcBackend.ORT and rt.last.shadow_da == 0.0
    finally:
        z.close()


def test_zmq_honours_the_servers_parity_tol(fake, ort1, monkeypatch):
    """x280d reports parity_tol (fp16 1e-2, fp32 5e-5; T-3.1): fp16 weights differ from ORT by ~7e-3 by design."""
    monkeypatch.setattr(wb.ZmqWbc, "CONNECT_TIMEOUT_MS", 500)
    srv = fake(info_extra={"parity_tol": 1e-2}, offset=5e-3)
    srv.mode = "wrong"
    z = wb.ZmqWbc(srv.url, ort1)
    try:
        assert z.parity_tol == 1e-2 and 4e-3 < z.parity_max_err < 6e-3
        assert z.info()["parity_tol_used"] == 1e-2 and z.info()["parity_tol"] == 1e-2
    finally:
        z.close()
    strict = fake(info_extra={"parity_tol": 5e-5}, offset=1e-4)  # fp32: the server's own bound is stricter
    strict.mode = "wrong"
    with pytest.raises(RuntimeError, match=r"tol 5e-05"):
        wb.ZmqWbc(strict.url, ort1)
    loose = fake(info_extra={"parity_tol": 1.0}, offset=5e-2)  # never looser than MAX_PARITY_TOL
    loose.mode = "wrong"
    with pytest.raises(RuntimeError, match=r"tol 0\.01"):
        wb.ZmqWbc(loose.url, ort1)
    for bad in ("1e-2", float("nan"), -1.0, 0, True, None, [1e-2]):
        assert wb.ZmqWbc._parity_tol({"parity_tol": bad}) == wb.ZmqWbc.PARITY_TOL, bad
    assert wb.ZmqWbc._parity_tol({}) == wb.ZmqWbc.PARITY_TOL


def test_shadow_client_has_no_fallback(fake, ort1, monkeypatch):
    monkeypatch.setattr(wb.ZmqWbc, "CONNECT_TIMEOUT_MS", 500)
    with pytest.raises(ValueError, match="fallback or a reference"):
        wb.ZmqWbc("tcp://127.0.0.1:1", None)
    srv = fake()
    z = wb.ZmqWbc(srv.url, None, reference=ort1, timeout_ms=5)
    try:
        assert z.fallback is None and z.reference is ort1 and z.info()["fallback"] is False
        x = unit_obs(np.random.default_rng(21), 1)[0]
        a, meta = z(WBC_BALANCE, x, 1, 1)
        assert not meta["fallback"] and np.array_equal(a, ort1(WBC_BALANCE, x))
        srv.mode = "silent"
        for i in range(3):
            a, meta = z(WBC_BALANCE, x, 2 + i, 2 + i)
            assert a is None and meta["fallback"] and meta["error"] == "timeout"
        assert z.latched  # a dead shadow costs at most three timeouts per episode
        t0 = time.perf_counter()
        a, meta = z(WBC_BALANCE, x, 9, 9)
        assert a is None and meta["error"] == "latched" and time.perf_counter() - t0 < 1e-3
    finally:
        z.close()


def test_runtime_with_ort_primary_and_remote_shadow(fake, ort1, monkeypatch):
    """T-3.6: --wbc ort --shadow URL. The action is always ORT's; shadow_da = |a_ort - a_remote|; a miss is NaN."""
    monkeypatch.setattr(wb.ZmqWbc, "CONNECT_TIMEOUT_MS", 500)
    srv = fake(info_extra={"parity_tol": 1e-2}, offset=2e-3)
    z = wb.ZmqWbc(srv.url, None, reference=ort1, timeout_ms=5)
    try:
        rt = wb.WbcRuntime(primary=ort1, shadow=z)
        rng = np.random.default_rng(22)
        x = unit_obs(rng, 1)[0]
        rt.tick = 5
        a = rt.call(WBC_WALK, x)
        assert np.array_equal(a, ort1(WBC_WALK, x)) and rt.last.shadow_da == 0.0
        assert rt.last.backend == WbcBackend.ORT and not rt.last.fallback
        assert srv.requests[-1][2]["tick"] == 5  # the shadow gets the tick and seq the primary used
        srv.mode = "wrong"
        rt.call(WBC_WALK, x)
        assert abs(rt.last.shadow_da - 2e-3) < 1e-4
        srv.mode = "silent"
        a = rt.call(WBC_WALK, x)
        assert np.isnan(rt.last.shadow_da) and np.array_equal(a, ort1(WBC_WALK, x))
        assert rt.n_shadow == 3 and rt.n_shadow_miss == 1
    finally:
        z.close()


# --- The D5 patch -------------------------------------------------------------------------------


def test_install_patch_unit(ort1, monkeypatch, tmp_path):
    import torch
    from gr00t_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy

    original = G1GearWbcPolicy.__dict__["load_onnx_policy"]
    monkeypatch.setenv("GR00T_WBC_SERVER", "127.0.0.1:1")     # must be bypassed, not dialled
    rt = wb.WbcRuntime(primary=ort1, dump=[])
    wb.install(rt)
    wb.install(rt)                                            # idempotent
    try:
        assert G1GearWbcPolicy.__dict__["load_onnx_policy"] is wb._patched_load_onnx_policy
        dummy = object.__new__(G1GearWbcPolicy)
        f = dummy.load_onnx_policy(f"{wb.ONNX_DIR}/{WBC_WALK}.onnx")
        assert f.rt_stem == WBC_WALK
        x = torch.randn(1, WBC_OBS_DIM)
        rt.tick = 7
        out = f(x)
        assert out.dtype == torch.float32 and out.shape == (1, 15)
        assert np.array_equal(out.numpy(), ort1(WBC_WALK, x.numpy()))
        assert rt.last.tick == 7 and rt.last.policy == WBC_WALK and np.isnan(rt.last.shadow_da)
        x.add_(1.0)                                           # the policy reuses its buffer in place
        assert not np.array_equal(rt.last.obs, x.numpy()) and len(rt.dump) == 1
        # An env .onnx that is not the file the in-process backend loaded is refused.
        other = tmp_path / "onnx"
        other.mkdir()
        shutil.copy(f"{wb.ONNX_DIR}/{WBC_WALK}.onnx", other / f"{WBC_BALANCE}.onnx")
        shutil.copy(f"{wb.ONNX_DIR}/{WBC_WALK}.onnx", other / f"{WBC_WALK}.onnx")
        wb.install(wb.WbcRuntime(primary=wb.OrtCpu(str(other))))
        with pytest.raises(RuntimeError, match="differs"):
            dummy.load_onnx_policy(f"{wb.ONNX_DIR}/{WBC_BALANCE}.onnx")
    finally:
        wb.uninstall()
    assert G1GearWbcPolicy.__dict__["load_onnx_policy"] is original and not wb.installed()
    wb.uninstall()                                            # idempotent too


# --- sim: the patched env is bit-identical to the stock one -------------------------------------


def make_g1_env():
    """The env as rollout_policy.get_groot_locomanip_env_fn builds it (without SyncVectorEnv)."""
    import gymnasium as gym
    from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401
    from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
    from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
    import robocasa  # noqa: F401

    env = gym.make("gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc", onscreen=False,
                   offscreen=True, enable_waist=True, randomize_cameras=False,
                   camera_names=["robot0_oak_egoview", "robot0_rs_tppview"])
    return WholeBodyControlWrapper(env, BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict())


@pytest.fixture(scope="module")
def g1_env():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "GR00T_WBC_SERVER" not in os.environ
    env = make_g1_env()
    yield env
    env.close()


WALK_TICKS = range(80, 140)  # nav 0.3 m/s: exercises the Walk policy


def rollout(env, n: int, seed: int, runtime=None) -> np.ndarray:
    """q43 after each of n steps of a hold action (nav on in WALK_TICKS)."""
    obs, _ = env.reset(seed=seed)
    qs = []
    for k in range(n):
        row = schema.hold_row(obs)
        if k in WALK_TICKS:
            row[schema.NAV] = [0.3, 0.0, 0.0]
        if runtime is not None:
            runtime.tick = k
        obs, _, _, _, _ = env.step(schema.row_to_env_action(row))
        qs.append(np.array(obs["q"], copy=True))
    return np.stack(qs)


def _lower(env):
    return env.wbc_policy.lower_body_policy


@pytest.mark.sim
def test_install_bit_identical_in_real_env(g1_env, capsys):
    from gr00t_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy

    from rtserve import wbc_numpy_ref

    original = G1GearWbcPolicy.__dict__["load_onnx_policy"]
    wb.uninstall()
    seed, n = 7, 200
    q_ref = rollout(g1_env, n, seed)                          # stock: the policy's own ORT sessions
    assert not hasattr(_lower(g1_env).policy_1, "rt_stem")

    dump = []
    rt = wb.WbcRuntime(primary=wb.OrtCpu("default"), dump=dump)
    wb.install(rt)
    try:
        q_pat = rollout(g1_env, n, seed, rt)                  # env.reset() re-created the policy
        lower = _lower(g1_env)
        assert lower.policy_1.rt_stem == WBC_BALANCE and lower.policy_2.rt_stem == WBC_WALK
        assert np.array_equal(q_pat, q_ref), np.abs(q_pat - q_ref).max()
        expect = [WBC_WALK if k in WALK_TICKS else WBC_BALANCE for k in range(n)]
        assert [d[0] for d in dump] == list(range(n)) and [d[1] for d in dump] == expect
        assert rt.n_calls == {WBC_BALANCE: n - len(WALK_TICKS), WBC_WALK: len(WALK_TICKS)}
        assert rt.last.tick == n - 1 and rt.last.backend == WbcBackend.ORT

        # The oracle, and ORT single vs default, on real observations of both policies.
        refs = wbc_numpy_ref.load_dir(wb.ONNX_DIR)
        single = wb.OrtCpu("single")
        for stem in WBC_POLICIES:
            obs = np.concatenate([d[2] for d in dump if d[1] == stem])
            act = np.concatenate([d[3] for d in dump if d[1] == stem])
            err = float(np.max(np.abs(act - refs[stem](obs))))
            d1 = max(float(np.max(np.abs(single(stem, o[None]) - a[None]))) for o, a in zip(obs, act))
            report(capsys, f"{len(obs)} real obs516 ({stem}): numpy f64 vs ORT max {err:.2e}; "
                           f"ORT single vs default max {d1:.2e}")
            assert err < 1e-5 and d1 < 1e-5

        # Survives the next reset (the policy is re-created again).
        before = sum(rt.n_calls.values())
        q2 = rollout(g1_env, 50, seed, rt)
        assert _lower(g1_env).policy_1.rt_stem == WBC_BALANCE
        assert np.array_equal(q2, q_ref[:50]) and sum(rt.n_calls.values()) == before + 50
    finally:
        wb.uninstall()
    assert G1GearWbcPolicy.__dict__["load_onnx_policy"] is original
    calls = sum(rt.n_calls.values())
    q3 = rollout(g1_env, 20, seed)
    assert not hasattr(_lower(g1_env).policy_1, "rt_stem") and sum(rt.n_calls.values()) == calls
    assert np.array_equal(q3, q_ref[:20])


@pytest.mark.sim
def test_loop_shadow_url_behind_ort(fake, ort1, tmp_path):
    """T-3.6 through rt_sim (VL mode, no RT loop): --wbc ort --shadow URL logs |a_ort - a_url| on every tick and
    never changes the action (the miss path, NaN and counted, is test_runtime_with_ort_primary_and_remote_shadow)."""
    import json

    import loop_utils as lu

    srv = fake(info_extra={"parity_tol": 1e-2}, offset=2e-3)
    srv.mode = "wrong"
    mock, url = lu.start_mock("detfn")
    try:
        run = lu.LoopRun(tmp_path, "shadow", "--mode", "vl", "--d-inj", "4", "--vla", url, "--wbc", "ort",
                         "--shadow", srv.url, "--shadow-timeout-ms", "20", "--seeds", "0", "--max-ticks", "120")
        d = run.finish(180)
    finally:
        lu.stop(mock)
    ticks = np.load(d / "ticks.npy")
    assert (ticks["wbc_be"] == WbcBackend.ORT).all() and not ticks["wbc_fb"].any()
    assert np.allclose(ticks["shadow_da"], 2e-3, atol=1e-4), ticks["shadow_da"][:5]
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["wbc_info"]["shadow"]["url"] == srv.url and cfg["wbc_info"]["shadow"]["parity_tol_used"] == 1e-2
    assert cfg["endpoints"]["shadow"] == srv.url
    sh = json.loads((d / "summary.json").read_text())["wbc"]["shadow"]
    assert sh["n_calls"] == len(ticks) and sh["n_miss"] == 0
