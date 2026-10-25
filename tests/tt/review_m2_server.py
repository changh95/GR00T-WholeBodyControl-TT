"""Adversarial review of the M2 CPU work (server_g1 / bench_g1), CPU only: ``import ttnn`` stays blocked.

    ttpy -m pytest -p no:cacheprovider $RT/tests/tt/review_m2_server.py -q

Each test states the behaviour the plan or the M1 contract asks for; a failing test is a demonstrated defect.
Servers bind 127.0.0.1 on random ports (never 5555-5572).
"""

import threading
import time

import torch
import zmq

from rtloop import schema, wire
from rttt import bench_g1 as B
from rttt import golden_io as gio
from rttt import server_g1 as S

RESERVED_PORTS = range(5555, 5573)


class Served:
    """A G1Server around an arbitrary model object, serving from a background thread."""

    def __init__(self, model, debug: bool = True, log_path=None):
        self.server = S.G1Server("127.0.0.1", 0, log_path, debug_endpoints=debug)
        assert self.server.port not in RESERVED_PORTS
        self.server.policy = S.G1Policy(model, {"label": "review"}, 0, fake=True)
        self.server.static_info = {"backend": "mock:review", "tree": "review"}
        self.url = self.server.addr
        self._stop = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.server.serve(lambda: self._stop, poll_ms=20)
        except BaseException as e:  # noqa: BLE001 - recorded for the test
            self.error = e

    def call(self, endpoint, data=None, timeout_ms=5000):
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self.url)
        try:
            s.send(wire.request(endpoint, data))
            if not s.poll(timeout_ms):
                return TimeoutError(endpoint)
            return wire.loads(s.recv())
        finally:
            s.close(linger=0)

    def stop(self):
        self._stop = True
        self.thread.join(5)
        self.server.close()


def golden_obs(sid: str) -> dict:
    return gio.load_obs(gio.GOLDEN_DIR / "obs" / f"{sid}.npz")


# --- 1. abandoned requests (schema.request_abandoned) -----------------------------------------------------------


def test_abandoned_request_is_skipped_like_the_mock():
    """mock_vla_server.py answers a request whose client already gave up (options.rt t_send_ns + give_up_ms before
    the receive stamp) with {"error": "stale ..."} and no inference, so requests abandoned during an outage do not
    each cost a full inference ahead of the live one. server_g1 must behave the same on the M2 path (same VlaClient,
    same give_up_ms)."""
    model = S.FakeG1Model()
    srv = Served(model, debug=False)
    try:
        now = time.monotonic_ns()
        rt = {"seq": 7, "gen": 0, "ep": 0, "k_obs": 0, "t_obs_ns": now - 2_000_000_000,
              "t_send_ns": now - 2_000_000_000, "give_up_ms": 500}
        assert schema.request_abandoned(rt, now)
        out = srv.call("get_action", {"observation": golden_obs("s00"), "options": {"rt": rt}})
        n_inferred = model.n_predict
    finally:
        srv.stop()
    assert isinstance(out, dict) and "stale" in str(out.get("error", "")), \
        f"server_g1 served an abandoned request: reply {type(out).__name__}, {n_inferred} inference(s) run"
    assert n_inferred == 0


# --- 2. T-2.4 ABAB with the default obs pair --------------------------------------------------------------------


class LagOneStateModel(S.FakeG1Model):
    """A stale-state bug of the kind T-2.4 exists to catch: the state used by a call is the PREVIOUS call's state
    (e.g. a trace input buffer written after execute_trace). The image reaches the model correctly."""

    def __init__(self):
        super().__init__()
        self._prev = None

    def predict_normalized(self, mi, noise):
        cur = mi.state.to(torch.float32).reshape(1, 1, -1).clone()
        used = cur if self._prev is None else self._prev
        self._prev = cur
        out = 0.5 * torch.tanh(noise.to(torch.float32))
        out[..., :43] += 0.1 * used[..., :43]
        out += 0.01 * mi.pixel_values.to(torch.float32).mean()
        self.n_predict += 1
        return out


def _abab_checks(obs_a, obs_b):
    srv = Served(LagOneStateModel(), debug=True)
    client = B.G1Client(srv.url, 20000)
    try:
        res = B.bench_abab(client, obs_a, obs_b, 8, B.fixed_noise(0))
    finally:
        client.close()
        srv.stop()
    return res, B.checks_of({"abab": res}, {})


def test_abab_default_pair_can_see_stale_state():
    """jobs/t2_abab.sh passes --obs $GOLDEN. T-2.4 must flag a stale-state bug, so the A/B pair bench_g1 picks from the
    golden (bench_g1.abab_pair, what main() uses for --abab) must differ in state; s00/s01 (both resets, bit-identical
    state) passed every T-2.4 check with the bug present before the fix (review:tt issue 1)."""
    obs_set = B.load_obs_set([str(gio.GOLDEN_DIR)])
    (na, a), (nb, b) = B.abab_pair(obs_set)
    same_state = all(a[k].tobytes() == b[k].tobytes() for k in schema.STATE_KEYS)
    res, checks = _abab_checks((na, a), (nb, b))
    passed = all(c["pass"] for c in checks)
    assert not passed, (f"T-2.4 passes with a lag-one stale-state model on the default pair {na}/{nb} "
                        f"(identical state: {same_state}): {[(c['name'], c['value']) for c in checks]}")


def test_abab_reset_vs_mid_pair_catches_stale_state():
    """The fix: A = a reset sample, B = a mid-episode one (different state and image) catches the same bug."""
    res, checks = _abab_checks(("s00", golden_obs("s00")), ("s10", golden_obs("s10")))
    assert not all(c["pass"] for c in checks), checks


# --- 3. a reply-side value that JSON cannot log kills the server ------------------------------------------------


def test_unloggable_rt_value_does_not_kill_the_server(tmp_path):
    """Every failure must come back as {"error"} and serving must continue (WP-2.3). _write_log runs after the reply
    and outside handle()'s try, so an options.rt value json.dumps cannot encode (bytes here) ends serve(). The real
    server always logs (default logs/server_g1/calls_*.jsonl; chip_lib.sh passes --log)."""
    srv = Served(S.FakeG1Model(), debug=False, log_path=tmp_path / "calls.jsonl")
    try:
        rt = {"seq": b"\x01", "gen": 0, "ep": 0, "k_obs": 0, "t_obs_ns": time.monotonic_ns()}
        first = srv.call("get_action", {"observation": golden_obs("s00"), "options": {"rt": rt}})
        time.sleep(0.2)
        ping = srv.call("ping", timeout_ms=2000)
        err = srv.error
    finally:
        srv.stop()
    assert isinstance(ping, dict) and ping.get("status") == "ok", \
        f"server stopped serving after one request (first reply {type(first).__name__}): {err!r}"
