"""T-2.1 (server half): rttt.server_g1 on MK with the fake model, CPU only (no ttnn, no device).

    ttpy -m pytest $RT/tests/tt/test_tt_server_protocol.py -q

The fake model runs the real G1 host pipeline (encode / decode through the MK port) around a deterministic
``predict_normalized``, so these tests cover the wire, the endpoints, the modality-key filter, noise seeding, error
replies, the debug-endpoint gate, the JSONL log, the CLI start-up order and default bind, the boot sanity check, and a
stock ``rtloop.vla_client.VlaClient`` round trip. Servers bind 127.0.0.1 on random ports (never 5555-5572).
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch
import zmq

from rtloop import schema, wire
from rtloop.schema import ChunkStatus
from rtloop.vla_client import VlaClient, check_modality_config
from rttt import golden_io as gio
from rttt import server_g1 as S

RT = Path(__file__).resolve().parents[2]
RESERVED_PORTS = range(5555, 5573)  # PLAN.md D15: never bound by tests

# The 13 keys the stock rollout sends beyond the 9 W-VLA keys (22 in all), as recorded by tests/official_arm.py's
# PolicyTap on the first request of seed 1 (scratchpad pt_gpu/official_equiv0/official_seed1.json "first_obs").
# PLAN.md §1.4.1 says 12; the recording has 13 (video.tpp_view included).
STOCK_EXTRA = {
    "ddq": ("float32", (1, 1, 43)), "dq": ("float32", (1, 1, 43)), "ego_view_image": ("float32", (1, 1, 480, 640, 3)),
    "floating_base_acc": ("float32", (1, 1, 6)), "floating_base_pose": ("float32", (1, 1, 7)),
    "floating_base_vel": ("float32", (1, 1, 6)), "q": ("float32", (1, 1, 43)), "tau_est": ("float32", (1, 1, 43)),
    "torso_ang_vel": ("float32", (1, 1, 3)), "torso_quat": ("float32", (1, 1, 4)),
    "tpp_view_image": ("float32", (1, 1, 480, 640, 3)), "video.tpp_view": ("uint8", (1, 1, 480, 640, 3)),
    "wrist_pose": ("float32", (1, 1, 14)),
}


def stock_obs(obs: dict) -> dict:
    out = dict(obs)
    rng = np.random.default_rng(7)
    for k, (dt, shape) in STOCK_EXTRA.items():
        out[k] = (rng.integers(0, 256, shape, dtype=np.uint8) if dt == "uint8"
                  else rng.standard_normal(shape).astype(np.float32))
    return out


def fake_args(**kw) -> argparse.Namespace:
    base = dict(fake_model=True, no_boot_check=False, cache_root=None, seed=0, golden=str(gio.GOLDEN_DIR),
                min_pcc=S.DEFAULT_MIN_PCC)
    base.update(kw)
    return argparse.Namespace(**base)


class Running:
    """A G1Server with the fake model, serving from a background thread."""

    def __init__(self, tmp_path: Path, debug: bool = False, seed: int = 0):
        self.server = S.G1Server("127.0.0.1", 0, tmp_path / "calls.jsonl", debug_endpoints=debug)
        assert self.server.port not in RESERVED_PORTS
        S.build(fake_args(seed=seed), self.server)
        self.url = self.server.addr
        self._stop = False
        self.thread = threading.Thread(target=self.server.serve, args=(lambda: self._stop,), kwargs={"poll_ms": 20},
                                       daemon=True)
        self.thread.start()
        self.ctx = zmq.Context()

    def call(self, endpoint: str, data: dict | None = None, timeout_ms: int = 10000):
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self.url)
        try:
            s.send(wire.request(endpoint, data))
            assert s.poll(timeout_ms), f"no reply to {endpoint}"
            return wire.loads(s.recv())
        finally:
            s.close(linger=0)

    def stop(self):
        self._stop = True
        self.thread.join(5)
        assert not self.thread.is_alive()
        self.server.close()
        self.ctx.destroy(linger=0)


@pytest.fixture
def srv(tmp_path):
    r = Running(tmp_path)
    yield r
    r.stop()


@pytest.fixture
def dbg(tmp_path):
    r = Running(tmp_path, debug=True)
    yield r
    r.stop()


def rt(seq: int, **kw) -> dict:
    d = {"seq": seq, "gen": 0, "ep": 0, "k_obs": 16 * seq, "t_obs_ns": time.monotonic_ns()}
    d.update(kw)
    return d


def get_action(r: Running, obs: dict, options=None):
    out = r.call("get_action", {"observation": obs, "options": options})
    assert isinstance(out, list) and len(out) == 2, out
    return out


# --- import / identity on MK -----------------------------------------------------------------------------------------


def test_imports_on_mk():
    import models.experimental.gr00t as pkg
    from rttt import g1_n16 as G

    assert "publish/worktree-mk" in pkg.__file__
    assert not (Path(pkg.__file__).parent / "libero").exists()  # MK has no libero/: the port lives in RT (R20)
    assert G.tree_stamp().startswith("MK@")
    assert "ttnn" not in sys.modules


def test_default_bind_is_localhost():
    a = S.make_parser().parse_args([])
    assert (a.host, a.port) == ("127.0.0.1", 5557)
    assert not a.debug_endpoints and not a.fake_model and a.min_pcc == 0.995
    assert a.cpus == "8,9,10,11" and a.torch_threads == 1  # PLAN.md §1.1 pin; one host-pipeline thread
    assert a.mk_arena_dtype == "auto"  # MK default (bfp8_b for n16); bf16 only as the WP-2.7 fallback


def test_real_path_builds_the_requested_policy(monkeypatch):
    """Without a device: the non-fake build asks load_tt_model for TTPolicy(mk_arena_dtype=...) (WP-2.7 switch)."""
    from rttt import g1_n16 as G

    seen = {}

    def fake_load(policy=None, cache_root=None, **kw):
        seen["policy"], seen["cache_root"] = policy, cache_root
        raise RuntimeError("stop before the device")

    monkeypatch.setattr(G, "load_tt_model", fake_load)
    s = S.G1Server(port=0)
    try:
        with pytest.raises(RuntimeError, match="stop before the device"):
            S.build(fake_args(fake_model=False, mk_arena_dtype="bf16"), s)
    finally:
        s.close()
    assert seen["policy"].mk_arena_dtype == "bf16" and seen["policy"].dit_backend == "megakernel"
    assert seen["cache_root"] is None  # -> g1_n16.default_cache_root()
    s = S.G1Server(port=0)
    try:
        assert s.addr.startswith("tcp://127.0.0.1:") and s.port not in RESERVED_PORTS
    finally:
        s.close()


# --- endpoints -------------------------------------------------------------------------------------------------------


def test_ping_info_modality(srv):
    assert srv.call("ping") == {"status": "ok", "message": "Server is running"}
    info = srv.call("info")
    assert info["backend"] == "mock:server_g1-fake" and info["proto"] == 2 and info["fake_model"] is True
    assert info["tree"].startswith("MK@") and info["model"].endswith("-bf16")
    assert info["boot_check"]["skipped"] == "fake model"
    mc = srv.call("get_modality_config")
    assert set(mc) == {"video", "state", "action", "language"}
    assert all("__ModalityConfig_class__" in v for v in mc.values())
    assert check_modality_config(mc) is True  # the loop's startup check against the 9-key request and A35


def test_vla_client_round_trip(srv, obs_factory):
    c = VlaClient(srv.url, give_up_ms=10000)
    try:
        assert c.ping(5000)
        assert c.check_modality_config()
        assert c.info()["backend"] == "mock:server_g1-fake"
        sent = rt(3, noise_seed=12345)
        c.send(schema.build_vla_obs({k: v[0, 0] if k != schema.LANGUAGE_KEY else v[0]
                                     for k, v in obs_factory(11).items()}), sent)
        rep = c.wait(10000)
        assert rep is not None and rep.status == ChunkStatus.OK, rep.error
        assert rep.rows.shape == (30, 35) and np.isfinite(rep.rows).all()
        info = rep.info
        assert info["rt"]["seq"] == 3 and info["rt"]["t_send_ns"] == rep.t_send_ns
        assert info["noise_seed"] == 12345 and info["backend"] == "mock:server_g1-fake"
        assert rep.t_send_ns <= info["t_recv_ns"] <= info["t_reply_ns"] <= rep.t_recv_ns
        for k in ("infer_ms", "encode_ms", "device_ms", "decode_ms"):
            assert info[k] >= 0.0, k
        assert info["infer_ms"] >= info["encode_ms"] + info["decode_ms"]
        assert info["tree"].startswith("MK@")
    finally:
        c.close()


def test_reply_shapes_and_options_none(srv, obs_factory):
    action, info = get_action(srv, obs_factory(12), None)  # what the stock rollout sends
    assert tuple(action) == tuple(f"action.{n}" for n, _ in schema.ACTION_GROUPS)
    for n, d in schema.ACTION_GROUPS:
        assert action[f"action.{n}"].shape == (1, 30, d) and action[f"action.{n}"].dtype == np.float32
    assert info["rt"] is None and info["noise_seed"] is None
    assert srv.call("reset", {"options": None}) == {}


def test_batch_of_two(srv, obs_factory):
    action, _ = get_action(srv, obs_factory(13, b=2), {"rt": rt(1, noise_seed=1)})
    assert action["action.navigate_command"].shape == (2, 30, 3)


def test_noise_seed_reproducible_and_fresh_otherwise(srv, obs_factory):
    obs = obs_factory(14)
    a1, _ = get_action(srv, obs, {"rt": rt(1, noise_seed=99)})
    a2, _ = get_action(srv, obs, {"rt": rt(2, noise_seed=99)})
    a3, _ = get_action(srv, obs, {"rt": rt(3, noise_seed=100)})
    b1, _ = get_action(srv, obs, {"rt": rt(4)})
    b2, _ = get_action(srv, obs, {"rt": rt(5)})
    for k in a1:
        assert a1[k].tobytes() == a2[k].tobytes(), k
    assert any(a1[k].tobytes() != a3[k].tobytes() for k in a1)
    assert any(b1[k].tobytes() != b2[k].tobytes() for k in b1)  # fresh noise per call without a seed


def test_key_filter_drops_the_stock_extra_keys(srv, obs_factory):
    obs = obs_factory(15)
    full = stock_obs(obs)
    assert len(full) == 22 and len(STOCK_EXTRA) == 13
    a_full, _ = get_action(srv, full, {"rt": rt(1, noise_seed=5)})
    a_nine, _ = get_action(srv, obs, {"rt": rt(2, noise_seed=5)})
    for k in a_nine:
        assert a_full[k].tobytes() == a_nine[k].tobytes(), k  # the extra keys change nothing
    info = srv.call("info")
    assert info["dropped_key_counts"] == {k: 1 for k in STOCK_EXTRA}


def test_key_filter_runs_before_split_batch(g1, obs_factory, monkeypatch):
    """split_batch must only ever see the 9 keys (an f32 camera copy would become a Python list)."""
    seen = []
    real = g1.split_batch
    monkeypatch.setattr(g1, "split_batch", lambda flat: seen.append(tuple(flat)) or real(flat))
    pol = S.G1Policy(S.FakeG1Model(), S.fake_stamp(), 0, fake=True)
    pol.predict(stock_obs(obs_factory(16)), None)
    assert seen == [schema.VLA_OBS_KEYS]


def test_errors_are_replies_and_serving_continues(srv, obs_factory):
    obs = obs_factory(17)
    cases = [
        ("kill", None, "Unknown endpoint"),
        ("nope", None, "Unknown endpoint"),
        ("get_action", {"observation": {k: v for k, v in obs.items() if k != "state.waist"}}, "lacks"),
        ("get_action", {"observation": dict(obs, **{"state.waist": obs["state.waist"].astype(np.float64)})}, "float32"),
        ("get_action", {"observation": dict(obs, **{schema.VIDEO_KEY: obs[schema.VIDEO_KEY][:, 0]})}, "uint8"),
        ("get_action", {"observation": dict(obs, **{schema.LANGUAGE_KEY: (" ".join(["apple"] * 40),)})},
         "exceeds L_max=24"),  # the static g1 layout's text budget
        ("get_action", {"bogus": 1}, "unexpected keyword"),
    ]
    for ep, data, needle in cases:
        out = srv.call(ep, data)
        assert isinstance(out, dict) and "error" in out and needle in out["error"], (ep, out)
    action, _ = get_action(srv, obs, {"rt": rt(1)})
    assert action["action.waist"].shape == (1, 30, 3)
    assert srv.call("info")["n_errors"] == len(cases)


def test_debug_endpoints_gated(srv, obs_factory):
    noise = np.zeros((1, 50, 128), np.float32)
    assert srv.call("predict_debug", {"observation": obs_factory(18), "noise": noise}) == \
        {"error": "debug endpoints disabled"}
    assert srv.call("x280_peek", {"x": 8, "y": 7, "addr": 0}) == {"error": "debug endpoints disabled"}


def test_predict_debug_matches_get_action(dbg, g1, obs_factory):
    obs = obs_factory(19)
    noise = g1.make_noise(4242)  # = the generator get_action reseeds from options.rt.noise_seed = 4242
    out = dbg.call("predict_debug", {"observation": obs, "noise": noise.numpy()})
    assert out["action_pred_norm"].shape == (1, 50, 128) and out["action_pred_norm"].dtype == np.float32
    kept, _ = g1.filter_obs(obs)
    mi, _ = g1.encode_obs(g1.split_batch(kept)[0])
    want = S.FakeG1Model().predict_normalized(mi, noise).numpy()
    assert np.array_equal(out["action_pred_norm"], want)
    action, _ = get_action(dbg, obs, {"rt": rt(1, noise_seed=4242)})
    for k in action:
        assert out["actions"][k].tobytes() == action[k].tobytes(), k
    for k in ("encode_ms", "device_ms", "decode_ms"):
        assert out[k] >= 0
    bad = dbg.call("predict_debug", {"observation": obs, "noise": noise.numpy().astype(np.float64)})
    assert "error" in bad and "float32" in bad["error"]
    peek = dbg.call("x280_peek", {"x": 8, "y": 7, "addr": 0})
    assert "error" in peek and "real device" in peek["error"]


def test_gpu_vla_server_cpu_noise_is_server_g1s_draw(tmp_path, obs_factory, monkeypatch):
    """verify:all D3 (review:tt #8): for the loop's --crn noise_seeds, the noise server_g1 hands its model (the fake
    model, captured at predict_normalized) is bit-identical to rtserve/gpu_vla_server.py --noise cpu's draw for the
    N1.6 action head's one ``torch.randn(size=(B, action_horizon, max_action_dim), dtype=<model dtype>)``: equal in
    fp32 before the cast and equal after the cast to the dtype the DiT asks for (shape and dtype from the DP1
    checkpoint's config.json, gr00t_n1d6.py:310-314). server_g1 reports the mode as info.noise."""
    from rtloop.loop import noise_seed
    from rtserve import gpu_vla_server as gv
    from rttt.convert_bf16 import DEFAULT_SRC

    cfg_path = Path(DEFAULT_SRC) / "config.json"
    if not cfg_path.exists():
        pytest.skip(f"no DP1 snapshot at {DEFAULT_SRC}")
    cfg = json.loads(cfg_path.read_text())
    shape, dtype = (1, cfg["action_horizon"], cfg["max_action_dim"]), getattr(torch, cfg["model_dtype"])
    assert shape == S.FakeG1Model().noise_shape and dtype == torch.bfloat16
    seen = []
    orig = S.FakeG1Model.predict_normalized

    def capture(self, mi, noise):
        seen.append(noise.detach().clone())
        return orig(self, mi, noise)

    monkeypatch.setattr(S.FakeG1Model, "predict_normalized", capture)
    r = Running(tmp_path)
    draws = []
    try:
        assert r.call("info")["noise"] == gv.NOISE_MODES[1] == "cpu"
        for i, seed in enumerate([noise_seed(2000, k) for k in range(3)] + [0, 2**64 - 1]):
            seen.clear()
            get_action(r, obs_factory(30 + i), {"rt": rt(i + 1, noise_seed=seed)})
            (tt,) = seen
            assert tuple(tt.shape) == shape and tt.dtype == torch.float32
            with gv.SeededNoise(seed) as sn:
                head = torch.randn(size=shape, dtype=dtype, device="cpu")  # the action head's call, on cpu here
            with gv.SeededNoise(seed):
                fp32 = torch.randn(shape)
            assert sn.calls == 1 and head.dtype == dtype
            assert torch.equal(fp32, tt) and fp32.numpy().tobytes() == tt.numpy().tobytes(), seed
            assert torch.equal(head, tt.to(dtype)), seed
            draws.append(tt)
        assert not any(torch.equal(a, b) for i, a in enumerate(draws) for b in draws[i + 1:])  # seed -> draw
    finally:
        r.stop()


def test_jsonl_log(tmp_path, obs_factory):
    r = Running(tmp_path)
    try:
        get_action(r, obs_factory(20), {"rt": rt(7, noise_seed=3)})
        r.call("nope")
        path = tmp_path / "calls.jsonl"
        for _ in range(100):  # written just after the reply is sent
            lines = path.read_text().splitlines() if path.exists() else []
            if len(lines) >= 2:
                break
            time.sleep(0.02)
        recs = [json.loads(l) for l in lines]
        ok = recs[0]
        assert ok["endpoint"] == "get_action" and ok["seq"] == 7 and ok["noise_seed"] == 3 and ok["B"] == 1
        for k in ("encode_ms", "device_ms", "decode_ms", "infer_ms", "wire_in_ms", "wire_out_ms", "t_recv_ns",
                  "t_reply_ns", "seq_len"):
            assert k in ok, k
        assert ok["seq_len"] == 117
        assert recs[1]["endpoint"] == "nope" and "Unknown endpoint" in recs[1]["error"]
    finally:
        r.stop()


# --- boot sanity check -----------------------------------------------------------------------------------------------


def _synthetic_golden(root: Path, g1, obs_factory, corrupt: bool = False) -> None:
    (root / "obs").mkdir(parents=True)
    (root / "samples").mkdir()
    index = [{"sid": "r00", "kind": "reset"}, {"sid": "m00", "kind": "mid"}]
    (root / "obs_index.json").write_text(json.dumps({"samples": index}))
    model = S.FakeG1Model()
    for i, s in enumerate(index):
        obs = obs_factory(30 + i)
        gio.save_obs(root / "obs" / f"{s['sid']}.npz", obs)
        noise = g1.make_noise(50 + i)
        mi, _ = g1.encode_obs(g1.split_batch(g1.filter_obs(obs)[0])[0])
        pred = model.predict_normalized(mi, noise).numpy()
        if corrupt and s["kind"] == "mid":
            pred = np.random.default_rng(0).standard_normal(pred.shape).astype(np.float32)
        np.savez(root / "samples" / f"{s['sid']}.npz", noise=noise.numpy(), **{"bf16.action_pred_norm": pred})


def test_boot_sanity_check(tmp_path, g1, obs_factory):
    _synthetic_golden(tmp_path / "ok", g1, obs_factory)
    pol = S.G1Policy(S.FakeG1Model(), S.fake_stamp(), 0, fake=True)
    out = S.boot_sanity_check(pol, tmp_path / "ok", 0.995)
    assert out["pass"] and out["samples"] == ["r00", "m00"] and min(out["pcc"]) > 0.999999
    _synthetic_golden(tmp_path / "bad", g1, obs_factory, corrupt=True)
    with pytest.raises(RuntimeError, match="boot sanity check failed"):
        S.boot_sanity_check(pol, tmp_path / "bad", 0.995)


def test_real_golden_has_the_boot_keys(golden_dir):
    if not (golden_dir / "obs_index.json").is_file() or not (golden_dir / "samples").is_dir():
        pytest.skip(f"G1 goldens not written yet under {golden_dir} (WP-2.2 make_g1_golden.py)")
    sids = S.boot_samples(golden_dir)
    assert len(sids) == 2
    for sid in sids:
        s = gio.load_sample(golden_dir, sid)
        assert s["noise"].shape == (1, 50, 128) and s["noise"].dtype == np.float32
        assert s["bf16.action_pred_norm"].shape == (1, 50, 128)
        assert tuple(s["obs"]) == schema.VLA_OBS_KEYS


# --- CLI -------------------------------------------------------------------------------------------------------------


def _spawn(*args: str, timeout_s: float = 60.0):
    p = subprocess.Popen([sys.executable, "-m", "rttt.server_g1", *args], cwd=str(RT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    lines, ready = [], None
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        line = p.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip())
        if line.startswith("server_g1 READY"):
            ready = line.split()[2]
            break
    return p, lines, ready


def test_cli_fake_server_start_order_and_shutdown(tmp_path, obs_factory):
    p, lines, url = _spawn("--fake-model", "--port", "0", "--cpus", "", "--log", str(tmp_path / "c.jsonl"),
                           "--max-idle-s", "60")
    try:
        assert url is not None, "\n".join(lines)
        assert url.startswith("tcp://127.0.0.1:") and int(url.rsplit(":", 1)[1]) not in RESERVED_PORTS
        i_bound = next(i for i, l in enumerate(lines) if "] bound tcp://127.0.0.1:" in l)
        i_loaded = next(i for i, l in enumerate(lines) if "] loaded in" in l)
        assert i_bound < i_loaded  # the port is held before the model (and so the device) is loaded
        c = VlaClient(url, give_up_ms=10000)
        try:
            assert c.ping(5000) and c.check_modality_config()
            assert c.info()["torch_threads"] == 1
            c.send(schema.build_vla_obs({k: v[0, 0] if k != schema.LANGUAGE_KEY else v[0]
                                         for k, v in obs_factory(21).items()}), rt(1))
            rep = c.wait(10000)
            assert rep is not None and rep.status == ChunkStatus.OK, rep.error
        finally:
            c.close()
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=20)
        assert p.returncode == 0, out
        assert "model released" in out
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()


def test_cli_busy_port_fails_before_loading():
    ctx = zmq.Context()
    s = ctx.socket(zmq.REP)
    port = s.bind_to_random_port("tcp://127.0.0.1")
    try:
        p = subprocess.run([sys.executable, "-m", "rttt.server_g1", "--fake-model", "--port", str(port),
                            "--cpus", "", "--log", "none"], cwd=str(RT), capture_output=True, text=True, timeout=60)
        assert p.returncode == 3, p.stdout + p.stderr
        assert "bind" in p.stdout and "loaded in" not in p.stdout
    finally:
        s.close(linger=0)
        ctx.term()


def test_cli_boot_check_failure_exits_2(tmp_path):
    """The real path without goldens: no READY, exit 2 (exercised with the fake model forced through the check)."""
    p = subprocess.run([sys.executable, "-c",
                        "import sys; from rttt import server_g1 as S; "
                        "S.build_orig = S.build\n"
                        "def b(a, s):\n"
                        "    a.fake_model = True; S.build_orig(a, s)\n"
                        "    S.boot_sanity_check(s.policy, a.golden, a.min_pcc)\n"
                        "S.build = b\n"
                        f"sys.exit(S.main(['--port', '0', '--cpus', '', '--log', 'none', '--golden', '{tmp_path}/none']))"],
                       cwd=str(RT), capture_output=True, text=True, timeout=60)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "READY" not in p.stdout and "start-up failed" in p.stdout


# --- abandoned requests, logging robustness (review:tt fixes) ------------------------------------------------------


def test_abandoned_request_is_answered_stale_without_inference(srv, obs_factory):
    """As rtserve/gpu_vla_server.py:120-128: a request received after the client's give-up gets {"error": "stale"}
    and runs no inference; a fresh request with the same stamps is served."""
    now = time.monotonic_ns()
    old = rt(5, t_send_ns=now - 2_000_000_000, give_up_ms=500)
    assert schema.request_abandoned(old, now)
    n0 = srv.server.policy.n_calls
    out = srv.call("get_action", {"observation": obs_factory(40), "options": {"rt": old}})
    assert isinstance(out, dict) and "stale request seq 5" in out["error"] and "500 ms" in out["error"]
    assert srv.server.policy.n_calls == n0
    fresh = rt(6, t_send_ns=time.monotonic_ns(), give_up_ms=500)
    action, info = get_action(srv, obs_factory(40), {"rt": fresh})
    assert info["rt"]["seq"] == 6 and action["action.waist"].shape == (1, 30, 3)
    info = srv.call("info")
    assert info["n_stale"] == 1 and info["n_errors"] == 0  # a stale skip is not a server error
    recs = [json.loads(l) for l in (Path(srv.server.log_path)).read_text().splitlines()]
    assert recs[0]["stale"] is True and recs[0]["seq"] == 5 and "infer_ms" not in recs[0]


def test_unloggable_rt_values_are_logged_as_text(srv, obs_factory):
    out = get_action(srv, obs_factory(41), {"rt": rt(b"\x01", gen=b"g", ep=0)})  # bytes: json cannot encode them
    assert isinstance(out, list)
    assert srv.call("ping") == {"status": "ok", "message": "Server is running"}
    for _ in range(100):
        lines = Path(srv.server.log_path).read_text().splitlines()
        if lines:
            break
        time.sleep(0.02)
    rec = json.loads(lines[0])
    assert rec["seq"] == "b'\\x01'" and srv.call("info")["n_log_errors"] == 0


def test_log_write_failure_is_counted_and_serving_continues(srv, obs_factory):
    srv.server._fh.close()  # every later write raises ValueError: I/O operation on closed file
    get_action(srv, obs_factory(42), {"rt": rt(1)})
    get_action(srv, obs_factory(42), {"rt": rt(2)})
    time.sleep(0.1)
    info = srv.call("info")
    assert info["n_log_errors"] == 2 and info["n_calls"] == 2
    srv.server._fh = None  # nothing left to close in stop()


class LagOneStateModel(S.FakeG1Model):
    """The stale-state bug T-2.4 exists for: each call uses the previous call's state (e.g. a trace input buffer
    written after execute_trace); the image reaches the model correctly."""

    def __init__(self):
        super().__init__()
        self._prev = None

    def predict_normalized(self, mi, noise):
        cur = mi.state.to(torch.float32).reshape(1, 1, -1).clone()
        used, self._prev = (cur if self._prev is None else self._prev), cur
        out = 0.5 * torch.tanh(noise.to(torch.float32))
        out[..., :43] += 0.1 * used[..., :43]
        out += 0.01 * mi.pixel_values.to(torch.float32).mean()
        self.n_predict += 1
        return out


def test_t24_default_pair_catches_a_lag_one_state_bug(tmp_path, golden_dir):
    """review:tt issue 1: with the A/B pair bench_g1 picks from the golden (s00 reset, s10 mid), T-2.4 fails a server
    whose state input lags one call; the old pair s00/s01 (identical state) passed it."""
    if not (golden_dir / "obs_index.json").is_file():
        pytest.skip("G1 goldens not written yet")
    from rttt import bench_g1 as B

    server = S.G1Server("127.0.0.1", 0, None, debug_endpoints=True)
    server.policy = S.G1Policy(LagOneStateModel(), {"label": "lag-one"}, 0, fake=True)
    server.static_info = {"backend": "mock:lag-one", "tree": "test"}
    stop = {"flag": False}
    t = threading.Thread(target=server.serve, args=(lambda: stop["flag"],), kwargs={"poll_ms": 20}, daemon=True)
    t.start()
    client = B.G1Client(server.addr, 20000)
    try:
        obs_set = B.load_obs_set([str(golden_dir)])
        a, b = B.abab_pair(obs_set)
        assert (a[0], b[0]) == ("s00", "s10")
        # the old pair first, on the fresh model (the lagged state carries over from one bench to the next)
        old = B.checks_of({"abab": B.bench_abab(client, obs_set[0], obs_set[1], 8, B.fixed_noise(0))}, {})
        bad = B.checks_of({"abab": B.bench_abab(client, a, b, 8, B.fixed_noise(0))}, {})
    finally:
        client.close()
        stop["flag"] = True
        t.join(5)
        server.close()
    assert not all(c["pass"] for c in bad), bad
    assert all(c["pass"] for c in old)  # the reason the pair rule exists


# --- x280_lim_write (LIM init method B) ---------------------------------------------------------------------------


def _lim_args(**kw):
    from x280 import regs

    x, y = regs.L2CPU_XY[3]
    base = {"x": x, "y": y, "addr": regs.LIM_BASE + 0x10_0000, "data": bytes(range(256)) * 16, "chunk": 64}
    base.update(kw)
    return base


def test_lim_write_gated_and_needs_the_device(srv, dbg):
    assert srv.call("x280_lim_write", _lim_args()) == {"error": "debug endpoints disabled"}
    out = dbg.call("x280_lim_write", _lim_args())
    assert "error" in out and "real device" in out["error"]


def test_lim_write_64b_lines_through_write_to_core(dbg):
    from x280 import regs

    seen = []
    dbg.server.noc_write = lambda dev, x, y, addr, data: seen.append((dev, x, y, addr, bytes(data)))
    args = _lim_args()
    out = dbg.call("x280_lim_write", args)
    assert out["n_bytes"] == 4096 and out["n_writes"] == 64, out
    assert [s[3] for s in seen] == [args["addr"] + 64 * i for i in range(64)]
    assert all(len(s[4]) == 64 and s[4] == args["data"][64 * i: 64 * i + 64] for i, s in enumerate(seen))
    assert {s[:3] for s in seen} == {(0, *regs.L2CPU_XY[3])}
    seen.clear()  # the last line of LIM and the spare tile (DP4) are writable too
    assert "error" not in dbg.call("x280_lim_write", _lim_args(addr=regs.LIM_END - 64, data=bytes(64)))
    assert "error" not in dbg.call("x280_lim_write", _lim_args(x=regs.L2CPU_XY[2][0], y=regs.L2CPU_XY[2][1],
                                                               addr=regs.LIM_BASE, data=bytes(64)))
    assert [s[3] for s in seen] == [regs.LIM_END - 64, regs.LIM_BASE]


@pytest.mark.parametrize("kw,needle", [
    ({"addr": "end-64", "data": bytes(128)}, "outside LIM"),
    ({"addr": "base-64", "data": bytes(64)}, "outside LIM"),
    ({"addr": "base+4"}, "aligned"),
    ({"chunk": 4}, "chunk must be 64"),
    ({"data": bytes(100)}, "whole 64 B lines"),
    ({"data": b""}, "whole 64 B lines"),
    ({"data": bytes(128 * 1024)}, "whole 64 B lines"),
    ({"data": "text"}, "data must be bytes"),
    ({"xy": "translated3"}, "read-only"),
    ({"xy": "tile0"}, "read-only"),
    ({"xy": "arc"}, "refuses"),
    ({"xy": (1, 2)}, "not an allowed tile"),
])
def test_lim_write_refusals_write_nothing(dbg, kw, needle):
    from x280 import regs

    seen = []
    dbg.server.noc_write = lambda *a: seen.append(a)
    kw = dict(kw)
    addr = {"end-64": regs.LIM_END - 64, "base-64": regs.LIM_BASE - 64, "base+4": regs.LIM_BASE + 4}
    if "addr" in kw:
        kw["addr"] = addr[kw["addr"]]
    xy = kw.pop("xy", None)
    if xy is not None:
        xy = {"translated3": regs.L2CPU_XY_TRANSLATED[3], "tile0": regs.L2CPU_XY[0], "arc": regs.ARC_XY}.get(xy, xy)
        kw.update(x=xy[0], y=xy[1], addr=regs.LIM_BASE)
    out = dbg.call("x280_lim_write", _lim_args(**kw))
    assert "error" in out and needle in out["error"], out
    assert seen == []


# --- status stream (PLAN.md §4.3 item 7) --------------------------------------------------------------------------


STATUS_KEYS = {"src", "pid", "t_ns", "seq", "backend", "uptime_s", "ready", "calls", "errors", "stale", "last_ms",
               "p50_ms", "device_duty", "trace", "cq", "core_map", "tree", "telemetry"}


class Sub:
    def __init__(self, url: str):
        self.ctx = zmq.Context()
        self.s = self.ctx.socket(zmq.SUB)
        self.s.setsockopt(zmq.SUBSCRIBE, b"status")
        self.s.setsockopt(zmq.LINGER, 0)
        self.s.connect(url)

    def take(self, n: int, timeout_s: float = 10.0) -> list:
        out, t_end = [], time.monotonic() + timeout_s
        while len(out) < n and time.monotonic() < t_end:
            if self.s.poll(200):
                topic, payload = self.s.recv_multipart()
                assert topic == b"status"
                out.append((time.monotonic(), wire.loads(payload)))
        return out

    def close(self):
        self.s.close(0)
        self.ctx.term()


def test_status_stream_contract(srv, obs_factory):
    srv.server.t_ready_ns = time.monotonic_ns()
    pub = S.StatusPublisher(srv.server, "0", 10).start()
    sub = Sub(pub.addr)
    try:
        assert pub.addr.startswith("tcp://127.0.0.1:") and int(pub.addr.rsplit(":", 1)[1]) not in RESERVED_PORTS
        first = sub.take(2)
        get_action(srv, obs_factory(43), {"rt": rt(1)})
        get_action(srv, obs_factory(44), {"rt": rt(2, noise_seed=7)})
        later = sub.take(4)
    finally:
        sub.close()
        pub.close()
    msgs = [m for _, m in first + later]
    assert len(msgs) == 6 and all(set(m) == STATUS_KEYS for m in msgs)
    info = srv.call("info")
    m0, m = msgs[0], msgs[-1]
    assert m["src"] == "server_g1" and m["pid"] == os.getpid() and m["backend"] == info["backend"]
    assert m["tree"] == info["tree"] and m["ready"] is True and m["telemetry"] is None
    assert [x["seq"] for x in msgs] == sorted({x["seq"] for x in msgs})
    assert all(b["t_ns"] > a["t_ns"] for a, b in zip(msgs, msgs[1:]))
    assert m0["calls"] == 0 and m0["last_ms"] is None and m0["p50_ms"] is None and m0["device_duty"] == 0.0
    assert m["calls"] == 2 and set(m["last_ms"]) == set(m["p50_ms"]) == {"encode", "device", "decode", "total"}
    assert m["last_ms"]["total"] >= m["last_ms"]["encode"] + m["last_ms"]["decode"]
    assert 0.0 <= m["device_duty"] <= 1.0 and m["errors"] == 0 and m["stale"] == 0
    assert m["trace"] is False and m["cq"] is None and m["core_map"] is None  # fake model: no trace, no device
    span = later[-1][0] - first[0][0]
    assert 2.0 <= (len(msgs) - 1) / span <= 12.0  # 10 Hz requested


def test_status_before_ready_and_rate_clamp(tmp_path):
    server = S.G1Server("127.0.0.1", 0, None)
    fast, slow = S.StatusPublisher(server, "0", 1000.0), S.StatusPublisher(server, "0", 0.1)
    try:
        assert abs(fast.period - 0.1) < 1e-9 and abs(slow.period - 0.5) < 1e-9  # clamped to 2-10 Hz
        m = server.status_msg(1)
        assert m["ready"] is False and m["backend"] is None and m["device_duty"] is None and m["calls"] == 0
    finally:
        fast.close()
        slow.close()
        server.close()


def test_status_pub_never_blocks_on_a_stalled_subscriber(srv, obs_factory):
    pub = S.StatusPublisher(srv.server, "0", 10).start()
    stalled = Sub(pub.addr)
    try:
        time.sleep(1.5)  # ~15 messages against a reader that never reads: SNDHWM 10 drops, never blocks
        t0 = time.perf_counter()
        get_action(srv, obs_factory(45), {"rt": rt(1)})
        assert time.perf_counter() - t0 < 5.0
        assert pub.n_sent + pub.n_dropped >= 10 and pub.thread.is_alive()
    finally:
        stalled.close()
        pub.close()


def test_device_duty_window():
    now = 10 * 10**9
    assert S.device_duty([], now, 0) == 0.0
    # 5 calls of 40 ms device time each in the last 5 s -> 0.04
    assert S.device_duty([(now - i * 10**9, 40.0) for i in range(5)], now, 0) == 0.04
    # since READY 1 s ago: one 40 ms call -> 0.04; a call that ended 10 ms into the window counts 10 ms
    assert S.device_duty([(now, 40.0)], now, now - 10**9) == 0.04
    assert S.device_duty([(now - 5 * 10**9 + 10_000_000, 40.0)], now, 0) == round(10 / 5000, 4)


def test_core_map_from_mk():
    """The status core_map numbers are MK's own: CoreMap built from the measured DRAM workers (CPU fixture)."""
    from models.experimental.gr00t.tt.megakernel.core_map import default_core_map

    cm = S.core_map_dict(default_core_map())
    assert cm["grid"] == [11, 10] and cm["megakernel"] == {"readers": 8, "hub": 1, "compute": 96, "spare": 5}
    assert len(cm["readers"]) == 8 and len(cm["spare"]) == 5 and cm["hub"] == [10, 9]
    assert sum(cm["megakernel"].values()) == 110


def test_static_status_of_reads_the_mk_fields():
    from types import SimpleNamespace as NS

    from models.experimental.gr00t.tt.megakernel.core_map import default_core_map

    backend = NS(num_command_queues=lambda device: 2)
    model = NS(captured=True, executor=NS(backend=backend, device=object()),
               head=NS(megakernel=NS(core_map=default_core_map())))
    st = S.static_status_of(model, fake=False)
    assert st["trace"] is True and st["cq"] == 2 and st["core_map"]["megakernel"]["compute"] == 96
    assert S.static_status_of(NS(), fake=False) == {"trace": None, "cq": None, "core_map": None}
    assert S.static_status_of(S.FakeG1Model(), fake=True) == {"trace": False, "cq": None, "core_map": None}


# --- CPU placement ------------------------------------------------------------------------------------------------


def test_parse_cpus_and_enforce_affinity():
    assert S.parse_cpus("0,1,8-11,16,17,24-27") == {0, 1, 8, 9, 10, 11, 16, 17, 24, 25, 26, 27}
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs two allowed CPUs")
    a = {allowed[0]}
    ready, done = threading.Event(), threading.Event()
    tid = {}

    def body():
        tid["t"] = threading.get_native_id()
        os.sched_setaffinity(0, a)
        ready.set()
        done.wait(10)

    t = threading.Thread(target=body)
    t.start()
    try:
        ready.wait(5)
        chk = S.enforce_affinity(set(allowed) - a)
        assert any(r["tid"] == tid["t"] and r["was"] == sorted(a) for r in chk["repinned"]), chk
        assert os.sched_getaffinity(tid["t"]) == set(allowed) - a
        assert S.enforce_affinity(set(allowed))["repinned"] == []
    finally:
        done.set()
        t.join()
        for t_id in os.listdir("/proc/self/task"):  # enforce_affinity narrowed every thread: give them back
            try:
                os.sched_setaffinity(int(t_id), set(allowed))
            except OSError:
                pass


def test_cli_status_pub_and_affinity(tmp_path):
    cpus = ",".join(map(str, sorted(os.sched_getaffinity(0))))
    env = dict(os.environ, TT_METAL_NUMA_BASED_AFFINITY="1", CHIP_SAFE_CPUS=cpus)
    p = subprocess.Popen([sys.executable, "-m", "rttt.server_g1", "--fake-model", "--port", "0", "--cpus", cpus,
                          "--log", "none", "--status-pub", "0", "--max-idle-s", "60"], cwd=str(RT), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, url, status = [], None, None
    try:
        t_end = time.monotonic() + 60
        while time.monotonic() < t_end and url is None:
            line = p.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip())
            if "] status PUB tcp://" in line:
                status = line.split("status PUB ")[1].split()[0]
            if line.startswith("server_g1 READY"):
                url = line.split()[2]
        assert url and status, "\n".join(lines)
        assert any("unset TT_METAL_NUMA_BASED_AFFINITY=1" in l for l in lines)
        sub = Sub(status)
        try:
            msgs = [m for _, m in sub.take(2)]
        finally:
            sub.close()
        assert len(msgs) == 2 and msgs[-1]["ready"] is True and msgs[-1]["backend"] == "mock:server_g1-fake"
        c = VlaClient(url, give_up_ms=10000)
        try:
            info = c.info()
        finally:
            c.close()
        assert info["status_pub"] == status and info["affinity_check"]["repinned"] == []
        assert info["affinity"] == sorted(S.parse_cpus(cpus))
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
    assert p.returncode == 0


def test_cli_refuses_cpus_outside_chip_safe_cpus():
    allowed = sorted(os.sched_getaffinity(0))
    env = dict(os.environ, CHIP_SAFE_CPUS=str(allowed[0]))
    outside = next((c for c in range(os.cpu_count() or 1) if c != allowed[0]), None)
    if outside is None:
        pytest.skip("single-CPU host")
    p = subprocess.run([sys.executable, "-m", "rttt.server_g1", "--fake-model", "--port", "0", "--cpus", str(outside),
                        "--log", "none"], cwd=str(RT), env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and "outside CHIP_SAFE_CPUS" in p.stdout and "bound" not in p.stdout


def test_no_ttnn():
    assert "ttnn" not in sys.modules
