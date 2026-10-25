"""The loop-side control integration (rtloop/loop.py + rtloop/control.py, WP-1.7, PLAN.md §1.4.5), tested
without the sim: the tick-boundary methods are driven on a Loop built with __new__ and fake backends.

tests/hud/loop_control.patch is the HUD lane's original change, applied to loop.py on 2026-10-25 (with the
--shadow additions on top); it is kept for the record. If loop.py ever loses ``_control_boundary`` the fixture
falls back to applying the patch to a temporary copy, so the tests still say whether it would apply."""

import argparse
import importlib.util
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
RT = HERE.parents[1]
sys.path.insert(0, str(RT / "scripts"))
import control_proto  # noqa: E402  (the shim: the same objects as rtloop.control)

from rtloop import clock  # noqa: E402
from rtloop import splicer as sp  # noqa: E402
from rtloop.schema import TICK_NS  # noqa: E402


@pytest.fixture(scope="module")
def loopmod(tmp_path_factory):
    import rtloop

    if importlib.util.find_spec("rtloop.control") is None:
        sys.modules["rtloop.control"] = control_proto
        rtloop.control = control_proto
    src = RT / "rtloop" / "loop.py"
    if "_control_boundary" in src.read_text():
        from rtloop import loop

        return loop
    d = tmp_path_factory.mktemp("patched")
    shutil.copy(src, d / "loop.py")
    p = subprocess.run(["patch", "-s", str(d / "loop.py"), str(HERE / "loop_control.patch")],
                       capture_output=True, text=True)
    if p.returncode != 0:
        # A skip, not a failure: M1 may be editing loop.py concurrently. Regenerate the patch if this shows.
        pytest.skip(f"loop_control.patch no longer applies to the current rtloop/loop.py: {p.stdout}{p.stderr}")
    spec = importlib.util.spec_from_file_location("rtloop_loop_patched", d / "loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeZmqWbc:
    def __init__(self, url):
        self.url, self.latched, self.consecutive = url, True, 3


class FakeVla:
    def __init__(self, url):
        self.url, self.in_flight = url, False


@pytest.fixture
def lp(loopmod, zmq_ctx):
    L = loopmod.Loop.__new__(loopmod.Loop)
    ort = SimpleNamespace(name="ort")
    gpu = FakeZmqWbc("tcp://127.0.0.1:1")
    L.args = argparse.Namespace(swap_wbc_every=None, no_shadow=False)
    L.wb = SimpleNamespace(ZmqWbc=FakeZmqWbc)
    L.ort = ort
    L.remote_shadow = None
    L.swap_every_s = None
    L.pauses = []
    L.runtime = SimpleNamespace(primary=gpu, shadow=ort)
    L.wbc_targets, L.wbc_name = {"gpu": gpu, "ort": ort}, "gpu"
    L.vla = FakeVla("tcp://127.0.0.1:2")
    L.vla_info = {"backend": "cuda:rtx5090:x@1"}
    L.vla_clients = {"gpu": (L.vla, L.vla_info), "tt": (FakeVla("tcp://127.0.0.1:3"), {"backend": "tt:p150a:x"})}
    L.vla_name = "gpu"
    L.control = pull = control_proto.ControlPull("tcp://127.0.0.1:*", ctx=zmq_ctx)
    L.cstate = control_proto.ControlState()
    L.control_log = []
    L.reset_requested = L._resumed = L.stop_requested = False
    L.n_ticks_total = L.n_pauses = L.n_operator_resets = L.n_vla_swaps = L.n_wbc_swaps = 0
    L.paused_ms = 0.0
    L.ep_wbc_swaps, L.ep_paused_ms = 0, 0.0
    L.pub_on, L.rt, L.ep = False, True, 0
    L.st = sp.new_state()
    L.pacer = clock.Pacer()
    L.pacer.anchor()
    push = control_proto.ControlPush(L.control.url, ctx=zmq_ctx)
    t_end = time.monotonic() + 2
    while not push.send("pause", "off") and time.monotonic() < t_end:  # wait for the connection
        time.sleep(0.01)
    yield L, push
    push.close()
    pull.close()


def boundary(L, k, expect=1, timeout=2.0):
    """Run _control_boundary until `expect` more commands have been applied or rejected."""
    want = L.cstate.n_applied + L.cstate.n_rejected + expect
    t_end = time.monotonic() + timeout
    end = False
    while L.cstate.n_applied + L.cstate.n_rejected < want and time.monotonic() < t_end:
        end |= L._control_boundary(k)
        time.sleep(0.002)
    return end


def test_wbc_swap_applies_from_the_next_tick(lp):
    L, push = lp
    boundary(L, 0)  # the fixture's connection probe ("pause off": a no-op)
    push.send("swap_wbc", "next")
    assert boundary(L, 10) is False
    assert L.runtime.primary is L.ort and L.runtime.shadow is None and L.wbc_name == "ort"
    gpu = L.wbc_targets["gpu"]
    push.send("swap_wbc", "gpu")
    boundary(L, 11)
    assert L.runtime.primary is gpu and L.runtime.shadow is L.ort  # ORT shadows a remote backend (D16)
    assert gpu.latched is False and gpu.consecutive == 0  # a latched episode never sticks to a swap
    assert L.n_wbc_swaps == 2 and L.ep_wbc_swaps == 2
    assert [e["action"] for e in L.control_log[-2:]] == ["swap_wbc", "swap_wbc"]


def test_vla_swap_waits_for_the_reset_window(lp):
    L, push = lp
    boundary(L, 0)
    L.st.d_hist[:] = [4, 4, 5]
    push.send("swap_vla", "next")
    boundary(L, 5)
    assert L.cstate.pending_vla == "tt" and L.vla_name == "gpu"  # nothing changes mid-episode
    L._swap_vla(L.cstate.pending_vla)  # what episode() does before the gen bump
    assert L.vla_name == "tt" and L.vla.url == "tcp://127.0.0.1:3" and L.vla_info["backend"].startswith("tt")
    assert L.cstate.pending_vla is None and L.st.d_hist == [] and L.n_vla_swaps == 1
    L.vla.in_flight = True
    with pytest.raises(AssertionError):
        L._swap_vla("gpu")


def test_reset_ends_the_episode(lp):
    L, push = lp
    boundary(L, 0)
    push.send("reset")
    assert boundary(L, 42) is True and L.reset_requested


def test_pause_freezes_between_ticks_and_reanchors(lp):
    L, push = lp
    boundary(L, 0)
    k = 300

    def resume():
        time.sleep(0.25)
        push.send("pause")

    th = threading.Thread(target=resume)
    push.send("pause")
    th.start()
    t0 = time.monotonic()
    end = boundary(L, k, expect=2, timeout=5)
    th.join()
    dt = time.monotonic() - t0
    assert end is False and not L.cstate.paused and L._resumed and L.n_pauses == 1
    assert 0.2 <= dt < 2.0 and L.paused_ms >= 200
    slack_ms = (L.pacer.deadline(k + 1) - clock.now_ns()) / 1e6
    assert 0 < slack_ms <= TICK_NS / 1e6  # T_{k+1} = resume + 20 ms: no slip, no catch-up burst


def test_reset_during_pause(lp):
    L, push = lp
    boundary(L, 0)
    threading.Timer(0.1, lambda: push.send("reset")).start()
    push.send("pause")
    assert boundary(L, 7, expect=2, timeout=5) is True and L.reset_requested and not L.cstate.paused


def test_rejected_and_auto_swap(lp):
    L, push = lp
    boundary(L, 0)
    push.send("swap_wbc", "x280")  # not registered
    boundary(L, 1)
    assert L.cstate.n_rejected == 1 and L.wbc_name == "gpu"
    L.swap_every_s = 0.1  # every 5 ticks (--swap-wbc-every 0.1)
    L.n_ticks_total = 0
    names = []
    for k in range(20):
        L._control_boundary(k)
        names.append(L.wbc_name)
    assert names[4] == "ort" and names[9] == "gpu" and names[14] == "ort" and names.count("ort") == 10


def test_boundary_is_cheap_when_idle(lp):
    L, push = lp
    boundary(L, 0)
    t0 = time.perf_counter_ns()
    for k in range(2000):
        L._control_boundary(k)
    per_us = (time.perf_counter_ns() - t0) / 2000 / 1e3
    assert per_us < 30, per_us  # it runs inside the 20 ms tick's slack; measured ~2-4 us
    L.control, L.swap_every_s = None, None
    t0 = time.perf_counter_ns()
    for k in range(2000):
        L._control_boundary(k)
    assert (time.perf_counter_ns() - t0) / 2000 / 1e3 < 5


def test_ort_primary_gets_the_remote_shadow_back(lp):
    """T-3.6 with swaps: ORT as primary is shadowed by the --shadow W-WBC; a remote primary by ORT (D16)."""
    L, push = lp
    boundary(L, 0)
    L.remote_shadow = FakeZmqWbc("tcp://127.0.0.1:9")
    push.send("swap_wbc", "ort")
    boundary(L, 3)
    assert L.runtime.primary is L.ort and L.runtime.shadow is L.remote_shadow
    push.send("swap_wbc", "gpu")
    boundary(L, 4)
    assert L.runtime.primary is L.wbc_targets["gpu"] and L.runtime.shadow is L.ort


def test_pauses_are_recorded_in_full(lp):
    L, push = lp
    boundary(L, 0)
    threading.Timer(0.1, lambda: push.send("pause", "off")).start()
    push.send("pause", "on")
    boundary(L, 12, expect=2, timeout=5)
    assert len(L.pauses) == 1 and L.pauses[0][:2] == [0, 12] and L.pauses[0][2] >= 50


def test_cli_flags(loopmod):
    a = loopmod.build_parser().parse_args(["--control", "--vla-alt", "gpu=tcp://127.0.0.1:5565",
                                           "--wbc-alt", "gpu=tcp://127.0.0.1:5556", "--swap-wbc-every", "10"])
    assert a.control == "tcp://127.0.0.1:5571" and a.swap_wbc_every == 10.0
    assert control_proto.parse_targets(a.vla_alt) == {"gpu": "tcp://127.0.0.1:5565"}
    b = loopmod.build_parser().parse_args([])
    assert b.control is None and b.vla_alt == "" and b.wbc_alt == "" and b.swap_wbc_every is None
    assert b.shadow is None and b.shadow_timeout_ms is None
    c = loopmod.build_parser().parse_args(["--shadow", "tcp://127.0.0.1:5558", "--shadow-timeout-ms", "3"])
    assert c.shadow == "tcp://127.0.0.1:5558" and c.shadow_timeout_ms == 3


def test_shadow_needs_the_ort_primary(loopmod):
    for argv in (["--shadow", "tcp://127.0.0.1:1", "--wbc", "tcp://127.0.0.1:2"],
                 ["--shadow", "tcp://127.0.0.1:1", "--no-shadow"]):
        with pytest.raises(SystemExit) as e:
            loopmod.main(["rtloop.loop", *argv])
        assert e.value.code == 2


def test_real_module_is_tested(loopmod):
    """loop.py carries the integration itself now (no patched temporary copy)."""
    assert loopmod.__name__ == "rtloop.loop" and hasattr(loopmod.Loop, "_control_boundary")
    assert control_proto.ControlPull is __import__("rtloop.control", fromlist=["x"]).ControlPull


# --- the real loop (VL mode: no pacing, no CPU pinning), driven through its PUB and control PULL ------------


class OrtWbcServer:
    """A W-WBC REP server in a thread that answers with in-process ORT (a stand-in for :5556/:5558)."""

    def __init__(self, offset: float = 0.0):
        import zmq

        from rtloop import wbc_backends as wb
        from rtloop import wire

        self.zmq, self.wire, self.ort, self.offset = zmq, wire, wb.OrtCpu(), np_f32(offset)
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1", 20000, 60000)
        assert not 5555 <= self.port <= 5574
        self.url = f"tcp://127.0.0.1:{self.port}"
        self.n = 0
        self._halt = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._halt.is_set():
            if not self.sock.poll(20):
                continue
            req = self.wire.loads(self.sock.recv())
            ep, data = req.get("endpoint"), req.get("data") or {}
            if ep == "info":
                out = {"backend": "cuda-graph:fake", "proto": 2, "policies": list(self.ort.policies),
                       "parity_tol": 1e-2}
            elif ep == "ping":
                out = {"status": "ok"}
            else:
                self.n += 1
                out = {"action": self.ort(data["policy"], data["obs"]) + self.offset, "tick": data.get("tick"),
                       "seq": data.get("seq"), "compute_us": 7.0}
            self.sock.send(self.wire.dumps(out))

    def stop(self):
        self._halt.set()
        self.thread.join(2)
        self.sock.close(linger=0)
        self.ctx.term()


def np_f32(x):
    import numpy as np

    return np.float32(x)


def _free_port() -> int:
    import socket

    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if not 5555 <= port <= 5574:
            return port


@pytest.mark.sim
def test_real_loop_applies_operator_control(tmp_path, zmq_ctx):
    """WBC swap from the next tick, a pause that is not a period, an operator reset that moves on to the next
    seed without an EpisodeLog record, all logged; analyze_run accepts the run (VL mode, so no RT loop)."""
    import json

    import numpy as np
    import zmq

    sys.path.insert(0, str(RT / "tests"))
    import loop_utils as lu
    from rtloop import wire
    from rtloop.schema import WbcBackend

    mock, url = lu.start_mock("detfn")
    gpu = OrtWbcServer()
    pub, ctl = f"tcp://127.0.0.1:{_free_port()}", f"tcp://127.0.0.1:{_free_port()}"
    run = lu.LoopRun(tmp_path, "ctl", "--mode", "vl", "--d-inj", "4", "--vla", url, "--wbc", "ort",
                     "--wbc-alt", f"gpu={gpu.url}", "--control", ctl, "--pub", pub, "--seeds", "0-1",
                     "--max-ticks", "300")
    sub = zmq_ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"tick")
    sub.setsockopt(zmq.RCVTIMEO, 120_000)
    sub.connect(pub)
    push = control_proto.ControlPush(ctl, ctx=zmq_ctx)

    def until(pred):
        while True:
            m = wire.loads(sub.recv_multipart()[1])
            if pred(m):
                return m

    try:
        until(lambda m: m["ep"] == 0 and m["k"] >= 10)
        assert push.send("swap_wbc", "gpu")
        until(lambda m: m["wbc_be"] == WbcBackend.GPU)
        assert push.send("pause", "on")
        time.sleep(0.4)
        assert push.send("pause", "off")
        k = until(lambda m: m["period_us"] != m["period_us"])["k"]  # the first tick after the pause (NaN period)
        assert push.send("reset")
        until(lambda m: m["ep"] == 1)
        d = run.finish(180)
    finally:
        run.kill()
        push.close()
        sub.close(linger=0)
        gpu.stop()
        lu.stop(mock)
    summary = json.loads((d / "summary.json").read_text())
    c = summary["control"]
    assert c["wbc_swaps"] == 1 and c["pauses"] == 1 and c["operator_resets"] == 1 and c["rejected"] == 0
    assert len(c["pauses_at"]) == 1 and c["pauses_at"][0][0] == 0 and c["pauses_at"][0][2] >= 300
    eps = [json.loads(ln) for ln in (d / "episodes.jsonl").read_text().splitlines()]
    assert [(e["ep"], e["seed"], e["end"]) for e in eps] == [(1, 1, "TRUNC")]  # ep 0 was the operator reset
    assert eps[0]["wbc_target"] == "gpu" and eps[0]["wbc"] == gpu.url and gpu.n > 0
    ticks = np.load(d / "ticks.npy")
    t0 = ticks[ticks["ep"] == 0]
    assert t0["k"].max() < 299 and t0["k"].max() >= k  # ended early, at the reset
    nan = np.isnan(t0["period_us"]) & (t0["k"] > 0)
    assert nan.sum() == 1 and int(t0["k"][nan][0]) == k  # the first tick after the pause carries no period
    assert (t0["wbc_be"][t0["k"] > int(t0["k"][nan][0])] == WbcBackend.GPU).all()
    ar = subprocess.run([sys.executable, str(RT / "scripts" / "analyze_run.py"), str(d), "--gates", "m1_sim_smoke",
                         "--no-plots"], capture_output=True, text=True)
    assert ar.returncode == 0, ar.stdout + ar.stderr
    v = json.loads((d / "verdict.json").read_text())
    assert v["metrics"]["operator_pauses"] == 1 and v["run_info"]["operator_resets"] == 1
