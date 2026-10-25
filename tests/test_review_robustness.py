"""Adversarial review (robustness and measurement): regression tests for defects found in review.

Each test states the property the code should have; each one failed on the code as reviewed and
passes since its fix (analyze_run verdicts on missing data, eval_paired verdict and child-process
hygiene, Publisher drop counter, RequestLog growth outside ticks, ZmqWbc with fds >= 1024, the
analyze_run VLA/shadow/in-flight metrics). Nothing here binds 5555-5572: servers use random
127.0.0.1 ports, and every process a test starts is killed in a ``finally``.
"""

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import textwrap
import time
import uuid

import numpy as np
import pytest
import zmq

from rtloop import telemetry
from rtloop.schema import TICK_DTYPE, WBC_BALANCE, Outcome
from rtloop.telemetry import Publisher, RequestLog, RunDir, TickLog

from test_telemetry import ar, ev, synth, write_run  # noqa: E402 (shared synthetic-run helpers)

RT = Path(__file__).resolve().parents[1]
EVAL = RT / "scripts" / "eval_paired.py"


def _ticks_only_run(root: Path, ticks: np.ndarray) -> Path:
    """A run dir as a SIGKILLed loop leaves it before its first episode end: ticks.npy only."""
    run = RunDir(root=root, tag="ticks_only", config={"synthetic": True})
    tl = TickLog(run, capacity=256)
    for i in range(len(ticks)):
        r = tl.next()
        for name in TICK_DTYPE.names:
            r[name] = ticks[i][name]
        tl.commit()
    tl.close()
    return run.path


# --- analyze_run: verdicts must not pass on missing data ----------------------------------------


def test_smoke_gate_fails_without_episodes_or_requests(tmp_path):
    ticks, _, _ = synth()
    ticks["safety"][-1] = 1  # the last tick tripped the joint safety monitor, then the process died
    run = _ticks_only_run(tmp_path, ticks)
    v = ar.analyze(run, "m1_sim_smoke", plots=False)
    assert v["metrics"]["n_episodes"] == 0 and v["metrics"]["n_requests"] == 0
    assert not v["pass"], {c["name"]: (c["value"], c["pass"]) for c in v["checks"]}


def test_viewer_gates_need_a_viewer(tmp_path):
    ticks, reqs, eps = synth()
    ticks["viewer_us"] = 0.0  # what the loop logs without --viewer (no ViewerLink)
    run = write_run(tmp_path, ticks, reqs, eps, config={"synthetic": True, "args": {"viewer": False}})
    v = ar.analyze(run, "m1_rt", plots=False)
    viewer = {c["name"]: c for c in v["checks"] if c["name"].startswith("viewer_")}
    assert viewer and all(c["pass"] is False and c["value"] is None for c in viewer.values()), viewer
    assert all("viewer off" in c["note"] for c in viewer.values())
    assert not v["pass"]


# --- eval_paired: verdict and child-process hygiene ---------------------------------------------


def test_eval_report_only_without_pairs_is_not_a_pass(tmp_path):
    ticks, reqs, eps = synth()
    ref = write_run(tmp_path / "ref", ticks, reqs, eps)
    empty = tmp_path / "new_empty"
    empty.mkdir()
    rc = ev.main(["--report-only", f"C1={ref}", f"C4={empty}", "--out", str(tmp_path / "rep")])
    rep = json.loads((tmp_path / "rep" / "report.json").read_text())
    assert rep["comparisons"][0]["n"] == 0
    assert rc != 0 and not rep["verdict"].startswith("PASS"), (rc, rep["verdict"])


def _marked_pythons(mark: str) -> list[int]:
    """pids of `python -c ... <mark> ...` processes (the stand-in loops), not bash or eval_paired."""
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if len(argv) > 2 and argv[0].endswith(b"python") and argv[1] == b"-c" and mark.encode() in argv:
            out.append(int(p.name))
    return out


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_eval_paired_signal_stops_its_loop_processes(tmp_path, sig):
    mark = f"rtloop_review_leak_{uuid.uuid4().hex[:8]}"
    # DEFAULT_LOOP_CMD verbatim, except that the loop is a sleeping python carrying a marker.
    loop_cmd = list(ev.DEFAULT_LOOP_CMD)
    assert 'simpy -m rtloop.loop "$@"' in loop_cmd[2]
    loop_cmd[2] = loop_cmd[2].replace('simpy -m rtloop.loop "$@"',
                                      f'simpy -c "import time; time.sleep(120)" {mark} "$@"')
    matrix = tmp_path / "m.yaml"
    matrix.write_text("name: leak\nseeds: '0-1'\narms:\n  - {name: A, args: '--mode rt'}\n"
                      "  - {name: B, args: '--mode sync'}\ncompare: [[A, B]]\n")
    ep = subprocess.Popen([sys.executable, str(EVAL), "--matrix", str(matrix), "--out", str(tmp_path / "out"),
                           "--block", "0", "--loop-cmd", shlex.join(loop_cmd)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t_end = time.monotonic() + 20
        while not _marked_pythons(mark) and time.monotonic() < t_end:
            time.sleep(0.1)
        assert _marked_pythons(mark), "stand-in loop never started"
        ep.send_signal(sig)
        ep.wait(45)
        time.sleep(1.0)
        survivors = _marked_pythons(mark)
        assert not survivors, f"loop processes alive after eval_paired exited: {survivors}"
    finally:
        if ep.poll() is None:
            ep.kill()
            ep.wait(10)
        for pid in _marked_pythons(mark):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


# --- telemetry: tick-path costs and counters ----------------------------------------------------


def test_publisher_counts_drops_to_a_stalled_subscriber(zmq_ctx):
    pub = Publisher("tcp://127.0.0.1:*", ctx=zmq_ctx, hwm=100)
    sub = zmq_ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 100)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(pub.url)
    time.sleep(0.3)
    img = np.zeros((120, 160, 3), np.uint8)
    n = 5000
    for i in range(n):
        pub.ego(i, i, img)
    time.sleep(0.3)
    got = 0
    while sub.poll(200):
        sub.recv_multipart()
        got += 1
    sub.close(0)
    pub.close()
    assert got < n
    assert pub.dropped >= n - got - 400, (pub.sent, pub.dropped, got)  # 400: in-flight socket buffers


def test_requestlog_never_grows_inside_add_after_a_reset_window(tmp_path):
    rl = RequestLog(tmp_path, capacity=4096)
    s43 = np.zeros(43, np.float32)
    seq = 0
    for _ in range(4000):  # episode 0
        seq += 1
        rl.add({"seq": seq, "ep": 0, "k_obs": seq, "state43_sent": s43, "outcome": Outcome.PENDING})
    rl.end_episode()  # reset window: the place to make room for the next episode
    grew = []
    for _ in range(200):  # episode 1, every add is in a tick
        seq += 1
        before = rl._arr
        rl.add({"seq": seq, "ep": 1, "k_obs": seq, "state43_sent": s43, "outcome": Outcome.PENDING})
        if rl._arr is not before:
            grew.append(seq)
    rl.close()
    assert not grew, f"array re-allocated inside add() at seq {grew}"


# --- ZmqWbc: select() on the zmq FD -------------------------------------------------------------


def test_zmqwbc_works_when_its_fd_is_above_1023(zmq_ctx):
    from rtloop import wbc_backends as wb
    from test_wbc_backend import FakeWbcServer

    ort = wb.OrtCpu()
    srv = FakeWbcServer(zmq_ctx, ort)
    pad = []
    try:
        while not pad or pad[-1] < 1030:  # a process with many open files (RLIMIT_NOFILE here is 1048576)
            pad.append(os.open("/dev/null", os.O_RDONLY))
        z = wb.ZmqWbc(srv.url, fallback=ort, ctx=zmq_ctx)
        assert z._fd >= 1024
        obs = np.random.default_rng(0).standard_normal((1, 516)).astype(np.float32)
        metas = [z(WBC_BALANCE, obs, tick=k, seq=k)[1] for k in range(3)]
        z.close()
        assert not any(m["fallback"] for m in metas), [m["error"] for m in metas]
    finally:
        for fd in pad:
            os.close(fd)
        srv.stop()


# --- analyze_run: what the VLA, shadow and in-flight gates measure ------------------------------


def _quantised_vla(reqs, rtt_ms=80.0, srv_ms=60.6, queue_ms=0.7):
    """Requests as the loop logs them against a 60 ms server: the client sees the reply at the
    next tick poll (80 ms), the server stamps its own receive and reply (60.6 ms apart)."""
    for q in reqs:
        if q["k_obs"]:
            q["t_recv_ns"] = q["t_send_ns"] + int(rtt_ms * 1e6)
            q["srv_t_recv_ns"] = q["t_send_ns"] + int(queue_ms * 1e6)
            q["srv_t_reply_ns"] = q["srv_t_recv_ns"] + int(srv_ms * 1e6)
            q["srv_infer_ms"] = srv_ms - 0.6


def test_vla_gate_uses_server_time_not_tick_quantised_rtt(tmp_path):
    """Integration open issue (a): a 60 ms VLA reads ~80 ms at the client (tick polls). The m1_rt
    gate (71.5 ms) must judge the server's service time; the client RTT is reported, ungated."""
    ticks, reqs, eps = synth()
    _quantised_vla(reqs)
    v = ar.analyze(write_run(tmp_path / "ok", ticks, reqs, eps), "m1_rt", plots=False)
    m = v["metrics"]
    assert m["vla_p50_ms"] == pytest.approx(60.6, abs=1e-3)  # srv_t_reply_ns - srv_t_recv_ns
    assert m["vla_rtt_p50_ms"] == pytest.approx(80.0, abs=1e-3)
    assert m["vla_queue_p99_ms"] == pytest.approx(0.7, abs=1e-3)
    assert v["pass"], v["table"]
    assert "vla_rtt_p50_ms" not in {c["metric"] for c in v["checks"]}
    ticks, reqs, eps = synth()
    _quantised_vla(reqs, rtt_ms=100.0, srv_ms=90.0)  # a server that really is slow
    v = ar.analyze(write_run(tmp_path / "slow", ticks, reqs, eps), "m1_rt", plots=False)
    assert {c["name"] for c in v["checks"] if c["pass"] is False} == {"vla_p50_ms_max"}


def test_vla_gate_falls_back_to_srv_infer_ms_and_fails_without_server_timings(tmp_path):
    ticks, reqs, eps = synth()  # srv_infer_ms = 55, no server stamps
    assert ar.analyze(write_run(tmp_path / "a", ticks, reqs, eps), "m1_rt", plots=False)["metrics"][
        "vla_p50_ms"] == pytest.approx(55.0)
    for q in reqs:
        q.pop("srv_infer_ms", None)  # a stock server: INFO is {}
    v = ar.analyze(write_run(tmp_path / "b", ticks, reqs, eps), "m1_rt", plots=False)
    c = next(c for c in v["checks"] if c["name"] == "vla_p50_ms_max")
    assert c["pass"] is False and c["value"] is None and "no timings" in c["note"]


def test_shadow_gate_is_na_for_ort_wbc_and_fails_without_shadow_on_remote(tmp_path):
    """Integration open issue (c): with --wbc ort there is nothing to shadow, so the m1_rt shadow
    gate reports n/a (neither pass nor fail); with a remote WBC, a missing shadow fails."""
    ticks, reqs, eps = synth()
    ticks["wbc_be"] = 0  # ORT
    ticks["shadow_da"] = np.nan
    ort = write_run(tmp_path / "ort", ticks, reqs, eps,
                    config={"synthetic": True, "args": {"viewer": True, "wbc": "ort"}})
    v = ar.analyze(ort, "m1_rt", plots=False)
    c = next(c for c in v["checks"] if c["name"] == "shadow_da_p99_max")
    assert c["pass"] is None and c["status"] == "n/a" and c["note"] == "no shadow (--wbc ort)"
    assert v["pass"], v["table"]  # n/a does not fail the verdict
    assert "shadow_da_p99_max" in v["table"] and "n/a" in v["table"]

    ticks, reqs, eps = synth()  # a remote WBC on every tick, but no shadow value logged (--no-shadow)
    ticks["shadow_da"] = np.nan
    remote = write_run(tmp_path / "remote", ticks, reqs, eps,
                       config={"synthetic": True, "args": {"viewer": True, "wbc": "tcp://127.0.0.1:1"}})
    v = ar.analyze(remote, "m1_rt", plots=False)
    c = next(c for c in v["checks"] if c["name"] == "shadow_da_p99_max")
    assert c["pass"] is False and "no shadow_da" in c["note"] and not v["pass"]


def test_inflight_counts_an_abandoned_request_still_on_the_server(tmp_path):
    """Request 4 times out at 500 ms while the server is still computing it; request 5 (sent right
    after) is taken only 300 ms later: two requests were in flight at the server, so the gate fails.
    The same run with request 5 taken after the usual 0.7 ms transfer counts 1."""
    def run(wait5_ms):
        t0, ms = 10**12, 1_000_000
        rows = []
        for i in range(3):  # normal requests: taken 0.7 ms after send, 60 ms service, seen at 80 ms
            ts = t0 + i * 320 * ms
            rows.append(dict(seq=i + 1, ep=0, k_obs=16 * (i + 1), t_send_ns=ts, t_recv_ns=ts + 80 * ms, d=4,
                             k_arr=16 * (i + 1) + 4, srv_infer_ms=60.0, srv_t_recv_ns=ts + int(0.7 * ms),
                             srv_t_reply_ns=ts + int(60.7 * ms), outcome=Outcome.APPLIED))
        ts = t0 + 1000 * ms
        rows.append(dict(seq=4, ep=0, k_obs=64, t_send_ns=ts, t_recv_ns=ts + 500 * ms, outcome=Outcome.TIMEOUT))
        ts5 = ts + 520 * ms
        rows.append(dict(seq=5, ep=0, k_obs=90, t_send_ns=ts5, t_recv_ns=ts5 + int((wait5_ms + 80) * ms), d=5,
                         k_arr=95, srv_infer_ms=60.0, srv_t_recv_ns=ts5 + int(wait5_ms * ms),
                         srv_t_reply_ns=ts5 + int((wait5_ms + 60) * ms), outcome=Outcome.APPLIED))
        d = tmp_path / f"w{wait5_ms}"
        d.mkdir()
        rl = RequestLog(d)
        for q in rows:
            q = dict(q)
            outcome = q.pop("outcome")
            rl.add(q)
            rl.update(q["seq"], outcome=outcome)
        rl.close()
        return telemetry.load_requests(d), telemetry.load_request_extras(d)

    reqs, extras = run(300.0)
    assert ar.inflight_max(reqs, extras) == 2
    assert ar.compute_metrics(np.zeros(0, TICK_DTYPE), reqs, [], extras)["vla_queue_max_ms"] == pytest.approx(300.0)
    reqs, extras = run(0.7)
    assert ar.inflight_max(reqs, extras) == 1
    assert ar.inflight_max(reqs, {}) == 1  # without the server stamps only the client view is left
    assert ar.inflight_max(reqs[:0], {}) is None


def test_requestlog_writes_no_file_inside_a_tick(tmp_path):
    """A terminal outcome (timeout, late reject, ...) is set inside a tick; its jsonl line waits for
    the reset window (end_episode), so a tick makes no write/flush syscall."""
    rl = RequestLog(tmp_path)
    s43 = np.zeros(43, np.float32)
    rl.add({"seq": 1, "ep": 0, "k_obs": 0, "state43_sent": s43, "outcome": Outcome.PENDING})
    size0 = rl.jsonl_path.stat().st_size
    for outcome in (Outcome.TIMEOUT, Outcome.LATE_REJECT, Outcome.ERROR, Outcome.GEN_DROP):
        rl.update(1, outcome=outcome)
    assert rl.jsonl_path.stat().st_size == size0 == 0
    rl.end_episode()
    assert rl.jsonl_path.stat().st_size > 0
    rl.close()


def test_dump_vla_obs_is_written_in_the_reset_window_not_in_the_tick(tmp_path):
    """--dump-vla-obs N: the ~0.9 MB np.savez per request used to run inside the tick (_audit);
    the observation is now queued and written by _flush_vla_dumps after the tick loop."""
    from review_harness import make_loop, run_episode

    lp = make_loop(tmp_path, "rt", max_ticks=60, d_of=lambda i: 4)
    lp.args.dump_vla_obs = 3
    lp.vla_dump_pending = []
    r = run_episode(lp)
    vdir = lp.run.path / "vla_obs"
    assert len(r.sent) > 3 and not vdir.exists()  # nothing written during the ticks
    assert [s for s, _ in lp.vla_dump_pending] == [x[2] for x in r.sent[:3]]
    lp._flush_vla_dumps()
    assert sorted(p.name for p in vdir.iterdir()) == [f"{x[2]:06d}.npz" for x in r.sent[:3]]
    assert lp.vla_dump_pending == []


def test_malloc_trim_reports_what_it_released():
    from rtloop import loop

    r = loop.malloc_trim()
    assert r is None or (set(r) == {"released_mb", "rss_mb", "ms"} and r["ms"] >= 0 and r["rss_mb"] > 0)


def test_fixed_mmap_threshold_returns_freed_big_blocks(tmp_path):
    """Integration open issue (b): with glibc's dynamic threshold a freed 32 MB block raises the
    threshold, so the next big blocks come from the heap and stay resident after free; with the
    loop's fixed 1 MiB threshold they are unmapped on free. Run in a child: mallopt is process-wide."""
    code = textwrap.dedent("""
        import ctypes, sys
        import numpy as np
        from rtloop import loop
        if sys.argv[1] == "fixed":
            assert loop.set_mmap_threshold(1 << 20)
        assert not loop.set_mmap_threshold(0)
        big = np.ones(32 << 20, np.uint8); del big   # the first big free: dynamic threshold -> 32 MB
        keep = []
        for i in range(8):                            # big transient blocks among long-lived small ones
            a = np.ones(8 << 20, np.uint8)
            keep.append(np.ones(4096, np.uint8))
            del a
        print(loop.rss_anon_mb())
    """)
    out = {}
    for mode in ("dynamic", "fixed"):
        p = subprocess.run([sys.executable, "-c", code, mode], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "PYTHONPATH": str(RT)})
        assert p.returncode == 0, p.stderr
        out[mode] = float(p.stdout.split()[-1])
    assert out["dynamic"] - out["fixed"] >= 6, out  # the freed 8 MB block stays resident only in "dynamic"
