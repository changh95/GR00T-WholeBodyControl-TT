"""WP-1.6 / WP-1.9: telemetry logs, the run stamp, analyze_run gates and eval_paired statistics, on synthetic data."""

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from rtloop.schema import REQUEST_DTYPE, TICK_DTYPE, TICK_NS, EpisodeEnd, Outcome, WbcBackend
from rtloop import telemetry, wire
from rtloop.telemetry import (EpisodeLog, Publisher, RequestLog, RunDir, TickLog, collect_config, load_episodes,
                              load_request_extras, load_requests, load_ticks)

RT = Path(__file__).resolve().parents[1]
ANALYZE = RT / "scripts" / "analyze_run.py"
EVAL = RT / "scripts" / "eval_paired.py"


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ar = _load_script(ANALYZE)
ev = _load_script(EVAL)


def _fill(tl: TickLog, k: int, ep: int = 0) -> None:
    r = tl.next()
    r["ep"] = ep
    r["k"] = k
    r["t_start_ns"] = 1_000 + k * TICK_NS
    r["cmd35"] = np.full(35, k, np.float32)
    tl.commit()


# --- RunDir -------------------------------------------------------------------------------------


def test_rundir_layout_latest_and_env_root(tmp_path, monkeypatch):
    a = RunDir(root=tmp_path, tag="x", config={"v": 1})
    b = RunDir(root=tmp_path, tag="x")
    assert a.path != b.path and a.path.name.endswith("_x") and b.path.name.endswith("_x")
    assert os.readlink(tmp_path / "latest") == b.path.name
    assert json.loads((a.path / "config.json").read_text()) == {"v": 1}
    a.update_config(info={"backend": "mock"})
    assert json.loads((a.path / "config.json").read_text())["info"] == {"backend": "mock"}
    monkeypatch.setenv(telemetry.RUNS_ENV, str(tmp_path / "envroot"))
    c = RunDir(tag="y")
    assert c.path.parent == tmp_path / "envroot"


# --- TickLog ------------------------------------------------------------------------------------


def test_ticklog_survives_missing_close(tmp_path):
    """130 rows, no close (crash): np.load sees the 100 rows published at the 50-row flushes."""
    run = RunDir(root=tmp_path, tag="crash")
    tl = TickLog(run)
    for k in range(130):
        _fill(tl, k)
    a = np.load(run.path / "ticks.npy")
    assert len(a) >= 100 and len(a) == 100
    assert np.array_equal(a["k"], np.arange(100)) and a["cmd35"][99, 0] == 99
    assert int((run.path / "ticks.len").read_text()) == 100
    assert len(load_ticks(run.path)) == 100
    tl.close()


def test_ticklog_survives_process_crash(tmp_path):
    code = textwrap.dedent(f"""
        import os, numpy as np
        from rtloop.telemetry import TickLog
        from pathlib import Path
        tl = TickLog(Path({str(tmp_path)!r}), capacity=64)
        for k in range(130):
            r = tl.next(); r["k"] = k; r["q43_meas"] = np.full(43, k, np.float32); tl.commit()
        os._exit(1)
    """)
    p = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=f"{RT}:{os.environ.get('PYTHONPATH', '')}"))
    assert p.returncode == 1
    a = np.load(tmp_path / "ticks.npy")
    assert len(a) == 100 and np.array_equal(a["k"], np.arange(100)) and a["q43_meas"][42, 7] == 42
    assert len(np.load(tmp_path / "ticks.npy", mmap_mode="r")) == 100


def test_ticklog_close_truncates(tmp_path):
    tl = TickLog(tmp_path)
    for k in range(130):
        _fill(tl, k)
    offset = tl._offset
    tl.close()
    tl.close()  # idempotent
    a = np.load(tmp_path / "ticks.npy")
    assert len(a) == 130 and np.array_equal(a["k"], np.arange(130))
    assert (tmp_path / "ticks.npy").stat().st_size == offset + 130 * TICK_DTYPE.itemsize
    assert int((tmp_path / "ticks.len").read_text()) == 130
    with pytest.raises(RuntimeError):
        tl.next()


def test_ticklog_grows_in_blocks(tmp_path):
    tl = TickLog(tmp_path, capacity=64)
    for k in range(230):
        _fill(tl, k)
    assert tl._cap == 256
    assert len(np.load(tmp_path / "ticks.npy")) == 200  # published so far
    tl.prefault(100)
    assert len(tl) == 230
    tl.close()
    a = np.load(tmp_path / "ticks.npy")
    assert np.array_equal(a["k"], np.arange(230)) and np.array_equal(a["cmd35"][:, 3], np.arange(230))


def test_ticklog_next_is_zeroed_view(tmp_path):
    tl = TickLog(tmp_path)
    r = tl.next()
    r["k"] = 5
    r["cmd35"] = 1.0
    r = tl.next()  # handed out again without a commit: cleared
    assert r["k"] == 0 and not r["cmd35"].any()
    r["k"] = 7
    tl.commit()
    assert tl.rows()[0]["k"] == 7  # filled in place
    with pytest.raises(RuntimeError):
        tl.commit()  # no next()
    tl.close()


def test_ticklog_row_cost(tmp_path):
    tl = TickLog(tmp_path)
    tl.prefault(2000)
    cmd = np.zeros(35, np.float32)
    t0 = time.perf_counter()
    for k in range(2000):
        r = tl.next()
        r["k"] = k
        r["compute_us"] = 1.0
        r["cmd35"] = cmd
        tl.commit()
    per_row_us = (time.perf_counter() - t0) / 2000 * 1e6
    tl.close()
    assert per_row_us < 50, per_row_us


def _mapped_rss_kb(path: Path) -> int:
    """Resident kB of this process's mappings of `path` (/proc/self/smaps)."""
    total, mine = 0, False
    with open("/proc/self/smaps") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and "-" in parts[0] and ":" in parts[3]:
                mine = len(parts) > 5 and parts[5] == str(path)
            elif mine and parts[0] == "Rss:":
                total += int(parts[1])
    return total


def test_ticklog_release_drops_resident_pages_and_keeps_data(tmp_path):
    """T-1.10: committed rows leave the process RSS in the reset window (0.9 MB per 1440 ticks would
    otherwise read as a leak); the data stays in the file and in the still-open log."""
    run = RunDir(root=tmp_path, tag="rel")
    tl = TickLog(run, capacity=8192)
    for k in range(5000):
        _fill(tl, k)
    tl.flush(sync=True)
    before = _mapped_rss_kb(tl.path)
    released = tl.release()
    after = _mapped_rss_kb(tl.path)
    assert released >= 5000 * TICK_DTYPE.itemsize - 2 * 4096 and before - after >= released // 1024 - 8
    assert np.array_equal(tl.rows()["k"], np.arange(5000))  # faults back in from the page cache
    assert tl.release() == 0  # nothing new committed since
    for k in range(5000, 6000):
        _fill(tl, k)
    assert 0 < tl.release() <= 1001 * TICK_DTYPE.itemsize + 4096  # only the new rows
    _fill(tl, 6000)
    tl.close()
    a = load_ticks(run.path)
    assert len(a) == 6001 and a["cmd35"][4999, 0] == 4999 and np.array_equal(a["k"], np.arange(6001))


# --- RequestLog ---------------------------------------------------------------------------------


def _req(seq, ep, k_obs, **kw):
    row = {"seq": seq, "gen": ep, "ep": ep, "k_obs": k_obs, "t_send_ns": 10 + seq, "state43_sent":
           np.arange(43, dtype=np.float32) * np.float32(0.1) + seq, "obs_sha1": hashlib.sha1(bytes([seq])).digest()}
    row.update(kw)
    return row


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_requestlog_update_jsonl_and_npz(tmp_path):
    rl = RequestLog(tmp_path, capacity=2)
    for seq in range(3):
        assert rl.add(_req(seq, 0, 16 * seq, encode_ms=1.5)) == seq
    rl.update(0, outcome=Outcome.APPLIED, k_arr=4, d=4, first_row=4, last_row=19, n_rows_used=16)
    rl.update(2, outcome=Outcome.APPLIED, k_arr=36, d=4)
    assert _jsonl(rl.jsonl_path) == []  # applied rows wait for the episode end
    rl.update(1, outcome=Outcome.LATE_REJECT, k_arr=46, d=30)
    assert _jsonl(rl.jsonl_path) == []  # terminal rows too: no file I/O inside a tick

    rl.add(_req(3, 1, 0))  # new episode: ep 0 is finalised
    lines = _jsonl(rl.jsonl_path)
    assert sorted(r["seq"] for r in lines) == [0, 1, 2]
    late = next(r for r in lines if r["seq"] == 1)
    assert late["outcome_name"] == "LATE_REJECT"
    assert late["encode_ms"] == 1.5 and late["obs_sha1"] == hashlib.sha1(bytes([1])).hexdigest()
    with np.load(rl.npz_path) as z:
        assert len(z["requests"]) == 3 and z["requests"].dtype == REQUEST_DTYPE
        assert list(z["k_obs"]) == [0, 16, 32]

    rl.update(0, n_rows_used=17)  # after finalisation: re-appended at the next episode end
    rl.update(3, outcome=Outcome.TIMEOUT)
    rl.close()
    rows = load_requests(tmp_path)
    assert len(rows) == 4
    assert list(rows["outcome"]) == [Outcome.APPLIED, Outcome.LATE_REJECT, Outcome.APPLIED, Outcome.TIMEOUT]
    assert rows["n_rows_used"][0] == 17 and rows["d"][1] == 30
    for seq in range(4):
        assert np.array_equal(rows["state43_sent"][seq], _req(seq, 0, 0)["state43_sent"])
        assert bytes(rows["obs_sha1"][seq]) == hashlib.sha1(bytes([seq])).digest()
    seq0 = [r for r in _jsonl(rl.jsonl_path) if r["seq"] == 0]
    assert len(seq0) == 2 and seq0[-1]["n_rows_used"] == 17
    os.remove(rl.npz_path)  # jsonl fallback keeps the last line per seq
    rows_j = load_requests(tmp_path)
    assert rows_j["n_rows_used"][0] == 17
    assert np.array_equal(rows_j["state43_sent"], rows["state43_sent"])
    assert np.array_equal(rows_j["obs_sha1"], rows["obs_sha1"])


def test_requestlog_drops_extras_two_episode_ends_old(tmp_path):
    """T-1.10: the per-row extras dict was kept for the whole run (~0.9 KB per request). It now lives until
    its row is two episode ends old; the jsonl keeps every extra, and a late update still lands."""
    rl = RequestLog(tmp_path)
    for ep in range(3):
        for j in range(4):
            seq = 4 * ep + j
            rl.add(_req(seq, ep, 16 * j, srv_t_recv_ns=1000 + seq, bytes=7))
            rl.update(seq, outcome=Outcome.APPLIED, srv_t_reply_ns=2000 + seq)
        rl.end_episode()
        kept = [i for i, x in enumerate(rl._extra) if x is not None]
        assert kept == list(range(4 * ep, 4 * ep + 4))  # the episode that just ended (the next one may update it)
    rl.update(1, n_rows_used=9, drain="late")  # a row whose extras are gone
    rl.close()
    extras = load_request_extras(tmp_path)
    assert sorted(extras) == list(range(12))
    assert (extras[1]["srv_t_recv_ns"], extras[1]["srv_t_reply_ns"], extras[1]["bytes"]) == (1001, 2001, 7)
    assert extras[1]["drain"] == "late" and extras[5]["srv_t_reply_ns"] == 2005
    rows = load_requests(tmp_path)
    assert rows["n_rows_used"][1] == 9 and rows["outcome"][1] == Outcome.APPLIED
    os.remove(rl.npz_path)
    assert load_requests(tmp_path)["n_rows_used"][1] == 9  # the re-appended line carries every array field


def test_requestlog_end_episode_explicit(tmp_path):
    rl = RequestLog(tmp_path)
    rl.add(_req(0, 0, 0, outcome=Outcome.APPLIED))
    rl.end_episode()
    assert len(_jsonl(rl.jsonl_path)) == 1 and len(load_requests(tmp_path)) == 1
    rl.close()
    assert len(_jsonl(rl.jsonl_path)) == 1


# --- EpisodeLog ---------------------------------------------------------------------------------


def test_episodelog_jsonable(tmp_path):
    el = EpisodeLog(tmp_path)
    el.add({"ep": np.uint32(0), "seed": np.int64(2000), "end": EpisodeEnd.SUCCESS, "success": np.bool_(True),
            "tick_p99": np.float32(20.5), "boundary_jump_p99": float("nan"), "nav": np.zeros(3, np.float32)})
    el.add({"ep": 1, "seed": 2001, "end": EpisodeEnd.SAFETY_FAULT, "success": False})
    el.close()
    eps = load_episodes(tmp_path)
    assert eps[0] == {"ep": 0, "seed": 2000, "end": "SUCCESS", "success": True, "tick_p99": 20.5,
                      "boundary_jump_p99": None, "nav": [0.0, 0.0, 0.0]}
    assert eps[1]["end"] == "SAFETY_FAULT"


# --- Publisher ----------------------------------------------------------------------------------


def test_publisher_no_subscriber_never_blocks():
    pub = Publisher("tcp://127.0.0.1:*")
    assert not pub.url.endswith(":*") and int(pub.url.rsplit(":", 1)[1]) not in range(5555, 5573)
    msg = {"k": 1, "ep": 0, "t_ns": 123, "compute_us": 6.5, "nav": [0.0, 0.0, 0.0]}
    t0 = time.perf_counter()
    for _ in range(10_000):
        pub.tick(msg)
    dt = time.perf_counter() - t0
    pub.close()
    assert dt < 0.1, dt


def test_publisher_drops_on_hwm_with_stalled_subscriber(zmq_ctx):
    import zmq

    pub = Publisher("tcp://127.0.0.1:*", hwm=100)
    sub = zmq_ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 10)
    sub.setsockopt(zmq.RCVBUF, 65536)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(pub.url)
    time.sleep(0.3)  # slow joiner
    img = np.zeros((120, 160, 3), np.uint8)
    t0 = time.perf_counter()
    for i in range(2000):  # 2000 x 57.6 kB, far beyond HWMs and socket buffers
        pub.ego(i, i, img)
    dt = time.perf_counter() - t0
    got = 0
    while sub.poll(200):
        sub.recv_multipart()
        got += 1
    pub.close()
    assert dt < 1.0, dt
    assert 0 < got < 2000


def test_publisher_topics_roundtrip(zmq_ctx):
    import zmq

    pub = Publisher("tcp://127.0.0.1:*")
    sub = zmq_ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(pub.url)
    time.sleep(0.3)
    row = np.zeros((), TICK_DTYPE)[()]
    row["k"] = 7
    row["nav_wbc"] = (0.2, 0.0, 0.1)
    pub.tick(telemetry.tick_msg(row))
    pub.req({"seq": np.uint32(3), "d": np.int16(4)})
    pub.ego(3, 16, np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3) % 251)
    pub.meta({"sim": "cpu"})
    got = {}
    while len(got) < 4 and sub.poll(1000):
        topic, payload = sub.recv_multipart()
        got[topic] = wire.loads(payload)
    pub.close()
    assert got[b"tick"]["k"] == 7 and got[b"tick"]["nav"] == pytest.approx([0.2, 0.0, 0.1])
    assert got[b"req"] == {"seq": 3, "d": 4}
    assert got[b"ego"]["k_obs"] == 16 and got[b"ego"]["img"].shape == (120, 160, 3)
    assert got[b"meta"] == {"sim": "cpu"}
    assert pub.sent == 4 and pub.dropped == 0


def test_publisher_disabled_is_noop():
    pub = Publisher(None)
    pub.tick({"k": 1})
    pub.close()
    assert pub.sent == 0


# --- collect_config -----------------------------------------------------------------------------


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True,
                   capture_output=True)


def test_collect_config_hashes_imported_dirty_files(tmp_path):
    repo = tmp_path / "tree"
    (repo / "pkg").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "tlm_clean.py").write_text("X = 1\n")
    (repo / "tlm_dirty.py").write_text("X = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    (repo / "tlm_dirty.py").write_text("X = 2\n")
    (repo / "tlm_new.py").write_text("X = 3\n")
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "tlm_sub.py").write_text("X = 4\n")
    nested = repo / "nested"
    nested.mkdir()
    _git(nested, "init", "-q")
    (nested / "tlm_nested.py").write_text("X = 5\n")
    _git(nested, "add", ".")
    _git(nested, "commit", "-qm", "n")
    sys.path[:0] = [str(repo), str(nested)]
    names = ("tlm_clean", "tlm_dirty", "tlm_new", "pkg", "pkg.tlm_sub", "tlm_nested")
    try:
        for name in names:
            importlib.import_module(name)
        cfg = collect_config(["loop", "--mode", "rt"], {"args": argparse.Namespace(mode="rt", s=16)},
                             trees={"T": repo})
    finally:
        sys.path.remove(str(repo))
        sys.path.remove(str(nested))
        for name in names:
            sys.modules.pop(name, None)
    got = {d["path"]: d for d in cfg["imported_dirty"]}
    assert set(got) == {"tlm_dirty.py", "tlm_new.py", "pkg/__init__.py", "pkg/tlm_sub.py", "nested/tlm_nested.py"}
    assert got["tlm_dirty.py"]["status"] == "M" and got["tlm_new.py"]["status"] == "??"
    assert got["tlm_dirty.py"]["sha256"] == hashlib.sha256(b"X = 2\n").hexdigest()
    assert cfg["git"]["T"]["dirty"] and len(cfg["git"]["T"]["head"]) == 40
    assert "nested/" in cfg["git"]["T"]["nested"] and len(cfg["git"]["T"]["nested"]["nested/"]["head"]) == 40
    assert cfg["args"] == {"mode": "rt", "s": 16} and cfg["argv"] == ["loop", "--mode", "rt"]


def test_collect_config_default_fields():
    t0 = time.perf_counter()
    cfg = collect_config(sys.argv, {"endpoints": {"vla": "tcp://127.0.0.1:5565"}})
    dt = time.perf_counter() - t0
    json.dumps(cfg, allow_nan=False)
    assert set(cfg["env"]) == set(telemetry.CONFIG_ENV)
    assert cfg["affinity"] == sorted(os.sched_getaffinity(0)) and cfg["hostname"] == socket.gethostname()
    assert set(cfg["git"]) == {"RT", "R", "W"}
    for name in ("R", "W"):
        assert len(cfg["git"][name]["head"]) == 40 and isinstance(cfg["git"][name]["status"], list)
    assert any(d["tree"] == "RT" and d["path"] == "rtloop/telemetry.py" for d in cfg["imported_dirty"]) or \
        cfg["git"]["RT"]["head"] is not None
    assert cfg["endpoints"]["vla"].endswith(":5565")
    assert dt < 5.0, dt


# --- analyze_run on synthetic runs --------------------------------------------------------------


def synth(n_eps: int = 3, K: int = 320, seed: int = 0):
    """A run that passes m1_rt: 20 ms +- 0.2 ms ticks, GPU WBC ~0.65 ms, d = 4, VLA 65 ms, one in flight."""
    rng = np.random.default_rng(seed)
    ticks = np.zeros(n_eps * K, TICK_DTYPE)
    reqs, eps, seq = [], [], 0
    for e in range(n_eps):
        t0 = 10**12 + e * 40 * 10**9
        t = ticks[e * K:(e + 1) * K]
        k = np.arange(K)
        t["ep"], t["k"] = e, k
        t["t_start_ns"] = t0 + k * TICK_NS + rng.integers(0, 200_000, K)
        t["late_us"] = (t["t_start_ns"] - t0 - k * TICK_NS) / 1e3
        t["compute_us"] = 6500 + rng.normal(0, 300, K)
        t["sim_us"], t["viewer_us"] = 5300, 800
        t["viewer_us"][0] = 1500
        t["wbc_rtt_us"] = 600 + np.abs(rng.normal(0, 80, K))
        t["wbc_dev_us"] = 150
        t["wbc_be"] = WbcBackend.GPU
        t["shadow_da"] = 1e-6
        ep_reqs = [dict(seq=seq, gen=e, ep=e, k_obs=0, k_arr=0, d=0, t_send_ns=t0 - 70_000_000,
                        t_recv_ns=t0 - 5_000_000, outcome=Outcome.APPLIED)]
        seq += 1
        for ko in range(16, K - 8, 16):
            ts = int(t["t_start_ns"][ko]) + 300_000
            ep_reqs.append(dict(seq=seq, gen=e, ep=e, k_obs=ko, k_arr=ko + 4, d=4, t_send_ns=ts,
                                t_recv_ns=ts + 65_000_000, srv_infer_ms=55.0, outcome=Outcome.APPLIED))
            seq += 1
        k_arr = np.array([q["k_arr"] for q in ep_reqs])
        active = np.searchsorted(k_arr, k, side="right") - 1
        k_obs = np.array([q["k_obs"] for q in ep_reqs])[active]
        t["row"] = t["age"] = k - k_obs
        t["rows_left"] = 30 - t["row"]
        t["chunk_seq"] = np.array([q["seq"] for q in ep_reqs])[active]
        t["splice"] = np.isin(k, k_arr)
        t["pelvis_z"] = 0.75
        reqs += ep_reqs
        eps.append({"ep": e, "seed": 2000 + e, "success": True, "end": "SUCCESS", "first_success_tick": K - 1})
    return ticks, reqs, eps


SYNTH_CONFIG = {"synthetic": True, "args": {"viewer": True}}  # T-1.4 conditions: viewer on, remote WBC


def write_run(root: Path, ticks, reqs, eps, tag: str = "synth", config: dict | None = None) -> Path:
    run = RunDir(root=root, tag=tag, config=SYNTH_CONFIG if config is None else config)
    tl = TickLog(run, capacity=256)
    for i in range(len(ticks)):
        r = tl.next()
        for name in TICK_DTYPE.names:
            r[name] = ticks[i][name]
        tl.commit()
    tl.close()
    rl = RequestLog(run)
    for q in reqs:
        q = dict(q)
        outcome = q.pop("outcome")
        rl.add(q)
        rl.update(q["seq"], outcome=outcome)
    rl.close()
    el = EpisodeLog(run)
    for e in eps:
        el.add(e)
    el.close()
    return run.path


def _failing(verdict) -> set[str]:
    return {c["name"] for c in verdict["checks"] if not c["pass"]}


def test_analyze_synthetic_run_passes_m1_rt(tmp_path):
    run = write_run(tmp_path, *synth())
    v = ar.analyze(run, "m1_rt", plots=False)
    assert v["pass"], v["table"]
    assert len(v["checks"]) == 18 and _failing(v) == set()  # 17 gates + run_data_complete
    assert v["checks"][0]["name"] == "run_data_complete" and v["checks"][0]["pass"] is True
    m = v["metrics"]
    assert m["n_ticks"] == 960 and m["d_p50"] == 4 and m["d_p99"] == 4 and m["inflight_max"] == 1
    assert m["vla_p50_ms"] == pytest.approx(55.0)  # gated: the server's time (srv_infer_ms here)
    assert m["vla_rtt_p50_ms"] == pytest.approx(65.0)  # reported: client-seen, tick-quantised
    assert m["viewer_k0_ms"] == pytest.approx(1.5)
    assert 19.8 < m["period_p50_ms"] < 20.2 and m["period_max_ms"] < 20.3
    saved = json.loads((run / "verdict.json").read_text())
    assert saved["gates"] == "m1_rt" and saved["pass"] is True
    assert {"name", "value", "threshold", "pass"} <= set(saved["checks"][0])


def _period_spike(t, r, e):
    sel = (t["ep"] == 0) & (t["k"] >= 100)
    t["t_start_ns"][sel] += 15_000_000


def _slip(t, r, e):
    t["slip"][50] = 1


def _overrun(t, r, e):
    t["compute_us"][::50] = 25_000


def _viewer_k0(t, r, e):
    t["viewer_us"][(t["ep"] == 1) & (t["k"] == 0)] = 5000


def _wbc_p99(t, r, e):
    t["wbc_rtt_us"][::40] = 3000


def _shadow(t, r, e):
    t["shadow_da"][::40] = 1e-3


def _d_late(t, r, e):
    for q in r[1::10]:
        if q["k_obs"]:
            q["d"], q["k_arr"] = 6, q["k_obs"] + 6


def _two_in_flight(t, r, e):
    r[5]["t_send_ns"] = r[4]["t_send_ns"] + 10_000_000
    r[5]["t_recv_ns"] = r[5]["t_send_ns"] + 65_000_000


def _starved(t, r, e):
    t["starved"][10:13] = 1


def _safety(t, r, e):
    e[1].update(end="SAFETY_FAULT", success=False)


def _slow_vla(t, r, e):
    for q in r:
        if q["k_obs"]:
            q["srv_infer_ms"] = 90.0


@pytest.mark.parametrize("mutate, failing", [
    (_period_spike, {"period_max_ms"}),
    (_slip, {"slips_max"}),
    (_overrun, {"overrun_frac_max"}),
    (_viewer_k0, {"viewer_k0_ms_max"}),
    (_wbc_p99, {"wbc_rtt_p99_ms"}),
    (_shadow, {"shadow_da_p99_max"}),
    (_d_late, {"d_p99_max"}),
    (_two_in_flight, {"inflight_max"}),
    (_starved, {"starved_ticks_max"}),
    (_safety, {"safety_faults_max"}),
    (_slow_vla, {"vla_p50_ms_max"}),
])
def test_analyze_crafted_failures(tmp_path, mutate, failing):
    ticks, reqs, eps = synth()
    mutate(ticks, reqs, eps)
    v = ar.analyze(write_run(tmp_path, ticks, reqs, eps), "m1_rt", plots=False)
    assert not v["pass"] and _failing(v) == failing, v["table"]


def test_analyze_wbc_excludes_ort_rows(tmp_path):
    ticks, reqs, eps = synth()
    fb = np.zeros(len(ticks), bool)
    fb[::48] = True  # ~2% fallback ticks: ORT after a 5 ms timeout
    ticks["wbc_be"][fb] = WbcBackend.ORT
    ticks["wbc_fb"][fb] = 1
    ticks["wbc_rtt_us"][fb] = 5000
    v = ar.analyze(write_run(tmp_path, ticks, reqs, eps), "m1_rt", plots=False)
    assert _failing(v) == {"wbc_fallback_frac_max"}
    assert v["metrics"]["wbc_rtt_p99_ms"] < 1.0 and v["metrics"]["wbc_fallback_frac"] == pytest.approx(fb.mean())


def test_analyze_ort_only_run_uses_all_rows(tmp_path):
    ticks, reqs, eps = synth()
    ticks["wbc_be"] = WbcBackend.ORT
    ticks["wbc_rtt_us"] = 50
    m = ar.analyze(write_run(tmp_path, ticks, reqs, eps), "m1_sim_smoke", plots=False)["metrics"]
    assert m["wbc_ort_only"] and m["wbc_rtt_p50_ms"] == pytest.approx(0.05)


def test_analyze_negctl(tmp_path):
    ticks, reqs, eps = synth()
    ticks["wbc_rtt_us"][::20] = 25_000  # 5% of remote calls > 20 ms
    run = write_run(tmp_path, ticks, reqs, eps)
    assert ar.analyze(run, "m1_negctl", plots=False)["pass"]
    v = ar.analyze(run, "m1_rt", plots=False)
    assert {"wbc_rtt_over20_frac_max", "wbc_rtt_p99_ms"} <= _failing(v)


def test_analyze_preroll_and_timeouts_excluded():
    reqs = np.zeros(6, REQUEST_DTYPE)
    reqs["ep"] = [0, 0, 1, 1, 2, 2]
    reqs["k_obs"] = [0, 16, 0, 16, 0, 16]
    reqs["d"] = [0, 4, 0, 4, 0, 0]
    reqs["outcome"] = [Outcome.APPLIED] * 5 + [Outcome.TIMEOUT]
    reqs["t_send_ns"] = [1, 100, 1000, 1100, 2000, 2100]
    reqs["t_recv_ns"] = [50, 165, 1050, 1165, 2050, 0]
    m = ar.compute_metrics(np.zeros(0, TICK_DTYPE), reqs, [])
    assert m["d_p50"] == 4 and m["d_max"] == 4 and m["n_applied"] == 5
    assert m["vla_rtt_p50_ms"] == pytest.approx(65e-6)  # ns -> ms of the two non-pre-roll replies
    assert m["inflight_max"] == 1


def test_analyze_unknown_gate_key_raises(tmp_path):
    gates = tmp_path / "gates.yaml"
    gates.write_text("bad:\n  period_p99_ms: 21.0\n  period_p99_typo_ms: 21.0\n")
    run = write_run(tmp_path, *synth(n_eps=1))
    with pytest.raises(ValueError, match="unknown gate key"):
        ar.analyze(run, "bad", gates, plots=False)
    with pytest.raises(ValueError):
        ar.apply_gates({"period_p99_ms": 1.0}, {"nope": 1})
    with pytest.raises(ValueError, match="not in"):
        ar.load_gates(gates, "m1_rt")


def test_analyze_cli_exit_codes(tmp_path):
    good = write_run(tmp_path / "a", *synth(n_eps=1))
    ticks, reqs, eps = synth(n_eps=1)
    _slip(ticks, reqs, eps)
    bad = write_run(tmp_path / "b", ticks, reqs, eps)
    gates = tmp_path / "g.yaml"
    gates.write_text("g:\n  slips_max: 0\n  mystery_metric: 1\n")

    def run(*args):
        return subprocess.run([sys.executable, str(ANALYZE), *map(str, args), "--no-plots"], capture_output=True,
                              text=True)

    p = run(good, "--gates", "m1_rt")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "VERDICT: PASS" in p.stdout and "period_p99_ms" in p.stdout
    p = run(bad, "--gates", "m1_rt")
    assert p.returncode == 1 and "VERDICT: FAIL" in p.stdout
    p = run(good, "--gates", "g", "--gates-file", gates)
    assert p.returncode == 2 and "mystery_metric" in p.stderr
    p = run(good, "--gates", "no_such_set")
    assert p.returncode == 2
    p = run(tmp_path / "missing", "--gates", "m1_rt")
    assert p.returncode == 2


def test_analyze_plots(tmp_path):
    pytest.importorskip("matplotlib")
    run = write_run(tmp_path, *synth(n_eps=2))
    v = ar.analyze(run, "m1_rt", plots=True)
    names = {Path(p).name for p in v["plots"]}
    assert names == {"histograms.png", "gantt_ep0.png", "boundary_jump_cdf.png"}
    assert all(Path(p).stat().st_size > 10_000 for p in v["plots"])


# --- eval_paired statistics ---------------------------------------------------------------------


@pytest.mark.parametrize("x, n, lo, hi", [
    (0, 10, 0.0, 0.27753),
    (5, 10, 0.23659, 0.76341),
    (25, 50, 0.366445, 0.633555),  # centre 0.5, half 1.959964 * sqrt(0.005384146) / 1.0768292
    (50, 50, 0.92865, 1.0),        # lower = n / (n + z^2)
    (1, 50, 0.003539, 0.104954),   # centre 0.0542469, half 0.0507076
])
def test_wilson_known_values(x, n, lo, hi):
    got = ev.wilson(x, n)
    assert got[0] == pytest.approx(lo, abs=1e-5) and got[1] == pytest.approx(hi, abs=1e-5)


@pytest.mark.parametrize("b, c, p", [
    (10, 3, 756 / 8192),  # 2 * (1 + 13 + 78 + 286) / 2^13
    (5, 0, 2 / 32),
    (1, 12, 28 / 8192),
    (7, 7, 1.0),
    (3, 2, 1.0),
    (0, 0, 1.0),
])
def test_mcnemar_exact_known_tables(b, c, p):
    assert ev.mcnemar_exact(b, c)[0] == pytest.approx(p, rel=1e-9)
    if b + c:
        assert ev.binom_two_sided_half(min(b, c), b + c) == pytest.approx(p, rel=1e-12)


def _pairs(n, b, c, both):
    sn = np.zeros(n, bool)
    sr = np.zeros(n, bool)
    sn[:b] = True
    sr[b:b + c] = True
    sn[b + c:b + c + both] = sr[b + c:b + c + both] = True
    return sn, sr


@pytest.mark.parametrize("b, c, delta, se, verdict", [
    (8, 5, 0.06, 0.0716100552, "no gross drop"),       # sqrt(13 - 9/50) / 50
    (2, 18, -0.32, 0.0771492061, "gross drop"),        # sqrt(20 - 256/50) / 50
    (0, 10, -0.20, 0.0565685425, "no gross drop"),     # L80 = -0.2476
    (0, 11, -0.22, 0.0585832740, "gross drop"),        # L80 = -0.2693
])
def test_paired_stats_hand_computed(b, c, delta, se, verdict):
    out = ev.paired_success(*_pairs(50, b, c, 15))
    assert (out["n"], out["b"], out["c"], out["both"]) == (50, b, c, 15)
    assert out["delta"] == pytest.approx(delta, abs=1e-12) and out["se"] == pytest.approx(se, abs=1e-9)
    assert out["lower80"] == pytest.approx(delta - 0.8416212336 * se, abs=1e-9)
    assert out["ci95"] == pytest.approx([delta - 1.959964 * se, delta + 1.959964 * se], abs=1e-6)
    assert out["verdict"] == verdict


def test_paired_stats_empty():
    out = ev.paired_success(np.zeros(0, bool), np.zeros(0, bool))
    assert out["n"] == 0 and out["delta"] is None and out["verdict"] == "no data"


def test_time_to_success_and_logrank():
    new = [{"success": True, "first_success_tick": 100} for _ in range(10)]
    ref = [{"success": True, "first_success_tick": 200} for _ in range(10)]
    t = ev.time_to_success(new, ref, 1440)
    assert t["rmst_diff"] == -100 and t["rmst_diff_ci95_boot"] == [-100, -100]
    assert t["logrank"]["o_minus_e"] == pytest.approx(5.0) and t["logrank"]["var"] == pytest.approx(2.5)
    assert t["logrank"]["p"] == pytest.approx(0.0015654, abs=1e-6)  # z = 5 / sqrt(2.5)
    fail = [{"success": False, "end": "TRUNC"} for _ in range(10)]
    t = ev.time_to_success(new, fail, 1440)
    assert t["rmst_ref"] == 1440 and t["rmst_diff"] == -1340


def test_seed_parsing_and_blocks():
    assert ev.parse_seeds("2000-2004,2010") == [2000, 2001, 2002, 2003, 2004, 2010]
    assert ev.parse_seeds([1, "3-4"]) == [1, 3, 4]
    assert ev.seed_blocks(ev.parse_seeds("2000-2049"), 10)[:2] == [(2000, 2009), (2010, 2019)]
    assert ev.seed_blocks([1, 2, 3, 7, 8], 0) == [(1, 3), (7, 8)]
    assert len(ev.seed_blocks(ev.parse_seeds("2000-2049"), 10)) == 5


def _episodes_dir(path: Path, seeds, success_fn, end_fail="TRUNC"):
    path.mkdir(parents=True)
    el = EpisodeLog(path)
    for s in seeds:
        ok = success_fn(s)
        el.add({"seed": s, "success": ok, "end": "SUCCESS" if ok else end_fail,
                "first_success_tick": 700 if ok else None})
    el.close()
    return path


def test_eval_report_only(tmp_path):
    c1 = _episodes_dir(tmp_path / "c1", range(2000, 2050), lambda s: s % 2 == 0)
    flip_on, flip_off = {2001, 2003, 2005}, {2000, 2002}

    def c4_ok(s):
        return (s % 2 == 0 or s in flip_on) and s not in flip_off

    a = _episodes_dir(tmp_path / "c4a", range(2000, 2025), c4_ok)
    b = _episodes_dir(tmp_path / "c4b", range(2025, 2050), c4_ok)
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, str(EVAL), "--report-only", f"C1={c1}", f"C4={a},{b}", "--compare", "C4,C1",
                        "--seeds", "2000-2049", "--out", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads((out / "report.json").read_text())
    arms = {x["arm"]: x for x in rep["arms"]}
    assert arms["C1"]["successes"] == 25 and arms["C4"]["successes"] == 26 and arms["C4"]["n"] == 50
    assert arms["C1"]["wilson95"] == pytest.approx([0.366445, 0.633555], abs=1e-6)
    cmp_ = rep["comparisons"][0]
    assert (cmp_["n"], cmp_["b"], cmp_["c"]) == (50, 3, 2) and cmp_["delta"] == pytest.approx(0.02)
    assert cmp_["p_mcnemar"] == pytest.approx(1.0) and cmp_["verdict"] == "no gross drop"
    md = (out / "report.md").read_text()
    assert "| C4 - C1 |" in md and "no gross drop" in md and "Wilson 95%" in md


def test_eval_report_only_gross_drop_exit(tmp_path):
    c1 = _episodes_dir(tmp_path / "c1", range(2000, 2050), lambda s: s < 2030)
    c4 = _episodes_dir(tmp_path / "c4", range(2000, 2050), lambda s: s < 2015, end_fail="FALL")
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, str(EVAL), "--report-only", f"C1={c1}", f"C4={c4}", "--out", str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 1, p.stdout + p.stderr
    rep = json.loads((out / "report.json").read_text())
    c = rep["comparisons"][0]
    assert (c["new"], c["ref"], c["b"], c["c"]) == ("C4", "C1", 0, 15) and c["verdict"] == "gross drop"
    assert c["falls_new"] == 35 and not c["falls_ok"]
    assert rep["verdict"].startswith("FAIL")


FAKE_LOOP = '''
import argparse, os, sys
from rtloop.telemetry import RunDir, EpisodeLog
ap = argparse.ArgumentParser()
ap.add_argument("--seeds"); ap.add_argument("--tag"); ap.add_argument("--mode"); ap.add_argument("--bias", type=int, default=0)
args, _ = ap.parse_known_args()
a, b = map(int, args.seeds.split("-"))
with open(os.environ["ORDER_FILE"], "a") as f:
    f.write(f"{args.tag} {a}-{b}\\n")
run = RunDir(tag=args.tag, config={"argv": sys.argv})
el = EpisodeLog(run)
for s in range(a, b + 1):
    ok = (s * 7) % 10 < 5 + args.bias
    el.add({"seed": s, "success": ok, "end": "SUCCESS" if ok else "TRUNC", "first_success_tick": 600 if ok else None})
el.close()
'''


def test_eval_runs_matrix_with_fake_loop(tmp_path, monkeypatch):
    fake = tmp_path / "fake_loop.py"
    fake.write_text(FAKE_LOOP)
    matrix = tmp_path / "m.yaml"
    matrix.write_text(textwrap.dedent("""
        name: fake
        seeds: "2000-2019"
        procs: 1
        block: 10
        common_args: "--mode sync"
        arms:
          - {name: C1, args: ""}
          - {name: C4, args: "--bias 1"}
        compare:
          - [C4, C1]
    """))
    order = tmp_path / "order.txt"
    env = dict(os.environ, ORDER_FILE=str(order))
    out = tmp_path / "out"
    cmd = [sys.executable, str(EVAL), "--matrix", str(matrix), "--out", str(out), "--loop-cmd",
           f"{sys.executable} {fake}"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert order.read_text().split("\n")[:4] == ["fake_C1 2000-2009", "fake_C4 2000-2009", "fake_C1 2010-2019",
                                                 "fake_C4 2010-2019"]  # blocks interleave across arms
    rep = json.loads((out / "report.json").read_text())
    arms = {x["arm"]: x for x in rep["arms"]}
    assert arms["C1"]["n"] == 20 and arms["C4"]["n"] == 20 and not arms["C1"]["missing_seeds"]
    assert len(arms["C4"]["run_dirs"]) == 2 and all("/runs/C4/" in d for d in arms["C4"]["run_dirs"])
    c = rep["comparisons"][0]
    assert (c["b"], c["c"]) == (2, 0)  # (7 s) % 10 == 5 for two seeds in 2000-2019
    assert len(list((out / "jobs").glob("*.json"))) == 4 and len(list((out / "logs").glob("*.log"))) == 4

    p = subprocess.run(cmd + ["--resume", "--procs", "2"], capture_output=True, text=True, env=env)
    assert p.returncode == 0 and p.stdout.count("[skip]") == 4
    assert len(order.read_text().split()) == 8  # nothing re-ran

    out2 = tmp_path / "out2"
    p = subprocess.run([*cmd[:5], str(out2), *cmd[6:], "--procs", "4", "--seeds", "2000-2004,2006"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads((out2 / "report.json").read_text())
    assert {x["arm"]: x["n"] for x in rep["arms"]} == {"C1": 6, "C4": 6}


def test_eval_job_failure_is_reported(tmp_path):
    matrix = tmp_path / "m.yaml"
    matrix.write_text("name: f\nseeds: '1-2'\narms:\n  - {name: A}\n  - {name: B}\ncompare: [[B, A]]\n")
    p = subprocess.run([sys.executable, str(EVAL), "--matrix", str(matrix), "--out", str(tmp_path / "o"),
                        "--loop-cmd", f"{sys.executable} -c 'import sys; sys.exit(4)'"],
                       capture_output=True, text=True)
    assert p.returncode == 3, p.stdout + p.stderr
    rep = json.loads((tmp_path / "o" / "report.json").read_text())
    assert len(rep["job_failures"]) == 2 and rep["verdict"].startswith("INCOMPLETE")


def test_eval_bad_matrix(tmp_path):
    matrix = tmp_path / "m.yaml"
    matrix.write_text("name: f\narms:\n  - {name: A}\ncompare: [[B, A]]\n")
    p = subprocess.run([sys.executable, str(EVAL), "--matrix", str(matrix), "--dry-run"], capture_output=True,
                       text=True)
    assert p.returncode == 2 and "unknown arm" in p.stderr


def test_example_matrices_parse():
    prim = ev.load_matrix(RT / "configs" / "eval_m1_primary.yaml")
    assert [a["name"] for a in prim["arms"]] == ["C1", "C4"] and prim["compare"][0]["gate"]
    assert ev.parse_seeds(prim["seeds"]) == list(range(2000, 2050)) and "--crn" in prim["common_args"]
    scr = ev.load_matrix(RT / "configs" / "eval_m1_screen.yaml")
    assert len(scr["arms"]) == 12 and len(ev.parse_seeds(scr["seeds"])) == 50
    assert [c for c in scr["compare"] if c["gate"]] == [{"new": "C4d4", "ref": "C1", "gate": True}]
    jobs = ev.plan_jobs(prim, Path("/x"), ["loop"], ev.parse_seeds(prim["seeds"]), 10)
    assert [j["id"] for j in jobs[:3]] == ["C1_2000-2009", "C4_2000-2009", "C1_2010-2019"]
    assert jobs[1]["cmd"][-4:] == ["--seeds", "2000-2009", "--tag", "m1_primary_C4"]


# --- analyze_run: the M2/M3 keys, operator pauses and resets, the remote shadow, x280d's heartbeat ring -----


def _gates(tmp_path, **keys) -> Path:
    import yaml

    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump({"g": keys}))
    return p


def _check(v, name):
    return next(c for c in v["checks"] if c["name"] == name)


def test_every_gates_yaml_set_uses_known_keys():
    import yaml

    sets = yaml.safe_load((RT / "gates.yaml").read_text())
    assert {"m1_rt", "m1_negctl", "m1_sim_smoke", "m2_rt", "m3_sim", "m3_rt", "m3_rt_slow"} <= set(sets)
    for name in sets:
        assert ar.load_gates(RT / "gates.yaml", name)
    m3 = sets["m3_rt"]
    assert m3["wbc_fallback_frac_max"] == 0.0 and m3["x280_heartbeat_continuity_min"] == 1.0  # §4.2 m3_rt row
    assert sets["m3_rt_slow"]["wbc_rtt_p99_ms"] == 9.0 and sets["m2_rt"]["starved_run_max"] == 10
    assert sets["m2_rt"]["vla_errors_max"] == 0 and sets["m2_rt"]["starved_frac_max"] == 0.001
    sim = sets["m3_sim"]  # §4.2: starvation "as m2" (D10); T-3.3 -> T-1.4: fallbacks <= 0.1%
    for name in ("m3_sim", "m3_rt", "m3_rt_slow"):
        assert {k: sets[name][k] for k in ("starved_frac_max", "starved_run_max")} == \
            {k: sets["m2_rt"][k] for k in ("starved_frac_max", "starved_run_max")}, name
        assert "starved_ticks_max" not in sets[name], name
    assert sim["wbc_fallback_frac_max"] == 0.001 and sim["wbc_rtt_p99_ms"] == 2.0


def test_analyze_starvation_and_vla_error_keys(tmp_path):
    ticks, reqs, eps = synth()
    ticks["starved"][100:112] = 1  # one run of 12 in episode 0 (0.0125 of 960 ticks)
    reqs[5]["outcome"] = Outcome.ERROR
    g = _gates(tmp_path, starved_frac_max=0.02, starved_run_max=10, vla_errors_max=0)
    v = ar.analyze(write_run(tmp_path, ticks, reqs, eps), "g", g, plots=False)
    m = v["metrics"]
    assert m["starved_run_max"] == 12 and m["starved_frac"] == pytest.approx(12 / len(ticks)) and m["vla_errors"] == 1
    assert _failing(v) == {"starved_run_max", "vla_errors_max"}


def test_analyze_excludes_operator_pauses(tmp_path):
    ticks, reqs, eps = synth(n_eps=1)
    ticks["t_start_ns"][150:] += 7_000_000_000  # a 7 s operator pause between tick 149 and 150
    ticks["period_us"][1:] = 20000.0
    ticks["period_us"][150] = np.nan  # what the loop logs on the first tick after a pause
    m = ar.analyze(write_run(tmp_path / "a", ticks, reqs, eps), "m1_rt", plots=False)
    assert m["pass"], m["table"]
    assert m["metrics"]["operator_pauses"] == 1 and m["metrics"]["period_max_ms"] < 21
    ticks["period_us"][150] = 20000.0  # no marker: the same gap is a timing failure
    v = ar.analyze(write_run(tmp_path / "b", ticks, reqs, eps), "m1_rt", plots=False)
    assert "period_max_ms" in _failing(v) and v["metrics"]["operator_pauses"] == 0


def test_analyze_allows_operator_resets(tmp_path):
    ticks, reqs, eps = synth(n_eps=3)
    eps = eps[:2]  # episode 2 was ended by HUD r: ticks, no EpisodeLog record
    run = write_run(tmp_path, ticks, reqs, eps)
    assert "run_data_complete" in _failing(ar.analyze(run, "m1_rt", plots=False))
    (run / "summary.json").write_text(json.dumps({"interrupted": False, "control": {"operator_resets": 1}}))
    v = ar.analyze(run, "m1_rt", plots=False)
    assert _check(v, "run_data_complete")["pass"] and v["run_info"]["operator_resets"] == 1


def test_analyze_remote_shadow_behind_ort(tmp_path):
    """T-3.6: --wbc ort --shadow URL. The shadow gates apply (not n/a), p99.9 is gated, misses are counted."""
    ticks, reqs, eps = synth()
    ticks["wbc_be"], ticks["wbc_rtt_us"] = WbcBackend.ORT, 60
    ticks["shadow_da"] = 6.8e-3
    ticks["shadow_da"][::100] = np.nan  # shadow misses
    ticks["shadow_da"][[5, 6, 7]] = 2e-2  # 0.3% excursions: p99 passes, p99.9 does not
    cfg = {"args": {"viewer": True, "wbc": "ort", "shadow": "tcp://127.0.0.1:5558"}}
    g = _gates(tmp_path, shadow_da_p99_max=1e-2, shadow_da_p999_max=1e-2)
    v = ar.analyze(write_run(tmp_path, ticks, reqs, eps, config=cfg), "g", g, plots=False)
    m = v["metrics"]
    assert m["shadow_remote"] and m["shadow_miss_frac"] == pytest.approx(np.isnan(ticks["shadow_da"]).mean())
    assert _check(v, "shadow_da_p99_max")["pass"] and not _check(v, "shadow_da_p999_max")["pass"]
    v = ar.analyze(write_run(tmp_path / "noshadow", ticks, reqs, eps, config={"args": {"wbc": "ort"}}), "g", g,
                   plots=False)
    assert _check(v, "shadow_da_p999_max")["status"] == "n/a"


def test_analyze_x280_frac_and_visible_wait(tmp_path):
    ticks, reqs, eps = synth()
    ticks["wbc_be"] = WbcBackend.X280
    ticks["wbc_be"][::200] = WbcBackend.ORT
    g = _gates(tmp_path, wbc_x280_frac_min=0.999, wbc_visible_wait_p99_ms=2.0)
    v = ar.analyze(write_run(tmp_path / "a", ticks, reqs, eps), "g", g, plots=False)
    assert v["metrics"]["wbc_x280_frac"] == pytest.approx(1 - len(ticks[::200]) / len(ticks))
    assert _failing(v) == {"wbc_x280_frac_min", "wbc_visible_wait_p99_ms"}
    assert "no early-dispatch ticks" in _check(v, "wbc_visible_wait_p99_ms")["note"]
    ticks["wbc_be"] = WbcBackend.X280_SIM
    ticks["wbc_early"] = 1
    ticks["wbc_rtt_us"] = 900
    v = ar.analyze(write_run(tmp_path / "b", ticks, reqs, eps), "g", g, plots=False)
    assert v["pass"] and v["metrics"]["wbc_visible_wait_p99_ms"] == pytest.approx(0.9)


def _ring(ticks, stall_at=(), start_ns=None, n=None):
    """An x280d_ticks.npy-shaped ring (TICK_LOG_DTYPE's fields that analyze_run reads, plus a few)."""
    dt = np.dtype([("t_recv_ns", "<i8"), ("tick", "<i8"), ("seq", "<u4"), ("status", "u1"), ("x280d_us", "<f4")])
    t = np.sort(ticks["t_start_ns"])
    if start_ns is not None:
        t = t[t >= start_ns]
    ring = np.zeros(len(t) if n is None else n, dt)
    ring["t_recv_ns"] = t[:len(ring)] + 1_000_000
    for i in stall_at:
        ring["status"][i] = ar.X280_ST_STALL
    return ring


def test_analyze_x280_heartbeat_continuity(tmp_path, monkeypatch):
    monkeypatch.delenv("CHIP_SESSION_DIR", raising=False)
    ticks, reqs, eps = synth()
    ticks["t_end_ns"] = ticks["t_start_ns"] + 7_000_000
    g = _gates(tmp_path, x280_heartbeat_continuity_min=1.0)
    run = write_run(tmp_path / "a", ticks, reqs, eps)
    v = ar.analyze(run, "g", g, plots=False)
    assert not v["pass"] and "no x280d ring" in _check(v, "x280_heartbeat_continuity_min")["note"]
    np.save(run / "x280d_ticks.npy", _ring(ticks))
    v = ar.analyze(run, "g", g, plots=False)
    assert v["pass"] and v["metrics"]["x280_heartbeat"]["rows"] == len(ticks)
    np.save(run / "x280d_ticks.npy", _ring(ticks, stall_at=[17]))
    v = ar.analyze(run, "g", g, plots=False)
    assert not v["pass"] and v["metrics"]["x280_heartbeat_continuity"] == pytest.approx(1 - 1 / len(ticks))
    late = _ring(ticks, start_ns=int(ticks["t_start_ns"][400]))  # wrapped: the ring misses the run's start
    np.save(run / "x280d_ticks.npy", late)
    v = ar.analyze(run, "g", g, plots=False)
    assert v["metrics"]["x280_heartbeat_continuity"] is None and "reach back" in v["metrics"]["x280_heartbeat"]["why"]
    early = _ring(ticks)[:400]  # dumped while the run was still going
    np.save(run / "x280d_ticks.npy", early)
    v = ar.analyze(run, "g", g, plots=False)
    assert v["metrics"]["x280_heartbeat_continuity"] is None and "ends before" in v["metrics"]["x280_heartbeat"]["why"]
    # elsewhere: --x280-ticks, then the session dir recorded in config.json
    (run / "x280d_ticks.npy").unlink()
    sess = tmp_path / "session"
    sess.mkdir()
    np.save(sess / "x280d_ticks.npy", _ring(ticks))
    assert ar.analyze(run, "g", g, plots=False, x280_ticks=sess / "x280d_ticks.npy")["pass"]
    run2 = write_run(tmp_path / "b", ticks, reqs, eps, config={**SYNTH_CONFIG, "env": {"CHIP_SESSION_DIR": str(sess)}})
    v = ar.analyze(run2, "g", g, plots=False)
    assert v["pass"] and v["metrics"]["x280_ring"] == str(sess / "x280d_ticks.npy")
    np.save(sess / "x280d_ticks.npy", np.zeros(3, [("a", "<i8")]))
    v = ar.analyze(run2, "g", g, plots=False)
    assert not v["pass"] and "not an x280d tick ring" in v["metrics"]["x280_ring"]


def test_collect_config_records_the_chip_session(monkeypatch):
    monkeypatch.setenv("CHIP_SESSION_DIR", "/x/y")
    cfg = collect_config(["loop"], {})
    assert cfg["env"]["CHIP_SESSION_DIR"] == "/x/y" and "CHIP_SESSION_ID" in cfg["env"]
