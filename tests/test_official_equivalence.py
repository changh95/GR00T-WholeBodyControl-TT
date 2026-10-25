"""T-1.3: the loop's own sync mode is bit-identical to the official rollout stack (PLAN.md §2 M1, WP-1.8).

Record: the GR00T N1.6 VLA (``rtserve.gpu_vla_server`` on a free port, or ``--vla`` / ``T13_VLA``)
sits behind a ``record-proxy`` mock, and ``rtloop.loop --mode sync --wbc ort --ort-opts default``
plays the seeds through it, so every chunk the loop executed is saved with the exact request it
answered. Replay: each seed then runs through the official stack (``tests/official_arm.py``: the env,
MultiStepWrapper(n_action_steps=20, max_episode_steps=1440, terminate_on_success=True),
SyncVectorEnv, PolicyClient and ``run_rollout_gymnasium_policy`` from ``rollout_policy.py``), in its
own subprocess, against a ``replay`` mock that answers with the recorded chunks and errors if a
request's state keys differ by any bit or its image sha1 differs.

Pass (PLAN.md T-1.3): 0 replay mismatches, and max |dq| = 0.0 on every tick both stacks ran (the
loop stops at the first success tick, the official stack at the end of that chunk). The loop's
TickLog keeps q in float32, so the loop runs through ``tests/loop_tap.py`` (the CLI in-process with
a passive float64 recorder on env.step, the twin of the official arm's recorder): q, dq and the
floating base pose/velocity are compared bit-exactly in float64, the TickLog's q43_meas / q43_cmd /
cmd35 / nav in float32. Every request's state is also compared bit-exactly by the replay, and both
arms report their ORT SessionOptions and env construction (gym.make kwargs, WBC config). Run:

    simpy -m pytest $RT/tests/test_official_equivalence.py -q          # starts its own GPU VLA
    T13_VLA=tcp://127.0.0.1:5565 T13_SEEDS=0-2 simpy -m pytest ...       # reuse a running server

The PLAN's form works too (tests/conftest.py registers the options; they win over the env vars):

    simpy -m pytest $RT/tests/test_official_equivalence.py -q --vla $VLA_GPU --seeds 0-4
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest
import zmq

from loop_utils import LoopRun, R, RT, load_run, start_mock, stop
from official_arm import ENV_NAME
from rtloop import wire

N_ACTION_STEPS = 20
MAX_TICKS = 1440
SRV_PYTHON = os.environ.get("SRV_PYTHON") or str(R / ".venv/bin/python")
MODEL = os.environ.get("VLA_CHECKPOINT", "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate")
SERVER_BOOT_S = 900
LOOP_TIMEOUT_S = 3600
ARM_TIMEOUT_S = 1200
UPSTREAM_TIMEOUT_MS = 120_000
FORBIDDEN_PORTS = range(5555, 5573)
EARLY_ENDS = ("FALL", "SAFETY_FAULT")  # the loop stops; the official stack would go on
STATE64 = ("q", "dq", "base_pose", "base_vel")  # recorded in float64 by both arms


# --- options ------------------------------------------------------------------------------------


def _option(config, name: str, env: str, default=None):
    v = config.getoption(name, default=None)
    return v if v is not None else os.environ.get(env, default)


def _parse_seeds(spec: str) -> list[int]:
    out = []
    for part in str(spec).split(","):
        a, sep, b = part.strip().partition("-")
        out.extend(range(int(a), int(b) + 1) if sep else [int(a)])
    return out


# --- servers ------------------------------------------------------------------------------------


def free_port() -> int:
    """A free port in 49152-65535 (zmq's random range, never 5555-5572)."""
    ctx = zmq.Context()
    s = ctx.socket(zmq.REP)
    try:
        return s.bind_to_random_port("tcp://127.0.0.1")
    finally:
        s.close(linger=0)
        ctx.term()


def rpc(url: str, endpoint: str, timeout_ms: int = 5000):
    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    try:
        s.connect(url)
        s.send(wire.request(endpoint))
        if not s.poll(timeout_ms, zmq.POLLIN):
            raise TimeoutError(f"{url} {endpoint}: no reply in {timeout_ms} ms")
        return wire.loads(s.recv())
    finally:
        s.close(linger=0)
        ctx.term()


class GpuVlaServer:
    """``srvpy -m rtserve.gpu_vla_server`` on a free port (CUDA visible for it only)."""

    def __init__(self, tmp: Path):
        self.port = free_port()
        self.url = f"tcp://127.0.0.1:{self.port}"
        self.log_path = tmp / "gpu_vla_server.log"
        self.log = open(self.log_path, "w")
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("GR00T_WBC_SERVER", None)
        env["PYTHONPATH"] = f"{RT}:{R}"
        cmd = [SRV_PYTHON, "-m", "rtserve.gpu_vla_server", "--port", str(self.port), "--model", MODEL,
               "--log", str(tmp / "gpu_vla_calls.jsonl")]
        self.t0 = time.monotonic()
        self.proc = subprocess.Popen(cmd, cwd=R, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def tail(self, n: int = 30) -> str:
        self.log.flush()
        return "".join(self.log_path.read_text(errors="replace").splitlines(True)[-n:])

    def wait_ready(self, timeout_s: float = SERVER_BOOT_S) -> float:
        while time.monotonic() - self.t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"gpu_vla_server exited {self.proc.returncode}\n{self.tail()}")
            if "gpu_vla_server ready" in self.log_path.read_text(errors="replace"):
                rep = rpc(self.url, "ping")
                assert isinstance(rep, dict) and rep.get("status") == "ok", rep
                return time.monotonic() - self.t0
            time.sleep(1.0)
        raise TimeoutError(f"gpu_vla_server not ready after {timeout_s} s\n{self.tail()}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        if not self.log.closed:
            self.log.close()


class TapLoopRun(LoopRun):
    """``tests/loop_tap.py --tap-out OUT -- <loop args>``: the loop CLI plus its float64 recorder."""

    def __init__(self, tmp_path: Path, name: str, tap_out: Path, *args: str):
        self.name = name
        self.root = Path(tmp_path) / name
        self.log_path = Path(tmp_path) / f"{name}.log"
        self.log = open(self.log_path, "w")
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{RT}:{R}"
        env.pop("GR00T_WBC_SERVER", None)
        cmd = [sys.executable, str(RT / "tests/loop_tap.py"), "--tap-out", str(tap_out), "--",
               *args, "--runs-root", str(self.root), "--tag", name]
        self.proc = subprocess.Popen(cmd, cwd=R, env=env, stdout=self.log, stderr=subprocess.STDOUT)


# --- recording ----------------------------------------------------------------------------------


def read_index(d: Path) -> list[dict]:
    return [json.loads(line) for line in (d / "index.jsonl").read_text().splitlines() if line.strip()]


def split_recording(rec_dir: Path, out: Path) -> dict[int, Path]:
    """One replay dir per loop episode (the replay mock serves a dir from its first entry)."""
    by_ep = defaultdict(list)
    for e in read_index(rec_dir):
        by_ep[int(e["rt"]["ep"])].append(e)
    dirs = {}
    for ep, entries in by_ep.items():
        d = out / f"ep{ep}"
        d.mkdir(parents=True)
        with open(d / "index.jsonl", "w") as f:
            for e in sorted(entries, key=lambda e: e["i"]):
                os.symlink(rec_dir / e["file"], d / e["file"])
                f.write(json.dumps(e) + "\n")
        modality = rec_dir / "modality_config.msgpack"
        if modality.exists():
            shutil.copy(modality, d / modality.name)
        dirs[ep] = d
    return dirs


def run_official(seed: int, replay_dir: Path, tmp: Path) -> dict:
    """One official arm (own process) against its own replay mock; returns its arrays and meta."""
    mock, url = start_mock("replay", "--replay-dir", str(replay_dir))
    port = int(url.rsplit(":", 1)[1])
    assert port not in FORBIDDEN_PORTS
    out = tmp / f"official_seed{seed}.npz"
    log_path = tmp / f"official_seed{seed}.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{RT}:{R}"
    env.pop("GR00T_WBC_SERVER", None)
    cmd = [sys.executable, str(RT / "tests/official_arm.py"), "--seed", str(seed), "--port", str(port),
           "--out", str(out), "--n-action-steps", str(N_ACTION_STEPS),
           "--max-episode-steps", str(MAX_TICKS)]
    try:
        with open(log_path, "w") as log:
            rc = subprocess.run(cmd, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=ARM_TIMEOUT_S).returncode
        replay = rpc(url, "info")
    finally:
        stop(mock)
    arrays = {}
    if out.exists():
        with np.load(out) as z:
            arrays = {k: z[k] for k in z.files}
    meta = json.loads(out.with_suffix(".json").read_text()) if out.with_suffix(".json").exists() else {}
    return {"rc": rc, "arrays": arrays, "meta": meta, "replay": replay, "log": log_path}


@pytest.fixture(scope="module")
def t13(tmp_path_factory, pytestconfig):
    tmp = tmp_path_factory.mktemp("official_equiv")
    seeds = _parse_seeds(_option(pytestconfig, "seeds", "T13_SEEDS", "0-4"))
    vla = _option(pytestconfig, "vla", "T13_VLA")
    if vla is not None and int(str(vla).rsplit(":", 1)[1]) == 5555:
        pytest.fail("T-1.3 must not use the existing :5555 server (PLAN.md D15)")
    server = proxy = loop = None
    out = {"seeds": seeds, "tmp": tmp}
    try:
        if vla is None:
            if not Path(SRV_PYTHON).exists():
                pytest.skip(f"server venv {SRV_PYTHON} missing")
            server = GpuVlaServer(tmp)
            out["server_boot_s"] = server.wait_ready()
            vla = server.url
        out["vla"] = vla
        rec_dir = tmp / "recording"
        proxy, proxy_url = start_mock("record-proxy", "--upstream", vla, "--record-dir", str(rec_dir),
                                      "--upstream-timeout-ms", str(UPSTREAM_TIMEOUT_MS))
        t0 = time.monotonic()
        tap = tmp / "loop_tap.npz"
        loop = TapLoopRun(tmp, "sync", tap, "--mode", "sync", "--vla", proxy_url, "--wbc", "ort",
                          "--ort-opts", "default", "--seeds", ",".join(map(str, seeds)),
                          "--max-ticks", str(MAX_TICKS), "--block-timeout-ms", str(UPSTREAM_TIMEOUT_MS))
        run_dir = loop.finish(LOOP_TIMEOUT_S)
        out["loop_wall_s"] = time.monotonic() - t0
        with np.load(tap) as z:
            out["tap"] = {k: z[k] for k in z.files}
        out["tap_meta"] = json.loads(tap.with_suffix(".json").read_text())
        out["loop_dir"] = run_dir
        out["ticks"], out["requests"], out["episodes"] = load_run(run_dir)
        out["config"] = json.loads((run_dir / "config.json").read_text())
        stop(proxy)
        proxy = None
        if server is not None:  # the official arms only talk to replay mocks
            server.stop()
            server = None
        out["recording"] = read_index(rec_dir)
        dirs = split_recording(rec_dir, tmp / "replay")
        out["official"] = {}
        t1 = time.monotonic()
        for ep, rec in enumerate(out["episodes"]):
            out["official"][rec["seed"]] = run_official(rec["seed"], dirs[ep], tmp)
        out["official_wall_s"] = time.monotonic() - t1
        yield out
    finally:
        if loop is not None:
            loop.kill()
        if proxy is not None:
            stop(proxy)
        if server is not None:
            server.stop()


# --- helpers over the fixture -------------------------------------------------------------------


def loop_episode(t13, seed: int):
    rec = next(e for e in t13["episodes"] if e["seed"] == seed)
    ticks = t13["ticks"][t13["ticks"]["ep"] == rec["ep"]]
    reqs = t13["requests"][t13["requests"]["ep"] == rec["ep"]]
    return rec, ticks, reqs


def tap_episode(t13, seed: int) -> dict:
    """The loop tap's float64 rows of one episode (episodes split at the tap's reset marks)."""
    rec = next(e for e in t13["episodes"] if e["seed"] == seed)
    tap = t13["tap"]
    starts = [int(x) for x in tap["resets"][:, 0]] + [len(tap["q"])]
    ep = rec["ep"]
    assert int(tap["resets"][ep, 1]) == seed, (ep, tap["resets"])
    return {k: tap[k][starts[ep]:starts[ep + 1]] for k in STATE64}


def expected_official_steps(rec: dict) -> int | None:
    """Steps the official stack runs: to the end of the success chunk, or the truncation."""
    if rec["end"] == "SUCCESS":
        return min(MAX_TICKS, (rec["first_success_tick"] // N_ACTION_STEPS + 1) * N_ACTION_STEPS)
    if rec["end"] == "TRUNC":
        return MAX_TICKS
    return None  # FALL / SAFETY_FAULT: the official stack runs on until the recording runs out


def first_diff(a: np.ndarray, b: np.ndarray):
    """(tick, column, |d|) of the first differing element of two (T, D) arrays, or None."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    bad = np.argwhere(d > 0)
    if not len(bad):
        return None
    k, j = (int(x) for x in bad[0])
    return k, j, float(d[k, j])


# --- tests --------------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.sim
def test_recording_complete(t13, capsys):
    eps, reqs, recs = t13["episodes"], t13["requests"], t13["recording"]
    assert [e["seed"] for e in eps] == t13["seeds"]
    assert t13["config"]["args"]["ort_opts"] == "default" and t13["config"]["args"]["wbc"] == "ort"
    assert len(recs) == len(reqs) and all(r["status"] == "ok" for r in recs)
    assert (reqs["outcome"] == 1).all() and (reqs["d"] == 0).all()  # APPLIED, sync
    assert [int(r["rt"]["seq"]) for r in recs] == [int(s) for s in reqs["seq"]]
    for e in eps:
        _, ticks, rq = loop_episode(t13, e["seed"])
        assert len(ticks) == e["n_ticks"] == e["k_end"] + 1
        assert list(rq["k_obs"]) == list(range(0, e["n_ticks"], N_ACTION_STEPS))
    up = np.array([r["upstream_ms"] for r in recs])
    with capsys.disabled():
        print(f"\n[T-1.3] VLA {t13['vla']} (boot {t13.get('server_boot_s', 0):.0f} s): "
              f"{len(recs)} chunks recorded, upstream p50 {np.median(up):.1f} ms; loop wall "
              f"{t13['loop_wall_s']:.0f} s; " + "; ".join(
                  f"seed {e['seed']}: {e['end']} at k={e['k_end']}" for e in eps))


@pytest.mark.gpu
@pytest.mark.sim
def test_zero_replay_mismatches(t13, capsys):
    lines = []
    for seed in t13["seeds"]:
        rec, ticks, rq = loop_episode(t13, seed)
        off = t13["official"][seed]
        meta, replay = off["meta"], off["replay"]
        assert replay["n_mismatch"] == 0, f"seed {seed}: {meta.get('error')} (log {off['log']})"
        n_rec = len(rq)
        assert replay["n_recorded"] == n_rec
        if rec["end"] in EARLY_ENDS:  # the official arm asks once more and finds the replay exhausted
            assert off["rc"] == 2 and "replay exhausted" in (meta.get("error") or ""), meta.get("error")
        else:
            assert off["rc"] == 0 and meta.get("error") is None, f"seed {seed}: {meta.get('error')}"
        assert replay["replay_pos"] == n_rec, (seed, replay)
        ticks_off = list(off["arrays"]["request_ticks"][:n_rec])
        assert ticks_off == list(rq["k_obs"]), f"seed {seed}: official request ticks {ticks_off}"
        assert meta["first_reset_kwargs"]["seed"] == str(seed)
        lines.append(f"seed {seed}: {replay['replay_pos']}/{n_rec} replayed, 0 mismatches, "
                     f"full GCs at {meta.get('full_gc_at')}")
    with capsys.disabled():
        print("\n[T-1.3] " + "; ".join(lines))


@pytest.mark.gpu
@pytest.mark.sim
def test_q_bit_identical_every_tick(t13, capsys):
    lines, worst = [], 0.0
    for seed in t13["seeds"]:
        rec, ticks, _ = loop_episode(t13, seed)
        a = t13["official"][seed]["arrays"]
        n_loop, n_off = len(ticks), len(a.get("q", ()))
        want = expected_official_steps(rec)
        if want is not None:
            assert n_off == want, f"seed {seed}: official ran {n_off} steps, expected {want}"
        n = min(n_loop, n_off)
        assert n == n_loop, f"seed {seed}: official stopped at {n_off} < loop {n_loop}"
        q_off = a["q"][:n].astype(np.float32)
        q_loop = ticks["q43_meas"][:n]
        dq = float(np.abs(q_off.astype(np.float64) - q_loop.astype(np.float64)).max())
        worst = max(worst, dq)
        fd = first_diff(q_off, q_loop)
        assert fd is None, f"seed {seed}: q differs first at tick {fd[0]}, joint {fd[1]}, |d| {fd[2]:.3g}"
        checks = {
            "cmd35": (a["cmd35"][:n], ticks["cmd35"][:n]),
            "q43_cmd": (a["q_cmd"][:n].astype(np.float32), ticks["q43_cmd"][:n]),
            "nav_wbc": (a["nav_wbc"][:n].astype(np.float32), ticks["nav_wbc"][:n]),
            "pelvis_z": (a["base_pose"][:n, 2:3].astype(np.float32), ticks["pelvis_z"][:n, None]),
        }
        for name, (x, y) in checks.items():
            fd = first_diff(x, y)
            assert fd is None, f"seed {seed}: {name} differs first at tick {fd[0]}, col {fd[1]}"
        lines.append(f"seed {seed}: {n} ticks compared (official {n_off}), max|dq| {dq}")
    with capsys.disabled():
        print(f"\n[T-1.3] max |dq43| = {worst} (official vs loop sync, f32); " + "; ".join(lines))


@pytest.mark.gpu
@pytest.mark.sim
def test_same_success_ticks(t13):
    for seed in t13["seeds"]:
        rec, _, _ = loop_episode(t13, seed)
        off = t13["official"][seed]
        succ = off["arrays"]["success"]
        first = int(np.argmax(succ)) if succ.any() else None
        if rec["end"] in EARLY_ENDS:
            succ = succ[: rec["n_ticks"]]
            first = int(np.argmax(succ)) if succ.any() else None
        assert first == rec["first_success_tick"], (seed, first, rec["first_success_tick"])
        if rec["end"] not in EARLY_ENDS:
            assert off["meta"]["rollout_success"] == rec["success"], seed


@pytest.mark.gpu
@pytest.mark.sim
def test_identical_wbc_session_options(t13):
    """PLAN.md C-14: both arms run ORT with default SessionOptions on the CPU provider."""
    wbc = t13["config"]["wbc_info"]
    assert wbc["backend"] == "ort-cpu" and wbc["opts"] == "default"
    for seed in t13["seeds"]:
        sessions = t13["official"][seed]["meta"]["wbc_sessions"]
        for name in ("policy_1", "policy_2"):
            s = sessions[name]
            assert s["providers"] == ["CPUExecutionProvider"], (seed, s)
            assert s["intra_op_num_threads"] == 0 and s["inter_op_num_threads"] == 0, (seed, s)


@pytest.mark.gpu
@pytest.mark.sim
def test_state_f64_bit_identical(t13, capsys):
    """Full-precision env state per tick, loop tap vs official recorder: identical bytes."""
    worst, lines = dict.fromkeys(STATE64, 0.0), []
    for seed in t13["seeds"]:
        rec, ticks, _ = loop_episode(t13, seed)
        lp = tap_episode(t13, seed)
        a = t13["official"][seed]["arrays"]
        n = len(lp["q"])
        # SystemExit inside env.step (SAFETY_FAULT) leaves the last TickLog row without a tap row
        assert n == rec["n_ticks"] or (rec["end"] == "SAFETY_FAULT" and n == rec["n_ticks"] - 1), (
            seed, n, rec["n_ticks"])
        assert np.array_equal(lp["q"].astype(np.float32), ticks["q43_meas"][:n]), seed  # same q the loop logged
        assert len(a["q"]) >= n, f"seed {seed}: official ran {len(a['q'])} < loop {n} steps"
        for k in STATE64:
            x, y = a[k][:n], lp[k]
            assert x.dtype == y.dtype == np.float64 and x.shape == y.shape, (seed, k, x.shape, y.shape)
            if x.tobytes() != y.tobytes():
                fd = first_diff(x, y)
                pytest.fail(f"seed {seed}: {k} (f64) differs first at tick {fd and fd[0]}, "
                            f"col {fd and fd[1]}, |d| {fd and fd[2]:.3g}")
            worst[k] = max(worst[k], float(np.abs(x - y).max()))
        lines.append(f"seed {seed}: {n} ticks")
    with capsys.disabled():
        print("\n[T-1.3] f64 bit-identical (" + ", ".join(f"max|d {k}| {v}" for k, v in worst.items())
              + "): " + "; ".join(lines))


@pytest.mark.gpu
@pytest.mark.sim
def test_identical_env_construction(t13):
    """PLAN.md T-1.3 rerun rule: same env id, gym.make kwargs, WBC config and wrapper class."""
    mine = t13["tap_meta"]["env"]
    assert mine["env_id"] == ENV_NAME, mine
    assert mine["make_kwargs"].get("offscreen") is True and mine["make_kwargs"].get("onscreen") is False, mine
    for seed in t13["seeds"]:
        assert t13["official"][seed]["meta"]["env"] == mine, seed
