"""The out-of-process viewer (rtloop.viewer + rtloop.viewer_proc; PLAN.md §1.1, R9).

Without a display: the seqlock never hands out a torn snapshot (writer here, reader in another process), the
model handoff round-trips bit-exactly through the MJB (xpos/xquat after mj_forward), superseded and closed
segments leave /dev/shm, and a viewer that dies is logged once while publishing goes on.
With a usable $DISPLAY (skipped otherwise; each opens a short-lived window): the viewer process follows the
generations in one window and close() terminates it; SIGKILL of the loop takes the viewer and the segments
with it (PR_SET_PDEATHSIG, the resource tracker).
sim: the real G1 env's model and state, handed off at two resets and after 300 ticks, reproduce the env's
xpos/xquat bit for bit; the tick-path publish cost is measured.
"""

import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from fault_utils import display_usable
from loop_utils import R, RT

from rtloop import viewer as vl
from rtloop.viewer_proc import StateReader

mujoco = pytest.importorskip("mujoco")

TOY = """<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="3 3 .1"/>
<body name="robot_pelvis" pos="0 0 1"><freejoint/><geom type="box" size=".2 .1 .3"/>{extra}</body>
<body name="target" mocap="true" pos="1 0 1"><geom type="sphere" size=".05" contype="0" conaffinity="0"/></body>
<site name="marker" pos="0 1 1" size=".05"/></worldbody></mujoco>"""
LIMB = ('<body pos="0 0 -.5"><joint type="hinge" axis="1 0 0"/><geom type="capsule" fromto="0 0 0 0 0 -.4" '
        'size=".05"/>{extra}</body>')


def toy(n_limbs: int = 0):
    extra = ""
    for _ in range(n_limbs):
        extra = LIMB.format(extra=extra)
    m = mujoco.MjModel.from_xml_string(TOY.format(extra=extra))
    return m, mujoco.MjData(m)


def segments(prefix: str) -> list[str]:
    return sorted(p.name for p in vl.MJB_DIR.glob(f"{prefix}_*"))


def forwarded(m, d):
    """mj_forward of a copy (the env's own data is mid-step: its xpos lag its qpos by one substep)."""
    c = copy.copy(d)
    mujoco.mj_forward(m, c)
    return c


def reproduce(reader: StateReader, gen_models: dict):
    """What the viewer process does with the latest snapshot: (seq, gen, k, model, data) after mj_forward."""
    seq, gen, k = reader.read()
    if gen not in gen_models:
        gen_models[gen] = reader.load_model(gen)
    m = gen_models[gen]
    d = mujoco.MjData(m)
    reader.apply(m, d)
    mujoco.mj_forward(m, d)
    return seq, gen, k, m, d


def assert_same_scene(ref, got) -> None:
    for f in ("xpos", "xquat", "geom_xpos", "geom_xmat", "site_xpos"):
        assert getattr(ref, f).tobytes() == getattr(got, f).tobytes(), f
    assert ref.time == got.time


def test_handoff_round_trip_and_cleanup():
    link = vl.ViewerLink()
    reader = None
    try:
        assert segments(link.prefix) == [f"{link.prefix}_s"]
        m, d = toy()
        d.qpos[:3] = [0.3, -0.2, 1.1]
        d.qpos[3:7] = np.array([0.9, 0.1, 0.3, 0.2]) / np.linalg.norm([0.9, 0.1, 0.3, 0.2])
        d.mocap_pos[0] = [1.5, 0.5, 0.7]
        d.time = 1.25
        m.geom_rgba[1] = [0.1, 0.2, 0.3, 0.4]  # what robosuite's visualize() would change at runtime
        info = link.handoff(m, d)
        assert info["gen"] == 1 and segments(link.prefix) == [f"{link.prefix}_m1", f"{link.prefix}_s"]
        reader = StateReader(link.prefix)
        models: dict = {}
        seq, gen, k, m1, d1 = reproduce(reader, models)
        assert (gen, k, seq) == (1, -1, 2)
        assert_same_scene(forwarded(m, d), d1)
        assert m1.geom_rgba.tobytes() == m.geom_rgba.tobytes()
        mujoco.mj_step(m, d)
        us = link.publish(7)
        assert 0 < us < 10_000
        seq, gen, k, _, d1 = reproduce(reader, models)
        assert (gen, k) == (1, 7) and d1.qvel.tobytes() == d.qvel.tobytes()
        assert_same_scene(forwarded(m, d), d1)

        m2, d2 = toy(n_limbs=3)  # a "hard reset": new model, other sizes
        d2.qpos[7:] = [0.4, -0.3, 0.2]
        mujoco.mj_forward(m2, d2)
        assert link.handoff(m2, d2)["gen"] == 2
        assert segments(link.prefix) == [f"{link.prefix}_m2", f"{link.prefix}_s"]  # gen 1 unlinked
        assert reader.load_model(1) is None  # a late viewer skips a superseded generation
        seq, gen, k, mm, dd = reproduce(reader, models)
        assert (gen, k) == (2, -1) and mm.nq == m2.nq
        assert_same_scene(d2, dd)  # d2 is forwarded: compare with the live data itself
    finally:
        if reader is not None:
            reader.close()
        link.close()
    assert segments(link.prefix) == []


READER = r"""
import sys, time
import numpy as np, mujoco
from rtloop.viewer_proc import StateReader
r = StateReader(sys.argv[1])
m = d = None
n = bad = 0
ks = set()
t_end = time.monotonic() + float(sys.argv[2])
while time.monotonic() < t_end:
    s = r.read()
    if s is None:
        continue
    seq, gen, k = s
    if m is None:
        m = r.load_model(gen)
        d = mujoco.MjData(m)
    r.apply(m, d)
    v = float(k)
    ok = ((d.qpos == v).all() and (d.qvel == v).all() and d.time == v and (d.mocap_pos == v).all()
          and (m.geom_rgba[:, 0] == np.float32(k % 4096)).all())
    bad += not ok
    n += 1
    ks.add(k)
r.close()
print(n, bad, len(ks), r.torn, flush=True)
"""


def test_seqlock_never_tears_across_processes(capsys):
    """Writer here as fast as Python goes; reader in another process. Every field of a snapshot carries the
    same counter, so a torn copy (fields from two publishes) or a header/payload mismatch is visible."""
    link = vl.ViewerLink()
    try:
        m, d = toy(n_limbs=8)
        link.handoff(m, d)
        env = dict(os.environ, PYTHONPATH=f"{RT}:{R}")
        p = subprocess.Popen([sys.executable, "-c", READER, link.prefix, "2.0"], env=env, cwd=RT,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        k = 0
        t_end = time.monotonic() + 4.0
        while p.poll() is None and time.monotonic() < t_end:
            k += 1
            d.qpos[:] = k
            d.qvel[:] = k
            d.mocap_pos[:] = k
            d.time = k
            m.geom_rgba[:, 0] = k % 4096
            link.publish(k)
        out, err = p.communicate(timeout=30)
        assert p.returncode == 0, err
        n, bad, distinct, torn = map(int, out.split())
    finally:
        link.close()
    with capsys.disabled():
        print(f"\n[viewer] seqlock: {k} publishes, {n} snapshots read ({distinct} distinct), {bad} inconsistent, "
              f"{torn} torn reads retried")
    assert bad == 0
    assert n > 1000 and distinct > 1000
    assert torn > 0  # the reader did run into writes in progress, and retried them


def test_dead_viewer_is_logged_once_and_publishing_goes_on(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DISPLAY", raising=False)  # GLFW cannot open a window: the viewer exits by itself
    link = vl.ViewerLink()
    try:
        info = link.spawn(tmp_path / "viewer.log")
        assert info["pid"] == link.proc.pid
        assert link.proc.wait(30) == 2
        m, d = toy()
        link.handoff(m, d)  # the reset-window check finds it dead
        for k in range(3 * vl.POLL_EVERY):
            link.publish(k)
        assert link.n == 3 * vl.POLL_EVERY + 1
        err = capsys.readouterr().err
        assert err.count("viewer process") == 1 and "the loop runs on without it" in err
        assert link.exited["rc"] == 2 and link.exited["k"] == -1
        out = link.close()
    finally:
        link.close()
    assert out["rc"] == 2 and "terminated" not in out
    assert "GLFW cannot open a window" in (tmp_path / "viewer.log").read_text()
    assert segments(link.prefix) == []


@pytest.fixture(scope="module")
def display():
    if not display_usable():
        pytest.skip("needs a usable $DISPLAY (opens a viewer window)")


def wait_for(cond, timeout_s: float = 30.0) -> bool:
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_viewer_process_follows_generations_and_closes(display, tmp_path, capsys):
    link = vl.ViewerLink(cpus=[6, 7])
    try:
        link.spawn(tmp_path / "viewer.log")
        m, d = toy()
        link.handoff(m, d)
        assert wait_for(lambda: link.stats()["viewer_gen"] == 1 and link.stats()["frames"] > 5), link.stats()
        m2, d2 = toy(n_limbs=2)
        link.handoff(m2, d2)
        for k in range(50):
            d2.qpos[7] = np.sin(k / 5)
            link.publish(k)
            time.sleep(0.02)
        assert wait_for(lambda: link.stats()["viewer_gen"] == 2), link.stats()
        assert link.check(0) and link.exited is None
        out = link.close()
    finally:
        link.close()
    log = (tmp_path / "viewer.log").read_text()
    assert out["models_loaded"] == 2 and out["frames"] > 5 and out["terminated"] and out["rc"] == 0, out
    assert {t["cpus"] for t in out["threads"]} == {"6-7"}, out["threads"]
    assert "gen 2 shown" in log and "exit (signal)" in log and "relaunching" not in log
    assert segments(link.prefix) == []
    with capsys.disabled():
        print(f"\n[viewer] window: {out['frames']} frames, {out['models_loaded']} models in one window, "
              f"threads {[t['cpus'] for t in out['threads']]}, rc {out['rc']} after SIGTERM")


LOOP_STANDIN = r"""
import sys, time
from pathlib import Path
import mujoco
from rtloop import viewer as vl
link = vl.ViewerLink()
link.spawn(Path(sys.argv[1]))
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="pelvis"><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
link.handoff(m, mujoco.MjData(m))
print(link.proc.pid, link.prefix, flush=True)
while True:
    link.publish(0)
    time.sleep(0.02)
"""


def test_viewer_and_segments_go_when_the_loop_is_killed(display, tmp_path):
    env = dict(os.environ, PYTHONPATH=f"{RT}:{R}")
    loop = subprocess.Popen([sys.executable, "-c", LOOP_STANDIN, str(tmp_path / "viewer.log")], env=env, cwd=RT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    viewer_pid = None
    try:
        viewer_pid, prefix = loop.stdout.readline().split()
        viewer_pid = int(viewer_pid)
        assert segments(prefix)
        assert wait_for(lambda: "gen 1 shown" in (tmp_path / "viewer.log").read_text())
        loop.kill()  # SIGKILL: no close(), no atexit
        loop.wait(10)
        assert wait_for(lambda: not Path(f"/proc/{viewer_pid}").exists() or
                        "Z" in Path(f"/proc/{viewer_pid}/stat").read_text().split()[2], 10), "viewer outlived the loop"
        assert wait_for(lambda: segments(prefix) == [], 10), segments(prefix)  # the loop's resource tracker
    finally:
        if loop.poll() is None:
            loop.kill()
            loop.wait(10)
        if viewer_pid and Path(f"/proc/{viewer_pid}").exists():
            os.kill(viewer_pid, signal.SIGKILL)
        loop.stdout.close()
        loop.stderr.close()


@pytest.mark.sim
def test_env_handoff_bit_identical_and_publish_cost(capsys):
    """The real G1 env: at two resets and over 300 ticks, the MJB + copied state reproduce the env's xpos/xquat."""
    from rtloop import sim_env
    from rtloop import wbc_backends as wb
    from rtloop.schema import NAV, hold_row, row_to_env_action

    wb.install(wb.WbcRuntime(wb.OrtCpu()))
    sim_env.patch_render_context_teardown()
    env = sim_env.make_env(onscreen=False)
    link = vl.ViewerLink()
    reader = None
    us, checked = [], 0
    try:
        sim_env.skip_first_step_check(env)
        h = None
        for seed in (0, 1):
            obs, _ = env.reset(seed=seed)
            h = h or sim_env.SimHandles(env)
            sim = h.robosuite.sim
            m, d = sim.model._model, sim.data._data
            info = link.handoff(m, d)
            reader = reader or StateReader(link.prefix)
            models: dict = {}
            _, gen, k, mv, dv = reproduce(reader, models)
            assert (gen, k) == (seed + 1, -1)
            assert_same_scene(d, dv)  # robosuite forwards after a reset: the live data is consistent
            checked += 1
            for k in range(300 if seed == 0 else 50):
                row = hold_row(obs)
                if 100 <= k < 200:
                    row[NAV] = [0.2, 0.0, 0.0]  # exercise Walk
                obs, *_ = env.step(row_to_env_action(row))
                us.append(link.publish(k))
                if k % 25 == 24:
                    _, gen, kk, mv, dv = reproduce(reader, models)
                    assert (gen, kk) == (seed + 1, k)
                    assert_same_scene(forwarded(m, d), dv)
                    checked += 1
            assert len(models) == 1 and info["mjb_mb"] > 1
    finally:
        if reader is not None:
            reader.close()
        link.close()
        env.close()
        wb.uninstall()
    us = np.asarray(us)
    with capsys.disabled():
        print(f"\n[viewer] G1 env: {checked} snapshots bit-identical (xpos, xquat, geom/site poses) at 2 resets "
              f"and every 25 ticks; MJB {info['mjb_mb']} MB, handoff {info['ms']} ms; publish over {len(us)} "
              f"ticks p50 {np.percentile(us, 50):.1f} / p99 {np.percentile(us, 99):.1f} / max {us.max():.1f} us")
    assert np.percentile(us, 50) < 50.0
