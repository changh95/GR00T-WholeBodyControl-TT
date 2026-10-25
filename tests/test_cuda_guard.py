"""T-1.7 CUDA/EGL guard: rt_sim never holds a CUDA context, and its ego frames do not depend on CUDA.

Two loop runs of the same three episodes (sync mode, mock `detfn`, ORT WBC, seeds 0-2, 60 ticks),
each under tests/guard_probe.py (the loop CLI in-process, passive recorders):

  hidden    the simpy environment: CUDA_VISIBLE_DEVICES="" (the shipped configuration);
  visible   CUDA_VISIBLE_DEVICES=0 and ``import torch`` first, but nothing asks torch about devices;
            the probe lets the loop past its own CUDA_VISIBLE_DEVICES == "" assertion and still
            fails the run if torch initialises CUDA or GR00T_WBC_SERVER is set.

Checks (PLAN.md §2 M1, T-1.7):
  1. hidden: CUDA_VISIBLE_DEVICES == "" at start, after all imports (env built), after each of the
     3 resets and at exit, and torch never initialised CUDA;
  2. both: at exit, while still alive, the process is absent from
     ``nvidia-smi --query-compute-apps=pid`` (EGL contexts are graphics-type and are not listed);
  3. the ego frame of every tick is a real render: uint8 (so never NaN), not all black, mean in a
     sane range, and no two ticks of an episode share a frame (a stale-buffer detector: robosuite's
     render-context teardown bug froze frames, see sim_env.patch_render_context_teardown);
  4. hidden vs visible: the ego frame of every (seed, tick) is bit-identical (sha1 of all 183 frames
     per run, full frames at ticks 0/30/59), and so are the float64 state, q and the RequestLog.

The viewer on/off half (test_viewer_on_off_frames) runs two VL loops (mock `detfn`, ORT WBC, seeds 0-1,
200 ticks), without and with ``--viewer``, and needs a usable $DISPLAY (the viewer process opens a GLFW
window); it skips without one. The env is built the same way in both (onscreen=False: the viewer is its own
process, rtloop.viewer_proc), so every ego frame, q43 and request obs_sha1 must be bit-identical, and the
viewer must really have run (two model generations shown, frames rendered, viewer_us logged).
"""

import json

import numpy as np
import pytest

from fault_utils import ProbeRun, display_usable, kill9, spawn_mock

SEEDS = "0-2"
N_EPS = 3
MAX_TICKS = 60
FRAMES_AT = "0,30,59"
TIMEOUT_S = 400
LOOP = ["--mode", "sync", "--wbc", "ort", "--seeds", SEEDS, "--max-ticks", str(MAX_TICKS)]


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cuda_guard")
    mock, url = spawn_mock("detfn")
    probe = ["--record-obs", "--frames-at", FRAMES_AT]
    runs = {}
    try:
        runs["hidden"] = ProbeRun(tmp, "hidden", probe, [*LOOP, "--vla", url])
        runs["visible"] = ProbeRun(tmp, "visible", [*probe, "--import-torch", "--allow-cuda-visible"],
                                   [*LOOP, "--vla", url], env={"CUDA_VISIBLE_DEVICES": "0"})
        for r in runs.values():
            r.finish(TIMEOUT_S)
        yield {name: {"probe": r.probe(), "obs": r.obs(), "run": r.load()} for name, r in runs.items()}
    finally:
        for r in runs.values():
            r.kill()
        kill9(mock)


@pytest.mark.sim
def test_cuda_hidden_after_imports_resets_and_exit(arms, capsys):
    p = arms["hidden"]["probe"]
    assert p["rc"] == 0 and p["error"] is None
    where = [c["where"] for c in p["checks"]]
    resets = [w for w in where if w.startswith("after reset")]
    assert "start" in where and "after imports" in where
    assert resets == [f"after reset (seed {s})" for s in range(N_EPS)], where
    for c in [*p["checks"], p["exit"]]:
        assert c["cvd"] == "", c
        assert c["torch_cuda_init"] is False, c
        assert c["gr00t_wbc_server"] is None, c
    # torch is imported by the sim stack (gr00t_wbc) and still never initialised CUDA
    assert p["exit"]["torch_imported"] is True
    with capsys.disabled():
        print(f"\n[T-1.7] CUDA_VISIBLE_DEVICES == '' at {len(p['checks'])} checkpoints "
              f"({', '.join(where[1:])}) and at exit; torch.cuda never initialised")


@pytest.mark.sim
@pytest.mark.parametrize("arm", ["hidden", "visible"])
def test_pid_not_a_cuda_compute_app(arms, arm, capsys):
    ex = arms[arm]["probe"]["exit"]
    assert ex["nvidia_compute_pids"] is not None, "nvidia-smi failed"
    assert ex["pid_listed"] is False, (arms[arm]["probe"]["pid"], ex["nvidia_compute_pids"])
    assert ex["torch_cuda_init"] is False
    with capsys.disabled():
        print(f"\n[T-1.7 {arm}] pid {arms[arm]['probe']['pid']} not in nvidia-smi compute apps "
              f"{ex['nvidia_compute_pids']} at exit")


@pytest.mark.sim
def test_visible_arm_really_saw_cuda(arms):
    p = arms["visible"]["probe"]
    assert p["rc"] == 0 and p["error"] is None
    assert all(c["cvd"] == "0" for c in [*p["checks"], p["exit"]])
    assert any(c["where"] == "probe: after import torch" and c["torch_imported"] for c in p["checks"])
    assert sum(c["where"].startswith("after reset") for c in p["checks"]) == N_EPS


@pytest.mark.sim
@pytest.mark.parametrize("arm", ["hidden", "visible"])
def test_ego_frames_are_real_renders(arms, arm, capsys):
    o = arms[arm]["obs"]
    assert o["frames"].dtype == np.uint8  # an integer image cannot hold NaN
    assert len(o["ep"]) == N_EPS * (MAX_TICKS + 1)  # o_0 .. o_60 per episode
    assert (o["img_max"] > 0).all(), "an all-black ego frame"
    assert ((o["img_mean"] > 10) & (o["img_mean"] < 245)).all(), (o["img_mean"].min(), o["img_mean"].max())
    for f in o["frames"]:
        assert f.std() > 10 and len(np.unique(f.reshape(-1, 3), axis=0)) > 100
    for ep in range(N_EPS):
        h = {bytes(x) for x in o["img_sha1"][o["ep"] == ep]}
        assert len(h) == MAX_TICKS + 1, f"ep {ep}: {MAX_TICKS + 1 - len(h)} repeated frames (stale render?)"
    with capsys.disabled():
        print(f"\n[T-1.7 {arm}] {len(o['ep'])} ego frames: none black, all distinct within an episode; "
              f"mean {o['img_mean'].min():.1f}..{o['img_mean'].max():.1f}")


@pytest.mark.sim
def test_frames_bit_identical_hidden_vs_visible(arms, capsys):
    a, b = arms["hidden"]["obs"], arms["visible"]["obs"]
    for key in ("ep", "k", "frame_idx"):
        assert np.array_equal(a[key], b[key]), key
    diff = np.nonzero((a["img_sha1"] != b["img_sha1"]).any(axis=1))[0]
    assert diff.size == 0, f"{diff.size} frames differ, first (ep, k) = {a['ep'][diff[0]], a['k'][diff[0]]}"
    assert a["frames"].tobytes() == b["frames"].tobytes()
    assert a["state43"].tobytes() == b["state43"].tobytes() and a["q"].tobytes() == b["q"].tobytes()
    (ta, ra, ea), (tb, rb, eb) = arms["hidden"]["run"], arms["visible"]["run"]
    assert [e["end"] for e in ea] == [e["end"] for e in eb] and len(ta) == len(tb)
    for field in ("q43_meas", "cmd35", "q15"):
        assert ta[field].tobytes() == tb[field].tobytes(), field
    assert np.array_equal(ra["obs_sha1"], rb["obs_sha1"]) and len(ra) == len(rb)
    with capsys.disabled():
        print(f"\n[T-1.7] CUDA hidden vs visible-but-unused: {len(a['ep'])} ego frames, "
              f"{len(ta)} ticks of q43 and {len(ra)} requests bit-identical")


VIEWER_LOOP = ["--mode", "vl", "--arm", "v1", "--d-inj", "4", "--wbc", "ort", "--seeds", "0-1",
               "--max-ticks", "200"]


@pytest.fixture(scope="module")
def viewer_arms(tmp_path_factory):
    if not display_usable():
        pytest.skip("T-1.7 viewer on vs off needs a usable $DISPLAY (--viewer opens a GLFW window)")
    tmp = tmp_path_factory.mktemp("viewer_on_off")
    mock, url = spawn_mock("detfn")
    runs = {}
    try:
        for name, extra in (("off", []), ("on", ["--viewer"])):
            runs[name] = ProbeRun(tmp, f"viewer_{name}", ["--record-obs"], [*VIEWER_LOOP, "--vla", url, *extra])
            runs[name].finish(TIMEOUT_S)
        yield {name: {"obs": r.obs(), "run": r.load(),
                      "config": json.loads((r.run_dir / "config.json").read_text())} for name, r in runs.items()}
    finally:
        for r in runs.values():
            r.kill()
        kill9(mock)


@pytest.mark.sim
def test_viewer_on_off_frames(viewer_arms, capsys):
    off, on = viewer_arms["off"], viewer_arms["on"]
    a, b = off["obs"], on["obs"]
    for key in ("ep", "k"):
        assert np.array_equal(a[key], b[key]), key
    diff = np.nonzero((a["img_sha1"] != b["img_sha1"]).any(axis=1))[0]
    assert diff.size == 0, f"{diff.size} ego frames differ, first (ep, k) = {a['ep'][diff[0]], a['k'][diff[0]]}"
    assert a["state43"].tobytes() == b["state43"].tobytes() and a["q"].tobytes() == b["q"].tobytes()
    (ta, ra, ea), (tb, rb, eb) = off["run"], on["run"]
    assert [(e["end"], e["k_end"]) for e in ea] == [(e["end"], e["k_end"]) for e in eb] and len(ta) == len(tb)
    for field in ("q43_meas", "q43_cmd", "cmd35", "q15"):
        assert ta[field].tobytes() == tb[field].tobytes(), field
    assert len(ra) == len(rb) and np.array_equal(ra["obs_sha1"], rb["obs_sha1"])
    # The viewer really ran in "on" (and not in "off"), and the env deviates from stock in neither.
    assert off["config"]["viewer"] is None and (ta["viewer_us"] == 0).all()
    end = on["config"]["viewer"]["end"]
    assert end["models_loaded"] == 2 and end["viewer_gen"] == 2 and end["frames"] > 0, end
    assert end["exited"] is None and end["terminated"] and end["rc"] == 0, end
    assert (tb["viewer_us"] > 0).all() and [e["viewer_handoff"]["gen"] for e in eb] == [1, 2]
    assert on["config"]["fidelity_deviations"] == off["config"]["fidelity_deviations"]
    assert not {"f", "g"} & set(on["config"]["fidelity_deviations"])
    with capsys.disabled():
        v = tb["viewer_us"].astype(np.float64)
        print(f"\n[T-1.7] viewer off vs on (VL, d_inj 4): {len(a['ep'])} ego frames, {len(ta)} ticks of q43 and "
              f"{len(ra)} requests bit-identical; viewer {end['frames']} frames, {end['models_loaded']} models; "
              f"publish p50 {np.percentile(v, 50):.1f} / p99 {np.percentile(v, 99):.1f} / max {v.max():.1f} us")
