"""T-1.2: VL mode with v1, s = 20, d_inj = 0, B = 0 is bit-identical to the loop's own sync mode.

Both runs ask a mock `detfn` VLA (a deterministic function of the request's state bytes and k_obs)
on a free port, run the WBC on in-process ORT, and play seeds 0-4 for up to 1440 ticks. VL with
d_inj = 0 blocks on the request it sends at k = 0, 20, 40, ... and executes row 0 at k_obs, exactly
like sync, so every logged quantity of the physics must match to the bit and the episodes must end
on the same tick with the same outcome (PLAN.md §2 M1, T-1.2).
"""

import numpy as np
import pytest

from loop_utils import LoopRun, load_run, start_mock, stop

SEEDS = (0, 4)
MAX_TICKS = 1440
TIMEOUT_S = 1200


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vl_equiv")
    mock, url = start_mock("detfn")
    common = ["--vla", url, "--wbc", "ort", "--seeds", f"{SEEDS[0]}-{SEEDS[1]}",
              "--max-ticks", str(MAX_TICKS)]
    vl = sy = None
    try:
        vl = LoopRun(tmp, "vl", "--mode", "vl", "--arm", "v1", "--s", "20", "--d-inj", "0", "--B", "0",
                     *common)
        sy = LoopRun(tmp, "sync", "--mode", "sync", *common)
        yield {"vl": load_run(vl.finish(TIMEOUT_S)), "sync": load_run(sy.finish(TIMEOUT_S))}
    finally:
        for r in (vl, sy):
            if r is not None:
                r.kill()
        stop(mock)


@pytest.mark.sim
def test_same_episodes_and_end_ticks(runs, capsys):
    (_, _, ev), (_, _, es) = runs["vl"], runs["sync"]
    n = SEEDS[1] - SEEDS[0] + 1
    assert len(ev) == len(es) == n
    keys = ("seed", "end", "k_end", "n_ticks", "success", "first_success_tick")
    for a, b in zip(ev, es, strict=True):
        assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    with capsys.disabled():
        print("\n[T-1.2] " + "; ".join(f"seed {e['seed']}: {e['end']} at k={e['k_end']}" for e in ev))


@pytest.mark.sim
def test_q43_bit_identical_every_tick(runs, capsys):
    (tv, _, ev), (ts, _, _) = runs["vl"], runs["sync"]
    assert len(tv) == len(ts) == sum(e["n_ticks"] for e in ev)
    worst = 0.0
    for ep in np.unique(tv["ep"]):
        a, b = tv[tv["ep"] == ep], ts[ts["ep"] == ep]
        assert np.array_equal(a["k"], b["k"]) and np.array_equal(a["k"], np.arange(len(a)))
        dq = np.abs(a["q43_meas"].astype(np.float64) - b["q43_meas"].astype(np.float64)).max(axis=1)
        worst = max(worst, float(dq.max()))
        assert dq.max() == 0.0, f"ep {ep}: first difference at k={int(np.argmax(dq > 0))}"
        for field in ("cmd35", "q43_cmd", "q15", "nav_wbc", "pelvis_z", "pelvis_rp", "row", "policy"):
            assert np.array_equal(a[field], b[field]), (ep, field)
    walk = int(tv["policy"].sum())
    with capsys.disabled():
        print(f"\n[T-1.2] {len(tv)} ticks, max |dq43| = {worst} (VL vs sync); Walk ticks {walk}")
    assert walk > 0, "detfn's nav schedule should exercise the Walk policy"


@pytest.mark.sim
def test_same_requests(runs):
    (_, rv, _), (_, rs, _) = runs["vl"], runs["sync"]
    assert len(rv) == len(rs)
    for field in ("ep", "k_obs", "k_arr", "d", "state43_sent", "obs_sha1", "first_row", "last_row",
                  "n_rows_used", "outcome"):
        assert np.array_equal(rv[field], rs[field]), field
    assert (rv["k_obs"] % 20 == 0).all() and (rv["d"] == 0).all()
    # The arms sway, so every ego frame differs: a frozen frame means the render context broke
    # (robosuite's teardown bug, see sim_env.patch_render_context_teardown); equal sha1 across the
    # two processes means the frames are real renders, not stale buffer memory.
    assert len({bytes(h) for h in rv["obs_sha1"]}) == len(rv)
    used = rv["n_rows_used"][rv["n_rows_used"] > 0]
    assert used.max() == 20 and (rv["first_row"][rv["n_rows_used"] > 0] == 0).all()
