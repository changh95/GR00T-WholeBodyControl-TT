"""T-1.2b: the alignment detector. Every executed arm target is the chunk row the index law names.

A mock `ramp` VLA answers each request with arm row i = state_sent(arms) + float32(i) * 1e-3 (in
float32), so the command logged at tick k identifies which row of which chunk was executed. From
ticks.npy and requests.npz alone, bit-exactly in float32 (PLAN.md §2 M1, T-1.2b):

  1. cmd35[ARMS](k) == state43_sent(k_obs)[arm slots] + float32(k - k_obs) * float32(1e-3) on every
     tick with a chunk (row i = k - k_obs: latency skipped, never re-anchored);
  2. state43_sent(k_obs) == q43_meas at the end of step k_obs - 1 (the request carries o_k, the
     state at the start of tick k, mapped from env q43 order to the VLA's state43 order);
  3. at each switch tick, k - k_obs equals the logged d.

Runs: VL at d_inj in {1, 3, 8} and RT against a 60 +- 5 ms mock; 2 episodes each, capped at 400
ticks; all four in parallel.
"""

import numpy as np
import pytest

from loop_utils import LoopRun, load_run, q43_to_state43, start_mock, stop
from rtloop.schema import ARMS, STATE_SLICES, Outcome

DELTA = np.float32(1e-3)  # mock_vla_server.RAMP_DELTA
ARM_STATE = np.r_[STATE_SLICES["left_arm"], STATE_SLICES["right_arm"]]  # state43 slots of A35 ARMS
MAX_TICKS = 400
VL_D = (1, 3, 8)
# The RT case is its own fixture and carries the `rt` marker, so `-m 'not rt'` runs the VL cases without ever
# starting a spin-paced RT loop (which must not run next to a real-time campaign).
NAMES = [f"vl_d{d}" for d in VL_D] + [pytest.param("rt_60ms", marks=pytest.mark.rt)]
TIMEOUT_S = 900
COMMON = ["--wbc", "ort", "--seeds", "0-1", "--max-ticks", str(MAX_TICKS), "--arm", "v1", "--s", "16", "--B", "0"]


@pytest.fixture(scope="module")
def vl_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("alignment")
    fast, url_fast = start_mock("ramp")
    loops = {}
    try:
        for d in VL_D:
            loops[f"vl_d{d}"] = LoopRun(tmp, f"vl_d{d}", "--mode", "vl", "--d-inj", str(d),
                                        "--vla", url_fast, *COMMON)
        yield {name: load_run(run.finish(TIMEOUT_S)) for name, run in loops.items()}
    finally:
        for run in loops.values():
            run.kill()
        stop(fast)


@pytest.fixture(scope="module")
def rt_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("alignment_rt")
    slow, url_slow = start_mock("ramp", "--latency-ms", "60", "--jitter-ms", "5", "--seed", "1")
    run = None
    try:
        run = LoopRun(tmp, "rt_60ms", "--mode", "rt", "--vla", url_slow, *COMMON)
        yield {"rt_60ms": load_run(run.finish(TIMEOUT_S))}
    finally:
        if run is not None:
            run.kill()
        stop(slow)


@pytest.fixture
def runs(request, name):
    """The run set that holds `name` (only that set's loops are started)."""
    return request.getfixturevalue("rt_runs" if name.startswith("rt") else "vl_runs")


def _by_seq(reqs):
    return {int(s): i for i, s in enumerate(reqs["seq"])}


@pytest.mark.sim
@pytest.mark.parametrize("name", NAMES)
def test_cmd_is_the_index_law_row(runs, name, capsys):
    ticks, reqs, eps = runs[name]
    assert len(eps) == 2 and len(ticks) == sum(e["n_ticks"] for e in eps)
    assert (ticks["row"] >= 0).all(), "the pre-roll chunk covers tick 0, so no tick may hold"
    assert (ticks["starved"] == 0).all()
    idx = _by_seq(reqs)
    j = np.array([idx[int(s)] for s in ticks["chunk_seq"]])
    assert np.array_equal(reqs["ep"][j], ticks["ep"]), "a tick executed another episode's chunk"
    assert (reqs["outcome"][j] == Outcome.APPLIED).all()
    i = ticks["k"] - reqs["k_obs"][j]
    assert np.array_equal(ticks["row"], i) and np.array_equal(ticks["age"], i)
    want = reqs["state43_sent"][j][:, ARM_STATE] + i.astype(np.float32)[:, None] * DELTA
    assert want.dtype == np.float32
    got = ticks["cmd35"][:, ARMS]
    bad = np.nonzero((got != want).any(axis=1))[0]
    assert bad.size == 0, f"{bad.size} ticks off the index law, first k={int(ticks['k'][bad[0]])}"
    with capsys.disabled():
        print(f"\n[T-1.2b {name}] {len(ticks)} ticks: cmd_arm == state_sent + (k - k_obs) * 1e-3 "
              f"bit-exactly; rows used {int(i.min())}..{int(i.max())}")


@pytest.mark.sim
@pytest.mark.parametrize("name", NAMES)
def test_state_sent_is_q43_after_previous_step(runs, name, capsys):
    ticks, reqs, _ = runs[name]
    row_of = {(int(e), int(k)): n for n, (e, k) in enumerate(zip(ticks["ep"], ticks["k"]))}
    checked = 0
    for r in reqs:
        if r["k_obs"] == 0:  # pre-roll: o_0 is the reset observation, not in the TickLog
            continue
        n = row_of[(int(r["ep"]), int(r["k_obs"]) - 1)]
        want = q43_to_state43(ticks["q43_meas"][n])
        assert np.array_equal(r["state43_sent"], want), (int(r["seq"]), int(r["k_obs"]))
        checked += 1
    assert checked >= 2 * (MAX_TICKS // 16 - 2)
    with capsys.disabled():
        print(f"\n[T-1.2b {name}] {checked} requests: state43_sent(k_obs) == q43_meas(k_obs - 1)")


@pytest.mark.sim
@pytest.mark.parametrize("name", NAMES)
def test_switch_offset_is_logged_d(runs, name, capsys):
    ticks, reqs, _ = runs[name]
    idx = _by_seq(reqs)
    sw = ticks[ticks["splice"] == 1]
    j = np.array([idx[int(s)] for s in sw["chunk_seq"]])
    off = sw["k"] - reqs["k_obs"][j]
    assert np.array_equal(off, reqs["d"][j])
    assert np.array_equal(sw["k"], reqs["k_arr"][j])
    applied = reqs[(reqs["outcome"] == Outcome.APPLIED) & (reqs["k_obs"] > 0)]
    if name.startswith("vl"):
        d_inj = int(name.split("_d")[1])
        assert (applied["d"] == d_inj).all()
        assert (off[sw["k"] > 0] == d_inj).all() and (off[sw["k"] == 0] == 0).all()
    else:
        assert ((applied["d"] >= 3) & (applied["d"] <= 5)).all(), np.unique(applied["d"])
    d, n = np.unique(applied["d"], return_counts=True)
    with capsys.disabled():
        print(f"\n[T-1.2b {name}] {len(sw)} switches, k - k_obs == d at each; d counts "
              f"{dict(zip(d.tolist(), n.tolist()))}")
