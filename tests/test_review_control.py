"""Adversarial review of the loop's control decisions (sim-free; tests/review_harness.py).

The harness runs the production tick code (Loop._ticks_async / _ticks_sync / _blocking_call /
_drain, the splicer, TickLog, RequestLog) against a fake env and a fake VLA whose reply is
pollable exactly d ticks after its k_obs, so RT with a fixed latency d is a deterministic
reference for VL at d_inj = d.

Regression (found in review, fixed in loop.py): VL used to register a due chunk in step 3b only
*after* the trigger, while RT registers it in step 2, *before* the trigger. Whenever the arrival
tick is also a trigger tick (d >= s_eff, or a LATE_REJECT retrigger), RT sent o_{k_arr} at k_arr
but VL waited until k_arr + 1, so VL at d_inj = d was not the RT policy at latency d (fewer
requests, older chunks, more starvation). Loop._ticks_async now runs 3b before and after the
trigger; CASES_LAG below are the cells that used to differ.
"""

from pathlib import Path

import numpy as np
import pytest

from review_harness import FakeEnv, make_loop, run_episode

from rtloop import splicer as sp
from rtloop.schema import ARMS, H, NAV, Outcome


def _run(tmp_path: Path, mode: str, arm: str, s: int, d: int, B: int = 0, max_ticks: int = 300,
         name: str = ""):
    lp = make_loop(tmp_path / f"{mode}_{arm}_{s}_{d}_{B}{name}", mode, arm=arm, s=s,
                   B=B, d_inj=d, d_init=d, max_ticks=max_ticks, d_of=lambda i: d)
    return run_episode(lp)


def _s_eff_at(arm: str, s: int, d: int) -> int:
    cfg = sp.SplicerConfig(arm=arm, s=s, d_init=d)
    return sp.s_eff(sp.new_state(), cfg)


# (arm, s, d): d < s_eff (always matched); d >= s_eff or late reject (used to lag one tick).
CASES_OK = [("v1", 16, d) for d in (1, 3, 4, 5, 8, 12)] + \
           [("v1", 10, d) for d in (3, 4, 8)] + \
           [("sync_aligned", 16, d) for d in (3, 4, 8)] + \
           [("async_noskip", 16, d) for d in (4, 8, 12)] + \
           [("sync_pause", 16, d) for d in (4, 12, 20)]
CASES_LAG = [("v1", 10, 12), ("sync_aligned", 16, 12), ("v1", 16, 14), ("v1", 16, 20),
             ("async_noskip", 16, 20), ("v1", 16, 26)]


@pytest.mark.parametrize("arm,s,d", CASES_OK)
def test_vl_equals_rt_at_fixed_latency(tmp_path, arm, s, d):
    rt, vl = _run(tmp_path, "rt", arm, s, d), _run(tmp_path, "vl", arm, s, d)
    assert [x[1] for x in rt.sent] == [x[1] for x in vl.sent]
    assert np.array_equal(rt.actions, vl.actions)


@pytest.mark.parametrize("arm,s,d", CASES_LAG)
def test_vl_equals_rt_when_arrival_is_a_trigger_tick(tmp_path, arm, s, d):
    assert d >= _s_eff_at(arm, s, d) or d >= H - 4
    rt, vl = _run(tmp_path, "rt", arm, s, d), _run(tmp_path, "vl", arm, s, d)
    k_rt, k_vl = [x[1] for x in rt.sent], [x[1] for x in vl.sent]
    first = next((i for i, (a, b) in enumerate(zip(k_rt, k_vl)) if a != b), None)
    assert first is None and len(k_rt) == len(k_vl), (
        f"request {first}: RT k_obs {k_rt[first]} vs VL {k_vl[first]}; "
        f"{len(k_rt)} vs {len(k_vl)} requests; starved RT {int(rt.ticks['starved'].sum())} "
        f"vs VL {int(vl.ticks['starved'].sum())}")
    assert np.array_equal(rt.actions, vl.actions)
    assert np.array_equal(rt.ticks["starved"], vl.ticks["starved"])


def test_vl_replan_period_equals_rt(tmp_path):
    """The former defect, quantified: v1 s=10 at d=12 (a DP-S sweep cell). RT sends at every
    arrival tick (period d = 12); VL used to send one tick later (period 13). Now both are 12."""
    rt, vl = _run(tmp_path, "rt", "v1", 10, 12), _run(tmp_path, "vl", "v1", 10, 12)
    k_rt = np.array([x[1] for x in rt.sent])
    k_vl = np.array([x[1] for x in vl.sent])
    assert set(np.diff(k_rt[1:])) == {12} and set(np.diff(k_vl[1:])) == {12}
    assert np.array_equal(k_rt, k_vl)
    # Both execute a fresh chunk from row d at the switch tick (the index law).
    for r in (rt, vl):
        t = r.ticks
        assert set(t["row"][t["splice"] == 1][1:]) == {12}


def test_vl_d0_s20_equals_sync(tmp_path):
    """T-1.2's property (VL d_inj=0, s=20 == sync) holds with 3b on both sides of the trigger."""
    vl = _run(tmp_path, "vl", "v1", 20, 0)
    sy = _run(tmp_path, "sync", "v1", 20, 0)
    assert [x[1] for x in vl.sent] == [x[1] for x in sy.sent] == list(range(0, 300, 20))
    assert np.array_equal(vl.actions, sy.actions)


# --- loop-level control invariants that hold (regression guards) -----------------------------


@pytest.mark.parametrize("mode,d", [("rt", 4), ("rt", 1), ("vl", 0), ("vl", 4), ("sync", 0)])
def test_request_carries_o_k_and_is_sent_at_k_obs(tmp_path, mode, d):
    """D6/D7: each request is sent at tick k_obs (never earlier: the env had stepped exactly k_obs
    times) and carries o_k: its state is q after step k_obs - 1 (FakeEnv: the arms commanded at
    k_obs - 1)."""
    r = _run(tmp_path, mode, "v1", 16 if mode != "sync" else 20, d)
    assert all(k_env == k_obs for k_env, k_obs, _ in r.sent)
    for q in r.reqs:
        k = int(q["k_obs"])
        want = np.zeros(14, np.float32) if k == 0 else r.actions[k - 1][ARMS]
        got = np.concatenate([q["state43_sent"][15:22], q["state43_sent"][22:29]])
        assert np.array_equal(got, want), k
        if mode == "rt" and k > 0 and q["outcome"] == Outcome.APPLIED:
            assert q["d"] == d >= 1


def test_preroll_d_excluded_from_d_hat_and_logged(tmp_path):
    lp = make_loop(tmp_path, "rt", d_init=7, max_ticks=120, d_of=lambda i: 0 if i == 0 else 3)
    lp.st.d_hist[:] = []
    r = run_episode(lp)
    pre = r.reqs[r.reqs["k_obs"] == 0]
    assert len(pre) == 1 and pre["d"][0] == 0 and pre["d_hat"][0] == 7
    assert 0 not in lp.st.d_hist and set(lp.st.d_hist) == {3}


def test_timeout_starve_ramp_and_recovery(tmp_path):
    """Request 2 (k_obs 32) times out at 32 + 26 = 58: chunk 1 (k_obs 16) runs out at row 30
    (k = 46), nav ramps to 0 in 5 ticks, the retrigger at 58 lands at 62 on row 4."""
    d_of = lambda i: "timeout" if i == 2 else 4  # noqa: E731
    lp = make_loop(tmp_path, "rt", max_ticks=120, d_of=d_of)
    r = run_episode(lp)
    t = r.ticks
    starved = np.flatnonzero(t["starved"])
    assert starved[0] == 46 and starved[-1] == 61 and len(starved) == 16
    assert t["row"][46] == 30
    nav0 = t["cmd35"][45][NAV]
    for n in range(1, 6):
        np.testing.assert_allclose(t["cmd35"][45 + n][NAV], nav0 * (1 - n / 5), rtol=1e-6, atol=1e-7)
    assert np.all(t["cmd35"][46:62][:, ARMS] == t["cmd35"][45][ARMS])  # arms held
    req = r.reqs[r.reqs["k_obs"] == 32][0]
    assert req["outcome"] == Outcome.TIMEOUT
    assert [x[1] for x in r.sent][:4] == [0, 16, 32, 58]
    assert t["splice"][62] == 1 and t["row"][62] == 4 and not lp.st.starve_fault


@pytest.mark.parametrize("end", ["success", "fall", "safety", "trunc"])
def test_episode_end_tick_accounting(tmp_path, end):
    env = FakeEnv(success_at=37 if end == "success" else None,
                  fall_at=52 if end == "fall" else None,
                  safety_at=61 if end == "safety" else None)
    lp = make_loop(tmp_path, "rt", max_ticks=100, env=env)
    r = run_episode(lp)
    want = {"success": ("SUCCESS", 37, 37), "fall": ("FALL", 52, None),
            "safety": ("SAFETY_FAULT", 61, None), "trunc": ("TRUNC", 99, None)}[end]
    assert (r.end.name, r.k_end, r.fst) == want
    assert len(r.ticks) == r.k_end + 1 and r.ticks["k"][-1] == r.k_end
    assert r.ticks["safety"][-1] == (end == "safety")


def test_drain_and_gen_across_episodes(tmp_path):
    """A request in flight at the episode end is drained (outcome DRAINED), the socket is free for
    the next pre-roll, and the next episode's requests carry gen + 1."""
    lp = make_loop(tmp_path, "rt", max_ticks=34, d_of=lambda i: 4)  # request at 32 still out at 33
    r1 = run_episode(lp, seed=5, ep=0)
    assert r1.drain == "ok"
    drained = r1.reqs[r1.reqs["outcome"] == Outcome.DRAINED]
    assert list(drained["k_obs"]) == [32]
    r2 = run_episode(lp, seed=6, ep=1)
    e2 = r2.reqs[r2.reqs["ep"] == 1]
    assert set(e2["gen"]) == {2} and set(r1.reqs[r1.reqs["ep"] == 0]["gen"]) == {1}
    assert (e2["outcome"] != Outcome.GEN_DROP).all()
