"""T-1.1 splicer table (PLAN.md §2 M1, §3.1-§3.2): pure-function tests, no sim, no sockets, no clocks.

Chunk rows identify themselves: every non-nav column of row i of chunk ``seq`` is 100*seq + i, so a
command that is not blended or held can be traced back to exactly one stored row.
"""

import math
import random

import numpy as np
import pytest

from rtloop import splicer as sp
from rtloop.schema import A35, ARMS, BLEND, H, HANDS, HEIGHT, NAV, WAIST, Chunk, Outcome

NAV0 = np.array([0.30, -0.20, 0.50], np.float32)
HOLD = np.full(A35, -7.0, np.float32)
HOLD[NAV] = 0.0
HOLD[HEIGHT] = 0.74
JOINTS = np.r_[ARMS, WAIST]
INFO_KEYS = {"row", "age", "rows_left", "starved", "paused", "splice", "jump_rad", "chunk_seq"}


def rows_for(seq: int) -> np.ndarray:
    rows = np.empty((H, A35), np.float32)
    rows[:, : NAV[0]] = (100 * seq + np.arange(H, dtype=np.float32))[:, None]
    rows[:, NAV] = NAV0 * np.float32(1.0 + 0.1 * (seq % 7)) + np.float32(0.001) * np.arange(H, dtype=np.float32)[:, None]
    return rows


def mk(seq: int, k_obs: int, gen: int = 0) -> Chunk:
    return Chunk(seq=seq, gen=gen, k_obs=k_obs, rows=rows_for(seq))


def preroll(st, cfg, seq: int = 0) -> Chunk:
    """Blocking first call at k = 0 (d = 0, kept out of the d statistics), as the loop does it."""
    sp.on_send(0, st)
    c = mk(seq, 0, st.gen)
    assert sp.on_arrival(0, c, st, cfg, measure=False) == Outcome.APPLIED
    return c


def drive(cfg, d: int, n_ticks: int, with_preroll: bool = False, vl: bool = False):
    """Discrete-event loop: every request's reply arrives exactly d ticks after it was sent.

    RT order per tick: arrival, trigger (+ send), select. VL: trigger (+ send), visibility at
    k_obs + d, select, so d = 0 executes row 0 on the tick of the send.
    """
    st = sp.new_state()
    chunks, ticks, outcomes = {}, [], []
    flight = None
    seq = 0
    if with_preroll:
        chunks[0] = preroll(st, cfg)
        seq = 1

    def arrive(k):
        nonlocal flight
        if flight is not None and flight[1] <= k:
            outcomes.append((flight[0].seq, sp.on_arrival(k, flight[0], st, cfg)))
            flight = None

    for k in range(n_ticks):
        if not vl:
            arrive(k)
        if sp.trigger(k, st, cfg):
            assert flight is None and not st.in_flight, "trigger fired with a request in flight"
            sp.on_send(k, st)
            c = mk(seq, k, st.gen)
            chunks[seq] = c
            flight = (c, k + d)
            seq += 1
        if vl:
            arrive(k)
        n_in_flight = int(st.in_flight)
        cmd, info = sp.select(k, st, cfg, HOLD)
        ticks.append(dict(info, k=k, in_flight=n_in_flight, cmd=cmd, fault=st.starve_fault))
    return st, chunks, ticks, outcomes


def executed_rows(ticks) -> dict:
    """seq -> rows actually executed (not hold, not starved, not paused)."""
    out = {}
    for t in ticks:
        if t["row"] >= 0 and not t["starved"] and not t["paused"]:
            out.setdefault(t["chunk_seq"], []).append(t["row"])
    return out


# --- d_hat, s_eff ---------------------------------------------------------------------------------


def test_d_hat_init_and_p90_ignores_one_outlier():
    cfg = sp.SplicerConfig(d_init=4)
    st = sp.new_state()
    assert sp.d_hat(st, cfg) == 4 and sp.d_hat_max(st, cfg) == 4
    st.d_hist = [4] * 9 + [19]
    assert sp.d_hat(st, cfg) == 4 and sp.d_hat_max(st, cfg) == 19
    st.d_hist = [19] + [4] * 9
    assert sp.d_hat(st, cfg) == 4
    st.d_hist = [4] * 8 + [19, 20]  # two outliers: the 9th order statistic sees one
    assert sp.d_hat(st, cfg) == 19
    assert sp.d_hat(st, sp.SplicerConfig(d_init=3)) == 19
    assert sp.d_hat(sp.new_state(), sp.SplicerConfig(d_init=3)) == 3


def test_d_hat_generalised_order_statistic():
    rng = random.Random(0)
    cfg = sp.SplicerConfig()
    for _ in range(500):
        n = rng.randint(1, 10)
        st = sp.new_state()
        st.d_hist = [rng.randint(0, 28) for _ in range(n)]
        assert sp.d_hat(st, cfg) == sorted(st.d_hist)[math.ceil(0.9 * n) - 1]


def test_d_hist_keeps_last_10_measured():
    cfg = sp.SplicerConfig()
    st = sp.new_state()
    for i, d in enumerate([20] * 5 + [3] * 10):
        sp.on_send(100 * i, st)
        sp.on_arrival(100 * i + d, mk(i, 100 * i), st, cfg)
    assert st.d_hist == [3] * 10 and sp.d_hat(st, cfg) == 3
    # a late-rejected reply is still a measured d; a gen drop and the pre-roll are not
    sp.on_send(5000, st)
    assert sp.on_arrival(5027, mk(99, 5000), st, cfg) == Outcome.LATE_REJECT
    assert st.d_hist[-1] == 27 and len(st.d_hist) == 10
    sp.on_send(6000, st)
    assert sp.on_arrival(6005, mk(100, 6000, gen=7), st, cfg) == Outcome.GEN_DROP
    assert st.d_hist[-1] == 27
    st2 = sp.new_state()
    preroll(st2, cfg)
    assert st2.d_hist == []


@pytest.mark.parametrize("s", [10, 16, 20])
def test_s_eff_never_below_one(s):
    for arm in ("v1", "async_noskip"):
        cfg = sp.SplicerConfig(arm=arm, s=s)
        for dh in range(0, 41):
            st = sp.new_state()
            st.d_hist = [dh]
            se = sp.s_eff(st, cfg)
            assert se == max(1, min(s, H - dh - cfg.margin)) and se >= 1
    cfg = sp.SplicerConfig(s=16)
    st = sp.new_state()
    for dh, want in [(0, 16), (4, 16), (12, 16), (13, 15), (14, 14), (27, 1), (28, 1)]:
        st.d_hist = [dh]
        assert sp.s_eff(st, cfg) == want, dh


def test_s_eff_sync_aligned_and_sync_arms():
    cfg = sp.SplicerConfig(arm="sync_aligned", sync_rows=20)
    st = sp.new_state()
    for dh in range(0, 28):
        st.d_hist = [dh]
        assert sp.s_eff(st, cfg) == max(1, 20 - dh) >= 1
    for arm in ("sync_pause", "sync_hold"):
        assert sp.s_eff(st, sp.SplicerConfig(arm=arm)) == 20


def test_config_validation():
    with pytest.raises(ValueError):
        sp.SplicerConfig(arm="rtc")
    with pytest.raises(ValueError):
        sp.SplicerConfig(sync_rows=31)


# --- trigger / on_send / on_failure / visible ----------------------------------------------------


def test_trigger_rules():
    cfg = sp.SplicerConfig(s=16)
    st = sp.new_state()
    assert sp.trigger(0, st, cfg)  # no chunk
    sp.on_send(0, st)
    assert st.in_flight and st.k_req == 0 and st.last_outcome == Outcome.PENDING
    assert not any(sp.trigger(k, st, cfg) for k in range(0, 200))  # never while in flight
    sp.on_arrival(4, mk(0, 0), st, cfg)
    assert st.pending is not None and not st.in_flight
    assert not sp.trigger(4, st, cfg)  # the pending chunk counts as active
    sp.select(4, st, cfg, HOLD)
    assert [sp.trigger(k, st, cfg) for k in (15, 16)] == [False, True]
    for bad in (Outcome.TIMEOUT, Outcome.ERROR):
        sp.on_send(16, st)
        sp.on_failure(17, st, bad)
        assert not st.in_flight and st.last_outcome == bad
        assert sp.trigger(17, st, cfg)  # retrigger at once, even 1 tick after the send


def test_visible():
    assert sp.visible(10, 10, 0) and not sp.visible(12, 10, 3) and sp.visible(13, 10, 3)


# --- the splicer table: one request, d in [0, 28], s in {10, 16, 20}, B in {0, 3, 5} ------------


@pytest.mark.parametrize("B", [0, 3, 5])
@pytest.mark.parametrize("s", [10, 16, 20])
@pytest.mark.parametrize("d", range(0, 29))
def test_table_single_request(d, s, B):
    cfg = sp.SplicerConfig(s=s, B=B)
    st = sp.new_state()
    a = preroll(st, cfg)
    prev = None
    for k in range(s):
        assert not sp.trigger(k, st, cfg)
        cmd, info = sp.select(k, st, cfg, HOLD)
        assert set(info) == INFO_KEYS
        assert info["row"] == k and info["chunk_seq"] == 0 and info["splice"] == (k == 0)
        np.testing.assert_array_equal(cmd, a.rows[k])
        prev = cmd
    assert sp.trigger(s, st, cfg)  # d_hat = d_init = 4 -> s_eff = s for s <= 24
    sp.on_send(s, st)
    b = mk(1, s)
    k = s
    while True:
        if k == s + d:
            out = sp.on_arrival(k, b, st, cfg)
            break
        cmd, info = sp.select(k, st, cfg, HOLD)
        assert info["chunk_seq"] == 0 and info["row"] == k
        assert info["starved"] == (k >= H)
        prev = cmd
        k += 1
    k_arr = s + d
    if d >= H - cfg.late_reject_rows:  # d >= 26
        assert out == Outcome.LATE_REJECT and st.pending is None
        assert sp.trigger(k_arr, st, cfg)  # retrigger at once
        cmd, info = sp.select(k_arr, st, cfg, HOLD)
        assert info["chunk_seq"] == 0 and info["starved"] == 1 and info["splice"] == 0
        return
    assert out == Outcome.APPLIED and b.k_arr == k_arr and b.k_base == s
    se = max(1, min(s, H - d - cfg.margin))
    assert sp.s_eff(st, cfg) == se
    assert sp.trigger(k_arr, st, cfg) == (d >= se)
    frozen = prev.copy()
    for j in range(0, B + 2):
        k = k_arr + j
        if d + j >= H:
            break
        cmd, info = sp.select(k, st, cfg, HOLD)
        assert info["row"] == d + j and info["age"] == d + j and info["chunk_seq"] == 1
        assert info["starved"] == 0 and info["splice"] == (j == 0)
        new = b.rows[d + j]
        np.testing.assert_array_equal(cmd[HANDS], new[HANDS])  # hands and height switch hard
        assert cmd[HEIGHT] == new[HEIGHT]
        if j < B:
            w = np.float32((j + 1) / (B + 1))
            want = (1 - w) * frozen[BLEND] + w * new[BLEND]
            np.testing.assert_allclose(cmd[BLEND], want, rtol=1e-6, atol=1e-5)
        else:
            np.testing.assert_array_equal(cmd, new)
        if j == 0:
            assert info["jump_rad"] == pytest.approx(float(np.linalg.norm(cmd[JOINTS] - frozen[JOINTS])), rel=1e-6)
        else:
            assert info["jump_rad"] == 0.0
    np.testing.assert_array_equal(st.last_row_at_switch, frozen)


# --- starvation, STARVE_FAULT, recovery, blend after starvation ----------------------------------


def _starve_then_recover(B: int):
    """Stall: request at 16 times out at 41, the retry times out at 66, the third (sent at 66)
    arrives at 82 with d = 16. Chunk 0 runs out at k = 30, so ticks 30..81 starve (52 ticks)."""
    cfg = sp.SplicerConfig(s=16, B=B)
    st = sp.new_state()
    a = preroll(st, cfg)
    ticks = {}
    b = None
    for k in range(0, 120):
        if k in (41, 66):
            sp.on_failure(k, st, Outcome.TIMEOUT)
        if k == 82:
            assert sp.on_arrival(k, b, st, cfg) == Outcome.APPLIED
        if sp.trigger(k, st, cfg):
            assert k in (16, 41, 66, 82)  # at 82: d_hat = 16 -> s_eff = 12 <= k - k_req
            sp.on_send(k, st)
            if k == 66:
                b = mk(1, 66)
        cmd, info = sp.select(k, st, cfg, HOLD)
        ticks[k] = (cmd, info, st.starve_n, st.starve_fault)
    return cfg, a, b, ticks


def test_starvation_hold_nav_ramp_fault_and_recovery():
    cfg, a, b, ticks = _starve_then_recover(B=0)
    last = a.rows[H - 1]
    assert ticks[29][1]["starved"] == 0 and ticks[29][1]["rows_left"] == 0
    for k in range(30, 82):
        cmd, info, starve_n, fault = ticks[k]
        n = k - 29
        assert info["starved"] == 1 and info["paused"] == 0 and info["row"] == k and info["chunk_seq"] == 0
        assert starve_n == n
        assert fault == (n >= cfg.starve_fault)  # STARVE_FAULT on the 50th consecutive starved tick
        held = np.setdiff1d(np.arange(A35), NAV)
        np.testing.assert_array_equal(cmd[held], last[held])  # arms, hands, waist, height held
        np.testing.assert_allclose(cmd[NAV], last[NAV] * max(0.0, 1.0 - n / cfg.starve_ramp), rtol=1e-6, atol=1e-7)
        if n >= cfg.starve_ramp:
            assert np.linalg.norm(cmd[NAV]) == 0.0  # Balance within 5 ticks
    assert ticks[78][3] is False and ticks[79][3] is True
    cmd, info, starve_n, fault = ticks[82]  # recovery on the next arrival: row k - k_obs
    assert info["starved"] == 0 and info["row"] == 16 and info["chunk_seq"] == 1 and info["splice"] == 1
    assert starve_n == 0 and fault is True  # the flag is per episode (latched until reset)
    np.testing.assert_array_equal(cmd, b.rows[16])
    for k in range(83, 96):
        assert ticks[k][1]["row"] == k - 66 and ticks[k][1]["starved"] == 0


def test_starve_then_arrival_blend_B5_starts_from_zero_nav():
    cfg, a, b, ticks = _starve_then_recover(B=5)
    held = ticks[81][0]
    assert np.all(held[NAV] == 0.0)
    for j in range(0, 7):
        cmd, info, _, _ = ticks[82 + j]
        new = b.rows[16 + j]
        assert info["row"] == 16 + j
        if j < 5:
            w = np.float32((j + 1) / 6)
            np.testing.assert_allclose(cmd[NAV], (1 - w) * np.zeros(3, np.float32) + w * new[NAV], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(cmd[ARMS], (1 - w) * a.rows[H - 1][ARMS] + w * new[ARMS], rtol=1e-6)
            np.testing.assert_array_equal(cmd[HANDS], new[HANDS])
        else:
            np.testing.assert_array_equal(cmd, new)


# --- gen drop and reset ------------------------------------------------------------------------


def test_gen_drop_after_reset_and_while_active():
    cfg = sp.SplicerConfig()
    st = sp.new_state(gen=0)
    preroll(st, cfg)
    for k in range(17):
        if sp.trigger(k, st, cfg):
            sp.on_send(k, st)
        sp.select(k, st, cfg, HOLD)
    old = mk(1, 16, gen=0)
    sp.reset(st, 1)  # episode end with a request still out
    assert st.act is None and st.pending is None and st.gen == 1 and st.in_flight and st.k_req == 16
    assert st.last_row is None and st.starve_n == 0 and not st.starve_fault
    assert not sp.trigger(17, st, cfg)  # the socket is still busy
    d_before = list(st.d_hist)
    assert sp.on_arrival(20, old, st, cfg) == Outcome.GEN_DROP
    assert st.pending is None and st.act is None and not st.in_flight and st.d_hist == d_before
    assert sp.trigger(20, st, cfg)
    cmd, info = sp.select(20, st, cfg, HOLD)
    assert info["row"] == -1 and info["chunk_seq"] == 0
    np.testing.assert_array_equal(cmd, HOLD)
    # defensive: a stale-gen reply never disturbs the active chunk and the next trigger fires at once
    sp.on_send(20, st)
    sp.on_arrival(24, mk(2, 20, gen=1), st, cfg)
    sp.select(24, st, cfg, HOLD)
    sp.on_send(25, st)
    st.gen = 2
    act = st.act
    assert sp.on_arrival(27, mk(3, 25, gen=1), st, cfg) == Outcome.GEN_DROP
    assert st.act is act and st.pending is None and sp.trigger(27, st, cfg)


def test_hold_before_first_chunk():
    cfg = sp.SplicerConfig()
    st = sp.new_state()
    hold = HOLD.copy()
    cmd, info = sp.select(0, st, cfg, hold)
    assert info == {"row": -1, "age": -1, "rows_left": -1, "starved": 0, "paused": 0, "splice": 0,
                    "jump_rad": 0.0, "chunk_seq": 0}
    assert cmd.dtype == np.float32 and cmd is not hold and st.last_row is not hold
    np.testing.assert_array_equal(st.last_row, hold)
    cmd[:] = 1.0
    np.testing.assert_array_equal(st.last_row, HOLD)
    np.testing.assert_array_equal(hold, HOLD)


# --- VL: the row executed at k_obs + d_inj is d_inj -------------------------------------------


@pytest.mark.parametrize("d_inj", [0, 1, 3])
def test_vl_row_at_visibility_equals_d_inj(d_inj):
    cfg = sp.SplicerConfig(s=16, d_init=d_inj)
    st, chunks, ticks, outcomes = drive(cfg, d_inj, 600, with_preroll=True, vl=True)
    assert len(chunks) > 30 and all(o == Outcome.APPLIED for _, o in outcomes)
    for seq, c in chunks.items():
        if seq == 0 or c.k_obs + d_inj >= len(ticks):
            continue
        t = ticks[c.k_obs + d_inj]
        assert t["chunk_seq"] == seq and t["row"] == d_inj and t["splice"] == 1


def test_vl_s20_d0_executes_rows_0_to_19_like_sync():
    cfg = sp.SplicerConfig(s=20, d_init=0)
    _, chunks, ticks, _ = drive(cfg, 0, 1440, with_preroll=True, vl=True)
    rows = executed_rows(ticks)
    assert all(rows[q] == list(range(20)) for q in list(rows)[:-1])
    assert all(t["row"] == t["k"] % 20 and t["chunk_seq"] == t["k"] // 20 for t in ticks)


# --- discrete-event driver over 3000 ticks ------------------------------------------------------


@pytest.mark.parametrize("d", range(1, 13))
def test_driver_fixed_latency_rows_within_d_to_s_plus_d(d):
    s = 16
    st, chunks, ticks, outcomes = drive(sp.SplicerConfig(s=s), d, 3000)
    rows = executed_rows(ticks)
    assert all(o == Outcome.APPLIED for _, o in outcomes)
    assert sum(t["starved"] for t in ticks) == 0
    assert max(t["in_flight"] for t in ticks) <= 1
    assert all(d <= r < s + d for rs in rows.values() for r in rs)
    for q in list(rows)[:-1]:  # every complete chunk executes exactly rows [d, s + d)
        assert rows[q] == list(range(d, s + d)), (q, rows[q])
    assert all(t["row"] == -1 for t in ticks[:d])  # hold until the first chunk lands
    assert st.starve_n == 0 and not st.starve_fault


@pytest.mark.parametrize("d", range(13, 26))
def test_driver_starvation_threshold(d):
    """Past d = 12 the margin is used up; the adaptive s_eff keeps rows < H up to d = 15 (period
    max(s_eff, d), rows [d, d + period)), and from d = 16 every cycle starves 2d - 30 ticks."""
    s = 16
    se = max(1, min(s, H - d - 2))
    period = max(se, d)
    st, chunks, ticks, outcomes = drive(sp.SplicerConfig(s=s), d, 3000)
    rows = executed_rows(ticks)
    starved = {}
    for t in ticks:
        if t["starved"]:
            starved[t["chunk_seq"]] = starved.get(t["chunk_seq"], 0) + 1
    for q in list(rows)[:-1]:
        assert rows[q] == list(range(d, min(H, d + period))), (q, rows[q])
        assert starved.get(q, 0) == max(0, d + period - H)
    total = sum(t["starved"] for t in ticks)
    assert (total > 0) == (d >= 16)
    assert not st.starve_fault  # never 50 consecutive ticks


@pytest.mark.parametrize("d", [13, 14, 15, 16])
def test_driver_preroll_transient(d):
    """Pre-roll chunk 0 runs on d_init = 4 (s_eff = 16), so it starves max(0, 16 + d - 30) ticks."""
    _, _, ticks, _ = drive(sp.SplicerConfig(s=16), d, 400, with_preroll=True)
    n0 = sum(t["starved"] for t in ticks if t["chunk_seq"] == 0)
    assert n0 == max(0, 16 + d - H)
    assert executed_rows(ticks)[0] == list(range(0, min(H, 16 + d)))


def test_driver_all_late_rejected_sets_starve_fault():
    st, _, ticks, outcomes = drive(sp.SplicerConfig(s=16), 26, 300, with_preroll=True)
    assert outcomes and all(o == Outcome.LATE_REJECT for _, o in outcomes)
    assert [t["k"] for t in ticks if t["starved"]][0] == 30
    assert [t["k"] for t in ticks if t["fault"]][0] == 30 + 50 - 1  # the 50th starved tick
    assert st.starve_fault and st.starve_n == 300 - 30 and all(t["fault"] for t in ticks[79:])


# --- arms -----------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 4, 8])
def test_arm_sync_pause(d):
    cfg = sp.SplicerConfig(arm="sync_pause", sync_rows=20)
    st, chunks, ticks, outcomes = drive(cfg, d, 1000, with_preroll=True)
    rows = executed_rows(ticks)
    assert all(rows[q] == list(range(20)) for q in list(rows)[:-1])
    assert all(t["starved"] == 0 for t in ticks) and not st.starve_fault and st.starve_n == 0
    for q, c in chunks.items():
        if q == 0 or c.k_arr < 0 or c.k_arr >= len(ticks):
            continue
        assert c.k_base == c.k_arr and c.k_obs == chunks[q - 1].k_base + 20
        t = ticks[c.k_arr]
        assert t["row"] == 0 and t["age"] == d and t["splice"] == 1
        prev_last = chunks[q - 1].rows[19]
        for n in range(1, d + 1):  # the pause: hold, nav ramps to 0 over 5 ticks
            p = ticks[c.k_obs + n - 1]
            assert p["paused"] == 1 and p["starved"] == 0 and p["chunk_seq"] == q - 1
            np.testing.assert_array_equal(p["cmd"][: NAV[0]], prev_last[: NAV[0]])
            np.testing.assert_allclose(p["cmd"][NAV], prev_last[NAV] * max(0.0, 1 - n / 5), rtol=1e-6, atol=1e-7)


def test_arm_sync_pause_long_wait_is_never_a_fault():
    cfg = sp.SplicerConfig(arm="sync_pause")
    st = sp.new_state()
    preroll(st, cfg)
    for k in range(300):
        if sp.trigger(k, st, cfg):
            assert k == 20
            sp.on_send(k, st)  # never answered
        cmd, info = sp.select(k, st, cfg, HOLD)
        if k >= 20:
            assert info["paused"] == 1 and info["starved"] == 0
        if k >= 24:
            assert np.all(cmd[NAV] == 0.0)
    assert st.starve_n == 0 and not st.starve_fault


def test_arm_sync_hold_keeps_nav():
    cfg = sp.SplicerConfig(arm="sync_hold")
    st, chunks, ticks, _ = drive(cfg, 6, 500, with_preroll=True)
    for q, c in chunks.items():
        if q == 0 or c.k_arr < 0 or c.k_arr >= len(ticks):
            continue
        prev_last = chunks[q - 1].rows[19]
        for k in range(c.k_obs, c.k_arr):
            assert ticks[k]["paused"] == 1
            np.testing.assert_array_equal(ticks[k]["cmd"], prev_last)  # nav held, walks blind
        assert ticks[c.k_arr]["row"] == 0
    assert not st.starve_fault


@pytest.mark.parametrize("d", [4, 8, 12])
def test_arm_async_noskip_starts_at_row_0(d):
    cfg = sp.SplicerConfig(arm="async_noskip", s=16)
    _, chunks, ticks, _ = drive(cfg, d, 1000)
    rows = executed_rows(ticks)
    assert all(rows[q] == list(range(16)) for q in list(rows)[:-1])
    for q, c in chunks.items():
        if c.k_arr < 0 or c.k_arr >= len(ticks):
            continue
        assert c.k_base == c.k_arr
        t = ticks[c.k_arr]
        assert t["row"] == 0 and t["age"] == d and t["chunk_seq"] == q
    assert sum(t["starved"] for t in ticks) == 0


@pytest.mark.parametrize("d", [3, 4, 8])
def test_arm_sync_aligned_switches_near_row_20(d):
    cfg = sp.SplicerConfig(arm="sync_aligned", sync_rows=20, d_init=d)
    _, chunks, ticks, _ = drive(cfg, d, 1000, with_preroll=True)
    rows = executed_rows(ticks)
    assert rows[0] == list(range(20))
    assert all(rows[q] == list(range(d, 20)) for q in list(rows)[1:-1])
    assert all(c.k_base == c.k_obs for c in chunks.values() if c.k_arr >= 0)
    assert sum(t["starved"] for t in ticks) == 0


def test_arm_v1_with_preroll_rows():
    _, chunks, ticks, _ = drive(sp.SplicerConfig(s=16), 4, 1000, with_preroll=True)
    rows = executed_rows(ticks)
    assert rows[0] == list(range(0, 20))
    assert all(rows[q] == list(range(4, 20)) for q in list(rows)[1:-1])
    assert all(c.k_base == c.k_obs for c in chunks.values() if c.k_arr >= 0)


# --- immutability --------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", sp.ARM_NAMES)
def test_select_never_mutates_stored_chunks(arm):
    cfg = sp.SplicerConfig(arm=arm, s=16, B=5)
    st, chunks, ticks, _ = drive(cfg, 18, 800, with_preroll=True)  # blends and starvation both occur
    if arm == "v1":
        assert sum(t["starved"] for t in ticks) > 0
    for q, c in chunks.items():
        np.testing.assert_array_equal(c.rows, rows_for(q))
    # the returned command is a copy: editing it changes neither the state nor the chunk
    cmd, _ = sp.select(800, st, cfg, HOLD)
    before = st.last_row.copy()
    cmd += 1.0
    np.testing.assert_array_equal(st.last_row, before)
    np.testing.assert_array_equal(HOLD[NAV], np.zeros(3, np.float32))
