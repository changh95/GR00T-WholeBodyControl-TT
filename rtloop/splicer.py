"""Chunk splicer: which A35 command each 50 Hz tick executes, and when to ask the VLA (PLAN.md §3.1-§3.2).

Pure functions over a plain-field ``SplicerState``: no I/O, no clocks, no logging, so the RT and VL
loops (and the M4b C port) make identical decisions from the same tick stamps. Per tick k the loop
calls ``on_arrival`` / ``on_failure`` (step 2; step 3b in VL), ``trigger`` + ``on_send`` (step 3) and
``select`` (step 4), in that order.

Index law: row = k - chunk.k_base. ``k_base = k_obs`` for v1 and sync_aligned (row i targets tick
k_obs + i: latency is skipped, never re-anchored); ``k_base = k_arr`` for sync_pause, sync_hold and
async_noskip (the chunk restarts at row 0 on arrival). C1 sync is a separate loop mode, not an arm.
"""

from dataclasses import dataclass, field

import numpy as np

from rtloop.schema import ARMS, BLEND, H, NAV, WAIST, Chunk, Outcome

ARM_NAMES = ("v1", "sync_pause", "sync_hold", "async_noskip", "sync_aligned")
_SYNC = ("sync_pause", "sync_hold")  # rows [0, sync_rows) from arrival, then pause until the next chunk
_FROM_ARRIVAL = ("sync_pause", "sync_hold", "async_noskip")  # k_base = k_arr
# Outcomes after which the next trigger() fires at once (nothing useful is coming).
_RETRIGGER = (Outcome.LATE_REJECT, Outcome.TIMEOUT, Outcome.ERROR, Outcome.GEN_DROP)
_JUMP = np.r_[ARMS, WAIST]  # jump_rad is measured on the joint targets, not on nav
D_HIST_LEN = 10
K_NEVER = -(10**9)


@dataclass
class SplicerConfig:
    arm: str = "v1"  # "v1" | "sync_pause" (C2) | "sync_hold" (C2b) | "async_noskip" (C3) | "sync_aligned" (C6)
    s: int = 16  # trigger period, ticks (v1, C3)
    B: int = 0  # crossfade ticks on a switch (0 = hard switch)
    margin: int = 2  # s_eff = max(1, min(s, H - d_hat - margin))
    late_reject_rows: int = 4  # reject a chunk if usable rows H - d <= this (d >= 26)
    starve_ramp: int = 5  # ticks to ramp nav to 0 when starved (or paused, C2)
    starve_fault: int = 50  # consecutive starved ticks -> STARVE_FAULT flag (episode continues)
    d_init: int = 4  # d_hat before any measurement (4 on the 5090, 3 on TT; VL: pass d_inj)
    sync_rows: int = 20  # rows a synchronous arm executes per chunk (C2/C2b/C6)

    def __post_init__(self):
        if self.arm not in ARM_NAMES:
            raise ValueError(f"unknown splicer arm {self.arm!r}; expected one of {ARM_NAMES}")
        if not 1 <= self.sync_rows <= H:
            raise ValueError(f"sync_rows must be in [1, {H}], got {self.sync_rows}")
        if self.s < 1 or self.B < 0 or self.starve_ramp < 0:
            raise ValueError(f"bad splicer config {self}")


@dataclass
class SplicerState:
    """Plain fields only, so the M4b C port can mirror it as a struct."""

    act: Chunk | None = None  # chunk being executed
    pending: Chunk | None = None  # arrived, switches in at pending.k_arr
    in_flight: bool = False  # a request has been sent and is not yet registered by on_arrival
    k_req: int = K_NEVER  # tick of the last sent request
    last_outcome: Outcome = Outcome.PENDING
    blend_t0: int = K_NEVER
    last_row: np.ndarray | None = None  # last executed A35 command (chunk row, hold, starve or pause)
    last_row_at_switch: np.ndarray | None = None  # last_row frozen on the switch tick (blend source)
    starve_n: int = 0
    starve_fault: bool = False
    d_hist: list[int] = field(default_factory=list)  # last D_HIST_LEN measured d
    gen: int = 0


def new_state(gen: int = 0) -> SplicerState:
    return SplicerState(gen=gen)


def d_hat(st: SplicerState, cfg: SplicerConfig) -> int:
    """p90 of the last 10 measured d (the 9th order statistic of 10), so one outlier is ignored.

    Generalised to n < 10 values as sorted(d)[ceil(0.9 n) - 1], in integer arithmetic.
    """
    n = len(st.d_hist)
    if n == 0:
        return cfg.d_init
    return sorted(st.d_hist)[(9 * n + 9) // 10 - 1]


def d_hat_max(st: SplicerState, cfg: SplicerConfig) -> int:
    """Max of the last 10 d (logged next to d_hat, RequestLog.d_hat_max)."""
    return max(st.d_hist) if st.d_hist else cfg.d_init


def s_eff(st: SplicerState, cfg: SplicerConfig) -> int:
    """Effective trigger period. The sync arms trigger on rows used up, so theirs is sync_rows (logged)."""
    if cfg.arm in _SYNC:
        return cfg.sync_rows
    dh = d_hat(st, cfg)
    if cfg.arm == "sync_aligned":
        return max(1, cfg.sync_rows - dh)  # the new chunk lands near row sync_rows, as sync switches
    return max(1, min(cfg.s, H - dh - cfg.margin))


def trigger(k: int, st: SplicerState, cfg: SplicerConfig) -> bool:
    """True if a VLA request should be sent at tick k (PLAN.md §3.1 "Trigger").

    Never while a request is in flight (VL: not yet visible). A chunk registered this tick but not yet
    switched in (``pending``) counts as the active one.
    """
    if st.in_flight:
        return False
    cur = st.pending if st.pending is not None else st.act
    if cur is None or st.last_outcome in _RETRIGGER:
        return True
    if cfg.arm in _SYNC:
        return k - cur.k_base >= cfg.sync_rows
    return k - st.k_req >= s_eff(st, cfg)


def on_send(k: int, st: SplicerState) -> None:
    st.k_req = k
    st.in_flight = True
    st.last_outcome = Outcome.PENDING


def on_arrival(k: int, chunk: Chunk, st: SplicerState, cfg: SplicerConfig, measure: bool = True) -> Outcome:
    """Register a reply at tick k (RT: the poll tick; VL: k_obs + d_inj).

    ``measure=False`` keeps d out of d_hist; the loop passes it for the pre-roll chunk, whose d = 0 is
    an artefact of the blocking first call (PLAN.md §1.2 lifecycle step 5). The late-reject rule is the
    same for every arm: a reply that old is past the 500 ms give-up anyway.
    """
    chunk.k_arr = k
    st.in_flight = False
    if chunk.gen != st.gen:
        st.last_outcome = Outcome.GEN_DROP
        return Outcome.GEN_DROP
    d = k - chunk.k_obs
    if measure:
        st.d_hist.append(d)
        if len(st.d_hist) > D_HIST_LEN:
            del st.d_hist[:-D_HIST_LEN]
    if d >= H - cfg.late_reject_rows:
        st.last_outcome = Outcome.LATE_REJECT
        return Outcome.LATE_REJECT
    chunk.k_base = k if cfg.arm in _FROM_ARRIVAL else chunk.k_obs
    st.pending = chunk
    st.last_outcome = Outcome.APPLIED
    return Outcome.APPLIED


def on_failure(k: int, st: SplicerState, outcome: Outcome) -> None:
    """Timeout, error or drain: the socket is free again and the next trigger() fires at once."""
    st.in_flight = False
    st.last_outcome = Outcome(outcome)


def visible(k: int, k_obs: int, d_inj: int) -> bool:
    """VL step 3b: the in-flight chunk becomes visible (is registered) exactly at k_obs + d_inj."""
    return k >= k_obs + d_inj


def _info(row, age, rows_left, starved, paused, splice, jump_rad, chunk_seq) -> dict:
    return {"row": row, "age": age, "rows_left": rows_left, "starved": starved, "paused": paused,
            "splice": splice, "jump_rad": jump_rad, "chunk_seq": chunk_seq}


def select(k: int, st: SplicerState, cfg: SplicerConfig, hold: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """The A35 command for tick k and its TickLog fields (PLAN.md §3.1 reference ``select``).

    ``hold`` (hold_row of this tick's obs) is used only while no chunk is active (pre-roll fallback).
    Past the last usable row (row >= H; for sync_pause/sync_hold row >= sync_rows) the last executed
    arm, hand, waist and height targets are held and nav ramps linearly to 0 over ``starve_ramp``
    ticks (sync_hold keeps nav). For v1, async_noskip and sync_aligned that is starvation (counted,
    STARVE_FAULT at ``starve_fault`` consecutive ticks); for the sync arms it is the designed pause
    (``paused`` = 1, never a fault). info["row"] is k - k_base even then (no row is executed);
    chunk_seq is 0 while row == -1.
    """
    splice = 0
    p = st.pending
    if p is not None and p.k_arr <= k:
        st.act, st.pending, st.blend_t0 = p, None, k
        st.last_row_at_switch = st.last_row  # what was actually executed just before the switch
        splice = 1
    act = st.act
    if act is None:
        cmd = np.array(hold, dtype=np.float32)
        st.last_row = cmd
        return cmd.copy(), _info(-1, -1, -1, 0, 0, 0, 0.0, 0)

    i = k - act.k_base
    if i < 0:
        raise ValueError(f"tick {k} precedes chunk {act.seq}'s k_base {act.k_base}")
    sync = cfg.arm in _SYNC
    limit = cfg.sync_rows if sync else H
    prev = st.last_row  # on a switch tick this is last_row_at_switch

    if i >= limit:
        base = st.last_row if st.last_row is not None else act.rows[limit - 1]
        cmd = base.copy()
        n = i - limit + 1  # 1 on the first starved / paused tick
        if cfg.arm != "sync_hold":
            r = cfg.starve_ramp
            # nav_n = nav_0 * (1 - n/r), written as a step from nav_{n-1} so no extra state is kept
            cmd[NAV] = 0.0 if n >= r else base[NAV] * np.float32((r - n) / (r - n + 1))
        if sync:
            starved, paused = 0, 1
        else:
            starved, paused = 1, 0
            st.starve_n += 1
            if st.starve_n >= cfg.starve_fault:
                st.starve_fault = True
    else:
        starved = paused = 0
        st.starve_n = 0
        cmd = act.rows[i].copy()  # never mutate a stored chunk
        if cfg.B and st.last_row_at_switch is not None:
            j = k - st.blend_t0
            if j < cfg.B:
                w = np.float32((j + 1) / (cfg.B + 1))
                cmd[BLEND] = (1 - w) * st.last_row_at_switch[BLEND] + w * cmd[BLEND]

    jump = float(np.linalg.norm(cmd[_JUMP] - prev[_JUMP])) if splice and prev is not None else 0.0
    st.last_row = cmd
    info = _info(i, k - act.k_obs, max(0, limit - 1 - i), starved, paused, splice, jump, act.seq)
    return cmd.copy(), info


def reset(st: SplicerState, gen: int) -> None:
    """New episode, instruction change or VLA swap: drop the chunks and bump gen.

    ``in_flight`` and ``k_req`` are kept because they mirror the socket: a request still out is
    gen-dropped when it arrives. ``d_hist`` describes the server, not the episode, so it is kept
    too; clear it on a VLA swap.
    """
    st.act = None
    st.pending = None
    st.gen = gen
    st.blend_t0 = K_NEVER
    st.last_row = None
    st.last_row_at_switch = None
    st.starve_n = 0
    st.starve_fault = False
