"""A sim-free harness around the real rtloop.loop.Loop tick code (review tests).

Drives ``Loop._ticks_async`` / ``Loop._ticks_sync`` / ``Loop._drain`` with a fake env (a few
deterministic joints, no MuJoCo), a fake VLA client whose replies become pollable exactly ``d``
ticks after their k_obs (RT) or at any time (VL/sync), and a pacer that never waits. Everything
else (splicer, _send, _on_reply, _step_and_log, TickLog, RequestLog) is the production code, so a
divergence here is a divergence of the loop's control decisions, not of the fakes.
"""

from collections import Counter
from types import SimpleNamespace

import numpy as np

from rtloop import loop as L
from rtloop import splicer as sp
from rtloop import telemetry
from rtloop.schema import (A35, ARMS, H, NAV, Q43_GROUP_INDICES, TICK_NS, ChunkStatus,
                           VlaBackend, state43)
from rtloop.vla_client import VlaReply

DELTA = np.float32(1e-3)


class FakeEnv:
    """q43 follows the commanded arms exactly one tick later (like a perfect position servo)."""

    def __init__(self, success_at: int | None = None, fall_at: int | None = None,
                 safety_at: int | None = None):
        self.k = 0
        self.q = np.zeros(43)
        self.success_at, self.fall_at, self.safety_at = success_at, fall_at, safety_at
        self.actions: list[np.ndarray] = []

    def obs(self) -> dict:
        o = {"q": self.q.copy(), "dq": np.zeros(43),
             "floating_base_pose": np.array([0, 0, 0.75, 1, 0, 0, 0], dtype=np.float64),
             "video.ego_view": np.full((4, 4, 3), self.k % 256, dtype=np.uint8),
             "annotation.human.task_description": "pick the apple"}
        for g, idx in Q43_GROUP_INDICES.items():
            o[f"state.{g}"] = self.q[idx].copy()
        if self.fall_at is not None and self.k > self.fall_at:
            o["floating_base_pose"][2] = 0.3
        return o

    def reset(self) -> dict:
        self.k = 0
        self.q[:] = 0.0
        self.actions = []
        return self.obs()

    def step(self, action: dict):
        if self.safety_at is not None and self.k == self.safety_at:
            raise SystemExit(1)
        row = np.concatenate([action[f"action.{n}"] for n in
                              ("left_arm", "right_arm", "left_hand", "right_hand", "waist",
                               "base_height_command", "navigate_command")])
        self.actions.append(row.astype(np.float32))
        self.q[Q43_GROUP_INDICES["left_arm"]] = row[0:7]
        self.q[Q43_GROUP_INDICES["right_arm"]] = row[7:14]
        k = self.k
        self.k += 1
        return self.obs(), 0.0, False, False, {"success": self.success_at is not None and k == self.success_at}


def ramp_rows(vla_obs: dict, rt: dict) -> np.ndarray:
    """mock 'ramp': arm row i = sent arm state + i * delta, nav 0.2 on odd rows of k_obs parity."""
    s = state43(vla_obs)
    rows = np.zeros((H, A35), dtype=np.float32)
    arms = np.concatenate([s[15:22], s[22:29]])  # STATE order: left_arm 15:22, right_arm 22:29
    for i in range(H):
        rows[i, ARMS] = arms + np.float32(i) * DELTA
    rows[:, 31] = 0.74
    rows[:, NAV] = np.float32(0.2)
    return rows


class FakeVla:
    """VlaClient stand-in. RT: poll() returns the reply once env.k >= k_obs + d(seq).

    ``d_of`` maps the request index (0 = pre-roll) to its delay in ticks, or to "timeout" (poll
    returns TIMEOUT at k_obs + 26, like the 500 ms give-up) / "error" (ERROR on the next poll).
    """

    def __init__(self, env: FakeEnv, d_of=lambda i: 4, rows_fn=ramp_rows):
        self.env, self.d_of, self.rows_fn = env, d_of, rows_fn
        self._rt = None
        self._obs = None
        self.give_up_ms = 500
        self.last_request_bytes = 0
        self.drained = None
        self.n = 0
        self.sent: list[tuple[int, int, int]] = []  # (env tick at send, k_obs, seq)
        self.stats = {}

    @property
    def in_flight(self) -> bool:
        return self._rt is not None

    @property
    def inflight_rt(self):
        return self._rt

    def send(self, vla_obs: dict, rt: dict) -> None:
        assert self._rt is None, "send while a request is in flight (REQ would raise EFSM)"
        self._rt = dict(rt, t_send_ns=int(self.env.k * TICK_NS + 1000))
        self._obs = vla_obs
        self._i = self.n
        self.n += 1
        self.sent.append((self.env.k, int(rt["k_obs"]), int(rt["seq"])))

    def _finish(self, status, error=None) -> VlaReply:
        rt, self._rt = self._rt, None
        rows = self.rows_fn(self._obs, rt) if status == ChunkStatus.OK else None
        return VlaReply(seq=rt["seq"], gen=rt["gen"], status=status, action=None, info={},
                        t_send_ns=rt["t_send_ns"], t_recv_ns=int(self.env.k * TICK_NS + 500),
                        ser_us=0.0, deser_us=0.0, error=error, rows=rows, rt=rt)

    def poll(self):
        if self._rt is None:
            return None
        d = self.d_of(self._i)
        k_obs = self._rt["k_obs"]
        if d == "error":
            return self._finish(ChunkStatus.ERROR, "error")
        if d == "timeout":
            return self._finish(ChunkStatus.TIMEOUT, "timeout") if self.env.k >= k_obs + 26 else None
        if self.env.k >= k_obs + d:
            return self._finish(ChunkStatus.OK)
        return None

    def wait(self, timeout_ms: int):
        if self._rt is None:
            return None
        d = self.d_of(self._i)
        if d in ("error", "timeout"):
            return self._finish(ChunkStatus.ERROR if d == "error" else ChunkStatus.TIMEOUT, d)
        return self._finish(ChunkStatus.OK)

    def drain(self, timeout_ms: int = 500) -> str:
        self.drained = None
        if self._rt is None:
            return "none"
        self.drained = self._finish(ChunkStatus.OK)
        return "ok"


class FakePacer:
    t0 = 0

    def anchor(self, t0_ns=None):
        return 0

    def deadline(self, k: int) -> int:
        return k * TICK_NS

    def wait(self, k: int):
        return k * TICK_NS, 0, False


def make_loop(tmp_path, mode: str, arm: str = "v1", s: int = 16, B: int = 0, d_inj: int = 4,
              d_init: int = 4, max_ticks: int = 200, d_of=lambda i: 4, env: FakeEnv | None = None,
              sync_rows: int = 20, crn: bool = False, rows_fn=ramp_rows):
    """A Loop with the production tick code and fake I/O (no env build, no sockets)."""
    lp = object.__new__(L.Loop)
    env = env or FakeEnv()
    lp.args = SimpleNamespace(max_ticks=max_ticks, d_inj=d_inj, crn=crn, block_timeout_ms=30000,
                              dump_vla_obs=0, vla="fake://", sync_rows=sync_rows, arm=arm, s=s, B=B)
    lp.mode = mode
    lp.rt, lp.vl, lp.sync = mode == "rt", mode == "vl", mode == "sync"
    lp.cfg = sp.SplicerConfig(arm=arm, s=s, B=B, d_init=d_inj if lp.vl else d_init,
                              sync_rows=sync_rows)
    lp.st = sp.new_state()
    lp.env = env
    lp.h = SimpleNamespace(lower=SimpleNamespace(cmd=np.zeros(3, np.float32)),
                           wbc=SimpleNamespace(last_action={"q": np.zeros(43)}))
    lp.vla = FakeVla(env, d_of, rows_fn)
    lp.vla_be = VlaBackend.MOCK
    lp.pacer = FakePacer()
    lp.runtime = SimpleNamespace(tick=0, last=None, n_calls={})
    lp.pub_on = False
    lp.pub = None
    lp.gen = 0
    lp.next_seq = 1
    lp.n_vla_dumped = 0
    lp.inflight_k_obs = -1
    lp.ep = lp.seed = lp.ep_req_i = 0
    lp.ep_outcomes = Counter()
    lp.usage, lp.jumps = {}, {}
    lp.stop_requested = False
    lp.run = telemetry.RunDir(tmp_path, f"{mode}_{arm}", {})
    lp.tlog = telemetry.TickLog(lp.run, capacity=4096)
    lp.rlog = telemetry.RequestLog(lp.run)
    return lp


def run_episode(lp, seed: int = 0, ep: int = 0):
    """The tick part of Loop.episode(): gen bump, splicer reset, pre-roll, ticks, drain."""
    obs = lp.env.reset()
    lp.gen += 1
    sp.reset(lp.st, lp.gen)
    lp.ep, lp.seed, lp.ep_req_i = ep, seed, 0
    lp.ep_outcomes = Counter()
    lp.usage, lp.jumps = {}, {}
    row0 = len(lp.tlog)
    chunk0 = lp._blocking_call(0, obs, preroll=True)
    if lp.sync:
        end, k_end, fst, stored = lp._ticks_sync(obs, chunk0)
    else:
        end, k_end, fst, stored = lp._ticks_async(obs)
    drain = lp._drain(k_end, stored)
    for seq, (first, last, n) in lp.usage.items():
        lp.rlog.update(seq, first_row=first, last_row=last, n_rows_used=n)
    lp.rlog.end_episode()
    ticks = lp.tlog.rows()[row0:].copy()
    return SimpleNamespace(end=end, k_end=k_end, fst=fst, drain=drain, ticks=ticks,
                           reqs=lp.rlog.rows().copy(), sent=list(lp.vla.sent),
                           actions=np.array(lp.env.actions))
