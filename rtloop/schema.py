"""Shared constants, array layouts and record dtypes of the rtloop control stack (PLAN.md §1.4).

Everything that crosses a module boundary is defined here: the 35-wide action row (A35), the
43-wide state vector, the 9-key VLA observation, the ChunkRecord, and the TickLog / RequestLog
dtypes. Layouts are packed little-endian so the records can later be mirrored by C structs on the
X280 (M4).
"""

from dataclasses import dataclass, field
import enum
import hashlib

import numpy as np

# --- Timing -------------------------------------------------------------------------------------

DT = 0.02  # control tick, s (50 Hz)
TICK_NS = 20_000_000
H = 30  # trained action-chunk rows (0.6 s); model rows 30..49 are untrained and never used

# --- A35 action row (decode order, PLAN.md §1.4.2) ---------------------------------------------

ACTION_GROUPS = (
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 7),
    ("right_hand", 7),
    ("waist", 3),
    ("base_height_command", 1),
    ("navigate_command", 3),
)
A35 = sum(n for _, n in ACTION_GROUPS)


def _slices(groups):
    out, i = {}, 0
    for name, n in groups:
        out[name] = slice(i, i + n)
        i += n
    return out


ACTION_SLICES = _slices(ACTION_GROUPS)
ARMS = np.r_[ACTION_SLICES["left_arm"], ACTION_SLICES["right_arm"]]
HANDS = np.r_[ACTION_SLICES["left_hand"], ACTION_SLICES["right_hand"]]
WAIST = np.arange(28, 31)
HEIGHT = 31
NAV = np.arange(32, 35)
# Crossfaded on a chunk switch when B > 0; hands and base height always switch hard.
BLEND = np.r_[ARMS, WAIST, NAV]

DEFAULT_BASE_HEIGHT = 0.74  # m, the value used in training

# --- State (VLA observation order) -------------------------------------------------------------

STATE_GROUPS = (
    ("left_leg", 6),
    ("right_leg", 6),
    ("waist", 3),
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 7),
    ("right_hand", 7),
)
S43 = sum(n for _, n in STATE_GROUPS)
STATE_SLICES = _slices(STATE_GROUPS)

# Indices into the env's 43-dof q vector, as robot_model.get_joint_group_indices() returns them
# for the G1 with waist (checked against the live robot model by tests/test_schema.py).
Q43_GROUP_INDICES = {
    "left_leg": list(range(0, 6)),
    "right_leg": list(range(6, 12)),
    "waist": list(range(12, 15)),
    "left_arm": list(range(15, 22)),
    "left_hand": list(range(22, 29)),
    "right_arm": list(range(29, 36)),
    "right_hand": list(range(36, 43)),
}

# --- VLA observation (W-VLA request, PLAN.md §1.4.1) -------------------------------------------

VIDEO_KEY = "video.ego_view"
LANGUAGE_KEY = "annotation.human.task_description"
STATE_KEYS = tuple(f"state.{name}" for name, _ in STATE_GROUPS)
VLA_OBS_KEYS = (VIDEO_KEY, *STATE_KEYS, LANGUAGE_KEY)
IMAGE_SHAPE = (480, 640, 3)


def build_vla_obs(obs: dict) -> dict:
    """The 9-key observation sent to the VLA, shaped (B=1, T=1, ...) as the stock rollout sends it.

    States are cast to float32 the way gymnasium's SyncVectorEnv casts them (float64 -> float32),
    so the request is bit-identical to the stock path's for the same env observation.
    """
    out = {VIDEO_KEY: np.ascontiguousarray(obs[VIDEO_KEY], dtype=np.uint8)[None, None].copy()}
    for key in STATE_KEYS:
        out[key] = np.asarray(obs[key], dtype=np.float32)[None, None].copy()
    out[LANGUAGE_KEY] = (str(obs[LANGUAGE_KEY]),)
    return out


def request_abandoned(rt, t_recv_ns: int) -> bool:
    """True if the client had already given up on this get_action when the server took it off its socket.

    VlaClient stamps options.rt with t_send_ns and give_up_ms (CLOCK_MONOTONIC, shared by client and server
    on one host). A REP server answers queued requests in order, so without this check every request
    abandoned during an outage would still cost a full inference before the live one is served. Requests
    without both stamps (stock clients) are never abandoned.
    """
    if not isinstance(rt, dict):
        return False
    t_send, give_up = rt.get("t_send_ns"), rt.get("give_up_ms")
    if not isinstance(t_send, int) or not isinstance(give_up, (int, float)) or give_up <= 0:
        return False
    return int(t_recv_ns) - t_send > give_up * 1_000_000


def state43(obs: dict) -> np.ndarray:
    """float32[43] in STATE_GROUPS order, taken from the env observation (or a VLA obs dict)."""
    parts = [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in STATE_KEYS]
    out = np.concatenate(parts)
    assert out.shape == (S43,), out.shape
    return out


def obs_sha1(vla_obs: dict) -> np.ndarray:
    """uint8[20] digest of the image and state bytes of a VLA observation (RequestLog audit)."""
    h = hashlib.sha1()
    for key in VLA_OBS_KEYS[:-1]:
        h.update(np.ascontiguousarray(vla_obs[key]).tobytes())
    return np.frombuffer(h.digest(), dtype=np.uint8).copy()


def actions_to_rows(action: dict) -> np.ndarray:
    """A W-VLA reply's action dict -> float32 (H, 35) rows in A35 order.

    Accepts ``action.<group>`` arrays shaped (1, T, D) or (T, D) with T >= H; keeps rows [0, H).
    """
    rows = np.empty((H, A35), dtype=np.float32)
    for name, n in ACTION_GROUPS:
        a = np.asarray(action[f"action.{name}"], dtype=np.float32)
        if a.ndim == 3:
            assert a.shape[0] == 1, a.shape
            a = a[0]
        assert a.ndim == 2 and a.shape[0] >= H and a.shape[1] == n, (name, a.shape)
        rows[:, ACTION_SLICES[name]] = a[:H]
    return rows


def rows_to_actions(rows: np.ndarray) -> dict:
    """float32 (T, 35) rows -> a W-VLA reply action dict with (1, T, D) arrays (mock servers)."""
    rows = np.asarray(rows, dtype=np.float32)
    return {f"action.{name}": rows[None, :, ACTION_SLICES[name]].copy() for name, _ in ACTION_GROUPS}


def row_to_env_action(row: np.ndarray) -> dict:
    """One A35 row -> the per-step action dict WholeBodyControlWrapper.step() takes.

    Same keys, shapes (D,) and dtype (float32) as MultiStepWrapper hands to the env for one step
    of a stock chunk.
    """
    row = np.asarray(row, dtype=np.float32)
    assert row.shape == (A35,), row.shape
    return {f"action.{name}": row[ACTION_SLICES[name]].copy() for name, _ in ACTION_GROUPS}


def hold_row(obs: dict) -> np.ndarray:
    """A35 row that holds the current upper-body pose, stands still at the default height."""
    row = np.zeros(A35, dtype=np.float32)
    for name in ("left_arm", "right_arm", "left_hand", "right_hand", "waist"):
        row[ACTION_SLICES[name]] = np.asarray(obs[f"state.{name}"], dtype=np.float32)
    row[HEIGHT] = DEFAULT_BASE_HEIGHT
    return row


# --- WBC (W-WBC, PLAN.md §1.4.3) ---------------------------------------------------------------

WBC_FRAME = 86
WBC_HISTORY = 6
WBC_OBS_DIM = WBC_FRAME * WBC_HISTORY  # 516
WBC_ACT_DIM = 15
WBC_BALANCE = "GR00T-WholeBodyControl-Balance"
WBC_WALK = "GR00T-WholeBodyControl-Walk"
WBC_POLICIES = (WBC_BALANCE, WBC_WALK)
WALK_THRESHOLD = 0.05  # |nav cmd| below this selects Balance (g1_gear_wbc_policy.py:257)


class WbcBackend(enum.IntEnum):
    ORT = 0
    GPU = 1
    X280 = 2
    X280_SIM = 3


class VlaBackend(enum.IntEnum):
    MOCK = 0
    GPU = 1
    TT = 2


# --- ChunkRecord (PLAN.md §1.4.2) --------------------------------------------------------------


class ChunkStatus(enum.IntEnum):
    OK = 0
    ERROR = 1
    TIMEOUT = 2
    BAD_SHAPE = 3


CHUNK_DTYPE = np.dtype(
    [
        ("seq", "<u4"),
        ("gen", "<u4"),
        ("k_obs", "<i8"),
        ("k_arr", "<i8"),
        ("t_obs_ns", "<i8"),
        ("t_send_ns", "<i8"),
        ("t_recv_ns", "<i8"),
        ("d", "<i2"),
        ("vla_be", "u1"),
        ("status", "u1"),
        ("pad", "u1", (12,)),
        ("rows", "<f4", (H, A35)),
    ],
    align=False,
)
assert CHUNK_DTYPE.itemsize == 4264 and CHUNK_DTYPE.fields["rows"][1] == 64


@dataclass
class Chunk:
    """One VLA action chunk and its timing stamps (all ticks are control-tick indices).

    ``k_base`` is the tick that row 0 targets: ``k_obs`` under the aligned policy (index law: row i
    targets tick k_obs + i), ``k_arr`` for the arms that restart a chunk at row 0 on arrival.
    """

    seq: int
    gen: int
    k_obs: int
    rows: np.ndarray  # float32 (H, 35)
    t_obs_ns: int = 0
    t_send_ns: int = 0
    t_recv_ns: int = 0
    k_arr: int = -1
    k_base: int = -1
    vla_be: int = VlaBackend.MOCK
    status: int = ChunkStatus.OK
    info: dict = field(default_factory=dict)  # server INFO, not part of the record

    @property
    def d(self) -> int:
        return self.k_arr - self.k_obs

    def to_record(self) -> np.ndarray:
        rec = np.zeros((), dtype=CHUNK_DTYPE)
        for name in ("seq", "gen", "k_obs", "k_arr", "t_obs_ns", "t_send_ns", "t_recv_ns",
                     "vla_be", "status"):
            rec[name] = getattr(self, name)
        rec["d"] = self.d
        rec["rows"] = self.rows
        return rec


# --- Request outcomes and episode ends ---------------------------------------------------------


class Outcome(enum.IntEnum):
    PENDING = 0
    APPLIED = 1
    LATE_REJECT = 2
    GEN_DROP = 3
    TIMEOUT = 4
    ERROR = 5
    DRAINED = 6


class EpisodeEnd(enum.IntEnum):
    SUCCESS = 0
    TRUNC = 1
    FALL = 2
    SAFETY_FAULT = 3


FALL_PELVIS_Z = 0.50  # m; standing is about 0.752 m
MAX_EPISODE_TICKS = 1440

# --- TickLog (PLAN.md §4.1) ---------------------------------------------------------------------

TICK_DTYPE = np.dtype(
    [
        ("ep", "<u4"),
        ("k", "<i8"),
        ("t_start_ns", "<i8"),
        ("t_end_ns", "<i8"),
        ("period_us", "<f4"),
        ("compute_us", "<f4"),
        ("late_us", "<f4"),
        ("slack_us", "<f4"),
        ("overrun", "u1"),
        ("slip", "u1"),
        ("sim_us", "<f4"),
        ("viewer_us", "<f4"),
        ("wbc_rtt_us", "<f4"),
        ("wbc_dev_us", "<f4"),
        ("wbc_be", "u1"),
        ("wbc_fb", "u1"),
        ("wbc_early", "u1"),
        ("policy", "u1"),  # 0 Balance, 1 Walk
        ("shadow_da", "<f4"),
        ("chunk_seq", "<u4"),
        ("row", "<i2"),  # -1 = no chunk (hold)
        ("age", "<i2"),
        ("rows_left", "<i2"),
        ("starved", "u1"),
        ("in_flight", "u1"),
        ("splice", "u1"),
        ("jump_rad", "<f4"),
        ("cmd35", "<f4", (A35,)),
        ("nav_wbc", "<f4", (3,)),
        ("q15", "<f4", (WBC_ACT_DIM,)),
        ("q43_meas", "<f4", (43,)),
        ("q43_cmd", "<f4", (43,)),
        ("dq_arm_absmax", "<f4"),
        ("pelvis_z", "<f4"),
        ("pelvis_rp", "<f4", (2,)),
        ("success", "u1"),
        ("safety", "u1"),
    ]
)

# --- RequestLog (PLAN.md §4.1; one row per VLA request) ----------------------------------------

REQUEST_DTYPE = np.dtype(
    [
        ("seq", "<u4"),
        ("gen", "<u4"),
        ("ep", "<u4"),
        ("k_obs", "<i8"),
        ("k_arr", "<i8"),
        ("d", "<i2"),
        ("d_hat", "<i2"),
        ("d_hat_max", "<i2"),
        ("s_eff", "<i2"),
        ("t_obs_ns", "<i8"),
        ("t_send_ns", "<i8"),
        ("t_recv_ns", "<i8"),
        ("noise_seed", "<u8"),
        ("ser_us", "<f4"),
        ("deser_us", "<f4"),
        ("srv_infer_ms", "<f4"),
        ("state43_sent", "<f4", (S43,)),
        ("obs_sha1", "u1", (20,)),
        ("first_row", "<i2"),
        ("last_row", "<i2"),
        ("n_rows_used", "<i2"),
        ("outcome", "u1"),
        ("switch_jump", "<f4"),
    ]
)
