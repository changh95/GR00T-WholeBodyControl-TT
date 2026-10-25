"""T-1.1 schema invariants (PLAN.md §1.4): layouts, record sizes, codec round trips, the 9-key VLA obs.

The @pytest.mark.sim test checks the hard-coded joint-group indices and the obs shapes against the
live G1 env (about 10 s to build).
"""

import numpy as np
import pytest

from rtloop import schema
from rtloop.schema import (A35, ACTION_GROUPS, ACTION_SLICES, ARMS, BLEND, CHUNK_DTYPE, H, HANDS, HEIGHT,
                           IMAGE_SHAPE, LANGUAGE_KEY, NAV, Q43_GROUP_INDICES, REQUEST_DTYPE, S43, STATE_GROUPS,
                           STATE_KEYS, STATE_SLICES, VIDEO_KEY, VLA_OBS_KEYS, WAIST, Chunk)

# state43 order (STATE_GROUPS) as indices into the env's q43 (Q43_GROUP_INDICES order)
Q43_TO_STATE43 = np.concatenate([Q43_GROUP_INDICES[name] for name, _ in STATE_GROUPS])


def _cover(slices, n):
    idx = np.concatenate([np.arange(n)[sl] for sl in slices])
    assert len(idx) == n and sorted(idx.tolist()) == list(range(n)), idx


def test_a35_slices_cover_0_to_34_disjointly():
    assert A35 == 35
    _cover([ACTION_SLICES[name] for name, _ in ACTION_GROUPS], A35)
    assert [ACTION_SLICES[name].start for name, _ in ACTION_GROUPS] == [0, 7, 14, 21, 28, 31, 32]
    np.testing.assert_array_equal(ARMS, np.r_[0:14])
    np.testing.assert_array_equal(HANDS, np.r_[14:28])
    np.testing.assert_array_equal(WAIST, np.r_[28:31])
    assert HEIGHT == 31
    np.testing.assert_array_equal(NAV, np.r_[32:35])
    # the crossfade set is arms + waist + nav; hands and base height always switch hard
    assert sorted(BLEND.tolist()) == sorted(np.r_[ARMS, WAIST, NAV].tolist())
    assert not set(BLEND.tolist()) & (set(HANDS.tolist()) | {HEIGHT})


def test_state_and_q43_layouts():
    assert S43 == 43
    _cover([STATE_SLICES[name] for name, _ in STATE_GROUPS], S43)
    q = np.concatenate([Q43_GROUP_INDICES[g] for g in Q43_GROUP_INDICES])
    assert sorted(q.tolist()) == list(range(43))
    assert Q43_GROUP_INDICES["left_leg"] + Q43_GROUP_INDICES["right_leg"] + Q43_GROUP_INDICES["waist"] == list(range(15))
    for name, n in STATE_GROUPS:
        assert len(Q43_GROUP_INDICES[name]) == n
    assert Q43_TO_STATE43.tolist() == list(range(22)) + list(range(29, 36)) + list(range(22, 29)) + list(range(36, 43))


def test_chunk_dtype_layout():
    assert CHUNK_DTYPE.itemsize == 4264
    offsets = {name: CHUNK_DTYPE.fields[name][1] for name in CHUNK_DTYPE.names}
    assert offsets == {"seq": 0, "gen": 4, "k_obs": 8, "k_arr": 16, "t_obs_ns": 24, "t_send_ns": 32,
                       "t_recv_ns": 40, "d": 48, "vla_be": 50, "status": 51, "pad": 52, "rows": 64}
    assert CHUNK_DTYPE["rows"].shape == (H, A35) and CHUNK_DTYPE["rows"].base == np.dtype("<f4")
    assert REQUEST_DTYPE["state43_sent"].shape == (S43,) and REQUEST_DTYPE["obs_sha1"].shape == (20,)


def test_chunk_to_record():
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((H, A35)).astype(np.float32)
    c = Chunk(seq=7, gen=3, k_obs=100, rows=rows, t_obs_ns=1, t_send_ns=2, t_recv_ns=3, k_arr=104,
              vla_be=schema.VlaBackend.GPU, status=schema.ChunkStatus.OK)
    rec = c.to_record()
    assert rec.dtype == CHUNK_DTYPE and rec.tobytes().__len__() == 4264
    assert (int(rec["seq"]), int(rec["gen"]), int(rec["k_obs"]), int(rec["k_arr"]), int(rec["d"])) == (7, 3, 100, 104, 4)
    assert int(rec["vla_be"]) == 1 and int(rec["status"]) == 0
    np.testing.assert_array_equal(rec["rows"], rows)


def test_rows_actions_round_trip():
    rng = np.random.default_rng(1)
    rows = rng.standard_normal((H, A35)).astype(np.float32)
    action = schema.rows_to_actions(rows)
    assert set(action) == {f"action.{name}" for name, _ in ACTION_GROUPS}
    for name, n in ACTION_GROUPS:
        a = action[f"action.{name}"]
        assert a.shape == (1, H, n) and a.dtype == np.float32
    back = schema.actions_to_rows(action)
    assert back.dtype == np.float32 and back.shape == (H, A35)
    assert back.tobytes() == rows.tobytes()  # bit-exact
    # 2-D (T, D) arrays and the model's 50 rows: only rows [0, 30) are kept
    rows50 = rng.standard_normal((50, A35)).astype(np.float32)
    a50 = {k: v[0] for k, v in schema.rows_to_actions(rows50).items()}
    np.testing.assert_array_equal(schema.actions_to_rows(a50), rows50[:H])
    with pytest.raises(AssertionError):
        schema.actions_to_rows({k: v[:, :10] for k, v in action.items()})  # too few rows


def test_row_to_env_action():
    row = np.arange(A35, dtype=np.float32) * 0.1
    act = schema.row_to_env_action(row)
    assert list(act) == [f"action.{name}" for name, _ in ACTION_GROUPS]
    for name, n in ACTION_GROUPS:
        a = act[f"action.{name}"]
        assert a.shape == (n,) and a.dtype == np.float32
        np.testing.assert_array_equal(a, row[ACTION_SLICES[name]])
    act["action.left_arm"][:] = 99.0  # copies, not views
    assert row[0] == 0.0
    with pytest.raises(AssertionError):
        schema.row_to_env_action(np.zeros(34, np.float32))


def synthetic_obs(seed: int = 0) -> dict:
    """An env-shaped observation: f64 states consistent with q43, a u8 image, a str instruction, extras."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1, 1, 43)
    obs = {"q": q, "dq": rng.standard_normal(43), "floating_base_pose": rng.standard_normal(7),
           VIDEO_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
           "video.tpp_view": rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
           LANGUAGE_KEY: "pick up the apple and place it on the plate"}
    for name, _ in STATE_GROUPS:
        obs[f"state.{name}"] = q[Q43_GROUP_INDICES[name]]
    return obs


def check_vla_obs(vo: dict, obs: dict):
    assert tuple(vo) == VLA_OBS_KEYS and len(vo) == 9
    v = vo[VIDEO_KEY]
    assert v.shape == (1, 1, *IMAGE_SHAPE) and v.dtype == np.uint8 and v.flags.c_contiguous
    np.testing.assert_array_equal(v[0, 0], obs[VIDEO_KEY])
    for name, n in STATE_GROUPS:
        a = vo[f"state.{name}"]
        assert a.shape == (1, 1, n) and a.dtype == np.float32
        np.testing.assert_array_equal(a[0, 0], np.asarray(obs[f"state.{name}"]).astype(np.float32))
    lang = vo[LANGUAGE_KEY]
    assert isinstance(lang, tuple) and len(lang) == 1 and isinstance(lang[0], str)
    assert lang[0] == str(obs[LANGUAGE_KEY])


def test_build_vla_obs_from_synthetic_obs():
    obs = synthetic_obs()
    vo = schema.build_vla_obs(obs)
    check_vla_obs(vo, obs)
    # copies: editing the request never reaches the env obs
    vo[VIDEO_KEY][...] = 0
    vo["state.left_arm"][...] = 0
    assert obs[VIDEO_KEY].any() and np.abs(obs["state.left_arm"]).sum() > 0
    assert schema.obs_sha1(schema.build_vla_obs(obs)).shape == (20,)


def test_obs_sha1_sees_one_bit():
    obs = synthetic_obs()
    h0 = schema.obs_sha1(schema.build_vla_obs(obs))
    vo = schema.build_vla_obs(obs)
    vo["state.right_hand"].view(np.uint32)[0, 0, 3] ^= 1
    assert not np.array_equal(schema.obs_sha1(vo), h0)
    vo = schema.build_vla_obs(obs)
    vo[LANGUAGE_KEY] = ("something else",)
    assert np.array_equal(schema.obs_sha1(vo), h0)  # images and states only


def test_state43_order():
    obs = synthetic_obs()
    s = schema.state43(obs)
    assert s.shape == (43,) and s.dtype == np.float32
    np.testing.assert_array_equal(s, np.concatenate([np.asarray(obs[k], np.float32) for k in STATE_KEYS]))
    np.testing.assert_array_equal(s, obs["q"][Q43_TO_STATE43].astype(np.float32))
    for name, _ in STATE_GROUPS:
        np.testing.assert_array_equal(s[STATE_SLICES[name]], np.asarray(obs[f"state.{name}"], np.float32))
    np.testing.assert_array_equal(schema.state43(schema.build_vla_obs(obs)), s)  # same on the request


def test_hold_row():
    obs = synthetic_obs()
    row = schema.hold_row(obs)
    assert row.shape == (A35,) and row.dtype == np.float32
    for name in ("left_arm", "right_arm", "left_hand", "right_hand", "waist"):
        np.testing.assert_array_equal(row[ACTION_SLICES[name]], np.asarray(obs[f"state.{name}"], np.float32))
    assert row[HEIGHT] == np.float32(schema.DEFAULT_BASE_HEIGHT)
    assert np.all(row[NAV] == 0.0)


@pytest.mark.sim
def test_live_env_joint_groups_and_vla_obs():
    import gymnasium as gym
    from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401  (registers the env ids)
    from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
    from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
    import robocasa  # noqa: F401

    env = gym.make(
        "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
        onscreen=False,
        offscreen=True,
        enable_waist=True,
        randomize_cameras=False,
        camera_names=["robot0_oak_egoview", "robot0_rs_tppview"],
    )
    env = WholeBodyControlWrapper(env, BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict())
    try:
        rm = env.robot_model
        for group, idx in Q43_GROUP_INDICES.items():
            assert list(rm.get_joint_group_indices(group)) == idx, group
        assert list(rm.get_joint_group_indices("lower_body")) == list(range(15))
        obs, _ = env.reset(seed=0)
        assert np.asarray(obs["q"]).shape == (43,)
        for name, n in STATE_GROUPS:
            st = np.asarray(obs[f"state.{name}"])
            assert st.shape == (n,) and st.dtype == np.float64
            np.testing.assert_array_equal(st, np.asarray(obs["q"])[Q43_GROUP_INDICES[name]])
        assert isinstance(obs[LANGUAGE_KEY], str)
        vo = schema.build_vla_obs(obs)
        check_vla_obs(vo, obs)
        assert vo[VIDEO_KEY].any()  # a real render, not an all-black frame
        np.testing.assert_array_equal(schema.state43(obs), np.asarray(obs["q"])[Q43_TO_STATE43].astype(np.float32))
    finally:
        env.close()
