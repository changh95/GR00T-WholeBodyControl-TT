"""PLAN.md §1.4.1: the W-VLA request the loop puts on the wire is the 9-key table, and the stock server accepts it.

One RT run (2 episodes x 120 ticks, ORT WBC) talks to tests/stub_policy_server.py: gr00t's own
PolicyServer + Gr00tSimPolicyWrapper(strict=True) with the checkpoint's UNITREE_G1 modality
configs and no model, which captures the raw bytes of every get_action request. The loop runs
under tests/guard_probe.py --record-obs, which keeps a float64 copy of o_k (the observation at the
start of tick k) for every tick. From those two independent records:

  1. every request was accepted by the real strict checks (flat wrapper, nested Gr00tPolicy,
     check_action on the reply) and was applied; the loop's startup modality-config check ran;
  2. the observation has exactly schema.VLA_OBS_KEYS, in order, with the table's dtypes and shapes;
     options is exactly {"rt": {seq, gen, ep, k_obs, t_obs_ns, t_send_ns, give_up_ms}} (no noise_seed
     without --crn; give_up_ms lets an RT-owned server skip requests the client abandoned);
  3. each request is <= 1.0 MB (decimal) and is the size the loop logged;
  4. its state arrays are bit-equal to np.float32 of the env observation of tick k_obs, its image to
     the env's ego frame of that tick, its instruction to the env's; the RequestLog audit
     (state43_sent, obs_sha1) agrees with the wire;
  5. the same wrapper rejects mutated copies (float64 state, missing time axis, float video, bare
     str language), so the acceptance in 1. is not vacuous.

`--no-affinity`: other agents run RT loops pinned to the tick CPU concurrently; two spinning tick
threads on one core would disturb both runs. Pinning is not what this test checks.
"""

import hashlib
import json

import numpy as np
import pytest

from fault_utils import ProbeRun, free_port, kill9, load_request_extras, spawn_stub_server
from rtloop import wire
from rtloop.schema import (IMAGE_SHAPE, LANGUAGE_KEY, STATE_GROUPS, STATE_KEYS, VIDEO_KEY,
                           VLA_OBS_KEYS, Outcome, obs_sha1)

MAX_TICKS = 120
MAX_BYTES = 1_000_000
RT_KEYS = {"seq", "gen", "ep", "k_obs", "t_obs_ns", "t_send_ns", "give_up_ms"}
TIMEOUT_S = 400


@pytest.fixture(scope="module")
def schema_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("obs_schema")
    cap = tmp / "capture"
    stub, url = spawn_stub_server(free_port(), cap)
    run = None
    try:
        run = ProbeRun(tmp, "rt_schema", ["--record-obs"],
                       ["--mode", "rt", "--no-affinity", "--vla", url, "--wbc", "ort",
                        "--seeds", "0-1", "--max-ticks", str(MAX_TICKS)])
        run_dir = run.finish(TIMEOUT_S)
        ticks, reqs, eps = run.load()
        raw = [p.read_bytes() for p in sorted(cap.glob("*.msgpack"))]
        log = [json.loads(line) for line in (cap / "log.jsonl").read_text().splitlines() if line.strip()]
        config = json.loads((run_dir / "config.json").read_text())
        yield {"run": run, "dir": run_dir, "ticks": ticks, "reqs": reqs, "eps": eps, "raw": raw,
               "log": log, "config": config, "obs": run.obs(), "extras": load_request_extras(run_dir)}
    finally:
        if run is not None:
            run.kill()
        kill9(stub)


def _decode_gr00t(raw: bytes) -> dict:
    """Decode with gr00t's own MsgSerializer (what the stock server runs), not rtloop.wire."""
    from gr00t.policy.server_client import MsgSerializer

    return MsgSerializer.from_bytes(raw)


def _env_row(o: dict) -> dict:
    return {(int(e), int(k)): i for i, (e, k) in enumerate(zip(o["ep"], o["k"]))}


@pytest.mark.sim
@pytest.mark.rt
def test_every_request_accepted_by_strict_wrapper(schema_run, capsys):
    reqs, log, eps = schema_run["reqs"], schema_run["log"], schema_run["eps"]
    assert len(eps) == 2 and all(e["end"] in ("TRUNC", "SUCCESS") for e in eps)
    assert len(reqs) >= 2 * (MAX_TICKS // 16), len(reqs)
    assert len(log) == len(reqs) == len(schema_run["raw"])
    bad = [e for e in log if not e["ok"]]
    assert not bad, f"rejected by Gr00tSimPolicyWrapper: {bad[:3]}"
    assert sorted(e["rt"]["seq"] for e in log) == sorted(int(s) for s in reqs["seq"])
    assert (reqs["outcome"] == Outcome.APPLIED).all(), np.unique(reqs["outcome"])
    # The loop asked the server for get_modality_config at startup and matched it to the table.
    assert schema_run["config"]["modality_config_checked"] is True
    with capsys.disabled():
        print(f"\n[obs schema] {len(log)} requests accepted by PolicyServer + "
              f"Gr00tSimPolicyWrapper(strict=True) (UNITREE_G1 modality configs), all APPLIED")


@pytest.mark.sim
@pytest.mark.rt
def test_keys_dtypes_shapes(schema_run):
    dims = dict(STATE_GROUPS)
    for raw in schema_run["raw"]:
        req = _decode_gr00t(raw)
        assert set(req) == {"endpoint", "data"} and req["endpoint"] == "get_action"
        data = req["data"]
        assert set(data) == {"observation", "options"}
        obs = data["observation"]
        assert tuple(obs) == VLA_OBS_KEYS, tuple(obs)
        v = obs[VIDEO_KEY]
        assert isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.shape == (1, 1, *IMAGE_SHAPE)
        for key in STATE_KEYS:
            a = obs[key]
            want = (1, 1, dims[key.split(".", 1)[1]])
            assert isinstance(a, np.ndarray) and a.dtype == np.float32 and a.shape == want, \
                (key, a.dtype, a.shape)
        lang = obs[LANGUAGE_KEY]  # a tuple on the loop side; msgpack has only arrays -> list
        assert isinstance(lang, list) and len(lang) == 1 and isinstance(lang[0], str) and lang[0]
        assert set(data["options"]) == {"rt"}
        rt = data["options"]["rt"]
        assert set(rt) == RT_KEYS, sorted(rt)
        assert all(isinstance(rt[k], int) for k in RT_KEYS)
        assert rt["t_obs_ns"] <= rt["t_send_ns"]


@pytest.mark.sim
@pytest.mark.rt
def test_payload_at_most_1MB(schema_run, capsys):
    sizes = [len(r) for r in schema_run["raw"]]
    assert max(sizes) <= MAX_BYTES, max(sizes)
    extras = schema_run["extras"]
    for raw in schema_run["raw"]:
        seq = wire.loads(raw)["data"]["options"]["rt"]["seq"]
        assert extras[seq]["bytes"] == len(raw)  # what the loop logged is what went out
    with capsys.disabled():
        print(f"\n[obs schema] request payload {min(sizes)}..{max(sizes)} B (<= {MAX_BYTES})")


@pytest.mark.sim
@pytest.mark.rt
def test_state_and_image_are_env_obs_of_k_obs(schema_run, capsys):
    o, reqs = schema_run["obs"], schema_run["reqs"]
    row_of, by_seq = _env_row(o), {int(s): i for i, s in enumerate(reqs["seq"])}
    lang = dict(zip(o["lang_ep"].tolist(), o["lang"].tolist()))
    for raw in schema_run["raw"]:
        obs = _decode_gr00t(raw)["data"]["observation"]
        rt = wire.loads(raw)["data"]["options"]["rt"]
        n = row_of[(rt["ep"], rt["k_obs"])]
        sent = np.concatenate([obs[k].reshape(-1) for k in STATE_KEYS])
        want = o["state43"][n].astype(np.float32)  # np.float32(env obs), the SyncVectorEnv cast
        assert sent.tobytes() == want.tobytes(), (rt["seq"], rt["k_obs"])
        img_sha1 = np.frombuffer(hashlib.sha1(obs[VIDEO_KEY].tobytes()).digest(), np.uint8)
        assert np.array_equal(img_sha1, o["img_sha1"][n]), (rt["seq"], rt["k_obs"])
        assert obs[LANGUAGE_KEY][0] == lang[rt["ep"]]
        i = by_seq[rt["seq"]]
        assert reqs["k_obs"][i] == rt["k_obs"] and reqs["ep"][i] == rt["ep"]
        assert reqs["state43_sent"][i].tobytes() == sent.tobytes()
        assert np.array_equal(reqs["obs_sha1"][i], obs_sha1(obs))
    k_obs = sorted({wire.loads(r)["data"]["options"]["rt"]["k_obs"] for r in schema_run["raw"]})
    with capsys.disabled():
        print(f"\n[obs schema] {len(schema_run['raw'])} requests: state == np.float32(env obs) and "
              f"ego frame == env frame at k_obs (k_obs in {k_obs[:4]}...{k_obs[-1]})")


@pytest.mark.sim
@pytest.mark.rt
def test_wrapper_rejects_mutated_requests(schema_run):
    from stub_policy_server import make_wrapper

    wrapper = make_wrapper()
    data = _decode_gr00t(schema_run["raw"][1])["data"]
    action, info = wrapper.get_action(**data)  # exactly what PolicyServer calls
    assert set(action) == {f"action.{g}" for g in ("left_arm", "right_arm", "left_hand", "right_hand",
                                                   "waist", "base_height_command", "navigate_command")}

    def mutated(key, value):
        obs = dict(data["observation"])
        obs[key] = value
        return {"observation": obs, "options": data["options"]}

    obs = data["observation"]
    bad = {
        "state float64": mutated("state.left_arm", obs["state.left_arm"].astype(np.float64)),
        "state without time axis": mutated("state.waist", obs["state.waist"][:, 0]),
        "video float32": mutated(VIDEO_KEY, obs[VIDEO_KEY].astype(np.float32)),
        "video without batch axis": mutated(VIDEO_KEY, obs[VIDEO_KEY][0]),
        "language as bare str": mutated(LANGUAGE_KEY, obs[LANGUAGE_KEY][0]),
    }
    accepted = []
    for name, d in bad.items():
        try:
            wrapper.get_action(**d)
            accepted.append(name)
        except AssertionError:
            pass
    assert not accepted, f"the strict wrapper accepted: {accepted}"
