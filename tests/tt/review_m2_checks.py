"""Adversarial review of rttt/m2_checks.py (T-2.8 VLA p50 drift), on synthetic runs (no sim, no device).

    simpy -m pytest -p no:cacheprovider $RT/tests/tt/review_m2_checks.py -q

The loop stamps a VLA reply's t_recv_ns at the tick poll that finds it (rtloop/vla_client.py poll(): once per tick),
so the client RTT is a multiple of 20 ms minus the send offset (scripts/analyze_run.py docstring: "replies are polled
once per tick, so it is quantised to 20 ms"). The server's own service time is in requests.jsonl as
srv_t_recv_ns / srv_t_reply_ns, which analyze_run.py gates as vla_p50_ms. T-2.8 asks for "VLA p50 drift < 5%".
"""

import json
import math
from pathlib import Path

from rtloop import telemetry
from rtloop.schema import Outcome
from rttt import m2_checks as M

TICK = 20_000_000
SEND_OFFSET = 300_000  # the loop sends ~0.3 ms after T_k
WIRE = 600_000  # request + reply transfer, ~0.6 ms


def quantised_run(root: Path, span_first_ms: float, span_last_ms: float, minutes: float = 30.0) -> Path:
    """A T-2.6/T-2.8-shaped RT run: one request every 16 ticks; server span span_first_ms for the first half,
    span_last_ms after; the client sees the reply at the first tick start after it arrived."""
    run = telemetry.RunDir(root, tag="t28", config={"args": {"viewer": False}})
    rl = telemetry.RequestLog(run)
    t0 = 10**12
    n_req = int(minutes * 60 * 50 / 16)
    for i in range(n_req):
        span = span_first_ms if i < n_req // 2 else span_last_ms
        k_obs = 16 * i
        t_tick = t0 + k_obs * TICK
        send = t_tick + SEND_OFFSET
        srv_recv = send + WIRE // 2
        srv_reply = srv_recv + int(span * 1e6)
        arrive = srv_reply + WIRE // 2
        t_recv = t0 + math.ceil((arrive - t0) / TICK) * TICK + 50_000  # the poll at the next tick start
        rl.add({"seq": i + 1, "ep": 0, "k_obs": k_obs + 1, "k_arr": (t_recv - t0) // TICK, "d": 3,
                "t_send_ns": send, "t_recv_ns": t_recv, "outcome": Outcome.APPLIED,
                "srv_t_recv_ns": srv_recv, "srv_t_reply_ns": srv_reply})
    rl.close()
    (run.path / "verdict.json").write_text(json.dumps({"metrics": {"starved_frac": 0.0, "starved_run_max": 0,
                                                                   "outcome_counts": {"APPLIED": n_req}}}))
    return run.path


def drift_check(run: Path) -> dict:
    res = M.soak_checks(run, 1800)
    return next(c for c in res["checks"] if c["name"] == "vla_p50_drift")


def test_t28_flags_a_real_10pct_server_slowdown(tmp_path):
    """Server p50 42 -> 46.5 ms (+10.7%): T-2.8 must fail. The quantised client RTT stays one value (59.75 ms)."""
    c = drift_check(quantised_run(tmp_path, 42.0, 46.5))
    assert not c["pass"], f"T-2.8 drift check passes a +10.7% server slowdown: measured drift {c['value']}"


def test_t28_does_not_flag_a_4pct_change(tmp_path):
    """Server p50 40.0 -> 38.5 ms (-3.75%) crosses the 40 ms tick boundary (reply at 40.9 vs 39.4 ms after T_k):
    T-2.8 must pass. The client RTT jumps from 59.75 to 39.75 ms (-33%)."""
    c = drift_check(quantised_run(tmp_path, 40.0, 38.5))
    assert c["pass"], f"T-2.8 drift check fails a -3.75% server change: measured drift {c['value']}"
