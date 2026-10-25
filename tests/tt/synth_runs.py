"""Synthetic run dirs in the loop's on-disk format (rtloop.telemetry), for the M2 checks and the S1b job tests.

No sim and no server: the TickLog / RequestLog / EpisodeLog an RT run on TT would write, built tick by tick, so
scripts/analyze_run.py and rttt/m2_checks.py read them exactly as a real run. Used by tests/tt/test_m2_checks.py and,
as a stand-in for rt_sim behind demo.sh, by tests/tt/soak_stub_loop.py.
"""

from pathlib import Path

from rtloop import telemetry
from rtloop.schema import Outcome, WbcBackend

TT_BACKEND = "tt:p150a:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate-bf16@29c1f369"  # server_g1's info.backend


def synth_run(root: Path, minutes: float = 2.0, starve_run: int = 0, starve_every: int = 0, error: bool = False,
              rtt_ms=(45.0, 45.0), viewer: bool = False, tag: str = "t26", backend: str = TT_BACKEND,
              vla: str = "tcp://127.0.0.1:5557", config: dict | None = None) -> Path:
    """A clean RT run on TT: 20 ms ticks, GPU WBC 0.6 ms, a request every 16 ticks with d = 3 and server stamps."""
    cfg = {"args": {"viewer": viewer, "wbc": "tcp://127.0.0.1:5556", "vla": vla}, "vla_info": {"backend": backend},
           **(config or {})}
    run = telemetry.RunDir(root, tag=tag, config=cfg)
    tl, rl, el = telemetry.TickLog(run), telemetry.RequestLog(run), telemetry.EpisodeLog(run)
    n_ticks = int(minutes * 60 * 50)
    t0 = 1_000_000_000_000
    ep_len = 1440
    seq = 0
    for k_all in range(n_ticks):
        ep, k = divmod(k_all, ep_len)
        r = tl.next()
        r["ep"], r["k"] = ep, k
        r["t_start_ns"] = t0 + k_all * 20_000_000 + ep * 3_000_000_000
        r["period_us"], r["compute_us"], r["wbc_rtt_us"], r["wbc_be"] = 20000.0, 6000.0, 600.0, WbcBackend.GPU
        r["viewer_us"] = 1500.0 if viewer else 0.0
        r["row"], r["age"] = k % 16, k % 16
        starved = (bool(starve_run and 100 <= k_all < 100 + starve_run)
                   or bool(starve_every and k_all % starve_every < 2))
        r["starved"] = 1 if starved else 0
        tl.commit()
        if k % 16 == 0:
            seq += 1
            frac = k_all / max(n_ticks - 1, 1)
            rtt = rtt_ms[0] + (rtt_ms[1] - rtt_ms[0]) * frac
            send = int(r["t_start_ns"]) + 300_000
            out = Outcome.ERROR if (error and seq == 5) else Outcome.APPLIED
            rl.add({"seq": seq, "ep": ep, "k_obs": k, "k_arr": k + 3, "d": 3, "t_send_ns": send,
                    "t_recv_ns": send + int(rtt * 1e6) + 2_000_000, "outcome": out,
                    "srv_t_recv_ns": send + 500_000, "srv_t_reply_ns": send + 500_000 + int(rtt * 1e6)})
        if k == ep_len - 1 or k_all == n_ticks - 1:
            el.add({"ep": ep, "seed": 3000 + ep, "end": "TRUNC"})
    tl.close()
    rl.close()
    el.close()
    return run.path
