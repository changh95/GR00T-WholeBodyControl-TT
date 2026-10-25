"""T-1.6 faults: the RT loop survives a stalled VLA, a dead WBC server, a dead VLA server and an arm jump.

Four RT runs (``--wbc ort`` unless stated, mock VLA `detfn` with 60 +- 5 ms latency on random free
ports, short episodes) run concurrently, each under tests/guard_probe.py (the loop CLI in-process;
``--trace-splicer`` records the STARVE_FAULT flag, which the TickLog does not carry). The test kills
and restarts only servers it started itself (SIGKILL through their Popen objects):

  stall     the mock's 14th get_action (seq 14, k_obs 208) sleeps 2 s (--stall-after 13 --stall-s 2)
  wbc_kill  --wbc on x280d serve --backend sim (the W-WBC server of demo.sh --x280-sim, a random port);
            SIGKILL at about tick 150 of episode 0, restarted on the same port at once; 3 episodes
  vla_kill  SIGKILL of the mock at about tick 100 of episode 0, restarted on the same port 2.5 s later
  jump      tests/fault_mock.py answers every episode-0 request with k_obs >= 100 with arm rows =
            state + 1.5 rad: the arm PD turns the step into a joint speed above the safety monitor's
            6 rad/s, whose sys.exit(1) the loop must catch (sync_env.py:516-528)

Pass rules (PLAN.md §2 M1, T-1.6): stall: `starved` set at row 30 of the last chunk, nav ramps to 0
in 5 ticks and |nav| < 0.05 (Balance) from then on, the STARVE_FAULT flag is raised on the 50th
consecutive starved tick, the episode continues, the next arrival ends the starvation, the process
stays alive, 0 slips. WBC killed: the first missed call is answered by ORT on the same tick (no tick
lost, no overrun), the episode is latched to ORT after 3 misses, and the next episode reconnects to
the restarted server. VLA killed: every abandoned request gives up 500-550 ms after its send (the
first tick poll after 500 ms), the socket is re-created and the request re-sent on the same tick,
the loop starves instead of hanging, and recovers when the server is back. Arm jump: SystemExit
caught -> SAFETY_FAULT, the next episode runs.

`--no-affinity`: other agents run RT loops pinned to the tick CPU concurrently; four more spinning
tick threads on the same core would make every run slip. Pinning is not what this test checks.
"""

import json
import threading
import time

import numpy as np
import pytest

from fault_utils import (ProbeRun, WbcServer, free_port, kill9, runs_of, spawn_fault_mock,
                         spawn_mock)
from rtloop.schema import ARMS, NAV, STATE_SLICES, WALK_THRESHOLD, Outcome, WbcBackend

LAT = ("--latency-ms", "60", "--jitter-ms", "5")
RT = ("--mode", "rt", "--no-affinity")
STALL_AFTER = 13  # 0-based get_action index: the pre-roll is 0, then one request every 16 ticks
MAX_TICKS = {"stall": 450, "vla_kill": 450, "wbc_kill": 400, "jump": 300}
VLA_KILL_AT, VLA_DEAD_S = 100, 2.5
WBC_KILL_AT = 150
JUMP_K, JUMP_RAD = 100, 1.5
ARM_VELOCITY_LIMIT = 6.0  # JointSafetyMonitor.ARM_VELOCITY_LIMIT (rad/s)
ARM_STATE = np.r_[STATE_SLICES["left_arm"], STATE_SLICES["right_arm"]]
GIVE_UP_MS = (500.0, 550.0)
WBC_TIMEOUT_US = 5000.0
TIMEOUT_S = 400
STARVE_FAULT_TICKS = 50
STARVE_RAMP = 5


def _wait_ready_bg(server: WbcServer, ev: dict) -> threading.Thread:
    def run():
        try:
            server.wait_ready(120)
            ev["wbc_ready_ns"] = time.monotonic_ns()
        except Exception as e:  # noqa: BLE001 - reported by the test
            ev["wbc_ready_error"] = repr(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


@pytest.fixture(scope="module")
def faults(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("faults")
    procs, wbcs, runs, ev = [], [], {}, {}
    try:
        stall_mock, stall_url = spawn_mock("detfn", *LAT, "--seed", "1", "--stall-after",
                                           str(STALL_AFTER), "--stall-s", "2")
        jump_mock, jump_url = spawn_fault_mock("detfn", *LAT, "--seed", "2", "--jump-ep", "0",
                                               "--jump-k", str(JUMP_K), "--jump-rad", str(JUMP_RAD))
        vla_mock, vla_url = spawn_mock("detfn", *LAT, "--seed", "3")
        wbc_vla, wbc_vla_url = spawn_mock("detfn", *LAT, "--seed", "4")
        procs += [stall_mock, jump_mock, vla_mock, wbc_vla]
        wbc = WbcServer(free_port(), tmp / "wbc_first.log")
        wbcs.append(wbc)
        wbc.wait_ready(120)

        def loop(name, vla, wbc_url="ort", seeds="0-1"):
            return [*RT, "--vla", vla, "--wbc", wbc_url, "--seeds", seeds,
                    "--max-ticks", str(MAX_TICKS[name])]

        runs["stall"] = ProbeRun(tmp, "stall", ["--trace-splicer"], loop("stall", stall_url))
        runs["jump"] = ProbeRun(tmp, "jump", [], loop("jump", jump_url))
        runs["vla_kill"] = ProbeRun(tmp, "vla_kill", ["--trace-splicer"], loop("vla_kill", vla_url))
        runs["wbc_kill"] = ProbeRun(tmp, "wbc_kill", [], loop("wbc_kill", wbc_vla_url, wbc.url, "0-2"))

        # Kill and restart the servers of vla_kill and wbc_kill while their episode 0 runs.
        vla_port = int(vla_url.rsplit(":", 1)[1])
        vla_state = wbc_state = 0
        ready_thread = None
        t_end = time.monotonic() + 300
        while vla_state < 2 or wbc_state < 2:
            assert time.monotonic() < t_end, f"fault orchestration timed out: {ev}"
            for name in ("vla_kill", "wbc_kill"):
                r = runs[name]
                assert r.proc.poll() is None, f"{name} exited early ({r.proc.returncode})\n{r.tail()}"
            if vla_state == 0 and runs["vla_kill"].ticks() >= VLA_KILL_AT:
                ev["vla_kill_ns"] = time.monotonic_ns()
                kill9(vla_mock)
                ev["vla_dead_ns"] = time.monotonic_ns()
                vla_state = 1
            elif vla_state == 1 and time.monotonic_ns() - ev["vla_dead_ns"] >= VLA_DEAD_S * 1e9:
                p, _ = spawn_mock("detfn", *LAT, "--seed", "5", port=vla_port)
                procs.append(p)
                ev["vla_restart_ns"] = time.monotonic_ns()
                vla_state = 2
            if wbc_state == 0 and runs["wbc_kill"].ticks() >= WBC_KILL_AT:
                ev["wbc_kill_ns"] = time.monotonic_ns()
                wbc.kill9()
                ev["wbc_dead_ns"] = time.monotonic_ns()
                wbc2 = WbcServer(wbc.port, tmp / "wbc_restart.log")
                wbcs.append(wbc2)
                ready_thread = _wait_ready_bg(wbc2, ev)
                wbc_state = 1
            elif wbc_state == 1 and not ready_thread.is_alive():
                assert "wbc_ready_ns" in ev, ev.get("wbc_ready_error")
                wbc_state = 2
            time.sleep(0.02)

        out = {}
        for name, r in runs.items():
            run_dir = r.finish(TIMEOUT_S)
            out[name] = {"run": r, "dir": run_dir, "data": r.load(),
                         "summary": json.loads((run_dir / "summary.json").read_text())}
        out["ev"] = ev
        yield out
    finally:
        for r in runs.values():
            r.kill()
        for p in procs:
            kill9(p)
        for w in wbcs:
            w.stop()


def _ep(data, ep):
    t, q, e = data
    return t[t["ep"] == ep], q[q["ep"] == ep], e[ep]


def _rtt_ms(q):
    return (q["t_recv_ns"].astype(np.int64) - q["t_send_ns"].astype(np.int64)) / 1e6


def _check_give_ups(q_all, run_name):
    """Every TIMEOUT left 500-550 ms after its send and was re-sent on the same tick."""
    to = q_all[q_all["outcome"] == Outcome.TIMEOUT]
    rtt = _rtt_ms(to)
    assert ((rtt >= GIVE_UP_MS[0]) & (rtt <= GIVE_UP_MS[1])).all(), (run_name, rtt)
    nxt = {}
    for i in range(len(q_all) - 1):
        nxt[int(q_all["seq"][i])] = q_all[i + 1]
    for r in to:
        n = nxt.get(int(r["seq"]))
        if n is None or n["ep"] != r["ep"]:
            continue
        gap_ms = (int(n["t_send_ns"]) - int(r["t_recv_ns"])) / 1e6
        assert 0 <= gap_ms < 5, (run_name, int(r["seq"]), gap_ms)  # step 2 gives up, step 3 re-sends
    return to, rtt


def _starve_fault_tick(trace, ep):
    """(first tick with the flag, starve_n there, flag latched to the end of the episode)."""
    tr = trace[trace[:, 0] == ep]
    on = np.nonzero(tr[:, 5])[0]
    if on.size == 0:
        return None, None, False
    return int(tr[on[0], 1]), int(tr[on[0], 4]), bool(tr[on[0]:, 5].all())


# --- VLA stall 2 s --------------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.rt
def test_stall_starves_at_row_30_and_ramps_nav(faults, capsys):
    t0, q0, e0 = _ep(faults["stall"]["data"], 0)
    starved = runs_of(t0["starved"])
    assert len(starved) == 1, starved
    ks, n = starved[0]
    assert np.array_equal(t0["k"], np.arange(len(t0)))
    # The chunk being executed when the stalled request was due: starvation begins at its row 30.
    assert t0["row"][ks] == 30 and t0["starved"][ks - 1] == 0 and t0["row"][ks - 1] == 29
    assert t0["chunk_seq"][ks] == t0["chunk_seq"][ks - 1]
    nav0 = t0["cmd35"][ks - 1][NAV]
    assert np.linalg.norm(nav0) > WALK_THRESHOLD, "the stall should hit a walking window (detfn nav 0.2)"
    for j in range(STARVE_RAMP):  # linear ramp to 0 over 5 ticks; arms, hands, waist, height held
        want = nav0 * np.float32(1 - (j + 1) / STARVE_RAMP)
        assert np.allclose(t0["cmd35"][ks + j][NAV], want, atol=1e-6), (j, t0["cmd35"][ks + j][NAV])
        keep = np.r_[0:32]
        assert np.array_equal(t0["cmd35"][ks + j][keep], t0["cmd35"][ks - 1][keep])
    late = slice(ks + STARVE_RAMP - 1, ks + n)
    assert (np.linalg.norm(t0["nav_wbc"][late], axis=1) < WALK_THRESHOLD).all()
    assert (t0["policy"][late] == 0).all(), "not in Balance within 5 starved ticks"
    first_balance = ks + int(np.argmax(t0["policy"][ks:] == 0))
    with capsys.disabled():
        print(f"\n[T-1.6 stall] starved from k={ks} (row 30 of chunk {int(t0['chunk_seq'][ks])}) for {n} "
              f"ticks; nav {nav0[0]:.2f} -> 0 by k={ks + STARVE_RAMP - 1}; Balance from k={first_balance}")


@pytest.mark.sim
@pytest.mark.rt
def test_stall_starve_fault_at_50_and_episode_continues(faults, capsys):
    d = faults["stall"]
    t0, q0, e0 = _ep(d["data"], 0)
    ks, n = runs_of(t0["starved"])[0]
    assert n >= STARVE_FAULT_TICKS + 10, n
    k_flag, starve_n, latched = _starve_fault_tick(d["run"].splicer_trace(), 0)
    assert k_flag == ks + STARVE_FAULT_TICKS - 1 and starve_n == STARVE_FAULT_TICKS, (k_flag, ks, starve_n)
    assert latched, "STARVE_FAULT must stay set for the rest of the episode"
    assert e0["starve_fault"] is True and e0["end"] == "TRUNC" and e0["n_ticks"] == MAX_TICKS["stall"]
    assert e0["k_end"] > k_flag + 100  # the episode kept running long after the flag
    t1, q1, e1 = _ep(d["data"], 1)
    assert e1["starve_fault"] is False and e1["n_starved"] == 0 and e1["end"] in ("TRUNC", "SUCCESS")
    with capsys.disabled():
        print(f"\n[T-1.6 stall] STARVE_FAULT raised at k={k_flag} (starved tick #{starve_n}); "
              f"episode ran on to k={e0['k_end']} ({e0['end']}); next episode clean")


@pytest.mark.sim
@pytest.mark.rt
def test_stall_recovers_on_next_arrival_no_slips(faults, capsys):
    d = faults["stall"]
    t, q, eps = d["data"]
    t0, q0, e0 = _ep(d["data"], 0)
    ks, n = runs_of(t0["starved"])[0]
    stalled = q0[q0["seq"] == STALL_AFTER + 1][0]
    assert stalled["outcome"] == Outcome.TIMEOUT and stalled["k_obs"] == 16 * STALL_AFTER
    to, rtt = _check_give_ups(q, "stall")
    applied = q0[(q0["outcome"] == Outcome.APPLIED) & (q0["k_arr"] > ks)]
    first = applied[np.argmin(applied["k_arr"])]
    k_rec = int(first["k_arr"])
    assert k_rec == ks + n, (k_rec, ks, n)  # the next arrival ends the starvation
    assert t0["splice"][k_rec] == 1 and t0["chunk_seq"][k_rec] == first["seq"]
    assert t0["row"][k_rec] == first["d"] and t0["starved"][k_rec:].sum() == 0
    assert d["run"].rc == 0 and int(t["slip"].sum()) == 0
    with capsys.disabled():
        print(f"\n[T-1.6 stall] {len(to)} give-ups at {rtt.min():.1f}..{rtt.max():.1f} ms; recovered "
              f"at k={k_rec} (seq {int(first['seq'])}, d={int(first['d'])}); process exit 0; "
              f"slips 0 in {len(t)} ticks; late max {t['late_us'].max():.1f} us")


# --- WBC server killed --------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.rt
def test_wbc_killed_ort_on_the_same_tick_then_latched(faults, capsys):
    d, ev = faults["wbc_kill"], faults["ev"]
    t0, _, e0 = _ep(d["data"], 0)
    assert np.array_equal(t0["k"], np.arange(len(t0))) and e0["n_ticks"] == MAX_TICKS["wbc_kill"]
    assert np.isfinite(t0["q15"]).all() and np.isfinite(t0["wbc_rtt_us"]).all()  # a WBC action every tick
    before = t0["t_end_ns"] < ev["wbc_kill_ns"]
    after = t0["t_start_ns"] > ev["wbc_dead_ns"]
    assert before.sum() > 100 and after.sum() > 100
    assert (t0["wbc_be"][before] == WbcBackend.X280_SIM).all() and (t0["wbc_fb"][before] == 0).all()
    assert (t0["wbc_fb"][after] == 1).all() and (t0["wbc_be"][after] == WbcBackend.ORT).all()
    fb = np.nonzero(t0["wbc_fb"])[0]
    k_miss = int(fb[0])
    assert np.array_equal(fb, np.arange(k_miss, len(t0))), "fallback must persist (latched) to episode end"
    misses = t0["wbc_rtt_us"][fb[:3]]
    assert ((misses >= WBC_TIMEOUT_US) & (misses < WBC_TIMEOUT_US + 1500)).all(), misses
    # Latched: ORT only, the server is not asked again this episode (no tick waits for the timeout).
    assert (t0["wbc_rtt_us"][fb[3:]] < WBC_TIMEOUT_US).all(), "after 3 misses the episode is latched to ORT"
    latched_p50_us = float(np.median(t0["wbc_rtt_us"][fb[3:]]))
    assert (t0["shadow_da"][fb] == 0.0).all(), "the fallback action is ORT's (the ORT shadow agrees exactly)"
    assert (t0["compute_us"][fb[:3]] < 20000).all() and t0["slip"].sum() == 0
    w = d["summary"]["wbc"]
    assert w["n_timeouts"] + w["n_errors"] >= 3 and w["n_latches"] == 1
    with capsys.disabled():
        print(f"\n[T-1.6 wbc] server SIGKILLed during k={k_miss}: ORT on that tick (rtt "
              f"{misses[0] / 1e3:.2f} ms incl. the 5 ms timeout, compute {t0['compute_us'][k_miss] / 1e3:.2f} ms), "
              f"3 misses {np.round(misses / 1e3, 2).tolist()} ms then latched ({len(fb)} ORT ticks, "
              f"rtt p50 {latched_p50_us:.0f} us)")


@pytest.mark.sim
@pytest.mark.rt
def test_wbc_reconnects_after_restart(faults, capsys):
    d, ev = faults["wbc_kill"], faults["ev"]
    t, _, eps = d["data"]
    assert len(eps) == 3 and d["run"].rc == 0
    later = eps[1:]
    for e in later:
        te = t[t["ep"] == e["ep"]]
        assert int(te["t_start_ns"][0]) > ev["wbc_ready_ns"], "episode started before the restart was ready"
        assert e["wbc_reset_ping"] is True and e["n_fallback"] == 0
        assert (te["wbc_be"] == WbcBackend.X280_SIM).all() and (te["wbc_fb"] == 0).all()
        assert float(np.nanmax(te["shadow_da"])) < 1e-2  # the restarted server serves the same MLPs (fp16 bound)
    assert int(t["slip"].sum()) == 0
    restart_s = (ev["wbc_ready_ns"] - ev["wbc_dead_ns"]) / 1e9
    with capsys.disabled():
        print(f"\n[T-1.6 wbc] restarted in {restart_s:.2f} s; episodes "
              f"{[e['ep'] for e in later]} reconnected (reset ping ok, 0 fallbacks, x280d sim on every tick)")


# --- VLA server killed --------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.rt
def test_vla_killed_gives_up_at_500ms_and_starves(faults, capsys):
    d, ev = faults["vla_kill"], faults["ev"]
    t, q, eps = d["data"]
    t0, q0, e0 = _ep(d["data"], 0)
    to, rtt = _check_give_ups(q, "vla_kill")
    assert len(to) >= 3 and (to["ep"] == 0).all()
    # Only requests that were out while the server was dead were abandoned.
    assert (to["t_send_ns"] > ev["vla_kill_ns"] - 100_000_000).all()
    assert (to["t_send_ns"] < ev["vla_restart_ns"]).all()
    stats = d["summary"]["vla_stats"]
    assert stats["timeout"] == len(to) and stats["recreate"] >= len(to)  # lazy pirate per give-up
    starved = runs_of(t0["starved"])
    assert len(starved) == 1, starved
    ks, n = starved[0]
    assert t0["row"][ks] == 30 and t0["starved"][ks - 1] == 0
    assert int(t0["t_start_ns"][ks]) > ev["vla_kill_ns"]
    assert (np.linalg.norm(t0["nav_wbc"][ks + STARVE_RAMP - 1: ks + n], axis=1) < WALK_THRESHOLD).all()
    assert (t0["policy"][ks + STARVE_RAMP - 1: ks + n] == 0).all()
    k_flag, starve_n, latched = _starve_fault_tick(d["run"].splicer_trace(), 0)
    assert n >= STARVE_FAULT_TICKS and k_flag == ks + STARVE_FAULT_TICKS - 1 and latched
    assert e0["starve_fault"] is True and e0["end"] == "TRUNC" and e0["n_timeouts"] == len(to)
    with capsys.disabled():
        print(f"\n[T-1.6 vla] server SIGKILLed at k~{int(np.argmax(t0['t_start_ns'] > ev['vla_kill_ns']))}: "
              f"{len(to)} give-ups at {rtt.min():.1f}..{rtt.max():.1f} ms after send, each re-sent on the "
              f"same tick on a new socket; starved from k={ks} (row 30) for {n} ticks, STARVE_FAULT at k={k_flag}")


@pytest.mark.sim
@pytest.mark.rt
def test_vla_killed_no_hang_and_recovers(faults, capsys):
    d, ev = faults["vla_kill"], faults["ev"]
    t, q, eps = d["data"]
    t0, q0, e0 = _ep(d["data"], 0)
    ks, n = runs_of(t0["starved"])[0]
    applied = q0[(q0["outcome"] == Outcome.APPLIED) & (q0["k_arr"] > ks)]
    first = applied[np.argmin(applied["k_arr"])]
    assert int(first["t_recv_ns"]) > ev["vla_restart_ns"]
    assert int(first["k_arr"]) == ks + n and t0["starved"][ks + n:].sum() == 0
    assert d["run"].rc == 0 and len(eps) == 2 and int(t["slip"].sum()) == 0
    t1, q1, e1 = _ep(d["data"], 1)
    assert e1["n_timeouts"] == 0 and e1["n_starved"] == 0 and e1["end"] in ("TRUNC", "SUCCESS")
    # Ticks kept coming at 50 Hz through the outage (no tick waited on the dead server; the m1
    # period_max gate is 30 ms, a blocking wait would be >= 500 ms).
    per = np.diff(t0["t_start_ns"].astype(np.int64)) / 1e6
    assert per.max() < 30.0, per.max()
    dead_s = (ev["vla_restart_ns"] - ev["vla_dead_ns"]) / 1e9
    with capsys.disabled():
        print(f"\n[T-1.6 vla] dead {dead_s:.2f} s; recovered at k={int(first['k_arr'])} (seq "
              f"{int(first['seq'])}, d={int(first['d'])}); tick period max {per.max():.3f} ms through the "
              f"outage; 2 episodes, exit 0, drain {[e['drain'] for e in eps]}")


# --- Arm jump -> safety monitor -----------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.rt
def test_arm_jump_safety_fault_then_next_episode(faults, capsys):
    d = faults["jump"]
    t, q, eps = d["data"]
    assert len(eps) == 2 and d["run"].rc == 0
    t0, q0, e0 = _ep(d["data"], 0)
    assert e0["end"] == "SAFETY_FAULT"
    k_end = e0["k_end"]
    assert np.nonzero(t0["safety"])[0].tolist() == [len(t0) - 1] and int(t0["k"][-1]) == k_end
    jump = q0[q0["k_obs"] >= JUMP_K]
    j = jump[np.argmin(jump["k_obs"])]
    assert j["outcome"] == Outcome.APPLIED and JUMP_K < j["k_arr"] <= k_end
    k_arr = int(j["k_arr"])
    want = j["state43_sent"][ARM_STATE] + np.float32(JUMP_RAD)  # what the fault mock sent, in f32
    assert np.array_equal(t0["cmd35"][k_arr][ARMS], want), "the executed arm command is the injected jump"
    assert (t0["dq_arm_absmax"][:k_arr] < ARM_VELOCITY_LIMIT).all()
    assert t0["dq_arm_absmax"][-1] > ARM_VELOCITY_LIMIT  # what the monitor saw at the faulting tick
    assert "[SAFETY] Initiating system shutdown" in d["run"].log_path.read_text(errors="replace")
    t1, q1, e1 = _ep(d["data"], 1)
    assert e1["end"] in ("TRUNC", "SUCCESS") and t1["safety"].sum() == 0
    assert e1["n_ticks"] == MAX_TICKS["jump"] or e1["end"] == "SUCCESS"
    assert e1["gen"] == e0["gen"] + 1
    with capsys.disabled():
        print(f"\n[T-1.6 jump] +{JUMP_RAD} rad arm chunk applied at k={k_arr}; |dq_arm| "
              f"{t0['dq_arm_absmax'][-1]:.2f} rad/s > {ARM_VELOCITY_LIMIT} at k={k_end}: SystemExit caught "
              f"-> SAFETY_FAULT; episode 1 ran {e1['n_ticks']} ticks ({e1['end']}); exit 0")
