"""rttt/m2_checks.py, gates.yaml's m2_rt and the M2 eval matrices, on synthetic runs (no sim, no device).

    simpy -m pytest $RT/tests/tt/test_m2_checks.py -q
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from rtloop import telemetry
from rtloop.schema import Outcome
from rttt import m2_checks as M

from .synth_runs import TT_BACKEND, synth_run

RT = Path(__file__).resolve().parents[2]
GATES = RT / "gates.yaml"  # m2_rt (merged from configs/gates_m2.proposed.yaml, which is superseded)
PROPOSED = RT / "configs" / "gates_m2.proposed.yaml"


def _analyze_run():
    spec = importlib.util.spec_from_file_location("analyze_run_m2", RT / "scripts" / "analyze_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def no_viewer_gates(tmp_path: Path) -> Path:
    out = tmp_path / "gates_no_viewer.yaml"
    assert M.main(["gates", str(GATES), str(out), "", "--no-viewer"]) == 0
    return out


def test_gates_yaml_m2_rt_has_the_t26_rows_and_the_proposal_is_superseded():
    ar = _analyze_run()
    gates = ar.load_gates(GATES, "m2_rt")  # raises on an unknown key
    assert gates["d_p99_max"] == 3 and gates["vla_p50_ms_max"] == pytest.approx(49.5)
    assert gates["vla_timeouts_max"] == 0 and gates["vla_errors_max"] == 0  # T-2.6/T-2.8: 0 TT errors, 0 hangs
    assert gates["starved_frac_max"] == pytest.approx(0.001) and gates["starved_run_max"] == 10
    assert "starved_ticks_max" not in gates  # M2 allows 0.1%, not m1_rt's 0
    assert PROPOSED.read_text().startswith("# SUPERSEDED")
    for job in (RT / "jobs").glob("*.sh"):  # no job reads or copies the proposal any more
        assert "gates_m2.proposed" not in job.read_text(), job
    m1 = ar.load_gates(RT / "gates.yaml", "m1_rt")
    for k in ("period_p99_ms", "period_max_ms", "overrun_frac_max", "slips_max", "wbc_rtt_p99_ms", "safety_faults_max",
              "inflight_max", "viewer_k0_ms_max", "wbc_fallback_frac_max", "shadow_da_p99_max"):
        assert gates[k] == m1[k], k  # the T-1.4 timing rows carry over unchanged


def test_derive_gates_from_bench(tmp_path):
    bench = tmp_path / "t23_bench.json"
    bench.write_text(json.dumps({"latency": {"rtt_ms": {"p50": 47.0}, "server_total_ms": {"p50": 43.0},
                                             "server_span_ms": {"p50": 41.0}}}))
    out = tmp_path / "g.yaml"
    assert M.main(["gates", str(GATES), str(out), str(bench)]) == 0
    assert out.read_text().startswith(f"# m2_rt of one session (rttt/m2_checks.py gates) from {GATES}")
    assert list(yaml.safe_load(out.read_text())) == ["m2_rt"]
    g = yaml.safe_load(out.read_text())["m2_rt"]
    assert g["vla_p50_ms_max"] == pytest.approx(45.1) and "viewer_k0_ms_max" in g  # the server span, not the RTT
    assert g["starved_run_max"] == 10 and g["vla_errors_max"] == 0  # the rest is gates.yaml's set, unchanged
    bench.write_text(json.dumps({"latency": {"rtt_ms": {"p50": 47.0}, "server_total_ms": {"p50": 43.0}}}))
    assert M.main(["gates", str(GATES), str(out), str(bench), "--no-viewer"]) == 0
    g = yaml.safe_load(out.read_text())["m2_rt"]
    assert g["vla_p50_ms_max"] == pytest.approx(47.3) and "viewer_k0_ms_max" not in g
    assert M.main(["gates", str(GATES), str(out), ""]) == 0
    assert yaml.safe_load(out.read_text())["m2_rt"]["vla_p50_ms_max"] == pytest.approx(49.5)
    _analyze_run().load_gates(out, "m2_rt")


@pytest.mark.parametrize("viewer", [False, True])
def test_clean_soak_passes_gates_and_checks(tmp_path, viewer):
    run = synth_run(tmp_path, viewer=viewer)
    gates = GATES if viewer else no_viewer_gates(tmp_path)
    v = _analyze_run().analyze(run, "m2_rt", gates, plots=False)
    assert v["pass"], [c for c in v["checks"] if not c["pass"]]
    assert v["metrics"]["vla_p50_ms"] == pytest.approx(45.0)
    res = M.soak_checks(run, 120)  # starvation and errors are gates.yaml rows now; no duplicate checks here
    assert res["pass"] and [c["name"] for c in res["checks"]] == ["vla_on_tensix"]
    assert res["vla_placement"]["device"] == "tensix" and res["vla_placement"]["label"] == "VLA Tensix"


def test_viewer_gates_need_a_viewer_run(tmp_path):
    run = synth_run(tmp_path, viewer=False)
    v = _analyze_run().analyze(run, "m2_rt", GATES, plots=False)
    assert {c["name"] for c in v["checks"] if not c["pass"]} == {"viewer_k0_ms_max", "viewer_p99_ms_max"}


@pytest.mark.parametrize("kw,bad", [({"starve_run": 12, "minutes": 5}, {"starved_run_max"}),
                                    ({"starve_every": 100}, {"starved_frac_max"}),
                                    ({"error": True}, {"vla_errors_max"})])
def test_starvation_and_tt_errors_fail_the_gates_yaml_set(tmp_path, kw, bad):
    """What m2_checks soak used to check itself (T-2.6 starvation, 0 TT error replies) fails analyze_run's m2_rt."""
    run = synth_run(tmp_path, **kw)
    v = _analyze_run().analyze(run, "m2_rt", no_viewer_gates(tmp_path), plots=False)
    assert {c["name"] for c in v["checks"] if not c["pass"]} == bad
    assert M.soak_checks(run, 120)["pass"]  # and is not counted twice


@pytest.mark.parametrize("backend,device,ok", [(TT_BACKEND, "tensix", True),
                                               ("cuda:rtx5090:cloudwalk-research/x@2a5b61cc", "gpu", False),
                                               ("mock:server_g1-fake", "host", False)])
def test_soak_needs_the_hud_to_place_the_vla_on_tensix(tmp_path, backend, device, ok):
    """T-2.6 "HUD shows vla=tensix from info": scripts/hud.py classify on the run's VLA stamp."""
    res = M.soak_checks(synth_run(tmp_path, minutes=0.1, backend=backend), 6)
    assert res["vla_placement"]["device"] == device and res["pass"] is ok


def test_soak_drift(tmp_path):
    """T-2.8's drift is gated with --gate-drift; a 30 min T-2.6 run reports it (PLAN's T-2.6 row does not gate it)."""
    ok = synth_run(tmp_path / "a", minutes=21, rtt_ms=(45.0, 46.0))
    _analyze_run().analyze(ok, "m2_rt", no_viewer_gates(tmp_path), plots=False)
    res = M.soak_checks(ok, 21 * 60, gate_drift=True)
    assert res["pass"] and res["checks"][-1]["name"] == "vla_p50_drift" and res["checks"][-1]["gated"]
    bad = synth_run(tmp_path / "b", minutes=21, rtt_ms=(45.0, 52.0))
    _analyze_run().analyze(bad, "m2_rt", no_viewer_gates(tmp_path), plots=False)
    res = M.soak_checks(bad, 21 * 60, gate_drift=True)
    assert [c["name"] for c in res["checks"] if not c["pass"]] == ["vla_p50_drift"] and not res["pass"]
    res = M.soak_checks(bad, 21 * 60)
    assert res["pass"] and not res["checks"][-1]["pass"] and not res["checks"][-1]["gated"]


def quantised_run(root: Path, span_first_ms: float, span_last_ms: float, minutes: float = 21.0,
                   stamps: bool = True) -> Path:
    """An RT run as the loop logs it: the client stamps t_recv_ns at the tick poll that finds the reply, so the RTT is
    quantised to 20 ms, while the server's own span (srv_t_recv_ns -> srv_t_reply_ns) is exact."""
    import math

    run = telemetry.RunDir(root, tag="t28", config={"args": {"viewer": False}})
    rl = telemetry.RequestLog(run)
    t0, tick, n = 10**12, 20_000_000, int(minutes * 60 * 50 / 16)
    for i in range(n):
        span = span_first_ms if i < n // 2 else span_last_ms
        send = t0 + 16 * i * tick + 300_000
        srv_recv, srv_reply = send + 300_000, send + 300_000 + int(span * 1e6)
        t_recv = t0 + math.ceil((srv_reply + 300_000 - t0) / tick) * tick + 50_000
        rec = {"seq": i + 1, "ep": 0, "k_obs": 16 * i + 1, "d": 3, "t_send_ns": send, "t_recv_ns": t_recv,
               "outcome": Outcome.APPLIED}
        if stamps:
            rec.update(srv_t_recv_ns=srv_recv, srv_t_reply_ns=srv_reply)
        rl.add(rec)
    rl.close()
    (run.path / "verdict.json").write_text(json.dumps({"metrics": {"starved_frac": 0.0, "starved_run_max": 0,
                                                                   "outcome_counts": {"APPLIED": n}}}))
    return run.path


@pytest.mark.parametrize("first,last,ok", [(42.0, 46.5, False),   # +10.7%: the quantised RTT does not move
                                           (40.0, 38.5, True),    # -3.75% across the 40 ms boundary: RTT -33%
                                           (45.0, 46.0, True)])
def test_t28_drift_is_on_the_server_time(tmp_path, first, last, ok):
    res = M.soak_checks(quantised_run(tmp_path, first, last), 21 * 60)
    c = res["checks"][-1]
    assert c["name"] == "vla_p50_drift" and c["pass"] is ok, res["drift"]
    assert abs(res["drift"]["drift"] - abs(last / first - 1)) < 1e-9
    assert res["drift"]["p50_first_ms"] == first and res["drift"]["p50_last_ms"] == last


def test_t28_drift_without_server_timings_fails(tmp_path):
    res = M.soak_checks(quantised_run(tmp_path, 45.0, 45.0, stamps=False), 21 * 60)
    assert res["drift"]["drift"] is None and not res["checks"][-1]["pass"]


def test_soak_cli_writes_report(tmp_path):
    run = synth_run(tmp_path)
    _analyze_run().analyze(run, "m2_rt", no_viewer_gates(tmp_path), plots=False)
    out = tmp_path / "t26_soak.json"
    assert M.main(["soak", str(run), "120", str(out)]) == 0
    assert json.loads(out.read_text())["pass"] is True
    bad = synth_run(tmp_path / "b", minutes=21, rtt_ms=(45.0, 52.0))
    assert M.main(["soak", str(bad), str(21 * 60), str(out)]) == 0
    assert M.main(["soak", str(bad), str(21 * 60), str(out), "--gate-drift"]) == 1


def _cfg(d: Path, vla_info: dict, code: dict) -> None:
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({
        "vla_info": vla_info,
        "imported_dirty": [{"tree": "RT", "path": p, "sha256": h, "status": "??"} for p, h in code.items()]
                          + [{"tree": "R", "path": "gr00t/x.py", "sha256": "zz", "status": " M"}]}))


def test_reuse_check(tmp_path):
    gpu_info = {"backend": "cuda:rtx5090:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc", "proto": 2,
                "model": "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate", "revision": "2a5b61cc7660bdef",
                "noise": "cpu"}
    tt_info = {"backend": {"checkpoint": "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate-bf16@deadbeef",
                           "source_checkpoint": "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc"},
               "model": "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate-bf16", "revision": "deadbeef00"}
    code = {"rtloop/loop.py": "aa", "rtloop/splicer.py": "bb"}
    _cfg(tmp_path / "tt/runs/C4tt/2000-2009/r1", tt_info, code)
    _cfg(tmp_path / "gpu/r1", gpu_info, code)
    rep = M.reuse_check(tmp_path / "tt/runs", [tmp_path / "gpu"])
    assert rep["reusable"], rep
    _cfg(tmp_path / "gpu2/r1", gpu_info, dict(code, **{"rtloop/splicer.py": "cc"}))
    rep = M.reuse_check(tmp_path / "tt/runs", [tmp_path / "gpu2"])
    assert not rep["reusable"] and rep["code_diff"] == ["rtloop/splicer.py"]
    other = dict(gpu_info, backend="cuda:rtx5090:nvidia/GR00T-N1.6-G1@897d0313", model="nvidia/GR00T-N1.6-G1",
                 revision="897d0313aa")
    _cfg(tmp_path / "gpu3/r1", other, code)
    rep = M.reuse_check(tmp_path / "tt/runs", [tmp_path / "gpu3"])
    assert not rep["reusable"] and rep["why"] == "checkpoint differs"
    assert M.main(["reuse", str(tmp_path / "tt/runs"), str(tmp_path / "gpu"), str(tmp_path / "rc.json")]) == 0
    assert M.main(["reuse", str(tmp_path / "tt/runs"), str(tmp_path / "nowhere"), str(tmp_path / "rc.json")]) == 1
    rc = tmp_path / "rc.json"
    assert M.main(["reuse", str(tmp_path / "tt/runs"), str(tmp_path / "gpu2"), str(rc)]) == 1
    assert M.main(["reuse", str(tmp_path / "tt/runs"), str(tmp_path / "gpu2"), str(rc), "--force"]) == 0  # code only
    assert json.loads(rc.read_text())["accepted"] and json.loads(rc.read_text())["forced"]


@pytest.mark.parametrize("noise", [None, "cuda"])
def test_reuse_never_accepts_unpaired_noise(tmp_path, noise):
    """T-1.9's GPU arm (CUDA-seeded noise; its server reported no noise mode) is not a --crn partner of TT: not
    reusable, not even with --force (T2_FORCE_REUSE), and the GPU-only pre-flight ("-") refuses it too."""
    gpu_info = {"backend": "cuda:rtx5090:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc",
                "model": "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate", "revision": "2a5b61cc7660bdef"}
    if noise:
        gpu_info["noise"] = noise
    tt_info = {"backend": "tt:p150a:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate-bf16@29c1f369",
               "source": "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc7660bdef", "noise": "cpu"}
    code = {"rtloop/loop.py": "aa"}
    _cfg(tmp_path / "tt/runs/C4tt/2000-2009/r1", tt_info, code)
    _cfg(tmp_path / "gpu/r1", gpu_info, code)
    rep = M.reuse_check(tmp_path / "tt/runs", [tmp_path / "gpu"], force=True)
    assert rep["checkpoint_match"] and rep["code_match"] and not rep["noise_paired"]
    assert not rep["reusable"] and not rep["accepted"] and "noise" in rep["why"]
    out = tmp_path / "rc.json"
    assert M.main(["reuse", str(tmp_path / "tt/runs"), str(tmp_path / "gpu"), str(out), "--force"]) == 1
    assert M.main(["reuse", "-", str(tmp_path / "gpu"), str(out)]) == 1
    assert json.loads(out.read_text())["gpu_noise"] == [str(noise)]
    gpu_info["noise"] = "cpu"
    _cfg(tmp_path / "gpu_ok/r1", gpu_info, code)
    assert M.main(["reuse", "-", str(tmp_path / "gpu_ok"), str(out)]) == 0


def test_tt_only_matrix(tmp_path):
    out = tmp_path / "m.yaml"
    mx = M.tt_only_matrix(RT / "configs" / "eval_m2_primary.yaml", out)
    assert [a["name"] for a in mx["arms"]] == ["C4tt"] and mx["compare"] == []
    assert yaml.safe_load(out.read_text())["name"] == "m2_primary"


@pytest.mark.parametrize("matrix,arms,urls", [
    ("eval_m2_sync.yaml", ["C1tt", "C1gpu"], ["tcp://127.0.0.1:5557", "tcp://127.0.0.1:5565"]),
    ("eval_m2_primary.yaml", ["C4tt", "C4gpu"], ["tcp://127.0.0.1:5557", "tcp://127.0.0.1:5565"]),
])
def test_matrices_dry_run(tmp_path, matrix, arms, urls):
    """eval_paired.py --dry-run: 5 seed blocks x 2 arms; each arm's --vla is the last one on its command line."""
    r = subprocess.run([sys.executable, str(RT / "scripts" / "eval_paired.py"), "--matrix", str(RT / "configs" / matrix),
                        "--out", str(tmp_path / "out"), "--dry-run"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert len(lines) == 10
    for arm, url in zip(arms, urls):
        mine = [ln for ln in lines if ln.startswith(f"{arm}_")]
        assert len(mine) == 5
        for ln in mine:
            vlas = [tok for i, tok in enumerate(ln.split()) if ln.split()[i - 1] == "--vla"]
            assert vlas[-1] == url and "--wbc tcp://127.0.0.1:5556" in ln and "--crn" in ln
    seeds = sorted({ln.split("--seeds ")[1].split()[0] for ln in lines})
    first = "1000" if "sync" in matrix else "2000"
    assert seeds[0].startswith(first) and len(seeds) == 5


def test_primary_gpu_arm_matches_t19():
    """C4gpu has T-1.9's C4 settings (same mode/arm/s/B/WBC/CRN/ticks/seeds/VLA), so the fresh, noise-paired GPU arm
    of T-2.7 is comparable with M1; it is run again, never reused (the matrix comment says why)."""
    m2 = yaml.safe_load((RT / "configs" / "eval_m2_primary.yaml").read_text())
    m1 = yaml.safe_load((RT / "configs" / "eval_m1_primary.yaml").read_text())
    arm = {a["name"]: a for a in m2["arms"]}["C4gpu"]
    m1c4 = {a["name"]: a for a in m1["arms"]}["C4"]
    tokens = lambda s: s.split()  # noqa: E731
    m2_args = tokens(m2["common_args"]) + tokens(arm["args"])
    m1_args = tokens(m1["common_args"]) + tokens(m1c4["args"])

    def opts(args):
        out, i = {}, 0
        while i < len(args):
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                out[args[i]] = args[i + 1]
                i += 2
            else:
                out[args[i]] = True
                i += 1
        return out

    o2, o1 = opts(m2_args), opts(m1_args)
    assert o2.pop("--d-init") == "4"  # the loop's default, which T-1.9 used implicitly
    assert o2 == o1 and m2["seeds"] == m1["seeds"]


# --- T-2.6's T-2.3 baseline (verify:all D5) --------------------------------------------------------------------------


def _session(root: Path, name: str, mtime: float, passed, arena: str | None, span: float, in_info: bool = True,
             in_bench: bool = False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    stamp = {"mk_arena_dtype": arena} if arena else {}
    bench = {"pass": passed, "latency": {"server_span_ms": {"p50": span}, "server_total_ms": {"p50": span - 0.5}},
             "info": {"stamp": stamp} if in_bench else {}}
    (d / "t23_bench.json").write_text(json.dumps(bench))
    if in_info:
        (d / "server_g1_info.json").write_text(json.dumps({"stamp": stamp}))
    os.utime(d / "t23_bench.json", (mtime, mtime))
    return d / "t23_bench.json"


def test_pick_bench_takes_the_newest_passing_bfp8_b_bench(tmp_path, capsys):
    """S1's layout: the default-arena S1 bench, then the newer WP-2.7 bf16 bench (pass false), then a fake session."""
    s1 = _session(tmp_path, "20261025_004501_1_s1", 1000, True, "bfp8_b", 44.12)
    _session(tmp_path, "20261025_004849_2_s1_wp27_bf16", 2000, False, "bf16", 47.06)
    _session(tmp_path, "20261025_010000_3_fake", 3000, True, None, 1.0)
    _session(tmp_path, "20261025_010100_4_bf16_pass", 4000, True, "bf16", 46.0)
    res = M.pick_bench(tmp_path)
    assert res["bench"] == str(s1) and res["p50_ms"] == pytest.approx(44.12)
    assert res["arena_from"] == "server_g1_info.json"
    assert len(res["skipped"]) == 3
    assert M.main(["bench", str(tmp_path)]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == str(s1) and "T-2.3 baseline:" in out.err and "pass False" in out.err
    gates = tmp_path / "g.yaml"
    assert M.main(["gates", str(GATES), str(gates), out.out.strip()]) == 0
    assert yaml.safe_load(gates.read_text())["m2_rt"]["vla_p50_ms_max"] == pytest.approx(48.532)  # not 51.77
    # A T-2.3-only session whose arena is only in the bench's own INFO (bench_g1 records it at start) also counts.
    clean = _session(tmp_path, "20261025_020000_5_t23_clean", 5000, True, "bfp8_b", 43.0, in_info=False, in_bench=True)
    assert M.pick_bench(tmp_path)["bench"] == str(clean)


def test_pick_bench_none(tmp_path):
    _session(tmp_path, "a", 1000, False, "bfp8_b", 44.0)
    assert M.pick_bench(tmp_path)["bench"] is None and M.main(["bench", str(tmp_path)]) == 1
    assert M.main(["bench", str(tmp_path / "empty")]) == 1


def test_pick_bench_on_the_real_s1_sessions():
    """Read-only regression on runs/sessions: whatever is newest, the pick is a passing bfp8_b bench, never WP-2.7's."""
    root = RT / "runs" / "sessions"
    if not (root / "20261025_004849_1127496_s1_wp27_bf16" / "t23_bench.json").exists():
        pytest.skip("no S1 artefacts on this machine")
    res = M.pick_bench(root)
    assert res["bench"] and "wp27" not in res["bench"] and res["arena"] == "bfp8_b"
    assert json.loads(Path(res["bench"]).read_text())["pass"] is True


# --- session matrices and the T-2.5 / T-2.7 server error check -----------------------------------------------------


@pytest.mark.parametrize("matrix", ["eval_m2_sync.yaml", "eval_m2_primary.yaml"])
def test_session_matrix_puts_the_session_endpoints_in(tmp_path, matrix):
    tt, gpu = "tcp://127.0.0.1:40001", "tcp://127.0.0.1:40002"
    out = tmp_path / "m.yaml"
    assert M.main(["matrix", str(RT / "configs" / matrix), str(out), tt, gpu, "ort"]) == 0
    r = subprocess.run([sys.executable, str(RT / "scripts" / "eval_paired.py"), "--matrix", str(out),
                        "--out", str(tmp_path / "o"), "--dry-run"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert len(lines) == 10 and all("--wbc ort" in ln and "--crn" in ln for ln in lines)
    assert not any(p in r.stdout for p in (":5557", ":5565", ":5556"))
    for ln in lines:
        vla = ln.split("--vla ")[-1].split()[0]
        assert vla == (tt if "tt_" in ln.split()[0] else gpu), ln
    src = yaml.safe_load((RT / "configs" / matrix).read_text())
    got = yaml.safe_load(out.read_text())
    assert [a["name"] for a in got["arms"]] == [a["name"] for a in src["arms"]] and got["seeds"] == src["seeds"]


def test_session_matrix_refuses_an_unknown_endpoint(tmp_path):
    src = tmp_path / "m.yaml"
    src.write_text(yaml.safe_dump({"name": "x", "common_args": "--wbc tcp://127.0.0.1:5556",
                                   "arms": [{"name": "a", "args": "--vla tcp://127.0.0.1:5555"}]}))
    assert M.main(["matrix", str(src), str(tmp_path / "o.yaml"), "tcp://127.0.0.1:1", "tcp://127.0.0.1:2", "ort"]) == 2


def _requests_run(d: Path, outcomes: list) -> None:
    run = telemetry.RunDir(d, tag="m2_sync_x", config={"args": {}})
    rl = telemetry.RequestLog(run)
    for i, o in enumerate(outcomes):
        rl.add({"seq": i + 1, "ep": 0, "k_obs": 16 * i, "d": 3, "t_send_ns": 10**12 + i, "t_recv_ns": 10**12 + i + 5,
                "outcome": o})
    rl.close()


def test_server_errors_per_arm(tmp_path):
    runs = tmp_path / "runs"
    _requests_run(runs / "C1tt" / "1000-1009", [Outcome.APPLIED] * 5)
    _requests_run(runs / "C1gpu" / "1000-1009", [Outcome.APPLIED] * 5)
    res = M.server_errors(runs)
    assert res["pass"] and res["per_arm"]["C1tt"] == {"runs": 1, "requests": 5, "ERROR": 0, "TIMEOUT": 0}
    _requests_run(runs / "C1tt" / "1010-1019", [Outcome.APPLIED, Outcome.ERROR, Outcome.TIMEOUT])
    out = tmp_path / "e.json"
    assert M.main(["errors", str(runs), str(out)]) == 1
    rep = json.loads(out.read_text())
    assert rep["per_arm"]["C1tt"]["ERROR"] == 1 and rep["per_arm"]["C1tt"]["TIMEOUT"] == 1
    assert {c["name"] for c in rep["checks"] if not c["pass"]} == {"C1tt_errors", "C1tt_timeouts"}
    assert M.main(["errors", str(tmp_path / "none"), str(out)]) == 1  # no runs at all is not a pass
