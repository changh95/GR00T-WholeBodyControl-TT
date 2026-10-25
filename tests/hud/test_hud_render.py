"""scripts/hud.py (PLAN.md §4.3, WP-1.7): the model folded from real telemetry bytes, headless rendering to PNG,
and the HUD process against a live, stalled, absent and flooding publisher (rtloop.telemetry.Publisher +
tick_msg via tests/hud/synth_pub.py). The Blackhole chip panel (§4.3 item 7) against synthetic x280d and server_g1
status streams (tests/hud/synth_status.py): the fields it shows, and malformed, absent and stalled streams.
Never opens a window: SDL_VIDEODRIVER=dummy, DISPLAY removed."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.pop("DISPLAY", None)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

HERE = Path(__file__).resolve().parent
RT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(RT / "scripts")]

import hud  # noqa: E402
from rtloop import control as cp  # noqa: E402
from rtloop import telemetry, wire  # noqa: E402
from rtloop.schema import TICK_DTYPE  # noqa: E402
from synth_pub import META, SynthPublisher, SynthStream  # noqa: E402
from synth_status import CORE_MAP_COUNTS, StatusPublisher, server_g1_msg, x280d_msg  # noqa: E402

# Never a real-time CPU (2-4, 6, 7, 12-15) or its SMT sibling (18-20, 22, 23, 28-31): PLAN.md §1.1.
AUX_CPUS = os.environ.get("RTLOOP_TEST_AUX_CPUS", "24,25")


def feed(model, profile="m1", eps=1, stop=None, roundtrip=True):
    """Fold SynthStream into a HudModel through the wire codec; stop=(ep, k) ends after that tick."""
    st = SynthStream(ticks_per_ep=500, profile=profile)
    for ep in range(eps):
        for topic, f in st.episode(ep):
            if topic == "ego":
                f = {"seq": f[0], "k_obs": f[1], "img": f[2]}
            model.ingest(topic.encode(), wire.loads(wire.dumps(f)) if roundtrip else f)
            if topic == "tick" and stop == (ep, f["k"]):
                return model
    return model


# --- model ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind,backend,url,device,label", [
    ("vla", "cuda:rtx5090:cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc", None, "gpu", "VLA 5090"),
    ("vla", "tt:p150a:cloudwalk@2a5b61cc", None, "tensix", "VLA Tensix"),
    ("vla", "tensix-n16-g1", None, "tensix", "VLA Tensix"),
    ("vla", "mock:detfn", None, "host", "VLA mock"),
    ("vla", "policy-server", "tcp://127.0.0.1:5555", "gpu", "VLA 5090 (cfg)"),
    ("vla", "policy-server", "tcp://127.0.0.1:5557", "tensix", "VLA Tensix (cfg)"),
    ("wbc", "cuda-graph:rtx5090", None, "gpu", "WBC 5090"),
    ("wbc", "ort-cpu", None, "host", "WBC ORT CPU"),
    ("wbc", "x280:l2cpu3:h0@800MHz", None, "x280", "WBC-MLP X280 @800 MHz"),
    ("wbc", "x280", None, "x280", "WBC-MLP X280 @800 MHz"),
    ("wbc", "x280-sim", None, "host", "WBC-MLP X280-sim"),
    ("wbc", "policy-server", "tcp://127.0.0.1:5555", "gpu", "WBC 5090 (cfg)"),
    ("wbc", None, None, "?", "WBC ?"),
])
def test_classify(kind, backend, url, device, label):
    c = hud.classify(kind, backend, url)
    assert (c["device"], c["label"]) == (device, label)
    assert c["cfg"] == (backend == "policy-server")


@pytest.mark.parametrize("profile,caption", [
    ("m1", "VLA 5090 / WBC 5090"),
    ("m2", "VLA Tensix / WBC 5090"),
    ("m3", "VLA Tensix / WBC-MLP X280 @800 MHz, 410 µs (mcycle)"),
])
def test_processor_map_follows_meta(profile, caption):
    m = feed(hud.HudModel(), profile, stop=(0, 60))
    p = m.processors()
    assert p["caption"] == caption and p["have_meta"]


def test_model_counts_and_episode_bookkeeping():
    m = feed(hud.HudModel(), "m1", eps=2)
    s = m.summary()
    assert s["rx"]["tick"] == 1000 and s["bad"] == 0 and s["dropped_ticks"] == 0
    assert s["episodes"] == 1 and s["successes"] == 1  # ep 0 succeeds on its last tick; ep 1 still open
    assert m.n_starved == 12 and m.max_starve_run == 12  # ep 1, k 300..311
    assert m.n_fallback == 4 and m.n_overrun == 10  # synth: k % 211 == 100, k % 97 == 50, per episode
    v = m.vla_stats()
    assert v["d_p50"] == 4 and v["s_eff"] == 16 and 62.0 <= v["p50_ms"] <= 66.0
    t = m.tick_stats()
    assert 20.0 <= t["period_p99_ms"] <= 20.05 and t["jitter_ms"] < 0.05
    assert m.active_req()["seq"] == m.last["seq"]
    seq, k_obs, img = m.ego_image()
    assert seq == m.last["seq"] and img.shape == (120, 160, 3)


def test_fall_and_dropped_ticks():
    m = hud.HudModel()
    row = np.zeros((), dtype=TICK_DTYPE)
    for k in (0, 1, 2, 5, 6):  # 3 and 4 dropped by the PUB
        row["k"], row["ep"], row["pelvis_z"] = k, 0, 0.75 if k < 6 else 0.3
        m.ingest(b"tick", wire.loads(wire.dumps(telemetry.tick_msg(row))))
    row["k"], row["ep"], row["pelvis_z"] = 0, 1, 0.75
    m.ingest(b"tick", telemetry.tick_msg(row))
    assert m.dropped_ticks == 2 and m.falls == 1 and m.episodes == 1 and m.successes == 0


def test_bad_messages_never_raise():
    m = hud.HudModel()
    m.ingest_raw([b"tick", b"\xc1\xc1"])
    m.ingest_raw([b"tick"])
    m.ingest(b"tick", {"no_k": 1})
    m.ingest(b"ego", {"seq": 1, "img": np.zeros((4, 4), np.float32)})
    m.ingest(b"tick", [1, 2])
    m.ingest(b"weird", {"a": 1})
    assert m.bad == 5 and m.rx["other"] == 1 and m.last is None


def test_status_transitions():
    m = hud.HudModel()
    assert m.status(0) == ("NO DATA", None)
    row = np.zeros((), dtype=TICK_DTYPE)
    m.ingest(b"tick", telemetry.tick_msg(row), t_ns=1_000_000_000)
    assert m.status(1_100_000_000)[0] == "LIVE"
    st, age = m.status(3_000_000_000)
    assert st == "STALE" and abs(age - 2.0) < 1e-9
    m.ingest(b"meta", {"paused": True, "processor": META["m1"]}, t_ns=3_000_000_000)
    assert m.status(3_100_000_000)[0] == "PAUSED"


def test_fallback_marks_the_tick_backend():
    m = feed(hud.HudModel(), "m3", stop=(0, 100))  # synth: tick 100 falls back to ORT
    p = m.processors()
    assert p["fallback"] and p["wbc_now"] == "ORT" and p["wbc"]["device"] == "x280"


# --- rendering --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg():
    import pygame

    pygame.display.init()
    pygame.font.init()
    yield pygame
    pygame.quit()


def render(pg, model, tmp_path, name, **kw):
    surf = pg.Surface((hud.W, hud.H))
    view = hud.HudView(pg)
    t0 = time.perf_counter()
    view.draw(surf, model, time.monotonic_ns(), 30.0, "test", **kw)
    ms = (time.perf_counter() - t0) * 1e3
    path = tmp_path / f"{name}.png"
    pg.image.save(surf, str(path))
    return surf, path, ms


def pixel(surf, xy):
    return tuple(surf.get_at(xy))[:3]


def test_render_states_to_png(pg, tmp_path):
    shots = {
        "nodata": hud.HudModel(),
        "m1": feed(hud.HudModel(), "m1", stop=(0, 120)),
        "m3_starved": feed(hud.HudModel(), "m3", eps=2, stop=(1, 305)),
    }
    out = {}
    for name, m in shots.items():
        surf, path, _ = render(pg, m, tmp_path, name)
        assert path.stat().st_size > 10_000
        assert pg.image.load(str(path)).get_size() == (1280, 720)
        out[name] = surf
    # status pill (top right) and the lit Blackhole chips differ by state
    assert pixel(out["nodata"], (1258, 22)) == hud.C["crit"]
    assert pixel(out["m1"], (1258, 22)) == hud.C["good"]
    # M1 (GPU VLA and WBC): Host, RTX 5090 and Blackhole rows; the Blackhole row's Tensix chip is unlit
    gpu_vla_chip, bh_row_chip = (44, 168), (41, 217)
    assert pixel(out["m1"], gpu_vla_chip) == hud.C["vla"] and pixel(out["m1"], bh_row_chip) != hud.C["vla"]
    # M3 (Tensix + X280, nothing on the GPU): no 5090 row, the Blackhole box moves up into its place
    assert pixel(out["m3_starved"], gpu_vla_chip) == hud.C["vla"]  # the Tensix chip now
    assert pixel(out["m3_starved"], (44, 202)) == hud.C["wbc"]  # X280: WBC-MLP on the box's second line
    # ego thumbnail is drawn (gradient, not the panel colour) only once a thumbnail arrived
    assert pixel(out["m1"], (1200, 300)) != hud.C["panel"]
    assert pixel(out["nodata"], (1200, 300)) == hud.C["panel"]


def test_frame_cost_fits_30_fps(pg, tmp_path):
    m = feed(hud.HudModel(), "m3", eps=2, stop=(1, 400))
    render(pg, m, tmp_path, "warm")
    costs = [render(pg, m, tmp_path, f"f{i}")[2] for i in range(10)]
    assert np.median(costs) < 20.0, costs  # 33 ms frame budget; measured ~4-8 ms


# --- the HUD process --------------------------------------------------------------------------------


def free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if not 5555 <= port <= 5572:
            return port


def hud_env():
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    env.update(SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONPATH=f"{RT}:{env.get('PYTHONPATH', '')}")
    return env


def run_hud(tmp_path, sub_url, *extra, timeout=60, control=None, x280=None, tt=None):
    """The HUD process, headless; every URL it uses is a random port (never 5555-5574 defaults)."""
    png = tmp_path / "hud.png"
    cmd = [sys.executable, str(RT / "scripts" / "hud.py"), "--headless", "--sub", sub_url,
           "--control", control or f"tcp://127.0.0.1:{free_port()}", "--png", str(png), "--cpus", AUX_CPUS,
           "--x280-status", x280 or f"tcp://127.0.0.1:{free_port()}",
           "--tt-status", tt or f"tcp://127.0.0.1:{free_port()}", *extra]
    p = subprocess.run(cmd, env=hud_env(), capture_output=True, text=True, timeout=timeout)
    assert p.returncode == 0, p.stderr[-3000:]
    summary = json.loads(p.stdout.strip().splitlines()[-1])
    assert png.exists() and summary["png"] == str(png)
    return summary


def test_hud_process_against_live_publisher(tmp_path):
    pub = SynthPublisher(profile="m3")
    pub.start()
    try:
        s = run_hud(tmp_path, pub.url, "--duration", "4")
    finally:
        pub.stop()
    assert s["status"] == "LIVE" and s["bad"] == 0
    assert s["rx"]["tick"] >= 120 and s["rx"]["meta"] >= 2 and s["rx"]["ego"] >= 5 and s["rx"]["req"] >= 5
    assert 25 <= s["fps"] <= 32 and s["dropped_ticks"] == 0
    assert s["max_frame_ms"] < 100


def test_hud_survives_absent_publisher(tmp_path):
    s = run_hud(tmp_path, f"tcp://127.0.0.1:{free_port()}", "--frames", "30")
    assert s["frames"] == 30 and s["status"] == "NO DATA" and sum(s["rx"].values()) == 0


def test_hud_survives_stalled_publisher(tmp_path):
    pub = SynthPublisher(stall_after=60)
    pub.start()
    try:
        s = run_hud(tmp_path, pub.url, "--duration", "4")
        assert pub.stalled.is_set()
    finally:
        pub.stop()
    assert s["status"] == "STALE" and s["frames"] >= 90
    assert 1 <= s["rx"]["tick"] <= 60


def test_hud_drops_on_overflow_and_keeps_its_frame_rate(tmp_path):
    pub = SynthPublisher(rate_hz=0, n_ticks=400_000, hwm=1000)  # as fast as the PUB can go
    pub.start()
    try:
        s = run_hud(tmp_path, pub.url, "--duration", "3", "--max-drain", "300", "--hwm", "200")
    finally:
        pub.stop()
    assert s["rx"]["tick"] > 0 and s["fps"] >= 15
    assert s["dropped_ticks"] > 0 or s["drain_capped_frames"] > 0
    assert pub.pub.dropped > 0 or s["dropped_ticks"] > 0  # overflow is dropped, never queued without bound


def test_hud_keys_reach_the_control_pull(tmp_path, zmq_ctx):
    pull = cp.ControlPull("tcp://127.0.0.1:*", ctx=zmq_ctx)
    pub = SynthPublisher()
    pub.start()
    try:
        s = run_hud(tmp_path, pub.url, "--frames", "40", "--inject-keys", "10:w,15:v,20:p,25:r,30:x",
                    control=pull.url)
    finally:
        pub.stop()
    got, t_end = [], time.monotonic() + 2
    while len(got) < 4 and time.monotonic() < t_end:
        got += pull.poll(now_ns=None)
        time.sleep(0.01)
    pull.close()
    assert s["control_sent"] == 4 and s["control_failed"] == 0
    assert [(c.cmd, c.to) for c in got] == [("swap_wbc", "next"), ("swap_vla", "next"), ("pause", None),
                                            ("reset", None)]


def test_hud_keys_without_a_loop_fail_visibly(tmp_path):
    s = run_hud(tmp_path, f"tcp://127.0.0.1:{free_port()}", "--frames", "20", "--inject-keys", "5:w")
    assert s["control_sent"] == 0 and s["control_failed"] == 1


# --- the Blackhole chip panel (§4.3 item 7) ---------------------------------------------------------------


def mk_available() -> bool:
    return bool(hud.mk_default_core_map())


def chip_model(profile="m3", x280=True, tt=True, now=None, x280_age_s=0.0, **x280_over):
    m = feed(hud.HudModel(), profile, stop=(0, 120))
    now = time.monotonic_ns() if now is None else now
    for i in (1, 2):
        if x280:
            m.chip.ingest(wire.loads(wire.dumps(x280d_msg(i, **x280_over))), now - int(x280_age_s * 1e9))
        if tt:
            m.chip.ingest(wire.loads(wire.dumps(server_g1_msg(i))), now)
    return m, now


CHIP_RECT = (452, 192, 400, 452)  # where HudView.draw puts the chip panel


class RecordingView(hud.HudView):
    """Collects every string the frame draws (with its anchor), so the test can check which fields are shown."""

    def __init__(self, pg, **kw):
        super().__init__(pg, **kw)
        self.drawn: list[str] = []
        self.at: list[tuple[str, tuple]] = []
        self.color: dict[str, object] = {}

    def text(self, surf, s, pos, *a, **kw):
        self.drawn.append(str(s))
        self.at.append((str(s), tuple(pos)))
        self.color[str(s)] = kw.get("color", a[1] if len(a) > 1 else "ink")
        return super().text(surf, s, pos, *a, **kw)

    def pill(self, surf, s, pos, *a, **kw):
        self.drawn.append(str(s))
        self.at.append((str(s), tuple(pos)))
        return super().pill(surf, s, pos, *a, **kw)

    def in_chip_panel(self) -> list[str]:
        x, y, w, h = CHIP_RECT
        return [s for s, (px, py) in self.at if x <= px <= x + w and y <= py <= y + h]

    def row(self, key: str) -> str:
        """The value drawn on a key's line: the nearest string to its right at the same y (chip panel columns)."""
        kx, ky = next(pos for s, pos in self.at if s == key)
        return min(((px, s) for s, (px, py) in self.at if py == ky and px > kx), default=(0, None))[1]


def draw(pg, m, now, chip="auto"):
    surf = pg.Surface((hud.W, hud.H))
    view = RecordingView(pg, chip=chip)
    view.draw(surf, m, now, 30.0, "test")
    return surf, view


def test_chip_model_states_gaps_and_restarts():
    c = hud.ChipModel()
    assert c.state("x280d", 0) == ("NO DATA", None) and not c.seen_any()
    for seq in (1, 2, 5):
        assert c.ingest(x280d_msg(seq), t_ns=seq * 10**8)
    assert c.gaps["x280d"] == 2 and c.restarts["x280d"] == 0 and c.rx["x280d"] == 3
    assert c.ingest(x280d_msg(1, pid=999), t_ns=6 * 10**8)  # a restarted x280d starts its seq again
    assert c.restarts["x280d"] == 1 and c.gaps["x280d"] == 2
    assert c.state("x280d", 7 * 10**8)[0] == "LIVE"
    st, age = c.state("x280d", 6 * 10**8 + int(hud.STATUS_STALE_S * 1e9) + 1)
    assert st == "STALE" and age > hud.STATUS_STALE_S
    assert c.state("server_g1", 10**9)[0] == "NO DATA" and c.seen_any()


def test_chip_model_rejects_malformed_frames():
    c = hud.ChipModel()
    good = [b"status", wire.dumps(x280d_msg(1))]
    assert c.ingest_raw(good)
    bad = [[b"status", b"\xc1\xc1"], [b"weird", good[1]], [b"status"], [b"status", wire.dumps([1, 2])],
           [b"status", wire.dumps({"src": "someone"})], [b"status", wire.dumps({"no_src": 1})],
           [b"status", b"\x00" * (hud.STATUS_MAX_BYTES + 1)]]
    for parts in bad:
        assert not c.ingest_raw(parts)
    assert c.bad == len(bad) and c.rx["x280d"] == 1 and c.get("x280d")["seq"] == 1


def test_core_layout_counts_coordinates_and_garbage():
    assert hud.core_layout(None) == {} and hud.core_layout({"grid": "11x10"}) == {}
    assert hud.core_layout({"grid": [0, 10]}) == {} and hud.core_layout({"grid": [11, 10, 1]}) == {}
    lay = hud.core_layout(CORE_MAP_COUNTS)
    assert lay["grid"] == (11, 10) and lay["counts"] == {"reader": 8, "hub": 1, "compute": 96, "spare": 5}
    if mk_available():  # counts only: the roles are placed from MK's own default map, which agrees
        assert lay["source"] == "MK default_core_map" and len(lay["cells"]) == 110
        assert lay["cells"][(0, 0)] == "reader" and lay["cells"][(10, 9)] == "hub"
        manifest = hud.mk_default_core_map()  # MK CoreMap.manifest(): coordinates reported directly
        rep = hud.core_layout(manifest)
        assert rep["source"] == "reported" and rep["cells"] == lay["cells"]
    else:
        assert lay["cells"] == {} and lay["source"] is None
    other = hud.core_layout({"grid": [11, 10], "megakernel": {"readers": 7, "hub": 1, "compute": 96, "spare": 6}})
    assert other["cells"] == {} and other["source"] is None  # disagreeing counts: never placed by MK's map
    own = hud.core_layout({"grid": [4, 2], "readers": [[0, 0], [9, 9]], "hub": [3, 1], "compute": [[1, 0]]})
    assert own["source"] == "reported" and own["cells"] == {(0, 0): "reader", (3, 1): "hub", (1, 0): "compute"}


def test_chip_telemetry_takes_the_freshest_live_source():
    m, now = chip_model()
    tel, src = m.chip.telemetry(now)
    assert src == "x280d" and tel["aiclk_mhz"] == 1350  # server_g1's telemetry is all None (no read API)
    m, now = chip_model(x280_age_s=5.0)
    assert m.chip.telemetry(now) == ({}, None)


TELEMETRY_NUMBERS = ("AICLK", "1350", "52.5", "VCORE", "750 mV", "TDP", "TIMER_HEARTBEAT", "via x280d")


@pytest.mark.parametrize("over,note", [
    ({"backend": "x280-sim:v2-fp16"}, "no CMFW telemetry (sim)"),  # the sim's numbers are invented (D4)
    ({"backend": "x280-sim:v2-fp16", "telemetry": None, "telemetry_source": "sim"}, "no CMFW telemetry (sim)"),
    ({"telemetry_source": "FakeDevice"}, "no CMFW telemetry (sim)"),  # a device stamp naming a fake source
    ({"telemetry": None, "telemetry_error": "CMFW table pointer 0"}, "no CMFW telemetry (x280d read failed)"),
    ({"telemetry": None}, "no CMFW telemetry (x280d reports none)"),
])
def test_chip_telemetry_row_never_shows_invented_or_absent_numbers(pg, over, note):
    """D4: the row is labelled by its source; a simulated source or a message with telemetry None shows why there
    are no numbers, never numbers (x280d_msg carries the synthetic TELEMETRY values unless overridden)."""
    m, now = chip_model(**over)
    assert m.chip.telemetry(now) == ({}, None)
    assert m.chip.telemetry_row(now) == {"tel": {}, "src": None, "source": None, "note": note}
    s = "\n".join(draw(pg, m, now)[1].in_chip_panel())
    assert note in s
    for never in TELEMETRY_NUMBERS:
        assert never not in s, never


def test_chip_telemetry_row_labels_its_source(pg):
    m, now = chip_model()
    row = m.chip.telemetry_row(now)
    assert (row["src"], row["source"], row["note"]) == ("x280d", "CMFW table", None) and row["tel"]["aiclk_mhz"] == 1350
    m, now = chip_model(telemetry_source="cmfw:arc-window")  # x280d names its own source: shown as sent
    assert m.chip.telemetry_row(now)["source"] == "cmfw:arc-window"
    assert "cmfw:arc-window via x280d" in "\n".join(draw(pg, m, now)[1].in_chip_panel())
    m, now = chip_model(x280=False)  # TT-only (M2): server_g1 has no CMFW read
    assert m.chip.telemetry_row(now)["note"] == "no CMFW telemetry (server_g1 has no CMFW read)"
    m.chip.ingest(server_g1_msg(9, backend="mock:server_g1-fake", telemetry={"aiclk_mhz": 1}), now)
    assert m.chip.telemetry_row(now)["note"] == "no CMFW telemetry (sim)"
    assert hud.telemetry_source({"backend": "x280:l2cpu3:h0@800MHz:v2-fp16"}) == ("CMFW table", False)
    assert hud.telemetry_source({"backend": "x280-sim"}) == ("sim", True)
    assert hud.telemetry_source(None) == ("CMFW table", False)


def test_l2cpu_noc_fallback_equals_regs(pg, monkeypatch):
    """D11: without x280.regs the floorplan places L2CPU0-3 at the same NoC0 rows as regs.L2CPU_XY (3, 9, 5, 7:
    tile order is not row order), from a copy that must equal it."""
    from x280 import regs

    want = {int(k): tuple(v) for k, v in regs.L2CPU_XY.items()}
    assert hud.L2CPU_NOC0_FALLBACK == want
    assert {t: hud.l2cpu_xy(t) for t in range(4)} == want
    monkeypatch.setattr(hud, "_X280_REFS", {"xy": {}, "states": {}})  # x280 tree not importable
    assert {t: hud.l2cpu_xy(t) for t in range(4)} == want
    m, now = chip_model()
    surf, view = draw(pg, m, now)
    assert pixel(surf, (830, 324)) == hud.C["serious"]  # L2CPU3 lit at row 7 (released, no state names)
    assert "L2CPU3 released" in "\n".join(view.in_chip_panel())


def test_chip_panel_shows_the_contract_fields(pg, tmp_path):
    m, now = chip_model()
    surf, view = draw(pg, m, now)
    pg.image.save(surf, str(tmp_path / "chip_live.png"))
    s = "\n".join(view.drawn)
    for want in ("BLACKHOLE p150a", "TT LIVE", "X280 LIVE", "Tensix 11×10 (logical)", "8 reader", "1 hub",
                 "96 compute", "5 spare", "L2CPU3 (8,7) idle", "L2CPU2 (8,5) spare", "L2CPU0 (8,3) reserved",
                 "2.6/39.4/0.1", "2.5/39.3/0.1", "42.2/42.0", "12 %", "6 / 0", "on / 2", "MK@009de892c89",
                 "X280 · x280d (L2CPU3)", "410.00 µs", "612/701 µs",
                 "100 / 0", "1 / 0", "0 / 0", "v2-fp16", "0x20001 1a2b3c4d",
                 "AICLK 1350 MHz", "L2CPUCLK3 800 MHz", "ASIC 52.5 °C", "VCORE 750 mV", "TDP 48 W",
                 "TDC 62 A", "in 55 W", "TIMER_HEARTBEAT 1002", "CMFW table via x280d"):
        assert want in s, want
    assert "TICK TIMING (last 60 s)" not in s  # the panel takes the histograms' place
    # two snapshots microseconds apart: no rate yet, so the hart shows its monitor state and the heartbeat waits
    assert (view.row("tile / hart"), view.row("heartbeat"), view.row("calls served")) == ("idle", "…", "100")
    for gone in ("heartbeat age", "cycles l/max", "duty of 20 ms", "328k/401k", "2.1 %"):  # constants, not repeated
        assert gone not in s, gone
    lit = pixel(surf, (830, 324))  # L2CPU3's box in the x = 8 column (right of its label): a live, idle hart
    assert lit == hud.C["wbc"], lit
    if mk_available():
        assert pixel(surf, (469, 241)) == hud.C["reader"]  # core (0, 0) is a DRAM reader on this card


def test_chip_panel_stale_x280_and_absent_server_g1(pg):
    m, now = chip_model(profile="m3", tt=False, x280_age_s=4.0, monitor_state=3)
    surf, view = draw(pg, m, now)
    s = "\n".join(view.drawn)
    assert "TT NO DATA" in s and "X280 STALE 4 s" in s and "no core map" in s
    assert "L2CPU3 (8,7) fault (stale)" in s  # the last known state, marked stale, never lit
    assert pixel(surf, (830, 324)) == hud.C["axis"]
    assert "chip telemetry (CMFW table): no live source" in s


def test_chip_panel_survives_every_field_malformed(pg):
    m = feed(hud.HudModel(), "m3", stop=(0, 120))
    now = time.monotonic_ns()
    m.chip.ingest({"src": "x280d", "pid": "x", "seq": None, "tile": "three", "tile_state": 5, "monitor_state": {},
                   "heartbeat_age_us": True, "cycles_last": "many", "cycles_max": [1], "compute_us_last": None,
                   "duty": "high", "ccache_ecc": 7, "variant": 2, "fw_ver": 1.5, "build_id": "b",
                   "telemetry": [1, 2]}, now)
    m.chip.ingest({"src": "server_g1", "core_map": {"grid": "11x10"}, "last_ms": [1, 2], "p50_ms": "fast",
                   "calls": float("nan"), "errors": "0", "trace": "yes", "cq": 2.5, "tree": 7,
                   "device_duty": float("inf"), "telemetry": {"aiclk_mhz": "fast"}}, now)
    surf, view = draw(pg, m, now)
    s = "\n".join(view.in_chip_panel())
    assert "TT LIVE" in s and "X280 LIVE" in s and "—" in s and "no core map" in s
    for never in ("nan", "inf", "None", "True", "many", "fast", "three", "yes"):
        assert never not in s, never


def test_chip_panel_auto_mode_and_the_c_key(pg):
    m1 = feed(hud.HudModel(), "m1", stop=(0, 120))
    view = hud.HudView(pg)
    assert not view.chip_shown(m1)  # M1: no device stream, no Tensix/X280 backend
    assert view.chip_shown(feed(hud.HudModel(), "m2", stop=(0, 120)))  # VLA on Tensix
    view.toggle_chip(m1)
    assert view.chip == "on" and view.chip_shown(m1)
    view.toggle_chip(m1)
    assert view.chip == "off" and not view.chip_shown(m1)
    m3, _ = chip_model()
    assert not hud.HudView(pg, chip="off").chip_shown(m3) and hud.HudView(pg, chip="on").chip_shown(m1)


def test_chip_frame_cost_fits_30_fps(pg, tmp_path):
    m, now = chip_model()
    render(pg, m, tmp_path, "warm")
    costs = [render(pg, m, tmp_path, f"c{i}")[2] for i in range(10)]
    assert np.median(costs) < 20.0, costs


def test_hud_process_with_live_status_streams(tmp_path):
    pub = SynthPublisher(profile="m3")
    xs, tt = StatusPublisher("x280d"), StatusPublisher("server_g1")
    for p in (pub, xs, tt):
        p.start()
    try:
        s = run_hud(tmp_path, pub.url, "--duration", "3", x280=xs.url, tt=tt.url)
    finally:
        for p in (pub, xs, tt):
            p.stop()
    assert s["status"] == "LIVE" and s["chip_panel"] is True
    assert s["chip_state"] == {"server_g1": "LIVE", "x280d": "LIVE"}
    assert s["chip"]["rx"]["x280d"] >= 10 and s["chip"]["rx"]["server_g1"] >= 10 and s["chip"]["bad"] == 0
    assert s["chip_telemetry"] == {"src": "x280d", "source": "CMFW table", "note": None}
    assert 25 <= s["fps"] <= 32


def test_hud_process_with_a_sim_x280d_shows_no_telemetry(tmp_path):
    """D4 end to end: a synthetic x280d --backend sim stream that still carries numbers (as kmd.FakeDevice's did)
    next to server_g1: the HUD process's telemetry row says so and shows none of them."""
    pub = SynthPublisher(profile="m3")
    xs = StatusPublisher("x280d", backend="x280-sim:v2-fp16")
    tt = StatusPublisher("server_g1")
    for p in (pub, xs, tt):
        p.start()
    try:
        s = run_hud(tmp_path, pub.url, "--duration", "2", x280=xs.url, tt=tt.url)
    finally:
        for p in (pub, xs, tt):
            p.stop()
    assert s["chip_state"] == {"server_g1": "LIVE", "x280d": "LIVE"} and s["chip"]["bad"] == 0
    assert s["chip_telemetry"] == {"src": None, "source": None, "note": "no CMFW telemetry (sim)"}


def test_hud_process_with_stalled_and_garbage_status(tmp_path):
    pub = SynthPublisher(profile="m3")
    xs = StatusPublisher("x280d", mode="stall", stall_after=5)
    tt = StatusPublisher("server_g1", mode="garbage")
    for p in (pub, xs, tt):
        p.start()
    try:
        s = run_hud(tmp_path, pub.url, "--duration", "4", x280=xs.url, tt=tt.url)
        assert xs.stalled.is_set()
    finally:
        for p in (pub, xs, tt):
            p.stop()
    assert s["chip_state"]["x280d"] == "STALE" and s["chip_state"]["server_g1"] == "LIVE"
    assert s["chip"]["bad"] >= 5 and s["chip"]["rx"]["x280d"] <= 5 and s["frames"] >= 90


def test_hud_process_without_status_publishers(tmp_path):
    s = run_hud(tmp_path, f"tcp://127.0.0.1:{free_port()}", "--frames", "20", "--chip", "on")
    assert s["frames"] == 20 and s["chip_panel"] is True
    assert s["chip_state"] == {"server_g1": "NO DATA", "x280d": "NO DATA"} and s["chip"]["bad"] == 0
    s = run_hud(tmp_path, f"tcp://127.0.0.1:{free_port()}", "--frames", "5", "--no-status")
    assert s["chip_panel"] is False and s["chip_state"] == {"server_g1": "NO DATA", "x280d": "NO DATA"}


def test_core_layout_server_g1_format():
    """rttt/server_g1.py core_map_dict: counts, plus reader/hub/spare coordinates; compute = the rest of the grid."""
    cm = {"grid": [4, 3], "megakernel": {"readers": 2, "hub": 1, "compute": 8, "spare": 1},
          "readers": [[0, 0], [0, 2]], "hub": [3, 2], "spare": [[3, 1]]}
    lay = hud.core_layout(cm)
    assert lay["source"] == "reported" and len(lay["cells"]) == 12
    assert [lay["cells"][c] for c in ((0, 0), (0, 2), (3, 2), (3, 1), (1, 1))] == \
        ["reader", "reader", "hub", "spare", "compute"]
    cm["megakernel"]["compute"] = 7  # the rest does not match the reported count: compute stays unplaced
    assert "compute" not in hud.core_layout(cm)["cells"].values()


def test_chip_panel_dp4_degraded_and_building(pg):
    """DP4 fp32 lights both tiles (x280d's `tiles`); x280d's degraded flag and server_g1's ready=False show on the
    source pills, so a LIVE stream that serves errors never looks healthy."""
    m, now = chip_model(degraded="interlock: PLL4 not at default",
                        tiles={"2": {"monitor_state": "busy", "n_runs": 7}})
    m.chip.ingest(server_g1_msg(3, ready=False), now)
    surf, view = draw(pg, m, now)
    s = "\n".join(view.in_chip_panel())
    assert "X280 DEGRADED" in s and "TT BUILDING" in s
    assert "L2CPU3 (8,7) idle" in s and "L2CPU2 (8,5) busy" in s and "L2CPU2 (8,5) spare" not in s
    assert pixel(surf, (830, 302)) == hud.C["wbc"]  # L2CPU2's box, lit by its own monitor state


def test_chip_panel_variant_and_dtype_from_x280d(pg):
    """x280d publishes variant ("v2") and dtype ("fp16") as two fields; the panel shows them together, once."""
    m, now = chip_model(variant="v2", dtype="fp16")
    s = "\n".join(draw(pg, m, now)[1].in_chip_panel())
    assert "v2-fp16" in s and "v2-fp16-fp16" not in s
    m, now = chip_model(variant="v2-fp16", dtype="fp16")
    assert "v2-fp16-fp16" not in "\n".join(draw(pg, m, now)[1].in_chip_panel())


# --- X280 runs: no RTX 5090 on the map, live X280 values (user request 2026-10-25) ------------------------------


def x280_stream(m, now, n=8, hz=5.0, hb_hz=50e6, calls_hz=50.0, hb_frozen_from=None, rx_lag_s=0.0, **over):
    """Feed m.chip an x280d stream as x280d sends it: n snapshots at hz ending at `now` - rx_lag_s, each taken
    snapshot_age_ms before its send time; the heartbeat (50 MHz mtime) and the counters advance with the snapshot
    time, the heartbeat stops at message hb_frozen_from. Publisher and HUD share CLOCK_MONOTONIC here."""
    rng = np.random.default_rng(3)
    t_end = now - int(rx_lag_s * 1e9)
    period_ms = int(1e3 / hz)
    hb = 0
    for i in range(n):
        t_send = t_end - (n - 1 - i) * period_ms * 10**6
        age_ms = 20 * int(rng.integers(0, period_ms // 20))  # 20 ms steps keep the counters exact integers
        off_ms = (t_send - t_end) // 10**6 - age_ms  # snapshot time relative to t_end
        if hb_frozen_from is None or i < hb_frozen_from:
            hb = 10**12 + int(hb_hz // 1000) * off_ms
        calls = 10_000 + off_ms * int(calls_hz) // 1000
        msg = x280d_msg(i + 1, **{"t_ns": t_send, "snapshot_age_ms": float(age_ms), "monitor_state": "idle",
                                  "heartbeat": hb, "calls_ok": calls, "n_runs": 7 + calls, "compute_us_last": 278.07,
                                  "cycles_last": 222_455, "cycles_max": 223_041, "duty": 0.0139, **over})
        m.chip.ingest(wire.loads(wire.dumps(msg)), t_send + 500_000)
    return m


def test_chip_model_rates_from_snapshot_times():
    """Rates use the publisher's clock at the moment each value was read (t_ns - snapshot_age_ms), over ~1 s of
    snapshots: exact for a counter linear in time, whatever the send jitter; no rate from a re-sent snapshot, across
    a restart, from a counter that went backwards or from a snapshot sent stale."""
    c = hud.ChipModel()
    now = 50 * 10**9
    for i in range(10):
        t = now - (9 - i) * 200_000_000
        age = (37 * i) % 150
        snap_s = (t - age * 10**6) / 1e9
        assert c.ingest(x280d_msg(i + 1, t_ns=t, snapshot_age_ms=float(age), heartbeat=int(50e6 * snap_s),
                                  calls_ok=int(1000 * snap_s), tiles={"2": {"n_runs": int(20 * snap_s)}}), t)
    assert abs(c.rate("x280d", "heartbeat") - 50e6) < 1.0
    assert abs(c.rate("x280d", "calls_ok") - 1000) < 2.0 and abs(c.rate("x280d", "n_runs", "2") - 20) < 2.0
    assert c.rate("x280d", "no_such_key") is None and c.rate("server_g1", "calls") is None
    n = len(c.hist["x280d"])
    last = dict(c.get("x280d"))
    c.ingest({**last, "seq": last["seq"] + 1, "t_ns": last["t_ns"] + 200_000_000,
              "snapshot_age_ms": last["snapshot_age_ms"] + 200.0}, now + 200_000_000)  # the same snapshot again
    assert len(c.hist["x280d"]) == n and abs(c.rate("x280d", "heartbeat") - 50e6) < 1.0
    c.ingest({**last, "seq": 99, "t_ns": now + 400_000_000, "snapshot_age_ms": 1500.0}, now + 400_000_000)
    assert c.rate("x280d", "heartbeat") is None  # sent 1.5 s after it was taken: x280d's serving thread is stuck
    c.ingest(x280d_msg(1, pid=777, t_ns=now + 600_000_000, heartbeat=5, calls_ok=5), now + 600_000_000)
    assert c.restarts["x280d"] == 1 and len(c.hist["x280d"]) == 1 and c.rate("x280d", "heartbeat") is None
    c.ingest(x280d_msg(2, pid=777, t_ns=now + 800_000_000, heartbeat=10_000_005, calls_ok=15), now + 800_000_000)
    assert abs(c.rate("x280d", "heartbeat") - 50e6) < 1.0 and abs(c.rate("x280d", "calls_ok") - 50) < 1e-6
    c.ingest(x280d_msg(3, pid=777, t_ns=now + 10**9, heartbeat=20_000_005, calls_ok=3), now + 10**9)
    assert c.rate("x280d", "calls_ok") is None  # went backwards against every sample in the span


def test_x280_column_shows_live_values(pg):
    """The values that move while x280d serves: hart and tile "serving 50/s" (not the momentary idle), the
    heartbeat's rate, x280d's own per-call µs, calls served with calls/s; compute with two decimals."""
    m, now = chip_model(x280=False)
    x280_stream(m, now)
    surf, view = draw(pg, m, now)
    s = "\n".join(view.in_chip_panel())
    assert view.row("tile / hart") == "serving 50/s" and view.color["serving 50/s"] == "ink"
    assert view.row("heartbeat") == "+50.0 M/s" and view.color["+50.0 M/s"] == "ink"
    assert view.row("x280d last/p99") == "612/701 µs" and view.row("compute last") == "278.07 µs"
    served = view.row("calls served")
    assert served.endswith(" · 50.0/s") and int(served.split(" · ")[0]) == m.chip.get("x280d")["calls_ok"]
    assert "L2CPU3 (8,7) serving 50/s" in s and "idle" not in s
    assert pixel(surf, (830, 324)) == hud.C["wbc"]  # lit as a serving hart
    assert m.chip.serving(now) == {3: pytest.approx(50.0)}


@pytest.mark.parametrize("case,hart,hb,tile_note", [
    ("frozen", "serving 50/s", "stopped", "serving 50/s"),  # the hart's heartbeat stopped: critical within 1 s
    ("stale", "idle", "stream stale", "idle (stale)"),  # nothing for 3 s: no rate is claimed
    ("fault", "fault", "+50.0 M/s", "fault"),  # the real monitor state, never "serving"
    ("parked", "parked/parked", "+50.0 M/s", "parked"),
    ("old", "serving 50/s", "age 35 µs", "serving 50/s"),  # a publisher without the raw counter
])
def test_x280_column_when_not_healthy(pg, case, hart, hb, tile_note):
    m, now = chip_model(x280=False)
    kw = {"frozen": {"hb_frozen_from": 2}, "stale": {"rx_lag_s": 3.0}, "fault": {"monitor_state": "fault"},
          "parked": {"monitor_state": "parked", "tile_state": "parked"}, "old": {}}[case]
    x280_stream(m, now, **kw)
    if case == "old":  # same stream, without the raw heartbeat: the HUD falls back to heartbeat_age_us
        m.chip.last["x280d"] = {k: v for k, v in m.chip.get("x280d").items() if k != "heartbeat"}
    surf, view = draw(pg, m, now)
    assert view.row("tile / hart") == hart
    got = view.row("heartbeat")
    assert got.startswith(hb), got
    assert view.color[got] == ("crit" if case in ("frozen", "stale") else "ink")
    assert f"L2CPU3 (8,7) {tile_note}" in "\n".join(view.in_chip_panel())


@pytest.mark.parametrize("profile,targets,gpu", [
    ("m1", None, True), ("m2", None, True),
    ("m3", None, False),  # demo.sh m3 --no-alt: Tensix VLA, X280 WBC, ORT the only alternative
    ("m3", {"vla": ["tt"], "wbc": ["x280", "ort", "gpu"]}, True),  # demo.sh m3 with the GPU WBC as key-w target
    ("m3", {"vla": ["tt", "cuda5090"], "wbc": ["x280", "ort"]}, True),
])
def test_processor_map_shows_the_gpu_only_when_the_run_uses_it(pg, profile, targets, gpu):
    m = feed(hud.HudModel(), profile, stop=(0, 120))
    if targets is not None:
        m.ingest(b"meta", {**m.meta, "targets": targets})
    assert m.gpu_in_run() is gpu
    surf, view = draw(pg, m, time.monotonic_ns())
    s = "\n".join(view.drawn)
    assert ("RTX 5090" in s) is gpu
    if not gpu:  # no GPU wording anywhere in the frame; the Blackhole box has an engine per line
        assert not [t for t in view.drawn if "5090" in t or "gpu" in t.lower()], s
        assert pixel(surf, (44, 168)) == hud.C["vla"] and pixel(surf, (44, 202)) == hud.C["wbc"]


def test_processor_map_layout_follows_the_run_not_the_tick(pg):
    """The decision comes from the run's meta (processor stamps and swap targets), so a fallback tick (ORT) or a WBC
    swap between configured targets never moves the boxes; before any meta the GPU is not shown."""
    assert hud.HudModel().gpu_in_run() is False
    m = hud.HudModel()
    seen = set()
    for topic, f in SynthStream(ticks_per_ep=300, profile="m3").episode(0):
        if topic == "ego":
            f = {"seq": f[0], "k_obs": f[1], "img": f[2]}
        m.ingest(topic.encode(), f)
        if m.meta:
            seen.add(m.gpu_in_run())
    assert seen == {False} and m.n_fallback >= 1
    m = feed(hud.HudModel(), "m3", stop=(0, 60))
    alt = {"vla": ["tt"], "wbc": ["x280", "ort", "gpu"]}
    m.ingest(b"meta", {**m.meta, "targets": alt})
    assert m.gpu_in_run()
    m.ingest(b"meta", {**m.meta, "targets": alt, "processor": {**m.meta["processor"], "wbc": "cuda-graph:rtx5090"},
                       "current": {"vla": "tt", "wbc": "gpu"}})  # swapped to the GPU target (key w)
    assert m.gpu_in_run()


def test_blackhole_box_notes_without_the_gpu(pg):
    """The freed row: each Blackhole engine gets a line with what it is doing, from the status streams, or the
    loop's round trip without them."""
    m, now = chip_model(x280=False)
    x280_stream(m, now)
    view = draw(pg, m, now)[1]
    assert "L2CPU3 · serving 50/s" in view.drawn and "device p50 39.3 ms" in view.drawn
    m = feed(hud.HudModel(), "m3", stop=(0, 120))  # no status streams (--status-pub off)
    view = draw(pg, m, time.monotonic_ns(), chip="off")[1]
    assert "RTT p50 64.0 ms" in view.drawn and any(t.startswith("RTT p50 ") and t.endswith(" µs") for t in view.drawn)


def test_heartbeat_not_stopped_by_a_resent_snapshot_at_2_hz(pg):
    """At --status-hz 2 x280d may send one snapshot twice (the serving thread refreshes it at most every 0.5 s), so
    the heartbeat holds one value for ~1 s on a healthy hart: "stopped" waits 2.5 snapshot periods there."""
    m, now = chip_model(x280=False)
    x280_stream(m, now - 1_050_000_000, n=6, hz=2.0)  # the last fresh snapshot arrived 1.05 s ago
    last = dict(m.chip.get("x280d"))
    m.chip.ingest({**last, "seq": last["seq"] + 1, "t_ns": last["t_ns"] + 500_000_000,
                   "snapshot_age_ms": last["snapshot_age_ms"] + 500.0}, now - 550_000_000)  # re-sent 0.5 s later
    assert draw(pg, m, now)[1].row("heartbeat") == "+50.0 M/s"
    got = draw(pg, m, now + 300_000_000)[1].row("heartbeat")
    assert got.startswith("stopped 1.3"), got
