"""rtloop HUD: the live telemetry panel next to the MuJoCo viewer (PLAN.md §4.3, §1.4.5, WP-1.7).

A separate process (simpy venv, pygame, 1280x720, 30 fps, pinned to CPUs 6-7 by demo.sh) that SUBscribes to
rt_sim's PUB (multipart [topic, msgpack via rtloop.wire]; topics tick / req / ego / meta) and PUSHes operator
commands to the loop's control PULL (keys v / w / r / p; rtloop/control.py). It can never stall the
loop: the PUB drops at its HWM, and each frame drains at most ``--max-drain`` messages within
``--drain-budget-ms``, leaving the rest to the SUB HWM. A stalled or absent publisher only turns the status
pill to STALE / NO DATA; ZMQ reconnects by itself when a loop (re)binds.

The Blackhole chip panel (§4.3 item 7) comes from a second SUB on the device processes' out-of-band status
PUBs: x280d on :5573 and server_g1 on :5574, multipart [b"status", msgpack] at 2-10 Hz (INTERFACES.md, "Device
status streams"). The HUD never sends a request to :5557 or :5558 (a REP serves one request at a time, so a HUD
poll would delay a control call). Absent publishers are normal (M1 has none); a missing or malformed field shows
as "—", never as a made-up value. The panel replaces the timing histograms and the nav bars when a device
stream or a Tensix/X280 backend is seen (``--chip auto``), or always/never (``on``/``off``); key c toggles it.

The model (``HudModel``) is plain Python over the decoded messages and is tested without pygame; the view
(``HudView``) draws one frame onto any pygame Surface, so ``--headless`` (SDL_VIDEODRIVER=dummy) can render
frames to PNG. The processor map comes from the ``meta`` topic's ``processor`` stamps (backend ``info``);
a stock server without ``info`` ("policy-server") is placed by its port and marked "(cfg)". A run with nothing on
the RTX 5090 (no stamp and no swap target there: an X280 run) gets no 5090 box (``HudModel.gpu_in_run``).

    simpy $RT/scripts/hud.py                                   # live, SUB :5570 (+ :5573, :5574), PUSH :5571
    simpy $RT/scripts/hud.py --headless --frames 60 --png /tmp/hud.png --sub tcp://127.0.0.1:PORT
"""

import argparse
from collections import OrderedDict, deque
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RT = HERE.parent
for _p in (str(RT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rtloop import control as cproto  # noqa: E402
from rtloop import wire  # noqa: E402

TELEMETRY_PUB = os.environ.get("TELEMETRY_PUB", "tcp://127.0.0.1:5570")
CONTROL_PULL = os.environ.get("CONTROL_PULL", "tcp://127.0.0.1:5571")
X280_STATUS = os.environ.get("X280_STATUS", "tcp://127.0.0.1:5573")
TT_STATUS = os.environ.get("TT_STATUS", "tcp://127.0.0.1:5574")
W, H = 1280, 720
TICK_MS = 20.0
TIMELINE_TICKS = 200  # 4 s at 50 Hz (§4.3 item 5)
STATS_TICKS = 3000  # 60 s window for p99 / histograms
N_REQ = 50  # §4.3 item 2: VLA latency over the last 50 requests
LIVE_S = 0.5
FALL_Z = 0.50  # m, schema.FALL_PELVIS_Z
WBC_BE = {0: "ORT", 1: "GPU", 2: "X280", 3: "X280-sim"}
TERMINAL = ("LATE_REJECT", "GEN_DROP", "TIMEOUT", "ERROR", "DRAINED")
STATUS_SRCS = ("server_g1", "x280d")
STATUS_STALE_S = 2.0  # the streams run at 2-10 Hz
STATUS_MAX_BYTES = 64 * 1024  # a status message is ~1-3 KB; anything this large is not one
RATE_SPAN_S = 1.0  # counter rates (x280d heartbeat, calls, runs) over about this much of the publisher's clock
RATE_MIN_S = 0.05  # ... and never over less: snapshots are >= 100 ms apart at 2-10 Hz
SNAP_STALE_MS = 1000.0  # a snapshot sent this late was not refreshed by x280d's serving thread: no rate from it
HB_STOP_S = 1.0  # x280d's heartbeat is critical once it has not advanced for this long
GPU_NAME_RE = re.compile(r"gpu|cuda|5090", re.I)  # a swap target on the RTX 5090 (demo.sh names it "gpu")
TELEMETRY_KEYS = ("aiclk_mhz", "l2cpuclk_mhz", "asic_temp_c", "vcore_mv", "tdp_w", "tdc_a", "input_power_w",
                  "timer_heartbeat")
# A simulated publisher's telemetry is invented (kmd.FakeDevice), so it is never shown: x280d --backend sim stamps
# "x280-sim:...", server_g1 --fake-model "mock:server_g1-fake"; x280d may also name its source (telemetry_source).
SIM_BACKEND_RE = re.compile(r"(x280-sim|mock)(:|$)")
SIM_SOURCE_RE = re.compile(r"sim|fake|mock|synth", re.I)
CORE_ROLES = ("reader", "hub", "compute", "spare")
# NoC0 (x, y) of L2CPU0-3, used only when x280.regs is not importable; must equal regs.L2CPU_XY (FW noc_init.c:366-367
# kL2CpuNoc0Y = {3, 9, 5, 7}: tile order is not row order). tests/hud/test_hud_render.py checks it against regs.
L2CPU_NOC0_FALLBACK = {0: (8, 3), 1: (8, 9), 2: (8, 5), 3: (8, 7)}

# Dark chart chrome and roles (dataviz reference palette, dark mode). VLA = slot 1, WBC = slot 2.
C = {
    "page": (13, 13, 13), "panel": (26, 26, 25), "grid": (44, 44, 42), "axis": (56, 56, 53),
    "ink": (240, 239, 236), "ink2": (195, 194, 183), "muted": (137, 135, 129),
    "vla": (57, 135, 229), "wbc": (217, 89, 38), "aqua": (25, 158, 112),
    "good": (12, 163, 12), "warn": (250, 178, 25), "serious": (236, 131, 90), "crit": (208, 59, 59),
    "reader": (25, 158, 112), "hub": (250, 178, 25), "compute": (57, 135, 229), "spare": (82, 81, 78),
}


# --- model -------------------------------------------------------------------------------------


def _f(x, default=math.nan) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v


def _pct(xs, q) -> float:
    a = np.asarray([x for x in xs if x is not None and math.isfinite(x)], dtype=np.float64)
    return float(np.percentile(a, q)) if a.size else math.nan


def _port(url) -> int | None:
    m = re.search(r":(\d+)$", str(url or ""))
    return int(m.group(1)) if m else None


def classify(kind: str, backend, url=None) -> dict:
    """Where one backend runs, from its ``info`` stamp: {device: host|gpu|tensix|x280|?, label, cfg}.

    ``cfg`` marks a placement inferred from the configured port (stock server, no ``info``)."""
    name = str(backend or "")
    low = name.lower()
    cfg = False
    if low in ("", "none"):
        return {"device": "?", "label": f"{kind.upper()} ?", "cfg": False}
    if low == "policy-server":  # stock PolicyServer: no info endpoint, place it by port (D15)
        cfg = True
        port = _port(url)
        low = {5555: "cuda", 5565: "cuda", 5556: "cuda-graph", 5557: "tt", 5558: "x280"}.get(port, "?")
    if kind == "vla":
        head = low.split(":")[0]
        if low.startswith("mock"):
            dev, label = "host", "VLA mock"
        elif head == "tt" or "tensix" in low or head.startswith("tt-"):
            dev, label = "tensix", "VLA Tensix"
        elif low.startswith("cuda"):
            dev, label = "gpu", "VLA 5090" if "5090" in low or cfg else "VLA GPU"
        else:
            dev, label = "?", f"VLA {name}"
    else:
        if low.startswith("ort"):
            dev, label = "host", "WBC ORT CPU"
        elif low.startswith("x280-sim"):
            dev, label = "host", "WBC-MLP X280-sim"
        elif low.startswith("x280"):
            m = re.search(r"(\d+)\s*mhz", low)
            dev, label = "x280", f"WBC-MLP X280 @{m.group(1) if m else 800} MHz"
        elif low.startswith("cuda"):
            dev, label = "gpu", "WBC 5090" if "5090" in low or cfg else "WBC GPU"
        else:
            dev, label = "?", f"WBC {name}"
    if cfg:
        label += " (cfg)"
    return {"device": dev, "label": label, "cfg": cfg}


# --- Blackhole chip status (x280d :5573, server_g1 :5574) -------------------------------------------


def _num(v) -> float | None:
    """A finite real number, else None (bools, strings, NaN and containers are not numbers)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _int(v) -> int | None:
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else None


def _sub(d, key) -> dict:
    v = d.get(key) if isinstance(d, dict) else None
    return v if isinstance(v, dict) else {}


def _tiles(v) -> list[int]:
    """x280d's ``tile``: one index (3), or a list for DP4 fp32 ([3, 2]); [] if neither."""
    if _int(v) is not None:
        return [int(v)]
    if isinstance(v, (list, tuple)):
        return [int(t) for t in v if _int(t) is not None]
    return []


def telemetry_source(msg) -> tuple[str, bool]:
    """(label, simulated) of a status message's ``telemetry``: the label is the publisher's own ``telemetry_source``
    when it sends one, else "sim" for a simulated backend and "CMFW table" otherwise (the one real source, §4.3
    item 7). A simulated source never has CMFW telemetry, whatever numbers it sends."""
    msg = msg if isinstance(msg, dict) else {}
    own = msg.get("telemetry_source")
    own = own.strip() if isinstance(own, str) and own.strip() else None
    backend = msg.get("backend") if isinstance(msg.get("backend"), str) else ""
    sim = bool(SIM_BACKEND_RE.match(backend)) or bool(own and SIM_SOURCE_RE.search(own))
    return own or ("sim" if sim else "CMFW table"), sim


_X280_REFS: dict | None = None


def x280_refs() -> dict:
    """L2CPU NoC0 coordinates (x280/regs.py) and monitor state names (x280/fw/mailbox_layout.py): the only
    sources of those numbers. {} parts when the x280 tree is not importable (the panel then shows indices)."""
    global _X280_REFS
    if _X280_REFS is None:
        out = {"xy": {}, "states": {}}
        try:
            from x280 import regs

            out["xy"] = {int(k): tuple(v) for k, v in regs.L2CPU_XY.items()}
            out["default_tile"] = int(regs.DEFAULT_TILE)
            out["spare_tile"] = int(regs.SPARE_TILE)
        except Exception:  # noqa: BLE001 - display only
            pass
        try:
            from x280.fw import mailbox_layout as ml

            out["states"] = {int(v): k.lower() for k, v in ml.STATE.items()}
        except Exception:  # noqa: BLE001
            pass
        _X280_REFS = out
    return _X280_REFS


def _mon_name(v) -> str | None:
    """x280d's monitor_state as a name (x280d sends the name; a number is looked up in mailbox_layout.STATE)."""
    if _int(v) is not None:
        return x280_refs()["states"].get(v)
    return v if isinstance(v, str) else None


def hart_note(xs: dict, rate: float | None) -> str | None:
    """What x280d's hart is doing: "serving 50/s" while its tile serves calls (``rate`` calls/s) and its monitor is
    idle or busy (a 5 Hz sample of a 0.28 ms call every 20 ms almost always finds it idle), else the monitor state's
    name (fault, parked, ...), None if unknown."""
    name = _mon_name(xs.get("monitor_state"))
    if rate is not None and rate >= 0.5 and name in (None, "idle", "busy") and xs.get("tile_state") != "burned":
        return f"serving {rate:.0f}/s"
    return name


def l2cpu_xy(tile: int) -> tuple[int, int]:
    """NoC0 (x, y) of L2CPU<tile>: x280/regs.py's, else the pinned copy L2CPU_NOC0_FALLBACK (display only)."""
    return tuple((x280_refs().get("xy") or {}).get(tile, L2CPU_NOC0_FALLBACK[tile]))


_MK_DEFAULT: dict | None = None


def mk_default_core_map() -> dict:
    """MK's own default core map (tt/megakernel/core_map.py default_core_map(): the measured DRAM workers of this
    p150a), loaded by path from $MK (pure Python, no ttnn). {} if MK is not there. Used only to place the roles
    when server_g1 reports counts without coordinates, and only if its counts and grid agree."""
    global _MK_DEFAULT
    if _MK_DEFAULT is None:
        _MK_DEFAULT = {}
        mk = Path(os.environ.get("MK") or Path.home() / "experiments/gr00t/publish/worktree-mk")
        path = mk / "models/experimental/gr00t/tt/megakernel/core_map.py"
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_hud_mk_core_map", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod  # dataclasses resolve the module by name
            spec.loader.exec_module(mod)
            _MK_DEFAULT = mod.default_core_map().manifest()
        except Exception:  # noqa: BLE001 - display only
            _MK_DEFAULT = {}
    return _MK_DEFAULT


def _coords(v) -> list[tuple[int, int]] | None:
    """[[x, y], ...] or [x, y] -> [(x, y), ...]; None if it is not a coordinate list."""
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(_int(c) is not None for c in v):
        return [(int(v[0]), int(v[1]))]
    if isinstance(v, (list, tuple)) and v and all(
            isinstance(c, (list, tuple)) and len(c) == 2 and all(_int(a) is not None for a in c) for c in v):
        return [(int(c[0]), int(c[1])) for c in v]
    return None


def core_layout(core_map) -> dict:
    """server_g1's static core_map -> {grid, counts {role: n}, cells {(x, y): role} or {}, source}.

    Accepts the contract's counts ({"grid": [11, 10], "megakernel": {"readers": 8, ...}}) and coordinates
    (readers/hub/compute/spare as [x, y] lists, at the top or under "megakernel": MK's CoreMap.manifest(), or
    rttt/server_g1.py's core_map_dict, which lists readers, hub and spares and leaves compute = the rest of the grid).
    With counts only, the roles are placed from MK's default map when its grid and counts agree."""
    if not isinstance(core_map, dict):
        return {}
    grid = core_map.get("grid")
    if not (isinstance(grid, (list, tuple)) and len(grid) == 2 and all(_int(g) and 0 < g <= 32 for g in grid)):
        return {}
    gx, gy = int(grid[0]), int(grid[1])
    mk = _sub(core_map, "megakernel")
    counts, cells = {}, {}
    for role in CORE_ROLES:
        for key in (role + "s", role):
            v = core_map.get(key, mk.get(key))
            if v is None:
                continue
            xy = _coords(v)
            if xy is not None:
                counts[role] = len(xy)
                cells.update({c: role for c in xy if 0 <= c[0] < gx and 0 <= c[1] < gy})
            elif _int(v) is not None:
                counts[role] = int(v)
            break
    rest = [(x, y) for y in range(gy) for x in range(gx) if (x, y) not in cells]
    if cells and "compute" not in set(cells.values()) and counts.get("compute") == len(rest):
        cells.update({c: "compute" for c in rest})  # server_g1: the compute cores are the rest of the grid
    source = "reported" if cells else None
    if not cells and counts:
        d = mk_default_core_map()
        if d and list(d.get("grid", ())) == [gx, gy]:
            dc = {r: _coords(d.get(r + "s", d.get(r))) or [] for r in CORE_ROLES}
            if all(len(dc[r]) == counts.get(r, len(dc[r])) for r in CORE_ROLES):
                cells = {c: r for r in CORE_ROLES for c in dc[r]}
                source = "MK default_core_map"
    return {"grid": (gx, gy), "counts": counts, "cells": cells, "source": source}


def _snap_ns(msg) -> int | None:
    """When the publisher took the snapshot (its CLOCK_MONOTONIC): t_ns, less x280d's snapshot_age_ms."""
    t = _int(msg.get("t_ns"))
    age = _num(msg.get("snapshot_age_ms"))
    if t is None or age is None or age < 0:
        return t
    return t - int(age * 1e6)


class ChipModel:
    """The latest status of each Blackhole process, from its out-of-band PUB (PLAN.md §4.3 item 7).

    Keeps the last well-formed message per ``src`` as received (fields are type-checked when shown), plus receive
    time, message count, seq gaps (dropped by the SNDHWM 10 PUB) and publisher restarts (pid changes). Counters are
    turned into rates over consecutive snapshots (``rate``), and the receive time at which x280d's raw heartbeat
    last advanced is kept: a single snapshot of a deterministic kernel says little, the movement says it is alive."""

    def __init__(self):
        self.last: dict[str, dict] = {}
        self.t_rx: dict[str, int] = {}
        self.rx = {s: 0 for s in STATUS_SRCS}
        self.gaps = {s: 0 for s in STATUS_SRCS}
        self.restarts = {s: 0 for s in STATUS_SRCS}
        self.bad = 0
        self.hist = {s: deque(maxlen=16) for s in STATUS_SRCS}  # (snapshot ns, msg), one per distinct snapshot
        self.hb_moved: dict[str, int] = {}  # receive ns of the last message whose heartbeat differed

    def ingest(self, msg, t_ns: int | None = None) -> bool:
        t_ns = time.monotonic_ns() if t_ns is None else t_ns
        src = msg.get("src") if isinstance(msg, dict) else None
        if src not in STATUS_SRCS:
            self.bad += 1
            return False
        prev = self.last.get(src)
        restarted = False
        if prev is not None:
            pid, seq = _int(msg.get("pid")), _int(msg.get("seq"))
            if pid is not None and _int(prev.get("pid")) is not None and pid != prev["pid"]:
                self.restarts[src] += 1
                restarted = True
            elif seq is not None and _int(prev.get("seq")) is not None and seq > prev["seq"] + 1:
                self.gaps[src] += seq - prev["seq"] - 1
        hist = self.hist[src]
        if restarted:
            hist.clear()
        t_snap = _snap_ns(msg)
        if t_snap is not None and (not hist or t_snap > hist[-1][0]):  # a re-sent snapshot is no new sample
            hist.append((t_snap, msg))
        hb = _int(msg.get("heartbeat"))
        if hb is not None and (prev is None or restarted or hb != _int(prev.get("heartbeat"))):
            self.hb_moved[src] = t_ns
        self.last[src] = msg
        self.t_rx[src] = t_ns
        self.rx[src] += 1
        return True

    def rate(self, src: str, key: str, tile: str | None = None) -> float | None:
        """Per-second increase of the counter ``key`` (under ``tiles[tile]`` if given) over the last ~RATE_SPAN_S of
        snapshots, on the publisher's clock at the moment the values were read. None without two samples of it, for
        a counter that only went backwards, or while the newest snapshot was sent stale (x280d did not refresh it)."""
        hist = self.hist.get(src) or ()
        if len(hist) < 2:
            return None

        def val(msg):
            return _int((_sub(_sub(msg, "tiles"), tile) if tile is not None else msg).get(key))

        t1, m1 = hist[-1]
        v1 = val(m1)
        late = _num(self.get(src).get("snapshot_age_ms"))
        if v1 is None or (late is not None and late > SNAP_STALE_MS):
            return None
        for t0, m0 in hist:  # the oldest sample inside the span
            v0 = val(m0)
            if v0 is not None and v0 <= v1 and RATE_MIN_S * 1e9 <= t1 - t0 <= RATE_SPAN_S * 1.25e9:
                return (v1 - v0) * 1e9 / (t1 - t0)
        return None

    def serving(self, now_ns: int) -> dict[int, float]:
        """Calls per second each x280d tile is serving while its stream is LIVE: Δn_runs of the tile's monitor (the
        top-level tile falls back to x280d's own calls_ok); tiles with no rate are left out."""
        msg = self.get("x280d")
        if self.state("x280d", now_ns)[0] != "LIVE":
            return {}
        tiles = _tiles(msg.get("tile"))
        out = {}
        for i, t in enumerate(tiles + [int(k) for k in _sub(msg, "tiles") if str(k).isdigit()]):
            r = self.rate("x280d", "n_runs") if i == 0 else self.rate("x280d", "n_runs", str(t))
            if i == 0 and r is None:
                r = self.rate("x280d", "calls_ok")
            if r is not None and t not in out:
                out[t] = r
        return out

    def ingest_raw(self, parts, t_ns: int | None = None) -> bool:
        """One multipart message as received: [b"status", msgpack]."""
        if len(parts) != 2 or bytes(parts[0]) != b"status" or len(parts[1]) > STATUS_MAX_BYTES:
            self.bad += 1
            return False
        try:
            msg = wire.loads(parts[1])
        except Exception:  # noqa: BLE001 - a garbled frame must never kill the HUD
            self.bad += 1
            return False
        return self.ingest(msg, t_ns)

    def state(self, src: str, now_ns: int) -> tuple[str, float | None]:
        """LIVE | STALE | NO DATA for one source, and the seconds since its last message."""
        t = self.t_rx.get(src)
        if t is None:
            return "NO DATA", None
        age = (now_ns - t) / 1e9
        return ("LIVE" if age < STATUS_STALE_S else "STALE"), age

    def get(self, src: str) -> dict:
        return self.last.get(src) or {}

    def seen_any(self) -> bool:
        return bool(self.last)

    def telemetry(self, now_ns: int) -> tuple[dict, str | None]:
        """The CMFW telemetry of the freshest live, non-simulated source that reports any (x280d first on a tie)."""
        best, src_best = {}, None
        for src in ("x280d", "server_g1"):
            msg = self.get(src)
            tel = _sub(msg, "telemetry")
            if (not tel or self.state(src, now_ns)[0] != "LIVE" or telemetry_source(msg)[1]
                    or not any(tel.get(k) is not None for k in TELEMETRY_KEYS)):
                continue
            if src_best is None or self.t_rx[src] > self.t_rx[src_best]:
                best, src_best = tel, src
        return best, src_best

    def telemetry_row(self, now_ns: int) -> dict:
        """What the telemetry row shows: {tel, src, source} with the numbers and their source label, or {note}
        saying why there are none: no live source, or "no CMFW telemetry (<why>)" for the live ones (x280d's
        reason first: it is the only process that can read the table)."""
        tel, src = self.telemetry(now_ns)
        if src is not None:
            return {"tel": tel, "src": src, "source": telemetry_source(self.get(src))[0], "note": None}
        live = [s for s in ("x280d", "server_g1") if self.state(s, now_ns)[0] == "LIVE"]
        if not live:
            return {"tel": {}, "src": None, "source": None, "note": "chip telemetry (CMFW table): no live source"}
        msg = self.get(live[0])
        if telemetry_source(msg)[1]:
            why = "sim"
        elif live[0] == "server_g1":
            why = "server_g1 has no CMFW read"
        elif isinstance(msg.get("telemetry_error") or msg.get("error"), str):
            why = "x280d read failed"
        else:
            why = "x280d reports none"
        return {"tel": {}, "src": None, "source": None, "note": f"no CMFW telemetry ({why})"}

    def summary(self) -> dict:
        return {"rx": dict(self.rx), "bad": self.bad, "gaps": dict(self.gaps), "restarts": dict(self.restarts)}


class HudModel:
    """Everything the HUD shows, folded from the telemetry stream (no pygame, no sockets)."""

    def __init__(self):
        self.ticks: deque = deque(maxlen=TIMELINE_TICKS)
        self.period: deque = deque(maxlen=STATS_TICKS)
        self.compute: deque = deque(maxlen=STATS_TICKS)
        self.rtt: deque = deque(maxlen=STATS_TICKS)
        self.reqs: deque = deque(maxlen=N_REQ)
        self.req_by_seq: OrderedDict = OrderedDict()
        self.req_bars: deque = deque(maxlen=64)  # (ep, k_obs, k_arr, outcome)
        self.ego: OrderedDict = OrderedDict()  # seq -> (k_obs, img)
        self.ego_latest = None
        self.meta: dict = {}
        self.last: dict | None = None
        self.rx = {"tick": 0, "req": 0, "ego": 0, "meta": 0, "other": 0}
        self.t_rx: dict[str, int] = {}
        self.bad = 0
        self.dropped_ticks = 0
        self.n_fallback = self.n_overrun = self.n_slip = self.n_starved = 0
        self.starve_run = self.max_starve_run = 0
        self.shadow_max = 0.0
        self.outcomes: dict[str, int] = {}
        self.episodes = self.successes = self.falls = 0
        self.ep_success = False
        self.inflight_since: int | None = None
        self.chip = ChipModel()

    # -- ingest

    def ingest(self, topic, msg, t_ns: int | None = None) -> None:
        t_ns = time.monotonic_ns() if t_ns is None else t_ns
        topic = topic.decode() if isinstance(topic, (bytes, bytearray)) else str(topic)
        if not isinstance(msg, dict):
            self.bad += 1
            return
        handler = {"tick": self._tick, "req": self._req, "ego": self._ego, "meta": self._meta}.get(topic)
        if handler is None:
            self.rx["other"] += 1
            return
        try:
            handler(msg)
        except (KeyError, TypeError, ValueError):
            self.bad += 1
            return
        self.rx[topic] += 1
        self.t_rx[topic] = t_ns

    def ingest_raw(self, parts, t_ns: int | None = None) -> None:
        """One multipart message as received: [topic, msgpack]."""
        if len(parts) != 2:
            self.bad += 1
            return
        try:
            msg = wire.loads(parts[1])
        except Exception:  # noqa: BLE001 - a garbled frame must never kill the HUD
            self.bad += 1
            return
        self.ingest(parts[0], msg, t_ns)

    def _end_episode(self, last: dict) -> None:
        self.episodes += 1
        if self.ep_success:
            self.successes += 1
        elif _f(last.get("pelvis_z"), 1.0) < FALL_Z:
            self.falls += 1
        self.ep_success = False
        self.inflight_since = None

    def _tick(self, m: dict) -> None:
        k, ep = int(m["k"]), int(m.get("ep", 0))
        last = self.last
        if last is not None:
            if ep != last["ep"]:
                self._end_episode(last)
            elif k > last["k"] + 1:
                self.dropped_ticks += k - last["k"] - 1
        t = {
            "k": k, "ep": ep, "period_us": _f(m.get("period_us")), "compute_us": _f(m.get("compute_us")),
            "late_us": _f(m.get("late_us")), "slack_us": _f(m.get("slack_us")),
            "overrun": int(m.get("overrun", 0)), "slip": int(m.get("slip", 0)),
            "sim_us": _f(m.get("sim_us")), "viewer_us": _f(m.get("viewer_us")),
            "wbc_rtt_us": _f(m.get("wbc_rtt_us")), "wbc_dev_us": _f(m.get("wbc_dev_us")),
            "wbc_be": int(m.get("wbc_be", 0)), "wbc_pol": int(m.get("wbc_pol", 0)),
            "wbc_fb": int(m.get("wbc_fb", 0)), "shadow_da": _f(m.get("shadow_da")),
            "seq": int(m.get("seq", 0)), "row": int(m.get("row", -1)), "rows_left": int(m.get("rows_left", 0)),
            "starved": int(m.get("starved", 0)), "inflight": int(m.get("inflight", 0)),
            "splice": int(m.get("splice", 0)), "jump_rad": _f(m.get("jump_rad")),
            "nav": [_f(v, 0.0) for v in (m.get("nav") or [0, 0, 0])][:3],
            "pelvis_z": _f(m.get("pelvis_z")), "success": int(m.get("success", 0)),
        }
        self.ticks.append(t)
        if k > 0 and t["period_us"] > 0:
            self.period.append(t["period_us"])
        self.compute.append(t["compute_us"])
        self.rtt.append(t["wbc_rtt_us"])
        self.n_fallback += t["wbc_fb"]
        self.n_overrun += t["overrun"]
        self.n_slip += t["slip"]
        if t["starved"]:
            self.n_starved += 1
            self.starve_run += 1
            self.max_starve_run = max(self.max_starve_run, self.starve_run)
        else:
            self.starve_run = 0
        if math.isfinite(t["shadow_da"]):
            self.shadow_max = max(self.shadow_max, t["shadow_da"])
        if t["success"]:
            self.ep_success = True
        if t["inflight"] and (last is None or not last.get("inflight") or ep != last["ep"]):
            self.inflight_since = k
        elif not t["inflight"]:
            self.inflight_since = None
        self.last = t

    def _req(self, m: dict) -> None:
        r = {
            "seq": int(m["seq"]), "gen": int(m.get("gen", 0)), "ep": int(m.get("ep", 0)),
            "k_obs": int(m.get("k_obs", -1)), "k_arr": int(m.get("k_arr", -1)), "d": int(m.get("d", -1)),
            "d_hat": m.get("d_hat"), "s_eff": m.get("s_eff"), "rtt_ms": _f(m.get("rtt_ms")),
            "ser_ms": _f(m.get("ser_ms")), "srv": m.get("srv") if isinstance(m.get("srv"), dict) else {},
            "outcome": str(m.get("outcome", "?")),
        }
        self.reqs.append(r)
        self.req_by_seq[r["seq"]] = r
        while len(self.req_by_seq) > 128:
            self.req_by_seq.popitem(last=False)
        self.req_bars.append((r["ep"], r["k_obs"], r["k_arr"], r["outcome"]))
        self.outcomes[r["outcome"]] = self.outcomes.get(r["outcome"], 0) + 1

    def _ego(self, m: dict) -> None:
        img = m["img"]
        if not (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
            raise ValueError("ego img must be uint8 (h, w, 3)")
        seq = int(m.get("seq", 0))
        self.ego[seq] = (int(m.get("k_obs", -1)), img)
        while len(self.ego) > 8:
            self.ego.popitem(last=False)
        self.ego_latest = seq

    def _meta(self, m: dict) -> None:
        self.meta = dict(m)

    # -- derived

    def status(self, now_ns: int) -> tuple[str, float | None]:
        """LIVE | PAUSED | STALE | NO DATA, and seconds since the last message of any topic."""
        if not self.t_rx:
            return "NO DATA", None
        age = (now_ns - max(self.t_rx.values())) / 1e9
        t_tick = self.t_rx.get("tick")
        tick_age = (now_ns - t_tick) / 1e9 if t_tick is not None else None
        if self.meta.get("paused") and age < 2.0:
            return "PAUSED", tick_age
        if tick_age is not None and tick_age < LIVE_S:
            return "LIVE", tick_age
        return "STALE", tick_age if tick_age is not None else age

    def processors(self) -> dict:
        """The processor map: vla / wbc placements plus what computed the latest tick."""
        proc = self.meta.get("processor") or {}
        ends = self.meta.get("endpoints") or {}
        vla = classify("vla", proc.get("vla"), ends.get("vla"))
        wbc = classify("wbc", proc.get("wbc"), ends.get("wbc"))
        t = self.last
        now_be = WBC_BE.get(t["wbc_be"], "?") if t else None
        fallback = bool(t and t["wbc_fb"])
        caption = f"{vla['label']} / {wbc['label']}"
        if wbc["device"] == "x280" and t and math.isfinite(t["wbc_dev_us"]):
            caption += f", {t['wbc_dev_us']:.0f} µs (mcycle)"
        return {"vla": vla, "wbc": wbc, "wbc_now": now_be, "fallback": fallback, "caption": caption,
                "have_meta": bool(proc)}

    def gpu_in_run(self) -> bool:
        """Whether this run puts anything on the RTX 5090: the VLA or WBC in use (meta ``processor``) or a swap
        target (meta ``targets``, the --vla-alt / --wbc-alt names). From the run's configuration, not from a tick, so
        the processor map keeps one layout for the whole run (a swap only moves between configured targets). False
        before any meta: an X280 run never shows the GPU, not even while the loop is starting."""
        if not self.meta.get("processor"):
            return False
        p = self.processors()
        if "gpu" in (p["vla"]["device"], p["wbc"]["device"]):
            return True
        targets = self.meta.get("targets") if isinstance(self.meta.get("targets"), dict) else {}
        names = [n for k in ("vla", "wbc") if isinstance(targets.get(k), (list, tuple)) for n in targets[k]]
        return any(GPU_NAME_RE.search(str(n)) for n in names)

    def active_req(self) -> dict | None:
        t = self.last
        return self.req_by_seq.get(t["seq"]) if t else None

    def ego_image(self):
        """(seq, k_obs, img) of the chunk being executed, else the newest thumbnail."""
        t = self.last
        if t and t["seq"] in self.ego:
            k_obs, img = self.ego[t["seq"]]
            return t["seq"], k_obs, img
        if self.ego_latest is not None and self.ego_latest in self.ego:
            k_obs, img = self.ego[self.ego_latest]
            return self.ego_latest, k_obs, img
        return None

    def vla_stats(self) -> dict:
        ok = [r for r in self.reqs if r["outcome"] == "APPLIED"]
        rtts = [r["rtt_ms"] for r in ok]
        infer = [_f(r["srv"].get("infer_ms")) for r in ok]
        split = {k: _pct([_f(r["srv"].get(f"{k}_ms")) for r in ok], 50) for k in ("encode", "device", "decode")}
        ds = [r["d"] for r in ok if r["d"] >= 0]
        last = self.reqs[-1] if self.reqs else None
        return {"last_ms": last["rtt_ms"] if last else math.nan, "p50_ms": _pct(rtts, 50),
                "p99_ms": _pct(rtts, 99), "infer_p50_ms": _pct(infer, 50), "split_p50_ms": split,
                "d_p50": _pct(ds, 50),
                "d_hat": last.get("d_hat") if last else None, "s_eff": last.get("s_eff") if last else None,
                "n": len(self.reqs)}

    def tick_stats(self) -> dict:
        per = np.asarray(self.period, dtype=np.float64) / 1e3
        per = per[np.isfinite(per)]
        return {"period_p99_ms": _pct(per, 99), "period_max_ms": float(per.max()) if per.size else math.nan,
                "jitter_ms": float(per.std()) if per.size > 1 else math.nan,
                "compute_p99_ms": _pct(self.compute, 99) / 1e3, "rtt_p99_us": _pct(self.rtt, 99),
                "rtt_p50_us": _pct(self.rtt, 50)}

    def summary(self) -> dict:
        return {"rx": dict(self.rx), "bad": self.bad, "dropped_ticks": self.dropped_ticks,
                "last_k": self.last["k"] if self.last else None, "last_ep": self.last["ep"] if self.last else None,
                "episodes": self.episodes, "successes": self.successes, "falls": self.falls,
                "fallbacks": self.n_fallback, "starved": self.n_starved, "chip": self.chip.summary()}


# --- view --------------------------------------------------------------------------------------


FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


class HudView:
    """Draws one frame of a HudModel onto a 1280x720 pygame Surface."""

    def __init__(self, pygame, chip: str = "auto"):
        self.pg = pygame
        self._fonts = {}
        self._text_cache: dict = {}
        self._ego_cache = (None, None)
        self.toast: tuple[str, tuple, float] | None = None
        self.chip = chip  # auto | on | off (key c flips it to on/off)

    def font(self, size: int, mono: bool = False, bold: bool = False):
        key = (size, mono, bold)
        f = self._fonts.get(key)
        if f is None:
            name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
            path = FONT_DIR / name
            f = self.pg.font.Font(str(path) if path.exists() else None, size)
            self._fonts[key] = f
        return f

    def text(self, surf, s, pos, size=15, color="ink", mono=False, bold=False, anchor="topleft"):
        col = C[color] if isinstance(color, str) else color
        key = (s, size, mono, bold, col)
        img = self._text_cache.get(key)
        if img is None:
            if len(self._text_cache) > 1500:
                self._text_cache.clear()
            img = self.font(size, mono, bold).render(str(s), True, col)
            self._text_cache[key] = img
        r = img.get_rect(**{anchor: pos})
        surf.blit(img, r)
        return r

    def panel(self, surf, rect, title=None):
        pg = self.pg
        pg.draw.rect(surf, C["panel"], rect, border_radius=6)
        if title:
            self.text(surf, title, (rect[0] + 12, rect[1] + 8), 13, "muted", bold=True)

    def pill(self, surf, s, pos, fill, ink="page", size=14, anchor="topleft"):
        img = self.font(size, bold=True).render(s, True, C[ink] if isinstance(ink, str) else ink)
        r = img.get_rect()
        box = self.pg.Rect(0, 0, r.w + 16, r.h + 6)
        setattr(box, anchor, pos)
        self.pg.draw.rect(surf, C[fill] if isinstance(fill, str) else fill, box, border_radius=box.h // 2)
        surf.blit(img, img.get_rect(center=box.center))
        return box

    # -- frame

    def chip_shown(self, m: HudModel) -> bool:
        """auto: once a device status stream arrived, or the processor map puts a backend on Tensix or the X280."""
        if self.chip in ("on", "off"):
            return self.chip == "on"
        p = m.processors()
        return m.chip.seen_any() or p["vla"]["device"] == "tensix" or p["wbc"]["device"] == "x280"

    def toggle_chip(self, m: HudModel) -> None:
        self.chip = "off" if self.chip_shown(m) else "on"

    def draw(self, surf, m: HudModel, now_ns: int, fps: float = 0.0, control: str = "") -> None:
        surf.fill(C["page"])
        self._top(surf, m, now_ns, fps)
        self._processors(surf, m, (16, 52, 420, 236), now_ns)
        self._vla(surf, m, (16, 296, 420, 170))
        self._wbc(surf, m, (16, 474, 420, 170))
        self._chunk(surf, m, (452, 52, 400, 132))
        if self.chip_shown(m):  # §4.3 item 7 in place of the histograms and the nav bars
            self._chip(surf, m, (452, 192, 400, 452), now_ns)
        else:
            self._timing(surf, m, (452, 192, 400, 274))
            self._nav(surf, m, (452, 474, 400, 170))
        self._ego(surf, m, (868, 52, 396, 340))
        self._episode(surf, m, (868, 400, 396, 244), now_ns, control)
        self._timeline(surf, m, (16, 652, 1248, 60))

    def _top(self, surf, m, now_ns, fps):
        meta = m.meta
        mode = meta.get("mode", "?")
        t = m.last
        where = f"ep {t['ep']}  k {t['k']}" if t else "no ticks yet"
        seed = meta.get("seed")
        if seed is not None:
            where += f"  seed {seed}"
        self.text(surf, "G1 apple→plate  ·  rtloop HUD", (16, 12), 20, "ink", bold=True)
        rtf = meta.get("rt_factor")
        rtf = f"   rt ×{_f(rtf):.3f}" if rtf is not None and math.isfinite(_f(rtf)) else ""
        self.text(surf, f"mode {mode}{rtf}   {where}", (360, 15), 16, "ink2", mono=True)
        status, age = m.status(now_ns)
        fill = {"LIVE": "good", "PAUSED": "warn", "STALE": "warn", "NO DATA": "crit"}[status]
        label = status if status in ("LIVE", "PAUSED") or age is None else f"{status} {age:.1f} s"
        if status == "NO DATA":
            label = "NO DATA — no publisher"
        box = self.pill(surf, label, (1264, 12), fill, anchor="topright")
        self.text(surf, f"HUD {fps:4.1f} fps", (box.left - 12, 16), 13, "muted", mono=True, anchor="topright")

    def _processors(self, surf, m, rect, now_ns):
        """Host CPU, RTX 5090 and Blackhole boxes. A run with nothing on the GPU (m.gpu_in_run(): e.g. demo.sh m3
        --no-alt) has no 5090 box: the Blackhole box takes its place and gives each engine a line with its live state."""
        pg = self.pg
        x, y, w, h = rect
        self.panel(surf, rect, "PROCESSOR MAP")
        p = m.processors()
        vla_dev, wbc_dev = p["vla"]["device"], p["wbc"]["device"]
        fb = p["fallback"]
        bh = [("Tensix: VLA", vla_dev == "tensix", "vla"), ("X280: WBC-MLP", wbc_dev == "x280" and not fb, "wbc")]
        rows = [
            ("Host CPU · Ryzen 9 7950X", [("sim + splicer", True, "ink2"),
                                                ("VLA", vla_dev == "host", "vla"),
                                                ("ORT fallback" if fb else "WBC", fb or wbc_dev == "host",
                                                 "serious" if fb else "wbc")]),
        ]
        gpu = m.gpu_in_run()
        if gpu:
            rows += [("RTX 5090", [("VLA", vla_dev == "gpu", "vla"), ("WBC", wbc_dev == "gpu" and not fb, "wbc")]),
                     ("Blackhole p150a", bh)]
        by = y + 30
        for name, chips in rows:
            box = pg.Rect(x + 12, by, w - 24, 46)
            lit_any = name.startswith("Host") or any(lit for _, lit, _ in chips)
            pg.draw.rect(surf, C["axis"] if lit_any else C["grid"], box, width=1, border_radius=5)
            self.text(surf, name, (box.x + 10, box.y + 4), 12, "ink2" if lit_any else "muted")
            cx = box.x + 10
            for label, lit, role in chips:
                img_w = self.font(12, bold=True).size(label)[0] + 16
                cx = self._proc_chip(surf, pg.Rect(cx, box.y + 22, img_w, 19), label, lit, role).right + 8
            by += 52
        if not gpu:  # the 5090's row and the Blackhole row as one box, an engine per line
            box = pg.Rect(x + 12, by, w - 24, 98)
            lit_any = any(lit for _, lit, _ in bh)
            pg.draw.rect(surf, C["axis"] if lit_any else C["grid"], box, width=1, border_radius=5)
            self.text(surf, "Blackhole p150a", (box.x + 10, box.y + 4), 12, "ink2" if lit_any else "muted")
            img_w = max(self.font(12, bold=True).size(label)[0] for label, _, _ in bh) + 16
            for i, ((label, lit, role), note) in enumerate(zip(bh, self._bh_notes(m, p, now_ns))):
                chip = self._proc_chip(surf, pg.Rect(box.x + 10, box.y + 25 + 34 * i, img_w, 19), label, lit, role)
                if lit and note:
                    self.text(surf, note, (chip.right + 10, chip.centery), 12, "ink2", anchor="midleft")
        cap = p["caption"] if p["have_meta"] else "waiting for meta (processor stamps)"
        col = "ink" if p["have_meta"] else "muted"
        if self.font(15, bold=True).size(cap)[0] <= w - 24:
            self.text(surf, cap, (x + 12, y + h - 26), 15, col, bold=True)
        else:  # e.g. "VLA Tensix / WBC-MLP X280 @800 MHz, 410 us (mcycle)": one backend per line
            head, sep, tail = cap.partition(" / ")
            self.text(surf, head + (" /" if sep else ""), (x + 12, y + h - 44), 14, col, bold=True)
            self.text(surf, tail, (x + 12, y + h - 25), 14, col, bold=True)

    def _proc_chip(self, surf, rect, label, lit, role):
        if lit:
            self.pg.draw.rect(surf, C[role], rect, border_radius=4)
            self.text(surf, label, rect.center, 12, "page", bold=True, anchor="center")
        else:
            self.pg.draw.rect(surf, C["grid"], rect, width=1, border_radius=4)
            self.text(surf, label, rect.center, 12, "muted", anchor="center")
        return rect

    def _bh_notes(self, m, p, now_ns) -> list[str]:
        """The two-line Blackhole box's live notes (Tensix, X280), from the status streams: server_g1's device p50
        and calls/s, the X280 tile and what its hart does. Without a live stream, the loop's own round trip p50."""
        chip, f = m.chip, self._fmt
        if chip.state("server_g1", now_ns)[0] == "LIVE":
            dev = _sub(chip.get("server_g1"), "p50_ms").get("device")
            cps = chip.rate("server_g1", "calls")
            tensix = " · ".join(([f"device p50 {f(dev)} ms"] if _num(dev) is not None else []) +
                                ([f"{cps:.1f} calls/s"] if cps is not None else []))
        else:
            v = m.vla_stats()["p50_ms"]
            tensix = f"RTT p50 {v:.1f} ms" if math.isfinite(v) else ""
        xs = chip.get("x280d")
        tiles = _tiles(xs.get("tile"))
        if chip.state("x280d", now_ns)[0] == "LIVE" and tiles:
            hart = hart_note(xs, chip.serving(now_ns).get(tiles[0]))
            x280 = " · ".join([",".join(f"L2CPU{t}" for t in tiles)] + ([hart] if hart else []))
        else:
            r = m.tick_stats()["rtt_p50_us"]
            x280 = f"RTT p50 {r:.0f} µs" if math.isfinite(r) else ""
        return [tensix, x280]

    def _kv(self, surf, x, y, key, val, color="ink", w=190):
        self.text(surf, key, (x, y), 13, "muted")
        self.text(surf, val, (x + w - 12, y), 14, color, mono=True, anchor="topright")

    def _vla(self, surf, m, rect):
        x, y, w, h = rect
        self.panel(surf, rect, "VLA")
        p = m.processors()
        s = m.vla_stats()
        self.text(surf, p["vla"]["label"], (x + 60, y + 7), 13, "vla", bold=True)
        ar = m.active_req()

        def ms(v):
            return "—" if not math.isfinite(v) else f"{v:6.1f} ms"

        c1, c2, ry = x + 12, x + 214, y + 32
        self._kv(surf, c1, ry, "RTT last", ms(s["last_ms"]))
        self._kv(surf, c1, ry + 22, "RTT p50 (50)", ms(s["p50_ms"]))
        self._kv(surf, c1, ry + 44, "RTT p99 (50)", ms(s["p99_ms"]))
        sp = s["split_p50_ms"]
        if all(math.isfinite(sp[k]) for k in ("encode", "device", "decode")):  # server_g1 reports the split
            self._kv(surf, c1, ry + 66, "srv e/d/d p50", f"{sp['encode']:.1f}/{sp['device']:.1f}/{sp['decode']:.1f}")
        else:
            self._kv(surf, c1, ry + 66, "server infer p50", ms(s["infer_p50_ms"]))
        d = ar["d"] if ar else None
        self._kv(surf, c2, ry, "d (active)", "—" if d is None else f"{d}", w=194)
        self._kv(surf, c2, ry + 22, "d̂ / s_eff", f"{s['d_hat'] if s['d_hat'] is not None else '-'} / "
                 f"{s['s_eff'] if s['s_eff'] is not None else '-'}", w=194)
        oc = m.outcomes
        self._kv(surf, c2, ry + 44, "late / timeout", f"{oc.get('LATE_REJECT', 0)} / {oc.get('TIMEOUT', 0)}",
                 "serious" if oc.get("LATE_REJECT") or oc.get("TIMEOUT") else "ink", w=194)
        self._kv(surf, c2, ry + 66, "error / gen drop", f"{oc.get('ERROR', 0)} / {oc.get('GEN_DROP', 0)}",
                 "crit" if oc.get("ERROR") else "ink", w=194)
        t = m.last
        busy = bool(t and t["inflight"])
        cx, cy = x + 24, y + h - 26
        if busy:
            ang = (time.monotonic() * 6.0) % (2 * math.pi)
            for i in range(8):
                a = ang + i * math.pi / 4
                col = tuple(int(c * (0.25 + 0.75 * i / 7)) for c in C["vla"])
                self.pg.draw.circle(surf, col, (int(cx + 8 * math.cos(a)), int(cy + 8 * math.sin(a))), 2)
            since = m.inflight_since
            age = f" for {(t['k'] - since) * TICK_MS:.0f} ms" if since is not None else ""
            self.text(surf, f"request in flight{age}", (cx + 18, cy - 8), 13, "ink2")
        else:
            self.pg.draw.circle(surf, C["grid"], (cx, cy), 8, width=1)
            self.text(surf, "idle", (cx + 18, cy - 8), 13, "muted")

    def _wbc(self, surf, m, rect):
        x, y, w, h = rect
        self.panel(surf, rect, "WBC (50 Hz)")
        p = m.processors()
        t = m.last
        st = m.tick_stats()
        self.text(surf, p["wbc"]["label"], (x + 110, y + 7), 13, "wbc", bold=True)

        def num(v, unit, fmt="{:6.2f}"):
            return "—" if v is None or not math.isfinite(v) else fmt.format(v) + unit

        c1, c2, ry = x + 12, x + 214, y + 32
        self._kv(surf, c1, ry, "period now", num(t["period_us"] / 1e3 if t else None, " ms"))
        self._kv(surf, c1, ry + 22, "period p99", num(st["period_p99_ms"], " ms"),
                 "crit" if math.isfinite(st["period_p99_ms"]) and st["period_p99_ms"] > 21 else "ink")
        self._kv(surf, c1, ry + 44, "jitter std", num(st["jitter_ms"], " ms", "{:6.3f}"))
        self._kv(surf, c1, ry + 66, "RTT now", num(t["wbc_rtt_us"] if t else None, " µs", "{:6.0f}"))
        self._kv(surf, c1, ry + 88, "RTT p99", num(st["rtt_p99_us"], " µs", "{:6.0f}"),
                 "crit" if math.isfinite(st["rtt_p99_us"]) and st["rtt_p99_us"] > 2000 else "ink")
        be = p["wbc_now"] or "—"
        self._kv(surf, c2, ry, "computed by", be, "serious" if p["fallback"] else "ink", w=194)
        self._kv(surf, c2, ry + 22, "device", num(t["wbc_dev_us"] if t else None, " µs", "{:6.1f}"), w=194)
        self._kv(surf, c2, ry + 44, "fallbacks", f"{m.n_fallback}", "serious" if m.n_fallback else "ink", w=194)
        self._kv(surf, c2, ry + 66, "overruns / slips", f"{m.n_overrun} / {m.n_slip}",
                 "crit" if m.n_overrun or m.n_slip else "ink", w=194)
        sh = t["shadow_da"] if t else math.nan
        self._kv(surf, c2, ry + 88, "shadow |Δa|", "—" if not math.isfinite(sh) else f"{sh:.1e}", w=194)
        if t:
            walk = t["wbc_pol"] == 1
            self.pill(surf, "WALK" if walk else "BALANCE", (x + w - 12, y + 6), "wbc" if walk else "axis",
                      ink="page" if walk else "ink", size=12, anchor="topright")

    def _chunk(self, surf, m, rect):
        pg = self.pg
        x, y, w, h = rect
        self.panel(surf, rect, "CHUNK (30 rows)")
        t = m.last
        ar = m.active_req()
        n, cw, gap = 30, 10, 2
        x0, y0 = x + 12, y + 50
        row = t["row"] if t else -1
        d = ar["d"] if ar and ar["d"] >= 0 else None
        s_eff = ar.get("s_eff") if ar else None
        for i in range(n):
            r = pg.Rect(x0 + i * (cw + gap), y0, cw, 26)
            if row >= 0 and i == row:
                pg.draw.rect(surf, C["ink"], r, border_radius=2)
            elif row >= 0 and d is not None and d <= i < row:
                pg.draw.rect(surf, C["vla"], r, border_radius=2)
            elif d is not None and i < d:
                pg.draw.rect(surf, C["grid"], r, border_radius=2)
            else:
                pg.draw.rect(surf, C["axis"], r, width=1, border_radius=2)
        if d is not None and s_eff:
            a, b = d, min(n, d + int(s_eff))
            xa, xb = x0 + a * (cw + gap), x0 + b * (cw + gap) - gap
            pg.draw.line(surf, C["ink2"], (xa, y0 - 6), (xb, y0 - 6), 2)
            pg.draw.line(surf, C["ink2"], (xa, y0 - 10), (xa, y0 - 3), 2)
            pg.draw.line(surf, C["ink2"], (xb, y0 - 10), (xb, y0 - 3), 2)
            self.text(surf, f"[d, s+d) = [{a}, {b})", (xb + 6, y0 - 16), 12, "ink2", mono=True)
        info = "no chunk" if not t or row < 0 else (f"seq {t['seq']}  row {row}  age {row * TICK_MS:.0f} ms  "
                                                    f"rows left {t['rows_left']}")
        self.text(surf, info, (x0, y0 + 34), 13, "ink2", mono=True)
        if t and t["starved"]:
            self.pill(surf, f"STARVED  {m.starve_run} ticks", (x + w - 12, y + 6), "crit", ink="ink", size=16,
                      anchor="topright")
        elif m.n_starved:
            self.text(surf, f"starved {m.n_starved} (max run {m.max_starve_run})", (x + w - 12, y + 9), 12,
                      "serious", anchor="topright")

    def _hist(self, surf, rect, values, lo, hi, nbins, limit, unit, label):
        pg = self.pg
        x, y, w, h = rect
        v = np.asarray([a for a in values if a is not None and math.isfinite(a)], dtype=np.float64)
        self.text(surf, label, (x, y), 12, "muted")
        base = y + h - 16
        pg.draw.line(surf, C["axis"], (x, base), (x + w, base), 1)
        if v.size == 0:
            self.text(surf, "no data", (x + w // 2, y + h // 2), 12, "muted", anchor="center")
            return
        edges = np.linspace(lo, hi, nbins + 1)
        counts = np.histogram(np.clip(v, lo, hi - 1e-9), bins=edges)[0]
        top = max(1, counts.max())
        bw = w / nbins
        for i, c in enumerate(counts):
            if c == 0:
                continue
            bh = max(2, int((h - 34) * c / top))
            over = edges[i] >= limit
            col = C["crit"] if over else C["vla"]
            pg.draw.rect(surf, col, pg.Rect(int(x + i * bw) + 1, base - bh, max(1, int(bw) - 2), bh),
                         border_top_left_radius=2, border_top_right_radius=2)
        lx = x + (limit - lo) / (hi - lo) * w
        pg.draw.line(surf, C["ink2"], (lx, y + 16), (lx, base), 1)
        self.text(surf, f"{lo:g}", (x, base + 2), 11, "muted", mono=True)
        self.text(surf, f"{hi:g}{unit}", (x + w, base + 2), 11, "muted", mono=True, anchor="topright")
        p50, p99 = np.percentile(v, 50), np.percentile(v, 99)
        self.text(surf, f"p50 {p50:.2f}  p99 {p99:.2f}{unit}  n {v.size}", (x + w, y), 12, "ink2", mono=True,
                  anchor="topright")

    def _timing(self, surf, m, rect):
        x, y, w, h = rect
        self.panel(surf, rect, "TICK TIMING (last 60 s)")
        per = [p / 1e3 for p in m.period]
        comp = [c / 1e3 for c in m.compute]
        self._hist(surf, (x + 12, y + 30, w - 24, 112), per, 18.0, 23.0, 20, 21.0, " ms", "period")
        self._hist(surf, (x + 12, y + 152, w - 24, 112), comp, 0.0, 25.0, 25, 20.0, " ms", "compute")

    def _nav(self, surf, m, rect):
        pg = self.pg
        x, y, w, h = rect
        self.panel(surf, rect, "NAV COMMAND (to the WBC)")
        t = m.last
        nav = t["nav"] if t else [0.0, 0.0, 0.0]
        rows = (("vx", nav[0], 0.5, "m/s"), ("vy", nav[1], 0.5, "m/s"), ("wz", nav[2], 1.0, "rad/s"))
        bx, bw = x + 60, w - 200
        for i, (name, v, span, unit) in enumerate(rows):
            yy = y + 36 + i * 30
            self.text(surf, name, (x + 14, yy), 14, "ink2", mono=True)
            track = pg.Rect(bx, yy + 3, bw, 12)
            pg.draw.rect(surf, C["grid"], track, border_radius=3)
            mid = bx + bw // 2
            pg.draw.line(surf, C["axis"], (mid, yy - 1), (mid, yy + 19), 1)
            frac = max(-1.0, min(1.0, v / span))
            if abs(frac) > 1e-4:
                a, b = sorted((mid, int(mid + frac * bw / 2)))
                pg.draw.rect(surf, C["wbc"], pg.Rect(a, yy + 3, max(2, b - a), 12), border_radius=3)
            self.text(surf, f"{v:+.3f} {unit}", (x + w - 14, yy), 14, "ink", mono=True, anchor="topright")
        if t:
            norm = math.sqrt(sum(v * v for v in nav))
            z = t["pelvis_z"]
            self.text(surf, f"|nav| {norm:.3f} (Walk > 0.05)   pelvis z "
                      + ("—" if not math.isfinite(z) else f"{z:.3f} m"), (x + 14, y + h - 30), 13,
                      "crit" if math.isfinite(z) and z < FALL_Z else "ink2", mono=True)

    def _ego(self, surf, m, rect):
        pg = self.pg
        x, y, w, h = rect
        self.panel(surf, rect, "EGO VIEW (image of the active chunk)")
        e = m.ego_image()
        box = pg.Rect(x + 6, y + 30, 384, 288)
        if e is None:
            pg.draw.rect(surf, C["grid"], box, width=1, border_radius=4)
            self.text(surf, "no thumbnail yet", box.center, 14, "muted", anchor="center")
            return
        seq, k_obs, img = e
        key = (seq, id(img))
        if self._ego_cache[0] != key:
            small = pg.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
            self._ego_cache = (key, pg.transform.smoothscale(small, box.size))
        surf.blit(self._ego_cache[1], box)
        self.pill(surf, f"k_obs {k_obs}  seq {seq}", (box.x + 6, box.bottom - 6), (0, 0, 0), ink="ink", size=12,
                  anchor="bottomleft")

    def _episode(self, surf, m, rect, now_ns, control):
        x, y, w, h = rect
        self.panel(surf, rect, "EPISODE")
        t = m.last
        meta = m.meta
        yy = y + 30
        if t:
            if t["success"] or m.ep_success:
                self.pill(surf, "✓ SUCCESS", (x + w - 12, y + 6), "good", ink="page", anchor="topright")
            elif math.isfinite(t["pelvis_z"]) and t["pelvis_z"] < FALL_Z:
                self.pill(surf, "✗ FALL", (x + w - 12, y + 6), "crit", ink="ink", anchor="topright")
        self._kv(surf, x + 12, yy, "episode / seed", f"{t['ep'] if t else '-'} / {meta.get('seed', '-')}", w=370)
        self._kv(surf, x + 12, yy + 22, "done / success / falls",
                 f"{m.episodes} / {m.successes} / {m.falls}", w=370)
        cur = meta.get("current") or {}
        pend = meta.get("pending_vla")
        swap = f"vla {cur.get('vla', '?')}" + (f" → {pend} at reset" if pend else "") + \
               f"   wbc {cur.get('wbc', '?')}"
        self._kv(surf, x + 12, yy + 44, "backends", swap if cur else "—", w=370)
        self._kv(surf, x + 12, yy + 66, "telemetry rx / dropped",
                 f"{sum(m.rx.values())} / {m.dropped_ticks} ticks" + (f" / {m.bad} bad" if m.bad else ""), w=370)
        self._kv(surf, x + 12, yy + 88, "control", control or "—", w=370)
        if self.toast and time.monotonic() < self.toast[2]:
            self.text(surf, self.toast[0], (x + 12, yy + 116), 14, self.toast[1], bold=True)
        self.text(surf, "v VLA  w WBC  r reset  p pause  c chip  q quit", (x + 12, y + h - 26), 13,
                  "muted")

    # -- Blackhole chip panel (§4.3 item 7)

    def _src_pill(self, surf, label, state, age, pos, msg=None):
        fill = {"LIVE": "good", "STALE": "warn", "NO DATA": "grid"}[state]
        ink = "page" if state != "NO DATA" else "muted"
        text = f"{label} {state}" if state != "STALE" or age is None else f"{label} STALE {age:.0f} s"
        if state == "LIVE" and isinstance(msg, dict):  # what the process says about itself
            if msg.get("error") or msg.get("degraded"):
                text, fill, ink = f"{label} DEGRADED", "crit", "ink"  # x280d serves error replies (ORT falls in)
            elif msg.get("ready") is False:
                text, fill = f"{label} BUILDING", "warn"  # server_g1 before its boot check passed
        return self.pill(surf, text, pos, fill, ink=ink, size=11, anchor="topright")

    def _chip(self, surf, m, rect, now_ns):
        pg = self.pg
        x, y, w, h = rect
        self.panel(surf, rect, "BLACKHOLE p150a")
        chip = m.chip
        tt, xs = chip.get("server_g1"), chip.get("x280d")
        st_tt, age_tt = chip.state("server_g1", now_ns)
        st_x, age_x = chip.state("x280d", now_ns)
        box = self._src_pill(surf, "X280", st_x, age_x, (x + w - 10, y + 6), xs)
        self._src_pill(surf, "TT", st_tt, age_tt, (box.left - 6, y + 6), tt)
        serving = chip.serving(now_ns)
        self._floorplan(surf, (x + 12, y + 30, w - 24, 150), tt if st_tt != "NO DATA" else {}, xs, st_tt, st_x,
                        serving)
        cw = (w - 36) // 2
        self._tensix_col(surf, (x + 12, y + 198, cw, 190), tt, st_tt)
        self._x280_col(surf, (x + 24 + cw, y + 198, cw, 190), xs, st_x, chip, serving, now_ns)
        self._telemetry_row(surf, (x + 12, y + h - 58, w - 24, 50), chip, now_ns)

    def _floorplan(self, surf, rect, tt, xs, st_tt, st_x, serving):
        """The logical Tensix worker grid coloured by megakernel role, and the NoC0 x = 8 column with the ARC and
        the four L2CPU tiles at their rows (x280/regs.py); x280d's tile is lit by its ledger and monitor state, and
        labelled "serving N/s" while it serves calls (``serving``: ChipModel.serving)."""
        pg = self.pg
        x, y, w, h = rect
        lay = core_layout(tt.get("core_map")) if isinstance(tt, dict) else {}
        if lay:
            gx, gy = lay["grid"]
            cell = max(4, min(12, (h - 34) // gy, 210 // gx))
            gap = 1 if cell < 8 else 2
            dim = st_tt != "LIVE"
            for cy in range(gy):
                for cx in range(gx):
                    role = lay["cells"].get((cx, cy))
                    r = pg.Rect(x + cx * (cell + gap), y + 14 + cy * (cell + gap), cell, cell)
                    if role is None:  # counts only and no matching placement: the grid, unlit
                        pg.draw.rect(surf, C["grid"], r, width=1, border_radius=2)
                    else:
                        col = C[role] if not dim else tuple(int(c * 0.45) for c in C[role])
                        pg.draw.rect(surf, col, r, border_radius=2)
            grid_w, grid_h = gx * (cell + gap), gy * (cell + gap)
            self.text(surf, f"Tensix {gx}×{gy} (logical)", (x, y), 11, "ink2")
        else:  # nothing reported: no grid is drawn (its size would be a guess)
            grid_w, grid_h = 154, 104
            r = pg.Rect(x, y + 14, grid_w, grid_h)
            pg.draw.rect(surf, C["grid"], r, width=1, border_radius=4)
            self.text(surf, "no core map", r.center, 11, "muted", anchor="center")
            self.text(surf, "Tensix (server_g1)", (x, y), 11, "muted")
        legend_y = y + 18 + grid_h
        counts = lay.get("counts") or {}
        if counts:
            lx = x
            for role in CORE_ROLES:
                if role not in counts:
                    continue
                pg.draw.rect(surf, C[role], pg.Rect(lx, legend_y + 3, 8, 8), border_radius=2)
                r = self.text(surf, f"{counts[role]} {role}", (lx + 11, legend_y), 11, "ink2")
                lx = r.right + 8
            if lay["source"] and lay["source"] != "reported":  # counts were reported, the placement is MK's
                self.text(surf, "· MK default map", (lx, legend_y + 1), 10, "muted")
        # the x = 8 column: ARC (8, 0) and L2CPU0..3
        refs = x280_refs()
        xy = refs.get("xy") or {}
        tiles = _tiles(xs.get("tile")) if isinstance(xs, dict) else []
        more = _sub(xs, "tiles")  # DP4 fp32: the second tile's monitor, keyed by tile index
        tiles += [int(k) for k in more if str(k).isdigit() and int(k) not in tiles]
        col_x, col_w = x + grid_w + 14, w - grid_w - 14
        rows = 12
        row_h = (h - 14) / rows
        self.text(surf, "NoC0 x=8", (col_x, y), 11, "ink2")
        arc = pg.Rect(col_x, int(y + 14), col_w, int(row_h) - 1)
        pg.draw.rect(surf, C["grid"], arc, width=1, border_radius=3)
        self.text(surf, "ARC (8,0)", (arc.x + 4, arc.y), 10, "muted")
        for tile in range(4):
            ty = l2cpu_xy(tile)[1]
            r = pg.Rect(col_x, int(y + 14 + ty * row_h), col_w, int(row_h * 1.6))
            label = f"L2CPU{tile} (8,{ty})" if xy else f"L2CPU{tile}"
            if tile in tiles:
                look = xs if tile == tiles[0] else {**_sub(more, str(tile)), "tile_state": xs.get("tile_state")}
                fill, ink, note = self._tile_look(look, st_x, serving.get(tile))
                pg.draw.rect(surf, C[fill], r, border_radius=3)
                self.text(surf, f"{label} {note}", (r.x + 4, r.y + 1), 10, ink, bold=True)
            else:
                pg.draw.rect(surf, C["grid"], r, width=1, border_radius=3)
                role = "spare" if tile == refs.get("spare_tile") else "reserved"  # §1.1: L2CPU0/1 reserved
                self.text(surf, f"{label} {role if xy else ''}".rstrip(), (r.x + 4, r.y + 1), 10, "muted")

    @staticmethod
    def _tile_look(xs, st_x, rate=None) -> tuple[str, str, str]:
        """(fill, ink, note) of x280d's tile: lit only while x280d is live and the hart answers; ``rate``: the calls/s
        it serves (the note is then "serving N/s", see hart_note)."""
        ledger = xs.get("tile_state") if isinstance(xs.get("tile_state"), str) else None
        mon_name = _mon_name(xs.get("monitor_state"))
        note = hart_note(xs, rate) or ledger or "?"
        if st_x != "LIVE":
            return "axis", "ink2", f"{note} (stale)"
        if ledger == "burned" or mon_name == "fault":
            return "crit", "ink", note
        if mon_name in ("idle", "busy") or note.startswith("serving"):
            return "wbc", "page", note
        if mon_name in ("parked", "reentered", "host_gone") or ledger == "parked":
            return "warn", "page", note
        return "serious", "page", note  # fresh / released without a monitor yet / unknown

    def _rows(self, surf, x, y, w, rows, live: bool):
        """key / value rows; a stale source's values are greyed, except a critical one."""
        for i, (k, v, col) in enumerate(rows):
            self.text(surf, k, (x, y + i * 17), 11, "muted")
            self.text(surf, v, (x + w, y + i * 17), 12, col if live or col == "crit" else "muted", mono=True,
                      anchor="topright")

    @staticmethod
    def _fmt(v, fmt="{:.1f}", unit="") -> str:
        v = _num(v)
        return "—" if v is None else fmt.format(v) + unit

    def _tensix_col(self, surf, rect, tt, state):
        x, y, w, h = rect
        self.text(surf, "TENSIX · server_g1", (x, y), 12, "vla" if state == "LIVE" else "muted", bold=True)
        last, p50 = _sub(tt, "last_ms"), _sub(tt, "p50_ms")
        f = self._fmt

        def edd(d):
            vals = [f(d.get(k)) for k in ("encode", "device", "decode")]
            return "—" if all(v == "—" for v in vals) else "/".join(vals)

        duty = _num(tt.get("device_duty"))
        trace = tt.get("trace")
        cq = _int(tt.get("cq"))
        tree = tt.get("tree") if isinstance(tt.get("tree"), str) else None
        errors = _int(tt.get("errors"))
        rows = [
            ("enc/dev/dec ms", edd(last), "ink"),
            ("  p50 (50)", edd(p50), "ink"),
            ("total last/p50", f"{f(last.get('total'))}/{f(p50.get('total'))}", "ink"),
            ("device duty", "—" if duty is None else f"{100 * duty:.0f} %", "ink"),
            ("calls / errors", f"{f(tt.get('calls'), '{:.0f}')} / {'—' if errors is None else errors}",
             "crit" if errors else "ink"),
            ("trace / CQ", f"{'—' if not isinstance(trace, bool) else ('on' if trace else 'off')} / "
                           f"{'—' if cq is None else cq}", "ink"),
            ("tree", (tree or "—")[:14], "ink2"),
            ("uptime", f(tt.get("uptime_s"), "{:.0f}", " s"), "ink2"),
        ]
        self._rows(surf, x, y + 20, w, rows, state == "LIVE")

    @staticmethod
    def _heartbeat(xs, chip, state, now_ns) -> tuple[str, str]:
        """(text, colour) of x280d's heartbeat row: the rate of the raw L1 counter (the hart's 50 MHz mtime, so
        "+50.0 M/s"), critical once it has not advanced for HB_STOP_S or the stream is not live. A publisher without
        the raw counter: its heartbeat_age_us (which a healthy hart keeps at ~0)."""
        if _int(xs.get("heartbeat")) is None:
            age = _num(xs.get("heartbeat_age_us"))
            if age is None:
                return "—", "ink"
            return (f"age {age:.0f} µs" if age < 10_000 else f"age {age / 1e3:.0f} ms"), "crit" if age > 5000 else "ink"
        if state != "LIVE":
            return "stream stale", "crit"
        moved = chip.hb_moved.get("x280d")
        still = (now_ns - moved) / 1e9 if moved is not None else 0.0
        snaps = [t for t, _ in chip.hist.get("x280d", ())]
        period = float(np.median(np.diff(snaps))) / 1e9 if len(snaps) > 2 else 0.0  # a re-sent snapshot at 2 Hz
        if still > max(HB_STOP_S, 2.5 * period):  # holds one value for ~1 s: not a stop
            return f"stopped {still:.1f} s", "crit"
        r = chip.rate("x280d", "heartbeat")
        if r is None:
            return "…", "ink"  # one snapshot so far
        div, unit = next((d, u) for d, u in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, "")) if r >= d or d == 1.0)
        return f"+{r / div:.1f} {unit}/s", "ink"

    def _x280_col(self, surf, rect, xs, state, chip, serving, now_ns):
        """x280d's column: rows that move while it serves (hart state with the calls/s, heartbeat rate, x280d's own
        per-call µs, calls served) ahead of the counters; the kernel's constant cycles and duty are not repeated."""
        x, y, w, h = rect
        tiles = _tiles(xs.get("tile"))
        tname = ",".join(f"L2CPU{t}" for t in tiles) or "?"
        self.text(surf, f"X280 · x280d ({tname})", (x, y), 12, "wbc" if state == "LIVE" else "muted", bold=True)
        f = self._fmt
        mon = xs.get("monitor_state")
        mon = x280_refs()["states"].get(mon, str(mon)) if _int(mon) is not None else (
            mon if isinstance(mon, str) else "—")
        hart = hart_note(xs, serving.get(tiles[0]) if tiles else None) or mon
        ledger = xs.get("tile_state") if isinstance(xs.get("tile_state"), str) else "—"
        tile_hart = hart if ledger in ("released", "—") else f"{ledger}/{hart}"  # released: the ledger's normal state
        hb, hb_col = self._heartbeat(xs, chip, state, now_ns)
        xl, xp = _num(xs.get("x280d_us_last")), _num(xs.get("x280d_us_p99"))
        x280d = "—" if xl is None and xp is None else f"{f(xl, '{:.0f}')}/{f(xp, '{:.0f}')} µs"
        calls = _int(xs.get("calls_ok"))
        cps = chip.rate("x280d", "calls_ok") if state == "LIVE" else None
        served = "—" if calls is None else f"{calls}" + ("" if cps is None else f" · {cps:.1f}/s")
        ecc = _sub(xs, "ccache_ecc")
        variant = xs.get("variant") if isinstance(xs.get("variant"), str) else "—"
        dtype = xs.get("dtype") if isinstance(xs.get("dtype"), str) else None  # x280d sends variant and dtype apart
        if dtype and variant != "—" and dtype not in variant:
            variant = f"{variant}-{dtype}"
        fw = _int(xs.get("fw_ver"))
        bid = _int(xs.get("build_id"))
        ecc_fail = _int(ecc.get("fail"))
        rows = [
            ("tile / hart", tile_hart[:16], "crit" if mon == "fault" or ledger == "burned" else "ink"),
            ("heartbeat", hb, hb_col),
            ("x280d last/p99", x280d, "ink"),
            ("compute last", f(xs.get("compute_us_last"), "{:.2f}", " µs"), "ink"),
            ("calls served", served, "ink"),
            ("runs / overrun", f"{f(xs.get('n_runs'), '{:.0f}')} / {f(xs.get('n_overrun'), '{:.0f}')}",
             "serious" if _num(xs.get("n_overrun")) else "ink"),
            ("reentry / BEU", f"{f(xs.get('n_reentry'), '{:.0f}')} / {f(xs.get('beu_accrued'), '{:.0f}')}",
             "serious" if _num(xs.get("beu_accrued")) else "ink"),
            ("ECC fix / fail", f"{f(ecc.get('fix'), '{:.0f}')} / {'—' if ecc_fail is None else ecc_fail}",
             "crit" if ecc_fail else "ink"),
            ("variant", variant[:14], "ink"),
            ("fw / build", f"{'—' if fw is None else f'{fw:#x}'} {'' if bid is None else f'{bid:08x}'}"[:16], "ink2"),
        ]
        self._rows(surf, x, y + 20, w, rows, state == "LIVE")

    def _telemetry_row(self, surf, rect, chip, now_ns):
        x, y, w, h = rect
        pg = self.pg
        pg.draw.line(surf, C["grid"], (x, y), (x + w, y), 1)
        row = chip.telemetry_row(now_ns)
        tel, src = row["tel"], row["src"]
        if src is None:  # never a number without a real source (a sim's are invented)
            self.text(surf, row["note"], (x, y + 6), 11, "muted")
            return
        f = self._fmt
        tiles = _tiles(chip.get("x280d").get("tile"))
        t = tiles[0] if tiles else x280_refs().get("default_tile", 3)
        l2 = tel.get("l2cpuclk_mhz")
        l2v = _num(l2[t]) if isinstance(l2, (list, tuple)) and len(l2) > t else None
        line1 = (f"AICLK {f(tel.get('aiclk_mhz'), '{:.0f}', ' MHz')}   L2CPUCLK{t} "
                 f"{'—' if l2v is None else f'{l2v:.0f} MHz'}   ASIC {f(tel.get('asic_temp_c'), '{:.1f}', ' °C')}")
        line2 = (f"VCORE {f(tel.get('vcore_mv'), '{:.0f}', ' mV')}   TDP {f(tel.get('tdp_w'), '{:.0f}', ' W')}   "
                 f"TDC {f(tel.get('tdc_a'), '{:.0f}', ' A')}   in {f(tel.get('input_power_w'), '{:.0f}', ' W')}")
        self.text(surf, line1, (x, y + 5), 12, "ink2", mono=True)
        self.text(surf, line2, (x, y + 22), 12, "ink2", mono=True)
        self.text(surf, f"TIMER_HEARTBEAT {f(tel.get('timer_heartbeat'), '{:.0f}')}", (x, y + 39), 10, "muted")
        self.text(surf, f"{row['source']} via {src}", (x + w, y + 39), 10, "muted", anchor="topright")

    def _timeline(self, surf, m, rect):
        pg = self.pg
        x, y, w, h = rect
        pg.draw.rect(surf, C["panel"], rect, border_radius=6)
        self.text(surf, "last 4 s: compute (red > 20 ms) · VLA requests · splices", (x + 12, y + 3), 11,
                  "muted")
        ticks = list(m.ticks)
        if not ticks:
            return
        k_now, ep_now = ticks[-1]["k"], ticks[-1]["ep"]
        px = (w - 24) / TIMELINE_TICKS
        right = x + w - 12
        base = y + h - 6
        bar_h = h - 30

        def kx(k):
            return right - (k_now - k + 1) * px

        for t in ticks:
            if t["ep"] != ep_now:
                continue
            c = t["compute_us"] / 1e3
            if not math.isfinite(c):
                continue
            bh = max(1, int(bar_h * min(c, 25.0) / 25.0))
            col = C["crit"] if t["overrun"] or c > TICK_MS else C["muted"]
            pg.draw.rect(surf, col, pg.Rect(int(kx(t["k"])), base - bh, max(1, int(px) - 1), bh))
            if t["starved"]:
                pg.draw.rect(surf, C["crit"], pg.Rect(int(kx(t["k"])), base + 1, max(1, int(px)), 3))
            if t["splice"]:
                pg.draw.line(surf, C["ink"], (kx(t["k"]), y + 16), (kx(t["k"]), base), 1)
        lim = base - int(bar_h * TICK_MS / 25.0)
        pg.draw.line(surf, C["axis"], (x + 12, lim), (right, lim), 1)
        ry = y + 16
        for ep, k_obs, k_arr, outcome in m.req_bars:
            if ep != ep_now or k_arr < k_now - TIMELINE_TICKS:
                continue
            a, b = max(kx(k_obs), x + 12), kx(k_arr) + px
            col = C["vla"] if outcome == "APPLIED" else C["serious"]
            pg.draw.rect(surf, col, pg.Rect(int(a), ry, max(2, int(b - a)), 5), border_radius=2)
        if ticks[-1]["inflight"] and m.inflight_since is not None:
            a = max(kx(m.inflight_since), x + 12)
            pg.draw.rect(surf, C["vla"], pg.Rect(int(a), ry, max(2, int(right - a)), 5), width=1, border_radius=2)


# --- process -----------------------------------------------------------------------------------


def drain(sock, model: HudModel, max_msgs: int, budget_ms: float) -> tuple[int, bool]:
    """Receive what is queued, within the caps; returns (n, capped). The rest waits or is dropped by HWM."""
    import zmq

    n, t_end = 0, time.monotonic_ns() + int(budget_ms * 1e6)
    while n < max_msgs:
        try:
            parts = sock.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return n, False
        model.ingest_raw(parts, time.monotonic_ns())
        n += 1
        if time.monotonic_ns() >= t_end:
            return n, True
    return n, True


def drain_status(sock, chip: ChipModel, max_msgs: int = 64) -> int:
    """The device status streams: a few messages per second, all of them taken (the SUB HWM bounds a backlog)."""
    import zmq

    n = 0
    while n < max_msgs:
        try:
            parts = sock.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            break
        chip.ingest_raw(parts, time.monotonic_ns())
        n += 1
    return n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hud.py", description=__doc__.split("\n")[0])
    ap.add_argument("--sub", default=TELEMETRY_PUB, help=f"telemetry PUB to subscribe to (default {TELEMETRY_PUB})")
    ap.add_argument("--control", default=CONTROL_PULL, help=f"loop control PULL (default {CONTROL_PULL})")
    ap.add_argument("--no-control", action="store_true", help="keys v/w/r/p send nothing")
    ap.add_argument("--x280-status", default=X280_STATUS, help=f"x280d status PUB (default {X280_STATUS})")
    ap.add_argument("--tt-status", default=TT_STATUS, help=f"server_g1 status PUB (default {TT_STATUS})")
    ap.add_argument("--no-status", action="store_true", help="do not subscribe to the device status streams")
    ap.add_argument("--chip", choices=("auto", "on", "off"), default="auto",
                    help="Blackhole chip panel: auto = once a device stream or a Tensix/X280 backend is seen")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--headless", action="store_true", help="SDL_VIDEODRIVER=dummy (tests, CI)")
    ap.add_argument("--frames", type=int, default=0, help="exit after N frames (0 = run until closed)")
    ap.add_argument("--duration", type=float, default=0.0, help="exit after S seconds")
    ap.add_argument("--png", default=None, help="save the last frame here at exit")
    ap.add_argument("--png-every", type=int, default=0, help="also save every Nth frame as <png stem>_<n>.png")
    ap.add_argument("--max-drain", type=int, default=4000, help="messages per frame at most")
    ap.add_argument("--drain-budget-ms", type=float, default=8.0, help="receive time per frame at most")
    ap.add_argument("--hwm", type=int, default=1000, help="SUB receive HWM (overflow is dropped)")
    ap.add_argument("--cpus", default=None, help="CPU affinity, e.g. 6,7 (PLAN.md §1.1)")
    ap.add_argument("--pos", default=None, help="window position X,Y")
    ap.add_argument("--summary", default=None, help="write the exit summary JSON here (also printed)")
    ap.add_argument("--inject-keys", default="", metavar="FRAME:KEY,...",
                    help="post these key presses at these frame numbers (tests, scripted demos), e.g. 30:w,60:p")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if args.pos:
        os.environ["SDL_VIDEO_WINDOW_POS"] = args.pos
    if args.cpus:
        os.sched_setaffinity(0, {int(c) for c in args.cpus.split(",") if c.strip()})

    import pygame
    import zmq

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("rtloop HUD")

    stop = {"flag": False}

    def on_signal(_signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    ctx = zmq.Context(io_threads=1)
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, args.hwm)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(args.sub)
    push = None if args.no_control else cproto.ControlPush(args.control, ctx=ctx)
    status_sub = None
    if not args.no_status:  # SUB only: nothing is ever sent to a device process (absent publishers are normal)
        status_sub = ctx.socket(zmq.SUB)
        status_sub.setsockopt(zmq.RCVHWM, 50)
        status_sub.setsockopt(zmq.LINGER, 0)
        status_sub.setsockopt(zmq.SUBSCRIBE, b"status")
        for url in dict.fromkeys(u for u in (args.x280_status, args.tt_status) if u):
            status_sub.connect(url)

    model, view = HudModel(), HudView(pygame, chip=args.chip)
    clock = pygame.time.Clock()
    frames, capped, max_frame_ms = 0, 0, 0.0
    t_start = time.monotonic()
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    png = Path(args.png) if args.png else None
    inject: dict[int, list[str]] = {}
    for item in filter(None, (p.strip() for p in args.inject_keys.split(","))):
        f, _, key = item.partition(":")
        inject.setdefault(int(f), []).append(key)
    try:
        while not stop["flag"]:
            t_f0 = time.monotonic_ns()
            for key in inject.pop(frames, ()):
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.key.key_code(key)))
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    stop["flag"] = True
                elif ev.type == pygame.KEYDOWN:
                    key = pygame.key.name(ev.key)
                    if key in ("q", "escape"):
                        stop["flag"] = True
                        continue
                    if key == "c":  # a view toggle, never a loop command
                        view.toggle_chip(model)
                        continue
                    kc = cproto.key_command(key)
                    if kc is None:
                        continue
                    if push is None:
                        view.toast = ("control disabled (--no-control)", "muted", time.monotonic() + 2.5)
                    elif push.send(*kc):
                        view.toast = (f"sent {kc[0]}" + (f" → {kc[1]}" if kc[1] else ""), "ink",
                                      time.monotonic() + 2.5)
                    else:
                        view.toast = ("not sent: no loop on the control PULL", "serious", time.monotonic() + 2.5)
            _, cap = drain(sub, model, args.max_drain, args.drain_budget_ms)
            capped += cap
            if status_sub is not None:
                drain_status(status_sub, model.chip)
            control = "off" if push is None else f"{push.url} sent {push.n_sent} failed {push.n_failed}"
            view.draw(screen, model, time.monotonic_ns(), fps, control)
            pygame.display.flip()
            frames += 1
            fps_n += 1
            now = time.monotonic()
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
            if png is not None and args.png_every and frames % args.png_every == 0:
                pygame.image.save(screen, str(png.with_name(f"{png.stem}_{frames:05d}{png.suffix}")))
            max_frame_ms = max(max_frame_ms, (time.monotonic_ns() - t_f0) / 1e6)
            if args.frames and frames >= args.frames:
                break
            if args.duration and now - t_start >= args.duration:
                break
            clock.tick(args.fps)
        if png is not None:
            pygame.image.save(screen, str(png))
    finally:
        elapsed = time.monotonic() - t_start
        status, age = model.status(time.monotonic_ns())
        out = {"frames": frames, "elapsed_s": round(elapsed, 3), "fps": round(frames / elapsed, 2) if elapsed else 0,
               "max_frame_ms": round(max_frame_ms, 2), "drain_capped_frames": capped, "status": status,
               "png": str(png) if png else None,
               "control_sent": push.n_sent if push else 0, "control_failed": push.n_failed if push else 0,
               "chip_panel": view.chip_shown(model),
               "chip_state": {s: model.chip.state(s, time.monotonic_ns())[0] for s in STATUS_SRCS},
               "chip_telemetry": {k: v for k, v in model.chip.telemetry_row(time.monotonic_ns()).items()
                                  if k != "tel"},  # what the telemetry row said last: its source, or why none
               **model.summary()}
        if push is not None:
            push.close()
        if status_sub is not None:
            status_sub.close(linger=0)
        sub.close(linger=0)
        ctx.term()
        pygame.quit()
        line = json.dumps(out)
        print(line, flush=True)
        if args.summary:
            Path(args.summary).write_text(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
