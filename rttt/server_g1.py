"""W-VLA on the Blackhole p150a Tensix for M2: GR00T N1.6 G1 behind the stock PolicyServer wire on 127.0.0.1:5557.

Port of ``TT/libero/server_n16.py`` (wire :66-112, policy :123-233, main loop :237-314) to the G1 checkpoint through
:mod:`rttt.g1_n16` (PLAN.md D10, WP-2.3). The wire is ``Isaac-GR00T-n1.6/gr00t/policy/server_client.py``'s
(msgpack, ndarrays as ``{"__ndarray_class__": True, "as_npy": np.save bytes}``, ``{"endpoint", "data"}`` requests,
``handler(**data)``), so the stock ``PolicyClient`` / rollout and :class:`rtloop.vla_client.VlaClient` both work.

Endpoints (frozen W-VLA contract, D1): ``ping``, ``info``, ``get_modality_config`` (``__ModalityConfig_class__``
objects built from the checkpoint's processor_config.json), ``reset`` (-> ``{}``), ``get_action(observation,
options=None)`` -> ``[action, info]`` with ``action.<group>`` f32 (B, 30, D) and info = the ``options.rt`` echo,
``t_recv_ns`` (stamped before the request is decoded), ``t_reply_ns``, ``infer_ms``, ``noise_seed``, ``backend``,
``encode_ms`` / ``device_ms`` / ``decode_ms`` (the TT split) and ``tree`` (PLAN.md §1.4.1). There is no ``kill``
endpoint. Every failure is replied as ``{"error": str}`` and serving
continues; a request the client has already abandoned (``options.rt`` ``t_send_ns`` + ``give_up_ms`` before the
receive stamp, :func:`rtloop.schema.request_abandoned`) is answered ``{"error": "stale ..."}`` without inference, as
``rtserve/mock_vla_server.py`` does, so requests queued during a stall do not each cost an inference ahead of the live
one. Test-only, behind ``--debug-endpoints`` (outside D1): ``predict_debug(observation, noise)`` (fixed noise in,
``action_pred_norm`` + decoded actions out, the same ``predict_normalized`` call as the per-call path),
``x280_peek(x, y, addr, n)`` (``ttnn.cluster.read_reg``, ``tt-metal/ttnn/cpp/ttnn-nanobind/cluster.cpp:172-186``;
only to cross-check x280d's E0 reads) and ``x280_lim_write(x, y, addr, data)`` (LIM init method B, §1.4.4:
``ttnn.cluster.write_to_core`` in 64 B-aligned 64 B chunks, ``cluster.cpp:95-120``; every line checked against
``x280/regs.py``, nothing outside LIM); without the flag they reply ``{"error": "debug endpoints disabled"}``.

Per request: the observation is cut to the 9 modality keys **before** ``split_batch`` (the stock rollout sends 13
more, two of them f32 480x640 camera copies), checked like ``Gr00tSimPolicyWrapper(strict=True)``, and served element
by element at batch 1: host encode -> ``predict_normalized`` (4 traces, megakernel DiT) -> host decode. Noise is
fresh per element from one CPU ``torch.Generator(--seed)``; ``options.rt.noise_seed`` (D17) reseeds it first, so
paired arms on this backend draw the same noise for the same request. For B = 1 that noise is exactly
``g1_n16.make_noise(noise_seed)`` (CPU generator, fp32 (1, 50, 128)), reported as ``info.noise = "cpu"``. One
JSONL line per get_action / predict_debug call (``--log``); a line that cannot be written is counted, never fatal.

Status stream (``--status-pub [URL]``, PLAN.md §4.3 item 7, default off; bare flag = ``tcp://127.0.0.1:5574``):
zmq PUB, SNDHWM 10, multipart ``[b"status", rtloop.wire.dumps(msg)]`` at ``--status-hz`` (2-10) from its own thread
and context; sends never block (dropped at the HWM) and the thread holds the stats lock only for a deque copy and
never calls ttnn. ``msg``: src, pid, t_ns (CLOCK_MONOTONIC), seq, backend, uptime_s, ready, calls (inference calls:
get_action + predict_debug), errors, stale, last_ms / p50_ms {encode, device, decode, total} (last 50 calls),
device_duty (device ms / wall ms over the last 5 s), trace, cq, core_map (MK's megakernel ``CoreMap`` of the live
model), tree, telemetry. ``telemetry`` is always None: ttnn exposes no read of the CMFW telemetry table
(``ttnn-nanobind/cluster.cpp`` binds only raw core reads/writes and the DRAM bank table), and UMD's Python
``FirmwareTelemetryReader`` hangs off a ``tt_umd.TTDevice.create(n)`` the caller opens itself
(``third_party/umd/nanobind/py_api_tt_device.cpp:241-270``), i.e. a second fd, which §4.3 forbids.

Start-up order: bind 127.0.0.1 first (a busy port fails before the device is opened), load (opens the device with 2
CQs), warm + capture on the first golden observation, then the boot **sanity** check: PCC >= ``--min-pcc`` (0.995)
of ``action_pred_norm[0, :30, :35]`` vs the bf16 G1 golden on two samples (one reset, one mid-episode), then
``READY``. ``--fake-model`` serves deterministic chunks through the real host pipeline without ttnn (protocol tests);
the boot check does not apply to it. All ttnn calls stay on the main thread (one model per process, not reentrant).
``--cpus`` (default 8-11, §1.1) pins the process before anything starts a thread; ``TT_METAL_NUMA_BASED_AFFINITY`` is
unset (it would rebind the main thread to every CPU but two, ``tt_metal/impl/device/device_manager.cpp:141-182``), and
after the build every thread outside ``--cpus`` is re-pinned and reported in ``info.affinity_check``.

    $DEV ttpy -m rttt.server_g1 --port 5557 [--debug-endpoints] [--status-pub]   # under chip_session.sh in S1
    ttpy -m rttt.server_g1 --fake-model --port 0 --cpus ''                         # CPU only, never opens the device
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import datetime
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import msgpack
import numpy as np
import torch
import zmq

from rtloop import schema, wire
from rttt import g1_n16 as G
from rttt import golden_io as gio

PROTO = 2
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5557  # PLAN.md D15
DEFAULT_MIN_PCC = 0.995  # boot sanity gate vs the bf16 golden (PLAN.md T-2 review fix; T-2.2 is the real gate)
DEFAULT_CPUS = "8,9,10,11"  # PLAN.md §1.1 process table
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "server_g1"
MAX_PEEK_WORDS = 256
MAX_LIM_WRITE_BYTES = 64 * 1024  # per request; x280d's method B sends 4 KiB
DEBUG_DISABLED = "debug endpoints disabled"
DEBUG_ENDPOINTS = ("predict_debug", "x280_peek", "x280_lim_write")
DEFAULT_STATUS_URL = "tcp://127.0.0.1:5574"  # PLAN.md §4.3 item 7 (x280d publishes on :5573)
DEFAULT_STATUS_HZ = 5.0
STATUS_HZ_RANGE = (2.0, 10.0)
STATUS_HWM = 10
STATUS_WINDOW = 50  # p50 over the last 50 calls
DUTY_WINDOW_S = 5.0
NOISE = "cpu"  # per-call noise: torch.Generator("cpu"), fp32 (info.noise)


class DebugDisabled(Exception):
    """A test-only endpoint was called on a server started without --debug-endpoints."""


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------- wire
class ModalityConfigJSON(dict):
    """Marker: serialised like the official ``ModalityConfig`` (``__ModalityConfig_class__`` + ``as_json``)."""


def _pre(x: Any) -> Any:
    # msgpack packs dict subclasses as plain maps without calling default(), so the marker is resolved first
    if isinstance(x, ModalityConfigJSON):
        return {"__ModalityConfig_class__": True, "as_json": dict(x)}
    if isinstance(x, dict):
        return {k: _pre(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_pre(v) for v in x]
    return x


def to_bytes(data: Any) -> bytes:
    return msgpack.packb(_pre(data), default=wire._encode)


def from_bytes(data: bytes) -> Any:
    return wire.loads(data)


# ---------------------------------------------------------------------------------------------------------------- models
class FakeG1Model:
    """Deterministic stand-in for ``Gr00tTT`` (no ttnn): ``action_pred_norm = 0.5 tanh(noise)`` plus the encoded state
    (dims [0, 43)) added to every row, so a chunk depends on the noise and on the request, like the real one."""

    def __init__(self) -> None:
        self.noise_shape = (1, G.MODEL_HORIZON, G.ACTION_DIM_PAD)
        self.embodiment_id = G.EMBODIMENT_ID
        self.layout_name = G.LAYOUT_NAME
        self.policy = None
        self.head = None
        self.n_predict = 0

    def predict_normalized(self, mi: Any, noise: torch.Tensor) -> torch.Tensor:
        e = int(torch.as_tensor(mi.embodiment_id).reshape(-1)[0])
        if e != self.embodiment_id or tuple(noise.shape) != self.noise_shape:
            raise ValueError(f"fake model: embodiment {e} / noise {tuple(noise.shape)}")
        out = 0.5 * torch.tanh(noise.to(torch.float32))
        out[..., : G.STATE_DIM] += 0.1 * mi.state.to(torch.float32).reshape(1, 1, -1)[..., : G.STATE_DIM]
        self.n_predict += 1
        return out

    def warm_and_capture(self, mi: Any, noise: Optional[torch.Tensor] = None) -> None:
        return None

    def release(self) -> None:
        return None


def fake_stamp() -> Dict[str, Any]:
    return {"label": "fake-g1-host-pipeline", "fake": True, "checkpoint": f"{G.REPO}@{G.SHA[:8]}",
            "source_checkpoint": f"{G.SRC_REPO}@{G.SRC_SHA[:8]}", "layout": G.LAYOUT_NAME,
            "embodiment_id": G.EMBODIMENT_ID, "denoising_steps": G.DENOISING_STEPS}


def warm_obs() -> Dict[str, Any]:
    """A G1 W-VLA observation (grey frame, zero state, the task instruction) when no golden is at hand."""
    obs = {schema.VIDEO_KEY: np.full((1, 1, *schema.IMAGE_SHAPE), 127, np.uint8),
           schema.LANGUAGE_KEY: ("pick up the apple, walk left and place the apple on the plate.",)}
    for (_, n), key in zip(schema.STATE_GROUPS, schema.STATE_KEYS, strict=True):
        obs[key] = np.zeros((1, 1, n), np.float32)
    return obs


# ---------------------------------------------------------------------------------------------------------------- golden
def boot_samples(golden_dir: Path) -> List[str]:
    """The two boot-check sids: the first reset sample and the first mid-episode one (index order)."""
    index = gio.load_index(golden_dir)
    if not index:
        raise FileNotFoundError(f"{golden_dir}: obs_index.json lists no samples")
    first = index[0]["sid"]
    mid = next((s["sid"] for s in index if s.get("kind") == "mid" and s["sid"] != first), None)
    return [first] + ([mid] if mid else [])


def boot_sanity_check(policy: "G1Policy", golden_dir: Path, min_pcc: float) -> Dict[str, Any]:
    """Warm + capture on the first golden sample, then PCC of ``action_pred_norm[0, :30, :35]`` vs the bf16 golden
    with the golden's own noise on the boot samples (golden layout: :mod:`rttt.golden_io`, written by
    ``make_g1_golden.py``: ``samples/<sid>.npz`` keys ``noise`` and ``bf16.action_pred_norm``). Raises below
    ``min_pcc``; returns the facts for ``info``."""
    golden_dir = Path(golden_dir).expanduser()
    sids = boot_samples(golden_dir)
    out: Dict[str, Any] = {"golden_dir": str(golden_dir), "samples": sids, "min_pcc": min_pcc, "pcc": []}
    for i, sid in enumerate(sids):
        s = gio.load_sample(golden_dir, sid)
        kept, _ = G.filter_obs(s["obs"])
        G.check_obs(kept)
        mi, _ = G.encode_obs(G.split_batch(kept)[0])
        noise = torch.from_numpy(np.ascontiguousarray(s["noise"], np.float32))
        if i == 0:
            t0 = time.perf_counter()
            policy.model.warm_and_capture(mi, noise)
            out["capture_s"] = round(time.perf_counter() - t0, 3)
        pred = policy.model.predict_normalized(mi, noise).detach().to("cpu", torch.float32).numpy()
        out["pcc"].append(gio.pcc(gio.valid(pred), gio.valid(s["bf16.action_pred_norm"])))
    out["pass"] = bool(min(out["pcc"]) >= min_pcc)
    if not out["pass"]:
        raise RuntimeError(f"boot sanity check failed: golden PCC {min(out['pcc']):.6f} < {min_pcc} on {sids}")
    return out


# ---------------------------------------------------------------------------------------------------------------- policy
class G1Policy:
    """The model plus the G1 host pipeline, the noise generator and the serving statistics."""

    def __init__(self, model: Any, stamp: Dict[str, Any], seed: int, fake: bool) -> None:
        self.model, self.stamp, self.fake = model, stamp, bool(fake)
        self.seed = int(seed)
        self.gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self.noise_shape = tuple(int(x) for x in model.noise_shape)
        self.n_calls = self.n_obs = 0
        self.lat: deque = deque(maxlen=2000)

    def run_one(self, one: Dict[str, Any], noise: torch.Tensor):
        """One element: host encode -> predict_normalized -> host decode, with the three times in ms."""
        t0 = time.perf_counter()
        mi, obs = G.encode_obs(one)
        t1 = time.perf_counter()
        pred = self.model.predict_normalized(mi, noise)
        t2 = time.perf_counter()
        acts = G.decode_actions(pred, obs)
        t3 = time.perf_counter()
        ms = {"encode_ms": (t1 - t0) * 1e3, "device_ms": (t2 - t1) * 1e3, "decode_ms": (t3 - t2) * 1e3}
        return pred, acts, ms, mi

    def predict(self, observation: Any, noise_seed: Optional[int]):
        """Filtered, checked, element-wise inference; returns (actions (B,30,D), ms sums, per-call facts)."""
        kept, dropped = G.filter_obs(observation)
        b = G.check_obs(kept)
        if noise_seed is not None:
            self.gen.manual_seed(int(noise_seed))
        per, tot = [], {"encode_ms": 0.0, "device_ms": 0.0, "decode_ms": 0.0}
        seq_len = n_text = None
        for one in G.split_batch(kept):
            noise = torch.randn(self.noise_shape, generator=self.gen, dtype=torch.float32)
            _, acts, ms, mi = self.run_one(one, noise)
            per.append(acts)
            for k in tot:
                tot[k] += ms[k]
            seq_len, n_text = int(mi.seq_len), int(mi.n_text)
        self.n_calls += 1
        self.n_obs += b
        return G.stack_actions(per), tot, {"B": b, "dropped": dropped, "seq_len": seq_len, "n_text": n_text}

    def info(self) -> Dict[str, Any]:
        lat = np.asarray(self.lat, np.float64)
        return {"n_calls": self.n_calls, "n_obs": self.n_obs, "seed": self.seed,
                "latency_ms_p50": float(np.median(lat)) if lat.size else None,
                "latency_ms_p90": float(np.percentile(lat, 90)) if lat.size else None,
                "latency_ms_max": float(lat.max()) if lat.size else None}

    def close(self) -> None:
        self.model.release()


# ---------------------------------------------------------------------------------------------------------------- status
class ServingStats:
    """Per-call timings for the status stream. The request path appends under ``lock`` and the status thread copies
    under it; every computation happens outside the lock, so neither side waits longer than a deque copy."""

    KEYS = ("encode", "device", "decode", "total")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.n_calls = 0
        self.last: Optional[Tuple[float, float, float, float]] = None
        self.recent: deque = deque(maxlen=STATUS_WINDOW)
        self.device: deque = deque()  # (t_end_ns, device_ms), pruned to the duty window on append

    def record(self, ms: Dict[str, float], total_ms: float) -> None:
        row = (float(ms["encode_ms"]), float(ms["device_ms"]), float(ms["decode_ms"]), float(total_ms))
        t = time.monotonic_ns()
        horizon = t - int(DUTY_WINDOW_S * 1e9)
        with self.lock:
            self.n_calls += 1
            self.last = row
            self.recent.append(row)
            self.device.append((t, row[1]))
            while self.device and self.device[0][0] < horizon:
                self.device.popleft()

    def snapshot(self):
        with self.lock:
            return self.n_calls, self.last, list(self.recent), list(self.device)


def device_duty(device: List[Tuple[int, float]], now_ns: int, since_ns: int) -> float:
    """Device ms / wall ms over the last DUTY_WINDOW_S (or since READY, if that is shorter); a call that ended inside
    the window counts at most the part of it that fits."""
    w0 = max(now_ns - int(DUTY_WINDOW_S * 1e9), since_ns)
    win_ms = (now_ns - w0) / 1e6
    if win_ms <= 0:
        return 0.0
    busy = sum(min(d, (t - w0) / 1e6) for t, d in device if t > w0)
    return round(min(max(busy / win_ms, 0.0), 1.0), 4)


def core_map_dict(cm: Any) -> Dict[str, Any]:
    """MK's megakernel ``CoreMap`` (``tt/megakernel/core_map.py``) as role counts plus the reader / hub / spare cores
    (logical coordinates); the compute cores are the rest of the grid."""
    return {"grid": [int(v) for v in cm.grid],
            "megakernel": {"readers": int(cm.n_readers), "hub": 1, "compute": int(cm.n_compute),
                           "spare": len(cm.spare)},
            "readers": [[int(c[0]), int(c[1])] for c in cm.readers], "hub": [int(cm.hub[0]), int(cm.hub[1])],
            "spare": [[int(c[0]), int(c[1])] for c in cm.spare]}


def static_status_of(model: Any, fake: bool) -> Dict[str, Any]:
    """trace / cq / core_map of the built model for the status stream, read once on the main thread (ttnn is
    main-thread only). A field the model does not expose stays None; the fake model has no trace and no device."""
    out: Dict[str, Any] = {"trace": False if fake else None, "cq": None, "core_map": None}
    if fake:
        return out
    try:
        out["trace"] = bool(model.captured)  # MK tt/model.py:1039-1041
    except Exception:  # noqa: BLE001 - unknown stays None
        pass
    try:
        ex = model.executor
        out["cq"] = int(ex.backend.num_command_queues(ex.device))  # MK tt/tracer.py:469-475
    except Exception:  # noqa: BLE001
        pass
    try:
        out["core_map"] = core_map_dict(model.head.megakernel.core_map)  # CoreMap.from_device, action_head.py:470-480
    except Exception:  # noqa: BLE001 - no megakernel (another DiT backend)
        pass
    return out


class StatusPublisher:
    """The out-of-band status PUB (module docstring). Own context and thread, so a slow HUD never shares an IO thread
    or a lock with the REP socket; a message that would block is dropped."""

    def __init__(self, server: "G1Server", url: str, hz: float = DEFAULT_STATUS_HZ) -> None:
        self.server = server
        self.period = 1.0 / min(max(float(hz), STATUS_HZ_RANGE[0]), STATUS_HZ_RANGE[1])
        self.ctx = zmq.Context(io_threads=1)
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, STATUS_HWM)
        self.sock.setsockopt(zmq.LINGER, 0)
        try:
            if str(url) in ("0", ""):
                self.sock.bind_to_random_port(f"tcp://{DEFAULT_HOST}")
            else:
                self.sock.bind(str(url))
        except zmq.ZMQError:
            self.sock.close(linger=0)
            self.ctx.term()
            raise
        self.addr = self.sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self.seq = self.n_sent = self.n_dropped = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="server_g1-status", daemon=True)

    def start(self) -> "StatusPublisher":
        self.thread.start()
        return self

    def _run(self) -> None:
        warned = False
        while not self._stop.wait(self.period):
            self.seq += 1
            try:
                self.sock.send_multipart([b"status", wire.dumps(self.server.status_msg(self.seq))], zmq.NOBLOCK)
                self.n_sent += 1
            except zmq.Again:
                self.n_dropped += 1
            except Exception as e:  # noqa: BLE001 - the status stream must never take the server down
                if not warned:
                    log(f"status PUB: {type(e).__name__}: {e} (further errors not logged)")
                    warned = True

    def close(self) -> None:
        self._stop.set()
        if self.thread.ident is not None:  # started
            self.thread.join(2.0)
        self.sock.close(linger=0)
        self.ctx.term()


# ---------------------------------------------------------------------------------------------------------------- x280
def check_lim_write(x: int, y: int, addr: int, n: int, chunk: int) -> None:
    """Raise ValueError unless ``n`` bytes at ``addr`` on (x, y) are whole LIM lines that x280/regs.py allows: a
    writable L2CPU tile at its NoC-0 coordinates (``regs.classify_write``; UMD maps an L2CPU's translated coordinates
    onto its NoC-0 ones, ``umd/device/coordinates/blackhole_coordinate_manager.cpp:215-221``, so these are what
    ``write_to_core`` expects) and every line inside [LIM_BASE, LIM_END). regs.py is the only source of the numbers."""
    from x280 import regs

    line = regs.LINE
    if chunk != line:
        raise ValueError(f"chunk must be {line} (one LIM line per write_to_core), got {chunk}")
    if n <= 0 or n % line or n > MAX_LIM_WRITE_BYTES:
        raise ValueError(f"data must be 1..{MAX_LIM_WRITE_BYTES // line} whole {line} B lines, got {n} B")
    if addr % line:
        raise ValueError(f"addr {addr:#x} is not {line} B aligned")
    if not (regs.LIM_BASE <= addr and addr + n <= regs.LIM_END):
        raise ValueError(f"[{addr:#x}, {addr + n:#x}) is outside LIM [{regs.LIM_BASE:#x}, {regs.LIM_END:#x})")
    for a in range(addr, addr + n, line):
        entry, _, why = regs.classify_write((x, y), a, line)
        if entry is None or entry.name != "LIM":
            why = why if entry is None else f"allowed as {entry.name}, not LIM"
            raise ValueError(f"x280/regs.py refuses a {line} B write at {a:#x} on {(x, y)}: {why}")


def _ttnn_write_to_core(device_id: int, x: int, y: int, addr: int, data: bytes) -> None:
    import ttnn

    ttnn.cluster.write_to_core(int(device_id), int(x), int(y), int(addr), data)


def enforce_affinity(cpus: set) -> Dict[str, Any]:
    """Re-pin every thread of this process that is not inside ``cpus``, and say which were. tt-metal pins its own
    threads only under TT_METAL_NUMA_BASED_AFFINITY (unset by main) or on multi-device meshes
    (``tt_metal/distributed/mesh_device.cpp:97-104``); this is the check that nothing else did."""
    repinned = []
    tids = sorted(int(t) for t in os.listdir("/proc/self/task"))
    for tid in tids:
        try:
            cur = os.sched_getaffinity(tid)
        except OSError:  # the thread exited
            continue
        if not cur <= cpus:
            rec: Dict[str, Any] = {"tid": tid, "was": sorted(cur)}
            try:
                os.sched_setaffinity(tid, cpus)
            except OSError as e:
                rec["error"] = str(e)
            repinned.append(rec)
    return {"cpus": sorted(cpus), "threads": len(tids), "repinned": repinned}


# ---------------------------------------------------------------------------------------------------------------- server
class G1Server:
    """REP socket + endpoint table; bound in the constructor so the port is held before the device is opened."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, log_path: Optional[Path] = None,
                 debug_endpoints: bool = False, ctx: Optional[zmq.Context] = None) -> None:
        self._own_ctx = ctx is None
        self.ctx = zmq.Context() if ctx is None else ctx
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        if int(port) == 0:
            self.port = self.sock.bind_to_random_port(f"tcp://{host}")
        else:
            self.sock.bind(f"tcp://{host}:{int(port)}")
            self.port = int(port)
        self.addr = self.sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self.debug_endpoints = bool(debug_endpoints)
        self.policy: Optional[G1Policy] = None
        self.static_info: Dict[str, Any] = {}
        self.static_status: Dict[str, Any] = {}  # trace / cq / core_map, set by build()
        self.t0_ns = time.monotonic_ns()
        self.t_ready_ns: Optional[int] = None
        self.t_recv_ns = 0
        self.n_errors = self.n_stale = self.n_log_errors = 0
        self.stats = ServingStats()
        self.status_addr: Optional[str] = None
        self.noc_write: Optional[Callable[..., None]] = None  # tests inject a recorder; None = ttnn (real model)
        self.dropped_counts: Counter = Counter()
        self.log_path = Path(log_path) if log_path else None
        self._fh = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.log_path, "a", buffering=1)
        self._rec: Optional[Dict[str, Any]] = None
        self.endpoints: Dict[str, Callable[..., Any]] = {
            "ping": self.ping,
            "info": self.info,
            "get_modality_config": self.get_modality_config,
            "reset": self.reset,
            "get_action": self.get_action,
            "predict_debug": self.predict_debug,
            "x280_peek": self.x280_peek,
            "x280_lim_write": self.x280_lim_write,
        }
        self.no_input = {"ping", "info", "get_modality_config"}

    # -- endpoints --------------------------------------------------------------------------------------------------
    def ping(self) -> Dict[str, Any]:
        return {"status": "ok", "message": "Server is running"}

    def info(self) -> Dict[str, Any]:
        out = dict(self.static_info)
        out.update(self.policy.info() if self.policy is not None else {})
        out.update(n_errors=self.n_errors, n_stale=self.n_stale, n_log_errors=self.n_log_errors,
                   dropped_key_counts=dict(self.dropped_counts), debug_endpoints=self.debug_endpoints,
                   addr=self.addr, status_pub=self.status_addr)
        return out

    def status_msg(self, seq: int) -> Dict[str, Any]:
        """One status-stream message (module docstring); called from the status thread. Reads plain attributes and
        one stats snapshot; unknown values are None, never estimated."""
        now = time.monotonic_ns()
        n, last, recent, dev = self.stats.snapshot()
        keys = ServingStats.KEYS
        st, si = self.static_status, self.static_info
        ready = self.t_ready_ns
        p50 = None
        if recent:
            cols = np.asarray(recent, np.float64)
            p50 = {k: round(float(np.median(cols[:, i])), 3) for i, k in enumerate(keys)}
        return {"src": "server_g1", "pid": os.getpid(), "t_ns": now, "seq": int(seq), "backend": si.get("backend"),
                "uptime_s": round((now - self.t0_ns) / 1e9, 3), "ready": ready is not None,
                "calls": n, "errors": self.n_errors, "stale": self.n_stale,
                "last_ms": dict(zip(keys, (round(v, 3) for v in last), strict=True)) if last else None, "p50_ms": p50,
                "device_duty": device_duty(dev, now, ready) if ready is not None else None,
                "trace": st.get("trace"), "cq": st.get("cq"), "core_map": st.get("core_map"), "tree": si.get("tree"),
                "telemetry": None}

    def get_modality_config(self) -> Dict[str, Any]:
        return {m: ModalityConfigJSON(v) for m, v in G.modality_config_json().items()}

    def reset(self, options: Any = None, **_: Any) -> Dict[str, Any]:
        return {}

    def _require_policy(self) -> G1Policy:
        if self.policy is None:
            raise RuntimeError("model not loaded")
        return self.policy

    def get_action(self, observation: Any, options: Any = None) -> Any:
        t_recv = self.t_recv_ns
        rt = options.get("rt") if isinstance(options, dict) else None
        if schema.request_abandoned(rt, t_recv):  # as rtserve/mock_vla_server.py: no inference
            self.n_stale += 1
            self._rec = {"endpoint": "get_action", "stale": True, "t_recv_ns": t_recv,
                         **{k: rt.get(k) for k in ("seq", "gen", "ep", "k_obs")}}
            return {"error": f"stale request seq {rt.get('seq')}: received {(t_recv - rt['t_send_ns']) / 1e6:.0f} ms "
                             f"after send, the client gave up after {rt['give_up_ms']} ms"}
        pol = self._require_policy()
        t0 = time.perf_counter()
        seed = rt.get("noise_seed") if isinstance(rt, dict) else None
        seed = None if seed is None else int(seed)
        action, ms, facts = pol.predict(observation, seed)
        infer_ms = (time.perf_counter() - t0) * 1e3
        pol.lat.append(infer_ms)
        self.stats.record(ms, infer_ms)
        self._count_dropped(facts["dropped"])
        info = {"rt": rt, "t_recv_ns": t_recv, "t_reply_ns": time.monotonic_ns(), "infer_ms": infer_ms,
                "noise_seed": seed, "backend": self.static_info.get("backend"),
                **ms, "tree": self.static_info.get("tree")}
        self._rec = {"endpoint": "get_action", "call": pol.n_calls,
                     **({k: rt.get(k) for k in ("seq", "gen", "ep", "k_obs")} if isinstance(rt, dict) else {}),
                     "noise_seed": seed, "infer_ms": round(infer_ms, 3), **{k: round(v, 3) for k, v in ms.items()},
                     "B": facts["B"], "seq_len": facts["seq_len"], "n_text": facts["n_text"],
                     "n_dropped": len(facts["dropped"]), "t_recv_ns": t_recv, "t_reply_ns": info["t_reply_ns"]}
        return [action, info]

    def predict_debug(self, observation: Any, noise: Any, options: Any = None) -> Dict[str, Any]:
        if not self.debug_endpoints:
            raise DebugDisabled(DEBUG_DISABLED)
        pol = self._require_policy()
        kept, dropped = G.filter_obs(observation)
        if G.check_obs(kept) != 1:
            raise ValueError("predict_debug takes one observation (B = 1)")
        n = np.asarray(noise)
        if n.dtype != np.float32 or tuple(n.shape) != pol.noise_shape or not np.isfinite(n).all():
            raise ValueError(f"noise must be finite float32 {pol.noise_shape}, got {n.dtype} {n.shape}")
        t0 = time.perf_counter()
        pred, acts, ms, mi = pol.run_one(G.split_batch(kept)[0], torch.from_numpy(np.ascontiguousarray(n)))
        self.stats.record(ms, (time.perf_counter() - t0) * 1e3)
        self._count_dropped(dropped)
        self._rec = {"endpoint": "predict_debug", **{k: round(v, 3) for k, v in ms.items()},
                     "seq_len": int(mi.seq_len), "n_dropped": len(dropped)}
        return {"action_pred_norm": np.ascontiguousarray(pred.detach().to("cpu", torch.float32).numpy()),
                "actions": acts, **ms}

    def x280_peek(self, x: int, y: int, addr: int, n: int = 1, stride: int = 4, device_id: int = 0) -> Dict[str, Any]:
        """``n`` u32 reads at ``addr + i * stride`` on NOC tile (x, y) through the open device's UC TLB window. The
        caller supplies every address (x280/regs.py is their only source); this endpoint holds none of its own."""
        if not self.debug_endpoints:
            raise DebugDisabled(DEBUG_DISABLED)
        pol = self._require_policy()
        if pol.fake:
            raise RuntimeError("x280_peek needs the real device (server started with --fake-model)")
        x, y, addr, n, stride = int(x), int(y), int(addr), int(n), int(stride)
        if not (1 <= n <= MAX_PEEK_WORDS) or addr % 4 or stride % 4 or stride <= 0 or addr < 0:
            raise ValueError(f"bad peek: addr {addr:#x} n {n} stride {stride} (u32-aligned, 1 <= n <= {MAX_PEEK_WORDS})")
        import ttnn

        t0 = time.monotonic_ns()
        vals = [int(ttnn.cluster.read_reg(int(device_id), x, y, addr + i * stride)) for i in range(n)]
        return {"x": x, "y": y, "addr": addr, "stride": stride, "values": vals, "t_ns": t0,
                "dt_us": (time.monotonic_ns() - t0) / 1e3}

    def x280_lim_write(self, x: int, y: int, addr: int, data: Any, chunk: int = 64,
                       device_id: int = 0) -> Dict[str, Any]:
        """LIM init method B (PLAN.md §1.4.4, T-3.E1b fallback; caller: ``x280d.py lim-init --method B``): ``data``
        written from ``addr`` on L2CPU tile (x, y) as one ``ttnn.cluster.write_to_core`` per 64 B line (the WC path,
        whose doc says to write a full 64-byte line first at an uninitialised LIM address, ``cluster.cpp:95-120``).
        Every line is checked by :func:`check_lim_write` before anything is written; nothing outside LIM."""
        if not self.debug_endpoints:
            raise DebugDisabled(DEBUG_DISABLED)
        pol = self._require_policy()
        x, y, addr, chunk, device_id = int(x), int(y), int(addr), int(chunk), int(device_id)
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise ValueError(f"data must be bytes, got {type(data).__name__}")
        check_lim_write(x, y, addr, len(data), chunk)
        write = self.noc_write or (None if pol.fake else _ttnn_write_to_core)
        if write is None:
            raise RuntimeError("x280_lim_write needs the real device (server started with --fake-model)")
        t0 = time.monotonic_ns()
        for off in range(0, len(data), chunk):
            write(device_id, x, y, addr + off, data[off:off + chunk])
        dt_us = (time.monotonic_ns() - t0) / 1e3
        self._rec = {"endpoint": "x280_lim_write", "x": x, "y": y, "addr": addr, "n_bytes": len(data),
                     "dt_us": round(dt_us, 1)}
        return {"x": x, "y": y, "addr": addr, "n_bytes": len(data), "n_writes": len(data) // chunk, "t_ns": t0,
                "dt_us": dt_us}

    def _count_dropped(self, dropped: tuple) -> None:
        for k in dropped:
            self.dropped_counts[k] += 1

    # -- loop -------------------------------------------------------------------------------------------------------
    def handle(self, frame: bytes) -> bytes:
        """One request -> reply bytes; any failure becomes ``{"error": str}`` (the stock client raises on it)."""
        self._rec = None
        ep = None
        t0 = time.perf_counter()
        try:
            req = from_bytes(frame)
            if not isinstance(req, dict):
                raise ValueError(f"request must be a map, got {type(req).__name__}")
            ep = req.get("endpoint", "get_action")
            if ep not in self.endpoints:
                raise ValueError(f"Unknown endpoint: {ep}")
            data = req.get("data") or {}
            if not isinstance(data, dict):
                raise ValueError(f"data must be a map, got {type(data).__name__}")
            if ep in DEBUG_ENDPOINTS and not self.debug_endpoints:
                raise DebugDisabled(DEBUG_DISABLED)
            t1 = time.perf_counter()
            res = self.endpoints[ep]() if ep in self.no_input else self.endpoints[ep](**data)
            t2 = time.perf_counter()
            out = to_bytes(res)
            if self._rec is not None:
                self._rec.update(wire_in_ms=round((t1 - t0) * 1e3, 3),
                                 wire_out_ms=round((time.perf_counter() - t2) * 1e3, 3))
        except Exception as e:  # noqa: BLE001 - reply like the stock server and keep serving
            self.n_errors += 1
            msg = DEBUG_DISABLED if isinstance(e, DebugDisabled) else f"{type(e).__name__}: {e}"
            if not isinstance(e, DebugDisabled):
                log(f"Error in server ({ep}): {msg}\n{traceback.format_exc()}")
            self._rec = {"endpoint": ep, "error": msg}
            out = to_bytes({"error": msg})
        return out

    def _write_log(self) -> None:
        """One JSONL line, after the reply. Echoed options.rt values are client data (any msgpack type), so the line is
        written with default=str, and a line that still fails (disk full, closed file) is counted, never fatal."""
        rec, self._rec = self._rec, None
        if self._fh is None or rec is None:
            return
        try:
            rec["t"] = now()
            self._fh.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:  # noqa: BLE001 - logging must not end serving
            self.n_log_errors += 1
            if self.n_log_errors == 1:
                log(f"JSONL log write failed ({type(e).__name__}: {e}); counting further failures in info")

    def serve(self, should_stop: Callable[[], bool], max_idle_s: float = 0.0, poll_ms: int = 200) -> None:
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        last = time.monotonic()
        while not should_stop():
            if not poller.poll(poll_ms):
                if max_idle_s > 0 and time.monotonic() - last > max_idle_s:
                    log(f"idle for {max_idle_s:.0f} s, exiting")
                    return
                continue
            last = time.monotonic()
            frame = self.sock.recv()
            self.t_recv_ns = time.monotonic_ns()  # before decode (the stale check uses it)
            reply = self.handle(frame)
            self.sock.send(reply)
            self._write_log()  # after the reply: logging never adds to the client's latency

    def close(self) -> None:
        try:
            if self.policy is not None:
                self.policy.close()
        finally:
            self.sock.close(linger=0)
            if self._own_ctx:
                self.ctx.term()
            if self._fh is not None:
                self._fh.close()


# ---------------------------------------------------------------------------------------------------------------- main
def build(args: argparse.Namespace, server: G1Server) -> Dict[str, Any]:
    """Install, load (opens the device unless --fake-model), warm + boot check; fills server.policy / static_info."""
    facts = G.install()
    tree = G.tree_stamp()
    runtime: Dict[str, Any] = {"runtime_root": None}  # the fake model compiles no kernels
    t0 = time.perf_counter()
    if args.fake_model:
        model, stamp = FakeG1Model(), fake_stamp()
        backend = "mock:server_g1-fake"
    else:
        runtime = G.pin_runtime_root()  # first: the device kernels must come from the imported tree
        from models.experimental.gr00t.tt.policy import TTPolicy

        policy = TTPolicy(mk_arena_dtype=getattr(args, "mk_arena_dtype", "auto"))  # WP-2.7 fallback: bf16
        model = G.load_tt_model(policy=policy,
                                cache_root=Path(args.cache_root).expanduser() if args.cache_root else None)
        stamp = G.backend_stamp(model)
        backend = f"tt:p150a:{G.REPO}@{G.SHA[:8]}"
    load_s = time.perf_counter() - t0
    server.policy = G1Policy(model, stamp, args.seed, fake=args.fake_model)
    log(f"loaded in {load_s:.1f} s; tree {tree}; backend {json.dumps(stamp)}")
    if args.fake_model or args.no_boot_check:
        why = "fake model" if args.fake_model else "--no-boot-check"
        mi, _ = G.encode_obs(G.split_batch(G.filter_obs(warm_obs())[0])[0])
        model.warm_and_capture(mi, G.make_noise(0))
        boot = {"skipped": why, "pass": None}
    else:
        boot = boot_sanity_check(server.policy, Path(args.golden), args.min_pcc)
    mi, _ = G.encode_obs(G.split_batch(G.filter_obs(warm_obs())[0])[0])
    warm = []
    for k in range(3):
        t = time.perf_counter()
        model.predict_normalized(mi, G.make_noise(k))
        warm.append(round((time.perf_counter() - t) * 1e3, 3))
    boot["warm_latency_ms"] = warm
    server.static_status = static_status_of(model, args.fake_model)
    server.static_info = {"backend": backend, "proto": PROTO, "model": G.REPO, "revision": G.SHA, "noise": NOISE,
                          "source": f"{G.SRC_REPO}@{G.SRC_SHA}", "tree": tree, "stamp": stamp, "install": facts,
                          "runtime_root": runtime["runtime_root"],
                          "boot_check": boot, "load_s": round(load_s, 3), "fake_model": bool(args.fake_model),
                          "pid": os.getpid(), "cache_root": str(args.cache_root or G.default_cache_root()),
                          "log": str(server.log_path) if server.log_path else None}
    return boot


def default_log_path() -> Path:
    return DEFAULT_LOG_DIR / f"calls_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl"


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="0 = a random free port (printed)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the per-call noise generator (fresh per call)")
    ap.add_argument("--golden", default=str(gio.GOLDEN_DIR))
    ap.add_argument("--min-pcc", type=float, default=DEFAULT_MIN_PCC)
    ap.add_argument("--no-boot-check", action="store_true", help="skip the golden sanity check (bring-up only)")
    ap.add_argument("--cache-root", default=None, help=f"weights cache (default {G.default_cache_root()})")
    ap.add_argument("--mk-arena-dtype", choices=("auto", "bf16", "bfp8_b"), default="auto",
                    help="megakernel weight arena (MK tt/policy.py:54-61; auto = bfp8_b for n16; bf16 = WP-2.7)")
    ap.add_argument("--log", default=None, help="per-call JSONL (default logs/server_g1/calls_<stamp>.jsonl; 'none')")
    ap.add_argument("--debug-endpoints", action="store_true",
                    help="enable predict_debug, x280_peek and x280_lim_write (test-only, outside D1)")
    ap.add_argument("--fake-model", action="store_true", help="deterministic chunks without ttnn (protocol tests)")
    ap.add_argument("--max-idle-s", type=float, default=0.0, help="exit after this many idle seconds (0 = never)")
    ap.add_argument("--cpus", default=DEFAULT_CPUS, help="CPU affinity ('' = inherit)")
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="torch / cv2 intra-op threads for the host pipeline (0 = library default)")
    ap.add_argument("--status-pub", nargs="?", const=DEFAULT_STATUS_URL, default=None, metavar="URL",
                    help=f"out-of-band status PUB (PLAN.md §4.3 item 7; bare flag = {DEFAULT_STATUS_URL}, "
                         "0 = a random port; default off)")
    ap.add_argument("--status-hz", type=float, default=DEFAULT_STATUS_HZ,
                    help=f"status message rate, clamped to {STATUS_HZ_RANGE[0]:g}-{STATUS_HZ_RANGE[1]:g} Hz")
    return ap


def set_host_threads(n: int) -> dict:
    """Host encode is small-tensor work: on 4 CPUs it costs ~2.4 ms with 1, 2 or 4 threads, but with 4 threads it
    rose to 30 ms median / 54 ms p90 once other processes loaded those CPUs, while 1 thread stayed at 2.5 ms
    (measured 2026-10-24 on this host, fake model). One thread is the robust default."""
    if n > 0:
        torch.set_num_threads(int(n))
        try:
            import cv2

            cv2.setNumThreads(int(n))
        except ImportError:
            pass
    return {"torch_threads": torch.get_num_threads()}


def parse_cpus(spec: str) -> set:
    """'8,9,10,11' or '0,1,8-11' -> {CPU ids}."""
    out: set = set()
    for part in str(spec).split(","):
        lo, _, hi = part.strip().partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv=None) -> int:
    a = make_parser().parse_args(argv)
    cpus = parse_cpus(a.cpus) if a.cpus else None
    safe = os.environ.get("CHIP_SAFE_CPUS")  # chip_lib.sh: a session confined to the non-real-time cores
    if cpus and safe and not cpus <= parse_cpus(safe):
        log(f"--cpus {a.cpus} is outside CHIP_SAFE_CPUS={safe}; refusing to start")
        return 2
    if cpus:
        os.sched_setaffinity(0, cpus)  # first: every thread started later (zmq, torch, tt-metal) inherits it
    numa = os.environ.pop("TT_METAL_NUMA_BASED_AFFINITY", None)  # read by tt-metal at device open (rtoptions.cpp)
    threads = set_host_threads(a.torch_threads)
    log_path = None if a.log == "none" else Path(a.log) if a.log else default_log_path()
    try:
        server = G1Server(a.host, a.port, log_path, a.debug_endpoints)
    except zmq.ZMQError as e:
        log(f"bind tcp://{a.host}:{a.port} failed: {e}")
        return 3
    log(f"bound {server.addr}; loading the {'fake' if a.fake_model else 'TT'} model (pid {os.getpid()})")
    if numa not in (None, "", "0"):
        log(f"unset TT_METAL_NUMA_BASED_AFFINITY={numa}: it would rebind the main thread off --cpus")
    pub = None
    if a.status_pub is not None:
        try:
            pub = StatusPublisher(server, a.status_pub, a.status_hz).start()
            server.status_addr = pub.addr
            log(f"status PUB {pub.addr} every {pub.period * 1e3:.0f} ms")
        except zmq.ZMQError as e:  # out-of-band: a taken status port never stops the VLA server
            log(f"status PUB {a.status_pub} not bound ({e}); serving without it")
    stop = {"flag": False}

    def _term(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    rc = 0
    try:
        try:
            boot = build(a, server)
        except Exception as e:  # noqa: BLE001 - no READY line, exit 2: chip_session.sh sees the failure
            log(f"start-up failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return 2
        if cpus:
            server.static_info["affinity_check"] = chk = enforce_affinity(cpus)
            if chk["repinned"]:
                log(f"re-pinned threads outside --cpus {a.cpus}: {json.dumps(chk['repinned'])}")
        server.static_info.update(threads, affinity=sorted(os.sched_getaffinity(0)))
        log(f"warm + boot check: {json.dumps(boot)}")
        server.t_ready_ns = time.monotonic_ns()
        print(f"server_g1 READY {server.addr} {json.dumps({k: server.static_info[k] for k in ('backend', 'tree')})}",
              flush=True)
        print(f"Server is ready and listening on {server.addr}", flush=True)
        server.serve(lambda: stop["flag"], a.max_idle_s)
    finally:
        info = server.info()
        log("shutting down: " + json.dumps({k: info.get(k) for k in ("n_calls", "n_obs", "n_errors", "n_stale",
                                                                      "latency_ms_p50", "latency_ms_p90")}))
        if pub is not None:
            pub.close()
        server.close()
        log("model released")
    return rc


if __name__ == "__main__":
    sys.exit(main())
