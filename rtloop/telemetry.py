"""Run directory, crash-safe tick log, request and episode logs, telemetry PUB and run stamp (PLAN.md §4.1, §1.4.5).

Only `TickLog.next/commit` and the `Publisher` sends sit on the tick path. They make no syscall except on every
50th commit (two small pwrite calls that publish the committed length) and one non-blocking zmq send. The tick
rows live in a MAP_SHARED memmap, so they are in the page cache the moment they are written and survive a
process crash without msync (which costs 3-10 ms and is kept off the tick path).
"""

import argparse
import ctypes
import enum
import hashlib
import io
import json
import math
import mmap
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat

from rtloop import wire
from rtloop.schema import REQUEST_DTYPE, TICK_DTYPE, Outcome

RT = Path(__file__).resolve().parents[1]
RUNS = RT / "runs"
RUNS_ENV = "RTLOOP_RUNS_ROOT"  # overrides RunDir's default root (eval_paired points each job at its own root)
TELEMETRY_PUB = "tcp://127.0.0.1:5570"
FLUSH_EVERY = 50  # ticks between publications of the committed length (1 s at 50 Hz)
# CHIP_SESSION_*: which device session (scripts/chip_session.sh) a run belongs to; analyze_run finds x280d's
# tick ring through CHIP_SESSION_DIR (x280_heartbeat_continuity, T-3.7).
CONFIG_ENV = ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "GR00T_WBC_SERVER",
              "CHIP_SESSION_ID", "CHIP_SESSION_DIR")
# Distributions whose versions decide bit-exact results (ORT kernels, MuJoCo physics, wire, env wrappers).
CONFIG_PACKAGES = ("numpy", "onnxruntime", "pyzmq", "msgpack", "mujoco", "gymnasium", "torch", "robosuite",
                   "robocasa", "PyOpenGL")
TERMINAL_OUTCOMES = frozenset(
    {Outcome.LATE_REJECT, Outcome.GEN_DROP, Outcome.TIMEOUT, Outcome.ERROR, Outcome.DRAINED}
)


def _trees() -> dict:
    """The three source trees a run depends on (env.sh exports R and W; defaults match it)."""
    r = Path(os.environ.get("R") or Path.home() / "experiments/gr00t/Isaac-GR00T-n1.6")
    w = Path(os.environ.get("W") or r / "external_dependencies/GR00T-WholeBodyControl")
    return {"RT": RT, "R": r, "W": w}


def jsonable(x):
    """Plain-JSON copy of x: numpy -> Python, enums -> names, non-finite floats -> None, bytes -> hex."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(jsonable(v) for v in x)
    if isinstance(x, enum.Enum):
        return x.name
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).hex()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, argparse.Namespace):
        return jsonable(vars(x))
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return repr(x)


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)


# --- Run directory ------------------------------------------------------------------------------


class RunDir:
    """`<root>/<stamp>_<tag>/`: every artefact of one invocation in one place, `<root>/latest` -> newest."""

    def __init__(self, root=None, tag: str = "", config: dict | None = None):
        self.root = Path(root or os.environ.get(RUNS_ENV) or RUNS)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tag = str(tag).replace("/", "_")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{self.tag}" if self.tag else ""
        for i in range(1000):  # mkdir is the atomic claim, so parallel runs never share a dir
            name = f"{stamp}{f'-{i}' if i else ''}{suffix}"
            try:
                (self.root / name).mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f"no free run dir name under {self.root}")
        self.path = self.root / name
        self._link_latest()
        self.config = dict(config or {})
        self.write_json("config.json", self.config)

    def _link_latest(self) -> None:
        tmp = self.root / f".latest.{os.getpid()}"
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        os.symlink(self.path.name, tmp)
        os.replace(tmp, self.root / "latest")

    def file(self, name: str) -> Path:
        return self.path / name

    def write_json(self, name: str, obj) -> Path:
        path = self.path / name
        _write_json_atomic(path, obj)
        return path

    def update_config(self, **fields) -> None:
        """Add stamps learned after start (backend `info`, endpoints) to config.json."""
        self.config.update(fields)
        self.write_json("config.json", self.config)


def _run_path(run) -> Path:
    return run.path if isinstance(run, RunDir) else Path(run)


# --- TickLog ------------------------------------------------------------------------------------


class TickLog:
    """`ticks.npy`: a TICK_DTYPE memmap filled in place, loadable with np.load at any moment.

    Every 50 commits the .npy header's shape and the `ticks.len` sidecar are rewritten to the committed count,
    so after a crash np.load returns exactly the rows published so far (numpy pads the header so its length
    does not depend on the shape's digits; checked at every rewrite). Capacity grows by re-mapping in blocks
    of `capacity` rows; close() truncates the file to the committed rows.
    """

    def __init__(self, run, capacity: int = 1 << 20):
        path = _run_path(run)
        self.path = path / "ticks.npy"
        self.len_path = path / "ticks.len"
        self._block = int(capacity)
        self._cap = int(capacity)
        self._mm = npformat.open_memmap(self.path, mode="w+", dtype=TICK_DTYPE, shape=(self._cap,))
        self._offset = int(self._mm.offset)
        with open(self.path, "rb") as f:
            self._version = npformat.read_magic(f)
        self._descr = npformat.dtype_to_descr(TICK_DTYPE)
        self._fd = os.open(self.path, os.O_RDWR)
        self._len_fd = os.open(self.len_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        self.n = 0  # committed rows
        self._released_to = 0  # address up to which release() has dropped pages of the current mapping
        self._issued = -1  # index of the last row handed out by next()
        self._closed = False
        self._publish(0)

    def __len__(self) -> int:
        return self.n

    def _header(self, n: int) -> bytes:
        buf = io.BytesIO()
        d = {"descr": self._descr, "fortran_order": False, "shape": (n,)}
        if self._version == (1, 0):
            npformat.write_array_header_1_0(buf, d)
        else:
            npformat.write_array_header_2_0(buf, d)
        out = buf.getvalue()
        if len(out) != self._offset:
            raise RuntimeError(f"npy header length changed ({len(out)} != {self._offset})")
        return out

    def _publish(self, n: int) -> None:
        os.pwrite(self._fd, self._header(n), 0)
        os.pwrite(self._len_fd, b"%-20d\n" % n, 0)

    def _grow(self) -> None:
        cap = self._cap + self._block
        os.ftruncate(self._fd, self._offset + cap * TICK_DTYPE.itemsize)
        self._mm = np.memmap(self.path, dtype=TICK_DTYPE, mode="r+", offset=self._offset, shape=(cap,))
        self._cap = cap
        self._released_to = 0  # a new mapping (new addresses)

    def next(self) -> np.void:
        """The next row, zeroed, as a writable view into the memmap (fill fields in place, then commit)."""
        if self._closed:
            raise RuntimeError("TickLog is closed")
        i = self.n
        if i >= self._cap:
            self._grow()
        if self._issued == i:  # handed out before without a commit: clear what was written
            self._mm[i] = 0
        self._issued = i
        return self._mm[i]

    def commit(self) -> None:
        if self._issued != self.n:
            raise RuntimeError("commit() without next()")
        self.n += 1
        if self.n % FLUSH_EVERY == 0:
            self._publish(self.n)

    def prefault(self, rows: int = 1440) -> None:
        """Touch the next `rows` rows (reset window) so first-write page faults (20-200 us) leave the episode."""
        while self.n + rows > self._cap:
            self._grow()
        self._mm[self.n + 1 : self.n + rows] = 0  # row n may already be handed out by next()

    def flush(self, sync: bool = False) -> None:
        """Publish the committed length now; sync=True also msyncs (reset windows only: 3-10 ms)."""
        self._publish(self.n)
        if sync:
            self._mm.flush()

    def release(self) -> int:
        """Drop the committed rows' pages from this process's RSS (reset window, after flush(sync=True)).

        The rows stay in the page cache and the file (MADV_DONTNEED on a MAP_SHARED file mapping only
        unmaps; dirty data is kept), and reading them again faults them back in. Without this the memmap
        adds 0.9 MB of resident pages per 1440-tick episode, 227 MB over T-1.10's 2 h, which would read as
        an RSS leak. Each call covers the rows committed since the last one; returns the bytes of that range
        (0 if madvise is unavailable)."""
        if self._closed or self.n == 0:
            return 0
        page = mmap.PAGESIZE
        base = self._mm.ctypes.data
        start = max((base // page) * page, self._released_to)  # from the header page on: re-read from the file
        end = ((base + self.n * TICK_DTYPE.itemsize) // page) * page  # keep row n's (partial) page
        if end <= start:
            return 0
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            rc = libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), mmap.MADV_DONTNEED)
        except (OSError, AttributeError):
            return 0
        if rc != 0:
            return 0
        self._released_to = end
        return end - start

    def rows(self) -> np.ndarray:
        """The committed rows (a view; valid until close)."""
        return self._mm[: self.n]

    def close(self) -> None:
        """Publish, msync and truncate to the committed rows; views from next()/rows() are dead afterwards."""
        if self._closed:
            return
        self._publish(self.n)
        self._mm.flush()
        self._mm = None
        os.ftruncate(self._fd, self._offset + self.n * TICK_DTYPE.itemsize)
        os.fsync(self._fd)
        os.close(self._fd)
        os.close(self._len_fd)
        self._closed = True


def load_ticks(run_dir, mmap: bool = False) -> np.ndarray:
    """ticks.npy of a run, cut to the committed length (min of header and sidecar after a crash)."""
    run_dir = Path(run_dir)
    path = run_dir / "ticks.npy"
    if not path.exists():
        return np.zeros(0, dtype=TICK_DTYPE)
    arr = np.load(path, mmap_mode="r" if mmap else None)
    try:
        n = int((run_dir / "ticks.len").read_text().split()[0])
        arr = arr[: min(len(arr), n)]
    except (FileNotFoundError, IndexError, ValueError):
        pass
    return arr


# --- RequestLog ---------------------------------------------------------------------------------


def _sha1_array(v) -> np.ndarray:
    if isinstance(v, str):
        v = bytes.fromhex(v)
    if isinstance(v, (bytes, bytearray)):
        v = np.frombuffer(bytes(v), dtype=np.uint8)
    v = np.asarray(v, dtype=np.uint8).reshape(-1)
    if v.shape != (20,):
        raise ValueError(f"obs_sha1 must be 20 bytes, got {v.shape}")
    return v


class RequestLog:
    """One REQUEST_DTYPE row per VLA request, kept in memory and updated by seq as the request plays out.

    `add` and `update` run inside ticks, so they do no file I/O and no allocation beyond the row's dict:
    every row is appended to `requests.jsonl` at its episode's end (a row updated after it was written is
    appended again at the next episode end; readers keep the last line per seq), and `end_episode()` grows the
    array ahead of need, so the next episode's `reserve` adds fit without a copy (the copy at 4096/8192/16384
    rows costs 1.6/3.0/4.0 ms). `requests.npz` (key `requests` = the structured array, plus one column per
    field) is rewritten atomically at every episode end and at close. The episode ends when `add` sees a new
    `ep`, on `end_episode()`, or at close; all three run in the reset window. Keys of `row` that are not
    REQUEST_DTYPE fields (e.g. server INFO timings) go to the jsonl only, and are dropped from memory once
    their row is two episode ends old (T-1.10: ~0.9 KB per request otherwise kept for the whole run); a later
    update of such a row re-appends its array fields plus the new extras, and readers merge extras per seq.
    """

    def __init__(self, run, capacity: int = 4096, reserve: int = 2048):
        path = _run_path(run)
        self.jsonl_path = path / "requests.jsonl"
        self.npz_path = path / "requests.npz"
        self._arr = np.zeros(capacity, dtype=REQUEST_DTYPE)
        self.reserve = int(reserve)  # free rows end_episode() guarantees (> one episode's requests)
        self.n_grow_in_add = 0  # fallback growths inside add(), i.e. inside a tick (should stay 0)
        self.n = 0
        self._by_seq: dict[int, int] = {}
        self._extra: list[dict | None] = []
        self._dirty: set[int] = set()
        self._ep: int | None = None
        self._ep_start = 0  # rows below it are written: end_episode() writes every row of its episode
        self._prev_ep_start = 0  # rows below it have had their extras dropped
        self._f = open(self.jsonl_path, "a")
        self._closed = False

    def __len__(self) -> int:
        return self.n

    def rows(self) -> np.ndarray:
        return self._arr[: self.n]

    def get(self, seq: int) -> np.void:
        return self._arr[self._by_seq[int(seq)]]

    def _set(self, i: int, fields: dict) -> None:
        for key, value in fields.items():
            if key in REQUEST_DTYPE.names:
                self._arr[key][i] = _sha1_array(value) if key == "obs_sha1" else value
            else:
                if self._extra[i] is None:
                    self._extra[i] = {}
                self._extra[i][key] = value
        if i < self._ep_start:
            self._dirty.add(i)

    def add(self, row: dict) -> int:
        """Register a sent request (needs `seq`; `ep` defaults to 0). Returns its row index."""
        ep = int(row.get("ep", 0))
        if self._ep is not None and ep != self._ep:
            self.end_episode()
        self._ep = ep
        if self.n == len(self._arr):  # fallback only: end_episode() keeps `reserve` rows free
            self._arr = np.concatenate([self._arr, np.zeros(len(self._arr), dtype=REQUEST_DTYPE)])
            self.n_grow_in_add += 1
        i = self.n
        self.n += 1
        self._extra.append({})
        self._by_seq[int(row["seq"])] = i
        self._set(i, row)
        return i

    def update(self, seq: int, **fields) -> None:
        self._set(self._by_seq[int(seq)], fields)

    def record(self, i: int) -> dict:
        r = self._arr[i]
        out = {name: jsonable(r[name]) for name in REQUEST_DTYPE.names}
        out["obs_sha1"] = bytes(r["obs_sha1"]).hex()
        try:
            out["outcome_name"] = Outcome(int(r["outcome"])).name
        except ValueError:
            out["outcome_name"] = str(int(r["outcome"]))
        out.update(jsonable(self._extra[i] or {}))
        return out

    def _write(self, i: int) -> None:
        self._f.write(json.dumps(self.record(i), allow_nan=False) + "\n")
        self._f.flush()
        self._dirty.discard(i)

    def end_episode(self) -> None:
        """Finalise the current episode's rows (jsonl) and rewrite requests.npz. Call in the reset window."""
        for i in sorted(self._dirty):
            self._write(i)
        for i in range(self._ep_start, self.n):
            self._write(i)
        for i in range(self._prev_ep_start, self._ep_start):
            self._extra[i] = None
        self._prev_ep_start, self._ep_start = self._ep_start, self.n
        self._ep = None
        if self.n + self.reserve > len(self._arr):
            grow = max(len(self._arr), self.reserve)
            self._arr = np.concatenate([self._arr, np.zeros(grow, dtype=REQUEST_DTYPE)])
        self.save_npz()

    def save_npz(self) -> None:
        rows = self._arr[: self.n]
        tmp = self.npz_path.with_name(".requests.tmp.npz")
        with open(tmp, "wb") as f:
            np.savez(f, requests=rows, **{name: rows[name] for name in REQUEST_DTYPE.names})
        os.replace(tmp, self.npz_path)

    def close(self) -> None:
        if self._closed:
            return
        self.end_episode()
        self._f.close()
        self._closed = True


def load_requests(run_dir) -> np.ndarray:
    """requests.npz of a run, else rebuilt from requests.jsonl (last line per seq wins), else empty."""
    run_dir = Path(run_dir)
    npz = run_dir / "requests.npz"
    if npz.exists():
        with np.load(npz) as z:
            return z["requests"].copy()
    path = run_dir / "requests.jsonl"
    if not path.exists():
        return np.zeros(0, dtype=REQUEST_DTYPE)
    last: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            last[int(rec["seq"])] = rec
    out = np.zeros(len(last), dtype=REQUEST_DTYPE)
    for i, rec in enumerate(sorted(last.values(), key=lambda r: int(r["seq"]))):
        for name in REQUEST_DTYPE.names:
            if name in rec and rec[name] is not None:
                out[name][i] = _sha1_array(rec[name]) if name == "obs_sha1" else rec[name]
    return out


def load_request_extras(run_dir) -> dict[int, dict]:
    """{seq: jsonl-only keys} of requests.jsonl, merged over a seq's lines (later lines win per key; a row
    updated after RequestLog dropped its extras is re-appended without them): server stamps (srv_t_recv_ns,
    srv_t_reply_ns), drain result, error text. {} when there is no jsonl."""
    path = Path(run_dir) / "requests.jsonl"
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out.setdefault(int(rec["seq"]), {}).update(
                (k, v) for k, v in rec.items() if k not in REQUEST_DTYPE.names)
    return out


# --- EpisodeLog ---------------------------------------------------------------------------------


class EpisodeLog:
    """`episodes.jsonl`, one object per finished episode (PLAN.md §4.1 fields), flushed per line."""

    def __init__(self, run):
        self.path = _run_path(run) / "episodes.jsonl"
        self._f = open(self.path, "a")
        self.records: list[dict] = []

    def add(self, rec: dict) -> None:
        rec = jsonable(rec)
        self._f.write(json.dumps(rec, allow_nan=False) + "\n")
        self._f.flush()
        self.records.append(rec)

    def close(self) -> None:
        self._f.close()


def load_episodes(run_dir) -> list[dict]:
    path = Path(run_dir) / "episodes.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- Telemetry PUB ------------------------------------------------------------------------------


def tick_msg(row: np.void) -> dict:
    """The §1.4.5 `tick` topic fields of one TickLog row."""
    return {
        "k": int(row["k"]), "ep": int(row["ep"]), "t_ns": int(row["t_start_ns"]),
        "period_us": float(row["period_us"]), "compute_us": float(row["compute_us"]),
        "late_us": float(row["late_us"]), "slack_us": float(row["slack_us"]),
        "overrun": int(row["overrun"]), "slip": int(row["slip"]), "sim_us": float(row["sim_us"]),
        "viewer_us": float(row["viewer_us"]), "wbc_rtt_us": float(row["wbc_rtt_us"]),
        "wbc_dev_us": float(row["wbc_dev_us"]), "wbc_be": int(row["wbc_be"]),
        "wbc_pol": int(row["policy"]), "wbc_fb": int(row["wbc_fb"]),
        "shadow_da": float(row["shadow_da"]), "seq": int(row["chunk_seq"]), "row": int(row["row"]),
        "rows_left": int(row["rows_left"]), "starved": int(row["starved"]),
        "inflight": int(row["in_flight"]), "splice": int(row["splice"]),
        "jump_rad": float(row["jump_rad"]), "nav": row["nav_wbc"].tolist(),
        "pelvis_z": float(row["pelvis_z"]), "success": int(row["success"]),
    }


class Publisher:
    """PUB socket for the HUD: multipart [topic, wire msgpack], NOBLOCK, dropped when the HWM is reached.

    The socket is an XPUB with XPUB_NODROP, so a message that would exceed a subscriber's HWM makes the
    NOBLOCK send raise zmq.Again (counted in `dropped`) instead of vanishing silently as on a plain PUB;
    the send still never blocks, so the HUD cannot stall the loop. With NODROP a full pipe drops the message
    for every subscriber (there is one, the HUD). SUB peers connect to it as to a PUB. `url` None makes every
    method a no-op (loop without --pub); a port of `*` binds a random free port (tests).
    """

    def __init__(self, url: str | None = TELEMETRY_PUB, ctx=None, hwm: int = 100):
        self.url = url
        self.sent = 0
        self.dropped = 0
        self._sock = None
        self._own_ctx = False
        if url is None:
            return
        import zmq

        self._zmq = zmq
        if ctx is None:
            ctx = zmq.Context(io_threads=1)
            self._own_ctx = True
        self._ctx = ctx
        sock = ctx.socket(zmq.XPUB)
        sock.setsockopt(zmq.XPUB_NODROP, 1)
        sock.setsockopt(zmq.SNDHWM, hwm)
        sock.setsockopt(zmq.LINGER, 0)
        base, _, port = url.rpartition(":")
        if port in ("*", "0"):
            self.url = f"{base}:{sock.bind_to_random_port(base)}"
        else:
            sock.bind(url)
        self._sock = sock

    def _send(self, topic: bytes, obj) -> None:
        if self._sock is None:
            return
        try:
            self._sock.send_multipart((topic, wire.dumps(obj)), flags=self._zmq.NOBLOCK)
            self.sent += 1
        except self._zmq.Again:
            self.dropped += 1

    def tick(self, fields: dict) -> None:
        self._send(b"tick", fields)

    def req(self, fields: dict) -> None:
        self._send(b"req", fields)

    def ego(self, seq: int, k_obs: int, img_small: np.ndarray) -> None:
        self._send(b"ego", {"seq": int(seq), "k_obs": int(k_obs), "img": np.ascontiguousarray(img_small)})

    def meta(self, fields: dict) -> None:
        self._send(b"meta", fields)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
            if self._own_ctx:
                self._ctx.term()


# --- Opt-in dumps -------------------------------------------------------------------------------


def save_obs516(run, dump: list) -> Path:
    """`obs516.npz` from WbcRuntime.dump entries (tick, policy, obs (1,516), action (1,15))."""
    path = _run_path(run) / "obs516.npz"
    ticks = np.array([d[0] for d in dump], dtype=np.int64)
    policy = np.array([str(d[1]) for d in dump])
    obs = np.stack([np.asarray(d[2], dtype=np.float32).reshape(-1) for d in dump]) if dump else np.zeros((0, 516), np.float32)
    act = np.stack([np.asarray(d[3], dtype=np.float32).reshape(-1) for d in dump]) if dump else np.zeros((0, 15), np.float32)
    np.savez(path, tick=ticks, policy=policy, obs=obs, action=act)
    return path


def save_vla_obs(run, seq: int, vla_obs: dict) -> Path:
    """`vla_obs/<seq:06d>.npz`: one sent VLA observation (language as a 0-d str array)."""
    d = _run_path(run) / "vla_obs"
    d.mkdir(exist_ok=True)
    path = d / f"{int(seq):06d}.npz"
    arrays = {k.replace(".", "__"): (np.asarray(v) if not isinstance(v, (tuple, list)) else np.array(v[0]))
              for k, v in vla_obs.items()}
    np.savez(path, **arrays)
    return path


# --- Run stamp ----------------------------------------------------------------------------------


def _git(path: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _porcelain_paths(lines: list[str]) -> list[tuple[str, str]]:
    """(status, path) per `git status --porcelain` line; renames give the new path."""
    out = []
    for line in lines:
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        out.append((code.strip() or code, path.strip('"')))
    return out


def git_state(path: Path) -> dict:
    """HEAD and `git status --porcelain` of one tree, plus the same for nested repos it lists as untracked."""
    path = Path(path)
    top = _git(path, "rev-parse", "--show-toplevel")
    head = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--porcelain", "--untracked-files=all")
    lines = status.splitlines() if status is not None else None
    st = {"path": str(path), "toplevel": top.strip() if top else None, "head": head.strip() if head else None,
          "status": lines, "dirty": bool(lines)}
    nested = {}
    for code, rel in _porcelain_paths(lines or []):
        if code == "??" and rel.endswith("/") and (Path(st["toplevel"] or path) / rel / ".git").exists():
            sub = git_state(Path(st["toplevel"] or path) / rel)
            sub.pop("nested", None)
            nested[rel] = sub
    if nested:
        st["nested"] = nested
    return st


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def imported_dirty(states: dict) -> list[dict]:
    """sha256 of every imported module file that is modified or untracked in one of the trees.

    A module belongs to the most specific tree containing it (W sits inside R). An untracked directory
    (`?? dir/`, e.g. a nested repo) marks every file under it as untracked.
    """
    trees = []
    for name, st in states.items():
        root = Path(st.get("toplevel") or st["path"]).resolve()
        files, dirs = {}, []
        for code, rel in _porcelain_paths(st.get("status") or []):
            if rel.endswith("/"):
                dirs.append((rel, code))
            else:
                files[rel] = code
        trees.append((len(str(root)), name, root, files, dirs))
    trees.sort(reverse=True)
    out, seen = [], set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        for _, name, root, files, dirs in trees:
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                continue
            code = files.get(rel) or next((c for d, c in dirs if rel.startswith(d)), None)
            if code is not None:
                out.append({"tree": name, "path": rel, "status": code, "sha256": _sha256(p)})
            break
    return sorted(out, key=lambda d: (d["tree"], d["path"]))


def package_versions(names=CONFIG_PACKAGES) -> dict:
    """{distribution: version} of the installed ones (importlib.metadata; nothing is imported)."""
    from importlib import metadata

    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def collect_config(argv, extra: dict | None = None, trees: dict | None = None) -> dict:
    """config.json content: what ran, from which source state, on which CPUs (PLAN.md §4.1 "Other files").

    `extra` (resolved CLI args, endpoints, backend info stamps, ORT options, fidelity deviations) is merged
    in at top level. `trees` overrides {RT, R, W} (tests).
    """
    trees = trees or _trees()
    states = {name: git_state(path) for name, path in trees.items()}
    cfg = {
        "argv": list(argv),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.executable,
        "numpy": np.__version__,
        "versions": package_versions(),
        "env": {k: os.environ.get(k) for k in CONFIG_ENV},
        "affinity": sorted(os.sched_getaffinity(0)),
        "git": states,
        "imported_dirty": imported_dirty(states),
    }
    sums = RT / "rtserve" / "vendor" / "SHA256SUMS"
    if sums.exists():
        cfg["vendor_sha256sums"] = sums.read_text().splitlines()
    for key, value in (extra or {}).items():
        cfg[key] = value
    return jsonable(cfg)
