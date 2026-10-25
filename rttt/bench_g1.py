#!/usr/bin/env python
"""Latency, stale-state (ABAB) and determinism bench of the G1 VLA server through ZMQ (PLAN.md T-2.3, T-2.4, WP-2.4).

    simpy $RT/rttt/bench_g1.py --endpoint $VLA_TT --n 500 --period-ms 320 --out $CHIP_SESSION_DIR/t23_bench.json
    simpy $RT/rttt/bench_g1.py --endpoint $VLA_TT --abab 200 --determinism --out ...   (server_g1 --debug-endpoints)

A client only: numpy, pyzmq, msgpack and rtloop's wire/schema (any interpreter; the jobs use simpy). It speaks the
W-VLA wire (PLAN.md §1.4.1) and the test-only ``predict_debug(observation, noise f32(1,50,128))`` endpoint.

Latency (--n): paced get_action calls, one every --period-ms, after --warmup calls (§4.4: 20), shaped like the loop's
(the 9 W-VLA keys, options.rt without noise_seed). Per call: client RTT (send -> reply, CLOCK_MONOTONIC), the
server's encode/device/decode/total ms from INFO, the wire time on each side from the server's t_recv_ns/t_reply_ns
(same host clock), and d_est = ceil((RTT + send offset) / 20 ms), the delay in ticks a 50 Hz loop polling at tick
starts would see (T-2.6 measures the real d). Every call over --slow-ms (100) is listed and attributed to its largest
component. T-2.3 thresholds (projections): RTT p50 <= 48 ms, p99 <= 55 ms; server total p50 <= 45 ms; d p99 <= 3.

ABAB (--abab N, T-2.4): predict_debug on obs A and B with one fixed noise, each called twice in a row (the second
call, whose predecessor was the same obs, is the single-call reference; the two must be bit-identical), then N calls
alternating A, B, A, B; every output must equal its reference bit-exactly (a stale image, state or cache from the
previous call shows up as a mismatch). A and B must give different outputs, or the check proves nothing. A is the
first observation and B the first later one whose 43 state values AND image both differ from A's (abab_pair): on the
n16_g1 golden s00-s09 are resets with bit-identical state, so B = s10 (mid-episode), and a server whose state input
lags one call cannot pass. No such pair exits 2.

Status stream (--status-sub URL): a SUB on server_g1 --status-pub for the length of the run; checks that messages
arrive (src server_g1, 1.5-12 Hz, the same backend as INFO, calls advancing) and stores the first and last message.

Determinism (--determinism): --replays (5) predict_debug calls with one fixed noise are bit-identical; two
get_action calls on the same obs without a noise seed differ (fresh noise per call, D10); two with the same
options.rt.noise_seed are identical (D17).

Obs: --obs PATH (repeatable): a golden dir (obs_index.json + obs/<sid>.npz, rttt/golden_io.py), a loop run dir or
its vla_obs/ dir, or one .npz of either kind. Default: the golden obs if present, else two synthetic obs (fine for
latency, meaningless for anything else). Writes --out JSON ({"pass", "checks": [{name, value, threshold, pass}],
"latency", "abab", "determinism", "info"}); exits 0 if every check passed, 1 if one failed, 2 on bad input or an
unreachable server.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
import zmq

RT = Path(__file__).resolve().parents[1]
if str(RT) not in sys.path:
    sys.path.insert(0, str(RT))

from rtloop import schema, wire  # noqa: E402
from rttt import golden_io as gio  # noqa: E402

TICK_MS = schema.DT * 1e3
NOISE_SHAPE = gio.NOISE_SHAPE
SYNTH_TASK = "pick up the apple and place it on the plate"  # synthetic obs only (latency smoke)
T23 = {"rtt_p50_ms": 48.0, "rtt_p99_ms": 55.0, "server_total_p50_ms": 45.0, "d_p99": 3}  # PLAN.md T-2.3


# --- client ---------------------------------------------------------------------------------------------------------


class ServerError(RuntimeError):
    """The server replied ``{"error": ...}`` (PolicyServer convention)."""


@dataclass
class Reply:
    value: object
    t_send_ns: int
    t_recv_ns: int
    ser_us: float
    deser_us: float
    nbytes: int

    @property
    def rtt_ms(self) -> float:
        return (self.t_recv_ns - self.t_send_ns) / 1e6


class G1Client:
    """Blocking REQ client of the W-VLA wire, one request at a time; a timeout re-creates the socket (lazy pirate)."""

    def __init__(self, url: str, timeout_ms: int = 30000, ctx: zmq.Context | None = None):
        self.url, self.timeout_ms = url, int(timeout_ms)
        self.ctx = ctx or zmq.Context.instance()
        self._open()

    def _open(self) -> None:
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.url)

    def call(self, endpoint: str, data: dict | None = None, timeout_ms: int | None = None) -> Reply:
        t0 = time.perf_counter_ns()
        msg = wire.request(endpoint, data)
        ser_us = (time.perf_counter_ns() - t0) / 1e3
        t_send = time.monotonic_ns()
        self.sock.send(msg)
        if not self.sock.poll(self.timeout_ms if timeout_ms is None else int(timeout_ms), zmq.POLLIN):
            self.sock.close(0)
            self._open()
            raise TimeoutError(f"{endpoint} on {self.url}: no reply in {timeout_ms or self.timeout_ms} ms")
        raw = self.sock.recv()
        t_recv = time.monotonic_ns()
        t1 = time.perf_counter_ns()
        value = wire.loads(raw)
        deser_us = (time.perf_counter_ns() - t1) / 1e3
        if isinstance(value, dict) and "error" in value:
            raise ServerError(f"{endpoint} on {self.url}: {value['error']}")
        return Reply(value, t_send, t_recv, ser_us, deser_us, len(raw))

    def ping(self, timeout_ms: int = 2000) -> bool:
        try:
            self.call("ping", timeout_ms=timeout_ms)
            return True
        except (TimeoutError, ServerError, zmq.ZMQError):
            return False

    def info(self, timeout_ms: int = 5000) -> dict:
        try:
            v = self.call("info", timeout_ms=timeout_ms).value
        except (TimeoutError, ServerError):
            return {}
        return v if isinstance(v, dict) else {}

    def get_action(self, obs: dict, rt: dict | None = None, timeout_ms: int | None = None) -> tuple[dict, dict, Reply]:
        """W-VLA get_action -> (action dict, INFO dict, Reply)."""
        r = self.call("get_action", {"observation": obs, "options": {"rt": rt} if rt is not None else None},
                      timeout_ms)
        if not (isinstance(r.value, (list, tuple)) and len(r.value) == 2 and isinstance(r.value[0], dict)):
            raise ServerError(f"get_action: expected [action, info], got {type(r.value).__name__}")
        action, info = r.value
        return action, (info if isinstance(info, dict) else {}), r

    def predict_debug(self, obs: dict, noise: np.ndarray, timeout_ms: int | None = None) -> tuple[dict, Reply]:
        """Test-only endpoint (server_g1 --debug-endpoints, PLAN.md §1.4.1) -> (reply dict, Reply)."""
        noise = np.ascontiguousarray(noise, dtype=np.float32)
        assert noise.shape == NOISE_SHAPE, noise.shape
        r = self.call("predict_debug", {"observation": obs, "noise": noise}, timeout_ms)
        v = r.value[0] if isinstance(r.value, (list, tuple)) and r.value else r.value
        if not (isinstance(v, dict) and "action_pred_norm" in v):
            raise ServerError("predict_debug: reply has no action_pred_norm")
        return v, r

    def close(self) -> None:
        self.sock.close(0)


def outputs(debug_reply: dict) -> dict:
    """The arrays of a predict_debug reply that must be bit-identical: action_pred_norm and every decoded action."""
    out = {"action_pred_norm": np.asarray(debug_reply["action_pred_norm"])}
    acts = debug_reply.get("actions") or {}
    out.update({k: np.asarray(v) for k, v in sorted(acts.items())})
    return out


def compare(a: dict, b: dict) -> tuple[bool, float]:
    """(bit-identical, max |a - b| over the shared keys); differing keys, shapes or dtypes are never identical."""
    same = a.keys() == b.keys()
    worst = 0.0
    for k in a.keys() & b.keys():
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            return False, math.inf
        same = same and x.tobytes() == y.tobytes()
        if x.size:
            worst = max(worst, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
    return same, worst


def action_bytes(action: dict) -> bytes:
    return b"".join(np.ascontiguousarray(action[k]).tobytes() for k in sorted(action))


def fixed_noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(NOISE_SHAPE).astype(np.float32)


# --- observations ---------------------------------------------------------------------------------------------------


def synthetic_obs(seed: int) -> dict:
    """A random but well-formed W-VLA observation (latency smoke only)."""
    rng = np.random.default_rng(seed)
    obs = {schema.VIDEO_KEY: rng.integers(0, 256, (1, 1, *schema.IMAGE_SHAPE), dtype=np.uint8)}
    for (_, n), key in zip(schema.STATE_GROUPS, schema.STATE_KEYS, strict=True):
        obs[key] = rng.uniform(-0.3, 0.3, (1, 1, n)).astype(np.float32)
    obs[schema.LANGUAGE_KEY] = (SYNTH_TASK,)
    return obs


def load_npz_obs(path: Path) -> dict:
    """One .npz in either key style: the loop's dump ('.' -> '__', telemetry.save_vla_obs) or golden_io's own."""
    with np.load(path) as z:
        dumped = gio.dump_key(schema.VIDEO_KEY) in z.files
    return gio.load_loop_dump(path) if dumped else gio.load_obs(path)


def load_obs_set(paths: list[str] | None) -> list[tuple[str, dict]]:
    """[(name, W-VLA obs)] from --obs paths (see the module doc)."""
    if not paths:
        if (gio.GOLDEN_DIR / "obs_index.json").exists():
            paths = [str(gio.GOLDEN_DIR)]
        else:
            return [("synthetic0", synthetic_obs(0)), ("synthetic1", synthetic_obs(1))]
    out: list[tuple[str, dict]] = []
    for p in map(Path, paths):
        if p.is_dir() and (p / "obs_index.json").exists():
            out += [(s["sid"], gio.load_obs(p / "obs" / f"{s['sid']}.npz")) for s in gio.load_index(p)]
        elif p.is_dir():
            files = sorted(p.glob("*.npz")) or sorted((p / "vla_obs").glob("*.npz"))
            if not files:
                raise ValueError(f"{p}: no .npz observations")
            out += [(f"{p.name}/{f.stem}", load_npz_obs(f)) for f in files]
        elif p.suffix == ".npz" and p.exists():
            out.append((p.stem, load_npz_obs(p)))
        else:
            raise ValueError(f"--obs {p}: not a golden dir, a vla_obs dir or an .npz")
    return out


def inputs_differ(a: dict, b: dict) -> dict:
    """Do two W-VLA observations differ in their 43 state values and in their image (bytes)?"""
    return {"state": schema.state43(a).tobytes() != schema.state43(b).tobytes(),
            "image": np.asarray(a[schema.VIDEO_KEY]).tobytes() != np.asarray(b[schema.VIDEO_KEY]).tobytes()}


def abab_pair(obs_set: list[tuple[str, dict]]) -> tuple[tuple[str, dict], tuple[str, dict]] | None:
    """T-2.4's (A, B): A = the first observation, B = the first later one that differs from A in state AND image, so
    a stale state and a stale image both change the output; None if there is no such observation."""
    if not obs_set:
        return None
    a = obs_set[0]
    for b in obs_set[1:]:
        if all(inputs_differ(a[1], b[1]).values()):
            return a, b
    return None


# --- status stream --------------------------------------------------------------------------------------------------


class StatusTap:
    """A SUB on a device process's out-of-band status PUB (server_g1 --status-pub, PLAN.md §4.3 item 7), read by its
    own thread for the length of a bench; the socket is only touched by that thread until close()."""

    def __init__(self, url: str, ctx: zmq.Context | None = None):
        self.url = url
        self.sock = (ctx or zmq.Context.instance()).socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"status")
        self.sock.connect(url)
        self.msgs: list[tuple[int, dict]] = []
        self.t0 = time.monotonic()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="status-tap", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.sock.poll(100, zmq.POLLIN):
                parts = self.sock.recv_multipart()
                if len(parts) == 2 and parts[0] == b"status":
                    msg = wire.loads(parts[1])
                    if isinstance(msg, dict):
                        self.msgs.append((time.monotonic_ns(), msg))

    def close(self, min_s: float = 1.5) -> dict:
        """Stop after at least ``min_s`` of listening (a 2 Hz stream sends 3 messages) and summarise."""
        time.sleep(max(0.0, min_s - (time.monotonic() - self.t0)))
        self._stop.set()
        self.thread.join(2.0)
        self.sock.close(0)
        return status_summary(self.url, self.msgs)


def status_summary(url: str, msgs: list[tuple[int, dict]]) -> dict:
    out: dict = {"url": url, "n": len(msgs)}
    if not msgs:
        return out
    first, last = msgs[0][1], msgs[-1][1]
    span_s = (msgs[-1][0] - msgs[0][0]) / 1e9
    seqs = [m.get("seq") for _, m in msgs]
    calls = [m.get("calls") for _, m in msgs if isinstance(m.get("calls"), int)]
    out.update(rate_hz=(len(msgs) - 1) / span_s if span_s > 0 else None, src=last.get("src"),
               backend=last.get("backend"), seq_gaps=sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1),
               calls_delta=calls[-1] - calls[0] if calls else None, first=first, last=last)
    return out


# --- statistics -----------------------------------------------------------------------------------------------------


def _stats(x) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if not x.size:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    return {"n": int(x.size), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()), "mean": float(x.mean())}


def _num(info: dict, key: str) -> float | None:
    v = info.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def server_total_ms(info: dict) -> float | None:
    """total_ms if the server reports it, else infer_ms, else t_reply_ns - t_recv_ns."""
    for key in ("total_ms", "infer_ms"):
        if (v := _num(info, key)) is not None:
            return v
    a, b = _num(info, "t_recv_ns"), _num(info, "t_reply_ns")
    return (b - a) / 1e6 if a is not None and b is not None else None


def attribute(call: dict) -> str:
    """The largest component of one call's RTT (encode, device, decode, server_other, wire_in, wire_out, wire)."""
    parts = {k: call.get(k) for k in ("encode_ms", "device_ms", "decode_ms")}
    total = call.get("server_total_ms")
    if total is not None:
        known = sum(v for v in parts.values() if v is not None)
        parts["server_other_ms"] = max(total - known, 0.0)
        if call.get("wire_in_ms") is not None and call.get("wire_out_ms") is not None:
            parts["wire_in_ms"], parts["wire_out_ms"] = call["wire_in_ms"], call["wire_out_ms"]
        else:
            parts["wire_ms"] = max(call["rtt_ms"] - total, 0.0)
    else:
        parts["unreported_ms"] = call["rtt_ms"]
    parts = {k: v for k, v in parts.items() if v is not None}
    return max(parts, key=parts.get).removesuffix("_ms")


def d_est(rtt_ms: float, send_offset_ms: float) -> int:
    """Ticks from k_obs to the first tick start at or after the reply (the loop sends early in tick k_obs)."""
    return max(1, math.ceil((rtt_ms + send_offset_ms) / TICK_MS))


# --- benches --------------------------------------------------------------------------------------------------------


def _rt(seq: int, k_obs: int, noise_seed: int | None = None) -> dict:
    now = time.monotonic_ns()
    rt = {"seq": seq, "gen": 0, "ep": 0, "k_obs": k_obs, "t_obs_ns": now, "t_send_ns": now}
    if noise_seed is not None:
        rt["noise_seed"] = int(noise_seed)
    return rt


def bench_latency(client: G1Client, obs_set, n: int, period_ms: float, warmup: int = 20, slow_ms: float = 100.0,
                  send_offset_ms: float = 0.5, max_consecutive_errors: int = 5) -> dict:
    seq = 0
    for i in range(warmup):
        seq += 1
        client.get_action(obs_set[i % len(obs_set)][1], _rt(seq, 0))
    calls, consecutive = [], 0
    period = period_ms / 1e3
    t_next = time.perf_counter()
    for i in range(n):
        now = time.perf_counter()
        if now < t_next:
            time.sleep(t_next - now)
        t_next = max(t_next + period, time.perf_counter())  # paced, no catch-up burst
        name, obs = obs_set[i % len(obs_set)]
        seq += 1
        rec: dict = {"i": i, "obs": name}
        try:
            _, info, r = client.get_action(obs, _rt(seq, i * 16))
        except (TimeoutError, ServerError) as e:
            rec["error"] = str(e)
            calls.append(rec)
            consecutive += 1
            if consecutive >= max_consecutive_errors:
                break
            continue
        consecutive = 0
        rec.update(rtt_ms=r.rtt_ms, ser_us=r.ser_us, deser_us=r.deser_us, reply_bytes=r.nbytes,
                   encode_ms=_num(info, "encode_ms"), device_ms=_num(info, "device_ms"),
                   decode_ms=_num(info, "decode_ms"), server_total_ms=server_total_ms(info))
        srv_in, srv_out = _num(info, "t_recv_ns"), _num(info, "t_reply_ns")
        rec["wire_in_ms"] = (srv_in - r.t_send_ns) / 1e6 if srv_in is not None else None
        rec["wire_out_ms"] = (r.t_recv_ns - srv_out) / 1e6 if srv_out is not None else None
        # what analyze_run.py gates as vla_p50_ms (srv_t_reply_ns - srv_t_recv_ns), so T-2.6 can be derived from it
        rec["server_span_ms"] = (srv_out - srv_in) / 1e6 if srv_in is not None and srv_out is not None else None
        rec["d_est"] = d_est(r.rtt_ms, send_offset_ms)
        calls.append(rec)
    ok = [c for c in calls if "error" not in c]
    slow = [dict(c, cause=attribute(c)) for c in ok if c["rtt_ms"] > slow_ms]
    d = [c["d_est"] for c in ok]
    out = {"n_requested": n, "n_calls": len(calls), "n_ok": len(ok), "n_errors": len(calls) - len(ok),
           "period_ms": period_ms, "warmup": warmup, "slow_ms": slow_ms, "send_offset_ms": send_offset_ms,
           "rtt_ms": _stats(c["rtt_ms"] for c in ok),
           "server_total_ms": _stats(c["server_total_ms"] for c in ok)}
    for k in ("server_span_ms", "encode_ms", "device_ms", "decode_ms", "wire_in_ms", "wire_out_ms", "ser_us",
              "deser_us"):
        out[k] = _stats(c[k] for c in ok)
    out["d_est"] = {"p50": int(np.percentile(d, 50, method="higher")) if d else None,
                    "p99": int(np.percentile(d, 99, method="higher")) if d else None,
                    "hist": {str(v): d.count(v) for v in sorted(set(d))}}
    out["slow_calls"] = slow
    out["slow_causes"] = {c: sum(s["cause"] == c for s in slow) for c in sorted({s["cause"] for s in slow})}
    out["errors"] = [c for c in calls if "error" in c][:20]
    out["calls"] = calls
    return out


def bench_abab(client: G1Client, obs_a: tuple[str, dict], obs_b: tuple[str, dict], n: int, noise: np.ndarray) -> dict:
    refs, repeat = {}, {}
    for tag, (_, obs) in (("A", obs_a), ("B", obs_b)):
        first = outputs(client.predict_debug(obs, noise)[0])
        second = outputs(client.predict_debug(obs, noise)[0])  # the reference: the previous input was this obs too
        repeat[tag] = compare(first, second)[0]
        refs[tag] = second
    distinct, ab_diff = compare(refs["A"], refs["B"])
    mismatches = []
    for i in range(n):
        tag = "AB"[i % 2]
        obs = (obs_a if tag == "A" else obs_b)[1]
        same, diff = compare(outputs(client.predict_debug(obs, noise)[0]), refs[tag])
        if not same:
            mismatches.append({"i": i, "obs": tag, "max_abs": diff})
    return {"n": n, "obs_a": obs_a[0], "obs_b": obs_b[0], "inputs_differ": inputs_differ(obs_a[1], obs_b[1]),
            "refs_repeat_identical": repeat, "a_b_distinct": not distinct, "a_b_max_abs": ab_diff,
            "n_mismatches": len(mismatches), "mismatches": mismatches[:50]}


def bench_determinism(client: G1Client, obs: tuple[str, dict], noise: np.ndarray, replays: int = 5,
                      crn_seed: int = 12345) -> dict:
    outs = [outputs(client.predict_debug(obs[1], noise)[0]) for _ in range(replays)]
    diffs = [compare(outs[0], o) for o in outs[1:]]
    a1, _, _ = client.get_action(obs[1], _rt(1, 0))
    a2, _, _ = client.get_action(obs[1], _rt(2, 0))
    s1, _, _ = client.get_action(obs[1], _rt(3, 0, crn_seed))
    s2, _, _ = client.get_action(obs[1], _rt(4, 0, crn_seed))
    return {"obs": obs[0], "replays": replays, "replays_identical": all(same for same, _ in diffs),
            "replay_max_abs": max((d for _, d in diffs), default=0.0),
            "unseeded_differ": action_bytes(a1) != action_bytes(a2),
            "same_seed_identical": action_bytes(s1) == action_bytes(s2), "crn_seed": crn_seed}


# --- checks and CLI -------------------------------------------------------------------------------------------------


def _check(name: str, value, threshold, ok: bool) -> dict:
    return {"name": name, "value": value, "threshold": threshold, "pass": bool(ok)}


def checks_of(res: dict, thr: dict) -> list[dict]:
    out = []
    if (lat := res.get("latency")) is not None:
        rtt, srv, d = lat["rtt_ms"], lat["server_total_ms"], lat["d_est"]
        out += [_check("latency_errors", lat["n_errors"], 0, lat["n_errors"] == 0),
                _check("rtt_p50_ms", rtt["p50"], thr["rtt_p50_ms"], rtt["p50"] is not None and rtt["p50"] <= thr["rtt_p50_ms"]),
                _check("rtt_p99_ms", rtt["p99"], thr["rtt_p99_ms"], rtt["p99"] is not None and rtt["p99"] <= thr["rtt_p99_ms"]),
                _check("server_total_p50_ms", srv["p50"], thr["server_total_p50_ms"],
                       srv["p50"] is not None and srv["p50"] <= thr["server_total_p50_ms"]),
                _check("d_est_p99", d["p99"], thr["d_p99"], d["p99"] is not None and d["p99"] <= thr["d_p99"])]
    if (ab := res.get("abab")) is not None:
        out += [_check("abab_mismatches", ab["n_mismatches"], 0, ab["n_mismatches"] == 0),
                _check("abab_refs_repeat_identical", ab["refs_repeat_identical"], True,
                       all(ab["refs_repeat_identical"].values())),
                _check("abab_a_b_distinct", ab["a_b_distinct"], True, ab["a_b_distinct"])]
    if (det := res.get("determinism")) is not None:
        out += [_check("fixed_noise_replays_identical", det["replays_identical"], True, det["replays_identical"]),
                _check("unseeded_calls_differ", det["unseeded_differ"], True, det["unseeded_differ"]),
                _check("same_noise_seed_identical", det["same_seed_identical"], True, det["same_seed_identical"])]
    if (st := res.get("status_stream")) is not None:
        rate, backend = st.get("rate_hz"), (res.get("info") or {}).get("backend")
        out += [_check("status_msgs", st["n"], ">= 2", st["n"] >= 2),
                _check("status_src", st.get("src"), "server_g1", st.get("src") == "server_g1"),
                _check("status_rate_hz", rate, "1.5-12", rate is not None and 1.5 <= rate <= 12.0),
                _check("status_backend", st.get("backend"), backend, st.get("backend") == backend),
                _check("status_calls_advance", st.get("calls_delta"), "> 0",
                       isinstance(st.get("calls_delta"), int) and st["calls_delta"] > 0)]
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (bytes, bytearray)):
        return f"<{len(x)} bytes>"
    return x


def summary(res: dict) -> str:
    lines = [f"bench_g1 {res['endpoint']}  backend {res['info'].get('backend', '?')}  tree {res['info'].get('tree', '?')}"]
    if (lat := res.get("latency")) is not None:
        r, s = lat["rtt_ms"], lat["server_total_ms"]
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"latency: {lat['n_ok']}/{lat['n_requested']} ok, RTT p50 {fmt(r['p50'])} p99 {fmt(r['p99'])} "
                     f"max {fmt(r['max'])} ms; server total p50 {fmt(s['p50'])} ms; d_est {lat['d_est']['hist']}; "
                     f"{len(lat['slow_calls'])} calls > {lat['slow_ms']:g} ms {lat['slow_causes']}")
    for c in res["checks"]:
        lines.append(f"  {c['name']:<32} {str(c['value']):>12}  (want {c['threshold']})  {'PASS' if c['pass'] else 'FAIL'}")
    lines.append(f"VERDICT: {'PASS' if res['pass'] else 'FAIL'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    ap.add_argument("--obs", action="append", help="golden dir, vla_obs dir, run dir or .npz (repeatable)")
    ap.add_argument("--n", type=int, default=0, help="latency calls (T-2.3: 500)")
    ap.add_argument("--period-ms", type=float, default=320.0, help="T-2.3: 320 (s = 16 ticks)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--slow-ms", type=float, default=100.0)
    ap.add_argument("--send-offset-ms", type=float, default=0.5, help="send time after T_k in the loop (d_est)")
    ap.add_argument("--abab", type=int, default=0, metavar="N", help="T-2.4 alternating calls (200)")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--timeout-ms", type=int, default=30000, help="per call")
    ap.add_argument("--rtt-p50-max", type=float, default=T23["rtt_p50_ms"])
    ap.add_argument("--rtt-p99-max", type=float, default=T23["rtt_p99_ms"])
    ap.add_argument("--server-p50-max", type=float, default=T23["server_total_p50_ms"])
    ap.add_argument("--d-p99-max", type=int, default=T23["d_p99"])
    ap.add_argument("--keep-calls", action="store_true", help="keep every call record in the JSON")
    ap.add_argument("--status-sub", default=None, metavar="URL",
                    help="also check server_g1's --status-pub stream at URL during the run")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if not (args.n > 0 or args.abab > 0 or args.determinism):
        ap.error("nothing to do: give --n N, --abab N and/or --determinism")
    try:
        obs_set = load_obs_set(args.obs)
    except (ValueError, OSError, AssertionError, KeyError) as e:
        print(f"bench_g1: bad --obs: {e}", file=sys.stderr)
        return 2
    pair = abab_pair(obs_set) if args.abab > 0 else None
    if args.abab > 0 and pair is None:
        print(f"bench_g1: --abab needs two observations that differ in both state and image (a stale state or image "
              f"would be invisible otherwise); got {[n for n, _ in obs_set][:6]}", file=sys.stderr)
        return 2
    client = G1Client(args.endpoint, args.timeout_ms)
    if not client.ping(timeout_ms=5000):
        print(f"bench_g1: {args.endpoint} does not answer ping", file=sys.stderr)
        return 2
    thr = {"rtt_p50_ms": args.rtt_p50_max, "rtt_p99_ms": args.rtt_p99_max,
           "server_total_p50_ms": args.server_p50_max, "d_p99": args.d_p99_max}
    res = {"endpoint": args.endpoint, "argv": sys.argv if argv is None else argv, "thresholds": thr,
           "obs": [name for name, _ in obs_set], "info": client.info(), "t_start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    noise = fixed_noise(args.noise_seed)
    tap = StatusTap(args.status_sub) if args.status_sub else None
    try:
        if args.n > 0:
            res["latency"] = bench_latency(client, obs_set, args.n, args.period_ms, args.warmup, args.slow_ms,
                                           args.send_offset_ms)
            if not args.keep_calls:
                res["latency"].pop("calls")
        if pair is not None:
            res["abab"] = bench_abab(client, pair[0], pair[1], args.abab, noise)
        if args.determinism:
            res["determinism"] = bench_determinism(client, obs_set[0], noise, args.replays)
    except (TimeoutError, ServerError) as e:
        res["fatal"] = str(e)
        print(f"bench_g1: {e}", file=sys.stderr)
    finally:
        if tap is not None:
            res["status_stream"] = tap.close()
    res["info_end"] = client.info()
    res["checks"] = checks_of(res, thr)
    if "fatal" in res:
        res["checks"].append(_check("no_fatal_error", res["fatal"], None, False))
    res["pass"] = bool(res["checks"]) and all(c["pass"] for c in res["checks"])
    client.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_jsonable(res), indent=1) + "\n")
    print(summary(res))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
