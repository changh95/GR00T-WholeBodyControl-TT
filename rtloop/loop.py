"""rt_sim: the 50 Hz control loop of the G1 apple-to-plate demo (PLAN.md §1.1, §1.2, §1.6, §3.1, §4.1).

One CUDA-free process with one tick thread. Every tick k: pace to T_k = t0 + k * 20 ms (RT), poll
the VLA, trigger a request with o_k (the observation at the start of tick k), in VL make the
in-flight chunk visible at k_obs + d_inj, select the A35 command (rtloop.splicer), step the env
(the lower-body WBC runs inside, through the D5 seam in rtloop.wbc_backends) and log. The VLA runs
in its own server process and is never waited for in RT.

Modes:
  rt    wall clock, absolute deadlines, no catch-up after a slip (the demo and the timing gates)
  vl    virtual latency: no pacing; a chunk becomes visible exactly at k_obs + d_inj (sweeps,
        equivalence); deterministic with a deterministic VLA
  sync  frozen-time reference (arm C1): at k % 20 == 0 call the VLA, wait, apply rows 0..19

Operator control (rtloop.control, HUD keys v/w/r/p) is polled once per tick at the tick boundary, after the
TickLog row and the PUB, so it is never on the observation -> actuation path. ``--shadow URL`` runs a remote
W-WBC as the shadow of the ORT primary (T-3.6).

    simpy -m rtloop.loop --mode rt --arm v1 --s 16 --B 0 --vla $VLA_MOCK --wbc ort --seeds 0-4
"""

import argparse
from collections import Counter
import ctypes
import gc
import itertools
import math
import os
import signal
import sys
import time

import numpy as np

from rtloop import affinity, clock, control, sim_env, telemetry, viewer
from rtloop import splicer as sp
from rtloop.schema import (ARMS, MAX_EPISODE_TICKS, TICK_NS, VIDEO_KEY, WAIST, WBC_BALANCE,
                           ChunkStatus, Chunk, EpisodeEnd, Outcome, VlaBackend, WbcBackend,
                           build_vla_obs, hold_row, obs_sha1, row_to_env_action, state43)
from rtloop.vla_client import VlaClient

MODES = ("rt", "vl", "sync")
U64 = (1 << 64) - 1
JUMP = np.r_[ARMS, WAIST]  # boundary jump is measured on the joint targets (as the splicer does)
OVERRUN_NS = TICK_NS
META_EVERY = 50  # ticks between `meta` telemetry messages (1 Hz)
EGO_STRIDE = 4  # 480x640 -> 120x160 thumbnail on the `ego` topic
DRAIN_MS = 500

# PLAN.md §1.1 "Fidelity deviations", recorded in config.json. (f) onscreen=True and (g) the in-env viewer
# relaunch are gone: with --viewer the env is still built as stock and the viewer is its own process.
FIDELITY = {
    "a": "physics waits for the WBC call inside the tick",
    "b": "state and image come from the same tick, with no sensor latency",
    "c": "sim knee kp 300/200 where the robot uses 200/200",
    "d": "the first chunk is fetched before t0 is anchored (pre-roll, as stock)",
    "e": "VLA replies are quantised to tick polls",
}


# --- small helpers ------------------------------------------------------------------------------


def splitmix64(x: int) -> int:
    """SplitMix64 finaliser: the D17 noise seed of one request."""
    x = (x + 0x9E3779B97F4A7C15) & U64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & U64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & U64
    return x ^ (x >> 31)


def noise_seed(ep_seed: int, req_index: int) -> int:
    """splitmix64(ep_seed << 20 | i) with i the request's index within the episode (pre-roll = 0),
    so two arms share noise for matching request indices whatever earlier episodes did."""
    return splitmix64(((int(ep_seed) << 20) | int(req_index)) & U64)


def parse_seeds(spec: str):
    """'A-B' (inclusive), 'A-' (open-ended), 'A', or a comma list of those -> an iterator of ints."""
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"bad --seeds {spec!r}")
    iters, open_ended = [], False
    for j, p in enumerate(parts):
        a, sep, b = p.partition("-")
        try:
            if not sep:
                iters.append([int(a)])
            elif b == "":
                if j != len(parts) - 1:
                    raise ValueError("an open range must come last")
                iters.append(itertools.count(int(a)))
                open_ended = True
            else:
                lo, hi = int(a), int(b)
                if hi < lo:
                    raise ValueError("empty range")
                iters.append(range(lo, hi + 1))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"bad --seeds {spec!r}: {e}") from None
    return itertools.chain(*iters), open_ended


def parse_duration(s: str) -> float:
    """Seconds from '120', '120s', '2m' or '2h'."""
    s = str(s).strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600}.get(s[-1:], None)
    try:
        return float(s[:-1]) * mult if mult else float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad duration {s!r}") from None


def vla_backend_of(info: dict) -> VlaBackend:
    name = str(info.get("backend", "")).lower()
    if name.startswith("mock"):
        return VlaBackend.MOCK
    if "tt" in name.split(":")[0] or "tensix" in name:
        return VlaBackend.TT
    return VlaBackend.GPU  # the RT-owned and the stock GPU servers


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def rss_anon_mb() -> float | None:
    """Anonymous resident memory (heap, mmapped arenas): what a leak grows; file pages excluded."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


M_MMAP_THRESHOLD = -3  # glibc mallopt parameter


def set_mmap_threshold(nbytes: int) -> bool:
    """Fix glibc's mmap threshold (mallopt M_MMAP_THRESHOLD); call before the env is built.

    Why RSS grew per episode: robosuite hard-resets on every env.reset() (a new MjModel, MjData and render
    context; the old ones are freed). The live objects stay the same (2 MjSim, 2 render contexts, 1 WBC
    policy) and Python-level allocations grow < 10 KB per reset (tracemalloc), with or without the
    render-context patch. But glibc's *dynamic* mmap threshold rises to the size of the first large block
    freed (up to 32 MB), so from then on the reset's big transient blocks come from the heap and arenas,
    interleave with long-lived small ones and fragment them: RSS rose 50-90 MB per episode, about 1 GB of
    it free heap. With the threshold fixed (default 4 MiB) those blocks are mmapped and unmapped on free
    (bare env + trim, 20 resets: +4 MB at 4 MiB, +17 MB at 1 MiB, against +97 MB with the dynamic
    threshold). Every per-tick allocation is below it (the largest are the 921,600-byte frames; 0 page
    faults per steady-state tick at 1 MiB), so the tick path is unchanged."""
    if nbytes <= 0:
        return False
    try:
        return bool(ctypes.CDLL("libc.so.6").mallopt(M_MMAP_THRESHOLD, int(nbytes)))
    except (OSError, AttributeError):
        return False


def malloc_trim() -> dict | None:
    """Give the C heap's free pages back to the OS (glibc malloc_trim(0)); reset window only.

    Returns what the reset left free (about 0.9 GB per reset under glibc's default threshold, see
    set_mmap_threshold) to the OS. The pages a tick touches again fault in once, in the first tick
    after the reset (450-1150 minor faults). Rebuilding the env instead does not help (it adds about
    120 MB once). Returns {released_mb, rss_mb, ms}, or None where libc has no malloc_trim (not glibc)."""
    try:
        trim = ctypes.CDLL("libc.so.6").malloc_trim
    except (OSError, AttributeError):
        return None
    before, t0 = rss_mb(), time.perf_counter()
    trim(0)
    after = rss_mb()
    return {"released_mb": round(before - after, 1), "rss_mb": round(after, 1),
            "ms": round((time.perf_counter() - t0) * 1e3, 1)}


def _pct(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else None


# --- the loop -----------------------------------------------------------------------------------


class Loop:
    """Owns the env, the clients, the splicer state and the logs of one rt_sim invocation."""

    viewer_link: viewer.ViewerLink | None = None  # --viewer: shared memory + the viewer process
    # Operator control off by default, so the tick code also runs on a Loop built without __init__ (the review
    # harness, tests/review_harness.py); __init__ sets them from the CLI.
    control = None                # rtloop.control.ControlPull with --control
    swap_every_s: float | None = None  # --swap-wbc-every
    remote_shadow = None          # --shadow URL: a fallback-free ZmqWbc
    n_ticks_total = 0
    reset_requested = _resumed = False

    def __init__(self, args: argparse.Namespace, argv: list[str]):
        self.args = args
        self.mode = args.mode
        self.rt, self.vl, self.sync = args.mode == "rt", args.mode == "vl", args.mode == "sync"
        pin = not args.no_affinity and (self.rt or args.affinity)
        self.tick_cpu = affinity.TICK_CPU if pin else None
        self.helper_cpu = affinity.HELPER_CPU if pin else None
        self.stop_requested = False
        self.t_run0 = time.monotonic()
        self.mmap_threshold_set = set_mmap_threshold(args.mmap_threshold_kb * 1024)  # before the env exists

        # Build everything with the main thread on the helper CPU, so library threads inherit it.
        affinity.pin_current(self.helper_cpu)
        sim_env.assert_cuda_free("start")
        self.ctx = affinity.zmq_context(self.helper_cpu)

        from rtloop import wbc_backends as wb

        self.wb = wb
        self.ort = wb.OrtCpu(args.onnx_dir or wb.ONNX_DIR, args.ort_opts)
        # T-3.6: a remote W-WBC shadows the ORT primary. No ORT fallback (a miss is NaN, never a 0 that reads as
        # parity), latched like a primary so a dead shadow costs at most three timeouts per episode.
        self.remote_shadow = None
        if args.shadow:
            self.remote_shadow = wb.ZmqWbc(args.shadow, fallback=None, reference=self.ort,
                                           timeout_ms=args.shadow_timeout_ms or args.wbc_timeout_ms,
                                           latch=not args.wbc_no_latch, ctx=self.ctx)
        if args.wbc == "ort":
            primary, shadow = self.ort, self.remote_shadow
        else:
            primary = wb.ZmqWbc(args.wbc, fallback=self.ort, timeout_ms=args.wbc_timeout_ms,
                                latch=not args.wbc_no_latch, ctx=self.ctx)
            shadow = None if args.no_shadow else self.ort  # D16: ORT shadow on every call
        self.runtime = wb.WbcRuntime(primary, shadow, dump=[] if args.dump_obs516 else None)
        wb.install(self.runtime)  # before the env builds its G1GearWbcPolicy (D5)
        # WBC swap targets (HUD key w, PLAN.md §1.4.5, WP-3.9): connected and parity-checked here, never
        # on the tick path. "ort" is always one; the primary is named by its backend (gpu, x280, ...).
        self.wbc_name = "ort" if args.wbc == "ort" else primary.backend.name.lower()
        self.wbc_targets = {self.wbc_name: primary, "ort": self.ort}
        for name, url in control.parse_targets(args.wbc_alt).items():
            if name in self.wbc_targets:
                raise ValueError(f"--wbc-alt {name}: name already used by {list(self.wbc_targets)}")
            self.wbc_targets[name] = wb.ZmqWbc(url, fallback=self.ort, timeout_ms=args.wbc_timeout_ms,
                                               latch=not args.wbc_no_latch, ctx=self.ctx)

        if args.viewer:
            # rtloop.viewer_proc, fed through shared memory, so the env below stays as stock (onscreen=False).
            # The link (its names go into config.json) exists from here; the process starts with the run dir.
            cpus = [] if args.no_affinity else affinity.parse_cpus(args.viewer_cpus)
            self.viewer_link = viewer.ViewerLink(cpus)
        self.render_patch = sim_env.patch_render_context_teardown()  # before the first MjSim exists
        self.env = sim_env.make_env(onscreen=False)
        self.n_env_checkers = sim_env.skip_first_step_check(self.env)
        self.h = sim_env.SimHandles(self.env)
        sim_env.assert_cuda_free("after imports")

        give_up = args.give_up_ms if self.rt else args.block_timeout_ms
        self.vla = VlaClient(args.vla, give_up_ms=give_up, ctx=self.ctx)
        if not self.vla.ping(5000):
            raise RuntimeError(f"VLA server {args.vla} does not answer ping")
        self.vla_info = self.vla.info()
        self.modality_checked = self.vla.check_modality_config()
        self.vla_be = vla_backend_of(self.vla_info)
        self.pub = telemetry.Publisher(args.pub, ctx=self.ctx) if args.pub else telemetry.Publisher(None)
        self.pub_on = bool(args.pub)
        # VLA swap targets (HUD key v, WP-2.8): pinged, info'd and modality-checked here; a swap is
        # applied in the next reset window, where gen is bumped anyway (PLAN.md §1.4.5).
        self.vla_name = self.vla_be.name.lower()
        self.vla_clients = {self.vla_name: (self.vla, self.vla_info)}
        for name, url in control.parse_targets(args.vla_alt).items():
            if name in self.vla_clients:
                raise ValueError(f"--vla-alt {name}: name already used by {list(self.vla_clients)}")
            alt = VlaClient(url, give_up_ms=give_up, ctx=self.ctx)
            if not alt.ping(5000):
                raise RuntimeError(f"--vla-alt {name}={url} does not answer ping")
            alt_info = alt.info()
            alt.check_modality_config()
            self.vla_clients[name] = (alt, alt_info)
        # Operator control (HUD keys v/w/r/p): PULL bound now, polled once per tick at the boundary.
        self.control = control.ControlPull(args.control, ctx=self.ctx) if args.control else None
        self.cstate = control.ControlState()
        self.control_log: list[dict] = []
        self.pauses: list[list] = []  # [ep, k, ms]: every pause, in full (the control log keeps the last 500)
        self.reset_requested = self._resumed = False
        self.swap_every_s = args.swap_wbc_every
        self.n_ticks_total = self.n_pauses = self.n_operator_resets = self.n_vla_swaps = self.n_wbc_swaps = 0
        self.paused_ms = 0.0
        self.ep_wbc_swaps, self.ep_paused_ms = 0, 0.0

        self.cfg = sp.SplicerConfig(arm=args.arm, s=args.s, B=args.B,
                                    d_init=args.d_inj if self.vl else args.d_init,
                                    sync_rows=args.sync_rows)
        self.st = sp.new_state()
        self.pacer = clock.Pacer(mode=args.pacing, spin_ns=int(args.spin_ms * 1e6))
        self.gen = 0
        self.next_seq = 1  # 0 means "no chunk" in the TickLog
        self.n_vla_dumped = 0
        self.vla_dump_pending: list = []  # (seq, vla_obs), written in the reset window
        self.inflight_k_obs = -1
        self.ep = self.seed = self.ep_req_i = 0
        self.ep_outcomes: Counter = Counter()
        self.usage: dict[int, list] = {}  # seq -> [first_row, last_row, n_rows]
        self.jumps: dict[int, float] = {}  # seq -> boundary jump at its switch tick
        self.n_episodes = 0
        self.interrupted = False
        self._slack_set = False

        tag = args.tag or (self.mode if self.sync else f"{self.mode}_{args.arm}")
        extra = {
            "args": vars(args), "mode": self.mode, "endpoints": {"vla": args.vla, "wbc": args.wbc,
                                                                 "pub": self.pub.url, "shadow": args.shadow},
            "vla_info": self.vla_info, "vla_be": self.vla_be.name,
            "modality_config_checked": self.modality_checked,
            "wbc_info": self._wbc_info(), "ort_opts": args.ort_opts,
            "splicer": vars(self.cfg) if not self.sync else None,
            "fidelity_deviations": dict(FIDELITY),
            "viewer": self.viewer_link.plan() if self.viewer_link is not None else None,
            "affinity_plan": {"tick_cpu": self.tick_cpu, "helper_cpu": self.helper_cpu,
                              "idle_sibling": affinity.TICK_SIBLING if pin else None,
                              "viewer_cpus": (self.viewer_link.cpus or None) if self.viewer_link else None},
            "pacing": {"mode": args.pacing if self.rt else None, "spin_ms": args.spin_ms},
            "malloc_mmap_threshold_kb": args.mmap_threshold_kb if self.mmap_threshold_set else None,
            "env_checkers_skipped": self.n_env_checkers,
            "render_context_teardown_patch": self.render_patch,
            "malloc_trim_per_reset": not args.no_malloc_trim,
            "control": {"url": self.control.url if self.control else None,
                        "vla_targets": {n: c.url for n, (c, _) in self.vla_clients.items()},
                        "vla_alt_info": {n: i for n, (c, i) in self.vla_clients.items() if c is not self.vla},
                        "wbc_targets": {n: getattr(b, "url", "in-process") for n, b in self.wbc_targets.items()},
                        "swap_wbc_every_s": args.swap_wbc_every},
            "rss_mb_start": rss_mb(),
        }
        self.run = telemetry.RunDir(args.runs_root, tag, telemetry.collect_config(argv, extra))
        self.tlog = telemetry.TickLog(self.run, capacity=max(4096, 16 * args.max_ticks))
        # An episode sends at most one request per tick (+ pre-roll retries): keep that much room.
        self.rlog = telemetry.RequestLog(self.run, reserve=max(2048, args.max_ticks + 64))
        self.elog = telemetry.EpisodeLog(self.run)
        print(f"rt_sim: run dir {self.run.path}", flush=True)
        if self.viewer_link is not None:
            self.run.update_config(viewer=self.viewer_link.spawn(self.run.file("viewer.log")))

    # --- setup info ---------------------------------------------------------------------------

    def _wbc_info(self) -> dict:
        p, sh = self.runtime.primary, self.runtime.shadow
        shadow = (None if sh is None else "ort" if isinstance(sh, self.wb.OrtCpu)
                  else {"url": sh.url, **{k: sh.info().get(k) for k in ("backend", "parity_tol_used", "timeout_ms")}})
        if isinstance(p, self.wb.ZmqWbc):
            return {**p.info(), "shadow": shadow}
        return {"backend": "ort-cpu", "opts": p.opts, "onnx_dir": p.onnx_dir, "sha256": p.sha256,
                "shadow": shadow}

    # --- requests -----------------------------------------------------------------------------

    def _send(self, k: int, obs: dict, t_obs_ns: int):
        """Step 3: send o_k (the 9 keys, D6/D7) with options.rt; returns (seq, vla_obs, k)."""
        st, cfg = self.st, self.cfg
        vla_obs = build_vla_obs(obs)
        seq = self.next_seq
        self.next_seq += 1
        req_i = self.ep_req_i
        self.ep_req_i += 1
        rt = {"seq": seq, "gen": self.gen, "ep": self.ep, "k_obs": k, "t_obs_ns": int(t_obs_ns)}
        ns = noise_seed(self.seed, req_i) if self.args.crn else 0
        if self.args.crn:
            rt["noise_seed"] = ns
        dh, dmax, se = sp.d_hat(st, cfg), sp.d_hat_max(st, cfg), sp.s_eff(st, cfg)
        self.vla.send(vla_obs, rt)
        if not self.sync:
            sp.on_send(k, st)
        self.inflight_k_obs = k
        self.rlog.add({"seq": seq, "gen": self.gen, "ep": self.ep, "k_obs": k, "k_arr": -1, "d": -1,
                       "d_hat": dh, "d_hat_max": dmax, "s_eff": se, "t_obs_ns": int(t_obs_ns),
                       "t_send_ns": self.vla.inflight_rt["t_send_ns"], "noise_seed": ns,
                       "state43_sent": state43(vla_obs), "first_row": -1, "last_row": -1,
                       "outcome": Outcome.PENDING, "seed": self.seed, "req_i": req_i,
                       "bytes": self.vla.last_request_bytes})
        return seq, vla_obs, k

    def _audit(self, sent) -> None:
        """After env.step (off the observation -> actuation path): sha1, opt-in dump, ego thumbnail.

        A --dump-vla-obs file is about 0.9 MB of disk I/O (0.8-1.2 ms), so the observation (a private copy
        from build_vla_obs) is only queued here and written in the reset window (_flush_vla_dumps)."""
        seq, vla_obs, k_obs = sent
        self.rlog.update(seq, obs_sha1=obs_sha1(vla_obs))
        if self.n_vla_dumped < self.args.dump_vla_obs:
            self.vla_dump_pending.append((seq, vla_obs))
            self.n_vla_dumped += 1
        if self.pub_on:
            self.pub.ego(seq, k_obs, vla_obs[VIDEO_KEY][0, 0, ::EGO_STRIDE, ::EGO_STRIDE])

    def _on_reply(self, k: int, rep, preroll: bool = False):
        """Register a finished request at tick k (RT: the poll tick; VL: k_obs + d_inj)."""
        rt = rep.rt or {}
        seq, k_obs = rep.seq, int(rt.get("k_obs", -1))
        info = rep.info if isinstance(rep.info, dict) else {}
        fields = {"t_recv_ns": rep.t_recv_ns, "ser_us": rep.ser_us, "deser_us": rep.deser_us}
        infer = info.get("infer_ms")
        if isinstance(infer, (int, float)):
            fields["srv_infer_ms"] = float(infer)
        for key in ("t_recv_ns", "t_reply_ns"):
            if isinstance(info.get(key), int):
                fields[f"srv_{key}"] = info[key]
        chunk = None
        if rep.status == ChunkStatus.OK:
            chunk = Chunk(seq=seq, gen=rep.gen, k_obs=k_obs, rows=rep.rows,
                          t_obs_ns=int(rt.get("t_obs_ns", 0)), t_send_ns=rep.t_send_ns,
                          t_recv_ns=rep.t_recv_ns, vla_be=int(self.vla_be), status=int(rep.status),
                          info=info)
            if self.sync:
                chunk.k_arr = chunk.k_base = k
                outcome = Outcome.APPLIED
            else:
                outcome = sp.on_arrival(k, chunk, self.st, self.cfg, measure=not preroll)
            fields.update(k_arr=k, d=k - k_obs)
        else:
            outcome = Outcome.TIMEOUT if rep.status == ChunkStatus.TIMEOUT else Outcome.ERROR
            if not self.sync:
                sp.on_failure(k, self.st, outcome)
            fields.update(error=rep.error, status=rep.status.name)
        fields["outcome"] = outcome
        self.rlog.update(seq, **fields)
        self.ep_outcomes[outcome] += 1
        if self.pub_on:
            self.pub.req({"seq": seq, "gen": rep.gen, "ep": self.ep, "k_obs": k_obs, "k_arr": k,
                          "d": k - k_obs, "d_hat": sp.d_hat(self.st, self.cfg),
                          "s_eff": sp.s_eff(self.st, self.cfg), "rtt_ms": rep.rtt_ms,
                          "ser_ms": rep.ser_us / 1e3,
                          "srv": {key: info.get(key) for key in
                                  ("infer_ms", "encode_ms", "device_ms", "decode_ms")},
                          "outcome": outcome.name})
        return outcome, chunk

    def _blocking_call(self, k: int, obs: dict, preroll: bool) -> Chunk:
        """Send o_k and wait for it with the sim frozen (pre-roll, sync); three attempts."""
        old = self.vla.give_up_ms
        self.vla.give_up_ms = max(old, self.args.block_timeout_ms)
        try:
            for _ in range(3):
                sent = self._send(k, obs, clock.now_ns())  # frozen time: T_k is "now"
                rep = self.vla.wait(self.args.block_timeout_ms)
                if rep is None:  # cannot happen (give-up <= timeout), but never leave REQ wedged
                    self.vla.drain(0)
                    rep = self.vla.drained
                outcome, chunk = self._on_reply(k, rep, preroll=preroll)
                self._audit(sent)
                if outcome == Outcome.APPLIED:
                    return chunk
                print(f"rt_sim: blocking VLA call at k={k} failed: {outcome.name} {rep.error}",
                      file=sys.stderr, flush=True)
            raise RuntimeError(f"VLA call at k={k} failed 3 times ({self.args.vla})")
        finally:
            self.vla.give_up_ms = old

    def _flush_vla_dumps(self) -> None:
        for seq, vla_obs in self.vla_dump_pending:
            telemetry.save_vla_obs(self.run, seq, vla_obs)
        self.vla_dump_pending = []

    def _drain(self, k_end: int, stored) -> str:
        """Lifecycle step 1: nothing may stay in flight across the reset (REQ cannot send again)."""
        if stored is not None:  # VL: arrived, never became visible
            self.rlog.update(stored.seq, outcome=Outcome.DRAINED, t_recv_ns=stored.t_recv_ns,
                             drain="stored")
            sp.on_failure(k_end, self.st, Outcome.DRAINED)
            self.ep_outcomes[Outcome.DRAINED] += 1
            return "stored"
        if not self.vla.in_flight:
            return "none"
        seq = int(self.vla.inflight_rt["seq"])
        res = self.vla.drain(DRAIN_MS)
        fields = {"outcome": Outcome.DRAINED, "drain": res}
        d = self.vla.drained
        if d is not None and res == "ok":
            fields.update(t_recv_ns=d.t_recv_ns, deser_us=d.deser_us)
        self.rlog.update(seq, **fields)
        if not self.sync:
            sp.on_failure(k_end, self.st, Outcome.DRAINED)
        self.ep_outcomes[Outcome.DRAINED] += 1
        return res

    def _warmup(self, obs: dict) -> None:
        """PLAN.md §4.4 "20 warm-up calls before each run" (blocking; not logged as requests)."""
        vo = build_vla_obs(obs)
        for i in range(self.args.warmup):
            rt = {"seq": 0, "gen": 0, "ep": 0, "k_obs": 0, "t_obs_ns": clock.now_ns(), "warmup": i}
            self.vla.call("get_action", {"observation": vo, "options": {"rt": rt}},
                          timeout_ms=self.args.block_timeout_ms)

    # --- one tick: step and log ---------------------------------------------------------------

    def _step_and_log(self, k, cmd, info, obs, t_start, late_ns, slip, in_flight, sent, prev_start):
        """Steps 5-7: env.step under the SystemExit guard, the TickLog row, the end conditions.

        Returns (obs_next, end | None, success)."""
        runtime = self.runtime
        runtime.tick = k
        t_sim = clock.now_ns()
        safety = False
        try:
            obs_next, _, _, _, step_info = self.env.step(row_to_env_action(cmd))
        except SystemExit:  # JointSafetyMonitor.trigger_system_shutdown (sync_env.py:516-528)
            safety, obs_next, step_info = True, obs, {}
        t_sim_end = clock.now_ns()
        success = bool(step_info.get("success", False))
        z, roll, pitch = sim_env.pelvis_state(obs_next)

        row = self.tlog.next()
        row["ep"] = self.ep
        row["k"] = k
        row["t_start_ns"] = t_start
        # NaN marks the first tick after an operator pause (prev_start None): it has no period, and
        # analyze_run leaves that interval out of the timing gates.
        row["period_us"] = math.nan if prev_start is None else ((t_start - prev_start) / 1e3 if k else 0.0)
        row["late_us"] = late_ns / 1e3
        row["slip"] = slip
        row["sim_us"] = (t_sim_end - t_sim) / 1e3
        row["viewer_us"] = self.viewer_link.publish(k) if self.viewer_link is not None else 0.0
        call = runtime.last
        if call is not None and call.tick == k:
            row["wbc_rtt_us"] = call.rtt_us
            row["wbc_dev_us"] = call.dev_us
            row["wbc_be"] = int(call.backend)
            row["wbc_fb"] = call.fallback
            row["policy"] = 0 if call.policy == WBC_BALANCE else 1
            row["shadow_da"] = call.shadow_da
        else:
            row["wbc_rtt_us"] = row["wbc_dev_us"] = row["shadow_da"] = math.nan
        row["chunk_seq"] = info["chunk_seq"]
        row["row"] = info["row"]
        row["age"] = info["age"]
        row["rows_left"] = info["rows_left"]
        row["starved"] = info["starved"]
        row["in_flight"] = in_flight
        row["splice"] = info["splice"]
        row["jump_rad"] = info["jump_rad"]
        row["cmd35"] = cmd
        lower = self.h.lower
        row["nav_wbc"] = lower.cmd
        q_cmd = self.h.wbc.last_action["q"]
        row["q43_cmd"] = q_cmd
        row["q15"] = q_cmd[sim_env.LOWER_BODY]
        row["q43_meas"] = obs_next["q"]
        row["dq_arm_absmax"] = float(np.max(np.abs(obs_next["dq"][sim_env.ARM_Q43])))
        row["pelvis_z"] = z
        row["pelvis_rp"] = (roll, pitch)
        row["success"] = success
        row["safety"] = safety

        r = info["row"]
        if r >= 0 and not info["starved"] and not info.get("paused", 0):
            u = self.usage.get(info["chunk_seq"])
            if u is None:
                self.usage[info["chunk_seq"]] = [r, r, 1]
            else:
                u[1] = r
                u[2] += 1
        if info["splice"]:
            self.jumps[info["chunk_seq"]] = info["jump_rad"]
        if sent is not None:
            self._audit(sent)

        t_end = clock.now_ns()
        row["t_end_ns"] = t_end
        compute = t_end - t_start
        row["compute_us"] = compute / 1e3
        row["slack_us"] = (self.pacer.deadline(k + 1) - t_end) / 1e3 if self.rt else 0.0
        row["overrun"] = compute > OVERRUN_NS
        self.tlog.commit()
        if self.pub_on:
            self.pub.tick(telemetry.tick_msg(row))
            if k % META_EVERY == 0:
                self._meta(k)

        if safety:
            return obs_next, EpisodeEnd.SAFETY_FAULT, success
        if success:
            return obs_next, EpisodeEnd.SUCCESS, success
        if sim_env.fell(z):
            return obs_next, EpisodeEnd.FALL, success
        if k + 1 >= self.args.max_ticks:
            return obs_next, EpisodeEnd.TRUNC, success
        return obs_next, None, success

    def _meta(self, k: int) -> None:
        wall = (clock.now_ns() - self.pacer.t0) / 1e9 if self.rt else None
        self.pub.meta({"processor": {"sim": "host-cpu", "vla": self.vla_info.get("backend", "policy-server"),
                                     "wbc": self._wbc_info().get("backend")},
                       "ep": self.ep, "k": k, "mode": self.mode,
                       "rt_factor": (k * TICK_NS / 1e9) / wall if wall else None,
                       "seed": self.seed,
                       "endpoints": {"vla": self.vla.url, "wbc": getattr(self.runtime.primary, "url", "ort")},
                       "current": {"vla": self.vla_name, "wbc": self.wbc_name},
                       "targets": {"vla": list(self.vla_clients), "wbc": list(self.wbc_targets)},
                       "pending_vla": self.cstate.pending_vla, "paused": self.cstate.paused,
                       "control": self.control.url if self.control else None})

    # --- operator control (PLAN.md §1.4.5; rtloop.control) ------------------------------------

    def _control_boundary(self, k: int) -> bool:
        """Between tick k and k+1 (after the TickLog commit and the PUB, so off the observation ->
        actuation path): poll the control PULL once and apply what arrived. True: end the episode (reset)."""
        self.n_ticks_total += 1
        if self.control is None and not self.swap_every_s:
            return False
        cmds = self.control.poll() if self.control is not None else []
        if control.auto_swap_due(self.n_ticks_total, self.swap_every_s):  # T-3.9
            cmds.append(control.Command("swap_wbc", control.NEXT))
        if not cmds:
            return False
        end = False
        for c in cmds:
            end |= self._apply_control(k, c)
        if self.cstate.paused and not end:
            end = self._pause(k)
        if self.pub_on:
            self._meta(k)  # the HUD's processor map follows a swap at once, not at the next 1 Hz meta
        return end

    def _apply_control(self, k: int, c) -> bool:
        if isinstance(c, control.ControlError):
            self.cstate = control.reject(self.cstate, c)
            print(f"rt_sim: control: rejected: {c}", file=sys.stderr, flush=True)
            return False
        self.cstate, act = control.apply(self.cstate, c, list(self.vla_clients), self.vla_name,
                                         list(self.wbc_targets), self.wbc_name)
        self.control_log.append({"ep": self.ep, "k": k, "cmd": c.cmd, "to": c.to, "id": c.id,
                                 "action": act.kind, "arg": act.to})
        if act.kind == "swap_wbc":
            self._swap_wbc(act.to)
        elif act.kind == "reset":
            self.reset_requested = True
            return True
        return False

    def _swap_wbc(self, name: str) -> None:
        """Stateless (D2): tick k+1's run_inference goes to the new backend. ORT shadows a remote one (D16);
        the ORT primary gets the --shadow W-WBC back, if there is one (T-3.6)."""
        be = self.wbc_targets[name]
        if isinstance(be, self.wb.ZmqWbc):
            be.latched, be.consecutive = False, 0  # its REQ is idle (calls are synchronous): no I/O here
            self.runtime.shadow = None if self.args.no_shadow else self.ort
        else:
            self.runtime.shadow = self.remote_shadow
        self.runtime.primary = be
        self.wbc_name = name
        self.n_wbc_swaps += 1
        self.ep_wbc_swaps += 1

    def _swap_vla(self, name: str) -> None:
        """Reset window only: the episode-end drain left nothing in flight, and the gen bump that follows
        discards anything the old server could still send."""
        assert not self.vla.in_flight, "VLA swap with a request in flight"
        self.vla, self.vla_info = self.vla_clients[name]
        self.vla_be = vla_backend_of(self.vla_info)
        self.vla_name = name
        self.st.d_hist.clear()  # d_hat describes the server, not the episode (splicer.reset)
        self.n_vla_swaps += 1
        self.cstate = control.vla_swapped(self.cstate)
        print(f"rt_sim: VLA swapped to {name} ({self.vla.url}, {self.vla_info.get('backend')})", flush=True)

    def _pause(self, k: int) -> bool:
        """Operator pause between tick k and k+1: the sim is frozen (the viewer process keeps showing the
        last published state), commands are still served, meta is re-published every second. On resume the
        pacer re-anchors so T_{k+1} = now + 20 ms (no slip, no catch-up burst). True: a reset ended it."""
        t0 = t_meta = clock.now_ns()
        self.n_pauses += 1
        end = False
        while self.cstate.paused and not self.stop_requested and not end:
            time.sleep(0.02)
            for c in self.control.poll():
                end |= self._apply_control(k, c)
            now = clock.now_ns()
            if self.pub_on and now - t_meta >= 1_000_000_000:
                self._meta(k)
                t_meta = now
        dt_ms = (clock.now_ns() - t0) / 1e6
        self.paused_ms += dt_ms
        self.ep_paused_ms += dt_ms
        self.pauses.append([self.ep, k, round(dt_ms, 1)])
        if self.rt:
            self.pacer.anchor(clock.now_ns() - k * TICK_NS)
        self._resumed = True
        return end

    # --- tick loops ---------------------------------------------------------------------------

    def _vl_visible(self, k: int, stored):
        """VL step 3b: if the in-flight request is due (k >= k_obs + d_inj), block for its reply and
        register it at k. Returns what is still stored (arrived, not yet visible)."""
        if self.st.in_flight and sp.visible(k, self.inflight_k_obs, self.args.d_inj):
            rep = stored if stored is not None else self.vla.wait(self.args.block_timeout_ms)
            stored = None
            if rep is None:
                self.vla.drain(0)
                rep = self.vla.drained
            self._on_reply(k, rep)
        return stored

    def _ticks_async(self, obs: dict):
        """RT and VL: PLAN.md §1.2 steps 1-7 with the splicer. Returns (end, k_end, first_success, stored).

        VL runs step 3b both before and after the trigger (PLAN.md §1.2 lists it after only). Before:
        a chunk that becomes visible at k registers ahead of trigger(k), exactly where RT registers a
        chunk that arrives at k, so when the arrival tick is also a trigger tick (d >= s_eff, or a
        late-reject retrigger) both send o_k at k and VL at d_inj = d stays the RT policy at latency
        d. After: d_inj = 0 blocks on the request sent in step 3 of the same tick (VL == sync)."""
        args, st, cfg, vla = self.args, self.st, self.cfg, self.vla
        rt, vl = self.rt, self.vl
        stored = None  # VL: a reply that has arrived but is not yet visible
        prev_start = 0
        if rt:
            self.pacer.anchor()
        for k in range(args.max_ticks):
            # 1 pace
            if rt:
                t_start, late, slip = self.pacer.wait(k)
                t_k = self.pacer.deadline(k)
            else:
                t_start, late, slip = clock.now_ns(), 0, False
                t_k = t_start
            # 2 poll
            if vla.in_flight:
                rep = vla.poll()
                if rep is not None:
                    if vl and rep.status == ChunkStatus.OK:
                        stored = rep  # registered at its visibility tick (3b)
                    else:
                        self._on_reply(k, rep)  # RT arrival, or a failure (nothing will be visible)
            # 3b (before the trigger) VL: a chunk sent earlier and due now registers here, where RT
            # registers its arrival (step 2), so a trigger at k_arr fires on the same tick in both
            if vl:
                stored = self._vl_visible(k, stored)
            # 3 trigger
            sent = self._send(k, obs, t_k) if sp.trigger(k, st, cfg) else None
            # 3b (after the trigger) VL with d_inj = 0: the request just sent is due this tick
            if vl:
                stored = self._vl_visible(k, stored)
            # 4 select
            in_flight = st.in_flight
            hold = hold_row(obs) if st.act is None and st.pending is None else None
            cmd, info = sp.select(k, st, cfg, hold)
            # 5-7
            obs, end, success = self._step_and_log(k, cmd, info, obs, t_start, late, slip,
                                                   in_flight, sent, prev_start)
            prev_start = t_start
            if end is not None:
                return end, k, (k if success else None), stored
            if self._control_boundary(k):  # operator reset (HUD r): end here, no outcome recorded
                return None, k, None, stored
            if self._resumed:  # tick k+1 after a pause has no meaningful period
                self._resumed, prev_start = False, None
            if self.stop_requested:
                return None, k, None, stored
        raise AssertionError("unreachable: TRUNC ends the last tick")

    def _ticks_sync(self, obs: dict, chunk: Chunk):
        """C1 sync_frozen (PLAN.md §1.6): no splicer, no pacing; rows 0..sync_rows-1 per call."""
        args = self.args
        n_rows = args.sync_rows
        rows, seq, k_base = chunk.rows, chunk.seq, 0
        last_cmd = None
        prev_start = 0
        for k in range(args.max_ticks):
            t_start = clock.now_ns()
            sent = None
            if k - k_base >= n_rows:
                chunk = self._blocking_call(k, obs, preroll=False)
                rows, seq, k_base = chunk.rows, chunk.seq, k
            i = k - k_base
            cmd = rows[i].copy()
            splice = int(i == 0)
            jump = float(np.linalg.norm(cmd[JUMP] - last_cmd[JUMP])) if splice and last_cmd is not None else 0.0
            info = {"row": i, "age": i, "rows_left": n_rows - 1 - i, "starved": 0, "paused": 0,
                    "splice": splice, "jump_rad": jump, "chunk_seq": seq}
            obs, end, success = self._step_and_log(k, cmd, info, obs, t_start, 0, False, False,
                                                   sent, prev_start)
            prev_start = t_start
            last_cmd = cmd
            if end is not None:
                return end, k, (k if success else None), None
            if self._control_boundary(k):
                return None, k, None, None
            if self._resumed:
                self._resumed, prev_start = False, None
            if self.stop_requested:
                return None, k, None, None
        raise AssertionError("unreachable: TRUNC ends the last tick")

    # --- episodes -----------------------------------------------------------------------------

    def episode(self, ep: int, seed: int) -> dict | None:
        args = self.args
        t_ep0 = time.monotonic()
        # Reset window, outside the clock, on the helper CPU (robosuite re-creates its renderer).
        affinity.pin_current(self.helper_cpu)
        gc.enable()
        gc.collect()
        obs, _ = self.env.reset(seed=seed)
        sim_env.assert_cuda_free(f"after reset (seed {seed})")
        wbc_ping = None
        if isinstance(self.runtime.primary, self.wb.ZmqWbc):
            wbc_ping = self.runtime.primary.reset_episode()
        if self.remote_shadow is not None:  # a latched shadow gets the new episode too
            self.remote_shadow.reset_episode()
        if self.cstate.pending_vla is not None:  # HUD v: applied here, where gen is bumped anyway
            self._swap_vla(self.cstate.pending_vla)
        self.ep_wbc_swaps, self.ep_paused_ms = 0, 0.0
        self.gen += 1
        sp.reset(self.st, self.gen)
        self.ep, self.seed, self.ep_req_i = ep, seed, 0
        self.ep_outcomes = Counter()
        self.usage, self.jumps = {}, {}
        viewer_handoff = None
        if self.viewer_link is not None:  # the reset rebuilt the MjModel: the viewer gets it as a new generation
            sim = self.h.robosuite.sim
            viewer_handoff = self.viewer_link.handoff(sim.model._model, sim.data._data)
        self.tlog.prefault(args.max_ticks)
        if ep == 0 and args.warmup:
            self._warmup(obs)
        row0 = len(self.tlog)
        n_calls0 = sum(self.runtime.n_calls.values())
        gc.collect()
        trim = None if args.no_malloc_trim else malloc_trim()
        gc.disable()
        t_reset_s = time.monotonic() - t_ep0

        # Pre-roll (blocks, sim frozen) still on the helper CPU: the first zero-copy send starts
        # pyzmq's garbage-collector thread (with its own zmq context), which must not inherit the
        # tick CPU. Then the tick thread moves to its CPU, and the tick loop anchors t0.
        t_pre = clock.now_ns()
        chunk0 = self._blocking_call(0, obs, preroll=True)
        preroll_ms = (clock.now_ns() - t_pre) / 1e6
        affinity.pin_current(self.tick_cpu)
        if not self._slack_set:
            self._slack_set = clock.set_timer_slack(1)
        if ep == 0:
            self.run.update_config(threads=affinity.thread_report(), main_thread_cpus=affinity.current(),
                                   timer_slack_1ns=self._slack_set)
        if self.sync:
            end, k_end, fst, stored = self._ticks_sync(obs, chunk0)
        else:
            end, k_end, fst, stored = self._ticks_async(obs)

        gc.enable()
        affinity.pin_current(self.helper_cpu)
        drain = self._drain(k_end, stored)
        self._flush_vla_dumps()
        for seq, (first, last, n) in self.usage.items():
            self.rlog.update(seq, first_row=first, last_row=last, n_rows_used=n)
        for seq, jump in self.jumps.items():
            self.rlog.update(seq, switch_jump=jump)
        if end is None:  # interrupted or operator reset: not an outcome, so no EpisodeLog record
            self.rlog.end_episode()
            self.tlog.flush(sync=True)
            self.tlog.release()
            if self.reset_requested and not self.stop_requested:
                self.reset_requested = False
                self.n_operator_resets += 1
                print(f"rt_sim: ep {ep} seed {seed}: operator reset at k={k_end}", flush=True)
                return {"ep": ep, "seed": seed, "end": "OPERATOR_RESET", "k_end": k_end}
            return None
        t = self.tlog.rows()[row0:]
        rec = self._episode_record(ep, seed, end, k_end, fst, drain, t, n_calls0, wbc_ping,
                                   t_reset_s, preroll_ms, viewer_handoff, time.monotonic() - t_ep0)
        rec["malloc_trim"] = trim
        self.rlog.end_episode()
        self.tlog.flush(sync=True)
        rec["ticklog_released_mb"] = round(self.tlog.release() / 2**20, 2)
        rec["rss_mb"], rec["rss_anon_mb"] = rss_mb(), rss_anon_mb()  # after the logs let go of their pages
        self.elog.add(rec)
        self.n_episodes += 1
        print(f"rt_sim: ep {ep} seed {seed}: {end.name} at k={k_end} "
              f"({rec['n_requests']} requests, starved {rec['n_starved']}, "
              f"tick p99 {rec['tick_p99_ms']} ms)", flush=True)
        return rec

    def _episode_record(self, ep, seed, end, k_end, fst, drain, t, n_calls0, wbc_ping,
                        t_reset_s, preroll_ms, viewer_handoff, wall_s) -> dict:
        args, oc = self.args, self.ep_outcomes
        splice = t["splice"].astype(bool)
        with_row = t["row"] >= 0
        per = np.diff(t["t_start_ns"]).astype(np.float64) / 1e6
        per = per[~np.isnan(t["period_us"][1:])]  # an operator pause is not a period (NaN marker)
        primary = self.runtime.primary
        return {
            "ep": ep, "seed": seed, "mode": self.mode, "arm": None if self.sync else args.arm,
            "d_inj": args.d_inj if self.vl else None, "s": args.s, "B": args.B, "crn": args.crn,
            "vla": self.vla.url, "vla_be": self.vla_be.name, "checkpoint": self.vla_info.get("backend"),
            "tree": self.vla_info.get("tree"), "wbc": getattr(primary, "url", "ort"),
            "vla_target": self.vla_name, "wbc_target": self.wbc_name,
            "n_wbc_swaps": self.ep_wbc_swaps, "paused_ms": self.ep_paused_ms,
            "wbc_be": (WbcBackend.ORT.name if isinstance(primary, self.wb.OrtCpu) else primary.backend.name),
            "ort_opts": args.ort_opts, "gen": self.gen,
            "success": end == EpisodeEnd.SUCCESS, "first_success_tick": fst, "end": end.name,
            "k_end": k_end, "n_ticks": int(len(t)),
            "starve_fault": bool(self.st.starve_fault) if not self.sync else False,
            "n_starved": int(t["starved"].sum()), "n_late": oc[Outcome.LATE_REJECT],
            "n_timeouts": oc[Outcome.TIMEOUT], "n_errors": oc[Outcome.ERROR],
            "n_gen_drop": oc[Outcome.GEN_DROP], "n_applied": oc[Outcome.APPLIED],
            "n_requests": self.ep_req_i,
            "n_flips": int((np.diff(t["policy"].astype(np.int16)) != 0).sum()),
            "n_fallback": int(t["wbc_fb"].sum()), "wbc_calls": sum(self.runtime.n_calls.values()) - n_calls0,
            "wbc_reset_ping": wbc_ping, "drain": drain,
            "boundary_jump_p99": _pct(t["jump_rad"][splice], 99),
            "tick_p99_ms": _pct(per, 99) if self.rt else None,
            "compute_p99_ms": (_pct(t["compute_us"], 99) or 0) / 1e3,
            "slips": int(t["slip"].sum()),
            "chunk_age_mean": float(t["age"][with_row].mean()) if with_row.any() else None,
            "t_reset_s": t_reset_s, "preroll_ms": preroll_ms, "viewer_handoff": viewer_handoff,
            "wall_s": wall_s, "rss_mb": rss_mb(),
        }

    def seeds(self):
        it, _ = parse_seeds(self.args.seeds)
        if self.args.n_episodes is not None:
            it = itertools.islice(it, self.args.n_episodes)
        return it

    def run_episodes(self) -> int:
        args = self.args
        for ep, seed in enumerate(self.seeds()):
            if self.stop_requested:
                break
            if args.duration is not None and time.monotonic() - self.t_run0 >= args.duration:
                break
            if self.episode(ep, seed) is None:
                break
        self.interrupted = self.stop_requested
        return 130 if self.interrupted else 0

    def close(self) -> None:
        """Flush every log, then close sockets, env and context (each step independent)."""
        steps = [
            ("viewer", self._close_viewer),
            ("vla_obs", self._flush_vla_dumps),
            ("obs516", lambda: self.runtime.dump is not None and telemetry.save_obs516(self.run, self.runtime.dump)),
            ("summary", self._summary),
            ("ticklog", self.tlog.close), ("requestlog", self.rlog.close), ("episodelog", self.elog.close),
            ("pub", self.pub.close), ("vla", self.vla.close),
            ("control", lambda: self.control is not None and self.control.close()),
            ("vla_alts", lambda: [c.close() for c, _ in self.vla_clients.values() if c is not self.vla]),
            ("wbc_alts", lambda: [b.close() for b in self.wbc_targets.values()
                                  if isinstance(b, self.wb.ZmqWbc) and b is not self.runtime.primary]),
            ("wbc", lambda: isinstance(self.runtime.primary, self.wb.ZmqWbc) and self.runtime.primary.close()),
            ("shadow", lambda: self.remote_shadow is not None and self.remote_shadow.close()),
            ("uninstall", self.wb.uninstall), ("env", self.env.close),
            ("ctx", lambda: self.ctx.destroy(linger=0)),
        ]
        for name, fn in steps:
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - keep closing the rest
                print(f"rt_sim: close {name}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    def _close_viewer(self) -> None:
        """Terminate the viewer process (it never holds up the loop) and record how it went in config.json."""
        if self.viewer_link is not None:
            end = self.viewer_link.close()
            self.run.update_config(viewer={**(self.run.config.get("viewer") or {}), "end": end})

    def _summary(self) -> None:
        def counts(z):
            return dict(n_fallbacks=z.n_fallbacks, n_timeouts=z.n_timeouts, n_errors=z.n_errors,
                        n_latches=z.n_latches, n_recreate=z.n_recreate)

        p = self.runtime.primary
        wbc = {"n_calls": self.runtime.n_calls}
        if isinstance(p, self.wb.ZmqWbc):
            wbc.update(counts(p))
        remotes = {n: b for n, b in self.wbc_targets.items() if isinstance(b, self.wb.ZmqWbc)}
        if len(remotes) > 1 or (remotes and p not in remotes.values()):  # swaps: every target's own counts
            wbc["targets"] = {n: {"n_calls": b.n_calls, **counts(b)} for n, b in remotes.items()}
        if self.remote_shadow is not None:
            wbc["shadow"] = {"url": self.remote_shadow.url, "n_calls": self.runtime.n_shadow,
                             "n_miss": self.runtime.n_shadow_miss, **counts(self.remote_shadow)}
        self.run.write_json("summary.json", {
            "episodes": self.n_episodes, "interrupted": self.interrupted, "ticks": len(self.tlog),
            "requests": self.next_seq - 1, "vla_stats": self.vla.stats, "wbc": wbc,
            "control": {"applied": self.cstate.n_applied, "rejected": self.cstate.n_rejected,
                        "last_error": self.cstate.last_error, "pauses": self.n_pauses,
                        "paused_ms": self.paused_ms, "operator_resets": self.n_operator_resets,
                        "vla_swaps": self.n_vla_swaps, "wbc_swaps": self.n_wbc_swaps,
                        "pauses_at": self.pauses, "log": self.control_log[-500:]},
            "rss_mb_end": rss_mb(), "wall_s": time.monotonic() - self.t_run0})


# --- CLI ----------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rtloop.loop", description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=MODES, default="rt")
    ap.add_argument("--arm", choices=sp.ARM_NAMES, default="v1", help="splicer arm (rt, vl)")
    ap.add_argument("--s", type=int, default=16, help="trigger period, ticks")
    ap.add_argument("--B", type=int, default=0, help="crossfade ticks on a switch")
    ap.add_argument("--d-inj", type=int, default=4, help="VL: ticks from k_obs until a chunk is visible")
    ap.add_argument("--d-init", type=int, default=4, help="RT: d_hat before any measurement (5090: 4, TT: 3)")
    ap.add_argument("--sync-rows", type=int, default=20, help="rows per chunk of the synchronous arms / sync mode")
    ap.add_argument("--vla", default=os.environ.get("VLA_MOCK", "tcp://127.0.0.1:5559"))
    ap.add_argument("--wbc", default="ort", help="'ort' (in process) or a W-WBC URL")
    ap.add_argument("--wbc-timeout-ms", type=int, default=5)
    ap.add_argument("--wbc-no-latch", action="store_true")
    ap.add_argument("--no-shadow", action="store_true", help="no ORT shadow behind a remote WBC")
    ap.add_argument("--shadow", default=None, metavar="URL",
                    help="T-3.6: a W-WBC that shadows the ORT primary (shadow_da = |a_ort - a_url|; a miss is NaN)")
    ap.add_argument("--shadow-timeout-ms", type=int, default=None, help="--shadow reply timeout (default --wbc-timeout-ms)")
    ap.add_argument("--ort-opts", choices=("single", "default"), default="single")
    ap.add_argument("--onnx-dir", default=None)
    ap.add_argument("--crn", action="store_true", help="options.rt.noise_seed (D17)")
    ap.add_argument("--seeds", default="0-", help="'A-B', 'A-' (open), 'A' or a comma list")
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--duration", type=parse_duration, default=None,
                    help="stop starting episodes after this wall time (120, 2m, 2h)")
    ap.add_argument("--max-ticks", type=int, default=MAX_EPISODE_TICKS)
    ap.add_argument("--viewer-cpus", default=",".join(map(str, affinity.AUX_CPUS)),
                    help="CPUs of the --viewer process ('' = unpinned; --no-affinity also unpins it)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--pub", nargs="?", const=telemetry.TELEMETRY_PUB, default=None,
                    help=f"telemetry PUB (default {telemetry.TELEMETRY_PUB})")
    ap.add_argument("--control", nargs="?", const=control.CONTROL_PULL, default=None,
                    help=f"operator control PULL for the HUD keys v/w/r/p (default {control.CONTROL_PULL})")
    ap.add_argument("--vla-alt", default="", metavar="NAME=URL[,...]",
                    help="VLA swap targets (HUD v; applied at the next reset)")
    ap.add_argument("--wbc-alt", default="", metavar="NAME=URL[,...]",
                    help="W-WBC swap targets besides 'ort' (HUD w; applied from the next tick)")
    ap.add_argument("--swap-wbc-every", type=float, default=None, metavar="S",
                    help="T-3.9: swap the WBC to the next target every S seconds of ticks")
    ap.add_argument("--dump-obs516", action="store_true")
    ap.add_argument("--dump-vla-obs", type=int, default=0, metavar="N")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--runs-root", default=None, help=f"default ${telemetry.RUNS_ENV} or $RT/runs")
    ap.add_argument("--no-affinity", action="store_true")
    ap.add_argument("--affinity", action="store_true", help="pin in vl/sync too (default: rt only)")
    ap.add_argument("--pacing", choices=clock.PACING, default="spin",
                    help="RT wait between ticks (see rtloop.clock): spin, hybrid (sleep then spin), sleep")
    ap.add_argument("--spin-ms", type=float, default=4.0, help="hybrid pacing: spin this long before T_k")
    ap.add_argument("--give-up-ms", type=int, default=500, help="RT: abandon a VLA request after this")
    ap.add_argument("--block-timeout-ms", type=int, default=30000,
                    help="pre-roll, VL visibility and sync calls (frozen time)")
    ap.add_argument("--warmup", type=int, default=0, help="blocking VLA calls before the first episode")
    ap.add_argument("--no-malloc-trim", action="store_true",
                    help="skip malloc_trim(0) in the reset window")
    ap.add_argument("--mmap-threshold-kb", type=int, default=4096,
                    help="fixed glibc mmap threshold (0 = glibc's dynamic default; RSS then grows per episode)")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    args = build_parser().parse_args(argv[1:])
    _, open_ended = parse_seeds(args.seeds)
    if open_ended and args.n_episodes is None and args.duration is None:
        build_parser().error("open-ended --seeds needs --n-episodes or --duration")
    if args.mode == "vl" and args.wbc != "ort":
        print("rt_sim: warning: VL runs should use --wbc ort (PLAN.md §1.6)", file=sys.stderr)
    if args.shadow and (args.no_shadow or args.wbc != "ort"):
        build_parser().error("--shadow URL shadows the ORT primary: it needs --wbc ort and no --no-shadow")
    loop = Loop(args, argv)

    def on_signal(signum, _frame):
        loop.stop_requested = True
        print(f"rt_sim: signal {signum}: stopping after this tick", file=sys.stderr, flush=True)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        return loop.run_episodes()
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())
