# rtloop module contracts (M1, and the M2/M3 contracts M1 code consumes)

This file is the contract between the M1 modules. `PLAN.md` is the design
(`docs/plan/vla-wbc-x280/PLAN.md`, §1–§4); this file fixes the Python APIs so
the modules can be built in parallel. `rtloop/schema.py` and `rtloop/wire.py` are already written
and are shared by everything; change them only by adding, never by renaming. The sections after
"viewer.py + viewer_proc.py" summarise the M2/M3 contracts (rttt, chip sessions, the X280 kernel,
mailbox, host layer and x280d) as far as the loop, the HUD and the jobs depend on them; the source
files named there stay authoritative.

## Conventions

- Run everything through the wrappers in `env.sh` (`source ~/experiments/GR00T-WholeBodyControl-TT/env.sh`):
  `simpy` = sim/client venv (CUDA hidden, EGL on GPU 0; has torch CPU, onnxruntime 1.23, pyzmq,
  msgpack, numpy 1.26, pytest), `srvpy` = server venv (CUDA torch, onnx, pyzmq; **no
  onnxruntime**). Both run with `cd $R` and `PYTHONPATH=$RT:$R`.
- Tests: `simpy -m pytest $RT/tests/<file> -q`. Unit tests bind servers to `127.0.0.1` on a
  **random free port** (`socket.bind_to_random_port`), never to 5555–5572. Tests that need the
  GPU or a sim env are marked `@pytest.mark.gpu` / `@pytest.mark.sim` (registered in
  `tests/conftest.py`).
- **Never** stop, restart or send `kill` to the server on 127.0.0.1:5555. Do not open the
  Tenstorrent device. Do not edit files outside `~/experiments/GR00T-WholeBodyControl-TT/`.
- Times are `time.monotonic_ns()` (CLOCK_MONOTONIC) ints. Ticks are ints. Arrays crossing an
  interface are float32 unless stated.
- Style: plain modules, type hints, short docstrings that say *why*; no frameworks. Keep the
  per-tick paths allocation-light (no pandas, no logging calls per tick).

## schema.py / wire.py (done)

See the files. Key items: `DT, H=30, A35=35, ACTION_SLICES, ARMS, HANDS, WAIST, HEIGHT, NAV,
BLEND`, `build_vla_obs(obs)`, `state43(obs)`, `actions_to_rows(action) -> (30,35)`,
`rows_to_actions(rows)`, `row_to_env_action(row)`, `hold_row(obs)`, `Chunk`, `CHUNK_DTYPE`,
`Outcome`, `EpisodeEnd`, `TICK_DTYPE`, `REQUEST_DTYPE`, `WbcBackend`, `VlaBackend`,
`WBC_POLICIES`. `wire.dumps/loads/request` speak the stock PolicyServer format.

## splicer.py (WP-1.2) — pure functions, no I/O, no clocks

```python
@dataclass
class SplicerConfig:
    arm: str = "v1"          # "v1" | "sync_pause" (C2) | "sync_hold" (C2b) | "async_noskip" (C3) | "sync_aligned" (C6)
    s: int = 16              # trigger period, ticks (v1, C3)
    B: int = 0               # crossfade ticks on a switch (0 = hard switch)
    margin: int = 2          # s_eff = max(1, min(s, H - d_hat - margin))
    late_reject_rows: int = 4  # reject a chunk if usable rows H - d <= this (d >= 26)
    starve_ramp: int = 5     # ticks to ramp nav to 0 when starved
    starve_fault: int = 50   # consecutive starved ticks -> STARVE_FAULT flag (episode continues)
    d_init: int = 4          # d_hat before any measurement (4 on the 5090, 3 on TT)
    sync_rows: int = 20      # rows a synchronous arm executes per chunk (C2/C2b/C6)

@dataclass
class SplicerState:          # plain fields only (portable to C in M4b)
    act: Chunk | None        # chunk being executed
    pending: Chunk | None    # arrived, switches in at pending.k_arr
    in_flight: bool          # a request has been sent and is not yet registered by on_arrival
    k_req: int               # tick of the last sent request (-10**9 initially)
    last_outcome: Outcome
    blend_t0: int
    last_row: np.ndarray | None          # last executed A35 command
    last_row_at_switch: np.ndarray | None
    starve_n: int
    starve_fault: bool
    d_hist: list[int]        # last 10 measured d
    gen: int

def new_state(gen: int = 0) -> SplicerState
def d_hat(st, cfg) -> int                 # p90 (9th order statistic) of the last 10 d; cfg.d_init if none
def s_eff(st, cfg) -> int
def trigger(k, st, cfg) -> bool           # PLAN.md §3.1 "Trigger", per arm (sync arms: when rows [0, sync_rows) are used up)
def on_send(k, st) -> None                # k_req = k, in_flight = True
def on_arrival(k, chunk, st, cfg) -> Outcome
    # chunk.k_arr = k; in_flight = False; gen mismatch -> GEN_DROP; d >= H - late_reject_rows ->
    # LATE_REJECT (retrigger next call of trigger); else APPLIED: chunk.k_base = k_obs (v1, C6)
    # or k_arr (C2, C2b, C3), st.pending = chunk; d appended to d_hist.
def on_failure(k, st, outcome) -> None    # timeout/error: in_flight = False, last_outcome = outcome
def select(k, st, cfg, hold: np.ndarray) -> tuple[np.ndarray, dict]
    # returns (cmd35 float32 copy, info) with info keys: row (-1 if none), age, rows_left,
    # starved, splice (1 on the switch tick), jump_rad (L2 over ARMS+WAIST between this command
    # and last_row on a switch tick, else 0), chunk_seq.
    # `hold` is hold_row(obs) of the current tick, used only before the first chunk (pre-roll
    # fallback). Starvation (row >= H, or the sync arms' pause while a request is in flight):
    # hold last arms/hands/waist/height, ramp nav linearly to 0 over starve_ramp ticks.
def reset(st, gen) -> None                # new episode / instruction / VLA swap: clears act/pending, gen = gen
```

Index law: row index = k − chunk.k_base. PLAN.md §3.1 has the reference `select()`; the
blend uses `last_row_at_switch` (the command executed on the tick before the switch).

## vla_client.py + rtserve/mock_vla_server.py (WP-1.3)

```python
@dataclass
class VlaReply:
    seq: int; gen: int; status: ChunkStatus
    action: dict | None      # decoded "action.*" arrays (None on error/timeout)
    info: dict               # server INFO ({} from stock servers)
    t_send_ns: int; t_recv_ns: int
    ser_us: float; deser_us: float
    error: str | None

class VlaClient:
    def __init__(self, url: str, give_up_ms: int = 500, ctx: zmq.Context | None = None)
    def call(self, endpoint: str, data: dict | None = None, timeout_ms: int = 5000)  # blocking; only when nothing is in flight
    def ping(self) -> bool
    def send(self, vla_obs: dict, rt: dict) -> None   # NOBLOCK; rt = options.rt (PLAN.md §1.4.1); raises if in flight
    @property in_flight -> bool
    @property inflight_rt -> dict | None              # the rt dict of the request in flight
    def poll(self) -> VlaReply | None                 # zero-timeout; on give-up (now - t_send > give_up_ms)
                                                      # re-creates the REQ socket (lazy pirate) and returns status TIMEOUT
    def wait(self, timeout_ms: int) -> VlaReply | None  # blocking variant (pre-roll, VL visibility, sync mode)
    def drain(self, timeout_ms: int = 500) -> str     # "none" | "ok" | "recreate"; for episode end
    def close(self)
```

`send` builds `{"endpoint": "get_action", "data": {"observation": vla_obs, "options": {"rt": rt}}}`.
A reply is `[action, info]` (a 2-list) or `{"error": ...}`.

`mock_vla_server.py` (simpy or srvpy; imports only numpy/zmq/msgpack + rtloop): a REP server on
`--port` (default 5559), endpoints `ping`, `info` (`{"backend": "mock:<mode>", "proto": 2}`),
`get_modality_config` (returns `{}`), `reset`, `get_action(observation, options=None)` →
`[rows_to_actions(rows of shape (30,35)), info]` where info echoes `options.rt` plus
`t_recv_ns`, `t_reply_ns`, `infer_ms`, `backend`. Modes (`--mode`):

- `hold`: every row = hold of the request's state (arms, hands, waist), height 0.74, nav 0.
- `detfn`: deterministic function of the request (state bits and `options.rt.k_obs` only):
  arms = state + 0.05·sin(2π·i/30 + φ) with φ derived from the state bytes; nav row i =
  [0.2, 0, 0] when `((k_obs + i) // 200) % 2 == 1` else 0 (exercises Walk); hands/waist held.
- `ramp`: arms row i = state arms + i·δ (δ = 1e-3 rad, computed in float32), nav row i = 0,
  hands/waist held, height 0.74. Used by the alignment test (T-1.2b).
- `record-proxy --upstream URL --record-dir D`: forwards each request to the upstream server and
  saves `{request observation, options, reply}` as `D/<seq:06d>.npz` + jsonl index.
- `replay --replay-dir D`: replies with the recorded reply of the next request; **errors** if the
  request's state arrays differ from the recorded ones by any bit (image compared by sha1).
- `--latency-ms L --jitter-ms J` (uniform ±J, seeded), `--stall-after N --stall-s T` (the
  N-th request sleeps T s), `--seed`.

## wbc_backends.py + rtserve/wbc_gpu_server.py + rtserve/gpu_vla_server.py (WP-1.4)

```python
class OrtCpu:                         # in-process onnxruntime, CPU provider
    def __init__(self, onnx_dir: str = $ONNX, opts: str = "single")   # "single": intra_op_num_threads=1; "default": default SessionOptions
    def __call__(self, policy: str, obs: np.ndarray) -> np.ndarray     # obs float32 (1,516) -> (1,15)

@dataclass
class WbcCall:                        # what the loop reads back after each env.step
    tick: int; policy: str; obs: np.ndarray; action: np.ndarray
    backend: WbcBackend; fallback: bool; rtt_us: float; dev_us: float; shadow_da: float

class ZmqWbc:                         # W-WBC client (REQ), one per process
    def __init__(self, url: str, fallback: OrtCpu | None, timeout_ms: int = 5, latch: bool = True,
                 proto: int | None = None, ctx=None, reference: OrtCpu | None = None)
    def __call__(self, policy: str, obs: np.ndarray, tick: int, seq: int) -> tuple[np.ndarray | None, dict]
        # meta: rtt_us, dev_us (server compute_us), fallback (bool), backend
        # timeout -> ORT for this tick, socket re-created; 3 consecutive -> ORT for the rest of the
        # episode unless latch=False; reset_episode() clears the latch.
        # fallback=None (a shadow, T-3.6): a miss returns (None, meta) instead of an ORT action.
    def info(self) -> dict; def reset_episode(self); def close(self)
    # Connect-time parity check on 8 random inputs per policy against ORT (`reference`, default the fallback):
    # tolerance = the server's info.parity_tol when it reports a finite positive one (x280d: fp16 1e-2, fp32 5e-5,
    # T-3.1), capped at MAX_PARITY_TOL = 1e-2, else PARITY_TOL = 1e-4. info() adds parity_tol_used, parity_max_err.

class WbcRuntime:                     # process-wide singleton the patched G1GearWbcPolicy calls
    primary                           # OrtCpu or ZmqWbc
    shadow                            # None, an OrtCpu (D16: behind a remote primary) or a fallback-free ZmqWbc
                                      # (T-3.6 --shadow URL, behind the ORT primary: called after it with the same
                                      # tick/seq; a miss is shadow_da = NaN, counted in n_shadow / n_shadow_miss)
    tick: int                         # set by the loop before each env.step
    last: WbcCall | None              # filled on every call
    dump: list | None                 # if a list, append (tick, policy, obs, action) (obs516 dumps)

def install(runtime: WbcRuntime) -> None
    # Class-level patch of gr00t_wbc.control.policy.g1_gear_wbc_policy.G1GearWbcPolicy.load_onnx_policy:
    # returns a closure `run_inference(input_tensor: torch.Tensor) -> torch.Tensor` (CPU float32 (1,15))
    # that calls runtime with the policy stem (Path(model_path).stem). No ORT session is created by
    # the policy object and the GR00T_WBC_SERVER hook is bypassed. Survives the per-reset re-creation
    # of the policy (WholeBodyControlWrapper.reset -> setup_wbc_policy). Idempotent.
def uninstall() -> None
```

`wbc_gpu_server.py` (srvpy): REP bound to `tcp://127.0.0.1:<--port 5556>` served from the
**main thread**, wrapping `rtserve/vendor/gear_wbc_policy.py:GearWbcPolicies` (vendored copy,
sha256 in `rtserve/vendor/SHA256SUMS`, checked at start). Endpoints: `ping`,
`info` → `{"backend": "cuda-graph:rtx5090", "proto": 2, "policies": [...]}`,
`gear_wbc_action(policy, obs, tick=None, seq=None)` → `{"action": (1,15) f32, "tick", "seq",
"compute_us"}`. Boot self-check: max |err| < 1e-4 against a pure-numpy float64 forward pass of
the ONNX initializers (the server venv has no onnxruntime) on 64 random inputs per policy.
`--onnx-dir` default `$ONNX`. `--spin` (default off) busy-polls the REP socket instead of blocking in
`recv`; only with a dedicated `--cpus` core whose SMT sibling is idle (T-1.4 measured it inconclusive
with CPU 20, the sibling of 4, busy). `demo.sh --wbc-spin` passes it.

`gpu_vla_server.py` (srvpy): builds `Gr00tPolicy(embodiment_tag=UNITREE_G1, model_path=--model,
device="cuda")` → `Gr00tSimPolicyWrapper` exactly as `$R/gr00t/eval/run_gr00t_server.py` does
(strict=True), serves the PolicyServer wire on `tcp://127.0.0.1:<--port 5565>` with endpoints
`ping`, `info` (`{"backend": "cuda:rtx5090:<repo>@<sha8>", "proto": 2}`), `get_modality_config`,
`reset`, `get_action(observation, options=None)`. If `options.rt.noise_seed` is present it calls
`torch.manual_seed` + `torch.cuda.manual_seed(noise_seed)` before inference (PLAN.md D17).
Returns `[action, info]` with info = `{"rt": <echo>, "t_recv_ns", "t_reply_ns", "infer_ms",
"noise_seed", "backend"}`. No WBC endpoint. `--model` default
`cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate`. A request the client already abandoned
(`schema.request_abandoned`) is answered `{"error": "stale ..."}` without inference. `--noise cpu`
(opt-in; default `cuda` keeps the M1 noise stream): with a noise_seed, the action head's one
`torch.randn` returns `randn(shape, generator=torch.Generator("cpu").manual_seed(seed), float32)`
cast to the dtype/device it asked for, which is exactly server_g1's draw, so TT and GPU arms with the
same seed are noise-paired (D17 across backends); info then adds `noise: "cpu"` and `randn_calls`. The `info`
endpoint reports `model`, `revision` and `noise` ("cuda" | "cpu"). Who starts it how: `scripts/demo.sh` with
`HF_HUB_OFFLINE=1` and `--model` = DP1's local snapshot dir (`rttt/convert_bf16.DEFAULT_SRC`, at SRC_SHA), plus
`--noise cpu` for m2/m3 runs (m1 keeps `cuda`); `jobs/_common.sh start_gpu_servers` the same for the TT-vs-GPU arms.
A server demo.sh finds already answering ping is reused as it is (with its own `noise`); the jobs' `pair` check
(`need_gpu_servers pair`) refuses one whose `info.noise` is not `cpu`.

## telemetry.py + scripts/analyze_run.py + scripts/eval_paired.py (WP-1.6, WP-1.9)

```python
class RunDir:
    def __init__(self, root=$RT/runs, tag: str = "", config: dict = {})  # creates <stamp>_<tag>/, updates runs/latest symlink, writes config.json
    path: Path

class TickLog:                        # np.memmap of TICK_DTYPE, grown in blocks, flushed every 50 ticks
    def __init__(self, run: RunDir, capacity: int = 1 << 20)
    def next(self) -> np.void         # the next row to fill in place (zeroed)
    def commit(self)                  # count it; flush every 50
    def close(self)                   # truncate to the used length (ticks.npy stays loadable with np.load)

class RequestLog:                     # REQUEST_DTYPE rows -> requests.npz at close + requests.jsonl per row
    def add(self, row: dict) -> int; def update(self, seq: int, **fields); def close(self)

class EpisodeLog:                     # episodes.jsonl, one json object per episode (PLAN.md §4.1 fields)
    def add(self, rec: dict)

class Publisher:                      # PUB tcp://127.0.0.1:5570, SNDHWM small, NOBLOCK; drops on HWM
    def tick(self, fields: dict); def req(self, fields: dict); def ego(self, seq, k_obs, img_small); def meta(self, fields: dict)

def collect_config(argv, extra: dict) -> dict  # resolved CLI, git SHAs + `git status --porcelain` of RT/R/W,
                                               # sha256 of imported dirty/untracked files, affinity, env vars
```

`analyze_run.py <run_dir> --gates <name>` loads `ticks.npy`, `requests.npz`, `episodes.jsonl`,
applies the thresholds of `gates.yaml[<name>]` (measured outside reset windows; tick 0 included),
writes `verdict.json` (`{"gates": name, "pass": bool, "checks": [{name, value, threshold, pass}]}`)
plus plots (period/lateness/d/age/WBC-RTT histograms, Gantt of the first episode) and exits 1
on FAIL. Gate sets in `gates.yaml`: `m1_rt`, `m1_negctl`, `m1_sim_smoke`, `m2_rt`, `m3_sim`, `m3_rt`,
`m3_rt_slow` (PLAN.md §4.2; the comments in the file say where a set departs from the proposals in
`configs/gates_m{2,3}.proposed.yaml` and why). Keys added for M2/M3: `starved_frac_max`,
`starved_run_max`, `vla_errors_max` (outcome ERROR), `shadow_da_p999_max`, `wbc_x280_frac_min` (ticks
with wbc_be X280 or X280_SIM), `wbc_visible_wait_p99_ms` (wbc_rtt_us of ticks with wbc_early = 1; no
data until the WP-3.8b client exists) and `x280_heartbeat_continuity_min` (x280d's ring, status 3 =
stall, inside the run's time window; `--x280-ticks PATH`, else `RUN/x280d_ticks.npy`, else
`$CHIP_SESSION_DIR/x280d_ticks.npy` from config.json or the environment). The interval that contains an
operator pause (TickLog period_us = NaN on the next tick) is not a period; operator resets
(summary.json control.operator_resets) may leave episodes without an EpisodeLog record. `eval_paired.py --matrix <yaml>` runs conditions × seeds (50 by default, DP10), then
reports per-arm success with Wilson 95% CIs, the paired difference with a 95% Wald CI and the
lower 80% bound, and exact McNemar p, into `<out>/report.json` and a markdown table.

## loop.py and friends (WP-1.5) — the integration

`simpy -m rtloop.loop --mode rt|vl|sync --arm v1 --s 16 --B 0 --d-inj N --vla URL --wbc ort|URL
[--wbc-timeout-ms 5] [--wbc-no-latch] [--ort-opts single|default] [--crn] --seeds A-B
[--n-episodes N] [--duration S] [--viewer [--viewer-cpus 6,7]] [--pub] [--dump-obs516] [--dump-vla-obs N]
[--max-ticks 1440] [--tag T] [--control [URL]] [--vla-alt NAME=URL,...] [--wbc-alt NAME=URL,...]
[--swap-wbc-every S] [--shadow URL [--shadow-timeout-ms MS]]`. Files: `rtloop/{sim_env.py, viewer.py,
viewer_proc.py, clock.py, affinity.py, control.py, loop.py}`. The operator-control flags are described in
"control.py" below; `--shadow URL` (T-3.6) needs `--wbc ort`: that W-WBC shadows the ORT primary with no ORT
fallback (latched after 3 misses per episode like a primary, reset at every reset window), timeout
`--shadow-timeout-ms` (default `--wbc-timeout-ms`); config.json records it under endpoints.shadow and
wbc_info.shadow, summary.json under wbc.shadow {n_calls, n_miss, ...}. Per-tick flow, episode lifecycle and run modes: PLAN.md §1.2 and §1.6, exactly, except for the
viewer (next section): it is no longer inside `env.step` (§1.2 step 5) and is no longer pre-launched in the
reset window (lifecycle step 4); deviations (f) and (g) of §1.1 no longer apply.

## viewer.py + viewer_proc.py — the onscreen viewer, out of process

With `--viewer` the env is built exactly as without it (`onscreen=False`, stock), and the window belongs to
a separate process, `python -m rtloop.viewer_proc`, which the loop spawns itself (same simpy interpreter,
`CUDA_VISIBLE_DEVICES=""`, `MUJOCO_GL=glfw`, `$DISPLAY` inherited, pinned to `--viewer-cpus`, default the aux
CPUs 6,7 it shares with the HUD; `--no-affinity` or `--viewer-cpus ''` leaves it unpinned). Visualisation is
never on the control path: the loop never waits for the viewer, and a dead viewer (crash, or the user closed
the window) is logged once and the run goes on.

```python
class ViewerLink:                      # loop side (rtloop/viewer.py), one per run, only with --viewer
    def __init__(self, cpus: list[int] | None = None)  # creates the state block (POSIX shm, 1 MiB)
    def plan(self) -> dict                             # config.json "viewer" before the process exists
    def spawn(self, log_path: Path) -> dict            # start viewer_proc (log: <run dir>/viewer.log)
    def handoff(self, model, data) -> dict             # reset window: {gen, mjb_mb, ms}
    def publish(self, k: int) -> float                 # tick path, after env.step: returns viewer_us
    def check(self, k: int) -> bool                    # alive? logs the first time it is not
    def stats(self) -> dict; def close(self, grace_s=3.0) -> dict   # SIGTERM, SIGKILL after grace_s

class StateReader:                     # viewer side (rtloop/viewer_proc.py); also used by the handoff test
    def __init__(self, prefix: str)
    def read(self) -> tuple[int, int, int] | None      # (seq, gen, k) of a consistent snapshot
    def apply(self, model, data) -> None               # snapshot -> model/data of its generation
    def load_model(self, gen: int) -> MjModel | None   # None: superseded meanwhile
```

Shared memory, all created by the loop with `multiprocessing.shared_memory` (its resource tracker unlinks them
if the loop dies without `close()`), named `rtv<loop pid>_<6 hex>`:

- `<prefix>_s`, the state block. Header of 16 int64 words: 0 `seq`, 1 `gen`, 2 `k` (-1 = the reset state),
  3 `t_ns`, 4 payload bytes, 5 loop pid, written by the loop only; 8 viewer pid, 9 frames, 10 last seq shown,
  11 generation shown, 12 models loaded, written by the viewer only (its own cache line). The payload at byte
  128, in `viewer.scene_fields(model)` order: `time`, `qpos`, `qvel`, `act`, `mocap_pos`, `mocap_quat`
  (float64), `geom_rgba`, `site_rgba` (float32), each 8-byte aligned. Seqlock: the tick thread (the only
  writer) makes `seq` odd, copies, sets `gen`/`k`, makes `seq` even; it never waits. The reader copies the
  payload and accepts it only if `seq` was even and unchanged, otherwise retries (x86-64 store/load order).
- `<prefix>_m<gen>`, one per reset: robosuite builds a new MjModel at every hard reset, and `handoff` saves it
  with `mj_saveModel` (MJB, 136 MB with the textures, 43-57 ms), publishes the reset state with the new `gen`,
  then unlinks the previous generation. The viewer loads `/dev/shm/<prefix>_m<gen>` when `gen` changes and
  shows it in the same window (a superseded generation it has not opened yet is skipped). Checked: after
  loading, `mj_forward` of the copied state reproduces the env's `xpos`/`xquat` (and geom/site poses) bit for
  bit, at resets and mid-episode (`tests/test_viewer.py`). Nothing else in this env's model changes during an
  episode (all MjModel arrays compared over 300 ticks); `geom_rgba`/`site_rgba` are copied anyway.

The viewer renders at about 60 Hz with the free tracking camera of the day-1 prototype `live_viewer.py` (git
history at c1d5b90: follows the pelvis, mouse orbits and zooms, collision geoms hidden as in robosuite's
mjviewer). It exits when its window is closed, on SIGINT/SIGTERM (the loop's `close()` sends SIGTERM on every
exit path: normal, SIGINT, SIGTERM), or when the loop dies (`PR_SET_PDEATHSIG` = SIGKILL, plus a parent-pid
check per frame). With no usable display it exits at once (rc 2, "GLFW cannot open a window" in `viewer.log`).

Recorded: config.json `viewer` (plan, pid, CPUs, shm prefix, log, DISPLAY; at close `end` = published,
generations, frames, models loaded, `exited` {rc, k} if it died early, the viewer's threads and CPUs, rc),
`affinity_plan.viewer_cpus`; each episode record `viewer_handoff` {gen, mjb_mb, ms}; each tick `viewer_us`
(the publish time, so the m1_rt viewer gates now measure the new path; 0 without `--viewer`).

## control.py (WP-1.7): HUD keys → loop, PUSH → PULL `tcp://127.0.0.1:5571`

`rtloop/control.py` (pure parse/validate/resolve/apply plus the two socket wrappers; `scripts/control_proto.py` is a
re-export shim). One msgpack dict per key press, `{"cmd": "swap_vla"|"swap_wbc"|"reset"|"pause", "to": NAME|"next"|
"on"|"off", "id": int, "t_ns": int}`. Unknown keys, wrong types, messages over 512 B and commands older than 2 s
(`t_ns`, CLOCK_MONOTONIC) are rejected and counted, so a key press queued while no loop listened is never applied
to a later run. The HUD's PUSH uses ZMQ_IMMEDIATE: with no loop listening the send fails visibly.

| Command | Semantics in rt_sim |
|---|---|
| `swap_wbc NAME\|next` | from the next tick (the MLP call is stateless, D2). A remote target gets the ORT shadow (D16); ORT as primary gets the `--shadow` W-WBC back, if any. A latched target is unlatched at the swap (no I/O). |
| `swap_vla NAME\|next` | queued; applied in the next reset window, before the gen bump, after the episode-end drain (nothing in flight); d_hist is cleared. `r` applies it at once. |
| `reset` | ends the episode at the tick boundary with no EpisodeLog record and moves on to the next seed |
| `pause [on\|off]` | freezes between ticks (toggle without `to`); commands are still served, meta is re-published at 1 Hz; on resume the RT pacer re-anchors so T_{k+1} = now + 20 ms (no slip). The next tick's `period_us` is NaN, and analyze_run leaves that interval out. |

Loop flags: `--control [URL]` (bare = `$CONTROL_PULL`), `--vla-alt NAME=URL,...` (pinged, info'd and
modality-checked at startup), `--wbc-alt NAME=URL,...` (ZmqWbc with the parity check; `ort` is always a target and
the primary is named by its backend: gpu, x280, x280_sim), `--swap-wbc-every S` (T-3.9: a synthetic `swap_wbc next`
every S seconds of ticks). The control PULL is polled once per tick after the TickLog commit and the PUB (about 2 µs
idle). The `meta` topic adds `seed`, `endpoints {vla, wbc}`, `current {vla, wbc}`, `targets {vla, wbc}`,
`pending_vla`, `paused`, `control`; the episode record adds `vla_target`, `wbc_target`, `n_wbc_swaps`, `paused_ms`
(and `vla`/`wbc` are the URLs in use); summary.json gets `control {applied, rejected, last_error, pauses, paused_ms,
operator_resets, vla_swaps, wbc_swaps, pauses_at [[ep, k, ms], ...], log (last 500 actions)}`.

## Device status streams (HUD chip panel, PLAN.md §4.3 item 7)

Each device process may publish an out-of-band status stream; `scripts/hud.py` subscribes to both (absent publishers
are normal) and never sends a request to `:5557` or `:5558`.

| Publisher | Flag (default off) | URL |
|---|---|---|
| x280d (`x280/x280d.py serve`) | `--status-pub [URL]` (`0` = random port), `--status-hz` (2-10); chip_session: `CHIP_X280_STATUS_PUB` (empty by default; `demo.sh --hud` passes `--status-pub` itself) | `$X280_STATUS` = `tcp://127.0.0.1:5573` |
| server_g1 (`rttt/server_g1.py`) | `--status-pub [URL]`, `--status-hz`; chip_session: `CHIP_VLA_STATUS_PUB` | `$TT_STATUS` = `tcp://127.0.0.1:5574` |

zmq PUB, SNDHWM 10, NOBLOCK sends (dropped, never blocking), multipart `[b"status", rtloop.wire.dumps(msg)]` at 2-10
Hz from a thread that never holds a request-path lock for longer than a field copy. `msg` keys are absent or None
when unknown, never invented:

- common: `src` ("x280d" | "server_g1"), `pid`, `t_ns` (CLOCK_MONOTONIC), `seq`, `backend` (as `info`), `uptime_s`.
- server_g1: `calls`, `errors`, `last_ms` / `p50_ms` {encode, device, decode, total} (last 50 calls), `device_duty`
  (device ms / wall ms over the last 5 s), `trace` (bool), `cq` (int), `core_map` (MK's live megakernel CoreMap:
  `grid`, `megakernel {readers, hub, compute, spare}` counts, plus `readers`/`hub`/`spare` coordinates; compute =
  the rest of the grid), `tree`, `telemetry` (always None: ttnn has no read of the CMFW table). Also `ready`,
  `stale`.
- x280d: `tile`, `tile_state` (ledger: fresh/released/parked/burned), `monitor_state` (the mailbox state as a
  lower-case **name**, never the number: boot, idle, busy, fault, reentered, host_gone, parked), `heartbeat` (the
  raw L1 heartbeat word, the hart's 50 MHz mtime, read once per snapshot by the heartbeat check that also gives
  `heartbeat_age_us`: no extra device read), `heartbeat_age_us` (time since that word last changed as this host saw
  it, as of the snapshot; kept for compatibility: a healthy hart's word moves between any two reads, so it stays ~0),
  `seq_done`, `cycles_last`, `cycles_max`, `compute_us_last`, `duty` (compute µs / 20 ms over the last 50 calls),
  `x280d_us_last` / `x280d_us_p50` / `x280d_us_p99` (x280d's request-in to reply-ready µs over the last 50 OK rows of
  the tick ring), `calls_ok` (W-WBC calls served with an action since start; 0 before any), `n_runs`, `n_overrun`,
  `n_reentry`, `beu_accrued`, `ccache_ecc {fix, fail}` (COUNT registers, at most 1 Hz), `variant` ("v3") and `dtype`
  ("fp16") as two fields, `fw_ver`, `build_id`, `telemetry`. The kernel is deterministic (`compute_us_last`,
  `cycles_*` and `duty` repeat call after call, 278.07 µs / 222,455 cycles for v3-fp16) and a 0.28 ms call every
  20 ms is almost never in flight at a snapshot, so `monitor_state` reads idle: what moves while x280d serves is
  `heartbeat`, `x280d_us_*`, `calls_ok` and `n_runs`. Extras beyond the shared keys:
  - `telemetry_source`: `cmfw` on the device (the CMFW table through the read-only ARC/CSM windows); `none (sim)` on
    `--backend sim`, whose `telemetry` is always None (its FakeDevice has no chip behind it); `cmfw (FakeDevice
    table, unit test)` only in unit tests. `info()` reports it too, plus `status_pub` (the bound URL).
  - `snapshot_age_ms`: how old the serving thread's snapshot was when sent (the serving thread builds it between
    requests, so a long admin command makes it stale; `heartbeat_age_us` is as of the snapshot).
  - `mode`, `degraded`; `tiles {idx: {...}}` (DP4: the second tile's monitor fields; the top level is `tile`'s).
  - `telemetry_error` (the CMFW read failed; `telemetry` stays None), or `error` (the whole snapshot failed: only
    the common keys, `mode`, `degraded` and `telemetry_source` are sent).
  - Flags: `serve --status-pub [URL]` (no value: `$X280_STATUS`; `0`: a random port), `--status-hz` (clamped 2-10).
- `telemetry` (CMFW table, fw `bh_arc/telemetry.h` tags, memory reads only through x280d's ARC window, decoded by
  `x280/regs.py decode_status_telemetry`): `aiclk_mhz`, `l2cpuclk_mhz` [4], `asic_temp_c`, `vcore_mv`, `tdp_w`,
  `tdc_a`, `input_power_w`, `timer_heartbeat`.

The HUD (`--x280-status`, `--tt-status`, `--no-status`, `--chip auto|on|off`, key c) marks a source STALE after 2 s,
shows the last values greyed, a DEGRADED pill for x280d's `degraded`/`error`, a BUILDING pill for server_g1's
`ready: false`, and places the core roles from the reported coordinates (counts only: MK's `default_core_map()` when
its grid and counts agree, labelled as such). The telemetry row shows numbers only from a LIVE source that is not
simulated (a backend `x280-sim:…` / `mock:…`, or a `telemetry_source` naming a sim or fake), labelled `<source> via
<src>` (`telemetry_source`, else "CMFW table"). Otherwise it says why there are none, never a number: "no CMFW
telemetry (sim)", "(server_g1 has no CMFW read)", "(x280d read failed)", "(x280d reports none)", or "chip telemetry
(CMFW table): no live source". The HUD's exit summary records the last row as `chip_telemetry {src, source, note}`.
Without `x280.regs` the floorplan uses `hud.L2CPU_NOC0_FALLBACK`, a pinned copy of `regs.L2CPU_XY` (rows 3, 9, 5, 7).
Counters become rates over ~1 s of consecutive snapshots on the publisher's clock at the moment they were read
(`t_ns - snapshot_age_ms`; a re-sent snapshot is no new sample, a restart starts over, a snapshot sent more than 1 s
late gives none). The X280 column shows: tile / hart ("serving N/s" from the tile's Δ`n_runs`/s, else Δ`calls_ok`/s,
while that is > 0 and the monitor is idle or busy; otherwise the monitor state, e.g. fault, parked, with the ledger
state when it is not released), heartbeat (Δ`heartbeat`/s, "+50.0 M/s"; critical "stopped N s" once it has not
advanced for 1 s (2.5 snapshot periods at a slower `--status-hz`), "stream stale" when the stream is; "age N µs" from a publisher without `heartbeat`), x280d
last/p99 (µs), compute last (µs, two decimals), calls served (`calls_ok` and calls/s), runs / overrun, reentry / BEU,
ECC fix / fail, variant, fw / build; the floorplan labels x280d's tile "serving N/s" the same way. The processor map
has no RTX 5090 box when nothing in the run is on the GPU: neither meta `processor` stamp classifies as GPU and no
meta `targets` name (the loop's swap targets, `--vla-alt` / `--wbc-alt`) matches gpu / cuda / 5090, and before any
meta (e.g. `demo.sh m3 --no-alt`, `jobs/m3_soak.sh`). It is decided from the run's configuration, never from a tick,
so the layout holds through fallbacks and swaps; the Blackhole box then takes the freed row and gives each engine a
line with its live state (server_g1's device p50 and calls/s; the X280 tile and "serving N/s"; without a live
stream the loop's round-trip p50).

## rttt (M2): `rttt/server_g1.py` on `:5557`

The W-VLA contract of §1.4.1 on the Tensix: endpoints `ping`, `info`, `get_modality_config`, `reset`,
`get_action(observation, options=None)` → `[action, info]`; no `kill`. info: the `rt` echo, `t_recv_ns` (before
decode), `t_reply_ns`, `infer_ms`, `noise_seed`, `backend`, `encode_ms`, `device_ms`, `decode_ms`, `tree` (the loop
forwards encode/device/decode on the `req` topic's `srv`, and the HUD shows the split). Backend stamps:
`tt:p150a:<repo>@<sha8>` (rtloop `vla_backend_of` → TT) and `mock:server_g1-fake` for `--fake-model` (→ MOCK).
`noise_seed` reseeds a CPU `torch.Generator` (fp32 (1, 50, 128), the same tensor as `rttt/g1_n16.make_noise(seed)`),
reported as `info.noise = "cpu"`; the GPU server draws the same noise only with `--noise cpu`
(`tests/tt/test_noise_cross_venv.py` compares the two draws across srvpy and ttpy, fp32 and after the bf16 cast).
torch's CPU generator keeps only a seed's low 32 bits (0 and 2**63 draw the same noise), on both sides alike. An
abandoned request gets `{"error": "stale ..."}` without inference. Test-only, behind
`--debug-endpoints`: `predict_debug(observation, noise)`, `x280_peek(x, y, addr, n)`, `x280_lim_write(x, y, addr,
data)` (LIM init method B, LIM only). The observation is cut to the 9 modality keys before `split_batch` (the stock
rollout sends 13 more). Per-call JSONL in `logs/server_g1/` by default (`--log`). T-2.1:
`ttpy -m pytest $RT/tests/tt/test_g1_host_cpu.py $RT/tests/tt/test_tt_server_protocol.py -q`. Kernel tree:
`env.sh`'s `ttpy` exports `TT_METAL_RUNTIME_ROOT=$MK` unless the caller set one (without it ttnn takes its root from
the resolved ttnn symlink, the live clone: S1, 2026-10-25), and server_g1's real-model path calls
`rttt/g1_n16.pin_runtime_root()` before ttnn is imported, which accepts that root, refuses any other preset one, and
is reported as `info.runtime_root`.

## M2/M3 device sessions: `scripts/chip_session.sh`, `scripts/start_vla.sh`

Run only under the guard (`$DEV`, or `$DEVX` for any `--x280` other than none):
`chip_session.sh [--x280 none|probe|boot:3|attach] [--vla none|tt] [--debug-endpoints] [--allow-reset] --job SCRIPT
[--job SCRIPT ...] [--tag T]`. Order: x280d `probe --json` (10 s) → if dirty: with `--allow-reset` (fuser empty)
`tt-smi -r 0`, 5 s, `scripts/tt_probe.py`, probe again; without: exit 3 → x280d `serve --port 5558 --tile 3
--no-release|--boot|--attach` (`CHIP_X280D_SERVE_ARGS` overrides) → server_g1 (READY = answers ping) → the jobs →
teardown in reverse (SIGTERM, SIGKILL after 10 s). Exit codes: 0; the first failing job's code (3 wins); 2 usage or
precondition; 3 X280 dirty; 4 probe/reset/tt_probe failure; 5 a server did not start or died; 6 a child survived
SIGKILL; 128+N on signal N. Jobs get `CHIP_SESSION_DIR`, `CHIP_SESSION_ID`, `CHIP_VLA_URL`, `CHIP_X280_URL`,
`CHIP_START_VLA` (= `start_vla.sh tt|start|stop|restart|status`, the only way a job restarts server_g1),
`CHIP_X280_MODE`, `CHIP_VLA_MODE`, `CHIP_DEBUG_ENDPOINTS`, `CHIP_VLA_PORT`, `CHIP_X280_PORT`, `CHIP_DEV_NODE`,
`CHIP_VLA_STATUS_PUB`, `CHIP_X280_STATUS_PUB`, `CHIP_SESSION_FAKE`, and `DEVICE_LOCK_TIMEOUT=1` (a nested `$DEV`
fails fast). A job exits 3 only for a real X280-dirty state. x280d probe contract: one JSON object `{dirty, reasons,
attachable, ...}`, exit 0 report / 3 dirty / 4 error; a probe killed at the timeout counts as dirty. Artefacts:
`runs/sessions/<id>_<tag>/`.

M3 boot (2026-10-25, E1F; S1c/S1d showed host-written LIM unusable): every M3 session runs `--x280 boot:3` with
`X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0 ..."`: x280d's stage-0 boot (the hart
zero-fills LIM, then the monitor; mode `booted`, as after `--boot`). The default `--boot` (LIM init method A) stays in
the code and is not used on silicon. Every M3 job takes either boot (`jobs/_x280.sh x280_boot_flag` records which),
none assumes a host-initialised LIM, and a restarted x280d is `serve --attach` after either. `demo.sh m3` (device)
builds that CHIP_X280D_SERVE_ARGS itself and refuses (exit 2, before the hold) without `X280_E1F_APPROVED=1` in the
caller's environment or with a CHIP_X280D_SERVE_ARGS that lacks `--stage0`; `--x280-sim` needs neither.

Placement (`scripts/chip_lib.sh`, shared by the session, `start_vla.sh` and the jobs):
- x280d: every `serve` the session starts gets `--cpus 5` (PLAN.md §1.1) unless its arguments set `--cpus` or
  `CHIP_SAFE_CPUS` excludes CPU 5 (then it stays on the session's CPUs), and `--status-pub $CHIP_X280_STATUS_PUB` only
  when that is set (empty by default: the stream adds a status snapshot after every reply, which gated timing runs do
  not pay for) and the arguments have none. One function, `x280d_start LOGFILE ARGS...`, adds both, also for the jobs
  that start x280d again (`jobs/_x280.sh x280_reattach`: T-3.8 after kill -9, S2's E3(d)).
- server_g1 (`vla_start`): runs under `taskset -c <its --cpus>` (default 8-11) with `OPENBLAS_NUM_THREADS`,
  `OMP_NUM_THREADS` and `MKL_NUM_THREADS` = 1 unless the caller set them, so its library pools never carry the
  session's mask (`info.affinity_check.repinned` stays []); `--torch-threads 0` ("library default") therefore also
  needs `OMP_NUM_THREADS` set to the wanted count.
- `CHIP_SAFE_CPUS` (a taskset list; unset = no pin): the session and every job pin themselves to it
  (`chip_pin_safe`), and everything that would take a real-time core refuses before it starts (exit 2; vla_start
  returns 5): a server_g1 `--cpus` outside it (vla_start and server_g1 itself), an x280d `--cpus` / `--sim-cpus` outside it (every
  x280d subcommand), the GPU servers a job would start (`start_gpu_servers`), the closed-loop jobs (`need_rt_cores`:
  t2_sync, t2_soak, t2_soak_2h, t2_primary, m3_eval, m3_faults, m3_soak, m3_shadow in rt mode; `rtloop.loop --mode rt` pins
  itself onto CPUs 2/3 whatever the job's mask) and `jobs/demo_m2.sh` / `demo_m3.sh`.

Job helpers and environment:
- `jobs/_common.sh` (every session job): `need_vla`, `need_debug_endpoints`, `need_rt_cores [WHAT]`,
  `need_gpu_servers wbc|vla|pair` (vla: the GPU VLA serves DP1@SRC_SHA; pair: also `info.noise` "cpu" and one seeded
  call replying `{noise: cpu, randn_calls: 1}`, saved as `gpu_vla_info.json`), `start_gpu_servers` (`T2_START_GPU=1`:
  wbc_gpu_server on CPU 4, gpu_vla_server on CPUs 12-15 from the local DP1 snapshot with `HF_HUB_OFFLINE=1` and
  `--noise cpu`), `session_matrix` (a matrix rewritten to this session's endpoints), `save_server_info`, and
  `job_exit` (a tool's exit 3 becomes 4). Env: `T2_DRY_RUN=1` (print the sim commands, run none), `T2_GPU_VLA`
  (default `$VLA_GPU`), `T2_WBC` (default `$WBC_GPU`; `ort` = the loop's in-process WBC), `T2_START_GPU`; T-2.6/T-2.8
  `T26_BENCH` (a `t23_bench.json`, or `projected`; unset = `rttt/m2_checks.py bench`'s pick, printed),
  `T2_SOAK_DURATION`, `T2_SOAK_TAG`, `T2_SOAK_DEMO_ARGS`, `T2_SOAK_LOOP_ARGS`, `T2_SOAK_GATE_DRIFT`; T-2.5
  `T2_ANCHOR_EPISODES`, `T2_SKIP_ANCHOR`; T-2.7 `T2_GPU_ARM_DIRS`, `T2_FORCE_REUSE`.
- `jobs/_x280.sh` (the M3 jobs): `need_x280 MODE...`, `x280 SUB ARGS` (= `x280d.py SUB --via $CHIP_X280_URL`),
  `x280_step NAME SUB ARGS` (JSON in `NAME.json`), `x280_reattach` (its `serve` arguments are the session's
  `CHIP_X280D_SERVE_ARGS`, unset = chip_session's default, with --stage0 / --boot / --no-release swapped for --attach
  and --log-dir for x280d_attach, every other serve flag kept; `x280_reattach_args` refuses, starting nothing, what it
  cannot carry: --backend, --sim-cpus, unknown flags, no or two mode flags), `x280_boot_flag` (the session x280d's
  mode flag: stage0 | boot | attach | no-release | ?), `x280_rtt_client_args` (`bench --rtt`'s client placement:
  `--cpus $X280_RTT_CLIENT_CPUS`, default 2,3 = rt_sim's tick and helper CPUs, '' = unpinned, dropped with a log line
  when outside CHIP_SAFE_CPUS; `--spin` with `X280_RTT_SPIN=1`), `x280_dirty WHY` (exit 3).
  Env: `X280_SLOW=1` (DP11), `X280_TIMEOUT_S`; S1c `S1C_E1A_STORE` (method A's store, default a64); S1d
  (`jobs/s1d.sh`) `S1D_PLAN`, `S1D_LINES`, `S1D_PROBE_LINES`, `S1D_PARTIAL`, `S1D_SEED`, `S1D_BASE`; T-3.7 `M3_T23_BENCH` (required: a `t23_bench.json` or `none`), `M3_EVAL_EPISODES`,
  `M3_EVAL_MAX_TICKS`, `M3_EVAL_SEEDS`, `M3_EVAL_LOOP_ARGS`, `M3_SKIP_SCREEN`; T-3.6 `M3_SHADOW_*`; T-3.8
  `M3_FAULT_DURATION`, `M3_FAULT_GAP_S` (30 s between the cases), `M3_FAULT_SEEDS`; T-3.5 `M3_RTT_N`; T-3.E4
  `E4_ITERS`, `E4_RTT_N`, `E4_TT_N`; T-3.E5a `X280_PARITY_OBS`, `X280_PARITY_N`, `X280_PARITY_N_V0`; T-3.9
  (`jobs/m3_soak.sh`) `M3_SOAK_DURATION` (7200), `M3_SOAK_SWAP_S` (10), `M3_SOAK_TAG` (t39), `M3_SOAK_SEEDS`,
  `M3_SOAK_DEMO_ARGS` (default `--hud --record`), `M3_SOAK_LOOP_ARGS`; S2 `X280_E3D`, `X280_E2M_ZERO_DEVICE`.
  CPU-test hooks only: `M3_EVAL_MODE=sync` / `M3_SHADOW_MODE=sync` (the smokes), `M3_LOOP_MODULE` (m3_faults.sh's
  loop, a stand-in in tests/x280), `DEMO_LOOP_MODULE` (demo.sh's, also under m3_soak.sh).
- M3 job results: S2 `s2_summary.json` (`boot`, the E2 rows incl. `selftest_no_rnmi`, `e3_watch_pll4_default`,
  `e3_relock_events` / `e3_conclusive`, reported); E4 `e4_choice.json` (`t35_gates_idle/busy`, `rtt_client`); E5a
  `e5_summary.json` (`obs`, `boot`, per run `rows` / `rows_served`); T-3.5 `t35_summary.json` (`checks` = PLAN T-3.5's
  x280d-side gates, `pass` from them alone, `reported` = client RTT and placement, TT latency); T-3.9
  `t39_summary.json`.
- `rttt/m2_checks.py` (simpy): `gates SRC DST [BENCH]` (m2_rt with vla_p50 = 1.1 x the bench's server p50), `bench
  ROOT [ARENA]` (the newest passing `t23_bench.json` of the bfp8_b arena), `soak RUN DUR OUT [--gate-drift]` (the
  HUD's `vla=tensix` placement and the VLA p50 drift), `errors RUNS OUT` (0 server errors or hangs per arm), `matrix
  SRC DST TT GPU WBC`, `tt-only MATRIX DST`, `reuse TT_RUNS|- GPU_DIRS OUT [--force]` (`--force` never overrides
  unpaired noise).
- `x280/tools/m3_jobs.py` (the M3 jobs' verdicts, stdlib only except m3-gates): `s1c-summary DIR E1A_OK METHOD
  [k=rc ...]` (S1c and the S2 GO/NO-GO, plus `ecc`: each LIM init's and check's CCACHE ECC deltas, events per line /
  per word read, last event address and the failed parts of each gate; reporting only), `s1d-summary DIR [k=rc ...]`
  (S1d: per store `clean` or not, `clean_variants`, `recommendation` {verdict host_init | two_pass | no_host_init |
  inconclusive, lim_init, text}, `lines` for the job log), `m3-gates GATES_YAML OUT BENCH|none DTYPE` (gates.yaml's m3_rt / m3_rt_slow
  with vla_p50 = 1.05 x the T-2.3 server span p50), `t37-summary VERDICT BEFORE AFTER DUMP BENCH|none OUT`, `t36-gates
  OUT DTYPE`, `t36-summary VERDICT DTYPE LOOP_RC OUT`, `t39-gates OUT` (period p99 <= 21 ms, 0 slips, 0 fallbacks),
  `t39-summary VERDICT TICKS BEFORE AFTER T0 T1 LOOP_RC OUT STATUS_JSONL...` (numpy: simpy; 0 traps and clean x280d
  counts across the soak, |da| <= 1e-2 at every WBC swap from the ORT shadow of its remote side, PLL4 default in every
  10 s x280d status read with no gap > 30 s); each prints its JSON as the last line, exit 0 / 1 / 2.
- T-2.2 (`jobs/t2_fidelity.sh`): `t22_fidelity.json` also records TT-vs-fp32 per sample and per group (`tt_fp32`,
  `tt_fp32_max`, `argmax_fp32`, not gated), and `t22_tt_outputs.npz` keeps every TT output as `<sid>/<key>`.

## X280 kernel (WP-3.1): `x280/fw/wbc_mlp.h`

The contract between the kernel, the firmware and x280d. `wbc_infer(blob, policy, variant, obs, act, scratch)` with
`WBC_SCRATCH_BYTES` = 4096 (960 floats used); returns 0 or `WBC_EINVAL` -1 … `WBC_EDTYPE` -9 (magic, version,
size, CRC, layout, policy, variant, dtype). Variants v0 (scalar fp32, the bit-parity reference), v1 (RVV fp32, riscv
only), v2 (fp16 weights widened with integer ops, fp32 math; RVV on riscv), v2s (scalar fp16), v3 (fp16, bit-identical
to v2 on every input with one vector op fewer per weight: v2's exact 2^112 scale on x[i]; RVV e32m8 asm on riscv, the
same arithmetic in scalar code on x86; the fp16 default since E4/E5a passed on silicon); `wbc_variants_built()` is 0b11101
on x86, 0b11111 on riscv+V. Blob v1: 320 B header (`magic` "WBCB" @0, version, hdr_bytes, total_len,
`crc32` @16 = zlib CRC-32 over bytes [20, total_len), wdtype, layout W^T [in][out], policy_mask, n_layers 7,
align 64, the source .onnx CRCs, 2 × 7 layer descriptors, a 32 B tag); weights transposed in 64 B-aligned arrays,
biases fp32. Arithmetic: every multiply and add rounded separately (RNE, no FMA, `-ffp-contract=off`, frm = 0), so
v0 x86 == v0 riscv == v1 and v2 x86 == v2 riscv bit for bit. Blobs load at `WBC_BLOB_BASE` 0x0801_1000 and must end
below `WBC_BLOB_LIMIT` 0x081E_0000: `wbc_fp16.blob` (both policies) 1,889,088 B → 0x081D_E340; `wbc_fp32_balance` /
`wbc_fp32_walk.blob` (one policy each, DP4) 1,882,688 B → 0x081D_CA40.

## W-X280 LIM mailbox v2: `x280/fw/mailbox.h` and `x280/fw/mailbox_layout.py`

`mailbox.h` is the C layout, every offset pinned by `_Static_assert`; `mailbox_layout.py` is the Python copy x280d,
`mailbox_sim.py` and the spike harness use (`tests/x280/test_mailbox_x86.py::test_layout_matches_c` checks it against
C). §1.4.4's table plus: L1 `fw_magic` @60 (= "WBC2" once the monitor runs), `fw_ver` = (2 << 16) | rev; L36 @36-59
`n_traps`, `n_crc_fail`, `variants_built`, `n_host_gone`, `entry_flags`, `n_crc_retry`; L37 `kind`, `load_seq` (=
the LOAD's seq_req) and `desc_crc32` over bytes 0-59; L39 records (entry, SELFTEST, BENCH, CALL); L0 flags bits 8-11
variant + bit 12 "variant set"; states 5 HOST_GONE and 6 PARKED; errors 0xC0C CRC … 0xC14 DESC (magic, cmd, arg,
kernel, blob, load CRC, no code, descriptor); `obs_crc32` covers the 2,112 B of L2-L34.

## x280 host layer (WP-3.3): `x280/regs.py`, `x280/host/kmd.py`

`x280/regs.py` is the only source of X280 addresses, NoC coordinates (L2CPU0-3 at (8,3), (8,9), (8,5), (8,7)),
expected values and the write allowlist; `x280/host/kmd.Kmd` is the only device path (ctypes ioctls on the keeper
fd, UC/WC windows, `libwcio.so` 64 B non-temporal stores; `kmd.FakeDevice` for the sim and the tests). x280d's
boot (`x280d.Session.boot`) follows the host layer's order: `open`, `wait_settled`, `read_l2cpu_reset`, then
in `l2cpu_batch` contexts `lim_init`, `write_image`, `write_gp_scratch` + `write_boot_vectors`, then the PLL4 walk
`pll4_step_to(200)`, `release` (the session's one L2CPU_RESET RMW), `pll4_step_to(800)`, and `set_noc_cleanup` +
`l2pf_setup`; T-3.4 checks the dry-run op list against §5.2.

kmd calls the boot and the HUD rely on (review:x280 F1-F9):
- `write_image(tile, image) -> crc`: boot step 3, full 64 B lines at 0x0800_0000, read back line by line (16 UC
  words per call) and recorded as (crc, len) for `release`; a later write into the image drops that record.
- `write_gp_scratch(tile, code)`, `write_boot_vectors(tile, reset_pc, rnmi, rnmi_exc)`: steps 4-5, read back. The GP
  scratch is written only before release, with one exception: E1F's stage-0 -> park-stub swap once the monitor is up.
- `boot_artefacts_missing(tile, image_crc, gp_code, stage0=False) -> str`: '' when the tile may leave reset, else why
  not: vectors, GP scratch, every LIM line initialised and the image CRC, all written and read back in this session.
- `release(tiles, ledger, image_crc, *, gp_code, stage0=False)`: step 7, refused before any device write or ledger
  change unless `boot_artefacts_missing` is '' for every tile; `stage0=True` (E1F) also needs `X280_E1F_APPROVED=1`.
- `stage0_enter(tile, entry=RESET_PC)`: E1F only, the entry word after the image, after which the monitor region is
  the hart's.
- `pll4_step_to(mhz)`: before every write, after its policy check, re-reads CNTL_1, CNTL_5, USE_POSTDIV, then CNTL_0
  and PLL_LOCK last, immediately before the store (and all five after the last write); on the first change (or
  USE_POSTDIV & 0xF != 0xF, CNTL_0 & 0x1F != 0x12, PLL_LOCK bit 4 clear) it writes nothing more, leaves PLL4 to CMFW
  and raises `InterlockError` (x280d: exit 3, tiles burned after a release).
- `lim_init(lo, hi, tile, *, pattern, store='a64')`: `LIM_INIT_STORES` a64 (the boot's method A: one 64 B NT zmm store
  + sfence per line), a64nf (one sfence per call), wc2x32 (two 32 B ymm stores per line through the WC mapping, UMD's
  pattern), uc8 (eight aligned 8 B UC stores per line); whole aligned lines, pre-release only, the same per-line
  policy / ownership / batch checks for every store; the 32 B and 8 B widths are their own allowlist entries
  (`regs` LIM_INIT_WC32 / LIM_INIT_UC8, LIM only, ctx lim_init only). libwcio ABI 3 (`wcio_wc_store_lines_2x32`,
  `wcio_uc_store64`; txn ops WC32, UC_W64).
- `FakeDevice.default_chip(..., ecc='poison'|'clean'|'granule')` (`FakeEcc`): the CCACHE's LIM data ECC as a model,
  per 8 B granule, every host LIM access kmd makes counted (at most one fix and one fail per access, LOW/HIGH = the
  last counting access): 'poison' = the S1c hypothesis (every write a read-modify-write, an uncorrectable pre-read
  poisons the footprint), 'clean' = its negation, 'granule' = only 8 B stores initialise. Without `ecc` the counters
  stay 0 as before. `regs.pll4_mhz(cntl1, cntl5, use_postdiv)` is CMFW's formula (fw `pll.c:413-475`).
- `read_words(xy, addr, n)`, `read_lim_words(addr, n, tile)`, `read_lim_line(addr, tile)`: checked UC reads of n <= 16
  words inside one 64 B line in one libwcio call (libwcio ABI 2, `wcio_uc_read_words`; kmd refuses an older `.so`:
  `make -C x280/host`).
- `read_rnmi_trigger(tile)` (a hart left in the park stub), `read_status_telemetry()` (the HUD's CMFW telemetry,
  memory reads only, table walk cached), `read_ccache_ecc_counts(tile) -> {fix, fail}`.
- `lim_init_check(lo, hi, tile) -> lines`: LIM init method B's guard (server_g1's `x280_lim_write` moves the bytes, not
  this fd): lim_init's release check, then per line the write policy, ownership and the open `l2cpu_batch`, exactly
  as `lim_init` / `wc_write_lines` would; writes nothing, marks nothing initialised, logs a `CHECK` op. x280d runs it
  over the whole range before the first request, then for each 4 KiB request inside its own `l2cpu_batch`.
- `FakeDevice.default_chip(..., telemetry=None)`: the CMFW-after-boot state with the chip-panel telemetry tags at 0
  unless a unit test passes `{tag: raw u32}`; kmd holds no invented chip values (the old `FAKE_TELEMETRY` is gone).
- Ownership: once a tile is released (or its state is unknown) the whole monitor region [0x0800_0000, 0x0801_0000) is
  the X280's (`regs.lim_line_owner`), `lim_init` is refused, and `read_beu` is allowed only then (never while the
  harts are in reset).

x280d's `Session.interlocks(released_ok=False)` adds `released_tiles`, `noc_translation` (CMFW TAG_NOC_TRANSLATION;
the translated cross-read only when it is 1), `pll_lock` and `pll4_walk_ready`, and is dirty for a released tile
unless `released_ok` (attach, probe), and when PLL4 is not walk-ready (CNTL_0 & 0x1F != 0x12 or PLL_LOCK bit 4 clear),
so boot and E1F stop before their first store and probe exits 3.

PLL4 on silicon (S1c, 2026-10-25, p150a, CMFW 19.12.0.0; `runs/sessions/*_s1c*/x280_probe.1.json`): before and after
a chip reset PLL4 read CNTL_0 0x12, CNTL_1 0x0040_0102 (refdiv 2, internal postdiv 1, fbdiv 64), CNTL_5 0x0101_0101,
USE_POSTDIV 0xF, PLL_LOCK 0x1F. The bh_arc source writes postdiv 0 (fw `pll.c:226`), but CMFW ignores the internal
postdiv (`pll.c:456-461`; L2CPUCLK = 50 * fbdiv / (refdiv * (CNTL_5 postdiv + 1)) = 800 MHz) and its clock driver
programs postdiv 1 from the devicetree on every L2CPU power-on, i.e. the keeper fd's power message
(`tt_blackhole_smc.dtsi:296-298`, `clock_control_tt_bh.c:648-650, 193-194`). So `regs.PLL4_CNTL_1_DEFAULT` is
0x0040_0102, compared exactly by `kmd.pll4_is_default` (`pll4_default_why` gives the reason); 0x0040_0002 or any
other CNTL_1 is dirty ("unexplained"), and no walk starts from it. Both walks keep refdiv and the postdiv byte as read
at their start (tt-bh-linux `clock.py` semantics) and change only fbdiv and the CNTL_5 postdivs, one step per write:
the 200 MHz state is CNTL_1 0x0080_0102 / CNTL_5 0x0F0F_0F0F, the 800 MHz end state 0x0040_0102 / 0x0101_0101 again,
VCO 1600-3200 MHz and L2CPUCLK <= 800 MHz at every step (`regs._self_check`). `FakeDevice.default_chip()` models
these silicon values.

## x280d (WP-3.4): `x280/x280d.py`, W-WBC on `:5558`

Subcommands: probe, boot, serve, attach, ping, e0, lim-init, lim-check, lim-ecc, selftest, load, call, park, e2, lim-bench,
parity, harts (`adopt|status|ping [--mask 0xe]`), bench (`--host-only`, `--rtt [--cpus LIST] [--spin]`, `--matrix`,
`--mh [--mh-plan full|quick] [--mh-repeats 3] [--mh-window-us 500] [--mh-bench-iters 200]`), hbrate, powertest, inject,
status, dump;
all accept `--dry-run`
(sim backend plus every NoC op and libwcio transaction on stdout) and `--via URL` (run in the serving daemon, which
holds the only fd, through the `x280d_admin` endpoint). `dump --via URL [--out DIR|file.npy]` (T-3.7; without
`--via` exit 2): the serving daemon writes its tick ring, oldest row first, as `x280d_ticks.npy` in DIR (default its
`--log-dir`) and replies `{ring (the path), rows, calls (W-WBC calls served), wrapped (calls > 65,536: the oldest are
gone), t_first_ns, t_last_ns}`. Exit codes: 0 ok, 1 failed, 2 usage or missing build artefact, 3 X280 dirty, 4 probe
error / tile unusable. `--cpus` (serve) and `--sim-cpus` take taskset lists (`5`, `5,21`, `8-9`; `--sim-cpus` is one
CPU per hart in tile order); one that is not a list, or, with `CHIP_SAFE_CPUS` set, one outside it, exits 2 before
anything starts. `bench --rtt` (T-3.5's client): `--cpus LIST` places it as rt_sim (its calling thread on the
lowest CPU, the zmq I/O thread on the highest: `2,3`), `--spin` busy-polls each reply (both exit 2 without
`--rtt`); its reply has `checks` = PLAN T-3.5's x280d-side gates only (x280d p99 <= compute p99 + 0.5 ms and <= 2 /
9 ms, host overhead p99 <= 0.3 ms with data on the device, the sim exempt; 0 timeouts and errors), `pass` = all of
them, and `reported` = the client RTT row (`client_rtt_p99_le_limit`, not gated) and the client placement.
`powertest --watch-s S` ends each watch with a PLL4 read (`pll4_default_after_each` gets one more entry).
`probe --tile LIST` with more than one tile, a tile outside `regs.WRITABLE_TILES`, or `--via URL` (S5a,
`jobs/s5a_probe.sh`): the read-only probe of those L2CPU tiles in that order (`Session.tiles_probe`), standalone on its
own keeper fd or on a `serve --no-release --probe-tiles LIST` daemon (admin `tiles_probe`); nothing is written. ARC
first; every tile is skipped, never read, in CMFW cable-fault mode (ERROR_STATUS0 = SCRATCH_RAM[4] bit 1: all tiles
outside column 15 clock-gated), and so is a tile not enabled, held in reset (its L2CPU_RESET[3:0] bit clear), at 0 MHz,
released, or any tile with PLL4 off its defaults; the BEUs are not read (harts in reset, review F7). In a probe session
every probe tile, the session's own included, refuses every write. Reply (schema 1): {probe "s5a", schema, backend, tiles_order, l2cpu_reset,
pll4, pll4_default, pll4_why, pll_lock, pll4_walk_ready, l2cpuclk_mhz, error_status0, cable_fault, l2cpuclk_telemetry_mhz, timer_heartbeat,
enabled_l2cpu, noc_translation, beu, tiles {t: {xy, enabled, released, l2cpuclk_mhz, probed, reads, then skipped, or
wayenable, ccache_config, ccache_config_translated + cross_read_equal (or cross_read), ecc, hart_status, niu0
{noc_node_id, noc_endpoint_id, niu_cfg_0, node, endpoint, translation_enable, tile_clock_disable}, tlb {small (224 x 4
words), large (32 x 3), small/large_distinct, small/large_nonzero}, error}}, l2cpu_reset_after, pll4_after, checks
{l2cpu_reset, pll4_default, enabled_l2cpu, wayenable, complete, no_cable_fault} (only these gate), reasons, warnings, dirty, verdict
clean|dirty, attachable false, reads, harvested}; exit 0 clean, 3 dirty, 4 only ENABLED_L2CPU != 0xF, 1 no report, 2 a
bad tile list. `serve --probe-tiles LIST` (with `--no-release` only, else exit 2): per extra tile a read-only CCACHE0
window (+ the translated one), and every xp prefix reaches NIU #0 (0x15_7000 B, transient read-only mappings).
serve: `--backend device|sim`, `--no-release` (S1c),
`--boot`, `--attach`, `--stage0` (E1F, needs X280_E1F_APPROVED=1; every M3 session's boot since 2026-10-25),
`--harts 4` (S3b, the 4-hart bench: with `--stage0` and X280_MH_APPROVED=1 only, L2CPU3 only; before the release harts
1-3's RNMI handler = 0x0800_0000 (crt0's secondary path, x280/fw/mh.c) and RNMI exception handler = 0x0800_0200
(`regs.MH_SEC_RNMI_EXC`), hart 0 unchanged; without it the boot is byte-identical to the default),
`--early`, `--slow` (12 ms deadline, DP11),
`--dtype fp16|fp32`, `--tile 3[,2]`, `--sim-cpus`, `--cpus`, `--status-pub`, `--log-dir` (default
`$CHIP_SESSION_DIR`). `--sim-ecc poison|clean|granule` (sim only, no sim hart: no-release, e0, lim-*): the sim chip
gets `kmd.FakeEcc`; with the device backend or a sim hart exit 2. `lim-init --store a64|a64nf|wc2x32|uc8` (method A's
host stores) and lim-init / lim-check report `ecc_delta` (counts, last fix/fail address and where it lies),
`ecc_events_per_line` / `ecc_events_per_word_read`; lim-check adds `checks` and `failed_checks` (its gate unchanged).
`lim-ecc [--plan a64,a64nf,wc2x32,uc8,a64+uc8] [--lines 1024] [--base 0x08011000] [--probe-lines 4] [--partial 16]
[--seed 11]` (S1d, no-release only): per plan item a never-written region, each store through `kmd.lim_init`, the
first probe lines written and read one access at a time with the counters read after each, then a full 4 B re-read;
after the item's last store partial 4 B writes and a second re-read. Reply: `variants[]` {name, lo, hi, steps[]
{store, probe_write, probe_read, probe_partial ([fix, fail] of each partial write and of its read-back), phases
{write, read1, partial, read2: counter deltas + event addresses}, derived {write_events_per_line,
accesses_per_line_min, probe_event_offsets, read1/read2_events_per_word, partial_fail_events}, mismatches, clean},
clean, full_lim_write_s}, `clean_variants`, `integrity` (L2CPU_RESET, WAYENABLE unchanged) = `pass`. W-WBC `info`: `backend` `x280:l2cpu<tiles>:h0@<MHz>MHz:<variant>-<dtype>` or
`x280-sim:<variant>-<dtype>` (rtloop: X280 / X280_SIM), `proto` 2, `policies`, `dtype`, `variant`, `parity_tol`
(fp16 1e-2, fp32 5e-5; ZmqWbc honours it), `policy_tile`, `compute_us_src` (mcycle, or host_turnaround in the sim),
`degraded`, `status_pub`. Replies add `x280d_us`, `mailbox_us`, `cycles`, `mb_seq` to the §1.4.3 fields; any
failure is an error reply (the client falls back to ORT). `--early`: `gear_wbc_dispatch` / `gear_wbc_collect`
(WP-3.8b). x280d_admin commands: status, info, park, inject, dump (the tick ring) and the subcommands above (`harts`:
harts_adopt, mh_status, mh_ping; `bench --mh`: bench_mh; `probe --tile LIST --via`: tiles_probe). `harts adopt` (one
pulse of RNMI bits 1-3, refused by kmd
without this session's --harts 4 vectors, the E1F 'stub' state and L36.mh's multi-hart bit, never with bit 0) replies
{pass, adopted, lost, harts {state, entries, mncause, mnepc, mode, mip, heartbeat_moving, beu}, ping {done, wake_ticks,
total_ticks, ready}, hart_status_before/after (raw 0x2001_0400)}; `bench --mh` replies {configs [{name, kernel, kernel0,
mask, pattern, kbit, runs [per-hart bytes/cycle, cycles/access, cycles/tick; agg accesses/bytes per mtime tick, GB/s,
skews, flags], medians}], interference {bench_idle, bench_loaded, slowdown_pct, background}, events_before/after (BEU
accrued of harts 0-3, CCACHE ECC counts), secondaries_after, parity, derived {scaling (S_N per kernel/pattern),
S4_K9, compute_bound, bank, p4_sweep_agg, anchors, t4_estimate_cycles, split_worth_it, verdict}, abort, pass}; a
--harts 4 daemon's `status` adds the MH lines per tile. Logs:
`x280d_ticks.npy` (ring of the last 65,536 calls, `TICK_LOG_DTYPE`: t_recv_ns, tick, seq, mb_seq, tile, policy,
status 0 ok / 1 timeout / 2 fault / 3 stall / 4 interlock / 5 bad / 6 degraded, early, x280d_us, mb_us, compute_us,
cycles, hb_age_us; written at exit and on `dump`) and `x280d_status.jsonl` (every 10 s), both in `--log-dir`.
