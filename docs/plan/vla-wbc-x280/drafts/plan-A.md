# G1 VLA + WBC, asynchronous, on one chip: plan A (interfaces first, real-robot fidelity)

Date 2026-10-24. Drafter A. Evidence base: `docs/plan/vla-wbc-x280/research/understand.json` (read in full) plus
source checks made for this draft (cited file:line). Nothing in any repository was changed, the Blackhole was not
opened, and the GR00T server on 127.0.0.1:5555 (pid 3134179) was not touched.

**Thesis.** First fix a robot-shaped process architecture and its message contracts, copied from the gr00t_wbc
real-robot stack. After that, every milestone is **one config-key change behind an unchanged interface**:

| Milestone | The only key that changes | Interface it sits behind (frozen in M1) |
|---|---|---|
| M1 (5090) | none (baseline: `vla.endpoint=tcp://127.0.0.1:5555`, `wbc.backend=gpu_lbstep`) | — |
| M2 (VLA on Tensix) | `vla.endpoint` → `tcp://127.0.0.1:5557` | PolicyServer `get_action` protocol |
| M3 (WBC on X280) | `wbc.backend` → `x280` | `LbStep` (72 f32 in, 15 f32 out, stateful) |
| M4 (optional) | `controller.placement` → `x280` | `LowState`/`LowCmd` + `ActionChunk` |

Each new backend must pass a **conformance suite** for its interface before integration (§2.0). The conformance suite
is the main deliverable of this draft.

### Conventions and citation shorthand

- Paths: `R` = `~/experiments/gr00t/Isaac-GR00T-n1.6`. `W` = `R/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc`.
  `T` = `~/experiments/gr00t/tt-metal/models/experimental/gr00t`. `P` = `~/experiments/gr00t/publish`.
  `G` = `~/experiments/gr00t/g1rt` (**new**, outside every git repo; `~/experiments/gr00t` itself is not a repo).
  `S` = `/tmp/claude-1000/-home-deepgadget-isaacsim/bfbe65f0-1b38-425e-84fc-4937eb211a38/scratchpad`.
- Interpreters: `$SIMPY` = `R/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python` (client
  venv: MuJoCo, robosuite, pyzmq 27.2, msgpack). `$SRVPY` = `R/.venv/bin/python` (server venv: torch+CUDA, the running
  server's interpreter). `$TTENV` = `source P/worktree-mk-env.sh` (tt-metal python_env with the frozen snapshot
  009de892). `$LUPY` = `~/.tenstorrent-venv/bin/python` (pyluwen 0.8.5). `G/env.sh` exports all four.
- Research citations: `[sim:N]`, `[serve:…]`, `[tt:N]`, `[x280:…]`, `[async:…]` = the `understand.json` section and
  the finding whose topic starts with that number or keyword. Measurements quoted here are from those sections.
- Time: tick `k` (50 Hz, `dt` = 0.02 s) is the **only** control time base. `t_sim = k·dt`. `CLOCK_MONOTONIC` ns is used
  only for latency accounting. A chunk row `i` targets tick `k_obs + i`.

---

## 0. Goal, scope, non-goals, decisions already made

**Goal.** A G1 locomanipulation demo (`gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`, apple to plate) in which
GR00T N1.6 3B (VLA) and the Decoupled-WBC lower-body RL policy (`gear_wbc` Balance/Walk MLPs, 516→15) run
**concurrently and asynchronously like a real robot**. The WBC runs on every 50 Hz tick and never waits for the VLA.
The VLA runs in the background and produces 30-step action chunks. The end state runs on one Blackhole p150a: the VLA
on Tensix and the WBC on the on-die SiFive X280.

**In scope.**
- The robot-like process split and frozen message schemas (§1).
- M1: the async loop on the RTX 5090. The sim runs on the host CPU; the VLA and the WBC are served from the GPU.
- M2: the VLA on Tensix (existing tt-metal port).
- M3: the WBC on the X280 (bare metal, in LIM).
- M4, optional: on-chip VLA→WBC handoff through device DRAM, the whole controller on the X280, and RTC v2.
- The evaluation protocol, including the synchronous baselines.

**Non-goals.**
- Real hardware or a ROS/DDS install (rclpy, unitree_sdk2py and cyclonedds are absent [sim:3]). We copy the structure,
  not the code.
- Linux on the X280.
- Retraining or fine-tuning the WBC.
- N1.5/N1.7.
- Multi-env vectorised RT runs (VL mode may run several robot processes).
- TensorRT on the 5090.
- Changing the running 5555 server.

**Decisions already made (user).**

| # | Decision | Consequence in this plan |
|---|---|---|
| U1 | VLA on Blackhole Tensix with the existing port (`T`, frozen snapshot `P/worktree-mk`, 009de892) | M2 runs from the snapshot env, never the live tree [tt:Working tree] |
| U2 | WBC on Blackhole's X280 cores. Treat it as technically possible; the tt-bh-linux "not concurrent with tt-metal" note is outdated/strategic | M3 assumes coexistence. §5 still orders the experiments so that a failure costs at most one `tt-smi -r`, scheduled at a session boundary |
| U3 | Real-robot-like async is mandatory | Wall-clock RT mode is the demo. The VL mode exists only for statistics and equivalence |
| U4 | M1 = async loop validated on the 5090: sim on the host CPU, **VLA + WBC from the GPU server** | M1's official config serves the WBC from the 5090 on its **own** process/port (5556), never on the VLA socket |
| U5 | Order: M1 → M2 (TT VLA) → M3 (X280 WBC) → M4 optional | Each step swaps exactly one backend (table above) |

**Decisions this draft commits to (A-decisions).**

| # | Decision | Basis |
|---|---|---|
| A1 | Five roles in separate OS processes: **robot** (sim), **controller** (50 Hz), **vla_proxy**, **VLA service**, **WBC service**. No threads share a GIL across roles | gr00t_wbc splits ControlPolicy / Teleop / Sim into processes (`W/control/main/teleop/run_g1_control_loop.py:87-131`) [sim:3]. A single REP server serialises the WBC behind the VLA [serve:Q1] |
| A2 | The robot process is the **clock master**. It publishes `LowState_k` at `T_k` and steps physics with `LowCmd_k`, or holds the last command if the deadline passes (motor-like) | Real G1: the controller reads rt/lowstate and writes rt/lowcmd [sim:3] |
| A3 | Fixed binary layouts (little-endian, 40 B header) for every 50 Hz message, generated from one Python table into a C header used by the X280 | Mailbox and C port need fixed layouts [x280: rec. 4] |
| A4 | The WBC boundary is the **stateful 72→15 `LbStep`**: the backend owns the 6×86 history, the last action, the 1-tick command lag and Balance/Walk selection | [sim:4 Proposed X280 interface]. This is what the X280 needs, so M1 freezes it |
| A5 | Chunk scheduling (trigger + splice) lives in the controller, as a pure function. The proxy only does inference | LeRobot RobotClient owns the action queue and decides when to send observations [async: LeRobot]. The logic is portable to C for M4 |
| A6 | Upper body: tick-indexed identity pass-through (no InterpolationPolicy) in v1 | The sim eval path is identity (`W/control/utils/n1_utils.py:65`). Avoids the monotonic/sim clock mix [sim:6 Time base] |
| A7 | No repository edits are needed for M1–M3. All new code is in `G`. TT G1 support is injected at runtime | [tt: rec. 'g1' LayoutConfig, runtime injection proven on CPU] |
| A8 | A **slaved shadow** `ort_core` WBC runs in the controller every tick next to any remote backend. It gives per-tick parity and a warm fail-over | Cheap: ORT MLP 0.08 ms paced [serve: Measured idle] |
| A9 | One **device-session owner** script (under `bin/with-device.sh`) launches every Blackhole process (TT VLA server, `x280d`). All device tests are clients of those processes | Only one flock holder. The probe or reset can only happen at session start, before any hart release [x280: Hart reset is once-only] |

---

## 1. Target architecture

### 1.1 Processes, ownership, rates

```
            +----------------------------- host (7950X) ------------------------------+
            |                                                                          |
 VL/RT cfg  |  P1 robot_sim (CUDA-free, EGL)        P2 controller (CUDA-free)          |
  ------->  |  MuJoCo 500 Hz, joint PD, cameras  <--PAIR-->  sched + WBC front-end     |
            |  episode control, safety monitor   LowState_k / CaptureReq_k / LowCmd_k |
            |        |  CameraFrame_k (PUSH)                 |  InferRequest (PUSH)     |
            |        v                                       v                          |
            |  P3 vla_proxy (CUDA-free) <------- ActionChunk (PUSH) ------+            |
            |        | REQ (PolicyServer msgpack)            | REQ LbStep (binary)      |
            +--------|---------------------------------------|---------------------------+
                     v                                       v
     P4 VLA service: 5090 GR00T server :5555 (M1)   P5 WBC service: wbc_gpu_server :5556 (M1/M2)
                     TT VLA server :5557 (M2+)                    x280d ipc (M3)  [+ shadow ort_core in P2]
```

| Proc | Code (new unless noted) | Interpreter | CPU core | Device | Rate | Sole owner of |
|---|---|---|---|---|---|---|
| P1 robot_sim | `G/robot_sim.py`, `G/viewer.py` | `$SIMPY`, `CUDA_VISIBLE_DEVICES=""`, `MUJOCO_GL=egl` | 2 | EGL only (no CUDA context: it corrupts EGL renders) | tick 50 Hz, physics 10×2 ms, PD 500 Hz | MuJoCo model/data, cameras, seeds, episode end, JointSafetyMonitor |
| P2 controller | `G/controller.py`, `G/sched.py`, `G/wbc/*` | `$SIMPY`, `CUDA_VISIBLE_DEVICES=""` | 3 | none | event-driven on `LowState`, 50 Hz | chunk/schedule state, WBC front-end (FK, packing), shadow WBC |
| P3 vla_proxy | `G/vla_proxy.py` | `$SIMPY`, `CUDA_VISIBLE_DEVICES=""` | 4 | none | per request (≈3 Hz) | the VLA REQ socket, observation builder |
| P4a GR00T server | existing, untouched (`R/gr00t/eval/run_gr00t_server.py`) | `$SRVPY` | server's | 5090 | per call | VLA on 5090 (5555) |
| P4b TT VLA server (M2) | `G/services/tt_vla_server.py` | `$TTENV` | 6–7 | Tensix + DRAM (UMD) | per call | the `Gr00tTT` model and all ttnn calls (one thread) |
| P5a wbc_gpu_server (M1) | `G/services/wbc_gpu_server.py` | `$SRVPY` | 5 | 5090, own CUDA context, highest-priority stream | 50 Hz | GPU WBC history and graphs |
| P5b x280d (M3) | `G/x280/x280d.py` | `$LUPY` | 5 | L2CPU tile(s), LIM only, 2 MiB KMD window, keeper fd | 50 Hz | L2CPU lifecycle, mailbox, firmware loader |

Ownership rules:
- Only P1 touches MuJoCo state.
- Only the active WBC backend owns the WBC history. The shadow keeps its own copy, slaved to the primary (A8).
- Only P3 talks to the VLA.
- Only the device-session children open `/dev/tenstorrent/0` (A9). Robot, controller and proxy never open it, in any
  milestone.
- New sockets never use port 5555, which the running server and gr00t_wbc's ImagePublish defaults both use [sim:6 Port
  collision]. Host-local links use `ipc:///tmp/g1rt/<run_id>/<name>`.

### 1.2 Real-robot mapping (why this shape)

| g1rt element | gr00t_wbc real-robot stack (template) | Unitree G1 |
|---|---|---|
| P1 robot_sim | `RoboCasaG1EnvServer` / `run_sim_loop.py` (sim plays the robot over DDS) [sim:3] | robot + motor drivers |
| `LowState` | `G1Env.observe()` (rt/lowstate, rt/secondary_imu, rt/dex3/*/state) | rt/lowstate |
| `LowCmd` | `env.queue_action` → rt/lowcmd (q, dq=0, tau=0, kp, kd) | rt/lowcmd, joint PD on the motors |
| `CameraFrame` | `realsense/color/image_raw` (`W/control/main/constants.py:1`) | head camera |
| P2 controller | `run_g1_control_loop.py` ControlPolicy at 50 Hz (`:87-131`) | onboard PC |
| `ActionChunk` | `ControlPolicy/upper_body_pose` goal (`constants.py:3`) with `target_time` list; depth-1, latest-wins (`W/control/utils/ros_utils.py:91,110-123`) | — |
| P3 vla_proxy | `run_teleop_policy_loop.py` (separate 20 Hz goal producer) [sim:3] | — |
| starvation / timeout | `G1DecoupledWholeBodyPolicy` 1 s goal timeout (`W/control/policy/g1_decoupled_whole_body_policy.py:91-110`) | — |

Documented fidelity deviations:
- (a) The robot waits for `LowCmd_k` until a deadline. A real robot applies its last command at 500 Hz; with about 1 ms
  of controller latency the difference is negligible, and waiting avoids a fake one-tick actuation delay that the sync
  baseline does not have. `robot.wait_cmd: none` is offered as a stress mode.
- (b) The camera is triggered (`CaptureReq`) rather than free-running, which makes runs deterministic.
  `robot.camera_hz: 30` is a demo option.
- (c) Sim PD gains differ from the real robot's MOTOR_KP (left knee 300 vs 200) [sim:1 PD].
- (d) State has no noise or latency.
- (e) Evaluation info (success, pelvis height) travels on a separate topic that the controller never reads.

### 1.3 Per-tick data flow (RT mode, M1 numbers)

`T_k = T_0 + k·20 ms` (absolute deadlines, CLOCK_MONOTONIC).

```
T_k +0.00  P1  publish LowState_k (state after physics step k-1)            ~0.03 ms IPC
     +0.05  P2  recv; drain chunk inbox -> sched.on_arrival(); sched.trigger(k)?
                 if trigger: send CaptureReq_k (same PAIR, before LowCmd) + InferRequest(seq, k_obs=k)
                 row35 = sched.row(k)  (hold/starvation/crossfade applied)
                 torso_rpy = FK(row.waist); LbInput_k -> backend.step (5556: p99 0.92 ms measured [serve:Q3/Q4])
                 shadow ort_core.step(LbInput_k) (0.08 ms); LowCmd_k = [q15 | arms+hands from row]
     +~1.0  P1  (if CaptureReq_k) render ego 640x480 (0.28 ms mean, 5.8 ms max [sim:5]) -> CameraFrame_k
                 recv LowCmd_k (deadline T_k + 8 ms; else hold previous, flag CMD_MISSED)
                 env.step({"q": q_des}): 10 substeps x 2 ms, PD 500 Hz (~4.5 ms; full tick 5.26 ms with 2 cams [sim:5])
                 publish Event EVAL_TICK (success, pelvis z)
     +~6.5  P1  sleep to T_{k+1}; if more than one tick late: re-anchor T_0, log SLIP (no catch-up burst)
async  P3  on InferRequest + CameraFrame_k: build 9-key obs (0.11 ms serialise [sim:2]) -> REQ get_action
                 (5090 trimmed p50 58.8 ms [serve: VLA latency]) -> ActionChunk -> P2; visible at k+3..k+4
```

Slack per tick is about 13 ms. In-loop video encoding (17–87 ms [sim:5 Things that do break]) is banned from P1.
Frames are logged and rendered offline, or the onscreen viewer is used (§2 T-1.9).

**VL (virtual-latency) mode.** Same processes, no pacing, `robot.wait_cmd: block`. `InferRequest.k_avail = k_obs +
d_inj`. The controller treats a received chunk as invisible before `k_avail`, and blocks at `k_avail` until it
arrives. With `d_inj = 0` this blocks inside tick `k_obs`. That works because `CaptureReq_k` precedes `LowCmd_k` on
the ordered PAIR socket, so P1 renders frame `k` while it is still in tick `k`. VL is deterministic given the scene
seed and a deterministic VLA (replay or seeded), so `d_inj=0, s=20` must reproduce the sync rollout exactly (T-1.1).

### 1.4 Message schemas (frozen in M1, `G/msgs.py` → generated `G/c/g1rt_msgs.h`)

All values are little-endian with natural alignment. f64 is used on the sim wire for state and commands so that T-1.1
can be bit-exact. Nothing downstream depends on f64, and a hardware driver would downcast.

**Header (40 B, every message)**

| off | field | type | meaning |
|---:|---|---|---|
| 0 | magic | u32 | 0x54523147 ("G1RT") |
| 4 | type | u16 | 1 LowState, 2 LowCmd, 3 CaptureReq, 4 CameraFrame, 5 InferRequest, 6 ActionChunk, 7 LbInput, 8 LbOutput, 9 Event |
| 6 | version | u16 | 1 |
| 8 | episode | u32 | generation; bumps on reset or instruction change |
| 12 | seq | u32 | per-(sender, type) counter |
| 16 | tick | i64 | control tick `k` in the episode (0 = first tick after reset) |
| 24 | t_mono_ns | i64 | sender CLOCK_MONOTONIC at send |
| 32 | flags | u32 | per type (below) |
| 36 | payload_bytes | u32 | |

**Joint index tables (frozen; verified against `robot_model` by T-1.0).**
- `J43` (robot_model order) [sim:1 inputs]: left_leg 0–5, right_leg 6–11, waist 12–14 (yaw, roll, pitch),
  left_arm 15–21, left_hand 22–28, right_arm 29–35, right_hand 36–42. Groups: `lower_body` = 0–14,
  `upper_body` = 12–42 (31 joints), `body` = 29 joints with no hands, expected to be 0–21 ∪ 29–35.
- `A35` (chunk row order): left_arm [0:7], right_arm [7:14], left_hand [14:21], right_hand [21:28], waist [28:31],
  base_height_command [31], navigate_command [32:35] (vx m/s, vy m/s, wz rad/s).
- VLA state keys are slices of `q` (`W/control/utils/n1_utils.py:152-173`): left_leg = q[0:6], right_leg = q[6:12],
  waist = q[12:15], left_arm = q[15:22], left_hand = q[22:29], right_arm = q[29:36], right_hand = q[36:43].

| Type (dir, rate, transport) | Payload layout (offset: field dtype[shape] unit) | Flags |
|---|---|---|
| 1 **LowState** (P1→P2 PAIR; mirrored on P1 PUB for loggers; 50 Hz; 856 B) | 0: `q` f64[43] rad J43 · 344: `dq` f64[43] rad/s · 688: `base_pose` f64[7] (pelvis xyz m world, quat wxyz) · 744: `base_vel` f64[6] (lin xyz m/s world, ang xyz rad/s pelvis frame = MuJoCo free-joint qvel) · 792: `torso_imu` f64[7] (quat wxyz, gyro xyz; logged, unused by the WBC) · 848: `t_sim` f64 s | b0 FIRST_TICK, b1 PREV_CMD_MISSED, b2 SAFETY_LATCHED |
| 2 **LowCmd** (P2→P1 PAIR; 50 Hz; 360 B; `tick` = answered LowState tick) | 0: `q_des` f64[43] rad J43 (lower 15 from the WBC; waist overridden by the WBC; arms and hands from the row) · 344: `policy` u32 (0 Balance, 1 Walk) · 348: `ctrl_us` f32 · 352: `chunk_seq` u32 · 356: `cursor_row` i32 | b0 HOLD, b1 END_EPISODE_REQ, b2 STARVED, b3 WBC_FAILOVER |
| 3 **CaptureReq** (P2→P1 PAIR, sent before LowCmd_k; 12 B) | 0: `cam_id` u16 (0 = robot0_oak_egoview) · 2: `w` u16=640 · 4: `h` u16=480 · 6: pad · 8: `req_seq` u32 | — |
| 4 **CameraFrame** (P1→P3 PUSH; per request; 921,960 B) | 0: `req_seq` u32 · 4: `cam_id` u16 · 6: `h` u16 · 8: `w` u16 · 10: `c` u16=3 · 12: `render_ms` f32 · 16: `q` f64[43] (snapshot from the same tick) · 360: `rgb` u8[480,640,3], HWC, upright (`render(...)[::-1]`) | — |
| 5 **InferRequest** (P2→P3 PUSH; ≈3 Hz; 40 B v1, +4200 B v2) | 0: `req_seq` u32 · 8: `k_obs` i64 · 16: `k_avail` i64 (−1 in RT) · 24: `d_hat` u16 ticks · 26: `s_used` u16 ticks · 28: `noise_seed` u32 (0 = server default) · 32: `rtc_len` u32 (0 in v1) · 40: `rtc_prefix` f32[30,35] A35 absolute (v2 only) | b0 RTC |
| 6 **ActionChunk** (P3→P2 PUSH; 4256 B) | 0: `req_seq` u32 · 4: `status` u32 (0 ok, 1 server_error, 2 timeout, 3 bad_shape) · 8: `k_obs` i64 · 16: `k_avail` i64 · 24: `t_req_tx_ns` i64 · 32: `t_resp_rx_ns` i64 · 40: `client_ser_ms` f32 · 44: `server_ms` f32 (NaN if not reported) · 48: `H` u16=30 · 50: `A` u16=35 · 56: `rows` f32[30,35] A35, **absolute** targets; row i ↔ tick k_obs+i | — |
| 7 **LbInput** (P2→WBC backend; 50 Hz; 296 B) | 0: `x` f32[72] = [0:29] q_body rad · [29:58] dq_body rad/s · [58:62] pelvis quat wxyz · [62:65] pelvis ang vel xyz rad/s (body frame) · [65:68] nav cmd (vx, vy m/s; wz rad/s) · [68] base height cmd m · [69:72] torso rpy cmd rad · 288: `policy_override` u32 (0xFF auto) | b0 RESET_HISTORY (tick 0) |
| 8 **LbOutput** (backend→P2; 136 B; `tick` echoed) | 0: `q15` f32[15] rad (J43 0–14: absolute targets = a·0.25 + default_angles) · 60: `a15` f32[15] raw MLP output · 120: `policy` u32 · 124: `compute_ns` u32 (backend side; X280 = cycles at 800 MHz) · 128: `status` u32 (0 ok, 1 tick_gap, 2 not_reset, 3 trap) | — |
| 9 **Event** (P1 PUB; loggers and P3; the controller reads only START/END) | `kind` u32: 1 EPISODE_START {seed u32, instruction char[256] utf-8}, 2 EPISODE_END {reason u32: 1 success, 2 timeout, 3 fall, 4 safety, 5 starved, 6 abort; n_ticks u32; first_success_tick i32}, 3 EVAL_TICK {success u8, pelvis_z f32, reward f32} | — |

`LbInput.x` is the 72-float split q_body 29 + dq_body 29 + quat 4 + gyro 3 + nav 3 + height 1 + rpy 3
[sim:4 Proposed X280 interface]. T-1.0 asserts these offsets against `msgs.py`.

**`LbStep` semantics** (normative; the reference is `W/control/policy/g1_gear_wbc_policy.py:102-275` and
`g1_decoupled_whole_body_policy.py:111-147`). At tick k:
1. If RESET_HISTORY: zero the 6×86 history, set `a_prev` = 0, and set (cmd, height, rpy)_prev to
   cmd_init [0,0,0], 0.74, [0,0,0] (`W/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml:37-39`).
2. Push the frame built from (q, dq, quat, gyro of tick k; (cmd, height, rpy)_prev; `a_prev`), using
   single_obs = [cmd·[2,2,0.5], height, rpy, ω·0.5, gravity_in_body(quat), (q_body − default_pad)·1.0, dq_body·0.05,
   a_prev] (`g1_gear_wbc_policy.py:144-163`; `yaml:21-29`). This reproduces the reference **1-tick command lag**
   [sim:4 WBC per-tick].
3. (cmd, height, rpy)_prev ← (nav, height, rpy) from `LbInput`.
4. policy = Balance if ‖nav‖ < 0.05, else Walk (`g1_gear_wbc_policy.py:257`). This uses the new cmd.
5. a = MLP(history); `a_prev` ← a; q15 = a·0.25 + default_angles.

Rules for backends:
- A backend must be called exactly once per tick (the gait dt is hard-coded to 0.02 [sim:4]).
- A tick that is not `last+1` (without RESET) returns status=tick_gap. The backend still computes.
- The controller computes torso_rpy from the commanded waist with the 3-joint FK
  (`g1_decoupled_whole_body_policy.py:124-134`). FK moves into the backend only in M4.

**VLA service contract** (external, unchanged across M1–M4: Isaac-GR00T PolicyServer, `R/gr00t/policy/server_client.py:15-46,54-149`).
- Request: msgpack `{"endpoint": "get_action", "data": {"observation": O, "options": X}}`, with ndarrays encoded as
  `{"__ndarray_class__": True, "as_npy": …}`.
- `O` holds exactly 9 keys:
  - `video.ego_view` u8 (1,1,480,640,3);
  - `state.{left_leg 6, right_leg 6, waist 3, left_arm 7, right_arm 7, left_hand 7, right_hand 7}` f32 (1,1,D);
  - `annotation.human.task_description` (str,).
  The server runs strict=True and checks dtypes and ndim [sim:2]. The trimmed payload is 0.92 MB vs 9.22 MB for the
  full dict, and the call takes 58.8 vs 81.4 ms p50 [serve: VLA latency].
- `X` (options): `obs_tick` int and `t_obs_ns` int (Gr00tPolicy ignores both, `R/gr00t/policy/gr00t_policy.py:320,580,613`);
  `noise_seed` int (honoured by the TT and replay services only); `noise` f32 (50,128) (debug, TT only); `rtc` dict
  (v2 only).
- Response: `[A, info]`, where A = `action.{left_arm, right_arm, left_hand, right_hand, waist, base_height_command,
  navigate_command}` f32 (1,30,D), absolute (arms already converted from relative to absolute against the obs-time
  state [sim:2]). `info` may carry `server_recv_ns, encode_ms, infer_ms, decode_ms`.
- Required endpoints: `ping`, `get_modality_config`, `get_action`, `reset`.

**Transport table**

| Link | ZMQ pattern | Endpoint | Semantics |
|---|---|---|---|
| P1↔P2 | PAIR | `ipc:///tmp/g1rt/<run>/rc` | ordered, lossless, lockstep with deadline |
| P1→P3 | PUSH/PULL, HWM 2 | `ipc://…/cam` | frames by request |
| P2→P3 | PUSH/PULL, HWM 2 | `ipc://…/req` | at most one request in flight (enforced by P2) |
| P3→P2 | PUSH/PULL, HWM 8 | `ipc://…/chunk` | P2 drains every tick. The newest valid `req_seq` wins (latest-wins mailbox, as in ros_utils depth 1) |
| P1→* | PUB/SUB | `ipc://…/ev` | events + LowState mirror; lossy allowed |
| P3→P4 | REQ/REP | `tcp://127.0.0.1:5555` (M1) / `:5557` (M2+) / `:5558` (replay) | RCVTIMEO 1000 ms. On timeout, recreate the socket (lazy pirate). The stock PolicyClient never applies its timeout (`server_client.py:162,210-211`) |
| P2→P5 | REQ/REP | `tcp://127.0.0.1:5556` (GPU) / `ipc://…/x280d` | RCVTIMEO 5 ms, lazy pirate. On timeout, use the shadow output for that tick. After 3 consecutive timeouts, fail over for the rest of the episode |

### 1.5 Stage mapping (one backend swap per milestone)

| Component | M1 (5090) | M2 (VLA on Tensix) | M3 (WBC on X280) | M4 (optional) |
|---|---|---|---|---|
| P1 robot_sim | host, EGL | same | same | same |
| P2 controller | host: sched + FK + packing | same | same | relay only; sched + FK + packing in C on X280 |
| WBC backend | `gpu_lbstep` → P5a :5556 (official, U4); `ort_core` = reference and shadow | **unchanged** `gpu_lbstep` | **`x280`** → P5b (LIM-resident C core), shadow `ort_core` | X280 runs the whole controller |
| P3 vla_proxy | host | same | same | same; also forwards the chunk descriptor |
| VLA service | GR00T server :5555 (5090, untouched) | **TT VLA server :5557** | same | + chunk tensor in DRAM read by the X280 |
| Device owner | 5090: server pid 3134179 + P5a | Blackhole: `device_session.sh` → P4b | + P5b in the same session | same |
| Config diff | — | `vla.endpoint` | `wbc.backend` | `controller.placement` |

Keeping `gpu_lbstep` through M2 means that M2 changes only the VLA. A GPU-less single-chip M2 run is the M3 pre-flight
config `wbc.backend=ort_core`.

---

## 2. Milestones, work packages, acceptance tests

### 2.0 Interface conformance suites (written in M1, run on every new backend)

| Suite | Command | Checks |
|---|---|---|
| `test_lb_backend.py` | `$SIMPY -m pytest G/tests/conformance/test_lb_backend.py --backend <name> [--endpoint …] --golden runs/g1rt/golden/ticks_v1.npz` | Reset semantics. 1-tick command lag (a cmd step at tick k changes q15 only via the tick k+1 frame). Balance/Walk switch exactly at ‖nav‖ = 0.05. Tick echo and tick_gap status. Per-tick parity on ≥ 5,000 golden ticks (≥ 500 Walk): max\|Δq15\| ≤ ε_backend (fp32 1e-5 rad; GPU 1e-4, the existing hook's gate `W/control/policy/g1_gear_wbc_policy.py:95`; fp16 2.5e-3) and policy choice 100% equal. Latency at 50 Hz p99 ≤ 2 ms |
| `test_vla_service.py` | `$SIMPY -m pytest G/tests/conformance/test_vla_service.py --endpoint tcp://127.0.0.1:<port>` | `ping`. `get_modality_config` keys equal those of 5555. `get_action` schema (7 keys, (1,30,D) f32, finite). An instruction over 24 tokens returns an error dict and the server stays alive [tt:2 Other constraints]. With `noise_seed`, same seed gives bit-identical output and different seeds give different output (TT and replay). p50 latency at 3 Hz is reported |
| `test_robot_contract.py` | `$SIMPY -m pytest G/tests/conformance/test_robot_contract.py` | `LowState` layout vs `robot_model` index groups. `CameraFrame` bytes equal the env observable image of the same tick. Deadline hold behaviour. A SystemExit from safety becomes `EPISODE_END{safety}` |

### 2.1 M1: async loop on the RTX 5090

Work packages (house style: h = agent-hours).

| WP | What | Files (all new under `G`, unless stated) | Deps | h | Device |
|---|---|---|---|---|---|
| WP-1.0 | Schemas + index tables + C header generator. Snapshot the research prototypes out of `/tmp` | `msgs.py`, `tools/gen_c_header.py` → `c/g1rt_msgs.h`, `tests/test_msgs.py`, `research_artifacts/` (copies of `S/plan_research/simloop/{async_proto,sim_timing}.py`, `S/plan_research/x280/tc/*`) | — | 6 | no |
| WP-1.1 | Transport (PAIR/PUSH/PUB helpers, lazy-pirate REQ with RCVTIMEO, drain-latest), deadline pacer with slip re-anchor (pattern of `S/live_viewer.py:68-89`), CPU pinning | `transport.py`, `clock.py`, `tests/test_transport.py` | 1.0 | 5 | no |
| WP-1.2 | **robot_sim**. `gym.make(env, enable_waist=True, camera_names=[ego,tpp], …)` **without** `WholeBodyControlWrapper` (`R/gr00t/eval/rollout_policy.py:95-119`); `env.step({"q": q_des})`. Seeded reset (`env.reset(seed=s)`; re-fetch the robosuite handle and re-apply observable `sampling_rate` after every reset [sim:6 Reset]). Capture-on-request render. Deadline hold. SystemExit catch around step (`W/control/envs/robocasa/sync_env.py:516-528`). Fall detection (pelvis z < 0.5 m). Success latch. RT/VL pacing. CUDA guard (`assert not torch.cuda.is_initialized()` if torch is imported). Optional onscreen viewer (camera patch copied from `S/live_viewer.py:36-65`) | `robot_sim.py`, `viewer.py`, `tests/conformance/test_robot_contract.py` | 1.0, 1.1 | 12 | no |
| WP-1.3 | **WBC core and backends**. `core.py`: numpy `LbStep` reference (§1.4 semantics). `fk.py`: torso rpy via pinocchio `robot_model`, plus a closed-form 3-joint version for C. `backends.py`: `RefController` (the unchanged gr00t_wbc `G1DecoupledWholeBodyPolicy` via `get_wbc_policy`, fed LowState dicts and `n1_utils.concat_action(row)` — the golden), `ort_core` (core.py + ORT `intra_op_num_threads=1`), `gpu_lbstep` (ZMQ), `legacy516` (core.py history + the stateless `gear_wbc_action` on 5555, used only as a negative control), and the slaved shadow | `wbc/core.py`, `wbc/fk.py`, `wbc/backends.py`, `tools/record_golden_ticks.py`, `tests/conformance/test_lb_backend.py` | 1.0 | 14 | no |
| WP-1.4 | **wbc_gpu_server** (meets U4 without restarting 5555). Its own process and CUDA context. A dedicated thread owns REP :5556 (zmq sockets are not shared across threads). Graphs are captured at startup, before serving (capture_error_mode global). A new `StreamGraphPolicy` runs H2D, replay, D2H **and** sync on one explicit stream with priority `priority_range()[1]` (−3 on the 5090). This fixes, in our copy, the `self.stream = current_stream()` bug of `R/gr00t/policy/gear_wbc_policy.py:83,85-92` [serve:Q2 latent bug]. It reuses `GearWbcMLP` by import. `LbStep` history runs on the CPU with core.py | `services/wbc_gpu_server.py` | 1.3 | 6 | 5090 |
| WP-1.5 | **Chunk scheduler** (pure functions over a plain state record, C-portable): `trigger(st,k)`, `on_arrival(st,chunk,k)`, `row(st,k)`. Implements §3 v1 plus the baseline modes `sync_hold` and `async_noskip` | `sched.py`, `tests/test_sched.py` | 1.0 | 8 | no |
| WP-1.6 | **controller**: §1.3 loop, row→J43 mapping, FK, backend + shadow, fail-over, CaptureReq/InferRequest, VL blocking on `k_avail` | `controller.py` | 1.1, 1.3, 1.5 | 8 | no |
| WP-1.7 | **vla_proxy**: CameraFrame + InferRequest → 9-key obs (state f32 cast = what SyncVectorEnv does [sim:2]). Lazy-pirate REQ. Options passthrough. `server_ms` from `info` when present | `vla_proxy.py`, `tests/test_vla_proxy.py`, `tests/conformance/test_vla_service.py` | 1.0, 1.1 | 6 | no |
| WP-1.8 | **Sync recorder + replay VLA**. `record_sync.py` reproduces the stock rollout (MultiStepWrapper n_action_steps=20, `R/gr00t/eval/sim/wrapper/multistep_wrapper.py:249-279`) for one env with seeded resets and the ORT CPU WBC (`GR00T_WBC_SERVER` unset). It logs per tick the obs state, the 9-key request, the chunk and the q passed to `env.step`. `replay_vla_server.py` speaks PolicyServer and returns recorded chunks keyed by (seed, k_obs); it rejects (error dict) if the request state differs from the recorded one by > 0 | `tools/record_sync.py`, `services/replay_vla_server.py` | 1.0 | 8 | 5090 (recording only) |
| WP-1.9 | Launcher + configs + per-process loggers + analysis (§4) | `launch.py`, `configs/{m1_rt_gpu,m1_rt_ort,m1_vl,m1_vl_equiv,m1_rt_legacy,m2_rt_tt,m3_rt_tt_x280}.yaml`, `log.py`, `tools/{analyze_run,check_equiv}.py`, `env.sh` | 1.2, 1.6, 1.7 | 10 | no |
| WP-1.10 | Eval harness: paired-seed matrix runner, Wilson CI, McNemar, non-inferiority | `tools/eval_matrix.py`, `tools/stats.py`, `configs/eval_m1.yaml` | 1.9 | 8 | 5090 (runs) |
| WP-1.11 | (Optional, DP1) Seeded GPU VLA: a second VLA copy that wraps `Gr00tPolicy` with `torch.manual_seed(options.noise_seed)` (noise is unseeded `torch.randn`, `R/gr00t/model/gr00t_n1d6/gr00t_n1d6.py:310-314`) | `services/gpu_vla_seeded_server.py` | 1.7 | 3 | 5090 (+7.3 GB) |

M1 total ≈ 94 h (≈ 2 calendar weeks with 2 lanes).

M1 acceptance tests. Setup: `source G/env.sh; cd ~/experiments/gr00t`. `runs/` lives at `~/experiments/gr00t/runs/g1rt/`.

| Test | Command | Pass thresholds | Closes |
|---|---|---|---|
| T-1.0 units | `$SIMPY -m pytest G/tests/test_msgs.py G/tests/test_sched.py G/tests/test_transport.py -q` | All pass. Header = 40 B. Python layouts equal C `offsetof` (clang-20 compile in-test). IPC round trip LowState→LowCmd p99 ≤ 0.2 ms over 10,000 ticks. Pacer jitter p99 ≤ 0.5 ms when idle. The REQ client returns within RCVTIMEO+2 ms when the server is killed | R3, R19 |
| T-1.1 **loop equivalence** | `$SIMPY -m G.tools.record_sync --seeds 0-9 --vla tcp://127.0.0.1:5555 --out runs/g1rt/sync_rec` ; `$SIMPY -m G.services.replay_vla_server --rec runs/g1rt/sync_rec --port 5558 &` ; `$SIMPY -m G.launch --config G/configs/m1_vl_equiv.yaml --seeds 0-9 --out runs/g1rt/t11` (VL, d_inj=0, s=20, `wbc.backend=ref`, VLA=:5558) ; `$SIMPY -m G.tools.check_equiv runs/g1rt/sync_rec runs/g1rt/t11` | Per tick, `q_des` and the next `LowState.q/dq` are **bit-identical** (max\|Δ\| = 0) for all ticks of all 10 episodes, and success flags and first-success ticks are equal. Fallback if a documented FP nondeterminism source is found: max\|Δq\| ≤ 1e-9 rad over the first 500 ticks and identical outcomes on 10/10 | R6, R7, index law |
| T-1.2 WBC parity | `$SIMPY -m G.tools.record_golden_ticks --episodes 20 --vla tcp://127.0.0.1:5555 --out runs/g1rt/golden/ticks_v1.npz` ; conformance `--backend ort_core` ; `$SRVPY -m G.services.wbc_gpu_server --port 5556 &` ; conformance `--backend gpu_lbstep --endpoint tcp://127.0.0.1:5556` | ort_core: max\|Δq15\| ≤ 1e-5 rad and policy 100% equal on ≥ 5,000 ticks incl. ≥ 500 Walk. gpu_lbstep: ≤ 1e-4 rad. The torso FK closed form matches pinocchio to ≤ 1e-9 rad | R7, R2 |
| T-1.3 RT timing, official M1 config | `$SIMPY -m G.launch --config G/configs/m1_rt_gpu.yaml --seeds 100-119 --out runs/g1rt/t13` ; `$SIMPY -m G.tools.analyze_run runs/g1rt/t13 --gates m1_rt` | Over 20 episodes: tick period p99 ≤ 21.0 ms; late publishes (> 2 ms after `T_k`) < 0.5%; cmd misses = 0; slips = 0; WBC round trip p99 ≤ 2.0 ms and 0 ticks > 20 ms; d p99 ≤ 5 ticks; requests in flight ≤ 1 always; starved ticks = 0; safety faults = 0; shadow parity p99 ≤ 1e-4 rad; VLA p50 within +10% of the idle trimmed 58.8 ms while P1 renders. These numbers also measure the untested CPU and GPU contention [serve: gaps] | R1, R5, R10, R12, R20g |
| T-1.3b negative control | as T-1.3 with `--set wbc.backend=legacy516 wbc.endpoint=tcp://127.0.0.1:5555` (5 episodes) | **Expected to fail**: ≥ 3% of ticks have WBC > 20 ms (measured synthetic 5.1% at the 0.4 s cadence [serve:Q1 measured]). This documents why the split is needed | R1 |
| T-1.4 success eval | `$SIMPY -m G.tools.eval_matrix --matrix G/configs/eval_m1.yaml --out runs/g1rt/m1_eval` ; `$SIMPY -m G.tools.stats runs/g1rt/m1_eval --primary A3 --baseline A0 --noninferiority 0.10` | §4.3. A3 (v1) non-inferior to A0 (sync-frozen) within 10 points (lower bound of the paired 95% CI of the difference > −0.10) at n = 160 paired, RT native. A3 better than A1 (sync-hold) in median time-to-success at VL d ≥ 5 (Mann–Whitney p < 0.05). 0 starved episodes | R4, R13, R27 |
| T-1.5 demo | `$SIMPY -m G.launch --config G/configs/m1_rt_gpu.yaml --set robot.viewer=true --seeds 200-209` | Tick period p99 ≤ 21 ms **with the onscreen viewer** (unmeasured so far [sim: gaps]). If it fails, fall back to an offscreen tpp stream at 25 Hz shown by a separate viewer process | R9 |

### 2.2 M2: VLA on Tensix (swap `vla.endpoint` only)

| WP | What | Files | Deps | h | Device |
|---|---|---|---|---|---|
| WP-2.1 | CPU-only bf16 conversion of the cloudwalk checkpoint: F32→BF16 for the 517 F32 tensors, keep the base 2-shard weight_map, copy config/processor/statistics. Asserts total_size = 6,573,217,664 and all dtypes BF16 [tt:2 dtype] | `tt/convert_cloudwalk_bf16.py` → `~/.cache/gr00t-g1-cloudwalk-bf16/` | DP3 | 4 | no |
| WP-2.2 | Runtime G1 support without repo edits. `g1` LayoutConfig = `dataclasses.replace(gr1, name='g1', embodiment_tag='unitree_g1', embodiment_id=8, valid_slice=(30,35))`, injected into `configs._CONFIGS` before importing `tt.model`. An HF_HUB_CACHE shim links the pinned sha `d0814e7ecb19` to the bf16 dir (the `_shim_snapshot` pattern). Dedicated `cache_root=~/.cache/gr00t-tt-g1-cloudwalk` to avoid cache-key poisoning [tt:2 cache] | `tt/g1_layout.py`, `tt/make_shim.py`, `tests/tt/test_g1_cpu.py` | 2.1 | 5 | no |
| WP-2.3 | G1 fidelity reference. Record 10 raw obs through P1+P3 (`raw_observation.npz` schema). Dump an fp32 golden with a fixed noise tensor, by a variant of `R/tt_golden/dump_golden_n16.py` (bf16 cloudwalk, `unitree_g1`, npz input) in a **separate** 5090 process (about 24 GB free; 5555 untouched) | `tt/record_raw_obs.py`, `tt/dump_golden_g1.py` | 1.7, 2.1 | 10 | 5090 |
| WP-2.4 | **TT VLA server**: PolicyServer-compatible REP on :5557. MsgSerializer is copied (≈40 lines; `$TTENV` has pyzmq 27.2 and msgpack 1.2.2). The main thread owns the device; one model per process (never build two megakernels [tt:3 Long-lived]). `from_pretrained('n16', 'unitree_g1', layout_name='g1', cache_root=…, num_command_queues=2)` + `warm_and_capture` at boot. **Fresh noise per call** from a running `torch.Generator`, or `options.noise_seed`/`options.noise`, because the default is fixed seed 42 [tt: Noise semantics]; see `T/tt/model.py:942-957`. Batch B is looped. Static-shape refusals become error dicts. Returns `info` timings | `services/tt_vla_server.py` | 2.2 | 8 | no (written), yes (run) |
| WP-2.5 | Device-session owner. `bin/with-device.sh G/bin/device_session.sh --tt-vla [--x280d]`. Unsets `TT_METAL_WATCHER` and `TT_METAL_TRACE_ALLOC_TRACKING` [tt:3; x280: watcher]. Sources `$TTENV`. Starts P4b, waits for ready, optionally starts P5b, and holds the flock for the whole session (`~/experiments/gr00t/bin/with-device.sh:1-10`). Probe/reset happens only at session start | `bin/device_session.sh` | 2.4 | 3 | yes |
| WP-2.6 | Device session S1: conformance, fidelity, latency, then closed loop (§T-2.x). E0/E1 of M3 run in the same session (§5.2) | — | 2.3–2.5, M1 | 10 | **yes** |

M2 total ≈ 40 h. Effort per [tt: rec.] is about 4–5 engineer-days.

| Test | Command | Pass thresholds | Closes |
|---|---|---|---|
| T-2.1 CPU parity | `$TTENV && python -m pytest G/tests/tt/test_g1_cpu.py -q` | G1 spec resolves (slot 8, state 43, action 35, H 30). Layout passes `assert_call`. Encode and decode are bit-identical to the Isaac processor on recorded G1 obs (max\|d\| = 0, reproducing [tt:1 Host pre/post]) | R14 |
| T-2.2 conformance | session up; `$SIMPY -m pytest G/tests/conformance/test_vla_service.py --endpoint tcp://127.0.0.1:5557` | All pass, including noise-seed determinism and survival after an error | R14, R16 |
| T-2.3 fidelity | `$SIMPY -m pytest G/tests/tt/test_g1_fidelity.py --endpoint tcp://127.0.0.1:5557 --golden runs/g1rt/golden/g1_fp32/` (a client; sends the golden's noise via `options.noise`) | On all 10 obs: PCC(actions, rows 0–29, 35 dims) ≥ 0.999. Arms PCC ≥ 0.999. Per-group max\|d\| ≤ 1.5× the GPU-bf16-vs-fp32 deviation of that group. 5 repeated calls are bit-identical. If it fails, retry with `mk_arena_dtype=bf16` [tt: risks] | R15 |
| T-2.4 latency | `$SIMPY -m G.tools.bench_vla --endpoint tcp://127.0.0.1:5557 --n 500 --period-ms 320` | Through ZMQ: p50 ≤ 48 ms, p99 ≤ 55 ms (projection 42 ms + 2.6 ms G1 encode + ZMQ [tt: summary]). Warm boot ≤ 10 s | R14 |
| T-2.5 sync closed loop | `$SIMPY -m G.tools.record_sync --seeds 0-49 --vla tcp://127.0.0.1:5557 --out runs/g1rt/sync_tt` vs `runs/g1rt/m1_eval/A0` same seeds | Success difference vs GPU sync within the 95% CI. A screen (n=50) must not be below −15 points | R15 |
| T-2.6 async RT (only `vla.endpoint` changed) | `$SIMPY -m G.launch --config G/configs/m2_rt_tt.yaml --seeds 100-119` + analyze `--gates m1_rt` ; then eval A3 at n=160 paired with `noise_seed` | All T-1.3 gates, with d p99 ≤ 4 ticks. Success non-inferior to M1 A3 within 10 points (paired) | R4 |

### 2.3 M3: WBC on the X280 (swap `wbc.backend` only)

| WP | What | Files | Deps | h | Device |
|---|---|---|---|---|---|
| WP-3.0 | **C WBC core** implementing `LbStep` exactly (§1.4): 72→frame, 6×86 ring, command lag, select, estimator 516-256-256-35 ELU → L2-norm latent, actor 121-512-256-256-15 ELU, post-process. Own `memcpy/memset`, a polynomial `exp`, no libcalls. Weight blob v1 (header, CRC32, fp32 or fp16, per policy). Built for x86 too and exposed as backend `x86c` via ctypes | `x280/wbc_core.{c,h}`, `x280/pack_weights.py` (ONNX initializers → blob), `x280/Makefile` (`make host`), `wbc/backends.py: X86CBackend` | 1.3 (golden ticks) | 14 | no |
| WP-3.1 | RISC-V image. clang-20 `--target=riscv64-unknown-elf -march=rv64gcv_zfh_zba_zbb -mabi=lp64d -mcmodel=medany -O2 -ffreestanding -fno-builtin -nostdlib`. crt0 parks harts ≠ 0 in wfi and sets mstatus.FS/VS. `csrw 0x7c1,0`. mtvec writes mcause/mepc/mtval to the mailbox. RNMI park handler. **Resident monitor**: LOAD(addr,len), FENCE_I, CALL(entry), PEEK/POKE, STEP (runs the loaded `LbStep`), PARK. LIM linker script at 0x0800_0000, length 0x1E0000 (prototype `S/plan_research/x280/tc/{crt0.S,lim.ld}` [x280: Toolchain]). No Zvfh until a trap-guarded probe passes; fp16 is widened with integer ops | `x280/{crt0.S,lim.ld,monitor.c,payload_wbc.c}`, `make x280` | 3.0 | 14 | no |
| WP-3.2 | **x280d**. pyluwen `PciChip` noc_read32/noc_write32 through its **KMD-allocated 2 MiB window only** (never the Hardcoded-index APIs [x280: Host TLB budget]). A legacy keeper fd `os.open('/dev/tenstorrent/0', O_RDWR)` held for the whole life (clock gating [x280: L2CPU clock gating]). `SET_NOC_CLEANUP` ioctl via ctypes writes `HOST_GONE` to mailbox line 0 [x280: Fd-close safety hook]. Full-line 64 B zero-init of every touched LIM line before any partial write [x280: LIM ECC]. Reset-aware tile budget (§5.2). Control endpoint `x280ctl` (e0…e5, load, call, status) and `LbStep` REP on ipc. A fake-L2CPU mode runs the x86 build of monitor + core behind the same mailbox code. pyzmq is installed into `~/.tenstorrent-venv` (pip) | `x280/x280d.py`, `x280/mailbox.py`, `x280/fake_l2cpu.py`, `tests/test_x280d_fake.py` | 3.0, 3.1 | 14 | no |
| WP-3.3 | Device session S1/S2 experiments E0–E4 (§5.2) | — | 2.5, 3.2, DP4 | 12 | **yes** |
| WP-3.4 | Kernel ladder: K0 scalar fp32 (Balance only in LIM; exactness), K1 RVV fp32 GEMV (row-dot e32/m4 + vfredusum, or column-broadcast), K2 fp16 weights for both policies in LIM. Each is measured with mcycle into the mailbox | `x280/payload_wbc.c` variants | 3.3 (E2, E4) | 12 | **yes** |
| WP-3.5 | `X280Backend` (ZMQ to x280d) + E5 + closed loop + soak | `wbc/backends.py` | 3.4, M2 | 10 | **yes** |

LIM map (WAYENABLE untouched; 1,966,080 B available [x280: L3 LIM capacity]):
- `0x0800_0000`: 64 KiB code, rodata, data, bss and stack.
- `0x0801_0000`: 4 KiB mailbox. One writer per 64 B line:
  - L0 host→X280 control {magic, cmd, seq_req, episode, tick, flags, args};
  - L1 X280→host status {seq_done, state, heartbeat, mcause, mepc, mtval, last_cycles, vlenb};
  - L2–L6 `LbInput` 72 f32;
  - L7–L9 `LbOutput`;
  - L10–L15 chunk descriptor (M4);
  - L16–L63 trap/debug ring.
- `0x0801_1000`: weight blob. Both fp16 policies (1,882,312 B, ends 0x081D_C8C8) or one fp32 policy, the same size.

Protocol:
1. Host writes L2–L6, then writes L0.seq_req=n (UC, Strict).
2. The X280 polls L0, computes, writes L7–L9, runs `fence rw,rw`, then writes L1.seq_done=n.
3. Host polls L1 with UC reads (first poll after 150 µs), with a deadline of 4 ms.

| Test | Command | Pass thresholds | Closes |
|---|---|---|---|
| T-3.0 x86 parity (no device) | `cd G/x280 && make host && $SIMPY -m pytest G/tests/conformance/test_lb_backend.py --backend x86c --precision fp32` (then `fp16`) ; `$SIMPY -m G.launch --config G/configs/m1_vl.yaml --set wbc.backend=x86c sched.d_inj=3 --seeds 0-49` | fp32: max\|Δq15\| ≤ 1e-5 rad. fp16: ≤ 2.5e-3 rad (raw error 7.4e-3 × 0.25 [x280: Weight precision]). Policy 100% equal. Closed-loop screen n=50: success difference vs ort_core within the paired 95% CI | R24, R26 |
| T-3.1 build/static | `make x280 && $SIMPY G/x280/check_image.py build/wbc.elf` | Image + data ≤ 64 KiB. All sections inside LIM. Undefined symbols = ∅. Weight blob ends ≤ 0x081E_0000 | R21 |
| T-3.2 fake L2CPU | `$SIMPY -m pytest G/tests/test_x280d_fake.py` | seq/torn-write protocol, timeouts, trap dump path, reset-aware tile logic (simulated `L2CPU_RESET` states) | R18, R26 |
| E0–E4 | `$LUPY -m G.x280.x280d ctl e0` … `e4` inside the session (§5.2) | thresholds in §5.2 | R18–R23, R29 |
| T-3.3 X280 parity (E5a) | `$SIMPY -m pytest G/tests/conformance/test_lb_backend.py --backend x280 --endpoint ipc:///tmp/g1rt/s/x280d` | Same ε as T-3.0 for the loaded precision. compute_ns p99 ≤ 2.0 ms (K1/K2), ≤ 3.0 ms (K0). Round trip controller↔X280 p99 ≤ compute + 0.3 ms | R23, R30 |
| T-3.4 closed loop (only `wbc.backend` changed) | `$SIMPY -m G.launch --config G/configs/m3_rt_tt_x280.yaml --seeds 100-119` + analyze `--gates m3_rt` ; eval A3 n=160 paired vs T-2.6 | All T-1.3 gates. 0 fail-overs. Heartbeat continuity 100%. TT VLA p50 within ±5% of T-2.4 (no interference). Shadow parity p99 ≤ ε. Success non-inferior to M2 within 10 points | R20, R30 |
| T-3.5 soak | 2 h RT continuous (about 200 episodes) | 0 traps, 0 fail-overs, L2CPU clock = 800 MHz throughout (tt-smi snapshots taken **before and after only**; no fd churn during the run) | R20 |

### 2.4 M4 (optional): on-chip handoff, whole controller on the X280, RTC v2

| WP | What | Files | Deps | h | Device |
|---|---|---|---|---|---|
| WP-4.1 | On-chip VLA→WBC handoff. The TT server copies the final action tensor into a **persistent** DRAM tensor at a fixed address (one tile, or ROW_MAJOR fp32 pages) and exposes `get_chunk_descriptor` {addr, layout, bank table from `ttnn.cluster.get_dram_bank_table`}. The X280 maps one 128 GiB **uncached** window per needed bank and reads it after the host bumps chunk_seq in L10 (seq re-check against torn reads). Decode in C: clip, min/max un-normalise, relative→absolute for arms with the k_obs state the X280 keeps in a 64-tick q ring [x280: rec. 5] | `services/tt_vla_server.py` (+endpoint), `x280/chunk_decode.c` | M3 | 20 | yes |
| WP-4.2 | Whole controller on the X280. `sched` (trigger, splice, starvation), row→J43, closed-form FK and LowCmd assembly in C. P2 becomes a relay: LowState→mailbox, mailbox→LowCmd, trigger→InferRequest | `x280/controller.c`, `controller.py --placement x280` | M3, 4.1 optional | 24 | yes |
| WP-4.3 | RTC v2 (§3.2). v2a guided (5090 reference only) and v2b soft-replacement on the 5090 first, then v2b in the TT megakernel Euler stage (three extra inputs Y, W, z; repo change in `T/tt/action_head.py:750-770` and the megakernel, DP8) | `services/gpu_vla_rtc_server.py`, TT patch | M2 | 16 + 30 | 5090, then yes |

| Test | Pass thresholds |
|---|---|
| T-4.1 | X280-decoded chunk equals host-decoded chunk to ≤ 1e-6 on 100 calls. Chunk visible to the X280 ≤ 1 ms after trace completion. Closed-loop screen n=50 within CI |
| T-4.2 | On replayed LowState/chunk streams from T-3.4 logs: trigger/splice decisions 100% identical to `sched.py`, q_des max\|Δ\| ≤ 1e-5 rad. RT gates as T-3.4 |
| T-4.3 | VL d ∈ {5, 8, 12}, n=160 paired: v2 success ≥ v1 + 5 points at some d ≥ 5, **or** boundary-jump p99 −30% with success non-inferior (−5 points). Otherwise v2 is not adopted |

---

## 3. Chunk scheduling policy

### 3.1 v1: async aligned receding horizon (implemented in `G/sched.py`, mirrored in C for M4)

| Parameter | Value | Rationale / source |
|---|---|---|
| Tick, horizon | dt = 20 ms; H = 30 rows (0.6 s). Never use DiT rows 30–49 (untrained, loss-masked) | [async: N1.6 action horizon] |
| Index law | row i ↔ tick k_obs + i. k_obs = the tick whose start state and ego frame were captured | Matches MultiStepWrapper, which applies row 0 on the tick after the obs (`multistep_wrapper.py:256-263`). Verified by T-1.1 |
| Trigger | At tick k, if no request is in flight and (k − k_obs_active) ≥ s_eff (or there is no active chunk): send CaptureReq_k + InferRequest. A trigger while busy is logged `skipped`, never queued | [async: v1 trigger]; RTC Alg. 1 |
| s_eff | min(s_cfg = **16**, 30 − d̂ − m), m = 2 | RTC constraint d ≤ s ≤ H − d [async: RTC] |
| d̂ (latency estimate) | max of the last b = 10 measured d (ticks). d_init = 5 in RT. = d_inj in VL | RTC Table 4 (b=10) |
| Latency compensation | On visibility at k_now: d = k_now − k_obs. Execute rows from d (skip the stale prefix). **Do not** re-anchor arms; they are absolute targets anchored at obs time | [sim:2]; `state_action_processor.py:445-484` via [async: N1.6 action representation] |
| Splice | Replace the whole not-yet-executed future. Blend B = 0 by default. Option B ∈ {3, 5}: linear crossfade a = (1−w)·old + w·new, w_j = (j+1)/(B+1), on arms, waist and nav only. Hands (near-binary) and base_height switch hard | [async: v1 splice]. Temporal ensembling is rejected (needs a call every tick; RTC shows it oscillates) |
| Late chunk | If usable rows 30 − d < 5 (d ≥ 26, 520 ms): reject (`late_reject`) and trigger immediately | [async: v1 splice] |
| Staleness | Drop any chunk whose `episode` differs from the current one (reset or instruction change). VLA RCVTIMEO 1000 ms → lazy pirate, `status=timeout`, immediate re-trigger | [sim: rec. ActionBuffer]; [serve: Client robustness] |
| Starvation | When the cursor reaches row 30 without a new chunk: hold the last executed arm, waist and hand targets and base_height; ramp nav linearly to [0,0,0] over 5 ticks (‖cmd‖ < 0.05 → Balance, `g1_gear_wbc_policy.py:257`). Resume from row d at the next arrival | [async: v1 stale/late] |
| Fault | 50 consecutive starved ticks (1.0 s) → `END_EPISODE_REQ` (reason starved). The robot keeps holding | Matches the 1 s goal timeout (`g1_decoupled_whole_body_policy.py:98`) and LeRobot `action_starvation_grace_s=1.0` |
| First chunk | Trigger at tick 0. RT: stand on hold targets (upper-body q of LowState_0, nav 0, height 0.74) until it is visible. VL: governed by d_inj | Real-robot behaviour |
| Balance/Walk | No hysteresis in v1 (fidelity to the evaluated setup). Flips are logged | DP6 |
| Expected (5090, d ≈ 3) | Rows used ≈ [3, 19) ⊂ the sync rows 0–19. Replan every 320 ms. GPU duty ≈ 19%. Chunk age ≈ d + s/2 ≈ 11 ticks | [async: v1 trigger] |
| Expected (TT, d ≈ 3) | Same s. Duty ≈ 14% at 45 ms | [tt: summary] |

Baseline and ablation modes, same code with different flags:
- `sync_frozen`: VL, d_inj=0, s=20.
- `sync_hold`: trigger when 20 rows are exhausted; hold everything, nav included, during inference; execute from row 0.
- `async_noskip`: period trigger, but execute from row 0 on arrival (negative control for latency compensation).
- `sync_aligned`: s_eff = 20 − d̂, so the switch happens near row 20 as in sync [sim: rec. Request scheduling].

### 3.2 v2: RTC (later option, M4)

- H = 30 and s = 15 (N1.6 has 30 trained rows, below the ≥ 32 the GR00T guide recommends for RTC, so s ≤ 15 and
  d ≤ 15 are hard ceilings [async: risks]).
- d̂ as in v1.
- Soft mask (RTC Eq. 5): W_i = 1 for i < d; c_i(e^{c_i}−1)/(e−1) with c_i = (H−s−i)/(H−s−d+1) for d ≤ i < H−s;
  0 beyond. For d = 3: [1,1,1,.815,.655,.518,.402,.305,.224,.158,.105,.065,.035,.015,.004,0,…].
- Guidance clip β = 5. Per-step weights at τ = 0, .25, .5, .75: [5, 3.33, 2.0, 3.33] [async: measurements].

Interface impact is none for P1, P2's protocol or the robot:
- P2 fills `InferRequest.rtc_prefix` with the active chunk's absolute rows for ticks [k_obs, k_obs + 30), NaN where
  undefined, and sets the RTC flag.
- P3 forwards it as `options.rtc = {prefix, d, s}`.
- The **service** re-encodes the prefix relative to the new obs state, with the per-index (30,7) statistics, pads to
  (50,128) and sets W = 0 on dims ≥ 35.

Variants:
- (a) Guided VJP. 5090 reference only: it needs autograd through the head and breaks inference_mode and graphs.
- (b) Gradient-free soft replacement: x ← x + W·((1−τ)z + τY − x) before each of the 4 Euler steps. Zero cost.
  Portable into the TT megakernel Euler stage (`T/tt/action_head.py:750-770`).
- (c) Training-time RTC (per-token τ; a fine-tune with d ~ U[0,10)). Best for Tensix; needs DP8.

Adopt v2 only if T-4.3 passes. Unit test: the round trip `apply_action → unapply_action` of the prefix ≤ 1e-6
[async: risks].

---

## 4. Measurement, logging, evaluation

### 4.1 Logging (each process logs its own side, joined offline on (episode, tick, seq))

Each process keeps an in-memory structured ring and flushes to `runs/g1rt/<run>/<proc>/ep<NNNN>.npz` **at episode
end**. Nothing is written in the tick path. The run directory also holds `config.yaml` (fully resolved), `git.txt`
(R/W sha and dirty diff hash, snapshot sha 009de892), `events.jsonl` and `summary.json`.

| Log | Per-record fields |
|---|---|
| robot (per tick) | episode, tick, T_k, t_pub_ns, t_cmdrx_ns, t_step_end_ns, step_ms, render_ms, cmd_missed, cmd_tick, slip, q[43], dq[43], base_pose[7], q_des_applied[43], success, pelvis_z, safety_fault |
| controller (per tick) | tick, t_staterx_ns, t_cmdtx_ns, ctrl_ms, wbc_rtt_ms, backend compute_ns, policy, shadow_max\|Δq15\|, failover, chunk_seq, cursor row, chunk_age (k − k_obs) ticks, rows_left, starved, in_flight, trigger/skipped, row35 commanded, nav sent, torso_rpy, boundary flag |
| controller (per request) | req_seq, k_obs, k_visible, d, d̂ used, s_used, first and last row used, outcome (applied / late_reject / stale_gen / timeout), boundary jump L2 (arms + waist) |
| proxy (per request) | req_seq, t_reqrx, t_framerx, t_send, t_recv, ser_ms, server info (recv_ns, encode/infer/decode ms), status, payload bytes |
| WBC service / x280d (per tick) | tick, recv→send µs, compute_ns, status, heartbeat (x280), mcycle/mtime pair |
| episode summary | seed, success, first_success_tick, end reason, falls, safety faults, starved ticks, n_requests, late rejects, Balance↔Walk flips, chunk-age mean/p99, boundary-jump p99, tick p99 |

Derived metrics (`tools/analyze_run.py`):
- Gantt plot: ticks vs request bars vs switches.
- Histograms: tick period, tick lateness, controller latency, WBC RTT, d, chunk age.
- Jitter metrics from the GR00T deployment guide (`R/getting_started/real_world_deployment.md:283-362`): intra-chunk
  acceleration, boundary jump and momentum shift, computed on the **executed** per-tick target sequence so that sync
  and async compare like for like.
- Gate files `gates/{m1_rt,m3_rt}.yaml` hold the thresholds of §2.

### 4.2 Run modes

- **RT** (wall clock): the demo and all timing gates. d is measured.
- **VL** (virtual latency): deterministic, unpaced. `d_inj` ∈ {0, 3, 5, 8, 12} (0/60/100/160/240 ms) emulates the
  5090, TT and slower services. Several robot processes may run in parallel against one VLA service.

### 4.3 Evaluation protocol

- **Scenes**: paired seeds `env.reset(seed=s)`. The stock rollout resets unseeded [sim:6 Reset]. Screening uses
  s ∈ 0–49; confirmation uses 1000–1159 (n = 160).
- **VLA noise**: the 5090 server is unseeded (`gr00t_n1d6.py:310-314`) and cannot be changed (U-rule). M1 pairs scenes
  only, which McNemar handles. With DP1 the seeded GPU copy (WP-1.11) uses `noise_seed = seed·1000 + req_seq`. The TT
  server (M2+) honours `noise_seed` natively, so M2/M3 comparisons are fully paired.
- **Arms**:

| Arm | Description | Role |
|---|---|---|
| A0 sync-frozen | stock MultiStepWrapper n=20, sim frozen during inference (via `record_sync.py`) | primary baseline (optimistic: hides latency) |
| A0′ stock sanity | README command (`R/examples/GR00T-WholeBodyControl/README.md:83-88`), unseeded, n=50 | checks that A0's harness matches stock within CI |
| A1 sync-hold | robot keeps stepping and holds during inference; rows from 0 | honest synchronous baseline (RTC's real-robot comparison) |
| A2 async-noskip | period trigger, rows from 0 | negative control for latency compensation |
| A3 v1 B=0, s=16 | **primary** | candidate |
| A4 v1 B=3 | crossfade | ablation |
| A5 sync-aligned s=20−d̂ | | ablation (DP2) |

- **Matrix**:
  - Screening at n = 50: A1–A5 × VL d_inj ∈ {0, 3, 5, 8, 12}, plus RT native. About 15 s per VL episode (1440 ticks ×
    about 5 ms + about 90 calls × 60 ms + 2.8 s reset), so a cell takes about 12.5 min and the screen about 6.5 h.
  - Confirmation at n = 160: A0 vs A3 at RT native and VL d = 3, 5 (about 40–85 min per cell).
- **Statistics**:
  - Primary metric: success = success latched at any tick ≤ 1440 (as in the sync eval). Report Wilson 95% CIs.
    McNemar exact test on paired outcomes. Non-inferiority margin 10 points.
  - Secondary metrics: time-to-success (Mann–Whitney), falls, safety faults, starvation, boundary jump, flips, tick
    jitter.
  - Power: 45% vs 30% needs n = 160 per arm; 40% vs 30% needs 353 [async: measurements]. The README warns of ±15%
    variance, so **no conclusions from n < 50**.
- **Per-stage re-runs**:
  - M2 repeats A0 and A3 with the TT server.
  - M3 repeats A3 with the X280 backend.
  - The paired difference of each milestone vs the previous one is the acceptance quantity (T-2.6, T-3.4), since each
    milestone changes one backend.

---

## 5. Risk register and X280 sequencing

### 5.1 Register

| # | Risk | Evidence | Mitigation | Closes (WP → test) |
|---|---|---|---|---|
| R1 | Head-of-line blocking: the WBC waits behind the VLA on the single REP server | `server_client.py:69-71,120-143`; `run_gr00t_server.py:106-110`. Measured back-to-back WBC p99 67 ms, 35.7% of ticks > 20 ms; 5.1% at the 0.4 s cadence [serve:Q1] | The WBC is never on the VLA socket. P5a has its own process and port (prototype p99 0.92 ms [serve:Q3/Q4]). Negative control T-1.3b | WP-1.4 → T-1.3, T-1.3b |
| R2 | `CudaGraphPolicy` syncs the wrong stream if used from another thread | `gear_wbc_policy.py:83,85-92` [serve:Q2 latent bug] | `StreamGraphPolicy` with one explicit high-priority stream; graphs captured before serving | WP-1.4 → T-1.2 |
| R3 | `PolicyClient.timeout_ms` is never applied, so a dead server freezes the sim | `server_client.py:162,210-211` | Own REQ client: RCVTIMEO + lazy pirate. The WBC has a 5 ms deadline + shadow | WP-1.1 → T-1.0 |
| R4 | Latency-induced distribution shift (async uses rows ≈ 3–19, sync 0–19); success is already 30–50% | [sim: risks]; [async: risks] | Index-law skip. VL d sweep. A1 honest baseline. Non-inferiority design | WP-1.10 → T-1.4, T-2.6 |
| R5 | A chunk-switch discontinuity trips the arm 6 rad/s limit → `sys.exit(1)`, which kills the process or silently hangs a thread | `sync_env.py:516-528`; `joint_safety.py:21-22,451` | P1 catches SystemExit → `EPISODE_END{safety}`. Boundary-jump metric. Crossfade B=3 if needed | WP-1.2, 1.5 → T-1.3 (0 faults) |
| R6 | Clock mixing: gr00t_wbc timeouts and InterpolationPolicy use `time.monotonic`; sim time or pauses inject stop goals | `g1_decoupled_whole_body_policy.py:27,47,91-110`; `interpolation_policy.py:130-134` | The tick is the only control time. `RefController` gets `time=None` and `set_goal` every tick. No InterpolationPolicy in v1 | WP-1.3 → T-1.1 |
| R7 | The re-implemented WBC breaks the 1-tick command lag or obs ordering | [sim:4 ordering note] | Normative `LbStep` spec. Golden ticks from the unchanged reference. Conformance suite. Bit-exact loop test | WP-1.3 → T-1.1, T-1.2 |
| R8 | Reset takes 2.5–2.8 s and re-creates the MjSim, so cached handles go stale | [sim:5 Things that do break]; `n1_utils.py:29-33` | Reset happens outside the paced loop. Handles are re-fetched. The pacer re-anchors. `EPISODE_START` resets P2/P3 state | WP-1.2 → test_robot_contract |
| R9 | In-loop video (17–87 ms spikes) or viewer cost breaks 20 ms | [sim:5] | No encoding in P1. Viewer gated by T-1.5, with a separate-process fallback | WP-1.2, 1.9 → T-1.5 |
| R10 | A CUDA context in the sim process corrupts EGL | task constraint; `gear_wbc_policy.py:4-7` | `CUDA_VISIBLE_DEVICES=""` for P1–P3. Runtime assertion | WP-1.2 → test_robot_contract |
| R11 | Port 5555 collisions | [sim:6 Port collision] | ipc endpoints. TCP only on 5556–5558 | WP-1.1 |
| R12 | CPU contention (sim + 118 server threads + EGL on 16 cores) is unmeasured | [serve: gaps] | Core pinning. Measured by the T-1.3 gates | WP-1.1 → T-1.3 |
| R13 | Balance/Walk flicker around ‖cmd‖ = 0.05 | [sim:6 Balance/Walk] | Logged. Hysteresis only by DP6 | WP-1.10 → T-1.4 (secondary) |
| R14 | TT G1 gaps: no `g1` layout, pinned sha, 517 F32 tensors, cache-key collision, fixed noise 42 | [tt:2], [tt: Noise semantics] | WP-2.1/2.2 runtime injection, bf16 conversion, separate cache_root, fresh or seeded noise | WP-2.1–2.4 → T-2.1, T-2.2 |
| R15 | TT fidelity on the fine-tune is unknown (evidence is GR1 base only, PCC 0.99923) | [tt:5] | G1 fp32 golden + PCC gates. Fallback `mk_arena_dtype=bf16` | WP-2.3 → T-2.3, T-2.5 |
| R16 | Card stability: building two megakernels in one process hangs; `with-device.sh` may reset the card | [tt:3 Long-lived]; `bin/with-device.sh:5-6` | One model per process. All device work inside `device_session.sh`. Probe only at session start. Restarts inside a session use `WITH_DEVICE_NO_PROBE=1` | WP-2.5 → T-2.2 |
| R17 | Working-tree drift (the live tree has Stage-3 WIP) | [tt: Working tree] | P4b runs only from `$TTENV` (009de892). The sha is logged in `git.txt` | WP-2.4 |
| R18 | **X280 hart release works once per chip reset** (HW bug) | `X/tt-isa-documentation/BlackholeA0/L2CPUTile/README.md:29-31` (verified) | Only the resident monitor is ever released. RNMI handlers are installed first. Tile budget and order in §5.2. A hung payload is recovered by RNMI and never needs a reset | WP-3.1, 3.2 → E2, T-3.2 |
| R19 | LIM ECC: a partial write to an uninitialised line can fault | [x280: LIM ECC] | Zero-init every touched line in full 64 B lines. Image padded to 64 B. E1 tests it **before** any release | WP-3.2 → E1 |
| R20 | L2CPU clock gated after the last L2CPU_ENABLE fd closes; PLL4 relocks on every fd open/close | [x280: clock gating; CMFW re-locks PLL4] | Keeper fd in x280d. No fd churn (tt-smi included) during runs. E3 characterises it. Plan for 800 MHz, never 1750 | WP-3.2 → E3, T-3.5 |
| R20g | GPU contention between EGL renders and the VLA on the same 5090 is unmeasured | [sim: gaps] | Measured in T-1.3 (VLA p50 within +10%) | T-1.3 |
| R21 | Memory collision: the X280's "own" GDDR is tt-metal banks 5/6/7 | [x280: tt-bh-linux root causes; DRAM bank layout] | LIM-only in M3. M4 reads only a tt-metal-allocated tensor, read-only | WP-3.1 → T-3.1; WP-4.1 |
| R22 | Host TLB exhaustion or clobbering UMD windows | [x280: Host TLB budget] | KMD-allocated 2 MiB windows only. pyluwen Hardcoded APIs banned (grep check in T-3.2) | WP-3.2 → T-3.2 |
| R23 | X280 micro-architecture unknowns (no Zvfh in the DT; VLEN 512 from marketing) | [x280: ISA] | E2 reads vlenb and misa. Trap-guarded probes. Integer fp16 widening. Scalar K0 baseline | WP-3.4 → E2, T-3.3 |
| R24 | fp16 weights (≈ 7e-3 raw error) change closed-loop behaviour | [x280: Weight precision] | x86 closed-loop screen first (T-3.0). K0 fp32 exactness. If fp16 fails, put Walk fp32 in a second tile's LIM or in DRAM (DP5) | WP-3.0, 3.4 → T-3.0, T-3.4 |
| R25 | Torn or stale reads (tt-metal writes DRAM bypassing X280 caches) | [x280: Coherence] | Uncached aliases + seq re-check | WP-4.1 → T-4.1 |
| R26 | No UART; failures are opaque | [x280: risks] | Mailbox trap dumps and heartbeat. The same C sources build on x86. Fake-L2CPU harness | WP-3.0, 3.2 → T-3.0, T-3.2 |
| R27 | Eval variance (±15%) | README; [async: measurements] | n = 160 paired; no conclusions below 50 | WP-1.10 → T-1.4 |
| R28 | Watcher on → host writes to non-LIM L2CPU addresses throw | `sanitize_noc_host.hpp:147-155` [x280: tt-metal knows] | `device_session.sh` unsets `TT_METAL_WATCHER` | WP-2.5 |
| R29 | Translated vs raw coordinate confusion → silent wrong-target accesses | [x280: risks] | Host side uses UMD identity coords (8,3/9/5/7). E0 cross-reads (8,7) and (8,29) | E0 |
| R30 | The X280 hop adds jitter on the 50 Hz critical path | estimate 5–25 µs + compute [x280: Performance estimate] | 4 ms deadline, slaved shadow, fail-over after 3 misses | WP-3.5 → T-3.3, T-3.4 |
| R31 | The research scripts and X280 reference sources live in `/tmp` scratch and may vanish | `S/plan_research/*` | WP-1.0 copies them into `G/research_artifacts/`. WP-3.1 re-clones tt-bh-linux and tt-isa-documentation at the commits cited | WP-1.0 |
| R32 | Load tests perturb the shared 5555 server | [serve: risks] | Loop tests use the replay VLA (:5558). Back-to-back load only in T-1.3b (5 episodes) | WP-1.8 |

### 5.2 X280 experiment sequencing under the once-only reset (no step ever resets the chip)

Invariants:
1. Nothing in M3 calls `tt-smi -r`, tt-bh-linux `boot.py` (it board-resets and writes WAYENABLE=0xF [x280: root
   causes]) or writes `CCACHE0_WAYENABLE`.
2. All X280 access happens inside one `device_session.sh` (flock held). The `with-device.sh` probe, the only path that
   may reset, runs **before** anything is released.
3. x280d opens its keeper fd and pyluwen window **before** any release and keeps them until session end.
4. The 5090 M1 path and the `ort_core` backend stay runnable, so an X280 failure never blocks the demo.

**Reset-aware tile budget.** At startup x280d reads `L2CPU_RESET` (ARC (8,0) 0x8003_0014) bits [7:4]:
- bit = 0 → the tile is available;
- bit = 1 with a live monitor heartbeat in its LIM → attach;
- bit = 1 without a heartbeat → spent until the next chip reset.

The release order is L2CPU3 (8,7, bit 7), then L2CPU2 (8,5, bit 6). L2CPU0 and L2CPU1 are reserved and untouched
until the monitor is proven twice [x280: rec. 2].

| Step | Session | What | Pass | On failure |
|---|---|---|---|---|
| E0 read-only | S1 (with the TT VLA server running) | L2CPU_RESET = 0x0F with bit 7 = 0; WAYENABLE = 0; PLL4 fbdiv 64 / postdiv 1; hart status at 0xFFFF_F7FE_FFF1_0400 (64-bit address path); (8,7) vs (8,29) read equal | all as expected; TT VLA outputs bit-identical with fixed noise before vs after, latency ±1% | stop M3 device work; nothing spent |
| E1 LIM, no release | S1 | full-line zero-fill, pattern write/readback, one 4 B UC partial write to an initialised line, all while the VLA runs at 3 Hz | readback exact; no bus error; VLA unchanged | fall back to the DRAM placement design; nothing spent |
| DP4 | — | **User approves spending L2CPU3's release** | — | — |
| E2 release | S2 | P2–P8 of [x280: rec. 1]: zero-init, image, RNMI + exception handlers, reset vectors, PLL4 → 200 MHz, set bit 7, PLL4 → 800 MHz, L2PF config; the monitor publishes magic/vlenb/misa/heartbeat | heartbeat at ≥ 1 kHz; vlenb = 64; mcycle/mtime ≈ 16 at 800 MHz | RNMI park + reload; if that fails, mark L2CPU3 spent and go to L2CPU2 in the next session; a chip reset only at session end and with user approval |
| E3 power | S2 | open and close a second fd; restart the TT VLA server with `WITH_DEVICE_NO_PROBE=1` while the keeper fd holds | heartbeat continuous (no gap > 1 ms); clock 800 MHz after each relock | if harts stop on a relock: forbid any fd churn during runs and document it |
| E4 bench | S2 | K0/K1/K2 GEMV and full MLP from LIM; NoC read latency to a DRAM bank and a Tensix L1 | K1/K2 MLP ≤ 2 ms (estimate 0.1–0.6 ms) | use scalar K0 (≤ 3 ms) |
| E5 | S3 | T-3.3 parity, then T-3.4 closed loop, T-3.5 soak | §2.3 | shadow fail-over keeps the run alive; debug offline on the x86 build |

---

## 6. Critical path, parallel lanes, decision points

**Critical path** (≈ 5–6 calendar weeks to M3):
1. WP-1.0 → {WP-1.2, 1.3, 1.5, 1.7} → WP-1.6 → WP-1.8/1.9 → T-1.1 → T-1.3 → T-1.4. M1 takes about 2 weeks; the
   T-1.4 compute is 1–2 days.
2. Device session S1: T-2.2 to T-2.6, plus E0/E1 (week 3–4).
3. DP4 → S2: E2–E4 (week 4–5).
4. S3: E5, T-3.4, T-3.5 (week 5–6).

**Parallel lanes** (file-disjoint):

| Lane | Content | Needs |
|---|---|---|
| A (loop, `$SIMPY`) | M1 WPs | 5090 for recording and eval runs |
| B (TT host prep) | WP-2.1, 2.2, 2.4 code, 2.3 golden | CPU, plus 5090 for the golden; starts day 1 |
| C (X280, device-free) | WP-3.0 (once the T-1.2 golden ticks exist), 3.1, 3.2, T-3.0–T-3.2 | clang-20 only |
| D (device, serialized by flock) | S1 → S2 → S3 | one owner at a time. Each session is a single `bin/with-device.sh G/bin/device_session.sh …` invocation; tests are ZMQ clients |

**Decision points / open questions for the user** (defaults in bold):

| DP | Question | Default | Needed before |
|---|---|---|---|
| DP1 | How to serve the WBC "from the GPU server" in M1 | **a separate process `wbc_gpu_server` on :5556 (no restart of 5555)**. Alternatives: add `--gear-wbc-port` + thread to `run_gr00t_server.py` and restart 5555; also the seeded second VLA copy (WP-1.11, +7.3 GB) | WP-1.4 |
| DP2 | Trigger cadence | **s = 16 fixed**; switch to sync-aligned 20 − d̂ if the A5 screen beats A3 | T-1.4 confirmation |
| DP3 | Checkpoint | **cloudwalk G1 (the one served today), bf16-converted for TT**; alternative nvidia/GR00T-N1.6-G1-PnPAppleToPlate (bf16 native) | WP-2.1 |
| DP4 | May we spend L2CPU3's one-time release, and when may a `tt-smi -r` happen (it kills every tt-metal session on the card)? | **yes, in S2; resets only at session end** | E2 |
| DP5 | X280 precision | **fp16 weights for both policies in LIM, if T-3.0/T-3.4 pass**; otherwise fp32 (Walk on a second tile or in DRAM) | WP-3.4 |
| DP6 | Balance/Walk hysteresis (a behaviour change) | **off** | T-1.4 |
| DP7 | May repos be edited (TT `configs.py` g1 layout, snapshot-dir override, server split)? | **no; runtime injection + `G`** | M2 |
| DP8 | RTC v2 variant / training-time RTC fine-tune | **v2b on the 5090, then TT, only if T-4.3 passes; no fine-tune** | M4 |
| DP9 | Install qemu-user (sudo) for RVV host tests | **not needed** (x86 build + fake L2CPU cover logic; E4 covers RVV) | WP-3.1 |
| DP10 | Eval budget (about 6.5 h screen + about 6 h confirmation per stage on the 5090) | **accept** | T-1.4 |

Open questions carried into the WPs:
- (1) The response alignment (row 0 ↔ tick k_obs) is inferred; T-1.1 settles it.
- (2) Onscreen viewer cost (T-1.5).
- (3) Whether pyluwen `PciChip` open/init disturbs a live UMD session (E0 checks VLA bit-identity).
- (4) Whether harts survive PLL4 relocks (E3).
- (5) Real G1 end-to-end TT latency (T-2.4) sets d̂ and whether s can drop to 10 for more reactivity.
