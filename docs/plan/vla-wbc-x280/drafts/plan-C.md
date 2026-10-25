# GR00T N1.6 VLA + Decoupled-WBC, concurrent and asynchronous, on one Blackhole p150a — plan C (demo-first)

Date 2026-10-24. Drafter C. Angle: **the shortest path to a visibly working async demo at every milestone**, where each
milestone swaps exactly one backend behind a wire that does not change.

Evidence: `research/understand.json`, cited as `U.<section> F<n>` (finding), `U.<section> rec<n>` (recommendation, 1-based)
or `U.<section> m` (measurements). Sections are `sim` = sim_control, `srv` = serving_gpu, `tt` = tt_port, `x` = x280 and
`as` = async_practice. I also read code for this draft, cited as `file:line`. No repository file was modified and the
device was not opened. The GR00T server on 127.0.0.1:5555 was only inspected with `ps`.

Path shorthand:
- `R` = `~/experiments/gr00t/Isaac-GR00T-n1.6`
- `W` = `R/external_dependencies/GR00T-WholeBodyControl`
- `TT` = `~/experiments/gr00t/tt-metal/models/experimental/gr00t` (the live tree)
- `MK` = `~/experiments/gr00t/publish/worktree-mk` (frozen snapshot 009de892)
- `RT` = `~/experiments/gr00t/rtloop` (**new**; all new code goes here)
- `P` = `~/experiments/gr00t/docs/plan/vla-wbc-x280`
- `S` = `/tmp/claude-1000/-home-deepgadget-isaacsim/bfbe65f0-1b38-425e-84fc-4937eb211a38/scratchpad`

Units: d = engineer-days, ms/µs as written, and "tick" = 20 ms (50 Hz).

---

## 0. Goal, scope, non-goals, decisions

### 0.1 Goal

A live demo of a Unitree G1 in MuJoCo doing `LMPnPAppleToPlateDC_G1_gear_wbc`. It behaves like a real robot:
- The lower-body WBC (gear_wbc Balance/Walk MLP, 516 -> 15) runs on **every 50 Hz tick**, paced to wall time.
- GR00T N1.6 runs **in the background** and delivers 30-step action chunks that are spliced in by tick stamp.

The end state has both on one p150a: the VLA on Tensix and the WBC on an X280 hart.

Every milestone ends with:
- a **runnable demo**: the live MuJoCo viewer plus a HUD that shows VLA latency, chunk age, WBC tick timing and which
  processor runs what;
- a **numeric go/no-go verdict** written by a script.

### 0.2 Demo ladder (what is on screen at each step)

| Step | Sim + control loop | VLA backend | WBC MLP backend | On screen | Exit gate |
|---|---|---|---|---|---|
| M1.0 (day ~3) | host CPU, 1 process | mock server (latency 60 ± 5 ms, deterministic chunks) | host onnxruntime CPU | viewer + HUD; robot stands and sways on mock chunks | T-M1-1, T-M1-2 |
| M1.1 (day ~4) | host | **RTX 5090**, the running server :5555 | host onnxruntime CPU | robot does the apple task asynchronously | T-M1-3a |
| **M1** | host | RTX 5090 :5555 | **RTX 5090 CUDA graph**, own process :5556 | HUD: "VLA 5090 / WBC 5090" | T-M1-1..7 |
| **M2** | host | **p150a Tensix** :5557 | RTX 5090 :5556 | HUD: "VLA Tensix / WBC 5090"; optional GPU<->TT swap between episodes | T-M2-1..7 |
| **M3** | host | Tensix :5557 | **X280 L2CPU3 hart 0** via x280d :5558 | HUD: "VLA Tensix / WBC X280 @800 MHz, n µs (mcycle)"; live WBC hot-swap X280<->GPU | T-M3-* |
| M4a | host (robot I/O only) | Tensix | **whole lower-body controller on X280** (history, Balance/Walk select, FK) | same HUD; host sends only proprio | T-M4a-* |
| M4b | host | Tensix -> device DRAM -> X280 | X280 runs splicer + decode + WBC | HUD: "chunk handoff on chip" | T-M4b-* |
| M4c | — | RTC v2 (5090 reference; gradient-free variant on Tensix) | — | A/B overlay | T-M4c-* |

### 0.3 Scope

- A new standalone repo `RT`: control loop, splicer, wire clients, telemetry and HUD, evaluation scripts, the TT G1 VLA
  server, the X280 firmware and its host daemon.
- **Through M3 no file in `R`, `W` or tt-metal is edited.** Integration uses runtime patches, following two precedents:
  - `S/live_viewer.py:37-65` monkeypatches `MjviewerRenderer`;
  - `TT/libero/n16.py:176-229` (`install()`) re-points the port's registries at a fine-tune process-locally.

### 0.4 Non-goals

- ROS2/DDS: not installed (U.sim F6). We copy the latest-wins semantics, not the code.
- Teleop or IK: the VLA path has no IK (U.sim F1).
- Linux on the X280, or reaching 1750 MHz (U.x rec10).
- Batch > 1 on TT (U.tt F7).
- RTC before v1 has been evaluated.
- Encoding video inside the loop (U.sim F11).
- Real hardware.
- Changing training. The one exception, training-time RTC, is listed as an option only.

### 0.5 Decisions already made (user)

| # | Decision |
|---|---|
| U1 | VLA = GR00T N1.6 3B on **p150a Tensix**, using the existing tt-metal port. |
| U2 | WBC = GR00T-WholeBodyControl Decoupled gear_wbc lower body on the p150a's **SiFive X280**. Assume it can run concurrently with tt-metal: the tt-bh-linux "cannot be used concurrently" note (tt-bh-linux README.md:41) is treated as outdated. Its concrete causes are listed in U.x F0, and this plan avoids each one (§5.2). |
| U3 | Real-robot-like asynchrony: the WBC runs every 50 Hz tick and the VLA runs in the background. |
| U4 | Order: M1 is the async loop on the 5090 (sim on host CPU, VLA and WBC on the GPU), then M2 TT, then M3 X280. |
| U5 | The MuJoCo sim process must never create a CUDA context, because doing so corrupts its EGL renders (stated in `R/gr00t/policy/gear_wbc_policy.py:1-8`). |

### 0.6 Decisions this plan makes

| # | Decision | Basis |
|---|---|---|
| D1 | All new code lives in `RT`, a new git repo. Isaac, WBC and tt-metal are imported and never edited (§0.3). | precedents above; dirty trees in `R` (`git status`: run_gr00t_server.py M, gear_wbc_policy.py ??) |
| D2 | **One single-threaded control process** holds the sim, the splicer, a non-blocking VLA client (one REQ, `poll(0)` each tick) and the WBC client. There are no threads on the tick path; the VLA "background" is its server process. | U.srv rec4; U.srv F6 (GIL is not an issue); prototype `S/plan_research/simloop/async_proto.py` held 20.00 ± 0.35 ms (U.sim m) |
| D3 | The WBC is **never** on the VLA socket. In M1 it runs in a **new, separate GPU process on :5556**, so the running :5555 server needs no restart. | U.srv F1 (WBC p99 67 ms behind the VLA); U.srv F2/E3 (a separate CUDA process gives p99 0.62 ms under VLA load) |
| D4 | **Two stable wires.** **W-VLA** is the PolicyServer protocol (`get_action`). **W-WBC v1** is `gear_wbc_action(policy, obs[1,516]) -> action[1,15]`. Each milestone changes only an endpoint URL. The whole-controller wire W-LBC v2 appears only in M4a. | `R/gr00t/policy/server_client.py:117-143`; the `W` hook's call shape (git diff of `g1_gear_wbc_policy.py`, `load_remote_policy`) |
| D5 | **The WBC backend seam is `G1GearWbcPolicy.load_onnx_policy`**, which returns `run_inference(t[1,516]) -> t[1,15]`. `W/gr00t_wbc/control/policy/g1_gear_wbc_policy.py:31-35` and `:56` show the existing seam; it is called every tick at `:257-264`. RT patches it at runtime with a process-wide singleton backend. The patch survives the per-reset policy re-creation (`W/.../utils/n1_utils.py:29-33`) and never re-runs the per-reset verification RPC. | code read |
| D6 | Cameras keep rendering every tick in M1-M3 (5.26 ms/tick, 3.76x real time). There is no on-demand render code. | U.sim F9/F10 (on-demand saves only 13%) |
| D7 | Splice v1 = **aligned receding horizon**: s = 16, no blend (B = 0), latency skip by tick stamp (§3). | U.as rec1-4 |
| D8 | **No-future-leak rule.** The paced sim computes tick k-1's physics about 15 ms before wall time k·dt. The observation o_k may be sent **no earlier than its wall deadline t0 + k·dt**, so the measured latency is honest. | derived from the paced loop (U.sim F10) |
| D9 | The HUD is a **separate pygame process** fed by a ZMQ PUB that drops messages on overflow, so it can never stall the loop. An in-viewer "status lamp" (a `user_scn` geom) is a stretch goal. MuJoCo 3.2.6's passive `Handle` has no text-overlay API (only `user_scn`, `sync`, `lock`; `mujoco/viewer.py:64-160` in the client venv). | code read; pygame 2.6.1 is present in the client venv |
| D10 | The TT VLA server is a **clone of the validated LIBERO TT server** `TT/libero/server_n16.py`. That server already speaks the PolicyServer wire, draws fresh seeded noise, has an `info` endpoint and a boot golden self-check, and ran **100/100 closed-loop episodes with 0 errors** (p50 56.4 ms with 2 cameras and S_pad 256). It runs from the frozen snapshot `MK`. | `TT/libero/server_n16.py:1-36,170-194`; `~/experiments/gr00t/libero_eval/n16/tt_spatial_r0_summary.json`; U.tt rec1. This corrects U.tt F16 ("no ZMQ server in the port"). |
| D11 | The X280 is driven by a **standalone daemon, `x280d`** (pyluwen, plus a legacy keeper fd). Memory is LIM only. Only L2CPU3 is released. **The first image is a resident monitor**, not the WBC. | U.x rec1, rec2, rec9 |
| D12 | **WBC fallback chain** in the client: primary (X280 or GPU) -> GPU -> in-process ORT CPU, with a 5 ms receive timeout. This is safe at any tick because the W-WBC v1 backends are stateless: the history stays on the host until M4a. | U.srv rec3 |
| D13 | Wall-clock mode (RT) is the demo. Virtual-latency mode (VL) is for statistics. | U.as rec6 |

---

## 1. Target architecture

### 1.1 Processes and ownership

```
 host CPU (Ryzen 9 7950X, 16C/32T)                         RTX 5090                 Blackhole p150a
 ┌─────────────────────────────────────────────┐
 │ rt_sim  (client venv, CUDA-free)             │  W-VLA   ┌───────────────┐  M1: running server :5555 (pid 3134179, untouched)
 │  ├ MuJoCo/robosuite env + onscreen viewer    │ ───────▶ │ VLA backend   │  M2+: server_g1 :5557 ──────────▶ Tensix (traces, DRAM)
 │  ├ 50 Hz tick loop (wall-clock deadlines)    │          └───────────────┘
 │  ├ ChunkSplicer (pure function)              │  W-WBC   ┌───────────────┐  M1/M2: wbc_gpu_server :5556 ─▶ 5090 CUDA graph
 │  ├ VlaClient (1 REQ, poll(0) per tick)       │ ───────▶ │ WBC backend   │  M3: x280d :5558 ─(LIM mailbox)─▶ X280 L2CPU3 h0
 │  ├ WbcClient (REQ, RCVTIMEO 5 ms, fallback)  │          └───────────────┘  fallback: in-process ORT CPU
 │  └ Telemetry PUB :5570 / control PULL :5571  │
 └─────────────────────────────────────────────┘
 hud.py (pygame, SUB :5570, PUSH :5571)    run dir: RT/runs/<stamp>_<tag>/
```

| Process | Interpreter | Owns | CPU pin (taskset) | Starts / stops |
|---|---|---|---|---|
| `rt_sim.py` | client venv `R/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python` (onnxruntime 1.23.2 CPU-only, MuJoCo 3.2.6, pyzmq 27.2, pygame) | MjSim, viewer, splicer, both clients | 2-3 | last up, first down |
| `wbc_gpu_server.py` | server venv `R/.venv/bin/python` | its own CUDA context (< 1 GB), 2 CUDA graphs | 4 | before `rt_sim` |
| GPU VLA server :5555 | server venv | 5090, cloudwalk checkpoint | (as is) | **never touched** |
| optional GPU VLA :5565 | server venv | about 7.3 GB on the 5090 (24 GB free) | 8-15 | only if DP1 = nvidia checkpoint (§6) |
| `server_g1.py` | tt-metal python_env via `MK` env | **/dev/tenstorrent/0 Tensix + DRAM**; one model per process (U.tt F11) | 8-15 | first inside `chip_session.sh` |
| `x280d.py` | new `RT/.venv-x280` (pyluwen 0.8.5, pyzmq, msgpack, numpy) | **L2CPU3 lifecycle**, keeper fd, LIM mailbox | 5 | after `server_g1` prints READY |
| `hud.py` | client venv | display only | 6-7 | any time |

**Device rules.**
- 5090: rtloop never sends `gear_wbc_action` to :5555. Its endpoint stays registered (`R/gr00t/eval/run_gr00t_server.py:106-113`) but unused.
- p150a: one `chip_session.sh` runs under one `bin/with-device.sh` flock. Inside it, start order is
  `server_g1` -> READY -> `x280d`, and teardown is the reverse. This keeps the guard's probe-and-reset
  (`~/experiments/gr00t/bin/with-device.sh:1-10`) strictly **before** any hart is released. See §5.2.

### 1.2 Per-tick data flow (tick k, wall deadline T_k = t0 + k·20 ms)

| # | Step | Where | Budget (measured basis) |
|---|---|---|---|
| 1 | Sleep until T_k. Record `late_us = now - T_k`. If more than one tick late: **slip**, re-anchor t0, log. | rt_sim | wake lateness p99 ≤ 0.33 ms (U.srv F6) |
| 2 | `VlaClient.poll(0)`. If a reply is ready: decode it (0.11 ms, U.sim m) and register it with `ChunkSplicer.on_arrival(chunk, k)`. | rt_sim | ≤ 0.2 ms |
| 3 | Trigger check (§3). If it fires: build the 9-key observation from o_k (the env obs returned at the end of tick k-1, released now per D8), cast dtypes, `send(NOBLOCK)`, stamp `k_obs = k`. | rt_sim | msgpack 0.11-0.14 ms + 0.92 MB send (U.sim m, U.srv F4) |
| 4 | `cmd35 = ChunkSplicer.select(k)`: row `k - k_obs` of the active chunk, or hold/ramp when starved. | rt_sim | < 0.05 ms |
| 5 | `env.step(cmd35)` = `WholeBodyControlWrapper.step` (`W/.../utils/n1_utils.py:35-48`): concat_action -> set_goal -> get_action. The patched `run_inference` sends **W-WBC v1** to the backend. Then safety check -> 10 × 2 ms physics + PD -> obs o_{k+1} -> `set_observation` -> `viewer.sync()` (robosuite base.py:478-479). | rt_sim + WBC backend | sim 4.9 ms + WBC round trip 0.6 ms (U.sim F9, U.srv F8) |
| 6 | Telemetry: write the tick record to the ring buffer, `PUB tick` (does not block), and every ~16 ticks `PUB ego` with a 160×120 thumbnail of the image sent. | rt_sim | ≤ 0.05 ms |
| 7 | Episode check: `info["success"]`, max 1440 ticks, fall check (pelvis z < 0.55 m; standing height is 0.752 m per U.sim F9), fault flags. | rt_sim | — |

Expected tick compute is 6-7 ms of 20 ms, leaving about 13 ms of slack. A WBC timeout (5 ms) plus the fallback still fits.

### 1.3 Wire schemas (exact)

All wires use msgpack, with numpy arrays encoded as `{"__ndarray_class__": True, "as_npy": np.save bytes}`
(`R/gr00t/policy/server_client.py:15-46`). REQ/REP over TCP on localhost. Timestamps are `time.monotonic_ns()`
(CLOCK_MONOTONIC, comparable across processes on this host).

**W-VLA request**: `{"endpoint": "get_action", "data": {"observation": OBS, "options": OPT}}`

The observation keys are derived at startup from `get_modality_config()` (`video.*` + `state.*` + language), exactly as the
serving probe trimmed them (`S/plan_research/serving/vla_latency.py:9-13`). Startup asserts they equal this table:

| key | dtype | shape | units / notes |
|---|---|---|---|
| `video.ego_view` | uint8 | (1,1,480,640,3) | RGB, the env's ego camera obs this tick (D6) |
| `state.left_leg`, `state.right_leg` | float32 | (1,1,6) each | rad |
| `state.waist` | float32 | (1,1,3) | rad (yaw, roll, pitch) |
| `state.left_arm`, `state.right_arm` | float32 | (1,1,7) each | rad |
| `state.left_hand`, `state.right_hand` | float32 | (1,1,7) each | rad |
| `annotation.human.task_description` | list[str] | len 1 | instruction (n_text 13 tokens, U.tt F3) |

- `OPT = {"rt": {"seq": u32, "gen": u32, "ep": u32, "k_obs": u32, "t_obs_ns": i64, "t_send_ns": i64}}`. U.srv rec4 says
  options are accepted and ignored by `Gr00tPolicy`. This is confirmed once in T-M1-3c; if it fails, send `None` and keep
  the meta on the client.
- Payload is 0.92 MB, against 9.22 MB for the full dict (U.srv F4).

**W-VLA reply**: `[ACT, INFO]`

| key | dtype | shape | units |
|---|---|---|---|
| `action.left_arm`, `action.right_arm` | float32 | (1,30,7) | rad, **absolute**, anchored to the state at k_obs (relative-to-absolute happens on the server, U.sim F3) |
| `action.left_hand`, `action.right_hand` | float32 | (1,30,7) | rad (left hand is constant 0 in the stats, U.sim F2) |
| `action.waist` | float32 | (1,30,3) | rad |
| `action.base_height_command` | float32 | (1,30,1) | m (0.74 in the stats) |
| `action.navigate_command` | float32 | (1,30,3) | vx, vy m/s (±0.5); wz rad/s (±1.0) |

- `INFO` is `{}` from the stock GPU server (`gr00t_policy.py:358`, U.srv rec6).
- From `server_g1` it is `{"rt": <echo>, "t_recv_ns", "t_reply_ns", "encode_ms", "device_ms", "decode_ms", "backend": <stamp label>}`.
- Rows 30..49 of the model's horizon are never used (U.as F0).

**Chunk record** (internal and logged; 4.2 KB):

`{seq u32, gen u32, ep u32, k_obs u32, t_obs_ns i64, t_recv_ns i64, k_arr u32, d u8, rows f32[30,35], vla_be u8}`

The 35 columns are in decode order: L arm 7 | R arm 7 | L hand 7 | R hand 7 | waist 3 | base_height 1 | nav 3.
**Index law: row i targets tick k_obs + i.** This matches sync: MultiStepWrapper applies row 0 on the first tick after the
observation (`R/gr00t/eval/sim/wrapper/multistep_wrapper.py:249-277`).

**W-WBC v1 request**:
`{"endpoint": "gear_wbc_action", "data": {"policy": "GR00T-WholeBodyControl-Balance" | "GR00T-WholeBodyControl-Walk", "obs": f32[1,516], "tick": u32, "seq": u32}}`

The obs is 6 frames of 86, oldest first, zero-filled at start. Each frame is
`[cmd·[2,2,0.5] 3 | height 1 | rpy_cmd 3 | ω·0.5 3 | gravity_b 3 | (q29 - default)·1.0 29 | dq29·0.05 29 | last_action 15]`
(`W/.../policy/g1_gear_wbc_policy.py:143-198`).

**W-WBC v1 reply**: `{"action": f32[1,15], "tick": u32, "seq": u32, "compute_us": f32}`

- The action is the raw policy output. The host computes `q15 = a·0.25 + default_angles` (`:268`).
- Endpoints `ping` and `info` -> `{"backend": "cuda-graph:rtx5090:prio-3" | "x280:l2cpu3:h0:<variant>@800MHz" | ...}`.
- Compatibility: a `PolicyServer` handler is called as `handler(**data)` (`server_client.py:139`). The stock
  `GearWbcPolicies.get_action(policy, obs)` therefore rejects `tick`/`seq`, so the client sends them only when `info`
  answers (v1 servers).

**W-LBC v2 (M4a only)**:
- Request: `{"tick": u32, "seq": u32, "in72": f32[72]}`, where `in72` = `q_body29` rad | `dq_body29` rad/s | pelvis `quat_wxyz 4` | `gyro 3` rad/s (body) | `nav 3` | `height 1` m | `waist_cmd 3` rad. The X280 does the 3-joint FK (U.sim F8).
- Reply: `{"q15": f32[15], "policy_id": u8, "compute_us": f32}`.

**Telemetry PUB** (`tcp://127.0.0.1:5570`, multipart `[topic, msgpack]`):

| topic | rate | fields |
|---|---|---|
| `tick` | 50 Hz | k, ep, t_ns, period_us, compute_us, late_us, slack_us, overrun, slip, sim_us, wbc_rtt_us, wbc_dev_us, wbc_be, wbc_pol (0 Balance / 1 Walk), wbc_fb, seq, row (= chunk age in ticks), rows_left, starved, inflight, splice, jump_rad, nav[3], pelvis_z, success |
| `req` | per request | seq, gen, ep, k_obs, k_arr, d, d_hat, s_eff, rtt_ms, ser_ms, srv{encode,device,decode}_ms, rows_used[2], outcome (applied / late_reject / stale_gen / timeout / error / skipped) |
| `ego` | ~3 Hz | seq, k_obs, uint8[120,160,3] raw (resized with INTER_AREA, no encoding) |
| `meta` | 1 Hz | processor map `{sim, vla, wbc}` (taken from the backends' `info` where available), config, rt_factor |

**Control PULL** (`tcp://127.0.0.1:5571`):
- `{"cmd": "swap_vla" | "swap_wbc", "to": <name>}`, applied at a tick boundary. A VLA swap bumps `gen`, so any in-flight reply is dropped.
- `{"cmd": "reset" | "pause"}`.

**X280 LIM mailbox (M3)**: LIM is 0x0800_0000-0x081D_FFFF, 1,966,080 B (U.x F11). Every touched line gets a full 64 B
zero write before any partial write (U.x F2). There is one writer per line.

| Address | Lines | Writer | Content |
|---|---|---|---|
| 0x0800_0000 | 64 KiB | host (LOAD) | monitor + kernel code/rodata/data/bss, 16 KiB stack |
| 0x0801_0000 | L0 | host | magic `0x57424331` ('WBC1'), cmd (0 NOP, 1 INFER, 2 LOAD, 3 FENCE_I, 4 CALL, 5 PARK), seq_req, policy (0/1), tick, flags |
| 0x0801_0040 | L1 | X280 | heartbeat u64, seq_done, state (0 boot, 1 idle, 2 busy, 3 fault), mcause, mepc, mtval, cycles_last |
| 0x0801_0080 | L2-L34 | host | obs f32[516] (2064 B, padded to 2112 B) |
| 0x0801_08C0 | L35 | X280 | action f32[15], seq_echo |
| 0x0801_0900 | L36 | X280 | ident: misa, vlenb, mhartid, mtime_hz, build_id, kernel_variant |
| 0x0801_0940 | L37 | host | LOAD descriptor (dst, len, crc32), weight-blob header offsets |
| 0x0801_1000 | — | host | weights: Balance + Walk fp16, 1,882,312 B, ending at 0x081D_C8C8 (U.x rec3) |

Per-tick transaction:
1. Host writes L2-L34 (write-combined block writes).
2. Host writes L0 with `seq_req` (UC strict write).
3. Host polls L1 `seq_done == seq_req` with UC reads, until a 4 ms deadline.
4. Host reads L35.
5. On the X280 side: spin on L0 -> `fence r,rw` -> compute -> write L35 -> `fence w,w` -> write L1.

Estimated mailbox overhead is 5-25 µs (U.x rec4, low confidence).

### 1.4 Rates

| Loop | Rate | Cost (measured unless marked est.) |
|---|---|---|
| physics + joint PD | 500 Hz (10 × 2 ms substeps) | 1.32 + 1.95 ms per tick (U.sim m) |
| control tick (sim + WBC host part) | **50 Hz** wall-paced | 5.26 ms with 2 cameras (U.sim F9) |
| WBC MLP | 50 Hz, 1 call/tick | ORT CPU 0.024-0.078 ms; GPU via ZMQ p99 0.92 ms (U.srv F3/F8); X280 0.1-0.6 ms est. (U.x F21) |
| VLA | about 3.1 Hz (s = 16 ticks) | 5090 trimmed 58.8 ms p50, 65-68 ms when paced (U.srv m); TT ≈ 42-45 ms est. (U.tt F10) |
| viewer sync | 50 Hz, inside `env.step` | **unmeasured**, closed by T-M1-3b |
| telemetry / HUD | 50 Hz PUB / 30 fps render | PUB send is µs-scale; the HUD is out of process |

### 1.5 Mapping across stages (only one row changes per milestone)

| Component | M1 | M2 | M3 | M4a | M4b |
|---|---|---|---|---|---|
| sim + tick loop + splicer | host `rt_sim` | = | = | = (robot I/O only) | = (robot I/O only) |
| VLA endpoint (W-VLA) | :5555 GPU | **:5557 Tensix** | = | = | = (+ chunk to DRAM) |
| WBC endpoint (W-WBC v1) | :5556 GPU | = | **:5558 X280** | **W-LBC v2** on X280 | X280 + splicer |
| obs history / Balance-Walk select / FK | host Python | = | = | **X280** | X280 |
| chunk splice + decode | host | = | = | = | **X280** |
| HUD processor map | 5090 / 5090 | Tensix / 5090 | Tensix / X280 | Tensix / X280 (full) | Tensix -> X280 |

---

## 2. Milestones, work packages and acceptance tests

Common environment for every command below:

```bash
export R=~/experiments/gr00t/Isaac-GR00T-n1.6 RT=~/experiments/gr00t/rtloop
export CPY=$R/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python   # sim/client
export SPY=$R/.venv/bin/python                                                                    # GPU servers
export XPY=$RT/.venv-x280/bin/python                                                              # x280d
cd $R && export PYTHONPATH=$RT:$R MUJOCO_GL=egl
# TT: source ~/experiments/gr00t/publish/worktree-mk-env.sh   (frozen snapshot 009de892 + python_env)
```

Thresholds live in `RT/gates.yaml`. `RT/scripts/analyze_run.py <run_dir> --gates <milestone>` writes
`<run_dir>/verdict.json` with PASS/FAIL per gate and exits non-zero on FAIL.

### M1 — async loop on the 5090 (≈ 9.5 d, about 7 on the critical path)

| WP | What | Files (all new) | Deps | Effort |
|---|---|---|---|---|
| WP-M1-00 | Scaffold the repo. **Rescue the /tmp prototypes** (`S/plan_research/simloop/{async_proto,sim_timing,payload_cost}.py`, `serving/{proto_server,vla_latency,capture_obs,obs_reset.pkl}`, `x280/tc/*`, `live_viewer.py`) into `RT/prototypes/`. Record the SHAs of the research clones (tt-isa-documentation, tt-bh-linux, tt-kmd 2.11.0, fw 19.12.0). Add `env.sh` and `preflight.py`. Preflight checks: ports 5556-5559 and 5570-5571 are free; :5555 answers `ping` (read-only); GPU has ≥ 4 GB free; `DISPLAY` is set; the sim imports with `torch.cuda.is_initialized() == False`. | `RT/{README.md,env.sh,gates.yaml}`, `RT/scripts/preflight.py`, `RT/prototypes/**` | — | 0.5 |
| WP-M1-01 | `schema.py`: dims, joint-group order, wire dataclasses, codec (reuse `MsgSerializer`). `splicer.py`: a pure-function ChunkSplicer (§3, kept C-portable: no object state beyond plain arrays). Unit tests. | `RT/rtloop/{schema,splicer}.py`, `RT/tests/test_splicer.py` | 00 | 1 |
| WP-M1-02 | `wbc_gpu_server.py` (server venv) wraps `gr00t.policy.gear_wbc_policy.GearWbcPolicies`. REP :5556 in the **main thread**, so the fixed `self.stream` in `CudaGraphPolicy` (`gear_wbc_policy.py:83,90`) is the correct stream (U.srv F7). It serves W-WBC v1 with tick/seq echo, `compute_us`, and `info`/`ping`. Boot self-check vs ORT: max\|err\| < 1e-4, the same as the hook's check. Client side, `wbc_backends.py`: `OrtCpuBackend` (CPU provider, intra_op 1), `ZmqWbcBackend` (RCVTIMEO 5 ms, lazy-pirate reconnect), the fallback chain, and `install(G1GearWbcPolicy)`, which patches `load_onnx_policy`. | `RT/scripts/wbc_gpu_server.py`, `RT/rtloop/wbc_backends.py` | 01 | 1 |
| WP-M1-03 | `vla_client.py`: its own zmq.Context and REQ socket, `send(NOBLOCK)`, `poll(0)`, a 1000 ms reply timeout with lazy-pirate recreate, and `build_vla_obs()` from the modality config. `mock_vla_server.py`, wire-identical to PolicyServer, has modes `hold` (state-anchored constant), `detfn` (a deterministic function of the obs: arm sinusoids plus a nav schedule that exercises Walk), `replay <requests.jsonl>`, `--latency-ms 60 --jitter-ms 5`, and `--stall-after N --stall-s T`. | `RT/rtloop/vla_client.py`, `RT/scripts/mock_vla_server.py` | 01 | 1 |
| WP-M1-04 | `sim_env.py` builds the env: copy of `live_viewer.py` `make_env_fn` and `patch_viewer_camera`, `onscreen=True`, a seeded `env.reset(seed=s)`, and observable tweaks re-applied after each reset (U.sim F10/F14). `rt_sim.py` is the tick loop of §1.2. Modes: `rt`, `vl --d-inj D`, `sync`, `sync-hold` and `noskip`; flags `--s`, `--blend B`. It also handles episode control, a `SystemExit` guard around `env.step` (the safety monitor calls `sys.exit(1)`, `W/.../envs/g1/utils/joint_safety.py:451-454`), re-anchoring after reset, the CUDA guard and `--pin`. | `RT/rtloop/sim_env.py`, `RT/rtloop/viewer.py`, `RT/scripts/rt_sim.py` | 01-03 | 2 |
| WP-M1-05 | `telemetry.py`: the PUB, a tick ring buffer written to `ticks.npz`, `requests.jsonl`, `episodes.jsonl`, `--dump-obs516` (2 KB/tick, used for the M3 parity sets) and `--dump-vla-obs N` (raw VLA requests, used for M2 goldens). `hud.py` is the pygame HUD of §4.3. | `RT/rtloop/telemetry.py`, `RT/scripts/hud.py` | 01 (parallel with 04) | 1.5 |
| WP-M1-06 | `analyze_run.py`: gates, plus plots (Gantt, tick period, d, chunk age, WBC round trip). `eval_paired.py`: conditions × seeds, Wilson CI, exact McNemar (§4.4). | `RT/scripts/{analyze_run,eval_paired}.py` | 05 | 1 |
| WP-M1-07 | `demo.sh m1\|m2\|m3`: preflight -> backends -> HUD -> `rt_sim`, with a teardown trap. `record_demo.sh` runs ffmpeg x11grab of the viewer and HUD windows **out of process**, following the LIBERO demo precedent (`libero_eval/n16/demo/raw_*_window.mp4.ffmpeg.log`). | `RT/scripts/{demo.sh,record_demo.sh}` | 04, 05 | 0.5 |
| WP-M1-08 | Tests: VL equivalence, obs schema, faults, CUDA guard. | `RT/tests/{test_vl_equivalence,test_obs_schema,test_faults,test_cuda_guard}.py` | 04 | 1 |
| WP-M1-09 | Evaluation runs and the M1 report. | `P/results/M1.md` | 06-08 | 1 + about 8 h machine time |

Acceptance tests for M1:

| ID | Checks | Threshold | Command |
|---|---|---|---|
| T-M1-1 | splicer logic | all pass: index law; drop on gen mismatch; late-reject at d ≥ 26; starvation hold + nav ramp to 0 over 5 ticks; fault at 50 starved ticks; B = 0 is identity; B ∈ {3,5} weights `(j+1)/(B+1)`; hands and base height switch hard | `$CPY -m pytest $RT/tests/test_splicer.py -q` |
| T-M1-2 | **VL equivalence (deterministic)** with the mock `detfn` VLA, `--wbc cpu` (ORT CPU, so the WBC is deterministic), seed 7, 1440 ticks | `vl --d-inj 0 --s 20` vs `sync`: joint trajectory **max\|Δq\| == 0** at every tick, identical success flags | `$CPY -m pytest $RT/tests/test_vl_equivalence.py -q` |
| T-M1-3a | obs and wire parity vs the stock path | 9 keys, dtypes and shapes equal `prototypes/obs_reset.pkl` trimmed per the modality config; one real :5555 call returns 7 keys of shape (1,30,D) float32 | `$CPY -m pytest $RT/tests/test_obs_schema.py -q` |
| T-M1-3b | **RT timing soak**: 5 min, real VLA :5555 + GPU WBC :5556 + viewer + HUD + recording | tick period p99 ≤ 21.0 ms and max ≤ 30 ms; overruns (compute > 20 ms) ≤ 0.5%; slips = 0; WBC round trip p50 ≤ 1.0 ms and p99 ≤ 2.0 ms; WBC timeouts ≤ 0.1%; d p50 ∈ {3,4} and d p99 ≤ 12; starved ticks = 0; no `SAFETY` print; viewer-sync p99 ≤ 3 ms (else viewer at 25 Hz, R9) | `$RT/scripts/demo.sh m1 --duration 300 --seeds 1000-1009 --tag soak && $CPY $RT/scripts/analyze_run.py $RT/runs/latest --gates m1_rt` |
| T-M1-3c | the options pass-through | the server returns no error with `options.rt` set | part of T-M1-3a |
| T-M1-4 | faults (mock VLA on :5559) | VLA stall 2 s: `starved` set within 1 tick of row 30, nav \|cmd\| < 0.05 within 5 ticks (Balance), fault flag at 50 ticks, recovery on the next arrival, **process alive, 0 slips**. WBC server killed: fallback to ORT CPU on the same tick (≤ 5 ms), HUD shows `cpu-fallback`, reconnect after restart. | `$CPY -m pytest $RT/tests/test_faults.py -q` |
| T-M1-5 | CUDA-free sim | `torch.cuda.is_initialized() is False` after build, after 3 resets and at exit; no EGL artefact (ego frame not all-black / NaN) | `$CPY -m pytest $RT/tests/test_cuda_guard.py -q` |
| T-M1-6 | **closed-loop screen**: 50 paired seeds | async-RT success ≥ sync-frozen − 15 pts and exact McNemar p ≥ 0.05 for "worse"; falls(async) ≤ falls(sync) + 2 | `$CPY $RT/scripts/eval_paired.py --conditions sync,async --seeds 1000-1049 --vla tcp://127.0.0.1:5555 --wbc gpu` |
| T-M1-7 | **M1 exit (confirm)**: 160 paired seeds | non-inferiority margin 10 pts (one-sided 95%) vs sync-frozen; plus the C0 anchor of §4.4 | same as T-M1-6 with `--seeds 1000-1159 --conditions sync,sync-hold,async` |

**Go/no-go for M1:** T-M1-1..5 must pass before any demo. T-M1-6 must pass to call M1 "demo-ready". T-M1-7 is the formal
exit, and it may run in parallel with M2.

### M2 — VLA on Tensix (≈ 5.5 d; the CPU parts can start on day 1)

| WP | What | Files | Deps | Effort |
|---|---|---|---|---|
| WP-M2-01 | `g1_n16.py` with `install()`, cloned from `TT/libero/n16.py:176-229`. It re-points `HF_REPOS["n16"]` and `HF_SNAPSHOT_SHAS["n16"]` at the G1 checkpoint (**default** `nvidia/GR00T-N1.6-G1-PnPAppleToPlate@897d0313`, which is local and whose index total_size is 6,573,217,664, the same byte count as the all-BF16 base. That implies no conversion is needed; the dtype scan is part of T-M2-1. See DP1.). It adds layout `g1` = `dataclasses.replace(gr1, name="g1", embodiment_tag="unitree_g1", embodiment_id=8, valid_slice=(30,35))` (U.tt F3, proven on CPU), clears the normalization lru caches, and asserts the spec: id 8, ego_view, state 43, action 35, horizon 30 (U.tt F4). It uses a dedicated `cache_root=~/.cache/gr00t-tt-g1/` (U.tt F6). Also `encode_obs`, `decode_actions`, `backend_stamp`. CPU test: host encode is bit-exact vs the Isaac processor on 10 recorded obs (reuse `S/plan_research/parity_*.py`). | `RT/tt/g1_n16.py`, `RT/tests/tt/test_g1_host_cpu.py` | — | 1 |
| WP-M2-01b | *Only if DP1 = cloudwalk*: a CPU F32->BF16 snapshot conversion keeping the 2-shard map. Check total_size == 6,573,217,664 and all dtypes BF16 (U.tt rec2). | `RT/tt/convert_bf16.py` | — | 0.5 |
| WP-M2-02 | G1 golden: fp32 GPU reference on 10 raw obs from `--dump-vla-obs` (M1.1). Use a variant of `R/tt_golden/dump_golden_n16.py`, run as a **separate** server-venv process; the 5090 has about 24 GB free, and the :5555 server must not be touched. Save the noise and `action_pred_norm`, plus a bf16 GPU run to give the deviation scale. | `RT/tt/make_g1_golden.py` -> `~/experiments/gr00t/golden/n16_g1/` | M1-05 (or a stock-path capture) | 1 |
| WP-M2-03 | `server_g1.py`, a copy of `TT/libero/server_n16.py` (wire layer, poller loop, bind-before-open, `info`). It uses fresh noise from a generator seeded with `--seed` (U.tt F15), a boot self-check vs the G1 golden (`--min-pcc 0.999`), a JSONL per call, and returns `INFO` per §1.3. Port :5557. One model and one device owner in the main thread (U.tt F11/F12). | `RT/tt/server_g1.py` | 01, 02 | 1 |
| WP-M2-04 | Device session S1 under `chip_session.sh --vla tt`: cold boot, capture, golden check, then a 500-call latency bench with recorded obs. | `RT/scripts/chip_session.sh`, `RT/tt/bench_g1.py` | 03 | 0.5 (device) |
| WP-M2-05 | Sync closed loop on TT with the **stock** client (`rollout_policy.py --policy_client_port 5557 --n_envs 1`), 50 episodes. Same-checkpoint GPU baseline: the :5555 server if DP1 = cloudwalk, else a second GPU server on :5565. | — (commands only) | 04 | 1 (machine) |
| WP-M2-06 | Async on TT: `demo.sh m2`, a 30-min soak and the 50-seed paired eval. | `P/results/M2.md` | 05, M1 exit | 1 |
| WP-M2-07 | (demo extra) VLA hot-swap GPU <-> TT between episodes via the HUD key `v` (control PULL, gen bump). | `RT/rtloop/control.py` | 06 | 0.5 |

Acceptance tests for M2:

| ID | Checks | Threshold | Command |
|---|---|---|---|
| T-M2-1 | host parity (CPU) | `input_ids`, `pixel_values`, state and 7 decoded groups bit-exact (max\|d\| = 0) on 10 obs; spec assertions | `source $MKENV && python -m pytest $RT/tests/tt/test_g1_host_cpu.py -q` |
| T-M2-2 | device fidelity at boot | `actions_pred_valid` PCC ≥ **0.999** vs fp32 golden on ≥ 5 obs; per-group max\|d\| ≤ 2× the GPU-bf16-vs-fp32 max\|d\| of the same obs; 5 replays with the same noise bit-identical | `bin/with-device.sh $RT/scripts/chip_session.sh --vla tt --check-only` |
| T-M2-3 | latency | server `total_ms` p50 ≤ 45 and p99 ≤ 50 over 500 calls; client round trip p50 ≤ 48 ms | `... --bench 500` -> `$RT/runs/<id>/tt_bench.json` |
| T-M2-4 | sync closed loop | success (50 eps) ≥ GPU same-checkpoint − 10 pts; 0 server errors or hangs | `GR00T_WBC_SERVER=127.0.0.1:5556 $CPY gr00t/eval/rollout_policy.py --policy_client_host 127.0.0.1 --policy_client_port 5557 --n_envs 1 --n_episodes 50 --max_episode_steps 1440 --n_action_steps 20 --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc` (the hook sends `{policy, obs}`, which a v1 server accepts) |
| T-M2-5 | async RT soak, 30 min | all T-M1-3b gates, with d p50 ≤ 3 and d_hat init 3; 0 TT errors; HUD `vla=tensix` read from `info` | `demo.sh m2 --duration 1800 && analyze_run.py ... --gates m2_rt` |
| T-M2-6 | paired eval, 50 seeds | async(TT) ≥ async(GPU, same checkpoint) − 15 pts (screen) | `eval_paired.py --conditions async@gpu,async@tt --seeds 1000-1049` |
| T-M2-7 | non-reentrancy and stale state | 200 alternating obs A/B/A/B: outputs equal the single-call outputs (fixed noise) | `bench_g1.py --abab 200` |

**Go/no-go for M2:** T-M2-1..3 gate any closed-loop run; T-M2-4..6 gate the demo.

### M3 — WBC MLP on the X280 (≈ 10 d; about 4.5 d of CPU-only work runs in parallel with M1/M2)

CPU-only work (no device):

| WP | What | Files | Deps | Effort |
|---|---|---|---|---|
| WP-M3-01 | Freestanding C forward pass: estimator 516-256-256-35 ELU, L2-normalise, actor 121-512-256-256-15 (U.x F16), with an `expf` polynomial and no libcalls. Weight packer ONNX -> blob in fp32 or fp16 with a header and CRC32. x86 build plus a ctypes parity test vs ORT on 10k logged obs516 and 256 random. | `RT/x280/kernel/{wbc_mlp.c,wbc_mlp.h}`, `RT/x280/tools/pack_weights.py`, `RT/tests/x280/test_kernel_x86.py` | M1-05 (logs) | 1.5 |
| WP-M3-02 | X280 firmware: `crt0.S` (harts ≠ 0 -> wfi), `lim.ld` (0x0800_0000, 0x1E0000), `monitor.c` (L0/L1 protocol: LOAD / FENCE_I / CALL / INFER / PARK; trap handler dumps mcause/mepc/mtval; RNMI park handler), CSR 0x7c1 = 0, mstatus.FS/VS, and an ident line. Kernel variants: v0 scalar, v1 RVV e32 (fp16 widened with integer ops until Zvfh is probed, U.x rec6). Toolchain: clang-20/lld-20 from the prototype in `S/plan_research/x280/tc` (U.x F18). Image ≤ 48 KiB. | `RT/x280/fw/{crt0.S,lim.ld,monitor.c,Makefile}` | — | 2 |
| WP-M3-03 | `x280d.py`. Holds the keeper legacy fd `os.open('/dev/tenstorrent/0', O_RDWR)` (U.x F7/F10) and uses pyluwen `PciChip` noc_read/noc_write (allocated 2 MiB windows only; never the "Hardcoded" index APIs, U.x F9). Subcommands `probe`, `lim-init`, `load`, `release --tile 3`, `ping`, `serve --port 5558`, plus `--dry-run`, which prints every NoC op. **Interlocks** (it refuses to proceed unless): WAYENABLE reads 0; L2CPU_RESET bits [3:0] = 0xF and bit 7 = 0 before release; PLL4 reads fbdiv 64 / postdiv 1 before and after; the L2CPU clock is not gated (telemetry). New venv `RT/.venv-x280` (pyluwen 0.8.5 matches `~/.tenstorrent-venv`; that venv has no pyzmq). | `RT/x280/x280d.py`, `RT/x280/requirements.txt` | 02 | 2 |
| WP-M3-04 | Mailbox simulator: `x280d --backend sim` runs the x86 kernel behind the same L0-L37 struct in a shared-memory file, so the whole M3 software path (rt_sim -> :5558 -> mailbox -> kernel) is tested without hardware. | `RT/x280/mailbox_sim.py` | 01, 03 | 0.5 |

Device experiments. They run strictly in order, each in its own `chip_session.sh` and each with its own go/no-go
(U.x rec8; safety rules in §5.2):

| WP | What | Effort |
|---|---|---|
| WP-M3-E0 | Read-only probe while `server_g1` is running: L2CPU_RESET, CCACHE0_CONFIG/WAYENABLE, PLL4, hart status. Read CCACHE0_CONFIG via both (8,7) and (8,29) to resolve the coordinate question (U.x F13 / R22). | 0.25 |
| WP-M3-E1 | LIM full-line zero-fill of everything to be touched, then a pattern write/readback, then one partial 4 B UC write to an initialised line, all while the TT VLA bench loop runs. No release. | 0.5 |
| WP-M3-E2 | **The one release**: P2-P8 of U.x rec1 on L2CPU3 (bit 7), with the monitor and RNMI/trap handlers pre-loaded. Hello: heartbeat, ident. Budget: 1 chip reset. | 0.5-1 |
| WP-M3-E3 | Power policy: with the heartbeat running, open/close a second fd 10×, then stop/start `server_g1` while the keeper fd is held. | 0.25 |
| WP-M3-E4 | Microbench via the monitor: GEMV 256×516 from LIM, fp16/fp32 × scalar/RVV; mailbox round trip from x280d. Pick the kernel variant. | 0.5 |
| WP-M3-E5 | WBC on X280: on-device parity, timing, `serve :5558`, closed loop, soak. | 1.5 |
| WP-M3-06 | Demo: `demo.sh m3`. HUD key `w` hot-swaps WBC X280 <-> GPU mid-episode (stateless, D12). "Proof switch": RNMI-park the hart, and the HUD must show the fallback to GPU within 1 tick, then the reload. | 0.5 |

Acceptance tests for M3:

| ID | Checks | Threshold | Command |
|---|---|---|---|
| T-M3-1 | kernel correctness (x86) | fp32 blob vs ORT max\|err\| ≤ **1e-4**; fp16 blob vs ORT ≤ **1e-2** (host measurement 6.8e-3 / 7.4e-3, U.x F17) on 10k logged obs + 256 random, Balance and Walk | `$CPY -m pytest $RT/tests/x280/test_kernel_x86.py -q` |
| T-M3-2 | software path without hardware | `demo.sh m3 --x280-backend sim`: all T-M1-3b gates; WBC round trip p99 ≤ 2 ms | `demo.sh m3 --x280-backend sim --duration 120 && analyze_run.py ... --gates m3_sim` |
| T-M3-E0 | probe | L2CPU_RESET = 0x0F with bit 7 = 0; WAYENABLE = 0; PLL4 fbdiv 64, postdiv 1; TT latency unchanged | `$XPY $RT/x280/x280d.py probe --json $RT/runs/<id>/e0.json` |
| T-M3-E1 | LIM non-interference | 100% pattern match; no fault on the partial write; TT p50 latency change ≤ 1%; TT outputs bit-identical with fixed noise | `$XPY x280d.py lim-init --pattern --while-bench` |
| T-M3-E2 | release and monitor | heartbeat increments; `vlenb == 64`; misa has V; mcycle/mtime ≈ 16 ± 1 (800 MHz / 50 MHz); TT still serving | `$XPY x280d.py release --tile 3 && $XPY x280d.py ping --n 1000` |
| T-M3-E3 | clock continuity | no heartbeat gap > 10 ms across 10 fd open/close cycles and one TT restart while the keeper fd is held; PLL4 back at 800 MHz | `$XPY x280d.py powertest` |
| T-M3-E5a | on-device parity | X280 output vs the x86 build of the same blob: max\|err\| ≤ 1e-5; vs ORT ≤ 1e-2 (fp16) on 10k obs | `$XPY x280d.py parity --obs $RT/runs/<m1>/obs516.npz` |
| T-M3-E5b | timing | X280 compute (mcycle) p99 ≤ **2.0 ms** (v1 target ≤ 0.6 ms); x280d round trip p99 ≤ **3.0 ms**; timeouts ≤ 0.1% | `demo.sh m3 --duration 600 && analyze_run.py ... --gates m3_rt` |
| T-M3-E5c | closed loop | 50 paired seeds: async(TT VLA + X280 WBC) ≥ async(TT VLA + GPU WBC) − 10 pts; falls ≤ M2 + 2 | `eval_paired.py --conditions async@tt+gpuwbc,async@tt+x280 --seeds 1000-1049` |
| T-M3-6 | hot-swap and soak | 30 min with a swap every 10 s: 0 slips; tick p99 ≤ 21 ms; \|Δa\| at swap ≤ 1e-2; 0 X280 faults | `demo.sh m3 --duration 1800 --swap-wbc-every 10` |

**Go/no-go for M3:**
- T-M3-1, T-M3-2 and the dry-run review gate **any** device work.
- Each E-step gates the next one.
- E5a-c gate the demo.

### M4 — options, in recommended order

| WP | What | Key tests | Effort |
|---|---|---|---|
| WP-M4a | **Whole lower-body controller on X280** (W-LBC v2). The X280 keeps the 6×86 history and last action, does Balance/Walk select (\|cmd\| < 0.05, `g1_gear_wbc_policy.py:257`), the 3-joint torso-RPY FK, and reproduces the 1-tick cmd lag and dt = 0.02 (U.sim F7/F8). Then a **free-running mode**: the X280 ticks off mtime (50 MHz) on latest-wins lowstate/lowcmd mailboxes, which is DDS semantics (U.sim F5). | obs-vector reconstruction max\|err\| ≤ 1e-6 vs Python on logged episodes; q15 ≤ 1e-3; closed loop ≥ M3 − 10 pts | 4-5 |
| WP-M4b | **On-chip VLA -> X280 handoff.** `server_g1` copies the normalized `action_pred` into a persistent DRAM buffer at a fixed address and bumps `chunk_seq`. The X280 reads it through an uncached 128 GiB window with translated coordinates from `ttnn.cluster.get_dram_bank_table`, checks seq for torn reads, then de-normalizes, splices and runs the WBC; the host only does robot I/O (U.x rec5). The buffer's tile layout must be confirmed first (U.x gap). | X280 chunk == host-decoded chunk (max\|d\| ≤ 1e-6); 10k chunks with 0 torn reads; VLA-done -> X280-sees-seq ≤ 1 ms | 4-6 |
| WP-M4c | **RTC v2** (§3.2): v2a guided on the 5090 as the reference; v2b gradient-free soft replacement, with a TT megakernel Euler-stage blend (`TT/tt/action_head.py:750-770`, plus Y/W/z inputs). | §4.4 adoption rule | 5-10 |
| WP-M4d | (optional) A 3-process robot split: sim-as-robot with shared-memory lowstate/lowcmd. This makes M4a's free-running mode meaningful. | T-M1-3b gates | 2 |

---

## 3. Chunk scheduling policy

### 3.1 v1 — aligned receding horizon (default)

| Parameter | Value | Basis |
|---|---|---|
| Horizon used | H = 30 rows (model rows 30-49 never used) | U.as F0 |
| Index law | row i ↔ tick k_obs + i; chunk age = `k - k_obs` = the row executed | U.as rec1; multistep_wrapper.py:256-277 |
| Trigger | At the start of tick k (D8), if no request is in flight and `k - k_obs_active ≥ s_eff` -> send o_k. The first request of an episode goes at tick 0; ticks before the first arrival run **hold**: initial upper-body q, nav 0, height 0.74, so the WBC is in Balance. One request in flight; a trigger that fires while one is in flight is logged as `skipped`, never queued. | U.as rec2 |
| s_cfg / s_eff | s_cfg = 16; `s_eff = clamp(s_cfg, d_hat, 30 - d_hat - m)` with m = 2 | U.as rec2 (RTC d ≤ s ≤ H − d) |
| d_hat | max of the last 10 measured d (ticks); initial value 5 (5090) / 3 (TT) | U.as F5 (RTC Alg. 1) |
| Latency compensation | none beyond the index law. Arms are already absolute targets anchored at k_obs, so **never re-anchor** | U.as F1 |
| On arrival (tick k_arr, d = k_arr − k_obs) | drop if `gen ≠ current` (reset, language change, VLA swap); **late-reject** if d ≥ 26 (fewer than 4 usable rows) and retrigger immediately; otherwise use the new chunk's row `k − k_obs_new` from tick k_arr on | U.as rec3 |
| Blend | B = 0 default. Option B ∈ {3, 5}: `a = (1−w)·old + w·new`, `w_j = (j+1)/(B+1)` on arms, waist and nav only; hands and base_height switch hard. Adopt only if the boundary jump drops without a success loss (§4.4). | U.as rec3, risk on hands |
| Starvation (row ≥ 30) | hold the last executed arm/waist/hand/height targets; ramp nav linearly to 0 over 5 ticks (puts the WBC in Balance); `starved` lights on the HUD; after 50 starved ticks (1 s) the episode is flagged **fault** (it keeps holding and is excluded from timing gates) | U.as rec4; mirrors the 1 s goal timeout (`W/.../g1_decoupled_whole_body_policy.py:94-109`) |
| Timeouts | VLA reply 1000 ms -> lazy-pirate socket recreate, outcome `timeout`, retrigger; WBC 5 ms -> fallback chain | U.srv F9 |
| Goal timeout safety | `set_goal` is called every tick, so the monotonic 1 s timeout never fires; the reset stall happens outside the paced loop | U.sim F13 |

Expected behaviour on the 5090 (d ≈ 3-4): rows [d, d+16), roughly [3, 19], are executed per chunk. That is inside the
sync baseline's rows 0-19. The mean chunk age is about d + 8 ≈ 11 ticks (220 ms), and GPU duty is about 19%. With
s_eff = 16, starvation needs d > 12 ticks (240 ms) (U.as m).

The splicer core is load-bearing, so its exact semantics are given here (Python first; ported to C in M4a):

```python
def select(k, st, cfg):                      # st: plain fields {act, pending, blend_from, blend_t0, ...}
    if st.pending is not None and st.pending.k_arr <= k:   # registered at tick k_arr by on_arrival()
        st.blend_from, st.blend_t0, st.act, st.pending = st.act, k, st.pending, None
    act = st.act
    if act is None: return hold(st)          # before first arrival
    i = k - act.k_obs
    if i >= 30: return starve(st, k)         # hold + nav ramp, fault after 50
    cmd = act.rows[i].copy()                 # never mutate a stored chunk
    if cfg.B and st.blend_from is not None and (j := k - st.blend_t0) < cfg.B:
        io = k - st.blend_from.k_obs
        if io < 30:
            w = (j + 1) / (cfg.B + 1)
            cmd[BLEND] = (1 - w) * st.blend_from.rows[io][BLEND] + w * cmd[BLEND]
    return cmd
```

### 3.2 v2 — real-time chunking (RTC), a later option

- Insertion point: `Gr00tN1d6ActionHead.get_action_with_features` (U.as F2), with H = 30 and s = 15.
- The prefix Y is the not-yet-executed rows of the active chunk, re-encoded **relative to the new obs state** with the
  per-index (30,7) statistics. The exponential mask W is that of RTC Eq. 5; for d = 3 it is
  `[1,1,1,.815,.655,.518,.402,.305,.224,.158,.105,.065,.035,.015,.004,0…]`. β = 5 (U.as rec7, rec8).

Variants:

| Variant | Where it runs | Notes |
|---|---|---|
| **v2a** guided (VJP) | 5090 only | the reference; breaks inference_mode and CUDA graphs |
| **v2b** gradient-free soft replacement | portable to the TT megakernel | `x += W·((1−τ)z + τY − x)` before each of the 4 Euler steps |
| **v2c** training-time RTC | needs a fine-tune with per-token AdaLN timesteps | best long-term fit for Tensix |

- Caveats: the Isaac guide wants chunks of 32 or more (G1 has 30), and IT-RTC degrades at H = 30 with high delay (U.as F3, F8).
- Rule: **v2 is adopted only if it beats v1 at d_inj ≥ 5, or lowers the boundary jump with no success loss** (§4.4).

---

## 4. Measurement, logging and evaluation

### 4.1 What is measured (all on the CLOCK_MONOTONIC timebase)

| Quantity | Definition |
|---|---|
| tick period | start(k) − start(k−1) |
| tick compute | end(k) − start(k) |
| lateness | start(k) − T_k |
| slack | T_{k+1} − end(k) |
| overrun / slip | compute > 20 ms / lateness > 20 ms (then re-anchor) |
| WBC round trip | client send -> recv around `run_inference` |
| WBC device time | `compute_us` reported by the backend (CUDA: host-timed graph replay + sync; X280: `cycles_last / f_MHz`) |
| VLA latency | t_recv − t_send (client); `encode/device/decode_ms` (TT server) |
| d | k_arr − k_obs (ticks) |
| chunk age | row executed = k − k_obs (ticks; × 20 ms) |
| boundary jump | \|\|old_row − new_row\|\|₂ over arms + waist at the splice tick; plus velocity-direction cosine and intra-chunk acceleration, per `R/getting_started/real_world_deployment.md:283-362` (U.as F3) |
| Walk/Balance flips | count of policy switches per episode (U.sim F15) |

### 4.2 Files per run (`RT/runs/<YYYYmmdd-HHMMSS>_<tag>/`)

- `config.json`: CLI, git SHAs of RT/R/W/MK, checkpoint and sha, endpoints, backend `info` stamps, CPU pins.
- `ticks.npz`: a structured array, one row per tick, with the `tick` topic fields plus cmd35 and q43.
- `obs516.npz` (opt-in).
- `requests.jsonl`, `episodes.jsonl` (seed, success, ticks, falls, starved, faults, flips, time-to-success).
- `vla_obs/` (opt-in raw requests for goldens).
- Plots and `verdict.json`.

### 4.3 HUD (`hud.py`, 30 fps, 1280×720, separate process)

1. **Processor map**, coloured by device: Host CPU | RTX 5090 | Blackhole [Tensix | X280]. Each box is lit by the
   component that runs there and labelled from the backends' `info` stamps, not from the CLI, so the demo cannot claim
   what is not running. For the stock :5555 server, the label is configured and marked "(cfg)".
2. **VLA panel**:
   - last, p50 and p99 latency over 50 requests, with server-side encode/device/decode when the backend provides them;
   - d, d_hat, s_eff, an in-flight spinner;
   - skipped / late / timeout counters.
3. **Chunk bar**: 30 cells, with executed rows filled, the cursor, and the [d, s+d) window. Shows age in ms, rows left,
   and a large red STARVED when starved.
4. **WBC panel**:
   - tick period now / p99 and its jitter std;
   - round trip now / p99 and device µs;
   - Balance/Walk indicator;
   - fallback, overrun and slip counters;
   - real-time factor.
5. **Timeline**, last 4 s:
   - tick compute bars (red if > 20 ms);
   - VLA request bars (send -> recv);
   - splice markers;
   - WBC round-trip sparkline.
6. **Ego thumbnail** of the image the active chunk was computed from, stamped with k_obs. Plus episode id, seed,
   success and falls.

Keys: `v` swaps VLA, `w` swaps WBC, `r` resets, `p` pauses.

The HUD can never stall the loop: PUB drops messages when nobody is subscribed or the high-water mark is reached.

### 4.4 Evaluation protocol

- **Pairing.** Scene seeds `s = 1000..1000+N−1` via `env.reset(seed=s)` (U.sim F14). The same seed list is used for every
  condition. VLA flow noise:
  - The stock :5555 server draws fresh unseeded noise (`gr00t_n1d6.py:310-314`) and cannot be changed, so on the GPU
    pairing is by scene only.
  - On TT, noise comes from a generator seeded with `--seed` (it is not reset per episode).
- **Episode rule** (identical in all conditions): success = `info["success"]` at any tick ≤ 1440. Episodes are not
  terminated early on success (`terminate_on_success=False`, multistep_wrapper.py:130). Falls and faults are recorded
  separately.

| Cond. | Name | Definition | Role |
|---|---|---|---|
| C0 | stock | unmodified `rollout_policy.py --n_envs 1 --n_action_steps 20`, unseeded, 50 eps | anchor: C1 must agree within the 95% CI (validates the reimplementation) |
| C1 | sync-frozen | `rt_sim --mode sync`: sim frozen during each call, rows 0-19 | the user-requested sync baseline (optimistic: hides latency) |
| C2 | sync-hold | sim keeps ticking and holds the last targets during the call, then rows 0-19 from arrival | the honest synchronous baseline (as in the RTC paper) |
| C3 | async-noskip | overlap at s = 16, but execute from row 0 on arrival | negative control for latency skip |
| C4 | **async v1** | §3.1, B = 0, RT mode | the demo policy |
| C5 | async v1 + blend | B = 5 | variant |
| C6 | v2a/v2b | M4c | option |

- **Virtual-latency sweeps**: C4 and C1 in VL mode at d_inj ∈ {0, 3, 5, 8, 12}, i.e. 0, 60, 100, 160 and 240 ms
  (emulating sync, 5090, TT and slower backends). The VL d = 0, s = 20 case must match C1 (T-M1-2 bit-exact with the
  mock; within the CI with the real VLA).
- **Tiers**:

| Tier | Episodes | Use |
|---|---|---|
| smoke | 5 | after any change |
| screen | 50 paired/arm | Wilson 95% CI about ±14 pts; catches gross regressions, decides variants |
| confirm | **160 paired/arm** | detects 45% vs 30% at α = 0.05, power 0.8 (U.as m); used for exit claims |

  A 10-point difference needs about 350 per arm. The README warns of ±15% run-to-run variance (U.as rec13), so
  10-20-episode comparisons are never used.
- **Statistics**: Wilson 95% CI per arm and the exact McNemar test on discordant pairs.
  - Non-inferiority: the one-sided 95% bound of (async − sync) is > −10 pts.
  - Secondary metrics: time-to-success, falls, starved ticks, boundary jump, flips.
- **Machine time**: RT ≈ 32 s/episode (28.8 s + 2.8 s reset), so 160 eps ≈ 85 min/arm. VL ≈ 12 s/episode, so ≈ 32 min/arm.

---

## 5. Risk register

### 5.1 Register (each row names the WP and the test that closes it)

| # | Risk | Evidence | Mitigation | Closes (WP -> test) |
|---|---|---|---|---|
| R1 | WBC head-of-line blocked behind the VLA on the single REP socket | measured p99 67 ms, 35.7% of ticks > 20 ms (U.srv F1); the loop is serial (`server_client.py:117-143`) | WBC in its own process/port (D3); the :5555 WBC endpoint is never used | WP-M1-02 -> T-M1-3b |
| R2 | A CUDA context in the sim corrupts EGL; **also, `CUDA_VISIBLE_DEVICES=""` crashes robosuite EGL init**: `egl_context.py:39-55` parses `""` and `int("")` raises. U.as rec5 suggests setting it. | user constraint; code read `W/gr00t_wbc/dexmg/gr00trobosuite/robosuite/renderers/context/egl_context.py:39-55` | client venv ORT is CPU-only (`get_available_providers()` = Azure, CPU); assert `torch.cuda.is_initialized() is False`; do not set `CUDA_VISIBLE_DEVICES` (or set it together with `MUJOCO_EGL_DEVICE_ID=0`, unverified) | WP-M1-04 -> T-M1-5 |
| R3 | A chunk-switch discontinuity exceeds 6 rad/s and triggers `sys.exit(1)` | U.sim F13; `joint_safety.py:451-454` | `SystemExit` guard -> episode fault + HUD; boundary-jump metric; optional B = 3/5 | WP-M1-04 -> T-M1-4, T-M1-6 |
| R4 | Latency shifts the rows executed (sync rows 0-19 vs async about 3-19), so success drops | U.as risk; baseline is only 30-50% | VL sweep; sync-hold baseline; s sweep {10,16,20,25} | WP-M1-09 -> T-M1-7 |
| R5 | `PolicyClient` never applies `timeout_ms`, so a stuck server freezes the sim | `server_client.py:170,210-211` | own client with `poll(0)`, timeouts and lazy pirate | WP-M1-03 -> T-M1-4 |
| R6 | Reset re-creates the MjSim and WBC (2.5-2.8 s), invalidating handles and re-running the hook's verification RPC | U.sim F11, F14; `n1_utils.py:29-33` | backend singleton through the patched `load_onnx_policy`; reset outside the paced loop; re-anchor; re-apply observable tweaks | WP-M1-04 -> T-M1-3b (0 slips across resets) |
| R7 | Serializing the full 9.2 MB obs stalls the tick (9.9 ms) | U.srv F4, risk | 9-key trim (0.92 MB, 0.14 ms) | WP-M1-03 -> T-M1-3a |
| R8 | GPU contention: EGL renders + VLA + WBC CUDA graph on one 5090 | not measured (U.sim gap) | separate WBC process (E3 under load p99 0.62 ms); measure in the soak | WP-M1-09 -> T-M1-3b |
| R9 | Onscreen viewer sync cost in the paced loop | unmeasured (U.sim gap) | measure `viewer.sync`; fall back to syncing every 2nd tick (a renderer patch) | WP-M1-04 -> T-M1-3b |
| R10 | Balance/Walk flicker around \|cmd\| = 0.05 | U.sim F15 | count flips; hysteresis only with user approval (DP5) | WP-M1-06 metric |
| R11 | Paced GPU slowdown (VLA 59 -> 65-68 ms) pushes d to 4 | U.srv F4 | adaptive d_hat; s_eff clamp | T-M1-3b (d p99 ≤ 12) |
| R12 | TT G1 config gaps: no g1 layout, pinned sha, F32 cloudwalk tensors, cache-key collision, fixed noise seed 42 | U.tt F3-F6, F15 | `g1_n16.install()` (LIBERO pattern); default bf16 nvidia checkpoint; dedicated cache_root; per-call generator noise | WP-M2-01/03 -> T-M2-1, T-M2-2, T-M2-7 |
| R13 | TT fidelity on a fine-tune is unknown (hand groups max\|d\| 0.0925 on GR1) | U.tt F13, risk | G1 golden gate PCC ≥ 0.999; closed-loop T-M2-4; `mk_arena_dtype=bf16` as a fallback | WP-M2-02/04 -> T-M2-2, T-M2-4 |
| R14 | Card stability: a second megakernel build hangs; the guard probe may `tt-smi -r` | U.tt F11 | one model per process; `chip_session.sh` under one flock; restart the process to change checkpoint | WP-M2-04 -> T-M2-5 |
| R15 | Live-tree drift (Stage-3 work in progress) changes numerics | U.tt F14 | run from `MK` (worktree-mk-env.sh); the server stamps the commit | WP-M2-03 -> T-M2-2 |
| R16 | **The X280 hart release works only once per chip reset**; a hang costs `tt-smi -r`, which kills the TT session | ISA `BlackholeA0/L2CPUTile/README.md` "Reset" (read; U.x F3) | resident monitor + RNMI/trap handlers before release; L2CPU3 only; LOAD/CALL for all later code; §5.2 sequencing | WP-M3-02/E2 -> T-M3-E2 |
| R17 | LIM ECC fault on a partial write to an uninitialised line | U.x F2 | full-line init of every touched line; image padded to 64 B | WP-M3-03/E1 -> T-M3-E1 |
| R18 | L2CPU clock gated when the last enabling fd closes (5 s grace), freezing the WBC; NoC access to a gated tile may hang the host | U.x F7 | keeper legacy fd in x280d; interlock reads the clock before any L2CPU access; the client heartbeat watchdog triggers the fallback | WP-M3-03/E3 -> T-M3-E3 |
| R19 | PLL4 relocks on every fd open/close (4× per message), causing jitter | U.x F6 | no fd churn and no `tt-smi` during runs; E3 measures the effect | WP-M3-E3 -> T-M3-E3, T-M3-6 |
| R20 | Host TLB exhaustion / pyluwen hard-coded TLB indices clobber UMD | U.x F9 | allocated 2 MiB windows only; x280d refuses the hard-coded APIs | WP-M3-03 -> code review + T-M3-E0 |
| R21 | Memory collision: the X280's local GDDR is tt-metal bank 5/6/7 | U.x F14 | LIM only in M3; M4b uses a tt-metal-allocated buffer | WP-M3-02 -> T-M3-E1 |
| R22 | L2CPU coordinate confusion (translated (8,26..29) vs UMD identity (8,3/9/5/7)) | U.x F13 | E0 reads the same register via both coordinates | WP-M3-E0 -> T-M3-E0 |
| R23 | `TT_METAL_WATCHER=1` rejects non-LIM L2CPU writes; alloc tracking slows replay (589 ms) | U.x F1; U.tt F11 | `chip_session.sh` unsets `TT_METAL_WATCHER` and `TT_METAL_TRACE_ALLOC_TRACKING` | WP-M2-04 -> T-M2-3 |
| R24 | Zvfh/Zvfhmin absent (not in the device tree) -> illegal instruction | U.x F15 | v0 scalar; v1 widens fp16 with integer ops; probe under the trap handler | WP-M3-02/E4 -> T-M3-E4 |
| R25 | fp16 weights (both policies must fit in LIM; fp32 for both is 3.76 MB) give about 7e-3 error vs ORT | U.x F16/F17 | parity gate 1e-2 + closed-loop non-inferiority; fp32 via DRAM only if DP3 = "exact" | WP-M3-01/E5 -> T-M3-1, T-M3-E5a/c |
| R26 | No console on the X280: failures show only as mailbox dumps | U.x risk | trap handler writes mcause/mepc/mtval to L1; x280d `--dry-run`; mailbox simulator first | WP-M3-04 -> T-M3-2 |
| R27 | The research prototypes and the X280 source clones live in /tmp and will be lost | paths under `S` | copy and pin in WP-M1-00 | WP-M1-00 |
| R28 | Port clash: gr00t_wbc image publisher and camera defaults use 5555 | U.sim F16 | never instantiate them; RT ports are 5556-5559 and 5570-5571; preflight checks | WP-M1-00 -> preflight |
| R29 | Evaluation variance (±15%) leads to false conclusions about blends or RTC | U.as risk | tiers + 160/arm for claims | WP-M1-06 -> T-M1-7 |
| R30 | The HUD or recording perturbs timing | in-loop video encoding costs 17-87 ms (U.sim F11) | separate HUD process, PUB drop semantics, ffmpeg x11grab out of process; the soak runs **with** the HUD and recording on | WP-M1-05/07 -> T-M1-3b |

### 5.2 Sequencing X280 experiments safely (reset constraints)

1. **Order is fixed: E0 -> E1 -> E2 -> E3 -> E4 -> E5.** Each step has a written go/no-go (T-M3-E*). There are no
   device steps until T-M3-1, T-M3-2 and an x280d `--dry-run` review have passed.
2. **One chip session owns everything.** `bin/with-device.sh RT/scripts/chip_session.sh` holds the flock. Its guard
   probe and any reset happen at session start, **before** any hart exists. Then `server_g1` starts and prints READY,
   and only then x280d starts. No other `with-device.sh` command, no `tt-smi` and no extra tt-metal process may run
   while harts are live: each fd open is a power message and four PLL4 relocks (U.x F6).
3. **Clock before access.** x280d touches an L2CPU tile only after it confirms the L2CPU clock is ungated (U.x F7) and
   it holds its keeper fd.
4. **Read-only, then LIM-only, then release.**
   - E0 writes nothing.
   - E1 writes LIM only (full lines), with no release.
   - E2 performs **the single release**: L2CPU3 (`L2CPU_RESET` bit 7), with RNMI/exception handlers and the monitor
     already in initialised LIM, all four reset vectors set, PLL4 stepped to 200 MHz for release and back to CMFW's
     800 MHz (U.x F5, rec1).
   - L2CPU0-2 stay unreleased as clean spares.
5. **Never re-release.** All later kernels go in through the monitor's LOAD + FENCE_I + CALL. A stuck kernel is pulled
   back with an RNMI (trigger `0xFFFF_F7FE_FFF1_0414`, U.x F4).
6. **Never** write `CCACHE0_WAYENABLE` (it is one-way, U.x F11), never touch `L2CPU_RESET[3:0]`, and never run
   tt-bh-linux `boot.py` (it board-resets and sets WAYENABLE = 0xF, U.x F0).
7. **Recovery budget**: one `tt-smi -r` per E2 attempt. The procedure:
   1. Stop `rt_sim`. Nothing in the demo depends on the X280 yet: the WBC fallback chain covers it.
   2. Stop `server_g1`.
   3. Run `bin/with-device.sh tt-smi -r 0`.
   4. Restart the chip session. The cost is 41 s cold or about 4 s warm plus capture (U.tt F10).
   5. Write an incident note in `P/results/M3-incidents.md`.
8. **Demo safety**: the client heartbeat watchdog falls back to GPU, then CPU, within one tick. `SET_NOC_CLEANUP` (a
   driver write when the fd closes, U.x F20) is a stretch goal that marks "host gone" so the X280 parks.

---

## 6. Critical path, parallel lanes, decision points

### 6.1 Schedule (working days, one engineer per lane)

```
Lane A (loop):   M1-00 ─ M1-01 ─┬ M1-02 ┐                         (M1.0 mock demo ≈ day 3; M1.1 real VLA ≈ day 4)
                                ├ M1-03 ┼ M1-04 ─ M1-08 ─ M1-06 ─ M1-09(screen) ═ M1 demo-ready ≈ day 7 ─ T-M1-7 (machine)
                                └ M1-05 ┘        └ M1-07
Lane B (TT):     M2-01 ─ (M2-01b) ─ M2-03 ── [dev S1: M2-04] ─ M2-05(sync, stock client) ─ M2-06 (needs M1 demo-ready) ≈ day 10-11
                          M2-02 needs M1-05 obs dumps (≈ day 4) or a stock-path capture
Lane C (X280):   M3-01 ─ M3-02 ─ M3-03 ─ M3-04 (all CPU-only, days 1-6) ─ [dev S2: E0,E1 with server_g1 up] ─ [S3: E2,E3] ─ [S4: E4,E5] ─ M3-06 ≈ day 15-17
M4:              after M3 exit; M4a then M4b; M4c independent (5090 lane) after T-M1-7
```

- **Critical path**: M1-01 -> M1-04 -> M1 demo-ready -> M2-06 -> M3-E5 -> M3-06.
- **Device sessions**, serialized on the one card:

| Session | Content | Duration |
|---|---|---|
| S1 | M2-04 | 0.5 d |
| S1b | M2-05/06 | about 1 d machine time |
| S2 | E0/E1 | 0.5 d |
| S3 | E2/E3, the release | 0.5 d; may include one reset |
| S4 | E4/E5 | 1.5 d |
| S5 | demo soak | — |

- Totals: M1 9.5 d, M2 5.5 d, M3 10 d, about 25 d of engineering. With three lanes, M3 demo-ready lands at about
  3.5 weeks.

### 6.2 Decision points and open questions (defaults in bold)

| # | Question | Default | Decide by |
|---|---|---|---|
| DP1 | Which G1 checkpoint is the demo's? cloudwalk (served now; 517 F32 tensors, needs bf16 conversion, U.tt F5), or nvidia/GR00T-N1.6-G1-PnPAppleToPlate (local, 6.57 GB, bf16, no conversion) | **nvidia for TT and the paired GPU baseline (second GPU server on :5565)**; switch to cloudwalk if its sync success on 50 eps is ≥ 10 pts higher | before WP-M2-01 |
| DP2 | Is a separate GPU WBC process on :5556 an acceptable reading of "VLA + WBC from the GPU server"? It avoids restarting :5555. | **yes** | M1 start |
| DP3 | X280 WBC precision: fp16 weights (max\|err\| about 7e-3 vs ORT, both policies in LIM), or exact fp32 (forces DRAM-resident weights, est. up to 9 ms/inference streamed) | **fp16** | before WP-M3-E4 |
| DP4 | OK to spend L2CPU3's one-time release in a session that may need `tt-smi -r` (tears down the TT session)? | **yes, in S3 with no demo scheduled** | before S3 |
| DP5 | Allow hysteresis on the Balance/Walk switch (a behaviour change)? | **no**; report flips only | after T-M1-6 |
| DP6 | Statistics depth: 160 paired/arm for exit claims, or demo-only screening (50/arm)? | **screen gates the demo; 160 for M1 exit only** | M1-09 |
| DP7 | Code location: new `~/experiments/gr00t/rtloop` repo, or inside Isaac-GR00T-n1.6 | **new repo** | M1-00 |
| DP8 | HUD form: separate pygame window, or in-viewer overlay (needs MuJoCo ≥ 3.3 text APIs or `user_scn` labels) | **pygame window + optional status lamp** | M1-05 |
| DP9 | M3 keeps obs history and Balance/Walk select on host Python (MLP only on X280), with the full controller deferred to M4a. Acceptable as "WBC on X280" for the M3 demo? | **yes; the HUD labels it "WBC MLP on X280" until M4a** | before M3 |
| DP10 | Should the RT demo use the sync cadence (s = 20 equivalent) or the faster s = 16? | **s = 16**; the s sweep reported in M1 | M1-09 |

**Open questions carried into WPs:**
- (a) The exact layout of the TT `action_pred` buffer, for M4b.
- (b) Whether harts keep running through a PLL4 bypass/relock (E3).
- (c) Real LIM/NoC latencies (E4).
- (d) Whether `options.rt` is accepted by `Gr00tSimPolicyWrapper` (T-M1-3c).
- (e) The viewer-sync cost (T-M1-3b).
- (f) Whether the TT G1 e2e lands near the GR1 bench (41 ms) or nearer the 2-camera LIBERO figure (56 ms) (T-M2-3).
