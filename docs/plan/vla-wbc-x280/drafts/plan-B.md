# GR00T N1.6 VLA + gear_wbc WBC, concurrent and asynchronous on one Blackhole p150a: plan B (risk-first)

Date 2026-10-24. Draft B of 2+. Angle: the riskiest unknowns get the cheapest experiments first, those experiments run as
parallel tracks in week 1, and later work is gated on what they find. Evidence base:
`docs/plan/vla-wbc-x280/research/understand.json`, cited as **U:sim / U:srv / U:tt / U:x / U:as** (sections sim_control,
serving_gpu, tt_port, x280, async_practice), plus `.F<n>` for a finding, `.rec<n>` for a recommendation (0-based, in
file order) and `.M` for measurements. I re-checked the load-bearing source claims and cite them as file:line (marked ✔).

**Path shorthand.** R = `~/experiments/gr00t/Isaac-GR00T-n1.6`; W = `R/external_dependencies/GR00T-WholeBodyControl`;
T = `~/experiments/gr00t/tt-metal/models/experimental/gr00t` (live, has moved on); P = `~/experiments/gr00t/publish/worktree-mk`
(frozen snapshot 009de892, the validated 41 ms / PCC 0.99923 code, U:tt.F14); **RT = `~/experiments/gr00t/rt`** (new,
holds all new code; `git init` it); S = `/tmp/claude-1000/-home-deepgadget-isaacsim/bfbe65f0-1b38-425e-84fc-4937eb211a38/scratchpad`.
DEV = `~/experiments/gr00t/bin/with-device.sh`.
**Interpreters.** CPY = `R/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python` (sim/client);
SPY = `R/.venv/bin/python` (GPU server); TPY = tt-metal `python_env` after `source ~/experiments/gr00t/publish/worktree-mk-env.sh`;
LPY = `~/.tenstorrent-venv/bin/python` (pyluwen 0.8.5, tt-smi; U:x.F19). All of these exist (checked).
**Sim env (every sim command):** `MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT:$R`.
`MUJOCO_EGL_DEVICE_ID` is required. With only `CUDA_VISIBLE_DEVICES=""`, robosuite parses `""` as a device list and
crashes (`int("")`, W/gr00t_wbc/dexmg/gr00trobosuite/robosuite/renderers/context/egl_context.py:39-55 ✔).
**Conventions.** Tick k, dt = 20 ms (50 Hz). H = 30 used chunk rows. d = delay in ticks (k_arrive − k_obs). s = ticks
between VLA observations. B = blend ticks. Times are CLOCK_MONOTONIC ns (`time.monotonic_ns()`). Angles are rad, linear
velocity is m/s, yaw rate is rad/s.

---

## 0. Goal, scope, non-goals, decisions

**Goal.** A demo of the Unitree G1 in MuJoCo (`gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`) that runs like a
real robot:
- The lower-body WBC (gear_wbc Balance/Walk MLPs, 516→15) runs on every 50 Hz tick, unconditionally.
- GR00T N1.6 (cloudwalk G1 fine-tune) runs in the background. It turns observations into 30-step action chunks, and the
  controller splices them in by tick stamp.
- End state: everything on one Blackhole p150a, with the VLA on Tensix and the WBC on the on-die SiFive X280.

**Scope.** M1 to M3 are mandatory and M4 is optional. M0, a set of risk spikes, comes first and overlaps M1.

**Non-goals.**
- A real G1, IsaacSim, or teleop/IK paths (the VLA path has no IK, U:sim.F0).
- Linux on the X280 (U:x.OQ1).
- 1750 MHz on the X280 (CMFW re-locks PLL4 to 800 MHz, U:x.F6).
- TensorRT or compiled speed-ups on the 5090, unless DP1 calls for them.
- New device kernels for the VLA in M1 to M3 (M2 is configuration and packaging only, U:tt summary).
- Retraining. The exception is the optional RTC v2c in M4.
- Batched (B>1) VLA in RT mode.

**Decisions already made by the user.**

| # | Decision |
|---|---|
| U1 | VLA on Blackhole Tensix, using the existing tt-metal port. |
| U2 | WBC on the Blackhole X280 cores. This plan treats it as possible and treats the tt-bh-linux "not concurrent" note as an artefact of that demo's flow (U:x.F0). |
| U3 | Real-robot-like async: the WBC runs every tick and the VLA runs in the background. |
| U4 | M1 is validated on the RTX 5090: sim on the host CPU, VLA and WBC served from the GPU. Then M2 (TT VLA), then M3 (X280 WBC). |

**Decisions this plan makes.**

| # | Decision | Basis |
|---|---|---|
| D1 | **There are two wire contracts and they stay frozen from M1 to M3.** (a) The VLA uses the Isaac-GR00T PolicyServer protocol, `get_action`, msgpack with npy blobs, so the existing client code works unchanged. (b) The WBC uses `gear_wbc_action(policy, obs[1,516]) → action[1,15]`, the endpoint that already exists (R/gr00t/eval/run_gr00t_server.py diff ✔). **Each milestone replaces exactly one server behind one port.** | U:srv.rec7, U:tt.F16/F18 |
| D2 | **The WBC boundary for M1 to M3 is the stateless 516-float MLP call, not the 72-float controller.** Obs building, the 6-frame history, torso FK, the 1-tick command lag and Balance/Walk selection stay in the verified Python (W/.../g1_gear_wbc_policy.py:102-167, 257 ✔). Why: the boundary is identical to today's endpoint, every call can be checked bit-for-bit against ORT, and the loop can fail over to ORT mid-episode without losing history. The cost is 2,064 B per tick instead of 288 B, which is negligible at 50 Hz. The 72-float stateful controller moves to the X280 in M4. | U:sim.F7/F8, U:x.rec3 |
| D3 | **The WBC is never served from the VLA's REP socket.** On the 5090 it gets its own process and port (`rtserve/gpu_wbc_server.py`, :5556). In a separate CUDA context it measured p99 0.62 ms under back-to-back VLA load, versus p99 67 ms on the shared socket. It runs in that process's main thread, so the CudaGraphPolicy stream bug (U:srv.F7) cannot trigger and no repo file changes. | U:srv.F1/F2 (E3) |
| D4 | **The sim process is CUDA-free and has one tick thread.** The VLA socket is non-blocking and polled with `poll(0)` once per tick, so there is no VLA thread and no GIL contention. The WBC call is a blocking REQ with a 5 ms receive timeout, after which the loop falls back to ORT in the same process. | U:srv.rec3, U:as.rec4 |
| D5 | **v1 scheduling is a tick-indexed chunk buffer (the "aligned receding horizon"), not InterpolationPolicy.** Row i targets tick k_obs+i, which is equivalent to gr00t_wbc `target_time = t_obs + 0.02·i`. | U:as.rec0-2, U:as.OQ0 |
| D6 | **Statistics come from virtual-latency (VL) mode; jitter and the demo from wall-clock (RT) mode.** In VL mode with d=0 and s=20, the harness must reproduce sync (`MultiStepWrapper`, n_action_steps=20) bit-exactly on replayed chunks. | U:sim.rec5, U:as.rec5 |
| D7 | **M2 runs from the frozen snapshot P** with: the cloudwalk checkpoint converted to bf16, a `g1` layout injected at runtime, a separate weight-cache root, fresh noise on every call, and a PolicyServer-compatible ZMQ server on :5557. No repo file changes. | U:tt.rec0-4, F15 |
| D8 | **The X280 runs bare metal from LIM only.** WAYENABLE is never written and the GDDR is never touched. The first image is a **resident monitor**, released on **L2CPU3 (8,7)** first. The clock stays at 800 MHz. Payloads are loaded through the monitor and never through a second release. | U:x.F3/F11, rec1-2 |
| D9 | **WBC weights:** v0 is scalar fp32 with Balance only, used for exactness. **Production M3 is fp16 weights with fp32 accumulate, both policies in one LIM** (1,882,312 B). The fallback is fp32, one policy per tile (L2CPU3 = Balance, L2CPU2 = Walk). bf16 is rejected (8e-2 error). | U:x.F16/F17 |
| D10 | **A separate `x280d` daemon** (LPY, pyluwen) owns the X280 lifecycle and holds a keeper fd. It serves the D1(b) contract on :5558. It never runs in the TT VLA process, so the X280 survives VLA server restarts. | U:x.rec8 |
| D11 | **Ports.** 5555 is the existing GPU server: never stopped, never used for new sockets, and used only as an optional VLA source and as the negative control. Then 5556 GPU WBC, 5557 TT VLA, 5558 x280d, and 5565 an optional dedicated GPU VLA instance (DP0). | U:sim.F16, task rules |
| D12 | **No repository file changes in M0 to M3.** Everything new lives in RT. M4's RTC work is the first planned repo change. | task rules, risk |

**Risk-first summary: the riskiest unknowns and the cheapest experiment for each** (full register in §5).

| Unknown (risk id) | Cheapest experiment that answers it | Cost | Gate → decision |
|---|---|---|---|
| Can an X280 hart run next to a live tt-metal session, given the once-only release? (R1-R6) | SP-4: E0 read-only registers plus E1 LIM init, **no hart release**, inside a TT session. Then SP-5: E2 resident monitor on L2CPU3 plus E3 power-policy survival. | 0.5 d + 1.5 d, ~3 h device | T3.0-T3.3 → **DP3**: M3 goes ahead, or the WBC stays on the host behind the same contract |
| Does the TT port reproduce the *G1 fine-tune* (no G1 golden exists)? (R13) | SP-3: bf16 conversion, `g1` layout, 20 recorded obs, GPU fp32 golden, 20 TT predicts | 2 d, ~1 h device | T2.2 → **DP2**: proceed / bf16 arena / switch checkpoint |
| How much success does async lose versus sync (rows a[d..d+s) instead of a[0..20))? (R18) | SP-1: VL harness with a **50-seed paired pilot** in 4 arms, before any RT work | 3 d build + 1 h compute | pilot drop → **DP1**: v1 is enough / start RTC v2 now |
| Does the RT loop hold 50 Hz with GPU VLA, EGL renders and a remote WBC all on one host? (R19/R20) | SP-2: the existing `S/plan_research/simloop/async_proto.py` structure with the real VLA plus `gpu_wbc_server`, 10-minute soak | 1 d | tick p99 ≤ 21 ms → proceed, else apply the R20 mitigations |
| Is fp16 X280 math close enough, and fast enough? (R10/R11) | SP-6: the same C kernel built for x86 and run against ORT on 10k logged obs. Speed comes later from E4. | 2 d CPU | max err ≤ 1e-2 (fp16) / 1e-5 (fp32) |

---

## 1. Target architecture

### 1.1 Processes, threads, device ownership

```
 ┌──────────────── P1 sim_robot  (CPY, CUDA-free, 1 tick thread, pinned core 2) ───────────────┐
 │  MuJoCo/robosuite env + WholeBodyControlWrapper (obs build, history, FK, safety)           │
 │  Splicer v1 (pure fn)   VLA client: REQ, poll(0)/tick   WBC client: REQ, 5 ms RCVTIMEO       │
 │  ORT CPU fallback (in-process)   TickLogger (ring buffer, flushed between episodes)          │
 └───────────────┬───────────────────────────────────────┬────────────────────────────────────┘
       M-1 VLA   │ tcp :5565|:5555 (M1)  :5557 (M2+)       │ M-3 WBC  tcp :5556 (M1,M2)  :5558 (M3)
 ┌───────────────▼─────────────┐            ┌─────────────▼───────────────────────────────────┐
 │ P2 VLA server (REP, 1 thr.) │            │ P3 WBC server (REP, 1 thread)                    │
 │  M1: run_gr00t_server.py    │            │  M1/M2: gpu_wbc_server.py (own CUDA ctx)         │
 │  M2+: tt_vla_server.py      │            │  M3: x280d.py ── LIM mailbox (M-4) ── X280 hart  │
 │   (TPY, owns Tensix+DRAM)   │            │       (LPY, keeper fd, 2 MiB KMD windows)        │
 └─────────────────────────────┘            └─────────────────────────────────────────────────┘
 P4 (optional) viewer/recorder: reads logged state, never runs in P1's tick thread.
```

| Resource | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| RTX 5090 CUDA | P2 VLA (eager, ~8.5 GB), P3 WBC graphs | P3 WBC; free for goldens | free (optional reference) | free |
| RTX 5090 EGL | P1 renders (ego on demand) | same | same | same |
| Tensix grid, DRAM banks 0-7, 4 GiB host windows, ~158 2 MiB windows | idle | **P2 TT (sole owner via UMD, 2 CQs, 64 MiB trace region)** | same | plus a persistent chunk tensor |
| L2CPU3 (8,7): harts, LIM, mailbox | untouched | untouched | **x280d** (monitor + WBC) | plus controller loop |
| L2CPU2 (8,5) | spare release | spare | spare (or fp32 Walk, D9 fallback) | |
| L2CPU0/1 | spare | spare | reserved: last-resort releases | |
| ARC L2CPU_RESET (0x8003_0014), PLL4 (0x8002_0500) | never | never | x280d `boot` only, once per chip-reset epoch | |
| Host cores (16C/32T) | P1 on core 2, P2/P3 on cores 8-15 | same | x280d on core 3 | |

### 1.2 Per-tick data flow (tick k, deadline t0 + k·20 ms)

1. **Pace.** RT mode sleeps to the absolute deadline. VL mode does not wait.
2. **Poll the VLA.** `sock.poll(0)`. If a reply is ready: decode it (0.11 ms, U:sim.M), stamp `k_arrive = k`, check `gen`,
   build a ChunkRecord (M-2) and hand it to the splicer. In VL mode the record is released only at `k_obs + d_inj`.
3. **Trigger** (§3). If no request is in flight and `k − k_obs_active ≥ s`: render `robot0_oak_egoview` 640×480 on demand
   (0.28 ms mean, 5.8 ms max spike), build the 9-key snapshot from the *current* state (end of tick k−1), send without
   blocking (serialize 0.11-0.14 ms) and set `k_obs = k`.
4. **Splice.** `cmd35 = splicer.command(k)` gives one absolute target row: arms, hands, waist, base height and nav.
5. **WBC.** `WholeBodyControlWrapper.step(row)` does: `concat_action`, then `set_goal` (every tick, so the 1 s goal timeout
   at W/.../g1_decoupled_whole_body_policy.py:98 ✔ never fires), then `get_action`. Inside `get_action`: torso FK and
   policy selection by `‖nav‖<0.05` (g1_gear_wbc_policy.py:257 ✔), then **`backend.mlp(policy, obs516)` over M-3**
   (GPU 0.6 ms p50, ORT 0.024-0.08 ms), then `q15 = a·0.25 + default` (yaml:28 ✔).
6. **Safety and physics.** JointSafetyMonitor (6 rad/s arm limit, joint_safety.py:21 ✔), then 10 × 2 ms robosuite
   substeps with Python PD (4.2 ms).
7. **Observe.** `set_observation`: the 86-float frame is pushed onto the 6-deep history. The command therefore enters the
   MLP one tick later (U:sim.F7).
8. **Bookkeeping.** Check success (`info['success']`), truncation (1440 ticks) and a fall (pelvis z < 0.50 m), then
   append the TickLog row.

Budget, from measurements in U:sim.M: steps 5-7 take 5.3 ms with cameras sampled every tick, or 4.5-4.7 ms with
on-demand renders. Trigger ticks add about 0.5 ms. The total is ≤ 7 ms of a 20 ms tick, leaving about 13 ms of slack.

### 1.3 Message schemas

**M-1 VLA request** (REQ → REP, `MsgSerializer` msgpack + npy, R/gr00t/policy/server_client.py ✔). About 0.92 MB.

| Field | dtype | shape | units / notes |
|---|---|---|---|
| `endpoint` | str | – | `"get_action"` |
| `data.observation["video.ego_view"]` | uint8 | (1,1,480,640,3) | RGB, `sim.render(camera_name="robot0_oak_egoview")[::-1]` at k_obs |
| `data.observation["state.left_leg"]`, `["state.right_leg"]` | float32 | (1,1,6) each | rad |
| `data.observation["state.waist"]` | float32 | (1,1,3) | rad |
| `data.observation["state.left_arm"]`, `["state.right_arm"]` | float32 | (1,1,7) each | rad |
| `data.observation["state.left_hand"]`, `["state.right_hand"]` | float32 | (1,1,7) each | rad |
| `data.observation["annotation.human.task_description"]` | tuple[str] | (1,) | instruction |
| `data.options` | dict | – | `{obs_tick:int64, t_obs_ns:int64, seq:uint32, gen:uint32, noise_seed:int64\|None}`. The GPU server accepts and ignores it (gr00t_policy.py:580, 613 ✔). The TT server echoes it and honours `noise_seed`. |

**M-1 reply.** `[actions, info]`, about 5.4 KB.

| Key | dtype | shape | units |
|---|---|---|---|
| `action.left_arm`, `action.right_arm` | float32 | (1,30,7) | rad, **absolute**. The server converts relative to absolute against the state at obs time; never re-anchor (U:as.F1). |
| `action.left_hand`, `action.right_hand` | float32 | (1,30,7) | rad. Treat as near-binary and never blend. |
| `action.waist` | float32 | (1,30,3) | rad (yaw, roll, pitch). Only feeds the WBC torso-RPY command (U:sim.F2). |
| `action.base_height_command` | float32 | (1,30,1) | m (0.74 constant in training) |
| `action.navigate_command` | float32 | (1,30,3) | vx, vy m/s; wz rad/s (training range ±0.5 / ±1.0) |
| `info` (TT server only) | dict | – | `{obs_tick, seq, gen, t_recv_ns, t_send_ns, encode_ms, device_ms, decode_ms, noise_seed}`. The GPU server returns `{}`. |

**M-2 ChunkRecord** (in P1; laid out as a C struct so it ports to X280 in M4; 4,248 B).

| Field | dtype | Meaning |
|---|---|---|
| seq | u32 | request sequence number |
| gen | u32 | generation: bumped on episode reset or instruction change |
| k_obs | i64 | tick whose start state the obs shows. Row i is meant for tick k_obs+i. |
| t_obs_ns, t_send_ns, t_recv_ns | i64 | CLOCK_MONOTONIC |
| k_arrive | i64 | first tick at which the chunk is usable |
| a | f32[30][35] | decode order: [0:7] L arm, [7:14] R arm, [14:21] L hand, [21:28] R hand, [28:31] waist, [31] base height, [32:35] nav |

DiT rows 30-49 are never used, because they are untrained and loss-masked (U:as.F0).

**M-3 WBC request** (REQ → REP, same serializer, about 2.3 KB).
`{"endpoint":"gear_wbc_action","data":{"policy":"GR00T-WholeBodyControl-Balance"|"GR00T-WholeBodyControl-Walk","obs":f32(1,516),"tick":i64}}`
returns `{"action": f32(1,15), "tick": i64, "compute_us": f32}`.
- `tick` and `compute_us` exist only on the new servers (5556/5558). The 5555 endpoint takes `(policy, obs)` and would
  reject a `tick` keyword, so the client sends `tick` only when `--wbc-proto v2` is set.
- `obs` is 6 frames × 86 floats, oldest first. Each frame:

| Slice | Content |
|---|---|
| [0:3] | nav cmd × [2, 2, 0.5] |
| [3] | height cmd, m |
| [4:7] | roll, pitch, yaw cmd, rad |
| [7:10] | pelvis ω × 0.5, body frame |
| [10:13] | gravity in the pelvis frame |
| [13:42] | q29 − default (default is 15 values padded with 0) |
| [42:71] | dq29 × 0.05 |
| [71:86] | last raw action (15) |

  Scales are from W/gr00t_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml:21-40 ✔.
- Output `action` is 15 values (legs 12 + waist 3), raw. The joint target is `q = a·0.25 + default_angles`.

**M-4 X280 LIM mailbox** (L2CPU3, uncached, 64 B lines, one writer per line, base `0x0801_0000`; U:x.rec3/rec4).

| Line | Offset | Writer | Content |
|---|---|---|---|
| L0 | +0x000 | host | u32 magic 'WBC1', u32 cmd (0 idle, 1 RUN_MLP, 2 LOAD, 3 FENCE_I+CALL, 4 PARK, 5 SELFTEST), u32 seq_req, u32 policy (0 Balance, 1 Walk), u64 tick, u32 flags (bit0 host_alive, bit1 HOST_GONE via SET_NOC_CLEANUP), u32 arg_len, u64 arg_addr, u64 arg_entry, u32 crc32 |
| L1 | +0x040 | fw | u64 heartbeat (mtime ticks), u32 seq_done, u32 state (0 boot, 1 idle, 2 busy, 3 fault, 4 parked), u64 mcause, u64 mepc, u64 mtval, u32 cycles_last, u32 cycles_max, u32 fw_ver, u32 mhartid, u32 vlenb |
| L2-L34 | +0x080 | host | f32 obs[516] (2,064 B, zero-padded to 2,112) |
| L35 | +0x8C0 | fw | f32 action[15], u32 seq_echo |
| L36-L39 | +0x900 | fw | misa, boot mtime, n_runs, n_overrun, 16-bin cycle histogram |
| L40-L63 | +0xA00 | – | reserved for M4 (M-6 frame, chunk descriptor) |

Protocol:
1. The host writes L2-L34 as full lines.
2. The host writes L0 with `seq_req = n+1` using a UC Strict write.
3. The firmware spins on `seq_req`, executes `fence rw,rw`, runs the MLP, writes L35, executes `fence w,w`, then writes
   L1 with `seq_done = n+1`.
4. The host polls L1 with UC reads (2 ms timeout), then reads L35 and checks `seq_echo`.

LIM map: `0x0800_0000` holds 64 KiB of code, rodata, bss and stack; `0x0801_0000` the 4 KiB mailbox; `0x0801_1000`
Balance then Walk fp16 weights (biases fp32), ending at `0x081D_C8C8`. That leaves 14,136 B below the LIM end at
`0x081E_0000`.

**M-5 log records**: see §4.

### 1.4 Stage mapping: one backend swap per milestone

| Component | Contract | M1 (5090) | M2 (Tensix) | M3 (X280) | M4 |
|---|---|---|---|---|---|
| Sim + tick loop + splicer | in-process | P1 | P1 | P1 | splicer and loop in C on X280; P1 follows the X280 clock |
| VLA | M-1 | GPU eager on :5565 (or :5555) | **`tt_vla_server.py` :5557** | :5557 | :5557, plus the chunk copied to a DRAM tensor (M4.1) |
| WBC MLP | M-3 | **`gpu_wbc_server.py` :5556** | :5556 | **`x280d` :5558 → M-4** | M-6 stateful controller on X280 |
| WBC fallback | in-process ORT | ORT | ORT | ORT | X280 hold |

P1 is configured only with `--vla tcp://H:P --wbc tcp://H:P --wbc-proto v1|v2`. No P1 code changes between M1 and M3.

### 1.5 Clocks

- **RT:** `t_deadline(k) = t0_ns + k·20,000,000` on CLOCK_MONOTONIC. If a tick ends more than one period late, re-anchor
  and log a `slip`. Sim time is `k·0.02` and is never passed to `get_action(time=...)` (U:sim.F13).
- **VL:** no pacing. A chunk becomes visible at exactly `k_obs + d_inj`; if it has not arrived by then, the loop blocks.
- **Reset** happens outside the RT clock (2.5-2.8 s, and the MjSim and WBC are re-created, n1_utils.py:29-33 ✔). After a
  reset the loop re-binds the WBC backend, re-applies the camera sampling tweak, bumps `gen` and re-anchors t0.

---

## 2. Milestones, work packages, acceptance tests

Effort is in engineer-hours (h). "dev" marks WPs that need the Blackhole and must run under DEV. Every test writes a JSON
file to `RT/runs/<test-id>/<timestamp>.json` that holds the gate values.

### M0: risk spikes (week 1; tracks A, B and C in parallel)

Each spike is the first slice of a later WP, not throw-away work, and ends in a go/no-go JSON.

| Spike | Built from | Answers | Gate |
|---|---|---|---|
| SP-1 async-drop pilot | WP-1.1 to 1.5, plus WP-1.7 in VL mode only | success loss from latency | §3 DP1 |
| SP-2 jitter soak | WP-1.6, plus the RT mode of WP-1.7 | 50 Hz holds with GPU VLA + EGL + remote WBC | T1.3 timing rows, 10-minute version |
| SP-3 TT G1 fidelity | WP-2.1 to 2.4, plus the TT-D1 session | G1 fine-tune fidelity on TT | T2.2 |
| SP-4 X280 no-release | WP-3.0, 3.1 (inside the TT-D1 hold) | registers, LIM ECC, coexistence | T3.0, T3.1 |
| SP-5 X280 release | WP-3.2 to 3.4 (X-D1 session) | once-only release, monitor, power policy | T3.2, T3.3 |
| SP-6 kernel parity | WP-3.5 x86 part | fp16/fp32 accuracy | T3.4 (x86 rows) |

### M1: async loop on the RTX 5090

| WP | What | Files (all new) | Deps | h | dev |
|---|---|---|---|---|---|
| WP-1.0 | Scaffold RT (`git init`), env scripts for CPY/SPY/TPY/LPY, README | `RT/README.md`, `RT/env/{sim.sh,server.sh,tt.sh,lpy.sh}` | – | 2 | – |
| WP-1.1 | Schemas M-1 to M-5 as dataclasses plus numpy structured dtypes; offsets and sizes as constants | `RT/rtloop/schema.py`, `RT/tests/test_schema.py` | 1.0 | 4 | – |
| WP-1.2 | Splicer v1 as **pure functions**: `trigger(state,k)`, `on_arrival(state,chunk,k)`, `command(state,k)`, with no hidden state (ports to C in M4) | `RT/rtloop/splicer.py`, `RT/tests/test_splicer.py` | 1.1 | 8 | – |
| WP-1.3 | Robot sim. Env built as in live_viewer.py:96-112 **without** MultiStepWrapper or SyncVectorEnv. `reset(seed)`. Camera `sampling_rate` lowered after every reset. On-demand ego render. 9-key snapshot. **Re-bind `lower_body_policy.policy_1/policy_2` to the backend after every reset.** `SystemExit` guard around `step` (U:sim.F13). Chunk-alignment probe for R28. | `RT/rtloop/sim_robot.py` | 1.1 | 10 | – |
| WP-1.4 | VLA client with backends ZMQ (REQ, poll(0), RCVTIMEO, lazy-pirate re-create, seq/gen), `ReplayVla` (recorded chunks in order) and `MockVla` (hold chunk, fixed latency) | `RT/rtloop/vla_client.py`, `RT/tests/test_vla_client.py` | 1.1 | 8 | – |
| WP-1.5 | WBC backends: `OrtCpu` (intra_op=1), `Zmq` (5 ms timeout, then fallback; 3 consecutive timeouts switch the episode to ORT), `Shadow` (primary plus reference, logs max\|Δ\|) | `RT/rtloop/wbc_backend.py`, `RT/tests/test_wbc_backend.py` | 1.1 | 6 | – |
| WP-1.6 | GPU WBC server in its own process: SPY, REP :5556, `GearWbcPolicies` imported from R/gr00t/policy/gear_wbc_policy.py and called from the main thread only, endpoints `ping` and `gear_wbc_action(policy,obs,tick=None)`, `taskset` to cores 8-15. Launcher for the optional dedicated VLA: `run_gr00t_server.py --model-path cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper --host 127.0.0.1 --port 5565` (no WBC dir). | `RT/rtserve/gpu_wbc_server.py`, `RT/rtserve/launch_gpu_vla.sh` | 1.0 | 6 | – |
| WP-1.7 | Main loop. Modes `vl`, `rt`; arms `sync_frozen`, `sync_hold`, `async_noskip`, `v1`. Episode control: success, 1440-tick truncation, fall, fault. `gc.disable()` during episodes and `gc.collect()` at reset. Core pinning. CLI. | `RT/rtloop/run_async.py`, `RT/rtloop/clock.py` | 1.2-1.5 | 10 | – |
| WP-1.8 | TickLogger ring buffer (flushed only between episodes), RequestLog, EpisodeLog. Analyzer that computes the gates and draws the Gantt chart and histograms. | `RT/rtloop/logger.py`, `RT/tools/analyze_run.py` | 1.7 | 8 | – |
| WP-1.9 | Eval matrix runner. Runs N VL sim processes in parallel against one server. Statistics: Wilson CI, McNemar, paired difference with bootstrap CI. | `RT/tools/eval_matrix.py`, `RT/tools/stats.py` | 1.8 | 6 | – |
| WP-1.10 | Recorder for replay chunks and raw obs: produces `ReplayVla` files and the M2 golden inputs (raw_observation.npz schema) | `RT/tools/record_obs.py` | 1.7 | 3 | – |
| WP-1.11 | Viewer option: the live_viewer.py camera patch inside P1 at ≤ 25 Hz, with its cost measured. The default is off plus offline render from logs. | `RT/rtloop/viewer.py` | 1.7 | 4 | – |
| WP-1.12 | Acceptance campaign: §4 matrix on the 5090 (mostly unattended compute) | `RT/runs/M1/*` | 1.9 | 12 | – |

**M1 acceptance tests**

| Test | Command | Pass threshold |
|---|---|---|
| T1.1 unit | `cd $RT && $CPY -m pytest tests/test_schema.py tests/test_splicer.py tests/test_vla_client.py tests/test_wbc_backend.py -q` | 100% pass. The splicer table covers d ∈ [0,28], s ∈ {10,16,17,20}, B ∈ {0,3,5}, late-reject, gen-drop, starvation ramp, 1 s fault. |
| T1.2a VL ≡ sync (own harness) | `$CPY -m rtloop.run_async --mode vl --arm v1 --d-inj 0 --s 20 --B 0 --vla replay:runs/replay/sf_s0-4 --wbc ort --seeds 0-4` versus `--arm sync_frozen` with the same replay | **max\|Δq\| = 0.0** over every tick of all 5 episodes, and identical success ticks |
| T1.2b own sync ≡ official sync | `$CPY -m pytest tests/test_multistep_equivalence.py` (feeds the same replay chunks through `MultiStepWrapper(n_action_steps=20)` from R/gr00t/eval/rollout_policy.py `create_eval_env`, with the env seeded) | max\|Δq\| = 0.0 for 5 seeds. This proves `sync_frozen` in RT is the official baseline. |
| T1.3 RT timing soak | `$CPY -m rtloop.run_async --mode rt --arm v1 --vla tcp://127.0.0.1:5565 --wbc tcp://127.0.0.1:5556 --wbc-proto v2 --seeds 1000-1019`, then `tools/analyze_run.py --gates m1` | Measured outside reset windows. Period: p99 ≤ 21.0 ms, max ≤ 40 ms. Overrun ticks (compute > 20 ms) ≤ 0.5%. Slips 0. WBC RTT p99 ≤ 2.0 ms. WBC fallbacks ≤ 0.1% of ticks. d p99 ≤ 5. Starved ticks 0. SystemExit 0. In-flight requests never > 1. |
| T1.4 negative control | Same as T1.3 with `--wbc tcp://127.0.0.1:5555 --wbc-proto v1` (shared REP), 2 episodes | > 3% of ticks with WBC RTT > 20 ms. This shows the instrument sees head-of-line blocking (U:srv.F1: 5.1% expected). |
| T1.5 success (VL) | `$CPY -m tools.eval_matrix --mode vl --arms sync_frozen,v1 --d-inj 3 --seeds 0-159 --procs 4` | Paired difference v1 − sync_frozen ≥ −10 pts (point estimate). Wilson CIs and McNemar p are reported. |
| T1.6 success (RT) | `$CPY -m tools.eval_matrix --mode rt --arms sync_hold,v1 --seeds 0-159` | v1 ≥ sync_frozen(T1.5) − 10 pts. v1 median time-to-success ≤ sync_hold's. 0 loop-attributable falls. |
| T1.7 liveness | `$CPY -m rtloop.run_async --mode rt --arm v1 --duration 2h --seeds 2000-` | 0 hangs. Every episode ends SUCCESS, TRUNC or FAULT. 0 orphan requests. P1 RSS growth < 50 MB. |

### M2: VLA on Tensix

| WP | What | Files | Deps | h | dev |
|---|---|---|---|---|---|
| WP-2.1 | CPU conversion of cloudwalk F32 tensors to BF16 (as Gr00tPolicy does at load, gr00t_policy.py:82-84 per U:tt.F5). Keeps the base 2-shard weight_map and copies config, processor_config, statistics and embodiment_id. Checks `total_size == 6,573,217,664` and that every dtype is BF16. Output: `~/.cache/gr00t-tt-g1/cloudwalk-bf16/`. | `RT/rttt/convert_cloudwalk_bf16.py` | – | 4 | – |
| WP-2.2 | Runtime setup with no file edits: HF_HUB_CACHE shim pointing `snapshots/d0814e7ecb19…` at the bf16 dir (gr00t_p150 `_shim_snapshot` pattern); inject the `g1` layout via `dataclasses.replace(gr1, name='g1', embodiment_tag='unitree_g1', embodiment_id=8, valid_slice=(30,35))`; cache_root `~/.cache/gr00t-tt-g1-cloudwalk`; `GR00T_TT_FORCE_CONVERT=1` on first build | `RT/rttt/g1_setup.py`, `RT/tests/test_tt_setup_cpu.py` | 2.1 | 4 | – |
| WP-2.3 | Record 20 raw G1 obs: 10 reset seeds plus 10 mid-episode obs from sync_frozen GPU runs | uses WP-1.10, output `RT/runs/golden_inputs/*.npz` | 1.10 | 2 | – |
| WP-2.4 | G1 goldens on the 5090 in a separate process (SPY; the running server is untouched; check that ≥ 20 GB VRAM is free first). fp32-sdpa and bf16-fa2 per sample with saved noise, adapted from R/tt_golden/dump_golden_n16.py (`--model-path <bf16 dir> --embodiment-tag unitree_g1 --npz`). | `RT/rttt/dump_g1_golden.py` | 2.1, 2.3 | 8 | – |
| WP-2.5 | TT VLA server (TPY on P). PolicyServer-compatible REP :5557 in the device-owning main thread with one lock. **Fresh noise** from a running `torch.Generator`, or `options.noise_seed` when given. Static-shape refusals become error replies, never a crash. `info` timings. MsgSerializer copied (about 40 lines, U:tt.F18). Launcher: `DEV bash -c 'source …/worktree-mk-env.sh; unset TT_METAL_TRACE_ALLOC_TRACKING; exec $TPY RT/rtserve/tt_vla_server.py --port 5557'` | `RT/rtserve/tt_vla_server.py`, `RT/rtserve/msg.py`, `RT/rtserve/launch_tt_vla.sh`, `RT/tests/test_tt_server_protocol.py` (CPU, fake model) | 2.2 | 10 | – |
| WP-2.6 | **TT-D1 device session** (about 1.5 h, also hosts SP-4). Cold build with the new cache root, then fidelity and latency tests. | `RT/tests/test_tt_g1_fidelity.py` | 2.4, 2.5 | 6 | dev |
| WP-2.7 | Closed loop: sync_frozen against TT, then v1 RT against TT (P1 unchanged except `--vla :5557`) | `RT/runs/M2/*` | 1.12, 2.6 | 10 | dev |
| WP-2.8 | *Conditional on T2.2 failing:* `mk_arena_dtype=bf16` (denoise about 4.71 vs 3.83 ms/step) and a re-run of TT-D1. If it still fails, DP2. | – | 2.6 | 6 | dev |

**M2 acceptance tests**

| Test | Command | Pass threshold |
|---|---|---|
| T2.1 CPU setup | `$TPY -m pytest RT/tests/test_tt_setup_cpu.py -q` | bf16 index total_size equals 6,573,217,664, all BF16. `assert_call` OK for layout g1, 503 kept / 534 skipped megakernel weights. Port encode/decode is bit-identical to the Isaac processor on the 20 recorded obs (max\|d\| = 0, as U:tt.F1 showed on synthetic frames). |
| T2.2 fidelity | `DEV $TPY -m pytest RT/tests/test_tt_g1_fidelity.py` | For each of the 20 obs with golden noise: `action_pred_valid` PCC versus fp32 ≥ 0.999 on average and ≥ 0.998 at minimum. Per-group decoded max\|d\| ≤ 2.0 × the GPU bf16-vs-fp32 max\|d\| on the same sample (constant groups left_hand and base_height are excluded). 5 replays bit-identical. ABAB stale-state test passes. |
| T2.3 latency | `$CPY -m rtloop.vla_client --bench tcp://127.0.0.1:5557 --n 500 --period 0.32` | Client RTT p50 ≤ 47 ms, p99 ≤ 55 ms. d p99 ≤ 3 ticks. Two calls with the same obs and no seed give *different* chunks (fresh noise). |
| T2.4 sync fidelity in closed loop | `eval_matrix --mode vl --arms sync_frozen --vla tcp://127.0.0.1:5557 --seeds 0-159`, paired with the T1.5 GPU sync_frozen arm | TT − GPU ≥ −10 pts |
| T2.5 async with TT | T1.3 command with `--vla tcp://127.0.0.1:5557`, then `eval_matrix --mode rt --arms v1 --seeds 0-159` | T1.3 timing gates hold. v1-TT ≥ GPU sync_frozen − 10 pts. |
| T2.6 soak | 2 h RT | 0 server error replies, 0 device hangs, VLA p50 drift < 5% between the first and last 10 minutes |

### M3: WBC on X280

X280 experiments follow the safe sequence in §5.2. WP-3.0, 3.1 and 3.4 are the only steps that touch ARC or L2CPU registers.

| WP | What | Files | Deps | h | dev |
|---|---|---|---|---|---|
| WP-3.0 | Register map and **E0 read-only probe**. Reads: L2CPU_RESET (expect 0x0F, bits 4-7 clear), CCACHE0_WAYENABLE (expect 0), PLL4 CNTL_1/5 (fbdiv 64, postdiv 1), hart status, L2CPUCLK telemetry. Each read goes through both `ttnn.cluster.read_reg` (in-process) and pyluwen `noc_read32`. Also checks that (8,7) and (8,29) return the same CCACHE0_CONFIG. **Audits pyluwen 0.8.5 to confirm it uses `allocate_tlb` and never a Hardcoded TLB index** (U:x.F9). Ledger file. | `RT/rtx280/{regs.py,ledger.py,probe.py}` | – | 6 | dev (read-only) |
| WP-3.1 | **E1, LIM without a release**: full-line zero-fill of all 1,966,080 B, pattern write and read-back, then one 4 B UC write to an initialised line. Runs while a TT VLA predict loop runs in the same hold. | `RT/rtx280/lim_test.py` | 3.0 | 4 | dev |
| WP-3.2 | Resident monitor firmware (clang-20, `--target=riscv64-unknown-elf -march=rv64gcv_zfh_zba_zbb -mabi=lp64d -mcmodel=medany -O2 -ffreestanding -nostdlib`, U:x.rec6). Contents: crt0 (harts with `mhartid%4≠0` park in wfi; `csrw 0x7c1,0`; FS/VS on), mtvec dump into L1, RNMI park path, the M-4 protocol (LOAD/FENCE_I+CALL/PARK/SELFTEST), heartbeat. The same mailbox C is built for x86 for unit tests. | `RT/fw/{crt0.S,lim.ld,monitor.c,trap.c,mailbox.h,Makefile}`, `RT/tests/test_mailbox_x86.py` | – | 16 | – |
| WP-3.3 | Boot and control tools. `boot.py` runs P1-P8 (U:x.rec0): preflight from E0 (refuse on any mismatch); LIM init; image write with read-back CRC; RNMI handlers plus the ISA-recommended 64 B GP-scratch park stub (`fence.i`, load the reset vector, jump; ISA L2CPUTile/README.md:31 ✔); reset vectors with read-back; PLL4 to 200 MHz; set the release bit; PLL4 to 800 MHz; L2PF config. It refuses whenever the ledger says the tile is already released. `ctl.py` provides status/load/call/park. | `RT/rtx280/{boot.py,ctl.py}` | 3.0, 3.2 | 10 | – |
| WP-3.4 | **X-D1 session (E2 + E3)**: release L2CPU3, then T3.2 and T3.3 | `RT/rtx280/power_survival.py` | 3.1, 3.3 | 6 | dev |
| WP-3.5 | WBC kernel: v0 scalar fp32; v1 RVV fp32 (column-broadcast GEMV, e32/m4); v2 fp16 weights widened with integer ops (no Zvfh until a trapped probe passes, U:x.F15); ELU via an exp polynomial; weight packer. Golden test of the x86 build against ORT through ctypes on 10k obs logged in M1. | `RT/fw/{wbc_mlp.c,elu.c}`, `RT/rtx280/pack_weights.py`, `RT/tests/test_fw_x86.py` | 1.8 | 16 | – |
| WP-3.6 | **X-D2 session (E4 + E5)**: GEMV microbench (LIM vs cached vs uncached), host mailbox RTT with VLA idle vs busy, kernel parity on device over the same 10k obs via `ctl.py` | `RT/rtx280/bench.py` | 3.4, 3.5 | 6 | dev |
| WP-3.7 | `x280d` daemon (LPY): keeper legacy fd (`os.open(O_RDWR)`), boot or **reattach** (heartbeat advancing → no release), REP :5558 speaking M-3 v2, SET_NOC_CLEANUP registering a HOST_GONE flag write (U:x.F20), a telemetry check that L2CPUCLK3 > 0 before any NoC access, watchdog (heartbeat stall > 5 ms → error reply, and P1 falls back) | `RT/rtx280/x280d.py` | 3.6 | 12 | – |
| WP-3.8 | Closed loop: first **Shadow** (ORT primary, X280 shadow, 20 episodes), then X280 primary for the §4 eval | `RT/runs/M3/*` | 2.7, 3.7 | 10 | dev |
| WP-3.9 | Demo launcher. One DEV hold, in this order: x280d (boot/reattach), TT VLA server, P1. No fd churn during the run. | `RT/demo/{up.sh,down.sh}` | 3.7 | 4 | dev |

**M3 acceptance tests**

| Test | Command | Pass threshold |
|---|---|---|
| T3.0 probe | `DEV $LPY RT/rtx280/probe.py --tile 3 --also-ttnn` (inside a TT session) | All values as expected. ttnn and pyluwen agree. pyluwen audit is clean. VLA outputs bit-identical and p50 latency within ±2% over 50 predicts during the probe. |
| T3.1 LIM | `DEV $LPY RT/rtx280/lim_test.py --tile 3 --with-vla-loop` | 1,966,080 B read back exactly, 0 faults. The partial write after init is OK. VLA bit-identical and latency within ±2%. |
| T3.2 monitor | `DEV $LPY RT/rtx280/boot.py --tile 3 --image RT/fw/build/monitor.bin && $LPY RT/rtx280/ctl.py selftest` | Heartbeat advances. `vlenb == 64`. misa has V. mcycle/mtime = 16 ± 5% (800 MHz / 50 MHz). LOAD+CALL of a checksum payload is correct 10/10. PARK via RNMI then re-CALL succeeds 10/10. |
| T3.3 power survival | `DEV $LPY RT/rtx280/power_survival.py --tile 3` | Heartbeat has no gap > 10 ms across 10× second-fd open/close, one `tt-smi -s` snapshot and one TT VLA server restart while the keeper fd is held. The clock reads 800 MHz after each event. Step (d), which gates the clock deliberately, is **optional and needs user approval** (§5.2). |
| T3.4 kernel parity | `$CPY -m pytest RT/tests/test_fw_x86.py` and `DEV $LPY RT/rtx280/bench.py --parity runs/M1/obs10k.npz` | fp32 (v0, v1) max\|err\| ≤ 1e-5 versus ORT. fp16 (v2) ≤ 1e-2. Device v0 bit-equal to the x86 build. |
| T3.5 timing | `DEV $LPY RT/rtx280/bench.py --rtt --n 30000` (10 minutes at 50 Hz, VLA running) | X280 compute p99 ≤ 1.0 ms (v1/v2; v0 ≤ 2.0 ms). Mailbox RTT excluding compute p99 ≤ 0.3 ms. x280d end-to-end p99 ≤ 1.5 ms. 0 timeouts. |
| T3.6 closed loop | T1.3 command with `--vla :5557 --wbc :5558`, then `eval_matrix --mode rt --arms v1 --seeds 0-159` | T1.3 timing gates hold. Success ≥ M2 v1-TT − 10 pts (paired). In shadow mode, per-tick \|a_x280 − a_ort\| p99.9 ≤ 1e-2 (fp16) or ≤ 1e-5 (fp32). |
| T3.7 fault injection | `RT/tests/fault_x280.sh` with three cases: kill x280d mid-episode; `ctl.py park`; firmware sees HOST_GONE after `kill -9` of x280d | P1 falls back to ORT within 1 tick. No SystemExit, no fall. The episode continues. x280d restart reattaches without a release. |

### M4 (optional)

| WP | What | Files | Deps | h | dev |
|---|---|---|---|---|---|
| WP-4.1 | **Whole lower-body controller on X280.** Contract M-6, 72 floats in: q_body29 (rad), dq_body29 (rad/s), pelvis quat wxyz 4, pelvis gyro 3 (body, rad/s), nav 3, height 1, torso-RPY cmd 3 (or 3 waist targets, with the closed-form FK on X280). Out: q15 plus a15. Obs build, history, the 1-tick lag and policy selection in C. Bit-parity test against Python on logged ticks. | `RT/fw/wbc_ctrl.c`, `RT/tests/test_ctrl_parity.py` | M3 | 24 | dev |
| WP-4.2 | Splicer v1 in C on X280. **X280 owns the 50 Hz clock** (mtime at 50 MHz). P1 becomes the follower ("robot"): it pushes state and pulls q43. | `RT/fw/splicer.c`, `RT/rtloop/robot_follower.py` | 4.1 | 24 | dev |
| WP-4.2a | Sim-side split into robot and controller processes, gr00t_wbc style (shared-memory lowstate/lowcmd, latest-wins) | `RT/rtloop/robot_proc.py` | 4.2 | 10 | – |
| WP-4.3 | **On-chip VLA→WBC handoff.** The TT server copies the decoded chunk into a persistent DRAM tensor at a fixed address. x280d sends the address and `get_dram_bank_table` once. The X280 reads through a 128 GiB uncached window, using `chunk_seq` against torn reads. The host still reads the chunk for the upper body (U:x.rec4). | `RT/rtserve/tt_vla_server.py` (M4 flag), `RT/fw/chunk_rx.c` | 4.2 | 24 | dev |
| WP-4.4 | **RTC v2.** (a) Guided reference on the 5090. (b) Gradient-free soft replacement, ported into the TT megakernel Euler stage. Optional (c) training-time RTC. Modifies R/gr00t/model/gr00t_n1d6/gr00t_n1d6.py:288-363, T/tt/action_head.py:750-770 and the megakernel (§3.2). | as listed (first repo changes) | M2 | 40+ | dev |

M4 acceptance:
- **T4.1:** C controller bit-parity, max\|Δq15\| ≤ 1e-6 over 10k logged ticks.
- **T4.2:** X280-clocked loop period p99 ≤ 20.2 ms and a follower lag of 0 ticks in 99.9% of ticks.
- **T4.3:** chunk read on X280 is byte-identical to the host copy for 10k chunks, with 0 torn reads.
- **T4.4:** v2 ≥ v1 + 5 pts at d_inj ≥ 5 or a boundary jump ≤ 0.5× v1's, with no loss of success.

---

## 3. Chunk scheduling policy

### 3.1 v1: "aligned receding horizon" (M1 to M3)

| Parameter | Value | Basis |
|---|---|---|
| Rows used | H = 30 (rows 30-49 are never used) | U:as.F0 |
| Index law (latency compensation) | `row = k − k_obs`. The first d rows are skipped implicitly (UMI-style). Never re-anchor relative arms. | U:as.F1/F13 |
| Trigger period | `s = min(s_cfg, H − d̂ − m)`, s_cfg = 16, m = 2. Sweep values {10, 16, 20−d̂}; the last matches the sync end row 19. | U:as.rec1, U:sim.rec2 |
| Delay estimate | d̂ = max of the last 10 measured d (ticks), d_init = 5 | RTC Alg. 1 (U:as.F5) |
| Concurrency | At most 1 request in flight. A trigger while busy is logged `skipped`, never queued. | U:srv.rec3 |
| Episode start | The first request is blocking. The robot holds the reset pose with nav = 0, so the WBC runs Balance. | U:as.rec1 |
| Splice | Hard switch at k_arrive (B = 0, default). Option B ∈ {3,5}: `a = (1−w_j)·old + w_j·new`, `w_j=(j+1)/(B+1)`, on arms, waist and nav only. Hands and base height always switch hard. If the old chunk runs out during a blend, its last row is reused. | U:as.rec2 |
| Late chunk | If `H − d < 4` (d ≥ 26, 520 ms): reject (`late_reject`) and trigger at once | U:as.rec2 |
| Staleness | Drop on `gen ≠ current`. Drop if `t_recv − t_obs > 520 ms`. | U:as.rec2/3 |
| Starvation | When the cursor reaches row 30 with nothing pending: hold the last arm, waist, hand and base-height rows, and ramp nav linearly to 0 over 5 ticks (‖cmd‖ < 0.05 means Balance). After 50 starved ticks (1 s) the episode is FAULT; the hold continues and requests continue. | U:as.rec3, F14 |
| Balance/Walk | No hysteresis in v1 (it would change behaviour). Flips are logged. Hysteresis is DP5. | U:sim.F15 |
| Safety | A `SystemExit` from JointSafetyMonitor is caught, the episode becomes FAULT, the hold pose is applied and the env resets | U:sim.F13 |

The expected chunk age is in [d, s+d) with mean ≈ d + s/2, which is 11 ticks (220 ms) at d=3 and s=16. Sync has 0..19 with
mean 9.5 (U:as.rec11). GPU duty ≈ 59/320 = 18%; TT duty ≈ 45/320 = 14%.

**Comparison arms.** All use the same code path, differing only in parameters.

| Arm | Definition |
|---|---|
| `sync_frozen` | VL, d_inj = 0, s = 20, B = 0. Identical to MultiStepWrapper; T1.2 proves it. |
| `sync_hold` | The honest synchronous robot. Trigger at row 20, hold row 19 for d ticks, then start the new chunk at row 0. |
| `async_noskip` | Negative control. v1 trigger, but the new chunk starts at row 0 on arrival. |
| `v1(B)` | As in the table above |

**Decision point DP1 (after the SP-1 pilot, 50 paired seeds per arm, VL d_inj ∈ {0,3,5}):**
- Drop of v1 at d=3 versus sync_frozen ≤ 10 pts: v1 is the M1-M3 policy and v2 stays in M4.
- 10-20 pts: run the s and B sweeps first, then RTC v2a/b on the 5090 in parallel with M2.
- More than 20 pts, or `async_noskip ≥ v1`: stop and debug alignment (R28) before anything else.

### 3.2 v2: RTC (M4, optional)

- Inpainting in `Gr00tN1d6ActionHead.get_action_with_features` (R/gr00t/model/gr00t_n1d6/gr00t_n1d6.py:288-363 ✔), with
  H = 30, s = 15, d ≤ 5, the exponential soft mask (for d=3: W = [1,1,1,.815,.655,.518,.402,.305,.224,.158,.105,.065,.035,.015,.004,0…])
  and β = 5 (U:as.rec6/7).
- The prefix Y is the not-yet-executed rows of the active chunk, re-encoded against the *new* obs state with the
  per-index (30,7) relative statistics. Round-trip unit test: `apply_action` then `unapply_action`.
- (a) Guided VJP: 5090 reference only. It conflicts with inference_mode and CUDA graphs and costs about +28% latency.
- (b) Soft replacement before each Euler step, `x += W·((1−τ)z + τY − x)`: zero extra cost, and the only variant that
  fits the megakernel. It adds Y, W and z as inputs to the Euler stage (T/tt/action_head.py:750-770, U:as.F17).
- (c) Training-time RTC: needs per-token AdaLN timesteps and a fine-tune (DP7).

---

## 4. Measurement and logging spec; evaluation protocol

### 4.1 Logs (written by P1; flushed only between episodes; `RT/runs/<run>/<episode>.npz` + `requests.jsonl` + `episodes.jsonl`)

**TickLog**, one row per tick, numpy structured array:

| Group | Fields |
|---|---|
| Timing | `k` i64, `t_start_ns` i64, `t_end_ns` i64, `period_us` f32, `compute_us` f32, `slack_us` f32, `overrun` u1, `slip` u1, `sim_us` f32, `render_us` f32 (trigger ticks) |
| WBC | `wbc_rtt_us` f32, `wbc_compute_us` f32 (server-reported), `wbc_backend` u1 (0 ORT, 1 GPU, 2 X280), `wbc_fallback` u1, `policy` u1 (0 Balance, 1 Walk) |
| Chunk | `chunk_seq` u32, `row` i16 (−1 = none), `age` i16 (ticks), `remaining` i16, `starved` u1, `in_flight` u1 |
| Commands and state | `cmd35` f32[35], `nav_wbc` f32[3], `q15` f32[15], `q_upper_meas` f32[17] (arms + waist), `pelvis_z` f32, `pelvis_rp` f32[2], `success` u1, `safety` u1 |

**RequestLog**, one row per VLA request:
- Identity and timing: `seq`, `gen`, `k_obs`, `t_obs_ns`, `t_send_ns`, `t_recv_ns`, `k_arrive`, `d`, `d̂`, `s`.
- Cost: `ser_us`, `deser_us`, plus the server's `info` timings (TT).
- Usage: `first_row`, `last_row`, `n_rows_used`, `outcome` ∈ {used, late_reject, gen_drop, timeout, error, skipped}.
- At the switch tick: boundary L2 jump (arms + waist), velocity-direction cosine and intra-chunk acceleration, using
  the formulas in R/getting_started/real_world_deployment.md:283-362 (U:as.F3).

**EpisodeLog:** `arm`, `mode`, `d_inj`, `s`, `B`, `seed`, `vla_backend`, `wbc_backend`, `success`, `t_success_ticks`,
`end` ∈ {SUCCESS, TRUNC, FAULT, FALL}, `n_starved`, `n_late`, `n_flips`, `n_fallback`, `max_boundary_jump`.

**Device telemetry (M2/M3):** x280d samples AICLK, L2CPUCLK3 and temperature at 1 Hz through its *already-open* pyluwen
handle. Never run tt-smi during a run: each fd open or close re-locks PLL4 four times (U:x.F6).

### 4.2 Gates computed by `tools/analyze_run.py`

| Gate | Threshold |
|---|---|
| Period | p99 ≤ 21 ms, max ≤ 40 ms |
| Overruns | ≤ 0.5% of ticks |
| WBC RTT p99 | ≤ 2 ms (GPU), ≤ 1.5 ms (X280) |
| VLA delay | d_p99 ≤ H − s − 2 |
| Starvation | 0 starved ticks in nominal runs |
| Chunk age | histogram checked against [d, s+d) |
| Falls | 0 loop-attributable falls |

Plots: a Gantt chart (ticks, request bars, switches) and histograms of period, d, age and RTT.

### 4.3 Evaluation protocol

- **Pairing.** Arm × seed cells use `env.reset(seed=k)` (rollout_policy and SyncVectorEnv do not seed, U:sim.F14). VLA
  noise stays stochastic on the GPU because Gr00tN1d6 calls `torch.randn` unseeded (gr00t_n1d6.py:310-314 ✔). Pairing
  removes scene variance only, which is what the McNemar test needs. TT runs may pass `noise_seed = hash(seed, call)` for
  replay debugging, but the statistics runs leave it unseeded to match the GPU reference.
- **Sizes.** Pilot: 50 episodes per arm, enough to catch drops of about 25 pts or more. Acceptance: **160 per arm**
  (45% vs 30% at α = .05 and power .8, U:as.M). 10-20 episode comparisons are banned because the README warns of ±15%
  (R/examples/GR00T-WholeBodyControl/README.md:9-13 ✔, reported 58%).
- **Matrix, M1 (VL):**

  | Arm | d_inj values |
  |---|---|
  | sync_frozen | {0} |
  | sync_hold | {3, 5} |
  | async_noskip | {3} |
  | v1 B=0 | {0, 3, 5, 8, 12} |
  | v1 B=5 | {3, 5} |

  That is 11 arms × 160 = 1,760 episodes. About 90 VLA calls per episode × 60 ms gives about 5.4 s of VLA time per
  episode, so roughly 3 h with 4 parallel sim processes.
- **Matrix, M1 RT** (native d): sync_hold and v1, 160 each, about 1.5 h per arm.
- **Matrix, M2:** sync_frozen-TT (VL), v1-TT (RT).
- **Matrix, M3:** v1-TT-X280 (RT), 160 each.
- **Controls.**
  - Server warm-up: 20 calls before each run.
  - One server instance per campaign. A dedicated GPU VLA on :5565 is preferred (DP0) so other clients of :5555 do not
    perturb latency.
  - Arms are interleaved per seed block to cancel thermal and clock drift.
  - P1 is pinned to core 2 and the servers to cores 8-15.
  - VRAM is checked before starting (≥ 8 GB free).
- **Statistics.**
  - Primary metric: success rate with Wilson 95% CIs, the paired difference with a bootstrap CI, and McNemar's test.
  - Secondary metrics: time-to-success, starvation, boundary jump, falls, flips.
- **Acceptance rule (all milestones):** paired point estimate ≥ −10 pts against the reference arm, with the CIs reported.
  Within an episode, a "drop" is attributed to the loop only when it is backed by log evidence (starved, fault, SystemExit
  or fallback).

---

## 5. Risk register

### 5.1 Register (each row names the WP and test that closes it)

| # | Risk | Evidence | Mitigation | Closes |
|---|---|---|---|---|
| R1 | **Harts leave reset only once per chip reset.** A hung payload means `tt-smi -r`, which kills the TT session. | ISA L2CPUTile/README.md:29-31 ✔; U:x.F3 | Resident monitor plus RNMI/GP-scratch park stub before the first release. Payloads only via LOAD/CALL. L2CPU3 first. Release ledger (§5.2). | WP-3.2/3.3 → T3.2 |
| R2 | tt-bh-linux flow breaks tt-metal (board reset, GDDR bank 5, 4 GiB windows, WAYENABLE=0xF) | tt-bh-linux README:40, boot.py:76-85,103 ✔; U:x.F0 | Never run boot.py. LIM-only. 2 MiB windows only. Never write WAYENABLE and assert it reads 0. | WP-3.0/3.1 → T3.0/T3.1 |
| R3 | A partial write to LIM before ECC is initialised can fault | ttnn-nanobind/cluster.cpp:118-120 ✔; U:x.F2 | Zero-fill every line in full-line writes first, and use full-line writes everywhere | WP-3.1 → T3.1 |
| R4 | The L2CPU clock is gated when no fd asks for it, and PLL4 re-locks on every fd open/close | U:x.F6/F7 (telemetry 0 / 800 MHz) | Keeper legacy fd in x280d. No fd churn or tt-smi during runs. Watchdog. | WP-3.4/3.7 → T3.3 |
| R5 | Host hangs on NoC access to a gated L2CPU tile | U:x.F7 | Check L2CPUCLK3 > 0 in telemetry before every access session. Probes run in a subprocess with a timeout. | WP-3.0/3.7 → T3.0 |
| R6 | The device guard runs `tt-smi -r` when its probe fails and no one holds the device, wiping the X280. A keeper fd also makes *other* DEV jobs wait (up to 1800 s) and then refuse. | bin/_device_guard.sh ✔ | One DEV hold per session that launches x280d, TT and P1 together (WP-3.9). The ledger detects a new epoch (bits 4-7 clear). Others' jobs queue on the flock as today. | WP-3.9 → T3.3 |
| R7 | pyluwen Hardcoded-TLB APIs clobber UMD windows; host TLB exhaustion | U:x.F9 | Audit 0.8.5 in WP-3.0. Use `allocate_tlb` 2 MiB windows only. Fall back to `ttnn.cluster.*` from a tt-metal process. | WP-3.0 → T3.0 |
| R8 | Watcher rejects non-LIM L2CPU addresses | sanitize_noc_host.hpp:147-155 (U:x.F1) | `TT_METAL_WATCHER` unset in every M3 session | WP-3.9 |
| R9 | Translated versus NOC0 L2CPU coordinates get mixed up and writes silently hit the wrong target | U:x risks | E0 reads CCACHE0_CONFIG at both (8,7) and (8,29). All addresses come from `regs.py` only. | WP-3.0 → T3.0 |
| R10 | X280 is slower than estimated (single L1D fill, bandwidth unknown) | U:x.F21 (low confidence) | Weights LIM-resident. v0 scalar (about 1.5 ms) still leaves a 13× margin. E4 measures it. | WP-3.6 → T3.5 |
| R11 | fp16 weights (7e-3) change closed-loop behaviour | U:x.F17 | Shadow mode first. Fallback to fp32 with one tile per policy (costs a second release). | WP-3.8 → T3.6 |
| R12 | Zvfh/Zvfhmin absent; VLEN unconfirmed | U:x.F15 | Integer widening. Trapped probe. `vlenb` reported in L1. | WP-3.5 → T3.2/T3.4 |
| R13 | **TT fidelity on the G1 fine-tune is unknown.** DiT and LLM layers 12-15 are fp32 in cloudwalk and run as bf16/bfp8 on TT, and hands deviate up to 1.5× GPU on GR1. | U:tt.F5/F13 | G1 golden. Gate at 2× the GPU bf16 deviation. bf16 arena fallback. DP2. | WP-2.4/2.6 → T2.2/T2.4 |
| R14 | The weight cache is keyed by the pinned sha, so G1 weights could poison base slot 8 | U:tt.F6 | Separate cache_root plus FORCE_CONVERT on first build | WP-2.2 → T2.1 |
| R15 | Working-tree drift (Stage-3 WIP in T) | U:tt.F14 | Run from P only | WP-2.5 |
| R16 | The port's default noise is fixed at seed 42 | T/tt/model.py:151, 942-947 ✔ | Fresh generator in the server | WP-2.5 → T2.3 |
| R17 | Two megakernel builds in one process hang the card | U:tt.F11 | One model per server process; restart to switch | WP-2.5 |
| R18 | **Async loses success versus sync** (rows a[d..] instead of a[0..]) | U:sim risks; U:as.F15 | SP-1 pilot in week 1, DP1, v2 track | WP-1.7 → T1.5/T1.6 |
| R19 | WBC blocked behind VLA on one REP socket (p99 67 ms) | server_client.py:70,117-143 ✔; U:srv.F1 | Separate process and port (D3). T1.4 proves the instrument detects it. | WP-1.6 → T1.3/T1.4 |
| R20 | Tick jitter from GIL, GC, serialization, EGL spikes (5.8 ms), EGL contending with GPU VLA (unmeasured), CPU contention (118 server threads) | U:sim.F11, U:srv gaps | One tick thread, gc off in episodes, trimmed payload, core pinning, renders only on trigger ticks, video encoding offline | WP-1.7 → T1.3 (SP-2) |
| R21 | Chunk-switch jump pushes an arm past 6 rad/s, `sys.exit`, and the thread dies silently | joint_safety.py:21 ✔, sync_env.py:525 ✔ | Catch SystemExit, log the boundary jump, B ∈ {3,5} option | WP-1.3 → T1.3 |
| R22 | Balance/Walk flicker near ‖cmd‖ = 0.05 | g1_gear_wbc_policy.py:257 ✔ | Log flips. Hysteresis is DP5. | WP-1.8 |
| R23 | Mixed time bases (monotonic vs sim) trigger the 1 s goal timeout or interpolation jumps | g1_decoupled_whole_body_policy.py:47,98 ✔ | `set_goal` every tick. Never pass sim time. IdentityPolicy only. | WP-1.3 |
| R24 | Reset stalls (2.5-2.8 s) and handles go stale because MjSim and WBC are re-created | n1_utils.py:29-33 ✔ | Reset outside the RT clock. Re-bind the backend and the observable tweak after reset. Bump gen. | WP-1.3/1.7 → T1.7 |
| R25 | CUDA in the sim corrupts EGL; `CUDA_VISIBLE_DEVICES=""` alone crashes EGL init | egl_context.py:39-55 ✔ | Set `MUJOCO_EGL_DEVICE_ID=0` as well. Assert `not torch.cuda.is_initialized()` at each episode end. | WP-1.3 → T1.1 |
| R26 | Statistical power (±15% variance) | README.md:13 ✔ | 160 per arm, paired seeds | WP-1.9 |
| R27 | Alignment of a[0] with t_obs is only inferred | U:sim gaps | Alignment probe: mean \|a_row0 − q_obs\| < mean \|a_row1 − q_obs\| over 500 chunks. async_noskip as a control. | WP-1.3 → T1.5 |
| R28 | GPU memory: a second VLA copy (7.3 GB) plus goldens on top of the running server (7.8 GB used of 32.6) | nvidia-smi ✔; U:srv risks | Sequence GPU jobs and check free VRAM before each | WP-1.6/2.4 |
| R29 | Device time is shared with other agents | bin/with-device.sh ✔ | Batched sessions: TT-D1 (+SP-4), X-D1, X-D2 | §6 |
| R30 | X280 debuggability (no UART) | U:x risks | mtvec dump to L1, cycle histogram, `ctl.py status` | WP-3.2 |
| R31 | NoC contention under VLA load raises mailbox latency | inferred | Measure idle versus busy in E4. Gate T3.5 is set with the VLA running. | WP-3.6 → T3.5 |

### 5.2 Sequencing X280 experiments safely under the once-only reset

**Release ledger** `~/experiments/gr00t/.x280_ledger.json`:
`{epoch_id, detected_at, tiles: {0..3: {state: fresh|released|parked|burned, image_crc, released_at, notes}}}`.
- A new epoch is detected when L2CPU_RESET bits 4-7 are all 0.
- `boot.py` refuses to set a bit that the ledger or the register says is already set.
- x280d *reattaches* when the heartbeat is advancing and the magic is valid.

**Order.** Each step runs only after the previous one passes. All steps run inside a DEV hold. `TT_METAL_WATCHER` is unset.
No other fd is opened during a step.

| Step | Releases spent | Worst case | Recovery | Preconditions |
|---|---|---|---|---|
| E0 read-only probe (inside TT-D1) | 0 | host hang on a gated tile | reboot (avoided by the telemetry check) | L2CPUCLK > 0 in telemetry |
| E1 LIM init and partial write (inside TT-D1) | 0 | LIM fault / wedged tile | `tt-smi -r` at the end of the session | E0 pass |
| Firmware unit tests on x86 | 0 | – | – | – |
| E2 monitor release on **L2CPU3**, at the **start** of the X-D1 hold, *before* the TT server builds | 1 (L2CPU3) | hart hangs | RNMI → park → reload. If RNMI fails: mark burned and `tt-smi -r`. This is cheap here because no TT session is up yet. | E1 pass; image CRC read back; RNMI and reset vectors read back; PLL4 at 800 MHz confirmed |
| E3 power survival: (a) second fd, (b) tt-smi snapshot, (c) TT server start/stop | 0 | heartbeat stops | park/reload; else chip reset | E2 pass |
| E3(d) all fds closed for 10 s, then reopen (optional) | 0 | host hang or frozen hart | reboot / chip reset | **user approval**; run last in the session |
| E4/E5 microbench and parity (payloads via LOAD/CALL) | 0 | payload hang | RNMI park | E3 pass |
| Closed loop (WP-3.8) | 0 | – | P1 falls back to ORT | T3.4/T3.5 pass |

Further rules:
- **L2CPU2 is used only** for the D9 fp32 fallback, or after root-causing an L2CPU3 failure. L2CPU0/1 are last resorts.
- If ≥ 2 tiles are burned in an epoch, stop and schedule a chip-reset window with the user.
- **Session start rule (demo):** inside one DEV hold, x280d runs first. It reattaches, or boots a fresh tile, with a
  chip reset only if every tile is used and the user allowed it. The TT VLA server starts next. Releases therefore
  always happen while nothing else is running, which turns R1's worst case from "tears down the TT session" into "costs
  a 1-minute restart".

---

## 6. Critical path, parallel tracks, decision points

### 6.1 Tracks and schedule (1 engineer plus agents; device sessions batched)

| Week | Track A: async loop (5090) | Track B: TT VLA | Track C: X280 |
|---|---|---|---|
| 1 | WP-1.0 to 1.5, 1.7 (VL) → **SP-1 pilot → DP1**; WP-1.6 → **SP-2 soak** | WP-2.1, 2.2, 2.3 (needs 1.10), 2.4, 2.5 (CPU) | WP-3.2 firmware, WP-3.5 x86 (**SP-6**) |
| 1-2 | WP-1.8 to 1.11 | **TT-D1 session (WP-2.6) → DP2** | **SP-4 (WP-3.0/3.1) inside TT-D1**; WP-3.3 |
| 2 | WP-1.12 campaign (T1.5-T1.7) | WP-2.8 if needed | **X-D1 (WP-3.4) → DP3** |
| 3 | M1 sign-off | WP-2.7 closed loop (T2.4-T2.6) | X-D2 (WP-3.6), WP-3.7 x280d |
| 4 | – | M2 sign-off | WP-3.8 shadow, then closed loop; WP-3.9 |
| 5 | – | – | M3 sign-off (T3.6/T3.7) |
| 6+ | M4 (optional) | M4.3 / RTC v2b | M4.1 / M4.2 |

**Critical path:** WP-1.1 → 1.2 → 1.3/1.4 → 1.7 (VL) → SP-1 → 1.8/1.9 → T1.5 campaign → WP-2.7 (T2.4/T2.5) → WP-3.8 (T3.6).
That is about 4.5 weeks. Track C finishes x280d (WP-3.7) by the end of week 3, so it has about 1 week of slack before WP-3.8
needs it. Track B's TT-D1 must finish before week 3. Every device step outside WP-2.7 and WP-3.8 is off the critical path.

**Effort totals:** M1 ≈ 87 h, M2 ≈ 44 h (+6 h conditional), M3 ≈ 90 h, M4 ≈ 120 h or more. Device time:
TT-D1 1.5 h, X-D1 2 h, X-D2 2 h, closed-loop campaigns about 12 h.

### 6.2 Decision points (data-driven)

| DP | When | Trigger and options |
|---|---|---|
| DP1 | after SP-1 (week 1) | §3.1 thresholds: v1 is enough / sweep plus RTC v2a/b early / debug alignment |
| DP2 | after TT-D1 | T2.2 fails, then bf16 arena (WP-2.8). Still fails: switch to the nvidia/GR00T-N1.6-G1-PnPAppleToPlate bf16 checkpoint (U:tt.OQ0) and re-baseline M1. |
| DP3 | after X-D1 | T3.2/T3.3 fail twice: escalate the BH-25/BH-28 behaviour to TT firmware owners. M3 proceeds with the WBC on host ORT behind :5558 (same contract, so the demo is not blocked) while X280 work continues. |
| DP4 | after WP-3.8 shadow | fp16 p99.9 > 1e-2 or success drop > 5 pts: switch to fp32 with two tiles (D9) |
| DP5 | after the M1 campaign | Walk/Balance flips > 1/s in more than 10% of episodes: adopt hysteresis (needs user OK) |

### 6.3 Open questions for the user

1. **DP0.** Use the running :5555 server for M1 VLA calls, or start a dedicated instance on :5565 (+7.3 GB VRAM; cleaner
   latency statistics)? This plan recommends :5565.
2. Which checkpoint for the demo: cloudwalk (fp32 parts; what this plan assumes) or the nvidia G1 bf16 fine-tune?
3. **X280 release budget:** may X-D1 spend L2CPU3's release, and may a `tt-smi -r` be run at a session start when
   needed? It ends any other tt-metal session, including other agents'. Is E3(d) (deliberate clock gating) allowed?
4. Is fp16-weight accuracy (about 7e-3) acceptable, or is fp32 required (two tiles, so two releases)?
5. Is the stateless 516-float WBC boundary OK for M3, with the whole controller moving to the X280 only in M4 (D2)?
6. Balance/Walk hysteresis: allowed if flicker shows up (DP5)?
7. RTC: is a training-time RTC fine-tune (v2c) in scope, or only the inference-time variants (v2a/b)?
8. Code home: a new repo at `~/experiments/gr00t/rt` (as assumed here), or inside Isaac-GR00T-n1.6?
9. Episode end: stop at the first success tick in every arm (MultiStepWrapper stops at the chunk boundary after success;
   the difference is ≤ 19 ticks and affects only time-to-success)?
