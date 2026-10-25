# GR00T N1.6 VLA + gear_wbc WBC, concurrent and asynchronous on one Blackhole p150a: PLAN (authoritative, rev 2)

Date 2026-10-24. Revision 2 applies the three reviews in `drafts/critiques.json` (control, tt, x280) to the merged
plan (drafts A/B/C, scored in `drafts/judge.md`). §7 lists every review issue and how it was resolved. The skeleton is
B's (risk first), with C's demo ladder, HUD and code reuse, and A's normative WBC spec, conformance suites and M4
schemas.

**Rev 2 documentation corrections (2026-10-25), no scope change:** the park stub jumps to crt0 at 0x0800_0000 and
never loads a reset vector (WP-3.2, §5.2 E1F/PARK; review:x280 F1); §1.4.4's 0x0800_0000 writer is the host only
before release (F4); §5.2 boot steps 3 and 6-8 read back and interlock as kmd does (F2, F5); §4.2 m3_sim d p99 ≤ 5 and
fallbacks ≤ 0.1% (T-3.3 = all T-1.4 gates); 13 extra stock keys, not 12 (§1.4.1, T-2.1); T-3.3 runs
`demo.sh m3 --x280-sim`; the S1 T-2.2 result is noted under T-2.2. Later the same day, to match the session jobs:
T-2.5/T-2.7 gate 0 server errors per arm; T-2.6/T-2.8 name their jobs (`t2_soak.sh`, `t2_soak_2h.sh`); T-2.7 and the
§4.4 M2 row run both arms fresh (T-1.9's GPU arm is not noise-paired); T-3.6 is a live shadow with ≤ 0.1% misses;
T-3.7's VLA band uses the named T-2.3 bench's server span; S1c ends in an S2 GO/NO-GO; §4.3 item 7: server_g1 has no
telemetry read.

**Evidence.**
- `research/understand.json`, cited as **U:sim / U:srv / U:tt / U:x / U:as** (its sections sim_control,
  serving_gpu, tt_port, x280 and async_practice), followed by the finding topic.
- Code checks made for this plan, marked **✔ file:line**. Checks made by a reviewer and re-checked by me are marked
  **✔r**.
- Primary-source clones for the X280 work live under `X` (below): tt-isa-documentation@0610df6, tt-bh-linux@9f9597d,
  tt-kmd-2.11.0@c05f224, luwen@ecef1a8 (0.10.0), fw-19.12.0@fe27c0e, tt-system-firmware@d12c837.

No repository file was changed, the Blackhole was not opened, and the GR00T server on 127.0.0.1:5555 was not touched.

**Paths.**

| Symbol | Path |
|---|---|
| `R` | `~/experiments/gr00t/Isaac-GR00T-n1.6` |
| `W` | `R/external_dependencies/GR00T-WholeBodyControl` |
| `TT` | `~/experiments/gr00t/tt-metal/models/experimental/gr00t` (live tree, untracked in git) |
| `MK` | `~/experiments/gr00t/publish/worktree-mk` (frozen snapshot 009de892, env `publish/worktree-mk-env.sh`) |
| `RT` | `~/experiments/gr00t/rtloop` (**new** git repo; all new code goes here) |
| `P` | `docs/plan/vla-wbc-x280` |
| `S` | `/tmp/claude-1000/-home-deepgadget-isaacsim/bfbe65f0-1b38-425e-84fc-4937eb211a38/scratchpad` |
| `X` | `S/plan_research/x280/src` (copied to `RT/third_party_refs/` in WP-1.0) |
| `DEV` | `~/experiments/gr00t/bin/with-device.sh` |
| `DEVX` | `env WITH_DEVICE_RESET_AFTER=1 $DEV`: **every hold that touches an L2CPU tile** (D13) |

**Interpreters and wrappers**, all defined in `RT/env.sh`:

| Wrapper | Interpreter | Used for |
|---|---|---|
| `simpy` | client venv `R/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python`, run with `cd $R`, `MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT:$R`, `GR00T_WBC_SERVER` unset | sim and clients |
| `srvpy` | server venv `R/.venv/bin/python`, run with `PYTHONPATH=$RT:$R` | GPU servers, goldens, Isaac reference |
| `ttpy` | `source MK env`, then `python`, with `PYTHONPATH=$PYTHONPATH:$RT` | TT server |
| `xpy` | `RT/.venv-x280/bin/python` (numpy + pyzmq + msgpack; **no pyluwen**) plus `RT/x280/host/libwcio.so` | x280d |

`MUJOCO_EGL_DEVICE_ID=0` is mandatory whenever `CUDA_VISIBLE_DEVICES` is empty. Without it robosuite's EGL init
evaluates `int("")` and crashes (✔ `W/gr00t_wbc/dexmg/gr00trobosuite/robosuite/renderers/context/egl_context.py:39-55`).

**Units.**
- Tick k: dt = 20 ms (50 Hz). H = 30 trained chunk rows.
- d = delay in ticks (k_arr − k_obs). s = ticks between VLA observations. B = crossfade ticks.
- Times are `CLOCK_MONOTONIC` ns. Angles in rad, linear velocity in m/s, yaw rate in rad/s.
- Effort in agent-hours (h). "pts" = percentage points of success rate.

---

## 0. Goal, scope, non-goals, decisions

### 0.1 Goal

A Unitree G1 in MuJoCo (`gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`, apple to plate) that behaves like a
real robot:
- The lower-body WBC (gear_wbc Balance/Walk MLPs, 516→15) runs on **every 50 Hz tick, unconditionally, paced to wall
  time**.
- GR00T N1.6 3B runs **in the background**. It turns observations into 30-step action chunks, which the controller
  splices in by tick stamp.
- End state: one Blackhole p150a, with the VLA on Tensix and the WBC on the on-die SiFive X280.

Every milestone ends in a runnable demo (live viewer plus HUD) and a scripted numeric verdict (`verdict.json`).

### 0.2 Demo ladder

| Step | VLA backend | WBC backend | On screen (HUD processor map, read from backend `info`) | Exit gate |
|---|---|---|---|---|
| M1.0 (≈ day 3) | mock server (60 ± 5 ms, deterministic chunks) | ORT CPU in process | robot stands and sways on mock chunks | T-1.1, T-1.2 |
| M1.1 (≈ day 5) | RTX 5090 | ORT CPU | apple task, async | T-1.4 (ORT variant) |
| **M1** | RTX 5090, RT-owned server `:5565` | **RTX 5090 CUDA graph, own process `:5556`** | "VLA 5090 / WBC 5090" | T-1.1…T-1.10 |
| **M2** | **p150a Tensix `:5557`** | RTX 5090 `:5556` | "VLA Tensix / WBC 5090"; GPU↔TT swap between episodes | T-2.* |
| **M3** | Tensix `:5557` | **X280 L2CPU3 hart 0 via x280d `:5558`** | "VLA Tensix / WBC-MLP X280 @800 MHz, n µs (mcycle)"; live X280↔GPU swap | T-3.* |
| M4a | Tensix | **stateful lower-body controller on X280** (history, select, FK) | "WBC X280 (full)" | T-4.1 |
| M4b | Tensix | X280 runs splicer + WBC and owns the 50 Hz clock | "controller on X280" | T-4.2 |
| M4c | Tensix → device DRAM → X280 | X280 | "chunk handoff on chip" | T-4.3 |
| M4d | RTC v2 (5090 reference; gradient-free on Tensix) | — | A/B overlay | T-4.4 |

### 0.3 Scope and non-goals

**In scope.**
- The new repo `RT`: control loop, splicer, wire clients, GPU VLA and WBC servers (RT-owned wrappers, no repo edit),
  mock/replay VLA, telemetry, HUD and eval scripts.
- The TT G1 VLA server.
- The X280 firmware, its host daemon and its own small KMD access layer.
- M1–M3 are mandatory; M4 is optional.

**Non-goals.**
- ROS2/DDS: not installed (U:sim "Template code is not runnable here"). We copy the semantics, not the code.
- Teleop and IK: the VLA path has no IK.
- Linux on the X280, or running it at 1750 MHz: CMFW re-locks PLL4 to 800 MHz (U:x "CMFW re-locks PLL4").
- ARC messages or pyluwen in the X280 runtime path (D12).
- TensorRT or torch.compile on the 5090. Batch > 1 on TT. Video encoding inside the control loop. Real hardware.
- Retraining. The only exception is the optional training-time RTC (DP8).

### 0.4 Decisions already made by the user

| # | Decision |
|---|---|
| U1 | The VLA runs on Blackhole Tensix with the existing tt-metal GR00T port. |
| U2 | The WBC runs on the Blackhole X280. Treat it as possible; the tt-bh-linux "not concurrent with tt-metal" note is outdated. Its concrete causes (full board reset, GDDR bank 5, 4 GiB host windows, WAYENABLE=0xF; U:x "Why tt-bh-linux cannot coexist") are each avoided here (§5.2). |
| U3 | Real-robot-like asynchrony: the WBC runs every tick and never waits for the VLA; the VLA runs in the background. |
| U4 | Order: M1 is the async loop validated on the RTX 5090 (sim on the host CPU, VLA and WBC served from the GPU), then M2 (TT VLA), then M3 (X280 WBC). |
| U5 | The MuJoCo sim process never creates a CUDA context, because that corrupts its EGL renders (✔ `R/gr00t/policy/gear_wbc_policy.py:1-8`). |

### 0.5 Decisions this plan makes

| # | Decision | Basis |
|---|---|---|
| D1 | **Two wire contracts, frozen from M1 to M3.** **W-VLA** is the Isaac-GR00T PolicyServer `get_action` (msgpack + npy), with an optional `options.rt` block that stock servers ignore. **W-WBC** is `gear_wbc_action(policy, obs[1,516]) → action[1,15]`, with optional `tick`/`seq`. Each milestone changes **one endpoint URL**. Test-only endpoints (`predict_debug`, `x280_peek`) exist only behind `--debug-endpoints` and are **outside** the frozen contract. | U:srv rec "Keep the endpoint and schema identical"; ✔ `R/gr00t/policy/server_client.py:117-143` |
| D2 | **The WBC boundary for M1–M3 is the stateless 516→15 MLP call.** Obs build, the 6×86 history, torso FK, the 1-tick command lag and the Balance/Walk switch stay in the verified Python. This makes every call bit-checkable against ORT and lets the client fail over at any tick with no state loss. The stateful 72→15 `LbStep` (§1.4.6) moves to the X280 in M4a. | ✔ `W/gr00t_wbc/control/policy/g1_gear_wbc_policy.py:102-198, 239-270`; U:sim "Proposed X280 interface" |
| D3 | **The WBC is never on the VLA's REP socket.** On the GPU it runs in `RT/rtserve/wbc_gpu_server.py`: its own process and CUDA context, serving REP `:5556` from its **main thread**, where `CudaGraphPolicy`'s captured stream is the calling thread's stream, so the latent stream bug cannot trigger. `GearWbcPolicies` is **vendored** into `RT/rtserve/vendor/gear_wbc_policy.py` with its sha256, because the source file is untracked in `R`. | U:srv "Q1 measured head-of-line blocking" (p99 67 ms shared) vs E3 (separate CUDA process p99 0.62 ms); ✔r `gear_wbc_policy.py:83-92`; ✔ `git status` in R: `?? gr00t/policy/gear_wbc_policy.py` |
| D4 | **One single-threaded, CUDA-free control process `rt_sim`** holds the sim, the splicer, a non-blocking VLA client (one REQ, `poll(0)` per tick) and a WBC client (REQ, `RCVTIMEO` 5 ms). The tick path runs on one thread pinned to CPU 2; library threads (viewer, ZMQ I/O, EGL) are pinned to CPU 3 (§1.1). The VLA "background" is its server process. | U:srv "Q2 GIL contention" (wake-up p99 ≤ 0.33 ms); U:sim mock prototype (20.00 ± 0.35 ms) |
| D5 | **The WBC seam is a class-level patch of `G1GearWbcPolicy.load_onnx_policy`.** It returns a closure over a process-wide backend singleton keyed by policy stem. The patch survives the per-reset policy re-creation and skips both ORT session creation and the hook's verification RPC. `GR00T_WBC_SERVER` is asserted unset in `rt_sim` and in preflight. | ✔r `g1_gear_wbc_policy.py:31-36, 56-73`; ✔r `W/gr00t_wbc/control/utils/n1_utils.py:29-33` |
| D6 | **The ego image is the env observation of the tick.** Cameras are sampled every tick: 5.26 ms/tick, 3.76× real time. This matches the stock path, which the equivalence tests need. Rendering on demand is an optimisation used only if T-1.4 fails. | U:sim "Real time with on-demand images" (saves only 13%) |
| D7 | **No future leak.** Observation o_k (state at the end of physics step k−1) is sent **no earlier than T_k = t0 + k·dt**, so measured latency is honest. In RT mode d ≥ 1 by construction. | The paced sim finishes step k−1 about 15 ms early |
| D8 | **Chunk scheduling v1 is the aligned receding horizon** (§3.1): s = 16, B = 0, latency skip by tick stamp. RTC is v2 (M4d). | U:as rec "V1 STRATEGY" |
| D9 | **RT (wall clock) mode is the demo and the timing gates. VL (virtual latency) mode is for statistics sweeps and equivalence.** | U:sim rec "Two evaluation modes"; U:as rec "V1 RUN MODES" |
| D10 | **The M2 server is a port of the validated LIBERO TT server**, `TT/libero/server_n16.py` (PolicyServer wire, fresh noise from a seeded generator, `info`, boot golden check, 100/100 closed-loop episodes with 0 errors). It is **copied into RT**, runs on MK by default (MK has no `libero/`, ✔), filters the request to the modality keys before `split_batch`, and binds `127.0.0.1`. DP6 chooses MK or the live tree. | ✔ `TT/libero/server_n16.py:142-207, 239`; ✔ `TT/libero/n16.py:255-266`; ✔r `libero_eval/n16/tt_spatial_r0_summary.json` |
| D11 | **One checkpoint for M1 exit, M2 and M3.** Default: **`cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc`**. In this env on the 5090 it measured 5/11 with ORT WBC and 8/20 with the GPU WBC. The official `nvidia/…@897d0313` (README 58%) measured **0/10** in the same env (balances, never grasps). Cloudwalk has 517 F32 tensors, so WP-2.1b (CPU F32→BF16, 4 h) is **expected**, not conditional. The nvidia checkpoint needs no conversion (all 1106 tensors BF16, base byte layout) and stays the alternative. It is served for M1 by the RT-owned GPU server on `:5565`. SP-7 confirms (DP1). | 2026-10-24 5090 runs (memory `vla-rl-single-chip-demo`); ✔r snapshot headers; U:tt "Cloudwalk checkpoint dtype mismatch" |
| D12 | **X280:** bare metal from LIM only. WAYENABLE and GDDR are never touched. The first image is a **resident monitor** (LOAD / FENCE_I / CALL / INFER / PARK / SELFTEST), released first on L2CPU3 (8,7). The clock stays at 800 MHz. The daemon `x280d` (xpy) serves W-WBC on `:5558` and reaches the chip **only through its own ctypes KMD layer** (`kmd.py`: ALLOCATE/CONFIGURE/FREE_TLB, SET_NOC_CLEANUP; UC and WC mmaps) and a C helper (`libwcio.so`) that writes LIM in 64 B-aligned full-line non-temporal stores. It never sends ARC messages and does not use pyluwen. | U:x rec (1)-(4), (9), (10); ✔r luwen maps only the UC aperture and copies u32-wise (`X/luwen/crates/luwen-kmd/src/lib.rs:412-429`, `pci.rs:441-520`); ✔ tt-kmd `ioctl.h:271-282` exposes `mmap_offset_wc` |
| D13 | **X280 session model: one hold per session, reset after every hold.** Every X280 session is one `DEVX` hold, so the guard resets the chip after the job (✔ `bin/_device_guard.sh:122-130`). Its session script probes, then starts x280d **before** the TT server; children are restarted inside the hold and **`DEV` is never nested**. One release per tile per session (§5.2; DP3). | ✔ `bin/_device_guard.sh`, `bin/with-device.sh`; ISA "Reset" (✔ `X/tt-isa-documentation/BlackholeA0/L2CPUTile/README.md:29-30`) |
| D14 | **No repository file changes in M0–M3.** M4d's TT megakernel blend is the first planned repo change (DP8). | task rules |
| D15 | **Ports.** 5555 is the existing server: never stopped; used only by SP-7/C0 (cloudwalk screen) and the negative control T-1.5. 5556 GPU WBC, 5557 TT VLA, 5558 x280d, 5559 mock/replay VLA, 5565 RT-owned GPU VLA, 5570 telemetry PUB, 5571 control PULL, 5572 server_g1 → x280d PUSH (M4c only), 5573 x280d status PUB, 5574 server_g1 status PUB (HUD chip panel, §4.3). | U:sim "Port collision" |
| D16 | **WBC fallback and shadow.** The client chain is primary → in-process ORT CPU (stateless, so safe at any tick). ORT also runs **every tick as a shadow** (0.024–0.08 ms) and logs max\|Δa\|. | U:srv "Measured idle latencies" |
| D17 | **Common random numbers (CRN) for paired evaluation.** With `--crn`, `rt_sim` sends `options.rt.noise_seed = splitmix64(ep_seed ≪ 20 \| seq)`. The RT-owned GPU server calls `torch.cuda.manual_seed(noise_seed)` before each call; server_g1 reseeds its generator. Noise stays fresh per call, so the policy distribution is unchanged; only arm-to-arm variance drops. | ✔r unseeded `torch.randn(..., device=cuda)` at `R/gr00t/model/gr00t_n1d6/gr00t_n1d6.py:310-314` |
| D18 | **Milestone exits are functional; success rates are reported, not gated.** User decision (2026-10-24): **50 episodes per arm is enough**. Each exit needs the functional and timing gates plus one 50-episode paired comparison per milestone, reported with Wilson CIs, the paired difference and exact McNemar p. With n = 50 the comparison only flags gross drops (lower 80% bound ≥ −25 pts); no non-inferiority claim is made. The n = 350 tier is dropped (§4.4). | user DP10; review C-5 (power simulation) |

### 0.6 Fixes to the drafts (details in `drafts/judge.md` §3–4; review fixes are in §7)

| Problem in a draft | Fix in this plan |
|---|---|
| A: `CUDA_VISIBLE_DEVICES=""` without `MUJOCO_EGL_DEVICE_ID` crashes EGL | `simpy` sets both; T-1.7 checks frames |
| A: restarts "with `WITH_DEVICE_NO_PROBE=1`" inside a held session deadlock on the flock | chip_session.sh starts and restarts every child itself; `DEV` is never nested (D13) |
| A/B/C: n = 160 used as if it supported a 10-point non-inferiority claim | D18 and §4.4 |
| C: "stock episodes do not end on success" (✗ `R/gr00t/eval/rollout_policy.py:466`) | first-success tick recorded in every arm; equivalence compares up to the official end |
| C: "run the LIBERO server from MK" (MK has no `libero/`) | ported into RT; import test on MK; live tree as fallback (DP6) |
| A: stateful WBC boundary and 5-process lockstep in M1 | stateless boundary for M1–M3; A's `LbStep` and LowState/LowCmd become the M4a/M4b contracts |

---

## 1. Target architecture

### 1.1 Processes, threads, ownership

```
 host CPU (Ryzen 9 7950X)                                RTX 5090                        Blackhole p150a
 ┌ rt_sim (simpy, CUDA-free; tick thread CPU 2, ───────┐
 │   viewer/ZMQ/EGL threads CPU 3, CPU 18 idle)        │ W-VLA  ┌─────────────┐ M1: gpu_vla_server :5565 (RT-owned)
 │ robosuite/MuJoCo env + WholeBodyControlWrapper      │──────▶ │ VLA backend │ M2+: server_g1 :5557 ─────UMD────▶ Tensix + DRAM
 │ 50 Hz pacer · ChunkSplicer (pure functions)         │        └─────────────┘     (:5555 cloudwalk: untouched)
 │ VlaClient: 1 REQ, send(NOBLOCK), poll(0) per tick   │ W-WBC  ┌─────────────┐ M1/M2: wbc_gpu_server :5556 ───▶ 5090 CUDA graph
 │ WbcClient: REQ, RCVTIMEO 5 ms → ORT fallback        │──────▶ │ WBC backend │ M3: x280d :5558 ─KMD 2 MiB WC/UC─▶ LIM mailbox ▶ X280 L2CPU3 h0
 │ ORT CPU shadow · TickLog memmap ring · PUB :5570    │        └─────────────┘
 └─────────────────────────────────────────────────────┘
 hud.py (SUB :5570, PUSH :5571)      chip_session.sh under DEV/DEVX: probe → [reset] → x280d → server_g1 → job
```

| Process | Interpreter | Rate | Owns | CPU pin |
|---|---|---|---|---|
| `rt_sim` (`RT/rtloop/loop.py`) | simpy | 50 Hz paced | MjSim, onscreen viewer, splicer state, VLA and WBC client sockets, ORT shadow, episode control, JointSafetyMonitor | tick thread 2; helper threads 3; CPU 18 (SMT sibling of 2) kept idle |
| `hud.py` | simpy | 30 fps | display only (SUB, drops on overflow) | 6–7 |
| `wbc_gpu_server.py` | srvpy | 50 Hz | its own CUDA context (< 1 GB), Balance/Walk CUDA graphs | 4 |
| `gpu_vla_server.py` `:5565` (RT-owned wrapper of Gr00tPolicy + Gr00tSimPolicyWrapper + PolicyServer) | srvpy | ≈ 3 Hz | about 8 GB on the 5090, the DP1 checkpoint | 12–15 |
| GPU VLA `:5555` (existing) | existing | – | untouched | – |
| `server_g1.py` | ttpy | ≈ 3 Hz | Tensix, DRAM, 2 CQs and traces **via UMD** (CHIP_IN_USE mutex: no second tt-metal process can share the chip, ✔r `tt_metal/third_party/umd/device/chip/local_chip.cpp:148-151`). Shares the device *node* with x280d. One model per process; all ttnn calls on the main thread | 8–11 |
| `x280d.py` | xpy | 50 Hz | L2CPU3 lifecycle, keeper legacy fd, ~5 KMD 2 MiB windows (LIM, ARC, CCACHE/L2PF, BEU, external peripherals), LIM mailbox, `SET_NOC_CLEANUP` | 5 |
| `chip_session.sh` | bash under `DEV`/`DEVX` | – | the p150a flock for the whole session; order probe → [reset → re-probe] → x280d → server_g1 → job; teardown in reverse | – |

**SMT siblings (measured 2026-10-24, 7950X):** 2↔18, 3↔19, 4↔20, 6↔22, 7↔23, 12–15↔28–31. Background work (builds, tests, agents) must stay off the siblings of the real-time cores; use 0, 1, 8–11, 16, 17, 24–27 (5↔21 is reserved for x280d in M3). T-1.4's first full-load WBC RTT p50 (1.004 ms) was taken with build agents on CPU 20, the sibling of the WBC server's CPU 4.

Affinity is applied in this order and recorded in `config.json`: the main thread runs on CPU 3 while it builds the env,
creates the ZMQ context (`ZMQ_THREAD_AFFINITY_CPU_ADD=3`) and launches the viewer, so every library thread inherits
CPU 3; then the main thread moves itself to CPU 2 for the tick loop. The same switch brackets the viewer relaunch in
each reset window (§1.2).

**Device resources by milestone.**

| Resource | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| 5090 CUDA | VLA `:5565`, WBC graphs `:5556` | WBC `:5556`; goldens (with `:5565` stopped) | optional reference | free |
| 5090 EGL | `rt_sim` renders | same | same | same |
| Tensix, DRAM banks 0–7, 4 GiB windows, about 158 2 MiB windows | idle | **server_g1** | same | plus the persistent chunk tensor (M4c) |
| L2CPU3 (8,7): harts, LIM | untouched | untouched (S1c: read, LIM write, no release) | **x280d**: monitor + WBC | plus controller and splicer |
| L2CPU2 (8,5) | – | – | spare (fp32 Walk under DP4) | – |
| L2CPU0/1 | – | – | reserved | – |
| ARC `L2CPU_RESET` 0x8003_0014, PLL4 0x8002_0500 | never | read only (S1c) | x280d `boot` only, once per session | – |

**Real-robot mapping** (template: gr00t_wbc's stack, U:sim "Real-robot loop structure").

| rtloop | gr00t_wbc real stack | Unitree G1 |
|---|---|---|
| `rt_sim` tick loop | `run_g1_control_loop.py` ControlPolicy at 50 Hz | onboard PC |
| VLA server + `VlaClient` | `run_teleop_policy_loop.py`, a separate goal producer | – |
| ChunkRecord mailbox with `gen` | `ControlPolicy/upper_body_pose` topic: depth 1, latest wins, `target_time` (row i ↔ `t_obs + 0.02·i`) | – |
| STARVE_FAULT after 50 ticks | `G1DecoupledWholeBodyPolicy` 1 s goal timeout | – |
| M4b LowState/LowCmd | rt/lowstate, rt/lowcmd | motor drivers, PD |

**Fidelity deviations**, all recorded in the run's `config.json`:
- (a) Physics waits for the WBC call inside the tick (at most 2 ms of 20; M3 early dispatch, WP-3.8b, hides it).
- (b) State and image come from the same tick, with no sensor latency.
- (c) The sim knee kp is 300/200 where the robot uses 200/200 (U:sim "PD and physics").
- (d) The first chunk is fetched before t0 is anchored (pre-roll, the same as stock).
- (e) VLA replies are quantised to tick polls, as on any 50 Hz controller.
- (f) With `--viewer`, the env is built with `onscreen=True` (stock: `onscreen=False`, ✔ `rollout_policy.py:104-109`;
  live viewer: `onscreen=True`, ✔ `S/live_viewer.py:100-108`). T-1.7 checks the ego frames are identical either way.
- (g) robosuite destroys the mjviewer on every reset and relaunches it inside the next `env.step`
  (✔ `robosuite/environments/base.py:270-272, 478-484`; ✔ `renderers/viewer/mjviewer_renderer.py:26-33`). rt_sim
  relaunches it in the reset window instead (§1.2), so no window creation lands in tick 0.

### 1.2 Per-tick data flow (tick k; deadline T_k = t0 + k·20 ms)

| # | Step | Budget (measured basis) |
|---|---|---|
| 1 | **Pace** (RT only). Sleep until T_k and record `late_us`. If more than one tick late, log a **slip** and re-anchor t0 (no catch-up burst). | wake-up p99 ≤ 0.33 ms (U:srv "Q2 GIL contention") |
| 2 | **Poll the VLA.** `poll(0)`. If a reply is ready, decode it, check `gen`, and build a ChunkRecord with `k_recv = k`. RT: register it with `on_arrival` (`k_arr = k`). VL: store it as *arrived, not visible*. | decode 0.11 ms |
| 3 | **Trigger** (§3.1). If it fires: take the 9 keys from o_k (D6, D7), cast state to f32 (1,1,D) and the image to u8 (1,1,480,640,3), `send(NOBLOCK)` with `options.rt`, and set `k_req = k`. The sent `state43` is kept for the RequestLog. | serialise 0.11–0.14 ms, 0.92 MB |
| 3b | **VL visibility** (VL only). If the in-flight record has `k_obs + d_inj ≤ k`: block until its reply has arrived, then register it with `on_arrival` at `k_arr = k`. With d_inj = 0 this blocks on the request sent in step 3 of the same tick, so row 0 executes at k_obs, exactly as in sync. | – |
| 4 | **Select.** `cmd35 = select(k)`: row `k − k_obs` of the active chunk, or hold/ramp when starved. | < 0.05 ms |
| 5 | **Step.** `env.step(cmd35)` runs `WholeBodyControlWrapper.step` (✔ `n1_utils.py:35-48`): `concat_action` → `set_goal` (every tick, so the 1 s monotonic timeout never fires) → `get_action`. Inside `get_action`: torso FK; Balance if ‖cmd‖ < 0.05, else Walk (✔ `g1_gear_wbc_policy.py:257`); the patched `run_inference` sends W-WBC; then `q15 = a·0.25 + default`. Then the JointSafetyMonitor, 10 × 2 ms physics with Python PD, o_{k+1}, `set_observation` (history push, hence the 1-tick command lag), and `viewer.sync()`. | sim 5.3 ms (2 cameras); WBC round trip 0.6 ms GPU / 0.03–0.08 ms ORT |
| 6 | **Log.** Write the TickLog row (memmap ring) and a non-blocking `PUB tick`. About every 16 ticks, `PUB ego` with a 160×120 thumbnail. | ≤ 0.05 ms |
| 7 | **Episode control.** Success is latched at the first tick with `info["success"]`. Truncate at 1440 ticks. A fall is pelvis z < 0.50 m (standing ≈ 0.752 m). A `SystemExit` from the safety monitor ends the episode as **SAFETY_FAULT**. 50 consecutive starved ticks set the **STARVE_FAULT** flag; the episode continues (§3.1). | – |

Total ≈ 6–7 ms of a 20 ms tick, leaving about 13 ms of slack. A 5 ms WBC timeout plus the ORT fallback still fits.

**Episode lifecycle.**
1. **Drain.** If a VLA request is in flight at episode end, `recv` with a timeout of up to 500 ms (the reset that
   follows takes 2.5 s anyway) and discard the reply by `gen`; if nothing arrives, close and re-create the REQ socket.
   Counted as `drain_ok` / `drain_recreate` (a REQ socket cannot send while a reply is outstanding).
2. **Reset** outside the clock: 2.5–2.8 s; MjSim and WBC are re-created (U:sim "Things that do break").
3. `gen += 1`.
4. **Viewer pre-launch** (with `--viewer`): with the main thread on CPU 3, call the env renderer's `update()` once
   (launch + sync, no physics) through `live_viewer.patch_viewer_camera`'s hook, assert the viewer handle exists, then
   move the main thread back to CPU 2.
5. **Pre-roll:** the first VLA call blocks with the sim not stepping. Chunk 0 has d = 0, as in stock. It is excluded
   from the d statistics.
6. Anchor t0 and start ticking from k = 0. Tick 0 is inside all timing gates; `viewer_us` at k = 0 is logged.
7. End at the first success tick (SUCCESS), at 1440 ticks (TRUNC), on FALL, or on SAFETY_FAULT.

The stock CLI ends at the chunk boundary after success (✔ `rollout_policy.py:466`, `multistep_wrapper.py:306-308`).
The success outcome is identical; the equivalence tests compare trajectories up to the stock end tick.

### 1.3 Rates

| Loop | Rate | Cost |
|---|---|---|
| Physics + joint PD | 500 Hz (10 × 2 ms substeps) | 1.32 + 1.95 ms per tick (U:sim "Sim timing") |
| Control tick (sim + host WBC part) | **50 Hz** wall-paced | 5.26 ms with 2 cameras |
| WBC MLP | 50 Hz, one call per tick | ORT 0.024–0.078 ms; GPU via ZMQ p99 0.92 ms; **X280: unknown until T-3.E2m** (below) |
| VLA | ≈ 3.1 Hz (s = 16) | 5090 trimmed 58.8 ms p50, 65–68 ms when paced; TT ≈ 42–47 ms (projection) |
| Telemetry PUB / HUD | 50 Hz / 30 fps | out of process |
| x280d health | 1 Hz direct PLL4/heartbeat reads on its open windows (no telemetry snapshot, no tt-smi) | – |

**X280 timing basis.** LIM is **not data-cacheable**: the cacheable ranges are the Zero Device, local GDDR and TLB
windows only (✔ `X/tt-isa-documentation/BlackholeA0/L2CPUTile/Caches.md:26-31`). Every weight load from LIM is an
uncached access, so U:x's 0.1–0.6 ms (which assumed 4–16 B/cycle cached) is withdrawn. Working estimates, low
confidence, at 800 MHz: vector kernels with 64 B uncached loads at 20–60 cycles per line give ≈ 0.4–1.1 ms (fp16
weights, 14.7k lines) and ≈ 0.7–2.2 ms (fp32); a scalar kernel at about 25 cycles per weight load gives ≈ 15 ms,
which only works as a parity reference. Code fetched from LIM is cached in L1I per Caches.md:13-18 (tt-bh-linux's
PMA table omits "I", README.md:273); E2m measures it. The first payload after the monitor selftest is a LIM
microbench (T-3.E2m) that replaces these numbers before WP-3.7 picks a kernel. Mitigations: early dispatch (WP-3.8b)
and the relaxed `m3_rt_slow` gate set (DP11).

### 1.4 Message schemas

#### 1.4.1 W-VLA request

`{"endpoint": "get_action", "data": {"observation": OBS, "options": OPT}}`. The wire is msgpack, with ndarrays as
`{"__ndarray_class__": True, "as_npy": <np.save bytes>}` (✔ `server_client.py:15-46`). The keys are derived from
`get_modality_config()` at startup and asserted to equal this table:

| Key | dtype | shape | Units / notes |
|---|---|---|---|
| `video.ego_view` | uint8 | (1,1,480,640,3) | RGB, the env's `robot0_oak_egoview` obs at k_obs |
| `state.left_leg`, `state.right_leg` | float32 | (1,1,6) each | rad |
| `state.waist` | float32 | (1,1,3) | rad (yaw, roll, pitch) |
| `state.left_arm`, `state.right_arm` | float32 | (1,1,7) each | rad |
| `state.left_hand`, `state.right_hand` | float32 | (1,1,7) each | rad |
| `annotation.human.task_description` | tuple[str] | (1,) | instruction (13 tokens; static layout L_max = 24) |

- `OPT = {"rt": {"seq": u32, "gen": u32, "ep": u32, "k_obs": i64, "t_obs_ns": i64, "t_send_ns": i64, "noise_seed": u64 (optional, D17)}}`.
  `BasePolicy.get_action` accepts `options` and Gr00tPolicy ignores it (✔ `R/gr00t/policy/policy.py:65-87`,
  `gr00t_policy.py:306-320`).
- The payload is 0.92 MB, against 9.22 MB for the full obs dict (latency 58.8 vs 81.4 ms p50).
- server_g1 drops every key that is not in the table before `split_batch`. The stock rollout (T-2.5 anchor) sends 13
  more keys (22 in all, recorded on seed 1 by `RT/tests/official_arm.py`; `video.tpp_view` included), among them two
  f32 (1,1,480,640,3) camera copies, which `split_batch` would otherwise turn into Python lists at about 17.5 ms per
  frame (✔r `TT/libero/n16.py:259-266`; ✔ `RT/tests/tt/test_tt_server_protocol.py` `STOCK_EXTRA`).

**W-VLA reply** `[ACT, INFO]`:

| Key | dtype | shape | Units |
|---|---|---|---|
| `action.left_arm`, `action.right_arm` | float32 | (1,30,7) | rad, **absolute**. Converted from relative to absolute on the server against the request's state (✔r `state_action_processor.py:461-486`, `action_chunking.py:340-355`); never re-anchor. |
| `action.left_hand`, `action.right_hand` | float32 | (1,30,7) | rad. Near-binary; never blended. |
| `action.waist` | float32 | (1,30,3) | rad; feeds only the WBC torso-RPY command |
| `action.base_height_command` | float32 | (1,30,1) | m (0.74 in training) |
| `action.navigate_command` | float32 | (1,30,3) | vx, vy m/s (±0.5); wz rad/s (±1.0) |

- `INFO` is `{}` from the stock server (✔r `gr00t_policy.py:358`).
- From `gpu_vla_server` (RT-owned): `{"rt": <echo>, "t_recv_ns", "t_reply_ns", "infer_ms", "noise_seed", "backend": "cuda:rtx5090:<ckpt@sha8>"}`.
- From `server_g1`: the same plus `"encode_ms", "device_ms", "decode_ms"` and
  `"tree": "MK@<git HEAD12>" | "live@<sha256(sorted contents of models/experimental/gr00t/{common,tt}/**/*.{py,cpp,hpp,h})[:12]>"`,
  computed from `models.experimental.gr00t.__file__` at startup. The live tree is untracked (✔r `git ls-files` = 0),
  so a git hash cannot identify it.
- DiT rows 30–49 are never used: they are untrained and loss-masked (U:as "N1.6 action horizon").

**Test-only endpoint (server_g1, `--debug-endpoints`; outside D1):**
`predict_debug(observation, noise: f32(1,50,128)) → {action_pred_norm: f32(1,50,128), actions: {…as above…}, encode_ms, device_ms, decode_ms}`.
It calls `predict_normalized(mi, noise)` exactly like the per-call path (✔ `TT/libero/server_n16.py:174-182`). It
serves T-2.2, T-2.4, `bench_g1.py --abab/--determinism` and the S1c bit-identity checks. Without the flag it replies
`{"error": "debug endpoints disabled"}`. An in-process ttnn test cannot replace it while server_g1 holds the chip
(CHIP_IN_USE).

#### 1.4.2 ChunkRecord

Internal and logged. Packed little-endian; a numpy structured dtype in `rtloop/schema.py` that mirrors a C struct.
4,264 B.

| off | Field | Type | Meaning |
|---:|---|---|---|
| 0 | seq | u32 | request sequence |
| 4 | gen | u32 | generation; bumped on reset, instruction change or VLA swap |
| 8 | k_obs | i64 | tick whose start state and image the observation shows |
| 16 | k_arr | i64 | tick at which the chunk became visible (RT: arrival tick; VL: k_obs + d_inj) |
| 24 | t_obs_ns | i64 | T_{k_obs} |
| 32 | t_send_ns | i64 | |
| 40 | t_recv_ns | i64 | |
| 48 | d | i16 | k_arr − k_obs |
| 50 | vla_be | u8 | 0 mock, 1 GPU, 2 TT |
| 51 | status | u8 | 0 ok, 1 error, 2 timeout, 3 bad_shape |
| 52 | pad | u8[12] | aligns `rows` to 64 B |
| 64 | rows | f32[30][35] | **A35** decode order: L arm [0:7], R arm [7:14], L hand [14:21], R hand [21:28], waist [28:31], base_height [31], nav [32:35] |

**Index law:** row i targets tick k_obs + i. This matches MultiStepWrapper, which applies row 0 on the first env step
after the observation (✔r `multistep_wrapper.py:256-269`). T-1.3 shows the loop reproduces this convention; it does
not show the model was trained with that convention (R13).

#### 1.4.3 W-WBC

**Request:**
`{"endpoint": "gear_wbc_action", "data": {"policy": "GR00T-WholeBodyControl-Balance" | "GR00T-WholeBodyControl-Walk", "obs": f32(1,516), "tick": i64, "seq": u32}}`.

**Reply:** `{"action": f32(1,15), "tick": i64, "seq": u32, "compute_us": f32}`.

- `action` is the raw policy output: legs 12 + waist 3. The host computes `q15 = a·0.25 + default_angles`.
- `tick`/`seq` are sent only to servers that answer `info` with `proto ≥ 2` (5556, 5558). The stock endpoint is called
  as `handler(**data)` and accepts only `(policy, obs)` (✔ `server_client.py:139`, `gear_wbc_policy.py:107`).
- Extra endpoints: `ping`; `info` → `{"backend": "cuda-graph:rtx5090" | "x280:l2cpu3:h0:<kernel>@<MHz>" | "x280-sim", "proto": 2, "policies": [...]}`.
- `obs` is 6 frames × 86 floats, oldest first, zero-filled at the start of an episode (✔r `g1_gear_wbc_policy.py:143-198`).
  Each frame:

| Slice | Content |
|---|---|
| [0:3] | nav cmd × [2, 2, 0.5] |
| [3] | height cmd, m |
| [4:7] | roll, pitch, yaw cmd, rad |
| [7:10] | pelvis ω × 0.5 (body frame) |
| [10:13] | gravity in the pelvis frame |
| [13:42] | q29 − default (15 defaults, padded with 0) |
| [42:71] | dq29 × 0.05 |
| [71:86] | last raw action (15) |

Because of the 1-tick command lag, the MLP input for tick k+1 is complete after `set_observation` at the end of tick
k; only the Balance/Walk choice depends on the tick-(k+1) command (✔r `g1_gear_wbc_policy.py:154-156, 184, 239-264`).
WP-3.8b's early dispatch uses this without changing any number.

#### 1.4.4 X280 LIM mailbox (M3, L2CPU3; layout v2)

LIM is 0x0800_0000–0x081D_FFFF, 1,966,080 B (U:x "L3 LIM capacity"). Rules:
- **Every LIM line is ECC-initialised before any partial write**, by the method E1a proves (§5.2): A = `libwcio` 64 B
  non-temporal stores through a WC mapping plus `sfence`; B = `ttnn.cluster.write_to_core` in 64 B-aligned 64 B
  chunks (server_g1 debug endpoint); C = a hart-run stage-0 initializer (E1F). pyluwen's UC u32 copies cannot do
  this (D12). tt-metal's sources disagree on what is sufficient: "write a full 64-byte line first"
  (✔ `ttnn/cpp/ttnn-nanobind/cluster.cpp:118-120`) versus "initialising it requires running code on the L2CPU itself"
  (✔ `tests/tt_metal/distributed/test_hd_sockets.cpp:463-467`). E1a settles which applies here.
- Each 64 B line has exactly one writer; the KMD cleanup write has its own line (L38).
- The mailbox is uncached on both sides. x280d configures each window once; there is no per-call TLB ioctl.

| Address | Lines | Writer | Content |
|---|---|---|---|
| 0x0800_0000 | 64 KiB | host before release (the image, boot step 3); X280 after release (.bss, .persist, stack: the whole 64 KiB region, ✔ `RT/x280/regs.py` `lim_line_owner`) | monitor + kernel code, rodata, data and bss, plus a 16 KiB stack; image ≤ 48 KiB. The E1F image is the only host write after a release, made before the entry word. Option E4c: stack and activations (≈ 25 KB) in the Zero Device 0x0A00_0000 (cached) |
| 0x0801_0000 | L0 | host | body: `magic u32 'WBC2'` @0, `cmd u32` @4 (0 NOP, 1 INFER, 2 LOAD, 3 FENCE_I, 4 CALL, 5 PARK, 6 SELFTEST, 7 BENCH), `policy u32` @8, `flags u32` @12 (b0 host_alive), `tick i64` @16, `arg_addr u64` @24, `arg_entry u64` @32, `arg_len u32` @40, `crc32 u32` @44 over bytes 0–43, `obs_crc32 u32` @48 over L2–L34; **`seq_req u32` @60, written last as its own 4 B UC write** |
| 0x0801_0040 | L1 | X280 | `heartbeat u64` (mtime; host reads hi, lo, hi), `seq_done u32`, `state u32` (0 boot, 1 idle, 2 busy, 3 fault, 4 reentered), `mcause u64`, `mepc u64`, `mtval u64`, `cycles_last u32`, `cycles_max u32`, `fw_ver u32`, `mhartid u32`, `vlenb u32` |
| 0x0801_0080 | L2–L34 | host | `obs f32[516]` (2,064 B, padded to 2,112 B) |
| 0x0801_08C0 | L35 | X280 | `action f32[15]`, `seq_echo u32` |
| 0x0801_0900 | L36 | X280 | ident: `misa u64`, `mtime_hz u32`, `build_id u32`, `kernel_variant u32`, `n_runs u32`, `n_overrun u32`, `n_reentry u32`, `beu_accrued u32` |
| 0x0801_0940 | L37 | host | LOAD descriptor (`dst u64`, `len u32`, `crc32 u32`), weight-blob header offsets |
| 0x0801_0980 | L38 | **KMD cleanup only** | `host_gone u32` = 0x474F_4E45 ('GONE'), written by `SET_NOC_CLEANUP` when the keeper fd closes; cleared by x280d at attach; outside every CRC |
| 0x0801_09C0 | L39 | X280 | E-test scratch: BEU snapshot, microbench results |
| 0x0801_0A00 | L40–L41 | x280d (from server_g1 via PUSH `:5572`) | M4c chunk descriptor + `chunk_seq` (§1.4.6) |
| 0x0801_0A80 | L42–L63 | – | reserved for M4a LbStep frames |
| 0x0801_1000 | – | host | weights: Balance + Walk, fp16 weights (937,984 B each) and fp32 biases (6,344 B each) = 1,888,656 B plus header, ending ≈ 0x081D_E1D0 + header; T-3.2 checks the real blob ends ≤ 0x081E_0000 |

Per-tick protocol:
1. The host writes L2–L34 as full WC lines, then L0's body (bytes 0–59) as one WC line, then `sfence`.
2. The host writes `seq_req = n` as a separate 4 B UC write.
3. The X280 spins on `seq_req`. When it changes, it executes `fence r,rw`, checks `crc32` and `obs_crc32` (a mismatch
   is retried for 50 µs, then reported as state 3 with `mcause = 0xC0C`), runs the MLP, writes L35, executes
   `fence w,w`, then writes L1 with `seq_done = n`.
4. The host polls L1 with UC reads (first poll after 100 µs, deadline 4 ms; 12 ms under `m3_rt_slow`), then reads L35
   and checks `seq_echo`.

Host overhead is measured in S1c on an initialised LIM region with no hart running (`x280d.py bench --host-only`):
33 WC line writes + one UC write + UC polls. The earlier 5–25 µs figure is withdrawn until then.

#### 1.4.5 Telemetry PUB `tcp://127.0.0.1:5570` and control PULL `:5571`

Multipart `[topic, msgpack]`. The PUB drops messages when the HWM is reached, so the HUD can never stall the loop.

| Topic | Rate | Fields |
|---|---|---|
| `tick` | 50 Hz | k, ep, t_ns, period_us, compute_us, late_us, slack_us, overrun, slip, sim_us, viewer_us, wbc_rtt_us, wbc_dev_us, wbc_be, wbc_pol, wbc_fb, shadow_da, seq, row, rows_left, starved, inflight, splice, jump_rad, nav[3], pelvis_z, success |
| `req` | per request | seq, gen, ep, k_obs, k_arr, d, d_hat, s_eff, rtt_ms, ser_ms, srv{infer,encode,device,decode}_ms, rows_used[2], outcome |
| `ego` | ≈ 3 Hz | seq, k_obs, uint8[120,160,3] |
| `meta` | 1 Hz | processor map {sim, vla, wbc} taken from `info` stamps, config hash, rt_factor |

Control PULL commands are applied at a tick boundary: `{"cmd": "swap_vla" | "swap_wbc", "to": <name>}` (a VLA swap
bumps `gen`), `{"cmd": "reset" | "pause"}`.

#### 1.4.6 M4 contracts

Frozen now, implemented only in M4. Source: plan A §1.4.

**`LbInput`** (M4a; 296 B): `x f32[72]`, `policy_override u32` (0xFF = auto), plus the header
`{tick i64, seq u32, flags u32 (b0 RESET_HISTORY)}`. The 72 floats are:

| Slice | Content |
|---|---|
| [0:29] | q_body, rad |
| [29:58] | dq_body, rad/s |
| [58:62] | pelvis quat wxyz |
| [62:65] | pelvis ω, rad/s (body frame) |
| [65:68] | nav (vx, vy m/s; wz rad/s) |
| [68] | base height, m |
| [69:72] | waist targets, rad (the X280 does the 3-joint FK) |

**`LbOutput`:** `q15 f32[15]` (rad, absolute = a·0.25 + default), `a15 f32[15]`, `policy u32`, `compute_cycles u32`,
`status u32` (0 ok, 1 tick_gap, 2 not_reset, 3 trap).

**`LbStep` semantics** (normative; reference `g1_gear_wbc_policy.py:102-275` plus
`g1_decoupled_whole_body_policy.py:111-147`; ✔r in review). At tick k:
1. If RESET_HISTORY: zero the 6×86 history, set `a_prev = 0`, and set (cmd, height, rpy)_prev = ([0,0,0], 0.74,
   [0,0,0]).
2. Push the frame built from (q, dq, quat, ω of tick k; (cmd, height, rpy)_prev; a_prev). This reproduces the
   **1-tick command lag**.
3. Set (cmd, height, rpy)_prev to the values in `LbInput`.
4. Select Balance if ‖nav‖ < 0.05, else Walk, using the *new* command.
5. Run `a = MLP(history)`, set `a_prev = a`, and output `q15 = a·0.25 + default`.

A backend is called exactly once per tick: the gait dt is hard-coded to 0.02 s.

**`LowState` / `LowCmd`** (M4b; shared memory, latest wins):
- `LowState`: q f64[43], dq f64[43], base_pose f64[7], base_vel f64[6], torso_imu f64[7], t_sim f64, plus header
  {tick, gen, t_mono_ns, flags}.
- `LowCmd`: q_des f64[43], policy u32, chunk_seq u32, cursor_row i32, flags (HOLD, STARVED, WBC_FAILOVER).

**Chunk descriptor** (M4c; mailbox L40–L41): `addr u64`, `layout u32` (0 = ROW_MAJOR f32 128 B pages, 1 = TILE bf16),
`page_size u32`, `n_pages u32`, `bank[8] {x u8, y u8, pad u16, offset u32}` from `ttnn.cluster.get_dram_bank_table`,
`chunk_seq u32`, `k_obs i64`. **Writer:** only server_g1 can obtain the bank table and tensor address (the call needs a
device opened in-process, ✔r `ttnn/cpp/ttnn-nanobind/cluster.cpp:191-230`) and knows when the last trace finished. It
sends `{addr, bank_table, chunk_seq, k_obs}` over PUSH `:5572` after the last trace completes; x280d, the only host
writer of LIM, writes L40–L41 (body first, `chunk_seq` last).

### 1.5 Stage mapping (exactly one row changes per milestone)

| Component | Contract | M1 | M2 | M3 | M4a | M4b | M4c |
|---|---|---|---|---|---|---|---|
| sim + tick loop | in-process | `rt_sim` | = | = | = | robot process only | = |
| splicer | in-process | host | = | = | = | **X280** | X280 |
| VLA | W-VLA | GPU `:5565` | **TT `:5557`** | = | = | = | **+ chunk in device DRAM** |
| WBC MLP | W-WBC | GPU `:5556` | = | **x280d `:5558`** | **LbStep on X280** | X280 | X280 |
| obs history / select / FK | – | host Python | = | = | X280 | X280 | X280 |
| fallback | in-process | ORT | ORT | ORT | ORT (LbStep port) | X280 hold | X280 hold |
| `rt_sim` CLI change | – | – | `--vla` | `--wbc` | `--wbc lbc:…` | `--placement x280` | server flag |

### 1.6 Clocks and run modes

- **RT:** ticks run on absolute `CLOCK_MONOTONIC` deadlines. d is measured from the arrival tick; d ≥ 1 by
  construction (replies are polled at tick start, requests leave ≈ 0.3 ms after T_k), so 65.5 ms on the 5090 gives
  d = ceil(65.5/20) = 4 and ≈ 45 ms on TT gives d = 3. Sim time is `k·0.02` and is never passed to `get_action(time=…)`.
- **VL:** no pacing. "In flight" means "not yet visible". A chunk becomes visible exactly at `k_obs + d_inj` (step
  3b), and the loop blocks there if its reply has not arrived. Deterministic given a deterministic VLA (mock or
  replay). Supports several processes in parallel against one VLA server, always with `--wbc ort` (each VL process
  runs ≈ 200 ticks/s and would overload one GPU WBC REP). Sweep d_inj ∈ {0, 3, 4, 5, 8, 12}: **4 = the 5090
  condition, 3 = the TT condition**.
- **sync** (arm C1): a separate, minimal frozen-time reference loop that does not use the splicer. It calls the VLA,
  applies rows 0–19 and repeats.

---

## 2. Milestones, work packages, acceptance tests

`source ~/experiments/gr00t/rtloop/env.sh` defines `R W RT DEV DEVX`, the wrappers `simpy srvpy ttpy xpy`, the URLs
`VLA_GPU=tcp://127.0.0.1:5565 VLA_TT=tcp://127.0.0.1:5557 WBC_GPU=tcp://127.0.0.1:5556 WBC_X280=tcp://127.0.0.1:5558`
and `ONNX=$W/gr00t_wbc/sim2mujoco/resources/robots/g1/policy`.

Thresholds live in `RT/gates.yaml`. `simpy $RT/scripts/analyze_run.py <run_dir> --gates <name>` writes
`<run_dir>/verdict.json` and exits non-zero on FAIL. Runs go to `$RT/runs/<stamp>_<tag>/`; `$RT/runs/latest` links to
the newest. Device jobs run as `$DEV $RT/scripts/chip_session.sh [--x280 none|probe|boot:3|attach] [--vla none|tt]
[--debug-endpoints] [--allow-reset] --job <script>`, and **as `$DEVX …` whenever `--x280 ≠ none`**. Job scripts live
in `RT/jobs/`. A job exits 3 to report "X280 state dirty"; chip_session then tears down and exits 3, and DEVX resets.
`--x280 probe` runs x280d with the keeper fd and windows but **never releases a hart** (S1c). Jobs that must start or
stop server_g1 mid-job (S1c, E3) do it through `RT/scripts/start_vla.sh`, which runs it as a child of chip_session,
never through `DEV`.

### M0: risk spikes (week 1, overlapping M1)

Each spike is the first slice of a later WP, not throw-away work, and ends in a go/no-go JSON.

| Spike | Built from | Question | Gate → decision |
|---|---|---|---|
| SP-1 async-drop pilot | WP-1.1–1.5, VL mode, GPU VLA, `--wbc ort` | How much success does latency cost? | 50 paired seeds (1000–1049); cells C1, C2@4, C3@{8,12}, C4@{4,8,12}; §3.4 rule → **DP-S** |
| SP-1b CRN calibration (optional) | SP-1 code | How much discordance does CRN remove? | C1 vs C1, 50 seeds, `--crn` on vs off; report discordance |
| SP-2 jitter soak | WP-1.4, 1.5 (RT) | Does 50 Hz hold with GPU VLA, EGL, remote WBC and the viewer? | 10 min with the T-1.4 timing rows |
| SP-3 TT G1 fidelity | WP-2.1–2.4, session S1 | Does TT reproduce the G1 fine-tune? | T-2.2 → DP1/DP6 |
| SP-4 X280 without release | WP-3.3, 3.4, session S1c | Registers, LIM ECC init, coexistence | T-3.E0, T-3.E1a, T-3.E1b |
| SP-5 X280 release | WP-3.6 (S2; needs DP3) | Once-only release, monitor, RNMI re-arm, LIM speed, power policy | T-3.E2, T-3.E2m, T-3.E3 → DP-X |
| SP-6 kernel parity | WP-3.1 (CPU) | Is fp16/fp32 accurate enough? | T-3.1 |
| SP-7 checkpoint screen | stock `rollout_policy.py` (C0) | nvidia vs cloudwalk, sync success | 50 eps each → DP1 |

SP-7 command, once per checkpoint (port 5565 = nvidia on `gpu_vla_server`, 5555 = cloudwalk on the existing server;
this is one of the two permitted uses of :5555, D15):

```
simpy gr00t/eval/rollout_policy.py --policy_client_host 127.0.0.1 --policy_client_port 5565 --n_episodes 50 \
    --max_episode_steps 1440 --n_action_steps 20 --n_envs 1 --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc
```

### M1: async loop on the RTX 5090 (≈ 102 h)

| WP | What | Files (all new) | Deps | h |
|---|---|---|---|---|
| WP-1.0 | Scaffold `RT` (`git init`) with `env.sh` (the four wrappers, `DEVX`, ports), `gates.yaml`, `preflight.py` and `.venv-x280`. Preflight: ports 5556–5559 and 5565–5572 free; `:5555` answers `ping` (read-only); VRAM free ≥ 4 GB; `DISPLAY` set; `GR00T_WBC_SERVER` unset; in a simpy subprocess, after importing the full sim stack, `os.environ["CUDA_VISIBLE_DEVICES"] == ""`. **Rescue `/tmp`:** copy `S/plan_research/{simloop,serving,x280/tc,parity_*}`, `S/review_stats*.py` and `S/live_viewer.py` into `RT/prototypes/`; copy `X/*` into `RT/third_party_refs/` with their SHAs. | `RT/{env.sh,gates.yaml,README.md}`, `RT/scripts/preflight.py`, `RT/prototypes/**`, `RT/third_party_refs/**` | – | 5 |
| WP-1.1 | `schema.py`: dims, joint-group indices (asserted against `robot_model`), A35 order, the ChunkRecord dtype, the RequestLog array dtype (`state43_sent f32[43]`, `obs_sha1 u8[20]`), wire dataclasses, `MsgSerializer` reuse, and the M4 contracts as dtypes. | `RT/rtloop/schema.py`, `RT/tests/test_schema.py` | 1.0 | 4 |
| WP-1.2 | `splicer.py`: pure functions `trigger`, `on_arrival`, `visible` (VL), `select` over a plain-array state record (portable to C). Arms v1, sync_hold (C2, C2b), async_noskip and sync_aligned. Blend from `last_row`; `starve()`/`hold()` update `last_row`; STARVE_FAULT flag. | `RT/rtloop/splicer.py`, `RT/tests/test_splicer.py` | 1.1 | 8 |
| WP-1.3 | `vla_client.py`: its own `zmq.Context`, REQ, `send(NOBLOCK)`, `poll(0)`, give-up after 500 ms with lazy-pirate re-create, episode-end drain (§1.2), seq/gen, `options.rt` (+ `noise_seed` with `--crn`), and the 9-key builder from the modality config. `mock_vla_server.py` modes: `hold`; `detfn` (deterministic function of the obs: arm sinusoids plus a nav schedule that exercises Walk); **`ramp`** (arm rows = request `state.left_arm/right_arm + i·δ` in f32, nav row i = f(i)); `record-proxy <upstream>`; `replay <dir>` (errors if a request's state keys differ from the recording by any bit); `--latency-ms/--jitter-ms`; `--stall-after N --stall-s T`. | `RT/rtloop/vla_client.py`, `RT/rtserve/mock_vla_server.py`, `RT/tests/test_vla_client.py` | 1.1 | 11 |
| WP-1.4 | `wbc_backends.py`: `OrtCpu` (CPU provider; `--ort-opts single` = `intra_op_num_threads=1` by default, `default` = default SessionOptions as in the official path); `ZmqWbc` (RCVTIMEO 5 ms, `--wbc-timeout-ms` override; one timeout → ORT for that tick; 3 in a row → ORT for the rest of the episode unless `--wbc-no-latch`; lazy pirate); `Shadow`; `install()`, the D5 class patch. `wbc_gpu_server.py` (srvpy): REP `:5556` in the main thread, wraps the **vendored** `GearWbcPolicies`, endpoints `ping`/`info`/`gear_wbc_action(policy, obs, tick=None, seq=None)` returning `compute_us`, boot self-check against ORT max\|err\| < 1e-4. `gpu_vla_server.py` (srvpy): builds `Gr00tPolicy` → `Gr00tSimPolicyWrapper` → `PolicyServer` exactly as `run_gr00t_server.py:69-103` does, binds `127.0.0.1`, applies `noise_seed` (D17) and fills INFO (§1.4.1); no WBC endpoint. | `RT/rtloop/wbc_backends.py`, `RT/rtserve/{wbc_gpu_server.py,gpu_vla_server.py,vendor/gear_wbc_policy.py,vendor/SHA256SUMS}`, `RT/tests/test_wbc_backend.py` | 1.1 | 11 |
| WP-1.5 | `sim_env.py`: the env as `rollout_policy.get_groot_locomanip_env_fn` builds it (✔ `rollout_policy.py:98-117`), with `onscreen=True` only under `--viewer` (deviation f), plus `WholeBodyControlWrapper`. Viewer via `patch_viewer_camera`, pre-launched in the reset window (§1.2). `env.reset(seed=s)`. `loop.py` modes `rt`, `vl`, `sync`; flags `--arm --s --B --d-inj --vla --wbc --wbc-timeout-ms --wbc-no-latch --ort-opts --crn --seeds --viewer --pub --dump-obs516 --dump-vla-obs N`. Thread affinity (§1.1). Pre-roll, drain, episode control, the `SystemExit` guard around `env.step` (✔ `sync_env.py:516-528`), env assertion `CUDA_VISIBLE_DEVICES == ""` after imports and after each reset, `gc.disable()` within episodes. | `RT/rtloop/{sim_env.py,viewer.py,loop.py,clock.py,affinity.py}` | 1.2–1.4 | 15 |
| WP-1.6 | `telemetry.py`: TickLog ring **backed by `np.memmap`** (`ticks.npy`, survives a crash), `requests.jsonl` + `requests.npz` (arrays), `episodes.jsonl`, the PUB, opt-in `obs516.npz` / `vla_obs/`. `config.json` includes `git status --porcelain` of RT/R/W and the sha256 of every imported dirty or untracked file. `analyze_run.py`: gates, Gantt chart, histograms (period, lateness, d, chunk age, WBC RTT), boundary metrics. | `RT/rtloop/telemetry.py`, `RT/scripts/analyze_run.py` | 1.5 | 11 |
| WP-1.7 | `hud.py` (§4.3), `control.py` (PULL commands), `demo.sh m1\|m2\|m3`, `record_demo.sh` (ffmpeg x11grab, out of process, pinned to 6–7). Off the critical path. | `RT/scripts/{hud.py,demo.sh,record_demo.sh}`, `RT/rtloop/control.py` | 1.6 | 8 |
| WP-1.8 | Equivalence, alignment, fault and guard tests: `test_vl_equivalence.py`, **`test_alignment.py`** (T-1.2b), `test_official_equivalence.py` (each arm in its own subprocess; the official env + `MultiStepWrapper(n_action_steps=20, terminate_on_success=True)` built by importing `rollout_policy`, a recorder wrapper between the layers, seeded reset, replayed chunks, identical ORT SessionOptions), `test_faults.py`, `test_cuda_guard.py`, `test_obs_schema.py`. | `RT/tests/*.py` | 1.5, 1.6 | 12 |
| WP-1.9 | `eval_paired.py`: matrix YAML → conditions × seeds, n procs for VL, seed blocks of 10 interleaved across arms, Wilson CI, paired Wald CI, exact McNemar, the §4.4 verdict classes, paired restricted-mean time-to-success with seed bootstrap and seed-stratified log-rank. Configs for screen, primary and secondary matrices. | `RT/scripts/eval_paired.py`, `RT/configs/eval_m1_{screen,primary,secondary}.yaml` | 1.6 | 9 |
| WP-1.10 | M1 campaign and report. | `P/results/M1.md` | 1.8, 1.9 | 8 + machine time |

**M1 acceptance tests.** Before any demo: T-1.1–T-1.7 pass. "Demo-ready": T-1.8 also passes. Formal exit: T-1.9 and
T-1.10, which may overlap M2.

| ID | Command | Pass threshold |
|---|---|---|
| T-1.1 units | `simpy -m pytest $RT/tests/test_schema.py $RT/tests/test_splicer.py $RT/tests/test_vla_client.py $RT/tests/test_wbc_backend.py -q` | 100% pass. The splicer table covers d ∈ [0,28], s ∈ {10,16,20}, B ∈ {0,3,5}, late-reject at d ≥ 26, gen drop, starvation hold plus a nav ramp to 0 over 5 ticks, STARVE_FAULT at 50 starved ticks, **VL: the row executed at k_obs + d_inj equals d_inj for d_inj ∈ {0,1,3}**, **starve then arrival with B = 5: nav during the blend equals (1−w)·0 + w·new**, s_eff for d̂ ∈ [0,27] (never < 1), and d̂ = p90 of the last 10. The REQ client gives up within 500 + 50 ms when the server is dead and drains or re-creates at episode end. ORT fallback happens on the same tick (≤ 5.5 ms). Request payload ≤ 1.0 MB. ORT `single` vs `default` outputs compared on 10k logged obs (reported). |
| T-1.2 VL ≡ own sync | `simpy -m pytest $RT/tests/test_vl_equivalence.py -q` (mock `detfn` on :5559, `--wbc ort`, seeds 0–4, 1440 ticks; `--mode vl --arm v1 --s 20 --d-inj 0 --B 0` vs `--mode sync`) | max\|Δq\| = 0.0 at every tick; identical success ticks |
| **T-1.2b alignment** | `simpy -m pytest $RT/tests/test_alignment.py -q` (mock `ramp` on :5559, `--wbc ort`; VL at d_inj ∈ {1,3,8} and RT with mock latency 60 ± 5 ms; 2 episodes each) | From `ticks.npy` and `requests.npz`, bit-exactly in f32: `cmd_arm(k) == state43_sent(k_obs)[arms] + (k − k_obs)·δ` on every tick; `state43_sent(k_obs) == q43_meas` at the end of step k_obs − 1; `k − k_obs` at each switch tick equals the logged d. This is the alignment detector. |
| T-1.3 own sync ≡ official | `simpy -m pytest $RT/tests/test_official_equivalence.py -q --vla $VLA_GPU --seeds 0-4` (records chunks in `--mode sync` through `record-proxy`, then replays them through the official stack; each arm in its own subprocess; `--ort-opts default`) | max\|Δq\| = 0.0 up to the official end tick; 0 replay state mismatches. Shows the loop equals the official convention. If Δq ≠ 0, first rerun with byte-identical SessionOptions and env, then root-cause. |
| T-1.4 RT timing (official M1 config) | `srvpy -m rtserve.gpu_vla_server --port 5565 --model <DP1 checkpoint, default cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate> &`; `srvpy -m rtserve.wbc_gpu_server --port 5556 --onnx-dir $ONNX &`; `$RT/scripts/demo.sh m1 --seeds 1000-1019 --hud --record --tag t14`; `simpy $RT/scripts/analyze_run.py $RT/runs/latest --gates m1_rt` | Measured outside reset windows, tick 0 included. Tick period p99 ≤ 21.0 ms, max ≤ 30 ms. Overruns (compute > 20 ms) ≤ 0.5%. Slips = 0. `viewer_us` at k = 0 ≤ 3 ms. WBC RTT p50 ≤ 1.0 ms, p99 ≤ 2.0 ms; WBC calls with RTT > 20 ms < 0.1%. WBC fallbacks ≤ 0.1% of ticks. Shadow max\|Δa\| p99 ≤ 1e-4. d p50 ∈ {3,4}, p99 ≤ 5. In-flight requests ≤ 1 always. Starved ticks = 0. SAFETY_FAULT = 0. VLA p50 ≤ 1.1 × its idle paced value (65 ms). `viewer.sync` p99 ≤ 3 ms. These gates are pre-registered but unmeasured under full load; on failure apply the R9/R10 ladder and rerun. |
| T-1.5 negative control (optional; uses :5555 for both) | `$RT/scripts/demo.sh m1 --vla tcp://127.0.0.1:5555 --wbc tcp://127.0.0.1:5555 --wbc-timeout-ms 200 --wbc-no-latch --seeds 1000-1001 --tag t15` (timing only, cloudwalk); `analyze_run.py --gates m1_negctl` | **Expected to fail the m1_rt gates:** ≥ 3% of WBC calls with RTT > 20 ms or timeouts (U:srv E2c: 5.1% at a 0.4 s cadence). Together with T-1.4's < 0.1% this shows the instrument sees head-of-line blocking. |
| T-1.6 faults | `simpy -m pytest $RT/tests/test_faults.py -q` (mock on :5559) | **VLA stall 2 s:** `starved` set within 1 tick of row 30; ‖nav‖ < 0.05 within 5 ticks (Balance); STARVE_FAULT flag at 50 ticks; the episode continues; recovery on the next arrival; process alive; 0 slips. **WBC server killed:** ORT on the same tick; reconnect after restart. **VLA server killed:** give-up at 500 ms → lazy pirate → starvation path, no hang. **Arm jump injected:** SystemExit caught → SAFETY_FAULT → reset. |
| T-1.7 CUDA/EGL guard | `simpy -m pytest $RT/tests/test_cuda_guard.py -q` | `CUDA_VISIBLE_DEVICES == ""` after all imports, after 3 resets and at exit; at exit, rt_sim's pid is absent from `nvidia-smi --query-compute-apps=pid` (EGL contexts are graphics-type and do not appear there). The ego frame is not all-black or NaN, and it is **bit-identical** (same seed and tick) across: CUDA hidden vs visible-but-unused; viewer on vs off. |
| T-1.8 screen | `simpy $RT/scripts/eval_paired.py --matrix $RT/configs/eval_m1_screen.yaml --out $RT/runs/m1_screen` (§4.4) | Triage only (MDE ≈ 22–25 pts): C4 at VL d_inj = 4: lower 80% bound of (C4 − C1) ≥ −25 pts; falls(C4) ≤ falls(C1) + 2. All cells finish with 0 loop errors. |
| T-1.9 comparison | `… --matrix $RT/configs/eval_m1_primary.yaml` (RT C1 (`--mode sync`) vs RT C4, **n = 50 paired**, seeds 2000–2049, `--crn`, GPU WBC) | Gross-drop gate only: lower 80% bound of (C4 − C1) ≥ −25 pts. Reported: Δ̂ with a two-sided 95% paired CI, exact McNemar p, per-arm Wilson CIs. 0 STARVE_FAULT episodes, 0 SAFETY_FAULT, 0 loop-attributable falls. Secondary (VL d-sweep, C3 vs C4) reported at n = 50 when run. |
| T-1.10 liveness | `simpy -m rtloop.loop --mode rt --arm v1 --vla $VLA_GPU --wbc $WBC_GPU --duration 2h --seeds 3000-` | 0 hangs. Every episode ends SUCCESS, TRUNC, FALL or SAFETY_FAULT. 0 orphan requests; `drain_recreate` counted and ≤ 1% of episodes. `rt_sim` RSS growth < 50 MB. |

### M2: VLA on Tensix (only `--vla` changes; ≈ 55 h + 10 h conditional)

| WP | What | Files | Deps | h |
|---|---|---|---|---|
| WP-2.1 | `g1_n16.py`, with `install()` ported from `TT/libero/n16.py:176-229`: re-point `HF_REPOS["n16"]` and `HF_SNAPSHOT_SHAS["n16"]` at the DP1 checkpoint; add layout `g1 = dataclasses.replace(gr1, name="g1", embodiment_tag="unitree_g1", embodiment_id=8, valid_slice=(30,35))` (proven on CPU, U:tt "UNITREE_G1 support: layout gap"); **no `_extend_layer_tables`** (MK already has `TL.MATMUL_CONFIGS`, `SDPA_CHUNKS`, `FORBIDDEN_SDPA_CHUNKS` and G1 reuses gr1's S_pad 128 geometry, ✔r review); clear the normalization `lru_cache`s; assert the spec (slot 8, `ego_view`, state 43, action 35, horizon 30, arms RELATIVE). Dedicated `cache_root=~/.cache/gr00t-tt-g1/<repo>@<sha8>/` (U:tt "Weight-cache key collision"). Backend stamp built from fields that exist on MK's TTPolicy (it has no `backbone_backend`). | `RT/rttt/g1_n16.py`, `RT/tests/tt/test_g1_host_cpu.py` | – | 6 |
| WP-2.1b | *Needed when DP1 = cloudwalk (the default, D11):* CPU F32→BF16 conversion with `HF_HUB_CACHE=~/.cache/gr00t-tt-g1/hub` for ttpy only, written as a distinct repo (`…PnPAppleToPlate-bf16`) with a synthetic sha; **never** written into `~/.cache/huggingface`. Keep the 2-shard map; assert total_size = 6,573,217,664 and every dtype BF16; `GR00T_TT_FORCE_CONVERT=1` on the first build. Preflight: ≥ 40 GB free (disk is 92% used, 157 GB free, ✔). | `RT/rttt/convert_bf16.py` | – | 4 |
| WP-2.2 | G1 goldens on the 5090. srvpy in a separate process, a variant of `R/tt_golden/dump_golden_n16.py`, reading 20 raw obs (10 reset + 10 mid-episode, from `--dump-vla-obs`). Per sample: fp32 and bf16 runs with saved noise and `action_pred_norm`. **Stop `:5565` first; require ≥ 20 GB free VRAM.** Plus `make_isaac_ref.py` (srvpy): on the same 20 obs, writes `input_ids`, `pixel_values`, normalised state and decoded actions from fixed `pred` arrays to `isaac_ref.npz` (python_env cannot import the Isaac processor: no albumentations, ✔r). | `RT/rttt/{make_g1_golden.py,make_isaac_ref.py}` → `~/experiments/gr00t/golden/n16_g1/` | 1.5 (dumps) | 9 |
| WP-2.3 | `server_g1.py`, ported from `TT/libero/server_n16.py`: wire layer, poller loop, bind `127.0.0.1` before device open, modality-key filter before `split_batch`, `info`, per-call JSONL, fresh noise from `torch.Generator(--seed)` (reseeded from `noise_seed` when present), boot **sanity** check: PCC ≥ 0.995 vs the **bf16** G1 golden on rows [:30] × dims [:35], every failure returned as `{"error": …}`, `info.tree` stamp. `--debug-endpoints`: `predict_debug` (§1.4.1) and `x280_peek` (`ttnn.cluster.read_reg`, E0 cross-check only). Protocol test against a fake model on CPU, and an **import-compatibility test against MK**. | `RT/rttt/server_g1.py`, `RT/tests/tt/test_tt_server_protocol.py` | 2.1, 2.2 | 12 |
| WP-2.4 | `chip_session.sh`, run under `DEV`/`DEVX`. Unsets `TT_METAL_WATCHER` and `TT_METAL_TRACE_ALLOC_TRACKING`. Order: (1) if `--x280 ≠ none`: `xpy x280d.py probe --json` (opens, reads L2CPU_RESET, WAYENABLE, CCACHE0_CONFIG, PLL4, heartbeat; closes; exits; 10 s timeout) = the **L2CPU health check** the guard's Tensix-only probe cannot do (✔ `_device_guard.sh:33-44`); (2) if the probe reports a dirty tile and `--allow-reset`: `fuser /dev/tenstorrent/0` must be empty, then `tt-smi -r 0`, wait 5 s, re-run the guard's probe snippet (`RT/scripts/tt_probe.py`) and step 1; without `--allow-reset`: exit 3; (3) start `x280d serve` (keeper fd); (4) start server_g1 (`--vla tt`), wait READY; (5) run `--job`; (6) teardown in reverse, children killed with SIGKILL after 10 s so the guard's post-job reset sees no users. Restarts happen only as children; `DEV` is **never** called inside. Also `bench_g1.py` (latency, ABAB, determinism via `predict_debug`) and the job scripts. | `RT/scripts/{chip_session.sh,tt_probe.py}`, `RT/rttt/bench_g1.py`, `RT/jobs/t2_*.sh` | 2.3 | 7 |
| WP-2.5 | Session **S1** (TT only, **4–6 h** device time): cold G1 build under the new cache root (the old LIBERO cache dir is empty, so fully cold), boot check, T-2.2…T-2.4. | – | 2.4 | 6 (device) |
| WP-2.6 | Closed loop in **S1b**: T-2.5 sync on TT; T-2.6 RT soak; T-2.7 primary (TT arm, ≈ 3.1 h); T-2.8 soak. | `P/results/M2.md` | 2.5, M1 demo-ready | 12 |
| WP-2.7 | *Conditional on T-2.2 failing:* `mk_arena_dtype=bf16` (denoise ≈ 4.71 vs 3.83 ms/step), re-run S1. If it still fails: DP1 (other checkpoint) or DP6 (live tree). | – | 2.5 | 6 |
| WP-2.8 | Demo extra: GPU↔TT VLA hot-swap between episodes (HUD key `v`, gen bump). | `RT/rtloop/control.py` | 2.6 | 3 |

**M2 acceptance tests.** T-2.1…T-2.4 gate any closed-loop run; T-2.5…T-2.7 gate the demo.

| ID | Command | Pass threshold |
|---|---|---|
| T-2.1 CPU | `ttpy -m pytest $RT/tests/tt/test_g1_host_cpu.py $RT/tests/tt/test_tt_server_protocol.py -q` | Spec asserts pass. `assert_call` OK for layout g1. 503 kept / 534 skipped megakernel weights. Host encode and decode equal `isaac_ref.npz` on the 20 recorded obs (max\|d\| = 0). `server_g1` imports and serves the fake model on MK; key filter drops the 13 extra stock keys; default bind is 127.0.0.1. |
| T-2.2 fidelity | `$DEV $RT/scripts/chip_session.sh --vla tt --debug-endpoints --job $RT/jobs/t2_fidelity.sh` (boot check, then `simpy -m pytest $RT/tests/tt/test_g1_fidelity.py --endpoint $VLA_TT --golden ~/experiments/gr00t/golden/n16_g1`, through `predict_debug` with the golden noise) | **Primary, vs the bf16 golden:** `action_pred_valid` PCC mean ≥ 0.999, min ≥ 0.998 over 20 samples. **Secondary, vs fp32:** per sample, PCC(TT, fp32) ≥ PCC(GPU-bf16, fp32) − 0.001. **Per group, absolute, vs the bf16 golden:** decoded max\|d\| ≤ max(2 × the largest GPU-bf16-vs-fp32 max\|d\| of that group over all 20 samples, 0.02 rad, or 0.02 m/s / rad/s for nav); left_hand and base_height excluded. 5 replays with fixed noise are bit-identical. *S1 2026-10-25: FAIL on navigate_command only (0.0703 vs 0.0318, s19 yaw ramp lag; device numerics, WP-2.7 no change); coordinator proceeds to S1b with T-2.5 as the arbiter, pending the user.* |
| T-2.3 latency | `… --job $RT/jobs/t2_bench.sh` (`bench_g1.py --n 500 --period-ms 320` through ZMQ; `:5565` stopped) | Client RTT p50 ≤ 48 ms, p99 ≤ 55 ms; server `total_ms` p50 ≤ 45 ms; d p99 ≤ 3. These are projections (2.63 + 39.4 + 0.15 ≈ 42.2 ms + ZMQ, never measured on G1): on failure with d p99 ≤ 4, re-baseline the latency rows in `gates.yaml` with a recorded rationale (s = 16 still does not starve). Every call over 100 ms is attributed to encode/device/decode (LIBERO: 2 of 1327 calls > 100 ms, max 380 ms). Two calls on the same obs without a seed give *different* chunks. |
| T-2.4 stale state | `… --debug-endpoints --job … bench_g1.py --abab 200` | Outputs for alternating A/B/A/B obs with fixed noise (`predict_debug`) equal the single-call outputs bit-exactly. |
| T-2.5 sync on TT | `simpy $RT/scripts/eval_paired.py --matrix $RT/configs/eval_m2_sync.yaml` (C1 on TT vs C1 on `:5565`, 50 paired seeds); anchor: `simpy gr00t/eval/rollout_policy.py --policy_client_host 127.0.0.1 --policy_client_port 5557 --n_envs 1 --n_episodes 50 --max_episode_steps 1440 --n_action_steps 20 --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc` | Lower 80% bound of (TT − GPU) ≥ −25 pts (triage). 0 server errors or hangs per arm (`rttt/m2_checks.py errors`). Stock anchor within the 95% CI of SP-7 for the same checkpoint. |
| T-2.6 async RT soak (30 min) | `CHIP_VLA_STATUS_PUB=$TT_STATUS $DEV $RT/scripts/chip_session.sh --vla tt --job $RT/jobs/t2_soak.sh --tag t26` (runs `demo.sh m2 --in-session --hud --record --no-alt` for 1800 s, then `analyze_run.py --gates m2_rt` on the session copy and `rttt/m2_checks.py soak`) | All `m2_rt` gates (§4.2): T-1.4 timing rows with d p99 ≤ 3 and VLA p50 ≤ 1.1 × the T-2.3 p50; starved ticks ≤ 0.1% and no starved run > 10 ticks. 0 TT error replies. HUD shows `vla=tensix` from `info`. |
| T-2.7 comparison | `eval_paired.py --matrix $RT/configs/eval_m2_primary.yaml` (RT C4 with TT vs RT C4 with GPU, **n = 50 paired**, seeds 2000–2049; session job `jobs/t2_primary.sh`; both arms run fresh against `gpu_vla_server --noise cpu`: T-1.9's GPU arm drew CUDA-seeded noise, so it is not `--crn` paired with TT and is never reused) | Gross-drop gate: lower 80% bound of (TT − GPU) ≥ −25 pts; Δ̂, CI and McNemar p reported. 0 server errors or hangs per arm (`rttt/m2_checks.py errors`). |
| T-2.8 soak | 2 h RT: `CHIP_VLA_STATUS_PUB=$TT_STATUS $DEV $RT/scripts/chip_session.sh --vla tt --job $RT/jobs/t2_soak_2h.sh --tag t28` (`t2_soak.sh` for 7200 s with the drift gated) | 0 error replies, 0 device hangs, VLA p50 drift < 5% between the first and last 10 minutes |

### M3: WBC MLP on the X280 (only `--wbc` changes; ≈ 107 h + 9 h conditional)

CPU-only work, starting on day 1:

| WP | What | Files | Deps | h |
|---|---|---|---|---|
| WP-3.1 | Freestanding C forward pass: estimator 516-256-256-35 ELU, L2-normalised latent, actor 121-512-256-256-15 ELU. Polynomial `expf`, own `memcpy`/`memset`, no libcalls, `-ffp-contract=off` for v0. Weight packer: ONNX initializers → blob (header, CRC32, fp32 or fp16 weights, fp32 biases, both policies; prints the end address). x86 build plus a ctypes parity test against ORT on 10k logged `obs516` and 256 random inputs. | `RT/x280/fw/{wbc_mlp.c,wbc_mlp.h,elu.c}`, `RT/x280/tools/pack_weights.py`, `RT/tests/x280/test_kernel_x86.py` | 1.6 (logs) | 16 |
| WP-3.2 | Firmware: `crt0.S` (harts ≠ 0 park in wfi; `csrw 0x7c1,0`; mstatus FS/VS on; on every entry increments `n_reentry` and re-arms RNMIs with `csrsi 0x353, 8`, i.e. mnstatus.NMIE); `lim.ld` (0x0800_0000, length 0x1E0000); `monitor.c` (§1.4.4 v2 protocol with CRC checks; trap handler dumps mcause/mepc/mtval to L1; SELFTEST reads every LIM line from the hart and reports the hart-0 BEU accrued register at 0x0170_0000 to L36); **GP-scratch park stub** ≤ 48 B at 0x2001_0100 (`fence.i`; spin while bit h of 0x2001_0414 is set; jump to crt0 at the fixed 0x0800_0000, `lim.ld:63`; it never loads the hart's reset vector, which E1F leaves pointing at 0x2001_0100 = this stub, where the hart would loop forever: review:x280 F1, ✔ `RT/x280/fw/park_stub.S`) and an RNMI-exception spin stub at +48; CSRs only by number (0x350–0x353), `mnret` only as `.word 0x70200073`; heartbeat; ident line. Payloads: **LIM microbench** (pointer chase, scalar stream, `vle32` at m1/m4/m8, a loop executing from LIM, optional Zero-Device working set), kernels v0 scalar fp32 (parity reference), v1 RVV fp32 (e32/m4 column-broadcast GEMV), v2 fp16 weights widened with **integer** ops (no Zvfh until a trap-guarded probe passes). Stage-0 initializer source (≤ 64 B, used only in E1F). clang-20 `--target=riscv64-unknown-elf -march=rv64gcv_zfh_zba_zbb -mabi=lp64d -mcmodel=medany -O2 -ffreestanding -fno-builtin -nostdlib`. x86 build of the monitor for unit tests. | `RT/x280/fw/{crt0.S,lim.ld,monitor.c,trap.c,park_stub.S,stage0.S,lim_bench.c,mailbox.h,Makefile}`, `RT/scripts/check_image.py`, `RT/tests/x280/test_mailbox_x86.py` | – | 19 |
| WP-3.2b | **RISC-V emulation before silicon.** Build spike from source in user space (no sudo; none is installed, ✔ `which spike qemu-riscv64` → none). Run the same image with `--isa=rv64gcv_zfh_zba_zbb --varch=vlen:512,elen:64 -m0x8000000:0x1e0000` and a build flag `SPIKE_HOST` that replaces the host by a scripted command list in LIM plus HTIF exit codes. Test SELFTEST, LOAD/FENCE_I/CALL, INFER bit-parity against the x86 build, the microbench payload, the stage-0 stub, and the illegal-instruction trap dump. The X280-only RNMI CSRs (0x350–0x353) and 0x7c1 are stubbed behind the flag. | `RT/x280/sim/{build_spike.sh,run_spike.py}`, `RT/tests/x280/test_spike.py` | 3.2 | 6 |
| WP-3.3 | Host access layer: `regs.py` (the only source of addresses, coordinates and expected values; a write allowlist), `ledger.py`, **`kmd.py`** (ctypes: `ALLOCATE_TLB`, `CONFIGURE_TLB`, `FREE_TLB`, `SET_NOC_CLEANUP` with the packed `tenstorrent_set_noc_cleanup` struct, argsz 32, on the keeper fd; UC and WC mmaps of each window, ✔ `X/tt-kmd-2.11.0/ioctl.h:271-360`), **`wcio.c`** → `libwcio.so` (64 B-aligned AVX-512 `_mm512_stream_si512` stores + `sfence`; 4 B UC accessors; logs every transaction's size and alignment). Windows, each configured once: LIM (L2CPU3, 0x0800_0000), ARC (8,0) 0x8000_0000 (PLL4, L2CPU_RESET, SCRATCH_RAM), CCACHE/L2PF 0x0200_0000, BEU 0x0160_0000, external peripherals 0xFFFF_F7FE_FFE0_0000. Unit tests against a fake device (memfd + recorded ioctls). | `RT/x280/{regs.py,ledger.py}`, `RT/x280/host/{kmd.py,wcio.c,Makefile}`, `RT/tests/x280/test_kmd_fake.py` | – | 12 |
| WP-3.4 | `x280d.py`. Keeper legacy fd `os.open('/dev/tenstorrent/0', O_RDWR)`, then waits ≥ 1 s and re-reads PLL4 before any L2CPU access (the open itself sends a power message, ✔ `chardev.c:954-1017`). `SET_NOC_CLEANUP` writes 'GONE' into L38 when the fd closes. **Interlocks**, read directly (telemetry is a 100 ms snapshot, ✔r fw `telemetry.c:278, 591-592`): WAYENABLE reads 0; `L2CPU_RESET[3:0] == 0xF`; the target bit is 0 before a release, or a live heartbeat for an attach; PLL4 `CNTL_1 (0x8002_0504) == 0x0040_0002`, `CNTL_5 (0x8002_0514) == 0x0101_0101`, `USE_POSTDIV (0x8002_051C) & 0xF == 0xF` before any L2CPU-tile access and after boot (✔ fw `pll.c:225-236`); ENABLED_L2CPU bit 3 set; no ARC messages ever. Subcommands `probe`, `lim-init --method A\|B`, `lim-check`, `boot`, `attach`, `ping`, `selftest`, `load`, `call`, `park`, `parity`, `bench [--host-only\|--rtt]`, `powertest`, `serve --port 5558 [--early]`, plus `--dry-run` (prints every NoC op and every libwcio transaction). Watchdog: heartbeat stall > 5 ms → error reply, the client falls back to ORT. `mailbox_sim.py` runs the x86 kernel behind the same struct in shared memory (`serve --backend sim`). | `RT/x280/{x280d.py,mailbox_sim.py,requirements.txt}`, `RT/tests/x280/test_x280d_sim.py` | 3.1–3.3 | 20 |

Device work, strictly in order under §5.2 (all sessions run under `DEVX`):

| WP | Session | What | h |
|---|---|---|---|
| WP-3.5 | S1c (≈ 2 h) | E0 read-only probe; E1a LIM init + checks with **no TT**; then server_g1 starts and E1b repeats the pattern on a second region with TT serving. No release. Exit: `jobs/s1c.sh` logs `S1c -> S2: GO` only if E0 with the keeper fd passed with `e0.pll4_walk_ready` true and E1a passed, else `S1c -> S2: NO-GO: <reason>` (`x280/tools/m3_jobs.py s1c-summary`). **2026-10-25:** S1c ran (`RT/runs/sessions/20261025_053543_2185222_s1c`): E0 passed; E1a and E1b failed on the CCACHE ECC alone (host NoC writes pre-read the SRAM; on never-written SRAM ≈ 94% of those pre-reads are uncorrectable and the CCACHE poisons the footprint). S1d (`…/20261025_071804_2506246_s1d`, `jobs/s1d.sh`): no host store (4/8/32/64 B, repeated) heals it, and poison and SRAM contents survive `tt-smi -r 0`. Host-written LIM is unusable: S1c → S2 NO-GO for a host-initialised LIM, so WP-3.E1F ran (R26 resolved by E1F). | 3 |
| WP-3.6 | S2 (≈ 3 h; DP3) | E2: release L2CPU3 at the start of the hold, before any TT build; monitor selftest; RNMI re-arm test. E2m: LIM microbench. E3: power survival, including server_g1 start/stop as a child. Optionally E3(d) last. | 6 |
| WP-3.7 | S3 (≈ 3 h) | E4 kernel bench (v0/v1/v2 × fp16/fp32; option E4c: stack + activations in a Zero-Device pinned L3 way via `CCACHE0_WAYMASK0`, no WAYENABLE write, ✔ `Caches.md:95-99`); mailbox RTT with the VLA idle vs busy; E5a on-device parity. Pick the kernel. | 8 |
| WP-3.8 | S4 (≈ 1 day) | Closed loop. (a) **Shadow run:** ORT primary, X280 shadow, 20 episodes. (b) X280 primary: timing soak, fault injection, evaluation, 2 h soak. | 12 |
| WP-3.8b | – | *Conditional on the chosen kernel's compute p99 > 1.5 ms:* **early dispatch.** A class-level patch of `set_observation` sends the tick-(k+1) obs516 with the predicted policy (`select(k+1)` peeked without mutation) right after physics of tick k; the tick-(k+1) `run_inference` uses the reply if the policy matches, else calls synchronously. Numerically identical (§1.4.3). | 6 |
| WP-3.9 | S5 | `demo.sh m3`: HUD key `w` hot-swaps X280↔GPU mid-episode (stateless, so safe). "Proof switch": RNMI-park the hart; the HUD must show the fallback within 1 tick, then the reload. | 5 |
| WP-3.E1F | S2′ (DP3) | *Conditional on E1a failing with methods A and B:* stage-0 hart initializer (§5.2). Spends the session's release. **2026-10-25: ran with the user's DP3(d) approval** (`RT/runs/sessions/20261025_092909_2787811_e1f`, `jobs/s1f_stage0.sh`): stage 0 zero-filled LIM from the hart, the monitor came up, SELFTEST passed on all 30,720 lines with BEU 0 and 0 traps, blob loaded, boot parity vs x86 0.0, v2-fp16 compute 0.385 ms at 800 MHz, x280d p50 0.66 / p99 0.69 ms, 500/500 ok. Only `bench --rtt`'s host-overhead p99 (301 µs vs the 0.3 ms gate, which stands: a finding) and its unpinned client's RTT p99 (2.8 ms; not a T-3.5 gate) missed. **R26 is resolved by E1F, and the stage-0 boot (`serve --stage0`, `X280_E1F_APPROVED=1`) is now the standard boot of every M3 session (S2–S5);** method A stays in the code, unused on silicon. | 3 |

**M3 acceptance tests.** T-3.1–T-3.4 (and T-3.2b before E2) and the dry-run review gate **any** device work. Each
E-step gates the next. T-3.5–T-3.9 gate the demo.

| ID | Command | Pass threshold |
|---|---|---|
| T-3.1 kernel (x86) | `simpy -m pytest $RT/tests/x280/test_kernel_x86.py -q` | Balance and Walk on 10k logged + 256 random inputs: fp32 max\|Δa\| ≤ 5e-5 (≤ 1.25e-5 rad after the 0.25 scale); fp16 ≤ 1e-2 (host measurement 6.8e-3 / 7.4e-3, U:x "Weight precision") |
| T-3.2 image | `make -C $RT/x280/fw x280 && simpy $RT/scripts/check_image.py $RT/x280/fw/build/monitor.elf` | Image ≤ 48 KiB; all sections inside LIM; undefined symbols = ∅; park stub ≤ 48 B and stage-0 stub ≤ 64 B; no access to CSRs 0x740–0x744; `mnret` present only as 0x70200073; the real packed weight blob ends ≤ 0x081E_0000 |
| T-3.2b spike | `simpy -m pytest $RT/tests/x280/test_spike.py -q` | SELFTEST, LOAD/FENCE_I/CALL 10/10; INFER output bit-identical to the x86 build on 1k obs (v0) and ≤ 1e-6 (v1/v2); trap dump correct for an illegal instruction; microbench and stage-0 stub run to completion. **Precondition of E2.** |
| T-3.3 software path, no HW | `$RT/scripts/demo.sh m3 --x280-sim --duration 120 --tag t33` (starts `x280d.py serve --backend sim --port 5558 --sim-cpus 21` on CPUs 5,21 unless :5558 answers; without `--x280-sim`, `demo.sh m3` opens a DEVX session and boots L2CPU3); `analyze_run.py --gates m3_sim` | All T-1.4 gates; WBC RTT p99 ≤ 2 ms |
| T-3.4 dry-run review | `xpy $RT/x280/x280d.py boot --tile 3 --dry-run > $RT/runs/dryrun_boot.txt && simpy -m pytest $RT/tests/x280/test_x280d_sim.py $RT/tests/x280/test_kmd_fake.py -q` | The op list equals the §5.2 boot sequence. Zero writes to `CCACHE0_WAYENABLE` or any GDDR address. **Exactly one write to 0x8003_0014, whose value equals the preceding read OR 0x80 with [3:0] == 0xF, followed by a read-back.** The libwcio transaction log shows every LIM-init transaction is a 64 B-aligned 64 B store followed by `sfence`, and no partial write precedes the init of its line. Every write target is in the `regs.py` allowlist. |
| T-3.E0 probe | `$DEVX $RT/scripts/chip_session.sh --x280 probe --vla none --debug-endpoints --job $RT/jobs/s1c.sh` (one S1c job: step 1 = E0 with the keeper fd only; step 2 = E1a; step 3 = `start_vla.sh tt` as a child, then E0 repeated with server_g1 serving; step 4 = E1b) | `L2CPU_RESET == 0x0F`; WAYENABLE = 0; PLL4 registers at the default values above; ENABLED_L2CPU bit 3 = 1; (8,7) and (8,29) read the same `CCACHE0_CONFIG`; the CCACHE ECC fix/fail counters are readable (offsets pinned in `regs.py` from SiFive's ccache0 layout; values recorded); `TIMER_HEARTBEAT` readable via the `SCRATCH_RAM[13]` pointer (✔ fw `status_reg.h:12-13`, luwen `blackhole.rs:155`) and increasing; `x280_peek` agrees with kmd.py on every register; TT outputs with fixed noise bit-identical before and after; TT p50 change ≤ 2%; max TT latency during each x280d fd open/close recorded. |
| T-3.E1a LIM, no TT | same job, step 2 (keeper fd only) | Method A over all 1,966,080 B; pattern read back 100%; 100 partial 4 B UC writes on initialised lines read back correctly; CCACHE ECC fail counters unchanged (Δ = 0); L2CPU_RESET and WAYENABLE unchanged. **SET_NOC_CLEANUP check:** register a cleanup to L38 on a separate test fd, close it, read 'GONE' back. |
| T-3.E1b LIM, TT serving | same job, steps 3–4 | Same checks on a second region; TT `predict_debug` outputs bit-identical before/after; TT p50 ±2%. If method A failed in E1a, method B is tried here; if both fail → E1F (DP3). |
| T-3.E2 release | `$DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla none --job $RT/jobs/s2_e2_e3.sh` | Heartbeat advances at ≥ 1 kHz. `vlenb` recorded (kernels are VLEN-agnostic; 64 expected). misa has V. mcycle/mtime = 16 ± 5% (800 MHz / 50 MHz). SELFTEST: every LIM line read by the hart, BEU accrued = 0, no RNMI cause 3. LOAD+CALL of a checksum payload correct 10/10. **RNMI re-arm:** deliberately hung payload (infinite loop) → PARK → CALL → hang → PARK on the same hart, 10/10, `n_reentry` incrementing each time. |
| T-3.E2m LIM bench | same job | Results recorded (cycles per uncached load by width, I-fetch from LIM cached or not, Zero-Device option); §1.3 and the WP-3.7 gates are re-derived from them before WP-3.7 starts. |
| T-3.E3 power | same job | No heartbeat gap > 10 ms across 10× **legacy** fd open/close (`os.open(..., O_RDWR)`, which sends power messages; O_APPEND opens do not, ✔r `chardev.c:954, 1017, 1074-1085`) with relock evidence (monitor-measured mcycle/mtime dip or PLL4 CNTL_0 bypass reads), and server_g1 start → stop → start with the keeper fd held. PLL4 at default values after each event. Max TT latency during each fd event recorded. Optional E3(d), with approval and last: all fds closed 10 s, then attach → heartbeat resumes (informs whether reattach is ever usable; the default stays reset-per-session). |
| T-3.E4 bench | `$DEVX … --x280 boot:3 --vla tt --job $RT/jobs/e4_bench.sh` | Results recorded and a kernel chosen. Target compute p99 ≤ 1.0 ms for v1/v2 (gate ≤ 2.0 ms for `m3_rt`, ≤ 8.0 ms for `m3_rt_slow`/DP11). v0 is a parity reference: timing recorded, not gated. |
| T-3.E5a device parity | `… --job $RT/jobs/e5_parity.sh` (`x280d.py parity --obs <m1 run>/obs516.npz`) | Against the x86 build of the same blob: v0 bit-identical, v1/v2 ≤ 1e-5. Against ORT: ≤ 1e-2 (fp16), ≤ 5e-5 (fp32), on 10k obs. |
| T-3.5 timing | `… --job $RT/jobs/m3_rtt.sh` (`x280d.py bench --rtt --n 30000`, 10 min at 50 Hz with the VLA running) | x280d end-to-end p99 ≤ compute p99 + 0.5 ms and ≤ 2.0 ms (v1/v2); host mailbox overhead p99 ≤ 0.3 ms; 0 timeouts. Under DP11: ≤ compute p99 + 0.5 ms and ≤ 9 ms, or with early dispatch the visible wait p99 ≤ 2.0 ms. |
| T-3.6 shadow | `… --job $RT/jobs/m3_shadow.sh` (20 eps, live `rt_sim --wbc ort --shadow $CHIP_X280_URL`) | Per-tick \|a_x280 − a_ort\| p99.9 ≤ 1e-2 (fp16) or ≤ 5e-5 (fp32); at most 0.1% missed shadow replies (a miss counts as a failed tick) |
| T-3.7 closed loop | `… --job $RT/jobs/m3_eval.sh` (RT soak of 20 eps, then `eval_m3_screen.yaml`) | All `m3_rt` (or DP11 `m3_rt_slow`) gates. 0 fallbacks. Heartbeat continuity 100%. TT VLA p50 within ±5% of T-2.3's server span p50 (`M3_T23_BENCH` must name that bench; the session's `vla_p50_ms_max` is 1.05 × it). Screen (50 paired, vs the M2 configuration): lower 80% bound ≥ −25 pts; the n = 350 primary only if DP10 asks. |
| T-3.8 fault injection | `… --job $RT/jobs/m3_faults.sh` | Three cases: `kill -9 x280d` (keeper fd closes → KMD writes 'GONE' to L38 → firmware idles); `x280d.py park`; firmware detects a CRC mismatch. In each, `rt_sim` falls back to ORT within 1 tick, with no SystemExit and no fall, and the episode continues. A restarted x280d attaches without a release (server_g1's fd keeps the clock on). |
| T-3.9 soak + swap | `demo.sh m3 --duration 7200 --swap-wbc-every 10` | 0 traps, 0 unplanned fallbacks, 0 slips. Tick p99 ≤ 21 ms. \|Δa\| at a swap ≤ 1e-2. PLL4 registers at default values in x280d's direct reads every 10 s (no tt-smi). **2026-10-25 (user):** the GPU WBC has no place in M3, so T-3.9 runs X280-only by default (`jobs/m3_soak.sh`, no swaps, `--no-alt`; the swap row does not apply; `M3_SOAK_SWAP_S=N` restores the cycle). |

### M4: options, in recommended order (≈ 130 h+)

| WP | What | Files | Deps | h |
|---|---|---|---|---|
| WP-4.1 (M4a) | Stateful lower-body controller on X280: `LbStep` 72→15 (§1.4.6) in C. History, 1-tick lag, select, closed-form 3-joint torso FK, dt = 0.02. x86 + spike builds; conformance suite against Python `RefController` golden ticks recorded from the unchanged gr00t_wbc stack. W-LBC endpoint on `:5558` with `proto 3`. | `RT/x280/fw/lbstep.c`, `RT/tests/m4/test_lbstep_parity.py`, `RT/tools/record_golden_ticks.py` | M3 | 24 |
| WP-4.2 (M4b) | Splicer and the whole 50 Hz controller in C on X280, clocked by mtime (50 MHz). `rt_sim` splits into a robot process (shared-memory LowState/LowCmd, latest wins; plan A's real-robot shape). The host relays VLA requests the X280 asks for. | `RT/x280/fw/{splicer.c,controller.c}`, `RT/rtloop/robot_proc.py` | 4.1 | 34 |
| WP-4.3 (M4c) | On-chip VLA→WBC handoff. server_g1 copies normalised `action_pred` into a persistent DRAM tensor at a fixed address and PUSHes `{addr, bank_table, chunk_seq, k_obs}` to x280d (`:5572`), which writes L40–L41 (§1.4.6). The X280 maps a 128 GiB **uncached** window per bank (translated coordinates), re-checks seq against torn reads, then de-normalises and converts relative to absolute against its own 64-tick q ring. **First confirm the tensor's tile layout** and that ISA tile D7 equals tt-metal bank 7. The host still reads the chunk for the upper body. | `RT/rttt/server_g1.py` (flag), `RT/x280/fw/chunk_rx.c` | 4.2 | 24 |
| WP-4.4 (M4d) | RTC v2 (§3.3). v2a guided and v2b soft replacement on the 5090, in an RT-owned server that **runtime-patches** `Gr00tN1d6ActionHead.get_action_with_features` (no Isaac repo edit). Then v2b in the TT megakernel Euler stage, with Y, W and z as inputs (`TT/tt/action_head.py:750-770`): a **repo change** (DP8). | `RT/rtserve/gpu_vla_rtc_server.py`, TT patch | M2 | 16 + 30 |

| ID | Pass threshold |
|---|---|
| T-4.1 | C `LbStep` vs Python on 10k logged ticks (≥ 500 Walk): obs-vector reconstruction max\|Δ\| ≤ 1e-6, q15 ≤ 1e-5 rad (fp32 weights) or ≤ 2.5e-3 rad (fp16), policy choice 100% equal. Closed-loop screen: lower 80% bound vs M3 ≥ −25 pts. |
| T-4.2 | Replayed T-3.7 logs: trigger and splice decisions 100% identical to `splicer.py`. X280-clocked period p99 ≤ 20.2 ms. Robot lag 0 ticks in 99.9% of ticks. RT gates as T-3.7. |
| T-4.3 | The X280-decoded chunk equals the host-decoded chunk to ≤ 1e-6 over 10k chunks, with 0 torn reads. VLA-done → X280-sees-seq ≤ 1 ms. Closed-loop screen as T-4.1. |
| T-4.4 | At VL d ∈ {5, 8, 12}, n = 350 paired: v2 success ≥ v1 + 5 pts at some d ≥ 5 with the one-sided 95% lower bound > 0, **or** boundary-jump p99 ≤ 0.7 × v1 with v2 not clearly inferior. Prefix round trip `apply_action → unapply_action` ≤ 1e-6. Otherwise v2 is not adopted. |

---

## 3. Chunk scheduling policy

### 3.1 v1: aligned receding horizon (M1–M3; `RT/rtloop/splicer.py`, ported to C in M4b)

| Parameter | Value | Basis |
|---|---|---|
| Horizon | H = 30 rows (0.6 s). Model rows 30–49 are never used. | U:as "N1.6 action horizon" |
| Index law | row i ↔ tick k_obs + i. Chunk age = k − k_obs = the row executed. | ✔r `multistep_wrapper.py:256-269`; T-1.3, T-1.2b |
| Trigger | At tick k (after T_k, D7), if no request is in flight **and** any of: `k − k_req ≥ s_eff`; no active chunk; or the last outcome ∈ {late_reject, timeout, error, gen_drop}. `k_req` is the tick of the last *sent* request. In VL, "in flight" = "not yet visible", so triggers do not depend on reply timing. | U:as rec "V1 TRIGGER" |
| s_eff | `max(1, min(s_cfg = 16, 30 − d̂ − m))`, m = 2. Single-in-flight already enforces s ≥ d. | RTC constraint d ≤ s ≤ H − d (U:as "PI Real-Time Chunking") |
| d̂ | **p90 of the last 10 measured d** (the 9th order statistic), so one outlier (e.g. a 380 ms TT call, d = 19) does not shrink s_eff for 10 requests; the max is logged too. d_init = 4 on the 5090, 3 on TT. In VL, d̂ = d_inj. | RTC Alg. 1 (max), bounded for TT outliers (R21) |
| Latency compensation | None beyond the index law. Arms are already absolute targets anchored at k_obs: **never re-anchor**. | U:as "N1.6 action representation"; ✔r review |
| On arrival (k_arr, d = k_arr − k_obs) | Drop if `gen ≠ current`. **Late-reject** if usable rows 30 − d ≤ 4 (d ≥ 26, 520 ms), then retrigger at once. Otherwise switch at k_arr to row k − k_obs_new. | U:as rec "V1 SPLICE" |
| Blend | B = 0 by default. Option B ∈ {3, 5}: `a = (1−w)·last_row + w·new`, `w_j = (j+1)/(B+1)`, on arms, waist and nav only, where **`last_row` is the last executed 35-vector** (updated by select, hold and starve). Hands and base_height switch hard. | U:as rec "V1 SPLICE"; review C-8 |
| Starvation (row ≥ 30) | Hold the last executed arm, waist, hand and height targets. Ramp nav linearly to [0,0,0] over 5 ticks, which puts the WBC in Balance. Keep requesting. Resume from row k − k_obs at the next arrival. | U:as rec "V1 STALE / LATE HANDLING" |
| Faults | **STARVE_FAULT**: 50 consecutive starved ticks (1 s) set the flag; the episode continues and can still succeed; it counts as loop-attributable. **SAFETY_FAULT**: `SystemExit` from the safety monitor; the episode ends as a failure. | 1 s goal timeout (✔ `g1_decoupled_whole_body_policy.py:94-109`); LeRobot `action_starvation_grace_s=1.0` |
| Timeouts | VLA: give up **500 ms** after send ((26 − 1) × 20 ms; any later reply would be late-rejected), lazy pirate, outcome `timeout`, retrigger. WBC: 5 ms → ORT. | U:srv "Client robustness" |
| Balance/Walk | No hysteresis in v1 (fidelity to the evaluated setup). Flips are logged. | DP5 |
| Episode start | Pre-roll (§1.2). Reset and instruction change bump `gen`. | – |

Expected behaviour, 5090 (d = 4): executed rows ≈ [4, 20), inside the sync rows 0–19 shifted by d; replan every 320 ms;
GPU duty ≈ 20%; mean chunk age ≈ d + 7.5 ≈ 11.5 ticks (230 ms), against 9.5 for sync; s = 16 tolerates d ≤ 12 ticks
(240 ms) before starving. TT (d = 3): the same s, with duty ≈ 14%.

```python
def select(k, st, cfg):   # st: plain fields {act, pending, blend_t0, starve_n, last_row, last_row_at_switch}
    if st.pending is not None and st.pending.k_arr <= k:           # registered by on_arrival() (RT) / visible() (VL)
        st.act, st.pending, st.blend_t0 = st.pending, None, k
        st.last_row_at_switch = st.last_row                        # what was actually executed (held/ramped or chunk row)
    if st.act is None:
        return hold(st)                                            # pre-roll only; sets st.last_row
    i = k - st.act.k_obs
    if i >= 30:
        return starve(st, k)                                       # hold + nav ramp; STARVE_FAULT at 50; sets st.last_row
    st.starve_n = 0
    cmd = st.act.rows[i].copy()                                    # never mutate a stored chunk
    if cfg.B and st.last_row_at_switch is not None and (j := k - st.blend_t0) < cfg.B:
        w = (j + 1) / (cfg.B + 1)
        cmd[BLEND] = (1 - w) * st.last_row_at_switch[BLEND] + w * cmd[BLEND]   # BLEND = arms, waist, nav
    st.last_row = cmd
    return cmd
```

`last_row_at_switch` is `st.last_row` frozen at the switch tick (the command actually executed just before the new
chunk took over), so a blend after starvation starts from the held targets and nav = 0, never from the stale chunk.
The blend is a crossfade from that fixed command; the §3.1 table's `last_row` in the formula means this frozen value.

### 3.2 Baseline and ablation arms (same code path, different parameters)

| Arm | Definition | Role |
|---|---|---|
| C0 stock | Unmodified `rollout_policy.py`, n_envs 1, n_action_steps 20, 1440 ticks, unseeded | Anchor: C1 must agree within the 95% CI |
| C1 sync_frozen | `--mode sync`: sim frozen during each call, rows 0–19 | User-requested sync baseline (optimistic: it hides latency) |
| C2 sync_pause | RT/VL: trigger when rows 0–19 are used up; during the d ticks of inference hold arms, hands, waist and height and **ramp nav to 0** (the robot pauses, as in RTC's synchronous baseline); then rows 0–19 from arrival, starting at row 0 | Honest synchronous robot |
| C2b sync_hold (optional) | As C2, but nav is held at its last value (walks blind for d ticks) | Sensitivity of the time metric to C2's definition |
| C3 async_noskip | s = 16, but the new chunk starts at row 0 on arrival | Negative control for latency compensation |
| **C4 v1** | §3.1, B = 0, s = 16 | **Primary / demo** |
| C5 v1 + blend | B = 5 | Ablation |
| C6 sync_aligned | s_eff = 20 − d̂ (switch near row 20, as sync does) | Ablation (U:sim rec "Request scheduling") |
| C7 v2a / v2b | §3.3 | M4d |

### 3.3 v2: real-time chunking (RTC), a later option (M4d)

- **Insertion point:** `Gr00tN1d6ActionHead.get_action_with_features` (`R/gr00t/model/gr00t_n1d6/gr00t_n1d6.py:288-363`).
  H = 30, s = 15, d̂ as in v1. N1.6's 30 rows are below the ≥ 32 the Isaac guide asks for, so s ≤ 15 and d ≤ 15 are
  hard ceilings.
- **Prefix Y:** the not-yet-executed absolute rows of the active chunk for ticks [k_obs_new, k_obs_new + 30), sent in
  `options.rtc`. The server re-encodes them **relative to the new observation's state**, normalises them with the
  per-index (30,7) statistics, pads to (50,128), and sets W = 0 on dims ≥ 35 and on rows past the old chunk.
- **Mask (RTC Eq. 5):** W_i = 1 for i < d; `c_i (e^{c_i} − 1)/(e − 1)` with `c_i = (H−s−i)/(H−s−d+1)` for d ≤ i < H−s;
  0 beyond. For d = 3: [1,1,1,.815,.655,.518,.402,.305,.224,.158,.105,.065,.035,.015,.004,0,…]. Guidance clip β = 5,
  which gives per-step weights [5, 3.33, 2.0, 3.33] at τ = 0, .25, .5, .75.
- **Variants:** **v2a** guided VJP (5090 reference only; breaks `inference_mode` and CUDA graphs; ≈ +28% latency);
  **v2b** gradient-free soft replacement `x ← x + W·((1−τ)z + τY − x)` before each of the 4 Euler steps (zero cost,
  portable to the TT megakernel); **v2c** training-time RTC (per-token AdaLN timesteps plus a fine-tune with
  d ~ U[0,10); best for Tensix; out of scope unless DP8 says otherwise).
- **Adoption:** only through T-4.4.

### 3.4 Pilot decision rule (DP-S, after SP-1)

SP-1 has 50 pairs per cell: SE of a paired difference ≈ 8–10 pts, so it detects only ≈ 22–25 pt changes at 80% power.
Its bands are triage, not claims. Alignment is not judged here: T-1.2b checks it deterministically. L80 is the lower
one-sided 80% bound of (C4@4 − C1) = Δ̂ − 0.84·SE.

| SP-1 result | Action |
|---|---|
| L80 ≥ −15 pts | v1 is the M1–M3 policy; v2 stays in M4 |
| −25 ≤ L80 < −15 pts | Run the s ∈ {10, 16, 20−d̂} and B ∈ {3, 5} VL sweeps first (≈ 2 h machine time), then start v2a/v2b on the 5090 in parallel with M2 |
| L80 < −25 pts | Confirm T-1.2b passes; then the latency cost is real: v2a/v2b become the next Lane-A work, in parallel with M2 |

C3 vs C4 is compared only at d_inj ∈ {8, 12}, where the latency skip should matter; a one-sided paired p < 0.05 for
C4 > C3 is reported as "latency compensation helps". It is never a stop trigger.

---

## 4. Measurement, logging and evaluation

### 4.1 Logs

Written by `rt_sim`. One run dir per invocation: `RT/runs/<stamp>_<tag>/`.

**TickLog** (`ticks.npy`, a structured `np.memmap` ring flushed every 50 ticks, so a crashing episode is kept):

| Group | Fields (dtype) |
|---|---|
| Timing | `k` i64, `t_start_ns` i64, `t_end_ns` i64, `period_us` f32, `compute_us` f32, `late_us` f32, `slack_us` f32, `overrun` u1, `slip` u1, `sim_us` f32, `viewer_us` f32 |
| WBC | `wbc_rtt_us` f32, `wbc_dev_us` f32 (server-reported), `wbc_be` u1 (0 ORT, 1 GPU, 2 X280, 3 X280-sim), `wbc_fb` u1, `wbc_early` u1, `policy` u1 (0 Balance, 1 Walk), `shadow_da` f32 |
| Chunk | `chunk_seq` u32, `row` i16 (−1 = none), `age` i16, `rows_left` i16, `starved` u1, `in_flight` u1, `splice` u1, `jump_rad` f32 |
| Commands and state | `cmd35` f32[35], `nav_wbc` f32[3], `q15` f32[15], `q43_meas` f32[43] (measured, end of step), `q43_cmd` f32[43], `dq_arm_absmax` f32 (measured, the safety monitor's quantity, ✔r `joint_safety.py:206-226`), `pelvis_z` f32, `pelvis_rp` f32[2], `success` u1, `safety` u1 |

**RequestLog** (`requests.jsonl` plus `requests.npz` for arrays, one row per VLA request):
- Identity and timing: seq, gen, ep, k_obs, t_obs_ns, t_send_ns, t_recv_ns, k_arr, d, d̂ (p90 and max), s_eff,
  noise_seed.
- Sent observation: `state43_sent` f32[43] (audits the relative→absolute anchoring), `obs_sha1`.
- Cost: ser_us and deser_us, plus the server's INFO timings when present.
- Usage: first_row, last_row, n_rows_used, outcome ∈ {applied, late_reject, gen_drop, timeout, error, drained}.
- At the switch tick: L2 jump over arms + waist, velocity-direction cosine and intra-chunk acceleration, using the
  formulas in `R/getting_started/real_world_deployment.md:283-362`, computed on the **executed** target sequence.

**EpisodeLog** (`episodes.jsonl`): arm, mode, d_inj, s, B, seed, crn, vla_be, wbc_be, checkpoint@sha, tree, success,
first_success_tick, end ∈ {SUCCESS, TRUNC, FALL, SAFETY_FAULT}, starve_fault, n_starved, n_late, n_flips,
n_fallback, drain, boundary_jump_p99, tick_p99, chunk_age_mean.

**Other files:**
- `config.json`: resolved CLI, git SHAs and `git status --porcelain` of RT/R/W, sha256 of every imported dirty or
  untracked file (R has `?? gr00t/policy/gear_wbc_policy.py` and `M gr00t/eval/run_gr00t_server.py`; W has
  `M …/g1_gear_wbc_policy.py` and an untracked `gr00trobosuite/`, ✔ `git status`), the vendored-file hashes, the MK or
  live-tree stamp, endpoints, backend `info` stamps, thread/CPU affinity, ORT options and the fidelity deviations of
  §1.1.
- `obs516.npz` and `vla_obs/` (opt-in), plots, `verdict.json`.
- Server-side JSONL: gpu_vla_server and server_g1 per call; x280d per tick in a ring (recv→send µs, cycles, status,
  heartbeat, early-dispatch hit/miss).
- Device health: PLL4 registers and heartbeat read by x280d every 10 s on its open windows. **Never run tt-smi during
  a run.** Legacy (non-O_APPEND) opens and the closes of fds with a power contribution each send a power message, and
  every message re-runs the PLL4 sequence four times (✔r `chardev.c:954-1090`; fw `bh_power.c:32-54`). tt-smi and
  pyluwen open O_APPEND; whether `tt-smi -s` sends an explicit power message is unverified, so the policy stays.

### 4.2 Gates (`RT/gates.yaml`, applied by `analyze_run.py`)

| Gate set | Period p99 / max | Overruns | Slips | WBC RTT p99 | d p99 | Starvation | Other |
|---|---|---|---|---|---|---|---|
| `m1_rt` | ≤ 21 / ≤ 30 ms | ≤ 0.5% | 0 | ≤ 2.0 ms; RTT > 20 ms < 0.1% | ≤ 5 | 0 ticks | shadow p99 ≤ 1e-4; fallbacks ≤ 0.1%; SAFETY_FAULT 0; viewer_us(k=0) ≤ 3 ms; VLA p50 ≤ 1.1 × 65 ms |
| `m1_negctl` | – | – | – | expected: ≥ 3% of calls > 20 ms or timeouts | – | – | – |
| `m2_rt` | ≤ 21 / ≤ 30 ms | ≤ 0.5% | 0 | ≤ 2.0 ms | ≤ 3 | ≤ 0.1%, runs ≤ 10 ticks | TT errors 0; VLA p50 ≤ 1.1 × T-2.3 p50 |
| `m3_sim` | ≤ 21 / ≤ 30 ms | ≤ 0.5% | 0 | ≤ 2.0 ms | ≤ 5 (T-1.4's row: T-3.3 runs the GPU VLA or the 60 ± 5 ms mock, d = 4; ≤ 3 assumes the TT VLA) | as m2 | fallbacks ≤ 0.1% (T-3.3 = all T-1.4 gates) |
| `m3_rt` | ≤ 21 / ≤ 30 ms | ≤ 0.5% | 0 | ≤ 2.0 ms | ≤ 3 | as m2 | fallbacks 0; heartbeat 100% |
| `m3_rt_slow` (DP11) | ≤ 21 / ≤ 30 ms | ≤ 0.5% | 0 | ≤ 9.0 ms, or ≤ 2.0 ms visible wait with early dispatch | ≤ 3 | as m2 | fallbacks 0; heartbeat 100% |

Plots: a Gantt chart (ticks against request bars and switches); histograms of period, lateness, d, chunk age
(expected within [d, s+d)) and WBC RTT; boundary-jump CDFs per arm.

### 4.3 HUD (`hud.py`, pygame, 1280×720, 30 fps, separate process)

1. **Processor map.** Boxes for Host CPU, RTX 5090 and Blackhole [Tensix | X280], lit by the component that runs there,
   labelled from the backends' `info` stamps; stamps from stock servers are marked "(cfg)".
2. **VLA panel.** Last, p50 and p99 latency over 50 requests, with the server split. d, d̂, s_eff, an in-flight
   spinner, and late/timeout counters.
3. **Chunk bar.** 30 cells: executed rows, the cursor, and the [d, s+d) window. Age in ms, rows left, and a large red
   STARVED when starved.
4. **WBC panel.** Tick period now and p99, jitter std, round trip now and p99, device µs, Balance/Walk indicator,
   fallback/overrun/slip counters, real-time factor.
5. **Timeline** of the last 4 s: compute bars (red if > 20 ms), VLA request bars, splice markers.
6. **Ego thumbnail** of the image the active chunk came from, stamped with k_obs. Episode, seed, success and falls.
7. **Blackhole chip panel** (user request 2026-10-24; M2 for Tensix, M3 for X280). A chip floorplan with the Tensix
   grid and the four L2CPU tiles, plus two status columns:
   - *Tensix (server_g1):* per-call encode/device/decode ms, device duty (device_ms / request period), calls and
     errors, trace/CQ mode, the megakernel core map (static: 8 readers + 1 hub + 96 compute + 5 spares on 11×10).
   - *X280 (x280d):* tile state from the ledger (fresh/released/parked/burned), monitor `state`, heartbeat age,
     per-call µs from `mcycle` (L1 `cycles_last`/`cycles_max`) and its duty within the 20 ms tick, `n_runs`,
     `n_overrun`, `n_reentry`, BEU accrued and CCACHE ECC counters, kernel variant (v1/v2, fp16/fp32), fw build id.
   - *Chip telemetry (CMFW table, ≈ 100 ms snapshot, fw `bh_arc/telemetry.h`):* AICLK, `L2CPUCLK3`, ASIC
     temperature, VCORE, TDP/TDC, input power, `TIMER_HEARTBEAT`. Read from memory only: no ARC messages and **no
     tt-smi** (§5.2 invariant 3; a new fd sends power messages).
   Data paths, chosen so the HUD never adds load to a control-path socket: per-call fields (device ms, X280
   `compute_us`, variant) ride in the existing replies and reach the HUD through the loop's `tick`/`req` topics; slow
   status comes from an out-of-band PUB in each device process (x280d `:5573`, server_g1 `:5574`, 2–10 Hz). The HUD
   never sends requests to `:5557` or `:5558`, because a REP socket serves one request at a time and a HUD poll would
   delay a WBC call. Telemetry is read by x280d through its ARC window (read-only and in the `regs.py` read
   allowlist). server_g1 publishes `telemetry: None`: ttnn has no read of the CMFW table, and a second fd would send
   power messages, so a TT-only session shows no chip telemetry.

Keys: `v` swaps VLA, `w` swaps WBC, `r` resets, `p` pauses. All timing gates are measured **with the HUD and recording
on**.

### 4.4 Evaluation protocol

> **DP10 decided (2026-10-24): 50 episodes per arm is enough.** Every "primary" below runs at n = 50 paired (seeds 2000–2049) and gates only gross drops (lower 80% bound ≥ −25 pts). Where this section still mentions n = 350 or n = 250, read n = 50; those tiers are not run.

**Pairing and variance reduction.**
- Scenes: seeds via `env.reset(seed=s)`; the stock rollout resets unseeded. Seed sets: tuning 0–99, screen 1000–1049,
  primary and secondary 2000–2349, liveness 3000–.
- VLA noise: the stock server draws unseeded noise (`gr00t_n1d6.py:310-314`). With `--crn` (D17), the RT-owned GPU
  server and server_g1 seed each call from `(ep_seed, seq)`. Arms with different trigger schedules share noise only
  for matching request indices, so CRN reduces but does not remove discordance; SP-1b measures by how much.
- WBC backend: VL matrices always run `--wbc ort` (recorded in EpisodeLog); RT arms run the milestone's WBC.

**Episode rule** (identical in every arm): success = success at any tick ≤ 1440, counted at the first success tick.
FALL and SAFETY_FAULT are failures and are attributed to the loop when the logs show a loop cause (starvation,
fallback, safety trip). STARVE_FAULT episodes keep their success outcome but are flagged loop-attributable; nominal
runs require 0.

**Tiers.**

| Tier | n (paired) | Use |
|---|---|---|
| smoke | 5 | after any change |
| screen | 50 | triage (MDE ≈ 22–25 pts at 80% power); picks variants; d-sweep curves |
| primary | **350** | one pre-registered comparison per milestone exit (MDE ≈ 8–9 pts) |

**Primary comparisons** (one look each, no extension, decided before the run):
- M1 (T-1.9): RT C4 vs RT C1, DP1 checkpoint, GPU WBC, `--crn`.
- M2 (T-2.7): RT C4 on TT vs RT C4 on GPU.
- M3: screen only by default (shadow and parity gates are the primary instrument for a WBC swap with ≈ 7e-3 numeric
  error); a primary run only if DP10 asks.

**Verdicts** for a primary comparison of (new − reference):
- **Gate (exit):** *not clearly inferior*: one-sided 95% upper bound ≥ −10 pts. At n = 350 and discordance 0.32–0.49
  this passes a correct loop with probability ≈ 1 at Δ = 0, and fails a true −20 pt drop with probability 0.85–0.95.
- **Claim (report):** *non-inferior* when the one-sided 95% lower bound > −10 pts (probability 0.85–0.95 at Δ = 0,
  0.38–0.50 at Δ = −5); otherwise *inconclusive* or *inferior*, reported as such. No "add 100 seeds" rule.
- Reported with Δ̂, a two-sided 95% paired Wald CI, exact McNemar p and per-arm Wilson CIs.
- A clearly inferior result triggers DP-S's v2 track, not a loop-bug hunt, provided T-1.2b passes.

**Secondary metrics** (reported, not gated):
- VL d-sweep: C4@{0,3,4,5,8,12} − C1, n = 50 each (screen seeds).
- C3 vs C4 at d_inj ∈ {8, 12}, n = 50.
- Time-to-success: C4 vs C2 at VL d_inj = 5, n = 250 (seeds 2000–2249), as paired censored data (failures censored at
  1440 ticks): paired restricted-mean difference with a seed bootstrap CI, plus a seed-stratified log-rank test; C2b
  optional.
- Falls, starvation, boundary jump, Balance/Walk flips, tick jitter.
- No conclusions from n < 50: the README warns of ±15% variance (✔ `R/examples/GR00T-WholeBodyControl/README.md:9-13`).

**Matrices and machine time.** VL ≈ 15 s/episode (1440 × 5 ms + 90 calls × ≈ 50 ms + 2.8 s reset); RT ≈ 32 s/episode.

| Matrix | Cells | Size | Time |
|---|---|---|---|
| M1 screen | VL: C1; C2@4; C3@{8,12}; C4@{0,3,4,5,8,12}; C5, C6@4. RT: C2, C4 | 12 VL + 2 RT cells × 50 | ≈ 1.5 h VL (4 procs, VLA-bound) + 55 min RT |
| M1 primary | RT: C1 (`--mode sync`), C4 | 2 × 350 | ≈ 6.2 h |
| M1 secondary | VL: C4@5, C2@5 (n = 250) | 2 × 250 | ≈ 1.5 h |
| M2 | C1 on TT vs GPU (screen); C4 TT vs GPU (primary; both arms fresh, the GPU VLA with `--noise cpu`) | 50 + 350 | ≈ 4 h in S1b |
| M3 | C4 TT+X280 vs C4 TT+GPU WBC (screen) | 50 | ≈ 1 h |

**Controls.** 20 warm-up calls before each run. One dedicated VLA instance per campaign (`:5565` or `:5557`). Arms
interleaved by seed blocks of 10 to cancel thermal and clock drift. Affinity as in §1.1. VRAM checked by preflight.

---

## 5. Risk register and X280 sequencing

### 5.1 Register (each row names the WP and the test that closes it)

| # | Risk | Evidence | Mitigation | Closes |
|---|---|---|---|---|
| R1 | WBC head-of-line blocked behind the VLA on one REP socket | U:srv "Q1 measured head-of-line blocking": p99 67 ms, 35.7% of ticks > 20 ms back-to-back, 5.1% at a 0.4 s cadence | Own process and port (D3). A negative control with VLA and WBC on the same REP (:5555) shows the instrument sees it. | WP-1.4 → T-1.4, T-1.5 |
| R2 | A CUDA context in the sim corrupts EGL; `CUDA_VISIBLE_DEVICES=""` alone crashes EGL init | U5; ✔ `egl_context.py:39-55` | `simpy` sets both variables. Env assertion after imports and resets; nvidia-smi compute-apps check; frame identity. | WP-1.5 → T-1.7 |
| R3 | A chunk-switch jump drives an arm past 6 rad/s, then `sys.exit(1)` | ✔ `sync_env.py:516-528`, `joint_safety.py:451-454` | `SystemExit` guard → SAFETY_FAULT and reset. `dq_arm_absmax` logged. Boundary-jump metric. Option B ∈ {3,5}. | WP-1.5 → T-1.4 (0), T-1.6 |
| R4 | Latency shifts the executed rows (a[d..d+s) instead of a[0..20)), so success drops (baseline only 30–58%) | U:sim risks; U:as risks | SP-1 in week 1, DP-S, VL d-sweep, sync_pause baseline, s/B sweeps, v2 track; primary comparison gates only gross drops | WP-1.9 → T-1.8, T-1.9 |
| R5 | `PolicyClient.timeout_ms` is never applied, so a dead server hangs the sim | U:srv "Client robustness" | Own client: `poll`, RCVTIMEO, 500 ms give-up, lazy pirate, episode-end drain | WP-1.3 → T-1.1, T-1.6 |
| R6 | Reset takes 2.5–2.8 s and re-creates MjSim and WBC, so handles go stale | U:sim "Things that do break"; ✔ `n1_utils.py:29-33` | Reset outside the clock. The class-level seam (D5) survives re-creation. Gen bump. Re-anchor. | WP-1.5 → T-1.10 |
| R7 | Serialising the full 9.2 MB obs stalls the tick (9.9 ms) | U:srv "VLA latency and payload" | 9-key trim (0.92 MB, 0.14 ms) | WP-1.3 → T-1.1 |
| R8 | GPU contention between EGL renders, two VLA instances and the WBC graph | unmeasured (U:sim gaps) | Separate WBC process (E3: p99 0.62 ms under load). Measured in the soak. | T-1.4 (VLA p50 ≤ 1.1×) |
| R9 | The onscreen viewer: relaunched inside tick 0 after every reset; `viewer.sync` cost unmeasured | ✔ `base.py:270-272, 478-484`; U:sim gaps | Pre-launch in the reset window (§1.2); `viewer_us` gated at k = 0 and p99. Fallback ladder: viewer every 2nd tick → offscreen tpp stream to the HUD. | WP-1.5 → T-1.4, T-1.7 |
| R10 | CPU contention: sim + server threads + EGL + viewer thread on 16 cores | U:srv gaps; ✔r `mjviewer_renderer.py:28` (render thread) | Tick thread alone on CPU 2 (sibling 18 idle); library threads on 3; servers on their own cores; `gc.disable()` within episodes | T-1.4 |
| R11 | Mixed time bases (monotonic vs sim) inject stop goals | ✔ `g1_decoupled_whole_body_policy.py:91-110` | `set_goal` every tick. Sim time is never passed. Identity upper body only. | WP-1.5 → T-1.2 |
| R12 | Balance/Walk flicker around ‖cmd‖ = 0.05 | ✔ `g1_gear_wbc_policy.py:257` | Flips logged; hysteresis only with DP5 | WP-1.6 → T-1.9 (secondary) |
| R13 | Off-by-one in k_obs stamping or the observation sent would shift every chunk silently; separately, whether the *model* was trained with row 0 ↔ t_obs is inferred, not checked | U:sim gaps ("alignment … inferred from delta_indices"); ✔r server anchors to the request's state | T-1.2b (bit-exact ramp test) detects loop misalignment; T-1.3 shows the loop equals the official convention. The training semantics stay an open question: settle offline when the dataset is available (does predicted relative row 0 match q_t or q_{t+1}?) or with a VL ±1 index-offset ablation (screen). | WP-1.8 → T-1.2b, T-1.3; §6.3(a) |
| R14 | Prototypes and X280 reference clones live in `/tmp` and will be lost | `S/plan_research/*` | Copied with their SHAs | WP-1.0 |
| R15 | Port clash: gr00t_wbc image publisher and camera defaults use 5555 | U:sim "Port collision" | Never instantiated; fixed RT ports; preflight | WP-1.0 |
| R16 | HUD or recording perturbs timing | in-loop video costs 17–87 ms (U:sim) | Out of process; PUB drop; gates measured with them on | WP-1.7 → T-1.4 |
| R17 | GPU memory: 8.5 GB (`:5555`) + ≈ 8 GB (`:5565`) + goldens ≈ 15 GB | U:srv risks | Stop `:5565` during WP-2.2; VRAM preflight | WP-2.2 |
| R18 | TT G1 configuration gaps: no g1 layout, pinned sha, F32 cloudwalk tensors, cache-key collision, fixed noise seed 42 | U:tt "UNITREE_G1 support", "Weight-cache key collision", "Noise semantics" | `install()` (LIBERO pattern); BF16 nvidia default; dedicated cache root; per-call generator noise; private HF cache for any conversion | WP-2.1/2.1b/2.3 → T-2.1, T-2.3 |
| R19 | TT fidelity on the G1 fine-tune is unknown (GR1-only evidence 0.999232 vs fp32; hands max\|d\| 0.0925) | U:tt "Correctness evidence for N1.6" | Gates against the deployed bf16 reference with an fp32-relative secondary and absolute per-group floors; boot check is a sanity gate (0.995); bf16-arena fallback; DP1/DP6 | WP-2.2/2.5 → T-2.2, T-2.5 |
| R20 | The LIBERO server exists only in the live tree | ✔ `ls`; ✔r MK and live are identical for configs/normalization/preprocessing/layers/weights/checkpoint; no live file newer than the LIBERO run | Port into RT; MK import test; live tree as fallback (DP6) | WP-2.3 → T-2.1 |
| R21 | TT latency outliers (LIBERO max 380 ms, including encode 95.65 ms) cause starvation; CPU contention with the GPU VLA during hot swap | ✔r `tt_spatial_r0_summary.json`, `…_server_calls.jsonl` | d̂ = p90 (outlier-bounded); s_eff clamp; 500 ms give-up; bounded starvation gate; server_g1 on 8–11 and `:5565` on 12–15 (or stopped); calls > 100 ms attributed | T-2.3, T-2.6 |
| R22 | Card stability: two megakernels hang the card; the guard probe may reset it | U:tt "Long-lived server loop" | One model per process; `chip_session`; never nest `DEV` | WP-2.4 |
| R23 | Live-tree drift (Stage-3 WIP); live tree is untracked, so git cannot identify it | U:tt "Working tree vs validated snapshot"; ✔r `git ls-files` = 0 | MK by default; `info.tree` content-hash stamp | WP-2.3 |
| R24 | **Harts leave reset only once per chip reset**; a hung hart without a working park costs the session | ✔ ISA `L2CPUTile/README.md:29-30` | Monitor first; RNMI park with an explicit re-arm handshake (RNMIs are level-triggered and entry clears NMIE; the ISA's recommended stub never re-arms, ✔ `RNMIs.md:15-34`, `README.md:30`); payloads only by LOAD/CALL; spike run first; release before the TT build; reset after every X280 hold | WP-3.2/3.2b/3.4/3.6 → T-3.2b, T-3.E2 |
| R25 | tt-bh-linux's flow breaks tt-metal (board reset, GDDR bank 5, 4 GiB windows, WAYENABLE) | U:x "Why tt-bh-linux cannot coexist" | Never run `boot.py`; LIM only; 2 MiB windows; WAYENABLE interlock | WP-3.4 → T-3.4, T-3.E0 |
| R26 | **LIM ECC: a partial write to an uninitialised line can fault, and host writes may not be able to initialise it** | ✔ `test_hd_sockets.cpp:463-467`: "These do not complete construction of an L2CPU socket, which would write to LIM; a partial-line write to LIM whose ECC has not been initialised can fault, and initialising it requires running code on the L2CPU itself." vs ✔ `cluster.cpp:118-120` ("write a full 64-byte line first"); ✔r luwen UC u32 copies | Method A (WC 64 B NT stores), B (ttnn `write_to_core`), C (hart-run stage-0, E1F); E1a before any TT or release, instrumented with readback, CCACHE ECC counters and the SET_NOC_CLEANUP check; the hart's SELFTEST reads every line with BEU check at E2 | WP-3.3/3.4 → T-3.4, T-3.E1a, T-3.E1b, T-3.E2 |
| R27 | L2CPU clock gated when no fd asks for it; PLL4 relocks on every power message (legacy opens and power-contributing closes) | U:x "L2CPU clock gating", "CMFW re-locks PLL4"; ✔r `chardev.c:954-1090` | Keeper legacy fd; wait ≥ 1 s after open; no fd churn or tt-smi during runs; E3 with legacy-fd churn and relock evidence | WP-3.6 → T-3.E3, T-3.9 |
| R28 | The host hangs on a NoC access to a gated tile; telemetry is up to 100 ms stale | U:x "L2CPU clock gating"; ✔r fw `telemetry.c:278, 591-592` | Interlocks read PLL4 directly before each batch of L2CPU accesses; ENABLED_L2CPU check; probes in a subprocess with a timeout | WP-3.4 → T-3.E0 |
| R29 | Guard interactions: other `DEV` jobs block on the flock for up to `DEVICE_LOCK_TIMEOUT` (default 7200 s) while a session holds it; the 1800 s fuser wait applies only to an fd still open after a hold (orphan); a nested `DEV` deadlocks; the guard's probe is Tensix-only and cannot see L2CPU damage | ✔ `with-device.sh:8-10`; ✔ `_device_guard.sh:33-44, 63-74, 122-130` | One hold per session; children only; x280d exits with the session; DEVX reset after X280 holds; chip_session's L2CPU health check; long holds scheduled with other users | WP-2.4 → T-3.E0 |
| R30 | Own KMD layer bugs: a wrong TLB config silently targets the wrong tile/address | new code (replaces pyluwen, whose 0.8.5 behaviour was unaudited and whose UC path cannot init LIM) | `regs.py` allowlist; fake-device unit tests; `--dry-run`; read-only E0 first with `x280_peek` cross-check | WP-3.3 → T-3.4, T-3.E0 |
| R31 | L2CPU coordinate confusion: translated (8,26..29) vs identity (8,3/9/5/7) | U:x risks | `regs.py` is the single source; E0 cross-read | T-3.E0 |
| R32 | The watcher rejects non-LIM L2CPU writes; alloc tracking slows replay (589 ms) | U:x "tt-metal already knows"; U:tt | `chip_session` unsets both | WP-2.4 |
| R33 | **X280 slower than budget: LIM data is uncached; the scalar kernel may take ≈ 15 ms**; Zvfh absent | ✔ `Caches.md:26-31`; U:x "Performance estimate" (low confidence) | E2m microbench before kernel choice; vector kernels; integer fp16 widening; Zero-Device pinned working set (E4c); early dispatch (WP-3.8b, numerically identical); `m3_rt_slow` (DP11) | WP-3.7/3.8b → T-3.E2m, T-3.E4, T-3.5 |
| R34 | fp16 weights (≈ 7e-3 error) shift closed-loop behaviour | U:x "Weight precision" | Shadow run, then screen; fp32 on two tiles (DP4) | WP-3.8 → T-3.6, T-3.7 |
| R35 | Memory collision: the X280's local GDDR is tt-metal banks 5/6/7 | U:x "DRAM bank layout" | LIM only in M3; M4c reads a tt-metal-allocated tensor, read-only | T-3.2, T-3.4; WP-4.3 |
| R36 | Torn or stale reads: the host's L0 write is several stores; the M4c chunk is written by tt-metal | ✔r luwen/libwcio write granularity; U:x "Coherence" | `seq_req` last and separate; CRC over body and obs; heartbeat read hi-lo-hi; uncached alias plus seq re-check in M4c | T-3.8, T-4.3 |
| R37 | No UART: X280 failures are opaque; no RISC-V emulator was installed | U:x risks; ✔ `which spike` → none | Spike runs (WP-3.2b); trap dump to L1; x86 builds; mailbox sim; dry-run | WP-3.2/3.2b/3.4 → T-3.2b, T-3.3 |
| R38 | Underpowered comparisons mislead (±15% variance); a 3-cell must-pass gate blocks correct loops | README; review C-5 simulation (n = 250: single NI pass 0.81 at Δ = 0, 0.63 for three cells) | D18: functional exits, one primary comparison, n = 350, one look, "not clearly inferior" gate, CRN | WP-1.9 |

### 5.2 Sequencing X280 experiments safely under the once-only reset

**Invariants (every step).**
1. Never write `CCACHE0_WAYENABLE` (it is one-way) or touch GDDR. Write `L2CPU_RESET` only by the one RMW of boot step
   7. Never run tt-bh-linux `boot.py` (it board-resets and sets WAYENABLE = 0xF).
2. All X280 access happens inside **one `DEVX` hold** run by `chip_session.sh`. The guard's probe runs at hold start,
   **before** any hart is released, and the guard resets the chip after the hold. Inside the hold every restart is a
   child process; `DEV` is never nested (nesting blocks on `.p150a.lock`).
3. x280d opens its keeper legacy fd and its windows **before** any L2CPU access, waits ≥ 1 s, and closes them only at
   session end. No other fd is opened during a step (no tt-smi, no second tt-metal process), except the deliberate
   E3 churn.
4. Before each batch of L2CPU-tile accesses: PLL4 registers at their default values (read directly) and ENABLED_L2CPU
   bit 3 set. ARC-tile registers are read first.
5. `TT_METAL_WATCHER` and `TT_METAL_TRACE_ALLOC_TRACKING` are unset. x280d sends no ARC messages.
6. The 5090 M2 configuration (WBC `:5556`) and the ORT fallback stay runnable, so an X280 failure never blocks the demo.

**Session model (D13, DP3).**
- Every X280 hold ends with a chip reset (DEVX), so each session starts on a fresh chip and spends at most one release
  per tile. The ledger `~/experiments/gr00t/.x280_ledger.json` is per hold:
  `{hold_id, started_at, timer_heartbeat_at_start, tiles: {0..3: {state: fresh|released|parked|burned, image_crc, released_at}}}`.
  `TIMER_HEARTBEAT` (monotonic since CMFW boot) is recorded as an epoch cross-check: a smaller value than the last
  hold's means the chip was reset in between.
- `x280d boot:3`: if `L2CPU_RESET[7:4] ≠ 0` at session start (someone else released a tile without a reset), refuse
  and exit 3 unless `--allow-reset`, in which case chip_session resets first (WP-2.4 order). Otherwise release.
- `attach` is used only inside a hold (x280d restart while server_g1 keeps the clock on, T-3.8).
- Reset cost per session (≈ 1 min in U:x, unverified) is measured at the end of S1c; budget 2 min plus the TT warm
  boot (≈ 4 s `from_pretrained` + capture).

**Order.**

| Step | Session | Releases | Worst case | Recovery | Preconditions |
|---|---|---|---|---|---|
| E0 read-only probe (keeper fd only, then with server_g1) | S1c | 0 | host hang on a gated tile | power cycle (avoided by the PLL4 interlock and subprocess timeouts) | T-3.4 passed; WP-3.3 tests |
| E1a LIM init (method A) + checks, **no TT** | S1c | 0 | LIM fault or wedged tile | job exits 3; DEVX resets after the hold | E0 pass |
| E1b same pattern on a second region with server_g1 serving (method B here if A failed) | S1c | 0 | TT disturbed | DEVX reset | E1a pass, or E1a failed only on method A |
| E1F stage-0 hart initializer (only if A and B fail) | S2′ | 1 (L2CPU3) | hart hangs in the stub | DEVX reset | **explicit DP3 approval**; T-3.2b |
| E2 release L2CPU3 at hold start, **before** server_g1 builds; SELFTEST; RNMI re-arm test | S2 | 1 | hart hangs | RNMI park → reload; else ledger `burned`, job exits 3, DEVX reset (cheap: no TT session is up) | E1a/E1b pass (or E1F); T-3.2b; DP3; image CRC, handlers and reset vectors read back; PLL4 defaults |
| E2m LIM microbench (LOAD/CALL) | S2 | 0 | payload hang | RNMI park | E2 pass |
| E3 power: 10 legacy fd open/close, server_g1 start/stop/start | S2 | 0 | heartbeat stops | park/reload; else exit 3 | E2m done |
| E3(d), optional: all fds closed 10 s, then attach | end of S2 | 0 | host hang or frozen hart | DEVX reset | **user approval**; runs last |
| E4 kernel bench, E5a parity (payloads via LOAD/CALL) | S3 | 1 | payload hang | RNMI park | E3 pass |
| Closed loop, soak, demo | S4, S5 | 1 each | – | `rt_sim` falls back to ORT | T-3.E5a, T-3.5 pass |

**E1F stage-0 initializer** (contingency; spends the session's release). Reset vectors for harts 0–3 point at a
≤ 64 B stub in the GP scratch (0x2001_0100, executable via L1I, ✔ `Caches.md:18`). Hart 0 enables mstatus.VS, sets
`vsetvli e8,m2` with vl = 64, zero-fills 0x0800_0000–0x081D_FFFF with one 64 B `vse8.v` per line, writes a done word
to the last LIM line, and spins on a LIM entry word; harts 1–3 spin in wfi. The host then writes the monitor image,
writes the entry address, the hart executes `fence.i` and jumps to crt0; only after the monitor heartbeat appears
does the host overwrite the GP scratch with the park stub. The RNMI handler during stage 0 points at the stub entry
(a re-run only re-zeroes LIM). After E1F the reset vectors stay at 0x2001_0100. That is harmless: a reset vector is
consumed only at reset (✔ `X/tt-isa-documentation/BlackholeA0/L2CPUTile/README.md:46`), and the park stub never
loads it but jumps to crt0 at the fixed 0x0800_0000 (`lim.ld:63`; review:x280 F1). kmd allows exactly one GP-scratch
rewrite after a release: this stage-0 → park-stub swap, once the monitor is up (✔ `RT/x280/host/kmd.py`
`write_gp_scratch`).

**2026-10-25 (S1c, S1d, E1F):** host-written LIM is unusable (S1c's E1a/E1b: host NoC writes pre-read the SRAM, ≈ 94% of the pre-reads of never-written lines are uncorrectable and the CCACHE poisons the footprint; S1d: no host store heals it, and poison and contents survive `tt-smi -r 0`), and E1F passed (SELFTEST on every line, BEU 0, 0 traps), so the stage-0 initializer is no longer a contingency: every remaining M3 session (S2, S3, S4, S5) boots with it (`CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0 …"`, `X280_E1F_APPROVED=1`; boot steps 2–5 above become E1F's order, each session still spends exactly one release), E1F meets E2's precondition column, and R26 is resolved.

**Tile policy.**
- L2CPU3 (8,7), bit 7, is the default.
- L2CPU2 (8,5) is used only for the DP4 fp32-Walk option or after a root-caused L2CPU3 failure. Under DP4 both tiles
  are released by the same RMW (bits 6|7) inside one PLL excursion, never while another tile is serving.
- L2CPU0/1 are last resorts. If ≥ 2 tiles are burned in one session, stop and review with the user.

**Boot sequence** (what T-3.4 checks; U:x rec (1)):
1. Keeper fd open; wait ≥ 1 s; read PLL4 (`CNTL_1 == 0x0040_0002`, `CNTL_5 == 0x0101_0101`, `USE_POSTDIV & 0xF ==
   0xF`), `L2CPU_RESET == 0x0F`, WAYENABLE = 0, ENABLED_L2CPU bit 3, `TIMER_HEARTBEAT`.
   **2026-10-25 (S1c):** silicon reads `CNTL_1 = 0x0040_0102` before and after a chip reset (internal postdiv 1, which CMFW ignores, ✔ fw `pll.c:456-461`, and its clock driver sets on the keeper fd's power message, ✔ fw `tt_blackhole_smc.dtsi:296-298`; the bh_arc source `pll.c:226` writes 0), so x280d checks exactly `0x0040_0102` here, after boot and in WP-3.4 (`0x0040_0002` is dirty, unexplained), and steps 6 and 8 keep refdiv and the postdiv byte as read, moving only fbdiv and the CNTL_5 postdivs (200 MHz: `CNTL_1 = 0x0080_0102`; ✔ `RT/x280/regs.py` `PLL4_CNTL_1_DEFAULT`).
2. LIM init of every line by the method E1a proved (A by default).
3. Write the image (padded to 64 B) and read it back line by line (`kmd.write_image`, 16 UC words per call); the
   read-back CRC is what step 7 checks.
4. Write the GP-scratch park stub and RNMI-exception spin stub; set the RNMI handler of harts 0–3 to the park stub and
   the RNMI-exception handler to the spin stub (0xFFFF_F7FE_FFF1_0418 + 16h, +8); read back.
5. Write all four reset vectors (0xFFFF_F7FE_FFF1_0000 + 8h) = 0x0800_0000 and read them back.
6. Step PLL4 to 200 MHz in clock.py's order, one step at a time: postdivs 1 → 15, then fbdiv 64 → 128
   (✔ `X/tt-bh-linux/clock.py:14-30`). Before every write (and after the last) re-read CNTL_0, CNTL_1, CNTL_5 and
   PLL_LOCK; on any change (a CMFW relock) write nothing more, leave PLL4 to CMFW and exit 3 (✔ `kmd.pll4_step_to`).
7. One RMW: `L2CPU_RESET = read | 0x80` (`| 0x40` too under DP4), [3:0] == 0xF; read back. `release()` refuses
   before any write unless this session did steps 2–5: every LIM line initialised, and the image (its CRC), the GP
   scratch, the handlers and the reset vectors written and read back (✔ `kmd.boot_artefacts_missing`).
8. Back to 800 MHz with a new `800: [64, [1,1,1,1]]` entry in the same order: fbdiv 128 → 64, then postdivs 15 → 1;
   assert the default register values. The same re-read before every write as step 6; a change here is exit 3 with
   the ledger marking the tiles burned, since the harts already run.
9. L2PF: write 0x15811 and 0x38c84e at 0x0203_0000 + 0x2000·h.
10. Wait for magic and heartbeat; SELFTEST.

**PARK protocol.** Host sets bit h of 0x2001_0414 → the hart takes the RNMI (NMIE ← 0) into the park stub, which runs
`fence.i` and spins while the bit is set → the host waits for ≥ 1 ms of heartbeat silence, then clears the bit → the
stub jumps to crt0 at the fixed 0x0800_0000 (never through the reset vector), which increments `n_reentry`, re-arms
with `csrsi 0x353, 8` and returns to the idle loop → the host checks `n_reentry` incremented.

---

## 6. Critical path, parallel lanes, decision points

### 6.1 Lanes and schedule (one engineer plus agents; device sessions serialised on the one card)

| Week | Lane A: loop (5090) | Lane B: TT VLA | Lane C: X280 (CPU-only until S1c) | Lane D: device holds |
|---|---|---|---|---|
| 1 | WP-1.0–1.5 → M1.0 demo (day 3), **SP-1 → DP-S**, **SP-2**, **SP-7 → DP1** | WP-2.1, 2.3 (CPU), 2.2 once the obs dumps exist (≈ day 4) | WP-3.1 (**SP-6**), 3.2, 3.3 | – |
| 2 | WP-1.6–1.9; T-1.1…T-1.8 (demo-ready) | WP-2.4 | WP-3.2b, 3.4; T-3.1…T-3.4, T-3.2b | **S1** (4–6 h): T-2.2…T-2.4; **S1c** (≈ 2 h + reset): E0, E1a, E1b |
| 3 | T-1.9 / T-1.10 campaign (≈ 10 machine hours) → M1 exit | WP-2.6: T-2.5, T-2.6 | – | **S1b** (TT closed loop, ≈ 1 day); **S2** (≈ 3 h): E2, E2m, E3 → DP-X |
| 4 | WP-1.7 polish, HUD | T-2.7, T-2.8 → M2 exit | WP-3.7, 3.8 shadow, (3.8b) | **S3** (≈ 3 h): E4, E5a; **S4** (≈ 1 day): T-3.5…T-3.8 |
| 5 | – | WP-2.8 | T-3.9, WP-3.9 → M3 exit, demo | **S5** demo soak |
| 6+ | M4d (5090 part) | M4c / M4d TT | M4a → M4b | M4 sessions |

**Critical path** (≈ 5–5.5 weeks to M3 exit):
1. WP-1.1 → 1.2 → 1.5 → 1.8 (T-1.2, T-1.2b, T-1.3)
2. → 1.9 → 1.10 (T-1.9)
3. → WP-2.6 (T-2.6) → T-2.7
4. → WP-3.8 (T-3.7) → T-3.9

Lane C's CPU work (≈ 73 h) is finished by week 2, and S2/S3 by week 4, leaving about half a week of slack before
WP-3.8. E2m can move the kernel choice (and trigger WP-3.8b), so S2 is the earliest point at which M3's timing is
known.

**Effort:** M1 ≈ 102 h, M2 ≈ 55 h (+10 conditional), M3 ≈ 107 h (+9 conditional), M4 ≈ 130 h+.

**Device time:** S1 4–6 h, S1c ≈ 2 h, S1b ≈ 1 day, S2 ≈ 3 h, S3 ≈ 3 h, S4 ≈ 1 day, S5 as needed, plus ≈ 2 min of reset
after each X280 hold. While a session holds the flock, other users wait up to 7200 s (R29), so schedule long holds.

### 6.2 Decision points (defaults in bold)

| DP | Question | Default | Decide by |
|---|---|---|---|
| DP1 | Demo checkpoint for M1 exit, M2 and M3 | **cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc** (measured 5/11 and 8/20 vs nvidia 0/10 in this env; needs WP-2.1b). nvidia@897d0313 (BF16, no conversion) only if SP-7 shows it within 5 pts of cloudwalk. | end of week 1 (SP-7) |
| DP2 | Is "VLA + WBC from the GPU server" satisfied by the RT-owned GPU VLA `:5565` plus a separate GPU WBC process `:5556` (no restart of `:5555`)? | **yes** | M1 start |
| DP3 | X280 device policy: (a) may each X280 session spend L2CPU3's release; (b) may every X280 hold run under DEVX (chip reset after the hold) and may chip_session run `tt-smi -r 0` at session start when no one else holds the device; (c) is E3(d) allowed; (d) is the E1F stage-0 branch allowed if host LIM init fails? | **yes / yes / yes, last in S2 / ask when it arises** | before S1c (b) and S2 (a, c) |
| DP4 | X280 precision | **fp16 weights, both policies on L2CPU3**; fp32 = one policy per tile (L2CPU3 Balance, L2CPU2 Walk; both released in one excursion) | after T-3.6 |
| DP5 | Balance/Walk hysteresis (a behaviour change) | **off**; adopt only if flips exceed 1/s in > 10% of episodes and the user agrees | after T-1.8 |
| DP6 | TT code base for M2 | **MK snapshot** (validated 41 ms / PCC 0.99923); the live tree (where the LIBERO server ran 100/100, unchanged since) if T-2.1/T-2.2 fail on MK | after S1 |
| DP7 | Code home | **new git repo `~/experiments/gr00t/rtloop`** | WP-1.0 |
| DP8 | RTC scope | **v2a/v2b on the 5090 by runtime patch; the TT megakernel edit only if v2b passes T-4.4; no v2c fine-tune** | M4 start |
| DP9 | Is "WBC MLP on X280" (history and select on the host) acceptable as the M3 claim, with the full controller in M4a? | **yes**; the HUD labels it "WBC-MLP X280" until M4a | before M3 |
| DP10 | Statistics budget | **Decided (user, 2026-10-24): 50 episodes per arm for every comparison; no n = 350 tier** | closed |
| DP11 | M3 timing fallback if no kernel reaches 2 ms p99 | **accept `m3_rt_slow` (WBC p99 ≤ 9 ms, tick gates unchanged) and enable early dispatch**; below that (> 8 ms compute), keep X280 as shadow and show ORT/GPU as primary | after S2 (E2m) / S3 |
| DP-S | Async policy after the pilot | §3.4 rule | after SP-1 |
| DP-X | M3 go/no-go after S2 | Go if T-3.E2, T-3.E2m and T-3.E3 pass. If they fail twice: escalate BH-25/BH-28 behaviour to the TT firmware owners, and keep M3's software path on `--wbc x280-sim`/ORT behind the same contract, so the demo is not blocked. | after S2 |

### 6.3 Open questions carried into WPs

| # | Question | Settled by |
|---|---|---|
| (a) | Does the loop follow the official row-0 convention (T-1.3) and stamp k_obs correctly (T-1.2b)? Whether the model was *trained* with row 0 ↔ t_obs stays open: offline dataset check or VL ±1 offset ablation | T-1.2b, T-1.3; R13 |
| (b) | `viewer.sync` cost under load; GPU contention between EGL and two VLA instances | T-1.4 |
| (c) | Real G1 TT end-to-end latency (≈ 42 ms projected vs 56 ms for 2-camera LIBERO) and the cause of the 380 ms outliers; this sets d̂ and whether s can drop to 10 | T-2.3 |
| (d) | Can host writes initialise LIM ECC (methods A/B), or does it need the hart (E1F)? | T-3.E1a/b |
| (e) | Whether harts survive PLL4 relocks and clock-gated gaps | T-3.E3, E3(d) |
| (f) | Real LIM uncached load cost, L1I caching of LIM code, vlenb, Zvfh presence | T-3.E2, T-3.E2m, T-3.E4 |
| (g) | Tile layout of the TT `action_pred` buffer; ISA D7 = tt-metal bank 7 | WP-4.3 |
| (h) | ~~Whether the LIBERO `install()` table extensions exist on MK~~ **Closed:** MK has `TL.MATMUL_CONFIGS` (layers.py:627), `SDPA_CHUNKS` (:935), `FORBIDDEN_SDPA_CHUNKS` (:977); G1 reuses gr1's S_pad 128 geometry, so no rows are needed (review T-12). | – |
| (i) | How much discordance CRN removes between arms | SP-1b |

---

## 7. Review log

Source: `drafts/critiques.json` (control, tt, x280). IDs: C = control, T = tt, X = x280. "Re-checked" means I verified
the cited evidence myself before applying the fix.

| ID | Sev. | Issue (short) | Resolution | Where |
|---|---|---|---|---|
| C-1 | major | T-1.5 cannot show head-of-line blocking (VLA on :5565; 5 ms timeout caps RTT) | T-1.5 now sends **both** VLA and WBC to :5555 with `--wbc-timeout-ms 200 --wbc-no-latch`; gate ≥ 3% of calls > 20 ms or timeouts; T-1.4 asserts < 0.1% | T-1.5, `m1_negctl`, D15, R1 |
| C-2 | major | VL d_inj = 0 unrealisable (visibility checked before the send) | Visibility moved to step 3b after the trigger; "in flight" = not yet visible in VL; T-1.1 checks row = d_inj for d_inj ∈ {0,1,3}; RT d ≥ 1 stated | §1.2, §1.6, §3.1, T-1.1, D7 |
| C-3 | major | No end-to-end alignment test at d > 0 | New mock `ramp` mode and **T-1.2b** (bit-exact f32 check of anchoring and switch ticks); `state43_sent` and `q43_meas` logged | WP-1.3, WP-1.8, T-1.2b, §4.1, R13 |
| C-4 | major | Viewer relaunch lands in tick 0; onscreen deviation unlisted | Re-checked (`base.py:270-272, 478-484`). Viewer pre-launched in the reset window; deviations (f), (g); `viewer_us` at k = 0 gated; T-1.7 frame identity viewer on/off | §1.1, §1.2, WP-1.5, T-1.4, T-1.7, R9 |
| C-5 | major | Confirm protocol underpowered, inconsistent (C2 missing, unpaired Mann–Whitney, three must-pass cells, α-inflating +100 look) | D18: functional exits; one primary comparison per milestone (n = 350, one look, no extension); gate "not clearly inferior", NI reported as a claim; time-to-success as paired censored data with C2@5 in a secondary matrix; CRN via an RT-owned GPU VLA wrapper that also returns server timings. Power figures re-computed (normal approximation) | §4.4, D17, D18, T-1.9, T-2.7, DP10, WP-1.4, WP-1.9 |
| C-6 | major | SP-1 rule noise-dominated; "C3 ≥ C4 → debug" fires ≈ 55% by chance | Trigger dropped (T-1.2b is the alignment detector); bands on the lower 80% bound; C3 vs C4 only at d ∈ {8,12} with one-sided p < 0.05; MDE stated | §3.4, SP-1 |
| C-7 | minor | d_inj = 3 is the TT condition, not the 5090 | d_inj = 4 added and used for the 5090 (SP-1, T-1.8); 3 kept for TT | §1.6, SP-1, T-1.8 |
| C-8 | minor | Blend source is the stored old row, re-injecting pre-ramp nav | Blend from the last executed row frozen at the switch; hold/starve update it; unit test "starve then arrival, B = 5" | §3.1, T-1.1 |
| C-9 | minor | FAULT defined two ways | STARVE_FAULT (flag, continue, loop-attributable) vs SAFETY_FAULT (ends the episode, failure); scoring in §4.4 | §1.2, §3.1, §4.4, T-1.6 |
| C-10 | minor | CUDA guard vacuous | Env assertion after imports/resets, nvidia-smi compute-apps pid check at exit, frame identity | WP-1.0, WP-1.5, T-1.7, R2 |
| C-11 | minor | Process-wide CPU 2 pin puts library threads on the tick CPU | Tick thread alone on CPU 2 (sibling 18 idle); viewer/ZMQ/EGL threads on 3 via inheritance order and `ZMQ_THREAD_AFFINITY_CPU_ADD`; recorded | §1.1, D4, R10 |
| C-12 | minor | 1000 ms timeout longer than useful; s_eff clamp ill-defined; max-of-10 d̂ fragile | Give-up at 500 ms; `s_eff = max(1, min(s_cfg, 30 − d̂ − m))`; d̂ = p90 of last 10 (max logged) | §3.1, WP-1.3, T-1.1 |
| C-13 | minor | In-flight request at episode end blocks the REQ socket | Drain ≤ 500 ms in the reset window or re-create; counted; T-1.10 bound | §1.2, WP-1.3, T-1.10 |
| C-14 | minor | T-1.3 bit equality exposed to ORT options, leaking class patch, stray `GR00T_WBC_SERVER` | Per-arm subprocesses; `--ort-opts default` in T-1.3; `GR00T_WBC_SERVER` asserted unset; rerun rule | T-1.3, WP-1.0, WP-1.4, D5 |
| C-15 | minor | Logging gaps; SHAs do not pin dirty files | Re-checked `git status`. Added `state43_sent`, `obs_sha1`, `q43_meas/cmd`, `dq_arm_absmax`; memmap ring; `git status` + sha256 of dirty/untracked imports; `GearWbcPolicies` vendored | §4.1, WP-1.1, WP-1.4, WP-1.6, D3 |
| C-16 | minor | R13 overclaims what T-1.3 settles | Rephrased; training-semantics question kept open with an offline check or ±1 ablation | R13, §1.4.2, §6.3(a) |
| C-17 | minor | VL runs could mix ORT and GPU WBC numerics | VL matrices always `--wbc ort`, recorded | §1.6, §4.4 |
| C-18 | minor | C2 walks blind; SP-7 inconsistent with C0 and D15 | C2 = pause (nav ramp to 0) with optional C2b; SP-7 `--n_envs 1`; D15 lists both :5555 uses | §3.2, SP-7, D15 |
| T-1 | major | Fixed-noise gates need an endpoint the frozen contract lacks; in-process fallback blocked by CHIP_IN_USE | Re-checked (`local_chip.cpp:148-151`, `server_n16.py:174-182`). `predict_debug` behind `--debug-endpoints`, outside D1; T-2.2, T-2.4, bench and S1c checks use it; +2 h | §1.4.1, D1, WP-2.3, T-2.2, T-2.4, T-3.E0/E1b |
| T-2 | major | Boot gate stricter than acceptance; ratio gate fragile; golden unspecified | Boot sanity ≥ 0.995 vs bf16; T-2.2 primary vs bf16 (mean ≥ 0.999, min ≥ 0.998); fp32-relative secondary; per-group absolute floors | WP-2.3, T-2.2, R19 |
| T-3 | minor | T-2.1 parity cannot run in python_env | `make_isaac_ref.py` (srvpy) → `isaac_ref.npz`; +1 h | WP-2.2, T-2.1 |
| T-4 | minor | Ownership wording and reset ordering under-specified; R29 imprecise | Table says server_g1 owns Tensix/DRAM/CQs via UMD and shares the node; chip_session order probe → [reset → re-probe] → x280d → server_g1 → job; R29 corrected (7200 s flock; 1800 s only for orphans) | §1.1, WP-2.4, R29 |
| T-5 | minor | pyluwen evidence is 0.10.0, pin is 0.8.5; no cleanup binding; no ARC lock | x280d drops pyluwen: own ctypes KMD layer, `SET_NOC_CLEANUP` via ioctl, no ARC messages at all; E0 cross-check via `x280_peek`; T-3.E0/E3 record max TT latency per fd event | D12, WP-3.3, WP-3.4, R30 |
| T-6 | minor | M4c descriptor writer unassigned | server_g1 PUSHes `{addr, bank_table, chunk_seq, k_obs}` on :5572; x280d stays the single host writer of LIM | §1.4.4, §1.4.6, D15, WP-4.3 |
| T-7 | minor | `split_batch` costs ≈ 35 ms on stock requests; bind on all interfaces | Re-checked (`n16.py:259-266`, `server_n16.py:239`). Modality-key filter before splitting; `--host 127.0.0.1` | D10, §1.4.1, WP-2.3, T-2.1 |
| T-8 | minor | Device-time budget does not add up; T-2.7 misplaced | T-2.7 moved to WP-2.6/S1b; S1 booked 4–6 h (cold build); X280 no-release work in its own S1c; M2 ≈ 55 h | WP-2.5, WP-2.6, §6.1 |
| T-9 | minor | `live@<hash>` cannot identify the untracked live tree | Content sha256 stamp computed from the imported package | §1.4.1, R23 |
| T-10 | minor | Cloudwalk conversion location unspecified; disk 92% full | Private `HF_HUB_CACHE`, distinct `-bf16` repo name, never `~/.cache/huggingface`; ≥ 40 GB preflight | WP-2.1b |
| T-11 | minor | M2 VLA gate uses a GPU baseline; CPU pins overlap | `m2_rt` VLA p50 vs T-2.3; server_g1 8–11, `:5565` 12–15 or stopped; calls > 100 ms attributed | §1.1, §4.2, T-2.3, T-2.6, R21 |
| T-12 | minor | Open question (h) closeable now | Closed; `_extend_layer_tables` dropped; stamp from MK fields; MK import test kept | §6.3(h), WP-2.1, R20 |
| X-1 | **blocker** | LIM-ECC mitigation unbuildable with pyluwen UC u32 writes; cited source says host init is insufficient; E1 cannot see a fault; no fallback | Re-checked (luwen `lib.rs:412-429`, `pci.rs:441-520`; `test_hd_sockets.cpp:463-467`; `cluster.cpp:118-120`; KMD `mmap_offset_wc`). Three init methods (A WC 64 B NT stores via libwcio, B ttnn `write_to_core`, C hart-run stage-0 E1F with DP3 approval); E1 split and instrumented (readback, CCACHE ECC counters, cleanup test), plus hart-side SELFTEST read with BEU check at E2; T-3.4 checks the helper's transaction log; R26 quotes the full comment. Deviation from the critic's fix: pyluwen is dropped entirely, not kept for ≤ 4 B writes, since the own layer maps UC too | §1.4.4, D12, WP-3.2/3.3/3.4, WP-3.E1F, T-3.4, T-3.E1a/b, T-3.E2, §5.2, R26 |
| X-2 | major | RNMI park disables itself after first use; no trigger-clear handshake; wrong CSR numbers possible | Re-checked (`RNMIs.md:4-6, 15-34, 61`; `README.md:30`). PARK handshake (§5.2); crt0 re-arms with `csrsi 0x353, 8`; numeric CSRs, `mnret` as a word; check_image rejects 0x740–0x744; T-3.E2 hung-payload PARK→CALL→hang→PARK 10/10 | WP-3.2, T-3.2, T-3.E2, §5.2, R24 |
| X-3 | major | Timing assumes cacheable data; LIM is uncached; scalar v0 may take ≈ 15 ms; mailbox estimate ignores host costs | Re-checked (`Caches.md:13-31`). §1.3 estimates withdrawn and replaced; E2m microbench first; Zero-Device option E4c; early dispatch WP-3.8b (my addition: numerically identical thanks to the 1-tick lag); `m3_rt_slow` via DP11; host overhead benched in S1c; windows configured once (no per-call ioctl) | §1.3, §1.4.3, §1.4.4, WP-3.2, WP-3.7, WP-3.8b, T-3.E2m, T-3.E4, §4.2, DP11, R33 |
| X-4 | major | Recovery paths cannot run (WITH_DEVICE_RESET_AFTER read by the guard; Tensix-only probe; no --allow-reset) | Re-checked (`_device_guard.sh:33-44, 122-130`). Every X280 hold runs under DEVX; jobs exit 3 on dirty state; chip_session's L2CPU health check at start; "next hold" wording removed | D13, WP-2.4, §2 conventions, §5.2, R29 |
| X-5 | major | One-shot image never runs on a RISC-V model | Re-checked (`which spike qemu-riscv64` → none). WP-3.2b builds spike in user space; T-3.2b is a precondition of E2 | WP-3.2b, T-3.2b, R37 |
| X-6 | major | L0 not atomic; KMD cleanup write breaks one-writer-per-line and CRC | Re-checked (KMD `ioctl.h:332-360`). Mailbox v2: `seq_req` last word, separate UC write; CRC over body and obs; HOST_GONE in its own line L38 outside CRCs, cleared at attach; heartbeat hi-lo-hi; cleanup validated in E1a | §1.4.4, T-3.E1a, T-3.8, R36 |
| X-7 | minor | "postdiv 1" ambiguous; no 800 MHz entry; second tile needs a second excursion | Re-checked (fw `pll.c:225-236`, `clock.py:14-30`). Exact register values; 800 MHz entry in clock.py's order; DP4 tiles in one RMW and one excursion | WP-3.4, §5.2, DP4 |
| X-8 | minor | "Every fd open/close relocks PLL4" too broad; E3 might not exercise it | Re-checked (`chardev.c:954, 1017, 1074-1085`). Claim narrowed; E3 churns legacy fds and requires relock evidence | §4.1, R27, T-3.E3 |
| X-9 | minor | Telemetry interlock up to 100 ms stale; harvest bits unchecked | PLL4 read directly before each batch; ENABLED_L2CPU bit 3 in E0 and boot | WP-3.4, §5.2, R28 |
| X-10 | minor | pyluwen 0.8.5 behaviour inferred from 0.10.0 source | Resolved by T-5: pyluwen not used | D12, WP-3.3 |
| X-11 | minor | T-3.4's "zero writes to L2CPU_RESET[3:0]" cannot pass with an RMW | Exact assertion: one write = read \| 0x80 with [3:0] == 0xF, then read-back | T-3.4 |
| X-12 | minor | First LIM writes run alongside a live TT session | E1a without TT, then E1b with TT | §5.2, WP-3.5 |
| X-13 | minor | Weight blob counted biases as fp16 | 1,888,656 B, end ≈ 0x081D_E1D0 + header; T-3.2 checks the real blob | §1.4.4, T-3.2 |
| X-14 | minor | Epoch test cannot tell "reset" from "never released" | Reset after every X280 hold makes the ledger per hold; `TIMER_HEARTBEAT` recorded as cross-check | §5.2 |

**Refuted or unverifiable claims** from the reviews that are not covered by an issue above, and what replaced them:
- "viewer.sync p99 ≤ 3 ms", "max ≤ 30 ms with HUD, recording, viewer and 2 cameras", "d p99 ≤ 5 under EGL plus two
  VLA contexts": kept as pre-registered T-1.4 gates, marked unmeasured, with the R9/R10 fallback ladder.
- "TT G1 fidelity ≥ 0.999 vs fp32": replaced by gates against the bf16 reference (T-2).
- "Server total_ms p50 ≤ 45 ms, RTT p99 ≤ 55 ms": kept as projections with a recorded re-baseline rule (T-2.3).
- "Reset-per-session ≈ 1 minute": measured in S1c; budget 2 min.
- "X280 0.1–0.6 ms / 1.2–1.8 ms scalar" and "mailbox 5–25 µs": withdrawn; E2m and the S1c host bench replace them.
- "L1I caches LIM fetches" (ISA vs tt-bh-linux PMA): measured in E2m.
- "VLEN 512, 8 fp32 FMA/cycle": T-3.E2 records vlenb; kernels are VLEN-agnostic.
- "Harts survive PLL4 bypass/relock and gating": T-3.E3 and E3(d).
- "ISA D5/D6/D7 = tt-metal banks 5/6/7": only matters in M4c; WP-4.3 checks it against `get_dram_bank_table`.
- "Effect of a partial write to uninitialised LIM": not provoked deliberately; E1a's instrumentation and E2's hart-side
  read detect it.
- "tt-smi -s sends no power message": not relied on; no tt-smi during runs.
- "Bit-exact T-1.3 across ORT SessionOptions": avoided by identical options (C-14).
