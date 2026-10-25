# Judge report: plans A, B and C for the async VLA + WBC demo (5090 → Tensix → X280)

Date 2026-10-24. Inputs: `research/understand.json` (read in full), `drafts/plan-{A,B,C}.md` (read in full), plus the
source checks listed in §3. Nothing in any repository was modified, the Blackhole was not opened, and the GR00T server on
:5555 (pid 3134179) was not touched (inspected with `ps` only).

## 1. Scores (1–10)

| Criterion (weight) | A: interfaces first | B: risk first | C: demo first |
|---|---|---|---|
| Correctness vs evidence (0.25) | 7 | **8** | 7 |
| Completeness vs rubric (0.15) | **9** | **9** | **9** |
| Feasibility / effort realism (0.20) | 5 | **8** | **8** |
| Risk handling (0.15) | 8 | **9** | 7 |
| Testability: numeric gates, runnable commands (0.15) | **9** | 8 | 8 |
| Demo value (0.10) | 6 | 6 | **10** |
| **Weighted total** | **7.25** | **8.10** | **7.95** |

**Skeleton chosen: B.** It has the best correctness and risk handling for the hard part (M3, where one mistake costs a
chip reset). C's demo ladder, HUD, existing-code reuse and seam choice are grafted onto it, and so are A's normative
WBC spec, conformance suites and M4 binary schemas. §4 lists every graft and fix.

## 2. Per-draft reasoning

### Plan A (interfaces first, five processes)

Strengths:
- The most exact contracts of the three: byte-offset message tables, joint-index tables, and a normative `LbStep`
  (stateful 72→15) specification that reproduces the 1-tick command lag.
- Conformance suites that every new backend must pass.
- A replay VLA server that rejects any request whose state differs from the recording, which proves observation
  equivalence as well as action equivalence.
- A reset-aware tile budget read from `L2CPU_RESET`.
- An evaluation matrix with machine-time estimates, a negative control (WBC on the shared REP socket), and a
  real-robot mapping table.

Flaws:
- **Over-built M1.** Five OS processes, PAIR lockstep with deadlines, a custom binary wire with a generated C header, a
  stateful WBC on the GPU server, and a slaved shadow history. That is a lot of new surface before the first async run.
  The research shows the GIL is not a problem (U:srv "Q2 GIL contention": wake-up p99 ≤ 0.33 ms), so a single process
  is enough. 94 h for this scope is optimistic.
- **Stateful WBC boundary from M1.** Fail-over then needs a slaved history copy. The stateless 516→15 boundary (B, C)
  matches the existing `gear_wbc_action` endpoint, so fail-over at any tick is trivial.
- **EGL crash.** A sets `CUDA_VISIBLE_DEVICES=""` for P1–P3 without `MUJOCO_EGL_DEVICE_ID`, which crashes robosuite's
  EGL init (§3 row 1).
- **flock deadlock.** "Restarts inside a session use `WITH_DEVICE_NO_PROBE=1`" would block on the flock the session
  already holds (§3 row 8).
- **Unneeded `StreamGraphPolicy`.** It is harmless but not needed in a dedicated single-threaded server.
- **Statistics.** A non-inferiority gate on the CI bound at n = 160 has power of only about 0.6 (§3 row 10).
- **Weak demo story.** No HUD and no visible demo per milestone.

### Plan B (risk first)

Strengths:
- Load-bearing claims are re-checked and marked ✔. It found the `MUJOCO_EGL_DEVICE_ID` requirement and showed that the
  stock `gear_wbc_action` rejects a `tick` keyword.
- The stateless WBC boundary is argued well: bit-checkable, fail-over without state loss, same endpoint as today.
- Week-1 risk spikes are the first slices of real WPs, each with a go/no-go.
- Two-step loop equivalence: own VL ≡ own sync, then own sync ≡ the official `MultiStepWrapper`. This is cheaper and
  stronger than A's cross-process bit test.
- Device resource table for each milestone.
- Release ledger with epoch detection.
- **E2 releases the hart at the start of a hold, before the TT server builds**, so a failed release costs a
  1-minute restart, not a TT session.
- Fault-injection test (T3.7).
- Correct analysis of the device guard: it busy-waits up to 1800 s and resets only when the probe fails and no one
  else holds the device.

Flaws:
- **Weak acceptance rule.** "Paired point estimate ≥ −10 pts" is not a non-inferiority test.
- **Misses the existing LIBERO TT server.** B writes the TT server from scratch (10 h) although a validated one
  exists (§3 row 4).
- **Optimistic eval time.** The VL matrix (11 arms × 160 in about 3 h) is plausible only because the VLA server is
  the bottleneck; that should be stated.
- **No demo affordances.** No HUD and no demo story.
- **One-session assumption.** The X280 plan does not say what happens to a released hart *between* device holds,
  after the keeper fd closes and the clock gates. The optional E3(d) is the only place this is touched.

### Plan C (demo first)

Strengths:
- A demo ladder with a visible result and a scripted verdict at every step.
- An honest HUD: the processor map comes from the backends' `info` stamps, not from the CLI. Live backend hot-swap
  and a "proof switch".
- **Found the validated LIBERO TT server** (`TT/libero/server_n16.py`). It speaks the PolicyServer wire, draws fresh
  seeded noise, has an `info` endpoint and a boot golden check, and ran 100/100 episodes with 0 errors. This corrects
  U:tt "Existing serving code".
- **Found that the nvidia G1 checkpoint is local and all-BF16**, so the TT port needs no conversion.
- The cleanest WBC seam: a class-level patch of `G1GearWbcPolicy.load_onnx_policy`.
- The "no future leak" rule: send o_k no earlier than T_k.
- A mock VLA with deterministic and stall modes, and splicer pseudocode.

Flaws:
- **Wrong claim about early termination.** C says stock episodes are "not terminated early on success
  (`terminate_on_success=False`)". The stock CLI sets it to True (§3 row 6).
- **LIBERO server location.** The server exists only in the live tree, not in the frozen snapshot MK. C's "clone it and
  run from MK" needs an import-compatibility check that C does not plan (§3 row 4).
- **Lenient gate.** `d p99 ≤ 12` is far looser than the measured 5090 latency supports (p99 70–80 ms → d ≤ 4–5).
- **Wrong fp32 claim.** "Exact fp32 forces DRAM weights" is false: one fp32 policy per tile fits in LIM (§3 row 11).
- **Only a statistical anchor.** C has no official-path equivalence test, only the C0 anchor "within CI".
- **Thinner X280 recovery.** No ledger or epoch detection, `SET_NOC_CLEANUP` is only a stretch goal, and the release
  happens after the TT server is READY, so a failed release also tears down the TT session.
- **Statistics.** It has the same n = 160 non-inferiority power problem as A.

## 3. Verification log (claims I checked in code)

| # | Claim (draft) | Verdict | Evidence |
|---|---|---|---|
| 1 | With `CUDA_VISIBLE_DEVICES=""`, robosuite EGL init crashes unless `MUJOCO_EGL_DEVICE_ID` is set (B, C) | ✔ B/C; ✗ A, which sets it for P1–P3 | `W/gr00t_wbc/dexmg/gr00trobosuite/robosuite/renderers/context/egl_context.py:39-55`: `""` is not None and not `isdigit()`, so `[int(x) for x in "".split(",")]` raises at :52 |
| 2 | The stock `gear_wbc_action` rejects extra kwargs such as `tick` (B, C) | ✔ | `R/gr00t/policy/server_client.py:139` calls `handler(**data)`; `R/gr00t/policy/gear_wbc_policy.py:107` defines `get_action(self, policy, obs)` |
| 3 | `CudaGraphPolicy` syncs the stream captured at construction, which is harmless when called from the constructing thread (B, C); A's `StreamGraphPolicy` is not needed | ✔ | `gear_wbc_policy.py:83` sets `self.stream = torch.cuda.current_stream(device)`; `:85-92` |
| 4 | A PolicyServer-compatible TT server already exists (C) | ✔, with a caveat | `TT/libero/server_n16.py:1-36` (wire), `:174-177` (fresh `torch.randn` from a seeded generator), `:142-171` (boot golden check). `libero_eval/n16/tt_spatial_r0_summary.json`: 100/100 episodes, 0 errors, server total p50 56.4 ms, **max 380 ms**. Caveat: `publish/worktree-mk/models/experimental/gr00t/` has **no `libero/`**. The run used the live tree (`bin/env.sh` sets `TT_METAL_HOME=.../tt-metal`; run 2026-09-28; no live-tree file is newer than the libero files) |
| 5 | The nvidia G1 checkpoint is local and BF16, with the same byte layout as base (C) | ✔ | `models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate/snapshots/897d0313…`: 1106 BF16 tensors, total_size 6,573,217,664. cloudwalk `2a5b61cc…`: 517 F32 + 589 BF16, 9,813,153,664. `statistics.json` and `config.json` are identical; `processor_config.json` / `embodiment_id.json` differ only by extra entries (`rl_info`, `oxe_droid`, `new_embodiment`), and unitree_g1 is slot 8 in both. The README's official result (58%) is for the nvidia checkpoint (`R/examples/GR00T-WholeBodyControl/README.md:9-13`) |
| 6 | Stock episodes do not end early on success (C) | ✗ | `R/gr00t/eval/rollout_policy.py:466` passes `terminate_on_success=True`; `multistep_wrapper.py:306` ends at the chunk boundary after any success. The success metric is unaffected; episode length and time-to-success are |
| 7 | The WBC seam is `G1GearWbcPolicy.load_onnx_policy` (C) | ✔ | `g1_gear_wbc_policy.py:31-36` (policy_1/2), `:56-73` (ORT or the `GR00T_WBC_SERVER` hook), `:257` (Balance/Walk switch); re-created on every reset (`n1_utils.py:29-33`) |
| 8 | Device guard behaviour (A, B) | B ✔; A ✗ | `bin/with-device.sh` holds `flock .p150a.lock`. `bin/_device_guard.sh`: `wait_for_free_device` waits up to `DEVICE_BUSY_TIMEOUT=1800` s and then refuses; it resets only if the probe fails **and** nobody holds the device; `WITH_DEVICE_NO_PROBE=1` skips the probe but **still takes the flock**, so a nested call inside a held session blocks (A's R16 mitigation). `WITH_DEVICE_RESET_AFTER=1` exists; no draft uses it |
| 9 | Harts can leave reset only once per chip reset (all) | ✔ | `…/tt-isa-documentation/BlackholeA0/L2CPUTile/README.md` "Reset" (clone 0610df6). Corollary none of the drafts states: **a chip reset restores the budget**, so a reset-per-session model is a valid fallback |
| 10 | n = 160 per arm supports a 10-point non-inferiority claim (A, B, C) | ✗ | 160 comes from a *superiority* calculation, 45% vs 30% (U:as measurements). For a paired one-sided 95% non-inferiority test with a 10-point margin, at true Δ = 0 and discordance 0.32–0.49, power is 0.56–0.72 at n = 160 and 0.73–0.88 at n = 250 (normal approximation, computed for this report) |
| 11 | Exact fp32 X280 weights force DRAM placement (C) | ✗ | LIM 1,966,080 B − 0x11000 (code + mailbox) = 1,896,448 B ≥ 1,882,312 B for one fp32 policy, so fp32 means one policy per tile (two releases), as B's D9 says |
| 12 | The client venv torch is CUDA-built but not initialised at import (relevant to all CUDA guards) | ✔ | client venv: torch 2.7.1+cu128, `torch.cuda.is_initialized()` False; ORT providers `['AzureExecutionProvider','CPUExecutionProvider']` |
| 13 | `~/.tenstorrent-venv` has pyluwen 0.8.5 but no pyzmq (C) | ✔ | site-packages listing. A's "pip install pyzmq into ~/.tenstorrent-venv" mutates the venv that tt-smi runs from; a separate venv is safer |
| 14 | The stock eval env is `gym.make(..., enable_waist=True, camera_names=[ego,tpp])` + `WholeBodyControlWrapper` (all) | ✔ | `rollout_policy.py:98-117` |
| 15 | A JointSafetyMonitor violation calls `sys.exit(1)` (all) | ✔ | `sync_env.py:516-528`; `joint_safety.py:451-454` |

## 4. What the synthesis takes and fixes

| From | Item | Where in PLAN.md |
|---|---|---|
| B | Skeleton: decisions, stateless 516→15 wire for M1–M3, single tick thread, risk spikes, resource table, ledger, two-step equivalence, fault injection, guard analysis | §0, §1, §2, §5 |
| C | Demo ladder, HUD and telemetry PUB, honest processor map, hot-swap, `load_onnx_policy` seam, no-future-leak rule, mock VLA modes, LIBERO TT server reuse, nvidia checkpoint default, splicer pseudocode | §0.2, §1.2, §1.4, §2, §3, §4.3 |
| A | Normative `LbStep` spec and LowState/LowCmd/LbInput schemas (M4), conformance suites, replay server with request-state check, negative control, reset-aware tile budget, real-robot mapping and fidelity deviations, eval timing estimates | §1.1, §1.4.6, §2, §5.2 |
| fix | EGL env (`MUJOCO_EGL_DEVICE_ID=0` together with `CUDA_VISIBLE_DEVICES=`), plus a test that the frames are identical | §1, T-1.7 |
| fix | Nested `with-device.sh` deadlock: the session script starts every child; DEV is never nested | §5.2 |
| fix | Non-inferiority power: confirmation runs n = 250 paired seeds, with an explicit inconclusive rule | §4.4 |
| fix | Episode end: first-success tick in every arm; equivalence compares up to the official end | §1.2, §4.4 |
| fix | Port the LIBERO server *code* into RT and run it against MK with an import-compatibility test; the live tree is the fallback (DP6) | WP-2.3, T-2.1 |
| fix | TT latency outliers (LIBERO max 380 ms): starvation gate as a bounded fraction, and root-cause any call > 200 ms | T-2.6 |
| fix | d gates tightened to the measured latency (5090 d p99 ≤ 5; TT ≤ 3) | T-1.4, T-2.3 |
| fix | fp32 X280 option = two tiles, not DRAM | DP4, R-X9 |
| new | X280 session model: the release happens inside one hold, before the TT build; "reset per session" is the default until reattach across holds is proven (E3d) | D13, §5.2 |
| new | Trigger keyed on the last *sent* request (no "skipped" log spam while a request is in flight); late-reject threshold made consistent (usable rows ≤ 4, i.e. d ≥ 26) | §3.1 |
| new | RTC v2a/v2b on the 5090 by a runtime patch in an RT-owned server, with no Isaac repo edit; only the TT megakernel change is a repo change | §3.3, WP-4.4 |
