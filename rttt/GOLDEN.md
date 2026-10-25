# G1 goldens (PLAN.md WP-2.2)

Reference outputs of the deployed GPU policy for the TT port of GR00T N1.6 on the Unitree G1. T-2.2 (fidelity)
and the server_g1 boot check compare against them, and T-2.1 (host encode/decode parity) uses `isaac_ref.npz`.

- Checkpoint: `cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc7660bdef4cf17ad18dc5dd97f13e4ce5` (DP1), from
  the local HF snapshot (shard sha256s are in `manifest.json`).
- Isaac-GR00T `5dc80c4a` (n1.6.1-release). torch 2.7.1+cu128, transformers 4.51.3, flash-attn 2.7.4.post1, RTX 5090.
- Output directory: `~/experiments/gr00t/golden/n16_g1/` (21 MB).

## Regenerate

Run the steps in this order (`source $RT/env.sh` in every shell). Step 3 uses the GPU and needs ≥ 20 GB free VRAM
(as step 1 did, through the GPU VLA server); the untouchable `:5555` server alone takes about 7.3 GB.
`check_served_bf16` also uses the GPU (≥ 10 GB).

```bash
# 1. Raw observations: a sync-mode loop against a W-VLA server that dumps every request it sends. The recorded set
#    was served by the M1 GPU VLA server (rtserve/gpu_vla_server.py, now only at the git tag gpu-baselines) on a free
#    port; another server (e.g. server_g1 in a device session) gives a new golden set.
P=40484   # that server's port (a free port outside 5555-5572)
nice -n 19 taskset -c 16,17,20-31 simpy -m rtloop.loop --mode sync --vla tcp://127.0.0.1:$P --wbc ort \
    --seeds 0-9 --crn --dump-vla-obs 1000 --tag wp22_golden_obs --runs-root $RT/runs/wp22
# 2. Pick the 20 observations -> obs/ + obs_index.json
simpy -m rttt.select_g1_obs --run $RT/runs/wp22/<stamp>_wp22_golden_obs
# 3. bf16 + fp32 goldens -> samples/ + manifest.json (about 1 min, peak 12.5 GB allocated)
nice -n 19 taskset -c 16,17,20-31 srvpy -m rttt.make_g1_golden
# 4. Isaac host reference -> isaac_ref.npz + isaac_ref.json (CPU only)
nice -n 19 taskset -c 16,17,20-31 srvpy -m rttt.make_isaac_ref
# 5. Checks: offline (numpy; works in simpy, srvpy and ttpy) and served-path identity (GPU, optional)
simpy -m rttt.check_golden
srvpy -m rttt.check_served_bf16
```

The source run is `$RT/runs/wp22/20261024_211739_wp22_golden_obs`: 10 episodes, seeds 0–9, `--crn`, 599 requests,
4/10 SUCCESS (seeds 1, 3, 8, 9) and 6 TRUNC. Steps 2–4 are deterministic: two full regenerations gave
byte-identical `obs/`, `samples/` and `isaac_ref.npz`. The JSON files differ only in their timestamps and the hashes
that follow from them. Step 1 was not repeated, so whether a second loop run reproduces the same observations is
unmeasured. Treat a new run as a new golden set.

## Which observations

`select_g1_obs.py` picks them:

- `s00`–`s09` (reset): the pre-roll request (k_obs = 0) of episodes 0–9, one per scene seed.
- `s10`–`s19` (mid): one request per episode, at a target fraction 0.1 … 0.9 of that episode's length
  (±15 %), chosen while the robot moves. The chunk answering the request either walks (the Walk policy on more
  than half its ticks and |nav| ≥ 0.15) or manipulates (arm command travel > 0.10 rad or a hand command change
  > 0.20 rad).

| sid | seed | k_obs / k_end | kind | sid | seed | k_obs / k_end | kind |
|---|---|---|---|---|---|---|---|
| s10 | 0 | 60 / 1439 | walk | s15 | 5 | 780 / 1439 | manip |
| s11 | 1 | 120 / 639 | manip | s16 | 6 | 920 / 1439 | walk |
| s12 | 2 | 400 / 1439 | manip (right-hand release, Δ 1.5 rad) | s17 | 7 | 1040 / 1439 | manip |
| s13 | 3 | 260 / 692 | manip | s18 | 8 | 1020 / 1269 | walk |
| s14 | 4 | 660 / 1439 | manip | s19 | 9 | 640 / 704 | walk (nav 0.99) |

All 20 observations share the instruction "pick up the apple, walk left and place the apple on the plate.", which
gives 117 prompt tokens and 81 image tokens.

## Formats

All arrays are little-endian, C-ordered `.npz` (np.savez). `rttt/golden_io.py` reads them with numpy only.

| file | content |
|---|---|
| `obs_index.json` | `samples[]`: sid, kind, source run/file/seq, ep, seed, k_obs, k_end, end, obs_sha1 (= `schema.obs_sha1`, also in the run's RequestLog), state43, motion scores, and the loop's CRN noise seed (not the golden noise). |
| `obs/<sid>.npz` | The W-VLA request exactly as sent: `video.ego_view` u8 (1,1,480,640,3), `state.<group>` f32 (1,1,D) for the 7 groups (43 dims), `annotation.human.task_description` 0-d str. `golden_io.load_obs()` returns the request dict (language as a 1-tuple). |
| `samples/<sid>.npz` | `noise` f32 (1,50,128); `noise_seed` i64; `bf16.noise_in` = bf16(noise) as f32; for `p` in bf16 and fp32: `p.action_pred_norm` f32 (1,50,128) and `p.action.<group>` f32 (1,30,D), the decoded W-VLA reply (arms absolute). |
| `manifest.json` | Checkpoint repo@sha and blob hashes, the definitions of both runs (weights, inputs, attention, noise), noise rule, `sample_keys` (shape/dtype), and per sample the sha256s, noise seed, checks and bf16-vs-fp32 statistics. `bf16_vs_fp32` holds the T-2.2 floors. Also VRAM, env and code hashes. |
| `isaac_ref.npz` | Stacked over the 20 samples in index order (`sid` (20,)). Encode: `in.input_ids` / `in.attention_mask` i64 (20,1,117), `in.pixel_values` f32 (20,1,1,3,252,252) (the collator returns a list of one (1,3,252,252) tensor), `in.image_sizes` i64 (20,1,2), `in.state` f32 (20,1,1,128) (the normalised, zero-padded state), `in.embodiment_id` i64 (20,1) = 8, and `text` (the chat-template string). Decode: `pred.uniform` f32 (20,1,50,128) = `default_rng(1000+i).uniform(-1.1, 1.1)` (9 % of the used values are outside [-1, 1], so the clip is exercised) and `dec.uniform.action.<group>` (20,1,30,D); `pred.golden_bf16` / `dec.golden_bf16.action.<group>` are the same for the bf16 golden prediction. |
| `isaac_ref.json` | Keys, shapes, dtypes, pred rule, sha256. |

Encode values are those from before Gr00tPolicy's bf16 cast (`gr00t_policy.py:326-338`). Decode is
`processor.decode_action(pred, unitree_g1, {k: state[None]})` (`gr00t_policy.py:347-352`).

## The two runs

- **bf16 (primary T-2.2 reference, = deployed):** Gr00tPolicy as a policy server builds it: `model.to(bfloat16)`,
  bf16 inputs, flash_attention_2 for the LLM and Siglip2, and the default diffusers SDPA for the DiT. Every sample
  equals `Gr00tSimPolicyWrapper.get_action` with the same noise bit for bit (pred and actions max|d| = 0). In a
  fresh process without torch's deterministic flags or the cuBLAS workspace setting (`check_served_bf16.py`), all
  20 are again bit-identical. Re-running a sample in the same process gives max|d| = 0.
- **fp32 (secondary reference):** the checkpoint at its stored precision. It has 517 F32 tensors (kept) and 589
  BF16 tensors (upcast). All 1106 were checked equal to the safetensors. Inputs are fp32, and attention is torch
  SDPA with the math backend (Siglip2 with the packed block-diagonal mask, as in `dump_golden_n16.py` `fp32_math`).
  Re-runs give max|d| = 0.
- **Noise:** `torch.randn((1,50,128), generator=torch.Generator('cpu').manual_seed(42 + i), dtype=float32)` for
  sample i. It is injected by replacing `torch.randn` for the single call at `gr00t_n1d6.py:310-314`. The bf16 run
  starts from `bf16(noise)` (round to nearest even, saved as `bf16.noise_in`). torch 2.7.1 and torch 2.11 (CPU,
  python_env) both regenerate the noise from its seed bit-exactly, but consumers should send the saved `noise`.

## Measured (GPU bf16 vs fp32, 20 samples)

`action_pred_valid` = `action_pred_norm[0, :30, :35]`.

| metric | min | mean | worst sample |
|---|---|---|---|
| PCC(bf16, fp32), action_pred_valid | 0.999288 | 0.999944 | s12 (all others ≥ 0.999960) |
| PCC(bf16, fp32), all decoded actions | 0.999782 | 0.999966 | s12 |
| max\|d\| action_pred_valid (normalised) | – | – | 0.2374 (s12, right_hand) |

Per group, decoded max|d| over the 20 samples. The T-2.2 floor is max(2 × max|d|, 0.02):

| group | max\|d\| | where (2nd) | T-2.2 floor | gated |
|---|---|---|---|---|
| left_arm (rad) | 0.00525 | s13 (s03 0.00508) | 0.0200 | yes |
| right_arm (rad) | 0.01024 | s13 (s14 0.00777) | 0.0205 | yes |
| left_hand (rad) | 0 | constant: min = max = 0 in statistics.json | 0.0200 | no |
| right_hand (rad) | 0.17724 | s12 (s18 0.00611) | 0.3545 | yes |
| waist (rad) | 0.00530 | s18 (s10 0.00443) | 0.0200 | yes |
| base_height_command (m) | 0 | constant: min = max = 0.74 | 0.0200 | no |
| navigate_command (m/s, rad/s) | 0.01589 | s19 (s12 0.01212) | 0.0318 | yes |

Notes for T-2.2:

- s12 sits on a right-hand release. Both runs drop the hand from 1.5 to 0 at row 12, but they disagree on the
  transition row 11 (dim 3: bf16 0.697 rad, fp32 0.519 rad). The right_hand floor (0.354 rad) comes from that one
  value. On the other 19 samples the right_hand max|d| is ≤ 0.0061 rad. Every other floor is the 0.02 minimum,
  except nav (0.032).
- The secondary gate needs, per sample, PCC(TT, fp32) ≥ PCC(GPU-bf16, fp32) − 0.001. Those per-sample values are
  `manifest.json` `samples[i].bf16_vs_fp32.pcc_valid`. The tightest is s12, where TT needs ≥ 0.998288.
- Read the floors with `golden_io.group_floors(manifest)`. Units are radians, except base_height (m) and nav
  (m/s for vx and vy, rad/s for wz).
- left_hand and base_height decode to constants whatever the prediction is, so T-2.2 rightly excludes them.

Not a latency measurement: the per-sample GPU time in the manifest (bf16 median 156 ms, fp32 186 ms) was taken at
nice 19 on non-tick CPUs while the GPU was shared with `:5555`. Peak allocation was 12.5 GB (fp32 phase).

## Consumers checked on the CPU (2026-10-24; no device)

- `ttpy -m pytest tests/tt/test_g1_host_cpu.py -k "isaac or no_ttnn"` passes (3 tests), using WP-2.1's g1 glue
  and this `isaac_ref.npz`. Host encode (input_ids, pixel_values, state, embodiment_id) and decode of both pred sets
  have max|d| = 0 on all 20 observations.
- The same parity holds for the bare MK host code (`common/preprocessing.encode` + `common/normalization.decode`,
  gr1 layout, HF shim, ttnn imports blocked).
- `RTTT_CPU_REF=1 ttpy -m pytest tests/tt/test_g1_host_cpu.py -k cpu_reference` passes. It runs the port's fp32 CPU
  reference on the BF16 copy with the golden noise on the boot samples, and compares against these goldens:
  PCC ≥ 0.9995 vs bf16, ≥ 0.9999 vs fp32, decoded max|d| ≤ 0.02. This cross-checks the goldens against an
  independent implementation.
