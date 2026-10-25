# rtloop tests (M1, plus the HUD, M2 and M3 CPU suites)

How to run each test, what it checks (test ids are from `~/experiments/gr00t/docs/plan/vla-wbc-x280/PLAN.md`
§2), and the result of the last full run (RTX 5090, Ryzen 9 7950X). Every command assumes
`source ~/experiments/gr00t/rtloop/env.sh`, which defines `simpy`, `srvpy`, `ttpy`, `xpy`, `$RT`, `$R` and the ports.

**CPUs.** Background work (tests, builds, agents) stays off the real-time CPUs of §1.1 (2-4, 6, 7, 12-15) and their
SMT siblings (18-20, 22, 23, 28-31), and off 5/21 (x280d): `BG="nice -n 10 taskset -c 0,1,8-11,16,17,24-27"` below.
During an M2/M3 device session server_g1 owns 8-11 (siblings 24-27), which leaves 0, 1, 16, 17. Tests marked `rt`
start a spin-paced `--mode rt` loop and tests marked `gpu` start the GPU servers: deselect both (`-m 'not rt and not
gpu'`) while a real-time campaign or the GPU servers are serving.

## Running

| What | Command | Time |
|---|---|---|
| Everything, sim and GPU tests included | `simpy -m pytest $RT/tests -q` | about 10 min |
| M1 only (leaves out the HUD, M2 and M3 lanes in `tests/hud`, `tests/tt`, `tests/x280`) | `simpy -m pytest $RT/tests -q --ignore=$RT/tests/hud --ignore=$RT/tests/tt --ignore=$RT/tests/x280` | about 9 min |
| M1 + HUD next to a real-time campaign (no RT loop, no GPU server) | `$BG simpy -m pytest $RT/tests -q -m 'not rt and not gpu' --ignore=$RT/tests/tt --ignore=$RT/tests/x280` (unset `DISPLAY`: no windows) | about 6 min |
| The deferred RT and GPU tests (on an otherwise idle machine) | `simpy -m pytest $RT/tests -q -m 'rt or gpu'` | about 5 min |
| Fast tests only (no env build, no GPU) | `simpy -m pytest $RT/tests -q -m 'not sim and not gpu'` | about 35 s |
| One file | `simpy -m pytest $RT/tests/<file> -q`; add `-s` to see the `[measure]` / `REPORT` lines | |

Rules every test follows:
- Servers bind `127.0.0.1` on random free ports, never 5555-5572. Nothing touches the GR00T server on :5555.
- Every process a test starts is stopped in a `finally`.
- Tests that build the MuJoCo G1 env are marked `sim` (about 10 s per env build). Tests that need the RTX 5090 are
  marked `gpu`; they start `rtserve.gpu_vla_server` / `rtserve.wbc_gpu_server` under `srvpy` themselves. Tests that
  start a `--mode rt` loop are marked `rt`: all of `test_faults.py` (8) and `test_obs_schema.py` (5), and the three
  `rt_60ms` cases of `test_alignment.py` (its RT run is a separate fixture, so `-m 'not rt'` never starts it).
- A test opens a window only when `$DISPLAY` is usable (GLFW can open it): the T-1.7 viewer on/off check and two
  tests in `test_viewer.py`. Each window lives a few seconds and its process is stopped in a `finally`. Without a
  display these tests skip.

**Last run (2026-10-25 01:48, next to the T-1.10 campaign):** `$BG simpy -m pytest $RT/tests -q -m 'not rt and not
gpu' --ignore=$RT/tests/tt --ignore=$RT/tests/x280` with `DISPLAY` unset gave **690 passed, 3 skipped, 0 failed, 26
deselected** in 283 s: 570 of the 573 non-rt, non-gpu M1 tests (the 3 skips need a display: the T-1.7 viewer on/off
check and two `test_viewer.py` tests) and all 120 `tests/hud` tests. The run before it, with another lane's x280d-sim
loop on the same safe cores, failed once in `tests/hud/test_loop_control.py::test_real_loop_applies_operator_control`
(its VL run's m1_sim_smoke period gates: compute p99 31.8 ms under that load); alone it passed twice. Deselected, to
run on an idle machine
(`simpy -m pytest $RT/tests -q -m 'rt or gpu'`): the 16 `rt` tests (`test_faults.py` 8, `test_obs_schema.py` 5,
`test_alignment.py` rt_60ms 3) and the 10 `gpu` tests (`test_official_equivalence.py` 7, `test_wbc_backend.py`
3). The M1 suite now has 599 tests (575 + 22 for control, `--shadow`, parity_tol, `--spin`, `--noise cpu` and the
M2/M3 analyze_run keys, + 2 in `test_env_sh.py`).

**Full run before that:** `simpy -m pytest $RT/tests -q` gave **873 passed, 6 skipped, 0 failed** in 594 s. 574 of the 575
M1 tests pass; the T-1.7 viewer check skips. The other 5 skips are in `tests/tt/test_g1_fidelity.py`, which needs
a live `server_g1`. The run before it (834 passed, 14 skipped) also skipped the 8 tests in
`tests/hud/test_loop_control.py`: a `loop.py` config line had moved under their patch's context. The line is back,
the patch applies again, and those 8 now pass. That run predates the out-of-process viewer; the runs after it
are listed under "Viewer process" below.

## HUD, M2 and M3 suites (CPU only: no GPU server, no device)

| PLAN id / WP | Command | What it checks |
|---|---|---|
| WP-1.7 HUD, control, demo | `$BG simpy -m pytest $RT/tests/hud -q` (about 1 min; headless, random ports, one `sim` test) | `rtloop/control.py` parse/validate/apply and the PUSH/PULL wrappers; the loop's control boundary (swap, pause, reset, auto swap, cost) and the real loop in VL mode driven through its PUB and PULL (`test_real_loop_applies_operator_control`); `hud.py` model, PNG rendering and the HUD process against live, stalled, absent and flooding publishers; the Blackhole chip panel against synthetic x280d/server_g1 status streams (fields shown, malformed/absent/stalled streams, DP4, degraded, MK core map placement; the telemetry row labelled by its source and never showing numbers from a sim or a telemetry-None message, also through the HUD process; the L2CPU NoC0 rows without `x280.regs` equal to `regs.L2CPU_XY`); `demo.sh` dry-run plans (m1/m2/m3, `--x280-sim` on CPUs 5,21, `--wbc-spin`, `--shadow`; the GPU VLA offline on DP1's local snapshot, `--noise cpu` in m2/m3) and two live runs with a stub loop; `demo.sh m3` on the device boots with `--stage0` and refuses before the hold without `X280_E1F_APPROVED=1` or with a `CHIP_X280D_SERVE_ARGS` lacking `--stage0` (a stub `with-device.sh` under a temp HOME records what the guard would be given; nothing reaches the real one) |
| env.sh `ttpy` | `$BG simpy -m pytest $RT/tests/test_env_sh.py -q` (about 2 s; skips without `publish/worktree-mk-env.sh`) | `ttpy` exports `TT_METAL_RUNTIME_ROOT=$MK` only if unset (S1's live-tree kernel build) and `rttt/g1_n16.pin_runtime_root` accepts it and refuses the live tree, with ttnn blocked |
| T-2.1 CPU | `ttpy -m pytest $RT/tests/tt/test_g1_host_cpu.py $RT/tests/tt/test_tt_server_protocol.py -q` (75 tests, about 20 s: 74 pass, 1 skips unless `RTTT_CPU_REF=1`); `RTTT_CPU_REF=1` adds the fp32 CPU reference vs the G1 goldens (about 13 GB RAM, 12 threads, 20 s) | the gate before any S1 session: spec asserts, layout g1, host encode/decode vs `isaac_ref.npz`, server_g1 on a fake model (wire, key filter, bind, status stream, `pin_runtime_root`, `--noise cpu` pairing in one interpreter) |
| WP-2.4 sessions | `$BG simpy -m pytest $RT/tests/tt/test_chip_session_fake.py $RT/tests/tt/test_bench_g1_fake.py $RT/tests/tt/test_m2_checks.py -q` (90 + 24 + 31 tests, about 50 s; one test needs `strace`) | every chip_session branch in fake mode (x280d's `--cpus 5` / `--status-pub` placement, `--x280 boot:3`'s default `serve ... --stage0` with `X280_E1F_APPROVED=1` and its exit-2 refusal without it, a `CHIP_X280D_SERVE_ARGS` passed through as given and to the jobs, server_g1 under `taskset` with one-thread pools, `CHIP_SAFE_CPUS`), `jobs/_common.sh` with its servers stubbed, bench_g1, the S1b checks of `rttt/m2_checks.py`. `test_g1_fidelity.py` (T-2.2, 5) skips without `G1_ENDPOINT`; it runs in S1 through `jobs/t2_fidelity.sh` |
| WP-2.6 S1b jobs | `$BG simpy -m pytest $RT/tests/tt/test_s1b_jobs_fake.py -q` (21 tests, about 50 s; the T-2.5 sync smoke alone about 30 s; skips it without `publish/worktree-mk-env.sh`) | t2_sync / t2_soak / t2_soak_2h / t2_primary in chip_session's fake mode on random ports: T-2.5 end to end at smoke size (two `--mode sync` loops against server_g1 `--fake-model` and a GPU stand-in, same noise_seeds per request), the T-2.6 / T-2.8 `demo.sh m2 --in-session` plans and a stub-loop soak through the session gates, T-2.7 through `eval_paired --dry-run`, the noise-pairing and bench refusals, every job's `CHIP_SAFE_CPUS` refusal. Never `--mode rt`, no GPU server, no port 5555-5574 |
| D17 noise pairing across venvs | `$BG simpy -m pytest $RT/tests/tt/test_noise_cross_venv.py -q` (1 test, about 5 s; skips without the GPU-server venv or `publish/worktree-mk-env.sh`) | `gpu_vla_server --noise cpu`'s draw under srvpy (its torch, CUDA hidden, through `RtPolicyServer.get_action` around a stand-in action head) equals server_g1's (`G1Policy.predict` on the fake model, and `g1_n16.make_noise`) under ttpy (MK's torch, ttnn blocked), byte for byte in fp32 and after the bf16 cast, for `noise_seed(2000, 0..4)`, `noise_seed(1049, 71)`, 0, 1, 2**63, 2**64-1; `randn_calls` 1. torch keeps only a seed's low 32 bits (0 and 2**63 collide, on both sides) |
| T-3.1 kernel (x86) | `$BG simpy -m pytest $RT/tests/x280/test_kernel_x86.py -q` (about 4 s; builds into a pytest temp dir) | Balance/Walk parity vs ORT on 10k logged + 256 random obs: fp32 <= 5e-5, fp16 <= 1e-2. The logged obs are `$WBC_OBS516_NPZ` (default `$RT/runs/x280/20261024_211935_obs516_kernel/obs516.npz`; those tests skip without it). Regenerate: `simpy -m rtserve.mock_vla_server --mode detfn --port 0` and `simpy -m rtloop.loop --mode vl --arm v1 --d-inj 4 --wbc ort --seeds 0-6 --dump-obs516 --tag obs516_kernel --runs-root $RT/runs/x280 --vla <mock url>`, both under `$BG` |
| T-3.2 image | `make -C $RT/x280/fw x280 && simpy $RT/scripts/check_image.py $RT/x280/fw/build/monitor.elf` | image <= 48 KiB inside LIM, no undefined symbols, stub sizes, CSR rules, `crc32_tab` (mon_crc32's slicing-by-8 tables in `.rodata` equal zlib's), blob end <= 0x081E_0000, a fence.i before hart 0's first call in crt0 (`hart0_fence_i`, review F10) |
| M3 firmware (x86) | `$BG simpy -m pytest $RT/tests/x280/test_mailbox_x86.py $RT/tests/x280/test_check_image.py $RT/tests/x280/test_crc32.py -q` | the monitor's x86 build behind the v2 mailbox; `mailbox_layout.py` == `mailbox.h` + `mh.h` (`test_layout_matches_c`); harts 1-3 as threads of the x86 build (`mon_x86_secondary_start`): adoption, ping, missing and hung harts reported within the bounded waits, mh_bench on x86 (counts, sinks, refusals, background/collect, a worker trap), each hart's LIM writes confined to its own lines, hart 0's warm entry keeping them and a cold boot starting a new epoch; `check_image.py`'s `amo`, `mh_layout` and `mh_bench` rules; `test_crc32.py`: `x280/fw/crc32_tab.h` is `x280/tools/gen_crc32_tab.py`'s output, its tables (header, x86 build, linked `monitor.elf`) equal zlib, and `mon_crc32` in the table build and the `-DMON_CRC_BITWISE` build (`build/x86-crcbit`) equals `zlib.crc32` for every length 0..4096 from every alignment 0..7, chained, and in its mailbox users (L0, obs, LOAD, L37, BENCH act_crc32) |
| T-3.2b spike (**precondition of E2**) | `$BG simpy -m pytest $RT/tests/x280/test_spike.py -q` (about 35 s, 8 spike processes; needs `x280/sim/build_spike.sh`'s output and `$WBC_OBS516_NPZ`, else the INFER tests skip; `T32B_REPORT=<path>` writes per-INFER instruction counts, `T32B_CRC_REPORT=<path>` the bitwise vs table monitor overhead). Re-run with T-3.2 after any change under `x280/fw` | SELFTEST, LOAD/FENCE_I/CALL, INFER parity vs x86, trap dump, microbench and stage-0 stub on spike; the hart's `mon_crc32` against zlib (`SH crc` lines: every length 0..520, 2,112, 4,096 from each alignment, chained); a `SPIKE_CRC_BITWISE=1` build whose INFER records, BENCH act_crc32 and blob LOAD equal the table build's (monitor overhead per INFER 124,305 bitwise, 13,552 slicing-by-8). Harts 1-3 (the 4-hart bench, S3b): the default script on spike -p4 byte-identical to -p1 (lim_only, x280_regions); the `SPIKE_MH=1` script on -p4 (adoption by the emulated RNMI: stat lines IDLE, mncause 2^63+2, WFI mode; ping done 0xE; mh_bench K1-K12 x P0-P3 x {1,3,F,E}, the P4 sweep and K3 under K6 with every result line's counts and sink checked against `mailbox_layout.mhb_sink`; a background job; a worker trap on hart 2 dumped in its aux line, hart 0's L1/L36 untouched); the one-writer audit (`run_spike audit_stores`, `-l --log-commits`: each hart stores only into its own lines, hart 0 only the ctl line of the MH region, the only shared line the declared LOAD handover of a result line, no AMO/LR/SC commits); no A-extension instruction in the image or any payload |
| T-3.2b tool | `$BG simpy -m pytest $RT/tests/x280/test_spike_tool.py -q` | `x280/sim/run_spike.py` itself (`$SPIKE`, `$RUN_SPIKE` in env.sh), its commit-log store audit (`audit_stores`) on two harts and its parser |
| WP-3.3 host layer | `$BG xpy -m pytest $RT/tests/x280/test_kmd_fake.py -q` (also under simpy; builds `x280/host/libwcio.so` with clang-20 if missing) | kmd against a fake device: ioctls, windows, allowlist, LIM init and boot rules; `write_boot_vectors(harts=4)` (only with `X280_MH_APPROVED=1`, before release, hart 0 kept on the park stub, the default write sequence unchanged), per-hart boot artefacts, RNMI bits 1-3 refused without this session's secondary vectors, the E1F 'stub' state and L36.mh's multi-hart bit, and never with bit 0; BEU of harts 1-3 only after release; S1d's store loops by disassembly (2 x 32 B `vmovdqu ymm` per line, 64-bit `movq` UC stores), the LIM_INIT_WC32 / LIM_INIT_UC8 entries (LIM and ctx lim_init only, pre-release only, refusals write nothing), every `lim_init` store's libwcio transactions; `FakeEcc` (off unless asked; 'poison' reproduces S1c E1a's fix/fail events per line, last event at the last line + 0x20 and 0.935 fail events per re-read; 'clean' / 'granule'); the PLL4 walk's policy-then-reads order with CNTL_0 and PLL_LOCK last, a relock starting between its reads, USE_POSTDIV re-read; `regs.pll4_mhz` against a transcription of fw `pll.c`'s GetExtPostdiv |
| WP-3.4 x280d (sim) | `$BG xpy -m pytest $RT/tests/x280/test_x280d_sim.py -q` (about 11 s; the rt_sim-client test needs onnxruntime, so it runs only under simpy) | x280d serve/admin/boot/park/attach/dump on the mailbox sim; a device `boot` / `serve --boot` (method A) refused without `X280_ALLOW_METHOD_A=1` (exit 2, before any Session; the message names `--stage0`), `Session.boot()` refusing the same on a non-sim session before any op, the sim and `--dry-run` booting as before; `--cpus` / `--sim-cpus` taskset lists and their `CHIP_SAFE_CPUS` refusal (exit 2, including `boot --dry-run --sim-cpus 8-9` inside and outside the safe set); boot / E1F / probe refused at the start when PLL4 is not walk-ready; `lim-ecc` (S1d) on `--sim-ecc` poison / clean / granule (regions, probes, the heal item, writes only inside its regions in ctx lim_init, refusals before any write, never twice on the same lines, never with a live or released hart); lim-init / lim-check ECC reporting with the gate unchanged; `bench --rtt`: `checks` = PLAN T-3.5's x280d-side gates alone decide `pass`, the client RTT is `reported`, `--cpus` / `--spin` place the client (tick thread on the lowest CPU, zmq I/O on the highest) and are refused without `--rtt` or outside `CHIP_SAFE_CPUS`, the device's host-overhead row fails without data; attach and PARK after a stage-0 boot without a release (T-3.8, E3(d)); T-3.8's kill -9 -> `serve --attach` -> `park --hold-s 2` sequence in one process with hart 0's L1I line of the GP scratch modelled (review F10: a stale stage-0 line re-runs stage 0, L1/L36 all 0, as on silicon in S4; crt0's entry fence.i and step E's FENCE_I drop it); `powertest --watch-s` ends with a PLL4 read; `--harts 4` (refused without `--stage0`, without `X280_MH_APPROVED=1` or off L2CPU3; its stage-0 dry-run differs from the default only in harts 1-3's six RNMI vector words), `harts adopt` + `bench --mh` (quick plan) + status on the sim and through a serving daemon, the refusal without the secondary vectors (and the sim stranding harts 1-3 when the trigger bits are forced on the old vectors), a hung secondary re-adopted by its own RNMI bit |
| M3 session jobs | `$BG simpy -m pytest $RT/tests/x280/test_m3_jobs.py -q` (60 tests, about 35 s, the two job smokes about 14 s each; under xpy the PyYAML / onnxruntime tests skip) | `x280/tools/m3_jobs.py` (T-3.7 gate sets from gates.yaml and the T-2.3 span, the t36 / t37 summaries, S1c's S2 GO/NO-GO and its `ecc` block on a copy of the recorded S1c session, S1d's `s1d-summary` verdict both ways); `jobs/m3_eval.sh` and `jobs/m3_shadow.sh` end to end in `--mode sync` against a mock VLA and `x280d serve --backend sim` (these two create a MuJoCo EGL context: safe cores only); `jobs/s1d.sh` end to end against `x280d serve --backend sim --no-release --sim-ecc poison|granule`; every M3 rt job's `CHIP_SAFE_CPUS` refusal (run with HOME on an empty dir, so a missed refusal still cannot start rt_sim); `x280_reattach`'s x280d command line (`--cpus 5`, `--status-pub`) and its arguments from `CHIP_X280D_SERVE_ARGS` (kept flags, `--stage0` -> `--attach` like `--boot`, refusals that start nothing); `x280_boot_flag`, `x280_rtt_client_args` (default 2,3, '' unpinned, dropped outside `CHIP_SAFE_CPUS`, `--spin`); every remaining M3 job's header names the stage-0 session and none checks `--boot` or runs lim-init; T-3.9's `t39-gates`, `swap_da` (swaps from wbc_be, the ORT shadow of the remote side, fallbacks and episode ends not swaps, missing shadow = no data) and `t39-summary` (traps, x280d counts, PLL4 reads and their gaps, loop rc); T-3.8's `t38-summary` (tick period inside an episode, the X280 serving again after each case's fallback, the PARK step; the recorded S4 session) |
| M3 sessions on the sim, stage 0 | `$BG simpy -m pytest $RT/tests/x280/test_m3_sessions_fake.py -q` (7 tests, about 4 min; skips without the firmware build) | chip_session.sh in its fake mode with a real `x280d --backend sim` booted by `--stage0` (`X280_E1F_APPROVED=1` for the FakeDevice only) and the stubbed tt-smi / tt_probe / fuser: the session boots by E1F's steps (no host LIM init) and refuses without the approval; `s2_e2_e3.sh` (E2, E2m, E3 with the watch's PLL4 reads; the sim's mtime/heartbeat timing rows recorded, not met; E3(d) restarts x280d as `serve --attach` from the stage-0 arguments, refused on the sim because its chip died with the first x280d); `e4_bench.sh` + `e5_parity.sh` + `m3_rtt.sh` in one `--vla tt` hold (the RTT client on the asked CPUs, T-3.5's gates vs the reported client RTT); `m3_shadow.sh` + `m3_eval.sh` (sync smokes, MuJoCo EGL: safe cores only) + `m3_faults.sh` (tests/x280/m3_stub_loop.py as rt_sim, `M3_LOOP_MODULE`; case 1's `serve --attach` refused on the sim, exit 3) in one hold; `m3_soak.sh` (T-3.9: demo.sh m3 --in-session with the stub loop, swaps, then the t39 gates and summary; and its failure on a fallback and a 2e-2 swap jump); `demo_m3.sh`; `s3b_mh.sh` on a stage-0 session with `--harts 4` (adopt, the quick bench, `s3b_mh.json` with the verdict) and its refusals (no `--harts 4`: exit 2 before any RNMI; no `X280_MH_APPROVED=1`: x280d refuses to serve). Never `--mode rt`, never CPU 5, random ports |
| S1d (device; after S1c) | `$DEVX $RT/scripts/chip_session.sh --x280 probe --vla none --job $RT/jobs/s1d.sh --tag s1d` (no release, no TT server) | E0, then `x280d lim-ecc`: each LIM init store (a64, a64nf, wc2x32, uc8, a64+uc8) on its own never-written region, the CCACHE ECC counters between phases and per access on probe lines, then E0 again; `s1d_summary.json` and the job log name the stores that leave dat_fail_count unchanged across a full re-read and the lim-init recommendation (none clean: S2 stays NO-GO, E1F needs DP3(d)). Exit 0 = the measurement is complete |
| M3 device holds after E1F (2026-10-25) | Every hold: `X280_E1F_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0[ ...]" $DEVX $RT/scripts/chip_session.sh --x280 boot:3 ...` (the stage-0 boot; one release and one reset per hold). A: `--vla none --job s2_e2_e3.sh --tag s2` (E3(d) last). B: `--vla tt --job e4_bench.sh --job e5_parity.sh --job m3_rtt.sh --tag s3` with `X280_PARITY_OBS=$RT/runs/x280/20261024_211935_obs516_kernel/obs516.npz`. C: `--vla tt --job m3_shadow.sh --job m3_eval.sh --job m3_faults.sh --tag s4` with `M3_T23_BENCH=$RT/runs/sessions/20261025_031915_1803710_s1b_t23/t23_bench.json T2_START_GPU=1 DISPLAY=:1`. D: `--vla tt --job m3_soak.sh --job demo_m3.sh --tag t39` with `--status-pub $X280_STATUS` in the serve args, `CHIP_VLA_STATUS_PUB=$TT_STATUS`, `DISPLAY=:1` and `DEMO_ARGS="--hud --record ..."` | A: `s2_summary.json` pass (DP-X). B: `e4_choice.json` (tier, m3_rt), `e5_summary.json` pass, `t35_summary.json` pass (PLAN T-3.5 gates, x280d side). C: `t36_summary.json`, the m3_rt verdict + `t37_summary.json` + the screen, `t38_summary.json`. D: `t39_summary.json` and the t39 verdict; the swap demo and the proof switch (`xpy $RT/x280/x280d.py park --via tcp://127.0.0.1:5558 --hold-s 2`) by eye. B only after A's GO; C only after B's E5a and T-3.5 pass (§5.2) |
| S3b (device; after S3, the user's approval 2026-10-25) | `X280_E1F_APPROVED=1 X280_MH_APPROVED=1 CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --stage0 --harts 4" $DEVX $RT/scripts/chip_session.sh --x280 boot:3 --vla none --job $RT/jobs/s3b_mh.sh --tag s3b` (one hold, one release, L2CPU3 only) | harts 1-3 adopted once (stat lines, mnepc = stage 0's 0x2001_013A, ping), `bench --mh` (masks x K1-K8 x P0-P3, P4 sweep, K3 under K6, K9/K10/K12, 3 repeats; BEU of harts 0-3 and CCACHE ECC before / every 16 CALLs / after, any event aborts; hart-0 BENCH v2 x200 idle and loaded; parity spot check); `s3b_mh.json`: S_N per kernel/pattern and `split_worth_it` (S_4(K9) >= 2.0 or compute-bound). Phase 2 (the MLP split) only if it says so; a 16-hart version only after the user has seen it |
| T-3.4 dry-run review | `xpy $RT/x280/x280d.py boot --tile 3 --dry-run > $RT/runs/dryrun_boot.txt && simpy -m pytest $RT/tests/x280/test_x280d_sim.py $RT/tests/x280/test_kmd_fake.py -q`; the stage-0 boot: `X280_E1F_APPROVED=1 xpy $RT/x280/x280d.py boot --tile 3 --stage0 --dry-run > $RT/runs/dryrun_stage0.txt` | the op list equals §5.2, one L2CPU_RESET RMW, LIM init transactions, allowlist; stage 0: no host LIM init, the image written after the done word |
| T-3.3 software path, no HW | `simpy -m rtserve.mock_vla_server --port 0 --mode detfn --latency-ms 60 --jitter-ms 5`; `xpy $RT/x280/x280d.py serve --backend sim --port P --sim-cpus 21` (fp16; DP4 fp32: `--dtype fp32 --tile 3,2` and two hart CPUs); `simpy -m rtloop.loop --mode rt --arm v1 --vla <mock> --wbc tcp://127.0.0.1:P --duration 120s --tag t33 --runs-root $RT/runs/x280`; `analyze_run.py <run> --gates m3_sim`. Or `demo.sh m3 --x280-sim --duration 120 --tag t33` | an `rt` run: on an idle machine only. All m3_sim gates (T-1.4's rows, WBC RTT p99 <= 2 ms, starvation as m2: <= 0.1% and no run over 10 ticks); ZmqWbc now accepts fp16 (x280d's parity_tol 1e-2) |
| Reviews | `ttpy -m pytest $RT/tests/tt/review_m2_server.py -q` (4); `$BG simpy -m pytest $RT/tests/tt/review_m2_checks.py $RT/tests/tt/review_m2_session.py -q` (3); `$BG simpy -m pytest $RT/tests/x280/review_x280_safety.py -q` (12) | adversarial-review tests; named `review_*`, so the default collection skips them: run them by path |

**X280 pins.** Every `tests/x280` test and X280 build pins to `X280_CPUS` (default `0,1,8-11,16,17,24-27`: off the
real-time cores and their SMT siblings) at nice 19: the test files, `x280/fw/Makefile` (`FW_PIN`) and
`x280/sim/build_spike.sh`; `x280/sim/run_spike.py` uses `RT_SPIKE_CPUS` (same default, intersected with the caller's
mask). This replaces the earlier `16,17,20-31`, which included 20, 22, 23 and 28-31, siblings of real-time cores.
`tests/x280` collects 339 tests (2026-10-25, 11:20, table CRC): kmd fake 85, x280d sim 76, m3_jobs 60, M3
sessions on the sim 7, mailbox 32, crc32 9, check_image 5, kernel 20, spike 19, spike tool 26. `tests/tt` under simpy
collects 174 (chip_session fake 92, bench_g1 fake 24,
m2_checks 31, S1b jobs 21, fidelity 5, cross-venv noise 1; the two T-2.1 files are ttpy-only and 75 more).

**Last run (2026-10-25 13:10-13:40, the 4-hart bench build: mh.c, mh_bench, `--harts 4`; build_id 3277753451, monitor.bin
28,444 B, mh_bench.bin 2,944 B; `taskset -c 0,1,16,17,24-27 nice -n 10`, `X280_CPUS` / `RT_SPIKE_CPUS` the same list,
next to the S3 hold):** `tests/x280` under simpy 387 passed; `review_x280_safety.py` 17 passed; xpy (`test_x280d_sim.py`,
`test_kmd_fake.py`, `test_m3_jobs.py`, `test_crc32.py`) 239 passed, 6 skipped; `tests/tt` (`test_chip_session_fake.py`,
`test_bench_g1_fake.py`, `test_m2_checks.py`) 147 passed; `tests/hud` 121 passed; T-3.2 `check_image.py` 0 FAIL, 57 PASS;
T-3.4 `runs/dryrun_boot.txt` / `runs/dryrun_stage0.txt` regenerated (445 image lines): every write op and step label equal
to the previous build's dry-runs apart from the image's line count; with `--harts 4` the stage-0 dry-run differs only in
harts 1-3's six RNMI vector words.

**Earlier run (2026-10-25 10:48-11:20, `taskset -c 0,1,16,17,24-27 nice -n 10` with `X280_CPUS` / `RT_SPIKE_CPUS` set to
the same list, next to hold A; slicing-by-8 `mon_crc32`, build_id 246828902, monitor.bin 23,780 B):** `tests/x280`
under simpy 339 passed; `review_x280_safety.py` 12 passed; xpy (`test_x280d_sim.py`, `test_kmd_fake.py`,
`test_m3_jobs.py`, `test_crc32.py`) 224 passed, 6 skipped; `tests/tt` (`test_chip_session_fake.py`,
`test_bench_g1_fake.py`, `test_m2_checks.py`) 147 passed; `tests/hud/test_demo_scripts.py` 22 passed; T-3.2
`check_image.py` 0 FAIL, 42 PASS; T-3.4 `runs/dryrun_boot.txt` and the stage-0 `runs/dryrun_stage0.txt` regenerated
(372 image lines). T-3.2b monitor overhead per INFER 124,305 -> 13,552 spike instructions, blob LOAD 184.2 M -> 86.7 M.

**Earlier run (2026-10-25 10:07-10:25, `$BG`, the stage-0 session work; no firmware change):** `tests/x280` under simpy
324 passed; `review_x280_safety.py` 12 passed; xpy
(`test_x280d_sim.py`, `test_kmd_fake.py`, `test_m3_jobs.py`) 214 passed, 6 skipped (no onnxruntime / PyYAML in
`.venv-x280`); `tests/tt` (`test_chip_session_fake.py`, `test_bench_g1_fake.py`, `test_m2_checks.py`,
`test_s1b_jobs_fake.py`) 166 passed; `tests/hud` 121 passed.

**Earlier run of the X280 suites (2026-10-25 06:40, `$BG`, S1d build):** `tests/x280` under simpy 298 passed;
`review_x280_safety.py` 12 passed; xpy (`test_x280d_sim.py`, `test_kmd_fake.py`, `test_m3_jobs.py`) 196 passed, 5
skipped (no onnxruntime / PyYAML in `.venv-x280`); T-3.2 `check_image.py` 0 FAIL, 41 PASS; T-3.4 `runs/dryrun_boot.txt`
regenerated (the same 307 writes as before: one L2CPU_RESET RMW 0x0F -> 0x8F, the same 240 PLL4 writes, now each
preceded by CNTL_1, CNTL_5, USE_POSTDIV, CNTL_0, PLL_LOCK). `tests/tt` was not re-run for S1d.

**Earlier full run (2026-10-25 02:55, `$BG`, next to the S1b session):** `tests/x280` under simpy 235 passed;
`review_x280_safety.py` 12 passed; xpy (`test_x280d_sim.py`, `test_kmd_fake.py`, `test_m3_jobs.py`) 133 passed, 5
skipped (no onnxruntime / PyYAML in `.venv-x280`); `tests/tt` under simpy 165 passed, 5 skipped (T-2.2 needs a live
server_g1); T-2.1 under ttpy 74 passed, 1 skipped (`RTTT_CPU_REF`); `review_m2_server.py` 4 and `review_m2_checks.py`
+ `review_m2_session.py` 3 passed; `tests/test_env_sh.py` + `tests/hud/test_demo_scripts.py` 23 passed.
`tests/x280/review_x280_safety.py` (12 tests, 12/12 in the X280 fix lane) is a `review_*` file, so run it by path as a
regression: `$BG simpy -m pytest $RT/tests/x280/review_x280_safety.py -q`.

## Tests by PLAN id

| PLAN id | File (tests) | What it checks | Last result |
|---|---|---|---|
| T-1.1 units | `test_schema.py` (11, 2 sim) | A35/state43 layouts, record dtypes and sizes, wire codec round trips, the 9-key VLA observation; joint-group indices against the live G1 robot model | pass |
| T-1.1 units | `test_splicer.py` (326) | The §3.1 splicer table: d in [0, 28], s in {10, 16, 20}, B in {0, 3, 5}; late reject at d >= 26; gen drop; starvation hold with a 5-tick nav ramp; STARVE_FAULT at 50 starved ticks; VL row at k_obs + d_inj equals d_inj; blend after starvation; s_eff never < 1; d̂ = p90 of the last 10 | pass |
| T-1.1 units | `test_vla_client.py` (52) | REQ client: round trips per mock mode, give-up within 500 + 50 ms with a lazy-pirate re-create, drain at episode end, payload <= 1.0 MB, record-proxy / replay with bit-exact state checks; the server skips a request its client already abandoned | pass; give-up at 501.9 ms (stalled server) and 502.7 ms (dead server); the live request after two give-ups is served in 300 ms (would be at least 540 ms if the abandoned one were also served) |
| T-1.1 units | `test_wbc_backend.py` (28, 3 gpu, 3 sim) | ORT `single` vs `default`; numpy float64 oracle vs ORT; ZmqWbc fallback on the same call, latch, reconnect; the server's parity_tol (capped at 1e-2); the fallback-free shadow client and `--wbc ort --shadow URL` (unit and through rt_sim in VL mode); wbc_gpu_server `--spin` (off by default, busy-polls when on; fake handler, no CUDA); gpu_vla_server `--noise cpu` draws server_g1's noise; the D5 patch in the real env is bit-identical; GPU WBC server parity and RTT; gpu_vla_server boot | pass; ORT single vs default max \|Δa\| = 0 (1000 random and 200 logged obs516); numpy vs ORT <= 1.7e-5; silent-server fallback 5.10-5.17 ms with a 5 ms timeout; GPU vs ORT <= 1.6e-5; W-WBC RTT paced at 50 Hz p50 0.508 / p99 0.777 ms; gpu_vla_server infer p50 59.5 ms |
| T-1.2 VL ≡ own sync | `test_vl_equivalence.py` (3, sim) | mock `detfn`, ORT WBC, seeds 0-4, 1440 ticks: `--mode vl --arm v1 --s 20 --d-inj 0 --B 0` against `--mode sync` | pass; 7200 ticks, max \|Δq43\| = 0.0; all 5 episodes TRUNC at k = 1439 in both; 3200 Walk ticks |
| T-1.2b alignment | `test_alignment.py` (12, sim) | mock `ramp`: VL at d_inj 1/3/8 and RT at 60 ± 5 ms; `cmd_arm == state_sent + (k − k_obs)·δ` bit-exact in float32 on every tick; `state43_sent(k_obs) == q43_meas(k_obs − 1)`; `k − k_obs` at each switch equals d | pass; 800 ticks, 48 requests and 50 switches per run; RT d {3: 15, 4: 33} |
| T-1.3 own sync ≡ official | `test_official_equivalence.py` (7, gpu + sim) | Records the real GR00T N1.6 chunks (its own `gpu_vla_server` on a free port behind `record-proxy`) in `--mode sync`, replays them through the stock `rollout_policy.py` stack in subprocesses; also `--vla URL --seeds 0-4` (now registered in `tests/conftest.py`) or `T13_VLA` / `T13_SEEDS` | pass; 335 chunks, upstream p50 62.7 ms; seed 1 SUCCESS at k = 927, seeds 0, 2-4 TRUNC; 72/47/72/72/72 replayed, 0 mismatches; max \|Δq\| = 0.0 on every compared tick, float32 TickLog and float64 q/dq/base pose/base velocity. Outcomes vary between runs because sync mode sends no CRN seed; the replay makes the comparison exact anyway |
| T-1.6 faults | `test_faults.py` (8, sim) | Four RT runs: VLA stall 2 s, WBC server SIGKILL + restart, VLA server SIGKILL + restart, +1.5 rad arm jump | pass; stall: starved from row 30 (k = 222) for 93 ticks, nav 0 and Balance within 4 ticks, STARVE_FAULT at the 50th starved tick, the episode runs on to TRUNC, 3 give-ups at 519.6-519.9 ms, recovery at d = 3 (d = 11 before the stale-request fix), 0 slips, late max 4.0 µs; WBC kill: ORT on the same tick (5.30 ms), latched after 3 misses, reconnected after a 1.30 s restart; VLA kill: 4 give-ups, period max 20.003 ms through a 2.59 s outage, recovery at k = 236; jump: \|dq_arm\| 13.2 rad/s caught as SAFETY_FAULT, the next episode runs 300 ticks |
| T-1.7 CUDA/EGL guard | `test_cuda_guard.py` (8, sim) | `CUDA_VISIBLE_DEVICES == ""` after imports, after 3 resets and at exit; rt_sim's pid absent from `nvidia-smi --query-compute-apps`; ego frames not black, distinct, and bit-identical with CUDA hidden vs visible-but-unused; viewer off vs on (needs a usable `$DISPLAY`, skips otherwise): two VL runs (mock `detfn`, ORT, d_inj 4, seeds 0-1, 200 ticks) with every ego frame, q43, the commands and every request `obs_sha1` bit-identical, and the on-run's viewer really ran (both model generations shown, frames rendered, `viewer_us` > 0 on every tick, terminated with rc 0); same fidelity-deviation list in both | 8 pass with `DISPLAY=:1`: hidden vs visible 183 frames, 180 ticks of q43 and 9 requests; viewer off vs on 402 ego frames, 400 ticks of q43 and 26 requests bit-identical, viewer 333 frames, publish p50 6.5 / p99 14.2 / max 19.6 µs. Headless: 7 pass, 1 skip |
| Viewer process | `test_viewer.py` (6: 3 fast, 2 need a display, 1 sim) | The seqlock never hands out a torn snapshot (writer in the test, reader in another process, every field carrying the same counter); the MJB handoff round-trips (toy models, two generations of different sizes; xpos, xquat, geom and site poses after `mj_forward`), a superseded generation is unlinked and skipped, `close()` leaves nothing in /dev/shm; a viewer that cannot open a window exits (rc 2) and is logged once while publishing goes on. Display: the viewer follows both generations in one window with all its threads on CPUs 6-7 and exits on SIGTERM; SIGKILL of the loop takes the viewer (PR_SET_PDEATHSIG) and the segments (resource tracker) with it. sim: the G1 env's model and state reproduce the env's xpos/xquat/geom/site poses bit for bit at 2 resets and every 25 ticks over 350 ticks | pass; 431,845 publishes against 204,103 reads, 0 inconsistent, 198,885 torn reads retried; G1: 16 snapshots bit-identical, MJB 135.9 MB, handoff 52.7 ms, publish p50 6.3 / p99 10.2 / max 37.0 µs |
| §1.4.1 request schema | `test_obs_schema.py` (5, sim) | Every RT request is accepted by gr00t's PolicyServer + `Gr00tSimPolicyWrapper(strict=True)`; exactly `VLA_OBS_KEYS` with the table's dtypes and shapes; `options.rt` keys; payload <= 1.0 MB; states bit-equal to float32 of the env observation at k_obs; mutated requests are rejected | pass; 16 requests, 923,378 B each |
| WP-1.6 / WP-1.9 | `test_telemetry.py` (68) | TickLog crash safety and page release; RequestLog jsonl/npz; Publisher; run stamp (git state, dirty-file hashes, package versions, chip session); `analyze_run` gates on crafted failures, every `gates.yaml` set, the M2/M3 keys (starvation, VLA errors, shadow p99.9 with a remote shadow, X280 share, visible wait, x280d heartbeat ring), operator pauses and resets; `eval_paired` statistics (Wilson, paired Wald, McNemar, RMST, log-rank) | pass |
| WP-1.5 helpers | `test_loop_units.py` (9) | Seed and duration parsing, the D17 noise seed, the pacer | pass |
| Review: control | `test_review_control.py` (38) | The production tick code against a fake env and a fake VLA with exact latency d: VL at d_inj = d equals RT at latency d in 24 (arm, s, d) cells, including the 6 that used to lag a tick (d >= s_eff, late reject); VL d_inj = 0 equals sync; each request carries o_k and leaves at k_obs; timeout, starvation and recovery; episode-end accounting; drain and gen | pass |
| Review: robustness | `test_review_robustness.py` (16) | No verdict passes on missing data; viewer gates need the viewer; the VLA gate uses server time; shadow n/a with `--wbc ort`; an abandoned request still on the server counts as in flight; `eval_paired` report-only without pairs is not a pass, and SIGINT/SIGTERM stop its loop processes; Publisher counts drops; RequestLog does no growth and no file I/O inside a tick; `--dump-vla-obs` writes outside the tick; ZmqWbc works with fd >= 1024; malloc trim and the fixed mmap threshold | pass |

## Smoke run (m1_sim_smoke)

The 2-minute RT smoke: a mock `detfn` VLA with 60 ± 5 ms latency on :5559, in-process ORT WBC, no viewer.

```
simpy -m rtserve.mock_vla_server --port 5559 --mode detfn --latency-ms 60 --jitter-ms 5 &
simpy -m rtloop.loop --mode rt --arm v1 --s 16 --B 0 --vla tcp://127.0.0.1:5559 --wbc ort --seeds 0- --duration 120 --tag m1_smoke_fixes
simpy $RT/scripts/analyze_run.py $RT/runs/<run dir> --gates m1_sim_smoke
```

Last result: `runs/20261024_220228_m1_smoke_fixes`, **PASS 7/7** (the 6 gates plus `run_data_complete`).
- 4 episodes (all TRUNC at k = 1439), 5760 ticks, 360 requests, all APPLIED.
- Period p99 20.000 ms, max 20.003 ms; 0 slips; wake-up lateness max 4.4 µs.
- Compute p50 6.67 / p99 9.85 / max 13.0 ms, 0 overruns. Tick 0 is 7.6-8.5 ms: the first touches after the
  reset window's `malloc_trim`.
- d {3: 123, 4: 233}; 0 starved ticks; in flight <= 1.
- VLA: server time p50 60.63 ms (the gated value); client-seen RTT p50 79.97 ms (polled once per tick, reported
  only); queue p50 0.60 ms.
- ORT WBC RTT p50 0.061 / p99 0.177 ms.
- RSS at episode end: 3088, 3090, 3083, 3086 MB. The integration smoke went 4026 → 4382 MB.
- The log has no `Exception ignored` teardown traceback any more.

An A/B run with glibc's malloc defaults (`--mmap-threshold-kb 0 --no-malloc-trim`), right afterwards on the same
mock, also passed 7/7. Its compute was p50 6.71 / p99 9.95 / max 12.9 ms, so the malloc settings do not slow the
tick path. Its RSS went 4085 → 4426 MB over 4 episodes.

With the viewer off, `analyze_run --gates m1_rt` now reports the viewer gates as "no data" (fail). With `--wbc ort`
it reports the shadow gate as "n/a (no shadow)".

## Viewer process (T-1.4 fix)

**Problem.** In the T-1.4 run `runs/20261024_223015_t14` (GPU VLA, GPU WBC, `--viewer`, HUD), robosuite's onscreen
viewer ran inside `env.step`: compute p50 went from 6.6 ms (no viewer) to 16.0 ms, `viewer.sync` cost p50 6.8 /
p99 9.1 ms (up to 13.2 ms at k = 0), `sim_us` p50 15.9 ms, 0.59% of ticks overran 20 ms and one period was
30.08 ms. A VL A/B (400 ticks, mock) gave bit-identical q and ego frames, so the viewer only cost time.

**Change.** The viewer is now its own process, `rtloop/viewer_proc.py`, spawned by the loop under `--viewer`
(INTERFACES.md, "viewer.py + viewer_proc.py"). The env is built as stock (`onscreen=False`) with or without
`--viewer`, so deviations (f) and (g) are gone from config.json. The tick thread only copies the scene state into
shared memory under a seqlock; that copy is `viewer_us`, so the m1_rt viewer gates measure the new path. The model
goes over as an MJB at every reset. The viewer runs on CPUs 6-7 (shared with the HUD), and the loop never waits
for it. `scripts/demo.sh` needs no change: the loop starts the viewer, the viewer inherits `DISPLAY`/`XAUTHORITY`
and the loop's process group, and its window title still starts with "MuJoCo" (`keep_viewer_on_top.py`).

**Measured** (no RT run here, T-1.9 had CPU 2; the T-1.4 rerun confirms under load):

| What | Result |
|---|---|
| Publish (`viewer_us`), G1 env, 350 ticks (`test_viewer.py`, 3 runs) | p50 5.8-6.3 / p99 8.7-10.2 / max 12.0-37.0 µs |
| Publish in VL loops with `--viewer` (600 and 400 ticks) | p50 6.2-10.7 / p99 14.2-26.1 / max 19.5-29.7 µs; k = 0: about 10 µs |
| `sim_us` with `--viewer` (VL, 600 ticks) | p50 5.49 / p99 7.21 ms (the in-process viewer: 11.96 ms p50 in the A/B) |
| Model handoff per reset (reset window) | MJB 135.9 MB in 43-57 ms (92-105 ms when written through the mapping) |
| Viewer side | first window 330-360 ms; a new generation in the same window 160-180 ms; about 60 Hz; all threads on CPUs 6-7 |
| Handoff check (`test_viewer.py`, sim) | after loading, `mj_forward` of the copied state gives the env's xpos, xquat, geom and site poses bit for bit: 16 snapshots, at 2 resets and every 25 ticks |
| Model changes during an episode | none: every MjModel array compared before and after 300 ticks (hold, then walk; seeds 0 and 1); the mocap eef targets do not move. `geom_rgba`/`site_rgba` are copied every tick anyway |
| Viewer SIGKILLed mid-run (VL, 3 episodes) | logged once (`viewer process ... exited (rc -9) at k=148`), all 3 episodes ran to TRUNC, shared memory removed |
| Window closed (WM_DELETE_WINDOW through Xlib) | viewer exits with rc 0, logged once, the run goes on |
| SIGTERM / SIGINT to the loop | loop rc 130, the viewer is terminated (rc 0), shared memory removed |
| SIGKILL of the loop | the viewer goes too (PR_SET_PDEATHSIG) and the loop's resource tracker unlinks the segments (`test_viewer.py`) |
| Viewer shutdown | the first version could crash at exit: rc -11 or rc 1 with `X_GLXDestroyContext`, in 3 of 12 cycles with the G1 model, because mujoco's daemon GUI thread raced the atexit `glfw.terminate`. Joining that thread fixed it: 12 of 12 cycles rc 0 |

**Runs after the change** (CPUs outside T-1.9's 2-4, 12-15 and their siblings; `DISPLAY=:1` where noted):
- Fast suite (`-m 'not sim and not gpu'`, without `tests/tt`, `tests/x280`), headless: 617 passed, 2 skipped (the
  two display tests in `test_viewer.py`). It includes `tests/hud/test_loop_control.py`: 8 passed, the HUD patch still
  applies (hunk 2 with fuzz 1: its last context line, `viewer.patch_viewer_camera()`, is gone by design).
- `test_review_control.py` 38 passed. T-1.2 `test_vl_equivalence.py` 3 passed (7200 ticks, max |dq43| = 0.0).
- T-1.7 `test_cuda_guard.py` 8 passed (`DISPLAY=:1`). `test_viewer.py` 6 passed (`DISPLAY=:1`).
- `test_obs_schema.py` 5 passed. `test_faults.py` 8 passed on 16 CPUs (0 slips; on 7 CPUs, four spinning RT loops
  were oversubscribed and 4 timing checks failed; those runs have no `--viewer`).
- T-1.2b `test_alignment.py`: the 9 VL tests passed (d_inj 1, 3, 8, run from a copy without the RT arm). The 3
  `rt_60ms` tests were not run: that loop has no `--no-affinity` and would have pinned CPU 2 during T-1.9.

Driver scripts: `/tmp/claude-1000/-home-deepgadget-isaacsim/bfbe65f0-1b38-425e-84fc-4937eb211a38/scratchpad/vproc/`
(`model_diff.py`, `robust.py`, `stress_close.py`, `vl_viewer.sh`).

## Not run as tests here

| PLAN id | Why |
|---|---|
| T-1.4 RT timing, official config | Needs `demo.sh m1 --hud --record` with the viewer on, which needs a GUI. A headless run cannot pass it by accident: its viewer gates fail with "no data". To be rerun after the viewer moved out of process (below): `viewer_us` is now the shared-memory publish and `sim_us` no longer contains `viewer.sync`. |
| T-1.5 negative control | Optional, and uses :5555 for both the VLA and the WBC. Agents do not run it. |
| T-1.8 screen, T-1.9 comparison | Campaigns of hours (`scripts/eval_paired.py --matrix configs/eval_m1_{screen,primary}.yaml`). The verdict logic is covered by `test_telemetry.py` and `test_review_robustness.py`. |
| T-1.10 liveness (2 h) | A 2 h RT run against the GPU servers. Its RSS criterion was measured in VL mode instead (below). |

## RSS over many episodes (T-1.10 risk)

Measured in VL mode (`--mode vl --arm v1 --s 16 --d-inj 4`, mock `detfn`, ORT WBC), 30 episodes of 1440 ticks per
configuration. RSS is taken at each episode end, after the TickLog has released its pages. Driver script:
`/tmp/claude-1000/-home-deepgadget-isaacsim/bfbe65f0-1b38-425e-84fc-4937eb211a38/scratchpad/fix/rss_vl3.py`.

| Configuration | ep 1 | ep 30 | ep 1 → 30 | Slope after ep 10 | 220-episode projection (≈ 2 h) |
|---|---:|---:|---:|---:|---:|
| glibc defaults (`--mmap-threshold-kb 0 --no-malloc-trim`) | 4066 MB | 4661 MB | +595 MB | 4.4 MB/ep | ≈ 925 MB |
| `malloc_trim` only (earlier code, 16 episodes) | 3199 MB | 3304 MB | +105 MB | n/a | n/a |
| 1 MiB mmap threshold + trim | 3072 MB | 3090 MB | +18 MB | 0.40 MB/ep | ≈ 83 MB |
| **4 MiB mmap threshold + trim (default)** | 3088 MB | 3093 MB | **+5 MB** | 0.32 MB/ep | ≈ 66 MB |

Cause: robosuite hard-resets on every `env.reset()`. It builds a new MjModel, MjData and render context and frees
the old ones.
- The live objects stay constant (2 MjSim, 2 render contexts, 1 WBC policy). Python allocations grow by less than
  10 KB per reset (tracemalloc). The growth is the same with and without the render-context patch.
- glibc's dynamic mmap threshold rises to the size of the first large block freed. After that, the reset's large
  transient blocks come from the heap and fragment it. About 1 GB of free heap stayed resident.
- Rebuilding the env every 10 episodes does not help: it adds about 120 MB once and nothing else changes. The
  loop therefore does not rebuild.

Fix in `rtloop/loop.py`:
- `set_mmap_threshold` runs before the env is built (`--mmap-threshold-kb`, default 4096). Every per-tick
  allocation is smaller than the threshold: 0 page faults per steady-state tick.
- `malloc_trim(0)` runs in the reset window. It returns 0.5-1.0 GB in 25-50 ms.
- `TickLog.release()` drops the memmap's committed pages, which would otherwise add 0.9 MB per episode.
- Each episode record now has `rss_anon_mb`, `malloc_trim` and `ticklog_released_mb`. config.json records
  `malloc_mmap_threshold_kb` and `malloc_trim_per_reset`.

What is left: about 0.3 MB per episode. About 0.12 MB of that is the RequestLog's in-memory rows, which it keeps by
design to rewrite `requests.npz`. A strictly linear projection is about 60 MB per 2 h, against T-1.10's 50 MB
gate, so T-1.10 should judge the growth after a warm-up (for example from episode 10) or on `rss_anon_mb`.
