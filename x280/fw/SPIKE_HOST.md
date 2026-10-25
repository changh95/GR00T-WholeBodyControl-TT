# SPIKE_HOST contract (x280/fw, PLAN.md WP-3.2 / WP-3.2b)

`-DSPIKE_HOST` builds the same monitor (crt0, monitor.c, trap.c, kernel) for spike. Spike has no PCIe
host, so a **scripted host compiled into the image** plays x280d: it writes the host lines in the §1.4.4
order (obs lines, L0 bytes 0-59 as one line, then `seq_req` alone; L37 for LOAD), runs the monitor's
poll loop until `seq_done` changes, and prints the raw mailbox lines. **The firmware reports; the
harness judges.** This file is the whole interface between this lane and the spike harness.

## Build

```
make -C $RT/x280/fw spike [SPIKE_BUILD=<dir>] [SPIKE_SCRIPT=<file.c>] [SPIKE_X280_REGIONS=1] \
     [SPIKE_BLOB=<blob>] [SPIKE_OBS=<obs.bin> SPIKE_OUT=<out.bin> SPIKE_N=<n>] \
     [SPIKE_POLICY=<0|1>] [SPIKE_VARIANT=<v>] [SPIKE_CRC_BITWISE=1] \
     [SPIKE_MH=1 SPIKE_SCRIPT=spike_script_mh.c [SPIKE_MH_AUDIT=1]]
```

| Output (in `SPIKE_BUILD`, default `build/spike`) | What |
|---|---|
| `monitor_spike.elf` / `.bin` / `.map` | the monitor + kernel + runner + embedded payloads, linked by `lim.ld` at 0x0800_0000 |
| `stage0_spike.elf` | `stage0.o` (byte-identical to `build/stage0.bin`) at 0x2001_0100 plus a timer harness |

Paths given to `SPIKE_BLOB/OBS/OUT` are made absolute at build time (spike resolves them against its own
cwd otherwise). Use a separate `SPIKE_BUILD` per configuration when building in parallel. The link reserves
`SPIKE_STACK` (default 2 KiB) for the stack instead of the X280 image's 16 KiB: the runner's ~14 KiB (script,
harness, obs staging, embedded payloads) sit below it, the deepest frame chain is ~1.2 KiB (-fstack-usage), and the
compact 4 KiB multi-hart region (`.mh` at 0x0800_E800, asm.h `MH_SPIKE` geometry: same ctl/stat/aux lines, 128 B trap
stacks, 1 KiB per-hart areas) sits between `.persist` and the stack. The X280 image's `.mh` is the 8 KiB at 0x0800_A000.
`SPIKE_MH=1` embeds `build/payloads/mh_bench.bin` (and only the park stub besides it) for the multi-hart script.

## Run

```
spike --isa=rv64gcv_zfh_zba_zbb_zvl512b_zve64d -m0x8000000:0x1e0000 [--instructions=N] monitor_spike.elf
```

* Memory: LIM only (`-m0x8000000:0x1e0000`). With `SPIKE_X280_REGIONS=1` the harness must also map
  `0x20010000:0x1000` (reset handler addresses, GP scratch, RNMI trigger) and `0x1700000:0x1000` (hart-0
  BEU page) as RAM; the runner then installs the real park stub there at boot (§5.2 boot steps 4-5) and
  every emulated RNMI goes through it.
* Tested with riscv-isa-sim 2c300cd (x280/sim/SHAS.txt), 1 hart, and 4 harts (`-p4`, run_spike `harts=4`) for the
  multi-hart script and the flag-off regression (the default script's output is byte-identical on -p1 and -p4: a
  wfi'd hart consumes its 5000-instruction quantum at once, and mtime moves 50 ticks per round, riscv/sim.cc:365-385).
  `mcycle` counts spike instructions and
  `mtime` ticks at 10 MHz (100 instructions per tick, riscv/sim.h:79-80): cycle numbers are not X280 cycles.
* `--instructions=N` is a safe timeout; a correct default-script run takes about 5 M instructions
  (~0.05 s), plus ~0.1 s per MB of blob for LOAD (the monitor's table CRC and `wbc_blob_check`'s bitwise one)
  and per-INFER cost (v0 fp32 ~1.7 M instructions, v1 ~40 k, v2 ~78 k).

## What is emulated

| X280 feature | Under SPIKE_HOST |
|---|---|
| CSRs 0x350-0x353 (mnscratch/mnepc/mncause/mnstatus) | memory variables in spike_host.c (spike traps the numbers, x280/sim/selftest/t_csr.c); `mnstatus.NMIE` = 1 at boot |
| `csrw 0x7c1, 0` | skipped |
| `csrsi 0x353, 8` (crt0 re-arm) | `plat_rnmi_rearm()` sets the emulated NMIE |
| host RNMI for PARK | taken right after PARK completes |
| host RNMI on a hung payload | a CLINT machine-timer interrupt (`mtimecmp` at 0x0200_4000) armed by `SH_CALL_HANG` |
| RNMI taken | only if emulated NMIE = 1; clears it, sets mncause = 2^63 + 2 and mnepc, jumps to the park stub (regions) or to crt0. If NMIE was not re-armed the run stops with exit code 0x7F (`SH rnmi_lost`) - on the X280 that hart would stay hung |
| RNMI trigger-bit spin | hart 0: not exercised (the bit is always 0). Harts 1-3 (`SH_MH_ADOPT`): spin in crt0's secondary path until the scripted host clears their bits |
| RNMI of harts 1-3 | `sh_emu[h]`, one 64 B line per hart (crt0.S reads `sh_emu + 64 h`); with `SPIKE_X280_REGIONS=1` a cold secondary waits in wfi with mie = MSIE, and when an IPI wakes it with its trigger bit set and its emulated NMIE = 1 it takes the RNMI itself: NMIE cleared, mncause 2^63 + 2, mnepc = its wait loop, jump to its RNMI handler address in the 0x2001_0418 + 16 h RAM page. Without the regions it waits in wfi with mie = 0 for ever, as on the X280 |
| BEU (0x0170_0000) | reads 0, or the mapped RAM page with `SPIKE_X280_REGIONS=1` |

## Output (stdout, one line per record, via the HTIF syscall proxy `write(1, ...)`)

```
SH boot L1=<hex> L36=<hex> L39=<hex>
SH crc off=<o> nmax=520 slices=<8|0> digest=<u32> chained=<u32>      (8 lines, o = 0..7, after the boot line)
SH step=<i> op=<name> sub=<k> seq=<n> res=<res> cyc=<c> cyc_sum=<s> L1=<hex> L35=<hex> L36=<hex> L39=<hex>
SH unexpected_entry L1=<hex>
SH rnmi_lost: ...
SH mhadopt step=<i> mask=<m> pre_ready=<r0> ready=<r1> polls=<n>                    (SH_MH_ADOPT)
SH mh step=<i> sub=<k> C=<hex> S1=<hex> A1=<hex> S2=<hex> A2=<hex> S3=<hex> A3=<hex>   (report SH_REP_MH)
SH mhb step=<i> sub=<k> J=<hex> R0=<hex> R1=<hex> R2=<hex> R3=<hex>                    (report SH_REP_MHB)
SH end steps=<n> timeouts=<t> entries=<e>
```

* `<hex>` is one 64 B mailbox line in memory order (128 hex digits). Decode with
  `x280/fw/mailbox_layout.py`: `unpack_l1`, `unpack_l35`, `unpack_l36`, `unpack_l39`
  (`unpack_lb_result(l39["data"])` for lim_bench results, `unpack_l39_mh` for an MB_CMD_MH ping,
  `unpack_mhb_summary` for an mh_bench CALL). The extra lines (printed after a step's `SH step` line when the step's
  `report` asks): `C`/`S<h>`/`A<h>` = the MH ctl line and hart h's stat / aux lines (`unpack_mh_ctl`, `unpack_mh_stat`,
  `unpack_mh_aux`), `J`/`R<h>` = mh_bench's job line and hart h's result line (`unpack_mhb_job`, `unpack_mhb_res`).
  `SH mhadopt`: `pre_ready` = mh_ready() while the trigger bits are still set (1: hart 0 only, the others spin in
  crt0), `ready` after they were cleared.
* `res`: `done` (seq_done reached), `reentered` (the step's command ended in a warm entry: PARK or an RNMI
  on a hang), `timeout` (no seq_done after 4e8 polls), `open_failed`, `bad_op`.
* `entries` counts monitor entries (1 cold + every warm entry).
* `SH crc`: `mon_crc32` as compiled for the hart, over the image's own first bytes at LIM base + `o`: `digest` is
  FNV-1a (32-bit, offset 2166136261, prime 16777619, each CRC as 4 little-endian bytes) over the CRCs of every
  length 0..`nmax`, then 2,112 and 4,096; `chained` is one 4,096 B CRC continued over pieces of 1, 2, .., 13, 1, ..
  bytes. The harness recomputes both with zlib from `monitor_spike.bin`. `slices` = 8 (table) or 0
  (`SPIKE_CRC_BITWISE=1`, `-DMON_CRC_BITWISE`). Printed once per cold boot, before the first step.
* `cyc`: `mcycle` from the host's `seq_req` write to the poll that saw `seq_done` for the step's last
  command, i.e. what the hart spent on it (monitor + kernel); 0 for a `reentered` command. `cyc_sum`: the
  same summed over the step's commands (every record of an `INFER_FILE`, whose `sub` is the record count).
  Under spike `mcycle` = instructions retired. `L1.cycles_last` is the kernel call alone.
* `INFER_FILE` reports `res=timeout` if any record did not complete.
* **HTIF exit code** = number of steps with `timeout`/`open_failed`/`bad_op` (0 = every step completed);
  0x7F = RNMI lost. Whether each completed step has the right state is for the harness to check.

## Script format

A C file that includes `spike_host.h` and defines `const struct sh_step sh_script[]`, ending with
`{ .op = SH_END }` (see `spike_script_default.c`). Fields not listed are 0.

| op | fields used | does |
|---|---|---|
| `SH_CMD` | `cmd policy flags tick arg_addr arg_entry arg_len count` | issues the command `count` times (INFER/BENCH send the current obs lines) |
| `SH_LOAD_BYTES` | `bytes..bytes_end dst kind` (`SH_BYTES(sym)`) | host-writes the bytes at `dst` (full lines, zero-padded), writes L37 (CRC, `load_seq`, `desc_crc32`), LOAD |
| `SH_LOAD_FILE` | `path dst kind` | the same with a file read via HTIF straight into LIM |
| `SH_INFER_CONST` | `policy flags tick count` | `count` INFERs, obs from the PRNG below with seed `tick + sub` |
| `SH_INFER_FILE` | `path path2 policy flags count` | INFER every 2,064 B record (f32[516] LE) of `path` (`count` 0 = all); appends 128 B per record to `path2`: the L35 line then the L1 line |
| `SH_CORRUPT_INFER` | - | INFER with obs_crc32 + 1: expect state 3, mcause 0xC0C, mtval 2 |
| `SH_POKE32` | `dst arg_len` | host 4 B write `arg_len` at `dst` (e.g. L38 = 0x474F4E45, the KMD cleanup), then 1e5 idle polls |
| `SH_PARK` | - | PARK; reported as `reentered` |
| `SH_CALL_HANG` | `arg_entry count delay_us` | `count` x: arm the timer (`delay_us`, default 200), CALL; each reported as `reentered` |
| `SH_MH_ADOPT` | `flags` (harts, 0 = 0xE), `delay_us` (default 100), `report` | `SPIKE_X280_REGIONS` only: the `--harts 4` RNMI vectors of harts 1-3 into the RAM page (handler 0x0800_0000, exception 0x0800_0200), trigger bits = `flags`, an IPI to each (spike stand-in for the RNMI's wake), `delay_us` of mtime, bits cleared, then up to 5 ms for mh_ready(); `res` done / timeout, then `SH mhadopt` |
| `SH_LOAD_PATTERN` | `dst arg_len tick` | host-writes `arg_len` bytes of xorshift64 (13, 7, 17; seed `tick` \| 1, one u64 per step) at `dst`, then LOAD kind DATA |
| `SH_MHB_SWEEP` | `sweep` (kernels, patterns, k, masks, hart-0 kernel, mode), `tick arg_addr arg_len report` | CALL mh_bench at MHB_BASE for every kernel x pattern x k x mask, `sub` = the combination index; one step line (+ `report` lines) each |
| any op | `report` | `SH_REP_MH` / `SH_REP_MHB` bits: the extra `SH mh` / `SH mhb` lines after each step line |

Embedded payloads (`spike_payloads.S`): `sh_bin_checksum`, `sh_bin_illegal`, `sh_bin_hang`,
`sh_bin_lim_bench` (all linked for 0x0801_1000; position-independent in practice: medany code, no absolute
data pointers) and `sh_bin_park` (the 64 B park blob); a `SPIKE_MH=1` build embeds `sh_bin_mh_bench` (linked at
0x081D_F400, absolute: LOAD it there) and `sh_bin_park` only.

**SH_INFER_CONST obs:** `x = (seed << 1) | 1` (uint32), then for i in 0..515:
`x ^= x << 13; x ^= x >> 17; x ^= x << 5; obs[i] = float(x >> 8) * 2^-23 - 1.0f` (exact in float32;
numpy: compute in float64 and cast).

## Default script (`spike_script_default.c`) and the expected results

| step names | expect |
|---|---|
| `nop` | state 1 |
| `selftest` | state 1; L39 kind 1, lines 30720, misa_v 1, vlenb 64 (with zvl512b) |
| `load_checksum`, `call_before_fence` | state 1, then state 3 mcause 0xC13 mtval 1 (loaded, not fenced) |
| `fence_i`, `call_checksum` x10 | state 1; L39 kind 3, `ret` = zlib crc32 of `monitor_spike.bin[0:4096]` every time |
| `load_illegal`, `fence_i`, `call_illegal`, `nop_after_trap` | state 3 mcause 2 mepc 0x0801_2000 mtval 0xC0001073; then state 1 |
| `load_lim_bench`, `fence_i`, `lb_*` | lb_result status 0 for tests 1-6 and 8, status 4 (denied) for `lb_zero_dev_denied` |
| `park`, `nop_after_park` | `reentered`, state 4, mcause 2^63+2, n_reentry 1; then state 1 |
| `load_hang`, `fence_i`, `call_hang` x10, `nop_after_hang` | each `reentered`, state 4, mepc 0x0801_3000, n_reentry 2..11; then state 1 |
| `infer_bad_obs_crc` | state 3 mcause 0xC0C mtval 2 |
| `kmd_gone`, `nop_while_gone`, `attach_clear`, `nop_after_attach` | state 5, state 5 (seq adopted, not run), state 1, state 1 |
| `unknown_cmd` | state 3 mcause 0xC0E mtval 9 |
| with `SPIKE_BLOB`: `load_blob`, `infer_const` x4, `bench` | state 1; actions in L35 (seq_echo = seq); L39 kind 2 nondet 0 |
| with `SPIKE_OBS`/`SPIKE_OUT`: `infer_file` | `path2` holds one 128 B record per obs |

Checked on 2026-10-24 with spike 2c300cd: default script `SH end steps=32 timeouts=0 entries=12`; with
`SPIKE_X280_REGIONS=1` and an fp16 blob, 35 steps, 0 timeouts; `INFER_FILE` on 40 N(0,1) obs was
bit-identical to the x86 build's `wbc_infer` for v2 (fp16, both policies), v0 and v1 (fp32 Walk; v1
against x86 v0).

## stage0_spike.elf

```
spike --isa=rv64gcv_zfh_zba_zbb_zvl512b_zve64d -m0x8000000:0x1e0000,0x20010000:0x1000 stage0_spike.elf
```

Exit 0 = stage 0 zero-filled all of LIM, wrote the done word 0x081E_0000 at 0x081D_FFC0 and jumped to the
entry word the harness wrote at 0x081D_FF80; 1 = a non-zero LIM word; 2 = wrong done word. The harness
(timer interrupt every 1000 mtime ticks) stands in for the host; stage 0's bytes are unchanged.

## Multi-hart script (`spike_script_mh.c`, `SPIKE_MH=1 SPIKE_X280_REGIONS=1`, run with `-p4`)

| step names | expect |
|---|---|
| `mh_ping_unadopted` | L39 kind 5 (MB_CMD_MH ping): done 0, ready 1; L36.mh = MB_MH_CAP \| 1; no stat line yet |
| `mh_adopt` | `SH mhadopt` mask 14, pre_ready 1, ready 15; stat lines IDLE, entries 1, mncause 2^63 + 2, mode WFI, mip MSIP |
| `mh_ping` | done 14, ready 15, gen 2, wake ticks within two spike rounds (100 ticks); L36.mh = MB_MH_CAP \| 0xF |
| `load_ws`, `load_mh_bench`, `fence_i` | a 64 KiB xorshift working set at 0x0802_0000; mh_bench at 0x081D_F400 |
| `mhb_sweep` (192), `mhb_p4` (7), `mhb_loaded_latency` | per CALL: summary status 0, done = mask; each result line current (tag), accesses a multiple of the kernel's chunk, bytes = accesses x bytes per access, sink = `mailbox_layout.mhb_sink` over the working set, t0 >= t_start |
| `mhb_bad_range` | status 2 (ERANGE: the working set covers the payload) |
| `mhb_bg_start`, `nop_during_bg`, `mhb_bg_collect` | dispatched 0xE; the NOP runs while harts 1-3 are BUSY; collect: done 0xE, the start's tag |
| `mhb_trap_h2`, `mh_ping_after_trap`, `mhb_after_trap` | done 0xB; hart 2's aux: mcause 2, mtval 0xC0001073, its stat: traps 1, IDLE; hart 0: n_traps 0; then done 0xE and 0xF |
| `selftest`, `nop_end` | hart 0 unharmed: state 1, n_traps 0, n_reentry 0, entries 1 |

`SPIKE_MH_AUDIT=1` keeps a short subset (adoption, ping, 12 CALLs, the trap) for run_spike's `audit_stores`
(`-l --log-commits`): tests/x280/test_spike.py checks the per-hart store lines (one writer per line) and that no
A-extension instruction commits.
