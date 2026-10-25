# R33 re-derived: per-INFER time on the X280 at 800 MHz with uncached LIM

Status: model, not a measurement. Inputs are spike instruction and memory-access counts (T-3.2b) and llvm-mca
compute cycles (WP-3.1). The one unknown is `c`, the cost in cycles of one uncached LIM access. T-3.E2m measures
`c`; put it into the formulas below before E4 picks a kernel (PLAN §2 T-3.E2m, R33).

## Update 2026-10-25: table-driven monitor CRC (the `C_m` row below is the old, bitwise one)

E1F on silicon (session 20261025_092909_2787811_e1f): v2-fp16 compute 0.385 ms, x280d p50 0.66 / p99 0.69 ms, host
overhead (x280d_us - compute_us) p50 276 / p99 301 us against T-3.5's 0.3 ms. Most of that is the monitor's
bitwise CRC. `mon_crc32` (monitor.c) is now slicing-by-8 over 8 KiB of tables in `.rodata` (`crc32_tab.h`,
generated and checked against zlib by `x280/tools/gen_crc32_tab.py`; `-DMON_CRC_BITWISE` keeps the bitwise loop for
tests). It is zlib-exact (x86: every length 0..4096 from every alignment, both builds; spike: the hart's build
against zlib and the bitwise build's INFER records, `tests/x280/test_crc32.py`, `test_spike.py`).

Measured costs (S2 E2m `lim_bench`, session 20261025_104409_3081695_s2): a dependent uncached LIM load (pointer
chase) takes 28.0 cycles, and so does each load of a scalar stream that uses every load at once. Independent loads
issued back to back cost 7.76 cycles each (SELFTEST: 8 x 8 B per line, 1,906,479 cycles for 245,760 loads). A
`vle32` costs 8.1 cycles per 64 B line. A table chain that goes byte by byte is one dependent load per byte
(>= 28 cycles/B). Slicing-by-8 issues 8 independent loads per 8 B, and the 4 for the high word do not depend on
the chain, so a step costs about 28 + 7 x 7.8 cycles. Slicing-by-4 costs about 28 + 3 x 7.8 per 4 B, about 30 %
more per byte.

| Per INFER, monitor only (spike, VLEN 512) | bitwise (before) | slicing-by-8 |
|---|---|---|
| Instructions `C_m` (T-3.2b overhead = command minus kernel, every variant) | 124,305 | 13,552 |
| Uncached accesses `N_m` (spike `--log-commits`; read path of an INFER) | 651 (322 loads + 329 stores) | 2,809 (2,475 loads, 2,156 of them table entries = 1 per CRC'd byte; 334 stores) |
| `C_m + N_m x 7.8` at 800 MHz (IPC 1) | 129.4 K cycles = 162 us | 35.5 K cycles = 44 us |
| Same, each 8 B step's first table load at the 28-cycle latency (+270 x 20) | | 41.0 K cycles = 51 us |
| Blob LOAD (monitor CRC + `wbc_blob_check`'s own bitwise CRC) | 184.2 M instructions | 86.7 M instructions |

Expected saving: 88 K to 94 K cycles, 110 to 117 us per INFER. That would put the E1F host overhead at about
160 / 185 us (p50 / p99) and x280d end to end at about 0.54 ms p50. Both numbers are predictions until the T-3.5 bench
runs again on silicon. The new image is 23,780 B (was 15,140 B); the 8 KiB of tables are data, not L1I lines.

## Inputs

| Quantity | v0 fp32 scalar | v1 RVV fp32 | v2 RVV fp16 | v2s fp16 scalar | Source |
|---|---|---|---|---|---|
| Kernel instructions per INFER (VLEN 512) | 1,712,077 | 40,042 | 78,287 | 3,588,086 | kernel lane, spike; T-3.2b re-measured within 0.1 % (v0 1,713,081 median) |
| Kernel compute cycles, data cached (`C_k`) | 1.64 M | 0.16 M | 0.36 M | 3.58 M | llvm-mca `-mcpu=sifive-x280`: v0 28 cyc / 8 MAC, v2s 61 / 8 MAC, v1 82 / row, v2 185 / row; 468,992 MACs |
| Kernel uncached LIM line accesses (`N_k`) | 536.6 K | 32.9 K | 18.3 K | 536.6 K | T-3.2b `spike -l --log-commits`, one Walk obs (kernel lane's analytic counts agree within 0.5 %) |
| Monitor work per INFER (`C_m`) | 124,305 instr | same | same | same | T-3.2b: whole command minus kernel, constant (bitwise CRC of L0 + 2,112 B obs) |
| Monitor uncached accesses (`N_m`) | about 0.7 K | same | same | same | monitor.c `read_command`: 16 L0 words, 264 obs u64 loads + 264 copy stores into `.bss`, L35/L1 stores, stack (`lim.ld`: stack and `.bss` are in LIM, uncached) |

Assumptions:
- L2CPUCLK = 800 MHz: PLL4 at the CMFW defaults (fw `pll.c:225-236`), 1 cycle = 1.25 ns.
- LIM data is never cached (ISA `Caches.md:24-31`: LIM is absent from the data-cacheable ranges). Code runs from
  LIM through L1I, which does cache it (`Caches.md:13-14`). The 15 KB image fits in the 32 KiB L1I, so instruction
  fetch is free once it is warm.
- One uncached access per 64 B line touched, `c` cycles each. A `vle32` at m4 touches 4 lines and counts 4.
- Monitor code is scalar with a dependent CRC chain: IPC 1 and no overlap with its loads.
- Kernel bounds: "serial" = `C_k + N_k * c` (blocking loads, nothing overlaps). "Overlap" = `max(C_k, N_k * c)`
  (vector loads in flight while the single vector pipe computes). Scalar v0/v2s on an in-order core sit at the
  serial end. The overlap figure is given only as a floor.
- `c` = 20 / 25 / 40 / 60 cycles: the R33 range ("20-60 cycles", U:x "Performance estimate", low confidence),
  plus 25 and 40 as interior points.

Formula: `T(c) = [C_m + N_m*c + K(c)] / 800 MHz`, where `K(c)` is the serial or the overlap kernel bound.

## Per-INFER time (monitor + kernel; not counting host PCIe or ZMQ time)

| c (cycles) | v0 (serial) | v1 overlap to serial | v2 overlap to serial | v2s (serial) |
|---|---|---|---|---|
| 20 | 15.6 ms | 1.00 to 1.20 ms | 0.63 to 1.08 ms | 18.1 ms |
| 25 | 19.0 ms | 1.21 to 1.41 ms | 0.75 to 1.20 ms | 21.4 ms |
| 40 | 29.1 ms | 1.84 to 2.04 ms | 1.11 to 1.56 ms | 31.5 ms |
| 60 | 42.5 ms | 2.68 to 2.88 ms | 1.58 to 2.03 ms | 44.9 ms |

The monitor alone is 0.17 to 0.21 ms (c = 20 to 60), mostly its 124 K CRC instructions. L1 `cycles_last` measures
the kernel only, so x280d's `compute_us` excludes it. The end-to-end time that T-3.5 gates includes it.

## Break-even values of c (serial bound unless marked)

| Budget | v1 | v2 | v0 |
|---|---|---|---|
| Compute target 1.0 ms (T-3.E4) | c <= 15.3 (overlap: <= 20.1) | c <= 16.6 (overlap: <= 35.6) | never |
| `m3_rt` gate 2.0 ms (T-3.5, x280d end to end; host time not included here) | c <= 39.2 | c <= 58.7 | never |
| `m3_rt_slow` / DP11 8 ms | c <= 182 | c <= 311 | c <= 8.6 |
| One 20 ms tick | always | always | c <= 26.5 (v2s: c <= 22.9) |

What this means for E4 and the DP4/DP11 decisions (the user decides):
- v2 (fp16, the DP4 default) meets the 2.0 ms gate over almost all of the R33 range. It meets the 1.0 ms target
  only if `c` <= ~17, or if vector loads really overlap compute (then up to ~35).
- v1 (fp32) needs `c` <= ~39 for the 2.0 ms gate. Above that, DP4 fp32 needs DP11 (`m3_rt_slow`) or early
  dispatch (WP-3.8b).
- v0 and v2s are parity references only: 15 to 45 ms per call, more than a 20 ms tick for `c` above ~23 to 26.
  Hence x280d caps them at 100 BENCH iterations (`BENCH_SCALAR_ITERS_MAX`). Their BENCH deadline comes from a
  10-iteration pilot, not from a fixed 50 ms per call (review F9; v2s at c = 60 is 44.9 ms).
- A blob LOAD is 184.2 M instructions (monitor CRC plus `wbc_blob_check`) plus 2 x 472 K word loads. That is
  0.25 to 0.30 s at IPC 1 and c = 20 to 60, so `LOAD_DEADLINE_S` = 5 s keeps more than 10x margin even at IPC 0.5.

## Cross-checks on silicon (E2m / E4)

- `c` itself: lim_bench tests 1 (pointer chase) and 2 (scalar stream) give scalar `c`. Tests 3, 4 and 5 (`vle32`
  m1/m4/m8) show whether vector line requests pipeline (overlap) or serialise.
- Stall cycles = L1 `cycles_last` minus the spike instruction count, divided by `N_k`. This must agree with the
  E2m `c`, or the one-access-per-line model is wrong: for example the access counts are off, or vector lines
  overlap.
- BENCH `cyc_min` vs the table: v2 at the measured `c` should land between the overlap and serial columns.
- If v2 turns out compute-bound (`cycles_last` near 0.36 M), the kernel lane's open speedups apply. If it is
  memory-bound, option E4c applies: stack and activations in a Zero-Device pinned L3 way, which removes the
  stack/`.bss` share of `N_k` and `N_m`.

## Multi-hart (harts 0-3 of L2CPU3): the T_N model and the decision rule (S3b, 2026-10-25)

S3 measured v2-fp16 INFER at 307,877 cycles on silicon (14,656 weight lines, ~21 cycles per 64 B) and S2's lim_bench
~7.9 B/cycle per hart for 64-512 B vector loads with a ~28-cycle dependent load. Whether harts 1-3 can share that
work depends on whether LIM serves several harts in parallel, which nothing documents (CCACHE0_CONFIG says 4 banks,
16 ways, 512 sets per bank, 64 B blocks; the bank-select bits are unknown). S3b (`jobs/s3b_mh.sh`) measures it with
the `mh_bench` payload (x280/fw/mh_bench.c) on adopted harts 1-3 (x280/fw/mh.c):

- per kernel and pattern, `S_N = agg(N) / agg(1)` with `agg` = accesses (or bytes) per mtime tick over the common
  span, mtime being the one global clock; mcycle is per hart and only gives per-hart rates;
- K9 is gemv_f16_rvv's single-group inner loop on one 64-column slice (one hart's share of the column split), K10
  the same vector ops on registers only, K12 its loads only.

Split by output columns (bit-identical to V2/V2S: every output keeps its summation order), the serial share is L2 + L6
(25,600 B, 2.7 % of 937,984 B) and there are six barriers per INFER:

    T_N ~ 307,877 x (0.027 + 0.973 x 1.15 / min(N, S)) + 6 x ~200 (barriers) + ~1,200 (dispatch) cycles

with S = the measured S_4(K9) and 1.15 the x[i] scalar load amortised over one register group per hart instead of
four. S = 4: ~97 K cycles (3.2x); S = 3: ~125 K (2.5x); S = 2: ~184 K (1.7x); S <= 1.3: >= 240 K (<= 1.3x).

Rule (x280d `mh_derived`, s3b_mh.json `split_worth_it`): build the split (phase 2) only if S_4(K9) >= 2.0, or if K9
is compute-bound (K9 within 15 % of K10 per row and >= 1.5x K12): then each hart's own vector unit scales the split
until LIM saturates. Readings of the other rows: K2 ~ K3 cycles per access means no overlap inside a hart (the split
pays almost linearly if S_4(P0) >= 3.4); S_4(P2) >= 1.5 x S_4(P1) and a P4 peak at the select bit means bank-bound
(stagger the slices); S_4 <= 1.3 everywhere with P1 ~ P2 and a growing K3 under load means a shared port (no gain).
End to end the split saves at most ~265 us of x280d's 661 us p50; the monitor's L0/obs CRC and ~276 us of host time
stay. A 16-hart version (all four tiles) needs the user's approval after this 4-hart result.

## v3: the single-hart kernel (2026-10-25, model; silicon check in E4/E5a)

**What limits v2 on one hart.** S3b's K9 (v2's per-group loop: vle16 m2, flw, vwmul, vand, vfmul 2^112, vfmul x, vfadd)
takes 39.01 cycles per 128 B and K10 (the same five vector ops, no loads) 39.006, while K12 (the loads alone) takes
28.3. So the loads already overlap the arithmetic and one hart is bound by its single vector arithmetic pipe at about
7.8 cycles per e32m4 op. The 2.62 GB/s (41 % of the 6.4 GB/s port) is the rate of five ops per 64 weights. It does not
come from too few loads in flight. Rescheduling alone (deeper look-ahead, wider LMUL) leaves the main loop where it is,
and only the tails would gain (below). The single-hart lever is fewer vector ops per weight.

**v3 = v2's arithmetic minus one op.** v2 forms p = (w' * 2^112) * x with w' = the fp16 bits shifted into fp32 position
(w * 2^-112). The first multiply is exact. v3 forms X = x * 2^112 once per row (one fmul.s) and p = w' * X. For
|x| < 2^16, X is exact too (subnormal x included), so w' * X equals w * x as a real number and rounds to the same p.
The adds, their order and the ELU are v2's. So the bits are v2's, and each weight group costs vwmul + vand + vfmul +
vfadd (4 ops, not 5). Every row ANDs |x| < 2^16 into a flag. A layer where some row fails (|x| >= 2^16, inf, NaN) is
computed again by v2's gemv_f16_rvv, which costs one v2 layer extra and has never happened on logged obs. Schedule
(inline asm in wbc_mlp.c, because clang reorders the intrinsic version): e32m8 with two groups of 128 outputs.
Each row's two 256 B weight loads go into the registers that the previous row's vwmul has just read, so each load has
7 ops (~109 cycles) to arrive in. x[i+1] loads during row i. The 35- and 15-column tails run in blocks of 4 rows,
interleaved to break each row's vwmul->vand->vfmul chain at short vl, with the next block's weights and x loading.

| Per INFER (one hart, 800 MHz) | v2 (measured 307,877) | v3 (model) | Basis |
|---|---|---|---|
| Main rows (1,782 x 256 outputs; 912 KB) | 278 K (4 x 39 per row) | 205-222 K (115-125 per row) | K9: 7.8 cycles per m4 op; v3's 4 m8 ops = 8 m4-equivalents. Low end: vwmul at its source width (llvm-mca's SiFive7 costs) |
| L2 + L6 tails (512 rows, vl 35 / 15) | ~20 K (residual) | 11-13 K (in-order bound 19 K) | L2: max(VA 20-32, LIM 2.1 lines x 8 + flw 7.8 = 24) per row; L6: LIM 1.45 lines x 8 + 7.8 = 20 per row |
| ELU (1,536 values, e32m4) | ~6.3 K | ~6.3 K | 24 strips x ~33 ops x 8; unchanged (shared with v2) |
| Biases, stores, concat/normalise, call and layer drains | ~3 K | ~3 K | unchanged |
| **Total** | **307.9 K = 0.385 ms** | **225-251 K = 0.28-0.31 ms** (central ~236 K, v3/v2 ~0.77) | high end: tails at their in-order bound |

Evidence on the CPU (no device):
- Bit identity: x86 v3 (the same arithmetic in scalar code) equals v2 on all 10,080 logged obs + 256 N(0,1) for both
  policies and on 31 crafted rows (guard edge 65535.996 / 65536, +-inf, NaN, FLT_MAX, subnormals, +-0, x 1e4..1e6
  scaled rows). The per-product identity holds for all 63,488 finite fp16 weights x ~6,500 x values with |x| < 2^16,
  and breaks at x = 2^16, so the guard sits exactly at the edge (tests/x280/test_kernel_x86.py). Spike (RVV, VLEN 512):
  v3 equals spike v2 and x86 v2 on 1,000 logged obs, on infer_const/BENCH, and on 16 crafted + 8 plain obs whose
  fallback rows show in the instruction counts (tests/x280/test_spike.py, T-3.2b).
- Spike instructions per INFER: v3 54,145 vs v2 78,287 (0.69). A row with a fallback in every layer costs 130,189.
- llvm-mca -mcpu=sifive-x280 (`make -f Makefile.kernel kernel-mca`): v3 main row 125 cycles (block RThroughput 120);
  v2's row of the same 256 outputs is 197 (RThroughput 164). The tails are 172 / 121 cycles per 4-row block (m4 / m1).
  This model is in order and charges a full LMUL per op, so the tail figures are an upper bound.
- No vector spills in gemv_f16_v3 (test_rv64_objects_are_freestanding); monitor.bin 28,444 -> 29,448 B (kernel text
  5,550 -> 6,590 B); v2's gemv_f16_rvv and forward compile to the same instructions as before; no Zvfh.

Silicon check (jobs/e4_bench.sh: the E4 matrix includes v3; e4_choice.json `v3_vs_v2` gives cyc_min per policy and
v3/v2 against 225-251 K; jobs/e5_parity.sh runs v3_fp16 against x86 v2 on the 10k obs, bit for bit):
- v3 in 225-251 K: the model holds, and the single-hart kernel is at its vector-pipe floor (4 ops per weight without Zvfh).
- v3 >= ~290 K: the arithmetic pipe was not the limit after all, and this section's reading of K9/K10 is wrong.
- v3 < 225 K: vwmul is cheaper than either reading. Fine.
The session default stays v2 (monitor pick_variant, x280d) until both pass.

Multi-hart consequence: a v3 hart streams ~4.1 B/cycle (512 B per ~125 cycles), about half of one tile's 7.97 B/cycle
vle ceiling (K5-K8). Two harts on one tile would saturate LIM (K9 already reached 5.05 GB/s on 2 harts), so for v3 the
4-tile split (one LIM port each) is the next step, not 4 harts on one tile.
