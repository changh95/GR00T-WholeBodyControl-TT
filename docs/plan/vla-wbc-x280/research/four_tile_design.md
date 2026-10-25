# 4-tile tensor-parallel gear_wbc on L2CPU0-3: research design (2026-10-25)

Research output of the S5 design lane (no device opened). The step plan S5-0 .. S5e is the work order; the unknowns are what S5a-S5c must measure.

## noc_access

DESIGN DOCUMENT: 4-tile tensor-parallel gear_wbc MLP on L2CPU0-3, with X280-side NoC all-gather. Research only; nothing was edited and the device was not opened. Paths are relative to /home/deepgadget/experiments/gr00t/rtloop (third_party_refs = TPR, ISA = TPR/tt-isa-documentation/BlackholeA0).

1. HOW A HART ISSUES NoC REQUESTS
- Mechanism. Each L2CPU tile has 256 outbound TLB windows (ISA L2CPUTile/README.md:3; TLBWindows.md:3-5): 224 small windows of 2 MiB and 32 large windows of 128 GiB.
- Apertures (x280 physical, TLBWindows.md:9-16; MemoryMap.md:28-39):
  - small uncached: 0x0004_3000_0000 + i*2MiB
  - large uncached: 0x0804_3000_0000 + j*128GiB
  - cached aliases: the same addresses + 0x4000_0000_0000
- Uncached access: every load or store becomes one NoC request.
- Cached access: 64 B line fills and writebacks at arbitrary times, plus L2 prefetch within 4 KiB. There is no coherence across tiles (TLBWindows.md:4-5; Caches.md:77). The design therefore uses uncached small windows only, and the cached aliases must be unreachable.
- Configuration registers:
  - small window i: x280 phys 0x2??0_0000 + 16*i, 16 B = {u64 local_offset; u32 noc_properties_lo; u32 noc_properties_hi}. "??" is any byte value; tt-bh-linux uses 0x2FF0_0000 (TPR/tt-bh-linux/docs/addressing.md:42).
  - large window j: 0x2??0_0E00 + 12*j, 12 B (TLBWindows.md:9-11, 20-34; MemoryMap.md:108-109).
  - From the NoC (the host) the same registers sit at the external-peripheral high alias 0xFFFF_F7FE_FFF0_0000 + 16*i (MemoryMap.md:11, 104). That is the alias x280d already uses for reset vectors and RNMI handlers while the harts are in reset (x280/regs.py:328-363; tt-bh-linux boot.py:232-239).
- Who configures them. Either the hart (physical address) or the host (high alias). CMFW and tt-kmd never touch them: there is no reference in TPR/tt-kmd*/ or fw-19.12.0/lib/tenstorrent/bh_arc. tt-bh-linux programs them from the X280 side (addressing.md:25-117).
- Address encoding. Small window: NoC address = (local_offset[42:0] << 21) | (aperture offset & 0x1F_FFFF). Large window: local_offset[26:0] << 37 (TLBWindows.md:36).
- noc_properties_lo (TLBWindows.md:38-51; the same layout as the struct in addressing.md:314-327):
  - x_end [5:0], y_end [11:6], x_start [17:12], y_start [23:18]
  - mcast [24] (Tensix-only rectangles)
  - ordering [26:25]: 0 default, 1 strict AXI, 2 posted writes, 3 counted writes
  - linked [27], static_vc [28], reserved [30:29], noc_sel [31]
- noc_properties_hi: static-VC buddy and class, keep/skip, exclusion, num_destinations_override (TLBWindows.md:53-69).
- Ordering semantics. The ordering bits mean the same as on the PCIe H2D TLBs (TLBWindows.md:71-72):
  - strict = AXI ordering (writes with the same AXI ID are not reordered; few transactions in flight);
  - posted = the AXI write ack is returned as soon as the data enters the NoC;
  - otherwise the request is NOC_CMD_RESP_MARKED and the ack is returned only after the NoC write ack (TPR/tt-isa-documentation/WormholeB0/NoC/Ordering.md:41-42; NoC/MemoryMap.md:71).
- Sizes. A 2 MiB small window covers the whole LIM of a peer. NoC packets carry 64 B flits, up to 16 KiB per packet (NoC/README.md:3, 28-33).
- NoC hardware alone gives no order between packets unless they use a static VC. Response packets can reorder. The only documented strong order is to wait for the response (Ordering.md:13-18, 20-28, 32).

2. CAN A HART WRITE ANOTHER L2CPU'S LIM?
- Yes, architecturally. A NoC request to an L2CPU tile sees 0x0..0x7FFF_FFFF_FFFF passed straight through to x280 physical space (MemoryMap.md:9; NoC/README.md:19). LIM is 0x0800_0000-0x081D_FFFF (MemoryMap.md:84; Caches.md:73, 92).
- This is the same NIU-slave to CCACHE path that every host LIM write takes today (regs.py:281-286). Only the initiator NIU differs. Nobody has demonstrated an L2CPU-to-L2CPU write yet; S5c is the first one.
- Remote LIM of tile t = NoC0 (8, Y_t) at 0x0800_0000 + off, with Y = {L2CPU0: 3, L2CPU1: 9, L2CPU2: 5, L2CPU3: 7} (ISA L2CPUTile/README.md:39-43; fw-19.12.0 noc_init.c:366-367; tt-bh-linux boot.py:25-30, README.md:350-356).
- Window for peer p:
  - local_offset = 0x0800_0000 >> 21 = 0x40
  - lo = 8 | Y_p<<6 | ordering<<25, i.e. Y3 0x0C8, Y5 0x148, Y7 0x1C8, Y9 0x248, plus 0x0200_0000 for strict ordering
  - hi = 0; mcast, linked, static_vc and noc_sel all 0
  - Peer byte `off` is then at aperture_p + off, because 0x0800_0000 is 2 MiB aligned.
- Use raw NoC0 coordinates only.
  - CMFW programs translation into every NIU, the L2CPUs included (noc_init.c:417-443, 655-690), but the tables are the identity for x=8 and y<12 (MakeIdentity :325-341). The L2CPU entries only add the aliases y=26..29 -> 3,9,5,7 (:562-568). So (8,{3,5,7,9}) reaches the same tile whether TAG_NOC_TRANSLATION is 1 or 0 (regs.py:105-111).
  - The translated aliases (8,26..29) target nothing when translation is off (hang, R28).
  - NoC1 coordinates differ ((8,8),(8,2),(8,6),(8,4): tt-bh-linux README.md:358-362; noc_init.c:343-357).
  - So: NoC0 and raw coordinates only.
- Route. NoC0 routes X first (NoC/MemoryMap.md:205, bit 28). All four tiles sit in column 8, so every route is pure Y, downward around the 12-row ring: hops = (Ydst - Ysrc) mod 12 (2 to 10; for example 7->5 is 10 hops and 5->7 is 2). The write ack travels the rest of the ring, so a round trip is always 12 hops. Packets pass through the ARC (8,0) and security (8,2) routers only in transit.
- Self-identification. A hart learns its own tile from NOC_NODE_ID, y at bits 11:6, of NIU#0 at x280 phys 0x2005_6044 (MemoryMap.md:121; NoC/MemoryMap.md:196-219).

3. HOW tt-bh-linux USES THE WINDOWS
- The demo Linux uses the direct memory port to its own GDDR, not windows: 0x4000_3000_0000 cached, with L2CPU3 in the upper 2 GiB of the shared D7 (console/l2cpu.cpp:30-45; misc/blackhole-p100-2.dts:97-120).
- docs/addressing.md:10-23, 119-144 describes X280-side window programming (TLB_2M_CONFIG_BASE 0x2ff00000):
  - a 128 GiB window to (DRAM x, y, 0), exposed as pmem;
  - host memory through PCIe (19,24) plus the iATU.
- README.md:55-58, 96-98, 337-338: "all L2CPUs can access all DRAMs" this way.
- The host side of tt-bh-linux uses only PCIe TLBs (console/l2cpu.cpp:69-70, 151-153).
- Lesson for this design: the same window reaches GDDR, host memory, Tensix L1 or ARC. Whoever can write a window's configuration or its aperture can reach all of them (see lockdown).

4. THE TP EXCHANGE BUILT ON IT
- Window index = peer tile index p (0..3). The tile's own index stays unconfigured and is blocked by PMP.
- TP region at TP_BASE = 0x0818_0000 in every tile's LIM, 64 KiB, aligned for a NAPOT PMP entry. It lies above the TP slice blob (from 0x0801_1000, about 0.5 MB) and below HOST_BENCH_BASE 0x081D_E400 (x280d.py:134). Contents:
  - +0x000: 16 flag lines, one per (source tile, source hart), one writer each (the mailbox.h:8-10 rule). Word k = tick seq of phase k: 0 OBS, 1 X1, 2 X2, 3 X4, 4 X5, 5 X6, 6 ABORT, 7 JOIN epoch.
  - +0x1000 and +0x5000: two parity-indexed buffer sets (seq & 1), each OBS[516] (33 lines), X1[256], X2[256], X4[512], X5[256], X6[256] f32, about 8.2 KiB.
  - +0x9000: local stats lines.
- Each sender writes only its own column slice into every receiver's X buffer, at the slice's offset. The receiver's GEMV therefore reads one contiguous x vector with no copy.
- Send sequence (per phase): vse32 the slice registers to the local X buffer and to the 3 apertures; `fence ow,ow` (MB_FENCE_REL, mailbox.h:353); then one 4 B flag store per peer (seq).
- Receive: poll your own flag words (uncached LIM, about 28 cycles per dependent load: R33_ESTIMATE.md:16-20). Seq values, not booleans, so a stale flag never matches.
- Why this is enough: each sender's writes to a slice are ordered by its fences, so stale data from an aborted tick always lands before that sender's current-seq data. Parity double-buffering removes the remaining overlap.
- Ordering assumption: the window is non-posted (mode 1 strict, as the KMD and x280d use on the host side: tt-kmd-2.11.0/blackhole.c:245, regs.py:65-67), and the X280 fence waits for the AXI B response, so data is committed in the target CCACHE before the flag leaves. Neither half is documented for the L2CPU outbound path; S5c proves it.
- Fallback protocol if S5c fails: a seq tag in word 15 of every 64 B data line (15 floats + tag), checked by the receiver.

5. CONSIDERED AND EXCLUDED
- MSI catcher (MSICatcher.md): it lives in the peer's 0x2000_0000 page, which also holds the peer's TLB configuration, so mapping it is ruled out.
- NIU request initiators: not usable on L2CPU (NoC/MemoryMap.md:60).
- DMA engine at 0x2008_0000: undocumented (MemoryMap.md:23).
- Linked or static VCs and multicast: never.

## lockdown

THREAT
- An outbound window is a raw NoC master. One wrong configuration word, or one stray store into an aperture, can reach:
  - Tensix L1 (the TT VLA), any GDDR (VLA weights), ARC (8,0) (CSM, reset unit, PLL), or PCIe (19,24) and through it host memory.
- Translated coordinates make this worse: x=17/18 are DRAM, (19,24) is PCIe, (8,0) is ARC (Coordinates.md:30-46).
- The reset values of the 256 configuration slots are unknown. If they are 0, every unconfigured window targets (0,0) = a GDDR tile on NoC0 (Coordinates.md:129).
- The hart's own memory port reaches GDDR with no window at all:
  - uncached 0x3000_0000-0x1_2FFF_FFFF, cached 0x4000_3000_0000 (MemoryMap.md:26, 34);
  - CPUs 0-3 -> D5, 4-7 -> D6, 8-15 -> D7 (ISA L2CPUTile/README.md:23; regs.py:523, 'tt-metal DRAM banks 5/6/7').
  - Today's monitor has no PMP (no pmp/TLB code in x280/fw), so this path is already open.

LAYER 1: THE HOST OWNS THE WINDOW CONFIGURATION (write policy as data in regs.py)
- Today the host may not write any TLB configuration: DENY X280_TLB_CONFIG and X280_TLB_CONFIG_HIGH (regs.py:533-536); the copies at 0x2010_0000-0x2FFF_FFFF fall under the X280_DMA DENY (:537-538). Keep that and add one narrow path, like the PLL exception (regs.py:549-551, 684-686):
  - ALLOW 'X280_TLB_TP', ctx 'tp_windows', covering only the high-alias slots 0..3 (0xFFFF_F7FE_FFF0_0000..0x3F).
  - Values come only from a pure function regs.tp_window(tile, peer): words (0x40, 0, 0x0200_0008 | Y_peer<<6, 0).
  - Before release only, like the vectors (kmd.write_boot_vectors, kmd.py:1653-1690). Written in E1F step A and read back.
  - DENY X280_TLB_CONFIG_HIGH gains exception='tp_windows'.
  - _self_check decodes every allowed value: x=8, y in {3,5,7,9} and not the tile's own y, mcast=linked=static=noc_sel=0, local_offset<<21 == LIM_BASE, hi == 0.
- kmd.boot_artefacts_missing refuses the release unless the windows were written and read back on every tile.
- New READ ranges: all 256 configuration slots of each tile, and NIU#0 id/cfg/counters (0xFFFF_F7FE_FFF5_6000+). x280d verifies all 256 slots:
  - after release, after every PARK/resume, and in the slow health tick;
  - the 3 configured slots must equal the table, all other slots their S5a reset values.
- Fallback if host writes do not take (S5b): the hart writes the slots once in crt0's cold path, from constants keyed by its NOC_NODE_ID, then locks PMP. The host must see a correct readback before TP_JOIN.

LAYER 2: HART PMP, LOCKED SO IT BINDS M-MODE (MemoryMap.md:53-56; 8 entries)
Set in crt0 cold entry before any C code, on hart 0 and in the secondary path; checked again on every warm entry (locked entries ignore writes; a mismatch sets PMP_BAD and TP refuses):
- e0-e2: NAPOT 64 KiB RW, L: the TP region inside each peer's aperture (0x4_3000_0000 + p*2MiB + 0x18_0000). These are left OFF in non-TP sessions.
- e3: NAPOT 0x2000_0000/256 MiB, R|X only, L. Covers TLB config and all its copies, NIU config/security fences (0x2005_6000/7000), DMA, RNMI/vectors, MSI. X is kept because the park stub runs from GP scratch (park_stub.S). Nothing in fw writes this page: only reads at plat.h:41-44 and park_stub.S.
- e4: OFF with pmpaddr = 0x3000_0000>>2; e5: TOR up to 2^47 with no permissions, L. Covers GDDR both ports, all other windows (small and large, cached and uncached) and the reserved copies (MemoryMap.md:26-41).
- e6: NAPOT 0x0201_0000/16 KiB, R only, L. CCACHE regs: WAYENABLE (increase-only, Caches.md:84), ECCINJECTERR, flush.
- e7: spare.
Entries e0-e2 outrank e5 by index. PMP never restricts inbound NoC (host) accesses, so the mailbox protocol is unchanged. A lock cannot be undone until a chip reset, which matches one release per chip reset (README.md:29-30).

LAYER 3: FIRMWARE
- One accessor tp_remote(peer, off) with a range check (off inside the TP region) and FAULT on violation.
- No code path writes 0x2000_0000-0x2FFF_FFFF (in the host-config variant).
- The monitor reports pmpcfg0 and pmpaddr0-7 plus its own TLB slot readback in L36/L39.
- Remote writes are gated on TP_JOIN, which the host sends only after every tile's E1F done word and SELFTEST.

LAYER 4: EVIDENCE PER SESSION
- PMP probes that are harmless if PMP were broken:
  - a store to TLB slot 255 (never used) must take a store access fault, mcause 7;
  - a load from 0x3000_0000 must take a load access fault, mcause 5 (a plain GDDR read if PMP failed);
  - never probe with a store into an aperture or GDDR.
- NIU MST/SLV write counters on both sides must equal the stores issued.
- TT predict_debug bit-identity before and after, plus the TT p50 gate (T-3.E0/E1b pattern).

WHAT REMAINS POSSIBLE
- A bug can corrupt a peer's TP region only. That stays inside X280 LIM and is caught by seq checks, the leader timeout and the bit-identity canary.

## ecc

EXPECTATION: CONFIRMED, WITH CONDITIONS
1. How the CCACHE handles NoC writes into LIM.
   - S1c/S1d (PLAN.md:618; s1d_summary.json): host NoC writes are handled at sub-line granularity with a pre-read.
     - a64/wc2x32 64 B lines reach the CCACHE as 2 x 32 B accesses (events at offset 32 of the line).
     - uc8 lines arrive as 8 accesses.
   - On never-written SRAM, about 0.93 pre-reads per word were uncorrectable and the footprint was poisoned.
   - No host store heals poison; poison and SRAM contents survive tt-smi -r 0.
   - Only hart full-line stores (E1F stage0.S:23-30, vse8 64 B per line) produce valid ECC.
2. After E1F the same path is clean.
   - The host writes 33 WC obs lines + L0 + a 4 B UC seq_req each tick, and the KMD makes its 4 B L38 cleanup write.
   - Evidence: S3b events_before/after ecc fail=0 fix=0 (s3b_mh.json); T37b 27,303 ticks with 0 faults, BEU 0, n_crc_fail 0 (t37_x280d_after.json).
   - Remote X280 writes arrive through the same target NIU slave -> AXI -> CCACHE path. So: the pre-reads of E1F-zeroed lines are clean, and deltas of 0 are expected.
3. Conditions this design must enforce.
   - (a) Never write a peer before its stage-0 done word and SELFTEST.
     - x280d sends TP_JOIN only after all four pass.
     - Followers accept no TP traffic before JOIN.
     - Senders check the receiver's JOIN epoch word.
   - (b) Write only inside [LIM_BASE, LIM_END), here only inside the TP region (PMP e0-e2 plus the firmware check).
     - The window's last 128 KiB (0x081E_0000-0x081F_FFFF) is the cache-way piece and is never zeroed.
   - (c) Remote vector stores are NoC writes, not hart stores, so they cannot heal anything: a poisoned peer line stays poisoned.

WHAT TO MEASURE (S5c, then every TP session)
- Target tile, before and after (regs.py:300-305 CCACHE0_ECC; host reads through the ccache{t} window):
  - dir/dat fix and fail counts, and the low/high address registers.
  - Reading a COUNT register clears only the ECC interrupt signal (regs.py:297-299).
- BEU cause/value/accrued of every hart on source and target (regs.py:321-326):
  - source side: an error response to a remote store should show up here, not as a hang;
  - S3b shows local_interrupt 0 (no RNMI from the BEU), enable 0xEE6.
- NIU counters, read by the hart at 0x2005_6200 + 4*idx (NoC/Counters.md:5-38):
  - source: MST_NONPOSTED/POSTED_WR_REQ_SENT, WR_DATA_WORD_SENT, MST_WR_ACK_RECEIVED;
  - target: SLV_*_WR_REQ_RECEIVED, SLV_*_WR_DATA_WORD_RECEIVED, SLV_WR_ACK_SENT.
  - These give the NoC write granularity of sw/sd/vse32 m1/m4/m8 and prove every write completed.
- Readback: the target hart re-reads its TP region (SELFTEST range variant, BEU check), and the host reads it line by line and compares against the expected pattern.
- Store-width sweep (4/8/64/256/512 B; strict and default ordering) on zeroed lines: expected 0 ECC events. Any fix/fail delta stops the session (DEVX reset).
- Per tick in production: the leader's L39 carries the BEU accrued of all four tiles' harts (each follower publishes it in its own L1), plus the CCACHE counts in the slow health tick.

## latency

A. DECOMPOSITION (column split; bit-identical)
- Ranks: r0 = L2CPU3 (leader; the current default tile, regs.py:37), r1 = L2CPU2, r2 = L2CPU1, r3 = L2CPU0.
- Per layer:
  - L0 (516->256): 64 cols per rank
  - L1: 64 cols per rank
  - L2 (256->35): replicated on all ranks. It is latency-bound per row, so splitting saves nothing; replicating removes one exchange.
  - normalisation and concatenation: replicated (identical code and inputs give identical bits)
  - L3 (121->512): 128 cols per rank
  - L4: 64 cols per rank
  - L5: 64 cols per rank, gathered to r0 only
  - L6 (256->15): on r0 only
- Why it is bit-identical: every output is computed as b + sum over ascending i, each op rounded separately (wbc_mlp.h:15-18). The column-broadcast GEMV has no cross-lane op (wbc_mlp.c:187-216) and ELU is element-wise (elu.c:75-90). So TP == v2 == the x86 build, and X86Ref of the full blob is the oracle.
- Extra freedom: L3 rows 0..85 are obs[430:516] (wbc_mlp.c:247), so they can be accumulated before L2 finishes, in the same order.
- Per-tile TP blob (pack_weights --tp): W^T[in][lo:hi]:
  - about 246 KB per policy, plus 7.7 KB for L6 on r0
  - about 0.5 MB per tile for both policies, against 1.89 MB today (blob end 0x081D_E340, x280d.py:134)

B. PER-EXCHANGE COST: T_x = P*ceil(N/64)*t_st + t_ack + t_flag + t_poll
- t_st: issue cost of one remote 64 B store, posted at the core until the fence. Unknown. A local vle32 costs 8.1 cycles per line (R33_ESTIMATE.md:16-20). Assumed 20 cycles (range 8-40).
- t_ack (the fence wait):
  - NoC part: 12-hop round trip x 9 cycles plus about 10 NIU cycles, roughly 118 NoC cycles = 90-150 ns at AICLK 0.8-1.35 GHz (NoC/README.md:62-66).
  - Bridges: X280 AXI/TileLink, target NIU -> CCACHE write, ack back: about 100-250 ns (unknown).
  - Total about 0.2-0.4 µs, 160-320 cycles.
- t_flag: 4 B one-way, about 0.1-0.2 µs.
- t_poll: about 28-44 cycles per round over 3 lines (dependent load 28 cycles, independent loads 7.8 cycles: R33_ESTIMATE.md:16-20).
- Per tick at t_st = 20:
  - OBS forward: 33 lines x 3 = 99 stores, about 3.0 µs
  - X1, X2, X5: 4 lines x 3 = 12 stores, about 0.8 µs each
  - X4: 24 stores, about 1.1 µs
  - X6: 4 stores to r0, about 0.6 µs
  - Total about 7 µs.
- If remote vector stores split into 8 B NoC writes (S5c tells), the total is about 17 µs. Add 1-2 µs of skew.

C. COMPUTE PER TILE, 4x1 (v2 kernel unchanged)
- Basis: c = 39 cycles per 64-lane group-row, the S3b K9 = K10 = 39.0 cycles/row. One hart is vector-ALU bound: 5 e32m4 ops of about 8 cycles. K12 (loads only) is 28.3.
- Calibration: v2 = 7,640 group-rows x 39 + about 9.9K = 307,877 cycles.
- Per layer:
  - L0: 516 x 39 = 20.1K
  - L1: 9.98K
  - L2 (replicated, 35 lanes, latency-bound): 256 x ~30 = 7.7K
  - L3: 121 x 2 x 39 = 9.4K
  - L4: 20.0K
  - L5: 10.0K
  - ELU, normalisation, concatenation on 384 elements: about 2.5K
  - L6 on r0 only: 256 x ~28 = 7.2K
- Total: about 87K cycles on r0, about 80K on the other ranks.
- Cross-checks:
  - the R33 T_N model (R33_ESTIMATE.md:121) with S = 4: each tile has its own LIM port and vector unit, and K10 scales 4x, so 307,877 x (0.027 + 0.973 x 1.15/4) = 94K;
  - byte model: 246 KB / 2.62 GB/s (K9 on one hart) = 94 µs + L6 + ELU = about 106 µs.

D. EXPECTED INFER (r0 mcycle; 800 MHz)
- 4x1: compute 106-118 µs + exchanges 7-17 µs, about 0.115-0.135 ms. Best estimate 0.125 ms, 3.1x faster than v2's 0.385 ms (and v1 fp32's 0.332 ms).
- 4x4:
  - Each tile streams its 246 KB through one LIM port that saturates at about 5.1 GB/s for this pattern (S3b K9: 2.62 on 1 hart, 5.05 on 2, 5.10 on 4; per-tile ceiling 6.4).
  - The port-bound floor is 48 µs. With 2 harts it is 48.7 µs, so 4x2 is about the same as 4x4.
  - Narrow slices (16 cols per hart, 32 B rows) carry a per-row fixed cost (flw x[i] + vle16 + loop, about 25-35 cycles). That gives 48-67K cycles over about 1,917 rows per hart, unless a software-pipelined kernel (path 1) reaches the port bound of about 20 cycles/row.
  - Plus: L2 7.7K, partly hidden by early L3 rows on harts 1-3; L6 7.2K; intra-tile barriers 6 x ~300; exchanges 7-17 µs; 12 remote flags polled per phase.
  - Total about 0.08-0.11 ms. Best estimate 0.09 ms, about 4.3x.
- Floor: L2 and L6 are latency-bound, about 19 µs, and only a lower per-row latency in path (1) removes it.

E. END TO END THROUGH x280d
- x280d time = host overhead + INFER. Host overhead today is p50 214 / p99 242 µs (T35 t35_summary.json); it includes the monitor's L0/obs CRC, about 44-51 µs (R33_ESTIMATE.md:16-30; monitor.c:203-237).
- So x280d p50 is about 0.34 ms for 4x1 and about 0.30 ms for 4x4, against 0.599 ms now. p99 is about 30 µs higher.
- After TP, the host path (strict-ordered obs write about 34 µs, CRC about 45 µs, ZMQ/Python) is the largest part.

## host_side

1. OBS IN, ACTION OUT
- Measured: one strict-ordered WC obs write (33 lines + L0 + 4 B seq_req) takes p50 34 µs (E1F s1f_host_partial.json write_us).
- Broadcasting to 4 tiles means 4 x 34 = 136 µs of serial host time per tick, 4x the PCIe traffic beside the TT VLA, and 4 CRCs (about 45 µs each, in parallel). That is about 100 µs worse.
- Recommended: send to the leader only, exactly as today (HostMailbox.issue, x280d.py:355-370). The leader checks the CRC and forwards the 33 obs lines to the 3 followers over the NoC (about 3-8 µs).
- Action: read from the leader only (L1 seq_done poll + L35, x280d.py:2352-2363). Followers are never touched on the tick path.
- Option: overlap the forward with the CRC (rank 0 with a smaller L0 share, or hart 1 checking the CRC in 4x4), or replace CRC32 with a vector checksum. That changes the protocol and helps the single-tile path too.

2. BOOT
- One L2CPU_RESET RMW: 0x0F -> 0xFF, bits 4-7 (ISA README.md:35-44). kmd.release already ORs every session tile's bit, writes once and reads back (kmd.py:1605-1651); it refuses masks not in regs.RELEASE_MASKS, so add 0xF0 (regs.py:84-86).
- PLL4 drives L2CPUCLK0-3 together: postdiv0..3 (fw pll.c:225-236), and bh_power.c:32-55 gates all four. So the single 800 -> 200 -> RMW -> 800 MHz excursion (kmd.py:1515 'one PLL4 excursion per session'; ISA README.md:33) releases all four at 200 MHz.
- E1F x4: boot_stage0 already loops over the tiles (x280d.py:711-790): GP scratch + vectors per tile; one RMW; 4 done words with a 2 s timeout each (failure -> ledger burned); image + CRC per tile; L2PF; SET_NOC_CLEANUP; monitor; park stub; SELFTEST.
- Add:
  - step A: tp_windows written and read back on every tile
  - step E: check the PMP report, optionally run v2 parity per tile with the full blob, then load the TP blob per rank, TP_JOIN on followers, TP_JOIN on the leader, TP_PING across all 12 pairs
  - then start server_g1 (E2 order: release before TT builds).

3. regs.py CHANGES
- WRITABLE_TILES = (0,1,2,3) behind X280_TP_APPROVED=1 (regs.py:39 today allows (2,3); classify_write :625).
- RELEASE_MASKS += 0xF0, plus 3-tile masks if a tile is excluded at session start.
- tp_window table, ALLOW X280_TLB_TP with its DENY exception, and READ ranges for the TLB slots and NIU#0 (see lockdown).
- TP region constants; lim_line_owner marks the TP lines x280-owned once released.
- The xp window prefix grows to cover the NIU page if the host reads NIU registers (window_specs :481-502). Otherwise the hart reports them.

4. kmd.py CHANGES
- Kmd(tiles) already enforces WRITABLE_TILES (kmd.py:813-815) and per-tile LimTracker / boot artefacts.
- Add write_tp_windows and read_tlb_slots, plus the release gate.
- Windows: 4 x {lim, ccache, beu, xp} + arc + csm = 18 of the 201 host TLBs.
- Keeper fd plus 3 O_APPEND fds, one SET_NOC_CLEANUP per fd (x280d.py:792-801). The KMD keeps one cleanup per fd (tt-kmd chardev.c:539) and runs it on close (:1030-1039). The keeper's power message enables all four L2CPU clocks (bh_power.c:32-55).
- FakeDevice: TLB slot registers and an inter-tile aperture model for the sim tests.

5. LEDGER
- Already covers tiles 0-3 with fresh, released, parked and burned (ledger.py:27-31).
- All four are marked released before the RMW.
- Keep the rule: 2 or more burned tiles in a session -> stop and review (PLAN.md §5.2; ledger.py:178-180).

6. x280d SESSION
- Session(mode='tp', tiles=(3,2,1,0)) lifts the current limits (x280d.py:462-473): fp16 on one tile, at most 2 tiles, harts=4 only on tile 3.
- policy_tile: both policies go to the TP group (:480-481); blobs per rank; X86Ref = the full v2 blob.
- Server: gear_wbc_action issues INFER|TP on the leader only (x280d.py:2375-2413).
- Follower heartbeats, states, BEU and slot verification move to the slow status tick (STATUS_HZ), never the tick path.
- HUD/status per tile (:2237 already handles a second tile).

7. FAILURE HANDLING ('one tile stalls')
- The leader has a TP deadline, for example 300 µs past the expected phase time. On expiry:
  - it writes L1 FAULT MB_ERR_TP_TIMEOUT, with mtval = bitmap of missing (rank, phase);
  - x280d raises MailboxFault and returns {'error'} at once;
  - ZmqWbc then runs ORT on the same call; 3 misses latch the episode to ORT (wbc_backends.py:7-12, 121-133, 243-305; default timeout 5 ms).
- Followers have their own deadline and abandon a tick when obs_seq moves ahead.
- They poll their own L38, so a KMD 'GONE' cleanup takes every tile out of TP.
- Recovery, off the tick path:
  - read the follower's L1 trap dump, then PARK (RNMI), resume and re-JOIN;
  - if PARK fails (for example a hart stuck on a NoC transaction): ledger burned, the episode stays on ORT;
  - x280d may then reload the leader with the full v2 blob (about 0.3 s, R33_ESTIMATE.md) and serve 1x1 v2 at 0.385 ms. The TP blob and the full blob cannot both fit in LIM;
  - or switch to pre-staged 3-rank blobs.

8. WHAT DP4 (--tile 3,2) ALREADY PROVIDES
- Multi-tile Kmd, windows, LimTracker and boot artefacts.
- One RMW with mask 0xC0 inside one PLL excursion.
- Per-tile HostMailbox, interlocks, ledger.
- Extra O_APPEND fds for per-tile cleanup.
- policy -> tile routing; second-tile status.
- What it lacks: more than 2 tiles, tiles 0 and 1, fp16 on more than one tile, --harts 4 off tile 3, TLB windows, any inter-tile coupling, the TP commands.

## step_plan

S5-0, CPU ONLY (before any device time)
- Tooling:
  - pack_weights --tp 4 --rank r slices; numpy check that the concatenated slice outputs equal the x86 v2 output.
  - Firmware: TP commands (TP_JOIN, LEAVE, PING, STATUS; INFER flag TP), tp_remote, the PMP setup in crt0, gemv on arbitrary O (no change: the tail loop handles 64/128).
  - x86 TP simulation: 4 threads with their own fake LIM; apertures map to the peers' fake LIMs; injected delay and reordering; bit-identity vs X86Ref on 10^4 observations; abort and timeout paths.
  - spike PMP tests: denied stores and loads trap with mcause 5/7 instead of hanging; warm-entry PMP check.
  - kmd fake tests for tp_windows: policy, readback, the release gate (tests/x280).
- Gate: all tests green, image <= 48 KiB.

S5a, PROBE ALL FOUR TILES (read-only, no release)
- Do: x280d probe --tile 3,2,1,0 with the keeper fd, then the T-3.E0 checks per tile:
  - L2CPU_RESET 0x0F, PLL4 default, WAYENABLE 0, ENABLED_L2CPU 0xF, TAG_NOC_TRANSLATION;
  - CCACHE0_CONFIG on raw and translated coordinates; CCACHE ECC counters and BEU baseline;
  - all 256 TLB slots per tile (reset values), the NIU#0 NOC_NODE_ID / NOC_ENDPOINT_ID / NIU_CFG_0 bit 14, hart status 0x2001_0400.
  - Repeat with server_g1 serving (TT bit-identity, p50 within 2%).
- Could go wrong:
  - a read of an undocumented alias or of a clock-gated tile hangs the host (R28);
  - a tile is harvested;
  - the baseline ECC counts are non-zero.
- Detection: subprocess timeouts, the ENABLED_L2CPU check first, recorded baselines.
- Containment: the READ allowlist covers only documented registers; DEVX reset; a harvested tile means a 3-rank plan.

S5b, E1F ON FOUR TILES + PER-TILE SELFTEST (spends all four releases)
- Do:
  - one RMW 0x0F -> 0xFF in one PLL excursion, with the windows written pre-release and read back;
  - four done words, four images, four cleanups, four SELFTESTs (30,720 lines, BEU 0, 0 traps);
  - per-tile PMP report and the safe PMP probes;
  - slots read back after release, to show they persist;
  - per-tile v2 INFER parity with the full blob (max_abs 0.0 vs x86), then the TP blob loads.
- Could go wrong:
  - a tile never seen before (L2CPU0-2) hangs in stage 0, or fails SELFTEST or ECC;
  - host writes do not reach the TLB slots;
  - the PMP lock breaks the park stub or traps legitimate code;
  - the PLL excursion is disturbed by a CMFW relock.
- Detection: per-tile timeouts, SELFTEST/BEU/ECC deltas, slot readback, mcause in L1, the pll4 re-read guard (kmd.pll4_step_to).
- Containment:
  - burned tiles go in the ledger; 2 or more burned -> stop;
  - slot writes failing -> the hart-configured fallback, next session;
  - PMP trouble only costs this session (lock lasts until chip reset);
  - DEVX reset after.

S5c, REMOTE-WRITE MICROBENCH (tp_bench, under the locked PMP)
- Do:
  - (1) Ping-pong round trip over all 12 ordered pairs: 4 B flag, 10^5 iterations, mcycle on the source; one-way = RTT/2. Clocks are 16 cycles per mtime tick, but mtime is not synchronised across tiles.
  - (2) Store sweep (sw, sd, vse32 m1/m4/m8): issue cycles per store, fence wait, and NIU MST/SLV counters giving bytes per NoC write and completeness.
  - (3) Ordering stress: K tagged lines + fence + flag, in strict and default modes (2 windows per peer, both PMP-allowed), 10^6 iterations per pair, 0 violations required. A no-fence negative control only shows whether reordering can be seen.
  - (4) Concurrent all-to-all 256 B slices.
  - (5) ECC and BEU on all tiles before and after, plus a readback of the TP region.
  - (6) TT serving, bit-identity and p50.
- Could go wrong:
  - a store is never acknowledged, so the hart hangs and RNMI may not interrupt it;
  - data overtakes the flag;
  - ECC events on remote partial writes;
  - TT perturbation.
- Detection: heartbeat stall and missing L39 results; violation counts; CCACHE and BEU deltas; TT gates.
- Containment: windows reach only peer LIM and PMP only the TP region; a hung tile is burned; DEVX reset.
- Decision: pick the ordering mode and protocol (fence vs tagged lines), and fill in t_st / t_ack in the model.

S5d, 4x1 TENSOR-PARALLEL INFER
- Do:
  - TP_JOIN on all ranks;
  - parity battery: 10^4 observations from golden and random sets, both policies; x280d TP vs X86Ref, max_abs exactly 0.0; vs ORT within 8.5e-4 (E5a);
  - per-phase mcycle and wait breakdown from every tile's stats line;
  - T-3.5 bench (30k ticks), then the T-3.7 soak with TT serving;
  - fault injection: PARK a follower mid-run -> leader timeout -> client ORT fallback within the call -> resume and re-JOIN; kill x280d -> 'GONE' on all four tiles.
- Gates: bit-identity; INFER p50 at or below about 0.14 ms; 0 faults in the soak; TT p50 within ±5% / 2%.
- Could go wrong: stale or partial slices (non-identical output), deadlock waiting for a flag, follower traps, LIM port interference, timeouts.
- Detection: per-tick X86Ref shadow in test mode, per-tile seq/abort counters, the leader deadline, ECC/BEU/NIU counters.
- Containment: the error reaches the client, which runs ORT that tick; recovery happens off the tick path; ledger.

S5e, 4x4 (and 4x2)
- Do:
  - adopt harts 1-3 on every tile (extend mh.c / X280_MH_APPROVED beyond tile 3), each with its own PMP;
  - intra-tile column split, each hart sending its own 64 B sub-slice with its own flag line;
  - early L3 rows on harts 1-3 while hart 0 runs L2;
  - the same parity and soak gates;
  - compare 4x1, 4x2 and 4x4.
- Could go wrong: adoption fails on tiles 0-2; LIM port saturation (S3b: S4 = 1.94) makes 4x4 no better than 4x2; flag-poll contention; per-row overhead on narrow slices.
- Detection: adoption masks (harts_adopt), per-hart cycle stats, S3b-style interference.
- Containment: run with whichever hart mask was adopted (4x1 floor).

## risks

[
 "A misconfigured or unconfigured TLB window is a raw NoC master into Tensix L1, GDDR (VLA weights), ARC (8,0) or PCIe/host. The host-owned slot table, readback and locked PMP leave only peer TP regions reachable. Today's monitor has no PMP at all, and its memory port already reaches GDDR (MemoryMap.md:26,34).",
 "Ordering: if the L2CPU outbound bridge acks early, or the X280 fence does not wait for AXI B responses, a flag can overtake data. Actions are then silently wrong but plausible. Proven or refuted by S5c stress tests; fallback is a seq tag in every data line, plus a bit-identity canary in S5d.",
 "A remote store whose target never responds can leave a hart stuck on a bus transaction. It may not take an RNMI, so PARK fails, the tile is burned for the session, and a chip reset follows. Only raw (8,{3,5,7,9}) targets on NoC0 are ever used; the translated (8,26..29) aliases hang when translation is off (R28).",
 "ECC: a remote write into a line that was never zeroed (a peer before its E1F done word, or the 0x081E_0000-0x081F_FFFF tail) poisons it, and the poison survives tt-smi -r 0 (S1d). Gated by TP_JOIN after SELFTEST and by the PMP sub-range.",
 "Four releases per session, three of them (L2CPU0-2) never exercised before. More chances to burn tiles; with 2 or more burned the session must stop (PLAN \u00a75.2).",
 "PMP lock is irreversible until a chip reset. A wrong entry (for example one denying the park stub's execute permission) breaks PARK for the session. Mitigation: spike tests, and a first silicon pass that probes the entries before setting the L bit.",
 "End-to-end gain is capped by host overhead (p50 214 \u00b5s, which includes the leader's CRC of about 45 \u00b5s). x280d goes from about 0.60 ms to about 0.34 ms (4x1) or about 0.30 ms (4x4), not 3-4x.",
 "4x4 may not beat 4x2 or 4x1 by much: the per-tile LIM port saturates at about 5.1 GB/s from 2 harts on (S3b), narrow 16-column slices pay a per-row fixed cost, and L2/L6 (about 19 \u00b5s) are latency-bound.",
 "A stalled follower blocks the leader every tick until it is detected. The leader deadline must sit well under the client's 5 ms timeout, and recovery (PARK/resume/re-JOIN) has to stay off the tick path.",
 "No single-tile fallback is resident in TP mode: the TP slices and the full 1.89 MB blob do not both fit in LIM. After a loss the fallback is ORT, then a reload of the full blob on the leader (about 0.3 s).",
 "Interference with the TT VLA: NoC0 column-8 traffic passes through the ARC and security routers (tiny volume, under 20 KB per tick). Four tiles busy-polling add power and heat. Covered by the TT p50 and bit-identity gates.",
 "Changing the host write policy (TLB slot writes from the host) widens a currently absolute DENY. It must stay value-allowlisted, pre-release only, and self-checked at import (regs._self_check).",
 "Firmware growth (TP code, PMP, stats) must stay within the 48 KiB image limit (lim.ld:116), and the monitor's one-writer-per-line rules extend to remote writers."
]

## unknowns

[
 "Reset values of the 256 TLB configuration slots per tile, and whether the host can write them via the high alias (0xFFFF_F7FE_FFF0_0000 + 16i) while the harts are in reset and have the values persist through release (S5a reads, S5b writes and reads back).",
 "The exact semantics of ordering modes 0/1/2/3 on the L2CPU outbound path (TLBWindows.md:47 only lists the names), and whether 'counted writes' is usable.",
 "Whether the X280 `fence ow,ow` waits for the bridge's AXI write responses on the uncached aperture (the PMA of 0x0004_3000_0000: I/O or memory, strongly ordered or not).",
 "NoC write granularity for sw/sd/vse32 m1/m4/m8 into an aperture (one 64 B NoC write or several beats), the number of outstanding stores per hart, and the per-store issue cost t_st.",
 "The actual one-way and round-trip latency between L2CPU tiles (bridges plus the target CCACHE), and the AICLK/NoC clock at the time.",
 "Standard PMP behaviour on the X280: whether the L bit binds M-mode, the granularity G, and whether vector element accesses and the system port are checked as expected. MemoryMap.md:53-56 only says there are eight regions.",
 "BEU and bus-error behaviour for a failing or unanswered remote store (precise fault, imprecise BEU event, or a hang), and whether RNMI can interrupt such a hart.",
 "Whether remote partial writes into E1F-zeroed lines are always ECC-clean (expected yes), and how much interference remote inbound writes cause on the target's LIM port while its hart streams weights.",
 "Whether L2CPU0-2 on this board come up cleanly with E1F and SELFTEST (never released by this project), and whether ENABLED_L2CPU is 0xF.",
 "Whether CLINT mtime is common across tiles (it affects cross-tile timestamps; RTT/2 is used instead).",
 "Per-row cost of narrow column slices (16 or 32 columns per hart) in the 4x4 split, and whether 4 harts per tile beat 2 (S3b: 1.93x at 2 harts, 1.94x at 4).",
 "Whether CMFW's NIU security fences (NoC/MemoryMap.md:27, undocumented) could further restrict L2CPU-originated writes. Not used.",
 "How much of the leader's obs CRC (about 45 \u00b5s) can be overlapped or replaced without weakening the PCIe-ordering guarantee (R36)."
]

## feasible

Yes, as far as the documentation and code show, but three behaviours are undocumented and S5c must prove them.

Why it should work:
- A hart can write any L2CPU's LIM through an uncached 2 MiB outbound window. The window is configured as local_offset 0x40 and NoC0 (8,{3,5,7,9}); the remote LIM is NoC 0x0800_0000. This is the same NoC -> CCACHE path the host already uses every tick.
- The column split (L2 replicated, L6 on the leader) keeps every output's summation order, so the result is bit-identical to v2 and to the x86 build.
- Remote writes into LIM that E1F has zeroed are expected to be ECC-clean.

What S5c must prove:
- Non-posted/strict ordering plus fence: data must land before the flag.
- The cost and NoC granularity of remote stores.
- Safe failure behaviour when a remote store goes wrong.
- If ordering fails, fall back to a seq tag in every data line.

Safety prerequisite: host-written, value-allowlisted window slots read back after release, plus a locked M-mode PMP limiting the harts to the peers' TP regions. Without these the design must not run next to the TT VLA.

Expected gains:

| Setup | INFER | x280d end to end (p50) |
|---|---|---|
| Today, single tile v2 | 0.385 ms | 0.60 ms |
| 4x1 | about 0.115-0.135 ms (about 3x) | about 0.34 ms |
| 4x4 | about 0.08-0.11 ms (about 4x; 4x2 likely the same) | about 0.30 ms |

After TP, host overhead dominates the end-to-end time.

Host side: obs goes to the leader only and is forwarded over the NoC; the action is read from the leader. Boot is one RMW 0x0F -> 0xFF in one PLL4 excursion, then E1F on all four tiles; DP4's multi-tile plumbing covers most of it. Changes are confined to:
- regs.py: WRITABLE_TILES, RELEASE_MASKS, the TLB-slot ALLOW/READ entries;
- kmd.py: slot write and readback, the release gate;
- x280d: TP mode and leader-only ticks;
- the monitor: TP, PMP;
- pack_weights: TP slices.

Next step: S5-0 is CPU-only; then the silicon checkpoints S5a -> S5e, each with a stop condition.
