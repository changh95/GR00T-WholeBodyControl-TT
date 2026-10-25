/* Stage-0 LIM ECC initialiser, <= 64 B in the GP scratch at 0x2001_0100 (PLAN.md §5.2 "E1F", R26).
 *
 * CONTINGENCY ONLY. It runs only in session S2' after E1a has failed with host methods A and B, it spends
 * L2CPU3's one release, and it needs the user's explicit DP3 approval at that time. Nothing in this
 * repository loads or runs it; x280d must never do so silently.
 *
 * Reset handler addresses of harts 0-3 point here. Hart 0 turns on mstatus.VS, zero-fills all of LIM
 * (0x0800_0000-0x081D_FFFF, MemoryMap.md:84, Caches.md:92) with one 64 B vse8.v per line (vl = 64: e8/m8
 * gives VLMAX = VLEN >= 64 for any legal VLEN, so the store is a whole line), writes the done word
 * 0x081E_0000 to the last LIM line (0x081D_FFC0), then spins on the entry word in the line before it
 * (0x081D_FF80, written by the host; one writer per line). When the entry word is non-zero it runs
 * fence.i and jumps there (crt0). Harts 1-3 wait in wfi.
 * sp is used only as a pointer, so that c.addi16sp/c.swsp/c.lwsp keep the stub inside 64 B.
 */
    .section .stage0, "ax"
    .option norelax
    .globl stage0_start
stage0_start:
    csrr    a0, mhartid
    bnez    a0, 3f
    li      a1, 0x200               /* mstatus.VS = Initial (bits 10:9) */
    csrs    mstatus, a1
    li      a2, 64
    vsetvli zero, a2, e8, m8, ta, ma
    vmv.v.i v0, 0
    lui     sp, 0x08000             /* LIM base */
    lui     a5, 0x081E0             /* LIM end, also the done word */
1:  vse8.v  v0, (sp)
    add     sp, sp, a2
    bltu    sp, a5, 1b
    addi    sp, sp, -128            /* 0x081D_FF80: entry word line */
    sw      a5, 64(sp)              /* done word in the last line 0x081D_FFC0 */
2:  lw      a0, 0(sp)
    beqz    a0, 2b
    fence.i
    jr      a0
3:  wfi
    j       3b
    .globl stage0_end
stage0_end:
    .org    stage0_start + 64       /* assembler error if the stub grew past 64 B */
