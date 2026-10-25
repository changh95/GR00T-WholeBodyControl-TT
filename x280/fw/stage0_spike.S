/* Spike harness for the stage-0 LIM initialiser (stage0.S, PLAN.md §5.2 E1F), linked with the very same
 * stage0.o as build/stage0.elf, at the same address 0x2001_0100.
 *
 * Spike cannot play the host that writes the entry word, so a machine timer interrupt does: s0_setup
 * (the ELF entry, 0x2001_0200) points mtvec at s0_tick, arms mtimecmp and jumps to stage0_start. Each tick
 * checks the done word (0x081E_0000 at 0x081D_FFC0); once it is there it writes the entry word
 * (0x081D_FF80) = s0_check and returns. Stage 0 then runs fence.i and jumps to s0_check, which verifies
 * that every word of LIM is zero except those two and exits through HTIF: 0 = pass, 1 = a non-zero word
 * (its address in a0 is not reported), 2 = the done word is wrong.
 * Memory the harness must map: -m0x8000000:0x1e0000,0x20010000:0x1000 (tohost lives in the latter, so
 * stage 0's zero-fill cannot clobber it).
 */
#include "asm.h"

    .section .text.s0harness, "ax"
    .option norelax
    .globl s0_setup
s0_setup:
    la      t0, s0_tick
    csrw    mtvec, t0
    li      t0, 0x0200BFF8          /* CLINT mtime (plat.h) */
    ld      t1, 0(t0)
    addi    t1, t1, 1000
    li      t0, 0x02004000          /* hart-0 mtimecmp */
    sd      t1, 0(t0)
    li      t0, MIE_MTIE
    csrs    mie, t0
    li      t0, MSTATUS_MIE
    csrs    mstatus, t0
    la      t0, stage0_start
    jr      t0

    .p2align 2
s0_tick:
    csrw    mscratch, t0
    la      t0, s0_save
    sd      t1, 0(t0)
    sd      t2, 8(t0)
    li      t1, 0x0200BFF8
    ld      t2, 0(t1)
    addi    t2, t2, 1000
    li      t1, 0x02004000
    sd      t2, 0(t1)               /* next tick */
    li      t1, 0x081DFFC0
    lwu     t2, 0(t1)
    li      t1, 0x081E0000
    bne     t2, t1, 1f
    li      t1, 0x081DFF80
    la      t2, s0_check
    sw      t2, 0(t1)               /* entry word: stage 0 jumps there */
    li      t1, -1
    li      t2, 0x02004000
    sd      t1, 0(t2)               /* no more ticks */
1:  ld      t1, 0(t0)
    ld      t2, 8(t0)
    csrr    t0, mscratch
    mret

s0_check:
    csrci   mstatus, MSTATUS_MIE
    li      t0, 0x08000000
    li      t1, 0x081E0000
    li      t3, 0x081DFF80
    li      t4, 0x081DFFC0
2:  beq     t0, t3, 3f
    beq     t0, t4, 3f
    ld      t2, 0(t0)
    bnez    t2, 8f
3:  addi    t0, t0, 8
    bltu    t0, t1, 2b
    li      t0, 0x081DFFC0
    lwu     t2, 0(t0)
    li      t1, 0x081E0000
    bne     t2, t1, 9f
    li      a0, 0
    j       7f
8:  li      a0, 1
    j       7f
9:  li      a0, 2
7:  slli    a0, a0, 1
    ori     a0, a0, 1
    la      t0, tohost
5:  ld      t1, 0(t0)
    bnez    t1, 5b
    sd      a0, 0(t0)
6:  j       6b

    .section .data.s0harness, "aw"
    .p2align 3
s0_save:
    .dword  0, 0
    .section .htif, "aw"
    .p2align 6
    .globl tohost, fromhost
tohost:
    .dword  0
fromhost:
    .dword  0
