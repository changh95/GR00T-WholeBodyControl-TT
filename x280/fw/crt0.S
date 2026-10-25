/* Entry of the X280 monitor image at 0x0800_0000 (PLAN.md WP-3.2).
 *
 * Every entry comes here: the cold one out of reset (reset handler address = 0x0800_0000, §5.2 boot
 * step 5) and every warm one through the GP-scratch park stub, which jumps to the hart's reset handler
 * address after an RNMI (park_stub.S). So this code must be correct on any number of entries:
 *   - harts != 0 that come out of reset (mnstatus.NMIE = 1) park in wfi; one that arrives by RNMI (NMIE = 0: under
 *     `x280d serve --harts 4` its RNMI handler address is this _start) takes the secondary path below into mh.c;
 *   - csrw 0x7c1, 0 clears the SiFive feature-disable chicken bits (L2CPUTile/README.md:49);
 *   - mstatus.FS and .VS = Initial (bits 14:13 and 10:9), MIE/MPRV off, mie = 0, fcsr = 0 (RNE);
 *   - .bss is cleared on every entry (.persist is not: monitor.c decides cold/warm from mnstatus.NMIE);
 *   - mon_entry() counts the entry (n_reentry on a warm one);
 *   - csrsi 0x353, 8 re-arms RNMIs: mnstatus.NMIE (bit 3) is cleared by hardware on every RNMI and only
 *     software can set it again (RNMIs.md:19, 32, 84); the ISA's suggested stub never does (README.md:30).
 * X280 CSRs are used by number only (0x350-0x353, RNMIs.md:63-70; 0x7c1). Under SPIKE_HOST, where spike
 * traps those numbers (x280/sim/selftest/t_csr.c), they are replaced by calls into spike_host.c.
 * Note: llvm-objdump prints 0x350-0x353 with the standard Smcsrind names (miselect, mireg, mireg2,
 * mireg3), e.g. "csrsi mireg3, 0x8" for the re-arm; on the X280 these numbers are the RNMI CSRs.
 */
#include "asm.h"

    .section .text.init, "ax"
    .option norelax
    .globl _start
_start:
    csrr    t0, mhartid
    bnez    t0, .Lother_hart
#if !defined(SPIKE_HOST)
    csrwi   0x7c1, 0
#endif
    li      t1, MSTATUS_FS_INITIAL | MSTATUS_VS_INITIAL
    csrs    mstatus, t1
    li      t1, MSTATUS_MIE | MSTATUS_MPRV
    csrc    mstatus, t1
    csrw    mie, zero
    csrw    fcsr, zero
    la      t0, trap_entry
    csrw    mtvec, t0
    la      sp, __stack_top
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:  call    mon_entry
#if defined(SPIKE_HOST)
    call    plat_rnmi_rearm
#else
    csrsi   0x353, 8
#endif
    call    monitor_main

/* Harts 1-3 (mh.c, the 4-hart bench design A.2). t0 = mhartid. Out of reset (NMIE = 1) the hart waits in wfi forever
 * with mie = 0, as before: only an RNMI moves it on. Arriving by RNMI (NMIE = 0; handler address = _start, RNMI
 * exception handler = mh_sec_rnmi_exc below):
 *   1. fence.i: its L1I may still hold the stage-0 GP-scratch line of an E1F boot (Caches.md:18-22);
 *   2. spin while its own bit of 0x2001_0414 is set (the cause is level-triggered, RNMIs.md:15);
 *   3. csrwi 0x7c1, 0 (the chicken bits are per hart, README.md:49), FS/VS Initial, MIE/MPRV off,
 *      mie = MSIE | MTIE (wfi wakes; with mstatus.MIE = 0 nothing traps), fcsr = 0;
 *   4. mtvec = mh_sec_trap_entry, sp = MH_STACK_TOP(h) (mh.h; MH_BASE + 0x800 (h + 1) on the X280);
 *   5. csrsi 0x353, 8: NMIE re-armed, so the host can RNMI the hart again (re-adoption);
 *   6. mh_secondary_main(h), which never returns.
 * Under SPIKE_HOST the RNMI CSRs are spike_host.c's per-hart memory copies (sh_emu[h], 64 B apart), and with the
 * X280 regions mapped a cold hart takes the scripted host's emulated RNMI itself: woken by an IPI, it finds its
 * trigger bit set and jumps to its RNMI handler address (0x2001_0418 + 16 h) with the emulated NMIE cleared. */
.Lother_hart:
    li      t1, 4
    bgeu    t0, t1, .Lcold_wfi
#if defined(SPIKE_HOST)
    la      t5, sh_emu
    slli    t1, t0, 6
    add     t5, t5, t1               /* t5 = &sh_emu[h] {mnstatus, mncause, mnepc} */
    ld      t1, 0(t5)
#else
    csrr    t1, 0x353                /* mnstatus */
#endif
    andi    t1, t1, 8                /* NMIE */
    bnez    t1, .Lcold
    fence.i
    lui     t2, PLAT_XP_BASE_HI20
    li      t3, 1
    sll     t3, t3, t0               /* 1 << h */
1:  lw      t4, PLAT_RNMI_TRIGGER_LO(t2)
    and     t4, t4, t3
    bnez    t4, 1b
#if !defined(SPIKE_HOST)
    csrwi   0x7c1, 0
#endif
    li      t1, MSTATUS_FS_INITIAL | MSTATUS_VS_INITIAL
    csrs    mstatus, t1
    li      t1, MSTATUS_MIE | MSTATUS_MPRV
    csrc    mstatus, t1
    li      t1, MIE_MSIE | MIE_MTIE
    csrw    mie, t1
    csrw    fcsr, zero
    la      t1, mh_sec_trap_entry
    csrw    mtvec, t1
    slli    t1, t0, MH_STACK_SHIFT
    li      sp, MH_STACK_BASE
    add     sp, sp, t1               /* MH_STACK_TOP(h) */
#if defined(SPIKE_HOST)
    ld      t1, 0(t5)
    ori     t1, t1, 8                /* stands in for csrsi 0x353, 8 */
    sd      t1, 0(t5)
#else
    csrsi   0x353, 8
#endif
    mv      a0, t0
    call    mh_secondary_main
    j       .Lcold_wfi
.Lcold:
#if defined(SPIKE_HOST) && defined(SPIKE_X280_REGIONS)
    li      t1, MIE_MSIE
    csrw    mie, t1
    lui     t2, PLAT_XP_BASE_HI20
    lui     t6, PLAT_CLINT_HI20
    slli    t1, t0, 2
    add     t6, t6, t1               /* t6 = &msip[h] */
.Lsh_wait:
    wfi
    sw      zero, 0(t6)
    fence   iorw, iorw
    lw      t4, PLAT_RNMI_TRIGGER_LO(t2)
    srl     t4, t4, t0
    andi    t4, t4, 1
    beqz    t4, .Lsh_wait
    ld      t1, 0(t5)                /* emulated RNMI, taken only while NMIE = 1 (RNMIs.md:18-22) */
    andi    t3, t1, 8
    beqz    t3, .Lsh_wait
    andi    t1, t1, -9
    sd      t1, 0(t5)
    li      t1, 1
    slli    t1, t1, 63
    ori     t1, t1, 2
    sd      t1, 8(t5)                /* mncause = 2^63 | 2 */
    la      t1, .Lsh_wait
    sd      t1, 16(t5)               /* mnepc */
    slli    t1, t0, 4
    add     t1, t1, t2
    ld      t1, PLAT_RNMI_HANDLER_LO(t1)
    jr      t1
#else
    csrw    mie, zero
#endif
.Lcold_wfi:
    wfi
    j       .Lcold_wfi

/* RNMI exception trap handler address of harts 1-3 under --harts 4 (fixed: regs.MH_SEC_RNMI_EXC, lim.ld ASSERT). An
 * exception in the secondary path before the re-arm (NMIE = 0) lands here; the hart waits in wfi (the host sees no
 * stat line; nothing else can be done for it until a chip reset). */
    .org    MH_SEC_RNMI_EXC_OFF      /* assembler error if crt0's .text.init grew past it */
    .globl  mh_sec_rnmi_exc
mh_sec_rnmi_exc:
    wfi
    j       mh_sec_rnmi_exc

/* Direct-mode trap vector. Nothing is saved: mon_trap() never returns to the trapping context. It runs
 * on its own stack so that a trap caused by a bad sp is still reported. */
    .text
    .p2align 2
    .globl trap_entry
trap_entry:
    csrw    mscratch, sp
    la      sp, __trap_stack_top
    csrr    a0, mcause
    csrr    a1, mepc
    csrr    a2, mtval
    csrr    a3, mscratch
    call    mon_trap
3:  wfi
    j       3b

/* Harts 1-3: trap vector (mtvec). Nothing is saved: mh_sec_trap() longjmps into the running job's guard, or returns,
 * and the hart restarts its idle loop on a fresh stack. Trap stack top of hart h = MH_TRAP_TOP(h) (mh.h). */
    .p2align 2
    .globl mh_sec_trap_entry
mh_sec_trap_entry:
    csrw    mscratch, sp
    csrr    a3, mhartid
    slli    t1, a3, MH_TRAP_SHIFT
    li      sp, MH_TRAP_STACK_BASE
    add     sp, sp, t1
    csrr    a0, mcause
    csrr    a1, mepc
    csrr    a2, mtval
    call    mh_sec_trap
    csrr    a0, mhartid
    .globl  plat_mh_restart
plat_mh_restart:                     /* a0 = hart: fresh stack, idle loop again */
    slli    t1, a0, MH_STACK_SHIFT
    li      sp, MH_STACK_BASE
    add     sp, sp, t1
    call    mh_secondary_resume
5:  wfi
    j       5b

/* After a trap outside any guard: fresh stack, back into the poll loop (monitor.c). */
    .globl plat_restart_after_trap
plat_restart_after_trap:
    la      sp, __stack_top
    call    monitor_resume
4:  wfi
    j       4b

/* int mon_setjmp(uint64_t b[26]); void mon_longjmp(uint64_t b[26], int v): callee-saved state only. */
    .globl mon_setjmp
mon_setjmp:
    sd      ra, 0(a0)
    sd      sp, 8(a0)
    sd      s0, 16(a0)
    sd      s1, 24(a0)
    sd      s2, 32(a0)
    sd      s3, 40(a0)
    sd      s4, 48(a0)
    sd      s5, 56(a0)
    sd      s6, 64(a0)
    sd      s7, 72(a0)
    sd      s8, 80(a0)
    sd      s9, 88(a0)
    sd      s10, 96(a0)
    sd      s11, 104(a0)
    fsd     fs0, 112(a0)
    fsd     fs1, 120(a0)
    fsd     fs2, 128(a0)
    fsd     fs3, 136(a0)
    fsd     fs4, 144(a0)
    fsd     fs5, 152(a0)
    fsd     fs6, 160(a0)
    fsd     fs7, 168(a0)
    fsd     fs8, 176(a0)
    fsd     fs9, 184(a0)
    fsd     fs10, 192(a0)
    fsd     fs11, 200(a0)
    li      a0, 0
    ret

    .globl mon_longjmp
mon_longjmp:
    ld      ra, 0(a0)
    ld      sp, 8(a0)
    ld      s0, 16(a0)
    ld      s1, 24(a0)
    ld      s2, 32(a0)
    ld      s3, 40(a0)
    ld      s4, 48(a0)
    ld      s5, 56(a0)
    ld      s6, 64(a0)
    ld      s7, 72(a0)
    ld      s8, 80(a0)
    ld      s9, 88(a0)
    ld      s10, 96(a0)
    ld      s11, 104(a0)
    fld     fs0, 112(a0)
    fld     fs1, 120(a0)
    fld     fs2, 128(a0)
    fld     fs3, 136(a0)
    fld     fs4, 144(a0)
    fld     fs5, 152(a0)
    fld     fs6, 160(a0)
    fld     fs7, 168(a0)
    fld     fs8, 176(a0)
    fld     fs9, 184(a0)
    fld     fs10, 192(a0)
    fld     fs11, 200(a0)
    seqz    a0, a1
    add     a0, a0, a1      /* setjmp returns v, or 1 when v == 0 */
    ret
