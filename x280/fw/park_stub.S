/* GP-scratch park stub, 64 B at 0x2001_0100 (PLAN.md WP-3.2, §5.2 PARK protocol).
 *
 * The host writes this blob into the 64 B general-purpose scratch (L2CPUTile/MemoryMap.md:113), which is
 * executable via L1I (Caches.md:18-19), and points every hart's RNMI trap handler address at +0 and its
 * RNMI exception trap handler address at +48 (0x2001_0418 + 16h and +8, RNMIs.md:48-59).
 *
 * +0  park (<= 48 B): fence.i (the host may have changed code in LIM: only the hart itself can invalidate
 *     its L1I, Caches.md:22); spin while bit h of the RNMI trigger word 0x2001_0414 is set (RNMIs.md:38-46;
 *     the cause is level-triggered, RNMIs.md:15); then jump to crt0 at the fixed 0x0800_0000 (lim.ld:63
 *     asserts _start there), which sees mnstatus.NMIE = 0 (warm entry) and re-arms with csrsi 0x353, 8.
 *     The ISA's suggested stub (README.md:30) loads the per-hart reset handler address instead; this one does
 *     not, because an E1F boot leaves that address at the stage-0 stub 0x2001_0100 = this stub, where the
 *     hart would loop with NMIE = 0 forever (review F1), and the ISA only defines the address as the pc taken
 *     out of reset (README.md:46). The spin gives the host time to replace code before the hart runs again.
 * +48 spin (16 B): the RNMI exception handler. An exception inside the park stub is unrecoverable
 *     (NMIE = 0), so the hart waits in wfi; the host sees it in the wfi status bits (0x2001_0400,
 *     MemoryMap.md:115).
 * Register use is free: nothing returns to the interrupted context.
 */
    .section .park, "ax"
    .option norelax
    .globl park_stub
park_stub:
    fence.i
    csrr    a1, mhartid
    lui     a0, 0x20010             /* a0 = 0x2001_0000 */
    li      a2, 1
    sll     a2, a2, a1              /* a2 = 1 << h */
1:  lw      a3, 0x414(a0)           /* RNMI trigger bits */
    and     a3, a3, a2
    bnez    a3, 1b
    lui     a0, 0x08000             /* a0 = 0x0800_0000: crt0 _start (lim.ld:63), never the reset vector */
    jr      a0
    .globl park_stub_end
park_stub_end:
    .org    park_stub + 48          /* assembler error if the stub grew past 48 B */
    .globl park_spin
park_spin:
    wfi
    j       park_spin
    .org    park_stub + 64
