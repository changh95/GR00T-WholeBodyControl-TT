/* Constants shared by the assembly files (crt0.S, stage0.S, park_stub.S, stage0_spike.S) and, for the multi-hart
 * region, by mh.h. Plain #defines only: this file is included from both C and assembly. */
#ifndef X280_ASM_H
#define X280_ASM_H

/* mstatus fields (RISC-V privileged spec): FS bits 14:13, VS bits 10:9 (01 = Initial), MIE bit 3,
 * MPRV bit 17. */
#define MSTATUS_MIE          (1 << 3)
#define MSTATUS_VS_INITIAL   (1 << 9)
#define MSTATUS_FS_INITIAL   (1 << 13)
#define MSTATUS_MPRV         (1 << 17)
#define MIE_MSIE             (1 << 3)
#define MIE_MTIE             (1 << 7)

/* Multi-hart region (mh.h, lim.ld .mh): a fixed 8 KiB of LIM between the monitor's .persist and its stack, outside
 * .bss (hart 0's crt0 clears .bss on every entry, also while harts 1-3 run). Per hart h = 1..3 (X280 and x86):
 *   stat/aux lines   MH_BASE + 0x80 h (+0x40)        written by hart h only (hart 0 writes the ctl line at +0)
 *   trap stack top   MH_BASE + 0x400 + 0x100 h       256 B each: mh_sec_trap_entry
 *   private block    MH_BASE + 0x800 h               256 B: the running job's guard (struct mh_priv)
 *   stack top        MH_BASE + 0x800 (h + 1)         2 KiB area minus the private block
 * MH_SEC_RNMI_EXC is the RNMI exception trap handler address of harts 1-3 under `x280d serve --harts 4`: a
 * `wfi; j .` spin at a fixed address inside crt0's .text.init (crt0.S .org; asserted in lim.ld and regs.py). */
#if !defined(SPIKE_HOST)
#define MH_BASE              0x0800A000
#define MH_SIZE              0x2000
#define MH_TRAP_STACK_BASE   (MH_BASE + 0x400)
#define MH_TRAP_SHIFT        8                  /* 256 B trap stack per hart */
#define MH_STACK_BASE        (MH_BASE + 0x800)
#define MH_STACK_SHIFT       11                 /* 2 KiB per hart: 256 B private block + stack */
#else
/* SPIKE_HOST only: the scripted host's code, script table, payload copies and buffers fill the monitor region up to
 * ~0x0800_E600, so spike runs a compact 4 KiB geometry at 0x0800_E800 under a 2 KiB hart-0 stack (lim.ld MH_SPIKE).
 * The line layout (ctl, stat, aux) is the same; the trap stacks are 128 B and the per-hart areas 1 KiB. */
#define MH_BASE              0x0800E800
#define MH_SIZE              0x1000
#define MH_TRAP_STACK_BASE   (MH_BASE + 0x200)
#define MH_TRAP_SHIFT        7
#define MH_STACK_BASE        (MH_BASE + 0x400)
#define MH_STACK_SHIFT       10
#endif
#define MH_SEC_RNMI_EXC_OFF  0x200
#define MH_SEC_RNMI_EXC      (0x08000000 + MH_SEC_RNMI_EXC_OFF)

/* External peripherals as the hart sees them (MemoryMap.md:111-119), for the assembly paths. */
#define PLAT_XP_BASE_HI20    0x20010        /* lui value: 0x2001_0000 */
#define PLAT_RNMI_TRIGGER_LO 0x414          /* 0x2001_0414 - 0x2001_0000 */
#define PLAT_RNMI_HANDLER_LO 0x418          /* hart h: + 16 h */
#define PLAT_CLINT_HI20      0x2000         /* lui value: CLINT 0x0200_0000 (msip at +4 h) */

#endif
