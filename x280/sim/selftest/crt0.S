/* Start-up and trap entry of the spike self-test images.
 *
 * Hart 0 installs the trap vector, turns on mstatus.FS/VS (Initial), zeroes .bss and runs main(); its return
 * value becomes the HTIF exit code. Other harts call secondary_main(hartid) (weak: wfi loop) on their own 4 KiB
 * of the stack. Unlike the firmware crt0 (PLAN.md WP-3.2) this does NOT touch CSR 0x7c1 or 0x353: spike traps
 * them (t_csr.c measures exactly which ones).
 */
    .section .text.init, "ax"
    .globl _start
_start:
    la      t0, trap_entry
    csrw    mtvec, t0
    li      t1, (1 << 9) | (1 << 13)    /* mstatus.VS = Initial (bits 10:9), FS = Initial (bits 14:13) */
    csrs    mstatus, t1
    csrr    a0, mhartid
    la      sp, __stack_top
    bnez    a0, 3f
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:  call    main
    call    htif_exit                   /* a0 = main's return value; does not return */
3:  slli    t0, a0, 12                  /* hart h: stack top - h * 4 KiB */
    sub     sp, sp, t0
    call    secondary_main
4:  wfi
    j       4b

    .weak   secondary_main
secondary_main:
    wfi
    j       secondary_main

/* Save every GPR except x0 into a 32 x 8 B frame on the current stack, call trap_handler(frame), restore
 * and mret. trap_handler may rewrite mepc (resume after the faulting instruction). */
    .align  2
    .globl  trap_entry
trap_entry:
    addi    sp, sp, -256
    sd      x1, 8(sp)
    sd      x3, 24(sp)
    sd      x4, 32(sp)
    sd      x5, 40(sp)
    sd      x6, 48(sp)
    sd      x7, 56(sp)
    sd      x8, 64(sp)
    sd      x9, 72(sp)
    sd      x10, 80(sp)
    sd      x11, 88(sp)
    sd      x12, 96(sp)
    sd      x13, 104(sp)
    sd      x14, 112(sp)
    sd      x15, 120(sp)
    sd      x16, 128(sp)
    sd      x17, 136(sp)
    sd      x18, 144(sp)
    sd      x19, 152(sp)
    sd      x20, 160(sp)
    sd      x21, 168(sp)
    sd      x22, 176(sp)
    sd      x23, 184(sp)
    sd      x24, 192(sp)
    sd      x25, 200(sp)
    sd      x26, 208(sp)
    sd      x27, 216(sp)
    sd      x28, 224(sp)
    sd      x29, 232(sp)
    sd      x30, 240(sp)
    sd      x31, 248(sp)
    mv      a0, sp
    call    trap_handler
    ld      x1, 8(sp)
    ld      x3, 24(sp)
    ld      x4, 32(sp)
    ld      x5, 40(sp)
    ld      x6, 48(sp)
    ld      x7, 56(sp)
    ld      x8, 64(sp)
    ld      x9, 72(sp)
    ld      x10, 80(sp)
    ld      x11, 88(sp)
    ld      x12, 96(sp)
    ld      x13, 104(sp)
    ld      x14, 112(sp)
    ld      x15, 120(sp)
    ld      x16, 128(sp)
    ld      x17, 136(sp)
    ld      x18, 144(sp)
    ld      x19, 152(sp)
    ld      x20, 160(sp)
    ld      x21, 168(sp)
    ld      x22, 176(sp)
    ld      x23, 184(sp)
    ld      x24, 192(sp)
    ld      x25, 200(sp)
    ld      x26, 208(sp)
    ld      x27, 216(sp)
    ld      x28, 224(sp)
    ld      x29, 232(sp)
    ld      x30, 240(sp)
    ld      x31, 248(sp)
    addi    sp, sp, 256
    mret
