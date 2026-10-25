  .section .text.init
  .globl _start
_start:
  csrr  t0, mhartid
  bnez  t0, park            # only hart 0 runs; others park
  csrwi 0x7c1, 0            # clear SiFive feature-disable chicken bits (ISA doc L2CPUTile/README.md:49)
  li    t1, (1<<9)|(1<<13)  # mstatus.VS=Initial(01<<9), FS=Initial(01<<13)
  csrs  mstatus, t1
  la    sp, __stack_top
  call  main
park:
  wfi
  j park
