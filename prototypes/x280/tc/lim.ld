OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { LIM (rwx) : ORIGIN = 0x08000000, LENGTH = 0x1E0000 } /* 15 x 128 KiB LIM at reset */
SECTIONS {
  .text : { *(.text.init) *(.text*) } > LIM
  .rodata : ALIGN(64) { *(.rodata*) *(.srodata*) } > LIM
  .data : ALIGN(64) { *(.data*) *(.sdata*) } > LIM
  .bss (NOLOAD) : ALIGN(64) { *(.bss*) *(.sbss*) *(COMMON) } > LIM
  . = ALIGN(16); . += 0x4000; __stack_top = .;
}
