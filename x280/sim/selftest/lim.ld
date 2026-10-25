/* Link script for the spike tool's self-test images (tests/x280/test_spike_tool.py), not the firmware.
 *
 * LIM: L3 as uncached scratchpad at 0x0800_0000 (tt-isa-documentation BlackholeA0/L2CPUTile/MemoryMap.md:84),
 * 0x1E0000 B = 15 x 128 KiB at reset (PLAN.md WP-3.2 lim.ld).
 * GPX: the start of the external-peripheral block at 0x2001_0000 (MemoryMap.md:111-119): per-hart reset
 * handler addresses (+0x000, 4 x 8 B), GP scratch (+0x100, 64 B), RNMI trigger bits (+0x414), RNMI handler
 * addresses (+0x418, 4 x 16 B). Spike models this block only as plain RAM (run_spike.py --x280-regions).
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {
  LIM (rwx) : ORIGIN = 0x08000000, LENGTH = 0x1E0000
  GPX (rwx) : ORIGIN = 0x20010000, LENGTH = 0x458
}
SECTIONS {
  .text : { KEEP(*(.text.init)) *(.text .text.*) } > LIM
  .rodata : ALIGN(64) { *(.rodata .rodata.* .srodata .srodata.*) } > LIM
  .data : ALIGN(64) { *(.data .data.* .sdata .sdata.*) } > LIM
  .htif : ALIGN(64) { KEEP(*(.htif)) } > LIM
  .bss (NOLOAD) : ALIGN(64) { __bss_start = .; *(.bss .bss.* .sbss .sbss.* COMMON) . = ALIGN(8); __bss_end = .; } > LIM
  .stack (NOLOAD) : ALIGN(64) { . += 0x4000; __stack_top = .; } > LIM
  .gpx_resetvec 0x20010000 : { KEEP(*(.gpx.resetvec)) } > GPX
  .gpx_scratch 0x20010100 : { KEEP(*(.gpx.scratch)) } > GPX
  ASSERT(SIZEOF(.gpx_resetvec) <= 32, "reset vectors are 4 x 8 B (MemoryMap.md:111)")
  ASSERT(SIZEOF(.gpx_scratch) <= 64, "GP scratch is 64 B (MemoryMap.md:113)")
}
