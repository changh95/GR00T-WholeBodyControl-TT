/* Link script of build/spike/stage0_spike.elf (stage0_spike.S): stage 0 at its real address in the GP
 * scratch (L2CPUTile/MemoryMap.md:113), the timer harness and tohost after it in the same 4 KiB page,
 * which spike maps as plain RAM (-m...,0x20010000:0x1000). */
OUTPUT_ARCH(riscv)
ENTRY(s0_setup)
SECTIONS
{
    .stage0 0x20010100 : { KEEP(*(.stage0)) }
    .text 0x20010200 : { *(.text.s0harness) *(.text .text.*) }
    .data : { *(.data.s0harness) *(.data .data.*) }
    .htif : ALIGN(64) { *(.htif) }
    /DISCARD/ : { *(.eh_frame .comment .note .note.*) }
}
ASSERT(stage0_start == 0x20010100, "stage 0 must sit at the GP scratch")
ASSERT(stage0_end - stage0_start <= 64, "stage 0 > 64 B")
ASSERT(tohost < 0x20011000, "tohost must stay in the GP-scratch page")
