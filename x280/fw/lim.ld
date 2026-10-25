/* Link script of the X280 monitor image (PLAN.md WP-3.2, §1.4.4).
 *
 * LIM = L3 as uncached scratchpad, 0x0800_0000 (tt-isa-documentation BlackholeA0/L2CPUTile/MemoryMap.md:84),
 * 15 x 128 KiB = 0x1E_0000 B at reset (Caches.md:73, 92). The monitor owns the first 64 KiB:
 *   0x0800_0000  .text (crt0 first: the reset handler address), .rodata, .data   = the loadable image
 *                .bss (cleared by crt0 on every entry), .persist (survives warm entries)
 *   0x0800_A000  .mh (NOLOAD, 8 KiB, mh.h): harts 1-3 ctl/stat lines, trap stacks, stacks; never cleared by crt0
 *                stack, __stack_size (16 KiB default) ending at 0x0801_0000
 *   0x0801_0000  mailbox, 64 lines (mailbox.h); 0x0801_1000 weights / payloads.
 * Nothing in LIM is data-cacheable (Caches.md:26-31); the stack is uncached too.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

__stack_size = DEFINED(__stack_size) ? __stack_size : 0x4000;
/* asm.h MH_BASE / MH_SIZE: 8 KiB at 0x0800_A000; SPIKE_HOST links (--defsym=MH_SPIKE=1) the compact 4 KiB at
 * 0x0800_E800 below their 2 KiB stack */
MH_BASE = DEFINED(MH_SPIKE) ? 0x0800E800 : 0x0800A000;
MH_SIZE = DEFINED(MH_SPIKE) ? 0x1000 : 0x2000;
__trap_stack_size = 0x400;

MEMORY
{
    LIM (rwx) : ORIGIN = 0x08000000, LENGTH = 0x1E0000
}

SECTIONS
{
    . = 0x08000000;
    .text : {
        KEEP(*(.text.init))
        *(.text .text.*)
    } > LIM
    .rodata : ALIGN(64) {
        *(.rodata .rodata.* .srodata .srodata.*)
    } > LIM
    .data : ALIGN(64) {
        *(.data .data.* .sdata .sdata.*)
        *(.htif)
        . = ALIGN(64);
    } > LIM
    __image_end = .;

    .bss (NOLOAD) : ALIGN(64) {
        __bss_start = .;
        *(.sbss .sbss.* .bss .bss.* COMMON)
        . = ALIGN(16);
        . += __trap_stack_size;
        __trap_stack_top = .;
        . = ALIGN(64);
        __bss_end = .;
    } > LIM
    .persist (NOLOAD) : ALIGN(64) {
        __persist_start = .;
        KEEP(*(.persist))
        . = ALIGN(64);
        __persist_end = .;
    } > LIM

    .mh MH_BASE (NOLOAD) : {
        __mh_start = .;
        . += MH_SIZE;
        __mh_end = .;
    } > LIM

    __stack_top = 0x08010000;
    __stack_bottom = __stack_top - __stack_size;
    __mailbox = 0x08010000;
    __payload_base = 0x08011000;

    /DISCARD/ : { *(.eh_frame .eh_frame_hdr .comment .note .note.*) }
}

ASSERT(_start == 0x08000000, "_start must be the reset handler address 0x0800_0000")
ASSERT(__image_end - 0x08000000 <= 48 * 1024, "loadable image exceeds 48 KiB (PLAN.md T-3.2)")
ASSERT(__persist_end <= __stack_bottom, "bss + persist overlap the stack")
ASSERT(__persist_end <= __mh_start, "bss + persist overlap the multi-hart region .mh (mh.h MH_BASE)")
ASSERT(__mh_end <= __stack_bottom, ".mh overlaps the stack")
ASSERT(DEFINED(MH_SPIKE) || (__mh_start == 0x0800A000 && __mh_end == 0x0800C000),
       ".mh must be [0x0800_A000, 0x0800_C000) (mh.h, regs.py)")
ASSERT(mh_sec_rnmi_exc == 0x08000200, "mh_sec_rnmi_exc must be the fixed MH_SEC_RNMI_EXC 0x0800_0200 (regs.py)")
ASSERT(__stack_top <= __mailbox, "stack overlaps the mailbox")
