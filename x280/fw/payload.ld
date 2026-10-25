/* Link script of LOAD/CALL payloads (lim_bench, test payloads; PLAN.md WP-3.2 "Payloads").
 *
 * A payload is one loadable blob placed by the host at PAYLOAD_ADDR (default 0x0801_1000, the first byte
 * after the mailbox, mailbox.h MB_PAYLOAD_BASE) with LOAD kind MB_LOAD_CODE, then FENCE_I, then CALL with
 * arg_entry = PAYLOAD_ADDR: `payload_entry` is linked first. .bss is emitted as zeros (objcopy
 * --set-section-flags) so the LOAD CRC covers it. Payloads run on the monitor's stack and call only the
 * mon_api function pointers they receive, so they have no undefined symbols.
 */
OUTPUT_ARCH(riscv)
ENTRY(payload_entry)

PAYLOAD_ADDR = DEFINED(PAYLOAD_ADDR) ? PAYLOAD_ADDR : 0x08011000;

SECTIONS
{
    . = PAYLOAD_ADDR;
    .text : {
        KEEP(*(.text.payload_entry))
        *(.text .text.*)
    }
    .rodata : ALIGN(8) { *(.rodata .rodata.* .srodata .srodata.*) }
    .data : ALIGN(8) { *(.data .data.* .sdata .sdata.*) }
    .bss : ALIGN(64) { *(.sbss .sbss.* .bss .bss.* COMMON) . = ALIGN(64); }
    __payload_end = .;
    /DISCARD/ : { *(.eh_frame .eh_frame_hdr .comment .note .note.*) }
}

ASSERT(payload_entry == PAYLOAD_ADDR, "payload_entry must be the first byte of the payload")
ASSERT(PAYLOAD_ADDR >= 0x08011000, "payloads live above the mailbox")
ASSERT(__payload_end <= 0x081E0000, "payload runs past the end of LIM")
