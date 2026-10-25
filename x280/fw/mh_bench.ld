/* Link script of the mh_bench payload (mh_bench.c; addresses in mh.h).
 *
 * Code at MHB_BASE 0x081D_F400 (after HOST_BENCH_BASE's BENCH array, which ends at 0x081D_F3A0 for 1000 iterations;
 * after the fp16 blob, which ends at 0x081D_E340), then the fixed data lines at MHB_DATA 0x081D_FE40: the job line
 * (hart 0) and one result line per hart (MHB_RES(h), hart h). The image is emitted up to MHB_END 0x081D_FF80, the E1F
 * entry-word line, so a LOAD rewrites the data lines too (zeros: no stale result line survives a LOAD). The payload
 * runs where it is linked (absolute data addresses), and `payload_entry` is its first byte.
 */
OUTPUT_ARCH(riscv)
ENTRY(payload_entry)

MHB_BASE = 0x081DF400;
MHB_DATA = 0x081DFE40;
MHB_END  = 0x081DFF80;

SECTIONS
{
    . = MHB_BASE;
    .text : {
        KEEP(*(.text.payload_entry))
        *(.text .text.*)
    }
    .rodata : ALIGN(8) { *(.rodata .rodata.* .srodata .srodata.*) }
    .data : ALIGN(8) { *(.data .data.* .sdata .sdata.*) }
    .bss : ALIGN(8) { *(.sbss .sbss.* .bss .bss.* COMMON) }
    __code_end = .;
    .mhb_data MHB_DATA : {
        FILL(0x00);
        . = MHB_END - 1;             /* lld: an absolute location */
        BYTE(0)
    }
    __payload_end = .;
    /DISCARD/ : { *(.eh_frame .eh_frame_hdr .comment .note .note.*) }
}

ASSERT(payload_entry == MHB_BASE, "payload_entry must be the first byte of mh_bench (MHB_BASE)")
ASSERT(__code_end <= MHB_DATA, "mh_bench code/data run into its job and result lines (MHB_DATA)")
ASSERT(__payload_end == MHB_END, "mh_bench must end exactly at MHB_END 0x081D_FF80 (the E1F entry-word line)")
