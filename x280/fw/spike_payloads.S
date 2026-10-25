/* Payload images for the SPIKE_HOST runner (spike_host.h). Paths come from the Makefile (SH_BIN_*),
 * which builds the payload .bin files and build/park_stub.bin before this file. Read-only data. A SPIKE_MH build
 * embeds only the park stub and mh_bench (the multi-hart script needs no other payload, and space is short). */
    .section .rodata.sh_bins, "a"
#if defined(SH_BIN_CHECKSUM)
    .p2align 6
    .globl sh_bin_checksum, sh_bin_checksum_end
sh_bin_checksum:
    .incbin SH_BIN_CHECKSUM
sh_bin_checksum_end:
#endif
#if defined(SH_BIN_ILLEGAL)
    .p2align 6
    .globl sh_bin_illegal, sh_bin_illegal_end
sh_bin_illegal:
    .incbin SH_BIN_ILLEGAL
sh_bin_illegal_end:
#endif
#if defined(SH_BIN_HANG)
    .p2align 6
    .globl sh_bin_hang, sh_bin_hang_end
sh_bin_hang:
    .incbin SH_BIN_HANG
sh_bin_hang_end:
#endif
#if defined(SH_BIN_LIM_BENCH)
    .p2align 6
    .globl sh_bin_lim_bench, sh_bin_lim_bench_end
sh_bin_lim_bench:
    .incbin SH_BIN_LIM_BENCH
sh_bin_lim_bench_end:
#endif
    .p2align 6
    .globl sh_bin_park, sh_bin_park_end
sh_bin_park:
    .incbin SH_BIN_PARK
sh_bin_park_end:
#if defined(SH_BIN_MH_BENCH)
    .p2align 6
    .globl sh_bin_mh_bench, sh_bin_mh_bench_end
sh_bin_mh_bench:
    .incbin SH_BIN_MH_BENCH
sh_bin_mh_bench_end:
#endif
