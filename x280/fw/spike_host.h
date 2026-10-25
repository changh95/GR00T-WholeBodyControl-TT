/* SPIKE_HOST scripted host (contract in SPIKE_HOST.md).
 *
 * Why: spike runs one image and has no PCIe host, so under -DSPIKE_HOST the monitor's idle loop is
 * driven by a command list compiled into the image. The runner plays x280d (writes host lines in the
 * §1.4.4 order, waits for seq_done) and prints every result as raw mailbox lines; the harness judges.
 * Results leave through HTIF (riscv-isa-sim fesvr/syscall.cc: openat/read/write/close via magic memory).
 */
#ifndef X280_SPIKE_HOST_H
#define X280_SPIKE_HOST_H

#include <stdint.h>
#include "mailbox.h"

enum sh_op {
    SH_END = 0,
    SH_CMD,            /* one command from the fields below, `count` times (default 1)                  */
    SH_LOAD_BYTES,     /* host-write [bytes, bytes_end) at dst (zero-padded to 64 B), L37, LOAD `kind`  */
    SH_LOAD_FILE,      /* the same with the bytes of file `path` (HTIF read straight into LIM)          */
    SH_INFER_CONST,    /* `count` INFERs on xorshift obs seeded by tick + i (SPIKE_HOST.md)              */
    SH_INFER_FILE,     /* INFER each 2,064 B f32[516] record of `path`; append L35+L1 (128 B) to `path2`  */
    SH_CORRUPT_INFER,  /* INFER whose obs_crc32 is off by one: expect FAULT / MB_ERR_CRC               */
    SH_POKE32,         /* host 4 B write of `arg_len` at `dst` (e.g. L38 = GONE, as the KMD cleanup)     */
    SH_PARK,           /* PARK command; spike stands in for the host's RNMI right away               */
    SH_CALL_HANG,      /* CALL arg_entry with a timer standing in for the RNMI after delay_us, `count` x */
    SH_MH_ADOPT,       /* harts `flags` (0 = 1-3): the --harts 4 RNMI vectors into the RAM page, trigger bits, IPIs (the  */
                       /* spike stand-in for the RNMI's wake), delay_us, bits cleared; wait for mh_ready (X280 regions)  */
    SH_LOAD_PATTERN,   /* host-write arg_len bytes of the xorshift64 pattern seeded by tick at dst, then LOAD kind DATA */
    SH_MHB_SWEEP,      /* CALL mh_bench (MHB_BASE) for every kernel x pattern x k x mask of `sweep`                      */
};

/* `report` bits: extra lines after the step line (SPIKE_HOST.md) */
#define SH_REP_MH   1u     /* SH mh: the ctl line and the stat/aux lines of harts 1-3 */
#define SH_REP_MHB  2u     /* SH mhb: mh_bench's job line and the four result lines */

struct sh_sweep {
    const uint8_t *kernels, *patterns, *kbits, *masks;
    uint8_t nk, np, nkb, nm;      /* nkb 0 = one k of 0 */
    uint8_t k0, mode;             /* hart-0 kernel override, MHB_MODE_* */
};

struct sh_step {
    uint32_t op;
    uint32_t cmd, policy, flags;
    int64_t  tick;
    uint64_t arg_addr, arg_entry;
    uint32_t arg_len, count;
    uint64_t dst;
    uint32_t kind, delay_us;
    const uint8_t *bytes, *bytes_end;
    const char *path, *path2;
    const char *name;
    const struct sh_sweep *sweep;
    uint32_t report;
};

extern const struct sh_step sh_script[];

/* spike_payloads.S: payload images embedded with .incbin (built by the Makefile first). */
extern const uint8_t sh_bin_checksum[], sh_bin_checksum_end[];
extern const uint8_t sh_bin_illegal[], sh_bin_illegal_end[];
extern const uint8_t sh_bin_hang[], sh_bin_hang_end[];
extern const uint8_t sh_bin_lim_bench[], sh_bin_lim_bench_end[];
extern const uint8_t sh_bin_park[], sh_bin_park_end[];
extern const uint8_t sh_bin_mh_bench[], sh_bin_mh_bench_end[];   /* SPIKE_MH builds only */

#define SH_BYTES(sym) .bytes = sym, .bytes_end = sym##_end

#endif /* X280_SPIKE_HOST_H */
