/* Multi-hart layout of the X280 monitor (harts 1-3 of L2CPU3), shared by mh.c, monitor.c, the mh_bench payload, the
 * x86 shim and, through mailbox_layout.py, by x280d (which reads the stat lines straight from LIM).
 *
 * Why a fixed region and not .bss: hart 0's crt0 clears .bss on every warm entry (crt0.S), also while harts 1-3 run;
 * and the host reads the stat lines without a symbol table. lim.ld places the NOLOAD .mh section at MH_BASE and asserts
 * that it ends below the stack. On x86 the same addresses are inside the fake LIM (plat_ptr).
 *
 * One writer per 64 B line (mailbox.h rules): the ctl line is written by hart 0 only; stat[h], aux[h], the trap stack,
 * private block and stack of hart h by hart h only. Hart 0 dispatches a job with a seqlock on ctl.gen (odd while it
 * writes the job fields, MB_FENCE_REL around them), then sets msip[h]; hart h reads gen, MB_FENCE_ACQ, the fields,
 * MB_FENCE_ACQ, gen again. No AMOs anywhere: the A extension is implemented at L1D (Caches.md:34) and LIM is not
 * data-cacheable (Caches.md:26-31), so every flag is a plain store by its one writer.
 *
 * The mh_bench payload (mh_bench.c) is linked at MHB_BASE; its data lines (job line, one result line per hart) sit at
 * fixed addresses at its end, so x280d reads them without the ELF. [MHB_BASE, MHB_END) ends at the E1F entry-word line.
 */
#ifndef X280_MH_H
#define X280_MH_H

#include <stdint.h>
#include <stddef.h>
#include "asm.h"
#include "mailbox.h"

#define MH_NHARTS         4u
#define MH_SEC_MASK       0xEu                 /* harts 1-3 */
#define MH_END            (MH_BASE + MH_SIZE)
#define MH_CTL_ADDR       ((uint32_t)MH_BASE)
#define MH_STAT_ADDR(h)   ((uint32_t)MH_BASE + 0x80u * (uint32_t)(h))
#define MH_AUX_ADDR(h)    ((uint32_t)MH_BASE + 0x80u * (uint32_t)(h) + 0x40u)
#define MH_TRAP_TOP(h)    ((uint32_t)MH_TRAP_STACK_BASE + ((uint32_t)(h) << MH_TRAP_SHIFT))
#define MH_PRIV_ADDR(h)   ((uint32_t)MH_STACK_BASE + (((uint32_t)(h) - 1u) << MH_STACK_SHIFT))
#define MH_STACK_TOP(h)   ((uint32_t)MH_STACK_BASE + ((uint32_t)(h) << MH_STACK_SHIFT))

#define MH_CTL_MAGIC      0x4C54434Du          /* "MCTL" */
#define MH_STAT_MAGIC     0x5441544Du          /* "MSTA" */

/* mh_stat.state */
#define MH_ST_OFF         0u                   /* never adopted (or a line from before hart 0's epoch) */
#define MH_ST_STARTING    1u                   /* inside mh_secondary_main's adoption self-test */
#define MH_ST_IDLE        2u                   /* idle loop: ready for a job */
#define MH_ST_BUSY        3u                   /* running job done_gen + 2 */
#define MH_ST_TRAPPED     4u                   /* a trap outside any job: back in the idle loop, see aux */

/* mh_stat.mode: how the idle loop waits */
#define MH_MODE_WFI       1u                   /* wfi with mie = MSIE | MTIE, mstatus.MIE = 0: no trap, wake only */
#define MH_MODE_POLL      2u                   /* the self-IPI did not show in mip or did not wake wfi: ~5 us polls */

#define MIP_MSIP          (1u << 3)
#define MIP_MTIP          (1u << 7)

struct mh_ctl {            /* MH_CTL_ADDR, hart 0 only */
    uint32_t magic;        /*  0  MH_CTL_MAGIC from hart 0's cold entry on */
    uint32_t epoch;        /*  4  stamp of that cold entry: a stat line is current only if it echoes it */
    uint32_t gen;          /*  8  seqlock: odd while hart 0 writes the job fields, then the job's (even) generation */
    uint32_t mask;         /* 12  harts asked to run job `gen` (bits 1-3) */
    uint64_t fn;           /* 16  uint64_t fn(uint64_t arg, uint32_t hart, const struct mh_wapi *) */
    uint64_t arg;          /* 24 */
    uint64_t t_dispatch;   /* 32  mtime just before gen was written */
    uint32_t n_dispatch;   /* 40  jobs dispatched since the cold entry */
    uint32_t rsvd[5];      /* 44 */
};

struct mh_stat {           /* MH_STAT_ADDR(h), hart h only */
    uint32_t magic;        /*  0  MH_STAT_MAGIC */
    uint32_t hartid;       /*  4  mhartid */
    uint32_t epoch;        /*  8  ctl.epoch at the adoption */
    uint32_t state;        /* 12  MH_ST_* */
    uint64_t heartbeat;    /* 16  mtime of the last idle-loop pass */
    uint32_t entries;      /* 24  RNMI entries (adoptions) in this epoch */
    uint32_t done_gen;     /* 28  last job generation completed (result in aux.result) */
    uint32_t trap_gen;     /* 32  last job generation that trapped (dump in aux) */
    uint32_t traps;        /* 36  traps taken in this epoch */
    uint64_t mncause;      /* 40  at the last adoption: 2^63 | 2 for the trigger-bit RNMI (RNMIs.md:76-77) */
    uint64_t mnepc;        /* 48  at the last adoption (stage 0's wfi loop 0x2001_013A on an E1F boot) */
    uint32_t mode;         /* 56  MH_MODE_* */
    uint32_t mip;          /* 60  mip right after the adoption self-IPI (bit 3 MSIP) */
};

struct mh_aux {            /* MH_AUX_ADDR(h), hart h only */
    uint64_t mcause;       /*  0  last trap dump */
    uint64_t mepc;         /*  8 */
    uint64_t mtval;        /* 16 */
    uint64_t result;       /* 24  return value of job done_gen */
    uint64_t t_seen;       /* 32  mtime when the hart picked up job gen_seen */
    uint32_t gen_seen;     /* 40 */
    uint32_t wakes;        /* 44  idle-loop passes since the adoption */
    uint32_t beu_accrued;  /* 48  own BEU accrued register at the adoption (0x0170_0000 + 0x1000 h + 0x20) */
    uint32_t wfi_ticks;    /* 52  self-test: mtime ticks spent in the wfi after the self-IPI */
    uint32_t fence_i;      /* 56  fence.i executed before payload jobs */
    uint32_t rsvd;         /* 60 */
};

/* Worker services (restricted: no heartbeat, no result: those write L1 / L39, hart 0's lines). */
struct mh_wapi {
    uint32_t version;      /* MH_WAPI_VERSION */
    uint32_t size;
    uint64_t mtime_hz;
    uint64_t (*mtime)(void);
    uint64_t (*mcycle)(void);
    void    *(*ptr)(uint64_t lim_phys);
    uint32_t (*beu_accrued)(uint32_t hart);   /* that hart's BEU accrued register; call it for your own hart */
    uint32_t rsvd;
};
#define MH_WAPI_VERSION 1u

typedef uint64_t (*mh_fn)(uint64_t arg, uint32_t hart, const struct mh_wapi *w);

/* ---- mh_bench payload (mh_bench.c, mh_bench.ld): code at MHB_BASE, fixed data lines at its end ---------------- */
#define MHB_BASE          0x081DF400u          /* after HOST_BENCH_BASE's 1000-iteration array (ends 0x081D_F3A0) */
#define MHB_DATA          0x081DFE40u          /* job line (hart 0); 2,624 B of code before it */
#define MHB_RES(h)        (MHB_DATA + 0x40u + 0x40u * (uint32_t)(h))   /* result line of hart h (hart h only) */
#define MHB_END           0x081DFF80u          /* = the E1F entry-word line: never reached */
#define MHB_MAX_BYTES     (MHB_END - MHB_BASE) /* 2,944 B incl. the data lines */

/* CALL arguments: policy = kernel [7:0] | hart-0 kernel override [15:8] (0 = same); flags = hart mask [3:0] |
 * pattern [7:4] | k [11:8] (P4 bank bit; K_TRAP: the hart that traps) | mode [13:12]; tick = window in mtime ticks
 * (0 = MHB_WINDOW_US); arg_addr / arg_len = working set (0 = MHB_WS_DEFAULT_*). */
#define MHB_K_LD8X8       1u    /* 8 independent ld per line */
#define MHB_K_LD8         2u    /* 1 ld per line */
#define MHB_K_CHAIN       3u    /* dependent chain, one load per line (ld; and zero; add) */
#define MHB_K_LW16        4u    /* 16 lw per line */
#define MHB_K_VLE8_M1     5u    /* vle8.v m1 = 64 B, accumulated with vadd.vv (e8) */
#define MHB_K_VLE8_M2     6u    /* 128 B */
#define MHB_K_VLE8_M4     7u    /* 256 B */
#define MHB_K_VLE8_M8     8u    /* 512 B */
#define MHB_K_MLP         9u    /* gemv_f16_rvv's inner loop on one 64-column slice: flw x + vle16 m2 + vwmul + vand +
                                   vfmul + vfmul + vfadd per 512 B row (wbc_mlp.c vload_f16 / VMAC) */
#define MHB_K_MLP_ALU     10u   /* the same vector ops on registers only */
#define MHB_K_ALU         11u   /* a 64-instruction ALU block per access, no LIM */
#define MHB_K_MLP_LD      12u   /* K9's loads only (flw x + vle16 m2 per row) */
#define MHB_K_TRAP        15u   /* tests only: hart k executes an illegal instruction, the others run K11 */
#define MHB_N_KERNELS     12u

#define MHB_P_PRIVATE     0u    /* hart h: quarter h of the working set, contiguous */
#define MHB_P_SAME_BANK   1u    /* quarter h, units at stride 4U, offset 0 (lines = 0 mod 4 for U = 64) */
#define MHB_P_OWN_BANK    2u    /* quarter h, stride 4U, offset h U (lines = h mod 4) */
#define MHB_P_SHARED      3u    /* every hart quarter 0, contiguous */
#define MHB_P_SWEEP       4u    /* whole set, stride 2^(k+2), offset h 2^k, k = 6..12 */

#define MHB_MODE_RUN      0u    /* dispatch, run hart 0's share, wait, summarise */
#define MHB_MODE_START    1u    /* dispatch to harts 1-3 and return (background; window <= 1 s) */
#define MHB_MODE_COLLECT  2u    /* wait for the background job, summarise */

#define MHB_OK            0u
#define MHB_EKERNEL       1u
#define MHB_ERANGE        2u    /* working set outside [MB_PAYLOAD_BASE, MB_LIM_END), misaligned or over the payload */
#define MHB_EPATTERN      3u
#define MHB_EAPI          4u    /* mon_api older than version 2 */
#define MHB_EWINDOW       5u    /* window 0 or > 1 s */
#define MHB_ENOVECTOR     6u
#define MHB_ENOJOB        7u    /* COLLECT without a background job */
#define MHB_EMODE         8u

#define MHB_WINDOW_US     500u
#define MHB_LEAD_US       50u   /* t_start = mtime + 50 us: every hart spins on CLINT mtime until then */
#define MHB_WS_DEFAULT_BASE 0x08012000u
#define MHB_WS_DEFAULT_LEN  0x001C0000u        /* 1.75 MiB of the registered blob (ends 0x081D_2000) */
#define MHB_WS_ALIGN      0x10000u             /* working set: 64 KiB aligned length, 64 B aligned base */

struct mhb_job {           /* MHB_DATA, hart 0 only */
    uint32_t tag;          /*  0  CALL seq of the run (result lines carry it) */
    uint32_t kernel;       /*  4  harts 1-3 */
    uint32_t kernel0;      /*  8  hart 0 */
    uint32_t pattern;      /* 12 */
    uint32_t kbit;         /* 16 */
    uint32_t mask;         /* 20  harts asked */
    uint64_t ws_base;      /* 24 */
    uint64_t ws_len;       /* 32 */
    uint64_t t_start;      /* 40  mtime */
    uint64_t t_end;        /* 48  mtime */
    uint32_t mode;         /* 56 */
    uint32_t dispatched;   /* 60  harts 1-3 the monitor dispatched (mh_run) */
};

struct mhb_res {           /* MHB_RES(h), hart h only */
    uint32_t hart;         /*  0 */
    uint32_t status;       /*  4  MHB_* */
    uint32_t tag;          /*  8  = job.tag when current */
    uint32_t kernel;       /* 12 */
    uint64_t bytes;        /* 16  bytes loaded */
    uint64_t accesses;     /* 24  units (lines, vector loads, rows, ALU blocks) */
    uint64_t cycles;       /* 32  own mcycle from the start barrier to the end */
    uint64_t t0;           /* 40  mtime at the start (after the barrier) */
    uint32_t dt;           /* 48  t1 - t0, mtime ticks */
    uint32_t beu_accrued;  /* 52  own BEU accrued at the end */
    uint64_t sink;         /* 56  checksum of what was read (mhb_sink in mailbox_layout.py) */
};

struct mhb_summary {       /* L39 bytes 24..63 (mon_api.result), hart 0 */
    uint32_t kernel;       /* 24 */
    uint32_t status;       /* 28 */
    uint32_t mask;         /* 32  asked */
    uint32_t done;         /* 36  harts whose result line carries the tag (bit 0 = hart 0) */
    uint64_t bytes;        /* 40  over the done harts */
    uint32_t span;         /* 48  max t1 - min t0 */
    uint32_t start_skew;   /* 52  max t0 - min t0 */
    uint32_t end_skew;     /* 56  max t1 - min t1 */
    uint32_t tag;          /* 60 */
};

#define MH_SA(t, f, o) _Static_assert(offsetof(struct t, f) == (o), #t "." #f " != " #o)
MH_SA(mh_ctl, gen, 8);        MH_SA(mh_ctl, fn, 16);       MH_SA(mh_ctl, arg, 24);     MH_SA(mh_ctl, t_dispatch, 32);
MH_SA(mh_ctl, n_dispatch, 40);
MH_SA(mh_stat, state, 12);    MH_SA(mh_stat, heartbeat, 16); MH_SA(mh_stat, done_gen, 28); MH_SA(mh_stat, traps, 36);
MH_SA(mh_stat, mncause, 40);  MH_SA(mh_stat, mnepc, 48);   MH_SA(mh_stat, mode, 56);   MH_SA(mh_stat, mip, 60);
MH_SA(mh_aux, result, 24);    MH_SA(mh_aux, t_seen, 32);   MH_SA(mh_aux, gen_seen, 40); MH_SA(mh_aux, fence_i, 56);
MH_SA(mhb_job, t_start, 40);  MH_SA(mhb_job, dispatched, 60);
MH_SA(mhb_res, bytes, 16);    MH_SA(mhb_res, t0, 40);      MH_SA(mhb_res, dt, 48);     MH_SA(mhb_res, sink, 56);
MH_SA(mhb_summary, kernel, 0); MH_SA(mhb_summary, bytes, 16); MH_SA(mhb_summary, tag, 36);
_Static_assert(sizeof(struct mh_ctl) == MB_LINE && sizeof(struct mh_stat) == MB_LINE, "ctl/stat line");
_Static_assert(sizeof(struct mh_aux) == MB_LINE && sizeof(struct mhb_job) == MB_LINE, "aux/job line");
_Static_assert(sizeof(struct mhb_res) == MB_LINE && sizeof(struct mhb_summary) == 40, "result line, summary");
_Static_assert(MH_STAT_ADDR(3) + 0x80u <= (uint32_t)MH_TRAP_STACK_BASE, "stat/aux lines below the trap stacks");
_Static_assert(MH_TRAP_TOP(3) <= (uint32_t)MH_STACK_BASE, "trap stacks below the stacks");
_Static_assert(MH_STACK_TOP(3) == MH_END, "three stack areas end the region");
_Static_assert(MH_BASE % 0x800 == 0 && MH_END <= MB_CODE_END, ".mh inside the monitor region");
_Static_assert(MHB_RES(3) + MB_LINE <= MHB_END && MHB_END == 0x081DFF80u, "mh_bench data lines");

#endif /* X280_MH_H */
