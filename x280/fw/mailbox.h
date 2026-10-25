/* X280 LIM mailbox, layout v2 (PLAN.md §1.4.4), shared by the monitor (monitor.c), its x86 test build,
 * the SPIKE_HOST runner and, through mailbox_layout.py, by every host-side reader.
 *
 * Why a header and not only prose: the host (x280d over a WC/UC window) and the hart agree on byte
 * offsets, so every offset below is pinned by a _Static_assert and mirrored in mailbox_layout.py, which
 * tests/x280/test_mailbox_x86.py checks against the compiled C layout (mon_x86_layout()).
 *
 * Rules (§1.4.4): every LIM line is ECC-initialised by the host before any partial write; each 64 B line
 * has exactly one writer (H = host, X = this hart, K = the KMD cleanup write); the mailbox is uncached on
 * both sides (LIM is not data-cacheable: tt-isa-documentation BlackholeA0/L2CPUTile/Caches.md:26-31).
 *
 * Per-tick protocol (§1.4.4 steps 1-4), as implemented by monitor.c:
 *   host: write L2-L34 (obs, full lines), then L0 bytes 0-59 (one full line), sfence;
 *         then seq_req = n as its own 4 B UC write.
 *   hart: spin on seq_req; on a change: MB_FENCE_ACQ; check L0.crc32 and L0.obs_crc32 (re-read and
 *         retry for MB_CRC_RETRY_US, then state FAULT with mcause MB_ERR_CRC); run the command; write L35;
 *         MB_FENCE_REL; write L1 (everything but seq_done); MB_FENCE_REL; write L1.seq_done = n.
 *   host: poll L1.seq_done with UC reads, then read state/L35 and check seq_echo.
 */
#ifndef X280_MAILBOX_H
#define X280_MAILBOX_H

#include <stdint.h>
#include <stddef.h>

/* ---- LIM map ----------------------------------------------------------------------------------- */
/* LIM ("L3 as uncached scratchpad") starts at 0x0800_0000 (L2CPUTile/MemoryMap.md:84); at reset 15 of the
 * 16 x 128 KiB L3 pieces are LIM (Caches.md:73), i.e. 0x1E_0000 B ending at 0x081E_0000 (Caches.md:92). */
#define MB_LIM_BASE        0x08000000u
#define MB_LIM_SIZE        0x001E0000u
#define MB_LIM_END         (MB_LIM_BASE + MB_LIM_SIZE)
#define MB_LINE            64u                 /* L3/L1 line size (Caches.md:7, 26) */
#define MB_IMAGE_MAX       (48u * 1024u)       /* loadable monitor image (PLAN.md T-3.2) */
#define MB_CODE_END        0x08010000u         /* image + bss + persist + stack must end here */
#define MB_BASE            0x08010000u         /* mailbox, 64 lines = 4 KiB */
#define MB_L(n)            (MB_BASE + MB_LINE * (uint32_t)(n))
#define MB_PAYLOAD_BASE    0x08011000u         /* weights / payloads / bench working sets (§1.4.4) */

/* ---- line numbers ------------------------------------------------------------------------------ */
#define MB_L0_CMD          0     /* H: command                                                     */
#define MB_L1_STATUS       1     /* X: heartbeat, seq_done, state, trap dump                        */
#define MB_L2_OBS          2     /* H: obs f32[516], lines 2..34                                    */
#define MB_OBS_LINES       33
#define MB_L35_ACT         35    /* X: action f32[15], seq_echo                                     */
#define MB_L36_IDENT       36    /* X: ident and counters                                           */
#define MB_L37_LOAD        37    /* H: LOAD descriptor                                              */
#define MB_L38_CLEANUP     38    /* K: host_gone, written only by SET_NOC_CLEANUP; outside all CRCs  */
#define MB_L39_SCRATCH     39    /* X: E-test scratch (entry record, SELFTEST, BENCH, CALL results)  */
#define MB_L40_CHUNK       40    /* x280d: M4c chunk descriptor, lines 40-41 (unused in M3)          */
#define MB_L42_LBSTEP      42    /* reserved for M4a LbStep frames, lines 42-63                     */

/* ---- constants ----------------------------------------------------------------------------------- */
#define MB_MAGIC           0x32434257u   /* bytes "WBC2" (L0.magic, and L1.fw_magic echoes it) */
#define MB_LAYOUT_VERSION  2u
#define MB_FW_REV          1u
#define MB_FW_VER          ((MB_LAYOUT_VERSION << 16) | MB_FW_REV)
#define MB_HOST_GONE       0x474F4E45u   /* L38.host_gone as a u32 (PLAN.md §1.4.4 'GONE') */
#define MB_CRC_RETRY_US    50u           /* CRC mismatch is re-read for 50 us before FAULT */

/* L0.cmd */
#define MB_CMD_NOP         0u
#define MB_CMD_INFER       1u
#define MB_CMD_LOAD        2u
#define MB_CMD_FENCE_I     3u
#define MB_CMD_CALL        4u
#define MB_CMD_PARK        5u
#define MB_CMD_SELFTEST    6u
#define MB_CMD_BENCH       7u
#define MB_CMD_MH          8u      /* multi-hart (mh.c): policy = MB_MH_OP_*; flags [3:0] = harts (0 = 1-3);
                                      arg_len = timeout in us (0 = MB_MH_TIMEOUT_US) */
#define MB_N_CMDS          9u

/* MB_CMD_MH operations (L0.policy) */
#define MB_MH_OP_PING      0u      /* a NOP job on the harts asked: done mask and wake latency in L39 (mb_l39_mh) */
#define MB_MH_TIMEOUT_US   1000u

/* L0.flags */
#define MB_FLAG_HOST_ALIVE     (1u << 0)
#define MB_FLAG_VARIANT_SHIFT  8u              /* INFER/BENCH: wbc_infer variant in bits 8..11 ... */
#define MB_FLAG_VARIANT_MASK   0xFu
#define MB_FLAG_VARIANT_SET    (1u << 12)      /* ... used only when this bit is set, else the default */

/* L1.state */
#define MB_STATE_BOOT       0u
#define MB_STATE_IDLE       1u   /* last command succeeded */
#define MB_STATE_BUSY       2u   /* a command is running (heartbeat may pause while the kernel runs) */
#define MB_STATE_FAULT      3u   /* last command failed: mcause/mepc/mtval say why */
#define MB_STATE_REENTERED  4u   /* warm entry through the park stub; mcause/mepc = mncause/mnepc */
#define MB_STATE_HOST_GONE  5u   /* L38 == MB_HOST_GONE: commands are not executed until it is cleared */
#define MB_STATE_PARKED     6u   /* PARK done: the hart sits in wfi until the host's RNMI */

/* L1.mcause when state == FAULT and the fault was found by the monitor, not by a hardware trap.
 * Hardware traps report the CSR values (exception codes < 64, or bit 63 for interrupts). */
#define MB_ERR_CRC          0xC0Cu  /* crc32 / obs_crc32 still wrong after 50 us; mtval b0 body, b1 obs  */
#define MB_ERR_MAGIC        0xC0Du  /* L0.magic != MB_MAGIC (CRC was right); mtval = magic              */
#define MB_ERR_CMD          0xC0Eu  /* unknown command; mtval = cmd                                     */
#define MB_ERR_ARG          0xC0Fu  /* bad argument; mtval = MB_ARG_* code, mepc = offending value      */
#define MB_ERR_KERNEL       0xC10u  /* wbc_infer < 0; mtval = return code (sign-extended)               */
#define MB_ERR_BLOB         0xC11u  /* no blob registered, or wbc_blob_check < 0 (mtval = return code)  */
#define MB_ERR_LOAD_CRC     0xC12u  /* LOAD region CRC differs; mepc = dst, mtval = CRC the hart computed */
#define MB_ERR_NOCODE       0xC13u  /* CALL entry not inside a LOADed and FENCE_I'd code region         */
#define MB_ERR_DESC         0xC14u  /* L37 desc_crc32 wrong or load_seq != seq_req                      */

#define MB_ARG_RANGE        1u      /* address range outside the allowed LIM area                       */
#define MB_ARG_ALIGN        2u
#define MB_ARG_POLICY       3u
#define MB_ARG_KIND         4u
#define MB_ARG_LEN          5u
#define MB_ARG_OVERLAP      6u      /* BENCH output array overlaps the registered blob or code          */

/* L37.kind */
#define MB_LOAD_CODE        0u      /* executable payload: CALL needs it plus FENCE_I                   */
#define MB_LOAD_BLOB        1u      /* weight blob: wbc_blob_check, then INFER/BENCH use it             */
#define MB_LOAD_DATA        2u      /* verify the CRC only                                              */

/* L39.kind */
#define MB_L39_ENTRY        4u
#define MB_L39_SELFTEST     1u
#define MB_L39_BENCH        2u
#define MB_L39_CALL         3u
#define MB_L39_MH           5u

/* L36.entry_flags */
#define MB_ENTRY_WARM       (1u << 0)
#define MB_ENTRY_RELOADED   (1u << 1)   /* warm entry, but the persistent block did not match this build */
#define MB_ENTRY_STUB_KERNEL (1u << 2)  /* image was built without the kernel lane's wbc_mlp.c */

/* L36.mh (offset 60, hart 0): multi-hart support and the harts hart 0 last saw ready (mh.c mh_ready) */
#define MB_MH_CAP           (1u << 8)   /* the image has crt0's secondary RNMI-entry path, mh.c and MB_CMD_MH */
#define MB_MH_READY_MASK    0xFu        /* bit h: hart h ready (bit 0, hart 0, whenever MB_MH_CAP is set) */

/* ---- lines ------------------------------------------------------------------------------------- */
/* L0 (host). Command arguments by command:
 *   INFER    policy = WBC_POLICY_*, flags variant bits; obs in L2-L34 (obs_crc32 checked).
 *   LOAD     descriptor in L37.
 *   CALL     arg_entry = entry address; the payload gets {policy, flags, tick, arg_addr, arg_len}.
 *   SELFTEST arg_addr/arg_len = LIM range to read (0/0 = all of LIM).
 *   BENCH    policy/flags as INFER; arg_len = iterations; arg_addr = optional LIM address of a u32
 *            per-iteration mcycle array (0 = none).
 * crc32 covers bytes 0-43 (magic .. arg_len); obs_crc32 covers the 2,112 B of L2-L34 including the
 * zero padding after obs[515]. Both are zlib CRC-32 (reflected 0xEDB88320, init/final xor ~0). */
struct mb_l0 {
    uint32_t magic;       /*  0 */
    uint32_t cmd;         /*  4 */
    uint32_t policy;      /*  8 */
    uint32_t flags;       /* 12 */
    int64_t  tick;        /* 16 */
    uint64_t arg_addr;    /* 24 */
    uint64_t arg_entry;   /* 32 */
    uint32_t arg_len;     /* 40 */
    uint32_t crc32;       /* 44  over bytes 0..43 */
    uint32_t obs_crc32;   /* 48  over L2..L34 */
    uint32_t rsvd[2];     /* 52 */
    uint32_t seq_req;     /* 60  written last, as its own 4 B UC write */
};
#define MB_L0_CRC_BYTES 44u

/* L1 (hart). heartbeat = mtime (host reads hi, lo, hi). seq_done is written last, after a release fence.
 * mcause/mepc/mtval: trap dump or MB_ERR_* (state FAULT), mncause/mnepc (state REENTERED), else 0.
 * cycles_*: mcycle around wbc_infer of the last / slowest INFER. */
struct mb_l1 {
    uint64_t heartbeat;   /*  0 */
    uint32_t seq_done;    /*  8 */
    uint32_t state;       /* 12 */
    uint64_t mcause;      /* 16 */
    uint64_t mepc;        /* 24 */
    uint64_t mtval;       /* 32 */
    uint32_t cycles_last; /* 40 */
    uint32_t cycles_max;  /* 44 */
    uint32_t fw_ver;      /* 48  MB_FW_VER */
    uint32_t mhartid;     /* 52 */
    uint32_t vlenb;       /* 56 */
    uint32_t fw_magic;    /* 60  MB_MAGIC once the monitor runs (addition to §1.4.4) */
};

struct mb_obs {           /* L2..L34 */
    float   obs[516];     /*    0 */
    uint8_t pad[48];      /* 2064 */
};

struct mb_l35 {           /* hart */
    float    action[15];  /*  0 */
    uint32_t seq_echo;    /* 60 */
};

/* L36 ident (hart); offsets 0-35 are §1.4.4, 36-63 are additions. */
struct mb_l36 {
    uint64_t misa;          /*  0 */
    uint32_t mtime_hz;      /*  8 */
    uint32_t build_id;      /* 12 */
    uint32_t kernel_variant;/* 16  variant of the last INFER/BENCH (the default before any) */
    uint32_t n_runs;        /* 20  successful INFERs since the last cold entry */
    uint32_t n_overrun;     /* 24  commands finished after the host had already moved seq_req on */
    uint32_t n_reentry;     /* 28  warm entries since the last cold entry */
    uint32_t beu_accrued;   /* 32  hart-0 BEU accrued register at the last SELFTEST */
    uint32_t n_traps;       /* 36  hardware traps taken */
    uint32_t n_crc_fail;    /* 40  commands failed with MB_ERR_CRC */
    uint32_t variants_built;/* 44  wbc_variants_built() bitmask, 0 for the stub kernel */
    uint32_t n_host_gone;   /* 48  transitions into MB_STATE_HOST_GONE */
    uint32_t entry_flags;   /* 52  MB_ENTRY_* of the last entry */
    uint32_t n_crc_retry;   /* 56  commands that saw at least one CRC mismatch (fixed by a re-read or not) */
    uint32_t mh;            /* 60  MB_MH_CAP | ready mask (MB_MH_READY_MASK), at each entry and MB_CMD_MH */
};

/* L37 LOAD descriptor (host): the host writes [dst, dst+len) itself (full 64 B lines), then this line,
 * then L0 with cmd LOAD. load_seq must equal the LOAD's seq_req (catches a stale descriptor);
 * desc_crc32 covers bytes 0..59. */
struct mb_l37 {
    uint64_t dst;           /*  0  64 B aligned, >= MB_PAYLOAD_BASE */
    uint32_t len;           /*  8  dst + len <= MB_LIM_END */
    uint32_t crc32;         /* 12  zlib CRC-32 of the len bytes */
    uint32_t kind;          /* 16  MB_LOAD_* */
    uint32_t load_seq;      /* 20 */
    uint32_t rsvd[9];       /* 24 */
    uint32_t desc_crc32;    /* 60  over bytes 0..59 */
};
#define MB_L37_CRC_BYTES 60u

struct mb_l38 {             /* KMD cleanup write only */
    uint32_t host_gone;     /*  0  MB_HOST_GONE or 0 (x280d clears it at attach) */
    uint32_t rsvd[15];
};

/* L39 E-test scratch (hart). kind/seq say which command wrote it. For CALL, bytes 24..63 belong to the
 * payload (mon_api.result); the monitor writes only 0..23 after the payload returns. */
struct mb_l39 {
    uint32_t kind;          /*  0  MB_L39_* */
    uint32_t seq;           /*  4 */
    uint64_t ret;           /*  8  ENTRY: n_reentry; SELFTEST: xor of all words; BENCH: iterations; CALL: return */
    uint64_t cycles;        /* 16  mcycle spent in the command */
    uint8_t  data[40];      /* 24  per-kind, see below */
};

struct mb_l39_entry {       /* data[] of MB_L39_ENTRY */
    uint32_t entry_kind;    /* 24  0 cold, 1 warm, 2 warm + reloaded */
    uint32_t mnstatus;      /* 28 */
    uint64_t mncause;       /* 32 */
    uint64_t mnepc;         /* 40 */
    uint64_t mcycle;        /* 48 */
    uint64_t mtime;         /* 56 */
};

struct mb_l39_selftest {    /* data[] of MB_L39_SELFTEST */
    uint32_t lines;         /* 24  lines read */
    uint32_t cyc_per_mtime_x1000; /* 28  mcycle/mtime over the scan, x1000 (expect 16000 at 800/50 MHz) */
    uint32_t beu_cause;     /* 32 */
    uint32_t beu_accrued_before; /* 36 */
    uint64_t beu_value;     /* 40 */
    uint32_t beu_accrued_after;  /* 48 */
    uint32_t misa_v;        /* 52  misa bit 21 ('V') */
    uint32_t vlenb;         /* 56 */
    uint32_t mtime_ticks;   /* 60  mtime ticks spent in the scan */
};

struct mb_l39_mh {          /* data[] of MB_L39_MH (MB_CMD_MH ping) */
    uint32_t op;            /* 24  MB_MH_OP_* */
    uint32_t mask;          /* 28  harts asked (bits 1-3) */
    uint32_t done;          /* 32  harts that completed the job */
    uint32_t gen;           /* 36  job generation (mh_ctl.gen) */
    uint32_t wake_ticks[3]; /* 40  harts 1-3: mtime ticks from the dispatch to the pick-up (~0: not picked up) */
    uint32_t total_ticks;   /* 52  dispatch to the last completion, or to the timeout */
    uint32_t ready;         /* 56  ready mask after the ping (bit 0 = hart 0) */
    uint32_t trapped;       /* 60  harts whose stat line counts traps in this epoch */
};

struct mb_l39_bench {       /* data[] of MB_L39_BENCH */
    uint32_t iters;         /* 24 */
    uint32_t cyc_min;       /* 28 */
    uint32_t cyc_max;       /* 32 */
    uint32_t nondet;        /* 36  iterations whose action bits differ from the first */
    uint64_t cyc_sum;       /* 40 */
    uint32_t act_crc32;     /* 48  CRC-32 of the last action (60 B) */
    int32_t  rc;            /* 52  last wbc_infer return code */
    uint32_t variant;       /* 56 */
    uint32_t policy;        /* 60 */
};

/* Arguments a CALLed payload receives (a copy of L0 taken after the CRC check). */
struct mb_call_args {
    uint32_t policy, flags;
    int64_t  tick;
    uint64_t arg_addr, arg_entry;
    uint32_t arg_len, seq;
};

/* Services the monitor hands a payload in a1: uint64_t entry(const mb_call_args *, const mon_api *).
 * ptr() turns a LIM physical address into a pointer (identity on the X280, the fake LIM on x86).
 * Version 2 appends the multi-hart services (mh.c; mh.h has the job contract): mh_ready() = harts ready (bit h, bit 0
 * always), mh_run(mask, fn, arg, timeout) dispatches fn(arg, h, wapi) to the ready harts of mask & 0xE, runs hart 0's
 * share itself when bit 0 is set, then waits up to timeout mtime ticks and returns the completed mask (timeout 0: no
 * wait, returns the dispatched mask); mh_wait(timeout) waits for the job in flight. wapi = the workers' services. */
struct mh_wapi;
struct mon_api {
    uint32_t version;       /* MON_API_VERSION */
    uint32_t size;          /* sizeof(struct mon_api) */
    uint64_t mtime_hz;
    uint64_t (*mtime)(void);
    uint64_t (*mcycle)(void);
    void     (*heartbeat)(void);
    void    *(*ptr)(uint64_t lim_phys);
    volatile uint8_t *result;   /* L39 bytes 24..63 */
    uint32_t result_len;        /* 40 */
    uint32_t rsvd;
    /* version 2 */
    uint32_t (*mh_ready)(void);
    uint32_t (*mh_run)(uint32_t mask, uint64_t fn, uint64_t arg, uint64_t timeout_ticks);
    uint32_t (*mh_wait)(uint64_t timeout_ticks);
    const struct mh_wapi *wapi;
};
#define MON_API_VERSION 2u

/* Offsets pinned (PLAN.md §1.4.4). */
#define MB_SA(t, f, o) _Static_assert(offsetof(struct t, f) == (o), #t "." #f " != " #o)
MB_SA(mb_l0, magic, 0);      MB_SA(mb_l0, cmd, 4);        MB_SA(mb_l0, policy, 8);
MB_SA(mb_l0, flags, 12);     MB_SA(mb_l0, tick, 16);      MB_SA(mb_l0, arg_addr, 24);
MB_SA(mb_l0, arg_entry, 32); MB_SA(mb_l0, arg_len, 40);   MB_SA(mb_l0, crc32, 44);
MB_SA(mb_l0, obs_crc32, 48); MB_SA(mb_l0, seq_req, 60);
MB_SA(mb_l1, heartbeat, 0);  MB_SA(mb_l1, seq_done, 8);   MB_SA(mb_l1, state, 12);
MB_SA(mb_l1, mcause, 16);    MB_SA(mb_l1, mepc, 24);      MB_SA(mb_l1, mtval, 32);
MB_SA(mb_l1, cycles_last, 40); MB_SA(mb_l1, cycles_max, 44); MB_SA(mb_l1, fw_ver, 48);
MB_SA(mb_l1, mhartid, 52);   MB_SA(mb_l1, vlenb, 56);     MB_SA(mb_l1, fw_magic, 60);
MB_SA(mb_obs, obs, 0);       MB_SA(mb_obs, pad, 2064);
MB_SA(mb_l35, action, 0);    MB_SA(mb_l35, seq_echo, 60);
MB_SA(mb_l36, misa, 0);      MB_SA(mb_l36, mtime_hz, 8);  MB_SA(mb_l36, build_id, 12);
MB_SA(mb_l36, kernel_variant, 16); MB_SA(mb_l36, n_runs, 20); MB_SA(mb_l36, n_overrun, 24);
MB_SA(mb_l36, n_reentry, 28); MB_SA(mb_l36, beu_accrued, 32); MB_SA(mb_l36, n_traps, 36);
MB_SA(mb_l36, n_crc_fail, 40); MB_SA(mb_l36, variants_built, 44); MB_SA(mb_l36, n_host_gone, 48);
MB_SA(mb_l36, entry_flags, 52); MB_SA(mb_l36, n_crc_retry, 56); MB_SA(mb_l36, mh, 60);
MB_SA(mb_l37, dst, 0);       MB_SA(mb_l37, len, 8);       MB_SA(mb_l37, crc32, 12);
MB_SA(mb_l37, kind, 16);     MB_SA(mb_l37, load_seq, 20); MB_SA(mb_l37, desc_crc32, 60);
MB_SA(mb_l38, host_gone, 0);
MB_SA(mb_l39, kind, 0);      MB_SA(mb_l39, seq, 4);       MB_SA(mb_l39, ret, 8);
MB_SA(mb_l39, cycles, 16);   MB_SA(mb_l39, data, 24);
_Static_assert(sizeof(struct mb_l0) == MB_LINE && sizeof(struct mb_l1) == MB_LINE, "L0/L1 size");
_Static_assert(sizeof(struct mb_obs) == MB_OBS_LINES * MB_LINE, "L2-L34 size");
_Static_assert(sizeof(struct mb_l35) == MB_LINE && sizeof(struct mb_l36) == MB_LINE, "L35/L36 size");
_Static_assert(sizeof(struct mb_l37) == MB_LINE && sizeof(struct mb_l38) == MB_LINE, "L37/L38 size");
_Static_assert(sizeof(struct mb_l39) == MB_LINE, "L39 size");
_Static_assert(sizeof(struct mb_l39_entry) == 40 && sizeof(struct mb_l39_selftest) == 40, "L39 data");
_Static_assert(sizeof(struct mb_l39_bench) == 40 && sizeof(struct mb_l39_mh) == 40, "L39 bench / mh data");
_Static_assert(MB_L(MB_L2_OBS) == 0x08010080u && MB_L(MB_L35_ACT) == 0x080108C0u, "obs/act address");
_Static_assert(MB_L(MB_L36_IDENT) == 0x08010900u && MB_L(MB_L37_LOAD) == 0x08010940u, "L36/L37 address");
_Static_assert(MB_L(MB_L38_CLEANUP) == 0x08010980u && MB_L(MB_L39_SCRATCH) == 0x080109C0u, "L38/L39");
_Static_assert(MB_L(MB_L40_CHUNK) == 0x08010A00u && MB_L(MB_L42_LBSTEP) == 0x08010A80u, "L40/L42");
_Static_assert(MB_L(64) == MB_PAYLOAD_BASE, "mailbox is 4 KiB");

/* ---- fences ------------------------------------------------------------------------------------ */
/* §1.4.4 asks for `fence r,rw` after seeing seq_req and `fence w,w` before L1. They are widened with the
 * device-I/O bits because the X280's PMA table gives LIM only "RWX A" (tt-bh-linux README.md:263-273)
 * and does not say whether it is a main-memory or an I/O region; the i/o variants order it either way. */
#if defined(__riscv)
#define MB_FENCE_ACQ() __asm__ volatile("fence ir, iorw" ::: "memory")
#define MB_FENCE_REL() __asm__ volatile("fence ow, ow" ::: "memory")
#else
#define MB_FENCE_ACQ() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define MB_FENCE_REL() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif

#endif /* X280_MAILBOX_H */
