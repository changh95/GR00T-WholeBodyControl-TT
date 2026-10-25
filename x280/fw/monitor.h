/* Internal interface of the resident monitor (monitor.c, trap.c) for crt0.S and the platform shims
 * (plat_x86.c, spike_host.c). Not part of the host contract: that is mailbox.h.
 */
#ifndef X280_MONITOR_H
#define X280_MONITOR_H

#include <stdint.h>
#include "mailbox.h"
#include "plat.h"
#include "wbc_api.h"
#include "mh.h"

/* L36.build_id and the persistent-block cookie: build/gen/build_id.c (Makefile, cksum of the sources). */
extern const uint32_t mon_build_id;
#define MON_BUILD_ID mon_build_id
#ifndef MON_DEFAULT_VARIANT
#define MON_DEFAULT_VARIANT WBC_VARIANT_V0
#endif
#define MON_COOKIE 0x53524550u   /* "PERS" */

/* Survives a warm entry (park stub -> crt0); zeroed on a cold entry or when the cookie, build id or size
 * do not match (a new image was written while the hart was parked). Section .persist (lim.ld). */
struct mon_persist {
    uint32_t cookie, build_id, size, entry_flags;
    uint32_t n_reentry, n_runs, n_overrun, n_traps;
    uint32_t n_crc_fail, n_crc_retry, n_host_gone, cycles_max;
    uint32_t kernel_variant, beu_accrued;
    uint32_t blob_ok, blob_len;
    uint64_t blob_addr;
    uint64_t code_lo, code_hi;
    uint32_t code_ok, code_fenced;
};

/* Zeroed on every entry (crt0.S clears .bss; the x86 shim clears it in mon_x86_boot/reenter). */
struct mon_rt {
    uint32_t last_seq;          /* seq_req value already handled (or adopted at entry) */
    uint32_t state;             /* mirror of L1.state */
    uint32_t restart_pending;   /* an unguarded trap restarted the loop: fail the current seq */
    volatile uint32_t guard_active;
    uint32_t trap_depth;
    uint32_t idle_iter;
    uint32_t mh_ready;          /* L36.mh ready mask: mh_ready() at entry and after each MB_CMD_MH */
    uint64_t last_hb;
    uint64_t fault_mcause, fault_mepc, fault_mtval;
    uint64_t entry_mnstatus, entry_mncause, entry_mnepc;
    mon_jmpbuf guard;
    struct mb_call_args args;
    struct mon_api api;
    float act[16];
    uint64_t obs[MB_OBS_LINES * MB_LINE / 8] __attribute__((aligned(64)));  /* checked copy of L2-L34 */
    uint8_t scratch[WBC_SCRATCH_BYTES] __attribute__((aligned(64)));
};

extern struct mon_persist g_persist;
extern struct mon_rt g_rt;

/* crt0.S calls mon_entry() (bookkeeping: cold/warm, n_reentry), then re-arms RNMIs (csrsi 0x353, 8),
 * then monitor_main(), which never returns. monitor_resume() is the re-entry after an unguarded trap. */
void mon_entry(void);
__attribute__((noreturn)) void monitor_main(void);
__attribute__((noreturn)) void monitor_resume(void);

void mon_start(void);       /* entry-time L1/L36/L39 writes (monitor_main's first half) */
void mon_poll_once(void);   /* one iteration of the idle loop: poll seq_req, run a command */
void mon_heartbeat(void);
uint32_t mon_crc32(uint32_t crc, const void *p, uint32_t n);   /* zlib-compatible, crc = 0 to start */
/* mon_crc32's implementation: slicing-by-8 over mon_crc32_tab (crc32_tab.h, .rodata), or the bitwise loop with
 * -DMON_CRC_BITWISE (tests only; no table). */
#if defined(MON_CRC_BITWISE)
#define MON_CRC_SLICES 0u
#else
#define MON_CRC_SLICES 8u
extern const uint32_t mon_crc32_tab[8][256];
#endif

/* mh.c, hart 0 side: mh_hart0_entry() at every entry (cold: the ctl line and a new epoch; warm: nothing, harts 1-3
 * may be running), mh_ready() = ready harts (bit 0 always), the mon_api v2 services, the MB_CMD_MH ping. */
void     mh_hart0_entry(int cold);
uint32_t mh_ready(void);
uint32_t mh_run(uint32_t mask, uint64_t fn, uint64_t arg, uint64_t timeout_ticks);
uint32_t mh_wait(uint64_t timeout_ticks);
uint32_t mh_ping(uint32_t mask, uint64_t timeout_ticks, struct mb_l39_mh *out);
extern const struct mh_wapi mh_wapi;
/* mh.c, harts 1-3: entered from crt0's secondary RNMI-entry path on the hart's own MH stack (never returns);
 * mh_secondary_resume() is the idle loop again after a trap outside any job (crt0 mh_sec_trap_entry, x86 shim);
 * mh_sec_trap() longjmps into the running job's guard, or returns when there is none. */
__attribute__((noreturn)) void mh_secondary_main(uint32_t hart);
__attribute__((noreturn)) void mh_secondary_resume(uint32_t hart);
void mh_sec_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval, uint64_t hart);

/* trap.c: called by trap_entry (crt0.S) on its own stack, or by the x86 shim to emulate a trap. */
__attribute__((noreturn)) void mon_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval, uint64_t sp);

#if defined(SPIKE_HOST)
__attribute__((noreturn)) void sh_run(void);                      /* spike_host.c */
__attribute__((noreturn)) void sh_emulated_rnmi(uint64_t mncause, uint64_t mnepc);
#endif

/* INFER/BENCH/LOAD go through these, so the x86 tests can swap in a fake kernel (plat_x86.c). */
#if defined(MON_X86)
extern int (*mon_x86_infer)(const void *, uint32_t, uint32_t, const float *, float *, void *);
extern int (*mon_x86_blob_check)(const void *, uint32_t);
#define MON_WBC_INFER      mon_x86_infer
#define MON_WBC_BLOB_CHECK mon_x86_blob_check
#else
#define MON_WBC_INFER      wbc_infer
#define MON_WBC_BLOB_CHECK wbc_blob_check
#endif

#define MB_PTR(T, line) ((volatile T *)plat_ptr(MB_L(line)))

#endif /* X280_MONITOR_H */
