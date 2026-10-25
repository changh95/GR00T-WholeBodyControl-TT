/* x86 platform shim of the monitor: a fake LIM buffer, fake CSRs/clock/BEU, emulated RNMI re-entry,
 * trap and hang payloads, and a small driver API for tests/x280/test_mailbox_x86.py (ctypes).
 *
 * Why: monitor.c and trap.c are compiled unchanged (-DMON_X86), so the protocol (CRC retry, seq ordering,
 * every command, trap recovery, warm entry) is tested on the host before any hart is released
 * (PLAN.md WP-3.2 "x86 build of the monitor for unit tests").
 *
 * Driver model: mon_x86_boot() = reset + crt0 + mon_start(); mon_x86_step(n) runs n iterations of the
 * poll loop and returns early when the "hart" parks (PARK or a nested trap) or hangs (hang payload);
 * mon_x86_reenter(mncause, mnepc) = the host's RNMI + park stub + crt0 (warm entry).
 * Harts 1-3 (mh.c): mon_x86_secondary_start(h, mncause, mnepc) = the host's RNMI on bit h with the --harts 4 vectors
 * + crt0's secondary path: a thread running mh_secondary_main(h) against the same fake LIM, with a thread-local hart id
 * (plat_mhartid), its own fake mnstatus / msip / mtimecmp / BEU, and a wfi that sleeps until its fake msip or
 * mtimecmp (or ~1 ms: a spurious wake, which the ISA allows). mon_x86_secondary_stop(h) ends the thread at its next
 * wfi. The fake mtime and mcycle are shared and advance atomically; only hart 0 calls the test clock hook.
 */
#define _POSIX_C_SOURCE 200809L   /* nanosleep under -std=c11 */
#include <stddef.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "monitor.h"

#define X86_EXPORT __attribute__((visibility("default")))

uint8_t *g_x86_lim;

enum { X86_RUNNING = 0, X86_PARKED = 1, X86_HUNG = 2 };

static struct {
    uint64_t mtime, mtime_step, mcycle, mcycle_step;
    uint64_t misa, mnstatus, mncause, mnepc;
    uint32_t vlenb, mhartid;
    uint32_t beu_cause, beu_accrued;
    uint64_t beu_value;
    void (*clock_hook)(uint64_t);
    uint64_t n_mtime;
} fk;

struct x86_stats {
    uint64_t fence_i, rearm, mtime_reads, calls, parks, hangs, traps_emulated, restarts;
};
static struct x86_stats st;

static jmp_buf drv;
static volatile int drv_status;

/* ---- harts 1-3 ---------------------------------------------------------------------------------------------- */
static __thread uint32_t tl_hart;     /* 0: hart 0 (whatever thread drives it); 1..3: that secondary's thread */
static struct {
    pthread_t th;
    volatile int running, stop, hang;
    uint64_t mnstatus, mncause, mnepc;
    volatile uint32_t msip;
    volatile uint64_t mtimecmp;
    uint32_t beu_accrued, rearm;
    jmp_buf base;
} sec[MH_NHARTS];

/* ---- kernel routing (fake kernel for protocol tests, else the linked wbc_infer) ---------------- */

#define FAKE_BLOB_MAGIC 0x454B4146u   /* "FAKE" */

static int fake_blob_check(const void *blob, uint32_t len)
{
    return (len >= 4 && *(const uint32_t *)blob == FAKE_BLOB_MAGIC) ? 0 : -2;
}

/* act[j] = obs[j] * 0.5 + obs[515] + (policy * 100 + variant * 10 + j); variant 15 fails with -8. */
static int fake_infer(const void *blob, uint32_t policy, uint32_t variant,
                      const float *obs, float *act, void *scratch)
{
    (void)blob; (void)scratch;
    if (variant == 15u) return -8;
    for (int j = 0; j < WBC_ACT_DIM; j++)
        act[j] = obs[j] * 0.5f + obs[515] + (float)(policy * 100u + variant * 10u + (uint32_t)j);
    return 0;
}

int (*mon_x86_infer)(const void *, uint32_t, uint32_t, const float *, float *, void *) = wbc_infer;
int (*mon_x86_blob_check)(const void *, uint32_t) = wbc_blob_check;

X86_EXPORT void mon_x86_use_fake_kernel(int on)
{
    mon_x86_infer = on ? fake_infer : wbc_infer;
    mon_x86_blob_check = on ? fake_blob_check : wbc_blob_check;
}

X86_EXPORT uint32_t mon_x86_kernel_is_stub(void) { return wbc_kernel_is_stub; }
X86_EXPORT uint32_t mon_x86_variants_built(void) { return mon_variants_built(); }

/* ---- plat.h primitives ---------------------------------------------------------------------------- */

uint64_t plat_mtime(void)
{
    __atomic_add_fetch(&st.mtime_reads, 1, __ATOMIC_RELAXED);
    uint64_t t = __atomic_add_fetch(&fk.mtime, fk.mtime_step, __ATOMIC_SEQ_CST);
    if (fk.clock_hook && !tl_hart) fk.clock_hook(++fk.n_mtime);
    return tl_hart ? t : fk.mtime;
}
uint64_t plat_mcycle(void) { return __atomic_add_fetch(&fk.mcycle, fk.mcycle_step, __ATOMIC_RELAXED); }
uint64_t plat_misa(void) { return fk.misa; }
uint32_t plat_vlenb(void) { return fk.vlenb; }
uint32_t plat_mhartid(void) { return tl_hart ? tl_hart : fk.mhartid; }
uint64_t plat_mnstatus(void) { return tl_hart ? sec[tl_hart].mnstatus : fk.mnstatus; }
uint64_t plat_mncause(void) { return tl_hart ? sec[tl_hart].mncause : fk.mncause; }
uint64_t plat_mnepc(void) { return tl_hart ? sec[tl_hart].mnepc : fk.mnepc; }
void plat_rnmi_rearm(void)
{
    if (tl_hart) { sec[tl_hart].mnstatus |= 8u; sec[tl_hart].rearm++; return; }
    fk.mnstatus |= 8u;
    st.rearm++;
}
/* st.fence_i counts hart 0's fence.i (mailbox_sim.py's model of its L1I GP-scratch line reads it); a secondary's own
 * (mh.c payload jobs) is in its mh_aux line. */
void plat_fence_i(void) { if (!tl_hart) __atomic_add_fetch(&st.fence_i, 1, __ATOMIC_RELAXED); }

/* multi-hart primitives (plat.h): fake CLINT msip/mtimecmp per hart, the calling hart's mip, a secondary's wfi */
void plat_msip_set(uint32_t h, uint32_t v) { if (h < MH_NHARTS) sec[h].msip = v; }
void plat_mtimecmp_set(uint32_t h, uint64_t t) { if (h < MH_NHARTS) sec[h].mtimecmp = t; }
uint64_t plat_mip(void)
{
    uint32_t h = tl_hart;
    return (sec[h].msip ? MIP_MSIP : 0u) | (__atomic_load_n(&fk.mtime, __ATOMIC_SEQ_CST) >= sec[h].mtimecmp ? MIP_MTIP : 0u);
}
void plat_wfi(void)
{
    uint32_t h = tl_hart;
    if (!h) return;                                    /* hart 0 never waits in wfi on x86 */
    struct timespec ts = { 0, 20000 };
    for (int i = 0;; i++) {
        if (sec[h].stop) longjmp(sec[h].base, 2);       /* mon_x86_secondary_stop: leave the thread */
        if (!sec[h].hang && (sec[h].msip || i >= 50 ||
                             __atomic_load_n(&fk.mtime, __ATOMIC_SEQ_CST) >= sec[h].mtimecmp)) return;
        nanosleep(&ts, 0);
    }
}
uint32_t plat_beu_accrued_hart(uint32_t h) { return h == 0 ? fk.beu_accrued : h < MH_NHARTS ? sec[h].beu_accrued : 0u; }
int plat_in_image(uint64_t addr) { (void)addr; return 0; }   /* x86 'payloads' are stand-ins: count the fence.i */
void plat_mh_restart(uint32_t h) { longjmp(sec[h].base, 1); }
void plat_mh_relax(void)
{
    struct timespec ts = { 0, 20000 };
    nanosleep(&ts, 0);
}

static void *sec_main(void *p)
{
    uint32_t h = (uint32_t)(uintptr_t)p;
    tl_hart = h;
    int r = setjmp(sec[h].base);
    if (r == 0) {
        plat_rnmi_rearm();                             /* crt0: csrsi 0x353, 8 before mh_secondary_main */
        mh_secondary_main(h);
    } else if (r == 1) {
        mh_secondary_resume(h);                        /* a trap outside any job (plat_mh_restart) */
    }
    sec[h].running = 0;
    return 0;
}

static void sec_stop(uint32_t h)
{
    if (!sec[h].th) return;
    sec[h].stop = 1;
    pthread_join(sec[h].th, 0);
    sec[h].th = 0;
    sec[h].running = 0;
    sec[h].stop = 0;
}

/* The host's RNMI on bit h with handler = crt0's _start, trigger cleared: hardware clears NMIE and records
 * mncause/mnepc (RNMIs.md:18-22), crt0's secondary path runs mh_secondary_main(h) on the hart's MH stack. */
X86_EXPORT int mon_x86_secondary_start(uint32_t h, uint64_t mncause, uint64_t mnepc)
{
    if (h == 0 || h >= MH_NHARTS || !g_x86_lim) return -1;
    sec_stop(h);
    sec[h].hang = 0;
    sec[h].msip = 0;
    sec[h].mnstatus &= ~8ull;
    sec[h].mncause = mncause;
    sec[h].mnepc = mnepc;
    sec[h].running = 1;
    if (pthread_create(&sec[h].th, 0, sec_main, (void *)(uintptr_t)h)) {
        sec[h].th = 0;
        sec[h].running = 0;
        return -2;
    }
    return 0;
}
X86_EXPORT void mon_x86_secondary_stop(uint32_t h) { if (h && h < MH_NHARTS) sec_stop(h); }
X86_EXPORT void mon_x86_secondary_stop_all(void) { for (uint32_t h = 1; h < MH_NHARTS; h++) sec_stop(h); }
X86_EXPORT int mon_x86_secondary_running(uint32_t h) { return h && h < MH_NHARTS ? sec[h].running : 0; }
X86_EXPORT uint64_t mon_x86_secondary_mnstatus(uint32_t h) { return h < MH_NHARTS ? sec[h].mnstatus : 0; }
X86_EXPORT uint32_t mon_x86_secondary_rearms(uint32_t h) { return h < MH_NHARTS ? sec[h].rearm : 0; }
X86_EXPORT void mon_x86_secondary_hang(uint32_t h, int on) { if (h && h < MH_NHARTS) sec[h].hang = on; }
X86_EXPORT void mon_x86_set_beu_hart(uint32_t h, uint32_t accrued) { if (h && h < MH_NHARTS) sec[h].beu_accrued = accrued; }
X86_EXPORT uint32_t mon_x86_msip(uint32_t h) { return h < MH_NHARTS ? sec[h].msip : 0; }

void plat_beu_read(uint32_t *cause, uint64_t *value, uint32_t *accrued)
{
    *cause = fk.beu_cause;
    *value = fk.beu_value;
    *accrued = fk.beu_accrued;
}

void plat_park_forever(void)
{
    st.parks++;
    drv_status = X86_PARKED;
    longjmp(drv, 1);
}

void plat_restart_after_trap(void)
{
    st.restarts++;
    longjmp(drv, 1);   /* status stays RUNNING; the next step handles g_rt.restart_pending */
}

/* ---- payloads -------------------------------------------------------------------------------------- */

typedef uint64_t (*x86_payload_fn)(const struct mb_call_args *, const struct mon_api *);
uint64_t x86_payload_checksum(const struct mb_call_args *, const struct mon_api *);
uint64_t x86_payload_illegal(const struct mb_call_args *, const struct mon_api *);
uint64_t x86_payload_hang(const struct mb_call_args *, const struct mon_api *);
uint64_t x86_payload_lim_bench(const struct mb_call_args *, const struct mon_api *);
uint64_t x86_payload_mh_bench(const struct mb_call_args *, const struct mon_api *);

void x86_emulate_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval);
void x86_emulate_hang(void);

void x86_emulate_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval)
{
    __atomic_add_fetch(&st.traps_emulated, 1, __ATOMIC_RELAXED);
    if (tl_hart) {                                     /* a secondary: its own trap path (crt0 mh_sec_trap_entry) */
        mh_sec_trap(mcause, mepc, mtval, tl_hart);
        plat_mh_restart(tl_hart);
    }
    mon_trap(mcause, mepc, mtval, 0);
}

void x86_emulate_hang(void)
{
    st.hangs++;
    drv_status = X86_HUNG;
    longjmp(drv, 1);
}

#define X86_MAX_PAYLOADS 16
static struct { uint64_t addr; x86_payload_fn fn; } payloads[X86_MAX_PAYLOADS];

/* kind: 1 checksum, 2 illegal instruction, 3 hang, 4 lim_bench, 5 callback (cb, returns normally), 6 mh_bench. */
X86_EXPORT int mon_x86_register_payload(uint64_t addr, int kind, x86_payload_fn cb)
{
    static const x86_payload_fn builtin[] = {
        0, x86_payload_checksum, x86_payload_illegal, x86_payload_hang,
        x86_payload_lim_bench, 0, x86_payload_mh_bench,
    };
    x86_payload_fn fn = kind == 5 ? cb : (kind >= 1 && kind <= 6) ? builtin[kind] : 0;
    if (!fn) return -1;
    for (int i = 0; i < X86_MAX_PAYLOADS; i++) {
        if (!payloads[i].fn || payloads[i].addr == addr) {
            payloads[i].addr = addr;
            payloads[i].fn = fn;
            return 0;
        }
    }
    return -2;
}

X86_EXPORT void mon_x86_clear_payloads(void) { memset(payloads, 0, sizeof payloads); }

uint64_t plat_call(uint64_t entry, const struct mb_call_args *a, const struct mon_api *api)
{
    st.calls++;
    for (int i = 0; i < X86_MAX_PAYLOADS; i++)
        if (payloads[i].fn && payloads[i].addr == entry) return payloads[i].fn(a, api);
    x86_emulate_trap(1u, entry, entry);   /* instruction access fault at the entry */
    return 0;
}

/* ---- driver API ------------------------------------------------------------------------------------- */

X86_EXPORT int mon_x86_attach(void *lim, uint64_t len)
{
    if (len < MB_LIM_SIZE) return -1;
    mon_x86_secondary_stop_all();                      /* no thread may outlive the LIM it was given */
    g_x86_lim = lim;
    memset(&fk, 0, sizeof fk);
    memset(&st, 0, sizeof st);
    memset(sec, 0, sizeof sec);
    for (uint32_t h = 0; h < MH_NHARTS; h++) sec[h].mnstatus = 8;   /* out of reset: NMIE = 1 */
    fk.mtime_step = 1;
    fk.mcycle_step = 16;
    fk.misa = (2ull << 62) | PLAT_MISA_V | 0x112D;   /* RV64 + IMAFDC-ish + V */
    fk.vlenb = 64;
    return 0;
}

X86_EXPORT void mon_x86_set_clock(uint64_t mtime, uint64_t mtime_step, uint64_t mcycle_step)
{
    fk.mtime = mtime;
    fk.mtime_step = mtime_step;
    fk.mcycle_step = mcycle_step;
}

X86_EXPORT uint64_t mon_x86_mtime(void) { return fk.mtime; }
X86_EXPORT void mon_x86_set_clock_hook(void (*hook)(uint64_t)) { fk.clock_hook = hook; fk.n_mtime = 0; }

X86_EXPORT void mon_x86_set_csrs(uint64_t misa, uint32_t vlenb, uint32_t mhartid)
{
    fk.misa = misa;
    fk.vlenb = vlenb;
    fk.mhartid = mhartid;
}

X86_EXPORT void mon_x86_set_beu(uint32_t cause, uint64_t value, uint32_t accrued)
{
    fk.beu_cause = cause;
    fk.beu_value = value;
    fk.beu_accrued = accrued;
}

X86_EXPORT void mon_x86_stats(struct x86_stats *out) { *out = st; }
X86_EXPORT uint64_t mon_x86_mnstatus(void) { return fk.mnstatus; }

static int enter(void)
{
    plat_fence_i();                  /* crt0: hart 0's fence.i on every entry */
    memset(&g_rt, 0, sizeof g_rt);   /* crt0 clears .bss on every entry */
    mon_entry();
    plat_rnmi_rearm();               /* crt0: csrsi 0x353, 8 */
    drv_status = X86_RUNNING;
    if (setjmp(drv)) return drv_status;
    mon_start();
    return X86_RUNNING;
}

/* Reset: NMIE = 1 out of reset (RNMIs.md:4). */
X86_EXPORT int mon_x86_boot(void)
{
    fk.mnstatus = 8;
    fk.mncause = 0;
    fk.mnepc = 0;
    return enter();
}

/* The host's RNMI: hardware clears NMIE and records mncause/mnepc (RNMIs.md:18-22); the park stub
 * jumps to the reset handler address, i.e. crt0. */
X86_EXPORT int mon_x86_reenter(uint64_t mncause, uint64_t mnepc)
{
    fk.mnstatus &= ~8ull;
    fk.mncause = mncause;
    fk.mnepc = mnepc;
    return enter();
}

X86_EXPORT int mon_x86_step(uint32_t n)
{
    if (drv_status != X86_RUNNING) return drv_status;
    if (setjmp(drv)) return drv_status;
    for (uint32_t i = 0; i < n; i++) mon_poll_once();
    return X86_RUNNING;
}

/* Poll until L1.seq_done == seq (and the hart is not mid-command) or max iterations. Returns the
 * driver status, or -1 on timeout. */
X86_EXPORT int mon_x86_run_until(uint32_t seq, uint32_t max_iters)
{
    volatile struct mb_l1 *l1 = MB_PTR(struct mb_l1, MB_L1_STATUS);
    for (uint32_t i = 0; i < max_iters; i++) {
        int r = mon_x86_step(1);
        if (r != X86_RUNNING) return r;
        if (l1->seq_done == seq && g_rt.state != MB_STATE_BUSY) return X86_RUNNING;
    }
    return -1;
}

/* For the threaded test: poll until *stop != 0. Runs without the GIL (ctypes releases it). */
X86_EXPORT int mon_x86_run_thread(volatile uint32_t *stop, uint64_t max_iters)
{
    if (drv_status != X86_RUNNING) return drv_status;
    if (setjmp(drv)) return drv_status;
    for (uint64_t i = 0; i < max_iters && !*stop; i++) mon_poll_once();
    return X86_RUNNING;
}

X86_EXPORT void mon_x86_corrupt_persist(void) { g_persist.build_id ^= 0x5A5A5A5Au; }
X86_EXPORT uint32_t mon_x86_build_id(void) { return MON_BUILD_ID; }
X86_EXPORT uint32_t mon_x86_crc32(uint32_t crc, const void *p, uint32_t n) { return mon_crc32(crc, p, n); }
/* 8 = slicing-by-8 over mon_crc32_tab (the default), 0 = the -DMON_CRC_BITWISE build (build/x86-crcbit). */
X86_EXPORT uint32_t mon_x86_crc_slices(void) { return MON_CRC_SLICES; }
X86_EXPORT const void *mon_x86_crc32_tab(void)
{
#if MON_CRC_SLICES
    return mon_crc32_tab;
#else
    return 0;
#endif
}

/* ---- layout table (checked against mailbox_layout.py) ---------------------------------------------- */

#define L(n, v) { n, (uint64_t)(v) }
static const struct { const char *name; uint64_t value; } layout[] = {
    L("MB_LIM_BASE", MB_LIM_BASE), L("MB_LIM_SIZE", MB_LIM_SIZE), L("MB_BASE", MB_BASE),
    L("MB_PAYLOAD_BASE", MB_PAYLOAD_BASE), L("MB_CODE_END", MB_CODE_END), L("MB_IMAGE_MAX", MB_IMAGE_MAX),
    L("MB_MAGIC", MB_MAGIC), L("MB_FW_VER", MB_FW_VER), L("MB_HOST_GONE", MB_HOST_GONE),
    L("MB_CRC_RETRY_US", MB_CRC_RETRY_US), L("MB_L0_CRC_BYTES", MB_L0_CRC_BYTES),
    L("MB_L37_CRC_BYTES", MB_L37_CRC_BYTES), L("MB_OBS_LINES", MB_OBS_LINES),
    L("MB_FLAG_VARIANT_SHIFT", MB_FLAG_VARIANT_SHIFT), L("MB_FLAG_VARIANT_SET", MB_FLAG_VARIANT_SET),
    L("line.L0", MB_L(MB_L0_CMD)), L("line.L1", MB_L(MB_L1_STATUS)), L("line.L2", MB_L(MB_L2_OBS)),
    L("line.L35", MB_L(MB_L35_ACT)), L("line.L36", MB_L(MB_L36_IDENT)), L("line.L37", MB_L(MB_L37_LOAD)),
    L("line.L38", MB_L(MB_L38_CLEANUP)), L("line.L39", MB_L(MB_L39_SCRATCH)),
    L("line.L40", MB_L(MB_L40_CHUNK)), L("line.L42", MB_L(MB_L42_LBSTEP)),
    L("l0.magic", offsetof(struct mb_l0, magic)), L("l0.cmd", offsetof(struct mb_l0, cmd)),
    L("l0.policy", offsetof(struct mb_l0, policy)), L("l0.flags", offsetof(struct mb_l0, flags)),
    L("l0.tick", offsetof(struct mb_l0, tick)), L("l0.arg_addr", offsetof(struct mb_l0, arg_addr)),
    L("l0.arg_entry", offsetof(struct mb_l0, arg_entry)), L("l0.arg_len", offsetof(struct mb_l0, arg_len)),
    L("l0.crc32", offsetof(struct mb_l0, crc32)), L("l0.obs_crc32", offsetof(struct mb_l0, obs_crc32)),
    L("l0.seq_req", offsetof(struct mb_l0, seq_req)),
    L("l1.heartbeat", offsetof(struct mb_l1, heartbeat)), L("l1.seq_done", offsetof(struct mb_l1, seq_done)),
    L("l1.state", offsetof(struct mb_l1, state)), L("l1.mcause", offsetof(struct mb_l1, mcause)),
    L("l1.mepc", offsetof(struct mb_l1, mepc)), L("l1.mtval", offsetof(struct mb_l1, mtval)),
    L("l1.cycles_last", offsetof(struct mb_l1, cycles_last)),
    L("l1.cycles_max", offsetof(struct mb_l1, cycles_max)), L("l1.fw_ver", offsetof(struct mb_l1, fw_ver)),
    L("l1.mhartid", offsetof(struct mb_l1, mhartid)), L("l1.vlenb", offsetof(struct mb_l1, vlenb)),
    L("l1.fw_magic", offsetof(struct mb_l1, fw_magic)),
    L("l35.action", offsetof(struct mb_l35, action)), L("l35.seq_echo", offsetof(struct mb_l35, seq_echo)),
    L("l36.misa", offsetof(struct mb_l36, misa)), L("l36.mtime_hz", offsetof(struct mb_l36, mtime_hz)),
    L("l36.build_id", offsetof(struct mb_l36, build_id)),
    L("l36.kernel_variant", offsetof(struct mb_l36, kernel_variant)),
    L("l36.n_runs", offsetof(struct mb_l36, n_runs)), L("l36.n_overrun", offsetof(struct mb_l36, n_overrun)),
    L("l36.n_reentry", offsetof(struct mb_l36, n_reentry)),
    L("l36.beu_accrued", offsetof(struct mb_l36, beu_accrued)),
    L("l36.n_traps", offsetof(struct mb_l36, n_traps)), L("l36.n_crc_fail", offsetof(struct mb_l36, n_crc_fail)),
    L("l36.variants_built", offsetof(struct mb_l36, variants_built)),
    L("l36.n_host_gone", offsetof(struct mb_l36, n_host_gone)),
    L("l36.entry_flags", offsetof(struct mb_l36, entry_flags)),
    L("l36.n_crc_retry", offsetof(struct mb_l36, n_crc_retry)),
    L("l37.dst", offsetof(struct mb_l37, dst)), L("l37.len", offsetof(struct mb_l37, len)),
    L("l37.crc32", offsetof(struct mb_l37, crc32)), L("l37.kind", offsetof(struct mb_l37, kind)),
    L("l37.load_seq", offsetof(struct mb_l37, load_seq)), L("l37.desc_crc32", offsetof(struct mb_l37, desc_crc32)),
    L("l38.host_gone", offsetof(struct mb_l38, host_gone)),
    L("l39.kind", offsetof(struct mb_l39, kind)), L("l39.seq", offsetof(struct mb_l39, seq)),
    L("l39.ret", offsetof(struct mb_l39, ret)), L("l39.cycles", offsetof(struct mb_l39, cycles)),
    L("l39.data", offsetof(struct mb_l39, data)),
    L("call.policy", offsetof(struct mb_call_args, policy)), L("call.arg_addr", offsetof(struct mb_call_args, arg_addr)),
    L("call.arg_len", offsetof(struct mb_call_args, arg_len)), L("call.seq", offsetof(struct mb_call_args, seq)),
    /* multi-hart (mh.h) */
    L("MB_CMD_MH", MB_CMD_MH), L("MB_L39_MH", MB_L39_MH), L("MB_MH_CAP", MB_MH_CAP), L("MB_MH_TIMEOUT_US", MB_MH_TIMEOUT_US),
    L("l36.mh", offsetof(struct mb_l36, mh)), L("MON_API_VERSION", MON_API_VERSION),
    L("api.mh_ready", offsetof(struct mon_api, mh_ready)), L("api.wapi", offsetof(struct mon_api, wapi)),
    L("MH_BASE", MH_BASE), L("MH_SIZE", MH_SIZE), L("MH_SEC_RNMI_EXC", MH_SEC_RNMI_EXC),
    L("MH_CTL_MAGIC", MH_CTL_MAGIC), L("MH_STAT_MAGIC", MH_STAT_MAGIC),
    L("mh.ctl", MH_CTL_ADDR), L("mh.stat1", MH_STAT_ADDR(1)), L("mh.aux1", MH_AUX_ADDR(1)), L("mh.stat3", MH_STAT_ADDR(3)),
    L("mh.trap_top1", MH_TRAP_TOP(1)), L("mh.priv1", MH_PRIV_ADDR(1)), L("mh.stack_top1", MH_STACK_TOP(1)),
    L("mh.stack_top3", MH_STACK_TOP(3)),
    L("mh_ctl.epoch", offsetof(struct mh_ctl, epoch)), L("mh_ctl.gen", offsetof(struct mh_ctl, gen)),
    L("mh_ctl.mask", offsetof(struct mh_ctl, mask)), L("mh_ctl.fn", offsetof(struct mh_ctl, fn)),
    L("mh_ctl.t_dispatch", offsetof(struct mh_ctl, t_dispatch)), L("mh_ctl.n_dispatch", offsetof(struct mh_ctl, n_dispatch)),
    L("mh_stat.state", offsetof(struct mh_stat, state)), L("mh_stat.heartbeat", offsetof(struct mh_stat, heartbeat)),
    L("mh_stat.entries", offsetof(struct mh_stat, entries)), L("mh_stat.done_gen", offsetof(struct mh_stat, done_gen)),
    L("mh_stat.trap_gen", offsetof(struct mh_stat, trap_gen)), L("mh_stat.traps", offsetof(struct mh_stat, traps)),
    L("mh_stat.mncause", offsetof(struct mh_stat, mncause)), L("mh_stat.mnepc", offsetof(struct mh_stat, mnepc)),
    L("mh_stat.mode", offsetof(struct mh_stat, mode)), L("mh_stat.mip", offsetof(struct mh_stat, mip)),
    L("mh_aux.result", offsetof(struct mh_aux, result)), L("mh_aux.t_seen", offsetof(struct mh_aux, t_seen)),
    L("mh_aux.gen_seen", offsetof(struct mh_aux, gen_seen)), L("mh_aux.wakes", offsetof(struct mh_aux, wakes)),
    L("mh_aux.beu_accrued", offsetof(struct mh_aux, beu_accrued)), L("mh_aux.fence_i", offsetof(struct mh_aux, fence_i)),
    L("MH_ST_IDLE", MH_ST_IDLE), L("MH_ST_BUSY", MH_ST_BUSY), L("MH_MODE_WFI", MH_MODE_WFI), L("MH_MODE_POLL", MH_MODE_POLL),
    L("MHB_BASE", MHB_BASE), L("MHB_DATA", MHB_DATA), L("MHB_END", MHB_END), L("mhb.res0", MHB_RES(0)),
    L("mhb.res3", MHB_RES(3)), L("MHB_WS_DEFAULT_BASE", MHB_WS_DEFAULT_BASE), L("MHB_WS_DEFAULT_LEN", MHB_WS_DEFAULT_LEN),
    L("MHB_WS_ALIGN", MHB_WS_ALIGN), L("MHB_WINDOW_US", MHB_WINDOW_US), L("MHB_LEAD_US", MHB_LEAD_US),
    L("MHB_K_MLP", MHB_K_MLP), L("MHB_K_TRAP", MHB_K_TRAP), L("MHB_N_KERNELS", MHB_N_KERNELS),
    L("mhb_job.t_start", offsetof(struct mhb_job, t_start)), L("mhb_job.dispatched", offsetof(struct mhb_job, dispatched)),
    L("mhb_res.tag", offsetof(struct mhb_res, tag)), L("mhb_res.cycles", offsetof(struct mhb_res, cycles)),
    L("mhb_res.dt", offsetof(struct mhb_res, dt)), L("mhb_res.sink", offsetof(struct mhb_res, sink)),
    L("l39mh.wake_ticks", offsetof(struct mb_l39_mh, wake_ticks)), L("l39mh.trapped", offsetof(struct mb_l39_mh, trapped)),
};
#undef L

X86_EXPORT int mon_x86_layout_n(void) { return (int)(sizeof layout / sizeof layout[0]); }
X86_EXPORT const char *mon_x86_layout_name(int i) { return layout[i].name; }
X86_EXPORT uint64_t mon_x86_layout_value(int i) { return layout[i].value; }
