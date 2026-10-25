/* Resident monitor for one X280 hart (PLAN.md WP-3.2, D12; mailbox layout v2 in mailbox.h).
 *
 * Why a monitor and not the WBC loop: harts leave reset once per chip reset
 * (tt-isa-documentation BlackholeA0/L2CPUTile/README.md:29-30), so the first image must be able to take
 * new code (LOAD / FENCE_I / CALL), survive a faulting payload (trap recovery) and be pulled back from a
 * hung one (RNMI park, crt0 re-arm). Commands: NOP INFER LOAD FENCE_I CALL PARK SELFTEST BENCH MH.
 *
 * Single writer per line: this file writes only L1, L35, L36 and L39 (and, through a CALLed payload,
 * L39 bytes 24-63 or a BENCH output array the host asked for). Host lines are only read. Harts 1-3 (mh.c) never
 * run this file's code; hart 0 talks to them through the MH region's ctl line (mh.h).
 * Freestanding: no libc; the same source builds for the X280, for spike (SPIKE_HOST) and for x86.
 */
#include "monitor.h"

struct mon_persist g_persist __attribute__((section(".persist"), aligned(64)));
struct mon_rt g_rt __attribute__((aligned(64)));

#define HB_TICKS        (PLAT_MTIME_HZ / 100000u)                 /* heartbeat every 10 us of mtime */
#define CRC_RETRY_TICKS (PLAT_MTIME_HZ / 1000000u * MB_CRC_RETRY_US)
#define RESTART_TICKS   (PLAT_MTIME_HZ / 10000u)                  /* 100 us pause after a trap restart */

/* ---- small helpers ------------------------------------------------------------------------------ */

static void mon_memset(void *d, int c, uint32_t n)
{
    volatile uint8_t *p = d;   /* volatile: keeps clang from turning this into a memset call */
    while (n--) *p++ = (uint8_t)c;
}

/* ---- zlib CRC-32 (reflected 0xEDB88320, init and final ~0): L0, obs, BENCH action, L37, LOAD data ----------------
 *
 * Default: slicing-by-8 over 8 KiB of tables in .rodata (crc32_tab.h, generated and checked against zlib by
 * x280/tools/gen_crc32_tab.py). LIM is not data-cacheable (Caches.md:26-31), so each table entry is an uncached
 * load: about 28 cycles when the next instruction needs it (S2 E2m pointer chase), about 7.8 each when issued back
 * to back (SELFTEST). One 8 B round issues 8 independent loads, and the 4 for the high word do not depend on the
 * CRC chain: ~28 + 7 x 7.8 cycles per 8 B against ~57 instructions per byte bitwise (R33: ~124 K instructions
 * per INFER, mostly this). -DMON_CRC_BITWISE keeps the bitwise loop (no table) for tests; both are zlib-exact. */
_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the word steps take bytes in little-endian order");

#if defined(MON_CRC_BITWISE)

static inline uint32_t crc_step32(uint32_t c, uint32_t w)
{
    c ^= w;   /* reflected CRC: 4 little-endian bytes at once */
    for (int k = 0; k < 32; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    return c;
}

static inline uint32_t crc_step8(uint32_t c, uint8_t b)
{
    c ^= b;
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    return c;
}

static inline uint32_t crc_step64(uint32_t c, uint64_t v)
{
    return crc_step32(crc_step32(c, (uint32_t)v), (uint32_t)(v >> 32));
}

#else

#include "crc32_tab.h"
#define CT mon_crc32_tab

static inline uint32_t crc_step8(uint32_t c, uint8_t b)
{
    return (c >> 8) ^ CT[0][(c ^ b) & 0xFFu];
}

static inline uint32_t crc_step32(uint32_t c, uint32_t w)
{
    c ^= w;
    return CT[3][c & 0xFFu] ^ CT[2][(c >> 8) & 0xFFu] ^ CT[1][(c >> 16) & 0xFFu] ^ CT[0][c >> 24];
}

/* 8 little-endian bytes: the high word's four lookups are independent of the chain, so they go first. */
static inline uint32_t crc_step64(uint32_t c, uint64_t v)
{
    uint32_t hi = (uint32_t)(v >> 32);
    uint32_t h = CT[3][hi & 0xFFu] ^ CT[2][(hi >> 8) & 0xFFu] ^ CT[1][(hi >> 16) & 0xFFu] ^ CT[0][hi >> 24];
    c ^= (uint32_t)v;
    return h ^ CT[7][c & 0xFFu] ^ CT[6][(c >> 8) & 0xFFu] ^ CT[5][(c >> 16) & 0xFFu] ^ CT[4][c >> 24];
}

#endif

/* zlib crc32(crc, p, n); crc = 0 to start, or a previous result to continue. Bytes up to 4 B alignment, one 4 B
 * step up to 8 B alignment, 8 B steps (each data word loaded one step ahead, so its uncached load overlaps the
 * table loads of the step before), then a 4 B step and bytes for the tail. */
uint32_t mon_crc32(uint32_t crc, const void *p, uint32_t n)
{
    const uint8_t *b = p;
    uint32_t c = ~crc;
    while (n && ((uintptr_t)b & 3u)) { c = crc_step8(c, *b++); n--; }
    if (n >= 4 && ((uintptr_t)b & 4u)) { c = crc_step32(c, *(const uint32_t *)b); b += 4; n -= 4; }
    if (n >= 8) {
        uint64_t v = *(const uint64_t *)b;
        for (b += 8, n -= 8; n >= 8; b += 8, n -= 8) {
            uint64_t next = *(const uint64_t *)b;
            c = crc_step64(c, v);
            v = next;
        }
        c = crc_step64(c, v);
    }
    if (n >= 4) { c = crc_step32(c, *(const uint32_t *)b); b += 4; n -= 4; }
    while (n--) c = crc_step8(c, *b++);
    return ~c;
}

static inline int overlaps(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1)
{
    return a0 < b1 && b0 < a1;
}

static uint64_t api_mtime(void) { return plat_mtime(); }
static uint64_t api_mcycle(void) { return plat_mcycle(); }
static void *api_ptr(uint64_t phys) { return plat_ptr(phys); }

void mon_heartbeat(void)
{
    uint64_t now = plat_mtime();
    MB_PTR(struct mb_l1, MB_L1_STATUS)->heartbeat = now;
    g_rt.last_hb = now;
}

static void set_state(uint32_t st)
{
    g_rt.state = st;
    MB_PTR(struct mb_l1, MB_L1_STATUS)->state = st;
}

static void write_ident(void)
{
    volatile struct mb_l36 *id = MB_PTR(struct mb_l36, MB_L36_IDENT);
    id->kernel_variant = g_persist.kernel_variant;
    id->n_runs = g_persist.n_runs;
    id->n_overrun = g_persist.n_overrun;
    id->n_reentry = g_persist.n_reentry;
    id->beu_accrued = g_persist.beu_accrued;
    id->n_traps = g_persist.n_traps;
    id->n_crc_fail = g_persist.n_crc_fail;
    id->n_host_gone = g_persist.n_host_gone;
    id->entry_flags = g_persist.entry_flags;
    id->n_crc_retry = g_persist.n_crc_retry;
    id->mh = MB_MH_CAP | (g_rt.mh_ready & MB_MH_READY_MASK);
}

/* Finish command `s`: L1 fields, L36 counters, release fence, then seq_done last (§1.4.4 step 3). */
static void complete(uint32_t s, uint32_t st, uint64_t mcause, uint64_t mepc, uint64_t mtval)
{
    volatile struct mb_l1 *l1 = MB_PTR(struct mb_l1, MB_L1_STATUS);
    volatile struct mb_l0 *l0 = MB_PTR(struct mb_l0, MB_L0_CMD);
    l1->mcause = mcause;
    l1->mepc = mepc;
    l1->mtval = mtval;
    if (l0->seq_req != s) g_persist.n_overrun++;   /* the host gave up on s and moved on */
    write_ident();
    set_state(st);
    mon_heartbeat();
    MB_FENCE_REL();
    l1->seq_done = s;
    g_rt.last_seq = s;
}

static void fail(uint32_t s, uint64_t code, uint64_t mepc, uint64_t mtval)
{
    complete(s, MB_STATE_FAULT, code, mepc, mtval);
}

/* The trap handler already dumped mcause/mepc/mtval to L1 (trap.c); finish `s` with that dump. */
static void fail_trapped(uint32_t s)
{
    complete(s, MB_STATE_FAULT, g_rt.fault_mcause, g_rt.fault_mepc, g_rt.fault_mtval);
}

/* Without MB_FLAG_VARIANT_SET the variant follows the registered blob's weight type (wbc_mlp.h header field
 * wdtype): fp16 -> V2 if built, else V2S; anything else -> MON_DEFAULT_VARIANT (V0, the fp32 reference). With the
 * fixed V0 default, an INFER on the DP4-default fp16 blob that forgot the variant bits failed with WBC_EDTYPE
 * (kernel lane open issue); x280d always sets them, this is the safety net. */
static uint32_t pick_variant(uint32_t flags)
{
    if (flags & MB_FLAG_VARIANT_SET) return (flags >> MB_FLAG_VARIANT_SHIFT) & MB_FLAG_VARIANT_MASK;
#if defined(WBC_WDTYPE_F16) && defined(WBC_VARIANT_V2S)
    if (g_persist.blob_ok &&
        ((const volatile wbc_blob_hdr_t *)plat_ptr(g_persist.blob_addr))->wdtype == WBC_WDTYPE_F16)
        return (mon_variants_built() >> WBC_VARIANT_V2) & 1u ? WBC_VARIANT_V2 : WBC_VARIANT_V2S;
#endif
    return MON_DEFAULT_VARIANT;
}

/* ---- reading a command ---------------------------------------------------------------------------- */

enum { RD_OK = 0, RD_ABANDON = -1 };   /* > 0: CRC failure bits (1 body, 2 obs) */

static int cmd_needs_obs(uint32_t cmd) { return cmd == MB_CMD_INFER || cmd == MB_CMD_BENCH; }

/* Snapshot L0 (and, for INFER/BENCH, copy L2-L34 into g_rt.obs) and check both CRCs. A mismatch is
 * re-read until MB_CRC_RETRY_US of mtime has passed. seq_req is re-read after every snapshot: if it has
 * moved on, the command is abandoned without a completion (the host gave up on it) and the newer one is
 * picked up by the next poll. Residual: a host that writes the next body before bumping seq_req while
 * the previous command is still running (i.e. after its own timeout) can have that body run twice. */
static int read_command(uint32_t s, struct mb_l0 *c)
{
    volatile uint32_t *w = (volatile uint32_t *)MB_PTR(struct mb_l0, MB_L0_CMD);
    volatile uint64_t *obs = (volatile uint64_t *)plat_ptr(MB_L(MB_L2_OBS));
    uint32_t *cw = (uint32_t *)c;
    uint64_t t0 = plat_mtime();
    int retried = 0;
    for (;;) {
        for (uint32_t i = 0; i < 16; i++) cw[i] = w[i];
        int bad = 0;
        if (mon_crc32(0, c, MB_L0_CRC_BYTES) != c->crc32) {
            bad |= 1;
        } else if (cmd_needs_obs(c->cmd)) {
            /* Every obs word read once, in order, into g_rt.obs, and the CRC is over those copies. Word i + 1 is
             * read before word i's CRC step, so its uncached load overlaps that step's table loads. */
            enum { N = MB_OBS_LINES * MB_LINE / 8 };
            uint32_t cr = ~0u;
            uint64_t v = obs[0];
            for (uint32_t i = 1; i < N; i++) {
                uint64_t next = obs[i];
                g_rt.obs[i - 1] = v;
                cr = crc_step64(cr, v);
                v = next;
            }
            g_rt.obs[N - 1] = v;
            cr = crc_step64(cr, v);
            if (~cr != c->obs_crc32) bad |= 2;
        }
        /* seqlock re-check: the bytes just read belong to `s` only if seq_req has not moved on */
        MB_FENCE_ACQ();
        if (w[15] != s) return RD_ABANDON;
        if (!bad) return RD_OK;
        if (!retried) { g_persist.n_crc_retry++; retried = 1; }
        if (plat_mtime() - t0 >= CRC_RETRY_TICKS) { g_persist.n_crc_fail++; return bad; }
    }
}

/* ---- commands --------------------------------------------------------------------------------------- */

static void cmd_infer(uint32_t s, const struct mb_l0 *c)
{
    if (c->policy > WBC_POLICY_WALK) { fail(s, MB_ERR_ARG, c->policy, MB_ARG_POLICY); return; }
    if (!g_persist.blob_ok) { fail(s, MB_ERR_BLOB, 0, 0); return; }
    uint32_t variant = pick_variant(c->flags);
    g_persist.kernel_variant = variant;
    const void *blob = plat_ptr(g_persist.blob_addr);
    volatile int rc = 0;
    volatile uint64_t cyc = 0;
    g_rt.guard_active = 1;
    if (mon_setjmp(g_rt.guard) == 0) {
        uint64_t c0 = plat_mcycle();
        rc = MON_WBC_INFER(blob, c->policy, variant, (const float *)g_rt.obs, g_rt.act, g_rt.scratch);
        cyc = plat_mcycle() - c0;
        g_rt.guard_active = 0;
    } else {
        fail_trapped(s);
        return;
    }
    if (rc < 0) { fail(s, MB_ERR_KERNEL, variant, (uint64_t)(int64_t)rc); return; }
    volatile struct mb_l35 *a = MB_PTR(struct mb_l35, MB_L35_ACT);
    for (int j = 0; j < WBC_ACT_DIM; j++) a->action[j] = g_rt.act[j];
    a->seq_echo = s;
    MB_FENCE_REL();   /* L35 before L1 (§1.4.4 step 3) */
    volatile struct mb_l1 *l1 = MB_PTR(struct mb_l1, MB_L1_STATUS);
    uint32_t c32 = cyc > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)cyc;
    if (c32 > g_persist.cycles_max) g_persist.cycles_max = c32;
    l1->cycles_last = c32;
    l1->cycles_max = g_persist.cycles_max;
    g_persist.n_runs++;
    complete(s, MB_STATE_IDLE, 0, 0, 0);
}

static void cmd_load(uint32_t s)
{
    volatile uint32_t *dw = (volatile uint32_t *)MB_PTR(struct mb_l37, MB_L37_LOAD);
    struct mb_l37 d;
    uint32_t *w = (uint32_t *)&d;
    for (uint32_t i = 0; i < 16; i++) w[i] = dw[i];
    if (mon_crc32(0, &d, MB_L37_CRC_BYTES) != d.desc_crc32 || d.load_seq != s) {
        fail(s, MB_ERR_DESC, d.load_seq, d.desc_crc32);
        return;
    }
    uint64_t lo = d.dst, hi = d.dst + d.len;
    if (d.kind > MB_LOAD_DATA) { fail(s, MB_ERR_ARG, d.kind, MB_ARG_KIND); return; }
    if (d.dst & (MB_LINE - 1)) { fail(s, MB_ERR_ARG, d.dst, MB_ARG_ALIGN); return; }
    if (d.len == 0) { fail(s, MB_ERR_ARG, d.len, MB_ARG_LEN); return; }
    if (lo < MB_PAYLOAD_BASE || hi > MB_LIM_END) { fail(s, MB_ERR_ARG, d.dst, MB_ARG_RANGE); return; }
    /* The host has already overwritten [lo, hi): forget whatever was registered there. */
    if (g_persist.blob_ok && overlaps(lo, hi, g_persist.blob_addr, g_persist.blob_addr + g_persist.blob_len))
        g_persist.blob_ok = 0;
    if (g_persist.code_ok && overlaps(lo, hi, g_persist.code_lo, g_persist.code_hi))
        g_persist.code_ok = 0;
    volatile uint32_t crc = 0;
    volatile int rc = 0;
    g_rt.guard_active = 1;
    if (mon_setjmp(g_rt.guard) == 0) {
        const uint8_t *p = plat_ptr(lo);
        uint32_t c = 0, left = d.len;
        while (left) {
            uint32_t n = left > 4096u ? 4096u : left;
            c = mon_crc32(c, p, n);
            p += n;
            left -= n;
            mon_heartbeat();
        }
        crc = c;
        if (crc == d.crc32 && d.kind == MB_LOAD_BLOB) rc = MON_WBC_BLOB_CHECK(plat_ptr(lo), d.len);
        g_rt.guard_active = 0;
    } else {
        fail_trapped(s);
        return;
    }
    if (crc != d.crc32) { fail(s, MB_ERR_LOAD_CRC, d.dst, crc); return; }
    if (d.kind == MB_LOAD_BLOB) {
        if (rc < 0) { fail(s, MB_ERR_BLOB, d.dst, (uint64_t)(int64_t)rc); return; }
        g_persist.blob_addr = lo;
        g_persist.blob_len = d.len;
        g_persist.blob_ok = 1;
    } else if (d.kind == MB_LOAD_CODE) {
        g_persist.code_lo = lo;
        g_persist.code_hi = hi;
        g_persist.code_ok = 1;
        g_persist.code_fenced = 0;
    }
    complete(s, MB_STATE_IDLE, 0, 0, 0);
}

static void cmd_fence_i(uint32_t s)
{
    plat_fence_i();
    if (g_persist.code_ok) g_persist.code_fenced = 1;
    complete(s, MB_STATE_IDLE, 0, 0, 0);
}

static void cmd_call(uint32_t s, const struct mb_l0 *c)
{
    uint64_t e = c->arg_entry;
    if (!g_persist.code_ok || !g_persist.code_fenced || e < g_persist.code_lo || e >= g_persist.code_hi ||
        (e & 1u)) {
        fail(s, MB_ERR_NOCODE, e, g_persist.code_ok | (g_persist.code_fenced << 1));
        return;
    }
    struct mb_call_args *a = &g_rt.args;
    a->policy = c->policy;
    a->flags = c->flags;
    a->tick = c->tick;
    a->arg_addr = c->arg_addr;
    a->arg_entry = e;
    a->arg_len = c->arg_len;
    a->seq = s;
    volatile uint64_t ret = 0, cyc = 0;
    g_rt.guard_active = 1;
    if (mon_setjmp(g_rt.guard) == 0) {
        uint64_t c0 = plat_mcycle();
        ret = plat_call(e, a, &g_rt.api);
        cyc = plat_mcycle() - c0;
        g_rt.guard_active = 0;
    } else {
        fail_trapped(s);
        return;
    }
    volatile struct mb_l39 *r = MB_PTR(struct mb_l39, MB_L39_SCRATCH);
    r->kind = MB_L39_CALL;
    r->seq = s;
    r->ret = ret;
    r->cycles = cyc;
    complete(s, MB_STATE_IDLE, 0, 0, 0);
}

static void cmd_selftest(uint32_t s, const struct mb_l0 *c)
{
    uint64_t lo = c->arg_addr ? c->arg_addr : MB_LIM_BASE;
    uint64_t len = c->arg_len ? c->arg_len : MB_LIM_SIZE;
    if ((lo | len) & (MB_LINE - 1)) { fail(s, MB_ERR_ARG, lo, MB_ARG_ALIGN); return; }
    if (lo < MB_LIM_BASE || lo + len > MB_LIM_END) { fail(s, MB_ERR_ARG, lo, MB_ARG_RANGE); return; }
    struct mb_l39_selftest st;
    mon_memset(&st, 0, sizeof st);
    plat_beu_read(&st.beu_cause, &st.beu_value, &st.beu_accrued_before);
    volatile uint64_t x = 0, c0 = 0, c1 = 0, t0 = 0, t1 = 0;
    volatile uint32_t lines = 0;
    g_rt.guard_active = 1;
    if (mon_setjmp(g_rt.guard) == 0) {
        t0 = plat_mtime();
        c0 = plat_mcycle();
        const volatile uint64_t *p = plat_ptr(lo);
        uint64_t acc = 0;
        uint32_t n = (uint32_t)(len / MB_LINE);
        for (uint32_t i = 0; i < n; i++, p += 8) {
            acc ^= p[0] ^ p[1] ^ p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7];   /* every byte of the line */
            if ((i & 511u) == 511u) mon_heartbeat();
            lines = i + 1;
        }
        c1 = plat_mcycle();
        t1 = plat_mtime();
        x = acc;
        g_rt.guard_active = 0;
    } else {
        fail_trapped(s);   /* mtval = the faulting address of a bad line */
        return;
    }
    uint32_t cause_after;
    uint64_t value_after;
    plat_beu_read(&cause_after, &value_after, &st.beu_accrued_after);
    st.lines = lines;
    st.mtime_ticks = (uint32_t)(t1 - t0);
    st.cyc_per_mtime_x1000 = t1 > t0 ? (uint32_t)((c1 - c0) * 1000u / (t1 - t0)) : 0;
    st.misa_v = (plat_misa() & PLAT_MISA_V) ? 1u : 0u;
    st.vlenb = plat_vlenb();
    g_persist.beu_accrued = st.beu_accrued_after;
    MB_PTR(struct mb_l36, MB_L36_IDENT)->beu_accrued = st.beu_accrued_after;
    volatile struct mb_l39 *r = MB_PTR(struct mb_l39, MB_L39_SCRATCH);
    volatile uint32_t *rd = (volatile uint32_t *)r->data;
    const uint32_t *sw = (const uint32_t *)&st;
    for (uint32_t i = 0; i < sizeof st / 4; i++) rd[i] = sw[i];
    r->kind = MB_L39_SELFTEST;
    r->seq = s;
    r->ret = x;
    r->cycles = c1 - c0;
    complete(s, MB_STATE_IDLE, 0, 0, 0);
}

static void cmd_bench(uint32_t s, const struct mb_l0 *c)
{
    if (c->policy > WBC_POLICY_WALK) { fail(s, MB_ERR_ARG, c->policy, MB_ARG_POLICY); return; }
    if (!g_persist.blob_ok) { fail(s, MB_ERR_BLOB, 0, 0); return; }
    uint32_t iters = c->arg_len ? c->arg_len : 1u;
    uint64_t out = c->arg_addr;
    if (out) {
        uint64_t end = out + 4ull * iters;
        if (out & 3u) { fail(s, MB_ERR_ARG, out, MB_ARG_ALIGN); return; }
        if (out < MB_PAYLOAD_BASE || end > MB_LIM_END) { fail(s, MB_ERR_ARG, out, MB_ARG_RANGE); return; }
        if ((g_persist.blob_ok &&
             overlaps(out, end, g_persist.blob_addr, g_persist.blob_addr + g_persist.blob_len)) ||
            (g_persist.code_ok && overlaps(out, end, g_persist.code_lo, g_persist.code_hi))) {
            fail(s, MB_ERR_ARG, out, MB_ARG_OVERLAP);
            return;
        }
    }
    uint32_t variant = pick_variant(c->flags);
    g_persist.kernel_variant = variant;
    const void *blob = plat_ptr(g_persist.blob_addr);
    static struct mb_l39_bench b;   /* .bss: keeps it off the (uncached) stack across the setjmp */
    static uint32_t first[WBC_ACT_DIM];
    mon_memset(&b, 0, sizeof b);
    b.cyc_min = 0xFFFFFFFFu;
    b.variant = variant;
    b.policy = c->policy;
    volatile uint64_t total = 0;
    g_rt.guard_active = 1;
    if (mon_setjmp(g_rt.guard) == 0) {
        volatile uint32_t *o = out ? (volatile uint32_t *)plat_ptr(out) : 0;
        for (uint32_t i = 0; i < iters; i++) {
            uint64_t c0 = plat_mcycle();
            int rc = MON_WBC_INFER(blob, c->policy, variant, (const float *)g_rt.obs, g_rt.act, g_rt.scratch);
            uint64_t d = plat_mcycle() - c0;
            uint32_t d32 = d > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)d;
            b.rc = rc;
            if (rc < 0) break;
            const uint32_t *ab = (const uint32_t *)g_rt.act;
            if (i == 0) {
                for (int j = 0; j < WBC_ACT_DIM; j++) first[j] = ab[j];
            } else {
                for (int j = 0; j < WBC_ACT_DIM; j++) if (ab[j] != first[j]) { b.nondet++; break; }
            }
            if (o) o[i] = d32;
            if (d32 < b.cyc_min) b.cyc_min = d32;
            if (d32 > b.cyc_max) b.cyc_max = d32;
            b.cyc_sum += d;
            b.iters = i + 1;
            mon_heartbeat();
        }
        total = b.cyc_sum;
        g_rt.guard_active = 0;
    } else {
        fail_trapped(s);
        return;
    }
    if (b.iters) {
        volatile struct mb_l35 *a = MB_PTR(struct mb_l35, MB_L35_ACT);
        for (int j = 0; j < WBC_ACT_DIM; j++) a->action[j] = g_rt.act[j];
        a->seq_echo = s;
        b.act_crc32 = mon_crc32(0, g_rt.act, WBC_ACT_DIM * 4);
    }
    volatile struct mb_l39 *r = MB_PTR(struct mb_l39, MB_L39_SCRATCH);
    volatile uint32_t *rd = (volatile uint32_t *)r->data;
    const uint32_t *bw = (const uint32_t *)&b;
    for (uint32_t i = 0; i < sizeof b / 4; i++) rd[i] = bw[i];
    r->kind = MB_L39_BENCH;
    r->seq = s;
    r->ret = b.iters;
    r->cycles = total;
    MB_FENCE_REL();
    if (b.rc < 0) { fail(s, MB_ERR_KERNEL, variant, (uint64_t)(int64_t)b.rc); return; }
    complete(s, MB_STATE_IDLE, 0, 0, 0);
}

/* MB_CMD_MH (mh.c): the ping is a NOP job on the ready harts of the mask; the record says which completed and how
 * long each took to pick it up. The secondaries' own stat lines are for the host to read (LIM, mh.h). */
static void cmd_mh(uint32_t s, const struct mb_l0 *c)
{
    if (c->policy != MB_MH_OP_PING) { fail(s, MB_ERR_ARG, c->policy, MB_ARG_POLICY); return; }
    uint32_t mask = c->flags & MH_SEC_MASK;
    uint32_t us = c->arg_len ? c->arg_len : MB_MH_TIMEOUT_US;
    if (!mask) mask = MH_SEC_MASK;
    if (us > 1000000u) { fail(s, MB_ERR_ARG, us, MB_ARG_LEN); return; }
    static struct mb_l39_mh m;   /* .bss, like cmd_bench's record */
    mon_memset(&m, 0, sizeof m);
    uint64_t c0 = plat_mcycle();
    uint32_t done = mh_ping(mask, (uint64_t)us * (PLAT_MTIME_HZ / 1000000u), &m);
    uint64_t cyc = plat_mcycle() - c0;
    g_rt.mh_ready = m.ready;
    volatile struct mb_l39 *r = MB_PTR(struct mb_l39, MB_L39_SCRATCH);
    volatile uint32_t *rd = (volatile uint32_t *)r->data;
    const uint32_t *mw = (const uint32_t *)&m;
    for (uint32_t i = 0; i < sizeof m / 4; i++) rd[i] = mw[i];
    r->kind = MB_L39_MH;
    r->seq = s;
    r->ret = done;
    r->cycles = cyc;
    complete(s, MB_STATE_IDLE, 0, 0, 0);
}

static void cmd_park(uint32_t s)
{
    complete(s, MB_STATE_PARKED, 0, 0, 0);
    plat_park_forever();   /* until the host's RNMI: park stub -> crt0 (warm) */
}

static void run_command(uint32_t s)
{
    struct mb_l0 c;
    set_state(MB_STATE_BUSY);
    int rd = read_command(s, &c);
    if (rd == RD_ABANDON) return;   /* last_seq unchanged: the newer seq_req is picked up next */
    if (rd > 0) { fail(s, MB_ERR_CRC, c.crc32, (uint64_t)rd); return; }
    if (c.magic != MB_MAGIC) { fail(s, MB_ERR_MAGIC, 0, c.magic); return; }
    switch (c.cmd) {
    case MB_CMD_NOP:      complete(s, MB_STATE_IDLE, 0, 0, 0); break;
    case MB_CMD_INFER:    cmd_infer(s, &c); break;
    case MB_CMD_LOAD:     cmd_load(s); break;
    case MB_CMD_FENCE_I:  cmd_fence_i(s); break;
    case MB_CMD_CALL:     cmd_call(s, &c); break;
    case MB_CMD_PARK:     cmd_park(s); break;
    case MB_CMD_SELFTEST: cmd_selftest(s, &c); break;
    case MB_CMD_BENCH:    cmd_bench(s, &c); break;
    case MB_CMD_MH:       cmd_mh(s, &c); break;
    default:              fail(s, MB_ERR_CMD, 0, c.cmd); break;
    }
}

static int host_gone(void)
{
    return MB_PTR(struct mb_l38, MB_L38_CLEANUP)->host_gone == MB_HOST_GONE;
}

/* x280d died (KMD cleanup wrote 'GONE' into L38): stop executing commands, adopt seq_req so a stale
 * one is never run after a new x280d attaches and clears L38. */
static void on_host_gone(uint32_t s)
{
    if (g_rt.state != MB_STATE_HOST_GONE) g_persist.n_host_gone++;
    complete(s, MB_STATE_HOST_GONE, 0, 0, 0);
}

/* ---- loop --------------------------------------------------------------------------------------------- */

void mon_poll_once(void)
{
    volatile struct mb_l0 *l0 = MB_PTR(struct mb_l0, MB_L0_CMD);
    if (g_rt.restart_pending) {
        /* An unguarded trap restarted the loop (trap.c): fail the command in flight, pause briefly so a
         * persistent fault cannot spin hot, then continue. */
        g_rt.restart_pending = 0;
        uint32_t s = l0->seq_req;
        uint64_t t0 = plat_mtime();
        while (plat_mtime() - t0 < RESTART_TICKS) {}
        fail_trapped(s);
        return;
    }
    uint32_t s = l0->seq_req;
    if (s != g_rt.last_seq) {
        if (host_gone()) { on_host_gone(s); return; }
        MB_FENCE_ACQ();   /* seq_req before the body and obs (§1.4.4 step 3) */
        run_command(s);
        return;
    }
    if (++g_rt.idle_iter & 15u) return;
    uint64_t now = plat_mtime();
    if (now - g_rt.last_hb < HB_TICKS) return;
    mon_heartbeat();
    int gone = host_gone();
    if (gone && g_rt.state != MB_STATE_HOST_GONE) {
        g_persist.n_host_gone++;
        MB_PTR(struct mb_l36, MB_L36_IDENT)->n_host_gone = g_persist.n_host_gone;
        set_state(MB_STATE_HOST_GONE);
    } else if (!gone && g_rt.state == MB_STATE_HOST_GONE) {
        set_state(MB_STATE_IDLE);
    }
}

/* Entry bookkeeping, before RNMIs are re-armed (crt0.S). NMIE is 1 only straight out of reset
 * (RNMIs.md:4) and is cleared by hardware on every RNMI (RNMIs.md:19), and software never clears it
 * (RNMIs.md:84), so NMIE = 0 here means we came through the park stub. */
void mon_entry(void)
{
    uint64_t mnstatus = plat_mnstatus();
    int cold = (mnstatus & 8u) != 0;
    int reloaded = 0;
    if (cold || g_persist.cookie != MON_COOKIE || g_persist.build_id != MON_BUILD_ID ||
        g_persist.size != sizeof g_persist) {
        reloaded = !cold;
        mon_memset(&g_persist, 0, sizeof g_persist);
        g_persist.cookie = MON_COOKIE;
        g_persist.build_id = MON_BUILD_ID;
        g_persist.size = sizeof g_persist;
        g_persist.kernel_variant = MON_DEFAULT_VARIANT;
        if (!cold) g_persist.n_reentry = 1;
    } else {
        g_persist.n_reentry++;
    }
    g_persist.entry_flags = (cold ? 0u : MB_ENTRY_WARM) | (reloaded ? MB_ENTRY_RELOADED : 0u) |
                            (wbc_kernel_is_stub ? MB_ENTRY_STUB_KERNEL : 0u);
    g_rt.entry_mnstatus = mnstatus;
    g_rt.entry_mncause = cold ? 0 : plat_mncause();
    g_rt.entry_mnepc = cold ? 0 : plat_mnepc();
}

/* Entry-time mailbox writes: L36 ident, L39 entry record, L1; then seq_done = the seq_req found
 * (adopted, never executed: a command in flight across a park is dropped and must be re-issued). */
void mon_start(void)
{
    volatile struct mb_l0 *l0 = MB_PTR(struct mb_l0, MB_L0_CMD);
    volatile struct mb_l1 *l1 = MB_PTR(struct mb_l1, MB_L1_STATUS);
    volatile struct mb_l36 *id = MB_PTR(struct mb_l36, MB_L36_IDENT);
    volatile struct mb_l39 *r = MB_PTR(struct mb_l39, MB_L39_SCRATCH);
    int warm = (g_persist.entry_flags & MB_ENTRY_WARM) != 0;

    mh_hart0_entry(!warm);           /* cold: a fresh ctl line and epoch; warm: harts 1-3 may be running */
    g_rt.mh_ready = mh_ready();
    struct mon_api *api = &g_rt.api;
    api->version = MON_API_VERSION;
    api->size = sizeof *api;
    api->mtime_hz = PLAT_MTIME_HZ;
    api->mtime = api_mtime;
    api->mcycle = api_mcycle;
    api->heartbeat = mon_heartbeat;
    api->ptr = api_ptr;
    api->result = (volatile uint8_t *)plat_ptr(MB_L(MB_L39_SCRATCH) + 24u);
    api->result_len = 40u;
    api->mh_ready = mh_ready;
    api->mh_run = mh_run;
    api->mh_wait = mh_wait;
    api->wapi = &mh_wapi;

    set_state(MB_STATE_BOOT);
    uint32_t s = l0->seq_req;
    g_rt.last_seq = s;

    id->misa = plat_misa();
    id->mtime_hz = PLAT_MTIME_HZ;
    id->build_id = MON_BUILD_ID;
    id->variants_built = mon_variants_built();
    write_ident();

    struct mb_l39_entry e;
    e.entry_kind = !warm ? 0u : (g_persist.entry_flags & MB_ENTRY_RELOADED) ? 2u : 1u;
    e.mnstatus = (uint32_t)g_rt.entry_mnstatus;
    e.mncause = g_rt.entry_mncause;
    e.mnepc = g_rt.entry_mnepc;
    e.mcycle = plat_mcycle();
    e.mtime = plat_mtime();
    volatile uint32_t *rd = (volatile uint32_t *)r->data;
    const uint32_t *ew = (const uint32_t *)&e;
    for (uint32_t i = 0; i < sizeof e / 4; i++) rd[i] = ew[i];
    r->kind = MB_L39_ENTRY;
    r->seq = s;
    r->ret = g_persist.n_reentry;
    r->cycles = e.mcycle;

    l1->mcause = warm ? g_rt.entry_mncause : 0;
    l1->mepc = warm ? g_rt.entry_mnepc : 0;
    l1->mtval = 0;
    l1->cycles_last = 0;
    l1->cycles_max = g_persist.cycles_max;
    l1->fw_ver = MB_FW_VER;
    l1->mhartid = plat_mhartid();
    l1->vlenb = plat_vlenb();
    l1->fw_magic = MB_MAGIC;
    mon_heartbeat();
    set_state(host_gone() ? MB_STATE_HOST_GONE : warm ? MB_STATE_REENTERED : MB_STATE_IDLE);
    MB_FENCE_REL();
    l1->seq_done = s;
}

void monitor_main(void)
{
    mon_start();
#if defined(SPIKE_HOST)
    sh_run();
#else
    for (;;) mon_poll_once();
#endif
}

void monitor_resume(void)
{
#if defined(SPIKE_HOST)
    sh_run();
#else
    for (;;) mon_poll_once();
#endif
}
