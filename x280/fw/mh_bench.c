/* 4-hart LIM bench payload (the 4-hart bench design, section C; contract and data lines in mh.h).
 *
 * Why: the MLP split across harts 0-3 pays only if LIM serves several harts in parallel. This payload measures
 * exactly that, with every hart of a mask running the same kernel on its own (or a shared) slice of the registered
 * weight blob, started together by a CLINT-mtime barrier, and each hart writing its own result line.
 *
 * LOAD (kind CODE) at MHB_BASE (0x081D_F400), FENCE_I, CALL arg_entry = MHB_BASE on hart 0:
 *   policy   kernel [7:0] (MHB_K_*), hart-0 kernel [15:8] (0 = the same): K3 on hart 0 under K6 load = loaded latency
 *   flags    hart mask [3:0] | pattern [7:4] (MHB_P_*) | k [11:8] | mode [13:12] (MHB_MODE_*)
 *   tick     window in mtime ticks (0 = MHB_WINDOW_US); at most 1 s
 *   arg_addr / arg_len  working set (0 = MHB_WS_DEFAULT_*): read only, inside [MB_PAYLOAD_BASE, MB_LIM_END), not over
 *            this payload; 64 B aligned base, length a multiple of 64 KiB. Nothing is written there, so INFER/BENCH
 *            may run concurrently (mode START), and the blob stays registered.
 * Hart 0 writes the job line (t_start = mtime + 50 us, t_end = t_start + window), dispatches the worker to the ready
 * harts of mask & 0xE through mon_api v2 (mh_run), runs its own share when bit 0 is set, waits, and summarises the
 * result lines into L39 bytes 24..63 (struct mhb_summary). Each worker spins on CLINT mtime (not LIM) until t_start,
 * takes mcycle and mtime, runs its kernel in ~4 KiB chunks until mtime >= t_end and writes MHB_RES(h).
 * Return value: the done mask.
 *
 * Unit stream of hart h (mailbox_layout.py mhb_stream reproduces it): addresses lo, lo + st, ... wrapping to lo at hi.
 *   K1-K8, unit U (64 B; K6-K8: 128/256/512 B), quarter Q = len / 4:
 *     P0 [base + hQ, +Q) stride U;  P1 the same with stride 4U;  P2 lo = base + hQ + hU, hi = base + (h+1)Q, stride 4U;
 *     P3 [base, base + Q) stride U; P4 lo = base + h 2^k, hi = base + len, stride 2^(k+2) (k = 6..12)
 *   K9/K10/K12: rows of 512 B, the hart's 128 B slice at +128 q (q = h; P3: q = 0), x[i] = floats from base, 2 KiB
 * Sink (checked against mailbox_layout.mhb_sink): K1 sum of the 8 words of each line, K2/K3 sum of word 0, K4 sum of the
 * 16 u32 words, K5-K8 sum over lanes of the per-lane byte sums mod 256 (vadd.vv e8 accumulators); 0 for K9-K12.
 * No AMO, LR or SC; the only stores are the job line (hart 0), the own result line (hart h) and L39 (hart 0).
 */
#include <stdint.h>
#include "mh.h"

#if defined(MON_X86)
#define MHB_ENTRY uint64_t x86_payload_mh_bench(const struct mb_call_args *a, const struct mon_api *api)
void x86_emulate_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval);   /* plat_x86.c */
#else
#define MHB_ENTRY __attribute__((section(".text.payload_entry"), used)) \
    uint64_t payload_entry(const struct mb_call_args *a, const struct mon_api *api)
#define MTIME_ADDR 0x0200BFF8ul   /* CLINT mtime (plat.h PLAT_MTIME) */
#endif

#define CHUNK 4096u
/* Inner loops stay loops (the payload must fit 2,560 B, and an unrolled timed loop would change what is measured). */
#define NOUNROLL _Pragma("clang loop unroll(disable)")

/* Unit footprint and bytes loaded per access of kernel k (K11: no LIM). The accesses per ~4 KiB chunk are each
 * kernel's own constant: 64 (K1-K5, K11), 4096 / U (K6-K8), 32 rows (K9, K10, K12). */
static const uint16_t KUNIT[MHB_N_KERNELS + 1] = { 0, 64, 64, 64, 64, 64, 128, 256, 512, 128, 128, 64, 128 };
static const uint16_t KBPA[MHB_N_KERNELS + 1] = { 0, 64, 8, 8, 64, 64, 128, 256, 512, 128, 0, 0, 128 };

struct run { uint64_t lo, hi, st, x, n, sink; uintptr_t d; uint64_t t_end; const struct mh_wapi *w; };

#if defined(MON_X86)
#define NOW(r) ((r)->w->mtime())
#else
#define NOW(r) (*(volatile uint64_t *)MTIME_ADDR)
#endif

#define TIMED(R, C, ...)                                                                      \
    do {                                                                                      \
        uint64_t p = (R)->lo, lo_ = (R)->lo, hi_ = (R)->hi, st_ = (R)->st, n_ = 0, a_ = 0;     \
        uintptr_t d_ = (R)->d;                                                                \
        for (;;) {                                                                            \
            NOUNROLL for (uint32_t j_ = 0; j_ < (C); j_++) {                                  \
                const volatile uint64_t *q = (const volatile uint64_t *)(p + d_);             \
                __VA_ARGS__;                                                                  \
                p += st_;                                                                     \
                if (p >= hi_) p = lo_;                                                        \
            }                                                                                 \
            n_ += (C);                                                                        \
            if (NOW(R) >= (R)->t_end) break;                                                  \
        }                                                                                     \
        (R)->n = n_;                                                                          \
        (R)->sink = a_;                                                                       \
    } while (0)

static void k_ld8x8(struct run *r) { TIMED(r, 64, a_ += q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7]); }
static void k_ld8(struct run *r) { TIMED(r, 64, a_ += q[0]); }
static void k_lw16(struct run *r)
{
    TIMED(r, 64, {
        const volatile uint32_t *w32 = (const volatile uint32_t *)q;
        a_ += (uint64_t)w32[0] + w32[1] + w32[2] + w32[3] + w32[4] + w32[5] + w32[6] + w32[7] + w32[8] + w32[9] +
              w32[10] + w32[11] + w32[12] + w32[13] + w32[14] + w32[15];
    });
}

/* K3: the next address depends on the loaded word (and zero), so each load waits for the previous one. */
static void k_chain(struct run *r)
{
#if defined(__riscv)
    TIMED(r, 64, {
        uint64_t v, t, pp = p + d_;
        (void)q;
        __asm__ volatile("ld %0, 0(%2)\n\tand %1, %0, zero\n\tadd %2, %2, %1" : "=&r"(v), "=&r"(t), "+r"(pp) :: "memory");
        a_ += v;
        p = pp - d_;   /* the next address depends on the loaded word */
    });
#else
    TIMED(r, 64, a_ += q[0]);
#endif
}

static void k_alu(struct run *r)
{
    uint64_t n = 0;
    for (;;) {
        NOUNROLL for (uint32_t j = 0; j < 64; j++) {
#if defined(__riscv)
#define ALU8 "addi t0, t0, 1\n addi t1, t1, 1\n addi t2, t2, 1\n addi t3, t3, 1\n" \
             "addi t4, t4, 1\n addi t5, t5, 1\n addi t6, t6, 1\n addi a7, a7, 1\n"
            __asm__ volatile(ALU8 ALU8 ALU8 ALU8 ALU8 ALU8 ALU8 ALU8 ::: "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7");
#else
            __asm__ volatile("" ::: "memory");
#endif
        }
        n += 64;
        if (NOW(r) >= r->t_end) break;
    }
    r->n = n;
    r->sink = 0;
}

#if defined(__riscv_vector)
/* K5-K8: vle8.v at LMUL m1..m8 (VL = 64 LMUL at VLEN 512; vtype from a register: vsetvl), each load added into the
 * u8 accumulator group at v16 (v8 / v16 are aligned for every LMUL up to m8). */
static void k_vle8(struct run *r, uint64_t lmul)
{
    uint64_t p = r->lo + r->d, lo = p, hi = r->hi + r->d, st = r->st, n = 0, s = 0, t_end = r->t_end;
    uint64_t vl = 64ull << lmul, per = CHUNK / vl, vtype = 0xC0u | lmul;   /* ta, ma, e8, m(2^lmul) */
    __asm__ volatile("vsetvl zero, %0, %1\n\tvmv.v.i v16, 0" :: "r"(vl), "r"(vtype) : "memory");
    for (;;) {
        NOUNROLL for (uint64_t j = 0; j < per; j++) {
            __asm__ volatile("vle8.v v8, (%0)\n\tvadd.vv v16, v16, v8" :: "r"(p) : "memory");
            p += st;
            if (p >= hi) p = lo;
        }
        n += per;
        if (NOW(r) >= t_end) break;
    }
    /* sum over lanes, widened to u16 (at most 256 lanes x 255 per vwredsumu: m8 as two m4 halves) */
    for (uint64_t half = 0; half < (lmul == 3 ? 2u : 1u); half++) {
        uint64_t x, rl = vl > 256 ? 256 : vl;
        if (half)
            __asm__ volatile("vsetivli zero, 1, e16, m1, ta, ma\n\tvmv.v.i v0, 0\n\tvsetvli zero, %1, e8, m4, ta, ma\n\t"
                             "vwredsumu.vs v0, v20, v0\n\tvsetivli zero, 1, e16, m1, ta, ma\n\tvmv.x.s %0, v0"
                             : "=r"(x) : "r"(rl) : "memory");
        else
            __asm__ volatile("vsetivli zero, 1, e16, m1, ta, ma\n\tvmv.v.i v0, 0\n\tvsetvli zero, %1, e8, m4, ta, ma\n\t"
                             "vwredsumu.vs v0, v16, v0\n\tvsetivli zero, 1, e16, m1, ta, ma\n\tvmv.x.s %0, v0"
                             : "=r"(x) : "r"(rl) : "memory");
        s += x & 0xFFFFu;
    }
    r->n = n;
    r->sink = s;
}

/* K9 / K10 / K12: wbc_mlp.c gemv_f16_rvv's single-group loop as clang emits it (vle16 at e32m4 = EMUL m2, flw x[i],
 * vwmul at e16m2, then e32m4 vand, vfmul 2^112, vfmul x[i], vfadd), one 64-column slice per row of 512 B. */
#define MLP_HEAD                                                                                   \
    uint64_t p = r->lo + r->d, lo = r->lo + r->d, hi = r->hi + r->d, n = 0;                        \
    uint64_t xp = r->x + r->d, xlo = xp, xhi = xp + 2048u, t_end = r->t_end;                       \
    uint64_t k8192 = 8192, kmask = 0x8FFFFFFFu;                                                    \
    float f2p112 = 0x1p112f, xi;                                                                   \
    __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma\n\tvmv.v.i v8, 0\n\tvmv.v.i v20, 1" :: "r"(64ul) : "memory");
#define MLP_TAIL                                                                                   \
            p += 512u;                                                                             \
            if (p >= hi) p = lo;                                                                   \
            xp += 4u;                                                                              \
            if (xp >= xhi) xp = xlo;                                                               \
        }                                                                                          \
        n += 32u;                                                                                  \
        if (NOW(r) >= t_end) break;                                                                \
    }                                                                                              \
    r->n = n;                                                                                      \
    r->sink = 0;

static void k_mlp(struct run *r)
{
    MLP_HEAD
    for (;;) {
        NOUNROLL for (uint32_t j = 0; j < 32u; j++) {
            __asm__ volatile("vle16.v v12, (%1)\n\tflw %0, 0(%2)\n\t"
                             "vsetvli zero, zero, e16, m2, ta, ma\n\tvwmul.vx v16, v12, %3\n\t"
                             "vsetvli zero, zero, e32, m4, ta, ma\n\tvand.vx v12, v16, %4\n\t"
                             "vfmul.vf v12, v12, %5\n\tvfmul.vf v12, v12, %0\n\tvfadd.vv v8, v8, v12"
                             : "=&f"(xi) : "r"(p), "r"(xp), "r"(k8192), "r"(kmask), "f"(f2p112) : "memory");
    MLP_TAIL
}

static void k_mlp_alu(struct run *r)
{
    MLP_HEAD
    xi = 1.0f;
    for (;;) {
        NOUNROLL for (uint32_t j = 0; j < 32u; j++) {
            __asm__ volatile("vsetvli zero, zero, e16, m2, ta, ma\n\tvwmul.vx v16, v20, %0\n\t"
                             "vsetvli zero, zero, e32, m4, ta, ma\n\tvand.vx v12, v16, %1\n\t"
                             "vfmul.vf v12, v12, %2\n\tvfmul.vf v12, v12, %3\n\tvfadd.vv v8, v8, v12"
                             :: "r"(k8192), "r"(kmask), "f"(f2p112), "f"(xi) : "memory");
    MLP_TAIL
}

static void k_mlp_ld(struct run *r)
{
    MLP_HEAD
    (void)k8192; (void)kmask; (void)f2p112;
    for (;;) {
        NOUNROLL for (uint32_t j = 0; j < 32u; j++) {
            __asm__ volatile("vle16.v v12, (%1)\n\tflw %0, 0(%2)" : "=&f"(xi) : "r"(p), "r"(xp) : "memory");
    MLP_TAIL
}
#else
/* No V (the x86 stand-in): the same unit stream and the same sink definition, in scalar code. */
static void k_vle8(struct run *r, uint64_t lmul)
{
    uint32_t vl = 64u << lmul;
    uint8_t acc[512] = {0};
    uint64_t p = r->lo, n = 0, s = 0;
    for (;;) {
        for (uint32_t j = 0; j < CHUNK / vl; j++) {
            const volatile uint8_t *b = (const volatile uint8_t *)(p + r->d);
            for (uint32_t i = 0; i < vl; i++) acc[i] = (uint8_t)(acc[i] + b[i]);
            p += r->st;
            if (p >= r->hi) p = r->lo;
        }
        n += CHUNK / vl;
        if (NOW(r) >= r->t_end) break;
    }
    for (uint32_t i = 0; i < vl; i++) s += acc[i];
    r->n = n;
    r->sink = s;
}
static void k_rows(struct run *r, int loads)
{
    uint64_t p = r->lo, n = 0;
    for (;;) {
        for (uint32_t j = 0; j < 32u; j++) {
            if (loads) (void)*(const volatile uint64_t *)(p + r->d);
            p += 512u;
            if (p >= r->hi) p = r->lo;
        }
        n += 32u;
        if (NOW(r) >= r->t_end) break;
    }
    r->n = n;
    r->sink = 0;
}
static void k_mlp(struct run *r) { k_rows(r, 1); }
static void k_mlp_alu(struct run *r) { k_rows(r, 0); }
static void k_mlp_ld(struct run *r) { k_rows(r, 1); }
#endif

static void run_kernel(struct run *r, uint32_t k)
{
    if (k == MHB_K_LD8X8) k_ld8x8(r);
    else if (k == MHB_K_LD8) k_ld8(r);
    else if (k == MHB_K_CHAIN) k_chain(r);
    else if (k == MHB_K_LW16) k_lw16(r);
    else if (k <= MHB_K_VLE8_M8) k_vle8(r, k - MHB_K_VLE8_M1);   /* K5-K8: m1, m2, m4, m8 */
    else if (k == MHB_K_MLP) k_mlp(r);
    else if (k == MHB_K_MLP_ALU) k_mlp_alu(r);
    else if (k == MHB_K_MLP_LD) k_mlp_ld(r);
    else k_alu(r);
}

/* The unit stream of hart h (see the header). */
static void stream(struct run *r, uint32_t k, uint32_t pat, uint32_t kbit, uint32_t h, uint64_t base, uint64_t len)
{
    uint64_t q = len / 4u, u = KUNIT[k];
    if (k == MHB_K_MLP || k == MHB_K_MLP_ALU || k == MHB_K_MLP_LD) {
        r->lo = base + 128u * (pat == MHB_P_SHARED ? 0u : h);
        r->hi = base + len;
        r->st = 512u;
        r->x = base;
        return;
    }
    switch (pat) {
    case MHB_P_PRIVATE:   r->lo = base + h * q; r->hi = r->lo + q; r->st = u; break;
    case MHB_P_SAME_BANK: r->lo = base + h * q; r->hi = r->lo + q; r->st = 4u * u; break;
    case MHB_P_OWN_BANK:  r->lo = base + h * q + h * u; r->hi = base + (h + 1u) * q; r->st = 4u * u; break;
    case MHB_P_SHARED:    r->lo = base; r->hi = base + q; r->st = u; break;
    default:              r->lo = base + ((uint64_t)h << kbit); r->hi = base + len; r->st = 4ull << kbit; break;
    }
}

static uint64_t rd_mcycle(const struct mh_wapi *w)
{
#if defined(__riscv)
    (void)w;
    uint64_t c;
    __asm__ volatile("csrr %0, mcycle" : "=r"(c));
    return c;
#else
    return w->mcycle();
#endif
}

/* The job of every hart, hart 0's share included (mh_run calls it there directly). */
static uint64_t mhb_worker(uint64_t job_addr, uint32_t h, const struct mh_wapi *w)
{
    const volatile struct mhb_job *jb = w->ptr(job_addr);
    volatile struct mhb_res *res = w->ptr(MHB_RES(h));
    uint32_t tag = jb->tag, k = h ? jb->kernel : jb->kernel0, pat = jb->pattern, kbit = jb->kbit;
    struct run r;
    r.w = w;
    r.d = (uintptr_t)w->ptr(MB_LIM_BASE) - MB_LIM_BASE;   /* 0 on the X280, the fake LIM's offset on x86 */
    r.t_end = jb->t_end;
    uint64_t t_start = jb->t_start, base = jb->ws_base, len = jb->ws_len;
    uint32_t kk = k;
    if (k == MHB_K_TRAP) kk = MHB_K_ALU;
    stream(&r, kk, pat, kbit, h, base, len);
    while (NOW(&r) < t_start) {}                           /* start barrier: CLINT mtime, no LIM traffic */
    uint64_t c0 = rd_mcycle(w), t0 = NOW(&r);
    if (k == MHB_K_TRAP && h == kbit) {
#if defined(MON_X86)
        x86_emulate_trap(2u, (uint64_t)(uintptr_t)mhb_worker, 0xc0001073u);
#else
        __asm__ volatile(".4byte 0xc0001073" ::: "memory");   /* csrrw x0, cycle, x0: illegal (payloads.c) */
#endif
    }
    run_kernel(&r, kk);
    uint64_t c1 = rd_mcycle(w), t1 = NOW(&r);
    res->hart = h;
    res->status = MHB_OK;
    res->kernel = k;
    res->bytes = r.n * KBPA[kk];
    res->accesses = r.n;
    res->cycles = c1 - c0;
    res->t0 = t0;
    res->dt = (uint32_t)(t1 - t0);
    res->beu_accrued = w->beu_accrued(h);
    res->sink = r.sink;
#if defined(__riscv)
    __asm__ volatile("fence ow, ow" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    res->tag = tag;                                        /* last: a reader that sees the tag sees the line */
    return r.n;
}

static void summarise(struct mhb_summary *sm, const struct mon_api *api, uint32_t tag, uint32_t mask)
{
    uint64_t t0_min = ~0ull, t0_max = 0, t1_min = ~0ull, t1_max = 0;
    for (uint32_t h = 0; h < MH_NHARTS; h++) {
        const volatile struct mhb_res *res = api->ptr(MHB_RES(h));
        if (!((mask >> h) & 1u) || res->tag != tag || res->status != MHB_OK) continue;
        uint64_t t0 = res->t0, t1 = t0 + res->dt;
        sm->done |= 1u << h;
        sm->bytes += res->bytes;
        if (t0 < t0_min) t0_min = t0;
        if (t0 > t0_max) t0_max = t0;
        if (t1 < t1_min) t1_min = t1;
        if (t1 > t1_max) t1_max = t1;
    }
    if (sm->done) {
        sm->span = (uint32_t)(t1_max - t0_min);
        sm->start_skew = (uint32_t)(t0_max - t0_min);
        sm->end_skew = (uint32_t)(t1_max - t1_min);
    }
}

/* The CALL's checks, in order; MHB_OK when the run may start. */
static uint32_t check(uint32_t k, uint32_t k0, uint32_t mask, uint32_t pat, uint32_t kbit, uint64_t window, uint64_t hz,
                      uint64_t base, uint64_t len)
{
    if (!(k - 1u < MHB_N_KERNELS || k == MHB_K_TRAP) || !(k0 - 1u < MHB_N_KERNELS || k0 == MHB_K_TRAP) || !mask ||
        (k == MHB_K_TRAP && kbit > 3u))
        return MHB_EKERNEL;
    if (pat > MHB_P_SWEEP || (pat == MHB_P_SWEEP && kbit - 6u > 6u)) return MHB_EPATTERN;
    if (!window || window > hz) return MHB_EWINDOW;
    if ((base & 63u) || !len || len % MHB_WS_ALIGN || base < MB_PAYLOAD_BASE || base + len > MB_LIM_END ||
        (base < MHB_END && MHB_BASE < base + len))
        return MHB_ERANGE;
    return MHB_OK;
}

MHB_ENTRY
{
    struct mhb_summary sm = {0};
    uint32_t k = a->policy & 0xFFu, k0 = (a->policy >> 8) & 0xFFu, mask = a->flags & 0xFu;
    uint32_t pat = (a->flags >> 4) & 0xFu, kbit = (a->flags >> 8) & 0xFu, mode = (a->flags >> 12) & 3u;
    uint64_t hz = api->mtime_hz, us = hz / 1000000u;
    volatile struct mhb_job *jb = api->ptr(MHB_DATA);
    sm.kernel = a->policy;
    sm.tag = a->seq;
    sm.mask = mask;
    if (api->version < 2u || api->size < sizeof(struct mon_api)) { sm.status = MHB_EAPI; goto out; }
    if (mode == MHB_MODE_COLLECT) {
        if (jb->mode != MHB_MODE_START || !jb->dispatched) { sm.status = MHB_ENOJOB; goto out; }
        uint64_t now = api->mtime(), end = jb->t_end;
        api->mh_wait((end > now ? end - now : 0) + hz / 100u);
        sm.kernel = jb->kernel;
        sm.tag = jb->tag;
        sm.mask = jb->dispatched;
        jb->mode = MHB_MODE_RUN;                           /* collected: a second COLLECT has nothing to wait for */
        summarise(&sm, api, sm.tag, sm.mask);
        goto out;
    }
    if (mode == MHB_MODE_START) mask &= MH_SEC_MASK;
    uint64_t window = a->tick > 0 ? (uint64_t)a->tick : us * MHB_WINDOW_US;
    uint64_t base = a->arg_addr ? a->arg_addr : MHB_WS_DEFAULT_BASE;
    uint64_t len = a->arg_len ? a->arg_len : MHB_WS_DEFAULT_LEN;
    sm.status = mode > MHB_MODE_COLLECT ? MHB_EMODE : check(k, k0 ? k0 : k, mask, pat, kbit, window, hz, base, len);
    if (sm.status) goto out;
    uint64_t lead = us * MHB_LEAD_US, t_start = api->mtime() + lead;
    jb->tag = a->seq;
    jb->kernel = k;
    jb->kernel0 = k0 ? k0 : k;
    jb->pattern = pat;
    jb->kbit = kbit;
    jb->mask = mask;
    jb->ws_base = base;
    jb->ws_len = len;
    jb->t_start = t_start;
    jb->t_end = t_start + window;
    jb->mode = mode;
    jb->dispatched = mask & MH_SEC_MASK & api->mh_ready();
#if defined(__riscv)
    __asm__ volatile("fence ow, ow" ::: "memory");          /* the job line before mh_run's gen store */
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    uint32_t r = api->mh_run(mask, (uint64_t)(uintptr_t)mhb_worker, MHB_DATA,
                             mode == MHB_MODE_START ? 0 : lead + window + hz / 100u);
    if (mode == MHB_MODE_START) {
        jb->dispatched = r;
        sm.done = r;                                       /* dispatched, still running */
    } else {
        summarise(&sm, api, a->seq, mask);
    }
out:;
    volatile uint32_t *o = (volatile uint32_t *)api->result;
    const uint32_t *sw = (const uint32_t *)&sm;
    for (uint32_t i = 0; i < sizeof sm / 4; i++) o[i] = sw[i];
    return sm.done;
}
