/* LIM microbench payload (PLAN.md WP-3.2 "Payloads", T-3.E2m): the numbers that replace the §1.3
 * estimates before WP-3.7 picks a kernel. LIM is not data-cacheable (L2CPUTile/Caches.md:26-31) and the
 * L1D has one outstanding fill (Caches.md:36), so every number here is "uncached latency/bandwidth".
 *
 * Loaded with LOAD (kind CODE) at PAYLOAD_ADDR, then FENCE_I, then CALL with arg_entry = PAYLOAD_ADDR:
 *   policy   = test id (LB_*)
 *   flags    = LB_FLAG_* (bit 0: allow the Zero-Device working set)
 *   arg_addr = working-set base (LIM, 64 B aligned; 0 = the first line after this payload)
 *   arg_len  = working-set bytes (0 = 64 KiB), a multiple of 64
 *   tick     = iterations (0 = one pass over the working set)
 * Result (mon_api.result = L39 bytes 24..63) = struct lb_result; return value = cycles.
 * The working set must not hold a registered blob: the pointer-chase test writes it.
 *
 * Tests
 *   1 ptr_chase   dependent 8 B loads, one per line, random cyclic order (Sattolo): load-to-use latency
 *   2 stream      sequential 8 B loads over the set: scalar uncached bandwidth
 *   3/4/5 vle32   unit-stride vle32.v at LMUL m1/m4/m8 over the set (riscv with V only)
 *   6 exec_loop   a 64-instruction straight-line ALU block run `iterations` times from LIM: cycles per
 *                 instruction tells whether instruction fetch from LIM is cached in L1I (Caches.md:13-14)
 *   7 zero_dev    ptr_chase in the cached Zero Device (0x0A00_0000, MemoryMap.md:86), opt-in via flags
 *                 bit 0; touches no cache configuration register (no WAYENABLE, no WAYMASK)
 *   8 mtime_ratio mcycle / mtime over ~1 ms of mtime, x1000 in `bytes` (expect 16000 at 800/50 MHz)
 */
#include <stdint.h>
#include "mailbox.h"

#define LB_PTR_CHASE   1u
#define LB_STREAM      2u
#define LB_VLE32_M1    3u
#define LB_VLE32_M4    4u
#define LB_VLE32_M8    5u
#define LB_EXEC_LOOP   6u
#define LB_ZERO_DEV    7u
#define LB_MTIME_RATIO 8u
#define LB_FLAG_ZERO_DEVICE 1u

#define LB_OK          0u
#define LB_ENOTEST     1u
#define LB_ERANGE      2u
#define LB_ENOVECTOR   3u
#define LB_EDENIED     4u

struct lb_result {          /* 40 B in L39 bytes 24..63 */
    uint32_t test;          /* 24 */
    uint32_t status;        /* 28  LB_* */
    uint64_t iters;         /* 32  loads, vector loads or loop iterations */
    uint64_t bytes;         /* 40  bytes touched (mtime_ratio: ratio x1000) */
    uint64_t cycles;        /* 48  mcycle */
    uint64_t ticks;         /* 56  mtime */
};
_Static_assert(sizeof(struct lb_result) == 40, "lb_result must fit L39 bytes 24..63");

#define ZERO_DEVICE_BASE 0x0A000000ull
#define ZERO_DEVICE_MAX  (32u * 1024u)   /* stays inside the 32 KiB L1D (Caches.md:26) */

#if defined(MON_X86)
#define LB_ENTRY uint64_t x86_payload_lim_bench(const struct mb_call_args *a, const struct mon_api *api)
#else
extern char __payload_end[];
#define LB_ENTRY __attribute__((section(".text.payload_entry"), used)) \
    uint64_t payload_entry(const struct mb_call_args *a, const struct mon_api *api)
#endif

static uint64_t xorshift(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

/* Build a single random cycle over the n lines at base (Sattolo), each line's first word holding the
 * physical address of the next; then chase it `iters` times. Returns cycles. */
static uint64_t ptr_chase(const struct mon_api *api, uint64_t base, uint32_t n, uint64_t iters,
                          volatile uint64_t *sink)
{
    volatile uint64_t *w0 = api->ptr(base);
    for (uint32_t i = 0; i < n; i++) w0[i * 8] = i;               /* identity index first */
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    for (uint32_t i = n - 1; i > 0; i--) {                         /* Sattolo: one cycle */
        uint32_t j = (uint32_t)(xorshift(&seed) % i);
        uint64_t t = w0[i * 8]; w0[i * 8] = w0[j * 8]; w0[j * 8] = t;
        if ((i & 1023u) == 0) api->heartbeat();
    }
    /* convert the permutation of indices into next-pointers (physical addresses) */
    for (uint32_t i = 0; i < n; i++) w0[i * 8] = base + w0[i * 8] * 64u;
    uintptr_t delta = (uintptr_t)w0 - (uintptr_t)base;   /* 0 on the X280, fake LIM offset on x86 */
    uint64_t p = base;
    uint64_t c0 = api->mcycle();
    for (uint64_t k = 0; k < iters; k++) {
        p = *(volatile uint64_t *)(uintptr_t)(p + delta);
        if ((k & 4095u) == 4095u) api->heartbeat();
    }
    uint64_t c = api->mcycle() - c0;
    *sink = p;
    return c;
}

#if defined(__riscv_vector)
#define VLOOP(LMUL)                                                                         \
    do {                                                                                    \
        uint64_t vl;                                                                        \
        __asm__ volatile("vsetvli %0, zero, e32, " LMUL ", ta, ma" : "=r"(vl));             \
        uint64_t step = vl * 4u;                                                            \
        for (uint64_t k = 0; k < iters; k++) {                                              \
            for (uint64_t off = 0; off + step <= len; off += step)                          \
                __asm__ volatile("vle32.v v8, (%0)" :: "r"(p0 + off) : "memory");           \
            api->heartbeat();                                                               \
        }                                                                                   \
        loads = iters * (len / step);                                                       \
    } while (0)
#endif

/* 64 dependent-free ALU instructions, all executed from LIM. */
#define ALU8 "addi t0, t0, 1\n addi t1, t1, 1\n addi t2, t2, 1\n addi t3, t3, 1\n" \
             "addi t4, t4, 1\n addi t5, t5, 1\n addi t6, t6, 1\n addi a7, a7, 1\n"

LB_ENTRY
{
    struct lb_result r = { a->policy, LB_OK, 0, 0, 0, 0 };
    uint64_t iters = a->tick > 0 ? (uint64_t)a->tick : 0;
    uint64_t len = a->arg_len ? a->arg_len : 64u * 1024u;
    uint64_t base = a->arg_addr;
#if defined(MON_X86)
    if (!base) base = MB_PAYLOAD_BASE + 0x10000u;
#else
    if (!base) base = ((uint64_t)(uintptr_t)__payload_end + 63u) & ~63ull;
#endif
    volatile uint64_t sink = 0;
    uint64_t t0 = api->mtime();
    switch (a->policy) {
    case LB_PTR_CHASE:
    case LB_ZERO_DEV: {
        if (a->policy == LB_ZERO_DEV) {
#if defined(MON_X86)
            r.status = LB_EDENIED; break;   /* no Zero Device behind the fake LIM */
#endif
            if (!(a->flags & LB_FLAG_ZERO_DEVICE)) { r.status = LB_EDENIED; break; }
            base = ZERO_DEVICE_BASE;
            if (len > ZERO_DEVICE_MAX) len = ZERO_DEVICE_MAX;
        } else if ((base | len) & 63u || base < MB_PAYLOAD_BASE || base + len > MB_LIM_END) {
            r.status = LB_ERANGE; break;
        }
        uint32_t n = (uint32_t)(len / 64u);
        if (n < 2) { r.status = LB_ERANGE; break; }
        if (!iters) iters = n;
        r.cycles = ptr_chase(api, base, n, iters, &sink);
        r.iters = iters;
        r.bytes = iters * 8u;
        break;
    }
    case LB_STREAM: {
        if ((base | len) & 63u || base < MB_LIM_BASE || base + len > MB_LIM_END) { r.status = LB_ERANGE; break; }
        if (!iters) iters = 1;
        const volatile uint64_t *p = api->ptr(base);
        uint64_t acc = 0, c0 = api->mcycle();
        for (uint64_t k = 0; k < iters; k++) {
            for (uint64_t i = 0; i < len / 8u; i++) acc += p[i];
            api->heartbeat();
        }
        r.cycles = api->mcycle() - c0;
        sink = acc;
        r.iters = iters * (len / 8u);
        r.bytes = iters * len;
        break;
    }
    case LB_VLE32_M1:
    case LB_VLE32_M4:
    case LB_VLE32_M8: {
#if defined(__riscv_vector)
        if ((base | len) & 63u || base < MB_LIM_BASE || base + len > MB_LIM_END) { r.status = LB_ERANGE; break; }
        if (!iters) iters = 1;
        uint64_t p0 = (uint64_t)(uintptr_t)api->ptr(base), loads = 0, c0 = api->mcycle();
        if (a->policy == LB_VLE32_M1) VLOOP("m1");
        else if (a->policy == LB_VLE32_M4) VLOOP("m4");
        else VLOOP("m8");
        r.cycles = api->mcycle() - c0;
        r.iters = loads;
        r.bytes = iters * len;
#else
        r.status = LB_ENOVECTOR;
#endif
        break;
    }
    case LB_EXEC_LOOP: {
        if (!iters) iters = 1000;
        uint64_t c0 = api->mcycle();
#if defined(__riscv)
        for (uint64_t k = 0; k < iters; k++)
            __asm__ volatile(ALU8 ALU8 ALU8 ALU8 ALU8 ALU8 ALU8 ALU8
                             ::: "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7");
#else
        for (uint64_t k = 0; k < iters; k++) __asm__ volatile("" ::: "memory");
#endif
        r.cycles = api->mcycle() - c0;
        r.iters = iters;
        r.bytes = iters * 64u * 4u;   /* instruction bytes fetched */
        break;
    }
    case LB_MTIME_RATIO: {
        uint64_t m0 = api->mtime(), c0 = api->mcycle(), want = api->mtime_hz / 1000u, m1;
        do { m1 = api->mtime(); } while (m1 - m0 < want);
        uint64_t c1 = api->mcycle();
        r.cycles = c1 - c0;
        r.iters = m1 - m0;
        r.bytes = (m1 > m0) ? (c1 - c0) * 1000u / (m1 - m0) : 0;
        break;
    }
    default:
        r.status = LB_ENOTEST;
        break;
    }
    r.ticks = api->mtime() - t0;
    volatile uint32_t *out = (volatile uint32_t *)api->result;
    const uint32_t *rw = (const uint32_t *)&r;
    for (uint32_t i = 0; i < sizeof r / 4; i++) out[i] = rw[i];
    (void)sink;
    return r.cycles;
}
