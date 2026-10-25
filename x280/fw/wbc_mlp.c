/* gear_wbc MLP forward pass, freestanding (PLAN.md WP-3.1; contract and graph in wbc_mlp.h).
 *
 * Variants (all compute y[o] = b[o] + sum_i w[o][i] x[i] in ascending i, separately rounded):
 *   v0  scalar fp32, 8 outputs per pass so x[i] is loaded once per 8 weights (bit-parity reference)
 *   v1  RVV fp32 column-broadcast GEMV: e32/m4, four register groups (4*VLMAX outputs, 256 at
 *       VLEN 512) accumulate in registers while each input streams one contiguous W^T row
 *   v2  fp16 weights: RVV on riscv (vle16 + integer widening multiply by 2^13 and mask + one exact
 *       fp32 scale by 2^112, no Zvfh); on x86 the scalar v2s code
 *   v2s scalar fp16 (the same widening per weight); built everywhere
 *   v3  fp16, bit-identical to v2 (see "v3" below): v2's exact 2^112 weight scale moves onto x[i], so
 *       a weight costs 4 vector ops instead of 5. riscv: e32m8, two register groups (256 outputs) with
 *       the next row's weights and x[i+1] loaded a row ahead; the 35/15-column tails run four rows a
 *       block with the next block loading. x86: the same arithmetic in scalar code
 * -DWBC_RVV_FMA makes v1/v2 use vfmacc (one rounding per MAC): faster, but then no longer
 * bit-identical to v0/v2s (E4 experiment only; v3 never fuses).
 *
 * Memory traffic per call (weights dominate; LIM is uncached, Caches.md:26-31): fp32 1,875,968 B
 * (29,312 lines), fp16 937,984 B (14,656 lines), plus 6,344 B of biases and 3,840 B of scratch.
 */
#include "wbc_mlp.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

/* Scratch layout, in floats: two ping-pong activation buffers, the actor input and the estimate. */
#define S_A    0      /* 512 */
#define S_B    512    /* 256 */
#define S_CAT  768    /* 121 (128) */
#define S_EST  896    /* 35 (64) */
_Static_assert((S_EST + 64) * 4 <= WBC_SCRATCH_BYTES, "scratch");

#define NORM_EPS 9.99999996e-13f  /* the graph's Clip min, (float)1e-12 (F.normalize eps) */
#define N_EST    35
#define N_VEL    3                /* est[0:3] go to the actor as-is */
#define N_FRAME  86               /* newest frame: obs[430:516] */

static const uint32_t k_in[WBC_N_LAYERS]  = { 516, 256, 256, 121, 512, 256, 256 };
static const uint32_t k_out[WBC_N_LAYERS] = { 256, 256,  35, 512, 256, 256,  15 };

static inline float as_f32(uint32_t u) { union { uint32_t u; float f; } c = { .u = u }; return c.f; }

/* ---- libc replacements ----------------------------------------------------------------------- */

void *wbc_memcpy(void *dst, const void *src, uint32_t n)
{
    uint8_t *d = dst;
    const uint8_t *s = src;
    if ((((uintptr_t)d | (uintptr_t)s | n) & 3u) == 0) {
        uint32_t *dw = dst;
        const uint32_t *sw = src;
        for (uint32_t i = 0; i < n / 4; i++) dw[i] = sw[i];
        return dst;
    }
    for (uint32_t i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

void *wbc_memset(void *dst, int c, uint32_t n)
{
    uint8_t *d = dst;
    for (uint32_t i = 0; i < n; i++) d[i] = (uint8_t)c;
    return dst;
}

/* For calls the compiler may emit on its own; weak, so the firmware can provide strong ones. */
#if defined(__x86_64__)
#define WBC_LIBC_ATTR __attribute__((weak, visibility("hidden")))
#else
#define WBC_LIBC_ATTR __attribute__((weak))
#endif
WBC_LIBC_ATTR void *memcpy(void *dst, const void *src, __SIZE_TYPE__ n)
{
    uint8_t *d = dst;
    const uint8_t *s = src;
    for (__SIZE_TYPE__ i = 0; i < n; i++) d[i] = s[i];
    return dst;
}
WBC_LIBC_ATTR void *memset(void *dst, int c, __SIZE_TYPE__ n)
{
    uint8_t *d = dst;
    for (__SIZE_TYPE__ i = 0; i < n; i++) d[i] = (uint8_t)c;
    return dst;
}

/* ---- CRC-32 ---------------------------------------------------------------------------------- */

static inline uint32_t crc_step32(uint32_t c)
{
#pragma clang loop unroll(full)
    for (int k = 0; k < 32; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    return c;
}

uint32_t wbc_crc32(uint32_t crc, const void *p, uint32_t n)
{
    const uint8_t *b = p;
    uint32_t c = ~crc;
    while (n && ((uintptr_t)b & 3u)) {            /* bytes up to word alignment */
        c ^= *b++;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        n--;
    }
    for (; n >= 4; n -= 4, b += 4)                /* one 4 B load per word (LIM loads are uncached) */
        c = crc_step32(c ^ *(const uint32_t *)b); /* little-endian: same as 4 byte steps */
    while (n--) {
        c ^= *b++;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

/* ---- GEMV kernels: y[0:O] = b + W x, W^T stored [I][O] ----------------------------------------- */

typedef void (*gemv_fn)(const void *wt, const float *b, const float *x, float *y, uint32_t I, uint32_t O);
typedef void (*elu_fn)(float *v, uint32_t n);

/* fp16 -> fp32: sign-extend, shift the 15 exponent+mantissa bits to fp32's position, keep the sign,
 * clear the three copied sign bits (28..30), then scale by 2^(127-15). The scale is exact and also
 * turns fp16 subnormals (about 430 weights per policy) into their exact fp32 value. */
static inline float h2f(uint16_t h)
{
    uint32_t u = (uint32_t)(int32_t)(int16_t)h;
    return as_f32((u << 13) & 0x8FFFFFFFu) * 0x1p112f;
}

float wbc_f16_to_f32(uint16_t h) { return h2f(h); }

#define DOT8_BODY(LOADW)                                                                  \
    uint32_t o = 0;                                                                       \
    for (; o + 8 <= O; o += 8) {                                                          \
        float a0 = b[o], a1 = b[o + 1], a2 = b[o + 2], a3 = b[o + 3];                     \
        float a4 = b[o + 4], a5 = b[o + 5], a6 = b[o + 6], a7 = b[o + 7];                 \
        const W_T *w = wt + o;                                                            \
        for (uint32_t i = 0; i < I; i++, w += O) {                                        \
            float xi = x[i];                                                              \
            a0 = a0 + LOADW(w[0]) * xi; a1 = a1 + LOADW(w[1]) * xi;                       \
            a2 = a2 + LOADW(w[2]) * xi; a3 = a3 + LOADW(w[3]) * xi;                       \
            a4 = a4 + LOADW(w[4]) * xi; a5 = a5 + LOADW(w[5]) * xi;                       \
            a6 = a6 + LOADW(w[6]) * xi; a7 = a7 + LOADW(w[7]) * xi;                       \
        }                                                                                 \
        y[o] = a0; y[o + 1] = a1; y[o + 2] = a2; y[o + 3] = a3;                           \
        y[o + 4] = a4; y[o + 5] = a5; y[o + 6] = a6; y[o + 7] = a7;                       \
    }                                                                                     \
    for (; o < O; o++) {                                                                  \
        float a = b[o];                                                                   \
        const W_T *w = wt + o;                                                            \
        for (uint32_t i = 0; i < I; i++, w += O) a = a + LOADW(w[0]) * x[i];              \
        y[o] = a;                                                                         \
    }

#define LOAD_F32(v) (v)
#define LOAD_F16(v) h2f(v)

static void gemv_f32_scalar(const void *wt_, const float *b, const float *x, float *y, uint32_t I, uint32_t O)
{
#define W_T float
    const W_T *wt = wt_;
    DOT8_BODY(LOAD_F32)
#undef W_T
}

static void gemv_f16_scalar(const void *wt_, const float *b, const float *x, float *y, uint32_t I, uint32_t O)
{
#define W_T uint16_t
    const W_T *wt = wt_;
    DOT8_BODY(LOAD_F16)
#undef W_T
}

/* ---- v3: v2's arithmetic with the weight scale on x[i] -----------------------------------------------
 *
 * v2 forms each product as p = (w' * 2^112) * x[i], w' = h2f's bit pattern before the scale (= w * 2^-112,
 * possibly an fp32 subnormal). The first multiply is exact (w' * 2^112 = w, an fp16 value), so p = RN(w x[i]).
 * v3 forms X = x[i] * 2^112 once per row and p = w' * X. For |x[i]| < 2^16 the scaling is exact as well
 * (|X| <= FLT_MAX, also for subnormal x[i]), so w' * X = w x[i] as a real number and RN gives the same p,
 * including signed zeros. For |x[i]| >= 2^16, inf or NaN, X would overflow or lose that exactness, so such a
 * row gets v2's two multiplies instead (x86 below: per row; riscv: the layer is computed again by v2's kernel).
 * The adds, their order and the ELU are v2's, so every output equals v2's bits.
 * The vector cost per weight group: vwmul + vand + vfmul + vfadd, v2's vfmul by 2^112 is gone (4 ops, not 5). */
#define V3_SCALE 0x1p112f   /* h2f's scale */
#define V3_XLIM  0x1p16f    /* |x| < 2^16 <=> x * 2^112 is finite, and then exact */

#if !defined(__riscv_vector)
/* h2f without the scale: the fp32 pattern of w * 2^-112 */
static inline float h2f_unscaled(uint16_t h)
{
    uint32_t u = (uint32_t)(int32_t)(int16_t)h;
    return as_f32((u << 13) & 0x8FFFFFFFu);
}

#define FOLD8(LOADW, XV)                                                                   \
    a0 = a0 + LOADW(w[0]) * (XV); a1 = a1 + LOADW(w[1]) * (XV);                            \
    a2 = a2 + LOADW(w[2]) * (XV); a3 = a3 + LOADW(w[3]) * (XV);                            \
    a4 = a4 + LOADW(w[4]) * (XV); a5 = a5 + LOADW(w[5]) * (XV);                            \
    a6 = a6 + LOADW(w[6]) * (XV); a7 = a7 + LOADW(w[7]) * (XV);

/* v3 on x86: gemv_f16_scalar's loops with v3's product, or v2's for a row with |x[i]| >= 2^16, inf or NaN */
static void gemv_f16_fold_scalar(const void *wt_, const float *b, const float *x, float *y, uint32_t I, uint32_t O)
{
    const uint16_t *wt = wt_;
    uint32_t o = 0;
    for (; o + 8 <= O; o += 8) {
        float a0 = b[o], a1 = b[o + 1], a2 = b[o + 2], a3 = b[o + 3];
        float a4 = b[o + 4], a5 = b[o + 5], a6 = b[o + 6], a7 = b[o + 7];
        const uint16_t *w = wt + o;
        for (uint32_t i = 0; i < I; i++, w += O) {
            const float xi = x[i];
            if (__builtin_fabsf(xi) < V3_XLIM) {
                const float X = xi * V3_SCALE;
                FOLD8(h2f_unscaled, X)
            } else {
                FOLD8(h2f, xi)
            }
        }
        y[o] = a0; y[o + 1] = a1; y[o + 2] = a2; y[o + 3] = a3;
        y[o + 4] = a4; y[o + 5] = a5; y[o + 6] = a6; y[o + 7] = a7;
    }
    for (; o < O; o++) {
        float a = b[o];
        const uint16_t *w = wt + o;
        for (uint32_t i = 0; i < I; i++, w += O) {
            const float xi = x[i];
            if (__builtin_fabsf(xi) < V3_XLIM) a = a + h2f_unscaled(w[0]) * (xi * V3_SCALE);
            else a = a + h2f(w[0]) * xi;
        }
        y[o] = a;
    }
}
#endif

#if defined(__riscv_vector)

#if defined(WBC_RVV_FMA)
#define VMAC(acc, w, xi, vl) __riscv_vfmacc_vf_f32m4((acc), (xi), (w), (vl))
#else
#define VMAC(acc, w, xi, vl) __riscv_vfadd_vv_f32m4((acc), __riscv_vfmul_vf_f32m4((w), (xi), (vl)), (vl))
#endif

static inline vfloat32m4_t vload_f32(const float *p, size_t vl) { return __riscv_vle32_v_f32m4(p, vl); }

/* h2f on vl lanes. The widening signed multiply by 2^13 is sign-extend + shift in one op (the
 * product fits in 29 bits); the SiFive7 model has one vector arithmetic pipe, so this saves 1 of 6
 * arithmetic ops per register group (llvm-mca -mcpu=sifive-x280: 216 -> 185 cycles per input row). */
static inline vfloat32m4_t vload_f16(const uint16_t *p, size_t vl)
{
    vint32m4_t s = __riscv_vwmul_vx_i32m4(__riscv_vle16_v_i16m2((const int16_t *)p, vl), (int16_t)8192, vl);
    vuint32m4_t u = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(s), 0x8FFFFFFFu, vl);
    return __riscv_vfmul_vf_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(u), 0x1p112f, vl);
}

/* Four full register groups per pass, then single-group passes for the tail (O = 35, 15). */
#define COLBCAST_BODY(VLOAD)                                                              \
    const size_t vm = __riscv_vsetvlmax_e32m4();                                          \
    uint32_t o = 0;                                                                       \
    for (; o + 4 * vm <= O; o += 4 * vm) {                                                \
        vfloat32m4_t a0 = __riscv_vle32_v_f32m4(b + o, vm);                               \
        vfloat32m4_t a1 = __riscv_vle32_v_f32m4(b + o + vm, vm);                          \
        vfloat32m4_t a2 = __riscv_vle32_v_f32m4(b + o + 2 * vm, vm);                      \
        vfloat32m4_t a3 = __riscv_vle32_v_f32m4(b + o + 3 * vm, vm);                      \
        const W_T *w = wt + o;                                                            \
        for (uint32_t i = 0; i < I; i++, w += O) {                                        \
            float xi = x[i];                                                              \
            a0 = VMAC(a0, VLOAD(w, vm), xi, vm);                                          \
            a1 = VMAC(a1, VLOAD(w + vm, vm), xi, vm);                                     \
            a2 = VMAC(a2, VLOAD(w + 2 * vm, vm), xi, vm);                                 \
            a3 = VMAC(a3, VLOAD(w + 3 * vm, vm), xi, vm);                                 \
        }                                                                                 \
        __riscv_vse32_v_f32m4(y + o, a0, vm);                                             \
        __riscv_vse32_v_f32m4(y + o + vm, a1, vm);                                        \
        __riscv_vse32_v_f32m4(y + o + 2 * vm, a2, vm);                                    \
        __riscv_vse32_v_f32m4(y + o + 3 * vm, a3, vm);                                    \
    }                                                                                     \
    while (o < O) {                                                                       \
        size_t vl = __riscv_vsetvl_e32m4(O - o);                                          \
        vfloat32m4_t a = __riscv_vle32_v_f32m4(b + o, vl);                                \
        const W_T *w = wt + o;                                                            \
        for (uint32_t i = 0; i < I; i++, w += O) a = VMAC(a, VLOAD(w, vl), x[i], vl);     \
        __riscv_vse32_v_f32m4(y + o, a, vl);                                              \
        o += (uint32_t)vl;                                                                \
    }

static void gemv_f32_rvv(const void *wt_, const float *b, const float *x, float *y, uint32_t I, uint32_t O)
{
#define W_T float
    const W_T *wt = wt_;
    COLBCAST_BODY(vload_f32)
#undef W_T
}

static void gemv_f16_rvv(const void *wt_, const float *b, const float *x, float *y, uint32_t I, uint32_t O)
{
#define W_T uint16_t
    const W_T *wt = wt_;
    COLBCAST_BODY(vload_f16)
#undef W_T
}

/* v3 on riscv (the arithmetic is gemv_f16_fold_scalar's; the derivation is above it).
 *
 * Why this schedule. One hart is bound by its single vector arithmetic pipe, not by LIM: S3b's K9 (v2's
 * per-group loop, loads included) and K10 (the same ops, no loads) both take 39 cycles per 128 B of weights. So
 * v3 drops one of v2's five arithmetic ops per weight group, then keeps that pipe fed:
 *   - main passes, e32m8: two register groups of 128 outputs (acc v8, v16), a row is 2 x 256 B of weights;
 *     each group's next-row weights load into the registers (v0, v4) its vwmul has just read, so a load has
 *     the following seven arithmetic ops (~112 cycles at 16 per m8 op) to arrive in; x[i + 1] loads during row
 *     i (a dependent uncached load is ~28 cycles) and X = x * 2^112 is one fmul.s;
 *   - the 35/15-column tails (vl < VLMAX) do too little per row to hide a load or a latency: four rows per
 *     block, run interleaved, each row's registers refilled with the next block's row right after its vwmul
 *     and the next block's x loading meanwhile, at the smallest LMUL that holds the columns (m4: 35, m1: 15).
 * Inline asm because clang reschedules the intrinsics version (it sinks the look-ahead loads to their uses,
 * adds vmv copies and spills m8 groups). The loops are branch-free: each row ANDs |x[i]| < 2^16 into `fast`
 * and when a row fails (|x| >= 2^16, inf, NaN) the whole layer is computed again by v2's gemv_f16_rvv, whose
 * bits v3 equals by construction. Real obs never get there; it costs one v2 layer on top. */

/* One main pass: 2 x 128 outputs at b/y, rows from w (row stride rowb bytes), n >= 1 rows of x. */
static uint64_t v3_pass_m8(const uint16_t *w, const float *b, const float *x, float *y, uint64_t n, uint64_t rowb)
{
    uint64_t fast = 1, c, t;
    float xc, X, ft;
    const uint64_t vm = __riscv_vsetvlmax_e32m8();
    __asm__ volatile(
        "vsetvli    zero, %[vm], e32, m8, ta, ma\n\t"
        "vle32.v    v8, (%[b])\n\t"
        "vle32.v    v16, (%[b1])\n\t"
        "vsetvli    zero, zero, e16, m4, ta, ma\n\t"
        "vle16.v    v0, (%[w])\n\t"
        "add        %[t], %[w], %[g1]\n\t"
        "vle16.v    v4, (%[t])\n\t"
        "flw        %[xc], 0(%[x])\n\t"
        "1:\n\t"                                  /* row i: x[i] in xc, its weights in v0 / v4, vtype e16m4 */
        "# LLVM-MCA-BEGIN v3_main_row\n\t"         /* make -f Makefile.kernel kernel-mca */
        "fmul.s     %[X], %[xc], %[sc]\n\t"
        "fabs.s     %[ft], %[xc]\n\t"
        "flt.s      %[c], %[ft], %[lim]\n\t"
        "and        %[fast], %[fast], %[c]\n\t"
        "addi       %[n], %[n], -1\n\t"
        "beqz       %[n], 2f\n\t"
        "add        %[w], %[w], %[rowb]\n\t"
        "addi       %[x], %[x], 4\n\t"
        "flw        %[xc], 0(%[x])\n\t"           /* x[i + 1] */
        "vwmul.vx   v24, v0, %[k13]\n\t"          /* group 0: sign-extend and shift (w * 2^-112 bits) */
        "vle16.v    v0, (%[w])\n\t"               /* row i + 1, group 0 */
        "vsetvli    zero, zero, e32, m8, ta, ma\n\t"
        "vand.vx    v24, v24, %[msk]\n\t"
        "vfmul.vf   v24, v24, %[X]\n\t"
        "vfadd.vv   v8, v8, v24\n\t"
        "vsetvli    zero, zero, e16, m4, ta, ma\n\t"
        "vwmul.vx   v24, v4, %[k13]\n\t"          /* group 1 */
        "add        %[t], %[w], %[g1]\n\t"
        "vle16.v    v4, (%[t])\n\t"               /* row i + 1, group 1 */
        "vsetvli    zero, zero, e32, m8, ta, ma\n\t"
        "vand.vx    v24, v24, %[msk]\n\t"
        "vfmul.vf   v24, v24, %[X]\n\t"
        "vfadd.vv   v16, v16, v24\n\t"
        "vsetvli    zero, zero, e16, m4, ta, ma\n\t"
        "j          1b\n\t"
        "# LLVM-MCA-END\n\t"
        "2:\n\t"                                  /* the last row: nothing left to load */
        "vwmul.vx   v24, v0, %[k13]\n\t"
        "vsetvli    zero, zero, e32, m8, ta, ma\n\t"
        "vand.vx    v24, v24, %[msk]\n\t"
        "vfmul.vf   v24, v24, %[X]\n\t"
        "vfadd.vv   v8, v8, v24\n\t"
        "vsetvli    zero, zero, e16, m4, ta, ma\n\t"
        "vwmul.vx   v24, v4, %[k13]\n\t"
        "vsetvli    zero, zero, e32, m8, ta, ma\n\t"
        "vand.vx    v24, v24, %[msk]\n\t"
        "vfmul.vf   v24, v24, %[X]\n\t"
        "vfadd.vv   v16, v16, v24\n\t"
        "vse32.v    v8, (%[y])\n\t"
        "vse32.v    v16, (%[y1])\n\t"
        : [w] "+&r"(w), [x] "+&r"(x), [n] "+&r"(n), [fast] "+&r"(fast), [c] "=&r"(c), [t] "=&r"(t),
          [xc] "=&f"(xc), [X] "=&f"(X), [ft] "=&f"(ft)
        : [vm] "r"(vm), [b] "r"(b), [b1] "r"(b + vm), [y] "r"(y), [y1] "r"(y + vm), [rowb] "r"(rowb),
          [g1] "r"(vm * 2), [k13] "r"(8192ul), [msk] "r"(0x8FFFFFFFul), [sc] "f"(V3_SCALE), [lim] "f"(V3_XLIM)
        : "memory", "vl", "vtype", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
    return fast;
}

/* Tail row k of a block: X_k into ft<k>, the |x| check, then fa<k> refilled with the next block's x. */
#define V3_TX_ASM(K, OFF)                                                                          \
        "fmul.s     ft" #K ", fa" #K ", %[sc]\n\t"                                                 \
        "fabs.s     ft4, fa" #K "\n\t"                                                             \
        "flt.s      %[c], ft4, %[lim]\n\t"                                                         \
        "and        %[fast], %[fast], %[c]\n\t"                                                    \
        "flw        fa" #K ", " #OFF "(%[x])\n\t"

/* One tail group: vl columns at b/y, rows from w (stride rowb bytes), nb >= 1 blocks of 4 rows of x. The four
 * rows of a block run interleaved (temps v0 v4 v12 v16), so no row waits on its own vwmul-vand-vfmul latency
 * chain at these short vl; the adds into v8 stay in row order. Each row's weights (v24..v30) are refilled with the
 * next block's row right after its vwmul. */
#define V3_TAIL_FN(NAME, L32, L16)                                                                 \
static uint64_t NAME(const uint16_t *w, const float *b, const float *x, float *y, uint64_t nb, uint64_t rowb, \
                     uint64_t vl)                                                                  \
{                                                                                                  \
    uint64_t fast = 1, c, wp;                                                                      \
    __asm__ volatile(                                                                              \
        "vsetvli    zero, %[vl], e32, " L32 ", ta, ma\n\t"                                         \
        "vle32.v    v8, (%[b])\n\t"                                                                \
        "vsetvli    zero, zero, e16, " L16 ", ta, ma\n\t"                                          \
        "mv         %[wp], %[w]\n\t"                                                               \
        "vle16.v    v24, (%[wp])\n\t"                                                              \
        "add        %[wp], %[wp], %[rowb]\n\t"                                                     \
        "vle16.v    v26, (%[wp])\n\t"                                                              \
        "add        %[wp], %[wp], %[rowb]\n\t"                                                     \
        "vle16.v    v28, (%[wp])\n\t"                                                              \
        "add        %[wp], %[wp], %[rowb]\n\t"                                                     \
        "vle16.v    v30, (%[wp])\n\t"                                                              \
        "flw        fa0, 0(%[x])\n\t"                                                              \
        "flw        fa1, 4(%[x])\n\t"                                                              \
        "flw        fa2, 8(%[x])\n\t"                                                              \
        "flw        fa3, 12(%[x])\n\t"                                                             \
        "1:\n\t"                         /* next block = this one + 4 rows, or this one again if last */ \
        "# LLVM-MCA-BEGIN v3_tail_" L32 "\n\t"                                                    \
        "addi       %[nb], %[nb], -1\n\t"                                                          \
        "snez       %[c], %[nb]\n\t"                                                               \
        "neg        %[c], %[c]\n\t"                                                                \
        "and        %[wp], %[blkb], %[c]\n\t"                                                      \
        "add        %[w], %[w], %[wp]\n\t"                                                         \
        "andi       %[c], %[c], 16\n\t"                                                            \
        "add        %[x], %[x], %[c]\n\t"                                                          \
        V3_TX_ASM(0, 0) V3_TX_ASM(1, 4) V3_TX_ASM(2, 8) V3_TX_ASM(3, 12)                           \
        "mv         %[wp], %[w]\n\t"                                                               \
        "vsetvli    zero, zero, e16, " L16 ", ta, ma\n\t"                                          \
        "vwmul.vx   v0, v24, %[k13]\n\t"                                                           \
        "vle16.v    v24, (%[wp])\n\t"                                                              \
        "add        %[wp], %[wp], %[rowb]\n\t"                                                     \
        "vwmul.vx   v4, v26, %[k13]\n\t"                                                           \
        "vle16.v    v26, (%[wp])\n\t"                                                              \
        "add        %[wp], %[wp], %[rowb]\n\t"                                                     \
        "vwmul.vx   v12, v28, %[k13]\n\t"                                                          \
        "vle16.v    v28, (%[wp])\n\t"                                                              \
        "add        %[wp], %[wp], %[rowb]\n\t"                                                     \
        "vwmul.vx   v16, v30, %[k13]\n\t"                                                          \
        "vle16.v    v30, (%[wp])\n\t"                                                              \
        "vsetvli    zero, zero, e32, " L32 ", ta, ma\n\t"                                          \
        "vand.vx    v0, v0, %[msk]\n\t"                                                            \
        "vand.vx    v4, v4, %[msk]\n\t"                                                            \
        "vand.vx    v12, v12, %[msk]\n\t"                                                          \
        "vand.vx    v16, v16, %[msk]\n\t"                                                          \
        "vfmul.vf   v0, v0, ft0\n\t"                                                               \
        "vfmul.vf   v4, v4, ft1\n\t"                                                               \
        "vfmul.vf   v12, v12, ft2\n\t"                                                             \
        "vfmul.vf   v16, v16, ft3\n\t"                                                             \
        "vfadd.vv   v8, v8, v0\n\t"                                                                \
        "vfadd.vv   v8, v8, v4\n\t"                                                                \
        "vfadd.vv   v8, v8, v12\n\t"                                                               \
        "vfadd.vv   v8, v8, v16\n\t"                                                               \
        "bnez       %[nb], 1b\n\t"                                                                 \
        "# LLVM-MCA-END\n\t"                                                                       \
        "vse32.v    v8, (%[y])\n\t"                                                                \
        : [w] "+&r"(w), [x] "+&r"(x), [nb] "+&r"(nb), [fast] "+&r"(fast), [c] "=&r"(c), [wp] "=&r"(wp) \
        : [vl] "r"(vl), [b] "r"(b), [y] "r"(y), [rowb] "r"(rowb), [blkb] "r"(4 * rowb),             \
          [k13] "r"(8192ul), [msk] "r"(0x8FFFFFFFul), [sc] "f"(V3_SCALE), [lim] "f"(V3_XLIM)         \
        : "memory", "vl", "vtype", "ft0", "ft1", "ft2", "ft3", "ft4", "fa0", "fa1", "fa2", "fa3",     \
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", \
          "v15", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"); \
    return fast;                                                                                   \
}

V3_TAIL_FN(v3_tail_m4, "m4", "m2")    /* 17..64 columns (L2: 35) */
V3_TAIL_FN(v3_tail_m1, "m1", "mf2")   /* 1..16 columns (L6: 15) */

static void gemv_f16_v3(const void *wt_, const float *b, const float *x, float *y, uint32_t I, uint32_t O)
{
    const uint16_t *wt = wt_;
    const uint32_t vm = (uint32_t)__riscv_vsetvlmax_e32m8();   /* 128 at VLEN 512 */
    const uint64_t rowb = 2ull * O;
    if (I == 0 || ((O & (2 * vm - 1)) && I % 4)) {   /* not a shape of this MLP: v2 (same bits) */
        gemv_f16_rvv(wt_, b, x, y, I, O);
        return;
    }
    uint64_t fast = 1;
    uint32_t o = 0;
    for (; o + 2 * vm <= O; o += 2 * vm) fast &= v3_pass_m8(wt + o, b + o, x, y + o, I, rowb);
    while (o < O) {
        const uint32_t left = O - o;
        uint32_t vl;
        if (left > __riscv_vsetvlmax_e32m1()) {
            vl = (uint32_t)__riscv_vsetvl_e32m4(left);
            fast &= v3_tail_m4(wt + o, b + o, x, y + o, I / 4, rowb, vl);
        } else {
            vl = (uint32_t)__riscv_vsetvl_e32m1(left);
            fast &= v3_tail_m1(wt + o, b + o, x, y + o, I / 4, rowb, vl);
        }
        o += vl;
    }
    if (!fast) gemv_f16_rvv(wt_, b, x, y, I, O);   /* some |x[i]| >= 2^16, inf or NaN: v2's ops on every row */
}
#endif /* __riscv_vector */

/* ---- forward pass ---------------------------------------------------------------------------- */

__attribute__((noinline))
static void forward(const void *const w[WBC_N_LAYERS], const float *const b[WBC_N_LAYERS],
                    gemv_fn gemv, elu_fn elu, const float *obs, float *act, float *s)
{
    float *A = s + S_A, *B = s + S_B, *cat = s + S_CAT, *est = s + S_EST;

    gemv(w[0], b[0], obs, A, 516, 256);  elu(A, 256);
    gemv(w[1], b[1], A, B, 256, 256);    elu(B, 256);
    gemv(w[2], b[2], B, est, 256, N_EST);

    wbc_memcpy(cat, obs + (WBC_OBS_DIM - N_FRAME), N_FRAME * 4);
    for (uint32_t j = 0; j < N_VEL; j++) cat[N_FRAME + j] = est[j];
    float ss = 0.0f;
    for (uint32_t j = N_VEL; j < N_EST; j++) ss = ss + est[j] * est[j];
    float nrm = __builtin_sqrtf(ss);      /* -fno-math-errno: sqrtss / fsqrt.s, correctly rounded */
    if (nrm < NORM_EPS) nrm = NORM_EPS;   /* Clip(min); NaN passes */
    for (uint32_t j = N_VEL; j < N_EST; j++) cat[N_FRAME + j] = est[j] / nrm;

    gemv(w[3], b[3], cat, A, 121, 512);  elu(A, 512);
    gemv(w[4], b[4], A, B, 512, 256);    elu(B, 256);
    gemv(w[5], b[5], B, A, 256, 256);    elu(A, 256);
    gemv(w[6], b[6], A, act, 256, WBC_ACT_DIM);
}

uint32_t wbc_variants_built(void)
{
    uint32_t m = (1u << WBC_VARIANT_V0) | (1u << WBC_VARIANT_V2) | (1u << WBC_VARIANT_V2S) | (1u << WBC_VARIANT_V3);
#if defined(__riscv_vector)
    m |= 1u << WBC_VARIANT_V1;
#endif
    return m;
}

int wbc_infer(const void *blob, uint32_t policy, uint32_t variant,
              const float *obs, float *act, void *scratch)
{
    if (!blob || !obs || !act || !scratch) return WBC_EINVAL;
    if (((uintptr_t)blob | (uintptr_t)obs | (uintptr_t)act | (uintptr_t)scratch) & 3u) return WBC_EINVAL;
    const wbc_blob_hdr_t *h = blob;
    if (h->magic != WBC_BLOB_MAGIC) return WBC_EMAGIC;
    if (h->version != WBC_BLOB_VERSION) return WBC_EVERSION;
    if (policy >= WBC_N_POLICIES || !((h->policy_mask >> policy) & 1u)) return WBC_EPOLICY;
    if (variant >= WBC_N_VARIANTS || !((wbc_variants_built() >> variant) & 1u)) return WBC_EVARIANT;
    uint32_t want = (variant == WBC_VARIANT_V0 || variant == WBC_VARIANT_V1) ? WBC_WDTYPE_F32 : WBC_WDTYPE_F16;
    if (h->wdtype != want) return WBC_EDTYPE;

    /* Cheap bounds check of this policy's arrays (wbc_blob_check, once per LOAD, does the rest). */
    const uint8_t *base = blob;
    const uint32_t total = h->total_len, esz = want == WBC_WDTYPE_F32 ? 4u : 2u;
    const void *w[WBC_N_LAYERS];
    const float *b[WBC_N_LAYERS];
    for (uint32_t l = 0; l < WBC_N_LAYERS; l++) {
        const wbc_layer_t *L = &h->layer[policy][l];
        uint32_t wo = L->w_off, bo = L->b_off;
        uint32_t wbytes = k_in[l] * k_out[l] * esz, bbytes = k_out[l] * 4u;
        if (((wo | bo) & 3u) || wo > total || total - wo < wbytes || bo > total || total - bo < bbytes)
            return WBC_ESIZE;
        w[l] = base + wo;
        b[l] = (const float *)(base + bo);
    }

    switch (variant) {
    case WBC_VARIANT_V0:
        forward(w, b, gemv_f32_scalar, wbc_elu, obs, act, scratch);
        break;
#if defined(__riscv_vector)
    case WBC_VARIANT_V1:
        forward(w, b, gemv_f32_rvv, wbc_elu_rvv, obs, act, scratch);
        break;
    case WBC_VARIANT_V2:
        forward(w, b, gemv_f16_rvv, wbc_elu_rvv, obs, act, scratch);
        break;
    case WBC_VARIANT_V3:
        forward(w, b, gemv_f16_v3, wbc_elu_rvv, obs, act, scratch);
        break;
#else
    case WBC_VARIANT_V3:
        forward(w, b, gemv_f16_fold_scalar, wbc_elu, obs, act, scratch);
        break;
    case WBC_VARIANT_V2:
#endif
    case WBC_VARIANT_V2S:
        forward(w, b, gemv_f16_scalar, wbc_elu, obs, act, scratch);
        break;
    default:
        return WBC_EVARIANT;
    }
    return WBC_OK;
}

/* ---- blob validation (once per LOAD) --------------------------------------------------------- */

int wbc_blob_check(const void *blob, uint32_t len)
{
    if (!blob || ((uintptr_t)blob & 3u)) return WBC_EINVAL;
    if (len < WBC_BLOB_HDR_BYTES) return WBC_ESIZE;
    const wbc_blob_hdr_t *h = blob;
    if (h->magic != WBC_BLOB_MAGIC) return WBC_EMAGIC;
    if (h->version != WBC_BLOB_VERSION) return WBC_EVERSION;
    const uint32_t total = h->total_len;
    if (total < WBC_BLOB_HDR_BYTES || total > len) return WBC_ESIZE;
    if (wbc_crc32(0, (const uint8_t *)blob + WBC_BLOB_CRC_FROM, total - WBC_BLOB_CRC_FROM) != h->crc32)
        return WBC_ECRC;
    if (h->hdr_bytes != WBC_BLOB_HDR_BYTES || h->layout != WBC_LAYOUT_WT || h->n_layers != WBC_N_LAYERS
        || h->align != WBC_BLOB_ALIGN || h->wdtype > WBC_WDTYPE_F16
        || h->policy_mask == 0 || (h->policy_mask >> WBC_N_POLICIES) != 0)
        return WBC_ELAYOUT;
    const uint32_t esz = h->wdtype == WBC_WDTYPE_F32 ? 4u : 2u;
    uint32_t end = WBC_BLOB_HDR_BYTES;   /* arrays must follow each other in canonical order */
    for (uint32_t p = 0; p < WBC_N_POLICIES; p++) {
        const int present = (h->policy_mask >> p) & 1u;
        for (uint32_t l = 0; l < WBC_N_LAYERS; l++) {
            const wbc_layer_t *L = &h->layer[p][l];
            if (!present) {
                if (L->in | L->out | L->w_off | L->b_off) return WBC_ELAYOUT;
                continue;
            }
            if (L->in != k_in[l] || L->out != k_out[l]) return WBC_ELAYOUT;
            const uint32_t wbytes = L->in * L->out * esz, bbytes = L->out * 4u;
            if ((L->w_off | L->b_off) & (WBC_BLOB_ALIGN - 1u)) return WBC_ELAYOUT;
            if (L->w_off > total || total - L->w_off < wbytes) return WBC_ESIZE;
            if (L->b_off > total || total - L->b_off < bbytes) return WBC_ESIZE;
            if (L->w_off < end || L->b_off < L->w_off + wbytes) return WBC_ELAYOUT;
            end = L->b_off + bbytes;
        }
    }
    return WBC_OK;
}
