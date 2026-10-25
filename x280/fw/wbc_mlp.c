/* gear_wbc MLP forward pass, freestanding (PLAN.md WP-3.1; contract and graph in wbc_mlp.h).
 *
 * Variants (all compute y[o] = b[o] + sum_i w[o][i] x[i] in ascending i, separately rounded):
 *   v0  scalar fp32, 8 outputs per pass so x[i] is loaded once per 8 weights (bit-parity reference)
 *   v1  RVV fp32 column-broadcast GEMV: e32/m4, four register groups (4*VLMAX outputs, 256 at
 *       VLEN 512) accumulate in registers while each input streams one contiguous W^T row
 *   v2  fp16 weights: RVV on riscv (vle16 + integer widening multiply by 2^13 and mask + one exact
 *       fp32 scale by 2^112, no Zvfh); on x86 the scalar v2s code
 *   v2s scalar fp16 (the same widening per weight); built everywhere
 * -DWBC_RVV_FMA makes v1/v2 use vfmacc (one rounding per MAC): faster, but then no longer
 * bit-identical to v0/v2s (E4 experiment only).
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
    uint32_t m = (1u << WBC_VARIANT_V0) | (1u << WBC_VARIANT_V2) | (1u << WBC_VARIANT_V2S);
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
#else
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
