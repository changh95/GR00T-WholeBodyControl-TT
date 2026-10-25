/* Polynomial expf and ELU(alpha = 1) for the gear_wbc MLP (PLAN.md WP-3.1). No libm.
 *
 * expf: Cody-Waite range reduction x = n ln2 + r, |r| <= ln2/2, then Cephes' degree-7 form
 *   exp(r) = 1 + r + r^2 P(r),  P = minimax degree 5 (Cephes expf.c coefficients),
 * and the 2^n scale built in the exponent field. n = round(x log2e) uses the 1.5*2^23 shifter, so
 * there is no float->int conversion (whose NaN/overflow behaviour differs between x86 and RISC-V).
 * Every step is one separately rounded IEEE operation (-ffp-contract=off), which makes the scalar
 * and the RVV version below bit-identical and the result the same on x86 and on the X280.
 *
 * Domain: x is clamped to [-87, 88] (2^n stays a normal number: n in [-126, 127]); NaN propagates.
 * exp(-87) = 1.6e-38, so ELU's exp(x) - 1 is exactly -1.0f below the clamp, as in float32 ORT.
 *
 * Measured error, exhaustive over all 2,237,530,114 float32 values in [-87, 88] against libm's
 * double exp/expm1 (`make -f Makefile.kernel kernel-expf-check`, 2026-10-24; the riscv build runs
 * the same IEEE operation sequence):
 *   wbc_expf:                  max 0.990 ulp (x = 70.4031), max relative error 8.16e-08
 *   ELU (exp(x) - 1), x <= 0:  max absolute error 4.82e-08 (x = -0.330415)
 * onnxruntime's float Elu also evaluates exp(x) - 1 in float32, so the two differ by about 1 ulp of 1.
 */
#include "wbc_mlp.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#define EXP_LO   (-87.0f)
#define EXP_HI   (88.0f)
#define LOG2E    1.44269504088896341f
#define SHIFTER  12582912.0f          /* 1.5 * 2^23: adding it rounds |t| < 2^22 to an integer (RNE) */
#define SHIFTER_BITS 0x4B400000u
#define LN2_HI   0.693359375f         /* 0x3F318000: 8 significant bits, so n * LN2_HI is exact */
#define LN2_LO   (-2.12194440e-4f)    /* ln2 - LN2_HI */
#define P0 1.9875691500e-4f           /* Cephes expf.c, highest order first */
#define P1 1.3981999507e-3f
#define P2 8.3334519073e-3f
#define P3 4.1665795894e-2f
#define P4 1.6666665459e-1f
#define P5 5.0000001201e-1f

static inline float as_f32(uint32_t u) { union { uint32_t u; float f; } c = { .u = u }; return c.f; }
static inline uint32_t as_u32(float f) { union { float f; uint32_t u; } c = { .f = f }; return c.u; }

float wbc_expf(float x)
{
    if (x < EXP_LO) x = EXP_LO;               /* comparisons are false for NaN: it passes through */
    if (x > EXP_HI) x = EXP_HI;
    float t = x * LOG2E;
    t = t + SHIFTER;
    uint32_t tb = as_u32(t);
    t = t - SHIFTER;                          /* n, exactly */
    float r = x - t * LN2_HI;
    r = r - t * LN2_LO;
    float z = r * r;
    float p = P0;
    p = p * r + P1;
    p = p * r + P2;
    p = p * r + P3;
    p = p * r + P4;
    p = p * r + P5;
    float e = p * z + r;
    e = e + 1.0f;
    uint32_t scale = (tb - SHIFTER_BITS + 127u) << 23;   /* 2^n, n in [-126, 127] */
    return e * as_f32(scale);
}

void wbc_elu(float *v, uint32_t n)
{
#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t i = 0; i < n; i++) {
        float x = v[i];
        v[i] = x > 0.0f ? x : wbc_expf(x) - 1.0f;
    }
}

#if defined(__riscv_vector)
/* The same operation sequence as wbc_expf/wbc_elu on vl lanes (e32, m4); lanes with x > 0 take x. */
void wbc_elu_rvv(float *v, uint32_t n)
{
    for (uint32_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t x = __riscv_vle32_v_f32m4(v + i, vl);
        vfloat32m4_t xc = __riscv_vfmerge_vfm_f32m4(x, EXP_LO, __riscv_vmflt_vf_f32m4_b8(x, EXP_LO, vl), vl);
        xc = __riscv_vfmerge_vfm_f32m4(xc, EXP_HI, __riscv_vmfgt_vf_f32m4_b8(xc, EXP_HI, vl), vl);
        vfloat32m4_t t = __riscv_vfmul_vf_f32m4(xc, LOG2E, vl);
        t = __riscv_vfadd_vf_f32m4(t, SHIFTER, vl);
        vuint32m4_t tb = __riscv_vreinterpret_v_f32m4_u32m4(t);
        t = __riscv_vfsub_vf_f32m4(t, SHIFTER, vl);
        vfloat32m4_t r = __riscv_vfsub_vv_f32m4(xc, __riscv_vfmul_vf_f32m4(t, LN2_HI, vl), vl);
        r = __riscv_vfsub_vv_f32m4(r, __riscv_vfmul_vf_f32m4(t, LN2_LO, vl), vl);
        vfloat32m4_t z = __riscv_vfmul_vv_f32m4(r, r, vl);
        vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(P0, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), P1, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), P2, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), P3, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), P4, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), P5, vl);
        vfloat32m4_t e = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, z, vl), r, vl);
        e = __riscv_vfadd_vf_f32m4(e, 1.0f, vl);
        vuint32m4_t sc = __riscv_vsll_vx_u32m4(
            __riscv_vadd_vx_u32m4(__riscv_vsub_vx_u32m4(tb, SHIFTER_BITS, vl), 127u, vl), 23, vl);
        e = __riscv_vfmul_vv_f32m4(e, __riscv_vreinterpret_v_u32m4_f32m4(sc), vl);
        e = __riscv_vfsub_vf_f32m4(e, 1.0f, vl);
        vfloat32m4_t y = __riscv_vmerge_vvm_f32m4(e, x, __riscv_vmfgt_vf_f32m4_b8(x, 0.0f, vl), vl);
        __riscv_vse32_v_f32m4(v + i, y, vl);
        i += (uint32_t)vl;
    }
}
#endif

#if defined(WBC_EXPF_CHECK_MAIN)
/* Hosted x86 self-check, never part of the kernel (`make -f Makefile.kernel kernel-expf-check`):
 * every float32 in [-87, 88] against libm's double exp/expm1. */
#include <math.h>
#include <stdio.h>

int main(void)
{
    double max_ulp = 0, max_rel = 0, max_elu = 0;
    float at_ulp = 0, at_rel = 0, at_elu = 0;
    unsigned long n = 0;
    const uint32_t ranges[2][2] = {
        { 0x00000000u, as_u32(EXP_HI) },   /* +0 .. 88 */
        { 0x80000000u, as_u32(EXP_LO) },   /* -0 .. -87 */
    };
    for (int r = 0; r < 2; r++) {
        for (uint64_t u = ranges[r][0]; u <= ranges[r][1]; u++, n++) {
            float x = as_f32((uint32_t)u);
            float y = wbc_expf(x);
            double ref = exp((double)x);
            int e;
            frexp(ref, &e);
            double err = fabs((double)y - ref);
            double ulp = err / ldexp(1.0, e - 24), rel = err / ref;
            if (ulp > max_ulp) { max_ulp = ulp; at_ulp = x; }
            if (rel > max_rel) { max_rel = rel; at_rel = x; }
            if (x <= 0.0f) {
                float v = x;
                wbc_elu(&v, 1);
                double ae = fabs((double)v - expm1((double)x));
                if (ae > max_elu) { max_elu = ae; at_elu = x; }
            }
        }
    }
    printf("n=%lu expf max_ulp=%.4f (x=%.9g) max_rel=%.4g (x=%.9g) elu max_abs=%.4g (x=%.9g)\n",
           n, max_ulp, at_ulp, max_rel, at_rel, max_elu, at_elu);
    return !(max_ulp < 4.0 && max_elu < 2.5e-7);
}
#endif
