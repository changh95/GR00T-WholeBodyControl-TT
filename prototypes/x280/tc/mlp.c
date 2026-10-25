#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
/* y[o] = b[o] + sum_i W[o][i]*x[i]; W stored row-major as bf16 bits, widened to fp32 via integer shift
   (needs only base V, not Zvfh/Zvfbfmin). */
static void gemv_bf16w_f32(const uint16_t *W, const float *b, const float *x, float *y, int O, int I) {
  for (int o = 0; o < O; o++) {
    const uint16_t *w = W + (size_t)o * I;
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.f, vlmax);
    for (int i = 0; i < I;) {
      size_t vl = __riscv_vsetvl_e16m2(I - i);
      vuint16m2_t wb = __riscv_vle16_v_u16m2(w + i, vl);
      vuint32m4_t w32 = __riscv_vsll_vx_u32m4(__riscv_vzext_vf2_u32m4(wb, vl), 16, vl);
      vfloat32m4_t wf = __riscv_vreinterpret_v_u32m4_f32m4(w32);
      vfloat32m4_t xv = __riscv_vle32_v_f32m4(x + i, vl);
      acc = __riscv_vfmacc_vv_f32m4_tu(acc, wf, xv, vl);
      i += vl;
    }
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(b[o], 1);
    y[o] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc, z, vlmax));
  }
}
static inline float expm1_approx(float v) { /* v<=0: placeholder poly; real kernel uses range-reduced exp */
  float r = v * (1.f + v * (0.5f + v * (1.f/6 + v * (1.f/24))));
  return r < -1.f ? -1.f : r;
}
static void elu(float *v, int n) { for (int i = 0; i < n; i++) if (v[i] < 0.f) v[i] = expm1_approx(v[i]); }
__attribute__((aligned(64))) uint16_t W0[256*516]; float B0[256];
__attribute__((aligned(64))) float x0[516], h0[256];
volatile uint32_t mbox[16];
int main(void) {
  for (;;) {
    while (mbox[0] == mbox[1]) {}
    gemv_bf16w_f32(W0, B0, x0, h0, 256, 516); elu(h0, 256);
    mbox[1] = mbox[0];
  }
}
