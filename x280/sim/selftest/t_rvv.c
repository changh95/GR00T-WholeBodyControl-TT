/* RVV at VLEN 512: vlenb, VLMAX per LMUL, and a vle32/vfmacc/vse32 strip-mined loop checked bit-exactly
 * against a scalar fmadd.s loop (vfmacc.vv is a fused multiply-add per element, one rounding, like fmadd.s).
 * Exit 0 on pass, else 0x100 + number of mismatching elements (or 0x200 + a geometry code). */
#include <stddef.h>
#include <riscv_vector.h>
#include "rt.h"

#define N 100   /* not a multiple of VLMAX (16 at e32m1), so the tail vl is exercised */
#define K 8

static float a[K][N], b[K][N], acc_v[N], acc_s[N];

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }

static float rnd(uint32_t *s) {   /* uniform in [-2, 2), exact float from 24 random bits */
    return (float)(int32_t)(lcg(s) >> 8) * (1.0f / (1 << 22)) - 2.0f;
}

static inline float fmadd_s(float x, float y, float z) {   /* scalar fused reference, no libcall */
    float r;
    __asm__("fmadd.s %0, %1, %2, %3" : "=f"(r) : "f"(x), "f"(y), "f"(z));
    return r;
}

static void put_kv(const char *k, uint64_t v) { htif_puts(k); htif_putdec(v); htif_putc(' '); }

int main(void) {
    uint64_t vlenb = CSR_READ(vlenb);
    size_t m1 = __riscv_vsetvlmax_e32m1(), m4 = __riscv_vsetvlmax_e32m4(), m8 = __riscv_vsetvlmax_e32m8();
    put_kv("vlenb=", vlenb);
    put_kv("vlmax_e32m1=", m1);
    put_kv("vlmax_e32m4=", m4);
    put_kv("vlmax_e32m8=", m8);
    htif_putc('\n');
    if (vlenb != 64 || m1 != 16 || m4 != 64 || m8 != 128) return 0x200 + 1;

    uint32_t seed = 12345;
    for (int k = 0; k < K; k++)
        for (int i = 0; i < N; i++) { a[k][i] = rnd(&seed); b[k][i] = rnd(&seed); }
    for (int i = 0; i < N; i++) acc_v[i] = acc_s[i] = rnd(&seed);

    size_t n_strips = 0, last_vl = 0;
    for (int k = 0; k < K; k++) {
        for (size_t i = 0; i < N;) {
            size_t vl = __riscv_vsetvl_e32m1(N - i);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[k][i], vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[k][i], vl);
            vfloat32m1_t vc = __riscv_vle32_v_f32m1(&acc_v[i], vl);
            vc = __riscv_vfmacc_vv_f32m1(vc, va, vb, vl);
            __riscv_vse32_v_f32m1(&acc_v[i], vc, vl);
            i += vl;
            n_strips++;
            last_vl = vl;
        }
        for (int i = 0; i < N; i++) acc_s[i] = fmadd_s(a[k][i], b[k][i], acc_s[i]);
    }
    put_kv("strips=", n_strips);
    put_kv("last_vl=", last_vl);
    if (n_strips != (size_t)K * ((N + 15) / 16) || last_vl != N % 16) { htif_putc('\n'); return 0x200 + 2; }

    int bad = 0;
    for (int i = 0; i < N; i++) {
        union { float f; uint32_t u; } x = {acc_v[i]}, y = {acc_s[i]};
        bad += x.u != y.u;
    }
    put_kv("mismatch=", (uint64_t)bad);
    union { float f; uint32_t u; } c0 = {acc_v[0]};
    htif_puts("acc0_bits=");
    htif_puthex(c0.u);
    htif_putc('\n');
    return bad ? 0x100 + bad : 0;
}
