/* gear_wbc Balance/Walk MLP forward pass for the X280 (PLAN.md WP-3.1, §1.4.3, §1.4.4).
 *
 * Shared C contract between the kernel lane (wbc_mlp.c, elu.c) and the firmware lane (monitor.c).
 * The names in the CONTRACT block are fixed; either lane may only add names below it.
 *
 * Freestanding: no libc, no libm, no libcalls; the same sources build for x86-64 (ctypes parity
 * test against onnxruntime) and for riscv64-unknown-elf (LIM image at 0x0800_0000).
 *
 * Graph (verified from both ONNX files by x280/tools/pack_weights.py --describe; opset 11):
 *   est  = Gemm(516->256) Elu Gemm(256->256) Elu Gemm(256->35)          (transB=1, alpha=beta=1, Elu alpha=1)
 *   lat  = est[3:35] / max(ReduceL2(est[3:35]), 1e-12f)                (Clip min 9.99999996e-13 = (float)1e-12)
 *   in   = concat(obs[430:516], est[0:3], lat)                          (121 = 86 newest frame + 3 + 32)
 *   act  = Gemm(121->512) Elu Gemm(512->256) Elu Gemm(256->256) Elu Gemm(256->15)
 *
 * Arithmetic contract (all variants): y[o] = (((b[o] + w[o][0]*x[0]) + w[o][1]*x[1]) + ...), every
 * multiply and add rounded separately (RNE, no fused multiply-add, -ffp-contract=off), so v0 on
 * x86 == v0 on riscv == v1 bit for bit, and v2 on x86 == v2 on riscv bit for bit. The hart must run
 * with frm = 0 (round to nearest even) and must not flush subnormals (RISC-V never does).
 * v3 == v2 bit for bit on every input: it rounds the same product w*x[i] and the same sums in the
 * same order, and only moves v2's exact 2^112 weight scale onto x[i] when |x[i]| < 2^16 (wbc_mlp.c).
 */
#ifndef WBC_MLP_H
#define WBC_MLP_H

#include <stdint.h>

/* ---- CONTRACT (fixed) ------------------------------------------------------------------------ */
#define WBC_OBS_DIM 516
#define WBC_ACT_DIM 15
#define WBC_POLICY_BALANCE 0   /* GR00T-WholeBodyControl-Balance.onnx */
#define WBC_POLICY_WALK    1   /* GR00T-WholeBodyControl-Walk.onnx */
#define WBC_VARIANT_V0 0  /* scalar fp32, -ffp-contract=off: bit-parity reference */
#define WBC_VARIANT_V1 1  /* RVV fp32 (riscv builds only) */
#define WBC_VARIANT_V2 2  /* fp16 weights widened with INTEGER ops (no Zvfh), fp32 math */
#define WBC_SCRATCH_BYTES 4096  /* activations: 960 floats used (3,840 B) */

int wbc_blob_check(const void *blob, uint32_t len);   /* 0 ok, <0 bad magic/version/CRC32/size */
int wbc_infer(const void *blob, uint32_t policy, uint32_t variant,
              const float *obs, float *act, void *scratch);  /* 0 ok, <0 error (e.g. variant not built) */

/* ---- Additions (kernel lane) ----------------------------------------------------------------- */

/* Scalar fp16 path on every build. On x86 WBC_VARIANT_V2 is this same code; on riscv V2 is the RVV
 * column-broadcast kernel and V2S is the scalar timing point for E4. Bit-identical to V2. */
#define WBC_VARIANT_V2S 3
/* fp16, bit-identical to V2 on every input, with one vector op fewer per weight: V2's exact 2^112
 * scale rides on x[i] (one scalar multiply per row) when |x[i]| < 2^16, else V2's ops are used (x86:
 * for that row; riscv: that layer is computed again by V2). riscv: e32m8 GEMV in asm with the next
 * row's weights and x[i+1] loaded a row ahead (wbc_mlp.c); x86: the same arithmetic in scalar code. */
#define WBC_VARIANT_V3 4
#define WBC_N_VARIANTS 5

/* Return codes (all < 0). wbc_infer never writes act on an error. */
#define WBC_OK        0
#define WBC_EINVAL   -1   /* NULL or misaligned pointer (all four need 4 B alignment) */
#define WBC_EMAGIC   -2
#define WBC_EVERSION -3
#define WBC_ESIZE    -4   /* len < header, total_len > len, or an array outside total_len */
#define WBC_ECRC     -5
#define WBC_ELAYOUT  -6   /* hdr_bytes, layout, n_layers, dims, alignment or overlap mismatch */
#define WBC_EPOLICY  -7   /* policy out of range or not present in the blob */
#define WBC_EVARIANT -8   /* variant unknown or not built for this target */
#define WBC_EDTYPE   -9   /* variant needs the other weight dtype (v0/v1: fp32, v2/v2s/v3: fp16) */

/* Weight blob v1 (written by x280/tools/pack_weights.py; little-endian; placed at WBC_BLOB_BASE).
 * Every array is 64 B aligned relative to the blob start. Weights are stored transposed,
 * W^T[in][out] row-major (column-major W), so one input i streams a contiguous out-vector
 * (column-broadcast GEMV); biases are always fp32. CRC-32 is zlib's (reflected 0xEDB88320, init
 * and final xor 0xFFFFFFFF) over bytes [WBC_BLOB_CRC_FROM, total_len). */
#define WBC_BLOB_MAGIC     0x42434257u  /* "WBCB" */
#define WBC_BLOB_VERSION   1u
#define WBC_BLOB_HDR_BYTES 320u
#define WBC_BLOB_CRC_FROM  20u
#define WBC_BLOB_ALIGN     64u
#define WBC_LAYOUT_WT      1u           /* W^T [in][out], 64 B aligned arrays */
#define WBC_WDTYPE_F32     0u
#define WBC_WDTYPE_F16     1u           /* IEEE binary16 */
#define WBC_N_POLICIES     2
#define WBC_N_LAYERS       7            /* estimator 0..2, actor 3..6 */
#define WBC_BLOB_BASE      0x08011000u  /* PLAN.md §1.4.4 weights line */
#define WBC_BLOB_LIMIT     0x081E0000u  /* LIM end, exclusive: 15 x 128 KiB from 0x0800_0000 at reset
                                           (tt-isa-documentation L2CPUTile/Caches.md:73, 92) */

typedef struct {
    uint32_t in, out;     /* GEMV shape */
    uint32_t w_off;       /* byte offset of W^T [in][out] from the blob start */
    uint32_t b_off;       /* byte offset of the fp32 bias [out] */
} wbc_layer_t;

typedef struct {
    uint32_t magic;          /*   0 */
    uint32_t version;        /*   4 */
    uint32_t hdr_bytes;      /*   8 */
    uint32_t total_len;      /*  12 */
    uint32_t crc32;          /*  16  over [20, total_len) */
    uint32_t wdtype;         /*  20 */
    uint32_t layout;         /*  24 */
    uint32_t policy_mask;    /*  28  bit p = policy p present (fp32: one policy fits in LIM, DP4) */
    uint32_t n_layers;       /*  32 */
    uint32_t align;          /*  36 */
    uint32_t src_crc32[2];   /*  40  zlib CRC-32 of each source .onnx file (0 if absent) */
    uint32_t reserved[4];    /*  48 */
    wbc_layer_t layer[WBC_N_POLICIES][WBC_N_LAYERS];  /*  64 .. 287; zeros for an absent policy */
    char tag[32];            /* 288  NUL-padded ASCII, e.g. "pack_weights v1 fp16" */
} wbc_blob_hdr_t;            /* 320 */

_Static_assert(sizeof(wbc_layer_t) == 16, "wbc_layer_t");
_Static_assert(sizeof(wbc_blob_hdr_t) == WBC_BLOB_HDR_BYTES, "wbc_blob_hdr_t");

/* Bitmask of the variants this build can run (bit v = variant v): x86 0b11101, riscv+V 0b11111. */
uint32_t wbc_variants_built(void);

/* zlib-compatible CRC-32 (bitwise, no table: a table in LIM would be uncached anyway). Start with
 * crc = 0; chain by passing the previous result. About 8 x 3 ALU ops per byte. */
uint32_t wbc_crc32(uint32_t crc, const void *p, uint32_t n);

/* The v2/v2s weight widening for one value (exported for the exhaustive x86 test): exact for every
 * finite binary16 including subnormals; +-inf/NaN inputs (never packed) map to finite values. */
float wbc_f16_to_f32(uint16_t h);

void *wbc_memcpy(void *dst, const void *src, uint32_t n);
void *wbc_memset(void *dst, int c, uint32_t n);

/* elu.c: polynomial expf and ELU(alpha = 1) = x > 0 ? x : expf(x) - 1, the formula onnxruntime's
 * Elu uses. Error bounds are documented (and measured exhaustively) in elu.c. */
float wbc_expf(float x);
void wbc_elu(float *v, uint32_t n);          /* scalar, in place */
#if defined(__riscv_vector)
void wbc_elu_rvv(float *v, uint32_t n);      /* RVV e32/m4, in place; bit-identical to wbc_elu */
#endif

#endif /* WBC_MLP_H */
