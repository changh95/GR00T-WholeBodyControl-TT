/* The firmware's view of the kernel contract (wbc_mlp.h, owned by the kernel lane, PLAN.md WP-3.1).
 *
 * Why a wrapper: the monitor must build (with wbc_stub.c) before or without the kernel lane's files, so
 * this falls back to a verbatim copy of the fixed CONTRACT block when wbc_mlp.h is absent.
 */
#ifndef X280_WBC_API_H
#define X280_WBC_API_H

#if __has_include("wbc_mlp.h")
#include "wbc_mlp.h"
#else
#include <stdint.h>
#define WBC_OBS_DIM 516
#define WBC_ACT_DIM 15
#define WBC_POLICY_BALANCE 0
#define WBC_POLICY_WALK    1
#define WBC_VARIANT_V0 0
#define WBC_VARIANT_V1 1
#define WBC_VARIANT_V2 2
#define WBC_SCRATCH_BYTES 16384
int wbc_blob_check(const void *blob, uint32_t len);
int wbc_infer(const void *blob, uint32_t policy, uint32_t variant,
              const float *obs, float *act, void *scratch);
#endif

#if WBC_SCRATCH_BYTES > 16384
#error "WBC_SCRATCH_BYTES must be <= 16384 (shared contract)"
#endif

/* wbc_variants_built() is a kernel-lane addition; the stub and the fallback contract report 0. */
uint32_t mon_variants_built(void);

/* 1 when the image links wbc_stub.c instead of the kernel lane's wbc_mlp.c. */
extern const uint32_t wbc_kernel_is_stub;

#endif /* X280_WBC_API_H */
