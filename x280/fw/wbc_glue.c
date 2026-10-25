/* Glue linked next to the kernel lane's wbc_mlp.c (never with wbc_stub.c): marks the image as carrying
 * the real kernel and forwards the kernel lane's optional variant bitmask to L36.variants_built.
 */
#include "wbc_api.h"

const uint32_t wbc_kernel_is_stub = 0;

uint32_t mon_variants_built(void)
{
#if defined(WBC_N_VARIANTS)
    return wbc_variants_built();
#else
    return 0xFFFFFFFFu;   /* the kernel did not say; "unknown" rather than "none" */
#endif
}
