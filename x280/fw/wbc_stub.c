/* Stand-in kernel, linked only when the kernel lane's Makefile.kernel / wbc_mlp.c are absent
 * (Makefile prints so). Every call fails, so a monitor built this way can still be tested for its
 * protocol, and an INFER on it reports state FAULT with mcause MB_ERR_KERNEL and mtval -1 instead of
 * returning a plausible action. L36.variants_built = 0 and L36.entry_flags has MB_ENTRY_STUB_KERNEL.
 */
#include "wbc_api.h"

const uint32_t wbc_kernel_is_stub = 1;

int wbc_blob_check(const void *blob, uint32_t len)
{
    (void)blob; (void)len;
    return -1;
}

int wbc_infer(const void *blob, uint32_t policy, uint32_t variant,
              const float *obs, float *act, void *scratch)
{
    (void)blob; (void)policy; (void)variant; (void)obs; (void)act; (void)scratch;
    return -1;
}

uint32_t mon_variants_built(void) { return 0; }
