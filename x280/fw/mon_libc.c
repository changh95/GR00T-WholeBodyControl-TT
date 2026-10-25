/* The four functions a freestanding C compiler may call on its own (struct copies, zero-inits) even
 * with -fno-builtin. Weak, like the kernel lane's memcpy/memset in wbc_mlp.c, so either set links and
 * the image has no undefined symbols (PLAN.md T-3.2). Byte loops: LIM is uncached, so the width of the
 * access matters less than not depending on libc.
 */
#include <stddef.h>
#include <stdint.h>

__attribute__((weak)) void *memcpy(void *dst, const void *src, size_t n)
{
    volatile uint8_t *d = dst;
    const uint8_t *s = src;
    while (n--) *d++ = *s++;
    return dst;
}

__attribute__((weak)) void *memset(void *dst, int c, size_t n)
{
    volatile uint8_t *d = dst;
    while (n--) *d++ = (uint8_t)c;
    return dst;
}

__attribute__((weak)) void *memmove(void *dst, const void *src, size_t n)
{
    volatile uint8_t *d = dst;
    const uint8_t *s = src;
    if (d < s) {
        while (n--) *d++ = *s++;
    } else {
        d += n;
        s += n;
        while (n--) *--d = *--s;
    }
    return dst;
}

__attribute__((weak)) int memcmp(const void *a, const void *b, size_t n)
{
    const uint8_t *x = a, *y = b;
    for (; n; n--, x++, y++)
        if (*x != *y) return *x < *y ? -1 : 1;
    return 0;
}
