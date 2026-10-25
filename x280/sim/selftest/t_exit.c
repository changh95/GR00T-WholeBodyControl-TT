/* HTIF exit code, console capture and counter report.
 * -DEXIT_CODE=n  value returned from main (default 0); -DSPIN_FOREVER  never exits (timeout test). */
#include "rt.h"

#ifndef EXIT_CODE
#define EXIT_CODE 0
#endif

int main(void) {
    htif_puts("hello from x280-sim\n");
#ifdef SPIN_FOREVER
    for (;;) __asm__ volatile("");
#endif
    uint64_t instret = CSR_READ(minstret), cycle = CSR_READ(mcycle);
    htif_puts("minstret=");
    htif_putdec(instret);
    htif_puts(" mcycle=");
    htif_putdec(cycle);
    htif_puts(" exit=");
    htif_putdec(EXIT_CODE);
    htif_putc('\n');
    return EXIT_CODE;
}
