/* HTIF console/exit and the C trap handler of the spike self-test images (protocol in rt.h). */
#include "rt.h"

volatile uint64_t tohost __attribute__((section(".htif"), aligned(64))) = 0;
volatile uint64_t fromhost __attribute__((section(".htif"), aligned(8))) = 0;

volatile struct trap_rec trap_last;
volatile int trap_resume;

static void htif_send(uint64_t v) {
    while (tohost) {}
    tohost = v;
}

void htif_putc(char c) { htif_send((1ull << 56) | (1ull << 48) | (uint8_t)c); }

void htif_puts(const char *s) { while (*s) htif_putc(*s++); }

void htif_puthex(uint64_t v) {
    htif_puts("0x");
    for (int i = 60; i >= 0; i -= 4) htif_putc("0123456789abcdef"[(v >> i) & 0xf]);
}

void htif_putdec(uint64_t v) {
    char buf[21];
    int n = 0;
    do { buf[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n) htif_putc(buf[--n]);
}

void htif_exit(uint64_t code) {
    htif_send((code << 1) | 1);
    for (;;) {}
}

/* Called by trap_entry (crt0.S) with the saved-GPR frame. */
void trap_handler(uint64_t *frame) {
    (void)frame;
    uint64_t mcause = CSR_READ(mcause), mepc = CSR_READ(mepc), mtval = CSR_READ(mtval);
    trap_last.count++;
    trap_last.mcause = mcause;
    trap_last.mepc = mepc;
    trap_last.mtval = mtval;
    if (trap_resume && !(mcause >> 63)) {
        uint16_t lo = *(volatile uint16_t *)mepc;   /* instruction length from its low bits */
        CSR_WRITE(mepc, mepc + ((lo & 3) == 3 ? 4 : 2));
        return;
    }
    htif_puts("TRAP mcause=");
    htif_puthex(mcause);
    htif_puts(" mepc=");
    htif_puthex(mepc);
    htif_puts(" mtval=");
    htif_puthex(mtval);
    htif_putc('\n');
    htif_exit(TRAP_EXIT_BASE + (mcause & 0xff));
}
