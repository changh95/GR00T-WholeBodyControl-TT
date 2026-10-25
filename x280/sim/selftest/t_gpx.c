/* The X280 regions run_spike.py adds next to LIM, used the way the park stub uses them (PLAN.md WP-3.2,
 * §5.2 PARK protocol): a stub loaded into the GP scratch (0x2001_0100, MemoryMap.md:113) runs fence.i, spins
 * while bit h of 0x2001_0414 (RNMIs.md:38-46) is set, loads the hart's reset handler address from
 * 0x2001_0000 + 8h (MemoryMap.md:111) and jumps to it. Spike has no RNMI source, so hart 0 jumps into the stub
 * itself; with -DTWO_HARTS hart 1 plays the host and clears the trigger bit after hart 0 is spinning.
 * Also reads/writes the hart-0 BEU page at 0x0170_0000 (MemoryMap.md:69), which spike models as plain RAM.
 * Exit 0 when the stub came back through the reset vector. */
#include "rt.h"

#define GPX_RESETVEC 0x20010000ul
#define GPX_RNMI_TRIGGER 0x20010414ul
#define BEU0 0x01700000ul

__attribute__((noreturn)) void resume_point(void);
extern char gp_stub[];

/* Hart 0's reset handler address, loaded from the ELF into 0x2001_0000. */
__attribute__((section(".gpx.resetvec"), used)) const uint64_t gpx_resetvec[4] = {
    (uint64_t)resume_point, 0, 0, 0};

__asm__(
    ".section .gpx.scratch, \"ax\"\n"
    ".globl gp_stub\n"
    "gp_stub:\n"
    "  fence.i\n"
    "  csrr t0, mhartid\n"
    "  li   t1, 0x20010414\n"
    "1: lw  t2, 0(t1)\n"
    "  srl  t2, t2, t0\n"
    "  andi t2, t2, 1\n"
    "  bnez t2, 1b\n"
    "  slli t0, t0, 3\n"
    "  li   t1, 0x20010000\n"
    "  add  t1, t1, t0\n"
    "  ld   t1, 0(t1)\n"
    "  jr   t1\n"
    ".globl gp_stub_end\n"
    "gp_stub_end:\n"
    ".text\n");
extern char gp_stub_end[];

static volatile uint32_t hart0_parked;

void secondary_main(uint64_t hartid) {
    if (hartid != 1) return;
    while (!hart0_parked) {}
    for (volatile int i = 0; i < 2000; i++) {}
    *(volatile uint32_t *)GPX_RNMI_TRIGGER &= ~1u;   /* "host" releases hart 0 */
}

void resume_point(void) {
    htif_puts("resumed via gp stub\n");
    htif_exit(0);
}

int main(void) {
    volatile uint32_t *beu = (volatile uint32_t *)BEU0;
    uint32_t beu0 = beu[0];
    beu[0] = 0x5a5a5a5a;
    htif_puts("beu0_initial=");
    htif_puthex(beu0);
    htif_puts(" beu0_readback=");
    htif_puthex(beu[0]);
    htif_puts(" stub_bytes=");
    htif_putdec((uint64_t)(gp_stub_end - gp_stub));
    htif_puts(" resetvec0=");
    htif_puthex(*(volatile uint64_t *)GPX_RESETVEC);
    htif_putc('\n');
    if (beu[0] != 0x5a5a5a5a) return 2;
#ifdef TWO_HARTS
    *(volatile uint32_t *)GPX_RNMI_TRIGGER |= 1u;
    hart0_parked = 1;
#endif
    __asm__ volatile("jr %0" :: "r"(gp_stub));
    __builtin_unreachable();
}
