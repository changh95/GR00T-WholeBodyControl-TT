/* Illegal-instruction trap. `illegal_site` holds 0xc0001073 (csrrw x0, cycle, x0, i.e. `unimp` without RVC:
 * a write to a read-only CSR, illegal in every configuration).
 * Default: the trap handler prints "TRAP mcause= mepc= mtval=" and exits with TRAP_EXIT_BASE + 2.
 * -DRESUME: the handler skips the instruction; main checks the record and exits 0. */
#include "rt.h"

extern char illegal_site[];

int main(void) {
#ifdef RESUME
    trap_resume = 1;
#endif
    htif_puts("before illegal\n");
    __asm__ volatile(".globl illegal_site\nillegal_site:\n.4byte 0xc0001073" ::: "memory");
    htif_puts("after illegal traps=");
    htif_putdec(trap_last.count);
    htif_puts(" mcause=");
    htif_puthex(trap_last.mcause);
    htif_puts(" mepc=");
    htif_puthex(trap_last.mepc);
    htif_puts(" mtval=");
    htif_puthex(trap_last.mtval);
    htif_putc('\n');
    int ok = trap_last.count == 1 && trap_last.mcause == 2 && trap_last.mepc == (uint64_t)illegal_site
             && trap_last.mtval == 0xc0001073u;
    return ok ? 0 : 1;
}
