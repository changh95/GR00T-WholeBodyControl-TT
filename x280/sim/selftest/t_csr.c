/* Which X280-only CSR accesses trap in spike. One line per access:
 *   "ACCESS <name> trap mcause=<hex>"  or  "ACCESS <name> ok"
 * X280 RNMI CSRs mnscratch/mnepc/mncause/mnstatus are 0x350-0x353 (tt-isa-documentation
 * BlackholeA0/L2CPUTile/RNMIs.md:6, 63-70); 0x7c1 holds the SiFive feature-disable chicken bits
 * (L2CPUTile/README.md:49); mnret is the word 0x70200073 (RNMIs.md:31). The standard Smrnmi CSRs
 * 0x740-0x744 and the always-present M-mode CSRs are included for contrast. Exit 0 when all ran. */
#include "rt.h"

static void report(const char *name, uint64_t traps_before) {
    htif_puts("ACCESS ");
    htif_puts(name);
    if (trap_last.count != traps_before) {
        htif_puts(" trap mcause=");
        htif_puthex(trap_last.mcause);
    } else {
        htif_puts(" ok");
    }
    htif_putc('\n');
}

#define TRY(name, insn) do { uint64_t n_ = trap_last.count; \
    __asm__ volatile(insn ::: "t0", "memory"); report(name, n_); } while (0)

int main(void) {
    trap_resume = 1;
    TRY("csrr_0x350", "csrr t0, 0x350");
    TRY("csrr_0x351", "csrr t0, 0x351");
    TRY("csrr_0x352", "csrr t0, 0x352");
    TRY("csrr_0x353", "csrr t0, 0x353");
    TRY("csrr_0x7c1", "csrr t0, 0x7c1");
    TRY("csrw_0x350", "csrw 0x350, zero");
    TRY("csrw_0x351", "csrw 0x351, zero");
    TRY("csrw_0x352", "csrw 0x352, zero");
    TRY("csrw_0x353", "csrw 0x353, zero");
    TRY("csrw_0x7c1", "csrw 0x7c1, zero");
    TRY("csrsi_0x353_8", "csrsi 0x353, 8");        /* the firmware's RNMI re-arm (PLAN.md WP-3.2) */
    TRY("csrwi_0x7c1_0", "csrwi 0x7c1, 0");        /* the firmware's chicken-bit clear */
    TRY("csrr_0x740", "csrr t0, 0x740");          /* standard Smrnmi mnscratch */
    TRY("csrr_0x741", "csrr t0, 0x741");          /* mnepc */
    TRY("csrr_0x742", "csrr t0, 0x742");          /* mncause */
    TRY("csrr_0x744", "csrr t0, 0x744");          /* mnstatus */
    TRY("csrr_misa", "csrr t0, misa");
    TRY("csrr_mhartid", "csrr t0, mhartid");
    TRY("csrr_mcycle", "csrr t0, mcycle");
    TRY("csrr_minstret", "csrr t0, minstret");
    TRY("csrr_vlenb", "csrr t0, vlenb");
    TRY("csrr_time", "csrr t0, time");              /* rdtime: needs Zicntr in spike */
    /* Last: if mnret did NOT trap it would jump to mnepc. */
    TRY("mnret", ".4byte 0x70200073");
    htif_puts("misa=");
    htif_puthex(CSR_READ(misa));
    htif_putc('\n');
    return 0;
}
