/* Machine-mode trap handler of the monitor (PLAN.md WP-3.2: "trap handler dumps mcause/mepc/mtval to L1,
 * state 3").
 *
 * Why it never returns to the faulting code: a trap in the monitor means a payload (CALL), the kernel
 * (INFER/BENCH) or a LIM read (SELFTEST/LOAD) went wrong, and the hart cannot be reset (README.md:29-30).
 * So the handler dumps the CSRs, then either longjmps back into the command that armed the guard (which
 * completes its seq with state FAULT) or, for a trap outside any guard, restarts the poll loop on a fresh
 * stack. A trap inside the handler itself leaves the hart in wfi, still reachable by the host's RNMI.
 */
#include "monitor.h"

void mon_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval, uint64_t sp)
{
    (void)sp;
#if defined(SPIKE_HOST)
    /* spike has no RNMI: a machine timer interrupt armed by the scripted host stands in for one. */
    if ((mcause >> 63) && (mcause & 0xFFu) == 7u) sh_emulated_rnmi((1ull << 63) | 2u, mepc);
#endif
    if (g_rt.trap_depth) plat_park_forever();
    g_rt.trap_depth = 1;
    g_persist.n_traps++;
    g_rt.fault_mcause = mcause;
    g_rt.fault_mepc = mepc;
    g_rt.fault_mtval = mtval;
    volatile struct mb_l1 *l1 = MB_PTR(struct mb_l1, MB_L1_STATUS);
    l1->mcause = mcause;
    l1->mepc = mepc;
    l1->mtval = mtval;
    l1->state = MB_STATE_FAULT;
    g_rt.state = MB_STATE_FAULT;
    MB_PTR(struct mb_l36, MB_L36_IDENT)->n_traps = g_persist.n_traps;
    MB_FENCE_REL();
    g_rt.trap_depth = 0;
    if (g_rt.guard_active) {
        g_rt.guard_active = 0;
        mon_longjmp(g_rt.guard, 1);
    }
    g_rt.restart_pending = 1;
    plat_restart_after_trap();
}
