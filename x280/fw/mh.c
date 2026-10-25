/* Multi-hart runtime of the monitor: harts 1-3 of L2CPU3 next to hart 0 (layout, line ownership and the job contract
 * in mh.h; the 4-hart bench design, sections A-B).
 *
 * Why it looks like this:
 * - Harts 1-3 are reachable only by an RNMI whose handler address points at code they never fetched (LIM): under
 *   `x280d serve --harts 4` their handler is crt0's _start and their RNMI exception handler the fixed MH_SEC_RNMI_EXC
 *   spin. crt0's secondary path (NMIE = 0) runs fence.i, waits for its own trigger bit to clear, sets up the hart like
 *   hart 0 (0x7c1, FS/VS, mtvec = mh_sec_trap_entry, its own MH stack), re-arms NMIE and calls mh_secondary_main().
 * - The secondaries never take an interrupt: mstatus.MIE stays 0, mie = MSIE | MTIE only makes wfi return (priv spec
 *   3.3.3: wfi resumes on any pending enabled interrupt, also with MIE = 0). An mtimecmp backstop every 100 us bounds
 *   each wfi, and an adoption self-test (self-IPI, mip read back, wfi) falls back to a ~5 us POLL loop when the IPI
 *   does not show up.
 * - A secondary writes only its own stat/aux lines, trap stack, private block and stack (MH region), plus whatever the
 *   job it runs writes (mh_bench: its own result line). It never calls trap.c, mon_heartbeat or api->result: those
 *   write L1/L36/L39 and longjmp hart 0's guard. Its traps go to mh_sec_trap() on its own trap stack.
 * - Hart 0 writes the ctl line only, with a seqlock on gen, and sends IPIs through the CLINT; every wait is bounded by
 *   mtime and keeps hart 0's heartbeat going. No AMO, LR or SC: plain stores by the one writer of each line.
 */
#include "monitor.h"

#define MH_IDLE_TICKS (PLAT_MTIME_HZ / 10000u)    /* 100 us: mtimecmp backstop of each idle wfi */
#define MH_POLL_TICKS (PLAT_MTIME_HZ / 200000u)   /* 5 us: one POLL-mode pass */
#define MH_HB_TICKS   (PLAT_MTIME_HZ / 100000u)   /* 10 us: hart 0's heartbeat while it waits for the harts */

#if defined(__riscv)
#define MH_FENCE() __asm__ volatile("fence iorw, iorw" ::: "memory")   /* CLINT stores vs LIM loads */
#else
#define MH_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif

struct mh_priv {           /* MH_PRIV_ADDR(h): hart h only (the bottom of its stack area) */
    mon_jmpbuf guard;      /* the running job's guard (mh_sec_trap longjmps here) */
    uint32_t guard_active, in_trap, job_gen, rsvd;
};
_Static_assert(sizeof(struct mh_priv) <= 256, "mh_priv must fit the 256 B private block");

#define CTL()   ((volatile struct mh_ctl *)plat_ptr(MH_CTL_ADDR))
#define STAT(h) ((volatile struct mh_stat *)plat_ptr(MH_STAT_ADDR(h)))
#define AUX(h)  ((volatile struct mh_aux *)plat_ptr(MH_AUX_ADDR(h)))
#define PRIV(h) ((struct mh_priv *)plat_ptr(MH_PRIV_ADDR(h)))

static uint64_t w_mtime(void) { return plat_mtime(); }
static uint64_t w_mcycle(void) { return plat_mcycle(); }
static void *w_ptr(uint64_t phys) { return plat_ptr(phys); }
static uint32_t w_beu(uint32_t h) { return h < MH_NHARTS ? plat_beu_accrued_hart(h) : 0u; }

const struct mh_wapi mh_wapi = {
    MH_WAPI_VERSION, sizeof(struct mh_wapi), PLAT_MTIME_HZ, w_mtime, w_mcycle, w_ptr, w_beu, 0,
};

/* The NOP job of MB_CMD_MH ping (inside the image: no fence.i needed). */
static uint64_t mh_nop(uint64_t arg, uint32_t h, const struct mh_wapi *w)
{
    (void)w;
    return arg ^ h;
}

static int stat_valid(uint32_t h, uint32_t epoch)
{
    volatile struct mh_stat *st = STAT(h);
    return st->magic == MH_STAT_MAGIC && st->hartid == h && st->epoch == epoch;
}

/* ---- hart 0 ------------------------------------------------------------------------------------------------ */

void mh_hart0_entry(int cold)
{
    volatile struct mh_ctl *c = CTL();
    if (!cold && c->magic == MH_CTL_MAGIC) return;   /* warm entry: harts 1-3 may be running a job, keep it all */
    c->magic = 0;
    MB_FENCE_REL();
    c->gen = 0;
    c->mask = 0;
    c->fn = 0;
    c->arg = 0;
    c->t_dispatch = 0;
    c->n_dispatch = 0;
    for (int i = 0; i < 5; i++) c->rsvd[i] = 0;
    c->epoch = (uint32_t)plat_mtime() | 1u;   /* a stat line from before this cold entry no longer counts */
    MB_FENCE_REL();
    c->magic = MH_CTL_MAGIC;
}

uint32_t mh_ready(void)
{
    volatile struct mh_ctl *c = CTL();
    uint32_t r = 1u;
    if (c->magic != MH_CTL_MAGIC) return r;
    uint32_t epoch = c->epoch;
    for (uint32_t h = 1; h < MH_NHARTS; h++)
        if (stat_valid(h, epoch) && STAT(h)->state == MH_ST_IDLE) r |= 1u << h;
    return r;
}

/* Seqlock write of one job, then the IPIs. Returns the job's generation. */
static uint32_t dispatch(uint32_t want, uint64_t fn, uint64_t arg)
{
    volatile struct mh_ctl *c = CTL();
    uint32_t g = c->gen & ~1u;
    c->gen = g + 1u;                 /* odd: the fields are being written */
    MB_FENCE_REL();
    c->mask = want;
    c->fn = fn;
    c->arg = arg;
    c->n_dispatch = c->n_dispatch + 1u;
    c->t_dispatch = plat_mtime();
    MB_FENCE_REL();
    c->gen = g + 2u;
    MB_FENCE_REL();                  /* gen (LIM) before the msip stores (CLINT) */
    for (uint32_t h = 1; h < MH_NHARTS; h++)
        if ((want >> h) & 1u) plat_msip_set(h, 1u);
    return g + 2u;
}

/* Wait until every hart of `want` finished job g (done or trapped) or `timeout` mtime ticks passed; returns the
 * harts that completed it. Hart 0's heartbeat keeps going (L1, its own line). */
static uint32_t wait_gen(uint32_t g, uint32_t want, uint64_t timeout)
{
    uint64_t t0 = plat_mtime(), hb = t0;
    uint32_t done = 0, over = 0;
    for (;;) {
        for (uint32_t h = 1; h < MH_NHARTS; h++) {
            uint32_t b = 1u << h;
            if (!(want & b) || (over & b)) continue;
            volatile struct mh_stat *st = STAT(h);
            if (st->done_gen == g) { done |= b; over |= b; }
            else if (st->trap_gen == g) over |= b;
        }
        if (over == want) break;
        uint64_t now = plat_mtime();
        if (now - t0 >= timeout) break;
        if (now - hb >= MH_HB_TICKS) { mon_heartbeat(); hb = now; }
        plat_mh_relax();
    }
    MB_FENCE_ACQ();                  /* done_gen before the job's results */
    return done;
}

uint32_t mh_run(uint32_t mask, uint64_t fn, uint64_t arg, uint64_t timeout_ticks)
{
    uint32_t want = mask & MH_SEC_MASK & mh_ready();
    uint32_t g = 0, done = 0;
    if (want) g = dispatch(want, fn, arg);
    if (mask & 1u) {                 /* hart 0's own share, on hart 0's stack, under the CALL's guard */
        ((mh_fn)(uintptr_t)fn)(arg, 0u, &mh_wapi);
        done |= 1u;
    }
    if (!timeout_ticks) return want | (mask & 1u);
    if (want) done |= wait_gen(g, want, timeout_ticks);
    return done;
}

uint32_t mh_wait(uint64_t timeout_ticks)
{
    volatile struct mh_ctl *c = CTL();
    uint32_t g = c->gen, want = c->mask & MH_SEC_MASK;
    if ((g & 1u) || !want || c->magic != MH_CTL_MAGIC) return 0;
    return wait_gen(g, want, timeout_ticks ? timeout_ticks : 1u);
}

uint32_t mh_ping(uint32_t mask, uint64_t timeout_ticks, struct mb_l39_mh *o)
{
    uint32_t ask = mask & MH_SEC_MASK;
    uint32_t want = ask & mh_ready();
    volatile struct mh_ctl *c = CTL();
    o->op = MB_MH_OP_PING;
    o->mask = ask;
    for (int i = 0; i < 3; i++) o->wake_ticks[i] = 0xFFFFFFFFu;
    uint64_t t0 = plat_mtime();
    uint32_t g = c->gen & ~1u, done = 0;
    if (want) {
        g = dispatch(want, (uint64_t)(uintptr_t)mh_nop, 0x9u);
        done = wait_gen(g, want, timeout_ticks);
    }
    uint64_t t1 = plat_mtime(), td = c->t_dispatch;
    uint32_t trapped = 0, epoch = c->epoch;
    for (uint32_t h = 1; h < MH_NHARTS; h++) {
        volatile struct mh_aux *ax = AUX(h);
        if ((want >> h) & 1u && ax->gen_seen == g) o->wake_ticks[h - 1] = (uint32_t)(ax->t_seen - td);
        if (stat_valid(h, epoch) && STAT(h)->traps) trapped |= 1u << h;
    }
    o->done = done;
    o->gen = g;
    o->total_ticks = (uint32_t)(t1 - t0);
    o->ready = mh_ready();
    o->trapped = trapped;
    return done;
}

/* ---- harts 1-3 ----------------------------------------------------------------------------------------------- */

static void idle_wait(uint32_t h, uint32_t mode)
{
    if (mode == MH_MODE_WFI) {
        plat_mtimecmp_set(h, plat_mtime() + MH_IDLE_TICKS);
        plat_wfi();
    } else {
        uint64_t t0 = plat_mtime();
        while (plat_mtime() - t0 < MH_POLL_TICKS) {}
    }
    plat_msip_set(h, 0u);
    MH_FENCE();                      /* the msip clear before the ctl.gen load: a new IPI is never lost */
}

static void run_job(uint32_t h, uint32_t g, uint64_t fn, uint64_t arg, uint64_t now)
{
    volatile struct mh_stat *st = STAT(h);
    volatile struct mh_aux *ax = AUX(h);
    struct mh_priv *pv = PRIV(h);
    ax->gen_seen = g;
    ax->t_seen = now;
    st->state = MH_ST_BUSY;
    if (!plat_in_image(fn)) {        /* a LOADed payload: the monitor's FENCE_I fenced hart 0 only (Caches.md:22) */
        plat_fence_i();
        ax->fence_i = ax->fence_i + 1u;
    }
    pv->job_gen = g;
    if (mon_setjmp(pv->guard) == 0) {
        pv->guard_active = 1u;
        uint64_t r = ((mh_fn)(uintptr_t)fn)(arg, h, &mh_wapi);
        pv->guard_active = 0u;
        ax->result = r;
        MB_FENCE_REL();              /* the job's results before done_gen */
        st->done_gen = g;
    }                                /* else: mh_sec_trap() dumped the trap into aux and wrote trap_gen */
    st->state = MH_ST_IDLE;
}

__attribute__((noreturn)) static void idle_loop(uint32_t h, uint32_t last)
{
    volatile struct mh_ctl *c = CTL();
    volatile struct mh_stat *st = STAT(h);
    volatile struct mh_aux *ax = AUX(h);
    uint32_t mode = st->mode;
    for (;;) {
        idle_wait(h, mode);
        uint64_t now = plat_mtime();
        st->heartbeat = now;
        ax->wakes = ax->wakes + 1u;
        if (c->gen == last) continue;
        uint32_t g, mask;
        uint64_t fn, arg;
        for (;;) {                   /* seqlock read of the job fields */
            g = c->gen;
            if (g & 1u) continue;
            MB_FENCE_ACQ();
            mask = c->mask;
            fn = c->fn;
            arg = c->arg;
            MB_FENCE_ACQ();
            if (c->gen == g) break;
        }
        if (g == last) continue;
        last = g;
        if ((mask >> h) & 1u) run_job(h, g, fn, arg, now);
    }
}

void mh_secondary_main(uint32_t h)
{
    volatile struct mh_ctl *c = CTL();
    volatile struct mh_stat *st = STAT(h);
    volatile struct mh_aux *ax = AUX(h);
    struct mh_priv *pv = PRIV(h);
    uint32_t epoch = c->epoch;
    int same = stat_valid(h, epoch);
    uint32_t entries = same ? st->entries + 1u : 1u, traps = same ? st->traps : 0u;
    pv->guard_active = 0;
    pv->in_trap = 0;
    pv->job_gen = 0;
    st->state = MH_ST_STARTING;
    st->hartid = h;
    st->epoch = epoch;
    st->entries = entries;
    st->traps = traps;
    st->mncause = plat_mncause();
    st->mnepc = plat_mnepc();
    st->heartbeat = plat_mtime();
    MB_FENCE_REL();
    st->magic = MH_STAT_MAGIC;
    /* adoption self-test: an IPI to itself must show in mip and end a wfi long before the mtimecmp backstop */
    plat_msip_set(h, 1u);
    MH_FENCE();
    uint64_t mip = plat_mip();
    plat_mtimecmp_set(h, plat_mtime() + MH_IDLE_TICKS);
    uint64_t t0 = plat_mtime();
    plat_wfi();
    uint64_t t1 = plat_mtime();
    plat_msip_set(h, 0u);
    MH_FENCE();
    uint32_t mode = ((mip & MIP_MSIP) && t1 - t0 < MH_IDLE_TICKS / 2u) ? MH_MODE_WFI : MH_MODE_POLL;
    if (!same) {
        ax->mcause = 0;
        ax->mepc = 0;
        ax->mtval = 0;
        ax->result = 0;
        ax->fence_i = 0;
    }
    ax->t_seen = 0;
    ax->gen_seen = 0;
    ax->wakes = 0;
    ax->beu_accrued = plat_beu_accrued_hart(h);
    ax->wfi_ticks = (uint32_t)(t1 - t0);
    ax->rsvd = 0;
    st->mode = mode;
    st->mip = (uint32_t)mip;
    uint32_t g = c->gen & ~1u;       /* a job dispatched before this adoption is not this hart's */
    st->done_gen = g;
    st->trap_gen = g;
    MB_FENCE_REL();
    st->state = MH_ST_IDLE;
    idle_loop(h, g);
}

void mh_secondary_resume(uint32_t h)
{
    uint32_t g = CTL()->gen & ~1u;
    STAT(h)->state = MH_ST_IDLE;
    idle_loop(h, g);
}

void mh_sec_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval, uint64_t hart)
{
    uint32_t h = (uint32_t)hart;
    if (h == 0 || h >= MH_NHARTS) for (;;) plat_wfi();
    struct mh_priv *pv = PRIV(h);
    volatile struct mh_stat *st = STAT(h);
    volatile struct mh_aux *ax = AUX(h);
    if (pv->in_trap) for (;;) plat_wfi();   /* a trap in the trap path: wait for the host's RNMI (re-adoption) */
    pv->in_trap = 1u;
    ax->mcause = mcause;
    ax->mepc = mepc;
    ax->mtval = mtval;
    st->traps = st->traps + 1u;
    if (pv->guard_active) {
        pv->guard_active = 0u;
        MB_FENCE_REL();
        st->trap_gen = pv->job_gen;
        pv->in_trap = 0u;
        mon_longjmp(pv->guard, 1);
    }
    st->state = MH_ST_TRAPPED;
    MB_FENCE_REL();
    pv->in_trap = 0u;                /* the caller restarts the idle loop on a fresh stack */
}
