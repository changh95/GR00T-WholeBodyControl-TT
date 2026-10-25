/* SPIKE_HOST: the scripted host, HTIF I/O and the RNMI emulation of the spike build (SPIKE_HOST.md).
 *
 * Only compiled with -DSPIKE_HOST. The monitor code is the X280 code; what changes is (1) the idle loop
 * is driven by sh_run() instead of running forever, (2) the X280-only RNMI CSRs 0x350-0x353 and 0x7c1,
 * which trap in spike (x280/sim/selftest/t_csr.c), are memory variables here, (3) the host's RNMI is
 * emulated: PARK jumps straight into the park path, a hung CALL is interrupted by a machine timer
 * interrupt (CLINT mtimecmp, plat.h) and then takes the same path. "The park path" is the real park
 * stub in the GP scratch when the harness maps it (SPIKE_X280_REGIONS), else crt0 directly.
 *
 * State that must survive an emulated warm entry (crt0 clears .bss) lives in .data.
 * Harts 1-3 (spike -p4): each has its own emulated mnstatus/mncause/mnepc (sh_emu[h], one 64 B line per hart, read by
 * crt0's secondary path); with the X280 regions a cold secondary takes SH_MH_ADOPT's emulated RNMI itself (crt0.S).
 */
#include "monitor.h"
#include "spike_host.h"
#include "asm.h"

/* ---- HTIF (riscv-isa-sim fesvr/htif.cc, fesvr/syscall.cc:209-221, 463-475) ------------------------ */
volatile uint64_t tohost __attribute__((section(".htif"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".htif"), aligned(8)));
static volatile uint64_t sh_magic[8] __attribute__((aligned(64)));

#define SYS_openat 56
#define SYS_close  57
#define SYS_read   63
#define SYS_write  64
#define AT_FDCWD   (-100)
#define O_RDONLY   0
#define O_WRONLY_CREAT_TRUNC (01 | 0100 | 01000)

static int64_t sh_syscall(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4)
{
    sh_magic[0] = n; sh_magic[1] = a0; sh_magic[2] = a1; sh_magic[3] = a2;
    sh_magic[4] = a3; sh_magic[5] = a4; sh_magic[6] = 0; sh_magic[7] = 0;
    __asm__ volatile("fence rw, rw" ::: "memory");
    while (tohost) {}
    tohost = (uint64_t)(uintptr_t)sh_magic;      /* even value, device 0: proxied syscall */
    while (!fromhost) {}
    fromhost = 0;
    __asm__ volatile("fence rw, rw" ::: "memory");
    return (int64_t)sh_magic[0];
}

__attribute__((noreturn)) static void sh_exit(uint32_t code)
{
    while (tohost) {}
    tohost = ((uint64_t)code << 1) | 1u;
    for (;;) {}
}

static uint32_t sh_strlen(const char *s) { uint32_t n = 0; while (s[n]) n++; return n; }

/* ---- output ---------------------------------------------------------------------------------------- */
static char sh_buf[1024];
static uint32_t sh_n;

static void out_s(const char *s) { while (*s && sh_n < sizeof sh_buf - 1) sh_buf[sh_n++] = *s++; }
static void out_u(uint64_t v)
{
    char t[24];
    int n = 0;
    do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n && sh_n < sizeof sh_buf - 1) sh_buf[sh_n++] = t[--n];
}
static void out_addr(const char *key, uint64_t addr)
{
    static const char hx[] = "0123456789abcdef";
    const volatile uint8_t *p = plat_ptr(addr);
    out_s(" ");
    out_s(key);
    out_s("=");
    for (uint32_t i = 0; i < MB_LINE && sh_n < sizeof sh_buf - 3; i++) {
        uint8_t b = p[i];
        sh_buf[sh_n++] = hx[b >> 4];
        sh_buf[sh_n++] = hx[b & 15];
    }
}
static void out_line(const char *key, uint32_t line) { out_addr(key, MB_L(line)); }
static void flush(void)
{
    sh_buf[sh_n++] = '\n';
    sh_syscall(SYS_write, 1, (uint64_t)(uintptr_t)sh_buf, sh_n, 0, 0);
    sh_n = 0;
}

/* ---- emulated X280 RNMI state (plat.h, SPIKE_HOST) ------------------------------------------------ */
/* One line per hart, each written only by its hart (crt0.S indexes it as sh_emu + 64 h: mnstatus, mncause, mnepc). */
struct sh_emu_hart { uint64_t mnstatus, mncause, mnepc, pad[5]; };
_Static_assert(sizeof(struct sh_emu_hart) == 64, "crt0.S: sh_emu + 64 h");
struct sh_emu_hart sh_emu[MH_NHARTS] __attribute__((section(".data"), aligned(64))) = {
    { 8u, 0, 0, {0} }, { 8u, 0, 0, {0} }, { 8u, 0, 0, {0} }, { 8u, 0, 0, {0} },
};
#define EMU (&sh_emu[plat_mhartid() & (MH_NHARTS - 1u)])

uint64_t plat_mnstatus(void) { return EMU->mnstatus; }
uint64_t plat_mncause(void) { return EMU->mncause; }
uint64_t plat_mnepc(void) { return EMU->mnepc; }
void plat_rnmi_rearm(void) { EMU->mnstatus |= 8u; }   /* stands in for csrsi 0x353, 8 */

void plat_beu_read(uint32_t *cause, uint64_t *value, uint32_t *accrued)
{
#if defined(SPIKE_X280_REGIONS)
    *cause = *(volatile uint32_t *)(PLAT_BEU0 + PLAT_BEU_CAUSE);
    *value = *(volatile uint64_t *)(PLAT_BEU0 + PLAT_BEU_VALUE);
    *accrued = *(volatile uint32_t *)(PLAT_BEU0 + PLAT_BEU_ACCRUED);
#else
    *cause = 0; *value = 0; *accrued = 0;
#endif
}

static void timer_off(void)
{
    __asm__ volatile("csrc mstatus, %0" :: "r"(MSTATUS_MIE) : "memory");
    __asm__ volatile("csrc mie, %0" :: "r"(MIE_MTIE) : "memory");
    *(volatile uint64_t *)PLAT_MTIMECMP0 = ~0ull;
}

static void timer_arm(uint32_t delay_us)
{
    *(volatile uint64_t *)PLAT_MTIMECMP0 = plat_mtime() + (uint64_t)delay_us * (PLAT_MTIME_HZ / 1000000u);
    __asm__ volatile("csrs mie, %0" :: "r"(MIE_MTIE) : "memory");
    __asm__ volatile("csrs mstatus, %0" :: "r"(MSTATUS_MIE) : "memory");
}

/* The host's RNMI: taken only when (emulated) NMIE is set, as on the X280 (RNMIs.md:18-22). */
void sh_emulated_rnmi(uint64_t mncause, uint64_t mnepc)
{
    timer_off();
    if (!(sh_emu[0].mnstatus & 8u)) {
        out_s("SH rnmi_lost: NMIE was not re-armed; on the X280 this hart would stay hung");
        flush();
        sh_exit(0x7F);
    }
    sh_emu[0].mnstatus &= ~8ull;
    sh_emu[0].mncause = mncause;
    sh_emu[0].mnepc = mnepc;
#if defined(SPIKE_X280_REGIONS)
    uint64_t target = PLAT_GP_SCRATCH;   /* the real park stub (installed at boot) */
#else
    uint64_t target = MB_LIM_BASE;       /* crt0, i.e. the reset handler address */
#endif
    __asm__ volatile("jr %0" :: "r"(target) : "memory");
    __builtin_unreachable();
}

void plat_park_forever(void)
{
    uint64_t pc;
    __asm__ volatile("auipc %0, 0" : "=r"(pc));
    sh_emulated_rnmi((1ull << 63) | 2u, pc);
}

/* ---- runner ------------------------------------------------------------------------------------------ */
enum { PH_IDLE = 0, PH_WAIT = 1 };
/* t_issue/cyc_last/cyc_sum: mcycle from the host's seq_req write to the poll that sees seq_done, i.e. the
 * hart-side cost of one command (monitor + kernel; spike retires 1 instruction per cycle, so these are
 * instruction counts there). cyc_sum adds up the commands of the current step (all records of an
 * INFER_FILE). */
static struct {
    uint32_t step, sub, phase, seq, timeouts, entries, booted, nsteps;
    uint64_t t_issue, cyc_last, cyc_sum;
} st __attribute__((section(".data"))) = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static uint32_t entered_this_boot;          /* .bss: 0 on every entry, 1 after the first sh_run */
static uint8_t obs_stage[MB_OBS_LINES * MB_LINE] __attribute__((aligned(64)));

#define SH_MAX_POLLS 400000000u

static volatile struct mb_l1 *L1(void) { return MB_PTR(struct mb_l1, MB_L1_STATUS); }

static void host_line_write(uint64_t addr, const void *src)   /* one full 64 B line, 8 x sd */
{
    volatile uint64_t *d = plat_ptr(addr);
    const uint64_t *s = src;
    for (int i = 0; i < 8; i++) d[i] = s[i];
}

static void host_write(uint64_t addr, const uint8_t *p, uint32_t n)   /* full lines, zero padded */
{
    uint64_t line[8];
    for (uint32_t off = 0; off < n; off += MB_LINE) {
        uint8_t *lb = (uint8_t *)line;
        for (uint32_t i = 0; i < MB_LINE; i++) lb[i] = off + i < n ? p[off + i] : 0;
        host_line_write(addr + off, line);
    }
}

static void report(const struct sh_step *s, const char *res)
{
    out_s("SH step=");
    out_u(st.step);
    out_s(" op=");
    out_s(s->name ? s->name : "?");
    out_s(" sub=");
    out_u(st.sub);
    out_s(" seq=");
    out_u(st.seq);
    out_s(" res=");
    out_s(res);
    out_s(" cyc=");
    out_u(st.cyc_last);
    out_s(" cyc_sum=");
    out_u(st.cyc_sum);
    out_line("L1", MB_L1_STATUS);
    out_line("L35", MB_L35_ACT);
    out_line("L36", MB_L36_IDENT);
    out_line("L39", MB_L39_SCRATCH);
    flush();
}

/* Extra lines of a step (`report` bits): the multi-hart lines (mh.h), all raw. */
static void report_extra(const struct sh_step *s)
{
    if (s->report & SH_REP_MH) {
        static const char *const sk[] = { 0, "S1", "S2", "S3" }, *const ak[] = { 0, "A1", "A2", "A3" };
        out_s("SH mh step=");
        out_u(st.step);
        out_s(" sub=");
        out_u(st.sub);
        out_addr("C", MH_CTL_ADDR);
        for (uint32_t h = 1; h < MH_NHARTS; h++) { out_addr(sk[h], MH_STAT_ADDR(h)); out_addr(ak[h], MH_AUX_ADDR(h)); }
        flush();
    }
    if (s->report & SH_REP_MHB) {
        static const char *const rk[] = { "R0", "R1", "R2", "R3" };
        out_s("SH mhb step=");
        out_u(st.step);
        out_s(" sub=");
        out_u(st.sub);
        out_addr("J", MHB_DATA);
        for (uint32_t h = 0; h < MH_NHARTS; h++) out_addr(rk[h], MHB_RES(h));
        flush();
    }
}

/* §1.4.4 host steps 1-2: obs lines (if any), L0 bytes 0-59 as one line, then seq_req alone. */
static uint32_t issue(const struct sh_step *s, uint32_t cmd, int with_obs, uint32_t obs_crc_delta)
{
    struct mb_l0 c;
    uint32_t *cw = (uint32_t *)&c;
    for (int i = 0; i < 16; i++) cw[i] = 0;
    c.magic = MB_MAGIC;
    c.cmd = cmd;
    c.policy = s->policy;
    c.flags = s->flags;
    c.tick = s->tick;
    c.arg_addr = s->arg_addr;
    c.arg_entry = s->arg_entry;
    c.arg_len = s->arg_len;
    if (with_obs) {
        host_write(MB_L(MB_L2_OBS), obs_stage, sizeof obs_stage);
        c.obs_crc32 = mon_crc32(0, obs_stage, sizeof obs_stage) + obs_crc_delta;
    }
    c.crc32 = mon_crc32(0, &c, MB_L0_CRC_BYTES);
    volatile struct mb_l0 *l0 = MB_PTR(struct mb_l0, MB_L0_CMD);
    c.seq_req = l0->seq_req;
    host_line_write(MB_L(MB_L0_CMD), &c);
    __asm__ volatile("fence w, w" ::: "memory");
    st.seq = L1()->seq_done + 1u;
    st.phase = PH_WAIT;
    st.cyc_last = 0;
    l0->seq_req = st.seq;
    st.t_issue = plat_mcycle();
    return st.seq;
}

static const char *wait_done(void)
{
    for (uint32_t i = 0; i < SH_MAX_POLLS; i++) {
        mon_poll_once();
        if (L1()->seq_done == st.seq && g_rt.state != MB_STATE_BUSY) {
            st.cyc_last = plat_mcycle() - st.t_issue;
            st.cyc_sum += st.cyc_last;
            st.phase = PH_IDLE;
            return "done";
        }
    }
    st.phase = PH_IDLE;
    st.timeouts++;
    return "timeout";
}

static void load_desc(uint64_t dst, uint32_t len, uint32_t kind)
{
    struct mb_l37 d;
    uint32_t *w = (uint32_t *)&d;
    for (int i = 0; i < 16; i++) w[i] = 0;
    d.dst = dst;
    d.len = len;
    d.crc32 = mon_crc32(0, plat_ptr(dst), len);
    d.kind = kind;
    d.load_seq = L1()->seq_done + 1u;   /* the seq issue() is about to use */
    d.desc_crc32 = mon_crc32(0, &d, MB_L37_CRC_BYTES);
    host_line_write(MB_L(MB_L37_LOAD), &d);
}

static int64_t sh_open(const char *path, int flags)
{
    return sh_syscall(SYS_openat, (uint64_t)(int64_t)AT_FDCWD, (uint64_t)(uintptr_t)path,
                      sh_strlen(path) + 1u, (uint64_t)flags, 0644);
}

static uint32_t read_full(int64_t fd, uint8_t *p, uint32_t n)
{
    uint32_t got = 0;
    while (got < n) {
        int64_t r = sh_syscall(SYS_read, (uint64_t)fd, (uint64_t)(uintptr_t)(p + got), n - got, 0, 0);
        if (r <= 0) break;
        got += (uint32_t)r;
    }
    return got;
}

/* Deterministic obs for SH_INFER_CONST: xorshift32 from (seed << 1) | 1, obs[i] = (x >> 8) * 2^-23 - 1. */
static void const_obs(uint32_t seed)
{
    uint32_t x = (seed << 1) | 1u;
    float *o = (float *)obs_stage;
    for (int i = 0; i < 516; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        o[i] = (float)(x >> 8) * (1.0f / 8388608.0f) - 1.0f;
    }
    for (uint32_t i = 516 * 4; i < sizeof obs_stage; i++) obs_stage[i] = 0;
}

static void run_step(const struct sh_step *s)
{
    uint32_t count = s->count ? s->count : 1u;
    switch (s->op) {
    case SH_CMD:
        for (; st.sub < count; st.sub++) {
            issue(s, s->cmd, s->cmd == MB_CMD_INFER || s->cmd == MB_CMD_BENCH, 0);
            report(s, wait_done());
            report_extra(s);
        }
        break;
    case SH_MH_ADOPT: {
#if defined(SPIKE_X280_REGIONS)
        uint32_t mask = (s->flags ? s->flags : MH_SEC_MASK) & MH_SEC_MASK, n = 0, pre;
        uint64_t hz_us = PLAT_MTIME_HZ / 1000000u;
        for (uint32_t h = 1; h < MH_NHARTS; h++) {   /* what x280d serve --harts 4 writes before the release */
            *(volatile uint64_t *)(PLAT_RNMI_HANDLER + 16u * h) = MB_LIM_BASE;
            *(volatile uint64_t *)(PLAT_RNMI_HANDLER + 16u * h + 8u) = MH_SEC_RNMI_EXC;
        }
        *(volatile uint32_t *)PLAT_RNMI_TRIGGER = mask;
        __asm__ volatile("fence iorw, iorw" ::: "memory");
        for (uint32_t h = 1; h < MH_NHARTS; h++)
            if ((mask >> h) & 1u) plat_msip_set(h, 1u);
        uint64_t t0 = plat_mtime();
        while (plat_mtime() - t0 < (uint64_t)(s->delay_us ? s->delay_us : 100u) * hz_us) {}
        pre = mh_ready();                              /* still spinning on their trigger bits: nobody ready */
        *(volatile uint32_t *)PLAT_RNMI_TRIGGER = 0;
        t0 = plat_mtime();
        while ((mh_ready() & mask) != mask && plat_mtime() - t0 < 5000u * hz_us) n++;
        uint32_t ready = mh_ready();
        report(s, (ready & mask) == mask ? "done" : "timeout");
        if ((ready & mask) != mask) st.timeouts++;
        out_s("SH mhadopt step=");
        out_u(st.step);
        out_s(" mask=");
        out_u(mask);
        out_s(" pre_ready=");
        out_u(pre);
        out_s(" ready=");
        out_u(ready);
        out_s(" polls=");
        out_u(n);
        flush();
        report_extra(s);
#else
        report(s, "bad_op");
        st.timeouts++;
#endif
        break;
    }
    case SH_LOAD_PATTERN: {
        uint64_t x = (uint64_t)s->tick | 1u;
        volatile uint64_t *d = plat_ptr(s->dst);
        for (uint32_t i = 0; i < s->arg_len / 8u; i++) {   /* full 64 B lines: arg_len is a multiple of 64 */
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            d[i] = x;
        }
        load_desc(s->dst, s->arg_len, MB_LOAD_DATA);
        issue(s, MB_CMD_LOAD, 0, 0);
        report(s, wait_done());
        break;
    }
    case SH_MHB_SWEEP: {
        const struct sh_sweep *w = s->sweep;
        uint32_t nkb = w->nkb ? w->nkb : 1u, total = (uint32_t)w->nk * w->np * nkb * w->nm;
        for (; st.sub < total; st.sub++) {
            uint32_t i = st.sub, mi = i % w->nm, kbi, pi, ki;
            i /= w->nm;
            kbi = i % nkb;
            i /= nkb;
            pi = i % w->np;
            ki = i / w->np;
            struct sh_step c = *s;
            c.policy = w->kernels[ki] | ((uint32_t)w->k0 << 8);
            c.flags = w->masks[mi] | ((uint32_t)w->patterns[pi] << 4) | ((uint32_t)(w->nkb ? w->kbits[kbi] : 0u) << 8) |
                      ((uint32_t)w->mode << 12);
            c.arg_entry = MHB_BASE;
            issue(&c, MB_CMD_CALL, 0, 0);
            report(s, wait_done());
            report_extra(s);
        }
        break;
    }
    case SH_LOAD_BYTES: {
        uint32_t n = (uint32_t)(s->bytes_end - s->bytes);
        host_write(s->dst, s->bytes, n);
        load_desc(s->dst, n, s->kind);
        issue(s, MB_CMD_LOAD, 0, 0);
        report(s, wait_done());
        break;
    }
    case SH_LOAD_FILE: {
        int64_t fd = sh_open(s->path, O_RDONLY);
        if (fd < 0) { report(s, "open_failed"); st.timeouts++; break; }
        uint8_t *dst = plat_ptr(s->dst);
        uint32_t n = read_full(fd, dst, (uint32_t)(MB_LIM_END - s->dst));
        sh_syscall(SYS_close, (uint64_t)fd, 0, 0, 0, 0);
        for (uint32_t i = n; i & (MB_LINE - 1); i++) dst[i] = 0;
        load_desc(s->dst, n, s->kind);
        issue(s, MB_CMD_LOAD, 0, 0);
        report(s, wait_done());
        break;
    }
    case SH_INFER_CONST:
        for (; st.sub < count; st.sub++) {
            const_obs((uint32_t)s->tick + st.sub);
            issue(s, MB_CMD_INFER, 1, 0);
            report(s, wait_done());
        }
        break;
    case SH_CORRUPT_INFER:
        const_obs(12345u);
        issue(s, MB_CMD_INFER, 1, 1u);
        report(s, wait_done());
        break;
    case SH_INFER_FILE: {
        int64_t in = sh_open(s->path, O_RDONLY);
        int64_t out = sh_open(s->path2, O_WRONLY_CREAT_TRUNC);
        if (in < 0 || out < 0) { report(s, "open_failed"); st.timeouts++; break; }
        const char *res = "done";
        uint32_t i = 0, max = s->count ? s->count : 0xFFFFFFFFu;   /* count 0 = every record */
        for (; i < max; i++) {
            if (read_full(in, obs_stage, 516 * 4) != 516 * 4) break;
            for (uint32_t k = 516 * 4; k < sizeof obs_stage; k++) obs_stage[k] = 0;
            st.sub = i;
            issue(s, MB_CMD_INFER, 1, 0);
            const char *r = wait_done();
            if (r[0] != 'd') res = r;    /* one record that never completed marks the whole step */
            sh_syscall(SYS_write, (uint64_t)out, (uint64_t)(uintptr_t)plat_ptr(MB_L(MB_L35_ACT)), 64, 0, 0);
            sh_syscall(SYS_write, (uint64_t)out, (uint64_t)(uintptr_t)plat_ptr(MB_L(MB_L1_STATUS)), 64, 0, 0);
        }
        sh_syscall(SYS_close, (uint64_t)in, 0, 0, 0, 0);
        sh_syscall(SYS_close, (uint64_t)out, 0, 0, 0, 0);
        st.sub = i;
        report(s, res);
        break;
    }
    case SH_POKE32:
        *(volatile uint32_t *)plat_ptr(s->dst) = s->arg_len;
        for (int i = 0; i < 100000; i++) mon_poll_once();   /* let the idle loop notice */
        report(s, "done");
        break;
    case SH_PARK:
        issue(s, MB_CMD_PARK, 0, 0);
        report(s, wait_done());      /* not reached: PARK re-enters through crt0 */
        break;
    case SH_CALL_HANG:
        for (; st.sub < count; st.sub++) {
            timer_arm(s->delay_us ? s->delay_us : 200u);
            issue(s, MB_CMD_CALL, 0, 0);
            const char *r = wait_done();   /* a hang re-enters through crt0 instead of returning */
            timer_off();
            report(s, r);
        }
        break;
    default:
        report(s, "bad_op");
        st.timeouts++;
        break;
    }
}

#if defined(SPIKE_X280_REGIONS)
/* What x280d does at boot steps 4-5 (§5.2), into the regions the harness maps as RAM. */
static void install_park_stub(void)
{
    host_write(PLAT_GP_SCRATCH, sh_bin_park, (uint32_t)(sh_bin_park_end - sh_bin_park));
#if defined(SPIKE_E1F_VECTORS)
    *(volatile uint32_t *)PLAT_RESETVEC = (uint32_t)PLAT_GP_SCRATCH;   /* what an E1F boot leaves (review F1) */
#else
    *(volatile uint32_t *)PLAT_RESETVEC = MB_LIM_BASE;   /* hart 0 reset handler address, lo */
#endif
    *(volatile uint32_t *)(PLAT_RESETVEC + 4) = 0;
    *(volatile uint32_t *)PLAT_RNMI_TRIGGER = 0;
}
#endif

/* mon_crc32 as built for the hart, against zlib (test_spike.py recomputes every value from monitor_spike.bin): for
 * each start alignment 0..7 over the image's own first bytes, an FNV-1a digest of the CRCs of every length
 * 0..SH_CRC_NMAX plus 2,112 (the obs) and 4,096, and one 4,096 B CRC chained over pieces of 1, 2, .., 13 bytes. */
#define SH_CRC_NMAX 520u
static uint32_t fnv32(uint32_t h, uint32_t v)
{
    for (int i = 0; i < 4; i++) { h ^= (v >> (8 * i)) & 0xFFu; h *= 16777619u; }
    return h;
}

static void crc_report(void)
{
    const uint8_t *src = plat_ptr(MB_LIM_BASE);
    for (uint32_t off = 0; off < 8; off++) {
        uint32_t h = 2166136261u;
        for (uint32_t n = 0; n <= SH_CRC_NMAX; n++) h = fnv32(h, mon_crc32(0, src + off, n));
        h = fnv32(h, mon_crc32(0, src + off, 2112u));
        h = fnv32(h, mon_crc32(0, src + off, 4096u));
        uint32_t c = 0, pos = 0, k = 1;
        while (pos < 4096u) {
            uint32_t len = 4096u - pos < k ? 4096u - pos : k;
            c = mon_crc32(c, src + off + pos, len);
            pos += len;
            k = k % 13u + 1u;
        }
        out_s("SH crc off=");
        out_u(off);
        out_s(" nmax=");
        out_u(SH_CRC_NMAX);
        out_s(" slices=");
        out_u(MON_CRC_SLICES);
        out_s(" digest=");
        out_u(h);
        out_s(" chained=");
        out_u(c);
        flush();
    }
}

/* After an interrupted wait: the next repetition of a counted step, else the next step. */
static void advance(void)
{
    const struct sh_step *s = &sh_script[st.step];
    uint32_t count = s->count ? s->count : 1u;
    st.phase = PH_IDLE;
    if ((s->op == SH_CALL_HANG || s->op == SH_CMD || s->op == SH_INFER_CONST || s->op == SH_MHB_SWEEP) &&
        st.sub + 1 < count) {
        st.sub++;
    } else {
        st.step++;
        st.sub = 0;
    }
}

void sh_run(void)
{
    if (!entered_this_boot) {
        entered_this_boot = 1;
        st.entries++;
        if (!st.booted) {
            st.booted = 1;
#if defined(SPIKE_X280_REGIONS)
            install_park_stub();
#endif
            out_s("SH boot");
            out_line("L1", MB_L1_STATUS);
            out_line("L36", MB_L36_IDENT);
            out_line("L39", MB_L39_SCRATCH);
            flush();
            crc_report();
        } else if (st.phase == PH_WAIT) {
            /* warm entry while waiting: the step's command was parked (PARK or an RNMI on a hang) */
            report(&sh_script[st.step], "reentered");
            advance();
        } else {
            out_s("SH unexpected_entry");
            out_line("L1", MB_L1_STATUS);
            flush();
        }
    } else if (st.phase == PH_WAIT) {
        /* restart after an unguarded trap: the monitor fails the seq in flight; collect that */
        report(&sh_script[st.step], wait_done());
        advance();
    }
    for (; sh_script[st.step].op != SH_END; st.step++, st.sub = 0) {
        st.cyc_sum = 0;
        run_step(&sh_script[st.step]);
    }
    out_s("SH end steps=");
    out_u(st.step);
    out_s(" timeouts=");
    out_u(st.timeouts);
    out_s(" entries=");
    out_u(st.entries);
    flush();
    sh_exit(st.timeouts);
}
