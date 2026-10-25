/* Platform layer of the X280 monitor: the only place that knows whether the code runs on the X280
 * (default riscv build), under spike (SPIKE_HOST) or on x86 against a fake LIM buffer (MON_X86).
 *
 * Why: monitor.c and trap.c must be one source for all three builds, so the x86 protocol tests and the
 * spike runs exercise the code that is loaded into LIM. Everything hardware-specific is here, with the
 * primary source of each address.
 */
#ifndef X280_PLAT_H
#define X280_PLAT_H

#include <stdint.h>
#include "mailbox.h"

#if !defined(MON_X86) && !defined(__riscv)
#error "build for riscv64 (X280 / SPIKE_HOST) or define MON_X86"
#endif

/* ---- addresses (x280 physical) -------------------------------------------------------------- */
/* CLINT at 0x0200_0000 (tt-isa-documentation BlackholeA0/L2CPUTile/MemoryMap.md:74; tt-bh-linux
 * misc/blackhole-p100-2.dts:129-131 "sifive,clint0"). mtime at +0xBFF8 and hart-0 mtimecmp at +0x4000 are
 * the SiFive clint0 offsets (fw-19.12.0 dts/vendor/tenstorrent/tt_grendel_smc.dtsi:117-118, 250-253:
 * clint@c8000000 with mtimer reg 0xc800bff8 / 0xc8004000; freedom-e-sdk@4518a22c
 * bsp/spike/metal-platform.h:18 METAL_RISCV_CLINT0_MTIME 49144). spike's CLINT uses the same map. */
#define PLAT_CLINT_BASE      0x02000000ul
#define PLAT_MTIME           (PLAT_CLINT_BASE + 0xBFF8ul)
#define PLAT_MTIMECMP0       (PLAT_CLINT_BASE + 0x4000ul)          /* SPIKE_HOST hart-0 runner only */
/* Per-hart CLINT registers, same clint0 layout (msip 4 B per hart at +0, mtimecmp 8 B per hart at +0x4000; riscv-isa-sim
 * riscv/clint.cc:31-33 for spike): used only by mh.c (the secondaries' wfi wake and hart 0's IPIs). */
#define PLAT_MSIP(h)         (PLAT_CLINT_BASE + 4ul * (h))
#define PLAT_MTIMECMP(h)     (PLAT_CLINT_BASE + 0x4000ul + 8ul * (h))
/* Hart-0 bus-error unit page (MemoryMap.md:69). Register offsets are SiFive buserror0's: cause +0,
 * value +8, enable +16, platform_interrupt +24, accrued +32, local_interrupt +40 (freedom-e-sdk@4518a22c
 * bsp/sifive-hifive-unmatched/metal-platform.h:88-93); read only, never written. */
#define PLAT_BEU0            0x01700000ul
#define PLAT_BEU(h)          (PLAT_BEU0 + 0x1000ul * (h))           /* hart h's page (MemoryMap.md:69-72) */
#define PLAT_BEU_CAUSE       0x00ul
#define PLAT_BEU_VALUE       0x08ul
#define PLAT_BEU_ACCRUED     0x20ul
/* External peripherals as seen by the hart (MemoryMap.md:111-119): reset handler addresses (8 B per
 * hart), 64 B GP scratch (park stub), RNMI trigger bits. Used by the SPIKE_HOST park emulation only. */
#define PLAT_RESETVEC        0x20010000ul
#define PLAT_GP_SCRATCH      0x20010100ul
#define PLAT_RNMI_TRIGGER    0x20010414ul
#define PLAT_RNMI_HANDLER    0x20010418ul                          /* hart h: + 16 h, exception handler + 8 */
/* Zero Device, cached (MemoryMap.md:86; Caches.md:9, 75): only the optional lim_bench working set. */
#define PLAT_ZERO_DEVICE     0x0A000000ul

/* mtime rate: 50 MHz on the X280 (tt-bh-linux misc/blackhole-p100-2.dts:14 timebase-frequency =
 * 0x2faf080); spike's CLINT ticks once per 100 instructions of a 1 GHz model = 10 MHz
 * (riscv-isa-sim riscv/sim.h:79-80). */
#if defined(SPIKE_HOST)
#define PLAT_MTIME_HZ        10000000u
#else
#define PLAT_MTIME_HZ        50000000u
#endif

/* misa bit for V ('V' - 'A' = 21). */
#define PLAT_MISA_V          (1ul << 21)

/* ---- setjmp for trap recovery ------------------------------------------------------------------ */
#if defined(MON_X86)
#include <setjmp.h>
typedef jmp_buf mon_jmpbuf;
#define mon_setjmp(b)     setjmp(b)
#define mon_longjmp(b, v) longjmp(b, v)
#else
typedef uint64_t mon_jmpbuf[26];   /* ra, sp, s0-s11, fs0-fs11 (crt0.S) */
int mon_setjmp(mon_jmpbuf b) __attribute__((returns_twice));
__attribute__((noreturn)) void mon_longjmp(mon_jmpbuf b, int v);
#endif

/* ---- primitives -------------------------------------------------------------------------------- */
#if defined(MON_X86)

extern uint8_t *g_x86_lim;            /* fake LIM, MB_LIM_SIZE bytes (plat_x86.c) */
uint64_t plat_mtime(void);
uint64_t plat_mcycle(void);
uint64_t plat_misa(void);
uint32_t plat_vlenb(void);
uint32_t plat_mhartid(void);
uint64_t plat_mnstatus(void);
uint64_t plat_mncause(void);
uint64_t plat_mnepc(void);
void     plat_rnmi_rearm(void);
void     plat_fence_i(void);
void     plat_beu_read(uint32_t *cause, uint64_t *value, uint32_t *accrued);
__attribute__((noreturn)) void plat_park_forever(void);
__attribute__((noreturn)) void plat_restart_after_trap(void);
uint64_t plat_call(uint64_t entry, const struct mb_call_args *a, const struct mon_api *api);
static inline void *plat_ptr(uint64_t phys) { return g_x86_lim + (phys - MB_LIM_BASE); }
/* multi-hart (mh.c): fake CLINT msip/mtimecmp per hart, mip of the calling hart, wfi of a secondary thread */
void     plat_msip_set(uint32_t hart, uint32_t v);
void     plat_mtimecmp_set(uint32_t hart, uint64_t t);
uint64_t plat_mip(void);
void     plat_wfi(void);
uint32_t plat_beu_accrued_hart(uint32_t hart);
int      plat_in_image(uint64_t addr);
__attribute__((noreturn)) void plat_mh_restart(uint32_t hart);
void     plat_mh_relax(void);   /* hart 0's wait loops: a short sleep, so the fake clock does not run away */

#else /* riscv: X280 or SPIKE_HOST */

#define PLAT_CSRR(csr) ({ uint64_t v_; __asm__ volatile("csrr %0, " #csr : "=r"(v_)); v_; })
static inline void *plat_ptr(uint64_t phys) { return (void *)(uintptr_t)phys; }
static inline uint64_t plat_mtime(void) { return *(volatile uint64_t *)PLAT_MTIME; }
static inline uint64_t plat_mcycle(void) { return PLAT_CSRR(mcycle); }
static inline uint64_t plat_misa(void) { return PLAT_CSRR(misa); }
static inline uint32_t plat_mhartid(void) { return (uint32_t)PLAT_CSRR(mhartid); }
static inline uint32_t plat_vlenb(void) {
#if defined(__riscv_vector)
    return (uint32_t)PLAT_CSRR(0xC22);   /* vlenb; mstatus.VS is on (crt0.S) */
#else
    return 0;
#endif
}
static inline void plat_fence_i(void) { __asm__ volatile("fence.i" ::: "memory"); }

#if defined(SPIKE_HOST)
/* spike has no X280 RNMI CSRs: 0x350-0x353 trap there (x280/sim/selftest/t_csr.c). They are emulated
 * in memory by spike_host.c, which also emulates the RNMI itself (SPIKE_HOST.md). */
uint64_t plat_mnstatus(void);
uint64_t plat_mncause(void);
uint64_t plat_mnepc(void);
void     plat_rnmi_rearm(void);
void     plat_beu_read(uint32_t *cause, uint64_t *value, uint32_t *accrued);
__attribute__((noreturn)) void plat_park_forever(void);
#else
/* X280 RNMI CSRs are 0x350-0x353 (mnscratch, mnepc, mncause, mnstatus), not Smrnmi's 0x740-0x744
 * (L2CPUTile/RNMIs.md:6, 63-70); mnstatus.NMIE is bit 3 (RNMIs.md:84). */
static inline uint64_t plat_mnstatus(void) { return PLAT_CSRR(0x353); }
static inline uint64_t plat_mncause(void) { return PLAT_CSRR(0x352); }
static inline uint64_t plat_mnepc(void) { return PLAT_CSRR(0x351); }
static inline void plat_rnmi_rearm(void) { __asm__ volatile("csrsi 0x353, 8" ::: "memory"); }
static inline void plat_beu_read(uint32_t *cause, uint64_t *value, uint32_t *accrued) {
    *cause = *(volatile uint32_t *)(PLAT_BEU0 + PLAT_BEU_CAUSE);
    *value = *(volatile uint64_t *)(PLAT_BEU0 + PLAT_BEU_VALUE);
    *accrued = *(volatile uint32_t *)(PLAT_BEU0 + PLAT_BEU_ACCRUED);
}
/* PARK: stop touching LIM and wait in wfi for the host's RNMI (§5.2 PARK protocol). NMIE is 1 here. */
__attribute__((noreturn)) static inline void plat_park_forever(void) {
    for (;;) __asm__ volatile("wfi" ::: "memory");
}
#endif

__attribute__((noreturn)) void plat_restart_after_trap(void);   /* crt0.S: fresh sp, monitor_resume() */

/* multi-hart (mh.c). The CLINT stores are 4 B (msip) and 8 B (mtimecmp) as in the clint0 layout. */
static inline void plat_msip_set(uint32_t h, uint32_t v) { *(volatile uint32_t *)PLAT_MSIP(h) = v; }
static inline void plat_mtimecmp_set(uint32_t h, uint64_t t) { *(volatile uint64_t *)PLAT_MTIMECMP(h) = t; }
static inline uint64_t plat_mip(void) { return PLAT_CSRR(mip); }
static inline void plat_wfi(void) { __asm__ volatile("wfi" ::: "memory"); }
extern char __image_end[];
static inline int plat_in_image(uint64_t a) { return a >= MB_LIM_BASE && a < (uint64_t)(uintptr_t)__image_end; }
#if defined(SPIKE_HOST) && !defined(SPIKE_X280_REGIONS)
static inline uint32_t plat_beu_accrued_hart(uint32_t h) { (void)h; return 0; }   /* no BEU pages mapped */
#else
static inline uint32_t plat_beu_accrued_hart(uint32_t h) { return *(volatile uint32_t *)(PLAT_BEU(h) + PLAT_BEU_ACCRUED); }
#endif
__attribute__((noreturn)) void plat_mh_restart(uint32_t hart);   /* crt0.S: hart h's fresh sp, mh_secondary_resume() */
static inline void plat_mh_relax(void) {}
static inline uint64_t plat_call(uint64_t entry, const struct mb_call_args *a, const struct mon_api *api) {
    typedef uint64_t (*payload_fn)(const struct mb_call_args *, const struct mon_api *);
    return ((payload_fn)(uintptr_t)entry)(a, api);
}
#endif

#endif /* X280_PLAT_H */
