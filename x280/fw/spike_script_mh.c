/* Multi-hart SPIKE_HOST script (SPIKE_HOST.md "Multi-hart script"): spike -p4 with the X280 regions mapped.
 *
 *   make -C $RT/x280/fw spike SPIKE_SCRIPT=spike_script_mh.c SPIKE_MH=1 SPIKE_X280_REGIONS=1 [SPIKE_MH_AUDIT=1]
 *
 * Hart 0 plays x280d: a ping before any adoption (nobody ready), SH_MH_ADOPT of harts 1-3 (their --harts 4 RNMI
 * vectors, the trigger bits, the emulated RNMI), a ping, then mh_bench: every kernel x pattern x mask on a 64 KiB
 * xorshift working set, the P4 bank-bit sweep, K3 on hart 0 under K6 on harts 1-3, a background job collected later,
 * a trap on hart 2 (then a ping: hart 2 is back in its idle loop), and plain monitor commands to show hart 0 unharmed.
 * SPIKE_MH_AUDIT=1 keeps a short subset for the one-writer audit (run with spike -l --log-commits).
 * Timing means nothing here (spike: one instruction per cycle, mtime every 100 instructions per hart).
 */
#include "spike_host.h"
#include "mh.h"

#define WS      0x08020000u          /* working set: SH_LOAD_PATTERN, 64 KiB */
#define WS_LEN  0x10000u
#define WIN     100                  /* mtime ticks per mh_bench window (spike: 10 MHz) */
#define MHB_CALL(name_, k_, flags_) { .op = SH_CMD, .name = name_, .cmd = MB_CMD_CALL, .arg_entry = MHB_BASE, \
    .policy = (k_), .flags = (flags_), .tick = WIN, .arg_addr = WS, .arg_len = WS_LEN, .report = SH_REP_MHB | SH_REP_MH }

#if defined(SPIKE_MH_AUDIT)
static const uint8_t K_AUDIT[] = { 1, 6, 9 };
static const uint8_t P_AUDIT[] = { 0, 3 };
static const uint8_t M_AUDIT[] = { 0xF, 0xE };
static const struct sh_sweep SW_MAIN = { K_AUDIT, P_AUDIT, 0, M_AUDIT, 3, 2, 0, 2, 0, 0 };
#else
static const uint8_t P_0123[] = { 0, 1, 2, 3 };
static const uint8_t MASKS[] = { 0x1, 0x3, 0xF, 0xE };
static const uint8_t K_ALL[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
static const uint8_t P_SWEEP[] = { 4 };
static const uint8_t K_P4[] = { 5 };
static const uint8_t KBITS[] = { 6, 7, 8, 9, 10, 11, 12 };
static const uint8_t M_F[] = { 0xF };
static const struct sh_sweep SW_MAIN = { K_ALL, P_0123, 0, MASKS, 12, 4, 0, 4, 0, 0 };
static const struct sh_sweep SW_P4 = { K_P4, P_SWEEP, KBITS, M_F, 1, 1, 7, 1, 0, 0 };
#endif

const struct sh_step sh_script[] = {
    { .op = SH_CMD, .name = "nop", .cmd = MB_CMD_NOP },
    { .op = SH_CMD, .name = "mh_ping_unadopted", .cmd = MB_CMD_MH, .flags = 0xE, .arg_len = 200, .report = SH_REP_MH },
    { .op = SH_MH_ADOPT, .name = "mh_adopt", .flags = 0xE, .delay_us = 100, .report = SH_REP_MH },
    { .op = SH_CMD, .name = "mh_ping", .cmd = MB_CMD_MH, .flags = 0xE, .arg_len = 2000, .report = SH_REP_MH },
    { .op = SH_LOAD_PATTERN, .name = "load_ws", .dst = WS, .arg_len = WS_LEN, .tick = 0x5EED },
    { .op = SH_LOAD_BYTES, .name = "load_mh_bench", .dst = MHB_BASE, .kind = MB_LOAD_CODE, SH_BYTES(sh_bin_mh_bench) },
    { .op = SH_CMD, .name = "fence_i", .cmd = MB_CMD_FENCE_I },
    { .op = SH_MHB_SWEEP, .name = "mhb_sweep", .sweep = &SW_MAIN, .tick = WIN, .arg_addr = WS, .arg_len = WS_LEN,
      .report = SH_REP_MHB },
#if !defined(SPIKE_MH_AUDIT)
    { .op = SH_MHB_SWEEP, .name = "mhb_p4", .sweep = &SW_P4, .tick = WIN, .arg_addr = WS, .arg_len = WS_LEN,
      .report = SH_REP_MHB },
    MHB_CALL("mhb_loaded_latency", MHB_K_VLE8_M2 | (MHB_K_CHAIN << 8), 0xF),
    { .op = SH_CMD, .name = "mhb_bad_range", .cmd = MB_CMD_CALL, .arg_entry = MHB_BASE, .policy = MHB_K_LD8,
      .flags = 0xF, .tick = WIN, .arg_addr = 0x081D0000u, .arg_len = 0x10000u, .report = SH_REP_MHB },   /* over itself */
    /* background: K6/P0 on harts 1-3 (mode START), a NOP and a SELFTEST on hart 0 meanwhile, then COLLECT */
    { .op = SH_CMD, .name = "mhb_bg_start", .cmd = MB_CMD_CALL, .arg_entry = MHB_BASE, .policy = MHB_K_VLE8_M2,
      .flags = 0xE | (MHB_MODE_START << 12), .tick = 5000, .arg_addr = WS, .arg_len = WS_LEN, .report = SH_REP_MHB },
    { .op = SH_CMD, .name = "nop_during_bg", .cmd = MB_CMD_NOP, .report = SH_REP_MH },
    { .op = SH_CMD, .name = "mhb_bg_collect", .cmd = MB_CMD_CALL, .arg_entry = MHB_BASE,
      .flags = MHB_MODE_COLLECT << 12, .report = SH_REP_MHB | SH_REP_MH },
#endif
    /* a worker trap on hart 2: dumped into its aux line, back to idle, missing from the done mask */
    MHB_CALL("mhb_trap_h2", MHB_K_TRAP, 0xF | (2u << 8)),
    { .op = SH_CMD, .name = "mh_ping_after_trap", .cmd = MB_CMD_MH, .flags = 0xE, .arg_len = 2000, .report = SH_REP_MH },
    MHB_CALL("mhb_after_trap", MHB_K_LD8X8, 0xF),
#if !defined(SPIKE_MH_AUDIT)
    { .op = SH_CMD, .name = "selftest", .cmd = MB_CMD_SELFTEST },
#endif
    { .op = SH_CMD, .name = "nop_end", .cmd = MB_CMD_NOP, .report = SH_REP_MH },
    { .op = SH_END },
};
