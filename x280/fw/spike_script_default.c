/* Default SPIKE_HOST script (SPIKE_HOST.md): what T-3.2b needs from the monitor, in the order a first
 * hardware session would do it. Every step prints one "SH step=..." line; the harness checks them.
 * Override with `make spike SPIKE_SCRIPT=<file.c>`.
 *
 * Addresses: payloads at MB_PAYLOAD_BASE (0x0801_1000), the illegal-instruction payload at +0x1000, the
 * hang payload at +0x2000; lim_bench's working set defaults to the first line after itself.
 */
#include "spike_host.h"
#include "wbc_api.h"

#define PAY MB_PAYLOAD_BASE

#ifndef SPIKE_POLICY
#define SPIKE_POLICY 0
#endif
#ifndef SPIKE_VARIANT
#define SPIKE_VARIANT 0
#endif
#ifndef SPIKE_N
#define SPIKE_N 0
#endif
#define VFLAGS (MB_FLAG_VARIANT_SET | ((uint32_t)SPIKE_VARIANT << MB_FLAG_VARIANT_SHIFT))

const struct sh_step sh_script[] = {
    { .op = SH_CMD, .name = "nop", .cmd = MB_CMD_NOP },
    { .op = SH_CMD, .name = "selftest", .cmd = MB_CMD_SELFTEST },
    /* LOAD/FENCE_I/CALL of a checksum payload, 10/10 (T-3.E2); ret = zlib crc32 of the first 4 KiB of
     * the monitor image itself, which the harness can compute from monitor_spike.bin. */
    { .op = SH_LOAD_BYTES, .name = "load_checksum", .dst = PAY, .kind = MB_LOAD_CODE, SH_BYTES(sh_bin_checksum) },
    { .op = SH_CMD, .name = "call_before_fence", .cmd = MB_CMD_CALL, .arg_entry = PAY },
    { .op = SH_CMD, .name = "fence_i", .cmd = MB_CMD_FENCE_I },
    { .op = SH_CMD, .name = "call_checksum", .cmd = MB_CMD_CALL, .arg_entry = PAY,
      .arg_addr = MB_LIM_BASE, .arg_len = 4096, .count = 10 },
    /* illegal instruction inside a payload: trap dump (mcause 2, mepc = entry), monitor survives */
    { .op = SH_LOAD_BYTES, .name = "load_illegal", .dst = PAY + 0x1000, .kind = MB_LOAD_CODE,
      SH_BYTES(sh_bin_illegal) },
    { .op = SH_CMD, .name = "fence_i", .cmd = MB_CMD_FENCE_I },
    { .op = SH_CMD, .name = "call_illegal", .cmd = MB_CMD_CALL, .arg_entry = PAY + 0x1000 },
    { .op = SH_CMD, .name = "nop_after_trap", .cmd = MB_CMD_NOP },
    /* LIM microbench payload, every test (T-3.E2m) on a small working set */
    { .op = SH_LOAD_BYTES, .name = "load_lim_bench", .dst = PAY, .kind = MB_LOAD_CODE, SH_BYTES(sh_bin_lim_bench) },
    { .op = SH_CMD, .name = "fence_i", .cmd = MB_CMD_FENCE_I },
    { .op = SH_CMD, .name = "lb_ptr_chase", .cmd = MB_CMD_CALL, .arg_entry = PAY, .policy = 1,
      .arg_len = 16384, .tick = 2000 },
    { .op = SH_CMD, .name = "lb_stream", .cmd = MB_CMD_CALL, .arg_entry = PAY, .policy = 2, .arg_len = 16384 },
    { .op = SH_CMD, .name = "lb_vle32_m1", .cmd = MB_CMD_CALL, .arg_entry = PAY, .policy = 3, .arg_len = 16384 },
    { .op = SH_CMD, .name = "lb_vle32_m4", .cmd = MB_CMD_CALL, .arg_entry = PAY, .policy = 4, .arg_len = 16384 },
    { .op = SH_CMD, .name = "lb_vle32_m8", .cmd = MB_CMD_CALL, .arg_entry = PAY, .policy = 5, .arg_len = 16384 },
    { .op = SH_CMD, .name = "lb_exec_loop", .cmd = MB_CMD_CALL, .arg_entry = PAY, .policy = 6, .tick = 200 },
    { .op = SH_CMD, .name = "lb_zero_dev_denied", .cmd = MB_CMD_CALL, .arg_entry = PAY, .policy = 7 },
    { .op = SH_CMD, .name = "lb_mtime_ratio", .cmd = MB_CMD_CALL, .arg_entry = PAY, .policy = 8 },
    /* PARK and warm re-entry */
    { .op = SH_PARK, .name = "park" },
    { .op = SH_CMD, .name = "nop_after_park", .cmd = MB_CMD_NOP },
    /* RNMI re-arm: hung payload -> RNMI -> warm entry -> CALL -> hang, 10/10 (T-3.E2) */
    { .op = SH_LOAD_BYTES, .name = "load_hang", .dst = PAY + 0x2000, .kind = MB_LOAD_CODE, SH_BYTES(sh_bin_hang) },
    { .op = SH_CMD, .name = "fence_i", .cmd = MB_CMD_FENCE_I },
    { .op = SH_CALL_HANG, .name = "call_hang", .arg_entry = PAY + 0x2000, .count = 10, .delay_us = 200 },
    { .op = SH_CMD, .name = "nop_after_hang", .cmd = MB_CMD_NOP },
    /* CRC fault path, host gone, unknown command */
    { .op = SH_CORRUPT_INFER, .name = "infer_bad_obs_crc" },
    { .op = SH_POKE32, .name = "kmd_gone", .dst = MB_L(MB_L38_CLEANUP), .arg_len = MB_HOST_GONE },
    { .op = SH_CMD, .name = "nop_while_gone", .cmd = MB_CMD_NOP },
    { .op = SH_POKE32, .name = "attach_clear", .dst = MB_L(MB_L38_CLEANUP), .arg_len = 0 },
    { .op = SH_CMD, .name = "nop_after_attach", .cmd = MB_CMD_NOP },
    { .op = SH_CMD, .name = "unknown_cmd", .cmd = 9 },
#if defined(SPIKE_BLOB)
    /* the real kernel on a real blob (make spike SPIKE_BLOB=... [SPIKE_VARIANT=v SPIKE_POLICY=p]) */
    { .op = SH_LOAD_FILE, .name = "load_blob", .dst = PAY, .kind = MB_LOAD_BLOB, .path = SPIKE_BLOB },
    { .op = SH_INFER_CONST, .name = "infer_const", .policy = SPIKE_POLICY, .flags = VFLAGS, .tick = 1,
      .count = 4 },
    { .op = SH_CMD, .name = "bench", .cmd = MB_CMD_BENCH, .policy = SPIKE_POLICY, .flags = VFLAGS, .arg_len = 3 },
#if defined(SPIKE_OBS) && defined(SPIKE_OUT)
    { .op = SH_INFER_FILE, .name = "infer_file", .policy = SPIKE_POLICY, .flags = VFLAGS, .count = SPIKE_N,
      .path = SPIKE_OBS, .path2 = SPIKE_OUT },
#endif
#endif
    { .op = SH_END },
};
