/* Test payloads for LOAD/FENCE_I/CALL (PLAN.md T-3.E2: "LOAD+CALL of a checksum payload correct 10/10";
 * "deliberately hung payload (infinite loop) -> PARK -> CALL -> hang -> PARK"; T-3.2b trap dump).
 *
 * One source, one payload per build: -DPAYLOAD_CHECKSUM, -DPAYLOAD_ILLEGAL or -DPAYLOAD_HANG. The x86
 * test build compiles all three with -DMON_X86 into the shim, where the illegal instruction and the hang
 * are emulated (plat_x86.c).
 */
#include <stdint.h>
#include "mailbox.h"

#if defined(MON_X86)
#define PAYLOAD_FN(name) uint64_t name(const struct mb_call_args *a, const struct mon_api *api)
void x86_emulate_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval);   /* plat_x86.c */
void x86_emulate_hang(void);
#else
#define PAYLOAD_FN(name) __attribute__((section(".text.payload_entry"), used)) \
    uint64_t payload_entry(const struct mb_call_args *a, const struct mon_api *api)
#endif

#if defined(PAYLOAD_CHECKSUM) || defined(MON_X86)
/* zlib CRC-32 of [arg_addr, arg_addr + arg_len); also written to result[0..7] = {crc, len}. */
PAYLOAD_FN(x86_payload_checksum)
{
    const volatile uint8_t *p = api->ptr(a->arg_addr);
    uint32_t c = ~0u;
    for (uint32_t i = 0; i < a->arg_len; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        if ((i & 4095u) == 4095u) api->heartbeat();
    }
    c = ~c;
    volatile uint32_t *r = (volatile uint32_t *)api->result;
    r[0] = c;
    r[1] = a->arg_len;
    return c;
}
#endif

#if defined(PAYLOAD_ILLEGAL) || defined(MON_X86)
/* 0xc0001073 = csrrw x0, cycle, x0 (`unimp` without RVC): a write to a read-only CSR, illegal everywhere;
 * expected dump: mcause 2, mepc = the payload entry, mtval = 0xc0001073 where the hart reports it. */
PAYLOAD_FN(x86_payload_illegal)
{
    (void)a; (void)api;
#if defined(MON_X86)
    x86_emulate_trap(2u, a->arg_entry, 0xc0001073u);
    return 0;
#else
    __asm__ volatile(".4byte 0xc0001073" ::: "memory");
    return 0xBAD;
#endif
}
#endif

#if defined(PAYLOAD_HANG) || defined(MON_X86)
/* Never returns and never touches the mailbox: only the host's RNMI (PARK) gets the hart back. */
PAYLOAD_FN(x86_payload_hang)
{
    (void)a; (void)api;
#if defined(MON_X86)
    x86_emulate_hang();
    return 0;
#else
    for (;;) __asm__ volatile("" ::: "memory");
#endif
}
#endif
