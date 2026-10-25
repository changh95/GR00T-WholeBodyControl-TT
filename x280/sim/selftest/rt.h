/* Minimal freestanding runtime of the spike self-test images: HTIF console + exit, CSR helpers, trap record.
 *
 * HTIF (riscv-isa-sim fesvr/htif.cc:203-207, 288-335; fesvr/device.cc:56-70; fesvr/syscall.cc:213-215):
 * spike finds `tohost`/`fromhost` by ELF symbol and polls tohost between instruction batches.
 *   tohost = (1 << 56) | (1 << 48) | ch   device 1 (bcd) command 1: write one console byte to spike's stdout
 *   tohost = (code << 1) | 1              device 0: exit; spike exits with `code` (and prints
 *                                         "*** FAILED *** (tohost = code)" on stderr when code != 0)
 * The target waits for tohost == 0 (the host clears it) before every write.
 */
#ifndef X280_SIM_RT_H
#define X280_SIM_RT_H
#include <stdint.h>

void htif_putc(char c);
void htif_puts(const char *s);
void htif_puthex(uint64_t v);   /* 0x + 16 hex digits */
void htif_putdec(uint64_t v);
__attribute__((noreturn)) void htif_exit(uint64_t code);

/* Filled by trap_handler for every trap. When trap_resume != 0 the handler resumes after the faulting
 * instruction (mepc += 2 or 4), else it prints the dump and exits with TRAP_EXIT_BASE + mcause. */
#define TRAP_EXIT_BASE 0xE00u
struct trap_rec { uint64_t count, mcause, mepc, mtval; };
extern volatile struct trap_rec trap_last;
extern volatile int trap_resume;

#define CSR_READ(csr) ({ uint64_t v_; __asm__ volatile("csrr %0, " #csr : "=r"(v_)); v_; })
#define CSR_WRITE(csr, v) __asm__ volatile("csrw " #csr ", %0" :: "r"((uint64_t)(v)) : "memory")

#endif
