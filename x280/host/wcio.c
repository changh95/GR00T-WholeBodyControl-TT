// libwcio: the only code that stores to Blackhole TLB mappings on behalf of x280d (PLAN.md WP-3.3, D12).
//
// Why it exists: a partial write to a LIM line whose ECC is uninitialised can fault (PLAN R26), and
// pyluwen-style UC u32 copies cannot initialise a line. Method A initialises a line with one 64 B
// AVX-512 non-temporal store (vmovntdq zmm) through a write-combining mapping, followed by sfence,
// so the PCIe tile sees a full 64 B write. Without AVX-512 the store path refuses and writes
// nothing: it never falls back to narrower stores, which would be partial writes.
//
// S1d (after S1c, 2026-10-25: method A's lines read back right but the CCACHE reports an uncorrectable ECC
// error on most later reads, i.e. a full 64 B host store does not initialise LIM ECC) adds two more store
// widths, each its own noinline loop so that its disassembly shows exactly which stores reach the mapping:
// two 32 B ymm stores per line into the WC mapping (UMD memcpy_to_device's pattern, which S1c saw arrive
// as ONE 64 B access per line), and aligned 8 B UC stores. Both are LIM-init stores only (kmd.lim_init,
// regs LIM_INIT_WC32 / LIM_INIT_UC8); nothing here decides where they may go.
//
// Every access is appended to a transaction log (op, address, size, alignment) that x280d prints in
// --dry-run and that tests/x280/test_kmd_fake.py checks. Single-threaded use only (x280d owns one
// thread that touches the device).

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cpuid.h>
#include <immintrin.h>

#define WCIO_ABI 3  // 2: wcio_uc_read_words; 3: wcio_wc_store_lines_2x32, wcio_uc_store64 (S1d)

enum { WCIO_OP_WC64 = 1, WCIO_OP_SFENCE = 2, WCIO_OP_UC_W32 = 3, WCIO_OP_UC_R32 = 4, WCIO_OP_REFUSED = 5,
       WCIO_OP_UC_W64 = 6, WCIO_OP_WC32 = 7 };

struct wcio_txn {       // 32 B, mirrored by kmd.py WcioTxn
    uint64_t seq;       // 0, 1, 2, ... since wcio_log_enable
    uint64_t addr;      // virtual address (0 for sfence)
    uint32_t size;      // bytes stored or loaded (0 for sfence)
    uint32_t value;     // UC value (low u32 of a UC_W64); first u32 of a WC line or WC32 half; errno for REFUSED
    uint8_t op;
    uint8_t align64;    // addr % 64 == 0
    uint8_t align4;     // addr % 4 == 0
    uint8_t reserved[5];
};

static struct wcio_txn *g_log;
static uint64_t g_cap, g_n;
static int g_forced_off;

int wcio_abi(void) { return WCIO_ABI; }

static void log_push(uint8_t op, const volatile void *addr, uint32_t size, uint32_t value) {
    if (!g_log) return;
    struct wcio_txn *t = &g_log[g_n % g_cap];
    uintptr_t a = (uintptr_t)addr;
    t->seq = g_n++;
    t->addr = a;
    t->size = size;
    t->value = value;
    t->op = op;
    t->align64 = (a % 64) == 0;
    t->align4 = (a % 4) == 0;
    memset(t->reserved, 0, sizeof t->reserved);
}

int wcio_log_enable(uint64_t capacity) {
    free(g_log);
    g_log = NULL;
    g_cap = g_n = 0;
    if (capacity == 0) return 0;
    g_log = calloc(capacity, sizeof *g_log);
    if (!g_log) return -ENOMEM;
    g_cap = capacity;
    return 0;
}

uint64_t wcio_log_total(void) { return g_n; }
uint64_t wcio_log_capacity(void) { return g_cap; }

// Copies the oldest still-held records with seq >= first, in order; returns how many were copied.
uint64_t wcio_log_copy(struct wcio_txn *out, uint64_t max, uint64_t first) {
    if (!g_log) return 0;
    uint64_t lo = g_n > g_cap ? g_n - g_cap : 0;
    if (first < lo) first = lo;
    uint64_t k = 0;
    for (uint64_t s = first; s < g_n && k < max; s++) out[k++] = g_log[s % g_cap];
    return k;
}

// AVX-512F usable: CPUID.(7,0):EBX[16] and the OS saves opmask/ZMM state (XCR0 bits 1,2,5,6,7).
int wcio_avx512_ok(void) {
    if (g_forced_off) return 0;
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & bit_OSXSAVE)) return 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d) || !(b & bit_AVX512F)) return 0;
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (lo & 0xE6) == 0xE6;
}

void wcio_force_no_avx512(int on) { g_forced_off = on != 0; }  // test hook for the refusal path

// AVX usable (the 2x32 B ymm store loop): CPUID.1:ECX.AVX[28], OSXSAVE, and the OS saves SSE/AVX state (XCR0 bits
// 1, 2). The AVX-512 test hook turns this off too, so the refusal path of every vector store loop can be tested.
int wcio_avx_ok(void) {
    if (g_forced_off) return 0;
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & bit_OSXSAVE) || !(c & bit_AVX)) return 0;
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (lo & 0x6) == 0x6;
}

// The store loop proper. Kept free of calls and logging so that its disassembly shows exactly
// which stores reach the mapping: vmovntdq zmm to dst, sfence (checked by test_kmd_fake.py).
__attribute__((target("avx512f"), noinline))
void wcio__stream_lines(void *dst, const void *src, uint64_t nlines, int fence_each) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    if (fence_each) {
        for (uint64_t i = 0; i < nlines; i++) {
            _mm512_stream_si512((__m512i *)(d + 64 * i), _mm512_loadu_si512((const void *)(s + 64 * i)));
            _mm_sfence();
        }
    } else {
        for (uint64_t i = 0; i < nlines; i++)
            _mm512_stream_si512((__m512i *)(d + 64 * i), _mm512_loadu_si512((const void *)(s + 64 * i)));
        _mm_sfence();
    }
}

// nlines full 64 B lines from src to the 64 B-aligned dst (a WC mapping), then sfence (after each
// line if fence_each, else once). Returns 0, -EINVAL (misaligned or empty) or -ENOTSUP (no
// AVX-512); on error nothing at all is written.
int wcio_wc_store_lines(void *dst, const void *src, uint64_t nlines, int fence_each) {
    int err = 0;
    if (nlines == 0 || ((uintptr_t)dst % 64) != 0) err = -EINVAL;
    else if (!wcio_avx512_ok()) err = -ENOTSUP;
    if (err) {
        log_push(WCIO_OP_REFUSED, dst, (uint32_t)(nlines * 64), (uint32_t)-err);
        return err;
    }
    wcio__stream_lines(dst, src, nlines, fence_each);
    const char *d = (const char *)dst, *s = (const char *)src;
    for (uint64_t i = 0; i < nlines; i++) {
        uint32_t first;
        memcpy(&first, s + 64 * i, 4);
        log_push(WCIO_OP_WC64, d + 64 * i, 64, first);
        if (fence_each) log_push(WCIO_OP_SFENCE, 0, 0, 0);
    }
    if (!fence_each) log_push(WCIO_OP_SFENCE, 0, 0, 0);
    return 0;
}

// S1d: each line as two regular (not non-temporal) 32 B ymm stores, low half first, through the WC mapping, so
// the CPU's write-combining buffer assembles the line (UMD memcpy_to_device Phase 1/2: _mm256_storeu_si256 x 2 per
// 64 B, no fence). The stores are inline asm so that the compiler can neither merge them into one zmm store nor
// turn the loop into a memcpy call (checked by disassembly in test_kmd_fake.py).
__attribute__((target("avx"), noinline))
void wcio__stream_lines_2x32(void *dst, const void *src, uint64_t nlines, int fence_each) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    for (uint64_t i = 0; i < nlines; i++) {
        __m256i lo = _mm256_loadu_si256((const __m256i *)(s + 64 * i));
        __m256i hi = _mm256_loadu_si256((const __m256i *)(s + 64 * i + 32));
        __asm__ volatile("vmovdqu %1, %0" : "=m"(*(__m256i *)(d + 64 * i)) : "x"(lo) : "memory");
        __asm__ volatile("vmovdqu %1, %0" : "=m"(*(__m256i *)(d + 64 * i + 32)) : "x"(hi) : "memory");
        if (fence_each) _mm_sfence();
    }
    if (!fence_each) _mm_sfence();
}

// nlines full 64 B lines from src to the 64 B-aligned dst (a WC mapping) as 2 x 32 B ymm stores each, then sfence
// (after each line if fence_each, else once). Returns 0, -EINVAL (misaligned or empty) or -ENOTSUP (no AVX); on
// error nothing at all is written. Logged as two WC32 records per line (+0, +32).
int wcio_wc_store_lines_2x32(void *dst, const void *src, uint64_t nlines, int fence_each) {
    int err = 0;
    if (nlines == 0 || ((uintptr_t)dst % 64) != 0) err = -EINVAL;
    else if (!wcio_avx_ok()) err = -ENOTSUP;
    if (err) {
        log_push(WCIO_OP_REFUSED, dst, (uint32_t)(nlines * 64), (uint32_t)-err);
        return err;
    }
    wcio__stream_lines_2x32(dst, src, nlines, fence_each);
    const char *d = (const char *)dst, *s = (const char *)src;
    for (uint64_t i = 0; i < nlines; i++) {
        for (int h = 0; h < 2; h++) {
            uint32_t first;
            memcpy(&first, s + 64 * i + 32 * h, 4);
            log_push(WCIO_OP_WC32, d + 64 * i + 32 * h, 32, first);
        }
        if (fence_each) log_push(WCIO_OP_SFENCE, 0, 0, 0);
    }
    if (!fence_each) log_push(WCIO_OP_SFENCE, 0, 0, 0);
    return 0;
}

// S1d: n aligned 8 B stores in address order, each exactly one 64-bit `mov` to a UC mapping (an aligned 8 B UC
// store is one PCIe write with 8 byte enables). volatile keeps them single, ordered and out of any memcpy idiom.
__attribute__((noinline))
void wcio__store_words64(void *dst, const void *src, uint64_t nwords) {
    volatile uint64_t *d = (volatile uint64_t *)dst;
    const unsigned char *s = (const unsigned char *)src;
    for (uint64_t i = 0; i < nwords; i++) {
        uint64_t v;
        memcpy(&v, s + 8 * i, 8);
        d[i] = v;
    }
}

// nwords (>= 1) aligned 8 B UC stores of src to dst. Returns 0 or -EINVAL (misaligned or empty: nothing written).
int wcio_uc_store64(void *dst, const void *src, uint64_t nwords) {
    if (nwords == 0 || ((uintptr_t)dst % 8) != 0) {
        log_push(WCIO_OP_REFUSED, dst, (uint32_t)(nwords * 8), EINVAL);
        return -EINVAL;
    }
    wcio__store_words64(dst, src, nwords);
    const char *d = (const char *)dst, *s = (const char *)src;
    for (uint64_t i = 0; i < nwords; i++) {
        uint32_t lo;
        memcpy(&lo, s + 8 * i, 4);
        log_push(WCIO_OP_UC_W64, d + 8 * i, 8, lo);
    }
    return 0;
}

void wcio_sfence(void) {
    _mm_sfence();
    log_push(WCIO_OP_SFENCE, 0, 0, 0);
}

// 4 B uncached accessors: exactly one 32-bit load or store each.
int wcio_uc_write32(void *addr, uint32_t value) {
    if ((uintptr_t)addr % 4) {
        log_push(WCIO_OP_REFUSED, addr, 4, EINVAL);
        return -EINVAL;
    }
    *(volatile uint32_t *)addr = value;
    log_push(WCIO_OP_UC_W32, addr, 4, value);
    return 0;
}

int wcio_uc_read32(const void *addr, uint32_t *out) {
    if ((uintptr_t)addr % 4) {
        log_push(WCIO_OP_REFUSED, addr, 4, EINVAL);
        return -EINVAL;
    }
    *out = *(const volatile uint32_t *)addr;
    log_push(WCIO_OP_UC_R32, addr, 4, *out);
    return 0;
}

// n (1..16) consecutive 4 B UC loads inside one 64 B line, in address order, in one call: x280d's L1/L35/L36
// reads (PLAN §1.4.4 step 4) without one ctypes round trip per word. Still exactly one 32-bit load per word, each
// logged as UC_R32 like wcio_uc_read32: a wider load from a UC mapping is not guaranteed to reach PCIe as one
// read, and the X280 writes these lines with 4/8 B stores. -EINVAL (nothing read) for n outside 1..16, a
// misaligned address or a range that crosses a line.
int wcio_uc_read_words(const void *addr, uint32_t n, uint32_t *out) {
    uintptr_t a = (uintptr_t)addr;
    if (n == 0 || n > 16 || a % 4 || (a % 64) + 4u * n > 64) {
        log_push(WCIO_OP_REFUSED, addr, 4u * n, EINVAL);
        return -EINVAL;
    }
    const volatile uint32_t *p = (const volatile uint32_t *)addr;
    for (uint32_t i = 0; i < n; i++) {
        out[i] = p[i];
        log_push(WCIO_OP_UC_R32, p + i, 4, out[i]);
    }
    return 0;
}

// u64 written by the X280 as one store, read by the host as hi, lo, hi (PLAN §1.4.4 L1 heartbeat):
// retried until both hi reads agree, at most 4 times; -EAGAIN if it never settles.
int wcio_uc_read64_hlh(const void *addr, uint64_t *out) {
    const char *p = (const char *)addr;
    if ((uintptr_t)addr % 8) {
        log_push(WCIO_OP_REFUSED, addr, 8, EINVAL);
        return -EINVAL;
    }
    for (int k = 0; k < 4; k++) {
        uint32_t hi0, lo, hi1;
        wcio_uc_read32(p + 4, &hi0);
        wcio_uc_read32(p, &lo);
        wcio_uc_read32(p + 4, &hi1);
        if (hi0 == hi1) {
            *out = ((uint64_t)hi1 << 32) | lo;
            return 0;
        }
    }
    return -EAGAIN;
}
