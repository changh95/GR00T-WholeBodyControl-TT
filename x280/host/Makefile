# libwcio.so for x280d (PLAN.md WP-3.3). Build: make -C $RT/x280/host
# No global -mavx512f: only wcio__stream_lines carries target("avx512f"), so the library loads on any
# x86-64 and the store path refuses (-ENOTSUP) instead of faulting when AVX-512 is missing.
CC ?= clang-20
ifeq ($(origin CC),default)
CC := clang-20
endif
CFLAGS ?= -O2 -g -std=c11 -fPIC -Wall -Wextra -Werror -fno-strict-aliasing
OBJDUMP ?= llvm-objdump-20

libwcio.so: wcio.c
	$(CC) $(CFLAGS) -shared -o $@ $<

# Disassembly of the store loops, for review (test_kmd_fake.py checks the same thing).
disasm: libwcio.so
	$(OBJDUMP) -d --no-show-raw-insn \
		--disassemble-symbols=wcio__stream_lines,wcio__stream_lines_2x32,wcio__store_words64 $<

clean:
	rm -f libwcio.so

.PHONY: disasm clean
