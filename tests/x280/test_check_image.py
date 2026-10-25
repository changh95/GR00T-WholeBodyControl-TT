"""scripts/check_image.py (T-3.2) must pass the real image and fail deliberately broken ones.

    simpy -m pytest $RT/tests/x280/test_check_image.py -q
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RT = Path(__file__).resolve().parents[2]
FW = RT / "x280" / "fw"
CHECK = RT / "scripts" / "check_image.py"
# Off the real-time cores 2,3,4,6,7,12-15 and their SMT siblings (task rule; override with X280_CPUS).
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
PIN = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]
CC = ["clang-20", "--target=riscv64-unknown-elf", "-march=rv64gcv_zfh_zba_zbb", "-mabi=lp64d", "-c"]


def _mod():
    spec = importlib.util.spec_from_file_location("check_image", CHECK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def built():
    r = subprocess.run(PIN + ["make", "-C", str(FW), "x280"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return FW / "build"


def _run(elf: Path, *extra) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(CHECK), str(elf), *extra], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_real_image_passes(built):
    code, out = _run(built / "monitor.elf")
    assert code == 0, out
    assert "PASS weight_blob" in out and "== 0 FAIL" in out
    assert "PASS crc32_tab" in out, out
    # harts 1-3: the fixed RNMI exception spin, the NOLOAD .mh region, the bench payload's placement, no AMO anywhere
    for needle in ("PASS mh_layout: mh_sec_rnmi_exc 0x8000200 = ['0x10500073', '0xffdff06f'], .mh [0x800a000, 0x800c000)",
                   "PASS mh_bench: entry 0x81df400", "PASS amo[monitor.elf]", "PASS amo[mh_bench.elf]",
                   "PASS hart0_fence_i: fence.i before hart 0's first call"):
        assert needle in out, (needle, out)


def test_crc32_tab_rule(built, tmp_path):
    """check_image's crc32_tab rule: one flipped table bit in a copy of monitor.elf is a FAIL."""
    m = _mod()
    e = m.Elf(built / "monitor.elf")
    assert m.crc32_tab_problem(e) == ""
    sym = e.sym("mon_crc32_tab")
    sec = next(s for s in e.sections if s["addr"] <= sym < s["addr"] + s["size"] and s["name"] == ".rodata")
    raw = bytearray(e.data)
    raw[sec["off"] + sym - sec["addr"] + 4 * 300] ^= 0x10        # table 1, entry 44
    bad = tmp_path / "monitor.elf"
    bad.write_bytes(bytes(raw))
    for name in ("park_stub.elf", "stage0.elf"):
        shutil.copy(built / name, tmp_path / name)
    assert "1 entries differ" in m.crc32_tab_problem(m.Elf(bad))
    code, out = _run(bad, "--skip-blob", "--payloads", str(tmp_path / "none"))
    assert code == 1 and "FAIL crc32_tab" in out, out


def _link(tmp: Path, name: str, asm: str, ld_args: list[str]) -> Path:
    src = tmp / f"{name}.S"
    src.write_text(asm)
    obj = tmp / f"{name}.o"
    subprocess.run(CC + ["-o", str(obj), str(src)], check=True)
    elf = tmp / f"{name}.elf"
    subprocess.run(["ld.lld-20", *ld_args, "-o", str(elf), str(obj)], check=True)
    return elf


BAD_MONITOR = """
    .section .text.init, "ax"
    .globl _start
_start:
    csrr  t0, 0x740           /* Smrnmi mnscratch: forbidden */
    .word 0x702000f3          /* mnret pattern with rd != 0 */
    .word 0x00b5302f          /* amoadd.d: an AMO in an image whose LIM is uncached */
    .weak not_defined
    call  not_defined         /* weak undefined */
    .space 70000              /* > 48 KiB image, and past the mailbox at 0x0801_0000 */
1:  j 1b
"""

BIG_PARK = """
    .section .park, "ax"
    .globl park_stub, park_stub_end, park_spin
park_stub:
    .space 52
park_stub_end:
park_spin:
    j park_spin
"""


def test_broken_images_fail(built, tmp_path):
    mon = _link(tmp_path, "monitor", BAD_MONITOR, ["-Ttext=0x08000000", "-e", "_start"])
    _link(tmp_path, "park_stub", BIG_PARK, ["--section-start=.park=0x20010100", "-e", "park_stub"])
    shutil.copy(built / "stage0.elf", tmp_path / "stage0.elf")
    code, out = _run(mon, "--skip-blob", "--payloads", str(tmp_path / "none"))
    assert code == 1, out
    for needle in ("FAIL image_size", "FAIL csr_smrnmi[monitor.elf]", "FAIL mnret[monitor.elf]",
                   "FAIL rnmi_rearm", "FAIL undefined[monitor.elf]: not_defined", "FAIL park_stub",
                   "FAIL sections_lim", "FAIL mh_layout", "FAIL amo[monitor.elf]", "FAIL hart0_fence_i"):
        assert needle in out, (needle, out)
    assert "PASS stage0" in out


def test_hart0_fence_i_rule(built, tmp_path):
    """hart0_fence_i: a fence.i between _start and hart 0's first call passes; none, or one only after that call, fails
    (T-3.8 case 2 on silicon, 2026-10-25: the image without it let the first RNMI after E1F run a stale stage 0)."""
    m = _mod()
    e = m.Elf(built / "monitor.elf")
    fenced, call = m.fence_i_before_first_call(e, e.sym("_start"))
    assert fenced and call is not None and m.word_at(e, e.sym("_start") + 8) == m.FENCE_I  # right after the branch
    head = ".text\n.globl _start\n_start:\n    csrr t0, mhartid\n    bnez t0, 2f\n"
    for name, body, want in (("good", "    fence.i\n    call f\n", True), ("none", "    call f\n", False),
                             ("late", "    call f\n    fence.i\n", False)):
        elf = _link(tmp_path, name, head + body + "2:  j 2b\nf:  ret\n", ["-Ttext=0x08000000", "-e", "_start"])
        assert m.fence_i_before_first_call(m.Elf(elf), 0x0800_0000)[0] is want, name


def test_mnret_mnemonic_in_source_is_caught(tmp_path):
    m = _mod()
    (tmp_path / "ok.S").write_text("    .word 0x70200073   /* mnret, as a word */\n# mnret in a comment\n")
    assert m.source_mnret_uses(tmp_path) == []
    (tmp_path / "bad.S").write_text("label:  mnret\n")
    (tmp_path / "bad.c").write_text('void f(void) { __asm__ volatile("mnret"); }\n')
    assert m.source_mnret_uses(tmp_path) == ["bad.S:1", "bad.c:1"]


def test_blob_over_lim_fails(built, tmp_path):
    big = tmp_path / "big.blob"
    big.write_bytes(bytes(0x081E_0000 - 0x0801_1000 + 64))
    code, out = _run(built / "monitor.elf", "--blob", str(big))
    assert code == 1 and "FAIL weight_blob" in out, out


def test_amo_scan_and_fixed_spin_rules(built, tmp_path):
    """amo_words finds LR/SC/AMO (major opcode 0x2F) words; the X280 image has none and holds `wfi; j .` at
    MH_SEC_RNMI_EXC (regs.py and lim.ld agree on the address)."""
    m = _mod()
    e = m.Elf(built / "monitor.elf")
    assert m.amo_words(e) == [] and m.word_at(e, m.MH_SEC_RNMI_EXC) == m.WFI
    assert m.word_at(e, m.MH_SEC_RNMI_EXC + 4) == m.J_SELF and e.sym("mh_sec_rnmi_exc") == m.MH_SEC_RNMI_EXC
    sys.path.insert(0, str(RT))
    from x280 import regs
    assert (regs.MH_SEC_RNMI_EXC, regs.MH_BASE, regs.MH_BASE + regs.MH_SIZE) == (m.MH_SEC_RNMI_EXC, m.MH_BASE, m.MH_END)
    assert (regs.MHB_BASE, regs.MHB_END) == (m.MHB_BASE, m.MHB_END)
    amo = _link(tmp_path, "amo", ".text\n.globl _start\n_start:\n    lr.w t0, (a0)\n    sc.w t1, t2, (a0)\n"
                "    amoswap.w.aq t3, t4, (a1)\n1: j 1b\n", ["-Ttext=0x08000000", "-e", "_start"])
    assert len(m.amo_words(m.Elf(amo))) == 3
