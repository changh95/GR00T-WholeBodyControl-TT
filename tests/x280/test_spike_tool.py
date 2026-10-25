"""WP-3.2b tool checks: the user-space spike build and x280/sim/run_spike.py (PLAN.md WP-3.2b).

Tiny images from x280/sim/selftest/ are built with clang-20 + ld.lld-20 at 0x0800_0000 and run on spike with
rv64gcv_zfh_zba_zbb at VLEN 512: HTIF exit codes (including codes > 255), console capture, timeout, instruction
counts, an RVV vle32/vfmacc loop, the illegal-instruction trap dump, which X280-only CSRs trap, and the
GP-scratch / BEU regions. The firmware's own spike run (T-3.2b, test_spike.py) is built on top of this.
Run: `simpy -m pytest $RT/tests/x280/test_spike_tool.py -q` (no device, no GPU, no network).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RT = Path(__file__).resolve().parents[2]
SIM = RT / "x280" / "sim"
SELFTEST = SIM / "selftest"
CLANG, LLD, OBJCOPY, OBJDUMP = "clang-20", "ld.lld-20", "llvm-objcopy-20", "llvm-objdump-20"
CFLAGS = ["--target=riscv64-unknown-elf", "-march=rv64gcv_zfh_zba_zbb", "-mabi=lp64d", "-mcmodel=medany",
          "-O2", "-ffreestanding", "-fno-builtin", "-nostdlib", "-ffp-contract=off", "-Wall", "-Wextra",
          "-Werror", f"-I{SELFTEST}"]


def _load_run_spike():
    spec = importlib.util.spec_from_file_location("run_spike", SIM / "run_spike.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_spike"] = mod   # dataclasses resolves annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


rs = _load_run_spike()

pytestmark = pytest.mark.skipif(
    not rs.SPIKE_BIN.exists() or not all(shutil.which(t) for t in (CLANG, LLD, OBJCOPY, OBJDUMP)),
    reason="needs x280/sim/build_spike.sh output and clang-20/ld.lld-20/llvm-objcopy-20/llvm-objdump-20")


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """build(name, *defines) -> ELF of selftest/<name>.c + crt0.S + rt.c, cached per (name, defines)."""
    out = tmp_path_factory.mktemp("spike_selftest")
    cache: dict[tuple, Path] = {}

    def _build(name: str, *defines: str) -> Path:
        key = (name, defines)
        if key not in cache:
            tag = "_".join([name, *(d.replace("=", "") for d in defines)])
            objs = []
            for src in ("crt0.S", "rt.c", f"{name}.c"):
                obj = out / f"{tag}.{src}.o"
                subprocess.run([CLANG, *CFLAGS, *(f"-D{d}" for d in defines), "-c", str(SELFTEST / src),
                                "-o", str(obj)], check=True, capture_output=True, text=True)
                objs.append(str(obj))
            elf = out / f"{tag}.elf"
            subprocess.run([LLD, "-T", str(SELFTEST / "lim.ld"), "-o", str(elf), *objs],
                           check=True, capture_output=True, text=True)
            cache[key] = elf
        return cache[key]
    return _build


def _accesses(stdout: str) -> dict[str, str]:
    return dict(re.findall(r"^ACCESS (\S+) (trap|ok)", stdout, re.M))


# --- configuration --------------------------------------------------------------------------------------

def test_isa_and_memory_arguments():
    assert rs.isa_string() == "rv64gcv_zfh_zba_zbb_zvl512b"     # --varch vlen:512,elen:64 -> Zvl512b
    assert rs.isa_string("rv64gcv_zfh_zba_zbb", None) == "rv64gcv_zfh_zba_zbb"
    with pytest.raises(ValueError):
        rs.isa_string(varch="vlen:512,elen:32")                  # V implies Zve64d
    with pytest.raises(ValueError):
        rs.isa_string(varch="vlen:500,elen:64")
    with pytest.raises(ValueError):
        rs.isa_string("rv64gcv_zvl256b", "vlen:512,elen:64")
    assert rs.mem_arg(rs.memory_regions(x280_regions=False)) == "-m0x8000000:0x1e0000"   # the plan's region
    assert rs.mem_arg(rs.memory_regions()) == "-m0x8000000:0x1e0000,0x20010000:0x1000,0x1700000:0x4000"
    gp_lo, gp_n = rs.GP_SCRATCH
    assert any(lo <= gp_lo and gp_lo + gp_n <= lo + n for lo, n in rs.memory_regions())
    assert any(lo <= 0x0170_0000 < lo + n for lo, n in rs.memory_regions())


def test_build_script_idempotent_and_recorded():
    """A second build_spike.sh run rebuilds nothing and records the pinned SHAs."""
    proc = subprocess.run(["bash", str(SIM / "build_spike.sh")], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "build " not in proc.stderr, proc.stderr          # every step skipped by its stamp
    shas = (SIM / "SHAS.txt").read_text()
    pinned = re.search(r"SPIKE_SHA:-([0-9a-f]{40})", (SIM / "build_spike.sh").read_text()).group(1)
    assert f"riscv-isa-sim {pinned}" in shas
    assert re.search(r"^dtc [0-9a-f]{40} ", shas, re.M)
    head = subprocess.run(["git", "-C", str(SIM / "build" / "riscv-isa-sim"), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == pinned
    assert subprocess.run([str(rs.SPIKE_BIN.parent / "dtc"), "--version"], capture_output=True).returncode == 0


# --- HTIF exit codes, console, timeout, counts ----------------------------------------------------------

@pytest.mark.parametrize("code", [0, 1, 42, 255, 256, 4660])
def test_htif_exit_code(build, code):
    res = rs.run(build("t_exit", f"EXIT_CODE={code}"), timeout_s=30)
    assert not res.timed_out
    assert res.exit_code == code, res.stderr                  # 256 and 4660 only survive via the stderr parse
    assert res.returncode == code & 0xFF
    assert res.ok == (code == 0)
    assert "hello from x280-sim\n" in res.stdout
    assert f" exit={code}\n" in res.stdout


def test_instruction_count_matches_minstret(build):
    res = rs.run(build("t_exit"), count_insns=True, timeout_s=60)
    assert res.ok, res.stderr
    m = re.search(r"minstret=(\d+) mcycle=(\d+)", res.stdout)
    minstret, mcycle = int(m.group(1)), int(m.group(2))
    assert 0 <= mcycle - minstret <= 2                         # spike: one instruction per cycle
    assert res.insns_by_hart == {0: res.insns}
    assert res.insns_image == res.insns - 5                    # spike's 5-instruction boot ROM stub
    # The log covers everything up to the exit poll: more than minstret at the print, bounded by the HTIF
    # console/exit polling (one tohost check per 5000-instruction batch per character).
    assert minstret < res.insns < minstret + 200_000
    assert res.traps == []


def test_timeout_kills_spike(build):
    elf = build("t_exit", "SPIN_FOREVER")
    res = rs.run(elf, timeout_s=2)
    assert res.timed_out and res.exit_code is None and not res.ok
    assert "hello from x280-sim" in res.stdout                 # console captured up to the kill
    assert res.wall_s < 10
    left = [p for p in os.listdir("/proc") if p.isdigit()
            and str(elf).encode() in _cmdline(p)]
    assert left == [], "spike survived its timeout"


def _cmdline(pid: str) -> bytes:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return b""


def test_rejects_elf_without_htif(build, tmp_path):
    stripped = tmp_path / "nohtif.elf"
    subprocess.run([OBJCOPY, "-N", "tohost", "-N", "fromhost", str(build("t_exit")), str(stripped)], check=True)
    with pytest.raises(ValueError, match="tohost"):
        rs.run(stripped)


def test_missing_region_is_reported(build):
    elf = build("t_gpx")
    with pytest.raises(ValueError, match="outside every memory region"):
        rs.run(elf, x280_regions=False)
    res = rs.run(elf, x280_regions=False, check_segments=False, timeout_s=10)
    assert res.exit_code is None and not res.timed_out
    assert "0x20010000 is invalid" in res.stderr


# --- RVV at VLEN 512 ------------------------------------------------------------------------------------

def test_rvv_vle32_vfmacc_vlen512(build):
    elf = build("t_rvv")
    dis = subprocess.run([OBJDUMP, "-d", str(elf)], capture_output=True, text=True, check=True).stdout
    for mnemonic in ("vle32.v", "vfmacc.vv", "vse32.v", "vsetvli"):
        assert mnemonic in dis
    res = rs.run(elf, timeout_s=30)
    assert res.ok, res.stdout + res.stderr
    assert "vlenb=64 vlmax_e32m1=16 vlmax_e32m4=64 vlmax_e32m8=128" in res.stdout
    assert "strips=56 last_vl=4 mismatch=0" in res.stdout      # bit-exact vs the scalar fmadd.s loop


def test_rvv_vlen_is_configurable(build):
    """Negative control: at VLEN 128 the same image sees vlenb 16 and fails its geometry check."""
    res = rs.run(build("t_rvv"), varch="vlen:128,elen:64", timeout_s=30)
    assert res.exit_code == 0x201
    assert "vlenb=16 " in res.stdout


# --- traps ----------------------------------------------------------------------------------------------

def test_illegal_instruction_trap_dump(build):
    elf = build("t_illegal")
    site = rs.elf_info(elf).symbols["illegal_site"]
    res = rs.run(elf, trace_traps=True, timeout_s=30)
    assert res.exit_code == 0xE00 + 2                          # TRAP_EXIT_BASE + mcause (illegal instruction)
    assert (f"TRAP mcause=0x{2:016x} mepc=0x{site:016x} mtval=0x{0xc0001073:016x}") in res.stdout
    assert "after illegal" not in res.stdout
    assert res.traps == [{"hart": 0, "kind": "exception", "name": "trap_illegal_instruction",
                          "epc": site, "tval": 0xc0001073}]   # spike's own record agrees with the dump


def test_illegal_instruction_resume(build):
    res = rs.run(build("t_illegal", "RESUME"), timeout_s=30)
    assert res.ok, res.stdout
    assert "after illegal traps=1 mcause=0x0000000000000002" in res.stdout


# Measured on spike 2c300cd: every access to the X280-only CSRs traps (mcause 2), so the firmware must
# stub csrw 0x7c1, csrsi 0x353 and any 0x350-0x353 access (and mnret) behind SPIKE_HOST.
X280_TRAPPING = ["csrr_0x350", "csrr_0x351", "csrr_0x352", "csrr_0x353", "csrr_0x7c1",
                 "csrw_0x350", "csrw_0x351", "csrw_0x352", "csrw_0x353", "csrw_0x7c1",
                 "csrsi_0x353_8", "csrwi_0x7c1_0", "mnret"]
STANDARD_SMRNMI = ["csrr_0x740", "csrr_0x741", "csrr_0x742", "csrr_0x744"]
CONTROLS_OK = ["csrr_misa", "csrr_mhartid", "csrr_mcycle", "csrr_minstret", "csrr_vlenb"]


def test_x280_only_csrs_trap(build):
    res = rs.run(build("t_csr"), trace_traps=True, timeout_s=30)
    assert res.ok, res.stdout + res.stderr
    acc = _accesses(res.stdout)
    assert {k for k, v in acc.items() if v == "trap"} == set(X280_TRAPPING + STANDARD_SMRNMI + ["csrr_time"])
    assert {k for k, v in acc.items() if v == "ok"} == set(CONTROLS_OK)
    assert len(re.findall(r" trap mcause=0x0000000000000002$", res.stdout, re.M)) == len(acc) - len(CONTROLS_OK)
    assert all(t["name"] == "trap_illegal_instruction" for t in res.traps)
    misa = int(re.search(r"misa=0x([0-9a-f]+)", res.stdout).group(1), 16)
    for ext in "IMAFDCV":
        assert misa >> (ord(ext) - ord("A")) & 1, ext


def test_csr_0x350_is_miselect_under_smcsrind(build):
    """Why the ISA string must stay without Smcsrind: 0x350 would stop trapping and silently be miselect."""
    res = rs.run(build("t_csr"), isa="rv64gcv_zfh_zba_zbb_smcsrind", timeout_s=30)
    acc = _accesses(res.stdout)
    assert acc["csrr_0x350"] == acc["csrw_0x350"] == "ok"
    assert acc["csrr_0x351"] == acc["csrr_0x353"] == acc["csrr_0x7c1"] == "trap"


# --- X280 regions: GP scratch, reset vectors, RNMI trigger word, BEU ------------------------------------

def test_gp_scratch_stub_and_beu_single_hart(build):
    elf = build("t_gpx")
    info = rs.elf_info(elf)
    assert info.symbols["gp_stub"] == rs.GP_SCRATCH[0]
    res = rs.run(elf, timeout_s=30)
    assert res.ok, res.stdout + res.stderr
    assert "beu0_initial=0x0000000000000000 beu0_readback=0x000000005a5a5a5a" in res.stdout
    assert f"resetvec0=0x{info.symbols['resume_point']:016x}" in res.stdout
    assert int(re.search(r"stub_bytes=(\d+)", res.stdout).group(1)) <= 48
    assert res.stdout.endswith("resumed via gp stub\n")


def test_gp_scratch_park_release_two_harts(build):
    """Hart 0 spins in the stub on bit 0 of 0x2001_0414 until hart 1 (playing the host) clears it."""
    res = rs.run(build("t_gpx", "TWO_HARTS"), harts=2, count_insns=True, timeout_s=30)
    assert res.ok, res.stdout + res.stderr
    assert res.stdout.endswith("resumed via gp stub\n")
    assert set(res.insns_by_hart) == {0, 1}


def test_store_audit_two_harts(build):
    """run(audit_stores=True): spike -l --log-commits, every committed LIM store's 64 B line with the harts that made it.
    In t_gpx TWO_HARTS hart 0 (bss, its stack) and hart 1 (its own 4 KiB stack) never store into the same line, and
    the trigger word 0x2001_0414 hart 1 clears is outside LIM, so it is not recorded."""
    res = rs.run(build("t_gpx", "TWO_HARTS"), harts=2, audit_stores=True, timeout_s=60)
    assert res.ok, res.stdout + res.stderr
    lines = res.stores_by_line
    assert lines and all(rs.LIM[0] <= a < rs.LIM[0] + rs.LIM[1] and a % 64 == 0 for a in lines)
    h0 = {a for a, hs in lines.items() if 0 in hs}
    h1 = {a for a, hs in lines.items() if 1 in hs}
    assert h0 and h1 and not h0 & h1, sorted(hex(a) for a in h0 & h1)
    assert res.amos == [] and all((h, a) in res.store_pcs for a, hs in lines.items() for h in hs)


def test_store_audit_parser_records_amos_and_store_lines():
    """The commit-line parser itself: a store, a load (not a store), an AMO (major opcode 0x2F) and a store outside LIM."""
    import threading
    r = rs._LogReader("/nonexistent", threading.Event(), audit=True)
    r._line(b"core   2: 3 0x0000000008000100 (0x00b53023) mem 0x0000000008010040 0x0000000000000007")    # sd
    r._line(b"core   2: 3 0x0000000008000104 (0x00053283) x5  0x0000000000000000 mem 0x0000000008010080")  # ld
    r._line(b"core   1: 3 0x0000000008000108 (0x00b5302f) x0  0x0000000000000000 mem 0x0000000008010100 "
            b"mem 0x0000000008010100 0x0000000000000001")                                                      # amoadd.d
    r._line(b"core   1: 3 0x000000000800010c (0x8082) mem 0x0000000020010414 0x0000000000000000")             # not LIM
    assert {a: sorted(h) for a, h in r.stores.items()} == {0x0801_0040: [2], 0x0801_0100: [1]}
    assert r.amos == [{"hart": 1, "pc": 0x0800_0108, "insn": 0x00B5302F}]
    assert r.store_pcs[(2, 0x0801_0040)] == 0x0800_0100


# --- dtc / DTB ------------------------------------------------------------------------------------------

def test_dtc_comes_from_the_install_prefix(build, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    res = rs.run(build("t_exit"), pin=False, timeout_s=30)
    assert res.ok, res.stderr


def test_prebuilt_dtb(build, tmp_path, monkeypatch):
    """The alternative to dtc at run time: a DTB made once and passed with --dtb."""
    spike, dtc = str(rs.SPIKE_BIN), str(rs.SPIKE_BIN.parent / "dtc")
    elf = build("t_exit")
    env = dict(os.environ, PATH=f"{rs.SPIKE_BIN.parent}:{os.environ['PATH']}")
    dts = subprocess.run([spike, f"--isa={rs.isa_string()}", rs.mem_arg(rs.memory_regions()), "--dump-dts",
                          str(elf)], capture_output=True, text=True, check=True, env=env).stdout
    dtb = tmp_path / "x280.dtb"
    subprocess.run([dtc, "-I", "dts", "-O", "dtb", "-o", str(dtb)], input=dts, text=True, check=True)
    shutil.copy(spike, tmp_path / "spike")                     # a spike with no dtc next to it ...
    monkeypatch.setenv("PATH", "/nonexistent")                 # ... and none on PATH
    res = rs.run(elf, pin=False, dtb=str(dtb), spike=tmp_path / "spike", timeout_s=30)
    assert res.ok, res.stderr


# --- CLI ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("defines,timeout,expect_rc", [((), 30, 0), (("EXIT_CODE=42",), 30, 1),
                                                        (("SPIN_FOREVER",), 2, rs.TIMEOUT_EXIT)])
def test_cli_exit_status(build, tmp_path, defines, timeout, expect_rc):
    out = tmp_path / "res.json"
    proc = subprocess.run([sys.executable, str(SIM / "run_spike.py"), str(build("t_exit", *defines)),
                           "--timeout", str(timeout), "--json", str(out)], capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == expect_rc, proc.stderr
    assert "hello from x280-sim" in proc.stdout
    data = json.loads(out.read_text())
    assert data["timed_out"] == (expect_rc == rs.TIMEOUT_EXIT)


def test_cli_explain_lists_unmodelled():
    proc = subprocess.run([sys.executable, str(SIM / "run_spike.py"), "--explain"], capture_output=True,
                          text=True, check=True)
    assert "rv64gcv_zfh_zba_zbb_zvl512b" in proc.stdout
    assert "RNMI delivery" in proc.stdout and "0x350-0x353" in proc.stdout
