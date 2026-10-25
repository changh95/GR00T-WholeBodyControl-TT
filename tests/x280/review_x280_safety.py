"""Adversarial X280 safety/correctness review of the M3 CPU lanes (PLAN.md §1.4.4, §5.2, D12, D13, R26-R28).

Why this file: each finding of the review is reproduced here against kmd.FakeDevice / the mailbox sim, so it can
be re-checked after a fix without a device. The F1-F8 tests were strict xfails until the fix lane closed them
(2026-10-25); they now pass, and each fix also has its own tests in the collected suites (test_kmd_fake.py,
test_x280d_sim.py, test_spike.py, test_mailbox_x86.py). The plain tests re-derive regs.py constants from the
primary sources.
Not collected by `pytest tests/` (name does not match test_*.py); run it explicitly:

    nice -n 19 taskset -c 0,1,8-11,16,17,24-27 bash -c 'source $RT/env.sh; simpy -m pytest $RT/tests/x280/review_x280_safety.py -q'

No device, no GPU, no ports.
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT))
REFS = RT / "third_party_refs"
FW = RT / "x280" / "fw"
HOST = RT / "x280" / "host"
# Off the real-time cores 2,3,4,6,7,12-15 and their SMT siblings (task rule; override with X280_CPUS).
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
NICE = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]

so, src = HOST / "libwcio.so", HOST / "wcio.c"
if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
    subprocess.run(NICE + ["make", "-s", "-C", str(HOST)], check=True)
_need = [FW / "build" / p for p in ("monitor.bin", "park_stub.bin", "stage0.bin", "x86/libmonitor_x86.so",
                                    "kernel/wbc_fp16.blob", "kernel/libwbc_mlp.so")]
if [p for p in _need if not p.exists()]:
    pytest.skip(f"build outputs missing: {[str(p) for p in _need if not p.exists()]}", allow_module_level=True)

from x280 import regs  # noqa: E402
from x280 import ledger as L  # noqa: E402
from x280.host import kmd as K  # noqa: E402
from x280 import x280d as D  # noqa: E402

ARC, T3 = regs.ARC_XY, regs.L2CPU_XY[3]
TAG_NOC_TRANSLATION = 40  # fw-19.12.0 lib/tenstorrent/bh_arc/telemetry.h:173; set by telemetry.c:224-228
MONITOR_REGION = (regs.IMAGE_BASE, regs.MAILBOX_BASE)  # .text..stack of the monitor (x280/fw/lim.ld:54-98)


def _src(rel: str) -> str:
    return (REFS / rel).read_text(errors="replace")


@pytest.fixture
def released(tmp_path):
    """A FakeDevice session that has done exactly what Session.boot does up to step 8 (L2CPU3 running)."""
    dev = K.FakeDevice.default_chip()
    k = K.Kmd(dev).open()
    dev.advance(1.5)
    k.read_l2cpu_reset()
    led = L.Ledger(tmp_path / "ledger.json")
    led.start_hold("review", 1)
    stub = (FW / "build" / "park_stub.bin").read_bytes()
    with k.l2cpu_batch(3):
        k.lim_init()
        crc = k.write_image(3, (FW / "build" / "monitor.bin").read_bytes())
        k.write_gp_scratch(3, stub)
        k.write_boot_vectors(3)
    k.pll4_step_to(200)
    k.release((3,), led, crc, gp_code=stub)
    k.pll4_step_to(800)
    yield dev, k
    if k.fd is not None:
        k.close()


@pytest.fixture
def hold(monkeypatch, tmp_path):
    monkeypatch.setenv("CHIP_SESSION_ID", f"review-hold-{os.getpid()}")
    return tmp_path / "ledger.json"


# --- findings ----------------------------------------------------------------------------------------------------

# REVIEW F1 (fixed; was a strict xfail): after an E1F (stage-0) boot the reset handler address stays
#   0x2001_0100, so the park stub jumps to itself instead of crt0: RNMI PARK can never return
def test_e1f_boot_then_rnmi_park_returns_to_crt0(monkeypatch, hold):
    monkeypatch.setenv("X280_E1F_APPROVED", "1")  # sim only: the chip is a FakeDevice
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot_stage0()
        lo, hi = s.chip.peek32(T3, regs.reset_vector(0)), s.chip.peek32(T3, regs.reset_vector(0) + 4)
        r = s.park(3)
        # fix (a): the vector may stay at the stage-0 stub (consumed only at reset); the park stub no longer reads it
        assert (hi << 32 | lo) in (regs.RESET_PC, regs.STAGE0_ENTRY), f"hart-0 reset handler address 0x{hi << 32 | lo:x}"
        assert r["pass"], (r, s.chip.status())
    finally:
        s.close()


# REVIEW F2 (fixed; was a strict xfail): kmd.release() performs the one-shot L2CPU_RESET RMW although
#   no LIM init, image, GP-scratch park stub, RNMI handlers or reset vectors were written
def test_release_refuses_without_boot_artefacts(tmp_path):
    dev = K.FakeDevice.default_chip()
    k = K.Kmd(dev).open()
    dev.advance(1.5)
    k.read_l2cpu_reset()
    led = L.Ledger(tmp_path / "l.json")
    led.start_hold("h", 1)
    k.pll4_step_to(200)
    try:
        with pytest.raises((K.PolicyError, K.InterlockError)):
            k.release((3,), led, None, gp_code=(FW / "build" / "park_stub.bin").read_bytes())
        assert dev.peek32(ARC, regs.L2CPU_RESET) == 0x0F
    finally:
        k.close()


# REVIEW F3 (fixed; was a strict xfail): lim_init (ctx 'lim_init') is exempt from the post-release
#   line-ownership rule: it rewrites the running image, stack and X280-owned L1/L35/L36/L39
def test_lim_init_refused_after_release(released):
    dev, k = released
    with k.l2cpu_batch(3):
        with pytest.raises(K.PolicyError):
            k.lim_init()
        with pytest.raises(K.PolicyError):
            k.wc_write_lines(regs.MB_STATUS, bytearray(64), _ctx="lim_init")


# REVIEW F4 (fixed; was a strict xfail): after release the monitor's .text/.bss/.persist/stack lines
#   [0x0800_0000, 0x0801_0000) count as host-owned; kmd accepts host stores into a live monitor
def test_monitor_region_not_host_writable_after_release(released):
    dev, k = released
    with k.l2cpu_batch(3):
        for a in (0x0800_0000, 0x0800_3B00, 0x0800_59C0, 0x0800_FFC0):  # .text, .bss, .persist, stack top line
            with pytest.raises(K.PolicyError):
                k.wc_write_lines(a, bytearray(64))
        with pytest.raises(K.PolicyError):
            k.uc_write32(0x0800_FFF0, 0)


# REVIEW F4b (fixed; was a strict xfail): `x280d load --dst` into the monitor region writes LIM before
#   the monitor's own range check (monitor.c cmd_load) refuses it
def test_admin_load_into_monitor_region_writes_nothing(hold):
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot()
        n = len(s.k.oplog)
        with pytest.raises((D.X280Error, K.KmdError)):
            D.run_admin(s, "load", data=bytes(256), dst=0x0800_1000)  # one digit off WEIGHTS_BASE 0x0801_1000
        hits = [str(o) for o in s.k.oplog[n:] if o.kind in ("WC64", "W32") and o.addr is not None
                and MONITOR_REGION[0] <= o.addr < MONITOR_REGION[1]]
        assert not hits, hits
    finally:
        s.close()


# REVIEW F5 (fixed; was a strict xfail): pll4_step_to writes a precomputed sequence blindly; a CMFW
#   relock mid-walk (bh_power.c:32-57 -> clock_control_tt_bh.c:316-358) is neither detected nor
#   stopped, the running hart is overclocked and the final read-back still passes
def test_pll4_walk_stops_on_a_concurrent_relock(tmp_path):
    dev = K.FakeDevice.default_chip()
    k = K.Kmd(dev).open()
    dev.advance(1.5)
    k.read_l2cpu_reset()
    led = L.Ledger(tmp_path / "l.json")
    led.start_hold("h", 1)
    stub = (FW / "build" / "park_stub.bin").read_bytes()
    with k.l2cpu_batch(3):
        k.lim_init()
        crc = k.write_image(3, (FW / "build" / "monitor.bin").read_bytes())
        k.write_gp_scratch(3, stub)
        k.write_boot_vectors(3)
    k.pll4_step_to(200)
    k.release((3,), led, crc, gp_code=stub)
    seen, n = [], [0]

    def relock_at_step_10():
        n[0] += 1
        if n[0] == 10:  # CMFW re-programs PLL4 from its cached settings (fbdiv 64, postdivs 1)
            dev.poke32(ARC, regs.PLL4_CNTL_1, regs.PLL4_CNTL_1_DEFAULT)
            dev.poke32(ARC, regs.PLL4_CNTL_5, regs.PLL4_CNTL_5_DEFAULT)
        seen.append(max(regs.pll4_mhz(dev.peek32(ARC, regs.PLL4_CNTL_1), dev.peek32(ARC, regs.PLL4_CNTL_5),
                                          dev.peek32(ARC, regs.PLL4_USE_POSTDIV))[1]))
    k.step_sleep = relock_at_step_10
    raised = False
    try:
        k.pll4_step_to(800)
    except (K.InterlockError, K.PolicyError):
        raised = True
    finally:
        k.close()
    assert raised or max(seen) <= 800.0, f"L2CPUCLK reached {max(seen):.0f} MHz with the harts running"


# REVIEW F6 (fixed; was a strict xfail): the translated (8,29) cross-read is issued without checking
#   CMFW's TAG_NOC_TRANSLATION; with translation off it targets a non-existent NoC row (R28)
def test_translated_cross_read_needs_noc_translation_enabled():
    dev = K.FakeDevice.default_chip()
    data = dev.peek32(ARC, regs.TELEMETRY_DATA_PTR)
    dev.poke32(ARC, data + 4 * TAG_NOC_TRANSLATION, 0)
    k = K.Kmd(dev, cross_read=True).open()
    dev.advance(1.5)
    try:
        with k.l2cpu_batch(3):
            with pytest.raises((K.InterlockError, K.PolicyError)):
                k.read_ccache_config(3, translated=True)
    finally:
        k.close()


# REVIEW F7 (fixed; was a strict xfail): x280d e0 reads the hart-0 Bus-Error Unit page while the harts
#   are in reset; not in PLAN T-3.E0's register list and not covered by any primary source
def test_e0_reads_no_per_hart_device_before_release(hold):
    s = D.Session("sim", ledger_path=hold, cross_read=True, hart=False)
    try:
        s.open()
        s.mode = "no_release"
        s.e0()
        assert s.chip.peek32(ARC, regs.L2CPU_RESET) == 0x0F
        beu = [str(o) for o in s.k.oplog if o.kind == "R32" and o.addr is not None
               and regs.beu_base(0) <= o.addr < regs.beu_base(3) + 0x1000]
        assert not beu, beu[:3]
    finally:
        s.close()


# REVIEW F8 (fixed; was a strict xfail): attach never looks at the RNMI trigger word; a hart left in the
#   park stub by a dead x280d (bit still set) is not resumed and attach fails
def test_attach_resumes_a_hart_left_in_the_park_stub(hold):
    s1 = D.Session("sim", ledger_path=hold)
    s1.boot()
    with s1.batch(3):
        s1.k.rnmi_set(3, 1)  # x280d dies inside park() before it clears the bit
    chip = s1.chip
    time.sleep(0.05)
    s1.close(keep_chip=True)
    s2 = D.Session("sim", ledger_path=hold, chip=chip)
    try:
        r = s2.attach()
        assert r["tiles"]["3"]["n_reentry"] == 1
    finally:
        s2.close()


# --- primary-source cross-checks (these pass) --------------------------------------------------------------------

def test_reset_unit_pll_and_arc_constants_match_fw_and_kmd():
    init_h = _src("fw-19.12.0/lib/tenstorrent/bh_arc/init.h")
    assert int(re.search(r"RESET_UNIT_L2CPU_RESET_REG_ADDR\s+(0x[0-9A-Fa-f]+)", init_h)[1], 16) == regs.L2CPU_RESET
    assert int(re.search(r"RESET_UNIT_DDR_RESET_REG_ADDR\s+(0x[0-9A-Fa-f]+)", init_h)[1], 16) == regs.DDR_RESET
    pll = _src("fw-19.12.0/lib/tenstorrent/bh_arc/pll.c")
    base1 = int(re.search(r"PLL_0_CNTL_PLL_CNTL_1_REG_ADDR\s+(0x[0-9A-Fa-f]+)", pll)[1], 16)
    base5 = int(re.search(r"PLL_0_CNTL_PLL_CNTL_5_REG_ADDR\s+(0x[0-9A-Fa-f]+)", pll)[1], 16)
    upd = int(re.search(r"PLL_0_CNTL_USE_POSTDIV_REG_ADDR\s+(0x[0-9A-Fa-f]+)", pll)[1], 16)
    step = int(re.search(r"#define PLL_CNTL_REG_OFFSET\s+(0x[0-9A-Fa-f]+)", pll)[1], 16)
    assert (base1 + 4 * step, base5 + 4 * step, upd + 4 * step) == (regs.PLL4_CNTL_1, regs.PLL4_CNTL_5,
                                                                    regs.PLL4_USE_POSTDIV)
    p4 = pll[pll.index("/* PLL4 - L2CPUCLK0,1,2,3 */"):]
    refdiv, postdiv, fbdiv = (int(x) for x in re.search(
        r"\.f\.refdiv = (\d+), \.f\.postdiv = (\d+), \.f\.fbdiv = (\d+)", p4).groups())
    pds = [int(x) for x in re.findall(r"\.f\.postdiv[0-3] = (\d+)", p4)[:4]]
    # CNTL_1: refdiv/fbdiv as the bh_arc source; its postdiv 0 is the source value, silicon (S1c) reads postdiv 1,
    # which CMFW's PLL4 devicetree node carries; CMFW ignores the internal postdiv (pll.c:456-461)
    assert regs.pll_cntl1(refdiv, postdiv, fbdiv) == regs.PLL4_CNTL_1_SOURCE and postdiv == 0
    assert regs.PLL4_CNTL_1_DEFAULT == regs.pll_cntl1(refdiv, 1, fbdiv)
    assert "/* 3. Internal postdiv - PLL_CNTL_1.postdiv */" in pll[pll.index("/* What we don't support: */"):]
    dts = _src("fw-19.12.0/dts/vendor/tenstorrent/tt_blackhole_smc.dtsi")
    node = dts[dts.index("pll4: pll@80020500"):]
    node = node[:node.index("};")]
    dt = {k: int(re.search(rf"\b{k} = <(\d+)>", node)[1]) for k in ("refdiv", "postdiv", "fbdiv")}
    assert regs.pll_cntl1(dt["refdiv"], dt["postdiv"], dt["fbdiv"]) == regs.PLL4_CNTL_1_DEFAULT
    assert [int(x) for x in re.search(r"post_divs = <([\d ]+)>", node)[1].split()] == pds
    assert "&pll4 {\n\tstatus = \"okay\";" in _src("fw-19.12.0/boards/tenstorrent/tt_blackhole/"
                                                   "tt_blackhole_tt_blackhole_smc.dts")
    assert regs.pll_cntl5(pds) == regs.PLL4_CNTL_5_DEFAULT
    bh = _src("tt-kmd-2.11.0/blackhole.c")
    assert (int(re.search(r"#define ARC_X (\d+)", bh)[1]), int(re.search(r"#define ARC_Y (\d+)", bh)[1])) == ARC
    assert int(re.search(r"#define ARC_MSI_FIFO (0x[0-9A-Fa-f]+)", bh)[1], 16) == regs.ARC_MSI_FIFO
    tel = _src("fw-19.12.0/lib/tenstorrent/bh_arc/telemetry.h")
    assert int(re.search(r"TAG_TIMER_HEARTBEAT (\d+)", tel)[1]) == regs.TAG_TIMER_HEARTBEAT
    assert int(re.search(r"TAG_ENABLED_L2CPU (\d+)", tel)[1]) == regs.TAG_ENABLED_L2CPU
    assert int(re.search(r"TAG_NOC_TRANSLATION (\d+)", tel)[1]) == TAG_NOC_TRANSLATION


def test_l2cpu_coordinates_and_x280_addresses_match_isa_and_fw():
    readme = _src("tt-isa-documentation/BlackholeA0/L2CPUTile/README.md")
    rows = re.findall(r"\|(\d)\|1\|Reset bit for CPUs \d+-\d+ \(L2CPU tile at NoC #0 coordinates X=(\d+),Y=(\d+)\)\|",
                      readme)
    assert {int(b) - 4: (int(x), int(y)) for b, x, y in rows} == regs.L2CPU_XY
    assert "`0x80030014`" in readme
    noc = _src("fw-19.12.0/lib/tenstorrent/bh_arc/noc_init.c")
    ys = [int(v) for v in re.search(r"kL2CpuNoc0Y\[\] = \{([^}]*)\}", noc)[1].split(",")]
    assert {i: (8, y) for i, y in enumerate(ys)} == regs.L2CPU_XY
    assert "memcpy(noc0.translate_table_y + 26, kL2CpuNoc0Y" in noc  # (8, 26+i) -> (8, kL2CpuNoc0Y[i])
    mm = _src("tt-isa-documentation/BlackholeA0/L2CPUTile/MemoryMap.md")
    assert "an offset of `0xFFFF_F7FE_DFF0_0000` should be added" in mm and regs.XP_HIGH_OFFSET == 0xFFFF_F7FE_DFF0_0000
    for lit in ("`0x0000_2001_0000` to `0x0000_2001_001F`|4x 8 B|Per-hart reset handler address",
                "`0x0000_2001_0100` to `0x0000_2001_013F`|64 B|General purpose scratch memory",
                "`0x0000_2001_0414` to `0x0000_2001_0417`|4 B|All harts [RNMI]",
                "`0x0000_0800_0000` to `0x0000_081F_FFFF`|2 MiB|[L3 as uncached scratchpad (LIM)]",
                "`0x0000_0170_0000` to `0x0000_0170_0FFF`|4 KiB|Hart 0 Bus-Error Unit"):
        assert lit in mm, lit
    rn = _src("tt-isa-documentation/BlackholeA0/L2CPUTile/RNMIs.md")
    assert "|`0x0000_2001_0418`|47 bits|Hart 0 RNMI trap handler address|" in rn
    assert "|`0x0000_2001_0420`|47 bits|Hart 0 RNMI exception trap handler address|" in rn
    assert "(encoding `0x70200073`)" in rn and "`0x350` through `0x353`" in rn
    assert regs.rnmi_handler(1) - regs.rnmi_handler(0) == 16 and regs.rnmi_exc_handler(0) - regs.rnmi_handler(0) == 8
    ca = _src("tt-isa-documentation/BlackholeA0/L2CPUTile/Caches.md")
    assert "`0x0000_0201_0008`|`CCACHE0_WAYENABLE`" in ca and regs.CCACHE0_WAYENABLE == 0x0201_0008
    assert "write `0x15811` to `L2PF1_BASIC_CONTROL` and `0x38c84e` to `L2PF1_USER_CONTROL`" in ca
    assert "`0x0000_0203_0000 + HART_ID * 0x2000`" in ca and regs.l2pf_base(3) == 0x0203_6000
    assert "the other 15x 128 KiB are configured as uncached scratchpad (LIM)" in ca
    assert regs.LIM_SIZE == 15 * 128 * 1024 and regs.LIM_END == 0x081E_0000


def test_noc_cleanup_struct_and_power_semantics_match_chardev():
    ch = _src("tt-kmd-2.11.0/chardev.c")
    assert "bool power_aware = file->f_flags & O_APPEND;" in ch  # chardev.c:954
    assert "if (!power_aware && !tt_dev->detached && !tt_dev->needs_hw_init) {" in ch  # chardev.c:1017
    assert "tt_cdev_release_noc_cleanup(priv);\n\ttenstorrent_memory_cleanup(priv);" in ch  # cleanup before TLB/power
    hdr = _src("tt-kmd-2.11.0/ioctl.h")
    body = re.search(r"struct tenstorrent_set_noc_cleanup \{([^}]*)\};", hdr)[1]
    assert [ln.split()[-1].rstrip(";") for ln in body.strip().splitlines()] == [n for n, _ in K.SetNocCleanup._fields_]
    import ctypes
    assert ctypes.sizeof(K.SetNocCleanup) == 32 and K.SetNocCleanup.addr.offset == 16 and K.SetNocCleanup.data.offset == 24


# --- harts 1-3 (the 4-hart bench, S3b; x280/fw/mh.h, mh.c, mh_bench.c) --------------------------------------------------

MH_SOURCES = [FW / n for n in ("mh.c", "mh.h", "mh_bench.c", "mh_bench.ld", "asm.h", "spike_script_mh.c")]
HEX_RE = re.compile(r"0x[0-9A-Fa-f_]+")


def _mh_hard_session(monkeypatch, hold, harts):
    monkeypatch.setenv("X280_E1F_APPROVED", "1")  # sim only: the chip is a FakeDevice
    monkeypatch.setenv("X280_MH_APPROVED", "1")
    s = D.Session("sim", ledger_path=hold, harts=harts)
    s.boot_stage0()
    return s


# MH-1: an RNMI on bits 1-3 with the boot's GP-scratch (park stub) vectors strands harts 1-3 until a chip reset (stale
#   stage-0 line in their L1I after E1F); kmd must refuse it on every standard boot, x280d must not even try
def test_no_rnmi_on_harts_1_3_with_gp_scratch_vectors(monkeypatch, hold):
    s = _mh_hard_session(monkeypatch, hold, 1)
    try:
        assert s.chip.peek32(T3, regs.rnmi_handler(1)) == regs.PARK_STUB
        with s.batch(3):
            for m in (0b0010, 0b0100, 0b1000, 0b1110, 0b1111):
                with pytest.raises(K.PolicyError):
                    s.k.rnmi_set(3, m)
        assert not s.harts_adopt(3)["pass"]
        assert not [o for o in s.k.oplog if o.kind == "W32" and o.addr == regs.RNMI_TRIGGER]
    finally:
        s.close()


# MH-2: a secondary must never store into hart 0's or the host's lines: only its own stat/aux/private/result lines change
def test_no_secondary_store_to_hart0_or_mailbox_lines(monkeypatch, hold):
    s = _mh_hard_session(monkeypatch, hold, 4)
    try:
        with s.batch(3):
            s.load_blob(3)
        assert s.harts_adopt(3)["pass"]
        hart = s.chip.hart(3)
        hart.stop()                                   # hart 0 (and harts 1-3) stopped: take a LIM snapshot
        before = bytes(s.chip.peek(T3, regs.LIM_BASE, regs.LIM_SIZE))
        for h in (1, 2, 3):                           # harts 1-3 alone, re-adopted: they idle and heartbeat
            assert hart.lib.mon_x86_secondary_start(h, (1 << 63) | 2, regs.STAGE0_SEC_WFI) == 0
        time.sleep(0.2)
        hart.lib.mon_x86_secondary_stop_all()
        after = bytes(s.chip.peek(T3, regs.LIM_BASE, regs.LIM_SIZE))
        changed = {(regs.LIM_BASE + i) & ~63 for i in range(len(before)) if before[i] != after[i]}
        allowed = {regs.mh_stat(h) for h in (1, 2, 3)} | {regs.mh_aux(h) for h in (1, 2, 3)}
        allowed |= {regs.MH_BASE + 0x800 * h + 64 * j for h in (1, 2, 3) for j in range(4)}   # the job guards
        assert changed and changed <= allowed, sorted(hex(a) for a in changed - allowed)
    finally:
        s.close()


# MH-3: the bench payload validates its working set inside [0x0801_1000, 0x081E_0000) and never over itself
def test_mh_bench_refuses_working_sets_outside_the_payload_area():
    src = (FW / "mh_bench.c").read_text()
    assert "base < MB_PAYLOAD_BASE || base + len > MB_LIM_END" in src
    assert "(base < MHB_END && MHB_BASE < base + len)" in src
    hdr = (FW / "mailbox.h").read_text()
    assert "#define MB_PAYLOAD_BASE    0x08011000u" in hdr and "#define MB_LIM_END         (MB_LIM_BASE + MB_LIM_SIZE)" in hdr


# MH-4: no constant or access to WAYENABLE 0x0201_0008, WAYMASK 0x0201_0800.., the Zero Device 0x0A00_0000, GDDR
#   (0x3000_0000 and up) or the X280 TLB window configuration (0x2000_0000-0x2000_0F7F) in the multi-hart sources
def test_mh_sources_hold_no_forbidden_address():
    allowed_data = {0x8FFFFFFF, 0xFFFFFFFF, 0x4C54434D, 0x5441544D, 0xC0001073, 0x5EED, 0xFFFF}   # masks, magics, insn
    bad = []
    for p in MH_SOURCES:
        for i, line in enumerate(p.read_text().splitlines(), 1):
            for lit in HEX_RE.findall(line):
                v = int(lit.replace("_", ""), 16)
                if v in allowed_data:
                    continue
                if (regs.CCACHE0_WAYENABLE <= v < regs.CCACHE0_WAYENABLE + 8
                        or regs.CCACHE0_WAYMASK0 <= v < regs.CCACHE0_WAYMASK0 + 38 * 8
                        or 0x0A00_0000 <= v < 0x0A20_0000 or v >= 0x3000_0000
                        or regs.XP_TLBCFG_PHYS_LO <= v <= regs.XP_TLBCFG_PHYS_HI and v != regs.XP_TLBCFG_PHYS_LO):
                    bad.append(f"{p.name}:{i}: {lit}")
    assert not bad, bad


# MH-5: tile 3 only; the RNMI bits of harts 1-3 never together with hart 0's PARK bit
def test_mh_is_l2cpu3_only_and_never_combines_with_park(monkeypatch, hold):
    with pytest.raises(ValueError, match="L2CPU3"):
        D.Session("sim", ledger_path=hold, tiles=(2,), harts=4)
    s = _mh_hard_session(monkeypatch, hold, 4)
    try:
        with s.batch(3):
            with pytest.raises(K.PolicyError, match="bit 0"):
                s.k.rnmi_set(3, 0b0011)
        assert s.harts_adopt(3)["pass"] and s.park(3)["pass"]          # PARK of hart 0 still works next to them
        with s.batch(3):
            assert s.mh_ping(3)["done"] == 0xE                         # they kept running through hart 0's PARK
    finally:
        s.close()
