"""Blackhole X280 register map for x280d: the ONLY source of addresses, NoC coordinates, register
offsets and expected values used by the host access layer (PLAN.md WP-3.3, §1.4.4, §5.2).

Every number carries a citation. Source tags (paths relative to RT = ~/experiments/GR00T-WholeBodyControl-TT):
  ISA  third_party_refs/tt-isa-documentation/BlackholeA0          (0610df6, SHAS.txt)
  KMD  third_party_refs/tt-kmd-2.11.0                             (c05f224)
  FW   third_party_refs/fw-19.12.0/lib/tenstorrent/bh_arc         (fe27c0e, = installed CMFW 19.12)
  BHL  third_party_refs/tt-bh-linux                               (9f9597d)
  LUW  third_party_refs/luwen                                     (ecef1a8)
  REF  x280/refs (SiFive ccache0/BEU layouts; URLs and sha256 in x280/refs/SOURCES.txt)
  PLAN docs/plan/vla-wbc-x280/PLAN.md (rev 2)

The write policy is data here (ALLOW, DENY) plus the pure classifier `classify_write`; the
stateful rules (one L2CPU_RESET RMW per session, the PLL4 step order, LIM init before any partial
write, pre-release-only vectors) are enforced by x280/host/kmd.py on top of it. Writes are refused
unless an ALLOW entry covers the whole access and no DENY entry overlaps it; a DENY entry is never
waived except by its one named `exception` (only the PLL block has one, see `PLL4_STEP`).
"""

from dataclasses import dataclass
from typing import Optional

# --- NoC coordinates (NoC #0) ------------------------------------------------------------------

NOC = 0  # every access uses NoC #0; the coordinates below are NoC #0 coordinates (ISA L2CPUTile/README.md:40-43)

ARC_XY = (8, 0)  # KMD blackhole.c:59-60 (ARC_X 8, ARC_Y 0); FW noc_init.c:419-420

# Tile index i = L2CPU_RESET bit 4+i. ISA L2CPUTile/README.md:40-43; BHL boot.py:25-30; FW noc_init.c:366-367
# (kL2CpuNoc0Y = {3, 9, 5, 7}).
L2CPU_XY = {0: (8, 3), 1: (8, 9), 2: (8, 5), 3: (8, 7)}

# Translated aliases (8, 26+i) -> (8, kL2CpuNoc0Y[i]): FW noc_init.c:562-568; ISA NoC/Coordinates.md:45. Used only
# for the read-only E0 cross-check (PLAN T-3.E0, R31); never written.
L2CPU_XY_TRANSLATED = {i: (8, 26 + i) for i in range(4)}

DEFAULT_TILE = 3  # PLAN §5.2 tile policy: L2CPU3 (8,7) is the default
SPARE_TILE = 2  # PLAN §5.2: L2CPU2 (8,5) only for DP4 or after a root-caused L2CPU3 failure
WRITABLE_TILES = (2, 3)  # L2CPU0/1 are last resorts needing user review (PLAN §5.2), so not writable here

# Translated DRAM tile coordinates (ISA NoC/Coordinates.md:35-42). Listed for documentation: no window may
# target any tile outside ALLOWED_TARGETS, which already excludes every DRAM tile, raw or translated.
DRAM_TRANSLATED_XY = tuple((x, y) for x in (17, 18) for y in range(12, 24))


def tile_of(xy: tuple) -> Optional[int]:
    """L2CPU tile index for raw or translated coordinates, else None."""
    for i, c in L2CPU_XY.items():
        if c == tuple(xy):
            return i
    for i, c in L2CPU_XY_TRANSLATED.items():
        if c == tuple(xy):
            return i
    return None


def is_translated(xy: tuple) -> bool:
    return tuple(xy) in L2CPU_XY_TRANSLATED.values()


# --- Host TLB windows (PCIe tile) --------------------------------------------------------------

TLB_2M = 1 << 21  # KMD blackhole.c:22-24 (202 x 2 MiB windows); ISA PCIExpressTile/HostToDeviceTLBs.md:5-9
TLB_2M_ALLOCATABLE = 201  # ids 0..200; 201 is the kernel's (KMD blackhole.c:42; HostToDeviceTLBs.md:9)
# "Strict AXI" (HostToDeviceTLBs.md:31); the KMD's own window uses it too (KMD blackhole.c:245). Strict on every
# window keeps the obs lines visible before seq_req (PLAN §1.4.4 per-tick protocol, R36).
ORDERING_STRICT = 1

# --- ARC tile (8,0) ----------------------------------------------------------------------------

RESET_UNIT_BASE = 0x8003_0000  # FW init.h:13 (RESET_UNIT_GLOBAL_RESET); BHL boot.py:77
DDR_RESET = 0x8003_0010  # FW init.h:15: GDDR controller resets -> DENY
L2CPU_RESET = 0x8003_0014  # ISA L2CPUTile/README.md:35; FW init.h:16; BHL boot.py:80, 83-84
L2CPU_RESET_TILE_RESETS = 0xF  # bits 3:0 "reserved; software should not modify" (README.md:39), set to 0xF by
#                                CMFW (FW reset.c:388-391, init.h:84-85 l2cpu_reset_n:4, l2cpu_risc_reset_n:4)
L2CPU_RESET_AT_SESSION_START = 0x0F  # PLAN §5.2 step 1 / T-3.E0; bits 31:8 read as zero (README.md:44)


def release_bit(tile: int) -> int:
    """L2CPU_RESET bit that releases tile `tile`'s harts (0 -> 1 transition): ISA README.md:40-43; BHL boot.py:80-84."""
    return 1 << (4 + tile)


# Masks the single boot RMW may OR in (PLAN §5.2 step 7: 0x80, plus 0x40 under DP4; L2CPU2 alone after a
# root-caused L2CPU3 failure).
RELEASE_MASKS = (release_bit(3), release_bit(2), release_bit(3) | release_bit(2))

SCRATCH_RAM_BASE = 0x8003_0400  # FW status_reg.h:12-14; KMD blackhole.c:61 (RESET_SCRATCH(N))


def scratch_ram(n: int) -> int:
    return SCRATCH_RAM_BASE + 4 * n  # FW status_reg.h:13-14


TELEMETRY_DATA_PTR = scratch_ram(12)  # FW status_reg.h:52; KMD blackhole.c:63 (ARC_TELEMETRY_DATA)
TELEMETRY_TABLE_PTR = scratch_ram(13)  # FW status_reg.h:60; KMD blackhole.c:62; LUW crates/luwen-api/src/chip/blackhole.rs:155
ARC_CSM_BASE = 0x1000_0000  # KMD telemetry.h:82; LUW blackhole.rs:812 (0x10000000..=0x1007FFFF)
ARC_CSM_SIZE = 1 << 19  # KMD telemetry.h:83
# Table layout: u32 version @0, u32 entry_count @4, entry_count x {u16 tag, u16 offset} @8 (FW telemetry.c:64-94);
# value of a tag = u32 at TELEMETRY_DATA_PTR + 4*offset (KMD blackhole.c:479-519, the algorithm kmd.py copies).
TELEMETRY_VERSION_MAJOR_MAX = 1  # KMD blackhole.c:489-496 rejects major > 1
TELEMETRY_MAX_ENTRIES = 1 << 16  # KMD telemetry.h:87
TAG_TIMER_HEARTBEAT = 32  # FW telemetry.h:149; incremented once per update (FW telemetry.c:549)
TAG_ENABLED_L2CPU = 37  # FW telemetry.h:164; = tile_enable.l2cpu_enabled (telemetry.c:444), bitmap 0-3 (harvesting.h:26)
# 1 only after CMFW enabled NoC translation (FW noc_init.c:590 -> telemetry.c:224-228); 0 when it is skipped
# (noc_init.c:604-610: cable-fault mode or the fwtable feature off) or torn down (noc_init.c:744), and 0 by
# zero-init (telemetry.c:447-449).
# The translated L2CPU aliases (8, 26+i) exist only when it is 1 (noc_init.c:562-568); Y = 26..29 appears only in
# the translated coordinate table (ISA NoC/Coordinates.md:17-45), so with translation off a host read there targets
# no tile and can hang (R28).
TAG_NOC_TRANSLATION = 40  # FW telemetry.h:173
TELEMETRY_PERIOD_S = 0.100  # FW telemetry.c:278, 591-592: a 100 ms snapshot, so interlocks read registers directly

# HUD chip-panel telemetry (PLAN §4.3 item 7), all read from the CMFW table in CSM (no ARC message, no new fd).
# tag -> (contract key, unit, decoder); the encodings are CMFW's own writers in FW telemetry.c update_telemetry().
TAG_VCORE = 6  # FW telemetry.h:51 "VCore voltage in millivolts"; written telemetry.c:464-466
TAG_TDP = 7  # FW telemetry.h:54 "TDP in watts"; telemetry.c:467-469 (float truncated to u32)
TAG_TDC = 8  # FW telemetry.h:57 "TDC in amperes"; telemetry.c:470-472 (float truncated to u32)
TAG_ASIC_TEMPERATURE = 11  # FW telemetry.h:66 signed 16.16; telemetry.c:473-476 via ConvertFloatToTelemetry (188-206)
TAG_AICLK = 14  # FW telemetry.h:75 MHz; telemetry.c:479-481: "lower 16 bits - current AICLK"
TAG_L2CPUCLK = (17, 18, 19, 20)  # FW telemetry.h:84-93 L2CPUCLK0..3 MHz; telemetry.c:504-522: lower 16 bits = current
TAG_INPUT_POWER = 54  # FW telemetry.h:243 "Input power in watts"; telemetry.c:541
TELEMETRY_Q16_ERROR = 0x8000_0000  # ConvertFloatToTelemetry's +-FLT_MAX marker (telemetry.c:193-195)
STATUS_TELEMETRY_TAGS = (TAG_AICLK, *TAG_L2CPUCLK, TAG_ASIC_TEMPERATURE, TAG_VCORE, TAG_TDP, TAG_TDC,
                         TAG_INPUT_POWER, TAG_TIMER_HEARTBEAT)


def decode_status_telemetry(raw: dict) -> dict:
    """Contract units from raw CMFW tag words (`raw`: tag -> u32). A tag the table does not hold stays None: a
    value is never invented. Decoders follow the CMFW writers cited at the TAG_* constants; the 16.16 temperature
    is CMFW's own inverse, ConvertTelemetryToFloat (FW telemetry.c:208-216: int32 / 65536, INT32_MIN = error)."""
    def lo16(tag):
        v = raw.get(tag)
        return None if v is None else v & 0xFFFF

    def u32(tag):
        return raw.get(tag)

    t = raw.get(TAG_ASIC_TEMPERATURE)
    temp = None if t is None or t == TELEMETRY_Q16_ERROR else (t - (1 << 32) if t & 0x8000_0000 else t) / 65536.0
    return {"aiclk_mhz": lo16(TAG_AICLK), "l2cpuclk_mhz": [lo16(g) for g in TAG_L2CPUCLK],
            "asic_temp_c": temp, "vcore_mv": u32(TAG_VCORE), "tdp_w": u32(TAG_TDP), "tdc_a": u32(TAG_TDC),
            "input_power_w": u32(TAG_INPUT_POWER), "timer_heartbeat": u32(TAG_TIMER_HEARTBEAT)}


def enabled_l2cpu_bit(tile: int) -> int:
    """Bit of ENABLED_L2CPU for `tile` (BHL boot.py:114 tests (enabled_l2cpu >> l2cpu) & 1 with l2cpu = tile index)."""
    return 1 << tile


ARC_MSI_FIFO = 0x800B_0000  # KMD blackhole.c:67: writing it runs the ARC message queue -> DENY (no ARC messages, D12)

# PLL controller (FW pll.c:22-32, 239-241: PLLn register = PLL0 register + 0x100*n).
PLL_BLOCK_LO = 0x8002_0000  # page holding PLL_CNTL_WRAPPER (0x8002_002C, _0040) and PLL0..4 (0x8002_0100..05FF)
PLL_BLOCK_HI = 0x8002_0FFF
PLL_LOCK = 0x8002_0040  # FW pll.c:30; bit n = PLLn locked (pll.c:133-139)


def pll_lock_bit(n: int) -> int:
    return 1 << n  # FW pll.c:133-139 (pll0_lock..pll4_lock)


# CNTL_0 {reset:1, pd:1, reset_lock:1, pd_bgr:1, bypass:1} (FW pll.c:34-40). A running PLL reads pd = 1 and
# bypass = 1 ("disable PLL bypass"): the last two writes of every CMFW (re)lock (FW pll.c:319, 331;
# clock_control_tt_bh.c:341, 353). During a relock it reads bypass = 0 / pd = 0.
PLL_CNTL_0_MASK = 0x1F
PLL_CNTL_0_RUNNING = 0x12
PLL4_BASE = 0x8002_0500  # FW pll.c:22 + 0x100*4 (pll.c:153, 239-241); BHL clock.py:10
PLL4_CNTL_0 = PLL4_BASE + 0x00  # FW pll.c:22, 34-40 (reset, pd, reset_lock, pd_bgr, bypass): read-only relock evidence
PLL4_CNTL_1 = PLL4_BASE + 0x04  # FW pll.c:23, 49-53 {refdiv u8, postdiv u8, fbdiv u16}; BHL clock.py:11, 33-38
PLL4_CNTL_5 = PLL4_BASE + 0x14  # FW pll.c:27, 101-106 {postdiv0..3 u8}; BHL clock.py:12, 21-24
PLL4_USE_POSTDIV = PLL4_BASE + 0x1C  # FW pll.c:29, 115-124
# CMFW defaults for PLL4 (L2CPUCLK0..3 = 800 MHz): FW pll.c:225-236 for CNTL_5 / USE_POSTDIV (both matched silicon).
# CNTL_1 is the SILICON value: S1c (2026-10-25, p150a, CMFW 19.12.0.0) read 0x0040_0102 before and after a chip
# reset (runs/sessions/20261025_050343_2067005_s1c/x280_probe.1.json, 20261025_050610_2077011_s1c_probe2/
# x280_probe.1.json). The bh_arc source writes internal postdiv 0 (FW pll.c:226 via ConfigPLLVco pll.c:246), but CMFW
# does not support the internal postdiv (pll.c:456-461) and derives every clock from refdiv, fbdiv and the external
# CNTL_5 postdivs only (pll.c:462-475, 490-493): L2CPUCLK = 50 * 64 / (2 * 2) = 800 MHz with either byte. CMFW's
# clock driver carries postdiv 1 for PLL4 (fw-19.12.0 dts/vendor/tenstorrent/tt_blackhole_smc.dtsi:296-298 ->
# drivers/clock_control/clock_control_tt_bh.c:648-650) and rewrites CNTL_1 from it (:193-194, 334) on every L2CPU
# power-on, i.e. on the keeper fd's power message (FW bh_power.c:32-44, 100-104; KMD chardev.c:986-988, 1017-1021),
# which x280d always waits out before reading PLL4. So the check is EXACT, postdiv byte included: any other value
# (the source's 0x0040_0002 too) is unexplained and stays dirty; masking the byte would accept an unknown writer.
PLL4_CNTL_1_DEFAULT = 0x0040_0102  # refdiv 2, postdiv 1, fbdiv 64 (S1c silicon)
PLL4_CNTL_1_SOURCE = 0x0040_0002  # FW pll.c:226 (postdiv 0): never read on silicon; reported dirty, never walked
PLL4_CNTL_5_DEFAULT = 0x0101_0101  # postdiv0..3 = 1
PLL4_USE_POSTDIV_MASK = 0xF  # use_postdiv0..3 = 1; compare (value & 0xF) == 0xF
PLL_VCO_MIN_MHZ, PLL_VCO_MAX_MHZ = 1600, 5000  # FW pll.c:18-19
PLL_REF_MHZ = 50  # implied by FW pll.c:226 (refdiv 2, fbdiv 64 -> "1600 MHz") and clock.py:16-17
PLL4_MAX_MHZ = 800  # L2CPUCLK bound at every walk step: the CMFW default, the fastest state either walk end has


def pll_fields(cntl1: int, cntl5: int) -> tuple:
    """(refdiv, postdiv, fbdiv, [postdiv0..3]) from the two registers (FW pll.c:49-53, 101-106)."""
    return (cntl1 & 0xFF, (cntl1 >> 8) & 0xFF, cntl1 >> 16, [(cntl5 >> (8 * i)) & 0xFF for i in range(4)])


def pll_cntl1(refdiv: int, postdiv: int, fbdiv: int) -> int:
    return (refdiv & 0xFF) | ((postdiv & 0xFF) << 8) | ((fbdiv & 0xFFFF) << 16)


def pll_cntl5(postdivs: list) -> int:
    return sum((p & 0xFF) << (8 * i) for i, p in enumerate(postdivs))


def pll_ext_postdiv(value: int, enabled: bool) -> int:
    """CMFW's effective external postdivider (FW pll.c:413-453 GetExtPostdiv): use_postdiv bit clear -> 1 (the output
    runs at the VCO frequency), field 0 -> 0 (clock disabled), 1..16 -> field + 1, above 16 -> (field + 1) * 2."""
    if not enabled:
        return 1
    if value == 0:
        return 0
    return value + 1 if value <= 16 else (value + 1) * 2


def pll4_mhz(cntl1: int, cntl5: int, use_postdiv: int) -> tuple:
    """(VCO MHz, [L2CPUCLK0..3 MHz]) exactly as CMFW derives them (FW pll.c:462-475 CalculateFreqFromPllRegs, 490-493
    GetVcoFreq): VCO = PLL_REF * fbdiv / refdiv, out_i = PLL_REF * fbdiv / (refdiv * eff_i) with eff_i =
    pll_ext_postdiv(CNTL_5 postdiv_i, USE_POSTDIV bit i), 0 MHz where eff_i = 0 (disabled). CNTL_1.postdiv is ignored,
    as CMFW does (pll.c:456-461). Floats, not CMFW's integer division: a bound checked on these is never looser."""
    refdiv, _, fbdiv, pds = pll_fields(cntl1, cntl5)
    vco = PLL_REF_MHZ / refdiv * fbdiv
    effs = [pll_ext_postdiv(p, bool((use_postdiv >> i) & 1)) for i, p in enumerate(pds)]
    return vco, [vco / e if e else 0.0 for e in effs]


def pll4_default_why(cntl1: int, cntl5: int, use_postdiv: int) -> str:
    """'' when PLL4 reads the CMFW state the interlocks require (PLAN §5.2 invariant 4), else why not."""
    out = []
    if cntl1 == PLL4_CNTL_1_SOURCE:
        out.append(f"CNTL_1 0x{cntl1:08x} is the bh_arc source value (postdiv 0, FW pll.c:226), not the "
                   f"0x{PLL4_CNTL_1_DEFAULT:08x} CMFW's clock driver writes on the keeper fd's power message (S1c "
                   "silicon): unexplained, not walked")
    elif cntl1 != PLL4_CNTL_1_DEFAULT:
        r, p, f, _ = pll_fields(cntl1, 0)
        out.append(f"CNTL_1 0x{cntl1:08x} (refdiv {r}, postdiv {p}, fbdiv {f}) != 0x{PLL4_CNTL_1_DEFAULT:08x}")
    if cntl5 != PLL4_CNTL_5_DEFAULT:
        out.append(f"CNTL_5 0x{cntl5:08x} != 0x{PLL4_CNTL_5_DEFAULT:08x}")
    if use_postdiv & PLL4_USE_POSTDIV_MASK != PLL4_USE_POSTDIV_MASK:
        out.append(f"USE_POSTDIV 0x{use_postdiv:x} & 0x{PLL4_USE_POSTDIV_MASK:x} != 0x{PLL4_USE_POSTDIV_MASK:x}")
    return "; ".join(out)


# clock.py's solutions table plus PLAN §5.2 step 8's new 800 MHz entry: mhz -> (fbdiv, [postdiv0..3]).
PLL4_SOLUTIONS = {200: (128, [15, 15, 15, 15]), 800: (64, [1, 1, 1, 1])}  # BHL clock.py:14-19 (200); PLAN §5.2


def pll4_step_writes(cntl1: int, cntl5: int, mhz: int) -> list:
    """Every (addr, value) write of tt-bh-linux's one-step-at-a-time PLL4 algorithm (BHL clock.py:26-31, 40-45,
    48-71): increasing postdivs first (field by field, +1 per write), then fbdiv by +-1, then decreasing postdivs.
    Every CNTL_1 write keeps refdiv and the postdiv byte exactly as `cntl1` holds them (clock.py reads CNTL_1 once,
    :58-60, and changes only fbdiv, :40-45), so only fbdiv and the CNTL_5 postdivs ever move."""
    fb_t, pd_t = PLL4_SOLUTIONS[mhz]
    refdiv, postdiv, fb, pd = pll_fields(cntl1, cntl5)
    out = []
    for i, t in enumerate(pd_t):
        while t > pd[i]:
            pd[i] += 1
            out.append((PLL4_CNTL_5, pll_cntl5(pd)))
    while fb != fb_t:
        fb += 1 if fb_t > fb else -1
        out.append((PLL4_CNTL_1, pll_cntl1(refdiv, postdiv, fb)))
    for i, t in enumerate(pd_t):
        while t < pd[i]:
            pd[i] -= 1
            out.append((PLL4_CNTL_5, pll_cntl5(pd)))
    return out


# The only two PLL sequences x280d may ever write (PLAN §5.2 steps 6 and 8), precomputed from the CMFW defaults.
# The 200 MHz state keeps the silicon refdiv and postdiv bytes: refdiv 2, postdiv 1, fbdiv 128 = 0x0080_0102.
PLL4_200_CNTL_1 = pll_cntl1(PLL4_CNTL_1_DEFAULT & 0xFF, (PLL4_CNTL_1_DEFAULT >> 8) & 0xFF, PLL4_SOLUTIONS[200][0])
PLL4_200_CNTL_5 = pll_cntl5(PLL4_SOLUTIONS[200][1])
PLL4_STEP = {
    200: (PLL4_CNTL_1_DEFAULT, PLL4_CNTL_5_DEFAULT,
          tuple(pll4_step_writes(PLL4_CNTL_1_DEFAULT, PLL4_CNTL_5_DEFAULT, 200)), PLL4_200_CNTL_1, PLL4_200_CNTL_5),
    800: (PLL4_200_CNTL_1, PLL4_200_CNTL_5,
          tuple(pll4_step_writes(PLL4_200_CNTL_1, PLL4_200_CNTL_5, 800)), PLL4_CNTL_1_DEFAULT, PLL4_CNTL_5_DEFAULT),
}  # mhz -> (start CNTL_1, start CNTL_5, writes, end CNTL_1, end CNTL_5)

# --- L2CPU tile: x280 physical space as seen from the NoC ---------------------------------------
# NoC 0x0..0x7FFF_FFFF_FFFF passes straight through to x280 physical addresses (ISA L2CPUTile/MemoryMap.md:9).

LIM_BASE = 0x0800_0000  # ISA MemoryMap.md:84; Caches.md:71
LIM_SIZE = 15 * 128 * 1024  # 1,966,080 B: 15 of 16 L3 pieces are LIM at reset (ISA Caches.md:68-73)
LIM_END = LIM_BASE + LIM_SIZE  # 0x081E_0000, exclusive
LINE = 64  # L3/L2/L1 line size (ISA Caches.md:7, 26, 42, 69). PLAN R26 assumed ECC granule = line; S1c (2026-10-25,
#            runs/sessions/20261025_053543_2185222_s1c) refutes it: after method A, 4 B reads of lines that read back
#            right still report uncorrectable errors, per 32 B half (S1d, jobs/s1d.sh, measures the store widths)
LIM_LINES = LIM_SIZE // LINE

CCACHE0_BASE = 0x0201_0000  # ISA MemoryMap.md:75; Caches.md:83
CCACHE0_CONFIG = CCACHE0_BASE + 0x000  # ISA Caches.md:83; REF freedom-e-sdk-...metal-platform.h:102; REF linux-v6.8-sifive_ccache.c:39
CCACHE0_WAYENABLE = CCACHE0_BASE + 0x008  # ISA Caches.md:84 (increase-only!); REF metal-platform.h:103; linux:48
WAYENABLE_EXPECTED = 0  # 1 cache way + 15 LIM pieces at reset (ISA Caches.md:73, 89-92); PLAN §5.2 step 1
CCACHE0_ECCINJECTERR = CCACHE0_BASE + 0x040  # REF metal-platform.h:104; linux:49
# ECC fix/fail counters, SiFive ccache0 layout (REF linux-v6.8-sifive_ccache.c:23-37; REF metal-platform.h:105-116).
# Reading a COUNT register clears its ECC interrupt signal (linux:210-240; freedom-metal sifive_ccache0.h:69-81):
# a side effect on interrupts only, never on cache configuration.
CCACHE0_ECC = {
    "dir_fix_low": CCACHE0_BASE + 0x100, "dir_fix_high": CCACHE0_BASE + 0x104, "dir_fix_count": CCACHE0_BASE + 0x108,
    "dir_fail_low": CCACHE0_BASE + 0x120, "dir_fail_high": CCACHE0_BASE + 0x124, "dir_fail_count": CCACHE0_BASE + 0x128,
    "dat_fix_low": CCACHE0_BASE + 0x140, "dat_fix_high": CCACHE0_BASE + 0x144, "dat_fix_count": CCACHE0_BASE + 0x148,
    "dat_fail_low": CCACHE0_BASE + 0x160, "dat_fail_high": CCACHE0_BASE + 0x164, "dat_fail_count": CCACHE0_BASE + 0x168,
}
CCACHE0_FLUSH64 = CCACHE0_BASE + 0x200  # ISA Caches.md:85
CCACHE0_FLUSH32 = CCACHE0_BASE + 0x240  # ISA Caches.md:86
CCACHE0_WAYMASK0 = CCACHE0_BASE + 0x800  # ISA Caches.md:87, 38 x u64 (Caches.md:101-111)


def l2pf_base(hart: int) -> int:
    return 0x0203_0000 + 0x2000 * hart  # ISA Caches.md:57; MemoryMap.md:77-80


L2PF_BASIC_CONTROL_VALUE = 0x15811  # ISA Caches.md:64; README.md:50; BHL boot.py:252
L2PF_USER_CONTROL_VALUE = 0x38C84E  # ISA Caches.md:64; README.md:50; BHL boot.py:253
L2PF_WRITES = tuple(w for h in range(4) for w in ((l2pf_base(h), L2PF_BASIC_CONTROL_VALUE),
                                                   (l2pf_base(h) + 4, L2PF_USER_CONTROL_VALUE)))  # PLAN §5.2 step 9


def beu_base(hart: int) -> int:
    return 0x0170_0000 + 0x1000 * hart  # ISA MemoryMap.md:69-72


BEU = {"cause": 0, "value": 8, "enable": 16, "platform_interrupt": 24, "accrued": 32, "local_interrupt": 40}
# REF freedom-e-sdk-4518a22-unmatched-metal-platform.h:88-93 (METAL_SIFIVE_BUSERROR0_*)

# External peripherals: x280 physical 0x2000_0000; from the NoC also at a high alias that bypasses the cores,
# address + 0xFFFF_F7FE_DFF0_0000 (ISA MemoryMap.md:104). x280d uses the high alias, as tt-bh-linux does
# (BHL boot.py:118, 232-239), because the harts are in reset while vectors are written.
XP_PHYS_BASE = 0x2000_0000  # ISA MemoryMap.md:22
XP_HIGH_OFFSET = 0xFFFF_F7FE_DFF0_0000  # ISA MemoryMap.md:104


def xp_high(phys: int) -> int:
    return phys + XP_HIGH_OFFSET


RESET_VECTOR_PHYS = 0x2001_0000  # 4 x 8 B per-hart reset pc (ISA MemoryMap.md:111), written lo then hi u32 (BHL boot.py:232-239)
GP_SCRATCH_PHYS = 0x2001_0100  # 64 B, executable through L1I (ISA MemoryMap.md:113; Caches.md:18-19)
GP_SCRATCH_SIZE = 64
HART_STATUS_PHYS = 0x2001_0400  # 2 B cease/halt/wfi/debug (ISA MemoryMap.md:115)
SUPPRESS_FETCH_PHYS = 0x2001_0402  # 2 B (ISA MemoryMap.md:116): never written
RNMI_TRIGGER_PHYS = 0x2001_0414  # bit h = RNMI for hart h, bits 4..31 reserved (ISA RNMIs.md:38-46; MemoryMap.md:118)
RNMI_HANDLER_PHYS = 0x2001_0418  # hart h: handler @ +16h, exception handler @ +16h+8 (ISA RNMIs.md:48-59; MemoryMap.md:119)
XP_TLBCFG_PHYS_LO, XP_TLBCFG_PHYS_HI = 0x2000_0000, 0x2000_0F7F  # X280 -> NoC TLB windows (ISA MemoryMap.md:108-109)
XP_DMA_PHYS_LO, XP_DMA_PHYS_HI = 0x2008_0000, 0x2FFF_FFFF  # DMA engine + its copies (ISA MemoryMap.md:23-25)

# S5a (docs/plan/vla-wbc-x280/research/four_tile_design.md step_plan): read-only probe of the outbound TLB window
# slots and NIU #0, from the NoC through the same high alias as the vectors (ISA MemoryMap.md:104). Reads only: the
# slots stay under DENY X280_TLB_CONFIG_HIGH and no ALLOW entry covers NIU #0.
TLB_SMALL_N, TLB_LARGE_N = 224, 32  # 256 windows: 224 x 2 MiB + 32 x 128 GiB (ISA README.md:3; TLBWindows.md:3, 11-14)
TLB_SMALL_WORDS = 4  # 16 B {u64 local_offset, u32 noc_properties_lo, u32 noc_properties_hi} (ISA TLBWindows.md:11, 22-27)
TLB_LARGE_WORDS = 3  # 12 B {u32 local_offset, u32 noc_properties_lo, u32 noc_properties_hi} (ISA TLBWindows.md:12, 29-34)
XP_TLB_LARGE_PHYS = XP_TLBCFG_PHYS_LO + 4 * TLB_SMALL_WORDS * TLB_SMALL_N  # 0x2000_0E00 (ISA MemoryMap.md:109)
NIU0_PHYS = 0x2005_6000  # NIU #0 (NoC #0) configuration / status, 4 KiB (ISA MemoryMap.md:121; NoC/MemoryMap.md:3)
NOC_ENDPOINT_TYPE_L2CPU = 0x0901  # NOC_ENDPOINT_ID bits 23:8 (ISA NoC/MemoryMap.md:218)
NOC_GRID = (17, 12)  # NOC_NODE_ID width / height "in practice" (ISA NoC/MemoryMap.md:202-203)
NIU_CFG_0_TILE_CLOCK_DISABLE = 1 << 12  # firmware sets it for fused-off tiles (ISA NoC/MemoryMap.md:252)
NIU_CFG_0_TRANSLATION = 1 << 14  # coordinate translation enable (ISA NoC/MemoryMap.md:254)


def tlb_small_cfg(i: int) -> int:
    """Small window i's configuration, NoC side: 0xFFFF_F7FE_FFF0_0000 + 16 i (ISA MemoryMap.md:104, 108)."""
    return xp_high(XP_TLBCFG_PHYS_LO + 4 * TLB_SMALL_WORDS * i)


def tlb_large_cfg(j: int) -> int:
    """Large window j's configuration, NoC side: 0xFFFF_F7FE_FFF0_0E00 + 12 j (ISA MemoryMap.md:104, 109)."""
    return xp_high(XP_TLB_LARGE_PHYS + 4 * TLB_LARGE_WORDS * j)


NIU0_NOC_NODE_ID = xp_high(NIU0_PHYS + 0x44)  # ISA NoC/MemoryMap.md:19, 193, 196-207
NIU0_NOC_ENDPOINT_ID = xp_high(NIU0_PHYS + 0x48)  # ISA NoC/MemoryMap.md:19, 194, 211-219
NIU0_NIU_CFG_0 = xp_high(NIU0_PHYS + 0x100)  # ISA NoC/MemoryMap.md:22, 230, 247-257


def decode_noc_node_id(v: int) -> dict:
    """NOC_NODE_ID fields (ISA NoC/MemoryMap.md:198-207)."""
    return {"x": v & 0x3F, "y": (v >> 6) & 0x3F, "width": (v >> 12) & 0x7F, "height": (v >> 19) & 0x7F,
            "x_first": bool(v >> 28 & 1)}


def decode_noc_endpoint_id(v: int) -> dict:
    """NOC_ENDPOINT_ID fields (ISA NoC/MemoryMap.md:215-219)."""
    return {"index": v & 0xFF, "type": (v >> 8) & 0xFFFF, "noc": (v >> 24) & 0xFF}


def reset_vector(hart: int) -> int:
    return xp_high(RESET_VECTOR_PHYS + 8 * hart)  # = 0xFFFF_F7FE_FFF1_0000 + 8h (PLAN §5.2 step 5; BHL boot.py:118)


def rnmi_handler(hart: int) -> int:
    return xp_high(RNMI_HANDLER_PHYS + 16 * hart)  # = 0xFFFF_F7FE_FFF1_0418 + 16h (PLAN §5.2 step 4)


def rnmi_exc_handler(hart: int) -> int:
    return rnmi_handler(hart) + 8


GP_SCRATCH = xp_high(GP_SCRATCH_PHYS)
HART_STATUS = xp_high(HART_STATUS_PHYS)
RNMI_TRIGGER = xp_high(RNMI_TRIGGER_PHYS)
HARTS_PER_TILE = 4  # ISA L2CPUTile/README.md:3, 40-43
RNMI_TRIGGER_VALID_MASK = (1 << HARTS_PER_TILE) - 1

# Values boot may write (PLAN §5.2 steps 4-5; WP-3.2): the harts start at the LIM image (or at the E1F stage-0
# stub in GP scratch); RNMIs land in the GP-scratch park stub (<= 48 B), RNMI exceptions in the spin stub at +48.
RESET_PC = LIM_BASE
PARK_STUB = GP_SCRATCH_PHYS
RNMI_EXC_STUB = GP_SCRATCH_PHYS + 48
STAGE0_ENTRY = GP_SCRATCH_PHYS  # E1F only (PLAN §5.2 "E1F stage-0 initializer")
# x280/fw/stage0.S: hart 0 zero-fills LIM, writes the done word (= LIM_END) into the last line, then spins on the
# host's entry word in the line before it (one writer per line) and jumps there.
STAGE0_ENTRY_WORD = 0x081D_FF80
STAGE0_DONE_WORD = 0x081D_FFC0
# stage0.S `3: wfi; j 3b` of harts 1-3 (build/stage0.elf): an E1F boot leaves them there with this line in their L1I.
STAGE0_SEC_WFI = GP_SCRATCH_PHYS + 0x3A

# --- harts 1-3 of the tile (x280/fw/mh.h, crt0.S; the 4-hart bench design A) ---------------------------------------
# `x280d serve --harts 4` (X280_MH_APPROVED=1, stage-0 boot only) writes, before the release, hart 1-3 RNMI handler =
# RESET_PC (crt0 _start, whose secondary path runs mh.c) and RNMI exception handler = MH_SEC_RNMI_EXC, a `wfi; j .` spin
# at a fixed LIM address (crt0.S .org; lim.ld asserts it). Hart 0 keeps PARK_STUB / RNMI_EXC_STUB, so PARK is unchanged.
MH_HARTS = 4
MH_SEC_RNMI_EXC = LIM_BASE + 0x200
MH_SEC_HARTS = (1, 2, 3)
MH_SEC_MASK = 0b1110  # their RNMI trigger bits
# The multi-hart region (NOLOAD .mh, lim.ld): in the monitor region, so X280-owned after the release; the host only
# reads it. ctl line (hart 0) at MH_BASE, stat / aux line of hart h at MH_BASE + 0x80 h (+0x40), written by hart h.
MH_BASE, MH_SIZE = 0x0800_A000, 0x2000
MH_CTL = MH_BASE


def mh_stat(hart: int) -> int:
    return MH_BASE + 0x80 * hart


def mh_aux(hart: int) -> int:
    return mh_stat(hart) + 0x40


# The mh_bench payload (x280/fw/mh_bench.ld): code at MHB_BASE, its job / result lines at its end, all before the E1F
# entry-word line; after HOST_BENCH_BASE's 1000-iteration BENCH array (x280d.py).
MHB_BASE, MHB_END = 0x081D_F400, 0x081D_FF80

# --- LIM mailbox layout v2 (PLAN §1.4.4; must equal x280/fw/mailbox.h of WP-3.2) -----------------

IMAGE_BASE = LIM_BASE  # monitor + kernel image, <= 48 KiB in a 64 KiB region (PLAN §1.4.4)
IMAGE_REGION = 0x1_0000
MAILBOX_BASE = 0x0801_0000
WEIGHTS_BASE = 0x0801_1000


def mb_line(n: int) -> int:
    """NoC/x280 address of mailbox line Ln (PLAN §1.4.4 table)."""
    return MAILBOX_BASE + LINE * n


MB_CMD = mb_line(0)  # host body bytes 0-59, then seq_req
MB_SEQ_REQ = MB_CMD + 60  # written last, as its own 4 B UC write (PLAN §1.4.4, X-6)
MB_STATUS = mb_line(1)  # X280
MB_OBS = mb_line(2)  # L2-L34 host, obs f32[516]
MB_OBS_LINES = 33
MB_ACTION = mb_line(35)  # X280
MB_IDENT = mb_line(36)  # X280
MB_LOAD_DESC = mb_line(37)  # host
MB_HOST_GONE = mb_line(38)  # KMD cleanup only (+ x280d's clear at attach)
HOST_GONE_VALUE = 0x474F_4E45  # 'GONE' (PLAN §1.4.4 L38)
MB_ESCRATCH = mb_line(39)  # X280
MB_IDENT_MH = MB_IDENT + 60  # L36.mh (x280/fw/mailbox.h): MH_CAP_BIT | ready mask, written by hart 0
MH_CAP_BIT = 1 << 8  # the monitor has crt0's secondary path, mh.c and MB_CMD_MH
MB_CHUNK = mb_line(40)  # L40-41 host (M4c)
X280_OWNED_LINES = frozenset((1, 35, 36, 39))  # mailbox line numbers the X280 writes (PLAN §1.4.4)
KMD_OWNED_LINES = frozenset((38,))
# The running monitor's own lines: .text/.rodata (host-written before release only), .bss with g_rt and the trap
# stack (crt0.S clears it with `sd` on every entry), .persist and the 16 KiB stack 0x0800_C000-0x0801_0000
# (x280/fw/lim.ld:24-56, 63-66). After the release the hart writes them and the park stub re-enters crt0 at
# 0x0800_0000, so from then on they are X280-owned (review F4); only E1F's image write, before the monitor first
# runs, is exempt.
MONITOR_REGION = (IMAGE_BASE, MAILBOX_BASE)
# monitor.c cmd_load: dst 64 B aligned, len > 0, [dst, dst+len) inside [MB_PAYLOAD_BASE, MB_LIM_END)
# (x280/fw/monitor.c:221-224; mailbox.h:31, 37). x280d checks the same before writing any data line.
LOAD_BASE = WEIGHTS_BASE


def in_monitor_region(addr: int) -> bool:
    return MONITOR_REGION[0] <= addr < MONITOR_REGION[1]


def lim_line_owner(addr: int, released: bool = False) -> str:
    """'x280' | 'kmd' | 'host' for the LIM line holding `addr` (one writer per line, PLAN §1.4.4). `released`: the
    tile's harts run, so the monitor region [0x0800_0000, 0x0801_0000) belongs to the X280 too."""
    if MAILBOX_BASE <= addr < MAILBOX_BASE + 64 * LINE:
        n = (addr - MAILBOX_BASE) // LINE
        if n in X280_OWNED_LINES:
            return "x280"
        if n in KMD_OWNED_LINES:
            return "kmd"
    if released and in_monitor_region(addr):
        return "x280"
    return "host"


# --- Windows (each configured once per session; PLAN §1.1 x280d row, WP-3.3) ---------------------
# KMD maps only a prefix of a window starting at its base (memory.c:1582-1596) and forbids splitting it
# (memory.c:1538-1550), so each window maps the shortest prefix that covers what x280d touches.


@dataclass(frozen=True)
class WindowSpec:
    name: str
    xy: tuple
    base: int  # NoC address, 2 MiB aligned (KMD blackhole.c:181-182 rejects anything else)
    map_len: int  # bytes mapped from base (page multiple)
    persistent: str  # 'rw' (LIM only), 'ro', or 'none' (mapped only inside a transient session)
    wc: bool  # also map write-combined (LIM only: libwcio full-line stores)
    why: str


XP_MAP_LEN = 0x11_1000  # the external-peripheral prefix up to the RNMI handlers (0xFFFF_F7FE_FFF1_0457)
XP_PROBE_MAP_LEN = 0x15_7000  # ... up to NIU #0's page 0xFFFF_F7FE_FFF5_6000..6FFF: S5a probe sessions only


def window_specs(tiles: tuple = (DEFAULT_TILE,), cross_read: bool = False, probe_tiles: tuple = ()) -> tuple:
    """The fixed window set: ARC registers, ARC CSM, and per tile LIM, CCACHE/L2PF, BEU, external peripherals.
    probe_tiles (S5a: `x280d probe --tile 3,2,1,0`, `serve --no-release --probe-tiles`): read-only tiles; a probe tile
    outside `tiles` gets only CCACHE0 (+ the translated cross-read) and the external-peripheral prefix, and with
    probe_tiles every xp prefix reaches NIU #0's page. Without probe_tiles the set is unchanged."""
    xp_len = XP_PROBE_MAP_LEN if probe_tiles else XP_MAP_LEN
    out = [
        WindowSpec("arc", ARC_XY, 0x8000_0000, 0x3_1000, "ro", False,
                   "PLL4 0x8002_0500 + reset unit 0x8003_0000..0FFF (L2CPU_RESET, SCRATCH_RAM); writes transient"),
        WindowSpec("csm", ARC_XY, ARC_CSM_BASE, ARC_CSM_SIZE, "ro", False, "telemetry table (TIMER_HEARTBEAT, ENABLED_L2CPU)"),
    ]
    for t in tiles:
        xy = L2CPU_XY[t]
        out += [
            WindowSpec(f"lim{t}", xy, LIM_BASE, LIM_SIZE, "rw", True, "LIM: image, mailbox, weights"),
            WindowSpec(f"ccache{t}", xy, 0x0200_0000, 0x3_8000, "ro", False,
                       "CCACHE0 regs (WAYENABLE read-only), ECC counters, L2PF 0x0203_0000..7FFF; writes transient"),
            WindowSpec(f"beu{t}", xy, 0x0160_0000, 0x10_4000, "ro", False, "hart BEUs 0x0170_0000..3FFF"),
            WindowSpec(f"xp{t}", xy, 0xFFFF_F7FE_FFE0_0000, xp_len, "none", False,
                       "external peripherals page 0xFFFF_F7FE_FFF1_0000 (vectors, GP scratch, RNMI); "
                       "the prefix's first MiB is reserved alias space, so this window is never mapped persistently"),
        ]
        if cross_read:
            out.append(WindowSpec(f"ccache{t}x", L2CPU_XY_TRANSLATED[t], 0x0200_0000, 0x1_1000, "ro", False,
                                  "translated-coordinate CCACHE0_CONFIG cross-read (T-3.E0)"))
    for t in dict.fromkeys(probe_tiles):
        if t in tiles:
            continue
        xy = L2CPU_XY[t]
        out += [
            WindowSpec(f"ccache{t}", xy, 0x0200_0000, 0x1_1000, "ro", False,
                       "S5a probe tile: CCACHE0 regs (WAYENABLE, CONFIG, ECC counters), read only"),
            WindowSpec(f"xp{t}", xy, 0xFFFF_F7FE_FFE0_0000, xp_len, "none", False,
                       "S5a probe tile: TLB window slots, hart status, NIU #0 through transient read-only mappings; "
                       "the prefix's first MiB is reserved alias space, so this window is never mapped persistently"),
        ]
        if cross_read:
            out.append(WindowSpec(f"ccache{t}x", L2CPU_XY_TRANSLATED[t], 0x0200_0000, 0x1_1000, "ro", False,
                                  "translated-coordinate CCACHE0_CONFIG cross-read (T-3.E0, S5a)"))
    return tuple(out)


ALLOWED_TARGETS = frozenset([ARC_XY] + list(L2CPU_XY.values()) + list(L2CPU_XY_TRANSLATED.values()))

# --- Address-range policy ----------------------------------------------------------------------


@dataclass(frozen=True)
class Range:
    name: str
    tiles: str  # 'arc' | 'l2cpu' | 'any'
    lo: int
    hi: int  # inclusive
    why: str
    exception: str = ""  # DENY only: the one privileged path that may write here
    ctx: str = ""  # ALLOW only: privileged path required ('' = plain LIM accessors)
    sizes: tuple = (4,)  # ALLOW only: access sizes in bytes
    values: tuple = ()  # ALLOW only: permitted 32-bit values ((): any, subject to the stateful rule)


GDDR_REACHING = "reaches GDDR (tt-metal DRAM banks 5/6/7, PLAN R35)"

DENY = (
    Range("CCACHE0_WAYENABLE", "l2cpu", CCACHE0_WAYENABLE, CCACHE0_WAYENABLE + 7,
          "increase-only: converts LIM to cache until an ASIC reset (ISA Caches.md:73, 84, 89-93); "
          "tt-bh-linux writes 0xF here (BHL boot.py:210-212), x280d never does (PLAN §5.2 invariant 1)"),
    Range("L2CPU_GDDR_AND_X280_TLB", "l2cpu", 0x0000_3000_0000, 0x0000_7FFF_FFFF_FFFF,
          f"local GDDR6 (uncached/cached), X280 TLB windows to the NoC and their copies (ISA MemoryMap.md:26-41): {GDDR_REACHING}"),
    Range("L2CPU_RESERVED_ALIASES", "l2cpu", 0x0000_8000_0000_0000, 0xFFFF_F7FE_FFEF_FFFF,
          f"reserved, in practice copies of the 2^47 passthrough incl. GDDR (ISA MemoryMap.md:10): {GDDR_REACHING}"),
    Range("X280_TLB_CONFIG", "l2cpu", XP_TLBCFG_PHYS_LO, XP_TLBCFG_PHYS_HI,
          f"X280 -> NoC TLB window configuration (ISA MemoryMap.md:108-109): {GDDR_REACHING}"),
    Range("X280_TLB_CONFIG_HIGH", "l2cpu", xp_high(XP_TLBCFG_PHYS_LO), xp_high(XP_TLBCFG_PHYS_HI),
          f"same, high alias (ISA MemoryMap.md:104, 108-109): {GDDR_REACHING}"),
    Range("X280_DMA", "l2cpu", XP_DMA_PHYS_LO, XP_DMA_PHYS_HI,
          f"DMA engine configuration and its copies (ISA MemoryMap.md:23-25): {GDDR_REACHING}"),
    Range("SUPPRESS_FETCH", "l2cpu", xp_high(SUPPRESS_FETCH_PHYS), xp_high(SUPPRESS_FETCH_PHYS) + 1,
          "suppress-fetch flags (ISA MemoryMap.md:116): untested path, PLAN avoids it"),
    Range("NIU_CONFIG", "any", 0xFFFF_FFFF_FF00_0000, 0xFFFF_FFFF_FFFF_FFFF,
          "NIU configuration incl. coordinate translation (ISA L2CPUTile/MemoryMap.md:13)"),
    Range("ARC_DDR_RESET", "arc", DDR_RESET, DDR_RESET + 3, "GDDR controller resets (FW init.h:15): would wipe tt-metal DRAM"),
    Range("ARC_RESET_UNIT_LOW", "arc", RESET_UNIT_BASE, DDR_RESET + 3,
          "global/ETH/DDR resets (FW init.h:13-15)"),
    Range("ARC_RESET_UNIT_HIGH", "arc", L2CPU_RESET + 4, RESET_UNIT_BASE + 0xFFF,
          "Tensix resets, SCRATCH, SCRATCH_RAM incl. the ARC message queue pointers (FW init.h:18-27; status_reg.h:16-87)"),
    Range("ARC_MSI_FIFO", "arc", ARC_MSI_FIFO, ARC_MSI_FIFO + 3, "ARC message trigger (KMD blackhole.c:67): no ARC messages (D12)"),
    Range("PLL", "arc", PLL_BLOCK_LO, PLL_BLOCK_HI,
          "PLL controller (FW pll.c:22-32, 239-241): CMFW owns it; only PLAN §5.2 steps 6/8 touch PLL4",
          exception="pll4_step"),
    Range("ARC_CSM", "arc", ARC_CSM_BASE, ARC_CSM_BASE + ARC_CSM_SIZE - 1, "CMFW memory (KMD telemetry.h:82-83): read-only"),
)

ALLOW = (
    Range("L2CPU_RESET", "arc", L2CPU_RESET, L2CPU_RESET + 3,
          "the single boot RMW (PLAN §5.2 step 7; ISA README.md:35-46)", ctx="l2cpu_reset_rmw",
          values=tuple(L2CPU_RESET_AT_SESSION_START | m for m in RELEASE_MASKS)),
    Range("PLL4_CNTL_1", "arc", PLL4_CNTL_1, PLL4_CNTL_1 + 3, "PLAN §5.2 steps 6/8 (BHL clock.py:40-45)", ctx="pll4_step"),
    Range("PLL4_CNTL_5", "arc", PLL4_CNTL_5, PLL4_CNTL_5 + 3, "PLAN §5.2 steps 6/8 (BHL clock.py:26-31)", ctx="pll4_step"),
    Range("LIM", "l2cpu", LIM_BASE, LIM_END - 1,
          "LIM image/mailbox/weights (PLAN §1.4.4): 64 B full-line WC stores, 4 B UC writes only on initialised lines",
          sizes=(4, LINE)),
    # S1d store widths (jobs/s1d.sh): LIM init stores only (kmd.lim_init, ctx lim_init, hence pre-release only and
    # whole 64 B lines per call, review F3). WC32: two 32 B ymm stores per line through the WC mapping (UMD
    # memcpy_to_device's pattern, which S1c E1b saw reach the CCACHE as one 64 B access). UC8: aligned 8 B UC stores,
    # one ECC granule if the granule is 8 B as S1c's counts suggest. Neither is a write path after LIM init.
    Range("LIM_INIT_WC32", "l2cpu", LIM_BASE, LIM_END - 1,
          "S1d LIM init variant: 2 x 32 B ymm stores per 64 B line through the WC mapping", ctx="lim_init",
          sizes=(32,)),
    Range("LIM_INIT_UC8", "l2cpu", LIM_BASE, LIM_END - 1,
          "S1d LIM init variant: 8 aligned 8 B UC stores per 64 B line", ctx="lim_init", sizes=(8,)),
    Range("L2PF", "l2cpu", l2pf_base(0), l2pf_base(3) + 7, "PLAN §5.2 step 9 (ISA Caches.md:57-64)", ctx="l2pf",
          values=(L2PF_BASIC_CONTROL_VALUE, L2PF_USER_CONTROL_VALUE)),
    Range("RESET_VECTORS", "l2cpu", reset_vector(0), reset_vector(3) + 7,
          "PLAN §5.2 step 5, before release only (ISA README.md:46)", ctx="boot_vectors",
          values=(RESET_PC, STAGE0_ENTRY, 0)),
    Range("RNMI_HANDLERS", "l2cpu", rnmi_handler(0), rnmi_exc_handler(3) + 7,
          "PLAN §5.2 step 4, before release only (ISA RNMIs.md:61); harts 1-3 RESET_PC / MH_SEC_RNMI_EXC only under "
          "--harts 4 (kmd.write_boot_vectors), hart 0 always the park stub", ctx="boot_vectors",
          values=(PARK_STUB, RNMI_EXC_STUB, RESET_PC, MH_SEC_RNMI_EXC, 0)),
    Range("GP_SCRATCH", "l2cpu", GP_SCRATCH, GP_SCRATCH + GP_SCRATCH_SIZE - 1,
          "park stub + RNMI-exception spin stub (PLAN WP-3.2; §5.2 step 4, E1F)", ctx="gp_scratch"),
    Range("RNMI_TRIGGER", "l2cpu", RNMI_TRIGGER, RNMI_TRIGGER + 3, "PARK protocol (PLAN §5.2; ISA RNMIs.md:38-46)",
          ctx="park"),
)

LIM_ALLOW = ("LIM", "LIM_INIT_WC32", "LIM_INIT_UC8")  # the ALLOW entries kmd applies its LIM line rules to

# S1d (jobs/s1d.sh): one fresh region per store variant, all inside the weights area [LOAD_BASE, LIM_END), away from
# the monitor region and the mailbox lines (their one-writer rules), on a chip no host has written since its reset.
S1D_BASE = LOAD_BASE

# Reads x280d may issue (anything else is refused: a stray read of an alias range can hang the host, R28).
READ = (
    Range("PLL_BLOCK", "arc", PLL_BLOCK_LO, PLL_BLOCK_HI, "FW pll.c:22-32"),
    Range("RESET_UNIT", "arc", RESET_UNIT_BASE, scratch_ram(63) + 3, "FW init.h:13-27; status_reg.h:12-87"),
    Range("CSM", "arc", ARC_CSM_BASE, ARC_CSM_BASE + ARC_CSM_SIZE - 1, "KMD telemetry.h:82-83"),
    Range("LIM", "l2cpu", LIM_BASE, LIM_END - 1, "PLAN §1.4.4"),
    Range("CCACHE0_REGS", "l2cpu", CCACHE0_BASE, CCACHE0_BASE + 0x16F, "ISA Caches.md:83-84; REF linux:23-49"),
    Range("CCACHE0_WAYMASK0", "l2cpu", CCACHE0_WAYMASK0, CCACHE0_WAYMASK0 + 38 * 8 - 1, "ISA Caches.md:87, 101-111"),
    Range("L2PF", "l2cpu", l2pf_base(0), l2pf_base(3) + 0x1FFF, "ISA MemoryMap.md:77-80"),
    Range("BEU", "l2cpu", beu_base(0), beu_base(3) + 0xFFF, "ISA MemoryMap.md:69-72"),
    Range("XP_PAGE", "l2cpu", reset_vector(0), rnmi_exc_handler(3) + 7, "ISA MemoryMap.md:111-119"),
    # S5a (four_tile_design.md step_plan): reset values of the 256 outbound TLB window slots and NIU #0's id/config,
    # high alias; hart status 0x2001_0400 is in XP_PAGE above (ISA MemoryMap.md:115)
    Range("X280_TLB_SMALL_CFG", "l2cpu", tlb_small_cfg(0), tlb_small_cfg(TLB_SMALL_N) - 1,
          "224 x 16 B small TLB window configuration (ISA MemoryMap.md:104, 108; TLBWindows.md:11, 22-27)"),
    Range("X280_TLB_LARGE_CFG", "l2cpu", tlb_large_cfg(0), tlb_large_cfg(TLB_LARGE_N) - 1,
          "32 x 12 B large TLB window configuration (ISA MemoryMap.md:104, 109; TLBWindows.md:12, 29-34)"),
    Range("NIU0_ID", "l2cpu", NIU0_NOC_NODE_ID, NIU0_NOC_ENDPOINT_ID + 3,
          "NIU #0 NOC_NODE_ID, NOC_ENDPOINT_ID, read only (ISA MemoryMap.md:104, 121; NoC/MemoryMap.md:19, 187-219)"),
    Range("NIU0_CFG_0", "l2cpu", NIU0_NIU_CFG_0, NIU0_NIU_CFG_0 + 3,
          "NIU #0 NIU_CFG_0: bit 12 tile clock disable, bit 14 translation (ISA MemoryMap.md:121; "
          "NoC/MemoryMap.md:22, 230, 247-257)"),
)
S5A_READ = ("X280_TLB_SMALL_CFG", "X280_TLB_LARGE_CFG", "NIU0_ID", "NIU0_CFG_0")


def _kind(xy: tuple) -> str:
    if tuple(xy) == ARC_XY:
        return "arc"
    return "l2cpu" if tile_of(xy) is not None else "none"


def _hits(ranges: tuple, kind: str, lo: int, hi: int) -> list:
    return [r for r in ranges if r.tiles in (kind, "any") and r.lo <= hi and lo <= r.hi]


def classify_write(xy: tuple, addr: int, size: int) -> tuple:
    """(allow entry or None, [overlapping DENY entries], refusal reason or '') for a write of `size` bytes.

    Pure data lookup: the caller still applies the entry's ctx/values and its stateful rule."""
    kind, hi = _kind(xy), addr + size - 1
    if kind == "none":
        return None, [], f"target {tuple(xy)} is not an allowed tile"
    if kind == "l2cpu" and (is_translated(xy) or tile_of(xy) not in WRITABLE_TILES):
        return None, [], f"tile {tuple(xy)} is read-only"
    denies = _hits(DENY, kind, addr, hi)
    allows = [r for r in ALLOW if r.tiles in (kind, "any") and r.lo <= addr and hi <= r.hi]
    if not allows:
        why = f"DENY {','.join(d.name for d in denies)}" if denies else "not in the write allowlist"
        return None, denies, f"0x{addr:x} on {tuple(xy)}: {why}"
    sized = [r for r in allows if size in r.sizes]  # LIM has one entry per store width (the S1d ones need ctx lim_init)
    if not sized:
        return None, denies, f"size {size} not permitted for {allows[0].name}"
    return sized[0], denies, ""


def classify_read(xy: tuple, addr: int, size: int = 4) -> str:
    """'' if the read is allowed, else the refusal reason."""
    kind = _kind(xy)
    if kind == "none":
        return f"target {tuple(xy)} is not an allowed tile"
    ok = any(r.tiles in (kind, "any") and r.lo <= addr and addr + size - 1 <= r.hi for r in READ)
    return "" if ok else f"read of 0x{addr:x} on {tuple(xy)} is not in the read allowlist"


def _self_check() -> None:
    """Import-time invariants: a typo here must fail loudly, never reach the chip."""
    assert PLL4_CNTL_1 == 0x8002_0504 and PLL4_CNTL_5 == 0x8002_0514 and PLL4_USE_POSTDIV == 0x8002_051C
    assert pll4_mhz(PLL4_CNTL_1_DEFAULT, PLL4_CNTL_5_DEFAULT, PLL4_USE_POSTDIV_MASK) == (1600.0, [800.0] * 4)
    assert pll4_mhz(PLL4_200_CNTL_1, PLL4_200_CNTL_5, PLL4_USE_POSTDIV_MASK) == (3200.0, [200.0] * 4)
    assert pll4_mhz(PLL4_CNTL_1_DEFAULT, PLL4_CNTL_5_DEFAULT, 0xE)[1] == [1600.0, 800.0, 800.0, 800.0]  # bit 0 clear
    assert [pll_ext_postdiv(v, True) for v in (0, 1, 16, 17)] == [0, 2, 17, 36]  # FW pll.c:441-448
    assert PLL4_200_CNTL_1 == 0x0080_0102 and not pll4_default_why(PLL4_CNTL_1_DEFAULT, PLL4_CNTL_5_DEFAULT, 0xF)
    for mhz, (c1, c5, writes, e1, e5) in PLL4_STEP.items():
        keep, phase = c1 & 0xFFFF, 0  # refdiv + postdiv bytes as read; phase 0 postdivs up, 1 fbdiv, 2 postdivs down
        for a, v in writes:  # one field changes by one per write, VCO in range, L2CPUCLK <= 800 (BHL clock.py:26-71)
            assert a in (PLL4_CNTL_1, PLL4_CNTL_5)
            if a == PLL4_CNTL_1:
                assert v & 0xFFFF == keep and abs((v >> 16) - (c1 >> 16)) == 1 and phase <= 1, (mhz, hex(v))
                phase, c1 = 1, v
            else:
                d = [n - o for n, o in zip(pll_fields(0, v)[3], pll_fields(0, c5)[3]) if n != o]
                assert len(d) == 1 and abs(d[0]) == 1 and phase <= (0 if d[0] > 0 else 2), (mhz, hex(v))
                phase, c5 = (0 if d[0] > 0 else 2), v
            vco, outs = pll4_mhz(c1, c5, PLL4_USE_POSTDIV_MASK)  # the walks start only with USE_POSTDIV & 0xF == 0xF
            assert PLL_VCO_MIN_MHZ <= vco <= PLL_VCO_MAX_MHZ and max(outs) <= PLL4_MAX_MHZ, (mhz, hex(c1), vco, outs)
        assert (c1, c5) == (e1, e5), mhz
    assert reset_vector(0) == 0xFFFF_F7FE_FFF1_0000 and rnmi_handler(0) == 0xFFFF_F7FE_FFF1_0418
    assert RNMI_TRIGGER == 0xFFFF_F7FE_FFF1_0414 and LOAD_BASE == 0x0801_1000 and MONITOR_REGION[1] == MB_CMD
    assert len(set(STATUS_TELEMETRY_TAGS) | {TAG_NOC_TRANSLATION, TAG_ENABLED_L2CPU}) == 13
    assert LIM_END == 0x081E_0000 and MB_HOST_GONE == 0x0801_0980 and MB_ACTION == 0x0801_08C0
    probe_set = window_specs((DEFAULT_TILE,), True, tuple(L2CPU_XY))  # S5a: four probe tiles next to the session's
    assert len({w.name for w in probe_set}) == len(probe_set)
    for w in window_specs(tuple(L2CPU_XY), cross_read=True) + probe_set:
        assert w.base % TLB_2M == 0 and 0 < w.map_len <= TLB_2M and w.map_len % 4096 == 0, w.name
        assert tuple(w.xy) in ALLOWED_TARGETS, w.name
        lo, hi, kind = w.base, w.base + w.map_len - 1, _kind(w.xy)
        gddr = [d.name for d in _hits(DENY, kind, lo, hi) if GDDR_REACHING in d.why]
        if w.persistent != "none":  # no persistent mapping may cover a GDDR-reaching range
            assert not gddr, (w.name, gddr)
        else:  # the external-peripheral prefix: mapped only inside a transient session, both ranges refused
            assert sorted(gddr) == ["L2CPU_RESERVED_ALIASES", "X280_TLB_CONFIG_HIGH"], (w.name, gddr)
        if w.persistent == "rw" or w.wc:  # the only writable persistent mapping is LIM, which overlaps no DENY
            assert w.name.startswith("lim") and not _hits(DENY, kind, lo, hi), w.name
    for a in ALLOW:  # an ALLOW entry may overlap a DENY entry only through that entry's named exception
        for d in _hits(DENY, a.tiles, a.lo, a.hi):
            assert d.exception and d.exception == a.ctx, (a.name, d.name)
    assert classify_write(L2CPU_XY[3], CCACHE0_WAYENABLE, 4)[0] is None
    for name, size in (("LIM_INIT_WC32", 32), ("LIM_INIT_UC8", 8)):  # the S1d widths: LIM only, ctx lim_init only
        r = next(a for a in ALLOW if a.name == name)
        assert (r.lo, r.hi, r.ctx, r.sizes) == (LIM_BASE, LIM_END - 1, "lim_init", (size,)), name
        assert classify_write(L2CPU_XY[3], LIM_BASE, size)[0] is r and classify_write(ARC_XY, L2CPU_RESET, size)[0] is None
        assert classify_write(L2CPU_XY[3], LIM_END - size + 8, size)[0] is None  # never past LIM's end
    assert {a.name for a in ALLOW if a.lo <= LIM_BASE <= a.hi and a.tiles == "l2cpu"} == set(LIM_ALLOW)
    assert classify_write(L2CPU_XY[3], LIM_BASE, 4)[0].name == classify_write(L2CPU_XY[3], LIM_BASE, LINE)[0].name == "LIM"
    assert S1D_BASE % LINE == 0 and not in_monitor_region(S1D_BASE) and S1D_BASE >= MAILBOX_BASE + 64 * LINE
    # harts 1-3: the spin and the MH region inside the monitor region (X280-owned after release, never host-written);
    # the payload between the BENCH array and the E1F words; one 16 B vector pair per hart
    assert MH_SEC_RNMI_EXC == 0x0800_0200 and MH_SEC_RNMI_EXC % LINE == 0 and in_monitor_region(MH_SEC_RNMI_EXC)
    assert in_monitor_region(MH_BASE) and in_monitor_region(MH_BASE + MH_SIZE - 1) and MH_BASE % 0x1000 == 0
    assert lim_line_owner(mh_stat(1), released=True) == "x280" and lim_line_owner(MH_CTL, released=False) == "host"
    assert MHB_END == STAGE0_ENTRY_WORD and MHB_BASE % LINE == 0 and MHB_BASE >= 0x081D_E400 + 4 * 1000
    assert STAGE0_SEC_WFI == 0x2001_013A and MH_SEC_MASK == RNMI_TRIGGER_VALID_MASK & ~1
    # S5a: slot and NIU #0 addresses as ISA MemoryMap.md:104, 108-109, 121 give them; readable on every tile, writable
    # on none; only a probe session's xp prefix reaches NIU #0 (the default one is unchanged)
    assert tlb_small_cfg(0) == 0xFFFF_F7FE_FFF0_0000 and tlb_large_cfg(0) == 0xFFFF_F7FE_FFF0_0E00
    assert tlb_small_cfg(TLB_SMALL_N) == tlb_large_cfg(0) and tlb_large_cfg(TLB_LARGE_N) - 1 == xp_high(XP_TLBCFG_PHYS_HI)
    assert (NIU0_NOC_NODE_ID, NIU0_NOC_ENDPOINT_ID, NIU0_NIU_CFG_0) == (0xFFFF_F7FE_FFF5_6044, 0xFFFF_F7FE_FFF5_6048,
                                                                        0xFFFF_F7FE_FFF5_6100)
    assert {r.name for r in READ if r.name in S5A_READ} == set(S5A_READ)
    xp_lo = next(w for w in window_specs() if w.name == f"xp{DEFAULT_TILE}").base
    for r in (r for r in READ if r.name in S5A_READ):
        assert r.hi < xp_lo + XP_PROBE_MAP_LEN, r.name
        for xy in L2CPU_XY.values():
            for a in (r.lo, r.hi - 3):
                assert not classify_read(xy, a, 4) and classify_write(xy, a, 4)[0] is None, (r.name, xy, hex(a))
    assert xp_lo + XP_MAP_LEN <= NIU0_NOC_NODE_ID


_self_check()
