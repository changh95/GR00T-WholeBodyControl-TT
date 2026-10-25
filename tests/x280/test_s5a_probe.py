"""S5a, the read-only probe of all four L2CPU tiles (docs/plan/vla-wbc-x280/research/four_tile_design.md step_plan S5a).

regs' READ entries for the outbound TLB window slots and NIU #0 (and no new write entry), kmd's read-only probe tiles,
x280d's `probe --tile 3,2,1,0` (standalone and --via a `serve --no-release --probe-tiles` daemon) on kmd.FakeDevice,
x280/tools/m3_jobs.py s5a-summary and jobs/s5a_probe.sh's wiring. No device is opened; servers bind random free ports
(never 5555-5574). The job's run through chip_session's fake mode is test_m3_sessions_fake.py's
test_s5a_probe_session.

    source $RT/env.sh
    X280_CPUS=0,1 nice -n 19 taskset -c 0,1 bash -c 'source $RT/env.sh && simpy -m pytest $RT/tests/x280/test_s5a_probe.py -q'
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT))

from x280 import regs  # noqa: E402

HOST = RT / "x280" / "host"
FW = RT / "x280" / "fw"
JOBS = RT / "jobs"
M3J = RT / "x280" / "tools" / "m3_jobs.py"
X280D = RT / "x280" / "x280d.py"
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
NICE = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]
XPY = RT / ".venv-x280" / "bin" / "python"
PY = str(XPY) if XPY.exists() else sys.executable
ARC = regs.ARC_XY
TILES = (3, 2, 1, 0)
L2CPU_XYS = set(regs.L2CPU_XY.values()) | set(regs.L2CPU_XY_TRANSLATED.values())


def _build():
    so, src = HOST / "libwcio.so", HOST / "wcio.c"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(NICE + ["make", "-s", "-C", str(HOST)], check=True)


_build()
from x280.host import kmd as K  # noqa: E402
from x280.tools import m3_jobs as M  # noqa: E402

NEED = [FW / "build" / p for p in ("monitor.bin", "park_stub.bin", "kernel/wbc_fp16.blob", "kernel/libwbc_mlp.so",
                                   "x86/libmonitor_x86.so")]
needs_build = pytest.mark.skipif(any(not p.exists() for p in NEED), reason="firmware/kernel build outputs missing")


def _s5a_addrs() -> set:
    """Every L2CPU-tile address the S5a probe reads, raw coordinates (the report's table)."""
    a = {regs.CCACHE0_WAYENABLE, regs.CCACHE0_CONFIG, *regs.CCACHE0_ECC.values(), regs.HART_STATUS,
         regs.NIU0_NOC_NODE_ID, regs.NIU0_NOC_ENDPOINT_ID, regs.NIU0_NIU_CFG_0}
    a |= {regs.tlb_small_cfg(i) + 4 * n for i in range(regs.TLB_SMALL_N) for n in range(regs.TLB_SMALL_WORDS)}
    a |= {regs.tlb_large_cfg(j) + 4 * n for j in range(regs.TLB_LARGE_N) for n in range(regs.TLB_LARGE_WORDS)}
    return a


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
    assert not 5555 <= p <= 5574
    return p


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(RT)}


@pytest.fixture
def hold(monkeypatch, tmp_path):
    monkeypatch.setenv("CHIP_SESSION_ID", f"test-s5a-{os.getpid()}")
    return tmp_path / "ledger.json"


# --- regs: READ entries only ------------------------------------------------------------------------------------------


def test_s5a_read_entries_are_the_documented_registers():
    """The four S5a READ entries: small slots 0xFFFF_F7FE_FFF0_0000 + 16 i (i < 224), large 0xFFFF_F7FE_FFF0_0E00 + 12 j
    (j < 32), NIU #0 NOC_NODE_ID / NOC_ENDPOINT_ID at +0x44 / +0x48 and NIU_CFG_0 at +0x100 of 0xFFFF_F7FE_FFF5_6000
    (ISA L2CPUTile/MemoryMap.md:104, 108-109, 121; NoC/MemoryMap.md); readable on each of L2CPU0-3, not on ARC, and not
    one word past their ends. Hart status 0x2001_0400 was already in XP_PAGE."""
    got = {r.name: (r.tiles, r.lo, r.hi) for r in regs.READ if r.name in regs.S5A_READ}
    assert got == {"X280_TLB_SMALL_CFG": ("l2cpu", 0xFFFF_F7FE_FFF0_0000, 0xFFFF_F7FE_FFF0_0DFF),
                   "X280_TLB_LARGE_CFG": ("l2cpu", 0xFFFF_F7FE_FFF0_0E00, 0xFFFF_F7FE_FFF0_0F7F),
                   "NIU0_ID": ("l2cpu", 0xFFFF_F7FE_FFF5_6044, 0xFFFF_F7FE_FFF5_604B),
                   "NIU0_CFG_0": ("l2cpu", 0xFFFF_F7FE_FFF5_6100, 0xFFFF_F7FE_FFF5_6103)}
    for r in regs.READ:
        if r.name in regs.S5A_READ:
            assert "MemoryMap.md" in r.why  # cited like every other entry
    assert regs.xp_high(regs.HART_STATUS_PHYS) == regs.HART_STATUS == 0xFFFF_F7FE_FFF1_0400
    for t, xy in regs.L2CPU_XY.items():
        for a in sorted(_s5a_addrs()):
            assert regs.classify_read(xy, a, 4) == "", (t, hex(a))
        for a in (regs.tlb_large_cfg(regs.TLB_LARGE_N), regs.NIU0_NOC_NODE_ID - 4, regs.NIU0_NOC_ENDPOINT_ID + 4,
                  regs.NIU0_NIU_CFG_0 + 4, regs.NIU0_PHYS + regs.XP_HIGH_OFFSET + 0x200):
            assert regs.classify_read(xy, a, 4), (t, hex(a))
    assert regs.classify_read(ARC, regs.NIU0_NOC_NODE_ID, 4) and regs.classify_read(ARC, regs.tlb_small_cfg(0), 4)


def test_s5a_adds_no_write_entry_and_keeps_the_writable_tiles():
    """S5a writes nothing: the ALLOW list, WRITABLE_TILES and RELEASE_MASKS are as at 1da8b04, and every S5a address is
    refused for writes on every tile, raw or translated."""
    assert [a.name for a in regs.ALLOW] == ["L2CPU_RESET", "PLL4_CNTL_1", "PLL4_CNTL_5", "LIM", "LIM_INIT_WC32",
                                            "LIM_INIT_UC8", "L2PF", "RESET_VECTORS", "RNMI_HANDLERS", "GP_SCRATCH",
                                            "RNMI_TRIGGER"]
    assert regs.WRITABLE_TILES == (2, 3) and regs.RELEASE_MASKS == (0x80, 0x40, 0xC0)
    for xy in L2CPU_XYS:
        for a in sorted(_s5a_addrs() - {regs.CCACHE0_WAYENABLE, regs.CCACHE0_CONFIG, *regs.CCACHE0_ECC.values()}):
            assert regs.classify_write(xy, a, 4)[0] is None, (xy, hex(a))


def test_window_specs_change_only_with_probe_tiles():
    """Without probe tiles the window set is the old one (xp prefix 0x11_1000); with them each probe tile outside the
    session's tiles gets CCACHE0 (read only), its translated cross-read and an xp prefix that reaches NIU #0, never
    a LIM, BEU or any writable mapping."""
    base = regs.window_specs((3,), cross_read=True)
    assert [w.name for w in base] == ["arc", "csm", "lim3", "ccache3", "beu3", "xp3", "ccache3x"]
    assert next(w for w in base if w.name == "xp3").map_len == 0x11_1000
    probe = regs.window_specs((3,), True, TILES)
    assert [w.name for w in probe] == ["arc", "csm", "lim3", "ccache3", "beu3", "xp3", "ccache3x",
                                       "ccache2", "xp2", "ccache2x", "ccache1", "xp1", "ccache1x",
                                       "ccache0", "xp0", "ccache0x"]
    for w in probe:
        if w.name.startswith("xp"):
            assert w.persistent == "none" and w.base + w.map_len > regs.NIU0_NIU_CFG_0 + 3
        elif w.name[-1] in "012" or w.name.endswith(("0x", "1x", "2x")):
            assert w.persistent == "ro" and not w.wc and w.map_len == 0x1_1000
    assert regs.window_specs((3,), True, ()) == base


# --- kmd: probe tiles are read only -----------------------------------------------------------------------------------


@pytest.fixture
def dev():
    return K.FakeDevice.default_chip()


def _kmd(dev, tiles=(3,), probe_tiles=TILES):
    k = K.Kmd(dev, tiles, cross_read=True, probe_tiles=probe_tiles).open()
    dev.advance(1.5)
    k.read_l2cpu_reset()
    return k


def test_kmd_probe_tiles_read_and_refuse_every_write(dev):
    """Probe tiles open a batch and read the slots and NIU #0 through transient read-only mappings; every write path
    on a probe-only tile (L2PF, RNMI trigger, vectors, GP scratch, SET_NOC_CLEANUP, the policy itself) is refused
    before any store, and the session tile's own paths are unchanged."""
    xy0 = regs.L2CPU_XY[0]
    dev.poke32(xy0, regs.tlb_small_cfg(7) + 8, 0x0200_00C8)
    dev.poke32(xy0, regs.tlb_large_cfg(31) + 8, 0xDEAD_BEEF)
    k = _kmd(dev)
    try:
        with k.l2cpu_batch(0):
            slots = k.read_tlb_slots(0)
            niu = k.read_niu0(0)
        assert len(slots["small"]) == 224 and len(slots["large"]) == 32
        assert slots["small"][7] == [0, 0, 0x0200_00C8, 0] and slots["large"][31] == [0, 0, 0xDEAD_BEEF]
        assert sum(map(any, slots["small"])) == 1 and sum(map(any, slots["large"])) == 1
        assert regs.decode_noc_node_id(niu["noc_node_id"]) == {"x": 8, "y": 3, "width": 17, "height": 12,
                                                                "x_first": True}
        assert regs.decode_noc_endpoint_id(niu["noc_endpoint_id"]) == {"index": 0, "type": 0x0901, "noc": 0}
        assert niu["niu_cfg_0"] == regs.NIU_CFG_0_TRANSLATION
        n_ops = len(k.oplog)
        for t in (2, 1, 0):
            with k.l2cpu_batch(t):
                for call in (lambda: k.l2pf_setup(t), lambda: k.rnmi_set(t, 1), lambda: k.write_boot_vectors(t),
                             lambda: k.write_gp_scratch(t, bytes(64)), lambda: k.set_noc_cleanup(t),
                             lambda: k._policy(regs.L2CPU_XY[t], regs.LIM_BASE, 64, None, ""),
                             lambda: k._policy(regs.L2CPU_XY[t], regs.RNMI_TRIGGER, 4, 1, "park")):
                    with pytest.raises(K.PolicyError):
                        call()
        assert not [o for o in k.oplog[n_ops:] if o.kind in ("W32", "WC64", "W64", "MMAP")]
        assert all(c[0] != "mmap" or c[5] == K.PROT_READ or c[2] == k.windows["lim3"].tlb_id for c in dev.calls)
        assert "SET_NOC_CLEANUP" not in dev.names()
        with k.l2cpu_batch(3):  # the session tile is a probe tile too: its xp window reaches NIU #0, and a probe
            assert regs.decode_noc_node_id(k.read_niu0(3)["noc_node_id"])["y"] == 7  # session writes no tile at all
            with pytest.raises(K.PolicyError):
                k.l2pf_setup(3)
    finally:
        k.close()
    with pytest.raises(K.PolicyError, match="WRITABLE_TILES"):
        K.Kmd(dev, (0,))
    with pytest.raises(K.PolicyError, match="not an L2CPU tile"):
        K.Kmd(dev, (3,), probe_tiles=(4,))
    k = K.Kmd(dev, (3,)).open()  # no probe tiles: no window for tile 0, the xp prefix stops before NIU #0
    dev.advance(1.5)
    try:
        with pytest.raises(K.PolicyError):
            with k.l2cpu_batch(0):
                pass
        with k.l2cpu_batch(3):
            with pytest.raises(K.PolicyError, match="does not cover"):
                k.read_niu0(3)
    finally:
        k.close()


# --- x280d Session.tiles_probe on the sim -----------------------------------------------------------------------------


def _session(hold, tiles=TILES, poke=None, **chip_kw):
    from x280 import x280d as D
    from x280 import mailbox_sim as MS
    chip = MS.SimChip.create((), **chip_kw)
    for (xy, addr), v in (poke or {}).items():
        chip.poke32(xy, addr, v)
    chip.update_l2cpuclk_telemetry()
    return D.Session("sim", ledger_path=hold, cross_read=True, hart=False, probe_tiles=tiles, chip=chip)


def _snapshot(dev) -> dict:
    return {key: bytes(v[1]) for key, v in dev.regions.items() if key[0] in L2CPU_XYS}


def _run(hold, tiles=TILES, **kw):
    """Session.tiles_probe on a fresh sim chip; returns (report, kmd ops, libwcio txns, L2CPU memory unchanged)."""
    s = _session(hold, tiles, **kw)
    for key in list(s.chip.lazy):  # the modelled reset values (NIU #0) are part of the state before the probe
        s.chip._region(*key)
    before = _snapshot(s.chip)
    try:
        s.open()
        s.mode = "no_release"
        rep = s.tiles_probe(tiles)
        ops, txn = list(s.k.oplog), s.k.txn_log()
    finally:
        s.close()
    after = _snapshot(s.chip)
    same = all(after[key] == b for key, b in before.items()) and all(
        not any(b) for key, b in after.items() if key not in before)
    assert "SET_NOC_CLEANUP" not in s.chip.names() and not [c for c in s.chip.calls if c[0] == "cleanup"]
    return rep, ops, txn, same


def _tile_of(xy):
    return regs.tile_of(xy) if xy is not None else None


@needs_build
def test_tiles_probe_reads_only_the_documented_addresses_and_writes_nothing(hold):
    from x280 import x280d as D
    rep, ops, txn, same = _run(hold)
    assert rep["verdict"] == "clean" and rep["dirty"] is False and rep["reasons"] == [] and D.tiles_probe_rc(rep) == 0
    assert rep["warnings"] == [] and all(rep["checks"].values()) and rep["harvested"] == []
    assert same  # not one byte of any L2CPU tile changed
    assert not [o for o in ops if o.kind in ("W32", "WC64", "W64", "REFUSED", "ASSUME_INIT")]
    assert {t["op"] for t in txn} == {"UC_R32"}  # libwcio: loads only, no store, no sfence
    reads = [o for o in ops if o.kind in ("R32", "RWORDS")]
    for o in reads:
        assert regs.classify_read(o.xy, o.addr, o.size) == "", str(o)
    for t in TILES:
        raw = {o.addr for o in reads if o.xy == regs.L2CPU_XY[t]}
        assert raw == _s5a_addrs(), (t, sorted(map(hex, raw ^ _s5a_addrs()))[:8])
        assert {o.addr for o in reads if o.xy == regs.L2CPU_XY_TRANSLATED[t]} == {regs.CCACHE0_CONFIG}
        tr = rep["tiles"][str(t)]
        assert tr["probed"] and tr["wayenable"] == 0 and tr["cross_read_equal"] and tr["reads"] >= len(_s5a_addrs())
        assert tr["niu0"]["node"]["y"] == regs.L2CPU_XY[t][1] and tr["tlb"]["small_nonzero"] == 0
    assert {t["xy"] for t in txn if t["xy"] is not None} <= L2CPU_XYS | {ARC}
    # cheapest guards first: L2CPU_RESET, PLL4, PLL_LOCK and the telemetry (ENABLED_L2CPU) before any tile read ...
    first = next(i for i, o in enumerate(reads) if _tile_of(o.xy) is not None)
    arc_before = {o.addr for o in reads[:first] if o.xy == ARC}
    assert {regs.L2CPU_RESET, regs.PLL4_CNTL_1, regs.PLL4_CNTL_5, regs.PLL4_USE_POSTDIV, regs.PLL_LOCK,
            regs.TELEMETRY_TABLE_PTR, regs.TELEMETRY_DATA_PTR} <= arc_before
    # ... then one tile after the other in the asked order, each behind its own PLL4 re-read (the batch)
    seq = []
    for o in reads[first:]:
        t = _tile_of(o.xy)
        if t is not None and (not seq or seq[-1] != t):
            seq.append(t)
    assert seq == list(TILES)
    assert rep["l2cpu_reset_after"] == 0x0F and rep["pll4_after"] == rep["pll4"]


@needs_build
@pytest.mark.parametrize("enabled,off", [(0x7, 3), (0xE, 0), (0xB, 2)])
def test_tiles_probe_skips_a_harvested_tile(hold, enabled, off):
    """ENABLED_L2CPU != 0xF: the disabled tile is reported and never read (no access to its raw or translated
    coordinates at all), the enabled ones are probed; dirty with the ENABLED_L2CPU reason, exit 4 (harvest only)."""
    from x280 import x280d as D
    rep, ops, txn, same = _run(hold, enabled_l2cpu=enabled)
    gone = {regs.L2CPU_XY[off], regs.L2CPU_XY_TRANSLATED[off]}
    assert not [o for o in ops if o.xy in gone and o.kind not in ("IOCTL", "MMAP", "MUNMAP", "NOTE")]
    assert not [t for t in txn if t["xy"] in gone]
    tr = rep["tiles"][str(off)]
    assert not tr["probed"] and "not enabled" in tr["skipped"] and "never read" in tr["skipped"] and tr["reads"] == 0
    assert "wayenable" not in tr and "tlb" not in tr
    assert all(rep["tiles"][str(t)]["probed"] for t in TILES if t != off)
    assert rep["verdict"] == "dirty" and rep["harvested"] == [off] and not rep["checks"]["enabled_l2cpu"]
    assert any("ENABLED_L2CPU" in r and "3-rank" in r for r in rep["reasons"]) and D.tiles_probe_rc(rep) == 4
    assert same


@needs_build
@pytest.mark.parametrize("cntl5,use_postdiv,off", [(0x0100_0101, 0xF, 2), (0x0101_0100, 0xF, 0), (0x0001_0101, 0xF, 3)])
def test_tiles_probe_skips_a_clock_gated_tile_and_reads_no_tile(hold, cntl5, use_postdiv, off):
    """CMFW gates an L2CPUCLK by writing its PLL4 postdiv to 0 (fw clock_control_tt_bh.c:361-388): that tile reads
    0 MHz and is skipped as clock off, and PLL4 off its defaults keeps every other tile unread too (§5.2 invariant 4,
    R28); dirty, exit 3."""
    from x280 import x280d as D
    rep, ops, txn, same = _run(hold, poke={(ARC, regs.PLL4_CNTL_5): cntl5, (ARC, regs.PLL4_USE_POSTDIV): use_postdiv})
    assert not [t for t in txn if t["xy"] in L2CPU_XYS]  # not one L2CPU-tile access
    assert rep["l2cpuclk_mhz"][off] == 0.0 and "clock off" in rep["tiles"][str(off)]["skipped"]
    for t in TILES:
        tr = rep["tiles"][str(t)]
        assert not tr["probed"] and tr["skipped"].endswith("never read") and tr["reads"] == 0
        if t != off:
            assert "invariant 4" in tr["skipped"]
    assert rep["verdict"] == "dirty" and not rep["checks"]["pll4_default"] and D.tiles_probe_rc(rep) == 3
    assert any(r.startswith("PLL4 not at the CMFW defaults") for r in rep["reasons"]) and same


@needs_build
def test_tiles_probe_reads_no_tile_in_cable_fault_mode(hold):
    """CMFW's cable-fault mode clock-gates every tile outside column 15 through NIU_CFG_0 (fw noc_init.c:278-282),
    which no ARC register shows except ERROR_STATUS0 = SCRATCH_RAM[4] bit 1 (fw reset.c:51-59, 341-348): every tile is
    skipped before any tile access; dirty, exit 3."""
    from x280 import x280d as D
    rep, ops, txn, same = _run(hold, poke={(ARC, regs.scratch_ram(4)): 1 << 1})
    assert not [t for t in txn if t["xy"] in L2CPU_XYS]
    assert rep["cable_fault"] and rep["error_status0"] == 2 and not rep["checks"]["no_cable_fault"]
    for t in TILES:
        tr = rep["tiles"][str(t)]
        assert not tr["probed"] and "cable-fault" in tr["skipped"] and tr["reads"] == 0
    assert rep["verdict"] == "dirty" and D.tiles_probe_rc(rep) == 3 and same


@needs_build
def test_tiles_probe_skips_a_tile_held_in_reset(hold):
    """A tile whose L2CPU_RESET[3:0] bit is clear is held in reset: skipped, never read (R28); dirty, exit 3."""
    from x280 import x280d as D
    rep, ops, txn, same = _run(hold, poke={(ARC, regs.L2CPU_RESET): 0x0B})
    assert not [t for t in txn if t["xy"] in (regs.L2CPU_XY[2], regs.L2CPU_XY_TRANSLATED[2])]
    tr = rep["tiles"]["2"]
    assert not tr["probed"] and "held in reset" in tr["skipped"] and tr["reads"] == 0
    assert all(rep["tiles"][str(t)]["probed"] for t in TILES if t != 2)
    assert rep["verdict"] == "dirty" and not rep["checks"]["l2cpu_reset"] and D.tiles_probe_rc(rep) == 3 and same


@needs_build
def test_tiles_probe_records_baselines_without_gating_on_them(hold):
    """Unknown values are recorded, never required: non-zero slots, ECC counts, a hart-status word and an odd
    NOC_NODE_ID leave the verdict clean and show up in the per-tile section (ECC and NIU as warnings)."""
    xy1 = regs.L2CPU_XY[1]
    poke = {(xy1, regs.tlb_small_cfg(5)): 0x40, (xy1, regs.tlb_small_cfg(5) + 8): 0x0200_0148,
            (xy1, regs.tlb_large_cfg(3) + 4): 0x1, (xy1, regs.CCACHE0_ECC["dat_fix_count"]): 3,
            (xy1, regs.HART_STATUS): 0x0F0F, (xy1, regs.NIU0_NOC_NODE_ID): 0x8 | 5 << 6}
    rep, ops, txn, same = _run(hold, poke=poke)
    assert rep["verdict"] == "clean" and all(rep["checks"].values())
    tr = rep["tiles"]["1"]
    assert tr["tlb"]["small"][5] == [0x40, 0, 0x0200_0148, 0] and tr["tlb"]["small_nonzero"] == 1
    assert tr["tlb"]["large"][3] == [0, 1, 0] and tr["tlb"]["large_nonzero"] == 1
    assert {"words": [0, 0, 0, 0], "slots": 223, "first": 0} in tr["tlb"]["small_distinct"]
    assert {"words": [0x40, 0, 0x0200_0148, 0], "slots": 1, "first": 5} in tr["tlb"]["small_distinct"]
    assert tr["ecc"]["dat_fix_count"] == 3 and tr["hart_status"] == 0x0F0F
    assert any(w.startswith("L2CPU1: CCACHE ECC counts") for w in rep["warnings"])
    assert any(w.startswith("L2CPU1: NOC_NODE_ID") for w in rep["warnings"]) and len(rep["warnings"]) == 2
    assert same


@needs_build
@pytest.mark.parametrize("poke,chip_kw,why,skipped", [
    ({(regs.L2CPU_XY[0], regs.CCACHE0_WAYENABLE): 1}, {}, "L2CPU0 CCACHE0_WAYENABLE = 1", None),
    ({}, {"l2cpu_reset": 0x4F}, "L2CPU_RESET = 0x4f", 2),
])
def test_tiles_probe_wayenable_and_a_released_tile_are_dirty(hold, poke, chip_kw, why, skipped):
    from x280 import x280d as D
    rep, ops, txn, same = _run(hold, poke=poke, **chip_kw)
    assert rep["verdict"] == "dirty" and any(r.startswith(why) for r in rep["reasons"]) and D.tiles_probe_rc(rep) == 3
    if skipped is not None:  # a released tile has a live hart: S5a releases none and does not read it
        assert "released" in rep["tiles"][str(skipped)]["skipped"]
        assert not [t for t in txn if t["xy"] in (regs.L2CPU_XY[skipped], regs.L2CPU_XY_TRANSLATED[skipped])]
    assert same


@needs_build
def test_tiles_probe_with_noc_translation_off_reads_no_translated_coordinate(hold):
    rep, ops, txn, same = _run(hold, noc_translation=0)
    assert not [t for t in txn if t["xy"] in set(regs.L2CPU_XY_TRANSLATED.values())]
    assert rep["verdict"] == "clean" and rep["noc_translation"] == 0
    assert all(rep["tiles"][str(t)]["cross_read"].startswith("skipped") for t in TILES)
    assert all(not rep["tiles"][str(t)]["niu0"]["translation_enable"] for t in TILES)
    assert [w for w in rep["warnings"] if "translation" in w] == rep["warnings"] and len(rep["warnings"]) == 1


@needs_build
def test_tiles_probe_json_schema(hold):
    """The report as `probe --tile LIST` prints it (x280d.S5A_SCHEMA 1): chip_session's probe contract keys, the
    ARC part, the verdict, and per tile either a probe section or a skip reason."""
    from x280 import x280d as D
    rep, *_ = _run(hold, enabled_l2cpu=0xE)
    rep = json.loads(json.dumps(rep, default=D._jsonable))
    top = {"probe": str, "schema": int, "backend": str, "tiles_order": list, "l2cpu_reset": int, "pll4": dict,
           "pll4_default": bool, "pll4_why": str, "pll_lock": int, "pll4_walk_ready": bool, "l2cpuclk_mhz": list,
           "error_status0": int, "cable_fault": bool, "l2cpuclk_telemetry_mhz": list, "timer_heartbeat": int, "enabled_l2cpu": int, "noc_translation": int,
           "beu": str, "tiles": dict, "l2cpu_reset_after": int, "pll4_after": dict, "checks": dict, "reasons": list,
           "warnings": list, "dirty": bool, "verdict": str, "attachable": bool, "reads": int, "harvested": list}
    assert set(rep) == set(top) and all(isinstance(rep[k], ty) for k, ty in top.items()), \
        {k: type(rep.get(k)) for k in top}
    assert rep["probe"] == "s5a" and rep["schema"] == D.S5A_SCHEMA == 1 and rep["verdict"] in ("clean", "dirty")
    assert set(rep["checks"]) == {"l2cpu_reset", "pll4_default", "enabled_l2cpu", "wayenable", "complete",
                                   "no_cable_fault"}
    assert set(rep["tiles"]) == {"0", "1", "2", "3"} and rep["tiles_order"] == list(TILES)
    common = {"xy", "enabled", "released", "l2cpuclk_mhz", "probed", "reads"}
    assert set(rep["tiles"]["0"]) == common | {"skipped"}
    probed = common | {"wayenable", "ccache_config", "ccache_config_translated", "cross_read_equal", "ecc",
                       "hart_status", "niu0", "tlb"}
    for t in ("1", "2", "3"):
        tr = rep["tiles"][t]
        assert set(tr) == probed and set(tr["ecc"]) == set(regs.CCACHE0_ECC)
        assert set(tr["niu0"]) == {"noc_node_id", "noc_endpoint_id", "niu_cfg_0", "node", "endpoint",
                                   "translation_enable", "tile_clock_disable"}
        assert set(tr["tlb"]) == {"small", "large", "small_distinct", "large_distinct", "small_nonzero",
                                  "large_nonzero"}
        assert [len(w) for w in tr["tlb"]["small"]] == [4] * 224 and [len(w) for w in tr["tlb"]["large"]] == [3] * 32


@needs_build
def test_tiles_probe_refuses_tiles_the_session_did_not_open(hold):
    from x280 import x280d as D
    s = _session(hold, tiles=(3, 2))
    try:
        s.open()
        with pytest.raises(D.X280Error, match="probe tiles"):
            s.tiles_probe((3, 2, 1))
        assert s.tiles_probe((2,))["tiles_order"] == [2]
    finally:
        s.close()
    with pytest.raises(ValueError, match="releases no hart"):
        D.Session("sim", ledger_path=hold, probe_tiles=TILES)


# --- the CLI: standalone, --via a no-release daemon, refusals ---------------------------------------------------------


@needs_build
def test_probe_cli_tile_list_dry_run_and_the_single_tile_probe_unchanged(hold, capsys):
    from x280 import x280d as D
    assert D.main(["probe", "--tile", "3,2,1,0", "--dry-run", "--ledger", str(hold)]) == 0
    out = capsys.readouterr().out
    rep = json.loads(out.strip().splitlines()[-1])
    assert rep["probe"] == "s5a" and rep["verdict"] == "clean" and rep["backend"] == "sim"
    opslog = out.split("## OPS", 1)[1].split("## TXN", 1)[0]
    assert not [ln for ln in opslog.splitlines() if ln[7:].startswith(("W32", "WC64", "W64", "REFUSED"))]
    assert not [ln for ln in opslog.splitlines() if ln[7:].startswith("IOCTL") and "SET_NOC_CLEANUP" in ln]
    maps = [ln for ln in opslog.splitlines() if ln[7:].startswith("MMAP")]
    assert maps and all(ln.endswith(" RO") or " lim3 " in ln for ln in maps)  # only the session tile's LIM is RW
    assert D.main(["probe", "--json", "--dry-run", "--ledger", str(hold)]) == 0  # chip_session's step 1, as before
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "probe" not in rep and rep["dirty"] is False and set(rep["tiles"]) == {"3"}
    for bad in ("3,3", "4", "3,2,1,0,5"):
        assert D.main(["probe", "--tile", bad, "--dry-run", "--ledger", str(hold)]) == 2


def _daemon(tmp_path, *extra):
    port = _free_port()
    log = open(tmp_path / f"x280d_{port}.log", "w")
    p = subprocess.Popen(NICE + [PY, str(X280D), "serve", "--backend", "sim", "--port", str(port), "--log-dir",
                                 str(tmp_path / f"x280d_{port}"), *extra], cwd=RT, env=_env(), stdout=log,
                         stderr=subprocess.STDOUT)
    url = f"tcp://127.0.0.1:{port}"
    for _ in range(120):
        r = subprocess.run(NICE + [PY, str(X280D), "ping", "--via", url, "--timeout-s", "1"], cwd=RT, env=_env(),
                           capture_output=True)
        if r.returncode == 0:
            return p, url
        assert p.poll() is None, (tmp_path / f"x280d_{port}.log").read_text()[-2000:]
    raise AssertionError("x280d did not answer ping")


def _stop(p):
    p.terminate()
    try:
        return p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()
        return p.wait(5)


def _probe_via(url, tiles="3,2,1,0"):
    r = subprocess.run(NICE + [PY, str(X280D), "probe", "--tile", tiles, "--via", url], cwd=RT, env=_env(),
                       capture_output=True, text=True, timeout=120)
    return r.returncode, json.loads(r.stdout.strip().splitlines()[-1])


@needs_build
def test_probe_via_a_no_release_daemon_with_probe_tiles(tmp_path):
    """jobs/s5a_probe.sh's two probes: `probe --tile LIST --via` on `serve --no-release --probe-tiles 3,2,1,0` (the
    session's keeper fd); a daemon without --probe-tiles answers with an error (exit 1); --probe-tiles without
    --no-release is refused before anything opens (exit 2)."""
    pytest.importorskip("zmq")
    p, url = _daemon(tmp_path, "--tile", "3", "--no-release", "--probe-tiles", "3,2,1,0")
    try:
        rc, rep = _probe_via(url)
        assert rc == 0 and rep["verdict"] == "clean" and rep["tiles_order"] == [3, 2, 1, 0]
        assert all(rep["tiles"][str(t)]["probed"] for t in TILES)
        rc2, rep2 = _probe_via(url)
        assert rc2 == 0 and M._diffs(M._probe_view(rep), M._probe_view(rep2)) == []  # repeatable
        rc, rep = _probe_via(url, "0,1")
        assert rc == 0 and rep["tiles_order"] == [0, 1] and set(rep["tiles"]) == {"0", "1"}
    finally:
        assert _stop(p) == 0
    p, url = _daemon(tmp_path, "--tile", "3", "--no-release")
    try:
        rc, rep = _probe_via(url)
        assert rc == 1 and "--probe-tiles" in rep["error"]
    finally:
        assert _stop(p) == 0
    for extra in (["--probe-tiles", "3,2,1,0"], ["--no-release", "--probe-tiles", "3,4"]):
        r = subprocess.run(NICE + [PY, str(X280D), "serve", "--backend", "sim", "--port", str(_free_port()), *extra],
                           cwd=RT, env=_env(), capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "--probe-tiles" in r.stderr, r.stderr[-2000:]


# --- m3_jobs s5a-summary and the job's wiring -------------------------------------------------------------------------


def _probe_rep(hold, **kw) -> dict:
    from x280 import x280d as D
    rep, *_ = _run(hold, **kw)
    return json.loads(json.dumps(rep, default=D._jsonable))


def _s5a_dir(tmp_path, keeper, tt, digests=({"a": "x"}, {"a": "x"}), p50=(40.0, 40.4)) -> Path:
    d = tmp_path / "sd"
    d.mkdir(exist_ok=True)
    (d / "s5a_probe_keeper.json").write_text(json.dumps(keeper) + "\n")
    (d / "s5a_probe_tt.json").write_text(json.dumps(tt) + "\n")
    for i, dig in enumerate(digests):
        (d / f"s5a_tt_digest_{i}.json").write_text(json.dumps(dig))
    for i, v in enumerate(p50):
        (d / f"s5a_tt_lat_{i}.json").write_text(json.dumps({"latency": {"rtt_ms": {"p50": v, "max": v + 3}}}))
    (d / "s5a_tt_lat_probe.json").write_text(json.dumps({"latency": {"rtt_ms": {"p50": 41.0, "max": 47.5}}}))
    return d


@needs_build
def test_s5a_summary_go_and_each_no_go(tmp_path, hold):
    rep = _probe_rep(hold)
    s = M.s5a_summary(_s5a_dir(tmp_path, rep, rep), {"probe_keeper": "0", "probe_tt": "0"})
    assert s["pass"] and s["s5b"]["go"] and s["probes_equal"] and s["tt_bit_identical"] and s["tt_p50_within_2pct"]
    assert s["lines"][-1] == "S5a -> S5b: GO" and s["tt_max_ms_during_probe"] == 47.5
    assert [ln.split(":")[0] for ln in s["lines"][:4]] == [f"S5a L2CPU{t} ({regs.L2CPU_XY[t][0]},{regs.L2CPU_XY[t][1]})"
                                                           for t in TILES]
    assert "small" not in s["tiles"]["3"]["tlb"] and s["tiles"]["3"]["tlb"]["small_nonzero"] == 0
    tt = json.loads(json.dumps(rep))
    tt["tiles"]["1"]["tlb"]["small"][9][2] = 0x1C8  # a slot changed between the probes
    tt["timer_heartbeat"] += 50  # never compared
    s = M.s5a_summary(_s5a_dir(tmp_path, rep, tt), {"probe_keeper": "0", "probe_tt": "0"})
    assert not s["pass"] and s["probe_diffs"] == ["tiles.1.tlb.small[9][2]: 0 -> 456"]
    assert s["lines"][-1].startswith("S5a -> S5b: NO-GO: the two probes differ (1)")
    s = M.s5a_summary(_s5a_dir(tmp_path, rep, rep, digests=({"a": "x"}, {"a": "y"}), p50=(40.0, 41.0)), {})
    assert not s["tt_bit_identical"] and not s["tt_p50_within_2pct"] and len(s["s5b"]["why"]) == 2
    harv = _probe_rep(hold, enabled_l2cpu=0x7)
    s = M.s5a_summary(_s5a_dir(tmp_path, harv, harv), {"probe_keeper": "4", "probe_tt": "4"})
    assert not s["pass"] and s["harvested"] == [3] and s["probes_equal"]
    assert any("re-plan with 3" in n for n in s["s5b"]["notes"]) and "skipped, not enabled" in s["lines"][0]
    (tmp_path / "sd" / "s5a_probe_tt.json").unlink()
    s = M.s5a_summary(tmp_path / "sd", {"probe_keeper": "4"})
    assert not s["pass"] and s["probe_tt"]["verdict"] == "no report" and s["probe_diffs"] is None


@needs_build
def test_s5a_summary_cli_is_stdlib_only(tmp_path, hold):
    rep = _probe_rep(hold)
    d = _s5a_dir(tmp_path, rep, rep)
    r = subprocess.run(["python3", "-I", str(M3J), "s5a-summary", str(d), "probe_keeper=0", "probe_tt=0"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((d / "s5a_summary.json").read_text())["lines"][-1] == "S5a -> S5b: GO"


def test_s5a_job_header_and_wiring():
    text = (JOBS / "s5a_probe.sh").read_text()
    assert ('CHIP_X280D_SERVE_ARGS="serve --port 5558 --tile 3 --no-release --probe-tiles 3,2,1,0" \\\n#       $DEVX '
            '$RT/scripts/chip_session.sh --x280 probe --vla none --debug-endpoints --job $RT/jobs/s5a_probe.sh '
            '--tag s5a') in text
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    for part in ("need_x280 probe", "need_debug_endpoints", "x280_step s5a_probe_keeper probe --tile",
                 "x280_step s5a_probe_tt probe --tile", 'm3_jobs.py" s5a-summary', '"$CHIP_START_VLA" tt',
                 "tt_digest", "tt_latency"):
        assert part in code, part
    assert "$DEV " not in code and "$DEVX" not in code and "tt-smi" not in code and "/dev/tenstorrent" not in code
    assert os.access(JOBS / "s5a_probe.sh", os.X_OK)


def _job(tmp_path, **env):
    sd = tmp_path / "session"
    sd.mkdir(exist_ok=True)
    e = {k: v for k, v in os.environ.items() if not k.startswith("CHIP_") and k != "DISPLAY"}
    e.update(CHIP_SESSION_DIR=str(sd), CHIP_X280_URL="tcp://127.0.0.1:9", CHIP_SESSION_ID=f"test-s5a-{os.getpid()}",
             CHIP_DEBUG_ENDPOINTS="1")
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(NICE + ["bash", str(JOBS / "s5a_probe.sh")], env=e, capture_output=True, text=True,
                          timeout=60), sd


def test_s5a_job_refuses_other_sessions_before_any_step(tmp_path):
    """Exit 2 with nothing run: another --x280 mode, no --debug-endpoints, or a daemon without --probe-tiles."""
    for env, why in (({"CHIP_X280_MODE": "boot:3"}, "needs --x280 probe"),
                     ({"CHIP_X280_MODE": "probe", "CHIP_DEBUG_ENDPOINTS": "0"}, "--debug-endpoints"),
                     ({"CHIP_X280_MODE": "probe", "CHIP_X280D_SERVE_ARGS": "serve --port 9 --tile 3 --no-release"},
                      "--probe-tiles")):
        r, sd = _job(tmp_path, **env)
        assert r.returncode == 2 and why in r.stderr, r.stderr[-2000:]
        assert not list(sd.glob("s5a_*"))
