"""A simulated L2CPU tile for `x280d --backend sim` and `--dry-run` (PLAN.md WP-3.4, T-3.3, R37).

Why: x280d must be exercised end to end before any hart is released, through the *same* code path it uses
on the chip. So the "chip" here is kmd.FakeDevice (memfd per 2 MiB NoC region, KMD ioctl/mmap/release
rules) and the "hart" is the monitor's own x86 build (x280/fw/build/x86/libmonitor_x86.so: monitor.c and
trap.c compiled unchanged with -DMON_X86, plus the real kernel). The hart runs in a thread (ctypes drops
the GIL) against the LIM bytes of the FakeDevice region, so x280d writes obs lines with libwcio WC
stores, bumps seq_req with a UC write and polls L1, exactly as on the device (§1.4.4).

What the chip side emulates, with the primary source of each behaviour:
- Release: the hart leaves reset when x280d's one L2CPU_RESET RMW sets bit 4+tile (ISA
  L2CPUTile/README.md:35-46). Before it starts, the boot artefacts x280d wrote are checked the way the
  silicon would consume them: reset handler address of hart 0 = 0x0800_0000 (MemoryMap.md:111), RNMI
  handler / exception handler = GP-scratch park stub / spin stub (RNMIs.md:48-59), GP scratch = the park
  stub, LIM image = monitor.bin. A mismatch leaves the hart "dead" (no heartbeat), as garbage would.
- RNMI: bit h of the trigger word 0x2001_0414 is level-triggered and taken only while mnstatus.NMIE = 1
  (RNMIs.md:15-22); the hart then runs the code in the GP scratch (its RNMI handler address, RNMIs.md:48-59).
  The park stub spins while the bit stays set and jumps to crt0 at the fixed 0x0800_0000 when it clears
  (park_stub.S), i.e. a warm entry (mon_x86_reenter, mncause = 2^63 | 2, RNMIs.md:76-77); stage0.bin there (E1F,
  before the host swapped in the park stub) re-zeroes LIM; anything else leaves the hart dead. A lost RNMI
  (NMIE = 0) leaves the hart where it was.
- KMD cleanup: FakeDevice already performs the registered SET_NOC_CLEANUP write when the keeper fd is
  released (chardev.c:1092-1115), so the monitor sees 'GONE' in L38 and idles.
- CMFW telemetry: TIMER_HEARTBEAT advances at 10 Hz (fw telemetry.c:278, 549), so x280d's epoch and E0
  "increasing" checks see a live counter; L2CPUCLK0..3 follow PLL4 at the same rate (telemetry.c:504-522). The
  other chip-panel tags (AICLK, temperature, power) are 0: there is no chip to report, so x280d's sim status
  stream carries telemetry None with telemetry_source 'none (sim)' (verify:all D4).
- E1F stage 0 (contingency, DP3(d)): with the reset vector at the GP scratch and stage0.bin there, the hart
  zero-fills LIM, writes the done word 0x081E_0000 to 0x081D_FFC0 and spins on the entry word 0x081D_FF80
  (stage0.S); a non-zero entry word = crt0 (a cold entry: no RNMI was taken).
- Payloads: the x86 monitor cannot execute riscv code, so a CODE LOAD whose bytes equal one of the
  firmware payload images (build/payloads/*.bin) registers the matching x86 stand-in at that address
  (plat_x86.c mon_x86_register_payload kinds 1-4, 6 = mh_bench); any other code faults on CALL, as unknown code
  would.
- Harts 1-3 (mh.c, the 4-hart bench design): out of reset they wait in wfi, in crt0 (method A) or in stage 0's
  wfi loop with that GP-scratch line in their L1I (E1F, stage0.S). Bit h of the trigger word is taken while their
  NMIE = 1: with the --harts 4 vectors (RNMI handler = 0x0800_0000, exception handler = MH_SEC_RNMI_EXC) the hart
  spins in crt0's secondary path until the bit clears, then mh_secondary_main runs as a thread of the x86 build
  (mon_x86_secondary_start, mncause 2^63 | 2, mnepc = stage 0's wfi 0x2001_013A after E1F); after E1F the boot's
  park-stub vectors strand it (the stale stage-0 line sends it back to wfi with NMIE = 0: never again usable);
  after method A they lead through the park stub into the same crt0 path. A running secondary with NMIE = 1 is
  re-adopted the same way (recovery of a hung one); with NMIE = 0 the RNMI is lost.

Not emulated: cycle counts (the x86 fake mcycle advances 16 per read, so x280d reports the host-measured
mailbox turnaround as compute_us under the sim), ECC, BEU events, PLL relock effects, weak ordering.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional

from x280 import regs
from x280.host import kmd as K

FW_BUILD = Path(__file__).resolve().parent / "fw" / "build"
LIB = FW_BUILD / "x86" / "libmonitor_x86.so"
PAYLOAD_KINDS = {"checksum": 1, "illegal": 2, "hang": 3, "lim_bench": 4, "mh_bench": 6}  # mon_x86_register_payload
X86_RUNNING, X86_PARKED, X86_HUNG = 0, 1, 2  # plat_x86.c enum
RNMI_MNCAUSE = (1 << 63) | 2  # RNMI from the per-hart trigger bit (ISA RNMIs.md:15, 76-77)
MNSTATUS_NMIE = 1 << 3  # ISA RNMIs.md:84
MTIME_STEP = 50  # fake mtime ticks per read: heartbeat (every 500 ticks, monitor.c HB_TICKS) every ~160 polls
MCYCLE_STEP = 16  # fake mcycle per read (meaningless; see the module docstring)
WATCH_PERIOD_S = 0.001
STAGE0_ENTRY_WORD, STAGE0_DONE_WORD = regs.STAGE0_ENTRY_WORD, regs.STAGE0_DONE_WORD  # stage0.S
SWITCH_INTERVAL_S = 0.0002


def _lib_signatures(lib: ctypes.CDLL) -> ctypes.CDLL:
    u32, u64, i32, vp = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p
    for name, res, args in (
        ("mon_x86_attach", i32, [vp, u64]), ("mon_x86_boot", i32, []), ("mon_x86_reenter", i32, [u64, u64]),
        ("mon_x86_run_thread", i32, [vp, u64]), ("mon_x86_set_clock", None, [u64, u64, u64]),
        ("mon_x86_set_csrs", None, [u64, u32, u32]), ("mon_x86_mnstatus", u64, []),
        ("mon_x86_use_fake_kernel", None, [i32]), ("mon_x86_register_payload", i32, [u64, i32, vp]),
        ("mon_x86_clear_payloads", None, []), ("mon_x86_kernel_is_stub", u32, []),
        ("mon_x86_variants_built", u32, []), ("mon_x86_build_id", u32, []), ("mon_x86_mtime", u64, []),
        ("mon_x86_secondary_start", i32, [u32, u64, u64]), ("mon_x86_secondary_stop", None, [u32]),
        ("mon_x86_secondary_stop_all", None, []), ("mon_x86_secondary_running", i32, [u32]),
        ("mon_x86_secondary_mnstatus", u64, [u32]), ("mon_x86_secondary_hang", None, [u32, i32]),
        ("mon_x86_set_beu_hart", None, [u32, u32]),
    ):
        f = getattr(lib, name)
        f.restype, f.argtypes = res, args
    return lib


def load_private_lib(path: Path = LIB, tag: str = "hart") -> ctypes.CDLL:
    """dlopen a private copy: the monitor's state is file-scope globals, so each simulated hart needs its own
    instance (a second dlopen of the same path would return the same one). The copy lives in a memfd, so
    nothing is written to the filesystem and a rebuild of the .so cannot change a running hart."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing: run make -C {FW_BUILD.parent} x86")
    fd = K.memfd(f"libmonitor_x86_{tag}")
    try:
        os.write(fd, Path(path).read_bytes())
        lib = ctypes.CDLL(f"/proc/self/fd/{fd}", mode=os.RTLD_LOCAL | os.RTLD_NOW)
    finally:
        os.close(fd)  # the mapping keeps the image alive
    return _lib_signatures(lib)


def known_payloads(build: Path = FW_BUILD) -> dict:
    """sha256 of each firmware payload image -> its x86 stand-in kind."""
    out = {}
    for name, kind in PAYLOAD_KINDS.items():
        p = build / "payloads" / f"{name}.bin"
        if p.exists():
            out[hashlib.sha256(p.read_bytes()).hexdigest()] = (name, kind)
    return out


class SimHart:
    """One simulated L2CPU tile: hart 0 (the monitor's poll loop in a thread) and harts 1-3 (cold in wfi until an
    RNMI adopts them: then mh_secondary_main threads of the same x86 build, see the module docstring)."""

    def __init__(self, chip: "SimChip", tile: int, lib_path: Path, image: Optional[bytes],
                 park_stub: Optional[bytes], cpu: Optional[int]):
        self.chip, self.tile, self.xy = chip, tile, regs.L2CPU_XY[tile]
        self.image, self.park_stub = image, park_stub
        self.cpu = cpu
        self.lib = load_private_lib(lib_path, f"t{tile}")
        view = chip._region(self.xy, regs.LIM_BASE)[1]  # the same memfd the LIM window maps
        self._lim = (ctypes.c_uint8 * regs.TLB_2M).from_buffer(view)
        if self.lib.mon_x86_attach(ctypes.addressof(self._lim), regs.LIM_SIZE) != 0:
            raise RuntimeError("mon_x86_attach failed")
        self.lib.mon_x86_use_fake_kernel(0)  # the real kernel (wbc_mlp.c) linked into libmonitor_x86.so
        self.lib.mon_x86_clear_payloads()
        self.lib.mon_x86_set_clock(0, MTIME_STEP, MCYCLE_STEP)
        self.lib.mon_x86_set_csrs((2 << 62) | (1 << 21) | 0x112D, 64, 0)  # RV64, V present, vlenb 64 (E2 expects 64)
        self.state = "reset"  # reset | running | parked | hung | rnmi | dead
        self.why = ""
        self.boots = self.rnmis = self.rnmi_lost = self.reentries = 0
        self.events: list = []
        self._stop = ctypes.c_uint32(0)
        self._thread: Optional[threading.Thread] = None
        self._exit = X86_RUNNING
        self._lock = threading.Lock()
        self._payloads = known_payloads(Path(lib_path).resolve().parents[1])
        st0 = Path(lib_path).resolve().parents[1] / "stage0.bin"
        self.stage0 = st0.read_bytes() if st0.exists() else None
        # harts 1-3: reset | cold (crt0 wfi, method A) | cold_e1f (stage 0's wfi) | rnmi (spinning on the trigger bit in
        # crt0's secondary path) | running (mh_secondary_main thread) | stranded | dead
        self.sec = {h: {"state": "reset", "why": "", "rnmis": 0, "lost": 0, "starts": 0, "mnepc": 0}
                    for h in regs.MH_SEC_HARTS}

    # -- run thread --
    def _run(self) -> None:
        """The monitor's poll loop in C until stopped, parked or hung. It never comes back to Python while it
        runs: a thread that needs the GIL between bursts waits up to the 5 ms switch interval behind x280d's
        polling thread, which is exactly the latency being measured."""
        if self.cpu is not None:
            try:
                os.sched_setaffinity(0, {self.cpu})  # this thread only (Linux): the spin never shares x280d's CPU
            except OSError:
                pass
        self._exit = self.lib.mon_x86_run_thread(ctypes.addressof(self._stop), 1 << 62)

    def _start(self) -> None:
        self._stop.value = 0
        self._exit = X86_RUNNING
        self._thread = threading.Thread(target=self._run, name=f"x280sim-hart{self.tile}", daemon=True)
        self._thread.start()
        self.state = "running"

    def _halt(self) -> None:
        t = self._thread
        if t is not None:
            self._stop.value = 1
            t.join(10.0)
            if t.is_alive():
                raise RuntimeError(f"sim hart {self.tile} did not stop")
            self._thread = None
            self._stop.value = 0

    def _event(self, what: str) -> None:
        self.events.append((time.monotonic(), what))
        del self.events[:-200]

    # -- what the silicon would check at release / warm entry --
    def _artefacts_ok(self, reset_pc: int = regs.RESET_PC, gp: Optional[bytes] = None) -> str:
        c = self.chip
        lo, hi = c.peek32(self.xy, regs.reset_vector(0)), c.peek32(self.xy, regs.reset_vector(0) + 4)
        if (hi << 32 | lo) != reset_pc:
            return f"reset handler address 0x{hi << 32 | lo:x} != 0x{reset_pc:x}"
        if gp is not None:
            return "" if c.peek(self.xy, regs.GP_SCRATCH, len(gp)) == gp else "GP scratch does not hold stage0.bin"
        for a, want in ((regs.rnmi_handler(0), regs.PARK_STUB), (regs.rnmi_exc_handler(0), regs.RNMI_EXC_STUB)):
            v = c.peek32(self.xy, a) | c.peek32(self.xy, a + 4) << 32
            if v != want:
                return f"RNMI handler word 0x{a:x} = 0x{v:x} != 0x{want:x}"
        if self.park_stub is not None and c.peek(self.xy, regs.GP_SCRATCH, len(self.park_stub)) != self.park_stub:
            return "GP scratch does not hold the park stub"
        if self.image is not None and c.peek(self.xy, regs.IMAGE_BASE, len(self.image)) != self.image:
            return "LIM image differs from monitor.bin"
        return ""

    def service(self) -> None:
        """One watcher step: release, RNMI entry/exit, run-thread exit."""
        with self._lock:
            c = self.chip
            if self.state == "stage0":
                entry = c.peek32(self.xy, STAGE0_ENTRY_WORD)
                if entry:
                    why = "" if entry == regs.RESET_PC else f"stage-0 entry word 0x{entry:x}"
                    if not why and self.image is not None and c.peek(self.xy, regs.IMAGE_BASE,
                                                                     len(self.image)) != self.image:
                        why = "LIM image differs from monitor.bin"
                    if why or self.lib.mon_x86_boot() != X86_RUNNING:
                        self.state, self.why = "dead", why or "mon_x86_boot did not return RUNNING"
                        return
                    self.boots += 1
                    self._event("stage 0 -> crt0 (cold entry)")
                    self._start()
                return
            if self.state == "reset":
                if c.peek32(regs.ARC_XY, regs.L2CPU_RESET) & regs.release_bit(self.tile):
                    lo = c.peek32(self.xy, regs.reset_vector(0))
                    if self.stage0 is not None and lo == regs.STAGE0_ENTRY:
                        why = self._artefacts_ok(regs.STAGE0_ENTRY, self.stage0)
                        if why:
                            self.state, self.why = "dead", why
                            return
                        self._stage0_run("release: stage 0 zero-filled LIM, waiting on the entry word")
                        self._secondaries_released("cold_e1f")
                        return
                    why = self._artefacts_ok()
                    if why:
                        self.state, self.why = "dead", why
                        self._event(f"release: hart does not start: {why}")
                        return
                    self.boots += 1
                    if self.lib.mon_x86_boot() != X86_RUNNING:
                        self.state, self.why = "dead", "mon_x86_boot did not return RUNNING"
                        return
                    self._event("release: cold entry")
                    self._start()
                    self._secondaries_released("cold")
                return
            self._service_secondaries()
            if self.state == "running" and self._thread is not None and not self._thread.is_alive():
                self._thread = None
                self.state = {X86_PARKED: "parked", X86_HUNG: "hung"}.get(self._exit, "hung")
                self._event(f"run thread exit -> {self.state}")
            trig = c.peek32(self.xy, regs.RNMI_TRIGGER) & 1
            if trig and self.state in ("running", "parked", "hung"):
                if not self.lib.mon_x86_mnstatus() & MNSTATUS_NMIE:
                    self.rnmi_lost += 1  # RNMIs.md:18: taken only while NMIE = 1
                    return
                self._halt()
                self.rnmis += 1
                self.state = "rnmi"  # in the park stub, spinning on the trigger bit
                self._event("RNMI taken: park stub")
            elif self.state == "rnmi" and not trig:
                # the RNMI handler is whatever the GP scratch holds (RNMIs.md:48-59): the park stub jumps to crt0 at
                # the fixed 0x0800_0000 (park_stub.S); E1F's stage 0 re-zeroes LIM; anything else is fatal
                gp = c.peek(self.xy, regs.GP_SCRATCH, regs.GP_SCRATCH_SIZE)
                if self.stage0 is not None and gp == self.stage0:
                    self._stage0_run("RNMI during stage 0: stage 0 again (re-zeroes LIM)")
                    return
                if self.park_stub is not None and gp != self.park_stub:
                    self.state, self.why = "dead", "RNMI handler code in the GP scratch is not the park stub"
                    return
                mnepc = c.peek32(self.xy, regs.MB_CMD + 32) | c.peek32(self.xy, regs.MB_CMD + 36) << 32
                if self.lib.mon_x86_reenter(RNMI_MNCAUSE, mnepc or regs.LIM_BASE) != X86_RUNNING:
                    self.state, self.why = "dead", "mon_x86_reenter did not return RUNNING"
                    return
                self.reentries += 1
                self._event("park stub -> crt0 (warm entry)")
                self._start()

    # -- harts 1-3 --
    def _secondaries_released(self, state: str) -> None:
        for s in self.sec.values():
            if s["state"] == "reset":
                s["state"] = state

    def _sec_vectors(self, h: int) -> tuple:
        c = self.chip
        rd = (lambda a: c.peek32(self.xy, a) | c.peek32(self.xy, a + 4) << 32)
        return rd(regs.rnmi_handler(h)), rd(regs.rnmi_exc_handler(h))

    def _service_secondaries(self) -> None:
        """Harts 1-3 against the level-triggered trigger bits (called under the lock)."""
        trig = self.chip.peek32(self.xy, regs.RNMI_TRIGGER)
        for h, s in self.sec.items():
            bit, st = (trig >> h) & 1, s["state"]
            if st == "running" and not self.lib.mon_x86_secondary_running(h):
                s["state"], s["why"] = "dead", "mh_secondary_main thread ended"
                continue
            if bit and st in ("cold", "cold_e1f", "running"):
                if st == "running" and not self.lib.mon_x86_secondary_mnstatus(h) & MNSTATUS_NMIE:
                    s["lost"] += 1  # RNMIs.md:18: taken only while NMIE = 1
                    continue
                s["rnmis"] += 1
                handler, exc = self._sec_vectors(h)
                gp = self.chip.peek(self.xy, regs.GP_SCRATCH, regs.GP_SCRATCH_SIZE)
                via_crt0 = handler == regs.RESET_PC and exc == regs.MH_SEC_RNMI_EXC
                via_park = handler == regs.PARK_STUB and self.park_stub is not None and gp == self.park_stub
                if st == "cold_e1f" and not via_crt0:
                    # its L1I still holds stage 0's line at 0x2001_0100 (only its own fence.i would drop it, Caches.md:22):
                    # stage 0 again, `bnez` back to wfi with NMIE = 0, out of reach until a chip reset
                    s["state"], s["why"] = "stranded", f"RNMI through 0x{handler:x} after E1F: stale stage-0 line, NMIE 0"
                    self._event(f"hart {h} stranded")
                    continue
                if not (via_crt0 or via_park):
                    s["state"], s["why"] = "dead", f"RNMI handler 0x{handler:x} / exception 0x{exc:x}"
                    continue
                if st == "running":
                    self.lib.mon_x86_secondary_stop(h)
                s["mnepc"] = regs.STAGE0_SEC_WFI if st == "cold_e1f" else 0
                s["state"] = "rnmi"
                self._event(f"hart {h} RNMI taken (handler 0x{handler:x})")
            elif not bit and st == "rnmi":
                if self.image is not None and self.chip.peek(self.xy, regs.IMAGE_BASE, len(self.image)) != self.image:
                    s["state"], s["why"] = "dead", "LIM image differs from monitor.bin"
                    continue
                if self.lib.mon_x86_secondary_start(h, RNMI_MNCAUSE, s["mnepc"]) != 0:
                    s["state"], s["why"] = "dead", "mon_x86_secondary_start failed"
                    continue
                s["state"] = "running"
                s["starts"] += 1
                self._event(f"hart {h}: crt0 secondary path -> mh_secondary_main")

    def inject_secondary_hang(self, h: int, on: bool = True) -> None:
        """A wedged secondary: its idle loop stops (heartbeat frozen); only its RNMI bit brings it back."""
        self.lib.mon_x86_secondary_hang(h, int(on))

    def _stage0_run(self, what: str) -> None:
        ctypes.memset(ctypes.addressof(self._lim), 0, regs.LIM_SIZE)  # one vse8.v per line (stage0.S)
        self.chip.poke32(self.xy, STAGE0_DONE_WORD, regs.LIM_END)
        self.state = "stage0"
        self._event(what)

    # -- test hooks --
    def inject_hang(self) -> None:
        """A wedged hart: the monitor stops polling, the heartbeat freezes; only an RNMI brings it back."""
        with self._lock:
            if self.state == "running":
                self._halt()
                self.state = "hung"
                self._event("injected hang")

    def note_code_load(self, dst: int, data: bytes) -> Optional[str]:
        name_kind = self._payloads.get(hashlib.sha256(bytes(data)).hexdigest())
        if name_kind is None:
            return None
        name, kind = name_kind
        if self.lib.mon_x86_register_payload(dst, kind, None) != 0:
            raise RuntimeError("payload table full")
        return name

    def stop(self) -> None:
        with self._lock:
            self._halt()
            self.lib.mon_x86_secondary_stop_all()
            if self.state == "running":
                self.state = "parked"

    def status(self) -> dict:
        return {"tile": self.tile, "state": self.state, "why": self.why, "boots": self.boots, "rnmis": self.rnmis,
                "rnmi_lost": self.rnmi_lost, "reentries": self.reentries, "cpu": self.cpu,
                "kernel_is_stub": bool(self.lib.mon_x86_kernel_is_stub()),
                "variants_built": int(self.lib.mon_x86_variants_built()),
                "build_id": int(self.lib.mon_x86_build_id()),
                "harts": {str(h): dict(s) for h, s in self.sec.items()}}


class SimChip(K.FakeDevice):
    """FakeDevice.default_chip() plus one SimHart per tile and a 1 ms watcher thread."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.harts: dict = {}
        self._watch: Optional[threading.Thread] = None
        self._halt_watch = threading.Event()
        self._t0 = time.monotonic()
        self._hb0 = 0

    @classmethod
    def create(cls, tiles: tuple = (regs.DEFAULT_TILE,), *, image: Optional[bytes] = None,
               park_stub: Optional[bytes] = None, lib: Path = LIB, cpus: Optional[list] = None,
               **chip_kw) -> "SimChip":
        """A CMFW-after-boot chip (L2CPU_RESET 0x0F, PLL4 defaults, telemetry) with tiles in reset.

        cpus: one CPU per hart for its spin thread (None: the hart thread inherits the caller's CPU mask)."""
        # The watcher is a Python thread that emulates level-triggered hardware (release, RNMI): it must not wait
        # the default 5 ms GIL switch interval behind x280d's polling thread, or a short RNMI pulse is missed.
        sys.setswitchinterval(min(sys.getswitchinterval(), SWITCH_INTERVAL_S))
        chip = cls.default_chip(**chip_kw)
        chip._hb0 = chip.peek32(regs.ARC_XY, chip.peek32(regs.ARC_XY, regs.TELEMETRY_DATA_PTR)
                                + 4 * regs.TAG_TIMER_HEARTBEAT)
        for i, t in enumerate(tiles):
            cpu = cpus[i] if cpus and i < len(cpus) else None
            chip.harts[t] = SimHart(chip, t, lib, image, park_stub, cpu)
        chip._watch = threading.Thread(target=chip._watcher, name="x280sim-watch", daemon=True)
        chip._watch.start()
        return chip

    def _watcher(self) -> None:
        last = None
        while not self._halt_watch.wait(WATCH_PERIOD_S):
            hb = self._hb0 + int((time.monotonic() - self._t0) / regs.TELEMETRY_PERIOD_S)
            if hb != last:  # one CMFW telemetry update (10 Hz): heartbeat + L2CPUCLK from PLL4 (telemetry.c:504-549)
                self.set_heartbeat(hb)
                self.update_l2cpuclk_telemetry()
                last = hb
            for h in list(self.harts.values()):
                try:
                    h.service()
                except Exception as e:  # noqa: BLE001 - a sim bug must show up as a dead hart, not a dead watcher
                    h.state, h.why = "dead", f"sim error: {type(e).__name__}: {e}"

    def hart(self, tile: int) -> SimHart:
        return self.harts[tile]

    def note_code_load(self, tile: int, dst: int, data: bytes) -> Optional[str]:
        return self.harts[tile].note_code_load(dst, data) if tile in self.harts else None

    def status(self) -> dict:
        return {str(t): h.status() for t, h in self.harts.items()}

    def shutdown(self) -> None:
        self._halt_watch.set()
        if self._watch is not None:
            self._watch.join(5.0)
        for h in self.harts.values():
            h.stop()
