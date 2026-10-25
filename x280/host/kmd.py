"""tt-kmd 2.11 host access layer for x280d (PLAN.md WP-3.3, D12): TLB windows, UC/WC mappings,
SET_NOC_CLEANUP, and the write policy of x280/regs.py. It replaces pyluwen, whose UC u32 copies cannot
initialise LIM lines (PLAN X-1, R26).

Why it is shaped this way:
- Only four ioctls exist here: ALLOCATE_TLB, CONFIGURE_TLB, FREE_TLB, SET_NOC_CLEANUP (KMD ioctl.h:25-28).
  No ARC message, power-state or reset ioctl is even defined (D12: x280d sends no ARC messages).
- Windows are allocated and configured once, at open (PLAN §1.4.4: no per-call TLB ioctl).
- The only persistently writable mappings are LIM (UC + WC). ARC, CCACHE/L2PF, BEU and CSM are mapped
  PROT_READ, so a stray store there faults in the host MMU instead of reaching the chip; the few legal
  writes (the boot RMW, the PLL4 steps, L2PF, vectors, GP scratch, RNMI trigger) go through a short-lived
  RW mapping that exists only inside the one function allowed to make them. The external-peripheral
  window is never mapped persistently, because KMD maps a window only from its base (memory.c:1582-1596)
  and that prefix includes reserved alias space. No window targets GDDR (regs._self_check).
- Every device access goes through libwcio (UC 4 B accessors, 64 B AVX-512 non-temporal line stores) and
  is recorded twice: an op log here (ioctls, mmaps, reads, writes, refusals) and libwcio's transaction log
  (size and alignment of every store). x280d --dry-run prints both against FakeDevice.
- The real device is opened only inside a DEVX hold (D13): see `assert_devx_hold`.

Release semantics (KMD chardev.c:1101-1115): closing the keeper fd, or the process dying, makes the KMD
perform the registered SET_NOC_CLEANUP write ('GONE' into L38). The file is released only after the last
mapping is gone, which on process exit happens automatically.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import ctypes
import errno
import fcntl
import itertools
import mmap as _mmap
import os
from pathlib import Path
import time
from typing import Callable, Optional
import zlib

from x280 import regs

DEVICE_PATH = "/dev/tenstorrent/0"
LIBWCIO = Path(__file__).resolve().parent / "libwcio.so"
KEEPER_SETTLE_S = 1.0  # legacy open sends a power message, PLL4 relocks (KMD chardev.c:986-988, 1017-1021; PLAN WP-3.4)

# --- tt-kmd ABI (KMD ioctl.h) ------------------------------------------------------------------

TENSTORRENT_IOCTL_MAGIC = 0xFA  # ioctl.h:12


def _io(nr: int) -> int:
    """_IO(TENSTORRENT_IOCTL_MAGIC, nr): no direction or size bits (ioctl.h:14-31; LUW luwen-kmd/src/ioctl.rs:4, 261)."""
    return (TENSTORRENT_IOCTL_MAGIC << 8) | nr


IOCTL_ALLOCATE_TLB = _io(11)  # ioctl.h:25
IOCTL_FREE_TLB = _io(12)  # ioctl.h:26
IOCTL_CONFIGURE_TLB = _io(13)  # ioctl.h:27
IOCTL_SET_NOC_CLEANUP = _io(14)  # ioctl.h:28
IOCTL_NAMES = {IOCTL_ALLOCATE_TLB: "ALLOCATE_TLB", IOCTL_FREE_TLB: "FREE_TLB",
               IOCTL_CONFIGURE_TLB: "CONFIGURE_TLB", IOCTL_SET_NOC_CLEANUP: "SET_NOC_CLEANUP"}

u8, u16, u32, u64 = ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64


class AllocateTlbIn(ctypes.Structure):  # ioctl.h:271-274
    _fields_ = [("size", u64), ("reserved", u64)]


class AllocateTlbOut(ctypes.Structure):  # ioctl.h:276-282
    _fields_ = [("id", u32), ("reserved0", u32), ("mmap_offset_uc", u64), ("mmap_offset_wc", u64), ("reserved1", u64)]


class AllocateTlb(ctypes.Structure):  # ioctl.h:284-287
    _fields_ = [("in_", AllocateTlbIn), ("out", AllocateTlbOut)]


class FreeTlbIn(ctypes.Structure):  # ioctl.h:289-291
    _fields_ = [("id", u32)]


class FreeTlbOut(ctypes.Structure):  # ioctl.h:293-294 (empty: size 0)
    _fields_ = []


class FreeTlb(ctypes.Structure):  # ioctl.h:296-299
    _fields_ = [("in_", FreeTlbIn), ("out", FreeTlbOut)]


class NocTlbConfig(ctypes.Structure):  # ioctl.h:301-314
    _fields_ = [("addr", u64), ("x_end", u16), ("y_end", u16), ("x_start", u16), ("y_start", u16),
                ("noc", u8), ("mcast", u8), ("ordering", u8), ("linked", u8), ("static_vc", u8),
                ("reserved0", u8 * 3), ("reserved1", u32 * 2)]


class ConfigureTlbIn(ctypes.Structure):  # ioctl.h:316-320
    _fields_ = [("id", u32), ("reserved", u32), ("config", NocTlbConfig)]


class ConfigureTlbOut(ctypes.Structure):  # ioctl.h:322-324
    _fields_ = [("reserved", u64)]


class ConfigureTlb(ctypes.Structure):  # ioctl.h:326-329
    _fields_ = [("in_", ConfigureTlbIn), ("out", ConfigureTlbOut)]


class SetNocCleanup(ctypes.Structure):  # ioctl.h:350-360; argsz must equal sizeof = 32 (chardev.c:515)
    _pack_ = 1
    _fields_ = [("argsz", u32), ("flags", u32), ("enabled", u8), ("x", u8), ("y", u8), ("noc", u8),
                ("reserved0", u32), ("addr", u64), ("data", u64)]


assert ctypes.sizeof(SetNocCleanup) == 32 and ctypes.sizeof(ConfigureTlb) == 48 and ctypes.sizeof(AllocateTlb) == 48

IOCTL_TYPES = {IOCTL_ALLOCATE_TLB: AllocateTlb, IOCTL_FREE_TLB: FreeTlb,
               IOCTL_CONFIGURE_TLB: ConfigureTlb, IOCTL_SET_NOC_CLEANUP: SetNocCleanup}

PROT_READ, PROT_WRITE, MAP_SHARED = _mmap.PROT_READ, _mmap.PROT_WRITE, _mmap.MAP_SHARED

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.restype = ctypes.c_int
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
MAP_FAILED = ctypes.c_void_p(-1).value


def libc_mmap(fd: int, length: int, prot: int, offset: int) -> int:
    addr = _libc.mmap(None, length, prot, MAP_SHARED, fd, offset)
    if addr in (None, MAP_FAILED):
        e = ctypes.get_errno()
        raise OSError(e, f"mmap(len=0x{length:x}, off=0x{offset:x}): {os.strerror(e)}")
    return addr


def libc_munmap(addr: int, length: int) -> None:
    if _libc.munmap(addr, length) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"munmap: {os.strerror(e)}")


def memfd(name: str) -> int:
    """Anonymous shared-memory fd (FakeDevice regions). Some standalone CPython builds (e.g. simpy's uv venv)
    lack os.memfd_create, so fall back to glibc's memfd_create(2) through ctypes."""
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, os.MFD_CLOEXEC)
    fn = _libc.memfd_create
    fn.restype, fn.argtypes = ctypes.c_int, [ctypes.c_char_p, ctypes.c_uint]
    fd = fn(name.encode(), 1)  # MFD_CLOEXEC = 1 (linux/memfd.h)
    if fd < 0:
        e = ctypes.get_errno()
        raise OSError(e, f"memfd_create: {os.strerror(e)}")
    return fd


# --- errors ------------------------------------------------------------------------------------


class KmdError(RuntimeError):
    pass


class PolicyError(KmdError):
    """A read or write the policy refuses; nothing reached the device."""


class InterlockError(KmdError):
    """A §5.2 precondition does not hold (PLL4, ENABLED_L2CPU, settle time, L2CPU_RESET state)."""


class DeviceAccessRefused(KmdError):
    """The real device may be opened only inside a DEVX hold (D13)."""


class WcioError(KmdError):
    pass


# --- libwcio -----------------------------------------------------------------------------------


class WcioTxn(ctypes.Structure):  # struct wcio_txn in wcio.c
    _fields_ = [("seq", u64), ("addr", u64), ("size", u32), ("value", u32), ("op", u8), ("align64", u8),
                ("align4", u8), ("reserved", u8 * 5)]


WCIO_OPS = {1: "WC64", 2: "SFENCE", 3: "UC_W32", 4: "UC_R32", 5: "REFUSED", 6: "UC_W64", 7: "WC32"}
WCIO_ABI = 3  # wcio.c WCIO_ABI

# LIM init stores (kmd.lim_init `store`; S1d, jobs/s1d.sh, compares them): name -> (mapping, libwcio loop, sfence after
# every line, what reaches the mapping). Only "a64" is the boot's (PLAN §5.2 step 2, method A); the others exist so S1d
# can find a host write that leaves the CCACHE's LIM ECC valid (S1c: a64 does not).
LIM_INIT_STORES = {
    "a64": ("wc", "nt64", True, "one 64 B non-temporal zmm store + sfence per line (method A)"),
    "a64nf": ("wc", "nt64", False, "64 B non-temporal zmm stores, one sfence per call"),
    "wc2x32": ("wc", "2x32", False, "two regular 32 B ymm stores per line (UMD memcpy_to_device), one sfence per call"),
    "uc8": ("uc", "w64", None, "eight aligned 8 B UC stores per line"),
}


class Wcio:
    """ctypes binding of libwcio.so (x280/host/wcio.c)."""

    def __init__(self, path: Path = LIBWCIO):
        if not Path(path).exists():
            raise WcioError(f"{path} missing: run make -C {Path(path).parent}")
        self.lib = lib = ctypes.CDLL(str(path))
        if lib.wcio_abi() != WCIO_ABI:
            raise WcioError(f"libwcio ABI {lib.wcio_abi()} != {WCIO_ABI}: rebuild with make -C {Path(path).parent}")
        vp, c_int = ctypes.c_void_p, ctypes.c_int
        for name, res, args in (
            ("wcio_wc_store_lines", c_int, [vp, vp, u64, c_int]), ("wcio_sfence", None, []),
            ("wcio_uc_write32", c_int, [vp, u32]), ("wcio_uc_read32", c_int, [vp, ctypes.POINTER(u32)]),
            ("wcio_uc_read64_hlh", c_int, [vp, ctypes.POINTER(u64)]), ("wcio_avx512_ok", c_int, []),
            ("wcio_uc_read_words", c_int, [vp, u32, ctypes.POINTER(u32)]),
            ("wcio_wc_store_lines_2x32", c_int, [vp, vp, u64, c_int]), ("wcio_uc_store64", c_int, [vp, vp, u64]),
            ("wcio_avx_ok", c_int, []),
            ("wcio_force_no_avx512", None, [c_int]), ("wcio_log_enable", c_int, [u64]),
            ("wcio_log_total", u64, []), ("wcio_log_capacity", u64, []),
            ("wcio_log_copy", u64, [ctypes.POINTER(WcioTxn), u64, u64]),
        ):
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args

    def avx512_ok(self) -> bool:
        return bool(self.lib.wcio_avx512_ok())

    def force_no_avx512(self, on: bool) -> None:
        self.lib.wcio_force_no_avx512(int(on))

    def store_lines(self, dst: int, src: int, nlines: int, fence_each: bool) -> None:
        rc = self.lib.wcio_wc_store_lines(dst, src, nlines, int(fence_each))
        if rc:
            raise WcioError(f"wcio_wc_store_lines: {os.strerror(-rc)} (nothing written)")

    def store_lines_2x32(self, dst: int, src: int, nlines: int, fence_each: bool) -> None:
        rc = self.lib.wcio_wc_store_lines_2x32(dst, src, nlines, int(fence_each))
        if rc:
            raise WcioError(f"wcio_wc_store_lines_2x32: {os.strerror(-rc)} (nothing written)")

    def uc_store64(self, dst: int, src: int, nwords: int) -> None:
        if self.lib.wcio_uc_store64(dst, src, nwords):
            raise WcioError(f"wcio_uc_store64(0x{dst:x}, {nwords}): misaligned or empty (nothing written)")

    def uc_write32(self, addr: int, value: int) -> None:
        if self.lib.wcio_uc_write32(addr, value & 0xFFFF_FFFF):
            raise WcioError(f"misaligned UC write at 0x{addr:x}")

    def uc_read32(self, addr: int) -> int:
        out = u32()
        if self.lib.wcio_uc_read32(addr, ctypes.byref(out)):
            raise WcioError(f"misaligned UC read at 0x{addr:x}")
        return out.value

    def uc_read_words(self, addr: int, n: int) -> list:
        """n (1..16) 4 B UC loads inside one 64 B line, one ctypes call."""
        out = (u32 * 16)()
        if self.lib.wcio_uc_read_words(addr, n, out):
            raise WcioError(f"uc_read_words(0x{addr:x}, {n}): misaligned or crosses a 64 B line")
        return list(out[:n])

    def uc_read64_hlh(self, addr: int) -> int:
        out = u64()
        rc = self.lib.wcio_uc_read64_hlh(addr, ctypes.byref(out))
        if rc:
            raise WcioError(f"wcio_uc_read64_hlh: {os.strerror(-rc)}")
        return out.value

    def log_enable(self, capacity: int) -> None:
        if self.lib.wcio_log_enable(capacity):
            raise WcioError("log_enable: out of memory")

    def log_total(self) -> int:
        return self.lib.wcio_log_total()

    def log(self, first: int = 0) -> list:
        """Held records with seq >= first, oldest first."""
        n = max(0, self.lib.wcio_log_total() - first)
        n = min(n, self.lib.wcio_log_capacity())
        buf = (WcioTxn * max(n, 1))()
        k = self.lib.wcio_log_copy(buf, n, first)
        return [buf[i] for i in range(k)]


def _buffer_addr(data) -> tuple:
    """(address, keepalive) of a contiguous byte buffer; read-only inputs are copied once."""
    try:
        c = (ctypes.c_char * len(memoryview(data).cast("B"))).from_buffer(data)
        return ctypes.addressof(c), c
    except (TypeError, ValueError):
        c = ctypes.create_string_buffer(bytes(data), len(bytes(data)))
        return ctypes.addressof(c), c


# --- DEVX hold check ---------------------------------------------------------------------------


def _ancestor_cmdlines(pid: int, depth: int = 32) -> list:
    out = []
    for _ in range(depth):
        if pid <= 1:
            break
        try:
            out.append(Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError):
            break
    return out


def assert_devx_hold() -> None:
    """Refuses unless this process runs under DEVX: WITH_DEVICE_RESET_AFTER=1 (PLAN §0 'DEVX') inherited through
    an ancestor _device_guard.sh (bin/_device_guard.sh:122-130 resets the chip after the hold, D13)."""
    if os.environ.get("WITH_DEVICE_RESET_AFTER") != "1":
        raise DeviceAccessRefused("WITH_DEVICE_RESET_AFTER=1 not set: X280 holds run only under $DEVX (D13)")
    if not any("_device_guard.sh" in c for c in _ancestor_cmdlines(os.getppid())):
        raise DeviceAccessRefused("no _device_guard.sh ancestor: not inside a device hold (D13)")


# --- backends ----------------------------------------------------------------------------------


class RealBackend:
    """/dev/tenstorrent/0 through os.open, fcntl.ioctl and libc mmap."""

    def __init__(self, path: str = DEVICE_PATH):
        self.path = path

    def open(self, power_aware: bool) -> int:
        assert_devx_hold()
        # Legacy (no O_APPEND) contributes all power flags but MAX_AI_CLK and sends a power message; O_APPEND
        # contributes none (KMD chardev.c:954, 986-988, 1017-1021, 1074-1083).
        return os.open(self.path, os.O_RDWR | os.O_CLOEXEC | (os.O_APPEND if power_aware else 0))

    def ioctl(self, fd: int, request: int, st: ctypes.Structure) -> None:
        fcntl.ioctl(fd, request, st, True)

    def mmap(self, fd: int, length: int, prot: int, offset: int) -> int:
        return libc_mmap(fd, length, prot, offset)

    def munmap(self, addr: int, length: int) -> None:
        libc_munmap(addr, length)

    def close(self, fd: int) -> None:
        os.close(fd)


# S1c E1a (runs/sessions/20261025_053543_2185222_s1c/s1c_e1a_check.json ecc_before): method A's 30,720 lines reached
# the CCACHE as 61,440 accesses of 32 B and counted 57,681 fix and 57,447 fail events.
S1C_E1A_ACCESSES, S1C_E1A_FIX, S1C_E1A_FAIL = 61_440, 57_681, 57_447


class FakeEcc:
    """FakeDevice's model of the L2CPU CCACHE's LIM data ECC (S1d's and S1c's checks; tests and x280d --sim-ecc only).

    Why it exists: S1c saw host writes of never-written LIM count ECC events, and later 4 B reads of the right bytes
    report uncorrectable errors; jobs/s1d.sh's verdict logic must be testable both under that hypothesis and under its
    negation. State per 8 B granule (the size S1c's counts suggest): FRESH (never written: power-up bits), OK (valid
    ECC), POISON (written after an uncorrectable pre-read: reads report uncorrectable, data intact). Each host access
    counts at most one fix and one fail event, and LOW/HIGH take the address of the last access that counted one
    (S1c: dat_fail_low = the last partial-write address, or L38 after the cleanup check).

    Modes: 'poison' (the S1c hypothesis): every host write is a read-modify-write of its footprint; an uncorrectable or
    poisoned pre-read poisons the whole footprint, a correctable or clean one leaves it OK. 'clean' (its negation): a
    write covering whole granules sets them OK without a pre-read; only a sub-granule write is a RMW. 'granule': as
    'poison', except that a write of exactly one aligned granule (an 8 B UC store) needs no pre-read, i.e. only kmd's
    'uc8' store initialises. A FRESH granule reads as correctable / uncorrectable with p_fix / p_fail (independent
    granules; the defaults reproduce S1c E1a's fix and fail events per 32 B access). How a store reaches the CCACHE is
    `shapes` (S1c: method A's 64 B NT store = two 32 B accesses, at +0 and +0x20; UMD's 2 x 32 B ymm stores = one 64 B
    access). The sim hart's own LIM accesses are not modelled: use it without a hart."""

    GRANULE = 8
    FRESH, OK, POISON = 0, 1, 2
    MODES = ("poison", "clean", "granule")
    # kmd op -> (bytes between consecutive stores, write?, ((offset, size) of each CCACHE access of one store))
    SHAPES = {"wc_nt64": (64, True, ((0, 32), (32, 32))), "wc_nt64_fenced": (64, True, ((0, 32), (32, 32))),
              "wc_2x32": (64, True, ((0, 64),)), "uc_w64": (8, True, ((0, 8),)), "uc_w32": (4, True, ((0, 4),)),
              "uc_r32": (4, False, ((0, 4),))}
    # What the address registers held before any event on silicon (S1c s1c_e0_keeper.json): garbage until a count moves.
    RESET_ADDRS = {"dat_fix_low": 0x7646_8FE6, "dat_fix_high": 16437, "dat_fail_low": 0xA7D6_F189,
                   "dat_fail_high": 1286, "dir_fix_low": 39424, "dir_fail_low": 25600}

    def __init__(self, mode: str, seed: int = 0, p_fix: Optional[float] = None, p_fail: Optional[float] = None,
                 shapes: Optional[dict] = None):
        if mode not in self.MODES:
            raise ValueError(f"FakeEcc mode {mode!r} (one of {self.MODES})")
        import numpy as np  # FakeDevice only: the device path never imports numpy here
        self.np, self.mode, self.seed = np, mode, seed
        k = 32 // self.GRANULE  # granules per 32 B access in S1c E1a
        self.p_fail = p_fail if p_fail is not None else 1 - (1 - S1C_E1A_FAIL / S1C_E1A_ACCESSES) ** (1 / k)
        self.p_fix = p_fix if p_fix is not None else 1 - (1 - S1C_E1A_FIX / S1C_E1A_ACCESSES) ** (1 / k)
        assert 0 <= self.p_fix and 0 <= self.p_fail and self.p_fix + self.p_fail <= 1
        self.shapes = {**self.SHAPES, **(shapes or {})}
        self.tiles: dict = {}  # tile -> {state, fresh, regs}
        self.accesses = {"read": 0, "write": 0}

    def _tile(self, t: int) -> dict:
        if t not in self.tiles:
            np = self.np
            n = regs.LIM_SIZE // self.GRANULE
            u = np.random.default_rng(self.seed * 16 + t).random(n)
            fresh = np.where(u < self.p_fail, 2, np.where(u < self.p_fail + self.p_fix, 1, 0)).astype(np.uint8)
            regs_ = {k: 0 for k in regs.CCACHE0_ECC}
            regs_.update(self.RESET_ADDRS)
            self.tiles[t] = {"state": np.zeros(n, np.uint8), "fresh": fresh, "regs": regs_}
        return self.tiles[t]

    def publish(self, dev: "FakeDevice", t: int) -> None:
        """The model's counter registers into the fake CCACHE0 page (kmd reads them through its normal window)."""
        for k, v in self._tile(t)["regs"].items():
            dev.poke32(regs.L2CPU_XY[t], regs.CCACHE0_ECC[k], v & 0xFFFF_FFFF)

    def _count(self, r: dict, kind: str, n: int, last_addr: int) -> None:
        if n:
            r[f"dat_{kind}_count"] = (r[f"dat_{kind}_count"] + n) & 0xFFFF_FFFF
            r[f"dat_{kind}_low"], r[f"dat_{kind}_high"] = last_addr & 0xFFFF_FFFF, last_addr >> 32

    def access(self, dev: "FakeDevice", t: int, op: str, addr: int, count: int) -> None:
        """`count` consecutive stores (or 4 B loads) of kind `op` from NoC `addr`, in address order; their footprints
        are disjoint, so the batch is evaluated at once."""
        np = self.np
        stride, write, shape = self.shapes[op]
        sizes = {n for _, n in shape}
        assert len(sizes) == 1, op
        size = sizes.pop()
        tt = self._tile(t)
        st, fresh, r = tt["state"], tt["fresh"], tt["regs"]
        acc = (addr + stride * np.arange(count, dtype=np.int64)[:, None]
               + np.array([o for o, _ in shape], dtype=np.int64)[None, :]).reshape(-1)  # issue order
        if acc.min() < regs.LIM_BASE or acc.max() + size > regs.LIM_END:
            raise ValueError(f"FakeEcc: {op} x{count} at 0x{addr:x} leaves LIM")
        off = acc - regs.LIM_BASE
        g0, g1 = off // self.GRANULE, (off + size - 1) // self.GRANULE  # the granules each access touches
        span = g1 - g0
        g = g0[:, None] + np.arange(int(span.max()) + 1)[None, :]
        inside = np.arange(g.shape[1])[None, :] <= span[:, None]
        g = np.where(inside, g, g0[:, None])  # padding columns repeat the first granule, masked out below
        cls = np.where(st[g] == self.POISON, 2, np.where(st[g] == self.FRESH, fresh[g], 0)) * inside
        if write:
            full = (off % self.GRANULE == 0) & (size % self.GRANULE == 0)
            skip = (full if self.mode == "clean" else full & (size == self.GRANULE) if self.mode == "granule"
                    else np.zeros(len(acc), bool))
            cls[skip] = 0
        fix_ev, fail_ev = (cls == 1).any(1), (cls == 2).any(1)
        if write:
            new = np.broadcast_to(np.where(fail_ev, self.POISON, self.OK).astype(np.uint8)[:, None], g.shape)
            st[g[inside]] = new[inside]
        self.accesses["write" if write else "read"] += len(acc)
        for kind, ev in (("fix", fix_ev), ("fail", fail_ev)):
            idx = np.flatnonzero(ev)
            self._count(r, kind, len(idx), int(acc[idx[-1]]) if len(idx) else 0)
        self.publish(dev, t)

    def counts(self, t: int) -> dict:
        """{fresh, ok, poison} granule counts of tile t (tests)."""
        st = self._tile(t)["state"]
        return {"fresh": int((st == self.FRESH).sum()), "ok": int((st == self.OK).sum()),
                "poison": int((st == self.POISON).sum())}


class FakeDevice:
    """A Blackhole for tests and --dry-run: one memfd per 2 MiB NoC region, KMD ioctl and mmap semantics
    (memory.c:943-1058, 1582-1596; chardev.c:499-541, 1030-1039, 1101-1115), every call recorded. With an ECC model
    (default_chip(ecc=...), FakeEcc) every host LIM access kmd makes also updates the CCACHE ECC counters."""

    MMAP_OFFSET_TLB_UC = 6 << 36  # KMD memory.c:264
    MMAP_OFFSET_TLB_WC = 7 << 36  # KMD memory.c:265

    def __init__(self, time0: float = 1000.0):
        self.regions = {}  # (xy, base) -> (memfd, mmap view)
        self.lazy = {}  # (xy, base) -> [(addr, u32)]: written when that region is first created (no memfd before)
        self.ioctls = []  # (fd, name, decoded fields, errno or 0)
        self.calls = []  # ('open'|'close'|'mmap'|'munmap'|'release'|'cleanup', ...)
        self.fds = {}
        self.tlb_owner = {}
        self.tlb_cfg = {}
        self.configure_count = {}
        self.live = {}  # addr -> (fd, tlb_id, length)
        self._next_fd = 1000
        self.now = time0
        self.ecc: Optional[FakeEcc] = None  # see default_chip(ecc=...)

    # clock for Kmd(clock=...) so tests never sleep
    def clock(self) -> float:
        return self.now

    def advance(self, s: float) -> None:
        self.now += s

    # memory model
    def _region(self, xy: tuple, base: int):
        xy = self.canon(xy)
        key = (xy, base)
        if key not in self.regions:
            fd = memfd(f"fakebh-{xy[0]}-{xy[1]}-{base:x}")
            os.ftruncate(fd, regs.TLB_2M)
            self.regions[key] = (fd, _mmap.mmap(fd, regs.TLB_2M))
            for addr, value in self.lazy.pop(key, ()):  # reset values of a region no one has touched yet
                self.regions[key][1][addr - base:addr - base + 4] = (value & 0xFFFF_FFFF).to_bytes(4, "little")
        return self.regions[key]

    @staticmethod
    def canon(xy: tuple) -> tuple:
        t = regs.tile_of(xy)
        return regs.L2CPU_XY[t] if t is not None else tuple(xy)

    def _view(self, xy: tuple, addr: int):
        base = addr & ~(regs.TLB_2M - 1)
        return self._region(xy, base)[1], addr - base

    def peek(self, xy: tuple, addr: int, n: int) -> bytes:
        v, off = self._view(xy, addr)
        return bytes(v[off:off + n])

    def poke(self, xy: tuple, addr: int, data: bytes) -> None:
        v, off = self._view(xy, addr)
        v[off:off + len(data)] = data

    def peek32(self, xy: tuple, addr: int) -> int:
        return int.from_bytes(self.peek(xy, addr, 4), "little")

    def poke32(self, xy: tuple, addr: int, value: int) -> None:
        self.poke(xy, addr, (value & 0xFFFF_FFFF).to_bytes(4, "little"))

    @classmethod
    def default_chip(cls, *, l2cpu_reset: int = regs.L2CPU_RESET_AT_SESSION_START, enabled_l2cpu: int = 0xF,
                     heartbeat: int = 123_456, noc_translation: int = 1, telemetry: Optional[dict] = None,
                     ecc: Optional[str] = None, ecc_seed: int = 0, **kw) -> "FakeDevice":
        """CMFW-after-boot state as S1c read it on silicon (regs.PLL4_CNTL_1_DEFAULT): L2CPU_RESET 0x0F, PLL4
        CNTL_0 0x12 / CNTL_1 0x0040_0102 / CNTL_5 0x0101_0101 / USE_POSTDIV 0xF, PLL_LOCK 0x1F, ENABLED_L2CPU 0xF,
        telemetry table (NoC translation on, as tt-metal needs on Blackhole), WAYENABLE 0, LIM unwritten. The
        chip-panel tags (AICLK, temperature, power ...) stay 0 unless a unit test passes `telemetry` ({tag: raw u32
        in the CMFW encoding}): this module holds no invented chip values, and x280d's sim never publishes telemetry
        (verify:all D4). `ecc` ('poison' | 'clean' | 'granule', FakeEcc): model the CCACHE's LIM data ECC (S1d tests);
        None leaves the counters at 0 as before."""
        d = cls(**kw)
        arc = regs.ARC_XY
        d.poke32(arc, regs.L2CPU_RESET, l2cpu_reset)
        d.poke32(arc, regs.PLL4_CNTL_0, 0x12)  # pd = 1, bypass = 1 (not bypassed): FW pll.c:34-40, 319, 330
        d.poke32(arc, regs.PLL4_CNTL_1, regs.PLL4_CNTL_1_DEFAULT)  # postdiv 1, as S1c read it (not pll.c:226's 0)
        d.poke32(arc, regs.PLL4_CNTL_5, regs.PLL4_CNTL_5_DEFAULT)
        d.poke32(arc, regs.PLL4_USE_POSTDIV, 0xF)
        d.poke32(arc, regs.PLL_LOCK, 0x1F)
        table = regs.ARC_CSM_BASE + 0x8000
        n_tags = 78  # FW telemetry.h:401 (TAG_COUNT)
        data = table + 8 + 4 * n_tags  # telemetry[] follows tag_table[] (FW telemetry.c:79-94)
        d.poke32(arc, regs.TELEMETRY_TABLE_PTR, table)
        d.poke32(arc, regs.TELEMETRY_DATA_PTR, data)
        d.poke32(arc, table, 0x0000_0100)  # TELEMETRY_VERSION (FW telemetry.h:26)
        d.poke32(arc, table + 4, n_tags)
        for i in range(n_tags):  # tag i+1 at offset i+1 (TELEM_OFFSET(tag) = tag, FW telemetry.h:404)
            d.poke32(arc, table + 8 + 4 * i, (i + 1) | ((i + 1) << 16))
        d.poke32(arc, data + 4 * regs.TAG_TIMER_HEARTBEAT, heartbeat)
        d.poke32(arc, data + 4 * regs.TAG_ENABLED_L2CPU, enabled_l2cpu)
        d.poke32(arc, data + 4 * regs.TAG_NOC_TRANSLATION, noc_translation)
        for tag, v in (telemetry or {}).items():
            d.poke32(arc, data + 4 * tag, v)
        d.update_l2cpuclk_telemetry()
        for t, xy in regs.L2CPU_XY.items():
            d.poke32(xy, regs.CCACHE0_CONFIG, 0x0609_1004)  # banks 4, ways 16, lg sets 9, lg block 6 (fake)
            d.poke32(xy, regs.CCACHE0_WAYENABLE, regs.WAYENABLE_EXPECTED)
            # NIU #0 (S5a): the documented fields only (ISA NoC/MemoryMap.md:198-219, 247-257): raw NoC #0 x/y, the
            # 17 x 12 grid, X-first routing; endpoint type L2CPU with tile index = the L2CPU index (assumed: the doc
            # says only "0 to 3"); NIU_CFG_0 with just the translation bit, as TAG_NOC_TRANSLATION. The 256 TLB window
            # slots stay 0 (their reset values are what S5a measures). Lazily: the region gets a memfd only once
            # something maps or pokes it, so a chip nobody probes holds no more fds than before.
            d.lazy[(xy, (regs.NIU0_PHYS + regs.XP_HIGH_OFFSET) & ~(regs.TLB_2M - 1))] = [
                (regs.NIU0_NOC_NODE_ID, xy[0] | xy[1] << 6 | regs.NOC_GRID[0] << 12 | regs.NOC_GRID[1] << 19 | 1 << 28),
                (regs.NIU0_NOC_ENDPOINT_ID, t | regs.NOC_ENDPOINT_TYPE_L2CPU << 8),
                (regs.NIU0_NIU_CFG_0, regs.NIU_CFG_0_TRANSLATION if noc_translation == 1 else 0)]
        if ecc:
            d.ecc = FakeEcc(ecc, seed=ecc_seed)
            d.ecc_access = d._ecc_access  # the hook kmd looks for (absent on RealBackend and on a fake without a model)
            for t in regs.L2CPU_XY:
                d.ecc.publish(d, t)
        return d

    def _ecc_access(self, xy: tuple, op: str, addr: int, count: int) -> None:
        t = regs.tile_of(xy)
        if self.ecc is not None and t is not None and regs.LIM_BASE <= addr < regs.LIM_END:
            self.ecc.access(self, t, op, addr, count)

    def set_heartbeat(self, value: int) -> None:
        data = self.peek32(regs.ARC_XY, regs.TELEMETRY_DATA_PTR)
        self.poke32(regs.ARC_XY, data + 4 * regs.TAG_TIMER_HEARTBEAT, value)

    def update_l2cpuclk_telemetry(self) -> None:
        """L2CPUCLK0..3 from the PLL4 registers, as CMFW's update does (FW telemetry.c:504-522 via
        clock_control_get_rate; regs.pll4_mhz: USE_POSTDIV and pll.c's postdiv special cases included)."""
        arc = regs.ARC_XY
        data = self.peek32(arc, regs.TELEMETRY_DATA_PTR)
        _, outs = regs.pll4_mhz(self.peek32(arc, regs.PLL4_CNTL_1), self.peek32(arc, regs.PLL4_CNTL_5),
                                self.peek32(arc, regs.PLL4_USE_POSTDIV))
        for tag, mhz in zip(regs.TAG_L2CPUCLK, outs, strict=True):
            self.poke32(arc, data + 4 * tag, int(mhz) & 0xFFFF)

    # backend interface
    def open(self, power_aware: bool) -> int:
        fd = self._next_fd
        self._next_fd += 1
        self.fds[fd] = {"power_aware": power_aware, "tlbs": set(), "cleanup": None, "open": True, "released": False}
        self.calls.append(("open", fd, power_aware))
        return fd

    def _err(self, fd: int, name: str, fields: dict, e: int):
        self.ioctls.append((fd, name, fields, e))
        raise OSError(e, f"fake {name}: {os.strerror(e)}")

    def ioctl(self, fd: int, request: int, st: ctypes.Structure) -> None:
        name = IOCTL_NAMES.get(request, f"0x{request:x}")
        f = self.fds.get(fd)
        if f is None or not f["open"]:
            self._err(fd, name, {}, errno.EBADF)
        if request not in IOCTL_TYPES:
            self._err(fd, name, {}, errno.ENOTTY)
        raw = IOCTL_TYPES[request].from_buffer(st)
        if request == IOCTL_ALLOCATE_TLB:
            fields = {"size": raw.in_.size}
            if raw.in_.size != regs.TLB_2M:  # fake supports 2 MiB only; tlb.c:20-33 matches the size to a kind
                self._err(fd, name, fields, errno.EINVAL)
            free = [i for i in range(regs.TLB_2M_ALLOCATABLE) if i not in self.tlb_owner]
            if not free:
                self._err(fd, name, fields, errno.ENOMEM)
            i = free[0]
            self.tlb_owner[i] = fd
            self.tlb_cfg[i] = None
            f["tlbs"].add(i)
            raw.out.id = i
            raw.out.mmap_offset_uc = self.MMAP_OFFSET_TLB_UC + i * regs.TLB_2M  # memory.c:976-984; blackhole.c:836-838
            raw.out.mmap_offset_wc = self.MMAP_OFFSET_TLB_WC + i * regs.TLB_2M
            fields["id"] = i
        elif request == IOCTL_FREE_TLB:
            i = raw.in_.id
            fields = {"id": i}
            if i not in f["tlbs"]:
                self._err(fd, name, fields, errno.EPERM)  # memory.c:1012-1015
            if any(v[1] == i for v in self.live.values()):
                self._err(fd, name, fields, errno.EBUSY)  # memory.c:1017-1024
            f["tlbs"].discard(i)
            del self.tlb_owner[i]
            del self.tlb_cfg[i]
        elif request == IOCTL_CONFIGURE_TLB:
            i, c = raw.in_.id, raw.in_.config
            fields = {"id": i, "addr": c.addr, "x_end": c.x_end, "y_end": c.y_end, "x_start": c.x_start,
                      "y_start": c.y_start, "noc": c.noc, "mcast": c.mcast, "ordering": c.ordering,
                      "linked": c.linked, "static_vc": c.static_vc}
            if i >= 256:
                self._err(fd, name, fields, errno.EINVAL)  # memory.c:1045-1046
            if i not in f["tlbs"]:
                self._err(fd, name, fields, errno.EPERM)  # memory.c:1052-1055
            if c.addr & (regs.TLB_2M - 1):
                self._err(fd, name, fields, errno.EINVAL)  # blackhole.c:181-182
            self.tlb_cfg[i] = ((c.x_end, c.y_end), c.addr)
            self.configure_count[i] = self.configure_count.get(i, 0) + 1
        else:  # SET_NOC_CLEANUP, validated exactly as chardev.c:499-541
            fields = {k: getattr(raw, k) for k, _ in SetNocCleanup._fields_}
            if (raw.argsz != ctypes.sizeof(SetNocCleanup) or raw.flags != 0 or raw.enabled > 1 or raw.addr & 3
                    or raw.noc > 1 or raw.x > 64 or raw.y > 64):
                self._err(fd, name, fields, errno.EINVAL)
            f["cleanup"] = dict(fields)
        self.ioctls.append((fd, name, fields, 0))

    def mmap(self, fd: int, length: int, prot: int, offset: int) -> int:
        f = self.fds.get(fd)
        kind, i = None, None
        for base, k in ((self.MMAP_OFFSET_TLB_UC, "uc"), (self.MMAP_OFFSET_TLB_WC, "wc")):
            if base <= offset < base + 256 * regs.TLB_2M and (offset - base) % regs.TLB_2M == 0:
                kind, i = k, (offset - base) // regs.TLB_2M  # offset must be a window start (memory.c:1582-1593)
        if f is None or kind is None or length > regs.TLB_2M or length <= 0:
            raise OSError(errno.EINVAL, "fake mmap: bad offset or length")
        if i not in f["tlbs"]:
            raise OSError(errno.EPERM, "fake mmap: window not owned by this fd")
        if self.tlb_cfg.get(i) is None:
            raise OSError(errno.EINVAL, "fake mmap: window never configured (a real KMD would map it anyway)")
        xy, base = self.tlb_cfg[i]
        memfd = self._region(xy, base)[0]
        addr = libc_mmap(memfd, length, prot, 0)
        self.live[addr] = (fd, i, length)
        self.calls.append(("mmap", fd, i, kind, length, prot))
        return addr

    def munmap(self, addr: int, length: int) -> None:
        fd, i, n = self.live.pop(addr)
        assert n == length, "fake munmap: length mismatch"
        libc_munmap(addr, length)
        self.calls.append(("munmap", fd, i, length))
        if not self.fds[fd]["open"]:
            self._maybe_release(fd)

    def close(self, fd: int) -> None:
        self.fds[fd]["open"] = False
        self.calls.append(("close", fd))
        self._maybe_release(fd)

    def _maybe_release(self, fd: int) -> None:
        """tt_cdev_release runs once the last reference (fd or mapping) is gone (chardev.c:1092-1115)."""
        f = self.fds[fd]
        if f["released"] or any(v[0] == fd for v in self.live.values()):
            return
        f["released"] = True
        c = f["cleanup"]
        if c and c["enabled"]:  # tt_cdev_release_noc_cleanup: noc_write32(x, y, addr, data & 0xFFFFFFFF)
            self.poke32((c["x"], c["y"]), c["addr"], c["data"] & 0xFFFF_FFFF)
            self._ecc_access((c["x"], c["y"]), "uc_w32", c["addr"], 1)  # the KMD's own 4 B write (no-op without a model)
            self.calls.append(("cleanup", fd, (c["x"], c["y"]), c["addr"], c["data"] & 0xFFFF_FFFF))
        for i in sorted(f["tlbs"]):
            del self.tlb_owner[i]
            del self.tlb_cfg[i]
        f["tlbs"].clear()
        self.calls.append(("release", fd))

    def names(self) -> list:
        return [n for _, n, _, _ in self.ioctls]


# --- op log ------------------------------------------------------------------------------------


class OpLog(deque):
    """The op log: a bounded deque, so a full log drops its oldest record per append (O(1)). It used to be a list
    halved in one `del` when full, which freed 65,536 records at once: a 2-7 ms stall every ~1,070 W-WBC calls on
    the device path (S3 2026-10-25, x280d_us outliers every 65,536 ops). Slices return lists, as before."""

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            return list(itertools.islice(self, start, stop, step)) if step > 0 else list(self)[i]
        return super().__getitem__(i)


@dataclass
class Op:
    kind: str  # OPEN CLOSE IOCTL MMAP MUNMAP R32 RWORDS R64HLH W32 W64 WC64 ASSUME_INIT CHECK REFUSED NOTE
    xy: Optional[tuple] = None
    addr: Optional[int] = None
    size: int = 0
    value: Optional[int] = None
    detail: str = ""

    def __str__(self) -> str:
        s = f"{self.kind:<11}"
        if self.xy is not None:
            s += f" ({self.xy[0]},{self.xy[1]})"
        if self.addr is not None:
            s += f" 0x{self.addr:016x}"
        if self.size:
            s += f" {self.size}B"
        if self.value is not None:
            s += f" = 0x{self.value:08x}"
        return s + (f"  {self.detail}" if self.detail else "")


@dataclass
class Window:
    spec: regs.WindowSpec
    tlb_id: int
    off_uc: int
    off_wc: int
    uc: int = 0  # persistent UC mapping (0: none)
    wc: int = 0  # persistent WC mapping (LIM only)
    uc_prot: int = 0


@dataclass
class _VaRange:
    lo: int
    hi: int
    xy: tuple
    noc_base: int
    kind: str
    name: str
    seq_lo: int
    seq_hi: Optional[int] = None


class LimTracker:
    """Lines written by a full 64 B store in this session, or vouched for by assume() (PLAN R26). 'Initialised' here
    means written whole by an init store: the precondition of kmd's partial-write rule. It does not mean valid ECC:
    S1c (2026-10-25) read uncorrectable errors back from lines method A had written; S1d (jobs/s1d.sh) measures which
    store, if any, gives valid ECC."""

    def __init__(self):
        self.bits = bytearray(regs.LIM_LINES)

    @staticmethod
    def _idx(addr: int) -> int:
        return (addr - regs.LIM_BASE) // regs.LINE

    def mark(self, addr: int, nlines: int) -> None:
        i = self._idx(addr)
        self.bits[i:i + nlines] = b"\x01" * nlines

    def is_init(self, addr: int, size: int = 4) -> bool:
        lo, hi = self._idx(addr), self._idx(addr + size - 1)
        return all(self.bits[lo:hi + 1])

    def count(self) -> int:
        return sum(self.bits)


@dataclass
class Batch:
    tile: int
    checked_at: float
    pll: dict = field(default_factory=dict)


# --- the access layer --------------------------------------------------------------------------


class Kmd:
    """One x280d session: keeper fd, fixed windows, checked accessors and the boot primitives of PLAN §5.2."""

    def __init__(self, backend, tiles: tuple = (regs.DEFAULT_TILE,), *, cross_read: bool = False,
                 wcio: Optional[Wcio] = None, clock: Optional[Callable[[], float]] = None,
                 log_capacity: int = 1 << 17, oplog_cap: int = 1 << 17,
                 step_sleep: Callable[[], None] = lambda: None, probe_tiles: tuple = ()):
        """probe_tiles (S5a): L2CPU tiles this session may only read, through regs.window_specs' probe windows; a probe
        tile outside `tiles` is never written and never mapped writable, whatever regs.WRITABLE_TILES says."""
        for t in tiles:
            if t not in regs.WRITABLE_TILES:
                raise PolicyError(f"tile {t} is not in regs.WRITABLE_TILES {regs.WRITABLE_TILES}")
        for t in probe_tiles:
            if t not in regs.L2CPU_XY:
                raise PolicyError(f"probe tile {t} is not an L2CPU tile {tuple(regs.L2CPU_XY)}")
        self.backend = backend
        self.tiles = tuple(tiles)
        self.probe_tiles = tuple(dict.fromkeys(probe_tiles))
        self.cross_read = cross_read
        self.clock = clock or (backend.clock if isinstance(backend, FakeDevice) else time.monotonic)
        self.step_sleep = step_sleep
        self.wcio = wcio or Wcio()
        self.wcio.log_enable(log_capacity)
        self.oplog = OpLog(maxlen=oplog_cap)  # serve runs for hours at ~60 ops per W-WBC call: keep the newest
        self.oplog_cap, self.oplog_dropped = oplog_cap, 0
        self.windows: dict = {}
        self.fd: Optional[int] = None
        self.t_open: Optional[float] = None
        self.lim = {t: LimTracker() for t in self.tiles}
        self.released = {t: None for t in self.tiles}  # None unknown (strict), False/True from L2CPU_RESET
        self.reset_rmw_done = False
        self._rmw_armed = False
        self.pll_mhz: Optional[int] = None
        self.pll_excursions = 0
        self.pll_interlock: Optional[str] = None  # why a PLL4 walk stopped (review F5)
        self._pll_expect: list = []
        self._pll_i = 0
        self._batches: dict = {}
        self._enabled_l2cpu: Optional[int] = None
        self._va: list = []
        self.noc_translation: Optional[int] = None  # CMFW TAG_NOC_TRANSLATION, read at open() when cross_read
        # Boot artefacts written AND read back in this session (release() refuses without them, review F2):
        # gp = GP-scratch bytes, vectors = (reset pc, RNMI handler, RNMI exception handler), image = (crc32, len).
        self._boot = {t: {"gp": None, "vectors": None, "image": None, "mh_vectors": False} for t in self.tiles}
        # --harts 4: True only inside write_boot_vectors(harts=4), so _policy accepts the secondary vector values there
        self._mh_arm = {t: False for t in self.tiles}
        # E1F only: None, then 'zeroing' (released onto stage 0: no host LIM write), 'zeroed' (done word seen:
        # image may be written), 'entered' (entry word written: crt0 runs), 'stub' (park stub swapped in).
        self._stage0 = {t: None for t in self.tiles}
        self._tele_cache: Optional[tuple] = None  # (table ptr, data ptr, {tag: CSM address}), as KMD caches it
        self._ecc_hook = getattr(backend, "ecc_access", None)  # FakeDevice's CCACHE ECC model only (FakeEcc)

    @classmethod
    def dry_run(cls, tiles: tuple = (regs.DEFAULT_TILE,), **kw) -> "Kmd":
        """A session against FakeDevice.default_chip(): every op is recorded, nothing touches hardware."""
        return cls(FakeDevice.default_chip(), tiles, **kw)

    # -- logging --
    def _op(self, kind: str, xy=None, addr=None, size=0, value=None, detail="") -> Op:
        op = Op(kind, None if xy is None else tuple(xy), addr, size, value, detail)
        if len(self.oplog) == self.oplog_cap:
            self.oplog_dropped += 1  # the deque drops its oldest record on this append
        self.oplog.append(op)
        return op

    def _refuse(self, exc: type, msg: str, xy=None, addr=None, size=0, value=None):
        self._op("REFUSED", xy, addr, size, value, msg)
        raise exc(msg)

    def _ecc(self, xy: tuple, op: str, addr: int, count: int) -> None:
        """Tell FakeDevice's ECC model about host LIM accesses just made (FakeEcc.SHAPES ops); no-op otherwise."""
        if self._ecc_hook is not None and regs.LIM_BASE <= addr < regs.LIM_END and regs.tile_of(xy) is not None:
            self._ecc_hook(tuple(xy), op, addr, count)

    def oplog_text(self) -> str:
        return "\n".join(str(o) for o in self.oplog)

    def txn_log(self) -> list:
        """libwcio transactions with the NoC target of each address: dicts op, seq, xy, noc, kind, size, align64."""
        out = []
        for t in self.wcio.log():
            d = {"seq": t.seq, "op": WCIO_OPS.get(t.op, str(t.op)), "va": t.addr, "size": t.size, "value": t.value,
                 "align64": bool(t.align64), "align4": bool(t.align4), "xy": None, "noc": None, "kind": None}
            for r in reversed(self._va):
                if r.lo <= t.addr < r.hi and r.seq_lo <= t.seq and (r.seq_hi is None or t.seq < r.seq_hi):
                    d.update(xy=r.xy, noc=r.noc_base + (t.addr - r.lo), kind=r.kind, window=r.name)
                    break
            out.append(d)
        return out

    # -- session --
    def _ioctl(self, request: int, st: ctypes.Structure, detail: str, fd: Optional[int] = None) -> None:
        fd = self.fd if fd is None else fd
        try:
            self.backend.ioctl(fd, request, st)
        except OSError as e:
            self._op("IOCTL", detail=f"{IOCTL_NAMES[request]} {detail} -> {e.strerror}")
            raise KmdError(f"{IOCTL_NAMES[request]} {detail}: {e}") from e
        self._op("IOCTL", detail=f"{IOCTL_NAMES[request]} {detail}")

    def open(self) -> "Kmd":
        """Keeper legacy fd, then every window allocated, configured and mapped exactly once."""
        if self.fd is not None:
            raise KmdError("already open")
        self.fd = self.backend.open(power_aware=False)
        self.t_open = self.clock()
        self._op("OPEN", detail=f"keeper fd {self.fd} legacy O_RDWR (power message; settle {KEEPER_SETTLE_S:.0f} s)")
        try:
            for spec in regs.window_specs(self.tiles, self.cross_read, self.probe_tiles):
                if regs.is_translated(spec.xy) and not self._noc_translation_on():
                    # review F6: (8, 26+i) exists only with CMFW's NoC translation on; never even allocate it
                    self._op("NOTE", spec.xy, spec.base, 0, None, f"{spec.name} not allocated: NoC translation off "
                                                                  f"(TAG_NOC_TRANSLATION={self.noc_translation})")
                    continue
                self._setup_window(spec)
        except BaseException:
            self.close()
            raise
        return self

    def _setup_window(self, spec: regs.WindowSpec) -> None:
        if (spec not in regs.window_specs(self.tiles, self.cross_read, self.probe_tiles)
                or tuple(spec.xy) not in regs.ALLOWED_TARGETS
                or spec.base % regs.TLB_2M or spec.map_len > regs.TLB_2M):
            self._refuse(PolicyError, f"window {spec.name} {spec.xy} 0x{spec.base:x} is not a regs window")
        a = AllocateTlb()
        a.in_.size = regs.TLB_2M
        self._ioctl(IOCTL_ALLOCATE_TLB, a, f"size=0x{regs.TLB_2M:x} for {spec.name}")
        w = Window(spec, a.out.id, a.out.mmap_offset_uc, a.out.mmap_offset_wc)
        self.windows[spec.name] = w
        c = ConfigureTlb()
        c.in_.id = w.tlb_id
        cfg = c.in_.config
        cfg.addr, (cfg.x_end, cfg.y_end) = spec.base, spec.xy
        cfg.noc, cfg.ordering = regs.NOC, regs.ORDERING_STRICT  # unicast, not linked (HostToDeviceTLBs.md:30-33)
        self._ioctl(IOCTL_CONFIGURE_TLB, c, f"id={w.tlb_id} {spec.name} ({spec.xy[0]},{spec.xy[1]}) "
                                            f"addr=0x{spec.base:x} noc={cfg.noc} ordering={cfg.ordering}")
        if spec.persistent in ("ro", "rw"):
            w.uc_prot = PROT_READ | (PROT_WRITE if spec.persistent == "rw" else 0)
            w.uc = self._map(w, "uc", w.uc_prot)
        if spec.wc:
            w.wc = self._map(w, "wc", PROT_READ | PROT_WRITE)

    def _map(self, w: Window, kind: str, prot: int) -> int:
        off = w.off_uc if kind == "uc" else w.off_wc
        va = self.backend.mmap(self.fd, w.spec.map_len, prot, off)
        self._va.append(_VaRange(va, va + w.spec.map_len, tuple(w.spec.xy), w.spec.base, kind, w.spec.name,
                                 self.wcio.log_total()))
        rw = "RW" if prot & PROT_WRITE else "RO"
        self._op("MMAP", w.spec.xy, w.spec.base, w.spec.map_len, None, f"{w.spec.name} id={w.tlb_id} {kind.upper()} {rw}")
        return va

    def _unmap(self, w: Window, va: int, kind: str) -> None:
        self.backend.munmap(va, w.spec.map_len)
        for r in self._va:
            if r.lo == va and r.seq_hi is None:
                r.seq_hi = self.wcio.log_total()
        self._op("MUNMAP", w.spec.xy, w.spec.base, w.spec.map_len, None, f"{w.spec.name} {kind.upper()}")

    def close(self) -> None:
        """Unmap, FREE_TLB, close the keeper fd (the KMD then performs the registered cleanup write)."""
        for w in list(self.windows.values()):
            for kind in ("uc", "wc"):
                va = getattr(w, kind)
                if va:
                    self._unmap(w, va, kind)
                    setattr(w, kind, 0)
            f = FreeTlb()
            f.in_.id = w.tlb_id
            try:
                self._ioctl(IOCTL_FREE_TLB, f, f"id={w.tlb_id} {w.spec.name}")
            except KmdError:
                pass  # released with the fd anyway (chardev.c tt_cdev_release_tlbs)
        self.windows.clear()
        if self.fd is not None:
            self.backend.close(self.fd)
            self._op("CLOSE", detail=f"keeper fd {self.fd} (KMD runs the SET_NOC_CLEANUP write, if registered)")
            self.fd = None

    def __enter__(self) -> "Kmd":
        return self.open() if self.fd is None else self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- address resolution --
    def _window(self, xy: tuple, addr: int, size: int) -> Window:
        for w in self.windows.values():
            s = w.spec
            if tuple(s.xy) == tuple(xy) and s.base <= addr and addr + size <= s.base + s.map_len:
                return w
        self._refuse(PolicyError, f"no window covers 0x{addr:x}+{size} on {tuple(xy)}", xy, addr, size)

    def txn_summary(self) -> dict:
        """What T-3.4 reads off the libwcio log: every WC store a 64 B-aligned 64 B line followed by sfence
        (per-line or per-batch), no partial LIM write before its line's full-line store."""
        tx = self.txn_log()
        wc = [i for i, t in enumerate(tx) if t["op"] == "WC64"]
        fenced, inited, partial_before_init = True, set(), 0
        init_ops = ("WC64", "WC32", "UC_W64")  # LIM init stores (UC_W64 / WC32: kmd.lim_init's S1d variants only)
        for i in wc:
            j = i + 1
            while j < len(tx) and tx[j]["op"] == "WC64":
                j += 1
            fenced &= j < len(tx) and tx[j]["op"] == "SFENCE"
        for t in tx:
            if t["op"] in init_ops and t["noc"] is not None:
                inited.add((t["xy"], t["noc"] & ~63))
            elif t["op"] == "UC_W32" and t["noc"] is not None and regs.tile_of(t["xy"]) is not None \
                    and regs.LIM_BASE <= t["noc"] < regs.LIM_END:
                partial_before_init += (t["xy"], t["noc"] & ~63) not in inited
        return {"records": len(tx), "wc64": len(wc), "wc64_all_aligned_64B": all(tx[i]["align64"] and tx[i]["size"] == 64
                                                                              for i in wc),
                "wc64_all_fenced": fenced, "uc_w32": sum(t["op"] == "UC_W32" for t in tx),
                "wc32": sum(t["op"] == "WC32" for t in tx), "uc_w64": sum(t["op"] == "UC_W64" for t in tx),
                "uc_r32": sum(t["op"] == "UC_R32" for t in tx), "refused": sum(t["op"] == "REFUSED" for t in tx),
                "lim_partial_before_init": partial_before_init,
                "log_wrapped": self.wcio.log_total() > len(tx)}

    # -- interlocks --
    def wait_settled(self) -> None:
        """Wait out KEEPER_SETTLE_S after the keeper open (FakeDevice: advance its clock instead of sleeping)."""
        if self.t_open is None:
            raise InterlockError("keeper fd not open")
        left = KEEPER_SETTLE_S - (self.clock() - self.t_open)
        if left > 0:
            if isinstance(self.backend, FakeDevice):
                self.backend.advance(left)
            else:
                time.sleep(left)
        self._op("NOTE", detail=f"settled {KEEPER_SETTLE_S:.1f} s after the keeper open")

    def _settled(self) -> None:
        if self.t_open is None:
            raise InterlockError("keeper fd not open")
        dt = self.clock() - self.t_open
        if dt < KEEPER_SETTLE_S:
            self._refuse(InterlockError, f"keeper fd opened {dt:.3f} s ago; L2CPU access needs >= {KEEPER_SETTLE_S} s")

    @contextmanager
    def l2cpu_batch(self, tile: int):
        """PLAN §5.2 invariant 4: PLL4 at its defaults (read directly, ARC first) and ENABLED_L2CPU bit set
        before a batch of L2CPU-tile accesses; accesses outside a batch are refused (R28)."""
        if tile not in self.tiles and tile not in self.probe_tiles:
            raise PolicyError(f"tile {tile} not in this session {self.tiles} (probe tiles {self.probe_tiles})")
        self._settled()
        pll = self.read_pll4()
        why = self.pll4_default_why(pll)
        if why:
            self._refuse(InterlockError, f"PLL4 not at CMFW defaults ({why}): {_hexd(pll)}", regs.ARC_XY, regs.PLL4_BASE)
        self.pll_mhz = 800
        en = self.enabled_l2cpu()
        if not en & regs.enabled_l2cpu_bit(tile):
            self._refuse(InterlockError, f"ENABLED_L2CPU=0x{en:x}: tile {tile} harvested/disabled")
        self._batches[tile] = self._batches.get(tile, 0) + 1
        try:
            yield Batch(tile, self.clock(), pll)
        finally:
            self._batches[tile] -= 1
            if not self._batches[tile]:
                del self._batches[tile]

    def _need_batch(self, xy: tuple, addr: int) -> None:
        t = regs.tile_of(xy)
        if t is not None and t not in self._batches:
            self._refuse(InterlockError, f"L2CPU{t} access outside l2cpu_batch()", xy, addr)

    # -- reads --
    def _read_checks(self, xy: tuple, addr: int, size: int, allow_uninit: bool) -> Window:
        why = regs.classify_read(xy, addr, size)
        if why:
            self._refuse(PolicyError, why, xy, addr, size)
        if regs.is_translated(xy) and self.noc_translation != 1:
            self._refuse(InterlockError, f"translated coordinate {tuple(xy)} with NoC translation off or unknown "
                                         f"(TAG_NOC_TRANSLATION={self.noc_translation}; FW noc_init.c:608-610)", xy, addr, size)
        self._need_batch(xy, addr)
        t = regs.tile_of(xy)
        if (t is not None and regs.LIM_BASE <= addr < regs.LIM_END and not allow_uninit
                and not self.lim[t].is_init(addr, size)):
            self._refuse(PolicyError, f"LIM read of uninitialised line 0x{addr & ~63:x}", xy, addr, size)
        w = self._window(xy, addr, size)
        if not w.uc:
            self._refuse(PolicyError, f"{w.spec.name} has no persistent mapping", xy, addr, size)
        return w

    def read32(self, xy: tuple, addr: int, *, allow_uninit: bool = False) -> int:
        """Checked 4 B UC read through a persistent RO/RW mapping."""
        w = self._read_checks(xy, addr, 4, allow_uninit)
        v = self.wcio.uc_read32(w.uc + addr - w.spec.base)
        self._op("R32", xy, addr, 4, v)
        self._ecc(xy, "uc_r32", addr, 1)
        return v

    def read_words(self, xy: tuple, addr: int, n: int, *, allow_uninit: bool = False) -> list:
        """n (1..16) consecutive checked 4 B UC reads inside one 64 B line, in one libwcio call: the policy is
        checked once for the whole range, and libwcio still logs every word as its own UC_R32."""
        if not 1 <= n <= 16 or addr % 4 or (addr % regs.LINE) + 4 * n > regs.LINE:
            self._refuse(PolicyError, f"read_words(0x{addr:x}, {n}) must stay inside one 64 B line", xy, addr, 4 * n)
        w = self._read_checks(xy, addr, 4 * n, allow_uninit)
        v = self.wcio.uc_read_words(w.uc + addr - w.spec.base, n)
        self._op("RWORDS", xy, addr, 4 * n, v[0], f"x{n}")
        self._ecc(xy, "uc_r32", addr, n)
        return v

    def read_lim32(self, addr: int, tile: int = regs.DEFAULT_TILE, **kw) -> int:
        return self.read32(regs.L2CPU_XY[tile], addr, **kw)

    def read_lim_words(self, addr: int, n: int, tile: int = regs.DEFAULT_TILE, **kw) -> list:
        return self.read_words(regs.L2CPU_XY[tile], addr, n, **kw)

    def read_lim_line(self, addr: int, tile: int = regs.DEFAULT_TILE, **kw) -> bytes:
        """One whole 64 B LIM line (16 UC words, one call): x280d's L1/L35/L36/L39 reads."""
        if addr % regs.LINE:
            self._refuse(PolicyError, f"line read at 0x{addr:x} is not 64 B aligned", regs.L2CPU_XY[tile], addr, 64)
        return b"".join(v.to_bytes(4, "little") for v in self.read_lim_words(addr, 16, tile, **kw))

    def read_lim64_hlh(self, addr: int, tile: int = regs.DEFAULT_TILE) -> int:
        """u64 the X280 writes (heartbeat), read hi, lo, hi (PLAN §1.4.4 L1)."""
        xy = regs.L2CPU_XY[tile]
        for a in (addr, addr + 4):
            if regs.classify_read(xy, a, 4) or not self.lim[tile].is_init(a):
                self._refuse(PolicyError, f"read64 at 0x{addr:x} not allowed", xy, addr, 8)
        self._need_batch(xy, addr)
        w = self._window(xy, addr, 8)
        v = self.wcio.uc_read64_hlh(w.uc + addr - w.spec.base)
        self._op("R64HLH", xy, addr, 8, v & 0xFFFF_FFFF, f"hi=0x{v >> 32:08x}")
        for a in (addr + 4, addr, addr + 4):  # libwcio's hi, lo, hi (a retry adds 3 more, not modelled)
            self._ecc(xy, "uc_r32", a, 1)
        return v

    def read_pll4(self) -> dict:
        a = regs.ARC_XY
        return {"cntl_0": self.read32(a, regs.PLL4_CNTL_0), "cntl_1": self.read32(a, regs.PLL4_CNTL_1),
                "cntl_5": self.read32(a, regs.PLL4_CNTL_5), "use_postdiv": self.read32(a, regs.PLL4_USE_POSTDIV)}

    @staticmethod
    def pll4_default_why(d: dict) -> str:
        """'' at the CMFW state (FW pll.c:225-236, CNTL_1 as S1c read it on silicon), else why not (regs)."""
        return regs.pll4_default_why(d["cntl_1"], d["cntl_5"], d["use_postdiv"])

    @staticmethod
    def pll4_is_default(d: dict) -> bool:
        """PLAN WP-3.4 interlock: CNTL_1 exactly 0x0040_0102 (postdiv byte included), CNTL_5, USE_POSTDIV."""
        return not Kmd.pll4_default_why(d)

    def read_l2cpu_reset(self) -> int:
        v = self.read32(regs.ARC_XY, regs.L2CPU_RESET)
        for t in self.tiles:
            self.released[t] = bool(v & regs.release_bit(t))
        return v

    def _telemetry_addrs(self, tags: tuple) -> tuple:
        """KMD's algorithm (blackhole.c:479-519): SCRATCH_RAM[13] table, SCRATCH_RAM[12] data, tag -> offset, every
        pointer checked against CSM. Returns (table ptr, data ptr, {tag: CSM address of its u32 value})."""
        a = regs.ARC_XY

        def in_csm(x, n):
            return regs.ARC_CSM_BASE <= x <= regs.ARC_CSM_BASE + regs.ARC_CSM_SIZE - n

        base, data = self.read32(a, regs.TELEMETRY_TABLE_PTR), self.read32(a, regs.TELEMETRY_DATA_PTR)
        if not (in_csm(base, 8) and in_csm(data, 4)):
            raise KmdError(f"telemetry not available (table 0x{base:x}, data 0x{data:x})")
        version = self.read32(a, base)
        if (version >> 16) & 0xFF > regs.TELEMETRY_VERSION_MAJOR_MAX:
            raise KmdError(f"unsupported telemetry version 0x{version:x}")
        n = self.read32(a, base + 4)
        if n > regs.TELEMETRY_MAX_ENTRIES or not in_csm(base + 8, 4 * n):
            raise KmdError(f"bad telemetry table: {n} entries at 0x{base + 8:x}")
        want, out = set(tags), {}
        for i in range(n):
            e = self.read32(a, base + 8 + 4 * i)
            tag, off = e & 0xFFFF, e >> 16
            if tag in want and in_csm(data + 4 * off, 4):
                out[tag] = data + 4 * off
                if len(out) == len(want):
                    break
        return base, data, out

    def read_telemetry(self, tags: tuple = (regs.TAG_TIMER_HEARTBEAT, regs.TAG_ENABLED_L2CPU)) -> dict:
        """Tag values, the table walked afresh (interlocks: no cached pointer survives a CMFW restart)."""
        _, _, addrs = self._telemetry_addrs(tags)
        return {tag: self.read32(regs.ARC_XY, ad) for tag, ad in addrs.items()}

    def _noc_translation_on(self) -> bool:
        """TAG_NOC_TRANSLATION == 1 (read once per session through the ARC/CSM windows; an absent tag counts as 0)."""
        if self.noc_translation is None:
            self.noc_translation = self.read_telemetry((regs.TAG_NOC_TRANSLATION,)).get(regs.TAG_NOC_TRANSLATION, 0)
        return self.noc_translation == 1

    def read_status_telemetry(self) -> dict:
        """The HUD chip-panel telemetry (regs.STATUS_TELEMETRY_TAGS) decoded to contract units, memory reads only.
        Tag addresses are resolved once from the table and cached, as the KMD does (KMD blackhole.c:470-525
        populate, telemetry.c:158 read by cached address); both table pointers are re-read on every call and a
        change re-resolves them. Tags the table lacks stay None (regs.decode_status_telemetry)."""
        a = regs.ARC_XY
        ptrs = (self.read32(a, regs.TELEMETRY_TABLE_PTR), self.read32(a, regs.TELEMETRY_DATA_PTR))
        if self._tele_cache is None or self._tele_cache[:2] != ptrs:
            self._tele_cache = self._telemetry_addrs(regs.STATUS_TELEMETRY_TAGS)
        raw = {tag: self.read32(a, addr) for tag, addr in self._tele_cache[2].items()}
        return regs.decode_status_telemetry(raw)

    def enabled_l2cpu(self) -> int:
        """ENABLED_L2CPU (fused, so read once per session)."""
        if self._enabled_l2cpu is None:
            self._enabled_l2cpu = self.read_telemetry((regs.TAG_ENABLED_L2CPU,)).get(regs.TAG_ENABLED_L2CPU, 0)
        return self._enabled_l2cpu

    def timer_heartbeat(self) -> int:
        return self.read_telemetry((regs.TAG_TIMER_HEARTBEAT,))[regs.TAG_TIMER_HEARTBEAT]

    def read_wayenable(self, tile: int = regs.DEFAULT_TILE) -> int:
        return self.read32(regs.L2CPU_XY[tile], regs.CCACHE0_WAYENABLE)

    def read_ccache_config(self, tile: int = regs.DEFAULT_TILE, translated: bool = False) -> int:
        xy = regs.L2CPU_XY_TRANSLATED[tile] if translated else regs.L2CPU_XY[tile]
        return self.read32(xy, regs.CCACHE0_CONFIG)

    def read_ccache_ecc(self, tile: int = regs.DEFAULT_TILE) -> dict:
        """SiFive ccache0 ECC fix/fail registers (regs.CCACHE0_ECC); COUNT reads clear the ECC interrupt only."""
        return {k: self.read32(regs.L2CPU_XY[tile], a) for k, a in regs.CCACHE0_ECC.items()}

    def read_ccache_ecc_counts(self, tile: int = regs.DEFAULT_TILE) -> dict:
        """{fix, fail}: corrected / uncorrected ECC event counts, directory + data (REF linux-v6.8-sifive_ccache.c:
        23-37 COUNT registers; regs.CCACHE0_ECC), 4 reads instead of the full 12-register dump."""
        xy = regs.L2CPU_XY[tile]
        e = regs.CCACHE0_ECC
        return {"fix": self.read32(xy, e["dir_fix_count"]) + self.read32(xy, e["dat_fix_count"]),
                "fail": self.read32(xy, e["dir_fail_count"]) + self.read32(xy, e["dat_fail_count"])}

    def read_beu(self, tile: int = regs.DEFAULT_TILE, hart: int = 0) -> dict:
        """A hart's Bus-Error Unit page, only once the tile's harts run: no source says a per-hart device answers
        while its hart is in reset, and a hung NoC read hangs the host (review F7, R28)."""
        if self.released.get(tile) is not True:
            self._refuse(InterlockError, f"BEU read with tile {tile} released={self.released.get(tile)} (harts in "
                                         "reset or unknown)", regs.L2CPU_XY[tile], regs.beu_base(hart), 4)
        return {k: self.read32(regs.L2CPU_XY[tile], regs.beu_base(hart) + off) for k, off in regs.BEU.items()}

    # -- the write path --
    def _probe_only(self, xy: tuple) -> Optional[int]:
        """The tile of `xy` if it is a read-only probe tile of this session (S5a), else None. The session's own tile
        counts too: a probe session (`serve --no-release --probe-tiles`) writes no L2CPU tile at all."""
        t = regs.tile_of(xy)
        return t if t is not None and t in self.probe_tiles else None

    def _policy(self, xy: tuple, addr: int, size: int, value: Optional[int], ctx: str) -> regs.Range:
        if self._probe_only(xy) is not None:
            self._refuse(PolicyError, f"L2CPU{regs.tile_of(xy)} is a read-only probe tile of this session (S5a)",
                         xy, addr, size, value)
        allow, denies, why = regs.classify_write(xy, addr, size)
        for d in denies:
            if not (d.exception and d.exception == ctx):
                self._refuse(PolicyError, f"DENY {d.name}: {d.why}", xy, addr, size, value)
        if allow is None:
            self._refuse(PolicyError, why, xy, addr, size, value)
        if allow.ctx != ctx and not (allow.name == "LIM" and ctx in ("", "lim_init", "image", "stage0_entry")):
            self._refuse(PolicyError, f"{allow.name} is writable only via {allow.ctx or 'the LIM accessors'}",
                         xy, addr, size, value)
        if allow.values and value is not None and value not in allow.values:
            self._refuse(PolicyError, f"value 0x{value:x} not permitted for {allow.name}", xy, addr, size, value)
        t = regs.tile_of(xy)
        if allow.name in regs.LIM_ALLOW:
            self._lim_rules(t, xy, addr, size, value, ctx)
        elif allow.name in ("PLL4_CNTL_1", "PLL4_CNTL_5"):
            if self._pll_i >= len(self._pll_expect) or self._pll_expect[self._pll_i] != (addr, value):
                exp = self._pll_expect[self._pll_i] if self._pll_i < len(self._pll_expect) else None
                self._refuse(PolicyError, f"PLL4 write out of the clock.py sequence (expected {exp})", xy, addr, size, value)
        elif allow.name == "L2CPU_RESET":
            if not self._rmw_armed:  # only release() arms it, once per session
                self._refuse(PolicyError, "L2CPU_RESET is written only by release()", xy, addr, size, value)
            self._rmw_armed = False
        elif allow.name in ("RESET_VECTORS", "RNMI_HANDLERS"):
            if self.released.get(t) is not False:
                self._refuse(PolicyError, f"{allow.name} only before release (tile {t} released={self.released.get(t)})",
                             xy, addr, size, value)
            if addr % 8:  # high word of a 47-bit address
                ok = value == 0
            elif allow.name == "RESET_VECTORS":
                ok = value in (regs.RESET_PC, regs.STAGE0_ENTRY)
            else:  # hart 0 always the park stub; harts 1-3 the --harts 4 pair only inside write_boot_vectors(harts=4)
                h, exc = divmod(addr - regs.rnmi_handler(0), 16)
                ok = value == (regs.PARK_STUB if exc == 0 else regs.RNMI_EXC_STUB) or (
                    h in regs.MH_SEC_HARTS and self._mh_arm.get(t) is True and
                    value == (regs.RESET_PC if exc == 0 else regs.MH_SEC_RNMI_EXC))
            if not ok:
                self._refuse(PolicyError, f"value 0x{value:x} not permitted at {allow.name} 0x{addr:x}", xy, addr, size, value)
        elif allow.name == "RNMI_TRIGGER":
            if value & ~regs.RNMI_TRIGGER_VALID_MASK:
                self._refuse(PolicyError, "RNMI trigger reserved bits", xy, addr, size, value)
        if t is not None:
            self._need_batch(xy, addr)
        return allow

    def _lim_rules(self, t: int, xy: tuple, addr: int, size: int, value: Optional[int], ctx: str) -> None:
        """One writer per line (PLAN §1.4.4) and R26. Once the tile is released (or its state is unknown) the
        X280-owned mailbox lines AND the monitor region (regs.MONITOR_REGION: code, .bss, .persist, stack) are the
        hart's (review F4); LIM init is a pre-release operation only (review F3). E1F: no host LIM write at all while
        stage 0 zero-fills, then the image only through write_image() before the entry word (stage0_enter)."""
        released = self.released.get(t) is not False
        s0 = self._stage0.get(t)
        if s0 == "zeroing":
            self._refuse(PolicyError, "E1F: stage 0 is zero-filling LIM; no host LIM write before its done word",
                         xy, addr, size, value)
        if ctx == "lim_init" and released:
            self._refuse(PolicyError, f"LIM init after release (tile {t} released={self.released.get(t)}): it would "
                                      "overwrite the running monitor", xy, addr, size, value)
        if (addr <= regs.STAGE0_ENTRY_WORD < addr + size) and s0 == "zeroed" and ctx != "stage0_entry":
            self._refuse(PolicyError, "E1F entry word: only stage0_enter() writes it", xy, addr, size, value)
        owner = regs.lim_line_owner(addr, released)
        if owner == "x280" and released and regs.in_monitor_region(addr) and ctx == "image" and s0 == "zeroed":
            owner = "host"  # E1F step D: the image, written once before the monitor has ever run
        if size == 4:
            if not self.lim[t].is_init(addr):  # PLAN R26: no partial write before the line is initialised
                self._refuse(PolicyError, f"partial write to uninitialised LIM line 0x{addr & ~63:x}", xy, addr, size, value)
            if owner == "kmd" and value != 0:
                self._refuse(PolicyError, "L38 (HOST_GONE) accepts only the attach clear (0)", xy, addr, size, value)
            if owner == "x280" and released:
                self._refuse(PolicyError, f"line 0x{addr & ~63:x} is X280-owned (one writer per line)", xy, addr, size, value)
        elif released and owner != "host":
            self._refuse(PolicyError, f"line 0x{addr:x} is not host-owned once the tile is released "
                                      f"({'monitor region' if regs.in_monitor_region(addr) else owner})",
                         xy, addr, size, value)

    def _lim_written(self, t: int, lo: int, hi: int) -> None:
        """A write into a verified image invalidates it: release() then needs write_image() again."""
        img = self._boot[t]["image"]
        if img is not None and lo < regs.IMAGE_BASE + img[1] and regs.IMAGE_BASE < hi:
            self._boot[t]["image"] = None

    def _w32(self, va_base: int, w: Window, addr: int, value: int, ctx: str,
             pre_store: Optional[Callable[[], None]] = None) -> None:
        """Policy, then `pre_store` (an interlock that must read the device as late as possible: it runs after the
        policy and raises to stop the store), then the one 4 B UC store."""
        xy = tuple(w.spec.xy)
        self._policy(xy, addr, 4, value, ctx)
        if pre_store is not None:
            pre_store()
        self.wcio.uc_write32(va_base + addr - w.spec.base, value)
        self._op("W32", xy, addr, 4, value, ctx)
        self._ecc(xy, "uc_w32", addr, 1)
        if ctx == "pll4_step":
            self._pll_i += 1

    @contextmanager
    def _transient(self, name: str, ctx: str, writable: bool = True):
        """A UC mapping of window `name` that exists only for one privileged ctx (RW only if `writable`; never for a
        read-only probe tile)."""
        w = self.windows[name]
        if writable and self._probe_only(w.spec.xy) is not None:
            self._refuse(PolicyError, f"{name}: L2CPU{regs.tile_of(w.spec.xy)} is a read-only probe tile (S5a): no "
                                      f"writable mapping for {ctx}", w.spec.xy, w.spec.base)
        va = self._map(w, "uc", PROT_READ | (PROT_WRITE if writable else 0))

        def w32(addr: int, value: int, pre_store: Optional[Callable[[], None]] = None) -> None:
            if not writable:
                self._refuse(PolicyError, f"{name} mapped read-only for {ctx}", w.spec.xy, addr, 4, value)
            self._w32(va, w, addr, value, ctx, pre_store)

        try:
            yield w32, lambda addr: self._tr32(va, w, addr)
        finally:
            self._unmap(w, va, "uc")

    def _tr32(self, va: int, w: Window, addr: int) -> int:
        xy = tuple(w.spec.xy)
        why = regs.classify_read(xy, addr, 4)
        if why:
            self._refuse(PolicyError, why, xy, addr, 4)
        if not w.spec.base <= addr <= w.spec.base + w.spec.map_len - 4:  # e.g. NIU #0 past a non-probe xp prefix
            self._refuse(PolicyError, f"{w.spec.name} does not cover 0x{addr:x}+4 (maps 0x{w.spec.map_len:x} B from "
                                      f"0x{w.spec.base:x})", xy, addr, 4)
        self._need_batch(xy, addr)
        v = self.wcio.uc_read32(va + addr - w.spec.base)
        self._op("R32", xy, addr, 4, v, "transient")
        return v

    # -- LIM (the only persistently writable target) --
    def uc_write32(self, addr: int, value: int, tile: int = regs.DEFAULT_TILE, *, _ctx: str = "") -> None:
        """4 B UC write to an initialised, host-owned LIM line (e.g. seq_req, the L38 clear at attach)."""
        w = self.windows[f"lim{tile}"]
        if not (regs.LIM_BASE <= addr < regs.LIM_END):
            self._refuse(PolicyError, f"uc_write32 is LIM-only (0x{addr:x})", w.spec.xy, addr, 4, value)
        self._w32(w.uc, w, addr, value, _ctx)
        self._lim_written(tile, addr, addr + 4)

    def _line_checks(self, xy: tuple, addr: int, n: int, tile: int, ctx: str) -> None:
        """What a 64 B-line LIM write of [addr, addr + n) must pass before any byte moves: whole aligned lines inside
        LIM; per line classify, deny, ownership (one writer per line) and the open batch. The first and last line,
        every non-host-owned line and the E1F entry line are checked one by one; the host-owned lines between them
        share the first line's verdict (same window, range and owner)."""
        if n == 0 or n % regs.LINE or addr % regs.LINE:
            self._refuse(PolicyError, f"WC writes are whole 64 B-aligned lines (0x{addr:x}, {n} B)", xy, addr, n)
        if not (regs.LIM_BASE <= addr and addr + n <= regs.LIM_END):
            self._refuse(PolicyError, f"WC write 0x{addr:x}+{n} outside LIM", xy, addr, n)
        released = self.released.get(tile) is not False
        for a in range(addr, addr + n, regs.LINE):  # every line: classify, deny, ownership, batch
            if (a == addr or regs.lim_line_owner(a, released) != "host" or a + regs.LINE == addr + n
                    or a == regs.STAGE0_ENTRY_WORD & ~63):
                self._policy(xy, a, regs.LINE, None, ctx)

    def wc_write_lines(self, addr: int, data, tile: int = regs.DEFAULT_TILE, *, fence_each: bool = False,
                       _ctx: str = "", _store: str = "nt64") -> int:
        """Full 64 B lines through the WC mapping (libwcio), then sfence; marks the lines initialised. `_store`: 'nt64'
        (one 64 B non-temporal store per line, every caller) or '2x32' (two 32 B ymm stores per line: S1d's LIM init
        variant, allowed only as regs LIM_INIT_WC32, i.e. in ctx lim_init)."""
        w = self.windows[f"lim{tile}"]
        xy = tuple(w.spec.xy)
        n = len(memoryview(data).cast("B"))
        if _store not in ("nt64", "2x32"):
            self._refuse(PolicyError, f"WC store kind {_store!r}", xy, addr, n)
        self._line_checks(xy, addr, n, tile, _ctx)
        if _store == "2x32":  # the 32 B width has its own allowlist entry: first and last store of the range
            self._policy(xy, addr, 32, None, _ctx)
            self._policy(xy, addr + n - 32, 32, None, _ctx)
        nlines = n // regs.LINE
        src, keep = _buffer_addr(data)
        if _store == "2x32":
            self.wcio.store_lines_2x32(w.wc + addr - w.spec.base, src, nlines, fence_each)
        else:
            self.wcio.store_lines(w.wc + addr - w.spec.base, src, nlines, fence_each)
        del keep
        self.lim[tile].mark(addr, nlines)
        self._lim_written(tile, addr, addr + n)
        self._op("WC64", xy, addr, regs.LINE, None, (f"x{nlines} lines{' fence_each' if fence_each else ''}"
                                                     f"{' 2x32' if _store == '2x32' else ''} {_ctx}").rstrip())
        self._ecc(xy, "wc_2x32" if _store == "2x32" else "wc_nt64_fenced" if fence_each else "wc_nt64", addr, nlines)
        return nlines

    def _uc8_write_lines(self, addr: int, data, tile: int, _ctx: str) -> int:
        """Full 64 B lines as aligned 8 B UC stores (libwcio wcio_uc_store64), S1d's 'uc8' LIM init variant: the same
        whole-line checks as a WC line write (range, per-line policy, ownership, batch), plus the 8 B width's own
        allowlist entry (regs LIM_INIT_UC8: ctx lim_init only); marks the lines initialised."""
        w = self.windows[f"lim{tile}"]
        xy = tuple(w.spec.xy)
        n = len(memoryview(data).cast("B"))
        self._line_checks(xy, addr, n, tile, _ctx)
        self._policy(xy, addr, 8, None, _ctx)
        self._policy(xy, addr + n - 8, 8, None, _ctx)
        src, keep = _buffer_addr(data)
        self.wcio.uc_store64(w.uc + addr - w.spec.base, src, n // 8)
        del keep
        nlines = n // regs.LINE
        self.lim[tile].mark(addr, nlines)
        self._lim_written(tile, addr, addr + n)
        self._op("W64", xy, addr, 8, None, f"x{n // 8} words ({nlines} lines) uc8 {_ctx}".rstrip())
        self._ecc(xy, "uc_w64", addr, n // 8)
        return nlines

    def lim_init(self, lo: int = regs.LIM_BASE, hi: int = regs.LIM_END, tile: int = regs.DEFAULT_TILE, *,
                 pattern: Optional[Callable[[int], bytes]] = None, chunk_lines: int = 2048, store: str = "a64") -> int:
        """Method A (PLAN §1.4.4): every line in [lo, hi) written by one 64 B NT store + sfence (store 'a64', the
        boot's). S1d's other stores (LIM_INIT_STORES: 'a64nf', 'wc2x32', 'uc8') write the same whole lines another
        way, under the same checks. Pre-release only: refused unless read_l2cpu_reset() showed this tile still in
        reset (review F3)."""
        if store not in LIM_INIT_STORES:
            raise PolicyError(f"lim_init store {store!r} (one of {', '.join(LIM_INIT_STORES)})")
        mapping, kind, fence_each, _ = LIM_INIT_STORES[store]
        self._lim_init_allowed(lo, hi, tile)
        done = 0
        for a in range(lo, hi, chunk_lines * regs.LINE):
            n = min(chunk_lines, (hi - a) // regs.LINE)
            buf = bytearray(n * regs.LINE) if pattern is None else bytearray(b"".join(pattern(a + i * regs.LINE)
                                                                                       for i in range(n)))
            if mapping == "uc":
                done += self._uc8_write_lines(a, buf, tile, _ctx="lim_init")
            else:
                done += self.wc_write_lines(a, buf, tile, fence_each=fence_each, _ctx="lim_init", _store=kind)
        return done

    def _lim_init_allowed(self, lo: int, hi: int, tile: int) -> None:
        """lim_init's per-call checks: a line-aligned range inside LIM, and the tile known to be in reset."""
        if lo % regs.LINE or hi % regs.LINE or not (regs.LIM_BASE <= lo < hi <= regs.LIM_END):
            raise PolicyError(f"lim_init range 0x{lo:x}..0x{hi:x}")
        if self.released.get(tile) is not False:
            self._refuse(PolicyError, f"lim_init: tile {tile} released={self.released.get(tile)} (a running or unknown "
                                      "hart; LIM init is a pre-release step)", regs.L2CPU_XY[tile], lo, hi - lo)

    def lim_init_check(self, lo: int, hi: int, tile: int = regs.DEFAULT_TILE) -> int:
        """Method B (server_g1's x280_lim_write: ttnn write_to_core, not this fd) moves no byte through kmd, so before
        each of its writes x280d asks here for exactly what lim_init and wc_write_lines would check for the same lines:
        the release state, then per line the write policy, ownership and the open l2cpu_batch (verify:all D12).
        Returns the line count; nothing is written and the lines are not marked initialised."""
        self._lim_init_allowed(lo, hi, tile)
        self._line_checks(regs.L2CPU_XY[tile], lo, hi - lo, tile, "lim_init")
        self._op("CHECK", regs.L2CPU_XY[tile], lo, hi - lo, None, "lim_init (method B, written by server_g1)")
        return (hi - lo) // regs.LINE

    def assume_lim_initialised(self, tile: int, lo: int, hi: int, reason: str) -> None:
        """attach (PLAN T-3.8): a previous x280d of this hold initialised LIM; recorded with its reason. E1F: the
        caller saw stage 0's done word, which ends the 'zeroing' state."""
        if not reason:
            raise PolicyError("assume_lim_initialised needs a reason")
        self.lim[tile].mark(lo, (hi - lo) // regs.LINE)
        if self._stage0.get(tile) == "zeroing" and (lo, hi) == (regs.LIM_BASE, regs.LIM_END):
            self._stage0[tile] = "zeroed"
        self._op("ASSUME_INIT", regs.L2CPU_XY[tile], lo, hi - lo, None, reason)

    def write_image(self, tile: int, image: bytes) -> int:
        """Boot step 3: the monitor image as full 64 B lines at IMAGE_BASE, read back line by line; the read-back
        CRC is recorded for release(). Before release, or in E1F between stage 0's done word and the entry word."""
        data = bytes(image) + bytes(-len(image) % regs.LINE)
        xy = regs.L2CPU_XY[tile]
        if not data or len(data) > regs.MAILBOX_BASE - regs.IMAGE_BASE:
            self._refuse(PolicyError, f"image of {len(data)} B does not fit the monitor region", xy, regs.IMAGE_BASE)
        if self.released.get(tile) is not False and self._stage0.get(tile) != "zeroed":
            self._refuse(PolicyError, f"write_image: tile {tile} released={self.released.get(tile)}, E1F state "
                                      f"{self._stage0.get(tile)}", xy, regs.IMAGE_BASE, len(data))
        self.wc_write_lines(regs.IMAGE_BASE, bytearray(data), tile, _ctx="image")
        back = b"".join(self.read_lim_line(regs.IMAGE_BASE + a, tile) for a in range(0, len(data), regs.LINE))
        crc = zlib.crc32(back)
        if back != data:
            raise KmdError(f"L2CPU{tile} image read-back CRC 0x{crc:08x} != 0x{zlib.crc32(data):08x}")
        self._boot[tile]["image"] = (crc, len(data))
        return crc

    def stage0_enter(self, tile: int, entry: int = regs.RESET_PC) -> None:
        """E1F step D end: the entry word that sends stage 0 into crt0 (stage0.S); from here on the monitor region is
        the hart's like after any release."""
        if self._stage0.get(tile) != "zeroed" or self._boot[tile]["image"] is None or entry != regs.RESET_PC:
            self._refuse(PolicyError, f"stage0_enter: E1F state {self._stage0.get(tile)}, image "
                                      f"{self._boot[tile]['image']}, entry 0x{entry:x}", regs.L2CPU_XY[tile],
                         regs.STAGE0_ENTRY_WORD, 4, entry)
        self.uc_write32(regs.STAGE0_ENTRY_WORD, entry, tile, _ctx="stage0_entry")
        self._stage0[tile] = "entered"

    # -- privileged primitives (PLAN §5.2 boot sequence, PARK) --
    def pll4_step_to(self, mhz: int) -> dict:
        """Steps 6/8: tt-bh-linux's one-step PLL4 walk between the CMFW default (800 MHz) and 200 MHz."""
        if self._batches:
            self._refuse(InterlockError, "PLL4 stepping with an L2CPU batch open")
        if mhz not in regs.PLL4_STEP:
            self._refuse(PolicyError, f"no PLL4 sequence to {mhz} MHz")
        if mhz == 200 and self.pll_excursions:
            self._refuse(PolicyError, "one PLL4 excursion per session (DP4 releases both tiles in one excursion)")
        c1, c5, writes, e1, e5 = regs.PLL4_STEP[mhz]
        d = self.read_pll4()
        if (d["cntl_1"], d["cntl_5"]) != (c1, c5) or d["use_postdiv"] & 0xF != 0xF:
            why = (self.pll4_default_why(d) if mhz == 200 else
                   f"expected the 200 MHz state CNTL_1 0x{c1:08x} CNTL_5 0x{c5:08x}")
            self._refuse(InterlockError, f"PLL4 not at the sequence start for {mhz} MHz ({why}): {_hexd(d)}")
        # clock.py semantics (BHL clock.py:58-60, 40-45): the walk is the one derived from CNTL_1/5 as just read, so
        # refdiv and the postdiv byte stay as read; with the exact start check it is the precomputed table
        if tuple(regs.pll4_step_writes(d["cntl_1"], d["cntl_5"], mhz)) != writes:
            self._refuse(InterlockError, f"PLL4 walk from {_hexd(d)} differs from regs.PLL4_STEP[{mhz}]")
        self._pll_expect, self._pll_i = list(writes), 0
        if mhz == 200:
            self.pll_excursions += 1
        self.pll_mhz = None
        try:
            with self._transient("arc", "pll4_step") as (w32, r32):
                cur = (c1, c5)
                for i, (addr, value) in enumerate(writes):
                    # review F5: re-read before EVERY write. The policy check runs first and the interlock reads last,
                    # right before the store (CNTL_0 and PLL_LOCK the very last), so nothing sits between them
                    w32(addr, value, pre_store=lambda cur=cur, i=i: self._pll_still(r32, cur, mhz, i, len(writes)))
                    cur = (value, cur[1]) if addr == regs.PLL4_CNTL_1 else (cur[0], value)
                    self.step_sleep()
                self._pll_still(r32, cur, mhz, len(writes), len(writes))
        finally:
            self._pll_expect, self._pll_i = [], 0
        d = self.read_pll4()
        if (d["cntl_1"], d["cntl_5"]) != (e1, e5):
            raise InterlockError(f"PLL4 read-back after the {mhz} MHz walk: {_hexd(d)}")
        self.pll_mhz = mhz
        return d

    def _pll_still(self, r32, cur: tuple, mhz: int, i: int, n: int) -> None:
        """PLL4 must still be exactly where the walk left it, locked and running, or CMFW relocked it meanwhile.
        CMFW re-programs PLL4 from its cached settings (fbdiv 64, postdivs 1) on every power message (FW
        bh_power.c:100-104 -> 32-57 -> clock_control_tt_bh.c:361-390, 316-358: bypass, power down, rewrite
        CNTL_1/5, relock): one more step of ours would then mix our fbdiv with its postdiv 1, i.e. L2CPUCLK up to
        ~2x nominal with the harts running. So on the first mismatch nothing more is written: PLL4 stays CMFW's.
        Read order: CNTL_1, CNTL_5, USE_POSTDIV (a cleared bit runs that L2CPUCLK at the VCO, FW pll.c:438-452), then
        CNTL_0 and PLL_LOCK last, the relock evidence nearest to the store that follows (pll4_step_to)."""
        c1, c5, up = r32(regs.PLL4_CNTL_1), r32(regs.PLL4_CNTL_5), r32(regs.PLL4_USE_POSTDIV)
        c0, lock = r32(regs.PLL4_CNTL_0), r32(regs.PLL_LOCK)
        if ((c1, c5) != cur or up & regs.PLL4_USE_POSTDIV_MASK != regs.PLL4_USE_POSTDIV_MASK
                or c0 & regs.PLL_CNTL_0_MASK != regs.PLL_CNTL_0_RUNNING or not lock & regs.pll_lock_bit(4)):
            self.pll_interlock = (f"PLL4 changed under the walk to {mhz} MHz before write {i + 1}/{n}: expected "
                                  f"CNTL_1 0x{cur[0]:08x} CNTL_5 0x{cur[1]:08x} USE_POSTDIV & 0x"
                                  f"{regs.PLL4_USE_POSTDIV_MASK:x} running and locked, read CNTL_1 0x{c1:08x} CNTL_5 "
                                  f"0x{c5:08x} USE_POSTDIV 0x{up:x} CNTL_0 0x{c0:x} PLL_LOCK 0x{lock:x} (CMFW "
                                  f"relock?): walk stopped, PLL4 left to CMFW")
            self._refuse(InterlockError, self.pll_interlock, regs.ARC_XY, regs.PLL4_BASE)

    def boot_artefacts_missing(self, tile: int, image_crc: Optional[int], gp_code: bytes, stage0: bool = False) -> str:
        """'' when tile `tile` may leave reset, else why not (PLAN §5.2 E2 preconditions, review F2). Everything is
        what THIS session wrote and read back: reset vectors and RNMI handlers (steps 4-5), the GP-scratch code,
        and for a normal boot every LIM line initialised (step 2) plus the image with this CRC (step 3). E1F
        (stage0=True) releases onto the stage-0 stub with LIM not yet initialised: that path needs DP3(d)."""
        b = self._boot[tile]
        want = (regs.STAGE0_ENTRY if stage0 else regs.RESET_PC, regs.PARK_STUB, regs.RNMI_EXC_STUB)
        if b["vectors"] is None:
            return "reset vectors and RNMI handlers were not written and read back in this session"
        if b["vectors"] != want:
            return f"vectors {tuple(hex(v) for v in b['vectors'])} != {tuple(hex(v) for v in want)}"
        # per hart, as read back: hart 0 the park stub always; harts 1-3 the same, or the --harts 4 pair
        sec = (want[0], regs.RESET_PC, regs.MH_SEC_RNMI_EXC) if b["mh_vectors"] else want
        by = b.get("by_hart") or []
        if len(by) != regs.HARTS_PER_TILE or by[0] != want or any(v != sec for v in by[1:]):
            return f"per-hart vectors {[tuple(hex(x) for x in v) for v in by]} != hart 0 {want}, harts 1-3 {sec}"
        if b["mh_vectors"]:
            if not stage0:
                return "the --harts 4 secondary vectors go only with the stage-0 (E1F) boot"
            if os.environ.get("X280_MH_APPROVED") != "1":
                return "the --harts 4 secondary vectors need the user's approval: X280_MH_APPROVED=1"
        code = bytes(gp_code or b"")
        if len(code) != regs.GP_SCRATCH_SIZE or not any(code):
            return f"gp_code must be the {regs.GP_SCRATCH_SIZE} B park stub (or E1F stage 0), got {len(code)} B"
        if b["gp"] != code:
            return "the GP scratch was not written with gp_code and read back in this session"
        if stage0:
            if os.environ.get("X280_E1F_APPROVED") != "1":
                return "E1F (stage 0) needs the user's explicit DP3(d) approval: X280_E1F_APPROVED=1"
            return ""
        if self.lim[tile].count() != regs.LIM_LINES:
            return f"LIM not initialised: {self.lim[tile].count()} of {regs.LIM_LINES} lines"
        if b["image"] is None:
            return "the monitor image was not written and read back in this session"
        if image_crc is None or b["image"][0] != image_crc & 0xFFFF_FFFF:
            return f"image CRC 0x{b['image'][0]:08x} read back != the release's image_crc {image_crc}"
        return ""

    def release(self, tiles: tuple, ledger, image_crc: Optional[int], *, gp_code: bytes, stage0: bool = False) -> tuple:
        """Step 7: the one L2CPU_RESET RMW of the session, value = read | mask with [3:0] == 0xF, then read-back.
        Refused unless every tile's boot artefacts were written and read back (boot_artefacts_missing). The ledger
        marks the tiles released before the write: an attempt spends the release."""
        if self.reset_rmw_done:
            self._refuse(PolicyError, "L2CPU_RESET was already written once this session")
        if self._batches:
            self._refuse(InterlockError, "release with an L2CPU batch open")
        mask = 0
        for t in tiles:
            if t not in self.tiles:
                self._refuse(PolicyError, f"tile {t} not in this session")
            mask |= regs.release_bit(t)
        if mask not in regs.RELEASE_MASKS:
            self._refuse(PolicyError, f"release mask 0x{mask:x} not in regs.RELEASE_MASKS")
        for t in tiles:
            why = self.boot_artefacts_missing(t, image_crc, gp_code, stage0)
            if why:
                self._refuse(PolicyError, f"release refused, L2CPU{t}: {why}")
        d = self.read_pll4()
        if self.pll_mhz != 200 or (d["cntl_1"], d["cntl_5"]) != (regs.PLL4_200_CNTL_1, regs.PLL4_200_CNTL_5):
            self._refuse(InterlockError, f"harts leave reset only with PLL4 at 200 MHz (ISA README.md:32-33): {_hexd(d)}")
        prev = self.read_l2cpu_reset()
        if prev != regs.L2CPU_RESET_AT_SESSION_START:
            self._refuse(InterlockError, f"L2CPU_RESET=0x{prev:x} at release (expected 0x0F; a tile was released "
                                         f"without a chip reset: exit 3, PLAN §5.2)", regs.ARC_XY, regs.L2CPU_RESET)
        for t in tiles:
            if ledger.state(t) != "fresh":
                self._refuse(PolicyError, f"ledger: tile {t} is {ledger.state(t)}, not fresh")
        value = prev | mask
        assert value & regs.L2CPU_RESET_TILE_RESETS == regs.L2CPU_RESET_TILE_RESETS
        for t in tiles:
            ledger.mark_released(t, image_crc)
            if stage0:
                self._stage0[t] = "zeroing"  # before the write, like the ledger: no host LIM write from here on
        self.reset_rmw_done = True  # before the write: an exception mid-write must not allow a retry
        self._rmw_armed = True
        try:
            with self._transient("arc", "l2cpu_reset_rmw") as (w32, _):
                w32(regs.L2CPU_RESET, value)
        finally:
            self._rmw_armed = False
        rb = self.read_l2cpu_reset()
        self._op("NOTE", regs.ARC_XY, regs.L2CPU_RESET, 4, rb, f"release RMW 0x{prev:02x} | 0x{mask:02x}, read-back")
        if rb != value:
            raise KmdError(f"L2CPU_RESET read-back 0x{rb:x} != written 0x{value:x}")
        return prev, value, rb

    def write_boot_vectors(self, tile: int, reset_pc: int = regs.RESET_PC, rnmi: int = regs.PARK_STUB,
                           rnmi_exc: int = regs.RNMI_EXC_STUB, *, harts: int = 1) -> None:
        """Steps 4-5: RNMI handler / exception handler and reset pc of harts 0-3, before release, read back.
        harts=4 (`x280d serve --harts 4`, X280_MH_APPROVED=1; the 4-hart bench): harts 1-3 get RNMI handler RESET_PC (crt0
        _start, whose secondary path adopts the hart into mh.c) and RNMI exception handler MH_SEC_RNMI_EXC; hart 0 keeps
        rnmi / rnmi_exc. The writes are the same addresses in the same order either way (harts=1: byte-identical)."""
        if self.released.get(tile) is not False:
            self._refuse(PolicyError, f"vectors only before release; read_l2cpu_reset() first (tile {tile})")
        if harts not in (1, regs.MH_HARTS):
            self._refuse(PolicyError, f"harts={harts}: 1 or {regs.MH_HARTS}")
        if harts == regs.MH_HARTS and os.environ.get("X280_MH_APPROVED") != "1":
            self._refuse(PolicyError, "the --harts 4 secondary RNMI vectors need the user's approval: X280_MH_APPROVED=1",
                         regs.L2CPU_XY[tile], regs.rnmi_handler(1), 4, regs.RESET_PC)
        writes, by_hart = [], []
        for h in range(regs.HARTS_PER_TILE):
            hv, he = (rnmi, rnmi_exc) if h == 0 or harts == 1 else (regs.RESET_PC, regs.MH_SEC_RNMI_EXC)
            writes += [(regs.rnmi_handler(h), hv), (regs.rnmi_handler(h) + 4, 0),
                       (regs.rnmi_exc_handler(h), he), (regs.rnmi_exc_handler(h) + 4, 0)]
            by_hart.append((reset_pc, hv, he))
        for h in range(regs.HARTS_PER_TILE):
            writes += [(regs.reset_vector(h), reset_pc), (regs.reset_vector(h) + 4, 0)]
        self._boot[tile].update(vectors=None, mh_vectors=False, by_hart=None)
        self._mh_arm[tile] = harts == regs.MH_HARTS
        try:
            with self._transient(f"xp{tile}", "boot_vectors") as (w32, r32):
                for a, v in writes:
                    w32(a, v)
                bad = [(hex(a), hex(r32(a)), hex(v)) for a, v in writes if r32(a) != v]
        finally:
            self._mh_arm[tile] = False
        if bad:
            raise KmdError(f"vector read-back mismatch: {bad}")
        self._boot[tile].update(vectors=(reset_pc, rnmi, rnmi_exc), mh_vectors=harts == regs.MH_HARTS, by_hart=by_hart)

    def write_gp_scratch(self, tile: int, code: bytes) -> None:
        """Park stub (+ RNMI-exception spin stub at +48) into the 64 B GP scratch, read back (WP-3.2, E1F). The GP
        scratch is the RNMI handler code, so after the release it is never rewritten, with one exception: E1F swaps
        stage 0 for the park stub once, after the monitor came up (PLAN §5.2 "E1F stage-0 initializer"). Hart 0 runs
        the new stub on its next RNMI only after a fence.i of its own has dropped stage 0's line from its L1I: crt0's
        on entry and the FENCE_I x280d sends after the swap (the host cannot invalidate a hart's L1I, Caches.md:22)."""
        if len(code) > regs.GP_SCRATCH_SIZE:
            raise PolicyError(f"GP scratch holds {regs.GP_SCRATCH_SIZE} B, got {len(code)}")
        swap = self._stage0.get(tile) == "entered"
        if self.released.get(tile) is not False and not swap:
            self._refuse(PolicyError, f"GP scratch (the RNMI handler code) only before release (tile {tile} "
                                      f"released={self.released.get(tile)}, E1F state {self._stage0.get(tile)})",
                         regs.L2CPU_XY[tile], regs.GP_SCRATCH, len(code))
        code = bytes(code) + b"\0" * (-len(code) % 4)
        words = [int.from_bytes(code[i:i + 4], "little") for i in range(0, len(code), 4)]
        self._boot[tile]["gp"] = None
        with self._transient(f"xp{tile}", "gp_scratch") as (w32, r32):
            for i, v in enumerate(words):
                w32(regs.GP_SCRATCH + 4 * i, v)
            bad = [i for i, v in enumerate(words) if r32(regs.GP_SCRATCH + 4 * i) != v]
        if bad:
            raise KmdError(f"GP scratch read-back mismatch at words {bad}")
        self._boot[tile]["gp"] = code
        if swap:
            self._stage0[tile] = "stub"

    def mh_rnmi_refusal(self, tile: int, harts_mask: int) -> str:
        """'' when RNMI bits 1-3 of `harts_mask` may be set (adoption / recovery of harts 1-3, mh.c), else why not. With the
        boot's default vectors an RNMI strands such a hart until a chip reset (stale stage-0 line in its L1I, or crt0's
        cold wfi with NMIE = 0), so: this session wrote and read back the --harts 4 secondary vectors; the tile runs the
        E1F-booted monitor with the park stub swapped in (state 'stub'); the monitor reports multi-hart support in L36.mh
        (read here); and never together with bit 0, hart 0's PARK."""
        if not harts_mask & regs.MH_SEC_MASK:
            return ""
        b = self._boot.get(tile) or {}
        if harts_mask & 1:
            return "bits 1-3 never together with bit 0 (PARK of hart 0)"
        if not b.get("mh_vectors"):
            return "this session did not write and read back the --harts 4 secondary RNMI vectors"
        if self.released.get(tile) is not True or self._stage0.get(tile) != "stub":
            return (f"tile {tile} released={self.released.get(tile)}, E1F state {self._stage0.get(tile)} "
                    "(needs the stage-0 boot with the park stub swapped in: 'stub')")
        mh = self.read_lim32(regs.MB_IDENT_MH, tile)
        if not mh & regs.MH_CAP_BIT:
            return f"the monitor does not report multi-hart support (L36.mh = 0x{mh:x})"
        return ""

    def rnmi_set(self, tile: int, harts_mask: int) -> int:
        """PARK protocol: set (or clear with 0) the level-triggered RNMI bits of harts (ISA RNMIs.md:15, 38-46). Bits 1-3
        only as mh_rnmi_refusal() allows (checked before the transient mapping exists)."""
        why = self.mh_rnmi_refusal(tile, harts_mask)
        if why:
            self._refuse(PolicyError, f"RNMI trigger 0x{harts_mask:x} refused: {why}", regs.L2CPU_XY[tile],
                         regs.RNMI_TRIGGER, 4, harts_mask)
        with self._transient(f"xp{tile}", "park") as (w32, r32):
            w32(regs.RNMI_TRIGGER, harts_mask)
            return r32(regs.RNMI_TRIGGER)

    def read_hart_status(self, tile: int) -> int:
        """cease/halt/wfi/debug bits (ISA MemoryMap.md:115), via a transient read-only use of the XP window."""
        with self._transient(f"xp{tile}", "xp_read", writable=False) as (_, r32):
            return r32(regs.HART_STATUS) & 0xFFFF

    def read_rnmi_trigger(self, tile: int) -> int:
        """The level-triggered RNMI bits (ISA RNMIs.md:15, 38-46), transient read-only XP mapping: a set bit h means
        hart h sits in the park stub (park_stub.S spins while it is set), e.g. after an x280d died inside PARK."""
        with self._transient(f"xp{tile}", "xp_read", writable=False) as (_, r32):
            return r32(regs.RNMI_TRIGGER) & regs.RNMI_TRIGGER_VALID_MASK

    def read_tlb_slots(self, tile: int) -> dict:
        """S5a: all 256 X280 -> NoC TLB window slots of a tile from the NoC side (regs.tlb_small_cfg / tlb_large_cfg,
        ISA TLBWindows.md:20-34): 224 x 4 and 32 x 3 words in one transient read-only XP mapping."""
        with self._transient(f"xp{tile}", "xp_read", writable=False) as (_, r32):
            small = [[r32(regs.tlb_small_cfg(i) + 4 * n) for n in range(regs.TLB_SMALL_WORDS)]
                     for i in range(regs.TLB_SMALL_N)]
            large = [[r32(regs.tlb_large_cfg(j) + 4 * n) for n in range(regs.TLB_LARGE_WORDS)]
                     for j in range(regs.TLB_LARGE_N)]
        return {"small": small, "large": large}

    def read_niu0(self, tile: int) -> dict:
        """S5a: NIU #0's NOC_NODE_ID, NOC_ENDPOINT_ID and NIU_CFG_0 through the external-peripheral high alias (ISA
        MemoryMap.md:121; NoC/MemoryMap.md:187-257), transient read-only XP mapping (a probe session's longer prefix)."""
        with self._transient(f"xp{tile}", "xp_read", writable=False) as (_, r32):
            return {"noc_node_id": r32(regs.NIU0_NOC_NODE_ID), "noc_endpoint_id": r32(regs.NIU0_NOC_ENDPOINT_ID),
                    "niu_cfg_0": r32(regs.NIU0_NIU_CFG_0)}

    def l2pf_setup(self, tile: int) -> None:
        """Step 9: L2PF1 basic/user control of harts 0-3 (ISA Caches.md:57-64), read back."""
        with self._transient(f"ccache{tile}", "l2pf") as (w32, r32):
            for a, v in regs.L2PF_WRITES:
                w32(a, v)
            bad = [(hex(a), hex(r32(a))) for a, v in regs.L2PF_WRITES if r32(a) != v]
        if bad:
            raise KmdError(f"L2PF read-back mismatch: {bad}")

    def set_noc_cleanup(self, tile: int = regs.DEFAULT_TILE, enabled: bool = True, fd: Optional[int] = None) -> SetNocCleanup:
        """Register (or clear) the KMD's close-time write of 'GONE' into L38 (KMD ioctl.h:331-360, chardev.c:1030-1039)."""
        xy, addr = regs.L2CPU_XY[tile], regs.MB_HOST_GONE
        if self._probe_only(xy) is not None:
            self._refuse(PolicyError, f"SET_NOC_CLEANUP on L2CPU{tile}, a read-only probe tile (S5a)", xy, addr, 4)
        allow, denies, why = regs.classify_write(xy, addr, 4)
        if allow is None or allow.name != "LIM" or denies or regs.lim_line_owner(addr) != "kmd":
            self._refuse(PolicyError, f"cleanup target not the allowlisted L38: {why}", xy, addr, 4)
        if enabled and not self.lim[tile].is_init(addr):  # the cleanup is itself a 4 B partial write (R26)
            self._refuse(PolicyError, "SET_NOC_CLEANUP before L38 is initialised", xy, addr, 4)
        s = SetNocCleanup(argsz=ctypes.sizeof(SetNocCleanup), flags=0, enabled=int(enabled), x=xy[0], y=xy[1],
                          noc=regs.NOC, reserved0=0, addr=addr, data=regs.HOST_GONE_VALUE)
        self._ioctl(IOCTL_SET_NOC_CLEANUP, s, f"({xy[0]},{xy[1]}) addr=0x{addr:x} data=0x{s.data:08x} "
                                              f"enabled={s.enabled} fd={self.fd if fd is None else fd}", fd)
        return s

    def open_test_fd(self) -> int:
        """Second fd for the E1a cleanup check; O_APPEND so it sends no power message (chardev.c:954, 1074-1083)."""
        fd = self.backend.open(power_aware=True)
        self._op("OPEN", detail=f"test fd {fd} O_APPEND (power-aware, no power message)")
        return fd

    def close_test_fd(self, fd: int) -> None:
        self.backend.close(fd)
        self._op("CLOSE", detail=f"test fd {fd}")


def _hexd(d: dict) -> str:
    return ", ".join(f"{k}=0x{v:08x}" for k, v in d.items())
