"""50 Hz pacing on absolute CLOCK_MONOTONIC deadlines (PLAN.md §1.2 step 1, §1.6 "RT").

Tick k is due at T_k = t0 + k * 20 ms. Waiting for an absolute deadline instead of "sleep 20 ms
minus compute" keeps the error from accumulating. A tick that starts more than one period late is
a *slip*: t0 is re-anchored so the loop resumes from now instead of firing a burst of catch-up
ticks with stale state.

How to wait (``mode``), since rt_sim has neither SCHED_FIFO nor a negative nice (RLIMIT_RTPRIO 0):
- ``sleep``: clock_nanosleep(TIMER_ABSTIME) with 1 ns timer slack. An idle CPU 2 attracts other
  processes' new tasks (here a ``watch -n0.1 nvidia-smi`` lands there about 3 % of the time), and
  EEVDF lets them finish their slice first: measured 2.4 % of wake-ups over 0.5 ms late, max 3.7 ms.
- ``hybrid``: sleep to T_k - spin_ns, then spin.
- ``spin`` (default): never sleep between ticks, so CPU 2 never looks idle and the scheduler places
  waking tasks on the 28 idle cores instead: 0 of 1000 wake-ups over 1 us late in the same test.
  It costs the one core the plan reserves for the tick thread anyway.
"""

import ctypes
import ctypes.util
import time

from rtloop.schema import TICK_NS

CLOCK_MONOTONIC = 1
TIMER_ABSTIME = 1
PR_SET_TIMERSLACK = 29

now_ns = time.monotonic_ns


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    _nanosleep = _libc.clock_nanosleep  # a CDLL call releases the GIL while it sleeps
    _nanosleep.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(_Timespec), ctypes.c_void_p)
    _nanosleep.restype = ctypes.c_int
    _prctl = _libc.prctl
except (OSError, AttributeError):  # not Linux/glibc: fall back to time.sleep
    _libc = _nanosleep = _prctl = None


def set_timer_slack(ns: int = 1) -> bool:
    """Timer slack of the calling thread (Linux default 50 us); True if it was applied."""
    if _prctl is None:
        return False
    return _prctl(PR_SET_TIMERSLACK, ctypes.c_ulong(int(ns)), 0, 0, 0) == 0


def sleep_until(t_ns: int) -> None:
    """Block until CLOCK_MONOTONIC >= t_ns (returns at once if it already is)."""
    if _nanosleep is None:
        rem = t_ns - now_ns()
        if rem > 0:
            time.sleep(rem / 1e9)
        return
    ts = _Timespec(t_ns // 1_000_000_000, t_ns % 1_000_000_000)
    while _nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, ctypes.byref(ts), None) != 0:
        pass  # EINTR: sleep again to the same absolute deadline


def spin_until(t_ns: int) -> None:
    """Busy-wait to t_ns. Pure Python, so another thread asking for the GIL gets it after at most
    one switch interval (5 ms); rt_sim's only such thread is pyzmq's buffer collector."""
    while now_ns() < t_ns:
        pass


PACING = ("spin", "hybrid", "sleep")


class Pacer:
    """Absolute-deadline pacer for one episode: ``anchor()`` once, then ``wait(k)`` per tick."""

    def __init__(self, period_ns: int = TICK_NS, mode: str = "spin", spin_ns: int = 4_000_000):
        if mode not in PACING:
            raise ValueError(f"pacing {mode!r} not in {PACING}")
        self.period_ns = int(period_ns)
        self.mode = mode
        self.spin_ns = int(spin_ns)
        self.t0 = 0
        self.slips = 0

    def _wait_until(self, t_ns: int) -> None:
        if self.mode == "spin":
            spin_until(t_ns)
        elif self.mode == "hybrid":
            if now_ns() < t_ns - self.spin_ns:
                sleep_until(t_ns - self.spin_ns)
            spin_until(t_ns)
        else:
            sleep_until(t_ns)

    def anchor(self, t0_ns: int | None = None) -> int:
        """T_0 = now (lifecycle step 6, after the pre-roll)."""
        self.t0 = now_ns() if t0_ns is None else int(t0_ns)
        self.slips = 0
        return self.t0

    def deadline(self, k: int) -> int:
        return self.t0 + k * self.period_ns

    def wait(self, k: int) -> tuple[int, int, bool]:
        """Sleep to T_k; returns (t_start_ns, late_ns, slip). On a slip, T_k is moved to now."""
        t_k = self.t0 + k * self.period_ns
        if now_ns() < t_k:
            self._wait_until(t_k)
        t = now_ns()
        late = t - t_k
        if late > self.period_ns:
            self.t0 = t - k * self.period_ns  # re-anchor: no catch-up burst
            self.slips += 1
            return t, late, True
        return t, late, False
