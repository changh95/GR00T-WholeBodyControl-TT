"""Thread placement of rt_sim (PLAN.md §1.1 "Affinity is applied in this order").

Library threads (EGL, zmq I/O, onnxruntime) inherit the affinity of the thread that creates them.
So the main thread first runs on the helper CPU (3) while it builds the env and creates the zmq
context, and only then moves itself to the tick CPU (2), whose SMT sibling (18) is left idle. The
reset window repeats the bracket because robosuite re-creates its renderer there. Linux
``sched_setaffinity(0, ...)`` applies to the calling thread only, which is exactly what this needs.
The onscreen viewer is a separate process (rtloop.viewer_proc) on the aux CPUs it shares with the HUD.
"""

import os
from pathlib import Path

TICK_CPU = 2
HELPER_CPU = 3
TICK_SIBLING = 18  # SMT sibling of CPU 2 on the 7950X (kept idle)
AUX_CPUS = (6, 7)  # the HUD and the viewer process (rtloop.viewer_proc)


def pin_current(cpu: int | set[int] | None) -> None:
    """Pin the calling thread; None leaves it alone (affinity disabled)."""
    if cpu is None:
        return
    os.sched_setaffinity(0, {cpu} if isinstance(cpu, int) else set(cpu))


def current() -> list[int]:
    return sorted(os.sched_getaffinity(0))


def zmq_context(cpu: int | None = HELPER_CPU):
    """A zmq.Context whose I/O thread runs on `cpu`.

    THREAD_AFFINITY_CPU_ADD only takes effect if set before the first socket starts the I/O thread;
    the I/O thread would inherit the creating thread's mask anyway, this makes it explicit.
    """
    import zmq

    ctx = zmq.Context(io_threads=1)
    if cpu is not None:
        ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, int(cpu))
    return ctx


def _cpus_allowed(status: str) -> str:
    for line in status.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return "?"


def parse_cpus(spec: str) -> list[int]:
    """'6,7' or '6-7' -> [6, 7]; '' -> []."""
    out: list[int] = []
    for part in str(spec).split(","):
        a, sep, b = part.strip().partition("-")
        if a:
            out.extend(range(int(a), int(b) + 1) if sep else [int(a)])
    return sorted(set(out))


def thread_report(pid: int | None = None) -> list[dict]:
    """Every thread of this process (or of `pid`) with its name and allowed CPUs (recorded in config.json)."""
    out = []
    main_tid = pid or os.getpid()
    tasks = Path(f"/proc/{pid}/task" if pid else "/proc/self/task")
    for task in sorted(tasks.iterdir(), key=lambda p: int(p.name)):
        try:
            comm = (task / "comm").read_text().strip()
            cpus = _cpus_allowed((task / "status").read_text())
        except OSError:  # thread exited meanwhile
            continue
        out.append({"tid": int(task.name), "name": comm, "cpus": cpus, "main": int(task.name) == main_tid})
    return out
