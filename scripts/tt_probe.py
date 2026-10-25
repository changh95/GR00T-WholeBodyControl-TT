#!/usr/bin/env python
"""The device guard's Tensix probe as a file, for chip_session.sh's re-probe after its mid-hold reset (PLAN.md WP-2.4
step 2). The five device lines are bin/_device_guard.sh:36-43 verbatim; the guard runs them with the environment of
:33-35, which chip_lib.sh's TT_PROBE default reproduces:

    env TT_METAL_HOME=$TTM PYTHONPATH=$TTM:$TTM/ttnn TT_METAL_CACHE=~/.cache/tt-metal-cache-gr00t \\
        TT_METAL_LOGGER_LEVEL=FATAL $TTM/python_env/bin/python $RT/scripts/tt_probe.py     (TTM = ~/experiments/gr00t/tt-metal)

It opens /dev/tenstorrent/0, so before importing ttnn it checks that an ancestor is the guard running under
with-device.sh's flock (flock -> bash _device_guard.sh -> ..., with-device.sh:10): that is the only place a device
open cannot collide with another holder. It reads /proc only and never opens the lock file.

Exit 0 = the card ran the 32x32 bf16 add; 1 = the op failed; 2 = refused (no guard ancestor).
--guard-check-only does the ancestor check and exits without importing ttnn (tests).
"""

import argparse
import os
import sys
import time


def _cmdline(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return []


def _ppid(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def guard_ancestor() -> int | None:
    """pid of the nearest ancestor running _device_guard.sh whose parent is `flock ... .p150a.lock ...`."""
    p = os.getppid()
    while p > 1:
        if any(os.path.basename(a) == "_device_guard.sh" for a in _cmdline(p)):
            parent = _cmdline(_ppid(p))
            if parent and os.path.basename(parent[0]) == "flock" and any(a.endswith(".p150a.lock") for a in parent):
                return p
        p = _ppid(p)
    return None


def probe() -> None:
    import torch
    import ttnn

    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    a = ttnn.from_torch(torch.ones(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.add(a, a)
    ttnn.synchronize_device(dev)
    out = ttnn.to_torch(b)
    ttnn.close_device(dev)
    assert float(out[0, 0, 0, 0]) == 2.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--guard-check-only", action="store_true", help="check the guard ancestor and exit")
    args = ap.parse_args(argv)
    guard = guard_ancestor()
    if guard is None:
        print("tt_probe: refused: not inside a with-device.sh hold (no _device_guard.sh ancestor under flock)",
              file=sys.stderr)
        return 2
    if args.guard_check_only:
        print(f"tt_probe: guard ancestor pid {guard}")
        return 0
    t0 = time.monotonic()
    try:
        probe()
    except AssertionError:
        print("tt_probe: FAILED: 1 + 1 != 2 on the device", file=sys.stderr)
        return 1
    print(f"tt_probe: ok in {time.monotonic() - t0:.1f} s (guard pid {guard})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
