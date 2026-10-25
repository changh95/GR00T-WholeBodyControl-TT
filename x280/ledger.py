"""Per-hold X280 tile ledger, ~/experiments/gr00t/.x280_ledger.json (PLAN.md §5.2 session model, D13, DP3).

Harts leave reset once per chip reset (ISA L2CPUTile/README.md:29-30) and every X280 hold ends with a chip
reset (DEVX), so the ledger describes one hold. Its record is exactly the §5.2 schema:

    {hold_id, started_at, timer_heartbeat_at_start,
     tiles: {"0".."3": {state: fresh|released|parked|burned, image_crc, released_at}}}

`start_hold` replaces the previous hold's record (which it first appends to `<path>.history.jsonl`) and
compares TIMER_HEARTBEAT with it as an epoch cross-check: the counter restarts at CMFW boot and advances
once per 100 ms telemetry update (FW telemetry.c:278, 549), so a value well below the previous hold's
start value plus the elapsed wall time means the chip was reset in between. Writes are atomic
(tmp + fsync + rename) under an flock.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Optional

from x280 import regs

LEDGER_PATH = Path.home() / "experiments" / "gr00t" / ".x280_ledger.json"
TILES = ("0", "1", "2", "3")
STATES = ("fresh", "released", "parked", "burned")
# A release is spent once attempted: there is no way back to fresh without a chip reset.
TRANSITIONS = {"fresh": {"released", "burned"}, "released": {"parked", "burned"},
               "parked": {"released", "burned"}, "burned": set()}
KEYS = ("hold_id", "started_at", "timer_heartbeat_at_start", "tiles")
TILE_KEYS = ("state", "image_crc", "released_at")
HEARTBEAT_HZ = 1.0 / regs.TELEMETRY_PERIOD_S
EPOCH_SLACK_TICKS = 50  # 5 s of telemetry updates, plus 5% of the expected advance


class LedgerError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_time(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def new_record(hold_id: str, timer_heartbeat: int, started_at: Optional[str] = None) -> dict:
    return {"hold_id": str(hold_id), "started_at": started_at or utc_now(),
            "timer_heartbeat_at_start": int(timer_heartbeat),
            "tiles": {t: {"state": "fresh", "image_crc": None, "released_at": None} for t in TILES}}


def validate(rec: dict) -> None:
    """Raises LedgerError unless `rec` has exactly the §5.2 keys and legal values."""
    if not isinstance(rec, dict) or tuple(sorted(rec)) != tuple(sorted(KEYS)):
        raise LedgerError(f"ledger keys {sorted(rec) if isinstance(rec, dict) else rec!r} != {sorted(KEYS)}")
    if not isinstance(rec["hold_id"], str) or not isinstance(rec["timer_heartbeat_at_start"], int):
        raise LedgerError("hold_id must be str and timer_heartbeat_at_start int")
    _parse_time(rec["started_at"])
    tiles = rec["tiles"]
    if not isinstance(tiles, dict) or tuple(sorted(tiles)) != TILES:
        raise LedgerError(f"tiles keys {sorted(tiles)} != {list(TILES)}")
    for t, v in tiles.items():
        if not isinstance(v, dict) or tuple(sorted(v)) != tuple(sorted(TILE_KEYS)):
            raise LedgerError(f"tile {t} keys {sorted(v)} != {sorted(TILE_KEYS)}")
        if v["state"] not in STATES:
            raise LedgerError(f"tile {t} state {v['state']!r}")
        if v["image_crc"] is not None and not (isinstance(v["image_crc"], int) and 0 <= v["image_crc"] < 1 << 32):
            raise LedgerError(f"tile {t} image_crc {v['image_crc']!r}")
        if v["released_at"] is not None:
            _parse_time(v["released_at"])
        if v["state"] == "fresh" and v["released_at"] is not None:
            raise LedgerError(f"tile {t} is fresh but has released_at")


def epoch_verdict(prev: Optional[dict], heartbeat_now: int, t_now: float) -> str:
    """'first_hold' | 'reset' | 'no_reset' | 'inconsistent' (heartbeat ahead of wall time)."""
    if prev is None:
        return "first_hold"
    elapsed = max(0.0, t_now - _parse_time(prev["started_at"]))
    expected = prev["timer_heartbeat_at_start"] + elapsed * HEARTBEAT_HZ
    slack = EPOCH_SLACK_TICKS + 0.05 * elapsed * HEARTBEAT_HZ
    if heartbeat_now < expected - slack:
        return "reset"
    if heartbeat_now > expected + slack:
        return "inconsistent"
    return "no_reset"


class Ledger:
    def __init__(self, path: Path = LEDGER_PATH):
        self.path = Path(path)
        self.history = self.path.with_name(self.path.name + ".history.jsonl")
        self._lock = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def read(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        rec = json.loads(self.path.read_text())
        validate(rec)
        return rec

    def _write(self, rec: dict) -> None:
        validate(rec)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def start_hold(self, hold_id: Optional[str], timer_heartbeat: int, now: Optional[float] = None) -> dict:
        """New per-hold record (all tiles fresh). Returns {'previous', 'epoch', 'record'}."""
        t_now = datetime.now(timezone.utc).timestamp() if now is None else now
        started_at = datetime.fromtimestamp(t_now, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._locked():
            prev = self.read()
            verdict = epoch_verdict(prev, timer_heartbeat, t_now)
            if prev is not None:
                with open(self.history, "a") as f:
                    f.write(json.dumps(prev, sort_keys=True) + "\n")
            rec = new_record(hold_id or f"{started_at}-pid{os.getpid()}", timer_heartbeat, started_at)
            self._write(rec)
        return {"previous": prev, "epoch": verdict, "record": rec}

    def _current(self) -> dict:
        rec = self.read()
        if rec is None:
            raise LedgerError(f"no hold started ({self.path} missing)")
        return rec

    def state(self, tile: int) -> str:
        return self._current()["tiles"][str(tile)]["state"]

    def transition(self, tile: int, new: str, *, image_crc: Optional[int] = None) -> dict:
        if new not in STATES:
            raise LedgerError(f"unknown state {new!r}")
        with self._locked():
            rec = self._current()
            entry = rec["tiles"][str(tile)]
            if new not in TRANSITIONS[entry["state"]]:
                raise LedgerError(f"tile {tile}: {entry['state']} -> {new} is not allowed")
            if entry["state"] == "fresh":
                entry["released_at"] = utc_now()
            entry["state"] = new
            if image_crc is not None:
                entry["image_crc"] = int(image_crc) & 0xFFFF_FFFF
            self._write(rec)
            return rec

    def mark_released(self, tile: int, image_crc: Optional[int]) -> dict:
        """Called BEFORE the L2CPU_RESET write: an attempted release spends the tile's one release."""
        return self.transition(tile, "released", image_crc=image_crc)

    def mark_parked(self, tile: int) -> dict:
        return self.transition(tile, "parked")

    def mark_resumed(self, tile: int) -> dict:
        return self.transition(tile, "released")

    def mark_burned(self, tile: int) -> dict:
        return self.transition(tile, "burned")

    def burned_count(self) -> int:
        """PLAN §5.2 tile policy: >= 2 burned tiles in one session means stop and review with the user."""
        return sum(v["state"] == "burned" for v in self._current()["tiles"].values())
