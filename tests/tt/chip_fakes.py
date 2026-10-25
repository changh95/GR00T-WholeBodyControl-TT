"""Stubs of every device command chip_session.sh runs, for tests/tt/test_chip_session_fake.py (never touch a device).

    python chip_fakes.py x280d probe --json | x280d serve --port P ...
    python chip_fakes.py tt-smi -r 0 | tt-probe | fuser NODE

Behaviour comes from files in $FAKE_DIR, read at each call, so one test can change it between steps:
  x280_state   clean | dirty | dirty_until_reset | hang | hang_until_reset | error | exit3 | attachable   (probe)
  x280_serve   ok | exit3 | die | ignore_term                                       (serve)
  ttsmi_rc, tt_probe_rc   exit codes (default 0); fuser_pids   pids printed by fuser (default none)
Every call appends "<monotonic> <what>" to $FAKE_DIR/events.log; tt-smi also creates $FAKE_DIR/reset_done.
"""

import json
import os
from pathlib import Path
import signal
import sys
import time

DIR = Path(os.environ["FAKE_DIR"])


def read(name: str, default: str = "") -> str:
    p = DIR / name
    return p.read_text().strip() if p.exists() else default


def event(msg: str) -> None:
    with open(DIR / "events.log", "a") as f:
        f.write(f"{time.monotonic():.6f} {msg}\n")


def probe() -> int:
    state = read("x280_state", "clean")
    if state in ("dirty_until_reset", "hang_until_reset"):
        state = "clean" if (DIR / "reset_done").exists() else state.split("_")[0]
    event(f"x280d probe {state}")
    if state == "hang":
        time.sleep(3600)
    if state == "error":
        print("Traceback: fake x280d crashed", file=sys.stderr)
        return 1
    if state == "exit3":
        return 3
    rep = {"dirty": state in ("dirty", "attachable"), "reasons": [f"fake {state}"] if state != "clean" else [],
           "l2cpu_reset": "0x8f" if state == "attachable" else "0x0f"}
    if state == "attachable":
        rep["attachable"] = True
    print("x280d: probe done (log line)")
    print(json.dumps(rep))
    return 0


def serve(argv: list[str]) -> int:
    import msgpack
    import zmq

    port = int(argv[argv.index("--port") + 1])
    assert not 5555 <= port <= 5572, "tests never bind the reserved ports"
    mode = read("x280_serve", "ok")
    event(f"x280d serve start {' '.join(argv)}")
    if mode == "exit3":
        event("x280d serve exit3")
        return 3
    if mode == "die":
        event("x280d serve died")
        return 1
    stop = {"flag": False}
    if mode == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{port}")
    event("x280d serve ready")
    while not stop["flag"]:
        if sock.poll(100):
            req = msgpack.unpackb(sock.recv(), raw=False)
            ep = req.get("endpoint") if isinstance(req, dict) else None
            sock.send(msgpack.packb({"status": "ok"} if ep == "ping" else {"error": f"fake x280d: {ep}"}))
    event("x280d serve stop")
    sock.close(0)
    ctx.term()
    return 0


def main(argv: list[str]) -> int:
    what, rest = argv[0], argv[1:]
    if what == "x280d":
        return probe() if rest[:1] == ["probe"] else serve(rest)
    if what == "tt-smi":
        event(f"tt-smi {' '.join(rest)}")
        (DIR / "reset_done").write_text("1")
        return int(read("ttsmi_rc", "0"))
    if what == "tt-probe":
        event("tt-probe")
        return int(read("tt_probe_rc", "0"))
    if what == "fuser":
        event(f"fuser {' '.join(rest)}")
        pids = read("fuser_pids")
        if pids:
            print(pids)
        print(f"{rest[-1]}:", file=sys.stderr)
        return 0 if pids else 1
    raise SystemExit(f"chip_fakes: unknown command {what}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
