"""Stand-in for rtloop.loop in the demo.sh live test: no sim, no GPU, no fixed ports.

It accepts the CLI demo.sh builds ("--pub", "--control", "--vla-alt", "--wbc-alt", "--swap-wbc-every", plus
anything else), prints the loop's "rt_sim: run dir" line, publishes the synthetic stream at 50 Hz on --pub,
and applies control commands exactly as the requested loop integration does: ControlPull polled once per
tick at the boundary, control_proto.apply on the registered targets, a WBC swap from the next tick, a VLA
swap queued for the reset window, reset ends the episode, pause holds between ticks. Every applied action
goes to <run>/control.jsonl so the test can check it. SIGINT/SIGTERM stop after the tick (exit 130).
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "scripts")]

from rtloop import control as cp  # noqa: E402
from rtloop import telemetry  # noqa: E402
from synth_pub import META, SynthStream, publish  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub", default=None)
    ap.add_argument("--control", default=None)
    ap.add_argument("--vla", default="")
    ap.add_argument("--wbc", default="ort")
    ap.add_argument("--vla-alt", default="")
    ap.add_argument("--wbc-alt", default="")
    ap.add_argument("--swap-wbc-every", type=float, default=None)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--tag", default="stub")
    ap.add_argument("--profile", default="m1")
    args, _unknown = ap.parse_known_args()

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    run = telemetry.RunDir(tag=args.tag, config={"argv": sys.argv})
    print(f"rt_sim: run dir {run.path}", flush=True)
    pub = telemetry.Publisher(args.pub)
    ctl = cp.ControlPull(args.control) if args.control else None
    vla_names = ["primary", *cp.parse_targets(args.vla_alt)]
    wbc_names = ["primary", "ort", *cp.parse_targets(args.wbc_alt)]
    vla_cur, wbc_cur = "primary", "primary"
    state = cp.ControlState()
    log = open(run.path / "control.jsonl", "a")
    stream = SynthStream(ticks_per_ep=500, profile=args.profile)
    t_end = time.monotonic() + args.duration
    n_ticks, ep = 0, 0
    t_next = time.monotonic()
    while not stop["flag"] and time.monotonic() < t_end:
        if state.pending_vla:  # reset window: the queued VLA swap takes effect, gen bumps
            vla_cur, state = state.pending_vla, cp.vla_swapped(state)
            log.write(json.dumps({"ep": ep, "applied": "vla", "to": vla_cur}) + "\n")
        end_ep = False
        for item in stream.episode(ep):
            if item[0] == "meta":
                item = ("meta", {**item[1], "processor": META[args.profile],
                                 "current": {"vla": vla_cur, "wbc": wbc_cur},
                                 "pending_vla": state.pending_vla, "paused": state.paused,
                                 "control": ctl.url if ctl else None})
            publish(pub, item)
            if item[0] != "tick":
                continue
            n_ticks += 1
            # --- tick boundary: the requested loop integration, verbatim in shape
            cmds = ctl.poll() if ctl else []
            if cp.auto_swap_due(n_ticks, args.swap_wbc_every):
                cmds.append(cp.Command("swap_wbc", cp.NEXT))
            for c in cmds:
                if isinstance(c, cp.ControlError):
                    state = cp.reject(state, c)
                    log.write(json.dumps({"rejected": str(c)}) + "\n")
                    continue
                state, act = cp.apply(state, c, vla_names, vla_cur, wbc_names, wbc_cur)
                if act.kind == "swap_wbc":
                    wbc_cur = act.to
                elif act.kind == "reset":
                    end_ep = True
                log.write(json.dumps({"ep": ep, "k": item[1]["k"], "cmd": c.cmd, "to": c.to,
                                      "action": act.kind, "arg": act.to}) + "\n")
            log.flush()
            while state.paused and not stop["flag"] and time.monotonic() < t_end:
                time.sleep(0.02)
                for c in (ctl.poll() if ctl else []):
                    if isinstance(c, cp.Command):
                        state, act = cp.apply(state, c, vla_names, vla_cur, wbc_names, wbc_cur)
                        log.write(json.dumps({"ep": ep, "cmd": c.cmd, "action": act.kind}) + "\n")
                        end_ep |= act.kind == "reset"
                t_next = time.monotonic()
            t_next += 0.02
            delay = t_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if end_ep or stop["flag"] or time.monotonic() >= t_end:
                break
        ep += 1
    log.close()
    run.write_json("summary.json", {"ticks": n_ticks, "episodes": ep, "vla": vla_cur, "wbc": wbc_cur,
                                    "control": {"applied": state.n_applied, "rejected": state.n_rejected}})
    if ctl:
        ctl.close()
    pub.close()
    print(f"rt_sim: stub done, {n_ticks} ticks", flush=True)
    return 130 if stop["flag"] else 0


if __name__ == "__main__":
    sys.exit(main())
