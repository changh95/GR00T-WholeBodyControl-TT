"""Pick the 20 golden observations (WP-2.2) from a sync-mode loop run that dumped every request.

10 "reset" samples are the pre-roll request (k_obs = 0) of the first 10 episodes, so each comes
from a different scene seed. 10 "mid" samples take one request per episode, spread over episode
time (target fraction 0.1 ... 0.9 of that episode's length, +-WINDOW_FRAC) and chosen while the
robot moves: even episodes prefer walking (the Walk policy on most ticks of the chunk that answered
the request and a nav command of at least WALK_NAV), odd episodes prefer manipulating (arm command
travel or hand command change over that chunk), each falling back to the other kind and then to
the nearest request.

    simpy -m rttt.select_g1_obs --run $RT/runs/wp22/<run> [--out ~/experiments/gr00t/golden/n16_g1]

Writes obs/<sid>.npz (W-VLA keys, golden_io.save_obs) and obs_index.json.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from rtloop import schema, telemetry
from rttt import golden_io as gio

WALK_NAV = 0.15  # m/s or rad/s; the Walk policy already engages at 0.05 (schema.WALK_THRESHOLD)
ARM_TRAVEL_RAD = 0.10  # ||arm cmd(last row) - arm cmd(first row)|| over one 20-row chunk
HAND_CHANGE_RAD = 0.20  # max |hand cmd change| over one chunk (grasp / release)
WINDOW_FRAC = 0.15  # mid samples stay within +-15% of the episode length of their target time


def chunk_motion(ticks: np.ndarray, ep: int, k_obs: int, n_rows: int) -> dict:
    """What the robot did while executing the chunk that answered the request at k_obs."""
    t = ticks[(ticks["ep"] == ep) & (ticks["k"] >= k_obs) & (ticks["k"] < k_obs + n_rows)]
    if len(t) < 2:
        return {"walk": False, "manip": False, "nav_max": 0.0, "walk_frac": 0.0, "arm_travel": 0.0,
                "hand_change": 0.0}
    cmd = t["cmd35"].astype(np.float64)
    nav_max = float(np.abs(cmd[:, schema.NAV]).max())
    arm_travel = float(np.linalg.norm(cmd[-1, schema.ARMS] - cmd[0, schema.ARMS]))
    hand_change = float(np.abs(cmd[:, schema.HANDS] - cmd[0, schema.HANDS]).max())
    walk_frac = float(t["policy"].mean())  # 1 = Walk
    walk = bool(walk_frac > 0.5 and nav_max >= WALK_NAV)
    manip = bool(arm_travel > ARM_TRAVEL_RAD or hand_change > HAND_CHANGE_RAD)
    return {"walk": walk, "manip": manip, "nav_max": nav_max, "walk_frac": walk_frac,
            "arm_travel": arm_travel, "hand_change": hand_change}


def pick_mid(cands: list[dict], target_k: float, window: float, prefer: str) -> tuple[dict, str]:
    """Nearest request of the preferred kind within +-window ticks of the target, else of the other
    kind, else the nearest request (kept spread over time rather than clustered on one phase)."""
    near = [c for c in cands if abs(c["k_obs"] - target_k) <= window]
    for kind in (prefer, "manip" if prefer == "walk" else "walk"):
        ok = [c for c in near if c["motion"][kind]]
        if ok:
            return min(ok, key=lambda c: abs(c["k_obs"] - target_k)), kind
    return min(cands, key=lambda c: abs(c["k_obs"] - target_k)), "idle"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="loop run dir with vla_obs/ (sync mode, --dump-vla-obs)")
    ap.add_argument("--out", default=str(gio.GOLDEN_DIR))
    ap.add_argument("--n", type=int, default=10, help="episodes; one reset and one mid sample each")
    args = ap.parse_args(argv)

    run, out = Path(args.run).resolve(), Path(args.out).expanduser()
    cfg = json.loads((run / "config.json").read_text())
    n_rows = int(cfg["args"]["sync_rows"])
    assert cfg["args"]["mode"] == "sync", "goldens come from a sync-mode run (one obs per chunk)"
    req = telemetry.load_requests(run)
    ticks = telemetry.load_ticks(run)
    eps = {e["ep"]: e for e in telemetry.load_episodes(run)}
    assert len(eps) >= args.n, f"{len(eps)} finished episodes < {args.n}"

    (out / "obs").mkdir(parents=True, exist_ok=True)
    samples = []

    def add(sid: str, kind: str, r: np.void, ep: dict, motion: dict, extra: dict) -> None:
        seq = int(r["seq"])
        src = run / "vla_obs" / f"{seq:06d}.npz"
        obs = gio.load_loop_dump(src)
        sha1 = schema.obs_sha1(obs)
        assert np.array_equal(sha1, r["obs_sha1"]), f"{src}: sha1 differs from the RequestLog"
        gio.save_obs(out / "obs" / f"{sid}.npz", obs)
        samples.append({"sid": sid, "kind": kind, "run": str(run), "src": str(src), "seq": seq,
                        "ep": int(r["ep"]), "seed": int(ep["seed"]), "k_obs": int(r["k_obs"]),
                        "k_end": int(ep["k_end"]), "end": ep["end"],
                        "frac": round(int(r["k_obs"]) / max(1, int(ep["k_end"])), 3),
                        "noise_seed_loop": int(r["noise_seed"]), "obs_sha1": sha1.tobytes().hex(),
                        "instruction": obs[schema.LANGUAGE_KEY][0],
                        "state43": [float(x) for x in schema.state43(obs)],
                        "motion": motion, **extra})

    ep_ids = sorted(eps)[: args.n]
    for i, e in enumerate(ep_ids):
        r = req[(req["ep"] == e)]
        r0 = r[np.argmin(r["k_obs"])]
        assert int(r0["k_obs"]) == 0
        add(f"s{i:02d}", "reset", r0, eps[e], chunk_motion(ticks, e, 0, n_rows), {})
    for j, e in enumerate(ep_ids):
        k_end = int(eps[e]["k_end"])
        r = req[(req["ep"] == e) & (req["k_obs"] >= n_rows) & (req["k_obs"] <= k_end - n_rows)]
        cands = [{"r": x, "k_obs": int(x["k_obs"]), "motion": chunk_motion(ticks, e, int(x["k_obs"]), n_rows)}
                 for x in r]
        target = (0.1 + 0.8 * j / max(1, args.n - 1)) * k_end
        prefer = "walk" if j % 2 == 0 else "manip"
        c, got = pick_mid(cands, target, WINDOW_FRAC * k_end, prefer)
        add(f"s{args.n + j:02d}", "mid", c["r"], eps[e], c["motion"],
            {"target_k": round(target, 1), "window": round(WINDOW_FRAC * k_end, 1), "prefer": prefer,
             "picked_as": got})

    index = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source_run": str(run),
             "source_args": cfg["args"], "vla_info": cfg.get("vla_info"),
             "rule": __doc__.split("\n\n")[1].replace("\n", " "),
             "thresholds": {"walk_nav": WALK_NAV, "walk_policy_frac": 0.5, "arm_travel_rad": ARM_TRAVEL_RAD,
                            "hand_change_rad": HAND_CHANGE_RAD, "window_frac": WINDOW_FRAC,
                            "chunk_rows": n_rows},
             "samples": samples}
    (out / "obs_index.json").write_text(json.dumps(index, indent=1))
    for s in samples:
        m = s["motion"]
        print(f"{s['sid']} {s['kind']:5s} seed {s['seed']} k_obs {s['k_obs']:4d}/{s['k_end']} "
              f"{s.get('picked_as', ''):5s} nav {m['nav_max']:.2f} walk {m['walk_frac']:.2f} arm {m['arm_travel']:.2f} "
              f"hand {m['hand_change']:.2f}")
    print(f"wrote {len(samples)} obs to {out / 'obs'} and {out / 'obs_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
