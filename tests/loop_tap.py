"""T-1.3 loop arm: the rtloop.loop CLI, in-process, with a passive float64 recorder on env.step.

Run by tests/test_official_equivalence.py in its own simpy subprocess:

    python tests/loop_tap.py --tap-out D/loop_tap.npz -- --mode sync --vla URL --wbc ort ...

Everything after ``--`` goes to ``rtloop.loop.main`` unchanged, so the run is the CLI's. The TickLog
keeps q in float32 only; the official arm records the env's float64 observation, so this tap does
the same on the loop's side to make the T-1.3 comparison exact at full precision. It replaces
``rtloop.sim_env.make_env`` (looked up at call time by the loop) with one that returns the same env
whose ``step``/``reset`` instance attributes copy q, dq, floating_base_pose and floating_base_vel
after each step and note each reset's seed. No value the loop or env computes is touched.
The gym.make spec kwargs and the WBC script config go to the ``.json`` next to the npz (env
identity, PLAN.md T-1.3 "rerun with byte-identical SessionOptions and env").
"""

import json
from pathlib import Path
import sys

import numpy as np

KEYS = (("q", "q"), ("dq", "dq"), ("base_pose", "floating_base_pose"), ("base_vel", "floating_base_vel"))


def env_identity(env) -> dict:
    """gym.make kwargs and the WholeBodyControlWrapper script config, as plain JSON (official_arm too)."""
    spec = getattr(env.env, "spec", None)
    return {"env_id": getattr(spec, "id", None),
            "make_kwargs": json.loads(json.dumps(dict(getattr(spec, "kwargs", {}) or {}), default=str)),
            "script_config": json.loads(json.dumps(env.script_config, default=str)),
            "wrapper": type(env).__module__ + "." + type(env).__qualname__}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv or argv[:1] != ["--tap-out"]:
        raise SystemExit("usage: loop_tap.py --tap-out OUT.npz -- <rtloop.loop args>")
    out = Path(argv[1])
    loop_args = argv[argv.index("--") + 1:]

    from rtloop import loop, sim_env

    rows = {k: [] for k, _ in KEYS}
    resets: list[list] = []  # [step index at reset, seed]
    ident: dict = {}
    stock_make_env = sim_env.make_env

    def make_env(*a, **kw):
        env = stock_make_env(*a, **kw)
        ident.update(env_identity(env))
        step, reset = env.step, env.reset

        def tap_step(action):
            res = step(action)
            obs = res[0]
            for k, src in KEYS:
                rows[k].append(np.array(obs[src], dtype=np.float64, copy=True))
            return res

        def tap_reset(**kwargs):
            seed = kwargs.get("seed")
            resets.append([len(rows["q"]), -1 if seed is None else int(seed)])
            return reset(**kwargs)

        env.step, env.reset = tap_step, tap_reset
        return env

    sim_env.make_env = make_env
    rc = None
    try:
        rc = loop.main(["rtloop.loop", *loop_args])
        return rc
    finally:
        out.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items() if v}
        arrays["resets"] = np.asarray(resets, dtype=np.int64).reshape(-1, 2)
        np.savez(out, **arrays)
        out.with_suffix(".json").write_text(json.dumps({"env": ident, "rc": rc,
                                                        "n_steps": len(rows["q"])}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
