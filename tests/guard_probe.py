"""The rtloop.loop CLI, in-process, with passive recorders for the fault, CUDA-guard and request tests.

Run by tests/test_faults.py, tests/test_cuda_guard.py and tests/test_obs_schema.py in their own
simpy subprocess (through fault_utils.ProbeRun):

    python tests/guard_probe.py --out D [--record-obs] [--frames-at 0,30,59] [--trace-splicer]
                                [--import-torch] [--allow-cuda-visible] -- <rtloop.loop args>

Everything after ``--`` goes to ``rtloop.loop.main`` unchanged, so the run is the CLI's. The hooks
replace module attributes the loop looks up at call time and change no value it computes:

  always            ``sim_env.assert_cuda_free`` is wrapped: every call the loop makes (start, after
                    imports, after each reset) is recorded with CUDA_VISIBLE_DEVICES, whether torch
                    is imported and whether it initialised CUDA; then the original runs. At exit
                    (after loop.close) the same is recorded once more, plus this pid's presence in
                    ``nvidia-smi --query-compute-apps=pid`` while the process is still alive.
  --record-obs      ``sim_env.make_env`` returns the same env whose ``reset``/``step`` instance
                    attributes copy the observation the loop is about to act on: per (episode, k)
                    the 7 ``state.*`` arrays and ``q`` in float64, the sha1, mean and max of the ego
                    frame, and the instruction. Entry k is o_k, the observation at the start of tick k
                    (the reset obs for k = 0, else what step k-1 returned).
  --frames-at K,..  also keep the full ego frames of ticks K of every episode.
  --trace-splicer   ``splicer.select`` is wrapped: after every call, (ep, k, row, starved,
                    starve_n, starve_fault) of the splicer state (the flag is not in the TickLog).
  --import-torch    ``import torch`` before anything else (T-1.7's "visible but unused" arm).
  --allow-cuda-visible  the recorder skips only the original's CUDA_VISIBLE_DEVICES == "" check
                    (GR00T_WBC_SERVER unset and torch CUDA uninitialised are still enforced), so the
                    loop can run with CUDA visible for the T-1.7 frame comparison.

Outputs in D: probe.json, and obs.npz / splicer.npz when asked for.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


def cuda_state(where: str) -> dict:
    torch = sys.modules.get("torch")
    return {"where": where, "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_imported": torch is not None,
            "torch_cuda_init": bool(torch is not None and torch.cuda.is_initialized()),
            "gr00t_wbc_server": os.environ.get("GR00T_WBC_SERVER")}


def compute_pids() -> list[int] | None:
    """PIDs with a CUDA context (EGL contexts are graphics-type and not listed); None on failure."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return sorted(int(x) for x in out.split() if x.strip().isdigit())


class ObsRecorder:
    """Copies of o_k for every (episode, k) the loop acts on; small arrays only, plus chosen frames."""

    def __init__(self, frames_at: set[int]):
        from rtloop.schema import STATE_KEYS

        self.state_keys = STATE_KEYS
        self.frames_at = frames_at
        self.ep = -1
        self.k = 0
        self.rows = {"ep": [], "k": [], "state43": [], "q": [], "img_sha1": [], "img_mean": [],
                     "img_max": []}
        self.lang: dict[int, str] = {}
        self.frames: list[np.ndarray] = []
        self.frame_idx: list[tuple[int, int]] = []

    def add(self, obs: dict) -> None:
        from rtloop.schema import VIDEO_KEY, LANGUAGE_KEY

        img = np.ascontiguousarray(obs[VIDEO_KEY])
        r = self.rows
        r["ep"].append(self.ep)
        r["k"].append(self.k)
        r["state43"].append(np.concatenate([np.array(obs[k], dtype=np.float64).reshape(-1)
                                            for k in self.state_keys]))
        r["q"].append(np.array(obs["q"], dtype=np.float64, copy=True))
        r["img_sha1"].append(np.frombuffer(hashlib.sha1(img.tobytes()).digest(), dtype=np.uint8))
        r["img_mean"].append(float(img.mean()))
        r["img_max"].append(int(img.max()))
        self.lang.setdefault(self.ep, str(obs[LANGUAGE_KEY]))
        if self.k in self.frames_at:
            self.frames.append(img.copy())
            self.frame_idx.append((self.ep, self.k))
        self.k += 1

    def wrap(self, env) -> None:
        step, reset = env.step, env.reset

        def rec_reset(**kwargs):
            res = reset(**kwargs)
            self.ep += 1
            self.k = 0
            self.add(res[0])
            return res

        def rec_step(action):
            res = step(action)  # a SystemExit from the safety monitor propagates unrecorded
            self.add(res[0])
            return res

        env.reset, env.step = rec_reset, rec_step

    def save(self, path: Path) -> None:
        r = self.rows
        arrays = {
            "ep": np.asarray(r["ep"], dtype=np.int64), "k": np.asarray(r["k"], dtype=np.int64),
            "state43": np.asarray(r["state43"], dtype=np.float64).reshape(-1, 43),
            "q": np.asarray(r["q"], dtype=np.float64).reshape(-1, 43),
            "img_sha1": np.asarray(r["img_sha1"], dtype=np.uint8).reshape(-1, 20),
            "img_mean": np.asarray(r["img_mean"], dtype=np.float64),
            "img_max": np.asarray(r["img_max"], dtype=np.int64),
            "lang_ep": np.asarray(sorted(self.lang), dtype=np.int64),
            "lang": np.asarray([self.lang[e] for e in sorted(self.lang)]),
            "frame_idx": np.asarray(self.frame_idx, dtype=np.int64).reshape(-1, 2),
        }
        if self.frames:
            arrays["frames"] = np.stack(self.frames)
        np.savez(path, **arrays)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: guard_probe.py --out D [options] -- <rtloop.loop args>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser(prog="guard_probe.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-obs", action="store_true")
    ap.add_argument("--frames-at", default="")
    ap.add_argument("--trace-splicer", action="store_true")
    ap.add_argument("--import-torch", action="store_true")
    ap.add_argument("--allow-cuda-visible", action="store_true")
    a = ap.parse_args(argv[:cut])
    loop_args = argv[cut + 1:]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    checks = [cuda_state("probe: start")]
    if a.import_torch:
        import torch  # noqa: F401 - imported, never asked about devices

        checks.append(cuda_state("probe: after import torch"))

    from rtloop import loop, sim_env
    from rtloop import splicer as sp

    stock_assert = sim_env.assert_cuda_free

    def assert_cuda_free(where: str) -> None:
        checks.append(cuda_state(where))
        if not a.allow_cuda_visible:
            return stock_assert(where)
        if "GR00T_WBC_SERVER" in os.environ:
            raise RuntimeError(f"{where}: GR00T_WBC_SERVER is set")
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            raise RuntimeError(f"{where}: torch initialised CUDA in rt_sim")
        return None

    sim_env.assert_cuda_free = assert_cuda_free

    rec = None
    if a.record_obs:
        frames_at = {int(x) for x in a.frames_at.split(",") if x.strip()}
        rec = ObsRecorder(frames_at)
        stock_make_env = sim_env.make_env

        def make_env(*args, **kw):
            env = stock_make_env(*args, **kw)
            rec.wrap(env)
            return env

        sim_env.make_env = make_env

    trace = None
    if a.trace_splicer:
        trace = []
        stock_select = sp.select
        ep_of_gen: dict[int, int] = {}

        def select(k, st, cfg, hold):
            cmd, info = stock_select(k, st, cfg, hold)
            ep = ep_of_gen.setdefault(st.gen, len(ep_of_gen))  # gen is bumped once per episode
            trace.append((ep, k, info["row"], info["starved"], st.starve_n, int(st.starve_fault)))
            return cmd, info

        sp.select = select

    rc, err = None, None
    try:
        rc = loop.main(["rtloop.loop", *loop_args])
    except BaseException as e:  # noqa: BLE001 - recorded, then re-raised
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise
    finally:
        exit_state = cuda_state("probe: exit")
        pids = compute_pids()
        exit_state.update(nvidia_compute_pids=pids, pid_listed=None if pids is None else os.getpid() in pids)
        if rec is not None:
            rec.save(out / "obs.npz")
        if trace is not None:
            np.savez(out / "splicer.npz", trace=np.asarray(trace, dtype=np.int64).reshape(-1, 6))
        (out / "probe.json").write_text(json.dumps(
            {"rc": rc, "pid": os.getpid(), "checks": checks, "exit": exit_state, "error": err,
             "argv": argv}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
