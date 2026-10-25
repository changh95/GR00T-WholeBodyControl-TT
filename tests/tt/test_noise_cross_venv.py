"""verify2 N6: gpu_vla_server --noise cpu draws server_g1's noise across the two real interpreters (PLAN.md D17).

test_tt_server_protocol.py::test_gpu_vla_server_cpu_noise_is_server_g1s_draw runs both sides in one interpreter
(ttpy, MK's torch), but in use gpu_vla_server runs under srvpy (the GPU venv's torch). A torch change in either venv
that moved the CPU randn stream would silently unpair the --crn TT-vs-GPU arms (T-2.5, T-2.7) while the session probe
(jobs/_common.sh check_gpu_vla_revision: info.noise and randn_calls only) still passed. So each side draws in its own
venv, through env.sh's wrappers, and the bytes are compared here:

- srvpy, CUDA hidden (CUDA_VISIBLE_DEVICES=): RtPolicyServer.get_action with its real seeding (torch.manual_seed,
  torch.cuda.manual_seed, SeededNoise) around a stand-in policy that makes the N1.6 action head's one call
  (gr00t_n1d6.py: torch.randn(size=(B, action_horizon, action_dim), dtype=<model dtype>, device=...), bf16 here),
  plus SeededNoise's fp32 draw.
- ttpy, ttnn blocked: server_g1's G1Policy.predict on the fake model, the noise captured at predict_normalized, and
  rttt/g1_n16.make_noise.

    simpy -m pytest $RT/tests/tt/test_noise_cross_venv.py -q    (about 15 s; skips when either venv is missing)

No server binds a port and no device or GPU is touched.
"""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

RT = Path(__file__).resolve().parents[2]
ENV_SH = RT / "env.sh"
SRV_PYTHON = Path.home() / "experiments/gr00t/Isaac-GR00T-n1.6/.venv/bin/python"  # env.sh SRV_PYTHON
MK_ENV = Path.home() / "experiments/gr00t/publish/worktree-mk-env.sh"  # what env.sh's ttpy sources
SHAPE = (1, 50, 128)  # (B, action_horizon, max_action_dim) of DP1, as server_g1's noise_shape

GPU_SIDE = r'''
import json, sys, types
import numpy as np, torch
from rtserve import gpu_vla_server as gv
seeds, out, shape = json.loads(sys.argv[1]), sys.argv[2], tuple(json.loads(sys.argv[3]))

class HeadOnly:  # the action head's one draw (on cpu: CUDA is hidden); the server's get_action seeds around it
    def get_action(self, observation, options=None):
        self.noise = torch.randn(size=shape, dtype=torch.bfloat16, device="cpu")
        return {}, {}

srv = types.SimpleNamespace(t_recv_ns=0, n_stale=0, noise="cpu", policy=HeadOnly(), backend="cpu:test", log=None)
bf16, fp32, calls = [], [], []
for s in seeds:
    _, info = gv.RtPolicyServer.get_action(srv, {}, {"rt": {"seq": 0, "noise_seed": s}})
    assert info["noise"] == "cpu" and info["noise_seed"] == s, info
    bf16.append(srv.policy.noise.view(torch.int16).numpy())  # the bf16 bits
    calls.append(info["randn_calls"])
    with gv.SeededNoise(s):
        fp32.append(torch.randn(shape).numpy())
np.savez(out, fp32=np.stack(fp32), bf16=np.stack(bf16), calls=np.array(calls))
print(json.dumps({"torch": torch.__version__, "cuda_initialized": torch.cuda.is_initialized()}))
'''

TT_SIDE = r'''
import importlib.abc, json, sys
class NoTtnn(importlib.abc.MetaPathFinder):  # as tests/tt/conftest.py: the device must not be opened
    def find_spec(self, name, path=None, target=None):
        if name == "ttnn" or name.startswith("ttnn."):
            raise ImportError("ttnn is blocked in the CPU TT tests")
sys.meta_path.insert(0, NoTtnn())
import numpy as np, torch
from rttt import g1_n16 as G, server_g1 as S
seeds, out = json.loads(sys.argv[1]), sys.argv[2]
G.install()
pol = S.G1Policy(S.FakeG1Model(), S.fake_stamp(), 0, fake=True)
seen, orig = [], pol.model.predict_normalized
pol.model.predict_normalized = lambda mi, noise: (seen.append(noise.detach().clone()), orig(mi, noise))[1]
draw, bf16, mk = [], [], []
for s in seeds:
    seen.clear()
    pol.predict(S.warm_obs(), s)  # server_g1's own path: reseed its generator, one draw per element
    (n,) = seen
    assert n.dtype == torch.float32, n.dtype
    draw.append(n.numpy())
    bf16.append(n.to(torch.bfloat16).view(torch.int16).numpy())  # the cast the GPU action head's dtype makes
    mk.append(G.make_noise(s).numpy())
np.savez(out, fp32=np.stack(draw), bf16=np.stack(bf16), make_noise=np.stack(mk))
print(json.dumps({"torch": torch.__version__, "ttnn_imported": "ttnn" in sys.modules,
                  "noise_shape": list(pol.noise_shape)}))
'''


def _seeds() -> list[int]:
    """The loop's real --crn noise_seeds (splitmix64 of seed << 20 | request index) and the u64 edge values."""
    noise_seed = pytest.importorskip("rtloop.loop", reason="rtloop.loop needs the sim venv (simpy)").noise_seed

    return [noise_seed(2000, k) for k in range(5)] + [noise_seed(1049, 71), 0, 1, 2**63, 2**64 - 1]


def _run(wrapper: str, script: str, *args: str, env: dict | None = None) -> dict:
    """``wrapper - ARGS`` (env.sh's srvpy / ttpy) reading ``script`` from stdin; its last stdout line as JSON."""
    base = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # each wrapper sets its own (simpy's has $R)
    r = subprocess.run(["bash", "-c", f'source "{ENV_SH}" && {wrapper} - "$@"', "_", *args], input=script,
                       env={**base, **(env or {})}, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{wrapper}: rc {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not SRV_PYTHON.exists(), reason=f"no GPU-server venv ({SRV_PYTHON})")
@pytest.mark.skipif(not MK_ENV.exists(), reason=f"no MK env ({MK_ENV}) for ttpy")
def test_gpu_vla_server_noise_equals_server_g1s_across_venvs(tmp_path):
    seeds = _seeds()
    gpu_npz, tt_npz = tmp_path / "gpu.npz", tmp_path / "tt.npz"
    gpu = _run("srvpy", GPU_SIDE, json.dumps(seeds), str(gpu_npz), json.dumps(SHAPE), env={"CUDA_VISIBLE_DEVICES": ""})
    tt = _run("ttpy", TT_SIDE, json.dumps(seeds), str(tt_npz))
    assert gpu["cuda_initialized"] is False and tt["ttnn_imported"] is False and tuple(tt["noise_shape"]) == SHAPE
    g, t = np.load(gpu_npz), np.load(tt_npz)
    assert g["fp32"].dtype == t["fp32"].dtype == np.float32 and g["fp32"].shape == t["fp32"].shape == (len(seeds), *SHAPE)
    assert (g["calls"] == 1).all(), g["calls"]  # the head's draw was the seeded one, and the only one
    for i, s in enumerate(seeds):
        why = f"seed {s}: srvpy torch {gpu['torch']} vs ttpy torch {tt['torch']}"
        assert g["fp32"][i].tobytes() == t["fp32"][i].tobytes(), why  # fp32, byte for byte
        assert g["bf16"][i].tobytes() == t["bf16"][i].tobytes(), why  # and after the cast to the head's bf16
        assert t["make_noise"][i].tobytes() == t["fp32"][i].tobytes(), why  # warm-up's make_noise is the same draw
    # A seed picks its own draw, up to torch's CPU generator (mt19937) keeping only a seed's low 32 bits: 0 and 2**63
    # draw the same noise, in both venvs alike, so pairing holds and two of the loop's u64 noise_seeds collide with
    # probability 2**-32.
    low = {s & 0xFFFFFFFF for s in seeds}
    assert len({g["fp32"][i].tobytes() for i in range(len(seeds))}) == len(low) == len(seeds) - 1
