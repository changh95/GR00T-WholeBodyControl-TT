"""env.sh's ttpy (verify:all D7, S1 2026-10-25): every MK process it runs gets TT_METAL_RUNTIME_ROOT=$MK unless the
caller set one, so tt-metal compiles MK's own kernels, and rttt/g1_n16.pin_runtime_root accepts that root. CPU only:
ttnn is blocked inside the child, so nothing here can open the device."""

import json
import os
from pathlib import Path
import subprocess

import pytest

RT = Path(__file__).resolve().parents[1]
MK_ENV = Path.home() / "experiments/gr00t/publish/worktree-mk-env.sh"
MK = Path.home() / "experiments/gr00t/publish/worktree-mk"

pytestmark = pytest.mark.skipif(not MK_ENV.exists(), reason=f"{MK_ENV} (the MK snapshot env) is not here")

BLOCK_TTNN = """
import json, os, sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "ttnn" or name.startswith("ttnn."):
            raise ImportError("ttnn blocked (CPU test)")
sys.meta_path.insert(0, Block())
"""


def ttpy(code: str, preset: str | None = None) -> subprocess.CompletedProcess:
    """`source env.sh; ttpy -c CODE` in a clean bash, with TT_METAL_RUNTIME_ROOT unset or preset."""
    env = {k: v for k, v in os.environ.items() if k != "TT_METAL_RUNTIME_ROOT"}
    if preset is not None:
        env["TT_METAL_RUNTIME_ROOT"] = preset
    script = f'source {RT}/env.sh; ttpy -c "$1"; rc=$?; echo "caller=${{TT_METAL_RUNTIME_ROOT:-unset}}"; exit $rc'
    return subprocess.run(["bash", "-c", script, "_", code], env=env, capture_output=True, text=True, timeout=120)


def test_ttpy_sets_the_mk_runtime_root_only_if_unset():
    p = ttpy("import os; print(os.environ.get('TT_METAL_RUNTIME_ROOT'))")
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.splitlines() == [str(MK), "caller=unset"]  # ttpy's subshell: the caller's env is unchanged
    p = ttpy("import os; print(os.environ.get('TT_METAL_RUNTIME_ROOT'))", preset="/some/other/root")
    assert p.stdout.splitlines() == ["/some/other/root", "caller=/some/other/root"]  # a caller's choice is kept


def test_pin_runtime_root_accepts_the_ttpy_root():
    code = BLOCK_TTNN + ("from rttt import g1_n16 as G\n"
                         "print(json.dumps({**G.pin_runtime_root(), 'ttnn': 'ttnn' in sys.modules}))")
    p = ttpy(code)
    assert p.returncode == 0, p.stderr[-2000:]
    got = json.loads(p.stdout.splitlines()[0])
    assert got == {"runtime_root": str(MK), "preset": True, "ttnn": False}
    p = ttpy(code, preset=str(Path.home() / "experiments/gr00t/tt-metal"))  # the live tree: refused
    assert p.returncode != 0 and "kernels would come from another tree" in p.stderr
