"""Shared fixtures of the CPU-only TT tests (T-2.1): ``ttpy -m pytest $RT/tests/tt/<file> -q``.

These tests must never open the Blackhole, so ``import ttnn`` is made to fail for the whole session: any code path
that would reach the device raises ImportError instead (the real-model path is exercised in device session S1).
"""

import importlib.abc
import sys
from pathlib import Path

import numpy as np
import pytest

RESERVED_PORTS = range(5555, 5573)  # PLAN.md D15: never bound by tests


class _NoTtnn(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "ttnn" or name.startswith("ttnn."):
            raise ImportError("ttnn is blocked in the CPU TT tests (the device must not be opened)")
        return None


if "ttnn" in sys.modules:
    raise RuntimeError("ttnn was imported before the TT CPU tests started")
sys.meta_path.insert(0, _NoTtnn())

try:  # only ttpy resolves the port; under simpy/srvpy (e.g. `pytest $RT/tests`) the ttpy-only files are skipped
    import models.experimental.gr00t as _port  # noqa: F401

    del _port
except ImportError:
    collect_ignore = ["test_g1_host_cpu.py", "test_tt_server_protocol.py"]


def pytest_addoption(parser):
    """The T-2.2 options of tests/tt/conftest_fidelity.py (the fidelity lane's file), so the plain PLAN.md command
    works; a second registration through ``-p tests.tt.conftest_fidelity`` is tolerated."""
    try:
        from .conftest_fidelity import pytest_addoption as fidelity_options
    except ImportError:
        return
    try:
        fidelity_options(parser)
    except ValueError:  # "option names already added": loaded with -p as well
        pass


def pytest_configure(config):
    config.addinivalue_line("markers", "device: needs the Blackhole (never collected on CPU)")


@pytest.fixture(scope="session", autouse=True)
def _ttnn_never_imported():
    yield
    assert "ttnn" not in sys.modules, "a test imported ttnn"


@pytest.fixture(scope="session")
def g1():
    """rttt.g1_n16 after install() (process-local registry patch onto the BF16 G1 copy)."""
    from rttt import g1_n16 as G

    G.install()
    return G


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    from rttt import golden_io as gio

    return gio.GOLDEN_DIR


def make_obs(seed: int = 0, instruction: str = "pick up the apple, walk left and place the apple on the plate.",
             b: int = 1) -> dict:
    """A G1 W-VLA observation (PLAN.md §1.4.1 shapes) with a random 480x640 frame and states in [-0.5, 0.5]."""
    from rtloop import schema

    rng = np.random.default_rng(seed)
    obs = {schema.VIDEO_KEY: rng.integers(0, 256, (b, 1, *schema.IMAGE_SHAPE), dtype=np.uint8),
           schema.LANGUAGE_KEY: (instruction,) * b}
    for (name, n), key in zip(schema.STATE_GROUPS, schema.STATE_KEYS):
        obs[key] = rng.uniform(-0.5, 0.5, (b, 1, n)).astype(np.float32)
    return obs


@pytest.fixture(scope="session")
def obs_factory():
    return make_obs
