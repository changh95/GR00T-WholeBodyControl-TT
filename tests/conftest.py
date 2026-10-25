import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs the RTX 5090 (run with srvpy servers)")
    config.addinivalue_line("markers", "sim: builds the MuJoCo G1 env (slow, ~10 s per env)")
    config.addinivalue_line("markers", "rt: starts a --mode rt loop (spin-paced; never while a real-time campaign "
                                       "runs: deselect with -m 'not rt')")


@pytest.fixture
def zmq_ctx():
    import zmq

    ctx = zmq.Context()
    yield ctx
    ctx.destroy(linger=0)


def pytest_addoption(parser):
    # T-1.3 (PLAN.md §2 M1): `pytest tests/test_official_equivalence.py --vla $VLA_GPU --seeds 0-4`.
    parser.addoption("--vla", default=None,
                     help="T-1.3 VLA URL (default: start rtserve.gpu_vla_server on a free port)")
    parser.addoption("--seeds", default=None, help="T-1.3 seed spec, e.g. 0-4 or 0,2,5")
