import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "sim: builds the MuJoCo G1 env (slow, ~10 s per env)")
    config.addinivalue_line("markers", "rt: starts a --mode rt loop (spin-paced; never while a real-time campaign "
                                       "runs: deselect with -m 'not rt')")


@pytest.fixture
def zmq_ctx():
    import zmq

    ctx = zmq.Context()
    yield ctx
    ctx.destroy(linger=0)

