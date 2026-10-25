"""pytest options of the T-2.2 fidelity test (tests/tt/test_g1_fidelity.py), kept out of the TT lane's conftest.py.

Load it either way (both give the same options; every option also has an environment fallback, so the test runs
with neither):

    PYTHONPATH=$RT/tests/tt simpy -m pytest -p conftest_fidelity $RT/tests/tt/test_g1_fidelity.py \
        --endpoint $VLA_TT --golden DIR
    G1_ENDPOINT=$VLA_TT G1_GOLDEN=DIR simpy -m pytest $RT/tests/tt/test_g1_fidelity.py

(``-p tests.tt.conftest_fidelity`` does not work under simpy: the client venv ships a site-packages ``tests``
package that shadows RT's.)

or, once tests/tt/conftest.py imports the hook (``from .conftest_fidelity import pytest_addoption``), with the
plain PLAN.md T-2.2 command line. A plain import (not ``pytest_plugins``) because a non-root conftest may not use
``pytest_plugins`` when pytest collects it late (``pytest $RT/tests``). The environment names live in the test
(test_g1_fidelity.ENV).
"""


def pytest_addoption(parser):
    g = parser.getgroup("g1-fidelity", "T-2.2 TT G1 fidelity (tests/tt/test_g1_fidelity.py)")
    g.addoption("--endpoint", default=None, help="server_g1 URL, started with --debug-endpoints (env G1_ENDPOINT)")
    g.addoption("--golden", default=None, help="WP-2.2 golden dir (env G1_GOLDEN; default rttt.golden_io.GOLDEN_DIR)")
    g.addoption("--fidelity-report", default=None, help="write per-sample numbers and verdicts here (env G1_FIDELITY_REPORT)")
    g.addoption("--fidelity-timeout-ms", default=None, help="per predict_debug call, default 60000 (env G1_FIDELITY_TIMEOUT_MS)")
    g.addoption("--fidelity-outputs", default=None,
                help="save every TT output here, npz '<sid>/<key>' (env G1_FIDELITY_OUTPUTS)")
