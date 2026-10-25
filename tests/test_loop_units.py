"""Unit tests of rt_sim's pure helpers: seeds/duration parsing, the D17 noise seed, the pacer."""

import argparse
import itertools
import math

import numpy as np
import pytest

from rtloop import clock, loop, sim_env
from rtloop.schema import TICK_NS, VlaBackend


def test_parse_seeds():
    it, open_ended = loop.parse_seeds("2000-2049")
    assert list(it) == list(range(2000, 2050)) and not open_ended
    it, open_ended = loop.parse_seeds("3000-")
    assert list(itertools.islice(it, 3)) == [3000, 3001, 3002] and open_ended
    it, _ = loop.parse_seeds("1,5-6,9")
    assert list(it) == [1, 5, 6, 9]
    for bad in ("", "5-3", "a-b", "1-,4"):
        with pytest.raises(argparse.ArgumentTypeError):
            loop.parse_seeds(bad)


def test_parse_duration():
    assert loop.parse_duration("120") == 120 and loop.parse_duration("90s") == 90
    assert loop.parse_duration("2m") == 120 and loop.parse_duration("2h") == 7200


def test_splitmix64_reference_and_noise_seed():
    # First output of SplitMix64 seeded with 0 (Vigna's reference implementation).
    assert loop.splitmix64(0) == 0xE220A8397B1DCDAF
    assert loop.noise_seed(2000, 3) == loop.splitmix64((2000 << 20) | 3)
    seeds = {loop.noise_seed(s, i) for s in range(2000, 2050) for i in range(100)}
    assert len(seeds) == 5000 and all(0 <= x < 2**64 for x in seeds)


def test_vla_backend_of():
    assert loop.vla_backend_of({"backend": "mock:detfn"}) == VlaBackend.MOCK
    assert loop.vla_backend_of({"backend": "tt:p150a:MK@abc"}) == VlaBackend.TT
    assert loop.vla_backend_of({"backend": "unknown:x@1"}) == VlaBackend.OTHER
    assert loop.vla_backend_of({}) == VlaBackend.OTHER  # stock PolicyServer: no info endpoint


@pytest.mark.parametrize("mode", clock.PACING)
def test_pacer_deadlines_and_slip(mode):
    p = clock.Pacer(period_ns=TICK_NS // 10, mode=mode, spin_ns=500_000)  # 2 ms ticks keep it quick
    t0 = p.anchor()
    lates = []
    for k in range(5):
        t, late, slip = p.wait(k)
        assert not slip and t >= t0 + k * p.period_ns
        lates.append(late)
    assert max(lates) < 1_500_000
    clock.sleep_until(clock.now_ns() + 3 * p.period_ns)  # miss more than one period
    t, late, slip = p.wait(5)
    assert slip and late > p.period_ns and p.deadline(5) == t  # re-anchored: no catch-up burst
    t6, late6, slip6 = p.wait(6)
    assert not slip6 and t6 >= t + p.period_ns


def test_pelvis_state_roll_pitch():
    a = 0.1
    pose = np.array([0, 0, 0.75, math.cos(a / 2), math.sin(a / 2), 0, 0])  # roll a about x
    z, roll, pitch = sim_env.pelvis_state({"floating_base_pose": pose})
    assert z == 0.75 and abs(roll - a) < 1e-12 and abs(pitch) < 1e-12
    pose = np.array([0, 0, 0.4, math.cos(a / 2), 0, math.sin(a / 2), 0])  # pitch a about y
    z, roll, pitch = sim_env.pelvis_state({"floating_base_pose": pose})
    assert abs(pitch - a) < 1e-12 and abs(roll) < 1e-12 and sim_env.fell(z)


def test_cli_rejects_unbounded_open_seeds(capsys):
    with pytest.raises(SystemExit) as e:
        loop.main(["rtloop.loop", "--seeds", "3000-"])
    assert e.value.code == 2 and "open-ended" in capsys.readouterr().err
