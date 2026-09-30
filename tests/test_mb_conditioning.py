import numpy as np

from axosim_demo.mb_conditioning import _bootstrap_mean_interval, _disc


def test_discrimination_handles_zero_and_direction():
    assert _disc({"CS+": 0, "CS-": 0}) == 0.0
    assert _disc({"CS+": 9, "CS-": 3}) == 0.5
    assert _disc({"CS+": 3, "CS-": 9}) == -0.5


def test_bootstrap_interval_is_deterministic_and_directional():
    values = np.array([-0.4, -0.2, -0.5, -0.3])
    first = _bootstrap_mean_interval(values, samples=5000, seed=7)
    second = _bootstrap_mean_interval(values, samples=5000, seed=7)
    assert first == second
    assert first[1] < 0
