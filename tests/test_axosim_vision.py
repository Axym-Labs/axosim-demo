import numpy as np
import pytest

from axosim_demo.vision_axosim import _pearson, _resample, _spatial_target_control


def test_pearson_handles_direction_and_constant_vectors():
    assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert _pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1)
    assert _pearson([1, 1, 1], [1, 2, 3]) == 0


def test_resampling_preserves_endpoints_and_feature_axis():
    values = np.array([[0, 2], [1, 4], [2, 6]], dtype=np.float32)
    result, time = _resample(values, 0.005, 0.0025)
    assert result.shape == (5, 2)
    np.testing.assert_allclose(result[[0, -1]], values[[0, -1]])
    np.testing.assert_allclose(time[[0, -1]], [0, 0.01])


def test_spatial_control_preserves_target_types_and_degree_multiset():
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    target_type = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    shuffled = _spatial_target_control(target, target_type, seed=7)
    for value in np.unique(target_type):
        mask = target_type == value
        np.testing.assert_array_equal(np.sort(shuffled[mask]), np.sort(target[mask]))
    assert not np.array_equal(shuffled, target)
