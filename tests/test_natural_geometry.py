import numpy as np

from axosim_demo.natural_geometry import (
    circular_distance_degrees,
    normalized_soma_to_mv,
    within_image_center,
)


def test_normalized_soma_is_inverted_to_millivolts():
    normalized = np.asarray([0.0, 0.77, 1.27], dtype=np.float32)
    np.testing.assert_allclose(
        normalized_soma_to_mv(normalized),
        np.asarray([-67.7, -60.0, -55.0], dtype=np.float32),
        atol=1e-5,
    )


def test_circular_distance_wraps_at_360_degrees():
    np.testing.assert_allclose(
        circular_distance_degrees(np.asarray([350, 0]), np.asarray([10, 180])),
        np.asarray([20, 180]),
    )


def test_within_image_center_removes_each_images_mean():
    values = np.asarray([[1, 4], [3, 8], [10, 3], [14, 7]], np.float32)
    image_index = np.asarray([0, 0, 1, 1])
    centered = within_image_center(values, image_index)
    np.testing.assert_allclose(centered[:2].mean(0), 0, atol=1e-7)
    np.testing.assert_allclose(centered[2:].mean(0), 0, atol=1e-7)
