import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from axosim_demo.natural_manifold import _analyze_settling_and_oscillation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_neural_film_v3_program.py"
spec = importlib.util.spec_from_file_location("film_v3", SCRIPT)
film_v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(film_v3)


def _contract(kinds, scene_kinds=None):
    dynamics = {
        "transition_approach_seconds": [5.0, 5.8],
        "post_transition_seconds": [1.0, 1.8],
        "reference_orbit_seconds": [3.5, 4.5],
        "oscillation_seconds": [1.8, 4.8],
        "minimum_orbit_approach_ratio": 1.25,
        "maximum_peak_frequency_drift_hz": 0.5,
        "minimum_peak_power_fraction": 0.35,
        "amplitude_ratio": [0.75, 1.35],
    }
    if scene_kinds is not None:
        dynamics["scene_kinds"] = scene_kinds
    return {
        "source_dt_seconds": 0.005,
        "dynamics": dynamics,
        "scenes": [{"name": f"s{i}", "kind": kind} for i, kind in enumerate(kinds)],
    }


def _oscillating_activity(scenes, rng):
    time = np.arange(0, 6, 0.005)
    phase = 2 * np.pi * 10 * time
    activity = []
    for offset in range(scenes):
        base = rng.normal(size=(1, 2, 8))
        wave = np.sin(phase)[:, None, None] * rng.normal(size=(1, 2, 8))
        activity.append(base * (offset + 1) + wave)
    return np.asarray(activity, dtype=np.float32), time


def test_dynamics_scores_only_configured_constant_input_kinds():
    rng = np.random.default_rng(0)
    activity, time = _oscillating_activity(4, rng)
    state_time = time[::5][time[::5] >= 1.0]
    pca = rng.normal(size=(4 * len(state_time), 12))
    kinds = ["video-clip", "video-clip", "held-frame", "held-frame"]

    default = _analyze_settling_and_oscillation(
        activity, time, pca, state_time, _contract(kinds)
    )
    held = _analyze_settling_and_oscillation(
        activity, time, pca, state_time, _contract(kinds, ["held-frame"])
    )

    assert default["per_scene"] == []
    assert [scene["name"] for scene in held["per_scene"]] == ["s2", "s3"]
    # The first held frame is approached from the preceding clip window.
    assert held["per_scene"][0]["transition_to_post_transition_orbit_distance_ratio"]
    assert held["per_scene"][0]["peak_frequency_drift_hz"] == pytest.approx(0)


def test_nearest_source_frame_keeps_original_speed():
    assert [film_v3.source_frame_index(k / 30, 60) for k in range(4)] == [0, 2, 4, 6]
    assert [film_v3.source_frame_index(k / 30, 30) for k in range(4)] == [0, 1, 2, 3]


def test_crossfade_endpoints_and_blurred_midpoint():
    black = Image.new("RGB", (64, 36), (0, 0, 0))
    white = Image.new("RGB", (64, 36), (255, 255, 255))
    assert np.asarray(film_v3.crossfade(black, white, 0)).max() == 0
    assert np.asarray(film_v3.crossfade(black, white, 1)).min() == 255
    middle = np.asarray(film_v3.crossfade(black, white, 0.5))
    assert 120 <= middle.mean() <= 135


def test_clip_windows_tile_the_displayed_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    still = tmp_path / "still.mp4"
    still.write_bytes(b"still")
    items = [
        {
            "name": "clip",
            "kind": "video-clip",
            "display_seconds": 40.0,
            "creator": "c",
            "page": "p",
            "license": "l",
        },
        {
            "name": "still",
            "kind": "held-frame",
            "display_seconds": 5.0,
            "creator": "c",
            "page": "p",
            "license": "l",
        },
    ]
    scenes = film_v3.scenes_for(items, [clip, still])
    assert len(scenes) == 9
    assert [scene["start_seconds"] for scene in scenes[:8]] == [
        5.0 * index for index in range(8)
    ]
    assert {scene["display_group"] for scene in scenes[:8]} == {"clip"}
    assert scenes[-1]["kind"] == "held-frame" and scenes[-1]["start_seconds"] == 0
