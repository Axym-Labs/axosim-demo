import numpy as np
import torch

from axosim_demo.limit_cycle import (
    apply_full_state_perturbation,
    phase_insensitive_orbit_distance,
    summarise_return,
)
from axosim_demo.goodfire_tikz import _number


def test_tikz_number_uses_pgfplots_compatible_positive_exponents():
    assert _number(1e10) == "1e10"


def test_orbit_distance_is_phase_insensitive():
    phase = torch.linspace(0, 2 * torch.pi, 16)[:-1]
    reference = torch.stack((torch.cos(phase), torch.sin(phase)), dim=1)
    samples = reference.roll(4, dims=0)
    distance = phase_insensitive_orbit_distance(samples, reference)
    torch.testing.assert_close(distance, torch.zeros_like(distance), atol=5e-4, rtol=0)


def test_full_state_perturbation_changes_hidden_and_only_free_activity():
    hidden = torch.zeros((3, 2))
    activity = torch.zeros((3, 4))
    free = torch.tensor([1, 2])
    statistics = {
        "hidden_scale": torch.tensor([2.0, 3.0]),
        "activity_scale": torch.tensor([1.0, 4.0, 2.0, 0.5]),
    }
    changed_hidden, changed_activity = apply_full_state_perturbation(
        hidden,
        activity,
        free,
        statistics,
        amplitude=0.25,
        seed=7,
    )
    assert torch.count_nonzero(changed_hidden[free]) == len(free) * 2
    assert torch.count_nonzero(changed_hidden[0]) == 0
    assert torch.count_nonzero(changed_activity[free]) == len(free) * 4
    assert torch.count_nonzero(changed_activity[0]) == 0


def test_return_summary_passes_contracting_trajectories():
    times = np.arange(0, 3.01, 0.02, dtype=np.float32)
    scenes = np.asarray(["a", "b"])
    amplitudes = np.asarray([0.1, 0.25])
    seeds = np.asarray([1, 2, 3])
    control = np.full((2, len(times)), 0.01, dtype=np.float32)
    measured = np.empty((2, 2, 3, len(times)), dtype=np.float32)
    for amplitude_index, amplitude in enumerate(amplitudes):
        measured[amplitude_index] = 0.01 + amplitude * np.exp(-times / 0.25)
    contract = {
        "return_test": {
            "late_window_seconds": [2.0, 3.0],
            "control_distance_quantile": 0.95,
            "late_tolerance_as_initial_fraction": 0.05,
            "maximum_median_excess_ratio": 0.2,
            "maximum_bootstrap_upper_excess_ratio": 0.5,
            "minimum_trial_return_fraction": 0.9,
            "maximum_return_seconds": 2.5,
            "required_consecutive_seconds": 0.5,
            "bootstrap_repetitions": 100,
            "bootstrap_seed": 4,
        }
    }
    result = summarise_return(
        measured, control, times, scenes, amplitudes, seeds, contract
    )
    assert result["passed"]
    assert all(item["return_fraction"] == 1 for item in result["by_amplitude"])
