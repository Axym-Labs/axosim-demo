import numpy as np
import pytest
import torch

from axosim_demo.neural import DEFAULT_CHECKPOINT, PersistentPopulation, OdorNavigationController


@pytest.fixture
def backend():
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("Pin-verified trained AxoSim-Lite checkpoint is not installed")
    torch.set_num_threads(1)
    return PersistentPopulation()


def test_streaming_preserves_recurrence_and_forecast_alignment(backend):
    p = backend.population
    torch.manual_seed(2)
    contacts = torch.rand(2, 24, 16) * .04
    with torch.no_grad():
        expected = p(contacts)
    actual = torch.cat([backend.step(x) for x in contacts.split(4, dim=1)], dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert torch.count_nonzero(actual[:, :4]) == 0
    assert backend.state.abs().sum() > 0
    backend.reset()
    assert torch.count_nonzero(backend.state) == 0


def test_checkpoint_hash_required(tmp_path):
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not trained weights")
    with pytest.raises(ValueError, match="SHA256"):
        PersistentPopulation(wrong)


def test_adapted_neural_outputs_change_navigation(backend):
    weights = np.zeros(16)
    weights[:8] = -2
    trained = OdorNavigationController(weights)
    frozen = OdorNavigationController()
    observation = {"odor_intensity": np.array([.7, .7]), "source_bearings_rad": np.array([.5, -.5])}
    for t in np.arange(0, .4, .004):
        out = trained(t, observation)
        baseline = frozen(t, observation)
    assert out["diagnostics"]["learned_value_A"] > .05
    assert abs(out["diagnostics"]["learned_value_B"]) < 1e-6
    assert not np.allclose(out["command"], baseline["command"])
    assert out["diagnostics"]["neural_blocks"] == 100


def test_skipped_callback_boundaries_do_not_read_future_sensors(backend):
    regular, skipped = OdorNavigationController(), OdorNavigationController()
    before = {"odor_intensity": np.array([.2, .2]), "source_bearings_rad": np.array([.5, -.5])}
    after = {"odor_intensity": np.array([.9, .1]), "source_bearings_rad": np.array([.5, -.5])}
    regular(0, before)
    skipped(0, before)
    for t in (.004, .008, .012, .016):
        regular(t, before)
    regular(.020, after)
    skipped(.020, after)
    torch.testing.assert_close(regular.backend.state, skipped.backend.state)
    np.testing.assert_allclose(regular.last_output, skipped.last_output)
