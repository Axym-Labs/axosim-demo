import numpy as np
import pytest
import torch

from axosim_demo.conditioning import ConditioningConfig, DopamineEligibilityRule, run_conditioning
from axosim_demo.neural import DEFAULT_CHECKPOINT, PersistentPopulation


@pytest.fixture(autouse=True)
def trained_checkpoint():
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("Pin-verified trained AxoSim-Lite checkpoint is not installed")
    torch.set_num_threads(1)


def test_reward_requires_eligible_contact_and_plasticity():
    backend = PersistentPopulation()
    rule = DopamineEligibilityRule(backend, ConditioningConfig())
    baseline = backend.population.synaptic_log_efficacy.clone()
    rule.update(np.zeros(16), 1)
    torch.testing.assert_close(backend.population.synaptic_log_efficacy, baseline)
    active = np.r_[np.ones(8), np.zeros(8)]
    rule.update(active, 0)
    torch.testing.assert_close(backend.population.synaptic_log_efficacy, baseline)
    rule.update(active, 1, plasticity=False)
    torch.testing.assert_close(backend.population.synaptic_log_efficacy, baseline)
    rule.update(active, 1)
    assert torch.all(backend.population.synaptic_log_efficacy[0, :8] < 0)
    assert torch.count_nonzero(backend.population.synaptic_log_efficacy[0, 8:]) == 0
    assert torch.count_nonzero(backend.population.synaptic_log_efficacy[1]) == 0


@pytest.mark.parametrize("rewarded_cue", [0, 1])
def test_conditioning_controls_and_cue_reversal(rewarded_cue):
    report = run_conditioning(config=ConditioningConfig(), rewarded_cue=rewarded_cue)
    arms = report["arms"]
    learned = arms["paired"]["learned_preference_difference"]
    assert learned > .1
    assert abs(arms["frozen"]["learned_preference_difference"]) < 1e-6
    assert abs(arms["dopamine_blocked"]["learned_preference_difference"]) < 1e-6
    assert abs(arms["unpaired"]["learned_preference_difference"]) < learned * .1
    assert arms["unpaired"]["dopamine_integral_seconds"] == pytest.approx(arms["paired"]["dopamine_integral_seconds"])
    assert all(a["shared_weights_frozen"] for a in arms.values())
