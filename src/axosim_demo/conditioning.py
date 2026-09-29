"""Reduced odor–reward motif using local dopamine-gated AxoSim adaptation.

No behavioral targets, labels, reward gradients or optimizer are used. A
presynaptic KC eligibility trace and compartment DAN signal depress actual
AxoSim contact efficacies. All neuronal weights and decoding remain frozen.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .neural import DEFAULT_CHECKPOINT, PersistentPopulation


@dataclass
class ConditioningConfig:
    trials: int = 12
    dt_seconds: float = .004
    cue_seconds: float = .4
    intertrial_seconds: float = .6
    eligibility_tau_seconds: float = .08
    learning_rate_per_second: float = .5
    reward_start_seconds: float = .2
    reward_seconds: float = .2
    minimum_log_efficacy: float = -3.0


class DopamineEligibilityRule:
    """Minimal KC terminal coincidence rule, not a fitted biochemical model."""

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.eligibility = torch.zeros(16, device=backend.device)
        self.integrated_dopamine = 0.0

    @torch.no_grad()
    def update(self, kc_activity, dopamine, *, plasticity=True):
        c = self.config
        decay = math.exp(-c.dt_seconds / c.eligibility_tau_seconds)
        self.eligibility.mul_(decay).add_(torch.as_tensor(kc_activity, device=self.backend.device), alpha=1-decay)
        self.integrated_dopamine += dopamine * c.dt_seconds
        if plasticity:
            parameter = self.backend.population.synaptic_log_efficacy[0]
            parameter.add_(self.eligibility, alpha=-c.learning_rate_per_second * dopamine * c.dt_seconds)
            parameter.clamp_(min=c.minimum_log_efficacy, max=0)


def _patch(activity):
    return np.broadcast_to(np.asarray(activity, dtype=np.float32)[None, None, :] / 8, (2, 4, 16)).copy()


def evaluate_cues(backend, config):
    """Reward-free trials with reset dynamical state, preserving efficacies."""
    result = {}
    for cue in (0, 1):
        backend.reset()
        for _ in range(round(.4 / config.dt_seconds)):
            backend.step(_patch(np.zeros(16)))
        activity = np.zeros(16)
        activity[cue*8:(cue+1)*8] = 1
        outputs = []
        for _ in range(round(config.cue_seconds / config.dt_seconds)):
            outputs.append(backend.step(_patch(activity)).mean(1).cpu().numpy())
        average = np.asarray(outputs)[len(outputs)//2:].mean(0)
        result[str(cue)] = {"plastic_soma": float(average[0, 1]), "reference_soma": float(average[1, 1]),
                            "approach_value": float(average[1, 1] - average[0, 1]),
                            "plastic_spike_related_output": float(average[0, 0])}
    return result


def run_conditioning(*, checkpoint=DEFAULT_CHECKPOINT, config=None, seed=0, device="cpu", rewarded_cue=0):
    c = config or ConditioningConfig()
    if c.dt_seconds != .004:
        raise ValueError("The pretrained AxoSim-Lite model has a fixed 4-ms cadence")
    if rewarded_cue not in (0, 1):
        raise ValueError("rewarded_cue must be 0 or 1")
    torch.set_num_threads(1)
    arms = {}
    for arm in ("paired", "unpaired", "frozen", "dopamine_blocked"):
        backend = PersistentPopulation(checkpoint, device=device)
        before = evaluate_cues(backend, c)
        backend.reset()
        rule = DopamineEligibilityRule(backend, c)
        rng = np.random.default_rng(seed)
        history = []
        for trial in range(c.trials):
            # Small KC gain variability is paired across all arms, not fitted.
            gains = rng.uniform(.85, 1.15, 8)
            duration = c.cue_seconds + c.intertrial_seconds
            for step in range(round(duration / c.dt_seconds)):
                t = step * c.dt_seconds
                activity = np.zeros(16)
                if t < c.cue_seconds:
                    activity[rewarded_cue*8:(rewarded_cue+1)*8] = gains
                reward_start = c.reward_start_seconds if arm != "unpaired" else .8
                reward = float(reward_start <= t < reward_start + c.reward_seconds - 1e-9)
                dopamine = 0.0 if arm == "dopamine_blocked" else reward
                backend.step(_patch(activity))
                rule.update(activity, dopamine, plasticity=arm != "frozen")
            history.append(backend.population.synaptic_log_efficacy[0].exp().tolist())
        after = evaluate_cues(backend, c)
        log_efficacy = backend.population.synaptic_log_efficacy[0].detach().cpu().tolist()
        difference = after[str(rewarded_cue)]["approach_value"] - after[str(1-rewarded_cue)]["approach_value"]
        arms[arm] = {"before": before, "after": after, "learned_preference_difference": difference,
                     "log_efficacy": log_efficacy, "efficacy_by_trial": history,
                     "dopamine_integral_seconds": rule.integrated_dopamine,
                     "shared_weights_frozen": all(not p.requires_grad for p in backend.population.neuron.parameters())}
    return {"scope": "reduced synthetic KC–MBON motif, not whole connectome or validated fly physiology",
            "checkpoint": backend.metadata, "config": asdict(c), "seed": seed, "rewarded_cue": rewarded_cue,
            "reward_pathway": "sugar proxy -> PAM-like DAN compartment pulse -> KC eligibility coincidence -> AxoSim synaptic_log_efficacy depression",
            "learning_rule": "d log efficacy = -eta * DAN * KC_eligibility * dt; no supervised objective",
            "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    results = [run_conditioning(checkpoint=args.checkpoint, seed=s, device=args.device, rewarded_cue=cue)
               for cue in (0, 1) for s in range(args.seeds)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": results}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(results),
                      "paired_preference": [r["arms"]["paired"]["learned_preference_difference"] for r in results]}))


if __name__ == "__main__":
    main()
