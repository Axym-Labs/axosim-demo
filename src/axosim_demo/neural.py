"""Persistent, causal AxoSim-Lite integration for an explicitly reduced motif.

The trained checkpoint is a mammalian neuronal surrogate. The input routes,
olfactory motif and motor decoding here are engineering hypotheses, not an
empirical fly connectome or a validated fly neuron model.
The navigation controller is a retired engineering prototype, not an accepted
scientific demonstration. PersistentPopulation remains a tested numerical adapter.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from axosim import AxoSimLite, AxoSimPopulation
from axosim.checkpoint import load_checkpoint

CHECKPOINT_SHA256 = "19a045bf5b62ca92ab547934ca031e31f464ff2421f0d07b60d8278189889626"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "data/checkpoints/axosim-lite-t24-s8.pt"


class PersistentPopulation:
    """Four-ms AxoSim state advancement with a causal forecast queue.

    Each call consumes four native one-ms samples, returns the forecast queued
    by the preceding call, and queues the newly produced forecast for the next
    four-ms block. It never resets recurrence between calls.
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, *, neurons=2, contacts=16,
                 device="cpu", expected_sha256=CHECKPOINT_SHA256):
        checkpoint = Path(checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"Checkpoint SHA256 mismatch: {digest}")
        neuron, metadata = load_checkpoint(checkpoint)
        if not isinstance(neuron, AxoSimLite):
            raise TypeError("The persistent population adapter requires AxoSim-Lite")
        # Route zero is a declared synthetic excitatory contact location in the
        # pretrained specimen. All 16 contacts remain independently mutable.
        self.population = AxoSimPopulation(
            neuron, morphology_indices=torch.zeros(neurons, dtype=torch.long),
            contact_branch_indices=torch.zeros(neurons, contacts, dtype=torch.long),
        ).to(device).eval()
        self.population.enable_adaptation_training(behavior=False, morphology=False, synaptic=False)
        self.device = torch.device(device)
        self.checkpoint_sha256 = digest
        self.metadata = {"checkpoint_sha256": digest, "model_kind": metadata["model_kind"],
                         "morphology": neuron.config.morphology_ids[0],
                         "route_branch": 0, "neurons": neurons, "contacts_per_neuron": contacts,
                         "biological_claim": "reduced synthetic motif; mammalian surrogate initialization"}
        self.reset()

    def reset(self):
        p = self.population
        self.state = p.neuron.initial_state(p.population_size, device=self.device, dtype=torch.float32)
        self.pending = torch.zeros(p.population_size, 4, 2, device=self.device)
        self.blocks = 0

    @torch.no_grad()
    def step(self, contact_patch):
        p = self.population
        patch = torch.as_tensor(contact_patch, dtype=torch.float32, device=self.device)
        expected = (p.population_size, 4, p.synaptic_efficacies_per_neuron)
        if tuple(patch.shape) != expected:
            raise ValueError(f"contact_patch must have shape {expected}")
        if not torch.isfinite(patch).all():
            raise ValueError("nonfinite contact inputs")
        # Small reduced motif only: full-contact features are safe here. The
        # whole-connectome importer must use flat, chunked exact routing.
        weighted = patch * p.synaptic_log_efficacy.exp().unsqueeze(1)
        features = p.neuron.route_features[p.morphology_indices[:, None], p.contact_branch_indices]
        summary = torch.bmm(weighted, features)
        cache = p.behavior_adapter + p.morphology_adapter[p.morphology_indices]
        forecast, self.state = p.neuron.step_p4(summary, self.state, adaptation_cache=cache)
        result = self.pending
        self.pending = forecast
        self.blocks += 1
        return result


class OdorNavigationController:
    """Callback compatible with body.record_demo.

    Four persistent surrogate neurons evaluate plastic/reference responses to
    the two odor channels. Their output difference is an engineered approach
    value. The body uses its upstream locomotion controller; no claim is made
    that connectome-derived descending neurons generate the gait.
    """

    label = "AxoSim: reduced dopamine-conditioned odor motif"

    def __init__(self, log_efficacy=None, *, checkpoint=DEFAULT_CHECKPOINT, device="cpu", backend=None):
        self.label = ("AxoSim: reduced dopamine-conditioned odor motif" if log_efficacy is not None
                      else "AxoSim: reduced odor motif / unconditioned")
        self.backend = backend if backend is not None else PersistentPopulation(checkpoint, neurons=4, device=device)
        if log_efficacy is not None:
            values = torch.as_tensor(log_efficacy, device=self.backend.device, dtype=torch.float32)
            if values.shape != (16,) or not torch.isfinite(values).all():
                raise ValueError("log_efficacy must contain 16 finite KC contact parameters")
            with torch.no_grad():
                self.backend.population.synaptic_log_efficacy[0].copy_(values)
                self.backend.population.synaptic_log_efficacy[2].copy_(values)
        self.metadata = dict(self.backend.metadata)
        self.metadata['contact_log_efficacy_sha256'] = hashlib.sha256(
            self.backend.population.synaptic_log_efficacy.detach().cpu().numpy().astype('<f4').tobytes()
        ).hexdigest()
        self.metadata.update({"mode": "reduced_motif", "native_dt_seconds": .001,
                              "forecast_block_seconds": .004, "adaptation_during_video": False,
                              "conditioned_weights_loaded": log_efficacy is not None,
                              "sensor_clock": "4-ms zero-order hold; missed boundaries use last observation, never future observation"})
        self.next_block = 0.0
        self.last_output = np.zeros((4, 2))
        self.previous_odors = None
        self.previous_time = None

    def __call__(self, time_seconds, observation):
        odors = np.clip(np.asarray(observation["odor_intensity"], dtype=float), 0, 1)
        if self.previous_time is not None and time_seconds < self.previous_time:
            raise ValueError("Controller clock must be monotonic")
        if self.previous_time is None and abs(time_seconds) > 1e-9:
            raise ValueError("Controller clock must start at zero")
        while time_seconds + 1e-9 >= self.next_block:
            # An observation arriving after a missed boundary cannot be used
            # retrospectively. Hold the preceding sensor sample instead.
            sample = (odors if abs(time_seconds-self.next_block) <= 1e-9
                      else self.previous_odors)
            patch = np.zeros((4, 4, 16), dtype=np.float32)
            patch[:2, :, :8] = sample[0] / 8
            patch[2:, :, 8:] = sample[1] / 8
            self.last_output = self.backend.step(patch).mean(1).cpu().numpy()
            self.next_block += .004
        self.previous_odors = odors.copy()
        self.previous_time = time_seconds
        # Channel one is the checkpoint's soma output in its native normalized
        # units, never mislabeled as an empirical fly voltage or spike rate.
        value = np.array([self.last_output[1, 1] - self.last_output[0, 1],
                          self.last_output[3, 1] - self.last_output[2, 1]])
        bearings = np.asarray(observation["source_bearings_rad"], dtype=float)
        turn = float(np.clip(np.sum((.08 * odors + value) * np.sin(bearings)) * 2, -.5, .5))
        # NeuroMechFly's turning controller uses bilateral drives near one.
        command = np.array([1.0 - turn, 1.0 + turn])
        return {"command": command, "diagnostics": {
            "odor_A": float(odors[0]), "odor_B": float(odors[1]),
            "learned_value_A": float(value[0]), "learned_value_B": float(value[1]),
            "neural_soma_A": float(self.last_output[0, 1]),
            "neural_soma_B": float(self.last_output[2, 1]),
            "forecast_warmup": time_seconds < .004,
            "AxoSim_soma_A": float(self.last_output[0, 1]),
            "AxoSim_soma_B": float(self.last_output[2, 1]),
            "AxoSim_forecast_A": float(self.last_output[0, 0]),
            "AxoSim_state_norm": float(self.backend.state.norm()),
            "neural_blocks": self.backend.blocks,
            "KC_A_efficacy": float(self.backend.population.synaptic_log_efficacy[0, :8].exp().mean()),
            "KC_B_efficacy": float(self.backend.population.synaptic_log_efficacy[0, 8:].exp().mean()),
        }}
