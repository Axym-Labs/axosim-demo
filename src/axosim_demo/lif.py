"""A declared LIF substitution in the demo's reduced sensory motif.

Constants and continuous equations follow Shiu et al.'s released model
(MIT; commit91bdd1e7dcf193f3e7ca5a8933497fcef63b7960). This adapter is NOT
their full-brain simulation: it shares the demo's synthetic contacts, fractional
drive, adaptation parameters, forecast latency and engineered motor decoder.
Use shiu_benchmark.py for the original unmodified Brian2 experiment.
"""
from types import SimpleNamespace

import numpy as np
import torch

from .neural import OdorNavigationController


class LIFPopulation:
    """Analytically integrated exponential-current LIF, 0.1-ms event clock."""

    def __init__(self, neurons=4, contacts=16):
        self.device = torch.device('cpu')
        self.population = SimpleNamespace(
            synaptic_log_efficacy=torch.zeros(neurons, contacts))
        self.metadata = {
            'backend': 'lif', 'model_kind': 'Shiu-equation LIF substitution',
            'neurons': neurons, 'contacts_per_neuron': contacts,
            'dt_ms': .1, 'tau_membrane_ms': 20., 'tau_synapse_ms': 5.,
            'rest_mv': -52., 'reset_mv': -52., 'threshold_mv': -45.,
            'refractory_ms': 2.2, 'delay_ms': 1.8, 'weight_mv': .275,
            'source_commit': '91bdd1e7dcf193f3e7ca5a8933497fcef63b7960',
            'biological_claim': 'reduced synthetic motif; not original whole-brain baseline',
            'input_contract': 'same fractional contacts as AxoSim; one event-bin per ms',
            'output_contract': 'actual spike indicator; soma=(mV+67.7)*0.1',
        }
        self.v = np.full(neurons, -52.)
        self.g = np.zeros(neurons)
        self.refractory = np.zeros(neurons, dtype=int)
        self.queue = np.zeros((19, neurons))
        self.tick = 0
        self.pending = torch.zeros(neurons, 4, 2)
        self.blocks = 0

    @property
    def state(self):
        return torch.as_tensor(np.stack([self.v, self.g]))

    def step(self, patch):
        patch = np.asarray(patch, dtype=float)
        expected = (len(self.v), 4, self.population.synaptic_log_efficacy.shape[1])
        if patch.shape != expected or not np.isfinite(patch).all():
            raise ValueError(f'Expected finite patch of shape {expected}')
        drive = (patch * self.population.synaptic_log_efficacy.exp().numpy()[:, None]).sum(-1)
        forecast = np.zeros((len(self.v), 4, 2), dtype=np.float32)
        em, es = np.exp(-.1 / 20), np.exp(-.1 / 5)
        for sample in range(4):
            spikes = np.zeros(len(self.v), dtype=bool)
            for substep in range(10):
                if substep == 0:
                    self.queue[(self.tick + 18) % 19] += .275 * drive[:, sample]
                active = self.refractory == 0
                self.v[active] = (-52 + (self.v[active] + 52) * em
                                  + self.g[active] * (es - em) / (1 - 20 / 5))
                self.g[active] *= es
                fired = active & (self.v > -45)
                spikes |= fired
                self.refractory = np.maximum(self.refractory - 1, 0)
                # Brian2 becomes active at t-lastspike >= 2.2 ms: the firing
                # tick itself already accounts for one of those 22 ticks.
                self.refractory[fired] = 21
                self.v[fired] = -52
                self.g[fired] = 0
                # Brian2's "unless refractory" also gates synaptic writes.
                self.g += self.queue[self.tick % 19] * (active & ~fired)
                self.queue[self.tick % 19] = 0
                self.tick += 1
            forecast[:, sample, 0] = spikes
            forecast[:, sample, 1] = (self.v + 67.7) * .1
        result = self.pending
        self.pending = torch.from_numpy(forecast)
        self.blocks += 1
        return result


class LIFNavigationController(OdorNavigationController):
    def __init__(self, log_efficacy=None):
        super().__init__(log_efficacy, backend=LIFPopulation())
        self.label = 'LIF: matched reduced odor motif'
        self.metadata['mode'] = 'reduced_motif_lif_substitution'
