"""Predeclared whole-connectome AxoSim sensory-clamp experiment.

This module preserves every released Shiu v630 pair edge and integer contact
count. The source graph lacks contact positions on dendritic morphologies, so
the implementation marginalizes each contact over the pretrained specimen's
route features. That is an explicit uniform-location null hypothesis, not a
recovered fly morphology. No network gain is fitted to the expected result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from axosim.checkpoint import load_checkpoint

from .connectome import ExactConnectome
from .neural import CHECKPOINT_SHA256, DEFAULT_CHECKPOINT


class StreamingCrossingDecoder:
    """Convert checkpoint spike scores into causal upward-threshold events."""

    def __init__(self, neurons, *, threshold=0.0, device='cpu'):
        self.neurons = int(neurons)
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.previous_above = torch.zeros(
            self.neurons, dtype=torch.bool, device=self.device,
        )

    def __call__(self, scores):
        if scores.ndim != 2 or tuple(scores.shape) != (self.neurons, 4):
            raise ValueError('scores must have shape (neurons, 4)')
        above = scores >= self.threshold
        prior = torch.cat((self.previous_above[:, None], above[:, :-1]), dim=1)
        events = above & ~prior
        self.previous_above = above[:, -1]
        return events


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class AxoSimWholeBrain:
    """One morphology-conditioned AxoSim state per released v630 neuron."""

    def __init__(
        self,
        graph,
        *,
        checkpoint=DEFAULT_CHECKPOINT,
        morphology_index=0,
        device='cuda',
        spike_threshold=0.0,
    ):
        checkpoint = Path(checkpoint)
        digest = _sha256(checkpoint)
        if digest != CHECKPOINT_SHA256:
            raise ValueError(f'Checkpoint SHA256 mismatch: {digest}')
        model, metadata = load_checkpoint(checkpoint)
        if model.config.patch_size != 4 or model.config.output_dim != 2:
            raise ValueError('Whole-brain runtime requires the AxoSim P4 two-output contract')
        if not 0 <= morphology_index < len(model.config.morphology_ids):
            raise ValueError('Unknown checkpoint morphology index')
        self.device = torch.device(device)
        self.graph = graph
        self.router = graph.torch_router(self.device)
        self.model = model.to(self.device).eval()
        self.morphology_index = int(morphology_index)
        self.route_feature = self.model.route_features[
            self.morphology_index
        ].mean(dim=0).to(self.device)
        self.adaptation_cache = self.model.compile_adaptation(
            self.model.behavior_adaptation_bank[self.morphology_index][None].to(self.device)
        ).expand(graph.n_neurons, -1)
        self.decoder = StreamingCrossingDecoder(
            graph.n_neurons, threshold=spike_threshold, device=self.device,
        )
        self.checkpoint_metadata = metadata
        self.checkpoint_sha256 = digest
        self.reset()

    def reset(self):
        self.state = self.model.initial_state(
            self.graph.n_neurons, device=self.device, dtype=torch.float32,
        )
        self.recurrent_events = torch.zeros(
            self.graph.n_neurons, 4, dtype=torch.bool, device=self.device,
        )
        self.pending_events = torch.zeros_like(self.recurrent_events)
        self.decoder.reset()

    @torch.inference_mode()
    def step(self, clamp_indices, clamp_events):
        clamp_indices = torch.as_tensor(clamp_indices, dtype=torch.long, device=self.device)
        clamp_events = torch.as_tensor(clamp_events, dtype=torch.bool, device=self.device)
        if tuple(clamp_events.shape) != (len(clamp_indices), 4):
            raise ValueError('clamp_events must have shape (clamped_neurons, 4)')
        events_to_route = self.recurrent_events
        signed_contacts = self.router.route_block(events_to_route.to(torch.float32))
        summaries = signed_contacts.unsqueeze(-1) * self.route_feature
        forecast, self.state = self.model.step_p4(
            summaries,
            self.state,
            adaptation_cache=self.adaptation_cache,
        )
        emitted = self.decoder(forecast[..., 0])
        emitted.index_copy_(0, clamp_indices, clamp_events)
        self.recurrent_events = self.pending_events
        self.pending_events = emitted
        return forecast, emitted, events_to_route, signed_contacts


def _root_indices(graph, root_ids):
    order = np.argsort(graph.neuron_ids)
    sorted_ids = graph.neuron_ids[order]
    positions = np.searchsorted(sorted_ids, np.asarray(root_ids, dtype=np.int64))
    if np.any(positions == len(sorted_ids)) or not np.array_equal(
        sorted_ids[np.minimum(positions, len(sorted_ids) - 1)], root_ids
    ):
        raise ValueError('Configured root ID is absent from the released graph')
    return order[positions]


def run_condition(
    runtime,
    *,
    clamp_indices,
    readout_indices,
    duration_ms,
    rate_hz,
    seed,
    storm_fraction=0.2,
    storm_steps=3,
):
    if duration_ms <= 0 or duration_ms % 4:
        raise ValueError('duration_ms must be a positive multiple of four')
    if not 0 <= rate_hz <= 1000:
        raise ValueError('rate_hz must lie in [0, 1000] for one-ms event bins')
    runtime.reset()
    rng = np.random.default_rng(seed)
    event_probability = -np.expm1(-rate_hz / 1000.0)
    event_rows = []
    population_counts = []
    readout_events = np.zeros((len(readout_indices), duration_ms + 4), dtype=bool)
    unstable_steps = 0
    started = time.monotonic()
    for block_start in range(0, duration_ms, 4):
        clamp = rng.random((len(clamp_indices), 4)) < event_probability
        _, emitted, _, _ = runtime.step(clamp_indices, clamp)
        counts = emitted.sum(dim=0).cpu().numpy().astype(np.int64)
        population_counts.extend(counts.tolist())
        readout_events[:, block_start + 4:block_start + 8] = emitted[
            readout_indices
        ].cpu().numpy()
        active = torch.nonzero(emitted, as_tuple=False)
        if len(active):
            rows = active.cpu().numpy().astype(np.int32)
            rows[:, 1] += block_start + 4
            event_rows.append(rows)
        fractions = counts / runtime.graph.n_neurons
        for fraction in fractions:
            unstable_steps = unstable_steps + 1 if fraction > storm_fraction else 0
            if unstable_steps >= storm_steps:
                raise RuntimeError(
                    f'Population storm guard: >{storm_fraction:.0%} active for '
                    f'{storm_steps} consecutive milliseconds'
                )
    sparse_events = (
        np.concatenate(event_rows, axis=0)
        if event_rows else np.empty((0, 2), dtype=np.int32)
    )
    return {
        'readout_events': readout_events,
        'population_counts': np.asarray(population_counts, dtype=np.int64),
        'sparse_events_neuron_time': sparse_events,
        'wall_seconds': time.monotonic() - started,
    }


def run_experiment(
    *,
    config,
    output,
    checkpoint=DEFAULT_CHECKPOINT,
    conditions=None,
    seeds=(0,),
    duration_ms=1000,
    device='cuda',
    morphology_index=0,
):
    config = Path(config)
    contract = json.loads(config.read_text())
    conditions = tuple(contract['conditions']) if conditions is None else tuple(conditions)
    graph = ExactConnectome.from_shiu(
        contract['files']['neurons']['path'],
        contract['files']['edges']['path'],
    )
    expected_hashes = {
        'neurons_sha256': contract['files']['neurons']['sha256'],
        'edges_sha256': contract['files']['edges']['sha256'],
    }
    for name, expected in expected_hashes.items():
        if graph.provenance[name] != expected:
            raise ValueError(f'{name} differs from the pinned experiment contract')
    runtime = AxoSimWholeBrain(
        graph,
        checkpoint=checkpoint,
        morphology_index=morphology_index,
        device=device,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    readout_names = list(contract['readouts'])
    readout_indices = _root_indices(
        graph, [contract['readouts'][name]['root_id'] for name in readout_names]
    )
    report = {
        'status': 'running',
        'scope': 'pretrained AxoSim whole-brain v630 sensory-clamp experiment',
        'checkpoint_sha256': runtime.checkpoint_sha256,
        'checkpoint_model_kind': runtime.checkpoint_metadata['model_kind'],
        'checkpoint_training_domain': 'mammalian neuronal surrogate; fly use is out of domain',
        'graph': graph.provenance,
        'n_neurons': graph.n_neurons,
        'n_edges': graph.n_edges,
        'anatomical_contact_count': int(graph.contact_count.sum(dtype=np.int64)),
        'duration_ms': duration_ms,
        'rate_hz': contract['rate_hz'],
        'conditions': list(conditions),
        'seeds': list(seeds),
        'readouts': dict(zip(readout_names, map(int, readout_indices))),
        'neural_contract': {
            'morphology_index': morphology_index,
            'morphology_id': runtime.model.config.morphology_ids[morphology_index],
            'contact_location_hypothesis': 'uniform marginal mean over all checkpoint route features',
            'network_gain': 1.0,
            'network_gain_fitted': False,
            'spike_score_threshold': 0.0,
            'event_extractor': 'causal upward threshold crossing',
            'forecast_block_ms': 4,
            'recurrent_route_delay_ms': 4,
            'sensory_clamp': 'independent Poisson processes binned to one-ms presence events; clamped outputs override recurrent outputs',
        },
        'results': [],
    }
    report_path = output / 'summary.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        for condition in conditions:
            if condition not in contract['conditions']:
                raise ValueError(f'Unknown condition: {condition}')
            clamp_indices = _root_indices(
                graph, contract['conditions'][condition]['root_ids']
            )
            for seed in seeds:
                if runtime.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(runtime.device)
                result = run_condition(
                    runtime,
                    clamp_indices=clamp_indices,
                    readout_indices=readout_indices,
                    duration_ms=duration_ms,
                    rate_hz=contract['rate_hz'],
                    seed=seed,
                )
                rates = result['readout_events'][:, 4:duration_ms + 4].sum(axis=1) * 1000 / duration_ms
                stem = f'{condition}_seed{seed}'
                np.savez_compressed(
                    output / f'{stem}.npz',
                    readout_events=result['readout_events'],
                    population_counts=result['population_counts'],
                    sparse_events_neuron_time=result['sparse_events_neuron_time'],
                )
                row = {
                    'condition': condition,
                    'seed': seed,
                    'clamped_neurons': len(clamp_indices),
                    'wall_seconds': result['wall_seconds'],
                    'total_network_events': int(result['sparse_events_neuron_time'].shape[0]),
                    'maximum_active_fraction': float(result['population_counts'].max(initial=0) / graph.n_neurons),
                    'readout_rate_hz': dict(zip(readout_names, map(float, rates))),
                }
                if runtime.device.type == 'cuda':
                    row['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(runtime.device)
                    row['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(runtime.device)
                report['results'].append(row)
                report_path.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(row), flush=True)
        comparisons = {}
        for comparison in contract.get('comparisons', []):
            readout = comparison['readout']
            positive = {
                r['seed']: r['readout_rate_hz'][readout]
                for r in report['results'] if r['condition'] == comparison['positive']
            }
            negative = {
                r['seed']: r['readout_rate_hz'][readout]
                for r in report['results'] if r['condition'] == comparison['negative']
            }
            comparisons[comparison['name']] = {
                str(seed): {
                    'difference_hz': positive[seed] - negative[seed],
                    'positive_greater_than_negative': positive[seed] > negative[seed],
                }
                for seed in positive.keys() & negative.keys()
            }
        report['comparisons_by_seed'] = comparisons
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/shiu_grooming.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--duration-ms', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--morphology-index', type=int, default=0)
    args = parser.parse_args()
    run_experiment(
        config=args.config,
        output=args.output,
        checkpoint=args.checkpoint,
        conditions=args.conditions,
        seeds=args.seeds,
        duration_ms=args.duration_ms,
        device=args.device,
        morphology_index=args.morphology_index,
    )


if __name__ == '__main__':
    main()
