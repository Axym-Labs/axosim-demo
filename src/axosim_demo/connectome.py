"""Lossless loading and routing of every edge in the released Shiu graph.

The release represents anatomical contacts as integer-count neuron-pair edges.
We preserve that representation, including duplicate rows; we do not claim to
recover contact-specific morphology or transmitter/receptor identities.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ExactConnectome:
    neuron_ids: np.ndarray
    source: np.ndarray
    target: np.ndarray
    signed_count: np.ndarray
    provenance: dict

    @classmethod
    def from_shiu(cls, neuron_path, edge_path):
        ids = pd.read_csv(neuron_path).iloc[:, 0].to_numpy(dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Neuron IDs must be unique')
        df = pd.read_parquet(edge_path)
        source = df['Presynaptic_Index'].to_numpy()
        target = df['Postsynaptic_Index'].to_numpy()
        for indices, id_col in ((source, 'Presynaptic_ID'), (target, 'Postsynaptic_ID')):
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError('Graph indices must be integers')
            if len(indices) and (indices.min() < 0 or indices.max() >= len(ids)):
                raise ValueError('Graph index outside neuron table')
            if not np.array_equal(ids[indices], df[id_col].to_numpy(dtype=np.int64)):
                raise ValueError('Root ID mapping differs from neuron table')
        counts = df['Connectivity'].to_numpy()
        signed = df['Excitatory x Connectivity'].to_numpy()
        signs = df['Excitatory'].to_numpy()
        if not all(np.issubdtype(a.dtype, np.integer) for a in (counts, signed, signs)):
            raise ValueError('Contact counts and signs must have integer dtypes')
        if not np.all(counts > 0) or not np.all(np.isin(signs, [-1, 1])):
            raise ValueError('Expected positive integer contact counts and known signs')
        if not np.array_equal(signed, counts * signs):
            raise ValueError('Signed counts inconsistent with contact counts')
        if len(signed) and np.abs(signed).max() > np.iinfo(np.int32).max:
            raise ValueError('Signed count exceeds exact int32 representation')
        if len(ids) > np.iinfo(np.int32).max:
            raise ValueError('Neuron count exceeds exact int32 index range')
        return cls(ids, source.astype(np.int32), target.astype(np.int32),
                   signed.astype(np.int32), {
                       'neurons_sha256': sha256(neuron_path),
                       'edges_sha256': sha256(edge_path),
                       'release': 'Shiu integer weighted pair graph; all rows retained',
                       'pruning': False, 'coalescing': False,
                   })

    @property
    def n_neurons(self):
        return len(self.neuron_ids)

    @property
    def n_edges(self):
        return len(self.source)

    @property
    def contact_count(self):
        return np.abs(self.signed_count)

    def route(self, presynaptic, efficacy=None):
        """Reference sum with one optional independently mutable efficacy per row."""
        x = np.asarray(presynaptic)
        if x.shape != (self.n_neurons,):
            raise ValueError('Expected one activity value per neuron')
        w = self.signed_count if efficacy is None else self.signed_count * efficacy
        return np.bincount(self.target, weights=x[self.source] * w,
                           minlength=self.n_neurons)

    def memory_report(self, feature_dim=118, behavior_values=88):
        degree = np.bincount(self.target, minlength=self.n_neurons)
        maximum = int(degree.max()) if degree.size else 0
        slots = self.n_neurons * maximum
        return {
            'neurons': self.n_neurons, 'edges': self.n_edges,
            'anatomical_contact_count': int(self.contact_count.sum(dtype=np.int64)),
            'max_in_degree': maximum,
            'in_degree_percentiles_50_95_99': np.percentile(degree, [50,95,99]).tolist(),
            'rectangular_contact_slots': slots,
            'padding_factor': slots / max(1, self.n_edges),
            'flat_topology_and_fp32_efficacy_bytes': self.n_edges * 16,
            'fp32_adam_parameter_grad_and_moments_bytes': self.n_edges * 16,
            'flat_all_edge_feature_tensor_fp32_bytes': self.n_edges * feature_dim * 4,
            'padded_all_edge_feature_tensor_fp32_bytes': slots * feature_dim * 4,
            'padded_contact_efficacy_fp32_bytes': slots * 4,
            'behavior_fp32_bytes': self.n_neurons * behavior_values * 4,
            'scope': 'Analytic component sizes; excludes allocator, activations and renderer. '
                     'Optimizer total includes parameters, so do not double-count efficacy.',
            'provenance': self.provenance,
        }

    def torch_router(self, device='cpu'):
        return TorchExactRouter(self, device)


class TorchExactRouter:
    """Flat all-edge reference runtime. No graph pruning or dense adjacency.

    Integer topology and FP32 arithmetic are unchanged from the selected routing
    equation. Floating-point reduction order can differ across CPU/CUDA. Synaptic
    delays and neuron dynamics are responsibilities of the caller.
    """
    def __init__(self, graph, device):
        import torch
        self.n_neurons = graph.n_neurons
        self.source = torch.as_tensor(graph.source, device=device)
        self.target = torch.as_tensor(graph.target, device=device)
        self.weight = torch.as_tensor(graph.signed_count, device=device, dtype=torch.float32)
        self.efficacy = torch.ones_like(self.weight)

    def __call__(self, activity):
        import torch
        if activity.shape != (self.n_neurons,):
            raise ValueError('Expected one activity value per neuron')
        messages = torch.index_select(activity, 0, self.source) * self.weight * self.efficacy
        out = torch.zeros(self.n_neurons, device=activity.device, dtype=activity.dtype)
        return out.index_add_(0, self.target, messages)

    def route_block(self, activity):
        """Route a short time block while retaining every stored edge row."""
        import torch
        if activity.ndim != 2 or activity.shape[0] != self.n_neurons:
            raise ValueError('Expected activity with shape (neurons, time)')
        messages = torch.index_select(activity, 0, self.source) * self.weight.unsqueeze(1) * self.efficacy.unsqueeze(1)
        out = torch.zeros(
            self.n_neurons, activity.shape[1],
            device=activity.device, dtype=activity.dtype,
        )
        return out.index_add_(0, self.target, messages)
