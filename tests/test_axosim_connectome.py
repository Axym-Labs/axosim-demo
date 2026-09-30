"""Protect graph direction, every released edge, and independent plastic efficacy."""
import numpy as np
import pandas as pd
import pytest
from axosim_demo.connectome import ExactConnectome


def graph_files(tmp_path):
    # IDs deliberately exceed float64's exact integer range.
    ids = np.array([720575940596125868, 720575940605825666, 720575940620159366])
    neurons = tmp_path / 'neurons.csv'
    pd.DataFrame({'root_id': ids, 'Completed': [True]*3}).to_csv(neurons, index=False)
    edges = tmp_path / 'edges.parquet'
    pd.DataFrame({
        'Presynaptic_ID': ids[[0, 0, 1, 2]], 'Postsynaptic_ID': ids[[1, 1, 2, 0]],
        'Presynaptic_Index': [0,0,1,2], 'Postsynaptic_Index': [1,1,2,0],
        'Connectivity': [1,3,2,1], 'Excitatory': [1,1,-1,1],
        'Excitatory x Connectivity': [1,3,-2,1],
    }).to_parquet(edges)
    return neurons, edges


def test_preserve_all_edges_ids_direction_and_duplicate_contacts(tmp_path):
    graph = ExactConnectome.from_shiu(*graph_files(tmp_path))
    assert graph.n_neurons == 3
    assert graph.n_edges == 4  # no threshold, no duplicate coalescing
    assert graph.contact_count.sum() == 7
    assert graph.neuron_ids[0] == 720575940596125868
    np.testing.assert_array_equal(graph.route(np.array([1., 2., 4.])), [4., 4., -4.])
    np.testing.assert_array_equal(graph.route(np.array([1., 2., 4.]), efficacy=np.array([2.,1.,1.,1.])), [4.,5.,-4.])


def test_reject_inconsistent_root_id_mapping(tmp_path):
    neurons, edges = graph_files(tmp_path)
    df = pd.read_parquet(edges)
    df.loc[0, 'Postsynaptic_Index'] = 2
    df.to_parquet(edges)
    with pytest.raises(ValueError, match='ID'):
        ExactConnectome.from_shiu(neurons, edges)


def test_torch_routing_matches_hand_computed_signed_sum(tmp_path):
    import torch
    graph = ExactConnectome.from_shiu(*graph_files(tmp_path))
    router = graph.torch_router('cpu')
    out = router(torch.tensor([1.,2.,4.]))
    torch.testing.assert_close(out, torch.tensor([4.,4.,-4.]))


def test_torch_block_routing_matches_independent_reference_steps(tmp_path):
    import torch
    graph = ExactConnectome.from_shiu(*graph_files(tmp_path))
    router = graph.torch_router('cpu')
    activity = torch.tensor([[1., 0., 2.], [2., 3., 0.], [4., 1., 1.]])
    expected = torch.stack([router(activity[:, t]) for t in range(3)], dim=1)
    torch.testing.assert_close(router.route_block(activity), expected)


def test_flat_memory_is_based_on_actual_edges_not_max_degree(tmp_path):
    graph = ExactConnectome.from_shiu(*graph_files(tmp_path))
    report = graph.memory_report()
    assert report['edges'] == 4
    assert report['max_in_degree'] == 2
    assert report['rectangular_contact_slots'] == 6
    assert report['flat_topology_and_fp32_efficacy_bytes'] == 4 * 16


def test_reject_fractional_contact_counts_instead_of_truncating(tmp_path):
    neurons, edges = graph_files(tmp_path)
    df = pd.read_parquet(edges)
    df['Connectivity'] = df['Connectivity'].astype(float)
    df['Excitatory x Connectivity'] = df['Excitatory x Connectivity'].astype(float)
    df.loc[0, 'Connectivity'] = 1.5
    df.loc[0, 'Excitatory x Connectivity'] = 1.5
    df.to_parquet(edges)
    with pytest.raises(ValueError, match='integer'):
        ExactConnectome.from_shiu(neurons, edges)
