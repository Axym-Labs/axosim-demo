import numpy as np
import torch
import pytest

from axosim_demo.lif import LIFPopulation, LIFNavigationController


def test_lif_exact_subthreshold_solution():
    p = LIFPopulation(neurons=1, contacts=1)
    p.v[:] = -50
    p.g[:] = 1
    p.step(np.zeros((1, 4, 1)))
    expected = -52 + 2 * np.exp(-4 / 20) + (np.exp(-4 / 5) - np.exp(-4 / 20)) / (1 - 20 / 5)
    np.testing.assert_allclose(p.v, expected, atol=1e-12)
    np.testing.assert_allclose(p.g, np.exp(-4 / 5), atol=1e-12)


def test_lif_causal_delay_and_finite_spiking():
    p = LIFPopulation(neurons=1, contacts=1)
    output = p.step(np.ones((1, 4, 1)) * 100)
    assert torch.count_nonzero(output) == 0  # same four-ms forecast contract
    outputs = [p.step(np.ones((1, 4, 1)) * 100) for _ in range(30)]
    output = torch.cat(outputs, 1)
    assert torch.isfinite(output).all()
    assert output[..., 0].sum() > 0
    assert np.max(p.v) <= -45


def test_lif_uses_same_adaptation_and_symmetric_motor_decoder():
    weights = np.zeros(16)
    weights[:8] = -2
    trained, frozen = LIFNavigationController(weights), LIFNavigationController()
    obs = {'odor_intensity': np.array([.7, .7]), 'source_bearings_rad': np.array([.5, -.5])}
    for t in np.arange(0, .4, .004):
        result, control = trained(t, obs), frozen(t, obs)
    assert result['diagnostics']['learned_value_A'] > 0
    np.testing.assert_allclose(control['command'], [1, 1])
    assert result['command'][0] < result['command'][1]


def test_lif_matches_brian2_delay_reset_and_refractory():
    b = pytest.importorskip('brian2')
    b.start_scope()
    b.prefs.codegen.target = 'numpy'
    b.defaultclock.dt = .1*b.ms
    neuron = b.NeuronGroup(1,
        'dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)\n'
        'dg/dt=-g/(5*ms):volt (unless refractory)',
        threshold='v > -45*mV', reset='v=-52*mV; g=0*mV',
        refractory=2.2*b.ms, method='linear')
    neuron.v = -52*b.mV
    source = b.SpikeGeneratorGroup(1, np.zeros(40,dtype=int), np.arange(40)*b.ms)
    synapse = b.Synapses(source,neuron,on_pre='g += 27.5*mV',delay=1.8*b.ms)
    synapse.connect()
    monitor = b.StateMonitor(neuron,'v',record=True,when='end')
    spikes = b.SpikeMonitor(neuron)
    b.Network(neuron,source,synapse,monitor,spikes).run(40*b.ms)
    p = LIFPopulation(neurons=1, contacts=1)
    predictions=[]
    for _ in range(10):
        p.step(np.ones((1,4,1))*100)
        predictions.append(p.pending.numpy()[0])
    predictions=np.concatenate(predictions)
    np.testing.assert_allclose(predictions[:,1]/.1-67.7, np.asarray(monitor.v[0]/b.mV)[9::10],atol=1e-5)
    expected=np.zeros(40)
    expected[np.floor(np.asarray(spikes.t/b.ms)).astype(int)]=1
    np.testing.assert_array_equal(predictions[:,0],expected)
