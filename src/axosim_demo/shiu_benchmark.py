"""Execute pinned, unmodified Shiu v630 Brian2 code as a scientific baseline.

This is intentionally separate from AxoSim: the reference defines the behavior
an AxoSim port must be compared against. Upstream code/data are downloaded into
ignored data/ paths and hash-checked before executing the MIT-licensed model.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
import traceback


def notebook_assignments(path, names):
    """Read only literal assignments; never execute notebook cells."""
    found = {}
    for cell in json.loads(Path(path).read_text())['cells']:
        if cell.get('cell_type') != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    missing = set(names) - found.keys()
    if missing:
        raise ValueError(f'Missing literal notebook assignments: {sorted(missing)}')
    return found


def verify_file(path, expected):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    return digest.hexdigest()


def run_baseline(root, config, output, conditions, seeds, backend='numpy'):
    import brian2 as b2
    import numpy as np
    import pandas as pd

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path(config).read_text())
    paths = {}
    for name, info in contract['files'].items():
        paths[name] = root / info['path']
        verify_file(paths[name], info['sha256'])
    desired = {v['notebook_variable']: v['root_ids'] for v in contract['conditions'].values() if v['notebook_variable']}
    desired.update({v['notebook_variable']: v['root_id'] for v in contract['readouts'].values()})
    if notebook_assignments(paths['notebook'], desired) != desired:
        raise ValueError('Configured populations differ from pinned upstream notebook')
    module_spec = importlib.util.spec_from_file_location('shiu_pinned', paths['model'])
    upstream = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(upstream)
    neuron_ids = pd.read_csv(paths['neurons'], index_col=0).index.to_numpy(dtype=np.int64)
    id_to_index = {int(root_id): i for i, root_id in enumerate(neuron_ids)}
    # Resolve every population now: never silently intersect absent root IDs.
    selected = {k: [id_to_index[r] for r in v['root_ids']] for k, v in contract['conditions'].items()}
    readout_indices = {k: id_to_index[v['root_id']] for k, v in contract['readouts'].items()}
    b2.prefs.codegen.target = backend
    report = {'contract': contract, 'brian2_version': b2.__version__,
              'numpy_version': np.__version__, 'codegen_target': backend,
              'n_neurons': len(neuron_ids), 'conditions': conditions, 'seeds': seeds,
              'results': [], 'scope': contract['scope'], 'status': 'running'}
    report_path = output / 'summary.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    for condition in conditions:
        for seed in seeds:
            started = time.monotonic()
            print(f'Running original v630 {condition} seed={seed}', flush=True)
            try:
                b2.start_scope()
                b2.defaultclock.dt = contract['dt_ms'] * b2.ms
                b2.seed(seed)
                params = dict(upstream.default_params)
                params['r_poi'] = contract['rate_hz'] * b2.Hz
                params['t_run'] = contract['duration_ms'] * b2.ms
                # Original network, equations, solver, weights, and monitor unchanged.
                neurons, synapses, monitor = upstream.create_model(paths['neurons'], paths['edges'], params)
                poisson, neurons = upstream.poi(neurons, selected[condition], [], params)
                net = b2.Network(neurons, synapses, monitor, *poisson)
                net.run(params['t_run'], report='text', report_period=60 * b2.second)
                spike_i = np.asarray(monitor.i[:])
                spike_t_ms = np.asarray(monitor.t[:] / b2.ms)
                readouts = {}
                saved = {}
                for name, idx in readout_indices.items():
                    times = spike_t_ms[spike_i == idx]
                    saved[name] = times
                    readouts[name] = {'root_id': int(neuron_ids[idx]), 'spike_count': len(times),
                                      'rate_hz': len(times) * 1000 / contract['duration_ms'],
                                      'first_spike_ms': float(times[0]) if len(times) else None}
                np.savez_compressed(output / f'{condition}_seed{seed}_readouts.npz', **saved)
                result = {'condition': condition, 'seed': seed, 'n_edges': len(synapses),
                          'stimulated_neurons': len(selected[condition]),
                          'total_network_spikes': len(spike_i), 'readouts': readouts,
                          'wall_seconds': time.monotonic() - started,
                          'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                report['results'].append(result)
                print(json.dumps(result), flush=True)
                del net, neurons, synapses, monitor, poisson, spike_i, spike_t_ms
            except Exception:
                report['status'] = 'failed'
                report['error'] = traceback.format_exc()
                report_path.write_text(json.dumps(report, indent=2) + '\n')
                raise
            report_path.write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'complete'
    ce = {r['seed']: r['readouts']['aBN1']['rate_hz'] for r in report['results'] if r['condition'] == 'jo_ce'}
    jf = {r['seed']: r['readouts']['aBN1']['rate_hz'] for r in report['results'] if r['condition'] == 'jo_f'}
    report['aBN1_direction_by_seed'] = {str(s): {'jo_ce_minus_jo_f_hz': ce[s] - jf[s],
                                                'jo_ce_greater_than_jo_f': ce[s] > jf[s]}
                                       for s in ce.keys() & jf.keys()}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, default=Path('configs/shiu_grooming.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', choices=['jo_ce', 'jo_f', 'baseline'], default=['jo_ce', 'jo_f'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--backend', choices=['numpy', 'cython'], default='numpy')
    args = parser.parse_args()
    run_baseline(args.root, args.config, args.output, args.conditions, args.seeds, args.backend)


if __name__ == '__main__':
    main()
