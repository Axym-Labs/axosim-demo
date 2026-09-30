"""Reproduce the pinned FlyVis moving-edge reference metrics.

This command is a comparison arm and task-contract check. FlyVis uses its own
trained graded point-neuron dynamics; its outputs are never presented as an
AxoSim demonstration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FLYVIS_COMMIT = '92b3845cc426dd309a1a0e1b3890156c42e14021'
PRETRAINED_ZIP_SHA256 = '71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1'


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_reference(*, output, root=None):
    if root is not None:
        import os
        os.environ['FLYVIS_ROOT_DIR'] = str(Path(root).resolve())
    from flyvis import results_dir, root_dir
    from flyvis.analysis.moving_bar_responses import (
        correlation_to_known_tuning_curves,
        direction_selectivity_index,
        dsi_correlation_to_known,
        preferred_direction,
    )
    from flyvis.datasets.moving_bar import MovingEdge
    from flyvis.network import NetworkView

    archive = Path(root_dir) / 'results_pretrained_models.zip'
    if not archive.exists() or _sha256(archive) != PRETRAINED_ZIP_SHA256:
        raise ValueError('FlyVis pretrained archive is absent or differs from its pinned checksum')
    protocol = {
        'offsets': [-10, 11],
        'intensities': [0, 1],
        'speeds': [19],
        'height': 80,
        'post_pad_mode': 'continue',
        't_pre': 1.0,
        't_post': 1.0,
        'dt': 1 / 200,
        'angles_deg': list(range(0, 360, 30)),
    }
    dataset = MovingEdge(
        offsets=protocol['offsets'],
        intensities=protocol['intensities'],
        speeds=protocol['speeds'],
        height=protocol['height'],
        post_pad_mode=protocol['post_pad_mode'],
        t_pre=protocol['t_pre'],
        t_post=protocol['t_post'],
        dt=protocol['dt'],
        angles=protocol['angles_deg'],
    )
    network_path = results_dir / 'flow/0000/000'
    network = NetworkView(network_path)
    responses = network.moving_edge_responses(dataset)
    dsi = direction_selectivity_index(responses)
    direction = preferred_direction(responses)
    tuning = correlation_to_known_tuning_curves(responses)
    t4c_dsi = float(dsi.custom.where(cell_type='T4c', intensity=1).item())
    t4c_direction = float(
        direction.custom.where(cell_type='T4c', intensity=1).item() / np.pi * 180
    )
    t4c_tuning = float(
        tuning.custom.where(cell_type='T4c', intensity=1).squeeze().item()
    )
    report = {
        'status': 'complete',
        'scope': 'official FlyVis reference arm; not AxoSim inference',
        'upstream_repository': 'https://github.com/TuragaLab/flyvis',
        'upstream_commit': FLYVIS_COMMIT,
        'pretrained_archive_sha256': PRETRAINED_ZIP_SHA256,
        'network': 'flow/0000/000',
        'network_best_checkpoint_sha256': _sha256(network_path / 'best_chkpt'),
        'protocol': protocol,
        'metrics': {
            'T4c_direction_selectivity_index': t4c_dsi,
            'T4c_preferred_direction_deg': t4c_direction,
            'median_dsi_correlation_to_known': float(
                dsi_correlation_to_known(dsi).median().item()
            ),
            'T4c_tuning_curve_correlation': t4c_tuning,
        },
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    run_reference(output=args.output, root=args.root)


if __name__ == '__main__':
    main()
