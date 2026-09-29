"""Compose matched scene recordings, LIF left and AxoSim right."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .video import _font


def compose_comparison(lif_dir, axosim_dir, output):
    lif_dir, axosim_dir, output = map(Path, (lif_dir, axosim_dir, output))
    metadata = [json.loads((p/'metadata.json').read_text()) for p in (lif_dir, axosim_dir)]
    for key in ('seed', 'frames', 'fps', 'simulated_duration_s', 'view', 'num_flies', 'camera', 'sources_mm',
                'physics_timestep_s', 'playback_speed', 'odor_model', 'body', 'scene'):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f'Comparison mismatch: {key}')
    if metadata[0]['controller_metadata'].get('backend') != 'lif':
        raise ValueError('Left video must be the declared LIF substitution')
    if 'checkpoint_sha256' not in metadata[1]['controller_metadata']:
        raise ValueError('Right video must identify its trained AxoSim checkpoint')
    if not metadata[0]['controller_metadata'].get('contact_log_efficacy_sha256'):
        raise ValueError('Comparison requires contact efficacy hashes; re-record with current code')
    for key in ('conditioning_arm', 'conditioning_seed', 'rewarded_cue', 'conditioned_weights_loaded', 'contact_log_efficacy_sha256'):
        if metadata[0]['controller_metadata'].get(key) != metadata[1]['controller_metadata'].get(key):
            raise ValueError(f'Controller comparison mismatch: {key}')
    output.mkdir(parents=True, exist_ok=True)
    readers = [imageio.get_reader(p/'scene.mp4') for p in (lif_dir, axosim_dir)]
    try:
        with imageio.get_writer(output/'comparison.mp4', fps=metadata[0]['fps'], codec='libx264',
                                quality=8, macro_block_size=1, ffmpeg_log_level='error') as writer:
            for i in range(metadata[0]['frames']):
                frames = [r.get_data(i) for r in readers]
                canvas = Image.fromarray(np.concatenate(frames, axis=1))
                draw = ImageDraw.Draw(canvas)
                width, height = frames[0].shape[1], frames[0].shape[0]
                def text(x,y,value,size):
                    draw.text((x,y),value,font=_font(size),fill='white',stroke_width=1,stroke_fill='#231E18')
                text(18, height-74, 'LIF baseline', 25)
                text(width+18, height-74, 'AxoSim-Lite', 25)
                text(18, height-44, 'Same reduced circuit, sensory mapping and motor decoder', 15)
                text(width+18, height-44, 'Trained surrogate; identical initial contact efficacies', 15)
                text(18, height-23, 'AxoSim - Axym Labs', 17)
                text(18, 14, 'Exploratory model substitution', 17)
                writer.append_data(np.asarray(canvas))
                if i == metadata[0]['frames']//2:
                    canvas.save(output/'preview.png')
    finally:
        for reader in readers:
            reader.close()
    result = {'left':str(lif_dir), 'right':str(axosim_dir), 'frames':metadata[0]['frames'],
              'fps':metadata[0]['fps'], 'claim':'Exploratory reduced-circuit model substitution; no demonstrated biological fly fidelity advantage.',
              'controls':'Same initial scene, seed, synthetic odor fields, contacts, efficacy changes, four-ms output latency, decoder and gait. Closed-loop inputs diverge after trajectories diverge.',
              'baseline':'Shiu-equation LIF substitution; not original full-brain Shiu simulation.'}
    (output/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    return output/'comparison.mp4'


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lif', type=Path, required=True)
    parser.add_argument('--axosim', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    print(compose_comparison(args.lif,args.axosim,args.output))
