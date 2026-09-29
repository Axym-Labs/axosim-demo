"""Composite full-frame footage with actual TikZ neural traces and a single title."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .video import NeuralPlotPanel, compose_frame


def recompose(directory, *, renderer='tikz', raw_filename='scene.mp4'):
    directory=Path(directory)
    metadata=json.loads((directory/'metadata.json').read_text())
    rows=metadata['controller_diagnostics']
    if len(rows)!=metadata['frames']:
        raise ValueError('One recorded diagnostic row is required for each frame')
    has_neural=any('AxoSim_soma_A' in r for r in rows)
    overlay=axes=panel=None
    if has_neural:
        if renderer=='tikz':
            from .tikz_plot import compile_neural_overlay, reveal_lines
            overlay=compile_neural_overlay(metadata,directory/'tikz-overlay')
            axes=Image.open(overlay['axes_path']).convert('RGBA')
        elif renderer=='raster':
            panel=NeuralPlotPanel(metadata['simulated_duration_s'],
                baseline=metadata['controller_metadata'].get('backend')=='lif')
        else:
            raise ValueError('Renderer must be tikz or raster')
    raw=imageio.get_reader(directory/raw_filename)
    temporary=directory/'recomposed.mp4'
    writer=imageio.get_writer(temporary,fps=metadata['fps'],codec='libx264',quality=9,
                             macro_block_size=1,ffmpeg_log_level='error')
    try:
        for i,row in enumerate(rows):
            scene=raw.get_data(i)
            if scene.shape[:2]!=(720,1280):
                raise ValueError('New overlay layout requires full-frame 1280x720 source footage')
            plot=None
            if overlay is not None:
                plot=np.asarray(Image.alpha_composite(axes,reveal_lines(overlay,row['time_s'])))
            elif panel is not None:
                panel.update(row['time_s'],row)
                plot=panel.render()
            frame=compose_frame(scene,plot)
            writer.append_data(frame)
            if i in (0,metadata['frames']//2,metadata['frames']-1):
                name='preview.png' if i==0 else ('midpoint.png' if i==metadata['frames']//2 else 'final.png')
                Image.fromarray(frame).save(directory/name)
    finally:
        raw.close();writer.close()
    temporary.replace(directory/'demo.mp4')
    if panel is not None:panel.save(directory/'neural_traces')
    metadata['plot_display']={
        'renderer':renderer if has_neural else None,'full_frame_scene':True,
        'transparent_plot_ground':'frosted translucent white local backdrop',
        'font':'Inter','description_inside_video':False,
        'panels':['native soma A/B','native spike logit A' if metadata['controller_metadata'].get('backend')!='lif' else 'spike fraction A per 4 ms','recurrent state norm'] if has_neural else [],
        'forecast_warmup_excluded_seconds':.004,'traces_fly_index':0,
        'compiled_plot_limits':'whole recorded run; animation reveals past samples only',
        'recomposed_from_saved_raw_frames':True}
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return directory/'demo.mp4'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',type=Path,nargs='+')
    parser.add_argument('--renderer',choices=['tikz','raster'],default='tikz')
    args=parser.parse_args()
    for directory in args.directories:print(recompose(directory,renderer=args.renderer),flush=True)


if __name__=='__main__':main()
