"""Render fixed first validation trace with actual Inter/PGFPlots layers.

No model inference, data selection, smoothing, or metric recomputation occurs.
The input is the saved retrospective mammalian validation panel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .tikz_plot import FONT_PATH, _number

WIDTH, HEIGHT = 1280, 720
MODELS = [('lif_joint', 'LIF · joint fit', 'Baseline'),
          ('axosim_lite', 'AxoSim-Lite', 'AxymPurple')]


def _node(x, y, text, size=12, color='Ink', anchor='west'):
    return (rf'\node[anchor={anchor},font=\sffamily\fontsize{{{size}}}{{{size*1.2}}}\selectfont,'
            rf'text={color},inner sep=0pt] at ({x*.75},{(HEIGHT-y)*.75}) {{{text}}};'+'\n')


def render_evidence(source, output):
    source, output = Path(source), Path(output)
    for program in ('tectonic', 'pdftocairo'):
        if not shutil.which(program):
            raise RuntimeError(f'{program} executable required for actual TikZ evidence rendering')
    output.mkdir(parents=True, exist_ok=True)
    data_path = source/'val-predictions.npz'; result_path = source/'results.json'
    with np.load(data_path) as loaded:
        data = {k: loaded[k] for k in loaded.files}
    models = json.loads(result_path.read_text())['splits']['val']['models']
    time = np.arange(data['target_clipped_mv'].shape[1])/1000
    mask = time >= .1
    if data['target_clipped_mv'].shape != (4, 2000):
        raise ValueError('Expected the fixed four-trace, 2000-sample validation panel')
    rectangles = []
    preamble = r'''\documentclass[border=0pt]{standalone}
\usepackage{fontspec}
\setsansfont{InterVariable.ttf}[Path=FONTDIR/]
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{Ink}{HTML}{20232A}
\definecolor{Teacher}{HTML}{111111}
\definecolor{Baseline}{HTML}{BDBDBD}
\definecolor{AxymPurple}{HTML}{3F21B6}
\definecolor{Grid}{HTML}{DCE9EC}
\begin{document}
\begin{tikzpicture}[x=1bp,y=1bp]
'''.replace('FONTDIR', str(FONT_PATH.resolve().parent))
    for layer in ('axes', 'lines'):
        pieces = [preamble]
        for col, (name, title, color) in enumerate(MODELS):
            x0, x1 = ((96, 600) if col == 0 else (736, 1240))
            if layer == 'axes':
                pieces.append(_node(x0, 44, title, 18, 'Ink' if col == 0 else color))
                pieces.append(_node(x0, 83, 'Soma voltage (mV)', 10.5))
                # Both panels use exactly the same teacher trace and target scale.
                pieces.append(_node(x1, 44, 'NEURON reference', 10.5, 'Teacher', 'east'))
                pieces.append(rf'\draw[Teacher,line width=.8pt] ({(x1-157)*.75},{(HEIGHT-44)*.75}) -- ({(x1-139)*.75},{(HEIGHT-44)*.75});'+'\n')
                metric = models[name]['per_trace'][0]
                pieces.append(_node(x0, 620,
                    f"RMSE {metric['clipped_target_rmse_mv']:.3f} mV   ·   F1 ±5 ms {metric['f1_5ms']:.3f}", 12))
            for kind, y0, y1 in (('voltage', 106, 405), ('events', 466, 535)):
                rect = [x0, y0, x1, y1]
                if layer == 'axes': rectangles.append(rect)
                options = [f'at={{({x0*.75}bp,{(HEIGHT-y1)*.75}bp)}}',
                    'anchor=south west', 'scale only axis', f'width={(x1-x0)*.75}bp',
                    f'height={(y1-y0)*.75}bp', 'xmin=.1', 'xmax=2', 'clip=true',
                    'axis background/.style={fill=none}',
                    'tick label style={font=\\sffamily\\fontsize{10.5}{13}\\selectfont,text=Ink}',
                    'label style={font=\\sffamily\\fontsize{12}{14}\\selectfont,text=Ink}']
                options += ['ymin=-80','ymax=-50'] if kind == 'voltage' else ['ymin=-.55','ymax=1.55']
                if layer == 'axes':
                    options += ['axis x line*=bottom','axis y line*=left',
                        'axis line style={Ink,line width=.45pt}', 'tick align=outside',
                        'tick style={Ink,line width=.4pt}']
                    if kind == 'voltage':
                        options += ['xtick=\\empty','ytick={-80,-70,-60,-50}',
                            'yticklabels={−80,−70,−60,−50}', 'ymajorgrids',
                            'grid style={Grid,line width=.35pt}']
                    else:
                        options += ['ytick={0,1}', 'yticklabels={Model,NEURON}',
                            'xtick={.5,1,1.5,2}', 'xticklabels={0.5,1,1.5,2}',
                            'xlabel={Time (s)}', 'xlabel style={at={(axis description cs:.5,-.45)},anchor=north}']
                else:
                    options += ['hide axis','xtick=\\empty','ytick=\\empty']
                pieces.append('\\begin{axis}['+','.join(options)+']\n')
                if layer == 'lines':
                    if kind == 'voltage':
                        for key, style in ((f'{name}_mv', f'{color},line width=1.05pt'+(',dashed' if col == 0 else '')),
                                           ('target_clipped_mv', 'Teacher,line width=.65pt')):
                            coords = ' '.join(f'({_number(t)},{_number(v)})' for t,v in zip(time[mask],data[key][0,mask]))
                            pieces.append(r'\addplot['+style+',no marks] coordinates {'+coords+'};\n')
                    else:
                        for key, level, style in (('target_spikes',1,'Teacher'),(f'{name}_spikes',0,color)):
                            times = time[mask & data[key][0].astype(bool)]
                            coords = ' '.join(f'({_number(t)},{level})' for t in times)
                            pieces.append(r'\addplot['+style+',only marks,mark=|,mark size=4pt,line width=1pt] coordinates {'+coords+'};\n')
                pieces.append('\\end{axis}\n')
        if layer == 'axes':
            pieces.append(_node(96, 683, 'AxoSim - Axym Labs', 17))
        pieces.append(r'\pgfresetboundingbox'+'\n'+rf'\path[use as bounding box] (0,0) rectangle ({WIDTH*.75},{HEIGHT*.75});'+'\n'+r'\end{tikzpicture}'+'\n'+r'\end{document}'+'\n')
        tex = output/f'{layer}.tex';tex.write_text(''.join(pieces))
        run = subprocess.run(['tectonic','--keep-logs','--outdir',str(output.resolve()),str(tex.resolve())],capture_output=True,text=True)
        (output/f'{layer}-build.txt').write_text(run.stdout+run.stderr)
        if run.returncode: raise RuntimeError(f'Tectonic failed: {output}/{layer}-build.txt')
        prefix = output/f'{layer}-hires'
        subprocess.run(['pdftocairo','-png','-transp','-singlefile','-scale-to-x',str(WIDTH*2),'-scale-to-y',str(HEIGHT*2),str(output/f'{layer}.pdf'),str(prefix)],check=True,capture_output=True)
        raster = Image.open(prefix.with_suffix('.png')).convert('RGBA').resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)
        raster.save(output/f'{layer}.png');prefix.with_suffix('.png').unlink()
        subprocess.run(['pdftocairo','-svg',str(output/f'{layer}.pdf'),str(output/f'{layer}.svg')],check=True,capture_output=True)
    axes = Image.open(output/'axes.png').convert('RGBA')
    lines = Image.open(output/'lines.png').convert('RGBA')
    proof = Image.new('RGBA',(WIDTH,HEIGHT),'white');proof.alpha_composite(axes);proof.alpha_composite(lines)
    proof.convert('RGB').save(output/'neuron-fidelity.png')
    # Full static vector version remains an actual TeX figure as well.
    # Overlay PDFs in one standalone TeX picture at their exact shared origin.
    combined = r'''\documentclass[border=0pt]{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}
\node[inner sep=0pt,anchor=south west] at (0,0) {\includegraphics{axes.pdf}};
\node[inner sep=0pt,anchor=south west] at (0,0) {\includegraphics{lines.pdf}};
\end{tikzpicture}
\end{document}
'''
    (output/'neuron-fidelity.tex').write_text(combined)
    run = subprocess.run(['tectonic','--outdir',str(output.resolve()),str((output/'neuron-fidelity.tex').resolve())],cwd=output,capture_output=True,text=True)
    if run.returncode: raise RuntimeError(run.stdout+run.stderr)
    subprocess.run(['pdftocairo','-svg',str(output/'neuron-fidelity.pdf'),str(output/'neuron-fidelity.svg')],check=True,capture_output=True)
    writer = imageio.get_writer(output/'neuron-fidelity-comparison.mp4',fps=30,codec='libx264',quality=8,macro_block_size=1)
    try:
        for frame in range(240):
            fraction = frame/239
            visible = lines.copy() if frame else Image.new('RGBA',lines.size)
            draw = ImageDraw.Draw(visible)
            if frame < 239:
                for x0,y0,x1,y1 in rectangles:
                    edge = x0+round((x1-x0)*fraction)
                    draw.rectangle((edge+1,y0-5,x1+5,y1+5),fill=(0,0,0,0))
            image = Image.new('RGBA',(WIDTH,HEIGHT),'white');image.alpha_composite(axes);image.alpha_composite(visible)
            writer.append_data(np.asarray(image.convert('RGB')))
            if frame in (0,120,239): image.convert('RGB').save(output/f'frame-{frame:03d}.png')
    finally: writer.close()
    metadata = {'trace_index':0,'selection':'same first validation trace as revision 2; no reselection',
        'source_predictions_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'source_results_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'frames':240,'fps':30,'video_seconds':8,'resolution':[WIDTH,HEIGHT],
        'shared_axes':{'time_seconds':[.1,2],'voltage_mv':[-80,-50]},'plot_rects':rectangles,
        'displayed_metrics':{name:models[name]['per_trace'][0] for name,_,_ in MODELS},
        'teacher':'Identical NEURON soma target clipped at -55 mV, black in both panels',
        'renderer':'Tectonic PGFPlots; Inter; static alpha layers progressively revealed',
        'title':'AxoSim - Axym Labs','scope':'retrospective mammalian validation; not fly fidelity',
        'warmup_excluded_ms':100,'metrics_use':'entire post-warmup first validation trace',
        'future_data_shown':False}
    (output/'visual-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(render_evidence(args.source,args.output),indent=2))


if __name__=='__main__':main()
