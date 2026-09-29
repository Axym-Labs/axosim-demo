"""Actual Inter/PGFPlots overlay, compiled once with transparent layer export.

Requires Tectonic and Poppler's pdftocairo executable. No Matplotlib rendering.
Data are saved controller diagnostics, never invented or inferred spike rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

FONT_PATH=Path(__file__).with_name('assets')/'fonts'/'InterVariable.ttf'


def _number(value):
    return format(float(value),'.7g')


def _label(value):
    # Explicit text tick labels retain Inter instead of TeX's serif math font.
    return format(float(value),'.3g').replace('-', '−')


def _tex_escape(value):
    substitutions={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                   '#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    return ''.join(substitutions.get(c,c) for c in str(value))


def _limits(series):
    all_values=np.concatenate(series)
    finite=all_values[np.isfinite(all_values)]
    if not len(finite):raise ValueError('No finite post-warmup diagnostic samples')
    lo=float(finite.min());hi=float(finite.max());spread=hi-lo
    padding=max(spread*.08,abs(lo)*.01,.002)
    low=lo-padding;high=hi+padding
    # Three numerically legible ticks with the actual extrema still in view.
    order=10**math.floor(math.log10(max(high-low,1e-12)))
    candidates=[]
    for factor in (.1,.2,.25,.5,1.,2.,2.5,5.,10.):
        step=factor*order
        start=math.ceil(low/step)*step
        values=np.arange(start,high+step*1e-8,step).tolist()
        candidates.append((abs(len(values)-3),abs(step-(high-low)/2),values))
    ticks=min(candidates,key=lambda item:item[:2])[2]
    return low,high,ticks


def compile_neural_overlay(metadata,output_dir,*,width=320,height=600):
    """Return two exact-size RGBA layer paths and progressive-reveal geometry.

    ``metadata`` is a JSON path or dict with ``controller_diagnostics`` rows.
    Coordinates are PIL pixels, origin upper left; data axes only appear in
    ``plot_rects``. PNG backgrounds are transparent. Titles/labels/axes live in
    axes.png; data paths live in lines.png; both cover the full same canvas.
    """
    if width<240 or height<420:raise ValueError('Overlay too small for readable Inter labels')
    for program in ('tectonic','pdftocairo'):
        if not shutil.which(program):raise RuntimeError(f'{program} executable required for TikZ overlays')
    if isinstance(metadata,(str,Path)):
        source=Path(metadata);meta=json.loads(source.read_text())
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        meta=metadata;source_hash=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    rows=meta['controller_diagnostics']
    # The first P4 output block is forecast padding, not observed neural activity.
    rows=[r for r in rows if float(r['time_s'])>=.004]
    if len(rows)<2:raise ValueError('Need at least two post-warmup diagnostic samples')
    time=np.array([float(r['time_s']) for r in rows])
    if not np.all(np.diff(time)>0):raise ValueError('Diagnostic timestamps must strictly increase')
    maximum=max(float(meta.get('simulated_duration_s',time[-1])),float(time[-1]))
    if not np.isfinite(time).all() or maximum<=0:raise ValueError('Invalid time extent')
    is_lif=meta.get('controller_metadata',{}).get('backend')=='lif'
    definitions=[('Soma output',('AxoSim_soma_A','AxoSim_soma_B')),
                 ('Spike fraction (4 ms)' if is_lif else 'Spike logit',('AxoSim_forecast_A',)),
                 ('State norm',('AxoSim_state_norm',))]
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    # Establish fixed plot rectangles before compiling, not from guessed crops.
    left=round(width*.17);right=width-12
    top_fraction=(.063,.372,.68);plot_height=round(height*.22)
    rectangles=[];series=[]
    for index,(_,keys) in enumerate(definitions):
        top=round(height*top_fraction[index]);rectangles.append([left,top,right,top+plot_height])
        columns=[np.array([float(row[key]) for row in rows]) for key in keys]
        if not all(np.isfinite(column).all() for column in columns):raise ValueError('Nonfinite neural diagnostic')
        series.append(columns)
    # Store the plotted numerical data next to the TeX/PDF exports.
    np.savez_compressed(output/'diagnostic-series.npz',time_s=time,
        **{key:series[i][j] for i,(_,keys) in enumerate(definitions) for j,key in enumerate(keys)})
    preamble=r'''\documentclass[border=0pt]{standalone}
\usepackage{fontspec}
\setmainfont{InterVariable.ttf}[Path=FONTDIR/]
\setsansfont{InterVariable.ttf}[Path=FONTDIR/]
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{AxymPurple}{HTML}{3F21B6}
\definecolor{AxymLight}{HTML}{8C7AD3}
\definecolor{AxymInk}{HTML}{20232A}
\definecolor{AxymGrid}{HTML}{DCE9EC}
\begin{document}
\begin{tikzpicture}[x=1bp,y=1bp]
'''.replace('FONTDIR',str(FONT_PATH.resolve().parent))
    if is_lif:
        preamble=preamble.replace('{3F21B6}','{BDBDBD}').replace('{8C7AD3}','{989898}')
    tick_times=np.linspace(0,maximum,3)
    for layer in ('axes','lines'):
        pieces=[preamble]
        for i,((title,keys),rect,columns) in enumerate(zip(definitions,rectangles,series)):
            x0,y0,x1,y1=rect
            low,high,yticks=_limits(columns)
            axis=[f'at={{({_number(x0*.75)}bp,{_number((height-y1)*.75)}bp)}}',
                  'anchor=south west','scale only axis',f'width={_number((x1-x0)*.75)}bp',
                  f'height={_number((y1-y0)*.75)}bp','xmin=0',f'xmax={_number(maximum)}',
                  f'ymin={_number(low)}',f'ymax={_number(high)}','clip=true',
                  'axis background/.style={fill=none}',
                  'tick label style={font=\\sffamily\\fontsize{10.5}{13}\\selectfont,text=AxymInk}',
                  'label style={font=\\sffamily\\fontsize{10.5}{13}\\selectfont,text=AxymInk}']
            if layer=='axes':
                axis+=['axis x line*=bottom','axis y line*=left','axis line style={AxymInk,line width=.45pt}',
                       'tick align=outside','tick style={AxymInk,line width=.4pt}',
                       'ymajorgrids','grid style={AxymGrid,line width=.35pt}',
                       'xtick={'+','.join(map(_number,tick_times))+'}',
                       'xticklabels={'+','.join(map(_label,tick_times))+'}',
                       'ytick={'+','.join(map(_number,yticks))+'}',
                       'yticklabels={'+','.join(map(_label,yticks))+'}']
                if i==2:axis+=['xlabel={Time (s)}','xlabel style={at={(axis description cs:.5,-.25)},anchor=north}']
            else:
                axis+=['hide axis','xtick=\\empty','ytick=\\empty']
            pieces.append('\\begin{axis}['+','.join(axis)+']\n')
            if layer=='lines':
                for j,values in enumerate(columns):
                    coords=' '.join(f'({_number(t)},{_number(v)})' for t,v in zip(time,values))
                    style='AxymPurple,solid' if j==0 else 'AxymLight,dashed'
                    pieces.append('\\addplot['+style+',line width=.8pt,no marks] coordinates {'+coords+'};\n')
            pieces.append('\\end{axis}\n')
            if layer=='axes':
                title_y=(height-y0+15)*.75
                pieces.append(f'\\node[anchor=west,font=\\sffamily\\fontsize{{12}}{{14}}\\selectfont,text=AxymInk,inner sep=0pt] at (0,{_number(title_y)}) {{{_tex_escape(title)}}};\n')
                if i==0:
                    # Small direct legend, no opaque box or panel chrome.
                    legend_y=(height-y0-12)*.75
                    pieces.append(f'\\node[anchor=west,font=\\sffamily\\fontsize{{8}}{{10}}\\selectfont,text=AxymPurple,inner sep=0pt] at ({_number((x0+8)*.75)},{_number(legend_y)}) {{A solid / B dashed}};\n')
        pieces.append(f'\\pgfresetboundingbox\n\\path[use as bounding box] (0,0) rectangle ({_number(width*.75)},{_number(height*.75)});\n\\end{{tikzpicture}}\n\\end{{document}}\n')
        tex=output/f'{layer}.tex';tex.write_text(''.join(pieces))
        run=subprocess.run(['tectonic','--keep-logs','--outdir',str(output.resolve()),str(tex.resolve())],capture_output=True,text=True)
        (output/f'{layer}-build.txt').write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError(f'Tectonic failed; see {output}/{layer}-build.txt')
        prefix=output/f'{layer}-hires'
        subprocess.run(['pdftocairo','-png','-transp','-singlefile','-scale-to-x',str(width*4),'-scale-to-y',str(height*4),str(output/f'{layer}.pdf'),str(prefix)],check=True,capture_output=True)
        image=Image.open(prefix.with_suffix('.png')).convert('RGBA').resize((width,height),Image.Resampling.LANCZOS)
        image.save(output/f'{layer}.png');prefix.with_suffix('.png').unlink()
        subprocess.run(['pdftocairo','-svg',str(output/f'{layer}.pdf'),str(output/f'{layer}.svg')],check=True,capture_output=True)
    result={'axes_path':str((output/'axes.png').resolve()),'lines_path':str((output/'lines.png').resolve()),
            'plot_rects':rectangles,'time_range':[0,maximum],'width':width,'height':height,
            'font_path':str(FONT_PATH.resolve()),'font_family':'Inter','backend':'lif' if is_lif else 'axosim','source_metadata_sha256':source_hash,
            'excluded_initial_padding_seconds':.004,'renderer':'Tectonic PGFPlots + Poppler alpha export',
            'transparent_background':True,'initial_missing_samples':'No fabricated zeros before first stored post-warmup diagnostic'}
    (output/'overlay.json').write_text(json.dumps(result,indent=2)+'\n')
    combined=Image.open(result['axes_path']).convert('RGBA')
    combined=Image.alpha_composite(combined,Image.open(result['lines_path']).convert('RGBA'))
    combined.save(output/'proof-transparent.png')
    white=Image.new('RGBA',combined.size,'white');white.alpha_composite(combined);white.convert('RGB').save(output/'proof-white.png')
    return result


def reveal_lines(overlay,time_seconds):
    """Reveal historical portions only; axes remain separate and unmasked."""
    image=Image.open(overlay['lines_path']).convert('RGBA')
    start,end=overlay['time_range'];fraction=np.clip((time_seconds-start)/(end-start),0,1)
    if fraction<=0:return Image.new('RGBA',image.size,(0,0,0,0))
    draw=ImageDraw.Draw(image)
    for x0,y0,x1,y1 in overlay['plot_rects']:
        edge=x0+int(round((x1-x0)*float(fraction)))
        if edge<x1:draw.rectangle((edge+1,max(0,y0-2),x1+2,min(image.height,y1+2)),fill=(0,0,0,0))
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(compile_neural_overlay(args.metadata,args.output),indent=2))


if __name__=='__main__':main()
