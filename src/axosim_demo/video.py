"""Full-frame footage with restrained Inter typography and frosted neural overlays.

Production recordings use actual TikZ/PGFPlots layers (tikz_plot.py). The small
raster panel is an explicit preview/test renderer, not the publication export.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PURPLE = '#3F21B6'
LIGHT_PURPLE = '#8C7AD3'
GRAY = '#777580'
PANEL_WIDTH, PANEL_HEIGHT = 320, 600


@lru_cache(maxsize=32)
def _font(size, bold=False):
    path=Path(__file__).parent/'assets/fonts/InterVariable.ttf'
    if path.exists():
        font=ImageFont.truetype(str(path),size)
        font.set_variation_by_axes([14,600 if bold else 400])
        return font
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


class NeuralPlotPanel:
    """Lightweight raster preview; recorded values only, transparent ground."""
    def __init__(self,duration,width=PANEL_WIDTH,height=PANEL_HEIGHT,baseline=False,population_size=4):
        self.duration,self.width,self.height=duration,width,height
        self.baseline,self.population_size=baseline,population_size
        self.rows=[]

    def update(self,time_s,diagnostics):
        self.rows.append({'time_s':float(time_s),**{
            k:float(v) for k,v in diagnostics.items()
            if np.isscalar(v) and not isinstance(v,str) and np.isfinite(v)}})

    def render(self):
        scale=2
        image=Image.new('RGBA',(self.width*scale,self.height*scale))
        draw=ImageDraw.Draw(image)
        def text(x,y,s,size=13,fill='#45434A',bold=False):
            draw.text((x*scale,y*scale),s,font=_font(size*scale,bold),fill=fill)
        def line(points,fill,width=1):
            draw.line([(x*scale,y*scale) for x,y in points],fill=fill,width=max(1,round(width*scale)))
        configs=[(['AxoSim_soma_A','AxoSim_soma_B'],'Soma output','Native units'),
                 (['AxoSim_forecast_A'],'Spike fraction' if self.baseline else 'Spike logit','4 ms average' if self.baseline else 'Native units'),
                 (['AxoSim_state_norm'],'Recurrent state','Norm')]
        for panel,(keys,title,units) in enumerate(configs):
            y=panel*190
            text(38,y+5,title,15,bold=True)
            text(38,y+26,units,11,fill='#78747F')
            left,right,top,bottom=38,self.width-12,y+54,y+149
            rows=[r for r in self.rows if r['time_s']>.004]
            vals=[r[k] for r in rows for k in keys if k in r]
            lo,hi=(min(vals),max(vals)) if vals else (-.05,.05)
            pad=max((hi-lo)*.12,.015)
            lo-=pad;hi+=pad
            for f in (0,.5,1):
                yy=bottom-f*(bottom-top)
                line([(left,yy),(right,yy)],'#D7DDE1',.5)
                value=lo+f*(hi-lo)
                text(0,yy-6,f'{value:.2g}',11)
            line([(left,top),(left,bottom),(right,bottom)],'#6B6772',.6)
            for f in (0,.5,1):
                xx=left+f*(right-left)
                line([(xx,bottom),(xx,bottom+4)],'#6B6772',.6)
                text(xx-5,bottom+8,f'{self.duration*f:g}',11)
            for ki,key in enumerate(keys):
                points=[(left+min(r['time_s']/self.duration,1)*(right-left),bottom-(r[key]-lo)/(hi-lo)*(bottom-top)) for r in rows if key in r]
                if len(points)>1:line(points,GRAY if self.baseline else (PURPLE if ki==0 else LIGHT_PURPLE),1.5)
            if panel==2:text((left+right)/2-20,bottom+29,'Time (s)',12)
        return np.asarray(image.resize((self.width,self.height),Image.Resampling.LANCZOS))

    def save(self,path):
        Image.fromarray(self.render()).save(Path(path).with_suffix('.png'))


def compose_frame(scene_rgb,panel_rgb=None,*,label='',time_s=0.,playback_speed=.25):
    """Preserve the full scene canvas. The only caption is the exact project title.

    ``label`` and timing arguments remain accepted for old callers; descriptions
    and playback information belong in the accompanying metadata/document now.
    """
    scene=Image.fromarray(np.asarray(scene_rgb,dtype=np.uint8)).convert('RGBA')
    if panel_rgb is not None:
        panel=Image.fromarray(np.asarray(panel_rgb,dtype=np.uint8)).convert('RGBA')
        padding=16
        x=scene.width-panel.width-32
        y=(scene.height-panel.height)//2
        box=(x-padding,y-padding,x+panel.width+padding,y+panel.height+padding)
        mask=Image.new('L',scene.size,0)
        ImageDraw.Draw(mask).rounded_rectangle(box,radius=22,fill=255)
        blurred=scene.filter(ImageFilter.GaussianBlur(15))
        glass=Image.new('RGBA',scene.size,(255,255,255,212))
        glass=Image.alpha_composite(blurred,glass)
        scene.paste(glass,(0,0),mask)
        edge=Image.new('RGBA',scene.size)
        ImageDraw.Draw(edge).rounded_rectangle(box,radius=22,outline=(255,255,255,150),width=1)
        scene=Image.alpha_composite(scene,edge)
        scene.alpha_composite(panel,(x,y))
    # Soft local shadow supports the title over foliage without a title card.
    title='AxoSim - Axym Labs'
    xy=(30,scene.height-54)
    shadow=Image.new('RGBA',scene.size)
    ImageDraw.Draw(shadow).text(xy,title,font=_font(25,True),fill=(0,0,0,240),stroke_width=2)
    scene=Image.alpha_composite(scene,shadow.filter(ImageFilter.GaussianBlur(4)))
    ImageDraw.Draw(scene).text(xy,title,font=_font(25,True),fill='white')
    return np.asarray(scene.convert('RGB'))
