"""Readable video composition using real pose, commands, and backend telemetry."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = '#0e171c'
PANEL = '#18272c'
INK = '#edf3ed'
MUTED = '#9db1ac'
CYAN = '#70d5c1'
AMBER = '#efb06a'


@lru_cache(maxsize=20)
def _font(size, bold=False):
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf')
    return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)


def compose_frame(third, first, time_s, command, path, observation, diagnostics,
                  label, playback_speed, speed):
    canvas = Image.new('RGB', (1280, 720), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), 'AXOSIM', font=_font(23, True), fill=INK)
    draw.text((152, 20), '/  EMBODIED FLY', font=_font(18), fill=MUTED)
    draw.text((24, 51), label[:82], font=_font(14), fill=CYAN)
    draw.text((1022, 22), f't = {time_s:5.3f} s', font=_font(20), fill=INK)
    draw.text((1022, 50), f'{playback_speed:g}x playback', font=_font(13), fill=MUTED)
    canvas.paste(Image.fromarray(third), (20, 86))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((32, 98, 222, 126), radius=5, fill=BG)
    draw.text((43, 103), '01  /  ARTICULATED BODY', font=_font(12, True), fill=INK)
    draw.rectangle((968, 86, 1260, 654), fill=PANEL)
    canvas.paste(Image.fromarray(first), (968, 86))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((968, 226, 1260, 250), fill=BG)
    draw.text((980, 230), '02 / FIRST PERSON · perspective', font=_font(12), fill=INK)
    draw.text((983, 260), 'Not compound-eye neural input', font=_font(12), fill=MUTED)
    draw.text((983, 293), 'DESCENDING DRIVE', font=_font(12, True), fill=MUTED)
    for j, (side, color) in enumerate(zip(('L', 'R'), (CYAN, AMBER))):
        y = 319 + j*29
        draw.text((983, y), side, font=_font(14, True), fill=color)
        draw.rounded_rectangle((1005, y+3, 1170, y+15), radius=4, fill=BG)
        magnitude = min(1., abs(float(command[j]))/2)
        if magnitude > 0:
            draw.rounded_rectangle((1005, y+3, 1005+165*magnitude, y+15), radius=4, fill=color)
        draw.text((1183, y-1), f'{command[j]:.2f}', font=_font(14), fill=INK)
    draw.text((983, 389), 'ARENA TRAJECTORY  /  mm', font=_font(12, True), fill=MUTED)
    bounds = (990, 420, 1238, 535)
    draw.rounded_rectangle(bounds, radius=5, fill=BG)
    def point(xy):
        return (float(1114 + 5.4*xy[0]), float(477-5.4*xy[1]))
    draw.line((990, 477, 1238, 477), fill='#2b3a3d')
    draw.line((1114, 420, 1114, 535), fill='#2b3a3d')
    for source, color in zip(observation['source_positions_mm'], (AMBER, CYAN)):
        x, y = point(source)
        draw.ellipse((x-4,y-4,x+4,y+4), fill=color)
    samples = path[::max(1,len(path)//200)]
    points = [point(p) for p in samples]
    if len(points) > 1:
        draw.line(points, fill=INK, width=2)
    x,y = point(path[-1])
    draw.ellipse((x-3,y-3,x+3,y+3), fill=CYAN)
    draw.text((983, 548), f'Speed   {speed:5.1f} mm/s', font=_font(14), fill=INK)
    priority = ('learned_value_A', 'learned_value_B', 'AxoSim_soma_A')
    keys = [k for k in priority if k in diagnostics] + [k for k in diagnostics if k not in priority]
    scalar_metrics = [(k, diagnostics[k]) for k in keys if np.isscalar(diagnostics[k]) and not isinstance(diagnostics[k], str)]
    metric_labels = {'learned_value_A': 'Learned A (a.u.)', 'learned_value_B': 'Learned B (a.u.)', 'AxoSim_soma_A': 'Soma A (native units)'}
    for i,(key,value) in enumerate(scalar_metrics[:3]):
        draw.text((983, 578+21*i), f'{metric_labels.get(key, key[:24])}  {float(value):.3g}', font=_font(12), fill=MUTED)
    if not scalar_metrics:
        draw.text((983, 579), 'Hybrid CPG + contact reflexes', font=_font(12), fill=MUTED)
        draw.text((983, 600), 'No neural activity in this baseline', font=_font(11), fill=MUTED)
    draw.text((24, 670), 'DROSOPHILA MELANOGASTER', font=_font(13, True), fill=INK)
    draw.text((24, 692), 'NeuroMechFly body / MuJoCo physics / bilateral descending drive', font=_font(11), fill=MUTED)
    draw.text((855, 675), 'Amber / teal: synthetic odor landmarks', font=_font(12), fill=MUTED)
    draw.text((855, 696), 'Actual joint kinematics and body telemetry', font=_font(11), fill=MUTED)
    return np.asarray(canvas)
