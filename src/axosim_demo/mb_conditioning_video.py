"""Render the measured full-connectome AxoSim conditioning result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tikz_plot import FONT_PATH, _number


WIDTH, HEIGHT = 1280, 720
INK = (32, 35, 42)
MUTED = (104, 108, 117)
PURPLE = (63, 33, 182)
PURPLE_LIGHT = (140, 122, 211)
BLUE = (53, 117, 166)
AMBER = (203, 128, 27)
PALE = (226, 233, 236)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size, *, weight=400):
    font = ImageFont.truetype(str(FONT_PATH), size=size)
    try:
        font.set_variation_by_axes([14, weight])
    except (AttributeError, OSError):
        pass
    return font


def _compile_effect_plot(summary, output, *, width=430, height=335):
    output = Path(output)
    for program in ("tectonic", "pdftocairo"):
        if not shutil.which(program):
            raise RuntimeError(f"{program} is required for the TikZ effect plot")
    arms = ("standard", "reversed", "no_plasticity", "dopamine_blocked")
    labels = ("standard", "reversed", "no plasticity", "DAN clamp off")
    analysis = summary["analysis"]["arms"]
    all_values = np.concatenate([analysis[name]["delta_D"] for name in arms])
    extent = max(0.2, float(np.max(np.abs(all_values))) * 1.18)
    tex = r'''\documentclass[border=0pt]{standalone}
\usepackage{fontspec}
\setsansfont{InterVariable.ttf}[Path=FONTDIR/]
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{AxymPurple}{HTML}{3F21B6}
\definecolor{AxymLight}{HTML}{8C7AD3}
\definecolor{Control}{HTML}{777B83}
\definecolor{Ink}{HTML}{20232A}
\definecolor{Grid}{HTML}{DCE9EC}
\begin{document}\begin{tikzpicture}[x=1bp,y=1bp]
\begin{axis}[
at={(48bp,38bp)},anchor=south west,scale only axis,width=8.5cm,height=5.25cm,
xmin=.5,xmax=4.5,ymin=YMIN,ymax=YMAX,clip=false,
axis x line*=bottom,axis y line*=left,axis line style={Ink,line width=.45pt},
tick align=outside,tick style={Ink,line width=.4pt},
tick label style={font=\sffamily\fontsize{9.5}{11}\selectfont,text=Ink},
label style={font=\sffamily\fontsize{10}{12}\selectfont,text=Ink},
xtick={1,2,3,4},xticklabels={LABELS},
x tick label style={rotate=15,anchor=north east},
ymajorgrids,grid style={Grid,line width=.35pt},ylabel={$\Delta D$ · neural score},
extra y ticks={0},extra y tick labels={},extra y tick style={grid=major,grid style={Ink,line width=.55pt}}]
'''.replace("FONTDIR", str(FONT_PATH.resolve().parent)).replace(
        "YMIN", _number(-extent)
    ).replace("YMAX", _number(extent)).replace("LABELS", ",".join(labels))
    rng = np.random.default_rng(20260930)
    for x, arm in enumerate(arms, start=1):
        color = "AxymPurple" if arm in ("standard", "reversed") else "Control"
        values = np.asarray(analysis[arm]["delta_D"], dtype=np.float64)
        jitter = np.linspace(-0.11, 0.11, len(values)) + rng.normal(0, 0.008, len(values))
        coords = " ".join(
            f"({_number(x + dx)},{_number(value)})"
            for dx, value in zip(jitter, values)
        )
        mean = analysis[arm]["mean_delta_D"]
        ci = analysis[arm]["bootstrap_95_ci"]
        tex += f"\\addplot[{color},only marks,mark=*,mark size=2pt] coordinates {{{coords}}};\n"
        tex += (
            f"\\draw[{color},line width=.8pt] (axis cs:{x},{_number(ci[0])}) -- "
            f"(axis cs:{x},{_number(ci[1])});\n"
        )
        tex += (
            f"\\draw[{color},line width=1.4pt] (axis cs:{_number(x-.17)},{_number(mean)}) -- "
            f"(axis cs:{_number(x+.17)},{_number(mean)});\n"
        )
    tex += rf"\end{{axis}}\pgfresetboundingbox\path[use as bounding box] (0,0) rectangle ({width*.75},{height*.75});" + "\n"
    tex += "\\end{tikzpicture}\\end{document}\n"
    path = output / "conditioning-effect.tex"
    path.write_text(tex)
    run = subprocess.run(
        ["tectonic", "--keep-logs", "--outdir", str(output.resolve()), str(path.resolve())],
        capture_output=True,
        text=True,
    )
    (output / "conditioning-effect-build.txt").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError("Tectonic failed; see conditioning-effect-build.txt")
    subprocess.run(
        ["pdftocairo", "-svg", str(output / "conditioning-effect.pdf"), str(output / "conditioning-effect.svg")],
        check=True,
        capture_output=True,
    )
    prefix = output / "conditioning-effect-hires"
    subprocess.run(
        ["pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", str(width*3), "-scale-to-y", str(height*3), str(output/"conditioning-effect.pdf"), str(prefix)],
        check=True,
        capture_output=True,
    )
    image = Image.open(prefix.with_suffix(".png")).convert("RGBA").resize(
        (width, height), Image.Resampling.LANCZOS
    )
    image.save(output / "conditioning-effect.png")
    prefix.with_suffix(".png").unlink()
    return image


def _curve(draw, points, *, fill, width):
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in points)
    t = np.linspace(0, 1, 48)[:, None]
    curve = (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )
    draw.line([tuple(x) for x in curve], fill=fill, width=width, joint="curve")


def _interpolate_history(history, phase):
    position = np.clip(phase * len(history), 0, len(history) - 1e-9)
    index = int(position)
    alpha = position - index
    previous = history[max(index - 1, 0)]
    current = history[index]
    efficacy = {}
    for name in ("all", "PPL105_core", "PAM08_core", "min"):
        efficacy[name] = (
            (1 - alpha) * previous["efficacy"][name]
            + alpha * current["efficacy"][name]
        )
    return index, current, efficacy


def _circuit_frame(summary, history, phase, effect_plot):
    index, current, efficacy = _interpolate_history(history, phase)
    image = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    heading = _font(14, weight=600)
    label = _font(16, weight=600)
    small = _font(13)
    metric = _font(18, weight=600)
    title = _font(25, weight=600)

    draw.text((42, 42), "CONNECTOME-CONSTRAINED CONDITIONING", font=heading, fill=MUTED)
    draw.text((42, 68), "actual AxoSim inference · schematic circuit view", font=small, fill=MUTED)

    odor_active = current["odor"]
    dan_active = current["dan_type"]
    # Sensory receptor sources.
    draw.ellipse((78, 250, 174, 346), fill=(246, 244, 253), outline=PURPLE, width=2)
    draw.text((104, 276), odor_active, font=label, fill=PURPLE)
    draw.text((74, 359), "olfactory receptor neurons", font=small, fill=MUTED)

    # A compact sample of the real 4,064-cell KC population; explicitly schematic.
    rng = np.random.default_rng(19)
    kc_points = np.column_stack((rng.uniform(290, 520, 90), rng.uniform(205, 462, 90)))
    for x, y in kc_points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(*PURPLE_LIGHT, 165))
    draw.text((335, 475), "4,064 Kenyon cells", font=label, fill=INK)
    draw.text((332, 499), "33,496 plastic pair edges", font=small, fill=MUTED)

    # Input and two compartment streams.
    _curve(draw, ((174, 298), (225, 298), (250, 333), (290, 333)), fill=(*PURPLE, 180), width=3)
    counts = [current["counts_PPL105"], current["counts_PAM08"]]
    max_count = max(1, max(max(row["counts_PPL105"], row["counts_PAM08"]) for row in history))
    mbon_nodes = ((652, 239, "PPL105 core", BLUE), (652, 397, "PAM08 core", AMBER))
    for j, (x, y, name, color) in enumerate(mbon_nodes):
        pulse = counts[j] / max_count
        radius = 34 + int(9 * pulse)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(*color, 42 + int(90*pulse)), outline=color, width=2)
        draw.text((x + 50, y - 21), name, font=label, fill=INK)
        draw.text((x + 50, y + 5), f"{counts[j]:,} emitted events", font=small, fill=MUTED)
    # Measured efficacy controls line opacity; strands are schematic, never a spatial graph.
    for j, (x, y, _, color) in enumerate(mbon_nodes):
        eff = efficacy["PPL105_core" if j == 0 else "PAM08_core"]
        opacity = int(45 + 180 * np.clip(eff, 0, 1))
        for offset in np.linspace(-70, 70, 9):
            _curve(
                draw,
                ((512, 333 + offset*.7), (560, 333 + offset*.5), (585, y + offset*.16), (x-35, y)),
                fill=(*color, opacity),
                width=2,
            )

    # Direct dopamine-to-compartment learning route.
    dan_positions = {"PPL105": (420, 129, BLUE), "PAM08": (420, 557, AMBER)}
    for name, (x, y, color) in dan_positions.items():
        active = name == dan_active
        radius = 25 if active else 20
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(*color, 135 if active else 28), outline=color, width=2)
        draw.text((x + 36, y - 11), name, font=label, fill=color if active else MUTED)
    _curve(draw, ((444, 129), (560, 125), (603, 185), (635, 220)), fill=(*BLUE, 205 if dan_active == "PPL105" else 65), width=3)
    _curve(draw, ((444, 557), (560, 560), (603, 451), (635, 416)), fill=(*AMBER, 205 if dan_active == "PAM08" else 65), width=3)
    draw.text((42, 545), f"trial  {current['trial']:02d} / 12", font=metric, fill=INK)
    draw.text((42, 578), f"paired input  {odor_active} + {dan_active or 'no dopamine'}", font=small, fill=MUTED)
    draw.text((42, 608), f"mean KC→MBON efficacy  {efficacy['all']:.4f}", font=small, fill=INK)

    # Seamless result overlay on the right.
    draw.text((820, 43), "CONTINGENCY REVERSAL", font=heading, fill=MUTED)
    verdict = "gate passed" if summary["passed"] else "negative result"
    draw.text((820, 69), verdict, font=metric, fill=PURPLE if summary["passed"] else INK)
    image.alpha_composite(effect_plot, (800, 105))
    analysis = summary["analysis"]
    standard = analysis["arms"]["standard"]["mean_delta_D"]
    reversed_value = analysis["arms"]["reversed"]["mean_delta_D"]
    draw.text((835, 461), f"standard  mean ΔD {standard:+.3f}", font=small, fill=INK)
    draw.text((835, 487), f"reversed  mean ΔD {reversed_value:+.3f}", font=small, fill=INK)
    draw.text((835, 513), f"opposite-sign pairs  {analysis['trained_opposite_sign_pairs']} / 8", font=small, fill=INK)
    draw.text((835, 552), "neural endpoint · no motor decoder", font=small, fill=MUTED)
    draw.text((835, 577), "8 independently trained replicates", font=small, fill=MUTED)

    draw.rounded_rectangle((22, 655, 337, 709), radius=13, fill=(255, 255, 255, 235))
    draw.text((42, 672), "AxoSim - Axym Labs", font=title, fill=INK)
    return image


def render_video(summary_path, output, *, fps=30, seconds=12):
    summary_path, output = Path(summary_path), Path(output)
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "complete":
        raise ValueError("Conditioning experiment is incomplete")
    if summary["graph"]["pruned"] or summary["graph"]["coalesced"]:
        raise ValueError("Conditioning film requires the retained full graph")
    history = summary["results"]["standard"][0]["history"]
    if len(history) != 24:
        raise ValueError("Expected 12 two-odor conditioning trials")
    output.mkdir(parents=True, exist_ok=True)
    effect_plot = _compile_effect_plot(summary, output)
    frames = int(round(fps * seconds))
    writer = imageio.get_writer(
        output / "dopamine-gated-conditioning.mp4",
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=1,
    )
    try:
        for frame in range(frames):
            # Ten seconds of training followed by a two-second measured-result hold.
            phase = min(frame / max(fps * 10 - 1, 1), 1.0)
            image = _circuit_frame(summary, history, phase, effect_plot)
            writer.append_data(np.asarray(image.convert("RGB")))
            if frame in (0, frames // 2, frames - 1):
                image.convert("RGB").save(output / f"frame-{frame:03d}.png")
    finally:
        writer.close()
    metadata = {
        "title": "AxoSim - Axym Labs",
        "video": "dopamine-gated-conditioning.mp4",
        "frames": frames,
        "fps": fps,
        "seconds": seconds,
        "resolution": [WIDTH, HEIGHT],
        "summary_sha256": _sha256(summary_path),
        "passed": summary["passed"],
        "scope": summary["scope"],
        "graph": summary["graph"],
        "analysis": summary["analysis"],
        "display": "schematic circuit geometry driven by measured standard-arm seed-1000 history; geometry is not anatomical position",
        "plot": "Tectonic/PGFPlots; every replicate, bootstrap interval, and mean shown",
        "controller_or_motor_output": False,
        "future_values_shown": "final aggregate plot is visible throughout; animated efficacy uses only its current or previous presentation",
    }
    (output / "visual-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render_video(args.summary, args.output), indent=2))


if __name__ == "__main__":
    main()
