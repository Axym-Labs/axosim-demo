"""Render the exact whole-connectome AxoSim scale run and its negative specificity result."""
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
PURPLE = np.array([63, 33, 182], dtype=np.float64)
INACTIVE = np.array([237, 241, 242], dtype=np.float64)


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


def _rates(axosim_summary, reference):
    result = {}
    for model, rows in (("reference", reference["results"]), ("axosim", axosim_summary["results"])):
        for condition in ("jo_ce", "jo_f"):
            selected = [row for row in rows if row["condition"] == condition]
            selected.sort(key=lambda row: row["seed"])
            if model == "reference":
                values = [row["readouts"]["aBN1"]["rate_hz"] for row in selected]
            else:
                values = [row["readout_rate_hz"]["aBN1"] for row in selected]
            if len(values) != 3:
                raise ValueError(f"Expected three fixed seeds for {model} {condition}")
            result[f"{model}_{condition}"] = values
    return result


def _compile_specificity_plot(rates, output, *, width=375, height=270):
    output = Path(output)
    for program in ("tectonic", "pdftocairo"):
        if not shutil.which(program):
            raise RuntimeError(f"{program} is required for the TikZ specificity plot")
    groups = ["reference_jo_ce", "reference_jo_f", "axosim_jo_ce", "axosim_jo_f"]
    colors = ["Baseline", "Baseline", "AxymPurple", "AxymPurple"]
    labels = ["Ref. C/E", "Ref. F", "Axo C/E", "Axo F"]
    tex = r'''\documentclass[border=0pt]{standalone}
\usepackage{fontspec}
\setsansfont{InterVariable.ttf}[Path=FONTDIR/]
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{AxymPurple}{HTML}{3F21B6}
\definecolor{Baseline}{HTML}{737780}
\definecolor{Ink}{HTML}{20232A}
\definecolor{Grid}{HTML}{DCE9EC}
\begin{document}\begin{tikzpicture}[x=1bp,y=1bp]
\begin{axis}[
at={(34bp,31bp)},anchor=south west,scale only axis,width=7.25cm,height=4.45cm,
xmin=.5,xmax=4.5,ymin=0,ymax=110,clip=false,
axis x line*=bottom,axis y line*=left,axis line style={Ink,line width=.45pt},
tick align=outside,tick style={Ink,line width=.4pt},
tick label style={font=\sffamily\fontsize{9.5}{11}\selectfont,text=Ink},
label style={font=\sffamily\fontsize{10}{12}\selectfont,text=Ink},
xtick={1,2,3,4},xticklabels={LABELS},ytick={0,50,100},
ymajorgrids,grid style={Grid,line width=.35pt},ylabel={aBN1 rate (Hz)}]
'''.replace("FONTDIR", str(FONT_PATH.resolve().parent)).replace("LABELS", ",".join(labels))
    jitter = (-.08, 0, .08)
    for x, (group, color) in enumerate(zip(groups, colors), start=1):
        values = rates[group]
        coords = " ".join(f"({_number(x+j)},{_number(v)})" for j, v in zip(jitter, values))
        mean = float(np.mean(values))
        tex += f"\\addplot[{color},only marks,mark=*,mark size=2.2pt] coordinates {{{coords}}};\n"
        tex += f"\\draw[{color},line width=1.15pt] (axis cs:{x-.18},{_number(mean)}) -- (axis cs:{x+.18},{_number(mean)});\n"
    tex += rf"\end{{axis}}\pgfresetboundingbox\path[use as bounding box] (0,0) rectangle ({width*.75},{height*.75});" + "\n"
    tex += "\\end{tikzpicture}\\end{document}\n"
    path = output / "specificity.tex"
    path.write_text(tex)
    run = subprocess.run(
        ["tectonic", "--keep-logs", "--outdir", str(output.resolve()), str(path.resolve())],
        capture_output=True,
        text=True,
    )
    (output / "specificity-build.txt").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError("Tectonic failed; see specificity-build.txt")
    subprocess.run(
        ["pdftocairo", "-svg", str(output / "specificity.pdf"), str(output / "specificity.svg")],
        check=True,
        capture_output=True,
    )
    prefix = output / "specificity-hires"
    subprocess.run(
        ["pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", str(width*3), "-scale-to-y", str(height*3), str(output/"specificity.pdf"), str(prefix)],
        check=True,
        capture_output=True,
    )
    image = Image.open(prefix.with_suffix(".png")).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
    image.save(output / "specificity.png")
    prefix.with_suffix(".png").unlink()
    return image


def _event_matrix(last_event, time_ms):
    age = np.maximum(time_ms - last_event, 0)
    intensity = np.exp(-age / 24.0)
    intensity[last_event < 0] = 0
    colors = INACTIVE[None, :] + intensity[:, None] * (PURPLE - INACTIVE)[None, :]
    return np.rint(colors).astype(np.uint8).reshape(280, 455, 3)


def render_video(axosim_summary, events, reference_results, output, *, fps=30, seconds=10):
    axosim_summary, events = Path(axosim_summary), Path(events)
    reference_results, output = Path(reference_results), Path(output)
    summary = json.loads(axosim_summary.read_text())
    reference = json.loads(reference_results.read_text())
    if summary["status"] != "complete" or summary["n_neurons"] != 127400:
        raise ValueError("Expected the completed exact 127,400-neuron run")
    if summary["graph"].get("pruning") or summary["graph"].get("coalescing"):
        raise ValueError("Scale film requires the exact unpruned, uncoalesced graph")
    with np.load(events) as loaded:
        key = "sparse_events_neuron_time" if "sparse_events_neuron_time" in loaded.files else "sparse_events_time_index"
        event_rows = loaded[key].astype(np.int64)
        population = loaded["population_counts"].astype(np.int64)
    order = np.argsort(event_rows[:, 1], kind="stable")
    event_rows = event_rows[order]
    rates = _rates(summary, reference)
    output.mkdir(parents=True, exist_ok=True)
    specificity = _compile_specificity_plot(rates, output)
    title_font, heading_font, metric_font, small_font = (
        _font(25, weight=600), _font(17, weight=600), _font(16, weight=600), _font(13)
    )
    total_frames = int(round(fps * seconds))
    last_event = np.full(summary["n_neurons"], -1e9, dtype=np.float64)
    cursor = 0
    writer = imageio.get_writer(
        output / "full-connectome-scale.mp4", fps=fps, codec="libx264", quality=9, macro_block_size=1
    )
    try:
        for frame in range(total_frames):
            biological_ms = min(frame / max(total_frames - 1, 1) * summary["duration_ms"], summary["duration_ms"])
            next_cursor = int(np.searchsorted(event_rows[:, 1], biological_ms, side="right"))
            new = event_rows[cursor:next_cursor]
            if len(new):
                last_event[new[:, 0]] = new[:, 1]
            cursor = next_cursor
            matrix = Image.fromarray(_event_matrix(last_event, biological_ms)).resize((819, 504), Image.Resampling.NEAREST)
            image = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 255))
            image.alpha_composite(matrix.convert("RGBA"), (42, 94))
            draw = ImageDraw.Draw(image, "RGBA")
            draw.text((42, 42), "INDEXED NEURON POPULATION", font=small_font, fill=(90, 94, 104))
            draw.text((42, 611), f"biological time  {biological_ms/1000:.3f} s", font=small_font, fill=INK)
            active = int(population[min(int(biological_ms), len(population) - 1)])
            draw.text((936, 57), "127,400 neurons", font=heading_font, fill=INK)
            draw.text((936, 89), "14,687,178 pair edges", font=small_font, fill=INK)
            draw.text((936, 114), "52,793,639 contacts", font=small_font, fill=INK)
            draw.text((936, 155), f"active now  {active:,}", font=small_font, fill=(63, 33, 182))
            allocated = max(row.get("peak_cuda_allocated_bytes", 0) for row in summary["results"])
            reserved = max(row.get("peak_cuda_reserved_bytes", 0) for row in summary["results"])
            wall = np.mean([row["wall_seconds"] for row in summary["results"] if row["condition"] != "baseline"])
            draw.text((936, 198), f"GPU allocated  {allocated/2**30:.2f} GiB", font=small_font, fill=INK)
            draw.text((936, 223), f"GPU reserved  {reserved/2**30:.2f} GiB", font=small_font, fill=INK)
            draw.text((936, 248), f"wall / biological second  {wall:.2f} s", font=small_font, fill=INK)
            draw.text((906, 314), "aBN1 specificity · negative result", font=metric_font, fill=INK)
            draw.text((906, 340), "expected C/E > F · AxoSim 0/3 seeds", font=small_font, fill=(90, 94, 104))
            image.alpha_composite(specificity, (887, 370))
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rounded_rectangle((22, 655, 337, 709), radius=13, fill=(255, 255, 255, 235))
            draw.text((42, 672), "AxoSim - Axym Labs", font=title_font, fill=INK)
            writer.append_data(np.asarray(image.convert("RGB")))
            if frame in (0, total_frames // 2, total_frames - 1):
                image.convert("RGB").save(output / f"frame-{frame:03d}.png")
    finally:
        writer.close()
    visual = {
        "title": "AxoSim - Axym Labs",
        "video": "full-connectome-scale.mp4",
        "frames": total_frames,
        "fps": fps,
        "seconds": seconds,
        "biological_seconds": summary["duration_ms"] / 1000,
        "resolution": [WIDTH, HEIGHT],
        "axosim_summary_sha256": _sha256(axosim_summary),
        "event_stream_sha256": _sha256(events),
        "reference_results_sha256": _sha256(reference_results),
        "graph": {
            "neurons": summary["n_neurons"],
            "pair_edges": summary["n_edges"],
            "contacts": summary["anatomical_contact_count"],
            "pruning": summary["graph"]["pruning"],
            "coalescing": summary["graph"]["coalescing"],
        },
        "display": "455 by 280 index grid; spatial position is neuron index, not anatomy; 24-ms causal event afterglow",
        "condition": "jo_ce seed 0",
        "rates_hz": rates,
        "specificity_result": "negative: JO-C/E > JO-F in 0/3 AxoSim seeds; original LIF reference passed 3/3",
        "renderer": "Pillow event field plus Tectonic/PGFPlots vector specificity plot",
        "future_events_shown": False,
    }
    (output / "visual-metadata.json").write_text(json.dumps(visual, indent=2) + "\n")
    return visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axosim-summary", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render_video(args.axosim_summary, args.events, args.reference_results, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
