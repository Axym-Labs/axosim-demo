"""Record and render the controlled AxoSim visual-circuit evidence film."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .neural import DEFAULT_CHECKPOINT
from .tikz_plot import FONT_PATH, _number
from .vision_axosim import (
    AxoSimFlyVis,
    DEFAULT_CONFIG,
    _dataset,
    _replace_responses,
)


WIDTH, HEIGHT = 1280, 720
LAYERS = ("L1", "Mi1", "T4c", "T5c")
PURPLE = np.array([63, 33, 182], dtype=np.float64)
PALE = np.array([202, 221, 225], dtype=np.float64)
WHITE = np.array([255, 255, 255], dtype=np.float64)
INK = (32, 35, 42)


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


def prepare_recording(
    *, config, root, scored_output, output, checkpoint=DEFAULT_CHECKPOINT, device="cuda"
):
    """Re-run one fixed held-out trial and save spatial AxoSim layer states."""
    config = Path(config)
    contract = json.loads(config.read_text())
    scored_output = Path(scored_output)
    scored_summary = json.loads((scored_output / "summary.json").read_text())
    if not scored_summary["video_gate_passed"]:
        raise RuntimeError("The frozen visual-circuit experiment did not pass its video gate")
    with np.load(scored_output / "selected-responses.npz") as loaded:
        scored = {name: loaded[name] for name in loaded.files}
    os.environ["FLYVIS_ROOT_DIR"] = str(Path(root).resolve())
    from flyvis import results_dir
    from flyvis.analysis.moving_bar_responses import peak_responses
    from flyvis.network import NetworkView

    view = NetworkView(results_dir / contract["flyvis_network"])
    network = view.init_network("best")
    protocol = contract["protocol"]
    dataset = _dataset(contract, protocol["angles_deg"])
    reference = view.moving_edge_responses(dataset)
    sample_matches = np.flatnonzero(
        (reference.angle.values == 30) & (reference.intensity.values == 1)
    )
    if len(sample_matches) != 1:
        raise RuntimeError("Expected exactly one held-out light-edge 30-degree trial")
    sample = int(sample_matches[0])
    selected = scored_summary["selected"]
    runtime = AxoSimFlyVis(
        network,
        checkpoint=checkpoint,
        morphology_index=selected["morphology_index"],
        edge_current_gain=selected["edge_current_gain"],
        device=device,
    )
    layer_index = np.concatenate(
        [np.asarray(getattr(network.connectome.nodes.layer_index, name)[:]) for name in LAYERS]
    )
    central, recorded = runtime.run_stimulus(
        reference.stimulus.values[sample, :, 0, :],
        source_dt=protocol["source_dt_seconds"],
        target_dt=protocol["axosim_dt_seconds"],
        record_index=layer_index,
    )
    inference_difference = float(np.max(np.abs(central - scored["axosim"][sample])))
    if inference_difference > 1e-5:
        raise RuntimeError(
            f"Movie inference differs from scored response by {inference_difference:g}"
        )
    recorded = recorded.reshape(len(recorded), len(LAYERS), 721)

    candidate = _replace_responses(reference, scored["axosim"], label="AxoSim")
    control = _replace_responses(reference, scored["spatial_control"], label="control")
    peaks = {
        "flyvis": peak_responses(reference).values[0],
        "axosim": peak_responses(candidate).values[0],
        "spatial_control": peak_responses(control).values[0],
    }
    cell_types = reference.cell_type.values.astype(str)
    t4c = int(np.flatnonzero(cell_types == "T4c")[0])
    tuning_mask = reference.intensity.values == 1
    order = np.argsort(reference.angle.values[tuning_mask])
    tuning_angles = reference.angle.values[tuning_mask][order]
    tuning = {
        name: values[tuning_mask, t4c][order].astype(float).tolist()
        for name, values in peaks.items()
    }

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coords = np.column_stack(
        (
            np.asarray(network.connectome.nodes.u[:])[layer_index[:721]],
            np.asarray(network.connectome.nodes.v[:])[layer_index[:721]],
        )
    )
    np.savez_compressed(
        output / "recording.npz",
        stimulus=reference.stimulus.values[sample, :, 0, :],
        activity=recorded,
        time=reference.time.values,
        u=coords[:, 0],
        v=coords[:, 1],
        central=central,
    )
    metadata = {
        "scope": "held-out 30-degree light edge through the selected full AxoSim visual circuit",
        "sample_index": sample,
        "angle_deg": 30,
        "intensity": 1,
        "layers": list(LAYERS),
        "nodes": scored_summary["graph"]["nodes"],
        "edges": scored_summary["graph"]["edges"],
        "morphology_index": selected["morphology_index"],
        "edge_current_gain": selected["edge_current_gain"],
        "scored_inference_max_abs_difference": inference_difference,
        "scored_summary_sha256": _sha256(scored_output / "summary.json"),
        "scored_responses_sha256": _sha256(scored_output / "selected-responses.npz"),
        "recording_sha256": _sha256(output / "recording.npz"),
        "tuning": {"angle_deg": tuning_angles.astype(float).tolist(), **tuning},
        "display_transform": "per-layer pre-stimulus-median subtraction; symmetric 99th-percentile scale",
    }
    (output / "recording-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _compile_tuning(metadata, output):
    for program in ("tectonic", "pdftocairo"):
        if not shutil.which(program):
            raise RuntimeError(f"{program} is required for the TikZ tuning plot")
    output = Path(output)
    tuning = metadata["tuning"]
    angles = np.asarray(tuning["angle_deg"])
    # Normalize each arm only for legible shape comparison; raw primary metrics remain
    # in the experiment report and are printed separately in the film.
    normalized = {}
    for name in ("flyvis", "axosim", "spatial_control"):
        values = np.asarray(tuning[name], dtype=np.float64)
        scale = np.max(np.abs(values))
        normalized[name] = values / scale if scale else values
    styles = {
        "flyvis": "Baseline,line width=.8pt",
        "axosim": "AxymPurple,line width=1.3pt",
        "spatial_control": "AxymLight,line width=.8pt,dashed",
    }
    labels = {"flyvis": "FlyVis", "axosim": "AxoSim", "spatial_control": "scrambled"}
    tex = r'''\documentclass[border=0pt]{standalone}
\usepackage{fontspec}
\setsansfont{InterVariable.ttf}[Path=FONTDIR/]
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{AxymPurple}{HTML}{3F21B6}
\definecolor{AxymLight}{HTML}{8C7AD3}
\definecolor{Baseline}{HTML}{666A73}
\definecolor{Ink}{HTML}{20232A}
\definecolor{Grid}{HTML}{DCE9EC}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
width=10.4cm,height=5.2cm,xmin=0,xmax=330,ymin=0,ymax=1.08,
axis x line*=bottom,axis y line*=left,axis line style={Ink,line width=.45pt},
tick align=outside,tick style={Ink,line width=.4pt},
tick label style={font=\sffamily\fontsize{9}{11}\selectfont,text=Ink},
label style={font=\sffamily\fontsize{9}{11}\selectfont,text=Ink},
xtick={0,90,180,270},xticklabels={0°,90°,180°,270°},
ytick={0,.5,1},yticklabels={0,.5,1},ymajorgrids,grid style={Grid,line width=.35pt},
xlabel={Edge direction},ylabel={Normalized T4c peak},
legend style={draw=none,fill=none,font=\sffamily\fontsize{8.5}{10}\selectfont,
at={(0.5,1.03)},anchor=south,legend columns=3,column sep=8pt},
clip=true]
'''.replace("FONTDIR", str(FONT_PATH.resolve().parent))
    for name in ("flyvis", "axosim", "spatial_control"):
        coords = " ".join(
            f"({_number(x)},{_number(y)})" for x, y in zip(angles, normalized[name])
        )
        tex += f"\\addplot[{styles[name]},no marks] coordinates {{{coords}}};\n"
        tex += f"\\addlegendentry{{{labels[name]}}}\n"
    tex += "\\end{axis}\n\\end{tikzpicture}\n\\end{document}\n"
    tex_path = output / "tuning.tex"
    tex_path.write_text(tex)
    run = subprocess.run(
        ["tectonic", "--keep-logs", "--outdir", str(output.resolve()), str(tex_path.resolve())],
        capture_output=True,
        text=True,
    )
    (output / "tuning-build.txt").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError("Tectonic failed; see tuning-build.txt")
    subprocess.run(
        ["pdftocairo", "-svg", str(output / "tuning.pdf"), str(output / "tuning.svg")],
        check=True,
        capture_output=True,
    )
    prefix = output / "tuning-hires"
    subprocess.run(
        [
            "pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", "920",
            "-scale-to-y", "460", str(output / "tuning.pdf"), str(prefix),
        ],
        check=True,
        capture_output=True,
    )
    image = Image.open(prefix.with_suffix(".png")).convert("RGBA")
    image.thumbnail((460, 230), Image.Resampling.LANCZOS)
    image.save(output / "tuning.png")
    prefix.with_suffix(".png").unlink()
    return image


def _hex_centers(u, v, center, radius):
    x = np.asarray(u, dtype=np.float64) + np.asarray(v, dtype=np.float64) / 2
    y = np.asarray(v, dtype=np.float64) * math.sqrt(3) / 2
    scale = radius / max(np.max(np.abs(x)), np.max(np.abs(y)), 1)
    return np.column_stack((center[0] + x * scale, center[1] + y * scale)), scale


def _mix_color(value):
    value = float(np.clip(value, -1, 1))
    end = PURPLE if value >= 0 else PALE
    return tuple(np.rint(WHITE + abs(value) * (end - WHITE)).astype(np.uint8))


def _draw_field(draw, centers, cell_radius, values, *, stimulus=False):
    for (x, y), value in zip(centers, values):
        if stimulus:
            gray = int(round(242 - 205 * float(np.clip(value, 0, 1))))
            color = (gray, gray, gray)
        else:
            color = _mix_color(value)
        points = [
            (
                x + cell_radius * math.cos(math.pi / 3 * k),
                y + cell_radius * math.sin(math.pi / 3 * k),
            )
            for k in range(6)
        ]
        draw.polygon(points, fill=color)


def render_video(recording_dir, scored_summary, output, *, fps=30, seconds=10):
    recording_dir, output = Path(recording_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((recording_dir / "recording-metadata.json").read_text())
    summary = json.loads(Path(scored_summary).read_text())
    with np.load(recording_dir / "recording.npz") as loaded:
        data = {name: loaded[name] for name in loaded.files}
    tuning = _compile_tuning(metadata, output)
    activity = data["activity"].astype(np.float64)
    baseline = np.median(activity[data["time"] < 1.0], axis=0)
    delta = activity - baseline
    scales = np.quantile(np.abs(delta), .99, axis=(0, 2))
    scales[scales < 1e-9] = 1
    delta /= scales[None, :, None]
    centers = [(165, 280), (430, 275), (665, 280), (865, 285)]
    radii = [125, 108, 96, 86]
    grids = [_hex_centers(data["u"], data["v"], center, radius) for center, radius in zip(centers, radii)]
    title_font, label_font, small_font, metric_font = (
        _font(25, weight=600), _font(17, weight=600), _font(13), _font(16, weight=600)
    )
    frames = int(round(fps * seconds))
    writer = imageio.get_writer(
        output / "controlled-visual-circuit.mp4",
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=1,
    )
    try:
        for frame in range(frames):
            fraction = min(frame / max(frames - 1, 1), 1)
            sample = min(int(round(fraction * (len(data["time"]) - 1))), len(data["time"]) - 1)
            image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
            draw = ImageDraw.Draw(image, "RGBA")
            for start, end in zip(centers[:-1], centers[1:]):
                draw.line((start[0] + 128, start[1], end[0] - 112, end[1]), fill=(220, 225, 230, 210), width=2)
                draw.polygon(
                    ((end[0] - 114, end[1] - 5), (end[0] - 104, end[1]), (end[0] - 114, end[1] + 5)),
                    fill=(140, 145, 155, 220),
                )
            stimulus_grid, stimulus_scale = grids[0]
            _draw_field(draw, stimulus_grid, max(1.45, stimulus_scale * .48), data["stimulus"][sample], stimulus=True)
            for (grid, grid_scale), values in zip(grids[1:], delta[sample, :3]):
                _draw_field(draw, grid, max(1.25, grid_scale * .48), values)
            display_labels = ("RETINA", "L1", "Mi1", "T4c")
            for label, center, radius in zip(display_labels, centers, radii):
                bbox = draw.textbbox((0, 0), label, font=label_font)
                draw.text((center[0] - (bbox[2] - bbox[0]) / 2, center[1] - radius - 38), label, font=label_font, fill=INK)
            draw.text((997, 98), "45,669 neurons", font=metric_font, fill=INK)
            draw.text((997, 126), "1,513,231 edges", font=small_font, fill=INK)
            draw.text((997, 158), "held-out tuning  r = 0.72", font=small_font, fill=(63, 33, 182))
            draw.text((997, 183), "scrambled graph  r = 0.08", font=small_font, fill=(90, 94, 104))
            draw.text((997, 208), "T4c direction error  2.0°", font=small_font, fill=INK)
            draw.rounded_rectangle((789, 414, 1254, 657), radius=16, fill=(255, 255, 255, 235), outline=(220, 225, 230, 230), width=1)
            image_rgba = image.convert("RGBA")
            image_rgba.alpha_composite(tuning, (792, 420))
            draw = ImageDraw.Draw(image_rgba, "RGBA")
            draw.text((42, 671), "AxoSim - Axym Labs", font=title_font, fill=INK)
            draw.text((42, 38), "ACTIVITY RELATIVE TO PRE-STIMULUS BASELINE", font=small_font, fill=(90, 94, 104))
            writer.append_data(np.asarray(image_rgba.convert("RGB")))
            if frame in (0, frames // 2, frames - 1):
                image_rgba.convert("RGB").save(output / f"frame-{frame:03d}.png")
    finally:
        writer.close()
    visual = {
        "title": "AxoSim - Axym Labs",
        "video": "controlled-visual-circuit.mp4",
        "frames": frames,
        "fps": fps,
        "seconds": seconds,
        "resolution": [WIDTH, HEIGHT],
        "playback": f"{float(data['time'][-1]) / seconds:.4f} biological seconds per video second",
        "source_recording_sha256": _sha256(recording_dir / "recording.npz"),
        "source_summary_sha256": _sha256(scored_summary),
        "inference_match_max_abs": metadata["scored_inference_max_abs_difference"],
        "display_only_transform": metadata["display_transform"],
        "plot": "actual Tectonic/PGFPlots vector figure, normalized per arm for shape visibility",
        "primary_metrics_are_raw": True,
        "metrics": {
            "held_out_mean_t4_t5_tuning_correlation": summary["selected_metrics"]["held_out_peak_tuning_correlation"]["mean"],
            "spatial_control_mean": summary["spatial_control_metrics"]["held_out_peak_tuning_correlation"]["mean"],
            "T4c_direction_error_deg": summary["selected_metrics"]["T4c_preferred_direction_error_deg"],
            "T4c_direction_selectivity_index": summary["selected_metrics"]["T4c_direction_selectivity_index"],
        },
        "future_samples_shown": False,
    }
    (output / "visual-metadata.json").write_text(json.dumps(visual, indent=2) + "\n")
    return visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--scored-output", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    prepare.add_argument("--device", default="cuda")
    render = sub.add_parser("render")
    render.add_argument("--recording", type=Path, required=True)
    render.add_argument("--summary", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_recording(
            config=args.config,
            root=args.root,
            scored_output=args.scored_output,
            output=args.output,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    else:
        result = render_video(args.recording, args.summary, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
