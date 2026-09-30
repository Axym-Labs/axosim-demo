"""Run and render a natural woodland movie through the frozen AxoSim visual circuit."""
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
from scipy.ndimage import map_coordinates

from .neural import DEFAULT_CHECKPOINT
from .tikz_plot import FONT_PATH, _number
from .vision_axosim import AxoSimFlyVis, DEFAULT_CONFIG
from .vision_video import INK, _hex_centers, _mix_color


SOURCE_SHA256 = "ded3617a45b61a956bba53a490fbbc9f3f8583a2ac06f60e59002286d81e2bda"
SOURCE_PAGE = "https://www.pexels.com/video/walking-into-the-woods-8381986/"
LICENSE_PAGE = "https://www.pexels.com/license/"
START_SECONDS = 20.0
DURATION_SECONDS = 8.0
SOURCE_DT = 0.005
LAYERS = ("L1", "Mi1", "T4c", "T5c")
WIDTH, HEIGHT = 1280, 720


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


def _sample_movie(source, u, v):
    """Return the fixed 20--28 s crop on the 721-column hexagonal field."""
    reader = imageio.get_reader(source)
    metadata = reader.get_meta_data()
    fps = float(metadata["fps"])
    width, height = metadata["size"]
    first = int(math.floor(START_SECONDS * fps))
    last = int(math.ceil((START_SECONDS + DURATION_SECONDS) * fps)) + 1
    x = np.asarray(u, dtype=np.float64) + np.asarray(v, dtype=np.float64) / 2
    y = np.asarray(v, dtype=np.float64) * math.sqrt(3) / 2
    x /= np.max(np.abs(x))
    y /= np.max(np.abs(y))
    # Work at 640x360: far above the 721-sample retinal resolution.
    sample_width, sample_height = 640, 360
    pixel_x = (0.5 + 0.47 * x) * (sample_width - 1)
    pixel_y = (0.5 + 0.47 * y) * (sample_height - 1)
    retinal_frames = []
    try:
        for frame_index in range(first, last + 1):
            rgb = Image.fromarray(reader.get_data(frame_index)).resize(
                (sample_width, sample_height), Image.Resampling.LANCZOS
            )
            array = np.asarray(rgb, dtype=np.float32) / 255
            luminance = (
                0.2126 * array[..., 0]
                + 0.7152 * array[..., 1]
                + 0.0722 * array[..., 2]
            )
            retinal_frames.append(
                map_coordinates(luminance, (pixel_y, pixel_x), order=1, mode="nearest")
            )
    finally:
        reader.close()
    retinal_frames = np.asarray(retinal_frames, dtype=np.float32)
    time = np.arange(0, DURATION_SECONDS, SOURCE_DT, dtype=np.float64)
    positions = (START_SECONDS + time) * fps - first
    lower = np.floor(positions).astype(int)
    alpha = (positions - lower)[:, None]
    stimulus = retinal_frames[lower] * (1 - alpha) + retinal_frames[lower + 1] * alpha
    return stimulus.astype(np.float32), time, {
        "fps": fps,
        "source_size": [width, height],
        "decoded_frame_range_inclusive": [first, last],
    }


def _metrics(activity, time):
    mask = time >= 1.0
    t4c = activity[mask, 2]
    return {
        "median_cell_temporal_std": float(np.median(np.std(t4c, axis=0))),
        "median_absolute_first_difference": float(np.median(np.abs(np.diff(t4c, axis=0)))),
    }


def run_experiment(
    *, source, config, root, scored_summary, output, checkpoint=DEFAULT_CHECKPOINT, device="cuda"
):
    source = Path(source)
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError("Natural-scene source differs from the frozen protocol")
    contract = json.loads(Path(config).read_text())
    summary = json.loads(Path(scored_summary).read_text())
    if not summary["video_gate_passed"]:
        raise RuntimeError("Selected visual-circuit arm has not passed its controlled gate")
    os.environ["FLYVIS_ROOT_DIR"] = str(Path(root).resolve())
    from flyvis import results_dir
    from flyvis.network import NetworkView

    view = NetworkView(results_dir / contract["flyvis_network"])
    network = view.init_network("best")
    selected = summary["selected"]
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
    first_layer = layer_index[:721]
    node_u = np.asarray(network.connectome.nodes.u[:])
    node_v = np.asarray(network.connectome.nodes.v[:])
    u, v = node_u[first_layer], node_v[first_layer]
    stimulus, time, media = _sample_movie(source, u, v)
    _, dynamic = runtime.run_stimulus(
        stimulus,
        source_dt=SOURCE_DT,
        target_dt=contract["protocol"]["axosim_dt_seconds"],
        record_index=layer_index,
    )
    static_stimulus = np.repeat(stimulus[:1], len(stimulus), axis=0)
    _, static = runtime.run_stimulus(
        static_stimulus,
        source_dt=SOURCE_DT,
        target_dt=contract["protocol"]["axosim_dt_seconds"],
        record_index=layer_index,
    )
    dynamic = dynamic.reshape(len(time), len(LAYERS), 721)
    static = static.reshape(len(time), len(LAYERS), 721)
    uniform_stimulus = np.repeat(stimulus.mean(axis=1, keepdims=True), 721, axis=1)
    _, uniform = runtime.run_stimulus(
        uniform_stimulus,
        source_dt=SOURCE_DT,
        target_dt=contract["protocol"]["axosim_dt_seconds"],
        record_index=layer_index,
    )
    uniform_static_stimulus = np.repeat(uniform_stimulus[:1], len(stimulus), axis=0)
    _, uniform_static = runtime.run_stimulus(
        uniform_static_stimulus,
        source_dt=SOURCE_DT,
        target_dt=contract["protocol"]["axosim_dt_seconds"],
        record_index=layer_index,
    )
    uniform = uniform.reshape(len(time), len(LAYERS), 721)
    uniform_static = uniform_static.reshape(len(time), len(LAYERS), 721)
    dynamic_metrics, static_metrics = _metrics(dynamic, time), _metrics(static, time)
    ratio = dynamic_metrics["median_cell_temporal_std"] / max(
        static_metrics["median_cell_temporal_std"], 1e-12
    )
    mask = time >= 1.0
    natural_cell_rms = np.sqrt(np.mean((dynamic[mask, 2] - static[mask, 2]) ** 2, axis=0))
    uniform_cell_rms = np.sqrt(
        np.mean((uniform[mask, 2] - uniform_static[mask, 2]) ** 2, axis=0)
    )
    natural_effect = float(np.median(natural_cell_rms))
    uniform_effect = float(np.median(uniform_cell_rms))
    spatial_effect_ratio = natural_effect / max(uniform_effect, 1e-12)
    finite = bool(
        np.isfinite(dynamic).all()
        and np.isfinite(static).all()
        and np.isfinite(uniform).all()
        and np.isfinite(uniform_static).all()
    )
    report = {
        "status": "complete",
        "scope": "out-of-sample natural-scene responsiveness of the frozen AxoSim visual circuit",
        "claim_boundary": "not locomotion, a calibrated compound eye, fly physiology or a behavior",
        "source": {
            "page": SOURCE_PAGE,
            "license": LICENSE_PAGE,
            "creator": "I Am Sorin",
            "title": "Walking Into the Woods",
            "sha256": SOURCE_SHA256,
            "crop_seconds": [START_SECONDS, START_SECONDS + DURATION_SECONDS],
            **media,
        },
        "graph": summary["graph"],
        "selected": selected,
        "dynamic_metrics": dynamic_metrics,
        "static_first_frame_control_metrics": static_metrics,
        "T4c_temporal_std_ratio_dynamic_over_static": ratio,
        "initial_video_gate_checks": {"finite": finite, "dynamic_exceeds_static": ratio > 1},
        "initial_video_gate_passed": finite and ratio > 1,
        "exploratory_spatial_content_followup": {
            "natural_minus_matched_static_median_cell_rms": natural_effect,
            "uniform_luminance_minus_matched_static_median_cell_rms": uniform_effect,
            "natural_over_uniform_ratio": spatial_effect_ratio,
        },
        "followup_video_gate_checks": {
            "finite": finite,
            "spatial_content_exceeds_global_luminance": spatial_effect_ratio > 1,
        },
        "followup_video_gate_passed": finite and spatial_effect_ratio > 1,
        "retinal_transform": "Rec.709 luma; centered planar hex field; bilinear spatial and linear temporal interpolation",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "natural-recording.npz",
        stimulus=stimulus,
        dynamic=dynamic,
        static=static,
        uniform=uniform,
        uniform_static=uniform_static,
        time=time,
        u=u,
        v=v,
    )
    report["recording_sha256"] = _sha256(output / "natural-recording.npz")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _trace_data(data):
    time = data["time"]
    dynamic = data["dynamic"].astype(np.float64)
    uniform = data["uniform"].astype(np.float64)
    base_mask = time < 1.0
    dynamic -= np.median(dynamic[base_mask], axis=0)
    uniform -= np.median(uniform[base_mask], axis=0)
    raw = {
        "L1": np.mean(dynamic[:, 0], axis=1),
        "Mi1": np.mean(dynamic[:, 1], axis=1),
        "T4c": np.mean(dynamic[:, 2], axis=1),
        "uniform T4c": np.mean(uniform[:, 2], axis=1),
    }

    def trailing_mean(values, samples=10):
        cumulative = np.concatenate(([0.0], np.cumsum(values)))
        index = np.arange(len(values))
        start = np.maximum(index + 1 - samples, 0)
        return (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)

    return {name: trailing_mean(values) for name, values in raw.items()}, dynamic


def _compile_trace_layers(time, traces, output, *, width=390, height=250):
    output = Path(output)
    for program in ("tectonic", "pdftocairo"):
        if not shutil.which(program):
            raise RuntimeError(f"{program} is required for the TikZ activity plot")
    values = np.concatenate(list(traces.values()))
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * .08, .01)
    styles = {
        "L1": "AxymLight,line width=.7pt",
        "Mi1": "AxymMid,line width=.8pt",
        "T4c": "AxymPurple,line width=1.2pt",
        "uniform T4c": "Baseline,line width=.7pt,dashed",
    }
    preamble = r'''\documentclass[border=0pt]{standalone}
\usepackage{fontspec}
\setsansfont{InterVariable.ttf}[Path=FONTDIR/]
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{AxymPurple}{HTML}{3F21B6}
\definecolor{AxymMid}{HTML}{8C7AD3}
\definecolor{AxymLight}{HTML}{B7AADF}
\definecolor{Baseline}{HTML}{686C75}
\definecolor{Ink}{HTML}{20232A}
\definecolor{Grid}{HTML}{DCE9EC}
\begin{document}\begin{tikzpicture}[x=1bp,y=1bp]
'''.replace("FONTDIR", str(FONT_PATH.resolve().parent))
    for layer in ("axes", "lines"):
        tex = preamble
        options = [
            "at={(12bp,31bp)}", "anchor=south west", "scale only axis", "width=8.35cm", "height=3.45cm",
            "xmin=0", "xmax=8", f"ymin={_number(lo-pad)}", f"ymax={_number(hi+pad)}", "clip=true",
            "tick label style={font=\\sffamily\\fontsize{10.5}{12}\\selectfont,text=Ink}",
            "label style={font=\\sffamily\\fontsize{10.5}{12}\\selectfont,text=Ink}",
        ]
        if layer == "axes":
            options += [
                "axis x line*=bottom", "axis y line*=left", "axis line style={Ink,line width=.45pt}",
                "tick align=outside", "tick style={Ink,line width=.4pt}", "xtick={0,4,8}",
                "ytick=\\empty", "ymajorgrids", "grid style={Grid,line width=.35pt}",
                "xlabel={Biological time (s)}",
                "legend style={draw=none,fill=none,font=\\sffamily\\fontsize{9}{10.5}\\selectfont,at={(.5,1.04)},anchor=south,legend columns=4,column sep=5pt}",
            ]
        else:
            options += ["hide axis", "xtick=\\empty", "ytick=\\empty"]
        tex += "\\begin{axis}[" + ",".join(options) + "]\n"
        if layer == "axes":
            for name in traces:
                tex += f"\\addlegendimage{{{styles[name]}}}\\addlegendentry{{{name}}}\n"
        else:
            # Plot every fifth 5-ms sample after the declared display-only
            # trailing mean; raw arrays remain saved without decimation.
            for name, values_for_trace in traces.items():
                coords = " ".join(
                    f"({_number(t)},{_number(v)})"
                    for t, v in zip(time[::5], values_for_trace[::5])
                )
                tex += f"\\addplot[{styles[name]},no marks] coordinates {{{coords}}};\n"
        tex += "\\end{axis}\n"
        tex += rf"\pgfresetboundingbox\path[use as bounding box] (0,0) rectangle ({width*.75},{height*.75});" + "\n"
        tex += "\\end{tikzpicture}\\end{document}\n"
        path = output / f"activity-{layer}.tex"
        path.write_text(tex)
        run = subprocess.run(
            ["tectonic", "--keep-logs", "--outdir", str(output.resolve()), str(path.resolve())],
            capture_output=True,
            text=True,
        )
        (output / f"activity-{layer}-build.txt").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f"Tectonic failed for activity-{layer}.tex")
        subprocess.run(
            ["pdftocairo", "-svg", str(output / f"activity-{layer}.pdf"), str(output / f"activity-{layer}.svg")],
            check=True,
            capture_output=True,
        )
        prefix = output / f"activity-{layer}-hires"
        subprocess.run(
            ["pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", str(width*3), "-scale-to-y", str(height*3), str(output/f"activity-{layer}.pdf"), str(prefix)],
            check=True,
            capture_output=True,
        )
        image = Image.open(prefix.with_suffix(".png")).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
        image.save(output / f"activity-{layer}.png")
        prefix.with_suffix(".png").unlink()


def _field(draw, centers, cell_radius, values):
    for (x, y), value in zip(centers, values):
        points = [
            (x + cell_radius * math.cos(math.pi / 3 * k), y + cell_radius * math.sin(math.pi / 3 * k))
            for k in range(6)
        ]
        draw.polygon(points, fill=_mix_color(value))


def _right_gradient():
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(730, WIDTH):
        alpha = int(238 * ((x - 730) / (WIDTH - 730)) ** .7)
        draw.line((x, 0, x, HEIGHT), fill=(255, 255, 255, alpha))
    return overlay


def render_video(source, recording_dir, output, *, fps=30):
    source, recording_dir, output = Path(source), Path(recording_dir), Path(output)
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError("Natural-scene source differs from the frozen protocol")
    report = json.loads((recording_dir / "summary.json").read_text())
    if not report["followup_video_gate_passed"]:
        raise RuntimeError("Natural-scene follow-up did not pass its video gate")
    with np.load(recording_dir / "natural-recording.npz") as loaded:
        data = {name: loaded[name] for name in loaded.files}
    output.mkdir(parents=True, exist_ok=True)
    traces, dynamic_delta = _trace_data(data)
    _compile_trace_layers(data["time"], traces, output)
    axes = Image.open(output / "activity-axes.png").convert("RGBA")
    lines = Image.open(output / "activity-lines.png").convert("RGBA")
    gradient = _right_gradient()
    centers, center_scale = _hex_centers(data["u"], data["v"], (1040, 205), 104)
    t4c_scale = np.quantile(np.abs(dynamic_delta[:, 2]), .99)
    t4c_scale = max(float(t4c_scale), 1e-9)
    title_font, label_font, small_font = _font(25, weight=600), _font(17, weight=600), _font(13)
    reader = imageio.get_reader(source)
    source_fps = float(reader.get_meta_data()["fps"])
    frames = int(round(DURATION_SECONDS * fps))
    writer = imageio.get_writer(
        output / "natural-woodland-vision.mp4", fps=fps, codec="libx264", quality=9, macro_block_size=1
    )
    try:
        for frame_index in range(frames):
            video_time = frame_index / fps
            source_index = int(round((START_SECONDS + video_time) * source_fps))
            background = Image.fromarray(reader.get_data(source_index)).convert("RGB").resize(
                (WIDTH, HEIGHT), Image.Resampling.LANCZOS
            ).convert("RGBA")
            background.alpha_composite(gradient)
            draw = ImageDraw.Draw(background, "RGBA")
            sample = min(int(round(video_time / SOURCE_DT)), len(data["time"]) - 1)
            _field(draw, centers, max(1.25, center_scale * .46), dynamic_delta[sample, 2] / t4c_scale)
            draw.text((984, 72), "T4c · AxoSim", font=label_font, fill=INK)
            ratio = report["exploratory_spatial_content_followup"]["natural_over_uniform_ratio"]
            draw.text((905, 305), f"spatial-content effect  {ratio:.2f}× uniform", font=small_font, fill=INK)
            draw.text((870, 351), "POPULATION ACTIVITY · 50 ms trailing mean", font=small_font, fill=INK)
            background.alpha_composite(axes, (858, 337))
            visible = lines.copy()
            edge = int(round(visible.width * min(video_time / DURATION_SECONDS, 1)))
            if edge < visible.width:
                ImageDraw.Draw(visible).rectangle((edge + 1, 0, visible.width, visible.height), fill=(0, 0, 0, 0))
            background.alpha_composite(visible, (858, 337))
            draw = ImageDraw.Draw(background, "RGBA")
            draw.rounded_rectangle((22, 655, 337, 709), radius=13, fill=(255, 255, 255, 220))
            draw.text((42, 672), "AxoSim - Axym Labs", font=title_font, fill=INK)
            writer.append_data(np.asarray(background.convert("RGB")))
            if frame_index in (0, frames // 2, frames - 1):
                background.convert("RGB").save(output / f"frame-{frame_index:03d}.png")
    finally:
        reader.close()
        writer.close()
    visual = {
        "title": "AxoSim - Axym Labs",
        "video": "natural-woodland-vision.mp4",
        "frames": frames,
        "fps": fps,
        "seconds": DURATION_SECONDS,
        "resolution": [WIDTH, HEIGHT],
        "source_video_sha256": SOURCE_SHA256,
        "source_crop_seconds": [START_SECONDS, START_SECONDS + DURATION_SECONDS],
        "recording_sha256": report["recording_sha256"],
        "metrics": {
            "dynamic": report["dynamic_metrics"],
            "static": report["static_first_frame_control_metrics"],
            "T4c_temporal_std_ratio_dynamic_over_static": report["T4c_temporal_std_ratio_dynamic_over_static"],
            "exploratory_spatial_content_followup": report["exploratory_spatial_content_followup"],
        },
        "plot": "Tectonic/PGFPlots vector layers with progressive historical reveal",
        "plot_display_transform": "50-ms causal trailing mean, then every fifth 5-ms sample; raw 5-ms arrays retained",
        "future_samples_shown": False,
        "claim_boundary": report["claim_boundary"],
    }
    (output / "visual-metadata.json").write_text(json.dumps(visual, indent=2) + "\n")
    return visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--scored-summary", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--device", default="cuda")
    render = sub.add_parser("render")
    render.add_argument("--source", type=Path, required=True)
    render.add_argument("--recording", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        result = run_experiment(
            source=args.source,
            config=args.config,
            root=args.root,
            scored_summary=args.scored_summary,
            output=args.output,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    else:
        result = render_video(args.source, args.recording, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
