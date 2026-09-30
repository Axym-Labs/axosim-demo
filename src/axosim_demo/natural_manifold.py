"""Run and render a 3D AxoSim visual-neuron manifold under real scenes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time as time_module

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

from .neural import DEFAULT_CHECKPOINT
from .tikz_plot import FONT_PATH
from .vision_axosim import AxoSimFlyVis


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs/natural_manifold_spatial.json"
)
WIDTH, HEIGHT = 1280, 720
INK = (32, 35, 42)
MUTED = (96, 100, 109)
LAYER_COLORS = np.asarray(
    [(140, 122, 211), (61, 147, 163), (63, 33, 182), (211, 132, 39)],
    dtype=np.float64,
)


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


def _retinal_coordinates(network):
    first = np.asarray(network.connectome.nodes.layer_index.L1[:])
    node_u = np.asarray(network.connectome.nodes.u[:])
    node_v = np.asarray(network.connectome.nodes.v[:])
    u, v = node_u[first], node_v[first]
    x = u.astype(np.float64) + v.astype(np.float64) / 2
    y = v.astype(np.float64) * math.sqrt(3) / 2
    return u, v, x / np.max(np.abs(x)), y / np.max(np.abs(y))


def _sample_scene(scene, *, duration, dt, retinal_x, retinal_y):
    path = Path(scene["path"])
    if _sha256(path) != scene["sha256"]:
        raise ValueError(f"Source hash mismatch for {scene['name']}")
    reader = imageio.get_reader(path)
    metadata = reader.get_meta_data()
    fps = float(metadata["fps"])
    width, height = metadata["size"]
    start = float(scene["start_seconds"])
    first = int(math.floor(start * fps))
    last = int(math.ceil((start + duration) * fps)) + 1
    sample_width, sample_height = 640, 360
    pixel_x = (0.5 + 0.47 * retinal_x) * (sample_width - 1)
    pixel_y = (0.5 + 0.47 * retinal_y) * (sample_height - 1)
    frames = []
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
            frames.append(
                map_coordinates(
                    luminance, (pixel_y, pixel_x), order=1, mode="nearest"
                )
            )
    finally:
        reader.close()
    frames = np.asarray(frames, dtype=np.float32)
    time = np.arange(0, duration, dt, dtype=np.float64)
    positions = (start + time) * fps - first
    lower = np.floor(positions).astype(int)
    alpha = (positions - lower)[:, None]
    stimulus = frames[lower] * (1 - alpha) + frames[lower + 1] * alpha
    return stimulus.astype(np.float32), time, {
        "fps": fps,
        "source_size": [width, height],
        "decoded_frame_range_inclusive": [first, last],
    }


def _classify_scene(activity, time, contract):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = contract["classification"]
    step = int(round(cfg["sample_seconds"] / contract["source_dt_seconds"]))
    train_lo, train_hi = cfg["train_seconds"]
    test_lo, test_hi = cfg["test_seconds"]
    sampled_time = time[::step]
    sampled = activity[:, ::step].reshape(len(activity), len(sampled_time), -1)
    train_mask = (sampled_time >= train_lo) & (sampled_time < train_hi)
    test_mask = (sampled_time >= test_lo) & (sampled_time < test_hi)
    x_train = sampled[:, train_mask].reshape(-1, sampled.shape[-1])
    x_test = sampled[:, test_mask].reshape(-1, sampled.shape[-1])
    y_train = np.repeat(np.arange(len(activity)), int(train_mask.sum()))
    y_test = np.repeat(np.arange(len(activity)), int(test_mask.sum()))
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(cfg["logistic_C"]),
            max_iter=2000,
            random_state=contract["embedding"]["random_seed"],
        ),
    )
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "confusion_matrix": confusion_matrix(
            y_test, prediction, labels=np.arange(len(activity))
        ).tolist(),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "features": x_train.shape[1],
    }


def _embed_neurons(activity, time, contract):
    from sklearn.decomposition import PCA
    from sklearn.manifold import trustworthiness
    import umap

    cfg = contract["embedding"]
    step = int(round(cfg["sample_seconds"] / contract["source_dt_seconds"]))
    mask = time[::step] >= contract["warmup_seconds"]
    # scene,time,layer,cell -> layer,cell,scene*time
    sampled = activity[:, ::step][:, mask]
    fingerprints = sampled.transpose(2, 3, 0, 1).reshape(2884, -1).astype(np.float64)
    mean = fingerprints.mean(axis=1, keepdims=True)
    std = fingerprints.std(axis=1, keepdims=True)
    constant = std[:, 0] < 1e-8
    standardized = np.divide(
        fingerprints - mean,
        std,
        out=np.zeros_like(fingerprints),
        where=std >= 1e-8,
    )
    pca = PCA(
        n_components=int(cfg["pca_components"]),
        svd_solver="randomized",
        random_state=int(cfg["random_seed"]),
    )
    pca_values = pca.fit_transform(standardized).astype(np.float32)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=int(cfg["n_neighbors"]),
        min_dist=float(cfg["min_dist"]),
        metric=cfg["metric"],
        random_state=int(cfg["random_seed"]),
        n_jobs=1,
    )
    embedding = reducer.fit_transform(pca_values).astype(np.float32)
    quality = float(
        trustworthiness(
            pca_values,
            embedding,
            n_neighbors=int(cfg["trustworthiness_neighbors"]),
        )
    )
    return embedding, pca_values, {
        "constant_neurons": int(constant.sum()),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "trustworthiness": quality,
    }


def run_experiment(
    *, config=DEFAULT_CONFIG, root, output, checkpoint=DEFAULT_CHECKPOINT, device="cuda"
):
    config = Path(config)
    contract = json.loads(config.read_text())
    vision_contract = json.loads(Path(contract["vision_config"]).read_text())
    scored = json.loads(Path(contract["scored_summary"]).read_text())
    if not scored["video_gate_passed"]:
        raise RuntimeError("Controlled visual arm has not passed its gate")
    os.environ["FLYVIS_ROOT_DIR"] = str(Path(root).resolve())
    from flyvis import results_dir
    from flyvis.network import NetworkView

    view = NetworkView(results_dir / vision_contract["flyvis_network"])
    network = view.init_network("best")
    selected = scored["selected"]
    runtime = AxoSimFlyVis(
        network,
        checkpoint=checkpoint,
        morphology_index=selected["morphology_index"],
        edge_current_gain=selected["edge_current_gain"],
        device=device,
    )
    layers = tuple(contract["record_layers"])
    record_index = np.concatenate(
        [np.asarray(getattr(network.connectome.nodes.layer_index, name)[:]) for name in layers]
    )
    u, v, retinal_x, retinal_y = _retinal_coordinates(network)
    dynamic, uniform, media = [], [], []
    source_dt = float(contract["source_dt_seconds"])
    started = time_module.monotonic()
    time = None
    for scene in contract["scenes"]:
        print(f"scene: {scene['name']}", flush=True)
        stimulus, time, source_metadata = _sample_scene(
            scene,
            duration=float(contract["duration_seconds"]),
            dt=source_dt,
            retinal_x=retinal_x,
            retinal_y=retinal_y,
        )
        normalization = contract.get("retinal_normalization")
        if normalization is not None:
            frame_mean = stimulus.mean(axis=1, keepdims=True)
            frame_std = stimulus.std(axis=1, keepdims=True)
            stimulus = (
                (stimulus - frame_mean) / np.maximum(frame_std, 1e-6)
                * float(normalization["standard_deviation"])
                + float(normalization["mean"])
            )
            stimulus = np.clip(
                stimulus,
                float(normalization["clip"][0]),
                float(normalization["clip"][1]),
            ).astype(np.float32)
        _, response = runtime.run_stimulus(
            stimulus,
            source_dt=source_dt,
            target_dt=vision_contract["protocol"]["axosim_dt_seconds"],
            record_index=record_index,
        )
        if normalization is None:
            uniform_stimulus = np.repeat(
                stimulus.mean(axis=1, keepdims=True), 721, axis=1
            )
        else:
            uniform_stimulus = np.full_like(
                stimulus, float(normalization["mean"])
            )
        _, uniform_response = runtime.run_stimulus(
            uniform_stimulus,
            source_dt=source_dt,
            target_dt=vision_contract["protocol"]["axosim_dt_seconds"],
            record_index=record_index,
        )
        dynamic.append(response.reshape(len(time), len(layers), 721))
        uniform.append(uniform_response.reshape(len(time), len(layers), 721))
        media.append(source_metadata)
    dynamic = np.asarray(dynamic, dtype=np.float32)
    uniform = np.asarray(uniform, dtype=np.float32)
    if not np.isfinite(dynamic).all() or not np.isfinite(uniform).all():
        raise RuntimeError("Nonfinite AxoSim manifold recording")
    dynamic_score = _classify_scene(dynamic, time, contract)
    uniform_score = _classify_scene(uniform, time, contract)
    embedding, pca_values, embedding_metrics = _embed_neurons(dynamic, time, contract)
    low = np.quantile(dynamic, 0.05, axis=(0, 1)).reshape(-1).astype(np.float32)
    high = np.quantile(dynamic, 0.95, axis=(0, 1)).reshape(-1).astype(np.float32)
    advantage = dynamic_score["balanced_accuracy"] - uniform_score["balanced_accuracy"]
    gate = (
        dynamic_score["balanced_accuracy"]
        >= contract["classification"]["minimum_natural_balanced_accuracy"]
        and advantage >= contract["classification"]["minimum_advantage_over_uniform"]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "manifold-recording.npz",
        dynamic=dynamic,
        uniform=uniform,
        time=time,
        embedding=embedding,
        pca=pca_values,
        activity_low=low,
        activity_high=high,
        layer_index=np.repeat(np.arange(len(layers)), 721),
        u=u,
        v=v,
    )
    report = {
        "status": "complete",
        "passed": gate,
        "scope": "3D neuron-response manifold for frozen AxoSim visual-circuit activity under real scenes",
        "claim_boundary": "visual separability and visualization; not anatomy, calibrated fly physiology, camera control, or behavior",
        "config_sha256": _sha256(config),
        "checkpoint_sha256": scored["axosim_checkpoint_sha256"],
        "graph": scored["graph"],
        "selected": selected,
        "retinal_normalization": contract.get(
            "retinal_normalization",
            "native Rec.709 frame luminance; uniform control preserves framewise mean",
        ),
        "layers": list(layers),
        "neurons": len(record_index),
        "scenes": [
            {**scene, **source_metadata}
            for scene, source_metadata in zip(contract["scenes"], media)
        ],
        "natural_scene_classification": dynamic_score,
        "uniform_luminance_classification": uniform_score,
        "natural_minus_uniform_balanced_accuracy": advantage,
        "gate": {
            "minimum_natural_balanced_accuracy": contract["classification"]["minimum_natural_balanced_accuracy"],
            "minimum_advantage_over_uniform": contract["classification"]["minimum_advantage_over_uniform"],
            "passed": gate,
        },
        "embedding": {
            **contract["embedding"],
            **embedding_metrics,
            "object": "one fixed point per neuron based on its response fingerprint across all natural scenes",
        },
        "runtime_wall_seconds": time_module.monotonic() - started,
    }
    report["recording_sha256"] = _sha256(output / "manifold-recording.npz")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _right_gradient():
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(500, WIDTH):
        fraction = (x - 500) / (WIDTH - 500)
        alpha = int(225 * (1 - (1 - fraction) ** 2))
        draw.line((x, 0, x, HEIGHT), fill=(255, 255, 255, alpha))
    return overlay


def _normalize_embedding(embedding):
    result = embedding.astype(np.float64).copy()
    for axis in range(3):
        lo, hi = np.quantile(result[:, axis], (0.01, 0.99))
        result[:, axis] = np.clip((result[:, axis] - (lo + hi) / 2) / max((hi - lo) / 2, 1e-9), -1.2, 1.2)
    return result


def _project(points, angle):
    cy, sy = math.cos(angle), math.sin(angle)
    cx, sx = math.cos(-0.28), math.sin(-0.28)
    rotate_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotate_x = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotated = points @ rotate_y.T @ rotate_x.T
    perspective = 1 / np.clip(3.2 - rotated[:, 2], 1.2, None)
    x = 956 + rotated[:, 0] * 535 * perspective
    y = 430 - rotated[:, 1] * 535 * perspective
    return np.column_stack((x, y)), rotated[:, 2], perspective


def _trailing_mean(values, samples=10):
    cumulative = np.concatenate(
        (np.zeros_like(values[:1]), np.cumsum(values, axis=0)), axis=0
    )
    indices = np.arange(len(values))
    starts = np.maximum(indices + 1 - samples, 0)
    return (cumulative[indices + 1] - cumulative[starts]) / (indices + 1 - starts)[:, None]


def render_video(*, config=DEFAULT_CONFIG, recording, summary, output, fps=30):
    contract = json.loads(Path(config).read_text())
    report = json.loads(Path(summary).read_text())
    if report["status"] != "complete":
        raise ValueError("Manifold experiment is incomplete")
    with np.load(recording) as data:
        dynamic = data["dynamic"].astype(np.float32)
        embedding = _normalize_embedding(data["embedding"])
        low = data["activity_low"].astype(np.float32)
        high = data["activity_high"].astype(np.float32)
        layer_index = data["layer_index"].astype(np.int64)
    flat_activity = dynamic.reshape(len(dynamic), len(dynamic[0]), -1)
    smoothed = np.asarray([_trailing_mean(scene) for scene in flat_activity])
    total_seconds = len(contract["scenes"]) * float(contract["duration_seconds"])
    total_frames = int(round(total_seconds * fps))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    readers = [imageio.get_reader(scene["path"]) for scene in contract["scenes"]]
    gradient = _right_gradient()
    title_font = _font(25, weight=600)
    heading_font = _font(14, weight=600)
    metric_font = _font(18, weight=600)
    small_font = _font(13)
    writer = imageio.get_writer(
        output / "real-scenes-neuron-manifold.mp4",
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=1,
    )
    try:
        for frame in range(total_frames):
            global_time = frame / fps
            scene_index = min(
                int(global_time // contract["duration_seconds"]),
                len(contract["scenes"]) - 1,
            )
            local_time = global_time - scene_index * contract["duration_seconds"]
            scene = contract["scenes"][scene_index]
            media = report["scenes"][scene_index]
            source_frame = int((scene["start_seconds"] + local_time) * media["fps"])
            background = Image.fromarray(readers[scene_index].get_data(source_frame)).convert("RGBA")
            background = background.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
            background.alpha_composite(gradient)
            draw = ImageDraw.Draw(background, "RGBA")
            activity_index = min(
                int(round(local_time / contract["source_dt_seconds"])),
                smoothed.shape[1] - 1,
            )
            activity = smoothed[scene_index, activity_index]
            normalized = np.clip((activity - low) / np.maximum(high - low, 1e-6), 0, 1)
            angle = 0.18 + global_time / total_seconds * 1.05
            projected, depth, perspective = _project(embedding, angle)
            order = np.argsort(depth)
            for neuron in order:
                x, y = projected[neuron]
                if not (540 <= x < 1272 and 25 <= y < 660):
                    continue
                level = float(normalized[neuron])
                base = LAYER_COLORS[layer_index[neuron]]
                color = np.rint(225 + (base - 225) * (0.2 + 0.8 * level)).astype(int)
                radius = max(1.0, (1.2 + 3.0 * level) * perspective[neuron] * 2.5)
                draw.ellipse(
                    (x-radius, y-radius, x+radius, y+radius),
                    fill=(*color.tolist(), int(55 + 195 * level)),
                )
            # Layer labels follow their projected centroids.
            for layer, name in enumerate(contract["record_layers"]):
                center = projected[layer_index == layer].mean(axis=0)
                draw.text(
                    (float(center[0]) + 8, float(center[1]) - 8),
                    name,
                    font=small_font,
                    fill=(*LAYER_COLORS[layer].astype(int).tolist(), 230),
                )
            draw.text((42, 42), f"REAL-WORLD STIMULUS · {scene['name'].upper()}", font=heading_font, fill=(255, 255, 255, 245), stroke_width=2, stroke_fill=(20, 22, 26, 115))
            draw.text((795, 42), "2,884 ACTUAL AXOSIM NEURONS", font=heading_font, fill=MUTED)
            verdict = "held-out frames separate" if report["passed"] else "negative result"
            draw.text((795, 67), verdict, font=metric_font, fill=(63, 33, 182) if report["passed"] else INK)
            natural = report["natural_scene_classification"]["balanced_accuracy"]
            uniform = report["uniform_luminance_classification"]["balanced_accuracy"]
            draw.text((795, 101), f"held-out clip decoding  {natural:.0%}", font=small_font, fill=INK)
            control_label = (
                "constant-gray control"
                if isinstance(report["retinal_normalization"], dict)
                else "uniform-luminance control"
            )
            draw.text((795, 126), f"{control_label}  {uniform:.0%}", font=small_font, fill=MUTED)
            draw.text((795, 151), f"UMAP trustworthiness  {report['embedding']['trustworthiness']:.3f}", font=small_font, fill=MUTED)
            draw.rounded_rectangle((22, 655, 337, 709), radius=13, fill=(255, 255, 255, 235))
            draw.text((42, 672), "AxoSim - Axym Labs", font=title_font, fill=INK)
            writer.append_data(np.asarray(background.convert("RGB")))
            if frame in (0, total_frames // 3, 2 * total_frames // 3, total_frames - 1):
                background.convert("RGB").save(output / f"frame-{frame:03d}.png")
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    metadata = {
        "title": "AxoSim - Axym Labs",
        "video": "real-scenes-neuron-manifold.mp4",
        "frames": total_frames,
        "fps": fps,
        "seconds": total_seconds,
        "resolution": [WIDTH, HEIGHT],
        "summary_sha256": _sha256(summary),
        "recording_sha256": _sha256(recording),
        "passed": report["passed"],
        "scenes": report["scenes"],
        "neurons": report["neurons"],
        "layers": report["layers"],
        "classification": {
            "natural": report["natural_scene_classification"],
            "uniform": report["uniform_luminance_classification"],
        },
        "embedding": report["embedding"],
        "display": "fixed 3D UMAP neuron coordinates; depth-sorted perspective; causal 50-ms activation mean; fixed per-neuron 5th/95th percentile scale",
        "renderer": "Pillow custom perspective renderer with Inter; no Matplotlib",
        "controller_or_motor_output": False,
        "future_activity_shown": False,
    }
    (output / "visual-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--device", default="cuda")
    render = subparsers.add_parser("render")
    render.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    render.add_argument("--recording", type=Path, required=True)
    render.add_argument("--summary", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_experiment(
            config=args.config,
            root=args.root,
            output=args.output,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    else:
        result = render_video(
            config=args.config,
            recording=args.recording,
            summary=args.summary,
            output=args.output,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
