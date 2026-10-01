"""Run and render a 3D AxoSim visual-neuron manifold under real scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time as time_module
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
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
    first = math.floor(start * fps)
    time = np.arange(0, duration, dt, dtype=np.float64)
    positions = (start + time) * fps - first
    lower = np.floor(positions).astype(int)
    last = first + int(lower.max()) + 1
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
                0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
            )
            frames.append(
                map_coordinates(luminance, (pixel_y, pixel_x), order=1, mode="nearest")
            )
    finally:
        reader.close()
    frames = np.asarray(frames, dtype=np.float32)
    alpha = (positions - lower)[:, None]
    stimulus = frames[lower] * (1 - alpha) + frames[lower + 1] * alpha
    return (
        stimulus.astype(np.float32),
        time,
        {
            "fps": fps,
            "source_size": [width, height],
            "decoded_frame_range_inclusive": [first, last],
        },
    )


def _classify_scene(activity, time, contract):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = contract["classification"]
    step = round(cfg["sample_seconds"] / contract["source_dt_seconds"])
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


def _embed_states(activity, time, contract):
    """Embed population states, so every manifold point is one moment in time."""
    import umap
    from sklearn.decomposition import PCA
    from sklearn.manifold import trustworthiness

    cfg = contract["embedding"]
    step = round(cfg["sample_seconds"] / contract["source_dt_seconds"])
    sampled_time = time[::step]
    mask = sampled_time >= contract["warmup_seconds"]
    sampled_time = sampled_time[mask]
    sampled = activity[:, ::step][:, mask]
    states = sampled.reshape(-1, sampled.shape[-2] * sampled.shape[-1]).astype(
        np.float64
    )
    mean = states.mean(axis=0, keepdims=True)
    std = states.std(axis=0, keepdims=True)
    constant = std[0] < 1e-8
    standardized = np.divide(
        states - mean,
        std,
        out=np.zeros_like(states),
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
    flat_embedding = reducer.fit_transform(pca_values).astype(np.float32)
    quality = float(
        trustworthiness(
            pca_values,
            flat_embedding,
            n_neighbors=int(cfg["trustworthiness_neighbors"]),
        )
    )
    embedding = flat_embedding.reshape(len(activity), len(sampled_time), 3)
    return (
        embedding,
        sampled_time.astype(np.float32),
        pca_values,
        {
            "constant_features": int(constant.sum()),
            "pca_explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "trustworthiness": quality,
        },
    )


def _analyze_settling_and_oscillation(
    activity,
    time,
    pca_values,
    state_time,
    contract,
):
    """Measure approach to a late orbit and its spectral stability."""
    from scipy.spatial.distance import cdist
    from sklearn.decomposition import PCA

    cfg = contract["dynamics"]
    image_indices = [
        index
        for index, scene in enumerate(contract["scenes"])
        if scene["kind"] in {"natural-image", "voyager-record"}
    ]
    state = pca_values.reshape(len(activity), len(state_time), -1)[..., :10]
    transition = (state_time >= cfg["transition_approach_seconds"][0]) & (
        state_time < cfg["transition_approach_seconds"][1]
    )
    post_transition = (state_time >= cfg["post_transition_seconds"][0]) & (
        state_time < cfg["post_transition_seconds"][1]
    )
    orbit = (state_time >= cfg["reference_orbit_seconds"][0]) & (
        state_time < cfg["reference_orbit_seconds"][1]
    )
    analysis = (time >= cfg["oscillation_seconds"][0]) & (
        time < cfg["oscillation_seconds"][1]
    )
    per_scene = []
    for scene_index in image_indices:
        reference = state[scene_index, orbit]
        approach_ratio = None
        if scene_index > 0:
            transition_distance = np.median(
                cdist(state[scene_index - 1, transition], reference).min(axis=1)
            )
            post_transition_distance = np.median(
                cdist(state[scene_index, post_transition], reference).min(axis=1)
            )
            approach_ratio = float(
                transition_distance / max(post_transition_distance, 1e-12)
            )
        values = activity[scene_index, analysis].reshape(int(analysis.sum()), -1)
        std = values.std(axis=0)
        variable = std > 1e-8
        standardized = (values[:, variable] - values[:, variable].mean(axis=0)) / std[
            variable
        ]
        component = PCA(n_components=3, random_state=0).fit_transform(standardized)[
            :, 0
        ]
        halves = np.array_split(component, 2)
        spectra = []
        for half in halves:
            index = np.arange(len(half), dtype=np.float64)
            residual = half - np.polyval(np.polyfit(index, half, 1), index)
            power = np.abs(np.fft.rfft(residual)) ** 2
            frequency = np.fft.rfftfreq(len(residual), contract["source_dt_seconds"])
            power[0] = 0
            peak = int(np.argmax(power))
            spectra.append(
                {
                    "peak_hz": float(frequency[peak]),
                    "peak_power_fraction": float(power[peak] / power.sum()),
                    "amplitude_standard_deviation": float(residual.std()),
                }
            )
        per_scene.append(
            {
                "name": contract["scenes"][scene_index]["name"],
                "transition_to_post_transition_orbit_distance_ratio": approach_ratio,
                "spectral_halves": spectra,
                "peak_frequency_drift_hz": abs(
                    spectra[0]["peak_hz"] - spectra[1]["peak_hz"]
                ),
                "amplitude_ratio": spectra[1]["amplitude_standard_deviation"]
                / spectra[0]["amplitude_standard_deviation"],
            }
        )
    passed = all(
        (
            scene["transition_to_post_transition_orbit_distance_ratio"] is None
            or scene["transition_to_post_transition_orbit_distance_ratio"]
            >= cfg["minimum_orbit_approach_ratio"]
        )
        and scene["peak_frequency_drift_hz"] <= cfg["maximum_peak_frequency_drift_hz"]
        and min(half["peak_power_fraction"] for half in scene["spectral_halves"])
        >= cfg["minimum_peak_power_fraction"]
        and cfg["amplitude_ratio"][0]
        <= scene["amplitude_ratio"]
        <= cfg["amplitude_ratio"][1]
        for scene in per_scene
    )
    return {
        "passed": passed,
        "scope": f"{len(per_scene) - 1} cross-image transitions followed by {len(per_scene)} sustained image intervals",
        "interpretation": "population states approach a late recurrent orbit across each image transition and retain a stable dominant oscillation under constant visual input",
        "claim_boundary": "descriptive evidence for a stable oscillatory trajectory; a limit-cycle attractor requires perturbation-and-return validation",
        "method": cfg,
        "median_transition_to_post_transition_orbit_distance_ratio": float(
            np.median(
                [
                    scene["transition_to_post_transition_orbit_distance_ratio"]
                    for scene in per_scene
                    if scene["transition_to_post_transition_orbit_distance_ratio"]
                    is not None
                ]
            )
        ),
        "per_scene": per_scene,
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
        [
            np.asarray(getattr(network.connectome.nodes.layer_index, name)[:])
            for name in layers
        ]
    )
    u, v, retinal_x, retinal_y = _retinal_coordinates(network)
    stimuli, media = [], []
    source_dt = float(contract["source_dt_seconds"])
    started = time_module.monotonic()
    time = None
    normalization = contract.get("retinal_normalization")
    for scene in contract["scenes"]:
        print(f"scene: {scene['name']}", flush=True)
        stimulus, time, source_metadata = _sample_scene(
            scene,
            duration=float(contract["duration_seconds"]),
            dt=source_dt,
            retinal_x=retinal_x,
            retinal_y=retinal_y,
        )
        if normalization is not None:
            frame_mean = stimulus.mean(axis=1, keepdims=True)
            frame_std = stimulus.std(axis=1, keepdims=True)
            stimulus = (stimulus - frame_mean) / np.maximum(frame_std, 1e-6) * float(
                normalization["standard_deviation"]
            ) + float(normalization["mean"])
            stimulus = np.clip(
                stimulus,
                float(normalization["clip"][0]),
                float(normalization["clip"][1]),
            ).astype(np.float32)
        stimuli.append(stimulus)
        media.append(source_metadata)

    # Run the film as one continuous experiment. Each generated segment contains
    # one second of leading context. The first context initializes the circuit;
    # later contexts overlap the previous segment's completed transition, so
    # they are not advanced twice. The analysis windows retain that preceding
    # second and introduce no hidden-state reset at a visible boundary.
    warmup_steps = round(float(contract["warmup_seconds"]) / source_dt)
    stride = len(time) - warmup_steps
    continuous_stimulus = np.concatenate(
        [stimuli[0], *[stimulus[warmup_steps:] for stimulus in stimuli[1:]]]
    )
    _, continuous_response = runtime.run_stimulus(
        continuous_stimulus,
        source_dt=source_dt,
        target_dt=vision_contract["protocol"]["axosim_dt_seconds"],
        record_index=record_index,
    )
    continuous_response = continuous_response.reshape(-1, len(layers), 721)
    dynamic = np.asarray(
        [
            continuous_response[index * stride : index * stride + len(time)]
            for index in range(len(stimuli))
        ],
        dtype=np.float32,
    )
    if normalization is None:
        uniform_stimulus = np.repeat(
            continuous_stimulus.mean(axis=1, keepdims=True), 721, axis=1
        )
    else:
        uniform_stimulus = np.full_like(
            continuous_stimulus, float(normalization["mean"])
        )
    _, continuous_uniform = runtime.run_stimulus(
        uniform_stimulus,
        source_dt=source_dt,
        target_dt=vision_contract["protocol"]["axosim_dt_seconds"],
        record_index=record_index,
    )
    continuous_uniform = continuous_uniform.reshape(-1, len(layers), 721)
    uniform = np.asarray(
        [
            continuous_uniform[index * stride : index * stride + len(time)]
            for index in range(len(stimuli))
        ],
        dtype=np.float32,
    )
    if not np.isfinite(dynamic).all() or not np.isfinite(uniform).all():
        raise RuntimeError("Nonfinite AxoSim manifold recording")
    dynamic_score = _classify_scene(dynamic, time, contract)
    uniform_score = _classify_scene(uniform, time, contract)
    state_embedding, state_time, pca_values, embedding_metrics = _embed_states(
        dynamic, time, contract
    )
    dynamics = _analyze_settling_and_oscillation(
        dynamic,
        time,
        pca_values,
        state_time,
        contract,
    )
    advantage = dynamic_score["balanced_accuracy"] - uniform_score["balanced_accuracy"]
    gate = (
        dynamic_score["balanced_accuracy"]
        >= contract["classification"]["minimum_stimulus_balanced_accuracy"]
        and advantage >= contract["classification"]["minimum_advantage_over_uniform"]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shards = []
    for scene_index, scene_dynamic in enumerate(dynamic):
        shard = output / f"manifold-dynamic-{scene_index + 1:02d}.npz"
        np.savez_compressed(shard, dynamic=scene_dynamic)
        shards.append(
            {
                "file": shard.name,
                "scene": contract["scenes"][scene_index]["name"],
                "shape": list(scene_dynamic.shape),
                "dtype": str(scene_dynamic.dtype),
                "sha256": _sha256(shard),
                "bytes": shard.stat().st_size,
            }
        )
    np.savez_compressed(
        output / "manifold-recording.npz",
        uniform_classification_activity=uniform[
            :, :: round(contract["classification"]["sample_seconds"] / source_dt)
        ].astype(np.float32),
        time=time,
        state_embedding=state_embedding,
        state_time=state_time,
        pca=pca_values,
        layer_index=np.repeat(np.arange(len(layers)), 721),
        u=u,
        v=v,
    )
    recording_manifest = {
        "format": "exact-float32-scene-shards-v1",
        "core_file": "manifold-recording.npz",
        "core_sha256": _sha256(output / "manifold-recording.npz"),
        "dynamic_shards": shards,
        "raw_dynamic_total_bytes": int(sum(item["bytes"] for item in shards)),
        "packaging": "raw 5 ms shards are hash-pinned and regenerable; the compact core contains exact 25 ms score samples and embedding coordinates",
    }
    (output / "recording-manifest.json").write_text(
        json.dumps(recording_manifest, indent=2) + "\n"
    )
    report = {
        "status": "complete",
        "passed": gate and dynamics["passed"],
        "scope": "3D neuron-response manifold for frozen AxoSim visual-circuit activity under natural scenes and Voyager Golden Record images",
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
        "stimulus_segment_classification": dynamic_score,
        "uniform_luminance_classification": uniform_score,
        "stimulus_minus_uniform_balanced_accuracy": advantage,
        "gate": {
            "minimum_stimulus_balanced_accuracy": contract["classification"][
                "minimum_stimulus_balanced_accuracy"
            ],
            "minimum_advantage_over_uniform": contract["classification"][
                "minimum_advantage_over_uniform"
            ],
            "passed": gate,
        },
        "embedding": {
            **contract["embedding"],
            **embedding_metrics,
            "object": f"one point per sampled 2,884-neuron population state; {len(contract['scenes'])} fixed trajectories correspond to natural-scene and Voyager image segments",
        },
        "settling_and_oscillation": dynamics,
        "temporal_execution": f"one continuous {float(contract['warmup_seconds']) + len(contract['scenes']) * (float(contract['duration_seconds']) - float(contract['warmup_seconds'])):.0f}-second AxoSim run; {len(contract['scenes'])} overlapping analysis windows retain one second of preceding transition context and contain no state reset at a visible boundary",
        "uniform_control_storage": "one continuous exact mean-gray control run is retained and windowed at the same absolute times as the stimulus program",
        "raw_recording": {
            "manifest": "recording-manifest.json",
            "manifest_sha256": _sha256(output / "recording-manifest.json"),
            "dynamic_shards": len(shards),
            "raw_dynamic_total_bytes": recording_manifest["raw_dynamic_total_bytes"],
        },
        "runtime_wall_seconds": time_module.monotonic() - started,
    }
    report["recording_sha256"] = _sha256(output / "manifold-recording.npz")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _right_gradient():
    """Fade real footage into a dark neural-space canvas without a panel edge."""
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (7, 8, 13, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(540, WIDTH):
        fraction = (x - 540) / (WIDTH - 540)
        alpha = int(246 * (fraction**0.72))
        draw.line((x, 0, x, HEIGHT), fill=(7, 8, 13, alpha))
    return overlay


def _normalize_embedding(embedding):
    result = embedding.astype(np.float64).copy()
    flat = result.reshape(-1, result.shape[-1])
    for axis in range(3):
        lo, hi = np.quantile(flat[:, axis], (0.01, 0.99))
        result[..., axis] = np.clip(
            (result[..., axis] - (lo + hi) / 2) / max((hi - lo) / 2, 1e-9),
            -1.2,
            1.2,
        )
    return result


def _project(points):
    """Project a fixed camera; motion in the film must come only from activity."""
    angle = 0.58
    cy, sy = math.cos(angle), math.sin(angle)
    cx, sx = math.cos(-0.36), math.sin(-0.36)
    rotate_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotate_x = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotated = points @ rotate_y.T @ rotate_x.T
    perspective = 1 / np.clip(3.45 - rotated[:, 2], 1.35, None)
    x = 1002 + rotated[:, 0] * 520 * perspective
    y = 384 - rotated[:, 1] * 520 * perspective
    return np.column_stack((x, y)), rotated[:, 2], perspective


def _causal_smooth_path(points, *, dt, time_constant_seconds=0.18):
    """Low-pass a displayed trajectory without consulting future states."""
    output = np.empty_like(points, dtype=np.float32)
    output[0] = points[0]
    rate = 1 - math.exp(-dt / time_constant_seconds)
    for index in range(1, len(points)):
        output[index] = output[index - 1] + rate * (points[index] - output[index - 1])
    return output


def _ramp(level):
    """Goodfire-inspired violet → magenta → gold activation ramp."""
    anchors = np.asarray(
        [(63, 33, 182), (113, 67, 214), (222, 51, 154), (255, 196, 87)],
        dtype=np.float64,
    )
    position = float(np.clip(level, 0, 1)) * (len(anchors) - 1)
    index = min(int(position), len(anchors) - 2)
    fraction = position - index
    return np.rint(
        anchors[index] * (1 - fraction) + anchors[index + 1] * fraction
    ).astype(int)


def _draw_manifold(image, projected_paths, depth_paths, active_scene, trail):
    """Draw the fixed state manifold and a bright causal path through it."""
    draw = ImageDraw.Draw(image, "RGBA")
    for scene_index, path in enumerate(projected_paths):
        valid = [
            tuple(point)
            for point in path
            if 650 <= point[0] < 1274 and 36 <= point[1] < 650
        ]
        if len(valid) > 1:
            alpha = 82 if scene_index == active_scene else 38
            draw.line(valid, fill=(121, 104, 181, alpha), width=2)
        order = np.argsort(depth_paths[scene_index])
        for point_index in order[::2]:
            x, y = path[point_index]
            if 650 <= x < 1274 and 36 <= y < 650:
                alpha = 116 if scene_index == active_scene else 58
                draw.ellipse(
                    (x - 1.7, y - 1.7, x + 1.7, y + 1.7), fill=(177, 171, 205, alpha)
                )

    if len(trail) > 1:
        halo = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        halo_draw = ImageDraw.Draw(halo, "RGBA")
        halo_draw.line(
            [tuple(point) for point in trail], fill=(225, 49, 161, 110), width=15
        )
        halo = halo.filter(ImageFilter.GaussianBlur(10))
        image.alpha_composite(halo)
        draw = ImageDraw.Draw(image, "RGBA")
        for index in range(1, len(trail)):
            age = index / (len(trail) - 1)
            color = _ramp(age)
            draw.line(
                (tuple(trail[index - 1]), tuple(trail[index])),
                fill=(*color.tolist(), int(40 + 205 * age)),
                width=max(3, int(3 + 5 * age)),
            )
        x, y = trail[-1]
        halo = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        halo_draw = ImageDraw.Draw(halo, "RGBA")
        halo_draw.ellipse((x - 28, y - 28, x + 28, y + 28), fill=(255, 103, 173, 190))
        halo = halo.filter(ImageFilter.GaussianBlur(15))
        image.alpha_composite(halo)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 239, 199, 255))


def render_video(*, config=DEFAULT_CONFIG, recording, summary, output, fps=30):
    contract = json.loads(Path(config).read_text())
    report = json.loads(Path(summary).read_text())
    if report["status"] != "complete":
        raise ValueError("Manifold experiment is incomplete")
    with np.load(recording) as data:
        state_embedding = data["state_embedding"].astype(np.float32)
        state_time = data["state_time"].astype(np.float32)
    state_embedding = _causal_smooth_path(
        state_embedding.reshape(-1, 3),
        dt=float(contract["embedding"]["sample_seconds"]),
    ).reshape(state_embedding.shape)
    state_embedding = _normalize_embedding(state_embedding)
    projected_paths = []
    depth_paths = []
    for scene_embedding in state_embedding:
        projected, depth, _ = _project(scene_embedding)
        projected_paths.append(projected)
        depth_paths.append(depth)
    projected_paths = np.asarray(projected_paths)
    depth_paths = np.asarray(depth_paths)
    continuous_projected_path = projected_paths.reshape(-1, 2)
    display_start = max(float(contract["warmup_seconds"]), 1.0)
    scene_seconds = float(contract["duration_seconds"]) - display_start
    total_seconds = len(contract["scenes"]) * scene_seconds
    total_frames = round(total_seconds * fps)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    readers = [imageio.get_reader(scene["path"]) for scene in contract["scenes"]]
    gradient = _right_gradient()
    title_font = _font(25, weight=600)
    writer = imageio.get_writer(
        output / "axosim-neural-landscape.mp4",
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        ffmpeg_params=[
            "-preset",
            "slow",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
        ],
    )
    try:
        for frame in range(total_frames):
            global_time = frame / fps
            scene_index = min(
                int(global_time // scene_seconds),
                len(contract["scenes"]) - 1,
            )
            local_time = display_start + global_time - scene_index * scene_seconds
            scene = contract["scenes"][scene_index]
            media = report["scenes"][scene_index]
            source_position = (scene["start_seconds"] + local_time) * media["fps"]
            source_frame = math.floor(source_position)
            source_alpha = source_position - source_frame
            frame_lower = Image.fromarray(
                readers[scene_index].get_data(source_frame)
            ).convert("RGBA")
            frame_upper = Image.fromarray(
                readers[scene_index].get_data(source_frame + 1)
            ).convert("RGBA")
            background = Image.blend(frame_lower, frame_upper, source_alpha)
            background = background.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
            background.alpha_composite(gradient)
            state_index = int(np.searchsorted(state_time, local_time, side="right") - 1)
            state_index = min(max(state_index, 0), len(state_time) - 1)
            global_state_index = scene_index * len(state_time) + state_index
            trail_start = max(
                0,
                global_state_index
                - round(1.6 / contract["embedding"]["sample_seconds"]),
            )
            trail = continuous_projected_path[trail_start : global_state_index + 1]
            _draw_manifold(
                background,
                projected_paths,
                depth_paths,
                scene_index,
                trail,
            )
            draw = ImageDraw.Draw(background, "RGBA")
            draw.text(
                (WIDTH - 58, HEIGHT - 46),
                "AxoSim - Axym Labs",
                font=title_font,
                fill=(255, 255, 255, 255),
                anchor="rs",
            )
            writer.append_data(np.asarray(background.convert("RGB")))
            if frame in (0, total_frames // 3, 2 * total_frames // 3, total_frames - 1):
                background.convert("RGB").save(output / f"frame-{frame:03d}.png")
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    metadata = {
        "title": "AxoSim - Axym Labs",
        "video": "axosim-neural-landscape.mp4",
        "frames": total_frames,
        "fps": fps,
        "seconds": total_seconds,
        "source_segments": len(contract["scenes"]),
        "resolution": [WIDTH, HEIGHT],
        "summary_sha256": _sha256(summary),
        "recording_sha256": _sha256(recording),
        "passed": report["passed"],
        "scenes": report["scenes"],
        "neurons": report["neurons"],
        "layers": report["layers"],
        "classification": {
            "stimulus_segments": report["stimulus_segment_classification"],
            "uniform": report["uniform_luminance_classification"],
        },
        "embedding": report["embedding"],
        "display": "fixed-camera temporal 3D UMAP; every point is one 2,884-neuron population state from a continuous AxoSim run; persistent full trajectories; cross-boundary causal 180-ms display smoothing and causal 1.6-second current-state trail; no activity field is drawn over the source image",
        "activity_color": "violet-magenta-gold marks trail recency only; it never denotes cell type",
        "display_start_seconds_per_scene": display_start,
        "camera_motion": False,
        "activity_blinking": False,
        "source_image_activity_overlay": False,
        "neural_state_resets_at_visible_boundaries": False,
        "text_in_frame": ["AxoSim - Axym Labs"],
        "title_style": {
            "font": "Inter",
            "fill_rgba": [255, 255, 255, 255],
            "outline": False,
            "anchor": "right baseline",
            "position_pixels": [WIDTH - 58, HEIGHT - 46],
            "right_margin_pixels": 58,
            "bottom_baseline_margin_pixels": 46,
        },
        "renderer": "Pillow custom perspective renderer with Inter; no Matplotlib",
        "encoding": "H.264 High, CRF 20, yuv420p, fast-start",
        "source_frame_sampling": "linear temporal interpolation at original playback speed; no segment looping or slow motion",
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
