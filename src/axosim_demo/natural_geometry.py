"""Measure and render AxoSim optic-flow geometry on natural photographs.

Every neural quantity in this module is a recorded AxoSim soma-voltage output.
The only experimental manipulation is a known translation of a photograph over
the FlyVis retinal lattice.  Motor policies and hand-authored neural responses
are deliberately outside this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
from scipy import sparse
from scipy.ndimage import map_coordinates, sobel
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import pearsonr, spearmanr

from .natural_manifold import _retinal_coordinates
from .vision_axosim import AxoSimFlyVis


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/natural_geometry.json"
PAPER = "#F4F1EA"
CARD = "#FFFFFF"
PLOT = "#E5EDF8"
INK = "#272727"
MUTED = "#6D6A65"
BLUE = "#397DD5"
GREEN = "#67AE4E"
MAGENTA = "#D93B9F"
ORANGE = "#F0932B"
CYCLIC = "twilight_shifted"
SOURCE_COLORS = {"imagenet": "#397DD5", "inaturalist": "#D93B9F"}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_soma_to_mv(values, *, bias_mv=-67.7, scale_per_mv=0.1):
    """Invert the normalization stored with the frozen AxoSim checkpoint."""
    return np.asarray(values, dtype=np.float32) / float(scale_per_mv) + float(bias_mv)


def circular_distance_degrees(left, right):
    """Shortest unsigned separation between angles in degrees."""
    delta = np.abs(np.asarray(left) - np.asarray(right)) % 360.0
    return np.minimum(delta, 360.0 - delta)


def within_image_center(values, image_index):
    """Remove each photograph's angle-averaged response from its trials."""
    values = np.asarray(values, dtype=np.float32)
    image_index = np.asarray(image_index)
    result = values.copy()
    for group in np.unique(image_index):
        mask = image_index == group
        result[mask] -= result[mask].mean(axis=0, keepdims=True)
    return result


def _load_contract(config):
    config = Path(config)
    return config, json.loads(config.read_text())


def _select_source_rows(rows, count, analysis_count, development_count, seed):
    """Select one image per class/species, then sample deterministically."""
    unique = []
    seen = set()
    for row in rows:
        key = row.get("leaf", row.get("display_name", row["image_id"]))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if len(unique) < development_count:
        raise RuntimeError(
            f"need {development_count} distinct development labels, found {len(unique)}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    distinct = list(np.asarray(unique, dtype=object)[order[: min(count, len(unique))]])
    selected_ids = {row["image_id"] for row in distinct}
    remaining = [row for row in rows if row["image_id"] not in selected_ids]
    if len(distinct) < count:
        extra_order = rng.permutation(len(remaining))[: count - len(distinct)]
        distinct.extend(np.asarray(remaining, dtype=object)[extra_order])
    chosen = np.asarray(distinct, dtype=object)
    development_ids = {
        row["image_id"]
        for row in np.asarray(unique, dtype=object)[order[:development_count]]
    }
    analysis_ids = {row["image_id"] for row in chosen[:analysis_count]}
    return [
        (
            row,
            "development"
            if row["image_id"] in development_ids
            else "final"
            if row["image_id"] in analysis_ids
            else "display-only",
        )
        for row in sorted(chosen, key=lambda row: row["image_id"])
    ]


def prepare_data(config=DEFAULT_CONFIG):
    """Create a compact, hash-pinned two-source natural-image manifest."""
    config_path, contract = _load_contract(config)
    input_root = Path(contract["input_root"])
    output_root = Path(contract["output_root"])
    count = int(contract["images_per_source"])
    analysis_count = int(contract.get("analysis_images_per_source", count))
    development_count = int(contract["development_images_per_source"])
    if not development_count < analysis_count <= count:
        raise ValueError(
            "expected development_images_per_source < analysis_images_per_source "
            "<= images_per_source"
        )
    selected = []
    source_manifests = {}
    for offset, source in enumerate(("imagenet", "inaturalist")):
        source_path = input_root / source / "manifest.json"
        source_manifest = json.loads(source_path.read_text())
        source_manifests[source] = {
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_path),
        }
        rows = _select_source_rows(
            source_manifest["rows"],
            count,
            analysis_count,
            development_count,
            int(contract["seed"]) + offset,
        )
        pinned = contract.get("analysis_image_ids", {}).get(source)
        if pinned:
            development_ids = set(pinned["development"])
            final_ids = set(pinned["final"])
            selected_ids = {row["image_id"] for row, _ in rows}
            missing = (development_ids | final_ids) - selected_ids
            if missing:
                raise RuntimeError(
                    f"pinned {source} analysis images absent from display cohort: "
                    f"{sorted(missing)}"
                )
        for row, default_split in rows:
            evaluation_split = (
                "development"
                if pinned and row["image_id"] in development_ids
                else "final"
                if pinned and row["image_id"] in final_ids
                else "display-only"
                if pinned
                else default_split
            )
            path = Path(row["path"])
            if _sha256(path) != row["sha256"]:
                raise ValueError(f"image hash mismatch: {path}")
            selected.append(
                {
                    "base_image_index": len(selected),
                    "evaluation_split": evaluation_split,
                    "source": source,
                    "image_id": row["image_id"],
                    "display_name": row.get("display_name", row.get("leaf", "image")),
                    "label_path": [
                        str(row.get("level0", source)),
                        str(row.get("level1", "dataset label")),
                        str(row.get("level2", row.get("leaf", "class"))),
                        str(row.get("display_name", row.get("leaf", "image"))),
                    ],
                    "path": str(path.resolve()),
                    "sha256": row["sha256"],
                    "source_url": row.get("source_url", "https://www.image-net.org/"),
                    "license": row.get("license", "ImageNet terms"),
                    "attribution": row.get("attribution", ""),
                }
            )
    manifest = {
        "scope": "natural photographs used as controlled optic-flow carriers",
        "selection": "one image per distinct ImageNet class or iNaturalist species; deterministic seeded sample",
        "evaluation_protocol": "per source, development_images_per_source images are used for representation choice and the next images up to analysis_images_per_source form the untouched final split",
        "display_protocol": "analysis_images_per_source controls the development/final scientific cohort; remaining images are used only for unsupervised raw-activation figures",
        "seed": contract["seed"],
        "source_manifests": source_manifests,
        "rows": selected,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "status": "complete",
        "config_sha256": _sha256(config_path),
        "manifest_sha256": _sha256(manifest_path),
        "images": len(selected),
        "images_per_source": {
            source: sum(row["source"] == source for row in selected)
            for source in SOURCE_COLORS
        },
        "analysis_images_per_source": {
            source: sum(
                row["source"] == source and row["evaluation_split"] != "display-only"
                for row in selected
            )
            for source in SOURCE_COLORS
        },
    }
    (output_root / "preparation-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _load_rgb_and_stats(path, size):
    with Image.open(path) as image:
        rgb_image = ImageOps.fit(
            ImageOps.exif_transpose(image).convert("RGB"),
            (size, size),
            Image.Resampling.LANCZOS,
        )
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    stats = np.asarray(
        [
            luminance.mean(),
            luminance.std(),
            np.hypot(sobel(luminance, 0), sobel(luminance, 1)).mean(),
            (rgb.max(axis=2) - rgb.min(axis=2)).mean(),
        ],
        np.float32,
    )
    return rgb, luminance, stats


def _motion_stimulus(luminance, retinal_x, retinal_y, angle_deg, cfg):
    """Translate an image at constant speed and sample the fly retinal lattice."""
    size = int(cfg["image_size_pixels"])
    radius = float(cfg["retinal_radius_fraction"])
    px = (0.5 + radius * np.asarray(retinal_x)) * (size - 1)
    py = (0.5 + radius * np.asarray(retinal_y)) * (size - 1)
    distance = np.linspace(
        -float(cfg["travel_pixels"]) / 2,
        float(cfg["travel_pixels"]) / 2,
        int(cfg["motion_frames"]),
    )
    angle = np.deg2rad(float(angle_deg))
    frames = [
        map_coordinates(
            luminance,
            (py + step * np.sin(angle), px + step * np.cos(angle)),
            order=1,
            mode="reflect",
        )
        for step in distance
    ]
    gray = np.full((int(cfg["gray_frames"]), 721), float(cfg["gray_level"]), np.float32)
    return np.concatenate((gray, np.asarray(frames, np.float32)))


def _load_runtime(contract, device):
    vision_contract = json.loads(Path(contract["vision_config"]).read_text())
    scored = json.loads(Path(contract["scored_summary"]).read_text())
    if not scored["video_gate_passed"]:
        raise RuntimeError("the controlled AxoSim visual-circuit arm has not passed")
    os.environ["FLYVIS_ROOT_DIR"] = str(Path(contract["flyvis_root"]).resolve())
    from flyvis import results_dir
    from flyvis.network import NetworkView

    view = NetworkView(results_dir / vision_contract["flyvis_network"])
    network = view.init_network("best")
    selected = scored["selected"]
    runtime = AxoSimFlyVis(
        network,
        checkpoint=contract["checkpoint"],
        morphology_index=selected["morphology_index"],
        edge_current_gain=selected["edge_current_gain"],
        device=device,
    )
    layers = tuple(contract["record_layers"])
    record_index = np.concatenate(
        [
            np.asarray(getattr(network.connectome.nodes.layer_index, layer)[:])
            for layer in layers
        ]
    )
    return (
        runtime,
        record_index,
        _retinal_coordinates(network),
        scored,
        vision_contract,
    )


def record_voltages(config=DEFAULT_CONFIG, *, device="cuda"):
    """Record actual T4/T5 AxoSim soma voltages for every image/direction trial."""
    config_path, contract = _load_contract(config)
    root = Path(contract["output_root"])
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        prepare_data(config_path)
    manifest = json.loads(manifest_path.read_text())
    runtime, record_index, retinal, scored, vision_contract = _load_runtime(
        contract, device
    )
    u, v, retinal_x, retinal_y = retinal
    cfg = contract["stimulus"]
    angles = np.asarray(cfg["directions_degrees"], np.float32)
    dt = float(cfg["dt_seconds"])
    final = int(cfg["average_final_frames"])
    voltage, image_index, direction, image_stats = [], [], [], []
    started = time.monotonic()
    for image_number, row in enumerate(manifest["rows"]):
        _, luminance, stats = _load_rgb_and_stats(
            row["path"], int(cfg["image_size_pixels"])
        )
        for angle in angles:
            stimulus = _motion_stimulus(
                luminance, retinal_x, retinal_y, float(angle), cfg
            )
            _, normalized = runtime.run_stimulus(
                stimulus,
                source_dt=dt,
                target_dt=float(vision_contract["protocol"]["axosim_dt_seconds"]),
                record_index=record_index,
            )
            mv = normalized_soma_to_mv(
                normalized,
                bias_mv=cfg["soma_bias_mv"],
                scale_per_mv=cfg["soma_scale_per_mv"],
            )
            voltage.append(mv[-final:].mean(axis=0))
            image_index.append(image_number)
            direction.append(angle)
            image_stats.append(stats)
        print(
            f"recorded image {image_number + 1}/{len(manifest['rows'])} "
            f"({row['source']})",
            flush=True,
        )
    voltage = np.asarray(voltage, np.float32)
    image_index = np.asarray(image_index, np.int16)
    direction = np.asarray(direction, np.float32)
    result_path = root / "axosim-optic-flow-voltage.npz"
    np.savez_compressed(
        result_path,
        voltage_mv=voltage,
        image_index=image_index,
        direction_degrees=direction,
        image_statistics=np.asarray(image_stats, np.float32),
        statistic_names=np.asarray(
            ["mean luminance", "RMS contrast", "edge energy", "mean saturation"]
        ),
        layer_index=np.repeat(np.arange(len(contract["record_layers"])), 721),
        layers=np.asarray(contract["record_layers"]),
        u=u,
        v=v,
    )
    summary = {
        "status": "complete",
        "scope": "frozen pretrained AxoSim neurons on the exact expanded FlyVis graph under translated natural photographs",
        "quantity": "AxoSim output channel 1, inverse-normalized soma membrane voltage in millivolts",
        "conversion": "V_mV = normalized_output / 0.1 - 67.7",
        "config_sha256": _sha256(config_path),
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint_sha256": scored["axosim_checkpoint_sha256"],
        "graph": scored["graph"],
        "selected_axosim_substitution": scored["selected"],
        "images": len(manifest["rows"]),
        "directions": angles.tolist(),
        "trials": len(voltage),
        "recorded_neurons": int(voltage.shape[1]),
        "record_layers": contract["record_layers"],
        "voltage_min_mv": float(voltage.min()),
        "voltage_max_mv": float(voltage.max()),
        "voltage_mean_mv": float(voltage.mean()),
        "voltage_std_mv": float(voltage.std()),
        "recording_sha256": _sha256(result_path),
        "wall_seconds": time.monotonic() - started,
        "controller_or_motor_output": False,
    }
    (root / "recording-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _standardize_train(values, train_mask):
    mean = values[train_mask].mean(axis=0)
    std = values[train_mask].std(axis=0)
    keep = std > 1e-7
    if keep.sum() < 10:
        raise RuntimeError("too few nonconstant voltage features")
    return (
        ((values[:, keep] - mean[keep]) / std[keep]).astype(np.float32),
        keep,
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def _umap_pair(values, cfg, seed):
    from sklearn.decomposition import PCA
    import umap

    n_components = min(int(cfg["pca_components"]), len(values) - 1, values.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    compact = pca.fit_transform(values)
    kwargs = {
        "n_neighbors": min(int(cfg["umap_neighbors"]), len(values) - 1),
        "min_dist": float(cfg["umap_min_dist"]),
        "metric": "cosine",
        "random_state": seed,
        "n_jobs": 1,
        "init": cfg.get("umap_init", "spectral"),
    }
    two = umap.UMAP(n_components=2, **kwargs).fit_transform(compact)
    three = umap.UMAP(n_components=3, **kwargs).fit_transform(compact)
    return two.astype(np.float32), three.astype(np.float32), pca


def _knn_geodesic(values):
    from sklearn.metrics.pairwise import cosine_similarity

    cosine = np.clip(cosine_similarity(values), -1.0, 1.0)
    angular = np.arccos(cosine)
    n = len(values)
    order = np.argsort(angular, axis=1)
    for k in range(2, min(n, 80)):
        graph = np.full((n, n), np.inf, np.float32)
        np.fill_diagonal(graph, 0)
        neighbor = order[:, 1 : k + 1]
        row = np.repeat(np.arange(n), k)
        col = neighbor.reshape(-1)
        graph[row, col] = angular[row, col]
        graph = np.minimum(graph, graph.T)
        binary = sparse.csr_matrix(np.isfinite(graph) & (graph > 0))
        if connected_components(binary, directed=False, return_labels=False) == 1:
            rr, cc = np.where(np.isfinite(graph) & (graph > 0))
            weighted = sparse.csr_matrix((graph[rr, cc], (rr, cc)), shape=(n, n))
            return cosine, shortest_path(weighted, directed=False), k
    raise RuntimeError("no connected nearest-neighbor graph below k=80")


def _learn_subspace(values, target, pair_i, pair_j, train_mask, cfg, seed, device):
    import torch
    from sklearn.decomposition import PCA

    dims = min(
        int(cfg["subspace_dimensions"]), values.shape[1], int(train_mask.sum()) - 1
    )
    init = PCA(n_components=dims, random_state=seed).fit(values[train_mask]).components_
    dev = torch.device(device)
    x = torch.as_tensor(values, dtype=torch.float32, device=dev)
    projection = torch.nn.Parameter(
        torch.as_tensor(init, dtype=torch.float32, device=dev)
    )
    raw_scale = torch.nn.Parameter(torch.tensor(0.6, device=dev))
    optimizer = torch.optim.Adam(
        [projection, raw_scale], lr=float(cfg["subspace_learning_rate"])
    )
    eligible = np.flatnonzero(train_mask[pair_i] & train_mask[pair_j])
    rng = np.random.default_rng(seed)
    batch_size = min(int(cfg["pair_batch_size"]), len(eligible))
    train_indices = np.flatnonzero(train_mask)
    losses = []
    for step in range(int(cfg["subspace_steps"])):
        chosen = rng.choice(eligible, batch_size, replace=False)
        ii = torch.as_tensor(pair_i[chosen], dtype=torch.long, device=dev)
        jj = torch.as_tensor(pair_j[chosen], dtype=torch.long, device=dev)
        truth = torch.as_tensor(target[chosen], dtype=torch.float32, device=dev)
        zi = torch.nn.functional.normalize(x.index_select(0, ii) @ projection.T, dim=1)
        zj = torch.nn.functional.normalize(x.index_select(0, jj) @ projection.T, dim=1)
        angle = torch.acos((zi * zj).sum(1).clamp(-0.999999, 0.999999))
        scale = torch.nn.functional.softplus(raw_scale)
        distance_loss = torch.nn.functional.smooth_l1_loss(scale * angle, truth)
        sample = rng.choice(train_indices, min(64, len(train_indices)), replace=False)
        sample_x = x.index_select(
            0, torch.as_tensor(sample, dtype=torch.long, device=dev)
        )
        sample_z = sample_x @ projection.T
        reconstruction = sample_z @ torch.linalg.pinv(projection).T
        reconstruction_loss = torch.mean((reconstruction - sample_x) ** 2)
        loss = (
            distance_loss
            + float(cfg["subspace_reconstruction_weight"]) * reconstruction_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step + 1 == int(cfg["subspace_steps"]):
            losses.append(float(loss.detach().cpu()))
    with torch.inference_mode():
        learned = (x @ projection.T).cpu().numpy()
        scale = float(torch.nn.functional.softplus(raw_scale).cpu())
        matrix = projection.cpu().numpy()
    normalized = learned / np.maximum(
        np.linalg.norm(learned, axis=1, keepdims=True), 1e-9
    )
    learned_distance = scale * np.arccos(
        np.clip(np.sum(normalized[pair_i] * normalized[pair_j], axis=1), -1, 1)
    )
    reconstruction = learned @ np.linalg.pinv(matrix).T
    explained = 1 - float(
        np.square(values - reconstruction).sum() / np.square(values).sum()
    )
    return (
        learned.astype(np.float32),
        learned_distance.astype(np.float32),
        matrix.astype(np.float32),
        reconstruction.astype(np.float32),
        {
            "dimensions": dims,
            "distance_scale": scale,
            "variance_explained": explained,
            "sampled_losses": losses,
        },
    )


def _safe_corr(function, left, right):
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(function(left, right).statistic)


def _bootstrap_interval(values, repeats, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, np.float64)
    means = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)]
    )
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _development_representation_cv(
    centered, direction, image_index, development_mask, seed
):
    """Choose a texture-invariant summary without consulting final photographs."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    shaped = centered.reshape(len(centered), 8, 721)
    candidates = {
        "layer_mean": shaped.mean(axis=2),
        "layer_mean_std": np.concatenate(
            (shaped.mean(axis=2), shaped.std(axis=2)), axis=1
        ),
        "layer_quantiles": np.concatenate(
            [
                np.quantile(shaped, value, axis=2)
                for value in (0.1, 0.25, 0.5, 0.75, 0.9)
            ],
            axis=1,
        ),
        "layer_moments": np.concatenate(
            (
                shaped.mean(axis=2),
                shaped.std(axis=2),
                shaped.max(axis=2),
                shaped.min(axis=2),
            ),
            axis=1,
        ),
        "spatial_pca20": centered,
    }
    local = np.flatnonzero(development_mask)
    folds = GroupKFold(6)
    scores = {}
    for name, values in candidates.items():
        fold_scores = []
        for train_local, test_local in folds.split(
            values[local], direction[local], image_index[local]
        ):
            if name == "spatial_pca20":
                model = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=20, random_state=seed),
                    LogisticRegression(C=0.3, max_iter=3000, random_state=seed),
                )
            else:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.3, max_iter=3000, random_state=seed),
                )
            model.fit(
                values[local][train_local], direction[local][train_local].astype(int)
            )
            predicted = model.predict(values[local][test_local])
            fold_scores.append(
                float(
                    balanced_accuracy_score(
                        direction[local][test_local].astype(int), predicted
                    )
                )
            )
        scores[name] = {
            "fold_balanced_accuracy": fold_scores,
            "mean_balanced_accuracy": float(np.mean(fold_scores)),
        }
    winner = max(scores, key=lambda name: scores[name]["mean_balanced_accuracy"])
    return {
        "protocol": "six-fold grouped cross-validation by complete base photograph on development images only",
        "candidates": scores,
        "selected": winner,
    }


def analyze(config=DEFAULT_CONFIG, *, device="cuda"):
    """Fit display embeddings and evaluate held-out-image direction geometry."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import NearestNeighbors

    config_path, contract = _load_contract(config)
    root = Path(contract["output_root"])
    manifest = json.loads((root / "manifest.json").read_text())
    with np.load(root / "axosim-optic-flow-voltage.npz") as recording:
        full_voltage = recording["voltage_mv"].astype(np.float32)
        full_image_index = recording["image_index"].astype(np.int16)
        full_direction = recording["direction_degrees"].astype(np.float32)
        full_image_statistics = recording["image_statistics"].astype(np.float32)
        statistic_names = recording["statistic_names"].astype(str)
    cfg = contract["analysis"]
    analysis_images = np.asarray(
        [
            row["base_image_index"]
            for row in manifest["rows"]
            if row["evaluation_split"] != "display-only"
        ],
        np.int16,
    )
    analysis_trial_mask = np.isin(full_image_index, analysis_images)
    voltage = full_voltage[analysis_trial_mask]
    image_index = full_image_index[analysis_trial_mask]
    direction = full_direction[analysis_trial_mask]
    image_statistics = full_image_statistics[analysis_trial_mask]
    test_images = np.asarray(
        [
            row["base_image_index"]
            for row in manifest["rows"]
            if row["evaluation_split"] == "final"
        ],
        np.int16,
    )
    test_mask = np.isin(image_index, test_images)
    train_mask = ~test_mask
    centered = within_image_center(voltage, image_index)
    representation_selection = _development_representation_cv(
        centered, direction, image_index, train_mask, int(contract["seed"])
    )
    if representation_selection["selected"] != "layer_mean_std":
        raise RuntimeError(
            "frozen layer_mean_std representation no longer wins development-only CV"
        )
    raw_x, raw_keep, _, _ = _standardize_train(centered, train_mask)
    activation_x, activation_keep, _, _ = _standardize_train(
        full_voltage, np.ones(len(full_voltage), dtype=bool)
    )
    shaped = centered.reshape(len(centered), len(contract["record_layers"]), 721)
    pooled = np.concatenate((shaped.mean(axis=2), shaped.std(axis=2)), axis=1)
    x, keep, feature_mean, feature_std = _standardize_train(pooled, train_mask)
    seed = int(contract["seed"])
    raw_two, raw_three, _ = _umap_pair(raw_x, cfg, seed)
    raw_display_cfg = cfg["raw_activation_umap"]
    activation_cfg = {
        **cfg,
        "umap_neighbors": raw_display_cfg["n_neighbors"],
        "umap_min_dist": raw_display_cfg["min_dist"],
        "umap_init": raw_display_cfg.get("init", "spectral"),
    }
    activation_two, activation_three, _ = _umap_pair(
        activation_x, activation_cfg, seed + 2
    )
    activation_neighbors = NearestNeighbors(n_neighbors=11).fit(activation_two)
    activation_neighbor_index = activation_neighbors.kneighbors(return_distance=False)[
        :, 1:
    ]
    activation_same_image_rate = float(
        np.mean(
            full_image_index[activation_neighbor_index] == full_image_index[:, None]
        )
    )
    pair_i, pair_j = np.triu_indices(len(x), 1)
    target = np.deg2rad(
        circular_distance_degrees(direction[pair_i], direction[pair_j])
    ).astype(np.float32)
    cosine, geodesic, knn_k = _knn_geodesic(raw_x)
    learned, learned_distance, projection, reconstruction, learned_meta = (
        _learn_subspace(x, target, pair_i, pair_j, train_mask, cfg, seed, device)
    )
    learned_two, learned_three, _ = _umap_pair(learned, cfg, seed + 1)
    residual = x - reconstruction
    residual_model = PCA(n_components=1, random_state=seed).fit(residual)
    residual_component = residual_model.transform(residual)[:, 0].astype(np.float32)
    full_centered = within_image_center(full_voltage, full_image_index)
    full_shaped = full_centered.reshape(
        len(full_centered), len(contract["record_layers"]), 721
    )
    full_pooled = np.concatenate(
        (full_shaped.mean(axis=2), full_shaped.std(axis=2)), axis=1
    )
    full_x = ((full_pooled[:, keep] - feature_mean[keep]) / feature_std[keep]).astype(
        np.float32
    )
    full_learned = full_x @ projection.T
    learned_display_cfg = cfg["learned_subspace_umap"]
    display_learned_two, display_learned_three, _ = _umap_pair(
        full_learned,
        {
            **cfg,
            "umap_neighbors": learned_display_cfg["n_neighbors"],
            "umap_min_dist": learned_display_cfg["min_dist"],
            "umap_init": learned_display_cfg.get("init", "spectral"),
        },
        seed + 3,
    )
    display_learned_neighbors = NearestNeighbors(n_neighbors=11).fit(
        display_learned_two
    )
    display_learned_neighbor_index = display_learned_neighbors.kneighbors(
        return_distance=False
    )[:, 1:]
    display_learned_same_direction_rate = float(
        np.mean(
            full_direction[display_learned_neighbor_index] == full_direction[:, None]
        )
    )
    display_learned_neighbor_error = float(
        circular_distance_degrees(
            full_direction[display_learned_neighbor_index],
            full_direction[:, None],
        ).mean()
    )
    full_reconstruction = full_learned @ np.linalg.pinv(projection).T
    activation_residual_component = residual_model.transform(
        full_x - full_reconstruction
    )[:, 0].astype(np.float32)
    linear_three_model = PCA(n_components=3, random_state=seed).fit(learned[train_mask])
    linear_three = linear_three_model.transform(learned).astype(np.float32)

    decoder = LogisticRegression(C=0.3, max_iter=3000, random_state=seed)
    decoder.fit(x[train_mask], direction[train_mask].astype(int))
    prediction = decoder.predict(x[test_mask]).astype(np.float32)
    truth = direction[test_mask]
    accuracy = float(balanced_accuracy_score(truth, prediction))
    circular_error = circular_distance_degrees(truth, prediction)
    per_image_accuracy, per_image_error = [], []
    for image in test_images:
        local = image_index[test_mask] == image
        per_image_accuracy.append(float(np.mean(prediction[local] == truth[local])))
        per_image_error.append(float(np.mean(circular_error[local])))
    rng = np.random.default_rng(seed + 90)
    null_accuracy = []
    repeats = int(cfg["bootstrap_replicates"])
    for _ in range(repeats):
        shuffled = truth.copy()
        for image in test_images:
            local = image_index[test_mask] == image
            shuffled[local] = rng.permutation(shuffled[local])
        null_accuracy.append(float(np.mean(prediction == shuffled)))
    permutation_p = float(
        (1 + np.sum(np.asarray(null_accuracy) >= accuracy)) / (repeats + 1)
    )

    pair_geodesic = geodesic[pair_i, pair_j].astype(np.float32)
    pair_cosine = cosine[pair_i, pair_j].astype(np.float32)
    categories = {
        "train-train": train_mask[pair_i] & train_mask[pair_j],
        "test-test": test_mask[pair_i] & test_mask[pair_j],
        "train-test": train_mask[pair_i] ^ train_mask[pair_j],
    }
    unique_angles = np.asarray(contract["stimulus"]["directions_degrees"])
    cube = np.empty(
        (len(analysis_images), len(unique_angles), voltage.shape[1]), np.float32
    )
    for local_image, image in enumerate(analysis_images):
        for angle_number, angle in enumerate(unique_angles):
            index = np.flatnonzero((image_index == image) & (direction == angle))
            if len(index) != 1:
                raise RuntimeError("recording does not form an image × direction grid")
            cube[local_image, angle_number] = centered[index[0]]
    spectrum = np.abs(np.fft.rfft(cube, axis=1)) ** 2
    harmonic_power = spectrum[:, 1:].mean(axis=(0, 2))
    harmonic_power /= max(float(harmonic_power.sum()), 1e-12)

    metrics = {
        "status": "complete",
        "scientific_object": "within-photograph direction modulation of AxoSim T4/T5 soma voltage, summarized by the spatial mean and standard deviation within each anatomically named cell class",
        "no_phylogeny_claim": True,
        "samples": len(x),
        "base_images": len(analysis_images),
        "visualization_base_images": len(manifest["rows"]),
        "visualization_samples": len(full_voltage),
        "train_images": int(
            sum(row["evaluation_split"] == "development" for row in manifest["rows"])
        ),
        "held_out_images": int(len(test_images)),
        "test_image_indices": test_images.tolist(),
        "directions": len(unique_angles),
        "development_representation_selection": representation_selection,
        "chance_balanced_accuracy": 1 / len(unique_angles),
        "held_out_direction_balanced_accuracy": accuracy,
        "held_out_direction_accuracy_95pct_image_bootstrap": _bootstrap_interval(
            per_image_accuracy, repeats, seed + 2
        ),
        "held_out_circular_mean_absolute_error_degrees": float(circular_error.mean()),
        "held_out_circular_mae_95pct_image_bootstrap": _bootstrap_interval(
            per_image_error, repeats, seed + 3
        ),
        "held_out_label_permutation_p": permutation_p,
        "raw_spearman_cosine_similarity_vs_angular_distance": _safe_corr(
            spearmanr, pair_cosine, target
        ),
        "raw_pearson_knn_geodesic_vs_angular_distance": _safe_corr(
            pearsonr, pair_geodesic, target
        ),
        "knn_k_smallest_connected": knn_k,
        "learned_subspace": learned_meta,
        "learned_pair_correlations": {
            name: _safe_corr(pearsonr, learned_distance[mask], target[mask])
            for name, mask in categories.items()
        },
        "nonconstant_geometry_features": int(keep.sum()),
        "raw_nonconstant_neurons": int(raw_keep.sum()),
        "activation_nonconstant_neurons": int(activation_keep.sum()),
        "activation_umap_same_photograph_10nn_rate": activation_same_image_rate,
        "activation_umap_role": "uncentered AxoSim soma voltage for the Goodfire raw-activation figures; excluded from direction decoding and learned-subspace fitting",
        "display_umap_selection": {
            "raw_activation": raw_display_cfg,
            "learned_subspace": learned_display_cfg,
        },
        "display_learned_umap_same_direction_10nn_rate": display_learned_same_direction_rate,
        "display_learned_umap_neighbor_direction_error_degrees": display_learned_neighbor_error,
        "recorded_neurons": int(voltage.shape[1]),
        "harmonic_power_fraction_k1_to_k6": harmonic_power.tolist(),
        "claim_gate": {
            "accuracy_above_twice_chance": accuracy > 2 / len(unique_angles),
            "permutation_p_below_0_05": permutation_p < 0.05,
            "held_out_learned_distance_correlation_positive": _safe_corr(
                pearsonr,
                learned_distance[categories["test-test"]],
                target[categories["test-test"]],
            )
            > 0,
        },
    }
    metrics["claim_gate"]["passed"] = all(metrics["claim_gate"].values())
    np.savez_compressed(
        root / "analysis.npz",
        centered_voltage_mv=centered,
        raw_standardized=raw_x,
        standardized=x,
        keep_features=keep,
        feature_mean=feature_mean,
        feature_std=feature_std,
        raw_umap_2d=raw_two,
        raw_umap_3d=raw_three,
        activation_umap_2d=activation_two,
        activation_umap_3d=activation_three,
        activation_residual_component=activation_residual_component,
        display_learned_umap_2d=display_learned_two,
        display_learned_umap_3d=display_learned_three,
        learned=learned,
        learned_umap_2d=learned_two,
        learned_umap_3d=learned_three,
        learned_linear_3d=linear_three,
        learned_linear_mean=linear_three_model.mean_,
        learned_linear_components=linear_three_model.components_,
        projection=projection,
        reconstruction=reconstruction,
        residual_component=residual_component,
        residual_explained=residual_model.explained_variance_ratio_[0],
        pair_i=pair_i,
        pair_j=pair_j,
        target_angular_distance=target,
        cosine_similarity=pair_cosine,
        geodesic_distance=pair_geodesic,
        learned_distance=learned_distance,
        train_mask=train_mask,
        test_mask=test_mask,
        test_images=test_images,
        prediction=prediction,
        truth=truth,
        analysis_image_index=image_index,
        analysis_direction_degrees=direction,
        image_statistics=image_statistics,
        statistic_names=statistic_names,
        harmonic_power=harmonic_power,
    )
    summary_path = root / "analysis-summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def render_plots(config=DEFAULT_CONFIG):
    """Render exact-composition Goodfire replicas with TikZ/PGFPlots."""
    from .goodfire_tikz import render_all

    return render_all(config)


def run_all(config=DEFAULT_CONFIG, *, device="cuda"):
    preparation = prepare_data(config)
    recording = record_voltages(config, device=device)
    result = analyze(config, device=device)
    plots = render_plots(config)
    return {
        "preparation": preparation,
        "recording": recording,
        "analysis": result,
        "plots": plots,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "record", "analyze", "render-plots", "all"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_data(args.config)
    elif args.command == "record":
        result = record_voltages(args.config, device=args.device)
    elif args.command == "analyze":
        result = analyze(args.config, device=args.device)
    elif args.command == "render-plots":
        result = render_plots(args.config)
    else:
        result = run_all(args.config, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
