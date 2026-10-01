"""Test phase-insensitive return to an AxoSim recurrent orbit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .goodfire_tikz import _axis, _compile, _coordinates, _number
from .natural_manifold import _retinal_coordinates, _sample_scene
from .neural import DEFAULT_CHECKPOINT
from .vision_axosim import AxoSimFlyVis


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/limit_cycle_return.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_retinal_frame(frame, normalisation):
    frame = np.asarray(frame, dtype=np.float32)
    if normalisation is None:
        return frame
    mean = frame.mean()
    standard_deviation = frame.std()
    result = (frame - mean) / max(float(standard_deviation), 1e-6)
    result = result * float(normalisation["standard_deviation"])
    result = result + float(normalisation["mean"])
    return np.clip(
        result,
        float(normalisation["clip"][0]),
        float(normalisation["clip"][1]),
    ).astype(np.float32)


def _scene_frame(scene, manifold_contract, retinal_x, retinal_y):
    source_dt = float(manifold_contract["source_dt_seconds"])
    sample_time = 2.0
    stimulus, _, _ = _sample_scene(
        scene,
        duration=sample_time + 2 * source_dt,
        dt=source_dt,
        retinal_x=retinal_x,
        retinal_y=retinal_y,
    )
    frame = stimulus[round(sample_time / source_dt)]
    return _normalise_retinal_frame(
        frame, manifold_contract.get("retinal_normalization")
    )


def _endogenous_activity_index(runtime):
    clamped = torch.zeros(runtime.n_nodes, dtype=torch.bool, device=runtime.device)
    clamped[runtime.input_index.unique()] = True
    return torch.nonzero(~clamped, as_tuple=False).squeeze(1)


def _state_statistics(
    hidden_states,
    activity_states,
    endogenous_index,
    *,
    minimum_scale,
):
    hidden = torch.stack(hidden_states).index_select(1, endogenous_index)
    activity = torch.stack(activity_states).index_select(1, endogenous_index)
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    activity_flat = activity.reshape(-1, activity.shape[-1])
    hidden_mean = hidden_flat.mean(0)
    activity_mean = activity_flat.mean(0)
    hidden_scale = hidden_flat.std(0).clamp_min(minimum_scale)
    activity_scale = activity_flat.std(0).clamp_min(minimum_scale)
    reference = torch.cat(
        (
            ((hidden - hidden_mean) / hidden_scale).flatten(1),
            ((activity - activity_mean) / activity_scale).flatten(1),
        ),
        dim=1,
    ).contiguous()
    statistics = {
        "hidden_mean": hidden_mean,
        "activity_mean": activity_mean,
        "hidden_scale": hidden_scale,
        "activity_scale": activity_scale,
    }
    return reference, statistics


def _standardised_state(hidden, activity, endogenous_index, statistics):
    hidden_values = (
        hidden.index_select(0, endogenous_index) - statistics["hidden_mean"]
    ) / statistics["hidden_scale"]
    activity_values = (
        activity.index_select(0, endogenous_index) - statistics["activity_mean"]
    ) / statistics["activity_scale"]
    return torch.cat((hidden_values.flatten(), activity_values.flatten()))


def apply_full_state_perturbation(
    hidden,
    activity,
    endogenous_index,
    statistics,
    *,
    amplitude,
    seed,
):
    """Displace every free recurrent-state component with a fixed random draw."""
    generator = torch.Generator(device=hidden.device)
    generator.manual_seed(int(seed))
    perturbed_hidden = hidden.clone()
    perturbed_activity = activity.clone()
    hidden_noise = torch.randn(
        (len(endogenous_index), hidden.shape[1]),
        generator=generator,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    perturbed_hidden[endogenous_index] += (
        hidden_noise * statistics["hidden_scale"] * float(amplitude)
    )
    activity_noise = torch.randn(
        (len(endogenous_index), activity.shape[1]),
        generator=generator,
        device=activity.device,
        dtype=activity.dtype,
    )
    perturbed_activity[endogenous_index] += (
        activity_noise * statistics["activity_scale"] * float(amplitude)
    )
    return perturbed_hidden, perturbed_activity


def phase_insensitive_orbit_distance(samples, reference, *, chunk_size=16):
    """Minimum RMS distance to any sampled orbit phase in the full state."""
    if samples.ndim != 2 or reference.ndim != 2 or samples.shape[1] != reference.shape[1]:
        raise ValueError("Samples and reference must be two-dimensional with equal width")
    dimensions = samples.shape[1]
    reference_norm = reference.square().sum(1) / dimensions
    distances = []
    for start in range(0, len(samples), chunk_size):
        values = samples[start : start + chunk_size]
        values_norm = values.square().sum(1)[:, None] / dimensions
        squared = values_norm + reference_norm[None]
        squared = squared - 2 * (values @ reference.T) / dimensions
        distances.append(squared.clamp_min(0).min(1).values.sqrt())
    return torch.cat(distances)


def _advance_branch(
    runtime,
    stimulus_block,
    hidden,
    activity,
    endogenous_index,
    statistics,
    *,
    post_seconds,
    sample_seconds,
):
    block_seconds = 0.004
    blocks = round(post_seconds / block_seconds)
    sample_blocks = round(sample_seconds / block_seconds)
    if not math.isclose(sample_blocks * block_seconds, sample_seconds):
        raise ValueError("distance_sample_seconds must be divisible by 4 ms")
    current = torch.zeros_like(activity)
    samples = [_standardised_state(hidden, activity, endogenous_index, statistics)]
    for block_index in range(1, blocks + 1):
        hidden, activity = runtime.advance_block(
            stimulus_block,
            hidden,
            activity,
            current=current,
        )
        if block_index % sample_blocks == 0:
            samples.append(
                _standardised_state(hidden, activity, endogenous_index, statistics)
            )
    return torch.stack(samples)


def _first_sustained_return(
    distance,
    sample_time,
    *,
    threshold,
    consecutive_seconds,
    maximum_seconds,
):
    required = round(consecutive_seconds / sample_time)
    within = np.asarray(distance) <= threshold
    for start in range(0, len(within) - required + 1):
        confirmed_at = (start + required - 1) * sample_time
        if confirmed_at > maximum_seconds + 1e-12:
            break
        if within[start : start + required].all():
            return float(confirmed_at)
    return None


def _cluster_bootstrap_median(rows, *, repetitions, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row["scene"], []).append(row["late_excess_ratio"])
    scenes = sorted(groups)
    medians = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        values = []
        for scene_index in rng.integers(0, len(scenes), len(scenes)):
            group = np.asarray(groups[scenes[scene_index]], dtype=np.float64)
            values.extend(group[rng.integers(0, len(group), len(group))])
        medians[repetition] = np.median(values)
    return np.quantile(medians, [0.025, 0.975]).tolist()


def summarise_return(distances, control_distances, times, scenes, amplitudes, seeds, contract):
    """Apply the preregistered positive-result gate to measured distances."""
    cfg = contract["return_test"]
    late = (times >= cfg["late_window_seconds"][0]) & (
        times <= cfg["late_window_seconds"][1]
    )
    rows = []
    for amplitude_index, amplitude in enumerate(amplitudes):
        for scene_index, scene in enumerate(scenes):
            control = control_distances[scene_index]
            control_median = float(np.median(control))
            control_floor = float(
                np.quantile(control, cfg["control_distance_quantile"])
            )
            for seed_index, seed in enumerate(seeds):
                distance = distances[amplitude_index, scene_index, seed_index]
                excess = np.maximum(distance - control, 0.0)
                initial_excess = max(float(excess[0]), 1e-12)
                late_excess = float(np.median(excess[late]))
                ratio = late_excess / initial_excess
                return_threshold = (
                    control_floor
                    + cfg["late_tolerance_as_initial_fraction"] * initial_excess
                )
                return_time = _first_sustained_return(
                    distance,
                    float(times[1] - times[0]),
                    threshold=return_threshold,
                    consecutive_seconds=cfg["required_consecutive_seconds"],
                    maximum_seconds=cfg["maximum_return_seconds"],
                )
                rows.append(
                    {
                        "scene": scene,
                        "amplitude_in_orbit_standard_deviations": float(amplitude),
                        "seed": int(seed),
                        "initial_orbit_distance": float(distance[0]),
                        "initial_excess_over_matched_control": initial_excess,
                        "control_median_orbit_distance": control_median,
                        "control_quantile_orbit_distance": control_floor,
                        "late_median_orbit_distance": float(np.median(distance[late])),
                        "late_excess_ratio": float(ratio),
                        "return_threshold": float(return_threshold),
                        "confirmed_return_seconds": return_time,
                        "returned": return_time is not None,
                    }
                )
    by_amplitude = []
    for amplitude_index, amplitude in enumerate(amplitudes):
        selected = [
            row
            for row in rows
            if row["amplitude_in_orbit_standard_deviations"] == float(amplitude)
        ]
        ratios = np.asarray([row["late_excess_ratio"] for row in selected])
        interval = _cluster_bootstrap_median(
            selected,
            repetitions=int(cfg["bootstrap_repetitions"]),
            seed=int(cfg["bootstrap_seed"]) + amplitude_index,
        )
        return_fraction = float(np.mean([row["returned"] for row in selected]))
        median = float(np.median(ratios))
        passed = (
            median <= cfg["maximum_median_excess_ratio"]
            and interval[1] <= cfg["maximum_bootstrap_upper_excess_ratio"]
            and return_fraction >= cfg["minimum_trial_return_fraction"]
        )
        by_amplitude.append(
            {
                "amplitude_in_orbit_standard_deviations": float(amplitude),
                "trials": len(selected),
                "median_late_excess_ratio": median,
                "scene_cluster_bootstrap_95_interval": interval,
                "return_fraction": return_fraction,
                "passed": bool(passed),
            }
        )
    return {
        "passed": all(item["passed"] for item in by_amplitude),
        "by_amplitude": by_amplitude,
        "trials": rows,
    }


def run_experiment(
    *,
    config=DEFAULT_CONFIG,
    root,
    output,
    checkpoint=DEFAULT_CHECKPOINT,
    device="cuda",
):
    config = Path(config)
    contract = json.loads(config.read_text())
    manifold_contract = json.loads(Path(contract["natural_manifold_config"]).read_text())
    vision_contract = json.loads(Path(contract["vision_config"]).read_text())
    scored = json.loads(Path(contract["scored_summary"]).read_text())
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
    _, _, retinal_x, retinal_y = _retinal_coordinates(network)
    scene_by_name = {scene["name"]: scene for scene in manifold_contract["scenes"]}
    scenes = list(contract["scenes"])
    if len(set(scenes)) != len(scenes) or any(name not in scene_by_name for name in scenes):
        raise ValueError("limit-cycle scene names must be unique members of the film")
    amplitudes = np.asarray(
        contract["perturbation"]["amplitudes_in_orbit_standard_deviations"],
        dtype=np.float64,
    )
    seeds = np.asarray(contract["perturbation"]["seeds"], dtype=np.int64)
    block_seconds = 0.004
    settling_blocks = round(contract["settling_seconds"] / block_seconds)
    reference_blocks = round(contract["reference_orbit_seconds"] / block_seconds)
    sample_seconds = float(contract["distance_sample_seconds"])
    post_seconds = float(contract["post_perturbation_seconds"])
    distance_times = np.arange(
        0,
        post_seconds + sample_seconds / 2,
        sample_seconds,
        dtype=np.float32,
    )
    distances = np.empty(
        (len(amplitudes), len(scenes), len(seeds), len(distance_times)),
        dtype=np.float32,
    )
    control_distances = np.empty((len(scenes), len(distance_times)), dtype=np.float32)
    endogenous_index = _endogenous_activity_index(runtime)
    recurrent_dimensions = len(endogenous_index) * (
        runtime.model.config.state_dim + 4
    )
    scene_metadata = []
    started = time.monotonic()
    for scene_index, scene_name in enumerate(scenes):
        print(f"scene: {scene_name}", flush=True)
        frame = _scene_frame(
            scene_by_name[scene_name], manifold_contract, retinal_x, retinal_y
        )
        stimulus_block = np.repeat(frame[None], 4, axis=0)
        hidden, activity = runtime.initial_recurrent_state()
        current = torch.zeros_like(activity)
        reference_hidden = []
        reference_activity = []
        for block_index in range(1, settling_blocks + 1):
            hidden, activity = runtime.advance_block(
                stimulus_block,
                hidden,
                activity,
                current=current,
            )
            if block_index > settling_blocks - reference_blocks:
                reference_hidden.append(hidden.clone())
                reference_activity.append(activity.clone())
        base_hidden = hidden.clone()
        base_activity = activity.clone()
        reference, statistics = _state_statistics(
            reference_hidden,
            reference_activity,
            endogenous_index,
            minimum_scale=float(
                contract["perturbation"]["minimum_component_scale"]
            ),
        )
        del reference_hidden, reference_activity
        control_states = _advance_branch(
            runtime,
            stimulus_block,
            base_hidden.clone(),
            base_activity.clone(),
            endogenous_index,
            statistics,
            post_seconds=post_seconds,
            sample_seconds=sample_seconds,
        )
        control_distances[scene_index] = (
            phase_insensitive_orbit_distance(control_states, reference).cpu().numpy()
        )
        del control_states
        measured_initial_displacements = []
        for amplitude_index, amplitude in enumerate(amplitudes):
            for seed_index, seed in enumerate(seeds):
                perturbed_hidden, perturbed_activity = apply_full_state_perturbation(
                    base_hidden,
                    base_activity,
                    endogenous_index,
                    statistics,
                    amplitude=float(amplitude),
                    seed=int(seed),
                )
                states = _advance_branch(
                    runtime,
                    stimulus_block,
                    perturbed_hidden,
                    perturbed_activity,
                    endogenous_index,
                    statistics,
                    post_seconds=post_seconds,
                    sample_seconds=sample_seconds,
                )
                measured = phase_insensitive_orbit_distance(states, reference)
                distances[amplitude_index, scene_index, seed_index] = (
                    measured.cpu().numpy()
                )
                measured_initial_displacements.append(float(measured[0].item()))
                del states, measured
        scene_metadata.append(
            {
                "name": scene_name,
                "retinal_frame_sha256": hashlib.sha256(
                    frame.astype("<f4", copy=False).tobytes()
                ).hexdigest(),
                "hidden_component_scale": statistics["hidden_scale"].cpu().tolist(),
                "activity_component_scale": statistics["activity_scale"].cpu().tolist(),
                "control_orbit_distance_median": float(
                    np.median(control_distances[scene_index])
                ),
                "measured_initial_orbit_distance_range": [
                    float(min(measured_initial_displacements)),
                    float(max(measured_initial_displacements)),
                ],
            }
        )
        del reference, base_hidden, base_activity, statistics
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    result = summarise_return(
        distances,
        control_distances,
        distance_times,
        scenes,
        amplitudes,
        seeds,
        contract,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    distances_path = output / "limit-cycle-distances.npz"
    np.savez_compressed(
        distances_path,
        time_seconds=distance_times,
        perturbation_orbit_distance=distances,
        control_orbit_distance=control_distances,
        scenes=np.asarray(scenes),
        amplitudes_in_orbit_standard_deviations=amplitudes,
        seeds=seeds,
    )
    summary = {
        "status": "complete",
        "passed": result["passed"],
        "hypothesis": "small displacements of the complete recurrent state return to the same phase-insensitive periodic orbit under constant visual input",
        "interpretation": (
            "supported local numerical limit-cycle attractor in the substituted model"
            if result["passed"]
            else "the preregistered local limit-cycle gate was not met"
        ),
        "claim_boundary": "local numerical attraction under six constant visual inputs in the frozen substituted AxoSim/FlyVis circuit; not calibrated fly physiology or a biologically delivered perturbation",
        "config_sha256": _sha256(config),
        "checkpoint_sha256": _sha256(checkpoint),
        "graph": scored["graph"],
        "selected": selected,
        "state_space": {
            "cells": runtime.n_nodes,
            "clamped_input_cells": int(runtime.n_nodes - len(endogenous_index)),
            "unclamped_recurrent_cells": int(len(endogenous_index)),
            "hidden_components_per_recurrent_cell": int(
                runtime.model.config.state_dim
            ),
            "activity_components_per_recurrent_cell": 4,
            "full_recurrent_dimensions": int(recurrent_dimensions),
            "distance": "minimum standardized RMS distance to any 4-ms reference-orbit phase",
        },
        "conditions": {
            "scenes": scene_metadata,
            "settling_seconds": contract["settling_seconds"],
            "reference_orbit_seconds": contract["reference_orbit_seconds"],
            "post_perturbation_seconds": post_seconds,
            "distance_sample_seconds": sample_seconds,
            "amplitudes_in_orbit_standard_deviations": amplitudes.tolist(),
            "seeds": seeds.tolist(),
        },
        "return_test": result,
        "runtime_wall_seconds": time.monotonic() - started,
        "distances_file": distances_path.name,
        "distances_sha256": _sha256(distances_path),
    }
    summary_path = output / "limit-cycle-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["return_test"]["by_amplitude"], indent=2), flush=True)
    print(f"passed: {summary['passed']}", flush=True)
    return summary


def analyse_recording(*, config, distances, summary):
    """Recompute the frozen return gate from persisted orbit distances."""
    contract = json.loads(Path(config).read_text())
    summary_path = Path(summary)
    report = json.loads(summary_path.read_text())
    with np.load(distances) as data:
        times = data["time_seconds"]
        measured = data["perturbation_orbit_distance"]
        control = data["control_orbit_distance"]
        scenes = data["scenes"].astype(str)
        amplitudes = data["amplitudes_in_orbit_standard_deviations"]
        seeds = data["seeds"]
    result = summarise_return(
        measured, control, times, scenes, amplitudes, seeds, contract
    )
    report["passed"] = result["passed"]
    report["interpretation"] = (
        "supported local numerical limit-cycle attractor in the substituted model"
        if result["passed"]
        else "the preregistered local limit-cycle gate was not met"
    )
    report["analysis_revision"] = (
        "excess orbit distance subtracts the matched unperturbed control at each "
        "elapsed time before normalization"
    )
    report["return_test"] = result
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result["by_amplitude"], indent=2), flush=True)
    print(f"passed: {report['passed']}", flush=True)
    return report


def _curve_coordinates(x, y):
    return _coordinates(np.asarray(x), np.asarray(y))


def render_figure(*, summary, distances, output):
    summary_path = Path(summary)
    distances_path = Path(distances)
    summary = json.loads(summary_path.read_text())
    with np.load(distances_path) as data:
        times = data["time_seconds"]
        measured = data["perturbation_orbit_distance"]
        control = data["control_orbit_distance"]
        scenes = data["scenes"].astype(str)
        amplitudes = data["amplitudes_in_orbit_standard_deviations"]
    ratios = np.empty_like(measured)
    for scene_index in range(len(scenes)):
        excess = np.maximum(
            measured[:, scene_index] - control[scene_index][None, None], 0
        )
        initial = np.maximum(excess[..., :1], 1e-12)
        ratios[:, scene_index] = excess / initial
    flattened = ratios.reshape(len(amplitudes), -1, len(times))
    medians = np.median(flattened, axis=1)
    lower = np.quantile(flattened, 0.1, axis=1)
    upper = np.quantile(flattened, 0.9, axis=1)
    ymax = max(1.05, min(2.0, float(np.max(upper[:, 1:])) * 1.05))
    colors = ("Axym", "AxymLight")
    styles = ("solid", "densely dashed")
    plots = []
    for index, amplitude in enumerate(amplitudes):
        plots.extend(
            [
                rf"\addplot[forget plot,color={colors[index]},opacity=.28,line width=.6pt,{styles[index]}] coordinates {{{_curve_coordinates(times, lower[index])}}};",
                rf"\addplot[forget plot,color={colors[index]},opacity=.28,line width=.6pt,{styles[index]}] coordinates {{{_curve_coordinates(times, upper[index])}}};",
                rf"\addplot[color={colors[index]},line width=2.2pt,{styles[index]}] coordinates {{{_curve_coordinates(times, medians[index])}}};",
            ]
        )
    plots.append(
        r"\addplot[forget plot,color=Muted,densely dotted,line width=1.1pt] coordinates {(0,.2) (3,.2)};"
    )
    left = _axis(
        110,
        105,
        880,
        520,
        (
            rf"xmin=0,xmax=3,ymin=0,ymax={_number(ymax)},"
            r"xlabel={Time after perturbation (s)},ylabel={Excess orbit distance / initial},"
            r"xtick={0,0.5,1,1.5,2,2.5,3},legend style={draw=none,fill=none,font=\sffamily\fontsize{9}{11}\selectfont,at={(.98,.97)},anchor=north east},"
            r"legend entries={$0.10\sigma$,$0.25\sigma$},"
        ),
        "\n".join(plots),
    )
    scene_labels = {
        "apple-on-table": "Apple",
        "voyager-earth": "Earth",
        "woodland-clearing": "Woods",
        "ocean-surface": "Ocean",
        "city-pedestrians-a": "City",
        "voyager-jupiter": "Jupiter",
    }
    right_plots = [
        r"\addplot[forget plot,color=Muted,densely dotted,line width=1.1pt] coordinates {(-.5,.2) (5.5,.2)};"
    ]
    late = (times >= 2.0) & (times <= 3.0)
    scene_medians = []
    for amplitude_index in range(len(amplitudes)):
        x = np.arange(len(scenes), dtype=np.float64) + (-0.09 if amplitude_index == 0 else 0.09)
        y = np.median(ratios[amplitude_index, :, :, :][:, :, late], axis=(1, 2))
        scene_medians.append(y)
        right_plots.append(
            rf"\addplot[only marks,mark={'*' if amplitude_index == 0 else 'square*'},mark size=3.2pt,color={colors[amplitude_index]}] coordinates {{{_curve_coordinates(x, y)}}};"
        )
    right = _axis(
        1090,
        105,
        410,
        520,
        (
            rf"xmin=-.5,xmax=5.5,ymin=0,ymax={_number(max(.25, float(np.max(scene_medians)) * 1.15))},"
            r"ylabel={Late excess ratio},xlabel={Constant visual input},"
            r"ytick={0,0.05,0.10,0.15,0.20,0.25},"
            r"xtick={0,1,2,3,4,5},xticklabels={"
            + ",".join(scene_labels[name] for name in scenes)
            + r"},x tick label style={rotate=28,anchor=east,font=\sffamily\fontsize{8}{10}\selectfont},"
        ),
        "\n".join(right_plots),
    )
    body = "\n".join(
        [
            r"\definecolor{Axym}{HTML}{3F21B6}",
            r"\definecolor{AxymLight}{HTML}{8C7AD3}",
            r"\fill[white] (0,0) rectangle (1600,740);",
            left,
            right,
            r"\node[anchor=west,font=\sffamily\bfseries\fontsize{18}{20}\selectfont,text=Ink] at (75,680) {a};",
            r"\node[anchor=west,font=\sffamily\bfseries\fontsize{18}{20}\selectfont,text=Ink] at (1055,680) {b};",
        ]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = _compile(output, "limit-cycle-return.png", 1600, 740, body)
    metadata = {
        "passed": summary["passed"],
        "source_summary": str(summary_path),
        "source_distances": str(distances_path),
        "png_sha256": _sha256(target),
        "pdf_sha256": _sha256(output / "limit-cycle-return.pdf"),
        "encoding": "panel a shows median and 10th/90th percentile full-state excess orbit distance; panel b shows scene medians over the 2-3 s late window; dotted line is the preregistered 0.20 median gate",
    }
    (output / "limit-cycle-return-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--root", type=Path, default=Path("data/flyvis"))
    run.add_argument(
        "--output", type=Path, default=Path("data/natural_manifold/limit-cycle")
    )
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--device", default="cuda")
    render = subparsers.add_parser("render")
    render.add_argument(
        "--summary",
        type=Path,
        default=Path("data/natural_manifold/limit-cycle/limit-cycle-summary.json"),
    )
    render.add_argument(
        "--distances",
        type=Path,
        default=Path("data/natural_manifold/limit-cycle/limit-cycle-distances.npz"),
    )
    render.add_argument(
        "--output", type=Path, default=Path("data/natural_manifold/limit-cycle/figures")
    )
    analyse = subparsers.add_parser("analyze")
    analyse.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    analyse.add_argument(
        "--summary",
        type=Path,
        default=Path("data/natural_manifold/limit-cycle/limit-cycle-summary.json"),
    )
    analyse.add_argument(
        "--distances",
        type=Path,
        default=Path("data/natural_manifold/limit-cycle/limit-cycle-distances.npz"),
    )
    args = parser.parse_args()
    if args.action == "run":
        run_experiment(
            config=args.config,
            root=args.root,
            output=args.output,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    elif args.action == "render":
        render_figure(summary=args.summary, distances=args.distances, output=args.output)
    else:
        analyse_recording(
            config=args.config, distances=args.distances, summary=args.summary
        )


if __name__ == "__main__":
    main()
