"""Run the frozen AxoSim substitution experiment on the FlyVis visual circuit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from axosim.checkpoint import load_checkpoint

from .neural import CHECKPOINT_SHA256, DEFAULT_CHECKPOINT
from .vision_reference import PRETRAINED_ZIP_SHA256, _sha256


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/flyvis_moving_edge.json"
SELECTIVITY_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def _tensor_sha256(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        array = np.asarray(tensor)
        digest.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denominator <= 1e-15 else float(np.dot(x, y) / denominator)


def _resample(values, source_dt, target_dt):
    """Linearly resample a time-by-feature array at a fixed cadence."""
    from scipy.interpolate import interp1d

    values = np.asarray(values)
    source_time = np.arange(len(values), dtype=np.float64) * source_dt
    target_time = np.arange(0, source_time[-1] + 1e-12, target_dt)
    result = interp1d(source_time, values, axis=0)(target_time)
    return result.astype(np.float32), target_time


def _spatial_target_control(target, target_type, *, seed):
    """Destroy retinotopy while preserving target type and target-degree multiset."""
    target = np.asarray(target, dtype=np.int64)
    target_type = np.asarray(target_type)
    shuffled = target.copy()
    rng = np.random.default_rng(seed)
    for value in np.unique(target_type):
        locations = np.flatnonzero(target_type == value)
        shuffled[locations] = rng.permutation(target[locations])
    return shuffled


def _dataset(config, angles):
    from flyvis.datasets.moving_bar import MovingEdge

    protocol = config["protocol"]
    return MovingEdge(
        offsets=tuple(protocol["offsets"]),
        intensities=protocol["intensities"],
        speeds=protocol["speeds"],
        height=protocol["height"],
        post_pad_mode=protocol["post_pad_mode"],
        t_pre=protocol["t_pre"],
        t_post=protocol["t_post"],
        dt=protocol["source_dt_seconds"],
        angles=list(angles),
    )


class AxoSimFlyVis:
    """Frozen AxoSim neurons on the exact expanded FlyVis graph."""

    def __init__(
        self,
        network,
        *,
        checkpoint,
        morphology_index,
        edge_current_gain,
        device,
        target_override=None,
    ):
        checkpoint = Path(checkpoint)
        if _sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("AxoSim checkpoint differs from the frozen protocol")
        model, metadata = load_checkpoint(checkpoint)
        if not 0 <= morphology_index < len(model.config.morphology_ids):
            raise ValueError("Unknown checkpoint morphology index")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.metadata = metadata
        self.morphology_index = int(morphology_index)
        self.edge_current_gain = float(edge_current_gain)
        params = network._param_api()
        self.source = network._source_indices.to(self.device)
        targets = (
            network._target_indices.detach().cpu().numpy()
            if target_override is None else np.asarray(target_override)
        )
        self.target = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        self.weight = params.edges.weight.detach().to(self.device).float()
        self.input_index = torch.as_tensor(
            network.stimulus.input_index[:].reshape(-1),
            dtype=torch.long,
            device=self.device,
        )
        self.central_index = np.asarray(network.connectome.central_cells_index[:])
        self.n_nodes = network.n_nodes
        self.route_feature = self.model.route_features[morphology_index].mean(0).to(
            self.device
        )
        self.adaptation_cache = self.model.compile_adaptation(
            self.model.behavior_adaptation_bank[morphology_index][None].to(self.device)
        ).expand(self.n_nodes, -1)

    def initial_recurrent_state(self):
        """Return the complete recurrent state at a graph-update boundary."""
        hidden = self.model.initial_state(
            self.n_nodes, device=self.device, dtype=torch.float32
        )
        activity = torch.zeros((self.n_nodes, 4), device=self.device)
        return hidden, activity

    @torch.inference_mode()
    def advance_block(self, stimulus_block, hidden, activity, *, current=None):
        """Advance one four-millisecond block without resetting recurrence.

        ``activity`` is updated at the externally clamped input indices before
        graph propagation. Callers that branch a trajectory must clone both
        ``hidden`` and ``activity`` first.
        """
        block = np.asarray(stimulus_block, dtype=np.float32)
        if block.shape != (4, 721):
            raise ValueError("Expected one stimulus block with shape (4, 721)")
        if current is None:
            current = torch.zeros_like(activity)
        photoreceptors = np.tile(block, (1, 8))
        activity.index_copy_(
            0,
            self.input_index,
            torch.as_tensor(photoreceptors.T, device=self.device),
        )
        messages = (
            activity.index_select(0, self.source)
            * self.weight[:, None]
            * self.edge_current_gain
        )
        current.zero_().index_add_(0, self.target, messages)
        forecast, hidden = self.model.step_p4(
            current.unsqueeze(-1) * self.route_feature,
            hidden,
            adaptation_cache=self.adaptation_cache,
        )
        activity = forecast[..., 1]
        if not torch.isfinite(activity).all() or not torch.isfinite(hidden).all():
            raise RuntimeError("Nonfinite AxoSim visual-circuit recurrent state")
        return hidden, activity

    @torch.inference_mode()
    def run_stimulus(
        self,
        stimulus,
        *,
        source_dt,
        target_dt,
        record_index=None,
    ):
        if stimulus.ndim != 2 or stimulus.shape[1] != 721:
            raise ValueError("Expected FlyVis stimulus with shape (time, 721)")
        resampled, target_time = _resample(stimulus, source_dt, target_dt)
        state, activity = self.initial_recurrent_state()
        current = torch.zeros_like(activity)
        central_activity = []
        record_index_tensor = None
        recorded_activity = []
        if record_index is not None:
            record_index_tensor = torch.as_tensor(
                np.asarray(record_index), dtype=torch.long, device=self.device
            )
        for start in range(0, len(resampled), 4):
            block = resampled[start:start + 4]
            if len(block) < 4:
                block = np.concatenate(
                    (block, np.repeat(block[-1:], 4 - len(block), axis=0)), axis=0
                )
            state, activity = self.advance_block(
                block,
                state,
                activity,
                current=current,
            )
            central_activity.append(activity[self.central_index].T.cpu().numpy())
            if record_index_tensor is not None:
                recorded_activity.append(
                    activity.index_select(0, record_index_tensor).T.cpu().numpy()
                )
        response = np.concatenate(central_activity, axis=0)[:len(resampled)]
        # step_p4 forecasts the following four milliseconds.
        prediction_time = target_time + target_dt
        source_time = np.arange(len(stimulus), dtype=np.float64) * source_dt
        from scipy.interpolate import interp1d

        aligned = interp1d(
            prediction_time,
            response,
            axis=0,
            bounds_error=False,
            fill_value=(response[0], response[-1]),
        )(source_time)
        aligned = aligned.astype(np.float32)
        if record_index_tensor is None:
            return aligned
        recorded = np.concatenate(recorded_activity, axis=0)[:len(resampled)]
        aligned_recorded = interp1d(
            prediction_time,
            recorded,
            axis=0,
            bounds_error=False,
            fill_value=(recorded[0], recorded[-1]),
        )(source_time).astype(np.float32)
        return aligned, aligned_recorded


def _replace_responses(reference, values, *, label):
    dataset = reference.copy(deep=False)
    dataset["responses"] = (
        ["network_id", "sample", "frame", "neuron"],
        np.asarray(values, dtype=np.float32)[None],
    )
    dataset.attrs = dict(reference.attrs)
    dataset.attrs["network_config"] = {"type": label}
    return dataset


def _peak_tuning_correlations(reference, candidate, *, angles):
    from flyvis.analysis.moving_bar_responses import peak_responses

    reference_peak = peak_responses(reference).values[0]
    candidate_peak = peak_responses(candidate).values[0]
    sample_angles = reference.angle.values
    intensities = reference.intensity.values
    cell_types = reference.cell_type.values.astype(str)
    result = {}
    for cell_type in SELECTIVITY_TYPES:
        intensity = 1 if cell_type.startswith("T4") else 0
        sample_mask = np.isin(sample_angles, angles) & (intensities == intensity)
        cell_index = int(np.flatnonzero(cell_types == cell_type)[0])
        result[cell_type] = _pearson(
            reference_peak[sample_mask, cell_index],
            candidate_peak[sample_mask, cell_index],
        )
    result["mean"] = float(np.mean(list(result.values())))
    return result


def _full_metrics(reference, candidate, held_out_angles):
    from flyvis.analysis.moving_bar_responses import (
        correlation_to_known_tuning_curves,
        direction_selectivity_index,
        dsi_correlation_to_known,
        preferred_direction,
    )

    dsi = direction_selectivity_index(candidate)
    direction = preferred_direction(candidate)
    tuning = correlation_to_known_tuning_curves(candidate)
    reference_direction = preferred_direction(reference)

    def item(array, cell_type, intensity=1):
        return float(
            array.custom.where(cell_type=cell_type, intensity=intensity).squeeze().item()
        )

    t4c_direction = item(direction, "T4c") / np.pi * 180
    t4c_reference_direction = item(reference_direction, "T4c") / np.pi * 180
    direction_error = abs((t4c_direction - t4c_reference_direction + 180) % 360 - 180)
    held_out = _peak_tuning_correlations(
        reference, candidate, angles=held_out_angles
    )
    return {
        "held_out_peak_tuning_correlation": held_out,
        "T4c_direction_selectivity_index": item(dsi, "T4c"),
        "T4c_preferred_direction_deg": t4c_direction,
        "T4c_reference_preferred_direction_deg": t4c_reference_direction,
        "T4c_preferred_direction_error_deg": direction_error,
        "median_dsi_correlation_to_known": float(
            dsi_correlation_to_known(dsi).median().item()
        ),
        "T4c_tuning_curve_correlation_to_known": item(tuning, "T4c"),
    }


def _run_dataset(runtime, dataset, reference, protocol):
    values = []
    for sample in range(len(dataset)):
        values.append(
            runtime.run_stimulus(
                reference.stimulus.values[sample, :, 0, :],
                source_dt=protocol["source_dt_seconds"],
                target_dt=protocol["axosim_dt_seconds"],
            )
        )
    return np.asarray(values)


def run_experiment(*, config, root, output, checkpoint, device="cuda"):
    config = Path(config)
    contract = json.loads(config.read_text())
    os.environ["FLYVIS_ROOT_DIR"] = str(Path(root).resolve())
    from flyvis import results_dir, root_dir
    from flyvis.network import NetworkView

    archive = Path(root_dir) / "results_pretrained_models.zip"
    if not archive.exists() or _sha256(archive) != PRETRAINED_ZIP_SHA256:
        raise ValueError("FlyVis pretrained archive differs from the frozen protocol")
    network_view = NetworkView(results_dir / contract["flyvis_network"])
    network = network_view.init_network("best")
    protocol = contract["protocol"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    training_dataset = _dataset(contract, protocol["training_angles_deg"])
    training_reference = network_view.moving_edge_responses(training_dataset)
    candidates = []
    for morphology_index in contract["calibration"]["morphology_indices"]:
        for gain in contract["calibration"]["edge_current_gains"]:
            runtime = AxoSimFlyVis(
                network,
                checkpoint=checkpoint,
                morphology_index=morphology_index,
                edge_current_gain=gain,
                device=device,
            )
            values = _run_dataset(runtime, training_dataset, training_reference, protocol)
            candidate = _replace_responses(
                training_reference,
                values,
                label=f"AxoSim morphology {morphology_index}, gain {gain}",
            )
            score = _peak_tuning_correlations(
                training_reference,
                candidate,
                angles=protocol["training_angles_deg"],
            )
            row = {
                "morphology_index": morphology_index,
                "edge_current_gain": gain,
                "training_peak_tuning_correlation": score,
            }
            candidates.append(row)
            print(json.dumps(row), flush=True)
    selected = max(
        candidates,
        key=lambda row: row["training_peak_tuning_correlation"]["mean"],
    )

    full_dataset = _dataset(contract, protocol["angles_deg"])
    full_reference = network_view.moving_edge_responses(full_dataset)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    runtime = AxoSimFlyVis(
        network,
        checkpoint=checkpoint,
        morphology_index=selected["morphology_index"],
        edge_current_gain=selected["edge_current_gain"],
        device=device,
    )
    selected_values = _run_dataset(runtime, full_dataset, full_reference, protocol)
    selected_dataset = _replace_responses(
        full_reference, selected_values, label="selected frozen AxoSim arm"
    )
    selected_metrics = _full_metrics(
        full_reference, selected_dataset, protocol["held_out_angles_deg"]
    )

    target_control = _spatial_target_control(
        network._target_indices.detach().cpu().numpy(),
        network.connectome.edges.target_type[:],
        seed=contract["control_seed"],
    )
    control_runtime = AxoSimFlyVis(
        network,
        checkpoint=checkpoint,
        morphology_index=selected["morphology_index"],
        edge_current_gain=selected["edge_current_gain"],
        device=device,
        target_override=target_control,
    )
    control_values = _run_dataset(
        control_runtime, full_dataset, full_reference, protocol
    )
    control_dataset = _replace_responses(
        full_reference, control_values, label="within-type target permutation control"
    )
    control_metrics = _full_metrics(
        full_reference, control_dataset, protocol["held_out_angles_deg"]
    )

    gray = np.full_like(full_reference.stimulus.values[0, :, 0, :], 0.5)
    gray_values = runtime.run_stimulus(
        gray,
        source_dt=protocol["source_dt_seconds"],
        target_dt=protocol["axosim_dt_seconds"],
    )
    after_warmup = gray_values[int(1 / protocol["source_dt_seconds"]):]
    peak_to_peak = np.ptp(after_warmup, axis=0)
    central_types = full_reference.cell_type.values.astype(str)
    maximum_index = int(np.argmax(peak_to_peak))
    gray_modulation = {
        "warmup_excluded_seconds": 1.0,
        "maximum_central_peak_to_peak": float(peak_to_peak[maximum_index]),
        "maximum_modulation_cell_type": str(central_types[maximum_index]),
        "median_central_standard_deviation": float(
            np.median(np.std(after_warmup, axis=0))
        ),
        "T4c_peak_to_peak": float(
            peak_to_peak[np.flatnonzero(central_types == "T4c")[0]]
        ),
    }

    gate = contract["video_gate"]
    mean_held_out = selected_metrics["held_out_peak_tuning_correlation"]["mean"]
    control_mean = control_metrics["held_out_peak_tuning_correlation"]["mean"]
    checks = {
        "positive_mean_held_out_tuning": (
            mean_held_out > gate["minimum_mean_held_out_t4_t5_correlation"]
        ),
        "T4c_direction_within_tolerance": (
            selected_metrics["T4c_preferred_direction_error_deg"]
            <= gate["maximum_t4c_preferred_direction_error_deg"]
        ),
        "spatial_control_reduces_primary_metric": control_mean < mean_held_out,
    }
    report = {
        "status": "complete",
        "scope": "frozen pretrained AxoSim substitution in the full FlyVis visual circuit",
        "claim_boundary": "visual-circuit model substitution; not validated fly electrophysiology",
        "config": str(config),
        "axosim_checkpoint_sha256": _sha256(checkpoint),
        "flyvis_pretrained_archive_sha256": _sha256(archive),
        "graph": {
            "nodes": network.n_nodes,
            "edges": network.n_edges,
            "expanded_graph_sha256": _tensor_sha256(
                network._source_indices.cpu().numpy(),
                network._target_indices.cpu().numpy(),
                network._param_api().edges.weight.detach().cpu().numpy(),
            ),
        },
        "candidate_count": len(candidates),
        "selected": selected,
        "selected_metrics": selected_metrics,
        "spatial_control_metrics": control_metrics,
        "constant_gray_control": gray_modulation,
        "video_gate_checks": checks,
        "video_gate_passed": all(checks.values()),
        "wall_seconds": time.monotonic() - started,
    }
    if device.startswith("cuda"):
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(
            torch.device(device)
        )
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(
            torch.device(device)
        )
    (output / "candidate-search.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "selected-responses.npz",
        axosim=selected_values,
        spatial_control=control_values,
        flyvis=full_reference.responses.values[0],
        constant_gray=gray_values,
        stimulus=full_reference.stimulus.values[:, :, 0, :],
        angles=full_reference.angle.values,
        intensities=full_reference.intensity.values,
        time=full_reference.time.values,
        cell_types=full_reference.cell_type.values,
    )
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_experiment(
        config=args.config,
        root=args.root,
        output=args.output,
        checkpoint=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
