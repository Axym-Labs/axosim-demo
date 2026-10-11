"""Runtime export and batched reference for the in-browser AxoSim connectome demo.

The browser runs the same frozen AxoSim-Lite neuron on the exact expanded FlyVis
graph used by ``vision_axosim.AxoSimFlyVis``. Because every cell shares one
morphology and one compiled adaptation cache, the per-cell update collapses to a
few small affine maps of the four propagated input currents. ``fold_parameters``
derives those maps from the checkpoint; ``BatchedFlyVis`` advances many images at
once with exactly the arithmetic the browser uses, and ``verify_folding`` checks
it against the unfolded PyTorch runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from .neural import DEFAULT_CHECKPOINT
from .vision_axosim import AxoSimFlyVis

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/browser_demo.json"
INPUT_TYPES = 8


def load_runtime(*, config=DEFAULT_CONFIG, root=ROOT / "data/flyvis", device="cuda"):
    """Return the selected frozen AxoSim FlyVis runtime and its network."""
    contract = json.loads(Path(config).read_text())
    vision = json.loads((ROOT / contract["vision_config"]).read_text())
    selected = json.loads((ROOT / contract["scored_summary"]).read_text())["selected"]
    os.environ["FLYVIS_ROOT_DIR"] = str(Path(root).resolve())
    from flyvis import results_dir
    from flyvis.network import NetworkView

    network = NetworkView(results_dir / vision["flyvis_network"]).init_network("best")
    runtime = AxoSimFlyVis(
        network,
        checkpoint=DEFAULT_CHECKPOINT,
        morphology_index=selected["morphology_index"],
        edge_current_gain=selected["edge_current_gain"],
        device=device,
    )
    return runtime, network, contract


def record_index(network, layers):
    return np.concatenate(
        [np.asarray(getattr(network.connectome.nodes.layer_index, name)[:]) for name in layers]
    ).astype(np.int64)


def retinal_lattice(network):
    """Normalized L1 column coordinates, identical to the natural-film sampler."""
    first = np.asarray(network.connectome.nodes.layer_index.L1[:])
    u = np.asarray(network.connectome.nodes.u[:])[first].astype(np.float64)
    v = np.asarray(network.connectome.nodes.v[:])[first].astype(np.float64)
    x = u + v / 2
    y = v * math.sqrt(3) / 2
    return x / np.max(np.abs(x)), y / np.max(np.abs(y))


@torch.no_grad()
def fold_parameters(runtime):
    """Collapse the shared AxoSim-Lite update into per-cell affine maps.

    Per cell with input currents ``c`` (4) and hidden state ``s`` (8):
    ``pre = P1 c + P2 s + p0``; ``s' = d s + (1 - d) g tanh(pre)``;
    ``a' = G1 s' + G2 c + g0`` where ``a'`` holds the four propagated soma outputs.
    """
    with torch.device("cpu"):
        return _fold_parameters(runtime)


def _fold_parameters(runtime):
    model = runtime.model
    cfg = model.config
    token, state, patch = cfg.token_dim, cfg.state_dim, cfg.patch_size
    dtype = torch.float64
    cache = runtime.adaptation_cache[0].to(dtype).cpu()
    slices = model.runtime_coefficient_slices
    part = {name: cache[selection] for name, selection in slices.items()}
    weight = model.input_projection.weight.detach().to(dtype).cpu()
    bias = model.input_projection.bias.detach().to(dtype).cpu()
    route = runtime.route_feature.detach().to(dtype).cpu()
    width = route.numel()
    projection = torch.stack(
        [weight[:, t * width:(t + 1) * width] @ route for t in range(patch)], dim=1
    )
    token_scale = 1.0 + part["token_gain"]
    projection = token_scale[:, None] * projection
    token_bias = (bias + part["token_offset"]) * token_scale
    support_in = model.support_input.weight.detach().to(dtype).cpu()
    support_rec = model.support_recurrent.weight.detach().to(dtype).cpu()
    p0 = (
        support_in @ token_bias
        + model.state_bias.detach().to(dtype).cpu()
        + part["state_offset"]
    )
    decay = torch.sigmoid(
        model.shared_decay_logit.detach().to(dtype).cpu() + part["decay_delta"]
    )
    rows = torch.arange(patch) * cfg.output_dim + 1
    output_scale = 1.0 + part["output_gain"][rows]
    head = model.output_head.weight.detach().to(dtype).cpu()[rows]
    head_bias = model.output_head.bias.detach().to(dtype).cpu()[rows]
    skip = model.skip_head.weight.detach().to(dtype).cpu()[rows]
    folded = {
        "P1": support_in @ projection,
        "P2": support_rec,
        "p0": p0,
        "decay": decay,
        "gain": 1.0 + part["state_gain"],
        "G1": output_scale[:, None] * head,
        "G2": output_scale[:, None] * (skip @ projection),
        "g0": (head_bias + skip @ token_bias) * output_scale + part["output_offset"][rows],
    }
    assert folded["P1"].shape == (state, patch) and folded["G1"].shape == (patch, state)
    return {name: value.float().numpy() for name, value in folded.items()}


def graph_csr(runtime):
    """Incoming-edge CSR sorted by (target, source), with weights times edge gain."""
    source = runtime.source.cpu().numpy().astype(np.int64)
    target = runtime.target.cpu().numpy().astype(np.int64)
    weight = (runtime.weight.cpu().numpy() * np.float32(runtime.edge_current_gain)).astype(
        np.float32
    )
    order = np.lexsort((source, target))
    source, target, weight = source[order], target[order], weight[order]
    row_ptr = np.zeros(runtime.n_nodes + 1, dtype=np.int64)
    np.add.at(row_ptr, target + 1, 1)
    row_ptr = np.cumsum(row_ptr)
    table, weight_index = np.unique(weight, return_inverse=True)
    if len(table) > 65535:
        raise ValueError("Edge weights do not fit a uint16 lookup table")
    return {
        "row_ptr": row_ptr.astype(np.int32),
        "col": source.astype(np.int32),
        "weight_index": weight_index.astype(np.uint16),
        "weight_table": table.astype(np.float32),
    }


class BatchedFlyVis:
    """Advance B independent copies of the circuit with the browser's arithmetic."""

    def __init__(self, runtime, network, contract, *, device="cuda"):
        self.device = torch.device(device)
        self.n = runtime.n_nodes
        self.inputs = runtime.input_index.numel()
        if not torch.equal(
            runtime.input_index.cpu(), torch.arange(self.inputs, device="cpu")
        ) or self.inputs != 721 * INPUT_TYPES:
            raise ValueError("Expected the photoreceptor inputs to occupy the first indices")
        graph = graph_csr(runtime)
        values = torch.as_tensor(graph["weight_table"][graph["weight_index"]])
        self.matrix = torch.sparse_csr_tensor(
            torch.as_tensor(graph["row_ptr"], dtype=torch.int64),
            torch.as_tensor(graph["col"], dtype=torch.int64),
            values,
            size=(self.n, self.n),
        ).to(self.device)
        folded = fold_parameters(runtime)
        self.folded = folded
        self.P1 = torch.as_tensor(folded["P1"].T.copy(), device=self.device)
        self.P2 = torch.as_tensor(folded["P2"].T.copy(), device=self.device)
        self.p0 = torch.as_tensor(folded["p0"], device=self.device)
        self.decay = torch.as_tensor(folded["decay"], device=self.device)
        self.gain = torch.as_tensor(folded["gain"], device=self.device)
        self.G1 = torch.as_tensor(folded["G1"].T.copy(), device=self.device)
        self.G2 = torch.as_tensor(folded["G2"].T.copy(), device=self.device)
        self.g0 = torch.as_tensor(folded["g0"], device=self.device)
        self.record = torch.as_tensor(
            record_index(network, contract["record_layers"]), device=self.device
        )

    def zeros(self, batch):
        hidden = torch.zeros((self.n, batch, 8), device=self.device)
        activity = torch.zeros((self.n, batch, 4), device=self.device)
        return hidden, activity

    @torch.no_grad()
    def step(self, stimulus, hidden, activity):
        """Advance one block. ``stimulus`` is (B, 4, 721); returns recorded (B, 4, R)."""
        batch = hidden.shape[1]
        clamp = stimulus.permute(2, 0, 1).repeat(INPUT_TYPES, 1, 1)
        activity[: self.inputs] = clamp
        current = (self.matrix @ activity.reshape(self.n, batch * 4)).reshape(
            self.n, batch, 4
        )
        pre = current @ self.P1 + hidden @ self.P2 + self.p0
        hidden = self.decay * hidden + (1.0 - self.decay) * self.gain * torch.tanh(pre)
        activity = hidden @ self.G1 + current @ self.G2 + self.g0
        recorded = activity.index_select(0, self.record).permute(1, 2, 0)
        return hidden, activity, recorded


@torch.no_grad()
def verify_folding(*, blocks=250, device="cuda", seed=0):
    """Compare folded batched updates with the unfolded runtime on random stimuli."""
    runtime, network, contract = load_runtime(device=device)
    batched = BatchedFlyVis(runtime, network, contract, device=device)
    rng = np.random.default_rng(seed)
    stimulus = np.clip(
        0.5 + 0.15 * rng.standard_normal((blocks, 4, 721)), 0.0, 1.0
    ).astype(np.float32)
    hidden, activity = runtime.initial_recurrent_state()
    current = torch.zeros_like(activity)
    folded_hidden, folded_activity = batched.zeros(1)
    worst_activity = worst_hidden = 0.0
    for block in stimulus:
        hidden, activity = runtime.advance_block(block, hidden, activity, current=current)
        folded_hidden, folded_activity, _ = batched.step(
            torch.as_tensor(block, device=batched.device)[None], folded_hidden, folded_activity
        )
        free = slice(batched.inputs, None)
        worst_activity = max(
            worst_activity,
            float((folded_activity[free, 0] - activity[free]).abs().max()),
        )
        worst_hidden = max(
            worst_hidden, float((folded_hidden[free, 0] - hidden[free]).abs().max())
        )
    scale = float(activity[batched.inputs:].abs().max())
    return {
        "blocks": blocks,
        "max_abs_activity_difference": worst_activity,
        "max_abs_hidden_difference": worst_hidden,
        "activity_scale": scale,
    }


def export_runtime(output, *, device="cuda"):
    """Write the browser runtime: graph CSR binaries plus a JSON descriptor."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime, network, contract = load_runtime(device=device)
    graph = graph_csr(runtime)
    files = {}
    for name in ("row_ptr", "col", "weight_index", "weight_table"):
        path = output / f"graph-{name.replace('_', '-')}.bin"
        array = graph[name]
        path.write_bytes(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())
        files[name] = {"path": path.name, "dtype": str(array.dtype), "length": int(array.size)}
    lattice_x, lattice_y = retinal_lattice(network)
    folded = fold_parameters(runtime)
    descriptor = {
        "nodes": int(runtime.n_nodes),
        "edges": int(len(graph["col"])),
        "input_cells": int(runtime.input_index.numel()),
        "columns": 721,
        "record_layers": contract["record_layers"],
        "record_index": record_index(network, contract["record_layers"]).tolist(),
        "lattice_x": np.round(lattice_x, 7).tolist(),
        "lattice_y": np.round(lattice_y, 7).tolist(),
        "stimulus": contract["stimulus"],
        "image_protocol": contract["image_protocol"],
        "model": {name: value.tolist() for name, value in folded.items()},
        "voltage_transform": {"scale": 0.1, "offset_mv": -67.7},
        "files": files,
    }
    (output / "runtime.json").write_text(json.dumps(descriptor, separators=(",", ":")))
    return {"output": str(output), "edges": descriptor["edges"], "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-folding")
    verify.add_argument("--blocks", type=int, default=250)
    export = sub.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-folding":
        result = verify_folding(blocks=args.blocks)
    else:
        result = export_runtime(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
