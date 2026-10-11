"""Database pass for the in-browser connectome demo.

Pass 1 fits the latent space (standardization + principal components) on a
stratified image subset. Pass 2 runs every database image through the exact
browser protocol (gray warmup, constant image hold, free-state perturbations)
and stores the statistics the browser compares against. Later steps fit the
UMAPs and write the compact binary assets the website loads.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import demo_reference as ref
from .browser_demo import DEFAULT_CONFIG, BatchedFlyVis, load_runtime, retinal_lattice

STAT_KEYS = (
    "latent_speed",
    "path_density_percentile",
    "approach_ratio",
    "peak_frequency_drift_hz",
    "min_peak_power_fraction",
    "amplitude_ratio",
    "orbit_frequency_hz",
    "stable_orbit",
    "orbit_radius",
    "orbit_speed",
    "orbit_density_percentile",
    "return_ratio",
    "returned_seeds",
    "return_time",
    "return_time_seeds",
    "initial_distance",
)


def read_manifest(db):
    rows = [json.loads(line) for line in (Path(db) / "manifest.jsonl").read_text().splitlines() if line]
    return sorted(rows, key=lambda row: row["id"])


def image_path(db, row):
    return Path(db) / "images/full" / row["id"][:2] / f"{row['id']}.jpg"


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths, lattice, stimulus_cfg):
        self.paths, self.lattice, self.cfg = paths, lattice, stimulus_cfg

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return image_samples(self.paths[index], self.lattice, self.cfg)


def image_samples(path, lattice, cfg):
    """Normalized 721-column retinal sample of one still image."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    frame = ref.frame_from_rgb(rgb, width=cfg["frame_width"], height=cfg["frame_height"])
    sample = ref.lattice_sample(frame, *lattice, scale=cfg["lattice_scale"])
    norm = cfg["normalization"]
    return ref.normalize_samples(
        sample, mean=norm["mean"], std=norm["standard_deviation"], clip=tuple(norm["clip"])
    )


class Runner:
    def __init__(self, *, config=DEFAULT_CONFIG, device="cuda"):
        runtime, network, contract = load_runtime(config=config, device=device)
        self.contract = contract
        self.protocol = contract["image_protocol"]
        self.sim = BatchedFlyVis(runtime, network, contract, device=device)
        self.lattice = retinal_lattice(network)
        self.device = torch.device(device)
        self.warm_hidden, self.warm_activity, self.warm_record = self._warmup()

    @torch.no_grad()
    def _warmup(self):
        hidden, activity = self.sim.zeros(1)
        gray = torch.full((1, 4, 721), self.contract["stimulus"]["gray_level"], device=self.device)
        records = []
        for _ in range(ref.blocks_for(self.protocol["warmup_seconds"])):
            hidden, activity, recorded = self.sim.step(gray, hidden, activity)
            records.append(recorded[0])
        return hidden, activity, torch.cat(records)

    def loader(self, paths, batch):
        return torch.utils.data.DataLoader(
            _Images(paths, self.lattice, self.contract["stimulus"]),
            batch_size=batch,
            num_workers=12,
            shuffle=False,
            collate_fn=np.stack,
        )

    @torch.no_grad()
    def hold(self, samples, *, sigma=False):
        """Run the constant-image hold for (B, 721) samples."""
        batch = samples.shape[0]
        stimulus = torch.as_tensor(samples, device=self.device)[:, None, :].expand(batch, 4, 721).contiguous()
        hidden = self.warm_hidden.repeat(1, batch, 1)
        activity = self.warm_activity.repeat(1, batch, 1)
        blocks = ref.blocks_for(self.protocol["hold_seconds"])
        lo, hi = (ref.sample_index(s) for s in self.protocol["perturbation"]["sigma_window_seconds"])
        free = self.sim.inputs
        total = square = None
        count = 0
        records = []
        for block in range(blocks):
            hidden, activity, recorded = self.sim.step(stimulus, hidden, activity)
            records.append(recorded)
            if sigma and lo <= block * 4 and block * 4 + 3 < hi:
                state = torch.cat([hidden[free:], activity[free:]], dim=-1).double()
                total = state if total is None else total + state
                square = state**2 if square is None else square + state**2
                count += 1
        recorded = torch.cat(records, dim=1)
        result = {"recorded": recorded, "hidden": hidden, "activity": activity, "stimulus": stimulus}
        if sigma:
            mean = total / count
            result["sigma"] = torch.sqrt((square / count - mean**2).clamp_min(0)).float()
        return result


def fit_latent(db, work, *, config=DEFAULT_CONFIG, batch=32):
    """Pass 1: standardization, principal components and density reference."""
    runner = Runner(config=config)
    cfg = runner.contract["latent"]
    rows = read_manifest(db)
    rng = np.random.default_rng(cfg["random_seed"])
    by_category = {}
    for index, row in enumerate(rows):
        by_category.setdefault(row["category"], []).append(index)
    per = int(np.ceil(cfg["fit_images"] / len(by_category)))
    chosen = []
    for members in by_category.values():
        chosen.extend(rng.choice(members, size=min(per, len(members)), replace=False).tolist())
    chosen = sorted(rng.choice(chosen, size=min(cfg["fit_images"], len(chosen)), replace=False).tolist())
    paths = [image_path(db, rows[i]) for i in chosen]
    states = []
    hold_samples = ref.blocks_for(runner.protocol["hold_seconds"]) * 4
    started = time.monotonic()
    for number, samples in enumerate(runner.loader(paths, batch)):
        recorded = runner.hold(samples)["recorded"]
        pick = torch.as_tensor(
            np.stack([rng.choice(hold_samples, cfg["fit_samples_per_image"], replace=False) for _ in range(len(samples))]),
            device=recorded.device,
        )
        states.append(recorded.gather(1, pick[..., None].expand(-1, -1, recorded.shape[-1])).cpu())
        if number % 10 == 0:
            print(f"fit pass {number * batch}/{len(paths)} {time.monotonic() - started:.0f}s", flush=True)
    states = torch.cat(states).reshape(-1, states[0].shape[-1]).double()
    mean = states.mean(0)
    scale = states.std(0, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    standardized = ((states - mean) / scale).to(runner.device)
    _, singular, vh = torch.linalg.svd(standardized - standardized.mean(0), full_matrices=False)
    components = vh[: cfg["pca_components"]]
    # Fix component signs so the export is deterministic.
    signs = torch.sign(components[torch.arange(len(components)), components.abs().argmax(1)])
    components = components * signs[:, None]
    explained = float((singular[: cfg["pca_components"]] ** 2).sum() / (singular**2).sum())
    latent = ref.Latent(mean.float().numpy(), scale.float().numpy(), components.float().cpu().numpy())
    projected = latent.project(states.float().to(runner.device))
    reference_index = rng.choice(len(projected), size=min(cfg["density_reference_states"], len(projected)), replace=False)
    reference, quant = quantize(projected[reference_index].cpu().numpy())
    reference_t = torch.as_tensor(reference, device=runner.device)
    radius = ref.knn_radius(reference_t, reference_t, cfg["density_neighbors"] + 1)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    np.savez(
        work / "latent.npz",
        mean=latent.mean,
        scale=latent.scale,
        components=latent.components,
        density_reference_q=(reference / quant).round().astype(np.int8),
        density_quant=quant,
        density_sorted_radius=np.sort(radius.cpu().numpy()).astype(np.float32),
        fit_states_latent=projected.cpu().numpy().astype(np.float32),
        fit_image_index=np.repeat(np.asarray(chosen), cfg["fit_samples_per_image"]),
        warm_record=runner.warm_record.cpu().numpy(),
    )
    summary = {"images": len(chosen), "states": int(len(states)), "explained_variance": explained}
    (work / "latent-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def quantize(values):
    """Symmetric per-dimension int8 quantization; returns dequantized values and scales."""
    quant = np.abs(values).max(axis=0) / 127.0
    quant = np.where(quant > 0, quant, 1.0).astype(np.float32)
    return (np.round(values / quant).clip(-127, 127) * quant).astype(np.float32), quant


def load_latent(work, device):
    data = np.load(Path(work) / "latent.npz")
    latent = ref.Latent(data["mean"], data["scale"], data["components"])
    density = {
        "reference": torch.as_tensor(data["density_reference_q"].astype(np.float32) * data["density_quant"], device=device),
        "sorted_radius": torch.as_tensor(data["density_sorted_radius"], device=device),
    }
    return latent, density


def run_database(db, work, *, config=DEFAULT_CONFIG, batch=32, chunk=1024, limit=None, seed=7):
    """Pass 2: statistics for every database image, resumable in chunks."""
    runner = Runner(config=config)
    latent, density = load_latent(work, runner.device)
    density["k"] = runner.contract["latent"]["density_neighbors"]
    rows = read_manifest(db)[:limit]
    out = Path(work) / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    tail = runner.warm_record
    started = time.monotonic()
    for first in range(0, len(rows), chunk):
        path = out / f"{first:06d}.npz"
        if path.exists():
            continue
        part = rows[first:first + chunk]
        collected = {key: [] for key in (*STAT_KEYS, "descriptor")}
        generator = torch.Generator(device=runner.device).manual_seed(seed + first)
        for samples in runner.loader([image_path(db, row) for row in part], batch):
            hold = runner.hold(samples, sigma=True)
            stats = ref.image_statistics(hold["recorded"], tail, latent, density, runner.protocol)
            stats.update(
                ref.perturbation_stability(
                    runner.sim,
                    hold["stimulus"],
                    hold["hidden"],
                    hold["activity"],
                    hold["sigma"],
                    hold["recorded"],
                    runner.protocol,
                    generator=generator,
                )
            )
            for key in collected:
                collected[key].append(stats[key].detach().cpu().numpy())
        np.savez(path, ids=np.asarray([row["id"] for row in part]), **{k: np.concatenate(v) for k, v in collected.items()})
        done = first + len(part)
        rate = done / (time.monotonic() - started)
        print(f"database {done}/{len(rows)} {rate:.1f} img/s", flush=True)
    return merge_chunks(work)


def merge_chunks(work):
    parts = [np.load(path) for path in sorted((Path(work) / "chunks").glob("*.npz"))]
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0].files}
    np.savez(Path(work) / "database-stats.npz", **merged)
    return {"images": int(len(merged["ids"])), "stable_fraction": float(merged["stable_orbit"].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-latent")
    run = sub.add_parser("run-db")
    for command in (fit, run):
        command.add_argument("--db", type=Path, required=True)
        command.add_argument("--work", type=Path, required=True)
        command.add_argument("--batch", type=int, default=32)
    run.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "fit-latent":
        result = fit_latent(args.db, args.work, batch=args.batch)
    else:
        result = run_database(args.db, args.work, batch=args.batch, limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


def write_test_vectors(db, work, output, *, config=DEFAULT_CONFIG, count=2, blocks=40):
    """Fixtures for the browser parity tests: samples, early recording, statistics."""
    runner = Runner(config=config)
    latent, density = load_latent(work, runner.device)
    density["k"] = runner.contract["latent"]["density_neighbors"]
    rows = read_manifest(db)[:count]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in rows:
        path = image_path(db, row)
        samples = image_samples(path, runner.lattice, runner.contract["stimulus"])
        hold = runner.hold(samples[None], sigma=True)
        stats = ref.image_statistics(hold["recorded"], runner.warm_record, latent, density, runner.protocol)
        generator = torch.Generator(device=runner.device).manual_seed(0)
        stability = ref.perturbation_stability(
            runner.sim, hold["stimulus"], hold["hidden"], hold["activity"], hold["sigma"],
            hold["recorded"], runner.protocol, generator=generator,
        )
        recorded = hold["recorded"][0].cpu().numpy().astype("<f4")
        name = row["id"]
        (output / f"{name}.recorded.bin").write_bytes(recorded.tobytes())
        cases.append(
            {
                "id": name,
                "image": str(path),
                "samples": samples.astype(float).round(7).tolist(),
                "early_blocks": blocks,
                "early_recorded_sum": float(recorded[: blocks * 4].astype(np.float64).sum()),
                "recorded_shape": list(recorded.shape),
                "stats": {
                    key: float(value[0]) for key, value in stats.items() if key != "descriptor"
                },
                "stability": {
                    "return_time": float(stability["return_time"][0]),
                    "return_ratio": float(stability["return_ratio"][0]),
                    "return_time_seeds": stability["return_time_seeds"][0].tolist(),
                },
                "descriptor": stats["descriptor"][0].cpu().numpy().astype(float).tolist(),
            }
        )
    (output / "warm-record.bin").write_bytes(runner.warm_record.cpu().numpy().astype("<f4").tobytes())
    (output / "cases.json").write_text(json.dumps(cases))
    return {"cases": len(cases)}


def export_latent(work, output):
    """latent.json + latent.bin for the browser (little-endian float32/int8)."""
    data = np.load(Path(work) / "latent.npz")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parts = [
        ("mean", data["mean"].astype("<f4")),
        ("scale", data["scale"].astype("<f4")),
        ("components", data["components"].astype("<f4")),
        ("density_quant", data["density_quant"].astype("<f4")),
        ("density_sorted_radius", data["density_sorted_radius"].astype("<f4")),
        ("density_reference_q", data["density_reference_q"].astype(np.int8)),
    ]
    layout, offset, blobs = {}, 0, []
    for name, array in parts:
        layout[name] = {"offset": offset, "dtype": str(array.dtype.name), "shape": list(array.shape)}
        blob = array.tobytes()
        blobs.append(blob)
        offset += len(blob)
    (output / "latent.bin").write_bytes(b"".join(blobs))
    (output / "latent.json").write_text(json.dumps(layout))
    return layout
