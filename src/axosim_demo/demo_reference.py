"""Reference image protocol and statistics for the in-browser connectome demo.

Every function here has a line-for-line JavaScript counterpart in the website's
demo (``src/demo/connectome/``). The database statistics that the browser
compares against are produced with these functions, so the browser's
percentiles compare like with like.
"""

from __future__ import annotations

import math

import numpy as np
import torch

SAMPLE_SECONDS = 0.004
BLOCK_SAMPLES = 4
BLOCK_SECONDS = SAMPLE_SECONDS * BLOCK_SAMPLES


def blocks_for(seconds):
    """Number of whole blocks covering ``seconds`` (ceil, identical in JS)."""
    return int(math.ceil(seconds / BLOCK_SECONDS - 1e-9))


def sample_index(seconds):
    return int(round(seconds / SAMPLE_SECONDS))


def _cdist(a, b):
    """Exact Euclidean distances (no matmul expansion, which loses precision near 0)."""
    return torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")


# ---------------------------------------------------------------- stimulus


def crop_box(width, height):
    """Integer centre crop to 16:9."""
    crop_width = min(width, int(round(height * 16 / 9)))
    crop_height = min(height, int(round(width * 9 / 16)))
    return (width - crop_width) // 2, (height - crop_height) // 2, crop_width, crop_height


def area_weights(n_in, n_out):
    """Box-filter resampling matrix (n_out, n_in); rows sum to one."""
    scale = n_in / n_out
    weights = np.zeros((n_out, n_in), dtype=np.float64)
    for out in range(n_out):
        start, stop = out * scale, (out + 1) * scale
        first, last = int(math.floor(start)), int(math.ceil(stop))
        for source in range(first, min(last, n_in)):
            overlap = min(stop, source + 1) - max(start, source)
            if overlap > 0:
                weights[out, source] = overlap
        weights[out] /= weights[out].sum()
    return weights


def luminance(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def frame_from_rgb(rgb, *, width=640, height=360, crop=True):
    """Luminance frame (height, width) from an RGB array, as the browser computes it."""
    lum = luminance(rgb)
    if crop:
        x0, y0, cw, ch = crop_box(lum.shape[1], lum.shape[0])
        lum = lum[y0:y0 + ch, x0:x0 + cw]
    return area_weights(lum.shape[0], height) @ lum @ area_weights(lum.shape[1], width).T


def lattice_sample(frame, lattice_x, lattice_y, *, scale=0.47):
    """Bilinear sample of a (H, W) frame at the 721 hexagonal column centres."""
    height, width = frame.shape
    px = (0.5 + scale * np.asarray(lattice_x)) * (width - 1)
    py = (0.5 + scale * np.asarray(lattice_y)) * (height - 1)
    x0 = np.clip(np.floor(px).astype(int), 0, width - 2)
    y0 = np.clip(np.floor(py).astype(int), 0, height - 2)
    fx, fy = px - x0, py - y0
    top = frame[y0, x0] * (1 - fx) + frame[y0, x0 + 1] * fx
    bottom = frame[y0 + 1, x0] * (1 - fx) + frame[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def normalize_samples(samples, *, mean=0.5, std=0.15, clip=(0.0, 1.0)):
    """Per-sample retinal normalization of a (..., 721) array."""
    samples = np.asarray(samples, dtype=np.float64)
    centre = samples.mean(axis=-1, keepdims=True)
    spread = samples.std(axis=-1, keepdims=True)
    value = (samples - centre) / np.maximum(spread, 1e-6) * std + mean
    return np.clip(value, clip[0], clip[1]).astype(np.float32)


def video_samples(frames, fps, seconds):
    """Linear-in-time interpolation of per-frame lattice samples at 4 ms."""
    frames = np.asarray(frames, dtype=np.float64)
    count = int(math.ceil(seconds / SAMPLE_SECONDS - 1e-9))
    position = np.arange(count, dtype=np.float64) * SAMPLE_SECONDS * fps
    lower = np.minimum(np.floor(position).astype(int), len(frames) - 1)
    upper = np.minimum(lower + 1, len(frames) - 1)
    alpha = (position - np.floor(position))[:, None]
    return frames[lower] * (1 - alpha) + frames[upper] * alpha


# ---------------------------------------------------------------- latent space


class Latent:
    """Standardized recorded voltage projected on fixed principal components."""

    def __init__(self, mean, scale, components):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)

    def project(self, states):
        states = torch.as_tensor(states)
        device = states.device
        mean = torch.as_tensor(self.mean, device=device)
        scale = torch.as_tensor(self.scale, device=device)
        components = torch.as_tensor(self.components, device=device)
        return ((states - mean) / scale) @ components.T


def knn_radius(query, reference, k):
    """Distance to the k-th nearest reference point (Euclidean), batched on torch."""
    query = torch.as_tensor(query, dtype=torch.float32)
    reference = torch.as_tensor(reference, dtype=torch.float32, device=query.device)
    radii = []
    for chunk in torch.split(query, 2048):
        distance = _cdist(chunk, reference)
        radii.append(distance.topk(k, largest=False).values[:, -1])
    return torch.cat(radii)


def density_percentile(radius, sorted_reference_radius):
    """Percent of reference states that are less dense (larger k-NN radius)."""
    reference = torch.as_tensor(sorted_reference_radius, device=radius.device)
    larger = reference.numel() - torch.searchsorted(reference, radius, right=True)
    return 100.0 * larger.float() / reference.numel()


# ---------------------------------------------------------------- statistics


def _block_cadence(start, stop):
    """Sample indices at the last sample of each block within [start, stop)."""
    first = start + (-(start + 1)) % BLOCK_SAMPLES
    return np.arange(first, stop, BLOCK_SAMPLES)


def mean_speed(latent, start, stop):
    """Mean latent speed (units/s) between consecutive block-cadence samples."""
    index = _block_cadence(start, stop)
    path = latent[..., index, :]
    step = torch.linalg.vector_norm(path[..., 1:, :] - path[..., :-1, :], dim=-1)
    return step.mean(dim=-1) / BLOCK_SECONDS


def leading_component(values, iterations=100):
    """First principal component scores of (B, T, F) via power iteration.

    Deterministic start vector so the browser reproduces it exactly.
    """
    centred = values - values.mean(dim=1, keepdim=True)
    vector = torch.ones(values.shape[0], values.shape[2], 1, device=values.device)
    vector = vector / torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    for _ in range(iterations):
        vector = centred.transpose(1, 2) @ (centred @ vector)
        vector = vector / torch.linalg.vector_norm(vector, dim=1, keepdim=True).clamp_min(1e-30)
    return (centred @ vector)[..., 0]


def spectral_halves(component):
    """Film-identical spectral summary of two halves of a (B, T) component."""
    halves = torch.tensor_split(component, 2, dim=1)
    results = []
    for half in halves:
        half = half.double()
        index = torch.arange(half.shape[1], dtype=torch.float64, device=half.device)
        design = torch.stack([index, torch.ones_like(index)], dim=1)
        coefficients = torch.linalg.lstsq(design, half.T).solution
        residual = half - (design @ coefficients).T
        power = torch.fft.rfft(residual, dim=1).abs() ** 2
        power[:, 0] = 0
        frequency = torch.fft.rfftfreq(residual.shape[1], SAMPLE_SECONDS).to(half.device)
        peak = power.argmax(dim=1)
        results.append(
            {
                "peak_hz": frequency[peak],
                "peak_power_fraction": power.gather(1, peak[:, None])[:, 0] / power.sum(dim=1),
                "amplitude": residual.std(dim=1, unbiased=False),
            }
        )
    return results


def image_statistics(recorded, warmup_tail, latent_model, density, protocol):
    """Statistics of one batch of image holds.

    ``recorded`` is (B, S, R) raw recorded output over the hold; ``warmup_tail``
    (S0, R) is the shared gray warmup recording ending at image onset.
    """
    orbit_cfg = protocol["orbit"]
    hold = recorded.shape[1]
    latent = latent_model.project(recorded)
    tail_latent = latent_model.project(warmup_tail)
    stats = {}
    speed_lo, speed_hi = (sample_index(s) for s in protocol["speed_seconds"])
    stats["latent_speed"] = mean_speed(latent, speed_lo, min(speed_hi, hold))
    path_index = _block_cadence(0, hold)
    path_radius = knn_radius(
        latent[:, path_index].reshape(-1, latent.shape[-1]), density["reference"], density["k"]
    ).reshape(latent.shape[0], -1)
    path_percentile = density_percentile(path_radius, density["sorted_radius"])
    stats["path_density_percentile"] = path_percentile.mean(dim=1)
    d_lo, d_hi = (sample_index(s) for s in protocol["descriptor_seconds"])
    stats["descriptor"] = latent[:, d_lo:d_hi].mean(dim=1)

    ref_lo, ref_hi = (sample_index(s) for s in orbit_cfg["reference_orbit_seconds"])
    components = orbit_cfg["approach_pca_components"]
    reference = latent[:, ref_lo:ref_hi, :components]
    tran_lo, tran_hi = (sample_index(s) for s in orbit_cfg["transition_seconds"])
    transition = tail_latent[tail_latent.shape[0] + tran_lo: tail_latent.shape[0] + tran_hi, :components]
    post_lo, post_hi = (sample_index(s) for s in orbit_cfg["post_transition_seconds"])
    post = latent[:, post_lo:post_hi, :components]
    transition = transition[None].expand(latent.shape[0], -1, -1)
    d_transition = _cdist(transition, reference).min(dim=-1).values.median(dim=1).values
    d_post = _cdist(post, reference).min(dim=-1).values.median(dim=1).values
    approach = d_transition / d_post.clamp_min(1e-12)

    osc_lo, osc_hi = (sample_index(s) for s in orbit_cfg["oscillation_seconds"])
    values = recorded[:, osc_lo:osc_hi]
    spread = values.std(dim=1, unbiased=False, keepdim=True)
    variable = spread > 1e-8
    standardized = torch.where(
        variable, (values - values.mean(dim=1, keepdim=True)) / spread.clamp_min(1e-8), 0.0
    )
    halves = spectral_halves(leading_component(standardized))
    drift = (halves[0]["peak_hz"] - halves[1]["peak_hz"]).abs()
    fraction = torch.minimum(halves[0]["peak_power_fraction"], halves[1]["peak_power_fraction"])
    amplitude_ratio = halves[1]["amplitude"] / halves[0]["amplitude"].clamp_min(1e-30)
    low, high = orbit_cfg["amplitude_ratio"]
    stable = (
        (approach >= orbit_cfg["minimum_orbit_approach_ratio"])
        & (drift <= orbit_cfg["maximum_peak_frequency_drift_hz"])
        & (fraction >= orbit_cfg["minimum_peak_power_fraction"])
        & (amplitude_ratio >= low)
        & (amplitude_ratio <= high)
    )
    stats.update(
        approach_ratio=approach,
        peak_frequency_drift_hz=drift,
        min_peak_power_fraction=fraction.float(),
        amplitude_ratio=amplitude_ratio.float(),
        orbit_frequency_hz=halves[1]["peak_hz"].float(),
        stable_orbit=stable,
    )
    orbit = latent[:, ref_lo:ref_hi]
    stats["orbit_radius"] = torch.sqrt(
        ((orbit - orbit.mean(dim=1, keepdim=True)) ** 2).sum(dim=-1).mean(dim=1)
    )
    stats["orbit_speed"] = mean_speed(latent, ref_lo, ref_hi)
    orbit_mask = (path_index >= ref_lo) & (path_index < ref_hi)
    stats["orbit_density_percentile"] = path_percentile[:, orbit_mask].mean(dim=1)
    return stats


@torch.no_grad()
def perturbation_stability(sim, image_stimulus, hidden, activity, sigma, recorded, protocol, *, generator):
    """Return-to-orbit score after free-state perturbation, batched over images.

    ``hidden``/``activity`` are the states at the end of the hold (n, B, *),
    ``sigma`` the per-component orbit standard deviations (n, B, 12) of free cells,
    ``recorded`` the hold recording (B, S, R) and ``image_stimulus`` (B, 4, 721).
    Returns the median normalized late excess distance over seeds and the number
    of seeds that returned.
    """
    cfg = protocol["perturbation"]
    batch = hidden.shape[1]
    seeds = len(cfg["seeds"])
    branches = seeds + 1
    lo, hi = (sample_index(s) for s in cfg["distance_orbit_seconds"])
    orbit = recorded[:, lo:hi]
    scale = orbit.std(dim=1, unbiased=False)
    mask = scale > cfg["sigma_floor"]
    scale = torch.where(mask, scale, torch.ones_like(scale))
    weight = mask.float()
    free = sim.inputs
    h = hidden.repeat(1, branches, 1)
    a = activity.repeat(1, branches, 1)
    for seed in range(seeds):
        columns = slice((seed + 1) * batch, (seed + 2) * batch)
        noise = torch.randn((h.shape[0] - free, batch, 12), device=h.device, generator=generator)
        noise = noise * cfg["amplitude_sigma"] * sigma
        h[free:, columns] += noise[..., :8]
        a[free:, columns] += noise[..., 8:]
    stimulus = image_stimulus.repeat(branches, 1, 1)
    steps = blocks_for(cfg["continuation_seconds"])
    distances = []
    count = weight.sum(dim=-1).clamp_min(1)
    orbit_scaled = orbit / scale[:, None, :] * weight[:, None, :]
    for _ in range(steps):
        h, a, rec = sim.step(stimulus, h, a)
        state = rec[:, -1].reshape(branches, batch, -1).permute(1, 0, 2)
        state_scaled = state / scale[:, None, :] * weight[:, None, :]
        rms = _cdist(state_scaled, orbit_scaled) / torch.sqrt(count)[:, None, None]
        distances.append(rms.min(dim=-1).values)
    distance = torch.stack(distances, dim=-1)
    control, perturbed = distance[:, :1], distance[:, 1:]
    excess = (perturbed - control) / perturbed[..., :1].clamp_min(1e-12)
    late = int(round(cfg["late_seconds"] / BLOCK_SECONDS))
    per_seed = excess[..., -late:].median(dim=-1).values
    # Integrated normalized excess: equals the time constant of an exponential return.
    return_time = excess.clamp_min(0).sum(dim=-1) * BLOCK_SECONDS
    return {
        "return_ratio": per_seed.median(dim=1).values,
        "returned_seeds": (per_seed < cfg["return_threshold"]).sum(dim=1),
        "return_time": return_time.median(dim=1).values,
        "return_time_seeds": return_time,
        "initial_distance": perturbed[..., 0].mean(dim=1),
    }
