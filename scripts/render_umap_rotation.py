"""Render the rotating side-by-side 3D UMAP loop for the AxoSim article.

Left: learned ten-dimensional motion subspace (Fig. 8), coloured by motion
direction. Right: raw T4/T5 voltage (the unused raw 3D UMAP), coloured by image
source as in Fig. 2. Both embeddings are the fitted 12,000-trial display
coordinates stored in ``analysis.npz``; nothing is refitted here.

Category ovals are drawn only for categories that pass the discernibility
criterion (``category_scores``), computed on the displayed points along the
video's own camera path. Ovals are the 1-sigma ellipses of each category's
projected points, so they track the cluster as it rotates. In the raw panel no
label grouping passes, so a few individual photographs (each photograph's 12
motion directions) are marked instead.

Outputs: MP4 (H.264), WebM (VP9), poster PNG, contact sheet, discernibility.json.

Example:
    .venv/bin/python scripts/render_umap_rotation.py \
        --data data/natural_geometry \
        --output ../axosim-demo-internal/02-interactive-demo/artifacts/umap-rotation
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.textpath import TextPath
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from axosim_demo.goodfire_tikz import INK, MUTED, PALETTE

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "src" / "axosim_demo" / "assets" / "fonts" / "InterVariable.ttf"

# Article colours. Directions use PALETTE[0..11] in sorted order exactly as
# Fig. 8 (_render_3d); sources use PALETTE[0] (ImageNet) / PALETTE[1]
# (iNaturalist) exactly as the Fig. 2 "Image source" panel.
SOURCE_LABEL = {"imagenet": "ImageNet", "inaturalist": "iNaturalist"}
SOURCE_SLOT = {"imagenet": 0, "inaturalist": 1}

W, H, FPS = 1600, 800, 30
DPI = 100
K_SIGMA = 1.0  # oval = 1-sigma ellipse of the projected category points
PURITY_MEAN_MIN = 0.5  # majority of the points inside the oval are the category
PURITY_FLOOR = 0.4  # ... and never fall far below that at any angle
LIFT_MIN = 2.0  # ... and at least twice the category's share of all points
LEADER_GAP = 16.0  # px between a small oval and its label
LEGEND_Y = 40.0
FADE = 0.75  # seconds per oval ease-in / ease-out
EXTENT_QUANTILE = 0.995


def rgb(hex_color):
    return np.array([int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return 1 - (1 - x) ** 3


# ----------------------------------------------------------------------- data
def sample_indices(count, size, seed):
    """Same deterministic display thinning as goodfire_tikz._sample_indices."""
    if size >= count:
        return np.arange(count)
    return np.random.default_rng(seed).choice(np.arange(count), size, replace=False)


def load(data_dir, points):
    data_dir = Path(data_dir)
    analysis = np.load(data_dir / "analysis.npz")
    learned = analysis["display_learned_umap_3d"].astype(np.float64)
    raw = analysis["activation_umap_3d"].astype(np.float64)
    rows = json.loads((data_dir / "manifest.json").read_text())["rows"]
    # record_voltages() iterates image-major, direction-minor over 0..330 deg.
    angles = np.arange(0, 360, 30)
    image = np.repeat(np.arange(len(rows)), len(angles))
    direction = np.tile(angles, len(rows))
    recording = data_dir / "axosim-optic-flow-voltage.npz"
    if recording.exists():
        with np.load(recording) as rec:
            assert np.array_equal(rec["image_index"], image)
            assert np.array_equal(rec["direction_degrees"], direction)
    assert len(image) == len(learned) == len(raw)
    keep = sample_indices(len(image), points, 44)  # 44 = Fig. 8 / raw 3D sample
    source = np.array([rows[i]["source"] for i in image])
    return {
        "learned": learned[keep],
        "raw": raw[keep],
        "image": image[keep],
        "direction": direction[keep],
        "source": source[keep],
        "rows": rows,
        "trial": keep,
    }


def normalise(values):
    values = values - values.mean(0)
    return values / np.sqrt((values**2).sum(1).mean())


# --------------------------------------------------------------------- camera
def camera(azimuth_deg, elevation_deg):
    """Rows: screen-right, screen-up, toward-viewer for a z-up turntable."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    right = np.array([np.cos(a), -np.sin(a), 0.0])
    forward = np.array([np.sin(a), np.cos(a), 0.0])  # into the screen at e=0
    up = np.array([0.0, 0.0, 1.0]) * np.cos(e) + forward * np.sin(e)
    toward = -forward * np.cos(e) + np.array([0.0, 0.0, 1.0]) * np.sin(e)
    return np.stack([right, up, toward])


def projected_ellipse(values, mask, rotation):
    mean = values[mask].mean(0)
    cov = np.cov(values[mask].T)
    proj = rotation[:2]
    return proj @ mean, proj @ cov @ proj.T


def inside(points2d, mean, cov, k=None):
    k = K_SIGMA if k is None else k
    diff = points2d - mean
    inv = np.linalg.inv(cov)
    return np.einsum("ij,jk,ik->i", diff, inv, diff) <= k * k


# ------------------------------------------------------------- discernibility
def silhouette(values, labels, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(values), min(n, len(values)), replace=False)
    x, lab = values[pick], labels[pick]
    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    a = np.zeros(len(x))
    b = np.full(len(x), np.inf)
    for c in np.unique(lab):
        member = lab == c
        mean_d = dist[:, member].sum(1)
        a[member] = mean_d[member] / max(member.sum() - 1, 1)
        b[~member] = np.minimum(b[~member], mean_d[~member] / member.sum())
    s = (b - a) / np.maximum(a, b)
    return {c: float(s[lab == c].mean()) for c in np.unique(lab)}


def category_scores(values, labels, azimuths, elevation, min_count=24):
    """Rotation-resolved purity of each category's projected K_SIGMA oval."""
    labels = np.asarray(labels)
    sil = silhouette(values, labels)
    rotations = [camera(a, elevation) for a in azimuths]
    scores = {}
    for c in np.unique(labels):
        member = labels == c
        if member.sum() < min_count:
            continue
        purity, recall = [], []
        for rot in rotations:
            p2 = values @ rot[:2].T
            mean, cov = projected_ellipse(values, member, rot)
            hit = inside(p2, mean, cov)
            purity.append((hit & member).sum() / max(hit.sum(), 1))
            recall.append((hit & member).sum() / member.sum())
        purity = np.asarray(purity)
        scores[str(c)] = {
            "n_points": int(member.sum()),
            "base_rate": float(member.mean()),
            "purity_mean": float(purity.mean()),
            "purity_min": float(purity.min()),
            "lift": float(purity.mean() / member.mean()),
            "recall_mean": float(np.mean(recall)),
            "silhouette_3d": sil[c],
            "discernible": bool(
                purity.mean() >= PURITY_MEAN_MIN
                and purity.min() >= PURITY_FLOOR
                and purity.mean() >= LIFT_MIN * member.mean()
            ),
        }
    return scores


def photograph_scores(values, image, azimuths, elevation):
    """Per-photograph oval purity (each photograph = its 12 motion directions)."""
    rotations = [camera(a, elevation) for a in azimuths]
    projections = [values @ rot[:2].T for rot in rotations]
    out = {}
    for c in np.unique(image):
        member = image == c
        if member.sum() < 4:
            continue
        purity = []
        for rot, p2 in zip(rotations, projections):
            mean, cov = projected_ellipse(values, member, rot)
            cov = cov + np.eye(2) * 1e-6
            hit = inside(p2, mean, cov)
            purity.append((hit & member).sum() / max(hit.sum(), 1))
        purity = np.asarray(purity)
        spread = float(np.sqrt(np.trace(np.cov(values[member].T))))
        out[int(c)] = {
            "n_points": int(member.sum()),
            "purity_mean": float(purity.mean()),
            "purity_min": float(purity.min()),
            "rms_radius": spread,
            "discernible": bool(
                purity.mean() >= PURITY_MEAN_MIN and purity.min() >= PURITY_FLOOR
            ),
        }
    return out


def pick_photographs(values, image, source, scores, count, elevation):
    """Pure, compact photographs whose labels never crowd during the turn.

    Candidates are discernible photographs with mean oval purity >= 0.9 inside
    the inner 85% of the cloud. Greedy max-min selection on the worst-case
    on-screen separation over the full rotation, with horizontal distance
    discounted (labels are wide), alternating ImageNet / iNaturalist.
    """
    ids = [c for c, s in scores.items() if s["discernible"] and s["purity_mean"] >= 0.9]
    ids.sort(key=lambda c: (-scores[c]["purity_min"], scores[c]["rms_radius"]))
    centres = np.array([values[image == c].mean(0) for c in ids])
    src = np.array([source[image == c][0] for c in ids])
    radius = np.sqrt((centres**2).sum(1))
    eligible = radius <= np.quantile(np.sqrt((values**2).sum(1)), 0.85)
    ids = [c for c, ok in zip(ids, eligible) if ok]
    centres, src = centres[eligible], src[eligible]
    screen = np.stack(
        [centres @ camera(a, elevation)[:2].T for a in np.arange(0, 360, 5)], 1
    )  # (candidates, azimuths, 2)
    chosen = [0]  # purest, most compact candidate
    while len(chosen) < count:
        want = "inaturalist" if len(chosen) % 2 else "imagenet"
        gap = np.full(len(ids), np.inf)
        for j in chosen:
            delta = np.abs(screen - screen[j][None])
            sep = np.maximum(delta[..., 0] / 3.0, delta[..., 1]).min(1)
            gap = np.minimum(gap, sep)
        gap[src != want] = -np.inf
        gap[chosen] = -np.inf
        chosen.append(int(np.argmax(gap)))
    return [ids[i] for i in chosen]


# ---------------------------------------------------------------------- fonts
def instanced_fonts(tmp):
    fonts = {}
    for name, weight in (("regular", 400), ("medium", 500), ("semibold", 600)):
        font = instantiateVariableFont(TTFont(FONT), {"wght": weight, "opsz": 14})
        path = Path(tmp) / f"Inter-{name}.ttf"
        font.save(path)
        fonts[name] = path
    return fonts


def text_size_px(text, prop, size_pt):
    prop = prop.copy()
    prop.set_size(size_pt)
    ext = TextPath((0, 0), text, prop=prop).get_extents()
    scale = DPI / 72
    return ext.width * scale, size_pt * scale


# --------------------------------------------------------------------- layout
class Panel:
    def __init__(self, values, colours, x0, width, top, bottom, elevation):
        self.values = values
        self.colours = colours
        self.elevation = elevation
        # Frame on the projected extent over the whole turn (0.5-99.5% of the
        # points at every angle), so rare outliers do not shrink the cloud;
        # they may reach into the margin and are clipped at the panel edge.
        q = (1 - EXTENT_QUANTILE, EXTENT_QUANTILE)
        u_lim, v_lim = [], []
        for az in np.arange(0, 360, 10):
            p = values @ camera(az, elevation)[:2].T
            u_lim.append(np.quantile(p[:, 0], q))
            v_lim.append(np.quantile(p[:, 1], q))
        u_half = np.abs(np.array(u_lim)).max()  # rotation axis stays centred
        v_lo, v_hi = np.min(v_lim), np.max(v_lim)
        self.scale = min(0.9 * width / (2 * u_half), (top - bottom) / (v_hi - v_lo))
        self.cx = x0 + width / 2
        self.cy = (top + bottom) / 2 - self.scale * (v_hi + v_lo) / 2
        self.depth_range = np.sqrt((values**2).sum(1)).max()

    def project(self, azimuth):
        rot = camera(azimuth, self.elevation)
        p = self.values @ rot.T
        xy = np.stack(
            [self.cx + self.scale * p[:, 0], self.cy + self.scale * p[:, 1]], 1
        )
        near = np.clip(0.5 + 0.5 * p[:, 2] / self.depth_range, 0, 1)
        return rot, xy, near

    def ellipse(self, rot, mask, min_radius_px=0.0):
        mean, cov = projected_ellipse(self.values, mask, rot)
        centre = np.array([self.cx, self.cy]) + self.scale * mean
        cov = cov * self.scale**2
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, (min_radius_px / K_SIGMA) ** 2)
        cov = evecs @ np.diag(evals) @ evecs.T
        return centre, cov


def ellipse_outline(centre, cov, k=None, n=96):
    k = K_SIGMA if k is None else k
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    chol = np.linalg.cholesky(cov)
    return centre + k * (chol @ np.stack([np.cos(t), np.sin(t)])).T


def _push(a, b, pad):
    """Minimal-axis displacement that moves rect a out of rect b ([cx, cy, w, h])."""
    dx, dy = a[0] - b[0], a[1] - b[1]
    ox = (a[2] + b[2]) / 2 + pad - abs(dx)
    oy = (a[3] + b[3]) / 2 + pad - abs(dy)
    if ox <= 0 or oy <= 0:
        return np.zeros(2)
    if oy < ox:
        return np.array([0.0, oy if dy >= 0 else -oy])
    return np.array([ox if dx >= 0 else -ox, 0.0])


def relax_labels(start, anchor, size, obstacles, bounds, iterations=40, pad=7.0):
    """Spring each label toward its anchor; push it off other labels/obstacles.

    Warm-started from the previous frame, so positions evolve continuously.
    """
    pos = start.copy()
    x0, x1, y0, y1 = bounds
    for _ in range(iterations):
        pos += 0.2 * (anchor - pos)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = _push(np.r_[pos[i], size[i]], np.r_[pos[j], size[j]], pad)
                pos[i] += d / 2
                pos[j] -= d / 2
            for ob in obstacles:
                pos[i] += _push(np.r_[pos[i], size[i]], ob, pad / 2)
        pos[:, 0] = np.clip(pos[:, 0], x0 + size[:, 0] / 2, x1 - size[:, 0] / 2)
        pos[:, 1] = np.clip(pos[:, 1], y0 + size[:, 1] / 2, y1 - size[:, 1] / 2)
    return pos


# ------------------------------------------------------------------- rendering
def render(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = load(args.data, args.points)
    frames = round(args.seconds * FPS)
    azimuths = args.azimuth0 + 360.0 * np.arange(frames) / frames
    times = np.arange(frames) / FPS

    learned = normalise(data["learned"])
    raw = normalise(data["raw"])
    direction = data["direction"]
    source = data["source"]
    image = data["image"]

    # ---- discernibility on the displayed points and the video's camera path
    score_az = azimuths[:: max(frames // 72, 1)]
    # Sectors of three adjacent directions centred on 0/90/180/270 degrees.
    sectors4 = (((direction + 30) % 360) // 90) * 90
    sectors6 = (((direction + 0) % 360) // 60) * 60
    rows = data["rows"]
    raw_groups = {
        "image_source": source,
        "label_group_coarse_label_path0": np.array(
            [rows[i]["label_path"][0] for i in image]
        ),
        # Fig. 2 "Taxonomic group": iNaturalist class, ImageNet second level.
        "label_group_fig2": np.array(
            [
                rows[i]["label_path"][0 if rows[i]["source"] == "inaturalist" else 1]
                for i in image
            ]
        ),
    }
    disc = {
        "criterion": (
            f"For each category, project the displayed points with the video's "
            f"camera at {len(score_az)} azimuths (elevation {args.elevation} deg) "
            f"and draw the category's {K_SIGMA}-sigma ellipse from its projected "
            f"mean and covariance. Purity = fraction of all points inside the "
            f"ellipse that belong to the category. A category is discernible "
            f"(gets an oval) if mean purity >= {PURITY_MEAN_MIN}, minimum "
            f"purity over the rotation >= {PURITY_FLOOR}, and mean purity >= "
            f"{LIFT_MIN} x the category's base rate (lift), i.e. the oval is "
            f"mostly its own category from every angle and clearly above chance. "
            f"Silhouette (3D, "
            f"Euclidean, 4,000-point sample) is reported as a secondary check."
        ),
        "points_displayed": len(image),
        "elevation_deg": args.elevation,
        "learned_subspace": {
            "directions_12": category_scores(
                learned, direction, score_az, args.elevation
            ),
            "sectors_6_of_60deg": category_scores(
                learned, sectors6, score_az, args.elevation
            ),
            "sectors_4_of_90deg_centered": category_scores(
                learned, sectors4, score_az, args.elevation
            ),
        },
        "raw_voltage": {
            kind: category_scores(raw, labels, score_az, args.elevation)
            for kind, labels in raw_groups.items()
        },
    }
    photo = photograph_scores(raw, image, score_az[::3], args.elevation)
    passing = np.array([s["discernible"] for s in photo.values()])
    disc["raw_voltage"]["photograph_summary"] = {
        "photographs": len(photo),
        "fraction_discernible": float(passing.mean()),
        "purity_mean_median": float(
            np.median([s["purity_mean"] for s in photo.values()])
        ),
        "note": f"{len(score_az[::3])} azimuths; per-photograph values in photograph_scores",
    }

    # ---- choose what gets ovals (driven by the criterion, not by hand)
    learned_sets = disc["learned_subspace"]
    candidates = {
        "directions_12": direction,
        "sectors_6_of_60deg": sectors6,
        "sectors_4_of_90deg_centered": sectors4,
    }
    passing = [
        k for k in candidates if all(s["discernible"] for s in learned_sets[k].values())
    ]
    # Finest fully discernible set, unless a coarser passing set is requested
    # for readability.
    left_kind = args.learned_set if args.learned_set in passing else passing[0]
    left_labels = candidates[left_kind]
    disc["learned_subspace"]["fully_discernible_sets"] = passing
    left_cats = [int(c) for c, s in learned_sets[left_kind].items() if s["discernible"]]
    raw_sets = disc["raw_voltage"]
    right_group = [
        (kind, c)
        for kind in raw_groups
        for c, s in raw_sets[kind].items()
        if s["discernible"]
    ]
    photos = pick_photographs(
        raw, image, source, photo, args.photographs, args.elevation
    )
    disc["raw_voltage"]["photograph_examples"] = {
        str(c): {
            "display_name": data["rows"][c]["display_name"],
            "source": data["rows"][c]["source"],
            **photo[c],
        }
        for c in photos
    }
    disc["raw_voltage"]["photograph_scores"] = {str(c): s for c, s in photo.items()}
    disc["ovals"] = {
        "learned_subspace": {"set": left_kind, "categories": left_cats},
        "raw_voltage": {
            "group_categories": [f"{k}:{c}" for k, c in right_group],
            "photograph_examples": [int(c) for c in photos],
            "photograph_selection": (
                "greedy max-min worst-case on-screen separation over the turn "
                "(horizontal distance / 3), sources alternating, among photographs "
                "with mean oval purity >= 0.9 in the inner 85% radius; the first "
                "is the purest, most compact one"
            ),
        },
    }
    (out / "discernibility.json").write_text(json.dumps(disc, indent=2) + "\n")

    # ---- figure
    tmp = Path(tempfile.mkdtemp(prefix="umap-rotation-"))
    fonts = instanced_fonts(tmp)
    prop = {k: FontProperties(fname=str(v)) for k, v in fonts.items()}

    left_colours = np.array([rgb(PALETTE[(d // 30) % 12]) for d in direction])
    right_colours = np.array([rgb(PALETTE[SOURCE_SLOT[s]]) for s in source])
    top, bottom = H - 80, 76
    panels = [
        Panel(learned, left_colours, 0, W / 2, top, bottom, args.elevation),
        Panel(raw, right_colours, W / 2, W / 2, top, bottom, args.elevation),
    ]

    # One spec per oval; min_r keeps tiny photograph clusters visible.
    def sector_title(c):
        deg = "\N{DEGREE SIGN}"
        if left_kind == "directions_12":
            return f"{c}{deg}"
        if left_kind == "sectors_6_of_60deg":
            return f"{c}\N{EN DASH}{c + 30}{deg}"
        return f"{c}{deg} \N{PLUS-MINUS SIGN} 30{deg}"

    def spec(panel, mask, colour, title, size=17.0, min_r=0.0, leader=False):
        return {
            "panel": panel,
            "mask": mask,
            "colour": colour,
            "title": title,
            "size": size,
            "min_r": min_r,
            "leader": leader,
        }

    specs = [
        spec(0, left_labels == c, rgb(PALETTE[(c // 30) % 12]), sector_title(c))
        for c in sorted(left_cats)
    ]
    group_slots = {}
    for kind, c in right_group:
        if kind == "image_source":
            slot, title = SOURCE_SLOT[c], SOURCE_LABEL[c]
        else:
            slot, title = 2 + len(group_slots) % 10, c
            group_slots[c] = slot
        specs.append(spec(1, raw_groups[kind] == c, rgb(PALETTE[slot]), title))
    for c in photos:
        name = rows[c]["display_name"]
        name = name[:1].upper() + name[1:].lower()
        colour = rgb(PALETTE[SOURCE_SLOT[rows[c]["source"]]])
        specs.append(spec(1, image == c, colour, name, 15.0, 10.0, leader=True))
    # Staggered ease-in alternating panels, ease-out mirrored before the loop end.
    order = sorted(range(len(specs)), key=lambda i: (specs[i]["panel"], i))
    left_ids = [i for i in order if specs[i]["panel"] == 0]
    right_ids = [i for i in order if specs[i]["panel"] == 1]
    interleaved = [x for pair in zip(left_ids, right_ids) for x in pair]
    interleaved += left_ids[len(right_ids) :] + right_ids[len(left_ids) :]
    step = min(0.32, 1.5 / max(len(specs) - 1, 1))
    for rank, i in enumerate(interleaved):
        specs[i]["t_in"] = 0.35 + rank * step
        specs[i]["t_out"] = args.seconds - 2.0 + rank * step * 0.6
    fade = FADE

    # Precompute ellipses and label anchors for every frame, then resolve
    # label overlaps and smooth label offsets in time (periodic).
    n_spec = len(specs)
    centres = np.zeros((frames, n_spec, 2))
    covs = np.zeros((frames, n_spec, 2, 2))
    label_xy = np.zeros((frames, n_spec, 2))
    label_wh = np.zeros((n_spec, 2))
    anchors_prev = None
    for i, s in enumerate(specs):
        w, h = text_size_px(s["title"], prop["semibold"], s["size"])
        label_wh[i] = (w + 22, h + 12)
    projections = []
    for f, az in enumerate(azimuths):
        per_panel = [p.project(az) for p in panels]
        projections.append(per_panel)
        anchors = np.zeros((n_spec, 2))
        obstacles = {0: [], 1: []}
        for i, s in enumerate(specs):
            rot = per_panel[s["panel"]][0]
            c, cov = panels[s["panel"]].ellipse(rot, s["mask"], s["min_r"])
            centres[f, i], covs[f, i] = c, cov
            top_y = c[1] + K_SIGMA * np.sqrt(cov[1, 1])
            lift = (
                LEADER_GAP + label_wh[i, 1] / 2 if s["leader"] else label_wh[i, 1] * 0.1
            )
            anchors[i] = (c[0], top_y + lift)
            if s["leader"]:  # small ovals must stay visible: labels avoid them
                r = K_SIGMA * np.sqrt(np.linalg.eigvalsh(cov).max())
                obstacles[s["panel"]].append(np.r_[c, 2 * r + 4, 2 * r + 4])
        if anchors_prev is None:
            label_xy[f] = anchors
        else:  # carry the previous offset so the solution moves continuously
            label_xy[f] = label_xy[f - 1] + (anchors - anchors_prev)
        for p in (0, 1):
            ids = [i for i in range(n_spec) if specs[i]["panel"] == p]
            if not ids:
                continue
            x0 = p * W / 2 + 14
            bounds = (x0, x0 + W / 2 - 28, bottom - 40, top + 30)
            label_xy[f, ids] = relax_labels(
                label_xy[f, ids], anchors[ids], label_wh[ids], obstacles[p], bounds
            )
        anchors_prev = anchors
    offsets = label_xy - centres
    offsets = gaussian_filter1d(offsets, sigma=3.0, axis=0, mode="wrap")
    label_xy = centres + offsets
    # Diagnostics: per-frame jumps while any part of the label is visible.
    vis_all = np.array(
        [
            [
                ease((t - s["t_in"]) / FADE) * (1 - ease((t - s["t_out"]) / FADE))
                for s in specs
            ]
            for t in times
        ]
    )
    shown = (vis_all[1:] > 0) & (vis_all[:-1] > 0)
    step_label = np.linalg.norm(np.diff(label_xy, axis=0), axis=-1)
    step_offset = np.linalg.norm(np.diff(offsets, axis=0), axis=-1)
    step_centre = np.linalg.norm(np.diff(centres, axis=0), axis=-1)
    motion = {
        "max_label_step_px": float(step_label[shown].max()),
        "max_label_offset_change_px": float(step_offset[shown].max()),
        "max_oval_centre_step_px": float(step_centre[shown].max()),
    }
    full_vis = (vis_all[1:] > 0.99) & (vis_all[:-1] > 0.99)
    motion["max_label_offset_change_px_fully_visible"] = float(
        step_offset[full_vis].max()
    )
    worst = np.unravel_index(
        np.argmax(np.where(full_vis, step_offset, 0)), full_vis.shape
    )
    motion["worst_offset_change"] = {
        "frame": int(worst[0]),
        "label": specs[worst[1]]["title"],
    }

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # static chrome
    titles = ("Learned motion subspace", "Raw T4/T5 voltage")
    for p, t in enumerate(titles):
        ax.text(
            p * W / 2 + W / 4,
            H - 46,
            t,
            ha="center",
            va="center",
            fontproperties=prop["medium"],
            fontsize=21,
            color=f"#{INK}",
        )
    ax.plot([W / 2, W / 2], [70, H - 70], color="#E4E2DA", lw=1.0, zorder=0)
    legend_y = LEGEND_Y
    legend_kw = {
        "va": "center",
        "fontproperties": prop["regular"],
        "fontsize": 14,
        "color": f"#{MUTED}",
    }
    # direction strip: 12 swatches, 0..330 degrees, same slots as Fig. 8
    gap = 19
    strip_w = 11 * gap
    x0 = W / 4 - strip_w / 2 + 52
    ax.text(x0 - 64, legend_y, "Direction", ha="right", **legend_kw)
    ax.text(x0 - 14, legend_y, "0\N{DEGREE SIGN}", ha="right", **legend_kw)
    ax.scatter(
        x0 + gap * np.arange(12),
        np.full(12, legend_y),
        s=95,
        color=[f"#{c}" for c in PALETTE[:12]],
        lw=0,
    )
    ax.text(x0 + strip_w + 14, legend_y, "330\N{DEGREE SIGN}", ha="left", **legend_kw)
    lx = 3 * W / 4 - 112
    for k, src in enumerate(("imagenet", "inaturalist")):
        x = lx + k * 140
        ax.scatter([x], [legend_y], s=95, color=f"#{PALETTE[SOURCE_SLOT[src]]}", lw=0)
        ax.text(x + 14, legend_y, SOURCE_LABEL[src], ha="left", **legend_kw)

    scatters = []
    for p in range(len(panels)):
        sc = ax.scatter([], [], s=[], lw=0, zorder=2)
        # Clip each cloud to its half so rare outliers never cross the divider.
        clip = Rectangle(
            (p * W / 2 + 4, 62), W / 2 - 8, H - 132, transform=ax.transData
        )
        sc.set_clip_path(clip)
        scatters.append(sc)
    oval_fill = [Polygon(np.zeros((3, 2)), closed=True, lw=0, zorder=3) for _ in specs]
    oval_halo = [
        Polygon(np.zeros((3, 2)), closed=True, fill=False, zorder=3.1) for _ in specs
    ]
    oval_edge = [
        Polygon(np.zeros((3, 2)), closed=True, fill=False, zorder=3.2) for _ in specs
    ]
    pills = [
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=12", zorder=4)
        for _ in specs
    ]
    texts = [
        ax.text(
            0,
            0,
            s["title"],
            ha="center",
            va="center",
            zorder=5,
            fontproperties=prop["semibold"],
            fontsize=s["size"],
        )
        for s in specs
    ]
    for artist in oval_fill + oval_halo + oval_edge + pills:
        ax.add_patch(artist)
    leaders = [
        ax.plot([], [], lw=1.4, zorder=3.3, solid_capstyle="round")[0] for _ in specs
    ]

    frame_dir = tmp / "frames"
    frame_dir.mkdir()
    base_size = args.point_size
    sheet_frames = np.linspace(0, frames, 9)[:-1].astype(int)
    full = [
        f
        for f in range(frames)
        if all(times[f] >= s["t_in"] + fade and times[f] <= s["t_out"] for s in specs)
    ]
    poster_frame = full[np.argmin(np.abs(azimuths[full] % 360 - args.poster_azimuth))]
    todo = range(frames)
    if args.preview:
        todo = sorted(set(sheet_frames.tolist()) | {poster_frame})
    for f in todo:
        t = times[f]
        vis = np.array(
            [
                ease((t - s["t_in"]) / fade) * (1 - ease((t - s["t_out"]) / fade))
                for s in specs
            ]
        )
        for p, (panel, sc) in enumerate(zip(panels, scatters)):
            rot, xy, near = projections[f][p]
            order = np.argsort(near)
            keep = 0.62 + 0.38 * near[:, None]  # fog distant points toward white
            colour = panel.colours * keep + (1 - keep)
            alpha = 0.50 + 0.45 * near
            # Dim points under visible labels of this panel.
            for i, s in enumerate(specs):
                if s["panel"] != p or vis[i] <= 0:
                    continue
                cx, cy = label_xy[f, i]
                w, h = label_wh[i] * (0.8 + 0.2 * vis[i])
                under = (np.abs(xy[:, 0] - cx) < w / 2 + 6) & (
                    np.abs(xy[:, 1] - cy) < h / 2 + 6
                )
                alpha = np.where(under, alpha * (1 - 0.8 * vis[i]), alpha)
            rgba = np.concatenate([colour, alpha[:, None]], 1)[order]
            diameter = base_size * (0.72 + 0.56 * near)
            sc.set_offsets(xy[order])
            sc.set_sizes((diameter[order] * 72 / DPI) ** 2)
            sc.set_facecolors(rgba)
        for i, s in enumerate(specs):
            v = vis[i]
            col = s["colour"]
            dark = col * 0.62
            if v <= 0.002:
                for artist in (
                    oval_fill[i],
                    oval_halo[i],
                    oval_edge[i],
                    pills[i],
                    texts[i],
                    leaders[i],
                ):
                    artist.set_visible(False)
                continue
            grow = 0.55 + 0.45 * v
            poly = ellipse_outline(centres[f, i], covs[f, i] * grow**2)
            oval_fill[i].set_xy(poly)
            oval_fill[i].set_facecolor((*col, 0.07 * v))
            oval_halo[i].set_xy(poly)
            oval_halo[i].set_edgecolor((1, 1, 1, 0.85 * v))
            oval_halo[i].set_linewidth(5.0)
            oval_edge[i].set_xy(poly)
            oval_edge[i].set_edgecolor((*dark, 0.95 * v))
            oval_edge[i].set_linewidth(2.0)
            lx_, ly_ = label_xy[f, i]
            lw_, lh_ = label_wh[i] * (0.8 + 0.2 * v)
            pills[i].set_bounds(lx_ - lw_ / 2, ly_ - lh_ / 2, lw_, lh_)
            pills[i].set_boxstyle("round", pad=0, rounding_size=lh_ / 2)
            pills[i].set_facecolor((1, 1, 1, 0.93 * v))
            pills[i].set_edgecolor((*dark, 0.9 * v))
            pills[i].set_linewidth(1.4)
            texts[i].set_position((lx_, ly_ + 1))
            texts[i].set_fontsize(s["size"] * (0.8 + 0.2 * v))
            texts[i].set_color((*(dark * 0.85), v))
            for artist in (
                oval_fill[i],
                oval_halo[i],
                oval_edge[i],
                pills[i],
                texts[i],
            ):
                artist.set_visible(True)
            if s["leader"]:
                ctr, cov = centres[f, i], covs[f, i] * grow**2
                reach = K_SIGMA * np.sqrt(cov[1, 1])
                above = ly_ >= ctr[1]
                rim = ctr[1] + (reach if above else -reach)
                foot = ly_ - lh_ / 2 if above else ly_ + lh_ / 2
                if (foot - rim) * (1 if above else -1) > 2:
                    leaders[i].set_data(
                        [
                            ctr[0],
                            np.clip(ctr[0], lx_ - lw_ / 2 + 10, lx_ + lw_ / 2 - 10),
                        ],
                        [rim, foot],
                    )
                    leaders[i].set_color((*dark, 0.9 * v))
                    leaders[i].set_visible(True)
                else:
                    leaders[i].set_visible(False)
        fig.savefig(frame_dir / f"{f:04d}.png", dpi=DPI, facecolor="white")
        if f % 50 == 0 and not args.preview:
            print(f"frame {f}/{frames}", flush=True)
    plt.close(fig)

    # ---- poster (all ovals fully shown), contact sheet, videos
    shutil.copy(frame_dir / f"{poster_frame:04d}.png", out / "umap-rotation-poster.png")
    thumbs = [
        Image.open(frame_dir / f"{f:04d}.png").convert("RGB") for f in sheet_frames
    ]
    tw, th = 800, 400
    sheet = Image.new("RGB", (2 * tw, 4 * th), "white")
    for k, im in enumerate(thumbs):
        sheet.paste(
            im.resize((tw, th), Image.Resampling.LANCZOS), ((k % 2) * tw, (k // 2) * th)
        )
    sheet.save(out / "umap-rotation-contact-sheet.png")

    if args.preview:
        shutil.rmtree(tmp)
        print(json.dumps(motion, indent=2))
        return None
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    pattern = str(frame_dir / "%04d.png")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-preset",
            "veryslow",
            "-crf",
            str(args.crf_h264),
            "-tune",
            "animation",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(out / "umap-rotation.mp4"),
        ],
        check=True,
    )
    for pass_ in (1, 2):
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                pattern,
                "-c:v",
                "libvpx-vp9",
                "-b:v",
                args.vp9_bitrate,
                "-row-mt",
                "1",
                "-deadline",
                "good",
                "-cpu-used",
                "1",
                "-aq-mode",
                "0",
                "-arnr-strength",
                "0",
                "-pass",
                str(pass_),
                "-passlogfile",
                str(tmp / "vp9"),
                "-pix_fmt",
                "yuv420p",
                "-an",
                *(
                    ["-f", "null", "/dev/null"]
                    if pass_ == 1
                    else [str(out / "umap-rotation.webm")]
                ),
            ],
            check=True,
        )
    if args.keep_frames:
        print(f"frames kept in {frame_dir}")
    else:
        shutil.rmtree(tmp)
    meta = {
        "frames": frames,
        "fps": FPS,
        "seconds": args.seconds,
        "points_per_panel": len(image),
        "poster_frame": int(poster_frame),
        "contact_sheet_frames": sheet_frames.tolist(),
        "motion_diagnostics": motion,
        "files_bytes": {
            p.name: p.stat().st_size for p in sorted(out.glob("umap-rotation*"))
        },
        "ovals": [
            {"panel": ("learned", "raw")[s["panel"]], "title": s["title"]}
            for s in specs
        ],
    }
    meta["command"] = "scripts/render_umap_rotation.py " + " ".join(
        f"--{k.replace('_', '-')} {v}"
        for k, v in vars(args).items()
        if k not in ("output", "data", "keep_frames", "preview")
    )
    (out / "render-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(ROOT / "data" / "natural_geometry"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--points", type=int, default=6000)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--elevation", type=float, default=40.0)
    parser.add_argument("--azimuth0", type=float, default=0.0)
    parser.add_argument("--poster-azimuth", type=float, default=180.0)
    parser.add_argument("--point-size", type=float, default=4.8)
    parser.add_argument("--photographs", type=int, default=4)
    parser.add_argument(
        "--learned-set",
        default="sectors_4_of_90deg_centered",
        choices=("directions_12", "sectors_6_of_60deg", "sectors_4_of_90deg_centered"),
    )
    parser.add_argument("--crf-h264", type=int, default=31)
    parser.add_argument("--vp9-bitrate", default="1550k")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--preview", action="store_true", help="contact sheet + poster only"
    )
    render(parser.parse_args())


if __name__ == "__main__":
    main()
