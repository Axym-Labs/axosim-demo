#!/usr/bin/env python3
"""Build the deterministic two-minute neural-film stimulus segments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageFilter, ImageOps


FPS = 30
DURATION_SECONDS = 16.0
WARMUP_SECONDS = 1.0
DISPLAY_SECONDS = 15.0
TRANSITION_SECONDS = 3.0
SIZE = (960, 540)
SEED = 20261001
APPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/9/9e/Apple_on_table.jpg"
APPLE_PAGE = "https://commons.wikimedia.org/wiki/File:Apple_on_table.jpg"
VIDEO_SOURCES = {
    "woodland": {
        "file": "woodland.mp4",
        "sha256": "ded3617a45b61a956bba53a490fbbc9f3f8583a2ac06f60e59002286d81e2bda",
        "page": "https://www.pexels.com/video/walking-into-the-woods-8381986/",
        "creator": "I Am Sorin",
    },
    "ocean": {
        "file": "ocean.mp4",
        "sha256": "6d9b2ca081223361c8b2d88541d3182f33ec2538ae6286b975487ce6daeff3e0",
        "page": "https://www.pexels.com/video/ocean-waves-19093263/",
        "creator": "Atlantic Ambience",
    },
    "city": {
        "file": "city.mp4",
        "sha256": "8921ca702d67972924389c61f4d9ca6b4b40a0ee6227dc569dcb245e71dc387a",
        "page": "https://www.pexels.com/video/people-walking-on-a-street-in-a-city-19329330/",
        "creator": "Ali Alcántara",
    },
}
COLORS = (
    np.asarray((74, 30, 160), dtype=np.uint8),
    np.asarray((218, 51, 136), dtype=np.uint8),
    np.asarray((245, 173, 67), dtype=np.uint8),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, SIZE, method=Image.Resampling.LANCZOS)


def video_frame(path: Path, seconds: float) -> Image.Image:
    reader = imageio.get_reader(path)
    try:
        fps = float(reader.get_meta_data()["fps"])
        position = seconds * fps
        lower = int(math.floor(position))
        alpha = position - lower
        first = Image.fromarray(reader.get_data(lower)).convert("RGB")
        second = Image.fromarray(reader.get_data(lower + 1)).convert("RGB")
    finally:
        reader.close()
    image = Image.blend(first, second, alpha)
    return ImageOps.fit(image, SIZE, method=Image.Resampling.LANCZOS)


def solid_color(rgb: np.ndarray) -> Image.Image:
    return Image.new("RGB", SIZE, tuple(int(value) for value in rgb))


def crossfade(previous: Image.Image, current: Image.Image, alpha: float) -> Image.Image:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    eased = alpha * alpha * (3 - 2 * alpha)
    blur = 24 * math.sin(math.pi * eased)
    if blur > 0.05:
        previous = previous.filter(ImageFilter.GaussianBlur(blur))
        current = current.filter(ImageFilter.GaussianBlur(blur))
    return Image.blend(previous, current, eased)


class NoiseSource:
    def __init__(self, seed: int):
        self.seed = seed

    def _key(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + index)
        return rng.integers(0, 256, size=(45, 80, 3), dtype=np.uint8).astype(np.float32)

    def frame(self, seconds: float) -> Image.Image:
        position = max(seconds, 0) * 6
        lower = int(math.floor(position))
        alpha = position - lower
        alpha = alpha * alpha * (3 - 2 * alpha)
        array = self._key(lower) * (1 - alpha) + self._key(lower + 1) * alpha
        image = Image.fromarray(array.astype(np.uint8), "RGB")
        image = image.filter(ImageFilter.GaussianBlur(1.1))
        return image.resize(SIZE, Image.Resampling.BICUBIC)


def write_segment(path: Path, frame_at) -> None:
    writer = imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        ffmpeg_params=["-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"],
    )
    try:
        for frame_index in range(round(DURATION_SECONDS * FPS) + 1):
            writer.append_data(np.asarray(frame_at(frame_index / FPS)))
    finally:
        writer.close()


def image_slot(current: Image.Image, following: Image.Image):
    transition_start = WARMUP_SECONDS + DISPLAY_SECONDS - TRANSITION_SECONDS

    def frame_at(seconds: float) -> Image.Image:
        if seconds < transition_start:
            return current
        alpha = (seconds - transition_start) / TRANSITION_SECONDS
        return crossfade(current, following, alpha)

    return frame_at


def color_slot(violet: Image.Image, magenta: Image.Image, gold: Image.Image):
    def frame_at(seconds: float) -> Image.Image:
        display = max(seconds - WARMUP_SECONDS, 0.0)
        if display < 3:
            return violet
        if display < 6:
            return crossfade(violet, magenta, (display - 3) / 3)
        if display < 9:
            return magenta
        if display < 12:
            return crossfade(magenta, gold, (display - 9) / 3)
        return gold

    return frame_at


def noise_slot(gold: Image.Image, noise: NoiseSource):
    def frame_at(seconds: float) -> Image.Image:
        display = max(seconds - WARMUP_SECONDS, 0.0)
        current = noise.frame(display)
        if display < TRANSITION_SECONDS:
            return crossfade(gold, current, display / TRANSITION_SECONDS)
        return current

    return frame_at


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/natural_manifold/sources"),
    )
    parser.add_argument(
        "--apple",
        type=Path,
        default=Path("data/natural_manifold/sources/apple-on-table.jpg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/natural_manifold/program-2min"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.apple.parent.mkdir(parents=True, exist_ok=True)
    if not args.apple.exists():
        request = urllib.request.Request(
            APPLE_URL,
            headers={"User-Agent": "AxoSimDemo/1.0 (https://axym.org)"},
        )
        with urllib.request.urlopen(request) as response:
            args.apple.write_bytes(response.read())

    video_paths = {
        name: args.source_dir / source["file"]
        for name, source in VIDEO_SOURCES.items()
    }
    for name, path in video_paths.items():
        if sha256(path) != VIDEO_SOURCES[name]["sha256"]:
            raise ValueError(f"Video hash mismatch for {name}")
    still_specs = [
        ("woodland-clearing", "woodland", 3.0),
        ("woodland-canopy", "woodland", 20.0),
        ("woodland-path", "woodland", 40.0),
        ("ocean-waves", "ocean", 5.0),
        ("city-street", "city", 2.0),
    ]
    natural_images = [
        video_frame(video_paths[source], seconds)
        for _, source, seconds in still_specs
    ]
    natural_images.append(fit_image(args.apple))
    violet, magenta, gold = [solid_color(color) for color in COLORS]

    frame_functions = [
        image_slot(image, following)
        for image, following in zip(natural_images, natural_images[1:] + [violet])
    ]
    frame_functions.extend(
        [
            color_slot(violet, magenta, gold),
            noise_slot(gold, NoiseSource(SEED)),
        ]
    )
    names = [
        "woodland-clearing",
        "woodland-canopy",
        "woodland-path",
        "ocean-waves",
        "city-street",
        "apple-on-table",
        "three-colors",
        "random-noise",
    ]
    kinds = ["natural-image"] * 6 + ["static-colors", "random-noise"]
    scenes = []
    for index, (name, kind, frame_at) in enumerate(zip(names, kinds, frame_functions)):
        path = args.output / f"segment-{index + 1:02d}-{name}.mp4"
        print(f"building {path}", flush=True)
        write_segment(path, frame_at)
        scenes.append(
            {
                "name": name,
                "kind": kind,
                "path": str(path),
                "duration_seconds": DURATION_SECONDS,
                "sha256": sha256(path),
            }
        )

    source_records = []
    for name, source, seconds in still_specs:
        record = VIDEO_SOURCES[source]
        source_records.append(
            {
                "name": name,
                "path": str(video_paths[source]),
                "sha256": record["sha256"],
                "frame_seconds": seconds,
                "page": record["page"],
                "license": "Pexels license",
                "attribution": record["creator"],
            }
        )
    source_records.append(
        {
            "name": "Apple on table",
            "path": str(args.apple),
            "sha256": sha256(args.apple),
            "page": APPLE_PAGE,
            "license": "public-domain",
            "attribution": "Kim Siever",
        }
    )
    manifest = {
        "fps": FPS,
        "resolution": list(SIZE),
        "duration_seconds_per_segment": DURATION_SECONDS,
        "display_seconds_per_segment": DISPLAY_SECONDS,
        "total_display_seconds": len(scenes) * DISPLAY_SECONDS,
        "transition": {
            "seconds": TRANSITION_SECONDS,
            "method": "smoothstep RGB interpolation with symmetric Gaussian blur",
            "placement": "the final three seconds of each image slot; the first three seconds of the noise slot",
        },
        "color_sequence": {
            "colors_rgb": [color.tolist() for color in COLORS],
            "schedule": "3 s color, 3 s transition, 3 s color, 3 s transition, 3 s color",
        },
        "random_seed": SEED,
        "sources": source_records,
        "scenes": scenes,
    }
    (args.output / "program-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
