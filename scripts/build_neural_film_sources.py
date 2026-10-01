#!/usr/bin/env python3
"""Build the deterministic two-minute image program for the neural film."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageFilter, ImageOps

FPS = 30
DURATION_SECONDS = 6.0
WARMUP_SECONDS = 1.0
DISPLAY_SECONDS = 5.0
TRANSITION_SECONDS = 1.0
SIZE = (960, 540)
APPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/9/9e/Apple_on_table.jpg"
APPLE_PAGE = "https://commons.wikimedia.org/wiki/File:Apple_on_table.jpg"
APPLE_SHA256 = "1e716e7849d6e5959cbfa06847e37b01edd0ef044c02f2de022435d7b2ac312e"
VOYAGER_CATALOG = (
    "https://science.nasa.gov/mission/voyager/golden-record-contents/images/"
)

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

# Commons includes only the Golden Record images that are public-domain federal
# works or too simple for copyright. Each file page records that determination.
VOYAGER_SOURCES = (
    {
        "name": "voyager-earth",
        "file": "earth.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Voyager_golden_record_12_earth.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_12_earth.gif",
        "sha256": "d4e4f6b6579161a89e8a40f78908b325a037ffeba7765719ab3f8686501bb057",
        "attribution": "NASA",
    },
    {
        "name": "voyager-solar-spectrum",
        "file": "spectra.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3b/Voyager_golden_record_8_spectra.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_8_spectra.gif",
        "sha256": "678dee913092f8b666d260bf86f74550cb7b63ee36d699ff0284977f7341bd3c",
        "attribution": "National Astronomy and Ionosphere Center",
    },
    {
        "name": "voyager-mercury",
        "file": "mercury.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f6/Voyager_golden_record_9_mercury.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_9_mercury.gif",
        "sha256": "5165a6a8f9c56c74b3ca57ea50ac9b9439346d502b81e4d2e11356d141e52dee",
        "attribution": "NASA",
    },
    {
        "name": "voyager-mars",
        "file": "mars.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Voyager_golden_record_10_mars.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_10_mars.gif",
        "sha256": "0fab8fb7593e0af559e8ca283a2c5bceacf561a73b9de2f3e1e217da41c5cb0b",
        "attribution": "NASA",
    },
    {
        "name": "voyager-jupiter",
        "file": "jupiter.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Voyager_golden_record_11_jupiter.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_11_jupiter.gif",
        "sha256": "acce9bbcb980881433109327a210b3a80b0bb310a08ace7d162edb82a13bf663",
        "attribution": "NASA",
    },
    {
        "name": "voyager-earth-egypt",
        "file": "earth-egypt.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/Voyager_golden_record_13_earth.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_13_earth.gif",
        "sha256": "51a5aad1371ff8b1ba15ff0edc6792ae6be15419d0586e2c26bb6d01b2349b22",
        "attribution": "NASA",
    },
    {
        "name": "voyager-supermarket",
        "file": "supermarket.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/df/Voyager_golden_record_77_supermarket.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_77_supermarket.gif",
        "sha256": "4aab5644742f3003edc9d313feed432ebcc10687ac0fbaadd13aebe3da74529a",
        "attribution": "NASA Ames Research Center",
    },
    {
        "name": "voyager-eating-and-drinking",
        "file": "eating-drinking.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/1f/Voyager_golden_record_82_feeding.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_82_feeding.gif",
        "sha256": "8fe08a11b4898995063bd87b14e6ba5e6380fdb09dca8d2a220610c3b9d92b3d",
        "attribution": "National Astronomy and Ionosphere Center",
    },
    {
        "name": "voyager-xray-hand",
        "file": "xray-hand.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Voyager_golden_record_99_xray.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_99_xray.gif",
        "sha256": "51401f6199274153be1b34d317f63db4183c9608fdbce05c654fadaef7fe470f",
        "attribution": "National Astronomy and Ionosphere Center",
    },
    {
        "name": "voyager-modern-highway",
        "file": "highway.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/fd/Voyager_golden_record_103_highway.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_103_highway.gif",
        "sha256": "7f34df32a39cc8ffc38e8e17a2a7cc035a8d70e1ba84d3d85002508d704f3a5a",
        "attribution": "National Astronomy and Ionosphere Center",
    },
    {
        "name": "voyager-arecibo",
        "file": "arecibo.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/18/Voyager_golden_record_110_arecibo.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_110_arecibo.gif",
        "sha256": "5810b6e8e081c39019af8e1d2f8a339c4b124eed850ad4e8c6e63351d9969a55",
        "attribution": "National Astronomy and Ionosphere Center",
    },
    {
        "name": "voyager-astronaut",
        "file": "astronaut.gif",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/26/Voyager_golden_record_112_astronaut.gif",
        "page": "https://commons.wikimedia.org/wiki/File:Voyager_golden_record_112_astronaut.gif",
        "sha256": "78369f7537c6f0df6a0e8bef71f09d8c79428b32772808e64c8c1e4ecd27fc38",
        "attribution": "NASA",
    },
)

NATURAL_STILLS = (
    ("woodland-clearing", "woodland", 3.0),
    ("ocean-surface", "ocean", 2.0),
    ("woodland-canopy", "woodland", 10.0),
    ("city-pedestrians-a", "city", 1.0),
    ("ocean-waves", "ocean", 7.0),
    ("woodland-path", "woodland", 18.0),
    ("city-pedestrians-b", "city", 3.5),
    ("ocean-horizon", "ocean", 12.0),
    ("woodland-grove", "woodland", 28.0),
    ("woodland-trees", "woodland", 38.0),
    ("woodland-trail", "woodland", 48.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pinned(path: Path, url: str, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AxoSimDemo/1.0 (https://axym.org)"},
        )
        with urllib.request.urlopen(request) as response:
            path.write_bytes(response.read())
    if sha256(path) != expected_sha256:
        raise ValueError(f"Source hash mismatch for {path}")


def fit_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, SIZE, method=Image.Resampling.LANCZOS)


def video_frame(path: Path, seconds: float) -> Image.Image:
    reader = imageio.get_reader(path)
    try:
        fps = float(reader.get_meta_data()["fps"])
        position = seconds * fps
        lower = math.floor(position)
        alpha = position - lower
        first = Image.fromarray(reader.get_data(lower)).convert("RGB")
        second = Image.fromarray(reader.get_data(lower + 1)).convert("RGB")
    finally:
        reader.close()
    image = Image.blend(first, second, alpha)
    return ImageOps.fit(image, SIZE, method=Image.Resampling.LANCZOS)


def crossfade(previous: Image.Image, current: Image.Image, alpha: float) -> Image.Image:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    eased = alpha * alpha * (3 - 2 * alpha)
    blur = 24 * math.sin(math.pi * eased)
    if blur > 0.05:
        previous = previous.filter(ImageFilter.GaussianBlur(blur))
        current = current.filter(ImageFilter.GaussianBlur(blur))
    return Image.blend(previous, current, eased)


def write_segment(path: Path, current: Image.Image, following: Image.Image) -> None:
    transition_start = WARMUP_SECONDS + DISPLAY_SECONDS - TRANSITION_SECONDS
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
            seconds = frame_index / FPS
            if seconds < transition_start:
                frame = current
            else:
                frame = crossfade(
                    current,
                    following,
                    (seconds - transition_start) / TRANSITION_SECONDS,
                )
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


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
    for stale in args.output.glob("segment-*.mp4"):
        stale.unlink()

    download_pinned(args.apple, APPLE_URL, APPLE_SHA256)
    video_paths = {
        name: args.source_dir / source["file"] for name, source in VIDEO_SOURCES.items()
    }
    for name, path in video_paths.items():
        if sha256(path) != VIDEO_SOURCES[name]["sha256"]:
            raise ValueError(f"Video hash mismatch for {name}")

    voyager_dir = args.source_dir / "voyager"
    voyager_images = []
    for source in VOYAGER_SOURCES:
        path = voyager_dir / source["file"]
        download_pinned(path, source["url"], source["sha256"])
        voyager_images.append(fit_image(path))

    natural_images = [
        video_frame(video_paths[source], seconds)
        for _, source, seconds in NATURAL_STILLS
    ]
    apple = fit_image(args.apple)
    natural_items = [
        {"name": name, "kind": "natural-image", "image": image}
        for (name, _, _), image in zip(NATURAL_STILLS, natural_images)
    ]
    voyager_items = [
        {"name": source["name"], "kind": "voyager-record", "image": image}
        for source, image in zip(VOYAGER_SOURCES, voyager_images)
    ]

    # Apple, one Voyager image, two natural scenes, then alternating batches of
    # three Voyager and three natural images. The final Voyager batch uses two
    # slots so the program contains exactly 24 five-second intervals.
    items = [
        {"name": "apple-on-table", "kind": "natural-image", "image": apple},
        voyager_items[0],
        natural_items[0],
        natural_items[1],
        *voyager_items[1:4],
        *natural_items[2:5],
        *voyager_items[4:7],
        *natural_items[5:8],
        *voyager_items[7:10],
        *natural_items[8:11],
        *voyager_items[10:12],
    ]
    expected_kinds = [
        "natural-image",
        "voyager-record",
        "natural-image",
        "natural-image",
        *(["voyager-record"] * 3),
        *(["natural-image"] * 3),
        *(["voyager-record"] * 3),
        *(["natural-image"] * 3),
        *(["voyager-record"] * 3),
        *(["natural-image"] * 3),
        *(["voyager-record"] * 2),
    ]
    if len(items) != 24 or [item["kind"] for item in items] != expected_kinds:
        raise RuntimeError("The frozen 24-slot stimulus order is invalid")

    scenes = []
    for index, item in enumerate(items):
        following = items[(index + 1) % len(items)]["image"]
        path = args.output / f"segment-{index + 1:02d}-{item['name']}.mp4"
        print(f"building {path}", flush=True)
        write_segment(path, item["image"], following)
        scenes.append(
            {
                "name": item["name"],
                "kind": item["kind"],
                "path": str(path),
                "duration_seconds": DURATION_SECONDS,
                "sha256": sha256(path),
            }
        )

    source_records = [
        {
            "name": "apple-on-table",
            "path": str(args.apple),
            "sha256": sha256(args.apple),
            "page": APPLE_PAGE,
            "license": "public domain",
            "attribution": "Kim Siever",
        }
    ]
    for name, source, seconds in NATURAL_STILLS:
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
    for source in VOYAGER_SOURCES:
        source_records.append(
            {
                "name": source["name"],
                "path": str(voyager_dir / source["file"]),
                "sha256": source["sha256"],
                "page": source["page"],
                "catalog": VOYAGER_CATALOG,
                "license": "public domain",
                "attribution": source["attribution"],
            }
        )

    manifest = {
        "fps": FPS,
        "resolution": list(SIZE),
        "duration_seconds_per_segment": DURATION_SECONDS,
        "display_seconds_per_segment": DISPLAY_SECONDS,
        "total_display_seconds": len(scenes) * DISPLAY_SECONDS,
        "sequence_rule": "apple; one Voyager; two natural; alternating three-Voyager and three-natural batches; final two-Voyager partial batch",
        "transition": {
            "seconds": TRANSITION_SECONDS,
            "method": "smoothstep RGB interpolation with symmetric Gaussian blur",
            "placement": "final one second of every five-second image interval",
        },
        "sources": source_records,
        "scenes": scenes,
    }
    (args.output / "program-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
