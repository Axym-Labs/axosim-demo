#!/usr/bin/env python3
"""Build the three-clip, three-still stimulus program for the v3 neural film.

The film shows three 40-second clips at their original playback speed and then
holds three still photographs for five seconds each. Every transition (a
one-second smoothstep RGB interpolation with symmetric Gaussian blur, as in the
earlier image film) is written into the stimulus files, so AxoSim sees exactly
the frames that the film displays.

The program is written in the layout that ``natural_manifold.run_experiment``
already uses: overlapping six-second analysis windows with a five-second
stride and one second of leading context. Each clip program file is 41 seconds
long (one second of pre-roll footage, 39 seconds of raw footage and the
one-second transition into the next item) and backs eight consecutive
windows. Each still has its own six-second file.

The script fills the config's ``scenes`` list and program-manifest digest, and
optionally exports the raw 40-second clips and the fitted stills for the web.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageFilter, ImageOps

WINDOW_SECONDS = 6.0
CONTEXT_SECONDS = 1.0
STRIDE_SECONDS = 5.0
# The 2026-10 image film blurred 960-pixel frames by up to 24 pixels; keep the
# same blur relative to frame width.
BLUR_FRACTION_OF_WIDTH = 24 / 960


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3 - 2 * alpha)


def crossfade(previous: Image.Image, current: Image.Image, alpha: float) -> Image.Image:
    """Smoothstep RGB interpolation with symmetric blur (as the image film)."""
    eased = smoothstep(alpha)
    blur = BLUR_FRACTION_OF_WIDTH * previous.width * math.sin(math.pi * eased)
    if blur > 0.05:
        previous = previous.filter(ImageFilter.GaussianBlur(blur))
        current = current.filter(ImageFilter.GaussianBlur(blur))
    return Image.blend(previous, current, eased)


def source_frame_index(seconds: float, fps: float) -> int:
    """Nearest source frame at original playback speed (no blending)."""
    return int(math.floor(seconds * fps + 0.5 + 1e-9))


class ClipSource:
    """Sequential nearest-frame reader for one source video."""

    def __init__(self, path: Path, size, centering=(0.5, 0.5)):
        self.reader = imageio.get_reader(path)
        self.fps = float(self.reader.get_meta_data()["fps"])
        self.size = tuple(size)
        self.centering = tuple(centering)
        self.position = -1
        self.frame = None

    def at(self, seconds: float) -> Image.Image:
        index = source_frame_index(seconds, self.fps)
        if index < self.position:
            raise ValueError("ClipSource reads forward only")
        while self.position < index:
            self.frame = self.reader.get_next_data()
            self.position += 1
        image = Image.fromarray(self.frame).convert("RGB")
        return ImageOps.fit(
            image, self.size, method=Image.Resampling.LANCZOS, centering=self.centering
        )

    def close(self):
        self.reader.close()


def fit_still(path: Path, size, crop_box=None) -> Image.Image:
    """Crop a 16:9 pixel box (no flip or retouch), or centre-fit the photograph."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if crop_box is None:
        return ImageOps.fit(image, tuple(size), method=Image.Resampling.LANCZOS)
    left, top, right, bottom = crop_box
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError(f"Crop box outside {path}")
    if abs((right - left) / (bottom - top) - size[0] / size[1]) > 0.01:
        raise ValueError(f"Crop box for {path} is not {size[0]}:{size[1]}")
    return image.crop(crop_box).resize(tuple(size), Image.Resampling.LANCZOS)


def writer(path: Path, fps: int, crf: int, *, extra=()):
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        ffmpeg_params=[
            "-preset",
            "slow",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            *extra,
        ],
    )


def clip_program_seconds(display_seconds: float) -> float:
    return CONTEXT_SECONDS + display_seconds


def build_clip_program(clip, following, program, path: Path):
    """Write pre-roll, raw footage and the transition into ``following``."""
    fps = int(program["fps"])
    size = program["resolution"]
    display = float(clip["display_seconds"])
    transition = float(program["transition_seconds"])
    total = clip_program_seconds(display)
    transition_start = total - transition
    source = ClipSource(Path(clip["path"]), size, clip.get("centering", (0.5, 0.5)))
    next_source = None
    if following["kind"] == "video-clip":
        next_source = ClipSource(
            Path(following["path"]), size, following.get("centering", (0.5, 0.5))
        )
    else:
        next_still = fit_still(Path(following["path"]), size, following.get("crop_box"))
    out = writer(path, fps, int(program["stimulus_crf"]))
    try:
        for frame_index in range(round(total * fps) + 1):
            seconds = frame_index / fps
            # Program time zero is one second before the displayed clip starts.
            current = source.at(
                float(clip["start_seconds"]) - CONTEXT_SECONDS + seconds
            )
            if seconds >= transition_start:
                elapsed = seconds - transition_start
                if next_source is not None:
                    target = next_source.at(
                        float(following["start_seconds"]) - CONTEXT_SECONDS + elapsed
                    )
                else:
                    target = next_still
                current = crossfade(current, target, elapsed / transition)
            out.append_data(np.asarray(current))
    finally:
        out.close()
        source.close()
        if next_source is not None:
            next_source.close()


def build_still_segment(still, following, program, path: Path):
    fps = int(program["fps"])
    size = program["resolution"]
    transition = float(program["transition_seconds"])
    image = fit_still(Path(still["path"]), size, still.get("crop_box"))
    target = None
    if following is not None:
        target = fit_still(Path(following["path"]), size, following.get("crop_box"))
    transition_start = WINDOW_SECONDS - transition
    out = writer(path, fps, int(program["stimulus_crf"]))
    try:
        for frame_index in range(round(WINDOW_SECONDS * fps) + 1):
            seconds = frame_index / fps
            frame = image
            if target is not None and seconds >= transition_start:
                frame = crossfade(
                    image, target, (seconds - transition_start) / transition
                )
            out.append_data(np.asarray(frame))
    finally:
        out.close()
    return image


def export_web_clip(clip, program, path: Path, *, size, crf, maxrate_kbps):
    """Export the raw displayed clip (no pre-roll, no transition) for the web."""
    fps = int(program["fps"])
    source = ClipSource(Path(clip["path"]), size, clip.get("centering", (0.5, 0.5)))
    out = writer(
        path,
        fps,
        crf,
        extra=(
            "-maxrate",
            f"{maxrate_kbps}k",
            "-bufsize",
            f"{2 * maxrate_kbps}k",
            "-movflags",
            "+faststart",
            "-an",
        ),
    )
    try:
        for frame_index in range(round(float(clip["display_seconds"]) * fps)):
            out.append_data(
                np.asarray(source.at(float(clip["start_seconds"]) + frame_index / fps))
            )
    finally:
        out.close()
        source.close()


def program_items(program):
    items = [{**clip, "kind": "video-clip"} for clip in program["clips"]]
    items += [{**still, "kind": "held-frame"} for still in program["stills"]]
    return items


def scenes_for(items, files):
    """One six-second analysis window per five seconds of displayed program."""
    scenes = []
    for item, file in zip(items, files):
        count = (
            round(float(item["display_seconds"]) / STRIDE_SECONDS)
            if item["kind"] == "video-clip"
            else 1
        )
        for index in range(count):
            scenes.append(
                {
                    "name": item["name"]
                    if count == 1
                    else f"{item['name']}-{index + 1:02d}",
                    "kind": item["kind"],
                    "display_group": item["name"],
                    "path": str(file),
                    "creator": item["creator"],
                    "page": item["page"],
                    "license": item["license"],
                    "start_seconds": index * STRIDE_SECONDS,
                    "sha256": sha256(file),
                }
            )
    return scenes


def export_web(items, program, directory: Path, extra_clips=()):
    """Raw clips (exact displayed segment, no transitions) and fitted stills."""
    directory.mkdir(parents=True, exist_ok=True)
    web = program["web_export"]
    records = []
    for item in [*items, *extra_clips]:
        if item["kind"] == "video-clip":
            path = directory / f"{item['name']}.mp4"
            print(f"exporting {path}", flush=True)
            export_web_clip(
                item,
                program,
                path,
                size=web["resolution"],
                crf=int(web["crf"]),
                maxrate_kbps=int(web["maxrate_kbps"]),
            )
        else:
            path = directory / f"{item['name']}.jpg"
            fit_still(
                Path(item["path"]), program["resolution"], item.get("crop_box")
            ).save(path, quality=int(web["jpeg_quality"]))
        records.append((item, path))
    return records


def validated_items(contract):
    program = contract["program"]
    if program["fps"] != contract["display_fps"]:
        raise ValueError("Program and display frame rates must match")
    if (
        contract["duration_seconds"] != WINDOW_SECONDS
        or contract["warmup_seconds"] != CONTEXT_SECONDS
    ):
        raise ValueError("The program layout assumes 6-second windows with 1 s context")
    items = program_items(program)
    for item in items:
        if sha256(Path(item["path"])) != item["source_sha256"]:
            raise ValueError(f"Source hash mismatch for {item['name']}")
        if item["kind"] == "video-clip":
            if float(item["display_seconds"]) % STRIDE_SECONDS:
                raise ValueError("Clip durations must be multiples of 5 seconds")
            if float(item["start_seconds"]) < CONTEXT_SECONDS:
                raise ValueError("Clips need one second of pre-roll footage")
        else:
            item["display_seconds"] = STRIDE_SECONDS
    return items


def build_program(contract, items, config_path: Path):
    program = contract["program"]
    output = Path(program["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("program-*.mp4"):
        stale.unlink()
    files, fitted_hashes = [], {}
    for index, item in enumerate(items):
        following = items[index + 1] if index + 1 < len(items) else None
        path = output / f"program-{index + 1:02d}-{item['name']}.mp4"
        print(f"building {path}", flush=True)
        if item["kind"] == "video-clip":
            if following is None:
                raise ValueError("The program must end with held stills")
            build_clip_program(item, following, program, path)
        else:
            image = build_still_segment(item, following, program, path)
            fitted_hashes[item["name"]] = hashlib.sha256(image.tobytes()).hexdigest()
        files.append(path)
    if len(set(fitted_hashes.values())) != len(fitted_hashes):
        raise ValueError("Held stills must differ")
    scenes = scenes_for(items, files)
    displayed = sum(float(item["display_seconds"]) for item in items)
    manifest = {
        "fps": program["fps"],
        "resolution": program["resolution"],
        "window_seconds": WINDOW_SECONDS,
        "context_seconds": CONTEXT_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "total_display_seconds": displayed,
        "continuous_run_seconds": CONTEXT_SECONDS + displayed,
        "sequence": [item["name"] for item in items],
        "transition": {
            "seconds": program["transition_seconds"],
            "method": "smoothstep RGB interpolation with symmetric Gaussian blur (peak 2.5% of frame width)",
            "placement": "final displayed second of every item except the last still; clip-to-clip transitions blend into the next clip's pre-roll footage; last still held",
        },
        "source_frame_sampling": "nearest source frame at original playback speed; no blending, looping or slow motion",
        "fitted_still_sha256": fitted_hashes,
        "items": [
            item | {"file": str(file), "file_sha256": sha256(file)}
            for item, file in zip(items, files)
        ],
        "scenes": scenes,
    }
    manifest_path = output / "program-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    contract["scenes"] = scenes
    contract["program_manifest"] = str(manifest_path)
    contract["program_manifest_sha256"] = sha256(manifest_path)
    config_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(scenes)} analysis windows, {displayed:.0f} s displayed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/natural_film_v3.json")
    )
    parser.add_argument("--web-clips", type=Path, help="also export web clips here")
    parser.add_argument(
        "--presets", type=Path, help="JSON list of additional web-only clips to export"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="only export web clips; keep the program",
    )
    args = parser.parse_args()
    contract = json.loads(args.config.read_text())
    items = validated_items(contract)
    if not args.skip_build:
        build_program(contract, items, args.config)
    if args.web_clips:
        extra = []
        if args.presets:
            extra = [
                {**clip, "kind": "video-clip"}
                for clip in json.loads(args.presets.read_text())["clips"]
            ]
            for clip in extra:
                if sha256(Path(clip["path"])) != clip["source_sha256"]:
                    raise ValueError(f"Source hash mismatch for {clip['name']}")
        export_web(items, contract["program"], args.web_clips, extra)


if __name__ == "__main__":
    main()
