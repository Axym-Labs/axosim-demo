"""Stage the connectome demo's video and image presets from a film config and a web-preset config.

Writes, from the slot-resolved configs (``slot_choices`` of the film config):

* ``--web-clips``: H.264 web clips of the film clips (the raw displayed 40 s
  windows) and the web-only presets, the fitted held stills, and ``clips.json``
  (provenance: source URL and hash, window, output hash);
* ``--demo-work``/clips.json: the preset list ``demo_database.run_videos`` and
  ``export_site`` read (id, short title, theme, credit, path);
* ``--site``: videos/<id>.mp4 and videos/<id>.jpg (640x360 thumbnail at the clip's
  ``still_seconds``), images/<still>.jpg (1280x720) and images/presets.json.

``export_site`` then adds videos/<id>.lattice.bin and videos/presets.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

LICENSE_NOTES = {
    "Pexels license": "https://www.pexels.com/license/ - free commercial use and modification; attribution not required but given here; do not resell unaltered copies or redistribute on stock platforms",
    "public domain (NASA media; no copyright in the United States)": "https://www.nasa.gov/nasa-brand-center/images-and-media/ - NASA imagery generally not subject to US copyright; NASA must not be implied to endorse; no NASA insignia used",
}


def _builder():
    path = Path(__file__).resolve().parent / "build_neural_film_v3_program.py"
    spec = importlib.util.spec_from_file_location("film_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(config, presets, web_clips, demo_work, site, image_stills, *, thumb_quality=85):
    build = _builder()
    contract = json.loads(Path(config).read_text())
    program = contract["program"]
    choices = contract.get("slot_choices")
    items = build.validated_items(contract)
    extra = [
        {**clip, "kind": "video-clip", "role": "web preset"}
        for clip in build.resolve_slots(json.loads(Path(presets).read_text())["clips"], choices)
    ]
    for clip in extra:
        if _sha256(clip["path"]) != clip["source_sha256"]:
            raise ValueError(f"Source hash mismatch for {clip['name']}")
    web_clips, site = Path(web_clips), Path(site)
    records = build.export_web(items, program, web_clips, extra)

    provenance = {"clips": [], "frames": []}
    demo_clips = []
    for item, path in records:
        common = {
            "file": path.name,
            "title": item["title"],
            "theme": item.get("theme", "Earth from orbit ('eyes of the Earth' triptych)"),
            "role": item.get("role", "film clip" if item["kind"] == "video-clip" else "film held frame"),
            "creator": item["creator"],
            "license": item["license"],
            "page": item["page"],
            "source_url": item["url"],
            "source_sha256": item["source_sha256"],
        }
        if item["kind"] == "video-clip":
            provenance["clips"].append(
                common
                | {
                    "source_start_seconds": item["start_seconds"],
                    "duration_seconds": item["display_seconds"],
                    "still_seconds": item.get("still_seconds"),
                    "resolution": program["web_export"]["resolution"],
                    "fps": program["fps"],
                    "audio": False,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
            demo_clips.append(
                {
                    "id": item["name"],
                    "title": item["demo_title"],
                    "theme": item["demo_theme"],
                    "creator": item["creator"],
                    "license": item["license"],
                    "page_url": item["page"],
                    "path": str(path.resolve()),
                }
            )
        else:
            provenance["frames"].append(
                common | {"nasa_id": item.get("nasa_id"), "crop_box_pixels": item.get("crop_box"), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
    provenance = {
        "description": "Web-ready clips for the AxoSim interactive demo. The film clips are exactly the 40-second raw segments displayed in the neural film (before the pre-inference transition blending); the stills are the fitted held frames. The other clips are web-only presets. H.264, 1280x720, 30 fps, no audio, +faststart.",
        "slot_choices": choices,
        "license_notes": LICENSE_NOTES,
        **provenance,
    }
    (web_clips / "clips.json").write_text(json.dumps(provenance, indent=1, ensure_ascii=False) + "\n")
    (Path(demo_work) / "clips.json").write_text(json.dumps(demo_clips, indent=1) + "\n")

    # Site: clips and thumbnails (old preset files removed first).
    from PIL import Image

    videos, images = site / "videos", site / "images"
    shutil.rmtree(videos, ignore_errors=True)
    shutil.rmtree(images, ignore_errors=True)
    videos.mkdir(parents=True)
    images.mkdir(parents=True)
    by_name = {item["name"]: item for item, _ in records}
    size = tuple(program["resolution"])
    for clip in demo_clips:
        item = by_name[clip["id"]]
        shutil.copy(web_clips / f"{clip['id']}.mp4", videos / f"{clip['id']}.mp4")
        source = build.ClipSource(Path(item["path"]), size, item.get("centering", (0.5, 0.5)))
        try:
            frame = source.at(float(item["still_seconds"]))
        finally:
            source.close()
        frame.resize((640, 360), Image.Resampling.LANCZOS).save(videos / f"{clip['id']}.jpg", quality=thumb_quality)
        if clip["id"] in image_stills:
            frame.save(images / f"{clip['id']}-still.jpg", quality=int(program["web_export"]["jpeg_quality"]))
    image_presets = []
    for still in program["stills"]:
        shutil.copy(web_clips / f"{still['name']}.jpg", images / f"{still['name']}.jpg")
        image_presets.append(
            {
                "id": still["name"],
                "title": still["title"],
                "file": f"{still['name']}.jpg",
                "creator": "NASA",
                "license": "Public domain",
                "page_url": still["page"],
            }
        )
    for clip in demo_clips:
        if clip["id"] in image_stills:
            image_presets.append(
                {
                    "id": f"{clip['id']}-still",
                    "title": f"{clip['title']} (video still)",
                    "file": f"{clip['id']}-still.jpg",
                    "creator": clip["creator"],
                    "license": clip["license"],
                    "page_url": clip["page_url"],
                }
            )
    missing = set(image_stills) - {clip["id"] for clip in demo_clips}
    if missing:
        raise ValueError(f"Image stills for unknown clips: {sorted(missing)}")
    (images / "presets.json").write_text(json.dumps(image_presets, indent=1) + "\n")
    return {"videos": [c["id"] for c in demo_clips], "image_presets": [p["id"] for p in image_presets]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--presets", type=Path, required=True)
    parser.add_argument("--web-clips", type=Path, required=True)
    parser.add_argument("--demo-work", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True, help="the website's public/demo/connectome")
    parser.add_argument("--image-stills", nargs="+", required=True, help="clip ids that also become image presets")
    args = parser.parse_args()
    result = stage(args.config, args.presets, args.web_clips, args.demo_work, args.site, args.image_stills)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
