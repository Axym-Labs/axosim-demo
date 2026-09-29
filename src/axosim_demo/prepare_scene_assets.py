"""Verify CC0 source hashes and reproduce the exact MuJoCo texture conversions.

Run with ``python -m axosim_demo.prepare_scene_assets``. Missing source files
are fetched only from their pinned URLs in the adjacent provenance manifest.
"""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

from .scene import ASSETS

SOURCE_CACHE = Path(__file__).resolve().parents[2] / "data/natural_scene"


def main():
    manifest = json.loads((ASSETS / "sources.json").read_text())
    for item in manifest["assets"]:
        target = (ASSETS if item.get("runtime") else SOURCE_CACHE) / item["file"]
        if not target.exists():
            request = Request(
                item["source_url"],
                headers={"User-Agent": "axosim-demo (research preview)"},
            )
            with urlopen(request, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Source hash mismatch: {item['file']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {item['file']}")
    # Known, hash-verified 16384×16384 photograph, rather than untrusted input.
    Image.MAX_IMAGE_PIXELS = 300_000_000
    crop = manifest["preprocessing"]["tabletop_crop.png"]
    with Image.open(SOURCE_CACHE / crop["source"]) as photo:
        photo.crop(tuple(crop["pixel_box"])).save(ASSETS / "tabletop_crop.png")
    source = SOURCE_CACHE / "textures/food_apple_01_diff_2k.jpg"
    with Image.open(source) as photo:
        photo.save(ASSETS / "textures/food_apple_01_diff_2k.png")
    print("Source hashes verified; tabletop and apple PNG assets regenerated.")


if __name__ == "__main__":
    main()
