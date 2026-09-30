#!/usr/bin/env python3
"""Download and verify the pinned flybrain mirrored connectome archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/mb_conditioning.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.config.read_text())
    source = contract["source"]
    output = args.output or ROOT / source["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/TheMrRaGe/flybrain/"
        f"{source['commit']}/brain_mirrored.npz"
    )
    partial = output.with_suffix(output.suffix + ".partial")
    urllib.request.urlretrieve(url, partial)
    digest = _sha256(partial)
    if digest != source["sha256"]:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"SHA256 mismatch: {digest}")
    partial.replace(output)
    print(json.dumps({"path": str(output), "sha256": digest, "url": url}, indent=2))


if __name__ == "__main__":
    main()
