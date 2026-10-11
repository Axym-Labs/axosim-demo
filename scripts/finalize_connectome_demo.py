"""Fit UMAPs on the simulated image database and export site assets, the bundled fallback and the data.axym.org tree.

Run from the repository root after `python -m axosim_demo demo-database run-db`. The website checkout is
expected at ~/main/pinned/axym-website.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from axosim_demo import demo_database as d

W = Path("data/demo-work"); DB = Path("data/demo-image-db")
SITE = Path.home() / "main/pinned/axym-website/public/demo/connectome"
UP = W / "upload/axosim-demo"
t = time.time()
merged = d.merge_chunks(W)  # all chunks = the 20k
print("merged", merged, flush=True)
print("umaps", d.fit_umaps(W), round(time.time() - t), flush=True)
for name in ("state-cloud.bin", "state-cloud.json", "reference-stats.json", "latent.bin", "latent.json"):
    (SITE / name).unlink(missing_ok=True)
shutil.rmtree(SITE / "videos", ignore_errors=True)
print("site", d.export_site(W, SITE, clips=W / "clips.json"), flush=True)
# copy web clips + posters back (export_site rewrote videos/)
clips = Path.home() / "main/workspace/axosim-demo-internal/02-interactive-demo/artifacts/film/web-clips"
ff = ".venv-vision/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
for c in json.loads((W / "clips.json").read_text()):
    shutil.copy(clips / f"{c['id']}.mp4", SITE / "videos" / f"{c['id']}.mp4")
    os.system(f"{ff} -loglevel error -y -ss 3 -i {clips}/{c['id']}.mp4 -frames:v 1 -vf scale=640:360 -q:v 4 {SITE}/videos/{c['id']}.jpg")
# fallback DB (flat image layout, bundled with the site)
fb_ids = json.loads((DB / "fallback/ids.json").read_text())
fb_ids = fb_ids if isinstance(fb_ids, list) else fb_ids["ids"]
FB = SITE / "fallback-db"
shutil.rmtree(FB, ignore_errors=True)
print("fallback", d.export_database(DB, W, FB / "db", ids=fb_ids, sharded=False), flush=True)
for size in ("full", "thumb"):
    shutil.copytree(DB / "fallback" / size, FB / "images" / size)
# upload tree for data.axym.org: db + sharded images of the 20k (hard links)
shutil.rmtree(UP, ignore_errors=True)
print("primary", d.export_database(DB, W, UP / "db"), flush=True)
ids = json.loads((UP / "db/db.json").read_text())["meta"]["ids"]
for size in ("full", "thumb"):
    for i in ids:
        dst = UP / "images" / size / i[:2]
        dst.mkdir(parents=True, exist_ok=True)
        os.link(DB / "images" / size / i[:2] / f"{i}.jpg", dst / f"{i}.jpg")
print("upload images", len(ids), "done", round(time.time() - t), flush=True)
