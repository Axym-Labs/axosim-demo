# Natural scene assets

The source assets are free public-domain **CC0-1.0** works from Poly Haven, not proprietary Axym assets. [Poly Haven license](https://polyhaven.com/license).

- [Food Apple 01](https://polyhaven.com/a/food_apple_01), **Oliver Harries**. Scanned mesh and 2K diffuse/normal/roughness maps. MuJoCo uses the diffuse map; normal and roughness maps are cached as reconstructable source assets in ignored `data/natural_scene/` but are not used in the current fixed-function renderer.
- [Wood Table 001](https://polyhaven.com/a/wood_table_001), **Dimitrios Savva** (photography), **Rico Cilliers** (processing). A 4096×4096 crop of the 16K photographed diffuse texture retains its physical scale: the 1.5-m source becomes a 375-mm crop.

`sources.json` records exact source download URLs, SHA-256 hashes and preprocessing. `python -m axosim_demo.prepare_scene_assets` downloads any missing sources into ignored `data/natural_scene/`, verifies them, and reproduces the crop and JPEG-to-PNG conversions required by MuJoCo.

The apple is transformed from glTF's +Y-up metres to MuJoCo's +Z-up millimetres and scaled to 65% of its source size, approximately 64 mm wide. Two small fruit-fragment meshes are authored for this demo and are thin geometric peel props textured with the photographed apple skin, not scanned chunk geometry. All fruit props are visual-only: the original flat ground remains the physical contact surface; food interaction and ingestion are not simulated.
