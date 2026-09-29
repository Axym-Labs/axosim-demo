# Woodland environment assets

Powered by [Poly Haven](https://polyhaven.com). All source assets below are public domain under [CC0 1.0](https://polyhaven.com/license).

- [Pine tree 01](https://polyhaven.com/a/pine_tree_01): photography Rob Tuytel, modeling Rico Cilliers.
- [Fern 02](https://polyhaven.com/a/fern_02): scanning Rob Tuytel, modeling Rico Cilliers.
- [Tree stump 01](https://polyhaven.com/a/tree_stump_01): Rob Tuytel.
- [Mossy rock set 01](https://polyhaven.com/a/rock_moss_set_01): Kless Gyzen.
- [Bark debris 01](https://polyhaven.com/a/bark_debris_01): photography Greg Zaal, modeling Jenelle van Heerden.
- [Forest leaves 02](https://polyhaven.com/a/forest_leaves_02): Rob Tuytel.
- [Forest slope](https://polyhaven.com/a/forest_slope): Andreas Mischok.

The authored layout reuses assets from Poly Haven's [Pine Forest collection](https://polyhaven.com/collections/pine_forest); it is not the unmodified full Blender project. `sources.json` pins source URLs, SHA-256 hashes and byte lengths. `scene.json` describes OBJ/PNG placements in millimetres. Run `python -m axosim_demo.prepare_forest_assets` to verify/download sources and regenerate the runtime geometry and images.

Visual processing: glTF +Y-up metres become MuJoCo +Z-up millimetres; object-gallery offsets are removed; source UV texture transforms are preserved. Pine canopy LOD retains 45,000 distributed triangle pairs per tree, while trunks/bark/dead branches remain unsimplified. Fern cards become actual cutout geometry by 14×14 triangle subdivision and alpha-map threshold 150/255. Diffuse maps become PNG. The distant photographed panorama is resized to 4096×2048 and mapped inside a 20-metre sphere. Ground photography is 3 metres across. No asset is claimed to be part of the neuronal or physical contact model.
