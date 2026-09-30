# AxoSim natural-vision geometry

This fork of [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym) runs a frozen pretrained AxoSim neuron model on the exact expanded FlyVis visual-circuit graph. The current demonstration measures neural responses directly; it contains no gait controller, steering rule, replay, behavior state machine, motor decoder, or simulated behavioral claim.

## Current result

The current display cohort contains **1,000 natural photographs**—500 from ImageNet-1k and 500 licensed research-grade iNaturalist observations spanning 494 species. Each moves in 12 known directions across the 721-column fly retinal lattice, producing 12,000 trials. Every trial runs the full 45,669-cell / 1,513,231-edge graph and records all 5,768 T4a–d and T5a–d neurons. AxoSim output channel 1 is converted back to soma membrane voltage with `V_mV = output / 0.1 - 67.7`.

The scientific score remains isolated to the original frozen 60-photograph cohort: 48 development photographs and 12 final photographs. The other 940 photographs affect the unsupervised display geometry only. The scored analysis removes each photograph's angle-averaged response. A development-only, six-fold grouped comparison selected the spatial mean and standard deviation within each anatomically named T4/T5 class over alternatives including a PCA of all retinal positions. Twelve additional photographs were then opened as a final test set:

- 12-way held-out balanced accuracy: **0.5625** (chance **0.0833**; image-bootstrap 95% interval **[0.4653, 0.6597]**)
- circular mean absolute error: **23.3°** (image-bootstrap 95% interval **[15.8°, 31.7°]**)
- within-image label-permutation test: **p = 0.00050**
- learned 10D distance correlation on held-out-image pairs: **r = 0.660**
- first circular harmonic: **77.4%** of direction-modulated power

The accuracy is one-of-12 optic-flow direction decoding on 144 trials from photographs absent from representation selection and decoder training. The distance correlation compares pairwise distance in the learned neural space with true circular motion-angle separation on those held-out photographs. It asks whether nearby directions have nearby neural states. Neither score measures image classification, taxonomy, behavior, or calibrated fly electrophysiology.

The claim gate passed. These numbers establish generalization of a direction signal to unseen natural photographs in this frozen substituted circuit. They do not establish calibrated fly electrophysiology, object recognition, behavior, or a learned phylogeny.

## Figure and video suite

The static suite reconstructs the individual figure compositions from Goodfire's [Finding the Tree of Life in Evo 2](https://www.goodfire.com/research/phylogeny-manifold) at their original pixel dimensions: the radial tree, sampling schematic, three-card UMAP, nearest-neighbor construction, two-panel distance card, three-panel held-out subspace card, residual card, statistic card, and raw/learned 3D views. The implementation uses Tectonic, TikZ and PGFPlots with Inter; it does not use Matplotlib. Every figure is retained as an editable `.tex` source, vector PDF and raster preview.

All 12,000 voltage samples fit the display embeddings. A six-setting sweep selected the raw-activation UMAP at 90 neighbors / 0.40 minimum distance for trustworthiness and canvas coverage, and the learned motion-space UMAP at 10 / 0.08 for trustworthiness and circular-neighbor fidelity. Only the rasterization step deterministically thins points to fit the vector renderer. The raw view shows the natural-image activation manifold; the learned view forms a clear cyclic motion manifold instead of the earlier centered-voltage noise cloud. The semantic substitutions are image labels for biological clades, optic-flow angle for phylogenetic distance, whole-image holdouts for clade holdouts, and AxoSim soma voltage for hidden activation. The tree documents ImageNet WordNet and iNaturalist taxonomic labels only; no neural fit or score uses it.

The retinotopic voltage fields and circular-harmonic figure adapt the natural-image and curve-response ideas in Goodfire's [Uncovering Neural Geometry in Vision Models](https://www.goodfire.com/research/bsf-vision). All plotted neural values derive from persisted AxoSim voltages. Goodfire's rotating 3D GIFs are represented as fixed-camera vector frames because presentation rotation was explicitly rejected.

The neural-state film uses pinned woodland, ocean and city footage, never ImageNet stills. It records 2,884 L1, Mi1, T4c and T5c cells over time, places the real-world video full frame, overlays measured retinotopic voltage and shows a fixed-camera 3D UMAP of the population state. A short causal trail follows the current response. The only in-frame text is `AxoSim - Axym Labs`; the presentation has no camera rotation or activity blinking. Its independent control separates spatial content at 1.000 held-out temporal-block balanced accuracy versus 0.333 for constant gray.

The first full-spatial representation is preserved as a negative development result. Its final-image classifier scored 0.0417 balanced accuracy and its raw nearest-neighbor geometry did not track motion angle. This motivated the development-only comparison of anatomically pooled summaries before the fresh final images were recorded.

## Reproduce

Use Python 3.12 on Linux:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-demo.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

Prepare the trained checkpoint; there is no random-weight fallback:

```bash
.venv/bin/python scripts/prepare_axosim_demo.py \
  --checkpoint-source /path/to/axosim-lite-t24-s8.pt
```

Then run the complete protocol:

```bash
PYTHONPATH=src .venv/bin/python -m axosim_demo natural-geometry prepare
PYTHONPATH=src .venv/bin/python -m axosim_demo natural-geometry record --device cuda
PYTHONPATH=src .venv/bin/python -m axosim_demo natural-geometry analyze --device cuda
PYTHONPATH=src .venv/bin/python -m axosim_demo natural-geometry render-plots
```

The real-world neural-state film has a separate stimulus and score contract:

```bash
PYTHONPATH=src .venv/bin/python -m axosim_demo natural-manifold run \
  --config configs/natural_manifold_spatial.json \
  --root data/flyvis --output data/natural_manifold/result --device cuda

PYTHONPATH=src .venv/bin/python -m axosim_demo natural-manifold render \
  --config configs/natural_manifold_spatial.json \
  --recording data/natural_manifold/result/manifold-recording.npz \
  --summary data/natural_manifold/result/summary.json \
  --output data/natural_manifold/rendered
```

The experiment contract is [`configs/natural_geometry.json`](configs/natural_geometry.json). It pins the AxoSim checkpoint SHA-256 (`19a045bf5b62ca92ab547934ca031e31f464ff2421f0d07b60d8278189889626`), the expanded graph hash, sampling seed, split, stimulus, fitting procedure, and display settings. Image manifests preserve source IDs, hashes, URLs, licenses and attributions.

## Supporting scientific work

- The same substituted circuit reached mean held-out T4/T5 moving-edge peak-tuning correlation 0.724 versus 0.079 after a within-type retinotopy scramble. T4c preferred-direction error was 1.97°, while its direction-selectivity magnitude remained weak at 0.044 versus the 0.627 FlyVis reference.
- Exact whole-connectome routing advances 127,400 AxoSim states while retaining 14,687,178 pair edges and 52,793,639 anatomical contacts at about 0.70 GiB peak CUDA allocation. Its first uniform dendritic-route hypothesis did not reproduce published grooming specificity.
- A dopamine-gated mushroom-body experiment retained all 162,517 released neurons and adapted only anatomical KC→MBON edges. Its preregistered reversal gate failed, so it remains a negative result rather than a demo.

The checkpoint's mammalian training domain limits biological interpretation, but it does not block the experiment. A body video remains out of scope until a validated neural-to-muscle chain generates movement without an inserted controller.

See [upstream attribution](UPSTREAM.md) and the private companion research record in `axosim-demo-internal`.
