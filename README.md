# AxoSim natural-vision geometry

This fork of [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym) runs a frozen pretrained AxoSim neuron model on the exact expanded FlyVis visual-circuit graph. The current demonstration measures neural responses directly; it contains no gait controller, steering rule, replay, behavior state machine, motor decoder, or simulated behavioral claim.

The two-minute film and interpretability results are published in [Finding Visual Dynamics in a Simulated Fly Brain](https://axym.org/work/axosim-fly-geometry/).

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

The two-minute neural-state film contains 24 five-second image intervals: twelve naturalistic images sampled from the pinned woodland, ocean and city footage plus a public-domain apple, and twelve public-domain images carried on the Voyager Golden Record. It opens with the apple, one Voyager image and two natural scenes, then alternates three-image Voyager and natural-scene batches. ImageNet is not used in the film. Every interval holds for four seconds and uses its final second for smoothstep RGB interpolation with symmetric blur. Every generated source frame passes through AxoSim before rendering, keeping the visible stimulus and neural state time-aligned.

The film records 2,884 L1, Mi1, T4c and T5c cells, places the stimulus full frame, and shows a fixed-camera 3D UMAP of the population state. A short causal trail follows the current response. No activity field is drawn over the source imagery. The only in-frame text is the unoutlined white Inter title `AxoSim - Axym Labs`, right-aligned beneath the state space; the presentation has no camera rotation or activity blinking. One continuous 121-second AxoSim run supplies every displayed state, with overlapping context windows that prevent neural resets at visible boundaries. A held-out temporal-block classifier separates the 24 stimulus intervals at **1.000** balanced accuracy versus **0.0806** for the time-matched continuous mean-gray control (nominal chance **0.0417**). Across 23 cross-image transitions, the median transition-to-post-transition distance from the late recurrent orbit decreases by a factor of **2.17**. In the steady portion of all 24 intervals, peak-frequency drift between consecutive 1.5-second windows is **0 Hz**, and oscillation-amplitude ratios range from **0.909 to 1.302**. These measurements support a stable oscillatory trajectory; establishing a limit-cycle attractor requires perturbation-and-return validation.

The perturbation-and-return experiment now provides that validation within the substituted model. It displaces all **478,812 free recurrent-state variables**—eight AxoSim hidden values and four propagated activity values for each of 39,901 unclamped recurrent cells—and measures phase-insensitive distance to the pre-perturbation orbit against a matched unperturbed continuation. Across six constant images, the **0.10σ** displacement returns in **29/30** trials and the **0.25σ** displacement returns in **27/30** trials. Both preregistered gates pass. This supports a local numerical limit-cycle attractor under the tested inputs; it does not establish the same attractor in fly physiology or under biologically delivered perturbations.

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

The neural-state film has a separate stimulus and score contract. Place the three pinned Pexels source clips at `data/natural_manifold/sources/{woodland,ocean,city}.mp4`; the builder verifies their hashes and downloads the public-domain apple and selected public-domain Voyager Golden Record files when needed:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_neural_film_sources.py

PYTHONPATH=src .venv/bin/python -m axosim_demo natural-manifold run \
  --config configs/natural_manifold_spatial.json \
  --root data/flyvis --output data/natural_manifold/result-2min --device cuda

PYTHONPATH=src .venv/bin/python -m axosim_demo natural-manifold render \
  --config configs/natural_manifold_spatial.json \
  --recording data/natural_manifold/result-2min/manifold-recording.npz \
  --summary data/natural_manifold/result-2min/summary.json \
  --output data/natural_manifold/rendered-2min
```

Run and render the full-state perturbation test:

```bash
PYTHONPATH=src .venv/bin/python -m axosim_demo limit-cycle run \
  --config configs/limit_cycle_return.json \
  --root data/flyvis \
  --output data/natural_manifold/limit-cycle \
  --device cuda

PYTHONPATH=src .venv/bin/python -m axosim_demo limit-cycle render \
  --summary data/natural_manifold/limit-cycle/limit-cycle-summary.json \
  --distances data/natural_manifold/limit-cycle/limit-cycle-distances.npz \
  --output data/natural_manifold/limit-cycle/figures
```

The experiment contract is [`configs/natural_geometry.json`](configs/natural_geometry.json). It pins the AxoSim checkpoint SHA-256 (`19a045bf5b62ca92ab547934ca031e31f464ff2421f0d07b60d8278189889626`), the expanded graph hash, sampling seed, split, stimulus, fitting procedure, and display settings. Image manifests preserve source IDs, hashes, URLs, licenses and attributions.

## Supporting scientific work

- The same substituted circuit reached mean held-out T4/T5 moving-edge peak-tuning correlation 0.724 versus 0.079 after a within-type retinotopy scramble. T4c preferred-direction error was 1.97°, while its direction-selectivity magnitude remained weak at 0.044 versus the 0.627 FlyVis reference.
- Exact whole-connectome routing advances 127,400 AxoSim states while retaining 14,687,178 pair edges and 52,793,639 anatomical contacts at about 0.70 GiB peak CUDA allocation. Its first uniform dendritic-route hypothesis did not reproduce published grooming specificity.
- A dopamine-gated mushroom-body experiment retained all 162,517 released neurons and adapted only anatomical KC→MBON edges. Its preregistered reversal gate failed, so it remains a negative result rather than a demo.

The checkpoint's mammalian training domain limits biological interpretation, but it does not block the experiment. A body video remains out of scope until a validated neural-to-muscle chain generates movement without an inserted controller.

See [upstream attribution](UPSTREAM.md) and the private companion research record in `axosim-demo-internal`.
