# AxoSim scientific fly demonstration — in development

This project forks [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym) to investigate a scientific AxoSim fly simulation. **No controller-free, scientifically validated fly-behavior demo is implemented yet.**

The earlier walking videos used an engineered odor-to-steering rule and a hybrid CPG controller. The proposed flight controller used an upstream RL policy, and grooming was recorded-motion replay. These do not meet the project's scientific requirement and are retired from the demo path. [Archived engineering notes](docs/ENGINEERING_PROTOTYPE.md) retain provenance; they are not scientific results. `python -m axosim_demo fly-demo` rejects the unsupported path explicitly rather than falling back to a controller.

An acceptable embodied demonstration must use AxoSim for the neural dynamics and connect measured sensory physiology, the appropriate neural graph, motor-neuron outputs, and a defensible neuromuscular/body model. An engineered steering rule, gait oscillator, behavior selector, trajectory follower, RL motor layer or replay cannot supply the behavior being claimed.

There is a concrete direct-coupling route to test. [Pugliese et al.'s released VNC simulation](https://github.com/smpuglie/Pugliese_2026) produces rhythms in anatomically identified leg motor neurons from its recurrent fly circuit. The [FANC atlas](https://pmc.ncbi.nlm.nih.gov/articles/PMC11348827/) identifies front-leg muscle targets, [Azevedo et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7347388/) measured adult motor-unit spikes and force, and the fork already contains [FlyMimic's](https://github.com/gizemozd/FlyMimic) anatomical foreleg muscle plant. Joining and validating those links is active work. The mammalian-trained checkpoint is an out-of-domain starting hypothesis, not a reason to stop implementation.

## What currently has evidence

- **Numerical AxoSim integration and sparse equivalence.** Persistent trained-Lite execution, morphology adaptation, contact efficacies and causal forecast alignment are checked against reference interfaces. The sparse change preserves contact operations without pruning; it does not alter the separate fused path underlying technical-report Figure 5.
- **Retrospective neuronal comparison.** On four existing mammalian NEURON validation traces, frozen Lite obtained 2.985 mV clipped-voltage RMSE versus 3.299/3.767 mV for train-fitted LIF baselines, and spike F1 within 5 ms of 0.476 versus 0/0.342. This is a small retrospective panel; prior checkpoint exposure is unverified. It establishes neither fly physiology nor architecture-only superiority.
- **Original Shiu neural benchmark.** Unmodified Brian2 code simulates all 127,400 released neurons and 14,687,178 weighted edges. Three seeds reproduce stronger aBN1 activation from JO-C/E than JO-F stimulation. This is the original LIF model, not an AxoSim fly model.
- **Exact connectome routing audit.** Released edges are retained, integer identities checked, and GPU routing compared against a CPU reference. This does not measure a complete adaptive brain/body run.
- **Experimental whole-brain AxoSim runtime.** The pretrained checkpoint now advances one causal AxoSim state for each of 127,400 v630 neurons while a flat GPU router retains all 14,687,178 pair edges and 52,793,639 anatomical contacts. It uses no fitted network gain and exposes the missing dendritic-contact map as a uniform route-marginalization hypothesis. This first hypothesis does not reproduce the published grooming or feeding specificity robustly across checkpoint morphologies, so its traces are retained as negative scientific results and are not rendered as a success demo.
- **Visual infrastructure.** Imported woodland geometry, static anatomical previews, full-frame composition, the Axym website's Inter font, actual TikZ/PGFPlots overlays, and camera framing that reserves space for the overlay. Static scene previews demonstrate assets, not behavior.

## Scientific evidence-film series

The supported films now deliberately stop at measured neural endpoints. Every AxoSim-labelled neural trace is produced by the frozen checkpoint at runtime; none of these films uses gait control, steering, a body trajectory, an RL policy, or recorded motion.

1. **Controlled visual circuit.** AxoSim replaces the neuron dynamics on the exact expanded FlyVis graph (45,669 cells, 1,513,231 edges). Selection uses alternating moving-edge directions; the film uses a held-out 30° stimulus. Mean held-out T4/T5 peak-tuning correlation is 0.724 versus 0.079 after a within-cell-type retinotopy scramble, and T4c preferred-direction error is 1.97°. T4c direction-selectivity magnitude remains weak (0.044 versus the 0.627 reference), so this supports tuning shape and graph dependence rather than calibrated fly physiology.
2. **Natural woodland vision.** The frozen selected visual circuit processes an independently sourced, forward-moving woodland film. A predeclared dynamic-versus-static temporal-variance test fails (ratio 0.740). A separately labeled exploratory spatial-content control passes: natural spatial structure produces 2.50× the T4c effect of a matched uniform-luminance movie. Both outcomes are retained.
3. **Real-scenes neuron manifold.** The same frozen visual circuit processes woodland, ocean, and city footage after a fixed per-frame mean/contrast transform. A fixed 3D UMAP places 2,884 L1/Mi1/T4c/T5c neurons by their response fingerprints and illuminates their measured AxoSim state over time. Held-out temporal blocks separate the three clips at 100% balanced accuracy; a constant-gray control is at chance (33.3%), and 15-neighbor embedding trustworthiness is 0.989. This is a within-clip neural representation result, not recognition of unseen scenes.
4. **Cellular AxoSim/LIF comparison.** A matched side-by-side film uses the identical retrospective mammalian NEURON target in both panels. Across four traces, AxoSim-Lite has lower clipped-voltage RMSE (2.985 mV versus 3.299/3.767 mV for two fitted LIF variants), while the first displayed trace retains AxoSim's slightly lower spike F1. It is a small retrospective cellular check, not fly validation.
5. **Dopamine-gated mushroom-body conditioning.** The frozen experiment instantiates the full 162,517-neuron `flybrain` graph, retains all 10,861,252 released pair rows, and permits adaptation only on 33,496 anatomical KC→MBON pair edges. PPL105/PAM08 reinforcement reaches those efficacies through the released DAN→MBON contact projection. Standard, reversed, no-plasticity, and DAN-clamp-off arms use eight independently trained replicates. The predeclared reversal gate failed: standard and reversed effects were both near zero with intervals spanning zero, and only one of eight paired replicates changed in opposite directions. The video presents this as a negative result.

All films use a 1280×720 white/full-frame composition, exact title `AxoSim - Axym Labs`, Inter, and PGFPlots/TikZ scientific plots. Static diagrams disclose when geometry is schematic rather than anatomical.

The exact 127,400-neuron v630 scale run remains a secondary engineering artifact rather than a featured film. It advances one AxoSim state per neuron and routes all 14,687,178 pair edges and 52,793,639 anatomical contacts at about 0.70 GiB peak CUDA allocation, but its index-grid view was visually weak and its aBN1 grooming-specificity result was negative.

## Install and run the supported checks

Use Python 3.12 on Linux:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-demo.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m axosim_demo --help
```

The dependency lock pins the original sparse-event fix. The required trained checkpoint is retained in the private companion repository and is not bundled publicly. There is no random-weight fallback:

```bash
.venv/bin/python scripts/prepare_axosim_demo.py \
  --checkpoint-source /path/to/axosim-lite-t24-s8.pt
.venv/bin/python -m axosim_demo neuron-fidelity \
  --data /path/to/retrospective-neuron-traces --output /path/to/results
.venv/bin/python -m axosim_demo shiu-benchmark \
  --seeds 0 1 2 --output /path/to/shiu-results
.venv/bin/python -m axosim_demo connectome-audit \
  --download --device cuda --output /path/to/memory-audit.json
.venv/bin/python -m axosim_demo whole-brain \
  --config configs/shiu_grooming.json --seeds 0 1 2 \
  --device cuda --output /path/to/axosim-whole-brain
.venv/bin/python scripts/prepare_mb_connectome.py
.venv/bin/python -m axosim_demo mb-conditioning \
  --config configs/mb_conditioning.json --device cuda \
  --output /path/to/conditioning-results
.venv/bin/python -m axosim_demo mb-conditioning-video \
  --summary /path/to/conditioning-results/summary.json \
  --output /path/to/conditioning-video
.venv/bin/python -m axosim_demo natural-manifold run \
  --config configs/natural_manifold_spatial.json \
  --root /path/to/flyvis-data \
  --output /path/to/manifold-results --device cuda
.venv/bin/python -m axosim_demo natural-manifold render \
  --config configs/natural_manifold_spatial.json \
  --recording /path/to/manifold-results/manifold-recording.npz \
  --summary /path/to/manifold-results/summary.json \
  --output /path/to/manifold-video
```

Checkpoint SHA-256: `19a045bf5b62ca92ab547934ca031e31f464ff2421f0d07b60d8278189889626`. It is a **mammalian-trained** Lite model; no validated fly checkpoint was found. The current 1-ms/P4 neural contract also cannot silently stand in for the original Shiu model's 0.1-ms integration and fractional-ms delay/refractory values.

Static woodland infrastructure can be inspected without running a behavior controller:

```bash
MUJOCO_GL=egl .venv/bin/python -m axosim_demo scene-preview --output /path/to/static-preview
```

The bundled CC0 scene has seven imported pine trees, ferns, bark, a stump and mossy rock. Its visual props have no contact forces, the physical ground remains flat, and native MuJoCo rendering is not photorealistic. The clean stills have no simulated activity or plot overlay. Source hashes, licenses, conversions and visual geometry reductions are recorded in `assets/forest_scene/`.

The moving-edge visual benchmark has a separate pinned reference extra. It reproduces the official FlyVis metrics and is explicitly a non-AxoSim comparison arm:

```bash
uv pip install --python .venv/bin/python -e '.[vision-reference]'
flyvis download-pretrained
.venv/bin/python -m axosim_demo vision-reference \
  --root /path/to/flyvis-data --output /path/to/vision-reference
.venv/bin/python -m axosim_demo vision-axosim \
  --root /path/to/flyvis-data --output /path/to/vision-axosim
```

TikZ overlays require `tectonic` and Poppler's `pdftocairo`. `axosim_demo.tikz_plot` compiles axes and traces separately, preserving vector exports and revealing recorded samples progressively. The compositor keeps full-frame footage, uses a translucent frosted backdrop, and shows only `AxoSim - Axym Labs` as the title. Inter's OFL and source pin are included with its font asset. Plot labels identify native outputs; they do not invent calibrated fly firing rates.

## Tests and research record

```bash
uv pip install --python .venv/bin/python pytest pytest-cov build setuptools wheel
MUJOCO_GL=egl .venv/bin/python -m pytest \
  tests/test_axosim_*.py tests/test_neuron_fidelity.py \
  tests/test_mb_conditioning.py tests/test_tikz_plot.py \
  tests/test_scene_overlay.py -q -o addopts=''
```

Engineering-body tests verify infrastructure only. They do not turn its retired controller into a scientific model. Optional Warp/RL/tutorial/network suites require their own environments.

The private [axosim-demo-internal](https://github.com/DavideWiest/axosim-demo-internal) repository contains the original prompt, subsequent corrections, task specification, literature/code audits, raw measurements, and clearly categorized artifacts. The current scientific boundary supersedes earlier video descriptions. See [upstream attribution](UPSTREAM.md) and [original FlyGym documentation](docs/UPSTREAM_README.md).
