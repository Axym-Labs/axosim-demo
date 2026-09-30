# AxoSim scientific fly demonstration

This project forks [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym) and tests where a pretrained AxoSim model can replace neuron dynamics in released fly circuits without adding a gait controller, steering rule, behavior state machine, trajectory replay, or arbitrary game-control decoder.

The current public demonstration is a neural film rather than an embodied-behavior claim. A validated AxoSim neuron-to-muscle path is still under development. The checkpoint's mammalian training domain limits biological interpretation, but it does not block testing the model on the fly circuit.

## Current demo: real scenes through an AxoSim fly visual circuit

The film sends licensed woodland, ocean, and city footage through a frozen AxoSim model on the exact expanded FlyVis graph: 45,669 cells and 1,513,231 edges. It records 2,884 L1, Mi1, T4c, and T5c neurons.

The right-side object is a fixed three-dimensional UMAP of population state over time. Every pale point represents all 2,884 recorded neuron values at one 25 ms sample. The three trajectories come from the three source clips. The camera and geometry never rotate. Violet-to-gold shows the causal 1.6-second history ending at the current state; it does not identify cell types. A soft field over the footage shows the strongest T4c/T5c responses at their retinotopic locations.

The visualization begins after one second of model warm-up. Its 180 ms smoothing is causal and affects display coordinates only; it does not change, feed back into, or rerun the AxoSim simulation. The only in-frame text is `AxoSim - Axym Labs`.

The controlled result behind the film is within-clip spatial-content separation:

- held-out temporal-block balanced accuracy: **1.000**
- matched constant-gray control: **0.333**
- difference: **0.667**
- temporal-state UMAP trustworthiness at 15 neighbors: **0.9586**
- PCA-50 variance retained: **0.9644**

This does not establish recognition of unseen scenes, calibrated fly physiology, behavior, or anatomical geometry. It shows that real spatial input produces distinct, reproducible activity states in the frozen AxoSim visual circuit.

The visual language follows two public Goodfire patterns: image-linked activation fields in [Uncovering Neural Geometry in Vision Models](https://www.goodfire.com/research/bsf-vision), and a bright time-local trajectory over persistent geometry in [Meandering on Manifolds](https://www.goodfire.com/research/stories-in-space). Axym's Inter typeface and violet palette are used without dashboard chrome or Matplotlib panels.

## Reproduce

Use Python 3.12 on Linux:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-demo.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

Prepare the required trained checkpoint. There is no random-weight fallback:

```bash
.venv/bin/python scripts/prepare_axosim_demo.py \
  --checkpoint-source /path/to/axosim-lite-t24-s8.pt
```

Run and render the current experiment:

```bash
.venv/bin/python -m axosim_demo natural-manifold run \
  --config configs/natural_manifold_spatial.json \
  --root /path/to/flyvis-data \
  --output /path/to/manifold-results --device cuda

.venv/bin/python -m axosim_demo natural-manifold render \
  --config configs/natural_manifold_spatial.json \
  --recording /path/to/manifold-results/manifold-recording.npz \
  --summary /path/to/manifold-results/summary.json \
  --output /path/to/current-demo
```

The renderer uses a fixed camera, Inter, a full-frame real scene, H.264 CRF 20, and no Matplotlib. Source videos and their SHA-256 hashes are pinned in the config. The checkpoint SHA-256 is `19a045bf5b62ca92ab547934ca031e31f464ff2421f0d07b60d8278189889626`.

## Supporting scientific work

The repository retains non-video experiments that constrain future demonstrations:

- Controlled moving-edge validation on the same graph reached mean held-out T4/T5 peak-tuning correlation 0.724 versus 0.079 after a within-type retinotopy scramble. T4c preferred-direction error was 1.97°, while direction-selectivity magnitude remained weak at 0.044 versus the 0.627 FlyVis reference.
- Exact whole-connectome routing advances 127,400 AxoSim states while retaining 14,687,178 pair edges and 52,793,639 anatomical contacts at about 0.70 GiB peak CUDA allocation. Its first uniform dendritic-route hypothesis did not reproduce published grooming specificity.
- A dopamine-gated mushroom-body experiment retained all 162,517 released neurons and adapted only anatomical KC→MBON edges. Its predeclared reversal gate failed, so it is preserved as a negative result rather than a demo.
- On four retrospective mammalian NEURON traces, frozen AxoSim-Lite had lower clipped-voltage RMSE than two fitted LIF baselines. The small panel and unknown checkpoint exposure prevent a fly-fidelity or architecture-superiority claim.

The direct embodiment route under study joins released VNC circuit-generated motor-neuron rhythms, FANC motor-to-muscle anatomy, measured adult motor-unit force, and a physical foreleg muscle plant. A body video becomes valid only when that neural-to-muscle chain generates the movement.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_axosim_vision.py tests/test_mb_conditioning.py \
  tests/test_axosim_whole_brain.py -q -o addopts=''
```

See [upstream attribution](UPSTREAM.md), the [archived engineering prototype notes](docs/ENGINEERING_PROTOTYPE.md), and the private companion research record in `axosim-demo-internal`.
