> RETIRED ENGINEERING PROTOTYPE. The commands and clips below do not meet the scientific demo requirement. The old top-level recording CLI has been removed. Retained for provenance only.

# AxoSim embodied fly demo

An articulated fruit fly in MuJoCo, with an AxoSim neural backend, multiview videos, and reproducible behavioral experiments. This is a fork of [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym); [upstream provenance](UPSTREAM.md) and Apache-2.0 attribution are retained.

The first implementation contains three distinct experiments:

- **Embodied adaptation:** a small, trained AxoSim-Lite motif learns odor–reward association through dopamine-gated contact efficacy updates. Its outputs influence a walking fly. Separate third-person and first-person videos show native neural traces beside a photographed tabletop and scanned fruit.
- **Original grooming-circuit baseline:** unmodified Shiu Brian2 code runs all 127,400 neurons and 14,687,178 released weighted edges. Three seeds reproduce stronger aBN1 activation from JO-C/E than JO-F stimulation.
- **Exact graph audit:** all released edges are imported without pruning and routed on GPU. The audit compares against a CPU reference and reports memory, including the cost of rectangular padding.

**The current neural video uses four surrogate evaluations, not the full connectome.** The pretrained neuronal surrogate was trained on mammalian morphologies. Its use here is an engineering starting point, not validated fly physiology. The inherited controller supplies leg movements; navigation uses synthetic odor concentrations and known source bearings. A separate grooming video replays measured foreleg kinematics. Full-connectome AxoSim coupling and neurally generated grooming remain implementation milestones.

## Install

Use Python 3.12 on Linux for the tested headless configuration:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-demo.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

The dependency lock pins the AxoSim sparse-event improvement. FlyGym's upstream `uv.lock` is retained for its own examples; use `requirements-demo.lock` for these experiments.

## Record a working body demo

```bash
MUJOCO_GL=egl .venv/bin/python -m axosim_demo \
  --backend body --duration 1.5 --output demo-runs/body
```

This produces `demo.mp4`, `preview.png`, `metadata.json`, and `telemetry.npz`. The 1280×720 video plays 1.5 simulated seconds over six seconds at 0.25× speed. The first-person perspective is cinematic, not a compound-eye neural sensor.

Use `--view fpp` for a separate primary first-person video. `--num-flies 3` creates three articulated flies with independent controller state in one physics scene. The neural panel reports the first fly; social behavior and explicit fly–fly contacts are not modeled. `--raw-video` also exports the camera without overlays. Scene asset licenses, original hashes and physical scale are recorded in [sources.json](src/axosim_demo/assets/natural_scene/sources.json). MuJoCo renders photographic diffuse maps; this is not a path-traced environment.

## Prepare graph and neural checkpoint

```bash
.venv/bin/python scripts/prepare_axosim_demo.py
```

Graph and original reference code download from an immutable Shiu commit and are checked against SHA-256 digests. Inputs remain in ignored `data/`.

The AxoSim video additionally requires the evaluated Lite checkpoint. It is retained with the run evidence in the private companion repository, `01-embodied-fly/artifacts/neural/axosim-lite-t24-s8.pt`; it is not bundled in this public fork. Stage it explicitly:

```bash
.venv/bin/python scripts/prepare_axosim_demo.py \
  --checkpoint-source /path/to/axosim-lite-t24-s8.pt
```

Expected SHA-256: `19a045bf5b62ca92ab547934ca031e31f464ff2421f0d07b60d8278189889626`. A missing or different checkpoint fails explicitly; there is no random-model fallback.

## Condition and record the AxoSim fly

```bash
.venv/bin/python -m axosim_demo.conditioning \
  --seeds 3 --output demo-runs/conditioning.json
MUJOCO_GL=egl .venv/bin/python -m axosim_demo \
  --conditioning demo-runs/conditioning.json --arm paired \
  --seed 0 --rewarded-cue 0 --output demo-runs/conditioned
```

Repeat with `--arm frozen`, `unpaired`, or `dopamine_blocked`; use `--rewarded-cue 1` to test reversed pairing. Training uses imposed KC activity and a reward-to-DAN pulse abstraction, with a causal eligibility trace acting on actual AxoSim contact efficacies. Shared neuronal weights stay frozen. Test videos contain no reward or plasticity. This is a mechanism demonstration, not a competitive biological-learning score.

## Compare LIF and AxoSim

```bash
MUJOCO_GL=egl .venv/bin/python -m axosim_demo --backend lif \
  --conditioning demo-runs/conditioning.json --duration 2 --raw-video \
  --output demo-runs/lif
MUJOCO_GL=egl .venv/bin/python -m axosim_demo --backend axosim \
  --conditioning demo-runs/conditioning.json --duration 2 --raw-video \
  --output demo-runs/axosim
.venv/bin/python -m axosim_demo.comparison --lif demo-runs/lif \
  --axosim demo-runs/axosim --output demo-runs/comparison
```

The video places LIF left and AxoSim right. It holds the reduced circuit, initial scene, sensory fields, contact efficacy changes, decoder and gait fixed. Closed-loop inputs diverge as trajectories diverge. The LIF adapter uses Shiu's equations/constants, with delay, refractory behavior and reset checked against Brian2. It is a substitution into this synthetic motif, not the original full-brain simulation. Different walking trajectories alone do not establish superior fly fidelity.

Native soma outputs and recurrent-state diagnostics are actual model values, not calibrated fly firing rates. The first four milliseconds are forecast padding and are masked. The models preserve their recurrent state throughout each video. The generic sparse-memory patch changes allocation order; the current small video circuit uses a separate dense contact path.

A separate physiological benchmark is available through `python -m axosim_demo.neuron_fidelity --data <local-neuron-traces> --output <results>`. On four retrospective mammalian validation traces, frozen Lite achieved 2.985 mV clipped-voltage RMSE versus 3.299 and 3.767 mV for train-fitted voltage-only and joint-objective LIF models. Pooled spike F1 within 5 ms was 0.476 versus 0 and 0.342. This small panel uses a different, calibrated LIF from the embodied substitution; it does not validate fly behavior or isolate architecture from pretraining. The private report retains all traces, contrary cases, fitting budgets and dataset-overlap uncertainty.

## Replay recorded grooming

```bash
MUJOCO_GL=egl .venv/bin/python -m axosim_demo.grooming \
  --output demo-runs/grooming
```

This downloads a pinned, hash-checked SeqIKPy recording and renders 14 measured foreleg angle channels. It uses forward kinematics without a neural controller or physical contact simulation; the caption identifies this explicitly. Unverified head/antenna synchronization is not assumed. It provides a body-motion reference for future neural grooming experiments.

## Reproduce the original grooming-circuit baseline

```bash
.venv/bin/python -m axosim_demo.shiu_benchmark \
  --seeds 0 1 2 --output demo-runs/shiu
```

The pinned protocol preserves original neuron equations, weights, delays, integration, and every graph edge. At 100 Hz stimulation, measured aBN1 rates were 26/28/25 Hz for JO-C/E and zero for JO-F. Descending aDN1 recruitment was only 0/1/0 Hz at this setting, so this is not an embodied grooming reproduction. The paper used 30 repeats and broader sweeps; three seeds are an initial replication.

## Audit connectivity memory

```bash
.venv/bin/python -m axosim_demo.audit --download --device cuda \
  --output demo-runs/memory-audit.json
```

On the RTX 5090, the full graph routing audit peaked at 353,513,472 allocated bytes and matched the CPU sum exactly on its binary test input. That is routing only; it does not measure a complete adaptive brain/body run. The importer preserves integer root IDs, signed counts, duplicate rows, and independent efficacy per released row. A released weighted edge already aggregates anatomical contacts; no contact-specific locations are invented.

## Verify

```bash
uv pip install --python .venv/bin/python pytest pytest-cov build setuptools wheel
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_axosim_*.py tests/test_neuron_fidelity.py -q -o addopts=''
MUJOCO_GL=egl .venv/bin/python -m pytest \
  --ignore=tests/warp -m 'not warp and not rl and not tutorial and not network' \
  -q -o addopts=''
```

The second command covers the inherited CPU/body implementation. Warp, RL, notebook, and live-network suites require their optional environments. Detailed research, original prompt/specification, raw measurements, videos, and provenance are in the private [axosim-demo-internal](https://github.com/DavideWiest/axosim-demo-internal) companion repository.

## Scientific sources

- [Shiu et al. 2024: computational brain model and original code](https://github.com/philshiu/Drosophila_brain_model)
- [NeuroMechFly v2](https://www.nature.com/articles/s41592-024-02497-y)
- [Özdil et al. 2026: antennal grooming coordination](https://www.nature.com/articles/s41467-026-72152-x)
- [Eon's technical description of the viral demonstration](https://eon.systems/updates/embodied-brain-emulation)
- [Hige et al. 2015: dopamine-dependent mushroom-body plasticity](https://www.sciencedirect.com/science/article/pii/S0896627315009824)

The original FlyGym documentation remains in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).
