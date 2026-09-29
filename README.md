# AxoSim embodied fly demo

An articulated fruit fly in MuJoCo, with an AxoSim neural backend, multiview videos, and reproducible behavioral experiments. This is a fork of [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym); [upstream provenance](UPSTREAM.md) and Apache-2.0 attribution are retained.

The first implementation contains three distinct experiments:

- **Embodied adaptation:** a small, trained AxoSim-Lite motif learns odor–reward association through dopamine-gated contact efficacy updates. Its outputs influence a walking fly. Videos combine third-person body view, a thorax-mounted first-person perspective, and measured diagnostics.
- **Original grooming-circuit baseline:** unmodified Shiu Brian2 code runs all 127,400 neurons and 14,687,178 released weighted edges. Three seeds reproduce stronger aBN1 activation from JO-C/E than JO-F stimulation.
- **Exact graph audit:** all released edges are imported without pruning and routed on GPU. The audit compares against a CPU reference and reports memory, including the cost of rectangular padding.

**The current video uses the reduced motif, not the full connectome.** The pretrained neuronal surrogate was trained on mammalian morphologies. Its use here is an engineering starting point, not validated fly physiology. The inherited controller supplies leg movements; navigation uses synthetic odor concentrations and known source bearings. Full-connectome AxoSim coupling and embodied foreleg grooming are the next implementation milestones.

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
uv pip install --python .venv/bin/python pytest pytest-cov
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_axosim_*.py -q -o addopts=''
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
