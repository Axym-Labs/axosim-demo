# Upstream provenance

This repository is a GitHub fork of [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym), the NeuroMechFly MuJoCo environment.

- Fork base: `38c8ec61034cd59bc5ba0de20688d4a3c0000d60` (FlyGym 2.1.0 development head, inspected 2026-09-29).
- Original Apache-2.0 license and source attribution are retained.
- Working starting demonstrations: `tutorials/4d_turning_controller.ipynb`, `flygym_demo/complex_terrain`, and `scripts/replay_behavior_cpu.py`.
- AxoSim additions live in `src/axosim_demo/` and `tests/test_axosim_*.py`.
- AxoSim is installed separately from [DavideWiest/axosim](https://github.com/DavideWiest/axosim).
- FlyWire/Shiu graph inputs are downloaded separately, with upstream commit and SHA-256 recorded. No threshold pruning is applied by the demo importer. A weighted neuron-pair edge is not an anatomical synapse; the original release has already aggregated anatomical contacts.

The upstream body controller supplies the leg motor program. Neural outputs supply descending commands. This interface does not recover the full biological ventral nerve cord.
