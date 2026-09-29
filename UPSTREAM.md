# Upstream provenance

This repository is a GitHub fork of [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym), the NeuroMechFly MuJoCo environment.

- Fork base: `38c8ec61034cd59bc5ba0de20688d4a3c0000d60` (FlyGym 2.1.0 development head, inspected 2026-09-29).
- Original Apache-2.0 license and source attribution are retained.
- Working starting demonstrations: `tutorials/4d_turning_controller.ipynb`, `flygym_demo/complex_terrain`, and `scripts/replay_behavior_cpu.py`.
- AxoSim additions live in `src/axosim_demo/` and `tests/test_axosim_*.py`.
- AxoSim is installed separately from [Axym-Labs/axosim](https://github.com/Axym-Labs/axosim).
- FlyWire/Shiu graph inputs are downloaded separately, with upstream commit and SHA-256 recorded. No threshold pruning is applied by the demo importer. A weighted neuron-pair edge is not an anatomical synapse; the original release has already aggregated anatomical contacts.

The upstream body controller supplies the leg motor program. Neural outputs supply descending commands. This interface does not recover the full biological ventral nerve cord.

Additional revision sources:

- Natural-scene photographs and scanned apple: [Poly Haven](https://polyhaven.com), CC0. Original authors, source hashes and transformations are retained in [asset provenance](src/axosim_demo/assets/natural_scene/NOTICE.md).
- Measured foreleg trajectories: [SeqIKPy](https://github.com/NeLy-EPFL/sequential-inverse-kinematics), Apache-2.0 repository, pinned at `f7f1dc9b09ce89c4f54b2005722d62f887623a7b`. The sample is downloaded separately and hash checked; no separate data license was found. Replay does not run a neural model.
- Embodied LIF equations and constants: [Shiu brain model](https://github.com/philshiu/Drosophila_brain_model), MIT, pinned at `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`. The small adapter is checked against Brian2; the original full-network experiment uses separately downloaded upstream code.
