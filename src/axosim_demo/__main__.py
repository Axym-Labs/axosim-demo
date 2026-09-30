"""Scientific evidence and visual infrastructure for a future AxoSim fly model.

The earlier engineered steering/CPG recordings are retired engineering prototypes,
not scientific AxoSim fly demonstrations. No implicit body-controller fallback.
"""
import argparse
import importlib
import sys

COMMANDS={
    'neuron-fidelity':'axosim_demo.neuron_fidelity',
    'connectome-audit':'axosim_demo.audit',
    'shiu-benchmark':'axosim_demo.shiu_benchmark',
    'whole-brain':'axosim_demo.whole_brain',
    'vision-reference':'axosim_demo.vision_reference',
    'vision-axosim':'axosim_demo.vision_axosim',
    'vision-video':'axosim_demo.vision_video',
    'natural-vision':'axosim_demo.natural_vision',
    'natural-manifold':'axosim_demo.natural_manifold',
    'whole-brain-video':'axosim_demo.whole_brain_video',
    'mb-conditioning':'axosim_demo.mb_conditioning',
    'mb-conditioning-video':'axosim_demo.mb_conditioning_video',
    'scene-preview':'axosim_demo.scene_preview',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=[*COMMANDS,'fly-demo'],nargs='?')
    args=parser.parse_args(sys.argv[1:2])
    remaining=sys.argv[2:]
    if args.command is None:
        parser.print_help()
        return
    if args.command=='fly-demo':
        parser.error('Scientific fly embodiment is not implemented: a fly-calibrated neural model and validated neuron-to-muscle/sensory coupling are required. Engineered steering, CPG, RL motor policies and motion replay are not accepted substitutes.')
    module=importlib.import_module(COMMANDS[args.command])
    sys.argv=[f'axosim_demo {args.command}',*remaining]
    module.main()


if __name__ == '__main__':
    main()
