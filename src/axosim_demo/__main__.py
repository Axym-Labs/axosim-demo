"""Record the embodied fly with a body baseline or conditioned AxoSim backend."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=1.5, help='Simulated seconds; footage is 0.25x')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--backend', choices=['body', 'axosim'], default='axosim')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--conditioning', type=Path, help='JSON from axosim_demo.conditioning')
    parser.add_argument('--rewarded-cue', type=int, choices=[0,1], default=0)
    parser.add_argument('--arm', choices=['paired','unpaired','frozen','dopamine_blocked'], default='paired')
    args = parser.parse_args()
    if args.backend == 'body' and args.conditioning:
        parser.error('--conditioning requires --backend axosim')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from .body import record_demo
    controller = None
    if args.backend == 'axosim':
        from .neural import DEFAULT_CHECKPOINT, OdorNavigationController
        checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
        if not checkpoint.exists():
            parser.error('AxoSim-Lite checkpoint required; pass --checkpoint or use --backend body. See README.')
        log_efficacy = None
        selected = None
        if args.conditioning:
            runs = json.loads(args.conditioning.read_text())['runs']
            selected = next((r for r in runs if r['seed'] == args.seed and r['rewarded_cue'] == args.rewarded_cue), None)
            if selected is None:
                parser.error('No conditioning run matches seed and rewarded cue')
            log_efficacy = selected['arms'][args.arm]['log_efficacy']
        controller = OdorNavigationController(log_efficacy, checkpoint=checkpoint)
        controller.metadata.update({'conditioning_arm': args.arm if selected else None,
                                    'conditioning_seed': args.seed if selected else None,
                                    'rewarded_cue': args.rewarded_cue if selected else None,
                                    'conditioned_weights_loaded': selected is not None})
    paths = record_demo(args.output, duration=args.duration, seed=args.seed, controller=controller)
    print(json.dumps({k:str(v) for k,v in paths.items()},indent=2))


if __name__ == '__main__':
    main()
