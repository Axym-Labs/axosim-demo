"""Measured foreleg grooming replay on NeuroMechFly; no neural motor claim.

The 200-frame 100 Hz SeqIKPy tutorial clip is recorded animal motion inferred
through pose estimation and inverse kinematics. We copy its 14 foreleg angles,
with no learned/synthetic motion, and hold unrecorded body parts in neutral pose.
MuJoCo forward kinematics renders the replay; physics dynamics are not stepped.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import urllib.request

import numpy as np

SOURCE_COMMIT = 'f7f1dc9b09ce89c4f54b2005722d62f887623a7b'
SOURCE_PATH = 'data/anipose_220807_Fly002_002/leg_joint_angles.pkl'
SOURCE_URL = f'https://raw.githubusercontent.com/NeLy-EPFL/sequential-inverse-kinematics/{SOURCE_COMMIT}/{SOURCE_PATH}'
SOURCE_SHA256 = 'bb496006aec5967c8644d914a9de0fef521ee42bef25afc458291e1c744b85a0'
SOURCE_FPS = 100


def source_key(dof):
    """Translate anatomical joint labels without changing angular signs/order."""
    child = dof.child
    if child.pos not in ('lf', 'rf'):
        return None
    shorthand = {'coxa': 'ThC', 'trochanterfemur': 'CTr', 'tibia': 'FTi', 'tarsus1': 'TiTa'}
    if child.link not in shorthand:
        return None
    return f'Angle_{child.pos.upper()}_{shorthand[child.link]}_{dof.axis.value}'


def load_trajectory(path):
    """Validate pinned bytes before reading this trusted upstream NumPy pickle."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Grooming source SHA256 mismatch; refusing to unpickle')
    data = pickle.loads(raw)
    if len(data) != 14 or any(np.asarray(v).shape != (200,) or not np.isfinite(v).all() for v in data.values()):
        raise ValueError('Expected 14 finite, synchronized 200-frame foreleg angles')
    return data


def sample_angles(data, time_s, source_fps=SOURCE_FPS):
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1 or not lengths or source_fps <= 0:
        raise ValueError('Trajectory must contain equal-length arrays and positive fps')
    last = (next(iter(lengths))-1)/source_fps
    if not np.isfinite(time_s) or time_s < 0 or time_s > last + 1e-12:
        raise ValueError('Replay time outside recorded range; no looping or extrapolation')
    frame = min(time_s * source_fps, next(iter(lengths))-1)
    lo, hi = int(np.floor(frame)), int(np.ceil(frame))
    return {k: float(v[lo] * (1-(frame-lo)) + v[hi] * (frame-lo)) for k, v in data.items()}


def record_grooming(output, source_file='data/grooming/leg_joint_angles.pkl', fps=30,
                    playback_speed=.25, camera=(6.5,-3.5,2.5), preview_only=False):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image
    from flygym import Simulation
    from flygym.anatomy import Skeleton, AxisOrder, JointPreset
    from flygym.compose import NeuroMechFly, KinematicPosePreset, TetheredWorld
    from flygym.utils.math import Rotation3D
    from axosim_demo.scene import decorate_world
    from axosim_demo.video import compose_frame

    output, source_file = Path(output), Path(source_file)
    output.mkdir(parents=True, exist_ok=True)
    if not source_file.exists():
        source_file.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            source_file.write_bytes(response.read())
    data = load_trajectory(source_file)
    fly = NeuroMechFly(name='grooming_fly')
    fly.add_joints(Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY),
                   neutral_pose=KinematicPosePreset.NEUTRAL)
    fly.colorize()
    # Camera axes look at the animal's front from a slight side elevation.
    pos = np.asarray(camera, dtype=float)
    z = pos / np.linalg.norm(pos)
    x = np.cross([0,0,1], z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    tracking = fly.add_tracking_camera(name='grooming_tpp', pos_offset=pos,
                                       rotation=Rotation3D('xyaxes', tuple(np.r_[x,y])), fovy=33)
    world = decorate_world(TetheredWorld())
    world.add_fly(fly, [0,0,.85], Rotation3D('quat', [1,0,0,0]))
    sim = Simulation(world)
    sim.reset()
    sim.mj_model.vis.global_.offwidth = 1280
    sim.mj_model.vis.global_.offheight = 720
    sim.mj_model.vis.quality.shadowsize = 2048
    sim.mj_model.vis.headlight.ambient = (.5, .5, .5)
    sim.mj_model.vis.headlight.diffuse = (.6, .6, .6)
    mappings = []
    for dof, joint in fly.jointdof_to_mjcfjoint.items():
        key = source_key(dof)
        if key in data:
            jid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
            mappings.append((key, dof.name, int(sim.mj_model.jnt_qposadr[jid])))
    if len(mappings) != 14:
        raise ValueError(f'Expected all 14 measured foreleg DoFs, found {len(mappings)}')
    duration = (len(next(iter(data.values())))-1)/SOURCE_FPS
    times = np.arange(0, duration+1e-10, playback_speed/fps)
    if preview_only:
        times = np.array([0,.4,.8,1.2,1.6,1.99])
    renderer = mujoco.Renderer(sim.mj_model, height=720, width=1280)
    writer = None if preview_only else imageio.get_writer(output/'grooming_tpp.mp4', fps=fps,
                    codec='libx264', quality=9, macro_block_size=1, ffmpeg_log_level='error')
    measured = []
    try:
        for i, time_s in enumerate(times):
            angles = sample_angles(data, float(time_s))
            for key, _, address in mappings:
                sim.mj_data.qpos[address] = angles[key]
            sim.mj_data.time = time_s
            mujoco.mj_forward(sim.mj_model, sim.mj_data)
            renderer.update_scene(sim.mj_data, camera=tracking.name)
            frame = Image.fromarray(compose_frame(
                renderer.render(), None,
                label='Recorded grooming · measured kinematic replay · no neural motor generation',
                time_s=float(time_s), playback_speed=playback_speed))
            if writer:
                writer.append_data(np.asarray(frame))
            if preview_only or i in (0, len(times)//2, len(times)-1):
                frame.save(output/f'frame_{i:04d}.png')
            measured.append([angles[key] for key,_,_ in mappings])
    finally:
        if writer:
            writer.close()
        renderer.close();sim.close()
    np.savez_compressed(output/'replayed_angles.npz', time_s=times, angles_rad=np.array(measured),
                        source_keys=np.array([key for key,_,_ in mappings]))
    metadata = {'source_url':SOURCE_URL,'source_sha256':SOURCE_SHA256,'source_commit':SOURCE_COMMIT,
        'source_repository':'NeLy-EPFL/sequential-inverse-kinematics','source_license':'Apache-2.0 repository license; no separate sample-data license found',
        'source_recording':'anipose_220807_Fly002_002; upstream tutorial extracts frames400:600',
        'source_frames':200,'source_fps':SOURCE_FPS,'duration_source_s':duration,
        'rendered_frames':len(times),'output_fps':fps,'playback_speed':playback_speed,
        'camera_offset_mm':list(camera),'resolution':[1280,720],
        'mapping':[{'source_key':k,'joint_dof':d} for k,d,_ in mappings],
        'interpolation':'Linear; no loop, extrapolation, sign changes, amplitude changes, or smoothing',
        'scope':'14 recorded foreleg angles rendered through MuJoCo forward kinematics. Root/body and unrecorded parts remain fixed in neutral pose. No dynamics/controller/brain simulation.',
        'limitations':['Head sample in same folder has250 frames vs forelegs200; not replayed because temporal alignment is unverified',
                        'Axis order fixed to measured yaw-pitch-roll; transfer to current NMF geometry is not a contact-force validation',
                        'Actual source recording and matched physical sensorimotor physiology are not reconstructed'],
        'neural_diagnostics':'None: no neural model is run in this replay', 'preview_only':preview_only}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-file',default='data/grooming/leg_joint_angles.pkl')
    parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args()
    record_grooming(args.output,args.source_file,preview_only=args.preview_only)


if __name__=='__main__':
    main()
