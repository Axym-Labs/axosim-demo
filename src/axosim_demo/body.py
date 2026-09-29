"""Actual NeuroMechFly body, inherited hybrid gait, and backend callback bridge.

Based on FlyGym's tutorials/4d_turning_controller.ipynb (Apache-2.0).
The gait controller is retained; the callback supplies bilateral descending drive.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym.utils.mjcf import GEOM_TYPES
from flygym_demo.complex_terrain import (
    HybridTurningController, HybridControllerObservation, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)
from axosim_demo.video import compose_frame

SOURCE_POSITIONS = np.array([[12.0, 5.0, 0.4], [12.0, -5.0, 0.4]])


class FlyBody:
    """Free articulated fly in a landmark arena (millimetres and seconds)."""

    def __init__(self, seed: int = 0):
        self.fly = make_locomotion_fly(name='axosim_fly', colorize=True)
        self.third_person = self.fly.add_tracking_camera(
            name='third_person', pos_offset=(-2.5, -9, 5.5),
            rotation=Rotation3D('xyaxes', (1, -0.24, 0, 0.12, 0.5, 0.86)), fovy=40,
        )
        # A conventional perspective camera attached ahead of the head, not retina.
        self.first_person = self.fly.add_tracking_camera(
            name='first_person', mode='fixed', pos_offset=(1.45, 0, 0.45),
            rotation=Rotation3D('xyaxes', (0, -1, 0, 0.25, 0, 1)), fovy=85,
        )
        world = FlatGroundWorld(half_size=40)
        world.mjcf_root.texture('checker').rgb1 = (0.66, 0.70, 0.63)
        world.mjcf_root.texture('checker').rgb2 = (0.69, 0.73, 0.66)
        world.mjcf_root.material('grid').texrepeat = (20, 20)
        world.mjcf_root.material('grid').reflectance = 0
        world.mjcf_root.texture('skybox').rgb1 = (.33, .47, .48)
        world.mjcf_root.texture('skybox').rgb2 = (.78, .84, .78)
        world.mjcf_root.worldbody.add_light(
            name='soft_key', pos=(-8, -10, 25), dir=(.2, .3, -1),
            diffuse=(.55, .53, .46), specular=(.12, .12, .12), castshadow=True,
        )
        # Decorated odor landmarks. Contact geometry/food ingestion is not modeled.
        for index, (pos, color) in enumerate(zip(SOURCE_POSITIONS, [(0.96, .55, .21, 1), (.20, .64, .62, 1)])):
            world.mjcf_root.worldbody.add_geom(
                name=f'odor_source_{index}', type=GEOM_TYPES['ellipsoid'],
                pos=pos, size=(1.3, 1.3, .45), rgba=color, contype=0, conaffinity=0,
            )
            world.mjcf_root.worldbody.add_geom(
                name=f'landmark_{index}', type=GEOM_TYPES['cylinder'],
                pos=(pos[0] + 2.5, pos[1], 1.4), size=(.2, 1.4), rgba=color,
                contype=0, conaffinity=0,
            )
        # Sparse, low landmarks give optic-flow structure without hiding the fly.
        for i, angle in enumerate(np.linspace(0, 2*np.pi, 32, endpoint=False)):
            world.mjcf_root.worldbody.add_geom(
                name=f'boundary_{i}', type=GEOM_TYPES['box'],
                pos=(20*np.cos(angle), 20*np.sin(angle), .18), size=(.45, .45, .18),
                rgba=(.30, .39, .33, 1), contype=0, conaffinity=0,
            )
        world.add_fly(self.fly, [0, 0, .8], Rotation3D('quat', [1, 0, 0, 0]),
                      bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                      add_ground_contact_sensors=False)
        self.sim = Simulation(world)
        self.sim.mj_model.vis.global_.offwidth = 960
        self.sim.mj_model.vis.global_.offheight = 600
        self.sim.mj_model.vis.quality.shadowsize = 2048
        preprogrammed = PreprogrammedSteps()
        dofs = self.fly.get_actuated_jointdofs_order('position')
        self.gait = HybridTurningController(timestep=self.sim.timestep,
                                           preprogrammed_steps=preprogrammed,
                                           output_dof_order=dofs)
        self.sim.reset()
        self.gait.reset(seed=seed)
        apply_locomotion_action(self.sim, self.fly.name, LocomotionAction(
            joint_angles=preprogrammed.default_pose_by_dof_order(dofs),
            adhesion_onoff=np.ones(6, dtype=bool)))
        self.sim.warmup()
        self.thorax_index = self.fly.get_bodysegs_order().index(BodySegment('c_thorax'))
        self.thorax_id = mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                        self.fly.bodyseg_to_mjcfbody[BodySegment('c_thorax')].name)

    def observe(self):
        position = self.sim.get_body_positions(self.fly.name)[self.thorax_index].copy()
        forward = self.sim.mj_data.xmat[self.thorax_id].reshape(3, 3)[:, 0]
        heading = float(np.arctan2(forward[1], forward[0]))
        offsets = SOURCE_POSITIONS - position
        distances = np.linalg.norm(offsets, axis=1)
        bearings = (np.arctan2(offsets[:, 1], offsets[:, 0]) - heading + np.pi) % (2*np.pi) - np.pi
        return {'position_mm': position, 'heading_rad': heading,
                'source_positions_mm': SOURCE_POSITIONS.copy(), 'source_distances_mm': distances,
                'source_bearings_rad': bearings, 'odor_intensity': np.exp(-distances/8.0)}

    def step(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError('Controller command must contain two finite descending drives.')
        if np.any(np.abs(command) > 2):
            raise ValueError('Descending drive outside validated gait range [-2, 2].')
        obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
        apply_locomotion_action(self.sim, self.fly.name, self.gait.step(command, obs))
        self.sim.step()
        if not np.isfinite(self.sim.mj_data.qpos).all():
            raise RuntimeError('Physics produced nonfinite positions.')
        return self.observe()

    def close(self):
        self.sim.close()


def record_demo(output_dir, duration: float = 2.0, seed: int = 0,
                controller: Callable | None = None, fps: int = 30,
                playback_speed: float = .25):
    """Stream a 1280×720 multiview recording and full-rate telemetry to disk.

    ``controller(time_s, observation)`` returns a length-2 drive or a dictionary
    ``{'command': drive, 'diagnostics': {name: scalar}}``. Time starts after the
    inherited 50 ms standing warmup. Set ``controller.label`` and optionally
    ``controller.metadata`` to identify the backend. The first-person view is
    cinematic perspective, explicitly not compound-eye neural input.
    """
    if not math.isfinite(duration) or duration <= 0 or fps <= 0 or not math.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError('duration, fps, and playback_speed must be positive and finite.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: output/name for key, name in [('video', 'demo.mp4'), ('preview', 'preview.png'),
                                              ('telemetry', 'telemetry.npz'), ('metadata', 'metadata.json')]}
    label = getattr(controller, 'label', 'Body baseline / upstream hybrid CPG')
    body = FlyBody(seed)
    nsteps = max(1, int(round(duration/body.sim.timestep)))
    time_s = (np.arange(nsteps)+1)*body.sim.timestep
    telemetry = {'time_s': time_s, 'position_mm': np.empty((nsteps, 3)),
                 'heading_rad': np.empty(nsteps), 'command': np.empty((nsteps, 2)),
                 'odor_intensity': np.empty((nsteps, 2)),
                 'joint_angles_rad': np.empty((nsteps, len(body.sim.get_joint_angles(body.fly.name)))),
                 'cpg_magnitudes': np.empty((nsteps, 6))}
    diagnostic_rows = []
    nframes = 0
    previous_position = body.observe()['position_mm']
    third = first = writer = None
    try:
        third = mujoco.Renderer(body.sim.mj_model, height=568, width=936)
        first = mujoco.Renderer(body.sim.mj_model, height=164, width=292)
        writer = imageio.get_writer(paths['video'], fps=fps, codec='libx264', quality=8,
                                    macro_block_size=1, ffmpeg_log_level='error')
        for i in range(nsteps):
            time = i*body.sim.timestep
            obs = body.observe()
            result = controller(time, obs) if controller is not None else np.array([1.2, .65] if time < duration/2 else [.65, 1.2])
            diagnostics = result.get('diagnostics', {}) if isinstance(result, dict) else {}
            command = np.asarray(result['command'] if isinstance(result, dict) else result, dtype=float)
            obs = body.step(command)
            for key in ('position_mm', 'heading_rad', 'odor_intensity'):
                telemetry[key][i] = obs[key]
            telemetry['command'][i] = command
            telemetry['joint_angles_rad'][i] = body.sim.get_joint_angles(body.fly.name)
            telemetry['cpg_magnitudes'][i] = body.gait.cpg_network.curr_magnitudes
            if time + 1e-12 >= nframes*playback_speed/fps:
                third.update_scene(body.sim.mj_data, camera=body.third_person.name)
                first.update_scene(body.sim.mj_data, camera=body.first_person.name)
                speed = np.linalg.norm(obs['position_mm']-previous_position)/(playback_speed/fps) if nframes else 0.
                frame = compose_frame(third.render(), first.render(), time_s[i], command,
                                      telemetry['position_mm'][:i+1], obs, diagnostics,
                                      label, playback_speed, speed)
                writer.append_data(frame)
                if nframes == 0:
                    Image.fromarray(frame).save(paths['preview'])
                previous_position = obs['position_mm'].copy()
                diagnostic_rows.append({'time_s': float(time_s[i]), **{k: float(v) for k, v in diagnostics.items() if np.isscalar(v) and not isinstance(v, str)}})
                nframes += 1
    finally:
        if writer is not None:
            writer.close()
        if first is not None:
            first.close()
        if third is not None:
            third.close()
        body.close()
    np.savez_compressed(paths['telemetry'], **telemetry)
    metadata = {'controller': label, 'controller_metadata': getattr(controller, 'metadata', {}),
                'seed': seed, 'physics_timestep_s': body.sim.timestep,
                'simulated_duration_s': nsteps*body.sim.timestep, 'warmup_s': .05,
                'frames': nframes, 'fps': fps, 'playback_speed': playback_speed,
                'video_duration_s': nframes/fps, 'resolution': [1280, 720],
                'first_person': 'Conventional perspective camera; not compound-eye neural input.',
                'odor_model': 'Two analytic scalar fields exp(-distance_mm/8); synthetic concentrations, not fluid dynamics.',
                'sources_mm': SOURCE_POSITIONS.tolist(), 'landmarks': 'Visual only; no reward delivery or ingestion physics.',
                'body': 'NeuroMechFly articulated mesh, upstream hybrid turning controller',
                'controller_diagnostics': diagnostic_rows,
                'displacement_mm': (telemetry['position_mm'][-1]-telemetry['position_mm'][0]).tolist()}
    paths['metadata'].write_text(json.dumps(metadata, indent=2)+'\n')
    return paths
