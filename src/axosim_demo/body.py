"""Retired engineering prototype: body, hybrid gait, and callback bridge.

Based on FlyGym's tutorials/4d_turning_controller.ipynb (Apache-2.0).
The gait controller is retained; the callback supplies bilateral descending drive.
This architecture does not meet the scientific fly-demo requirement. It is kept
only for infrastructure tests and provenance, and is absent from the demo CLI.
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
from flygym_demo.complex_terrain import (
    HybridTurningController,
    HybridControllerObservation,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)
from axosim_demo.scene import decorate_world

SOURCE_POSITIONS = np.array([[12.0, 5.0, 0.4], [12.0, -5.0, 0.4]])


class FlyBody:
    """Free articulated fly in a landmark arena (millimetres and seconds)."""

    def __init__(self, seed: int = 0, num_flies: int = 1, spawn_positions=None):
        if num_flies not in (1, 2, 3, 4):
            raise ValueError("num_flies must be 1–4 for this scene.")
        if spawn_positions is not None and np.asarray(spawn_positions).shape != (
            num_flies,
            3,
        ):
            raise ValueError("spawn_positions must have shape (num_flies,3).")
        self.spawn_positions = (
            np.asarray(spawn_positions, dtype=float).tolist()
            if spawn_positions is not None
            else [
                [0, 0, 0.8]
                if i == 0
                else [-10 * i, 8 * ((i + 1) // 2) * (1 if i % 2 else -1), 0.8]
                for i in range(num_flies)
            ]
        )
        if not np.isfinite(self.spawn_positions).all():
            raise ValueError("spawn_positions must be finite.")
        self.flies = [
            make_locomotion_fly(name=f"axosim_fly_{i}", colorize=True)
            for i in range(num_flies)
        ]
        self.fly = self.flies[0]
        self.third_person = self.fly.add_tracking_camera(
            name="third_person",
            pos_offset=(-3, -10, 2.2) if num_flies == 1 else (-3, -19, 6),
            rotation=Rotation3D("xyaxes", (0.957, -0.287, 0, 0.061, 0.203, 0.977)),
            fovy=40,
        )
        self.first_person = self.fly.add_tracking_camera(
            name="first_person",
            mode="fixed",
            pos_offset=(1.45, 0, 0.45),
            rotation=Rotation3D("xyaxes", (0, -1, 0, 0.25, 0, 1)),
            fovy=85,
        )
        world = decorate_world(FlatGroundWorld(half_size=187.5))
        if num_flies > 1:
            self.third_person = world.mjcf_root.worldbody.add_camera(
                name="multi_camera",
                pos=(-6, -42, 20),
                xyaxes=(1, 0, 0, 0, 0.414, 0.910),
                fovy=45,
                mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
            )
        for i, fly in enumerate(self.flies):
            spawn = self.spawn_positions[i]
            world.add_fly(
                fly,
                spawn,
                Rotation3D("quat", [1, 0, 0, 0]),
                bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                add_ground_contact_sensors=False,
            )
        self.sim = Simulation(world)
        self.sim.mj_model.vis.global_.offwidth = 1280
        self.sim.mj_model.vis.global_.offheight = 720
        self.sim.mj_model.vis.quality.shadowsize = 4096
        self.sim.mj_model.vis.headlight.ambient = (0.50, 0.50, 0.50)
        self.sim.mj_model.vis.headlight.diffuse = (0.60, 0.60, 0.60)
        self.gaits = []
        self.thorax_indices = []
        self.thorax_ids = []
        self.sim.reset()
        for i, fly in enumerate(self.flies):
            steps = PreprogrammedSteps()
            dofs = fly.get_actuated_jointdofs_order("position")
            gait = HybridTurningController(
                timestep=self.sim.timestep,
                preprogrammed_steps=steps,
                output_dof_order=dofs,
            )
            gait.reset(seed=seed + i)
            self.gaits.append(gait)
            apply_locomotion_action(
                self.sim,
                fly.name,
                LocomotionAction(
                    joint_angles=steps.default_pose_by_dof_order(dofs),
                    adhesion_onoff=np.ones(6, dtype=bool),
                ),
            )
            self.thorax_indices.append(
                fly.get_bodysegs_order().index(BodySegment("c_thorax"))
            )
            self.thorax_ids.append(
                mujoco.mj_name2id(
                    self.sim.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    fly.bodyseg_to_mjcfbody[BodySegment("c_thorax")].name,
                )
            )
        self.gait = self.gaits[0]
        self.thorax_index = self.thorax_indices[0]
        self.thorax_id = self.thorax_ids[0]
        self.sim.warmup()
        self.camera_framing = self._reserve_overlay_space()

    def _reserve_overlay_space(self, width=1280, height=720, clear_width=912):
        """Translate only the TPP camera; neural sensing/physics are untouched."""
        model, data = self.sim.mj_model, self.sim.mj_data
        camera_id = model.camera(self.third_person.name).id
        rotation = data.cam_xmat[camera_id].reshape(3, 3)
        target = np.mean([self.observe(i)['position_mm'] for i in range(len(self.flies))], axis=0)
        coordinates = (target-data.cam_xpos[camera_id]) @ rotation
        depth = -coordinates[2]
        focal = height/(2*np.tan(np.deg2rad(model.cam_fovy[camera_id])/2))
        projected = width/2+focal*coordinates[0]/depth
        shift = (projected-clear_width/2)*depth/focal
        # MuJoCo tracking uses these compiled reference offsets; changing cam_pos
        # alone would not move an already compiled tracking camera.
        field = 'cam_pos0' if len(self.flies)==1 else 'cam_poscom0'
        getattr(model,field)[camera_id] += shift*rotation[:,0]
        mujoco.mj_forward(model,data)
        return {'target_x_pixels':clear_width/2,'canvas':[width,height],
                'clear_scene_width_pixels':clear_width,'rightward_shift_mm':float(shift),
                'runtime_offset_field':field,
                'runtime_offset_mm':getattr(model,field)[camera_id].tolist(),
                'tracked_target':'thorax' if len(self.flies)==1 else 'population center of mass'}

    def observe(self, index=0):
        position = self.sim.get_body_positions(self.flies[index].name)[
            self.thorax_indices[index]
        ].copy()
        forward = self.sim.mj_data.xmat[self.thorax_ids[index]].reshape(3, 3)[:, 0]
        heading = float(np.arctan2(forward[1], forward[0]))
        offsets = SOURCE_POSITIONS - position
        distances = np.linalg.norm(offsets, axis=1)
        bearings = (np.arctan2(offsets[:, 1], offsets[:, 0]) - heading + np.pi) % (
            2 * np.pi
        ) - np.pi
        return {
            "position_mm": position,
            "heading_rad": heading,
            "source_positions_mm": SOURCE_POSITIONS.copy(),
            "source_distances_mm": distances,
            "source_bearings_rad": bearings,
            "odor_intensity": np.exp(-distances / 8.0),
        }

    def step(self, command):
        commands = np.asarray(command, dtype=float)
        expected = (2,) if len(self.flies) == 1 else (len(self.flies), 2)
        if commands.shape != expected or not np.isfinite(commands).all():
            raise ValueError(f"Controller command must have finite shape {expected}.")
        if np.any(np.abs(commands) > 2):
            raise ValueError("Descending drive outside validated gait range [-2, 2].")
        for fly, gait, cmd in zip(self.flies, self.gaits, commands.reshape(-1, 2)):
            obs = HybridControllerObservation.from_sim(self.sim, fly.name)
            apply_locomotion_action(self.sim, fly.name, gait.step(cmd, obs))
        self.sim.step()
        if not np.isfinite(self.sim.mj_data.qpos).all():
            raise RuntimeError("Physics produced nonfinite positions.")
        return self.observe()

    def close(self):
        self.sim.close()


def record_demo(
    output_dir,
    duration: float = 2.0,
    seed: int = 0,
    controller: Callable | None = None,
    fps: int = 30,
    playback_speed: float = 0.25,
    *,
    view="tpp",
    raw_video=False,
    num_flies=1,
    controller_factory=None,
    spawn_positions=None,
    plot_renderer="tikz",
):
    """Stream a 1280×720 primary-view recording and full-rate telemetry to disk.

    ``controller(time_s, observation)`` returns a length-2 drive or a dictionary
    ``{'command': drive, 'diagnostics': {name: scalar}}``. Time starts after the
    inherited 50 ms standing warmup. Set ``controller.label`` and optionally
    ``controller.metadata`` to identify the backend. The first-person view is
    cinematic perspective, explicitly not compound-eye neural input.
    """
    if (
        not math.isfinite(duration)
        or duration <= 0
        or fps <= 0
        or not math.isfinite(playback_speed)
        or playback_speed <= 0
    ):
        raise ValueError(
            "duration, fps, and playback_speed must be positive and finite."
        )
    if view not in ("tpp", "fpp"):
        raise ValueError("view must be tpp or fpp")
    if num_flies > 1 and controller is not None and controller_factory is None:
        raise ValueError(
            "Multiple neural flies require controller_factory for independent state."
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        key: output / name
        for key, name in [
            ("video", "demo.mp4"),
            ("preview", "preview.png"),
            ("telemetry", "telemetry.npz"),
            ("metadata", "metadata.json"),
        ]
    }
    label = getattr(controller, "label", "Body baseline / upstream hybrid CPG")
    body = FlyBody(seed, num_flies=num_flies, spawn_positions=spawn_positions)
    controllers = (
        [controller_factory(i) for i in range(num_flies)]
        if controller_factory
        else [controller] * num_flies
    )
    if controller_factory:
        controller = controllers[0]
        label = getattr(controller, "label", "Controller")
    neural_count = getattr(controller, "metadata", {}).get("neurons")
    caption = (
        label
        + (f"; N={neural_count}" if neural_count else "")
        + ("; perspective camera" if view == "fpp" else "")
    )
    if num_flies > 1:
        caption = f"{num_flies} independent flies; N={neural_count or 0} per fly; neural traces: fly 1"
    if plot_renderer not in ('tikz','raster'):
        raise ValueError('plot_renderer must be tikz or raster')
    if raw_video:
        paths["raw_video"] = output / "scene.mp4"
    raw_path = output / ('scene.mp4' if raw_video else '.scene.mp4')
    nsteps = max(1, int(round(duration / body.sim.timestep)))
    time_s = (np.arange(nsteps) + 1) * body.sim.timestep
    telemetry = {
        "time_s": time_s,
        "position_mm": np.empty((nsteps, 3)),
        "heading_rad": np.empty(nsteps),
        "command": np.empty((nsteps, 2)),
        "odor_intensity": np.empty((nsteps, 2)),
        "joint_angles_rad": np.empty(
            (nsteps, len(body.sim.get_joint_angles(body.fly.name)))
        ),
        "cpg_magnitudes": np.empty((nsteps, 6)),
    }
    diagnostic_rows = []
    nframes = 0
    telemetry["all_positions_mm"] = np.empty((nsteps, num_flies, 3))
    telemetry["all_commands"] = np.empty((nsteps, num_flies, 2))
    renderer = writer = raw_writer = None
    try:
        renderer = mujoco.Renderer(body.sim.mj_model, height=720, width=1280)
        writer = imageio.get_writer(
            raw_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        for i in range(nsteps):
            time = i * body.sim.timestep
            results = []
            commands = []
            for fly_index, ctrl in enumerate(controllers):
                observation = body.observe(fly_index)
                result = (
                    ctrl(time, observation)
                    if ctrl is not None
                    else np.array([1.2, 0.65] if time < duration / 2 else [0.65, 1.2])
                )
                results.append(result)
                commands.append(
                    np.asarray(
                        result["command"] if isinstance(result, dict) else result,
                        dtype=float,
                    )
                )
            command = commands[0]
            diagnostics = (
                results[0].get("diagnostics", {})
                if isinstance(results[0], dict)
                else {}
            )
            obs = body.step(commands[0] if num_flies == 1 else commands)
            telemetry["all_positions_mm"][i] = [
                body.observe(j)["position_mm"] for j in range(num_flies)
            ]
            telemetry["all_commands"][i] = commands
            for key in ("position_mm", "heading_rad", "odor_intensity"):
                telemetry[key][i] = obs[key]
            telemetry["command"][i] = command
            telemetry["joint_angles_rad"][i] = body.sim.get_joint_angles(body.fly.name)
            telemetry["cpg_magnitudes"][i] = body.gait.cpg_network.curr_magnitudes
            if time + 1e-12 >= nframes * playback_speed / fps:
                renderer.update_scene(
                    body.sim.mj_data,
                    camera=(
                        body.third_person if view == "tpp" else body.first_person
                    ).name,
                )
                scene = renderer.render()
                writer.append_data(scene)
                if nframes == 0:
                    Image.fromarray(scene).save(paths["preview"])
                diagnostic_rows.append(
                    {
                        "time_s": float(time_s[i]),
                        **{
                            k: float(v)
                            for k, v in diagnostics.items()
                            if np.isscalar(v) and not isinstance(v, str)
                        },
                    }
                )
                nframes += 1
    finally:
        if writer is not None:
            writer.close()
        if raw_writer is not None:
            raw_writer.close()
        if renderer is not None:
            renderer.close()
        body.close()
    np.savez_compressed(paths["telemetry"], **telemetry)
    metadata = {
        "controller": label,
        "controller_metadata": getattr(controller, "metadata", {}),
        "camera": {
            "position_coordinates": "world"
            if num_flies > 1 and view == "tpp"
            else "thorax-local",
            "quaternion_wxyz": body.sim.mj_model.camera(
                (body.third_person if view == "tpp" else body.first_person).name
            ).quat.tolist(),
            "position_offset_mm": list(
                (body.third_person if view == "tpp" else body.first_person).pos
            ),
            "vertical_fov_degrees": float(
                (body.third_person if view == "tpp" else body.first_person).fovy
            ),
        },
        "controllers_by_fly": [
            {
                "label": getattr(c, "label", "Body baseline"),
                "metadata": getattr(c, "metadata", {}),
            }
            for c in controllers
        ],
        "spawn_positions_mm": body.spawn_positions,
        "view": view,
        "plot_renderer": plot_renderer,
        "camera_framing": body.camera_framing if view=='tpp' else {'centered_perspective':True},
        "num_flies": num_flies,
        "neural_diagnostics_fly_index": 0,
        "neural_diagnostics_sampling": "at video frame times; native surrogate units, not firing rates",
        "scene": "Photographed tabletop and scanned CC0 apple; source provenance in assets/natural_scene/sources.json",
        "interfly_interactions": "Independent articulated flies in one physics world; no social controller or explicit fly–fly contacts",
        "seed": seed,
        "physics_timestep_s": body.sim.timestep,
        "simulated_duration_s": nsteps * body.sim.timestep,
        "warmup_s": 0.05,
        "frames": nframes,
        "fps": fps,
        "playback_speed": playback_speed,
        "video_duration_s": nframes / fps,
        "resolution": [1280, 720],
        "first_person": "Conventional perspective camera; not compound-eye neural input.",
        "odor_model": "Two analytic scalar fields exp(-distance_mm/8); synthetic concentrations, not fluid dynamics.",
        "sources_mm": SOURCE_POSITIONS.tolist(),
        "landmarks": "Visual only; no reward delivery or ingestion physics.",
        "body": "NeuroMechFly articulated mesh, upstream hybrid turning controller",
        "controller_diagnostics": diagnostic_rows,
        "displacement_mm": (
            telemetry["position_mm"][-1] - telemetry["position_mm"][0]
        ).tolist(),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    from .render_saved_recording import recompose
    recompose(output, renderer=plot_renderer, raw_filename=raw_path.name)
    if not raw_video:
        raw_path.unlink()
    return paths
