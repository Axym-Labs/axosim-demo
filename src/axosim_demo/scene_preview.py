"""Static woodland asset preview; no neural model, motor controller, or simulation step."""

import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mujoco
from flygym.compose import FlatGroundWorld, NeuroMechFly, KinematicPosePreset
from flygym.anatomy import AxisOrder, JointPreset, Skeleton
from flygym.utils.math import Rotation3D
from .forest_scene import decorate_forest_world, scene_metadata


def render_scene_preview(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    world = decorate_forest_world(FlatGroundWorld(half_size=1500))
    # Geometry and neutral joint rest positions only: no actuator, neural model,
    # gait/CPG/controller, simulation integrator or locomotion tutorial imports.
    fly = NeuroMechFly(name="static_neuromechfly")
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY
    )
    fly.add_joints(
        skeleton,
        neutral_pose=KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            AxisOrder.YAW_PITCH_ROLL
        ),
    )
    fly.colorize()
    world.add_fly(fly, (0, 0, 1.1), Rotation3D("quat", [1, 0, 0, 0]))
    cameras = [
        ("tpp", (-3, -10, 1.65), (0, 0.0, 1.1), 40),
        ("establishing", (-1500, -2200, 1250), (0, 2200, 2500), 65),
        ("fpp", (1.45, 0, 1.55), (30, 0, 2), 85),
    ]
    resolved_cameras = []
    for name, pos, target, fovy in cameras:
        back = np.array(pos) - target
        back = back / np.linalg.norm(back)
        right = np.cross([0, 0, 1], back)
        right = right / np.linalg.norm(right)
        up = np.cross(back, right)
        if name == "tpp":
            focal = 720 / (2 * np.tan(np.deg2rad(fovy) / 2))
            shift = right * (
                (640 - 456) * np.linalg.norm(np.array(pos) - target) / focal
            )
            pos = np.array(pos) + shift
            target = np.array(target) + shift
        resolved_cameras.append(
            {
                "name": name,
                "pos_mm": np.asarray(pos).tolist(),
                "target_mm": np.asarray(target).tolist(),
                "fovy": fovy,
            }
        )
        world.mjcf_root.worldbody.add_camera(
            name=name, pos=pos, xyaxes=tuple(right) + tuple(up), fovy=fovy
        )
    model = world.mjcf_root.compile()
    data = mujoco.MjData(model)
    # Display the documented neutral pose directly, without actuator or stepping.
    for joint_id in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE):
        address = model.jnt_qposadr[joint_id]
        data.qpos[address] = model.qpos_spring[address]
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.map.zfar = 50000 / model.stat.extent
    model.vis.map.znear = 0.03 / model.stat.extent
    model.vis.headlight.ambient = (0.5, 0.5, 0.5)
    model.vis.headlight.diffuse = (0.6, 0.6, 0.6)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    for name, *_ in cameras:
        renderer.update_scene(data, camera=name)
        image = Image.fromarray(renderer.render())
        image.save(out / f"{name}-static-asset-preview.png")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 670, 1280, 720), fill=(255, 255, 255))
        draw.text((16, 680), "AxoSim - Axym Labs", fill=(63, 33, 182), font=font)
        draw.text(
            (300, 680),
            "Static environment / neutral body asset preview — no behavior simulation",
            fill=(50, 50, 50),
            font=font,
        )
        image.save(out / f"{name}.png")
    renderer.close()
    (out / "metadata.json").write_text(
        json.dumps(
            {
                **scene_metadata(),
                "simulation_steps": 0,
                "neural_model": None,
                "motor_controller": None,
                "camera_units": "mm",
                "cameras": resolved_cameras,
                "actuator_count": model.nu,
                "model_geoms": model.ngeom,
                "model_meshes": model.nmesh,
            },
            indent=2,
        )
        + "\n"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/forest-preview")
    args = parser.parse_args()
    print(render_scene_preview(args.output))


if __name__ == "__main__":
    main()
