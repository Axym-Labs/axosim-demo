"""Photographed tabletop and scanned fruit, with unchanged flat contact physics.

CC0 assets and exact download hashes are recorded in assets/natural_scene/sources.json.
The displayed environment uses diffuse texture maps; MuJoCo is not a PBR renderer.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from flygym.utils.mjcf import add_material, add_texture, GEOM_TYPES

ASSETS = Path(__file__).parent / "assets" / "natural_scene"


@lru_cache(maxsize=1)
def _apple_mesh():
    spec = json.loads((ASSETS / "food_apple_01.gltf").read_text())
    binary = (ASSETS / "food_apple_01.bin").read_bytes()
    arrays = []
    for accessor in spec["accessors"]:
        view = spec["bufferViews"][accessor["bufferView"]]
        components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[accessor["type"]]
        dtype = {5126: "<f4", 5123: "<u2"}[accessor["componentType"]]
        arrays.append(
            np.frombuffer(
                binary,
                dtype=dtype,
                count=accessor["count"] * components,
                offset=view["byteOffset"],
            )
            .reshape(-1, components)
            .copy()
        )
    vertices, normals, uv, indices = arrays
    # glTF +Y up, MuJoCo +Z up. Convert metres to mm (apple ~98 mm wide).
    vertices = vertices[:, [0, 2, 1]] * [1000, -1000, 1000]
    normals = normals[:, [0, 2, 1]] * [1, -1, 1]
    return vertices, normals, uv, indices.reshape(-1, 3)


def decorate_world(world):
    """Add real-world visual assets to an existing world before compilation.

    Visual props have no contact forces; the inherited ground plane is unchanged.
    A 375-mm photographed tabletop crop retains the source's physical texture scale.
    """
    spec = world.mjcf_root
    add_texture(
        spec,
        name="photographed_wood",
        type="2d",
        file=str(ASSETS / "tabletop_crop.png"),
    )
    add_material(
        spec,
        name="tabletop",
        texture="photographed_wood",
        texrepeat=(1, 1),
        reflectance=0,
        specular=0.12,
        shininess=0.2,
    )
    if hasattr(world, "ground_geom"):
        world.ground_geom.material = "tabletop"
        world.ground_geom.size = (187.5, 187.5, 1)
    else:
        spec.worldbody.add_geom(
            name="visual_tabletop",
            type=GEOM_TYPES["plane"],
            size=(187.5, 187.5, 1),
            material="tabletop",
            contype=0,
            conaffinity=0,
        )
    spec.texture("skybox").rgb1 = (0.60, 0.64, 0.60)
    spec.texture("skybox").rgb2 = (0.90, 0.87, 0.77)
    spec.worldbody.add_light(
        name="window",
        pos=(-40, -30, 65),
        dir=(0.2, 0.25, -1),
        diffuse=(0.65, 0.62, 0.55),
        specular=(0.15, 0.15, 0.12),
        castshadow=True,
    )
    add_texture(
        spec,
        name="apple_photo",
        type="2d",
        file=str(ASSETS / "textures/food_apple_01_diff_2k.png"),
    )
    add_material(
        spec, name="apple_skin", texture="apple_photo", specular=0.22, shininess=0.35
    )
    vertices, normals, uv, faces = _apple_mesh()
    spec.add_mesh(
        name="scanned_apple",
        uservert=vertices.ravel(),
        usernormal=normals.ravel(),
        usertexcoord=uv.ravel(),
        userface=faces.ravel(),
        userfacenormal=faces.ravel(),
        userfacetexcoord=faces.ravel(),
        scale=(0.65, 0.65, 0.65),
    )
    # 64-mm apple, far enough behind the 0–25 mm walking area to avoid intersection.
    spec.worldbody.add_geom(
        name="whole_apple",
        type=GEOM_TYPES["mesh"],
        meshname="scanned_apple",
        material="apple_skin",
        pos=(40, 30, 0),
        contype=0,
        conaffinity=0,
    )
    # Small irregular fruit fragments mark the analytic odor origins. These are
    # authored thin peel meshes with photographic skin, not scanned chunk geometry.
    rng = np.random.default_rng(17)
    for i, position in enumerate(((12, 5, 0.035), (12, -5, 0.035))):
        v = np.array(
            [
                [-1, -0.8, -0.3],
                [1.2, -0.7, -0.3],
                [0.9, 0.9, -0.3],
                [-0.8, 0.7, -0.3],
                [-0.6, -0.5, 0.45],
                [0.8, -0.5, 0.7],
                [0.65, 0.7, 0.55],
                [-0.7, 0.6, 0.35],
            ]
        )
        v += rng.normal(0, 0.06, v.shape)
        v[:, 2] *= 0.07
        uv = np.tile(
            np.array([[0.13, 0.25], [0.28, 0.25], [0.28, 0.40], [0.13, 0.40]]), (2, 1)
        )
        f = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ]
        )
        spec.add_mesh(
            name=f"fruit_fragment_{i}",
            uservert=v.ravel(),
            userface=f.ravel(),
            usertexcoord=uv.ravel(),
            userfacetexcoord=f.ravel(),
        )
        spec.worldbody.add_geom(
            name=f"odor_source_{i}",
            type=GEOM_TYPES["mesh"],
            meshname=f"fruit_fragment_{i}",
            pos=position,
            material="apple_skin",
            contype=0,
            conaffinity=0,
        )
    return world
