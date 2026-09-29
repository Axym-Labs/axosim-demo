"""Imported CC0 woodland geometry shared by FlyGym and the flight renderer."""

import json
from pathlib import Path

ASSETS = Path(__file__).parent / "assets" / "forest_scene"
SCENE_DESCRIPTION = (
    "Poly Haven woodland: genuine 3D pine trees, ferns, scanned rocks, "
    "stump and bark; photographic forest floor and distant panorama. "
    "Visual props are non-colliding; locomotion uses the inherited flat contact plane."
)


def forest_scene_manifest():
    """Return portable OBJ/PNG placements in millimetres, with absolute paths."""
    result = json.loads((ASSETS / "scene.json").read_text())
    result["ground_texture"] = str(ASSETS / result["ground_texture"])
    for item in result["objects"]:
        for key in ("mesh_path", "texture_path"):
            if item.get(key):
                item[key] = str(ASSETS / item[key])
    return result


def scene_metadata():
    return {
        "description": SCENE_DESCRIPTION,
        "asset_manifest": str(ASSETS / "scene.json"),
        "source_manifest": str(ASSETS / "sources.json"),
        "license": "CC0-1.0",
        "physical_scale": {
            "units": "mm",
            "ground_photo_width_mm": 3000,
            "panorama_radius_mm": 20000,
            "source_tree_height_mm_approx": 20000,
            "per_object_scale": "explicit in scene.json",
        },
        "rendering_limits": [
            "Diffuse lighting, not full PBR",
            "Finite ground-photo detail at macro scale",
            "Visual pine canopy LOD and geometrically approximated fern cutouts",
            "Decorative props have no contact forces; physical ground is flat",
        ],
        "collection": "https://polyhaven.com/collections/pine_forest",
    }


def decorate_forest_world(world):
    """Decorate a FlyGym MjSpec with imported geometry before compilation."""
    from flygym.utils.mjcf import add_material, add_texture, GEOM_TYPES

    data = forest_scene_manifest()
    spec = world.mjcf_root
    add_texture(spec, name="forest_floor_photo", type="2d", file=data["ground_texture"])
    add_material(
        spec,
        name="forest_floor",
        texture="forest_floor_photo",
        texrepeat=(1, 1),
        reflectance=0,
        specular=0.05,
        shininess=0.1,
    )
    if hasattr(world, "ground_geom"):
        world.ground_geom.material = "forest_floor"
        world.ground_geom.size = (1500, 1500, 1)
    else:
        spec.worldbody.add_geom(
            name="visual_forest_floor",
            type=GEOM_TYPES["plane"],
            size=(1500, 1500, 1),
            material="forest_floor",
            contype=0,
            conaffinity=0,
        )
    meshes = {}
    textures = {}
    for item in data["objects"]:
        path = item["mesh_path"]
        if path not in meshes:
            meshname = f"forest_mesh_{len(meshes)}"
            spec.add_mesh(name=meshname, file=path)
            meshes[path] = meshname
        texture = item["texture_path"]
        if texture not in textures:
            texname = f"forest_texture_{len(textures)}"
            matname = f"forest_material_{len(textures)}"
            add_texture(spec, name=texname, type="2d", file=texture)
            add_material(
                spec,
                name=matname,
                texture=texname,
                specular=0.03,
                shininess=0.1,
                emission=1 if "panorama" in item["name"] else 0,
            )
            textures[texture] = matname
        # Mesh scale is per asset in MuJoCo, so use distinct scaled asset for each size.
        scale = item["scale"]
        name = item["name"]
        meshname = meshes[path]
        if scale != [1, 1, 1]:
            meshname = name + "_scaled_mesh"
            spec.add_mesh(name=meshname, file=path, scale=scale)
        spec.worldbody.add_geom(
            name=name,
            type=GEOM_TYPES["mesh"],
            meshname=meshname,
            material=textures[texture],
            pos=item["pos_mm"],
            quat=item["quat_wxyz"],
            contype=0,
            conaffinity=0,
        )
    spec.worldbody.add_light(
        name="forest_sun",
        pos=(-100, -200, 600),
        dir=(0.2, 0.3, -1),
        diffuse=(0.65, 0.63, 0.54),
        specular=(0.1, 0.1, 0.08),
        castshadow=True,
    )
    # Forest spans metres while the fly is millimetres. Explicit extent keeps near
    # clipping small; the distant photographic horizon requires a long far plane.
    spec.stat.extent = 100
    spec.visual.map.znear = 0.0005
    spec.visual.map.zfar = 500
    return world
