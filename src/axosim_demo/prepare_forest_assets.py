"""Rebuild woodland OBJ/PNG assets from pinned CC0 source downloads.

Python -m axosim_demo.prepare_forest_assets requires about 1.2 GB source cache.
Only the visual foliage is simplified; no neural or physical model is modified.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from .forest_scene import ASSETS as O

S = Path(__file__).resolve().parents[2] / "data/forest_scene/source"
O.mkdir(exist_ok=True, parents=True)


def obj(path, v, uv, f):
    with path.open("w") as o:
        np.savetxt(o, v, fmt="v %.5f %.5f %.5f")
        np.savetxt(o, uv, fmt="vt %.6f %.6f")
        ff = f + 1
        np.savetxt(
            o, np.stack([ff, ff], axis=2).reshape(-1, 6), fmt="f %d/%d %d/%d %d/%d"
        )


def import_model(name):
    d = json.loads((S / name / (name + ".gltf")).read_text())
    buf = np.memmap(S / name / d["buffers"][0]["uri"], mode="r", dtype="u1")

    def a(i):
        z = d["accessors"][i]
        b = d["bufferViews"][z["bufferView"]]
        ty = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[z["componentType"]]
        n = {"VEC3": 3, "VEC2": 2, "SCALAR": 1}[z["type"]]
        dt = np.dtype(ty)
        return np.ndarray(
            (z["count"], n),
            dtype=dt,
            buffer=buf,
            offset=b.get("byteOffset", 0) + z.get("byteOffset", 0),
            strides=(b.get("byteStride", n * dt.itemsize), dt.itemsize),
        )

    out = []
    for mi, m in enumerate(d["meshes"]):
        parts = []
        for pi, p in enumerate(m["primitives"]):
            v = a(p["attributes"]["POSITION"])
            uv = a(p["attributes"]["TEXCOORD_0"])
            f = a(p["indices"]).reshape(-1, 3)
            if len(f) > 150000:
                # Distributed paired-triangle foliage subset: visual LOD only, untouched brain/physics.
                sel = np.linspace(0, len(f) // 2 - 1, 45000, dtype=int) * 2
                f = f[np.stack([sel, sel + 1], axis=1).ravel()]
            unique, inv = np.unique(f, return_inverse=True)
            v = v[unique].astype(float)[:, [0, 2, 1]] * [1000, -1000, 1000]
            uv = uv[unique].copy()
            f = inv.reshape(-1, 3)
            mat = d["materials"][p["material"]]["pbrMetallicRoughness"].get(
                "baseColorTexture"
            )
            tex = None
            if mat:
                tr = mat.get("extensions", {}).get("KHR_texture_transform", {})
                uv = uv * np.array(tr.get("scale", [1, 1])) + tr.get("offset", [0, 0])
                im = d["images"][d["textures"][mat["index"]]["source"]]["uri"]
                tex = O / (Path(im).stem + ".png")
                Image.open(S / name / im).save(tex)
            dest = O / f"{name}_{mi}_{pi}.obj"
            obj(dest, v, uv, f)
            parts.append(
                {
                    "mesh_path": dest.name,
                    "texture_path": tex.name if tex else None,
                    "bounds": [v.min(0).tolist(), v.max(0).tolist()],
                    "triangles": len(f),
                }
            )
        out.append(parts)
    print(name, sum(x["triangles"] for y in out for x in y), flush=True)
    return out


def cutout(path):
    vs = []
    uvs = []
    fs = []
    for line in path.read_text().splitlines():
        a = line.split()
        if a[0] == "v":
            vs.append(list(map(float, a[1:])))
        elif a[0] == "vt":
            uvs.append(list(map(float, a[1:])))
        elif a[0] == "f":
            fs.append([int(x.split("/")[0]) - 1 for x in a[1:]])
    vs = np.array(vs)
    uvs = np.array(uvs)
    fs = np.array(fs)
    mask = np.array(
        Image.open(S / "fern_02/textures/fern_02_alpha_1k.png").convert("L")
    )
    n = 14
    bary = []
    faces = []
    idx = {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            idx[(i, j)] = len(bary)
            bary.append([1 - (i + j) / n, i / n, j / n])
    for i in range(n):
        for j in range(n - i):
            faces.append([idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]])
            if j < n - i - 1:
                faces.append([idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]])
    bary = np.array(bary)
    faces = np.array(faces)
    ov = []
    ou = []
    of = []
    off = 0
    for f in fs:
        uv = bary @ uvs[f]
        v = bary @ vs[f]
        cent = uv[faces].mean(1)
        xy = np.clip((cent * np.array(mask.shape[::-1])).astype(int), 0, 1023)
        keep = mask[xy[:, 1], xy[:, 0]] > 150
        ff = faces[keep]
        if not len(ff):
            continue
        unique, inv = np.unique(ff, return_inverse=True)
        ov.append(v[unique])
        ou.append(uv[unique])
        of.append(inv.reshape(-1, 3) + off)
        off += len(unique)
    obj(path, np.concatenate(ov), np.concatenate(ou), np.concatenate(of))
    print(path, len(np.concatenate(of)), flush=True)


def main():
    import hashlib
    from urllib.request import Request, urlopen

    manifest = json.loads((O / "sources.json").read_text())
    for item in manifest["sources"]:
        path = S / item["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            request = Request(
                item["url"], headers={"User-Agent": "axosim-demo (research preview)"}
            )
            with urlopen(request, timeout=180) as response, path.open("wb") as target:
                while chunk := response.read(4 * 1024 * 1024):
                    target.write(chunk)
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"Source hash mismatch: {path}")
    for n in [
        "fern_02",
        "tree_stump_01",
        "rock_moss_set_01",
        "bark_debris_01",
        "pine_tree_01",
    ]:
        import_model(n)
    Image.open(S / "forest_leaves_02/diffuse.jpg").save(O / "forest_floor.png")
    # Lower camera sees true geometry; panorama only supplies distant forest/horizon.
    im = Image.open(S / "forest_slope/panorama.jpg")
    im.resize((4096, 2048)).save(O / "forest_panorama.png")
    u = np.linspace(0, 1, 129)
    vv = np.linspace(0.002, 0.998, 65)
    U, V = np.meshgrid(u, vv)
    theta = 2 * np.pi * U
    phi = np.pi * V
    v = (
        np.stack(
            [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
            axis=-1,
        ).reshape(-1, 3)
        * 20000
    )
    uv = np.stack([U, 1 - V], axis=-1).reshape(-1, 2)
    f = []
    for j in range(64):
        for i in range(128):
            z = j * 129 + i
            f.extend([[z, z + 129, z + 1], [z + 1, z + 129, z + 130]])
    obj(O / "forest_panorama.obj", v, uv, np.array(f)[:, ::-1])

    for path in O.glob("fern_02_*.obj"):
        cutout(path)
    # MuJoCo ignores per-pixel diffuse alpha. Leaf silhouettes were converted
    # to real geometry above, by subdividing 14x14 and sampling the CC0 alpha map.
    used = json.loads((O / "scene.json").read_text())["objects"]
    keep = {x["mesh_path"] for x in used} | {x["texture_path"] for x in used}
    keep |= {"forest_floor.png"}
    for path in O.iterdir():
        if path.suffix in (".obj", ".png") and path.name not in keep:
            path.unlink()
    print("Verified pinned source hashes and rebuilt forest runtime assets.")


if __name__ == "__main__":
    main()
