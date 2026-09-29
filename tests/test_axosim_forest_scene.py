"""Static asset compilation and visual smoke test, with no motor simulation."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from axosim_demo.forest_scene import forest_scene_manifest
from axosim_demo.scene_preview import render_scene_preview


def test_imported_forest_and_controller_free_preview(tmp_path):
    scene = forest_scene_manifest()
    trees = [x for x in scene["objects"] if x["name"].startswith("pine_tree")]
    assert len({tuple(x["pos_mm"]) for x in trees}) == 7
    assert all(
        Path(x[k]).is_file()
        for x in scene["objects"]
        for k in ["mesh_path", "texture_path"]
    )
    result = render_scene_preview(tmp_path)
    metadata = json.loads((result / "metadata.json").read_text())
    assert metadata["simulation_steps"] == 0
    assert metadata["actuator_count"] == 0
    assert metadata["motor_controller"] is None and metadata["neural_model"] is None
    for name in ["tpp", "fpp", "establishing"]:
        image = np.asarray(Image.open(result / f"{name}.png"))
        assert image.shape == (720, 1280, 3)
        assert image[:670].std() > 20
        assert np.mean(np.all(image[:670] < 5, axis=-1)) < 0.05
