"""Behavioral contracts for the real articulated body and streamed recording."""
import importlib.util
import json

import imageio.v2 as imageio
import numpy as np


def test_body_recording_has_moving_pose_and_decodable_multiview_video(tmp_path):
    # A missing physics step, dropped controller command, or broken encoder fails.
    assert importlib.util.find_spec('axosim_demo.body') is not None, 'body adapter missing'
    from axosim_demo.body import record_demo

    seen = []
    def controller(time, observation):
        seen.append(observation)
        return {'command': np.array([1.2, 0.4]), 'diagnostics': {'rate_hz': 12.0}}
    controller.label = 'Test drive: no neural model'
    result = record_demo(tmp_path, duration=0.04, seed=9, controller=controller, fps=20, playback_speed=0.2)
    data = np.load(result['telemetry'])
    assert len(data['time_s']) == 400
    assert np.isfinite(data['position_mm']).all()
    assert np.max(np.ptp(data['joint_angles_rad'], axis=0)) > 0.01
    np.testing.assert_allclose(data['command'], np.tile([1.2, 0.4], (400, 1)))
    assert seen[0]['odor_intensity'].shape == (2,)
    assert np.isfinite(seen[0]['heading_rad'])
    frames = imageio.mimread(result['video'])
    assert len(frames) == 4
    assert frames[0].shape == (720, 1280, 3)
    assert frames[0][100:550, 30:900].std() > 10
    metadata = json.loads(result['metadata'].read_text())
    assert metadata['controller'] == controller.label
    assert metadata['playback_speed'] == 0.2
