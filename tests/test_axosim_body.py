"""Behavioral contracts for the real articulated body and streamed recording."""

import importlib.util
import json

import imageio.v2 as imageio
import numpy as np


def test_body_recording_has_moving_pose_and_decodable_multiview_video(tmp_path):
    # A missing physics step, dropped controller command, or broken encoder fails.
    assert importlib.util.find_spec("axosim_demo.body") is not None, (
        "body adapter missing"
    )
    from axosim_demo.body import record_demo

    seen = []

    def controller(time, observation):
        seen.append(observation)
        return {"command": np.array([1.2, 0.4]), "diagnostics": {"rate_hz": 12.0}}

    controller.label = "Test drive: no neural model"
    result = record_demo(
        tmp_path,
        duration=0.04,
        seed=9,
        controller=controller,
        fps=20,
        playback_speed=0.2,
        plot_renderer="raster",
    )
    data = np.load(result["telemetry"])
    assert len(data["time_s"]) == 400
    assert np.isfinite(data["position_mm"]).all()
    assert np.max(np.ptp(data["joint_angles_rad"], axis=0)) > 0.01
    np.testing.assert_allclose(data["command"], np.tile([1.2, 0.4], (400, 1)))
    assert seen[0]["odor_intensity"].shape == (2,)
    assert np.isfinite(seen[0]["heading_rad"])
    frames = imageio.mimread(result["video"])
    assert len(frames) == 4
    assert frames[0].shape == (720, 1280, 3)
    assert frames[0][100:550, 30:900].std() > 10
    metadata = json.loads(result["metadata"].read_text())
    assert metadata["controller"] == controller.label
    assert metadata["playback_speed"] == 0.2


def test_separate_primary_views_preserve_full_frame_with_overlay(tmp_path):
    from axosim_demo.body import record_demo

    def controller(time, observation):
        return {
            "command": np.array([1.0, 1.0]),
            "diagnostics": {
                "AxoSim_soma_A": np.sin(time * 30),
                "AxoSim_soma_B": np.cos(time * 30),
                "AxoSim_state_norm": 1 + time,
            },
        }

    controller.label = "AxoSim reduced motif, N=4"
    out = record_demo(
        tmp_path, duration=0.02, controller=controller, view="fpp", raw_video=True,
        plot_renderer="raster",
    )
    frame = imageio.mimread(out["video"])[0]
    raw = imageio.mimread(out["raw_video"])[0]
    assert raw.shape == frame.shape == (720, 1280, 3)
    assert frame[100:600,960:1200].mean() > raw[100:600,960:1200].mean()
    assert np.abs(frame[0].astype(float)-raw[0].astype(float)).mean() < 3
    assert json.loads(out["metadata"].read_text())["view"] == "fpp"


def test_multiple_flies_have_independent_physical_state():
    from axosim_demo.body import FlyBody

    body = FlyBody(seed=0, num_flies=3)
    try:
        before = np.array([body.observe(i)["position_mm"] for i in range(3)])
        for _ in range(100):
            body.step([[0.8, 1.2], [1.2, 0.8], [1.0, 1.0]])
        after = np.array([body.observe(i)["position_mm"] for i in range(3)])
        assert body.sim.mj_model.nq > 3 * 70
        assert np.isfinite(after).all()
        assert np.linalg.norm(after - before) > 0.001
        assert (
            np.min(
                np.linalg.norm(after[:, None, :] - after[None, :, :], axis=-1)
                + np.eye(3) * 100
            )
            > 2
        )
    finally:
        body.close()


def test_tpp_target_is_centered_in_the_uncovered_region():
    from axosim_demo.body import FlyBody
    body=FlyBody()
    try:
        model,data=body.sim.mj_model,body.sim.mj_data
        camera=model.camera(body.third_person.name)
        relative=(body.observe()['position_mm']-data.cam_xpos[camera.id]) @ data.cam_xmat[camera.id].reshape(3,3)
        focal=360/np.tan(np.deg2rad(model.cam_fovy[camera.id])/2)
        projected=640+focal*relative[0]/(-relative[2])
        assert abs(projected-456)<1
    finally:
        body.close()
