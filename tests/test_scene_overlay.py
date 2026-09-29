import numpy as np
from axosim_demo.video import compose_frame


def test_overlay_does_not_expand_canvas_or_render_description():
    source=np.full((720,1280,3),70,dtype=np.uint8)
    plot=np.zeros((600,320,4),dtype=np.uint8)
    frame=compose_frame(source,plot,label='This must remain outside the video')
    np.testing.assert_array_equal(frame,compose_frame(source,plot,label='Different description'))
    assert frame.shape==source.shape
    np.testing.assert_array_equal(frame[10,1000],source[10,1000])
    assert frame[100,1000,0]>200


def test_frosted_overlay_retains_scene_information():
    plot=np.zeros((600,320,4),dtype=np.uint8)
    dark=compose_frame(np.full((720,1280,3),20,dtype=np.uint8),plot)
    light=compose_frame(np.full((720,1280,3),200,dtype=np.uint8),plot)
    assert light[100,1000,0]>dark[100,1000,0]
