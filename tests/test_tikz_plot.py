"""Verify layer geometry and truthful data selection without a TeX dependency."""
import numpy as np
import pytest
from PIL import Image

from axosim_demo.tikz_plot import _limits, compile_neural_overlay, reveal_lines


def test_reveal_only_keeps_historical_x_coordinates(tmp_path):
    data=np.full((100,120,4),255,dtype=np.uint8)
    data[:10]=0;data[91:]=0;data[:,:20]=0;data[:,101:]=0
    path=tmp_path/'lines.png';Image.fromarray(data).save(path)
    overlay={'lines_path':str(path),'time_range':[0,2], 'plot_rects':[[20,10,100,90]]}
    assert not np.asarray(reveal_lines(overlay,0))[:,:,3].any()
    actual=np.asarray(reveal_lines(overlay,1))
    assert np.array_equal(actual[:,20:61],data[:,20:61])
    assert not actual[:,61:,3].any()
    assert np.array_equal(np.asarray(reveal_lines(overlay,2)),data)


def test_compiler_missing_dependency_is_explicit(monkeypatch,tmp_path):
    monkeypatch.setattr('axosim_demo.tikz_plot.shutil.which',lambda _:None)
    with pytest.raises(RuntimeError,match='tectonic executable required'):
        compile_neural_overlay({},tmp_path)


def test_legible_y_ticks_include_narrow_state_range():
    lo,hi,ticks=_limits([np.array([11.7,12.1,12.5])])
    assert lo<11.7 and hi>12.5
    assert len(ticks)>=2
