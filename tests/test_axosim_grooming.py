import numpy as np
import pytest
from flygym.anatomy import JointDOF, BodySegment, RotationAxis


def dof(parent, child, axis):
    return JointDOF(BodySegment(parent), BodySegment(child), RotationAxis(axis))


def test_dof_mapping_preserves_anatomy_and_rotation_axis():
    from axosim_demo.grooming import source_key
    assert source_key(dof('c_thorax', 'lf_coxa', 'yaw')) == 'Angle_LF_ThC_yaw'
    assert source_key(dof('rf_coxa', 'rf_trochanterfemur', 'roll')) == 'Angle_RF_CTr_roll'
    assert source_key(dof('lf_trochanterfemur', 'lf_tibia', 'pitch')) == 'Angle_LF_FTi_pitch'
    assert source_key(dof('lm_tibia', 'lm_tarsus1', 'pitch')) is None
    assert source_key(dof('c_thorax', 'c_head', 'pitch')) is None


def test_interpolation_recovers_samples_without_loop_or_extrapolation():
    from axosim_demo.grooming import sample_angles
    data = {'angle': np.array([1., 2., 4.])}
    assert sample_angles(data, .01, 100)['angle'] == 2.
    assert sample_angles(data, .005, 100)['angle'] == 1.5
    with pytest.raises(ValueError, match='range'):
        sample_angles(data, .025, 100)


def test_hash_is_checked_before_unpickling(tmp_path):
    from axosim_demo.grooming import load_trajectory
    p = tmp_path / 'untrusted.pkl'
    p.write_bytes(b'not a pickle')
    with pytest.raises(ValueError, match='SHA256'):
        load_trajectory(p)
