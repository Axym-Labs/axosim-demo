import numpy as np
from axosim_demo.neuron_fidelity import lif_predict,event_counts,f1,logit_events


def test_lif_rest_and_refractory():
    p=np.array([20.,3.,8.,-2.,-2.,-76.,-50.,20.,2.])
    z=np.zeros((1,20))
    v,s=lif_predict(z,z,p)
    np.testing.assert_allclose(v,-76.)
    assert not s.any()
    p[3]=.7
    v,s=lif_predict(np.ones((1,20))*10,z,p)
    assert s.any()
    assert np.diff(np.flatnonzero(s[0])).min()>=3
    np.testing.assert_allclose(v[s],-55.)


def test_event_matching_is_one_to_one_and_respects_warmup():
    predicted=np.zeros((1,20),bool);expected=predicted.copy()
    predicted[0,[1,10]]=True;expected[0,[1,9,11]]=True
    assert event_counts(predicted,expected,tolerance=2,warmup=5)==(1,1,2)
    assert f1((1,1,2))==2/3
    assert event_counts(predicted,expected,tolerance=0,warmup=0)==(1,2,3)


def test_logit_threshold_does_not_count_plateau_samples_as_many_events():
    logits=np.array([[-5.,-4.,2.,2.,2.,-2.,-3.,1.,-3.]])
    events=logit_events(logits,0.)
    assert events.sum()==2
    assert not events[0,0]
