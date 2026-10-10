import numpy as np
import pytest
from abpr.query_aware_ranking import score,fuse,retrieval_metrics

def test_exact_baseline_blend_zero():
    base=np.arange(12,dtype=np.float32).reshape(3,4)-6
    new=np.ones_like(base)*5
    assert np.array_equal(fuse(base,new,0.),base)

def test_probabilistic_score_retrieves_correct():
    q=-np.ones((2,40),np.float32)
    q[0,0]=1;q[0,1]=0;q[1,0]=0;q[1,1]=1
    p=np.full((2,40),.5,np.float32)
    p[0,:2]=[.95,.04];p[1,:2]=[.05,.93]
    for mode in ('loglik','expected','margin'):
        s=score(q,p,method=mode)
        assert s.shape==(2,2)
        assert np.all(np.argmin(s,axis=1)==[0,1]),(mode,s)

def test_unknown_not_used():
    q=-np.ones((1,40),np.float32);q[0,0]=1
    p=np.full((2,40),.5,np.float32);p[:,0]=[.9,.1]
    p[0,1]=.01;p[1,1]=.99
    s=score(q,p)
    assert s[0,0]<s[0,1]

def test_query_metrics():
    d=np.array([[.1,.4,.8],[.8,.1,.4]],np.float32)
    out=retrieval_metrics(d,[0,1],[0,1,2])
    assert out['mAP']==1 and out['Rank-1']==1

def test_missing_values_rejected():
    q=np.full((2,40),.5,np.float32);p=np.full((3,40),.5,np.float32)
    with pytest.raises(ValueError): score(q,p)
