from pathlib import Path
import tempfile
import numpy as np
from abpr.retrieval_reranker import AttributeRanker, fuse_distances


def test_ranker_shapes_mask_and_roundtrip():
    r=AttributeRanker(np.ones(40),np.ones(40),np.zeros(40),blend=0.5)
    p=np.array([[.90]*40,[.10]*40,[.70]*40],dtype=np.float32)
    q=np.array([[1]+[-1]*39,[0]+[-1]*39],dtype=np.float32)
    dist=r.distance(q,p)
    assert dist.shape==(2,3)
    assert dist[0,0] < dist[0,1]
    assert dist[1,0] > dist[1,1]
    assert np.isfinite(dist).all()
    f=fuse_distances(dist,dist,0.5)
    assert np.isfinite(f).all()
    assert np.array_equal(fuse_distances(dist,np.zeros_like(dist),0.),dist)
    with tempfile.TemporaryDirectory() as t:
        name=Path(t)/'ranker.npz';r.save(name)
        r2=AttributeRanker.load(name)
        assert np.allclose(r2.distance(q,p),dist)
        assert r2.blend==.5


def test_exact_val_metric():
    from train_retrieval_ranker import evaluate_exact_retrieval
    dist=np.array([[0.2,0.5,0.7],[0.9,0.4,0.1]],dtype=np.float32)
    ids=np.array([10,20,30]); queries=np.array([10,30])
    score=evaluate_exact_retrieval(dist,ids,queries)
    assert score['val_subset_mAP']==1.
    assert score['val_subset_R1']==1.
