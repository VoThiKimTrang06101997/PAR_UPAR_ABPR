import numpy as np
import pytest
from abpr.domain_robust_calibration import calibrate_probabilities,query_distance,fuse_distances,semantic_metrics,domain_macro


def fixture():
 rng=np.random.default_rng(12)
 return rng.integers(-1,2,(5,40)).astype(np.float32),rng.uniform(.02,.98,(90,40)).astype(np.float32),np.full(40,.4,np.float32)


def test_identity_and_shrink():
 _,p,prior=fixture()
 output=calibrate_probabilities(p,scale=np.ones(40),bias=np.zeros(40),priors=prior)
 assert np.allclose(output,p,atol=5e-6)
 reduced=calibrate_probabilities(p,scale=np.ones(40),bias=np.zeros(40),priors=prior,shrink=.12)
 assert np.isfinite(reduced).all() and np.max(np.abs(reduced-p))>0


def test_missing_attribute_is_ignored():
 q,p,pr=fixture()
 q[0,:]=-1
 for method in ('loglik','expected'):
  dist=query_distance(q,p,method=method,priors=pr)
  assert dist.shape==(5,90)
  assert np.max(abs(dist[0]))<1e-6


def test_identity_exact_blend_and_ranking():
 q,p,pr=fixture()
 original=query_distance(q,p,priors=pr)
 arbitrary=np.random.default_rng(1).random(original.shape,dtype=np.float32)
 assert np.array_equal(fuse_distances(original,arbitrary,0),original)
 score=fuse_distances(original,arbitrary,.2)
 assert np.isfinite(score).all() and score.shape==original.shape


def test_semantic_and_macro():
 gids=np.array([7,7,2,2,5,5,8,8]);ids=np.array([7,2,5,8])
 d=np.ones((4,8),np.float32)
 for i,qid in enumerate(ids):d[i,gids==qid]=0
 m=semantic_metrics(d,ids,gids)
 assert m['mAP']==1 and m['Rank_1']==1
 macro=domain_macro(d,ids,gids,np.array([0,0,1,1,2,2,0,0]))
 assert set(macro['per_domain'])=={'0','1','2'}
 assert macro['macro']['mAP']==1


def test_bad_input_errors():
 q,p,pr=fixture()
 q[0,0]=.5
 with pytest.raises(ValueError):query_distance(q,p,priors=pr)
 with pytest.raises(ValueError):calibrate_probabilities(p,scale=np.ones(30),bias=np.zeros(40),priors=pr)
 with pytest.raises(ValueError):fuse_distances(np.ones((2,4)),np.ones((2,3)),.1)


def test_fit_synthetic_gradient_small():
 # Smoke test parameter training only; not a claimed real-dataset result.
 from train_domain_robust_calibrator import fit_attribute_calibrator
 rng=np.random.default_rng(12)
 y=rng.integers(0,2,(140,40)).astype(np.float32)
 probs=np.clip(y*.7+.15+rng.normal(0,.04,y.shape),.01,.99).astype(np.float32)
 domains=np.arange(len(y))%3
 (scale,bias),report=fit_attribute_calibrator(probs,y,domains,steps=4,lr=.005)
 assert scale.shape==(40,) and bias.shape==(40,)
 assert report['train_images']>0 and np.isfinite(scale).all()
