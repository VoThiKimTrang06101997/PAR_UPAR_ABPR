import numpy as np
import torch

from abpr.model import (
    sample_sparse_queries,
    probability_degree_match_loss,
    degree_match_listwise_loss,
    calibrated_attribute_distances,
    track2_metrics,
    apply_affine_calibration,
)


def test_sparse_queries_preserve_shape_and_known_values():
    q=torch.tensor([[1.,0.,1.,0.]+[-1.]*36,[0.,1.,0.,1.]+[-1.]*36])
    out=sample_sparse_queries(q,min_known=2,max_known=3)
    assert out.shape==q.shape
    assert torch.all((out==-1) | (out==0) | (out==1))
    keep=out>=0
    assert torch.all(out[keep]==q[keep])


def test_degree_match_losses_are_finite():
    logits=torch.randn(4,40,requires_grad=True)
    q=torch.full((4,40),-1.); q[:,0]=torch.tensor([1.,0.,1.,0.]); q[:,1]=1.
    y=torch.randint(0,2,(4,40)).float(); valid=torch.ones_like(y)
    a=probability_degree_match_loss(logits,q,y,valid)
    b=degree_match_listwise_loss(logits,q,y,valid)
    assert torch.isfinite(a) and torch.isfinite(b)
    (a+b).backward()
    assert logits.grad is not None


def test_match_distance_prefers_matching_probabilities():
    probs=torch.tensor([[.95,.05],[.05,.95]],dtype=torch.float32)
    q=np.array([[1,0],[0,1]],dtype=np.float32)
    d=calibrated_attribute_distances(probs,q,distance_kind='match')
    assert d[0,0] < d[0,1]
    assert d[1,1] < d[1,0]


def test_affine_calibration_identity():
    p=torch.tensor([[.2,.8],[.7,.3]])
    out=apply_affine_calibration(p,[1,1],[0,0])
    assert torch.allclose(p,out,atol=1e-6)


def test_track2_metrics_perfect_ranking():
    labels=np.array([[1,0],[0,1],[1,1]],dtype=np.int16)
    queries=np.array([[1,0],[0,1]],dtype=np.int16)
    d=np.array([[0.,2.,1.],[2.,0.,1.]],dtype=np.float32)
    m=track2_metrics(d,queries,labels)
    assert m['Rank-1']==1.0
    assert m['mAP']>0.99
    assert m['mADM']>0.0


def test_unknown_query_bits_are_ignored():
    probs=torch.tensor([[.9,.1],[.9,.9]],dtype=torch.float32)
    q=np.array([[1,-1]],dtype=np.float32)
    d=calibrated_attribute_distances(probs,q,distance_kind='match')
    assert abs(float(d[0,0]-d[0,1])) < 1e-6


def test_visual_prototype_head_shapes_and_gradients():
    from abpr.prototype import AttributePrototypeHead, masked_prototype_bce
    head=AttributePrototypeHead(in_dim=32,prototype_dim=16,num_attributes=40)
    fmap=torch.randn(3,32,8,4,requires_grad=True)
    global_feat=fmap.mean(dim=(2,3))
    logits,feat,sp,sn=head(fmap,global_feat)
    assert logits.shape==(3,40)
    assert feat.shape==(3,40,16)
    y=torch.randint(0,2,(3,40)).float(); valid=torch.ones_like(y)
    loss=masked_prototype_bce(logits,y,valid)
    loss.backward()
    assert head.pos_prototypes.grad is not None
    assert fmap.grad is not None


def test_prototype_domain_alignment_is_finite():
    from abpr.prototype import AttributePrototypeHead, prototype_domain_alignment_loss
    head=AttributePrototypeHead(in_dim=24,prototype_dim=12,num_attributes=40)
    fmap=torch.randn(9,24,6,3)
    global_feat=fmap.mean(dim=(2,3))
    _,feat,_,_=head(fmap,global_feat)
    y=torch.randint(0,2,(9,40)).float(); valid=torch.ones_like(y)
    domains=torch.tensor([0,0,0,1,1,1,2,2,2])
    pos,neg=head.normalized_prototypes()
    loss=prototype_domain_alignment_loss(feat,y,valid,domains,pos,neg,min_count=1)
    assert torch.isfinite(loss)


def test_probability_blend_endpoints():
    from abpr.prototype import blend_attribute_probabilities
    a=torch.zeros(2,40)+.2; b=torch.zeros(2,40)+.8
    assert torch.allclose(blend_attribute_probabilities(a,b,0.0),a)
    mid=blend_attribute_probabilities(a,b,0.5)
    assert torch.allclose(mid,torch.full_like(mid,.5))
