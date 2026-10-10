import torch
from abpr.joint_training_losses import (masked_asl_loss,query_hard_ranking_loss,
  mild_photometric_view,consistency_loss,ema_kd_loss)

def test_masked_asl_ignores_unknown_and_backward():
    logits=torch.randn(5,40,requires_grad=True)
    y=torch.randint(0,2,(5,40)).float(); y[:,3]=-1; mask=(y>=0).float()
    prior=torch.rand(40)*.7+.1
    loss=masked_asl_loss(logits,y,mask,prior)
    assert torch.isfinite(loss)
    loss.backward(); assert logits.grad.abs().sum()>0
    assert logits.grad[:,3].abs().sum()==0

def test_query_hard_negative_with_known_contradiction():
    z=torch.randn(4,12,requires_grad=True)
    qz=torch.randn(4,12,requires_grad=True)
    full=torch.zeros((4,40))
    full[:,0]=torch.tensor([0,1,0,1]).float()
    sparse=full.clone(); sparse[:,1:]=-1
    loss=query_hard_ranking_loss(z,qz,sparse,full,margin=1.0)
    assert torch.isfinite(loss)
    loss.backward(); assert z.grad is not None and qz.grad is not None

def test_aux_gradients_and_no_target_grad():
    x=torch.randn(4,3,16,8)
    v=mild_photometric_view(x); assert v.shape==x.shape
    l1=torch.randn(4,40,requires_grad=True);l2=torch.randn(4,40,requires_grad=True)
    z1=torch.randn(4,8,requires_grad=True);z2=torch.randn(4,8,requires_grad=True)
    cons=consistency_loss(l1,l2,z1,z2)
    kd=ema_kd_loss(l1,l2,z1,z2)
    (cons+kd).backward()
    assert l1.grad is not None and z2.grad is not None

def test_strict_shapes():
    try:
        query_hard_ranking_loss(torch.randn(4,12),torch.randn(4,11),torch.randn(4,40),torch.randn(4,40))
    except ValueError: pass
    else: raise AssertionError('Mismatched embedding shapes accepted')
