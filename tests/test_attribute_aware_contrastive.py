import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from abpr.attribute_aware_contrastive import multilabel_supcon, query_hard_negative_loss, AttributeAwareContrastive

def test_backward():
    torch.manual_seed(4)
    z=torch.randn(5,12,requires_grad=True)
    qz=torch.randn(5,12,requires_grad=True)
    q=torch.tensor([[1,0,1,0,1,0], [1,0,1,0,0,0], [1,0,1,0,1,0], [1,0,1,1,0,0], [0,0,0,0,0,0]],dtype=torch.float)
    sup=multilabel_supcon(z,q)
    hn=query_hard_negative_loss(z,qz,q)
    assert torch.isfinite(sup) and torch.isfinite(hn)
    (sup+hn).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert qz.grad is not None and torch.isfinite(qz.grad).all()

def test_missing_labels_not_imputed():
    z=torch.randn(3,8,requires_grad=True)
    q=torch.tensor([[1,-1,0,-1],[1,0,0,-1],[1,-1,1,-1]],dtype=torch.float)
    sup=multilabel_supcon(z,q,min_shared=4)
    assert sup.item()==0.0
    assert torch.isfinite(query_hard_negative_loss(z,z,q,min_shared=4))

def test_wrapped_loss_updates_embedding():
    z=torch.randn(4,16,requires_grad=True)
    qz=torch.randn(4,16,requires_grad=True)
    q=torch.tensor([[1,0,1,0,1],[1,0,1,0,0],[1,0,1,0,1],[1,0,0,0,0]],dtype=torch.float)
    def original(image_emb,query_emb,query_states,temperature=0.08):
        return (image_emb-query_emb).pow(2).mean()
    fn=AttributeAwareContrastive(original,warmup_steps=1)
    objective=fn(z,qz,q)
    objective.backward()
    assert fn.steps==1 and torch.isfinite(objective)
    assert z.grad.abs().sum()>0 and qz.grad.abs().sum()>0
