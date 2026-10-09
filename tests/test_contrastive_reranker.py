"""Synthetic CPU checks; do not represent real UPAR performance."""
import sys,unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from abpr.contrastive_reranker import ContrastiveRanker,fuse_distances
from train_contrastive_retrieval import train_contrastive,contrastive_losses,ContrastiveHead,synth_queries


class TestContrastive(unittest.TestCase):
    def test_training_and_save_load(self):
        rng=np.random.default_rng(7)
        labels=rng.integers(0,2,size=(180,40)).astype(np.float32)
        labels[rng.random((180,40))<0.12]=-1
        attr_vectors=rng.normal(size=(40,64)).astype(np.float32)
        feat=np.clip(labels,0,1)@attr_vectors+rng.normal(0,0.5,size=(180,64))
        model,logs=train_contrastive(feat.astype(np.float32),labels,np.arange(180)%3,
                      steps=3,batch_size=32,output_dim=32,device='cpu')
        q=synth_queries(labels[:5],rng)
        d=model.distance(q,feat[:11])
        self.assertEqual(d.shape,(5,11))
        self.assertTrue(np.isfinite(d).all())
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'r.npz';model.save(p)
            loaded=ContrastiveRanker.load(p)
            np.testing.assert_allclose(loaded.distance(q,feat[:11]),d,rtol=1e-6)

    def test_identity_blend(self):
        b=np.array([[.2,.3,.4]],dtype=np.float32)
        other=np.array([[.9,.5,.2]],dtype=np.float32)
        self.assertIs(fuse_distances(b,other,0.0),b)
        self.assertTrue(np.isfinite(fuse_distances(b,other,0.4)).all())

    def test_unknown_query_not_used(self):
        proj=np.eye(8,dtype=np.float32)
        pos=np.ones((40,8),dtype=np.float32);neg=-pos
        model=ContrastiveRanker(proj,pos,neg)
        embeddings=np.random.default_rng(8).normal(size=(7,8)).astype(np.float32)
        q=np.full((1,40),-1,dtype=np.float32)
        self.assertTrue(np.all(model.distance(q,embeddings)==1.0))

    def test_multi_positive_and_missing_label_finite(self):
        torch.manual_seed(7)
        net=ContrastiveHead(16,12)
        f=torch.randn(12,16)
        y=torch.randint(0,2,(12,40)).float()
        y[0,12:]=-1
        rng=np.random.default_rng(10)
        q=torch.tensor(synth_queries(y.numpy(),rng),dtype=torch.float32)
        loss,detail=contrastive_losses(net,f,y,q,torch.arange(12)%3)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(t.grad is not None for t in net.parameters()))

if __name__=='__main__':unittest.main()
