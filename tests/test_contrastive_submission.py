from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from abpr.contrastive_reranker import ContrastiveRanker


def test_rank_gallery_contrastive_integration():
    spec=importlib.util.spec_from_file_location('abpr_contrastive_submission_test',ROOT/'submission_run.py')
    sub=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    class FakeRuntime:
        attribute_names=sub.CANONICAL_NAMES
        def encode_gallery(self,gallery):
            g=len(gallery)
            return (np.tile(np.linspace(.1,.9,40,dtype=np.float32),(g,1)) + np.arange(g)[:,None]*.003),np.eye(g,8,dtype=np.float32)
        def distance(self,query,probs,embed):
            return np.broadcast_to(np.arange(len(probs),dtype=np.float32)[None,:],(len(query),len(probs))).copy()
    with TemporaryDirectory() as directory:
        tmp=Path(directory)
        assets=tmp/'assets';assets.mkdir()
        shutil.copy2(ROOT/'abpr'/'contrastive_reranker.py',tmp/'contrastive_reranker.py')
        sys.path.insert(0,str(tmp))
        model=ContrastiveRanker(np.eye(8,dtype=np.float32),
                  np.tile(np.eye(8,dtype=np.float32)[0],(40,1)),
                  np.tile(-np.eye(8,dtype=np.float32)[0],(40,1)),blend=0.5)
        model.save(assets/'contrastive.npz')
        sub.HERE=tmp
        sub._MODEL=FakeRuntime()
        s={'gallery':[{'image_path':f'/fake/{i}.png'} for i in range(5)],
           'queries':[[1]+[-1]*39,[0]+[-1]*39],
           'attribute_names':sub.CANONICAL_NAMES}
        score=sub.rank_gallery(s)['distances']
        assert score.shape==(2,5)
        assert np.isfinite(score).all()
        assert not np.allclose(score[0],score[1])
