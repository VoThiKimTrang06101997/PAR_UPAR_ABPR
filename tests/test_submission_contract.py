"""Unit tests for the competition contract, using a mock runtime only in tests."""
import importlib.util, sys, types
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class FakeABPRRuntime:
    calls=0
    def __init__(self, model_path):
        assert Path(model_path).exists()
        self.device='cpu'
        self.attribute_names=[
            'Age-Young','Age-Adult','Age-Old','Gender-Female',
            'Hair-Length-Short','Hair-Length-Long','Hair-Length-Bald','UpperBody-Length-Short',
            'UpperBody-Color-Black','UpperBody-Color-Blue','UpperBody-Color-Brown','UpperBody-Color-Green',
            'UpperBody-Color-Grey','UpperBody-Color-Orange','UpperBody-Color-Pink','UpperBody-Color-Purple',
            'UpperBody-Color-Red','UpperBody-Color-White','UpperBody-Color-Yellow','UpperBody-Color-Other',
            'LowerBody-Length-Short','LowerBody-Color-Black','LowerBody-Color-Blue','LowerBody-Color-Brown',
            'LowerBody-Color-Green','LowerBody-Color-Grey','LowerBody-Color-Orange','LowerBody-Color-Pink',
            'LowerBody-Color-Purple','LowerBody-Color-Red','LowerBody-Color-White','LowerBody-Color-Yellow',
            'LowerBody-Color-Other','LowerBody-Type-Trousers&Shorts','LowerBody-Type-Skirt&Dress',
            'Accessory-Backpack','Accessory-Bag','Accessory-Glasses-Normal','Accessory-Glasses-Sun','Accessory-Hat',
        ]
    def encode_gallery(self, gallery):
        self.__class__.calls+=1
        vals=np.asarray([float(i)/10.0 for i in range(len(gallery))],np.float32)
        p=np.tile(np.linspace(.1,.9,40,dtype=np.float32)[None,:],(len(gallery),1))+vals[:,None]
        return p,np.zeros((len(gallery),8),np.float32)
    def distance(self, q,p,e):
        return -(q @ p.T)


def load_runner(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules,'abpr_runtime',types.SimpleNamespace(ABPRRuntime=FakeABPRRuntime))
    module_file=tmp_path/'run.py';module_file.write_bytes((ROOT/'submission_run.py').read_bytes())
    asset=tmp_path/'assets';asset.mkdir(exist_ok=True)
    (asset/'model.pt').write_bytes(b'test-only mock model')
    spec=importlib.util.spec_from_file_location('subtest',module_file)
    obj=importlib.util.module_from_spec(spec);sys.modules['subtest']=obj;spec.loader.exec_module(obj)
    return obj


def test_large_cpu_gallery_no_prior_fallback(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    n=12001
    gallery=[{'image_path':f'/test/{i}.jpg'} for i in range(n)]
    query=np.zeros((2,40),dtype=np.float32)
    query[0,:10]=1;query[1,20:30]=1
    before=FakeABPRRuntime.calls
    res=runner.rank_gallery({'gallery':gallery,'queries':query,'attribute_names':runner.CANONICAL_NAMES})['distances']
    assert FakeABPRRuntime.calls==before+1, 'must use actual encode_gallery even on large CPU gallery'
    assert res.shape==(2,n) and np.isfinite(res).all()
    assert res.std()>0


def test_image_dict_string_contract(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch)
    g=['a.jpg',{'path':'b.jpg'},{'image_path':'c.jpg'},{'filename':'d.jpg'}]
    q=np.ones((1,40),np.float32)
    out=runner.rank_gallery({'gallery':g,'queries':q,'attribute_names':runner.CANONICAL_NAMES})['distances']
    assert out.shape==(1,4)


def test_missing_model_is_hard_error(tmp_path,monkeypatch):
    import pytest
    runner=load_runner(tmp_path,monkeypatch)
    runner.MODEL_PATH.unlink()
    with pytest.raises(FileNotFoundError): runner.load_model()
