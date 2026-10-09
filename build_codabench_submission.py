"""Package trained Track-2 runtime and execute the exact ZIP using REAL public images.
No simulated fallback check can pass for a missing/broken trained network.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

REQUIRED=('run.py','metadata.yaml','abpr_runtime.py','assets/model.pt')

def safe_extract(zf,destination):
    destination=Path(destination).resolve()
    for info in zf.infolist():
        dest=(destination/info.filename).resolve()
        if not dest.is_relative_to(destination): raise ValueError(f'Unsafe ZIP member: {info.filename}')
    zf.extractall(destination)


def get_sample_gallery(official_root, source_root, count=4):
    import pandas as pd
    sys.path.insert(0,str(source_root))
    from abpr.core import ImageResolver, detect_image_column, ATTRIBUTE_NAMES
    csv=official_root/'data/annotations/task2/val/gt.csv'
    if not csv.exists(): raise FileNotFoundError(f'Missing organizer VAL annotations {csv}')
    frame=pd.read_csv(csv)
    col=detect_image_column(frame)
    resolver=ImageResolver(official_root/'data',official_root,extra_roots=[source_root,source_root/'data'])
    # Spread the sample across the table instead of drawing neighboring near-identical frames.
    indices=[(i*(len(frame)-1))//max(count-1,1) for i in range(count)]
    paths=[]
    for i in indices:
        paths.append(str(resolver.resolve(str(frame.iloc[i][col]))))
    names=list(ATTRIBUTE_NAMES)
    row=frame.iloc[indices[0]]
    query=[float(row.get(n, 0)) for n in names]
    query=[float(v) if v in (0,1,0.0,1.0) else 0.0 for v in query]
    return {'gallery':[{'image_path':p} for p in paths], 'queries':[query,query[::-1]], 'attribute_names':names}


def smoke(package, sample, timeout, device):
    runner=r'''
import importlib.util,json,numpy as np,os,sys
from pathlib import Path
sys.path.insert(0,os.getcwd())
from pathlib import Path
spec=importlib.util.spec_from_file_location('run',Path('run.py').resolve())
m=importlib.util.module_from_spec(spec);sys.modules['run']=m;spec.loader.exec_module(m)
sample=json.loads(Path(sys.argv[1]).read_text())
res=m.rank_gallery(sample)
assert isinstance(res,dict) and 'distances' in res
s=np.asarray(res['distances'],dtype=np.float32)
assert s.shape==(len(sample['queries']),len(sample['gallery'])),s.shape
assert np.isfinite(s).all()
assert float(np.std(s,axis=1).max())>1e-7,'Constant ranking'
print(json.dumps({'passed':True,'shape':list(s.shape),'min':float(s.min()),'max':float(s.max()),'std':float(s.std())}))
'''
    with tempfile.TemporaryDirectory(prefix='abpr_packaged_') as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(package) as zf: safe_extract(zf,tmp)
        (tmp/'smoke_sample.json').write_text(json.dumps(sample),encoding='utf-8')
        (tmp/'smoke.py').write_text(runner,encoding='utf-8')
        env=os.environ.copy(); env['ABPR_FORCE_FALLBACK']='0'; env['PYTHONUNBUFFERED']='1'
        if device=='cpu': env['CUDA_VISIBLE_DEVICES']=''
        p=subprocess.run([sys.executable,'-u','smoke.py','smoke_sample.json'],cwd=str(tmp),env=env,capture_output=True,text=True,timeout=timeout)
        if p.returncode!=0: raise RuntimeError('Packaged inference FAILED\nSTDOUT:\n'+p.stdout[-3000:]+'\nSTDERR:\n'+p.stderr[-7000:])
        return {'status':'passed','stdout_tail':p.stdout[-1000:],'stderr_tail':p.stderr[-1500:],'device_test':device}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime-zip',required=True)
    p.add_argument('--result-dir',required=True)
    p.add_argument('--name',default='Submission_ABPR')
    p.add_argument('--official-root',required=True)
    p.add_argument('--source-root',required=True)
    p.add_argument('--smoke-timeout',type=int,default=900)
    p.add_argument('--smoke-device',choices=['auto','cpu'],default='auto')
    p.add_argument('--ranker-checkpoint',type=Path,default=None)
    p.add_argument('--contrastive-checkpoint',type=Path,default=None)
    a=p.parse_args()
    out=Path(a.result_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    source=Path(a.source_root).resolve(); official=Path(a.official_root).resolve()
    runtime=Path(a.runtime_zip).resolve()
    if not runtime.exists():raise FileNotFoundError(runtime)
    temp=out/a.name;shutil.rmtree(temp,ignore_errors=True);temp.mkdir()
    with zipfile.ZipFile(runtime) as zf: safe_extract(zf,temp)
    # Current official competition contract, not guessed metadata.
    metadata=official/'examples/task2/sample_code_submission/metadata.yaml'
    if not metadata.is_file():raise FileNotFoundError(metadata)
    shutil.copy2(metadata,temp/'metadata.yaml')
    shutil.copy2(source/'submission_run.py',temp/'run.py')
    ranker_applied=False
    if a.ranker_checkpoint is not None:
        ranker_path=a.ranker_checkpoint.resolve()
        if not ranker_path.is_file():raise FileNotFoundError(ranker_path)
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('retrieval_reranker',source/'abpr'/'retrieval_reranker.py')
        helper=module_from_spec(spec);spec.loader.exec_module(helper)
        r=helper.AttributeRanker.load(ranker_path)
        (temp/'assets').mkdir(exist_ok=True,parents=True)
        shutil.copy2(ranker_path,temp/'assets'/'ranker.npz')
        shutil.copy2(source/'abpr'/'retrieval_reranker.py',temp/'retrieval_reranker.py')
        ranker_applied=r.blend>0
        print('Ranker blend selected by VAL:',r.blend,flush=True)
    contrastive_applied=False
    if a.contrastive_checkpoint is not None:
        contrastive_path=a.contrastive_checkpoint.resolve()
        if not contrastive_path.is_file():raise FileNotFoundError(contrastive_path)
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('contrastive_reranker',source/'abpr'/'contrastive_reranker.py')
        module=module_from_spec(spec);spec.loader.exec_module(module)
        ranker=module.ContrastiveRanker.load(contrastive_path)
        (temp/'assets').mkdir(exist_ok=True,parents=True)
        shutil.copy2(contrastive_path,temp/'assets'/'contrastive.npz')
        shutil.copy2(source/'abpr'/'contrastive_reranker.py',temp/'contrastive_reranker.py')
        contrastive_applied=ranker.blend>0
        print('Contrastive blend selected by public VAL:',ranker.blend,flush=True)
    for f in REQUIRED:
        if not (temp/f).is_file():raise FileNotFoundError('Missing in submission: '+f)
    # Do not package unrelated prior-only or prior-jitter assets.
    for f in (temp/'assets').glob('*prior*.json'): f.unlink()
    archive=out/(a.name+'.zip')
    if archive.exists(): archive.unlink()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as zf:
        for item in sorted(temp.rglob('*')):
            if item.is_file() and '__pycache__' not in item.parts:
                zf.write(item,arcname=item.relative_to(temp).as_posix())
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:raise RuntimeError('Invalid ZIP CRC')
        for f in REQUIRED:assert f in zf.namelist(),f
        assert zf.read('metadata.yaml')==metadata.read_bytes()
    sample=get_sample_gallery(official,source)
    check=smoke(archive,sample,a.smoke_timeout,a.smoke_device)
    manifest={
        'submission_zip':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'size_mib':round(archive.stat().st_size/1048576,2),
        'runtime_zip':str(runtime),'trained_model_inference_smoke_test':check,
        'official_metadata_match':True,
        'ranker_packaged':a.ranker_checkpoint is not None,
        'ranker_active':ranker_applied,
        'contrastive_packaged':a.contrastive_checkpoint is not None,
        'contrastive_active':contrastive_applied,
        'critical':'No prior fallback. Failure in worker will be visible in ingestion logs.'
    }
    (out/(a.name+'_manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
