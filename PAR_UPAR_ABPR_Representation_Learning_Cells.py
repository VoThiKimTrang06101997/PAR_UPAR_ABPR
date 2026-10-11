#@title 32) Representation Learning — Restore exact public split and existing SigLIP caches (NO re-encoding)
from pathlib import Path
import os,sys,json,hashlib,base64,zlib,subprocess
import numpy as np,pandas as pd
from tqdm.auto import tqdm

SOURCE_ROOT=Path(globals().get('SOURCE_ROOT','/content/PAR_UPAR_ABPR'))
OFFICIAL_ROOT=Path(globals().get('OFFICIAL_ROOT','/content/UPAR-Challenge-2027'))
DRIVE_ROOT=Path(globals().get('DRIVE_ROOT','/content/drive/MyDrive/PedestrianAttributeRecognition'))
RESULT_DIR=Path(globals().get('RESULT_DIR',DRIVE_ROOT/'ABPR_Results'))
SIGLIP_DIR=Path(globals().get('SIGLIP_DIR',RESULT_DIR/'siglip2_retrieval'))
REPR_DIR=SIGLIP_DIR/'representation_learning';REPR_DIR.mkdir(parents=True,exist_ok=True)
SIGLIP_MODEL_ID='google/siglip2-base-patch16-224'
TRAIN_PER_DOMAIN=int(globals().get('TRAIN_PER_DOMAIN',1500))

# Optionally clone source into a fresh Colab runtime; never overwrite local edits.
if not (SOURCE_ROOT/'abpr/core.py').is_file():
    if SOURCE_ROOT.exists():raise RuntimeError('Invalid SOURCE_ROOT exists: '+str(SOURCE_ROOT))
    print('Clone trainer source...')
    subprocess.run(['git','clone','--depth','1','https://github.com/VoThiKimTrang06101997/PAR_UPAR_ABPR.git',str(SOURCE_ROOT)],check=True)
if not (OFFICIAL_ROOT/'data/annotations/task2/val/gt.csv').is_file():
    raise FileNotFoundError('Missing official Task 2 VAL annotations; mount/prepare data with Cells 1–3 first: '+str(OFFICIAL_ROOT))
sys.path.insert(0,str(SOURCE_ROOT))
# Restore original helper exactly when a fresh runtime has lost Cell 7 variables.
if not (SOURCE_ROOT/'abpr_siglip2_pipeline.py').is_file():
    (SOURCE_ROOT/'abpr_siglip2_pipeline.py').write_bytes(zlib.decompress(base64.b64decode("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")))
import abpr_siglip2_pipeline as sp
qa_module=SOURCE_ROOT/'abpr/query_aware_ranking.py'
if not qa_module.is_file():
    qa_module.parent.mkdir(parents=True,exist_ok=True)
    qa_module.write_bytes(zlib.decompress(base64.b64decode("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")))
from abpr.core import ATTRIBUTE_NAMES,detect_image_column,infer_domain_id,ImageResolver
from abpr.model import load_query_csv
module_path=Path('/content/abpr_repr_learning.py')
module_path.parent.mkdir(parents=True,exist_ok=True)
module_path.write_bytes(zlib.decompress(base64.b64decode("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")))
sys.path.insert(0,str(module_path.parent))
import abpr_repr_learning as reprlearn

manifest_file=SIGLIP_DIR/'fixed_public_val_split.json'
if not manifest_file.is_file():raise FileNotFoundError('Run Cell 8 to create exact FIT/AUDIT split: '+str(manifest_file))
manifest=json.loads(manifest_file.read_text())
FIT_IDS=np.asarray(manifest['fit_ids'],np.int64)
AUDIT_IDS=np.asarray(manifest['audit_ids'],np.int64)
SELECT_IDS=np.r_[FIT_IDS,AUDIT_IDS]
assert len(set(SELECT_IDS))==len(SELECT_IDS)
ATTR=list(ATTRIBUTE_NAMES)
assert len(ATTR)==40 and len(set(ATTR))==40
VAL_CSV=OFFICIAL_ROOT/'data/annotations/task2/val/gt.csv'
TRAIN_CSV=OFFICIAL_ROOT/'data/annotations/task2/train/gt.csv'
assert TRAIN_CSV.is_file(),TRAIN_CSV
VAL_FRAME=pd.read_csv(VAL_CSV)
VAL_COL=detect_image_column(VAL_FRAME)
VAL_NAMES=VAL_FRAME[VAL_COL].astype(str).tolist()
VAL_LABELS=VAL_FRAME[ATTR].to_numpy(np.float32)
GIDS=sp.ids_from_official(OFFICIAL_ROOT/'data/annotations/task2/val/ids.csv',VAL_NAMES)
_,Q=load_query_csv(OFFICIAL_ROOT/'data/annotations/task2/val/queries.csv')
Q=np.asarray(Q,np.float32)
assert Q.shape[1]==40 and np.isin(Q,[-1,0,1]).all()
assert sp.digest_lines(VAL_NAMES)==manifest['gallery_paths_sha256'], 'VAL gallery/order differs from saved split'
assert manifest['gallery_count']==len(VAL_NAMES) and SELECT_IDS.max()<len(Q)

cache=SIGLIP_DIR/'baseline_fixed_queries.npy'
meta=SIGLIP_DIR/'baseline_fixed_queries.json'
assert cache.is_file() and meta.is_file(),'Run Cell 9 once to cache baseline distances'
BASE_DIST=np.load(cache,mmap_mode='r',allow_pickle=False)
assert BASE_DIST.shape==(len(SELECT_IDS),len(VAL_NAMES)),(BASE_DIST.shape,len(SELECT_IDS),len(VAL_NAMES))
assert np.isfinite(BASE_DIST).all(),'Corrupt baseline distances'
info=json.loads(meta.read_text())
assert info.get('shape',list(BASE_DIST.shape))==list(BASE_DIST.shape)
print('[32 PASS] Exact public VAL:',len(VAL_NAMES),'gallery; FIT',len(FIT_IDS),'AUDIT',len(AUDIT_IDS))
print('Baseline ZIP cache:',info.get('source_zip','(not recorded)'))
print('Representation output:',REPR_DIR)


#@title 33) TRAIN-ONLY representation data + strict feature-to-image cache signatures
from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
from tqdm.auto import tqdm

TRAIN_FRAME=pd.read_csv(TRAIN_CSV)
tr_col=detect_image_column(TRAIN_FRAME)
assert all(k in TRAIN_FRAME for k in ATTR)
TRAIN_SAMPLE=sp.balanced_sample(TRAIN_FRAME,tr_col,infer_domain_id,TRAIN_PER_DOMAIN,seed=42)
TRAIN_NAMES=TRAIN_SAMPLE[tr_col].astype(str).tolist()
TRAIN_LABELS=TRAIN_SAMPLE[ATTR].to_numpy(np.float32)
TRAIN_DOMAINS=np.asarray([infer_domain_id(s) for s in TRAIN_NAMES],dtype=np.int64)
assert np.isin(TRAIN_LABELS,[-1,0,1]).all()
assert np.array_equal(np.unique(TRAIN_DOMAINS),[0,1,2])

# No broad os.walk() over Drive. Use the same ImageResolver as Cell 10/11.
roots=[Path(OFFICIAL_ROOT)/'data',Path(OFFICIAL_ROOT),Path(SOURCE_ROOT),DRIVE_ROOT,
       DRIVE_ROOT/'data',DRIVE_ROOT/'datasets',DRIVE_ROOT/'ABPR_Data',Path('/content/ABPR_real_image_links')]
roots += [Path(x) for x in globals().get('EXTRA_DATA_ROOTS',[])]
RESOLVER=ImageResolver(OFFICIAL_ROOT/'data',OFFICIAL_ROOT,extra_roots=[r for r in roots if r.is_dir()])
if 'VAL_PATHS' not in globals() or len(VAL_PATHS)!=len(VAL_NAMES):
    VAL_PATHS=[str(RESOLVER.resolve(p)) for p in tqdm(VAL_NAMES,desc='Restore exact VAL path signature')]
if 'paths_train' in globals() and 'raw_train' in globals() and list(raw_train)==TRAIN_NAMES and len(paths_train)==len(TRAIN_NAMES):
    TRAIN_PATHS=[str(p) for p in paths_train]
else:
    TRAIN_PATHS=[str(RESOLVER.resolve(p)) for p in tqdm(TRAIN_NAMES,desc='Restore exact TRAIN path signature')]

def check_cache(p,ordered_paths):
    p=Path(p);met=p.with_suffix('.json')
    if not p.is_file() or not met.is_file():
        raise FileNotFoundError(f'{p}: missing feature file/metadata. Run Cells 10–11, no re-training needed.')
    j=json.loads(met.read_text())
    expected=sp.digest_lines([SIGLIP_MODEL_ID,len(ordered_paths),*ordered_paths])
    if j.get('signature')!=expected or j.get('completed')!=len(ordered_paths):
        raise RuntimeError(f'Image path ordering/model cache mismatch for {p}. Do not train on misaligned features. Restore original symlinks or rerun Cell 10/11.')
    arr=np.load(p,mmap_mode='r',allow_pickle=False)
    assert arr.shape[0]==len(ordered_paths) and arr.ndim==2 and np.isfinite(arr).all()
    return arr,j

TRAIN_FEAT,TRAIN_META=check_cache(SIGLIP_DIR/'train_siglip2.npy',TRAIN_PATHS)
SIG_VAL,VAL_META=check_cache(SIGLIP_DIR/'val_siglip2.npy',VAL_PATHS)
assert TRAIN_FEAT.shape[1]==SIG_VAL.shape[1]
assert sp.digest_lines(VAL_NAMES)==manifest['gallery_paths_sha256']
REPR_SIGNATURE=sp.digest_lines([SIGLIP_MODEL_ID,TRAIN_META['signature'],sp.digest_lines(ATTR),
                              hashlib.sha256(TRAIN_LABELS.tobytes()).hexdigest(),TRAIN_PER_DOMAIN])
print('[33 PASS] TRAIN features:',TRAIN_FEAT.shape,'VAL:',SIG_VAL.shape)
print('Source rows:',np.bincount(TRAIN_DOMAINS,minlength=3).tolist())
print('TRAIN known-label proportion:',round(float((TRAIN_LABELS!=-1).mean()),4))
print('All labels for TRAIN-only training; no VAL labels enter training.')


#@title 34) Train ONE lightweight image/query representation adapter — resumable TRAIN-only
REPR_EPOCHS=12  #@param {type:'integer'}
REPR_BATCH=192 #@param {type:'integer'}
REPR_LR=0.0007 #@param {type:'number'}
FORCE_RETRAIN=False #@param {type:'boolean'}

# Uses frozen 4500 SigLIP feature vectors, not a 33k-image GPU re-encoding job.
# Negative mining excludes unknown labels; queries vary from sparse to dense.
REPR_CHECKPOINT,REPR_HISTORY=reprlearn.fit_model(
    TRAIN_FEAT,TRAIN_LABELS,TRAIN_DOMAINS,REPR_DIR,REPR_SIGNATURE,
    epochs=REPR_EPOCHS,batch_size=REPR_BATCH,lr=REPR_LR,seed=2027,
    force=FORCE_RETRAIN)
assert REPR_CHECKPOINT.is_file()
print('TRAIN-only best representation checkpoint:',REPR_CHECKPOINT)
print('This checkpoint was selected by a disjoint TRAIN holdout, NOT public VAL.')


#@title 35) Encode public VAL with learned 128-D representation + attribute logits (resume)
import hashlib,json,os,torch,numpy as np
from pathlib import Path

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for piece in iter(lambda:f.read(4*1024*1024),b''):h.update(piece)
    return h.hexdigest()

REPR_MODEL,REPR_DEVICE,REPR_CK=reprlearn.load_model(REPR_CHECKPOINT)
assert REPR_CK['signature']==REPR_SIGNATURE
REPR_ART=REPR_DIR/'val_repr_features.npz'
REPR_META=REPR_DIR/'val_repr_features.json'
FEATURE_SIGNATURE=sp.digest_lines([hash_file(REPR_CHECKPOINT),VAL_META['signature'],sp.digest_lines(ATTR)])
use_cache=False
if REPR_ART.is_file() and REPR_META.is_file():
    use_cache=json.loads(REPR_META.read_text()).get('signature')==FEATURE_SIGNATURE
if use_cache:
    with np.load(REPR_ART,allow_pickle=False) as f:
        REPR_IMAGE=np.asarray(f['embedding'],dtype=np.float32)
        REPR_LOGITS=np.asarray(f['logits'],dtype=np.float32)
    print('[CACHE HIT] learned VAL representation')
else:
    REPR_IMAGE,REPR_LOGITS=reprlearn.image_outputs(REPR_MODEL,SIG_VAL,REPR_DEVICE,chunk=1024)
    tmp=REPR_ART.with_suffix('.partial.npz')
    np.savez_compressed(tmp,embedding=REPR_IMAGE,logits=REPR_LOGITS)
    os.replace(tmp,REPR_ART)
    REPR_META.write_text(json.dumps({'signature':FEATURE_SIGNATURE,'gallery_count':len(VAL_NAMES),
                                     'checkpoint':str(REPR_CHECKPOINT)},indent=2))
assert REPR_IMAGE.shape==(len(VAL_NAMES),128) and REPR_LOGITS.shape==(len(VAL_NAMES),40)
assert np.isfinite(REPR_IMAGE).all() and np.isfinite(REPR_LOGITS).all()
REPR_QUERY=reprlearn.query_outputs(REPR_MODEL,Q[SELECT_IDS],REPR_DEVICE)
REPR_SEM_DIST,REPR_ATTR_DIST=reprlearn.distances(REPR_QUERY,REPR_IMAGE,REPR_LOGITS,Q[SELECT_IDS])
assert REPR_SEM_DIST.shape==REPR_ATTR_DIST.shape==BASE_DIST.shape
assert np.isfinite(REPR_SEM_DIST).all() and np.isfinite(REPR_ATTR_DIST).all()
print('[35 PASS] Image representation:',REPR_IMAGE.shape,'attribute logits:',REPR_LOGITS.shape)
print('Candidate distance matrices:',REPR_SEM_DIST.shape)
print('No ground-truth VAL_LABELS/GIDS used by representation scoring.')


#@title 36) FIT-only evaluation — choose representation embedding/attribute/mix + fusion
import pandas as pd,numpy as np,json
from tqdm.auto import tqdm

nfit=len(FIT_IDS)
BF=np.asarray(BASE_DIST[:nfit],np.float32)
BASE_FIT=sp.semantic_metrics(BF,FIT_IDS,GIDS,Q[FIT_IDS],VAL_LABELS)
BASE_FIT={k:float(BASE_FIT[k]) for k in ('mADM','mAP','Rank-1')}
print('Baseline FIT:',BASE_FIT)
records=[]
configs=[(method,blend) for method in ('embedding','attribute','mix')
         for blend in (0.05,0.10,0.20,0.35)]
base_one=np.bincount(GIDS,minlength=len(Q))[FIT_IDS]==1
old_top=GIDS[np.argmin(BF,axis=1)]==FIT_IDS
for method,blend in tqdm(configs,desc='36 / FIT representation candidates',unit='config'):
    candidate=reprlearn.combine_scores(BF,REPR_SEM_DIST[:nfit],REPR_ATTR_DIST[:nfit],method,blend)
    scores=sp.semantic_metrics(candidate,FIT_IDS,GIDS,Q[FIT_IDS],VAL_LABELS)
    single_delta=(float((GIDS[np.argmin(candidate,axis=1)]==FIT_IDS)[base_one].mean())-
                  float(old_top[base_one].mean())) if base_one.any() else 0.0
    r={'method':method,'blend':float(blend),'mADM':float(scores['mADM']),
       'mAP':float(scores['mAP']),'Rank-1':float(scores['Rank-1']),
       'single_positive_delta_Rank1':single_delta}
    for k in ('mADM','mAP','Rank-1'):r['delta_'+k]=r[k]-BASE_FIT[k]
    r['fit_pass']=bool(r['delta_mADM']>=0.003 and r['delta_mAP']>=-0.005 and
                       r['delta_Rank-1']>=-0.008 and single_delta>=-0.012)
    records.append(r)
REPR_FIT=pd.DataFrame(records).sort_values(['mADM','mAP'],ascending=False).reset_index(drop=True)
REPR_FIT.to_csv(REPR_DIR/'repr_fit_trials.csv',index=False)
winners=REPR_FIT[REPR_FIT.fit_pass]
REPR_CHOICE=(winners.iloc[0] if len(winners) else REPR_FIT.iloc[0]).to_dict()
REPR_FIT_PASS=bool(REPR_CHOICE['fit_pass'])
(REPR_DIR/'repr_selected_fit.json').write_text(json.dumps({'choice':REPR_CHOICE,
  'fit_pass':REPR_FIT_PASS,'rule':'FIT-only, audit never used for candidate choice'},indent=2))
print('\nBEST FIT candidates:\n',REPR_FIT.head(8).to_string(index=False))
print('\nSelected:',REPR_CHOICE,'FIT gate:',REPR_FIT_PASS)


#@title 37) Public AUDIT — compare SAME-gallery retrieval + domain/single-positive gates
from abpr.query_aware_ranking import retrieval_metrics
from abpr.model import track2_metrics
import numpy as np,json,pandas as pd
from tqdm.auto import tqdm

BA=np.asarray(BASE_DIST[nfit:],np.float32)
REPR_AUD_SCORE=reprlearn.combine_scores(BA,REPR_SEM_DIST[nfit:],REPR_ATTR_DIST[nfit:],
                                REPR_CHOICE['method'],REPR_CHOICE['blend'])
baseline=sp.semantic_metrics(BA,AUDIT_IDS,GIDS,Q[AUDIT_IDS],VAL_LABELS)
candidate=sp.semantic_metrics(REPR_AUD_SCORE,AUDIT_IDS,GIDS,Q[AUDIT_IDS],VAL_LABELS)
baseline={k:float(baseline[k]) for k in ('mADM','mAP','Rank-1')}
candidate={k:float(candidate[k]) for k in ('mADM','mAP','Rank-1')}
pos_count=np.bincount(GIDS,minlength=len(Q))[AUDIT_IDS]
one=pos_count==1
old_r1=GIDS[np.argmin(BA,axis=1)]==AUDIT_IDS
new_r1=GIDS[np.argmin(REPR_AUD_SCORE,axis=1)]==AUDIT_IDS
single_base=float(old_r1[one].mean()) if one.any() else None
single_cand=float(new_r1[one].mean()) if one.any() else None
D=np.asarray([infer_domain_id(s) for s in VAL_NAMES])
domain_rows=[]
for d,domain in tqdm(enumerate(('Market1501','PA100k','PETA')),total=3,desc='37 / domain evaluation',unit='domain'):
    mask=D==d
    relevant=np.isin(AUDIT_IDS,np.unique(GIDS[mask]))
    ids=AUDIT_IDS[relevant]
    if not len(ids):continue
    for label,matrix in [('baseline',BA),('representation',REPR_AUD_SCORE)]:
        score=matrix[relevant][:,mask]
        metric=retrieval_metrics(score,ids,GIDS[mask])
        metric.update(track2_metrics(score,Q[ids],VAL_LABELS[mask]))
        domain_rows.append({'domain':domain,'method':label,'queries':len(ids),
                            'mADM':float(metric['mADM']),'mAP':float(metric['mAP']),
                            'Rank-1':float(metric['Rank-1'])})
REPR_DOMAIN=pd.DataFrame(domain_rows)
REPR_DOMAIN.to_csv(REPR_DIR/'repr_audit_domains.csv',index=False)
domain_ok=True
for domain in REPR_DOMAIN.domain.unique():
    a=REPR_DOMAIN[REPR_DOMAIN.domain==domain].set_index('method')
    domain_ok &= (a.loc['representation','mADM']>=a.loc['baseline','mADM']-0.01)
single_ok=single_base is None or single_cand>=single_base-0.007
REPR_AUDIT_ACCEPTED=bool(REPR_FIT_PASS and candidate['mADM']>=baseline['mADM']+0.005
    and candidate['mAP']>=baseline['mAP']-0.005 and candidate['Rank-1']>=baseline['Rank-1']-0.005
    and domain_ok and single_ok)
REPR_REPORT={'method':'TRAIN-only masked attribute–image representation + multi-positive hard-negative ranking',
  'source':'public VAL exploratory ONLY; never hidden result',
  'previous_audit_repeatedly_inspected':True,
  'baseline_fit':BASE_FIT,'selected_fit':REPR_CHOICE,'fit_pass':REPR_FIT_PASS,
  'baseline_audit':baseline,'representation_audit':candidate,
  'audit_gain_mADM':candidate['mADM']-baseline['mADM'],
  'audit_gain_mAP':candidate['mAP']-baseline['mAP'],
  'single_positive_queries':int(one.sum()),'single_positive_Rank1_baseline':single_base,
  'single_positive_Rank1_candidate':single_cand,
  'domain_noninferior':bool(domain_ok),'single_positive_noninferior':bool(single_ok),
  'audit_accepted':REPR_AUDIT_ACCEPTED,
  'external_pretrained_model':'google/siglip2-base-patch16-224',
  'automatic_submission_allowed':False,
  'note':'No submission ZIP: requires challenge pretrained weights permission and a fully offline end-to-end gallery encoding runtime.'}
(REPR_DIR/'repr_audit_report.json').write_text(json.dumps(REPR_REPORT,indent=2,ensure_ascii=False))
print('AUDIT baseline:',baseline)
print('AUDIT representation:',candidate)
print('Single-positive R1:',single_base,'->',single_cand)
print('Domain breakdown:\n',REPR_DOMAIN.to_string(index=False))
print('REPR_AUDIT_ACCEPTED:',REPR_AUDIT_ACCEPTED)


#@title 38) Export representation-learning experiment report — NOT a Codabench submission ZIP
from pathlib import Path
import json,hashlib,zipfile
from tqdm.auto import tqdm
README=REPR_DIR/'README_submission_gate.txt'
README.write_text("""TRAIN-only representation fine-tuning of frozen SigLIP2 feature vectors.
The new image and query representation is 128-dimensional; unknown query/label
attributes (-1) are masked. Candidate selection was on public FIT only.
The public AUDIT has been repeatedly inspected during earlier experiments,
therefore it is exploratory, not an untouched independent validation set.
NO gallery ground truth or query-relevance IDs were used to train the adapter
or score gallery images. The Codabench hidden result was NOT measured.
Before submission: confirm SigLIP2 external pretrained weights eligibility,
package model and preprocessing offline, regenerate hidden gallery features,
check 40 attribute ordering, run an actual public ingestion smoke test.
Do not upload this experiment report ZIP as a Codabench submission.
""",encoding='utf-8')
items=[REPR_DIR/'repr_fit_trials.csv',REPR_DIR/'repr_selected_fit.json',
       REPR_DIR/'repr_audit_report.json',REPR_DIR/'repr_audit_domains.csv',
       REPR_DIR/'repr_training_history.json',Path(reprlearn.__file__),README]
REPR_REPORT_ZIP=REPR_DIR/'ABPR_Representation_Learning_Report.zip'
with zipfile.ZipFile(REPR_REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in tqdm(items,desc='38 / export experiment report',unit='file'):
        assert f.is_file(),f
        z.write(f,arcname=f.name)
print('REPORT:',REPR_REPORT_ZIP)
print('Fitted on TRAIN, selected on FIT; public AUDIT:',REPR_REPORT['representation_audit'])
if REPR_AUDIT_ACCEPTED:
    print('Public candidate promising; NOT submit-ready without legal offline SigLIP2 runtime.')
else:
    print('KEEP existing 0.5389 hidden baseline. Representation candidate failed gate.')
print('Send the report ZIP here for next decision.')
