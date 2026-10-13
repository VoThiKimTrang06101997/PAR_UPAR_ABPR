# ## Dense-Query / Single-Positive-Aware Representation Learning — continuation (Cells 39–45)

**Goal:** After the earlier Hybrid Prototype / SigLIP2 / HATA / TCR experiments, train **one** new TRAIN-only image/query projection head on cached frozen SigLIP2 features. Increase query density, use explicit-conflict hard negatives, balanced-source batches, and a paired-image top-1 objective. No VAL labels or ID relevance map enter training or scoring. Evaluation uses the same original baseline distance matrix and the same frozen FIT/AUDIT query IDs.

**Caveats:** This is an experimental method; it is **not guaranteed** to improve Codabench hidden mADM. The public AUDIT has been inspected repeatedly in previous experiments; all gains are exploratory. An experiment ZIP is not an eligible submission ZIP. SigLIP2 eligibility and a fully offline runner must be established separately. **Do not upload report ZIP to Codabench.**

#@title 39) Restore exact TRAIN/VAL metadata + install dense-query method (NO DATA DOWNLOAD)
import sys,os,json,hashlib,zlib,base64,importlib
from pathlib import Path
import numpy as np,pandas as pd
from tqdm.auto import tqdm

SOURCE_ROOT=Path(globals().get('SOURCE_ROOT','/content/PAR_UPAR_ABPR'))
OFFICIAL_ROOT=Path(globals().get('OFFICIAL_ROOT','/content/UPAR-Challenge-2027'))
DRIVE_ROOT=Path(globals().get('DRIVE_ROOT','/content/drive/MyDrive/PedestrianAttributeRecognition'))
RESULT_DIR=Path(globals().get('RESULT_DIR',DRIVE_ROOT/'ABPR_Results'))
SIGLIP_DIR=Path(globals().get('SIGLIP_DIR',RESULT_DIR/'siglip2_retrieval'))
SIGLIP_MODEL_ID=globals().get('SIGLIP_MODEL_ID','google/siglip2-base-patch16-224')
TRAIN_PER_DOMAIN=int(globals().get('TRAIN_PER_DOMAIN',1500))
DENSE_DIR=SIGLIP_DIR/'densequery_repr_method';DENSE_DIR.mkdir(parents=True,exist_ok=True)
assert SOURCE_ROOT.is_dir(),f'Source missing: {SOURCE_ROOT}. Run Cell 1 first.'
sys.path.insert(0,str(SOURCE_ROOT))

# Self-contained new source; does not touch original repository or trainer.
raw=zlib.decompress(base64.b64decode('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'))
module_file=DENSE_DIR/'abpr_densequery_repr.py'
if not module_file.is_file() or module_file.read_bytes()!=raw:
    module_file.write_bytes(raw)
sys.path.insert(0,str(DENSE_DIR))
import abpr_densequery_repr as dense
importlib.reload(dense)

from abpr.core import ATTRIBUTE_NAMES,detect_image_column,infer_domain_id,ImageResolver
from abpr.model import load_query_csv
try:
    import abpr_siglip2_pipeline as sp
except ImportError as exc:
    raise RuntimeError('Module abpr_siglip2_pipeline missing. Run earlier Cell 7 first (no need to retrain).') from exc

ATTR=list(ATTRIBUTE_NAMES)
assert len(ATTR)==40 and len(set(ATTR))==40
VAL_CSV=OFFICIAL_ROOT/'data/annotations/task2/val/gt.csv'
TRAIN_CSV=OFFICIAL_ROOT/'data/annotations/task2/train/gt.csv'
VAL_FRAME=pd.read_csv(VAL_CSV)
VAL_COL=detect_image_column(VAL_FRAME)
VAL_NAMES=VAL_FRAME[VAL_COL].astype(str).tolist()
VAL_LABELS=VAL_FRAME[ATTR].to_numpy(np.float32)
_,Q=load_query_csv(OFFICIAL_ROOT/'data/annotations/task2/val/queries.csv')
Q=np.asarray(Q,np.float32)
GIDS=sp.ids_from_official(OFFICIAL_ROOT/'data/annotations/task2/val/ids.csv',VAL_NAMES)

split_file=SIGLIP_DIR/'fixed_public_val_split.json'
if not split_file.is_file():raise FileNotFoundError(f'Missing FIT/AUDIT split: {split_file}. Run Cell 8.')
manifest=json.loads(split_file.read_text(encoding='utf-8'))
FIT_IDS=np.asarray(manifest['fit_ids'],np.int64);AUDIT_IDS=np.asarray(manifest['audit_ids'],np.int64)
SELECT_IDS=np.r_[FIT_IDS,AUDIT_IDS]
assert sp.digest_lines(VAL_NAMES)==manifest['gallery_paths_sha256'],'Original gallery ordering changed!'
assert len(set(SELECT_IDS))==len(SELECT_IDS) and GIDS.shape==(len(VAL_NAMES),)
assert Q.ndim==2 and Q.shape[1]==40 and VAL_LABELS.shape==(len(VAL_NAMES),40)

base_file=SIGLIP_DIR/'baseline_fixed_queries.npy'
assert base_file.is_file(),f'Run Cell 9 to generate exact submission baseline cache: {base_file}'
BASE_DIST=np.load(base_file,mmap_mode='r',allow_pickle=False)
assert BASE_DIST.shape==(len(SELECT_IDS),len(VAL_NAMES)) and np.isfinite(BASE_DIST).all()

# TRAIN sampling must be IDENTICAL to original Cell 11/33 for feature-cache alignment.
TRAIN_FRAME=pd.read_csv(TRAIN_CSV)
tr_col=detect_image_column(TRAIN_FRAME)
TRAIN_SAMPLE=sp.balanced_sample(TRAIN_FRAME,tr_col,infer_domain_id,TRAIN_PER_DOMAIN,seed=42)
TRAIN_NAMES=TRAIN_SAMPLE[tr_col].astype(str).tolist()
TRAIN_LABELS=TRAIN_SAMPLE[ATTR].to_numpy(np.float32)
TRAIN_DOMAINS=np.asarray([infer_domain_id(p) for p in TRAIN_NAMES],np.int64)
assert np.isin(TRAIN_LABELS,[-1,0,1]).all() and np.isin(TRAIN_DOMAINS,[0,1,2]).all()

roots=[OFFICIAL_ROOT/'data',OFFICIAL_ROOT,SOURCE_ROOT,DRIVE_ROOT,DRIVE_ROOT/'data',
       DRIVE_ROOT/'datasets',DRIVE_ROOT/'ABPR_Data',Path('/content/ABPR_real_image_links')]
roots.extend(Path(x) for x in globals().get('EXTRA_DATA_ROOTS',[]))
RESOLVER=ImageResolver(OFFICIAL_ROOT/'data',OFFICIAL_ROOT,extra_roots=[x for x in roots if x.is_dir()])

def strict_cached_features(cache,paths):
    cache=Path(cache);meta=cache.with_suffix('.json')
    assert cache.is_file() and meta.is_file(),f'Run Cell 10/11 to recreate missing cache: {cache}'
    entry=json.loads(meta.read_text(encoding='utf-8'))
    expected=sp.digest_lines([SIGLIP_MODEL_ID,len(paths),*paths])
    if entry.get('signature')!=expected or entry.get('completed')!=len(paths):
        raise RuntimeError(f'SigLIP model/path signature mismatch: {cache}. Do not pair features with incorrect images.')
    data=np.load(cache,mmap_mode='r',allow_pickle=False)
    assert data.ndim==2 and data.shape[0]==len(paths) and np.isfinite(data).all()
    return data,entry

if 'VAL_PATHS' not in globals() or len(VAL_PATHS)!=len(VAL_NAMES):
    VAL_PATHS=[str(RESOLVER.resolve(s)) for s in tqdm(VAL_NAMES,desc='39 / Restore true gallery paths')]
if 'TRAIN_PATHS' not in globals() or len(TRAIN_PATHS)!=len(TRAIN_NAMES):
    TRAIN_PATHS=[str(RESOLVER.resolve(s)) for s in tqdm(TRAIN_NAMES,desc='39 / Restore TRAIN paths')]
TRAIN_FEAT,TRAIN_META=strict_cached_features(SIGLIP_DIR/'train_siglip2.npy',TRAIN_PATHS)
SIG_VAL,VAL_META=strict_cached_features(SIGLIP_DIR/'val_siglip2.npy',VAL_PATHS)
assert TRAIN_FEAT.shape[1]==SIG_VAL.shape[1]
DENSE_SIGNATURE=sp.digest_lines([SIGLIP_MODEL_ID,TRAIN_META['signature'],VAL_META['signature'],
    sp.digest_lines(ATTR),hashlib.sha256(TRAIN_LABELS.tobytes()).hexdigest(),TRAIN_PER_DOMAIN,
    hashlib.sha256(raw).hexdigest()])
print('[39 PASS] TRAIN:',TRAIN_FEAT.shape,'VAL:',SIG_VAL.shape,'FIT/AUDIT:',len(FIT_IDS),len(AUDIT_IDS))
print('Using original best submission distances:',BASE_DIST.shape)
print('No re-encoding SigLIP2, no source modifications, no VAL labels used in training.')


#@title 40) TRAIN-ONLY dense-query feature-level representation (resumable, GPU optional)
DENSE_EPOCHS=12  #@param {type:'integer'}
DENSE_BATCH=192   #@param {type:'integer'}
DENSE_LR=0.0005   #@param {type:'number'}
DENSE_FORCE_RETRAIN=False #@param {type:'boolean'}

# This is a NEW head trained on cached frozen SigLIP2 image features.
# It does NOT fine-tune the original ConvNeXt/SigLIP2 visual backbone.
DENSE_CKPT,DENSE_HISTORY=dense.fit(
    TRAIN_FEAT,TRAIN_LABELS,TRAIN_DOMAINS,DENSE_DIR,DENSE_SIGNATURE,
    epochs=DENSE_EPOCHS,batch_size=DENSE_BATCH,lr=DENSE_LR,
    seed=2027,force=DENSE_FORCE_RETRAIN)
assert DENSE_CKPT.is_file()
print('[40 PASS] TRAIN-only selected checkpoint:',DENSE_CKPT)
print('[TRAIN holdout] Last epoch:',DENSE_HISTORY[-1])


#@title 41) Produce NEW 128-D embeddings and per-attribute evidence (NO new image inference)
import hashlib,os,json,numpy as np

def hash_file_stream(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()

DENSE_MODEL,DENSE_DEVICE,DENSE_CK=dense.load_model(DENSE_CKPT)
assert DENSE_CK['signature']==DENSE_SIGNATURE
CACHE_FILE=DENSE_DIR/'densequery_val_outputs.npz'
CACHE_META=DENSE_DIR/'densequery_val_outputs.json'
OUTPUT_SIGNATURE=sp.digest_lines([hash_file_stream(DENSE_CKPT),VAL_META['signature'],sp.digest_lines(ATTR)])
using_cache=(CACHE_FILE.is_file() and CACHE_META.is_file() and
    json.loads(CACHE_META.read_text()).get('signature')==OUTPUT_SIGNATURE)
if using_cache:
    with np.load(CACHE_FILE,allow_pickle=False) as arr:
        DENSE_IMG=np.asarray(arr['image'],np.float32)
        DENSE_LOGITS=np.asarray(arr['logits'],np.float32)
    print('[CACHE HIT] exact 33k-gallery projection')
else:
    DENSE_IMG,DENSE_LOGITS=dense.image_outputs(DENSE_MODEL,SIG_VAL,DENSE_DEVICE)
    partial=CACHE_FILE.with_name(CACHE_FILE.stem+'.partial.npz')
    np.savez_compressed(partial,image=DENSE_IMG,logits=DENSE_LOGITS)
    os.replace(partial,CACHE_FILE)
    CACHE_META.write_text(json.dumps({'signature':OUTPUT_SIGNATURE,
        'gallery_count':len(VAL_NAMES),'model':str(DENSE_CKPT)},indent=2))
assert DENSE_IMG.shape==(len(VAL_NAMES),128)
assert DENSE_LOGITS.shape==(len(VAL_NAMES),40)

known=np.isin(TRAIN_LABELS,[0,1]);freq=(TRAIN_LABELS==1).sum(0)/np.maximum(known.sum(0),1)
# All evidence weights are computed solely from TRAIN labels, never VAL.
DENSE_ATTR_WEIGHTS=np.clip((np.maximum(freq,0.01)+0.02)**(-0.20),0.6,1.7).astype('float32')
DENSE_Q_EMBED=dense.query_outputs(DENSE_MODEL,Q[SELECT_IDS],DENSE_DEVICE)
DENSE_SEM,DENSE_EVID=dense.distances(DENSE_Q_EMBED,DENSE_IMG,DENSE_LOGITS,
    Q[SELECT_IDS],DENSE_ATTR_WEIGHTS)
assert DENSE_SEM.shape==DENSE_EVID.shape==BASE_DIST.shape
assert np.isfinite(DENSE_SEM).all() and np.isfinite(DENSE_EVID).all()
print('[41 PASS] New distances:',DENSE_SEM.shape,
      '| new representation logits:',DENSE_LOGITS.shape)


#@title 42) Select ONE candidate on public FIT (protect Rank-1 and single-positive)
import json,numpy as np,pandas as pd
from tqdm.auto import tqdm

N_FIT=len(FIT_IDS)
BASE_FIT_SCORE=np.asarray(BASE_DIST[:N_FIT],np.float32)
base_metrics=sp.semantic_metrics(BASE_FIT_SCORE,FIT_IDS,GIDS,Q[FIT_IDS],VAL_LABELS)
DENSE_FIT_BASE={k:float(base_metrics[k]) for k in ('mADM','mAP','Rank-1')}
num_pos=np.bincount(GIDS,minlength=len(Q))[FIT_IDS]
ones=num_pos==1
old_r1=(GIDS[np.argmin(BASE_FIT_SCORE,axis=1)]==FIT_IDS)
old_single=float(old_r1[ones].mean()) if ones.any() else float('nan')

trials=[]
for method in ('evidence','semantic','hybrid'):
    for alpha in tqdm((0.025,0.05,0.08,0.12,0.18),desc='42 / FIT '+method,unit='weight'):
        candidate=dense.fusion(BASE_FIT_SCORE,DENSE_SEM[:N_FIT],DENSE_EVID[:N_FIT],method,alpha)
        m=sp.semantic_metrics(candidate,FIT_IDS,GIDS,Q[FIT_IDS],VAL_LABELS)
        new_r1=(GIDS[np.argmin(candidate,axis=1)]==FIT_IDS)
        new_single=float(new_r1[ones].mean()) if ones.any() else float('nan')
        delta={k:float(m[k])-DENSE_FIT_BASE[k] for k in ('mADM','mAP','Rank-1')}
        gate=bool(delta['mADM']>=0.003 and delta['mAP']>=-0.003 and
                  delta['Rank-1']>=-0.0042 and (not ones.any() or new_single>=old_single))
        trials.append({'method':method,'alpha':float(alpha),
            **{k:float(m[k]) for k in ('mADM','mAP','Rank-1')},
            'gain_mADM':delta['mADM'],'gain_mAP':delta['mAP'],
            'gain_Rank1':delta['Rank-1'],
            'single_positive_Rank1':new_single,
            'single_positive_delta':new_single-old_single,
            'fit_gate':gate})

DENSE_FIT_TRIALS=pd.DataFrame(trials).sort_values(['mADM','mAP'],ascending=False).reset_index(drop=True)
DENSE_FIT_TRIALS.to_csv(DENSE_DIR/'densequery_fit_trials.csv',index=False)
eligible=DENSE_FIT_TRIALS[DENSE_FIT_TRIALS.fit_gate]
DENSE_FIT_PASSED=bool(len(eligible))
DENSE_CHOICE=(eligible.iloc[0] if len(eligible) else DENSE_FIT_TRIALS.iloc[0]).to_dict()
(DENSE_DIR/'densequery_choice.json').write_text(json.dumps({
    'baseline_fit':DENSE_FIT_BASE,'baseline_single_R1':old_single,
    'candidate':DENSE_CHOICE,'fit_gate':DENSE_FIT_PASSED,
    'method_selection':'FIT only. AUDIT must not be used to select alpha.'},indent=2))
print('[FIT baseline]',DENSE_FIT_BASE)
print('Top FIT trials:\n',DENSE_FIT_TRIALS.head(8).to_string(index=False))
print('[SELECTED]',DENSE_CHOICE,'FIT_GATE:',DENSE_FIT_PASSED)


#@title 43) PUBLIC AUDIT — exact gallery + single-positive/domain rejection gates
import numpy as np,pandas as pd,json
from tqdm.auto import tqdm
from abpr.query_aware_ranking import retrieval_metrics
from abpr.model import track2_metrics

BA=np.asarray(BASE_DIST[N_FIT:],np.float32)
CAND=dense.fusion(BA,DENSE_SEM[N_FIT:],DENSE_EVID[N_FIT:],
                  DENSE_CHOICE['method'],DENSE_CHOICE['alpha'])
# Always evaluate the actual new candidate, EVEN IF it failed FIT.
# This avoids the previous misleading baseline fallback reporting an equal score.
bm=sp.semantic_metrics(BA,AUDIT_IDS,GIDS,Q[AUDIT_IDS],VAL_LABELS)
cm=sp.semantic_metrics(CAND,AUDIT_IDS,GIDS,Q[AUDIT_IDS],VAL_LABELS)
DENSE_AUD_BASE={k:float(bm[k]) for k in ('mADM','mAP','Rank-1')}
DENSE_AUD_CAND={k:float(cm[k]) for k in ('mADM','mAP','Rank-1')}

one=(np.bincount(GIDS,minlength=len(Q))[AUDIT_IDS]==1)
r1b=(GIDS[np.argmin(BA,axis=1)]==AUDIT_IDS)
r1c=(GIDS[np.argmin(CAND,axis=1)]==AUDIT_IDS)
old_single=float(r1b[one].mean()) if one.any() else None
new_single=float(r1c[one].mean()) if one.any() else None

D=np.array([infer_domain_id(s) for s in VAL_NAMES])
rows=[]
for d,domain in tqdm(enumerate(('Market1501','PA100k','PETA')),total=3,
                      desc='43 / AUDIT domains',unit='domain'):
    mask=(D==d);subset=np.isin(AUDIT_IDS,np.unique(GIDS[mask]));ids=AUDIT_IDS[subset]
    if not len(ids):continue
    for label,mat in [('baseline',BA),('densequery',CAND)]:
        sub=mat[subset][:,mask]
        m=retrieval_metrics(sub,ids,GIDS[mask]);m.update(track2_metrics(sub,Q[ids],VAL_LABELS[mask]))
        rows.append({'domain':domain,'method':label,'queries':len(ids),
            **{k:float(m[k]) for k in ('mADM','mAP','Rank-1')}})
DENSE_DOMAIN_REPORT=pd.DataFrame(rows)
DENSE_DOMAIN_REPORT.to_csv(DENSE_DIR/'densequery_audit_domains.csv',index=False)
worst_domain_ok=True
for domain in DENSE_DOMAIN_REPORT.domain.unique():
    a=DENSE_DOMAIN_REPORT[DENSE_DOMAIN_REPORT.domain==domain].set_index('method')
    worst_domain_ok &= bool(a.loc['densequery','mADM']>=a.loc['baseline','mADM']-0.003)

DENSE_AUDIT_ACCEPTED=bool(DENSE_FIT_PASSED and
    DENSE_AUD_CAND['mADM']>=DENSE_AUD_BASE['mADM']+0.005 and
    DENSE_AUD_CAND['mAP']>=DENSE_AUD_BASE['mAP']-0.003 and
    DENSE_AUD_CAND['Rank-1']>=DENSE_AUD_BASE['Rank-1'] and
    (old_single is None or new_single>=old_single) and worst_domain_ok)
DENSE_REPORT={
    'method':'TRAIN-only Dense-Query Single-Positive-Aware Representation Learning',
    'basis':'cached frozen SigLIP 2, train domain balance, explicit negative masks, multi-positive and paired-image hard loss',
    'baseline_audit':DENSE_AUD_BASE,'candidate_audit':DENSE_AUD_CAND,
    'best_fit':DENSE_CHOICE,'fit_pass':DENSE_FIT_PASSED,
    'single_positive_audit_queries':int(one.sum()),
    'single_positive_Rank1_baseline':old_single,'single_positive_Rank1_candidate':new_single,
    'domain_noninferior':bool(worst_domain_ok),'audit_accepted':DENSE_AUDIT_ACCEPTED,
    'public_audit_already_inspected_in_previous_experiments':True,
    'external_pretrained_model':SIGLIP_MODEL_ID,
    'codabench_submission_ready':False,
    'warning':'Public exploratory AUDIT only; hidden performance unknown. Offline SigLIP weights and runtime legality not verified.'}
(DENSE_DIR/'densequery_audit_report.json').write_text(json.dumps(DENSE_REPORT,indent=2,ensure_ascii=False))
print('PUBLIC baseline:',DENSE_AUD_BASE)
print('NEW representation:',DENSE_AUD_CAND)
print('Single-positive R1:',old_single,'->',new_single)
print(DENSE_DOMAIN_REPORT.to_string(index=False))
print('DENSE_AUDIT_ACCEPTED:',DENSE_AUDIT_ACCEPTED)


#@title 44) Audit investigation — per-query AP, Top-1 recoveries + failure groups
from tqdm.auto import tqdm
import pandas as pd,numpy as np

def exact_per_query_ap(d,ids,gids):
    # positive defined by official semantic IDs. ground truth is ONLY for evaluation.
    res=[]
    for i in tqdm(range(len(ids)),desc='44 / AP diagnostics',unit='query'):
        order=np.argsort(d[i],kind='stable')
        relevance=(gids[order]==ids[i])
        n=int(relevance.sum())
        positions=np.flatnonzero(relevance)
        ap=float((np.arange(1,n+1)/(positions+1)).mean()) if n else 0.
        res.append((ap,int(relevance[0]) if len(relevance) else 0,n))
    return np.array(res)

PB=exact_per_query_ap(BA,AUDIT_IDS,GIDS)
PC=exact_per_query_ap(CAND,AUDIT_IDS,GIDS)
DENSE_QUERY_DETAILS=pd.DataFrame({
    'query_id':AUDIT_IDS,
    'positives_in_gallery':PB[:,2].astype('int32'),
    'baseline_AP':PB[:,0],'candidate_AP':PC[:,0],
    'delta_AP':PC[:,0]-PB[:,0],
    'baseline_R1':PB[:,1].astype('int32'),
    'candidate_R1':PC[:,1].astype('int32'),
})
DENSE_QUERY_DETAILS['single_positive']=DENSE_QUERY_DETAILS.positives_in_gallery.eq(1)
DENSE_QUERY_DETAILS['recovered_top1']=(DENSE_QUERY_DETAILS.baseline_R1==0)&(DENSE_QUERY_DETAILS.candidate_R1==1)
DENSE_QUERY_DETAILS['lost_top1']=(DENSE_QUERY_DETAILS.baseline_R1==1)&(DENSE_QUERY_DETAILS.candidate_R1==0)
DENSE_QUERY_DETAILS.to_csv(DENSE_DIR/'densequery_query_errors.csv',index=False)
print('Recovered Top-1:',int(DENSE_QUERY_DETAILS.recovered_top1.sum()),
      'Lost Top-1:',int(DENSE_QUERY_DETAILS.lost_top1.sum()))
print('Single-positive recover/loss:',int(DENSE_QUERY_DETAILS.query('single_positive').recovered_top1.sum()),
      int(DENSE_QUERY_DETAILS.query('single_positive').lost_top1.sum()))
print('Worst AP drops:\n',DENSE_QUERY_DETAILS.nsmallest(10,'delta_AP').to_string(index=False))


#@title 45) Export experiment ZIP (REPORT ONLY, not Codabench submission)
import zipfile,hashlib,json
from pathlib import Path
from tqdm.auto import tqdm

readme=DENSE_DIR/'README_SUBMISSION_RESTRICTIONS.txt'
readme.write_text("This is a PUBLIC-VALIDATION EXPERIMENT REPORT, NOT A CODABENCH SUBMISSION.\n"
    "No hidden Codabench evaluation was performed. AUDIT was repeatedly inspected.\n"
    "Training uses TRAIN only; FIT selects, AUDIT evaluates.\n"
    "External SigLIP2 eligibility and a complete offline Codabench runtime remain unverified.\n"
    "Keep the previous 0.5389 hidden scoring submission.\n", encoding='utf-8')
items=[
    DENSE_DIR/'densequery_fit_trials.csv',
    DENSE_DIR/'densequery_choice.json',
    DENSE_DIR/'densequery_audit_report.json',
    DENSE_DIR/'densequery_audit_domains.csv',
    DENSE_DIR/'densequery_query_errors.csv',
    DENSE_DIR/'densequery_training_history.json',
    DENSE_DIR/'abpr_densequery_repr.py',readme]
assert all(p.is_file() for p in items)
DENSE_REPORT_ZIP=DENSE_DIR/'ABPR_DenseQuery_Experiment_Report.zip'
with zipfile.ZipFile(DENSE_REPORT_ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in tqdm(items,desc='45 / Package reports',unit='file'):
        z.write(p,arcname=p.name)
print('RESULT REPORT:',DENSE_REPORT_ZIP)
print('PUBLIC AUDIT ACCEPTED:',DENSE_AUDIT_ACCEPTED)
print('UPLOAD TO CODABENCH: NO — experiment ZIP is not a self-contained run.py submission')
print('Recommended: keep proven hidden baseline. Share this report for further source-level changes.')
