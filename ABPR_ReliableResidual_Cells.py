#@title 46) Continuation preflight — original DenseQuery model/cache, exact FIT/AUDIT alignment
from pathlib import Path
import json, hashlib, os, sys, base64, zlib, importlib, zipfile
import numpy as np, pandas as pd
from tqdm.auto import tqdm

SOURCE_ROOT=Path(globals().get('SOURCE_ROOT','/content/PAR_UPAR_ABPR'))
OFFICIAL_ROOT=Path(globals().get('OFFICIAL_ROOT','/content/UPAR-Challenge-2027'))
DRIVE_ROOT=Path(globals().get('DRIVE_ROOT','/content/drive/MyDrive/PedestrianAttributeRecognition'))
SIGLIP_DIR=Path(globals().get('SIGLIP_DIR',DRIVE_ROOT/'ABPR_Results/siglip2_retrieval'))
DENSE_DIR=SIGLIP_DIR/'densequery_repr_method'
RESIDUAL_DIR=SIGLIP_DIR/'reliable_residual_method'
RESIDUAL_DIR.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(SOURCE_ROOT));sys.path.insert(0,str(DENSE_DIR));sys.path.insert(0,str(RESIDUAL_DIR))

source=zlib.decompress(base64.b64decode('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'))
assert hashlib.sha256(source).hexdigest()=='f3d8f062556aa32c65fbb8bf85e95a2d2b40308f961fa8b15d1e301c35156ce5'
module_file=RESIDUAL_DIR/'abpr_reliable_residual.py'
if not module_file.exists() or module_file.read_bytes()!=source: module_file.write_bytes(source)
import abpr_reliable_residual as rr
importlib.reload(rr)
from abpr.core import ATTRIBUTE_NAMES,detect_image_column,infer_domain_id
from abpr.model import load_query_csv
try:
 import abpr_densequery_repr as dense
 import abpr_siglip2_pipeline as sp
except ImportError as exc:
 raise RuntimeError('Missing earlier source files. Run Cell 39 first; no training required.') from exc

ATTR=list(ATTRIBUTE_NAMES)
assert len(ATTR)==40
val_csv=OFFICIAL_ROOT/'data/annotations/task2/val/gt.csv'
val_frame=pd.read_csv(val_csv)
VAL_NAMES=val_frame[detect_image_column(val_frame)].astype(str).tolist()
VAL_LABELS=val_frame[ATTR].to_numpy('float32')  # ONLY for evaluation, NEVER scoring
_,Q=load_query_csv(OFFICIAL_ROOT/'data/annotations/task2/val/queries.csv')
Q=np.asarray(Q,'float32')
GIDS=sp.ids_from_official(OFFICIAL_ROOT/'data/annotations/task2/val/ids.csv',VAL_NAMES)
manifest_file=SIGLIP_DIR/'fixed_public_val_split.json'
assert manifest_file.is_file(),'Run Cell 8 first: fixed split missing'
manifest=json.loads(manifest_file.read_text())
FIT_IDS=np.asarray(manifest['fit_ids'],np.int64)
AUDIT_IDS=np.asarray(manifest['audit_ids'],np.int64)
SELECT_IDS=np.r_[FIT_IDS,AUDIT_IDS]
assert len(np.unique(SELECT_IDS))==len(SELECT_IDS)
assert sp.digest_lines(VAL_NAMES)==manifest['gallery_paths_sha256'], 'Gallery order changed!'
BASE_DIST=np.load(SIGLIP_DIR/'baseline_fixed_queries.npy',mmap_mode='r',allow_pickle=False)
assert BASE_DIST.shape==(len(SELECT_IDS),len(VAL_NAMES))
assert np.isfinite(BASE_DIST).all()

# DenseQuery checkpoint, exact VAL outputs from previous Cell 40/41.
DENSE_CKPT=DENSE_DIR/'densequery_best.pt'
assert DENSE_CKPT.is_file(), 'Run Cell 40 (train DenseQuery) first'
output_file=DENSE_DIR/'densequery_val_outputs.npz'
output_meta=DENSE_DIR/'densequery_val_outputs.json'
assert output_file.is_file() and output_meta.is_file(),'Run Cell 41 first'
DENSE_MODEL,DENSE_DEVICE,DENSE_CK=dense.load_model(DENSE_CKPT)
def sha256_stream(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
val_meta_file=SIGLIP_DIR/'val_siglip2.json'
assert val_meta_file.is_file(),val_meta_file
VAL_META=json.loads(val_meta_file.read_text())
expected=sp.digest_lines([sha256_stream(DENSE_CKPT),VAL_META['signature'],sp.digest_lines(ATTR)])
assert json.loads(output_meta.read_text())['signature']==expected,'Stale DenseQuery gallery cache! Run Cell 41'
with np.load(output_file,allow_pickle=False) as a:
 DENSE_IMG=np.asarray(a['image'],'float32')
 DENSE_LOGITS=np.asarray(a['logits'],'float32')
assert DENSE_IMG.shape==(len(VAL_NAMES),128) and DENSE_LOGITS.shape==(len(VAL_NAMES),40)
DENSE_Q_EMBED=dense.query_outputs(DENSE_MODEL,Q[SELECT_IDS],DENSE_DEVICE)
DENSE_SEM=-(np.asarray(DENSE_Q_EMBED,'float32') @ DENSE_IMG.T)
assert DENSE_SEM.shape==BASE_DIST.shape and np.isfinite(DENSE_SEM).all()

# TRAIN labels and features. If existing runtime was reset, rerun Cell 39
# to recover TRAIN_FEAT with checked full image-path signatures (no training).
required=('TRAIN_FEAT','TRAIN_LABELS','TRAIN_DOMAINS','TRAIN_META')
missing=[k for k in required if k not in globals()]
if missing: raise RuntimeError(f'Missing {missing}. Run Cell 39 again, then rerun Cell 46 (NO train/inference).')
TRAIN_FEAT=np.asarray(TRAIN_FEAT,'float32')
TRAIN_LABELS=np.asarray(TRAIN_LABELS,'float32')
TRAIN_DOMAINS=np.asarray(TRAIN_DOMAINS,'int64')
assert len(TRAIN_FEAT)==len(TRAIN_LABELS)==len(TRAIN_DOMAINS)
assert TRAIN_LABELS.shape[1]==40 and np.isin(TRAIN_LABELS,[-1,0,1]).all()
assert np.isin(TRAIN_DOMAINS,[0,1,2]).all()
assert TRAIN_META['signature']==json.loads((SIGLIP_DIR/'train_siglip2.json').read_text())['signature']
RESIDUAL_SIGNATURE=sp.digest_lines([rr.VERSION,sha256_stream(DENSE_CKPT),TRAIN_META['signature'],
    hashlib.sha256(TRAIN_LABELS.tobytes()).hexdigest(),sp.digest_lines(ATTR)])
print('[46 PASS] gallery',len(VAL_NAMES),'FIT',len(FIT_IDS),'AUDIT',len(AUDIT_IDS))
print('Train frozen features:',TRAIN_FEAT.shape,'Dense gallery:',DENSE_IMG.shape)
print('Baseline and DenseQuery are aligned. No VAL labels used in model training.')

#@title 47) TRAIN-only residual attribute representation learning (resume + progress)
RESIDUAL_EPOCHS=12 #@param {type:'integer'}
RESIDUAL_BATCH=192 #@param {type:'integer'}
RESIDUAL_LR=0.0007 #@param {type:'number'}
RESIDUAL_FORCE_RETRAIN=False #@param {type:'boolean'}

# Frozen DenseQuery image representation -> trainable small 128->96->40 residual.
# Uses TRAIN features and TRAIN labels, never public VAL labels or hidden data.
TRAIN_DENSE_IMG,TRAIN_DENSE_LOGITS=dense.image_outputs(DENSE_MODEL,TRAIN_FEAT,DENSE_DEVICE,chunk=1024)
assert TRAIN_DENSE_IMG.shape==(len(TRAIN_FEAT),128) and TRAIN_DENSE_LOGITS.shape==(len(TRAIN_FEAT),40)
RES_CKPT,RES_HISTORY,RES_HOLDOUT=rr.train_residual(
 TRAIN_DENSE_IMG,TRAIN_DENSE_LOGITS,TRAIN_LABELS,TRAIN_DOMAINS,
 RESIDUAL_DIR,RESIDUAL_SIGNATURE,epochs=RESIDUAL_EPOCHS,
 batch_size=RESIDUAL_BATCH,lr=RESIDUAL_LR,force=RESIDUAL_FORCE_RETRAIN)
print('[47 PASS] TRAIN-only residual checkpoint:',RES_CKPT)
print('Last TRAIN holdout:',RES_HISTORY[-1])

#@title 48) TRAIN-only reliability calibration + predicted gallery attribute evidence (no VAL labels)
import os
# Recompute TRAIN residual on actual TRAIN subset; never use public VAL labels to calibrate.
RES_TRAIN_DELTA=rr.residual_delta(RES_CKPT,TRAIN_DENSE_IMG)
RES_MIX,RES_WEIGHTS,RES_STATS=rr.calibration_policy(
 TRAIN_DENSE_LOGITS,RES_TRAIN_DELTA,TRAIN_LABELS,TRAIN_DOMAINS,RES_HOLDOUT,ATTR)
RES_STATS_DF=pd.DataFrame(RES_STATS)
RES_STATS_DF.to_csv(RESIDUAL_DIR/'train_attribute_reliability.csv',index=False)
print('Attributes with >0 residual correction:',int((RES_MIX>0).sum()),'/40')
print(RES_STATS_DF[['attribute','positive','negative','residual_blend','reliability_weight']].to_string(index=False))

# Cache prediction only. All computations use image features and TRAIN-calibrated parameters.
VAL_RES_CACHE=RESIDUAL_DIR/'residual_val_logits.npy'
VAL_RES_META=RESIDUAL_DIR/'residual_val_logits.json'
signature=sp.digest_lines([sha256_stream(RES_CKPT),expected,RES_MIX.tolist(),RES_WEIGHTS.tolist()])
if VAL_RES_CACHE.is_file() and VAL_RES_META.is_file() and json.loads(VAL_RES_META.read_text()).get('signature')==signature:
 CAL_LOGITS=np.load(VAL_RES_CACHE,mmap_mode='r',allow_pickle=False)
 print('[CACHE HIT] calibrated logits',CAL_LOGITS.shape)
else:
 DELTA=rr.residual_delta(RES_CKPT,DENSE_IMG)
 CAL_LOGITS=np.asarray(DENSE_LOGITS+DELTA*RES_MIX[None,:],'float32')
 tmp=VAL_RES_CACHE.with_name(VAL_RES_CACHE.stem+'.partial.npy')
 np.save(tmp,CAL_LOGITS);os.replace(tmp,VAL_RES_CACHE)
 VAL_RES_META.write_text(json.dumps({'signature':signature,'gallery_count':len(VAL_NAMES),
    'label_use':'TRAIN only','checkpoint':str(RES_CKPT)},indent=2))
assert CAL_LOGITS.shape==(len(VAL_NAMES),40) and np.isfinite(CAL_LOGITS).all()
RES_ATTR_DIST=rr.attribute_nll(Q[SELECT_IDS],CAL_LOGITS,RES_WEIGHTS)
assert RES_ATTR_DIST.shape==BASE_DIST.shape and np.isfinite(RES_ATTR_DIST).all()
print('[48 PASS] TRAIN-calibrated distances',RES_ATTR_DIST.shape)

#@title 49) FIT-only model selection — DenseQuery semantic + reliability residual, global or topK
from abpr.model import track2_metrics
from abpr.query_aware_ranking import retrieval_metrics
N_FIT=len(FIT_IDS)
FIT_BASE=np.asarray(BASE_DIST[:N_FIT],'float32')
FIT_SEM=np.asarray(DENSE_SEM[:N_FIT],'float32')
FIT_ATTR=np.asarray(RES_ATTR_DIST[:N_FIT],'float32')
def metric3(mat,ids):
 m=sp.semantic_metrics(mat,ids,GIDS,Q[ids],VAL_LABELS)
 return {key:float(m[key]) for key in ('mADM','mAP','Rank-1')}
base=metric3(FIT_BASE,FIT_IDS)
ones=np.bincount(GIDS,minlength=len(Q))[FIT_IDS]==1
base_one=float((GIDS[np.argmin(FIT_BASE,axis=1)]==FIT_IDS)[ones].mean()) if ones.any() else None

# NOTE: method selection uses FIT labels; AUDIT is not consulted here.
# Keep original DenseQuery champion (semantic alpha=0.05) as a reference.
configs=[('global',0.05,0.00,0)]
for mode in ('global','topk'):
 for alpha in (0.025,0.05,0.08):
  for beta in (0.02,0.05,0.08):
   configs.append((mode,alpha,beta,200 if mode=='topk' else 0))
trials=[]
for mode,alpha,beta,k in tqdm(configs,desc='49 / FIT candidate selection',unit='candidate'):
 dist=rr.fuse(FIT_BASE,FIT_SEM,FIT_ATTR,alpha,beta,mode=mode,topk=k if k else 200)
 m=metric3(dist,FIT_IDS)
 one=float((GIDS[np.argmin(dist,axis=1)]==FIT_IDS)[ones].mean()) if ones.any() else None
 delta={key:m[key]-base[key] for key in base}
 ok=(delta['mADM']>=0.003 and delta['mAP']>=-0.003 and delta['Rank-1']>=-0.0042 and
      (one is None or one>=base_one))
 trials.append(dict(mode=mode,alpha=float(alpha),beta=float(beta),topk=k,
                    **m,gain_mADM=delta['mADM'],gain_mAP=delta['mAP'],gain_Rank1=delta['Rank-1'],
                    single_positive_R1=one,single_positive_delta=None if one is None else one-base_one,
                    fit_gate=bool(ok)))
RES_FIT_TRIALS=pd.DataFrame(trials).sort_values(['mADM','mAP'],ascending=False).reset_index(drop=True)
RES_FIT_TRIALS.to_csv(RESIDUAL_DIR/'residual_fit_trials.csv',index=False)
eligible=RES_FIT_TRIALS[RES_FIT_TRIALS.fit_gate]
RES_FIT_OK=bool(len(eligible))
RES_CHOICE=(eligible.iloc[0] if RES_FIT_OK else RES_FIT_TRIALS.iloc[0]).to_dict()
(RESIDUAL_DIR/'residual_fit_choice.json').write_text(json.dumps({
 'baseline':base,'baseline_single_R1':base_one,'selected':RES_CHOICE,'fit_pass':RES_FIT_OK,
 'selection':'FIT only; AUDIT not used to select the model'},indent=2))
print('FIT baseline',base,'| FIT single-positive R1',base_one)
print(RES_FIT_TRIALS.head(10).to_string(index=False))
print('SELECTED',RES_CHOICE)

#@title 50) FIT robustness — query strata, per-attribute diagnostics and score identity checks
# This is a FIT-only stress test, NOT a second independent validation split.
FIT_OLD=np.asarray(BASE_DIST[:N_FIT],'float32')
assert np.array_equal(rr.fuse(FIT_OLD,FIT_SEM,FIT_ATTR,0,0),FIT_OLD)

RES_FIT_CAND=rr.fuse(FIT_OLD,FIT_SEM,FIT_ATTR,
 float(RES_CHOICE['alpha']),float(RES_CHOICE['beta']),
 mode=RES_CHOICE['mode'],topk=int(RES_CHOICE['topk'] or 200))
fit_ap_old,fit_r1_old=rr.per_query_ap(FIT_OLD,FIT_IDS,GIDS)
fit_ap_new,fit_r1_new=rr.per_query_ap(RES_FIT_CAND,FIT_IDS,GIDS)
positive_counts=np.bincount(GIDS,minlength=len(Q))[FIT_IDS]
rows=[]
for name,mask in [('single_positive',positive_counts==1),('2_to_4_positives',(positive_counts>=2)&(positive_counts<=4)),
                  ('5plus_positives',positive_counts>=5)]:
 if not mask.any():continue
 rows.append({'group':name,'queries':int(mask.sum()),'baseline_mAP':float(fit_ap_old[mask].mean()),
              'candidate_mAP':float(fit_ap_new[mask].mean()),'baseline_R1':float(fit_r1_old[mask].mean()),
              'candidate_R1':float(fit_r1_new[mask].mean())})
RES_FIT_GROUPS=pd.DataFrame(rows)
RES_FIT_GROUPS.to_csv(RESIDUAL_DIR/'residual_fit_groups.csv',index=False)
print(RES_FIT_GROUPS.to_string(index=False))
print('[50 PASS] No evidence/validation-label leakage in scoring; only FIT labels used for selection.')

#@title 51) Exploratory public AUDIT ONCE — same-gallery metrics, domains, single-positive & paired bootstrap
AUD_B=np.asarray(BASE_DIST[N_FIT:],'float32')
AUD_C=rr.fuse(AUD_B,DENSE_SEM[N_FIT:],RES_ATTR_DIST[N_FIT:],
 float(RES_CHOICE['alpha']),float(RES_CHOICE['beta']),
 mode=RES_CHOICE['mode'],topk=int(RES_CHOICE['topk'] or 200))
RES_AUD_BASE=metric3(AUD_B,AUDIT_IDS)
RES_AUD_CAND=metric3(AUD_C,AUDIT_IDS)
bs_ap,bs_r1=rr.per_query_ap(AUD_B,AUDIT_IDS,GIDS)
cs_ap,cs_r1=rr.per_query_ap(AUD_C,AUDIT_IDS,GIDS)
positive_n=np.bincount(GIDS,minlength=len(Q))[AUDIT_IDS]
one=positive_n==1
old_single=float(bs_r1[one].mean()) if one.any() else None
new_single=float(cs_r1[one].mean()) if one.any() else None

# Domain metrics computed with source-specific gallery, matching earlier reports.
DOMAIN_IDX=np.asarray([infer_domain_id(s) for s in VAL_NAMES])
rows=[]
for domain,idx in tqdm([('Market1501',0),('PA100k',1),('PETA',2)],desc='51 / AUDIT domains'):
 mask=DOMAIN_IDX==idx
 domain_q=np.isin(AUDIT_IDS,np.unique(GIDS[mask]))
 for label,mat in [('baseline',AUD_B),('candidate',AUD_C)]:
  part=mat[domain_q][:,mask];qid=AUDIT_IDS[domain_q]
  if len(qid)==0:continue
  m=retrieval_metrics(part,qid,GIDS[mask]);m.update(track2_metrics(part,Q[qid],VAL_LABELS[mask]))
  rows.append({'domain':domain,'method':label,'queries':len(qid),
               **{k:float(m[k]) for k in ('mADM','mAP','Rank-1')}})
RES_AUD_DOMAIN=pd.DataFrame(rows)
RES_AUD_DOMAIN.to_csv(RESIDUAL_DIR/'residual_audit_domains.csv',index=False)
domain_ok=True
for d in RES_AUD_DOMAIN.domain.unique():
 m=RES_AUD_DOMAIN[RES_AUD_DOMAIN.domain==d].set_index('method')
 domain_ok&=bool(m.loc['candidate','mADM']>=m.loc['baseline','mADM']-0.003)

# Paired, exploratory bootstrap for mAP/Rank1 only: mADM is not decomposed here.
rng=np.random.default_rng(927);n=len(AUDIT_IDS)
delta_ap=cs_ap-bs_ap;delta_r1=cs_r1-bs_r1
bootstrap=[]
for _ in tqdm(range(500),desc='51 / Paired bootstrap',unit='resample'):
 idx=rng.integers(0,n,size=n)
 bootstrap.append((float(delta_ap[idx].mean()),float(delta_r1[idx].mean())))
ci=np.quantile(np.asarray(bootstrap),[0.025,0.975],axis=0)
RES_AUD_OK=bool(RES_FIT_OK and
 RES_AUD_CAND['mADM']>=RES_AUD_BASE['mADM']+0.005 and
 RES_AUD_CAND['mAP']>=RES_AUD_BASE['mAP']-0.003 and
 RES_AUD_CAND['Rank-1']>=RES_AUD_BASE['Rank-1'] and
 (old_single is None or new_single>=old_single) and domain_ok)
RES_AUD_REPORT={
 'method':'TRAIN-only reliability-calibrated residual attribute representation',
 'baseline_audit':RES_AUD_BASE,'candidate_audit':RES_AUD_CAND,
 'selected_on_fit':RES_CHOICE,'fit_pass':bool(RES_FIT_OK),
 'single_positive_baseline_R1':old_single,'single_positive_candidate_R1':new_single,
 'domain_noninferior':bool(domain_ok),'audit_accepted':RES_AUD_OK,
 'mAP_gain_paired_bootstrap_CI95':[float(ci[0,0]),float(ci[1,0])],
 'rank1_gain_paired_bootstrap_CI95':[float(ci[0,1]),float(ci[1,1])],
 'caution':'Public AUDIT has been reused in earlier experiments; this is exploratory, not a fresh test.',
 'submission_ready':False,'reason':'External SigLIP2 weights/legality and an end-to-end offline hidden runtime are unverified.'}
(RESIDUAL_DIR/'residual_audit_report.json').write_text(json.dumps(RES_AUD_REPORT,indent=2))
print('AUDIT baseline:',RES_AUD_BASE)
print('AUDIT residual candidate:',RES_AUD_CAND)
print('Single-positive Rank-1:',old_single,'->',new_single)
print(RES_AUD_DOMAIN.to_string(index=False))
print('Paired bootstrap mAP gain CI95:',RES_AUD_REPORT['mAP_gain_paired_bootstrap_CI95'])
print('RESIDUAL_AUDIT_ACCEPTED:',RES_AUD_OK)
print('DO NOT submit this report ZIP to Codabench.')

#@title 52) Export complete reproducible REPORT ZIP — no automatic submission
README=RESIDUAL_DIR/'README_METHOD_AND_SUBMISSION.txt'
README.write_text('TRAIN-only reliability-calibrated residual attribute adaptation.\n'
    'Base: existing hidden-0.5389 submission; DenseQuery semantics from frozen SigLIP2.\n'
    'Public FIT selects the candidate; public AUDIT is exploratory and previously inspected.\n'
    'No hidden Codabench scores were obtained.\n'
    'A report ZIP is NOT an executable Codabench submission.\n'
    'To deploy the method, package the ORIGINAL baseline plus frozen SigLIP2, DenseQueryNet and\n'
    'residual checkpoints with fully offline inference. Verify competition rules and runtime.\n',encoding='utf-8')
items=[RESIDUAL_DIR/p for p in [
 'abpr_reliable_residual.py','residual_best.pt','residual_training_history.json',
 'train_attribute_reliability.csv','residual_fit_trials.csv','residual_fit_choice.json',
 'residual_fit_groups.csv','residual_audit_domains.csv','residual_audit_report.json',
 'README_METHOD_AND_SUBMISSION.txt']]
assert all(p.is_file() for p in items),[str(p) for p in items if not p.is_file()]
RES_REPORT_ZIP=RESIDUAL_DIR/'ABPR_ReliableResidual_Experiment_Report.zip'
with zipfile.ZipFile(RES_REPORT_ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in tqdm(items,desc='52 / Package evidence',unit='file'):
  z.write(p,arcname=p.name)
print('REPORT ZIP:',RES_REPORT_ZIP)
print('AUDIT accepted:',RES_AUD_OK)
print('UPLOAD TO CODABENCH: NO — report package, not a tested offline submission.')
print('Share this report to decide whether another representational change is justified.')