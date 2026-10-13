# ==================================================================
#@title 54) ITSELF-inspired ATTENTION — Install method + STRICT restore of DINO/HATA caches
from pathlib import Path
import os, sys, json, hashlib, base64, zlib, importlib
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

# This is an ITSELF-INSPIRED 40-attribute-token adaptation, NOT the original
# multi-layer ViT patch-attention MARS or a reproduced ITSELF result.
# Run notebook Cells 39–52 first; do NOT retrain the DINO/HATA backbones.
REQUIRED_STATE = [
    'DINO_TRAIN_TOKENS','DINO_VAL_TOKENS','DINO_TRAIN_LABELS',
    'DINO_TRAIN_DOMAINS','DINO_TRAIN_PATHS','VAL_PATHS',
    'HATA_TRAIN_PRED','HATA_VAL_TRACK1','HATA_RUN_DIR','HATA_BEST',
    'DINO_TO_TRACK2','TRACK1_ATTR','ATTR','Q','SELECT_IDS',
    'FIT_IDS','AUDIT_IDS','GIDS','VAL_LABELS','VAL_NAMES',
    'BASE_DIST','HATA_METHODS','HATA_FIT_WIN','dt',
    'DINO_RUN_SIGNATURE'
]
missing=[key for key in REQUIRED_STATE if key not in globals()]
if missing:
    raise RuntimeError('Required cached variables missing: '+str(missing)+
        '. Restore notebook Cells 39–52 (DINO/HATA cache and evaluation); no need to retrain backbone.')

ATTR=list(ATTR);TRACK1_ATTR=list(TRACK1_ATTR)
assert len(ATTR)==len(TRACK1_ATTR)==40 and set(ATTR)==set(TRACK1_ATTR)
assert np.array_equal(np.array(DINO_TO_TRACK2),
  np.array([TRACK1_ATTR.index(name) for name in ATTR]))
assert DINO_TRAIN_TOKENS.shape[1:]==DINO_VAL_TOKENS.shape[1:]==(40,192)
assert DINO_TRAIN_LABELS.shape==(len(DINO_TRAIN_TOKENS),40)
assert HATA_TRAIN_PRED.shape==(len(DINO_TRAIN_TOKENS),40)
assert HATA_VAL_TRACK1.shape==(len(DINO_VAL_TOKENS),40)
assert len(VAL_NAMES)==len(VAL_PATHS)==len(GIDS)==len(VAL_LABELS)==len(DINO_VAL_TOKENS)
assert len(SELECT_IDS)==len(FIT_IDS)+len(AUDIT_IDS)
assert np.array_equal(np.asarray(SELECT_IDS),np.r_[FIT_IDS,AUDIT_IDS])
assert np.asarray(BASE_DIST).shape==(len(SELECT_IDS),len(VAL_NAMES))
assert Path(HATA_BEST).is_file(), 'Missing existing HATA checkpoint'
assert np.isin(np.asarray(DINO_TRAIN_LABELS),[-1,0,1]).all()
assert np.isin(np.asarray(Q[SELECT_IDS]),[-1,0,1]).all()

MODULE_CODE=zlib.decompress(base64.b64decode('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'))
assert hashlib.sha256(MODULE_CODE).hexdigest()=='9bb830dadc3915058c4c0baa1852dc4099b5db5a6eedc455f19c07563eecfb03'
ATTENTION_MODULE_FILE=Path('/content/abpr_attentive_token_adapter.py')
ATTENTION_MODULE_FILE.write_bytes(MODULE_CODE)
sys.path.insert(0,'/content')
import abpr_attentive_token_adapter as itatt
itatt=importlib.reload(itatt)

ATTENTION_ROOT=Path(HATA_RUN_DIR)/'itself_inspired_attention'
ATTENTION_ROOT.mkdir(parents=True,exist_ok=True)
ATTENTION_TRAIN_SIGNATURE=itatt.digest(
    DINO_RUN_SIGNATURE, str(Path(HATA_BEST).resolve()),
    hashlib.sha256(Path(HATA_BEST).read_bytes()).hexdigest(),
    np.asarray(DINO_TRAIN_LABELS,dtype=np.float32),
    np.asarray(DINO_TRAIN_DOMAINS,dtype=np.int64),
    list(map(str,DINO_TRAIN_PATHS)),TRACK1_ATTR)
ATTENTION_VAL_SIGNATURE=itatt.digest(
    ATTENTION_TRAIN_SIGNATURE,list(map(str,VAL_PATHS)),TRACK1_ATTR)

HATA_REF_SOURCE=HATA_METHODS[HATA_FIT_WIN['method']]
HATA_REF_ALL=dt.fusion(np.asarray(BASE_DIST,np.float32),
    np.asarray(HATA_REF_SOURCE,np.float32),float(HATA_FIT_WIN['alpha']))
assert HATA_REF_ALL.shape==np.asarray(BASE_DIST).shape
assert np.isfinite(HATA_REF_ALL).all()
print('[54 PASS] Reusing frozen DINO cache, HATA teacher, and prior FIT choice.')
print('TRAIN:',len(DINO_TRAIN_TOKENS),'VAL:',len(DINO_VAL_TOKENS),
      'FIT:',len(FIT_IDS),'AUDIT:',len(AUDIT_IDS))
print('Source:',ATTENTION_MODULE_FILE,'out:',ATTENTION_ROOT)
print('Prior HATA FIT winner:',HATA_FIT_WIN)
print('No VAL labels are used to train attention or compute candidate distances.')


# ==================================================================
#@title 55) ATTENTION — TRAIN-only group-aware attention adapter (resume, progressive token selection)
# Trains a small attention adapter on frozen DINO tokens; HATA logits act as teacher.
# Caches and the 195.9 MiB ConvNeXt checkpoint remain UNMODIFIED.
ATTN_EPOCHS=10  #@param {type:'integer'}
ATTN_BATCH_SIZE=64  #@param {type:'integer'}
ATTN_LR=0.0007  #@param {type:'number'}
ATTN_RANK_WEIGHT=0.12  #@param {type:'number'}
ATTN_KD_WEIGHT=0.04  #@param {type:'number'}
ATTN_FORCE_NEW=False  #@param {type:'boolean'}

ATTN_RUN_DIR=ATTENTION_ROOT/f'ep{ATTN_EPOCHS}_lr{ATTN_LR}_rw{ATTN_RANK_WEIGHT}'
ATTN_RUN_DIR.mkdir(parents=True,exist_ok=True)
# Signature includes TRAIN feature identity & teacher checkpoint; never silently resume
# a model trained on another dataset/checkpoint/order.
ATTN_MODEL,ATTN_BEST,ATTN_TRAIN_HISTORY=itatt.fit(
    DINO_TRAIN_TOKENS,HATA_TRAIN_PRED,DINO_TRAIN_LABELS,
    DINO_TRAIN_DOMAINS,TRACK1_ATTR,ATTN_RUN_DIR,
    signature=ATTENTION_TRAIN_SIGNATURE,epochs=ATTN_EPOCHS,
    batch_size=ATTN_BATCH_SIZE,lr=ATTN_LR,
    rank_weight=ATTN_RANK_WEIGHT,kd_weight=ATTN_KD_WEIGHT,
    seed=42,force=ATTN_FORCE_NEW,patience=5)
print('[55 PASS] Best TRAIN-only checkpoint:',ATTN_BEST)
print('[TRAIN HOLDOUT]',ATTN_TRAIN_HISTORY[-1] if ATTN_TRAIN_HISTORY else 'checkpoint resumed')


# ==================================================================
#@title 56) ATTENTION — encode cached 33,407 gallery images + query-conditioned 40-attribute distances
# Reuses frozen DINO features. NO image re-encoding, NO VAL labels in scoring.
ATTN_CK_SHA=hashlib.sha256(Path(ATTN_BEST).read_bytes()).hexdigest()
ATTN_FEATURE_SIGNATURE=itatt.digest(ATTENTION_VAL_SIGNATURE,ATTN_CK_SHA)
ATTN_VAL_TRACK1=itatt.infer(ATTN_MODEL,DINO_VAL_TOKENS,HATA_VAL_TRACK1,
    ATTN_RUN_DIR/'attention_val_logits.npy',signature=ATTN_FEATURE_SIGNATURE,
    batch_size=80)
assert ATTN_VAL_TRACK1.shape==(len(VAL_PATHS),40)
ATTN_VAL_40=np.asarray(ATTN_VAL_TRACK1,np.float32)[:,DINO_TO_TRACK2]
# Strict label-free score: negative log-likelihood of query-specified attributes.
ATTN_DIST=itatt.weighted_query_nll(Q[SELECT_IDS],ATTN_VAL_40,chunk=48)
assert ATTN_DIST.shape==np.asarray(BASE_DIST).shape
assert np.isfinite(ATTN_DIST).all()
print('[56 PASS] ATTENTION DISTANCE:',ATTN_DIST.shape)
print('[RANGE]',float(ATTN_DIST.min()),float(ATTN_DIST.max()))
print('No VAL labels or semantic GIDS used by attention scoring.')


# ==================================================================
#@title 57) ATTENTION — select ONLY on FIT, compare against accepted HATA and the original baseline
from abpr.query_aware_ranking import retrieval_metrics
from abpr.model import track2_metrics
from abpr.core import infer_domain_id

ATTN_NFIT=len(FIT_IDS)
ATTN_B_FIT=np.asarray(BASE_DIST[:ATTN_NFIT],np.float32)
ATTN_H_FIT=np.asarray(HATA_REF_ALL[:ATTN_NFIT],np.float32)

def attention_metrics(scores,query_ids):
    return {k:float(v) for k,v in dino_metrics(scores,query_ids).items()
           if k in ('mADM','mAP','Rank-1')}

ATTN_BASE_FIT=attention_metrics(ATTN_B_FIT,FIT_IDS)
ATTN_HATA_FIT=attention_metrics(ATTN_H_FIT,FIT_IDS)
positive_counts=np.bincount(np.asarray(GIDS,dtype=np.int64),minlength=len(Q))[FIT_IDS]
single=positive_counts==1
h_top=(GIDS[np.argmin(ATTN_H_FIT,axis=1)]==FIT_IDS)

ATTN_ALPHA_GRID=[0.00,0.025,0.05,0.075,0.10,0.15,0.20]
records=[]
for reference,mat in [('baseline',ATTN_B_FIT),('HATA',ATTN_H_FIT)]:
    for alpha in tqdm(ATTN_ALPHA_GRID,desc='ATTENTION FIT fusion '+reference,unit='trial'):
        cand=itatt.combine(mat,ATTN_DIST[:ATTN_NFIT],alpha)
        mm=attention_metrics(cand,FIT_IDS)
        cand_top=(GIDS[np.argmin(cand,axis=1)]==FIT_IDS)
        delta_single=float(cand_top[single].mean()-h_top[single].mean()) if single.any() else 0.0
        row={'reference':reference,'alpha':float(alpha),**mm,
             'gain_vs_hata_mADM':mm['mADM']-ATTN_HATA_FIT['mADM'],
             'gain_vs_base_mADM':mm['mADM']-ATTN_BASE_FIT['mADM'],
             'delta_mAP_vs_HATA':mm['mAP']-ATTN_HATA_FIT['mAP'],
             'delta_R1_vs_HATA':mm['Rank-1']-ATTN_HATA_FIT['Rank-1'],
             'delta_single_R1_vs_HATA':delta_single}
        # Thresholds are screening heuristics, not organizer criteria.
        row['fit_pass']=bool(alpha>0 and
             row['gain_vs_hata_mADM']>=0.0015 and
             row['gain_vs_base_mADM']>=0.005 and
             row['delta_mAP_vs_HATA']>=-0.004 and
             row['delta_R1_vs_HATA']>=-0.005 and
             delta_single>=-0.007)
        records.append(row)
ATTN_FIT_TABLE=pd.DataFrame(records).sort_values(['mADM','mAP'],ascending=False).reset_index(drop=True)
ATTN_FIT_TABLE.to_csv(ATTN_RUN_DIR/'attention_fit_trials.csv',index=False)
winners=ATTN_FIT_TABLE[ATTN_FIT_TABLE.fit_pass]
ATTN_FIT_WIN=(winners.iloc[0] if len(winners) else ATTN_FIT_TABLE.iloc[0]).to_dict()
ATTN_FIT_PASS=bool(ATTN_FIT_WIN['fit_pass'])
itatt.atomic_json(ATTN_RUN_DIR/'attention_fit_choice.json',
    {'winner':ATTN_FIT_WIN,'fit_pass':ATTN_FIT_PASS,
     'baseline_fit':ATTN_BASE_FIT,'hata_fit':ATTN_HATA_FIT,
     'selection':'FIT-only; AUDIT not used to pick alpha or reference'})
print('[BASELINE FIT]',ATTN_BASE_FIT)
print('[HATA FIT]',ATTN_HATA_FIT)
print('[ATTENTION FIT TOP]\n',ATTN_FIT_TABLE.head(10).to_string(index=False))
print('[WINNER]',ATTN_FIT_WIN,'FIT PASS:',ATTN_FIT_PASS)


# ==================================================================
#@title 58) ATTENTION — AUDIT once + domain/rare-positive safeguards (public exploratory)
# AUDIT here was viewed repeatedly by earlier notebooks; it is NOT fresh holdout.
ATTN_B_AUD=np.asarray(BASE_DIST[ATTN_NFIT:],np.float32)
ATTN_H_AUD=np.asarray(HATA_REF_ALL[ATTN_NFIT:],np.float32)
ATTN_A_BASE=attention_metrics(ATTN_B_AUD,AUDIT_IDS)
ATTN_A_HATA=attention_metrics(ATTN_H_AUD,AUDIT_IDS)
ATTN_A_REF=ATTN_B_AUD if ATTN_FIT_WIN['reference']=='baseline' else ATTN_H_AUD
ATTN_A_CAND=itatt.combine(ATTN_A_REF,ATTN_DIST[ATTN_NFIT:],ATTN_FIT_WIN['alpha'])
ATTN_A_NEW=attention_metrics(ATTN_A_CAND,AUDIT_IDS)

pos_count=np.bincount(GIDS,minlength=len(Q))[AUDIT_IDS]
mask_single=pos_count==1
prev_top=(GIDS[np.argmin(ATTN_H_AUD,axis=1)]==AUDIT_IDS)
new_top=(GIDS[np.argmin(ATTN_A_CAND,axis=1)]==AUDIT_IDS)
prev_single=float(prev_top[mask_single].mean()) if mask_single.any() else None
new_single=float(new_top[mask_single].mean()) if mask_single.any() else None

source_domains=np.asarray([infer_domain_id(x) for x in VAL_NAMES],dtype=np.int64)
AUDIT_DOMAIN_ROWS=[]
for domain in tqdm((0,1,2),desc='Attention AUDIT per source domain',unit='domain'):
    gm=source_domains==domain
    local_ids=AUDIT_IDS[np.isin(AUDIT_IDS,np.unique(GIDS[gm]))]
    if not len(local_ids):continue
    row_index={int(a):i for i,a in enumerate(AUDIT_IDS)}
    ix=[row_index[int(i)] for i in local_ids]
    for label,dist in [('baseline',ATTN_B_AUD),('HATA',ATTN_H_AUD),('attention',ATTN_A_CAND)]:
        s=dist[ix][:,gm]
        m1=retrieval_metrics(s,local_ids,GIDS[gm])
        m1.update(track2_metrics(s,Q[local_ids],VAL_LABELS[gm]))
        AUDIT_DOMAIN_ROWS.append({'domain':int(domain),'method':label,'queries':len(local_ids),
            **{k:float(m1[k]) for k in ('mADM','mAP','Rank-1')}})
ATTN_AUDIT_DOMAINS=pd.DataFrame(AUDIT_DOMAIN_ROWS)
ATTN_AUDIT_DOMAINS.to_csv(ATTN_RUN_DIR/'attention_audit_domains.csv',index=False)
per_domain_ok=True
for d in (0,1,2):
    t=ATTN_AUDIT_DOMAINS[ATTN_AUDIT_DOMAINS.domain==d].set_index('method')
    if 'HATA' in t.index and 'attention' in t.index:
        per_domain_ok &= bool(t.loc['attention','mADM']>=t.loc['HATA','mADM']-0.007)

ATTN_AUDIT_ACCEPTED=bool(ATTN_FIT_PASS and
    ATTN_A_NEW['mADM']>=ATTN_A_HATA['mADM']+0.002 and
    ATTN_A_NEW['mAP']>=ATTN_A_HATA['mAP']-0.004 and
    ATTN_A_NEW['Rank-1']>=ATTN_A_HATA['Rank-1']-0.005 and
    (new_single is None or new_single>=prev_single-0.007) and per_domain_ok)

ATTN_AUDIT_REPORT={
 'method':'ITSELF-inspired attentive attribute-token refinement, not original GRAB/MARS',
 'paper':'https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_ITSELF_Attention_Guided_Fine-Grained_Alignment_for_Vision-Language_Retrieval_WACV_2026_paper.html',
 'train_only':True,'pretrained_DINO_source':'Track1 user checkpoint',
 'public_AUDIT_previously_inspected':True,'hidden_score':None,
 'checkpoint':str(ATTN_BEST),'selection':ATTN_FIT_WIN,
 'fit_baseline':ATTN_BASE_FIT,'fit_HATA':ATTN_HATA_FIT,
 'audit_baseline':ATTN_A_BASE,'audit_HATA':ATTN_A_HATA,'audit_attention':ATTN_A_NEW,
 'gain_over_HATA_mADM':float(ATTN_A_NEW['mADM']-ATTN_A_HATA['mADM']),
 'single_positive_HATA_R1':prev_single,'single_positive_attention_R1':new_single,
 'domains_ok':bool(per_domain_ok),'fit_pass':ATTN_FIT_PASS,
 'audit_accepted':ATTN_AUDIT_ACCEPTED,
 'eligible_for_codabench':None,
 'note':'Public VAL only; all query/label arrays used solely for public metrics. No submission created.'}
itatt.atomic_json(ATTN_RUN_DIR/'attention_audit_report.json',ATTN_AUDIT_REPORT)
print('BASELINE AUDIT:',ATTN_A_BASE)
print('PRIOR HATA AUDIT:',ATTN_A_HATA)
print('NEW ATTENTION AUDIT:',ATTN_A_NEW)
print('Single-positive R1 HATA -> new:',prev_single,'->',new_single)
print('DOMAIN AUDIT:\n',ATTN_AUDIT_DOMAINS.to_string(index=False))
print('ATTN_AUDIT_ACCEPTED:',ATTN_AUDIT_ACCEPTED)


# ==================================================================
#@title 59) ATTENTION — export reproducible report ZIP (NOT a Codabench submission)
import zipfile,hashlib
ATTN_README=ATTN_RUN_DIR/'README.txt'
ATTN_README.write_text('ITSELF-inspired Attribute Token Attention (research extension).\nUses frozen DINOv2 Track1 pooled 40 attribute tokens and HATA teacher predictions.\nDoes not use ITSELF multi-layer patch attention; not a reproduction of ITSELF.\nAttentive bank (40 tokens), group-aware learnable attention, progressive top-k budget.\nTRAIN-only fitting/holdout, public FIT for selection, public AUDIT for exploratory check.\nAUDIT has been inspected before in previous experiments, so findings are exploratory.\nDO NOT upload this report ZIP to Codabench.\nTo submit, implement/test complete offline inference on hidden images, confirm rules and licensing.\n',encoding='utf-8')
ATTN_REPORT_ZIP=ATTN_RUN_DIR/'ABPR_ITSELF_Inspired_Attention_Experiment_Report.zip'
report_files=[ATTN_MODULE_FILE if 'ATTN_MODULE_FILE' in globals() else ATTENTION_MODULE_FILE,
    ATTN_RUN_DIR/'attention_fit_trials.csv',ATTN_RUN_DIR/'attention_fit_choice.json',
    ATTN_RUN_DIR/'attention_audit_domains.csv',ATTN_RUN_DIR/'attention_audit_report.json',
    ATTN_RUN_DIR/'attention_train_history.json',ATTN_README]
with zipfile.ZipFile(ATTN_REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for f in tqdm(report_files,desc='Pack ATTENTION research report',unit='file'):
        f=Path(f)
        if f.is_file():z.write(f,arcname=f.name)
print('[EXPERIMENT REPORT]',ATTN_REPORT_ZIP)
print('[MODEL CHECKPOINT]',ATTN_BEST)
print('[PUBLIC ATTENTION AUDIT ACCEPTED]',ATTN_AUDIT_ACCEPTED)
print('[CURRENT HIDDEN BASELINE] 0.549873 mADM — not predicted from public results')
print('[SUBMISSION] NOT EXPORTED. Need offline runtime, rules and ingestion verification.')


# ==================================================================
#@title 60) ATTENTION — optional next-action summary (submission gate is NEVER automatic)
print('='*74)
print('NEXT ACTION — ATTENTION ABPR')
print('='*74)
if not ATTN_FIT_PASS:
    print('FIT gate FAILED. Keep existing submission. Do not use hidden Codabench to tune these same alpha trials.')
elif not ATTN_AUDIT_ACCEPTED:
    print('FIT improved, but AUDIT gate FAILED. Keep existing submission.')
else:
    print('Public FIT/AUDIT gates passed. This warrants building an end-to-end OFFLINE runtime, NOT immediate submission.')
    print('Must reproduce attention tokens on new hidden gallery images using the frozen DINO checkpoint.')
    print('Bundle all weights and code; test 40-attribute order, (queries,gallery) output, GPU memory, and organizer pretrained-weight policy.')
print('Existing hidden mADM: 0.549873; target 0.65–0.68 has NOT been demonstrated.')
print('Source notebook: PAR_UPAR_ABPR_Joint_Training(10).ipynb + Cells 54–60')
