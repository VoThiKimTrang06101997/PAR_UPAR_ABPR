

# =============================================================
#@title 47) HATA — restore frozen DINO caches, source and signatures
from pathlib import Path
import sys, json, hashlib, base64, zlib
import numpy as np
import torch
from tqdm.auto import tqdm

# Run Cells 39–45 first. This is a continuation, not a fresh download/training pipeline.
NEED = ['DINO_TRAIN_TOKENS','DINO_TRAIN_LOGITS','DINO_VAL_TOKENS','DINO_VAL_LOGITS',
        'DINO_TRAIN_LABELS','DINO_TRAIN_DOMAINS','DINO_TRAIN_PATHS','VAL_PATHS',
        'DINO_RUN_SIGNATURE','DINO_OUTPUT','DINO_CHECKPOINT','DINO_CK_SHA',
        'DINO_HEAD','DINO_HEAD_PATH','DINO_TO_TRACK2','DINO_RAW_DIST','DINO_HEAD_DIST',
        'BASE_DIST','FIT_IDS','AUDIT_IDS','SELECT_IDS','GIDS','Q','VAL_LABELS','VAL_NAMES',
        'ATTR','TRACK1_ATTR','dt','dino_metrics','DINO_WIN','DINO_TTA_FLIP']
missing=[x for x in NEED if x not in globals()]
if missing:
    raise RuntimeError('Colab lost source/cache variables: '+str(missing)+
        '. Restore via Cells 39–44 (and 43/44 evaluation) in the uploaded notebook, without retraining frozen DINO.')
HATA_ROOT=Path(DINO_OUTPUT)/'hata_representation'
HATA_ROOT.mkdir(parents=True,exist_ok=True)
HATA_FILE=Path('/content/abpr_hata_method.py')
HATA_FILE.write_bytes(zlib.decompress(base64.b64decode('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')))
sys.path.insert(0,'/content')
import abpr_hata_method as hata

assert len(ATTR)==len(TRACK1_ATTR)==40 and set(ATTR)==set(TRACK1_ATTR)
assert list(np.asarray(TRACK1_ATTR)[DINO_TO_TRACK2])==list(ATTR)
assert np.asarray(DINO_TRAIN_TOKENS).shape==(len(DINO_TRAIN_LABELS),40,192)
assert np.asarray(DINO_VAL_TOKENS).shape==(len(VAL_PATHS),40,192)
assert np.asarray(BASE_DIST).shape==(len(SELECT_IDS),len(VAL_PATHS))
assert np.array_equal(np.asarray(SELECT_IDS),np.r_[FIT_IDS,AUDIT_IDS])
assert len(GIDS)==len(VAL_NAMES)==len(VAL_LABELS)==len(VAL_PATHS)
assert np.isin(DINO_TRAIN_LABELS,[-1,0,1]).all()

# Validate actual cached frozen DINO features against their image ordering/model hashes.
for split,paths in [('train',DINO_TRAIN_PATHS),('val',VAL_PATHS)]:
    cache_dir=Path(DINO_OUTPUT)/f'{split}_features'
    meta=json.loads((cache_dir/'meta.json').read_text())
    expected=dt.digest_lines([DINO_CK_SHA,dt.digest_lines(TRACK1_ATTR),dt.digest_lines(paths),
                              'flip'+str(DINO_TTA_FLIP)])
    assert meta['signature']==expected and meta['completed']==len(paths),(
        'DINO cache/order mismatch',split,meta.get('completed'),len(paths))

DINO_HEAD_BEST=Path(DINO_HEAD_PATH).with_name('dino_token_head_best.pt')
assert DINO_HEAD_BEST.is_file(),'Run Cell 41 first for the selected old DINO head.'
DINO_HEAD_CK_SHA=hata.digest(DINO_HEAD_BEST.stat().st_size,dt.hash_file(DINO_HEAD_BEST))
HATA_SIGNATURE=hata.digest(DINO_RUN_SIGNATURE,DINO_HEAD_CK_SHA,dt.digest_lines(TRACK1_ATTR))
print('[PASS] DINO frozen features, 40 attribute order, baseline matrix')
print('[TRAIN]',len(DINO_TRAIN_LABELS),'[VAL]',len(VAL_NAMES),'[FIT]',len(FIT_IDS),'[AUDIT]',len(AUDIT_IDS))
print('[NEW SOURCE]',HATA_FILE)



# =============================================================
#@title 48) HATA TRAIN-only hierarchical attribute-token adapter — resumable
# Only TRAIN labels optimize HATA. Freeze existing DINO teacher & DINO token head.
HATA_EPOCHS=12  #@param {type:'integer'}
HATA_BATCH=96   #@param {type:'integer'}
HATA_LR=0.001   #@param {type:'number'}
HATA_RANK_WEIGHT=0.12 #@param {type:'number'}
HATA_FORCE_NEW=False  #@param {type:'boolean'}
assert HATA_EPOCHS>=1 and 24<=HATA_BATCH<=512

# Old TRAIN token-head predictions are the frozen anchor, in Track1 attribute order.
DINO_TRAIN_HEAD=dt.head_logits(DINO_HEAD,DINO_TRAIN_TOKENS,DINO_TRAIN_LOGITS)
assert DINO_TRAIN_HEAD.shape==(len(DINO_TRAIN_LABELS),40)
HATA_RUN_DIR=HATA_ROOT/f'lr{HATA_LR}_ep{HATA_EPOCHS}_rank{HATA_RANK_WEIGHT}'
HATA_TRAIN_SIGNATURE=hata.digest(HATA_SIGNATURE,DINO_TRAIN_LABELS,DINO_TRAIN_DOMAINS,
                                 HATA_EPOCHS,HATA_LR,HATA_RANK_WEIGHT)
HATA_MODEL,HATA_HISTORY,HATA_BEST=hata.fit(
    DINO_TRAIN_TOKENS,DINO_TRAIN_HEAD,DINO_TRAIN_LABELS,DINO_TRAIN_DOMAINS,
    TRACK1_ATTR,HATA_RUN_DIR,HATA_TRAIN_SIGNATURE,epochs=int(HATA_EPOCHS),
    batch_size=int(HATA_BATCH),lr=float(HATA_LR),query_weight=float(HATA_RANK_WEIGHT),
    seed=2027,force=HATA_FORCE_NEW,patience=4)
(HATA_RUN_DIR/'train_history.json').write_text(json.dumps(HATA_HISTORY,indent=2))
print('[HATA TRAIN-ONLY BEST]',HATA_BEST)
print('No DINO backbone retraining, no public VAL-label training.')



# =============================================================
#@title 49) HATA encode gallery + calibrated attribute query distances
import numpy as np,hashlib,json

# Reuse DINO head VAL predictions computed in Cell 42; restore native Track1 order.
if 'DINO_HEAD_40' in globals():
    DINO_VAL_HEAD=np.asarray(DINO_HEAD_40,dtype=np.float32)[:,np.argsort(DINO_TO_TRACK2)]
else:
    DINO_VAL_HEAD=dt.head_logits(DINO_HEAD,DINO_VAL_TOKENS,DINO_VAL_LOGITS)
assert DINO_VAL_HEAD.shape==(len(VAL_NAMES),40)

HATA_BEST_SHA=dt.hash_file(HATA_BEST)
HATA_FEATURE_SIG=hata.digest(HATA_SIGNATURE,HATA_BEST_SHA,len(VAL_PATHS))
HATA_VAL_TRACK1=hata.infer(HATA_MODEL,DINO_VAL_TOKENS,DINO_VAL_HEAD,
                          HATA_RUN_DIR/'val_hata_logits.npy',batch_size=384,
                          expected_signature=HATA_FEATURE_SIG)
# Diagnostic reliable attributes are derived only from TRAIN labels.
HATA_TRAIN_PRED=hata.infer(HATA_MODEL,DINO_TRAIN_TOKENS,DINO_TRAIN_HEAD,
                          HATA_RUN_DIR/'train_hata_logits.npy',batch_size=384,
                          expected_signature=hata.digest(HATA_FEATURE_SIG,'train',len(DINO_TRAIN_PATHS)))
_,HATA_TRAIN_HOLD=hata.balanced_indices(DINO_TRAIN_LABELS,DINO_TRAIN_DOMAINS,2027)
HATA_RELIABILITY=hata.reliability(HATA_TRAIN_PRED[HATA_TRAIN_HOLD],DINO_TRAIN_LABELS[HATA_TRAIN_HOLD])
np.save(HATA_RUN_DIR/'reliability_train_only.npy',HATA_RELIABILITY)

HATA_VAL_40=np.asarray(HATA_VAL_TRACK1,dtype=np.float32)[:,DINO_TO_TRACK2]
HATA_R40=HATA_RELIABILITY[DINO_TO_TRACK2]
HATA_UNWEIGHTED=hata.weighted_query_nll(Q[SELECT_IDS],HATA_VAL_40,chunk=48)
HATA_WEIGHTED=hata.weighted_query_nll(Q[SELECT_IDS],HATA_VAL_40,weights=HATA_R40,chunk=48)
assert HATA_UNWEIGHTED.shape==HATA_WEIGHTED.shape==BASE_DIST.shape
assert np.isfinite(HATA_WEIGHTED).all()
print('[PASS] HATA representation logits',HATA_VAL_40.shape)
print('[DISTANCE]',HATA_WEIGHTED.shape,'[RELIABILITY]',HATA_R40.min(),HATA_R40.max())



# =============================================================
#@title 50) HATA FIT-only search — protect single-positive queries
import numpy as np,pandas as pd,json
from tqdm.auto import tqdm

HATA_ALPHA_GRID=[0.02,0.035,0.05,0.075,0.10,0.15,0.20] # Small fusion weights avoid overwhelming baseline.
N_FIT=len(FIT_IDS)
HATA_BF=np.asarray(BASE_DIST[:N_FIT],np.float32)
HATA_BASE_FIT=dino_metrics(HATA_BF,FIT_IDS)
HATA_PCOUNTS=np.bincount(GIDS,minlength=len(Q))[FIT_IDS]
HATA_SINGLE=(HATA_PCOUNTS==1)
HATA_OLD_TOP=GIDS[np.argmin(HATA_BF,axis=1)]==FIT_IDS
HATA_METHODS={'HATA_unweighted':HATA_UNWEIGHTED,'HATA_train_reliable':HATA_WEIGHTED,
              'previous_DINO_head':DINO_HEAD_DIST}
rows=[]
for meth,dm in HATA_METHODS.items():
    for alpha in tqdm(HATA_ALPHA_GRID,desc=f'FIT {meth}',unit='fusion'):
        f=dt.fusion(HATA_BF,dm[:N_FIT],float(alpha))
        m=dino_metrics(f,FIT_IDS)
        top=GIDS[np.argmin(f,axis=1)]==FIT_IDS
        single_diff=float(top[HATA_SINGLE].mean()-HATA_OLD_TOP[HATA_SINGLE].mean()) if HATA_SINGLE.any() else 0.
        delta={k:float(m[k]-HATA_BASE_FIT[k]) for k in ('mADM','mAP','Rank-1')}
        ok=(delta['mADM']>=.003 and delta['mAP']>=-.005 and delta['Rank-1']>=-.005
            and single_diff>=-.007)
        rows.append({'method':meth,'alpha':float(alpha),**m,
                     'delta_mADM':delta['mADM'],'delta_mAP':delta['mAP'],
                     'delta_Rank-1':delta['Rank-1'],
                     'single_positive_rank1_delta':single_diff,'fit_pass':bool(ok)})
HATA_FIT_TABLE=pd.DataFrame(rows).sort_values(['mADM','mAP'],ascending=False)
HATA_FIT_TABLE.to_csv(HATA_RUN_DIR/'hata_fit_trials.csv',index=False)
valid=HATA_FIT_TABLE.query('fit_pass == True')
HATA_FIT_WIN=(valid.iloc[0] if len(valid) else HATA_FIT_TABLE.iloc[0]).to_dict()
HATA_FIT_PASS=bool(HATA_FIT_WIN['fit_pass'])
HATA_BEATS_OLD_FIT=bool(HATA_FIT_WIN['method']!='previous_DINO_head'
                        and HATA_FIT_WIN['mADM']>=max(
                            HATA_BASE_FIT['mADM'],float(DINO_WIN.get('mADM',-999)))+.001)
print('[BASE FIT]',HATA_BASE_FIT)
print('[TOP 10 FIT]\n',HATA_FIT_TABLE.head(10).to_string(index=False))
print('[FIT winner]',HATA_FIT_WIN)
print('[HATA true improvement beyond prior DINO?]',HATA_BEATS_OLD_FIT)



# =============================================================
#@title 51) HATA exploratory AUDIT — mADM/mAP/Rank-1, domains and single-positive
import numpy as np,pandas as pd,json
from abpr.core import infer_domain_id
from abpr.query_aware_ranking import retrieval_metrics
from abpr.model import track2_metrics
from tqdm.auto import tqdm

HATA_BA=np.asarray(BASE_DIST[N_FIT:],np.float32)
HATA_NEW=HATA_METHODS[HATA_FIT_WIN['method']]
HATA_CA=dt.fusion(HATA_BA,HATA_NEW[N_FIT:],HATA_FIT_WIN['alpha'])
HATA_A_BASE=dino_metrics(HATA_BA,AUDIT_IDS)
HATA_A_CAND=dino_metrics(HATA_CA,AUDIT_IDS)
# Prior DINO reference (selected only on FIT, never retuned on AUDIT).
HATA_PRIOR_SRC=(DINO_RAW_DIST if DINO_WIN.get('method')=='DINO_raw' else DINO_HEAD_DIST)
HATA_PRIOR=dt.fusion(HATA_BA,HATA_PRIOR_SRC[N_FIT:],float(DINO_WIN['alpha']))
HATA_A_PRIOR=dino_metrics(HATA_PRIOR,AUDIT_IDS)
count=np.bincount(GIDS,minlength=len(Q))[AUDIT_IDS]
single=(count==1)
old_top=GIDS[np.argmin(HATA_BA,axis=1)]==AUDIT_IDS
new_top=GIDS[np.argmin(HATA_CA,axis=1)]==AUDIT_IDS
single_b=float(old_top[single].mean()) if single.any() else None
single_c=float(new_top[single].mean()) if single.any() else None
vdomains=np.asarray([infer_domain_id(p) for p in VAL_NAMES],dtype=np.int8)
rows=[]
for d in tqdm(range(3),desc='HATA AUDIT by domain',unit='domain'):
    mask=vdomains==d
    ids=AUDIT_IDS[np.isin(AUDIT_IDS,np.unique(GIDS[mask]))]
    if len(ids)==0:continue
    index={int(k):i for i,k in enumerate(AUDIT_IDS)}
    rr=[index[int(x)] for x in ids]
    for name,matrix in [('baseline',HATA_BA),('candidate',HATA_CA),('prior_DINO',HATA_PRIOR)]:
        ds=matrix[rr][:,mask]
        mm=retrieval_metrics(ds,ids,GIDS[mask])
        mm.update(track2_metrics(ds,Q[ids],VAL_LABELS[mask]))
        rows.append({'domain':d,'method':name,'queries':len(ids),
                     **{k:float(mm[k]) for k in ('mADM','mAP','Rank-1')}})
HATA_DOM_TABLE=pd.DataFrame(rows)
HATA_DOM_TABLE.to_csv(HATA_RUN_DIR/'hata_audit_domains.csv',index=False)
noninferior=True
for d in range(3):
    sub=HATA_DOM_TABLE[HATA_DOM_TABLE.domain==d].set_index('method')
    if {'baseline','candidate'}.issubset(set(sub.index)):
        noninferior &= bool(sub.loc['candidate','mADM']>=sub.loc['baseline','mADM']-.01)
single_ok=single_b is None or single_c>=single_b-.007
HATA_AUDIT_ACCEPTED=bool(HATA_FIT_PASS and HATA_BEATS_OLD_FIT
    and HATA_A_CAND['mADM']>=HATA_A_BASE['mADM']+.005
    and HATA_A_CAND['mAP']>=HATA_A_BASE['mAP']-.005
    and HATA_A_CAND['Rank-1']>=HATA_A_BASE['Rank-1']-.005
    and HATA_A_CAND['mADM']>HATA_A_PRIOR['mADM']+.001
    and noninferior and single_ok)
HATA_REPORT={'method':'HATA hierarchical group-conditioned attribute-token representation (research proposal)',
  'train_only':True,'public_audit_previously_inspected':True,
  'checkpoint':str(HATA_BEST),'fit_baseline':HATA_BASE_FIT,
  'fit_selected':HATA_FIT_WIN,'fit_pass':HATA_FIT_PASS,'beats_old_DINO_fit':HATA_BEATS_OLD_FIT,
  'audit_baseline':HATA_A_BASE,'audit_prior_DINO':HATA_A_PRIOR,'audit_candidate':HATA_A_CAND,
  'audit_gain_mADM':HATA_A_CAND['mADM']-HATA_A_BASE['mADM'],
  'audit_gain_vs_previous_DINO':HATA_A_CAND['mADM']-HATA_A_PRIOR['mADM'],
  'single_positive_baseline':single_b,'single_positive_candidate':single_c,
  'domain_ok':bool(noninferior),'single_positive_ok':bool(single_ok),
  'audit_accepted':HATA_AUDIT_ACCEPTED,'hidden_codabench_score':None,
  'note':'PUBLIC exploratory VAL only. No VAL labels/IDs used to train or score. Not Codabench submission.'}
(HATA_RUN_DIR/'hata_audit_report.json').write_text(json.dumps(HATA_REPORT,indent=2,ensure_ascii=False))
print('AUDIT baseline:',HATA_A_BASE)
print('AUDIT earlier DINO:',HATA_A_PRIOR)
print('AUDIT HATA:',HATA_A_CAND)
print('Single-positive Rank1:',single_b,'->',single_c)
print('Domain metrics:\n',HATA_DOM_TABLE.to_string(index=False))
print('HATA_AUDIT_ACCEPTED:',HATA_AUDIT_ACCEPTED)



# =============================================================
#@title 52) HATA export experiment evidence ZIP (NOT a Codabench submission)
import zipfile,hashlib,json
from pathlib import Path

HATA_REPORT_ZIP=HATA_RUN_DIR/'ABPR_HATA_Representation_Experiment_Report.zip'
(HATA_RUN_DIR/'README.txt').write_text(
    'HATA hierarchical group-conditioned attribute-token adapter. Frozen Track1 DINOv2 tokens; '
    'TRAIN-only masked BCE plus explicit-conflict query ranking. Checkpoint selected using TRAIN holdout.\n'
    'FIT selected ranking blend; AUDIT previously repeatedly inspected, exploratory only.\n'
    'NOT a Codabench submission. For hidden, runtime must compute DINO tokens and HATA predictions '
    'for arbitrary gallery images offline, with challenge eligibility confirmed.\n',encoding='utf-8')
artifacts=[HATA_BEST,HATA_RUN_DIR/'train_history.json',HATA_RUN_DIR/'hata_fit_trials.csv',
           HATA_RUN_DIR/'hata_audit_domains.csv',HATA_RUN_DIR/'hata_audit_report.json',
           HATA_RUN_DIR/'reliability_train_only.npy',HATA_FILE,HATA_RUN_DIR/'README.txt']
with zipfile.ZipFile(HATA_REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for file in tqdm(artifacts,desc='HATA export report',unit='file'):
        assert Path(file).is_file(),file
        z.write(file,Path(file).name)
print('[EXPERIMENT REPORT]',HATA_REPORT_ZIP)
print('[CHECKPOINT]',HATA_BEST)
print('[AUDIT accepted]',HATA_AUDIT_ACCEPTED)
print('[SUBMISSION] NOT exported: offline runtime, licensing, and ingestion not yet verified.')

