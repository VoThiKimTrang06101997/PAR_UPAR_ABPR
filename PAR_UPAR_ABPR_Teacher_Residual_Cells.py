#@title 60) TCR — install teacher-constrained residual method + verify genuine DINO/HATA caches
from pathlib import Path
import base64,zlib,hashlib,sys,json
import numpy as np
from tqdm.auto import tqdm

required=['DINO_TRAIN_TOKENS','DINO_VAL_TOKENS','DINO_TRAIN_LABELS','DINO_TRAIN_DOMAINS',
          'DINO_TRAIN_PATHS','VAL_PATHS','HATA_TRAIN_PRED','HATA_VAL_TRACK1','HATA_BEST',
          'DINO_CK_SHA','DINO_TTA_FLIP','DINO_OUTPUT','TRACK1_ATTR','ATTR','DINO_TO_TRACK2',
          'Q','GIDS','VAL_LABELS','VAL_NAMES','SELECT_IDS','FIT_IDS','AUDIT_IDS',
          'BASE_DIST','HATA_REF_ALL','HATA_FIT_WIN','HATA_R40','DINO_RUN_SIGNATURE','dt','dino_metrics','hata']
missing=[x for x in required if x not in globals()]
if missing:
    raise RuntimeError('Missing in this Colab runtime: '+str(missing)+
      '. Restore cached states using original Cells 39–59. Frozen DINO tokens can be loaded from Drive; avoid retraining if saved.')
assert len(ATTR)==len(TRACK1_ATTR)==40 and set(ATTR)==set(TRACK1_ATTR)
assert np.array_equal(np.asarray(DINO_TO_TRACK2),[TRACK1_ATTR.index(n) for n in ATTR])
assert len(DINO_TRAIN_TOKENS)==len(HATA_TRAIN_PRED)==len(DINO_TRAIN_LABELS)==len(DINO_TRAIN_PATHS)
assert DINO_TRAIN_TOKENS.shape[1:]==DINO_VAL_TOKENS.shape[1:]==(40,192)
assert len(VAL_NAMES)==len(VAL_PATHS)==len(GIDS)==len(VAL_LABELS)==len(DINO_VAL_TOKENS)
assert np.asarray(HATA_VAL_TRACK1).shape==(len(VAL_NAMES),40)
assert np.array_equal(np.asarray(SELECT_IDS),np.r_[FIT_IDS,AUDIT_IDS])
assert np.asarray(BASE_DIST).shape==np.asarray(HATA_REF_ALL).shape==(len(SELECT_IDS),len(VAL_NAMES))
assert Path(HATA_BEST).is_file()
if 'manifest' in globals():
    assert dt.digest_lines(VAL_NAMES)==manifest['gallery_paths_sha256'], 'VAL order changed; STOP'
for split,paths in [('train',DINO_TRAIN_PATHS),('val',VAL_PATHS)]:
    meta_file=Path(DINO_OUTPUT)/f'{split}_features'/'meta.json'
    assert meta_file.is_file(),meta_file
    cm=json.loads(meta_file.read_text())
    expected=dt.digest_lines([DINO_CK_SHA,dt.digest_lines(TRACK1_ATTR),dt.digest_lines(paths),'flip'+str(DINO_TTA_FLIP)])
    assert cm.get('signature')==expected and cm.get('completed')==len(paths),f'Stale {split} DINO cache; STOP'
source=zlib.decompress(base64.b64decode('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'))
assert hashlib.sha256(source).hexdigest()=="b3473fdcb9b5efa4a27031938104e47014de4c2486f2d6b32daf954baf04191e", 'Corrupt embedded module'
TCR_MODULE=Path('/content/abpr_teacher_residual.py');TCR_MODULE.write_bytes(source)
sys.path.insert(0,'/content')
import importlib,abpr_teacher_residual as tcr
tcr=importlib.reload(tcr)
TCR_DIR=Path(DINO_OUTPUT)/'teacher_constrained_residual'
TCR_DIR.mkdir(parents=True,exist_ok=True)
TCR_SIGNATURE=tcr.digest(DINO_RUN_SIGNATURE,hashlib.sha256(Path(HATA_BEST).read_bytes()).hexdigest(),
                         dt.digest_lines(TRACK1_ATTR),np.asarray(DINO_TRAIN_LABELS,dtype=np.float32),
                         np.asarray(DINO_TRAIN_DOMAINS,dtype=np.int64),dt.digest_lines(DINO_TRAIN_PATHS))
print('[60 PASS] TRAIN',len(DINO_TRAIN_TOKENS),'VAL',len(DINO_VAL_TOKENS),'FIT',len(FIT_IDS),'AUDIT',len(AUDIT_IDS))
print('Teacher:',HATA_BEST,'Output:',TCR_DIR)
print('All training labels come from TRAIN; validation labels are not used for scoring.')

#@title 61) TCR — TRAIN-only residual representation learning (zero-init, resume, progress)
TCR_EPOCHS=12 #@param {type:'integer'}
TCR_BATCH=96 #@param {type:'integer'}
TCR_LR=0.001 #@param {type:'number'}
TCR_RANK_WEIGHT=0.08 #@param {type:'number'}
TCR_MAX_CORRECTION=0.8 #@param {type:'number'}
TCR_FORCE_NEW=False #@param {type:'boolean'}

# This computes losses from TRAIN annotations only; frozen DINO and HATA weights are untouched.
assert 1<=TCR_EPOCHS<=200 and 16<=TCR_BATCH<=512
TCR_RUN=TCR_DIR/f'ep{TCR_EPOCHS}_lr{TCR_LR}_rank{TCR_RANK_WEIGHT}_cap{TCR_MAX_CORRECTION}'
TCR_MODEL,TCR_HISTORY,TCR_BEST=tcr.train(
    DINO_TRAIN_TOKENS,HATA_TRAIN_PRED,DINO_TRAIN_LABELS,DINO_TRAIN_DOMAINS,
    TCR_RUN,TCR_SIGNATURE,epochs=int(TCR_EPOCHS),batch_size=int(TCR_BATCH),
    lr=float(TCR_LR),rank_weight=float(TCR_RANK_WEIGHT),
    max_correction=float(TCR_MAX_CORRECTION),seed=8719,
    force=bool(TCR_FORCE_NEW))
print('[61 PASS] Best TRAIN-holdout checkpoint:',TCR_BEST)
print('Backbone and HATA teacher NOT trained again.')

#@title 62) TCR — resume VAL logits cache and compute label-free query distances
import numpy as np,hashlib,json
from pathlib import Path
TCR_CK_HASH=dt.hash_file(TCR_BEST)
TCR_VAL_SIGNATURE=tcr.digest(TCR_SIGNATURE,TCR_CK_HASH,dt.digest_lines(VAL_PATHS),len(VAL_PATHS))
TCR_VAL_NATIVE=tcr.infer(TCR_MODEL,DINO_VAL_TOKENS,HATA_VAL_TRACK1,
                          TCR_RUN/'val_teacher_residual_logits.npy',
                          signature=TCR_VAL_SIGNATURE,batch_size=192)
TCR_VAL_40=np.asarray(TCR_VAL_NATIVE,dtype=np.float32)[:,DINO_TO_TRACK2]
assert TCR_VAL_40.shape==(len(VAL_PATHS),40)
TCR_NLL=hata.weighted_query_nll(Q[SELECT_IDS],TCR_VAL_40,weights=HATA_R40,chunk=48)
assert TCR_NLL.shape==BASE_DIST.shape and np.isfinite(TCR_NLL).all()
# Sanity: teacher + bounded residual should not produce uncontrolled large logit changes.
max_delta=float(np.max(np.abs(np.asarray(TCR_VAL_NATIVE)-np.asarray(HATA_VAL_TRACK1))))
assert max_delta<=float(TCR_MAX_CORRECTION)+1e-4,(max_delta,TCR_MAX_CORRECTION)
print('[62 PASS] Same-gallery distance:',TCR_NLL.shape,'max logit correction:',round(max_delta,6))
print('No VAL_LABELS or GIDS used to compute these scores.')

#@title 63) TCR — FIT-only candidate selection, protect Rank-1 and single-positive
import numpy as np,pandas as pd,json
from tqdm.auto import tqdm
NF=len(FIT_IDS)
TCR_HATA_F=np.asarray(HATA_REF_ALL[:NF],np.float32)
TCR_BASE_F=np.asarray(BASE_DIST[:NF],np.float32)
TCR_F_B=dino_metrics(TCR_BASE_F,FIT_IDS)
TCR_F_H=dino_metrics(TCR_HATA_F,FIT_IDS)
count=np.bincount(np.asarray(GIDS,dtype=np.int64),minlength=len(Q))[FIT_IDS]
single=count==1
h_top=(np.asarray(GIDS)[np.argmin(TCR_HATA_F,axis=1)]==FIT_IDS)

def tcr_fused(method,alpha,idx):
    b=np.asarray(BASE_DIST[idx],np.float32)
    h=np.asarray(HATA_REF_ALL[idx],np.float32)
    correction=np.asarray(TCR_NLL[idx],np.float32)
    if method=='replace_HATA':return dt.fusion(b,correction,float(HATA_FIT_WIN['alpha']))
    if method=='residual_on_HATA':return dt.fusion(h,correction,float(alpha))
    if method=='residual_on_baseline':return dt.fusion(b,correction,float(alpha))
    raise ValueError(method)

configs=[('replace_HATA',float(HATA_FIT_WIN['alpha']))]+[(m,a) for m in
         ('residual_on_HATA','residual_on_baseline') for a in (0.02,0.04,0.075,0.10,0.15)]
rows=[]
for method,alpha in tqdm(configs,desc='TCR FIT candidate evaluation',unit='trial'):
    ix=np.arange(NF)
    dd=tcr_fused(method,alpha,ix)
    m=dino_metrics(dd,FIT_IDS)
    top=(np.asarray(GIDS)[np.argmin(dd,axis=1)]==FIT_IDS)
    sr=float(top[single].mean()-h_top[single].mean()) if single.any() else 0.
    row={'method':method,'alpha':float(alpha),**{k:float(m[k]) for k in ('mADM','mAP','Rank-1')},
         'gain_mADM_vs_HATA':float(m['mADM']-TCR_F_H['mADM']),
         'delta_mAP_vs_HATA':float(m['mAP']-TCR_F_H['mAP']),
         'delta_R1_vs_HATA':float(m['Rank-1']-TCR_F_H['Rank-1']),
         'delta_single_R1_vs_HATA':sr}
    # Screening thresholds are researcher choices, NOT organizer criteria.
    row['fit_pass']=bool(row['gain_mADM_vs_HATA']>=.002 and
                          row['delta_mAP_vs_HATA']>=-.002 and
                          row['delta_R1_vs_HATA']>=-.005 and sr>=-.007)
    rows.append(row)
TCR_FIT_TABLE=pd.DataFrame(rows).sort_values(['mADM','mAP','Rank-1'],ascending=False)
TCR_FIT_TABLE.to_csv(TCR_RUN/'tcr_fit_trials.csv',index=False)
passed=TCR_FIT_TABLE[TCR_FIT_TABLE.fit_pass]
TCR_WIN=(passed.iloc[0] if len(passed) else TCR_FIT_TABLE.iloc[0]).to_dict()
TCR_FIT_PASS=bool(TCR_WIN['fit_pass'])
tcr.safe_json(TCR_RUN/'tcr_fit_choice.json',{'winner':TCR_WIN,'fit_pass':TCR_FIT_PASS,
    'fit_baseline':{k:float(TCR_F_B[k]) for k in ('mADM','mAP','Rank-1')},
    'fit_HATA':{k:float(TCR_F_H[k]) for k in ('mADM','mAP','Rank-1')},
    'selection':'FIT-only; AUDIT not used to select method/alpha'})
print('FIT baseline:',TCR_F_B,'FIT HATA:',TCR_F_H)
print('Top FIT trials:\n',TCR_FIT_TABLE.head(10).to_string(index=False))
print('Selected:',TCR_WIN,'FIT PASS:',TCR_FIT_PASS)

#@title 64) TCR — AUDIT once, mADM/mAP/Rank-1, domains, single-positive safeguard
import numpy as np,pandas as pd,json
from tqdm.auto import tqdm
from abpr.core import infer_domain_id
from abpr.query_aware_ranking import retrieval_metrics
from abpr.model import track2_metrics
indices=np.arange(NF,len(SELECT_IDS))
TCR_A_B=np.asarray(BASE_DIST[indices],np.float32)
TCR_A_H=np.asarray(HATA_REF_ALL[indices],np.float32)
TCR_A_C=tcr_fused(TCR_WIN['method'],TCR_WIN['alpha'],indices)
TCR_A_BASE=dino_metrics(TCR_A_B,AUDIT_IDS)
TCR_A_HATA=dino_metrics(TCR_A_H,AUDIT_IDS)
TCR_A_NEW=dino_metrics(TCR_A_C,AUDIT_IDS)
pos=np.bincount(np.asarray(GIDS,dtype=np.int64),minlength=len(Q))[AUDIT_IDS]
single=pos==1
oldtop=np.asarray(GIDS)[np.argmin(TCR_A_H,axis=1)]==AUDIT_IDS
newtop=np.asarray(GIDS)[np.argmin(TCR_A_C,axis=1)]==AUDIT_IDS
single_old=float(oldtop[single].mean()) if single.any() else None
single_new=float(newtop[single].mean()) if single.any() else None
source_domains=np.asarray([infer_domain_id(x) for x in VAL_NAMES],dtype=np.int64)
rows=[]
index_map={int(x):j for j,x in enumerate(AUDIT_IDS)}
for d in tqdm(range(3),desc='TCR AUDIT per domain',unit='domain'):
    mask=source_domains==d
    local_ids=AUDIT_IDS[np.isin(AUDIT_IDS,np.unique(np.asarray(GIDS)[mask]))]
    if len(local_ids)==0:continue
    rr=[index_map[int(x)] for x in local_ids]
    for name,scores in [('baseline',TCR_A_B),('HATA',TCR_A_H),('TCR',TCR_A_C)]:
        sub=scores[rr][:,mask]
        mm=retrieval_metrics(sub,local_ids,np.asarray(GIDS)[mask]);
        mm.update(track2_metrics(sub,Q[local_ids],np.asarray(VAL_LABELS)[mask]))
        rows.append({'domain':int(d),'method':name,'queries':len(local_ids),
                     **{k:float(mm[k]) for k in ('mADM','mAP','Rank-1')}})
TCR_DOM=pd.DataFrame(rows);TCR_DOM.to_csv(TCR_RUN/'tcr_audit_domains.csv',index=False)
domains_ok=True
for d in range(3):
    sub=TCR_DOM[TCR_DOM.domain==d].set_index('method')
    if {'HATA','TCR'}.issubset(sub.index):
        domains_ok &= bool(sub.loc['TCR','mADM']>=sub.loc['HATA','mADM']-.008)
single_ok=(single_old is None or single_new >= single_old-.007)
TCR_AUDIT_ACCEPTED=bool(TCR_FIT_PASS and
  TCR_A_NEW['mADM'] >= TCR_A_HATA['mADM']+.002 and
  TCR_A_NEW['mAP'] >= TCR_A_HATA['mAP']-.002 and
  TCR_A_NEW['Rank-1'] >= TCR_A_HATA['Rank-1']-.005 and domains_ok and single_ok)
TCR_REPORT={'method':'TRAIN-only teacher-constrained residual attribute representation',
 'checkpoint':str(TCR_BEST),'checkpoint_sha256':TCR_CK_HASH,
 'baseline_public_audit':{k:float(TCR_A_BASE[k]) for k in ('mADM','mAP','Rank-1')},
 'HATA_public_audit':{k:float(TCR_A_HATA[k]) for k in ('mADM','mAP','Rank-1')},
 'TCR_public_audit':{k:float(TCR_A_NEW[k]) for k in ('mADM','mAP','Rank-1')},
 'fit_winner':TCR_WIN,'fit_pass':bool(TCR_FIT_PASS),
 'single_positive_HATA_R1':single_old,'single_positive_TCR_R1':single_new,
 'domain_ok':bool(domains_ok),'single_positive_ok':bool(single_ok),
 'audit_accepted':bool(TCR_AUDIT_ACCEPTED),'hidden_codabench_score':None,
 'public_audit_previously_inspected':True,
 'eligibility_external_pretrained_confirmed':False,
 'note':'Exploratory public VAL. No VAL labels used in model training or score generation. NOT a Codabench submission.'}
tcr.safe_json(TCR_RUN/'tcr_audit_report.json',TCR_REPORT)
print('PUBLIC baseline:',TCR_A_BASE,'\nHATA:',TCR_A_HATA,'\nTCR:',TCR_A_NEW)
print('SINGLE-POSITIVE HATA -> TCR:',single_old,'->',single_new)
print('DOMAIN AUDIT:\n',TCR_DOM.to_string(index=False))
print('TCR_AUDIT_ACCEPTED:',TCR_AUDIT_ACCEPTED)
print('Public AUDIT has been inspected repeatedly. Hidden performance UNKNOWN.')

#@title 65) TCR — package experiment evidence (NOT Codabench submission)
from pathlib import Path
import zipfile,json,hashlib
from tqdm.auto import tqdm
readme=TCR_RUN/'README.txt'
readme.write_text('Teacher-Constrained Residual Representation for Attribute-Based Person Retrieval.\n'
 'Inputs: frozen DINO Track1 40x192 tokens, HATA teacher logits. TRAIN-only BCE + multi-positive TRAIN query ranking + teacher preservation.\n'
 'Checkpoint selection: TRAIN holdout; fusion selection: public FIT; AUDIT exploratory (seen earlier).\n'
 'Does not reproduce a named paper, not an official Codabench ingestion/runtime ZIP.\n'
 'Do not use public VAL_LABELS or GIDS in hidden inference. Offline runtime on hidden images is NOT implemented by this experiment.\n'
 'Using Track1 DINO weights on Track2 requires verification of challenge eligibility and model licensing.\n',encoding='utf-8')
TCR_REPORT_ZIP=TCR_RUN/'ABPR_Teacher_Residual_Experiment_Report.zip'
files=[TCR_MODULE,TCR_BEST,TCR_RUN/'train_history.json',TCR_RUN/'tcr_fit_trials.csv',
       TCR_RUN/'tcr_fit_choice.json',TCR_RUN/'tcr_audit_domains.csv',
       TCR_RUN/'tcr_audit_report.json',readme]
with zipfile.ZipFile(TCR_REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for f in tqdm(files,desc='Package TCR report',unit='file'):
        f=Path(f);assert f.is_file(),f
        z.write(f,arcname=f.name)
print('REPORT:',TCR_REPORT_ZIP)
print('Checkpoint:',TCR_BEST)
print('TCR AUDIT ACCEPTED:',TCR_AUDIT_ACCEPTED)
print('Never submit report ZIP to Codabench.')

#@title 66) TCR — decision dashboard and next research fork
import json
r=json.loads((TCR_RUN/'tcr_audit_report.json').read_text())
print('='*74)
print('CONSERVATIVE REPRESENTATION CONTINUATION — DECISION')
print('='*74)
for k in ['baseline_public_audit','HATA_public_audit','TCR_public_audit']:
    print(k, r[k])
print('Best FIT:',r['fit_winner'])
print('FIT passed:',r['fit_pass'],'AUDIT accepted:',r['audit_accepted'])
if not r['audit_accepted']:
    print('RECOMMENDATION: KEEP existing HATA/best verified hidden submission.')
    print('Do not repeatedly retune against the previously inspected public AUDIT.')
    print('Next methodological lever is a carefully controlled domain-holdout and backbone-level fine-tune,')
    print('which requires new training/validation design and more GPU; no 0.65–0.68 guarantee.')
else:
    print('Promising public validation signal only. Still NOT Codabench-ready.')
    print('Before submission: confirm eligibility, implement offline frozen-DINO/HATA/TCR runtime,')
    print('run ingestion smoke test and validate memory/time on 29k+ gallery.')
print('Experiment ZIP:',TCR_REPORT_ZIP)