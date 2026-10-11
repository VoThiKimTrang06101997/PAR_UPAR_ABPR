#@title 39) DINO Track1 -> ABPR Track2 — restore, verify checkpoint, gallery, 40-attribute ordering
from pathlib import Path
import os,sys,json,hashlib,subprocess,importlib,zipfile,time
import numpy as np,pandas as pd,torch
from tqdm.auto import tqdm
from google.colab import drive

DINO_ROOT=Path('/content/drive/MyDrive/PedestrianAttributeRecognition')
if not DINO_ROOT.parent.is_dir(): drive.mount('/content/drive')
DINO_OUTPUT=DINO_ROOT/'ABPR_Results'/'dino_attribute_token_transfer'
DINO_OUTPUT.mkdir(parents=True,exist_ok=True)
SOURCE_ROOT=Path('/content/PAR_UPAR_ABPR')
TRACK1_ROOT=Path('/content/PAR_UPAR')
OFFICIAL_ROOT=Path('/content/UPAR-Challenge-2027')
EXPECTED_GIT={SOURCE_ROOT:'https://github.com/VoThiKimTrang06101997/PAR_UPAR_ABPR.git',
              TRACK1_ROOT:'https://github.com/VoThiKimTrang06101997/PAR_UPAR.git'}
for folder,url in EXPECTED_GIT.items():
    if not folder.is_dir():
        subprocess.run(['git','clone','--depth','1',url,str(folder)],check=True)
    elif (folder/'.git').is_dir():
        actual=subprocess.check_output(['git','-C',str(folder),'remote','get-url','origin'],text=True).strip()
        if actual.rstrip('/').removesuffix('.git')!=url.rstrip('/').removesuffix('.git'):
            raise RuntimeError(f'Unexpected git origin at {folder}: {actual}')
    print('[SOURCE]',folder)
for needed in (TRACK1_ROOT/'strong_par.py',SOURCE_ROOT/'abpr/core.py',SOURCE_ROOT/'abpr/model.py'):
    if not needed.is_file():
        raise FileNotFoundError('Required project module missing: '+str(needed)+
          '. Run Track1 source setup Cell 3 or Track2 source setup Cell 2 as appropriate.')
sys.path[:0]=[str(SOURCE_ROOT),str(TRACK1_ROOT)]
from abpr.core import ATTRIBUTE_NAMES,detect_image_column,ImageResolver
from abpr.model import load_query_csv,track2_metrics
from abpr.query_aware_ranking import retrieval_metrics
from strong_par import ATTRIBUTE_NAMES as TRACK1_ATTRIBUTE_NAMES
ATTR=list(ATTRIBUTE_NAMES);TRACK1_ATTR=list(TRACK1_ATTRIBUTE_NAMES)
assert len(ATTR)==len(TRACK1_ATTR)==40
if set(ATTR)!=set(TRACK1_ATTR):
    print('TRACK1 names:',TRACK1_ATTR);print('TRACK2 names:',ATTR)
    raise ValueError('Different 40 attribute vocabularies. Do not transfer without explicit audited mapping.')
DINO_TO_TRACK2=np.array([TRACK1_ATTR.index(name) for name in ATTR],dtype=np.int64)
print('[ORDER] Track1 -> Track2 permutation:',DINO_TO_TRACK2.tolist())

# Self-contained DINO runtime and transfer training module; no huggingface weights downloaded.
import base64,zlib
transfer_bytes=zlib.decompress(base64.b64decode('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'))
dino_bytes=zlib.decompress(base64.b64decode('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'))
DINO_MODULE=Path('/content/abpr_dino_transfer.py')
DINO_MODEL_FILE=Path('/content/dino_transfer_model.py')
DINO_MODULE.write_bytes(transfer_bytes)
DINO_MODEL_FILE.write_bytes(dino_bytes)
sys.path.insert(0,'/content')
import abpr_dino_transfer as dt

DINO_CHECKPOINT=DINO_ROOT/'Checkpoints'/'DinoAttributeTraining'/'dinov2_attribute_seed42_best.pt'
if not DINO_CHECKPOINT.is_file():raise FileNotFoundError('Track1 DINO checkpoint missing: '+str(DINO_CHECKPOINT))
DINO_CK_SHA=dt.hash_file(DINO_CHECKPOINT)
print('[TRACK1 DINO CKPT]',DINO_CHECKPOINT,'SHA256',DINO_CK_SHA[:20])

SIGLIP_DIR=DINO_ROOT/'ABPR_Results'/'siglip2_retrieval'
manifest_file=SIGLIP_DIR/'fixed_public_val_split.json'
baseline_file=SIGLIP_DIR/'baseline_fixed_queries.npy'
if not manifest_file.is_file() or not baseline_file.is_file():
    raise FileNotFoundError('Need Cell 8–9: missing fixed_public_val_split.json or baseline_fixed_queries.npy')
manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
BEST_SUBMISSION_ZIP=DINO_ROOT/'ABPR_Results'/'Submission_ABPR.zip'
if not BEST_SUBMISSION_ZIP.is_file():
    raise FileNotFoundError('Original Track2 0.5389 submission ZIP missing: '+str(BEST_SUBMISSION_ZIP))
BASELINE_INFO=SIGLIP_DIR/'baseline_fixed_queries.json'
if not BASELINE_INFO.is_file():
    raise FileNotFoundError('Missing exact baseline cache signature JSON from Cell 9: '+str(BASELINE_INFO))
FIT_IDS=np.array(manifest['fit_ids'],dtype=np.int64)
AUDIT_IDS=np.array(manifest['audit_ids'],dtype=np.int64)
SELECT_IDS=np.r_[FIT_IDS,AUDIT_IDS]
import hashlib
baseline_signature=hashlib.sha256()
for item in [dt.hash_file(BEST_SUBMISSION_ZIP),manifest['gallery_paths_sha256'],*SELECT_IDS]:
    baseline_signature.update(str(item).encode('utf-8')); baseline_signature.update(b'\n')
cacheinfo=json.loads(BASELINE_INFO.read_text(encoding='utf-8'))
if cacheinfo.get('signature')!=baseline_signature.hexdigest():
    raise RuntimeError('BASE_DIST is not an exact output of the current baseline ZIP. Rerun Cell 9; refusing stale cache.')
BASE_DIST=np.load(baseline_file,mmap_mode='r',allow_pickle=False)
VAL_CSV=OFFICIAL_ROOT/'data/annotations/task2/val/gt.csv'
TRAIN_CSV=OFFICIAL_ROOT/'data/annotations/task2/train/gt.csv'
for p in (VAL_CSV,TRAIN_CSV):
    if not p.is_file():raise FileNotFoundError('Official annotations missing: '+str(p))
val=pd.read_csv(VAL_CSV);train=pd.read_csv(TRAIN_CSV)
for frame in (val,train):
    if not set(ATTR).issubset(frame.columns):raise RuntimeError('40 attribute columns missing')
valcol=detect_image_column(val);traincol=detect_image_column(train)
VAL_NAMES=val[valcol].astype(str).tolist()
assert dt.digest_lines(VAL_NAMES)==manifest['gallery_paths_sha256'], 'Gallery order changed — STOP'
assert len(VAL_NAMES)==BASE_DIST.shape[1] and BASE_DIST.shape[0]==len(SELECT_IDS)

# Exact path/id mapping, never positional join of id.csv
ids=pd.read_csv(OFFICIAL_ROOT/'data/annotations/task2/val/ids.csv',header=None,dtype=str)
lookup={}
for a,b in ids.iloc[:,:2].itertuples(index=False,name=None):
    k=str(a).strip().replace('\\','/').lstrip('./')
    try:v=int(float(b))
    except (TypeError,ValueError):
        if k.lower() in ('path','image','image_path'):continue
        raise
    if k in lookup and lookup[k]!=v:raise ValueError('Conflicting ID mapping: '+k)
    lookup[k]=v
GIDS=np.array([lookup[str(v).strip().replace('\\','/').lstrip('./')] for v in VAL_NAMES],np.int32)
_,Q=load_query_csv(OFFICIAL_ROOT/'data/annotations/task2/val/queries.csv')
Q=np.asarray(Q,np.float32)
VAL_LABELS=val[ATTR].to_numpy(np.float32) # evaluation ONLY
assert Q.shape[1]==40 and np.isin(Q,[-1,0,1]).all()
assert 0<=GIDS.min() and GIDS.max()<len(Q) and SELECT_IDS.max()<len(Q)

def dino_metrics(dist,ids,gt=VAL_LABELS):
    m=retrieval_metrics(np.asarray(dist,np.float32),np.asarray(ids),GIDS)
    m.update(track2_metrics(np.asarray(dist,np.float32),Q[ids],gt))
    return {k:float(m[k]) for k in ('mADM','mAP','Rank-1')}

DATA_ROOT=OFFICIAL_ROOT/'data'
DINO_RESOLVER=ImageResolver(DATA_ROOT,OFFICIAL_ROOT,
    extra_roots=[p for p in [DINO_ROOT/'datasets',DINO_ROOT/'ABPR_Data',DINO_ROOT/'data'] if p.is_dir()])
# Reuse VAL_PATHS only if exact correspondence to VAL_NAMES can be checked.
print('[PREPARE] Resolve source image paths; no recursive Google Drive scan')
VAL_PATHS=[str(DINO_RESOLVER.resolve(x)) for x in tqdm(VAL_NAMES,desc='Exact gallery filenames',unit='img')]
assert len(VAL_PATHS)==len(VAL_NAMES) and all(Path(p).is_file() for p in VAL_PATHS)

# Balanced sample TRAIN only. No VAL labels used for training.
from abpr.core import infer_domain_id
DINO_TRAIN_PER_DOMAIN=1500 #@param {type:'integer'}
DINO_BATCH=16 #@param {type:'integer'}
DINO_TTA_FLIP=True #@param {type:'boolean'}
assert DINO_TRAIN_PER_DOMAIN>100 and DINO_BATCH>=1
rng=np.random.default_rng(42)
train_domains=np.array([infer_domain_id(x) for x in train[traincol].astype(str)])
selected=[]
for domain in (0,1,2):
    subset=np.flatnonzero(train_domains==domain)
    if not len(subset):raise RuntimeError('Missing TRAIN domain '+str(domain))
    selected.extend(rng.choice(subset,size=min(DINO_TRAIN_PER_DOMAIN,len(subset)),replace=False).tolist())
rng.shuffle(selected)
DINO_TRAIN_FRAME=train.iloc[selected].reset_index(drop=True)
DINO_TRAIN_PATHS=[str(DINO_RESOLVER.resolve(x)) for x in tqdm(DINO_TRAIN_FRAME[traincol].astype(str),desc='TRAIN image filenames',unit='img')]
DINO_TRAIN_LABELS=DINO_TRAIN_FRAME[TRACK1_ATTR].to_numpy(np.float32)
DINO_TRAIN_DOMAINS=np.array([infer_domain_id(x) for x in DINO_TRAIN_FRAME[traincol].astype(str)])
assert set(np.unique(DINO_TRAIN_DOMAINS))=={0,1,2}
DINO_RUN_SIGNATURE=dt.digest_lines([DINO_CK_SHA,dt.digest_lines(TRACK1_ATTR),str(DINO_TTA_FLIP),
                                   dt.digest_lines(DINO_TRAIN_PATHS),dt.digest_lines(VAL_PATHS)])
(DINO_OUTPUT/'run_signature.json').write_text(json.dumps({'signature':DINO_RUN_SIGNATURE,
  'teacher_sha256':DINO_CK_SHA,'train_count':len(DINO_TRAIN_PATHS),'gallery_count':len(VAL_PATHS),
  'track1_attr':TRACK1_ATTR,'track2_attr':ATTR,'flip':DINO_TTA_FLIP},indent=2))
print('[PASS] FIT',len(FIT_IDS),'AUDIT',len(AUDIT_IDS),'gallery',len(VAL_PATHS),'TRAIN',len(DINO_TRAIN_PATHS))
print('[CAUTION] Source AUDIT has already been inspected in past runs; exploratory result, not independent hidden score.')


#@title 40) Frozen Track1 DINOv2 attribute-token features — RESUMABLE TRAIN + full VAL
import gc
DINO_DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
if DINO_DEVICE!='cuda':raise RuntimeError('Enable Colab GPU: Runtime > Change runtime type > GPU')
DINO_TEACHER,DINO_CK_META=dt.load_teacher(DINO_CHECKPOINT,DINO_MODEL_FILE,DINO_DEVICE)
print('[TEACHER] DINOv2-S; attributes',DINO_TEACHER.num_attributes,
      'validation mA metadata:',DINO_CK_META.get('validation_metrics',{}).get('mA'))
train_sig=dt.digest_lines([DINO_CK_SHA,dt.digest_lines(TRACK1_ATTR),dt.digest_lines(DINO_TRAIN_PATHS),'flip'+str(DINO_TTA_FLIP)])
val_sig=dt.digest_lines([DINO_CK_SHA,dt.digest_lines(TRACK1_ATTR),dt.digest_lines(VAL_PATHS),'flip'+str(DINO_TTA_FLIP)])
DINO_TRAIN_TOKENS,DINO_TRAIN_LOGITS=dt.cache_teacher(DINO_TEACHER,DINO_TRAIN_PATHS,
  DINO_OUTPUT/'train_features',train_sig,batch_size=DINO_BATCH,flip=DINO_TTA_FLIP,device=DINO_DEVICE)
DINO_VAL_TOKENS,DINO_VAL_LOGITS=dt.cache_teacher(DINO_TEACHER,VAL_PATHS,
  DINO_OUTPUT/'val_features',val_sig,batch_size=DINO_BATCH,flip=DINO_TTA_FLIP,device=DINO_DEVICE)
assert DINO_TRAIN_TOKENS.shape==(len(DINO_TRAIN_PATHS),40,192)
assert DINO_VAL_TOKENS.shape==(len(VAL_PATHS),40,192)
print('[PASS] Frozen DINO features cached; TRAIN',DINO_TRAIN_TOKENS.shape,
      'VAL',DINO_VAL_TOKENS.shape)
del DINO_TEACHER;gc.collect();torch.cuda.empty_cache()


#@title 41) TRAIN-only DINO attribute-token representation adapter — no backbone training
DINO_EPOCHS=12 #@param {type:'integer'}
DINO_HEAD_BATCH=128 #@param {type:'integer'}
DINO_HEAD_LR=0.003 #@param {type:'number'}
DINO_RANK_WEIGHT=0.07 #@param {type:'number'}
DINO_FULL_QUERY_PROB=0.65 #@param {type:'number'}
DINO_HEAD_PATH=DINO_OUTPUT/'dino_token_head_last.pt'
assert DINO_EPOCHS>=1 and DINO_HEAD_BATCH>=12
DINO_HEAD,DINO_TRAIN_HISTORY=dt.train_head(DINO_TRAIN_TOKENS,DINO_TRAIN_LOGITS,
    DINO_TRAIN_LABELS,DINO_TRAIN_DOMAINS,DINO_HEAD_PATH,epochs=DINO_EPOCHS,
    batch_size=DINO_HEAD_BATCH,lr=DINO_HEAD_LR,rank_weight=DINO_RANK_WEIGHT,
    full_query_probability=DINO_FULL_QUERY_PROB)
(DINO_OUTPUT/'train_history.json').write_text(json.dumps(DINO_TRAIN_HISTORY,indent=2))
print('[PASS] TRAIN-only best head:',DINO_HEAD_PATH.with_name('dino_token_head_best.pt'))


#@title 42) DINO-only vs DINO-adapted query distances — public VAL labels NOT used for scoring
DINO_RAW_40=np.asarray(DINO_VAL_LOGITS,np.float32)[:,DINO_TO_TRACK2]
DINO_HEAD_40=dt.head_logits(DINO_HEAD,DINO_VAL_TOKENS,DINO_VAL_LOGITS)[:,DINO_TO_TRACK2]
DINO_Q=Q[SELECT_IDS]
DINO_RAW_DIST=dt.query_nll(DINO_Q,DINO_RAW_40,chunk=64)
DINO_HEAD_DIST=dt.query_nll(DINO_Q,DINO_HEAD_40,chunk=64)
assert DINO_RAW_DIST.shape==DINO_HEAD_DIST.shape==BASE_DIST.shape
assert np.isfinite(DINO_HEAD_DIST).all()
print('DINO scores:',DINO_RAW_DIST.shape,'baseline',BASE_DIST.shape)


#@title 43) FIT-only model selection: raw DINO, learned DINO and fusion, protect single-positive queries
from tqdm.auto import tqdm
DINO_ALPHA_GRID=[0.0,0.02,0.05,0.1,0.2,0.35,0.5,1.0]
N_FIT=len(FIT_IDS)
BF=np.asarray(BASE_DIST[:N_FIT],np.float32)
base_fit=dino_metrics(BF,FIT_IDS)
count=np.bincount(GIDS,minlength=len(Q))[FIT_IDS]
single=count==1
old_top=(GIDS[np.argmin(BF,axis=1)]==FIT_IDS)
fit_rows=[]
for method,newmat in [('DINO_raw',DINO_RAW_DIST),('DINO_token_head',DINO_HEAD_DIST)]:
    for a in tqdm(DINO_ALPHA_GRID,desc='FIT DINO '+method,unit='fusion'):
        scores=dt.fusion(BF,newmat[:N_FIT],a)
        m=dino_metrics(scores,FIT_IDS)
        newtop=(GIDS[np.argmin(scores,axis=1)]==FIT_IDS)
        sd=float(newtop[single].mean()-old_top[single].mean()) if single.any() else 0.
        row={'method':method,'alpha':a,**m,'single_positive_rank1_delta':sd,
             'delta_mADM':m['mADM']-base_fit['mADM'],
             'delta_mAP':m['mAP']-base_fit['mAP'],
             'delta_Rank-1':m['Rank-1']-base_fit['Rank-1']}
        row['fit_pass']=bool(a>0 and row['delta_mADM']>=.003
               and row['delta_mAP']>=-.005 and row['delta_Rank-1']>=-.005
               and sd>=-.007)
        fit_rows.append(row)
DINO_FIT_TABLE=pd.DataFrame(fit_rows).sort_values(['mADM','mAP'],ascending=False)
DINO_FIT_TABLE.to_csv(DINO_OUTPUT/'dino_fit_trials.csv',index=False)
possible=DINO_FIT_TABLE[DINO_FIT_TABLE.fit_pass]
if len(possible):
    DINO_WIN=possible.iloc[0].to_dict();DINO_FIT_PASS=True
else:
    DINO_WIN={'method':'BASELINE','alpha':0.0};DINO_FIT_PASS=False
(DINO_OUTPUT/'dino_fit_selection.json').write_text(json.dumps({'selection':DINO_WIN,'fit_pass':DINO_FIT_PASS,'baseline':base_fit},indent=2))
print('FIT baseline',base_fit)
print(DINO_FIT_TABLE.head(8).to_string(index=False))
print('SELECTED (FIT ONLY):',DINO_WIN,'FIT_PASS:',DINO_FIT_PASS)


#@title 44) AUDIT: same-gallery mADM, mAP, Rank-1, each domain, single-positive gate
from abpr.core import infer_domain_id
BA=np.asarray(BASE_DIST[N_FIT:],np.float32)
raw=(DINO_RAW_DIST if DINO_WIN['method']=='DINO_raw' else DINO_HEAD_DIST)
CA=dt.fusion(BA,raw[N_FIT:],float(DINO_WIN['alpha'])) if DINO_FIT_PASS else BA.copy()
base_audit=dino_metrics(BA,AUDIT_IDS)
cand_audit=dino_metrics(CA,AUDIT_IDS)
# Single-positive ID group. These IDs are used ONLY for evaluation, not scoring.
positive_count=np.bincount(GIDS,minlength=len(Q))[AUDIT_IDS]
one=positive_count==1
correct_b=GIDS[np.argmin(BA,axis=1)]==AUDIT_IDS
correct_c=GIDS[np.argmin(CA,axis=1)]==AUDIT_IDS
single_b=float(correct_b[one].mean()) if one.any() else None
single_c=float(correct_c[one].mean()) if one.any() else None
val_domains=np.array([infer_domain_id(p) for p in VAL_NAMES])
domain_rows=[]
for d in tqdm(range(3),desc='DINO source-domain AUDIT',unit='domain'):
    gmask=val_domains==d
    qmask=np.isin(AUDIT_IDS,np.unique(GIDS[gmask]))
    subset_ids=AUDIT_IDS[qmask]
    if not len(subset_ids):continue
    for label,scores in [('baseline',BA),('dino_candidate',CA)]:
        small=scores[qmask][:,gmask]
        m=retrieval_metrics(small,subset_ids,GIDS[gmask])
        m.update(track2_metrics(small,Q[subset_ids],VAL_LABELS[gmask]))
        domain_rows.append({'domain':d,'method':label,'queries':len(subset_ids),
           **{k:float(m[k]) for k in ('mADM','mAP','Rank-1')}})
domain_table=pd.DataFrame(domain_rows)
domain_table.to_csv(DINO_OUTPUT/'dino_audit_domains.csv',index=False)
domain_ok=True
for d in range(3):
    v=domain_table[domain_table.domain==d].set_index('method')
    if len(v)==2:domain_ok&=bool(v.loc['dino_candidate','mADM']>=v.loc['baseline','mADM']-.01)
single_ok=(single_b is None or single_c>=single_b-.007)
DINO_AUDIT_ACCEPTED=bool(DINO_FIT_PASS
  and cand_audit['mADM']>=base_audit['mADM']+.005
  and cand_audit['mAP']>=base_audit['mAP']-.005
  and cand_audit['Rank-1']>=base_audit['Rank-1']-.005
  and domain_ok and single_ok)
DINO_AUDIT_REPORT={'method':'DINOv2-S Track1 frozen attribute-token transfer to Track2 ABPR',
  'baseline_fit':base_fit,'selected_fit':DINO_WIN,'fit_pass':DINO_FIT_PASS,
  'audit_baseline':base_audit,'audit_candidate':cand_audit,
  'audit_gain_mADM':cand_audit['mADM']-base_audit['mADM'],
  'single_positive_rank1_baseline':single_b,'single_positive_rank1_candidate':single_c,
  'domain_ok':bool(domain_ok),'single_positive_ok':bool(single_ok),
  'audit_accepted':bool(DINO_AUDIT_ACCEPTED),
  'audit_repeatedly_inspected':True,'hidden_codabench_score':None,
  'note':'PUBLIC VAL ONLY. No private labels or query IDs used for scoring.'}
(DINO_OUTPUT/'dino_audit_report.json').write_text(json.dumps(DINO_AUDIT_REPORT,indent=2))
print('AUDIT baseline:',base_audit)
print('AUDIT DINO:',cand_audit)
print('Single-positive R1:',single_b,'->',single_c)
print(domain_table.to_string(index=False))
print('DINO_AUDIT_ACCEPTED:',DINO_AUDIT_ACCEPTED)


#@title 45) Export DINO transfer experiment results ZIP (NOT a Codabench submission)
import zipfile
DINO_REPORT_ZIP=DINO_OUTPUT/'ABPR_DINO_Transfer_Experiment_Report.zip'
items=[DINO_OUTPUT/n for n in ('run_signature.json','train_history.json','dino_fit_trials.csv',
    'dino_fit_selection.json','dino_audit_report.json','dino_audit_domains.csv')]
items += [DINO_MODULE,DINO_MODEL_FILE]
README=DINO_OUTPUT/'README.txt'
README.write_text('DINOv2-S Track1 -> Track2 ABPR. Frozen pretrained attribute-token features; train head on TRAIN only.\n'
 'FIT selected candidate, AUDIT exploratory (previously inspected). ZIP here is NOT Codabench submission.\n'
 'Check external pretrained eligibility and validate fully offline runtime before using a Codabench submission.\n')
items.append(README)
with zipfile.ZipFile(DINO_REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in tqdm(items,desc='Export DINO experiment report',unit='file'):
        assert p.is_file(),p
        z.write(p,p.name)
print('[REPORT]',DINO_REPORT_ZIP)
print('[DINO AUDIT PASS]',DINO_AUDIT_ACCEPTED)
print('DO NOT upload experiment report as Codabench submission.')


#@title 46) OPTIONAL — export end-to-end OFFLINE DINO+baseline runtime (only if gates pass)
# This prepares a candidate, NOT an automatic Codabench submission.
EXPORT_DINO_SUBMISSION=False #@param {type:'boolean'}
EXTERNAL_PRETRAINED_ELIGIBLE=False #@param {type:'boolean'}
if not EXPORT_DINO_SUBMISSION:
    print('[SKIP] No candidate export requested. Submit existing baseline, if needed.')
else:
    import shutil,tempfile,stat,zipfile
    if not DINO_AUDIT_ACCEPTED:raise RuntimeError('DINO AUDIT gate failed; refusing ZIP creation')
    if not EXTERNAL_PRETRAINED_ELIGIBLE:raise RuntimeError('Verify challenge allows Track1 pretraining before setting True')
    ORIGINAL_ZIP=Path(BEST_SUBMISSION_ZIP)
    if not ORIGINAL_ZIP.is_file():raise FileNotFoundError('Cannot locate Track2 baseline submission ZIP; set BEST_SUBMISSION_ZIP explicitly')
    wrapper_code=Path('/content/abpr_dino_runtime_wrapper.py')
    if not wrapper_code.is_file():
        wrapper_code.write_bytes(zlib.decompress(base64.b64decode('eNqVV11v2zYUffev4PJCqtXUJiv6YEPA0jZtAwxNkAV7mGEIlETbXChSIalExrD/vntJSbaTtMPyIIvk4bnf9yonJye3lld3Z+SjqXkpdLUlttNeNmJOaqHEhntBRM8rT7bSeWNlxRUpuRNKakH8Vmiy7pwgny6/XZHAdUpcZaxw2excKfIo5GbrHeG6JpWpBWk650kpSAtQvhE1Met14JIa6cifl9cLog3hVSWcI96QtiuVrIgC/RSwnpyczGTTGuuJcanbufQvZ/RsbU0DpH6rZEmG82tYjljdNe2OcEd0m4Id1TbeuL78bURfNqBP3A2AB+mk0eOpt1y7tbGNGxnjD4jLOi/V7Obq6jZHiawo1lKJokgycINRD4IlWcut0H4m18R5yxCbgJUerQYTMlR8Pr5kUjthPXubTthkFMrL1ha11KaICgmLNtU+6n10UDTgbzXq/wmOzr23suy8uD6/mRUfzn+/yL8ZLRbF7cX5x68XN8Pq68X5p+H149W3z5dfwmI2q8WaKMPrSMyS+YzA30aZEnIi0KUjUxpI0uF+wIHpAUOkC5Yj59wK31kd3gPItaLKjx2b4V6B5kW3KlNxD4FhNFgLKbkpxoSkKXrrDcVNqbkqIJmzdkeTQB4tfsIOtnTAGvhREsNHssBYxCO3xJ1M80as8kCxCBvoCWEz0YN2EcnC6YGs7NBbcXtwKKZsOHWMRZWPQ1cZvZabDGEU0whovOg9SyJNdZeHFA0URwSCV1vQqvU0bXg7OSunVdvRdKjGwmi1yz9z5cTAt97ktaw8q+6WFBZ0lSzgJ2tNy2gLQbIcvFvTFAMVjpY07PESI8Kdp6v8baCqxYOsBMjrak4x6FFRXGbSFfyBS4W3WEKgnAUJikXfjFn4NFXZXoOodPrqFaiQHN2KvnYeGlYxmRKWdIV1BFv5re3E8a18uu4Ni6onmXjgEC/w+n0noYKLjQVmduAuTOUYyCVthN+aGszPKfbAwps7oYstRIzG+gjiQknVPpvMukXYV0CBoL3o6UIpnP9eiCcBBaL+b6gndZ45DNl+4LK9HfH6f3rpsLZnv0ZjpIbshjEjQlVAUWBPgaS/KzZcKWF3rOU7VGxoLs8KqIwlmL14JyKgF+S6zbjj1vIdK5fgN7AKhDqwC07WAPa/nEU4h4i4XAFipFlSPkapwKp3WA33h5QT8L4TVr5IO6iWT9Bhg67GDEI9M7flrfgpZ0podp+k+DMAk2QOKQ8F8gdXnbiw1lhGr4a+duQyEkhII13DfbUdmt3UAoMJ+ZSvPszoYjIRlF8QWff58vgGhKoWPcP3hMDcI/iG4yo4LFrh1w0mddlJVRfTgGTJYmgCGlvWVGEwAoHCC+ugjWUREp2Bg9ehg0XT+h1jh25I371Nnnm3REPzhvfsNJUaZEzVKCtrwikELWJRdwi/DbMWdNwIGKxHEgI+2VcrKOrz4XQZrs7D83UAriZc7/LlfoVypBcNikGGPR/+PUq/jV8YmWlBOCKXNFhe4MwHdXGQy2beu4y3AKkZuJfJJoNZ8ICfA/TmyweaJPtq7PuhRYB21R3r3YutJEiOON55aBKQ6bE7p7XftWLgCP49fZ+K0NPrPNJkATF08+TYpLXgfuqfrO/TWPAF7/onbeNpz/SeF/DR19LVMePIenZImyGS/Xya/Jg/douN9C5nyLGkcUVX0TSWvA7Uz/eTN2fPmEKTnZjw2VnxAtfzk6dsOOTmg2Iv67UYhH1P1uzQhdh6j76gBupwwCJTGvcOLoYCO0plLADM0mSVR/QoLYPpAc/wucyS5TyF3hCTXItHrHbsebtCK8Xu00icVttO3+Xv3w0FB/8P1IiEF/xOw0aXwuU0SpiygKt2y6cyHabF3wfNeh6YIKdDd07+mf0L5SIvXw==')))
    out=DINO_OUTPUT/'UPAR2027_Track2_DINOTransfer_Candidate.zip'
    import torch
    with tempfile.TemporaryDirectory(prefix='dino_abpr_pack_') as wd:
        wd=Path(wd);sub=wd/'submission';sub.mkdir()
        with zipfile.ZipFile(ORIGINAL_ZIP) as z:
            for e in z.infolist():
                if e.is_dir():continue
                target=(sub/e.filename).resolve()
                if not target.is_relative_to(sub.resolve()) or stat.S_ISLNK(e.external_attr>>16):
                    raise RuntimeError('Unsafe input ZIP entry: '+e.filename)
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(e) as src,target.open('wb') as dest:shutil.copyfileobj(src,dest)
        orig=sub/'run.py'
        if not orig.is_file():raise RuntimeError('Baseline ZIP missing run.py')
        orig.rename(sub/'original_run.py')
        shutil.copy2(wrapper_code,sub/'run.py')
        shutil.copy2(DINO_MODULE,sub/'abpr_dino_transfer.py')
        shutil.copy2(DINO_MODEL_FILE,sub/'dino_transfer_model.py')
        torch.save({'state':DINO_CK_META['inference_model_state'],
                    'cfg':DINO_CK_META.get('dinov2_config',{})},sub/'dino_teacher.pt')
        shutil.copy2(DINO_HEAD_PATH.with_name('dino_token_head_best.pt'),sub/'dino_token_head_best.pt')
        runtime_cfg={'method':DINO_WIN['method'],'alpha':float(DINO_WIN['alpha']),
                     'track1_attributes':TRACK1_ATTR,'track2_attributes':ATTR,
                     'tta_flip':DINO_TTA_FLIP,'microbatch':max(1,min(8,DINO_BATCH))}
        (sub/'dino_transfer_config.json').write_text(json.dumps(runtime_cfg,indent=2))
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for p in tqdm(sorted(sub.rglob('*')),desc='Package offline DINO candidate',unit='file'):
                if p.is_file():z.write(p,p.relative_to(sub).as_posix())
        print('[CANDIDATE]',out,'size MB:',round(out.stat().st_size/2**20,1))
        # Smoke test in newly extracted directory with real 2 image query payload
        sys.path.insert(0,str(sub))
        import importlib.util
        spec=importlib.util.spec_from_file_location('dino_candidate_probe',sub/'run.py')
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        payload={'gallery':[{'image_path':VAL_PATHS[0]},{'image_path':VAL_PATHS[1]}],
                 'queries':Q[SELECT_IDS[:2]].tolist(),'attribute_names':ATTR}
        with torch.inference_mode():
            mod.load_model();got=mod.rank_gallery(payload)
        scores=np.asarray(got['distances'],np.float32)
        assert scores.shape==(2,2) and np.isfinite(scores).all()
        print('[LOCAL SMOKE PASS] Shape',scores.shape)
        sys.modules.pop(spec.name,None);sys.path.remove(str(sub))
    print('[WARNING] Local smoke test does not prove Codabench GPU, time, dependency or official rules compliance.')
