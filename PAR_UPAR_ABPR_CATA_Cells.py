#@title 54) CATA — Restore exact HATA/Baseline and install new TRAIN-only method
from pathlib import Path
import sys,os,json,hashlib,base64,zlib
import numpy as np,pandas as pd,torch
from tqdm.auto import tqdm

# This follows the completed DINO/HATA Cells 39–52 of this notebook.
# If Colab runtime was reset, rerun Cells 39–40 and 47–52 first (cached features load without recomputing).
required=['dt','DINO_OUTPUT','DINO_TO_TRACK2','DINO_RUN_SIGNATURE','DINO_CK_SHA',
          'DINO_CHECKPOINT','DINO_MODEL_FILE','DINO_RESOLVER','DINO_VAL_TOKENS','DINO_VAL_LOGITS',
          'VAL_NAMES','VAL_PATHS','ATTR','TRACK1_ATTR','BASE_DIST','FIT_IDS','AUDIT_IDS',
          'SELECT_IDS','Q','GIDS','VAL_LABELS','dino_metrics','BEST_SUBMISSION_ZIP','HATA_RUN_DIR']
missing=[x for x in required if x not in globals()]
if missing:raise RuntimeError('Missing notebook state '+str(missing)+'; rerun the cached setup Cells 39–40 and HATA Cells 47–52 before Cell 54.')
import importlib
installed=Path('/content/abpr_cross_token.py')
installed.write_bytes(zlib.decompress(base64.b64decode('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')))
sys.path.insert(0,'/content');import abpr_cross_token as cata
cata=importlib.reload(cata)
CATA_ROOT=Path(DINO_OUTPUT)/'cross_attribute_token_attention';CATA_ROOT.mkdir(parents=True,exist_ok=True)
FIT_IDS=np.asarray(FIT_IDS,dtype=np.int64);AUDIT_IDS=np.asarray(AUDIT_IDS,dtype=np.int64)
SELECT_IDS=np.asarray(SELECT_IDS,dtype=np.int64)
assert np.array_equal(SELECT_IDS,np.r_[FIT_IDS,AUDIT_IDS]);N_FIT=len(FIT_IDS)
assert len(ATTR)==40 and np.shape(DINO_VAL_TOKENS)==(len(VAL_NAMES),40,192)
assert np.shape(DINO_VAL_LOGITS)==(len(VAL_NAMES),40)
assert np.shape(BASE_DIST)==(len(SELECT_IDS),len(VAL_NAMES))
assert np.isfinite(np.asarray(BASE_DIST)).all()
# Verify entire gallery feature cache corresponds to exact image order, teacher and transform.
vcache=Path(DINO_OUTPUT)/'val_features'/'meta.json'
meta=json.loads(vcache.read_text())
expected=dt.digest_lines([DINO_CK_SHA,dt.digest_lines(TRACK1_ATTR),dt.digest_lines(VAL_PATHS),'flipTrue'])
assert meta['signature']==expected and meta['completed']==len(VAL_NAMES) and meta['flipped'] is True, 'DINO cache signature/order mismatch — stop'
manifest_path=Path(DINO_ROOT)/'ABPR_Results'/'siglip2_retrieval'/'fixed_public_val_split.json'
manifest=json.loads(manifest_path.read_text())
assert dt.digest_lines(VAL_NAMES)==manifest['gallery_paths_sha256'], 'Gallery order differs from baseline cache'
base_meta=Path(DINO_ROOT)/'ABPR_Results'/'siglip2_retrieval'/'baseline_fixed_queries.json'
base_info=json.loads(base_meta.read_text())
expected_base=dt.digest_lines([dt.hash_file(BEST_SUBMISSION_ZIP),manifest['gallery_paths_sha256'],*SELECT_IDS])
assert base_info['signature']==expected_base, 'Baseline ZIP changed! Re-run Cell 9 before comparing.'
print('[PASS] Frozen DINO cache:',DINO_VAL_TOKENS.shape, 'public gallery:',len(VAL_NAMES))
print('[PASS] EXACT cached baseline ZIP hash:',dt.hash_file(BEST_SUBMISSION_ZIP)[:20])
print('[CAUTION] Hidden Codabench mADM 0.549873 is NOT the same as public VAL mADM.')
print('[CAUTION] Existing AUDIT was inspected repeatedly: exploratory, NOT pristine blind test.')


#@title 55) CATA — Expand TRAIN-only DINO attribute-token cache (resumable)
# Increase to 8000 only if Colab disk and time allow; 5000/domain = 15000 TRAIN images.
CATA_TRAIN_PER_DOMAIN=5000 #@param {type:'integer'}
CATA_DINO_BATCH=16 #@param {type:'integer'}
CATA_SEED=741 #@param {type:'integer'}
assert 1500<=CATA_TRAIN_PER_DOMAIN<=8000
from abpr.core import infer_domain_id,detect_image_column
import shutil,gc
trainframe=pd.read_csv(OFFICIAL_ROOT/'data/annotations/task2/train/gt.csv')
traincol=detect_image_column(trainframe)
assert set(TRACK1_ATTR).issubset(trainframe.columns)
train_domains=np.asarray([infer_domain_id(x) for x in trainframe[traincol].astype(str)],np.int64)
assert set(np.unique(train_domains))=={0,1,2}
rng=np.random.default_rng(CATA_SEED)
selected=[]
for d in range(3):
    ids=np.flatnonzero(train_domains==d);rng.shuffle(ids)
    n=min(len(ids),CATA_TRAIN_PER_DOMAIN)
    selected.extend(ids[:n].tolist())
CATA_FRAME=trainframe.iloc[selected].reset_index(drop=True)
CATA_LABELS=CATA_FRAME[TRACK1_ATTR].to_numpy(np.float32)
CATA_DOMAINS=train_domains[np.asarray(selected)]
print('[TRAIN ONLY] total:',len(CATA_FRAME),'domain counts:',np.bincount(CATA_DOMAINS,minlength=3).tolist())
train_names=CATA_FRAME[traincol].astype(str).tolist()
# Refuse data overlap with public VAL, independent of labels.
assert not (set(train_names)&set(VAL_NAMES)), 'Data leakage: TRAIN overlaps public VAL'
CATA_TRAIN_PATHS=[str(DINO_RESOLVER.resolve(x)) for x in tqdm(train_names,desc='Resolve REAL TRAIN image paths',unit='img')]
assert len(set(CATA_TRAIN_PATHS))==len(CATA_TRAIN_PATHS), 'Repeated TRAIN image paths'
if shutil.disk_usage('/content').free < 2*2**30:raise RuntimeError('Free at least 2 GiB local disk before DINO feature caching')
CATA_DATA_SIGNATURE=dt.digest_lines([DINO_CK_SHA,dt.digest_lines(TRACK1_ATTR),
                                       dt.digest_lines(CATA_TRAIN_PATHS),'flipTrue'])
CATA_EXPT=CATA_ROOT/f'train{len(CATA_FRAME)}_seed{CATA_SEED}'
CATA_EXPT.mkdir(parents=True,exist_ok=True)
# Resume from its exact filenames/signature; no VAL labels involved.
DINO_TEACHER,DINO_META=dt.load_teacher(DINO_CHECKPOINT,DINO_MODEL_FILE,'cuda' if torch.cuda.is_available() else 'cpu')
CATA_TOKENS,CATA_LOGITS=dt.cache_teacher(DINO_TEACHER,CATA_TRAIN_PATHS,
     CATA_EXPT/'train_features',CATA_DATA_SIGNATURE,batch_size=CATA_DINO_BATCH,
     flip=True,device='cuda' if torch.cuda.is_available() else 'cpu')
del DINO_TEACHER;gc.collect()
if torch.cuda.is_available():torch.cuda.empty_cache()
assert np.shape(CATA_TOKENS)==(len(CATA_LABELS),40,192)
print('[PASS] TRAIN frozen tokens:',CATA_TOKENS.shape,'cache:',CATA_EXPT)


#@title 56) CATA — Train cross-token attention and hard-negative representation (resume)
CATA_EPOCHS=9 #@param {type:'integer'}
CATA_BATCH=96 #@param {type:'integer'}
CATA_LR=0.00045 #@param {type:'number'}
CATA_STEPS_PER_EPOCH=0 #@param {type:'integer'}
assert len(CATA_LABELS)==len(CATA_TOKENS)
CATA_RUN_SIG=cata.digest(CATA_DATA_SIGNATURE,dt.digest_lines(TRACK1_ATTR),len(CATA_LABELS),
                         f'epochs={CATA_EPOCHS}|bs={CATA_BATCH}|lr={CATA_LR}|steps={CATA_STEPS_PER_EPOCH}')
CATA_RUN=CATA_EXPT/f'head_e{CATA_EPOCHS}_b{CATA_BATCH}_lr{CATA_LR}'
CATA_RUN.mkdir(parents=True,exist_ok=True)
CATA_MODEL,CATA_HISTORY,CATA_BEST=cata.train(CATA_TOKENS,CATA_LOGITS,CATA_LABELS,CATA_DOMAINS,
          CATA_RUN/'cross_token_last.pt',CATA_RUN_SIG,epochs=CATA_EPOCHS,batch=CATA_BATCH,
          lr=CATA_LR,steps_per_epoch=CATA_STEPS_PER_EPOCH,seed=CATA_SEED)
(CATA_RUN/'train_history.json').write_text(json.dumps(CATA_HISTORY,indent=2))
print('[PASS] New representation best TRAIN-holdout checkpoint:',CATA_BEST)
print('[IMPORTANT] Reduced TRAIN-holdout BCE does not guarantee better hidden ABPR retrieval.')


#@title 57) CATA — Score SAME public gallery; reconstruct HATA reference exactly
CATA_CK_SHA=dt.hash_file(CATA_BEST)
CATA_VAL_SIG=cata.digest(CATA_CK_SHA,meta['signature'],dt.digest_lines(VAL_PATHS))
CATA_VAL_LOGITS=cata.infer(CATA_MODEL,DINO_VAL_TOKENS,DINO_VAL_LOGITS,
                           CATA_RUN/'cata_val_logits.npy',CATA_VAL_SIG,batch=256)
CATA_VAL_TRACK2=np.asarray(CATA_VAL_LOGITS[:,DINO_TO_TRACK2],np.float32)
CATA_DISTS=cata.query_distance(Q[SELECT_IDS],CATA_VAL_TRACK2,chunk=48)
assert CATA_DISTS.shape==BASE_DIST.shape
# HATA is the strongest prior reference verified in the uploaded notebook.
import abpr_hata_method as hata
hata_meta=Path(HATA_RUN_DIR)/'hata_audit_report.json'
hata_rep=json.loads(hata_meta.read_text())
hata_win=hata_rep['fit_selected'];hata_method=hata_win['method'];hata_alpha=float(hata_win['alpha'])
if hata_method != 'HATA_unweighted':
    raise RuntimeError('Stored HATA reference is different from the audited unweighted method; inspect HATA report first')
hata_logits_file=Path(HATA_RUN_DIR)/'val_hata_logits.npy'
HATA_LOG=np.load(hata_logits_file,mmap_mode='r',allow_pickle=False)
assert HATA_LOG.shape==(len(VAL_NAMES),40)
HATA_DISTS=hata.weighted_query_nll(Q[SELECT_IDS],np.asarray(HATA_LOG[:,DINO_TO_TRACK2],np.float32),chunk=48)
HATA_REFERENCE=cata.fuse(BASE_DIST,HATA_DISTS,hata_alpha)
HATA_REF_AUD=dino_metrics(HATA_REFERENCE[N_FIT:],AUDIT_IDS)
assert abs(HATA_REF_AUD['mADM']-hata_rep['audit_candidate']['mADM']) < 1e-4, (
    'Reconstructed HATA AUDIT differs from the original log! Check cache signature/alignment.')
print('[PASS] CATA matrix:',CATA_DISTS.shape)
print('[PASS] HATA reference exactly reconstructed:',HATA_REF_AUD,'alpha',hata_alpha)


#@title 58) CATA — FIT-only method/weight selection (protect HATA and single-positives)
from tqdm.auto import tqdm
FBASE=np.asarray(BASE_DIST[:N_FIT],np.float32)
FHATA=HATA_REFERENCE[:N_FIT]
FCATA=CATA_DISTS[:N_FIT]
base_m=dino_metrics(FBASE,FIT_IDS);ref_m=dino_metrics(FHATA,FIT_IDS)
positive_counts=np.bincount(GIDS,minlength=len(Q))[FIT_IDS]
single=positive_counts==1
old_r1=GIDS[np.argmin(FHATA,axis=1)]==FIT_IDS
CATA_ALPHA_GRID=(.02,.035,.05,.075,.10,.15,.20)
# Fusions normalize row-wise; all scoring uses only model predictions, not VAL labels.
combos=[]
for alpha in CATA_ALPHA_GRID:
    combos.append(('baseline_plus_CATA',alpha,cata.fuse(FBASE,FCATA,alpha)))
    combos.append(('HATA_plus_CATA',alpha,cata.fuse(FHATA,FCATA,alpha)))
    combos.append(('baseline_HATA_CATA',alpha,cata.fuse(FBASE,.5*FHATA+.5*FCATA,alpha)))
rows=[]
for name,alpha,scores in tqdm(combos,desc='FIT 21 CATA configurations',unit='config'):
    mm=dino_metrics(scores,FIT_IDS)
    top=GIDS[np.argmin(scores,axis=1)]==FIT_IDS
    single_delta=float(top[single].mean()-old_r1[single].mean()) if single.any() else 0
    # Gating is relative to stronger HATA, not weaker baseline.
    accepted=bool(mm['mADM']>=ref_m['mADM']+.003 and
                  mm['mAP']>=ref_m['mAP']-.003 and
                  mm['Rank-1']>=ref_m['Rank-1']-.005 and single_delta>=-.007)
    rows.append({'method':name,'alpha':float(alpha),**mm,'single_R1_delta_vs_HATA':single_delta,
                 'delta_mADM_vs_HATA':mm['mADM']-ref_m['mADM'], 'fit_pass':accepted})
CATA_FIT_TABLE=pd.DataFrame(rows).sort_values(['mADM','mAP'],ascending=False)
CATA_FIT_TABLE.to_csv(CATA_RUN/'fit_trials.csv',index=False)
eligible=CATA_FIT_TABLE[CATA_FIT_TABLE.fit_pass]
CATA_FIT_PASS=bool(len(eligible))
CATA_WIN=eligible.iloc[0].to_dict() if CATA_FIT_PASS else {'method':'HATA_reference','alpha':0.0,'fit_pass':False}
print('BASELINE FIT:',base_m,'HATA FIT:',ref_m)
print(CATA_FIT_TABLE.head(10).to_string(index=False))
print('[FIT GATE]',CATA_FIT_PASS,'WINNER',CATA_WIN)


#@title 59) CATA — Public AUDIT: compare baseline/HATA/new, domains + single-positive
from abpr.core import infer_domain_id
from abpr.query_aware_ranking import retrieval_metrics
from abpr.model import track2_metrics
BA=np.asarray(BASE_DIST[N_FIT:],np.float32)
HA=np.asarray(HATA_REFERENCE[N_FIT:],np.float32)
if CATA_FIT_PASS:
    method=CATA_WIN['method'];a=float(CATA_WIN['alpha']);CA0=CATA_DISTS[N_FIT:]
    if method=='baseline_plus_CATA': CATA_AUD=cata.fuse(BA,CA0,a)
    elif method=='HATA_plus_CATA': CATA_AUD=cata.fuse(HA,CA0,a)
    elif method=='baseline_HATA_CATA': CATA_AUD=cata.fuse(BA,.5*HA+.5*CA0,a)
    else:raise ValueError(method)
else:
    CATA_AUD=HA.copy() # fallback is HATA, NOT a test of rejected CATA candidate
b=dino_metrics(BA,AUDIT_IDS);h=dino_metrics(HA,AUDIT_IDS);n=dino_metrics(CATA_AUD,AUDIT_IDS)
print('[AUDIT BASELINE]',b,'[AUDIT HATA]',h,'[AUDIT CATA-selected]',n)
counts=np.bincount(GIDS,minlength=len(Q))[AUDIT_IDS];single=counts==1
correct_h=GIDS[np.argmin(HA,axis=1)]==AUDIT_IDS
correct_n=GIDS[np.argmin(CATA_AUD,axis=1)]==AUDIT_IDS
single_h=float(correct_h[single].mean()) if single.any() else None
single_n=float(correct_n[single].mean()) if single.any() else None
vd=np.asarray([infer_domain_id(x) for x in VAL_NAMES])
rows=[];dom_ok=True
for d in tqdm((0,1,2),desc='AUDIT domain checks',unit='domain'):
    mask=vd==d
    ids=AUDIT_IDS[np.isin(AUDIT_IDS,np.unique(GIDS[mask]))]
    index={int(v):i for i,v in enumerate(AUDIT_IDS)}
    take=[index[int(v)] for v in ids]
    for method,arr in [('baseline',BA),('HATA',HA),('CATA',CATA_AUD)]:
        sc=arr[take][:,mask]
        ret=retrieval_metrics(sc,ids,GIDS[mask]);ret.update(track2_metrics(sc,Q[ids],VAL_LABELS[mask]))
        rows.append({'domain':int(d),'model':method,'queries':len(ids),
                     **{k:float(ret[k]) for k in ('mADM','mAP','Rank-1')}})
    xx=pd.DataFrame(rows);sub=xx[xx.domain==d].set_index('model')
    dom_ok &= bool(sub.loc['CATA','mADM']>=sub.loc['HATA','mADM']-.01)
CATA_DOMAIN_TABLE=pd.DataFrame(rows)
CATA_DOMAIN_TABLE.to_csv(CATA_RUN/'audit_domains.csv',index=False)
CATA_AUDIT_ACCEPTED=bool(CATA_FIT_PASS and n['mADM']>=h['mADM']+.005 and
                         n['mAP']>=h['mAP']-.003 and n['Rank-1']>=h['Rank-1']-.005 and
                         (single_h is None or single_n>=single_h-.007) and dom_ok)
CATA_REPORT={'method':'CATA cross-attribute DINO token attention with TRAIN-only hard-negative ranking',
  'hidden_current_mADM_from_user':.549873,'hidden_expected_or_guaranteed':False,
  'base_audit':b,'HATA_audit':h,'candidate_audit':n,'fit_win':CATA_WIN,
  'fit_pass':bool(CATA_FIT_PASS),'audit_accepted':bool(CATA_AUDIT_ACCEPTED),
  'audit_domains':CATA_DOMAIN_TABLE.to_dict(orient='records'),
  'single_positive_Rank1_HATA':single_h,'single_positive_Rank1_new':single_n,
  'train_count':len(CATA_LABELS),'train_checksum':CATA_DATA_SIGNATURE,'checkpoint':str(CATA_BEST),
  'warning':'AUDIT previously inspected repeatedly; exploratory. External DINO eligibility unconfirmed.'}
(CATA_RUN/'cata_audit_report.json').write_text(json.dumps(CATA_REPORT,indent=2))
print(CATA_DOMAIN_TABLE.to_string(index=False))
print('[CATA AUDIT ACCEPTED]',CATA_AUDIT_ACCEPTED)
print('[NOTE] If CATA FIT gate is false, CATA column represents fallback HATA; NOT a new method result.')


#@title 60) CATA — Reproducibility artifact + submission decision (NO automatic upload)
import zipfile,hashlib
README='CATA / UPAR 2027 Track 2 exploratory continuation.\nTRAIN-only: DINO frozen 40x192 tokens, cross-attribute attention, domain-balanced sampling, hard-negative ranking.\nVAL labels only used in metrics. FIT chooses fusion. AUDIT was already inspected in prior experiments.\nDo not upload this experiment ZIP as Codabench submission. It is not a run.py runtime.\nFor an actual submission, deploy full offline DINO+cross-token checkpoint inference, verify output [Q,G], memory, integrity, licences and organizer permission.\nThere is no confirmed hidden score >0.65 from this experiment.\n'
(CATA_RUN/'README.txt').write_text(README)
zip_path=CATA_RUN/'ABPR_CATA_Experiment_Report.zip'
items=[CATA_RUN/'README.txt',CATA_RUN/'cata_audit_report.json',CATA_RUN/'fit_trials.csv',
       CATA_RUN/'audit_domains.csv',CATA_RUN/'train_history.json',CATA_BEST,
       Path('/content/abpr_cross_token.py')]
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zp:
    for p in tqdm(items,desc='Package CATA evidence',unit='file'):
        if not Path(p).is_file():raise FileNotFoundError(p)
        zp.write(p,arcname=Path(p).name)
print('[REPORT]',zip_path)
print('[BASELINE hidden Codabench mADM]',.549873)
if CATA_AUDIT_ACCEPTED:
    print('[RESULT] CATA outperforms HATA on public exploratory AUDIT. NEXT: eligibility + offline runtime parity; NOT ready to upload.')
else:
    print('[RESULT] Keep best prior submission/HATA; CATA has not passed FIT/AUDIT requirements. DO NOT submit it.')
print('[DISCLAIMER] No hidden >0.65–0.68 result verified. ZIP above is a research report, not Codabench submission.')
