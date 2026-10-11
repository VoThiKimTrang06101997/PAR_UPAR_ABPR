

# =======================================================
#@title 27) Rank-Local Method — restore exact artifacts + verify predicted-attribute cache
from pathlib import Path
import os,sys,json,hashlib,time,importlib,base64,zlib
import numpy as np,pandas as pd
from tqdm.auto import tqdm

SOURCE_ROOT=Path(globals().get('SOURCE_ROOT','/content/PAR_UPAR_ABPR'))
OFFICIAL_ROOT=Path(globals().get('OFFICIAL_ROOT','/content/UPAR-Challenge-2027'))
DRIVE_ROOT=Path(globals().get('DRIVE_ROOT','/content/drive/MyDrive/PedestrianAttributeRecognition'))
RESULT_DIR=Path(globals().get('RESULT_DIR',DRIVE_ROOT/'ABPR_Results'))
SIGLIP_DIR=Path(globals().get('SIGLIP_DIR',RESULT_DIR/'siglip2_retrieval'))
METHOD_DIR=Path(globals().get('METHOD_DIR',SIGLIP_DIR/'error_guided_method'))
RLOCAL_DIR=SIGLIP_DIR/'rank_local_method';RLOCAL_DIR.mkdir(parents=True,exist_ok=True)
SIGLIP_MODEL_ID='google/siglip2-base-patch16-224'
sys.path.insert(0,str(SOURCE_ROOT))
from abpr.core import ATTRIBUTE_NAMES,detect_image_column,ImageResolver
from abpr.model import load_query_csv

# Install source from this notebook; no git pull or external source ZIP required.
source=zlib.decompress(base64.b64decode('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')).decode('utf-8')
module_path=Path('/content/abpr_rank_local.py')
module_path.write_text(source,encoding='utf-8')
sys.path.insert(0,str(module_path.parent))
import abpr_rank_local as rl
importlib.reload(rl)

if 'sp' not in globals():
    for folder in [Path('/content'),SOURCE_ROOT]:
        if (folder/'abpr_siglip2_pipeline.py').is_file():
            sys.path.insert(0,str(folder));import abpr_siglip2_pipeline as sp;break
if 'sp' not in globals():
    raise FileNotFoundError('Missing abpr_siglip2_pipeline.py; run Cell 7 (source install).')
if 'eg' not in globals():
    for folder in [Path('/content'),SOURCE_ROOT]:
        if (folder/'abpr_error_guided_rerank.py').is_file():
            sys.path.insert(0,str(folder));import abpr_error_guided_rerank as eg;break
if 'eg' not in globals():
    raise FileNotFoundError('Missing abpr_error_guided_rerank.py; run Cell 20A.')

# Strictly use original split; never generate new FIT/AUDIT IDs here.
manifest_path=SIGLIP_DIR/'fixed_public_val_split.json'
assert manifest_path.is_file(),f'Run original Cell 8 first: {manifest_path}'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
FIT_IDS=np.asarray(manifest['fit_ids'],np.int64)
AUDIT_IDS=np.asarray(manifest['audit_ids'],np.int64)
SELECT_IDS=np.r_[FIT_IDS,AUDIT_IDS]
assert len(SELECT_IDS)==len(set(SELECT_IDS.tolist()))
ATTR=list(ATTRIBUTE_NAMES);assert len(ATTR)==40
VAL_CSV=OFFICIAL_ROOT/'data/annotations/task2/val/gt.csv'
assert VAL_CSV.is_file(),VAL_CSV
VAL_FRAME=pd.read_csv(VAL_CSV)
VAL_COL=detect_image_column(VAL_FRAME)
VAL_NAMES=VAL_FRAME[VAL_COL].astype(str).tolist()
VAL_LABELS=VAL_FRAME[ATTR].to_numpy(np.float32)
GIDS=np.asarray(sp.ids_from_official(OFFICIAL_ROOT/'data/annotations/task2/val/ids.csv',VAL_NAMES),dtype=np.int64)
_,Q=load_query_csv(OFFICIAL_ROOT/'data/annotations/task2/val/queries.csv')
Q=np.asarray(Q,np.float32)
assert Q.ndim==2 and Q.shape[1]==40 and int(SELECT_IDS.max())<len(Q)
assert len(VAL_NAMES)==len(GIDS)==len(VAL_LABELS)==manifest['gallery_count']
assert len(Q)==manifest['query_count']
assert sp.digest_lines(VAL_NAMES)==manifest['gallery_paths_sha256'],'Gallery order changed; STOP'

# Baseline is the exact cached original submission's distance matrix.
base_file=SIGLIP_DIR/'baseline_fixed_queries.npy'
assert base_file.is_file(),f'Run Cell 9: missing {base_file}'
BASE_DIST=np.load(base_file,mmap_mode='r',allow_pickle=False)
expected=(len(SELECT_IDS),len(VAL_NAMES))
assert BASE_DIST.shape==expected, (BASE_DIST.shape,expected)
assert np.isfinite(BASE_DIST).all(),'NaN/Inf in baseline'

# Predicted-attribute penalties (not gallery GT labels) from Cell 22.
pen_file=METHOD_DIR/'predicted_contradiction_penalty.npy'
gate_file=METHOD_DIR/'train_only_gate.json'
adapter_file=SIGLIP_DIR/'train_only_adapter.npz'
sig_file=SIGLIP_DIR/'val_siglip2.npy'
for label,p in [('penalty Cell 22',pen_file),('TRAIN-only gate',gate_file),('adapter Cell 11',adapter_file),('SigLIP cache Cell 10',sig_file)]:
    if not p.is_file():raise FileNotFoundError(f'Missing {label}: {p}. Run prerequisite cells.')
CONTRADICTION_ALL=np.load(pen_file,mmap_mode='r',allow_pickle=False)
assert CONTRADICTION_ALL.shape==expected and np.isfinite(CONTRADICTION_ALL).all()
GATE_META=json.loads(gate_file.read_text())
assert len(GATE_META['hi'])==40 and len(GATE_META['lo'])==40

# Verify SigLIP feature cache was created against EXACT SAME paths and pretrained model.
meta_file=sig_file.with_suffix('.json')
assert meta_file.is_file(),f'SigLIP feature manifest missing: {meta_file}'
meta=json.loads(meta_file.read_text())
VAL_PATHS=globals().get('VAL_PATHS')
if VAL_PATHS is None or len(VAL_PATHS)!=len(VAL_NAMES):
    DATA_ROOT=OFFICIAL_ROOT/'data'
    root_guesses=[DATA_ROOT,OFFICIAL_ROOT,DRIVE_ROOT/'data',DRIVE_ROOT/'UPAR_Data',
                  Path('/content/data'),Path('/content/datasets'),Path('/content/PAR_UPAR/data')]
    root_guesses += [Path(x) for x in globals().get('EXTRA_DATA_ROOTS',[])]
    RESOLVER=ImageResolver(DATA_ROOT,OFFICIAL_ROOT,extra_roots=[r for r in root_guesses if r.is_dir()])
    VAL_PATHS=[str(RESOLVER.resolve(raw)) for raw in tqdm(VAL_NAMES,desc='27 / verify exact VAL image paths',unit='image')]
    if not all(Path(p).is_file() for p in VAL_PATHS):raise FileNotFoundError('Missing real images; rerun Cell 3')
actual_signature=sp.digest_lines([SIGLIP_MODEL_ID,len(VAL_PATHS),*map(str,VAL_PATHS)])
assert meta.get('signature')==actual_signature and int(meta.get('completed',-1))==len(VAL_PATHS), (
    'SigLIP cache not aligned to the exact gallery/order/model. Rerun Cells 10 and 22; never force-match files.')

# Check a few predicted penalty rows against the actual SigLIP image features and trained adapter.
with np.load(adapter_file,allow_pickle=False) as a:
    w=np.asarray(a['weight'],np.float32);b=np.asarray(a['bias'],np.float32)
SIG_VAL=np.load(sig_file,mmap_mode='r',allow_pickle=False)
assert SIG_VAL.shape[0]==len(VAL_NAMES) and w.shape[1]==40 and w.shape[0]==SIG_VAL.shape[1]
weights=np.ones(40,np.float32)
for j,name in enumerate(ATTR):
    if name.startswith(('Accessory-','Hair-Length-','UpperBody-Color-','LowerBody-Color-')):
        weights[j]=1.35
sample_rows=np.unique(np.linspace(0,len(SELECT_IDS)-1,num=min(5,len(SELECT_IDS)),dtype=int))
check=eg.contradiction_penalty(Q[SELECT_IDS[sample_rows]],SIG_VAL,w,b,GATE_META,attr_weights=weights)
difference=float(np.max(np.abs(check-np.asarray(CONTRADICTION_ALL[sample_rows],np.float32))))
assert difference<2e-4, f'Penalty-cache mismatch (max error={difference}); rerun Cell 22'

# Smoke: zero strength reproduces exact baseline, no label/ID input to scorer.
small=np.asarray(BASE_DIST[:3,:25],np.float32)
p_small=np.asarray(CONTRADICTION_ALL[:3,:25],np.float32)
ix=rl.topk_indices(small,15)
assert np.array_equal(rl.rank_local_score(small,p_small,ix,top_k=15,strength=0,tau=1),small)
print('[PASS] Baseline and predicted-attribute feature orders verified')
print('FIT=',len(FIT_IDS),'AUDIT=',len(AUDIT_IDS),'Gallery=',len(VAL_NAMES))
print('Predictive gate usable directions:',sum(GATE_META['usable_hi']),sum(GATE_META['usable_lo']))
print('Zero-strength scorer returns exact baseline. Ready for FIT-only search.')



# =======================================================
#@title 28) FIT-only Rank-Local hyperparameter search — real progress, no retraining
from tqdm.auto import tqdm
import numpy as np,pandas as pd,json,time
from abpr.query_aware_ranking import retrieval_metrics
from abpr.model import track2_metrics

F_N=len(FIT_IDS)
FIT_BASE=np.asarray(BASE_DIST[:F_N],np.float32)
FIT_P=np.asarray(CONTRADICTION_ALL[:F_N],np.float32)
FIT_TOP=rl.topk_indices(FIT_BASE,800)
FIT_BMET=sp.semantic_metrics(FIT_BASE,FIT_IDS,GIDS,Q[FIT_IDS],VAL_LABELS)
FIT_BMET={k:float(FIT_BMET[k]) for k in ('mADM','mAP','Rank-1')}
print('FIT ORIGINAL baseline:',FIT_BMET)
# Small grid: 3 x 2 x 2 = 12 configs; tune exclusively on FIT.
RLOCAL_GRID=[{'top_k':int(k),'strength':float(g),'tau':float(t),'cap':0.65}
             for k in (50,200,800) for g in (0.2,0.5) for t in (0.75,1.5)]
RLOCAL_ROWS=[]
for cfg in tqdm(RLOCAL_GRID,desc='28 / FIT-only local reranking',unit='trial'):
    out=rl.rank_local_score(FIT_BASE,FIT_P,FIT_TOP,**cfg)
    metrics=sp.semantic_metrics(out,FIT_IDS,GIDS,Q[FIT_IDS],VAL_LABELS)
    metrics={k:float(metrics[k]) for k in ('mADM','mAP','Rank-1')}
    RLOCAL_ROWS.append({**cfg,**metrics,'delta_mADM':metrics['mADM']-FIT_BMET['mADM'],
                        'delta_mAP':metrics['mAP']-FIT_BMET['mAP'],
                        'delta_Rank1':metrics['Rank-1']-FIT_BMET['Rank-1']})
    del out
RLOCAL_FIT=pd.DataFrame(RLOCAL_ROWS).sort_values(['mADM','mAP','Rank-1'],ascending=False).reset_index(drop=True)
RLOCAL_FIT.to_csv(RLOCAL_DIR/'rank_local_fit_trials.csv',index=False)
print(RLOCAL_FIT.to_string(index=False))
print('Saved:',RLOCAL_DIR/'rank_local_fit_trials.csv')



# =======================================================
#@title 29) FIT-only stability + single-positive guard — choose ONE config before AUDIT
import numpy as np,pandas as pd,json
from tqdm.auto import tqdm

# This is an internal stability check on the *same FIT* subset, not independent new data.
TOP_N=5  #@param {type:'integer'}
N_FOLDS=3 #@param {type:'integer'}
ranked=RLOCAL_FIT.head(max(1,min(TOP_N,len(RLOCAL_FIT))))
folds=np.array_split(np.random.default_rng(2027).permutation(len(FIT_IDS)),N_FOLDS)
baseline_q=rl.perquery_ranking_stats(FIT_BASE,FIT_IDS,GIDS)
positive_count=baseline_q['positive_count']
rows=[]
for item in tqdm(list(ranked.itertuples(index=False)),desc='29 / FIT fold stability',unit='config'):
    cfg={'top_k':int(item.top_k),'strength':float(item.strength),'tau':float(item.tau),'cap':float(item.cap)}
    out=rl.rank_local_score(FIT_BASE,FIT_P,FIT_TOP,**cfg)
    after=rl.perquery_ranking_stats(out,FIT_IDS,GIDS)
    fold_gains=[]
    for folds_rows in folds:
        subset_ids=FIT_IDS[folds_rows]
        old=sp.semantic_metrics(FIT_BASE[folds_rows],subset_ids,GIDS,Q[subset_ids],VAL_LABELS)
        new=sp.semantic_metrics(out[folds_rows],subset_ids,GIDS,Q[subset_ids],VAL_LABELS)
        fold_gains.append(float(new['mADM']-old['mADM']))
    one=positive_count==1
    one_delta=float(np.mean(after['rank1'][one]-baseline_q['rank1'][one])) if one.any() else 0.
    fit_gain=float(item.delta_mADM)
    # Conservative FIT gate: benchmark gain + no material loss in other metrics.
    passed=bool(fit_gain>=0.003 and item.delta_mAP>=-0.004 and
                item.delta_Rank1>=-0.004 and sum(x>0 for x in fold_gains)>=2 and
                one_delta>=-0.01)
    rows.append({**cfg,'FIT_gain_mADM':fit_gain,'FIT_gain_mAP':float(item.delta_mAP),
                 'FIT_gain_Rank1':float(item.delta_Rank1),'single_positive_delta_Rank1':one_delta,
                 'positive_folds':int(sum(x>0 for x in fold_gains)),
                 'fold_mADM_gains':json.dumps(fold_gains),'fit_pass':passed})
RLOCAL_STABILITY=pd.DataFrame(rows)
RLOCAL_STABILITY.to_csv(RLOCAL_DIR/'rank_local_fit_stability.csv',index=False)
passed=RLOCAL_STABILITY[RLOCAL_STABILITY.fit_pass]
chosen=passed.iloc[0] if len(passed) else RLOCAL_STABILITY.iloc[0]
RLOCAL_CONFIG={key:float(chosen[key]) if key!='top_k' else int(chosen[key])
               for key in ('top_k','strength','tau','cap')}
RLOCAL_FIT_PASS=bool(chosen.fit_pass)
(RLOCAL_DIR/'rank_local_selected_fit.json').write_text(json.dumps({
    'config':RLOCAL_CONFIG,'fit_pass':RLOCAL_FIT_PASS,
    'selection':'FIT-only; AUDIT not used for selection',
    'all_candidates':RLOCAL_STABILITY.to_dict(orient='records')},indent=2),encoding='utf-8')
print(RLOCAL_STABILITY.to_string(index=False))
print('Selected CONFIG:',RLOCAL_CONFIG,'FIT_PASS:',RLOCAL_FIT_PASS)
print('Next Cell 30 is an EXPLORATORY public AUDIT check (prior AUDIT was already inspected).')



# =======================================================
#@title 30) Public AUDIT — compare metrics, domains, single-positive + paired bootstrap
import numpy as np,pandas as pd,json
from tqdm.auto import tqdm
from abpr.query_aware_ranking import retrieval_metrics
from abpr.model import track2_metrics

nfit=len(FIT_IDS)
BA=np.asarray(BASE_DIST[nfit:],np.float32)
PA=np.asarray(CONTRADICTION_ALL[nfit:],np.float32)
ATOP=rl.topk_indices(BA,800)
A_NEW=rl.rank_local_score(BA,PA,ATOP,**RLOCAL_CONFIG)

AUDIT_BASE_MET=sp.semantic_metrics(BA,AUDIT_IDS,GIDS,Q[AUDIT_IDS],VAL_LABELS)
AUDIT_NEW_MET=sp.semantic_metrics(A_NEW,AUDIT_IDS,GIDS,Q[AUDIT_IDS],VAL_LABELS)
AUDIT_BASE_MET={k:float(AUDIT_BASE_MET[k]) for k in ('mADM','mAP','Rank-1')}
AUDIT_NEW_MET={k:float(AUDIT_NEW_MET[k]) for k in ('mADM','mAP','Rank-1')}

oldq=rl.perquery_ranking_stats(BA,AUDIT_IDS,GIDS)
newq=rl.perquery_ranking_stats(A_NEW,AUDIT_IDS,GIDS)
RLOCAL_PAIRED_CI=rl.paired_bootstrap_perquery(oldq,newq,repeats=500)
summary=[]
for group,mask in [('all',np.ones(len(AUDIT_IDS),bool)),
                   ('single_positive',oldq['positive_count']==1),
                   ('multiple_positives',oldq['positive_count']>1)]:
    if not mask.any():continue
    old_sub=rl.perquery_ranking_stats(BA[mask],AUDIT_IDS[mask],GIDS)
    new_sub=rl.perquery_ranking_stats(A_NEW[mask],AUDIT_IDS[mask],GIDS)
    for method,r in [('baseline',old_sub),('candidate',new_sub)]:
        summary.append({'group':group,'method':method,'queries':int(mask.sum()),
                        'mAP':float(np.mean(r['ap'])),'Rank-1':float(np.mean(r['rank1']))})
RLOCAL_GROUPS=pd.DataFrame(summary)
RLOCAL_GROUPS.to_csv(RLOCAL_DIR/'rank_local_audit_group_metrics.csv',index=False)

from abpr.core import infer_domain_id
VAL_DOMAINS=np.asarray([infer_domain_id(p) for p in VAL_NAMES],np.int8)
domain_rows=[]
for d,name in tqdm(enumerate(('Market1501','PA100k','PETA')),total=3,desc='30 / AUDIT per-domain',unit='domain'):
    gallery_mask=VAL_DOMAINS==d
    relevant_query_mask=np.isin(AUDIT_IDS,np.unique(GIDS[gallery_mask]))
    ids=AUDIT_IDS[relevant_query_mask]
    if not len(ids):continue
    for label,mat in [('baseline',BA),('candidate',A_NEW)]:
        sub=mat[relevant_query_mask][:,gallery_mask]
        r=retrieval_metrics(sub,ids,GIDS[gallery_mask])
        t=track2_metrics(sub,Q[ids],VAL_LABELS[gallery_mask])
        domain_rows.append({'domain':name,'method':label,'queries':len(ids),
                            'mADM':float(t['mADM']),'mAP':float(r['mAP']),
                            'Rank-1':float(r['Rank-1'])})
RLOCAL_DOMAIN=pd.DataFrame(domain_rows)
RLOCAL_DOMAIN.to_csv(RLOCAL_DIR/'rank_local_audit_domain_metrics.csv',index=False)
d_safe=True
for name in RLOCAL_DOMAIN.domain.unique():
    subset=RLOCAL_DOMAIN[RLOCAL_DOMAIN.domain==name].set_index('method')
    d_safe &= bool(subset.loc['candidate','mADM']>=subset.loc['baseline','mADM']-0.01)
sg=RLOCAL_GROUPS[RLOCAL_GROUPS.group=='single_positive'].set_index('method')
single_safe=bool(len(sg)==2 and sg.loc['candidate','Rank-1']>=sg.loc['baseline','Rank-1']-0.007)
RLOCAL_AUDIT_ACCEPTED=bool(RLOCAL_FIT_PASS and
     AUDIT_NEW_MET['mADM']>=AUDIT_BASE_MET['mADM']+0.005 and
     AUDIT_NEW_MET['mAP']>=AUDIT_BASE_MET['mAP']-0.005 and
     AUDIT_NEW_MET['Rank-1']>=AUDIT_BASE_MET['Rank-1']-0.005 and
     d_safe and single_safe)
RLOCAL_REPORT={'source':'Public VAL only — NOT Codabench hidden',
 'previous_public_audit_inspected':True,
 'method':'Rank-local confidence-gated attribute contradiction reranking',
 'gallery_ground_truth_used_by_scorer':False,
 'candidate_requires_SigLIP2_runtime_weights':True,
 'fit_pass':RLOCAL_FIT_PASS,'config':RLOCAL_CONFIG,
 'baseline_fit':FIT_BMET,'baseline_audit':AUDIT_BASE_MET,'candidate_audit':AUDIT_NEW_MET,
 'audit_delta_mADM':AUDIT_NEW_MET['mADM']-AUDIT_BASE_MET['mADM'],
 'paired_bootstrap_AP_R1_only':RLOCAL_PAIRED_CI,
 'domain_noninferior':bool(d_safe),'single_positive_noninferior':single_safe,
 'audit_accepted':RLOCAL_AUDIT_ACCEPTED,
 'automatic_submission_allowed':False,
 'note':'No hidden-result guarantee. No ZIP export until a compliant offline runtime computes SigLIP image features on hidden gallery.'}
(RLOCAL_DIR/'rank_local_audit_report.json').write_text(json.dumps(RLOCAL_REPORT,indent=2,ensure_ascii=False),encoding='utf-8')
print('AUDIT ORIGINAL:',AUDIT_BASE_MET)
print('AUDIT Rank-Local:',AUDIT_NEW_MET)
print('AUDIT mADM gain:',round(RLOCAL_REPORT['audit_delta_mADM'],6))
print('\nSingle-positive and other groups:\n',RLOCAL_GROUPS.to_string(index=False))
print('\nPer-domain:\n',RLOCAL_DOMAIN.to_string(index=False))
print('AP/Rank1 paired-bootstrap (not mADM CI):',RLOCAL_PAIRED_CI)
print('RLOCAL_AUDIT_ACCEPTED:',RLOCAL_AUDIT_ACCEPTED)



# =======================================================
#@title 31) Export new method report and code, with explicit Codabench submission gate
import zipfile,hashlib,json,time
from pathlib import Path
files_to_include=[RLOCAL_DIR/'rank_local_fit_trials.csv',
 RLOCAL_DIR/'rank_local_fit_stability.csv',
 RLOCAL_DIR/'rank_local_selected_fit.json',
 RLOCAL_DIR/'rank_local_audit_report.json',
 RLOCAL_DIR/'rank_local_audit_group_metrics.csv',
 RLOCAL_DIR/'rank_local_audit_domain_metrics.csv',
 Path(rl.__file__)]
README=RLOCAL_DIR/'README_submission_decision.txt'
README.write_text("""Error-guided rank-local ABPR evaluation — public VAL ONLY.
No hidden metric was measured. Model selection used FIT only; however previous work
inspected AUDIT, therefore AUDIT is exploratory and not a fresh independent holdout.
Gallery ground-truth labels and relevance IDs MUST NEVER enter deployment scoring.
This scorer depends on a SigLIP2 predicted-attribute penalty cache generated on VAL.
A real submission MUST regenerate that penalty on hidden gallery, with legal external
pretrained weights, correct 40 attributes, image ordering and an offline runtime test.
This report is NOT a ready-to-submit Codabench ZIP.
""",encoding='utf-8')
files_to_include.append(README)
RLOCAL_REPORT_ZIP=RLOCAL_DIR/'ABPR_RankLocal_Experiment_Report.zip'
with zipfile.ZipFile(RLOCAL_REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in tqdm(files_to_include,desc='31 / export diagnostic artifact',unit='file'):
        assert f.is_file(),f
        z.write(f,arcname=f.name)
print('REPORT:',RLOCAL_REPORT_ZIP)
print('FIT:',RLOCAL_FIT_PASS,'AUDIT:',RLOCAL_AUDIT_ACCEPTED)
if not RLOCAL_AUDIT_ACCEPTED:
    print('DECISION: KEEP current hidden 0.5389 baseline. Do not submit this method.')
else:
    print('PROMISING public candidate, BUT do not submit this report ZIP.')
    print('Next requirement: permitted SigLIP2 weights + full offline inference wrapper + Docker smoke test.')
print('Upload ABPR_RankLocal_Experiment_Report.zip to this chat for detailed next decision.')

