# ===== 20A) Source bootstrap =====
#@title 20A) Install source-only error-guided scoring code (no pip installs)
from pathlib import Path
import base64,zlib
method_file=Path('/content/abpr_error_guided_rerank.py')
method_file.write_bytes(zlib.decompress(base64.b64decode('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')))
print('Installed scorer at',method_file, 'bytes:',method_file.stat().st_size)


# ===== 21) Load report + fit TRAIN-only domain model (NO new backbone training) =====
#@title 21) ERROR-GUIDED ABPR — inspect diagnostics + TRAIN-only source-domain model
from pathlib import Path
import sys,json,hashlib,zipfile,shutil,time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Run after Cell 12 and Cell 15–20; no repeated inference required.
METHOD_DIR=Path(SIGLIP_DIR)/'error_guided_method'
METHOD_DIR.mkdir(parents=True,exist_ok=True)
assert Path(BEST_SUBMISSION_ZIP).is_file(), 'Missing baseline 0.5389 submission ZIP'
assert Path(TRAIN_CSV).is_file(), 'Missing official TRAIN gt.csv'
assert len(ATTR)==40 and len(VAL_NAMES)==len(GIDS)==len(VAL_LABELS)
assert len(SELECT_IDS)==len(FIT_IDS)+len(AUDIT_IDS)
assert np.array_equal(np.asarray(SELECT_IDS),np.r_[FIT_IDS,AUDIT_IDS])

# Self-contained module is provided in the extension notebook (Cell 21 bootstrap)
assert (Path('/content')/'abpr_error_guided_rerank.py').is_file() or (Path(SOURCE_ROOT)/'abpr_error_guided_rerank.py').is_file(), (
    'Method source missing: run the bootstrap cell supplied before this Cell')
if (Path('/content')/'abpr_error_guided_rerank.py').is_file():
    sys.path.insert(0,'/content')
else:
    sys.path.insert(0,str(SOURCE_ROOT))
import abpr_error_guided_rerank as eg

REPORT_DIR=Path(SIGLIP_DIR)/'error_analysis'
report_file=REPORT_DIR/'diagnostic_summary.json'
assert report_file.is_file(), f'Run Cell 15–20 first: {report_file}'
DIAG=json.loads(report_file.read_text(encoding='utf-8'))
print('Queries:',DIAG['queries'],'Gallery:',DIAG['gallery'])
print('Wrong Top1 explicit contradiction:',DIAG['wrong_top1_contradiction_fraction'])
print('Single positive query fraction:',DIAG['single_ID_positive_query_rate'])
print('Top errors:',DIAG['top_10_attribute_conflicts'])

TRAIN_FRAME=pd.read_csv(TRAIN_CSV)
assert set(ATTR).issubset(TRAIN_FRAME.columns)
from abpr.core import detect_image_column
tr_col=detect_image_column(TRAIN_FRAME)
TRAIN_LABEL_MATRIX=TRAIN_FRAME[ATTR].to_numpy(np.float32)
TRAIN_DOMAIN_VECTOR=np.array([eg.domain_from_path(s) for s in TRAIN_FRAME[tr_col].astype(str)],np.int64)
assert np.isin(TRAIN_DOMAIN_VECTOR,[0,1,2]).all(), 'Unknown TRAIN source-domain path'
TRAIN_DOMAIN_MODEL=eg.fit_domain_nb(TRAIN_LABEL_MATRIX,TRAIN_DOMAIN_VECTOR,smoothing=30)
np.savez_compressed(METHOD_DIR/'train_only_domain_nb.npz',prob=TRAIN_DOMAIN_MODEL['prob'],counts=TRAIN_DOMAIN_MODEL['counts'],attrs=np.asarray(ATTR))
VAL_DOMAINS=np.array([eg.domain_from_path(s) for s in VAL_NAMES],np.int8)
assert np.isin(VAL_DOMAINS,[0,1,2]).all()
print('TRAIN domain rows:',dict(zip(eg.DOMAINS,np.bincount(TRAIN_DOMAIN_VECTOR,minlength=3))))
print('VAL gallery rows:',dict(zip(eg.DOMAINS,np.bincount(VAL_DOMAINS,minlength=3))))
# IMPORTANT: Neither diagnostic VAL labels nor GIDS are used to compute candidate scores.


# ===== 22) Prepare predicted attribute contradictions (SigLIP adapter optional) =====
#@title 22) Build TRAIN-calibrated predicted-attribute penalty (optional, NO VAL label features)
import numpy as np,json,hashlib
RUN_CONTRADICTION_GATE=True #@param {type:'boolean'}
# External SigLIP 2 weights may not be eligible for Codabench; this is a public VAL experiment only.
GATE_TRUST_PRECISION=0.90 #@param {type:'number'}

if 'BASE_DIST' not in globals():
    BASE_DIST=np.load(Path(SIGLIP_DIR)/'baseline_fixed_queries.npy',mmap_mode='r')
BASE_DIST=np.asarray(BASE_DIST,dtype=np.float32)
assert BASE_DIST.shape==(len(SELECT_IDS),len(VAL_PATHS))
assert np.isfinite(BASE_DIST).all()

GATE_AVAILABLE=False
CONTRADICTION_ALL=None
GATE_META=None
ADAPTER_PATH=Path(SIGLIP_DIR)/'train_only_adapter.npz'
if RUN_CONTRADICTION_GATE:
    sig_cache=Path(SIGLIP_DIR)/'val_siglip2.npy'
    train_cache=Path(SIGLIP_DIR)/'train_siglip2.npy'
    if not (ADAPTER_PATH.is_file() and sig_cache.is_file() and train_cache.is_file()):
        print('[SKIP GATE] Missing SigLIP feature or adapter cache; domain-only method remains runnable.')
    else:
        with np.load(ADAPTER_PATH,allow_pickle=False) as a:
            w=np.asarray(a['weight'],np.float32);b=np.asarray(a['bias'],np.float32)
        SIG_VAL_FOR_METHOD=np.load(sig_cache,mmap_mode='r',allow_pickle=False)
        TRAIN_FEAT_FOR_METHOD=np.load(train_cache,mmap_mode='r',allow_pickle=False)
        sample=sp.balanced_sample(TRAIN_FRAME,tr_col,__import__('abpr.core',fromlist=['infer_domain_id']).infer_domain_id,TRAIN_PER_DOMAIN,seed=42)
        y_sample=sample[ATTR].to_numpy(np.float32)
        ordered_train_names=sample[tr_col].astype(str).tolist()
        # Verify original exact file ordering and model signature in the feature cache.
        # Never silently pair features with labels from a different sample order.
        if 'raw_train' in globals() and list(raw_train)==ordered_train_names and 'paths_train' in globals():
            ordered_train_paths=[str(p) for p in paths_train]
        else:
            ordered_train_paths=[str(RESOLVER.resolve(p)) for p in tqdm(ordered_train_names,desc='Verify TRAIN feature order')]
        def exact_cache_ok(cache,ordered_paths):
            meta=cache.with_suffix('.json')
            if not meta.is_file():return False
            info=json.loads(meta.read_text())
            expected=sp.digest_lines([SIGLIP_MODEL_ID,len(ordered_paths),*ordered_paths])
            return (info.get('signature')==expected and info.get('completed')==len(ordered_paths))
        cache_ok=exact_cache_ok(train_cache,ordered_train_paths) and exact_cache_ok(sig_cache,VAL_PATHS)
        if (not cache_ok or len(TRAIN_FEAT_FOR_METHOD)!=len(y_sample) or
                SIG_VAL_FOR_METHOD.shape[0]!=len(VAL_NAMES)):
            print('[SKIP GATE] Wrong model/image ordering or unfinished SigLIP cache. Use domain-only or rerun Cell 10–11.')
        else:
            gate=eg.train_confidence_gate(TRAIN_FEAT_FOR_METHOD,y_sample,w,b,
                                          min_precision=GATE_TRUST_PRECISION,min_support=12)
            GATE_META={key:np.asarray(v).tolist() for key,v in gate.items()}
            (METHOD_DIR/'train_only_gate.json').write_text(json.dumps(GATE_META,indent=2),encoding='utf-8')
            print('TRAIN-only reliable + predictions:',int(gate['usable_hi'].sum()),'/40')
            print('TRAIN-only reliable - predictions:',int(gate['usable_lo'].sum()),'/40')
            print('Computing contradiction penalties from PREDICTED attributes only (NO VAL_LABELS)...')
            # Fixed attribute weights; selection of gamma happens only on FIT.
            attr_w=np.ones(40,np.float32)
            for j,name in enumerate(ATTR):
                if name.startswith(('Accessory-', 'Hair-Length-', 'UpperBody-Color-', 'LowerBody-Color-')):
                    attr_w[j]=1.35
            CONTRADICTION_ALL=eg.contradiction_penalty(Q[SELECT_IDS],SIG_VAL_FOR_METHOD,w,b,gate,attr_weights=attr_w)
            assert CONTRADICTION_ALL.shape==BASE_DIST.shape and np.isfinite(CONTRADICTION_ALL).all()
            GATE_AVAILABLE=bool(gate['usable_hi'].sum()+gate['usable_lo'].sum()>=5)
            np.save(METHOD_DIR/'predicted_contradiction_penalty.npy',CONTRADICTION_ALL)
            print('Penalty prepared:',CONTRADICTION_ALL.shape,'gate enabled:',GATE_AVAILABLE)
print('No VAL_LABELS/GIDS used by scorer. These are used ONLY by metrics below.')


# ===== 23) FIT-only hyperparameter search with progress =====
#@title 23) FIT-only search — domain calibration + optional high-confidence attribute penalties
from tqdm.auto import tqdm
import pandas as pd,numpy as np

FIT_N=len(FIT_IDS)
B_FIT=BASE_DIST[:FIT_N]
Q_FIT=Q[FIT_IDS]
P_FIT=None if not GATE_AVAILABLE else CONTRADICTION_ALL[:FIT_N]
from abpr.query_aware_ranking import retrieval_metrics
from abpr.model import track2_metrics

def eg_metrics(mat,ids):
    m=sp.semantic_metrics(np.asarray(mat,np.float32),np.asarray(ids),GIDS,Q[ids],VAL_LABELS)
    return {k:float(m[k]) for k in ('mADM','mAP','Rank-1')}

FIT_BASELINE_METRICS=eg_metrics(B_FIT,FIT_IDS)
GRID=[{'method':'domain','alpha':float(a),'beta':float(b),'temp':float(temperature)}
      for a,b in [(0,0.08),(0,0.20),(0,0.4),(.15,0),(.35,0),(.15,.08),(.15,.20),(.35,.08),(.35,.20)]
      for temperature in (.4,1.0)]
if GATE_AVAILABLE:
    GRID += [{'method':'gate','gamma':float(g)} for g in (.05,.10,.20,.35)]
    GRID += [{'method':'domain_gate','alpha':float(a),'beta':float(b),'gamma':float(g),'temp':0.6}
             for a,b,g in [(.15,.08,.05),(.15,.2,.10),(.35,.08,.10),(.35,.2,.20)]]

FIT_TRIALS=[]
for config in tqdm(GRID,desc='23 / FIT-only model-selection',unit='trial'):
    try:
        scores=eg.apply_candidate(B_FIT,Q_FIT,VAL_DOMAINS,TRAIN_DOMAIN_MODEL,config,penalty=P_FIT)
        mm=eg_metrics(scores,FIT_IDS)
        FIT_TRIALS.append({**config,**mm,
                           'delta_mADM':mm['mADM']-FIT_BASELINE_METRICS['mADM'],
                           'delta_mAP':mm['mAP']-FIT_BASELINE_METRICS['mAP'],
                           'delta_Rank1':mm['Rank-1']-FIT_BASELINE_METRICS['Rank-1']})
    except Exception as e:
        raise RuntimeError(f'Failed method config {config}: {e}') from e
FIT_TRIALS=pd.DataFrame(FIT_TRIALS).sort_values(['mADM','mAP','Rank-1'],ascending=False)
FIT_TRIALS.to_csv(METHOD_DIR/'fit_trials.csv',index=False)
print('FIT BASELINE:',FIT_BASELINE_METRICS)
print('TOP FIT:',FIT_TRIALS.head(10).reindex(columns=['method','alpha','beta','temp','gamma','mADM','delta_mADM','mAP','Rank-1']).to_string(index=False))
EG_WIN=FIT_TRIALS.iloc[0].dropna().to_dict()
EG_CONFIG={k:EG_WIN[k] for k in ('method','alpha','beta','gamma','temp') if k in EG_WIN}
EG_FIT_PASS=bool(EG_WIN['mADM']>=FIT_BASELINE_METRICS['mADM']+.003 and
                 EG_WIN['mAP']>=FIT_BASELINE_METRICS['mAP']-.008 and
                 EG_WIN['Rank-1']>=FIT_BASELINE_METRICS['Rank-1']-.008)
print('Selected:',EG_CONFIG,'FIT_PASS:',EG_FIT_PASS)


# ===== 24) AUDIT gate, per-domain and one-positive checks — no audit tuning =====
#@title 24) Held-out AUDIT — full mADM/mAP/R1, per-source and single-ID-positive
from tqdm.auto import tqdm
import numpy as np,pandas as pd,json
AUDIT_N=len(AUDIT_IDS)
B_AUD=BASE_DIST[len(FIT_IDS):]
AUD_BASE=eg_metrics(B_AUD,AUDIT_IDS)
# Important: even if FIT fails, calculate exploratory AUDIT candidate, but NEVER tune on AUDIT.
with tqdm(total=3,desc='24 / AUDIT metrics and safety checks',unit='step') as progress:
    P_AUD=None if not GATE_AVAILABLE else CONTRADICTION_ALL[len(FIT_IDS):]
    EG_AUDIT_SCORES=eg.apply_candidate(B_AUD,Q[AUDIT_IDS],VAL_DOMAINS,TRAIN_DOMAIN_MODEL,EG_CONFIG,penalty=P_AUD)
    progress.update(1)
    AUD_CAND=eg_metrics(EG_AUDIT_SCORES,AUDIT_IDS)
    progress.update(1)
    eval_rows=[]
    for d in (0,1,2):
        mask=VAL_DOMAINS==d
        source_ids=np.unique(GIDS[mask])
        query_mask=np.isin(AUDIT_IDS,source_ids)
        ids=AUDIT_IDS[query_mask]
        if not len(ids):continue
        for method,values in [('baseline',B_AUD),('candidate',EG_AUDIT_SCORES)]:
            s=values[query_mask][:,mask]
            mr=retrieval_metrics(s,ids,GIDS[mask])
            mt=track2_metrics(s,Q[ids],VAL_LABELS[mask])
            eval_rows.append({'domain':eg.DOMAINS[d],'method':method,'queries':len(ids),
                              'mADM':float(mt['mADM']),'mAP':float(mr['mAP']),
                              'Rank-1':float(mr['Rank-1'])})
    progress.update(1)
DOMAIN_CHECK=pd.DataFrame(eval_rows)
DOMAIN_CHECK.to_csv(METHOD_DIR/'audit_domains.csv',index=False)
base_domain={r.domain:r.mADM for r in DOMAIN_CHECK.itertuples() if r.method=='baseline'}
cand_domain={r.domain:r.mADM for r in DOMAIN_CHECK.itertuples() if r.method=='candidate'}
domain_safe=all(cand_domain[k]>=v-0.010 for k,v in base_domain.items())

one_positive=np.isin(AUDIT_IDS,np.unique(GIDS)) & (np.array([(GIDS==qid).sum() for qid in AUDIT_IDS])==1)
ONEPOS_CHECK=[]
for label,mask in [('single_positive',one_positive),('multiple_positives',~one_positive)]:
    if mask.sum()==0:continue
    for method,scores in [('baseline',B_AUD),('candidate',EG_AUDIT_SCORES)]:
        mr=retrieval_metrics(scores[mask],AUDIT_IDS[mask],GIDS)
        ONEPOS_CHECK.append({'group':label,'method':method,'queries':int(mask.sum()),
                             'mAP':float(mr['mAP']),'Rank-1':float(mr['Rank-1'])})
ONEPOS_CHECK=pd.DataFrame(ONEPOS_CHECK)
ONEPOS_CHECK.to_csv(METHOD_DIR/'audit_single_positive.csv',index=False)

# Cross-domain positives are vulnerable to overconfident source-domain priors.
source_counts=np.array([len(np.unique(VAL_DOMAINS[GIDS==qid])) for qid in AUDIT_IDS])
CROSS_DOMAIN=[]
for group,mask in [('one_source',source_counts==1),('multiple_sources',source_counts>1)]:
    if not mask.any():continue
    for method,scores in [('baseline',B_AUD),('candidate',EG_AUDIT_SCORES)]:
        mm=eg_metrics(scores[mask],AUDIT_IDS[mask])
        CROSS_DOMAIN.append({'group':group,'method':method,'queries':int(mask.sum()),**mm})
CROSS_CHECK=pd.DataFrame(CROSS_DOMAIN)
CROSS_CHECK.to_csv(METHOD_DIR/'audit_cross_domain.csv',index=False)
try:
    orig_cross=float(CROSS_CHECK.query("group=='multiple_sources' and method=='baseline'").iloc[0]['mADM'])
    cand_cross=float(CROSS_CHECK.query("group=='multiple_sources' and method=='candidate'").iloc[0]['mADM'])
    cross_safe=cand_cross>=orig_cross-0.008
except IndexError:
    cross_safe=True

# Attribute-wise Top1 contradiction counts from official VAL labels: REPORT ONLY.
# None of these labels are used to produce distances or select hyperparameters.
orig_top=np.argmin(B_AUD,axis=1)
new_top=np.argmin(EG_AUDIT_SCORES,axis=1)
ATT_IMPROVEMENTS=[]
for j,name in enumerate(ATTR):
    known=Q[AUDIT_IDS,j]!=-1
    old_bad=known & (VAL_LABELS[orig_top,j]!=-1) & (VAL_LABELS[orig_top,j]!=Q[AUDIT_IDS,j])
    new_bad=known & (VAL_LABELS[new_top,j]!=-1) & (VAL_LABELS[new_top,j]!=Q[AUDIT_IDS,j])
    ATT_IMPROVEMENTS.append({'attribute':name,'audit_queries':int(known.sum()),
        'baseline_top1_contradictions':int(old_bad.sum()),
        'candidate_top1_contradictions':int(new_bad.sum()),
        'contradictions_removed':int(old_bad.sum()-new_bad.sum())})
ATTR_RECOVERY=pd.DataFrame(ATT_IMPROVEMENTS).sort_values('contradictions_removed',ascending=False)
ATTR_RECOVERY.to_csv(METHOD_DIR/'attribute_top1_recovery.csv',index=False)

AUDIT_ACCEPTED_ERROR_GUIDED=bool(
    EG_FIT_PASS and
    AUD_CAND['mADM']>=AUD_BASE['mADM']+.005 and
    AUD_CAND['mAP']>=AUD_BASE['mAP']-.005 and
    AUD_CAND['Rank-1']>=AUD_BASE['Rank-1']-.005 and domain_safe and cross_safe
)
EG_REPORT={
    'source':'public VAL only, not hidden Codabench',
    'method':'Source-domain calibration and predicted-attribute contradiction reranking',
    'source_train_only':True,
    'gallery_labels_used_in_scoring':False,
    'fit_queries':int(len(FIT_IDS)),'audit_queries':int(len(AUDIT_IDS)),
    'best_fit':EG_WIN,'selected_config':EG_CONFIG,'fit_baseline':FIT_BASELINE_METRICS,
    'fit_pass':bool(EG_FIT_PASS),'audit_baseline':AUD_BASE,'audit_candidate':AUD_CAND,
    'domain_noninferior':bool(domain_safe), 'cross_domain_noninferior':bool(cross_safe),
    'audit_accepted':bool(AUDIT_ACCEPTED_ERROR_GUIDED),
    'can_export_domain_only':bool(EG_CONFIG['method']=='domain' and AUDIT_ACCEPTED_ERROR_GUIDED),
    'prior_audit_inspected':True,
    'warning':'Previous error analysis included public AUDIT labels; this is not a fully untouched independent validation.'
}
for key in ['best_fit','selected_config']:
    EG_REPORT[key]={k:(float(v) if isinstance(v,np.floating) else v) for k,v in EG_REPORT[key].items()}
(METHOD_DIR/'error_guided_audit_report.json').write_text(json.dumps(EG_REPORT,indent=2,ensure_ascii=False,default=float))
print('AUDIT baseline:',AUD_BASE)
print('AUDIT candidate:',AUD_CAND)
print('PER DOMAIN:');print(DOMAIN_CHECK.to_string(index=False))
print('CROSS DOMAIN:');print(CROSS_CHECK.to_string(index=False))
print('TOP ATTRIBUTE RECOVERY:');print(ATTR_RECOVERY.head(12).to_string(index=False))
print('SINGLE VS MULTIPLE:');print(ONEPOS_CHECK.to_string(index=False))
print('AUDIT_ACCEPTED_ERROR_GUIDED:',AUDIT_ACCEPTED_ERROR_GUIDED)
print('Safe domain-only packaging eligible:',EG_REPORT['can_export_domain_only'])


# ===== 25) GATED offline ZIP export (domain-only, preserve baseline) =====
#@title 25) OPTIONAL: export domain-only candidate to OFFLINE Codabench ZIP + smoke test
# To prevent accidental submission of SigLIP pretrained weights in an ineligible environment,
# only the source-only domain method can be exported here.
EXPORT_ERROR_GUIDED_ZIP=False #@param {type:'boolean'}
import zipfile,tempfile,shutil,subprocess,sys,os,stat,json
from pathlib import Path
if not EXPORT_ERROR_GUIDED_ZIP:
    print('Export OFF. Check method report first; enable only if approved.')
else:
    if not EG_REPORT['can_export_domain_only']:
        raise RuntimeError('Export blocked: selected method not domain-only OR failed FIT/AUDIT gate.')
    archive=Path(BEST_SUBMISSION_ZIP)
    PACKAGE_DIR=METHOD_DIR/'submission_domain_only'
    if PACKAGE_DIR.exists():shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True)
    with zipfile.ZipFile(archive) as src:
        for m in src.infolist():
            dst=(PACKAGE_DIR/m.filename).resolve()
            if not dst.is_relative_to(PACKAGE_DIR.resolve()) or m.filename.startswith('/'):
                raise ValueError('Unsafe base archive entry')
            if stat.S_IFMT(m.external_attr>>16)==stat.S_IFLNK:
                raise ValueError('ZIP symlink not allowed')
        src.extractall(PACKAGE_DIR)
    assert (PACKAGE_DIR/'run.py').is_file(), 'Baseline run.py missing'
    (PACKAGE_DIR/'run.py').rename(PACKAGE_DIR/'base_run.py')
    source_method=Path(eg.__file__)
    shutil.copy2(source_method,PACKAGE_DIR/'abpr_error_guided_rerank.py')
    assets=PACKAGE_DIR/'assets';assets.mkdir(parents=True,exist_ok=True)
    shutil.copy2(METHOD_DIR/'train_only_domain_nb.npz',assets/'domain_nb.npz')
    (assets/'domain_rerank_config.json').write_text(json.dumps(EG_CONFIG,indent=2,default=float))
    (PACKAGE_DIR/'run.py').write_text('''\
# Offline re-ranking wrapper: preserve original runtime, adjust query/domain scores.
from pathlib import Path
import numpy as np,json
import base_run
import abpr_error_guided_rerank as rerank
ROOT=Path(__file__).resolve().parent
with np.load(ROOT/'assets/domain_nb.npz',allow_pickle=False) as n:
    MODEL={'prob':n['prob'], 'counts':n['counts']}
    ATTR_ORDER=[str(x) for x in n['attrs']]
CONFIG=json.loads((ROOT/'assets/domain_rerank_config.json').read_text())
def load_model(*args,**kwargs):
    return base_run.load_model(*args,**kwargs)
def rank_gallery(payload):
    original=base_run.rank_gallery(payload)
    if not isinstance(original,dict) or 'distances' not in original:
        raise RuntimeError('Unexpected original submission output')
    query=np.asarray(payload['queries'],np.float32)
    assert query.ndim==2 and query.shape[1]==len(ATTR_ORDER)
    passed=payload.get('attribute_names')
    if passed is not None:
        passed=list(map(str,passed))
        if passed!=ATTR_ORDER:
            if set(passed)!=set(ATTR_ORDER) or len(passed)!=len(ATTR_ORDER):
                raise ValueError('Attribute names mismatch; refusing to reorder ambiguously')
            order=[passed.index(x) for x in ATTR_ORDER]
            query=query[:,order]
    gallery=payload['gallery']
    paths=[e.get('image_path',e.get('path','')) if isinstance(e,dict) else str(e) for e in gallery]
    domains=np.asarray([rerank.domain_from_path(p) for p in paths],np.int8)
    old=np.asarray(original['distances'],np.float32)
    if old.shape!=(len(query),len(paths)) or not np.isfinite(old).all():
        raise ValueError('Baseline output shape / finite check failed')
    improved=rerank.domain_adjusted(old,query,domains,MODEL,alpha=CONFIG.get('alpha',0.0),
               beta=CONFIG.get('beta',0.0),temp=CONFIG.get('temp',3.0))
    original['distances']=improved.tolist()
    return original
''',encoding='utf-8')
    ZIP_OUT=METHOD_DIR/'Submission_ABPR_ErrorGuided_Domain.zip'
    with zipfile.ZipFile(ZIP_OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as zout:
        for f in sorted(PACKAGE_DIR.rglob('*')):
            if f.is_file():zout.write(f,f.relative_to(PACKAGE_DIR).as_posix())
    with zipfile.ZipFile(ZIP_OUT) as z:
        assert z.testzip() is None
        assert {'run.py','base_run.py','metadata.yaml','abpr_runtime.py','assets/model.pt',
                'abpr_error_guided_rerank.py','assets/domain_nb.npz','assets/domain_rerank_config.json'}<=set(z.namelist())
    # Offline smoke test 2 queries x 5 images, and original-vs-wrapper shape.
    smoke='''import os,sys,json,numpy as np
os.environ['HF_HUB_OFFLINE']='1'
import run,base_run
p=json.load(open('smoke.json'))
a=np.asarray(base_run.rank_gallery(p)['distances'],np.float32)
b=np.asarray(run.rank_gallery(p)['distances'],np.float32)
assert a.shape==b.shape==(2,5) and np.isfinite(b).all()
print('OFFLINE SMOKE PASS',b.shape)
'''
    with tempfile.TemporaryDirectory(prefix='abpr_eg_smoke_') as td:
        with zipfile.ZipFile(ZIP_OUT) as z:z.extractall(td)
        obj={'gallery':[{'image_path':str(x)} for x in VAL_PATHS[:5]],
             'queries':Q[SELECT_IDS[:2]].tolist(),'attribute_names':list(ATTR)}
        (Path(td)/'smoke.json').write_text(json.dumps(obj))
        (Path(td)/'smoke.py').write_text(smoke)
        p=subprocess.run([sys.executable,'-u','smoke.py'],cwd=td,text=True,capture_output=True,
                         timeout=1200,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
        print(p.stdout[-3000:]);print(p.stderr[-3000:])
        if p.returncode:raise RuntimeError('Offline smoke failed, DO NOT submit.')
    print('CANDIDATE ZIP:',ZIP_OUT)
    print('Not submitted automatically. Verify official rules, size, Docker limits before uploading.')


# ===== 26) Archive scores, metrics and method code =====
#@title 26) Save reproducible evaluation report ZIP (can send here for further analysis)
import zipfile,hashlib,json
EG_REPORT_ZIP=METHOD_DIR/'ABPR_ErrorGuided_Experiment_Report.zip'
paths=[METHOD_DIR/'fit_trials.csv',METHOD_DIR/'audit_domains.csv',METHOD_DIR/'audit_single_positive.csv',
       METHOD_DIR/'error_guided_audit_report.json',METHOD_DIR/'audit_cross_domain.csv',METHOD_DIR/'attribute_top1_recovery.csv',METHOD_DIR/'train_only_domain_nb.npz',Path(eg.__file__)]
if (METHOD_DIR/'train_only_gate.json').is_file():paths.append(METHOD_DIR/'train_only_gate.json')
with zipfile.ZipFile(EG_REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zout:
    for p in tqdm(paths,desc='26 / package method evaluation',unit='file'):
        assert p.is_file(),p
        zout.write(p,arcname=p.name)
print('READY:',EG_REPORT_ZIP,round(EG_REPORT_ZIP.stat().st_size/1024/1024,3),'MiB')
print('Decision:', 'Consider Codabench ONLY after offline smoke and eligibility checks' if AUDIT_ACCEPTED_ERROR_GUIDED else 'KEEP existing 0.5389 ZIP — insufficient public audit gain')
