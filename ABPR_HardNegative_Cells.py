

#@title 52) HNM — Strict preflight + install TRAIN-only hard-negative metric adapter
from pathlib import Path
import sys,os,json,hashlib,importlib
import numpy as np,pandas as pd
from tqdm.auto import tqdm

# This continuation requires the restored, signature-checked setup from Cell 46.
# If runtime was reset, run Cell 39 then Cell 46. They restore source/metadata/caches without retraining.
required_vars=['sp','dense','rr','TRAIN_FEAT','TRAIN_LABELS','TRAIN_DOMAINS',
               'DENSE_MODEL','DENSE_DEVICE','DENSE_IMG','DENSE_Q_EMBED',
               'DENSE_SEM','DENSE_CKPT','SIGLIP_DIR','sha256_stream','Q','VAL_NAMES','VAL_LABELS','GIDS',
               'FIT_IDS','AUDIT_IDS','SELECT_IDS','BASE_DIST','ATTR','TRAIN_META','manifest']
missing=[k for k in required_vars if k not in globals()]
if missing:
    raise RuntimeError('Missing prior runtime variables: '+', '.join(missing)+
       '. Run Cell 39 then Cell 46 to restore checked metadata. Also run Cell 41 if needed. No retraining required.')

HNM_DIR=Path(SIGLIP_DIR)/'hardnegative_metric_method'
HNM_DIR.mkdir(parents=True,exist_ok=True)
HNM_MODULE=HNM_DIR/'abpr_hardnegative_metric_adapter.py'
# Do not silently use a different version of the implementation.
HNM_SOURCE = r'''"""Frozen DenseQuery feature-space residual metric adapter for ABPR Track 2.
TRAIN labels only for optimization. VAL labels allowed ONLY in notebook evaluations.
No offline/Codabench inference runtime is bundled with this training experiment.
"""
from __future__ import annotations
import os, json, math, random, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

VERSION = 'hnuma-trainonly-20261013'

def normalize_np(x):
    a=np.asarray(x,np.float32)
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-8)

class ResidualBranch(nn.Module):
    def __init__(self, dim, hidden=96):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,dim))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        return F.normalize(x+0.35*self.net(x),dim=-1)

class PairedAdapter(nn.Module):
    def __init__(self, dim=128, hidden=96):
        super().__init__()
        self.image=ResidualBranch(dim,hidden)
        self.query=ResidualBranch(dim,hidden)
    def forward(self,x,q):
        return self.image(x),self.query(q)


def positive_mask(y):
    """Multi-positive when all observed TRAIN attribute values agree exactly."""
    known=(y==0)|(y==1)
    matches=((y[:,None,:]==y[None,:,:]) | ~(known[:,None,:]&known[None,:,:])).all(-1)
    # A missing label is not proof of positive: require matching observation mask.
    same_known=(known[:,None,:]==known[None,:,:]).all(-1)
    pos=matches & same_known & known.any(-1)[:,None]
    pos.fill_diagonal_(True)
    return pos


def mp_contrastive(images,queries,labels,temp=0.12,teacher_weight=0.25,
                   teacher_scores=None):
    sim=queries@images.T
    pos=positive_mask(labels)
    diag=torch.diag(sim)
    logits=sim/temp
    loss=(torch.logsumexp(logits,dim=1)
        -torch.logsumexp(logits.masked_fill(~pos,-1e4),dim=1)).mean()
    neg=logits.masked_fill(pos,-1e4).max(1).values
    hard=F.softplus(neg-logits.diagonal()).mean()
    if teacher_scores is None:
        drift=sim.sum()*0.
    else:
        # Relative scores: teacher geometry regularizer discourages loss of retrieval robustness.
        drift=F.mse_loss(sim,teacher_scores)
    return loss+0.15*hard+teacher_weight*drift, (float(loss.detach()),float(hard.detach()),float(drift.detach()))

@torch.inference_mode()
def holdout_probe(model,x,q,y,device):
    model.eval()
    tx=torch.as_tensor(np.asarray(x,np.float32),device=device)
    tq=torch.as_tensor(np.asarray(q,np.float32),device=device)
    labs=torch.as_tensor(np.asarray(y,np.float32),device=device)
    im,qu=model(tx,tq)
    sim=(qu@im.T).cpu().numpy()
    target=positive_mask(labs).cpu().numpy()
    ranks=np.argsort(-sim,axis=1,kind='stable')
    pos=target[np.arange(len(ranks))[:,None],ranks]
    count=target.sum(1)
    positions=np.argmax(pos,axis=1)
    r1=float(pos[:,0].mean())
    mrr=float((1/(positions+1)).mean())
    # All identical labels are considered valid positives in TRAIN-only holdout.
    return {'r1':r1,'mrr':mrr,'positive_mean':float(count.mean())}


def _save(path,model,opt,ep,best,signature,cfg,history):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.pt')
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                'epoch':int(ep),'best':float(best),'signature':signature,
                'cfg':cfg,'history':history},tmp)
    os.replace(tmp,path)


def fit(x,q,y,domains,out,signature,epochs=8,batch=192,lr=0.00015,seed=427,
        force=False):
    X=normalize_np(x);Q=normalize_np(q);Y=np.asarray(y,np.float32);D=np.asarray(domains,np.int64)
    assert X.shape==Q.shape and X.ndim==2 and X.shape[1]==128
    assert Y.shape==(len(X),40) and D.shape==(len(X),)
    assert np.isin(Y,[-1,0,1]).all() and np.isin(D,[0,1,2]).all()
    assert np.isfinite(X).all() and np.isfinite(Q).all()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=dict(version=VERSION,epochs=int(epochs),batch=int(batch),lr=float(lr),seed=int(seed),
             input_dim=128,hidden=96,holdout_fraction=0.2,teacher_weight=0.25,
             training='domain-balanced-multi-positive-hard-negative',training_labels='TRAIN-only')
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    train=[];val=[]
    for d in (0,1,2):
        idx=np.flatnonzero(D==d);rng.shuffle(idx)
        assert len(idx)>=25, f'Domain {d} too small'
        hold=max(5,round(.2*len(idx)))
        val.extend(idx[:hold]);train.extend(idx[hold:])
    train=np.asarray(train,dtype=np.int64);val=np.asarray(val,dtype=np.int64)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=PairedAdapter().to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.01)
    last=out/'metric_adapter_latest.pt';best_file=out/'metric_adapter_best.pt'
    start=0;best=-1e10;history=[]
    if last.exists() and not force:
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['signature']!=signature or state['cfg']!=cfg:
            raise RuntimeError('Existing checkpoint has a different training signature/config; use a new output folder')
        net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        start=int(state['epoch'])+1;best=float(state['best']);history=state['history']
        print('[RESUME] Adapter epoch',start+1,flush=True)
    buckets=[train[D[train]==d] for d in range(3)]
    steps=max(1,math.ceil(len(train)/batch))
    for ep in range(start,epochs):
        net.train();logs=[]
        for _ in tqdm(range(steps),desc=f'Adapter ep {ep+1}/{epochs}',unit='batch',leave=False):
            sel=np.concatenate([rng.choice(buckets[d],batch//3+int(d<batch%3),replace=True)
                                for d in range(3)])
            rng.shuffle(sel)
            ix=torch.as_tensor(X[sel],device=device)
            qx=torch.as_tensor(Q[sel],device=device)
            yy=torch.as_tensor(Y[sel],device=device)
            a,b=net(ix,qx)
            teacher_score=qx@ix.T
            loss,detail=mp_contrastive(a,b,yy,teacher_scores=teacher_score)
            # Preserve identity and prevent large embedding displacement.
            loss=loss+0.05*((a-ix).square().sum(-1).mean()+(b-qx).square().sum(-1).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.0);opt.step()
            logs.append([float(loss.detach()),*detail])
        v=holdout_probe(net,X[val],Q[val],Y[val],device)
        score=v['r1']+0.20*v['mrr']
        stat={'epoch':ep+1,'loss':float(np.mean(logs,axis=0)[0]),'holdout_r1':v['r1'],
              'holdout_mrr':v['mrr'],'holdout_positive_mean':v['positive_mean']}
        history.append(stat)
        if score>best:
            best=score;_save(best_file,net,opt,ep,best,signature,cfg,history)
        _save(last,net,opt,ep,best,signature,cfg,history)
        (out/'metric_training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print('[TRAIN-ONLY]',stat,flush=True)
    if not best_file.exists():raise FileNotFoundError(best_file)
    return best_file,history


def load(path,device=None):
    state=torch.load(path,map_location='cpu',weights_only=False)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net=PairedAdapter().to(device).eval();net.load_state_dict(state['model'])
    return net,device,state

@torch.inference_mode()
def encode(model,vectors,branch,device='cpu',batch=1024):
    out=[]
    for i in tqdm(range(0,len(vectors),batch),desc='Encode '+branch,unit='batch'):
        z=torch.as_tensor(np.asarray(vectors[i:i+batch],np.float32),device=device)
        transformed=(model.image if branch=='image' else model.query)(F.normalize(z,dim=-1))
        out.append(transformed.cpu().numpy())
    return np.concatenate(out,axis=0).astype(np.float32)

def candidate_distance(query,images,batch=32):
    q=normalize_np(query);x=normalize_np(images)
    out=np.empty((len(q),len(x)),np.float32)
    for i in tqdm(range(0,len(q),batch),desc='New retrieval distances',unit='query batch'):
        out[i:i+batch]=-(q[i:i+batch]@x.T)
    assert np.isfinite(out).all()
    return out


def per_query_ap(scores,ids,gids):
    ids=np.asarray(ids);gids=np.asarray(gids)
    ap=np.zeros(len(ids),np.float32);r1=np.zeros(len(ids),np.float32)
    for i in tqdm(range(len(ids)),desc='Evaluate paired AP/Rank1',unit='query',leave=False):
        order=np.argsort(scores[i],kind='stable');pos=(gids[order]==ids[i])
        n=int(pos.sum());r1[i]=float(pos[0]);where=np.flatnonzero(pos)
        ap[i]=float((np.arange(1,n+1)/(where+1)).mean()) if n else 0.
    return ap,r1
'''
if not HNM_MODULE.is_file() or HNM_MODULE.read_text(encoding='utf-8') != HNM_SOURCE:
    HNM_MODULE.write_text(HNM_SOURCE,encoding='utf-8')
sys.path.insert(0,str(HNM_DIR))
import abpr_hardnegative_metric_adapter as hm
importlib.reload(hm)

assert len(ATTR)==40 and list(ATTR)==list(__import__('abpr.core',fromlist=['ATTRIBUTE_NAMES']).ATTRIBUTE_NAMES)
assert len(VAL_NAMES)==len(GIDS)==len(VAL_LABELS)==len(DENSE_IMG)
assert np.array_equal(np.r_[FIT_IDS,AUDIT_IDS],SELECT_IDS)
assert tuple(BASE_DIST.shape)==(len(SELECT_IDS),len(VAL_NAMES))
assert DENSE_SEM.shape==BASE_DIST.shape and DENSE_IMG.shape[1]==128
assert np.asarray(TRAIN_FEAT).shape[0]==len(TRAIN_LABELS)==len(TRAIN_DOMAINS)
assert sp.digest_lines(VAL_NAMES)==manifest['gallery_paths_sha256']
assert np.isfinite(DENSE_IMG).all() and np.isfinite(BASE_DIST).all()
assert np.isin(TRAIN_LABELS,[-1,0,1]).all() and np.isin(TRAIN_DOMAINS,[0,1,2]).all()
assert TRAIN_META['signature']==json.loads((Path(SIGLIP_DIR)/'train_siglip2.json').read_text())['signature']

HNM_SIG=sp.digest_lines([hm.VERSION,sha256_stream(DENSE_CKPT),TRAIN_META['signature'],
  manifest['gallery_paths_sha256'],hashlib.sha256(np.asarray(TRAIN_LABELS,np.float32).tobytes()).hexdigest(),
  sp.digest_lines(ATTR)])
N_FIT_HNM=len(FIT_IDS)
print('[52 PASS] Gallery',len(VAL_NAMES),'TRAIN',len(TRAIN_FEAT),'FIT',len(FIT_IDS),'AUDIT',len(AUDIT_IDS))
print('Frozen backbone and DenseQuery remain untouched; training labels are TRAIN-only.')
print('Output:',HNM_DIR)


# ======================================================================

#@title 53) HNM — TRAIN-only paired residual representation training (resume + progress)
import numpy as np,torch,json,hashlib
HNM_EPOCHS=8  #@param {type:'integer'}
HNM_BATCH=192 #@param {type:'integer'}
HNM_LR=0.00015 #@param {type:'number'}
HNM_FORCE_RETRAIN=False #@param {type:'boolean'}

# Construct exact paired image/query features from frozen trained DenseQuery.
# This stage uses only TRAIN SigLIP features and official TRAIN labels; NO VAL labels.
train_cache=HNM_DIR/'train_dense_image_query.npz'
train_cache_meta=HNM_DIR/'train_dense_image_query.json'
sig=sp.digest_lines([HNM_SIG,'dense_image_query_cache'])
cache_ok=(train_cache.is_file() and train_cache_meta.is_file() and
    json.loads(train_cache_meta.read_text()).get('signature')==sig)
if cache_ok:
    with np.load(train_cache,allow_pickle=False) as a:
        HNM_TRAIN_IMG=np.asarray(a['image'],np.float32)
        HNM_TRAIN_Q=np.asarray(a['query'],np.float32)
    print('[CACHE HIT] Paired TRAIN representations',HNM_TRAIN_IMG.shape)
else:
    HNM_TRAIN_IMG,_=dense.image_outputs(DENSE_MODEL,TRAIN_FEAT,DENSE_DEVICE,chunk=512)
    HNM_TRAIN_Q=dense.query_outputs(DENSE_MODEL,TRAIN_LABELS,DENSE_DEVICE,chunk=512)
    tmp=train_cache.with_name('train_dense_image_query.partial.npz')
    np.savez_compressed(tmp,image=HNM_TRAIN_IMG,query=HNM_TRAIN_Q)
    os.replace(tmp,train_cache)
    train_cache_meta.write_text(json.dumps({'signature':sig,'rows':len(HNM_TRAIN_IMG)},indent=2))
assert HNM_TRAIN_IMG.shape==HNM_TRAIN_Q.shape==(len(TRAIN_LABELS),128)
assert np.isfinite(HNM_TRAIN_IMG).all() and np.isfinite(HNM_TRAIN_Q).all()

HNM_CKPT,HNM_HISTORY=hm.fit(HNM_TRAIN_IMG,HNM_TRAIN_Q,TRAIN_LABELS,TRAIN_DOMAINS,
    HNM_DIR,HNM_SIG,epochs=HNM_EPOCHS,batch=HNM_BATCH,lr=HNM_LR,force=HNM_FORCE_RETRAIN)
print('[53 PASS] TRAIN-only checkpoint:',HNM_CKPT)
print('Best holdout TRAIN-domain proxy:',max((r['holdout_r1'] for r in HNM_HISTORY),default=None))


# ======================================================================

#@title 54) HNM — Resume 33k gallery encoding + obtain attribute-query distance
import json,hashlib,os,numpy as np
HNM_VAL_FILE=HNM_DIR/'hnm_val_retrieval.npz'
HNM_VAL_META=HNM_DIR/'hnm_val_retrieval.json'
HNM_OUTPUT_SIG=sp.digest_lines([HNM_SIG,sha256_stream(HNM_CKPT),manifest['gallery_paths_sha256'],
                                 hashlib.sha256(Q[SELECT_IDS].astype('float32').tobytes()).hexdigest()])
use_cache=HNM_VAL_FILE.is_file() and HNM_VAL_META.is_file() and json.loads(HNM_VAL_META.read_text()).get('signature')==HNM_OUTPUT_SIG
if use_cache:
    with np.load(HNM_VAL_FILE,allow_pickle=False) as a:
        HNM_VAL_IMAGE=np.asarray(a['image'],np.float32)
        HNM_VAL_QUERY=np.asarray(a['query'],np.float32)
    print('[CACHE HIT] new representation:',HNM_VAL_IMAGE.shape,HNM_VAL_QUERY.shape)
else:
    model,device,ck=hm.load(HNM_CKPT)
    assert ck['signature']==HNM_SIG
    HNM_VAL_IMAGE=hm.encode(model,DENSE_IMG,'image',device,batch=1024)
    HNM_VAL_QUERY=hm.encode(model,DENSE_Q_EMBED,'query',device,batch=512)
    tmp=HNM_VAL_FILE.with_name('hnm_val_retrieval.partial.npz')
    np.savez_compressed(tmp,image=HNM_VAL_IMAGE,query=HNM_VAL_QUERY)
    os.replace(tmp,HNM_VAL_FILE)
    HNM_VAL_META.write_text(json.dumps({'signature':HNM_OUTPUT_SIG,'gallery':len(HNM_VAL_IMAGE),
       'queries':len(HNM_VAL_QUERY),'checkpoint':str(HNM_CKPT)},indent=2))
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
assert HNM_VAL_IMAGE.shape==(len(VAL_NAMES),128)
assert HNM_VAL_QUERY.shape==(len(SELECT_IDS),128)
HNM_DIST=hm.candidate_distance(HNM_VAL_QUERY,HNM_VAL_IMAGE,batch=32)
assert HNM_DIST.shape==BASE_DIST.shape and np.isfinite(HNM_DIST).all()
print('[54 PASS] New semantic distances:',HNM_DIST.shape)


# ======================================================================

#@title 55) HNM — FIT-only model selection against baseline AND DenseQuery champion
import numpy as np,pandas as pd,json
from tqdm.auto import tqdm

# Three options: teacher DenseQuery; student HNM; 50:50 teacher/student semantic.
# Selection is from FIT only; AUDIT does not influence parameters.
N_FIT_HNM=len(FIT_IDS)
HNM_REF_NAME='DenseQuery(alpha=0.05)'
def metrics3(scores,ids):
    m=sp.semantic_metrics(scores,ids,GIDS,Q[ids],VAL_LABELS)
    return {k:float(m[k]) for k in ('mADM','mAP','Rank-1')}
B=np.asarray(BASE_DIST[:N_FIT_HNM],np.float32)
TEACH=np.asarray(DENSE_SEM[:N_FIT_HNM],np.float32)
STUD=np.asarray(HNM_DIST[:N_FIT_HNM],np.float32)
HNM_REF_FIT=rr.fuse(B,TEACH,TEACH,alpha=0.05,beta=0.)
HNM_FIT_BASE=metrics3(B,FIT_IDS);HNM_FIT_REF=metrics3(HNM_REF_FIT,FIT_IDS)
pos_counts=np.bincount(GIDS,minlength=len(Q))[FIT_IDS]
single=pos_counts==1
ref_top=(GIDS[np.argmin(HNM_REF_FIT,axis=1)]==FIT_IDS)
ref_single=float(ref_top[single].mean()) if single.any() else None

trials=[]; semantic_modes=('new','teacher_new_half')
for mode in tqdm(semantic_modes,desc='55 / FIT representation modes'):
    sem=STUD if mode=='new' else 0.5*STUD+0.5*TEACH
    for alpha in (0.02,0.035,0.05,0.07,0.10):
        candidate=rr.fuse(B,sem,sem,alpha=float(alpha),beta=0.)
        m=metrics3(candidate,FIT_IDS)
        top=(GIDS[np.argmin(candidate,axis=1)]==FIT_IDS)
        one=float(top[single].mean()) if single.any() else None
        good=(m['mADM']>=HNM_FIT_REF['mADM']+0.001 and
              m['mADM']>=HNM_FIT_BASE['mADM']+0.003 and
              m['mAP']>=HNM_FIT_REF['mAP']-0.002 and
              m['Rank-1']>=HNM_FIT_REF['Rank-1']-0.0042 and
              (one is None or one>=ref_single))
        trials.append({'mode':mode,'alpha':float(alpha),**m,
             'gain_mADM_over_teacher':float(m['mADM']-HNM_FIT_REF['mADM']),
             'single_R1':one,'fit_pass':bool(good)})
HNM_FIT_TABLE=pd.DataFrame(trials).sort_values(['mADM','mAP'],ascending=False).reset_index(drop=True)
HNM_FIT_TABLE.to_csv(HNM_DIR/'hnm_fit_trials.csv',index=False)
allowed=HNM_FIT_TABLE[HNM_FIT_TABLE.fit_pass]
HNM_FIT_PASS=bool(len(allowed))
HNM_CHOICE=(allowed.iloc[0] if HNM_FIT_PASS else HNM_FIT_TABLE.iloc[0]).to_dict()
(HNM_DIR/'hnm_fit_choice.json').write_text(json.dumps({'teacher_fit':HNM_FIT_REF,'baseline_fit':HNM_FIT_BASE,
    'selected':HNM_CHOICE,'fit_pass':HNM_FIT_PASS},indent=2),encoding='utf-8')
print('FIT original:',HNM_FIT_BASE)
print('FIT DenseQuery champion:',HNM_FIT_REF)
print(HNM_FIT_TABLE.head(10).to_string(index=False))
print('SELECTED:',HNM_CHOICE,'FIT_PASS:',HNM_FIT_PASS)


# ======================================================================

#@title 56) HNM — Public AUDIT exploratory comparison + single-positive/domain guards
import numpy as np,pandas as pd,json
from tqdm.auto import tqdm
from abpr.model import track2_metrics
from abpr.query_aware_ranking import retrieval_metrics

N=N_FIT_HNM
AUD_B=np.asarray(BASE_DIST[N:],np.float32)
AUD_T=np.asarray(DENSE_SEM[N:],np.float32)
AUD_S=np.asarray(HNM_DIST[N:],np.float32)
HNM_REF_AUD=rr.fuse(AUD_B,AUD_T,AUD_T,alpha=.05,beta=0.)
sem=AUD_S if HNM_CHOICE['mode']=='new' else .5*AUD_S+.5*AUD_T
HNM_CAND_AUD=rr.fuse(AUD_B,sem,sem,alpha=float(HNM_CHOICE['alpha']),beta=0.)
orig=metrics3(AUD_B,AUDIT_IDS);teacher=metrics3(HNM_REF_AUD,AUDIT_IDS);candidate=metrics3(HNM_CAND_AUD,AUDIT_IDS)
num_pos=np.bincount(GIDS,minlength=len(Q))[AUDIT_IDS]
one=num_pos==1
r1_ref=(GIDS[np.argmin(HNM_REF_AUD,axis=1)]==AUDIT_IDS)
r1_cand=(GIDS[np.argmin(HNM_CAND_AUD,axis=1)]==AUDIT_IDS)
ref_one=float(r1_ref[one].mean()) if one.any() else None
cand_one=float(r1_cand[one].mean()) if one.any() else None
rows=[]
for d,name in tqdm(enumerate(('Market1501','PA100k','PETA')),total=3,desc='56 / source domains'):
    mask=np.asarray([infer_domain_id(s)==d for s in VAL_NAMES]);domain_ids=AUDIT_IDS[np.isin(AUDIT_IDS,GIDS[mask])]
    if len(domain_ids)==0:continue
    qix=np.flatnonzero(np.isin(AUDIT_IDS,domain_ids))
    for label,mat in [('baseline',AUD_B),('teacher',HNM_REF_AUD),('candidate',HNM_CAND_AUD)]:
        sub=mat[qix][:,mask]
        mm=retrieval_metrics(sub,AUDIT_IDS[qix],GIDS[mask])
        mm.update(track2_metrics(sub,Q[AUDIT_IDS[qix]],VAL_LABELS[mask]))
        rows.append({'domain':name,'method':label,'queries':len(qix),
            **{k:float(mm[k]) for k in ('mADM','mAP','Rank-1')}})
HNM_DOMAIN=pd.DataFrame(rows)
HNM_DOMAIN.to_csv(HNM_DIR/'hnm_audit_domains.csv',index=False)
# Define the gate relative to the stronger existing DenseQuery champion.
check_domains=[]
for name in HNM_DOMAIN.domain.unique():
    t=HNM_DOMAIN[(HNM_DOMAIN.domain==name)&(HNM_DOMAIN.method=='teacher')].iloc[0]
    c=HNM_DOMAIN[(HNM_DOMAIN.domain==name)&(HNM_DOMAIN.method=='candidate')].iloc[0]
    check_domains.append((c.mADM>=t.mADM-.003) and (c.mAP>=t.mAP-.003))
accepted=bool(HNM_FIT_PASS and candidate['mADM']>=teacher['mADM']+.002 and
    candidate['mAP']>=teacher['mAP']-.002 and candidate['Rank-1']>=teacher['Rank-1'] and
    (ref_one is None or cand_one>=ref_one) and all(check_domains))

# Pairwise differences in AP and top-1, not an mADM confidence interval.
ref_ap,ref_r1=hm.per_query_ap(HNM_REF_AUD,AUDIT_IDS,GIDS)
new_ap,new_r1=hm.per_query_ap(HNM_CAND_AUD,AUDIT_IDS,GIDS)
rng=np.random.default_rng(22013)
idx=rng.integers(0,len(AUDIT_IDS),(500,len(AUDIT_IDS)))
d_ap=(new_ap-ref_ap);d_r1=(new_r1-ref_r1)
ci_ap=np.quantile(d_ap[idx].mean(axis=1),[.025,.975]).tolist()
ci_r1=np.quantile(d_r1[idx].mean(axis=1),[.025,.975]).tolist()
HNM_AUDIT_ACCEPTED=accepted
HNM_AUDIT_REPORT={
 'baseline':orig,'previous_champion':teacher,'new_candidate':candidate,
 'fit_chosen':HNM_CHOICE,'fit_pass':bool(HNM_FIT_PASS),
 'audit_gain_over_champion_mADM':float(candidate['mADM']-teacher['mADM']),
 'single_R1_champion':ref_one,'single_R1_candidate':cand_one,
 'bootstrap_mAP_delta_CI95':ci_ap,'bootstrap_R1_delta_CI95':ci_r1,
 'audit_accepted':accepted,'submission_ready':False,
 'note':'Public AUDIT reused in many earlier experiments; exploratory only. External SigLIP2 eligibility and an offline runtime remain unverified.'}
(HNM_DIR/'hnm_audit_report.json').write_text(json.dumps(HNM_AUDIT_REPORT,indent=2))
print('Original:',orig)
print('Previous champion:',teacher)
print('New candidate:',candidate)
print('Single-positive R1 champion -> candidate:',ref_one,'->',cand_one)
print(HNM_DOMAIN.to_string(index=False))
print('HNM_AUDIT_ACCEPTED:',accepted)
print('No Codabench submission ZIP is generated by these experiment cells.')


# ======================================================================

#@title 57) HNM — Export reproducible experiment report (NOT submission)
import json,zipfile,hashlib
from pathlib import Path
from tqdm.auto import tqdm
readme=HNM_DIR/'README_EXPERIMENT_ONLY.txt'
readme.write_text('ABPR HARD-NEGATIVE METRIC ADAPTATION | PUBLIC VALIDATION RESEARCH ONLY\n'
  'Model trained on TRAIN labels; FIT chosen; AUDIT exploratory and repeatedly examined earlier.\n'
  'Cache feature order is checked. Never use public VAL labels for runtime scoring.\n'
  'Not a Codabench package. Runtime offline weights & pretrained model eligibility are unverified.\n',encoding='utf-8')
items=[HNM_DIR/'abpr_hardnegative_metric_adapter.py',HNM_DIR/'metric_training_history.json',
       HNM_DIR/'hnm_fit_trials.csv',HNM_DIR/'hnm_fit_choice.json',
       HNM_DIR/'hnm_audit_report.json',HNM_DIR/'hnm_audit_domains.csv',readme]
missing=[str(p) for p in items if not p.is_file()]
assert not missing,missing
HNM_REPORT_ZIP=HNM_DIR/'ABPR_HardNegative_Experiment_Report.zip'
with zipfile.ZipFile(HNM_REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for item in tqdm(items,desc='57 / package reproducibility report',unit='file'):
        z.write(item,arcname=item.name)
print('[REPORT]',HNM_REPORT_ZIP)
print('COMPARE TO CHAMPION:',HNM_AUDIT_REPORT['previous_champion'],HNM_AUDIT_REPORT['new_candidate'])
print('RECOMMENDATION:', 'Eligible for OFFLINE-runtime review (not auto-submit)' if HNM_AUDIT_ACCEPTED else 'KEEP existing 0.5389 hidden baseline.')
