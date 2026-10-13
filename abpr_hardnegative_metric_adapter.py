"""Frozen DenseQuery feature-space residual metric adapter for ABPR Track 2.
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
