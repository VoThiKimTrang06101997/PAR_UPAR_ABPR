"""TRAIN-only dense-query, single-positive-aware representation learning.

Input: frozen SigLIP2 image vectors plus official TRAIN attribute labels.
No VAL labels/semantic IDs are used in training or scoring.
Public FIT/AUDIT labels are read by the notebook exclusively for evaluation.
"""
from __future__ import annotations
import json, math, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class DenseQueryNet(nn.Module):
    def __init__(self, input_dim, hidden=256, embed=128, drop=0.10):
        super().__init__()
        self.image = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden),
            nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, embed))
        self.image_skip = nn.Linear(input_dim, embed, bias=False)
        self.attributes = nn.Sequential(nn.LayerNorm(embed), nn.Linear(embed,40))
        self.query = nn.Sequential(nn.Linear(80, hidden), nn.GELU(),
            nn.Linear(hidden, embed))
        self.embed = embed

    def image_forward(self, x):
        z = F.normalize(self.image(x) + 0.15 * self.image_skip(x), dim=-1)
        return z, self.attributes(z)

    def query_forward(self,q):
        valid = (q==0)|(q==1)
        signed = torch.where(valid, q*2.-1., torch.zeros_like(q))
        rep = torch.cat([signed, valid.float()], dim=-1)
        return F.normalize(self.query(rep),dim=-1)


def sample_dense_queries(y, epoch, epochs, rng):
    """Sample known TRAIN attributes, always keep unknown as -1.
    Second half shifts toward dense queries to reduce train-test mismatch.
    """
    y = np.asarray(y, np.float32)
    q = np.full_like(y,-1.0)
    for i in range(len(y)):
        valid = np.flatnonzero(np.isin(y[i],[0,1]))
        if len(valid)==0: continue
        low = min(len(valid), 12 if epoch<epochs//2 else 24)
        n = int(rng.integers(low, len(valid)+1))
        chosen = rng.choice(valid,size=n,replace=False)
        q[i,chosen] = y[i,chosen]
    return q


def safe_multi_positive_loss(z,qz,q,y,temp=0.13,margin=0.12):
    """Exact known-attribute matches are positive, explicit conflicts negative;
    label-unknown cross-pairs are neither negative nor positive.
    Each paired query/image is always a positive by construction.
    """
    known_q = (q==0)|(q==1)
    known_y = (y==0)|(y==1)
    contradiction = ((q[:,None,:]!=y[None,:,:]) & known_q[:,None,:] & known_y[None,:,:]).any(-1)
    full_known = (known_q[:,None,:] & (~known_y[None,:,:])).any(-1)
    positive = (~contradiction) & (~full_known) & known_q.any(1)[:,None]
    n = len(y)
    positive[torch.arange(n,device=y.device),torch.arange(n,device=y.device)] = True
    negative = contradiction
    eligible = positive|negative
    logits = (qz@z.T)/temp
    mask_p = logits.masked_fill(~positive,-1e4)
    mask_e = logits.masked_fill(~eligible,-1e4)
    row_loss = torch.logsumexp(mask_e,dim=1)-torch.logsumexp(mask_p,dim=1)
    valid = negative.any(dim=1)
    mp = row_loss[valid].mean() if valid.any() else logits.sum()*0.
    best_pos = mask_p.amax(dim=1)
    hard_neg = logits.masked_fill(~negative,-1e4).amax(dim=1)
    hard = F.relu(margin + hard_neg[valid]*temp - best_pos[valid]*temp).mean() if valid.any() else logits.sum()*0.
    # A second objective targets the paired image: encourages discriminative top-1.
    # Other valid multi-positives remain legal via the multi-positive term above.
    diagonal = logits.diagonal()
    neg_lse = torch.logsumexp(logits.masked_fill(~negative,-1e4),dim=1)
    pair = F.softplus(neg_lse[valid]-diagonal[valid]).mean() if valid.any() else logits.sum()*0.
    return mp,hard,pair


def masked_attribute_loss(logits,y,pos_weights):
    valid = (y==0)|(y==1)
    targets = y.clamp(0,1)
    bce = F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    reweight = torch.where(targets==1,pos_weights[None,:],1.0)
    # per attribute normalization for label-frequency robustness
    col_denom = valid.sum(dim=0).clamp_min(1)
    return ((bce*reweight*valid.float()).sum(0)/col_denom).mean()


def prototype_alignment(z,y,dom):
    """Light positive-attribute cross-domain center alignment on TRAIN batch.
    Prevents excessive domain-specific geometry; never uses public VAL.
    """
    terms=[]
    for a in range(40):
        centroids=[]
        for d in (0,1,2):
            mask=(y[:,a]==1)&(dom==d)
            if int(mask.sum())>=2:centroids.append(z[mask].mean(0))
        for i in range(len(centroids)):
            for j in range(i+1,len(centroids)):
                terms.append((centroids[i]-centroids[j]).square().mean())
    return torch.stack(terms).mean() if terms else z.sum()*0.


def holdout_eval(model,X,Y,device,batch=256,topk=None):
    """TRAIN-only holdout retrieval; each image is positive for its own query.
    Reports a *proxy* metric, not UPAR 2027 public/hidden score.
    """
    model.eval(); feats=[];query=[];num=0.;den=0.
    with torch.inference_mode():
        for i in range(0,len(X),batch):
            xb=torch.tensor(np.asarray(X[i:i+batch],np.float32),device=device)
            yb=torch.tensor(np.asarray(Y[i:i+batch],np.float32),device=device)
            zb,lb=model.image_forward(xb)
            qb=model.query_forward(yb)
            known=((yb==0)|(yb==1))
            num+=float((F.binary_cross_entropy_with_logits(lb,yb.clamp(0,1),reduction='none')*known).sum())
            den+=float(known.sum())
            feats.append(zb.cpu());query.append(qb.cpu())
    z=torch.cat(feats);q=torch.cat(query)
    diag=(q*z).sum(-1)
    similarity=q@z.T
    ranks = (similarity>diag[:,None]+1e-8).sum(1).numpy()+1
    # Self-image retrieval in a TRAIN-only heldout gallery. Important caveat:
    # 'Rank-1' here does not equal challenge retrieval Rank-1.
    return {'holdout_bce':num/max(1,den),'holdout_self_R1':float((ranks==1).mean()),
            'holdout_self_mrr':float(np.mean(1./ranks)),'holdout_n':int(len(Y))}


def save_ck(path,model,opt,epoch,best,cfg,signature,history,rng):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.pt')
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':int(epoch),
        'best':float(best),'config':cfg,'signature':signature,'history':history,
        'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},tmp)
    os.replace(tmp,path)


def fit(X,Y,D,out,signature,epochs=12,batch_size=192,lr=0.0005,seed=2027,force=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    X=np.asarray(X,np.float32); Y=np.asarray(Y,np.float32); D=np.asarray(D,np.int64)
    assert X.ndim==2 and Y.shape==(len(X),40) and D.shape==(len(X),)
    assert np.isfinite(X).all() and np.isin(Y,[-1,0,1]).all() and np.isin(D,[0,1,2]).all()
    assert all((D==d).sum()>=15 for d in (0,1,2))
    assert (Y!=-1).sum()>100
    cfg={'in':X.shape[1],'hidden':256,'embed':128,'drop':0.1,'batch':int(batch_size),
        'epochs':int(epochs),'lr':float(lr),'seed':int(seed),'holdout':0.2,
        'ranking_weight':0.35,'hard_weight':0.12,'pair_weight':0.15,'domain_weight':0.02}
    seed_everything(seed); rng=np.random.default_rng(seed)
    training=[];holdout=[]
    for d in (0,1,2):
        ids=np.where(D==d)[0];rng.shuffle(ids)
        n=max(2,int(round(0.2*len(ids))))
        holdout.extend(ids[:n]);training.extend(ids[n:])
    training=np.array(training,dtype=np.int64);holdout=np.array(holdout,dtype=np.int64)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=DenseQueryNet(X.shape[1]).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.02)
    last=out/'densequery_latest.pt';best_file=out/'densequery_best.pt'
    start=0; best=-float('inf');history=[]
    if last.is_file() and not force:
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck.get('signature')!=signature or ck.get('config')!=cfg:
            raise RuntimeError('Resume checkpoint signature/config mismatch. Use a new folder, do not overwrite original.')
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start=int(ck['epoch'])+1;best=float(ck['best']);history=ck['history']
        rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'])
        if device=='cuda' and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
        print('[RESUME] DenseQuery next epoch',start+1,flush=True)
    known=np.isin(Y[training],[0,1]);pos=(Y[training]==1).sum(0)
    weights=np.clip((known.sum(0)-pos+1)/np.maximum(1,pos+1),0.8,4.0).astype(np.float32)
    pw=torch.from_numpy(weights).to(device)
    train_buckets=[training[D[training]==d] for d in (0,1,2)]
    steps=max(1,math.ceil(len(training)/batch_size))
    if start>=epochs and best_file.is_file():return best_file,history
    for ep in range(start,epochs):
        net.train();logs=[]
        for _ in tqdm(range(steps),desc=f'DenseQuery epoch {ep+1}/{epochs}',leave=False):
            selections=[]
            for d,b in enumerate(train_buckets):
                count=batch_size//3+int(d<batch_size%3)
                selections.extend(rng.choice(b,size=count,replace=True))
            inds=np.asarray(selections,dtype=np.int64);rng.shuffle(inds)
            xb=torch.tensor(X[inds],device=device);yb=torch.tensor(Y[inds],device=device)
            db=torch.tensor(D[inds],device=device)
            q_np=sample_dense_queries(Y[inds],ep,epochs,rng)
            qb=torch.tensor(q_np,device=device)
            z,logits=net.image_forward(xb);qz=net.query_forward(qb)
            attr=masked_attribute_loss(logits,yb,pw)
            mp,hard,pair=safe_multi_positive_loss(z,qz,qb,yb)
            dom=prototype_alignment(z,yb,db)
            loss=attr+0.35*mp+0.12*hard+0.15*pair+0.02*dom
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.0);opt.step()
            logs.append((float(loss.detach()),float(attr.detach()),float(mp.detach()),float(pair.detach())))
        v=holdout_eval(net,X[holdout],Y[holdout],device)
        # Holdout-only selection. Weighted toward recovering a single exact paired image.
        select=v['holdout_self_R1'] + 0.15*v['holdout_self_mrr']-0.025*v['holdout_bce']
        stats={'epoch':ep+1,'train_loss':float(np.mean(logs,axis=0)[0]),
               'train_attr':float(np.mean(logs,axis=0)[1]),'train_mp':float(np.mean(logs,axis=0)[2]),
               'train_pair':float(np.mean(logs,axis=0)[3]),**v,'holdout_selection':float(select)}
        history.append(stats)
        if select>best:
            best=select;save_ck(best_file,net,opt,ep,best,cfg,signature,history,rng)
        save_ck(last,net,opt,ep,best,cfg,signature,history,rng)
        (out/'densequery_training_history.json').write_text(json.dumps(history,indent=2))
        print('[TRAIN-ONLY]',stats,flush=True)
    return best_file,history


def load_model(path,device=None):
    ck=torch.load(Path(path),map_location='cpu',weights_only=False)
    cfg=ck['config']; dev=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=DenseQueryNet(cfg['in'],cfg['hidden'],cfg['embed'],cfg['drop']).to(dev)
    model.load_state_dict(ck['model']);model.eval()
    return model,dev,ck

@torch.inference_mode()
def image_outputs(model,features,device,chunk=1024):
    zs=[];ls=[]
    for s in tqdm(range(0,len(features),chunk),desc='DenseQuery VAL features',unit='batch'):
        x=torch.tensor(np.asarray(features[s:s+chunk],np.float32),device=device)
        z,l=model.image_forward(x);zs.append(z.cpu().numpy());ls.append(l.cpu().numpy())
    return np.concatenate(zs).astype('float32'),np.concatenate(ls).astype('float32')

@torch.inference_mode()
def query_outputs(model,queries,device,chunk=512):
    zs=[]
    for s in range(0,len(queries),chunk):
        q=torch.tensor(np.asarray(queries[s:s+chunk],np.float32),device=device)
        zs.append(model.query_forward(q).cpu().numpy())
    return np.concatenate(zs).astype('float32')


def distances(query_z,gallery_z,logits,queries,weights,chunk=24):
    qs=np.asarray(queries,np.float32);qz=np.asarray(query_z,np.float32)
    gz=np.asarray(gallery_z,np.float32); l=np.asarray(logits,np.float32)
    w=np.asarray(weights,np.float32)
    assert qs.shape[1]==l.shape[1]==len(w)==40
    assert len(qs)==len(qz) and len(gz)==len(l)
    lp=-np.logaddexp(0,-l);ln=-np.logaddexp(0,l)
    semantic=[];evidence=[]
    for s in tqdm(range(0,len(qs),chunk),desc='DenseQuery retrieval',unit='query batch'):
        q=qs[s:s+chunk]; pos=(q==1)*w;neg=(q==0)*w
        denom=np.maximum((pos+neg).sum(1,keepdims=True),1e-8)
        attr=-(pos@lp.T+neg@ln.T)/denom
        cosine=-(qz[s:s+chunk]@gz.T)
        semantic.append(cosine.astype('float32'));evidence.append(attr.astype('float32'))
    return np.concatenate(semantic),np.concatenate(evidence)


def row_std(x):
    x=np.asarray(x,np.float32)
    return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-5)


def fusion(base,embed,evidence,method,alpha):
    base=np.asarray(base,np.float32)
    if alpha==0: return base.copy()
    if method=='semantic':new=embed
    elif method=='evidence':new=evidence
    elif method=='hybrid':new=0.50*row_std(embed)+0.50*row_std(evidence)
    else:raise KeyError(method)
    return ((1-alpha)*row_std(base)+alpha*row_std(new)).astype('float32')
