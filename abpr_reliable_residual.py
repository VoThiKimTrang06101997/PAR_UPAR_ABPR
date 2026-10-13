"""TRAIN-only reliability-calibrated residual attribute evidence for ABPR Track2.
No public/hidden labels are accepted by train(), infer() or scoring functions.
"""
from pathlib import Path
import hashlib, json, os, math
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

VERSION='residual-attribute-train-only-20261012'

class ResidualAttributeHead(nn.Module):
    def __init__(self,dim=128,hidden=96):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,40))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self,z):
        return self.net(z)

def _probs(logits):
    a=np.clip(np.asarray(logits,dtype=np.float64),-40,40)
    return (1/(1+np.exp(-a))).astype('float32')

def _save(path,payload):
    path=Path(path);tmp=path.with_suffix('.partial.pt')
    torch.save(payload,tmp);os.replace(tmp,path)

def train_residual(train_image,train_teacher_logits,train_labels,train_domains,out,signature,
                   epochs=12,batch_size=192,lr=0.0007,seed=927,force=False):
    X=np.asarray(train_image,dtype=np.float32)
    T=np.asarray(train_teacher_logits,dtype=np.float32)
    Y=np.asarray(train_labels,dtype=np.float32)
    D=np.asarray(train_domains,dtype=np.int64)
    assert X.ndim==2 and X.shape[1]==128 and X.shape[0]==T.shape[0]==Y.shape[0]==len(D)
    assert T.shape[1]==Y.shape[1]==40 and np.isfinite(X).all() and np.isfinite(T).all()
    assert np.isin(Y,[-1,0,1]).all() and np.isin(D,[0,1,2]).all()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config=dict(dim=128,hidden=96,seed=seed,batch=batch_size,epochs=epochs,lr=lr,wd=0.03)
    best_path=out/'residual_best.pt';last_path=out/'residual_latest.pt'
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    net=ResidualAttributeHead().to('cuda' if torch.cuda.is_available() else 'cpu')
    device=next(net.parameters()).device
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.03)
    rg=np.random.default_rng(seed)
    training=[];holdout=[]
    for d in (0,1,2):
        ids=np.flatnonzero(D==d);rg.shuffle(ids)
        if len(ids)<15:raise ValueError('TRAIN domain has <15 rows')
        nh=max(4,int(0.2*len(ids)))
        holdout.extend(ids[:nh]);training.extend(ids[nh:])
    training=np.asarray(training,dtype=np.int64)
    holdout=np.asarray(holdout,dtype=np.int64)
    weights=np.clip(((Y[training]==0).sum(0)+2)/((Y[training]==1).sum(0)+2),0.8,4.0).astype('float32')
    wt=torch.as_tensor(weights,device=device)
    history=[];best=float('inf');begin=0
    if last_path.exists() and not force:
        ck=torch.load(last_path,map_location='cpu',weights_only=False)
        if ck.get('signature')!=signature or ck.get('config')!=config:
            raise RuntimeError('Residual resume checkpoint config/signature mismatch. Do not silently mix runs.')
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        begin=int(ck['epoch'])+1;history=ck['history'];best=float(ck['best'])
        torch.set_rng_state(ck['torch_rng'])
        if torch.cuda.is_available() and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
        print('[RESUME] residual at epoch',begin+1,flush=True)
    groups=[training[D[training]==d] for d in (0,1,2)]
    Xdev=torch.from_numpy(X).to(device);Tdev=torch.from_numpy(T).to(device);Ydev=torch.from_numpy(Y).to(device)
    for epoch in range(begin,epochs):
        net.train();losses=[];rgen=np.random.default_rng(seed+epoch+104729)
        steps=max(1,math.ceil(len(training)/batch_size))
        for _ in tqdm(range(steps),desc=f'Residual epoch {epoch+1}/{epochs}',unit='batch',leave=False):
            take=[]
            for d,part in enumerate(groups):
                n=batch_size//3+int(d<batch_size%3)
                take.extend(rgen.choice(part,size=n,replace=True))
            ix=torch.from_numpy(np.asarray(take,dtype=np.int64)).to(device)
            y=Ydev[ix];t=Tdev[ix];r=net(Xdev[ix])
            known=((y==0)|(y==1))
            bce=nn.functional.binary_cross_entropy_with_logits(t+r,y.clamp(0,1),reduction='none')
            weighted=bce*torch.where(y==1,wt[None,:],1.)
            loss=(weighted*known).sum(0).div(known.sum(0).clamp_min(1)).mean()+0.07*r.square().mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),2.0);opt.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.inference_mode():
            o=[]
            for pos in range(0,len(holdout),512):
                ix=torch.from_numpy(holdout[pos:pos+512]).to(device)
                o.append((Tdev[ix]+net(Xdev[ix])).cpu().numpy())
        pred=np.concatenate(o);true=Y[holdout];mask=np.isin(true,[0,1])
        p=_probs(pred);brier=float(np.sum(((p-true.clip(0,1))**2)*mask)/max(mask.sum(),1))
        metric=brier+0.001*float(np.mean(losses))
        record={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'holdout_masked_brier':brier,'selection':metric}
        history.append(record)
        payload={'model':net.cpu().state_dict(),'optimizer':opt.state_dict(),'signature':signature,'config':config,
                 'epoch':epoch,'best':min(best,metric),'history':history,'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        net.to(device)
        if metric<best:
            best=metric;_save(best_path,payload)
        _save(last_path,payload)
        (out/'residual_training_history.json').write_text(json.dumps(history,indent=2))
        print('RESIDUAL TRAIN',record,flush=True)
    if not best_path.is_file():raise RuntimeError('No trained residual checkpoint')
    return best_path,history,holdout

@torch.inference_mode()
def residual_delta(checkpoint,image_features,batch=1024):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ResidualAttributeHead().to('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(ck['model']);model.eval()
    device=next(model.parameters()).device
    out=[]
    for st in tqdm(range(0,len(image_features),batch),desc='Residual inference',unit='batch'):
        x=torch.as_tensor(np.asarray(image_features[st:st+batch],dtype=np.float32),device=device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out,axis=0).astype('float32')

def calibration_policy(teacher_logits,residual_logits,labels,domains,holdout,attribute_names):
    t=np.asarray(teacher_logits,dtype='float32')[holdout]
    r=np.asarray(residual_logits,dtype='float32')[holdout]
    y=np.asarray(labels,dtype='float32')[holdout]
    d=np.asarray(domains)[holdout]
    choices=[0.0,0.25,0.50,1.0]
    mixes=[];weights=[];rows=[]
    for a in range(40):
        known=np.isin(y[:,a],[0,1]);positive=int((y[:,a]==1).sum());negative=int((y[:,a]==0).sum())
        if positive<8 or negative<8:
            alpha=0.;trust=0.45;losses=[]
        else:
            losses=[]
            for m in choices:
                v=_probs(t[:,a]+m*r[:,a]);err=(v-y[:,a])**2
                # Cross-domain uniform Brier on TRAIN holdout only
                dm=[float(err[known&(d==j)].mean()) for j in (0,1,2) if (known&(d==j)).sum()>=8]
                losses.append(float(np.mean(dm) if dm else err[known].mean()))
            best=int(np.argmin(losses));alpha=choices[best]
            base=float(losses[0]);improvement=(base-min(losses))/max(base,1e-6)
            trust=float(np.clip(0.65+1.5*improvement,0.65,1.2))
        mixes.append(alpha);weights.append(trust)
        rows.append(dict(attribute=str(attribute_names[a]),positive=positive,negative=negative,
                         residual_blend=alpha,reliability_weight=trust,
                         holdout_brier_base=float(losses[0]) if losses else None,
                         holdout_brier_selected=float(min(losses)) if losses else None))
    return np.asarray(mixes,'float32'),np.asarray(weights,'float32'),rows

def attribute_nll(queries,logits,attr_weights,chunk=32):
    q=np.asarray(queries,'float32');l=np.asarray(logits,'float32')
    w=np.asarray(attr_weights,'float32')
    assert q.ndim==2 and q.shape[1]==40 and l.ndim==2 and l.shape[1]==40 and w.shape==(40,)
    assert np.isin(q,[-1,0,1]).all() and np.isfinite(l).all()
    # nll for value 1 / value 0
    lp=np.logaddexp(0.,-l);ln=np.logaddexp(0.,l)
    out=np.empty((len(q),len(l)),dtype='float32')
    for s in tqdm(range(0,len(q),chunk),desc='TRAIN-calibrated attribute NLL',unit='query batch'):
        sub=q[s:s+chunk];p=(sub==1)*w;n=(sub==0)*w
        denom=np.maximum((p+n).sum(1,keepdims=True),1e-8)
        out[s:s+len(sub)]=((p@lp.T+n@ln.T)/denom).astype('float32')
    return out

def zscore(arr):
    x=np.asarray(arr,'float32')
    return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-5)

def fuse(base,semantic,attribute,alpha,beta,mode='global',topk=200):
    b=np.asarray(base,dtype='float32');s=np.asarray(semantic,dtype='float32');a=np.asarray(attribute,dtype='float32')
    assert b.shape==s.shape==a.shape and np.isfinite(b).all() and alpha>=0 and beta>=0 and alpha+beta<=1
    if alpha==0 and beta==0:return b.copy()
    if mode=='global':
        return ((1-alpha-beta)*zscore(b)+alpha*zscore(s)+beta*zscore(a)).astype('float32')
    if mode!='topk':raise ValueError(mode)
    n,k=b.shape;kk=min(int(topk),k)
    if kk<2:return b.copy()
    out=b.copy()
    # Keep all gallery members outside baseline topK at exactly original distance;
    # change ordering only among original TopK by permuting those exact distances.
    for i in range(n):
        top=np.argpartition(b[i],kk-1)[:kk]
        ord0=np.argsort(b[i,top],kind='stable');idx=top[ord0]
        ss=s[i,idx];aa=a[i,idx]
        v=(1-alpha-beta)*zscore(b[i,idx][None,:])[0]+alpha*zscore(ss[None,:])[0]+beta*zscore(aa[None,:])[0]
        ord1=np.argsort(v,kind='stable')
        out[i,idx[ord1]]=b[i,idx]
    return out

def per_query_ap(distance,ids,gallery_ids):
    mat=np.asarray(distance);ids=np.asarray(ids);g=np.asarray(gallery_ids)
    ap=np.zeros(len(ids),dtype='float32');r1=np.zeros(len(ids),dtype='float32')
    for i in tqdm(range(len(ids)),desc='Paired AP / Rank-1',unit='query'):
        ix=np.argsort(mat[i],kind='stable');positive=(g[ix]==ids[i]);n=int(positive.sum())
        r1[i]=float(positive[0])
        if n:
            positions=np.flatnonzero(positive)
            ap[i]=float((np.arange(1,n+1)/(positions+1)).mean())
    return ap,r1
