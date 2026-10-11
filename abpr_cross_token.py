"""Cross-Attribute Token Attention, frozen DINO representations, TRAIN-only supervised learning.
Research experiment for UPAR27 Track2. Never uses VAL labels to train or score.
"""
import os, json, math, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


def digest(*items):
    h=hashlib.sha256()
    for item in items:
        if isinstance(item,np.ndarray):
            a=np.ascontiguousarray(item)
            h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
        else:h.update(str(item).encode())
        h.update(b'\n')
    return h.hexdigest()


class CrossTokenHead(nn.Module):
    """Residual attribute-token attention model. 40 DINO attribute tokens, 192 channels."""
    def __init__(self,nattr=40,dim=192,dropout=.12,scale=.35):
        super().__init__()
        self.scale=float(scale)
        self.norm=nn.LayerNorm(dim)
        self.pos=nn.Parameter(torch.randn(nattr,dim)*.012)
        enc=nn.TransformerEncoderLayer(d_model=dim,nhead=4,dim_feedforward=384,
                                       dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.attn=nn.TransformerEncoder(enc,num_layers=1,enable_nested_tensor=False)
        self.outnorm=nn.LayerNorm(dim)
        self.attr_readout=nn.Parameter(torch.randn(nattr,dim)*.012)
        self.bias=nn.Parameter(torch.zeros(nattr))
        self.dropout=nn.Dropout(dropout)

    def forward(self,tokens,old_logits):
        if tokens.ndim!=3 or tokens.shape[1:]!=(40,192):
            raise ValueError(f'Expected [B,40,192] features, got {tuple(tokens.shape)}')
        h=self.norm(tokens.float())+self.pos[None]
        h=self.attn(h)
        h=self.outnorm(self.dropout(h))
        delta=(h*self.attr_readout[None]).sum(-1)/math.sqrt(h.shape[-1])+self.bias
        return old_logits.float()+self.scale*delta


def _loss(out,y,pos_weights,domains,query_weight=.16,anchor_weight=.02,old=None):
    v=(y==0)|(y==1)
    target=y.clamp(0,1)
    ww=torch.where(target==1,pos_weights[None,:],1.0)
    cell=F.binary_cross_entropy_with_logits(out,target,reduction='none')*v*ww
    per=(cell.sum(1)/v.sum(1).clamp_min(1))
    domain_loss=torch.stack([per[domains==d].mean() for d in (0,1,2) if (domains==d).any()])
    loss_cls=domain_loss.mean()+.08*domain_loss.std(unbiased=False)
    # Full 40-attr query from TRAIN labels (75%); sparse 2-10 attributes otherwise.
    full=torch.rand((len(y),1),device=y.device)<.75
    keep=v&(full|(torch.rand_like(y)<.2))
    shortage=keep.sum(1)<2
    if shortage.any():
        r=torch.rand_like(y).masked_fill(~v,-1.)
        idx=torch.topk(r,k=min(10,y.shape[1]),dim=1).indices
        fill=torch.zeros_like(v).scatter(1,idx,True)&v
        keep=torch.where(shortage[:,None],fill,keep)
    p=((y==1)&keep).float(); n=((y==0)&keep).float(); denom=(p+n).sum(1).clamp_min(1)
    scores=(p@F.logsigmoid(out).T+n@F.logsigmoid(-out).T)/denom[:,None]
    bad=((p@((y==0)&v).float().T+n@((y==1)&v).float().T)>0)
    bad.fill_diagonal_(False)
    possible=bad.any(1)
    if possible.any():
        hardest=scores.masked_fill(~bad,-1e6).max(1).values
        loss_rank=F.relu(.12+hardest[possible]-scores.diagonal()[possible]).mean()
    else:loss_rank=out.sum()*0.0
    anchor=F.smooth_l1_loss(out,old) if old is not None else out.sum()*0
    return loss_cls+float(query_weight)*loss_rank+float(anchor_weight)*anchor,loss_cls,loss_rank


def train(tokens,logits,labels,domains,path,signature,epochs=9,batch=96,lr=.00045,
          steps_per_epoch=0,seed=73,patience=3):
    """Resumable, strict signature. Per-domain TRAIN-only stratified holdout. Never VAL."""
    labels=np.asarray(labels,np.float32); domains=np.asarray(domains,np.int64)
    n=len(labels)
    assert np.shape(tokens)==(n,40,192) and np.shape(logits)==(n,40)
    assert set(np.unique(domains))=={0,1,2}
    assert np.isin(labels,[-1,0,1]).all()
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    groups=[];hold=[]
    for d in range(3):
        ix=np.flatnonzero(domains==d);rng.shuffle(ix)
        if len(ix)<100:raise RuntimeError(f'TRAIN domain {d} has too few examples')
        nh=max(24,int(.12*len(ix)))
        groups.append(ix[nh:]);hold.extend(ix[:nh])
    hold=np.asarray(hold,np.int64)
    obs=(labels==0)|(labels==1)
    tr=np.concatenate(groups)
    pos=((labels[tr]==1)&obs[tr]).sum(0);neg=((labels[tr]==0)&obs[tr]).sum(0)
    pw=np.clip(np.sqrt((neg+1)/(pos+1)),.7,2.5).astype(np.float32)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CrossTokenHead().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.015)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=lr*.2)
    wt=torch.as_tensor(pw,device=device)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    best=path.with_name('cross_token_best.pt')
    config={'sig':signature,'epochs':epochs,'batch':batch,'lr':lr,'steps':steps_per_epoch,'seed':seed,
            'arch':'CATA_1layer_v1','train_samples':n,'patience':patience}
    start=0;best_loss=float('inf');hist=[];bad=0
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['config']!=config:raise RuntimeError('Training config/data changed: choose a new run directory')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['next_epoch'];best_loss=ck['best_holdout'];hist=ck['history'];bad=ck['bad']
        rng.bit_generator.state=ck['rng_np'];torch.set_rng_state(ck['rng_torch'])
    def load(ix):
        # Efficient memmap indexing; only batch enters GPU memory.
        t=torch.from_numpy(np.asarray(tokens[ix],dtype=np.float32).copy()).to(device)
        z=torch.from_numpy(np.asarray(logits[ix],dtype=np.float32).copy()).to(device)
        y=torch.from_numpy(np.asarray(labels[ix],dtype=np.float32).copy()).to(device)
        d=torch.from_numpy(np.asarray(domains[ix],dtype=np.int64).copy()).to(device)
        return t,z,y,d
    nsteps=steps_per_epoch or max(1,math.ceil(len(tr)/batch))
    for epoch in tqdm(range(start,epochs),desc='CATA training (TRAIN only)',unit='epoch'):
        model.train();train_losses=[]
        for _ in tqdm(range(nsteps),desc=f'Epoch {epoch+1} minibatches',unit='batch',leave=False):
            ix=np.concatenate([rng.choice(g,size=batch//3+int(d<batch%3),replace=True)
                               for d,g in enumerate(groups)])
            t,z,y,d=load(ix)
            out=model(t,z)
            loss,cls,rnk=_loss(out,y,wt,d,old=z)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.0);optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval(); total=0.; total_pos=0.; count=0
        with torch.inference_mode():
            for i in range(0,len(hold),192):
                t,z,y,d=load(hold[i:i+192]);out=model(t,z)
                val=(y==0)|(y==1)
                err=F.binary_cross_entropy_with_logits(out,y.clamp(0,1),reduction='none')*val
                total+=float(err.sum());count+=int(val.sum())
        holdout=total/max(1,count); sched.step()
        row={'epoch':epoch+1,'loss':float(np.mean(train_losses)),'train_holdout_BCE':holdout}
        hist.append(row)
        if holdout<best_loss-0.0001:
            best_loss=holdout;bad=0
            torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'config':config,'holdout_BCE':holdout},best)
        else:bad+=1
        tmp=path.with_suffix('.tmp')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(), 'scheduler':sched.state_dict(),
                    'next_epoch':epoch+1,'history':hist,'best_holdout':best_loss,'bad':bad,'config':config,
                    'rng_np':rng.bit_generator.state,'rng_torch':torch.get_rng_state()},tmp)
        os.replace(tmp,path)
        print(f'[CATA epoch {epoch+1}] TRAIN={row["loss"]:.5f} TRAIN-holdout BCE={holdout:.5f} best={best_loss:.5f}',flush=True)
        if bad>=patience:print('[CATA] Early stopping based on TRAIN-only holdout');break
    if not best.is_file():raise RuntimeError('No best model checkpoint found')
    ck=torch.load(best,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']);model.eval()
    return model,hist,best


@torch.inference_mode()
def infer(model,tokens,logits,outfile,signature,batch=256):
    """Ordered and resume-safe full gallery logits cache. Never uses gallery labels."""
    outfile=Path(outfile);outfile.parent.mkdir(parents=True,exist_ok=True)
    meta=outfile.with_suffix('.json');n=len(logits)
    if outfile.is_file() and meta.is_file():
        info=json.loads(meta.read_text())
        if info.get('sig')!=signature or info.get('n')!=n:raise RuntimeError('Old cache differs; use separate run directory')
        arr=np.load(outfile,mmap_mode='r+');done=int(info['completed'])
        if arr.shape!=(n,40) or done>n or done<0:raise RuntimeError('Invalid resume state')
    elif outfile.exists() or meta.exists():raise RuntimeError('Orphan feature/meta file; investigate')
    else:
        arr=np.lib.format.open_memmap(outfile,mode='w+',dtype=np.float32,shape=(n,40));done=0
        meta.write_text(json.dumps({'sig':signature,'n':n,'completed':0}))
    device=next(model.parameters()).device
    for a in tqdm(range(done,n,batch),desc='CATA full VAL inference',unit='image-batch'):
        b=min(n,a+batch)
        t=torch.from_numpy(np.asarray(tokens[a:b],dtype=np.float32).copy()).to(device)
        z=torch.from_numpy(np.asarray(logits[a:b],dtype=np.float32).copy()).to(device)
        logits2=model(t,z).cpu().numpy()
        if not np.isfinite(logits2).all():raise FloatingPointError('Bad logits')
        arr[a:b]=logits2;arr.flush()
        tmp=meta.with_suffix('.json.tmp')
        tmp.write_text(json.dumps({'sig':signature,'n':n,'completed':b}))
        os.replace(tmp,meta)
    del arr
    return np.load(outfile,mmap_mode='r')


def query_distance(query,logits,chunk=64):
    """Masked attribute probability NLL; no VAL label use."""
    q=np.asarray(query,np.float32);z=np.asarray(logits,np.float32)
    if q.ndim!=2 or z.ndim!=2 or q.shape[1]!=40 or z.shape[1]!=40:raise ValueError('40 labels expected')
    lp=np.logaddexp(0,-z);ln=np.logaddexp(0,z)
    out=np.empty((len(q),len(z)),np.float32)
    for a in tqdm(range(0,len(q),chunk),desc='Query-conditioned distances',unit='query-batch'):
        qb=q[a:a+chunk];p=(qb==1).astype(np.float32);n=(qb==0).astype(np.float32)
        out[a:a+len(qb)]=(p@lp.T+n@ln.T)/np.maximum((p+n).sum(1,keepdims=True),1)
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite distances')
    return out


def fuse(base,secondary,alpha):
    if alpha==0:return np.asarray(base,np.float32)
    x=np.asarray(base,np.float32);y=np.asarray(secondary,np.float32)
    if x.shape!=y.shape:raise ValueError('Mismatch ranking matrix shape')
    def zn(a):return (a-a.mean(1,keepdims=True))/np.maximum(a.std(1,keepdims=True),1.e-6)
    return (1-alpha)*zn(x)+alpha*zn(y)
