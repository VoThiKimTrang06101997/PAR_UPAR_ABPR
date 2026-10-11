"""HATA: Hierarchical Attribute-Token Adaptation for ABPR.
Research extension, not a published model. TRAIN-only learning; no VAL labels in scoring.
"""
from __future__ import annotations
import hashlib, json, math, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


def digest(*items):
    h=hashlib.sha256()
    for x in items:
        if isinstance(x,(np.ndarray,)):
            a=np.ascontiguousarray(x);h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        else:h.update(str(x).encode())
        h.update(b'\n')
    return h.hexdigest()


def group_ids(attributes):
    assert len(attributes)==40 and len(set(attributes))==40
    groups=[]
    for a in attributes:
        if a.startswith('Age-'):groups.append(0)
        elif a.startswith('Gender-'):groups.append(1)
        elif a.startswith('Hair-'):groups.append(2)
        elif a.startswith('UpperBody-'):groups.append(3)
        elif a.startswith('LowerBody-'):groups.append(4)
        elif a.startswith('Accessory-'):groups.append(5)
        else:raise ValueError(f'Unrecognized attribute group: {a}')
    return np.asarray(groups,dtype=np.int64)

class HierarchicalAttributeAdapter(nn.Module):
    def __init__(self,group_indices,dim=192,hidden=64,dropout=.10,scale=.35):
        super().__init__()
        self.register_buffer('group_indices',torch.as_tensor(group_indices,dtype=torch.long))
        self.n_groups=int(max(group_indices))+1
        self.norm=nn.LayerNorm(dim)
        self.token_projection=nn.Linear(dim,hidden)
        self.group_projection=nn.Linear(dim,hidden,bias=False)
        self.attr_embedding=nn.Parameter(torch.randn(40,hidden)*.02)
        self.readout=nn.Parameter(torch.randn(40,hidden)*.02)
        self.bias=nn.Parameter(torch.zeros(40))
        self.dropout=nn.Dropout(dropout)
        self.scale=float(scale)
    def forward(self,tokens,head_logits):
        b,c,d=tokens.shape
        if c!=40:raise ValueError('Expected 40 DINO attribute tokens')
        h=self.norm(tokens.float())
        gv=[]
        for k in range(self.n_groups):gv.append(h[:,self.group_indices==k].mean(1))
        gv=torch.stack(gv,dim=1)
        z=F.gelu(self.token_projection(h)+self.group_projection(gv)[:,self.group_indices,:]+self.attr_embedding[None,:,:])
        z=self.dropout(z)
        delta=(z*self.readout[None,:,:]).sum(-1)/math.sqrt(z.shape[-1])+self.bias[None,:]
        return head_logits.float()+self.scale*delta


def query_negative_loss(logits,labels,observed,full_probability=.75,margin=.15,seed_fraction=.16):
    """Batch query ranking: negatives require an explicitly observed attribute conflict.
    Positive is own image. No unknown label is treated as negative.
    """
    B,C=logits.shape
    y=labels.clamp(0,1)
    keep=((torch.rand((B,1),device=logits.device)<full_probability)&observed) | ((torch.rand_like(logits)<seed_fraction)&observed)
    insufficient=keep.sum(1)<2
    if insufficient.any():
        # Guarantee known labels when possible. Never promote an unknown label.
        rand=torch.rand_like(logits).masked_fill(~observed,-1)
        take=torch.topk(rand,k=min(10,C),dim=1).indices
        fill=torch.zeros_like(observed).scatter(1,take,True)&observed
        keep=torch.where(insufficient[:,None],fill,keep)
    positive=((y==1)&keep).float();negative=((y==0)&keep).float()
    known=(positive+negative).sum(1).clamp_min(1)
    ll_pos=F.logsigmoid(logits);ll_neg=F.logsigmoid(-logits)
    scores=(positive@ll_pos.T+negative@ll_neg.T)/known[:,None]
    conflicts=((positive@((labels==0)&observed).float().T)+(negative@((labels==1)&observed).float().T))>0
    conflicts.fill_diagonal_(False)
    valid=conflicts.any(1)
    if not valid.any():return logits.sum()*0
    hardest=scores.masked_fill(~conflicts,-1.e6).max(1).values
    return F.relu(margin+hardest[valid]-scores.diagonal()[valid]).mean()


def masked_bce(pred,target,obs,pos_weights):
    t=target.clamp(0,1)
    w=torch.where(t>0.5,pos_weights[None,:],1.)
    return (F.binary_cross_entropy_with_logits(pred,t,reduction='none')*w*obs).sum()/obs.sum().clamp_min(1)


def balanced_indices(y,dom,seed,holdout=0.15):
    rng=np.random.default_rng(seed)
    training=[];hold=[]
    for d in range(3):
        indices=np.flatnonzero(dom==d)
        if len(indices)<12:raise RuntimeError(f'TRAIN domain {d} requires >=12 samples')
        rng.shuffle(indices)
        nh=max(1,int(len(indices)*holdout))
        hold.extend(indices[:nh]); training.extend(indices[nh:])
    return np.asarray(training,dtype=np.int64),np.asarray(hold,dtype=np.int64)


def fit(tokens,base_logits,labels,domains,attrs,output_dir,signature,epochs=10,batch_size=96,lr=1.e-3,
        query_weight=.12,feature_weight=.015,seed=42,force=False,patience=4):
    """Frozen DINO tokens -> tiny HATA adapter. Checkpoints best TRAIN holdout, resumes epochs."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    last=out/'hata_last.pt';best=out/'hata_best.pt'
    Y=np.asarray(labels,np.float32);D=np.asarray(domains,np.int64)
    assert Y.ndim==2 and Y.shape[1]==40 and set(np.unique(D))=={0,1,2}
    assert tokens.shape==(len(Y),40,192) and base_logits.shape==(len(Y),40)
    assert np.isin(Y,[-1,0,1]).all(), 'Masked training labels must be -1,0,1'
    grouping=group_ids(attrs)
    config={'signature':signature,'epochs':epochs,'batch':batch_size,'lr':lr,'query_weight':query_weight,
            'feature_weight':feature_weight,'seed':seed,'attributes':list(attrs)}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    tr,hold=balanced_indices(Y,D,seed)
    obs=np.isin(Y,[0,1])
    pos=((Y==1)&obs)[tr].sum(0);neg=((Y==0)&obs)[tr].sum(0)
    weights=torch.tensor(np.clip(np.sqrt((neg+1)/(pos+1)),.7,3.0),dtype=torch.float32,device=device)
    net=HierarchicalAttributeAdapter(grouping).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs),eta_min=lr*.10)
    train_by_domain=[tr[D[tr]==d] for d in range(3)]
    rng=np.random.default_rng(seed)
    history=[];start=0;best_loss=math.inf;bad=0
    if last.is_file() and not force:
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['config']!=config:raise RuntimeError('Existing HATA checkpoint has different config/signature. Set force=True or change outdir.')
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        start=int(ck['next_epoch']);best_loss=float(ck['best_holdout']);history=list(ck['history']);bad=int(ck.get('bad',0))
        rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'])
    def get_rows(ix):
        # Index memmapped cache on demand, never load 33k VAL tokens into GPU.
        x=torch.from_numpy(np.array(tokens[ix],dtype=np.float32,copy=True)).to(device)
        z=torch.from_numpy(np.array(base_logits[ix],dtype=np.float32,copy=True)).to(device)
        y=torch.from_numpy(np.array(Y[ix],dtype=np.float32,copy=True)).to(device)
        return x,z,y,((y==0)|(y==1))
    for ep in tqdm(range(start,epochs),desc='HATA TRAIN-only epochs',unit='epoch'):
        net.train();step_losses=[]
        steps=max(1,math.ceil(len(tr)/batch_size))
        for _ in tqdm(range(steps),desc=f'HATA epoch {ep+1}',unit='batch',leave=False):
            sizes=[batch_size//3+int(d<batch_size%3) for d in range(3)]
            indices=np.concatenate([rng.choice(train_by_domain[d],size=sizes[d],replace=True) for d in range(3)])
            x,z,y,v=get_rows(indices)
            new=net(x,z)
            loss_cls=masked_bce(new,y,v,weights)
            loss_rank=query_negative_loss(new,y,v)
            loss_anchor=F.smooth_l1_loss(new,z)
            loss=loss_cls+query_weight*loss_rank+feature_weight*loss_anchor
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.0);opt.step()
            step_losses.append(float(loss.detach()))
        net.eval();val_total=0.;n=0
        with torch.inference_mode():
            for a in range(0,len(hold),256):
                x,z,y,v=get_rows(hold[a:a+256]);new=net(x,z)
                # TRAIN holdout supervised masked loss only.
                l=masked_bce(new,y,v,weights);val_total+=float(l)*len(y);n+=len(y)
        val_total/=max(n,1);scheduler.step()
        row={'epoch':ep+1,'train_loss':float(np.mean(step_losses)),'train_holdout_masked_bce':val_total}
        history.append(row)
        if val_total<best_loss-1e-5:
            best_loss=val_total;bad=0
            torch.save({'model':net.cpu().state_dict(),'config':config,'best_holdout':best_loss},best)
            net.to(device)
        else:bad+=1
        ck={'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'next_epoch':ep+1,'config':config,'best_holdout':best_loss,'history':history,
            'bad':bad,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state()}
        torch.save(ck,last)
        print(f'[HATA epoch {ep+1}] TRAIN loss={row["train_loss"]:.4f} TRAIN-holdout BCE={val_total:.4f}',flush=True)
        if bad>=patience:print('[HATA] Early stop on TRAIN holdout',flush=True);break
    if not best.is_file():raise RuntimeError('No best HATA checkpoint created')
    ck=torch.load(best,map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);net.to(device).eval()
    return net,history,best

@torch.inference_mode()
def infer(net,tokens,base_logits,out_file,batch_size=320,expected_signature=None):
    """Resume-safe inference cache: restart at last fully flushed batch."""
    out_file=Path(out_file);meta=out_file.with_suffix('.json')
    n=len(tokens);start=0
    if out_file.is_file() and meta.is_file():
        info=json.loads(meta.read_text())
        if info.get('signature')!=expected_signature or info.get('shape')!=[n,40]:
            raise RuntimeError('Existing inference cache mismatched. Use a new checkpoint/cache directory.')
        mm=np.load(out_file,mmap_mode='r+')
        if mm.shape!=(n,40):raise RuntimeError('Malformed cached HATA logits')
        start=int(info.get('completed',0))
        if not 0<=start<=n:raise RuntimeError('Invalid cache completion counter')
        if start==n:return np.load(out_file,mmap_mode='r')
    elif out_file.exists() or meta.exists():
        raise RuntimeError('Incomplete HATA cache metadata/files; investigate before resuming')
    else:
        out_file.parent.mkdir(parents=True,exist_ok=True)
        mm=np.lib.format.open_memmap(out_file,mode='w+',dtype='float32',shape=(n,40))
        meta.write_text(json.dumps({'signature':expected_signature,'completed':0,'shape':[n,40]},indent=2))
    device=next(net.parameters()).device
    for a in tqdm(range(start,n,batch_size),desc='HATA gallery logits',unit='batch'):
        b=min(n,a+batch_size)
        x=torch.from_numpy(np.array(tokens[a:b],dtype=np.float32,copy=True)).to(device)
        z=torch.from_numpy(np.array(base_logits[a:b],dtype=np.float32,copy=True)).to(device)
        mm[a:b]=net(x,z).float().cpu().numpy();mm.flush()
        temp=meta.with_suffix('.json.tmp')
        temp.write_text(json.dumps({'signature':expected_signature,'completed':b,'shape':[n,40]},indent=2))
        os.replace(temp,meta)
    del mm
    return np.load(out_file,mmap_mode='r')


def weighted_query_nll(queries,image_logits,weights=None,chunk=40):
    q=np.asarray(queries,np.float32);img=np.asarray(image_logits,np.float32)
    assert q.ndim==2 and img.ndim==2 and q.shape[1]==img.shape[1]==40
    if weights is None:weights=np.ones(40,np.float32)
    w=np.asarray(weights,np.float32);assert w.shape==(40,) and (w>0).all()
    pos=np.logaddexp(0,-img);neg=np.logaddexp(0,img)
    out=np.empty((len(q),len(img)),np.float32)
    for a in tqdm(range(0,len(q),chunk),desc='HATA query-conditioned distance',unit='query-batch'):
        s=q[a:a+chunk];p=(s==1).astype(np.float32)*w;n=(s==0).astype(np.float32)*w
        div=(p+n).sum(1,keepdims=True).clip(1e-6)
        out[a:a+len(s)]=(p@pos.T+n@neg.T)/div
    return out


def reliability(train_logits,labels,domains=None):
    """TRAIN-derived reliability only. Prefer leave-as-is if no labels valid."""
    l=np.asarray(train_logits,np.float32);y=np.asarray(labels,np.float32)
    assert l.shape==y.shape and y.shape[1]==40
    correct=((l>0).astype(np.float32)==y)
    out=np.zeros(40,np.float32)
    for c in range(40):
        sel=np.isin(y[:,c],[0,1]);p=sel&(y[:,c]==1);n=sel&(y[:,c]==0)
        if min(p.sum(),n.sum())<8:out[c]=.5;continue
        balanced=.5*(correct[p,c].mean()+correct[n,c].mean())
        out[c]=float(balanced)
    # modest weights: avoid amplifying noisy rare attributes
    return np.clip(.6+1.2*(out-.5),.45,1.25).astype(np.float32)
