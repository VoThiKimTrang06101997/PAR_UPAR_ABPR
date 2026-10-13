"""Research extension inspired by ITSELF (WACV 2026), NOT a reproduction.
Frozen DINO attribute-token bank -> attention refiner. TRAIN-only supervision.
The cached tokens are 40 attribute-pooled tokens, NOT ViT patch-attention maps;
therefore this is an attribute-token adaptation of GRAB/ATS ideas, not original MARS.
"""
from pathlib import Path
import hashlib, json, math, os, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


def digest(*pieces):
    h = hashlib.sha256()
    for piece in pieces:
        if isinstance(piece, np.ndarray):
            a = np.ascontiguousarray(piece)
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
        else:
            h.update(str(piece).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def atomic_json(path, data):
    p=Path(path);p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False), encoding='utf-8')
    os.replace(tmp,p)


def attribute_groups(names):
    groups=[]
    for a in names:
        found=None
        for i,prefix in enumerate(('Age-','Gender-','Hair-','UpperBody-','LowerBody-','Accessory-')):
            if a.startswith(prefix):found=i;break
        if found is None:raise ValueError('Unrecognized attribute '+a)
        groups.append(found)
    if len(groups)!=40 or len(set(names))!=40:raise ValueError('Need exactly 40 unique attributes')
    return np.asarray(groups, np.int64)


class AttentiveAttributeRefiner(nn.Module):
    """40 semantic queries attend 40 frozen attribute tokens + 6 group tokens.
    Group and own-attribute bias reduce shortcut correlations. Adaptive top-k
    retains more context early, then focuses on salient tokens (ATS-inspired).
    """
    def __init__(self, groups, width=96, token_dim=192, dropout=0.08, residual_scale=0.35):
        super().__init__()
        self.register_buffer('group_idx',torch.as_tensor(groups,dtype=torch.long))
        self.register_buffer('same_attribute',torch.eye(40,dtype=torch.float32))
        self.n_groups = int(np.max(groups))+1
        self.width = width
        self.scale = float(residual_scale)
        self.norm = nn.LayerNorm(token_dim)
        self.key = nn.Linear(token_dim,width,bias=False)
        self.value = nn.Linear(token_dim,width,bias=False)
        self.group_value = nn.Linear(token_dim,width,bias=False)
        self.attr_q = nn.Parameter(torch.randn(40,width)*0.02)
        self.attr_readout = nn.Parameter(torch.randn(40,width)*0.02)
        self.attr_bias = nn.Parameter(torch.zeros(40))
        self.gate = nn.Parameter(torch.tensor(-1.5))
        self.dropout=nn.Dropout(dropout)

    def forward(self, tokens, prior_logits, topk=40, return_attention=False):
        if tokens.ndim!=3 or tokens.shape[1:]!=(40,192):
            raise ValueError('Expected tokens [B,40,192]')
        if prior_logits.shape != (tokens.shape[0],40):
            raise ValueError('Expected prior logits [B,40]')
        x=self.norm(tokens.float())
        key=F.normalize(self.key(x),dim=-1)
        value=self.value(x)
        q=F.normalize(self.attr_q,dim=-1)
        weights=torch.einsum('ch,bkh->bck',q,key)*math.sqrt(self.width)
        group_matches=(self.group_idx[:,None]==self.group_idx[None,:]).float()
        weights=weights+0.30*group_matches[None,:,:]+0.50*self.same_attribute[None,:,:]
        k=int(max(1,min(40,topk)))
        if k<40:
            idx=weights.topk(k,dim=-1).indices
            mask=torch.zeros_like(weights,dtype=torch.bool).scatter_(-1,idx,True)
            mask=mask | self.same_attribute.bool()[None,:,:]
            weights=weights.masked_fill(~mask, -1.e4)
        attn=F.softmax(weights,dim=-1)
        context=torch.einsum('bck,bkh->bch',attn,value)
        group_tokens=torch.stack([x[:,self.group_idx==g,:].mean(dim=1) for g in range(self.n_groups)],dim=1)
        group_context=self.group_value(group_tokens)[:,self.group_idx,:]
        fused=self.dropout(F.gelu(context+0.2*group_context))
        delta=(fused*self.attr_readout[None,:,:]).sum(-1)/math.sqrt(self.width)+self.attr_bias
        delta=torch.tanh(delta)
        logits=prior_logits.float()+self.scale*torch.sigmoid(self.gate)*delta
        return (logits,attn) if return_attention else logits


def hard_negative_query_loss(logits, labels, observed, full_query_prob=0.75, margin=0.12):
    """Only explicitly contradictory TRAIN labels form negatives."""
    B,C=logits.shape
    full=(torch.rand((B,1),device=logits.device)<full_query_prob)
    known=observed
    subset=(torch.rand_like(logits)<0.2)&known
    keep=torch.where(full,known,subset)
    low=keep.sum(1)<2
    if low.any():
        rand=torch.rand_like(logits).masked_fill(~known,-1)
        pick=rand.topk(k=min(10,C),dim=1).indices
        keep=torch.where(low[:,None],torch.zeros_like(keep).scatter_(1,pick,True)&known,keep)
    pos=((labels==1)&keep).float();neg=((labels==0)&keep).float()
    count=(pos+neg).sum(1).clamp_min(1)
    sim=(pos@F.logsigmoid(logits).T+neg@F.logsigmoid(-logits).T)/count[:,None]
    conflicts=((pos@((labels==0)&known).float().T)+(neg@((labels==1)&known).float().T))>0
    conflicts.fill_diagonal_(False)
    valid=conflicts.any(1)
    if not valid.any():return logits.sum()*0
    hard=sim.masked_fill(~conflicts,-1.e5).max(dim=1).values
    return F.relu(margin+hard[valid]-sim.diagonal()[valid]).mean()


def stratified_split(domain,seed=42,hold_fraction=0.15):
    rng=np.random.default_rng(seed);train=[];hold=[]
    for d in (0,1,2):
        pool=np.flatnonzero(domain==d)
        if len(pool)<20:raise ValueError(f'TRAIN domain {d} has <20 samples')
        rng.shuffle(pool);nh=max(4,int(round(hold_fraction*len(pool))))
        hold.extend(pool[:nh]);train.extend(pool[nh:])
    return np.array(train,dtype=np.int64),np.array(hold,dtype=np.int64)


def fit(tokens, teacher_logits, labels, domains, attributes, out_dir, signature,
        epochs=10,batch_size=64,lr=0.0007,rank_weight=0.12,kd_weight=0.04,
        diversity_weight=0.002,seed=42,force=False,patience=5):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    last=out/'attention_last.pt';best=out/'attention_best.pt'
    Y=np.asarray(labels,np.float32);D=np.asarray(domains,np.int64)
    if Y.shape!=(len(tokens),40) or teacher_logits.shape!=(len(tokens),40):
        raise ValueError('TRAIN sample count and attribute order mismatch')
    if not np.isin(Y,[-1,0,1]).all():raise ValueError('TRAIN labels must be -1/0/1')
    grouping=attribute_groups(attributes)
    cfg=dict(signature=signature,epochs=int(epochs),batch_size=int(batch_size),lr=float(lr),rank_weight=float(rank_weight),kd_weight=float(kd_weight),diversity_weight=float(diversity_weight),seed=int(seed),attrs=list(attributes))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    tr,hold=stratified_split(D,seed)
    observed=np.isin(Y,[0,1]);pos=((Y[tr]==1)&observed[tr]).sum(0);neg=((Y[tr]==0)&observed[tr]).sum(0)
    class_weights=torch.as_tensor(np.clip(np.sqrt((neg+1)/(pos+1)),0.7,2.5),device=device,dtype=torch.float32)
    net=AttentiveAttributeRefiner(grouping).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.008)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=lr*0.1)
    rng=np.random.default_rng(seed)
    start=0;best_val=float('inf');bad=0;history=[]
    if last.is_file() and not force:
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['config']!=cfg:raise RuntimeError('Existing checkpoint configuration changed: change out_dir or FORCE=True')
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        start=ck['next_epoch'];best_val=ck['best_val'];bad=ck['bad'];history=ck['history']
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
    pools=[tr[D[tr]==d] for d in range(3)]
    def batch_rows(index):
        # Memmap -> CPU copy -> GPU, not entire VAL features.
        tok=torch.from_numpy(np.array(tokens[index],dtype=np.float32,copy=True)).to(device)
        prior=torch.from_numpy(np.array(teacher_logits[index],dtype=np.float32,copy=True)).to(device)
        labels=torch.from_numpy(np.array(Y[index],dtype=np.float32,copy=True)).to(device)
        return tok,prior,labels,((labels==0)|(labels==1))
    for epoch in range(start,epochs):
        net.train();run=[];steps=max(1,math.ceil(len(tr)/batch_size))
        for step in tqdm(range(steps),desc=f'Attention epoch {epoch+1}/{epochs}',leave=False,unit='batch'):
            parts=[rng.choice(pool,size=batch_size//3+int(d<batch_size%3),replace=True) for d,pool in enumerate(pools)]
            idx=np.concatenate(parts);rng.shuffle(idx)
            tok,prior,y,obs=batch_rows(idx)
            frac=(epoch+step/max(steps,1))/max(1,epochs)
            topk=int(round(40-(40-10)*frac))
            output,attn=net(tok,prior,topk=topk,return_attention=True)
            target=y.clamp(0,1);w=torch.where(y==1,class_weights[None,:],torch.ones_like(y))
            bce=(F.binary_cross_entropy_with_logits(output,target,reduction='none')*obs*w).sum()/obs.sum().clamp_min(1)
            ranking=hard_negative_query_loss(output,y,obs)
            kd=F.mse_loss(torch.sigmoid(output),torch.sigmoid(prior))
            # Attention diversity, penalize class-row collapse; subtract inherent diagonal first
            gram=attn@attn.transpose(-1,-2)
            off=gram*(1.-torch.eye(40,device=device)[None,:,:])
            diverse=off.sum()/(len(idx)*40*39)
            loss=bce+rank_weight*ranking+kd_weight*kd+diversity_weight*diverse
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.0);opt.step()
            run.append(float(loss.item()))
        scheduler.step()
        net.eval();hold_losses=[]
        with torch.inference_mode():
            for chunk in np.array_split(hold,max(1,math.ceil(len(hold)/192))):
                tok,prior,y,obs=batch_rows(chunk)
                p=net(tok,prior,topk=10);loss=(F.binary_cross_entropy_with_logits(p,y.clamp(0,1),reduction='none')*obs).sum()/obs.sum().clamp_min(1)
                hold_losses.append((float(loss.item()),int(obs.sum().item())))
        val=sum(x*w for x,w in hold_losses)/max(1,sum(w for _,w in hold_losses))
        row={'epoch':epoch+1,'train_loss':float(np.mean(run)),'train_holdout_BCE':float(val),'active_topk':topk}
        history.append(row);print('[ATTN TRAIN-only]',row,flush=True)
        if val<best_val-1.e-5:
            best_val=val;bad=0
            torch.save({'model':net.state_dict(),'config':cfg,'best_holdout':best_val,'epoch':epoch+1},best)
        else:bad+=1
        ck={'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'config':cfg,'next_epoch':epoch+1,'best_val':best_val,'bad':bad,'history':history,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state()}
        torch.save(ck,last);atomic_json(out/'attention_train_history.json',{'config':cfg,'history':history,'best_holdout':best_val})
        if bad>=patience:
            print('[EARLY STOP] TRAIN-holdout not improving');break
    if not best.is_file():raise RuntimeError('No best attention checkpoint; inspect training output')
    net.load_state_dict(torch.load(best,map_location='cpu',weights_only=False)['model']);net.eval()
    return net,best,history


@torch.inference_mode()
def infer(net,tokens,prior_logits,out_file,signature,batch_size=80):
    out_file=Path(out_file);meta=out_file.with_suffix('.json');n=len(tokens)
    if prior_logits.shape!=(n,40) or tokens.shape!=(n,40,192):
        raise ValueError('Frozen DINO token cache shape invalid')
    expected={'signature':signature,'shape':[n,40]}
    if out_file.exists() and meta.exists():
        info=json.loads(meta.read_text())
        if any(info.get(k)!=v for k,v in expected.items()):raise RuntimeError('Attention inference cache signature differs!')
        mm=np.load(out_file,mmap_mode='r+');cursor=int(info.get('completed',0))
        if mm.shape!=(n,40) or not 0<=cursor<=n:raise RuntimeError('Attention inference cache corrupt')
        if cursor==n:return np.load(out_file,mmap_mode='r')
    elif out_file.exists() or meta.exists():raise RuntimeError('Partial/missing cache metadata')
    else:
        out_file.parent.mkdir(parents=True,exist_ok=True)
        mm=np.lib.format.open_memmap(out_file,mode='w+',dtype='float32',shape=(n,40))
        cursor=0;atomic_json(meta,{**expected,'completed':0})
    device=next(net.parameters()).device
    for a in tqdm(range(cursor,n,batch_size),desc='Attention encode VAL attribute tokens',unit='batch'):
        b=min(n,a+batch_size)
        t=torch.from_numpy(np.array(tokens[a:b],dtype=np.float32,copy=True)).to(device)
        prior=torch.from_numpy(np.array(prior_logits[a:b],dtype=np.float32,copy=True)).to(device)
        mm[a:b]=net(t,prior,topk=10).cpu().numpy();mm.flush()
        atomic_json(meta,{**expected,'completed':b})
    del mm
    return np.load(out_file,mmap_mode='r')


def weighted_query_nll(queries,logits,weights=None,chunk=48):
    queries=np.asarray(queries,dtype=np.float32)
    logits=np.asarray(logits,dtype=np.float32)
    if queries.ndim!=2 or queries.shape[1]!=40 or logits.ndim!=2 or logits.shape[1]!=40:
        raise ValueError('Expected query and gallery with 40 ordered attributes')
    if not np.isin(queries,[-1,0,1]).all():raise ValueError('Queries must be -1/0/1')
    weight=np.ones(40,np.float32) if weights is None else np.asarray(weights,np.float32)
    if weight.shape!=(40,) or (weight<=0).any():raise ValueError('Invalid weights')
    pos=np.logaddexp(0,-logits);neg=np.logaddexp(0,logits)
    out=np.empty((len(queries),len(logits)),np.float32)
    for a in tqdm(range(0,len(queries),chunk),desc='Attention query distance',unit='batch'):
        q=queries[a:a+chunk];p=(q==1).astype(np.float32)*weight;n=(q==0).astype(np.float32)*weight
        out[a:a+len(q)]=(p@pos.T+n@neg.T)/np.maximum((p+n).sum(1,keepdims=True),1e-6)
    return out


def rowz(arr):
    x=np.asarray(arr,dtype=np.float32);mean=x.mean(1,keepdims=True);std=x.std(1,keepdims=True).clip(1e-5)
    return (x-mean)/std

def combine(base,candidate,alpha):
    if float(alpha)==0:return np.array(base,copy=True)
    return (1-float(alpha))*rowz(base)+float(alpha)*rowz(candidate)
