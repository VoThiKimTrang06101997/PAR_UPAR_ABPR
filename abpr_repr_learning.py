"""TRAIN-only representation learning for UPAR Track-2 ABPR.
Frozen SigLIP2 features -> small image/query towers, masked BCE + multi-positive
semantic-query ranking. No VAL labels, IDs or distances enter optimization.
"""
from __future__ import annotations
import hashlib,json,math,os,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

class AttributeQueryNet(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128, dropout=.12):
        super().__init__()
        self.image_net=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,embed_dim))
        self.attr_net=nn.Linear(embed_dim,40)
        self.query_tokens=nn.Parameter(torch.randn(40,2,embed_dim)*.025)
        self.query_norm=nn.LayerNorm(embed_dim)
        self.input_dim=input_dim;self.embed_dim=embed_dim;self.hidden=hidden;self.dropout=dropout
    def image_features(self,x):
        h=self.image_net(F.normalize(x.float(),dim=-1))
        return F.normalize(h,dim=-1),self.attr_net(h)
    def query_features(self,q):
        q=q.float();present=(q==0)|(q==1)
        # No implicit class for -1 (unknown)
        indices=q.clamp(0,1).long()
        tok=self.query_tokens[torch.arange(40,device=q.device)[None,:],indices]
        summary=(tok*present[:,:,None]).sum(1)/present.sum(1,keepdim=True).clamp_min(1).sqrt()
        return F.normalize(self.query_norm(summary),dim=-1)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def sample_query(y,epoch,total_epochs,generator):
    """Curriculum: sparse/mid/full known attributes with no fabricated annotations."""
    out=np.full_like(y,-1,dtype=np.float32)
    for i in range(len(y)):
        valid=np.flatnonzero((y[i]==0)|(y[i]==1))
        if not len(valid):continue
        u=generator.random()
        if u < .30: low,high=2,6
        elif u < .65: low,high=6,14
        elif u < .85: low,high=14,25
        else: low,high=25,40
        # Later epochs emphasize longer queries, matching current 40-known VAL.
        if epoch>=total_epochs//2 and generator.random()<.35:low,high=20,40
        n=min(len(valid),int(generator.integers(low,high+1)))
        if n==0:n=1
        choice=generator.choice(valid,size=n,replace=False)
        out[i,choice]=y[i,choice]
    return out

def train_loss(model,x,y,domain,epoch,epochs,generator,pos_weight,ranking_w=.30,hard_w=.06,domain_w=.008):
    q_np=sample_query(y.detach().cpu().numpy(),epoch,epochs,generator)
    q=torch.as_tensor(q_np,device=x.device)
    z,logits=model.image_features(x)
    qz=model.query_features(q)
    known=(y==0)|(y==1)
    targets=y.clamp(0,1)
    weight=torch.where(targets==1,pos_weight[None,:],torch.ones_like(targets))
    loss_attr=(F.binary_cross_entropy_with_logits(logits,targets,reduction='none')*known.float()*weight).sum()/known.sum().clamp_min(1)
    q_known=(q==0)|(q==1)
    # Positive requires all query labels known and equal; unknown image label is NOT a negative.
    match=(q_known[:,None,:] & known[None,:,:] & (q[:,None,:]==y[None,:,:]))
    conflicts=(q_known[:,None,:] & known[None,:,:] & (q[:,None,:]!=y[None,:,:]))
    needed=q_known.sum(1)[:,None]
    positive=(match.sum(2)==needed) & (needed>0)
    negative=conflicts.any(2)
    eligible=positive|negative
    scores=(qz @ z.T)/.15
    valid=positive.any(1)&negative.any(1)
    if valid.any():
        lp=torch.logsumexp(scores.masked_fill(~positive,-1e4),dim=1)
        la=torch.logsumexp(scores.masked_fill(~eligible,-1e4),dim=1)
        loss_rank=(la-lp)[valid].mean()
        s_pos=scores.masked_fill(~positive,-1e4).amax(1)
        s_neg=scores.masked_fill(~negative,-1e4).amax(1)
        hard=F.relu(s_neg[valid]-s_pos[valid]+.25).mean()
    else:
        loss_rank=logits.sum()*0;hard=logits.sum()*0
    # Weak domain mean discrepancy, from TRAIN only, not label leakage.
    means=[z[domain==d].mean(0) for d in (0,1,2) if (domain==d).sum()>=2]
    dom=torch.stack([(a-b).square().mean() for i,a in enumerate(means) for b in means[i+1:]]).mean() if len(means)>1 else z.sum()*0
    total=loss_attr+ranking_w*loss_rank+hard_w*hard+domain_w*dom
    return total,{'loss':float(total.detach()),'attr':float(loss_attr.detach()),'rank':float(loss_rank.detach()),'hard':float(hard.detach())}

def masked_bce(model,x,y,device,batch=256):
    model.eval();total=0.;den=0.
    with torch.no_grad():
        for s in range(0,len(x),batch):
            X=torch.tensor(np.asarray(x[s:s+batch],np.float32),device=device)
            Y=torch.tensor(np.asarray(y[s:s+batch],np.float32),device=device)
            _,logits=model.image_features(X)
            valid=(Y==0)|(Y==1)
            total+=float((F.binary_cross_entropy_with_logits(logits,Y.clamp(0,1),reduction='none')*valid).sum())
            den+=int(valid.sum())
    return total/max(1,den)

def save_checkpoint(path,model,opt,epoch,best,signature,config,history):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.pt')
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'signature':signature,'config':config,'history':history},tmp)
    os.replace(tmp,path)

def fit_model(X,Y,D,output,signature,epochs=12,batch_size=192,lr=.0007,seed=2027,holdout=.18,force=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    latest=output/'repr_latest.pt';best_path=output/'repr_best.pt'
    X=np.asarray(X,np.float32);Y=np.asarray(Y,np.float32);D=np.asarray(D,np.int64)
    assert X.ndim==2 and Y.shape==(len(X),40) and len(D)==len(X)
    assert np.isfinite(X).all() and np.isin(Y,[-1,0,1]).all() and np.isin(D,[0,1,2]).all()
    seed_all(seed);device='cuda' if torch.cuda.is_available() else 'cpu';gen=np.random.default_rng(seed)
    # Separate source-stratified TRAIN holdout, never use public VAL for epochs.
    train=[];valid=[]
    for d in (0,1,2):
        ids=np.where(D==d)[0];gen.shuffle(ids)
        v=max(1,int(round(len(ids)*holdout)));valid.extend(ids[:v]);train.extend(ids[v:])
    train=np.asarray(train);valid=np.asarray(valid)
    net=AttributeQueryNet(X.shape[1]).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.02)
    cfg={'input_dim':int(X.shape[1]),'hidden':256,'embed_dim':128,'dropout':.12,'epochs':int(epochs),'batch_size':int(batch_size),'lr':float(lr),'seed':int(seed),'holdout':float(holdout)}
    start=0;best=float('inf');history=[]
    if latest.is_file() and not force:
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck.get('signature')!=signature or ck.get('config')!=cfg:
            raise RuntimeError('Existing checkpoint/cache signature or training config differs. Change output folder or explicitly FORCE_RETRAIN=True.')
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=int(ck['epoch'])+1;best=float(ck['best']);history=list(ck['history'])
        print('[RESUME]',start,'/',epochs,'epochs; best TRAIN-only holdout BCE:',best,flush=True)
    known=((Y[train]==0)|(Y[train]==1)); pos=(Y[train]==1).sum(0)
    pos_w=np.clip((known.sum(0)-pos+1)/np.maximum(1,pos+1),.8,4.).astype(np.float32)
    PW=torch.tensor(pos_w,device=device)
    Xv=X[valid];Yv=Y[valid]
    if start>=epochs and best_path.is_file():print('[CACHE] Representation checkpoint complete:',best_path);return best_path,history
    for ep in range(start,epochs):
        net.train();losses=[]
        # BALANCED DOMAIN SAMPLING WITH REPLACEMENT, TRAIN ONLY.
        buckets=[train[D[train]==d] for d in (0,1,2)]
        if any(len(b)==0 for b in buckets):raise ValueError('Every TRAIN domain needs samples')
        steps=max(1,math.ceil(len(train)/batch_size))
        for _ in tqdm(range(steps),desc=f'Representation ep {ep+1}/{epochs}',leave=False):
            samples=[]
            for d,b in enumerate(buckets):
                count=batch_size//3+int(d<batch_size%3)
                samples.extend(gen.choice(b,size=count,replace=True))
            gen.shuffle(samples);ids=np.asarray(samples,dtype=np.int64)
            xb=torch.tensor(X[ids],device=device);yb=torch.tensor(Y[ids],device=device)
            db=torch.tensor(D[ids],device=device)
            # Training curriculum; full dense queries are simulated only from KNOWN TRAIN labels.
            loss,log=train_loss(net,xb,yb,db,ep,epochs,gen,PW)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.0);opt.step()
            losses.append(log)
        valid_loss=masked_bce(net,Xv,Yv,device)
        summary={'epoch':ep+1,'train_loss':float(np.mean([r['loss'] for r in losses])),'train_rank_loss':float(np.mean([r['rank'] for r in losses])),'train_attr_loss':float(np.mean([r['attr'] for r in losses])),'holdout_BCE':float(valid_loss)}
        history.append(summary)
        if valid_loss<best:
            best=valid_loss
            save_checkpoint(best_path,net,opt,ep,best,signature,cfg,history)
        save_checkpoint(latest,net,opt,ep,best,signature,cfg,history)
        (output/'repr_training_history.json').write_text(json.dumps(history,indent=2))
        print('REPR:',summary,flush=True)
    return best_path,history

def load_model(path,device=None):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    cfg=ck['config'];device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=AttributeQueryNet(cfg['input_dim'],cfg['hidden'],cfg['embed_dim'],cfg['dropout']).to(device)
    model.load_state_dict(ck['model']);model.eval()
    return model,device,ck

@torch.inference_mode()
def image_outputs(model,features,device,chunk=2048):
    # Returns normalized learned embeddings + logits, in original gallery order.
    z=[];l=[]
    for start in tqdm(range(0,len(features),chunk),desc='Representation gallery features',unit='batch'):
        x=torch.tensor(np.asarray(features[start:start+chunk],np.float32),device=device)
        zi,li=model.image_features(x);z.append(zi.cpu().numpy().astype('float32'));l.append(li.cpu().numpy().astype('float32'))
    return np.concatenate(z),np.concatenate(l)

@torch.inference_mode()
def query_outputs(model,queries,device,chunk=256):
    zs=[]
    for start in range(0,len(queries),chunk):
        q=torch.tensor(np.asarray(queries[start:start+chunk],np.float32),device=device)
        zs.append(model.query_features(q).cpu().numpy().astype('float32'))
    return np.concatenate(zs)

def distances(query_z,gallery_z,logits,queries,chunk=48):
    """RETURN two score matrices. 0=best distance. Scores from learned outputs only."""
    query_z=np.asarray(query_z,np.float32);gallery_z=np.asarray(gallery_z,np.float32)
    queries=np.asarray(queries,np.float32); logits=np.asarray(logits,np.float32)
    assert len(logits)==len(gallery_z) and queries.shape==(len(query_z),40)
    learned=[];attributes=[]
    lp=-np.logaddexp(0,-logits);ln=-np.logaddexp(0,logits)
    for s in tqdm(range(0,len(queries),chunk),desc='Representation retrieval queries',unit='batch'):
        q=queries[s:s+chunk];positive=(q==1).astype(np.float32);negative=(q==0).astype(np.float32)
        denom=np.maximum((positive+negative).sum(1,keepdims=True),1.)
        attr=-((positive@lp.T)+(negative@ln.T))/denom
        cos=-(query_z[s:s+chunk]@gallery_z.T)
        learned.append(cos.astype('float32'));attributes.append(attr.astype('float32'))
    return np.concatenate(learned),np.concatenate(attributes)

def std_rows(x):
    x=np.asarray(x,np.float32)
    return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-5)

def combine_scores(base,semantic,attributes,method,alpha):
    base=np.asarray(base,np.float32)
    if alpha==0:return base.copy()
    new={'embedding':semantic,'attribute':attributes,'mix':.5*std_rows(semantic)+.5*std_rows(attributes)}[method]
    return ((1-alpha)*std_rows(base)+alpha*std_rows(new)).astype('float32')
