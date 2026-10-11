"""DINOv2 Track1 -> ABPR Track2 exploratory representation transfer.
Frozen teacher inference, TRAIN-only attribute-token adapter, strict image-order caches.
No VAL labels are used to train or score candidates.
"""
from __future__ import annotations
import hashlib,json,os,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def digest_lines(items):
    h=hashlib.sha256()
    for x in items:h.update(str(x).encode());h.update(b'\n')
    return h.hexdigest()

def load_teacher(checkpoint,model_py,device):
    import importlib.util
    module_spec=importlib.util.spec_from_file_location('dino_transfer_model_v1',str(model_py))
    assert module_spec and module_spec.loader
    import sys
    module=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=module
    module_spec.loader.exec_module(module)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=dict(ck.get('dinov2_config',{}));cfg.pop('pretrained',None);cfg['trainable_last']=0
    net=module.DinoAttributePAR(pretrained=False,**cfg)
    state=ck.get('inference_model_state')
    if state is None:raise RuntimeError('Track1 DINO checkpoint does not contain inference_model_state')
    net.load_state_dict(state,strict=True)
    net=net.to(device).eval();net.requires_grad_(False)
    return net,ck

def _save_json_atomic(path,obj):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(temp,path)

def build_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((252,126),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

@torch.inference_mode()
def cache_teacher(teacher,paths,out_dir,signature,batch_size=16,flip=True,device='cuda'):
    """Strict cache with byte-level signature and resumable ordered memmaps."""
    from PIL import Image
    paths=list(map(str,paths));out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    meta_file=out_dir/'meta.json'; feat_file=out_dir/'tokens.npy';logit_file=out_dir/'logits.npy'
    n=len(paths); expected={'signature':str(signature),'n':n,'flipped':bool(flip),'transform':'bicubic252x126_imagenet_v1'}
    if meta_file.exists():
        info=json.loads(meta_file.read_text())
        if any(info.get(k)!=v for k,v in expected.items()):
            raise RuntimeError(f'Feature cache signature mismatch: {out_dir}; use different directory')
        if not feat_file.exists() or not logit_file.exists():
            raise RuntimeError('Incomplete cache files missing; manually investigate '+str(out_dir))
        f=np.load(feat_file,mmap_mode='r+');l=np.load(logit_file,mmap_mode='r+')
        assert f.shape==(n,40,192) and l.shape==(n,40)
        cursor=int(info.get('completed',0));assert 0<=cursor<=n
    else:
        if feat_file.exists() or logit_file.exists():
            raise RuntimeError('Cache array exists without metadata: '+str(out_dir))
        f=np.lib.format.open_memmap(feat_file,mode='w+',dtype='float16',shape=(n,40,192))
        l=np.lib.format.open_memmap(logit_file,mode='w+',dtype='float16',shape=(n,40))
        cursor=0;_save_json_atomic(meta_file,{**expected,'completed':0})
    tfm=build_transform()
    pbar=tqdm(total=n,initial=cursor,desc='DINO frozen encode '+out_dir.name,unit='img',dynamic_ncols=True)
    try:
        for a in range(cursor,n,batch_size):
            piece=paths[a:a+batch_size];xs=[]
            for image_path in piece:
                if not Path(image_path).is_file():raise FileNotFoundError(image_path)
                with Image.open(image_path) as im:xs.append(tfm(im.convert('RGB')))
            x=torch.stack(xs).to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=(str(device).startswith('cuda'))):
                aux=teacher(x,return_aux=True)
                if flip:
                    aux2=teacher(x.flip(-1),return_aux=True)
                    # flip changes token order only spatially; tokens here are per-attribute pooled vectors
                    toks=.5*(aux['features'].float()+aux2['features'].float())
                    logits=.5*(aux['logits'].float()+aux2['logits'].float())
                else:
                    toks=aux['features'].float();logits=aux['logits'].float()
            f[a:a+len(piece)]=toks.cpu().numpy().astype(np.float16)
            l[a:a+len(piece)]=logits.cpu().numpy().astype(np.float16)
            f.flush();l.flush()
            _save_json_atomic(meta_file,{**expected,'completed':a+len(piece)})
            pbar.update(len(piece));del x,xs,aux
    finally:pbar.close()
    return np.load(feat_file,mmap_mode='r'),np.load(logit_file,mmap_mode='r')

class AttributeTokenHead(nn.Module):
    def __init__(self,num_attrs=40,dim=192,scale=.3):
        super().__init__()
        self.scale=float(scale)
        self.norm=nn.LayerNorm(dim)
        self.weight=nn.Parameter(torch.zeros(num_attrs,dim))
        self.bias=nn.Parameter(torch.zeros(num_attrs))
    def forward(self,tokens,base_logits):
        corrections=torch.einsum('bcd,cd->bc',self.norm(tokens),self.weight)/math.sqrt(tokens.shape[-1])+self.bias
        return base_logits+self.scale*corrections

def train_head(train_tokens,train_logits,labels,domains,path,epochs=12,batch_size=128,lr=.003,seed=42,rank_weight=.07,steps_per_epoch=0,full_query_probability=.65):
    """Only TRAIN attribute labels and TRAIN image features; no public VAL labels."""
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T=np.asarray(train_tokens);L=np.asarray(train_logits);Y=np.asarray(labels,np.float32);D=np.asarray(domains,np.int64)
    assert T.shape==(len(Y),40,192) and L.shape==(len(Y),40) and Y.shape[1]==40
    valid=np.isin(Y,[0,1]);assert valid.sum()>0
    train_indices=[]; hold_indices=[]
    for d in (0,1,2):
        ix=np.flatnonzero(D==d);rng.shuffle(ix);k=max(1,int(round(.12*len(ix))));hold_indices.extend(ix[:k]);train_indices.extend(ix[k:])
    train_indices=np.asarray(train_indices);hold_indices=np.asarray(hold_indices)
    X=torch.as_tensor(T.astype(np.float32),device=device);Z=torch.as_tensor(L.astype(np.float32),device=device)
    YY=torch.as_tensor(Y,device=device);VV=torch.as_tensor(valid,device=device)
    ps=(valid&(Y==1)).sum(0);ng=(valid&(Y==0)).sum(0)
    weights=torch.as_tensor(np.clip(np.sqrt((ng+1)/(ps+1)),.7,3.),device=device,dtype=torch.float32)
    head=AttributeTokenHead().to(device);opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.002)
    groups=[train_indices[D[train_indices]==d] for d in (0,1,2)]
    if any(not len(i) for i in groups):raise ValueError('Each domain requires TRAIN images')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    best=float('inf'); start=0; history=[]
    if path.is_file():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if (ck.get('epoch_count')!=int(epochs) or ck.get('seed')!=seed or
            ck.get('train_count')!=len(Y) or ck.get('full_query_probability')!=full_query_probability):
            raise RuntimeError('Resume configuration differs; use another checkpoint')
        head.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);start=int(ck['next_epoch']);best=float(ck['best_loss']);history=ck['history']
    for ep in tqdm(range(start,epochs),desc='TRAIN-only DINO attribute-token head',unit='epoch'):
        head.train();losses=[]
        nsteps=steps_per_epoch or max(1,int(np.ceil(len(train_indices)/batch_size)))
        for _ in tqdm(range(nsteps),desc=f'Head epoch {ep+1}',leave=False):
            sel=np.concatenate([rng.choice(g,size=batch_size//3+int(d<batch_size%3),replace=True) for d,g in enumerate(groups)])
            ids=torch.as_tensor(sel,device=device)
            t=X[ids]; l=Z[ids];y=YY[ids];v=VV[ids]
            pred=head(t,l);target=y.clamp(0,1)
            w=torch.where(target==1,weights[None,:],1.)
            bce=(F.binary_cross_entropy_with_logits(pred,target,reduction='none')*w*v).sum()/v.sum().clamp_min(1)
            # Sample sparse TRAIN-derived queries (2-10 known attrs). Compare own positive
            # against the hardest in-batch image with at least one observed conflict.
            random=torch.rand_like(y);known=v
            # Track2 official queries often specify all 40 attributes. Mix
            # full TRAIN-label queries with sparse augmentation to bridge that shift.
            full=(torch.rand((len(y),1),device=device)<full_query_probability)
            keep=torch.where(full,known,(random<.18)&known)
            for _k in range(2):
                low=keep.sum(1)<2
                if low.any():
                    fill=torch.topk(random.masked_fill(~known,-1),k=10,dim=1).indices
                    proposed=torch.zeros_like(keep).scatter(1,fill,True)&known
                    keep=torch.where(low[:,None],proposed,keep)
            qpos=((y==1)&keep).float();qneg=((y==0)&keep).float()
            lp=F.logsigmoid(pred);ln=F.logsigmoid(-pred)
            norm=(qpos+qneg).sum(1).clamp_min(1)
            similarity=(qpos@lp.T+qneg@ln.T)/norm[:,None]
            conflict=(((qpos@((y==0)&v).float().T)+(qneg@((y==1)&v).float().T))>0)
            conflict.fill_diagonal_(False)
            hardest=similarity.masked_fill(~conflict,-1e6).max(1).values
            good=conflict.any(1)
            if good.any():
                rank=F.relu(.10+hardest[good]-similarity.diagonal()[good]).mean()
            else:rank=pred.sum()*0
            loss=bce+float(rank_weight)*rank+.0005*(head.weight**2).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),3);opt.step()
            losses.append(float(loss.item()))
        head.eval()
        with torch.inference_mode():
            vals=[]
            for ids0 in np.array_split(hold_indices,max(1,int(np.ceil(len(hold_indices)/512)))):
                if not len(ids0):continue
                ii=torch.as_tensor(ids0,device=device)
                p=head(X[ii],Z[ii]);y=YY[ii].clamp(0,1);v=VV[ii]
                vals.append((float((F.binary_cross_entropy_with_logits(p,y,reduction='none')*v).sum()),int(v.sum())))
            loss_val=sum(x for x,_ in vals)/max(1,sum(z for _,z in vals))
        history.append({'epoch':ep+1,'train_loss':float(np.mean(losses)),'train_holdout_masked_BCE':float(loss_val)})
        print(f'[DINO head] epoch={ep+1}/{epochs} loss={np.mean(losses):.5f} TRAIN-holdout={loss_val:.5f}',flush=True)
        # Save last for resume; separate best state for final inference
        if loss_val<best:
            best=loss_val
            torch.save({'state':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},'train_holdout_BCE':best,'epoch':ep+1},path.with_name('dino_token_head_best.pt'))
        torch.save({'state':head.state_dict(),'optimizer':opt.state_dict(),'next_epoch':ep+1,'best_loss':best,'history':history,
                    'epoch_count':int(epochs),'train_count':len(Y),'seed':seed,
                    'full_query_probability':full_query_probability},path)
    best_file=path.with_name('dino_token_head_best.pt')
    head.load_state_dict(torch.load(best_file,map_location='cpu',weights_only=False)['state']);head.eval()
    return head,history

@torch.inference_mode()
def head_logits(head,tokens,logits,batch=512):
    device=next(head.parameters()).device
    out=np.empty(np.shape(logits),np.float32)
    for a in tqdm(range(0,len(logits),batch),desc='DINO attribute-token head inference',unit='batch'):
        b=min(len(logits),a+batch)
        x=torch.as_tensor(np.asarray(tokens[a:b],np.float32),device=device)
        y=torch.as_tensor(np.asarray(logits[a:b],np.float32),device=device)
        out[a:b]=head(x,y).float().cpu().numpy()
    return out

def query_nll(queries,image_logits,chunk=128):
    """[queries, gallery] lower better, no query/gallery labels used."""
    q=np.asarray(queries,np.float32);z=np.asarray(image_logits,np.float32)
    if q.ndim!=2 or z.ndim!=2 or q.shape[1]!=40 or z.shape[1]!=40:raise ValueError('Expected 40 attributes')
    lp=np.logaddexp(0,-z);ln=np.logaddexp(0,z)  # -log P(y=1), -log P(y=0)
    ans=np.empty((len(q),len(z)),np.float32)
    for a in tqdm(range(0,len(q),chunk),desc='DINO query-conditional distance',unit='query batch'):
        qb=q[a:a+chunk]
        pos=(qb==1).astype(np.float32);neg=(qb==0).astype(np.float32)
        den=(pos+neg).sum(1).clip(1)[:,None]
        ans[a:a+len(qb)]=(pos@lp.T+neg@ln.T)/den
    if not np.isfinite(ans).all():raise FloatingPointError('Nonfinite query distances')
    return ans

def row_z(x):
    x=np.asarray(x,np.float32)
    return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)

def fusion(base,candidate,alpha):
    if alpha==0:return np.asarray(base,np.float32)
    return (1-alpha)*row_z(base)+alpha*row_z(candidate)
