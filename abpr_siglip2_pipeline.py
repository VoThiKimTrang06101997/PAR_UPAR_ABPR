"""Frozen SigLIP 2 + TRAIN-only attribute adapter for UPAR Track 2.

No hidden data, no synthetic labels. Strict public path -> semantic ID alignment.
The scoring code is also reused by the offline submission wrapper.
"""
from __future__ import annotations
import hashlib, json, os, random, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

MODEL_ID = 'google/siglip2-base-patch16-224'


def digest_lines(seq):
    h=hashlib.sha256()
    for x in seq: h.update(str(x).encode('utf-8'));h.update(b'\n')
    return h.hexdigest()


def load_siglip(model_path=MODEL_ID, device=None):
    from transformers import AutoModel, AutoProcessor
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype=torch.float16 if str(device).startswith('cuda') else torch.float32
    processor=AutoProcessor.from_pretrained(str(model_path))
    try:
        model=AutoModel.from_pretrained(str(model_path),torch_dtype=dtype)
    except TypeError:
        model=AutoModel.from_pretrained(str(model_path),dtype=dtype)
    model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,processor,device


def tensor_feature(result):
    if isinstance(result,torch.Tensor):return result
    for key in ('pooler_output','image_embeds','text_embeds'):
        value=getattr(result,key,None)
        if isinstance(value,torch.Tensor):return value
    if isinstance(result,(tuple,list)) and len(result)>0:
        return tensor_feature(result[0])
    raise TypeError(f'Unexpected SigLIP2 feature output: {type(result)}')


def _processor_image_batch(processor,imgs,device):
    x=processor(images=imgs,return_tensors='pt')
    return {k:v.to(device) for k,v in x.items() if isinstance(v,torch.Tensor)}


def image_embeddings(paths,model,processor,device,cache,batch_size=16):
    """Resumeable numpy memmap, keyed by model and *ordered* image paths."""
    paths=[str(p) for p in paths]
    cache=Path(cache);cache.parent.mkdir(parents=True,exist_ok=True)
    meta=cache.with_suffix('.json')
    signature=digest_lines([MODEL_ID,len(paths),*paths])
    shape=None;completed=0
    if meta.is_file() and cache.is_file():
        info=json.loads(meta.read_text())
        if info.get('signature')==signature:
            shape=tuple(info['shape']);completed=int(info.get('completed',0))
            if shape[0]!=len(paths):raise ValueError('Cached gallery length mismatch')
            print(f'[CACHE] {cache}: {completed}/{len(paths)}',flush=True)
        else:
            print('[CACHE] Different image order/model: refusing stale cache',flush=True)
    if shape is None:
        # Dimensionality inferred from first image, not hardcoded.
        with Image.open(paths[0]) as im: first=im.convert('RGB')
        with torch.inference_mode():
            data=_processor_image_batch(processor,[first],device)
            vec=tensor_feature(model.get_image_features(**data))
            dim=int(vec.reshape(1,-1).shape[1])
        shape=(len(paths),dim);completed=0
        arr=np.lib.format.open_memmap(cache,mode='w+',dtype='float16',shape=shape)
        arr.flush();del arr
        meta.write_text(json.dumps({'signature':signature,'shape':list(shape),'completed':0}))
    arr=np.lib.format.open_memmap(cache,mode='r+',dtype='float16',shape=shape)
    for start in tqdm(range(completed,len(paths),batch_size),desc=f'SigLIP2 images [{len(paths)}]'):
        group=paths[start:start+batch_size]
        imgs=[]
        for p in group:
            try:
                with Image.open(p) as im: imgs.append(im.convert('RGB'))
            except Exception as exc: raise RuntimeError(f'Cannot read real image: {p}') from exc
        with torch.inference_mode():
            data=_processor_image_batch(processor,imgs,device)
            vec=tensor_feature(model.get_image_features(**data)).float()
            vec=F.normalize(vec,dim=-1)
        out=vec.cpu().numpy().astype('float16')
        if out.shape!=(len(group),shape[1]):raise ValueError(f'Bad vision features {out.shape}')
        arr[start:start+len(group)]=out
        arr.flush()
        meta.write_text(json.dumps({'signature':signature,'shape':list(shape),'completed':start+len(group)}))
        del vec,data,imgs
    del arr
    return np.load(cache,mmap_mode='r')


def prompt_pair(attribute):
    """Use the organizer's names; predictable positive/negative captions, no manual label remapping."""
    name=str(attribute).replace('&',' and ').replace('-', ' ').replace('_',' ').lower()
    tokens=name.split()
    if tokens[:2]==['upperbody','color']:
        thing=f'{" ".join(tokens[2:])} upper-body clothing'
    elif tokens[:2]==['lowerbody','color']:
        thing=f'{" ".join(tokens[2:])} lower-body clothing'
    elif tokens[0]=='age':
        thing=f'a person of {" ".join(tokens[1:])} age'
    elif tokens[0]=='gender':
        thing=f'a {" ".join(tokens[1:])} pedestrian'
    else:
        thing=name+' attribute'
    return (f'A surveillance photo of a pedestrian with {thing}.',
            f'A surveillance photo of a pedestrian without {thing}.')


def text_pairs(names,model,processor,device,batch_size=16):
    sentences=[]
    for name in names:sentences.extend(prompt_pair(name))
    groups=[]
    for start in tqdm(range(0,len(sentences),batch_size),desc='SigLIP2 attribute prompts'):
        tok=processor(text=sentences[start:start+batch_size],padding='max_length',max_length=64,return_tensors='pt')
        inp={k:v.to(device) for k,v in tok.items() if isinstance(v,torch.Tensor) and k in ('input_ids','attention_mask','position_ids')}
        with torch.inference_mode():vec=tensor_feature(model.get_text_features(**inp)).float()
        groups.append(F.normalize(vec,dim=-1).cpu().numpy())
    mat=np.concatenate(groups).reshape(len(names),2,-1)
    return np.asarray(mat,dtype=np.float32)


def semantic_query_embedding(q,text_matrix):
    """Unknown (-1) attributes contribute nothing; negative labels use negative prompts."""
    q=np.asarray(q,dtype=np.float32)
    if q.ndim!=2 or q.shape[1]!=text_matrix.shape[0]:raise ValueError('Attribute query order/shape mismatch')
    one=(q==1).astype(np.float32);zero=(q==0).astype(np.float32)
    text=np.asarray(text_matrix,np.float32)
    embedding=one @ text[:,0,:]+zero @ text[:,1,:]
    embedding/=np.maximum(np.linalg.norm(embedding,axis=1,keepdims=True),1.e-8)
    return embedding


def prompt_distance(q,features,text_matrix):
    query=semantic_query_embedding(q,text_matrix)
    a=np.asarray(features,dtype=np.float32)
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8)
    return -(query @ a.T)


def adapter_distance(q,features,weight,bias):
    """Negative log-likelihood, with unknown query attributes ignored."""
    q=np.asarray(q,np.float32);x=np.asarray(features,np.float32)
    w=np.asarray(weight,np.float32);b=np.asarray(bias,np.float32)
    z=x @ w + b
    # Stable log-sigmoid
    lp=-np.logaddexp(0,-z);ln=-np.logaddexp(0,z)
    p=(q==1).astype(np.float32);n=(q==0).astype(np.float32)
    count=np.maximum(p.sum(1)+n.sum(1),1.)
    return -((p @ lp.T)+(n @ ln.T))/count[:,None]


def standardize_rows(x):
    x=np.asarray(x,np.float32)
    return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-5)


def combine(base,prompt,adapter,kind,blend):
    """For blend=0 retain exact baseline values, no normalization."""
    if blend==0:return np.asarray(base,np.float32)
    if kind=='prompt': new=prompt
    elif kind=='adapter':new=adapter
    elif kind=='prompt_adapter':new=0.5*standardize_rows(prompt)+0.5*standardize_rows(adapter)
    else:raise ValueError(f'Unknown candidate {kind}')
    return (1-blend)*standardize_rows(base)+blend*standardize_rows(new)


def balanced_sample(frame,image_col,domain_of,per_domain,seed=42):
    rng=np.random.default_rng(seed)
    sampled=[]
    for domain in (0,1,2):
        rows=np.flatnonzero(np.array([domain_of(v) for v in frame[image_col].astype(str)],dtype=int)==domain)
        if not len(rows):raise RuntimeError(f'No TRAIN samples for domain {domain}')
        sampled.extend(rng.choice(rows,size=min(per_domain,len(rows)),replace=False).tolist())
    rng.shuffle(sampled)
    return frame.iloc[sampled].reset_index(drop=True)


def train_adapter(features,labels,domain_ids,save_to,epochs=6,batch_size=256,lr=0.002,seed=42,mp_weight=0.08):
    """One short training run on frozen TRAIN feature vectors only."""
    torch.manual_seed(seed);np.random.seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    X=torch.as_tensor(np.asarray(features,np.float32),device=device)
    Y=torch.as_tensor(np.asarray(labels,np.float32),device=device)
    D=torch.as_tensor(np.asarray(domain_ids,dtype=np.int64),device=device)
    if X.shape[0]!=Y.shape[0] or Y.shape[1]!=40:raise ValueError('TRAIN feature/label shape mismatch')
    seen=(Y==0)|(Y==1)
    positive=((Y==1)&seen).sum(0).float();tot=seen.sum(0).float()
    pos_w=((tot-positive+1)/(positive+1)).clamp(1,6)
    head=torch.nn.Linear(X.shape[1],40).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.01)
    domain_rows=[torch.where(D==d)[0] for d in range(3)]
    if any(not len(rows) for rows in domain_rows):raise ValueError('All three TRAIN domains required')
    rng=np.random.default_rng(seed)
    for ep in range(epochs):
        nsteps=max(1,int(np.ceil(X.shape[0]/batch_size)))
        totals=[]
        for step in tqdm(range(nsteps),desc=f'TRAIN-only adapter epoch {ep+1}/{epochs}',leave=False):
            chunks=[]
            for d,rows in enumerate(domain_rows):
                pick=torch.randint(len(rows),(batch_size//3 + int(d<batch_size%3),),device=device)
                chunks.append(rows[pick])
            indices=torch.cat(chunks)
            logits=head(X[indices]);yi=Y[indices];valid=seen[indices]
            target=yi.clamp(0,1)
            weight=torch.where(target==1,pos_w[None,:],torch.ones_like(target))
            bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight*valid.float()
            loss=bce.sum()/valid.sum().clamp_min(1)
            if mp_weight>0 and step%3==0:
                # multi-positive query retrieval: 4 known features per image (when available).
                known=valid.float()
                rnd=torch.rand_like(yi)
                chosen=(rnd<0.25)&valid
                short=(chosen.sum(1)<2)
                if short.any():
                    kth=torch.topk(rnd.masked_fill(~valid,-1),k=min(4,rnd.shape[1]),dim=1).indices
                    forced=torch.zeros_like(valid).scatter(1,kth,True)&valid
                    chosen=torch.where(short[:,None],forced,chosen)
                qpos=((yi==1)&chosen).float();qneg=((yi==0)&chosen).float()
                lp=F.logsigmoid(logits);ln=F.logsigmoid(-logits)
                scores=(qpos@lp.T+qneg@ln.T)/(qpos.sum(1)+qneg.sum(1)).clamp_min(1)[:,None]
                matches=qpos @ (yi==1).float().T + qneg @ (yi==0).float().T
                incompatible=qpos @ (yi==0).float().T+qneg@(yi==1).float().T
                pos=matches >= (qpos.sum(1)+qneg.sum(1))[:,None]-1e-4
                neg=incompatible>0
                eligible=pos|neg
                has_neg=neg.any(1)&pos.any(1)
                if has_neg.any():
                    log_pos=torch.logsumexp(scores.masked_fill(~pos,-1e4),dim=1)
                    log_all=torch.logsumexp(scores.masked_fill(~eligible,-1e4),dim=1)
                    loss=loss+mp_weight*(log_all[has_neg]-log_pos[has_neg]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            totals.append(float(loss.detach()))
        print(f'Adapter epoch={ep+1} avg_loss={np.mean(totals):.5f}',flush=True)
    save_to=Path(save_to);save_to.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(save_to,weight=head.weight.detach().cpu().numpy().T,bias=head.bias.detach().cpu().numpy(),
                        train_rows=np.int32(len(X)),seed=np.int32(seed))
    return save_to


def ids_from_official(csv_path,raw_image_names):
    import pandas as pd
    ids=pd.read_csv(csv_path,header=None,dtype=str)
    if ids.shape[1]<2:raise ValueError('ids.csv must have path/query-id columns')
    lookup={}
    for path,v in ids.iloc[:,:2].itertuples(index=False,name=None):
        key=str(path).strip().replace('\\','/').lstrip('./')
        try: n=int(float(v))
        except (ValueError,TypeError):
            if key.lower() in ('path','image','image_path'):continue
            raise
        if key in lookup and lookup[key]!=n:raise ValueError(f'Conflicting IDs: {key}')
        lookup[key]=n
    out=[]
    for x in raw_image_names:
        key=str(x).strip().replace('\\','/').lstrip('./')
        if key not in lookup:raise ValueError(f'Query ID missing for exact image path {key}')
        out.append(lookup[key])
    return np.asarray(out,dtype=np.int32)


def semantic_metrics(dist,query_ids,gallery_ids,queries,gt_labels):
    from abpr.query_aware_ranking import retrieval_metrics
    from abpr.model import track2_metrics
    result=retrieval_metrics(dist,np.asarray(query_ids),np.asarray(gallery_ids))
    result.update(track2_metrics(np.asarray(dist,np.float32),queries,gt_labels))
    return result
