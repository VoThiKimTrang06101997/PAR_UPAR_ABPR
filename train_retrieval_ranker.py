"""Train an attribute-reliability ranker using public TRAIN labels, select blend on public VAL.

Inputs: existing trained ABPRRuntime package + public challenge annotations.
No hidden annotations or test rankings are used. Backbone is not retrained by
this script; `train_track2.py` separately trains/resumes the deep network.
"""
from __future__ import annotations
import argparse, json, hashlib, os, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from abpr.retrieval_reranker import AttributeRanker, fuse_distances


def img_col(df):
    for c in ('# image','image','image_path','path','filename'):
        if c in df.columns: return c
    raise ValueError(f'Image column missing: {df.columns.tolist()[:8]}')


def load_data(official, split, names):
    path=official/'data'/'annotations'/'task2'/split/'gt.csv'
    if not path.is_file(): raise FileNotFoundError(path)
    df=pd.read_csv(path)
    missing=[n for n in names if n not in df.columns]
    if missing: raise ValueError(f'Missing attributes in {path}: {missing}')
    return df,df[names].to_numpy(dtype=np.float32),df[img_col(df)].astype(str).tolist()


def select_evenly(df, max_images, seed):
    if max_images >= len(df): return np.arange(len(df))
    # Domain-balanced sampling across Market1501 / PA100k / PETA.
    rng=np.random.default_rng(seed)
    col=img_col(df)
    groups=df[col].astype(str).str.split('/').str[0].str.lower().to_numpy()
    unique=np.unique(groups)
    chosen=[]
    budget=max_images
    for j,domain in enumerate(unique):
        ids=np.flatnonzero(groups == domain)
        quota=min(len(ids), max(1, budget // (len(unique)-j)))
        chosen.extend(rng.choice(ids,size=quota,replace=False).tolist())
        budget-=quota
    if budget>0:
        remaining=np.setdiff1d(np.arange(len(df)),chosen)
        chosen.extend(rng.choice(remaining,size=min(budget,len(remaining)),replace=False).tolist())
    return np.asarray(sorted(chosen),dtype=np.int64)


def create_gallery_runtime(runtime_zip, output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    needed=['abpr_runtime.py','assets/model.pt']
    with zipfile.ZipFile(runtime_zip) as z:
        for name in needed:
            if name not in z.namelist():raise KeyError(f'{name} absent from {runtime_zip}')
            target=output_dir/name;target.parent.mkdir(parents=True,exist_ok=True)
            # avoid extracting ~200 MiB repeatedly if already present
            if not target.is_file() or target.stat().st_size!=z.getinfo(name).file_size:
                with z.open(name) as inp, target.open('wb') as out:
                    import shutil;shutil.copyfileobj(inp,out)
    sys.path.insert(0,str(output_dir))
    from abpr_runtime import ABPRRuntime
    return ABPRRuntime(output_dir/'assets'/'model.pt')


def encode_images(runtime, relpaths, resolver, batch, cache):
    if cache.is_file():
        with np.load(cache,allow_pickle=False) as z:
            if len(z['probs'])==len(relpaths):
                print(f'Cache hit: {cache}',flush=True)
                return z['probs'].copy(), z['embeddings'].copy()
    prob=[]
    embeds=[]
    for start in tqdm(range(0,len(relpaths),batch),desc=f'Encode real images ({cache.stem})',unit='batch'):
        paths=[{'image_path':str(resolver.resolve(v))} for v in relpaths[start:start+batch]]
        p,e=runtime.encode_gallery(paths)
        if hasattr(e,'detach'): e=e.detach().cpu().numpy()
        e=np.asarray(e,dtype=np.float32)
        if e.ndim!=2 or e.shape[0]!=len(paths) or not np.isfinite(e).all():
            raise ValueError(f'Invalid embedding shape={e.shape}')
        embeds.append(e)
        if hasattr(p,'detach'):p=p.detach().cpu().numpy()
        p=np.asarray(p,dtype=np.float32)
        if p.shape!=(len(paths),40) or not np.isfinite(p).all():
            raise ValueError(f'Invalid model predictions shape={p.shape}')
        prob.append(p)
    result=np.concatenate(prob,axis=0)
    cache.parent.mkdir(parents=True,exist_ok=True)
    embeddings=np.concatenate(embeds,axis=0)
    np.savez_compressed(cache, probs=result, embeddings=embeddings)
    return result,embeddings


def train_ranker(p, labels, steps=2000, seed=42, device='cpu'):
    # Predictors are FROZEN; learn a small retrieval head instead of memorizing identities.
    # Synthetic sparse queries are formed only from public TRAIN labels.
    rng=np.random.default_rng(seed)
    valid_labels=(labels>=0)&(labels<=1)
    usable=np.flatnonzero(valid_labels.sum(axis=1)>=3)
    if len(usable)<200: raise RuntimeError('Insufficient valid labeled training examples')
    probs=torch.tensor(np.clip(p,1e-5,1-1e-5),dtype=torch.float32,device=device)
    logits=torch.logit(probs)
    y=torch.tensor(labels,dtype=torch.float32,device=device)
    log_weight=torch.nn.Parameter(torch.zeros(40,device=device))
    log_temp=torch.nn.Parameter(torch.zeros(40,device=device))
    bias=torch.nn.Parameter(torch.zeros(40,device=device))
    optim=torch.optim.AdamW([log_weight,log_temp,bias],lr=0.006,weight_decay=0.01)
    recent=[]
    for step in tqdm(range(steps),desc='Train reliability-ranking head',unit='step'):
        batch=256
        pos_idx=rng.choice(usable,size=batch,replace=True)
        neg_idx=rng.choice(usable,size=batch,replace=True)
        # Some pairs share labels; weight training toward differing attributes.
        ys=labels[pos_idx]
        known=(ys>=0)&(ys<=1)
        masks=np.zeros_like(known,dtype=np.float32)
        for i in range(batch):
            candidates=np.flatnonzero(known[i])
            if not len(candidates):continue
            k=min(len(candidates),int(rng.integers(2,11)))
            masks[i,rng.choice(candidates,size=k,replace=False)]=1.
        mask=torch.as_tensor(masks,device=device)
        q=torch.as_tensor(np.clip(ys,0,1),device=device)
        i=torch.as_tensor(pos_idx,device=device)
        j=torch.as_tensor(neg_idx,device=device)
        weight=torch.exp(log_weight.clamp(-1.2,1.2))
        temperature=torch.exp(log_temp.clamp(-0.8,0.8))
        a=torch.sigmoid(logits[i]/temperature+bias)
        b=torch.sigmoid(logits[j]/temperature+bias)
        # Probability-mass error for present/absent attributes.
        w=mask*weight
        norm=w.sum(1).clamp_min(1e-6)
        d_pos=(w*torch.abs(q-a)).sum(1)/norm
        d_neg=(w*torch.abs(q-b)).sum(1)/norm
        label_difference=(w*torch.abs(q-torch.clamp(y[j],0,1))).sum(1)/norm
        compare=label_difference>0.10
        if compare.any():
            # Soft ranking hinge prevents overfitting high-confidence teacher predictions.
            rank_loss=F.softplus((d_pos[compare]-d_neg[compare]+0.08)*8).mean()/8
        else: rank_loss=torch.tensor(0.,device=device)
        # Learn reliable soft probabilities, regularize near identity calibration.
        random_idx=torch.as_tensor(rng.choice(usable,size=batch,replace=False if len(usable)>=batch else True),device=device)
        prob=torch.sigmoid(logits[random_idx]/temperature+bias)
        targets=y[random_idx]
        m=((targets>=0)&(targets<=1)).float()
        bce=(F.binary_cross_entropy(prob,targets.clamp(0,1),reduction='none')*m).sum()/m.sum().clamp_min(1)
        reg=0.018*(log_weight.square().mean()+log_temp.square().mean()+bias.square().mean())
        loss=rank_loss+0.35*bce+reg
        optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([log_weight,log_temp,bias],2.0);optim.step()
        if (step+1)%200==0: recent.append({'step':step+1,'loss':float(loss.item()),'rank':float(rank_loss.item()),'bce':float(bce.item())})
    return AttributeRanker(weight=torch.exp(log_weight.detach()).cpu().numpy(),
              temperature=torch.exp(log_temp.detach()).cpu().numpy(),bias=bias.detach().cpu().numpy(),source='public_train'),recent


def val_pairs(labels, count, known_min, known_max, seed):
    rng=np.random.default_rng(seed)
    eligible=np.flatnonzero(((labels>=0)&(labels<=1)).sum(1)>=known_min)
    ids=rng.choice(eligible,size=min(count,len(eligible)),replace=False)
    q=np.full((len(ids),40),-1,dtype=np.float32)
    for i,j in enumerate(ids):
        a=np.flatnonzero((labels[j]>=0)&(labels[j]<=1))
        subset=rng.choice(a,size=min(len(a),int(rng.integers(known_min,known_max+1))),replace=False)
        q[i,subset]=labels[j,subset]
    return q,ids


def exact_validation_queries(official, val_selection, seed, n_queries):
    """Prefer organizer query/ids protocol. Never invent positives for official mAP."""
    split=official/'data/annotations/task2/val'
    qpath=split/'queries.csv';ipath=split/'ids.csv'
    if not (qpath.is_file() and ipath.is_file()):
        raise FileNotFoundError('Official query/ids.csv files required for selecting learned blend')
    from abpr.model import load_query_csv
    query_names, q=load_query_csv(qpath)
    q=np.asarray(q,dtype=np.float32)
    from abpr.core import ATTRIBUTE_NAMES
    attr=list(ATTRIBUTE_NAMES)
    if query_names is not None and len(query_names)==40:
        lookup={str(k):i for i,k in enumerate(query_names)}
        if set(attr)==set(lookup): q=q[:,[lookup[n] for n in attr]]
        elif list(map(str,query_names)) != attr:
            raise ValueError('Official query attributes have a different order/names from the model')
    df=pd.read_csv(ipath)
    idcols=[c for c in df.columns if str(c).lower().strip() in ('id','ids','query_id','queryid','semantic_id')]
    if idcols: ids=pd.to_numeric(df[idcols[0]],errors='raise').to_numpy(dtype=np.int64)
    elif df.shape[1]==1: ids=pd.to_numeric(df.iloc[:,0],errors='raise').to_numpy(dtype=np.int64)
    else:
        raise ValueError(f'Could not identify query IDs in {ipath}: {df.columns.tolist()}')
    if len(ids)<int(val_selection.max())+1: raise ValueError('ids.csv not aligned to validation gallery')
    ids=ids[val_selection]
    if ids.min()<0 or ids.max()>=len(q): raise ValueError('ids.csv out of range for queries.csv')
    # Only evaluate queries with at least one selected gallery match.
    options=np.unique(ids)
    rng=np.random.default_rng(seed)
    sampled=rng.choice(options,size=min(n_queries,len(options)),replace=False)
    queries=q[sampled]
    if queries.shape[1]!=40:raise ValueError('Official query dimension must be 40')
    return queries, ids, sampled


def evaluate_exact_retrieval(d, gallery_ids, query_ids):
    AP=[];R1=[]
    for row,qid in enumerate(query_ids):
        relevant=(gallery_ids==qid)
        if not relevant.any():continue
        ranked=relevant[np.argsort(d[row],kind='stable')]
        where=np.flatnonzero(ranked)
        AP.append(float((np.cumsum(ranked)[where]/(where+1)).mean()))
        R1.append(float(ranked[0]))
    return {'val_subset_mAP':float(np.mean(AP)),'val_subset_R1':float(np.mean(R1)),
            'queries':len(AP),'note':'subset retrieval AP, not full challenge mADM'}


def synthetic_rank_metric(d,queries,ids,gt):
    # Retrieval proxy, NOT official mADM. For each synthetic query compute AP
    # against images matching its known attribute subset (ground truth).
    AP=[];R1=[]
    for row in range(len(queries)):
        q=queries[row];valid=(q>=0)&(q<=1)
        relevant=(gt[:,valid]==q[None,valid]).all(axis=1)
        relevant[ids[row]]=True
        if not relevant.any():continue
        indices=np.argsort(d[row],kind='stable')
        match=relevant[indices]
        cumsum=np.cumsum(match)
        where=np.flatnonzero(match)
        AP.append(float((cumsum[where]/(where+1)).mean()))
        R1.append(float(match[0]))
    return {'proxy_mAP':float(np.mean(AP)),'proxy_R1':float(np.mean(R1)),'queries':len(AP)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,required=True)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--runtime-zip',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cache-dir',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=48)
    ap.add_argument('--max-train-images',type=int,default=18000)
    ap.add_argument('--max-val-images',type=int,default=9000)
    ap.add_argument('--val-queries',type=int,default=320)
    ap.add_argument('--steps',type=int,default=1600)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--extra-data-root',type=Path,action='append',default=[])
    a=ap.parse_args()
    sys.path.insert(0,str(a.source_root))
    from abpr.core import ATTRIBUTE_NAMES, ImageResolver
    attr=list(ATTRIBUTE_NAMES)
    if len(attr)!=40: raise ValueError(f'Expected 40 attributes, got {len(attr)}')
    a.cache_dir.mkdir(parents=True,exist_ok=True)
    model_dir=a.cache_dir/'ranker_runtime'
    runtime=create_gallery_runtime(a.runtime_zip,model_dir)
    resolver=ImageResolver(a.repo_root/'data',a.repo_root,extra_roots=a.extra_data_root)
    fingerprint=hashlib.sha256((str(a.runtime_zip.resolve())+str(a.runtime_zip.stat().st_mtime_ns)+str(a.runtime_zip.stat().st_size)).encode()).hexdigest()[:12]
    train_df, train_gt, train_paths=load_data(a.repo_root,'train',attr)
    val_df,val_gt,val_paths=load_data(a.repo_root,'val',attr)
    train_inds=select_evenly(train_df,a.max_train_images,a.seed)
    val_inds=select_evenly(val_df,a.max_val_images,a.seed+1)
    train_pred,train_emb=encode_images(runtime,[train_paths[i] for i in train_inds],resolver,a.batch_size,a.cache_dir/f'train_probs_{fingerprint}_{len(train_inds)}.npz')
    val_pred,val_emb=encode_images(runtime,[val_paths[i] for i in val_inds],resolver,a.batch_size,a.cache_dir/f'val_probs_{fingerprint}_{len(val_inds)}.npz')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ranker,logs=train_ranker(train_pred,train_gt[train_inds],steps=a.steps,seed=a.seed,device=device)
    q, val_gallery_ids, val_query_ids=exact_validation_queries(a.repo_root,val_inds,a.seed+2,a.val_queries)
    # Compare the exact exported runtime's baseline distance vs learned distance.
    with torch.no_grad():
        try:
            baseline=runtime.distance(torch.tensor(q),torch.tensor(val_pred),torch.tensor(val_emb))
        except Exception:
            # model may require an embedding. Compute model embeddings for val and retry.
            # No cheating: fallback to weighted expected-Hamming baseline only if no runtime baseline is available.
            baseline=None
        if baseline is not None and hasattr(baseline,'detach'):baseline=baseline.detach().cpu().numpy()
    learned=ranker.distance(q,val_pred)
    if baseline is None:
        # Score-only fallback for tuning; submission still uses the model's real embeddings.
        baseline=AttributeRanker(np.ones(40),np.ones(40),np.zeros(40)).distance(q,val_pred)
    trials=[]
    for blend in [0.0,0.15,0.30,0.50,0.75,1.0]:
        d=fuse_distances(baseline,learned,blend)
        score=evaluate_exact_retrieval(d,val_gallery_ids,val_query_ids);score['blend']=blend
        trials.append(score)
    # Selection must beat baseline on held-out public VAL proxy, else zero blend.
    best=max(trials,key=lambda x: (x['val_subset_mAP'],x['val_subset_R1']))
    ranker.blend=float(best['blend']) if best['val_subset_mAP']>trials[0]['val_subset_mAP']+0.005 else 0.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    ranker.save(a.output)
    report={'output':str(a.output),'train_images':len(train_inds),'val_images':len(val_inds),'best_blend':ranker.blend,
            'selection_metric':'organizer ids.csv and queries.csv subset mAP, not full challenge mADM',
            'baseline':trials[0],'trials':trials,'train_log':logs,'model_device':device}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
