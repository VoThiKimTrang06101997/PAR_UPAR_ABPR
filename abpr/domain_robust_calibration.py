"""Domain-robust post-hoc attribute calibration and query-aware retrieval.

No test labels, test domain flags, or path identities enter inference. All distances
are smaller-is-better; zero-blend reproduces the original runtime distance exactly.
"""
from __future__ import annotations
import numpy as np

N_ATTR = 40


def ensure(q, p, priors):
    q=np.asarray(q,np.float32); p=np.asarray(p,np.float32); prior=np.asarray(priors,np.float32)
    if q.ndim!=2 or q.shape[1]!=N_ATTR or p.ndim!=2 or p.shape[1]!=N_ATTR: raise ValueError('Expected [Q,40] queries and [G,40] probabilities')
    if prior.shape!=(N_ATTR,): raise ValueError('Priors must have shape (40,)')
    if not np.isin(q,(-1,0,1)).all(): raise ValueError('Query attributes must be -1/0/1')
    if not (np.isfinite(p).all() and np.isfinite(prior).all()): raise ValueError('Nonfinite probabilities or priors')
    return q,np.clip(p,1e-5,1.-1e-5),np.clip(prior,0.01,0.99)


def calibrate_probabilities(probabilities, *, scale, bias, priors, shrink=0.0):
    p=np.asarray(probabilities,np.float32)
    a=np.asarray(scale,np.float32); b=np.asarray(bias,np.float32); prior=np.asarray(priors,np.float32)
    if p.ndim!=2 or p.shape[1]!=40 or a.shape!=(40,) or b.shape!=(40,) or prior.shape!=(40,):
        raise ValueError('Invalid calibration parameter dimensions')
    if not 0.0<=float(shrink)<=0.4:raise ValueError('Shrink must be in [0,0.4]')
    if not np.isfinite(p).all() or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite input')
    logit=np.log(np.clip(p,1e-5,1.-1e-5))-np.log1p(-np.clip(p,1e-5,1.-1e-5))
    x=np.clip(logit*a[None,:]+b[None,:],-30.,30.)
    adjusted=(1.0/(1.0+np.exp(-x))).astype(np.float32)
    return ((1.-shrink)*adjusted+shrink*prior[None,:]).astype(np.float32)


def query_distance(queries, probabilities, *, method='expected', priors, negative_weight=1.0,
                   rarity_power=0.0, query_batch=48):
    q,p,prior=ensure(queries,probabilities,priors)
    if method not in ('expected','loglik'):raise ValueError('Unknown method')
    if not 0<=negative_weight<=2 or not 0<=rarity_power<=1:raise ValueError('Invalid ranking weights')
    pw=(1/np.maximum(prior,.01))**rarity_power
    nw=(1/np.maximum(1-prior,.01))**rarity_power
    pw=pw/np.mean(pw);nw=nw/np.mean(nw)
    a=((1-p) if method=='expected' else -np.log(p)).T
    b=(p if method=='expected' else -np.log1p(-p)).T
    distances=np.empty((len(q),len(p)),dtype=np.float32)
    for s in range(0,len(q),query_batch):
        qq=q[s:s+query_batch]
        pos=(qq==1).astype(np.float32)*pw
        neg=(qq==0).astype(np.float32)*nw*negative_weight
        den=(pos.sum(axis=1)+neg.sum(axis=1)).clip(1e-6)
        distances[s:s+len(qq)]=(pos@a+neg@b)/den[:,None]
    return distances


def fuse_distances(original, robust, blend):
    baseline=np.asarray(original,np.float32)
    if not np.isfinite(baseline).all() or baseline.ndim!=2:raise ValueError('Invalid baseline distance')
    if not 0<=float(blend)<=1:raise ValueError('Invalid fusion coefficient')
    if float(blend)==0:return baseline.copy()
    alternative=np.asarray(robust,np.float32)
    if alternative.shape!=baseline.shape or not np.isfinite(alternative).all():raise ValueError('Invalid alternative distance')
    def norm(x):
        mean=np.mean(x,axis=1,keepdims=True,dtype=np.float64)
        sd=np.std(x,axis=1,keepdims=True,dtype=np.float64)
        return ((x-mean)/np.maximum(sd,1e-5)).astype(np.float32)
    return ((1.-blend)*norm(baseline)+blend*norm(alternative)).astype(np.float32)


def semantic_metrics(distances, query_ids, gallery_ids):
    d=np.asarray(distances,np.float32);ids=np.asarray(query_ids,np.int64);g=np.asarray(gallery_ids,np.int64)
    if d.shape!=(len(ids),len(g)):raise ValueError('Distance/id shape mismatch')
    if not np.isfinite(d).all():raise ValueError('Nonfinite distances')
    aps=[];r1=[];r5=[];r10=[];inp=[]
    for i,qid in enumerate(ids):
        relevant=(g==qid)
        n=int(relevant.sum())
        if n==0:continue
        order=np.argsort(d[i],kind='stable')
        pos=np.flatnonzero(relevant[order])+1
        aps.append(float(np.mean(np.arange(1,n+1)/pos)))
        r1.append(float(pos[0]<=1));r5.append(float(pos[0]<=5));r10.append(float(pos[0]<=10))
        inp.append(float(n/pos[-1]))
    return dict(mAP=float(np.mean(aps)) if aps else 0.,
                Rank_1=float(np.mean(r1)) if r1 else 0.,
                Rank_5=float(np.mean(r5)) if r5 else 0.,
                Rank_10=float(np.mean(r10)) if r10 else 0.,
                mINP=float(np.mean(inp)) if inp else 0.,
                valid_queries=len(aps))


def domain_macro(distances, query_ids, gallery_ids, domains, num_domains=3):
    """Semantic retrieval per named source domain, with a domain-restricted gallery.
    This is a diagnostic, NOT official hidden challenge mADM.
    """
    g=np.asarray(gallery_ids,np.int64);domain=np.asarray(domains,np.int32);qid=np.asarray(query_ids,np.int64)
    d=np.asarray(distances,np.float32)
    if len(g)!=len(domain) or d.shape!=(len(qid),len(g)):raise ValueError('Domain metric shape mismatch')
    per_domain={}
    for k in range(num_domains):
        mask=(domain==k)
        if not np.any(mask):continue
        qmask=np.isin(qid,np.unique(g[mask]))
        if not np.any(qmask):continue
        per_domain[str(k)]=semantic_metrics(d[np.ix_(qmask,mask)],qid[qmask],g[mask])
    keys=['mAP','Rank_1','Rank_5','Rank_10','mINP']
    macro={k:float(np.mean([m[k] for m in per_domain.values()])) for k in keys} if per_domain else {}
    worst={k:float(min([m[k] for m in per_domain.values()])) for k in keys} if per_domain else {}
    return {'macro':macro,'worst_domain':worst,'per_domain':per_domain}
