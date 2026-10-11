"""Source-only domain calibration and optional predicted-attribute contradiction reranking.

No validation gallery labels or semantic IDs are accepted in scorer functions.
Query labels (attribute vectors) are inputs of challenge, not ground-truth image labels.
"""
from __future__ import annotations
import numpy as np

DOMAINS = ('Market1501', 'PA100k', 'PETA')


def domain_from_path(s):
    t = str(s).lower().replace('\\', '/')
    if 'market1501' in t or 'market-1501' in t:
        return 0
    if 'pa100k' in t:
        return 1
    if 'peta' in t:
        return 2
    return -1


def fit_domain_nb(train_attrs, train_domains, smoothing=30.):
    """Bernoulli Naive Bayes calibrated on TRAIN only, not VAL.

    The input labels may be 0,1,-1 (missing). Domain weights are balanced.
    """
    y = np.asarray(train_attrs, dtype=np.float32)
    domains = np.asarray(train_domains, dtype=np.int64)
    if y.ndim != 2 or y.shape[1] != 40 or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError('Expected TRAIN annotations of shape [N,40] with -1/0/1')
    if len(domains) != len(y) or not np.isin(domains, [0,1,2]).all():
        raise ValueError('Unexpected TRAIN domain IDs')
    p = np.empty((3,40), np.float32)
    n = np.empty((3,), np.int64)
    for d in range(3):
        v = y[domains==d]
        if len(v)==0: raise ValueError(f'No TRAIN records for domain {d}')
        n[d] = len(v)
        positives = (v==1).sum(axis=0)
        valid = ((v==1) | (v==0)).sum(axis=0)
        p[d] = (positives+smoothing*0.5)/(valid+smoothing)
    return {'prob':p, 'counts':n}


def predict_domains(query, model, temp=3.0):
    """P(source domain | query), uniform prior, conservative temperature."""
    q = np.asarray(query, np.float32)
    if q.ndim != 2 or q.shape[1]!=40 or not np.isin(q,[-1,0,1]).all():
        raise ValueError('Expected query [Q,40] with -1/0/1')
    p = np.clip(np.asarray(model['prob'],np.float32), 1e-4,1-1e-4)
    yes=(q==1).astype(np.float32)
    no=(q==0).astype(np.float32)
    ll = yes @ np.log(p).T + no @ np.log1p(-p).T
    # Mean per known attribute to stop the 40-attribute vectors from overconcentrating.
    ll /= np.maximum((yes+no).sum(axis=1,keepdims=True),1.)
    ll /= float(temp)
    ll -= ll.max(axis=1,keepdims=True)
    s=np.exp(ll)
    return s/np.maximum(s.sum(axis=1,keepdims=True),1e-9)


def standardize(base, eps=1e-5):
    x=np.asarray(base,np.float32)
    mu=x.mean(axis=1,keepdims=True,dtype=np.float64)
    sd=x.std(axis=1,keepdims=True,dtype=np.float64)
    return ((x-mu)/np.maximum(sd,eps)).astype(np.float32)


def domain_adjusted(base, query, gallery_domains, nb_model, alpha=0.0,beta=0.0,
                    temp=3.0):
    """Return lower-is-better distance. Does not accept relevance labels."""
    dist=np.asarray(base,np.float32)
    q=np.asarray(query,np.float32)
    domains=np.asarray(gallery_domains)
    if dist.shape != (len(q),len(domains)) or not np.isfinite(dist).all():
        raise ValueError('Bad distance shape or nonfinite distance values')
    if alpha==0 and beta==0:
        return dist.copy()
    out=standardize(dist)
    if alpha:
        local=np.zeros_like(out)
        for d in range(3):
            mask=domains==d
            if not mask.any():continue
            sub=dist[:,mask]
            local[:,mask]=standardize(sub)
        # Leave unknown-domain images on global scale.
        local[:,domains<0]=out[:,domains<0]
        out=(1-float(alpha))*out+float(alpha)*local
    if beta:
        posterior=predict_domains(q,nb_model,temp=temp)
        correction=-np.log(np.clip(posterior,1e-7,1.))
        correction-=correction.mean(axis=1,keepdims=True)
        for d in range(3):
            mask=domains==d
            if mask.any():out[:,mask] += float(beta)*correction[:,d,None]
    return out.astype(np.float32)


def train_confidence_gate(train_feats,train_labels,weight,bias,min_precision=.90,
                          min_support=12):
    """Train-only thresholds to flag confidently wrong predictions.

    Note: if adapter was fitted on these rows, reliability can be optimistic;
    kept as exploratory only. No VAL labels enter thresholds or inference.
    """
    features=np.asarray(train_feats,np.float32)
    y=np.asarray(train_labels,np.float32)
    w=np.asarray(weight,np.float32)
    b=np.asarray(bias,np.float32)
    if features.shape[0]!=len(y) or y.shape[1]!=40 or features.shape[1]!=w.shape[0]:
        raise ValueError('Train features/labels/head dimension mismatch')
    logits=features@w+b
    p=1/(1+np.exp(-np.clip(logits,-50,50)))
    hi=np.ones(40,np.float32)*1.01
    lo=np.ones(40,np.float32)*(-.01)
    usable_hi=np.zeros(40,dtype=bool)
    usable_lo=np.zeros(40,dtype=bool)
    precision_hi=np.full(40,np.nan,np.float32)
    precision_lo=np.full(40,np.nan,np.float32)
    # Hold-out labels only TRAIN, threshold gates on prediction confidence.
    for j in range(40):
        known=np.isin(y[:,j],[0,1]); pj=p[known,j]; lj=y[known,j]
        for t in np.linspace(.6,.98,20):
            picked=pj>=t
            if picked.sum()>=min_support and (lj[picked]==1).mean()>=min_precision:
                hi[j]=t;usable_hi[j]=True;precision_hi[j]=(lj[picked]==1).mean();break
        for t in np.linspace(.4,.02,20):
            picked=pj<=t
            if picked.sum()>=min_support and (lj[picked]==0).mean()>=min_precision:
                lo[j]=t;usable_lo[j]=True;precision_lo[j]=(lj[picked]==0).mean();break
    return {'hi':hi,'lo':lo,'usable_hi':usable_hi,'usable_lo':usable_lo,
            'precision_hi':precision_hi,'precision_lo':precision_lo}


def contradiction_penalty(queries, image_features, weight,bias,gate,
                          attr_weights=None,query_batch=48):
    """Positive penalty only when predicted feature contradicts a known query attr.
    No ground-truth gallery labels; works on unseen gallery images.
    """
    q=np.asarray(queries,np.float32)
    features=np.asarray(image_features,np.float32)
    z=features@np.asarray(weight,np.float32)+np.asarray(bias,np.float32)
    prob=1/(1+np.exp(-np.clip(z,-50,50)))
    hi=np.asarray(gate['hi'],np.float32)
    lo=np.asarray(gate['lo'],np.float32)
    # Query says 0 but image predicted 1 -> mismatch; query says 1, predicted 0 -> mismatch.
    mismatch_zero=np.maximum(0,prob-hi[None,:]) / np.maximum(1-hi[None,:],1e-3)
    mismatch_one=np.maximum(0,lo[None,:]-prob) / np.maximum(lo[None,:],1e-3)
    mismatch_zero[:,~np.asarray(gate['usable_hi'],bool)]=0
    mismatch_one[:,~np.asarray(gate['usable_lo'],bool)]=0
    weights=np.ones(40,np.float32) if attr_weights is None else np.asarray(attr_weights,np.float32)
    scores=np.empty((len(q),len(features)),np.float32)
    for start in range(0,len(q),query_batch):
        end=min(start+query_batch,len(q))
        neg=(q[start:end]==0).astype(np.float32)*weights
        pos=(q[start:end]==1).astype(np.float32)*weights
        norm=np.maximum((neg+pos).sum(axis=1,keepdims=True),1.)
        scores[start:end]=(neg@mismatch_zero.T+pos@mismatch_one.T)/norm
    return scores


def combine_penalty(base, penalty, gamma):
    """Add score proportional to predicted contradiction; no gallery labels."""
    base=np.asarray(base,np.float32)
    if gamma==0:return base.copy()
    penalty=np.asarray(penalty,np.float32)
    if base.shape!=penalty.shape:raise ValueError('Penalty must match distance shape')
    # Standardize positive penalty to robust scale, avoid exploding on all-constant rows.
    scale=np.maximum(penalty.std(axis=1,keepdims=True),0.02)
    return (standardize(base)+float(gamma)*penalty/scale).astype(np.float32)


def apply_candidate(base, query, domains, model, config, penalty=None):
    kind=config['method']
    if kind=='domain':
        return domain_adjusted(base,query,domains,model,
                               alpha=config.get('alpha',0),beta=config.get('beta',0),
                               temp=config.get('temp',3))
    if kind=='gate':
        if penalty is None:raise ValueError('gate candidate requires predicted image features')
        return combine_penalty(base,penalty,config['gamma'])
    if kind=='domain_gate':
        if penalty is None:raise ValueError('gate candidate requires predicted image features')
        out=domain_adjusted(base,query,domains,model,alpha=config.get('alpha',0),
                             beta=config.get('beta',0),temp=config.get('temp',3))
        return combine_penalty(out,penalty,config['gamma'])
    raise ValueError('Unknown method')
