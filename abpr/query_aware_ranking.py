"""Probabilistic ABPR distances with a strictly zero-blend original baseline.

Training is not performed here: weights are TRAIN-label priors; mixing parameters
must be selected on FIT public validation queries and verified on unseen AUDIT
public validation queries. This module never uses image paths or IDs to rank.
"""
from __future__ import annotations
import numpy as np

N_ATTRIBUTES = 40

def _inputs(queries, probabilities, priors=None):
    q = np.asarray(queries, dtype=np.float32)
    p = np.asarray(probabilities, dtype=np.float32)
    if q.ndim != 2 or p.ndim != 2 or q.shape[1] != N_ATTRIBUTES or p.shape[1] != N_ATTRIBUTES:
        raise ValueError(f'Expected queries [Q,40], probabilities [G,40], got {q.shape}, {p.shape}')
    if not (np.isfinite(q).all() and np.isfinite(p).all()):
        raise ValueError('Non-finite query or probabilities')
    if not np.isin(q, (-1.0, 0.0, 1.0)).all():
        raise ValueError('Unknown attributes must be exactly -1; known values must be 0 or 1')
    prior = np.full(40, 0.5, dtype=np.float32) if priors is None else np.asarray(priors, np.float32)
    if prior.shape != (40,) or not np.isfinite(prior).all():
        raise ValueError('Priors must be 40 finite float values')
    return q, np.clip(p, 1e-5, 1 - 1e-5), np.clip(prior, 0.02, 0.98)


def score(queries, probabilities, *, method='loglik', negative_weight=1.0,
          rarity_power=0.0, priors=None, query_batch=64):
    """Smaller-is-better distances. No attribute-label oracle at inference."""
    q, p, prior = _inputs(queries, probabilities, priors)
    if method not in ('loglik', 'expected', 'margin'):
        raise ValueError(f'Unknown score method: {method}')
    if not 0 <= negative_weight <= 3 or not 0 <= rarity_power <= 1:
        raise ValueError('Weights outside supported ranges')
    pos_weight = np.power(1.0 / np.maximum(prior, 1e-3), rarity_power)
    neg_weight = np.power(1.0 / np.maximum(1 - prior, 1e-3), rarity_power)
    pos_weight /= np.mean(pos_weight)
    neg_weight /= np.mean(neg_weight)
    out = np.empty((len(q), len(p)), dtype=np.float32)
    if method == 'loglik':
        a = -np.log(p).T.copy()
        b = -np.log1p(-p).T.copy()
    elif method == 'expected':
        a = (1-p).T.copy()
        b = p.T.copy()
    else:
        a = (-p).T.copy()
        b = p.T.copy()
    for start in range(0, len(q), query_batch):
        qq = q[start:start+query_batch]
        wp = (qq == 1).astype(np.float32) * pos_weight
        wn = (qq == 0).astype(np.float32) * neg_weight * negative_weight
        total = (wp.sum(axis=1) + wn.sum(axis=1)).clip(1e-6)
        out[start:start+len(qq)] = ((wp @ a + wn @ b) / total[:,None]).astype(np.float32)
    return out


def _normalize_per_query(d):
    d = np.asarray(d, np.float32)
    # fixed within each query, cannot reorder baseline images
    mean = np.mean(d, axis=1, keepdims=True, dtype=np.float64)
    scale = np.std(d, axis=1, keepdims=True, dtype=np.float64)
    return ((d - mean) / np.maximum(scale, 1e-5)).astype(np.float32)


def fuse(baseline, novel, blend):
    baseline = np.asarray(baseline, np.float32)
    if float(blend) <= 0: return baseline.copy() # EXACT baseline reproduction
    novel = np.asarray(novel, np.float32)
    if baseline.shape != novel.shape:
        raise ValueError(f'Distance shape mismatch {baseline.shape} vs {novel.shape}')
    if not np.isfinite(baseline).all() or not np.isfinite(novel).all():
        raise ValueError('Non-finite distances')
    if not (0.0 <= float(blend) <= 1.0): raise ValueError('Invalid blend')
    return ((1-blend)*_normalize_per_query(baseline) + blend*_normalize_per_query(novel)).astype(np.float32)


def retrieval_metrics(distances, qids, gallery_ids):
    """Exact official semantic-ID AP/Rank-1 proxy for held-out public query IDs.

    This is NOT the challenge's domain-averaged mADM.
    """
    distances = np.asarray(distances)
    qids = np.asarray(qids, np.int64)
    gids = np.asarray(gallery_ids, np.int64)
    if distances.shape != (len(qids),len(gids)):
        raise ValueError('Scoring shape mismatch')
    aps=[]; top1=[]; skipped=0
    for i,qid in enumerate(qids):
        nrel=int(np.sum(gids==qid))
        if nrel==0: skipped+=1;continue
        ind=np.argsort(distances[i], kind='stable')
        relevance=(gids[ind]==qid)
        hit=np.flatnonzero(relevance)
        aps.append(float(((np.arange(1,len(hit)+1))/(hit+1)).sum()/nrel))
        top1.append(float(relevance[0]))
    return {'mAP':float(np.mean(aps)) if aps else 0.,'Rank-1':float(np.mean(top1)) if top1 else 0.,
            'evaluated_queries':len(aps),'missing_positive_queries':skipped}
