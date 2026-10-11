"""Inference-only, query-conditional rank-local contradiction penalty.

Inputs: baseline distances [queries, gallery] (smaller is better), a predicted-attribute
contradiction matrix derived WITHOUT gallery ground truth, and rank-only hyperparameters.
Never use GIDS or VAL_LABELS to construct the candidate distances.
"""
from __future__ import annotations
import numpy as np


def topk_indices(base: np.ndarray, max_k: int) -> np.ndarray:
    d = np.asarray(base, dtype=np.float32)
    if d.ndim != 2 or d.shape[1] < 2 or not np.isfinite(d).all():
        raise ValueError('Invalid baseline distance matrix')
    k = min(int(max_k), d.shape[1]); assert k >= 2
    out = np.empty((d.shape[0], k), dtype=np.int32)
    for i in range(d.shape[0]):
        part = np.argpartition(d[i], k-1)[:k]
        out[i] = part[np.argsort(d[i, part], kind='stable')]
    return out


def rank_local_score(base: np.ndarray, predicted_penalty: np.ndarray,
                     sorted_top_indices: np.ndarray, *, top_k: int,
                     strength: float, tau: float, cap: float = 0.65) -> np.ndarray:
    """Keep baseline scores outside top_k, penalize predicted contradictions locally.

    Local scale and damped adjustments depend only on the model's distances/predictions.
    The same function can run on unseen gallery if runtime embeddings are available.
    """
    b = np.asarray(base, dtype=np.float32)
    p = np.asarray(predicted_penalty, dtype=np.float32)
    idx = np.asarray(sorted_top_indices)
    if b.shape != p.shape or b.ndim != 2 or idx.ndim != 2 or idx.shape[0] != len(b):
        raise ValueError('Baseline/penalty/top-k shape mismatch')
    if not np.isfinite(b).all() or not np.isfinite(p).all():
        raise ValueError('NaN/Inf in distance or penalty')
    k = min(int(top_k), idx.shape[1]);
    if k < 2 or not (0 <= float(strength) <= 5) or float(tau) <= 0 or float(cap) <= 0:
        raise ValueError('Invalid rank-local parameters')
    if strength == 0:
        return b.copy()
    out = b.copy()
    rows = np.arange(len(b))
    chosen = idx[:, :k]
    db = np.take_along_axis(b, chosen, axis=1)
    pb = np.take_along_axis(p, chosen, axis=1)
    local_std = np.maximum(np.std(db, axis=1, keepdims=True), 1e-6)
    spread = np.maximum(db[:, -1:] - db[:, :1], 1e-6)
    # Prediction-based penalty relative to the least contradictory top-k image.
    excess = np.maximum(0.0, pb - np.min(pb, axis=1, keepdims=True))
    denom = np.maximum(np.quantile(excess, 0.90, axis=1, keepdims=True), 0.025)
    relative_risk = np.minimum(excess / denom, 3.0)
    margin = np.maximum(db - db[:, :1], 0.0)
    locality = np.exp(-margin / (float(tau) * local_std))
    amount = float(strength) * local_std * relative_risk * locality
    amount = np.minimum(amount, float(cap) * spread)
    out[rows[:, None], chosen] = db + amount.astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite reranked distances')
    return out


def perquery_ranking_stats(distances, ids, gallery_ids):
    """AP, reciprocal rank and R1 by exact semantic-ID relevance (evaluation only)."""
    dist = np.asarray(distances, np.float32)
    ids = np.asarray(ids, np.int64)
    gallery_ids = np.asarray(gallery_ids, np.int64)
    if dist.shape != (len(ids), len(gallery_ids)):
        raise ValueError('Evaluation mismatch')
    ap = np.zeros(len(ids), np.float64)
    rank1 = np.zeros(len(ids), np.float64)
    first_rank = np.zeros(len(ids), np.int64)
    count = np.zeros(len(ids), np.int64)
    for i, qid in enumerate(ids):
        positive = gallery_ids == qid
        n = int(positive.sum()); count[i] = n
        if n == 0:
            continue
        order = np.argsort(dist[i], kind='stable')
        hit = positive[order]
        positions = np.flatnonzero(hit) + 1
        ap[i] = np.mean(np.arange(1, n + 1) / positions)
        first_rank[i] = int(positions[0]); rank1[i] = float(hit[0])
    return dict(ap=ap, rank1=rank1, first_rank=first_rank, positive_count=count)


def paired_bootstrap_perquery(before, after, repeats=400, seed=2027):
    """Simple paired bootstrap for AP and Rank1 only; NOT a CI for mADM."""
    rng = np.random.default_rng(seed)
    out = {}
    for metric in ('ap', 'rank1'):
        diffs = np.asarray(after[metric], np.float64) - np.asarray(before[metric], np.float64)
        idx = rng.integers(0, len(diffs), size=(repeats, len(diffs)))
        boot = diffs[idx].mean(axis=1)
        out[metric] = {'delta': float(diffs.mean()),
                       'bootstrap_ci_95': [float(np.quantile(boot, .025)),
                                           float(np.quantile(boot, .975))],
                       'p_bootstrap_positive': float((boot > 0).mean())}
    return out
