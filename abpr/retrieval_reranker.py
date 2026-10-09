"""Lightweight learned query-image distance for Track 2 ABPR.

Train exclusively on public TRAIN image annotations/predictions. Tune ensemble
strength exclusively on official public VAL. All 40 attribute names remain in
the order used by the existing ABPRRuntime.
"""
from __future__ import annotations
import numpy as np

N_ATTR = 40


def sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


class AttributeRanker:
    def __init__(self, weight, temperature, bias, blend=0., source="train"):
        self.weight = np.asarray(weight, dtype=np.float32)
        self.temperature = np.asarray(temperature, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.blend = float(blend)
        self.source = str(source)
        if any(a.shape != (N_ATTR,) for a in (self.weight, self.temperature, self.bias)):
            raise ValueError('ranker parameters must each have 40 values')
        if not np.isfinite(self.weight).all() or np.any(self.weight <= 0):
            raise ValueError('attribute weights must be positive and finite')
        if not np.isfinite(self.temperature).all() or np.any(self.temperature <= 0):
            raise ValueError('attribute temperature must be positive and finite')
        if not 0. <= self.blend <= 1.:
            raise ValueError('blend must be within [0,1]')

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z['weight'], z['temperature'], z['bias'],
                       blend=float(z['blend']), source=str(z['source']))

    def save(self, path):
        np.savez_compressed(path, weight=self.weight, temperature=self.temperature,
                            bias=self.bias, blend=np.float32(self.blend),
                            source=np.asarray(self.source))

    def calibrated(self, p):
        p = np.asarray(p, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != N_ATTR:
            raise ValueError(f'gallery probabilities must have 40 columns: {p.shape}')
        p = np.clip(p, 1e-5, 1.-1e-5)
        logits = np.log(p) - np.log1p(-p)
        return sigmoid(logits / self.temperature[None] + self.bias[None]).astype(np.float32)

    def distance(self, queries, gallery_probs, batch_queries=128):
        p = self.calibrated(gallery_probs)
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != N_ATTR:
            raise ValueError('query dimension must equal 40')
        out = np.empty((len(q), len(p)), dtype=np.float32)
        # Query values -1 are unknown and MUST NOT contribute to the distance.
        for start in range(0, len(q), batch_queries):
            qs = q[start:start+batch_queries]
            valid = (qs >= 0) & (qs <= 1)
            weights = valid * self.weight[None]
            denom = np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
            qq = np.clip(qs, 0, 1)
            # Matrix formulation avoids allocation of [Q,G,40] ~ many GiB.
            # |q-p| = q(1-p) + (1-q)p, for binary queries.
            positive = (weights * qq) @ (1-p).T
            negative = (weights * (1-qq)) @ p.T
            out[start:start+len(qs)] = (positive + negative) / denom
        if not np.isfinite(out).all():
            raise RuntimeError('Non-finite learned retrieval distance')
        return out


def fuse_distances(base, learned, blend):
    """Per-query scale normalization; blend=0 exactly preserves the baseline."""
    b = np.asarray(base, dtype=np.float32)
    if blend <= 0:
        return b
    l = np.asarray(learned, dtype=np.float32)
    if b.shape != l.shape:
        raise ValueError(f'base vs ranker shapes differ: {b.shape}, {l.shape}')
    if blend >= 1:
        return l
    def normalize(x):
        center = np.median(x, axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True).clip(1e-5)
        return (x-center)/sd
    z = (1-blend)*normalize(b) + blend*normalize(l)
    if not np.isfinite(z).all():
        raise RuntimeError('Non-finite fused retrieval distance')
    return z.astype(np.float32)
