"""Contrastive image/query retrieval head for ABPR Task 2.

The backbone is pre-trained separately; this file learns a new image projection
and attribute query prototypes on TRAIN embeddings. No hidden labels are used.
All scores are smaller-is-better and no prior/hash fallback is provided.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

N_ATTR = 40


class ContrastiveRanker:
    def __init__(self, projection, pos_proto, neg_proto, blend=0.0,
                 temperature=0.08, source='public_train',
                 best_val_map=None):
        self.projection = np.asarray(projection, dtype=np.float32)
        self.pos_proto = np.asarray(pos_proto, dtype=np.float32)
        self.neg_proto = np.asarray(neg_proto, dtype=np.float32)
        self.blend = float(blend)
        self.temperature = float(temperature)
        self.source = str(source)
        self.best_val_map = None if best_val_map is None else float(best_val_map)
        if self.projection.ndim != 2 or self.projection.shape[1] < 8:
            raise ValueError('projection must have shape [embedding_dim, output_dim]')
        out_dim = self.projection.shape[1]
        if self.pos_proto.shape != (N_ATTR, out_dim) or self.neg_proto.shape != (N_ATTR, out_dim):
            raise ValueError('positive/negative prototype shapes must be [40, output_dim]')
        if not (0.0 <= self.blend <= 1.0):
            raise ValueError('contrastive blend must be between 0 and 1')
        if not all(np.isfinite(x).all() for x in (self.projection,self.pos_proto,self.neg_proto)):
            raise ValueError('non-finite contrastive checkpoint')

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as pack:
            return cls(pack['projection'],pack['pos_proto'],pack['neg_proto'],
                       float(pack['blend']), float(pack['temperature']),
                       str(pack['source']),
                       float(pack['best_val_map']) if 'best_val_map' in pack else None)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path, projection=self.projection, pos_proto=self.pos_proto,
                            neg_proto=self.neg_proto, blend=np.float32(self.blend),
                            temperature=np.float32(self.temperature),
                            source=np.asarray(self.source),
                            best_val_map=np.float32(self.best_val_map if self.best_val_map is not None else np.nan))

    @staticmethod
    def _norm(x):
        x=np.asarray(x, dtype=np.float32)
        return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

    def distance(self, queries, embeddings, batch_queries=128):
        q=np.asarray(queries,dtype=np.float32)
        e=np.asarray(embeddings,dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != N_ATTR:
            raise ValueError(f'Expected queries [Q,40], got {q.shape}')
        if e.ndim != 2 or e.shape[1] != self.projection.shape[0]:
            raise ValueError(f'Expected embedding shape [G,{self.projection.shape[0]}], got {e.shape}')
        if not (np.isfinite(q).all() and np.isfinite(e).all()):
            raise ValueError('Nonfinite queries/embeddings')
        img=self._norm(self._norm(e) @ self.projection)
        output=np.zeros((len(q),len(img)),dtype=np.float32)
        for start in range(0,len(q),batch_queries):
            qb=q[start:start+batch_queries]
            valid=(qb >= 0) & (qb <= 1)
            active=valid.astype(np.float32)
            positive=active*(qb > 0.5)
            negative=active*(qb <= 0.5)
            proto=positive @ self.pos_proto + negative @ self.neg_proto
            # Do not let a completely unknown query generate nonzero artifacts.
            proto=self._norm(proto)
            score=proto @ img.T
            d=(1.-score).astype(np.float32)
            empty=~valid.any(axis=1)
            if empty.any():d[empty]=1.0
            output[start:start+len(qb)]=d
        if not np.isfinite(output).all():
            raise RuntimeError('Contrastive distance contains NaN/inf')
        return output


def fuse_distances(base, contrastive, blend):
    base=np.asarray(base,dtype=np.float32)
    if blend <= 0.0:return base  # EXACT baseline, including the ranking scale.
    contrastive=np.asarray(contrastive,dtype=np.float32)
    if base.shape!=contrastive.shape:raise ValueError('Distance shape mismatch')
    if blend >= 1.0:return contrastive
    def norm(x):
        return (x-np.median(x,axis=1,keepdims=True)) / np.maximum(x.std(axis=1,keepdims=True),1e-5)
    ans=((1.0-blend)*norm(base)+blend*norm(contrastive)).astype(np.float32)
    if not np.isfinite(ans).all():raise RuntimeError('Nonfinite fused distances')
    return ans
