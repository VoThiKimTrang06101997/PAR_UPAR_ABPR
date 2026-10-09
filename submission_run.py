"""Track-2 submission. Uses trained pixels, never a prior/hash fallback.

Official public contract:
  load_model() -> None
  predict_attributes(gallery, attribute_names) -> probabilities [G, A]
  rank_gallery(sample) -> {'distances': float32 ndarray [Q, G]}

A submission that cannot execute the model must FAIL with its actual exception,
not silently emit a constant/gallery-name ranking that lowers the score.
"""
from __future__ import annotations
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / 'assets' / 'model.pt'
_MODEL = None

CANONICAL_NAMES = [
    'Age-Young','Age-Adult','Age-Old','Gender-Female',
    'Hair-Length-Short','Hair-Length-Long','Hair-Length-Bald','UpperBody-Length-Short',
    'UpperBody-Color-Black','UpperBody-Color-Blue','UpperBody-Color-Brown','UpperBody-Color-Green',
    'UpperBody-Color-Grey','UpperBody-Color-Orange','UpperBody-Color-Pink','UpperBody-Color-Purple',
    'UpperBody-Color-Red','UpperBody-Color-White','UpperBody-Color-Yellow','UpperBody-Color-Other',
    'LowerBody-Length-Short','LowerBody-Color-Black','LowerBody-Color-Blue','LowerBody-Color-Brown',
    'LowerBody-Color-Green','LowerBody-Color-Grey','LowerBody-Color-Orange','LowerBody-Color-Pink',
    'LowerBody-Color-Purple','LowerBody-Color-Red','LowerBody-Color-White','LowerBody-Color-Yellow',
    'LowerBody-Color-Other','LowerBody-Type-Trousers&Shorts','LowerBody-Type-Skirt&Dress',
    'Accessory-Backpack','Accessory-Bag','Accessory-Glasses-Normal','Accessory-Glasses-Sun','Accessory-Hat',
]


def _log(message):
    print('[ABPR inference] ' + str(message), file=sys.stderr, flush=True)


def _norm(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).lower().replace('&','and'))


def _gallery_records(items):
    if items is None:
        raise ValueError('sample.gallery is missing')
    if hasattr(items, 'to_dict') and not isinstance(items, (dict, list, tuple, np.ndarray)):
        try: items=items.to_dict('records')
        except TypeError: pass
    records=[]
    for item in items:
        if isinstance(item, dict):
            d=dict(item)
            if not d.get('image_path'):
                for key in ('path','image','filename','file','img_path'):
                    if d.get(key) is not None:
                        d['image_path']=os.fspath(d[key]);break
            if not d.get('image_path'):
                raise ValueError(f'Gallery dict lacks image path keys: {list(item)[:12]}')
        elif isinstance(item,(str,os.PathLike)):
            d={'image_path':os.fspath(item)}
        else:
            raise TypeError(f'Unsupported gallery entry: {type(item).__name__}')
        records.append(d)
    return records


def _cols(names, target):
    names=[str(n) for n in names]
    if len(names)!=len(set(map(_norm,names))):
        raise ValueError('Duplicate normalized attribute names in incoming sample')
    lookup={_norm(n):i for i,n in enumerate(names)}
    missing=[n for n in target if _norm(n) not in lookup]
    if missing: raise ValueError(f'Missing challenge attributes: {missing}')
    return [lookup[_norm(n)] for n in target]


def load_model():
    global _MODEL
    if _MODEL is not None: return
    if os.environ.get('ABPR_FORCE_FALLBACK')=='1':
        raise RuntimeError('Prior/hash fallback is prohibited; trained model required')
    if not MODEL_PATH.is_file(): raise FileNotFoundError(f'Trained checkpoint missing: {MODEL_PATH}')
    # Lazy import: if dependencies fail on worker, report real traceback.
    from abpr_runtime import ABPRRuntime
    _MODEL=ABPRRuntime(MODEL_PATH)
    device=str(getattr(_MODEL,'device','unknown'))
    _log(f'TRAINED checkpoint loaded; device={device}; weights={MODEL_PATH.stat().st_size/1024**2:.1f} MiB')


def _predict_model(gallery):
    load_model()
    if len(gallery)==0: raise ValueError('Empty gallery')
    start=time.monotonic()
    output=_MODEL.encode_gallery(gallery)
    if isinstance(output, tuple):
        probs, embeddings = output[:2]
    else: raise TypeError('ABPRRuntime.encode_gallery must return (probabilities, embeddings)')
    if hasattr(probs,'detach'): probs=probs.detach().cpu().numpy()
    probs=np.asarray(probs,dtype=np.float32)
    if probs.shape!=(len(gallery),len(CANONICAL_NAMES)):
        raise ValueError(f'Bad probabilities shape {probs.shape} expected ({len(gallery)},40)')
    if not np.isfinite(probs).all(): raise ValueError('Non-finite probabilities')
    spread=float(np.std(probs,axis=0).mean())
    if len(gallery)>=4 and spread<1e-7:
        raise RuntimeError('DEGENERATE model: predicted attributes are constant over gallery')
    _log(f'Encoded {len(gallery)} REAL gallery images in {time.monotonic()-start:.1f}s; avg per-attribute std={spread:.5f}')
    return probs,embeddings


def predict_attributes(gallery, attribute_names):
    records=_gallery_records(gallery)
    probs,_=_predict_model(records)
    model_names=list(getattr(_MODEL,'attribute_names', CANONICAL_NAMES))
    if len(model_names)!=probs.shape[1]: raise ValueError('Runtime attribute_names length mismatch')
    return probs[:,_cols(model_names,attribute_names)]


def rank_gallery(sample):
    if not isinstance(sample,dict): raise TypeError('rank_gallery expects a dict sample')
    gallery=_gallery_records(sample.get('gallery'))
    names=sample.get('attribute_names',CANONICAL_NAMES)
    names=list(names)
    queries=np.asarray(sample['queries'],dtype=np.float32)
    if queries.ndim==1: queries=queries[None,:]
    if queries.ndim!=2 or queries.shape[1]!=len(names):
        raise ValueError(f'Query shape {queries.shape} incompatible with {len(names)} names')
    probs,embeddings=_predict_model(gallery)
    model_names=list(getattr(_MODEL,'attribute_names',CANONICAL_NAMES))
    indices=_cols(names,model_names)
    queries_model=queries[:,indices]
    # Runtime expects tensors; encode_gallery returns NumPy after our validation.
    # Previous submissions failed on numpy.ndarray.float() without this bridge.
    import torch
    tensor_probs=torch.as_tensor(probs,dtype=torch.float32,device='cpu')
    tensor_embeddings=(torch.as_tensor(embeddings,dtype=torch.float32,device='cpu')
                       if embeddings is not None else None)
    try:
        scores=_MODEL.distance(queries_model,tensor_probs,tensor_embeddings)
    except TypeError as exc:
        # Some provider runtimes use NumPy matrix multiplication rather than
        # tensor APIs. Retry NumPy only for an explicit NumPy/Tensor mixing
        # exception; never swallow genuine model errors.
        msg=str(exc)
        if 'numpy.ndarray' not in msg or 'Tensor' not in msg:
            raise
        scores=_MODEL.distance(queries_model,probs,embeddings)
    if hasattr(scores,'detach'): scores=scores.detach().cpu().numpy()
    scores=np.asarray(scores,dtype=np.float32)

    # Optional TRAIN-supervised attribute-ranker. If public VAL does not show a
    # measurable gain, artifact has blend=0 and the baseline remains exact.
    ranker_file=HERE/'assets'/'ranker.npz'
    if ranker_file.is_file():
        from retrieval_reranker import AttributeRanker, fuse_distances
        ranker=AttributeRanker.load(ranker_file)
        if ranker.blend>0:
            learned=ranker.distance(queries_model,probs)
            scores=fuse_distances(scores,learned,ranker.blend)
            _log(f'Learned attribute-ranker applied (blend={ranker.blend:.2f})')
        else:
            _log('Ranker public-VAL gate preferred baseline; blend=0')
    # TRAIN-supervised multi-label contrastive retrieval head. The weights were
    # fit on public TRAIN; its blend was selected using public VAL only.
    contrastive_file=HERE/'assets'/'contrastive.npz'
    if contrastive_file.is_file():
        from contrastive_reranker import ContrastiveRanker, fuse_distances as fuse_contrastive
        contrastive=ContrastiveRanker.load(contrastive_file)
        if contrastive.blend>0:
            if embeddings is None:
                raise RuntimeError('Contrastive ranker requires image embeddings from trained model')
            if hasattr(embeddings,'detach'):
                np_embeddings=embeddings.detach().cpu().numpy()
            else:
                np_embeddings=np.asarray(embeddings,dtype=np.float32)
            enhanced=contrastive.distance(queries_model,np_embeddings)
            scores=fuse_contrastive(scores,enhanced,contrastive.blend)
            _log(f'TRAIN-supervised contrastive ranking applied (blend={contrastive.blend:.2f})')
        else:
            _log('Contrastive public-VAL gate preferred baseline; blend=0')
    expected=(len(queries),len(gallery))
    if scores.shape!=expected: raise RuntimeError(f'Unexpected distance shape {scores.shape}, expected {expected}')
    if not np.isfinite(scores).all(): raise RuntimeError('Non-finite retrieval distance')
    if len(gallery)>=4 and float(scores.std(axis=1).max())<1e-7:
        raise RuntimeError('DEGENERATE rankings: every image receives the same score')
    _log(f'Non-degenerate model ranking returned shape={scores.shape}, distance range [{scores.min():.4f},{scores.max():.4f}]')
    return {'distances':scores}
