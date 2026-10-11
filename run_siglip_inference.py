"""Optional Track 2 submission wrapper: original baseline + frozen SigLIP2.

Works OFFLINE when assets/siglip2 model/processor are packaged and transformers
is preinstalled in the grading environment. Do not use unless smoke test passes.
"""
from pathlib import Path
import gc,os
import numpy as np
import torch
import base_run
from abpr_siglip2_pipeline import image_embeddings, prompt_distance, adapter_distance, combine, text_pairs, load_siglip
HERE=Path(__file__).resolve().parent
_MODEL=None
_PROCESSOR=None
_DEVICE=None
_TEXT=None
_CONFIG=None


def _setup():
    global _MODEL,_PROCESSOR,_DEVICE,_TEXT,_CONFIG
    if _MODEL is not None:return
    import json
    _CONFIG=json.loads((HERE/'assets/siglip2_config.json').read_text())
    _MODEL,_PROCESSOR,_DEVICE=load_siglip(HERE/'assets/siglip2')
    from abpr_siglip2_pipeline import text_pairs
    _TEXT=text_pairs(base_run.CANONICAL_NAMES,_MODEL,_PROCESSOR,_DEVICE)


def load_model():
    base_run.load_model()
    _setup()


def predict_attributes(gallery,attribute_names):
    return base_run.predict_attributes(gallery,attribute_names)


def rank_gallery(sample):
    import tempfile
    # IMPORTANT: existing best ZIP baseline (e.g. Query-Aware) executes unmodified.
    output=base_run.rank_gallery(sample)
    b=np.asarray(output['distances'],np.float32)
    alpha=float(_get_config()['blend'])
    if alpha==0:return output
    # Free ConvNeXt GPU allocations before loading SigLIP2 to reduce peak usage.
    if getattr(base_run,'_MODEL',None) is not None:
        base_run._MODEL=None
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    _setup()
    gal=base_run._gallery_records(sample['gallery'])
    paths=[str(r['image_path']) for r in gal]
    names=list(sample.get('attribute_names',base_run.CANONICAL_NAMES))
    keys={base_run._norm(x):i for i,x in enumerate(names)}
    q=np.asarray(sample['queries'],np.float32)
    if q.ndim==1:q=q[None,:]
    q=q[:,[keys[base_run._norm(n)] for n in base_run.CANONICAL_NAMES]]
    # Feature arrays written ONLY to temporary runtime disk (not grading labels).
    with tempfile.TemporaryDirectory(prefix='abpr_siglip2_hidden_') as tmp:
        x=image_embeddings(paths,_MODEL,_PROCESSOR,_DEVICE,Path(tmp)/'gallery.npy',batch_size=int(_get_config().get('batch_size',16)))
        pdist=prompt_distance(q,x,_TEXT)
        with np.load(HERE/'assets/adapter.npz',allow_pickle=False) as adapter:
            adist=adapter_distance(q,x,adapter['weight'],adapter['bias'])
        scores=combine(b,pdist,adist,_get_config()['method'],alpha)
    if scores.shape!=b.shape or not np.isfinite(scores).all():
        raise RuntimeError(f'Invalid fused ranking {scores.shape} vs {b.shape}')
    return {'distances':np.asarray(scores,np.float32)}


def _get_config():
    global _CONFIG
    if _CONFIG is None:
        import json
        _CONFIG=json.loads((HERE/'assets/siglip2_config.json').read_text())
    return _CONFIG
