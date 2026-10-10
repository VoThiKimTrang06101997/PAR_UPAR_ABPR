"""FIT/AUDIT public-VAL calibration for ABPR trained pixel-model ranking.

Uses public TRAIN labels only for attribute priors, never public VAL image
labels to fit a neural network; query IDs are used exclusively as public
validation relevance for selection. Outputs a no-op (blend=0) when audit
fails. Does not claim to calculate official hidden mADM.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from abpr.query_aware_ranking import score, fuse, retrieval_metrics


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--repo-root',required=True,type=Path)
    p.add_argument('--runtime-zip',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cache-dir',required=True,type=Path)
    p.add_argument('--max-gallery',type=int,default=0,help='0 = complete public VAL gallery')
    p.add_argument('--fit-queries',type=int,default=240)
    p.add_argument('--audit-queries',type=int,default=240)
    p.add_argument('--min-audit-madm-gain',type=float,default=0.003)
    p.add_argument('--max-rank1-regression',type=float,default=0.005)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--extra-data-root',type=Path,action='append',default=[])
    return p.parse_args()


def paths_to_ids(repo, frame, col):
    ids_path=repo/'data/annotations/task2/val/ids.csv'
    ids=pd.read_csv(ids_path,header=None,dtype=str)
    # Challenge ids.csv has exactly two columns without headers: image path, query ID.
    if ids.shape[1]<2: raise ValueError(f'Expected 2 columns in {ids_path}')
    lookup={}
    for path,qid in ids.iloc[:,:2].itertuples(index=False,name=None):
        key=str(path).strip().replace('\\','/').lstrip('./')
        try: val=int(float(qid))
        except (ValueError,TypeError):
            if key.lower() in ('image','image_path','path'):continue
            raise ValueError(f'Invalid semantic query ID: {path},{qid}')
        if key in lookup and lookup[key]!=val:raise ValueError('Duplicated path maps to different IDs')
        lookup[key]=val
    mapped=[];missing=[]
    for raw in frame[col].astype(str):
        key=raw.strip().replace('\\','/').lstrip('./')
        if key not in lookup:missing.append(key);mapped.append(-1)
        else:mapped.append(lookup[key])
    if missing:raise ValueError(f'Missing {len(missing)} image-path/ID alignments (first={missing[:3]}). Refuse positional join.')
    return np.asarray(mapped,np.int32)


def get_priors(repo):
    from abpr.core import ATTRIBUTE_NAMES
    train=repo/'data/annotations/task2/train/gt.csv'
    df=pd.read_csv(train,usecols=list(ATTRIBUTE_NAMES))
    values=df[list(ATTRIBUTE_NAMES)].to_numpy(dtype=np.float32)
    mask=(values==0)|(values==1)
    sums=((values==1)&mask).sum(0)
    den=mask.sum(0)
    priors=(sums+1.)/(den+2.)
    return np.clip(priors.astype(np.float32),0.02,0.98)


def load_queries(repo):
    from abpr.core import ATTRIBUTE_NAMES
    from abpr.model import load_query_csv
    _,q=load_query_csv(repo/'data/annotations/task2/val/queries.csv')
    q=np.asarray(q,np.float32)
    if q.ndim!=2 or q.shape[1]!=len(ATTRIBUTE_NAMES):
        raise ValueError(f'Query shape {q.shape} does not match official 40 attributes')
    return q


def extract_model(runtime_zip, directory):
    with zipfile.ZipFile(runtime_zip) as z:
        required={'abpr_runtime.py','assets/model.pt'}
        if not required.issubset(z.namelist()):raise ValueError('Runtime ZIP lacks model weights and runtime')
        for name in required:
            dst=(directory/name).resolve()
            if not dst.is_relative_to(directory.resolve()):raise ValueError('Unsafe file path')
            dst.parent.mkdir(exist_ok=True,parents=True)
            dst.write_bytes(z.read(name))
    sys.path.insert(0,str(directory))
    from abpr_runtime import ABPRRuntime
    return ABPRRuntime(directory/'assets/model.pt')


def get_features(a, frame, col):
    import hashlib
    from abpr.core import ImageResolver
    a.cache_dir.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256()
    with a.runtime_zip.open('rb') as f:
        for blob in iter(lambda:f.read(2**20),b''):identity.update(blob)
    identity.update(str(len(frame)).encode());identity.update('|'.join(frame[col].astype(str)).encode())
    fname=a.cache_dir/f'public_val_features_{identity.hexdigest()[:16]}.npz'
    if fname.is_file():
        with np.load(fname,allow_pickle=False) as cached:
            probs=cached['probs'];emb=cached['emb']
        print('Model-specific feature cache HIT:',fname,flush=True)
        return probs,emb
    resolver=ImageResolver(a.repo_root/'data',a.repo_root,extra_roots=a.extra_data_root)
    records=[]
    for raw in tqdm(frame[col].astype(str),desc='Resolve VAL real images'):
        path=resolver.resolve(raw)
        if not Path(path).is_file():raise FileNotFoundError(f'VAL image not found: {raw} -> {path}')
        records.append({'image_path':str(path)})
    with tempfile.TemporaryDirectory(prefix='abpr_runtime_') as tmp:
        runtime=extract_model(a.runtime_zip,Path(tmp))
        result=runtime.encode_gallery(records)
        if not isinstance(result,tuple) or len(result)<2:raise TypeError('ABPRRuntime.encode_gallery must return (probs,emb)')
        probs,emb=result[:2]
        if hasattr(probs,'detach'):probs=probs.detach().cpu().numpy()
        if hasattr(emb,'detach'):emb=emb.detach().cpu().numpy()
        probs=np.asarray(probs,dtype=np.float32)
        emb=np.asarray(emb,dtype=np.float32) if emb is not None else np.empty((len(probs),0),np.float32)
    if probs.shape!=(len(frame),40) or not np.isfinite(probs).all():
        raise ValueError(f'Invalid pixel-model features {probs.shape}')
    np.savez_compressed(fname,probs=probs,emb=emb)
    print('Cached VAL REAL image features:',fname,flush=True)
    return probs,emb


def get_baseline(a, query_array, probs, emb, query_indices):
    with tempfile.TemporaryDirectory(prefix='abpr_runtime_') as tmp:
        runtime=extract_model(a.runtime_zip,Path(tmp))
        p=torch.as_tensor(probs,dtype=torch.float32)
        e=torch.as_tensor(emb,dtype=torch.float32) if emb.shape[1] else None
        results=[]
        for ids in tqdm(np.array_split(query_indices,max(1,int(np.ceil(len(query_indices)/48)))),desc='Baseline distance batches'):
            d=runtime.distance(query_array[ids],p,e)
            if hasattr(d,'detach'):d=d.detach().cpu().numpy()
            d=np.asarray(d,np.float32)
            if d.shape!=(len(ids),len(probs)) or not np.isfinite(d).all():
                raise ValueError(f'Invalid runtime baseline score shape {d.shape}')
            results.append(d)
    return np.concatenate(results,axis=0)


def main():
    a=arguments()
    sys.path.insert(0,str(a.source_root))
    from abpr.core import detect_image_column, ATTRIBUTE_NAMES
    from abpr.model import track2_metrics
    val=a.repo_root/'data/annotations/task2/val/gt.csv'
    if not val.is_file():raise FileNotFoundError(val)
    full=pd.read_csv(val)
    col=detect_image_column(full)
    q=load_queries(a.repo_root)
    # Select only genuine public VAL images; no fake dataset rows or synthetic scores.
    rng=np.random.default_rng(a.seed)
    if a.max_gallery and a.max_gallery < len(full):
        selected=np.sort(rng.choice(len(full),size=a.max_gallery,replace=False))
        frame=full.iloc[selected].reset_index(drop=True)
    else:frame=full
    gids=paths_to_ids(a.repo_root,frame,col)
    labels=frame[list(ATTRIBUTE_NAMES)].to_numpy(np.float32)
    valid_ids=np.intersect1d(np.unique(gids),np.arange(len(q)))
    needed=a.fit_queries+a.audit_queries
    if len(valid_ids)<needed:
        raise ValueError(f'Only {len(valid_ids)} public VAL queries have positives: need {needed}. Lower fit/audit counts.')
    rng.shuffle(valid_ids)
    fit_ids=valid_ids[:a.fit_queries]
    audit_ids=valid_ids[a.fit_queries:needed]
    print(f'Gallery={len(frame)}, queries={len(q)}, FIT={len(fit_ids)}, AUDIT={len(audit_ids)}',flush=True)
    priors=get_priors(a.repo_root)
    probs,emb=get_features(a,frame,col)
    merged=np.concatenate([fit_ids,audit_ids])
    baseline=get_baseline(a,q,probs,emb,merged)
    base_fit=baseline[:len(fit_ids)];base_audit=baseline[len(fit_ids):]
    fit_base=retrieval_metrics(base_fit,fit_ids,gids)
    audit_base=retrieval_metrics(base_audit,audit_ids,gids)
    fit_base.update(track2_metrics(base_fit,q[fit_ids],labels))
    audit_base.update(track2_metrics(base_audit,q[audit_ids],labels))
    print('Baseline FIT',fit_base,'AUDIT',audit_base,flush=True)
    trial_results=[];current={'method':'none','blend':0.,'negative_weight':1.,'rarity_power':0.}
    for method in ['loglik','expected']:
        for neg in [0.75,1.0]:
            for rarity in [0.,0.35]:
                values=score(q[fit_ids],probs,method=method,negative_weight=neg,rarity_power=rarity,priors=priors)
                for blend in [0.05,0.12,0.25]:
                    dist=fuse(base_fit,values,blend)
                    m=retrieval_metrics(dist,fit_ids,gids)
                    m.update(track2_metrics(dist,q[fit_ids],labels))
                    trial={'method':method,'blend':blend,'negative_weight':neg,'rarity_power':rarity,**m}
                    trial_results.append(trial)
    trial_results.sort(key=lambda d:(d['mADM'],d['mAP'],d['Rank-1']),reverse=True)
    if trial_results and trial_results[0]['mADM']>=fit_base['mADM']+0.002 and trial_results[0]['Rank-1']>=fit_base['Rank-1']-a.max_rank1_regression:
        current=trial_results[0]
        new=score(q[audit_ids],probs,method=current['method'],negative_weight=current['negative_weight'],rarity_power=current['rarity_power'],priors=priors)
        dist=fuse(base_audit,new,current['blend'])
        audit_candidate=retrieval_metrics(dist,audit_ids,gids)
        audit_candidate.update(track2_metrics(dist,q[audit_ids],labels))
    else:
        audit_candidate=dict(audit_base)
    accepted=bool(current['blend']>0 and
       audit_candidate['mADM']>=audit_base['mADM']+a.min_audit_madm_gain and
       audit_candidate['Rank-1']>=audit_base['Rank-1']-a.max_rank1_regression)
    selected=current if accepted else {'method':'loglik','blend':0.,'negative_weight':1.,'rarity_power':0.}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,method=np.array(selected['method']),blend=np.float32(selected['blend']),
        negative_weight=np.float32(selected['negative_weight']),rarity_power=np.float32(selected['rarity_power']),priors=priors)
    report={ 'official_hidden_mADM':None,'note':'Public VAL track2_metrics (mADM etc) + exact semantic-ID check; NOT hidden Codabench mADM',
      'gallery_images':len(frame),'query_count':len(q),'fit_query_ids':fit_ids.tolist(),'audit_query_ids':audit_ids.tolist(),
      'baseline_fit':fit_base,'baseline_audit':audit_base,'candidate_audit':audit_candidate,
      'best_fit_trial':trial_results[0] if trial_results else None,'top_fit_trials':trial_results[:8],
      'selected':selected,'audit_accepted':accepted,'model_zip':str(a.runtime_zip),
      'artifact':str(a.output),'metric_gain_audit_mADM':round(audit_candidate['mADM']-audit_base['mADM'],6)}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
