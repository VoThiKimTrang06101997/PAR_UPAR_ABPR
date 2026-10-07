from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from abpr.core import ATTRIBUTE_NAMES, ImageResolver, attribute_error_weights, seed_everything
from abpr.model import (
    ABPRNet, UPARDataset, build_eval_transform, find_task2_split_files, load_query_csv,
    fit_affine_calibration, calibrated_attribute_distances, track2_metrics,
)
from abpr.prototype import blend_attribute_probabilities


def parse_args():
    p=argparse.ArgumentParser(description='Calibrate retrieval on the organizer Track-2 validation split.')
    p.add_argument('--repo-root',default='/content/UPAR-Challenge-2027')
    p.add_argument('--checkpoint',action='append',required=True,help='Repeat for a probability ensemble.')
    p.add_argument('--output',default='')
    p.add_argument('--extra-data-root',action='append',default=[])
    p.add_argument('--batch-size',type=int,default=40)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--max-calib-gallery',type=int,default=16000)
    p.add_argument('--max-calib-queries',type=int,default=420)
    p.add_argument('--top-full-trials',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tta-flip',action='store_true')
    return p.parse_args()


@torch.no_grad()
def encode_one(model, loader, device, tta_flip=False, desc='Encode validation'):
    probs=[]; proto_probs=[]; emb=[]; labels=[]; valid=[]; domains=[]
    model.eval()
    for b in tqdm(loader,desc=desc,dynamic_ncols=True,leave=False):
        x=b['image'].to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            logits,z,_,proto_logits,_,_,_=model.encode_images(x,return_prototype=True)
            if tta_flip:
                lf,zf,_,pf,_,_,_=model.encode_images(torch.flip(x,dims=[3]),return_prototype=True)
                logits=(logits+lf)*.5; proto_logits=(proto_logits+pf)*.5
                z=torch.nn.functional.normalize((z+zf)*.5,dim=-1)
        probs.append(torch.sigmoid(logits.float()).cpu())
        proto_probs.append(torch.sigmoid(proto_logits.float()).cpu())
        emb.append(z.float().cpu())
        q=b['query'].numpy().astype(np.int16); labels.append(q); valid.append((q>=0).astype(np.float32)); domains.append(b['domain'].numpy())
    return torch.cat(probs),torch.cat(proto_probs),torch.cat(emb),np.concatenate(labels),np.concatenate(valid),np.concatenate(domains)


def score_selection(m):
    # Codabench primary metric is mADM; the others break ties and penalize degenerate ranking.
    return float(.88*m['mADM']+.05*m['mAP']+.025*m['Rank-1']+.015*m['Rank-5']+.01*m['Rank-10']+.02*m['mINP'])


def select_calibration_subset(queries, labels, max_q, max_g, seed):
    rng=np.random.default_rng(seed)
    qidx=np.arange(len(queries))
    if len(qidx)>max_q:
        counts=(queries>=0).sum(1); chosen=[]
        for k in sorted(np.unique(counts)):
            idx=np.where(counts==k)[0]; share=max(1,int(round(max_q*len(idx)/len(queries))))
            chosen.extend(rng.choice(idx,size=min(share,len(idx)),replace=False).tolist())
        if len(chosen)>max_q: chosen=rng.choice(np.asarray(chosen),size=max_q,replace=False).tolist()
        qidx=np.asarray(chosen,dtype=np.int64)
    if len(labels)<=max_g:
        return qidx,np.arange(len(labels))
    # Preserve every exact-positive example for selected queries when feasible, then sample negatives.
    positive=set()
    for qi in qidx:
        q=queries[qi]; valid=q>=0
        if not valid.any(): continue
        y=labels[:,valid]
        hit=np.where(((y>=0).all(1) & (y==q[valid][None,:]).all(1)))[0]
        positive.update(hit.tolist())
        if len(positive)>=max_g: break
    positive=np.asarray(sorted(positive),dtype=np.int64)
    if len(positive)>max_g:
        positive=rng.choice(positive,size=max_g,replace=False)
    pool=np.setdiff1d(np.arange(len(labels)),positive)
    remain=max_g-len(positive)
    neg=rng.choice(pool,size=min(remain,len(pool)),replace=False) if remain>0 else np.asarray([],dtype=np.int64)
    gidx=np.concatenate([positive,neg]); rng.shuffle(gidx)
    return qidx,gidx


def load_checkpoint(path, device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=ABPRNet(ck['backbone'],int(ck['embed_dim']),pretrained=False)
    state=ck.get('inference_model_state') or ck.get('ema_model_state') or ck.get('model_state')
    model.load_state_dict(state,strict=True)
    return ck,model.to(device).eval()


def main():
    args=parse_args(); seed_everything(args.seed)
    root=Path(args.repo_root); ck_paths=[Path(x) for x in args.checkpoint]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gallery_csv,query_csv=find_task2_split_files(root,'val')
    gdf=pd.read_csv(gallery_csv); _,queries=load_query_csv(query_csv)
    print('Validation gallery:',gallery_csv,'rows=',len(gdf))
    print('Validation queries:',query_csv,'rows=',len(queries))

    checkpoints=[]; models=[]
    for p in ck_paths:
        ck,m=load_checkpoint(p,device); checkpoints.append(ck); models.append(m)
    h=int(checkpoints[0].get('image_height',384)); w=int(checkpoints[0].get('image_width',192))
    for ck in checkpoints:
        if int(ck.get('image_height',h))!=h or int(ck.get('image_width',w))!=w:
            raise RuntimeError('All ensemble checkpoints must use the same image size.')
    resolver=ImageResolver(root/'data',root,extra_roots=args.extra_data_root)
    ds=UPARDataset(gdf,resolver,build_eval_transform(h,w))
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=device.type=='cuda',persistent_workers=args.num_workers>0)

    probs_all=[]; proto_all=[]; emb_all=[]; labels=valid=domains=None
    for i,m in enumerate(models):
        p,pp,e,y,v,d=encode_one(m,dl,device,tta_flip=args.tta_flip,desc=f'Encode model {i+1}/{len(models)}')
        probs_all.append(p); proto_all.append(pp); emb_all.append(e)
        if labels is None: labels,valid,domains=y,v,d
    probs=torch.stack(probs_all).mean(0)
    proto_probs=torch.stack(proto_all).mean(0)
    emb=torch.nn.functional.normalize(torch.stack(emb_all).mean(0),dim=-1)

    from abpr.model import apply_affine_calibration
    prototype_mixes=(0.0,0.15,0.30,0.45,0.60)
    calibration_by_mix={}
    print('Fitting affine calibration for prototype/fused probability blends...')
    for pmix in tqdm(prototype_mixes,desc='Prototype calibration',dynamic_ncols=True):
        mixed=blend_attribute_probabilities(probs,proto_probs,pmix)
        scale,bias=fit_affine_calibration(mixed,labels,valid,steps=180,lr=.035)
        pcalibrated=apply_affine_calibration(mixed,scale,bias)
        base_weights=np.asarray(attribute_error_weights(pcalibrated,labels,valid,tau=.18),dtype=np.float32)
        calibration_by_mix[float(pmix)]={'probs':mixed,'scale':scale,'bias':bias,'base_weights':base_weights}

    priors=np.asarray(checkpoints[0].get('priors',[.5]*40),dtype=np.float32)
    if priors.shape!=(40,):
        yclean=np.where(valid>0,np.clip(labels,0,1),0); priors=(yclean*valid).sum(0)/np.maximum(valid.sum(0),1); priors=np.clip(priors,.01,.99).astype(np.float32)

    qidx,gidx=select_calibration_subset(queries,labels,args.max_calib_queries,args.max_calib_gallery,args.seed)
    qcal=queries[qidx]; esub=emb[gidx]; ysub=labels[gidx]

    # Query embeddings are optional auxiliary evidence. Average across ensemble models.
    qz=[]
    with torch.inference_mode():
        qt=torch.as_tensor(queries,dtype=torch.float32,device=device)
        for m in models: qz.append(m.query_encoder(qt).float().cpu())
    qz_all=torch.nn.functional.normalize(torch.stack(qz).mean(0),dim=-1); qz_sub=qz_all[qidx]

    grid=[]
    for pmix in prototype_mixes:
        for kind in ('match','nll','l1'):
            for reliability_power in (0.0,.5,1.0):
                for rarity in (0.0,.25,.45):
                    for negw in (.55,.75,1.0):
                        for mix in (0.0,.025):
                            grid.append((pmix,kind,reliability_power,rarity,negw,mix))

    records=[]
    for pmix,kind,rpow,rarity,negw,mix in tqdm(grid,desc='Calibration grid',dynamic_ncols=True):
        cal=calibration_by_mix[float(pmix)]
        psub=cal['probs'][gidx]
        weights=np.power(np.maximum(cal['base_weights'],1e-4),rpow).astype(np.float32); weights/=max(weights.mean(),1e-6)
        d=calibrated_attribute_distances(
            psub,qcal,weights,None,kind,esub,qz_sub,mix,priors,rarity,negw,
            calibration_scale=cal['scale'],calibration_bias=cal['bias'])
        m=track2_metrics(d,qcal,ysub)
        records.append({'prototype_mix':float(pmix),'distance_kind':kind,'reliability_power':rpow,'positive_rarity_power':rarity,'negative_weight':negw,'embedding_mix':mix,'metrics':m,'selection':score_selection(m)})
    records.sort(key=lambda r:r['selection'],reverse=True)

    full=[]
    for rec in tqdm(records[:max(1,args.top_full_trials)],desc='Full validation finalists',dynamic_ncols=True):
        cal=calibration_by_mix[float(rec['prototype_mix'])]
        weights=np.power(np.maximum(cal['base_weights'],1e-4),float(rec['reliability_power'])).astype(np.float32); weights/=max(weights.mean(),1e-6)
        d=calibrated_attribute_distances(
            cal['probs'],queries,weights,None,rec['distance_kind'],emb,qz_all,float(rec['embedding_mix']),priors,float(rec['positive_rarity_power']),float(rec['negative_weight']),
            calibration_scale=cal['scale'],calibration_bias=cal['bias'])
        m=track2_metrics(d,queries,labels)
        outrec={**{k:v for k,v in rec.items() if k!='metrics'},'metrics':m,'selection':score_selection(m),'attribute_weights':weights.tolist(),
                'calibration_scale':cal['scale'].tolist(),'calibration_bias':cal['bias'].tolist()}
        full.append(outrec); print(json.dumps(outrec))
    full.sort(key=lambda r:r['selection'],reverse=True); best=full[0]

    out=Path(args.output) if args.output else ck_paths[0].with_name('abpr_retrieval_calibrated.pt')
    package={
        'format':'upar2027-track2-abpr-prototype-calibrated',
        'models':[],
        'image_height':h,'image_width':w,'attribute_names':ATTRIBUTE_NAMES,
        'attribute_weights':best['attribute_weights'],'calibration_scale':best['calibration_scale'],'calibration_bias':best['calibration_bias'],
        'distance_kind':best['distance_kind'],'embedding_mix':float(best['embedding_mix']),'prototype_mix':float(best['prototype_mix']),
        'positive_rarity_power':float(best['positive_rarity_power']),'negative_weight':float(best['negative_weight']),
        'tta_flip':bool(args.tta_flip),'priors':priors.tolist(),
        'validation':{'metrics':best['metrics'],'selection':best['selection']},
        'retrieval_calibration':{
            'selected':best,'full_finalists':full,'grid_top20':records[:20],
            'val_gallery_csv':str(gallery_csv),'val_query_csv':str(query_csv),
            'val_gallery_size':int(len(labels)),'val_query_size':int(len(queries)),
            'calib_gallery_size':int(len(gidx)),'calib_query_size':int(len(qidx)),
            'tta_flip':bool(args.tta_flip),
        },
        'source_checkpoints':[p.name for p in ck_paths],
    }
    for ck,p in zip(checkpoints,ck_paths):
        state=ck.get('inference_model_state') or ck.get('ema_model_state') or ck.get('model_state')
        package['models'].append({'seed':int(ck.get('seed',0)),'backbone':ck['backbone'],'embed_dim':int(ck['embed_dim']),'state':state,'source_checkpoint':p.name})
    torch.save(package,out)
    out.with_suffix('.json').write_text(json.dumps({'output':str(out),'selected':best,'source_checkpoints':[str(x) for x in ck_paths]},indent=2),encoding='utf-8')
    trials_csv=out.with_name(out.stem+'_trials.csv')
    pd.DataFrame([{**{k:v for k,v in r.items() if k!='metrics'},**r['metrics']} for r in records]).to_csv(trials_csv,index=False)
    print('\nSELECTED CALIBRATION'); print(json.dumps({'output':str(out),'selected':best},indent=2)); print('Trials:',trials_csv)


if __name__=='__main__':
    main()
