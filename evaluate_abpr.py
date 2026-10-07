from __future__ import annotations

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from abpr.core import ImageResolver
from abpr.model import UPARDataset, build_eval_transform, find_task2_split_files, load_query_csv, calibrated_attribute_distances, track2_metrics
from abpr.prototype import blend_attribute_probabilities


def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo-root',default='/content/UPAR-Challenge-2027'); p.add_argument('--checkpoint',required=True)
    p.add_argument('--extra-data-root',action='append',default=[]); p.add_argument('--batch-size',type=int,default=40); p.add_argument('--num-workers',type=int,default=2); a=p.parse_args()
    root=Path(a.repo_root); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    gallery_csv,query_csv=find_task2_split_files(root,'val'); gdf=pd.read_csv(gallery_csv); _,queries=load_query_csv(query_csv)
    from abpr.model import ABPRNet
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolver=ImageResolver(root/'data',root,extra_roots=a.extra_data_root)
    h=int(ck.get('image_height',384)); w=int(ck.get('image_width',192)); ds=UPARDataset(gdf,resolver,build_eval_transform(h,w)); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=='cuda',persistent_workers=a.num_workers>0)
    probs_models=[]; proto_models=[]; emb_models=[]; labels=[]
    for mi,item in enumerate(ck.get('models') or [ck]):
        m=ABPRNet(item.get('backbone',ck.get('backbone','convnext_small')),int(item.get('embed_dim',ck.get('embed_dim',320))),pretrained=False)
        state=item.get('state') or item.get('inference_model_state') or item.get('ema_model_state') or item.get('model_state'); m.load_state_dict(state,strict=True); m=m.to(device).eval()
        ps=[]; proto_local=[]; es=[]; local_labels=[]
        for b in tqdm(dl,desc=f'Evaluate model {mi+1}',dynamic_ncols=True,leave=False):
            x=b['image'].to(device,non_blocking=True)
            with torch.inference_mode(), torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits,z,_,proto_logits,_,_,_=m.encode_images(x,return_prototype=True)
                if ck.get('tta_flip',False):
                    lf,zf,_,pf,_,_,_=m.encode_images(torch.flip(x,dims=[3]),return_prototype=True); logits=(logits+lf)*.5; proto_logits=(proto_logits+pf)*.5; z=torch.nn.functional.normalize((z+zf)*.5,dim=-1)
            ps.append(torch.sigmoid(logits.float()).cpu()); proto_local.append(torch.sigmoid(proto_logits.float()).cpu()); es.append(z.float().cpu())
            if mi==0: local_labels.append(b['query'].numpy().astype(np.int16))
        probs_models.append(torch.cat(ps)); proto_models.append(torch.cat(proto_local)); emb_models.append(torch.cat(es))
        if mi==0: labels=np.concatenate(local_labels)
    probs=torch.stack(probs_models).mean(0); proto_probs=torch.stack(proto_models).mean(0); probs=blend_attribute_probabilities(probs,proto_probs,float(ck.get('prototype_mix',0.0))); emb=torch.nn.functional.normalize(torch.stack(emb_models).mean(0),dim=-1)
    qz=[]
    qt=torch.as_tensor(queries,dtype=torch.float32,device=device)
    for item in ck.get('models') or [ck]:
        m=ABPRNet(item.get('backbone',ck.get('backbone','convnext_small')),int(item.get('embed_dim',ck.get('embed_dim',320))),pretrained=False)
        state=item.get('state') or item.get('inference_model_state') or item.get('ema_model_state') or item.get('model_state'); m.load_state_dict(state,strict=True); m=m.to(device).eval()
        with torch.inference_mode(): qz.append(m.query_encoder(qt).float().cpu())
    qz=torch.nn.functional.normalize(torch.stack(qz).mean(0),dim=-1)
    d=calibrated_attribute_distances(probs,queries,ck.get('attribute_weights'),None,ck.get('distance_kind','match'),emb,qz,float(ck.get('embedding_mix',0.)),ck.get('priors'),float(ck.get('positive_rarity_power',0.)),float(ck.get('negative_weight',1.)),calibration_scale=ck.get('calibration_scale'),calibration_bias=ck.get('calibration_bias'))
    report={'checkpoint':str(a.checkpoint),'gallery':str(gallery_csv),'queries':str(query_csv),'params':{k:ck.get(k) for k in ['distance_kind','embedding_mix','prototype_mix','positive_rarity_power','negative_weight','tta_flip']},'metrics':track2_metrics(d,queries,labels)}
    print(json.dumps(report,indent=2))
    Path(a.checkpoint).with_suffix('.eval.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__': main()
