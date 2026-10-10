from __future__ import annotations



from pathlib import Path

import argparse

import json

import math

import time



import numpy as np

import pandas as pd

import torch

from torch.utils.data import DataLoader

from tqdm.auto import tqdm



from abpr.core import (

    ATTRIBUTE_NAMES, ImageResolver, discover_labeled_csv,

    split_train_val, make_domain_balanced_sampler, ModelEMA, domain_loss,

    estimate_attribute_priors, attribute_error_weights, seed_everything, grad_reverse,

)

from abpr.model import (

    ABPRNet, UPARDataset, TrainConfig,

    build_train_transform, build_eval_transform,

    find_task2_split_files, load_query_csv, sample_sparse_queries,

    MaskedFocalLoss, group_consistency_loss, alignment_loss,

    degree_of_match_contrastive_loss, probability_degree_match_loss,

    degree_match_listwise_loss, calibrated_attribute_distances,

    track2_metrics,

)

from abpr.attribute_aware_contrastive import multilabel_supcon, query_hard_negative_loss

from abpr.prototype import (

    masked_prototype_bce,

    prototype_domain_alignment_loss,

    prototype_separation_loss,

    blend_attribute_probabilities,

)





def parse_args():

    p = argparse.ArgumentParser(description='Train Track-2 attribute-based person retrieval model.')

    p.add_argument('--repo-root', default='/content/UPAR-Challenge-2027')

    p.add_argument('--extra-data-root', action='append', default=[])

    p.add_argument('--checkpoint-dir', default='/content/drive/MyDrive/PedestrianAttributeRecognition/ABPR_Checkpoints')

    p.add_argument('--result-dir', default='/content/drive/MyDrive/PedestrianAttributeRecognition/ABPR_Results')

    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--epochs', type=int, default=24)

    p.add_argument('--batch-size', type=int, default=16)

    p.add_argument('--eval-batch-size', type=int, default=40)

    p.add_argument('--num-workers', type=int, default=2)

    p.add_argument('--backbone', choices=['convnext_tiny','convnext_small','convnext_base','efficientnet_b0','efficientnet_v2_s'], default='convnext_small')

    p.add_argument('--embed-dim', type=int, default=320)

    p.add_argument('--image-height', type=int, default=384)

    p.add_argument('--image-width', type=int, default=192)

    p.add_argument('--backbone-lr', type=float, default=4e-5)

    p.add_argument('--head-lr', type=float, default=2e-4)

    p.add_argument('--weight-decay', type=float, default=5e-4)

    p.add_argument('--val-fraction', type=float, default=.12)

    p.add_argument('--max-val-queries', type=int, default=450)

    p.add_argument('--query-min-known', type=int, default=2)

    p.add_argument('--query-max-known', type=int, default=10)

    p.add_argument('--save-every-steps', type=int, default=800)

    p.add_argument('--restart', action='store_true')
    # Fine-tune trained Prototype directly; original training remains unchanged
    # when --finetune-from is omitted.
    p.add_argument('--finetune-from', default=None,
                   help='Existing Prototype checkpoint for strict warm-start')
    p.add_argument('--supcon-weight', type=float, default=0.0)
    p.add_argument('--hard-weight', type=float, default=0.0)
    p.add_argument('--contrastive-warmup-steps', type=int, default=250)


    return p.parse_args()





@torch.no_grad()

def encode_loader(model, loader, device, tta_flip=False, desc='Encode validation'):

    model.eval(); probs=[]; proto_probs=[]; emb=[]; labels=[]; valid_all=[]; domains=[]

    for b in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):

        x=b['image'].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):

            logits,z,_,proto_logits,_,_,_=model.encode_images(x, return_prototype=True)

            if tta_flip:

                lf,zf,_,pf,_,_,_=model.encode_images(torch.flip(x, dims=[3]), return_prototype=True)

                logits=(logits+lf)*0.5

                proto_logits=(proto_logits+pf)*0.5

                z=torch.nn.functional.normalize((z+zf)*0.5,dim=-1)

        probs.append(torch.sigmoid(logits.float()).cpu())

        proto_probs.append(torch.sigmoid(proto_logits.float()).cpu())

        emb.append(z.float().cpu())

        q=b['query'].numpy().astype(np.int16); labels.append(q); valid_all.append((q>=0).astype(np.float32)); domains.append(b['domain'].numpy())

    return torch.cat(probs),torch.cat(proto_probs),torch.cat(emb),np.concatenate(labels),np.concatenate(valid_all),np.concatenate(domains)





def select_query_subset(queries: np.ndarray, max_queries: int, seed: int):

    if len(queries) <= max_queries:

        return queries

    rng=np.random.default_rng(seed)

    counts=(queries>=0).sum(1); chosen=[]

    for k in sorted(np.unique(counts)):

        idx=np.where(counts==k)[0]

        share=max(1,int(round(max_queries*len(idx)/len(queries))))

        chosen.extend(rng.choice(idx,size=min(share,len(idx)),replace=False).tolist())

    if len(chosen)>max_queries:

        chosen=rng.choice(np.asarray(chosen),size=max_queries,replace=False).tolist()

    elif len(chosen)<max_queries:

        remain=np.setdiff1d(np.arange(len(queries)),np.asarray(chosen,dtype=np.int64))

        need=min(max_queries-len(chosen),len(remain))

        if need>0: chosen.extend(rng.choice(remain,size=need,replace=False).tolist())

    return queries[np.asarray(chosen,dtype=np.int64)]





@torch.no_grad()

def evaluate_retrieval(model, loader, device, queries, train_priors, max_queries=450, seed=42):

    probs,proto_probs,emb,labels,valid,domains=encode_loader(model,loader,device,tta_flip=False)

    q=select_query_subset(np.asarray(queries,dtype=np.float32),max_queries,seed)

    with torch.inference_mode():

        qz=model.query_encoder(torch.as_tensor(q,dtype=torch.float32,device=device)).float().cpu()



    candidates=[]

    for proto_mix in (0.0, 0.15, 0.30, 0.45, 0.60):

        mixed=blend_attribute_probabilities(probs,proto_probs,proto_mix)

        weights=attribute_error_weights(mixed,labels,valid,tau=.20)

        d=calibrated_attribute_distances(

            mixed,q,attribute_weights=weights,attribute_temperatures=None,distance_kind='match',

            gallery_emb=emb,query_emb=qz,embedding_mix=0.02,priors=train_priors,

            positive_rarity_power=0.20,negative_weight=0.75,

        )

        metrics=track2_metrics(d,q,labels)

        selection=.90*metrics['mADM']+.04*metrics['mAP']+.02*metrics['Rank-1']+.015*metrics['Rank-5']+.015*metrics['Rank-10']+.01*metrics['mINP']

        candidates.append((float(selection),float(proto_mix),metrics,weights))

    candidates.sort(key=lambda x:x[0],reverse=True)

    selection,proto_mix,metrics,weights=candidates[0]

    return {

        'metrics':metrics,'selection':float(selection),'num_val_queries_used':int(len(q)),

        'distance_kind':'match','embedding_mix':.02,'positive_rarity_power':.20,'negative_weight':.75,

        'prototype_mix':float(proto_mix),'attribute_weights':weights.tolist(),

    }





def append_history(path: Path, row: dict):

    path.parent.mkdir(parents=True,exist_ok=True)

    frame=pd.DataFrame([row])

    if path.exists():

        old=pd.read_csv(path)

        if 'epoch' in old.columns:

            old=old[old['epoch']!=row['epoch']]

        frame=pd.concat([old,frame],ignore_index=True)

    frame.to_csv(path,index=False)





def capture_rng_state():

    state={'torch':torch.get_rng_state(),'numpy':np.random.get_state()}

    if torch.cuda.is_available(): state['cuda']=torch.cuda.get_rng_state_all()

    return state





def restore_rng_state(state):

    if not state: return

    try: torch.set_rng_state(state['torch'])

    except Exception: pass

    try: np.random.set_state(state['numpy'])

    except Exception: pass

    if torch.cuda.is_available() and state.get('cuda') is not None:

        try: torch.cuda.set_rng_state_all(state['cuda'])

        except Exception: pass





def make_scheduler(optimizer, total_steps: int, warmup_ratio: float):

    warmup=max(1,int(total_steps*warmup_ratio))

    def fn(step):

        if step < warmup: return max(step,1)/warmup

        progress=(step-warmup)/max(total_steps-warmup,1)

        return 0.5*(1.0+math.cos(math.pi*min(max(progress,0.0),1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer,fn)





def main():

    args=parse_args(); seed_everything(args.seed)
    if args.finetune_from and (args.supcon_weight + args.hard_weight <= 0):
        raise ValueError('Fine-tune requested but both contrastive loss weights are zero')

    repo=Path(args.repo_root); data_root=repo/'data'

    ckdir=Path(args.checkpoint_dir); outdir=Path(args.result_dir); ckdir.mkdir(parents=True,exist_ok=True); outdir.mkdir(parents=True,exist_ok=True)



    train_csv=discover_labeled_csv(repo,'train')

    try: val_csv=discover_labeled_csv(repo,'val')

    except FileNotFoundError: val_csv=None

    train_df=pd.read_csv(train_csv)

    if val_csv is not None: val_df=pd.read_csv(val_csv)

    else: train_df,val_df=split_train_val(train_df,args.val_fraction,args.seed)



    try:

        val_gallery_csv,val_query_csv=find_task2_split_files(repo,'val')

        _,val_queries=load_query_csv(val_query_csv)

        val_gallery_df=pd.read_csv(val_gallery_csv)

        if all(c in val_gallery_df.columns for c in ATTRIBUTE_NAMES):

            val_df=val_gallery_df

        print('Track-2 validation gallery:',val_gallery_csv,'rows=',len(val_df))

        print('Track-2 validation queries:',val_query_csv,'rows=',len(val_queries))

    except Exception as exc:

        print('[warn] exact Track-2 validation queries unavailable:',repr(exc))

        labels=val_df[ATTRIBUTE_NAMES].to_numpy(np.float32)

        val_queries=np.unique(labels,axis=0)



    cfg=TrainConfig(

        backbone=args.backbone,embed_dim=args.embed_dim,image_height=args.image_height,image_width=args.image_width,

        batch_size=args.batch_size,eval_batch_size=args.eval_batch_size,epochs=args.epochs,

        backbone_lr=args.backbone_lr,head_lr=args.head_lr,weight_decay=args.weight_decay,num_workers=args.num_workers,

        query_min_known=args.query_min_known,query_max_known=args.query_max_known,

    )

    resolver=ImageResolver(data_root,repo,extra_roots=args.extra_data_root)

    tr=UPARDataset(train_df,resolver,build_train_transform(cfg.image_height,cfg.image_width))

    va=UPARDataset(val_df,resolver,build_eval_transform(cfg.image_height,cfg.image_width))

    sampler=make_domain_balanced_sampler(tr.domains,args.seed)

    train_loader=DataLoader(tr,batch_size=cfg.batch_size,sampler=sampler,num_workers=cfg.num_workers,pin_memory=True,persistent_workers=cfg.num_workers>0,drop_last=False)

    val_loader=DataLoader(va,batch_size=cfg.eval_batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=True,persistent_workers=cfg.num_workers>0,drop_last=False)



    train_valid=((tr.targets==0)|(tr.targets==1)).astype(np.float32)

    train_targets=np.where(train_valid>0,tr.targets,0)

    priors=estimate_attribute_priors(train_targets,train_valid)



    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Device:',device,'Backbone:',cfg.backbone,'Image:',cfg.image_height,'x',cfg.image_width,'Train:',len(tr),'Val:',len(va))

    model=ABPRNet(cfg.backbone,cfg.embed_dim,pretrained=True).to(device)

    ema=ModelEMA(model,cfg.ema_decay); ema.module=ema.module.to(device)

    attr_criterion=MaskedFocalLoss(gamma=cfg.focal_gamma,label_smoothing=0.0)



    backbone_params=[]; head_params=[]

    for name,param in model.named_parameters():

        if name.startswith(('features.','avgpool.')): backbone_params.append(param)

        else: head_params.append(param)

    optimizer=torch.optim.AdamW([

        {'params':backbone_params,'lr':cfg.backbone_lr},

        {'params':head_params,'lr':cfg.head_lr},

    ],weight_decay=cfg.weight_decay)

    total_steps=max(1,cfg.epochs*len(train_loader)); scheduler=make_scheduler(optimizer,total_steps,cfg.warmup_ratio)

    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')



    prefix=f'abpr_prototype_{cfg.backbone}_seed{args.seed}'

    last_path=ckdir/f'{prefix}_last.pt'; best_path=ckdir/f'{prefix}_best.pt'; hist_path=outdir/f'{prefix}_history.csv'

    if args.restart:

        for p in (last_path,best_path,hist_path): p.unlink(missing_ok=True)



    # Protect the baseline: fine-tuning must write to a DIFFERENT directory.
    finetune_from = Path(args.finetune_from).expanduser().resolve() if args.finetune_from else None
    if finetune_from is not None:
        if not finetune_from.is_file():
            raise FileNotFoundError(f'Fine-tune baseline checkpoint not found: {finetune_from}')
        if finetune_from.parent == ckdir.resolve():
            raise RuntimeError('Fine-tune checkpoint directory cannot equal baseline directory')
        if args.restart:
            raise RuntimeError('Do not use --restart for fine-tune. Select an empty separate checkpoint directory.')
        if args.supcon_weight < 0 or args.hard_weight < 0:
            raise ValueError('Contrastive weights must be non-negative')
        baseline_state = torch.load(finetune_from, map_location='cpu', weights_only=False)
        if not isinstance(baseline_state, dict):
            raise RuntimeError('Baseline checkpoint is not a mapping')
        if baseline_state.get('backbone', args.backbone) != args.backbone:
            raise RuntimeError('Baseline backbone does not match requested backbone')
        if int(baseline_state.get('embed_dim', args.embed_dim)) != args.embed_dim:
            raise RuntimeError('Baseline embed_dim does not match requested embed_dim')
        weights = next((baseline_state[k] for k in
                        ('inference_model_state', 'ema_model_state', 'model_state')
                        if isinstance(baseline_state.get(k), dict)), None)
        if weights is None:
            raise RuntimeError('Baseline checkpoint has no usable trained state_dict')
        model.load_state_dict(weights, strict=True)
        ema.module.load_state_dict(weights, strict=True)
        del weights, baseline_state
        print(f'[WARM START] Strictly loaded trained Prototype: {finetune_from}', flush=True)
        print(f'[FINE-TUNE] backbone_lr={cfg.backbone_lr} head_lr={cfg.head_lr} '
              f'supcon={args.supcon_weight} hard={args.hard_weight}', flush=True)

    start_epoch=0; best_selection=-1.0; global_step=0

    if last_path.exists() and not args.restart:

        st=torch.load(last_path,map_location='cpu',weights_only=False)

        if st.get('format')=='upar2027-track2-abpr-prototype':

            if finetune_from is not None:
                saved_from = st.get('finetune_from')
                if saved_from != str(finetune_from):
                    raise RuntimeError('Resume checkpoint belongs to a different experiment '
                                       f'or baseline: {saved_from!r} != {str(finetune_from)!r}')
                if (float(st.get('supcon_weight', -1)) != args.supcon_weight
                        or float(st.get('hard_weight', -1)) != args.hard_weight):
                    raise RuntimeError('Resume checkpoint contrastive weights differ from current run')
            model.load_state_dict(st['model_state'],strict=True); ema.module.load_state_dict(st['ema_model_state'],strict=True)

            optimizer.load_state_dict(st['optimizer']); scheduler.load_state_dict(st['scheduler'])

            if st.get('scaler'): scaler.load_state_dict(st['scaler'])

            # Mid-epoch checkpoints restart only the current epoch; completed epochs resume at the next one.

            start_epoch=int(st['epoch']) if st.get('mid_epoch',False) else int(st['epoch'])+1

            best_selection=float(st.get('best_selection',-1)); global_step=int(st.get('global_step',0)); restore_rng_state(st.get('rng_state'))

            print(f'✓ Auto-resume {last_path} -> epoch {start_epoch+1}/{cfg.epochs}, global_step={global_step}')

        else:

            print('[warn] checkpoint format differs; starting from pretrained backbone.')



    def save_state(epoch:int, mid_epoch:bool, validation=None):

        payload={

            'format':'upar2027-track2-abpr-prototype','epoch':int(epoch),'mid_epoch':bool(mid_epoch),'global_step':int(global_step),

            'seed':args.seed,'model_state':model.state_dict(),'ema_model_state':ema.module.state_dict(),
            'finetune_from':str(finetune_from) if finetune_from else None,
            'supcon_weight':float(args.supcon_weight),'hard_weight':float(args.hard_weight),

            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),

            'best_selection':float(best_selection),'validation':validation or {},'priors':priors,

            'backbone':cfg.backbone,'embed_dim':cfg.embed_dim,'image_height':cfg.image_height,'image_width':cfg.image_width,

            'config':vars(cfg),'attribute_names':ATTRIBUTE_NAMES,'rng_state':capture_rng_state(),

            'attribute_weights':[1.0]*len(ATTRIBUTE_NAMES),'attribute_temperatures':[1.0]*len(ATTRIBUTE_NAMES),

            'distance_kind':'match','embedding_mix':.0,'prototype_mix':0.0,'positive_rarity_power':0.0,'negative_weight':1.0,'tta_flip':False,

        }

        torch.save(payload,last_path)

        return payload



    for epoch in range(start_epoch,cfg.epochs):

        model.train(); run=0.; steps=0; start=time.time()

        prog=tqdm(train_loader,desc=f'ABPR seed{args.seed} {epoch+1}/{cfg.epochs}',dynamic_ncols=True)

        for b in prog:

            x=b['image'].to(device,non_blocking=True); y=b['target'].to(device); valid=b['valid'].to(device)

            full_q=b['query'].to(device); dom=b['domain'].to(device)

            sparse_q=sample_sparse_queries(full_q,cfg.query_min_known,cfg.query_max_known)

            progress=global_step/max(total_steps,1)

            grl=cfg.grl_lambda*(2/(1+math.exp(-10*progress))-1)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):

                logits,z,feat,proto_logits,attr_feat,_,_=model.encode_images(x, return_prototype=True)

                qz=model.query_encoder(sparse_q)

                # Same gradient-reversal domain regularization as the original forward path.

                dlogits=model.domain_head(grad_reverse(feat,grl))

                l_attr=attr_criterion(logits,y,valid)

                l_group=group_consistency_loss(logits,y,valid)

                l_align=alignment_loss(z,qz)

                l_embed=degree_of_match_contrastive_loss(z,qz,sparse_q,y,valid,temperature=cfg.contrast_temperature,beta=7.0)

                l_prob=probability_degree_match_loss(logits,sparse_q,y,valid)

                l_list=degree_match_listwise_loss(logits,sparse_q,y,valid,temperature=.10,target_beta=8.0)

                l_domain=domain_loss(dlogits,dom)

                l_proto=masked_prototype_bce(proto_logits,y,valid)

                pos_proto,neg_proto=model.prototype_head.normalized_prototypes()

                l_proto_domain=prototype_domain_alignment_loss(attr_feat,y,valid,dom,pos_proto,neg_proto,min_count=2)

                l_proto_sep=prototype_separation_loss(pos_proto,neg_proto,max_cosine=.15)

                proto_ramp=min(1.0,max(0.05,progress/max(cfg.prototype_warmup_fraction,1e-6)))

                loss=(cfg.attr_loss_weight*l_attr + cfg.probability_dom_weight*l_prob + cfg.listwise_dom_weight*l_list +

                      cfg.align_loss_weight*l_align + cfg.dom_contrast_weight*l_embed + cfg.group_loss_weight*l_group +

                      cfg.domain_loss_weight*l_domain + proto_ramp*(cfg.prototype_bce_weight*l_proto +

                      cfg.prototype_domain_weight*l_proto_domain + cfg.prototype_separation_weight*l_proto_sep))

                # DIRECT in-graph fine-tuning: z and qz are live tensors from
                # model.encode_images(x) / model.query_encoder(sparse_q).
                # full_q provides observed TRAIN labels for SupCon; sparse_q
                # is the actual query sent to the query encoder.
                if finetune_from is not None:
                    aa_ramp = min(1.0, (global_step + 1) /
                                  max(1, args.contrastive_warmup_steps))
                    l_supcon = multilabel_supcon(z, full_q) if args.supcon_weight else z.sum()*0.0
                    l_hard = query_hard_negative_loss(z, qz, sparse_q) if args.hard_weight else z.sum()*0.0
                    extra = aa_ramp * (args.supcon_weight*l_supcon + args.hard_weight*l_hard)
                    if not torch.isfinite(extra):
                        raise FloatingPointError('Non-finite attribute-aware contrastive loss')
                    loss = loss + extra
                    if (global_step + 1) % 100 == 0:
                        print(f'[CONTRASTIVE] step={global_step+1} '
                              f'supcon={float(l_supcon.detach()):.4f} '
                              f'hard={float(l_hard.detach()):.4f} '
                              f'weight_ramp={aa_ramp:.3f}', flush=True)

            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)

            scaler.step(optimizer); scaler.update(); scheduler.step(); ema.update(model)

            run+=float(loss.item()); steps+=1; global_step+=1

            prog.set_postfix(loss=f'{loss.item():.4f}',focal=f'{l_attr.item():.3f}',proto=f'{l_proto.item():.3f}',pdom=f'{l_proto_domain.item():.3f}',DoM=f'{l_prob.item():.3f}',rank=f'{l_list.item():.3f}',lr=f'{optimizer.param_groups[0]["lr"]:.1e}')

            if args.save_every_steps>0 and global_step%args.save_every_steps==0:

                save_state(epoch,mid_epoch=True,validation=None)

                print(f'\n✓ recovery checkpoint: {last_path} @ step {global_step}')



        ev=evaluate_retrieval(ema.module,val_loader,device,val_queries,priors,args.max_val_queries,args.seed)

        selection=float(ev['selection']); train_loss=run/max(steps,1)

        if selection>best_selection: best_selection=selection

        payload=save_state(epoch,mid_epoch=False,validation=ev)

        row={'epoch':epoch+1,'global_step':global_step,'train_loss':train_loss,'selection':selection,'prototype_mix':ev.get('prototype_mix',0.0),'elapsed_min':(time.time()-start)/60,**ev['metrics']}

        append_history(hist_path,row)

        print(json.dumps({'epoch':epoch+1,'train_loss':train_loss,'retrieval':ev},indent=2))

        print('✓ saved last:',last_path,'history:',hist_path)

        if selection>=best_selection-1e-12:

            torch.save(payload,best_path); print('★ Saved new best:',best_path)



    if not best_path.exists():

        shutil_target=last_path

        best_path.write_bytes(shutil_target.read_bytes())

    best=torch.load(best_path,map_location='cpu',weights_only=False)

    ema.module.load_state_dict(best['ema_model_state'],strict=True); ema.module=ema.module.to(device).eval()

    final_ev=evaluate_retrieval(ema.module,val_loader,device,val_queries,priors,min(len(val_queries),1000),args.seed)

    best['inference_model_state']={k:v.detach().cpu() for k,v in ema.module.state_dict().items()}

    best['validation']=final_ev; best['priors']=priors

    torch.save(best,best_path)

    (outdir/f'{prefix}_validation.json').write_text(json.dumps(final_ev,indent=2),encoding='utf-8')

    print('\nFINAL BEST:',best_path); print(json.dumps(final_ev,indent=2))





if __name__=='__main__':

    main()
