"""Train attribute-aware multi-positive contrastive retrieval + hard negatives.

Image embeddings are extracted from the existing trained Prototype ABPR checkpoint.
This script TRAINs a new image projection and 40 positive/negative attribute
prototypes. Source TRAIN is the only optimization data. Source VAL is used for
choosing the blend against the existing model (and optional reliability ranker).
No IDs, identities or labels from hidden/test data are used.
"""
from __future__ import annotations
import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from abpr.contrastive_reranker import ContrastiveRanker, fuse_distances
from abpr.retrieval_reranker import AttributeRanker, fuse_distances as fuse_reliability
from train_retrieval_ranker import (
    load_data,select_evenly,create_gallery_runtime,encode_images,
    exact_validation_queries,evaluate_exact_retrieval,
)


class ContrastiveHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.proj=torch.nn.Linear(input_dim,output_dim,bias=False)
        torch.nn.init.orthogonal_(self.proj.weight)
        self.positive=torch.nn.Parameter(torch.randn(40,output_dim)*0.02)
        self.negative=torch.nn.Parameter(torch.randn(40,output_dim)*0.02)

    def image(self,x):
        return F.normalize(self.proj(F.normalize(x,dim=-1)),dim=-1)

    def query(self,q):
        valid=(q >= 0)&(q <= 1)
        pos=((q > 0.5)&valid).float()
        neg=((q <= 0.5)&valid).float()
        raw=pos @ self.positive + neg @ self.negative
        return F.normalize(raw,dim=-1)


def synth_queries(labels, rng, min_known=3, max_known=9):
    """Make masked positive/negative queries from public TRAIN labels only."""
    q=np.full(labels.shape,-1.0,dtype=np.float32)
    for i in range(len(q)):
        known=np.flatnonzero((labels[i]>=0)&(labels[i]<=1))
        if len(known)<min_known:continue
        count=min(len(known),int(rng.integers(min_known,max_known+1)))
        cols=rng.choice(known,size=count,replace=False)
        q[i,cols]=labels[i,cols]
    return q


def contrastive_losses(head, feats, labels, query, domains, temperature=0.10,
                       use_image_supcon=True):
    """Masked multi-positive InfoNCE + query-aware hard negatives + label SupCon.

    Unknown training labels (-1) never create a supervised negative.
    Other true positives in a batch are not treated as negatives.
    """
    device=feats.device
    n=len(feats)
    img=head.image(feats)
    qemb=head.query(query)
    active=(query>=0)&(query<=1)                      # [B,A]
    avail=(labels>=0)&(labels<=1)                    # [B,A]
    # [query_i, image_j, attribute_k]
    known=active[:,None,:]&avail[None,:,:]
    mismatch=known & (torch.abs(query[:,None,:]-labels.clamp(0,1)[None,:,:])>0.5)
    fully_known=(known.sum(-1)==active.sum(-1)[:,None])
    positive=fully_known & ~mismatch.any(-1) & (active.sum(-1)[:,None]>=2)
    negative=fully_known & mismatch.any(-1)
    valid=positive|negative
    diag=torch.eye(n,device=device,dtype=torch.bool)
    # query_i intentionally comes from label_i (provided >=3 known)
    positive=positive|diag
    valid=valid|diag
    logits=(qemb @ img.T)/temperature
    logits=logits.masked_fill(~valid,-1e4)
    # Multi-positive supervised InfoNCE, NOT an ordinary single-identity CLIP loss.
    logp=F.log_softmax(logits,dim=1)
    targets=positive.float()/positive.sum(1,keepdim=True).clamp_min(1.)
    query_loss=-(targets*logp).sum(1).mean()
    # Hard-negative mining among candidates that actually disagree on known bits.
    hard_logits=logits.masked_fill(~negative,-1e4)
    hardest=hard_logits.max(dim=1).values
    pos_sim=(qemb*img).sum(-1)/temperature
    good=negative.any(dim=1)
    hard_loss=(F.softplus((hardest[good]-pos_sim[good]+1.0))
               .mean() if good.any() else 0.*query_loss)
    # Soft label-aware image-image contrastive positives with cross-domain boost.
    label_loss=0.*query_loss
    if use_image_supcon:
        both=avail[:,None,:]&avail[None,:,:]
        agreements=((labels[:,None,:]-labels[None,:,:]).abs()<0.5)&both
        count=both.sum(-1)
        overlap=agreements.sum(-1)/count.clamp_min(1)
        positives=(count>=5)&(overlap>=0.80)&(~diag)
        # Domain-balanced positives are prioritized without excluding same-domain.
        weights=positives.float()*(1.0+0.5*(domains[:,None]!=domains[None,:]).float())
        use=weights.sum(-1)>0
        if use.any():
            image_logits=(img@img.T)/temperature
            image_logits=image_logits.masked_fill(diag | (count<5),-1e4)
            image_logp=F.log_softmax(image_logits,dim=1)
            normalized=weights/weights.sum(-1,keepdim=True).clamp_min(1)
            label_loss=-(normalized[use]*image_logp[use]).sum(-1).mean()
    # Prevent the projection from discarding all teacher feature geometry.
    with torch.no_grad():base_sim=F.normalize(feats,dim=-1)@F.normalize(feats,dim=-1).T
    student_sim=img@img.T
    preserve_loss=(student_sim-base_sim).square().mean()
    total=query_loss+0.30*hard_loss+0.12*label_loss+0.15*preserve_loss
    return total,{'qi':float(query_loss.detach()),'hard':float(hard_loss.detach()),
                  'supcon':float(label_loss.detach()),'preserve':float(preserve_loss.detach()),
                  'mined_query_count':int(good.sum().detach())}


def train_contrastive(emb, labels, domains, steps=1400, seed=42,
                      output_dim=128,batch_size=128,lr=0.001,device='cpu',
                      resume_path=None,save_every=200,fingerprint=None):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    labels=np.asarray(labels,dtype=np.float32)
    eligible=np.flatnonzero(((labels>=0)&(labels<=1)).sum(1)>=3)
    if len(eligible)<batch_size:raise RuntimeError('Not enough public TRAIN attribute-labeled images')
    inp=torch.as_tensor(np.asarray(emb,dtype=np.float32),device=device)
    labs=torch.as_tensor(labels,device=device)
    domains=torch.as_tensor(domains,dtype=torch.long,device=device)
    head=ContrastiveHead(inp.shape[1],output_dim).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.002)
    logs=[]
    start_step=0
    if resume_path is not None:
        resume_path=Path(resume_path)
        if resume_path.is_file():
            state=torch.load(resume_path,map_location=device,weights_only=False)
            if state.get('fingerprint') != fingerprint or state.get('input_dim')!=inp.shape[1] or state.get('output_dim')!=output_dim:
                raise RuntimeError('Resume state belongs to another runtime/feature configuration. Move it or pass --reset-train-state.')
            head.load_state_dict(state['head'])
            optimizer.load_state_dict(state['optimizer'])
            rng.bit_generator.state=state['rng']
            logs=state.get('logs',[])
            start_step=state['step']
            print(f'Auto-resumed contrastive head from step {start_step}/{steps}: {resume_path}',flush=True)
    bar=tqdm(range(start_step,steps),initial=start_step,total=steps,desc='Train multi-label contrastive + hard negatives',unit='step')
    for step in bar:
        rows=rng.choice(eligible,size=batch_size,replace=False)
        y=labels[rows]
        query=synth_queries(y,rng)
        idx=torch.as_tensor(rows,dtype=torch.long,device=device)
        qt=torch.as_tensor(query,device=device)
        loss,details=contrastive_losses(head,inp[idx],labs[idx],qt,domains[idx])
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite contrastive loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.0)
        optimizer.step()
        if (step+1)%25==0:
            bar.set_postfix(loss=f'{loss.item():.3f}',hard=details['mined_query_count'])
        if (step+1)%100==0 or step==steps-1:
            logs.append({'step':step+1,'loss':round(float(loss.detach()),5),**details})
        if resume_path is not None and ((step+1)%save_every==0 or step==steps-1):
            state={'head':head.state_dict(),'optimizer':optimizer.state_dict(),
                   'rng':rng.bit_generator.state,'logs':logs,'step':step+1,
                   'fingerprint':fingerprint,'input_dim':inp.shape[1],'output_dim':output_dim}
            resume_path.parent.mkdir(parents=True,exist_ok=True)
            atomic=resume_path.with_suffix('.tmp')
            torch.save(state,atomic)
            atomic.replace(resume_path)
    return ContrastiveRanker(
        projection=head.proj.weight.detach().cpu().numpy().T,
        pos_proto=head.positive.detach().cpu().numpy(),
        neg_proto=head.negative.detach().cpu().numpy(),
        source='public_train',temperature=0.10),logs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--runtime-zip',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--reliability-checkpoint',type=Path,default=None)
    parser.add_argument('--extra-data-root',type=Path,action='append',default=[])
    parser.add_argument('--batch-size',type=int,default=48)
    parser.add_argument('--max-train-images',type=int,default=18000)
    parser.add_argument('--max-val-images',type=int,default=9000)
    parser.add_argument('--val-queries',type=int,default=320)
    parser.add_argument('--steps',type=int,default=1400)
    parser.add_argument('--train-batch-size',type=int,default=128)
    parser.add_argument('--output-dim',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--min-val-gain',type=float,default=0.005)
    parser.add_argument('--train-state',type=Path,default=None)
    parser.add_argument('--reset-train-state',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.source_root))
    from abpr.core import ATTRIBUTE_NAMES,ImageResolver
    names=list(ATTRIBUTE_NAMES)
    if len(names)!=40:raise RuntimeError(f'Expected 40 official attributes, got {len(names)}')
    args.cache_dir.mkdir(exist_ok=True,parents=True)
    runtime=create_gallery_runtime(args.runtime_zip,args.cache_dir/'ranker_runtime')
    resolver=ImageResolver(args.repo_root/'data',args.repo_root,extra_roots=args.extra_data_root)
    fingerprint=hashlib.sha256((str(args.runtime_zip.resolve())+str(args.runtime_zip.stat().st_mtime_ns)+str(args.runtime_zip.stat().st_size)).encode()).hexdigest()[:12]
    train_df,train_labels,train_paths=load_data(args.repo_root,'train',names)
    val_df,val_labels,val_paths=load_data(args.repo_root,'val',names)
    train_ids=select_evenly(train_df,args.max_train_images,args.seed)
    val_ids=select_evenly(val_df,args.max_val_images,args.seed+1)
    tp,te=encode_images(runtime,[train_paths[i] for i in train_ids],resolver,args.batch_size,
                       args.cache_dir/f'train_probs_{fingerprint}_{len(train_ids)}.npz')
    vp,ve=encode_images(runtime,[val_paths[i] for i in val_ids],resolver,args.batch_size,
                       args.cache_dir/f'val_probs_{fingerprint}_{len(val_ids)}.npz')
    # Make domain labels from image directory, no domain leakage from VAL.
    domains=np.array([str(train_paths[i]).split('/')[0].lower() for i in train_ids])
    _,domain_ids=np.unique(domains,return_inverse=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Prototype cache: train={len(tp)} val={len(vp)} emb_dim={te.shape[1]} device={device}',flush=True)
    if args.reset_train_state and args.train_state is not None and args.train_state.exists():
        args.train_state.unlink()
    ranker,history=train_contrastive(te,train_labels[train_ids],domain_ids,
                         steps=args.steps,seed=args.seed,output_dim=args.output_dim,
                         batch_size=args.train_batch_size,device=device,
                         resume_path=args.train_state,fingerprint=fingerprint)
    q,vg_ids,vq_ids=exact_validation_queries(args.repo_root,val_ids,args.seed+2,args.val_queries)
    with torch.no_grad():
        base=runtime.distance(torch.as_tensor(q,dtype=torch.float32),
                              torch.as_tensor(vp,dtype=torch.float32),
                              torch.as_tensor(ve,dtype=torch.float32))
        if hasattr(base,'detach'):base=base.detach().cpu().numpy()
    base=np.asarray(base,dtype=np.float32)
    if args.reliability_checkpoint is not None:
        reliability=AttributeRanker.load(args.reliability_checkpoint)
        if reliability.blend>0:
            base=fuse_reliability(base,reliability.distance(q,vp),reliability.blend)
    novel=ranker.distance(q,ve)
    candidates=[]
    for strength in [0.,0.1,0.2,0.35,0.5,0.7,1.0]:
        score=evaluate_exact_retrieval(fuse_distances(base,novel,strength),vg_ids,vq_ids)
        candidates.append({'blend':strength,**score})
    best=max(candidates,key=lambda d:(d['val_subset_mAP'],d['val_subset_R1']))
    valid_gain=(best['val_subset_mAP'] > candidates[0]['val_subset_mAP'] + args.min_val_gain)
    rank1_guard=(best['val_subset_R1'] >= candidates[0]['val_subset_R1'] - 0.01)
    ranker.blend=float(best['blend'] if valid_gain and rank1_guard else 0.0)
    ranker.best_val_map=float(best['val_subset_mAP'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    ranker.save(args.output)
    report={'method':'multi-label multi-positive contrastive + query-aware hard-negative + soft label SupCon',
            'source_train_images':len(train_ids),'public_val_images':len(val_ids),
            'embedding_dim':int(te.shape[1]),'output_dim':args.output_dim,'train_steps':args.steps,
            'baseline_reliability_included':args.reliability_checkpoint is not None,
            'baseline':candidates[0],'val_trials':candidates,'best_blend':ranker.blend,
            'min_val_gain':args.min_val_gain,'rank1_no_regression_guard':0.01,
            'selection_metric':'official query IDs subset mAP; NOT hidden mADM',
            'train_history':history,'checkpoint':str(args.output)}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
