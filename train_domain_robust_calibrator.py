"""Fit a small, regularized, domain-balanced attribute calibrator on TRAIN images.

It never trains on public VAL or hidden test. The original ConvNeXt weights remain
unchanged. Output is a candidate, gated later using strictly disjoint VAL queries.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,tempfile,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch
from tqdm.auto import tqdm


def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument('--source-root',type=Path,required=True);p.add_argument('--repo-root',type=Path,required=True)
 p.add_argument('--runtime-zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--max-per-domain',type=int,default=5500)
 p.add_argument('--steps',type=int,default=600);p.add_argument('--lr',type=float,default=.015)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--extra-data-root',type=Path,action='append',default=[])
 return p.parse_args()


def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()


def get_features(a,frame,col):
 from abpr.core import ImageResolver
 a.cache_dir.mkdir(parents=True,exist_ok=True)
 runtime_digest=sha_file(a.runtime_zip)
 h=hashlib.sha256((runtime_digest+'|'.join(frame[col].astype(str))).encode()).hexdigest()[:18]
 output=a.cache_dir/f'train_pixels_{h}.npz'
 if output.is_file():
  with np.load(output,allow_pickle=False) as z: probs=z['probs']
  if probs.shape==(len(frame),40) and np.isfinite(probs).all():
   print('TRAIN feature cache HIT:',output,flush=True);return probs,runtime_digest
 resolver=ImageResolver(a.repo_root/'data',a.repo_root,extra_roots=a.extra_data_root)
 records=[]
 for raw in tqdm(frame[col].astype(str),desc='Resolve real TRAIN pixels'):
  path=Path(resolver.resolve(raw))
  if not path.is_file():raise FileNotFoundError(f'TRAIN image missing: {raw}')
  records.append({'image_path':str(path)})
 with tempfile.TemporaryDirectory(prefix='abpr_domain_robust_') as tmp:
  tmp=Path(tmp)
  with zipfile.ZipFile(a.runtime_zip) as z:
   for name in ('abpr_runtime.py','assets/model.pt'):
    if name not in z.namelist():raise FileNotFoundError(f'Runtime missing {name}')
    path=tmp/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
  sys.path.insert(0,str(tmp))
  from abpr_runtime import ABPRRuntime
  model=ABPRRuntime(tmp/'assets/model.pt')
  feats=model.encode_gallery(records)
  if not isinstance(feats,tuple):raise TypeError('Runtime must return (probs,emb)')
  probs=feats[0]
  if hasattr(probs,'detach'):probs=probs.detach().cpu().numpy()
  probs=np.asarray(probs,np.float32)
 if probs.shape!=(len(frame),40) or not np.isfinite(probs).all():raise RuntimeError('Invalid TRAIN feature shape or values')
 np.savez_compressed(output,probs=probs)
 print('Cached TRAIN real pixel probabilities:',output,flush=True)
 return probs,runtime_digest


def fit_attribute_calibrator(probs,y,domains,steps=600,lr=.015,seed=42):
 from torch.nn import functional as F
 rng=np.random.default_rng(seed)
 y=np.asarray(y,np.float32);p=np.asarray(probs,np.float32);d=np.asarray(domains,np.int32)
 valid=np.isin(y,(0,1));yp=np.clip(y,0,1)
 if p.shape!=y.shape or p.shape[1]!=40 or len(d)!=len(y):raise ValueError('TRAIN labels and predictions misaligned')
 ids_fit=[];ids_hold=[]
 for domain in sorted(set(d.tolist())):
  pool=np.where(d==domain)[0];rng.shuffle(pool)
  n=max(1,int(len(pool)*.2));ids_hold.extend(pool[:n]);ids_fit.extend(pool[n:])
 if not ids_hold or not ids_fit:raise RuntimeError('Not enough data for stratified TRAIN holdout')
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 xp=torch.as_tensor(np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1-1e-5)),device=dev)
 yy=torch.as_tensor(yp,device=dev);mask=torch.as_tensor(valid.astype(np.float32),device=dev)
 tdomain=torch.as_tensor(d,device=dev)
 fit_ix=torch.as_tensor(ids_fit,dtype=torch.long,device=dev)
 hold_ix=torch.as_tensor(ids_hold,dtype=torch.long,device=dev)
 log_s=torch.nn.Parameter(torch.zeros(40,device=dev));b=torch.nn.Parameter(torch.zeros(40,device=dev))
 opt=torch.optim.AdamW([log_s,b],lr=lr,weight_decay=0.0)
 unique=np.unique(d)
 def loss_for(idx,grad):
  a=torch.exp(log_s);logits=xp[idx]*a+b
  elem=F.binary_cross_entropy_with_logits(logits,yy[idx],reduction='none')*mask[idx]
  doms=tdomain[idx]
  per=[]
  for k in unique:
   j=(doms==int(k));den=mask[idx][j].sum().clamp(min=1.)
   per.append(elem[j].sum()/den)
  z=torch.stack(per)
  base=.7*z.mean()+.3*torch.logsumexp(z*8,dim=0)/8
  penalty=.03*((a-1)**2).mean()+.01*(b**2).mean()
  return base+(penalty if grad else 0),z
 with torch.no_grad():
  init,init_domains=loss_for(hold_ix,False)
 print('TRAIN holdout initial BCE:',float(init),'domain BCE:',init_domains.cpu().numpy().round(4).tolist(),flush=True)
 best=float(init);saved=(np.ones(40,np.float32),np.zeros(40,np.float32));history=[]
 for step in tqdm(range(1,steps+1),desc='TRAIN domain-balanced attribute calibration'):
  opt.zero_grad(set_to_none=True)
  loss,_=loss_for(fit_ix,True)
  if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
  loss.backward();opt.step()
  with torch.no_grad():
   log_s.clamp_(-0.75,0.75);b.clamp_(-1.5,1.5)
  if step%25==0 or step==steps:
   with torch.no_grad():
    held,values=loss_for(hold_ix,False)
    score=float(held)
    history.append({'step':step,'holdout_bce':score,'per_domain_bce':values.detach().cpu().numpy().tolist()})
    if score<best-0.00005:
     best=score;saved=(torch.exp(log_s).cpu().numpy().copy(),b.cpu().numpy().copy())
 print('TRAIN holdout best BCE:',best,'improvement:',float(init)-best,flush=True)
 return saved,dict(initial=float(init),best=best,improvement=float(init)-best,history=history,
                   train_images=int(len(fit_ix)),holdout_images=int(len(hold_ix)),domain_counts={str(k):int((d==k).sum()) for k in unique})


def main():
 a=parse_args();sys.path.insert(0,str(a.source_root))
 from abpr.core import ATTRIBUTE_NAMES,detect_image_column,infer_domain_id
 train=a.repo_root/'data/annotations/task2/train/gt.csv'
 if not train.is_file():raise FileNotFoundError(train)
 df=pd.read_csv(train);col=detect_image_column(df)
 rng=np.random.default_rng(a.seed);indices=[]
 for domain in (0,1,2):
  pool=np.asarray([i for i,v in enumerate(df[col].astype(str)) if infer_domain_id(v)==domain],dtype=np.int32)
  if len(pool)<100:raise RuntimeError(f'Insufficient genuine TRAIN images for domain {domain}: {len(pool)}')
  selected=rng.choice(pool,size=min(a.max_per_domain,len(pool)),replace=False)
  indices.extend(selected.tolist())
 frame=df.iloc[sorted(indices)].reset_index(drop=True)
 y=frame[list(ATTRIBUTE_NAMES)].to_numpy(np.float32)
 mask=np.isin(y,(0,1));priors=((y==1)&mask).sum(0).astype(np.float32)/(mask.sum(0)+1e-6)
 priors=np.clip(priors,.02,.98)
 domain=np.asarray([infer_domain_id(v) for v in frame[col].astype(str)],dtype=np.int32)
 print('Training calibrator on real TRAIN images:',len(frame), 'domain counts',np.bincount(domain,minlength=3).tolist(),flush=True)
 probs,runtime_sha=get_features(a,frame,col)
 (scale,bias),report=fit_attribute_calibrator(probs,y,domain,steps=a.steps,lr=a.lr,seed=a.seed)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output,scale=scale,bias=bias,priors=priors,runtime_sha=np.array(runtime_sha))
 report.update({'source':'REAL TRAIN images only','runtime_sha256':runtime_sha,'checkpoint':str(a.output),'trained_weights_modified':False})
 a.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)

if __name__=='__main__':main()
