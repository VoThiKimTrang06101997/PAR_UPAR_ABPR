"""Gate TRAIN-only attribute calibration against original runtime on disjoint public VAL queries.

Reports both semantic-ID ranking and source-domain breakdown; never estimates hidden mADM.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from tqdm.auto import tqdm


def args_parse():
 p=argparse.ArgumentParser()
 p.add_argument('--source-root',type=Path,required=True);p.add_argument('--repo-root',type=Path,required=True)
 p.add_argument('--runtime-zip',type=Path,required=True);p.add_argument('--train-calibrator',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True)
 p.add_argument('--fit-queries',type=int,default=300);p.add_argument('--audit-queries',type=int,default=300)
 p.add_argument('--max-gallery',type=int,default=0);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--min-madm-gain',type=float,default=.003);p.add_argument('--max-map-regression',type=float,default=.003)
 p.add_argument('--max-rank1-regression',type=float,default=.005);p.add_argument('--extra-data-root',type=Path,action='append',default=[])
 return p.parse_args()


def read_trained(path,runtime_sha):
 with np.load(path,allow_pickle=False) as z:
  scale=z['scale'];bias=z['bias'];priors=z['priors'];hashval=str(z['runtime_sha'].item())
 if hashval!=runtime_sha:raise RuntimeError('Calibrator was trained for DIFFERENT runtime checkpoint: retrain.')
 if scale.shape!=(40,) or bias.shape!=(40,) or priors.shape!=(40,):raise ValueError('Bad calibration dimensions')
 return scale,bias,priors


def main():
 a=args_parse();sys.path.insert(0,str(a.source_root))
 from abpr.core import ATTRIBUTE_NAMES, detect_image_column, infer_domain_id
 from abpr.model import track2_metrics
 from abpr.domain_robust_calibration import calibrate_probabilities,query_distance,fuse_distances,semantic_metrics,domain_macro
 from train_domain_robust_calibrator import sha_file
 from calibrate_query_aware import get_features,get_baseline,load_queries,paths_to_ids
 import hashlib
 runtime_sha=sha_file(a.runtime_zip)
 scale,bias,priors=read_trained(a.train_calibrator,runtime_sha)
 source=a.repo_root/'data/annotations/task2/val/gt.csv';frame=pd.read_csv(source)
 col=detect_image_column(frame);rng=np.random.default_rng(a.seed)
 if a.max_gallery>0 and a.max_gallery<len(frame):
  selected=np.sort(rng.choice(len(frame),size=a.max_gallery,replace=False));frame=frame.iloc[selected].reset_index(drop=True)
 ids=paths_to_ids(a.repo_root,frame,col)
 labels=frame[list(ATTRIBUTE_NAMES)].to_numpy(np.float32)
 domains=np.asarray([infer_domain_id(x) for x in frame[col].astype(str)],dtype=np.int32)
 if np.any((domains<0)|(domains>2)):raise ValueError('Unknown public VAL domain')
 queries=load_queries(a.repo_root)
 valid=np.intersect1d(np.unique(ids),np.arange(len(queries))); rng.shuffle(valid)
 required=a.fit_queries+a.audit_queries
 if required>len(valid):raise RuntimeError('Not enough VAL semantic IDs for disjoint FIT/AUDIT')
 fit=valid[:a.fit_queries];audit=valid[a.fit_queries:required];qids=np.r_[fit,audit]
 print(f'Gallery={len(frame)}, total_queries={len(queries)}, FIT={len(fit)}, AUDIT={len(audit)}',flush=True)
 probs,emb=get_features(a,frame,col)
 baseline=get_baseline(a,queries,probs,emb,qids)
 fit_orig,audit_orig=baseline[:len(fit)],baseline[len(fit):]
 def metrics(dist,qindex):
  local=semantic_metrics(dist,qindex,ids)
  diag=domain_macro(dist,qindex,ids,domains)
  challenge=track2_metrics(dist,queries[qindex],labels)
  return {'semantic':local,'domain':diag,'mADM':float(challenge['mADM']),
          'track2_mAP':float(challenge.get('mAP',0.)),
          'track2_Rank_1':float(challenge.get('Rank-1',0.))}
 base_fit=metrics(fit_orig,fit);base_audit=metrics(audit_orig,audit)
 print('BASE FIT:',json.dumps(base_fit,indent=2),flush=True)
 print('BASE AUDIT:',json.dumps(base_audit,indent=2),flush=True)
 candidates=[]
 for shrink in tqdm((0.,.05,.12),desc='Calibration shrink candidates'):
  p2=calibrate_probabilities(probs,scale=scale,bias=bias,priors=priors,shrink=shrink)
  for method in ('expected','loglik'):
   for neg in (.75,1.):
    dd=query_distance(queries[fit],p2,method=method,negative_weight=neg,priors=priors)
    for blend in (.05,.12,.25):
     fused=fuse_distances(fit_orig,dd,blend)
     met=metrics(fused,fit)
     eligible=(met['mADM']>=base_fit['mADM']+.001 and
       met['semantic']['mAP']>=base_fit['semantic']['mAP']-a.max_map_regression and
       met['semantic']['Rank_1']>=base_fit['semantic']['Rank_1']-a.max_rank1_regression and
       met['domain']['macro']['mAP']>=base_fit['domain']['macro']['mAP']-a.max_map_regression)
     candidates.append({'method':method,'shrink':shrink,'negative_weight':neg,'blend':blend,
        'mADM':met['mADM'],'semantic_mAP':met['semantic']['mAP'],
        'semantic_Rank1':met['semantic']['Rank_1'],'domain_macro_mAP':met['domain']['macro']['mAP'],
        'eligible_fit':bool(eligible)})
 eligible=[x for x in candidates if x['eligible_fit']]
 eligible.sort(key=lambda x:(x['mADM']+.35*x['semantic_mAP']+.25*x['domain_macro_mAP'],x['semantic_Rank1']),reverse=True)
 choice=eligible[0] if eligible else None
 audited=None;accepted=False
 if choice is not None:
  p2=calibrate_probabilities(probs,scale=scale,bias=bias,priors=priors,shrink=choice['shrink'])
  alternative=query_distance(queries[audit],p2,method=choice['method'],negative_weight=choice['negative_weight'],priors=priors)
  aud=fuse_distances(audit_orig,alternative,choice['blend'])
  audited=metrics(aud,audit)
  accepted=bool(audited['mADM']>=base_audit['mADM']+a.min_madm_gain and
    audited['semantic']['mAP']>=base_audit['semantic']['mAP']-a.max_map_regression and
    audited['semantic']['Rank_1']>=base_audit['semantic']['Rank_1']-a.max_rank1_regression and
    audited['domain']['macro']['mAP']>=base_audit['domain']['macro']['mAP']-a.max_map_regression and
    audited['domain']['worst_domain']['mAP']>=base_audit['domain']['worst_domain']['mAP']-.015)
 saved=choice if accepted else {'method':'expected','blend':0.,'shrink':0.,'negative_weight':1.}
 a.output.parent.mkdir(exist_ok=True,parents=True)
 np.savez_compressed(a.output,scale=scale,bias=bias,priors=priors,
   method=np.array(saved['method']),blend=np.float32(saved['blend']),
   shrink=np.float32(saved['shrink']),negative_weight=np.float32(saved['negative_weight']),runtime_sha=np.array(runtime_sha))
 report={'model_sha256':runtime_sha,'hidden_score':None,'official_hidden_mADM':None,
   'note':'Public VAL only. Domain macro semantic-ID metrics are diagnostics; no test examples/labels used.',
   'gallery_size':len(frame),'fit_ids':fit.tolist(),'audit_ids':audit.tolist(),
   'baseline_fit':base_fit,'baseline_audit':base_audit,'candidate_audit':audited,
   'best_fit_candidate':choice,'top_fit_candidates':candidates[:5],
   'candidate_accepted':accepted,'selected':saved,
   'audit_madm_gain':(audited['mADM']-base_audit['mADM']) if audited is not None else 0.,
   'audit_semantic_map_gain':(audited['semantic']['mAP']-base_audit['semantic']['mAP']) if audited is not None else 0.}
 a.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
