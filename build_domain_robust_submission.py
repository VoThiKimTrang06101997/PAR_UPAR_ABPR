"""Build a 5-file, self-contained Track-2 submission with an audit-gated scorer.

No extra ranker assets at inference. Baseline model is never modified. ZIP root
contains run.py, abpr_runtime.py, requirements.txt, metadata.yaml, assets/model.pt.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np,pandas as pd


def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--source-root',type=Path,required=True);p.add_argument('--repo-root',type=Path,required=True)
 p.add_argument('--runtime-zip',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
 p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--smoke-real',action='store_true');p.add_argument('--allow-baseline-build',action='store_true')
 p.add_argument('--extra-data-root',type=Path,action='append',default=[])
 args=p.parse_args();src=args.source_root
 meta=args.repo_root/'examples/task2/sample_code_submission/metadata.yaml'
 for path in (meta,args.runtime_zip,args.calibration,src/'submission_run.py',src/'abpr/domain_robust_calibration.py'):
  if not path.is_file():raise FileNotFoundError(path)
 reportpath=args.calibration.with_suffix('.json')
 if not reportpath.is_file():raise FileNotFoundError(reportpath)
 report=json.loads(reportpath.read_text(encoding='utf-8'))
 accepted=report.get('candidate_accepted') is True
 if not accepted and not args.allow_baseline_build:
  raise RuntimeError('Candidate FAILED public AUDIT; refusing candidate submission. Use --allow-baseline-build only for original baseline.')
 model_sha=sha_file(args.runtime_zip)
 if model_sha!=report.get('model_sha256'):raise RuntimeError('Runtime checksum differs from validated model; calibration cannot be reused')
 with np.load(args.calibration,allow_pickle=False) as arr:
  config={k:float(arr[k]) for k in ('blend','shrink','negative_weight')}
  config.update(method=str(arr['method'].item()),scale=arr['scale'].astype(float).tolist(),
                bias=arr['bias'].astype(float).tolist(),priors=arr['priors'].astype(float).tolist())
  if str(arr['runtime_sha'].item())!=model_sha:raise RuntimeError('Trained calibrator SHA differs from runtime')
 if not accepted and config['blend']>0:raise RuntimeError('Unaccepted calibration is not baseline')
 if config['blend']<0 or config['blend']>1:raise ValueError('Invalid blend')
 run= (src/'submission_run.py').read_text(encoding='utf-8')
 first='    # Optional TRAIN-supervised attribute-ranker.'
 last='    expected=(len(queries),len(gallery))'
 if first not in run or last not in run:raise RuntimeError('submission_run.py marker mismatch: inspect actual repo source before patching')
 begin=run.index(first);end=run.index(last,begin)
 if accepted:
  inject='''    # Domain-robust score, TRAIN-supervised and disjoint VAL-audited.
    cfg = _DR_CONFIG
    adjusted = calibrate_probabilities(probs, scale=cfg['scale'], bias=cfg['bias'],
                                       priors=cfg['priors'], shrink=cfg['shrink'])
    novel = query_distance(queries_model, adjusted, method=cfg['method'],
                           negative_weight=cfg['negative_weight'], priors=cfg['priors'])
    scores = fuse_distances(scores, novel, cfg['blend'])
    _log(f"Domain-robust audit-selected calibration active: blend={cfg['blend']:.2f}")
'''
 else:inject='    # Strict original baseline distance; no unapproved postprocessing.\n'
 run=run[:begin]+inject+run[end:]
 if accepted:
  source_text=(src/'abpr/domain_robust_calibration.py').read_text(encoding='utf-8')
  tree=ast.parse(source_text)
  helpers=('ensure','calibrate_probabilities','query_distance','fuse_distances')
  defs={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
  if any(x not in defs for x in helpers):raise ValueError('Ranking implementation missing')
  code='\n\n'.join(ast.get_source_segment(source_text,defs[x]) for x in helpers)
  run+='\n\nN_ATTR=40\n_DR_CONFIG='+repr(config)+'\n\n'+code+'\n'
 for banned in ('from query_aware_ranking import',"'query_aware.npz'","'ranker.npz'","'contrastive.npz'"):
  if banned in run:raise RuntimeError(f'Unremoved optional scorer dependency: {banned}')
 compile(run,'run.py','exec')
 output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
 tmp=output/'submission_building'
 if tmp.exists():shutil.rmtree(tmp)
 (tmp/'assets').mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(args.runtime_zip) as z:
  required={'abpr_runtime.py','assets/model.pt'}
  if not required.issubset(z.namelist()):raise FileNotFoundError('Runtime lacks assets/model.pt or abpr_runtime.py')
  for file in required:(tmp/file).write_bytes(z.read(file))
  if 'requirements.txt' in z.namelist():
   (tmp/'requirements.txt').write_bytes(z.read('requirements.txt'))
  elif (src/'requirements.txt').is_file():shutil.copyfile(src/'requirements.txt',tmp/'requirements.txt')
  else:raise FileNotFoundError('No requirements.txt in runtime or repo')
 (tmp/'run.py').write_text(run,encoding='utf-8')
 (tmp/'metadata.yaml').write_bytes(meta.read_bytes())
 allowed={'run.py','metadata.yaml','requirements.txt','abpr_runtime.py','assets/model.pt'}
 actual={p.relative_to(tmp).as_posix() for p in tmp.rglob('*') if p.is_file()}
 if actual!=allowed:raise RuntimeError(f'Unexpected submission entries {actual.symmetric_difference(allowed)}')
 if args.smoke_real:
  sys.path.insert(0,str(src))
  from abpr.core import detect_image_column,ImageResolver,ATTRIBUTE_NAMES
  data=pd.read_csv(args.repo_root/'data/annotations/task2/val/gt.csv')
  col=detect_image_column(data);resolver=ImageResolver(args.repo_root/'data',args.repo_root,extra_roots=args.extra_data_root)
  indices=np.linspace(0,len(data)-1,8,dtype=int)
  sample={'gallery':[{'image_path':str(resolver.resolve(str(data.iloc[int(i)][col])))} for i in indices],
    'queries':[data.iloc[int(i)][list(ATTRIBUTE_NAMES)].astype(float).replace({np.nan:-1}).to_numpy().tolist() for i in (indices[0],indices[-1])],
    'attribute_names':list(ATTRIBUTE_NAMES)}
  # Construct a fresh process as Codabench would, eliminating in-memory imports.
  with tempfile.TemporaryDirectory() as work:
   payload=Path(work)/'sample.json';payload.write_text(json.dumps(sample),encoding='utf-8')
   code="""import json,numpy as np,run,sys
sample=json.load(open(sys.argv[1]));d=np.asarray(run.rank_gallery(sample)['distances']);assert d.shape==(2,8) and np.isfinite(d).all();print('REAL PIXEL SMOKE PASSED',d.shape,d.std())
"""
   r=subprocess.run([sys.executable,'-c',code,str(payload)],cwd=str(tmp),text=True,capture_output=True,timeout=900)
   print(r.stdout,flush=True)
   if r.returncode:raise RuntimeError('Smoke failed.\n'+r.stderr[-7000:])
 submission=output/'submission'
 if submission.exists():shutil.rmtree(submission)
 tmp.rename(submission)
 zip_path=output/'Submission_ABPR.zip'
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for file in sorted(submission.rglob('*')):
   if file.is_file():z.write(file,file.relative_to(submission).as_posix())
 with zipfile.ZipFile(zip_path) as z:
  if z.testzip():raise RuntimeError('Corrupted output ZIP')
  if set(z.namelist())!=allowed:raise RuntimeError('Incorrect ZIP root files')
  if z.read('metadata.yaml')!=meta.read_bytes():raise RuntimeError('Metadata mismatch')
 manifest=dict(accepted=accepted,sha256_model_zip=model_sha,files=sorted(allowed),submission=str(zip_path),smoke_real=args.smoke_real)
 (output/'Submission_ABPR_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('CLEAN SUBMISSION:',submission,'\nFINAL ZIP:',zip_path,'\nMANIFEST:',manifest,flush=True)

if __name__=='__main__':main()
