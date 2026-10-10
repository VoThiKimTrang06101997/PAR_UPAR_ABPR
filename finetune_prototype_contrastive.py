"""Run official Prototype trainer with native contrastive fine-tune options.

Unlike the former launcher, this file does NOT rewrite functions via AST, monkeypatch
`train_abpr.main`, or depend on a variable literally named `query`.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser(description='Warm-start Prototype and fine-tune its backbone')
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--baseline-checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--result-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--eval-batch-size',type=int,default=16)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--backbone',default='convnext_small')
    p.add_argument('--embed-dim',type=int,default=320)
    p.add_argument('--image-height',type=int,default=384)
    p.add_argument('--image-width',type=int,default=192)
    p.add_argument('--max-val-queries',type=int,default=500)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--supcon-weight',type=float,default=.40)
    p.add_argument('--hard-weight',type=float,default=.70)
    p.add_argument('--warmup-steps',type=int,default=250)
    p.add_argument('--save-every-steps',type=int,default=300)
    p.add_argument('--extra-data-root',action='append',default=[])
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--restart',action='store_true')
    return p.parse_args()


def verify_training_graph(path:Path):
    source=path.read_text(encoding='utf-8')
    tree=ast.parse(source)
    main=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'),None)
    if main is None:
        raise RuntimeError('train_abpr.py lacks main()')
    names={n.id for n in ast.walk(main) if isinstance(n,ast.Name)}
    required={'full_q','sparse_q','z','qz','loss','l_supcon','l_hard','extra','finetune_from'}
    missing=required-names
    if missing:
        raise RuntimeError('Native trainer patch not installed. Missing: '+repr(sorted(missing)))
    if not any(isinstance(n,ast.AugAssign) for n in ast.walk(main)):
        pass
    if not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets)
               and isinstance(n.value,ast.BinOp) and isinstance(n.value.op,ast.Add)
               and isinstance(n.value.left,ast.Name) and n.value.left.id=='loss'
               and isinstance(n.value.right,ast.Name) and n.value.right.id=='extra'
               for n in ast.walk(main)):
        raise RuntimeError('Expected loss = loss + extra in original training graph')
    if not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='backward'
               for n in ast.walk(main)):
        raise RuntimeError('Training graph has no backward() call')
    args={a.dest if isinstance(a,ast.Attribute) else None for a in []}
    for flag in ('--finetune-from','--supcon-weight','--hard-weight','--contrastive-warmup-steps'):
        if flag not in source:
            raise RuntimeError(f'Missing trainer CLI option {flag}')
    return True


def run(a):
    root=a.source_root.resolve(); trainer=root/'train_abpr.py'; baseline=a.baseline_checkpoint.resolve()
    ckdir=a.checkpoint_dir.resolve(); result=a.result_dir.resolve()
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    if not baseline.is_file():
        raise FileNotFoundError(baseline)
    if a.restart:
        raise RuntimeError('Restart is disabled for baseline safety. Use a different new fine-tune directory.')
    if ckdir==baseline.parent:
        raise RuntimeError('Do not overwrite baseline checkpoint directory')
    if not a.epochs>0 or not a.lr>0:
        raise ValueError('Epochs and LR must be positive')
    if not (a.supcon_weight>=0 and a.hard_weight>=0 and a.supcon_weight+a.hard_weight>0):
        raise ValueError('At least one contrastive loss weight must be positive')
    verify_training_graph(trainer)
    official=a.repo_root.resolve()
    for kind in ('train','val'):
        p=official/'data'/'annotations'/'task2'/kind/'gt.csv'
        if not p.is_file():
            raise FileNotFoundError(p)
    if a.audit_only:
        import torch
        checkpoint=torch.load(baseline,map_location='cpu',weights_only=False)
        if not isinstance(checkpoint,dict):
            raise ValueError('Baseline checkpoint must be a dict')
        if checkpoint.get('backbone',a.backbone)!=a.backbone:
            raise RuntimeError('Baseline backbone mismatch')
        if int(checkpoint.get('embed_dim',a.embed_dim))!=a.embed_dim:
            raise RuntimeError('Baseline embedding dimension mismatch')
        key=next((k for k in ('inference_model_state','ema_model_state','model_state')
                  if isinstance(checkpoint.get(k),dict)),None)
        if key is None:
            raise RuntimeError('Baseline checkpoint missing trained state')
        print(json.dumps({'audit':'PASSED','method':'NATIVE training graph (no AST injection)',
                          'query_tensor':'sparse_q','labels_for_supcon':'full_q',
                          'baseline_state':key,'baseline':str(baseline),
                          'parameters':len(checkpoint[key])},indent=2),flush=True)
        return 0
    ckdir.mkdir(parents=True,exist_ok=True); result.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(trainer),
         '--repo-root',str(official),'--checkpoint-dir',str(ckdir),
         '--result-dir',str(result),'--finetune-from',str(baseline),
         '--seed',str(a.seed),'--epochs',str(a.epochs),
         '--batch-size',str(a.batch_size),'--eval-batch-size',str(a.eval_batch_size),
         '--num-workers',str(a.num_workers),'--backbone',str(a.backbone),
         '--embed-dim',str(a.embed_dim),'--image-height',str(a.image_height),
         '--image-width',str(a.image_width),'--max-val-queries',str(a.max_val_queries),
         '--backbone-lr',str(a.lr),'--head-lr',str(a.lr),
         '--supcon-weight',str(a.supcon_weight),'--hard-weight',str(a.hard_weight),
         '--contrastive-warmup-steps',str(a.warmup_steps),
         '--save-every-steps',str(a.save_every_steps)]
    for p in a.extra_data_root:
        if Path(p).is_dir():cmd.extend(['--extra-data-root',str(p)])
    print('RUNNING NATIVE Prototype TRAINER:', ' '.join(cmd),flush=True)
    process=subprocess.run(cmd,cwd=str(root),env=dict(os.environ,PYTHONUNBUFFERED='1'))
    return process.returncode


if __name__=='__main__':
    sys.exit(run(parse_args()))
