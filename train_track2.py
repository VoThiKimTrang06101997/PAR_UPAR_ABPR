"""Launch the repository's real prototype ABPR training, not inference-only tuning.

Preserves original train_abpr.py, which owns the prototype losses and model definition.
No hidden-test data, no shortcut to leaderboard scores. Training resumes from *_last.pt.
"""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path


def run(command,cwd,log_path):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print('\n>>>',' '.join(map(str,command)),flush=True)
    env=dict(os.environ,PYTHONUNBUFFERED='1')
    proc=subprocess.Popen(command,cwd=str(cwd),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace',bufsize=1)
    try:
        with log_path.open('a',encoding='utf-8') as log:
            for line in proc.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired:proc.kill()
        raise
    if code:raise subprocess.CalledProcessError(code,command)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--result-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,action='append',default=None)
    p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--batch-size',type=int,default=12)
    p.add_argument('--eval-batch-size',type=int,default=24)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--backbone',default='convnext_small')
    p.add_argument('--image-height',type=int,default=384)
    p.add_argument('--image-width',type=int,default=192)
    p.add_argument('--embed-dim',type=int,default=320)
    p.add_argument('--max-val-queries',type=int,default=500)
    p.add_argument('--save-every-steps',type=int,default=500)
    p.add_argument('--restart',action='store_true')
    p.add_argument('--extra-data-root',type=Path,action='append',default=[])
    a=p.parse_args()
    source=a.source_root.resolve(); official=a.repo_root.resolve()
    a.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    a.result_dir.mkdir(parents=True,exist_ok=True)
    train=official/'data/annotations/task2/train/gt.csv'
    val=official/'data/annotations/task2/val/gt.csv'
    if not (train.is_file() and val.is_file()):raise FileNotFoundError('Missing official labeled train / val annotations')
    if train.resolve()==val.resolve():raise RuntimeError('Train and val are identical file path')
    if a.epochs<1:raise ValueError('epochs must be positive')
    if a.seed is None:a.seed=[42]
    if not (source/'train_abpr.py').exists():raise FileNotFoundError(source/'train_abpr.py')
    checkpoints=[]
    for seed in a.seed:
        prefix=f'abpr_prototype_{a.backbone}_seed{seed}'
        best=a.checkpoint_dir/f'{prefix}_best.pt'
        last=a.checkpoint_dir/f'{prefix}_last.pt'
        print(f'\nTRAIN REAL MODEL: {prefix}; resume_candidate={last if last.exists() else "none"}')
        command=[sys.executable,'-u',str(source/'train_abpr.py'),
            '--repo-root',str(official),
            '--checkpoint-dir',str(a.checkpoint_dir), '--result-dir',str(a.result_dir),
            '--seed',str(seed), '--epochs',str(a.epochs),
            '--batch-size',str(a.batch_size),'--eval-batch-size',str(a.eval_batch_size),
            '--num-workers',str(a.num_workers), '--backbone',a.backbone,
            '--embed-dim',str(a.embed_dim), '--image-height',str(a.image_height),
            '--image-width',str(a.image_width), '--max-val-queries',str(a.max_val_queries),
            '--query-min-known','2','--query-max-known','10',
            '--save-every-steps',str(a.save_every_steps)]
        if a.restart:command.append('--restart')
        for r in a.extra_data_root:command.extend(['--extra-data-root',str(r)])
        run(command,source,a.result_dir/(prefix+'_train.log'))
        if not best.exists():raise RuntimeError(f'Training completed without best checkpoint: {best}')
        print(f'Best checkpoint: {best} ({best.stat().st_size/1048576:.1f} MiB)')
        checkpoints.append(best)
    print(json.dumps({'trained_checkpoints':list(map(str,checkpoints)),'epochs_target':a.epochs},indent=2))

if __name__=='__main__':main()
