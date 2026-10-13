"""TRAIN-only teacher-constrained 40-attribute residual adapter for ABPR.

Consumes frozen DINO 40x192 attribute tokens and HATA teacher logits;
never needs public VAL labels or semantic IDs for training/inference.
This is a research experiment, not a Codabench runtime/submission.
"""
from __future__ import annotations
import hashlib, json, os, random, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


def digest(*values):
    h = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        else:
            h.update(str(value).encode())
        h.update(b'\x00')
    return h.hexdigest()


def safe_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


class ResidualAdapter(nn.Module):
    """Zero-initialized uncertainty-gated, teacher-constrained local token correction."""
    def __init__(self, dim=192, num_attrs=40, width=48, max_correction=0.8):
        super().__init__()
        self.dim, self.num_attrs = dim, num_attrs
        self.max_correction = float(max_correction)
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Sequential(nn.Linear(dim, width), nn.GELU())
        self.attr_weight = nn.Parameter(torch.zeros(num_attrs, width))
        self.attr_bias = nn.Parameter(torch.zeros(num_attrs))

    def forward(self, token, teacher, return_delta=False):
        if token.ndim != 3 or token.shape[1:] != (self.num_attrs, self.dim):
            raise ValueError(f'Expected [batch,{self.num_attrs},{self.dim}] tokens, got {tuple(token.shape)}')
        if teacher.shape != token.shape[:2]:
            raise ValueError('HATA teacher logits shape does not match frozen DINO tokens')
        hidden = self.projection(self.norm(token))
        raw = (hidden * self.attr_weight[None, :, :]).sum(-1) + self.attr_bias[None, :]
        uncertainty = 1. - (torch.sigmoid(teacher.detach()) * 2. - 1.).abs()
        gate = uncertainty.clamp(min=0.03)
        delta = self.max_correction * torch.tanh(raw) * gate
        logits = teacher.detach() + delta
        return (logits, delta) if return_delta else logits


def balanced_bce(logits, target, posweight):
    known = target >= 0
    y = target.clamp(min=0)
    raw = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    w = torch.where(y > .5, posweight[None, :], 1.)
    return (raw * w * known).sum() / (known * w).sum().clamp(min=1.)


def multi_positive_rank(logits, labels, rng, dense_prob=0.8, temperature=0.20):
    """Query-to-many-positive softmax loss using TRAIN labels only.

    A gallery sample is positive if all query-known AND gallery-known attributes match;
    unknown gallery labels are not treated as positive evidence.
    """
    n, a = labels.shape
    if n <= 1:
        return logits.sum() * 0.
    q = labels.clone()
    dense = rng.random(n) < dense_prob
    for i in range(n):
        if dense[i]:
            continue
        good = torch.where(q[i] >= 0)[0]
        if len(good) > 2:
            take = torch.as_tensor(rng.choice(good.cpu().numpy(), size=min(len(good), int(rng.integers(2, 11))), replace=False),device=q.device)
            drop = torch.ones(a, dtype=torch.bool, device=q.device)
            drop[take] = False
            q[i, drop] = -1
    m = (q >= 0).float()
    target = q.clamp(min=0)
    # Each image is scored against every query in the local TRAIN minibatch.
    nll = (F.softplus(logits)[None, :, :] - target[:, None, :] * logits[None, :, :])
    dist = (nll * m[:, None, :]).sum(-1) / m.sum(-1).clamp(min=1)[:, None]
    known_both = (q[:, None, :] >= 0) & (labels[None, :, :] >= 0)
    mismatch = ((q[:, None, :] != labels[None, :, :]) & known_both).any(-1)
    compatible = ~mismatch
    # Gallery entries missing query labels are not proven positives.
    completely_known = ~(((q[:, None, :] >= 0) & (labels[None, :, :] < 0)).any(-1))
    positive = compatible & completely_known
    # Each image matches its own query, provided it has known attributes.
    valid = positive.any(-1) & mismatch.any(-1) & (m.sum(-1) > 0)
    if not valid.any():
        return logits.sum() * 0.
    scores = -dist / temperature
    logp = F.log_softmax(scores, dim=1)
    positive_logmass = torch.logsumexp(logp.masked_fill(~positive, -1e4), dim=1)
    return -positive_logmass[valid].mean()


def split_train_holdout(domains, seed=8719, holdout_fraction=.16):
    rng = np.random.default_rng(seed)
    train, hold = [], []
    for d in np.unique(domains):
        idx = np.flatnonzero(np.asarray(domains) == d)
        rng.shuffle(idx)
        k = max(1, round(len(idx) * holdout_fraction))
        hold.extend(idx[:k]); train.extend(idx[k:])
    return np.asarray(train,np.int64),np.asarray(hold,np.int64)


def train(tokens, teacher, labels, domains, folder, signature, *, epochs=12, batch_size=96,
          lr=.001, rank_weight=.08, seed=8719, patience=4, max_correction=.8,
          force=False, device=None):
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    bestfile, lastfile = folder/'teacher_residual_best.pt', folder/'teacher_residual_last.pt'
    histfile = folder/'train_history.json'
    tokens = np.asarray(tokens); teacher=np.asarray(teacher); labels=np.asarray(labels); domains=np.asarray(domains)
    n=len(tokens)
    assert tokens.shape == (n,40,192) and teacher.shape == labels.shape == (n,40)
    assert np.isfinite(teacher).all() and np.isin(labels,[-1,0,1]).all()
    assert len(domains)==n and n>20
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    setup = {'signature':signature,'epochs':int(epochs),'batch_size':int(batch_size),'lr':float(lr),
             'rank_weight':float(rank_weight),'seed':int(seed),'max_correction':float(max_correction)}
    signature2=digest(json.dumps(setup,sort_keys=True))
    model=ResidualAdapter(max_correction=max_correction).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.012)
    tr_idx,ho_idx=split_train_holdout(domains,seed=seed)
    ytr=labels[tr_idx]
    pos=((ytr==1).sum(0)+1).astype('float32')
    neg=((ytr==0).sum(0)+1).astype('float32')
    posweight=np.clip(np.sqrt(neg/pos),0.7,3.).astype('float32')
    pw=torch.from_numpy(posweight).to(device)
    start_epoch=0;best_hold=float('inf');bad=0;history=[]
    if lastfile.is_file() and not force:
        ck=torch.load(lastfile,map_location='cpu',weights_only=False)
        if ck['signature']!=signature2:raise RuntimeError('TRAIN resume signature differs; use another output directory or FORCE_NEW=True')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        for state in opt.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(device)
        start_epoch=ck['epoch'];best_hold=float(ck['best_hold']);bad=ck['bad'];history=ck['history']
        print('[RESUME] teacher residual from epoch',start_epoch,'/',epochs,flush=True)
        if (start_epoch>=epochs or bad>=patience) and bestfile.is_file():
            model.load_state_dict(torch.load(bestfile,map_location='cpu',weights_only=False)['model'])
            return model.eval(),history,bestfile
    elif bestfile.is_file() and not force:
        ck=torch.load(bestfile,map_location='cpu',weights_only=False)
        if ck['signature']!=signature2:raise RuntimeError('Existing best checkpoint signature differs')
        # Avoid accidentally using a result with no compatible last state.
        raise RuntimeError('Best exists but LAST not found. Do not overwrite; use FORCE_NEW=True or restore LAST.')

    rng=np.random.default_rng(seed+start_epoch)
    for ep in range(start_epoch,epochs):
        model.train()
        # Domain-balanced reshuffle, not using any public VAL labels.
        domains_here=domains[tr_idx]
        shuffled=np.concatenate([rng.permutation(tr_idx[domains_here==d]) for d in np.unique(domains_here)])
        rng.shuffle(shuffled)
        losses=[]
        for st in tqdm(range(0,len(shuffled),batch_size),desc=f'Residual TRAIN epoch {ep+1}/{epochs}',leave=False):
            ii=shuffled[st:st+batch_size]
            if len(ii)<2:continue
            x=torch.from_numpy(np.array(tokens[ii],dtype='float32',copy=True)).to(device)
            t=torch.from_numpy(np.array(teacher[ii],dtype='float32',copy=True)).to(device)
            y=torch.from_numpy(np.array(labels[ii],dtype='float32',copy=True)).to(device)
            pred,delta=model(x,t,return_delta=True)
            bce=balanced_bce(pred,y,pw)
            rank=multi_positive_rank(pred,y,rng)
            preservation=delta.square().mean()
            loss=bce+rank_weight*rank+0.06*preservation
            if not torch.isfinite(loss):raise RuntimeError('Non-finite TRAIN loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); vl=[]
        with torch.inference_mode():
            for st in range(0,len(ho_idx),batch_size):
                ii=ho_idx[st:st+batch_size]
                x=torch.from_numpy(np.array(tokens[ii],dtype='float32',copy=True)).to(device)
                t=torch.from_numpy(np.array(teacher[ii],dtype='float32',copy=True)).to(device)
                y=torch.from_numpy(np.array(labels[ii],dtype='float32',copy=True)).to(device)
                pred,delta=model(x,t,return_delta=True)
                vl.append((len(ii),float((balanced_bce(pred,y,pw)+0.06*delta.square().mean()).cpu())))
        val_loss=sum(c*v for c,v in vl)/sum(c for c,_ in vl)
        row={'epoch':ep+1,'train_loss':float(np.mean(losses)),'train_holdout_bce_reg':float(val_loss)}
        history.append(row)
        improved=val_loss<best_hold-1e-5
        if improved:
            best_hold=val_loss;bad=0
            torch.save({'signature':signature2,'model':model.state_dict(),'config':setup,'holdout':best_hold},bestfile)
        else:bad+=1
        torch.save({'signature':signature2,'epoch':ep+1,'model':model.state_dict(),
                    'optimizer':opt.state_dict(),'history':history,'best_hold':best_hold,'bad':bad},lastfile)
        safe_json(histfile,{'setup':setup,'signature':signature2,'history':history,'best_holdout':best_hold})
        print('Residual epoch=',ep+1,'train_loss=',round(row['train_loss'],5),
              'TRAIN holdout=',round(val_loss,5),'best=',round(best_hold,5),flush=True)
        if bad>=patience:
            print('[EARLY STOP] no TRAIN holdout improvement',flush=True)
            break
    ck=torch.load(bestfile,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']);model.eval()
    return model,history,bestfile


def load_checkpoint(file, device=None):
    ck=torch.load(file,map_location='cpu',weights_only=False)
    model=ResidualAdapter(max_correction=ck['config']['max_correction'])
    model.load_state_dict(ck['model'])
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device).eval(),ck


def infer(model, tokens, teacher, out_file, signature, batch_size=160):
    """Resume .npy cache only if metadata signature matches image/teacher/model order."""
    out_file=Path(out_file);out_file.parent.mkdir(parents=True,exist_ok=True)
    met=out_file.with_suffix('.json')
    n=len(tokens)
    if np.asarray(tokens).shape!=(n,40,192) or np.asarray(teacher).shape!=(n,40):
        raise ValueError('DINO token/HATA teacher shapes do not align')
    completed=0
    if out_file.is_file() and met.is_file():
        info=json.loads(met.read_text(encoding='utf-8'))
        if info.get('signature')!=signature or info.get('shape')!=[n,40]:
            raise RuntimeError('Stale representation cache! Must not mix image ordering. Use new method output folder.')
        completed=int(info.get('completed',0))
        out=np.load(out_file,mmap_mode='r+',allow_pickle=False)
        if completed==n:
            if np.isfinite(out).all():return out
            raise RuntimeError('Completed cache contains NaN/Inf')
    else:
        if out_file.is_file() or met.is_file():
            raise RuntimeError('Partial cache without matching metadata: refuse unsafe resume')
        out=np.lib.format.open_memmap(out_file,mode='w+',dtype='float32',shape=(n,40))
        safe_json(met,{'signature':signature,'shape':[n,40],'completed':0})
    dev=next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for st in tqdm(range(completed,n,batch_size),desc='Residual encode VAL',unit='batch'):
            end=min(n,st+batch_size)
            x=torch.from_numpy(np.asarray(tokens[st:end],dtype='float32')).to(dev)
            t=torch.from_numpy(np.asarray(teacher[st:end],dtype='float32')).to(dev)
            pred=model(x,t).cpu().numpy().astype('float32')
            if not np.isfinite(pred).all():raise RuntimeError('Invalid non-finite inference output')
            out[st:end]=pred
            if end==n or (end//batch_size)%12==0:
                out.flush();safe_json(met,{'signature':signature,'shape':[n,40],'completed':end})
    out.flush();safe_json(met,{'signature':signature,'shape':[n,40],'completed':n})
    return out


if __name__=='__main__':
    # Synthetic smoke test does not validate actual user data/results.
    rng=np.random.default_rng(17)
    tok=rng.standard_normal((36,40,192),dtype=np.float32)
    base=rng.standard_normal((36,40),dtype=np.float32)
    lab=rng.integers(0,2,(36,40)).astype(np.float32)
    dom=np.repeat(np.arange(3),12)
    m=ResidualAdapter()
    assert torch.equal(m(torch.from_numpy(tok[:3]),torch.from_numpy(base[:3])),torch.from_numpy(base[:3]))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        model,h,best=train(tok,base,lab,dom,d,'smoke',epochs=2,batch_size=12,device='cpu')
        got=infer(model,tok,base,Path(d)/'infer.npy','signature',batch_size=10)
        assert got.shape==(36,40) and np.isfinite(got).all()
    print('[SMOKE PASS]')
