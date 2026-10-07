from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable
import copy
import math
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


ATTRIBUTE_NAMES = [
    'Age-Young','Age-Adult','Age-Old','Gender-Female',
    'Hair-Length-Short','Hair-Length-Long','Hair-Length-Bald',
    'UpperBody-Length-Short',
    'UpperBody-Color-Black','UpperBody-Color-Blue','UpperBody-Color-Brown',
    'UpperBody-Color-Green','UpperBody-Color-Grey','UpperBody-Color-Orange',
    'UpperBody-Color-Pink','UpperBody-Color-Purple','UpperBody-Color-Red',
    'UpperBody-Color-White','UpperBody-Color-Yellow','UpperBody-Color-Other',
    'LowerBody-Length-Short',
    'LowerBody-Color-Black','LowerBody-Color-Blue','LowerBody-Color-Brown',
    'LowerBody-Color-Green','LowerBody-Color-Grey','LowerBody-Color-Orange',
    'LowerBody-Color-Pink','LowerBody-Color-Purple','LowerBody-Color-Red',
    'LowerBody-Color-White','LowerBody-Color-Yellow','LowerBody-Color-Other',
    'LowerBody-Type-Trousers&Shorts','LowerBody-Type-Skirt&Dress',
    'Accessory-Backpack','Accessory-Bag','Accessory-Glasses-Normal',
    'Accessory-Glasses-Sun','Accessory-Hat',
]
NUM_ATTRIBUTES = len(ATTRIBUTE_NAMES)
DOMAIN_NAMES = ['Market1501', 'PA100k', 'PETA']


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_domain_id(path: str) -> int:
    s = str(path).replace('\\', '/').lower()
    if 'market1501' in s or 'market-1501' in s or '/market/' in s:
        return 0
    if 'pa100k' in s or 'pa-100k' in s:
        return 1
    if 'peta' in s:
        return 2
    return -1


def detect_image_column(df: pd.DataFrame) -> str:
    for candidate in ('# image', 'image', 'image_path', 'path', 'filename', 'img', 'file'):
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        sample = df[col].astype(str).head(100).str.lower()
        if sample.str.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')).any():
            return col
    raise RuntimeError('Could not detect an image-path column.')


def _is_labeled_csv(path: Path) -> bool:
    try:
        cols = set(pd.read_csv(path, nrows=2).columns)
    except Exception:
        return False
    return all(name in cols for name in ATTRIBUTE_NAMES)


def discover_labeled_csv(repo_root: str | Path, split: str = 'train') -> Path:
    """Find a labeled UPAR CSV without assuming one exact 2027 directory layout."""
    repo_root = Path(repo_root)
    data = repo_root / 'data'
    split = split.lower()
    prioritized = []
    if split == 'train':
        prioritized = [
            data / 'annotations' / 'task2' / 'train' / 'gt.csv',
            data / 'annotations' / 'task1' / 'train' / 'gt.csv',
            data / 'phase1' / 'train' / 'train.csv',
            data / 'phase1' / 'annotations' / 'train.csv',
        ]
    elif split == 'val':
        prioritized = [
            data / 'annotations' / 'task2' / 'val' / 'gt.csv',
            data / 'annotations' / 'task1' / 'val' / 'gt.csv',
            data / 'phase1' / 'val' / 'val.csv',
            data / 'phase1' / 'annotations' / 'val.csv',
        ]
    for p in prioritized:
        if p.exists() and _is_labeled_csv(p):
            return p
    if data.exists():
        tokens = ('train',) if split == 'train' else ('val', 'valid', 'validation')
        candidates = []
        for p in data.rglob('*.csv'):
            low = p.as_posix().lower()
            if any(t in low for t in tokens) and _is_labeled_csv(p):
                candidates.append(p)
        if candidates:
            candidates.sort(key=lambda p: (len(p.parts), len(str(p))))
            return candidates[0]
    raise FileNotFoundError(f'Could not find a labeled {split} CSV under {repo_root}.')


def find_task2_files(repo_root: str | Path) -> tuple[Optional[Path], Optional[Path]]:
    """Locate Track-2 public gallery/query CSVs across UPAR layouts.

    The 2027 repository may expose ``annotations/task2/val/queries.csv`` and
    use the sibling labeled ``gt.csv`` as the public gallery table rather than
    a separately named ``val_imgs.csv``.  Older starter kits use
    ``val_task2/val_imgs.csv`` + ``val_queries.csv``.  Support both.
    """
    repo_root = Path(repo_root)
    data = repo_root / 'data'
    gallery_names = {
        'val_imgs.csv', 'gallery.csv', 'val_gallery.csv', 'images.csv',
        'imgs.csv', 'val_images.csv',
    }
    query_names = {'val_queries.csv', 'queries.csv', 'query.csv'}
    gallery = query = None

    if data.exists():
        csvs = list(data.rglob('*.csv'))
        for p in csvs:
            low = p.name.lower()
            whole = p.as_posix().lower()
            if 'task2' not in whole and 'retriev' not in whole and 'val_task2' not in whole:
                continue
            if gallery is None and low in gallery_names:
                gallery = p
            if query is None and low in query_names:
                query = p

        # 2027 fallback: queries.csv and gt.csv are siblings.  gt.csv has an
        # image column plus the 40 labels and is therefore a valid gallery list.
        if query is not None and gallery is None:
            for name in ('gt.csv', 'val.csv'):
                candidate = query.parent / name
                if candidate.exists():
                    try:
                        df = pd.read_csv(candidate, nrows=2)
                        detect_image_column(df)
                        gallery = candidate
                        break
                    except Exception:
                        pass

        # Last targeted fallback for the current annotations tree.
        if gallery is None:
            for candidate in (
                data / 'annotations' / 'task2' / 'val' / 'gt.csv',
                data / 'phase1' / 'val_task2' / 'val_imgs.csv',
            ):
                if candidate.exists():
                    gallery = candidate
                    break
        if query is None:
            for candidate in (
                data / 'annotations' / 'task2' / 'val' / 'queries.csv',
                data / 'phase1' / 'val_task2' / 'val_queries.csv',
            ):
                if candidate.exists():
                    query = candidate
                    break

    return gallery, query


class ImageResolver:
    """Resolve UPAR image paths across the official repo and optional extra roots.

    The organizer annotations and the user data-preparation script may live in
    different checkouts in Colab.  This resolver therefore searches all known
    roots and only falls back to a basename index as a last resort.
    """
    def __init__(self, data_root: str | Path, repo_root: str | Path, extra_roots=None):
        self.data_root = Path(data_root)
        self.repo_root = Path(repo_root)
        # Search the current 2027 layout plus legacy UPAR/phase1 layouts.
        roots = [
            self.data_root,
            self.repo_root,
            self.repo_root / 'data',
            self.data_root / 'phase1',
            self.repo_root / 'data' / 'phase1',
            self.data_root / 'datasets',
            self.repo_root / 'datasets',
        ]
        env_roots = [x for x in os.environ.get('UPAR_EXTRA_ROOTS', '').split(os.pathsep) if x]
        for x in (extra_roots or []):
            roots.append(Path(x))
        for x in env_roots:
            roots.append(Path(x))
        # Common Colab layout used by the user's previous notebooks.
        for x in ('/content/PAR_UPAR_ABPR', '/content/PAR_UPAR_ABPR/data', '/content/PAR_UPAR', '/content/PAR_UPAR/data'):
            roots.append(Path(x))
        seen=set(); self.roots=[]
        for r in roots:
            key=str(r)
            if key not in seen:
                seen.add(key); self.roots.append(r)
        self._index = None

    @staticmethod
    def _clean(value) -> Path:
        raw=str(value).strip().strip('\"').strip("'").replace('\\','/')
        return Path(raw)

    def _direct_candidates(self, raw: Path):
        """Generate likely on-disk locations without building a global index.

        UPAR annotations use canonical names such as ``Market1501/...`` while
        the original archives are sometimes extracted as ``Market-1501`` or
        ``Market-1501-v15.09.15``.  The same issue occurs for PA100K.  Resolve
        these aliases explicitly before falling back to the basename index.
        """
        yield raw
        parts = list(raw.parts)
        low = [x.lower() for x in parts]

        alias_groups = {
            'market1501': [
                'Market1501', 'market1501', 'Market-1501',
                'Market-1501-v15.09.15', 'market_1501',
            ],
            'pa100k': ['PA100k', 'PA100K', 'PA-100K', 'pa100k'],
            'peta': ['PETA', 'peta'],
        }
        token_alias = {
            'market1501': 'market1501',
            'market-1501': 'market1501',
            'market_1501': 'market1501',
            'pa100k': 'pa100k',
            'pa-100k': 'pa100k',
            'peta': 'peta',
        }

        for root in self.roots:
            yield root / raw

            # CSVs sometimes store paths beginning with data/ or phase1/.
            if parts and parts[0].lower() == 'data':
                yield root / Path(*parts[1:])
            if parts and parts[0].lower() == 'phase1':
                yield root / Path(*parts[1:])

            # Find the dataset token in the annotation path and try the common
            # extraction-directory aliases used by the official/legacy kits.
            hit = None
            for i, token in enumerate(low):
                key = token_alias.get(token)
                if key is not None:
                    hit = (i, key)
                    break
            if hit is not None:
                i, key = hit
                tail = Path(*parts[i+1:]) if i + 1 < len(parts) else Path()
                canonical_tail = Path(*parts[i:])
                yield root / canonical_tail
                for alias in alias_groups[key]:
                    yield root / alias / tail
                    # Some archives contain one extra wrapper folder.
                    if key == 'market1501':
                        yield root / alias / 'Market-1501-v15.09.15' / tail
                    elif key == 'pa100k':
                        yield root / alias / 'release_data' / tail

    def _build_index(self):
        if self._index is not None:
            return
        self._index = {}
        for root in self.roots:
            if not root.exists():
                continue
            for p in root.rglob('*'):
                if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:
                    self._index.setdefault(p.name, p)

    def resolve(self, value) -> Path:
        raw = self._clean(value)
        for p in self._direct_candidates(raw):
            if p.exists() and p.is_file():
                return p
        self._build_index()
        match = self._index.get(raw.name)
        if match is None:
            searched=', '.join(str(x) for x in self.roots)
            raise FileNotFoundError(f'Cannot resolve image: {value}. Searched roots: {searched}')
        return match


def build_train_transform(height: int = 256, width: int = 128):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            (height, width), scale=(0.72, 1.0), ratio=(0.38, 0.58),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.25, hue=0.06)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.08),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.15),
        transforms.RandomAutocontrast(p=0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.18), ratio=(0.3,3.3), value='random'),
    ])


def build_eval_transform(height: int = 256, width: int = 128):
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])


class UPARAttributeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, resolver: ImageResolver, transform):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_col = detect_image_column(self.df)
        self.paths = self.df[self.image_col].astype(str).tolist()
        self.targets = self.df[ATTRIBUTE_NAMES].to_numpy(dtype=np.float32)
        self.domains = np.asarray([infer_domain_id(p) for p in self.paths], dtype=np.int64)
        self.resolver = resolver
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        path = self.resolver.resolve(self.paths[index])
        with Image.open(path) as im:
            image = self.transform(im.convert('RGB'))
        y = self.targets[index].copy()
        valid = ((y == 0) | (y == 1)).astype(np.float32)
        y_clean = np.where(valid > 0, y, 0).astype(np.float32)
        query = np.where(valid > 0, y, -1).astype(np.float32)
        return {
            'image': image,
            'target': torch.from_numpy(y_clean),
            'valid': torch.from_numpy(valid),
            'query': torch.from_numpy(query),
            'domain': int(self.domains[index]),
            'path': self.paths[index],
            'index': int(index),
        }


def split_train_val(df: pd.DataFrame, val_fraction: float = 0.12, seed: int = 42):
    """Domain-aware holdout when the starter has no labeled validation CSV."""
    image_col = detect_image_column(df)
    domains = np.asarray([infer_domain_id(x) for x in df[image_col].astype(str)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    unique = sorted(set(domains.tolist()))
    for d in unique:
        idx = np.where(domains == d)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) > 4 else max(1, len(idx)//4)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def make_domain_balanced_sampler(domains: np.ndarray, seed: int = 42):
    domains = np.asarray(domains, dtype=np.int64)
    weights = np.ones(len(domains), dtype=np.float64)
    known = domains >= 0
    for d in sorted(set(domains[known].tolist())):
        count = max(int((domains == d).sum()), 1)
        weights[domains == d] = 1.0 / count
    if (~known).any():
        weights[~known] = 1.0 / max(int((~known).sum()), 1)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(domains), True, generator=g)


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = float(p), float(alpha), float(eps)

    def forward(self, x):
        if not self.training or x.ndim != 4 or x.size(0) < 2:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        # AMP-safe: compute feature statistics in FP32.
        original_dtype = x.dtype
        x32 = x.float()
        mu = x32.mean((2,3), keepdim=True)
        var = x32.var((2,3), keepdim=True, unbiased=False)
        sig = torch.sqrt(var.clamp_min(0.0) + self.eps)
        mu_det, sig_det = mu.detach(), sig.detach()
        x_norm = (x32 - mu_det) / sig_det.clamp_min(self.eps)
        perm = torch.randperm(x32.size(0), device=x32.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x32.size(0),1,1,1)).to(x32.device, torch.float32)
        out = x_norm * (lam*sig_det + (1-lam)*sig_det[perm]) + (lam*mu_det + (1-lam)*mu_det[perm])
        if not torch.isfinite(out).all():
            raise RuntimeError('MixStyle generated NaN/Inf.')
        return out.to(original_dtype)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradientReverse.apply(x, lambd)


class QueryEncoder(nn.Module):
    """Encodes a 40-state attribute query into the same space as image embeddings."""
    def __init__(self, embed_dim: int = 192, negative_token_scale: float = 0.20):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.negative_token_scale = float(negative_token_scale)
        self.pos_basis = nn.Parameter(torch.randn(NUM_ATTRIBUTES, embed_dim) * 0.02)
        self.neg_basis = nn.Parameter(torch.randn(NUM_ATTRIBUTES, embed_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, query):
        q = query.float()
        valid = q >= 0
        pos = (q > 0.5) & valid
        neg = (q <= 0.5) & valid
        tokens = pos.unsqueeze(-1) * self.pos_basis.unsqueeze(0)
        tokens = tokens + self.negative_token_scale * neg.unsqueeze(-1) * self.neg_basis.unsqueeze(0)
        count = (pos.float() + self.negative_token_scale * neg.float()).sum(1, keepdim=True).clamp(min=1.0)
        z = tokens.sum(1) / count.sqrt()
        z = z + self.mlp(z)
        return F.normalize(z, dim=-1)


class ABPRNet(nn.Module):
    """
    Hybrid PAR + retrieval model.

    The classification head keeps the strong PAR signal from the user's existing
    PAR_UPAR work. The projection/query heads optimize the actual Track-2 task.
    """
    def __init__(self, backbone: str = 'efficientnet_b0', embed_dim: int = 192, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.embed_dim = int(embed_dim)
        if backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            try:
                base = efficientnet_b0(weights=weights)
            except Exception:
                base = efficientnet_b0(weights=None)
            self.features = base.features
            self.avgpool = base.avgpool
            feat_dim = base.classifier[1].in_features
            self.mix_positions = {2: self._make_mix(0.45), 4: self._make_mix(0.35)}
        elif backbone == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            try:
                base = convnext_tiny(weights=weights)
            except Exception:
                base = convnext_tiny(weights=None)
            self.features = base.features
            self.avgpool = base.avgpool
            feat_dim = base.classifier[2].in_features
            self.mix_positions = {2: self._make_mix(0.35), 4: self._make_mix(0.25)}
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.dropout = nn.Dropout(0.60)
        self.attr_head = nn.Linear(feat_dim, NUM_ATTRIBUTES)
        self.image_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(embed_dim, embed_dim),
        )
        self.query_encoder = QueryEncoder(embed_dim=embed_dim, negative_token_scale=0.20)
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.20), nn.Linear(256, len(DOMAIN_NAMES))
        )

    @staticmethod
    def _make_mix(p):
        return MixStyle(p=p, alpha=0.1)

    def extract_features(self, x):
        for idx, block in enumerate(self.features):
            x = block(x)
            key = str(idx)
            # ModuleDict keeps MixStyle registered in state/model mode.
            if hasattr(self, 'mixstyles') and key in self.mixstyles:
                x = self.mixstyles[key](x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def _ensure_mixstyles(self):
        # Backward-compatible lazy registration for checkpoints made from this source.
        if not hasattr(self, 'mixstyles'):
            self.mixstyles = nn.ModuleDict({str(k): v for k, v in self.mix_positions.items()})

    def train(self, mode: bool = True):
        self._ensure_mixstyles()
        return super().train(mode)

    def eval(self):
        self._ensure_mixstyles()
        return super().eval()

    def forward(self, x, query=None, grl_lambda: float = 0.0, return_domain: bool = False):
        self._ensure_mixstyles()
        feat = self.extract_features(x)
        attr_logits = self.attr_head(self.dropout(feat))
        image_emb = F.normalize(self.image_proj(feat), dim=-1)
        query_emb = self.query_encoder(query) if query is not None else None
        if not return_domain:
            return attr_logits, image_emb, query_emb
        domain_logits = self.domain_head(grad_reverse(feat, grl_lambda))
        return attr_logits, image_emb, query_emb, domain_logits


class MaskedAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = map(float, (gamma_neg, gamma_pos, clip, eps))

    def forward(self, logits, targets, valid):
        # AMP-safe: probability/log arithmetic stays in FP32.
        logits = logits.float(); targets = targets.float(); valid = valid.float()
        p = torch.sigmoid(logits).clamp(self.eps, 1.0-self.eps)
        pn = 1.0 - p
        if self.clip > 0:
            pn = (pn + self.clip).clamp(max=1.0)
        pn = pn.clamp(self.eps, 1.0)
        loss = targets * torch.log(p) + (1-targets) * torch.log(pn)
        pt = p * targets + pn * (1-targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1-targets)
        loss = -loss * torch.pow((1-pt).clamp_min(0.0), gamma) * valid
        out = loss.sum() / valid.sum().clamp(min=1.0)
        if not torch.isfinite(out):
            raise RuntimeError('MaskedAsymmetricLoss generated NaN/Inf.')
        return out


class MaskedBalancedBCELoss(nn.Module):
    """Weighted BCE + label smoothing for calibrated cross-domain PAR.

    The UPAR challenge baseline family is classification-centric.  For ABPR we
    want useful probabilities rather than only hard labels, so this loss keeps
    BCE geometry while balancing rare attributes. Unknown labels are masked.
    """
    def __init__(self, priors, label_smoothing: float = 0.08, max_weight: float = 4.0):
        super().__init__()
        priors = torch.as_tensor(priors, dtype=torch.float32).clamp(0.01, 0.99)
        # Equalize positive/negative contributions per attribute, then normalize.
        w_pos = 0.5 / priors
        w_neg = 0.5 / (1.0 - priors)
        w_pos = w_pos.clamp(0.25, float(max_weight))
        w_neg = w_neg.clamp(0.25, float(max_weight))
        norm = (priors * w_pos + (1.0-priors) * w_neg).clamp_min(1e-6)
        self.register_buffer('w_pos', w_pos / norm)
        self.register_buffer('w_neg', w_neg / norm)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets, valid):
        logits = logits.float(); targets = targets.float(); valid = valid.float()
        smooth = targets * (1.0-self.label_smoothing) + 0.5*self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, smooth, reduction='none')
        weight = targets*self.w_pos[None,:] + (1.0-targets)*self.w_neg[None,:]
        loss = bce * weight * valid
        out = loss.sum() / valid.sum().clamp(min=1.0)
        if not torch.isfinite(out):
            raise RuntimeError('MaskedBalancedBCELoss generated NaN/Inf.')
        return out


def domain_loss(domain_logits, domains):
    valid = domains >= 0
    if valid.sum() == 0:
        return domain_logits.sum() * 0.0
    return F.cross_entropy(domain_logits[valid], domains[valid].long())


def alignment_loss(image_emb, query_emb):
    return (1.0 - (image_emb * query_emb).sum(-1)).mean()


def multi_positive_contrastive_loss(image_emb, query_emb, query_states, temperature: float = 0.08):
    """InfoNCE with exact-query duplicates treated as additional positives."""
    logits = image_emb @ query_emb.t() / float(temperature)
    q = query_states.detach()
    same = (q[:, None, :] == q[None, :, :]).all(dim=-1)
    eye = torch.eye(q.size(0), device=q.device, dtype=torch.bool)
    positives = same | eye

    def one_direction(scores, pos_mask):
        all_lse = torch.logsumexp(scores, dim=1)
        pos_scores = scores.masked_fill(~pos_mask, float('-inf'))
        pos_lse = torch.logsumexp(pos_scores, dim=1)
        return (all_lse - pos_lse).mean()

    return 0.5 * (one_direction(logits, positives) + one_direction(logits.t(), positives.t()))


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            src = model_state[k].detach()
            if torch.is_floating_point(v):
                v.mul_(self.decay).add_(src, alpha=1.0-self.decay)
            else:
                v.copy_(src)


@dataclass
class TrainConfig:
    image_height: int = 256
    image_width: int = 128
    backbone: str = 'convnext_tiny'
    embed_dim: int = 192
    batch_size: int = 48
    eval_batch_size: int = 96
    epochs: int = 14
    lr: float = 1e-4
    weight_decay: float = 5e-4
    attr_loss_weight: float = 1.0
    align_loss_weight: float = 0.10
    contrast_loss_weight: float = 0.08
    domain_loss_weight: float = 0.03
    grl_lambda: float = 0.15
    temperature: float = 0.08
    ema_decay: float = 0.999
    num_workers: int = 2


def estimate_attribute_priors(targets: np.ndarray, valid: np.ndarray) -> np.ndarray:
    targets, valid = np.asarray(targets), np.asarray(valid)
    pos = (targets * valid).sum(0)
    den = valid.sum(0)
    return ((pos + 1.0) / (den + 2.0)).astype(np.float32)


def build_queries_from_labels(labels: np.ndarray, max_queries: int = 600, seed: int = 42):
    labels = np.asarray(labels, dtype=np.int16)
    uniq, inverse = np.unique(labels, axis=0, return_inverse=True)
    if len(uniq) > max_queries:
        rng = np.random.default_rng(seed)
        # Prefer recurring queries, then add random singletons.
        counts = np.bincount(inverse, minlength=len(uniq))
        recurring = np.where(counts >= 2)[0]
        singleton = np.where(counts < 2)[0]
        chosen = recurring.tolist()
        if len(chosen) > max_queries:
            chosen = rng.choice(chosen, size=max_queries, replace=False).tolist()
        else:
            need = max_queries - len(chosen)
            if need > 0 and len(singleton) > 0:
                chosen += rng.choice(singleton, size=min(need, len(singleton)), replace=False).tolist()
        uniq = uniq[np.asarray(chosen, dtype=np.int64)]
    return uniq.astype(np.float32)


def semantic_match_scores(probs: torch.Tensor, queries: torch.Tensor, priors: torch.Tensor,
                          negative_weight: float = 0.15, eps: float = 1e-6):
    """
    Bernoulli attribute agreement. Positive query attributes receive rarity-aware
    weights; negatives are deliberately weaker so the many zero attributes do not
    swamp the discriminative positives.
    """
    probs = probs.clamp(eps, 1-eps)
    queries = queries.float()
    valid = queries >= 0
    pos = (queries > 0.5) & valid
    neg = (queries <= 0.5) & valid
    priors = priors.clamp(1e-4, 1-1e-4)
    w_pos = (-torch.log(priors)).clamp(0.25, 4.0)
    w_neg = (-torch.log(1-priors)).clamp(0.05, 1.5) * float(negative_weight)
    # [Q,G,A]
    lp = torch.log(probs)[None, :, :]
    ln = torch.log(1-probs)[None, :, :]
    weight = pos[:,None,:].float()*w_pos[None,None,:] + neg[:,None,:].float()*w_neg[None,None,:]
    ll = pos[:,None,:].float()*lp + neg[:,None,:].float()*ln
    return (ll*weight).sum(-1) / weight.sum(-1).clamp(min=1e-6)


def _row_zscore(x: torch.Tensor):
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp(min=1e-6)


def hybrid_retrieval_scores(gallery_probs: torch.Tensor, gallery_emb: torch.Tensor,
                            queries: torch.Tensor, query_emb: torch.Tensor,
                            priors: torch.Tensor, blend: float = 0.65,
                            negative_weight: float = 0.15):
    sem = semantic_match_scores(gallery_probs, queries, priors, negative_weight=negative_weight)
    emb = query_emb @ gallery_emb.t()
    sem_z, emb_z = _row_zscore(sem), _row_zscore(emb)
    return float(blend) * sem_z + (1.0-float(blend)) * emb_z


def relevance_matrix(queries: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    queries = np.asarray(queries)
    gallery_labels = np.asarray(gallery_labels)
    out = np.zeros((len(queries), len(gallery_labels)), dtype=bool)
    for i, q in enumerate(queries):
        valid = q >= 0
        if valid.sum() == 0:
            continue
        out[i] = (gallery_labels[:, valid] == q[valid]).all(axis=1)
    return out


def retrieval_metrics(scores: np.ndarray, relevance: np.ndarray) -> dict:
    scores = np.asarray(scores)
    relevance = np.asarray(relevance, dtype=bool)
    aps, r1s = [], []
    for i in range(scores.shape[0]):
        rel = relevance[i]
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        order = np.argsort(-scores[i], kind='stable')
        ranked = rel[order]
        r1s.append(float(ranked[0]))
        cumsum = np.cumsum(ranked)
        ranks = np.arange(1, len(ranked)+1)
        ap = float(((cumsum / ranks) * ranked).sum() / n_rel)
        aps.append(ap)
    mAP = float(np.mean(aps)) if aps else 0.0
    r1 = float(np.mean(r1s)) if r1s else 0.0
    h = float(2*mAP*r1/max(mAP+r1, 1e-12))
    return {'mAP': mAP, 'Rank1': r1, 'Hmean_mAP_R1': h, 'num_queries': len(aps)}


# ---------------------------------------------------------------------------
# Track-2 / mADM-aligned retrieval helpers
# ---------------------------------------------------------------------------

def calibrate_probabilities(probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply scalar temperature to probabilities through the logit domain."""
    p = probs.float().clamp(1e-6, 1.0 - 1e-6)
    t = max(float(temperature), 1e-4)
    if abs(t - 1.0) < 1e-8:
        return p
    logits = torch.log(p / (1.0 - p)) / t
    return torch.sigmoid(logits)


def attribute_reliability_weights(
    probs: torch.Tensor | np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray | None = None,
    power: float = 1.0,
) -> np.ndarray:
    """
    Estimate per-attribute reliability from validation balanced accuracy.

    The returned weights are normalized to mean 1.  power=0 gives uniform
    weights.  This is an inference calibration aid; the challenge metric still
    treats the semantic attributes themselves equally.
    """
    p = probs.detach().cpu().numpy() if torch.is_tensor(probs) else np.asarray(probs)
    y = np.asarray(labels)
    if valid is None:
        valid = (y >= 0).astype(np.float32)
    valid = np.asarray(valid).astype(bool)
    pred = p >= 0.5

    bal = np.full(y.shape[1], 0.5, dtype=np.float64)
    for a in range(y.shape[1]):
        va = valid[:, a]
        if not va.any():
            continue
        ya = y[va, a] > 0.5
        pa = pred[va, a]
        pos = ya
        neg = ~ya
        tpr = float((pa[pos] == ya[pos]).mean()) if pos.any() else 0.5
        tnr = float((pa[neg] == ya[neg]).mean()) if neg.any() else 0.5
        bal[a] = 0.5 * (tpr + tnr)

    # Map chance-level reliability to a small positive value and normalize.
    signal = np.clip((bal - 0.5) / 0.5, 0.02, 1.0)
    w = np.power(signal, float(power))
    w = w / max(float(w.mean()), 1e-8)
    return w.astype(np.float32)


def expected_hamming_distances(
    gallery_probs: torch.Tensor,
    queries: torch.Tensor | np.ndarray,
    attribute_weights: torch.Tensor | np.ndarray | None = None,
    temperature: float = 1.0,
    gallery_emb: torch.Tensor | None = None,
    query_emb: torch.Tensor | None = None,
    embedding_mix: float = 0.0,
) -> torch.Tensor:
    """
    Expected weighted Hamming/L1 distance between binary attribute queries and
    gallery attribute probabilities.

    This matches the organizer's sample baseline geometry:
        sum |q - p|
    but supports per-attribute reliability weights and an optional small
    image/query embedding term.  Smaller values are better.

    The implementation avoids allocating [Q,G,A], which would be several GB for
    the official 893 x 29248 x 40 evaluation tensor.
    """
    p = calibrate_probabilities(gallery_probs, temperature=temperature).cpu()
    q = torch.as_tensor(queries, dtype=torch.float32).cpu()
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != q.shape[1]:
        raise ValueError(f'Expected p=[G,A], q=[Q,A], got {tuple(p.shape)}, {tuple(q.shape)}')

    if attribute_weights is None:
        w = torch.ones(p.shape[1], dtype=torch.float32)
    else:
        w = torch.as_tensor(attribute_weights, dtype=torch.float32).flatten().cpu()
        if len(w) != p.shape[1]:
            raise ValueError(f'attribute_weights has {len(w)} values, expected {p.shape[1]}')
        w = w.clamp(min=1e-6)

    valid = q >= 0
    q01 = torch.where(valid, q.clamp(0, 1), torch.zeros_like(q))
    wq = valid.float() * w[None, :]

    # sum_a w_a * [p_a + q_a * (1 - 2 p_a)] over valid query attributes.
    # First term depends on each query when missing attributes are allowed.
    d_attr = wq @ p.t() + (q01 * wq) @ (1.0 - 2.0 * p).t()
    denom = wq.sum(1, keepdim=True).clamp(min=1e-6)
    d_attr = d_attr / denom

    mix = float(embedding_mix)
    if mix > 0.0:
        if gallery_emb is None or query_emb is None:
            raise ValueError('gallery_emb and query_emb are required when embedding_mix > 0')
        gz = F.normalize(gallery_emb.float().cpu(), dim=-1)
        qz = F.normalize(query_emb.float().cpu(), dim=-1)
        # cosine distance [0, 2], rescaled approximately to [0, 1]
        d_emb = (1.0 - qz @ gz.t()) * 0.5
        d_attr = (1.0 - mix) * d_attr + mix * d_emb
    return d_attr


def madm_metric(
    distances: np.ndarray | torch.Tensor,
    queries: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict:
    """
    Mean Average Degree of Match (mADM), following the UPAR definition.

    For each query:
      DoM@k = fraction of matching attributes at rank k
      DoM_norm = max(0, (DoM - mean_gallery_DoM) / (1 - mean_gallery_DoM))
      Prec_DoM@k = cumulative mean of DoM_norm
    The final per-query ADM is then calculated analogously to AP, replacing
    binary precision with Prec_DoM while retaining the exact-match relevance
    indicator at the AP accumulation positions.

    Distances are sorted ascending (smaller = better).
    """
    d = distances.detach().cpu().numpy() if torch.is_tensor(distances) else np.asarray(distances)
    q_all = np.asarray(queries)
    y_all = np.asarray(gallery_labels)
    if d.shape != (len(q_all), len(y_all)):
        raise ValueError(f'distances shape {d.shape} != {(len(q_all), len(y_all))}')

    adms = []
    for qi, q in enumerate(q_all):
        q_valid = q >= 0
        if not q_valid.any():
            continue

        y = y_all[:, q_valid]
        qv = q[q_valid]
        known = y >= 0
        n_known = known.sum(1)
        # Number of matching known attributes for every gallery item.
        matches = ((y == qv[None, :]) & known).sum(1)
        dom = np.divide(
            matches,
            np.maximum(n_known, 1),
            out=np.zeros_like(matches, dtype=np.float64),
            where=n_known > 0,
        )

        # Exact positives require every queried attribute to be known and match.
        rel = (n_known == int(q_valid.sum())) & (matches == int(q_valid.sum()))
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue

        mean_dom = float(dom.mean())
        denom = max(1.0 - mean_dom, 1e-12)
        dom_norm = np.maximum(0.0, (dom - mean_dom) / denom)

        order = np.argsort(d[qi], kind='stable')
        ranked_rel = rel[order]
        ranked_dom = dom_norm[order]
        prec_dom = np.cumsum(ranked_dom) / np.arange(1, len(order) + 1)
        adm = float((prec_dom * ranked_rel).sum() / n_rel)
        adms.append(adm)

    return {
        'mADM': float(np.mean(adms)) if adms else 0.0,
        'num_queries_mADM': int(len(adms)),
    }


def madm_aligned_metrics(
    distances: np.ndarray | torch.Tensor,
    queries: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict:
    """Convenience diagnostic: mADM + legacy mAP/Rank-1 for the same ranking."""
    d = distances.detach().cpu().numpy() if torch.is_tensor(distances) else np.asarray(distances)
    rel = relevance_matrix(queries, gallery_labels)
    legacy = retrieval_metrics(-d, rel)  # retrieval_metrics expects higher score = better
    out = dict(legacy)
    out.update(madm_metric(d, queries, gallery_labels))
    return out


# ---------------------------------------------------------------------------
# Track-2 calibration + error-weighted distance helpers
# ---------------------------------------------------------------------------

def fit_attribute_temperatures(
    probs: torch.Tensor | np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray | None = None,
    grid=(0.60, 0.75, 0.90, 1.00, 1.15, 1.35, 1.60, 2.00),
) -> np.ndarray:
    """Fit one monotone temperature per attribute by validation Brier loss."""
    p = probs.detach().cpu().numpy() if torch.is_tensor(probs) else np.asarray(probs)
    y = np.asarray(labels, dtype=np.float32)
    if valid is None:
        valid = (y >= 0)
    valid = np.asarray(valid).astype(bool)
    out = np.ones(p.shape[1], dtype=np.float32)
    eps = 1e-6
    logit = np.log(np.clip(p,eps,1-eps) / np.clip(1-p,eps,1-eps))
    for a in range(p.shape[1]):
        m = valid[:,a]
        if m.sum() < 16 or np.unique(y[m,a]).size < 2:
            continue
        best=(float('inf'),1.0)
        for t in grid:
            pc=1.0/(1.0+np.exp(-np.clip(logit[m,a]/float(t),-30,30)))
            brier=float(np.mean((pc-y[m,a])**2))
            if brier < best[0]: best=(brier,float(t))
        out[a]=best[1]
    return out


def apply_attribute_temperatures(
    probs: torch.Tensor,
    temperatures: torch.Tensor | np.ndarray | None,
) -> torch.Tensor:
    p=probs.float().clamp(1e-6,1-1e-6)
    if temperatures is None:
        return p
    t=torch.as_tensor(temperatures,dtype=torch.float32).flatten().cpu().clamp_min(1e-4)
    if len(t)!=p.shape[1]:
        raise ValueError(f'Expected {p.shape[1]} temperatures, got {len(t)}')
    logits=torch.log(p.cpu()/(1-p.cpu())) / t[None,:]
    return torch.sigmoid(logits)


def attribute_error_weights(
    probs: torch.Tensor | np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray | None = None,
    tau: float = 0.10,
) -> np.ndarray:
    """Error weighting: more reliable attributes get more retrieval weight.

    We estimate per-attribute MSE after calibration and convert -MSE to a
    softmax-like weight. tau controls how aggressively unreliable attributes are
    suppressed. Weights are normalized to mean 1 for stable distance scales.
    """
    p=probs.detach().cpu().numpy() if torch.is_tensor(probs) else np.asarray(probs)
    y=np.asarray(labels,dtype=np.float32)
    if valid is None: valid=(y>=0)
    valid=np.asarray(valid).astype(bool)
    mse=np.full(p.shape[1],0.25,dtype=np.float64)
    for a in range(p.shape[1]):
        m=valid[:,a]
        if m.any(): mse[a]=float(np.mean((p[m,a]-y[m,a])**2))
    tau=max(float(tau),1e-4)
    z=np.exp(-(mse-mse.min())/tau)
    z=np.clip(z,1e-4,None)
    z=z/max(float(z.mean()),1e-8)
    return z.astype(np.float32)


def calibrated_attribute_distances(
    gallery_probs: torch.Tensor,
    queries: torch.Tensor | np.ndarray,
    attribute_weights: torch.Tensor | np.ndarray | None = None,
    attribute_temperatures: torch.Tensor | np.ndarray | None = None,
    distance_kind: str = 'l2',
    gallery_emb: torch.Tensor | None = None,
    query_emb: torch.Tensor | None = None,
    embedding_mix: float = 0.0,
) -> torch.Tensor:
    """Memory-efficient weighted L1/L2 query-to-gallery distance.

    q is binary (or -1 for unknown).  L2 is the squared Euclidean distance;
    the square root is omitted because it preserves ranking.  No [Q,G,A]
    tensor is materialized, so 893 x 29248 x 40 remains practical.
    """
    p=apply_attribute_temperatures(gallery_probs,attribute_temperatures).cpu()
    q=torch.as_tensor(queries,dtype=torch.float32).cpu()
    if p.ndim!=2 or q.ndim!=2 or p.shape[1]!=q.shape[1]:
        raise ValueError(f'Bad p/q shapes: {tuple(p.shape)} / {tuple(q.shape)}')
    if attribute_weights is None:
        w=torch.ones(p.shape[1],dtype=torch.float32)
    else:
        w=torch.as_tensor(attribute_weights,dtype=torch.float32).flatten().cpu().clamp_min(1e-6)
    if len(w)!=p.shape[1]: raise ValueError('attribute_weights size mismatch')
    valid=q>=0
    q01=torch.where(valid,q.clamp(0,1),torch.zeros_like(q))
    wq=valid.float()*w[None,:]
    denom=wq.sum(1,keepdim=True).clamp_min(1e-6)
    kind=str(distance_kind).lower()
    if kind=='l1':
        d=wq@p.t() + (q01*wq)@(1.0-2.0*p).t()
    elif kind=='l2':
        # (q-p)^2 = p^2 + q*(1-2p) for q in {0,1}
        d=wq@(p*p).t() + (q01*wq)@(1.0-2.0*p).t()
    else:
        raise ValueError(f'Unsupported distance_kind={distance_kind}')
    d=d/denom
    mix=float(embedding_mix)
    if mix>0:
        if gallery_emb is None or query_emb is None:
            raise ValueError('Embeddings required when embedding_mix > 0')
        gz=F.normalize(gallery_emb.float().cpu(),dim=-1)
        qz=F.normalize(query_emb.float().cpu(),dim=-1)
        d_emb=(1.0-qz@gz.t())*0.5
        # normalize row-wise to reduce scale mismatch before a small fusion.
        d0=(d-d.mean(1,keepdim=True))/d.std(1,keepdim=True).clamp_min(1e-6)
        de=(d_emb-d_emb.mean(1,keepdim=True))/d_emb.std(1,keepdim=True).clamp_min(1e-6)
        d=(1.0-mix)*d0+mix*de
    return d
