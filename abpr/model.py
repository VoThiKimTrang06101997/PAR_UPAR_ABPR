from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
)

from .core import (
    ATTRIBUTE_NAMES, NUM_ATTRIBUTES, DOMAIN_NAMES,
    ImageResolver, detect_image_column, infer_domain_id,
    MixStyle, grad_reverse, estimate_attribute_priors,
    fit_attribute_temperatures, apply_attribute_temperatures,
    attribute_error_weights,
)
from .prototype import AttributePrototypeHead


# -----------------------------------------------------------------------------
# Challenge-file discovery: explicitly prefer validation over train.
# Earlier generic recursive discovery could pick task2/train before validation.
# This resolver explicitly prioritizes the organizer validation retrieval split.
# -----------------------------------------------------------------------------

def find_task2_split_files(repo_root: str | Path, split: str = 'val') -> tuple[Path, Path]:
    repo_root = Path(repo_root)
    data = repo_root / 'data'
    split = split.lower()

    explicit = []
    if split == 'val':
        explicit = [
            (data / 'annotations' / 'task2' / 'val' / 'gt.csv', data / 'annotations' / 'task2' / 'val' / 'queries.csv'),
            (data / 'phase1' / 'val_task2' / 'val_imgs.csv', data / 'phase1' / 'val_task2' / 'val_queries.csv'),
        ]
    elif split == 'train':
        explicit = [
            (data / 'annotations' / 'task2' / 'train' / 'gt.csv', data / 'annotations' / 'task2' / 'train' / 'queries.csv'),
        ]
    else:
        raise ValueError(f'Unsupported split={split}')

    for g, q in explicit:
        if g.exists() and q.exists():
            return g, q

    # Conservative fallback: require both files from the same directory and
    # prefer paths whose directory name matches the requested split.
    candidates = []
    if data.exists():
        for q in data.rglob('queries.csv'):
            whole = q.as_posix().lower()
            if 'task2' not in whole:
                continue
            g = q.parent / 'gt.csv'
            if not g.exists():
                continue
            score = 0
            if f'/{split}/' in whole or q.parent.name.lower() == split:
                score -= 100
            if split == 'val' and 'train' in whole:
                score += 100
            candidates.append((score, len(q.parts), g, q))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], str(x[2])))
        _, _, g, q = candidates[0]
        return g, q

    raise FileNotFoundError(f'Could not find Track-2 {split} gt.csv + queries.csv under {repo_root}')


def load_query_csv(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    path = Path(path)
    df = pd.read_csv(path)
    if all(c in df.columns for c in ATTRIBUTE_NAMES):
        q = df[ATTRIBUTE_NAMES].apply(pd.to_numeric, errors='coerce').fillna(-1).to_numpy(np.float32)
        return df, q

    excluded = {'id', 'query_id', 'index', '#', 'name', 'query'}
    numeric = []
    for c in df.columns:
        if str(c).strip().lower() in excluded:
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        # Attribute columns are almost entirely in {-1,0,1}.
        vals = s.dropna().to_numpy()
        good = np.isin(vals, [-1, 0, 1]).mean() if len(vals) else 0.0
        if s.notna().mean() >= 0.8 and good >= 0.95:
            numeric.append(c)
    if len(numeric) < NUM_ATTRIBUTES:
        raise RuntimeError(f'Cannot identify {NUM_ATTRIBUTES} attribute columns in {path}; columns={list(df.columns)}')
    q = df[numeric[:NUM_ATTRIBUTES]].apply(pd.to_numeric, errors='coerce').fillna(-1).to_numpy(np.float32)
    return df, q


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def build_train_transform(height: int = 320, width: int = 160):
    # Preserve the full pedestrian body; large random crops can remove fine attributes.
    # fine clothing/accessory attributes are easily lost by large random crops.
    pad_h = max(8, int(round(height * 0.05)))
    pad_w = max(4, int(round(width * 0.05)))
    return transforms.Compose([
        transforms.Resize((height + pad_h, width + pad_w), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop((height, width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.18, hue=0.04)
        ], p=0.7),
        transforms.RandomGrayscale(p=0.04),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.18, scale=(0.02, 0.10), ratio=(0.4, 2.5), value='random'),
    ])


def build_eval_transform(height: int = 320, width: int = 160):
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class UPARDataset(Dataset):
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


def sample_sparse_queries(
    full_queries: torch.Tensor,
    min_known: int = 2,
    max_known: int = 10,
) -> torch.Tensor:
    """Randomly hide known labels so query training matches sparse ABPR queries.

    Earlier training aligned image embeddings with almost-complete 40-attribute label
    vectors. Track-2 queries are sparse. This augmentation trains the same query
    encoder on the input geometry it sees at inference.
    """
    q = full_queries.detach().clone().float()
    out = torch.full_like(q, -1.0)
    bsz = q.shape[0]
    for i in range(bsz):
        known = torch.where(q[i] >= 0)[0]
        if len(known) == 0:
            continue
        lo = min(max(1, min_known), len(known))
        hi = min(max(lo, max_known), len(known))
        k = int(torch.randint(lo, hi + 1, (1,), device=q.device).item()) if hi > lo else lo

        pos = known[q[i, known] > 0.5]
        neg = known[q[i, known] <= 0.5]
        selected = []
        # Keep one positive whenever available; it helps discriminative sparse queries.
        if len(pos) > 0 and k > 0:
            selected.append(pos[torch.randint(0, len(pos), (1,), device=q.device)].item())
        # Keep one negative when possible as well.
        if len(neg) > 0 and len(selected) < k:
            selected.append(neg[torch.randint(0, len(neg), (1,), device=q.device)].item())

        remaining = known[~torch.isin(known, torch.as_tensor(selected, device=q.device, dtype=known.dtype))] if selected else known
        need = k - len(selected)
        if need > 0 and len(remaining) > 0:
            perm = torch.randperm(len(remaining), device=q.device)[:need]
            selected += remaining[perm].tolist()
        idx = torch.as_tensor(selected, dtype=torch.long, device=q.device)
        out[i, idx] = q[i, idx]
    return out


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class QueryEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, negative_token_scale: float = 0.35):
        super().__init__()
        self.negative_token_scale = float(negative_token_scale)
        self.pos_basis = nn.Parameter(torch.randn(NUM_ATTRIBUTES, embed_dim) * 0.02)
        self.neg_basis = nn.Parameter(torch.randn(NUM_ATTRIBUTES, embed_dim) * 0.02)
        self.count_embed = nn.Sequential(
            nn.Linear(2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, query):
        q = query.float()
        valid = q >= 0
        pos = (q > 0.5) & valid
        neg = (q <= 0.5) & valid
        tok = pos.unsqueeze(-1) * self.pos_basis.unsqueeze(0)
        tok = tok + self.negative_token_scale * neg.unsqueeze(-1) * self.neg_basis.unsqueeze(0)
        denom = (pos.float() + self.negative_token_scale * neg.float()).sum(1, keepdim=True).clamp_min(1.0)
        z = tok.sum(1) / denom.sqrt()
        counts = torch.stack([
            pos.float().sum(1) / NUM_ATTRIBUTES,
            neg.float().sum(1) / NUM_ATTRIBUTES,
        ], dim=1)
        z = z + self.count_embed(counts)
        z = z + self.mlp(z)
        return F.normalize(z, dim=-1)


class SpatialAttributeHead(nn.Module):
    """Attribute-specific spatial attention head for local clothing/accessory cues."""
    def __init__(self, in_dim: int, value_dim: int = 256, num_attributes: int = NUM_ATTRIBUTES):
        super().__init__()
        self.attn = nn.Conv2d(in_dim, num_attributes, kernel_size=1, bias=True)
        self.value = nn.Conv2d(in_dim, value_dim, kernel_size=1, bias=False)
        self.attr_weight = nn.Parameter(torch.randn(num_attributes, value_dim) * 0.02)
        self.attr_bias = nn.Parameter(torch.zeros(num_attributes))

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        b, _, h, w = fmap.shape
        attn = self.attn(fmap).flatten(2)                # [B,A,N]
        attn = torch.softmax(attn, dim=-1)
        value = self.value(fmap).flatten(2)              # [B,D,N]
        pooled = torch.einsum('ban,bdn->bad', attn, value)
        return (pooled * self.attr_weight.unsqueeze(0)).sum(-1) + self.attr_bias.unsqueeze(0)


class StripeAttributeHead(nn.Module):
    """Attribute-specific soft selection over horizontal body stripes.

    Pedestrian attributes are strongly localized (head, torso, legs, accessories).
    This head keeps the implementation lightweight while preserving location cues.
    """
    def __init__(self, in_dim: int, num_attributes: int = NUM_ATTRIBUTES, stripes: int = 4):
        super().__init__()
        self.stripes = int(stripes)
        self.attr_stripe_logits = nn.Parameter(torch.zeros(num_attributes, self.stripes))
        self.attr_weight = nn.Parameter(torch.randn(num_attributes, in_dim) * 0.02)
        self.attr_bias = nn.Parameter(torch.zeros(num_attributes))

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,S,C]
        pooled = F.adaptive_avg_pool2d(fmap, (self.stripes, 1)).squeeze(-1).transpose(1, 2)
        weights = torch.softmax(self.attr_stripe_logits, dim=-1)  # [A,S]
        attr_feat = torch.einsum('as,bsc->bac', weights, pooled)
        return (attr_feat * self.attr_weight.unsqueeze(0)).sum(-1) + self.attr_bias.unsqueeze(0)


class ABPRNet(nn.Module):
    def __init__(self, backbone: str = 'convnext_tiny', embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone_name = str(backbone)
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
            mix_cfg = {2: 0.30, 4: 0.20}
        elif backbone == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            try:
                base = convnext_tiny(weights=weights)
            except Exception:
                base = convnext_tiny(weights=None)
            self.features = base.features
            self.avgpool = base.avgpool
            feat_dim = base.classifier[2].in_features
            mix_cfg = {2: 0.25, 4: 0.15}
        elif backbone == 'convnext_small':
            weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            try:
                base = convnext_small(weights=weights)
            except Exception:
                base = convnext_small(weights=None)
            self.features = base.features
            self.avgpool = base.avgpool
            feat_dim = base.classifier[2].in_features
            mix_cfg = {2: 0.22, 4: 0.12}
        elif backbone == 'convnext_base':
            weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            try:
                base = convnext_base(weights=weights)
            except Exception:
                base = convnext_base(weights=None)
            self.features = base.features
            self.avgpool = base.avgpool
            feat_dim = base.classifier[2].in_features
            mix_cfg = {2: 0.18, 4: 0.10}
        elif backbone == 'efficientnet_v2_s':
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            try:
                base = efficientnet_v2_s(weights=weights)
            except Exception:
                base = efficientnet_v2_s(weights=None)
            self.features = base.features
            self.avgpool = base.avgpool
            feat_dim = base.classifier[-1].in_features
            mix_cfg = {2: 0.20, 4: 0.10}
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.mixstyles = nn.ModuleDict({str(k): MixStyle(p=v, alpha=0.1) for k, v in mix_cfg.items()})
        self.global_attr_head = nn.Linear(feat_dim, NUM_ATTRIBUTES)
        self.spatial_attr_head = SpatialAttributeHead(feat_dim, value_dim=min(320, feat_dim))
        self.stripe_attr_head = StripeAttributeHead(feat_dim, NUM_ATTRIBUTES, stripes=4)
        self.prototype_head = AttributePrototypeHead(
            feat_dim,
            prototype_dim=min(256, max(128, embed_dim)),
            num_attributes=NUM_ATTRIBUTES,
            init_scale=8.0,
        )
        # Per-attribute 4-way fusion: global / spatial / stripes / prototype-margin.
        # Prototype starts with a smaller prior contribution and earns more weight
        # only if it improves the supervised retrieval objective.
        init_fusion = torch.zeros(NUM_ATTRIBUTES, 4)
        init_fusion[:, 3] = -1.10
        self.attr_fusion_logits = nn.Parameter(init_fusion)
        self.image_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(embed_dim, embed_dim),
        )
        self.query_encoder = QueryEncoder(embed_dim=embed_dim, negative_token_scale=0.35)
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.15), nn.Linear(256, len(DOMAIN_NAMES))
        )

    def extract_feature_map(self, x):
        for idx, block in enumerate(self.features):
            x = block(x)
            key = str(idx)
            if key in self.mixstyles:
                x = self.mixstyles[key](x)
        return x

    def encode_images(self, x, return_prototype: bool = False):
        fmap = self.extract_feature_map(x)
        feat = torch.flatten(self.avgpool(fmap), 1)
        g = self.global_attr_head(feat)
        s = self.spatial_attr_head(fmap)
        r = self.stripe_attr_head(fmap)
        proto_logits, attr_feat, sim_pos, sim_neg = self.prototype_head(fmap, feat)
        fusion = torch.softmax(self.attr_fusion_logits, dim=-1)
        logits = (fusion[:, 0].unsqueeze(0) * g +
                  fusion[:, 1].unsqueeze(0) * s +
                  fusion[:, 2].unsqueeze(0) * r +
                  fusion[:, 3].unsqueeze(0) * proto_logits)
        emb = F.normalize(self.image_proj(feat), dim=-1)
        if return_prototype:
            return logits, emb, feat, proto_logits, attr_feat, sim_pos, sim_neg
        return logits, emb, feat

    def forward(self, x, query=None, grl_lambda: float = 0.0, return_domain: bool = False):
        logits, emb, feat = self.encode_images(x)
        qz = self.query_encoder(query) if query is not None else None
        if not return_domain:
            return logits, emb, qz
        dlogits = self.domain_head(grad_reverse(feat, grl_lambda))
        return logits, emb, qz, dlogits


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

class MaskedFocalLoss(nn.Module):
    """Masked focal BCE without inverse-frequency class weighting.

    UPAR retrieval is sensitive to probability calibration. Heavy positive-class
    reweighting can improve classification balance while degrading retrieval score
    calibration. Focal modulation emphasizes hard examples without permanently
    changing the positive/negative base cost.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets, valid):
        logits = logits.float(); targets = targets.float(); valid = valid.float()
        smooth = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, smooth, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1.0 - targets) * (1.0 - p)
        focal = (1.0 - pt).clamp_min(0.0).pow(self.gamma)
        return (bce * focal * valid).sum() / valid.sum().clamp_min(1.0)


GROUPS = [
    [0, 1, 2],          # age
    [4, 5, 6],          # hair length/bald
    list(range(8, 20)),  # upper color
    list(range(21, 33)), # lower color
    [33, 34],           # trousers/shorts vs skirt/dress when exactly one is annotated positive
]


def group_consistency_loss(logits, targets, valid):
    losses = []
    for group in GROUPS:
        idx = torch.as_tensor(group, device=logits.device)
        y = targets[:, idx]
        v = valid[:, idx] > 0.5
        eligible = v.all(dim=1) & ((y > 0.5).sum(dim=1) == 1)
        if eligible.any():
            cls = torch.argmax(y[eligible], dim=1)
            losses.append(F.cross_entropy(logits[eligible][:, idx], cls))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def alignment_loss(image_emb, query_emb):
    return (1.0 - (image_emb * query_emb).sum(-1)).mean()


def degree_of_match_contrastive_loss(
    image_emb: torch.Tensor,
    query_emb: torch.Tensor,
    query_states: torch.Tensor,
    gallery_targets: torch.Tensor,
    gallery_valid: torch.Tensor,
    temperature: float = 0.07,
    beta: float = 7.0,
):
    """Soft retrieval loss: target mass follows in-batch degree-of-match.

    This is closer to mADM than ordinary one-positive InfoNCE because partially
    matching images are treated as useful neighbors instead of hard negatives.
    """
    q = query_states.float()
    valid_q = q >= 0
    q01 = torch.where(valid_q, q.clamp(0, 1), torch.zeros_like(q))
    y = gallery_targets.float()
    yv = gallery_valid > 0.5

    # [Q,B,A], batches are modest so this tensor is safe.
    both = valid_q[:, None, :] & yv[None, :, :]
    matches = (q01[:, None, :] == y[None, :, :]) & both
    denom = both.sum(-1).clamp_min(1)
    dom = matches.sum(-1).float() / denom.float()

    logits = (query_emb @ image_emb.t()) / float(temperature)
    soft_target = torch.softmax(float(beta) * dom.detach(), dim=1)
    return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def probability_degree_match_loss(
    logits: torch.Tensor,
    query_states: torch.Tensor,
    gallery_targets: torch.Tensor,
    gallery_valid: torch.Tensor,
) -> torch.Tensor:
    """Directly regress probabilistic query/gallery match toward ground-truth DoM.

    For a known query bit q and predicted gallery probability p, the expected
    match is p for q=1 and (1-p) for q=0. Averaging over known query attributes
    gives a differentiable approximation of Degree-of-Match. We supervise the
    complete in-batch QxG matrix, not only the diagonal pair.
    """
    p = torch.sigmoid(logits.float())
    q = query_states.float()
    valid_q = q >= 0
    q01 = torch.where(valid_q, q.clamp(0, 1), torch.zeros_like(q))
    y = gallery_targets.float()
    yv = gallery_valid > 0.5

    both = valid_q[:, None, :] & yv[None, :, :]
    denom = both.sum(-1).clamp_min(1).float()
    pred_each = q01[:, None, :] * p[None, :, :] + (1.0 - q01[:, None, :]) * (1.0 - p[None, :, :])
    pred_dom = (pred_each * both.float()).sum(-1) / denom
    true_match = (q01[:, None, :] == y[None, :, :]) & both
    true_dom = true_match.sum(-1).float() / denom
    mask = both.any(-1)
    if not mask.any():
        return logits.sum() * 0.0
    return F.smooth_l1_loss(pred_dom[mask], true_dom.detach()[mask], beta=0.08)


def degree_match_listwise_loss(
    logits: torch.Tensor,
    query_states: torch.Tensor,
    gallery_targets: torch.Tensor,
    gallery_valid: torch.Tensor,
    temperature: float = 0.10,
    target_beta: float = 8.0,
) -> torch.Tensor:
    """Listwise ranking loss that orders gallery samples by Degree-of-Match."""
    p = torch.sigmoid(logits.float())
    q = query_states.float()
    valid_q = q >= 0
    q01 = torch.where(valid_q, q.clamp(0, 1), torch.zeros_like(q))
    y = gallery_targets.float(); yv = gallery_valid > 0.5
    both = valid_q[:, None, :] & yv[None, :, :]
    denom = both.sum(-1).clamp_min(1).float()
    pred_each = q01[:, None, :] * p[None, :, :] + (1.0 - q01[:, None, :]) * (1.0 - p[None, :, :])
    pred = (pred_each * both.float()).sum(-1) / denom
    true = (((q01[:, None, :] == y[None, :, :]) & both).sum(-1).float() / denom).detach()
    valid_rows = both.any(-1).sum(-1) > 1
    if not valid_rows.any():
        return logits.sum() * 0.0
    logp = F.log_softmax(pred[valid_rows] / float(temperature), dim=1)
    target = F.softmax(float(target_beta) * true[valid_rows], dim=1)
    return -(target * logp).sum(1).mean()


# -----------------------------------------------------------------------------
# Validation / metrics
# -----------------------------------------------------------------------------

def relevance_matrix(queries: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    queries = np.asarray(queries)
    gallery_labels = np.asarray(gallery_labels)
    out = np.zeros((len(queries), len(gallery_labels)), dtype=bool)
    for i, q in enumerate(queries):
        valid = q >= 0
        if not valid.any():
            continue
        y = gallery_labels[:, valid]
        qv = q[valid]
        known = y >= 0
        out[i] = known.all(axis=1) & (y == qv[None, :]).all(axis=1)
    return out


def _per_query_madm(distances: np.ndarray, queries: np.ndarray, gallery_labels: np.ndarray):
    d = np.asarray(distances)
    q_all = np.asarray(queries)
    y_all = np.asarray(gallery_labels)
    values = []
    for qi, q in enumerate(q_all):
        q_valid = q >= 0
        if not q_valid.any():
            values.append(np.nan); continue
        y = y_all[:, q_valid]
        qv = q[q_valid]
        known = y >= 0
        n_known = known.sum(1)
        matches = ((y == qv[None, :]) & known).sum(1)
        dom = np.divide(matches, np.maximum(n_known, 1), out=np.zeros_like(matches, dtype=np.float64), where=n_known > 0)
        rel = (n_known == int(q_valid.sum())) & (matches == int(q_valid.sum()))
        n_rel = int(rel.sum())
        if n_rel == 0:
            values.append(np.nan); continue
        mean_dom = float(dom.mean())
        dom_norm = np.maximum(0.0, (dom - mean_dom) / max(1.0 - mean_dom, 1e-12))
        order = np.argsort(d[qi], kind='stable')
        ranked_rel = rel[order]
        ranked_dom = dom_norm[order]
        prec_dom = np.cumsum(ranked_dom) / np.arange(1, len(order) + 1)
        values.append(float((prec_dom * ranked_rel).sum() / n_rel))
    return np.asarray(values, dtype=np.float64)


def track2_metrics(distances: np.ndarray | torch.Tensor, queries: np.ndarray, gallery_labels: np.ndarray) -> dict:
    d = distances.detach().cpu().numpy() if torch.is_tensor(distances) else np.asarray(distances)
    rel = relevance_matrix(queries, gallery_labels)
    aps=[]; r1=[]; r5=[]; r10=[]; inps=[]
    for i in range(len(queries)):
        n_rel = int(rel[i].sum())
        if n_rel == 0:
            continue
        order = np.argsort(d[i], kind='stable')
        rr = rel[i][order]
        cs = np.cumsum(rr)
        ranks = np.arange(1, len(rr)+1)
        aps.append(float(((cs / ranks) * rr).sum() / n_rel))
        r1.append(float(rr[:1].any()))
        r5.append(float(rr[:5].any()))
        r10.append(float(rr[:10].any()))
        last = int(np.where(rr)[0][-1]) + 1
        inps.append(float(n_rel / last))

    adm = _per_query_madm(d, queries, gallery_labels)
    adm = adm[np.isfinite(adm)]

    def ms(x):
        a=np.asarray(x,dtype=np.float64)
        return (float(a.mean()) if len(a) else 0.0, float(a.std()) if len(a) else 0.0)
    mAP,mAPs=ms(aps); a1,a1s=ms(r1); a5,a5s=ms(r5); a10,a10s=ms(r10); mi,mis=ms(inps); ma,mas=ms(adm)
    return {
        'mAP': mAP, 'mAP_std': mAPs,
        'Rank-1': a1, 'Rank-1_std': a1s,
        'Rank-5': a5, 'Rank-5_std': a5s,
        'Rank-10': a10, 'Rank-10_std': a10s,
        'mINP': mi, 'mINP_std': mis,
        'mADM': ma, 'mADM_std': mas,
        'num_queries': int(len(aps)),
    }


def apply_affine_calibration(
    probs: torch.Tensor,
    scale: torch.Tensor | np.ndarray | None = None,
    bias: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    p = probs.float().cpu().clamp(1e-6, 1.0 - 1e-6)
    if scale is None and bias is None:
        return p
    s = torch.ones(p.shape[1]) if scale is None else torch.as_tensor(scale, dtype=torch.float32).flatten().cpu()
    b = torch.zeros(p.shape[1]) if bias is None else torch.as_tensor(bias, dtype=torch.float32).flatten().cpu()
    if len(s) != p.shape[1] or len(b) != p.shape[1]:
        raise ValueError('Affine calibration size mismatch')
    logits = torch.log(p / (1.0 - p))
    return torch.sigmoid(logits * s[None, :] + b[None, :])


def fit_affine_calibration(
    probs: torch.Tensor,
    labels: np.ndarray,
    valid: np.ndarray,
    steps: int = 160,
    lr: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one positive scale and one bias per attribute using masked BCE."""
    p = probs.float().cpu().clamp(1e-5, 1.0 - 1e-5)
    base_logits = torch.log(p / (1.0 - p))
    m = torch.as_tensor(valid, dtype=torch.float32)
    y_raw = torch.as_tensor(labels, dtype=torch.float32)
    y = torch.where(m > 0, y_raw.clamp(0, 1), torch.zeros_like(y_raw))
    raw_scale = torch.zeros(p.shape[1], requires_grad=True)
    bias = torch.zeros(p.shape[1], requires_grad=True)
    opt = torch.optim.Adam([raw_scale, bias], lr=float(lr))
    for _ in range(int(steps)):
        scale = F.softplus(raw_scale) + 0.25
        z = base_logits * scale[None, :] + bias[None, :]
        loss = (F.binary_cross_entropy_with_logits(z, y, reduction='none') * m).sum() / m.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    scale = (F.softplus(raw_scale) + 0.25).detach().clamp(0.25, 4.0).numpy().astype(np.float32)
    b = bias.detach().clamp(-3.0, 3.0).numpy().astype(np.float32)
    return scale, b


# -----------------------------------------------------------------------------
# Retrieval distance
# -----------------------------------------------------------------------------

def _query_weight_matrix(
    queries: torch.Tensor,
    attribute_weights: torch.Tensor,
    priors: Optional[torch.Tensor] = None,
    positive_rarity_power: float = 0.0,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    valid = queries >= 0
    pos = (queries > 0.5) & valid
    neg = (queries <= 0.5) & valid
    base = attribute_weights[None, :].expand(len(queries), -1)

    pos_factor = torch.ones_like(base)
    if priors is not None and positive_rarity_power > 0:
        pr = priors.clamp(1e-4, 1.0-1e-4)
        rarity = (-torch.log(pr)).pow(float(positive_rarity_power))
        rarity = rarity / rarity.mean().clamp_min(1e-6)
        rarity = rarity.clamp(0.35, 3.5)
        pos_factor = rarity[None, :].expand_as(base)

    w = pos.float() * base * pos_factor + neg.float() * base * float(negative_weight)
    return w


def calibrated_attribute_distances(
    gallery_probs: torch.Tensor,
    queries: torch.Tensor | np.ndarray,
    attribute_weights: torch.Tensor | np.ndarray | None = None,
    attribute_temperatures: torch.Tensor | np.ndarray | None = None,
    distance_kind: str = 'nll',
    gallery_emb: torch.Tensor | None = None,
    query_emb: torch.Tensor | None = None,
    embedding_mix: float = 0.0,
    priors: torch.Tensor | np.ndarray | None = None,
    positive_rarity_power: float = 0.0,
    negative_weight: float = 1.0,
    calibration_scale: torch.Tensor | np.ndarray | None = None,
    calibration_bias: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    p = apply_attribute_temperatures(gallery_probs, attribute_temperatures).cpu().clamp(1e-6, 1-1e-6)
    p = apply_affine_calibration(p, calibration_scale, calibration_bias).clamp(1e-6, 1-1e-6)
    q = torch.as_tensor(queries, dtype=torch.float32).cpu()
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != q.shape[1]:
        raise ValueError(f'Bad p/q shapes: {tuple(p.shape)} / {tuple(q.shape)}')
    w = torch.ones(p.shape[1], dtype=torch.float32) if attribute_weights is None else torch.as_tensor(attribute_weights, dtype=torch.float32).flatten().cpu().clamp_min(1e-6)
    if len(w) != p.shape[1]:
        raise ValueError('attribute_weights size mismatch')
    pr = None if priors is None else torch.as_tensor(priors, dtype=torch.float32).flatten().cpu()
    if pr is not None and len(pr) != p.shape[1]:
        raise ValueError('priors size mismatch')

    valid = q >= 0
    q01 = torch.where(valid, q.clamp(0,1), torch.zeros_like(q))
    wq = _query_weight_matrix(q, w, pr, positive_rarity_power, negative_weight)
    denom = wq.sum(1, keepdim=True).clamp_min(1e-6)

    kind = str(distance_kind).lower()
    if kind == 'match':
        # Negative expected Degree-of-Match. Smaller distance = better match.
        sim = wq @ (1.0 - p).t() + (q01*wq) @ (2.0*p - 1.0).t()
        d = -sim
    elif kind == 'l1':
        d = wq @ p.t() + (q01*wq) @ (1.0 - 2.0*p).t()
    elif kind == 'l2':
        d = wq @ (p*p).t() + (q01*wq) @ (1.0 - 2.0*p).t()
    elif kind == 'nll':
        neglog0 = -torch.log1p(-p)
        delta = torch.log1p(-p) - torch.log(p)
        d = wq @ neglog0.t() + (q01*wq) @ delta.t()
    else:
        raise ValueError(f'Unsupported distance_kind={distance_kind}')
    d = d / denom

    mix = float(embedding_mix)
    if mix > 0:
        if gallery_emb is None or query_emb is None:
            raise ValueError('Embeddings required when embedding_mix > 0')
        gz = F.normalize(gallery_emb.float().cpu(), dim=-1)
        qz = F.normalize(query_emb.float().cpu(), dim=-1)
        de = (1.0 - qz @ gz.t()) * 0.5
        d0 = (d-d.mean(1,keepdim=True))/d.std(1,keepdim=True).clamp_min(1e-6)
        de = (de-de.mean(1,keepdim=True))/de.std(1,keepdim=True).clamp_min(1e-6)
        d = (1.0-mix)*d0 + mix*de
    return d


@dataclass
class TrainConfig:
    backbone: str = 'convnext_small'
    embed_dim: int = 320
    image_height: int = 384
    image_width: int = 192
    batch_size: int = 16
    eval_batch_size: int = 40
    epochs: int = 24
    backbone_lr: float = 4e-5
    head_lr: float = 2e-4
    weight_decay: float = 5e-4
    num_workers: int = 2
    ema_decay: float = 0.9994
    attr_loss_weight: float = 1.0
    probability_dom_weight: float = 0.18
    listwise_dom_weight: float = 0.08
    align_loss_weight: float = 0.03
    dom_contrast_weight: float = 0.05
    group_loss_weight: float = 0.03
    domain_loss_weight: float = 0.01
    prototype_bce_weight: float = 0.18
    prototype_domain_weight: float = 0.05
    prototype_separation_weight: float = 0.01
    prototype_warmup_fraction: float = 0.20
    grl_lambda: float = 0.10
    contrast_temperature: float = 0.08
    query_min_known: int = 2
    query_max_known: int = 10
    focal_gamma: float = 2.0
    warmup_ratio: float = 0.06
