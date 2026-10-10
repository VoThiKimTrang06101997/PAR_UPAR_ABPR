"""One-pass attribute-balanced/query-aware/domain-generalized ABPR objectives.
TRAIN labels only. Mask unknown attributes. No hidden labels or metadata at inference.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def masked_asl_loss(logits, labels, valid, priors, gamma_neg=3.0, gamma_pos=0.0, clip=0.03):
    """Asymmetric loss averaged over observed attributes; -1 is never negative.

    Positive weights are clipped to avoid rare-label overamplification.
    Priors are estimated only from public TRAIN ground truth.
    """
    logits = logits.float()
    labels = labels.float()
    mask = (valid > 0) & ((labels == 0) | (labels == 1))
    if not bool(mask.any()):
        return logits.sum() * 0.0
    p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    pn = (1-p + float(clip)).clamp(max=1.0)
    y = labels.clamp(0, 1)
    pw = torch.sqrt((1.0-priors.float().clamp(0.01,0.99)) /
                    priors.float().clamp(0.01,0.99)).clamp(0.7, 2.5)
    pos = -y * torch.log(p) * (1-p).pow(gamma_pos) * pw.view(1,-1)
    neg = -(1-y) * torch.log(pn.clamp_min(1e-6)) * (1-pn).pow(gamma_neg)
    losses = (pos + neg) * mask.float()
    per_attr = losses.sum(0) / mask.float().sum(0).clamp_min(1)
    observed = mask.any(0)
    return per_attr[observed].mean()


def query_hard_ranking_loss(z, qz, sparse_q, full_labels, margin=0.12):
    """Hard negatives must contradict a known query attribute.

    The diagonal is the anchor image matching its own sampled query.
    Unknown gallery labels cannot create a contradiction.
    """
    z = F.normalize(z.float(), dim=-1)
    qz = F.normalize(qz.float(), dim=-1)
    query = sparse_q.float()
    gallery = full_labels.float()
    if z.ndim != 2 or z.shape != qz.shape or query.shape != gallery.shape:
        raise ValueError('Incompatible embedding/query shapes')
    if z.size(0) != query.size(0):
        raise ValueError('Mismatched batch size')
    if z.size(0) < 2:
        return z.sum() * 0.0
    known = (query[:,:,None] != -1) & (gallery.T[None,:,:] != -1)
    # Shape [B(query), A(attr), B(gallery)]
    conflict = known & (query[:,:,None] != gallery.T[None,:,:])
    confirmed_neg = conflict.any(dim=1)
    confirmed_neg.fill_diagonal_(False)
    valid_row = confirmed_neg.any(dim=1)
    if not bool(valid_row.any()):
        return (z.sum() + qz.sum()) * 0.0
    sim = qz @ z.T
    hardest = sim.masked_fill(~confirmed_neg, -1e4).max(dim=1).values
    matched = sim.diagonal()
    return F.relu(float(margin) + hardest[valid_row] - matched[valid_row]).mean()


def mild_photometric_view(x):
    """Small normalized-space brightness/contrast noise; no cropping or label flip."""
    b = x.size(0)
    scale = torch.empty((b,1,1,1), device=x.device, dtype=x.dtype).uniform_(0.97, 1.03)
    offset = torch.empty((b,1,1,1), device=x.device, dtype=x.dtype).uniform_(-0.025,0.025)
    return (x * scale + offset).contiguous()


def consistency_loss(primary_logits, view_logits, primary_z, view_z):
    p_ref = torch.sigmoid(primary_logits.float()).detach()
    p_aug = torch.sigmoid(view_logits.float())
    cls = F.mse_loss(p_aug, p_ref)
    emb = (1 - F.cosine_similarity(F.normalize(view_z.float(),dim=-1),
                                   F.normalize(primary_z.float().detach(),dim=-1),dim=-1)).mean()
    return cls + 0.25 * emb


def ema_kd_loss(student_logits, teacher_logits, student_z=None, teacher_z=None, temp=2.0):
    t = float(temp)
    target = torch.sigmoid(teacher_logits.float().detach()/t)
    cls = F.binary_cross_entropy_with_logits(student_logits.float()/t, target)*t*t
    if student_z is None or teacher_z is None:
        return cls
    emb = (1-F.cosine_similarity(F.normalize(student_z.float(),dim=-1),
                                 F.normalize(teacher_z.float().detach(),dim=-1),dim=-1)).mean()
    return cls + 0.25*emb
