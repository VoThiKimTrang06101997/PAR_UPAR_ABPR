"""Backbone-level multi-label contrastive and query-hard-negative losses.

Call from the original training graph, on *current minibatch* image and query
embeddings, so gradients update the Prototype encoder/backbone. It never loads
VAL labels. Missing labels (-1) never become positives or negatives by default.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _labels_and_masks(queries: torch.Tensor):
    q = torch.as_tensor(queries)
    if q.ndim != 2:
        raise ValueError(f'Expected [B,A] query states, got {tuple(q.shape)}')
    known = (q == 0) | (q == 1)
    binary = q == 1
    both = known[:, None, :] & known[None, :, :]
    agree = ((binary[:, None, :] == binary[None, :, :]) & both).sum(dim=-1)
    disagree = ((binary[:, None, :] != binary[None, :, :]) & both).sum(dim=-1)
    shared = both.sum(dim=-1)
    shared_pos = (binary[:, None, :] & binary[None, :, :] & both).sum(dim=-1)
    return shared, agree, disagree, shared_pos


def multilabel_supcon(image_emb: torch.Tensor, queries: torch.Tensor,
                      temperature: float = 0.12, min_shared: int = 4,
                      min_agreement: float = 0.80):
    """Soft-positive SupCon. Valid image pairs need known, agreeing labels and
    at least one shared positive label. Unknown attributes are ignored."""
    z = F.normalize(image_emb.float(), dim=-1)
    b = z.shape[0]
    if b < 2:
        return z.sum() * 0.0
    shared, agree, disagree, pos = _labels_and_masks(queries)
    valid_pos = (shared >= min_shared) & (agree.float() / shared.clamp_min(1) >= min_agreement) & (pos >= 1)
    valid_pos.fill_diagonal_(False)
    weights = (agree.float() / shared.clamp_min(1)).pow(2) * valid_pos.float()
    # Numpy/Pytorch dtype-agnostic; gradients must flow into z.
    weights = weights.to(z.dtype)
    logit = (z @ z.T) / max(float(temperature), 1e-4)
    mask_eye = torch.eye(b, dtype=torch.bool, device=z.device)
    logit = logit.masked_fill(mask_eye, -1e4)
    log_prob = logit - torch.logsumexp(logit, dim=1, keepdim=True)
    den = weights.sum(dim=1)
    selected = den > 0
    if not bool(selected.any()):
        return z.sum() * 0.0
    per_anchor = -(weights * log_prob).sum(dim=1) / den.clamp_min(1e-6)
    return per_anchor[selected].mean()


def query_hard_negative_loss(image_emb: torch.Tensor, query_emb: torch.Tensor,
                             queries: torch.Tensor, margin: float = 0.12,
                             min_shared: int = 4, min_matching: int = 2,
                             max_disagree: int = 4):
    """Within minibatch rank the matched image above a similar but provably
    contradictory image (at least one conflicting *known* query attribute).
    Does not treat missing labels as 0, and does not mine from VAL.
    """
    z = F.normalize(image_emb.float(), dim=-1)
    qz = F.normalize(query_emb.float(), dim=-1)
    if z.shape != qz.shape:
        raise ValueError(f'Image/query embedding mismatch: {tuple(z.shape)} vs {tuple(qz.shape)}')
    b = z.size(0)
    if b < 2:
        return z.sum() * 0.0
    shared, agree, disagree, _ = _labels_and_masks(queries)
    cand = (shared >= min_shared) & (agree >= min_matching) & (disagree >= 1) & (disagree <= max_disagree)
    cand.fill_diagonal_(False)
    sims = qz @ z.T
    hardest = sims.masked_fill(~cand, -1e4).max(dim=1).values
    selected = cand.any(dim=1)
    if not bool(selected.any()):
        return sims.sum() * 0.0
    positive = sims.diagonal()
    return F.relu(float(margin) + hardest[selected] - positive[selected]).mean()


class AttributeAwareContrastive:
    """Wrap the original InfoNCE without replacing classification/prototype losses.
    Increasing warmup slowly is safer than injecting strong contrastive gradients
    into an already trained model on the first update.
    """
    def __init__(self, original, supcon_weight: float = 0.40,
                 hard_weight: float = 0.70, warmup_steps: int = 250):
        self.original = original
        self.supcon_weight = float(supcon_weight)
        self.hard_weight = float(hard_weight)
        self.warmup_steps = max(1, int(warmup_steps))
        self.steps = 0
        self.last_stats = {}

    def __call__(self, image_emb, query_emb, query_states, temperature=0.08, **kwargs):
        self.steps += 1
        if query_emb is None:
            raise RuntimeError('Training must provide query embeddings for contrastive loss')
        base = self.original(image_emb, query_emb, query_states, temperature=temperature, **kwargs)
        sup = multilabel_supcon(image_emb, query_states)
        hard = query_hard_negative_loss(image_emb, query_emb, query_states)
        ramp = min(1., self.steps / self.warmup_steps)
        self.last_stats = {'info_nce':float(base.detach()), 'supcon':float(sup.detach()),
                           'hard':float(hard.detach()), 'ramp':ramp}
        return base + ramp * (self.supcon_weight * sup + self.hard_weight * hard)
