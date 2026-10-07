from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import NUM_ATTRIBUTES


class AttributePrototypeHead(nn.Module):
    """Positive/negative visual prototypes for each pedestrian attribute.

    The head builds an attribute-specific visual feature from a spatially attended
    feature map and the global backbone feature. For each attribute ``a`` it learns
    two normalized prototype vectors: ``P[a,+]`` and ``P[a,-]``. The prototype logit
    is a scaled cosine-margin:

        scale * (cos(h_a, P[a,+]) - cos(h_a, P[a,-]))

    This is intentionally lightweight: it adds semantic structure to the Track-2
    representation without replacing the existing global/spatial/stripe heads.
    """

    def __init__(
        self,
        in_dim: int,
        prototype_dim: int = 192,
        num_attributes: int = NUM_ATTRIBUTES,
        init_scale: float = 8.0,
    ):
        super().__init__()
        self.num_attributes = int(num_attributes)
        self.prototype_dim = int(prototype_dim)

        self.attn = nn.Conv2d(in_dim, self.num_attributes, kernel_size=1, bias=True)
        self.value = nn.Conv2d(in_dim, self.prototype_dim, kernel_size=1, bias=False)
        self.global_proj = nn.Linear(in_dim, self.prototype_dim, bias=False)
        self.local_gate_logits = nn.Parameter(torch.zeros(self.num_attributes))

        self.pos_prototypes = nn.Parameter(
            F.normalize(torch.randn(self.num_attributes, self.prototype_dim), dim=-1) * 0.10
        )
        self.neg_prototypes = nn.Parameter(
            F.normalize(torch.randn(self.num_attributes, self.prototype_dim), dim=-1) * 0.10
        )
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)).log())

    def normalized_prototypes(self):
        pos = F.normalize(self.pos_prototypes.float(), dim=-1)
        neg = F.normalize(self.neg_prototypes.float(), dim=-1)
        return pos, neg

    def forward(self, fmap: torch.Tensor, global_feat: torch.Tensor):
        # Attribute-local feature [B,A,D]
        attn = torch.softmax(self.attn(fmap).flatten(2), dim=-1)
        values = self.value(fmap).flatten(2)
        local = torch.einsum('ban,bdn->bad', attn, values)

        # Global feature contributes differently to each attribute. This helps
        # coarse labels (gender/age) while spatial attention remains useful for
        # small/local labels such as hat, bag, and glasses.
        glob = self.global_proj(global_feat).unsqueeze(1).expand(-1, self.num_attributes, -1)
        local_gate = torch.sigmoid(self.local_gate_logits).view(1, self.num_attributes, 1)
        attr_feat = F.normalize(local_gate * local + (1.0 - local_gate) * glob, dim=-1)

        pos, neg = self.normalized_prototypes()
        sim_pos = torch.einsum('bad,ad->ba', attr_feat.float(), pos)
        sim_neg = torch.einsum('bad,ad->ba', attr_feat.float(), neg)
        scale = self.logit_scale.exp().clamp(1.0, 30.0)
        logits = scale * (sim_pos - sim_neg)
        return logits.to(attr_feat.dtype), attr_feat, sim_pos, sim_neg


def masked_prototype_bce(proto_logits, targets, valid):
    proto_logits = proto_logits.float()
    targets = targets.float()
    valid = valid.float()
    loss = F.binary_cross_entropy_with_logits(proto_logits, targets, reduction='none')
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def prototype_domain_alignment_loss(
    attr_feat: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    domains: torch.Tensor,
    pos_prototypes: torch.Tensor,
    neg_prototypes: torch.Tensor,
    min_count: int = 2,
):
    """Align domain-specific positive/negative centroids to shared prototypes.

    This directly targets hidden-domain robustness: Market1501, PA-100K and PETA
    samples for the same attribute/class are encouraged to share a visual direction.
    Unknown labels are ignored. Domains/classes without enough samples in the batch
    are skipped rather than producing noisy centroids.
    """
    feat = F.normalize(attr_feat.float(), dim=-1)
    y = targets.float()
    v = valid.float()
    pos = F.normalize(pos_prototypes.float(), dim=-1)
    neg = F.normalize(neg_prototypes.float(), dim=-1)

    losses = []
    for d in torch.unique(domains.detach()):
        if int(d.item()) < 0:
            continue
        dm = (domains == d).float().unsqueeze(1)
        for cls, proto in ((1.0, pos), (0.0, neg)):
            cls_mask = y if cls > 0.5 else (1.0 - y)
            w = v * cls_mask * dm
            count = w.sum(0)  # [A]
            keep = count >= float(min_count)
            if not torch.any(keep):
                continue
            centroid = torch.einsum('ba,bad->ad', w, feat) / count.clamp_min(1.0).unsqueeze(-1)
            centroid = F.normalize(centroid, dim=-1)
            cos = (centroid * proto).sum(-1)
            losses.append((1.0 - cos[keep]).mean())

    if not losses:
        return attr_feat.sum() * 0.0
    return torch.stack(losses).mean()


def prototype_separation_loss(pos_prototypes, neg_prototypes, max_cosine: float = 0.15):
    """Prevent the positive and negative prototype of the same attribute collapsing."""
    pos = F.normalize(pos_prototypes.float(), dim=-1)
    neg = F.normalize(neg_prototypes.float(), dim=-1)
    cos = (pos * neg).sum(-1)
    return F.relu(cos - float(max_cosine)).mean()


def blend_attribute_probabilities(
    fused_probs: torch.Tensor,
    prototype_probs: torch.Tensor | None,
    prototype_mix: float,
) -> torch.Tensor:
    if prototype_probs is None:
        return fused_probs
    mix = float(max(0.0, min(0.95, prototype_mix)))
    if mix <= 0:
        return fused_probs
    return (1.0 - mix) * fused_probs + mix * prototype_probs
