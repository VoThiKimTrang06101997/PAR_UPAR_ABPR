# PAR_UPAR_ABPR — Prototype-Enhanced Track-2 Retrieval

This repository trains an attribute-based person retrieval model for UPAR Track 2. The current method keeps the strong retrieval-aware pipeline (ConvNeXt, sparse-query training, Degree-of-Match losses, calibration, ensemble/TTA) and adds **positive/negative visual prototypes for every one of the 40 attributes**.

## Method

For each pedestrian attribute `a`, the model learns two normalized prototype vectors:

```text
P[a,+]  positive prototype
P[a,-]  negative prototype
```

The backbone feature map is converted into an attribute-specific feature `h_a` using spatial attention plus a global feature branch. Prototype evidence is computed with a cosine margin:

```text
prototype_logit[a]
  = scale * ( cos(h_a, P[a,+]) - cos(h_a, P[a,-]) )
```

The final attribute logit is a learned per-attribute fusion of four branches:

```text
global + spatial-attention + body-stripe + prototype-margin
```

The prototype branch is not forced to dominate. It starts with a lower fusion prior, and validation calibration later searches how much prototype probability should contribute to Track-2 ranking.

## Prototype losses

Training adds three prototype-specific objectives:

1. **Prototype BCE** — the positive/negative margin itself must predict the attribute correctly.
2. **Cross-domain prototype alignment** — domain-specific positive/negative feature centroids from Market1501, PA-100K and PETA are pulled toward the same shared prototypes.
3. **Prototype separation** — positive and negative prototypes for the same attribute are prevented from collapsing.

Prototype loss weights ramp up during the early part of training to avoid noisy prototypes destabilizing the pretrained backbone.

## Retrieval-aware training retained

The training loop still includes:

- focal multi-label classification loss;
- sparse query augmentation (`-1 = unknown`);
- probability Degree-of-Match supervision;
- listwise Degree-of-Match ranking loss;
- image/query embedding alignment;
- domain adversarial regularization;
- group/ontology consistency;
- EMA model;
- domain-balanced sampling.

## Prototype-aware calibration

After training, `calibrate_retrieval.py` evaluates mixtures such as:

```text
0% prototype / 100% fused probabilities
15% prototype / 85% fused probabilities
30% prototype / 70% fused probabilities
45% prototype / 55% fused probabilities
60% prototype / 40% fused probabilities
```

For each candidate it also searches retrieval distance, reliability weighting, rarity weighting, negative-query weighting and optional embedding fusion. The selected `prototype_mix` is stored inside the exported runtime package and used identically by local validation and Codabench inference.

## Checkpoint and resume

Prototype runs are isolated from older incompatible checkpoints:

```text
/content/drive/MyDrive/PedestrianAttributeRecognition/ABPR_Checkpoints/
  abpr_prototype_convnext_small_seed42_last.pt
  abpr_prototype_convnext_small_seed42_best.pt
```

`last.pt` includes model, EMA model, optimizer, scheduler, AMP scaler, epoch/global step and RNG state. The training script automatically resumes when the matching prototype checkpoint exists.

CSV history is stored under:

```text
/content/drive/MyDrive/PedestrianAttributeRecognition/ABPR_Results/
```

## Colab

Use:

```text
notebooks/PAR_UPAR_ABPR_Track2_Prototype.ipynb
```

The notebook clones the GitHub repository and runs directly from source. It contains no `%%writefile` source-generation cells.

## Validation discipline

Calibration is pinned to the organizer's Track-2 **validation** gallery/query pair rather than a recursively discovered training pair. Before submission, score the **exported runtime** locally; that is the closest check of what Codabench will execute.

## Score expectation

The prototype method is designed to improve semantic stability under domain shift and retrieval matching, but no source-code change can guarantee a hidden-test mADM above 0.60. Use validation mADM, mAP, Rank-1 and mINP to decide whether the prototype branch is genuinely helping before submitting.
