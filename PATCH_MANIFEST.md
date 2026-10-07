# Prototype Retrieval Patch Manifest

This patch adds positive/negative visual attribute prototypes to the canonical, non-versioned Track-2 ABPR pipeline.

## Add

- `abpr/prototype.py`
  - `AttributePrototypeHead`
  - masked prototype BCE
  - cross-domain prototype alignment
  - positive/negative prototype separation
  - fused/prototype probability blending
- `notebooks/PAR_UPAR_ABPR_Track2_Prototype.ipynb`

## Replace / modify

- `abpr/model.py`
  - adds a fourth prototype branch to the existing global/spatial/stripe fusion
  - exposes attribute-specific prototype features/logits
  - adds prototype loss weights to `TrainConfig`
- `abpr/runtime.py`
  - mirrors the prototype architecture inside the standalone Codabench runtime
  - applies the selected validation `prototype_mix`
- `abpr/__init__.py`
  - exports prototype APIs
- `train_abpr.py`
  - direct prototype BCE supervision
  - cross-domain alignment to shared positive/negative prototypes
  - prototype separation regularization
  - warm-up/ramp for prototype-specific losses
  - checkpoint selection searches prototype/fused probability blends
  - Google Drive resume/checkpoint behavior retained
- `calibrate_retrieval.py`
  - jointly calibrates prototype mix + per-attribute affine calibration + retrieval distance settings
- `evaluate_abpr.py`
  - evaluates the selected prototype probability blend
- `export_abpr_runtime.py`
  - stores `prototype_mix` in the final runtime package
- `inspect_checkpoint.py`
  - reports prototype mix
- `tests/test_abpr.py`
  - prototype forward/backward, domain alignment and probability blend tests
- `README.md`

## Unchanged core scripts but still part of the repo

- `predict_task2.py`
- `score_task2_local.py`
- `submission_run.py`
- `build_codabench_submission.py`
- `abpr/core.py`
- `requirements.txt`

## Checkpoint naming

Prototype runs use separate filenames such as:

```text
abpr_prototype_convnext_small_seed42_last.pt
abpr_prototype_convnext_small_seed42_best.pt
```

This avoids accidentally loading an incompatible older non-prototype checkpoint while preserving automatic resume for prototype runs.

## Important

A 0.60+ Codabench mADM is a target, not a guaranteed result. The patch is designed to make the model better aligned with sparse attribute retrieval and hidden-domain generalization; final hidden-test performance must still be measured by the competition scorer.
