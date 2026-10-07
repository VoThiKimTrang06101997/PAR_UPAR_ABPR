"""
UPAR Challenge 2027 Track 2 — model-backed ABPR inference adapter.

The organizer imports this file and calls rank_gallery(sample).  Unlike the
official sample baseline, this adapter actually loads assets/model.pt and uses
the gallery pixels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np

from abpr_runtime import ABPRRuntime


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "assets" / "model.pt"

_RUNTIME: ABPRRuntime | None = None


def _key(name: str) -> str:
    """Normalize attribute spelling enough to survive punctuation/case changes."""
    s = str(name).strip().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "", s)


def load_model() -> None:
    global _RUNTIME
    if _RUNTIME is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model asset: {MODEL_PATH}")
        _RUNTIME = ABPRRuntime(MODEL_PATH)


def _model_indices_for_incoming(attribute_names: list[str]) -> list[int]:
    """
    For each incoming challenge attribute, return its position in model output.
    """
    assert _RUNTIME is not None
    model_map = {_key(n): i for i, n in enumerate(_RUNTIME.attribute_names)}
    indices = []
    missing = []
    for name in attribute_names:
        k = _key(name)
        if k not in model_map:
            missing.append(name)
        else:
            indices.append(model_map[k])
    if missing:
        raise ValueError(
            "Challenge attribute names do not match the trained model. "
            f"Missing mappings: {missing}"
        )
    if len(indices) != len(attribute_names):
        raise RuntimeError("Attribute mapping produced wrong number of columns.")
    return indices


def _queries_to_model_order(
    queries: np.ndarray,
    attribute_names: list[str],
) -> np.ndarray:
    """
    Organizer queries are ordered by sample['attribute_names']; reorder them to
    the 40-column order used while training the model.
    """
    assert _RUNTIME is not None
    incoming_map = {_key(n): i for i, n in enumerate(attribute_names)}
    cols = []
    missing = []
    for model_name in _RUNTIME.attribute_names:
        k = _key(model_name)
        if k not in incoming_map:
            missing.append(model_name)
        else:
            cols.append(incoming_map[k])
    if missing:
        raise ValueError(
            "Cannot reorder challenge queries into model attribute order. "
            f"Missing: {missing}"
        )
    return np.asarray(queries, dtype=np.float32)[:, cols]


def predict_attributes(
    gallery: list[dict[str, Any]],
    attribute_names: list[str],
) -> np.ndarray:
    """
    Predict gallery attribute probabilities in the exact order requested by the
    organizer.  This helper mirrors the official starter signature.
    """
    load_model()
    assert _RUNTIME is not None
    probs, _ = _RUNTIME.encode_gallery(gallery)
    model_cols = _model_indices_for_incoming(attribute_names)
    return probs[:, model_cols].numpy().astype(np.float32, copy=False)


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Return a [num_queries, num_gallery] float32 distance matrix.
    Smaller is better, exactly as required by the official Track-2 starter.
    """
    load_model()
    assert _RUNTIME is not None

    gallery = sample["gallery"]
    incoming_names = list(sample["attribute_names"])
    queries_in = np.asarray(sample["queries"], dtype=np.float32)

    if queries_in.ndim != 2:
        raise ValueError(f"Expected 2-D queries, got shape={queries_in.shape}")
    if queries_in.shape[1] != len(incoming_names):
        raise ValueError(
            f"Query width {queries_in.shape[1]} != "
            f"len(attribute_names) {len(incoming_names)}"
        )

    queries_model = _queries_to_model_order(queries_in, incoming_names)
    probs, emb = _RUNTIME.encode_gallery(gallery)
    distances = _RUNTIME.distance(
        queries_model,
        probs,
        emb,
    ).numpy().astype(np.float32, copy=False)

    expected = (len(queries_in), len(gallery))
    if distances.shape != expected:
        raise RuntimeError(
            f"Bad distance shape {distances.shape}; expected {expected}"
        )
    if not np.isfinite(distances).all():
        raise RuntimeError("Non-finite distances produced by submission model.")

    return {"distances": distances}
