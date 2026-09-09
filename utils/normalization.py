"""
normalization.py — Per-model input normalization statistics.

Extracted from utils/dali_pipeline.py so the stats can be resolved without
importing NVIDIA DALI. The orchestration layer must be able to answer
"what normalization does this (model, dataset) pair need?" on a machine that
has no DALI installed (CPU dev boxes, CI), and dali_pipeline imports
``nvidia.dali`` at module scope.

utils/dali_pipeline.py re-exports every name defined here, so existing
``from utils.dali_pipeline import norm_for_model`` imports keep working.
This module is the single source of truth for the values.
"""

from __future__ import annotations

# Default normalization: standard ImageNet statistics (used by torchvision and
# most timm checkpoints). Some checkpoints — e.g. timm's mobilenetv1_100.ra4 —
# were trained with mean=std=0.5 instead, so mean/std are configurable per model.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)

# Normalization matching each model's pretrained timm checkpoint. Most use
# standard ImageNet stats, but timm's mobilenetv1_100.ra4 checkpoint was trained
# with inception-style mean=std=0.5 — feeding it ImageNet-normalized inputs
# collapses accuracy (~17% instead of ~73%).
_MODEL_NORM = {
    "resnet18":    (IMAGENET_MEAN, IMAGENET_STD),
    "resnet50":    (IMAGENET_MEAN, IMAGENET_STD),
    "mobilenetv1": (HALF_MEAN, HALF_STD),
    "mobilenetv2": (IMAGENET_MEAN, IMAGENET_STD),
}


def norm_for_model(arch: str):
    """Return the (mean, std) normalization matching a model's pretrained checkpoint.

    Falls back to standard ImageNet stats for unknown architectures. Callers that
    must not silently accept an unknown architecture should check membership in
    ``_MODEL_NORM`` first — see orchestration.registry.resolve_normalization,
    which refuses unknown (model, dataset) pairs instead of falling back.
    """
    return _MODEL_NORM.get(arch, (IMAGENET_MEAN, IMAGENET_STD))
