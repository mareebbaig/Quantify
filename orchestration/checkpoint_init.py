"""
checkpoint_init.py — Initialise a freshly-built model's weights.

Extracted verbatim (behaviour-preserving) from examples/train_imagenet_qat.py so
build_run can reach it without importing an example script. The original names
are kept as aliases in that module, so
``from examples.train_imagenet_qat import _load_ptq_checkpoint`` still works
(tests/test_load_ptq_checkpoint_bn_fusion.py relies on it).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.bn_fusion import fuse_bn_into_conv


def load_pretrained_weights(model: nn.Module, model_name: str) -> nn.Module:
    """Load timm float weights into a freshly built quantized model.

    The timm id for each registered model lives in the registry
    (ModelEntry.timm_name); models without one are left untouched.
    """
    import timm

    from .registry import get_model
    from utils.weight_mapping import load_timm_weights

    timm_name = get_model(model_name).timm_name
    if timm_name is None:
        print(f"[pretrained] No timm weights configured for {model_name}, skipping.")
        return model
    print(f"[pretrained] Loading timm {timm_name} …")
    float_model = timm.create_model(timm_name, pretrained=True)
    float_model.eval()
    return load_timm_weights(model, float_model, model_name)


def load_ptq_checkpoint(model: nn.Module, ckpt_path: str) -> tuple[nn.Module, bool]:
    """
    Load a checkpoint produced by examples/find_perfect_lsbs_imagenet_ptq.py,
    typically the activations-mode run chained from a weights-mode run via
    that script's --init-from-ckpt so both roles are calibrated.

    Uses strict=False and does NOT reset calibration buffers — search_done /
    search_result_lsb / annealing_alpha are loaded as-is so the PTQ-found
    LSBs are what QAT starts from. Missing/unexpected keys are reported but
    not fatal: a checkpoint produced with a different --mode / model variant
    than the one being constructed here will legitimately have mismatched
    quantizer buffers for the role that wasn't searched.

    If the checkpoint was produced with --fuse-bn (or was itself a QAT
    checkpoint saved from such a run), its model_state_dict has BatchNorm
    folded into the preceding conv/linear (conv gained a bias, BatchNorm
    became Identity) — loading that into a freshly built model that still
    has separate, randomly-initialized BatchNorm layers would leave BatchNorm
    untrained and silently produce garbage output. Detect this via
    extra.fuse_bn and fuse this model's BatchNorm the same way before
    loading, so the module structures match.

    Returns (model, bn_fused) so the caller can propagate fuse_bn=True into
    subsequent checkpoint saves, allowing further chained runs to work.
    """
    print(f"[init-from-ptq] Loading {ckpt_path} …")
    payload = torch.load(ckpt_path, map_location="cpu")
    bn_fused = False
    if payload.get("extra", {}).get("fuse_bn"):
        n_fused = fuse_bn_into_conv(model)
        bn_fused = True
        print(f"[init-from-ptq] Checkpoint was produced with --fuse-bn; fused "
              f"{n_fused} BatchNorm layer(s) into preceding conv/linear weights "
              f"to match its module structure.")
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    if incompatible.missing_keys:
        print(f"[init-from-ptq] Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"[init-from-ptq] Unexpected keys: {incompatible.unexpected_keys}")
    metrics = payload.get("metrics", {})
    if metrics:
        print(f"[init-from-ptq] Checkpoint metrics: {metrics}")
    return model, bn_fused
