"""Model inspection helpers — parameter counts, per-layer breakdowns."""

from __future__ import annotations

from typing import Tuple

from torch import nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts for ``model``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def summarize_parameters(model: nn.Module) -> None:
    """Print a per-layer parameter table plus totals.

    Each parameter is attributed to exactly one module (the one that
    registered it), so the per-row numbers sum to the grand total.
    """
    rows = []
    for name, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own == 0:
            continue
        rows.append((name or "<root>", type(module).__name__, own))

    if not rows:
        print("Model has no parameters.")
        return

    name_w = max(len(r[0]) for r in rows + [("Module", "", 0)])
    type_w = max(len(r[1]) for r in rows + [("", "Type",  0)])
    sep_w  = name_w + type_w + 17

    print(f"{'Module':<{name_w}}  {'Type':<{type_w}}  {'Params':>12}")
    print("-" * sep_w)
    for name, cls, p in rows:
        print(f"{name:<{name_w}}  {cls:<{type_w}}  {p:>12,}")
    print("-" * sep_w)

    total, trainable = count_parameters(model)
    label_w = name_w + type_w + 2
    print(f"{'Total':<{label_w}}  {total:>12,}")
    if trainable != total:
        print(f"{'Trainable':<{label_w}}  {trainable:>12,}")