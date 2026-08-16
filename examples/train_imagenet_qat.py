"""
train_imagenet_qat.py — ImageNet QAT with the V2 harness.

Supports ResNet-18, ResNet-50, MobileNetV1, MobileNetV2 with configurable
fixed-point or coefficient-based weight quantization.

Dataset is loaded from Hugging Face (ILSVRC/imagenet-1k by default).

Usage examples
--------------
# ResNet-18, 8-bit weights + activations, load torchvision pretrained:
python examples/train_imagenet_qat.py \\
    --model resnet18 --act-bits 8 --weight-bits 8 --bias-bits 8 --pretrained

# ResNet-50, coefficient weights from a file:
python examples/train_imagenet_qat.py \\
    --model resnet50 \\
    --act-bits 8 --weight-coeffs /path/to/coefficients.txt --bias-bits 8 --pretrained

# MobileNetV2 4-bit:
python examples/train_imagenet_qat.py \\
    --model mobilenetv2 --act-bits 4 --weight-bits 4 --bias-bits 8 --pretrained
"""

from __future__ import annotations

import argparse
import os  # still used for os.path.splitext in experiment name
import warnings

warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from quantizers.manager import QuantizerManager
from orchestration.checkpoint_init import (
    load_pretrained_weights,
    load_ptq_checkpoint as _load_ptq_checkpoint,
)
from orchestration.registry import AUGMENTATIONS, get_model
from orchestration.run_builder import build_run, make_injectors
from orchestration.run_spec import LRScheduleSpec, QuantSpec, RunSpec
from training_harness.config_v2 import TrainerConfigV2, QATScheduleConfigV2
from training_harness.config import CheckpointConfig
from training_harness.lr_finder import find_lr
from utils.bn_fusion import fuse_bn_into_conv
from utils.run_utils import env_default, next_run_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _print_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Print every parsed argument, grouped the same way --help groups them."""
    print(f"\n{'='*70}")
    print("  train_imagenet_qat.py — arguments")
    print(f"{'='*70}")

    seen: set = set()
    for group in parser._action_groups:
        rows = []
        for action in group._group_actions:
            # --flag/--no-flag pairs (e.g. --mixed-precision/--no-mixed-precision)
            # share one dest and both land in the same group; only list it once.
            if action.dest == "help" or action.dest in seen:
                continue
            seen.add(action.dest)
            rows.append((action.dest, getattr(args, action.dest)))
        if not rows:
            continue

        print(f"\n  [{group.title}]")
        width = max(len(dest) for dest, _ in rows)
        for dest, value in rows:
            print(f"    {dest:<{width}} : {value}")

    print(f"\n{'='*70}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ImageNet QAT — model and quantization selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Model -------------------------------------------------------------
    p.add_argument(
        "--model",
        choices=["resnet18", "resnet50", "mobilenetv1", "mobilenetv2"],
        default="resnet18",
        help="Network architecture",
    )
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="Load torchvision pretrained float weights before QAT "
             "(supported for resnet18, resnet50, mobilenetv2)",
    )

    # ---- Data --------------------------------------------------------------
    d = p.add_argument_group("data")
    d.add_argument(
        "--data-dir",
        type=str,
        default=env_default("IMAGENET_DALI_PATH"),
        metavar="PATH",
        help="Path to ImageFolder dataset (train/ and val/ subdirs). "
             "When set, uses NVIDIA DALI instead of the HuggingFace dataloader. "
             "Extract with: python scripts/extract_imagenet.py --output-dir PATH. "
             "Defaults to $IMAGENET_DALI_PATH if set.",
    )
    d.add_argument(
        "--hf-dataset",
        type=str,
        default="ILSVRC/imagenet-1k",
        help="Hugging Face dataset name (ignored when --data-dir is set)",
    )
    d.add_argument(
        "--num-workers", type=int, default=20,
        help="HuggingFace DataLoader workers (ignored when --data-dir is set)",
    )
    d.add_argument(
        "--dali-threads", type=int, default=4,
        help="DALI CPU preprocessing threads (used when --data-dir is set). "
             "DALI offloads most work to GPU so 4 is usually enough.",
    )
    d.add_argument("--randaugment-n", type=int, default=2,
                   help="Number of RandAugment transforms per image")
    d.add_argument("--randaugment-m", type=int, default=7,
                   help="RandAugment magnitude")

    # ---- Quantization ------------------------------------------------------
    q = p.add_argument_group("quantization")
    q.add_argument("--act-bits", type=int, default=8, help="Activation bit width")

    wq = q.add_mutually_exclusive_group()
    wq.add_argument(
        "--weight-bits",
        type=int,
        default=8,
        help="Weight bit width — uses FixedPointPerTensorWeightQuant",
    )
    wq.add_argument(
        "--weight-coeffs",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to coefficient file — uses CoefficientPerTensorWeightQuant "
             "(mutually exclusive with --weight-bits)",
    )
    q.add_argument("--bias-bits", type=int, default=8, help="Bias bit width")
    q.add_argument(
        "--weight-lsb-subtract",
        type=int,
        default=0,
        metavar="N",
        help="After loading --init-from-ptq, subtract N from every weight quantizer's "
             "LSB position (finer grid). Implicitly disables all activation quantizers. "
             "A before/after table is printed as a sanity check.",
    )

    # ---- Training ----------------------------------------------------------
    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Enable AMP (autocast + GradScaler). Disable if Brevitas fake-quant "
             "ops cause NaN losses during QAT (use --no-mixed-precision).",
    )
    t.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    t.add_argument(
        "--prefetch-factor",
        type=int,
        default=3,
        help="DataLoader prefetch factor (batches queued per worker ahead of GPU)",
    )
    t.add_argument(
        "--mixup",
        type=float,
        default=0.1,
        help="MixUp Beta distribution alpha (0 = off)",
    )
    t.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="CutMix Beta distribution alpha (0 = off)",
    )
    t.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of applying mixup or cutmix per batch",
    )
    t.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix are enabled",
    )
    t.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Label smoothing: via Mixup when active, else LabelSmoothingCE (0 = off)",
    )
    t.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        help="Random Erasing probability (0 = off)",
    )
    t.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="EMA decay for shadow model (validation uses EMA weights). Set 0 to disable.",
    )
    t.add_argument(
        "--repeat-aug",
        type=int,
        default=1,
        metavar="N",
        help="Repeated augmentation: each image appears N times per epoch with different "
             "augmentations (HuggingFace dataloader only; N=1 = off).",
    )

    # ---- QAT schedule ------------------------------------------------------
    s = p.add_argument_group("qat schedule")
    s.add_argument(
        "--float-warmup-epochs",
        type=int,
        default=30,
        help="Epochs of float-only training. QAT starts when val_loss plateaus "
             "for --plateau-patience epochs OR this epoch limit is reached.",
    )
    s.add_argument(
        "--plateau-patience",
        type=int,
        default=10,
        help="Epochs of no val_loss improvement before QAT is triggered",
    )
    s.add_argument(
        "--annealing-steps",
        type=int,
        default=20,
        help="Forward passes over which each quantizer anneals 0→1",
    )
    s.add_argument(
        "--qat-gap",
        type=int,
        default=300,
        help="Forward passes between successive quantizer activations (staggered cascade)",
    )

    # ---- Output ------------------------------------------------------------
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to output/imagenet_qat_<model> "
             "(e.g. output/imagenet_qat_resnet18).",
    )
    p.add_argument(
        "--new-run-dir",
        action="store_true",
        help="Auto-increment the output directory if it already exists "
             "(output/imagenet_qat_<model> → output/imagenet_qat_<model>_1 → …). "
             "Useful for keeping each run's checkpoints separate.",
    )
    p.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override the auto-generated experiment name",
    )

    # ---- Dry-run (for quick config validation) -----------------------------
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only --dry-run-batches batches per epoch (for fast config testing)",
    )
    p.add_argument(
        "--dry-run-batches",
        type=int,
        default=10,
        help="Number of batches per epoch when --dry-run is active",
    )

    # ---- LR Finder ---------------------------------------------------------
    lr = p.add_argument_group("lr finder")
    lr.add_argument(
        "--find-lr",
        action="store_true",
        help=(
            "Run the two-phase LR Range Test instead of normal training. "
            "Uses whichever model weights are active at startup "
            "(--pretrained or default init)."
        ),
    )
    lr.add_argument(
        "--find-lr-sweep-start", type=float, default=1e-8,
        help="Start of the LR sweep (default: 1e-8)",
    )
    lr.add_argument(
        "--find-lr-sweep-end", type=float, default=1e-2,
        help="End of the LR sweep (default: 1e-2)",
    )
    lr.add_argument(
        "--find-lr-steps", type=int, default=100,
        help="Number of steps in the LR sweep (default: 100)",
    )
    lr.add_argument(
        "--find-lr-calib-steps", type=int, default=10,
        help="Calibration pre-pass steps in Phase 1 (default: 10)",
    )

    # ---- Reduce LR on plateau -----------------------------------------------
    rlr = p.add_argument_group("reduce lr on plateau")
    rlr.add_argument(
        "--cosine-lr",
        action="store_true",
        default=False,
        help="Use a per-step linear-warmup + cosine-annealing LR schedule "
             "instead of ReduceLROnPlateau. Mutually exclusive with the "
             "--reduce-lr-* options: cosine steps every batch and would "
             "overwrite any plateau-triggered reduction, so ReduceLROnPlateau "
             "is disabled when this is set.",
    )
    rlr.add_argument(
        "--cosine-warmup-frac", type=float, default=0.1,
        help="Fraction of total steps spent in linear warmup (--cosine-lr only)",
    )
    rlr.add_argument(
        "--cosine-eta-min", type=float, default=1e-6,
        help="Final LR at the end of cosine annealing (--cosine-lr only)",
    )
    rlr.add_argument("--reduce-lr-patience", type=int, default=20,
                     help="Epochs of no improvement before reducing LR (default: 5)")
    rlr.add_argument("--reduce-lr-factor", type=float, default=0.5,
                     help="Multiplicative factor applied to LR on plateau (default: 0.5)")
    rlr.add_argument("--reduce-lr-min-lr", type=float, default=1e-8,
                     help="Lower bound on LR (default: 1e-8)")
    rlr.add_argument(
        "--reduce-lr-metric", type=str, default="val_loss",
        help="Metric monitored by ReduceLROnPlateau (e.g. val_loss, val_acc)",
    )

    # ---- Init from a PTQ checkpoint -----------------------------------------
    ptq = p.add_argument_group("ptq init")
    ptq.add_argument(
        "--init-from-ptq",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a checkpoint produced by "
            "examples/find_perfect_lsbs_imagenet_ptq.py — typically the "
            "activations-mode run, chained from a weights-mode run via that "
            "script's --init-from-ckpt so both roles are calibrated. Loaded "
            "with strict=False after model construction (and after "
            "--pretrained, if both are given — the checkpoint's weights win). "
            "Automatically sets preserve_calibrated_quantizers=True so the "
            "PTQ-found LSBs survive the float-warmup -> QAT transition instead "
            "of being reset and re-derived from scratch. Consider pairing with "
            "--float-warmup-epochs 0 since the model is already calibrated."
        ),
    )

    # ---- Pretrained + in-process LSB search --------------------------------
    pq = p.add_argument_group("pretrained qat")
    pq.add_argument(
        "--pretrained-qat",
        action="store_true",
        help=(
            "Load pretrained float weights, fold BatchNorm into the preceding "
            "conv/linear weights, run the full PTQ LSB search (weights -> bias "
            "-> activations, the same greedy per-quantizer search as "
            "examples/find_perfect_lsbs_imagenet_ptq.py) in-process, then start "
            "QAT from the calibrated model. BatchNorm is always fused (QAT "
            "trains the BN-folded deployment graph), so no separate flag is "
            "needed. The searched model is cached to --pretrained-qat-cache so "
            "the (slow) search runs only once; later runs with the same "
            "model/bits/radius reuse it. Mutually exclusive with "
            "--init-from-ptq. Consider pairing with --float-warmup-epochs 0 "
            "since the model is already calibrated."
        ),
    )
    pq.add_argument(
        "--ptq-search-radius",
        type=int,
        default=7,
        help="LSB positions tested on each side of the calibrated value during "
             "the --pretrained-qat search (0 = calibrated position only).",
    )
    pq.add_argument(
        "--ptq-eval-batches",
        type=int,
        default=None,
        help="Validation batches per LSB candidate during the --pretrained-qat "
             "search (None = full validation set; smaller = faster search).",
    )
    pq.add_argument(
        "--pretrained-qat-cache",
        type=str,
        default=None,
        metavar="PATH",
        help="Where the --pretrained-qat LSB-searched checkpoint is cached. "
             "Defaults to output/pretrained_qat_cache/<model>_W<w>_A<a>_B<b>"
             "_r<radius>.pt.",
    )
    pq.add_argument(
        "--force-lsb-search",
        action="store_true",
        help="Re-run the --pretrained-qat LSB search even if a cached "
             "checkpoint already exists (overwrites it).",
    )

    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"output/imagenet_qat_{args.model}"

    if args.pretrained_qat and args.init_from_ptq:
        p.error("--pretrained-qat and --init-from-ptq are mutually exclusive: "
                "the former runs the LSB search itself, the latter loads one.")
    if args.pretrained_qat and args.weight_coeffs:
        p.error("--pretrained-qat requires fixed-point weights (--weight-bits); "
                "the LSB search does not apply to coefficient weights.")

    _print_args(p, args)
    return args


# ---------------------------------------------------------------------------
# Quantizer factories
# ---------------------------------------------------------------------------

# These wrap orchestration.run_builder.make_injectors so the argparse-shaped
# call sites (and any external caller) keep working. The construction itself
# now lives in one place, driven by a QuantSpec.

def _make_weight_quant(args: argparse.Namespace):
    return make_injectors(_quant_spec_from_args(args))[0]


def _make_act_quant(args: argparse.Namespace):
    return make_injectors(_quant_spec_from_args(args))[1]


def _make_bias_quant(args: argparse.Namespace):
    return make_injectors(_quant_spec_from_args(args))[2]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(args, weight_quant, act_quant, bias_quant) -> nn.Module:
    """Build the named architecture. Delegates to the model registry, which is
    now the single place mapping a model name to a constructor."""
    return get_model(args.model).build(
        args.num_classes, (weight_quant, act_quant, bias_quant)
    )


def _load_pretrained(model: nn.Module, args) -> nn.Module:
    """Load timm float weights. The name → timm id mapping moved into the
    model registry (ModelEntry.timm_name)."""
    return load_pretrained_weights(model, args.model)


def _default_pretrained_qat_cache(args) -> str:
    """Deterministic cache path for the --pretrained-qat LSB-searched model.

    Keyed on everything that changes the search result (model, per-role bit
    widths, search radius) so different configs never collide and a matching
    config reuses the same file across runs. BatchNorm is always fused, so it
    isn't part of the key.
    """
    tag = f"{args.model}_W{args.weight_bits}_A{args.act_bits}_B{args.bias_bits}"
    tag += f"_r{args.ptq_search_radius}"
    return os.path.join("output", "pretrained_qat_cache", f"{tag}.pt")


def _run_pretrained_qat_search(args, model: nn.Module, device, train_loader, val_loader) -> None:
    """Run the weights -> bias -> activations LSB search in-process on `model`.

    Mirrors examples/find_perfect_lsbs_imagenet_ptq.py, but drives all three
    roles over a single in-memory model (each role searched with the previously
    searched roles kept active) instead of chaining checkpoints across three
    separate processes. Leaves every quantizer calibrated (search_done=True,
    annealing_alpha=1) so QAT can start immediately with
    preserve_calibrated_quantizers=True.
    """
    from pathlib import Path
    from examples.find_perfect_lsbs_imagenet_ptq import (
        _assign_descriptive_ids,
        _evaluate,
        search_role_lsbs,
    )

    out_dir = Path(args.output_dir) / "ptq_lsb_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ptq_search.log"

    _assign_descriptive_ids(model)
    mgr = QuantizerManager()
    mgr.quantization_start_gap = 2
    loss_fn = nn.CrossEntropyLoss()
    # Near-zero LR: weights barely move, but a training-mode forward+backward is
    # what lets each quantizer's calibration fire (see base_quantizer.forward).
    search_optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)

    # Baseline (float) pass: force every quantizer to calibrated-passthrough so
    # eval mode neither raises nor calibrates, and so this first forward pass
    # establishes each quantizer's forward-execution order for the search.
    for q in mgr.quantizers.values():
        q.search_done.fill_(True)
        q.annealing_alpha.data.fill_(0.0)
    print("[pretrained-qat] Baseline evaluation (no quantization) …")
    _evaluate(model, val_loader, loss_fn, device, args.ptq_eval_batches, label="baseline")

    # One calibration batch, reused for every quantizer (weights barely move).
    for _imgs, _lbls in train_loader:
        calib_images = _imgs.to(device)
        calib_labels = _lbls.to(device).long()
        break

    role_bits = {
        "weight":     args.weight_bits,
        "bias":       args.bias_bits,
        "activation": args.act_bits,
    }
    active_roles: set[str] = set()
    for role in ("weight", "bias", "activation"):
        if not any(q.quantizer_role == role for q in mgr.quantizers.values()):
            print(f"\n[pretrained-qat] ── LSB search: {role} — SKIPPED "
                  f"(model has no {role} quantizers)")
            continue
        print(f"\n[pretrained-qat] ── LSB search: {role} ({role_bits[role]}b) "
              f"─ keeping active: {sorted(active_roles) or 'none'}")
        search_role_lsbs(
            model=model,
            target_role=role,
            bit_width=role_bits[role],
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            search_radius=args.ptq_search_radius,
            eval_batches=args.ptq_eval_batches,
            out_dir=out_dir,
            log_path=log_path,
            calib_images=calib_images,
            calib_labels=calib_labels,
            optimizer=search_optimizer,
            active_roles=set(active_roles),
        )
        active_roles.add(role)

    print("\n[pretrained-qat] Final evaluation (all quantizers active) …")
    _evaluate(model, val_loader, loss_fn, device, args.ptq_eval_batches, label="final")


def _prepare_pretrained_qat(args, model: nn.Module, device, train_loader, val_loader) -> tuple[nn.Module, bool]:
    """Load-or-run the --pretrained-qat LSB search. Returns (model, bn_fused).

    On a cache hit, the freshly built `model` is populated from the cached
    checkpoint (BatchNorm re-fused first to match its BN-folded structure) via
    the same _load_ptq_checkpoint path used by --init-from-ptq, so the search
    is skipped entirely. Otherwise the search runs and its result is cached.
    """
    cache_path = args.pretrained_qat_cache or _default_pretrained_qat_cache(args)

    if os.path.exists(cache_path) and not args.force_lsb_search:
        print(f"[pretrained-qat] Using cached LSB-searched checkpoint: {cache_path}")
        print(f"                 (delete it or pass --force-lsb-search to re-run the search)")
        return _load_ptq_checkpoint(model, cache_path)

    if args.force_lsb_search and os.path.exists(cache_path):
        print(f"[pretrained-qat] --force-lsb-search: re-running search (overwriting {cache_path})")
    else:
        print(f"[pretrained-qat] No cache at {cache_path}; running LSB search …")

    model = _load_pretrained(model, args)
    # QAT trains the BN-folded deployment graph, so always fuse BatchNorm into
    # the preceding conv/linear weights before the search calibrates against
    # that same weight distribution.
    n_fused = fuse_bn_into_conv(model)
    bn_fused = True
    print(f"[pretrained-qat] Fused {n_fused} BatchNorm layer(s) into preceding conv/linear weights.")

    model = model.to(device)
    _run_pretrained_qat_search(args, model, device, train_loader, val_loader)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "metrics": {},
        "config": {
            "model": args.model,
            "weight_bits": args.weight_bits,
            "act_bits": args.act_bits,
            "bias_bits": args.bias_bits,
            "search_radius": args.ptq_search_radius,
        },
        "extra": {
            "pretrained_qat": True,
            "fuse_bn": bn_fused,
            "role_bit_widths": {
                "weight": args.weight_bits,
                "bias": args.bias_bits,
                "activation": args.act_bits,
            },
        },
    }
    torch.save(payload, cache_path)
    print(f"[pretrained-qat] Saved LSB-searched checkpoint: {cache_path}")
    return model, bn_fused


def _disable_act_quant_proxies(model: nn.Module) -> None:
    """Set disable_quant=True on all activation proxies (leaves weight/bias proxies alone)."""
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector, BiasQuantProxyFromInjector
    for m in model.modules():
        if hasattr(m, "disable_quant") and not isinstance(
            m, (WeightQuantProxyFromInjector, BiasQuantProxyFromInjector)
        ):
            m.disable_quant = True


def _apply_weight_lsb_subtract(model: nn.Module, delta: int) -> None:
    """
    Subtract `delta` from every weight quantizer's search_result_lsb buffer,
    then disable all activation quantizer proxies.

    Prints a before/after table for each adjusted quantizer so the caller can
    verify the shift is correct before training starts.
    """
    from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer

    col = 62
    print(f"\n[weight-lsb-subtract] Subtracting {delta} from all weight quantizer LSBs")
    print(f"  {'Module path':<{col}}  {'Before':>6}  {'After':>6}  Check")
    print(f"  {'-'*col}  {'-'*6}  {'-'*6}  -----")

    n = 0
    for name, module in model.named_modules():
        if not isinstance(module, FixedPointPerTensorQuantizer):
            continue
        if "weight_quant" not in name:
            continue

        before = int(module.search_result_lsb.item())
        after  = before - delta
        module.search_result_lsb.fill_(after)
        readback = int(module.search_result_lsb.item())
        ok = "OK" if readback == after else f"MISMATCH (got {readback})"
        print(f"  {name:<{col}}  {before:>6}  {after:>6}  {ok}")
        n += 1

    if n == 0:
        print("  WARNING: no weight quantizers found — load a PTQ checkpoint first.")
    else:
        print(f"\n  {n} quantizer(s) adjusted.")

    _disable_act_quant_proxies(model)
    print("  Activation quantizer proxies disabled for this run.\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class HFDatasetWrapper(Dataset):
    """Wraps a Hugging Face dataset for use with PyTorch DataLoader."""
    def __init__(self, hf_dataset, preprocess):
        self.hf_dataset = hf_dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        img = self.preprocess(item["image"].convert("RGB"))
        return img, item["label"]


class RepeatAugSampler(torch.utils.data.Sampler):
    """
    Repeated augmentation sampler: each unique image appears n_repeats times
    in consecutive slots so that, with an appropriate batch_size, every batch
    contains batch_size // n_repeats unique images each seen n_repeats times
    with independent random augmentations.

    Only meaningful when batch_size is a multiple of n_repeats.
    """

    def __init__(self, dataset: Dataset, n_repeats: int = 2, shuffle: bool = True) -> None:
        self._n = len(dataset)
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self._n * self.n_repeats

    def __iter__(self):
        import random
        indices = list(range(self._n))
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            for _ in range(self.n_repeats):
                yield idx


# Loader construction moved to orchestration/registry.py (DATASETS["imagenet"]),
# so there is one DALI call site instead of two drifting copies. The
# HuggingFace path was already dead (it raised "Deprecated. Use dali instead.");
# a run without --data-dir now fails in resolve_data_dir with a message naming
# both --data-dir and $IMAGENET_DALI_PATH.


# ---------------------------------------------------------------------------
# argparse → RunSpec adapter
# ---------------------------------------------------------------------------

def _quant_spec_from_args(args: argparse.Namespace) -> QuantSpec:
    """Map the quantization flags onto a QuantSpec.

    --weight-bits and --weight-coeffs are a mutually exclusive argparse group,
    but --weight-bits keeps its default (8) in the namespace even when
    --weight-coeffs was passed. QuantSpec models the exclusivity properly, so
    weight_bits is cleared when coefficients win — matching the old
    _make_weight_quant, which checked weight_coeffs first.
    """
    using_coeffs = bool(args.weight_coeffs)
    return QuantSpec(
        weight_bits=None if using_coeffs else args.weight_bits,
        weight_coeffs=args.weight_coeffs if using_coeffs else None,
        act_bits=args.act_bits,
        bias_bits=args.bias_bits,
        weight_lsb_subtract=args.weight_lsb_subtract,
    )


def spec_from_args(args: argparse.Namespace) -> RunSpec:
    """Build the RunSpec this command line describes.

    Every value here comes straight from a flag — the CLI's defaults and
    behaviour are unchanged; they are just expressed as data now. Kept as a
    separate function so the regression test can assert, without launching a
    run, that a given command line still produces the config the old
    imperative main() built.
    """
    output_dir = next_run_dir(args.output_dir) if args.new_run_dir else args.output_dir

    quant = _quant_spec_from_args(args)
    weight_desc = quant.describe()
    exp_name = args.experiment_name or \
        f"{args.model}_{weight_desc}_A{args.act_bits}_B{args.bias_bits}"

    # Only record deviations from the preset, so a default command line
    # serializes as a clean "imagenet_default" with no overrides.
    preset = "imagenet_default"
    preset_defaults = AUGMENTATIONS[preset].params
    overrides = {
        key: value
        for key, value in (("randaugment_n", args.randaugment_n),
                           ("randaugment_m", args.randaugment_m))
        if value != preset_defaults[key]
    }

    training = TrainerConfigV2(
        experiment_name=exp_name,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=1.0,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,

        dry_run=args.dry_run,
        dry_run_batches=args.dry_run_batches,

        qat=QATScheduleConfigV2(
            float_warmup_epochs=args.float_warmup_epochs,
            plateau_metric="val_loss",
            plateau_patience=args.plateau_patience,
            plateau_min_delta=1e-4,
            annealing_steps=args.annealing_steps,
            quantization_start_gap=args.qat_gap,
            freeze_bn_at_qat=True,
            track_scale_factors=True,
            preserve_calibrated_quantizers=bool(args.init_from_ptq or args.pretrained_qat),
        ),

        checkpoint=CheckpointConfig(
            monitor_metric="val_acc",
            monitor_mode="max",
            top_k=3,
            save_last=True,
        ),

        early_stopping_patience=None,
        reduce_lr_on_plateau=not args.cosine_lr,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_min_lr=args.reduce_lr_min_lr,
        reduce_lr_metric=args.reduce_lr_metric,

        mixup=args.mixup,
        cutmix=args.cutmix,
        mixup_prob=args.mixup_prob,
        mixup_switch_prob=args.mixup_switch_prob,
        smoothing=args.smoothing,
        reprob=args.reprob,
        num_classes=args.num_classes,
        ema_decay=args.ema_decay,
    )

    return RunSpec(
        model=args.model,
        dataset="imagenet",
        augmentation=preset,
        augmentation_overrides=overrides,
        experiment_name=exp_name,
        output_dir=output_dir,
        data_dir=args.data_dir,
        dali_threads=args.dali_threads,
        init_checkpoint=args.init_from_ptq,
        pretrained=args.pretrained,
        pretrained_qat=args.pretrained_qat,
        pretrained_qat_cache=args.pretrained_qat_cache,
        ptq_search_radius=args.ptq_search_radius,
        ptq_eval_batches=args.ptq_eval_batches,
        force_lsb_search=args.force_lsb_search,
        quant=quant,
        optimizer="adamw",
        lr_schedule=LRScheduleSpec(
            kind="cosine" if args.cosine_lr else "plateau",
            cosine_warmup_frac=args.cosine_warmup_frac,
            cosine_eta_min=args.cosine_eta_min,
        ),
        training=training,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Thin adapter: argv → RunSpec → build_run → fit.

    Every flag, default and behaviour is unchanged; the object construction
    that used to live here now lives in orchestration.run_builder.build_run,
    so the CLI and a programmatic launcher build runs the same way.
    """
    args = parse_args()

    spec = spec_from_args(args)
    handle = build_run(spec, tee_stdout=True)

    # ── LR Finder mode ───────────────────────────────────────────────────────
    if args.find_lr:
        find_lr(
            model=handle.model,
            optimizer=handle.optimizer,
            train_loader=handle.train_loader,
            loss_fn=nn.CrossEntropyLoss(),
            device="auto",
            calibration_steps=args.find_lr_calib_steps,
            sweep_start_lr=args.find_lr_sweep_start,
            sweep_end_lr=args.find_lr_sweep_end,
            sweep_steps=args.find_lr_steps,
            out_dir=os.path.join(handle.run_dir, "lr_finder"),
            grad_clip_norm=1.0,
        )
        return
    # ─────────────────────────────────────────────────────────────────────────

    trainer = handle.trainer

    print("\nPre-training evaluation (eval mode, quantization disabled):")
    trainer.evaluate(handle.val_loader,   label="val  ")
    trainer.evaluate(handle.train_loader, label="train")
    print()

    # When --weight-lsb-subtract is active, re-disable activation proxies after
    # every epoch so QAT activation (which re-enables all proxies) can't undo it.
    epoch_hook = None
    if args.weight_lsb_subtract:
        def epoch_hook(trainer, epoch, snap):
            _disable_act_quant_proxies(trainer.model)

    tracker = handle.fit(after_epoch_hook=epoch_hook)

    best_acc = tracker.best_value("val_acc", "max")
    print(f"\nDone. Best val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
