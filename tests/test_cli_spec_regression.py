"""
test_cli_spec_regression.py — the train_imagenet_qat CLI must not drift.

main() is now a thin argparse → RunSpec → build_run adapter. The risk that
introduces is silent behaviour drift: a flag that used to reach the trainer
quietly stops doing so, and a full-scale run trains with the wrong settings for
hours before anyone notices.

This test pins that down. ``_legacy_config_from_args`` is a VERBATIM copy of
the TrainerConfigV2 construction that lived in main() before the refactor
(train_imagenet_qat.py:933-982 at commit 6b2827b). For a range of command
lines, the config the adapter produces is compared to it field by field —
including every nested sub-config. If someone edits the adapter and changes an
effective default, this fails.

Nothing here launches a run: the comparison is pure config construction, so it
needs no ImageNet, no DALI, and no GPU.
"""

import dataclasses
import sys

import pytest

from examples.train_imagenet_qat import parse_args, spec_from_args
from orchestration.run_spec import RunSpec
from training_harness.config import CheckpointConfig
from training_harness.config_v2 import QATScheduleConfigV2, TrainerConfigV2


# ---------------------------------------------------------------------------
# The pre-refactor construction, copied verbatim. Do not "clean this up" —
# its whole value is being an independent copy of the old behaviour.
# ---------------------------------------------------------------------------

def _legacy_exp_name(args):
    """From train_imagenet_qat.py:853-858 (pre-refactor)."""
    import os

    weight_desc = (
        f"coeffs_{os.path.splitext(os.path.basename(args.weight_coeffs))[0]}"
        if args.weight_coeffs
        else f"W{args.weight_bits}"
    )
    return args.experiment_name or \
        f"{args.model}_{weight_desc}_A{args.act_bits}_B{args.bias_bits}"


def _legacy_config_from_args(args) -> TrainerConfigV2:
    """From train_imagenet_qat.py:933-982 (pre-refactor)."""
    return TrainerConfigV2(
        experiment_name=_legacy_exp_name(args),
        output_dir=args.output_dir,
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


# ---------------------------------------------------------------------------
# Command lines to pin. Each exercises a different corner of the flag surface.
# ---------------------------------------------------------------------------

COMMAND_LINES = {
    "defaults": [],
    "resnet50_4bit": [
        "--model", "resnet50", "--weight-bits", "4", "--act-bits", "4",
        "--bias-bits", "8", "--pretrained",
    ],
    "mobilenetv1_full_training_flags": [
        "--model", "mobilenetv1", "--epochs", "42", "--batch-size", "512",
        "--lr", "3e-4", "--weight-decay", "5e-5", "--no-mixed-precision",
        "--mixup", "0.3", "--cutmix", "0.7", "--mixup-prob", "0.9",
        "--mixup-switch-prob", "0.4", "--smoothing", "0.05", "--reprob", "0.1",
        "--ema-decay", "0.998", "--num-classes", "1000", "--num-workers", "12",
    ],
    "cosine_schedule": [
        "--cosine-lr", "--cosine-warmup-frac", "0.05", "--cosine-eta-min", "1e-7",
    ],
    "plateau_schedule_tuned": [
        "--reduce-lr-patience", "7", "--reduce-lr-factor", "0.25",
        "--reduce-lr-min-lr", "1e-9", "--reduce-lr-metric", "val_acc",
    ],
    "qat_schedule": [
        "--float-warmup-epochs", "0", "--plateau-patience", "3",
        "--annealing-steps", "77", "--qat-gap", "11",
    ],
    "init_from_ptq": [
        "--init-from-ptq", "/tmp/ptq_calibrated_model.pt", "--float-warmup-epochs", "0",
    ],
    "pretrained_qat": [
        "--pretrained-qat", "--ptq-search-radius", "3", "--ptq-eval-batches", "5",
    ],
    "coefficient_weights": [
        "--weight-coeffs", "/tmp/coefficients.txt", "--act-bits", "6",
    ],
    "explicit_experiment_and_output": [
        "--experiment-name", "my_run", "--output-dir", "output/custom_place",
    ],
    "dry_run": ["--dry-run", "--dry-run-batches", "3"],
    "weight_lsb_subtract": [
        "--init-from-ptq", "/tmp/ptq.pt", "--weight-lsb-subtract", "2",
    ],
    "augmentation_overrides": ["--randaugment-n", "3", "--randaugment-m", "12"],
    "dali_and_data": [
        "--data-dir", "/data/imagenet", "--dali-threads", "16",
    ],
}


@pytest.fixture
def parsed(monkeypatch):
    """parse_args() reads sys.argv; hand it a command line."""
    def _parse(argv):
        monkeypatch.setattr(sys, "argv", ["train_imagenet_qat.py", *argv])
        return parse_args()
    return _parse


@pytest.mark.parametrize("name", sorted(COMMAND_LINES))
def test_adapter_builds_the_legacy_config(name, parsed):
    """Field-by-field equality with what the old imperative main() built."""
    args = parsed(COMMAND_LINES[name])

    expected = _legacy_config_from_args(args)
    actual = spec_from_args(args).training

    mismatches = []
    for field in dataclasses.fields(TrainerConfigV2):
        want = getattr(expected, field.name)
        got = getattr(actual, field.name)
        if field.name == "run_id":
            # The spec assigns a run_id up front (the old path let the trainer
            # do it); everything else must match exactly.
            assert got, "spec must carry a run_id"
            continue
        if want != got:
            mismatches.append(f"  {field.name}: legacy={want!r} adapter={got!r}")

    assert not mismatches, (
        f"CLI drift for {name!r} — the adapter no longer builds what main() did:\n"
        + "\n".join(mismatches)
    )


@pytest.mark.parametrize("name", sorted(COMMAND_LINES))
def test_adapter_produces_a_valid_serializable_spec(name, parsed):
    """Whatever the CLI accepts must also survive the manifest round-trip."""
    spec = spec_from_args(parsed(COMMAND_LINES[name]))
    assert RunSpec.from_json(spec.to_json()) == spec


# ---------------------------------------------------------------------------
# Ingredient identity — the part the old path did not express at all
# ---------------------------------------------------------------------------

def test_model_and_dataset_land_in_the_spec(parsed):
    spec = spec_from_args(parsed(["--model", "resnet50"]))
    assert spec.model == "resnet50"
    assert spec.dataset == "imagenet"
    assert spec.augmentation == "imagenet_default"


def test_default_output_dir_matches_the_historical_cli_default(parsed):
    """parse_args fills output_dir with output/imagenet_qat_<model> (:434)."""
    spec = spec_from_args(parsed(["--model", "mobilenetv2"]))
    assert spec.output_dir == "output/imagenet_qat_mobilenetv2"


def test_quant_flags_land_in_the_quant_spec(parsed):
    spec = spec_from_args(parsed([
        "--weight-bits", "4", "--act-bits", "6", "--bias-bits", "8",
    ]))
    assert spec.quant.weight_bits == 4
    assert spec.quant.act_bits == 6
    assert spec.quant.bias_bits == 8
    assert spec.quant.weight_coeffs is None


def test_coefficient_weights_clear_weight_bits(parsed):
    """--weight-bits keeps its argparse default even when --weight-coeffs wins;
    QuantSpec models the exclusivity, matching the old _make_weight_quant which
    checked weight_coeffs first."""
    spec = spec_from_args(parsed(["--weight-coeffs", "/tmp/c.txt"]))
    assert spec.quant.weight_coeffs == "/tmp/c.txt"
    assert spec.quant.weight_bits is None


def test_default_command_line_records_no_augmentation_overrides(parsed):
    """A default run must serialize as a clean preset with nothing overridden."""
    spec = spec_from_args(parsed([]))
    assert spec.augmentation_overrides == {}


def test_randaugment_flags_are_recorded_as_overrides(parsed):
    spec = spec_from_args(parsed(["--randaugment-n", "3", "--randaugment-m", "12"]))
    assert spec.augmentation_overrides == {"randaugment_n": 3, "randaugment_m": 12}
    assert spec.augmentation_params()["randaugment_m"] == 12


def test_lr_schedule_choice_is_captured(parsed):
    assert spec_from_args(parsed([])).lr_schedule.kind == "plateau"

    cosine = spec_from_args(parsed(["--cosine-lr", "--cosine-eta-min", "1e-7"]))
    assert cosine.lr_schedule.kind == "cosine"
    assert cosine.lr_schedule.cosine_eta_min == 1e-7


def test_init_checkpoint_and_pretrained_flags_are_captured(parsed):
    spec = spec_from_args(parsed(["--init-from-ptq", "/tmp/x.pt"]))
    assert spec.init_checkpoint == "/tmp/x.pt"
    assert spec.training.qat.preserve_calibrated_quantizers is True

    pre = spec_from_args(parsed(["--pretrained"]))
    assert pre.pretrained is True
    assert pre.init_checkpoint is None
