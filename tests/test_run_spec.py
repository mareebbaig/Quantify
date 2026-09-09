"""
test_run_spec.py — RunSpec validation, serialization, and normalization resolution.

These tests deliberately import nothing heavy: the registry defers torch /
brevitas / DALI imports into its builder functions, so ingredient names can be
validated on a machine with no DALI installed.
"""

import json

import pytest

from orchestration.registry import RegistryError, resolve_normalization
from orchestration.run_spec import LRScheduleSpec, QuantSpec, RunSpec, RunSpecError
from training_harness.config import CheckpointConfig
from training_harness.config_v2 import QATScheduleConfigV2, TrainerConfigV2
from utils.normalization import HALF_MEAN, HALF_STD, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Defaults and derived identity
# ---------------------------------------------------------------------------

def test_defaults_are_derived():
    spec = RunSpec(model="resnet18", dataset="imagenet")

    assert spec.augmentation == "imagenet_default"
    assert spec.run_id  # auto timestamp
    # Same naming rule the CLI has always used (train_imagenet_qat.py:858).
    assert spec.experiment_name == "resnet18_W8_A8_B8"
    # Same default output root the CLI has always used (:434).
    assert spec.output_dir == "output/imagenet_qat_resnet18"
    # Identity is mirrored into the nested training config, which the harness reads.
    assert spec.training.experiment_name == "resnet18_W8_A8_B8"
    assert spec.training.run_id == spec.run_id


def test_experiment_name_uses_coefficient_stem():
    spec = RunSpec(
        model="resnet18", dataset="imagenet",
        quant=QuantSpec(weight_bits=None, weight_coeffs="/tmp/my_coeffs.txt"),
    )
    assert spec.experiment_name == "resnet18_coeffs_my_coeffs_A8_B8"


def test_run_dir_is_per_run():
    """Each run gets its own directory, so two runs cannot share a checkpoint
    pool (the eviction bug in checkpointing.py:343,363)."""
    a = RunSpec(model="resnet18", dataset="imagenet", run_id="run-a")
    b = RunSpec(model="resnet18", dataset="imagenet", run_id="run-b")

    assert a.run_dir != b.run_dir
    assert a.run_dir.endswith("run-a")
    assert "runs" in a.run_dir


def test_dataset_supplies_num_classes():
    spec = RunSpec(model="mnist_cnn", dataset="mnist")
    assert spec.training.num_classes == 10


def test_explicit_num_classes_is_not_overridden():
    spec = RunSpec(
        model="resnet18", dataset="imagenet",
        training=TrainerConfigV2(num_classes=100),
    )
    assert spec.training.num_classes == 100


def test_lr_schedule_kind_drives_the_config_flag():
    plateau = RunSpec(model="resnet18", dataset="imagenet")
    cosine = RunSpec(model="resnet18", dataset="imagenet",
                     lr_schedule=LRScheduleSpec(kind="cosine"))

    assert plateau.training.reduce_lr_on_plateau is True
    assert cosine.training.reduce_lr_on_plateau is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_unknown_model_rejected():
    with pytest.raises(RegistryError, match="unknown model"):
        RunSpec(model="resnet99", dataset="imagenet")


def test_unknown_dataset_rejected():
    with pytest.raises(RegistryError, match="unknown dataset"):
        RunSpec(model="resnet18", dataset="cifar100")


def test_unknown_augmentation_preset_rejected():
    with pytest.raises(RegistryError, match="unknown augmentation preset"):
        RunSpec(model="resnet18", dataset="imagenet", augmentation="does_not_exist")


def test_augmentation_preset_must_match_dataset():
    with pytest.raises(RunSpecError, match="belongs to dataset"):
        RunSpec(model="resnet18", dataset="imagenet", augmentation="mnist_default")


def test_unknown_augmentation_override_rejected():
    """An override that silently does nothing is the same class of bug as a
    silently-wrong normalization."""
    with pytest.raises(RegistryError, match="no parameter"):
        RunSpec(model="resnet18", dataset="imagenet",
                augmentation_overrides={"randaugment_q": 3})


@pytest.mark.parametrize("bits", [0, -1, 33])
def test_insane_bit_widths_rejected(bits):
    with pytest.raises(RunSpecError, match="outside the supported range"):
        RunSpec(model="resnet18", dataset="imagenet", quant=QuantSpec(weight_bits=bits))


def test_weight_bits_and_coeffs_are_mutually_exclusive():
    with pytest.raises(RunSpecError, match="mutually exclusive"):
        RunSpec(model="resnet18", dataset="imagenet",
                quant=QuantSpec(weight_bits=8, weight_coeffs="c.txt"))


def test_one_weight_mode_is_required():
    with pytest.raises(RunSpecError, match="must be set"):
        RunSpec(model="resnet18", dataset="imagenet",
                quant=QuantSpec(weight_bits=None, weight_coeffs=None))


def test_pretrained_qat_and_init_checkpoint_are_mutually_exclusive():
    with pytest.raises(RunSpecError, match="mutually exclusive"):
        RunSpec(model="resnet18", dataset="imagenet",
                pretrained_qat=True, init_checkpoint="/tmp/ckpt.pt")


def test_unknown_optimizer_rejected():
    with pytest.raises(RunSpecError, match="optimizer"):
        RunSpec(model="resnet18", dataset="imagenet", optimizer="sgd")


def test_unknown_lr_schedule_rejected():
    with pytest.raises(RunSpecError, match="kind"):
        RunSpec(model="resnet18", dataset="imagenet",
                lr_schedule=LRScheduleSpec(kind="onecycle"))


# ---------------------------------------------------------------------------
# Model quantization contracts
# ---------------------------------------------------------------------------

def test_fixed_contract_model_rejects_other_bit_widths():
    """MNISTQuantNet hardcodes its quantizers; a 4-bit request would be
    silently ignored, leaving the manifest and checkpoint lying about the run."""
    with pytest.raises(RegistryError, match="hardcodes its quantizers"):
        RunSpec(model="mnist_cnn", dataset="mnist", quant=QuantSpec(weight_bits=4))


def test_fixed_contract_model_accepts_its_own_bit_widths():
    spec = RunSpec(model="mnist_cnn", dataset="mnist",
                   quant=QuantSpec(weight_bits=8, act_bits=8, bias_bits=8))
    assert spec.quant.weight_bits == 8


def test_fixed_contract_model_rejects_other_num_classes():
    with pytest.raises(RegistryError, match="hardwires num_classes"):
        RunSpec(model="mnist_cnn", dataset="mnist",
                training=TrainerConfigV2(num_classes=100))


# ---------------------------------------------------------------------------
# Normalization — the (model, dataset) pair landmine
# ---------------------------------------------------------------------------

def test_mobilenetv1_imagenet_resolves_to_half_stats():
    """timm's mobilenetv1_100.ra4 was trained with mean=std=0.5. Feeding it
    ImageNet stats collapses top-1 from ~73% to ~17% — silently."""
    mean, std = resolve_normalization("mobilenetv1", "imagenet")
    assert mean == HALF_MEAN
    assert std == HALF_STD
    assert mean != IMAGENET_MEAN


def test_resnet18_imagenet_resolves_to_imagenet_stats():
    mean, std = resolve_normalization("resnet18", "imagenet")
    assert (mean, std) == (IMAGENET_MEAN, IMAGENET_STD)


def test_mnist_resolves_to_mnist_stats():
    mean, std = resolve_normalization("mnist_cnn", "mnist")
    assert (mean, std) == ((0.1307,), (0.3081,))


def test_untested_pair_is_refused_by_default():
    with pytest.raises(RegistryError, match="untested pair"):
        resolve_normalization("resnet18", "mnist")


def test_untested_pair_warns_when_explicitly_allowed():
    with pytest.warns(RuntimeWarning, match="untested pair"):
        resolve_normalization("resnet18", "mnist", allow_untested_pair=True)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _rich_spec() -> RunSpec:
    """A spec exercising every nested structure, including the tuple-valued
    secondary_checkpoint_metrics that JSON would otherwise turn into lists."""
    return RunSpec(
        model="mobilenetv2",
        dataset="imagenet",
        augmentation="imagenet_default",
        augmentation_overrides={"randaugment_m": 9},
        run_id="2026-01-02_030405",
        experiment_name="explicit_name",
        output_dir="output/somewhere",
        data_dir="/data/imagenet",
        dali_threads=8,
        init_checkpoint="/ckpts/ptq.pt",
        pretrained=True,
        quant=QuantSpec(weight_bits=4, act_bits=6, bias_bits=8, weight_lsb_subtract=2),
        lr_schedule=LRScheduleSpec(kind="cosine", cosine_warmup_frac=0.2,
                                   cosine_eta_min=1e-7),
        training=TrainerConfigV2(
            epochs=7, batch_size=128, learning_rate=3e-4, num_classes=1000,
            api_port=0, mixup=0.2, ema_decay=0.999,
            qat=QATScheduleConfigV2(float_warmup_epochs=2, annealing_steps=50),
            checkpoint=CheckpointConfig(monitor_metric="val_acc", monitor_mode="max"),
            secondary_checkpoint_metrics=[("train_loss", "min")],
        ),
    )


def test_dict_round_trip_is_identical():
    spec = _rich_spec()
    assert RunSpec.from_dict(spec.to_dict()) == spec


def test_json_round_trip_is_identical():
    spec = _rich_spec()
    assert RunSpec.from_json(spec.to_json()) == spec


def test_to_dict_is_json_serializable():
    """The manifest and the checkpoint's extra["run_spec"] both store this dict,
    so it must survive json.dumps without a custom encoder."""
    json.dumps(_rich_spec().to_dict())


def test_file_round_trip(tmp_path):
    spec = _rich_spec()
    path = spec.write_json(str(tmp_path / "run.json"))
    assert RunSpec.read_json(path) == spec


def test_from_dict_rejects_unknown_fields():
    data = _rich_spec().to_dict()
    data["a_field_that_does_not_exist"] = 1
    with pytest.raises(RunSpecError, match="unknown RunSpec field"):
        RunSpec.from_dict(data)


def test_nested_configs_survive_round_trip_as_dataclasses():
    restored = RunSpec.from_json(_rich_spec().to_json())

    assert isinstance(restored.training, TrainerConfigV2)
    assert isinstance(restored.training.qat, QATScheduleConfigV2)
    assert isinstance(restored.training.checkpoint, CheckpointConfig)
    assert isinstance(restored.quant, QuantSpec)
    assert isinstance(restored.lr_schedule, LRScheduleSpec)
    assert restored.training.qat.float_warmup_epochs == 2
    # Tuples, not the lists JSON produced.
    assert restored.training.secondary_checkpoint_metrics == [("train_loss", "min")]


def test_augmentation_params_merge_preset_with_overrides():
    spec = RunSpec(model="resnet18", dataset="imagenet",
                   augmentation_overrides={"randaugment_m": 9})
    params = spec.augmentation_params()

    assert params["randaugment_m"] == 9   # override
    assert params["randaugment_n"] == 2   # preset default
    assert params["crop"] == 224
