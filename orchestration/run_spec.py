"""
run_spec.py — RunSpec: a training run described entirely as data.

The audit that motivated this module found that a run's *training dynamics*
were already data (TrainerConfigV2) while a run's *identity* — which model,
which dataset, which augmentation, which starting checkpoint — existed only as
argparse flags plus imperative wiring inside train_imagenet_qat.main(). RunSpec
closes that gap: everything needed to launch a run is one serializable object.

    spec = RunSpec(model="resnet18", dataset="imagenet",
                   training=TrainerConfigV2(epochs=150, batch_size=1024))
    handle = build_run(spec)
    handle.fit()

RunSpec is also the on-disk format for two things that read it back:
  * <run_dir>/run.json          — the run manifest a launcher globs
  * checkpoint payload ["extra"]["run_spec"] — checkpoint provenance

--------------------------------------------------------------------------
What is deliberately NOT in RunSpec
--------------------------------------------------------------------------
Anything TrainerConfigV2 already owns stays there — there is exactly one
source of truth per value. So learning_rate, weight_decay, batch_size, epochs,
num_classes, num_workers, the whole QAT schedule, checkpoint policy, EMA,
early stopping, mixup/cutmix/smoothing/random-erasing, and api_port live in
``spec.training``, not beside it. RunSpec adds only what TrainerConfigV2 lacks.

``optimizer`` and ``lr_schedule`` carry the *choice* of algorithm (previously
hardcoded: AdamW at train_imagenet_qat.py:895, cosine-vs-plateau at :920-931);
the numbers those algorithms consume stay in TrainerConfigV2.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from training_harness.config import CheckpointConfig, LoggingConfig
from training_harness.config_v2 import QATScheduleConfigV2, TrainerConfigV2

from .registry import (
    RegistryError,
    augmentation_names,
    dataset_names,
    get_augmentation,
    get_dataset,
    get_model,
    model_names,
    resolve_augmentation_params,
    validate_model_quant_contract,
)

# Bit widths outside this range are almost certainly a typo rather than an
# experiment; the quantizers themselves document 8-16b as the working range.
MIN_BIT_WIDTH = 1
MAX_BIT_WIDTH = 32

VALID_OPTIMIZERS = ("adamw",)
VALID_LR_SCHEDULES = ("plateau", "cosine", "none")


class RunSpecError(ValueError):
    """Raised when a RunSpec describes a run that cannot be built."""


# ---------------------------------------------------------------------------
# Sub-specs
# ---------------------------------------------------------------------------

@dataclass
class QuantSpec:
    """Quantizer configuration — the flat scalars that drive the injectors.

    ``weight_bits`` and ``weight_coeffs`` are mutually exclusive, mirroring the
    CLI's mutually-exclusive group (train_imagenet_qat.py:145-159): fixed-point
    weights at N bits, or coefficient weights read from a file.
    """

    weight_bits: Optional[int] = 8
    weight_coeffs: Optional[str] = None
    act_bits: int = 8
    bias_bits: int = 8
    weight_lsb_subtract: int = 0
    """Subtract N from every weight quantizer's LSB after loading an init
    checkpoint (finer grid). Implicitly disables activation quantizers."""

    def validate(self) -> None:
        if self.weight_coeffs is not None and self.weight_bits is not None:
            raise RunSpecError(
                "QuantSpec: weight_bits and weight_coeffs are mutually exclusive "
                "(fixed-point weights OR coefficient weights, not both). "
                "Set weight_bits=None when using weight_coeffs."
            )
        if self.weight_coeffs is None and self.weight_bits is None:
            raise RunSpecError(
                "QuantSpec: one of weight_bits or weight_coeffs must be set."
            )
        for name in ("weight_bits", "act_bits", "bias_bits"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise RunSpecError(f"QuantSpec.{name} must be an int, got {value!r}")
            if not (MIN_BIT_WIDTH <= value <= MAX_BIT_WIDTH):
                raise RunSpecError(
                    f"QuantSpec.{name}={value} is outside the supported range "
                    f"[{MIN_BIT_WIDTH}, {MAX_BIT_WIDTH}]"
                )
        if self.weight_lsb_subtract < 0:
            raise RunSpecError(
                f"QuantSpec.weight_lsb_subtract must be >= 0, "
                f"got {self.weight_lsb_subtract}"
            )

    def describe(self) -> str:
        """Short tag used to build experiment names — matches the naming rule
        at train_imagenet_qat.py:853-858 so existing run names are unchanged."""
        if self.weight_coeffs:
            stem = os.path.splitext(os.path.basename(self.weight_coeffs))[0]
            return f"coeffs_{stem}"
        return f"W{self.weight_bits}"


@dataclass
class LRScheduleSpec:
    """Which LR schedule drives the run.

    "plateau" — ReduceLROnPlateau, stepped once per epoch by the harness.
    "cosine"  — per-step linear warmup + cosine annealing.
    "none"    — no scheduler.

    plateau and cosine are mutually exclusive by construction: cosine steps
    every batch and would overwrite any plateau-triggered reduction on the next
    batch (train_imagenet_qat.py:916-919). ``kind`` is authoritative — RunSpec
    keeps ``training.reduce_lr_on_plateau`` in sync with it.
    """

    kind: str = "plateau"
    cosine_warmup_frac: float = 0.1
    cosine_eta_min: float = 1e-6

    def validate(self) -> None:
        if self.kind not in VALID_LR_SCHEDULES:
            raise RunSpecError(
                f"LRScheduleSpec.kind must be one of {list(VALID_LR_SCHEDULES)}, "
                f"got {self.kind!r}"
            )
        if not (0.0 <= self.cosine_warmup_frac <= 1.0):
            raise RunSpecError(
                f"LRScheduleSpec.cosine_warmup_frac must be in [0, 1], "
                f"got {self.cosine_warmup_frac}"
            )


# ---------------------------------------------------------------------------
# RunSpec
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """Everything needed to launch one training run.

    Required: ``model`` and ``dataset``. Everything else has a default, and
    identity fields (run_id / experiment_name / output_dir) are derived in
    __post_init__ when not supplied — so a spec is always fully resolved once
    constructed, and serializing then reloading it reproduces the same run.
    """

    # ---- Identity ----------------------------------------------------------
    model: str
    dataset: str
    augmentation: Optional[str] = None
    """Pixel-level preset name. Defaults to the dataset's default preset."""

    augmentation_overrides: Dict[str, Any] = field(default_factory=dict)
    """Deviations from the preset, e.g. {"randaugment_m": 9}. Keys must be
    parameters the preset declares. Exists so the CLI's --randaugment-n/-m
    keep working without decomposing the preset."""

    run_id: Optional[str] = None
    experiment_name: Optional[str] = None
    output_dir: Optional[str] = None
    """Base output directory. The run's own files land in
    <output_dir>/runs/<run_id>/ — see build_run."""

    # ---- Ingredients -------------------------------------------------------
    data_dir: Optional[str] = None
    """Dataset root. Falls back to the dataset's env var, then its default."""

    dali_threads: int = 4
    """CPU threads for the DALI pipeline (ImageNet only)."""

    init_checkpoint: Optional[str] = None
    """Checkpoint to initialise weights from (the --init-from-ptq path).
    Loaded with strict=False, calibration state preserved."""

    pretrained: bool = False
    """Load timm float weights before QAT."""

    pretrained_qat: bool = False
    """Load timm weights, fuse BN, and run the PTQ LSB search in-process."""

    pretrained_qat_cache: Optional[str] = None
    ptq_search_radius: int = 7
    ptq_eval_batches: Optional[int] = None
    force_lsb_search: bool = False

    quant: QuantSpec = field(default_factory=QuantSpec)
    optimizer: str = "adamw"
    lr_schedule: LRScheduleSpec = field(default_factory=LRScheduleSpec)

    allow_untested_pair: bool = False
    """Proceed past an unrecognised (model, dataset) normalization pair with a
    warning instead of an error. See registry.resolve_normalization."""

    # ---- Training dynamics -------------------------------------------------
    training: TrainerConfigV2 = field(default_factory=TrainerConfigV2)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.validate()

        dataset_entry = get_dataset(self.dataset)
        if self.augmentation is None:
            self.augmentation = dataset_entry.default_augmentation

        # A dataset knows how many classes it has. Fill it in only when the
        # caller left TrainerConfigV2 at its own default, so an explicit choice
        # (e.g. the CLI's --num-classes for an ImageNet subset) always wins.
        default_num_classes = TrainerConfigV2().num_classes
        if (self.training.num_classes == default_num_classes
                and dataset_entry.num_classes != default_num_classes):
            self.training.num_classes = dataset_entry.num_classes
        if self.run_id is None:
            # Same format the trainer would otherwise generate (trainer_v2.py:145).
            self.run_id = time.strftime("%Y-%m-%d_%H%M%S")
        if self.experiment_name is None:
            self.experiment_name = f"{self.model}_{self.quant.describe()}_" \
                                   f"A{self.quant.act_bits}_B{self.quant.bias_bits}"
        if self.output_dir is None:
            # For dataset="imagenet" this reproduces the CLI's historical
            # default exactly: output/imagenet_qat_<model> (:434).
            self.output_dir = f"output/{self.dataset}_qat_{self.model}"

        # Identity fields are duplicated into the nested training config because
        # the harness reads them from there (experiment name -> checkpoint
        # filenames and log dir; run_id -> log dir).
        self.training.experiment_name = self.experiment_name
        self.training.run_id = self.run_id
        # Normalise to tuples so a spec built from JSON (which has only lists)
        # compares equal to the same spec built in Python.
        self.training.secondary_checkpoint_metrics = [
            tuple(pair) for pair in self.training.secondary_checkpoint_metrics
        ]
        # lr_schedule.kind is authoritative over the config's boolean so the
        # serialized spec can never contradict itself.
        self.training.reduce_lr_on_plateau = (self.lr_schedule.kind == "plateau")

        # Re-validate the cross-ingredient rules now that defaults are filled.
        self._validate_combination()

    def validate(self) -> None:
        """Field-level validation: names exist, scalars are sane."""
        get_model(self.model)          # raises RegistryError on unknown name
        get_dataset(self.dataset)

        self.quant.validate()
        self.lr_schedule.validate()

        if self.optimizer not in VALID_OPTIMIZERS:
            raise RunSpecError(
                f"RunSpec.optimizer must be one of {list(VALID_OPTIMIZERS)}, "
                f"got {self.optimizer!r}"
            )
        if self.pretrained_qat and self.init_checkpoint:
            raise RunSpecError(
                "pretrained_qat and init_checkpoint are mutually exclusive: the "
                "former runs the LSB search itself, the latter loads one."
            )
        if self.pretrained_qat and self.quant.weight_coeffs:
            raise RunSpecError(
                "pretrained_qat requires fixed-point weights (weight_bits); the "
                "LSB search does not apply to coefficient weights."
            )
        if self.dali_threads < 1:
            raise RunSpecError(f"dali_threads must be >= 1, got {self.dali_threads}")

    def _validate_combination(self) -> None:
        """Cross-ingredient validation: preset belongs to the dataset, overrides
        are real parameters, the model can honour the requested quantization."""
        aug_entry = get_augmentation(self.augmentation)
        if aug_entry.dataset != self.dataset:
            raise RunSpecError(
                f"augmentation preset {self.augmentation!r} belongs to dataset "
                f"{aug_entry.dataset!r}, not {self.dataset!r}"
            )
        # Raises on unknown override keys.
        resolve_augmentation_params(self.augmentation, self.augmentation_overrides)
        validate_model_quant_contract(self.model, self.quant, self.training.num_classes)

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> str:
        """Directory holding EVERYTHING this run produces.

        Per-run isolation: two runs never share a checkpoint pool, so a second
        run can no longer inherit — and evict — the first run's top-K
        checkpoints (checkpointing.py:343,363).
        """
        return os.path.join(self.output_dir, "runs", self.run_id)

    def augmentation_params(self) -> Dict[str, Any]:
        """Preset defaults merged with this spec's overrides."""
        return resolve_augmentation_params(self.augmentation, self.augmentation_overrides)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form. JSON-safe: no tuples, no non-primitive values."""
        data = dataclasses.asdict(self)
        # asdict preserves tuples; JSON turns them into lists. Normalise here so
        # to_dict() and json round-trips agree.
        data["training"]["secondary_checkpoint_metrics"] = [
            list(pair) for pair in self.training.secondary_checkpoint_metrics
        ]
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def write_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunSpec":
        """Rebuild a RunSpec from to_dict() output.

        Nested dataclasses are reconstructed explicitly — dataclasses.asdict is
        lossy in that direction (it produces plain dicts).
        """
        data = dict(data)
        quant = QuantSpec(**data.pop("quant", {}) or {})
        lr_schedule = LRScheduleSpec(**data.pop("lr_schedule", {}) or {})
        training = _trainer_config_from_dict(data.pop("training", {}) or {})
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise RunSpecError(f"unknown RunSpec field(s): {sorted(unknown)}")
        return cls(quant=quant, lr_schedule=lr_schedule, training=training, **data)

    @classmethod
    def from_json(cls, text: str) -> "RunSpec":
        return cls.from_dict(json.loads(text))

    @classmethod
    def read_json(cls, path: str) -> "RunSpec":
        with open(path) as f:
            return cls.from_json(f.read())


def _trainer_config_from_dict(data: Dict[str, Any]) -> TrainerConfigV2:
    """Rebuild a TrainerConfigV2 (and its three nested sub-configs) from a dict."""
    data = dict(data)
    qat = data.pop("qat", None)
    checkpoint = data.pop("checkpoint", None)
    logging_cfg = data.pop("logging", None)
    # JSON turns the (metric, mode) tuples into lists; restore tuples so a
    # round-tripped config compares equal to the original.
    secondary = data.pop("secondary_checkpoint_metrics", None)

    config = TrainerConfigV2(**data)
    if qat is not None:
        config.qat = QATScheduleConfigV2(**qat)
    if checkpoint is not None:
        config.checkpoint = CheckpointConfig(**checkpoint)
    if logging_cfg is not None:
        config.logging = LoggingConfig(**logging_cfg)
    if secondary is not None:
        config.secondary_checkpoint_metrics = [tuple(pair) for pair in secondary]
    return config
