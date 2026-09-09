"""
registry.py — Name → ingredient tables for the orchestration layer.

Three registries, each keyed by the name a RunSpec carries:

  MODELS         "resnet18"         -> how to construct that network
  DATASETS       "imagenet"         -> how to build its loaders
  AUGMENTATIONS  "imagenet_default" -> the pixel-level preset for a dataset

plus ``resolve_normalization``, which answers the (model, dataset) pair
question that no single table can answer on its own (see below).

Import cost is kept low on purpose: every heavy dependency (torch, brevitas,
DALI, timm, torchvision) is imported *inside* the builder functions, not at
module scope. That lets a launcher — or a test — enumerate and validate
ingredient names on a machine with no DALI installed.

--------------------------------------------------------------------------
Model quantization contracts
--------------------------------------------------------------------------
The models in this repo do NOT share a constructor signature. Three distinct
contracts exist, and ``ModelEntry.quant_contract`` names which one applies:

  "injectors"  __init__(num_classes, weight_quant, act_quant, bias_quant)
               Takes Brevitas injector *classes*. The four ImageNet models.

  "bit_ints"   __init__(num_classes, weight_bit_width, act_bit_width)
               Takes plain ints and builds its own quantizers internally.
               (No model registered yet — the CIFAR models use this.)

  "fixed"      __init__()
               Quantizers hardcoded inside the module; nothing is
               configurable. MNISTQuantNet. A spec that asks for bit widths
               other than the model's built-in 8/8/8 is REJECTED rather than
               silently ignored — see validate_model_quant_contract.

--------------------------------------------------------------------------
Known wart (for the later architecture pass)
--------------------------------------------------------------------------
``mnist_cnn`` is defined in examples/mnist_qat_v2.py, so this module imports
from examples/. That is the wrong direction — examples should depend on the
framework, not the reverse. It is a lazy, function-scoped import so it costs
nothing until the model is actually built, but the model belongs in models/.
Left in place here to keep this slice a pure extraction (moving it would
break `from examples.mnist_qat_v2 import MNISTQuantNet` in two tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from utils.normalization import _MODEL_NORM, norm_for_model


class RegistryError(ValueError):
    """Raised when a name is not in a registry, or an ingredient combination
    is rejected (unknown normalization pair, incompatible quant contract)."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    """One entry in the model registry.

    Attributes:
        name:              Registry key, as it appears in a RunSpec.
        quant_contract:    "injectors" | "bit_ints" | "fixed" (see module docstring).
        default_num_classes: Output size this model is normally built with.
        fixed_num_classes: If set, the ONLY output size this model supports
                           (contract "fixed" models hardwire their classifier).
        timm_name:         timm model id for --pretrained float weights, or None.
        datasets:          Datasets this model is known to work on. Used by
                           resolve_normalization to reject untested pairs.
        build:             (num_classes, injectors) -> nn.Module. `injectors` is
                           the (weight_quant, act_quant, bias_quant) triple; a
                           "fixed" model ignores it.
    """
    name: str
    quant_contract: str
    default_num_classes: int
    timm_name: Optional[str]
    datasets: Tuple[str, ...]
    build: Callable[..., Any]
    fixed_num_classes: Optional[int] = None


def _build_resnet18(num_classes, injectors):
    from models.resnet_quant import QuantResNet18
    w, a, b = injectors
    return QuantResNet18(num_classes, w, a, b)


def _build_resnet50(num_classes, injectors):
    from models.resnet_quant import QuantResNet50
    w, a, b = injectors
    return QuantResNet50(num_classes, w, a, b)


def _build_mobilenetv1(num_classes, injectors):
    from models.mobilenetv1_quant import QuantMobileNetV1
    w, a, b = injectors
    return QuantMobileNetV1(num_classes, w, a, b)


def _build_mobilenetv2(num_classes, injectors):
    from models.mobilenetv2_quant import QuantMobileNetV2
    w, a, b = injectors
    # Keyword form matches the original call site in train_imagenet_qat.py:491 —
    # this model's positional slots 2 and 3 are bit-width ints, not injectors.
    return QuantMobileNetV2(num_classes, weight_quant=w, act_quant=a, bias_quant=b)


def _build_mnist_cnn(num_classes, injectors):
    # See "Known wart" in the module docstring.
    from examples.mnist_qat_v2 import MNISTQuantNet
    return MNISTQuantNet()


MODELS: Dict[str, ModelEntry] = {
    "resnet18": ModelEntry(
        name="resnet18", quant_contract="injectors", default_num_classes=1000,
        timm_name="resnet18.a1_in1k", datasets=("imagenet",), build=_build_resnet18,
    ),
    "resnet50": ModelEntry(
        name="resnet50", quant_contract="injectors", default_num_classes=1000,
        timm_name="resnet50.a1_in1k", datasets=("imagenet",), build=_build_resnet50,
    ),
    "mobilenetv1": ModelEntry(
        name="mobilenetv1", quant_contract="injectors", default_num_classes=1000,
        timm_name="mobilenetv1_100.ra4_e3600_r224_in1k", datasets=("imagenet",),
        build=_build_mobilenetv1,
    ),
    "mobilenetv2": ModelEntry(
        name="mobilenetv2", quant_contract="injectors", default_num_classes=1000,
        timm_name="mobilenetv2_100.ra_in1k", datasets=("imagenet",),
        build=_build_mobilenetv2,
    ),
    "mnist_cnn": ModelEntry(
        name="mnist_cnn", quant_contract="fixed", default_num_classes=10,
        fixed_num_classes=10, timm_name=None, datasets=("mnist",),
        build=_build_mnist_cnn,
    ),
}

# Bit widths a "fixed"-contract model is hardwired to, from the injector class
# defaults in quantizers/base_injector.py:33,42.
FIXED_CONTRACT_BITS = {"weight_bits": 8, "act_bits": 8, "bias_bits": 8}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetEntry:
    """One entry in the dataset registry.

    Attributes:
        name:                 Registry key.
        num_classes:          Output size for this dataset.
        input_shape:          (C, H, W) of one sample — used to build the ONNX
                              dummy input, replacing the hardcoded (1,3,224,224).
        default_augmentation: Preset used when a RunSpec names none.
        requires_data_dir:    True when there is no download fallback (ImageNet).
        data_dir_env:         Environment variable consulted for the root.
        default_data_dir:     Root used when neither spec nor env supplies one.
        build_loaders:        (spec, aug_params, norm) -> (train_loader, val_loader)
    """
    name: str
    num_classes: int
    input_shape: Tuple[int, int, int]
    default_augmentation: str
    build_loaders: Callable[..., Any]
    requires_data_dir: bool = False
    data_dir_env: Optional[str] = None
    default_data_dir: Optional[str] = None


def _build_imagenet_loaders(spec, aug_params, norm):
    """DALI ImageFolder loaders — the same call train_imagenet_qat.py:817 made."""
    from utils.dali_pipeline import build_dali_loaders

    mean, std = norm
    data_dir = resolve_data_dir(spec)
    print(f"Building DALI loaders from {data_dir} …")
    print(f"  normalization for {spec.model}: mean={mean} std={std}")
    train_loader, val_loader = build_dali_loaders(
        data_dir=data_dir,
        batch_size=spec.training.batch_size,
        num_threads=spec.dali_threads,
        randaugment_n=aug_params["randaugment_n"],
        randaugment_m=aug_params["randaugment_m"],
        crop=aug_params["crop"],
        resize_shorter=aug_params["resize_shorter"],
        mean=mean,
        std=std,
    )
    print(f"  train: {len(train_loader):,} batches   val: {len(val_loader):,} batches")
    return train_loader, val_loader


def _build_mnist_loaders(spec, aug_params, norm):
    """torchvision MNIST — transforms lifted verbatim from examples/mnist_qat_v2.py:107-118."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    mean, std = norm
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = resolve_data_dir(spec)
    num_workers = spec.training.num_workers
    train_loader = DataLoader(
        datasets.MNIST(root, train=True, download=True, transform=transform),
        batch_size=spec.training.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.MNIST(root, train=False, download=True, transform=transform),
        batch_size=spec.training.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


DATASETS: Dict[str, DatasetEntry] = {
    "imagenet": DatasetEntry(
        name="imagenet", num_classes=1000, input_shape=(3, 224, 224),
        default_augmentation="imagenet_default", build_loaders=_build_imagenet_loaders,
        requires_data_dir=True, data_dir_env="IMAGENET_DALI_PATH",
    ),
    "mnist": DatasetEntry(
        name="mnist", num_classes=10, input_shape=(1, 28, 28),
        default_augmentation="mnist_default", build_loaders=_build_mnist_loaders,
        default_data_dir="./data",
    ),
}


def resolve_data_dir(spec) -> str:
    """Resolve a dataset root from spec → environment → registry default.

    Raises RegistryError when a dataset that has no download fallback
    (ImageNet) ends up with no root at all.
    """
    import os

    entry = get_dataset(spec.dataset)
    if spec.data_dir:
        return spec.data_dir
    if entry.data_dir_env:
        from_env = os.environ.get(entry.data_dir_env)
        if from_env:
            return from_env
    if entry.default_data_dir:
        return entry.default_data_dir
    raise RegistryError(
        f"dataset {spec.dataset!r} needs a data directory: set RunSpec.data_dir "
        f"or the {entry.data_dir_env} environment variable."
    )


# ---------------------------------------------------------------------------
# Augmentation presets (pixel-level only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AugmentationEntry:
    """A named pixel-level augmentation preset for one dataset.

    ``params`` holds today's values AND doubles as the allow-list of keys a
    RunSpec may override via ``augmentation_overrides``.

    Scope note: this covers the *pixel-level* half of augmentation only. The
    batch-level half (mixup / cutmix / label smoothing / random erasing) stays
    in TrainerConfigV2, where it already lived. Slice 1 deliberately does not
    unify the two — see docs/llm/RUNSPEC_AND_BUILD_RUN.md.
    """
    name: str
    dataset: str
    params: Dict[str, Any] = field(default_factory=dict)


AUGMENTATIONS: Dict[str, AugmentationEntry] = {
    # Values are today's defaults: train_imagenet_qat.py:136-139 (randaugment)
    # and utils/dali_pipeline.py build_dali_loaders defaults (crop, resize).
    "imagenet_default": AugmentationEntry(
        name="imagenet_default", dataset="imagenet",
        params={"randaugment_n": 2, "randaugment_m": 7,
                "crop": 224, "resize_shorter": 256},
    ),
    # MNIST has no pixel-level knobs — ToTensor + Normalize only.
    "mnist_default": AugmentationEntry(
        name="mnist_default", dataset="mnist", params={},
    ),
}


def resolve_augmentation_params(augmentation: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a preset's defaults with a spec's overrides.

    Unknown override keys are rejected — an override that silently does nothing
    is the same class of bug as a silently-wrong normalization.
    """
    entry = get_augmentation(augmentation)
    unknown = set(overrides) - set(entry.params)
    if unknown:
        raise RegistryError(
            f"augmentation preset {augmentation!r} has no parameter(s) "
            f"{sorted(unknown)}; valid keys: {sorted(entry.params) or '(none)'}"
        )
    return {**entry.params, **overrides}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def model_names() -> Tuple[str, ...]:
    return tuple(sorted(MODELS))


def dataset_names() -> Tuple[str, ...]:
    return tuple(sorted(DATASETS))


def augmentation_names() -> Tuple[str, ...]:
    return tuple(sorted(AUGMENTATIONS))


def get_model(name: str) -> ModelEntry:
    try:
        return MODELS[name]
    except KeyError:
        raise RegistryError(
            f"unknown model {name!r}; registered models: {list(model_names())}"
        ) from None


def get_dataset(name: str) -> DatasetEntry:
    try:
        return DATASETS[name]
    except KeyError:
        raise RegistryError(
            f"unknown dataset {name!r}; registered datasets: {list(dataset_names())}"
        ) from None


def get_augmentation(name: str) -> AugmentationEntry:
    try:
        return AUGMENTATIONS[name]
    except KeyError:
        raise RegistryError(
            f"unknown augmentation preset {name!r}; registered presets: "
            f"{list(augmentation_names())}"
        ) from None


# ---------------------------------------------------------------------------
# Cross-ingredient validation
# ---------------------------------------------------------------------------

def resolve_normalization(model: str, dataset: str, *, allow_untested_pair: bool = False):
    """Return the (mean, std) an input pipeline must use for this pair.

    Normalization is a property of the (model, dataset) PAIR, not of either
    alone: timm's mobilenetv1_100.ra4 checkpoint was trained with mean=std=0.5,
    and feeding it ImageNet-normalized inputs collapses top-1 from ~73% to ~17%
    (utils/normalization.py). Because that failure is silent — the run trains,
    it just trains badly — an unrecognised pair is REFUSED rather than
    defaulted. Pass allow_untested_pair=True to downgrade to a loud warning.

    Raises:
        RegistryError: on an unknown name, or an untested pair without the opt-in.
    """
    model_entry = get_model(model)
    dataset_entry = get_dataset(dataset)

    tested = dataset in model_entry.datasets
    if not tested:
        message = (
            f"untested pair: model {model!r} has only been run on "
            f"{list(model_entry.datasets)}, not {dataset!r}. Input normalization "
            f"is a property of the (model, dataset) pair — a wrong choice trains "
            f"silently and badly (see utils/normalization.py). Set "
            f"allow_untested_pair=True on the RunSpec to proceed anyway."
        )
        if not allow_untested_pair:
            raise RegistryError(message)
        import warnings
        warnings.warn(f"[normalization] {message}", RuntimeWarning, stacklevel=2)

    if dataset == "imagenet":
        if model not in _MODEL_NORM and not allow_untested_pair:
            raise RegistryError(
                f"no ImageNet normalization recorded for model {model!r}; "
                f"known: {sorted(_MODEL_NORM)}. Refusing to fall back to default "
                f"ImageNet stats — set allow_untested_pair=True to accept them."
            )
        # utils/normalization.py stays the single source of truth for these.
        return norm_for_model(model)

    if dataset == "mnist":
        # Same stats as examples/mnist_qat_v2.py:109.
        return ((0.1307,), (0.3081,))

    raise RegistryError(f"no normalization rule for dataset {dataset!r}")


def validate_model_quant_contract(model: str, quant, num_classes: int) -> None:
    """Reject specs whose quantization the named model cannot honour.

    A "fixed"-contract model hardcodes its quantizers and its classifier width,
    so a spec asking for 4-bit weights would be silently ignored — the run would
    train at 8-bit while its manifest and checkpoint claimed 4-bit. Refuse instead.

    Raises:
        RegistryError: when the model cannot deliver what the spec asks for.
    """
    entry = get_model(model)

    if entry.fixed_num_classes is not None and num_classes != entry.fixed_num_classes:
        raise RegistryError(
            f"model {model!r} hardwires num_classes={entry.fixed_num_classes} "
            f"but the spec asks for {num_classes}."
        )

    if entry.quant_contract != "fixed":
        return

    if quant.weight_coeffs is not None:
        raise RegistryError(
            f"model {model!r} hardcodes its quantizers and cannot use "
            f"coefficient weights (weight_coeffs)."
        )
    actual = {"weight_bits": quant.weight_bits,
              "act_bits": quant.act_bits,
              "bias_bits": quant.bias_bits}
    mismatched = {k: v for k, v in actual.items() if v != FIXED_CONTRACT_BITS[k]}
    if mismatched:
        raise RegistryError(
            f"model {model!r} hardcodes its quantizers at "
            f"{FIXED_CONTRACT_BITS} (quantizers/base_injector.py defaults) but the "
            f"spec asks for {mismatched}. The request would be silently ignored, "
            f"so it is refused instead."
        )
