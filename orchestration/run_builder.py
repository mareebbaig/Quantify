"""
run_builder.py — build_run(spec) -> RunHandle.

Turns a RunSpec into a fully-wired, ready-to-fit QATTrainerV2. This is the
object-construction logic that used to live inline in
examples/train_imagenet_qat.main(); the CLI is now a thin adapter over it
(argparse -> RunSpec -> build_run), so the programmatic and command-line doors
build byte-identical runs.

build_run assembles inputs. It does NOT touch fit() or the training loop.

--------------------------------------------------------------------------
Construction order (it matters)
--------------------------------------------------------------------------
 1. Resolve identity + create the run directory
 2. Optional stdout/stderr tee into the run directory
 3. Resolve normalization from the (model, dataset) PAIR   <- before loaders
 4. Build dataloaders                                      <- before the model:
       --pretrained-qat's LSB search needs the val loader and a calibration
       batch, which is why main() built data first (train_imagenet_qat.py:875)
 5. Build quantizer injector classes
 6. Build the model
 7. Initialise weights: pretrained_qat -> pretrained -> init_checkpoint
       (same precedence as train_imagenet_qat.py:884-889)
 8. Apply weight_lsb_subtract
 9. Optimizer, then LR schedule
10. Assemble TrainerConfigV2 with a run-scoped output_dir
11. Construct QATTrainerV2, embedding the spec as checkpoint provenance
12. Read back the bound API port; write run.json and latest.json

--------------------------------------------------------------------------
Known wart (for the later architecture pass)
--------------------------------------------------------------------------
``pretrained_qat=True`` lazily imports _prepare_pretrained_qat from
examples/train_imagenet_qat.py — orchestration reaching into examples/, the
wrong direction. That helper drives the in-process LSB search and is itself
coupled to examples/find_perfect_lsbs_imagenet_ptq.py, so moving it is a
bigger extraction than this slice should carry. The import is function-scoped
and only fires when the flag is set.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from training_harness.config import CheckpointConfig, LoggingConfig
from training_harness.config_v2 import TrainerConfigV2
from training_harness.schedulers import WarmupCosineScheduler
from training_harness.trainer_v2 import QATTrainerV2
from utils.run_utils import setup_output_tee

from .checkpoint_init import load_pretrained_weights, load_ptq_checkpoint
from .registry import get_dataset, get_model, resolve_normalization
from .run_spec import RunSpec

MANIFEST_NAME = "run.json"
LATEST_POINTER_NAME = "latest.json"


# ---------------------------------------------------------------------------
# Quantizer injectors
# ---------------------------------------------------------------------------

def make_injectors(quant) -> Tuple[type, type, type]:
    """Build the (weight, activation, bias) Brevitas injector classes.

    Same subclass-with-an-overridden-attribute construction the CLI used
    (train_imagenet_qat.py:451-475), driven by a QuantSpec instead of argparse.
    """
    from quantizers.coefficient_per_tensor_weights import CoefficientPerTensorWeightQuant
    from quantizers.fixedpoint_per_tensor import (
        FixedPointPerTensorActivationQuant,
        FixedPointPerTensorBiasQuant,
        FixedPointPerTensorWeightQuant,
    )

    if quant.weight_coeffs:
        _fp = quant.weight_coeffs

        class WeightQuant(CoefficientPerTensorWeightQuant):
            filepath = _fp
    else:
        _wb = quant.weight_bits

        class WeightQuant(FixedPointPerTensorWeightQuant):
            bit_width = _wb

    _ab = quant.act_bits

    class ActQuant(FixedPointPerTensorActivationQuant):
        bit_width = _ab

    _bb = quant.bias_bits

    class BiasQuant(FixedPointPerTensorBiasQuant):
        bit_width = _bb

    return WeightQuant, ActQuant, BiasQuant


# ---------------------------------------------------------------------------
# Run handle
# ---------------------------------------------------------------------------

@dataclass
class RunHandle:
    """A built, not-yet-started run.

    Carries the assembled objects plus the addresses an orchestrator needs:
    where the run writes (``run_dir``), where its manifest is
    (``manifest_path``), and which port its dashboard bound to (``api_port``).
    """

    spec: RunSpec
    config: TrainerConfigV2
    trainer: QATTrainerV2
    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: Any
    val_loader: Any
    run_dir: str
    manifest_path: Optional[str] = None
    api_port: Optional[int] = None
    bn_fused: bool = False

    def fit(self, **kwargs):
        """Run the training loop. Passes through to QATTrainerV2.fit()."""
        return self.trainer.fit(**kwargs)

    @property
    def dashboard_url(self) -> Optional[str]:
        """Base URL of this run's monitoring API, or None when it has no server."""
        if self.api_port is None:
            return None
        return f"http://{self.config.api_host}:{self.api_port}/api/v1/"


# ---------------------------------------------------------------------------
# build_run
# ---------------------------------------------------------------------------

def build_run(
    spec: RunSpec,
    *,
    tee_stdout: bool = False,
    write_manifest: bool = True,
) -> RunHandle:
    """Assemble everything a RunSpec describes and return a ready-to-fit handle.

    Args:
        spec:           The run, as data. Already validated by its constructor.
        tee_stdout:     Duplicate stdout/stderr into <run_dir>/run.log. The CLI
                        passes True (preserving its existing behaviour);
                        programmatic callers default to False so a library call
                        does not hijack the process's streams.
        write_manifest: Write <run_dir>/run.json and <output_dir>/latest.json.

    Returns:
        RunHandle — call .fit() to start training.
    """
    model_entry = get_model(spec.model)
    dataset_entry = get_dataset(spec.dataset)

    # ── 1. Identity + run directory ────────────────────────────────────
    run_dir = spec.run_dir
    os.makedirs(run_dir, exist_ok=True)

    # ── 2. Tee ─────────────────────────────────────────────────────────
    if tee_stdout:
        setup_output_tee(run_dir)

    _print_header(spec, run_dir)

    # ── 3. Normalization from the (model, dataset) PAIR ────────────────
    norm = resolve_normalization(
        spec.model, spec.dataset, allow_untested_pair=spec.allow_untested_pair
    )

    # ── 4. Dataloaders (before the model — see module docstring) ───────
    aug_params = spec.augmentation_params()
    train_loader, val_loader = dataset_entry.build_loaders(spec, aug_params, norm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 5-6. Injectors, then model ─────────────────────────────────────
    injectors = make_injectors(spec.quant)
    model = model_entry.build(spec.training.num_classes, injectors)

    # ── 7. Weight initialisation ───────────────────────────────────────
    bn_fused = False
    if spec.pretrained_qat:
        model, bn_fused = _prepare_pretrained_qat(spec, model, device, train_loader, val_loader)
    elif spec.pretrained:
        model = load_pretrained_weights(model, spec.model)
    if spec.init_checkpoint:
        model, bn_fused = load_ptq_checkpoint(model, spec.init_checkpoint)

    # ── 8. Weight LSB shift ────────────────────────────────────────────
    if spec.quant.weight_lsb_subtract:
        from examples.train_imagenet_qat import _apply_weight_lsb_subtract
        _apply_weight_lsb_subtract(model, spec.quant.weight_lsb_subtract)

    # ── 9. Optimizer + LR schedule ─────────────────────────────────────
    optimizer = _build_optimizer(spec, model)
    scheduler = _build_scheduler(spec, optimizer, train_loader)

    # ── 10. Run-scoped training config ─────────────────────────────────
    config = _build_config(spec, run_dir)

    # ── 11. Trainer, with the spec as checkpoint provenance ────────────
    dummy_input = torch.zeros(1, *dataset_entry.input_shape)
    extra_fields: Dict[str, Any] = {"run_spec": spec.to_dict()}
    if bn_fused:
        extra_fields["fuse_bn"] = True

    trainer = QATTrainerV2(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        scheduler=scheduler,
        onnx_dummy_input=dummy_input,
        extra_checkpoint_fields=extra_fields,
    )

    # ── 12. Bound port + manifest ──────────────────────────────────────
    api_port = _bound_api_port(trainer)

    handle = RunHandle(
        spec=spec, config=config, trainer=trainer, model=model, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader, run_dir=run_dir,
        api_port=api_port, bn_fused=bn_fused,
    )

    if write_manifest:
        handle.manifest_path = write_run_manifest(handle)
        write_latest_pointer(handle)

    return handle


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def _build_optimizer(spec: RunSpec, model: nn.Module) -> torch.optim.Optimizer:
    """AdamW with the spec's LR/weight-decay — the CLI's hardcoded choice
    (train_imagenet_qat.py:895), now reachable from the spec."""
    if spec.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=spec.training.learning_rate,
            weight_decay=spec.training.weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {spec.optimizer!r}")


def _build_scheduler(spec: RunSpec, optimizer, train_loader):
    """Per-step cosine schedule, or None when the harness's epoch-stepped
    ReduceLROnPlateau is driving (train_imagenet_qat.py:916-931)."""
    if spec.lr_schedule.kind != "cosine":
        return None
    total_steps = len(train_loader) * spec.training.epochs
    warmup_steps = int(total_steps * spec.lr_schedule.cosine_warmup_frac)
    print(f"[lr-schedule] Cosine: total_steps={total_steps:,} "
          f"warmup={warmup_steps:,} "
          f"eta_min={spec.lr_schedule.cosine_eta_min} (ReduceLROnPlateau disabled)")
    return WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=spec.lr_schedule.cosine_eta_min,
    )


def _build_config(spec: RunSpec, run_dir: str) -> TrainerConfigV2:
    """Return the spec's TrainerConfigV2 re-pointed at the run-scoped directory.

    The spec's own ``output_dir`` is the *base*; the harness writes into
    <base>/runs/<run_id>/ so checkpoints, plots and logs are per-run. Nothing in
    training_harness/config_v2.py changes — direct TrainerConfigV2 users keep
    their current layout.
    """
    config = spec.training
    config.output_dir = run_dir
    config.experiment_name = spec.experiment_name
    config.run_id = spec.run_id
    return config


def _prepare_pretrained_qat(spec: RunSpec, model, device, train_loader, val_loader):
    """Run (or reuse a cache of) the in-process PTQ LSB search.

    See "Known wart" in the module docstring: this reaches into examples/.
    """
    from examples.train_imagenet_qat import _prepare_pretrained_qat as _legacy

    return _legacy(_legacy_args(spec), model, device, train_loader, val_loader)


def _legacy_args(spec: RunSpec):
    """Compatibility shim: an argparse-shaped namespace for helpers that still
    take one. Kept in one place so the architecture pass can delete it whole."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model=spec.model,
        weight_bits=spec.quant.weight_bits,
        act_bits=spec.quant.act_bits,
        bias_bits=spec.quant.bias_bits,
        weight_coeffs=spec.quant.weight_coeffs,
        num_classes=spec.training.num_classes,
        output_dir=spec.run_dir,
        ptq_search_radius=spec.ptq_search_radius,
        ptq_eval_batches=spec.ptq_eval_batches,
        pretrained_qat_cache=spec.pretrained_qat_cache,
        force_lsb_search=spec.force_lsb_search,
    )


def _bound_api_port(trainer: QATTrainerV2) -> Optional[int]:
    """The port the monitoring API actually bound to.

    With ``api_port=0`` the OS picks a free port, so the requested value is not
    the answer — DashboardAPIServer.port is (server.py:237-239). Returns None
    when the run has no API server, or when binding failed.
    """
    server = getattr(trainer, "api_server", None)
    if server is None:
        return None
    return server.port


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def write_run_manifest(handle: RunHandle) -> str:
    """Write <run_dir>/run.json — the record a launcher globs to list runs.

    Holds the full spec (so the run is reproducible from the manifest alone)
    plus the runtime facts only the live process knows: bound port and pid.
    """
    import json

    manifest = {
        "run_id": handle.spec.run_id,
        "experiment_name": handle.spec.experiment_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": os.getpid(),
        "api_port": handle.api_port,
        "api_host": handle.config.api_host,
        "dashboard_url": handle.dashboard_url,
        "run_dir": os.path.abspath(handle.run_dir),
        "checkpoint_dir": os.path.abspath(handle.config.checkpoint_dir),
        "log_dir": os.path.abspath(handle.config.log_dir),
        "plot_dir": os.path.abspath(handle.config.plot_dir),
        "spec": handle.spec.to_dict(),
    }
    path = os.path.join(handle.run_dir, MANIFEST_NAME)
    os.makedirs(handle.run_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[run] Manifest → {os.path.abspath(path)}")
    return path


def write_latest_pointer(handle: RunHandle) -> str:
    """Write <output_dir>/latest.json pointing at this run.

    Per-run directories mean there is no longer a fixed
    <output_dir>/checkpoints/last.pt to reference from a shell script. This
    pointer restores a stable, machine-readable address for "the newest run
    under this base directory". A file rather than a symlink because symlinks
    need elevation on Windows.
    """
    import json

    last_ckpt = os.path.join(handle.config.checkpoint_dir, "last.pt")
    pointer = {
        "run_id": handle.spec.run_id,
        "experiment_name": handle.spec.experiment_name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": os.path.abspath(handle.run_dir),
        "manifest": os.path.abspath(os.path.join(handle.run_dir, MANIFEST_NAME)),
        "checkpoint_dir": os.path.abspath(handle.config.checkpoint_dir),
        "last_checkpoint": os.path.abspath(last_ckpt),
        "api_port": handle.api_port,
    }
    base = handle.spec.output_dir
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, LATEST_POINTER_NAME)
    with open(path, "w") as f:
        json.dump(pointer, f, indent=2, default=str)
    return path


def _print_header(spec: RunSpec, run_dir: str) -> None:
    """The run banner the CLI has always printed (train_imagenet_qat.py:860-873),
    plus the ingredient names the spec now makes explicit."""
    print(f"\n{'═'*60}")
    print(f"  Experiment : {spec.experiment_name}")
    print(f"  Run ID     : {spec.run_id}")
    print(f"  Model      : {spec.model}")
    print(f"  Dataset    : {spec.dataset}")
    print(f"  Augment    : {spec.augmentation}"
          f"{f'  overrides={spec.augmentation_overrides}' if spec.augmentation_overrides else ''}")
    print(f"  Weight Q   : {spec.quant.describe()}")
    print(f"  Act Q      : A{spec.quant.act_bits}")
    print(f"  Bias Q     : B{spec.quant.bias_bits}")
    print(f"  Pretrained : {spec.pretrained_qat or spec.pretrained}"
          f"{'  (+ in-process LSB search)' if spec.pretrained_qat else ''}")
    if spec.init_checkpoint:
        print(f"  Init from  : {spec.init_checkpoint}")
    print(f"  AMP        : {spec.training.mixed_precision}")
    print(f"  Run dir    : {os.path.abspath(run_dir)}")
    print(f"{'═'*60}\n")
