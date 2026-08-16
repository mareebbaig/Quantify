"""
test_run_builder.py — build_run() end-to-end on MNIST.

Exercises the plumbing a launcher depends on: a RunSpec becomes a runnable
trainer, the run's dashboard binds a discoverable port, the manifest lands on
disk, checkpoints carry their provenance, and every output is under a per-run
directory.

MNIST is used only because it is small and already vendored under data/. Draw
no training-quality conclusions from it — the run is two dry-run batches.
"""

import json
import os
import urllib.request

import pytest
import torch

from orchestration.run_builder import build_run
from orchestration.run_spec import QuantSpec, RunSpec
from training_harness.config import CheckpointConfig, LoggingConfig
from training_harness.config_v2 import QATScheduleConfigV2, TrainerConfigV2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST_ROOT = os.path.join(REPO_ROOT, "data")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(MNIST_ROOT, "MNIST")),
    reason="MNIST data not vendored under data/",
)


def _mnist_spec(output_dir: str, *, api_port=None, run_id="smoke-run") -> RunSpec:
    return RunSpec(
        model="mnist_cnn",
        dataset="mnist",
        run_id=run_id,
        output_dir=output_dir,
        data_dir=MNIST_ROOT,
        quant=QuantSpec(weight_bits=8, act_bits=8, bias_bits=8),
        training=TrainerConfigV2(
            epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            num_workers=0,          # keep the test single-process on Windows
            device="cpu",
            dry_run=True,
            dry_run_batches=2,
            api_port=api_port,
            smoothing=0.0,          # avoid pulling in timm's LabelSmoothing
            logging=LoggingConfig(log_every_n_steps=1, save_plots=False),
            qat=QATScheduleConfigV2(
                float_warmup_epochs=1,
                plateau_patience=1,
                annealing_steps=4,
                quantization_start_gap=2,
            ),
            checkpoint=CheckpointConfig(monitor_metric="val_acc", monitor_mode="max",
                                        top_k=2, save_last=True),
        ),
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One built-and-fitted run, shared by the assertions below.

    Scoped to the module because fit() mutates the QuantizerManager singleton;
    building several runs in one process is exactly the state-crossing the
    harness warns about.
    """
    output_dir = str(tmp_path_factory.mktemp("runs_base"))
    spec = _mnist_spec(output_dir, api_port=0)
    handle = build_run(spec)
    tracker = handle.fit()
    return handle, tracker


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------

def test_build_run_returns_a_wired_trainer(built):
    handle, _ = built

    assert handle.trainer is not None
    assert handle.model is not None
    assert handle.optimizer is not None
    assert handle.train_loader is not None and handle.val_loader is not None
    # The optimizer choice the spec asked for.
    assert isinstance(handle.optimizer, torch.optim.AdamW)


def test_training_actually_proceeds(built):
    """V2 records one snapshot per phase — a baseline val pass at epoch -1, then
    a train and a val snapshot for each epoch — so assert on the epochs that
    ran rather than on a snapshot count."""
    _, tracker = built

    train_epochs = {
        snapshot.epoch for snapshot in tracker.history
        if "train_loss" in snapshot.metrics
    }
    val_epochs = {
        snapshot.epoch for snapshot in tracker.history
        if "val_loss" in snapshot.metrics and snapshot.epoch >= 0
    }

    assert train_epochs == {0, 1}, f"both epochs should have trained, got {train_epochs}"
    assert val_epochs == {0, 1}, f"both epochs should have validated, got {val_epochs}"


def test_onnx_dummy_input_comes_from_the_dataset(built):
    """Previously hardcoded to (1, 3, 224, 224) — wrong for anything but ImageNet."""
    handle, _ = built
    assert tuple(handle.trainer._onnx_dummy_input.shape) == (1, 1, 28, 28)


# ---------------------------------------------------------------------------
# Per-run output isolation
# ---------------------------------------------------------------------------

def test_outputs_are_under_a_per_run_directory(built):
    handle, _ = built
    run_dir = os.path.abspath(handle.run_dir)

    assert run_dir.endswith(os.path.join("runs", "smoke-run"))
    for path in (handle.config.checkpoint_dir, handle.config.log_dir,
                 handle.config.plot_dir):
        assert os.path.abspath(path).startswith(run_dir), \
            f"{path} escapes the run directory"


def test_two_runs_do_not_share_a_checkpoint_pool(tmp_path):
    """The latent bug per-run isolation fixes: CheckpointManager reloads
    checkpoint_index.json from its save_dir (checkpointing.py:363) and prunes
    across it (:343), so a second run in a shared directory could evict the
    first run's checkpoints. Different run_ids must not collide."""
    base = str(tmp_path / "base")
    first = _mnist_spec(base, run_id="run-one")
    second = _mnist_spec(base, run_id="run-two")

    assert first.run_dir != second.run_dir
    first_ckpts = os.path.join(first.run_dir, "checkpoints")
    second_ckpts = os.path.join(second.run_dir, "checkpoints")
    assert first_ckpts != second_ckpts


# ---------------------------------------------------------------------------
# Manifest + port discovery
# ---------------------------------------------------------------------------

def test_manifest_is_written_and_holds_the_spec(built):
    handle, _ = built

    assert handle.manifest_path and os.path.exists(handle.manifest_path)
    with open(handle.manifest_path) as f:
        manifest = json.load(f)

    assert manifest["run_id"] == "smoke-run"
    assert manifest["experiment_name"] == handle.spec.experiment_name
    assert manifest["pid"] == os.getpid()
    # The spec inside the manifest must rebuild the same run.
    assert RunSpec.from_dict(manifest["spec"]) == handle.spec


def test_latest_pointer_is_written(built):
    """Per-run directories removed the fixed <output_dir>/checkpoints/last.pt
    address; latest.json restores a stable one for shell scripts."""
    handle, _ = built
    pointer_path = os.path.join(handle.spec.output_dir, "latest.json")

    assert os.path.exists(pointer_path)
    with open(pointer_path) as f:
        pointer = json.load(f)

    assert pointer["run_id"] == "smoke-run"
    assert pointer["last_checkpoint"].endswith("last.pt")
    assert os.path.abspath(handle.run_dir) == pointer["run_dir"]


def test_bound_api_port_is_discoverable(built):
    """api_port=0 means the OS picks; the launcher must learn which one."""
    handle, _ = built

    assert isinstance(handle.api_port, int) and handle.api_port > 0
    assert handle.dashboard_url.endswith(f":{handle.api_port}/api/v1/")

    with open(handle.manifest_path) as f:
        assert json.load(f)["api_port"] == handle.api_port


def test_dashboard_api_answers_on_the_bound_port(built):
    handle, _ = built

    with urllib.request.urlopen(f"{handle.dashboard_url}status", timeout=10) as response:
        status = json.load(response)

    assert status["run_id"] == "smoke-run"
    assert status["trainer_version"] == "v2"


def test_no_api_server_when_api_port_is_unset(tmp_path):
    spec = _mnist_spec(str(tmp_path / "no_api"), api_port=None, run_id="no-api")
    handle = build_run(spec)

    assert handle.api_port is None
    assert handle.dashboard_url is None


# ---------------------------------------------------------------------------
# Checkpoint provenance
# ---------------------------------------------------------------------------

def test_checkpoint_carries_the_run_spec(built):
    """The audit's asymmetry — PTQ checkpoints were self-describing while QAT
    checkpoints were not — closed by embedding the spec in extra."""
    handle, _ = built
    last = os.path.join(handle.config.checkpoint_dir, "last.pt")
    assert os.path.exists(last)

    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert "run_spec" in payload.get("extra", {})

    restored = RunSpec.from_dict(payload["extra"]["run_spec"])
    assert restored == handle.spec
    # The three things a checkpoint could not previously tell you.
    assert restored.model == "mnist_cnn"
    assert restored.dataset == "mnist"
    assert restored.augmentation == "mnist_default"
