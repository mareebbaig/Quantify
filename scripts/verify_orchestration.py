"""
verify_orchestration.py — one-command check that the RunSpec / build_run layer works.

Run it from the repo root:

    python scripts/verify_orchestration.py

Every check prints [OK] or [FAIL] with a plain-English description. The script
exits 0 only if all of them pass. It needs no GPU, no DALI and no ImageNet —
it uses the MNIST data already vendored under data/, trains for a few batches,
and deletes everything it wrote when it finishes.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_results: list[tuple[bool, str, str]] = []


def check(description: str):
    """Run a check function, record OK/FAIL, never raise."""
    def decorator(fn):
        try:
            detail = fn() or ""
            _results.append((True, description, str(detail)))
        except Exception as exc:  # noqa: BLE001 — a failed check must not stop the run
            _results.append((False, description, f"{type(exc).__name__}: {exc}"))
            if os.environ.get("VERIFY_TRACEBACK"):
                traceback.print_exc()
        return fn
    return decorator


def main() -> int:
    print("=" * 72)
    print("  Verifying the orchestration layer (RunSpec + build_run)")
    print("=" * 72)
    print()

    # ── 1. The package imports ────────────────────────────────────────────
    @check("orchestration package imports")
    def _():
        from orchestration import RunSpec, build_run  # noqa: F401
        return "orchestration.RunSpec, orchestration.build_run"

    from orchestration import RunSpec, build_run
    from orchestration.registry import (
        RegistryError, dataset_names, model_names, resolve_normalization,
    )
    from orchestration.run_spec import QuantSpec, RunSpecError
    from training_harness.config_v2 import QATScheduleConfigV2, TrainerConfigV2

    # ── 2. Ingredients are addressable by name ────────────────────────────
    @check("models and datasets are listable by name")
    def _():
        models, datasets = model_names(), dataset_names()
        assert "resnet18" in models and "mnist_cnn" in models, models
        assert "imagenet" in datasets and "mnist" in datasets, datasets
        return f"models={list(models)} datasets={list(datasets)}"

    # ── 3. A run survives a JSON round-trip ───────────────────────────────
    @check("a run description survives being saved and reloaded as JSON")
    def _():
        spec = RunSpec(model="resnet18", dataset="imagenet",
                       quant=QuantSpec(weight_bits=4, act_bits=6, bias_bits=8),
                       training=TrainerConfigV2(epochs=42, batch_size=256))
        assert RunSpec.from_json(spec.to_json()) == spec, "round-trip changed the spec"
        return f"{spec.experiment_name}, epochs=42 -> JSON -> identical"

    # ── 4. The normalization landmine ─────────────────────────────────────
    @check("mobilenetv1+imagenet resolves to mean=std=0.5, NOT ImageNet stats")
    def _():
        mean, std = resolve_normalization("mobilenetv1", "imagenet")
        assert mean == (0.5, 0.5, 0.5) and std == (0.5, 0.5, 0.5), (mean, std)
        other, _std = resolve_normalization("resnet18", "imagenet")
        assert other != mean, "resnet18 should NOT get the 0.5 stats"
        return f"mobilenetv1={mean}  resnet18={other}"

    @check("an untested model+dataset combination is refused, not silently guessed")
    def _():
        try:
            resolve_normalization("resnet18", "mnist")
        except RegistryError:
            return "resnet18+mnist rejected as expected"
        raise AssertionError("untested pair was NOT refused")

    # ── 5. Nonsense is rejected ───────────────────────────────────────────
    @check("invalid run descriptions are rejected with clear errors")
    def _():
        cases = {
            "unknown model": lambda: RunSpec(model="resnet99", dataset="imagenet"),
            "unknown dataset": lambda: RunSpec(model="resnet18", dataset="cifar100"),
            "nonsense bit width": lambda: RunSpec(model="resnet18", dataset="imagenet",
                                                  quant=QuantSpec(weight_bits=99)),
            "wrong augmentation": lambda: RunSpec(model="resnet18", dataset="imagenet",
                                                  augmentation="mnist_default"),
        }
        for label, build in cases.items():
            try:
                build()
            except (RunSpecError, RegistryError):
                continue
            raise AssertionError(f"{label} was NOT rejected")
        return f"all {len(cases)} bad descriptions rejected"

    # ── 6. The CLI still builds what it always did ────────────────────────
    @check("the train_imagenet_qat CLI still produces its original settings")
    def _():
        import dataclasses

        sys.argv = ["train_imagenet_qat.py", "--model", "resnet50",
                    "--weight-bits", "4", "--epochs", "80", "--batch-size", "512",
                    "--lr", "3e-4", "--cosine-lr"]
        with contextlib.redirect_stdout(io.StringIO()):
            from examples.train_imagenet_qat import parse_args, spec_from_args
            config = spec_from_args(parse_args()).training

        expected = {"experiment_name": "resnet50_W4_A8_B8", "epochs": 80,
                    "batch_size": 512, "learning_rate": 3e-4,
                    "reduce_lr_on_plateau": False, "grad_clip_norm": 1.0,
                    "num_classes": 1000}
        for field, want in expected.items():
            got = getattr(config, field)
            assert got == want, f"{field}: expected {want!r}, got {got!r}"
        assert len(dataclasses.fields(TrainerConfigV2)) > 20
        return f"{len(expected)} settings match the pre-refactor values"

    # ── 7. A real run, end to end ─────────────────────────────────────────
    if not os.path.isdir(os.path.join(REPO_ROOT, "data", "MNIST")):
        _results.append((False, "MNIST training run",
                         "data/MNIST not found — cannot run the training check"))
        return _report()

    workspace = tempfile.mkdtemp(prefix="verify_orchestration_")
    base_dir = os.path.join(workspace, "output")
    handle = None
    try:
        spec = RunSpec(
            model="mnist_cnn", dataset="mnist",
            output_dir=base_dir, data_dir=os.path.join(REPO_ROOT, "data"),
            training=TrainerConfigV2(
                epochs=2, batch_size=64, num_workers=0, device="cpu",
                dry_run=True, dry_run_batches=3, api_port=0, smoothing=0.0,
                qat=QATScheduleConfigV2(float_warmup_epochs=1, annealing_steps=4,
                                        quantization_start_gap=2),
            ),
        )

        print("  ... training a tiny MNIST run (a few seconds, output hidden)\n")
        noise = io.StringIO()
        with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
            handle = build_run(spec)
            tracker = handle.fit()

        @check("a run can be started from a description alone, and trains")
        def _():
            epochs = {s.epoch for s in tracker.history if "train_loss" in s.metrics}
            assert epochs == {0, 1}, f"expected epochs 0 and 1, trained {epochs}"
            return f"trained epochs {sorted(epochs)}"

        @check("the run's dashboard is reachable on a port the launcher knows")
        def _():
            assert isinstance(handle.api_port, int) and handle.api_port > 0
            url = f"{handle.dashboard_url}status"
            with urllib.request.urlopen(url, timeout=10) as response:
                status = json.load(response)
            assert status["run_id"] == spec.run_id, status["run_id"]
            return f"port {handle.api_port} answered /status"

        @check("the run wrote a manifest (run.json) describing itself")
        def _():
            with open(handle.manifest_path) as f:
                manifest = json.load(f)
            assert RunSpec.from_dict(manifest["spec"]) == spec, "manifest spec differs"
            assert manifest["api_port"] == handle.api_port
            return f"run.json holds the full spec + port {manifest['api_port']}"

        @check("latest.json points at the newest run")
        def _():
            with open(os.path.join(base_dir, "latest.json")) as f:
                pointer = json.load(f)
            assert pointer["run_id"] == spec.run_id
            assert os.path.exists(pointer["last_checkpoint"]), pointer["last_checkpoint"]
            return "pointer resolves to an existing last.pt"

        @check("everything the run wrote is inside its own private folder")
        def _():
            run_dir = os.path.abspath(handle.run_dir)
            assert run_dir.endswith(os.path.join("runs", spec.run_id)), run_dir
            for path in (handle.config.checkpoint_dir, handle.config.log_dir,
                         handle.config.plot_dir):
                assert os.path.abspath(path).startswith(run_dir), path
            return f"checkpoints, logs and plots all under runs/{spec.run_id}/"

        @check("a checkpoint knows which model, dataset and augmentation made it")
        def _():
            import torch

            path = os.path.join(handle.config.checkpoint_dir, "last.pt")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            provenance = RunSpec.from_dict(payload["extra"]["run_spec"])
            assert provenance == spec, "checkpoint spec differs from the run's spec"
            return (f"model={provenance.model} dataset={provenance.dataset} "
                    f"augmentation={provenance.augmentation} "
                    f"weights=W{provenance.quant.weight_bits}")

    finally:
        if handle is not None and handle.trainer.api_server is not None:
            with contextlib.suppress(Exception):
                handle.trainer.api_server.shutdown()
        shutil.rmtree(workspace, ignore_errors=True)

    return _report()


def _report() -> int:
    width = max(len(description) for _ok, description, _detail in _results)
    for ok, description, detail in _results:
        print(f"  [{'OK' if ok else 'FAIL'}] {description.ljust(width)}   {detail}")

    failed = [description for ok, description, _ in _results if not ok]
    print()
    print("=" * 72)
    if failed:
        print(f"  {len(failed)} of {len(_results)} checks FAILED:")
        for description in failed:
            print(f"    - {description}")
        print("\n  Re-run with VERIFY_TRACEBACK=1 for full tracebacks.")
        print("=" * 72)
        return 1
    print(f"  All {len(_results)} checks passed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
