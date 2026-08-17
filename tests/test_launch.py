"""
test_launch.py — the spec-file launcher and its terminal-status marker.

The failure-path tests are the point of this file. Before this module existed,
a crashed run and a cleanly finished run were byte-identical on disk, so
nothing could tell them apart after the process was gone. These tests assert
that every exit path now leaves a definitive marker, and that a failure still
exits nonzero so a supervising parent can trust Popen.returncode too.

MNIST here is plumbing only — no training-quality conclusions.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from orchestration.launch import (
    PHASE_BUILD,
    PHASE_LOAD,
    PHASE_TRAIN,
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_FILENAME,
    launch_from_spec_file,
    resolve_status_path,
)
from orchestration.run_spec import QuantSpec, RunSpec
from training_harness.config import CheckpointConfig, LoggingConfig
from training_harness.config_v2 import QATScheduleConfigV2, TrainerConfigV2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST_ROOT = os.path.join(REPO_ROOT, "data")

needs_mnist = pytest.mark.skipif(
    not os.path.isdir(os.path.join(MNIST_ROOT, "MNIST")),
    reason="MNIST data not vendored under data/",
)


def _mnist_spec(output_dir: str, run_id: str = "launch-test", **kwargs) -> RunSpec:
    return RunSpec(
        model="mnist_cnn",
        dataset="mnist",
        run_id=run_id,
        output_dir=output_dir,
        data_dir=MNIST_ROOT,
        quant=QuantSpec(weight_bits=8, act_bits=8, bias_bits=8),
        training=TrainerConfigV2(
            epochs=1,
            batch_size=32,
            num_workers=0,
            device="cpu",
            dry_run=True,
            dry_run_batches=2,
            api_port=None,          # no dashboard: proves status.json does not need one
            smoothing=0.0,
            logging=LoggingConfig(log_every_n_steps=1, save_plots=False),
            qat=QATScheduleConfigV2(float_warmup_epochs=1, annealing_steps=4,
                                    quantization_start_gap=2),
            checkpoint=CheckpointConfig(monitor_metric="val_acc", monitor_mode="max",
                                        top_k=2, save_last=True),
        ),
        **kwargs,
    )


def _write_spec(spec: RunSpec, tmp_path) -> str:
    return spec.write_json(str(tmp_path / "spec.json"))


def _read_status(spec: RunSpec) -> dict:
    with open(os.path.join(spec.run_dir, STATUS_FILENAME)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Status path cascade — where the marker lands when things go wrong early
# ---------------------------------------------------------------------------

def test_status_path_prefers_the_run_directory():
    spec = RunSpec(model="resnet18", dataset="imagenet",
                   run_id="RID", output_dir="output/base")
    path = resolve_status_path(spec=spec, spec_path="/tmp/spec.json")
    assert path == os.path.join(spec.run_dir, STATUS_FILENAME)


def test_status_path_falls_back_to_raw_json_fields():
    """A spec that parses but fails validation still names its run directory."""
    raw = {"model": "nonexistent", "dataset": "imagenet",
           "output_dir": "output/base", "run_id": "RID"}
    path = resolve_status_path(spec=None, raw=raw, spec_path="/tmp/spec.json")
    assert path == os.path.join("output/base", "runs", "RID", STATUS_FILENAME)


def test_status_path_falls_back_beside_the_spec_file():
    """Malformed JSON: nothing is known except where the spec file was."""
    path = resolve_status_path(spec=None, raw=None, spec_path="/tmp/spec.json")
    assert path == "/tmp/spec.json.status.json"


def test_status_path_is_none_when_nothing_is_known():
    assert resolve_status_path() is None


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

@needs_mnist
def test_successful_run_writes_a_finished_marker(tmp_path):
    spec = _mnist_spec(str(tmp_path / "out"))
    launch_from_spec_file(_write_spec(spec, tmp_path))

    assert os.path.exists(os.path.join(spec.run_dir, "run.json")), "manifest missing"
    status = _read_status(spec)

    assert status["status"] == STATUS_FINISHED
    assert status["phase"] == PHASE_TRAIN
    assert status["error"] is None
    assert status["finished_at"] and status["started_at"]
    assert status["duration_s"] >= 0
    assert status["run_id"] == "launch-test"
    assert status["epochs_completed"] == 1
    assert status["pid"] == os.getpid()


@needs_mnist
def test_finished_marker_records_the_best_checkpoint(tmp_path):
    """Recovering this from outside the process would need a two-file join;
    in-process the CheckpointManager already holds the ranked records."""
    spec = _mnist_spec(str(tmp_path / "out"), run_id="best-test")
    launch_from_spec_file(_write_spec(spec, tmp_path))

    best = _read_status(spec)["best"]
    assert best is not None
    assert best["metric"] == "val_acc"
    assert isinstance(best["value"], float)
    assert os.path.exists(best["checkpoint"])


@needs_mnist
def test_run_log_captures_output(tmp_path):
    """tee_stdout=True is forced by the launcher, so a run leaves a log even
    though build_run defaults it to False."""
    spec = _mnist_spec(str(tmp_path / "out"), run_id="log-test")
    launch_from_spec_file(_write_spec(spec, tmp_path))

    log_path = os.path.join(spec.run_dir, "run.log")
    assert os.path.exists(log_path)
    with open(log_path, encoding="utf-8", errors="replace") as f:
        assert "Experiment" in f.read()


# ---------------------------------------------------------------------------
# Failure paths — the reason this module exists
# ---------------------------------------------------------------------------

@needs_mnist
def test_training_failure_writes_a_failed_marker_and_reraises(tmp_path, monkeypatch):
    from training_harness.trainer_v2 import QATTrainerV2

    def _boom(self, **kwargs):
        raise RuntimeError("simulated training explosion")

    monkeypatch.setattr(QATTrainerV2, "fit", _boom)

    spec = _mnist_spec(str(tmp_path / "out"), run_id="fail-test")
    with pytest.raises(RuntimeError, match="simulated training explosion"):
        launch_from_spec_file(_write_spec(spec, tmp_path))

    status = _read_status(spec)
    assert status["status"] == STATUS_FAILED
    assert status["phase"] == PHASE_TRAIN
    assert status["error"]["type"] == "RuntimeError"
    assert "simulated training explosion" in status["error"]["message"]
    assert "Traceback" in status["error"]["traceback"]


def test_build_failure_writes_a_marker_with_build_phase(tmp_path):
    """A missing dataset root fails inside build_run, before any manifest is
    written -- which is exactly why runs must be enumerated by directory
    rather than by globbing run.json."""
    spec = RunSpec(
        model="resnet18", dataset="imagenet", run_id="build-fail",
        output_dir=str(tmp_path / "out"),
        data_dir=str(tmp_path / "definitely_not_a_dataset"),
        training=TrainerConfigV2(epochs=1, num_workers=0, device="cpu"),
    )
    with pytest.raises(BaseException):
        launch_from_spec_file(_write_spec(spec, tmp_path))

    status = _read_status(spec)
    assert status["status"] == STATUS_FAILED
    assert status["phase"] == PHASE_BUILD
    assert status["error"] is not None
    assert status["best"] is None and status["epochs_completed"] is None
    assert not os.path.exists(os.path.join(spec.run_dir, "run.json")), \
        "a build failure should have no manifest — that is the case being covered"


def test_invalid_spec_writes_a_marker_next_to_the_run_dir(tmp_path):
    """Parses as JSON, fails RunSpec validation: output_dir and run_id are
    still readable from the raw dict, so the marker lands in the run dir."""
    spec_path = str(tmp_path / "bad.json")
    output_dir = str(tmp_path / "out")
    with open(spec_path, "w") as f:
        json.dump({"model": "no_such_model", "dataset": "imagenet",
                   "output_dir": output_dir, "run_id": "bad-spec"}, f)

    with pytest.raises(BaseException):
        launch_from_spec_file(spec_path)

    expected = os.path.join(output_dir, "runs", "bad-spec", STATUS_FILENAME)
    with open(expected) as f:
        status = json.load(f)
    assert status["status"] == STATUS_FAILED
    assert status["phase"] == PHASE_LOAD
    assert "no_such_model" in status["error"]["message"]


def test_malformed_spec_writes_a_marker_beside_the_spec_file(tmp_path):
    """Not even valid JSON — the only thing known is where the file was."""
    spec_path = str(tmp_path / "broken.json")
    with open(spec_path, "w") as f:
        f.write("{ this is not json")

    with pytest.raises(BaseException):
        launch_from_spec_file(spec_path)

    with open(f"{spec_path}.status.json") as f:
        status = json.load(f)
    assert status["status"] == STATUS_FAILED
    assert status["phase"] == PHASE_LOAD


# ---------------------------------------------------------------------------
# Subprocess: both signals must agree — marker on disk AND nonzero exit
# ---------------------------------------------------------------------------

def _run_child(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": REPO_ROOT, "PYTHONUTF8": "1"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )


@needs_mnist
def test_subprocess_success_exits_zero_with_a_finished_marker(tmp_path):
    spec = _mnist_spec(str(tmp_path / "out"), run_id="subproc-ok")
    spec_path = _write_spec(spec, tmp_path)

    result = _run_child(f"""
        import sys
        from orchestration.launch import main
        sys.exit(main(["--spec", {spec_path!r}]))
    """)

    assert result.returncode == 0, f"stderr:\n{result.stderr[-3000:]}"
    assert _read_status(spec)["status"] == STATUS_FINISHED


@needs_mnist
def test_subprocess_failure_exits_nonzero_and_leaves_a_marker(tmp_path):
    """The test that proves an orchestrator can trust BOTH signals: the process
    exit code and the on-disk marker must agree that the run failed."""
    spec = _mnist_spec(str(tmp_path / "out"), run_id="subproc-fail")
    spec_path = _write_spec(spec, tmp_path)

    result = _run_child(f"""
        from training_harness.trainer_v2 import QATTrainerV2

        def _boom(self, **kwargs):
            raise RuntimeError("simulated training explosion")

        QATTrainerV2.fit = _boom

        from orchestration.launch import main
        main(["--spec", {spec_path!r}])
    """)

    # Signal 1: the process exited nonzero (re-raise preserved).
    assert result.returncode != 0, "launcher swallowed the exception"
    assert "simulated training explosion" in result.stderr

    # Signal 2: the marker on disk says the same thing.
    status = _read_status(spec)
    assert status["status"] == STATUS_FAILED
    assert status["phase"] == PHASE_TRAIN
    assert "simulated training explosion" in status["error"]["message"]
    assert "Traceback" in status["error"]["traceback"]


def test_missing_spec_file_exits_nonzero(tmp_path):
    result = _run_child(f"""
        from orchestration.launch import main
        main(["--spec", {str(tmp_path / "nope.json")!r}])
    """)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
