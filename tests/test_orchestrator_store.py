"""
test_orchestrator_store.py — run discovery and state classification.

The classification rules are the contract in docs/llm/RUNSPEC_AND_BUILD_RUN.md:

    status.json present  -> authoritative
    absent + pid alive   -> running
    absent + pid dead    -> crashed

Every class a run can be in is seeded here, including the two that a naive
implementation loses: a build failure (status.json but NO run.json, which a
manifest glob would skip entirely) and a crash (no marker at all).
"""

import json
import os

import pytest

from orchestration.service import store
from orchestration.service.process import process_alive

DEAD_PID = 999_999_998  # implausible; asserted dead below


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _manifest(pid: int, run_id: str = "r", **extra) -> dict:
    manifest = {
        "run_id": run_id,
        "experiment_name": "resnet18_W8_A8_B8",
        "created_at": "2026-08-18T10:00:00",
        "pid": pid,
        "api_port": 8765,
        "api_host": "127.0.0.1",
        "dashboard_url": "http://127.0.0.1:8765/api/v1/",
        "spec": {
            "model": "resnet18", "dataset": "imagenet",
            "augmentation": "imagenet_default",
            "quant": {"weight_bits": 8, "act_bits": 8, "bias_bits": 8},
            "training": {"epochs": 90},
        },
    }
    manifest.update(extra)
    return manifest


def _seed(base, run_id: str, *, manifest: dict = None, status: dict = None,
          spec: dict = None) -> str:
    run_dir = os.path.join(str(base), "imagenet_qat_resnet18", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    if manifest is not None:
        _write(os.path.join(run_dir, "run.json"), manifest)
    if status is not None:
        _write(os.path.join(run_dir, "status.json"), status)
    if spec is not None:
        _write(os.path.join(run_dir, "spec.json"), spec)
    return run_dir


def test_dead_pid_constant_is_actually_dead():
    """Guards the fixtures below: if this pid existed, 'crashed' tests would lie."""
    assert not process_alive(DEAD_PID)


# ---------------------------------------------------------------------------
# The classification table, case by case
# ---------------------------------------------------------------------------

def test_finished_run(tmp_path):
    run_dir = _seed(tmp_path, "finished", manifest=_manifest(DEAD_PID), status={
        "status": "finished", "phase": "train",
        "started_at": "2026-08-18T10:00:00", "finished_at": "2026-08-18T12:00:00",
        "duration_s": 7200.0, "epochs_completed": 90,
        "best": {"metric": "val_acc", "value": 0.7213, "epoch": 74},
        "error": None,
    })
    record = store.classify_run_dir(run_dir)

    assert record.state == store.STATE_FINISHED
    assert record.best_metric == "val_acc" and record.best_value == 0.7213
    assert record.epochs_completed == 90
    assert record.model == "resnet18" and record.dataset == "imagenet"
    assert record.quant == "W8 A8 B8"


def test_status_marker_wins_over_a_live_pid(tmp_path):
    """A marker is authoritative even if the pid is somehow still alive --
    e.g. pid reuse, or a lingering parent."""
    run_dir = _seed(tmp_path, "marked", manifest=_manifest(os.getpid()),
                    status={"status": "finished", "phase": "train"})
    assert store.classify_run_dir(run_dir).state == store.STATE_FINISHED


def test_failed_run_carries_the_traceback(tmp_path):
    run_dir = _seed(tmp_path, "failed", manifest=_manifest(DEAD_PID), status={
        "status": "failed", "phase": "train",
        "error": {"type": "RuntimeError", "message": "CUDA out of memory",
                  "traceback": "Traceback (most recent call last): ..."},
    })
    record = store.classify_run_dir(run_dir)

    assert record.state == store.STATE_FAILED
    assert record.error["type"] == "RuntimeError"
    assert "Traceback" in record.error["traceback"]


def test_interrupted_run(tmp_path):
    run_dir = _seed(tmp_path, "interrupted", manifest=_manifest(DEAD_PID),
                    status={"status": "interrupted", "phase": "train"})
    assert store.classify_run_dir(run_dir).state == store.STATE_INTERRUPTED


def test_build_failure_has_a_marker_but_no_manifest(tmp_path):
    """The case a run.json glob would silently drop."""
    run_dir = _seed(
        tmp_path, "buildfail",
        status={"status": "failed", "phase": "build", "pid": DEAD_PID,
                "started_at": "2026-08-18T10:00:00",
                "error": {"type": "RegistryError", "message": "dataset needs a data directory",
                          "traceback": "..."}},
        spec={"model": "resnet50", "dataset": "imagenet",
              "augmentation": "imagenet_default",
              "quant": {"weight_bits": 4, "act_bits": 8, "bias_bits": 8},
              "training": {"epochs": 12}},
    )
    record = store.classify_run_dir(run_dir)

    assert record.state == store.STATE_FAILED
    assert record.has_status and not record.has_manifest
    # Identity recovered from spec.json, which the launcher writes before starting.
    assert record.model == "resnet50"
    assert record.quant == "W4 A8 B8"


def test_crashed_run_has_a_manifest_and_a_dead_pid(tmp_path):
    run_dir = _seed(tmp_path, "crashed", manifest=_manifest(DEAD_PID))
    record = store.classify_run_dir(run_dir)

    assert record.state == store.STATE_CRASHED
    assert "not running" in record.reason


def test_running_run_has_a_manifest_and_a_live_pid(tmp_path):
    """Uses this process's pid: a real live process, not a mocked one."""
    run_dir = _seed(tmp_path, "running", manifest=_manifest(os.getpid()))
    record = store.classify_run_dir(run_dir)

    assert record.state == store.STATE_RUNNING
    assert record.pid == os.getpid()
    assert record.dashboard_url == "http://127.0.0.1:8765/api/v1/"


def test_empty_run_dir_is_crashed(tmp_path):
    """Died between mkdir and the manifest write -- nothing to go on."""
    run_dir = _seed(tmp_path, "empty")
    record = store.classify_run_dir(run_dir)

    assert record.state == store.STATE_CRASHED
    assert "no manifest" in record.reason


def test_pid_reuse_is_blunted_by_created_at(tmp_path):
    """A live pid whose process predates the run belongs to something else.

    Only enforceable where process start times are readable (Linux /proc);
    elsewhere the check degrades to plain liveness and this run stays 'running'.
    """
    from orchestration.service.process import process_start_time

    run_dir = _seed(tmp_path, "reused",
                    manifest=_manifest(os.getpid(), created_at="2099-01-01T00:00:00"))
    record = store.classify_run_dir(run_dir)

    if process_start_time(os.getpid()) is None:
        assert record.state == store.STATE_RUNNING   # cannot tell; documented
    else:
        assert record.state == store.STATE_CRASHED   # pid predates the run


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discovery_finds_every_class_and_sorts_newest_first(tmp_path):
    _seed(tmp_path, "a", manifest=_manifest(DEAD_PID, created_at="2026-08-18T09:00:00"),
          status={"status": "finished"})
    _seed(tmp_path, "b", manifest=_manifest(DEAD_PID, created_at="2026-08-18T11:00:00"))
    _seed(tmp_path, "c", status={"status": "failed", "phase": "build"})
    _seed(tmp_path, "d", manifest=_manifest(os.getpid(), created_at="2026-08-18T10:00:00"))

    records = store.discover_runs([str(tmp_path)])
    by_id = {r.run_id: r.state for r in records}

    assert by_id == {
        "a": store.STATE_FINISHED,
        "b": store.STATE_CRASHED,
        "c": store.STATE_FAILED,
        "d": store.STATE_RUNNING,
    }
    stamps = [r.created_at for r in records if r.created_at]
    assert stamps == sorted(stamps, reverse=True), "newest first"


def test_discovery_handles_a_root_that_is_itself_an_output_dir(tmp_path):
    run_dir = os.path.join(str(tmp_path), "runs", "direct")
    os.makedirs(run_dir)
    _write(os.path.join(run_dir, "run.json"), _manifest(DEAD_PID, run_id="direct"))

    records = store.discover_runs([str(tmp_path)])
    assert [r.run_id for r in records] == ["direct"]


def test_discovery_ignores_unrelated_directories(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "some_experiment", "checkpoints"))
    os.makedirs(os.path.join(str(tmp_path), "logs"))
    assert store.discover_runs([str(tmp_path)]) == []


def test_discovery_tolerates_a_missing_root(tmp_path):
    assert store.discover_runs([str(tmp_path / "nope")]) == []


def test_find_run_by_id(tmp_path):
    _seed(tmp_path, "wanted", manifest=_manifest(DEAD_PID, run_id="wanted"),
          status={"status": "finished"})
    _seed(tmp_path, "other", manifest=_manifest(DEAD_PID, run_id="other"))

    assert store.find_run([str(tmp_path)], "wanted").state == store.STATE_FINISHED
    assert store.find_run([str(tmp_path)], "absent") is None


def test_corrupt_json_does_not_break_the_scan(tmp_path):
    """A half-written manifest must degrade, not crash the listing."""
    run_dir = _seed(tmp_path, "corrupt")
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        f.write("{ not json")

    record = store.classify_run_dir(run_dir)
    assert record.state == store.STATE_CRASHED
    assert not record.has_manifest


def test_live_status_returns_none_without_a_port(tmp_path):
    run_dir = _seed(tmp_path, "noport", manifest=_manifest(os.getpid(), api_port=None))
    assert store.fetch_live_status(store.classify_run_dir(run_dir)) is None
