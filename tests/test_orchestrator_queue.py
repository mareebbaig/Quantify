"""
test_orchestrator_queue.py — the one-worker queue, launching, and stopping.

Two of these matter more than the rest and are deliberately strict:

  * test_three_runs_execute_strictly_one_at_a_time  — real MNIST subprocesses,
    asserting the GPU is never double-booked. Serialisation is a correctness
    requirement here (one GPU, and a process-global QuantizerManager), not a
    preference.
  * test_reconcile_*                                — an orchestrator restart
    must not lose the queue, must not orphan a live run, and must never
    relaunch something blindly.
"""

import json
import os
import subprocess
import sys
import time

import pytest

from orchestration.run_spec import QuantSpec, RunSpec
from orchestration.service import store
from orchestration.service.launcher import launch_run, prepare_spec
from orchestration.service.process import RunProcess, process_alive
from orchestration.service.queue import (
    STATE_CANCELLED,
    STATE_QUEUED,
    STATE_RUNNING,
    QueueEntry,
    RunQueue,
)
from training_harness.config import CheckpointConfig, LoggingConfig
from training_harness.config_v2 import QATScheduleConfigV2, TrainerConfigV2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST_ROOT = os.path.join(REPO_ROOT, "data")

needs_mnist = pytest.mark.skipif(
    not os.path.isdir(os.path.join(MNIST_ROOT, "MNIST")),
    reason="MNIST data not vendored under data/",
)


def mnist_spec(output_dir: str, run_id: str, *, epochs: int = 1) -> RunSpec:
    """A deliberately tiny run: this exercises plumbing, not training."""
    return RunSpec(
        model="mnist_cnn", dataset="mnist", run_id=run_id,
        output_dir=str(output_dir), data_dir=MNIST_ROOT,
        quant=QuantSpec(weight_bits=8, act_bits=8, bias_bits=8),
        training=TrainerConfigV2(
            epochs=epochs, batch_size=32, num_workers=0, device="cpu",
            dry_run=True, dry_run_batches=2, api_port=0, smoothing=0.0,
            logging=LoggingConfig(log_every_n_steps=1, save_plots=False),
            qat=QATScheduleConfigV2(float_warmup_epochs=1, annealing_steps=4,
                                    quantization_start_gap=2),
            checkpoint=CheckpointConfig(monitor_metric="val_acc", monitor_mode="max",
                                        top_k=1, save_last=True),
        ),
    )


def drain(queue: RunQueue, timeout_s: float = 300.0, poll_s: float = 0.5) -> None:
    """Block until nothing is queued or running."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        snapshot = queue.snapshot()
        if not snapshot["pending"] and not snapshot["running"]:
            return
        time.sleep(poll_s)
    raise AssertionError(f"queue did not drain within {timeout_s}s: {queue.snapshot()}")


# ---------------------------------------------------------------------------
# Launch seam
# ---------------------------------------------------------------------------

def test_prepare_spec_forces_a_free_port_and_an_absolute_path(tmp_path):
    """Every launched run must get an OS-assigned port (so queued runs cannot
    collide) and an absolute output dir (the child runs with cwd=repo root)."""
    spec = mnist_spec(tmp_path / "out", "prep")
    spec.training.api_port = 8765
    spec.output_dir = "output/relative"

    prepared = prepare_spec(spec)

    assert prepared.training.api_port == 0
    assert os.path.isabs(prepared.output_dir)
    # The caller's spec is untouched.
    assert spec.training.api_port == 8765
    assert spec.output_dir == "output/relative"


@needs_mnist
def test_launch_run_spawns_a_supervised_child(tmp_path):
    process = launch_run(mnist_spec(tmp_path / "out", "spawn"))
    try:
        assert process.pid > 0
        assert os.path.exists(os.path.join(process.run_dir, "spec.json"))
        # The bound port is recovered from the child's manifest, not stdout.
        assert isinstance(process.api_port, int) and process.api_port > 0
        assert os.path.exists(os.path.join(process.run_dir, "run.json"))
        assert process.wait(timeout=300) == 0
    finally:
        if process.is_running():
            process.kill()

    # Raw child output and the teed training log are separate files on purpose.
    assert os.path.exists(os.path.join(process.run_dir, "launcher.log"))
    assert os.path.exists(os.path.join(process.run_dir, "run.log"))
    assert store.classify_run_dir(process.run_dir).state == store.STATE_FINISHED


@needs_mnist
def test_launcher_log_is_not_double_written(tmp_path):
    """run.log is teed by the run itself; pointing the child's stdout at the
    same file would duplicate every line. They must stay distinct files."""
    process = launch_run(mnist_spec(tmp_path / "out", "logsplit"))
    process.wait(timeout=300)

    with open(os.path.join(process.run_dir, "run.log"), encoding="utf-8",
              errors="replace") as f:
        run_log = f.read()

    banner_count = run_log.count("  Run dir    :")
    assert banner_count == 1, f"run.log shows the banner {banner_count}x (duplicated tee)"


# ---------------------------------------------------------------------------
# Queue mechanics (fake launcher — fast, deterministic)
# ---------------------------------------------------------------------------

class _FakeProcess(RunProcess):
    """A process we can end on command, to drive queue transitions."""

    def __init__(self, run_dir):
        super().__init__(pid=os.getpid(), run_dir=str(run_dir), api_port=1234)
        self._done = False

    def poll(self):
        return 0 if self._done else None

    def finish(self, status="finished"):
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "status.json"), "w") as f:
            json.dump({"status": status, "phase": "train"}, f)
        self._done = True


def test_queue_runs_one_entry_at_a_time(tmp_path):
    launched = []

    def fake_launch(spec):
        process = _FakeProcess(spec.run_dir)
        launched.append(process)
        return process

    queue = RunQueue(str(tmp_path / "state"), poll_interval_s=0.05,
                     launch_fn=fake_launch)
    for i in range(3):
        queue.enqueue(mnist_spec(tmp_path / "out", f"q{i}"))
    queue.start()
    try:
        time.sleep(0.5)
        assert len(launched) == 1, "worker started more than one run"
        assert len(queue.snapshot()["pending"]) == 2

        launched[0].finish()
        time.sleep(1.0)
        assert len(launched) == 2, "worker did not advance after the first finished"
        assert len(queue.snapshot()["pending"]) == 1

        launched[1].finish()
        time.sleep(1.0)
        assert len(launched) == 3
    finally:
        for process in launched:
            process.finish()
        queue.stop()


def test_cancel_removes_a_pending_entry(tmp_path):
    queue = RunQueue(str(tmp_path / "state"), launch_fn=lambda spec: None)
    first = queue.enqueue(mnist_spec(tmp_path / "out", "c1"))
    second = queue.enqueue(mnist_spec(tmp_path / "out", "c2"))

    assert queue.cancel(second.entry_id).state == STATE_CANCELLED
    assert [e["entry_id"] for e in queue.snapshot()["pending"]] == [first.entry_id]

    with pytest.raises(KeyError):
        queue.cancel("nope")


def test_cancelling_a_running_entry_is_refused(tmp_path):
    queue = RunQueue(str(tmp_path / "state"), launch_fn=lambda spec: None)
    entry = queue.enqueue(mnist_spec(tmp_path / "out", "c3"))
    entry.state = STATE_RUNNING

    with pytest.raises(ValueError, match="use stop"):
        queue.cancel(entry.entry_id)


def test_a_failed_launch_marks_the_entry_failed(tmp_path):
    def explode(spec):
        raise RuntimeError("no GPU today")

    queue = RunQueue(str(tmp_path / "state"), poll_interval_s=0.05, launch_fn=explode)
    queue.enqueue(mnist_spec(tmp_path / "out", "boom"))
    queue.start()
    try:
        drain(queue, timeout_s=20)
    finally:
        queue.stop()

    history = queue.snapshot()["history"]
    assert history[0]["state"] == store.STATE_FAILED
    assert "no GPU today" in history[0]["error"]


def test_queue_state_survives_a_restart(tmp_path):
    state_dir = str(tmp_path / "state")
    queue = RunQueue(state_dir, launch_fn=lambda spec: None)
    queue.enqueue(mnist_spec(tmp_path / "out", "persist1"))
    queue.enqueue(mnist_spec(tmp_path / "out", "persist2"))

    reloaded = RunQueue(state_dir, launch_fn=lambda spec: None)
    assert [e["run_id"] for e in reloaded.snapshot()["pending"]] == ["persist1", "persist2"]


# ---------------------------------------------------------------------------
# Reconciliation — what happens when the orchestrator restarts
# ---------------------------------------------------------------------------

def _seed_entry(queue: RunQueue, run_dir, *, state, pid, run_id) -> QueueEntry:
    entry = QueueEntry(entry_id=run_id, spec={"model": "mnist_cnn", "dataset": "mnist"},
                       state=state, run_id=run_id, run_dir=str(run_dir), pid=pid)
    queue._entries.append(entry)
    queue._save()
    return entry


def test_reconcile_marks_a_dead_run_with_no_marker_as_crashed(tmp_path):
    queue = RunQueue(str(tmp_path / "state"), launch_fn=lambda spec: None)
    run_dir = tmp_path / "out" / "runs" / "dead"
    os.makedirs(run_dir)
    _seed_entry(queue, run_dir, state=STATE_RUNNING, pid=999_999_998, run_id="dead")

    notes = queue.reconcile()

    entry = queue.get_entry("dead")
    assert entry.state == store.STATE_CRASHED
    assert "no status marker" in entry.note
    assert any("crashed" in n for n in notes)


def test_reconcile_recovers_the_terminal_status_from_the_marker(tmp_path):
    queue = RunQueue(str(tmp_path / "state"), launch_fn=lambda spec: None)
    run_dir = tmp_path / "out" / "runs" / "ended"
    os.makedirs(run_dir)
    with open(run_dir / "status.json", "w") as f:
        json.dump({"status": "finished", "phase": "train"}, f)
    _seed_entry(queue, run_dir, state=STATE_RUNNING, pid=999_999_998, run_id="ended")

    queue.reconcile()

    assert queue.get_entry("ended").state == store.STATE_FINISHED


def test_reconcile_adopts_a_run_that_is_still_alive(tmp_path):
    """The orchestrator restarted while training continued: pick it back up
    rather than orphaning it or declaring it dead."""
    queue = RunQueue(str(tmp_path / "state"), launch_fn=lambda spec: None)
    run_dir = tmp_path / "out" / "runs" / "alive"
    os.makedirs(run_dir)
    _seed_entry(queue, run_dir, state=STATE_RUNNING, pid=os.getpid(), run_id="alive")

    queue.reconcile()

    entry = queue.get_entry("alive")
    assert entry.state == STATE_RUNNING
    assert entry.adopted is True
    assert "adopted" in entry.note


def test_reconcile_never_relaunches_and_keeps_the_queue(tmp_path):
    """Restarting must not silently re-spend GPU hours, and must not drop
    pending work."""
    launches = []
    queue = RunQueue(str(tmp_path / "state"),
                     launch_fn=lambda spec: launches.append(spec))
    run_dir = tmp_path / "out" / "runs" / "midflight"
    os.makedirs(run_dir)
    _seed_entry(queue, run_dir, state=STATE_RUNNING, pid=999_999_998, run_id="midflight")
    queue.enqueue(mnist_spec(tmp_path / "out", "still-pending"))

    queue.reconcile()

    assert launches == [], "reconcile relaunched a run on its own"
    assert queue.get_entry("midflight").state == store.STATE_CRASHED
    pending = queue.snapshot()["pending"]
    assert [e["run_id"] for e in pending] == ["still-pending"]


@needs_mnist
def test_restart_reconciliation_against_a_real_killed_run(tmp_path):
    """End to end: launch a real child, kill it, and prove a fresh orchestrator
    classifies it as crashed and carries the rest of the queue forward."""
    state_dir = str(tmp_path / "state")
    queue = RunQueue(state_dir, poll_interval_s=0.05)

    spec = mnist_spec(tmp_path / "out", "killed", epochs=40)
    process = launch_run(spec)
    _seed_entry(queue, process.run_dir, state=STATE_RUNNING, pid=process.pid,
                run_id="killed")
    queue.enqueue(mnist_spec(tmp_path / "out", "survivor"))

    process.kill()
    process.wait(timeout=60)
    assert not process_alive(process.pid)

    # A brand-new orchestrator, reading only the persisted state file.
    restarted = RunQueue(state_dir, launch_fn=lambda spec: None)
    restarted.reconcile()

    assert restarted.get_entry("killed").state == store.STATE_CRASHED
    assert [e["run_id"] for e in restarted.snapshot()["pending"]] == ["survivor"]


# ---------------------------------------------------------------------------
# The real thing: three MNIST runs, strictly serialised
# ---------------------------------------------------------------------------

@needs_mnist
def test_three_runs_execute_strictly_one_at_a_time(tmp_path):
    """Real subprocesses. Two training processes must never overlap: one GPU,
    and a process-global QuantizerManager."""
    queue = RunQueue(str(tmp_path / "state"), poll_interval_s=0.2)
    run_ids = ["serial-a", "serial-b", "serial-c"]
    for run_id in run_ids:
        queue.enqueue(mnist_spec(tmp_path / "out", run_id))

    overlaps = []
    order = []

    queue.start()
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            snapshot = queue.snapshot()
            running = snapshot["running"]
            if len(running) > 1:
                overlaps.append([e["run_id"] for e in running])
            for entry in running:
                if entry["run_id"] and entry["run_id"] not in order:
                    order.append(entry["run_id"])
            if not snapshot["pending"] and not running:
                break
            time.sleep(0.2)
        else:
            pytest.fail(f"queue did not drain: {queue.snapshot()}")
    finally:
        queue.stop()

    assert overlaps == [], f"runs overlapped: {overlaps}"
    assert order == run_ids, f"ran out of order: {order}"

    history = {e["run_id"]: e for e in queue.snapshot()["history"]}
    for run_id in run_ids:
        assert history[run_id]["state"] == store.STATE_FINISHED
        assert history[run_id]["returncode"] == 0
        run_dir = os.path.join(str(tmp_path / "out"), "runs", run_id)
        assert store.classify_run_dir(run_dir).state == store.STATE_FINISHED


@needs_mnist
def test_stop_frees_the_worker_and_starts_the_next_run(tmp_path):
    """Stopping is only complete when the process is gone -- and only then may
    the next run start."""
    queue = RunQueue(str(tmp_path / "state"), poll_interval_s=0.2)
    first = queue.enqueue(mnist_spec(tmp_path / "out", "stop-me", epochs=200))
    queue.enqueue(mnist_spec(tmp_path / "out", "next-up"))

    queue.start()
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            entry = queue.get_entry(first.entry_id)
            if entry.state == STATE_RUNNING and entry.pid:
                break
            time.sleep(0.2)
        else:
            pytest.fail("first run never started")

        # Short halt window so the ladder escalates promptly in a test.
        queue.request_stop(first.entry_id, halt_timeout_s=5.0)
        assert queue.get_entry(first.entry_id).stop_stage in {"halt", "sigterm"}

        drain(queue, timeout_s=300)
    finally:
        queue.stop()

    history = {e["run_id"]: e for e in queue.snapshot()["history"]}
    assert "stop-me" in history and "next-up" in history
    assert history["next-up"]["state"] == store.STATE_FINISHED, \
        "the worker did not free after the stop"
