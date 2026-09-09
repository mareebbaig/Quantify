"""
test_orchestrator_e2e.py — POST a spec, watch a real run go from queued to finished.

The one test that exercises every layer at once: the HTTP API, the queue
worker, a real subprocess launched through the launch_run seam, the bound-port
readback, and the classification of the finished run from disk.

MNIST, two dry-run batches. Plumbing only.
"""

import os
import time

import pytest

from orchestration.service.app import create_app
from orchestration.service.queue import RunQueue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST_ROOT = os.path.join(REPO_ROOT, "data")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(MNIST_ROOT, "MNIST")),
    reason="MNIST data not vendored under data/",
)


def _spec_body(output_dir: str, run_id: str) -> dict:
    return {
        "model": "mnist_cnn",
        "dataset": "mnist",
        "run_id": run_id,
        "output_dir": output_dir,
        "data_dir": MNIST_ROOT,
        "quant": {"weight_bits": 8, "act_bits": 8, "bias_bits": 8},
        "training": {
            "epochs": 1, "batch_size": 32, "num_workers": 0, "device": "cpu",
            "dry_run": True, "dry_run_batches": 2, "api_port": 0,
            "smoothing": 0.0, "num_classes": 10,
            "logging": {"save_plots": False, "log_every_n_steps": 1},
            "qat": {"float_warmup_epochs": 1, "annealing_steps": 4,
                    "quantization_start_gap": 2},
        },
    }


def test_post_a_spec_and_watch_it_run(tmp_path):
    output_dir = str(tmp_path / "out")
    queue = RunQueue(str(tmp_path / "state"), poll_interval_s=0.2)
    app = create_app(roots=[output_dir], run_queue=queue, start_worker=True)
    app.config["TESTING"] = True
    client = app.test_client()

    try:
        response = client.post("/api/v1/queue", json=_spec_body(output_dir, "e2e"))
        assert response.status_code == 202
        entry_id = response.get_json()["entry_id"]

        # It should be seen running, with a discovered dashboard port, before
        # it finishes -- that is the launch path working end to end.
        saw_running = False
        port = None
        deadline = time.time() + 300
        while time.time() < deadline:
            entry = queue.get_entry(entry_id)
            if entry.state == "running":
                saw_running = True
                port = port or entry.api_port
            if entry.state not in {"queued", "launching", "running", "stopping"}:
                break
            time.sleep(0.2)
        else:
            pytest.fail(f"run never reached a terminal state: {queue.snapshot()}")

        entry = queue.get_entry(entry_id)
        assert saw_running, "run went terminal without ever being seen as running"
        assert entry.state == "finished", f"unexpected end state: {entry.to_dict()}"
        assert entry.returncode == 0
        assert isinstance(port, int) and port > 0, "bound dashboard port was never recovered"

        # And the API now lists it as a finished run, classified from disk.
        runs = {r["run_id"]: r for r in client.get("/api/v1/runs").get_json()["runs"]}
        assert runs["e2e"]["state"] == "finished"
        assert runs["e2e"]["model"] == "mnist_cnn"

        detail = client.get("/api/v1/runs/e2e").get_json()
        assert detail["status"]["status"] == "finished"
        assert detail["manifest"]["spec"]["model"] == "mnist_cnn"
        assert detail["best_value"] is not None

        # The run's own directory carries everything the docs promise.
        run_dir = os.path.join(output_dir, "runs", "e2e")
        for name in ("run.json", "status.json", "spec.json", "run.log", "launcher.log"):
            assert os.path.exists(os.path.join(run_dir, name)), f"missing {name}"
    finally:
        queue.stop()


def test_relaunch_a_finished_run_end_to_end(tmp_path):
    """The repertoire flow: finish a run, then queue a descendant from its spec."""
    output_dir = str(tmp_path / "out")
    queue = RunQueue(str(tmp_path / "state"), poll_interval_s=0.2)
    app = create_app(roots=[output_dir], run_queue=queue, start_worker=True)
    app.config["TESTING"] = True
    client = app.test_client()

    try:
        client.post("/api/v1/queue", json=_spec_body(output_dir, "ancestor"))
        deadline = time.time() + 300
        while time.time() < deadline:
            if not queue.snapshot()["pending"] and not queue.snapshot()["running"]:
                break
            time.sleep(0.2)
        else:
            pytest.fail("ancestor run did not finish")

        checkpoint = os.path.join(output_dir, "runs", "ancestor", "checkpoints", "last.pt")
        assert os.path.exists(checkpoint)

        response = client.post("/api/v1/runs/ancestor/relaunch",
                               json={"init_checkpoint": checkpoint, "epochs": 1})
        assert response.status_code == 202
        child = response.get_json()
        assert child["run_id"] != "ancestor"

        spec = queue.get_entry(child["entry_id"]).spec
        assert spec["init_checkpoint"] == checkpoint
        assert spec["model"] == "mnist_cnn"
    finally:
        queue.stop()
