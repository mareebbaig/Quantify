"""
test_orchestrator_api.py — the orchestrator's REST API and UI routes.

Driven through Flask's test client with the worker disabled, so these are fast
and deterministic: the queue mechanics themselves are covered in
test_orchestrator_queue.py.
"""

import json
import os

import pytest

from orchestration.service.app import create_app
from orchestration.service.queue import RunQueue

DEAD_PID = 999_999_998


@pytest.fixture
def app(tmp_path):
    """An orchestrator over an empty root, with the worker thread off."""
    root = tmp_path / "root"
    root.mkdir()
    queue = RunQueue(str(tmp_path / "state"), launch_fn=lambda spec: None)
    application = create_app(roots=[str(root)], run_queue=queue, start_worker=False)
    application.config["TESTING"] = True
    application.config["_ROOT"] = root
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def seed_run(root, run_id, *, manifest=None, status=None):
    run_dir = root / "base" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (run_dir / "run.json").write_text(json.dumps(manifest))
    if status is not None:
        (run_dir / "status.json").write_text(json.dumps(status))
    return run_dir


def a_manifest(run_id="r1", pid=DEAD_PID, **extra):
    manifest = {
        "run_id": run_id, "experiment_name": "mnist_cnn_W8_A8_B8",
        "created_at": "2026-08-18T10:00:00", "pid": pid,
        "api_port": None, "api_host": "127.0.0.1", "dashboard_url": None,
        "spec": {
            "model": "mnist_cnn", "dataset": "mnist",
            "augmentation": "mnist_default",
            "quant": {"weight_bits": 8, "act_bits": 8, "bias_bits": 8},
            "run_id": run_id, "output_dir": "output/mnist",
            "training": {"epochs": 3, "num_classes": 10},
        },
    }
    manifest.update(extra)
    return manifest


def a_spec_body(**overrides):
    body = {
        "model": "mnist_cnn", "dataset": "mnist",
        "quant": {"weight_bits": 8, "act_bits": 8, "bias_bits": 8},
        "training": {"epochs": 2, "batch_size": 32, "num_classes": 10},
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------

def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def test_registry_lists_real_ingredients(client):
    """The launch form is driven by this, so it must reflect the registry
    rather than a hardcoded copy."""
    data = client.get("/api/v1/registry").get_json()

    models = {m["name"] for m in data["models"]}
    assert {"resnet18", "resnet50", "mobilenetv1", "mobilenetv2", "mnist_cnn"} <= models
    assert {d["name"] for d in data["datasets"]} == {"imagenet", "mnist"}
    mnist = next(m for m in data["models"] if m["name"] == "mnist_cnn")
    assert mnist["datasets"] == ["mnist"]


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_runs_list_is_empty_initially(client):
    assert client.get("/api/v1/runs").get_json()["runs"] == []


def test_runs_list_classifies_each_run(app, client):
    root = app.config["_ROOT"]
    seed_run(root, "done", manifest=a_manifest("done"),
             status={"status": "finished", "best": {"metric": "val_acc", "value": 0.91}})
    seed_run(root, "dead", manifest=a_manifest("dead"))
    seed_run(root, "live", manifest=a_manifest("live", pid=os.getpid()))

    runs = {r["run_id"]: r for r in client.get("/api/v1/runs").get_json()["runs"]}

    assert runs["done"]["state"] == "finished"
    assert runs["done"]["best_value"] == 0.91
    assert runs["dead"]["state"] == "crashed"
    assert runs["live"]["state"] == "running"


def test_run_detail_includes_manifest_and_status(app, client):
    root = app.config["_ROOT"]
    seed_run(root, "r1", manifest=a_manifest("r1"),
             status={"status": "failed", "phase": "train",
                     "error": {"type": "RuntimeError", "message": "boom",
                               "traceback": "Traceback..."}})

    data = client.get("/api/v1/runs/r1").get_json()

    assert data["state"] == "failed"
    assert data["manifest"]["run_id"] == "r1"
    assert data["status"]["error"]["type"] == "RuntimeError"


def test_unknown_run_is_404(client):
    assert client.get("/api/v1/runs/nope").status_code == 404
    assert client.get("/api/v1/runs/nope/live").status_code == 404


def test_live_endpoint_503s_without_a_reachable_dashboard(app, client):
    root = app.config["_ROOT"]
    seed_run(root, "noport", manifest=a_manifest("noport", pid=os.getpid()))

    response = client.get("/api/v1/runs/noport/live")
    assert response.status_code == 503
    assert response.get_json()["state"] == "running"


# ---------------------------------------------------------------------------
# Enqueue + validation
# ---------------------------------------------------------------------------

def test_enqueue_accepts_a_valid_spec(client):
    response = client.post("/api/v1/queue", json=a_spec_body())
    assert response.status_code == 202

    entry = response.get_json()
    assert entry["state"] == "queued"
    assert entry["model"] == "mnist_cnn"
    assert client.get("/api/v1/queue").get_json()["pending"][0]["entry_id"] == entry["entry_id"]


def test_enqueue_rejects_an_unknown_model(client):
    response = client.post("/api/v1/queue", json=a_spec_body(model="resnet99"))
    assert response.status_code == 400
    assert "unknown model" in response.get_json()["error"]


def test_enqueue_rejects_an_untested_pair(client):
    """RunSpec's own refusal, surfaced through the API rather than duplicated:
    a wrong (model, dataset) normalization trains silently and badly."""
    body = a_spec_body(model="resnet18", dataset="mnist")
    response = client.post("/api/v1/queue", json=body)

    assert response.status_code == 400
    assert "untested pair" in response.get_json()["error"]


def test_enqueue_rejects_bad_bit_widths(client):
    body = a_spec_body(quant={"weight_bits": 99, "act_bits": 8, "bias_bits": 8})
    response = client.post("/api/v1/queue", json=body)
    assert response.status_code == 400


def test_enqueue_rejects_an_empty_body(client):
    assert client.post("/api/v1/queue", json={}).status_code == 400


def test_queue_snapshot_shape(client):
    client.post("/api/v1/queue", json=a_spec_body())
    data = client.get("/api/v1/queue").get_json()

    assert set(data) == {"running", "pending", "history", "worker_busy"}
    assert data["worker_busy"] is False


def test_cancel_a_pending_entry(client):
    entry = client.post("/api/v1/queue", json=a_spec_body()).get_json()

    assert client.delete(f"/api/v1/queue/{entry['entry_id']}").status_code == 200
    assert client.get("/api/v1/queue").get_json()["pending"] == []
    assert client.delete("/api/v1/queue/nope").status_code == 404


# ---------------------------------------------------------------------------
# Relaunch — the repertoire flow
# ---------------------------------------------------------------------------

def test_relaunch_reuses_the_prior_spec_with_a_new_identity(app, client):
    root = app.config["_ROOT"]
    seed_run(root, "parent", manifest=a_manifest("parent"),
             status={"status": "finished"})

    response = client.post("/api/v1/runs/parent/relaunch",
                           json={"init_checkpoint": "/ckpts/best.pt", "epochs": 25})
    assert response.status_code == 202

    entry = response.get_json()
    assert entry["model"] == "mnist_cnn"
    assert entry["epochs"] == 25
    assert entry["run_id"] != "parent", "relaunch must not reuse the parent's run_id"

    queued = client.get("/api/v1/queue").get_json()["pending"][0]
    spec = app.config["QUEUE"].get_entry(queued["entry_id"]).spec
    assert spec["init_checkpoint"] == "/ckpts/best.pt"


def test_relaunch_without_overrides_repeats_the_run(app, client):
    root = app.config["_ROOT"]
    seed_run(root, "again", manifest=a_manifest("again"), status={"status": "finished"})

    entry = client.post("/api/v1/runs/again/relaunch", json={}).get_json()
    spec = app.config["QUEUE"].get_entry(entry["entry_id"]).spec

    assert spec["model"] == "mnist_cnn"
    assert spec["init_checkpoint"] is None


def test_relaunch_of_an_unknown_run_is_404(client):
    assert client.post("/api/v1/runs/ghost/relaunch", json={}).status_code == 404


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

def test_stop_on_an_unmanaged_run_is_404(app, client):
    root = app.config["_ROOT"]
    seed_run(root, "foreign", manifest=a_manifest("foreign", pid=os.getpid()))
    assert client.post("/api/v1/runs/foreign/stop", json={}).status_code == 404


def test_stop_on_a_queued_entry_is_409(app, client):
    """Not yet running: cancel it instead of stopping it."""
    client.post("/api/v1/queue", json=a_spec_body())
    entry = app.config["QUEUE"].snapshot()["pending"][0]

    response = client.post(f"/api/v1/runs/{entry['run_id']}/stop", json={})
    assert response.status_code == 409
    assert "not running" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# UI routes render
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/queue", "/launch"])
def test_ui_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"Quantify Orchestrator" in response.data


def test_run_detail_page_renders(app, client):
    root = app.config["_ROOT"]
    seed_run(root, "shown", manifest=a_manifest("shown"),
             status={"status": "finished",
                     "best": {"metric": "val_acc", "value": 0.9, "epoch": 2,
                              "checkpoint": "/tmp/best.pt"}})

    response = client.get("/runs/shown")
    assert response.status_code == 200
    assert b"shown" in response.data
    assert b"val_acc" in response.data


def test_run_detail_page_404s_for_unknown_run(client):
    response = client.get("/runs/ghost")
    assert response.status_code == 404
    assert b"No run" in response.data


def test_launch_page_offers_only_valid_pairs(client):
    """The form is registry-driven, so an untested pair is not offered in the
    first place rather than rejected on submit."""
    body = client.get("/launch").data.decode()
    assert '"mnist_cnn": ["mnist"]' in body.replace("'", '"')
