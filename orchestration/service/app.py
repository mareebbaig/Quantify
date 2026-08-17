"""
app.py — the orchestrator service: REST API + a thin UI.

Four capabilities, each assembled from pieces that already existed:

    list       run.json + status.json + the classification rules in store.py
    launch     python -m orchestration.launch, via the launch_run seam
    supervise  the run's own /api/v1/status and /api/v1/control/halt
    link       manifest["dashboard_url"], from build_run's bound-port readback

The UI links OUT to each run's own dashboard for deep monitoring; it does not
reimplement the charts.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request

from ..registry import AUGMENTATIONS, DATASETS, MODELS
from ..run_spec import RunSpec, RunSpecError
from ..registry import RegistryError
from . import store
from .launcher import REPO_ROOT
from .queue import DEFAULT_HALT_TIMEOUT_S, RunQueue

DEFAULT_ROOT = os.path.join(REPO_ROOT, "output")
DEFAULT_STATE_DIR = os.path.join(DEFAULT_ROOT, ".orchestrator")


def create_app(
    roots: Optional[List[str]] = None,
    state_dir: str = DEFAULT_STATE_DIR,
    *,
    run_queue: Optional[RunQueue] = None,
    start_worker: bool = True,
) -> Flask:
    """Build the orchestrator app.

    Args:
        roots:        Directories to scan for runs. Runs launched outside these
                      (e.g. by hand via the CLI) will not appear -- see the
                      --api-port follow-up in docs/llm/ORCHESTRATOR.md.
        state_dir:    Where queue.json lives.
        run_queue:    Inject a queue (tests); one is created otherwise.
        start_worker: Start the worker thread. Tests that drive the queue by
                      hand pass False.
    """
    app = Flask(__name__)
    app.config["ROOTS"] = [os.path.abspath(r) for r in (roots or [DEFAULT_ROOT])]
    app.config["QUEUE"] = run_queue or RunQueue(state_dir)

    if start_worker:
        app.config["QUEUE"].start()

    _register_api(app)
    _register_ui(app)
    return app


def _roots(app: Flask) -> List[str]:
    return app.config["ROOTS"]


def _queue(app: Flask) -> RunQueue:
    return app.config["QUEUE"]


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

def _spec_from_payload(payload: Dict[str, Any]) -> RunSpec:
    """Build and pre-flight a RunSpec from a request body.

    Validation is RunSpec's own -- unknown model, augmentation preset belonging
    to another dataset, a fixed-contract model asked for the wrong bit widths.
    The orchestrator adds no rules and so cannot drift from the launcher's.
    """
    if not isinstance(payload, dict) or not payload:
        raise RunSpecError("request body must be a RunSpec JSON object")
    spec = RunSpec.from_dict(payload)
    _preflight_normalization(spec)
    return spec


def _preflight_normalization(spec: RunSpec) -> None:
    """Refuse an untested (model, dataset) pair at submit time.

    RunSpec construction does NOT check this -- resolve_normalization is called
    inside build_run, in the child process. Without this the orchestrator would
    accept the spec, queue it, spend a launch on it and only then fail during
    build. Calling the same registry function here moves that answer to the
    moment the user can act on it; it is a lookup in a table, so it costs
    nothing and cannot disagree with what build_run will decide.
    """
    from ..registry import resolve_normalization

    resolve_normalization(spec.model, spec.dataset,
                          allow_untested_pair=spec.allow_untested_pair)


def _relaunch_spec(record: store.RunRecord, overrides: Dict[str, Any]) -> RunSpec:
    """Derive a new spec from a previous run, with overrides applied.

    This is the repertoire flow the project exists for: take a finished run,
    point it at one of its own checkpoints, and continue. run_id is cleared so
    the new run gets its own directory rather than colliding with its ancestor.
    """
    source = (record.manifest or {}).get("spec")
    if source is None:
        spec_file = os.path.join(record.run_dir, store.SPEC_FILENAME)
        source = store._read_json(spec_file)
    if source is None:
        raise RunSpecError(
            f"run {record.run_id} has neither a manifest nor a spec file to relaunch from"
        )

    data = dict(source)
    data["run_id"] = None            # fresh identity
    data.pop("experiment_name", None)  # re-derived unless overridden

    if "init_checkpoint" in overrides:
        data["init_checkpoint"] = overrides["init_checkpoint"]
    if overrides.get("experiment_name"):
        data["experiment_name"] = overrides["experiment_name"]
    if overrides.get("output_dir"):
        data["output_dir"] = overrides["output_dir"]

    training = dict(data.get("training") or {})
    for key in ("epochs", "learning_rate", "batch_size", "weight_decay"):
        if overrides.get(key) is not None:
            training[key] = overrides[key]
    data["training"] = training

    spec = RunSpec.from_dict(data)
    _preflight_normalization(spec)
    return spec


def _records(app: Flask) -> List[store.RunRecord]:
    return store.discover_runs(_roots(app))


def _find(app: Flask, run_id: str) -> Optional[store.RunRecord]:
    return store.find_run(_roots(app), run_id)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _register_api(app: Flask) -> None:

    @app.get("/api/v1/health")
    def health():
        return jsonify({"ok": True, "roots": _roots(app)})

    @app.get("/api/v1/registry")
    def registry():
        """Ingredient names, so the launch form is driven by the real registry."""
        return jsonify({
            "models": [
                {"name": name, "datasets": list(entry.datasets),
                 "quant_contract": entry.quant_contract,
                 "default_num_classes": entry.default_num_classes}
                for name, entry in sorted(MODELS.items())
            ],
            "datasets": [
                {"name": name, "num_classes": entry.num_classes,
                 "default_augmentation": entry.default_augmentation,
                 "requires_data_dir": entry.requires_data_dir}
                for name, entry in sorted(DATASETS.items())
            ],
            "augmentations": [
                {"name": name, "dataset": entry.dataset, "params": entry.params}
                for name, entry in sorted(AUGMENTATIONS.items())
            ],
        })

    @app.get("/api/v1/runs")
    def runs():
        return jsonify({"runs": [r.to_dict() for r in _records(app)]})

    @app.get("/api/v1/runs/<run_id>")
    def run_detail(run_id):
        record = _find(app, run_id)
        if record is None:
            return jsonify({"error": f"no such run: {run_id}"}), 404
        payload = record.to_dict()
        payload["manifest"] = record.manifest
        payload["status"] = record.status
        if record.state == store.STATE_RUNNING:
            payload["live"] = store.fetch_live_status(record)
        entry = _queue(app).entry_for_run(run_id)
        payload["queue_entry"] = entry.summary if entry else None
        return jsonify(payload)

    @app.get("/api/v1/runs/<run_id>/live")
    def run_live(run_id):
        record = _find(app, run_id)
        if record is None:
            return jsonify({"error": f"no such run: {run_id}"}), 404
        live = store.fetch_live_status(record)
        if live is None:
            return jsonify({"error": "run has no reachable dashboard",
                            "state": record.state}), 503
        return jsonify(live)

    @app.post("/api/v1/runs/<run_id>/stop")
    def run_stop(run_id):
        body = request.get_json(silent=True) or {}
        timeout = float(body.get("halt_timeout_s", DEFAULT_HALT_TIMEOUT_S))
        entry = _queue(app).entry_for_run(run_id)
        if entry is None:
            return jsonify({"error": f"run {run_id} is not managed by this queue"}), 404
        try:
            entry = _queue(app).request_stop(entry.entry_id, halt_timeout_s=timeout)
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(entry.summary), 202

    @app.post("/api/v1/runs/<run_id>/relaunch")
    def run_relaunch(run_id):
        record = _find(app, run_id)
        if record is None:
            return jsonify({"error": f"no such run: {run_id}"}), 404
        overrides = request.get_json(silent=True) or {}
        try:
            spec = _relaunch_spec(record, overrides)
        except (RunSpecError, RegistryError) as exc:
            return jsonify({"error": str(exc)}), 400
        entry = _queue(app).enqueue(spec)
        return jsonify(entry.summary), 202

    @app.get("/api/v1/queue")
    def queue_state():
        return jsonify(_queue(app).snapshot())

    @app.post("/api/v1/queue")
    def queue_add():
        payload = request.get_json(silent=True) or {}
        try:
            spec = _spec_from_payload(payload)
        except (RunSpecError, RegistryError) as exc:
            return jsonify({"error": str(exc)}), 400
        entry = _queue(app).enqueue(spec)
        return jsonify(entry.summary), 202

    @app.delete("/api/v1/queue/<entry_id>")
    def queue_cancel(entry_id):
        try:
            entry = _queue(app).cancel(entry_id)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(entry.summary)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _register_ui(app: Flask) -> None:

    @app.get("/")
    def ui_runs():
        return render_template("runs.html", runs=[r.to_dict() for r in _records(app)],
                               roots=_roots(app))

    @app.get("/queue")
    def ui_queue():
        return render_template("queue.html", queue=_queue(app).snapshot())

    @app.get("/launch")
    def ui_launch():
        return render_template(
            "launch.html",
            models=sorted(MODELS),
            datasets=sorted(DATASETS),
            augmentations=sorted(AUGMENTATIONS),
            model_datasets={name: list(e.datasets) for name, e in MODELS.items()},
            dataset_augmentations={
                name: [a for a, e in AUGMENTATIONS.items() if e.dataset == name]
                for name in DATASETS
            },
        )

    @app.get("/runs/<run_id>")
    def ui_run_detail(run_id):
        record = _find(app, run_id)
        if record is None:
            return render_template("not_found.html", run_id=run_id), 404
        live = (store.fetch_live_status(record)
                if record.state == store.STATE_RUNNING else None)
        entry = _queue(app).entry_for_run(run_id)
        return render_template(
            "run_detail.html",
            run=record.to_dict(),
            manifest=record.manifest,
            status=record.status,
            live=live,
            entry=entry.summary if entry else None,
        )
