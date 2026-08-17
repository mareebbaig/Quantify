"""
store.py — find runs on disk and say what state each one is in.

Implements the classification contract from docs/llm/RUNSPEC_AND_BUILD_RUN.md:

    status.json present   -> authoritative (finished / failed / interrupted)
    absent + pid alive    -> running, live detail from the run's own dashboard
    absent + pid dead     -> crashed

--------------------------------------------------------------------------
Enumerate by DIRECTORY, never by globbing run.json
--------------------------------------------------------------------------
The manifest is written at the END of build_run, so a run that dies during
build has a status.json and no run.json. Globbing manifests would silently
hide exactly the failures worth seeing. Every scan here walks run
*directories* and reads whichever files are present.

A run directory can therefore legitimately contain: both files (normal), only
run.json (still running), only status.json (build failure), or neither (died
between mkdir and the manifest write — classified crashed, reason recorded).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .process import pid_matches_run

MANIFEST_FILENAME = "run.json"
STATUS_FILENAME = "status.json"
SPEC_FILENAME = "spec.json"

# States a run can be in as seen from disk. The queue adds queued / launching /
# stopping / cancelled on top of these.
STATE_RUNNING = "running"
STATE_FINISHED = "finished"
STATE_FAILED = "failed"
STATE_INTERRUPTED = "interrupted"
STATE_CRASHED = "crashed"

TERMINAL_STATES = {STATE_FINISHED, STATE_FAILED, STATE_INTERRUPTED, STATE_CRASHED}

_LIVE_TIMEOUT_S = 2.0


@dataclass
class RunRecord:
    """One run, as the orchestrator sees it."""

    run_id: str
    run_dir: str
    state: str
    reason: Optional[str] = None
    experiment_name: Optional[str] = None
    model: Optional[str] = None
    dataset: Optional[str] = None
    augmentation: Optional[str] = None
    quant: Optional[str] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    epochs_completed: Optional[int] = None
    total_epochs: Optional[int] = None
    best_metric: Optional[str] = None
    best_value: Optional[float] = None
    pid: Optional[int] = None
    api_port: Optional[int] = None
    dashboard_url: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    has_manifest: bool = False
    has_status: bool = False
    manifest: Optional[Dict[str, Any]] = field(default=None, repr=False)
    status: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def sort_key(self) -> str:
        return self.created_at or ""

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "state": self.state,
            "reason": self.reason,
            "experiment_name": self.experiment_name,
            "model": self.model,
            "dataset": self.dataset,
            "augmentation": self.augmentation,
            "quant": self.quant,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "epochs_completed": self.epochs_completed,
            "total_epochs": self.total_epochs,
            "best_metric": self.best_metric,
            "best_value": self.best_value,
            "pid": self.pid,
            "api_port": self.api_port,
            "dashboard_url": self.dashboard_url,
            "error": self.error,
            "has_manifest": self.has_manifest,
            "has_status": self.has_status,
            "is_terminal": self.is_terminal,
        }
        return data


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def iter_run_dirs(roots: Iterable[str]) -> Iterable[str]:
    """Yield every run directory under the configured roots.

    Two layouts are recognised, matching how RunSpec derives output paths:

        <root>/<base>/runs/<run_id>/   the usual case, base per dataset+model
        <root>/runs/<run_id>/          when a root is itself an output_dir

    Directories, not manifests -- see the module docstring.
    """
    seen: set = set()
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue

        candidates: List[Path] = []
        direct = root_path / "runs"
        if direct.is_dir():
            candidates.append(direct)
        try:
            candidates.extend(p for p in root_path.glob("*/runs") if p.is_dir())
        except OSError:
            pass

        for runs_dir in candidates:
            try:
                entries = sorted(runs_dir.iterdir())
            except OSError:
                continue
            for entry in entries:
                if not entry.is_dir():
                    continue
                resolved = str(entry.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    yield str(entry)


def discover_runs(roots: Iterable[str], *, check_liveness: bool = True) -> List[RunRecord]:
    """Classify every run under the roots, newest first."""
    records = [classify_run_dir(d, check_liveness=check_liveness) for d in iter_run_dirs(roots)]
    records.sort(key=lambda r: r.sort_key, reverse=True)
    return records


def find_run(roots: Iterable[str], run_id: str) -> Optional[RunRecord]:
    """Locate a single run by id, or None."""
    for run_dir in iter_run_dirs(roots):
        if os.path.basename(os.path.normpath(run_dir)) == run_id:
            return classify_run_dir(run_dir)
    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_run_dir(run_dir: str, *, check_liveness: bool = True) -> RunRecord:
    """Read a run directory and decide what state it is in.

    The rules are the contract in docs/llm/RUNSPEC_AND_BUILD_RUN.md, in order:
    a status marker wins outright; without one, a live pid means running and a
    dead pid means crashed.
    """
    manifest = _read_json(os.path.join(run_dir, MANIFEST_FILENAME))
    status = _read_json(os.path.join(run_dir, STATUS_FILENAME))
    run_id = os.path.basename(os.path.normpath(run_dir))

    record = RunRecord(
        run_id=run_id,
        run_dir=os.path.abspath(run_dir),
        state=STATE_CRASHED,
        has_manifest=manifest is not None,
        has_status=status is not None,
        manifest=manifest,
        status=status,
    )
    _fill_identity(record, manifest, status)

    # 1. A terminal marker is authoritative, whatever the process is doing.
    if status is not None:
        record.state = status.get("status") or STATE_FAILED
        record.reason = f"status.json: {record.state}"
        record.finished_at = status.get("finished_at")
        record.duration_s = status.get("duration_s")
        record.epochs_completed = status.get("epochs_completed")
        record.error = status.get("error")
        best = status.get("best") or {}
        record.best_metric = best.get("metric")
        record.best_value = best.get("value")
        return record

    # 2. No marker: a run that never wrote a manifest cannot be checked for
    #    liveness at all -- it died between mkdir and the manifest write.
    if manifest is None:
        record.state = STATE_CRASHED
        record.reason = "no manifest and no status marker"
        return record

    # 3. No marker, manifest present: liveness decides.
    if not check_liveness:
        record.state = STATE_RUNNING
        record.reason = "liveness check skipped"
        return record

    created_epoch = _parse_timestamp(manifest.get("created_at"))
    if pid_matches_run(record.pid, created_epoch):
        record.state = STATE_RUNNING
        record.reason = f"pid {record.pid} alive, no status marker"
    else:
        record.state = STATE_CRASHED
        record.reason = (
            f"pid {record.pid} not running and no status marker "
            "(killed, OOM-killer, or power loss)"
        )
    return record


def _fill_identity(record: RunRecord, manifest: Optional[dict], status: Optional[dict]) -> None:
    """Populate the descriptive fields from whichever files exist."""
    if manifest:
        record.experiment_name = manifest.get("experiment_name")
        record.created_at = manifest.get("created_at")
        record.pid = manifest.get("pid")
        record.api_port = manifest.get("api_port")
        record.dashboard_url = manifest.get("dashboard_url")
        spec = manifest.get("spec") or {}
        record.model = spec.get("model")
        record.dataset = spec.get("dataset")
        record.augmentation = spec.get("augmentation")
        record.quant = _quant_summary(spec.get("quant"))
        training = spec.get("training") or {}
        record.total_epochs = training.get("epochs")

    if status:
        record.experiment_name = record.experiment_name or status.get("experiment_name")
        record.created_at = record.created_at or status.get("started_at")
        record.pid = record.pid or status.get("pid")

    # A build failure has no manifest; recover identity from the spec file the
    # launcher wrote before starting.
    if record.model is None:
        spec_file = _read_json(os.path.join(record.run_dir, SPEC_FILENAME))
        if spec_file:
            record.model = spec_file.get("model")
            record.dataset = spec_file.get("dataset")
            record.augmentation = spec_file.get("augmentation")
            record.quant = _quant_summary(spec_file.get("quant"))
            record.experiment_name = record.experiment_name or spec_file.get("experiment_name")
            training = spec_file.get("training") or {}
            record.total_epochs = record.total_epochs or training.get("epochs")


def _quant_summary(quant: Optional[dict]) -> Optional[str]:
    """Render the quantization as the short W/A/B tag used in run names."""
    if not quant:
        return None
    if quant.get("weight_coeffs"):
        weights = "coeffs"
    else:
        weights = f"W{quant.get('weight_bits')}"
    return f"{weights} A{quant.get('act_bits')} B{quant.get('bias_bits')}"


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file, or None if missing/unreadable/mid-write."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _parse_timestamp(value: Optional[str]) -> Optional[float]:
    """Parse the "%Y-%m-%dT%H:%M:%S" stamps the manifest writes."""
    if not value:
        return None
    try:
        return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Live detail
# ---------------------------------------------------------------------------

def fetch_live_status(record: RunRecord, timeout_s: float = _LIVE_TIMEOUT_S) -> Optional[Dict[str, Any]]:
    """Proxy the run's own /api/v1/status.

    The training process already reports phase, epoch, progress, ETA, current
    LR, best metric, pause state and scheduler detail -- far more than the
    orchestrator could reconstruct from disk, and always current. Returns None
    when the run has no dashboard or is not answering; that is not an error,
    since a run is classified running by pid, not by its port.
    """
    if not record.api_port:
        return None

    import urllib.error
    import urllib.request

    host = (record.manifest or {}).get("api_host") or "127.0.0.1"
    url = f"http://{host}:{record.api_port}/api/v1/status"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def post_control(record: RunRecord, endpoint: str, body: Optional[dict] = None,
                 timeout_s: float = 5.0) -> Optional[Dict[str, Any]]:
    """POST to one of the run's control endpoints (e.g. "control/halt").

    Returns the decoded response, or None if the run has no reachable
    dashboard -- which the stop ladder treats as "graceful is unavailable,
    escalate".
    """
    if not record.api_port:
        return None

    import urllib.error
    import urllib.request

    host = (record.manifest or {}).get("api_host") or "127.0.0.1"
    url = f"http://{host}:{record.api_port}/api/v1/{endpoint.lstrip('/')}"
    payload = json.dumps(body or {}).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None
