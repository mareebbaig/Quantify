"""
queue.py — one run at a time, in order, surviving a restart.

Why one worker and not a scheduler: the box has one GPU (concurrent ImageNet
runs at batch 1024 exhaust it) and DALI pins device_id=0 unconditionally
(utils/dali_pipeline.py:193). Concurrency is not a tuning knob here, it is a
correctness problem, so the queue serialises by construction.

Only a TERMINAL state frees the worker. A paused run still holds its GPU
memory, and pause is reversible, so a paused run keeps the worker occupied --
which is the behaviour you want.

--------------------------------------------------------------------------
Reconciliation on restart
--------------------------------------------------------------------------
The queue is persisted, so an orchestrator restart resumes rather than
forgets. For each entry that was not terminal:

    queued                      -> stays queued
    launching/running, alive    -> ADOPTED: supervised again by pid + marker
    launching/running, dead,
        status.json present     -> that terminal status is recorded
    launching/running, dead,
        no marker               -> crashed

Nothing is ever relaunched automatically. Re-running a run is a decision with
GPU-hours attached, so it stays an explicit user action.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..run_spec import RunSpec
from . import store
from .launcher import launch_run
from .process import RunProcess, process_alive

QUEUE_FILENAME = "queue.json"

STATE_QUEUED = "queued"
STATE_LAUNCHING = "launching"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_CANCELLED = "cancelled"

ACTIVE_STATES = {STATE_QUEUED, STATE_LAUNCHING, STATE_RUNNING, STATE_STOPPING}

# How long to keep looking for status.json after the process exits. The
# launcher writes the marker before the interpreter exits, so this is normally
# satisfied immediately; the wait covers filesystem flush ordering. Exceeding
# it means the process died without recording anything -- a crash.
MARKER_GRACE_S = 10.0

# Stop ladder timings. The halt default is generous because halt lands on an
# epoch boundary and an ImageNet epoch is minutes long; a slow halt is not a
# hang. Overridable per request.
DEFAULT_HALT_TIMEOUT_S = 900.0
SIGTERM_TIMEOUT_S = 30.0


@dataclass
class QueueEntry:
    """One submitted run, from enqueue to terminal state."""

    entry_id: str
    spec: Dict[str, Any]
    state: str = STATE_QUEUED
    run_id: Optional[str] = None
    run_dir: Optional[str] = None
    pid: Optional[int] = None
    api_port: Optional[int] = None
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    terminal_status: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
    stop_stage: Optional[str] = None
    stop_deadline: Optional[float] = None
    adopted: bool = False
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def summary(self) -> Dict[str, Any]:
        """The compact form the API and UI list."""
        spec = self.spec or {}
        quant = spec.get("quant") or {}
        return {
            "entry_id": self.entry_id,
            "state": self.state,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "model": spec.get("model"),
            "dataset": spec.get("dataset"),
            "augmentation": spec.get("augmentation"),
            "experiment_name": spec.get("experiment_name"),
            "epochs": (spec.get("training") or {}).get("epochs"),
            "weight_bits": quant.get("weight_bits"),
            "act_bits": quant.get("act_bits"),
            "bias_bits": quant.get("bias_bits"),
            "enqueued_at": self.enqueued_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "terminal_status": self.terminal_status,
            "returncode": self.returncode,
            "api_port": self.api_port,
            "pid": self.pid,
            "stop_stage": self.stop_stage,
            "stop_seconds_left": self.stop_seconds_left,
            "adopted": self.adopted,
            "error": self.error,
            "note": self.note,
        }

    @property
    def stop_seconds_left(self) -> Optional[int]:
        """Seconds remaining on the current stop rung, for the UI countdown."""
        if self.stop_deadline is None:
            return None
        return max(0, int(round(self.stop_deadline - time.time())))


class RunQueue:
    """A single-worker run queue with a persisted, reconcilable state file."""

    def __init__(
        self,
        state_dir: str,
        *,
        poll_interval_s: float = 1.0,
        launch_fn=launch_run,
        history_limit: int = 200,
    ):
        self.state_dir = os.path.abspath(state_dir)
        self.poll_interval_s = poll_interval_s
        self._launch_fn = launch_fn
        self._history_limit = history_limit

        self._lock = threading.RLock()
        self._entries: List[QueueEntry] = []
        self._processes: Dict[str, RunProcess] = {}
        self._worker: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._wake = threading.Event()

        os.makedirs(self.state_dir, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def state_path(self) -> str:
        return os.path.join(self.state_dir, QUEUE_FILENAME)

    def _load(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        known = {f for f in QueueEntry.__dataclass_fields__}
        self._entries = [
            QueueEntry(**{k: v for k, v in entry.items() if k in known})
            for entry in data.get("entries", [])
        ]

    def _save(self) -> None:
        """Rewrite the state file atomically: a torn queue.json would be worse
        than a slightly stale one."""
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "entries": [e.to_dict() for e in self._entries],
        }
        tmp = f"{self.state_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self.state_path)
        except OSError as exc:
            print(f"[queue] WARNING: could not persist queue state: {exc}")

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(self) -> List[str]:
        """Bring persisted state back in line with reality after a restart.

        Returns human-readable notes about what changed, for the log and the UI.
        """
        notes: List[str] = []
        with self._lock:
            for entry in self._entries:
                if entry.state not in {STATE_LAUNCHING, STATE_RUNNING, STATE_STOPPING}:
                    continue

                alive = process_alive(entry.pid)
                if alive:
                    # Still training. Adopt it: no Popen, so RunProcess falls
                    # back to pid-based supervision.
                    self._processes[entry.entry_id] = RunProcess(
                        pid=entry.pid, run_dir=entry.run_dir or "",
                        api_port=entry.api_port, adopted=True,
                    )
                    entry.adopted = True
                    entry.state = STATE_RUNNING
                    entry.note = "adopted after orchestrator restart"
                    notes.append(f"{entry.run_id}: adopted (pid {entry.pid} still alive)")
                    continue

                # Process is gone. The marker, if any, is authoritative.
                marker = self._read_marker(entry)
                if marker:
                    self._mark_terminal(entry, marker.get("status", store.STATE_FAILED),
                                        note="recovered from status.json after restart")
                    notes.append(f"{entry.run_id}: {entry.terminal_status} (from status.json)")
                else:
                    self._mark_terminal(entry, store.STATE_CRASHED,
                                        note="process gone and no status marker at restart")
                    notes.append(f"{entry.run_id}: crashed (no marker, pid dead)")

            if notes:
                self._save()
        return notes

    def _read_marker(self, entry: QueueEntry) -> Optional[dict]:
        if not entry.run_dir:
            return None
        path = os.path.join(entry.run_dir, store.STATUS_FILENAME)
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _mark_terminal(self, entry: QueueEntry, status: str, *,
                       note: Optional[str] = None) -> None:
        entry.state = status
        entry.terminal_status = status
        entry.ended_at = entry.ended_at or time.strftime("%Y-%m-%dT%H:%M:%S")
        entry.stop_stage = None
        entry.stop_deadline = None
        if note:
            entry.note = note
        self._processes.pop(entry.entry_id, None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, spec: RunSpec) -> QueueEntry:
        """Add a validated spec to the back of the queue."""
        entry = QueueEntry(
            entry_id=uuid.uuid4().hex[:12],
            spec=spec.to_dict(),
            run_id=spec.run_id,
            run_dir=os.path.abspath(spec.run_dir),
            enqueued_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        with self._lock:
            self._entries.append(entry)
            self._trim_history()
            self._save()
        self._wake.set()
        return entry

    def cancel(self, entry_id: str) -> QueueEntry:
        """Cancel a pending entry. Running entries must be stopped, not cancelled."""
        with self._lock:
            entry = self._get(entry_id)
            if entry is None:
                raise KeyError(f"no such queue entry: {entry_id}")
            if entry.state != STATE_QUEUED:
                raise ValueError(
                    f"entry {entry_id} is {entry.state}, not {STATE_QUEUED}; "
                    "use stop for a run that has already started"
                )
            entry.state = STATE_CANCELLED
            entry.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save()
            return entry

    def request_stop(self, entry_id: str, *,
                     halt_timeout_s: float = DEFAULT_HALT_TIMEOUT_S) -> QueueEntry:
        """Begin the stop ladder for a running entry.

        Graceful first: the run's own control API halts it at the next epoch
        boundary, which runs _post_training() -- final plots, best-checkpoint
        restore, status marker. Only if that does not land within
        halt_timeout_s does the worker escalate to SIGTERM and then SIGKILL,
        both of which skip finalisation entirely.
        """
        with self._lock:
            entry = self._get(entry_id)
            if entry is None:
                raise KeyError(f"no such queue entry: {entry_id}")
            if entry.state not in {STATE_RUNNING, STATE_LAUNCHING}:
                raise ValueError(f"entry {entry_id} is {entry.state}, not running")

            record = store.classify_run_dir(entry.run_dir) if entry.run_dir else None
            accepted = None
            if record is not None:
                accepted = store.post_control(record, "control/halt", {"confirm": True})

            entry.state = STATE_STOPPING
            if accepted is not None:
                entry.stop_stage = "halt"
                entry.stop_deadline = time.time() + halt_timeout_s
                entry.note = "halt requested; will finish the current epoch"
            else:
                # No reachable dashboard (api_port null or bind failed), so the
                # graceful rung is unavailable -- go straight to SIGTERM.
                entry.stop_stage = "sigterm"
                entry.stop_deadline = time.time() + SIGTERM_TIMEOUT_S
                entry.note = "no reachable dashboard; sent SIGTERM"
                proc = self._processes.get(entry.entry_id)
                if proc is not None:
                    proc.terminate()
            self._save()
            self._wake.set()
            return entry

    def snapshot(self) -> Dict[str, Any]:
        """Queue state for the API and UI."""
        with self._lock:
            entries = [e.summary for e in self._entries]
            running = [e for e in entries if e["state"] in
                       {STATE_RUNNING, STATE_LAUNCHING, STATE_STOPPING}]
            pending = [e for e in entries if e["state"] == STATE_QUEUED]
            history = [e for e in entries if e["state"] not in ACTIVE_STATES]
            return {
                "running": running,
                "pending": pending,
                "history": list(reversed(history)),
                "worker_busy": bool(running),
            }

    def get_entry(self, entry_id: str) -> Optional[QueueEntry]:
        with self._lock:
            return self._get(entry_id)

    def entry_for_run(self, run_id: str) -> Optional[QueueEntry]:
        with self._lock:
            for entry in reversed(self._entries):
                if entry.run_id == run_id:
                    return entry
            return None

    def _get(self, entry_id: str) -> Optional[QueueEntry]:
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def _trim_history(self) -> None:
        terminal = [e for e in self._entries if e.state not in ACTIVE_STATES]
        excess = len(terminal) - self._history_limit
        if excess > 0:
            drop = {id(e) for e in terminal[:excess]}
            self._entries = [e for e in self._entries if id(e) not in drop]

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Reconcile, then run the worker in a daemon thread."""
        notes = self.reconcile()
        for note in notes:
            print(f"[queue] reconciled: {note}")
        if self._worker is not None and self._worker.is_alive():
            return
        self._shutdown.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="run-queue",
                                        daemon=True)
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker thread. Does NOT stop a running training process --
        that keeps going and is adopted on the next start."""
        self._shutdown.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 — the worker must never die
                print(f"[queue] worker error: {type(exc).__name__}: {exc}")
            self._wake.wait(self.poll_interval_s)
            self._wake.clear()

    def _tick(self) -> None:
        """One supervision step: advance the active run, or start the next."""
        with self._lock:
            active = next((e for e in self._entries
                           if e.state in {STATE_LAUNCHING, STATE_RUNNING, STATE_STOPPING}),
                          None)

        if active is not None:
            self._supervise(active)
            return

        with self._lock:
            nxt = next((e for e in self._entries if e.state == STATE_QUEUED), None)
        if nxt is not None:
            self._launch(nxt)

    def _launch(self, entry: QueueEntry) -> None:
        with self._lock:
            entry.state = STATE_LAUNCHING
            entry.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save()

        try:
            spec = RunSpec.from_dict(entry.spec)
            process = self._launch_fn(spec)
        except BaseException as exc:  # noqa: BLE001 — a failed spawn is a failed run
            with self._lock:
                entry.error = f"{type(exc).__name__}: {exc}"
                self._mark_terminal(entry, store.STATE_FAILED,
                                    note="failed to launch")
                self._save()
            print(f"[queue] launch failed for {entry.run_id}: {entry.error}")
            return

        with self._lock:
            self._processes[entry.entry_id] = process
            entry.state = STATE_RUNNING
            entry.pid = process.pid
            entry.api_port = process.api_port
            entry.run_dir = process.run_dir
            # The child generates nothing new here: run_id came from the spec.
            self._save()
        print(f"[queue] launched {entry.run_id} (pid {process.pid}, "
              f"port {process.api_port})")

    def _supervise(self, entry: QueueEntry) -> None:
        """Advance one active entry: escalate a stop, or detect termination."""
        process = self._processes.get(entry.entry_id)
        if process is None:
            # Launching, or an entry whose process we lost track of.
            if entry.state == STATE_LAUNCHING:
                return
            self._settle_without_process(entry)
            return

        if entry.state == STATE_STOPPING:
            self._escalate_stop(entry, process)

        if process.is_running():
            return

        # The process has exited: that is the authority. Give the marker a
        # bounded moment to appear (it is written just before exit), then read
        # it -- this is the sub-second window where the API can still say
        # "running" while the process is on its way out.
        marker = self._await_marker(entry)
        returncode = process.returncode

        with self._lock:
            entry.returncode = returncode
            if marker:
                status = marker.get("status", store.STATE_FAILED)
                note = None
                # Both signals exist; disagreement is worth surfacing.
                if status == store.STATE_FINISHED and returncode not in (0, None):
                    note = (f"status.json says finished but the process exited "
                            f"{returncode}; trusting the marker")
                self._mark_terminal(entry, status, note=note)
            else:
                self._mark_terminal(
                    entry, store.STATE_CRASHED,
                    note=f"process exited ({returncode}) without writing a status marker",
                )
            self._save()
        print(f"[queue] {entry.run_id} -> {entry.terminal_status} "
              f"(returncode {returncode})")
        self._wake.set()

    def _settle_without_process(self, entry: QueueEntry) -> None:
        """An entry marked running with no handle (mid-reconcile edge)."""
        if process_alive(entry.pid):
            return
        marker = self._read_marker(entry)
        with self._lock:
            self._mark_terminal(entry, (marker or {}).get("status", store.STATE_CRASHED),
                                note="settled without a process handle")
            self._save()

    def _await_marker(self, entry: QueueEntry) -> Optional[dict]:
        deadline = time.time() + MARKER_GRACE_S
        while time.time() < deadline:
            marker = self._read_marker(entry)
            if marker:
                return marker
            time.sleep(0.2)
        return self._read_marker(entry)

    def _escalate_stop(self, entry: QueueEntry, process: RunProcess) -> None:
        """Move down the stop ladder when the current rung times out."""
        if entry.stop_deadline is None or time.time() < entry.stop_deadline:
            return

        with self._lock:
            if entry.stop_stage == "halt":
                entry.stop_stage = "sigterm"
                entry.stop_deadline = time.time() + SIGTERM_TIMEOUT_S
                entry.note = "halt did not land in time; sent SIGTERM"
                self._save()
                print(f"[queue] {entry.run_id}: halt timed out, sending SIGTERM")
                process.terminate()
            elif entry.stop_stage == "sigterm":
                entry.stop_stage = "sigkill"
                entry.stop_deadline = time.time() + SIGTERM_TIMEOUT_S
                entry.note = "SIGTERM did not land; sent SIGKILL"
                self._save()
                print(f"[queue] {entry.run_id}: SIGTERM timed out, sending SIGKILL")
                process.kill()
