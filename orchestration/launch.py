"""
launch.py — start a run from a spec file, and record how it ended.

    python -m orchestration.launch --spec /path/to/run.json

This is the launch path an orchestrator spawns as a subprocess. It exists to
close two gaps:

1. **Launchable from a spec.** The CLI (examples/train_imagenet_qat.py) cannot
   round-trip a RunSpec -- it hardcodes dataset="imagenet" and has no flags for
   several spec fields -- so a saved run could not be relaunched faithfully.
   Here the spec file *is* the entire interface.

2. **Reliably knowing how a run ended.** QATTrainerV2.fit() has no try/finally
   and nothing wrote a completion marker, so a crashed run and a cleanly
   finished run were byte-identical on disk. This module wraps fit() and writes
   a terminal-status file on every exit path.

The division of labour is deliberate: **the launcher owns lifecycle, the
trainer stays a pure training loop.** Nothing in training_harness/ is modified
to make this work.

--------------------------------------------------------------------------
Where the status file goes
--------------------------------------------------------------------------
Normally ``<run_dir>/status.json``, beside the run manifest. But a spec can
fail to load *before* a run directory is known, so the path is resolved by a
cascade -- see resolve_status_path(). The orchestrator, which writes the spec
file in the first place, can always compute every candidate.

--------------------------------------------------------------------------
What this module is NOT
--------------------------------------------------------------------------
No queue, no run listing, no liveness classification, no subprocess
supervision, no UI. Those belong to the orchestrator that spawns this.
A SIGKILL (or an unhandled SIGTERM) leaves no marker by design -- the
orchestrator treats "no status.json + dead pid" as crashed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

STATUS_FILENAME = "status.json"

# Terminal states. "running" is deliberately absent: this file is only ever
# written once, at the end. A run in progress has no status.json at all, which
# is what makes "no marker + dead pid => crashed" a sound inference.
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"

# Which stage the run was in when it ended.
PHASE_LOAD = "load"    # reading / validating the spec file
PHASE_BUILD = "build"  # build_run(): loaders, model, trainer construction
PHASE_TRAIN = "train"  # fit()


# ---------------------------------------------------------------------------
# Status path resolution
# ---------------------------------------------------------------------------

def resolve_status_path(
    spec: Any = None,
    raw: Optional[Dict[str, Any]] = None,
    spec_path: Optional[str] = None,
) -> Optional[str]:
    """Return the best available path for the terminal-status file.

    The earliest point at which ``<run_dir>/status.json`` is guaranteed is
    *after* the RunSpec is constructed: run_dir is a property derived from
    output_dir and run_id, both of which __post_init__ fills in. A spec that
    fails to load or validate never reaches that point, so the marker would be
    lost exactly when it is most interesting -- hence this cascade:

      1. ``<spec.run_dir>/status.json``            -- spec constructed
      2. ``<output_dir>/runs/<run_id>/status.json`` -- raw JSON parsed but
         validation failed; both fields are read straight from the dict
      3. ``<spec_path>.status.json``                -- file unreadable or
         malformed; falls back beside the spec, which the caller knows

    Returns None only when there is no spec_path either (nothing to key on).
    """
    if spec is not None:
        try:
            return os.path.join(spec.run_dir, STATUS_FILENAME)
        except Exception:  # noqa: BLE001 — a broken spec must not hide the real error
            pass

    if raw:
        output_dir = raw.get("output_dir")
        run_id = raw.get("run_id")
        if output_dir and run_id:
            return os.path.join(str(output_dir), "runs", str(run_id), STATUS_FILENAME)

    if spec_path:
        return f"{spec_path}.status.json"

    return None


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------

def build_status_record(
    *,
    status: str,
    phase: str,
    started_at: float,
    spec: Any = None,
    handle: Any = None,
    error: Optional[BaseException] = None,
) -> Dict[str, Any]:
    """Assemble the terminal-status record.

    Every lookup is best-effort: this runs while an exception is in flight, so
    a failure to collect an optional field must never mask the original error.
    """
    now = time.time()
    record: Dict[str, Any] = {
        "status": status,
        "phase": phase,
        "run_id": _safe(lambda: spec.run_id),
        "experiment_name": _safe(lambda: spec.experiment_name),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "duration_s": round(now - started_at, 1),
        "pid": os.getpid(),
        "run_dir": _safe(lambda: os.path.abspath(handle.run_dir))
                   or _safe(lambda: os.path.abspath(spec.run_dir)),
        "epochs_completed": _epochs_completed(handle),
        "best": _best_checkpoint(handle),
        "metrics": _safe(lambda: handle.trainer.tracker.summary()),
        "error": _error_record(error),
    }
    return record


def _safe(fn):
    """Evaluate fn(), returning None on any failure."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 — optional field; never mask the real error
        return None


def _error_record(error: Optional[BaseException]) -> Optional[Dict[str, Any]]:
    if error is None:
        return None
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


def _epochs_completed(handle: Any) -> Optional[int]:
    """Count distinct epochs that produced metrics.

    Not MetricsTracker.summary()["total_epochs"], which counts *snapshots* --
    V2 commits one per phase, plus a baseline validation pass at epoch -1, so
    two epochs of training report five. Epochs below zero are the baseline pass
    and are excluded.
    """
    def _count():
        history = handle.trainer.tracker.history
        return len({snap.epoch for snap in history if snap.epoch >= 0})
    return _safe(_count)


def _best_checkpoint(handle: Any) -> Optional[Dict[str, Any]]:
    """The best checkpoint this run produced, or None if it never saved one.

    Cheap here, because CheckpointManager holds the ranked records in memory
    (checkpointing.py:216) and the config names the metric they are ranked by.
    Reconstructing this from outside the process would need a two-file join of
    checkpoint_index.json and the manifest.
    """
    def _record():
        manager = handle.trainer.checkpoint_mgr
        record = manager.best_checkpoint_record()
        if record is None:
            return None
        return {
            "metric": handle.config.checkpoint.monitor_metric,
            "value": record.metric_value,
            "epoch": record.epoch,
            "checkpoint": os.path.abspath(record.path),
        }
    return _safe(_record)


def write_status(path: Optional[str], record: Dict[str, Any]) -> Optional[str]:
    """Write the status record, never raising.

    A failure to write the marker (unwritable directory, full disk) must not
    replace the exception that caused the run to end -- that would turn a
    useful "CUDA out of memory" into a baffling "permission denied". Failures
    are reported on stderr and swallowed.
    """
    if path is None:
        print("[launch] WARNING: no writable location for the status file; "
              "the run's outcome was not recorded.", file=sys.stderr)
        return None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
    except OSError as exc:
        print(f"[launch] WARNING: could not write status file {path}: {exc}",
              file=sys.stderr)
        return None
    print(f"[launch] Status ({record['status']}) → {os.path.abspath(path)}")
    return path


# ---------------------------------------------------------------------------
# The launcher
# ---------------------------------------------------------------------------

def launch_from_spec_file(spec_path: str):
    """Load a spec, build the run, train it, and record how it ended.

    Wraps both build and training: a build failure (missing dataset root,
    unreadable init checkpoint, OOM constructing the model) leaves exactly the
    same silence as a training crash, so both are covered.

    On success writes status "finished" and returns the MetricsTracker. On
    failure writes "failed" (or "interrupted" for Ctrl-C) and **re-raises**, so
    the process still exits nonzero and a supervising parent can read the
    outcome from Popen.returncode as well as from the file.
    """
    from .run_builder import build_run
    from .run_spec import RunSpec

    started_at = time.time()
    spec = None
    raw: Optional[Dict[str, Any]] = None
    handle = None
    phase = PHASE_LOAD

    def _finish(status: str, error: Optional[BaseException] = None) -> None:
        write_status(
            resolve_status_path(spec=spec, raw=raw, spec_path=spec_path),
            build_status_record(status=status, phase=phase, started_at=started_at,
                                spec=spec, handle=handle, error=error),
        )

    try:
        # Parsed separately from RunSpec.read_json so that a spec which parses
        # as JSON but fails validation still yields output_dir / run_id for the
        # status path cascade.
        with open(spec_path, encoding="utf-8") as f:
            raw = json.load(f)
        spec = RunSpec.from_dict(raw)
        print(f"[launch] Loaded spec: {spec.experiment_name} "
              f"({spec.model} / {spec.dataset} / {spec.augmentation})")

        phase = PHASE_BUILD
        # tee_stdout=True is not build_run's default and is required here:
        # setup_output_tee is what creates <run_dir>/run.log, and without it a
        # crashed run leaves no traceback on disk at all.
        handle = build_run(spec, tee_stdout=True)

        phase = PHASE_TRAIN
        tracker = handle.fit()

    except KeyboardInterrupt as exc:
        _finish(STATUS_INTERRUPTED, exc)
        raise
    except BaseException as exc:  # noqa: BLE001 — record every exit, then re-raise
        _finish(STATUS_FAILED, exc)
        raise

    # Reached only on a clean return from fit(), which has already run
    # _post_training() -> mark_finished(). This marker is written strictly
    # afterwards and is the on-disk source of truth for post-mortem status;
    # mark_finished()'s api_metrics.jsonl event is unchanged and only exists
    # when the run had api_port set.
    _finish(STATUS_FINISHED)
    return tracker


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m orchestration.launch",
        description="Run a training job described by a RunSpec JSON file.",
    )
    parser.add_argument(
        "--spec", required=True, metavar="PATH",
        help="Path to a RunSpec JSON file (as written by RunSpec.write_json).",
    )
    args = parser.parse_args(argv)

    launch_from_spec_file(args.spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
