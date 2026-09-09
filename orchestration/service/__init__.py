"""
orchestration.service — list, launch and supervise runs on this machine.

The layer above orchestration/: where run_spec + build_run + launch describe
and start ONE run, this manages MANY of them.

    python -m orchestration.service        # UI + API on :8090

Design, in one breath: runs are discovered by walking run *directories* and
classified from run.json + status.json; new runs are launched as subprocesses
through the single launch_run() seam; exactly one trains at a time because the
box has one GPU; live detail and graceful stops are proxied to each run's own
dashboard rather than reimplemented.

Modules:
    process.py   process_alive() and RunProcess (owned or adopted)
    launcher.py  launch_run(spec) -> RunProcess -- the one spawn seam
    store.py     discovery + the state classification rules
    queue.py     the one-worker queue, persistence and reconciliation
    app.py       Flask API + UI

See docs/llm/ORCHESTRATOR.md.
"""

from .launcher import launch_run
from .process import RunProcess, process_alive
from .queue import QueueEntry, RunQueue
from .store import RunRecord, classify_run_dir, discover_runs

__all__ = [
    "launch_run",
    "RunProcess",
    "process_alive",
    "RunQueue",
    "QueueEntry",
    "RunRecord",
    "discover_runs",
    "classify_run_dir",
]
