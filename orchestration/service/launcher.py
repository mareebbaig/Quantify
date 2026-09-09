"""
launcher.py — launch_run(spec) -> RunProcess. The one spawn seam.

Everything above this module deals in RunSpec and RunProcess. Nothing above it
holds a Popen, reads the child's files, or knows the run is even local — this
function is the single place that would be swapped for an HTTP call to a
per-box agent if this ever became multi-machine.

Why a subprocess rather than build_run() in-process:

  * QuantizerManager is a process-global singleton (quantizers/manager.py:48),
    so two runs in one process would corrupt each other's quantizer state.
  * A training crash must not take the orchestrator down with it.
  * A separate process gives an exit code, which nothing else records.

Why the spec-JSON launcher rather than the CLI: the CLI cannot round-trip a
spec — it hardcodes dataset="imagenet" and has no flags for
augmentation_overrides or allow_untested_pair.

--------------------------------------------------------------------------
The two log files
--------------------------------------------------------------------------
``<run_dir>/launcher.log``  the child's raw stdout/stderr, as redirected here.
                            Covers the window before the tee is installed —
                            import errors, a malformed spec — which run.log
                            structurally cannot.
``<run_dir>/run.log``       written by the run itself: orchestration.launch
                            forces tee_stdout=True and setup_output_tee mirrors
                            the training output into it.

They are separate on purpose. Pointing the child's stdout at run.log would make
_Tee write every line twice into the same file, once through the inherited
stream and once through its own handle.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Optional

from ..run_spec import RunSpec
from .process import RunProcess

# <repo>/orchestration/service/launcher.py -> <repo>
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPEC_FILENAME = "spec.json"
LAUNCHER_LOG = "launcher.log"
MANIFEST_FILENAME = "run.json"


def prepare_spec(spec: RunSpec) -> RunSpec:
    """Return a launch-ready copy of the spec.

    Two adjustments, both required for a supervised run and neither mutating
    the caller's object:

    * ``api_port = 0`` — the OS assigns a free port, so queued runs can never
      collide on one. A fixed port would make the second run silently
      dashboard-less: the bind fails, the trainer ignores the failure
      (trainer_v2.py:268), and training continues unmonitorable.
    * ``output_dir`` absolute — the default is relative to the working
      directory, and the child is spawned with cwd=<repo root>. Resolving it
      here keeps the run's location independent of how the orchestrator
      was started.
    """
    launch_spec = RunSpec.from_dict(spec.to_dict())   # deep copy + revalidate
    launch_spec.training.api_port = 0
    launch_spec.output_dir = os.path.abspath(launch_spec.output_dir)
    return launch_spec


def launch_run(
    spec: RunSpec,
    *,
    python_executable: Optional[str] = None,
    repo_root: str = REPO_ROOT,
    port_wait_s: float = 60.0,
) -> RunProcess:
    """Spawn a run as a supervised subprocess and return a handle to it.

    Args:
        spec:              What to run. Copied and adjusted by prepare_spec().
        python_executable: Interpreter for the child (default: this one).
        repo_root:         Working directory for the child, so its relative
                           imports and any relative paths resolve.
        port_wait_s:       How long to wait for the child's run.json to appear
                           so the bound dashboard port can be recovered.

    Returns:
        RunProcess — pid, run_dir, and api_port when the manifest showed up.

    A missing manifest is not an error: the run may have failed during build,
    which the store classifies correctly from status.json.
    """
    launch_spec = prepare_spec(spec)
    run_dir = launch_spec.run_dir
    os.makedirs(run_dir, exist_ok=True)

    spec_path = os.path.join(run_dir, SPEC_FILENAME)
    launch_spec.write_json(spec_path)

    log_path = os.path.join(run_dir, LAUNCHER_LOG)
    log_file = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")

    command = [
        python_executable or sys.executable,
        "-m", "orchestration.launch",
        "--spec", spec_path,
    ]
    env = {**os.environ, "PYTHONUTF8": "1"}

    try:
        popen = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    finally:
        # The child holds its own duplicate of the descriptor.
        log_file.close()

    api_port = wait_for_bound_port(run_dir, timeout_s=port_wait_s, popen=popen)

    return RunProcess(
        pid=popen.pid,
        run_dir=run_dir,
        popen=popen,
        api_port=api_port,
        spec_path=spec_path,
    )


def wait_for_bound_port(
    run_dir: str,
    *,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.25,
    popen: Optional[subprocess.Popen] = None,
) -> Optional[int]:
    """Poll the run's manifest for the port its dashboard actually bound to.

    The manifest is written inside build_run, before fit() starts, so it
    normally appears within seconds. Reading it beats parsing stdout: with
    api_port=0 the requested port is not the answer, and only the running
    process knows what the OS handed it.

    Returns None if the manifest never appears, if it records no port (the bind
    failed), or if the child exits first.
    """
    manifest_path = os.path.join(run_dir, MANIFEST_FILENAME)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    return json.load(f).get("api_port")
            except (OSError, json.JSONDecodeError):
                pass  # mid-write; try again
        if popen is not None and popen.poll() is not None:
            # Child is gone — one last look in case it wrote the manifest on
            # its way out, then give up.
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, encoding="utf-8") as f:
                        return json.load(f).get("api_port")
                except (OSError, json.JSONDecodeError):
                    return None
            return None
        time.sleep(poll_interval_s)

    return None
