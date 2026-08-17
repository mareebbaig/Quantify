"""
process.py — process liveness and control, for runs we own and runs we adopt.

Two things live here:

``process_alive(pid)``
    Is that pid still running? Needed to classify a run whose status.json is
    absent (see docs/llm/RUNSPEC_AND_BUILD_RUN.md).

``RunProcess``
    A uniform handle over a launched run, whether we hold its ``Popen`` (we
    spawned it) or only its pid (we adopted it after an orchestrator restart).
    Everything above the launcher talks to this, never to ``Popen`` directly —
    it is the seam a multi-machine version would replace.

--------------------------------------------------------------------------
Why not os.kill(pid, 0)
--------------------------------------------------------------------------
The obvious liveness check is ``os.kill(pid, 0)``, and on Windows it is
actively destructive: CPython documents that any signal other than
CTRL_C_EVENT / CTRL_BREAK_EVENT is routed to TerminateProcess, so
``os.kill(pid, 0)`` *kills the process it was asked about*. On a Windows dev
box a naive liveness check would therefore terminate the very runs it is
inspecting. Windows goes through OpenProcess + GetExitCodeProcess instead.

No psutil: ctypes is stdlib and this is the only place platform detail leaks.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Optional

IS_WINDOWS = sys.platform == "win32"

# Windows API constants (winnt.h / processthreadsapi.h)
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_TERMINATE = 0x0001
_STILL_ACTIVE = 259


def process_alive(pid: Optional[int]) -> bool:
    """Return True if a process with this pid currently exists.

    False for None or a nonsensical pid. Says nothing about *which* process:
    pids are recycled, so callers that care should corroborate with a start
    time (see process_start_time) or the run's recorded created_at.
    """
    if not pid or pid <= 0:
        return False

    if IS_WINDOWS:
        return _alive_windows(pid)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _alive_windows(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        # A process that genuinely exits with code 259 is indistinguishable
        # from a running one here. Rare enough to accept, documented so it is
        # not a surprise.
        return exit_code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def process_start_time(pid: int) -> Optional[float]:
    """Unix timestamp when the process started, or None if unavailable.

    Linux only (reads /proc). Used to blunt pid reuse: if a process started
    before the run that claims it, the pid has been recycled. Everywhere else
    this returns None and callers fall back to plain liveness.
    """
    if IS_WINDOWS or not os.path.isdir("/proc"):
        return None
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            fields = f.read().rsplit(b")", 1)[1].split()
        starttime_ticks = int(fields[19])          # field 22, 1-indexed
        hz = os.sysconf("SC_CLK_TCK")
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime "):
                    boot = float(line.split()[1])
                    return boot + starttime_ticks / hz
    except (OSError, IndexError, ValueError):
        return None
    return None


def pid_matches_run(pid: Optional[int], created_at_epoch: Optional[float]) -> bool:
    """Liveness, corroborated against when the run claims to have started.

    A live pid whose process began *before* the run did has been recycled and
    belongs to something else. Where start times are unavailable this is just
    process_alive().
    """
    if not process_alive(pid):
        return False
    if created_at_epoch is None:
        return True
    started = process_start_time(int(pid))
    if started is None:
        return True
    # A minute of slack: created_at is written seconds after the process began.
    return started <= created_at_epoch + 60.0


class RunProcess:
    """A launched run, supervised either through its Popen or by pid alone.

    Adopted runs (orchestrator restarted while training continued) have no
    Popen, so every operation degrades to a pid-based equivalent. Callers do
    not need to know which case they hold.
    """

    def __init__(
        self,
        pid: int,
        run_dir: str,
        *,
        popen: Optional[subprocess.Popen] = None,
        api_port: Optional[int] = None,
        spec_path: Optional[str] = None,
        adopted: bool = False,
    ):
        self.pid = pid
        self.run_dir = run_dir
        self.api_port = api_port
        self.spec_path = spec_path
        self.adopted = adopted
        self._popen = popen
        self._returncode: Optional[int] = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def poll(self) -> Optional[int]:
        """Exit code if the process has ended, else None.

        With a Popen this is the real exit status. For an adopted run only
        liveness is observable, so a finished process reports a synthetic 0 —
        the authoritative outcome for those comes from status.json, which the
        store reads anyway.
        """
        if self._popen is not None:
            self._returncode = self._popen.poll()
            return self._returncode
        if process_alive(self.pid):
            return None
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def is_running(self) -> bool:
        return self.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        return self._returncode

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        """Block until the process exits or timeout elapses. None on timeout."""
        if self._popen is not None:
            try:
                return self._popen.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return None
        deadline = None if timeout is None else time.time() + timeout
        while process_alive(self.pid):
            if deadline is not None and time.time() >= deadline:
                return None
            time.sleep(0.2)
        return self.poll()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def terminate(self) -> None:
        """Ask the process to stop (SIGTERM; TerminateProcess on Windows).

        Nothing in the training process handles SIGTERM, so this is abrupt:
        _post_training() does not run and no status marker is written. It is
        the middle rung of the stop ladder, below a graceful dashboard halt.
        """
        if self._popen is not None:
            self._popen.terminate()
            return
        self._signal_pid(signal.SIGTERM)

    def kill(self) -> None:
        """Last resort. Uncatchable on POSIX; no marker will be written."""
        if self._popen is not None:
            self._popen.kill()
            return
        self._signal_pid(getattr(signal, "SIGKILL", signal.SIGTERM))

    def _signal_pid(self, sig) -> None:
        """Signal an adopted process, which we have no Popen for."""
        if IS_WINDOWS:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(_PROCESS_TERMINATE, False, self.pid)
            if handle:
                try:
                    kernel32.TerminateProcess(handle, 1)
                finally:
                    kernel32.CloseHandle(handle)
            return
        try:
            os.kill(self.pid, sig)
        except OSError:
            pass  # already gone

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "run_dir": self.run_dir,
            "api_port": self.api_port,
            "adopted": self.adopted,
            "returncode": self._returncode,
        }
