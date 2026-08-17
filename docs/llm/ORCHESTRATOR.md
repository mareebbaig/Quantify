# The orchestrator — list, launch, supervise (single machine)

The layer above `orchestration/`: where `RunSpec` + `build_run` + `launch.py`
describe and start **one** run, this manages **many**.

```bash
python -m orchestration.service                       # UI + API on :8090
python -m orchestration.service --port 9000 --root output --root /data/experiments
```

Four capabilities, each assembled from pieces that already existed:

| Capability | Built on |
|---|---|
| **List** | `run.json` + `status.json` + the classification rules below |
| **Launch** | `python -m orchestration.launch --spec`, via the `launch_run` seam |
| **Supervise** | each run's own `/api/v1/status` and `/api/v1/control/halt` |
| **Link** | `manifest["dashboard_url"]`, from `build_run`'s bound-port readback |

Genuinely new here: process spawning and liveness, the queue and its
reconciliation, the Flask service, the UI. Everything else is assembly.

## Layout

```
orchestration/service/
    process.py    process_alive(), RunProcess (owned or adopted)
    launcher.py   launch_run(spec) -> RunProcess   <- the one spawn seam
    store.py      run discovery + state classification
    queue.py      one-worker queue, persistence, reconciliation
    app.py        Flask API + UI
    __main__.py   python -m orchestration.service
```

## Run state machine

```
        enqueue              worker picks up          child exits
queued ──────────> queued ──> launching ──> running ──────────────> finished
   │                              │            │                    failed
   │ cancel                       │ spawn      │ stop               interrupted
   ▼                              │ failed     ▼                    crashed
cancelled                         ▼         stopping ───────────────┘
                               failed
```

`queued`, `launching`, `stopping` and `cancelled` are orchestrator states and
live in `queue.json`. `running` plus the four terminal states are **derived
from disk** and are equally valid for runs the orchestrator never launched.

## Listing

### Enumerate by directory, never by globbing `run.json`

The manifest is written at the *end* of `build_run`, so a run that fails during
build has a `status.json` and **no** `run.json`. A manifest glob would silently
drop exactly the failures worth seeing. `store.iter_run_dirs` walks
directories and reads whichever files are present:

```
<root>/<base>/runs/<run_id>/     the usual case
<root>/runs/<run_id>/            when a root is itself an output_dir
```

### Classification

Implemented verbatim from `docs/llm/RUNSPEC_AND_BUILD_RUN.md`:

| `status.json` | pid | Result |
|---|---|---|
| present | — | **authoritative**: `finished` / `failed` / `interrupted` |
| absent | alive | `running`; live detail proxied from the run's dashboard |
| absent | dead | `crashed` |
| absent, and no `run.json` either | — | `crashed` ("died between mkdir and the manifest write") |

A marker wins even against a live pid — pids get reused, markers do not lie.
pid reuse is blunted by comparing the process start time against the
manifest's `created_at` (`process_start_time`, Linux `/proc` only; elsewhere it
degrades to plain liveness, which the tests assert explicitly rather than
paper over).

Identity for a build-failed run (no manifest) is recovered from the
`spec.json` the launcher writes before starting.

### Roots

Runs are only found under the configured `--root` directories (default
`<repo>/output`). **A run launched by hand outside them will not appear** —
that is the known `--api-port` follow-up, not a bug in the listing.

## Launching

```python
launch_run(spec) -> RunProcess     # orchestration/service/launcher.py
```

**The single spawn seam.** Nothing above it holds a `Popen`, reads the child's
files, or knows the run is local — everything goes through `RunProcess`. It is
the one function a multi-machine version would replace with a call to a per-box
agent.

It spawns `python -m orchestration.launch --spec <path>` with `cwd=<repo root>`
and `stdin=DEVNULL`, having first:

- forced `training.api_port = 0`, so the OS assigns a free port and queued runs
  can never collide. A fixed port would make the second run silently
  dashboard-less: the bind fails, the trainer ignores the failure
  (`trainer_v2.py:268`), and training continues unmonitorable.
- made `output_dir` absolute, since the default is relative and the child runs
  with a different working directory.
- written the spec to `<run_dir>/spec.json`.

The **bound port is recovered by polling `<run_dir>/run.json`**, not by parsing
stdout: with `api_port=0` the requested port is not the answer, and only the
running process knows what it got. A missing manifest is not an error — the run
may have failed during build, which the store classifies correctly.

Why a subprocess rather than calling `build_run` in-process: `QuantizerManager`
is a process-global singleton (`quantizers/manager.py:48`), so two runs in one
process corrupt each other; a training crash must not take the orchestrator
down; and a child process yields an exit code, which nothing else records.

### The two log files

| File | Written by | Contains |
|---|---|---|
| `<run_dir>/launcher.log` | the orchestrator, as the child's stdout/stderr | everything, **including the pre-tee window** — import errors, a malformed spec |
| `<run_dir>/run.log` | the run itself, via `tee_stdout=True` | the training output |

They are separate **on purpose**. Pointing the child's stdout at `run.log` would
make `_Tee` write every line twice into that file: once through the inherited
stream and once through its own handle. A test asserts the banner appears
exactly once.

## The queue

**One worker. Not a scheduler.** The box has one GPU (concurrent ImageNet runs
at batch 1024 exhaust it) and DALI pins `device_id=0` unconditionally
(`utils/dali_pipeline.py:193`). Serialisation is a correctness requirement, not
a preference.

Only a **terminal** state frees the worker. A paused run still holds its GPU
memory and is resumable, so it keeps the worker busy — which is what you want.

**Terminal detection**: the process exit is the authority. Once
`RunProcess.poll()` reports an exit, the worker looks for `status.json` for a
bounded grace period (10s; the launcher writes it *before* the interpreter
exits, so this is normally satisfied at once). No marker after the grace period
means the process died without recording anything — `crashed`. The returncode is
recorded alongside, and a disagreement between the two signals is noted in the
entry rather than hidden.

State lives in `<state_dir>/queue.json` (default `output/.orchestrator/`),
rewritten atomically via temp + `os.replace`.

### Reconciliation on restart

`RunQueue.start()` reconciles before the worker begins:

| Recorded | pid | Action |
|---|---|---|
| `queued` | — | stays queued |
| `launching`/`running` | alive | **adopted** — supervised again by pid + marker |
| `launching`/`running` | dead, marker present | that terminal status is recorded |
| `launching`/`running` | dead, no marker | `crashed` |

**Nothing is ever relaunched automatically.** Re-running a run costs GPU hours,
so it stays an explicit user action. Adoption means an orchestrator restart
during training picks the run back up instead of orphaning it; since an adopted
run has no `Popen`, `RunProcess` degrades to pid-based supervision — the same
abstraction a remote agent would need.

## Stopping

Graceful first, escalating only on a timeout:

1. **halt** — `POST /api/v1/control/halt` with `{"confirm": true}` on the run's
   own dashboard. Ends the run after the current epoch, which still runs
   `_post_training()`: final plots, best-checkpoint restore, status marker.
   Default window **900s**, overridable per request via `halt_timeout_s` — an
   ImageNet epoch is minutes long, so a slow halt is not a hang. The UI shows
   the current rung and a live countdown so it does not look frozen.
2. **SIGTERM** — 30s. Nothing handles it, so this skips finalisation entirely.
3. **SIGKILL** — uncatchable; no marker, and the store then classifies the run
   `crashed`.

A run with no reachable dashboard (`api_port` null, or the bind failed) skips
straight to SIGTERM.

**Pause is not stop.** A paused run still holds the GPU and still occupies the
worker.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | liveness + configured roots |
| GET | `/api/v1/registry` | model / dataset / augmentation names for the launch form |
| GET | `/api/v1/runs` | every run, classified |
| GET | `/api/v1/runs/<id>` | manifest + status (+ live snapshot when running) |
| GET | `/api/v1/runs/<id>/live` | proxied `/api/v1/status`; 503 if unreachable |
| POST | `/api/v1/runs/<id>/stop` | start the stop ladder; body `{halt_timeout_s?}` |
| POST | `/api/v1/runs/<id>/relaunch` | new spec from a prior run + overrides → enqueue |
| GET | `/api/v1/queue` | `{running, pending, history, worker_busy}` |
| POST | `/api/v1/queue` | validate a spec → enqueue |
| DELETE | `/api/v1/queue/<entry_id>` | cancel a pending entry |

UI routes: `/` (runs), `/queue`, `/launch`, `/runs/<id>`.

### Validation, and one thing RunSpec does *not* check

Submitted specs are validated by constructing a `RunSpec`, so the orchestrator
adds no rules of its own and cannot drift from the launcher's.

One gap worth knowing: **`RunSpec` construction does not check the (model,
dataset) normalization pair** — `resolve_normalization` is called inside
`build_run`, in the child process. Without help, a bad pair would be accepted,
queued, launched, and only fail at build time. `app._preflight_normalization`
therefore calls the same registry function at submit time, so the refusal
arrives when the user can act on it. It is a table lookup, so it cannot
disagree with what `build_run` will later decide.

## UI

Flask + Jinja, one CSS file, vanilla JS for auto-refresh and the stop
countdown. The palette is lifted from `dashboard/index.html` so the two read as
one system. Views: run list (sortable, live runs badged and linked to their
dashboard), queue (pending order, current run with stop, recent history),
launch form (registry-driven, so untested pairs are not even offered), and run
detail (status marker, live snapshot when running, traceback on failure,
relaunch).

It links **out** to each run's dashboard for charts and in-run controls; it
does not reimplement them.

## Tests

| File | Covers |
|---|---|
| `tests/test_orchestrator_store.py` | all six run classes, including build-failed (no manifest) and crashed; discovery layouts; corrupt JSON; pid-reuse |
| `tests/test_orchestrator_queue.py` | the launch seam, `launcher.log` not double-written, strict one-at-a-time execution of 3 real MNIST runs, cancel, failed launch, persistence, all four reconciliation cases, stop frees the worker |
| `tests/test_orchestrator_api.py` | every endpoint, validation refusals, relaunch, UI routes render |
| `tests/test_orchestrator_e2e.py` | POST a spec → subprocess → port recovered → finished → listed; relaunch from a finished run |

The two that matter most are `test_three_runs_execute_strictly_one_at_a_time`
(real subprocesses; asserts the GPU is never double-booked) and
`test_restart_reconciliation_against_a_real_killed_run` (kill a real child,
restart from the persisted file, assert `crashed` and an intact queue).

MNIST throughout, dry-run batches. Plumbing only — no training-quality claims.

## Known limits and follow-ups

- **Hand-launched runs are invisible.** The CLI still has no `--api-port` flag
  and writes outside the configured roots unless told otherwise. Wiring that
  flag is the smallest next step to make screen-session runs appear here.
- **No dependency chaining.** "Run B after A finishes" is not supported.
  `scripts/run_imagenet_ptq_qat_pipeline.sh` is that idea in bash — sequential
  stages where each feeds the next's `--init-from-ptq` — and is the obvious
  model when it is wanted.
- **Single machine only.** `launch_run` is the seam; nothing above it may
  assume it can see a `Popen` or the local filesystem.
- **No GPU scheduling.** One worker, period. Multiple GPUs would need
  `device_id` plumbed through `build_dali_loaders` first, which is not exposed
  today.
- **`os.kill(pid, 0)` is unsafe on Windows** — CPython routes any signal other
  than `CTRL_C_EVENT`/`CTRL_BREAK_EVENT` to `TerminateProcess`, so the obvious
  liveness check *kills the process it asks about*. `process.py` uses
  `OpenProcess` + `GetExitCodeProcess` there instead. Any future code checking
  pids must do the same.
