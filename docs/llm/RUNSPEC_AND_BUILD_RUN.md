# RunSpec and build_run — describing a run as data

Slice 1 of the orchestration layer. This document is the written record of what
was built, what was deliberately left out, and which warts a later architecture
pass should unwind.

## Why this exists

An audit of run-launching found a clean split:

- A run's **training dynamics** were already data — `TrainerConfigV2`
  (`training_harness/config_v2.py:87`) captures optimization, the QAT schedule,
  checkpoint policy, EMA, early stopping, batch-level augmentation, and the API
  port.
- A run's **identity** — which model, which dataset, which augmentation, which
  starting checkpoint — existed only as argparse flags plus ~150 lines of
  imperative wiring inside `examples/train_imagenet_qat.main()`.

You could not describe a run as data, so you could not start one
programmatically, list one, or tell what a checkpoint belonged to. `RunSpec`
moves identity into data; `build_run` turns that data into a running trainer.

```python
from orchestration import RunSpec, build_run
from training_harness.config_v2 import TrainerConfigV2

spec = RunSpec(model="resnet18", dataset="imagenet",
               data_dir="/data/imagenet",
               training=TrainerConfigV2(epochs=150, batch_size=1024, api_port=0))
handle = build_run(spec)
print(handle.dashboard_url)   # http://127.0.0.1:54123/api/v1/
handle.fit()
```

The CLI is now a thin adapter over the same path, so both doors build identical
runs:

```
argv → parse_args() → spec_from_args() → RunSpec → build_run() → RunHandle.fit()
```

## Package layout

| File | Role |
|---|---|
| `orchestration/run_spec.py` | `RunSpec`, `QuantSpec`, `LRScheduleSpec`, validation, JSON round-trip |
| `orchestration/registry.py` | `MODELS` / `DATASETS` / `AUGMENTATIONS`, normalization pair resolution |
| `orchestration/run_builder.py` | `build_run()`, `RunHandle`, manifest writing |
| `orchestration/checkpoint_init.py` | Weight init helpers extracted from the CLI script |
| `utils/normalization.py` | Per-model normalization stats, extracted so they resolve without DALI |

`orchestration/` is a sibling of `training_harness/`, not part of it: the
registry imports `models/`, and the harness must stay agnostic about which
models exist.

## RunSpec schema

### Top level

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model` | `str` | *required* | Registry key (`resnet18`, `resnet50`, `mobilenetv1`, `mobilenetv2`, `mnist_cnn`) |
| `dataset` | `str` | *required* | Registry key (`imagenet`, `mnist`) |
| `augmentation` | `str` | dataset's default | Pixel-level preset name |
| `augmentation_overrides` | `dict` | `{}` | Deviations from the preset; keys must be preset parameters |
| `run_id` | `str` | `%Y-%m-%d_%H%M%S` | Unique per run; names the output directory |
| `experiment_name` | `str` | `<model>_W<n>_A<n>_B<n>` | Prefix for checkpoint filenames and log dirs |
| `output_dir` | `str` | `output/<dataset>_qat_<model>` | **Base** dir; the run writes to `<base>/runs/<run_id>/` |
| `data_dir` | `str \| None` | `None` | Dataset root; falls back to the dataset's env var, then its default |
| `dali_threads` | `int` | `4` | DALI CPU threads (ImageNet only) |
| `init_checkpoint` | `str \| None` | `None` | Checkpoint to start weights from (`--init-from-ptq`) |
| `pretrained` | `bool` | `False` | Load timm float weights |
| `pretrained_qat` | `bool` | `False` | Load timm weights, fuse BN, run the PTQ LSB search in-process |
| `pretrained_qat_cache` | `str \| None` | `None` | Where the LSB-searched model is cached |
| `ptq_search_radius` | `int` | `7` | LSB positions tested each side |
| `ptq_eval_batches` | `int \| None` | `None` | Validation batches per LSB candidate |
| `force_lsb_search` | `bool` | `False` | Re-run the search even on a cache hit |
| `quant` | `QuantSpec` | 8/8/8 | See below |
| `optimizer` | `str` | `"adamw"` | Currently the only supported choice |
| `lr_schedule` | `LRScheduleSpec` | plateau | See below |
| `allow_untested_pair` | `bool` | `False` | Downgrade the normalization pair check to a warning |
| `training` | `TrainerConfigV2` | defaults | Training dynamics, unchanged |

### `QuantSpec`

| Field | Default | Notes |
|---|---|---|
| `weight_bits` | `8` | Mutually exclusive with `weight_coeffs` |
| `weight_coeffs` | `None` | Path to a coefficient file; set `weight_bits=None` when using it |
| `act_bits` | `8` | |
| `bias_bits` | `8` | |
| `weight_lsb_subtract` | `0` | Shift every weight quantizer's LSB after loading an init checkpoint |

### `LRScheduleSpec`

| Field | Default | Notes |
|---|---|---|
| `kind` | `"plateau"` | `plateau` \| `cosine` \| `none` |
| `cosine_warmup_frac` | `0.1` | Fraction of total steps spent warming up |
| `cosine_eta_min` | `1e-6` | Final LR |

`kind` is **authoritative** over `training.reduce_lr_on_plateau`: `__post_init__`
keeps the boolean in sync so a serialized spec can never contradict itself.
Cosine and plateau cannot coexist — cosine steps every batch and would overwrite
any plateau-triggered reduction on the next one.

### What is deliberately NOT in RunSpec

There is exactly one source of truth per value. Anything `TrainerConfigV2`
already owns stays there and is reached as `spec.training.<field>`:
`learning_rate`, `weight_decay`, `batch_size`, `epochs`, `num_classes`,
`num_workers`, the QAT schedule, checkpoint policy, EMA, early stopping,
mixup/cutmix/smoothing/random-erasing, and `api_port`.

`optimizer` and `lr_schedule` carry only the *choice* of algorithm; the numbers
those algorithms consume live in `TrainerConfigV2`.

## Validation

A `RunSpec` validates on construction — an invalid run cannot be represented.

**Field level** (`RunSpecError`): unknown model / dataset / preset; bit widths
outside 1–32; `weight_bits` and `weight_coeffs` both (or neither) set;
`pretrained_qat` with `init_checkpoint`; `pretrained_qat` with coefficient
weights; unknown optimizer or schedule kind.

**Cross-ingredient** (`RegistryError`): a preset that belongs to a different
dataset; an override key the preset does not declare; a model that cannot
honour the requested quantization (see contracts below).

## The three model quantization contracts

The models in this repo do **not** share a constructor signature, so
`ModelEntry.quant_contract` names which applies:

| Contract | Signature | Models |
|---|---|---|
| `injectors` | `(num_classes, weight_quant, act_quant, bias_quant)` — Brevitas injector *classes* | the four ImageNet models |
| `bit_ints` | `(num_classes, weight_bit_width, act_bit_width)` — plain ints, builds its own quantizers | none registered yet (the CIFAR models use this) |
| `fixed` | `()` — quantizers hardcoded inside the module | `mnist_cnn` |

A `fixed`-contract model **rejects** a spec asking for anything other than its
built-in 8/8/8 or its hardwired `num_classes`, rather than silently ignoring the
request. Silently ignoring it would leave the manifest and the checkpoint
claiming a 4-bit run that actually trained at 8-bit.

## Normalization: resolved from the (model, dataset) PAIR

This is the sharpest silent-failure in the system. Normalization belongs to the
*pair*, not to either half: timm's `mobilenetv1_100.ra4` checkpoint was trained
with `mean=std=0.5`, and feeding it ImageNet stats collapses top-1 from ~73% to
~17% — the run trains fine, it just trains badly, and it looks like a bad recipe.

`registry.resolve_normalization(model, dataset)` therefore **refuses** an
unrecognised pair instead of falling back to a default. `allow_untested_pair=True`
on the spec downgrades it to a loud `RuntimeWarning`.

`utils/normalization.py` remains the single source of truth for the values;
`utils/dali_pipeline.py` re-exports every name from it, so existing
`from utils.dali_pipeline import norm_for_model` imports still work. The
extraction exists so normalization resolves on machines with no DALI installed.

## build_run contract

```python
build_run(spec: RunSpec, *, tee_stdout: bool = False,
          write_manifest: bool = True) -> RunHandle
```

`build_run` **assembles inputs only**. It does not touch `fit()` or the training
loop.

- `tee_stdout` — duplicate stdout/stderr into `<run_dir>/run.log`. The CLI passes
  `True`; programmatic callers default to `False` so a library call does not
  hijack the process's streams.
- `write_manifest` — write `run.json` and `latest.json`.

### Construction order (it matters)

1. Resolve identity, create the run directory
2. Optional stdout/stderr tee
3. **Resolve normalization from the pair** — before loaders
4. **Build dataloaders — before the model**: `pretrained_qat`'s LSB search needs
   the val loader and a calibration batch
5. Build quantizer injector classes
6. Build the model via the registry
7. Initialise weights: `pretrained_qat` → else `pretrained` → then
   `init_checkpoint` layered on top (unchanged precedence)
8. Apply `weight_lsb_subtract`
9. Optimizer, then LR schedule
10. Point `TrainerConfigV2` at the run-scoped directory
11. Construct `QATTrainerV2`, embedding the spec as checkpoint provenance
12. Read back the bound API port; write `run.json` and `latest.json`

### RunHandle

| Attribute | Meaning |
|---|---|
| `spec`, `config`, `trainer`, `model`, `optimizer`, `train_loader`, `val_loader` | the assembled objects |
| `run_dir` | where this run writes |
| `manifest_path` | path to `run.json` |
| `api_port` | the port the dashboard actually bound to (`None` if no server) |
| `dashboard_url` | `http://<host>:<port>/api/v1/`, or `None` |
| `.fit(**kwargs)` | passes through to `QATTrainerV2.fit()` |

## Output layout and per-run isolation

```
<output_dir>/                        base, e.g. output/imagenet_qat_resnet18
├── latest.json                      pointer to the newest run
└── runs/<run_id>/                   the run directory
    ├── run.json                     manifest: full spec + bound port + pid
    ├── status.json                  terminal status — written when the run ENDS
    ├── run.log                      teed stdout/stderr (CLI and launched runs)
    ├── checkpoints/                 top-K pool, last.pt, checkpoint_index.json
    ├── plots/
    └── logs/<experiment_name>/<run_id>/   hparams.json, metrics.csv, api_metrics.jsonl
```

**This fixes a real latent bug.** `CheckpointManager` reloads
`checkpoint_index.json` from its save directory (`checkpointing.py:363`) and
prunes across the merged pool (`:343`). Before per-run directories, a second run
sharing an `output_dir` inherited the first run's top-K records and could
**evict its checkpoint files**. Separate directories cure it.

Two consequences:

- **On-disk paths moved.** There is no longer a fixed
  `<output_dir>/checkpoints/last.pt`. `latest.json` is the stable machine-readable
  address for "the newest run under this base directory" — a file rather than a
  symlink because symlinks need elevation on Windows.
  `scripts/run_imagenet_ptq_qat_pipeline.sh` was updated to read it.
- **`logs/` nests `run_id` twice** (once from the run directory, once from
  `ExperimentLogger.run_dir`, `logger.py:60`). Cosmetically redundant, but it
  leaves the logger untouched. Worth flattening in the architecture pass.

## Launching from a spec file

`build_run` constructs a run in-process. To start one as its own process —
which is what an orchestrator does — use the launcher:

```bash
python -m orchestration.launch --spec /path/to/run.json
```

One flag; the spec file is the entire interface. It does
`RunSpec.read_json` → `build_run(spec, tee_stdout=True)` → `fit()`, wrapped in
the lifecycle handling below. Also importable, for an orchestrator that wants
to run something in-process:

```python
from orchestration.launch import launch_from_spec_file
tracker = launch_from_spec_file("/path/to/run.json")
```

Two things the launcher does that `build_run` alone does not:

- **Forces `tee_stdout=True`.** `build_run` defaults it to `False` so a library
  call does not hijack the process's streams, but a launched run must leave a
  log: `setup_output_tee` (`run_utils.py:69`) is what creates
  `<run_dir>/run.log`, and since the launcher re-raises, an exception's
  traceback is teed there as well as recorded in `status.json`.
- **Writes the terminal-status marker**, below.

The CLI (`examples/train_imagenet_qat.py`) is *not* a substitute: it hardcodes
`dataset="imagenet"` (`:834`) and has no flags for `augmentation_overrides` or
`allow_untested_pair`, so a spec cannot round-trip through argv.

## Terminal status (`status.json`)

### Why it exists

`QATTrainerV2.fit()` has no `try/finally`, and `_post_training()`
(`trainer_v2.py:950`) is called at `:652` on the normal path only. So before
this, a crashed run and a cleanly finished run were **byte-identical on disk** —
a manifest, a partial `metrics.csv`, and a dead pid. Nothing could tell them
apart once the process was gone.

The launcher closes that by wrapping `build_run` **and** `fit()` in
try/except and writing a marker on every exit path. **`trainer_v2.py` is not
modified**: the launcher owns lifecycle, the trainer stays a training loop.

### Schema

```json
{
  "status": "finished",
  "phase": "train",
  "run_id": "2026-08-17_174444",
  "experiment_name": "resnet18_W8_A8_B8",
  "started_at": "2026-08-17T17:44:44",
  "finished_at": "2026-08-17T19:02:11",
  "duration_s": 4647.3,
  "pid": 31427,
  "run_dir": "/abs/path/.../runs/2026-08-17_174444",
  "epochs_completed": 80,
  "best": {
    "metric": "val_acc", "value": 0.7213, "epoch": 74,
    "checkpoint": "/abs/.../checkpoints/..._epoch0074_metric0.721300.pt"
  },
  "metrics": { "total_epochs": 165, "best_val_acc": 0.7213, "final_val_acc": 0.7198 },
  "error": null
}
```

| Field | Notes |
|---|---|
| `status` | `finished` \| `failed` \| `interrupted` (Ctrl-C). No `running` value exists — the file is written once, at the end |
| `phase` | Where it ended: `load` (reading/validating the spec), `build` (`build_run`), `train` (`fit`) |
| `epochs_completed` | Distinct epochs that produced metrics. **Use this, not `metrics.total_epochs`** — `MetricsTracker.summary()` counts *snapshots*, and V2 commits one per phase plus a baseline pass, so 2 epochs reports 5 |
| `best` | Best checkpoint by the monitored metric; `null` if none was saved. Cheap in-process (`checkpointing.py:216`); from outside it would need a two-file join |
| `error` | `null` on success; otherwise `{type, message, traceback}` |

Every optional field is best-effort: collection runs while an exception is in
flight, so a failed lookup yields `null` rather than masking the real error.
Likewise a failure to *write* the marker is reported on stderr and swallowed —
turning "CUDA out of memory" into "permission denied" would help nobody.

### Where it lands

Normally `<run_dir>/status.json`. But a spec can fail before a run directory is
known, so the path is resolved by a cascade
(`launch.py:resolve_status_path`) — the earliest point at which
`<run_dir>` is *guaranteed* is after the `RunSpec` is constructed, since
`run_dir` derives from `output_dir` and `run_id`, which `__post_init__` fills:

| Situation | Marker goes to |
|---|---|
| Spec constructed (covers all build and train failures) | `<run_dir>/status.json` |
| JSON parsed but validation failed | `<output_dir>/runs/<run_id>/status.json`, read straight from the raw dict |
| File missing or malformed JSON | `<spec_path>.status.json` |

The orchestrator writes the spec file, so it can always compute every
candidate.

### Ordering vs. the existing finalization

On success: `fit()` → `_post_training()` → `mark_finished()`
(`collector.py:141`, appends a `"training finished"` event to
`api_metrics.jsonl` and closes it) → `fit()` returns → **then** `status.json`.
Strictly sequential and single-threaded, so no race and no double-reporting:
the jsonl event feeds the live API's history, `status.json` is the post-mortem
record.

On failure `_post_training()` never runs, so no jsonl event is written at all —
the marker is the only record. Note also that `mark_finished()`'s event only
exists when the run had `api_port` set (the collector is created only then,
`trainer_v2.py:251`), whereas `status.json` is written unconditionally. **The
marker is the more reliable signal; key on it.**

A SIGKILL — or an unhandled SIGTERM, since no signal handler is installed —
leaves no marker. That is intentional and not worth defending against; see the
consumption rules below.

## How the orchestrator will consume this

Not built yet (next slice), but the contract it depends on:

**Enumerate runs by directory, NOT by globbing `run.json`.** The manifest is
written at the *end* of `build_run`, so a run that fails during build has a
`status.json` and **no `run.json`** — precisely the failures you least want to
lose. Glob `<root>/*/runs/*/` and read whichever files are present.

Classification, given a run directory:

| `status.json` | pid alive | Conclusion |
|---|---|---|
| present | — | Authoritative. Use its `status` |
| absent | yes | Running. Live detail from the run's own `/api/v1/status` |
| absent | no | **Crashed** (SIGKILL, power loss, OOM-killer) |

Liveness needs `pid` from `run.json`; combine with `created_at` to blunt pid
reuse. `psutil` is not a dependency — `os.kill(pid, 0)` suffices on Linux.

Deliberately still absent, for the next slice: the run queue, the listing
service, liveness classification, subprocess supervision, the UI, stale-manifest
cleanup, and the `--api-port` CLI wire-up (unnecessary for launched runs, since
`api_port` is a spec field and `build_run` reads the bound port back).

`training_harness/config_v2.py` is **not** modified — direct `TrainerConfigV2`
users (`examples/dashboard_demo.py`, `examples/mnist_qat_v2.py`) keep their
existing layout. Only runs that go through `build_run` are run-scoped.

## Checkpoint provenance

Every checkpoint a `build_run` run saves carries its spec:

```python
payload["extra"]["run_spec"]   # RunSpec.to_dict()
```

This closes the asymmetry the audit found — PTQ checkpoints were self-describing
(`find_perfect_lsbs_imagenet_ptq.py:1249-1263`) while QAT checkpoints, which
embedded only `TrainerConfigV2`, could not tell you their model, dataset, or
augmentation. Recover it with:

```python
payload = torch.load(path, map_location="cpu", weights_only=False)
spec = RunSpec.from_dict(payload["extra"]["run_spec"])
spec.model, spec.dataset, spec.augmentation, spec.quant.weight_bits
```

`checkpointing.py` needed no changes — `extra` already flowed through
`_build_payload` (`:43-64`). Secondary best-checkpoint pools now pass the same
`extra` (`trainer_v2.py:553`), so a checkpoint reloaded from a secondary pool is
as self-describing as one from the primary.

## Port assignment

`spec.training.api_port` is passed through unchanged: `0` lets the OS pick a free
port, an explicit number requests that one, `None` disables the API. Either way
`build_run` reads the **actually bound** port back from
`DashboardAPIServer.port` (`server.py:237-239`) and records it in `run.json`, so
a launcher can link to the dashboard without guessing.

Note: `examples/train_imagenet_qat.py` still has **no `--api-port` flag**, so CLI
runs expose no dashboard. Adding that flag is the obvious next step and was left
out of this slice to keep the CLI surface unchanged.

## Scope: what slice 1 deliberately did not do

Two cuts, taken on purpose:

1. **Models are name-addressable, not architecture-in-JSON.** The spec names a
   model and a factory builds it. No layer-by-layer description.
2. **Augmentations are named presets, not decomposed recipes.** A preset maps to
   today's existing setup for that dataset. The DALI `@pipeline_def` was **not**
   parameterized — presets only supply values to parameters `build_dali_loaders`
   already exposed. Pixel-level augmentation still cannot be varied freely.

Also out of scope and still absent: the orchestrator service, run listing, any
UI, CIFAR-10 (deferred until a CIFAR pair can actually be tested end to end),
multi-machine concerns, and any V1 consolidation.

Augmentation remains **split**: the pixel-level half is the preset, the
batch-level half (mixup / cutmix / smoothing / random erasing) is still
`TrainerConfigV2` fields. Unifying them is a later slice.

## Known warts for the architecture pass

1. **`orchestration/` imports from `examples/`** — the wrong direction, in two
   places, both lazy and function-scoped:
   - `registry._build_mnist_cnn` imports `MNISTQuantNet` from
     `examples/mnist_qat_v2.py`. The model belongs in `models/`; moving it would
     break `from examples.mnist_qat_v2 import MNISTQuantNet` in two tests.
   - `run_builder._prepare_pretrained_qat` imports the legacy helper from
     `examples/train_imagenet_qat.py`, which is itself coupled to
     `examples/find_perfect_lsbs_imagenet_ptq.py`.
2. **`run_builder._legacy_args`** — a `SimpleNamespace` shim for helpers that
   still expect an argparse namespace. Kept in one function so it can be deleted
   whole once those helpers take a spec.
3. **Vestigial CLI flags.** `--hf-dataset`, `--prefetch-factor`, and
   `--repeat-aug` no longer reach anything (the HuggingFace loader path was
   already dead before this work). They are still parsed, so the CLI surface is
   unchanged; they should be removed deliberately.
4. **Double `run_id` nesting** under `logs/`, described above.
5. **`examples/train_custom_cifar10.py` is bit-rotted** — it constructs
   `TrainerConfig` fields that do not exist (`workdir`, `amp`, `checkpoint_path`)
   and calls `trainer.train()`, which does not exist either (V1 exposes `fit()`).
   It is **not** a `build_run` target and CIFAR is not registered.

## Tests

| File | Covers |
|---|---|
| `tests/test_run_spec.py` | validation rejections, JSON/dict round-trip identity, normalization pair resolution, model quant contracts |
| `tests/test_run_builder.py` | MNIST end-to-end: build → fit, per-run dirs, manifest, bound port, live dashboard, checkpoint provenance |
| `tests/test_cli_spec_regression.py` | **strict** field-by-field equality between the adapter's config and a verbatim copy of the pre-refactor `main()` construction, across 14 command lines |
| `tests/test_launch.py` | the launcher: status-path cascade, success/failure/build-failure/invalid-spec/malformed-spec markers, and a subprocess test asserting a failure produces **both** a nonzero exit code and an on-disk marker |

The regression guard is the important one. `_legacy_config_from_args` in
`tests/test_cli_spec_regression.py` is an intentional verbatim copy of the old
code — do not "clean it up"; its value is being an independent witness. If
someone edits the adapter and changes an effective default, that test fails.

MNIST in `test_run_builder.py` exercises plumbing only. Draw no training-quality
conclusions from it.
