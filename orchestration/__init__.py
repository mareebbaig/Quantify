"""
orchestration — describe a training run as data, then launch it.

This is the layer *above* training_harness: training_harness knows how to run a
loop given a model and loaders; orchestration knows how to turn a name ("the
resnet18 / imagenet / imagenet_default run, starting from checkpoint X") into
those objects.

    from orchestration import RunSpec, build_run

    spec = RunSpec(model="resnet18", dataset="imagenet")
    handle = build_run(spec)
    handle.fit()

Or, from a spec file on disk — the path an orchestrator spawns:

    python -m orchestration.launch --spec /path/to/run.json

Layout:
    run_spec.py     RunSpec / QuantSpec / LRScheduleSpec — the data
    registry.py     name -> model / dataset / augmentation preset
    run_builder.py  build_run(spec) -> RunHandle — the assembly
    launch.py       spec file -> a run, plus the terminal-status marker

Deliberately NOT here (slice 1 scope): the orchestrator service, run listing,
any UI, and pixel-level augmentation decomposition. See
docs/llm/RUNSPEC_AND_BUILD_RUN.md.
"""

from .registry import (
    AUGMENTATIONS,
    DATASETS,
    MODELS,
    RegistryError,
    augmentation_names,
    dataset_names,
    model_names,
    resolve_normalization,
)
from .run_builder import RunHandle, build_run, make_injectors
from .run_spec import LRScheduleSpec, QuantSpec, RunSpec, RunSpecError

# launch.py is exposed lazily (PEP 562), following training_harness/api. It is
# run as a script via `python -m orchestration.launch`, and importing it here
# eagerly would load it twice — once as orchestration.launch when the package
# initialises, then again as __main__ — which Python warns about and which
# would give the two copies separate module-level constants.
_LAZY = {"launch_from_spec_file", "STATUS_FINISHED", "STATUS_FAILED",
         "STATUS_INTERRUPTED"}


def __getattr__(name):
    if name in _LAZY:
        from . import launch
        return getattr(launch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RunSpec",
    "QuantSpec",
    "LRScheduleSpec",
    "RunSpecError",
    "build_run",
    "RunHandle",
    "make_injectors",
    "launch_from_spec_file",
    "STATUS_FINISHED",
    "STATUS_FAILED",
    "STATUS_INTERRUPTED",
    "MODELS",
    "DATASETS",
    "AUGMENTATIONS",
    "RegistryError",
    "model_names",
    "dataset_names",
    "augmentation_names",
    "resolve_normalization",
]
