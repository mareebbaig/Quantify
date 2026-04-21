"""
Where to save and load data?
This workspace enviroment describes and manages where to save/load datasets, models, and other data created or needed at runtime.

Every example follows the same layout::

    <root>/
    ├── data/          # dataset downloads
    ├── checkpoints/   # model weights
    └── logs/          # CSV logs, tensorboard, etc.


The root is controlled by (in priority order):

1. The ``--workdir`` CLI flag.
2. The ``$QATLAB_WORKDIR`` environment variable, combined with the
   example's ``name`` (e.g. ``$QATLAB_WORKDIR/cifar10_vgg``).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_WORKDIR_ENV = "QUANT_WORKDIR"
DEFAULT_ROOT = None


@dataclass(frozen=True)
class Workspace:
    """Bundle of directories produced by a single training run.

    ``root`` is the only real attribute; the subdirectories are
    derived from it so there is a single source of truth.
    """

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> "Workspace":
        """Create every subdirectory if it doesn't already exist."""
        for d in (self.data, self.checkpoints, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def at(cls, root: os.PathLike | str) -> "Workspace":
        """Build a Workspace rooted at ``root`` (expanded & resolved),
        and create its subdirectories."""
        return cls(root=Path(root).expanduser().resolve()).ensure()


def add_workspace_args(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    env_var: str = DEFAULT_WORKDIR_ENV,
) -> None:
    """Register a ``--workdir`` flag on ``parser``.

    Parameters
    ----------
    parser
        Your example's ``argparse.ArgumentParser``.
    name
        Short identifier for this example (e.g. ``"cifar10_vgg"``).
        Used to build a per-example subdirectory when the env var
        fallback is active.
    env_var
        Environment variable consulted for the base directory when
        ``--workdir`` is not given on the command line.
    """
    env_root = os.environ.get(env_var)
    if env_root:
        default = Path(env_root) / name
    else:
        if DEFAULT_ROOT is None:
            raise Exception("Set enviroment variable " + env_root)
        default = DEFAULT_ROOT / name

    parser.add_argument(
        "--workdir",
        type=Path,
        default=default,
        help=(
            f"Base directory for data, checkpoints and logs. "
            f"Can also be controlled via the ${env_var} environment "
            f"variable (in which case the example's name '{name}' "
            f"is appended). Default: %(default)s"
        ),
    )


def workspace_from_args(args: argparse.Namespace) -> Workspace:
    """Create and prepare a :class:`Workspace` from parsed CLI args."""
    return Workspace.at(args.workdir)
