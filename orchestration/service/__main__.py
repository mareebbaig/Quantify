"""
__main__.py — run the orchestrator.

    python -m orchestration.service                       # http://127.0.0.1:8090
    python -m orchestration.service --port 9000
    python -m orchestration.service --root output --root /data/experiments

Starting it reconciles the persisted queue first (see queue.RunQueue.reconcile):
a run still alive is adopted, a dead one is resolved from its status marker or
recorded as crashed, and pending entries stay pending. Nothing is relaunched
automatically.
"""

from __future__ import annotations

import argparse
import sys

from .app import DEFAULT_ROOT, DEFAULT_STATE_DIR, create_app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m orchestration.service",
        description="List, launch and supervise training runs on this machine.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8090,
                        help="Port to serve on (default: 8090)")
    parser.add_argument("--root", action="append", metavar="DIR",
                        help="Directory to scan for runs; repeatable "
                             f"(default: {DEFAULT_ROOT})")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR,
                        help="Where the queue state file lives")
    args = parser.parse_args(argv)

    app = create_app(roots=args.root, state_dir=args.state_dir)

    print(f"[orchestrator] Roots      : {app.config['ROOTS']}")
    print(f"[orchestrator] Queue state: {app.config['QUEUE'].state_path}")
    print(f"[orchestrator] UI         : http://{args.host}:{args.port}/")
    print(f"[orchestrator] API        : http://{args.host}:{args.port}/api/v1/")

    # threaded=True so a slow live-status proxy to a run's dashboard cannot
    # block the whole UI.
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
