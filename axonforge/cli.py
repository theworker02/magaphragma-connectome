"""AxonForge CLI."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="axonforge")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Run API (+ dashboard static) on port 8741")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8741)
    s.set_defaults(func=cmd_serve)

    d = sub.add_parser("demo", help="Submit demo volume and run adaptive locally")
    d.set_defaults(func=cmd_demo)

    t = sub.add_parser("status", help="Print runtime status JSON")
    t.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return int(args.func(args))


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from axonforge.api import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    from axonforge.runtime import RUNTIME

    print(json.dumps(RUNTIME.submit_volume(reset=True), indent=2))
    print(json.dumps(RUNTIME.request_inference(run_adaptive=True), indent=2))
    print(json.dumps(RUNTIME.status(), indent=2))
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    from axonforge.runtime import RUNTIME

    print(json.dumps(RUNTIME.status(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
