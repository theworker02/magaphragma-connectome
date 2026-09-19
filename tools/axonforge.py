#!/usr/bin/env python3
"""Connectome tools entry — AxonForge (serve / demo / switch)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/axonforge.py", description="AxonForge tool entry")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Run API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8741)
    s.set_defaults(func=_serve)

    d = sub.add_parser("demo", help="Submit demo volume and run adaptive")
    d.set_defaults(func=_demo)

    t = sub.add_parser("status", help="Runtime + switch status")
    t.set_defaults(func=_status)

    sw = sub.add_parser("switch", help="Light switch: on | off | toggle | autonomous")
    sw.add_argument("action", choices=["on", "off", "toggle", "autonomous", "manual"])
    sw.add_argument("--enable", action="store_true", help="For autonomous: turn autonomous mode on")
    sw.add_argument("--disable", action="store_true", help="For autonomous: turn autonomous mode off")
    sw.set_defaults(func=_switch)

    ls = sub.add_parser("list-tools", help="Show Connectome tools registry")
    ls.set_defaults(func=_list_tools)

    args = ap.parse_args(argv)
    return int(args.func(args))


def _serve(args: argparse.Namespace) -> int:
    from axonforge.cli import cmd_serve
    return cmd_serve(args)


def _demo(args: argparse.Namespace) -> int:
    from axonforge.cli import cmd_demo
    return cmd_demo(args)


def _status(args: argparse.Namespace) -> int:
    from axonforge.runtime import RUNTIME
    from axonforge.switch import SWITCH
    print(json.dumps({"runtime": RUNTIME.status(), "switch": SWITCH.as_dict()}, indent=2))
    return 0


def _switch(args: argparse.Namespace) -> int:
    from axonforge.switch import SWITCH
    act = args.action
    if act == "on":
        SWITCH.set_power(True)
    elif act == "off":
        SWITCH.set_power(False)
    elif act == "toggle":
        SWITCH.toggle()
    elif act == "autonomous":
        SWITCH.set_autonomous(True if not args.disable else False)
    elif act == "manual":
        SWITCH.set_autonomous(False)
    print(json.dumps(SWITCH.as_dict(), indent=2))
    return 0


def _list_tools(_args: argparse.Namespace) -> int:
    import registry as tools_registry
    print(json.dumps(tools_registry.as_dict_list(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
