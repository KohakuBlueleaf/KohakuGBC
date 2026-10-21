"""``gbc``: the command line of KohakuGBC.

gbc run CONFIG [CONFIG ...]     build the machine from configuration layers and run its source
gbc check CONFIG [CONFIG ...]   build and check the machine without starting any unit
"""

import argparse
import json
import sys
import time

from togeari import no_unit_start

from kohakugbc import __version__
from kohakugbc.machine import build


def run(layers: list[str]) -> int:
    machine, source, sink = build(*layers)
    if source is None:
        print("the configuration names no source", file=sys.stderr)
        return 2
    start = time.time()
    with machine:
        summary = machine.run(source, sink)
    print(json.dumps({"seconds": round(time.time() - start, 2), "summary": summary}, default=str))
    return 0


def check(layers: list[str]) -> int:
    with no_unit_start():
        machine, _, _ = build(*layers)
    for warning in machine.warnings:
        print("warning:", warning)
    print("ok:", ", ".join(machine.code.entries))
    machine.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gbc", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"kohakugbc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check"):
        p = sub.add_parser(name)
        p.add_argument("configs", nargs="+")
    args = parser.parse_args(argv)
    match args.command:
        case "run":
            return run(args.configs)
        case "check":
            return check(args.configs)
    return 2


if __name__ == "__main__":
    sys.exit(main())
