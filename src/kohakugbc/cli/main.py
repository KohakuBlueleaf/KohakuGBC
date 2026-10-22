"""``gbc``: the command line of KohakuGBC.

gbc run CONFIG [CONFIG ...]     build the machine from configuration layers and run its source
gbc check CONFIG [CONFIG ...]   build and check the machine without starting any unit
gbc host CONFIG ... --slot S    serve the units of slot S's entry to remote units
gbc doctor [CONFIG ...]         versions, devices, entries; a configuration's placement plan
"""

import argparse
import json
import sys
import time
from importlib import metadata

from togeari import no_unit_start

from kohakugbc import __version__
from kohakugbc.engines import host as engine_host
from kohakugbc.engines import placement, registry
from kohakugbc.engines.host import EngineHost, wire
from kohakugbc.machine import build, config


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


def host(layers: list[str], slot: str, listen: str, authkey_env: str) -> int:
    """Serve the units of one slot's entry (from the configuration) to remote units."""
    cfg = config.load(*layers)
    units = registry.build_slot(slot, cfg["engines"].get(slot), {}).units
    address, _ = wire.address(listen)
    server = EngineHost(units, address, engine_host.authkey({"authkey_env": authkey_env}))
    bound = server.start()
    print(f"serving {[u.name for u in units]} at {bound[0]}:{bound[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def doctor(layers: list[str]) -> int:
    """Versions, devices, the entries of every slot; with configuration layers, the engines'
    placement plan per device and whether it fits."""
    print(f"kohakugbc {__version__}")
    for name, version in sorted(versions().items()):
        print(f"  {name} {version}")
    print("devices:")
    for device in placement.devices("auto"):
        mem = placement.memory(device)
        free = (
            f"{mem[0] / placement.GIB:.1f} of {mem[1] / placement.GIB:.1f} GiB free" if mem else ""
        )
        print(f"  {device} {free}")
    print("entries:")
    for slot in registry.SLOTS:
        names = sorted(n for s, n in registry.ENTRIES if s == slot)
        print(f"  {slot}: {', '.join(names)}")
    if not layers:
        return 0
    cfg = config.load(*layers)
    slots = registry.build(cfg["engines"])
    needs = [n for s in slots.values() for n in s.needs]
    print("placement:")
    for need in needs:
        print(f"  {need.device}: {need.owner} {need.what} {need.bytes / placement.GIB:.2f} GiB")
    try:
        placement.check(needs, cfg["machine"].get("placement_headroom", 0.05))
    except ValueError as exc:
        print(exc)
        return 1
    print("  fits")
    return 0


def versions() -> dict:
    out = {}
    for name in ("togeari", "kohakunode", "torch", "transformers", "vllm", "httpx"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gbc", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"kohakugbc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check", "host", "doctor"):
        p = sub.add_parser(name)
        p.add_argument("configs", nargs="*" if name == "doctor" else "+")
        if name == "host":
            p.add_argument("--slot", required=True, choices=registry.SLOTS)
            p.add_argument("--listen", default="127.0.0.1:7501")
            p.add_argument("--authkey-env", default="KOHAKUGBC_HOST_KEY")
    args = parser.parse_args(argv)
    match args.command:
        case "run":
            return run(args.configs)
        case "check":
            return check(args.configs)
        case "host":
            return host(args.configs, args.slot, args.listen, args.authkey_env)
        case "doctor":
            return doctor(args.configs)
    return 2


if __name__ == "__main__":
    sys.exit(main())
