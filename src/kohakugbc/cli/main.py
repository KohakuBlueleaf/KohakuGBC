"""``gbc``: the command line of KohakuGBC.

gbc run CONFIG [CONFIG ...]     build the machine from configuration layers and run its source
gbc serve CONFIG [CONFIG ...]   keep the machine running, images streamed in over HTTP
gbc check CONFIG [CONFIG ...]   build and check the machine without starting any unit
gbc host CONFIG ... --slot S    serve the units of slot S's entry (--slot repeats) to remote units
gbc doctor [CONFIG ...]         versions, devices, entries; a configuration's placement plan
"""

import argparse
import contextlib
import json
import os
import pickle
import signal
import sys
import threading
import time
from importlib import metadata
from pathlib import Path

from togeari import no_unit_start
from togeari.cli.commands.serve import serve_until_signal
from togeari.plugins import Inspector
from togeari.plugins.telemetry import Telemetry
from togeari.serve import Service

from kohakugbc import __version__
from kohakugbc.engines import host as engine_host
from kohakugbc.engines import placement, registry
from kohakugbc.engines.host import EngineHost, wire
from kohakugbc.io.service import Jobs
from kohakugbc.machine import build, build_parts, config


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


def serve(args) -> int:
    """The machine as ``togeari serve``'s HTTP service: images in over ``POST /jobs``, graphs
    back as they finish and into the configuration's sink; ``--telemetry`` for ``togeari
    dash``."""
    parts = build_parts(*args.configs)
    jobs = Jobs(args.inbox, parts.controls)
    plugins = []
    if args.telemetry:
        plugins = [Inspector(), Telemetry(args.telemetry, inspector_token=True)]
    host, _, port = args.listen.rpartition(":")
    service = Service(
        parts.machine,
        parts.sink,
        jobs.prepare,
        jobs.present,
        host,
        int(port),
        os.environ.get(args.token_env) if args.token_env else None,
        plugins,
    )
    return serve_until_signal(service, sys.stdout)


def check(layers: list[str]) -> int:
    with no_unit_start():
        machine, _, _ = build(*layers)
    for warning in machine.warnings:
        print("warning:", warning)
    print("ok:", ", ".join(machine.code.entries))
    machine.stop()
    return 0


def host(args) -> int:
    """Serve the units of slots' entries to remote units: ``--slot`` names slots of the
    configuration, or ``--engines-file`` holds the slots' settings (pickled)."""
    if args.engines_file:
        engines = pickle.loads(Path(args.engines_file).read_bytes())
    else:
        if not args.configs or not args.slot:
            print("gbc host: CONFIG ... --slot S, or --engines-file", file=sys.stderr)
            return 2
        cfg = config.load(*args.configs)
        engines = {s: cfg["engines"].get(s) for s in args.slot}
    served = engine_host.serve(engines)
    address, _ = wire.address(args.listen)
    key = engine_host.authkey({"authkey_env": args.authkey_env})
    server = EngineHost(served.units, address, key, shapes=served.shapes, lazy=args.lazy)
    bound = server.start()
    if args.ready_file:
        ready = Path(args.ready_file)
        info = {"address": bound, "units": served.by_slot, "shapes": served.slot_shapes}
        ready.with_suffix(".tmp").write_bytes(pickle.dumps(info))
        ready.with_suffix(".tmp").replace(ready)
    print(f"serving {[u.name for u in served.units]} at {bound[0]}:{bound[1]}", flush=True)
    if args.watch_stdin:
        threading.Thread(target=_until_stdin_closes, args=(server,), daemon=True).start()
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def _until_stdin_closes(server) -> None:
    """Stop the host when its stdin ends (the process that started it ended). The descriptor
    is read raw: a thread blocked in the buffered reader aborts the interpreter's exit."""
    with contextlib.suppress(OSError, ValueError):
        while os.read(sys.stdin.fileno(), 4096):
            pass
    server.stopping.set()


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
    for name in ("run", "serve", "check", "host", "doctor"):
        p = sub.add_parser(name)
        p.add_argument("configs", nargs="*" if name in ("doctor", "host") else "+")
        if name == "serve":
            p.add_argument("--listen", default="127.0.0.1:8790")
            p.add_argument("--inbox", default="runs/serve/inbox", help="posted pictures")
            p.add_argument("--telemetry", default=None, help="telemetry directory for the dash")
            p.add_argument("--token-env", default=None, help="env variable holding the token")
        if name == "host":
            p.add_argument("--slot", action="append", choices=registry.SLOTS)
            p.add_argument("--engines-file")
            p.add_argument("--ready-file")
            p.add_argument("--lazy", action="store_true")
            p.add_argument("--watch-stdin", action="store_true")
            p.add_argument("--listen", default="127.0.0.1:7501")
            p.add_argument("--authkey-env", default="KOHAKUGBC_HOST_KEY")
    args = parser.parse_args(argv)
    match args.command:
        case "run":
            return run(args.configs)
        case "serve":
            return serve(args)
        case "check":
            return check(args.configs)
        case "host":
            return host(args)
        case "doctor":
            return doctor(args.configs)
    return 2


if __name__ == "__main__":
    sys.exit(main())
