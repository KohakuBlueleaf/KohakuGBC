"""Engine hosts the machine starts: one ``gbc host`` process per named host, its environment (a
card through ``CUDA_VISIBLE_DEVICES``) set before anything in it loads.

A host serves the units of the slots it is given (``engines``, settings per slot as in the
machine's engine configuration) on a loopback port, lazily: it builds the slots, writes what it
serves to its ready file (its address, its units per slot and each slot's shape: the lane's
adapter, former, select and lane options) and loads a unit when its remote unit starts. The
process inherits the machine's stdout and stderr; it ends with the machine's process, and a host
whose process ended is started again by the next remote unit that looks for it (``locate``).
"""

import atexit
import contextlib
import os
import pickle
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

KEY_ENV = "KOHAKUGBC_HOST_KEY"


class HostProcess:
    def __init__(
        self,
        name: str,
        engines: dict,
        env: dict | None = None,
        ready_timeout_s: float = 300.0,
        order: list | None = None,
    ):
        """``order``: the host's slots in start stages (``[["vlm"], ["detect", "embed"]]``: the
        VLM's units up before the others load); slots it leaves out start at once."""
        self.name, self.engines = name, dict(engines)
        self.env = {k: str(v) for k, v in (env or {}).items()}
        self.ready_timeout_s = ready_timeout_s
        self.order = [list(stage) for stage in (order or [])]
        self.authkey = secrets.token_hex(16).encode()
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.address: tuple[str, int] | None = None
        self.units: dict[str, list[str]] = {}
        self.shapes: dict[str, object] = {}
        self.dir = Path(tempfile.mkdtemp(prefix=f"gbc-host-{name}-"))
        # The hosts whose units are up before this one's load (``after``), and this one's
        # units up so far; ``up`` is set once every unit is up or one failed (``failed``)
        self.after: list[HostProcess] = []
        self.progress = threading.Condition()
        self.started: set[str] = set()
        self.failed: str | None = None
        self.up = threading.Event()
        atexit.register(self.close)

    def earlier(self, slot: str) -> set[str]:
        """The units of the slots in ``order``'s stages before ``slot``'s."""
        out: set[str] = set()
        for stage in self.order:
            if slot in stage:
                return out
            out |= {u for s in stage for u in self.units.get(s, [])}
        return set()

    def wait_after(self, timeout_s: float, slot: str | None = None) -> None:
        """Wait until every ``after`` host's units are up, then (for a unit of ``slot``) the
        units of this host's earlier stages; raises when one failed or the wait runs out."""
        deadline = time.monotonic() + timeout_s
        for host in self.after:
            if not host.up.wait(max(0.0, deadline - time.monotonic())):
                raise TimeoutError(f"host {self.name!r}: {host.name!r} not up in {timeout_s}s")
            if host.failed:
                raise RuntimeError(f"host {self.name!r}: {host.name!r} failed: {host.failed}")
        needed = self.earlier(slot) if slot is not None else set()
        with self.progress:
            while not needed <= self.started:
                if self.failed:
                    raise RuntimeError(f"host {self.name!r}: an earlier unit failed: {self.failed}")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"host {self.name!r}: {sorted(needed)} not up in time")
                self.progress.wait(min(left, 1.0))

    def unit_up(self, unit: str) -> None:
        with self.progress:
            self.started.add(unit)
            self.progress.notify_all()
            if self.started >= {u for names in self.units.values() for u in names}:
                self.up.set()

    def unit_failed(self, unit: str, why: str) -> None:
        with self.progress:
            self.failed = self.failed or f"{unit}: {why}"
            self.progress.notify_all()
        self.up.set()

    def launch(self) -> None:
        """Start the process and wait for its ready file; raises when it ends first."""
        spec, ready = self.dir / "engines.pkl", self.dir / "ready.pkl"
        spec.write_bytes(pickle.dumps(self.engines))
        ready.unlink(missing_ok=True)
        command = [sys.executable, "-m", "kohakugbc.cli.main", "host", "--engines-file", str(spec)]
        command += ["--ready-file", str(ready), "--listen", "127.0.0.1:0", "--lazy"]
        command += ["--watch-stdin"]
        env = {**os.environ, **self.env, KEY_ENV: self.authkey.decode()}
        # The host ends when its stdin closes: when this process ends, however it ends
        proc = subprocess.Popen(command, env=env, stdin=subprocess.PIPE)
        deadline = time.monotonic() + self.ready_timeout_s
        while not ready.exists():
            if proc.poll() is not None:
                raise RuntimeError(f"engine host {self.name!r} exited ({proc.returncode})")
            if time.monotonic() > deadline:
                proc.kill()
                waited = self.ready_timeout_s
                raise TimeoutError(f"engine host {self.name!r}: not ready in {waited}s")
            time.sleep(0.05)
        info = pickle.loads(ready.read_bytes())
        self.proc, self.address = proc, tuple(info["address"])
        self.units, self.shapes = info["units"], info["shapes"]

    def locate(self) -> tuple[str, int]:
        """The host's address, after starting the process again when it ended."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self.launch()
            return self.address

    def close(self) -> None:
        """End the process (its units stop on the way out), killing it after 30 s."""
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(10)


def launch_all(hosts: dict) -> dict[str, HostProcess]:
    """Every host of ``hosts`` (name -> ``{"env", "engines", "after", "order"}``) started at
    once; one error names every host that failed, and the started ones are closed again.
    ``after``: hosts whose units are up before this host's units load; ``order``: the host's
    slots in start stages (``HostProcess``)."""
    out = {
        name: HostProcess(
            name, h["engines"], h.get("env"), h.get("ready_timeout_s", 300.0), h.get("order")
        )
        for name, h in hosts.items()
    }
    for name, h in hosts.items():
        unknown = [a for a in h.get("after", []) if a not in out or a == name]
        if unknown:
            raise ValueError(f"engines.hosts.{name}.after: {unknown} are not other hosts")
        out[name].after = [out[a] for a in h.get("after", [])]
    for name in out:
        seen, todo = set(), list(out[name].after)
        while todo:
            host = todo.pop()
            if host.name == name:
                raise ValueError(f"engines.hosts.{name}.after: a cycle back to {name!r}")
            if host.name not in seen:
                seen.add(host.name)
                todo += host.after
    errors: dict[str, BaseException] = {}

    def run(name: str, host: HostProcess) -> None:
        try:
            host.launch()
        except BaseException as exc:  # noqa: BLE001
            errors[name] = exc

    threads = [threading.Thread(target=run, args=item, daemon=True) for item in out.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        for host in out.values():
            host.close()
        lines = "\n  ".join(f"{n}: {e}" for n, e in errors.items())
        raise ValueError(f"engine hosts failed to start:\n  {lines}")
    return out
