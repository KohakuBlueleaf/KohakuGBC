"""The engine host: one node's engine units served to remote units over the network.

One pump thread makes every call into the engines (Togeari's B3: one caller thread), one reader
thread per connection hands it the connection's messages, and the engines' ``done`` callbacks
send results from their own threads. State (capacity, load, readiness, submits taken) goes to
each connection when it changes, and at least every ``heartbeat_s``.
"""

import queue
import threading
import time
from multiprocessing.connection import Listener

from togeari import WorkItem

from kohakugbc.engines.host import wire


class _Payload:
    """The adapter of a hosted item: the payload arrived prepared; the raw reply goes back."""

    protocols = {"*"}
    light = True

    def check(self, signature) -> list[str]:
        return []

    def prepare(self, inputs: dict, protocol):
        return inputs["payload"]

    def decode(self, raw, protocol):
        return raw


PAYLOAD = _Payload()


class _Connection:
    def __init__(self, conn, unit):
        self.conn, self.unit = conn, unit
        self.send_lock = threading.Lock()
        self.tickets: dict = {}
        self.seen = 0
        self.last_state: dict | None = None
        self.last_sent = 0.0
        self.alive = True

    def send(self, message) -> None:
        with self.send_lock:
            if not self.alive:
                return
            try:
                self.conn.send(message)
            except (OSError, EOFError, BrokenPipeError):
                self.alive = False


class EngineHost:
    """Serves ``units`` (started engines, by name) at ``address`` with ``authkey``."""

    def __init__(self, units: list, address: tuple[str, int], authkey: bytes, heartbeat_s=0.2):
        self.units = {u.name: u for u in units}
        self.address, self.authkey = address, authkey
        self.heartbeat_s = heartbeat_s
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.connections: list[_Connection] = []
        self.listener: Listener | None = None
        self.stopping = threading.Event()

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self) -> tuple[str, int]:
        for unit in self.units.values():
            unit.start()
        self.listener = Listener(self.address, authkey=self.authkey)
        threading.Thread(target=self._accept, name="host-accept", daemon=True).start()
        threading.Thread(target=self._pump, name="host-pump", daemon=True).start()
        return self.listener.address

    def stop(self) -> None:
        self.stopping.set()
        if self.listener is not None:
            self.listener.close()
        for c in list(self.connections):
            c.alive = False
            c.conn.close()
        for unit in self.units.values():
            unit.stop()

    def serve_forever(self) -> None:
        while not self.stopping.wait(1.0):
            pass

    # ---- connections -----------------------------------------------------------------------

    def _accept(self) -> None:
        while not self.stopping.is_set():
            try:
                conn = self.listener.accept()
            except (OSError, EOFError):
                if self.stopping.is_set():
                    return
                continue
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn) -> None:
        try:
            hello = conn.recv()
        except (OSError, EOFError):
            conn.close()
            return
        name = hello[1] if len(hello) > 1 else None
        unit = self.units.get(name) if name else next(iter(self.units.values()), None)
        if hello[0] != "hello" or unit is None:
            conn.send(("refused", f"no unit {name!r}; serving {sorted(self.units)}"))
            conn.close()
            return
        c = _Connection(conn, unit)
        conn.send(
            (
                "welcome",
                {
                    "kind": unit.kind,
                    "max_batch": getattr(unit, "max_batch", 1),
                    "protocol": unit.protocol,
                    "spec": unit.spec(),
                },
            )
        )
        self.inbox.put(("open", c, None))
        while c.alive:
            try:
                message = conn.recv()
            except (OSError, EOFError):
                break
            if message[0] == "bye":
                break
            self.inbox.put((message[0], c, message))
        c.alive = False
        self.inbox.put(("close", c, None))

    # ---- the pump: every engine call ------------------------------------------------------------

    def _pump(self) -> None:
        while not self.stopping.is_set():
            try:
                kind, c, message = self.inbox.get(timeout=0.02)
            except queue.Empty:
                kind = None
            match kind:
                case "open":
                    self.connections.append(c)
                case "close":
                    self._close(c)
                case "submit":
                    self._submit(c, message[1], message[2])
                case "cancel":
                    ticket = c.tickets.get(message[1])
                    if ticket is not None:
                        c.unit.cancel(ticket)
            self._states()

    def _submit(self, c: _Connection, ticket, payloads: list) -> None:
        c.seen += 1
        items = [
            WorkItem(c.unit.name, {"payload": p}, PAYLOAD, protocol=c.unit.protocol)
            for p in payloads
        ]
        work = items if c.unit.kind == "batch" else items[0]

        def done(_, results, c=c, ticket=ticket):
            c.tickets.pop(ticket, None)
            c.send(("done", ticket, [wire.encode(r) for r in results]))

        try:
            c.tickets[ticket] = c.unit.submit(work, done)
        except Exception as exc:  # noqa: BLE001
            c.send(("done", ticket, [("item", repr(exc))] * len(items)))

    def _close(self, c: _Connection) -> None:
        if c in self.connections:
            self.connections.remove(c)
        for ticket in list(c.tickets.values()):
            c.unit.cancel(ticket)
        c.conn.close()

    def _states(self) -> None:
        now = time.monotonic()
        for c in list(self.connections):
            load = c.unit.load()
            state = {
                "capacity": c.unit.capacity(),
                "running": load.running,
                "waiting": load.waiting,
                "ready": bool(c.unit.ready()),
                "seen": c.seen,
                "extra": load.extra,
            }
            if state != c.last_state or now - c.last_sent >= self.heartbeat_s:
                c.last_state, c.last_sent = state, now
                c.send(("state", state))
