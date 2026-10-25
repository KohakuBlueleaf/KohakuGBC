"""The engine host: one process's engine units served to remote units.

One pump thread makes every call into the engines (Togeari's B3: one caller thread), one reader
thread per connection hands it the connection's messages, and the engines' ``done`` callbacks
send results from their own threads. State (capacity, load, readiness, submits taken) goes to
each connection when it changes, and at least every ``heartbeat_s``.

An eager host starts every unit before it listens. A lazy one listens first and starts a unit
when a connection asks (``("start",)``, on a thread of its own; the pump leaves the unit alone
until it is up) and stops it on ``("stop",)``.
"""

import contextlib
import queue
import threading
import time
from multiprocessing.connection import Listener

from togeari import WorkItem
from togeari.engines.common.transport import release

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


class _Hosted:
    """A served unit and where its start stands: ``up`` once started, ``error`` its failure."""

    def __init__(self, unit, shape):
        self.unit, self.shape = unit, shape
        self.up = False
        self.starting = False
        self.error: str | None = None


class _Connection:
    def __init__(self, conn, hosted: _Hosted, transport: str):
        self.conn, self.hosted, self.unit = conn, hosted, hosted.unit
        self.codec = wire.codec(transport)
        self.send_lock = threading.Lock()
        self.tickets: dict = {}
        # Result blocks by ticket, held until the remote unit has read them
        self.held: dict = {}
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
            except wire.CLOSED:
                self.alive = False

    def free(self, ticket=None) -> None:
        """Release the result blocks of ``ticket`` (every ticket's when None)."""
        with self.send_lock:
            if ticket is None:
                blocks = [b for owned in self.held.values() for b in owned]
                self.held.clear()
            else:
                blocks = self.held.pop(ticket, [])
        release(blocks, unlink=True)


class EngineHost:
    """Serves ``units`` (by name) at ``address`` with ``authkey``. ``shapes``: per unit name,
    the lane's side of its slot (sent in the welcome). ``lazy``: listen first, start units when
    asked."""

    def __init__(
        self,
        units: list,
        address: tuple[str, int],
        authkey: bytes,
        heartbeat_s=0.2,
        shapes: dict | None = None,
        lazy: bool = False,
    ):
        shapes = shapes or {}
        self.hosted = {u.name: _Hosted(u, shapes.get(u.name)) for u in units}
        self.address, self.authkey = address, authkey
        self.heartbeat_s = heartbeat_s
        self.lazy = lazy
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.connections: list[_Connection] = []
        self.listener: Listener | None = None
        self.stopping = threading.Event()

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self) -> tuple[str, int]:
        if not self.lazy:
            for h in self.hosted.values():
                h.unit.start()
                h.up = True
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
            with contextlib.suppress(OSError):
                c.conn.close()
            c.free()
        for h in self.hosted.values():
            if h.up:
                h.up = False
                h.unit.stop()

    def serve_forever(self) -> None:
        while not self.stopping.wait(1.0):
            pass

    # ---- connections -----------------------------------------------------------------------

    def _accept(self) -> None:
        while not self.stopping.is_set():
            try:
                conn = self.listener.accept()
            except wire.CLOSED:
                if self.stopping.is_set():
                    return
                continue
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn) -> None:
        try:
            hello = conn.recv()
        except wire.CLOSED:
            conn.close()
            return
        name = hello[1] if len(hello) > 1 else None
        options = hello[2] if len(hello) > 2 else {}
        hosted = self.hosted.get(name) if name else next(iter(self.hosted.values()), None)
        transport = options.get("transport", "inband")
        if hello[0] != "hello" or hosted is None or transport not in wire.TRANSPORTS:
            why = f"no unit {name!r}; serving {sorted(self.hosted)}"
            if transport not in wire.TRANSPORTS:
                why = f"transport {transport!r}: one of {wire.TRANSPORTS}"
            conn.send(("refused", why))
            conn.close()
            return
        c = _Connection(conn, hosted, transport)
        unit = hosted.unit
        welcome = {
            "kind": unit.kind,
            "max_batch": getattr(unit, "max_batch", 1),
            "protocol": unit.protocol,
            "spec": unit.spec(),
            "shape": hosted.shape,
        }
        conn.send(("welcome", welcome))
        self.inbox.put(("open", c, None))
        while c.alive:
            try:
                message = conn.recv()
            except wire.CLOSED:
                break
            if message[0] == "bye":
                break
            if message[0] == "release":
                c.free(message[1])
                continue
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
                case "start":
                    self._start(c.hosted)
                case "stop":
                    self._stop(c.hosted)
                case "started":
                    message.up, message.starting = True, False
                case "failed":
                    h, why = message
                    h.starting, h.error = False, why
            self._states()

    def _start(self, h: _Hosted) -> None:
        """Start ``h`` on a thread of its own; the pump hears back (``started`` / ``failed``)."""
        if h.up or h.starting:
            return
        h.starting, h.error = True, None

        def run():
            try:
                h.unit.start()
            except Exception as exc:  # noqa: BLE001
                self.inbox.put(("failed", None, (h, f"{type(exc).__name__}: {exc}")))
                return
            self.inbox.put(("started", None, h))

        threading.Thread(target=run, name=f"host-start-{h.unit.name}", daemon=True).start()

    def _stop(self, h: _Hosted) -> None:
        if h.up:
            h.up = False
            h.unit.stop()

    def _submit(self, c: _Connection, ticket, packet) -> None:
        c.seen += 1
        try:
            payloads, _ = c.codec.decode(packet, copy=True)
        except Exception as exc:  # noqa: BLE001
            c.send(("done", ticket, self._packet(c, ticket, [("item", f"decode: {exc!r}")])))
            return
        finally:
            if packet.blocks:
                c.send(("release", ticket))
        if not c.hosted.up:
            why = ("unit", f"{c.unit.name} is not started", None)
            c.send(("done", ticket, self._packet(c, ticket, [why] * len(payloads))))
            return
        items = [
            WorkItem(c.unit.name, {"payload": p}, PAYLOAD, protocol=c.unit.protocol)
            for p in payloads
        ]
        work = items if c.unit.kind == "batch" else items[0]

        def done(_, results, c=c, ticket=ticket):
            c.tickets.pop(ticket, None)
            c.send(("done", ticket, self._packet(c, ticket, [wire.encode(r) for r in results])))

        try:
            c.tickets[ticket] = c.unit.submit(work, done)
        except Exception as exc:  # noqa: BLE001
            c.send(("done", ticket, self._packet(c, ticket, [("item", repr(exc))] * len(items))))

    def _packet(self, c: _Connection, ticket, results: list):
        """``results`` as a packet; its blocks are held until the remote unit has read them."""
        try:
            packet, owned = c.codec.encode(results)
        except Exception as exc:  # noqa: BLE001
            packet, owned = c.codec.encode([("item", f"encode: {exc!r}")] * len(results))
        if owned:
            with c.send_lock:
                c.held[ticket] = owned
        return packet

    def _close(self, c: _Connection) -> None:
        if c in self.connections:
            self.connections.remove(c)
        for ticket in list(c.tickets.values()):
            c.unit.cancel(ticket)
        c.free()
        # ``stop`` may have closed it already
        with contextlib.suppress(OSError):
            c.conn.close()

    def _states(self) -> None:
        now = time.monotonic()
        for c in list(self.connections):
            h = c.hosted
            state = {"capacity": 0, "running": 0, "waiting": 0, "extra": {}}
            if h.up:
                load = c.unit.load()
                state = {
                    "capacity": c.unit.capacity(),
                    "running": load.running,
                    "waiting": load.waiting,
                    "extra": load.extra,
                }
            state.update(ready=h.up and bool(c.unit.ready()), error=h.error, seen=c.seen)
            if state != c.last_state or now - c.last_sent >= self.heartbeat_s:
                c.last_state, c.last_sent = state, now
                c.send(("state", state))
