# Engines on other nodes

A Togeari machine is one dispatcher in one process. An engine on another node runs under an
**engine host** (`kohakugbc.engines.host.EngineHost`, `gbc host`), and the machine's lane holds a
**remote unit** for it (`RemoteUnit`, the `remote` entry). Any entry of any slot runs under a
host: the host builds the slot's units from the same configuration, and the lane takes the
entry's adapter, former and lane options.

```
gbc host configs/engines/vlm.py --slot vlm --listen 0.0.0.0:7501   # on the engine node
```

```python
engines = {
    "vlm": {
        "entry": "remote",
        "hosts": ["10.0.0.5:7501/replay-vlm-0"],
        "authkey_env": "KOHAKUGBC_HOST_KEY",
        "runs": {"entry": "replay", "trace": "runs/traces/poc30/text"},
    }
}
```

When the machine's node cannot build the hosted entry (vLLM on a Linux node, a machine on
Windows), the lane names what it needs instead of `runs`:

```python
engines = {
    "vlm": {
        "entry": "remote",
        "hosts": ["10.0.0.5:7593/vllm-0"],
        "adapter": "vllm",
        "kind": "stream",
    }
}
```

`adapter` is one of `kohakugbc.engines.adapters.ADAPTERS` (`vllm`, `torch`, `torch-text`);
`kind`, `protocol`, `max_batch` and `wait_ms` describe the units.

## Wire

Pickled tuples over `multiprocessing.connection`, authenticated by a shared key
(`KOHAKUGBC_HOST_KEY`): only key holders can send, which is what makes unpickling safe. A unit
says hello with the unit it wants; the host answers with the unit's kind, batch size, protocol
and spec. Submits carry prepared payloads; results come back per ticket; the host sends state
(capacity, load, readiness, and how many of this connection's submits it has taken) when it
changes and at least every 0.2 s.

## The host

One pump thread makes every call into the engines (submit, cancel, capacity, load, as Togeari's
contract asks of a dispatcher); one reader thread per connection hands it the connection's
messages; results go back from the engines' own threads. A hosted item is a Togeari work item
whose adapter returns the prepared payload and passes the raw reply back: the machine decodes it
with the lane's adapter.

## The remote unit

| Member | Behaviour |
|---|---|
| `submit` | queues the work for a sender thread (payloads are prepared there) and returns a ticket |
| `capacity` | the host's last reported capacity, less the submits the host has not taken yet |
| `load` | the host's running and waiting, plus the submits on the way |
| results | a reader thread answers each ticket; a capacity rise wakes the lane |
| connection lost | every open ticket is answered with `UnitError`; `ready()` reconnects |

The lane treats a remote unit like any other: it fails over on `UnitError`, and selection,
batching and admission read its capacity and load.

## Checks

`tests/engines/test_remote.py`: remote stream and batch units keep Togeari's engine contract
(`check_engine`, with the host killed mid-check: the next submit is a `UnitError`); the real
detector in a `gbc host` process gives the machine the PoC's graphs; a machine on Windows runs on
vLLM served by `gbc host` in WSL (`KOHAKUGBC_REMOTE_VLM`).
