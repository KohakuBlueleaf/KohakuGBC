# Engines in other processes

A Togeari machine is one dispatcher in one process. An engine in another process, on this node
or another, runs under an **engine host** (`kohakugbc.engines.host.EngineHost`, `gbc host`), and
the machine's lane holds a **remote unit** for it (`RemoteUnit`, the `remote` entry): a
connector. Any entry of any slot runs under a host: the host builds the slot's units from the
same configuration, and the lane takes the entry's adapter, former and lane options.

## Hosts the machine starts

`engines["hosts"]` names engine host processes, each with its environment and the slots it
serves; the machine starts them all at once before it builds its lanes, and a `remote` entry
names them in its `hosts`:

```python
engines = {
    "hosts": {
        "vlm0": {"env": {"CUDA_VISIBLE_DEVICES": 0}, "engines": {"vlm": {...}}},
        "aux0": {"env": {"CUDA_VISIBLE_DEVICES": 0}, "engines": {"detect": {...}, "embed": {...}}},
    },
    "vlm": {"entry": "remote", "hosts": ["vlm0"]},
    "detect": {"entry": "remote", "hosts": ["aux0"]},
    "embed": {"entry": "remote", "hosts": ["aux0"]},
}
```

A spawned host (`kohakugbc.engines.spawn.HostProcess`) is `gbc host --engines-file F
--ready-file R --listen 127.0.0.1:0 --lazy --watch-stdin`: its environment is set before
anything in it loads (one card through `CUDA_VISIBLE_DEVICES`, seen as `cuda:0`), it builds its
slots (their placement checked on its card), listens, and writes its address, its units per slot
and each slot's shape (adapter, former, select, lane options; pickled) to the ready file. It
loads a unit when that unit's remote unit starts, and stops it when the remote unit stops. It
ends when the machine's process ends (its stdin closes); a remote unit that finds its host's
process ended starts it again. Units are named `<slot>@<host>` (`<slot><i>@<host>` for several).
A host's `after` names hosts whose units are up before its own load, and its `order` stages its
own slots (`[["vlm"], ["detect", "embed"]]`): vLLM refuses to start with less than its
`gpu_memory_utilization` of the card free, so a card's VLM loads before what shares its card.
`qwen35_card.py` puts each card's engines in one host (`GBC_HOST_LAYOUT=card`: the in-process
machine's memory per card) or in a VLM host and a SAM 3 + EmbeddingGemma host (`split`: one more
CUDA context and allocator per card; on the RTX 5090 at a 12 GiB KV cache the second host runs
out of memory by 32 MiB).
With every slot on hosts, the machine's process holds connectors only and never touches a card.
`configs/machine/qwen35_card.py` runs this way by default (`GBC_HOSTING=process`): per card one
VLM host and one host for SAM 3 and EmbeddingGemma.

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
(`KOHAKUGBC_HOST_KEY`; a spawned host gets a random one): only key holders can send, which is
what makes unpickling safe. A unit says hello with the unit it wants and its transport; the host
answers with the unit's kind, batch size, protocol, spec and slot shape. Submits carry prepared
payloads and results come back per ticket, each as a packet (`kohakugbc.engines.host.wire`):
under `shm` (one node; spawned hosts' default) a buffer of 1 MiB and up (an image's pixels, a
tensor) travels in a shared-memory block its writer holds until the reader has copied it and
sent `release`; under `inband` everything is in the message. The host sends state (capacity,
load, readiness, a start's error, and how many of this connection's submits it has taken) when
it changes and at least every 0.2 s.

## The host

One pump thread makes every call into the engines (submit, cancel, capacity, load, as Togeari's
contract asks of a dispatcher); one reader thread per connection hands it the connection's
messages; results go back from the engines' own threads. An eager host starts its units before
it listens; a lazy one starts a unit on a thread of its own when a remote unit asks, and the
pump leaves it alone until it is up. A hosted item is a Togeari work item whose adapter returns
the prepared payload and passes the raw reply back: the machine decodes it with the lane's
adapter.

## The remote unit

| Member | Behaviour |
|---|---|
| `start` | connects; for a spawned host's unit, asks the host to start it and waits until it is up |
| `submit` | queues the work for a sender thread (payloads are prepared and encoded there) and returns a ticket |
| `capacity` | the host's last reported capacity, less the submits the host has not taken yet |
| `load` | the host's running and waiting, plus the submits on the way |
| results | a reader thread answers each ticket; a capacity rise wakes the lane |
| connection lost | every open ticket is answered with `UnitError`; `ready()` reconnects (starting an ended spawned host's process again, and its unit) |

The lane treats a remote unit like any other: it fails over on `UnitError`, and selection,
batching and admission read its capacity and load.

## Checks

Runs on the cards: `scripts/bench/machine/card.sh` runs the machine on its engine hosts
(`configs/machine/qwen35_card.py`, `GBC_HOSTING=process`), every metric on disk; the numbers are
in [the flagship run](../performance/flagship.md).
