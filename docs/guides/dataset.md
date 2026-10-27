# The GBC dataset format

A GBC dataset holds images, their GBC graphs and a free-form metadata dict per image. Every
part is joined by one **sample key** (the image's id in its source), cut into **shards** that
stand alone. `kohakugbc.io.dataset` writes and reads it (`pip install "kohakugbc[dataset]"`:
kohakuvault, pyarrow).

## Layout

```
<root>/
  dataset.json                  what the dataset is: format, version, shards, header
  shards/
    shard_00000/
      images.zip                the images: a KohakuClip image shard
      graphs.db                 the graphs: KohakuVault, key -> GbcGraph msgpack (zstd)
      meta.db                   the metadata: KohakuVault, key -> dict
      manifest.parquet          one row per sample, in image order
    shard_00001/
    ...
```

| part | stores | why |
|---|---|---|
| `images.zip` | each image as encoded (`<key>.<ext>`), no re-encode | KohakuClip's image shard: a stored zip any zip tool opens, plus an index member read in place, so KohakuClip's `ImageReader` (decode, crop, resize off the GIL) reads it directly |
| `graphs.db` | the graph in KohakuGBC's typed msgpack in a zstd frame (`GbcGraph.to_msgpack`) | about 7.6x smaller than JSON and 5x faster to load ([graph store](../internals/graph-store.md)); lossless (a graph the typed form cannot carry is stored raw) |
| `meta.db` | any msgpack-able dict | whatever the source knows: tags, scores, dates, the caption run; the format sets no schema |
| `manifest.parquet` | `key`, `index`, `w`, `h`, `codec`, `image_bytes`, `graph_bytes`, plus the writer's own columns | filter or sample without opening a shard |

### Images

`images.zip` is byte-identical to what KohakuClip's `pack_images` writes for the same
members and header (checked on 3,000 real images, synthetic JPEG / PNG and a 4.4 GiB zip64
shard). It is a stored (uncompressed) zip with zip64 end records:

- one member `<key>.<ext>` per image, in sample order (`ext`: jpg, webp, png, jxl, obu);
- the last member `__index__.bin`: `KCIMG1\0\0`, a u64 length, a JSON header (`{"images": n,
  ...}`, sorted, compact; the dataset header is copied in), padding to 8 bytes, then one
  24-byte record per image: absolute offset (u64) and size (u32) of its bytes, stored width
  and height (u32 each), codec (1 jpeg, 2 av1, 3 jxl, 4 webp, 5 png), color flags, 2 pad
  bytes.

Images are stored as they were captioned, so the graph's boxes (`[0, 1]` of the image) and
sizes refer to exactly these pixels. A format KohakuClip does not decode (GIF) is stored as
lossless PNG of its first frame.

### Graphs and metadata

Both are KohakuVault files (SQLite) keyed by the sample key, table `graphs` / `meta`. A
finished shard's files use a rollback journal, not WAL: each is one self-contained file,
safe to copy, upload or read from a network filesystem.

### `dataset.json`

```json
{
 "format": "kohakugbc-dataset", "version": 1, "samples": 112802,
 "shards": [{"name": "shard_00000", "samples": 26871, "bytes": 4294970000}, ...],
 "images": "kohakuclip-zip",
 "graphs": {"codec": "gbc-msgpack", "zstd": 3, "table": "graphs"},
 "meta": {"codec": "kohakuvault-autopack", "table": "meta"},
 "header": {"name": "...", "source": "...", "...": "..."}
}
```

It lists only finished shards: a writer rewrites it (atomically) as each shard closes, so a
reader never sees a half-written shard, and an interrupted export keeps what it finished.

## Sharding

A shard is the unit of storage, transfer and distribution:

- **Whole on its own.** A shard folder carries its images, graphs, metadata and manifest.
  Copy, upload, download or delete one without touching the rest.
- **Append-only growth.** A dataset grows by new shards after the last
  (`DatasetWriter` on an existing root continues the numbering); nothing is re-indexed.
- **Split by shards.** Distributed jobs take whole shards (`split(shards, rank, world)`:
  every `world`-th from `rank`), so each rank opens only its own files.

Shard size is the writer's choice: `shard_bytes` (image + graph bytes, default 4 GiB) and / or
`per_shard` (samples); a shard closes at whichever comes first.

| fewer, larger shards | more, smaller shards |
|---|---|
| fewer files and SQLite handles; long sequential reads | finer splits across ranks; smaller units to retry, download or drop |

Rules that settle it:

- at least as many shards as the ranks that will split them (a rank with no shard idles);
- 1-8 GiB per shard (Hugging Face takes files up to 50 GB; a few GiB uploads and resumes
  well; tiny shards turn a dataset into thousands of files);
- the in-process loader is random access across all its shards, so DataLoader workers need
  no more shards than ranks do.

For 100k anime images at a 1024-px side (~16 GB) that is 4 shards of 4 GiB; for 10M images,
a few hundred.

## Reading

```python
from kohakugbc.io.dataset import GbcDataset, read_info, split

ds = GbcDataset("danbooru-gbc-27b")       # every shard; image bytes, graph and meta
len(ds), ds.keys()[:3], "7243972" in ds
s = ds[0]                                 # by position (shard order) ...
s = ds["7243972"]                         # ... or by key
s.key, s.w, s.h, s.codec, s.image         # image: the stored bytes
s.graph                                   # kohakugbc._rs.GbcGraph
s.meta                                    # dict
ds.manifest                               # pyarrow table of every row, plus "shard"
```

| argument | does |
|---|---|
| `shards` | read only these shards (e.g. `split(names, rank, world)`) |
| `image` | `"bytes"` (default), `"pil"` (decoded RGB), `None` (skip) |
| `graphs`, `meta` | `False` skips that store |
| `transform` | `ds[i]` returns `transform(sample)` instead of the `Sample` |

Files open on first use in each process, and a pickled dataset carries no open handle, so one
`GbcDataset` passed to a PyTorch `DataLoader` with workers needs nothing else. KohakuClip can
also read the images alone, with its own crops and decode threads:

```python
from kohakuclip import ImageReader
reader = ImageReader([f"{root}/shards/{n}/images.zip" for n in ds.shards], size=512, threads=16)
batch = reader.read(list(range(256))).image   # uint8 [256, 3, 512, 512]; ids follow ds order
```

## Rendering for training

The dataset stores what was made; turning it into a training sample belongs to the consumer,
in `transform`:

```python
def to_prompt(s):
    text = s.graph.prompt(width=s.w, height=s.h)              # the GBC prompt text
    tags = ", ".join(s.meta.get("general", [])[:30])            # whatever the meta offers
    return s.image, f"{tags}\n{text}"

ds = GbcDataset(root, image="pil", transform=to_prompt)
```

`GbcGraph.prompt` / `prompt_dag` and `kohakugbc.graph.prompt` (`render`, `to_moss`, detail
levels) are the renderers ([graph store and prompts](../internals/graph-store.md)).

## Writing

```python
from kohakugbc.io.dataset import DatasetWriter

with DatasetWriter(root, shard_bytes=4 << 30, header={"name": "my-set"}) as w:
    for key, image_bytes, (width, height), codec, graph_line, meta in source:
        w.add(key, image_bytes, width, height, codec, graph_line, meta, vertices=..., any_column=...)
```

`add(key, image, w, h, codec, graph, meta=None, **row)`: `image` is already encoded (`codec`
one of jpeg, webp, png, jxl, av1; w / h its pixel size); `graph` a `GbcGraph`, its JSON line or
its msgpack; `meta` any dict; `row` keywords become manifest columns. Keys must be unique.
