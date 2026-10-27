# kohakugbc-data

The data side of [KohakuGBC](https://github.com/KohakuBlueleaf/KohakuGBC) (Graph-Based
Captioning), without the captioning machine: GBC graphs (store, resolution, prompt rendering)
and the GBC dataset format. Depends on numpy, pillow, pyarrow and kohakuvault only; one wheel
per platform for CPython 3.11+.

```bash
pip install kohakugbc-data
```

## Graphs

```python
from kohakugbc_data import GbcGraph

g = GbcGraph.from_json(line)  # one GBC graph (ml-gbc JSON), or from_msgpack
blob = g.to_msgpack()  # typed msgpack + zstd: ~7.6x smaller than JSON
g = GbcGraph.from_msgpack(blob)
text = g.prompt(width=1024, height=1024)  # the GBC prompt text (boxes in MOSS layout space)
```

`GbcGraph.prompt` / `prompt_dag` render natively; `kohakugbc_data.prompt` holds the Python
renderers (`from_gbc`, `render`, `parse`, `to_moss` for the MOSS-Image layout draft, detail
levels in `kohakugbc_data.prompt.detail`), equal to the native ones.

## Datasets

A GBC dataset: images, graphs and a metadata dict per image, joined by a sample key, in
self-contained shards.

```
<root>/dataset.json             format, version, the shard list, the dataset header
<root>/shards/shard_XXXXX/
    images.zip                  KohakuClip image shard: stored zip + __index__.bin
    graphs.db                   KohakuVault "graphs": key -> GbcGraph msgpack (zstd)
    meta.db                     KohakuVault "meta": key -> dict (any content)
    manifest.parquet            key, index, w, h, codec, sizes, extra columns
```

```python
from kohakugbc_data.dataset import DatasetWriter, GbcDataset, read_info, split

ds = GbcDataset(root, image="pil")  # or "bytes" / None; graphs=, meta=, shards=
s = ds[0]  # or ds["<key>"]: key, w, h, codec, image, graph, meta
ds = GbcDataset(root, transform=lambda s: (s.image, s.graph.prompt(width=s.w, height=s.h)))

with DatasetWriter(out, shard_bytes=4 << 30, header={"name": "my-set"}) as w:
    w.add(key, image_bytes, width, height, "webp", graph_line, {"tags": tags})
```

`images.zip` is byte-identical to what KohakuClip writes for the same images, so
`kohakuclip.ImageReader` reads it directly. Shards close at `shard_bytes` and / or
`per_shard`; distributed jobs take whole shards (`split(names, rank, world)`). The full format:
[docs/guides/dataset.md](https://github.com/KohakuBlueleaf/KohakuGBC/blob/main/docs/guides/dataset.md).

## License

Apache-2.0.
