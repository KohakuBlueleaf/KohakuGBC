"""The GBC dataset format: images, graphs and metadata, sharded, joined by sample key.

::

    <root>/dataset.json             format, version, the shard list (name, samples), header
    <root>/shards/shard_XXXXX/
        images.zip                  KohakuClip image shard (``images``)
        graphs.db                   KohakuVault "graphs": key -> GbcGraph msgpack (zstd)
        meta.db                     KohakuVault "meta": key -> dict (any content)
        manifest.parquet            key, index (in images.zip), w, h, codec, sizes, extras

A shard folder is whole on its own; ``dataset.json`` lists the complete ones. Needs the
``dataset`` extra (kohakuvault, pyarrow).
"""

from kohakugbc_data.dataset.reader import GbcDataset, Sample, read_info, split
from kohakugbc_data.dataset.writer import DatasetWriter

__all__ = ["DatasetWriter", "GbcDataset", "Sample", "read_info", "split"]
