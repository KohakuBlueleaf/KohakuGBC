"""Image shards in KohakuClip's layout, written and read here (no KohakuClip needed).

A shard is a stored (uncompressed) zip with zip64 end records: one member ``<name>.<ext>`` per
image, in order, then the image index ``__index__.bin`` as the last member::

    b"KCIMG1\\0\\0" | u64 json length | json header | pad to 8 | one 24-byte record per image

The header is a json object (``{"images": n, ...}``, keys sorted, compact). A record holds,
little-endian: the absolute offset (u64) and size (u32) of the image's bytes in the archive, its
stored width and height (u32 each), codec (u8), color flags (u8) and two pad bytes. The bytes
equal those of ``kohakuclip._core.pack_images`` for the same members and header, so KohakuClip's
``ImageReader`` reads these shards.
"""

import json
import mmap
import os
import struct
import zlib
from pathlib import Path

import numpy as np

INDEX_NAME = "__index__.bin"
MAGIC = b"KCIMG1\0\0"
CODECS = {"jpeg": 1, "av1": 2, "jxl": 3, "webp": 4, "png": 5}
EXTENSIONS = {"jpeg": "jpg", "av1": "obu", "jxl": "jxl", "webp": "webp", "png": "png"}
# KohakuClip's color flags per codec: JPEG / AV1 full-range BT.601, WebP limited-range BT.601,
# JPEG XL / PNG RGB
COLORS = {"jpeg": 2, "av1": 2, "webp": 0, "jxl": 4, "png": 4}
RECORD = np.dtype(
    [
        ("off", "<u8"),
        ("size", "<u4"),
        ("w", "<u4"),
        ("h", "<u4"),
        ("codec", "u1"),
        ("color", "u1"),
        ("pad", "u1", (2,)),
    ]
)

_LOCAL, _CENTRAL = 0x04034B50, 0x02014B50
_EOCD, _EOCD64, _LOCATOR = 0x06054B50, 0x06064B50, 0x07064B50
_ZIP64_EXTRA = 0x0001
_U32 = 0xFFFFFFFF


def index_bytes(header: dict, records: np.ndarray) -> bytes:
    """The ``__index__.bin`` member of ``records`` (``RECORD``) with ``header`` (the image count
    is set here)."""
    head = json.dumps(
        {**header, "images": len(records)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    out = MAGIC + struct.pack("<Q", len(head)) + head
    out += b"\0" * (-len(out) % 8)
    return out + records.astype(RECORD, copy=False).tobytes()


class ImageShardWriter:
    """Writes one shard: ``add`` each image (already encoded), then ``close``."""

    def __init__(self, path: str | os.PathLike, header: dict | None = None):
        self.path = Path(path)
        self.header = dict(header or {})
        self.file = open(self.path, "wb")  # noqa: SIM115 (open until close)
        self.pos = 0
        self.central = bytearray()
        self.members = 0
        self.records: list[tuple] = []

    def _member(self, name: str, data: bytes) -> int:
        """Append a stored member; returns the absolute offset of its data."""
        if len(data) > _U32:
            raise ValueError(f"{name}: zip member over 4 GiB")
        raw = name.encode()
        at = self.pos
        shared = struct.pack(
            "<HHHHHIIIHH", 45, 0, 0, 0, 0x21, zlib.crc32(data), len(data), len(data), len(raw), 0
        )
        local = struct.pack("<I", _LOCAL) + shared + raw
        self.file.write(local)
        self.file.write(data)
        self.pos += len(local) + len(data)

        big = at >= _U32
        entry = struct.pack("<IH", _CENTRAL, 45) + shared[:-2]
        entry += struct.pack("<H", 12 if big else 0) + b"\0" * 10
        entry += struct.pack("<I", _U32 if big else at) + raw
        if big:
            entry += struct.pack("<HHQ", _ZIP64_EXTRA, 8, at)
        self.central += entry
        self.members += 1
        return at + len(local)

    def add(self, name: str, data: bytes, w: int, h: int, codec: str) -> int:
        """Store one image (``name`` with its extension, e.g. ``<key>.webp``); returns its index."""
        off = self._member(name, data)
        self.records.append((off, len(data), w, h, CODECS[codec], COLORS[codec], (0, 0)))
        return len(self.records) - 1

    def close(self) -> None:
        records = np.array(self.records, dtype=RECORD)
        self._member(INDEX_NAME, index_bytes(self.header, records))
        cd_offset, cd_size = self.pos, len(self.central)
        n = self.members
        end = bytes(self.central)
        end += struct.pack("<IQHHIIQQQQ", _EOCD64, 44, 45, 45, 0, 0, n, n, cd_size, cd_offset)
        end += struct.pack("<IIQI", _LOCATOR, 0, cd_offset + cd_size, 1)
        end += struct.pack(
            "<IHHHHIIH",
            _EOCD,
            0,
            0,
            min(n, 0xFFFF),
            min(n, 0xFFFF),
            min(cd_size, _U32),
            min(cd_offset, _U32),
            0,
        )
        self.file.write(end)
        self.file.close()

    def __enter__(self) -> "ImageShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.close()
        else:
            self.file.close()


def _central_directory(f, size: int) -> tuple[int, int, int]:
    """(member count, central directory size, its offset), zip64 aware."""
    tail_len = min(size, 22 + 65535 + 20)
    f.seek(size - tail_len)
    tail = f.read(tail_len)
    at = tail.rfind(struct.pack("<I", _EOCD))
    if at < 0:
        raise ValueError("not a zip archive")
    if at >= 20 and struct.unpack_from("<I", tail, at - 20)[0] == _LOCATOR:
        f.seek(struct.unpack_from("<Q", tail, at - 20 + 8)[0])
        record = f.read(56)
        if struct.unpack_from("<I", record)[0] != _EOCD64:
            raise ValueError("bad zip64 end of central directory")
        return struct.unpack_from("<QQQ", record, 32)
    count, cd_size, cd_offset = struct.unpack_from("<HII", tail, at + 10)
    return count, cd_size, cd_offset


def _data_offset(f, header: int) -> int:
    f.seek(header)
    head = f.read(30)
    name_len, extra_len = struct.unpack_from("<HH", head, 26)
    return header + 30 + name_len + extra_len


class ImageShard:
    """One shard read by its index: ``len``, ``header``, ``records`` (``RECORD``) and the bytes
    of image ``i``. The index must be the last member (as both writers put it)."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.file = open(self.path, "rb")  # noqa: SIM115 (open until close)
        size = os.fstat(self.file.fileno()).st_size
        _, cd_size, cd_offset = _central_directory(self.file, size)
        name = INDEX_NAME.encode()
        for extra_len in (0, 12):
            entry_len = 46 + len(name) + extra_len
            if entry_len > cd_size:
                continue
            self.file.seek(cd_offset + cd_size - entry_len)
            entry = self.file.read(entry_len)
            (sig,) = struct.unpack_from("<I", entry)
            name_len, got_extra, comment = struct.unpack_from("<HHH", entry, 28)
            if (sig, name_len, got_extra, comment) == (_CENTRAL, len(name), extra_len, 0) and (
                entry[46 : 46 + len(name)] == name
            ):
                (header,) = struct.unpack_from("<I", entry, 42)
                if header == _U32:
                    (header,) = struct.unpack_from("<Q", entry, 46 + len(name) + 4)
                break
        else:
            raise ValueError(f"{self.path}: no image index as the last member")
        start = _data_offset(self.file, header)
        self.file.seek(start)
        if self.file.read(8) != MAGIC:
            raise ValueError(f"{self.path}: bad image index magic")
        (head_len,) = struct.unpack("<Q", self.file.read(8))
        self.header = json.loads(self.file.read(head_len))
        records_at = start + 16 + head_len + (-(16 + head_len) % 8)
        count = self.header["images"]
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.records = np.frombuffer(self.map, dtype=RECORD, count=count, offset=records_at)

    def __len__(self) -> int:
        return len(self.records)

    def read(self, i: int) -> bytes:
        r = self.records[i]
        off = int(r["off"])
        return self.map[off : off + int(r["size"])]

    def close(self) -> None:
        self.records = None
        self.map.close()
        self.file.close()
