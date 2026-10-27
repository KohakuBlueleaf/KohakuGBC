//! A record as msgpack bytes (positional arrays), in a zstd frame by default.

use super::record::{Record, FORMAT};

const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];
/// zstd 3: 7.6x JSON at 41 us per write; 19 reaches 8.1x (docs/internals/graph-store.md).
pub const DEFAULT_ZSTD: i32 = 3;

pub fn pack(rec: &Record, zstd_level: Option<i32>) -> Result<Vec<u8>, String> {
    let bytes = rmp_serde::to_vec(rec).map_err(|e| format!("msgpack encode: {e}"))?;
    match zstd_level {
        None => Ok(bytes),
        Some(level) => zstd::bulk::compress(&bytes, level).map_err(|e| format!("zstd: {e}")),
    }
}

/// The record in `data`: msgpack, or msgpack in a zstd frame (told apart by the zstd magic).
pub fn unpack(data: &[u8]) -> Result<Record, String> {
    let plain;
    let bytes = if data.starts_with(&ZSTD_MAGIC) {
        plain = zstd::stream::decode_all(data).map_err(|e| format!("zstd: {e}"))?;
        &plain[..]
    } else {
        data
    };
    let rec: Record = rmp_serde::from_slice(bytes).map_err(|e| format!("msgpack decode: {e}"))?;
    if rec.format != FORMAT {
        return Err(format!(
            "record format {} (this build reads {FORMAT})",
            rec.format
        ));
    }
    Ok(rec)
}
