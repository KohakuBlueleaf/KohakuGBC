//! One answer line stored as an index-linked graph (msgpack positional arrays); a graph the
//! typed form cannot carry exactly is kept as its raw value. See docs/internals/graph-store.md.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The record layout's version, the first field of every record.
pub const FORMAT: u32 = 1;

/// Desc forms: `{label, text}`; the full ml-gbc caption (`full_label` `<desc>-<vertex>`,
/// null scores); anything else, kept raw.
pub const DESC_SHORT: u8 = 0;
pub const DESC_FULL: u8 = 1;
pub const DESC_RAW: u8 = 2;

/// `(form, label, text, raw)`; `raw` holds the whole desc dict for `DESC_RAW` only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Desc(pub u8, pub String, pub String, pub Option<Value>);

/// Tag scores by name index, in order: all in 1e-4 steps (`.0`), else all as `f64` (`.1`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Scores(pub Vec<(u32, u16)>, pub Vec<(u32, f64)>);

/// `(left, top, right, bottom, confidence)` of the image.
pub type Bbox = (f64, f64, f64, f64, Option<f64>);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vertex {
    pub id: String,
    pub label: String,
    pub bbox: Option<Bbox>,
    pub descs: Vec<Desc>,
    /// `(target vertex, text)`.
    pub out_edges: Vec<(u32, String)>,
    /// Out-edge indices in graph order (vertex by vertex, each vertex's out-edges in order).
    pub in_edges: Vec<u32>,
    pub sub_masks: Vec<u32>,
    pub super_masks: Vec<u32>,
    /// Absent (`None`) is a vertex without a `tags` key.
    pub tags: Option<Scores>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphTags {
    pub image: Scores,
    /// `(tag name, category name)`, both string-table indices.
    pub categories: Vec<(u32, u32)>,
    pub regions: Vec<(String, Scores)>,
    pub crops: Vec<(String, Scores)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub strings: Vec<String>,
    pub vertices: Vec<Vertex>,
    pub img_url: Option<String>,
    pub img_path: Option<String>,
    pub original_caption: Option<String>,
    pub short_caption: Option<String>,
    pub detail_caption: Option<String>,
    pub img_size: Option<Vec<i64>>,
    pub mask_inside_threshold: f64,
    pub tags: Option<GraphTags>,
}

/// One answer line, or one bare graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub format: u32,
    /// The top-level keys around the graph, in order (empty for a bare graph).
    pub head: Vec<(String, Value)>,
    /// Where `graph` sits among the head keys; `None` for a bare graph.
    pub graph_at: Option<u32>,
    /// The graph in typed form, else (`raw`) as stored; both `None` is a null graph.
    pub graph: Option<Graph>,
    pub raw: Option<Value>,
}
