//! Records in, graph out: the data of one resolution.

use serde::Serialize;

/// `[entity_id, label, text]` of a record's ids or children.
#[derive(Clone, Debug)]
pub struct EntityRef {
    pub id: String,
    pub label: String,
    pub text: Option<String>,
    /// The id ends in `_<digits>` (Python's `str.isdigit` on the part after the last `_`).
    pub numbered: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Bbox {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub confidence: Option<f64>,
}

impl Bbox {
    pub const FULL: Bbox = Bbox {
        left: 0.0,
        top: 0.0,
        right: 1.0,
        bottom: 1.0,
        confidence: None,
    };
}

/// `[text, desc_label, [entity_id, ...]]`.
#[derive(Clone, Debug)]
pub struct Desc {
    pub text: String,
    pub label: String,
    pub ids: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Record {
    pub ids: Vec<EntityRef>,
    pub bbox: Option<Bbox>,
    pub descs: Vec<Desc>,
    pub children: Vec<EntityRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Edge {
    pub source: String,
    pub text: String,
    pub target: String,
}

#[derive(Debug, Serialize)]
pub struct ClipScores {
    pub scores: Option<()>,
    pub truncation: bool,
}

#[derive(Debug, Serialize)]
pub struct Caption {
    pub text: String,
    pub label: String,
    pub full_label: String,
    pub statistics: Option<()>,
    pub clip_scores: ClipScores,
    pub toxicity_scores: Option<()>,
}

#[derive(Debug, Serialize)]
pub struct Vertex {
    pub vertex_id: String,
    pub bbox: Bbox,
    pub label: String,
    pub descs: Vec<Caption>,
    pub in_edges: Vec<Edge>,
    pub out_edges: Vec<Edge>,
    pub sub_masks: Vec<String>,
    pub super_masks: Vec<String>,
}

/// The dict `GbcGraphFull.model_dump()` gives, field for field.
#[derive(Debug, Serialize)]
pub struct Graph {
    pub vertices: Vec<Vertex>,
    pub img_url: Option<()>,
    pub img_path: Option<String>,
    pub original_caption: Option<String>,
    pub short_caption: Option<String>,
    pub detail_caption: Option<String>,
    pub img_size: Option<[i64; 2]>,
    pub mask_inside_threshold: f64,
}

pub const DESC_LABELS: [&str; 7] = [
    "short",
    "relation",
    "composition",
    "detail",
    "original",
    "hardcode",
    "bagofwords",
];
pub const VERTEX_LABELS: [&str; 4] = ["image", "entity", "composition", "relation"];
