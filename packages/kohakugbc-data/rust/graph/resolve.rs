//! Resolution of one image's records into a GBC graph, written from ml-gbc's definition of the
//! conversion; the Python reference (`kohakugbc.graph.reference`) is the oracle it is checked
//! against by digest.

use std::collections::HashMap;

use indexmap::IndexMap;

use super::model::{
    Bbox, Caption, ClipScores, Desc, Edge, EntityRef, Graph, Record, Vertex, DESC_LABELS,
    VERTEX_LABELS,
};

pub type Error = String;

/// The GBC graph of one image, or None when no record has a description.
pub fn resolve(
    records: &[Record],
    img_path: Option<String>,
    img_size: Option<[i64; 2]>,
    threshold: f64,
) -> Result<Option<Graph>, Error> {
    let (nodes, mapping) = parse(records)?;
    if nodes.is_empty() {
        return Ok(None);
    }
    let mut vertices = nodes
        .values()
        .map(|node| vertex(node, &mapping))
        .collect::<Result<Vec<_>, _>>()?;
    link(&mut vertices);
    mask(&mut vertices, threshold);
    let (original, short, detail) = captions(&vertices);
    Ok(Some(Graph {
        vertices,
        img_url: None,
        img_path,
        original_caption: original,
        short_caption: short,
        detail_caption: detail,
        img_size,
        mask_inside_threshold: threshold,
    }))
}

type Mapping = HashMap<String, Vec<String>>;

// ---- ids, numbered instances, relation split ------------------------------------------------

fn parse(records: &[Record]) -> Result<(IndexMap<String, Record>, Mapping), Error> {
    // Vertex id -> its record, in first-seen order; entity id -> the vertex ids it points to
    let mut nodes: IndexMap<String, Record> = IndexMap::new();
    let mut mapping: Mapping = HashMap::new();
    let mut numbered: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut relational: Vec<&Record> = Vec::new();

    for rec in records {
        if rec.descs.is_empty() {
            continue;
        }
        let Some(first) = rec.ids.first() else {
            return Err("a record without ids".into());
        };
        if first.label == "relation" {
            relational.push(rec);
            continue;
        }

        // Every joined id maps to the first; numbered ids (cup_1) gather under their stem
        nodes.insert(first.id.clone(), rec.clone());
        for e in &rec.ids {
            if mapping.contains_key(&e.id) {
                return Err(format!("entity id {:?} appears twice", e.id));
            }
            if e.label != first.label {
                return Err(format!(
                    "joined ids of {:?} have labels {}, {}",
                    first.id, first.label, e.label
                ));
            }
            mapping.insert(e.id.clone(), vec![first.id.clone()]);
            if e.numbered {
                let stem = e.id.rsplit_once('_').map_or("", |(s, _)| s);
                numbered
                    .entry(stem.to_string())
                    .or_default()
                    .push(first.id.clone());
            }
        }
    }

    // A stem with no composition vertex points to its instances
    for (stem, firsts) in numbered {
        mapping.entry(stem).or_insert(firsts);
    }

    for rec in relational {
        let split = split_relation(rec, &nodes, &mapping)?;
        for (id, node) in split {
            mapping
                .entry(id.clone())
                .or_insert_with(|| vec![id.clone()]);
            nodes.insert(id, node);
        }
    }
    Ok((nodes, mapping))
}

fn node<'a>(
    id: &str,
    nodes: &'a IndexMap<String, Record>,
    mapping: &Mapping,
) -> Result<(&'a Record, String), Error> {
    let vertex_id = mapping
        .get(id)
        .and_then(|v| v.first())
        .ok_or_else(|| format!("a relation names {id:?}, which has no vertex"))?;
    Ok((&nodes[vertex_id.as_str()], vertex_id.clone()))
}

/// One vertex per set of entities the answers mention, linked from the asked vertex. The asked
/// vertex comes back with its new children, at the position of the first split vertex.
fn split_relation(
    rec: &Record,
    nodes: &IndexMap<String, Record>,
    mapping: &Mapping,
) -> Result<IndexMap<String, Record>, Error> {
    if rec.ids.len() != 1 {
        return Err("a relation record names exactly one asked vertex".into());
    }
    let (asked, asked_id) = node(&rec.ids[0].id, nodes, mapping)?;
    let mut parent = asked.clone();

    let mut split: IndexMap<String, Record> = IndexMap::new();
    for d in &rec.descs {
        let mut assoc = d.ids.clone();
        assoc.sort();
        let vertex_id = format!("[{}]", assoc.join("|"));
        if !split.contains_key(&vertex_id) {
            let mut children = Vec::with_capacity(assoc.len());
            let mut boxes = Vec::with_capacity(assoc.len());
            for id in &assoc {
                let (child, _) = node(id, nodes, mapping)?;
                // The edge label is the id's last part, by ml-gbc's design
                let text = id.rsplit_once('_').map_or(id.as_str(), |(_, t)| t);
                children.push(EntityRef {
                    id: id.clone(),
                    label: child.ids[0].label.clone(),
                    text: Some(text.to_string()),
                    numbered: false,
                });
                boxes.push(child.bbox);
            }
            for c in &children {
                parent.children.push(EntityRef {
                    id: vertex_id.clone(),
                    label: "relation".into(),
                    text: c.text.clone(),
                    numbered: false,
                });
            }
            let bbox = union(&boxes, &vertex_id)?;
            split.insert(
                vertex_id.clone(),
                Record {
                    ids: vec![EntityRef {
                        id: vertex_id.clone(),
                        label: "relation".into(),
                        text: None,
                        numbered: false,
                    }],
                    bbox: Some(bbox),
                    descs: Vec::new(),
                    children,
                },
            );
            // Placeholder that keeps the asked vertex's position; filled once all are known
            if !split.contains_key(&asked_id) {
                split.insert(asked_id.clone(), asked.clone());
            }
        }
        split[&vertex_id].descs.push(Desc {
            text: d.text.clone(),
            label: d.label.clone(),
            ids: vec![vertex_id.clone()],
        });
    }
    if let Some(slot) = split.get_mut(&asked_id) {
        *slot = parent;
    }
    Ok(split)
}

fn union(boxes: &[Option<Bbox>], vertex_id: &str) -> Result<Bbox, Error> {
    let mut it = boxes.iter();
    let first = it
        .next()
        .ok_or_else(|| format!("relation vertex {vertex_id} has no entities"))?;
    let none = || format!("relation vertex {vertex_id} includes a vertex without a box");
    let mut u = first.ok_or_else(none)?;
    u.confidence = None;
    for b in it {
        let b = b.ok_or_else(none)?;
        // Python's min / max keep the first of equal values
        if b.left < u.left {
            u.left = b.left;
        }
        if b.top < u.top {
            u.top = b.top;
        }
        if b.right > u.right {
            u.right = b.right;
        }
        if b.bottom > u.bottom {
            u.bottom = b.bottom;
        }
    }
    Ok(u)
}

// ---- vertices -----------------------------------------------------------------------------

/// Python's `str.isspace` on every character (`str.strip() == ""`).
fn blank(text: &str) -> bool {
    text.chars()
        .all(|c| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

fn vertex(node: &Record, mapping: &Mapping) -> Result<Vertex, Error> {
    let head = &node.ids[0];
    let id = &head.id;
    if mapping.get(id).map(|v| v.as_slice()) != Some(std::slice::from_ref(id)) {
        return Err(format!("vertex id {id:?} does not map to itself"));
    }
    let label = if VERTEX_LABELS.contains(&head.label.as_str()) {
        head.label.clone()
    } else {
        "entity".to_string()
    };

    let mut descs = Vec::with_capacity(node.descs.len());
    for d in &node.descs {
        if !DESC_LABELS.contains(&d.label.as_str()) {
            return Err(format!("description label {:?}", d.label));
        }
        descs.push(Caption {
            text: d.text.clone(),
            label: d.label.clone(),
            full_label: format!("{}-{}", d.label, label),
            statistics: None,
            clip_scores: ClipScores {
                scores: None,
                truncation: false,
            },
            toxicity_scores: None,
        });
    }

    // One out-edge per named phrase to every vertex its id maps to
    let mut out_edges = Vec::new();
    for c in &node.children {
        let Some(text) = &c.text else {
            return Err(format!("vertex {id:?} names a child without text"));
        };
        if blank(text) {
            continue;
        }
        for target in mapping.get(&c.id).into_iter().flatten() {
            out_edges.push(Edge {
                source: id.clone(),
                text: text.clone(),
                target: target.clone(),
            });
        }
    }

    Ok(Vertex {
        vertex_id: id.clone(),
        bbox: node.bbox.unwrap_or(Bbox::FULL),
        label,
        descs,
        in_edges: Vec::new(),
        out_edges,
        sub_masks: Vec::new(),
        super_masks: Vec::new(),
    })
}

// ---- in-edges and masks -------------------------------------------------------------------

/// Vertex indices sorted by id (byte order of UTF-8 is code point order, as Python sorts).
fn sorted_ids(vertices: &[Vertex]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..vertices.len()).collect();
    order.sort_by(|&a, &b| vertices[a].vertex_id.cmp(&vertices[b].vertex_id));
    order
}

/// In-edges from out-edges: each child gathers its in-edges by source in id order, each source's
/// in its out-edge order, without duplicates; out-edges to unknown vertices are dropped.
fn link(vertices: &mut [Vertex]) {
    let mut index: HashMap<String, usize> = HashMap::new();
    for (i, v) in vertices.iter().enumerate() {
        index.insert(v.vertex_id.clone(), i);
    }
    for s in sorted_ids(vertices) {
        let out = std::mem::take(&mut vertices[s].out_edges);
        let mut kept = Vec::with_capacity(out.len());
        for e in out {
            let Some(&c) = index.get(&e.target) else {
                continue;
            };
            if !vertices[c].in_edges.contains(&e) {
                vertices[c].in_edges.push(e.clone());
            }
            kept.push(e);
        }
        vertices[s].out_edges = kept;
    }
}

fn area(left: f64, top: f64, right: f64, bottom: f64) -> f64 {
    (right - left).max(0.0) * (bottom - top).max(0.0)
}

fn mostly_inside(inner: &Bbox, outer: &Bbox, threshold: f64) -> bool {
    let a = area(inner.left, inner.top, inner.right, inner.bottom);
    let overlap = area(
        inner.left.max(outer.left),
        inner.top.max(outer.top),
        inner.right.min(outer.right),
        inner.bottom.min(outer.bottom),
    );
    let ratio = if a > 0.0 { overlap / a } else { 1.0 };
    ratio > threshold
}

/// For every vertex, the others (in id order) mostly inside it, and those it is mostly inside.
fn mask(vertices: &mut [Vertex], threshold: f64) {
    let order = sorted_ids(vertices);
    let boxes: Vec<Bbox> = vertices.iter().map(|v| v.bbox).collect();
    for &v in &order {
        let mut sub = Vec::new();
        let mut sup = Vec::new();
        for &a in &order {
            if vertices[a].vertex_id == vertices[v].vertex_id {
                continue;
            }
            if mostly_inside(&boxes[a], &boxes[v], threshold) {
                sub.push(vertices[a].vertex_id.clone());
            }
            if mostly_inside(&boxes[v], &boxes[a], threshold) {
                sup.push(vertices[a].vertex_id.clone());
            }
        }
        vertices[v].sub_masks = sub;
        vertices[v].super_masks = sup;
    }
}

// ---- graph captions -----------------------------------------------------------------------

type Captions = (Option<String>, Option<String>, Option<String>);

fn captions(vertices: &[Vertex]) -> Captions {
    let (mut original, mut short, mut detail) = (None, None, None);
    if let Some(image) = vertices.iter().find(|v| v.label == "image") {
        for d in &image.descs {
            let slot = match d.label.as_str() {
                "original" => &mut original,
                "short" => &mut short,
                "detail" => &mut detail,
                _ => continue,
            };
            if slot.is_none() {
                *slot = Some(d.text.clone());
            }
        }
    }
    (original, short, detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(id: &str, label: &str, text: Option<&str>) -> EntityRef {
        let numbered = id
            .rsplit_once('_')
            .is_some_and(|(_, s)| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()));
        EntityRef {
            id: id.into(),
            label: label.into(),
            text: text.map(String::from),
            numbered,
        }
    }

    fn b(l: f64, t: f64, rr: f64, bb: f64) -> Option<Bbox> {
        Some(Bbox {
            left: l,
            top: t,
            right: rr,
            bottom: bb,
            confidence: Some(0.5),
        })
    }

    fn d(text: &str, label: &str, ids: &[&str]) -> Desc {
        Desc {
            text: text.into(),
            label: label.into(),
            ids: ids.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn records() -> Vec<Record> {
        vec![
            Record {
                ids: vec![r("", "image", None)],
                bbox: None,
                descs: vec![d("A girl and two cups.", "detail", &[])],
                children: vec![
                    r("girl", "entity", Some("girl")),
                    r("cup", "entity", Some("cups")),
                ],
            },
            Record {
                ids: vec![r("girl", "entity", Some("girl"))],
                bbox: b(0.1, 0.1, 0.5, 0.9),
                descs: vec![d("A girl.", "detail", &[])],
                children: vec![],
            },
            Record {
                ids: vec![r("cup_1", "entity", Some("cups"))],
                bbox: b(0.6, 0.6, 0.7, 0.7),
                descs: vec![d("A mug.", "detail", &[])],
                children: vec![],
            },
            Record {
                ids: vec![r("cup_2", "entity", Some("cups"))],
                bbox: b(0.75, 0.6, 0.85, 0.7),
                descs: vec![d("A cup.", "detail", &[])],
                children: vec![],
            },
            Record {
                ids: vec![r("", "relation", None)],
                bbox: None,
                descs: vec![d("The girl holds cup 1.", "relation", &["girl", "cup_1"])],
                children: vec![],
            },
        ]
    }

    #[test]
    fn numbered_instances_and_relation_split() {
        let g = resolve(&records(), Some("x.png".into()), None, 0.85)
            .unwrap()
            .unwrap();
        let ids: Vec<&str> = g.vertices.iter().map(|v| v.vertex_id.as_str()).collect();
        assert_eq!(ids, ["", "girl", "cup_1", "cup_2", "[cup_1|girl]"]);

        // "cup" has no composition vertex: the image points to both instances
        let image = &g.vertices[0];
        let targets: Vec<&str> = image.out_edges.iter().map(|e| e.target.as_str()).collect();
        assert_eq!(
            targets,
            ["girl", "cup_1", "cup_2", "[cup_1|girl]", "[cup_1|girl]"]
        );
        assert_eq!(g.vertices[4].bbox.left, 0.1);
        assert_eq!(g.vertices[4].bbox.right, 0.7);
        assert_eq!(g.detail_caption.as_deref(), Some("A girl and two cups."));
    }

    #[test]
    fn unknown_relation_entity_is_an_error() {
        let mut recs = records();
        recs[4].descs[0].ids = vec!["ghost".into()];
        assert!(resolve(&recs, None, None, 0.85).is_err());
    }

    #[test]
    fn python_blank() {
        assert!(blank(" \t\x1f"));
        assert!(!blank(" a "));
    }
}
