//! A stored GBC graph as the prompt DAG (`kohakugbc_data.prompt.gbc.from_gbc`).

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use super::dag::{check, Dag, PVertex, Relation};
use super::pystr::collapse;
use crate::store::record::{Graph, Scores, Vertex};

pub struct Options {
    pub captions: Vec<String>,
    pub boxes: bool,
    pub tag_threshold: Option<f64>,
    pub tag_spaces: bool,
    pub root: String,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            captions: ["detail", "composition", "short", "hardcode"]
                .map(String::from)
                .to_vec(),
            boxes: true,
            tag_threshold: Some(0.35),
            tag_spaces: true,
            root: "image".into(),
        }
    }
}

fn clean(s: &str) -> Option<String> {
    let c = collapse(s);
    (!c.is_empty()).then_some(c)
}

/// `{target: edge}` in first-seen target order with each target's last edge, split into the
/// edges to relation vertices and the others.
fn kids<'a>(g: &'a Graph, v: &'a Vertex, relation: bool) -> Vec<(u32, &'a str)> {
    let mut order: Vec<u32> = Vec::new();
    let mut text: HashMap<u32, &str> = HashMap::new();
    for (t, s) in &v.out_edges {
        if text.insert(*t, s.as_str()).is_none() {
            order.push(*t);
        }
    }
    order
        .into_iter()
        .filter(|t| (g.vertices[*t as usize].label == "relation") == relation)
        .map(|t| (t, text[&t]))
        .collect()
}

/// Each non-relation vertex's handle, in walk order (the image first).
pub fn handles(g: &Graph, root: &str) -> Result<IndexMap<u32, String>, String> {
    let image = g
        .vertices
        .iter()
        .position(|v| v.label == "image")
        .ok_or("a graph without an image vertex")? as u32;
    let mut handle: IndexMap<u32, String> = IndexMap::new();
    handle.insert(image, root.to_string());
    let mut used: HashSet<String> = HashSet::from([root.to_string()]);
    fn walk(g: &Graph, v: u32, handle: &mut IndexMap<u32, String>, used: &mut HashSet<String>) {
        for (t, text) in kids(g, &g.vertices[v as usize], false) {
            if handle.contains_key(&t) {
                continue;
            }
            let base = clean(text)
                .or_else(|| clean(&g.vertices[t as usize].id))
                .unwrap_or_else(|| "vertex".into())
                .replace('[', "(")
                .replace(']', ")");
            let (mut h, mut k) = (base.clone(), 1);
            while used.contains(&h) {
                k += 1;
                h = format!("{base} {k}");
            }
            used.insert(h.clone());
            handle.insert(t, h);
            walk(g, t, handle, used);
        }
    }
    walk(g, image, &mut handle, &mut used);
    Ok(handle)
}

fn scores<'a>(g: &'a Graph, s: &Scores) -> Vec<(&'a str, f64)> {
    let q =
        s.0.iter()
            .map(|&(i, q)| (g.strings[i as usize].as_str(), q as f64 / 10000.0));
    let f =
        s.1.iter()
            .map(|&(i, x)| (g.strings[i as usize].as_str(), x));
    q.chain(f).collect()
}

/// The DAG of a typed GBC graph, boxes in [0, 1] of the image.
pub fn from_gbc(g: &Graph, o: &Options) -> Result<Dag, String> {
    let handle = handles(g, &o.root)?;
    let image = *handle.keys().next().unwrap();
    let mut vertices: IndexMap<String, PVertex> = IndexMap::new();
    for (&vi, h) in &handle {
        let v = &g.vertices[vi as usize];

        // The first desc of each label, then the first label of `captions` the vertex has
        let mut first: HashMap<&str, &str> = HashMap::new();
        for d in &v.descs {
            first.entry(d.1.as_str()).or_insert(d.2.as_str());
        }
        let text = o
            .captions
            .iter()
            .find_map(|c| first.get(c.as_str()))
            .and_then(|t| clean(t));
        let bbox = match v.bbox {
            Some((l, t, r, b, _)) if o.boxes && vi != image => {
                let c = |x: f64| x.clamp(0.0, 1.0);
                let (x0, y0, x1, y1) = (c(l), c(t), c(r), c(b));
                Some([x0, y0, x0.max(x1), y0.max(y1)])
            }
            _ => None,
        };
        let tags = match (&v.tags, o.tag_threshold) {
            (Some(s), Some(th)) => {
                let mut kept: Vec<(&str, f64)> =
                    scores(g, s).into_iter().filter(|t| t.1 >= th).collect();
                kept.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let names: Vec<String> = kept
                    .into_iter()
                    .map(|(t, _)| {
                        if o.tag_spaces {
                            t.replace('_', " ")
                        } else {
                            t.to_string()
                        }
                    })
                    .collect();
                (!names.is_empty()).then_some(names)
            }
            _ => None,
        };
        let mut relations = Vec::new();
        for (t, _) in kids(g, v, true) {
            let rel = &g.vertices[t as usize];
            let mut who: Vec<String> = Vec::new();
            for (x, _) in &rel.out_edges {
                let name = handle
                    .get(x)
                    .ok_or("a relation names a vertex with no handle")?;
                if !who.contains(name) {
                    who.push(name.clone());
                }
            }
            for d in &rel.descs {
                if d.1 == "relation" {
                    if let Some(text) = clean(&d.2) {
                        let text = text.replace('[', "(").replace(']', ")");
                        relations.push(Relation {
                            participants: who.clone(),
                            text,
                        });
                    }
                }
            }
        }
        let children = kids(g, v, false)
            .into_iter()
            .map(|(t, _)| handle[&t].clone())
            .collect();
        vertices.insert(
            h.clone(),
            PVertex {
                text,
                bbox,
                tags,
                children,
                relations,
            },
        );
    }
    let dag = Dag {
        root: o.root.clone(),
        vertices,
    };
    check(&dag)?;
    Ok(dag)
}
