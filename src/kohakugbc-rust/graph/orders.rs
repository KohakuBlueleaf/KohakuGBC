//! Counts and depth of a resolved graph (`kohakugbc.graph.orders.stats`).

use std::collections::{BTreeMap, HashMap, HashSet};

use super::model::Graph;

pub struct Stats {
    pub vertices: usize,
    pub labels: BTreeMap<String, usize>,
    pub edges: usize,
    pub captions: usize,
    pub depth: i64,
    pub roots: usize,
    pub leaves: usize,
}

/// Image vertices first, then every other vertex without in-edges.
fn roots(g: &Graph) -> Vec<usize> {
    let images = (0..g.vertices.len()).filter(|&i| g.vertices[i].label == "image");
    let others = (0..g.vertices.len())
        .filter(|&i| g.vertices[i].label != "image" && g.vertices[i].in_edges.is_empty());
    images.chain(others).collect()
}

/// The depth of a BFS from the roots.
fn depth(g: &Graph) -> i64 {
    let index: HashMap<&str, usize> = g
        .vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (v.vertex_id.as_str(), i))
        .collect();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut current = roots(g);
    let mut depth = -1;
    while !current.is_empty() {
        depth += 1;
        let mut following = Vec::new();
        for v in current {
            if !seen.insert(v) {
                continue;
            }
            for e in &g.vertices[v].out_edges {
                let t = index[e.target.as_str()];
                if !seen.contains(&t) {
                    following.push(t);
                }
            }
        }
        current = following;
    }
    depth
}

pub fn stats(g: &Graph) -> Stats {
    let mut labels = BTreeMap::new();
    for v in &g.vertices {
        *labels.entry(v.label.clone()).or_insert(0) += 1;
    }
    Stats {
        vertices: g.vertices.len(),
        labels,
        edges: g.vertices.iter().map(|v| v.out_edges.len()).sum(),
        captions: g.vertices.iter().map(|v| v.descs.len()).sum(),
        depth: depth(g),
        roots: roots(g).len(),
        leaves: g.vertices.iter().filter(|v| v.out_edges.is_empty()).count(),
    }
}
