//! The prompt DAG and its check (`kohakugbc_data.prompt.dag`).

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;

use super::pystr::strip;

pub static MENTION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\[\]\n]+)\]").unwrap());
pub const BOX_TEXT: &str = r"\[(-?[0-9]+),(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)\]";
static BOX_LIKE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!("^{BOX_TEXT}( |$)")).unwrap());

#[derive(Clone, Debug, PartialEq)]
pub struct Relation {
    pub participants: Vec<String>,
    pub text: String,
}

/// `(left, top, right, bottom)` in [0, 1].
pub type Box4 = [f64; 4];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PVertex {
    pub text: Option<String>,
    pub bbox: Option<Box4>,
    pub tags: Option<Vec<String>>,
    pub children: Vec<String>,
    pub relations: Vec<Relation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dag {
    pub root: String,
    pub vertices: IndexMap<String, PVertex>,
}

fn mention_whole(h: &str) -> bool {
    !h.is_empty() && !h.contains(['[', ']', '\n'])
}

/// `Ok` when `dag` is one DAG under its root that text can carry; the error names why not.
pub fn check(dag: &Dag) -> Result<(), String> {
    let vs = &dag.vertices;
    if !vs.contains_key(&dag.root) {
        return Err(format!("root [{}] is not a vertex", dag.root));
    }
    for (h, v) in vs {
        if h.is_empty() || h != strip(h) || !mention_whole(h) {
            return Err(format!(
                "handle {h:?}: empty, padded, or holds brackets or a newline"
            ));
        }
        if BOX_LIKE.is_match(&format!("[{h}]")) {
            return Err(format!("handle {h:?}: reads as a box"));
        }
        let texts = v.text.iter().chain(v.tags.iter().flatten());
        for s in texts {
            if s.contains('\n') || s != strip(s) || s.is_empty() {
                return Err(format!(
                    "[{h}]: text or tag {s:?} is empty, padded or multi-line"
                ));
            }
        }
        if v.tags.iter().flatten().any(|t| t.contains(',')) {
            return Err(format!("[{h}]: a tag holds a comma"));
        }
        if v.bbox.is_none() && v.text.as_deref().is_some_and(|t| BOX_LIKE.is_match(t)) {
            return Err(format!(
                "[{h}]: text opens like a box but the vertex has none"
            ));
        }
        if let Some([l, t, r, b]) = v.bbox {
            if !(0.0 <= l && l <= r && r <= 1.0 && 0.0 <= t && t <= b && b <= 1.0) {
                return Err(format!(
                    "[{h}]: box {:?} is not (left, top, right, bottom) in [0, 1]",
                    v.bbox
                ));
            }
        }
        if v.children.iter().collect::<HashSet<_>>().len() != v.children.len() {
            return Err(format!("[{h}]: a child listed twice"));
        }
        for c in &v.children {
            if !vs.contains_key(c) || *c == dag.root {
                return Err(format!("[{h}]: child [{c}] is not a vertex below the root"));
            }
        }
        for r in &v.relations {
            if r.text.contains('\n') || r.text != strip(&r.text) {
                return Err(format!(
                    "[{h}]: relation text {:?} is padded or multi-line",
                    r.text
                ));
            }
            let set: HashSet<_> = r.participants.iter().collect();
            if r.participants.is_empty() || set.len() != r.participants.len() {
                return Err(format!(
                    "[{h}]: a relation without participants, or one named twice"
                ));
            }
            if r.participants.iter().any(|p| !vs.contains_key(p)) {
                return Err(format!(
                    "[{h}]: relation participant not a vertex: {:?}",
                    r.participants
                ));
            }
            let named = MENTION.captures_iter(&r.text).map(|c| c[1].to_string());
            if named.into_iter().any(|m| !set.contains(&m)) {
                return Err(format!(
                    "[{h}]: relation text names a handle outside its participants"
                ));
            }
        }
    }

    // Every vertex reachable from the root, and no cycle (colour DFS)
    let mut state: HashMap<&str, u8> = HashMap::new();
    state.insert(&dag.root, 1);
    let mut stack: Vec<(&str, usize)> = vec![(&dag.root, 0)];
    while let Some(&mut (h, ref mut i)) = stack.last_mut() {
        let kids = &vs[h].children;
        if *i == kids.len() {
            state.insert(h, 2);
            stack.pop();
            continue;
        }
        let c = kids[*i].as_str();
        *i += 1;
        match state.get(c) {
            Some(1) => return Err(format!("cycle through [{c}]")),
            Some(_) => {}
            None => {
                state.insert(c, 1);
                stack.push((c, 0));
            }
        }
    }
    if state.len() != vs.len() {
        let mut missing: Vec<&String> = vs
            .keys()
            .filter(|k| !state.contains_key(k.as_str()))
            .collect();
        missing.sort();
        return Err(format!(
            "vertices with no parent path from the root: {missing:?}"
        ));
    }
    Ok(())
}
