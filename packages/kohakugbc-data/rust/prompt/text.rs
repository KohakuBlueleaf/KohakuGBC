//! The prompt DAG as text, and text back as the DAG (`kohakugbc_data.prompt.text`).

use std::collections::HashMap;
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;

use super::coords::{Coords, HEADER};
use super::dag::{check, Dag, PVertex, Relation, BOX_TEXT, MENTION};
use super::pystr::{rstrip, splitlines, strip};

const INDENT: &str = "  ";
static HEAD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"^\[([^\[\]\n]+)\](?: {BOX_TEXT})?(?: (.+))?$")).unwrap()
});
static RELATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^relation((?: \[[^\[\]\n]+\])*):(?: (.+))?$").unwrap());
static TAGS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^tags:(?: (.+))?$").unwrap());

fn head(handle: &str, v: &PVertex, coords: &Coords) -> String {
    let mut s = format!("[{handle}]");
    if let Some(b) = &v.bbox {
        let [a, b2, c, d] = coords.write(b);
        s += &format!(" [{a},{b2},{c},{d}]");
    }
    if let Some(t) = &v.text {
        s.push(' ');
        s += t;
    }
    s
}

fn tags_line(tags: &[String]) -> String {
    if tags.is_empty() {
        "tags:".into()
    } else {
        format!("tags: {}", tags.join(", "))
    }
}

/// The text lines with each vertex's content under `host[vertex]`, and every vertex's parents
/// in the order its places appear.
fn lines(
    dag: &Dag,
    host: &HashMap<String, String>,
    coords: &Coords,
) -> (Vec<String>, IndexMap<String, Vec<String>>) {
    let vs = &dag.vertices;
    let root = &vs[&dag.root];
    let mut out = vec![format!("# {}", head(&dag.root, root, coords))];
    if let Some(t) = &root.tags {
        out.push(tags_line(t));
    }
    let mut places: IndexMap<String, Vec<String>> = IndexMap::new();
    fn walk(
        dag: &Dag,
        host: &HashMap<String, String>,
        coords: &Coords,
        h: &str,
        depth: usize,
        out: &mut Vec<String>,
        places: &mut IndexMap<String, Vec<String>>,
    ) {
        let vs = &dag.vertices;
        let pad = INDENT.repeat(depth);
        for c in &vs[h].children {
            places.entry(c.clone()).or_default().push(h.to_string());
            if host[c] != h {
                out.push(format!("{pad}- [{c}]"));
                continue;
            }
            out.push(format!("{pad}- {}", head(c, &vs[c], coords)));
            if let Some(t) = &vs[c].tags {
                out.push(format!("{pad}{INDENT}{}", tags_line(t)));
            }
            walk(dag, host, coords, c, depth + 1, out, places);
        }
        for r in &vs[h].relations {
            let who: String = r.participants.iter().map(|p| format!(" [{p}]")).collect();
            let text = if r.text.is_empty() {
                String::new()
            } else {
                format!(" {}", r.text)
            };
            out.push(format!("{pad}- relation{who}:{text}"));
        }
    }
    walk(dag, host, coords, &dag.root, 0, &mut out, &mut places);
    (out, places)
}

/// The parent each vertex's content sits under: its `first` or `last` place in the text.
pub fn hosts(dag: &Dag, placement: &str) -> Result<HashMap<String, String>, String> {
    if placement != "first" && placement != "last" {
        return Err(format!("placement {placement:?}: one of (first, last)"));
    }
    let mut host: HashMap<String, String> = HashMap::new();
    fn first(dag: &Dag, h: &str, host: &mut HashMap<String, String>) {
        for c in &dag.vertices[h].children {
            if !host.contains_key(c) {
                host.insert(c.clone(), h.to_string());
                first(dag, c, host);
            }
        }
    }
    first(dag, &dag.root, &mut host);
    if placement == "first" {
        return Ok(host);
    }

    // "last": moving a vertex's content moves only its own subtree, so re-placing at the last
    // place until nothing moves settles; any host is a real parent, so every round is valid
    let coords = Coords::default();
    for _ in 0..(4 * dag.vertices.len() + 1) {
        let (_, places) = lines(dag, &host, &coords);
        let moved: Vec<(String, String)> = places
            .iter()
            .filter_map(|(c, ps)| {
                let last = ps.last().unwrap();
                (host[c] != *last).then(|| (c.clone(), last.clone()))
            })
            .collect();
        if moved.is_empty() {
            break;
        }
        host.extend(moved);
    }
    Ok(host)
}

pub fn render(dag: &Dag, coords: &Coords, placement: &str, header: bool) -> Result<String, String> {
    check(dag)?;
    let (body, _) = lines(dag, &hosts(dag, placement)?, coords);
    let mut out = String::new();
    if header {
        out += &coords.header();
        out.push('\n');
    }
    for l in body {
        out += &l;
        out.push('\n');
    }
    Ok(out)
}

fn merge(
    handle: &str,
    v: &mut PVertex,
    text: Option<String>,
    bbox: Option<[f64; 4]>,
    tags: Option<Vec<String>>,
) -> Result<(), String> {
    fn one<T: PartialEq + std::fmt::Debug>(
        h: &str,
        name: &str,
        old: &mut Option<T>,
        new: Option<T>,
    ) -> Result<(), String> {
        let Some(new) = new else { return Ok(()) };
        if let Some(o) = old {
            if *o != new {
                return Err(format!(
                    "[{h}]: {name} given twice and differs: {o:?} / {new:?}"
                ));
            }
        }
        *old = Some(new);
        Ok(())
    }
    one(handle, "text", &mut v.text, text)?;
    one(handle, "box", &mut v.bbox, bbox)?;
    one(handle, "tags", &mut v.tags, tags)
}

type Parts = (String, Option<String>, Option<[f64; 4]>);

fn parts(c: &regex::Captures, coords: &Coords, n: usize) -> Result<Parts, String> {
    let bbox = match c.get(2) {
        None => None,
        Some(_) => {
            let mut ints = [0i64; 4];
            for (k, slot) in ints.iter_mut().enumerate() {
                *slot = c[k + 2]
                    .parse()
                    .map_err(|_| format!("line {n}: box value {:?} is out of range", &c[k + 2]))?;
            }
            Some(coords.read(ints).map_err(|e| format!("line {n}: {e}"))?)
        }
    };
    Ok((
        c[1].to_string(),
        c.get(6).map(|m| m.as_str().to_string()),
        bbox,
    ))
}

/// The DAG a prompt text says, its boxes read in the header's coords (else `coords`).
pub fn parse(text: &str, coords: &Coords) -> Result<Dag, String> {
    let mut lines: Vec<(usize, &str)> = splitlines(text)
        .into_iter()
        .enumerate()
        .filter(|(_, l)| !strip(l).is_empty())
        .map(|(i, l)| (i + 1, rstrip(l)))
        .collect();
    let mut coords = coords.clone();
    if let Some(&(n, first)) = lines.first() {
        if HEADER.is_match(first) {
            coords = Coords::from_header(first).map_err(|e| format!("line {n}: {e}"))?;
            lines.remove(0);
        }
    }
    let Some(&(n, first)) = lines.first().filter(|(_, l)| l.starts_with("# ")) else {
        return Err("a prompt starts with the root, `# [handle] ...`".into());
    };
    let m = HEAD
        .captures(&first[2..])
        .ok_or_else(|| format!("line {n}: the root is `# [handle] [a,b,c,d] text`"))?;
    let (root, rtext, rbox) = parts(&m, &coords, n)?;
    let mut vs: IndexMap<String, PVertex> = IndexMap::new();
    vs.insert(
        root.clone(),
        PVertex {
            text: rtext,
            bbox: rbox,
            ..Default::default()
        },
    );
    let mut parents: Vec<String> = vec![root.clone()];
    let mut last: (Option<String>, i64) = (Some(root.clone()), -1);
    for &(n, line) in &lines[1..] {
        let body = line.trim_start_matches(' ');
        let lead = line.chars().count() - body.chars().count();
        if lead % INDENT.len() != 0 {
            return Err(format!(
                "line {n}: indent of {lead} is not a multiple of {}",
                INDENT.len()
            ));
        }
        let depth = lead / INDENT.len();
        if let Some(t) = TAGS.captures(body) {
            if depth as i64 != last.1 + 1 || last.0.is_none() {
                return Err(format!(
                    "line {n}: a tags line sits one level under its item"
                ));
            }
            let tags: Vec<String> = t
                .get(1)
                .map_or("", |m| m.as_str())
                .split(',')
                .map(strip)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            let h = last.0.clone().unwrap();
            merge(&h, vs.get_mut(&h).unwrap(), None, None, Some(tags))?;
            continue;
        }
        if !body.starts_with("- ") || depth >= parents.len() {
            return Err(format!(
                "line {n}: an item is `- [handle] ...` at most one level deeper"
            ));
        }
        let parent = parents[depth].clone();
        if let Some(r) = RELATION.captures(&body[2..]) {
            let rtext = r.get(2).map_or("", |m| m.as_str());
            let mut who: Vec<String> = Vec::new();
            let named = MENTION
                .captures_iter(&r[1])
                .chain(MENTION.captures_iter(rtext));
            for c in named {
                if !who.iter().any(|w| w == &c[1]) {
                    who.push(c[1].to_string());
                }
            }
            vs[&parent].relations.push(Relation {
                participants: who,
                text: rtext.to_string(),
            });
            parents.truncate(depth + 1);
            last = (None, -2);
            continue;
        }
        let m = HEAD
            .captures(&body[2..])
            .ok_or_else(|| format!("line {n}: an item is `- [handle] [a,b,c,d] text`"))?;
        let (handle, itext, bbox) = parts(&m, &coords, n)?;
        if handle == root {
            return Err(format!(
                "line {n}: the root [{root}] cannot sit under another vertex"
            ));
        }
        let v = vs.entry(handle.clone()).or_default();
        merge(&handle, v, itext, bbox, None)?;
        if vs[&parent].children.contains(&handle) {
            return Err(format!("line {n}: [{handle}] is under [{parent}] twice"));
        }
        vs[&parent].children.push(handle.clone());
        parents.truncate(depth + 1);
        parents.push(handle.clone());
        last = (Some(handle), depth as i64);
    }
    let dag = Dag { root, vertices: vs };
    check(&dag)?;
    Ok(dag)
}
