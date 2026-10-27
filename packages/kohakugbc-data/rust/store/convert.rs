//! An answer line's JSON value to its record and back; `from_value` keeps a graph raw unless the
//! typed form gives the same value back.

use std::collections::HashMap;

use serde_json::{Map, Number, Value};

use super::record::{
    Bbox, Desc, Graph, GraphTags, Record, Scores, Vertex, DESC_FULL, DESC_RAW, DESC_SHORT, FORMAT,
};

type R<T> = Result<T, String>;

const VERTEX_KEYS: [&str; 8] = [
    "vertex_id",
    "bbox",
    "label",
    "descs",
    "in_edges",
    "out_edges",
    "sub_masks",
    "super_masks",
];
const GRAPH_KEYS: [&str; 8] = [
    "vertices",
    "img_url",
    "img_path",
    "original_caption",
    "short_caption",
    "detail_caption",
    "img_size",
    "mask_inside_threshold",
];
const TAG_KEYS: [&str; 4] = ["image", "categories", "regions", "crops"];
const SCORE_STEP: f64 = 10000.0;

#[derive(Default)]
struct Strings {
    list: Vec<String>,
    index: HashMap<String, u32>,
}

impl Strings {
    fn id(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.index.get(s) {
            return i;
        }
        let i = self.list.len() as u32;
        self.list.push(s.to_string());
        self.index.insert(s.to_string(), i);
        i
    }
}

fn obj<'a>(v: &'a Value, what: &str) -> R<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| format!("{what} is not an object"))
}

fn arr<'a>(v: &'a Value, what: &str) -> R<&'a Vec<Value>> {
    v.as_array().ok_or_else(|| format!("{what} is not a list"))
}

fn string(v: &Value, what: &str) -> R<String> {
    v.as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{what} is not a string"))
}

fn opt_string(v: Option<&Value>, what: &str) -> R<Option<String>> {
    match v {
        None => Err(format!("{what} is missing")),
        Some(Value::Null) => Ok(None),
        Some(x) => string(x, what).map(Some),
    }
}

/// A float that came from a JSON float (an integer literal would not give itself back).
fn float(v: &Value, what: &str) -> R<f64> {
    match v {
        Value::Number(n) if n.is_f64() => Ok(n.as_f64().unwrap()),
        _ => Err(format!("{what} is not a float")),
    }
}

fn same_keys(m: &Map<String, Value>, want: &[&str], what: &str) -> R<()> {
    if m.len() == want.len() && want.iter().all(|k| m.contains_key(*k)) {
        Ok(())
    } else {
        Err(format!(
            "{what} keys {:?} are not {want:?}",
            m.keys().collect::<Vec<_>>()
        ))
    }
}

fn scores(v: &Value, strings: &mut Strings, what: &str) -> R<Scores> {
    let m = obj(v, what)?;
    let mut pairs = Vec::with_capacity(m.len());
    for (k, s) in m {
        pairs.push((strings.id(k), float(s, what)?));
    }
    let step = |s: f64| (s * SCORE_STEP).round();
    let exact = pairs.iter().all(|&(_, s)| {
        (0.0..=u16::MAX as f64 / SCORE_STEP).contains(&s) && step(s) / SCORE_STEP == s
    });
    Ok(if exact {
        Scores(
            pairs.iter().map(|&(i, s)| (i, step(s) as u16)).collect(),
            Vec::new(),
        )
    } else {
        Scores(Vec::new(), pairs)
    })
}

fn scores_value(s: &Scores, strings: &[String]) -> Value {
    let mut m = Map::new();
    for &(i, q) in &s.0 {
        m.insert(strings[i as usize].clone(), f(q as f64 / SCORE_STEP));
    }
    for &(i, x) in &s.1 {
        m.insert(strings[i as usize].clone(), f(x));
    }
    Value::Object(m)
}

fn f(x: f64) -> Value {
    Value::Number(Number::from_f64(x).expect("a finite float"))
}

fn named_scores(v: &Value, strings: &mut Strings, what: &str) -> R<Vec<(String, Scores)>> {
    obj(v, what)?
        .iter()
        .map(|(k, s)| Ok((k.clone(), scores(s, strings, what)?)))
        .collect()
}

fn graph_tags(v: &Value, strings: &mut Strings) -> R<GraphTags> {
    let m = obj(v, "graph tags")?;
    same_keys(m, &TAG_KEYS, "graph tags")?;
    let categories = obj(&m["categories"], "categories")?
        .iter()
        .map(|(k, c)| Ok((strings.id(k), strings.id(&string(c, "a category")?))))
        .collect::<R<Vec<_>>>()?;
    Ok(GraphTags {
        image: scores(&m["image"], strings, "image tags")?,
        categories,
        regions: named_scores(&m["regions"], strings, "region tags")?,
        crops: named_scores(&m["crops"], strings, "crop tags")?,
    })
}

fn bbox(v: &Value) -> R<Option<Bbox>> {
    if v.is_null() {
        return Ok(None);
    }
    let m = obj(v, "bbox")?;
    same_keys(m, &["left", "top", "right", "bottom", "confidence"], "bbox")?;
    let conf = match &m["confidence"] {
        Value::Null => None,
        c => Some(float(c, "confidence")?),
    };
    Ok(Some((
        float(&m["left"], "left")?,
        float(&m["top"], "top")?,
        float(&m["right"], "right")?,
        float(&m["bottom"], "bottom")?,
        conf,
    )))
}

fn bbox_value(b: &Option<Bbox>) -> Value {
    let Some((l, t, r, bt, c)) = *b else {
        return Value::Null;
    };
    let mut m = Map::new();
    m.insert("left".into(), f(l));
    m.insert("top".into(), f(t));
    m.insert("right".into(), f(r));
    m.insert("bottom".into(), f(bt));
    m.insert("confidence".into(), c.map_or(Value::Null, f));
    Value::Object(m)
}

const FULL_KEYS: [&str; 6] = [
    "text",
    "label",
    "full_label",
    "statistics",
    "clip_scores",
    "toxicity_scores",
];

fn desc(v: &Value, vertex_label: &str) -> R<Desc> {
    let m = obj(v, "desc")?;
    let label = m.get("label").and_then(Value::as_str);
    let text = m.get("text").and_then(Value::as_str);
    if let (Some(label), Some(text)) = (label, text) {
        if m.len() == 2 {
            return Ok(Desc(DESC_SHORT, label.into(), text.into(), None));
        }
        let full = format!("{label}-{vertex_label}");
        if same_keys(m, &FULL_KEYS, "desc").is_ok()
            && m["full_label"].as_str() == Some(full.as_str())
            && m["statistics"].is_null()
            && m["toxicity_scores"].is_null()
            && m["clip_scores"] == clip_default()
        {
            return Ok(Desc(DESC_FULL, label.into(), text.into(), None));
        }
    }
    Ok(Desc(
        DESC_RAW,
        String::new(),
        String::new(),
        Some(v.clone()),
    ))
}

fn clip_default() -> Value {
    let mut m = Map::new();
    m.insert("scores".into(), Value::Null);
    m.insert("truncation".into(), Value::Bool(false));
    Value::Object(m)
}

fn desc_value(d: &Desc, vertex_label: &str) -> Value {
    let mut m = Map::new();
    match d.0 {
        DESC_SHORT => {
            m.insert("label".into(), Value::String(d.1.clone()));
            m.insert("text".into(), Value::String(d.2.clone()));
        }
        DESC_FULL => {
            m.insert("text".into(), Value::String(d.2.clone()));
            m.insert("label".into(), Value::String(d.1.clone()));
            m.insert(
                "full_label".into(),
                Value::String(format!("{}-{vertex_label}", d.1)),
            );
            m.insert("statistics".into(), Value::Null);
            m.insert("clip_scores".into(), clip_default());
            m.insert("toxicity_scores".into(), Value::Null);
        }
        _ => return d.3.clone().unwrap_or(Value::Null),
    }
    Value::Object(m)
}

fn edge_value(source: &str, text: &str, target: &str) -> Value {
    let mut m = Map::new();
    m.insert("source".into(), Value::String(source.into()));
    m.insert("text".into(), Value::String(text.into()));
    m.insert("target".into(), Value::String(target.into()));
    Value::Object(m)
}

fn edge(v: &Value) -> R<(String, String, String)> {
    let m = obj(v, "edge")?;
    same_keys(m, &["source", "text", "target"], "edge")?;
    Ok((
        string(&m["source"], "edge source")?,
        string(&m["text"], "edge text")?,
        string(&m["target"], "edge target")?,
    ))
}

/// The typed form of a GBC graph dict.
pub fn graph(v: &Value) -> R<Graph> {
    let g = obj(v, "graph")?;
    let has_tags = g.contains_key("tags");
    let mut keys: Vec<&str> = GRAPH_KEYS.to_vec();
    if has_tags {
        keys.push("tags");
    }
    same_keys(g, &keys, "graph")?;
    let vs = arr(&g["vertices"], "vertices")?;
    let mut index: HashMap<&str, u32> = HashMap::with_capacity(vs.len());
    for (i, v) in vs.iter().enumerate() {
        let id = obj(v, "vertex")?
            .get("vertex_id")
            .and_then(Value::as_str)
            .ok_or("a vertex without a string vertex_id")?;
        if index.insert(id, i as u32).is_some() {
            return Err(format!("vertex id {id:?} twice"));
        }
    }
    let at = |id: &str| {
        index
            .get(id)
            .copied()
            .ok_or_else(|| format!("no vertex {id:?}"))
    };

    // Out-edges first: in-edges name them by their graph-order index
    let mut strings = Strings::default();
    let mut vertices = Vec::with_capacity(vs.len());
    let mut edge_index: HashMap<(u32, u32, String), u32> = HashMap::new();
    let mut n_edges = 0u32;
    for v in vs {
        let m = obj(v, "vertex")?;
        let has_vtags = m.contains_key("tags");
        let mut want: Vec<&str> = VERTEX_KEYS.to_vec();
        if has_vtags {
            want.push("tags");
        }
        same_keys(m, &want, "vertex")?;
        let id = string(&m["vertex_id"], "vertex_id")?;
        let label = string(&m["label"], "vertex label")?;
        let me = at(&id)?;
        let mut out_edges = Vec::new();
        for e in arr(&m["out_edges"], "out_edges")? {
            let (source, text, target) = edge(e)?;
            if source != id {
                return Err(format!("an out-edge of {id:?} from {source:?}"));
            }
            let t = at(&target)?;
            edge_index.entry((me, t, text.clone())).or_insert(n_edges);
            n_edges += 1;
            out_edges.push((t, text));
        }
        let masks = |k: &str| -> R<Vec<u32>> {
            arr(&m[k], k)?.iter().map(|x| at(&string(x, k)?)).collect()
        };
        vertices.push(Vertex {
            bbox: bbox(&m["bbox"])?,
            descs: arr(&m["descs"], "descs")?
                .iter()
                .map(|d| desc(d, &label))
                .collect::<R<_>>()?,
            out_edges,
            in_edges: Vec::new(),
            sub_masks: masks("sub_masks")?,
            super_masks: masks("super_masks")?,
            tags: if has_vtags {
                Some(scores(&m["tags"], &mut strings, "vertex tags")?)
            } else {
                None
            },
            id,
            label,
        });
    }
    for (v, raw) in vertices.iter_mut().zip(vs) {
        for e in arr(&raw["in_edges"], "in_edges")? {
            let (source, text, target) = edge(e)?;
            let key = (at(&source)?, at(&target)?, text);
            let i = edge_index.get(&key).ok_or("an in-edge with no out-edge")?;
            v.in_edges.push(*i);
        }
    }
    let size = match &g["img_size"] {
        Value::Null => None,
        s => Some(
            arr(s, "img_size")?
                .iter()
                .map(|x| x.as_i64().ok_or("img_size holds a non-integer"))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };
    let tags = if has_tags {
        Some(graph_tags(&g["tags"], &mut strings)?)
    } else {
        None
    };
    Ok(Graph {
        strings: strings.list,
        vertices,
        img_url: opt_string(g.get("img_url"), "img_url")?,
        img_path: opt_string(g.get("img_path"), "img_path")?,
        original_caption: opt_string(g.get("original_caption"), "original_caption")?,
        short_caption: opt_string(g.get("short_caption"), "short_caption")?,
        detail_caption: opt_string(g.get("detail_caption"), "detail_caption")?,
        img_size: size,
        mask_inside_threshold: float(&g["mask_inside_threshold"], "mask_inside_threshold")?,
        tags,
    })
}

fn opt(s: &Option<String>) -> Value {
    s.as_ref().map_or(Value::Null, |x| Value::String(x.clone()))
}

/// The GBC graph dict of a typed graph.
pub fn graph_value(g: &Graph) -> Value {
    let ids: Vec<&str> = g.vertices.iter().map(|v| v.id.as_str()).collect();
    let mut flat: Vec<(usize, usize)> = Vec::new();
    for (vi, v) in g.vertices.iter().enumerate() {
        flat.extend((0..v.out_edges.len()).map(|k| (vi, k)));
    }
    let edge_of = |vi: usize, k: usize| {
        let (t, text) = &g.vertices[vi].out_edges[k];
        edge_value(ids[vi], text, ids[*t as usize])
    };
    let mut vertices = Vec::with_capacity(g.vertices.len());
    for (vi, v) in g.vertices.iter().enumerate() {
        let mut m = Map::new();
        m.insert("vertex_id".into(), Value::String(v.id.clone()));
        m.insert("bbox".into(), bbox_value(&v.bbox));
        m.insert("label".into(), Value::String(v.label.clone()));
        m.insert(
            "descs".into(),
            Value::Array(v.descs.iter().map(|d| desc_value(d, &v.label)).collect()),
        );
        let ins = v.in_edges.iter().map(|&i| {
            let (svi, k) = flat[i as usize];
            edge_of(svi, k)
        });
        m.insert("in_edges".into(), Value::Array(ins.collect()));
        let outs = (0..v.out_edges.len()).map(|k| edge_of(vi, k));
        m.insert("out_edges".into(), Value::Array(outs.collect()));
        let names = |xs: &[u32]| {
            Value::Array(
                xs.iter()
                    .map(|&i| Value::String(ids[i as usize].into()))
                    .collect(),
            )
        };
        m.insert("sub_masks".into(), names(&v.sub_masks));
        m.insert("super_masks".into(), names(&v.super_masks));
        if let Some(t) = &v.tags {
            m.insert("tags".into(), scores_value(t, &g.strings));
        }
        vertices.push(Value::Object(m));
    }
    let mut m = Map::new();
    m.insert("vertices".into(), Value::Array(vertices));
    m.insert("img_url".into(), opt(&g.img_url));
    m.insert("img_path".into(), opt(&g.img_path));
    m.insert("original_caption".into(), opt(&g.original_caption));
    m.insert("short_caption".into(), opt(&g.short_caption));
    m.insert("detail_caption".into(), opt(&g.detail_caption));
    let size = g.img_size.as_ref().map_or(Value::Null, |s| {
        Value::Array(s.iter().map(|&x| Value::Number(x.into())).collect())
    });
    m.insert("img_size".into(), size);
    m.insert("mask_inside_threshold".into(), f(g.mask_inside_threshold));
    if let Some(t) = &g.tags {
        let s = &g.strings;
        let named = |xs: &[(String, Scores)]| {
            let mut o = Map::new();
            for (k, sc) in xs {
                o.insert(k.clone(), scores_value(sc, s));
            }
            Value::Object(o)
        };
        let mut cats = Map::new();
        for &(k, c) in &t.categories {
            cats.insert(s[k as usize].clone(), Value::String(s[c as usize].clone()));
        }
        let mut tm = Map::new();
        tm.insert("image".into(), scores_value(&t.image, s));
        tm.insert("categories".into(), Value::Object(cats));
        tm.insert("regions".into(), named(&t.regions));
        tm.insert("crops".into(), named(&t.crops));
        m.insert("tags".into(), Value::Object(tm));
    }
    Value::Object(m)
}

/// The record of an answer line (an object with `graph`) or of a bare graph.
pub fn from_value(v: Value) -> Record {
    let mut rec = Record {
        format: FORMAT,
        head: Vec::new(),
        graph_at: None,
        graph: None,
        raw: None,
    };
    let body = match v {
        Value::Object(m) if m.contains_key("graph") && !m.contains_key("vertices") => {
            let mut body = Value::Null;
            for (i, (k, x)) in m.into_iter().enumerate() {
                if k == "graph" {
                    rec.graph_at = Some(i as u32);
                    body = x;
                } else {
                    rec.head.push((k, x));
                }
            }
            body
        }
        other => other,
    };
    if body.is_null() {
        return rec;
    }
    match graph(&body) {
        Ok(g) if graph_value(&g) == body => rec.graph = Some(g),
        _ => rec.raw = Some(body),
    }
    rec
}

/// The answer line (or bare graph) a record stores.
pub fn to_value(rec: &Record) -> Value {
    let body = match (&rec.graph, &rec.raw) {
        (Some(g), _) => graph_value(g),
        (None, Some(r)) => r.clone(),
        (None, None) => Value::Null,
    };
    let Some(at) = rec.graph_at else {
        return body;
    };
    let mut m = Map::new();
    let mut body = Some(body);
    for (i, (k, x)) in rec.head.iter().enumerate() {
        if i == at as usize {
            m.insert("graph".into(), body.take().unwrap());
        }
        m.insert(k.clone(), x.clone());
    }
    if let Some(b) = body {
        m.insert("graph".into(), b);
    }
    Value::Object(m)
}
