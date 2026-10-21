//! `kohakugbc._rs.resolve`: records from Python, the graph as JSON text and its stats back.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use super::model::{Bbox, Desc, EntityRef, Record};
use super::orders::stats;
use super::resolve::resolve as resolve_core;

/// `[entity_id, label, text]`; `numbered` uses Python's own `str.isdigit` on the id's suffix.
fn entity_ref(py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<EntityRef> {
    let id: String = item.get_item(0)?.extract()?;
    let label: String = item.get_item(1)?.extract()?;
    let text: Option<String> = item.get_item(2)?.extract()?;
    let numbered = match id.rsplit_once('_') {
        Some((_, suffix)) => PyString::new(py, suffix)
            .call_method0("isdigit")?
            .extract::<bool>()?,
        None => false,
    };
    Ok(EntityRef {
        id,
        label,
        text,
        numbered,
    })
}

fn refs(py: Python<'_>, items: &Bound<'_, PyAny>) -> PyResult<Vec<EntityRef>> {
    items
        .try_iter()?
        .map(|item| entity_ref(py, &item?))
        .collect()
}

fn bbox(item: &Bound<'_, PyAny>) -> PyResult<Option<Bbox>> {
    if item.is_none() {
        return Ok(None);
    }
    Ok(Some(Bbox {
        left: item.get_item(0)?.extract()?,
        top: item.get_item(1)?.extract()?,
        right: item.get_item(2)?.extract()?,
        bottom: item.get_item(3)?.extract()?,
        confidence: item.get_item(4)?.extract()?,
    }))
}

fn record(py: Python<'_>, rec: &Bound<'_, PyAny>) -> PyResult<Record> {
    let mut descs = Vec::new();
    for d in rec.get_item("descs")?.try_iter()? {
        let d = d?;
        descs.push(Desc {
            text: d.get_item(0)?.extract()?,
            label: d.get_item(1)?.extract()?,
            ids: d.get_item(2)?.extract()?,
        });
    }
    Ok(Record {
        ids: refs(py, &rec.get_item("ids")?)?,
        bbox: bbox(&rec.get_item("box")?)?,
        descs,
        children: refs(py, &rec.get_item("children")?)?,
    })
}

/// The GBC graph of one image as JSON text (None without described records), and its stats.
#[pyfunction]
#[pyo3(signature = (records, img_path, img_size=None, threshold=0.85))]
pub fn resolve<'py>(
    py: Python<'py>,
    records: &Bound<'py, PyAny>,
    img_path: Option<String>,
    img_size: Option<Vec<i64>>,
    threshold: f64,
) -> PyResult<(Option<String>, Bound<'py, PyDict>)> {
    let recs = records
        .try_iter()?
        .map(|r| record(py, &r?))
        .collect::<PyResult<Vec<_>>>()?;
    let size = match img_size.as_deref() {
        None => None,
        Some(&[w, h]) => Some([w, h]),
        Some(_) => return Err(PyValueError::new_err("img_size is [width, height]")),
    };

    // Resolution and serialisation run without the GIL
    let out = py.detach(move || -> Result<Option<(String, _)>, String> {
        let Some(graph) = resolve_core(&recs, img_path, size, threshold)? else {
            return Ok(None);
        };
        let text = serde_json::to_string(&graph).map_err(|e| e.to_string())?;
        Ok(Some((text, stats(&graph))))
    });
    let out = out.map_err(PyValueError::new_err)?;

    let d = PyDict::new(py);
    let Some((text, s)) = out else {
        d.set_item("vertices", 0)?;
        d.set_item("labels", PyDict::new(py))?;
        d.set_item("edges", 0)?;
        d.set_item("captions", 0)?;
        d.set_item("depth", -1)?;
        return Ok((None, d));
    };
    let labels = PyDict::new(py);
    for (k, v) in &s.labels {
        labels.set_item(k, v)?;
    }
    d.set_item("vertices", s.vertices)?;
    d.set_item("labels", labels)?;
    d.set_item("edges", s.edges)?;
    d.set_item("captions", s.captions)?;
    d.set_item("depth", s.depth)?;
    d.set_item("roots", s.roots)?;
    d.set_item("leaves", s.leaves)?;
    Ok((Some(text), d))
}
