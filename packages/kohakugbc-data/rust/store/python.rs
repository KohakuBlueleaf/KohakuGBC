//! `kohakugbc_data._rs.GbcGraph`: one answer line held in Rust; JSON or msgpack in, either out, and
//! its prompt.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::Value;

use super::convert::{from_value, to_value};
use super::pack::{pack, unpack, DEFAULT_ZSTD};
use super::record::Record;
use crate::prompt::gbc::{from_gbc, Options};
use crate::prompt::python::{coords, PromptDag};
use crate::prompt::text::render;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pyclass(module = "kohakugbc_data._rs", name = "GbcGraph", frozen)]
pub struct GbcGraph {
    rec: Record,
}

impl GbcGraph {
    fn options(
        captions: Option<Vec<String>>,
        boxes: bool,
        tag_threshold: Option<f64>,
        tag_spaces: bool,
        root: &str,
    ) -> Options {
        let mut o = Options {
            boxes,
            tag_threshold,
            tag_spaces,
            root: root.into(),
            ..Default::default()
        };
        if let Some(c) = captions {
            o.captions = c;
        }
        o
    }

    fn graph(&self) -> PyResult<&super::record::Graph> {
        self.rec
            .graph
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the graph is null or kept raw, not typed"))
    }
}

#[pymethods]
impl GbcGraph {
    /// An answer line (`{"key", "ok", "graph", ...}`) or a bare graph dict, as JSON text.
    #[staticmethod]
    fn from_json(py: Python<'_>, text: &str) -> PyResult<Self> {
        let text = text.to_owned();
        let rec = py.detach(move || {
            serde_json::from_str::<Value>(&text)
                .map(from_value)
                .map_err(|e| e.to_string())
        });
        Ok(GbcGraph {
            rec: rec.map_err(err)?,
        })
    }

    /// Bytes `to_msgpack` wrote (plain or zstd).
    #[staticmethod]
    fn from_msgpack(py: Python<'_>, data: Vec<u8>) -> PyResult<Self> {
        let rec = py.detach(move || unpack(&data)).map_err(err)?;
        Ok(GbcGraph { rec })
    }

    /// msgpack bytes in a zstd frame of level `zstd` (None: plain msgpack).
    #[pyo3(signature = (zstd=Some(DEFAULT_ZSTD)))]
    fn to_msgpack<'py>(&self, py: Python<'py>, zstd: Option<i32>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| pack(&self.rec, zstd)).map_err(err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// The answer line (or bare graph) as compact JSON text, equal to the one read in.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| serde_json::to_string(&to_value(&self.rec)))
            .map_err(|e| err(e.to_string()))
    }

    #[getter]
    fn key(&self) -> Option<String> {
        self.rec
            .head
            .iter()
            .find(|(k, _)| k == "key")
            .and_then(|(_, v)| v.as_str().map(String::from))
    }

    /// The graph is held in typed form (else raw or null).
    #[getter]
    fn typed(&self) -> bool {
        self.rec.graph.is_some()
    }

    #[getter]
    fn num_vertices(&self) -> usize {
        self.rec.graph.as_ref().map_or(0, |g| g.vertices.len())
    }

    /// The graph's prompt DAG (`kohakugbc_data.prompt.from_gbc`).
    #[pyo3(signature = (captions=None, boxes=true, tag_threshold=Some(0.35), tag_spaces=true, root="image"))]
    fn prompt_dag(
        &self,
        py: Python<'_>,
        captions: Option<Vec<String>>,
        boxes: bool,
        tag_threshold: Option<f64>,
        tag_spaces: bool,
        root: &str,
    ) -> PyResult<PromptDag> {
        let o = Self::options(captions, boxes, tag_threshold, tag_spaces, root);
        let g = self.graph()?;
        let dag = py.detach(|| from_gbc(g, &o)).map_err(err)?;
        Ok(PromptDag { dag })
    }

    /// The graph's prompt text in one call (`prompt_dag(...).render(...)`).
    #[pyo3(signature = (width=1024, height=1024, space="abs_centralized", order="yxyx", scale=1000, rounding="enclose", placement="first", header=true, captions=None, boxes=true, tag_threshold=Some(0.35), tag_spaces=true, root="image"))]
    #[allow(clippy::too_many_arguments)]
    fn prompt(
        &self,
        py: Python<'_>,
        width: i64,
        height: i64,
        space: &str,
        order: &str,
        scale: i64,
        rounding: &str,
        placement: &str,
        header: bool,
        captions: Option<Vec<String>>,
        boxes: bool,
        tag_threshold: Option<f64>,
        tag_spaces: bool,
        root: &str,
    ) -> PyResult<String> {
        let c = coords(width, height, space, order, scale, rounding)?;
        let o = Self::options(captions, boxes, tag_threshold, tag_spaces, root);
        let g = self.graph()?;
        py.detach(|| from_gbc(g, &o).and_then(|d| render(&d, &c, placement, header)))
            .map_err(err)
    }
}
