//! `kohakugbc_data._rs.PromptDag`: the prompt DAG held in Rust, its text in and out.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::coords::Coords;
use super::dag::Dag;
use super::text::{parse, render};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

pub fn coords(
    width: i64,
    height: i64,
    space: &str,
    order: &str,
    scale: i64,
    rounding: &str,
) -> PyResult<Coords> {
    Coords::new(width, height, space, order, scale, rounding).map_err(err)
}

#[pyclass(module = "kohakugbc_data._rs", name = "PromptDag", frozen)]
pub struct PromptDag {
    pub dag: Dag,
}

#[pymethods]
impl PromptDag {
    /// The DAG a prompt text says; boxes read in the header's coords, else in these.
    #[staticmethod]
    #[pyo3(signature = (text, width=1024, height=1024, space="abs_centralized", order="yxyx", scale=1000))]
    fn parse(
        py: Python<'_>,
        text: String,
        width: i64,
        height: i64,
        space: &str,
        order: &str,
        scale: i64,
    ) -> PyResult<Self> {
        let c = coords(width, height, space, order, scale, "enclose")?;
        let dag = py.detach(move || parse(&text, &c)).map_err(err)?;
        Ok(PromptDag { dag })
    }

    #[pyo3(signature = (width=1024, height=1024, space="abs_centralized", order="yxyx", scale=1000, rounding="enclose", placement="first", header=true))]
    #[allow(clippy::too_many_arguments)]
    fn render(
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
    ) -> PyResult<String> {
        let c = coords(width, height, space, order, scale, rounding)?;
        py.detach(|| render(&self.dag, &c, placement, header))
            .map_err(err)
    }

    #[getter]
    fn root(&self) -> &str {
        &self.dag.root
    }

    fn __len__(&self) -> usize {
        self.dag.vertices.len()
    }

    /// `{"root", "vertices": {handle: {"text", "box", "tags", "children", "relations"}}}`,
    /// a box a `(left, top, right, bottom)` tuple, a relation `(participants, text)`.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let vs = PyDict::new(py);
        for (h, v) in &self.dag.vertices {
            let d = PyDict::new(py);
            d.set_item("text", &v.text)?;
            d.set_item("box", v.bbox.map(|b| PyTuple::new(py, b)).transpose()?)?;
            d.set_item("tags", &v.tags)?;
            d.set_item("children", PyList::new(py, &v.children)?)?;
            let rels = v
                .relations
                .iter()
                .map(|r| (r.participants.clone(), r.text.clone()))
                .collect::<Vec<_>>();
            d.set_item("relations", rels)?;
            vs.set_item(h, d)?;
        }
        let out = PyDict::new(py);
        out.set_item("root", &self.dag.root)?;
        out.set_item("vertices", vs)?;
        Ok(out)
    }
}
