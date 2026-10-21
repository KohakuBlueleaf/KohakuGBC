//! `kohakugbc._rs`: the native part of KohakuGBC.

use pyo3::prelude::*;

pub mod graph;

/// The crate version, which is the package version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(graph::python::resolve, m)?)?;
    Ok(())
}
