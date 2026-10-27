//! `kohakugbc_data._rs`: the native part of KohakuGBC's data package (graph resolution, the
//! graph store, prompts).

use pyo3::prelude::*;

pub mod graph;
pub mod prompt;
pub mod store;

/// The crate version, which is the package version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(graph::python::resolve, m)?)?;
    m.add_class::<store::python::GbcGraph>()?;
    m.add_class::<prompt::python::PromptDag>()?;
    Ok(())
}
