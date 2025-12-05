mod python;
mod wavelet_matrix;
mod traits;
mod dynamic_wavelet_matrix;

use pyo3::prelude::*;
use crate::python::{
    wavelet_matrix::PyWaveletMatrix,
    dynamic_wavelet_matrix::PyDynamicWaveletMatrix,
};


#[pymodule(name = "wavelet_matrix")]
fn py_wavelet_matrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWaveletMatrix>()?;
    m.add_class::<PyDynamicWaveletMatrix>()?;
    Ok(())
}
