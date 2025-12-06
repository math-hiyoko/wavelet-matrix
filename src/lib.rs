mod dynamic_wavelet_matrix;
mod python;
mod traits;
mod wavelet_matrix;

use crate::python::{
    dynamic_wavelet_matrix::PyDynamicWaveletMatrix, wavelet_matrix::PyWaveletMatrix,
};
use pyo3::prelude::*;

#[pymodule(name = "wavelet_matrix")]
fn py_wavelet_matrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWaveletMatrix>()?;
    m.add_class::<PyDynamicWaveletMatrix>()?;
    Ok(())
}
