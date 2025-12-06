use std::u8;

use crate::{
    traits::wavelet_matrix::WaveletMatrixTrait, wavelet_matrix::wavelet_matrix::WaveletMatrix,
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyInt, PyList, PySequence},
};

enum WaveletMatrixEnum {
    U8(WaveletMatrix<u8>),
    U16(WaveletMatrix<u16>),
    U32(WaveletMatrix<u32>),
    U64(WaveletMatrix<u64>),
    U128(WaveletMatrix<u128>),
    BigUint(WaveletMatrix<BigUint>),
}

/// A Wavelet Matrix data structure for efficient rank, select, and quantile queries.
///
/// The Wavelet Matrix decomposes a sequence into multiple bit vectors,
/// one for each bit position. This allows for efficient queries on the sequence.
#[pyclass(unsendable, name = "WaveletMatrix")]
pub(crate) struct PyWaveletMatrix {
    inner: WaveletMatrixEnum,
}

#[pymethods]
impl PyWaveletMatrix {
    /// Creates a new Wavelet Matrix from the given list or tuple of integers.
    #[new]
    pub(crate) fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let values: Vec<BigUint> = data
            .clone()
            .cast_into::<PySequence>()
            .map_err(|_| PyValueError::new_err("Input must be a list or tuple"))?
            .try_iter()?
            .map(|item| {
                item?.extract::<BigUint>().map_err(|_| {
                    PyValueError::new_err("Input elements must be non-negative integers")
                })
            })
            .collect::<PyResult<_>>()?;
        let bit_width = values.iter().map(|v| v.bits()).max().unwrap_or(0) as usize;
        let wv: WaveletMatrixEnum = match bit_width {
            0..=8 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u8())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u8"))?;
                WaveletMatrixEnum::U8(WaveletMatrix::<u8>::new(&values))
            }
            9..=16 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u16())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u16"))?;
                WaveletMatrixEnum::U16(WaveletMatrix::<u16>::new(&values))
            }
            17..=32 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u32())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u32"))?;
                WaveletMatrixEnum::U32(WaveletMatrix::<u32>::new(&values))
            }
            33..=64 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u64())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u64"))?;
                WaveletMatrixEnum::U64(WaveletMatrix::<u64>::new(&values))
            }
            65..=128 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u128())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u128"))?;
                WaveletMatrixEnum::U128(WaveletMatrix::<u128>::new(&values))
            }
            _ => WaveletMatrixEnum::BigUint(WaveletMatrix::<BigUint>::new(&values)),
        };
        Ok(PyWaveletMatrix { inner: wv })
    }

    /// Returns the length of the Wavelet Matrix.
    pub(crate) fn __len__(&self) -> PyResult<usize> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => Ok(wm.len()),
            WaveletMatrixEnum::U16(wm) => Ok(wm.len()),
            WaveletMatrixEnum::U32(wm) => Ok(wm.len()),
            WaveletMatrixEnum::U64(wm) => Ok(wm.len()),
            WaveletMatrixEnum::U128(wm) => Ok(wm.len()),
            WaveletMatrixEnum::BigUint(wm) => Ok(wm.len()),
        }
    }

    /// Gets the value at the specified index.
    pub(crate) fn __getitem__(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyInt>> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.access(index).map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U16(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            WaveletMatrixEnum::U32(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            WaveletMatrixEnum::U64(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            WaveletMatrixEnum::U128(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            WaveletMatrixEnum::BigUint(wm) => wm
                .access(index)
                .map(|value| value.into_pyobject(py).unwrap().unbind()),
        }
    }

    /// Access the value at the specified index.
    pub(crate) fn access(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyInt>> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.access(index).map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U16(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            WaveletMatrixEnum::U32(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            WaveletMatrixEnum::U64(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            WaveletMatrixEnum::U128(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            WaveletMatrixEnum::BigUint(wm) => wm
                .access(index)
                .map(|value| value.into_pyobject(py).unwrap().unbind()),
        }
    }

    /// Counts the occurrences of the given value in the range [0, end).
    pub(crate) fn rank(&self, value: &Bound<'_, PyInt>, end: usize) -> PyResult<usize> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.rank(&value.extract::<u8>()?, end),
            WaveletMatrixEnum::U16(wm) => wm.rank(&value.extract::<u16>()?, end),
            WaveletMatrixEnum::U32(wm) => wm.rank(&value.extract::<u32>()?, end),
            WaveletMatrixEnum::U64(wm) => wm.rank(&value.extract::<u64>()?, end),
            WaveletMatrixEnum::U128(wm) => wm.rank(&value.extract::<u128>()?, end),
            WaveletMatrixEnum::BigUint(wm) => wm.rank(&value.extract::<BigUint>()?, end),
        }
    }

    /// Finds the position of the k-th occurrence of the given value.
    pub(crate) fn select(&self, value: &Bound<'_, PyInt>, kth: usize) -> PyResult<Option<usize>> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.select(&value.extract::<u8>()?, kth),
            WaveletMatrixEnum::U16(wm) => wm.select(&value.extract::<u16>()?, kth),
            WaveletMatrixEnum::U32(wm) => wm.select(&value.extract::<u32>()?, kth),
            WaveletMatrixEnum::U64(wm) => wm.select(&value.extract::<u64>()?, kth),
            WaveletMatrixEnum::U128(wm) => wm.select(&value.extract::<u128>()?, kth),
            WaveletMatrixEnum::BigUint(wm) => wm.select(&value.extract::<BigUint>()?, kth),
        }
    }

    /// Finds the k-th smallest value in the range [start, end).
    pub(crate) fn quantile(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        kth: usize,
    ) -> PyResult<Py<PyInt>> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U16(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U32(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U64(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U128(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::BigUint(wm) => wm
                .quantile(start, end, kth)
                .map(|value| value.into_pyobject(py).unwrap().unbind()),
        }
    }

    /// Finds the top-k most frequent elements in the range [start, end).
    #[pyo3(signature = (start, end, k=None))]
    pub(crate) fn topk(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        k: Option<usize>,
    ) -> PyResult<Py<PyList>> {
        let result = match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.topk(start, end, k),
            WaveletMatrixEnum::U16(wm) => wm.topk(start, end, k),
            WaveletMatrixEnum::U32(wm) => wm.topk(start, end, k),
            WaveletMatrixEnum::U64(wm) => wm.topk(start, end, k),
            WaveletMatrixEnum::U128(wm) => wm.topk(start, end, k),
            WaveletMatrixEnum::BigUint(wm) => wm.topk(start, end, k),
        };
        result.and_then(|value| {
            let pyobject = value.into_pyobject(py)?;
            let pylist = pyobject.cast_into::<PyList>()?;
            Ok(pylist.unbind())
        })
    }

    /// Computes the sum of values in the range [start, end).
    pub(crate) fn range_sum(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
    ) -> PyResult<Py<PyInt>> {
        let result = match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::U16(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::U32(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::U64(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::U128(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::BigUint(wm) => wm.range_sum(start, end),
        };
        result.and_then(|value| {
            let pyobject = value.into_pyobject(py)?;
            Ok(pyobject.unbind())
        })
    }

    /// Finds the intersection of values in the two ranges [start1, end1) and [start2, end2).
    pub(crate) fn range_intersection(
        &self,
        py: Python<'_>,
        start1: usize,
        end1: usize,
        start2: usize,
        end2: usize,
    ) -> PyResult<Py<PyList>> {
        let result = match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.range_intersection(start1, end1, start2, end2),
            WaveletMatrixEnum::U16(wm) => wm.range_intersection(start1, end1, start2, end2),
            WaveletMatrixEnum::U32(wm) => wm.range_intersection(start1, end1, start2, end2),
            WaveletMatrixEnum::U64(wm) => wm.range_intersection(start1, end1, start2, end2),
            WaveletMatrixEnum::U128(wm) => wm.range_intersection(start1, end1, start2, end2),
            WaveletMatrixEnum::BigUint(wm) => wm.range_intersection(start1, end1, start2, end2),
        };
        result.and_then(|value| {
            let pyobject = value.into_pyobject(py)?;
            let pylist = pyobject.cast_into::<PyList>()?;
            Ok(pylist.unbind())
        })
    }

    /// Counts the number of elements within the optional range [lower, upper) in the range [start, end).
    #[pyo3(signature = (start, end, lower=None, upper=None))]
    pub fn range_freq(
        &self,
        _py: Python<'_>,
        start: usize,
        end: usize,
        lower: Option<Bound<'_, PyInt>>,
        upper: Option<Bound<'_, PyInt>>,
    ) -> PyResult<usize> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => {
                let lower = lower.map(|value| value.extract::<u8>().ok());
                let upper = upper.map(|value| value.extract::<u8>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(0)
                } else {
                    wm.range_freq(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::U16(wm) => {
                let lower = lower.map(|value| value.extract::<u16>().ok());
                let upper = upper.map(|value| value.extract::<u16>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(0)
                } else {
                    wm.range_freq(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::U32(wm) => {
                let lower = lower.map(|value| value.extract::<u32>().ok());
                let upper = upper.map(|value| value.extract::<u32>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(0)
                } else {
                    wm.range_freq(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::U64(wm) => {
                let lower = lower.map(|value| value.extract::<u64>().ok());
                let upper = upper.map(|value| value.extract::<u64>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(0)
                } else {
                    wm.range_freq(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::U128(wm) => {
                let lower = lower.map(|value| value.extract::<u128>().ok());
                let upper = upper.map(|value| value.extract::<u128>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(0)
                } else {
                    wm.range_freq(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::BigUint(wm) => {
                let lower = lower.map(|value| value.extract::<BigUint>().ok());
                let upper = upper.map(|value| value.extract::<BigUint>().ok());
                wm.range_freq(
                    start,
                    end,
                    lower.flatten().as_ref(),
                    upper.flatten().as_ref(),
                )
            }
        }
    }

    /// Lists all elements within the optional range [lower, upper) in the range [start, end).
    #[pyo3(signature = (start, end, lower=None, upper=None))]
    pub fn range_list(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        lower: Option<Bound<'_, PyInt>>,
        upper: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Py<PyList>> {
        let result = match &self.inner {
            WaveletMatrixEnum::U8(wm) => {
                let lower = lower.map(|value| value.extract::<u8>().ok());
                let upper = upper.map(|value| value.extract::<u8>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(vec![])
                } else {
                    wm.range_list(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::U16(wm) => {
                let lower = lower.map(|value| value.extract::<u16>().ok());
                let upper = upper.map(|value| value.extract::<u16>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(vec![])
                } else {
                    wm.range_list(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::U32(wm) => {
                let lower = lower.map(|value| value.extract::<u32>().ok());
                let upper = upper.map(|value| value.extract::<u32>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(vec![])
                } else {
                    wm.range_list(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::U64(wm) => {
                let lower = lower.map(|value| value.extract::<u64>().ok());
                let upper = upper.map(|value| value.extract::<u64>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(vec![])
                } else {
                    wm.range_list(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::U128(wm) => {
                let lower = lower.map(|value| value.extract::<u128>().ok());
                let upper = upper.map(|value| value.extract::<u128>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(vec![])
                } else {
                    wm.range_list(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                }
            }
            WaveletMatrixEnum::BigUint(wm) => {
                let lower = lower.map(|value| value.extract::<BigUint>().ok()).flatten();
                let upper = upper.map(|value| value.extract::<BigUint>().ok()).flatten();
                wm.range_list(start, end, lower.as_ref(), upper.as_ref())
            }
        };
        result.and_then(|value| {
            let pyobject = value.into_pyobject(py)?;
            let pylist = pyobject.cast_into::<PyList>()?;
            Ok(pylist.unbind())
        })
    }

    /// Finds the k largest values in the range [start, end).
    #[pyo3(signature = (start, end, k=None))]
    fn range_maxk(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        k: Option<usize>,
    ) -> PyResult<Py<PyList>> {
        let result = match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.range_maxk(start, end, k),
            WaveletMatrixEnum::U16(wm) => wm.range_maxk(start, end, k),
            WaveletMatrixEnum::U32(wm) => wm.range_maxk(start, end, k),
            WaveletMatrixEnum::U64(wm) => wm.range_maxk(start, end, k),
            WaveletMatrixEnum::U128(wm) => wm.range_maxk(start, end, k),
            WaveletMatrixEnum::BigUint(wm) => wm.range_maxk(start, end, k),
        };
        result.and_then(|value| {
            let pyobject = value.into_pyobject(py)?;
            let pylist = pyobject.cast_into::<PyList>()?;
            Ok(pylist.unbind())
        })
    }

    /// Finds the k smallest values in the range [start, end).
    #[pyo3(signature = (start, end, k=None))]
    pub fn range_mink(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        k: Option<usize>,
    ) -> PyResult<Py<PyList>> {
        let result = match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.range_mink(start, end, k),
            WaveletMatrixEnum::U16(wm) => wm.range_mink(start, end, k),
            WaveletMatrixEnum::U32(wm) => wm.range_mink(start, end, k),
            WaveletMatrixEnum::U64(wm) => wm.range_mink(start, end, k),
            WaveletMatrixEnum::U128(wm) => wm.range_mink(start, end, k),
            WaveletMatrixEnum::BigUint(wm) => wm.range_mink(start, end, k),
        };
        result.and_then(|value| {
            let pyobject = value.into_pyobject(py)?;
            let pylist = pyobject.cast_into::<PyList>()?;
            Ok(pylist.unbind())
        })
    }

    /// Finds the previous value before upper bound in the range [start, end).
    #[pyo3(signature = (start, end, lower=None, upper=None))]
    pub fn prev_value(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        lower: Option<Bound<'_, PyInt>>,
        upper: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Option<Py<PyInt>>> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => {
                let lower = lower.map(|value| value.extract::<u8>().ok());
                let upper = upper.map(|value| value.extract::<u8>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.prev_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::U16(wm) => {
                let lower = lower.map(|value| value.extract::<u16>().ok());
                let upper = upper.map(|value| value.extract::<u16>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.prev_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::U32(wm) => {
                let lower = lower.map(|value| value.extract::<u32>().ok());
                let upper = upper.map(|value| value.extract::<u32>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.prev_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::U64(wm) => {
                let lower = lower.map(|value| value.extract::<u64>().ok());
                let upper = upper.map(|value| value.extract::<u64>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.prev_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::U128(wm) => {
                let lower = lower.map(|value| value.extract::<u128>().ok());
                let upper = upper.map(|value| value.extract::<u128>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.prev_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::BigUint(wm) => {
                let lower = lower.map(|value| value.extract::<BigUint>().ok()).flatten();
                let upper = upper.map(|value| value.extract::<BigUint>().ok()).flatten();
                wm.prev_value(start, end, lower.as_ref(), upper.as_ref())
                    .map(|value| value.map(|value| value.into_pyobject(py).unwrap().unbind()))
            }
        }
    }

    /// Finds the next value after lower bound in the range [start, end).
    #[pyo3(signature = (start, end, lower=None, upper=None))]
    pub fn next_value(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        lower: Option<Bound<'_, PyInt>>,
        upper: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Option<Py<PyInt>>> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => {
                let lower = lower.map(|value| value.extract::<u8>().ok());
                let upper = upper.map(|value| value.extract::<u8>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.next_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::U16(wm) => {
                let lower = lower.map(|value| value.extract::<u16>().ok());
                let upper = upper.map(|value| value.extract::<u16>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.next_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::U32(wm) => {
                let lower = lower.map(|value| value.extract::<u32>().ok());
                let upper = upper.map(|value| value.extract::<u32>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.next_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::U64(wm) => {
                let lower = lower.map(|value| value.extract::<u64>().ok());
                let upper = upper.map(|value| value.extract::<u64>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.next_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::U128(wm) => {
                let lower = lower.map(|value| value.extract::<u128>().ok());
                let upper = upper.map(|value| value.extract::<u128>().ok());
                if lower.is_some_and(|lower| lower.is_none()) {
                    Ok(None)
                } else {
                    wm.next_value(
                        start,
                        end,
                        lower.flatten().as_ref(),
                        upper.flatten().as_ref(),
                    )
                    .map(|value| value.map(|value| PyInt::new(py, value).into()))
                }
            }
            WaveletMatrixEnum::BigUint(wm) => {
                let lower = lower.map(|value| value.extract::<BigUint>().ok()).flatten();
                let upper = upper.map(|value| value.extract::<BigUint>().ok()).flatten();
                wm.next_value(start, end, lower.as_ref(), upper.as_ref())
                    .map(|value| value.map(|value| value.into_pyobject(py).unwrap().unbind()))
            }
        }
    }
}
