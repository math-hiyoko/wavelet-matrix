use std::u8;

use crate::{
    dynamic_wavelet_matrix::dynamic_wavelet_matrix::DynamicWaveletMatrix,
    traits::{
        dynamic_wavelet_matrix::DynamicWaveletMatrixTrait, wavelet_matrix::WaveletMatrixTrait,
    },
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyInt, PyList, PySequence},
};

enum DynamicWaveletMatrixEnum {
    U8(DynamicWaveletMatrix<u8>),
    U16(DynamicWaveletMatrix<u16>),
    U32(DynamicWaveletMatrix<u32>),
    U64(DynamicWaveletMatrix<u64>),
    U128(DynamicWaveletMatrix<u128>),
    BigUint(DynamicWaveletMatrix<BigUint>),
}

/// A Wavelet Matrix data structure for efficient rank, select, and quantile queries.
///
/// The Wavelet Matrix decomposes a sequence into multiple bit vectors,
/// one for each bit position. This allows for efficient queries on the sequence.
#[pyclass(unsendable, name = "DynamicWaveletMatrix")]
pub(crate) struct PyDynamicWaveletMatrix {
    inner: DynamicWaveletMatrixEnum,
}

#[pymethods]
impl PyDynamicWaveletMatrix {
    /// Creates a new Wavelet Matrix from the given list or tuple of integers.
    #[new]
    pub(crate) fn new(
        data: &Bound<'_, PyAny>,
        max_bit: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Self> {
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
        let max_bit = max_bit
            .map(|max_bit| max_bit.extract::<usize>().ok())
            .flatten();
        let bit_width =
            (values.iter().map(|v| v.bits()).max().unwrap_or(0) as usize).max(max_bit.unwrap_or(0));
        let wv: DynamicWaveletMatrixEnum = match bit_width {
            0..=8 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u8())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u8"))?;
                DynamicWaveletMatrixEnum::U8(DynamicWaveletMatrix::<u8>::new(&values, max_bit)?)
            }
            9..=16 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u16())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u16"))?;
                DynamicWaveletMatrixEnum::U16(DynamicWaveletMatrix::<u16>::new(&values, max_bit)?)
            }
            17..=32 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u32())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u32"))?;
                DynamicWaveletMatrixEnum::U32(DynamicWaveletMatrix::<u32>::new(&values, max_bit)?)
            }
            33..=64 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u64())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u64"))?;
                DynamicWaveletMatrixEnum::U64(DynamicWaveletMatrix::<u64>::new(&values, max_bit)?)
            }
            65..=128 => {
                let values = values
                    .iter()
                    .map(|v| v.to_u128())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyRuntimeError::new_err("Value out of range for u128"))?;
                DynamicWaveletMatrixEnum::U128(DynamicWaveletMatrix::<u128>::new(&values, max_bit)?)
            }
            _ => DynamicWaveletMatrixEnum::BigUint(DynamicWaveletMatrix::<BigUint>::new(
                &values, max_bit,
            )?),
        };
        Ok(PyDynamicWaveletMatrix { inner: wv })
    }

    /// Returns the length of the Wavelet Matrix.
    pub(crate) fn __len__(&self) -> PyResult<usize> {
        match &self.inner {
            DynamicWaveletMatrixEnum::U8(wm) => Ok(wm.len()),
            DynamicWaveletMatrixEnum::U16(wm) => Ok(wm.len()),
            DynamicWaveletMatrixEnum::U32(wm) => Ok(wm.len()),
            DynamicWaveletMatrixEnum::U64(wm) => Ok(wm.len()),
            DynamicWaveletMatrixEnum::U128(wm) => Ok(wm.len()),
            DynamicWaveletMatrixEnum::BigUint(wm) => Ok(wm.len()),
        }
    }

    /// Gets the value at the specified index.
    pub(crate) fn __getitem__(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyInt>> {
        match &self.inner {
            DynamicWaveletMatrixEnum::U8(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U16(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U32(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U64(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U128(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::BigUint(wm) => wm
                .access(index)
                .map(|value| value.into_pyobject(py).unwrap().unbind()),
        }
    }

    /// Access the value at the specified index.
    pub(crate) fn access(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyInt>> {
        match &self.inner {
            DynamicWaveletMatrixEnum::U8(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U16(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U32(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U64(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U128(wm) => {
                wm.access(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::BigUint(wm) => wm
                .access(index)
                .map(|value| value.into_pyobject(py).unwrap().unbind()),
        }
    }

    /// Counts the occurrences of the given value in the range [0, end).
    pub(crate) fn rank(&self, value: &Bound<'_, PyInt>, end: usize) -> PyResult<usize> {
        match &self.inner {
            DynamicWaveletMatrixEnum::U8(wm) => wm.rank(&value.extract::<u8>()?, end),
            DynamicWaveletMatrixEnum::U16(wm) => wm.rank(&value.extract::<u16>()?, end),
            DynamicWaveletMatrixEnum::U32(wm) => wm.rank(&value.extract::<u32>()?, end),
            DynamicWaveletMatrixEnum::U64(wm) => wm.rank(&value.extract::<u64>()?, end),
            DynamicWaveletMatrixEnum::U128(wm) => wm.rank(&value.extract::<u128>()?, end),
            DynamicWaveletMatrixEnum::BigUint(wm) => wm.rank(&value.extract::<BigUint>()?, end),
        }
    }

    /// Finds the position of the k-th occurrence of the given value.
    pub(crate) fn select(&self, value: &Bound<'_, PyInt>, kth: usize) -> PyResult<Option<usize>> {
        match &self.inner {
            DynamicWaveletMatrixEnum::U8(wm) => wm.select(&value.extract::<u8>()?, kth),
            DynamicWaveletMatrixEnum::U16(wm) => wm.select(&value.extract::<u16>()?, kth),
            DynamicWaveletMatrixEnum::U32(wm) => wm.select(&value.extract::<u32>()?, kth),
            DynamicWaveletMatrixEnum::U64(wm) => wm.select(&value.extract::<u64>()?, kth),
            DynamicWaveletMatrixEnum::U128(wm) => wm.select(&value.extract::<u128>()?, kth),
            DynamicWaveletMatrixEnum::BigUint(wm) => wm.select(&value.extract::<BigUint>()?, kth),
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
            DynamicWaveletMatrixEnum::U8(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            DynamicWaveletMatrixEnum::U16(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            DynamicWaveletMatrixEnum::U32(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            DynamicWaveletMatrixEnum::U64(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            DynamicWaveletMatrixEnum::U128(wm) => wm
                .quantile(start, end, kth)
                .map(|value| PyInt::new(py, value).into()),
            DynamicWaveletMatrixEnum::BigUint(wm) => wm
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
            DynamicWaveletMatrixEnum::U8(wm) => wm.topk(start, end, k),
            DynamicWaveletMatrixEnum::U16(wm) => wm.topk(start, end, k),
            DynamicWaveletMatrixEnum::U32(wm) => wm.topk(start, end, k),
            DynamicWaveletMatrixEnum::U64(wm) => wm.topk(start, end, k),
            DynamicWaveletMatrixEnum::U128(wm) => wm.topk(start, end, k),
            DynamicWaveletMatrixEnum::BigUint(wm) => wm.topk(start, end, k),
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
            DynamicWaveletMatrixEnum::U8(wm) => wm.range_sum(start, end),
            DynamicWaveletMatrixEnum::U16(wm) => wm.range_sum(start, end),
            DynamicWaveletMatrixEnum::U32(wm) => wm.range_sum(start, end),
            DynamicWaveletMatrixEnum::U64(wm) => wm.range_sum(start, end),
            DynamicWaveletMatrixEnum::U128(wm) => wm.range_sum(start, end),
            DynamicWaveletMatrixEnum::BigUint(wm) => wm.range_sum(start, end),
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
            DynamicWaveletMatrixEnum::U8(wm) => wm.range_intersection(start1, end1, start2, end2),
            DynamicWaveletMatrixEnum::U16(wm) => wm.range_intersection(start1, end1, start2, end2),
            DynamicWaveletMatrixEnum::U32(wm) => wm.range_intersection(start1, end1, start2, end2),
            DynamicWaveletMatrixEnum::U64(wm) => wm.range_intersection(start1, end1, start2, end2),
            DynamicWaveletMatrixEnum::U128(wm) => wm.range_intersection(start1, end1, start2, end2),
            DynamicWaveletMatrixEnum::BigUint(wm) => {
                wm.range_intersection(start1, end1, start2, end2)
            }
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
            DynamicWaveletMatrixEnum::U8(wm) => {
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
            DynamicWaveletMatrixEnum::U16(wm) => {
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
            DynamicWaveletMatrixEnum::U32(wm) => {
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
            DynamicWaveletMatrixEnum::U64(wm) => {
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
            DynamicWaveletMatrixEnum::U128(wm) => {
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
            DynamicWaveletMatrixEnum::BigUint(wm) => {
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
            DynamicWaveletMatrixEnum::U8(wm) => {
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
            DynamicWaveletMatrixEnum::U16(wm) => {
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
            DynamicWaveletMatrixEnum::U32(wm) => {
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
            DynamicWaveletMatrixEnum::U64(wm) => {
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
            DynamicWaveletMatrixEnum::U128(wm) => {
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
            DynamicWaveletMatrixEnum::BigUint(wm) => {
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
            DynamicWaveletMatrixEnum::U8(wm) => wm.range_maxk(start, end, k),
            DynamicWaveletMatrixEnum::U16(wm) => wm.range_maxk(start, end, k),
            DynamicWaveletMatrixEnum::U32(wm) => wm.range_maxk(start, end, k),
            DynamicWaveletMatrixEnum::U64(wm) => wm.range_maxk(start, end, k),
            DynamicWaveletMatrixEnum::U128(wm) => wm.range_maxk(start, end, k),
            DynamicWaveletMatrixEnum::BigUint(wm) => wm.range_maxk(start, end, k),
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
            DynamicWaveletMatrixEnum::U8(wm) => wm.range_mink(start, end, k),
            DynamicWaveletMatrixEnum::U16(wm) => wm.range_mink(start, end, k),
            DynamicWaveletMatrixEnum::U32(wm) => wm.range_mink(start, end, k),
            DynamicWaveletMatrixEnum::U64(wm) => wm.range_mink(start, end, k),
            DynamicWaveletMatrixEnum::U128(wm) => wm.range_mink(start, end, k),
            DynamicWaveletMatrixEnum::BigUint(wm) => wm.range_mink(start, end, k),
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
            DynamicWaveletMatrixEnum::U8(wm) => {
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
            DynamicWaveletMatrixEnum::U16(wm) => {
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
            DynamicWaveletMatrixEnum::U32(wm) => {
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
            DynamicWaveletMatrixEnum::U64(wm) => {
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
            DynamicWaveletMatrixEnum::U128(wm) => {
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
            DynamicWaveletMatrixEnum::BigUint(wm) => {
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
            DynamicWaveletMatrixEnum::U8(wm) => {
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
            DynamicWaveletMatrixEnum::U16(wm) => {
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
            DynamicWaveletMatrixEnum::U32(wm) => {
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
            DynamicWaveletMatrixEnum::U64(wm) => {
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
            DynamicWaveletMatrixEnum::U128(wm) => {
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
            DynamicWaveletMatrixEnum::BigUint(wm) => {
                let lower = lower.map(|value| value.extract::<BigUint>().ok()).flatten();
                let upper = upper.map(|value| value.extract::<BigUint>().ok()).flatten();
                wm.next_value(start, end, lower.as_ref(), upper.as_ref())
                    .map(|value| value.map(|value| value.into_pyobject(py).unwrap().unbind()))
            }
        }
    }

    fn insert(&mut self, index: usize, value: &Bound<'_, PyInt>) -> PyResult<()> {
        match &mut self.inner {
            DynamicWaveletMatrixEnum::U8(wm) => wm.insert(index, &value.extract::<u8>()?),
            DynamicWaveletMatrixEnum::U16(wm) => wm.insert(index, &value.extract::<u16>()?),
            DynamicWaveletMatrixEnum::U32(wm) => wm.insert(index, &value.extract::<u32>()?),
            DynamicWaveletMatrixEnum::U64(wm) => wm.insert(index, &value.extract::<u64>()?),
            DynamicWaveletMatrixEnum::U128(wm) => wm.insert(index, &value.extract::<u128>()?),
            DynamicWaveletMatrixEnum::BigUint(wm) => wm.insert(index, &value.extract::<BigUint>()?),
        }
    }

    fn remove(&mut self, py: Python<'_>, index: usize) -> PyResult<Py<PyInt>> {
        match &mut self.inner {
            DynamicWaveletMatrixEnum::U8(wm) => {
                wm.remove(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U16(wm) => {
                wm.remove(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U32(wm) => {
                wm.remove(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U64(wm) => {
                wm.remove(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::U128(wm) => {
                wm.remove(index).map(|value| PyInt::new(py, value).into())
            }
            DynamicWaveletMatrixEnum::BigUint(wm) => wm
                .remove(index)
                .map(|value| value.into_pyobject(py).unwrap().unbind()),
        }
    }

    fn update(
        &mut self,
        py: Python<'_>,
        index: usize,
        value: &Bound<'_, PyInt>,
    ) -> PyResult<Py<PyInt>> {
        match &mut self.inner {
            DynamicWaveletMatrixEnum::U8(wm) => wm
                .update(index, &value.extract::<u8>()?)
                .map(|old_value| PyInt::new(py, old_value).into()),
            DynamicWaveletMatrixEnum::U16(wm) => wm
                .update(index, &value.extract::<u16>()?)
                .map(|old_value| PyInt::new(py, old_value).into()),
            DynamicWaveletMatrixEnum::U32(wm) => wm
                .update(index, &value.extract::<u32>()?)
                .map(|old_value| PyInt::new(py, old_value).into()),
            DynamicWaveletMatrixEnum::U64(wm) => wm
                .update(index, &value.extract::<u64>()?)
                .map(|old_value| PyInt::new(py, old_value).into()),
            DynamicWaveletMatrixEnum::U128(wm) => wm
                .update(index, &value.extract::<u128>()?)
                .map(|old_value| PyInt::new(py, old_value).into()),
            DynamicWaveletMatrixEnum::BigUint(wm) => wm
                .update(index, &value.extract::<BigUint>()?)
                .map(|old_value| old_value.into_pyobject(py).unwrap().unbind()),
        }
    }
}
