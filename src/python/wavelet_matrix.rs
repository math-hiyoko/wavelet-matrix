use num_bigint::BigUint;
use num_traits::ToPrimitive;
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyInt, PyList, PySequence, PySlice, PySliceIndices},
};

use crate::{
    traits::wavelet_matrix::wavelet_matrix::WaveletMatrixTrait,
    wavelet_matrix::wavelet_matrix::WaveletMatrix,
};

#[derive(Clone)]
enum WaveletMatrixEnum {
    U8(WaveletMatrix<u8>),
    U16(WaveletMatrix<u16>),
    U32(WaveletMatrix<u32>),
    U64(WaveletMatrix<u64>),
    U128(WaveletMatrix<u128>),
    BigUint(WaveletMatrix<BigUint>),
}
/// A Wavelet Matrix for fast queries on a static sequence of integers.
///
/// Wavelet Matrix decomposes values into bit layers and supports queries such as:
/// - access(i): read value at position i
/// - rank(v, end): count v in a prefix
/// - select(v, kth): find position of k-th v
/// - quantile(l, r, kth): k-th smallest in a range
/// - rich range queries: topk, range_sum, range_freq, range_list, ...
///
/// This class automatically chooses an internal representation based on input values.
///
/// ### Construction
/// #### Time / Space Complexity
///
/// - Time: `O(N log V)`
/// - Space: `O(N log V)`
///
/// where:
/// - `N` = length of the sequence
/// - `V` = value domain / range (roughly related to max bit-width)
///
/// ```python
/// from wavelet_matrix import WaveletMatrix
/// wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
/// ```
#[derive(Clone)]
#[pyclass(name = "WaveletMatrix")]
pub(crate) struct PyWaveletMatrix {
    inner: WaveletMatrixEnum,
}

#[pymethods]
impl PyWaveletMatrix {
    /// Creates a new Wavelet Matrix from the given list or tuple of integers.
    #[new]
    fn new(py: Python<'_>, data: &Bound<'_, PySequence>) -> PyResult<Self> {
        let values: Vec<BigUint> = data
            .try_iter()?
            .map(|item| {
                item?.extract::<BigUint>().map_err(|_| {
                    PyValueError::new_err("Input elements must be non-negative integers")
                })
            })
            .collect::<PyResult<_>>()?;

        py.detach(move || {
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
        })
    }

    /// Returns the length of the Wavelet Matrix.
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(move || match &self.inner {
            WaveletMatrixEnum::U8(wm) => Ok(wm.len()),
            WaveletMatrixEnum::U16(wm) => Ok(wm.len()),
            WaveletMatrixEnum::U32(wm) => Ok(wm.len()),
            WaveletMatrixEnum::U64(wm) => Ok(wm.len()),
            WaveletMatrixEnum::U128(wm) => Ok(wm.len()),
            WaveletMatrixEnum::BigUint(wm) => Ok(wm.len()),
        })
    }

    /// Gets the value at the specified index.
    fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        macro_rules! getitem_impl {
            ($wm:expr) => {
                if let Ok(index) = index.extract::<usize>() {
                    let value = py.detach(move || $wm.access(index))?;
                    return Ok(value.into_pyobject(py)?.unbind().into());
                } else if let Ok(slice) = index.clone().cast_into::<PySlice>() {
                    let PySliceIndices {
                        start,
                        step,
                        slicelength,
                        ..
                    } = slice.indices($wm.len() as isize)?;
                    let values = py.detach(move || -> PyResult<Vec<_>> {
                        let mut index = start;
                        let mut values = Vec::with_capacity(slicelength as usize);
                        for _ in 0..slicelength {
                            index = (index + $wm.len() as isize) % ($wm.len() as isize);
                            values.push($wm.access(index as usize)?);
                            index += step;
                        }
                        Ok(values)
                    })?;
                    return Ok(PyList::new(py, &values)?.unbind().into());
                } else {
                    return Err(PyTypeError::new_err(
                        "index must be a non-negative integer or a slice",
                    ));
                }
            };
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => getitem_impl!(wm),
            WaveletMatrixEnum::U16(wm) => getitem_impl!(wm),
            WaveletMatrixEnum::U32(wm) => getitem_impl!(wm),
            WaveletMatrixEnum::U64(wm) => getitem_impl!(wm),
            WaveletMatrixEnum::U128(wm) => getitem_impl!(wm),
            WaveletMatrixEnum::BigUint(wm) => getitem_impl!(wm),
        }
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(move || match &self.inner {
            WaveletMatrixEnum::U8(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U16(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U32(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U64(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U128(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::BigUint(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(move || match &self.inner {
            WaveletMatrixEnum::U8(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U16(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U32(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U64(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U128(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::BigUint(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
        })
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(move || Ok(self.clone()))
    }

    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        py.detach(move || Ok(self.clone()))
    }

    /// Return the entire sequence as a Python list.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(N log V)`  
    /// - Space: `O(N)`
    ///
    /// #### Examples
    /// ```python
    /// wm.values()
    /// # [5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
    /// ```
    fn values(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        match &self.inner {
            WaveletMatrixEnum::U8(wm) => {
                Ok(PyList::new(py, &py.detach(move || wm.values())?)?.unbind())
            }
            WaveletMatrixEnum::U16(wm) => {
                Ok(PyList::new(py, &py.detach(move || wm.values())?)?.unbind())
            }
            WaveletMatrixEnum::U32(wm) => {
                Ok(PyList::new(py, &py.detach(move || wm.values())?)?.unbind())
            }
            WaveletMatrixEnum::U64(wm) => {
                Ok(PyList::new(py, &py.detach(move || wm.values())?)?.unbind())
            }
            WaveletMatrixEnum::U128(wm) => {
                Ok(PyList::new(py, &py.detach(move || wm.values())?)?.unbind())
            }
            WaveletMatrixEnum::BigUint(wm) => {
                Ok(PyList::new(py, &py.detach(move || wm.values())?)?.unbind())
            }
        }
    }

    /// Return data[index].
    ///
    /// #### Complexity
    ///
    /// - Time: `O(log V)`  
    /// - Space: `O(1)`
    ///
    /// #### Examples
    /// ```python
    /// wm.access(3)
    /// # 5
    /// ```
    fn access(&self, py: Python<'_>, index: &Bound<'_, PyInt>) -> PyResult<Py<PyInt>> {
        let index = index
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("index must be a non-negative integer"))?;

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => py
                .detach(move || wm.access(index))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U16(wm) => py
                .detach(move || wm.access(index))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U32(wm) => py
                .detach(move || wm.access(index))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U64(wm) => py
                .detach(move || wm.access(index))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U128(wm) => py
                .detach(move || wm.access(index))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::BigUint(wm) => py
                .detach(move || wm.access(index))
                .map(|value| value.into_pyobject(py).unwrap().unbind()),
        }
    }

    /// Count occurrences of value in the prefix range [0, end).
    ///
    /// #### Complexity
    ///
    /// - Time: `O(log V)`  
    /// - Space: `O(1)`
    ///
    /// #### Examples
    /// ```python
    /// wm.rank(5, 9)
    /// # 4
    /// ```
    fn rank(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
    ) -> PyResult<usize> {
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;

        macro_rules! rank_impl {
            ($wm:expr, $number_type:ty) => {{
                let value = match value.extract::<$number_type>() {
                    Ok(value) => value,
                    Err(_) => return Ok(0usize),
                };
                return py.detach(move || $wm.rank(&value, end));
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => rank_impl!(wm, u8),
            WaveletMatrixEnum::U16(wm) => rank_impl!(wm, u16),
            WaveletMatrixEnum::U32(wm) => rank_impl!(wm, u32),
            WaveletMatrixEnum::U64(wm) => rank_impl!(wm, u64),
            WaveletMatrixEnum::U128(wm) => rank_impl!(wm, u128),
            WaveletMatrixEnum::BigUint(wm) => rank_impl!(wm, BigUint),
        }
    }

    /// Return the index of the kth occurrence of value (1-indexed).  
    /// Returns None if it does not exist.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(log V)` (amortized)
    /// - Space: `O(1)`
    ///
    /// #### Examples
    /// ```python
    /// wm.select(5, 4)
    /// # 6
    /// ```
    fn select(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyInt>,
        kth: &Bound<'_, PyInt>,
    ) -> PyResult<Option<usize>> {
        let kth = kth
            .extract::<usize>()
            .map_err(|_| PyValueError::new_err("kth must be a positive integer"))?;

        macro_rules! select_impl {
            ($wm:expr, $number_type:ty) => {{
                let value = match value.extract::<$number_type>() {
                    Ok(value) => value,
                    Err(_) => return Ok(None),
                };
                return py.detach(move || $wm.select(&value, kth));
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => select_impl!(wm, u8),
            WaveletMatrixEnum::U16(wm) => select_impl!(wm, u16),
            WaveletMatrixEnum::U32(wm) => select_impl!(wm, u32),
            WaveletMatrixEnum::U64(wm) => select_impl!(wm, u64),
            WaveletMatrixEnum::U128(wm) => select_impl!(wm, u128),
            WaveletMatrixEnum::BigUint(wm) => select_impl!(wm, BigUint),
        }
    }

    /// Return the k-th smallest value in [start, end) (1-indexed).
    ///
    /// #### Complexity
    ///
    /// - Time: `O(log V)`  
    /// - Space: `O(1)`
    ///
    /// #### Examples
    /// ```python
    /// wm.quantile(2, 12, 8)
    /// # 5
    /// ```
    fn quantile(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
        kth: &Bound<'_, PyInt>,
    ) -> PyResult<Py<PyInt>> {
        let start = start
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start must be a non-negative integer"))?;
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;
        let kth = kth
            .extract::<usize>()
            .map_err(|_| PyValueError::new_err("kth must be a positive integer"))?;

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => py
                .detach(move || wm.quantile(start, end, kth))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U16(wm) => py
                .detach(move || wm.quantile(start, end, kth))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U32(wm) => py
                .detach(move || wm.quantile(start, end, kth))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U64(wm) => py
                .detach(move || wm.quantile(start, end, kth))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::U128(wm) => py
                .detach(move || wm.quantile(start, end, kth))
                .map(|value| PyInt::new(py, value).into()),
            WaveletMatrixEnum::BigUint(wm) => py
                .detach(move || wm.quantile(start, end, kth))
                .map(|value| value.into_pyobject(py).unwrap().unbind()),
        }
    }

    /// Return the most frequent values in [start, end).  
    /// Result items look like {"value": x, "count": c}.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(L (log L) (log V))`  
    /// - Space: `O(L)`  
    /// - L = number of distinct values in the range  
    ///
    /// #### Examples
    /// ```python
    /// wm.topk(1, 10, 2)
    /// # [{'value': 5, 'count': 3}, {'value': 1, 'count': 2}]
    /// ```
    #[pyo3(signature = (start, end, k=None))]
    fn topk(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
        k: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Py<PyList>> {
        let start = start
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start must be a non-negative integer"))?;
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;
        let k = match k {
            Some(k) => Some(
                k.extract::<usize>()
                    .map_err(|_| PyValueError::new_err("k must be a positive integer"))?,
            ),
            None => None,
        };

        macro_rules! topk_impl {
            ($wm:expr) => {{
                let result = py
                    .detach(move || $wm.topk(start, end, k))?
                    .iter()
                    .map(|(value, count)| {
                        let dict = PyDict::new(py);
                        dict.set_item("value", value)?;
                        dict.set_item("count", count)?;
                        Ok(dict)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                return Ok(PyList::new(py, result)?.unbind());
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => topk_impl!(wm),
            WaveletMatrixEnum::U16(wm) => topk_impl!(wm),
            WaveletMatrixEnum::U32(wm) => topk_impl!(wm),
            WaveletMatrixEnum::U64(wm) => topk_impl!(wm),
            WaveletMatrixEnum::U128(wm) => topk_impl!(wm),
            WaveletMatrixEnum::BigUint(wm) => topk_impl!(wm),
        }
    }

    /// Sum of values in [start, end).
    ///
    /// #### Complexity
    ///
    /// - Time: `O(L log V)`  
    /// - Space: `O(L)`
    ///
    /// #### Examples
    /// ```python
    /// wm.range_sum(2, 8)
    /// # 24
    /// ```
    fn range_sum(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
    ) -> PyResult<Py<PyInt>> {
        let start = start
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start must be a non-negative integer"))?;
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;

        let result = py.detach(move || match &self.inner {
            WaveletMatrixEnum::U8(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::U16(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::U32(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::U64(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::U128(wm) => wm.range_sum(start, end),
            WaveletMatrixEnum::BigUint(wm) => wm.range_sum(start, end),
        })?;
        Ok(result.into_pyobject(py)?.unbind())
    }

    /// Intersection of values between two ranges.  
    /// Each item: {"value": x, "count1": a, "count2": b}.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(L log V)`  
    /// - Space: `O(L)`
    ///
    /// #### Examples
    /// ```python
    /// wm.range_intersection(0, 6, 6, 11)
    /// # [{'value': 1, 'count1': 1, 'count2': 1}, {'value': 5, 'count1': 3, 'count2': 2}]
    /// ```
    fn range_intersection(
        &self,
        py: Python<'_>,
        start1: &Bound<'_, PyInt>,
        end1: &Bound<'_, PyInt>,
        start2: &Bound<'_, PyInt>,
        end2: &Bound<'_, PyInt>,
    ) -> PyResult<Py<PyList>> {
        let start1 = start1
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start1 must be a non-negative integer"))?;
        let end1 = end1
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end1 must be a non-negative integer"))?;
        let start2 = start2
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start2 must be a non-negative integer"))?;
        let end2 = end2
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end2 must be a non-negative integer"))?;

        macro_rules! range_intersection_impl {
            ($wm:expr) => {{
                let result = py
                    .detach(move || $wm.range_intersection(start1, end1, start2, end2))?
                    .iter()
                    .map(|(value, count1, count2)| {
                        let dict = PyDict::new(py);
                        dict.set_item("value", value)?;
                        dict.set_item("count1", count1)?;
                        dict.set_item("count2", count2)?;
                        Ok(dict)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                return Ok(PyList::new(py, result)?.unbind());
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => range_intersection_impl!(wm),
            WaveletMatrixEnum::U16(wm) => range_intersection_impl!(wm),
            WaveletMatrixEnum::U32(wm) => range_intersection_impl!(wm),
            WaveletMatrixEnum::U64(wm) => range_intersection_impl!(wm),
            WaveletMatrixEnum::U128(wm) => range_intersection_impl!(wm),
            WaveletMatrixEnum::BigUint(wm) => range_intersection_impl!(wm),
        }
    }

    /// Count elements c in [start, end) such that lower <= c < upper.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(log V)`  
    /// - Space: `O(1)`
    ///
    /// #### Examples
    /// ```python
    /// wm.range_freq(1, 9, 4, 6)
    /// # 4
    /// ```
    #[pyo3(signature = (start, end, lower=None, upper=None))]
    fn range_freq(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
        lower: Option<Bound<'_, PyInt>>,
        upper: Option<Bound<'_, PyInt>>,
    ) -> PyResult<usize> {
        let start = start
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start must be a non-negative integer"))?;
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;

        macro_rules! range_freq_impl {
            ($wm:expr, $number_type:ty) => {{
                let lower = lower.map(|value| value.extract::<$number_type>().ok());
                let upper = upper.map(|value| value.extract::<$number_type>().ok());
                if lower.as_ref().is_some_and(|lower| lower.is_none()) {
                    return Ok(0);
                } else {
                    return py.detach(move || {
                        $wm.range_freq(
                            start,
                            end,
                            lower.flatten().as_ref(),
                            upper.flatten().as_ref(),
                        )
                    });
                }
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => range_freq_impl!(wm, u8),
            WaveletMatrixEnum::U16(wm) => range_freq_impl!(wm, u16),
            WaveletMatrixEnum::U32(wm) => range_freq_impl!(wm, u32),
            WaveletMatrixEnum::U64(wm) => range_freq_impl!(wm, u64),
            WaveletMatrixEnum::U128(wm) => range_freq_impl!(wm, u128),
            WaveletMatrixEnum::BigUint(wm) => range_freq_impl!(wm, BigUint),
        }
    }

    /// List distinct values c in [start, end) satisfying lower <= c < upper, with counts.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(L log V)`  
    /// - Space: `O(L)`
    ///
    /// #### Examples
    /// ```python
    /// wm.range_list(1, 9, 4, 6)
    /// # [{'value': 4, 'count': 1}, {'value': 5, 'count': 3}]
    /// ```
    #[pyo3(signature = (start, end, lower=None, upper=None))]
    fn range_list(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
        lower: Option<Bound<'_, PyInt>>,
        upper: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Py<PyList>> {
        let start = start
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start must be a non-negative integer"))?;
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;

        macro_rules! range_list_impl {
            ($wm:expr, $number_type:ty) => {{
                let lower = lower.map(|value| value.extract::<$number_type>().ok());
                let upper = upper.map(|value| value.extract::<$number_type>().ok());
                if lower.as_ref().is_some_and(|lower| lower.is_none()) {
                    return Ok(PyList::empty(py).into());
                } else {
                    let result = py
                        .detach(move || {
                            $wm.range_list(
                                start,
                                end,
                                lower.flatten().as_ref(),
                                upper.flatten().as_ref(),
                            )
                        })?
                        .iter()
                        .map(|(value, count)| {
                            let dict = PyDict::new(py);
                            dict.set_item("value", value)?;
                            dict.set_item("count", count)?;
                            Ok(dict)
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    return Ok(PyList::new(py, result)?.unbind());
                }
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => range_list_impl!(wm, u8),
            WaveletMatrixEnum::U16(wm) => range_list_impl!(wm, u16),
            WaveletMatrixEnum::U32(wm) => range_list_impl!(wm, u32),
            WaveletMatrixEnum::U64(wm) => range_list_impl!(wm, u64),
            WaveletMatrixEnum::U128(wm) => range_list_impl!(wm, u128),
            WaveletMatrixEnum::BigUint(wm) => range_list_impl!(wm, BigUint),
        }
    }

    /// Return the k largest values in [start, end) with counts.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(k log V)`  
    /// - Space: `O(k)`
    ///
    /// #### Examples
    /// ```python
    /// wm.range_maxk(1, 9, 2)
    /// # [{'value': 6, 'count': 1}, {'value': 5, 'count': 3}]
    /// ```
    #[pyo3(signature = (start, end, k=None))]
    fn range_maxk(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
        k: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Py<PyList>> {
        let start = start
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start must be a non-negative integer"))?;
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;
        let k = match k {
            Some(k) => Some(
                k.extract::<usize>()
                    .map_err(|_| PyValueError::new_err("k must be a positive integer"))?,
            ),
            None => None,
        };

        macro_rules! range_maxk_impl {
            ($wm:expr) => {{
                let result = py
                    .detach(move || $wm.range_maxk(start, end, k))?
                    .iter()
                    .map(|(value, count)| {
                        let dict = PyDict::new(py);
                        dict.set_item("value", value)?;
                        dict.set_item("count", count)?;
                        Ok(dict)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                return Ok(PyList::new(py, result)?.unbind());
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => range_maxk_impl!(wm),
            WaveletMatrixEnum::U16(wm) => range_maxk_impl!(wm),
            WaveletMatrixEnum::U32(wm) => range_maxk_impl!(wm),
            WaveletMatrixEnum::U64(wm) => range_maxk_impl!(wm),
            WaveletMatrixEnum::U128(wm) => range_maxk_impl!(wm),
            WaveletMatrixEnum::BigUint(wm) => range_maxk_impl!(wm),
        }
    }

    /// Return the k smallest values in [start, end) with counts.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(k log V)`  
    /// - Space: `O(k)`
    ///
    /// #### Examples
    /// ```python
    /// wm.range_mink(1, 9, 2)
    /// # [{'value': 1, 'count': 2}, {'value': 2, 'count': 1}]
    /// ```
    #[pyo3(signature = (start, end, k=None))]
    fn range_mink(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
        k: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Py<PyList>> {
        let start = start
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start must be a non-negative integer"))?;
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;
        let k = match k {
            Some(k) => Some(
                k.extract::<usize>()
                    .map_err(|_| PyValueError::new_err("k must be a positive integer"))?,
            ),
            None => None,
        };

        macro_rules! range_mink_impl {
            ($wm:expr) => {{
                let result = py
                    .detach(move || $wm.range_mink(start, end, k))?
                    .iter()
                    .map(|(value, count)| {
                        let dict = PyDict::new(py);
                        dict.set_item("value", value)?;
                        dict.set_item("count", count)?;
                        Ok(dict)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                return Ok(PyList::new(py, result)?.unbind());
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => range_mink_impl!(wm),
            WaveletMatrixEnum::U16(wm) => range_mink_impl!(wm),
            WaveletMatrixEnum::U32(wm) => range_mink_impl!(wm),
            WaveletMatrixEnum::U64(wm) => range_mink_impl!(wm),
            WaveletMatrixEnum::U128(wm) => range_mink_impl!(wm),
            WaveletMatrixEnum::BigUint(wm) => range_mink_impl!(wm),
        }
    }

    /// Return the maximum value c in [start, end) such that c < upper.  
    /// Returns None if no value matches.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(log V)`  
    /// - Space: `O(1)`
    ///
    /// #### Examples
    /// ```python
    /// wm.prev_value(1, 9, 7)
    /// # 6
    /// ```
    #[pyo3(signature = (start, end, upper=None))]
    fn prev_value(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
        upper: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Option<Py<PyInt>>> {
        let start = start
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start must be a non-negative integer"))?;
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;

        macro_rules! prev_value_impl {
            ($wm:expr, $number_type:ty) => {{
                let upper = upper.map(|value| value.extract::<$number_type>().ok());
                return Ok(py
                    .detach(move || $wm.prev_value(start, end, upper.flatten().as_ref()))?
                    .map(|value| value.into_pyobject(py).unwrap().unbind()));
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => prev_value_impl!(wm, u8),
            WaveletMatrixEnum::U16(wm) => prev_value_impl!(wm, u16),
            WaveletMatrixEnum::U32(wm) => prev_value_impl!(wm, u32),
            WaveletMatrixEnum::U64(wm) => prev_value_impl!(wm, u64),
            WaveletMatrixEnum::U128(wm) => prev_value_impl!(wm, u128),
            WaveletMatrixEnum::BigUint(wm) => prev_value_impl!(wm, BigUint),
        }
    }

    /// Return the minimum value c in [start, end) such that lower <= c.  
    /// Returns None if no value matches.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(log V)`  
    /// - Space: `O(1)`
    ///
    /// #### Examples
    /// ```python
    /// wm.next_value(1, 9, 3)
    /// 4
    /// ```
    #[pyo3(signature = (start, end, lower=None))]
    fn next_value(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyInt>,
        end: &Bound<'_, PyInt>,
        lower: Option<Bound<'_, PyInt>>,
    ) -> PyResult<Option<Py<PyInt>>> {
        let start = start
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("start must be a non-negative integer"))?;
        let end = end
            .extract::<usize>()
            .map_err(|_| PyIndexError::new_err("end must be a non-negative integer"))?;

        macro_rules! next_value_impl {
            ($wm:expr, $number_type:ty) => {{
                let lower = lower.map(|value| value.extract::<$number_type>().ok());
                if lower.as_ref().is_some_and(|lower| lower.is_none()) {
                    return Ok(None);
                } else {
                    return Ok(py
                        .detach(move || $wm.next_value(start, end, lower.flatten().as_ref()))?
                        .map(|value| value.into_pyobject(py).unwrap().unbind()));
                }
            }};
        }

        match &self.inner {
            WaveletMatrixEnum::U8(wm) => next_value_impl!(wm, u8),
            WaveletMatrixEnum::U16(wm) => next_value_impl!(wm, u16),
            WaveletMatrixEnum::U32(wm) => next_value_impl!(wm, u32),
            WaveletMatrixEnum::U64(wm) => next_value_impl!(wm, u64),
            WaveletMatrixEnum::U128(wm) => next_value_impl!(wm, u128),
            WaveletMatrixEnum::BigUint(wm) => next_value_impl!(wm, BigUint),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_wavelet_matrix_u8() {
        Python::attach(|py| {
            let elements = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
            let pylist = PyList::new(py, &elements).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let wm = PyWaveletMatrix::new(py, pysequence).unwrap();

            assert_eq!(wm.__len__(py).unwrap(), elements.len());
            assert_eq!(
                wm.__getitem__(py, &PyInt::new(py, 4))
                    .unwrap()
                    .extract::<u8>(py)
                    .unwrap(),
                elements[4]
            );
            assert_eq!(
                wm.__getitem__(py, &PySlice::new(py, 2, 6, 1))
                    .unwrap()
                    .extract::<Vec<u8>>(py)
                    .unwrap(),
                elements[2..6].to_vec()
            );
            assert_eq!(
                wm.__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__repr__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__copy__(py).unwrap().__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.values(py).unwrap().extract::<Vec<u8>>(py).unwrap(),
                elements
            );
            assert_eq!(
                wm.access(py, &PyInt::new(py, 3))
                    .unwrap()
                    .extract::<u8>(py)
                    .unwrap(),
                elements[3]
            );
            assert_eq!(
                wm.rank(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 9))
                    .unwrap(),
                4
            );
            assert_eq!(
                wm.select(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 4))
                    .unwrap(),
                Some(6usize)
            );
            assert_eq!(
                wm.quantile(
                    py,
                    &PyInt::new(py, 2),
                    &PyInt::new(py, 12),
                    &PyInt::new(py, 8)
                )
                .unwrap()
                .extract::<u8>(py)
                .unwrap(),
                elements[2]
            );
            assert_eq!(
                wm.topk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 10),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_sum(py, &PyInt::new(py, 2), &PyInt::new(py, 8))
                    .unwrap()
                    .extract::<u8>(py)
                    .unwrap(),
                24
            );
            assert_eq!(
                wm.range_intersection(
                    py,
                    &PyInt::new(py, 0),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 11)
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_freq(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap(),
                4
            );
            assert_eq!(
                wm.range_list(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_maxk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_mink(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.prev_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 7))
                )
                .unwrap()
                .unwrap()
                .extract::<u8>(py)
                .unwrap(),
                elements[7]
            );
            assert_eq!(
                wm.next_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 3))
                )
                .unwrap()
                .unwrap()
                .extract::<u8>(py)
                .unwrap(),
                elements[1]
            );
        });
    }

    #[test]
    fn test_wavelet_matrix_u16() {
        Python::attach(|py| {
            let elements = vec![
                5 << 8,
                4 << 8,
                5 << 8,
                5 << 8,
                2 << 8,
                1 << 8,
                5 << 8,
                6 << 8,
                1 << 8,
                3 << 8,
                5 << 8,
                0 << 8,
            ];
            let pylist = PyList::new(py, &elements).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let wm = PyWaveletMatrix::new(py, pysequence).unwrap();

            assert_eq!(wm.__len__(py).unwrap(), elements.len());
            assert_eq!(
                wm.__getitem__(py, &PyInt::new(py, 4))
                    .unwrap()
                    .extract::<u16>(py)
                    .unwrap(),
                elements[4]
            );
            assert_eq!(
                wm.__getitem__(py, &PySlice::new(py, 2, 6, 1))
                    .unwrap()
                    .extract::<Vec<u16>>(py)
                    .unwrap(),
                elements[2..6].to_vec()
            );
            assert_eq!(
                wm.__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__repr__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__copy__(py).unwrap().__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.values(py).unwrap().extract::<Vec<u16>>(py).unwrap(),
                elements
            );
            assert_eq!(
                wm.access(py, &PyInt::new(py, 3))
                    .unwrap()
                    .extract::<u16>(py)
                    .unwrap(),
                elements[3]
            );
            assert_eq!(
                wm.rank(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 9))
                    .unwrap(),
                4
            );
            assert_eq!(
                wm.select(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 4))
                    .unwrap(),
                Some(6usize)
            );
            assert_eq!(
                wm.quantile(
                    py,
                    &PyInt::new(py, 2),
                    &PyInt::new(py, 12),
                    &PyInt::new(py, 8)
                )
                .unwrap()
                .extract::<u16>(py)
                .unwrap(),
                elements[2]
            );
            assert_eq!(
                wm.topk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 10),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_sum(py, &PyInt::new(py, 2), &PyInt::new(py, 8))
                    .unwrap()
                    .extract::<u16>(py)
                    .unwrap(),
                24 << 8
            );
            assert_eq!(
                wm.range_intersection(
                    py,
                    &PyInt::new(py, 0),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 11)
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_freq(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap(),
                4
            );
            assert_eq!(
                wm.range_list(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_maxk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_mink(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.prev_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 7 << 8))
                )
                .unwrap()
                .unwrap()
                .extract::<u16>(py)
                .unwrap(),
                elements[7]
            );
            assert_eq!(
                wm.next_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 3 << 8))
                )
                .unwrap()
                .unwrap()
                .extract::<u16>(py)
                .unwrap(),
                elements[1]
            );
        });
    }

    #[test]
    fn test_wavelet_matrix_u32() {
        Python::attach(|py| {
            let elements = vec![
                5 << 16,
                4 << 16,
                5 << 16,
                5 << 16,
                2 << 16,
                1 << 16,
                5 << 16,
                6 << 16,
                1 << 16,
                3 << 16,
                5 << 16,
                0 << 16,
            ];
            let pylist = PyList::new(py, &elements).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let wm = PyWaveletMatrix::new(py, pysequence).unwrap();

            assert_eq!(wm.__len__(py).unwrap(), elements.len());
            assert_eq!(
                wm.__getitem__(py, &PyInt::new(py, 4))
                    .unwrap()
                    .extract::<u32>(py)
                    .unwrap(),
                elements[4]
            );
            assert_eq!(
                wm.__getitem__(py, &PySlice::new(py, 2, 6, 1))
                    .unwrap()
                    .extract::<Vec<u32>>(py)
                    .unwrap(),
                elements[2..6].to_vec()
            );
            assert_eq!(
                wm.__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__repr__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__copy__(py).unwrap().__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.values(py).unwrap().extract::<Vec<u32>>(py).unwrap(),
                elements
            );
            assert_eq!(
                wm.access(py, &PyInt::new(py, 3))
                    .unwrap()
                    .extract::<u32>(py)
                    .unwrap(),
                elements[3]
            );
            assert_eq!(
                wm.rank(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 9))
                    .unwrap(),
                4
            );
            assert_eq!(
                wm.select(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 4))
                    .unwrap(),
                Some(6usize)
            );
            assert_eq!(
                wm.quantile(
                    py,
                    &PyInt::new(py, 2),
                    &PyInt::new(py, 12),
                    &PyInt::new(py, 8)
                )
                .unwrap()
                .extract::<u32>(py)
                .unwrap(),
                elements[2]
            );
            assert_eq!(
                wm.topk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 10),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_sum(py, &PyInt::new(py, 2), &PyInt::new(py, 8))
                    .unwrap()
                    .extract::<u32>(py)
                    .unwrap(),
                24 << 16
            );
            assert_eq!(
                wm.range_intersection(
                    py,
                    &PyInt::new(py, 0),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 11)
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_freq(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap(),
                4
            );
            assert_eq!(
                wm.range_list(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_maxk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_mink(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.prev_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 7 << 16))
                )
                .unwrap()
                .unwrap()
                .extract::<u32>(py)
                .unwrap(),
                elements[7]
            );
            assert_eq!(
                wm.next_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 3 << 16))
                )
                .unwrap()
                .unwrap()
                .extract::<u32>(py)
                .unwrap(),
                elements[1]
            );
        });
    }

    #[test]
    fn test_wavelet_matrix_u64() {
        Python::attach(|py| {
            let elements = vec![
                5 << 32,
                4 << 32,
                5 << 32,
                5 << 32,
                2 << 32,
                1 << 32,
                5 << 32,
                6 << 32,
                1 << 32,
                3 << 32,
                5 << 32,
                0 << 32,
            ];
            let pylist = PyList::new(py, &elements).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let wm = PyWaveletMatrix::new(py, pysequence).unwrap();

            assert_eq!(wm.__len__(py).unwrap(), elements.len());
            assert_eq!(
                wm.__getitem__(py, &PyInt::new(py, 4))
                    .unwrap()
                    .extract::<u64>(py)
                    .unwrap(),
                elements[4]
            );
            assert_eq!(
                wm.__getitem__(py, &PySlice::new(py, 2, 6, 1))
                    .unwrap()
                    .extract::<Vec<u64>>(py)
                    .unwrap(),
                elements[2..6].to_vec()
            );
            assert_eq!(
                wm.__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__repr__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__copy__(py).unwrap().__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.values(py).unwrap().extract::<Vec<u64>>(py).unwrap(),
                elements
            );
            assert_eq!(
                wm.access(py, &PyInt::new(py, 3))
                    .unwrap()
                    .extract::<u64>(py)
                    .unwrap(),
                elements[3]
            );
            assert_eq!(
                wm.rank(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 9))
                    .unwrap(),
                4
            );
            assert_eq!(
                wm.select(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 4))
                    .unwrap(),
                Some(6usize)
            );
            assert_eq!(
                wm.quantile(
                    py,
                    &PyInt::new(py, 2),
                    &PyInt::new(py, 12),
                    &PyInt::new(py, 8)
                )
                .unwrap()
                .extract::<u64>(py)
                .unwrap(),
                elements[2]
            );
            assert_eq!(
                wm.topk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 10),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_sum(py, &PyInt::new(py, 2), &PyInt::new(py, 8))
                    .unwrap()
                    .extract::<u64>(py)
                    .unwrap(),
                24 << 32
            );
            assert_eq!(
                wm.range_intersection(
                    py,
                    &PyInt::new(py, 0),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 11)
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_freq(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap(),
                4
            );
            assert_eq!(
                wm.range_list(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_maxk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_mink(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.prev_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 7u64 << 32))
                )
                .unwrap()
                .unwrap()
                .extract::<u64>(py)
                .unwrap(),
                elements[7]
            );
            assert_eq!(
                wm.next_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 3u64 << 32))
                )
                .unwrap()
                .unwrap()
                .extract::<u64>(py)
                .unwrap(),
                elements[1]
            );
        });
    }

    #[test]
    fn test_wavelet_matrix_u128() {
        Python::attach(|py| {
            let elements = vec![
                5 << 64,
                4 << 64,
                5 << 64,
                5 << 64,
                2 << 64,
                1 << 64,
                5 << 64,
                6 << 64,
                1 << 64,
                3 << 64,
                5 << 64,
                0 << 64,
            ];
            let pylist = PyList::new(py, &elements).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let wm = PyWaveletMatrix::new(py, pysequence).unwrap();

            assert_eq!(wm.__len__(py).unwrap(), elements.len());
            assert_eq!(
                wm.__getitem__(py, &PyInt::new(py, 4))
                    .unwrap()
                    .extract::<u128>(py)
                    .unwrap(),
                elements[4]
            );
            assert_eq!(
                wm.__getitem__(py, &PySlice::new(py, 2, 6, 1))
                    .unwrap()
                    .extract::<Vec<u128>>(py)
                    .unwrap(),
                elements[2..6].to_vec()
            );
            assert_eq!(
                wm.__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__repr__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__copy__(py).unwrap().__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.values(py).unwrap().extract::<Vec<u128>>(py).unwrap(),
                elements
            );
            assert_eq!(
                wm.access(py, &PyInt::new(py, 3))
                    .unwrap()
                    .extract::<u128>(py)
                    .unwrap(),
                elements[3]
            );
            assert_eq!(
                wm.rank(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 9))
                    .unwrap(),
                4
            );
            assert_eq!(
                wm.select(py, &PyInt::new(py, elements[0]), &PyInt::new(py, 4))
                    .unwrap(),
                Some(6usize)
            );
            assert_eq!(
                wm.quantile(
                    py,
                    &PyInt::new(py, 2),
                    &PyInt::new(py, 12),
                    &PyInt::new(py, 8)
                )
                .unwrap()
                .extract::<u128>(py)
                .unwrap(),
                elements[2]
            );
            assert_eq!(
                wm.topk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 10),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_sum(py, &PyInt::new(py, 2), &PyInt::new(py, 8))
                    .unwrap()
                    .extract::<u128>(py)
                    .unwrap(),
                24 << 64
            );
            assert_eq!(
                wm.range_intersection(
                    py,
                    &PyInt::new(py, 0),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 11)
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_freq(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap(),
                4
            );
            assert_eq!(
                wm.range_list(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, elements[1])),
                    Some(PyInt::new(py, elements[7]))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_maxk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_mink(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.prev_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 7u128 << 64))
                )
                .unwrap()
                .unwrap()
                .extract::<u128>(py)
                .unwrap(),
                elements[7]
            );
            assert_eq!(
                wm.next_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 3u128 << 64))
                )
                .unwrap()
                .unwrap()
                .extract::<u128>(py)
                .unwrap(),
                elements[1]
            );
        });
    }

    #[test]
    fn test_wavelet_matrix_biguint() {
        Python::attach(|py| {
            let elements = vec![
                BigUint::from(5u32) << 128,
                BigUint::from(4u32) << 128,
                BigUint::from(5u32) << 128,
                BigUint::from(5u32) << 128,
                BigUint::from(2u32) << 128,
                BigUint::from(1u32) << 128,
                BigUint::from(5u32) << 128,
                BigUint::from(6u32) << 128,
                BigUint::from(1u32) << 128,
                BigUint::from(3u32) << 128,
                BigUint::from(5u32) << 128,
                BigUint::from(0u32) << 128,
            ];
            let pylist = elements.clone().into_pyobject(py).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let wm = PyWaveletMatrix::new(py, pysequence).unwrap();

            assert_eq!(wm.__len__(py).unwrap(), elements.len());
            assert_eq!(
                wm.__getitem__(py, &PyInt::new(py, 4))
                    .unwrap()
                    .extract::<BigUint>(py)
                    .unwrap(),
                elements[4]
            );
            assert_eq!(
                wm.__getitem__(py, &PySlice::new(py, 2, 6, 1))
                    .unwrap()
                    .extract::<Vec<BigUint>>(py)
                    .unwrap(),
                elements[2..6].to_vec()
            );
            assert_eq!(
                wm.__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__repr__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.__copy__(py).unwrap().__str__(py).unwrap(),
                format!("WaveletMatrix({:?})", elements)
            );
            assert_eq!(
                wm.values(py).unwrap().extract::<Vec<BigUint>>(py).unwrap(),
                elements
            );
            assert_eq!(
                wm.access(py, &PyInt::new(py, 3))
                    .unwrap()
                    .extract::<BigUint>(py)
                    .unwrap(),
                elements[3]
            );
            assert_eq!(
                wm.rank(
                    py,
                    &elements[0].clone().into_pyobject(py).unwrap(),
                    &PyInt::new(py, 9)
                )
                .unwrap(),
                4
            );
            assert_eq!(
                wm.select(
                    py,
                    &elements[0].clone().into_pyobject(py).unwrap(),
                    &PyInt::new(py, 4)
                )
                .unwrap(),
                Some(6usize)
            );
            assert_eq!(
                wm.quantile(
                    py,
                    &PyInt::new(py, 2),
                    &PyInt::new(py, 12),
                    &PyInt::new(py, 8)
                )
                .unwrap()
                .extract::<BigUint>(py)
                .unwrap(),
                elements[2]
            );
            assert_eq!(
                wm.topk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 10),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_sum(py, &PyInt::new(py, 2), &PyInt::new(py, 8))
                    .unwrap()
                    .extract::<BigUint>(py)
                    .unwrap(),
                BigUint::from(24u32) << 128
            );
            assert_eq!(
                wm.range_intersection(
                    py,
                    &PyInt::new(py, 0),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 6),
                    &PyInt::new(py, 11)
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_freq(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(elements[1].clone().into_pyobject(py).unwrap()),
                    Some(elements[7].clone().into_pyobject(py).unwrap())
                )
                .unwrap(),
                4
            );
            assert_eq!(
                wm.range_list(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(elements[1].clone().into_pyobject(py).unwrap()),
                    Some(elements[7].clone().into_pyobject(py).unwrap())
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_maxk(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.range_mink(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(PyInt::new(py, 2))
                )
                .unwrap()
                .extract::<Vec<Py<PyDict>>>(py)
                .unwrap()
                .len(),
                2
            );
            assert_eq!(
                wm.prev_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(
                        (BigUint::from(7u128) << 128usize)
                            .into_pyobject(py)
                            .unwrap()
                    )
                )
                .unwrap()
                .unwrap()
                .extract::<BigUint>(py)
                .unwrap(),
                elements[7]
            );
            assert_eq!(
                wm.next_value(
                    py,
                    &PyInt::new(py, 1),
                    &PyInt::new(py, 9),
                    Some(
                        (BigUint::from(3u128) << 128usize)
                            .into_pyobject(py)
                            .unwrap()
                    )
                )
                .unwrap()
                .unwrap()
                .extract::<BigUint>(py)
                .unwrap(),
                elements[1]
            );
        });
    }
}
