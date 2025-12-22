use crate::{
    traits::wavelet_matrix::wavelet_matrix::WaveletMatrixTrait,
    wavelet_matrix::wavelet_matrix::WaveletMatrix,
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyInt, PyList, PySequence, PySlice, PySliceIndices},
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
/// A Wavelet Matrix data structure for efficient rank, select, and quantile queries.
///
/// The Wavelet Matrix decomposes a sequence into multiple bit vectors,
/// one for each bit position. This allows for efficient queries on the sequence.  
/// This class supports various integer types, automatically selecting
/// the appropriate internal representation based on the input data.  
#[derive(Clone)]
#[pyclass(unsendable, name = "WaveletMatrix")]
pub(crate) struct PyWaveletMatrix {
    inner: WaveletMatrixEnum,
}

#[pymethods]
impl PyWaveletMatrix {
    /// Creates a new Wavelet Matrix from the given list or tuple of integers.
    #[new]
    pub(crate) fn new(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let values: Vec<BigUint> = data
            .clone()
            .cast_into::<PySequence>()
            .map_err(|_| PyValueError::new_err("Input must be an Iterable object"))?
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
    pub(crate) fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
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
    pub(crate) fn __getitem__(
        &self,
        py: Python<'_>,
        index: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
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

    pub(crate) fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(move || match &self.inner {
            WaveletMatrixEnum::U8(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U16(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U32(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U64(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::U128(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
            WaveletMatrixEnum::BigUint(wm) => Ok(format!("WaveletMatrix({:?})", wm.values()?)),
        })
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
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

    /// Get all values in the Wavelet Matrix as a list.
    ///
    /// # Complexity
    ///
    /// - Time: `O(N log V)`  
    ///
    /// where:
    /// - `N` = length of the sequence  
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Exapmles
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.values()
    /// [5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
    /// ```
    pub(crate) fn values(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
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

    /// Access the value at the specified index.
    ///
    /// # Complexity
    ///
    /// - Time: `O(log V)`  
    ///
    /// where:
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.access(3)
    /// 5
    /// ```
    pub(crate) fn access(&self, py: Python<'_>, index: &Bound<'_, PyInt>) -> PyResult<Py<PyInt>> {
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

    /// Counts the occurrences of the given value in the range [0, end).
    ///
    /// # Complexity
    ///
    /// - Time: `O(log V)`  
    ///
    /// where:
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.rank(5, 9)
    /// 4
    /// ```
    pub(crate) fn rank(
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

    /// Finds the position of the k-th occurrence of the given value.
    ///
    /// # Complexity
    ///
    /// - Time: `O(log V)` (amortized)
    ///
    /// where:
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.select(5, 4)
    /// 6
    /// ```
    pub(crate) fn select(
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

    /// Find the k-th smallest value in the range [start, end) (1-indexed).
    ///
    /// # Complexity
    ///
    /// - Time: `O(log V)`  
    ///
    /// where:
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.quantile(2, 12, 8)
    /// 5
    /// ```
    pub(crate) fn quantile(
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

    /// Finds the top-k most frequent elements in the range [start, end).
    ///
    /// # Complexity
    ///
    /// - Time: `O(L (log L) (log V))`  
    ///
    /// where:
    /// - `L` = the number of distinct values in the range `[start, end)`
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.topk(1, 10, 2)
    /// [{'value': 5, 'count': 3}, {'value': 1, 'count': 2}]
    /// ```
    #[pyo3(signature = (start, end, k=None))]
    pub(crate) fn topk(
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

    /// Computes the sum of values in the range [start, end).
    ///
    /// # Complexity
    ///
    /// - Time: `O(L log V)`  
    ///
    /// where:
    /// - `L` = the number of distinct values `c` in the range `[start, end)`
    ///   that satisfy `lower <= c < upper`
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.range_sum(2, 8)
    /// 24
    /// ```
    pub(crate) fn range_sum(
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

    /// Finds the intersection of values in the two ranges [start1, end1) and [start2, end2).
    ///
    /// # Complexity
    ///
    /// - Time: `O(L log V)`  
    ///
    /// where:
    /// - `L` = the number of distinct values `c` in the intersection of the two ranges
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.range_intersection(0, 6, 6, 11)
    /// [{'value': 1, 'count1': 1, 'count2': 1}, {'value': 5, 'count1': 3, 'count2': 2}]
    /// ```
    pub(crate) fn range_intersection(
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

    /// Counts the number of elements c in the range [start, end) such that lower <= c < upper.
    ///
    /// # Complexity
    ///
    /// - Time: `O(log V)`  
    ///
    /// where:
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.range_freq(1, 9, 4, 6)
    /// 4
    /// ```
    #[pyo3(signature = (start, end, lower=None, upper=None))]
    pub fn range_freq(
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

    /// Lists all elements c in the range [start, end) such that lower <= c < upper.
    ///
    /// # Complexity
    ///
    /// - Time: `O(L log V)`  
    ///
    /// where:
    /// - `L` = the number of distinct values `c` in the range `[start, end)`
    ///   that satisfy `lower <= c < upper`
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.range_list(1, 9, 4, 6)
    /// [{'value': 4, 'count': 1}, {'value': 5, 'count': 3}]
    /// ```
    #[pyo3(signature = (start, end, lower=None, upper=None))]
    pub fn range_list(
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

    /// Finds the k largest values in the range [start, end).
    ///
    /// # Complexity
    ///
    /// - Time: `O(k log V)`  
    ///
    /// where:
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.range_maxk(1, 9, 2)
    /// [{'value': 6, 'count': 1}, {'value': 5, 'count': 3}]
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

    /// Finds the k smallest values in the range [start, end).
    ///
    /// # Complexity
    ///
    /// - Time: `O(k log V)`  
    ///
    /// where:
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.range_mink(1, 9, 2)
    /// [{'value': 1, 'count': 2}, {'value': 2, 'count': 1}]
    /// ```
    #[pyo3(signature = (start, end, k=None))]
    pub fn range_mink(
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

    /// Finds the maximum value c in the range [start, end) such that c < upper.
    ///
    /// # Complexity
    ///
    /// - Time: `O(log V)`  
    ///
    /// where:
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.prev_value(1, 9, 7)
    /// 6
    /// ```
    #[pyo3(signature = (start, end, upper=None))]
    pub fn prev_value(
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

    /// Finds the minimum value c in the range [start, end) such that lower <= c.
    ///
    /// # Complexity
    ///
    /// - Time: `O(log V)`  
    ///
    /// where:
    /// - `V` = range of possible values (max value domain)
    ///
    /// # Examples
    /// ```python
    /// >>> from wavelet_matrix import WaveletMatrix
    /// >>> wm = WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    /// >>> wm.next_value(1, 9, 3)
    /// 4
    /// ```
    #[pyo3(signature = (start, end, lower=None))]
    pub fn next_value(
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
