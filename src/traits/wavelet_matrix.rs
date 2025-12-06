use super::bit_vector::BitVectorTrait;
use num_bigint::{BigUint, ToBigUint};
use num_traits::{One, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
};
use std::{
    cmp::PartialEq,
    collections::{BinaryHeap, HashMap},
    iter::zip,
    ops::{BitAnd, BitOr, BitOrAssign, Shl, ShlAssign, Shr},
};

/// A Wavelet Matrix data structure for efficient rank, select, and quantile queries.
///
/// The Wavelet Matrix decomposes a sequence into multiple bit vectors,
/// one for each bit position. This allows for efficient queries on the sequence.
pub(crate) trait WaveletMatrixTrait<NumberType, BitVectorType>
where
    NumberType: BitAnd<NumberType, Output = NumberType>
        + BitOr<NumberType, Output = NumberType>
        + BitOrAssign
        + Clone
        + One
        + Ord
        + PartialEq
        + Shl<usize, Output = NumberType>
        + ShlAssign<usize>
        + ToBigUint
        + Zero
        + 'static,
    for<'a> &'a NumberType: Shl<usize, Output = NumberType> + Shr<usize, Output = NumberType>,
    BitVectorType: BitVectorTrait,
{
    /// Get the length of the Wavelet Matrix.
    fn len(&self) -> usize;

    /// Get the height (number of layers) of the Wavelet Matrix.
    fn height(&self) -> usize;

    /// Get the bit vectors (layers) of the Wavelet Matrix.
    fn get_layers(&self) -> &[BitVectorType];

    /// Get the number of zeros in each layer.
    fn get_zeros(&self) -> &[usize];

    /// Get the begin index for each unique value.
    #[inline]
    fn begin_index(&self, value: &NumberType) -> Option<usize> {
        let mut start = 0usize;
        let mut end = self.len();
        for (i, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let bit = (value >> (self.height() - i - 1) & NumberType::one()).is_one();
            if bit {
                start = zero + layer.rank(bit, start).ok()?;
                end = zero + layer.rank(bit, end).ok()?;
            } else {
                start = layer.rank(bit, start).ok()?;
                end = layer.rank(bit, end).ok()?;
            }
            debug_assert!(end <= self.len());
        }

        debug_assert!(start <= end);
        if start == end { None } else { Some(start) }
    }

    /// Get the value at the specified position.
    fn access(&self, mut index: usize) -> PyResult<NumberType> {
        if index >= self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        let mut result = NumberType::zero();
        for (layer, zero) in zip(self.get_layers(), self.get_zeros()) {
            let bit = layer.access(index)?;
            result <<= 1;
            if bit {
                result |= NumberType::one();
                index = zero + layer.rank(bit, index)?;
            } else {
                index = layer.rank(bit, index)?;
            }
            debug_assert!(index < self.len());
        }

        Ok(result)
    }

    /// Count the number of occurrences of a value in the range [0, end).
    fn rank(&self, value: &NumberType, mut end: usize) -> PyResult<usize> {
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        let begin_index = match self.begin_index(value) {
            Some(index) => index,
            None => return Ok(0usize),
        };

        for (i, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let bit = (value >> (self.height() - i - 1) & NumberType::one()).is_one();
            if bit {
                end = zero + layer.rank(bit, end)?;
            } else {
                end = layer.rank(bit, end)?;
            }
            debug_assert!(end <= self.len());
        }

        debug_assert!(begin_index <= end);
        Ok(end - begin_index)
    }

    /// Find the position of the k-th occurrence of a value (1-indexed).
    fn select(&self, value: &NumberType, kth: usize) -> PyResult<Option<usize>> {
        if kth.is_zero() {
            return Err(PyValueError::new_err("kth must be greater than 0"));
        }
        let begin_index = match self.begin_index(value) {
            Some(index) => index,
            None => return Ok(None),
        };

        let mut index = begin_index + kth - 1;
        for (i, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate().rev() {
            let bit = (value >> (self.height() - i - 1) & NumberType::one()).is_one();
            if bit {
                index -= zero;
            }
            index = match layer.select(bit, index + 1)? {
                Some(index) => index,
                None => return Ok(None),
            };
            debug_assert!(index < self.len());
        }

        Ok(Some(index))
    }

    /// Find the k-th smallest value in the range [start, end) (1-indexed).
    fn quantile(&self, start: usize, end: usize, mut kth: usize) -> PyResult<NumberType> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if kth.is_zero() {
            return Err(PyValueError::new_err("kth must be greater than 0"));
        }
        if kth > end - start {
            return Err(PyValueError::new_err("kth is larger than the range size"));
        }

        let mut left = start;
        let mut right = end;
        let mut result = NumberType::zero();
        for (layer, zero) in zip(self.get_layers(), self.get_zeros()) {
            let count_zeros = layer.rank(false, right)? - layer.rank(false, left)?;
            let bit = if kth <= count_zeros {
                false
            } else {
                kth -= count_zeros;
                true
            };

            result <<= 1;
            if bit {
                result |= NumberType::one();
                left = zero + layer.rank(bit, left)?;
                right = zero + layer.rank(bit, right)?;
            } else {
                left = layer.rank(bit, left)?;
                right = layer.rank(bit, right)?;
            }
            debug_assert!(right <= self.len());
        }

        Ok(result)
    }

    // Count values in [start, end) with the top-k highest frequencies.
    fn topk(
        &self,
        start: usize,
        end: usize,
        k: Option<usize>,
    ) -> PyResult<Vec<HashMap<String, BigUint>>> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if k.is_some_and(|k| k.is_zero()) {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        let k = k.unwrap_or(end - start);

        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        struct QueueItem<T> {
            len: usize,
            depth: usize,
            start: usize,
            end: usize,
            value: T,
        }
        let mut heap = BinaryHeap::new();
        heap.push(QueueItem::<NumberType> {
            len: end - start,
            depth: 0,
            start,
            end,
            value: NumberType::zero(),
        });

        let mut result = Vec::new();
        while let Some(QueueItem {
            len,
            depth,
            start,
            end,
            value,
        }) = heap.pop()
        {
            if depth == self.height() {
                result.push(HashMap::from([
                    (
                        "value".to_string(),
                        value
                            .to_biguint()
                            .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                    ),
                    (
                        "count".to_string(),
                        len.to_biguint()
                            .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                    ),
                ]));

                if result.len() == k {
                    break;
                }
                continue;
            }

            let layer = &self.get_layers()[depth];
            let zero = self.get_zeros()[depth];

            let left_zero = layer.rank(false, start)?;
            let right_zero = layer.rank(false, end)?;
            if left_zero < right_zero {
                heap.push(QueueItem {
                    len: right_zero - left_zero,
                    depth: depth + 1,
                    start: left_zero,
                    end: right_zero,
                    value: &value << 1usize,
                });
            }

            let left_one = zero + layer.rank(true, start)?;
            let right_one = zero + layer.rank(true, end)?;
            if left_one < right_one {
                heap.push(QueueItem {
                    len: right_one - left_one,
                    depth: depth + 1,
                    start: left_one,
                    end: right_one,
                    value: (&value << 1usize) | NumberType::one(),
                });
            }
        }

        Ok(result)
    }

    /// Get the sum of elements in the range [start, end).
    fn range_sum(&self, start: usize, end: usize) -> PyResult<BigUint> {
        let result = self.range_list(start, end, None, None)?.iter().try_fold(
            BigUint::zero(),
            |acc, item| -> PyResult<BigUint> {
                let value = item
                    .get("value")
                    .ok_or(PyRuntimeError::new_err("invalid value type"))?;
                let count = item
                    .get("count")
                    .ok_or(PyRuntimeError::new_err("invalid count type"))?;

                Ok(acc + value * count)
            },
        )?;

        Ok(result)
    }

    /// Get the intersection of two ranges [start1, end1) and [start2, end2).
    fn range_intersection(
        &self,
        start1: usize,
        end1: usize,
        start2: usize,
        end2: usize,
    ) -> PyResult<Vec<HashMap<String, BigUint>>> {
        if start1 >= end1 {
            return Err(PyValueError::new_err("start1 must be less than end1"));
        }
        if end1 > self.len() {
            return Err(PyIndexError::new_err("end1 index out of bounds"));
        }
        if start2 >= end2 {
            return Err(PyValueError::new_err("start2 must be less than end2"));
        }
        if end2 > self.len() {
            return Err(PyIndexError::new_err("end2 index out of bounds"));
        }

        struct StackItem<T> {
            start1: usize,
            end1: usize,
            start2: usize,
            end2: usize,
            value: T,
        }
        let mut stack = vec![StackItem {
            start1,
            end1,
            start2,
            end2,
            value: NumberType::zero(),
        }];

        for (layer, zero) in zip(self.get_layers(), self.get_zeros()) {
            let mut next_stack = Vec::new();

            for StackItem {
                start1,
                end1,
                start2,
                end2,
                value,
            } in stack
            {
                let left1_zero = layer.rank(false, start1)?;
                let right1_zero = layer.rank(false, end1)?;
                let left2_zero = layer.rank(false, start2)?;
                let right2_zero = layer.rank(false, end2)?;
                if left1_zero < right1_zero && left2_zero < right2_zero {
                    next_stack.push(StackItem {
                        start1: left1_zero,
                        end1: right1_zero,
                        start2: left2_zero,
                        end2: right2_zero,
                        value: &value << 1,
                    });
                }

                let left1_one = zero + layer.rank(true, start1)?;
                let right1_one = zero + layer.rank(true, end1)?;
                let left2_one = zero + layer.rank(true, start2)?;
                let right2_one = zero + layer.rank(true, end2)?;
                if left1_one < right1_one && left2_one < right2_one {
                    next_stack.push(StackItem {
                        start1: left1_one,
                        end1: right1_one,
                        start2: left2_one,
                        end2: right2_one,
                        value: (&value << 1) | NumberType::one(),
                    });
                }
            }

            stack = next_stack;
        }

        let result = stack
            .into_iter()
            .map(
                |StackItem {
                     start1,
                     end1,
                     start2,
                     end2,
                     value,
                 }| {
                    Ok(HashMap::from([
                        (
                            "value".to_string(),
                            value
                                .to_biguint()
                                .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                        ),
                        (
                            "count1".to_string(),
                            (end1 - start1)
                                .to_biguint()
                                .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                        ),
                        (
                            "count2".to_string(),
                            (end2 - start2)
                                .to_biguint()
                                .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                        ),
                    ]))
                },
            )
            .collect::<PyResult<Vec<_>>>()?;

        Ok(result)
    }

    /// Get the total count of values c in the range [start, end) such that lower <= c < upper.
    fn range_freq(
        &self,
        start: usize,
        end: usize,
        lower: Option<&NumberType>,
        upper: Option<&NumberType>,
    ) -> PyResult<usize> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if lower
            .zip(upper)
            .is_some_and(|(lower, upper)| lower >= upper)
        {
            return Err(PyValueError::new_err("lower must be less than upper"));
        }

        struct StackItem<T> {
            start: usize,
            end: usize,
            value: T,
        }
        let mut stack = vec![StackItem {
            start,
            end,
            value: NumberType::zero(),
        }];

        for (depth, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let mut next_stack = Vec::new();

            for StackItem { start, end, value } in stack {
                let left_zero = layer.rank(false, start)?;
                let right_zero = layer.rank(false, end)?;
                let next_value_zero = &value << 1;
                if lower.is_none_or(|lower| lower >> (self.height() - depth - 1) <= next_value_zero)
                    && upper
                        .is_none_or(|upper| next_value_zero <= upper >> (self.height() - depth - 1))
                    && left_zero < right_zero
                {
                    next_stack.push(StackItem {
                        start: left_zero,
                        end: right_zero,
                        value: next_value_zero,
                    });
                }

                let left_one = zero + layer.rank(true, start)?;
                let right_one = zero + layer.rank(true, end)?;
                let next_value_one = (&value << 1) | NumberType::one();
                if lower
                    .is_none_or(|lower| (lower >> (self.height() - depth - 1)) <= next_value_one)
                    && upper.is_none_or(|upper| {
                        next_value_one <= (upper >> (self.height() - depth - 1))
                    })
                    && left_one < right_one
                {
                    next_stack.push(StackItem {
                        start: left_one,
                        end: right_one,
                        value: next_value_one,
                    });
                }
            }

            stack = next_stack;
        }

        let result = stack
            .iter()
            .filter(|StackItem { value, .. }| {
                lower.is_none_or(|lower| lower <= value) && upper.is_none_or(|upper| value < upper)
            })
            .map(|StackItem { start, end, .. }| end - start)
            .sum();
        Ok(result)
    }

    /// Get a list of values c in the range [start, end) such that lower <= c < upper.
    fn range_list(
        &self,
        start: usize,
        end: usize,
        lower: Option<&NumberType>,
        upper: Option<&NumberType>,
    ) -> PyResult<Vec<HashMap<String, BigUint>>> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if lower
            .zip(upper)
            .is_some_and(|(lower, upper)| lower >= upper)
        {
            return Err(PyValueError::new_err("lower must be less than upper"));
        }

        struct StackItem<T> {
            start: usize,
            end: usize,
            value: T,
        }
        let mut stack = vec![StackItem {
            start,
            end,
            value: NumberType::zero(),
        }];

        for (depth, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let mut next_stack = Vec::new();

            for StackItem { start, end, value } in stack {
                let left_zero = layer.rank(false, start)?;
                let right_zero = layer.rank(false, end)?;
                let next_value_zero = &value << 1;
                if lower.is_none_or(|lower| lower >> (self.height() - depth - 1) <= next_value_zero)
                    && upper
                        .is_none_or(|upper| next_value_zero <= upper >> (self.height() - depth - 1))
                    && left_zero < right_zero
                {
                    next_stack.push(StackItem {
                        start: left_zero,
                        end: right_zero,
                        value: next_value_zero,
                    });
                }

                let left_one = zero + layer.rank(true, start)?;
                let right_one = zero + layer.rank(true, end)?;
                let next_value_one = (&value << 1) | NumberType::one();
                if lower
                    .is_none_or(|lower| (lower >> (self.height() - depth - 1)) <= next_value_one)
                    && upper.is_none_or(|upper| {
                        next_value_one <= (upper >> (self.height() - depth - 1))
                    })
                    && left_one < right_one
                {
                    next_stack.push(StackItem {
                        start: left_one,
                        end: right_one,
                        value: next_value_one,
                    });
                }
            }

            stack = next_stack;
        }

        let result = stack
            .into_iter()
            .filter(|StackItem { value, .. }| {
                lower.is_none_or(|lower| lower <= value) && upper.is_none_or(|upper| value < upper)
            })
            .map(|StackItem { start, end, value }| {
                Ok(HashMap::from([
                    (
                        "value".to_string(),
                        value
                            .to_biguint()
                            .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                    ),
                    (
                        "count".to_string(),
                        (end - start)
                            .to_biguint()
                            .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                    ),
                ]))
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(result)
    }

    /// Get values in [start, end) with the top-k maximum values.
    fn range_maxk(
        &self,
        start: usize,
        end: usize,
        k: Option<usize>,
    ) -> PyResult<Vec<HashMap<String, BigUint>>> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if k.is_some_and(|k| k.is_zero()) {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        let k = k.unwrap_or(end - start);

        struct StackItem<T> {
            start: usize,
            end: usize,
            value: T,
        }
        let mut stack = vec![StackItem {
            start,
            end,
            value: NumberType::zero(),
        }];

        for (layer, zero) in zip(self.get_layers(), self.get_zeros()) {
            let mut next_stack = Vec::new();

            for StackItem { start, end, value } in stack {
                let left_one = zero + layer.rank(true, start)?;
                let right_one = zero + layer.rank(true, end)?;
                let next_value_one = (&value << 1) | NumberType::one();
                if left_one < right_one {
                    next_stack.push(StackItem {
                        start: left_one,
                        end: right_one,
                        value: next_value_one,
                    });
                }

                let left_zero = layer.rank(false, start)?;
                let right_zero = layer.rank(false, end)?;
                let next_value_zero = &value << 1;
                if left_zero < right_zero {
                    next_stack.push(StackItem {
                        start: left_zero,
                        end: right_zero,
                        value: next_value_zero,
                    });
                }

                if next_stack.len() > k {
                    break;
                }
            }

            stack = next_stack;
        }

        let result = stack
            .into_iter()
            .map(|StackItem { start, end, value }| {
                Ok(HashMap::from([
                    (
                        "value".to_string(),
                        value
                            .to_biguint()
                            .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                    ),
                    (
                        "count".to_string(),
                        (end - start)
                            .to_biguint()
                            .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                    ),
                ]))
            })
            .take(k)
            .collect::<PyResult<Vec<_>>>()?;

        Ok(result)
    }

    /// Get values in [start, end) with the top-k minimum values.
    fn range_mink(
        &self,
        start: usize,
        end: usize,
        k: Option<usize>,
    ) -> PyResult<Vec<HashMap<String, BigUint>>> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if k.is_some_and(|k| k.is_zero()) {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        let k = k.unwrap_or(end - start);

        struct StackItem<T> {
            start: usize,
            end: usize,
            value: T,
        }
        let mut stack = vec![StackItem {
            start,
            end,
            value: NumberType::zero(),
        }];

        for (layer, zero) in zip(self.get_layers(), self.get_zeros()) {
            let mut next_stack = Vec::new();

            for StackItem { start, end, value } in stack {
                let left_zero = layer.rank(false, start)?;
                let right_zero = layer.rank(false, end)?;
                let next_value_zero = &value << 1;
                if left_zero < right_zero {
                    next_stack.push(StackItem {
                        start: left_zero,
                        end: right_zero,
                        value: next_value_zero,
                    });
                }

                let left_one = zero + layer.rank(true, start)?;
                let right_one = zero + layer.rank(true, end)?;
                let next_value_one = (&value << 1) | NumberType::one();
                if left_one < right_one {
                    next_stack.push(StackItem {
                        start: left_one,
                        end: right_one,
                        value: next_value_one,
                    });
                }

                if next_stack.len() > k {
                    break;
                }
            }

            stack = next_stack;
        }

        let result = stack
            .into_iter()
            .map(|StackItem { start, end, value }| {
                Ok(HashMap::from([
                    (
                        "value".to_string(),
                        value
                            .to_biguint()
                            .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                    ),
                    (
                        "count".to_string(),
                        (end - start)
                            .to_biguint()
                            .ok_or(PyRuntimeError::new_err("to_biguint failed"))?,
                    ),
                ]))
            })
            .take(k)
            .collect::<PyResult<Vec<_>>>()?;

        Ok(result)
    }

    /// Get the maximum value c in the range [start, end) such that lower <= c < upper.
    fn prev_value(
        &self,
        start: usize,
        end: usize,
        lower: Option<&NumberType>,
        upper: Option<&NumberType>,
    ) -> PyResult<Option<NumberType>> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if lower
            .zip(upper)
            .is_some_and(|(lower, upper)| lower >= upper)
        {
            return Err(PyValueError::new_err("lower must be less than upper"));
        }

        if self.height().is_zero() {
            if lower.is_none_or(|lower| lower <= &NumberType::zero())
                && upper.is_none_or(|upper| &NumberType::zero() < upper)
            {
                return Ok(Some(NumberType::zero()));
            } else {
                return Ok(None);
            }
        }

        struct StackItem<T> {
            start: usize,
            end: usize,
            value: T,
        }
        let mut stack = vec![StackItem {
            start,
            end,
            value: NumberType::zero(),
        }];

        for (depth, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let mut next_stack = Vec::new();

            for StackItem { start, end, value } in stack {
                let left_one = zero + layer.rank(true, start)?;
                let right_one = zero + layer.rank(true, end)?;
                let next_value_one = (&value << 1) | NumberType::one();
                if lower.is_none_or(|lower| lower >> (self.height() - depth - 1) <= next_value_one)
                    && upper
                        .is_none_or(|upper| next_value_one <= upper >> (self.height() - depth - 1))
                    && left_one < right_one
                {
                    next_stack.push(StackItem {
                        start: left_one,
                        end: right_one,
                        value: next_value_one,
                    });
                }

                let left_zero = layer.rank(false, start)?;
                let right_zero = layer.rank(false, end)?;
                let next_value_zero = &value << 1;
                if lower.is_none_or(|lower| lower >> (self.height() - depth - 1) <= next_value_zero)
                    && upper
                        .is_none_or(|upper| next_value_zero <= upper >> (self.height() - depth - 1))
                    && left_zero < right_zero
                {
                    next_stack.push(StackItem {
                        start: left_zero,
                        end: right_zero,
                        value: next_value_zero,
                    });
                }

                if depth + 1 == self.height() && next_stack.len() >= 1 {
                    let max_value = next_stack
                        .iter()
                        .filter(|item| {
                            lower.is_none_or(|lower| lower <= &item.value)
                                && upper.is_none_or(|upper| &item.value < upper)
                        })
                        .next();
                    match max_value {
                        Some(item) => return Ok(Some(item.value.clone())),
                        None => continue,
                    }
                }
            }

            stack = next_stack;
        }

        Ok(None)
    }

    /// Get the minimum value c in the range [start, end) such that lower <= c < upper.
    fn next_value(
        &self,
        start: usize,
        end: usize,
        lower: Option<&NumberType>,
        upper: Option<&NumberType>,
    ) -> PyResult<Option<NumberType>> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if lower
            .zip(upper)
            .is_some_and(|(lower, upper)| lower >= upper)
        {
            return Err(PyValueError::new_err("lower must be less than upper"));
        }

        if self.height().is_zero() {
            if lower.is_none_or(|lower| lower <= &NumberType::zero())
                && upper.is_none_or(|upper| &NumberType::zero() < upper)
            {
                return Ok(Some(NumberType::zero()));
            } else {
                return Ok(None);
            }
        }

        struct StackItem<T> {
            start: usize,
            end: usize,
            value: T,
        }
        let mut stack = vec![StackItem {
            start,
            end,
            value: NumberType::zero(),
        }];

        for (depth, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let mut next_stack = Vec::new();

            for StackItem { start, end, value } in stack {
                let left_zero = layer.rank(false, start)?;
                let right_zero = layer.rank(false, end)?;
                let next_value_zero = &value << 1;
                if lower.is_none_or(|lower| lower >> (self.height() - depth - 1) <= next_value_zero)
                    && upper
                        .is_none_or(|upper| next_value_zero <= upper >> (self.height() - depth - 1))
                    && left_zero < right_zero
                {
                    next_stack.push(StackItem {
                        start: left_zero,
                        end: right_zero,
                        value: next_value_zero,
                    });
                }

                let left_one = zero + layer.rank(true, start)?;
                let right_one = zero + layer.rank(true, end)?;
                let next_value_one = (&value << 1) | NumberType::one();
                if lower
                    .is_none_or(|lower| (lower >> (self.height() - depth - 1)) <= next_value_one)
                    && upper.is_none_or(|upper| {
                        next_value_one <= (upper >> (self.height() - depth - 1))
                    })
                    && left_one < right_one
                {
                    next_stack.push(StackItem {
                        start: left_one,
                        end: right_one,
                        value: next_value_one,
                    });
                }

                if depth + 1 == self.height() && next_stack.len() >= 1 {
                    let min_value = next_stack
                        .iter()
                        .filter(|item| {
                            lower.is_none_or(|lower| lower <= &item.value)
                                && upper.is_none_or(|upper| &item.value < upper)
                        })
                        .next();
                    match min_value {
                        Some(item) => return Ok(Some(item.value.clone())),
                        None => continue,
                    }
                }
            }

            stack = next_stack;
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::Python;
    use std::marker::PhantomData;

    use super::*;
    use crate::traits::{bit_vector::SampleBitVector, bit_width::BitWidth};

    struct SampleWaveletMatrix<NumberType> {
        layers: Vec<SampleBitVector>,
        zeros: Vec<usize>,
        height: usize,
        len: usize,
        phantom: PhantomData<NumberType>,
    }

    impl<NumberType> SampleWaveletMatrix<NumberType>
    where
        NumberType: BitAnd<NumberType, Output = NumberType> + BitWidth + Clone + One + Ord,
        for<'a> &'a NumberType: Shr<usize, Output = NumberType>,
    {
        fn new(data: &Vec<NumberType>) -> Self {
            let mut values = data.clone();
            let height = values.iter().max().map_or(0usize, |max| max.bit_width());
            let len = values.len();
            let mut layers: Vec<SampleBitVector> = Vec::with_capacity(height);
            let mut zeros: Vec<usize> = Vec::with_capacity(height);

            for i in 0..height {
                let mut bits = Vec::with_capacity(len);
                let mut zero_values = Vec::new();
                let mut one_values = Vec::new();
                for value in values.iter() {
                    let bit = (value >> (height - i - 1) & NumberType::one()).is_one();
                    bits.push(bit);
                    if bit {
                        one_values.push(value.clone());
                    } else {
                        zero_values.push(value.clone());
                    }
                }
                layers.push(SampleBitVector::new(bits));
                zeros.push(zero_values.len());
                values = [zero_values, one_values].concat();
            }

            SampleWaveletMatrix {
                layers,
                zeros,
                height,
                len,
                phantom: PhantomData,
            }
        }
    }

    impl<NumberType> WaveletMatrixTrait<NumberType, SampleBitVector> for SampleWaveletMatrix<NumberType>
    where
        NumberType: BitAnd<NumberType, Output = NumberType>
            + BitOr<NumberType, Output = NumberType>
            + BitOrAssign
            + Clone
            + One
            + Ord
            + PartialEq
            + Shl<usize, Output = NumberType>
            + ShlAssign<usize>
            + ToBigUint
            + Zero
            + 'static,
        for<'a> &'a NumberType: Shl<usize, Output = NumberType> + Shr<usize, Output = NumberType>,
    {
        fn get_layers(&self) -> &[SampleBitVector] {
            &self.layers
        }

        fn get_zeros(&self) -> &[usize] {
            &self.zeros
        }

        fn height(&self) -> usize {
            self.height
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    fn create_dummy_u8() -> SampleWaveletMatrix<u8> {
        let elements: Vec<u8> = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        SampleWaveletMatrix::new(&elements)
    }

    fn create_dummy_biguint() -> SampleWaveletMatrix<BigUint> {
        let elements: Vec<BigUint> = [5u32, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
            .into_iter()
            .map(BigUint::from)
            .collect();
        SampleWaveletMatrix::new(&elements)
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let wv_u8 = SampleWaveletMatrix::<u8>::new(&Vec::new());
        assert_eq!(wv_u8.len(), 0);
        assert_eq!(wv_u8.height(), 0);
        assert_eq!(
            wv_u8.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(wv_u8.rank(&0u8, 0).unwrap(), 0);
        assert_eq!(wv_u8.select(&0u8, 1).unwrap(), None);
        assert_eq!(
            wv_u8.quantile(0, 0, 1).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u8.topk(0, 0, Some(1)).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u8.range_sum(0, 0).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u8
                .range_intersection(0, 0, 0, 0)
                .unwrap_err()
                .to_string(),
            "ValueError: start1 must be less than end1"
        );
        assert_eq!(
            wv_u8.range_freq(0, 0, None, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u8.range_list(0, 0, None, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u8.range_maxk(0, 0, Some(1)).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u8.range_mink(0, 0, Some(1)).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u8.prev_value(0, 0, None, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u8.next_value(0, 0, None, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );

        let wv_biguint = SampleWaveletMatrix::<BigUint>::new(&Vec::new());
        assert_eq!(wv_biguint.len(), 0);
        assert_eq!(wv_biguint.height(), 0);
        assert_eq!(
            wv_biguint.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(wv_biguint.rank(&0u32.into(), 0).unwrap(), 0);
        assert_eq!(wv_biguint.select(&0u32.into(), 1).unwrap(), None);
        assert_eq!(
            wv_biguint.quantile(0, 0, 1).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_biguint.topk(0, 0, Some(1)).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_biguint.range_sum(0, 0).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_biguint
                .range_intersection(0, 0, 0, 0)
                .unwrap_err()
                .to_string(),
            "ValueError: start1 must be less than end1"
        );
        assert_eq!(
            wv_biguint
                .range_freq(0, 0, None, None)
                .unwrap_err()
                .to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_biguint
                .range_list(0, 0, None, None)
                .unwrap_err()
                .to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_biguint
                .range_maxk(0, 0, Some(1))
                .unwrap_err()
                .to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_biguint
                .range_mink(0, 0, Some(1))
                .unwrap_err()
                .to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_biguint
                .prev_value(0, 0, None, None)
                .unwrap_err()
                .to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_biguint
                .next_value(0, 0, None, None)
                .unwrap_err()
                .to_string(),
            "ValueError: start must be less than end"
        );
    }

    #[test]
    fn test_all_zero() {
        Python::initialize();

        let wv_u8 = SampleWaveletMatrix::<u8>::new(&vec![0u8; 64]);
        assert_eq!(wv_u8.len(), 64);
        assert_eq!(wv_u8.height(), 0);
        assert_eq!(wv_u8.access(1).unwrap(), 0u8);
        assert_eq!(wv_u8.rank(&0u8, 1).unwrap(), 1);
        assert_eq!(wv_u8.select(&0u8, 1).unwrap(), Some(0));
        assert_eq!(wv_u8.quantile(0, 64, 1).unwrap(), 0u8);
        assert_eq!(wv_u8.topk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u8.range_sum(0, 64).unwrap(), 0u32.into());
        assert_eq!(wv_u8.range_freq(0, 64, None, None).unwrap(), 64usize);
        assert_eq!(wv_u8.range_list(0, 64, None, None).unwrap().len(), 1);
        assert_eq!(wv_u8.range_maxk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u8.range_mink(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u8.prev_value(0, 64, None, None).unwrap(), Some(0u8));
        assert_eq!(wv_u8.next_value(0, 64, None, None).unwrap(), Some(0u8));

        let wv_biguint = SampleWaveletMatrix::<BigUint>::new(&vec![0u32.into(); 64]);
        assert_eq!(wv_biguint.len(), 64);
        assert_eq!(wv_biguint.height(), 0);
        assert_eq!(wv_biguint.access(1).unwrap(), 0u32.into());
        assert_eq!(wv_biguint.rank(&0u32.into(), 1).unwrap(), 1);
        assert_eq!(wv_biguint.select(&0u32.into(), 1).unwrap(), Some(0));
        assert_eq!(wv_biguint.quantile(0, 64, 1).unwrap(), 0u32.into());
        assert_eq!(wv_biguint.topk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_biguint.range_sum(0, 64).unwrap(), 0u32.into());
        assert_eq!(wv_biguint.range_freq(0, 64, None, None).unwrap(), 64usize);
        assert_eq!(wv_biguint.range_list(0, 64, None, None).unwrap().len(), 1);
        assert_eq!(wv_biguint.range_maxk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_biguint.range_mink(0, 64, None).unwrap().len(), 1);
        assert_eq!(
            wv_biguint.prev_value(0, 64, None, None).unwrap(),
            Some(0u32.into())
        );
        assert_eq!(
            wv_biguint.next_value(0, 64, None, None).unwrap(),
            Some(0u32.into())
        );
    }

    #[test]
    fn test_max_value() {
        Python::initialize();

        let wv_u8 = SampleWaveletMatrix::<u8>::new(&vec![u8::MAX; 64]);
        assert_eq!(wv_u8.len(), 64);
        assert_eq!(wv_u8.height(), 8);
        assert_eq!(wv_u8.access(1).unwrap(), u8::MAX);
        assert_eq!(wv_u8.rank(&u8::MAX, 1).unwrap(), 1);
        assert_eq!(wv_u8.select(&u8::MAX, 1).unwrap(), Some(0));
        assert_eq!(wv_u8.quantile(0, 64, 1).unwrap(), u8::MAX);
        assert_eq!(wv_u8.topk(0, 64, None).unwrap().len(), 1);
        assert_eq!(
            wv_u8.range_sum(0, 64).unwrap(),
            (u8::MAX as u32 * 64).into()
        );
        assert_eq!(wv_u8.range_freq(0, 64, None, None).unwrap(), 64usize);
        assert_eq!(wv_u8.range_list(0, 64, None, None).unwrap().len(), 1);
        assert_eq!(wv_u8.range_maxk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u8.range_mink(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u8.prev_value(0, 64, None, None).unwrap(), Some(u8::MAX));
        assert_eq!(wv_u8.next_value(0, 64, None, None).unwrap(), Some(u8::MAX));
    }

    #[test]
    fn test_access() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(wv_u8.access(6).unwrap(), 5u8);

        let wv_biguint = create_dummy_biguint();
        assert_eq!(wv_biguint.access(6).unwrap(), 5u32.into());
    }

    #[test]
    fn test_rank() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(wv_u8.rank(&5u8, 9).unwrap(), 4usize);

        let wv_biguint = create_dummy_biguint();
        assert_eq!(wv_biguint.rank(&5u32.into(), 9).unwrap(), 4usize);
    }

    #[test]
    fn test_select() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(wv_u8.select(&5u8, 4).unwrap(), Some(6usize));
        assert_eq!(wv_u8.select(&5u8, 6).unwrap(), None);

        let wv_biguint = create_dummy_biguint();
        assert_eq!(wv_biguint.select(&5u32.into(), 4).unwrap(), Some(6usize));
        assert_eq!(wv_biguint.select(&5u32.into(), 6).unwrap(), None);
    }

    #[test]
    fn test_quantile() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(wv_u8.quantile(2, 12, 8).unwrap(), 5u8);

        let wv_biguint = create_dummy_biguint();
        assert_eq!(wv_biguint.quantile(2, 12, 8).unwrap(), 5u32.into());
    }

    #[test]
    fn test_topk() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        let result_u8 = wv_u8.topk(1, 10, Some(2)).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(5u8)));
        assert_eq!(result_u8[0].get("count"), Some(&BigUint::from(3usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(1u8)));
        assert_eq!(result_u8[1].get("count"), Some(&BigUint::from(2usize)));

        let wv_biguint = create_dummy_biguint();
        let result_biguint = wv_biguint.topk(1, 10, Some(2)).unwrap();
        assert_eq!(result_biguint.len(), 2);
        assert_eq!(result_biguint[0].get("value"), Some(&BigUint::from(5u32)));
        assert_eq!(result_biguint[0].get("count"), Some(&BigUint::from(3usize)));
        assert_eq!(result_biguint[1].get("value"), Some(&BigUint::from(1u32)));
        assert_eq!(result_biguint[1].get("count"), Some(&BigUint::from(2usize)));
    }

    #[test]
    fn test_range_sum() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(wv_u8.range_sum(2, 8).unwrap(), 24u32.into());

        let wv_biguint = create_dummy_biguint();
        assert_eq!(wv_biguint.range_sum(2, 8).unwrap(), 24u32.into());
    }

    #[test]
    fn test_range_intersection() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        let result_u8 = wv_u8.range_intersection(0, 6, 6, 11).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(1u8)));
        assert_eq!(result_u8[0].get("count1"), Some(&BigUint::from(1usize)));
        assert_eq!(result_u8[0].get("count2"), Some(&BigUint::from(1usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(5u8)));
        assert_eq!(result_u8[1].get("count1"), Some(&BigUint::from(3usize)));
        assert_eq!(result_u8[1].get("count2"), Some(&BigUint::from(2usize)));

        let wv_biguint = create_dummy_biguint();
        let result_biguint = wv_biguint.range_intersection(0, 6, 6, 11).unwrap();
        assert_eq!(result_biguint.len(), 2);
        assert_eq!(result_biguint[0].get("value"), Some(&BigUint::from(1u32)));
        assert_eq!(
            result_biguint[0].get("count1"),
            Some(&BigUint::from(1usize))
        );
        assert_eq!(
            result_biguint[0].get("count2"),
            Some(&BigUint::from(1usize))
        );
        assert_eq!(result_biguint[1].get("value"), Some(&BigUint::from(5u32)));
        assert_eq!(
            result_biguint[1].get("count1"),
            Some(&BigUint::from(3usize))
        );
        assert_eq!(
            result_biguint[1].get("count2"),
            Some(&BigUint::from(2usize))
        );
    }

    #[test]
    fn test_range_freq() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(
            wv_u8.range_freq(1, 9, Some(&4u8), Some(&6u8)).unwrap(),
            4usize
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint
                .range_freq(1, 9, Some(&4u32.into()), Some(&6u32.into()))
                .unwrap(),
            4usize,
        );
    }

    #[test]
    fn test_range_list() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        let result_u8 = wv_u8.range_list(1, 9, Some(&4u8), Some(&6u8)).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(4u8)));
        assert_eq!(result_u8[0].get("count"), Some(&BigUint::from(1usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(5u8)));
        assert_eq!(result_u8[1].get("count"), Some(&BigUint::from(3usize)));

        let wv_biguint = create_dummy_biguint();
        let result_biguint = wv_biguint
            .range_list(1, 9, Some(&4u32.into()), Some(&6u32.into()))
            .unwrap();
        assert_eq!(result_biguint.len(), 2);
        assert_eq!(result_biguint[0].get("value"), Some(&BigUint::from(4u32)));
        assert_eq!(result_biguint[0].get("count"), Some(&BigUint::from(1usize)));
        assert_eq!(result_biguint[1].get("value"), Some(&BigUint::from(5u32)));
        assert_eq!(result_biguint[1].get("count"), Some(&BigUint::from(3usize)));
    }

    #[test]
    fn test_range_maxk() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        let result_u8 = wv_u8.range_maxk(1, 9, Some(2)).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(6u8)));
        assert_eq!(result_u8[0].get("count"), Some(&BigUint::from(1usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(5u8)));
        assert_eq!(result_u8[1].get("count"), Some(&BigUint::from(3usize)));

        let wv_biguint = create_dummy_biguint();
        let result_biguint = wv_biguint.range_maxk(1, 9, Some(2)).unwrap();
        assert_eq!(result_biguint.len(), 2);
        assert_eq!(result_biguint[0].get("value"), Some(&BigUint::from(6u32)));
        assert_eq!(result_biguint[0].get("count"), Some(&BigUint::from(1usize)));
        assert_eq!(result_biguint[1].get("value"), Some(&BigUint::from(5u32)));
        assert_eq!(result_biguint[1].get("count"), Some(&BigUint::from(3usize)));
    }

    #[test]
    fn test_range_mink() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        let result_u8 = wv_u8.range_mink(1, 9, Some(2)).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(1u8)));
        assert_eq!(result_u8[0].get("count"), Some(&BigUint::from(2usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(2u8)));
        assert_eq!(result_u8[1].get("count"), Some(&BigUint::from(1usize)));

        let wv_biguint = create_dummy_biguint();
        let result_biguint = wv_biguint.range_mink(1, 9, Some(2)).unwrap();
        assert_eq!(result_biguint.len(), 2);
        assert_eq!(result_biguint[0].get("value"), Some(&BigUint::from(1u32)));
        assert_eq!(result_biguint[0].get("count"), Some(&BigUint::from(2usize)));
        assert_eq!(result_biguint[1].get("value"), Some(&BigUint::from(2u32)));
        assert_eq!(result_biguint[1].get("count"), Some(&BigUint::from(1usize)));
    }

    #[test]
    fn test_prev_value() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(
            wv_u8.prev_value(1, 9, Some(&4u8), Some(&7u8)).unwrap(),
            Some(6u8),
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint
                .prev_value(1, 9, Some(&4u32.into()), Some(&7u32.into()))
                .unwrap(),
            Some(6u32.into()),
        );
    }

    #[test]
    fn test_next_value() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(
            wv_u8.next_value(1, 9, Some(&3u8), Some(&5u8)).unwrap(),
            Some(4u8),
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint
                .next_value(1, 9, Some(&3u32.into()), Some(&5u32.into()))
                .unwrap(),
            Some(4u32.into()),
        );
    }
}
