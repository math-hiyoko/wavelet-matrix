use crate::traits::{bit_vector::bit_vector::BitVectorTrait, utils::bit_width::BitWidth};
use num_bigint::{BigUint, ToBigUint};
use num_traits::{One, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
};
use std::{
    cmp::PartialEq,
    collections::BinaryHeap,
    iter::{once, zip},
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
        + BitWidth
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
        for (depth, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let bit = (value >> (self.height() - depth - 1) & NumberType::one()).is_one();
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

    /// Get all values in the Wavelet Matrix as a vector.
    fn values(&self) -> PyResult<Vec<NumberType>> {
        let mut indices = (0..self.len()).collect::<Vec<usize>>();
        let mut values = vec![NumberType::zero(); self.len()];
        for (depth, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let bits = layer.values()?;
            let rank = once([0usize; 2])
                .chain(bits.iter().scan([0usize; 2], |acc, &bit| {
                    acc[bit as usize] += 1;
                    Some(*acc)
                }))
                .collect::<Vec<_>>();
            for (index, value) in zip(indices.iter_mut(), values.iter_mut()) {
                let bit = bits[*index];
                if bit {
                    *value |= NumberType::one() << (self.height() - depth - 1);
                    *index = zero + rank[*index][bit as usize];
                } else {
                    *index = rank[*index][bit as usize];
                }
                debug_assert!(*index <= self.len());
            }
        }
        Ok(values)
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
            debug_assert!(index <= self.len());
        }

        Ok(result)
    }

    /// Count the number of occurrences of a value in the range [0, end).
    fn rank(&self, value: &NumberType, mut end: usize) -> PyResult<usize> {
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if value.bit_width() > self.height() {
            return Ok(0usize);
        }

        let begin_index = match self.begin_index(value) {
            Some(index) => index,
            None => return Ok(0usize),
        };

        for (depth, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let bit = (value >> (self.height() - depth - 1) & NumberType::one()).is_one();
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
        if value.bit_width() > self.height() {
            return Ok(None);
        }

        let begin_index = match self.begin_index(value) {
            Some(index) => index,
            None => return Ok(None),
        };

        let mut index = begin_index + kth - 1;
        for (depth, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate().rev() {
            let bit = (value >> (self.height() - depth - 1) & NumberType::one()).is_one();
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
    fn quantile(&self, mut start: usize, mut end: usize, mut kth: usize) -> PyResult<NumberType> {
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

        let mut result = NumberType::zero();
        for (layer, zero) in zip(self.get_layers(), self.get_zeros()) {
            let count_zeros = layer.rank(false, end)? - layer.rank(false, start)?;
            let bit = if kth <= count_zeros {
                false
            } else {
                kth -= count_zeros;
                true
            };

            result <<= 1;
            if bit {
                result |= NumberType::one();
                start = zero + layer.rank(bit, start)?;
                end = zero + layer.rank(bit, end)?;
            } else {
                start = layer.rank(bit, start)?;
                end = layer.rank(bit, end)?;
            }
            debug_assert!(start < end && end <= self.len());
        }

        Ok(result)
    }

    // Count values in [start, end) with the top-k highest frequencies.
    fn topk(
        &self,
        start: usize,
        end: usize,
        k: Option<usize>,
    ) -> PyResult<Vec<(NumberType, usize)>> {
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
                result.push((value, len));
                if result.len() == k {
                    break;
                }
                continue;
            }

            let layer = &self.get_layers()[depth];
            let zero = self.get_zeros()[depth];

            let start_zero = layer.rank(false, start)?;
            let end_zero = layer.rank(false, end)?;
            debug_assert!(start_zero <= end_zero);
            if start_zero != end_zero {
                heap.push(QueueItem {
                    len: end_zero - start_zero,
                    depth: depth + 1,
                    start: start_zero,
                    end: end_zero,
                    value: &value << 1usize,
                });
            }

            let start_one = zero + layer.rank(true, start)?;
            let end_one = zero + layer.rank(true, end)?;
            debug_assert!(start_one <= end_one);
            if end_one != start_one {
                heap.push(QueueItem {
                    len: end_one - start_one,
                    depth: depth + 1,
                    start: start_one,
                    end: end_one,
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
            |acc, (value, count)| -> PyResult<BigUint> {
                let value = value
                    .to_biguint()
                    .ok_or(PyRuntimeError::new_err("failed to convert to BigUint"))?;
                let count = count
                    .to_biguint()
                    .ok_or(PyRuntimeError::new_err("failed to convert to BigUint"))?;

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
    ) -> PyResult<Vec<(NumberType, usize, usize)>> {
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
                let start1_zero = layer.rank(false, start1)?;
                let end1_zero = layer.rank(false, end1)?;
                debug_assert!(start1_zero <= end1_zero);
                let start2_zero = layer.rank(false, start2)?;
                let end2_zero = layer.rank(false, end2)?;
                debug_assert!(start2_zero <= end2_zero);
                if start1_zero != end1_zero && start2_zero != end2_zero {
                    next_stack.push(StackItem {
                        start1: start1_zero,
                        end1: end1_zero,
                        start2: start2_zero,
                        end2: end2_zero,
                        value: &value << 1,
                    });
                }

                let start1_one = zero + layer.rank(true, start1)?;
                let end1_one = zero + layer.rank(true, end1)?;
                debug_assert!(start1_one <= end1_one);
                let start2_one = zero + layer.rank(true, start2)?;
                let end2_one = zero + layer.rank(true, end2)?;
                debug_assert!(start2_one <= end2_one);
                if start1_one != end1_one && start2_one != end2_one {
                    next_stack.push(StackItem {
                        start1: start1_one,
                        end1: end1_one,
                        start2: start2_one,
                        end2: end2_one,
                        value: (&value << 1) | NumberType::one(),
                    });
                }
            }

            stack = next_stack;
        }

        let result = stack
            .iter()
            .map(
                |StackItem {
                     start1,
                     end1,
                     start2,
                     end2,
                     value,
                 }| (value.clone(), end1 - start1, end2 - start2),
            )
            .collect::<Vec<_>>();
        Ok(result)
    }

    /// Get the total count of values c in the range [start, end) such that c < upper.
    #[inline]
    fn range_freq_less(
        &self,
        mut start: usize,
        mut end: usize,
        upper: &NumberType,
    ) -> PyResult<usize> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if upper.bit_width() > self.height() {
            return Ok(end - start);
        }

        let mut count = 0usize;
        for (depth, (layer, zero)) in zip(self.get_layers(), self.get_zeros()).enumerate() {
            let bit = (upper >> (self.height() - depth - 1) & NumberType::one()).is_one();
            if bit {
                count += layer.rank(false, end)? - layer.rank(false, start)?;
                start = zero + layer.rank(bit, start)?;
                end = zero + layer.rank(bit, end)?;
            } else {
                start = layer.rank(bit, start)?;
                end = layer.rank(bit, end)?;
            }
        }

        Ok(count)
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

        let upper_count = match upper {
            Some(upper) => self.range_freq_less(start, end, upper)?,
            None => end - start,
        };
        let lower_count = match lower {
            Some(lower) => self.range_freq_less(start, end, lower)?,
            None => 0usize,
        };
        Ok(upper_count - lower_count)
    }

    /// Get a list of values c in the range [start, end) such that lower <= c < upper.
    fn range_list(
        &self,
        start: usize,
        end: usize,
        lower: Option<&NumberType>,
        upper: Option<&NumberType>,
    ) -> PyResult<Vec<(NumberType, usize)>> {
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
                let start_zero = layer.rank(false, start)?;
                let end_zero = layer.rank(false, end)?;
                debug_assert!(start_zero <= end_zero);
                let next_value_zero = &value << 1;
                if start_zero != end_zero
                    && lower
                        .is_none_or(|lower| lower >> (self.height() - depth - 1) <= next_value_zero)
                    && upper
                        .is_none_or(|upper| next_value_zero <= upper >> (self.height() - depth - 1))
                {
                    next_stack.push(StackItem {
                        start: start_zero,
                        end: end_zero,
                        value: next_value_zero,
                    });
                }

                let start_one = zero + layer.rank(true, start)?;
                let end_one = zero + layer.rank(true, end)?;
                debug_assert!(start_one <= end_one);
                let next_value_one = (&value << 1) | NumberType::one();
                if start_one != end_one
                    && lower
                        .is_none_or(|lower| lower >> (self.height() - depth - 1) <= next_value_one)
                    && upper
                        .is_none_or(|upper| next_value_one <= upper >> (self.height() - depth - 1))
                {
                    next_stack.push(StackItem {
                        start: start_one,
                        end: end_one,
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
            .map(|StackItem { start, end, value }| (value, end - start))
            .collect::<Vec<_>>();

        Ok(result)
    }

    /// Get values in [start, end) with the top-k maximum values.
    fn range_maxk(
        &self,
        start: usize,
        end: usize,
        k: Option<usize>,
    ) -> PyResult<Vec<(NumberType, usize)>> {
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
                let start_one = zero + layer.rank(true, start)?;
                let end_one = zero + layer.rank(true, end)?;
                debug_assert!(start_one <= end_one);
                let next_value_one = (&value << 1) | NumberType::one();
                if start_one != end_one {
                    next_stack.push(StackItem {
                        start: start_one,
                        end: end_one,
                        value: next_value_one,
                    });
                }

                if next_stack.len() >= k {
                    break;
                }

                let start_zero = layer.rank(false, start)?;
                let end_zero = layer.rank(false, end)?;
                debug_assert!(start_zero <= end_zero);
                let next_value_zero = &value << 1;
                if start_zero != end_zero {
                    next_stack.push(StackItem {
                        start: start_zero,
                        end: end_zero,
                        value: next_value_zero,
                    });
                }

                if next_stack.len() >= k {
                    break;
                }
            }

            stack = next_stack;
        }

        let result = stack
            .into_iter()
            .map(|StackItem { start, end, value }| (value, end - start))
            .take(k)
            .collect::<Vec<_>>();

        Ok(result)
    }

    /// Get values in [start, end) with the top-k minimum values.
    fn range_mink(
        &self,
        start: usize,
        end: usize,
        k: Option<usize>,
    ) -> PyResult<Vec<(NumberType, usize)>> {
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
                let start_zero = layer.rank(false, start)?;
                let end_zero = layer.rank(false, end)?;
                debug_assert!(start_zero <= end_zero);
                let next_value_zero = &value << 1;
                if start_zero != end_zero {
                    next_stack.push(StackItem {
                        start: start_zero,
                        end: end_zero,
                        value: next_value_zero,
                    });
                }

                if next_stack.len() >= k {
                    break;
                }

                let start_one = zero + layer.rank(true, start)?;
                let end_one = zero + layer.rank(true, end)?;
                debug_assert!(start_one <= end_one);
                let next_value_one = (&value << 1) | NumberType::one();
                if start_one != end_one {
                    next_stack.push(StackItem {
                        start: start_one,
                        end: end_one,
                        value: next_value_one,
                    });
                }

                if next_stack.len() >= k {
                    break;
                }
            }

            stack = next_stack;
        }

        let result = stack
            .into_iter()
            .map(|StackItem { start, end, value }| (value, end - start))
            .take(k)
            .collect::<Vec<_>>();

        Ok(result)
    }

    /// Get the maximum value c in the range [start, end) such that c < upper.
    fn prev_value(
        &self,
        start: usize,
        end: usize,
        upper: Option<&NumberType>,
    ) -> PyResult<Option<NumberType>> {
        let count = self.range_freq(start, end, None, upper)?;
        if count.is_zero() {
            return Ok(None);
        }

        let value = self.quantile(start, end, count)?;
        Ok(Some(value))
    }

    /// Get the minimum value c in the range [start, end) such that lower <= c.
    fn next_value(
        &self,
        start: usize,
        end: usize,
        lower: Option<&NumberType>,
    ) -> PyResult<Option<NumberType>> {
        let count = self.range_freq(start, end, lower, None)?;
        if count.is_zero() {
            return Ok(None);
        }

        let value = self.quantile(start, end, end - start - count + 1)?;
        Ok(Some(value))
    }
}

#[cfg(test)]
mod tests {
    use pyo3::Python;
    use std::marker::PhantomData;

    use super::*;
    use crate::traits::{bit_vector::bit_vector::SampleBitVector, utils::bit_width::BitWidth};

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
            + BitWidth
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
        assert_eq!(wv_u8.values().unwrap(), Vec::<u8>::new());
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
            wv_u8.prev_value(0, 0, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u8.next_value(0, 0, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );

        let wv_biguint = SampleWaveletMatrix::<BigUint>::new(&Vec::new());
        assert_eq!(wv_biguint.len(), 0);
        assert_eq!(wv_biguint.height(), 0);
        assert_eq!(wv_biguint.values().unwrap(), Vec::<BigUint>::new());
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
            wv_biguint.prev_value(0, 0, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_biguint.next_value(0, 0, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
    }

    #[test]
    fn test_all_zero() {
        Python::initialize();

        let wv_u8 = SampleWaveletMatrix::<u8>::new(&vec![0u8; 64]);
        assert_eq!(wv_u8.len(), 64);
        assert_eq!(wv_u8.height(), 0);
        assert_eq!(wv_u8.values().unwrap(), vec![0u8; 64]);
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
        assert_eq!(wv_u8.prev_value(0, 64, None).unwrap(), Some(0u8));
        assert_eq!(wv_u8.next_value(0, 64, None).unwrap(), Some(0u8));

        let wv_biguint = SampleWaveletMatrix::<BigUint>::new(&vec![0u32.into(); 64]);
        assert_eq!(wv_biguint.len(), 64);
        assert_eq!(wv_biguint.height(), 0);
        assert_eq!(wv_biguint.values().unwrap(), vec![0u32.into(); 64]);
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
            wv_biguint.prev_value(0, 64, None).unwrap(),
            Some(0u32.into())
        );
        assert_eq!(
            wv_biguint.next_value(0, 64, None).unwrap(),
            Some(0u32.into())
        );
    }

    #[test]
    fn test_max_value() {
        Python::initialize();

        let wv_u8 = SampleWaveletMatrix::<u8>::new(&vec![u8::MAX; 64]);
        assert_eq!(wv_u8.len(), 64);
        assert_eq!(wv_u8.height(), 8);
        assert_eq!(wv_u8.values().unwrap(), vec![u8::MAX; 64]);
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
        assert_eq!(wv_u8.prev_value(0, 64, None).unwrap(), Some(u8::MAX));
        assert_eq!(wv_u8.next_value(0, 64, None).unwrap(), Some(u8::MAX));
    }

    #[test]
    fn test_values() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(
            wv_u8.values().unwrap(),
            vec![5u8, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0],
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint.values().unwrap(),
            vec![
                5u32.into(),
                4u32.into(),
                5u32.into(),
                5u32.into(),
                2u32.into(),
                1u32.into(),
                5u32.into(),
                6u32.into(),
                1u32.into(),
                3u32.into(),
                5u32.into(),
                0u32.into()
            ],
        );
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
        assert_eq!(
            wv_u8.topk(1, 10, Some(2)).unwrap(),
            vec![(5u8, 3usize), (1u8, 2usize)],
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint.topk(1, 10, Some(2)).unwrap(),
            vec![(5u32.into(), 3usize), (1u32.into(), 2usize)],
        );
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
        assert_eq!(
            wv_u8.range_intersection(0, 6, 6, 11).unwrap(),
            vec![(1u8, 1usize, 1usize), (5u8, 3usize, 2usize),],
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint.range_intersection(0, 6, 6, 11).unwrap(),
            vec![(1u32.into(), 1usize, 1usize), (5u32.into(), 3usize, 2usize),],
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
        assert_eq!(
            wv_u8.range_list(1, 9, Some(&4u8), Some(&6u8)).unwrap(),
            vec![(4u8, 1usize), (5u8, 3usize),],
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint
                .range_list(1, 9, Some(&4u32.into()), Some(&6u32.into()))
                .unwrap(),
            vec![(4u32.into(), 1usize), (5u32.into(), 3usize),],
        );
    }

    #[test]
    fn test_range_maxk() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(
            wv_u8.range_maxk(1, 9, Some(2)).unwrap(),
            vec![(6u8, 1usize), (5u8, 3usize),],
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint.range_maxk(1, 9, Some(2)).unwrap(),
            vec![(6u32.into(), 1usize), (5u32.into(), 3usize),],
        );
    }

    #[test]
    fn test_range_mink() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(
            wv_u8.range_mink(1, 9, Some(2)).unwrap(),
            vec![(1u8, 2usize), (2u8, 1usize),],
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint.range_mink(1, 9, Some(2)).unwrap(),
            vec![(1u32.into(), 2usize), (2u32.into(), 1usize),],
        );
    }

    #[test]
    fn test_prev_value() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(wv_u8.prev_value(1, 9, Some(&7u8)).unwrap(), Some(6u8),);

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint.prev_value(1, 9, Some(&7u32.into())).unwrap(),
            Some(6u32.into()),
        );
    }

    #[test]
    fn test_next_value() {
        Python::initialize();

        let wv_u8 = create_dummy_u8();
        assert_eq!(wv_u8.next_value(1, 9, Some(&3u8)).unwrap(), Some(4u8),);

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint.next_value(1, 9, Some(&3u32.into())).unwrap(),
            Some(4u32.into()),
        );
    }
}
