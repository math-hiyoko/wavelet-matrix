use super::wavelet_matrix::WaveletMatrixTrait;
use crate::traits::{
    bit_vector::dynamic_bit_vector::DynamicBitVectorTrait, utils::bit_width::BitWidth,
};
use num_bigint::ToBigUint;
use num_traits::{One, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyValueError},
};
use std::{
    cmp::PartialEq,
    iter::zip,
    ops::{BitAnd, BitOr, BitOrAssign, Shl, ShlAssign, Shr},
};

pub(crate) trait DynamicWaveletMatrixTrait<NumberType, BitVectorType>:
    WaveletMatrixTrait<NumberType, BitVectorType>
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
    BitVectorType: DynamicBitVectorTrait,
{
    fn len(&mut self) -> &mut usize;

    fn get_layers_and_zeros(&mut self) -> (&mut [BitVectorType], &mut [usize]);

    /// Inserts a value at the specified index.
    fn insert(&mut self, mut index: usize, value: &NumberType) -> PyResult<()> {
        if index > *self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if value.bit_width() > self.height() {
            return Err(PyValueError::new_err("value exceeds the maximum value"));
        }
        *self.len() += 1;

        let height = self.height();
        let (layers, zeros) = self.get_layers_and_zeros();
        for (i, (layer, zero)) in zip(layers, zeros).enumerate() {
            let bit = (value >> (height - i - 1) & NumberType::one()).is_one();
            layer.insert(index, bit)?;
            if bit {
                index = *zero + layer.rank(bit, index)?;
            } else {
                index = layer.rank(bit, index)?;
                *zero += 1;
            }
        }

        Ok(())
    }

    /// Removes a value at the specified index.
    fn remove(&mut self, mut index: usize) -> PyResult<NumberType> {
        if index >= *self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        *self.len() -= 1;

        let (layers, zeros) = self.get_layers_and_zeros();
        let mut result = NumberType::zero();
        for (layer, zero) in zip(layers, zeros) {
            let bit = layer.remove(index)?;
            result <<= 1;
            if bit {
                index = *zero + layer.rank(bit, index)?;
                result |= NumberType::one();
            } else {
                debug_assert!(*zero > 0);
                index = layer.rank(bit, index)?;
                *zero -= 1;
            }
        }

        Ok(result)
    }

    /// Updates the value at the specified index.
    fn update(&mut self, index: usize, value: &NumberType) -> PyResult<NumberType> {
        if index >= *self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if value.bit_width() > self.height() {
            return Err(PyValueError::new_err("value exceeds the maximum value"));
        }
        let removed_value = self.remove(index)?;
        self.insert(index, value)?;

        Ok(removed_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{
        bit_vector::dynamic_bit_vector::SampleDynamicBitVector, utils::bit_width::BitWidth,
    };
    use num_bigint::BigUint;
    use pyo3::Python;
    use std::marker::PhantomData;

    struct SampleDynamicWaveletMatrix<NumberType> {
        layers: Vec<SampleDynamicBitVector>,
        zeros: Vec<usize>,
        height: usize,
        len: usize,
        phantom: PhantomData<NumberType>,
    }

    impl<NumberType> SampleDynamicWaveletMatrix<NumberType>
    where
        NumberType: BitAnd<NumberType, Output = NumberType> + BitWidth + Clone + One + Ord,
        for<'a> &'a NumberType: Shr<usize, Output = NumberType>,
    {
        fn new(data: &Vec<NumberType>, max_bit: Option<usize>) -> PyResult<Self> {
            let mut values = data.clone();
            let max_width = values.iter().max().map_or(0usize, |max| max.bit_width());
            if max_bit.is_some_and(|max_bit| max_bit < max_width) {
                return Err(PyValueError::new_err(
                    "max_bit is less than the maximum bit width of the data",
                ));
            }
            let height = max_bit.unwrap_or(max_width);
            let len = values.len();
            let mut layers: Vec<SampleDynamicBitVector> = Vec::with_capacity(height);
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
                layers.push(SampleDynamicBitVector::new(&bits));
                zeros.push(zero_values.len());
                values = [zero_values, one_values].concat();
            }

            Ok(SampleDynamicWaveletMatrix {
                layers,
                zeros,
                height,
                len,
                phantom: PhantomData,
            })
        }
    }

    impl<NumberType> WaveletMatrixTrait<NumberType, SampleDynamicBitVector>
        for SampleDynamicWaveletMatrix<NumberType>
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
        fn get_layers(&self) -> &[SampleDynamicBitVector] {
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

    impl<NumberType> DynamicWaveletMatrixTrait<NumberType, SampleDynamicBitVector>
        for SampleDynamicWaveletMatrix<NumberType>
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
        fn len(&mut self) -> &mut usize {
            &mut self.len
        }

        fn get_layers_and_zeros(&mut self) -> (&mut [SampleDynamicBitVector], &mut [usize]) {
            (&mut self.layers, &mut self.zeros)
        }
    }

    fn create_dummy_u8() -> SampleDynamicWaveletMatrix<u8> {
        let elements: Vec<u8> = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        SampleDynamicWaveletMatrix::new(&elements, None).unwrap()
    }

    fn create_dummy_biguint() -> SampleDynamicWaveletMatrix<BigUint> {
        let elements: Vec<BigUint> = [5u32, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
            .into_iter()
            .map(BigUint::from)
            .collect();
        SampleDynamicWaveletMatrix::new(&elements, None).unwrap()
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let wv_u8 = SampleDynamicWaveletMatrix::<u8>::new(&Vec::new(), None).unwrap();
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

        let wv_biguint = SampleDynamicWaveletMatrix::<BigUint>::new(&Vec::new(), None).unwrap();
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

        let wv_u8 = SampleDynamicWaveletMatrix::<u8>::new(&vec![0u8; 64], None).unwrap();
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

        let wv_biguint =
            SampleDynamicWaveletMatrix::<BigUint>::new(&vec![0u32.into(); 64], None).unwrap();
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

        let wv_u8 = SampleDynamicWaveletMatrix::<u8>::new(&vec![u8::MAX; 64], None).unwrap();
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
            vec![5u8, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
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
            ]
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
            vec![(5u8, 3usize), (1u8, 2usize),],
        );

        let wv_biguint = create_dummy_biguint();
        assert_eq!(
            wv_biguint.topk(1, 10, Some(2)).unwrap(),
            vec![(5u32.into(), 3usize), (1u32.into(), 2usize),],
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

    #[test]
    fn test_insert() {
        Python::initialize();

        let mut wv_u8 = create_dummy_u8();
        wv_u8.insert(4, &5u8).unwrap();
        assert_eq!(
            wv_u8.insert(4, &8u8).unwrap_err().to_string(),
            "ValueError: value exceeds the maximum value"
        );
        assert_eq!(wv_u8.access(4).unwrap(), 5u8);
        assert_eq!(wv_u8.len(), 13);

        let mut wv_biguint = create_dummy_biguint();
        wv_biguint.insert(4, &5u32.into()).unwrap();
        assert_eq!(
            wv_biguint.insert(4, &8u32.into()).unwrap_err().to_string(),
            "ValueError: value exceeds the maximum value"
        );
        assert_eq!(wv_biguint.access(4).unwrap(), 5u32.into());
        assert_eq!(wv_biguint.len(), 13);
    }

    #[test]
    fn test_remove() {
        Python::initialize();

        let mut wv_u8 = create_dummy_u8();
        wv_u8.remove(4).unwrap();
        assert_eq!(wv_u8.access(4).unwrap(), 1u8);
        assert_eq!(wv_u8.len(), 11);

        let mut wv_biguint = create_dummy_biguint();
        wv_biguint.remove(4).unwrap();
        assert_eq!(wv_biguint.access(4).unwrap(), 1u32.into());
        assert_eq!(wv_biguint.len(), 11);
    }

    #[test]
    fn test_update() {
        Python::initialize();

        let mut wv_u8 = create_dummy_u8();
        wv_u8.update(4, &5u8).unwrap();
        assert_eq!(
            wv_u8.update(4, &8u8).unwrap_err().to_string(),
            "ValueError: value exceeds the maximum value"
        );
        assert_eq!(wv_u8.access(4).unwrap(), 5u8);
        assert_eq!(wv_u8.len(), 12);

        let mut wv_biguint = create_dummy_biguint();
        wv_biguint.update(4, &5u32.into()).unwrap();
        assert_eq!(
            wv_biguint.update(4, &8u32.into()).unwrap_err().to_string(),
            "ValueError: value exceeds the maximum value"
        );
        assert_eq!(wv_biguint.access(4).unwrap(), 5u32.into());
        assert_eq!(wv_biguint.len(), 12);
    }

    #[test]
    fn test_insert_remove_values() {
        Python::initialize();

        let mut wv_u8 = SampleDynamicWaveletMatrix::new(&vec![], Some(3)).unwrap();
        let elements: Vec<u8> = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];

        for (index, &element) in elements.iter().enumerate() {
            wv_u8.insert(index, &element).unwrap();
            assert_eq!(wv_u8.access(index).unwrap(), element);
        }
        assert_eq!(wv_u8.len(), elements.len());

        for &element in &elements {
            assert_eq!(wv_u8.remove(0).unwrap(), element);
        }
        assert_eq!(wv_u8.len(), 0);

        for &element in elements.iter().rev() {
            wv_u8.insert(0, &element).unwrap();
            assert_eq!(wv_u8.access(0).unwrap(), element);
        }
        assert_eq!(wv_u8.len(), elements.len());

        for (index, &element) in elements.iter().enumerate().rev() {
            assert_eq!(wv_u8.remove(index).unwrap(), element);
        }
        assert_eq!(wv_u8.len(), 0);
    }
}
