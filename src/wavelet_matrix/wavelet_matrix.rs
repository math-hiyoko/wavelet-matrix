use std::{
    collections::HashMap,
    hash::Hash,
    ops::{BitAnd, BitOr, BitOrAssign, Shl, Shr, ShlAssign},
};
use num_bigint::ToBigUint;
use num_traits::{One, Zero};
use super::bit_vector::BitVector;
use crate::traits::{
    bit_width::BitWidth, wavelet_matrix::WaveletMatrixTrait,
};

pub(crate) struct WaveletMatrix<NumberType> {
    layers: Vec<BitVector>,
    zeros: Vec<usize>,
    begin_index: HashMap<NumberType, usize>,
    height: usize,
    len: usize,
}

impl<NumberType> WaveletMatrix<NumberType>
where
    NumberType: BitAnd<NumberType, Output = NumberType> + BitWidth + Clone + Hash + One + Ord,
    for<'a> &'a NumberType: Shr<usize, Output = NumberType>,
{
    pub(crate) fn new(data: &Vec<NumberType>) -> Self {
        let mut values = data.clone();
        let height = values.iter().max().map_or(0usize, |max| max.bit_width());
        let len = values.len();
        let mut layers: Vec<BitVector> = Vec::with_capacity(height);
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
            layers.push(BitVector::new(&bits));
            zeros.push(zero_values.len());
            values = [zero_values, one_values].concat();
        }

        let mut begin_index = HashMap::new();
        values.iter().enumerate().for_each(|(i, v)| {
            begin_index.entry(v.clone()).or_insert(i);
        });

        WaveletMatrix {
            layers,
            zeros,
            begin_index,
            height,
            len,
        }
    }
}

impl<NumberType> WaveletMatrixTrait<NumberType, BitVector>
    for WaveletMatrix<NumberType>
where
    NumberType: BitAnd<NumberType, Output = NumberType>
        + BitOr<NumberType, Output = NumberType>
        + BitOrAssign
        + Clone
        + Hash
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
    #[inline]
    fn get_layers(&self) -> &[BitVector] {
        &self.layers
    }

    #[inline]
    fn get_zeros(&self) -> &[usize] {
        &self.zeros
    }

    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn begin_index(&self, value: &NumberType) -> Option<usize> {
        self.begin_index.get(value).copied()
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use super::*;

    fn create_u8() -> WaveletMatrix<u8> {
        let elements: Vec<u8> = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        WaveletMatrix::new(&elements)
    }

    fn create_biguint() -> WaveletMatrix<BigUint> {
        let elements: Vec<BigUint> = [5u32, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
            .into_iter()
            .map(BigUint::from)
            .collect();
        WaveletMatrix::new(&elements)
    }

    #[test]
    fn test_empty() {
        let wv_u8 = WaveletMatrix::<u8>::new(&Vec::new());
        assert_eq!(wv_u8.len(), 0);
        assert_eq!(wv_u8.height(), 0);
        assert_eq!(wv_u8.access(0).unwrap_err().to_string(), "IndexError: index out of bounds");
        assert_eq!(wv_u8.rank(&0u8, 0).unwrap(), 0);
        assert_eq!(wv_u8.select(&0u8, 0).unwrap_err().to_string(), "ValueError: kth must be greater than 0");
        assert_eq!(wv_u8.quantile(0, 0, 1).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_u8.topk(0, 0, Some(1)).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_u8.range_sum(0, 0).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_u8.range_intersection(0, 0, 0, 0).unwrap_err().to_string(), "ValueError: start1 must be less than end1");
        assert_eq!(wv_u8.range_freq(0, 0, None, None).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_u8.range_list(0, 0, None, None).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_u8.range_maxk(0, 0, Some(1)).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_u8.range_mink(0, 0, Some(1)).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_u8.prev_value(0, 0, None, None).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_u8.next_value(0, 0, None, None).unwrap_err().to_string(), "ValueError: start must be less than end");

        let wv_biguint = WaveletMatrix::<BigUint>::new(&Vec::new());
        assert_eq!(wv_biguint.len(), 0);
        assert_eq!(wv_biguint.height(), 0);
        assert_eq!(wv_biguint.access(0).unwrap_err().to_string(), "IndexError: index out of bounds");
        assert_eq!(wv_biguint.rank(&0u32.into(), 0).unwrap(), 0);
        assert_eq!(wv_biguint.select(&0u32.into(), 0).unwrap_err().to_string(), "ValueError: kth must be greater than 0");
        assert_eq!(wv_biguint.quantile(0, 0, 1).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_biguint.topk(0, 0, Some(1)).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_biguint.range_sum(0, 0).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_biguint.range_intersection(0, 0, 0, 0).unwrap_err().to_string(), "ValueError: start1 must be less than end1");
        assert_eq!(wv_biguint.range_freq(0, 0, None, None).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_biguint.range_list(0, 0, None, None).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_biguint.range_maxk(0, 0, Some(1)).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_biguint.range_mink(0, 0, Some(1)).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_biguint.prev_value(0, 0, None, None).unwrap_err().to_string(), "ValueError: start must be less than end");
        assert_eq!(wv_biguint.next_value(0, 0, None, None).unwrap_err().to_string(), "ValueError: start must be less than end");
    }

    #[test]
    fn test_all_zero() {
        let wv_u8 = WaveletMatrix::<u8>::new(&vec![0u8; 64]);
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

        let wv_biguint = WaveletMatrix::<BigUint>::new(&vec![0u32.into(); 64]);
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
        let wv_u8 = WaveletMatrix::<u8>::new(&vec![u8::MAX; 64]);
        assert_eq!(wv_u8.len(), 64);
        assert_eq!(wv_u8.height(), 8);
        assert_eq!(wv_u8.access(1).unwrap(), u8::MAX);
        assert_eq!(wv_u8.rank(&u8::MAX, 1).unwrap(), 1);
        assert_eq!(wv_u8.select(&u8::MAX, 1).unwrap(), Some(0));
        assert_eq!(wv_u8.quantile(0, 64, 1).unwrap(), u8::MAX);
        assert_eq!(wv_u8.topk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u8.range_sum(0, 64).unwrap(), (u8::MAX as u32 * 64).into());
        assert_eq!(wv_u8.range_freq(0, 64, None, None).unwrap(), 64usize);
        assert_eq!(wv_u8.range_list(0, 64, None, None).unwrap().len(), 1);
        assert_eq!(wv_u8.range_maxk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u8.range_mink(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u8.prev_value(0, 64, None, None).unwrap(), Some(u8::MAX));
        assert_eq!(wv_u8.next_value(0, 64, None, None).unwrap(), Some(u8::MAX));
    }

    #[test]
    fn test_access() {
        let wv_u8 = create_u8();
        assert_eq!(wv_u8.access(6).unwrap(), 5u8);

        let wv_biguint = create_biguint();
        assert_eq!(wv_biguint.access(6).unwrap(), 5u32.into());
    }

    #[test]
    fn test_rank() {
        let wv_u8 = create_u8();
        assert_eq!(wv_u8.rank(&5u8, 9).unwrap(), 4usize);

        let wv_biguint = create_biguint();
        assert_eq!(wv_biguint.rank(&5u32.into(), 9).unwrap(), 4usize);
    }

    #[test]
    fn test_select() {
        let wv_u8 = create_u8();
        assert_eq!(wv_u8.select(&5u8, 4).unwrap(), Some(6usize));
        assert_eq!(wv_u8.select(&5u8, 6).unwrap(), None);

        let wv_biguint = create_biguint();
        assert_eq!(wv_biguint.select(&5u32.into(), 4).unwrap(), Some(6usize));
        assert_eq!(wv_biguint.select(&5u32.into(), 6).unwrap(), None);
    }

    #[test]
    fn test_quantile() {
        let wv_u8 = create_u8();
        assert_eq!(wv_u8.quantile(2, 12, 8).unwrap(), 5u8);

        let wv_biguint = create_biguint();
        assert_eq!(wv_biguint.quantile(2, 12, 8).unwrap(), 5u32.into());
    }

    #[test]
    fn test_topk() {
        let wv_u8 = create_u8();
        let result_u8 = wv_u8.topk(1, 10, Some(2)).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(5u8)));
        assert_eq!(result_u8[0].get("count"), Some(&BigUint::from(3usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(1u8)));
        assert_eq!(result_u8[1].get("count"), Some(&BigUint::from(2usize)));

        let wv_biguint = create_biguint();
        let result_biguint = wv_biguint.topk(1, 10, Some(2)).unwrap();
        assert_eq!(result_biguint.len(), 2);
        assert_eq!(result_biguint[0].get("value"), Some(&BigUint::from(5u32)));
        assert_eq!(result_biguint[0].get("count"), Some(&BigUint::from(3usize)));
        assert_eq!(result_biguint[1].get("value"), Some(&BigUint::from(1u32)));
        assert_eq!(result_biguint[1].get("count"), Some(&BigUint::from(2usize)));
    }

    #[test]
    fn test_range_sum() {
        let wv_u8 = create_u8();
        assert_eq!(wv_u8.range_sum(2, 8).unwrap(), 24u32.into());

        let wv_biguint = create_biguint();
        assert_eq!(wv_biguint.range_sum(2, 8).unwrap(), 24u32.into());
    }

    #[test]
    fn test_range_intersection() {
        let wv_u8 = create_u8();
        let result_u8 = wv_u8.range_intersection(0, 6, 6, 11).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(1u8)));
        assert_eq!(result_u8[0].get("count1"), Some(&BigUint::from(1usize)));
        assert_eq!(result_u8[0].get("count2"), Some(&BigUint::from(1usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(5u8)));
        assert_eq!(result_u8[1].get("count1"), Some(&BigUint::from(3usize)));
        assert_eq!(result_u8[1].get("count2"), Some(&BigUint::from(2usize)));

        let wv_biguint = create_biguint();
        let result_biguint = wv_biguint.range_intersection(0, 6, 6, 11).unwrap();
        assert_eq!(result_biguint.len(), 2);
        assert_eq!(result_biguint[0].get("value"), Some(&BigUint::from(1u32)));
        assert_eq!(result_biguint[0].get("count1"), Some(&BigUint::from(1usize)));
        assert_eq!(result_biguint[0].get("count2"), Some(&BigUint::from(1usize)));
        assert_eq!(result_biguint[1].get("value"), Some(&BigUint::from(5u32)));
        assert_eq!(result_biguint[1].get("count1"), Some(&BigUint::from(3usize)));
        assert_eq!(result_biguint[1].get("count2"), Some(&BigUint::from(2usize)));
    }

    #[test]
    fn test_range_freq() {
        let wv_u8 = create_u8();
        assert_eq!(wv_u8.range_freq(1, 9, Some(&4u8), Some(&6u8)).unwrap(), 4usize);

        let wv_biguint = create_biguint();
        assert_eq!(
            wv_biguint.range_freq(1, 9, Some(&4u32.into()), Some(&6u32.into())).unwrap(),
            4usize,
        );
    }

    #[test]
    fn test_range_list() {
        let wv_u8 = create_u8();
        let result_u8 = wv_u8.range_list(1, 9, Some(&4u8), Some(&6u8)).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(4u8)));
        assert_eq!(result_u8[0].get("count"), Some(&BigUint::from(1usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(5u8)));
        assert_eq!(result_u8[1].get("count"), Some(&BigUint::from(3usize)));

        let wv_biguint = create_biguint();
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
        let wv_u8 = create_u8();
        let result_u8 = wv_u8.range_maxk(1, 9, Some(2)).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(6u8)));
        assert_eq!(result_u8[0].get("count"), Some(&BigUint::from(1usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(5u8)));
        assert_eq!(result_u8[1].get("count"), Some(&BigUint::from(3usize)));

        let wv_biguint = create_biguint();
        let result_biguint = wv_biguint.range_maxk(1, 9, Some(2)).unwrap();
        assert_eq!(result_biguint.len(), 2);
        assert_eq!(result_biguint[0].get("value"), Some(&BigUint::from(6u32)));
        assert_eq!(result_biguint[0].get("count"), Some(&BigUint::from(1usize)));
        assert_eq!(result_biguint[1].get("value"), Some(&BigUint::from(5u32)));
        assert_eq!(result_biguint[1].get("count"), Some(&BigUint::from(3usize)));
    }

    #[test]
    fn test_range_mink() {
        let wv_u8 = create_u8();
        let result_u8 = wv_u8.range_mink(1, 9, Some(2)).unwrap();
        assert_eq!(result_u8.len(), 2);
        assert_eq!(result_u8[0].get("value"), Some(&BigUint::from(1u8)));
        assert_eq!(result_u8[0].get("count"), Some(&BigUint::from(2usize)));
        assert_eq!(result_u8[1].get("value"), Some(&BigUint::from(2u8)));
        assert_eq!(result_u8[1].get("count"), Some(&BigUint::from(1usize)));

        let wv_biguint = create_biguint();
        let result_biguint = wv_biguint.range_mink(1, 9, Some(2)).unwrap();
        assert_eq!(result_biguint.len(), 2);
        assert_eq!(result_biguint[0].get("value"), Some(&BigUint::from(1u32)));
        assert_eq!(result_biguint[0].get("count"), Some(&BigUint::from(2usize)));
        assert_eq!(result_biguint[1].get("value"), Some(&BigUint::from(2u32)));
        assert_eq!(result_biguint[1].get("count"), Some(&BigUint::from(1usize)));
    }

    #[test]
    fn test_prev_value() {
        let wv_u8 = create_u8();
        assert_eq!(
            wv_u8.prev_value(1, 9, Some(&4u8), Some(&7u8)).unwrap(),
            Some(6u8),
        );

        let wv_biguint = create_biguint();
        assert_eq!(
            wv_biguint.prev_value(1, 9, Some(&4u32.into()), Some(&7u32.into())).unwrap(),
            Some(6u32.into()),
        );
    }

    #[test]
    fn test_next_value() {
        let wv_u8 = create_u8();
        assert_eq!(
            wv_u8.next_value(1, 9, Some(&3u8), Some(&5u8)).unwrap(),
            Some(4u8),
        );

        let wv_biguint = create_biguint();
        assert_eq!(
            wv_biguint.next_value(1, 9, Some(&3u32.into()), Some(&5u32.into())).unwrap(),
            Some(4u32.into()),
        );
    }
}
