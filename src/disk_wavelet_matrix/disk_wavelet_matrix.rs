use std::{iter, marker, mem, ops};

use bytemuck::{Pod, cast_slice, cast_slice_mut};
use memmap2::{Mmap, MmapMut};
use num_bigint::ToBigUint;
use num_traits::{One, PrimInt, Unsigned};
use pyo3::{PyResult, exceptions::PyRuntimeError};
use rayon::prelude::*;
use tempfile::tempfile;

use super::disk_bit_vector::DiskBitVector;
use crate::traits::{
    bit_vector::bit_vector::BlockType, utils::bit_width::BitWidth,
    wavelet_matrix::wavelet_matrix::WaveletMatrixTrait,
};

#[derive(Clone)]
pub(crate) struct DiskWaveletMatrix<NumberType> {
    layers: Vec<DiskBitVector>,
    zeros_count_per_layer: Vec<usize>,
    height: usize,
    len: usize,
    phantom: marker::PhantomData<NumberType>,
}

impl<NumberType> DiskWaveletMatrix<NumberType>
where
    NumberType: BitWidth + One + PrimInt + Unsigned + Pod + Send + Sync,
    for<'a> &'a NumberType: ops::Shr<usize, Output = NumberType>,
{
    pub(crate) fn new(data: Mmap) -> PyResult<Self> {
        assert!(data.len().is_multiple_of(mem::size_of::<NumberType>()));
        let len = data.len() / mem::size_of::<NumberType>();

        let mut values = data;
        let values_data: &[NumberType] = cast_slice(&values[..]);
        let height = values_data
            .par_iter()
            .max()
            .map_or(0usize, |max| max.bit_width());

        let mut zeros_count_per_layer = Vec::with_capacity(height);
        let mut layer_blocks_vec = Vec::with_capacity(height);
        for i in 0..height {
            let current_layer_bits_file = tempfile().map_err(PyRuntimeError::new_err)?;
            current_layer_bits_file
                .set_len(
                    (len.div_ceil(BlockType::BITS as usize) * mem::size_of::<BlockType>()) as u64,
                )
                .map_err(PyRuntimeError::new_err)?;
            #[allow(unsafe_code)]
            let mut current_layer_bits = unsafe {
                MmapMut::map_mut(&current_layer_bits_file).map_err(PyRuntimeError::new_err)?
            };
            assert!(
                current_layer_bits
                    .len()
                    .is_multiple_of(mem::size_of::<BlockType>())
            );
            let current_layer_bits_data: &mut [BlockType] =
                cast_slice_mut(&mut current_layer_bits[..]);
            let values_data: &[NumberType] = cast_slice(&values[..]);
            values_data
                .par_iter()
                .map(|value| (value >> (height - i - 1) & NumberType::one()).is_one())
                .chunks(BlockType::BITS as usize)
                .zip(current_layer_bits_data.par_iter_mut())
                .for_each(|(bits_chunk, block)| {
                    bits_chunk.iter().enumerate().for_each(|(j, &bit)| {
                        if bit {
                            *block |= BlockType::one() << j;
                        }
                    });
                });
            let zeros_count = len
                - current_layer_bits_data
                    .par_iter()
                    .map(|&block| block.count_ones() as usize)
                    .sum::<usize>();

            let mut next_values = MmapMut::map_anon(len * mem::size_of::<NumberType>())
                .map_err(PyRuntimeError::new_err)?;
            assert!(
                next_values
                    .len()
                    .is_multiple_of(mem::size_of::<NumberType>())
            );
            let next_values_data: &mut [NumberType] = cast_slice_mut(&mut next_values[..]);
            let mut zero_index = 0usize;
            let mut one_index = zeros_count;
            for (bit, value) in iter::zip(
                current_layer_bits_data
                    .iter()
                    .flat_map(|block| {
                        (0..BlockType::BITS as usize)
                            .map(move |i| ((block >> i) & BlockType::one()).is_one())
                    })
                    .take(len),
                values_data.iter(),
            ) {
                if bit {
                    next_values_data[one_index] = *value;
                    one_index += 1;
                } else {
                    next_values_data[zero_index] = *value;
                    zero_index += 1;
                }
            }

            zeros_count_per_layer.push(zeros_count);
            layer_blocks_vec.push((
                current_layer_bits
                    .make_read_only()
                    .map_err(PyRuntimeError::new_err)?,
                current_layer_bits_file,
            ));
            values = next_values
                .make_read_only()
                .map_err(PyRuntimeError::new_err)?;
        }

        let layers = layer_blocks_vec
            .into_par_iter()
            .map(|(blocks, blocks_file)| DiskBitVector::new(blocks, blocks_file, len))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            layers,
            zeros_count_per_layer,
            height,
            len,
            phantom: marker::PhantomData,
        })
    }
}

impl<NumberType> WaveletMatrixTrait<NumberType, DiskBitVector> for DiskWaveletMatrix<NumberType>
where
    NumberType: ops::BitOrAssign
        + BitWidth
        + PrimInt
        + Unsigned
        + ops::ShlAssign<usize>
        + ToBigUint
        + Send
        + Sync
        + Pod
        + 'static,
    for<'a> &'a NumberType:
        ops::Shl<usize, Output = NumberType> + ops::Shr<usize, Output = NumberType>,
{
    #[inline]
    fn get_layers(&self) -> &[DiskBitVector] {
        &self.layers
    }

    #[inline]
    fn get_zeros_count_per_layer(&self) -> &[usize] {
        &self.zeros_count_per_layer
    }

    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use pyo3::Python;

    use super::*;

    fn create_u8() -> DiskWaveletMatrix<u8> {
        let elements: Vec<u8> = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        let mut mmap = MmapMut::map_anon(elements.len() * mem::size_of::<u8>()).unwrap();
        let mmap_data: &mut [u8] = cast_slice_mut(&mut mmap[..]);
        mmap_data.copy_from_slice(&elements);
        DiskWaveletMatrix::new(mmap.make_read_only().unwrap()).unwrap()
    }

    fn create_u128() -> DiskWaveletMatrix<u128> {
        let elements: Vec<u128> = vec![5u128, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        let mut mmap = MmapMut::map_anon(elements.len() * mem::size_of::<u128>()).unwrap();
        let mmap_data: &mut [u128] = cast_slice_mut(&mut mmap[..]);
        mmap_data.copy_from_slice(&elements);
        DiskWaveletMatrix::new(mmap.make_read_only().unwrap()).unwrap()
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let mmap_empty = MmapMut::map_anon(0).unwrap();
        let wv_u8 = DiskWaveletMatrix::<u8>::new(mmap_empty.make_read_only().unwrap()).unwrap();
        assert_eq!(wv_u8.len(), 0);
        assert_eq!(wv_u8.height(), 0);
        assert_eq!(wv_u8.values().unwrap(), Vec::<u8>::new());
        assert_eq!(
            wv_u8.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(wv_u8.rank(&0u8, 0).unwrap(), 0);
        assert_eq!(
            wv_u8.select(&0u8, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
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

        let mmap_empty = MmapMut::map_anon(0).unwrap();
        let wv_u128 = DiskWaveletMatrix::<u128>::new(mmap_empty.make_read_only().unwrap()).unwrap();
        assert_eq!(wv_u128.len(), 0);
        assert_eq!(wv_u128.height(), 0);
        assert_eq!(wv_u128.values().unwrap(), Vec::<u128>::new());
        assert_eq!(
            wv_u128.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(wv_u128.rank(&0u128, 0).unwrap(), 0);
        assert_eq!(
            wv_u128.select(&0u128, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
        assert_eq!(
            wv_u128.quantile(0, 0, 1).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u128.topk(0, 0, Some(1)).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u128.range_sum(0, 0).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u128
                .range_intersection(0, 0, 0, 0)
                .unwrap_err()
                .to_string(),
            "ValueError: start1 must be less than end1"
        );
        assert_eq!(
            wv_u128
                .range_freq(0, 0, None, None)
                .unwrap_err()
                .to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u128
                .range_list(0, 0, None, None)
                .unwrap_err()
                .to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u128.range_maxk(0, 0, Some(1)).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u128.range_mink(0, 0, Some(1)).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u128.prev_value(0, 0, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u128.next_value(0, 0, None).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
    }

    #[test]
    fn test_all_zero() {
        Python::initialize();

        let mmap_u8_all_zero = MmapMut::map_anon(64 * mem::size_of::<u8>()).unwrap();
        let wv_u8 =
            DiskWaveletMatrix::<u8>::new(mmap_u8_all_zero.make_read_only().unwrap()).unwrap();
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

        let mmap_u128_all_zero = MmapMut::map_anon(64 * mem::size_of::<u128>()).unwrap();
        let wv_u128 =
            DiskWaveletMatrix::<u128>::new(mmap_u128_all_zero.make_read_only().unwrap()).unwrap();
        assert_eq!(wv_u128.len(), 64);
        assert_eq!(wv_u128.height(), 0);
        assert_eq!(wv_u128.values().unwrap(), vec![0u128; 64]);
        assert_eq!(wv_u128.access(1).unwrap(), 0u128);
        assert_eq!(wv_u128.rank(&0u128, 1).unwrap(), 1);
        assert_eq!(wv_u128.select(&0u128, 1).unwrap(), Some(0));
        assert_eq!(wv_u128.quantile(0, 64, 1).unwrap(), 0u128);
        assert_eq!(wv_u128.topk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u128.range_sum(0, 64).unwrap(), 0u128.into());
        assert_eq!(wv_u128.range_freq(0, 64, None, None).unwrap(), 64usize);
        assert_eq!(wv_u128.range_list(0, 64, None, None).unwrap().len(), 1);
        assert_eq!(wv_u128.range_maxk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u128.range_mink(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u128.prev_value(0, 64, None).unwrap(), Some(0u128));
        assert_eq!(wv_u128.next_value(0, 64, None).unwrap(), Some(0u128));
    }

    #[test]
    fn test_max_value() {
        Python::initialize();

        let mut mmap_u8_max_value = MmapMut::map_anon(64 * mem::size_of::<u8>()).unwrap();
        let mmap_u8_data: &mut [u8] = cast_slice_mut(&mut mmap_u8_max_value[..]);
        mmap_u8_data.fill(u8::MAX);
        let wv_u8 =
            DiskWaveletMatrix::<u8>::new(mmap_u8_max_value.make_read_only().unwrap()).unwrap();
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

        let mut mmap_u128_max_value = MmapMut::map_anon(64 * mem::size_of::<u128>()).unwrap();
        let mmap_u128_data: &mut [u128] = cast_slice_mut(&mut mmap_u128_max_value[..]);
        mmap_u128_data.fill(u128::MAX);
        let wv_u128 =
            DiskWaveletMatrix::<u128>::new(mmap_u128_max_value.make_read_only().unwrap()).unwrap();
        assert_eq!(wv_u128.len(), 64);
        assert_eq!(wv_u128.height(), 128);
        assert_eq!(wv_u128.values().unwrap(), vec![u128::MAX; 64]);
        assert_eq!(wv_u128.access(1).unwrap(), u128::MAX);
        assert_eq!(wv_u128.rank(&u128::MAX, 1).unwrap(), 1);
        assert_eq!(wv_u128.select(&u128::MAX, 1).unwrap(), Some(0));
        assert_eq!(wv_u128.quantile(0, 64, 1).unwrap(), u128::MAX);
        assert_eq!(wv_u128.topk(0, 64, None).unwrap().len(), 1);
        assert_eq!(
            wv_u128.range_sum(0, 64).unwrap(),
            BigUint::from(u128::MAX) * 64u128,
        );
        assert_eq!(wv_u128.range_freq(0, 64, None, None).unwrap(), 64usize);
        assert_eq!(wv_u128.range_list(0, 64, None, None).unwrap().len(), 1);
        assert_eq!(wv_u128.range_maxk(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u128.range_mink(0, 64, None).unwrap().len(), 1);
        assert_eq!(wv_u128.prev_value(0, 64, None).unwrap(), Some(u128::MAX));
        assert_eq!(wv_u128.next_value(0, 64, None).unwrap(), Some(u128::MAX));
    }

    #[test]
    fn test_values() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(
            wv_u8.values().unwrap(),
            vec![5u8, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128.values().unwrap(),
            vec![5u128, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        );
    }

    #[test]
    fn test_access() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(wv_u8.access(6).unwrap(), 5u8);

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.access(6).unwrap(), 5u128);
    }

    #[test]
    fn test_rank() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(wv_u8.rank(&5u8, 9).unwrap(), 4usize);

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.rank(&5u128, 9).unwrap(), 4usize);
    }

    #[test]
    fn test_select() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(wv_u8.select(&5u8, 4).unwrap(), Some(6usize));
        assert_eq!(wv_u8.select(&5u8, 6).unwrap(), None);

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.select(&5u128, 4).unwrap(), Some(6usize));
        assert_eq!(wv_u128.select(&5u128, 6).unwrap(), None);
    }

    #[test]
    fn test_quantile() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(wv_u8.quantile(2, 12, 8).unwrap(), 5u8);

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.quantile(2, 12, 8).unwrap(), 5u128);
    }

    #[test]
    fn test_topk() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(
            wv_u8.topk(1, 10, Some(2)).unwrap(),
            vec![(5u8, 3usize), (1u8, 2usize),],
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128.topk(1, 10, Some(2)).unwrap(),
            vec![(5u128, 3usize), (1u128, 2usize),],
        );
    }

    #[test]
    fn test_range_sum() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(wv_u8.range_sum(2, 8).unwrap(), 24u32.into());

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.range_sum(2, 8).unwrap(), 24u32.into());
    }

    #[test]
    fn test_range_intersection() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(
            wv_u8.range_intersection(0, 6, 6, 11).unwrap(),
            vec![(1u8, 1usize, 1usize), (5u8, 3usize, 2usize),],
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128.range_intersection(0, 6, 6, 11).unwrap(),
            vec![(1u128, 1usize, 1usize), (5u128, 3usize, 2usize),],
        );
    }

    #[test]
    fn test_range_freq() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(
            wv_u8.range_freq(1, 9, Some(&4u8), Some(&6u8)).unwrap(),
            4usize
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128
                .range_freq(1, 9, Some(&4u128), Some(&6u128))
                .unwrap(),
            4usize,
        );
    }

    #[test]
    fn test_range_list() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(
            wv_u8.range_list(1, 9, Some(&4u8), Some(&6u8)).unwrap(),
            vec![(4u8, 1usize), (5u8, 3usize),],
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128
                .range_list(1, 9, Some(&4u128), Some(&6u128))
                .unwrap(),
            vec![(4u128, 1usize), (5u128, 3usize),],
        );
    }

    #[test]
    fn test_range_maxk() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(
            wv_u8.range_maxk(1, 9, Some(2)).unwrap(),
            vec![(6u8, 1usize), (5u8, 3usize),],
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128.range_maxk(1, 9, Some(2)).unwrap(),
            vec![(6u128, 1usize), (5u128, 3usize),],
        );
    }

    #[test]
    fn test_range_mink() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(
            wv_u8.range_mink(1, 9, Some(2)).unwrap(),
            vec![(1u8, 2usize), (2u8, 1usize),],
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128.range_mink(1, 9, Some(2)).unwrap(),
            vec![(1u128, 2usize), (2u128, 1usize),],
        );
    }

    #[test]
    fn test_prev_value() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(wv_u8.prev_value(1, 9, Some(&7u8)).unwrap(), Some(6u8),);

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.prev_value(1, 9, Some(&7u128)).unwrap(), Some(6u128),);
    }

    #[test]
    fn test_next_value() {
        Python::initialize();

        let wv_u8 = create_u8();
        assert_eq!(wv_u8.next_value(1, 9, Some(&3u8)).unwrap(), Some(4u8),);

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.next_value(1, 9, Some(&3u128)).unwrap(), Some(4u128),);
    }
}
