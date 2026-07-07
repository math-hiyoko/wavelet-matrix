use std::{fs, iter, mem};

use bytemuck::{cast_slice, cast_slice_mut};
use memmap2::{Mmap, MmapMut};
use num_integer::Integer;
use num_traits::{One, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
};
use tempfile::tempfile;

use crate::traits::{
    bit_vector::bit_vector::{BitVectorTrait, BlockType},
    utils::bit_select::BitSelect,
};

const SELECT_INDEX_INTERVAL: usize = 512;

pub(crate) struct DiskBitVector {
    len: usize,
    ranks: Mmap,
    ranks_file: fs::File,
    blocks: Mmap,
    blocks_file: fs::File,
    select_index: [Mmap; 2],
    select_index_file: [fs::File; 2],
}

impl DiskBitVector {
    pub(super) fn new(blocks: Mmap, blocks_file: fs::File, len: usize) -> PyResult<Self> {
        assert!(blocks.len().is_multiple_of(mem::size_of::<BlockType>()));
        let blocks_data: &[BlockType] = cast_slice(&blocks[..]);

        // Build the rank index structure.
        let ranks_file = tempfile().map_err(PyRuntimeError::new_err)?;
        ranks_file
            .set_len((blocks_data.len() + 1) as u64 * mem::size_of::<usize>() as u64)
            .map_err(PyRuntimeError::new_err)?;
        #[allow(unsafe_code)]
        let mut ranks = unsafe { MmapMut::map_mut(&ranks_file).map_err(PyRuntimeError::new_err)? };
        let ranks_data: &mut [usize] = cast_slice_mut(&mut ranks[..]);
        iter::once(0usize)
            .chain(blocks_data.iter().scan(0usize, |acc, block| {
                *acc += block.count_ones() as usize;
                Some(*acc)
            }))
            .enumerate()
            .for_each(|(index, rank)| ranks_data[index] = rank);

        let select_index_file = [
            {
                let file = tempfile().map_err(PyRuntimeError::new_err)?;
                file.set_len(
                    (((len - ranks_data.last().unwrap()) / SELECT_INDEX_INTERVAL + 2)
                        * mem::size_of::<usize>()) as u64,
                )
                .map_err(PyRuntimeError::new_err)?;
                file
            },
            {
                let file = tempfile().map_err(PyRuntimeError::new_err)?;
                file.set_len(
                    ((ranks_data.last().unwrap() / SELECT_INDEX_INTERVAL + 2)
                        * mem::size_of::<usize>()) as u64,
                )
                .map_err(PyRuntimeError::new_err)?;
                file
            },
        ];
        #[allow(unsafe_code)]
        let mut select_index_0 =
            unsafe { MmapMut::map_mut(&select_index_file[0]).map_err(PyRuntimeError::new_err)? };
        #[allow(unsafe_code)]
        let mut select_index_1 =
            unsafe { MmapMut::map_mut(&select_index_file[1]).map_err(PyRuntimeError::new_err)? };
        let select_index_data: [&mut [usize]; 2] = [
            cast_slice_mut(&mut select_index_0[..]),
            cast_slice_mut(&mut select_index_1[..]),
        ];
        select_index_data[0][0] = 0;
        select_index_data[1][0] = 0;
        let mut count = [0usize, 0usize];
        for (index, bit) in blocks_data
            .iter()
            .flat_map(|block| {
                (0..BlockType::BITS as usize)
                    .map(move |i| ((block >> i) & BlockType::one()) as usize)
            })
            .take(len)
            .enumerate()
        {
            count[bit] += 1;
            let (count_div, count_rem) = count[bit].div_rem(&SELECT_INDEX_INTERVAL);
            if count_rem.is_zero() {
                select_index_data[bit][count_div] = index;
            }
        }
        select_index_data[0][count[0] / SELECT_INDEX_INTERVAL + 1] = len;
        select_index_data[1][count[1] / SELECT_INDEX_INTERVAL + 1] = len;

        Ok(Self {
            len,
            ranks: ranks.make_read_only().map_err(PyRuntimeError::new_err)?,
            ranks_file,
            blocks,
            blocks_file,
            select_index: [
                select_index_0
                    .make_read_only()
                    .map_err(PyRuntimeError::new_err)?,
                select_index_1
                    .make_read_only()
                    .map_err(PyRuntimeError::new_err)?,
            ],
            select_index_file,
        })
    }
}

impl Clone for DiskBitVector {
    fn clone(&self) -> Self {
        let ranks_file = tempfile().unwrap();
        ranks_file.set_len(self.ranks.len() as u64).unwrap();
        #[allow(unsafe_code)]
        let mut ranks = unsafe { MmapMut::map_mut(&ranks_file).unwrap() };
        ranks.copy_from_slice(&self.ranks[..]);

        let blocks_file = tempfile().unwrap();
        blocks_file.set_len(self.blocks.len() as u64).unwrap();
        #[allow(unsafe_code)]
        let mut blocks = unsafe { MmapMut::map_mut(&blocks_file).unwrap() };
        blocks.copy_from_slice(&self.blocks[..]);

        let select_index_file = [
            {
                let file = tempfile().unwrap();
                file.set_len(self.select_index[0].len() as u64).unwrap();
                file
            },
            {
                let file = tempfile().unwrap();
                file.set_len(self.select_index[1].len() as u64).unwrap();
                file
            },
        ];
        let mut select_index = [
            #[allow(unsafe_code)]
            unsafe {
                MmapMut::map_mut(&select_index_file[0]).unwrap()
            },
            #[allow(unsafe_code)]
            unsafe {
                MmapMut::map_mut(&select_index_file[1]).unwrap()
            },
        ];
        select_index[0].copy_from_slice(&self.select_index[0][..]);
        select_index[1].copy_from_slice(&self.select_index[1][..]);

        let [select_index_0, select_index_1] = select_index;

        Self {
            len: self.len,
            ranks: ranks.make_read_only().unwrap(),
            ranks_file,
            blocks: blocks.make_read_only().unwrap(),
            blocks_file,
            select_index: [
                select_index_0.make_read_only().unwrap(),
                select_index_1.make_read_only().unwrap(),
            ],
            select_index_file,
        }
    }
}

impl BitVectorTrait for DiskBitVector {
    #[inline]
    fn values(&self) -> PyResult<Vec<BlockType>> {
        Ok(cast_slice(&self.blocks[..]).to_vec())
    }

    #[inline]
    fn access(&self, index: usize) -> PyResult<bool> {
        if index >= self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        let (block_index, bit_index) = index.div_rem(&(BlockType::BITS as usize));
        let blocks_data: &[BlockType] = cast_slice(&self.blocks[..]);
        Ok(((blocks_data[block_index] >> bit_index) & BlockType::one()).is_one())
    }

    #[inline]
    fn rank(&self, bit: bool, end: usize) -> PyResult<usize> {
        if end > self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if self.len.is_zero() {
            return Ok(0);
        }
        if !bit {
            return Ok(end - self.rank(true, end)?);
        }

        let (block_index, bit_index) = end.div_rem(&(BlockType::BITS as usize));
        let ranks_data: &[usize] = cast_slice(&self.ranks[..]);
        let blocks_data: &[BlockType] = cast_slice(&self.blocks[..]);
        let mut rank = ranks_data[block_index];
        if block_index < blocks_data.len() {
            rank += (blocks_data[block_index]
                & ((BlockType::one() << bit_index) - BlockType::one()))
            .count_ones() as usize;
        }
        Ok(rank)
    }

    #[inline]
    fn select(&self, bit: bool, mut kth: usize) -> PyResult<Option<usize>> {
        if kth.is_zero() {
            return Err(PyValueError::new_err("kth must be greater than 0"));
        }
        if kth > self.rank(bit, self.len)? {
            return Ok(None);
        }

        let select_index_data: [&[usize]; 2] = [
            cast_slice(&self.select_index[0][..]),
            cast_slice(&self.select_index[1][..]),
        ];
        let ranks_data: &[usize] = cast_slice(&self.ranks[..]);
        let blocks_data: &[BlockType] = cast_slice(&self.blocks[..]);

        let block_index = {
            let mut left = select_index_data[bit as usize][(kth - 1) / SELECT_INDEX_INTERVAL]
                / (BlockType::BITS as usize);
            let mut right = select_index_data[bit as usize][kth / SELECT_INDEX_INTERVAL + 1]
                .div_ceil(BlockType::BITS as usize);
            debug_assert!(right <= self.blocks.len());
            while left + 1 < right {
                let mid = (left + right) / 2;
                let rank_at_mid = if bit {
                    ranks_data[mid]
                } else {
                    mid * (BlockType::BITS as usize) - ranks_data[mid]
                };
                if rank_at_mid < kth {
                    left = mid;
                } else {
                    right = mid;
                }
            }
            left
        };

        kth -= if bit {
            ranks_data[block_index]
        } else {
            block_index * (BlockType::BITS as usize) - ranks_data[block_index]
        };
        let index = blocks_data[block_index].bit_select(bit, kth).unwrap()
            + block_index * (BlockType::BITS as usize);

        Ok(Some(index))
    }
}

#[cfg(test)]
mod tests {
    use pyo3::Python;

    use super::*;

    fn create_disk_bit_vector(bits: Vec<bool>) -> DiskBitVector {
        let len = bits.len();
        let blocks_file = tempfile().unwrap();
        blocks_file
            .set_len((len.div_ceil(BlockType::BITS as usize) * mem::size_of::<BlockType>()) as u64)
            .unwrap();
        #[allow(unsafe_code)]
        let mut blocks = unsafe { MmapMut::map_mut(&blocks_file).unwrap() };
        let blocks_data: &mut [BlockType] = cast_slice_mut(&mut blocks[..]);
        bits.chunks(BlockType::BITS as usize)
            .enumerate()
            .for_each(|(index, chunk)| {
                blocks_data[index] =
                    chunk
                        .iter()
                        .enumerate()
                        .fold(BlockType::zero(), |acc, (i, &b)| {
                            if b {
                                acc | (BlockType::one() << i)
                            } else {
                                acc
                            }
                        })
            });

        DiskBitVector::new(blocks.make_read_only().unwrap(), blocks_file, len).unwrap()
    }

    fn create_dummy() -> DiskBitVector {
        let bits = [true, false, true, true, false, true, false, false].repeat(999);
        create_disk_bit_vector(bits)
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let bv = create_disk_bit_vector(vec![]);

        assert_eq!(bv.values().unwrap(), Vec::<BlockType>::new());
        assert_eq!(
            bv.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(bv.rank(true, 0).unwrap(), 0);
        assert_eq!(bv.rank(false, 0).unwrap(), 0);
        assert_eq!(bv.select(true, 1).unwrap(), None);
        assert_eq!(bv.select(false, 1).unwrap(), None);
    }

    #[test]
    fn test_exact_block() {
        Python::initialize();

        let bits = vec![true; 1024];
        let bv = create_disk_bit_vector(bits);

        for i in 0..1024 {
            assert!(bv.access(i).unwrap());
            assert_eq!(bv.rank(true, i + 1).unwrap(), i + 1);
            assert_eq!(bv.rank(false, i + 1).unwrap(), 0);
            assert_eq!(bv.select(true, i + 1).unwrap(), Some(i));
            assert_eq!(bv.select(false, i + 1).unwrap(), None);
        }
    }

    #[test]
    fn test_values() {
        Python::initialize();

        let bv = create_dummy();
        assert_eq!(
            bv.values().unwrap(),
            [true, false, true, true, false, true, false, false]
                .repeat(999)
                .chunks(BlockType::BITS as usize)
                .map(|chunk| {
                    chunk
                        .iter()
                        .enumerate()
                        .fold(BlockType::zero(), |acc, (i, &b)| {
                            if b {
                                acc | (BlockType::one() << i)
                            } else {
                                acc
                            }
                        })
                })
                .collect::<Vec<BlockType>>()
        );
    }

    #[test]
    fn test_access() {
        Python::initialize();

        let bv = create_dummy();

        assert!(bv.access(0).unwrap());
        assert!(!bv.access(1001).unwrap());
        assert!(bv.access(2002).unwrap());
        assert!(bv.access(3003).unwrap());
        assert!(!bv.access(4004).unwrap());
        assert!(bv.access(5005).unwrap());
        assert!(!bv.access(6006).unwrap());
        assert!(!bv.access(7007).unwrap());
        assert_eq!(
            bv.access(7992).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
    }

    #[test]
    fn test_rank() {
        Python::initialize();

        let bv = create_dummy();

        assert_eq!(bv.rank(true, 0).unwrap(), 0);
        assert_eq!(bv.rank(true, 1001).unwrap(), 501);
        assert_eq!(bv.rank(true, 2002).unwrap(), 1001);
        assert_eq!(bv.rank(true, 3003).unwrap(), 1502);
        assert_eq!(bv.rank(true, 4004).unwrap(), 2003);
        assert_eq!(bv.rank(true, 5005).unwrap(), 2503);
        assert_eq!(bv.rank(true, 6006).unwrap(), 3004);
        assert_eq!(bv.rank(true, 7007).unwrap(), 3504);
        assert_eq!(bv.rank(true, 7992).unwrap(), 3996);
        assert_eq!(
            bv.rank(true, 7993).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );

        assert_eq!(bv.rank(false, 0).unwrap(), 0);
        assert_eq!(bv.rank(false, 1001).unwrap(), 500);
        assert_eq!(bv.rank(false, 2002).unwrap(), 1001);
        assert_eq!(bv.rank(false, 3003).unwrap(), 1501);
        assert_eq!(bv.rank(false, 4004).unwrap(), 2001);
        assert_eq!(bv.rank(false, 5005).unwrap(), 2502);
        assert_eq!(bv.rank(false, 6006).unwrap(), 3002);
        assert_eq!(bv.rank(false, 7007).unwrap(), 3503);
        assert_eq!(bv.rank(false, 7992).unwrap(), 3996);
        assert_eq!(
            bv.rank(false, 7993).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
    }

    #[test]
    fn test_select() {
        Python::initialize();

        let bv = create_dummy();

        assert_eq!(
            bv.select(true, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
        assert_eq!(bv.select(true, 1).unwrap(), Some(0));
        assert_eq!(bv.select(true, 1000).unwrap(), Some(1997));
        assert_eq!(bv.select(true, 2000).unwrap(), Some(3997));
        assert_eq!(bv.select(true, 3000).unwrap(), Some(5997));
        assert_eq!(bv.select(true, 3996).unwrap(), Some(7989));
        assert_eq!(bv.select(true, 3997).unwrap(), None);

        assert_eq!(
            bv.select(false, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
        assert_eq!(bv.select(false, 1).unwrap(), Some(1));
        assert_eq!(bv.select(false, 1000).unwrap(), Some(1999));
        assert_eq!(bv.select(false, 2000).unwrap(), Some(3999));
        assert_eq!(bv.select(false, 3000).unwrap(), Some(5999));
        assert_eq!(bv.select(false, 3996).unwrap(), Some(7991));
        assert_eq!(bv.select(false, 3997).unwrap(), None);
    }
}
