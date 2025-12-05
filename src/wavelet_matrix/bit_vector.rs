use num_integer::Integer;
use num_traits::{One, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
};
use crate::traits::{
    bit_select::BitSelect,
    bit_vector::BitVectorTrait,
};

type BlockType = u64;


pub(crate) struct BitVector {
    len: usize,
    ranks: Vec<usize>,
    blocks: Vec<BlockType>,
}

impl BitVector {
    pub(super) fn new(bits: &Vec<bool>) -> Self {
        let len = bits.len();
        // Pack blocks into BitType words
        let blocks: Vec<BlockType> = bits
            .chunks(BlockType::BITS as usize)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(BlockType::zero(), |acc, (i, &bit)| {
                        if bit {
                            acc | (BlockType::one() << i)
                        } else {
                            acc
                        }
                    })
            })
            .collect();
        // Build the rank index structure.
        let ranks: Vec<usize> = std::iter::once(0usize)
            .chain(blocks.iter().scan(0usize, |acc, block| {
                *acc += block.count_ones() as usize;
                Some(*acc)
            }))
            .collect();

        Self { len, ranks, blocks }
    }
}

impl BitVectorTrait for BitVector {
    #[inline]
    fn access(&self, index: usize) -> PyResult<bool> {
        if index >= self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        let (block_index, bit_index) = index.div_rem(&(BlockType::BITS as usize));
        Ok(((self.blocks[block_index] >> bit_index) & BlockType::one()).is_one())
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
        let mut rank = self.ranks[block_index];
        if block_index < self.blocks.len() {
            rank += (self.blocks[block_index] & ((1 << bit_index) - BlockType::one())).count_ones() as usize;
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

        // Find the block containing the k-th bit
        let block_index = {
            let mut left = 0usize;
            let mut right = self.ranks.len();
            while left + 1 < right {
                let mid = (left + right) / 2;
                let rank_at_mid = if bit {
                    self.ranks[mid]
                } else {
                    (mid * BlockType::BITS as usize).min(self.len) - self.ranks[mid]
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
            self.ranks[block_index]
        } else {
            (block_index * BlockType::BITS as usize).min(self.len) - self.ranks[block_index]
        };
        let index = block_index * BlockType::BITS as usize
            + self.blocks[block_index]
                .bit_select(bit, kth)
                .ok_or(PyRuntimeError::new_err("Failed to find the k-th bit"))?;
        Ok(Some(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy() -> BitVector {
        let bits = vec![true, false, true, true, false, true, false, false].repeat(999);
        BitVector::new(&bits)
    }

    #[test]
    fn test_empty() {
        let bv = BitVector::new(&vec![]);

        assert_eq!(bv.access(0).unwrap_err().to_string(), "IndexError: index out of bounds");
        assert_eq!(bv.rank(true, 0).unwrap(), 0);
        assert_eq!(bv.rank(false, 0).unwrap(), 0);
        assert_eq!(bv.select(true, 1).unwrap(), None);
        assert_eq!(bv.select(false, 1).unwrap(), None);
    }

    #[test]
    fn test_exact_block() {
        let bits = vec![true; 1024];
        let bv = BitVector::new(&bits);

        for i in 0..1024 {
            assert_eq!(bv.access(i).unwrap(), true);
            assert_eq!(bv.rank(true, i + 1).unwrap(), i + 1);
            assert_eq!(bv.rank(false, i + 1).unwrap(), 0);
            assert_eq!(bv.select(true, i + 1).unwrap(), Some(i));
            assert_eq!(bv.select(false, i + 1).unwrap(), None);
        }
    }

    #[test]
    fn test_access() {
        let bv = create_dummy();

        assert_eq!(bv.access(0).unwrap(), true);
        assert_eq!(bv.access(1001).unwrap(), false);
        assert_eq!(bv.access(2002).unwrap(), true);
        assert_eq!(bv.access(3003).unwrap(), true);
        assert_eq!(bv.access(4004).unwrap(), false);
        assert_eq!(bv.access(5005).unwrap(), true);
        assert_eq!(bv.access(6006).unwrap(), false);
        assert_eq!(bv.access(7007).unwrap(), false);
        assert_eq!(bv.access(7992).unwrap_err().to_string(), "IndexError: index out of bounds");
    }

    #[test]
    fn test_rank() {
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
        assert_eq!(bv.rank(true, 7993).unwrap_err().to_string(), "IndexError: index out of bounds");

        assert_eq!(bv.rank(false, 0).unwrap(), 0);
        assert_eq!(bv.rank(false, 1001).unwrap(), 500);
        assert_eq!(bv.rank(false, 2002).unwrap(), 1001);
        assert_eq!(bv.rank(false, 3003).unwrap(), 1501);
        assert_eq!(bv.rank(false, 4004).unwrap(), 2001);
        assert_eq!(bv.rank(false, 5005).unwrap(), 2502);
        assert_eq!(bv.rank(false, 6006).unwrap(), 3002);
        assert_eq!(bv.rank(false, 7007).unwrap(), 3503);
        assert_eq!(bv.rank(false, 7992).unwrap(), 3996);
        assert_eq!(bv.rank(false, 7993).unwrap_err().to_string(), "IndexError: index out of bounds");
    }

    #[test]
    fn test_select() {
        let bv = create_dummy();

        assert_eq!(bv.select(true, 0).unwrap_err().to_string(), "ValueError: kth must be greater than 0");
        assert_eq!(bv.select(true, 1).unwrap(), Some(0));
        assert_eq!(bv.select(true, 1000).unwrap(), Some(1997));
        assert_eq!(bv.select(true, 2000).unwrap(), Some(3997));
        assert_eq!(bv.select(true, 3000).unwrap(), Some(5997));
        assert_eq!(bv.select(true, 3996).unwrap(), Some(7989));
        assert_eq!(bv.select(true, 3997).unwrap(), None);

        assert_eq!(bv.select(false, 0).unwrap_err().to_string(), "ValueError: kth must be greater than 0");
        assert_eq!(bv.select(false, 1).unwrap(), Some(1));
        assert_eq!(bv.select(false, 1000).unwrap(), Some(1999));
        assert_eq!(bv.select(false, 2000).unwrap(), Some(3999));
        assert_eq!(bv.select(false, 3000).unwrap(), Some(5999));
        assert_eq!(bv.select(false, 3996).unwrap(), Some(7991));
        assert_eq!(bv.select(false, 3997).unwrap(), None);
    }
}
