use num_traits::{One, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
};
use std::{cmp::max, mem::replace};

use crate::traits::{
    bit_vector::{bit_vector::BitVectorTrait, dynamic_bit_vector::DynamicBitVectorTrait},
    utils::bit_select::BitSelect,
};

type BlockType = u64;
const BITS_SIZE_LIMIT: usize = BlockType::BITS as usize / 2;

#[derive(Debug, Clone)]
enum DynamicBitVectorNode {
    Internal {
        left: Box<DynamicBitVectorNode>,
        right: Box<DynamicBitVectorNode>,
        left_total: usize,
        left_ones: usize,
        balance: isize,
    },
    Leaf {
        bits: BlockType,
    },
}

impl DynamicBitVectorNode {
    fn new(bits: &[bool]) -> Self {
        if bits.is_empty() {
            return Self::Leaf {
                bits: BlockType::zero(),
            };
        }

        struct DynamicBitVectorNodeBuildItem {
            node: Box<DynamicBitVectorNode>,
            total: usize,
            ones: usize,
            height: usize,
        }

        let mut nodes = bits
            .chunks(BITS_SIZE_LIMIT)
            .map(|chunk| DynamicBitVectorNodeBuildItem {
                node: Box::new(DynamicBitVectorNode::Leaf {
                    bits: chunk
                        .iter()
                        .enumerate()
                        .fold(BlockType::zero(), |acc, (i, &bit)| {
                            if bit {
                                acc | (BlockType::one() << i)
                            } else {
                                acc
                            }
                        }),
                }),
                total: chunk.len(),
                ones: chunk.iter().filter(|&&b| b).count(),
                height: 0usize,
            })
            .collect::<Vec<DynamicBitVectorNodeBuildItem>>();

        fn merge_nodes(
            left: &DynamicBitVectorNodeBuildItem,
            right: &DynamicBitVectorNodeBuildItem,
        ) -> DynamicBitVectorNodeBuildItem {
            let balance = right.height as isize - left.height as isize;
            debug_assert!(
                (-1..=1).contains(&balance),
                "unbalanced tree detected during build"
            );
            let internal = Box::new(DynamicBitVectorNode::Internal {
                left: left.node.clone(),
                right: right.node.clone(),
                left_total: left.total,
                left_ones: left.ones,
                balance,
            });
            DynamicBitVectorNodeBuildItem {
                node: internal,
                total: left.total + right.total,
                ones: left.ones + right.ones,
                height: max(left.height, right.height) + 1,
            }
        }

        while nodes.len() > 1 {
            let mut next_nodes = (0..nodes.len() - 1)
                .step_by(2)
                .map(|i| merge_nodes(&nodes[i], &nodes[i + 1]))
                .collect::<Vec<DynamicBitVectorNodeBuildItem>>();
            if nodes.len() % 2 == 1 {
                let left = next_nodes.pop().unwrap();
                let right = nodes.last().unwrap();
                next_nodes.push(merge_nodes(&left, right));
            }
            nodes = next_nodes;
        }

        nodes[0].node.as_ref().clone()
    }

    #[inline]
    fn values(&self, len: usize) -> Vec<bool> {
        match self {
            DynamicBitVectorNode::Leaf { bits } => {
                debug_assert!(len <= BlockType::BITS as usize);
                (0..len)
                    .map(|i| ((bits >> i) & BlockType::one()).is_one())
                    .collect()
            }
            DynamicBitVectorNode::Internal {
                left,
                right,
                left_total,
                ..
            } => {
                debug_assert!(*left_total < len);
                [left.values(*left_total), right.values(len - *left_total)].concat()
            }
        }
    }

    ///     D(self)       B(self)
    ///    / \           / \
    ///   B   E    ->   A   D
    ///  / \               / \
    /// A   C             C   E
    ///
    /// returns difference in height after rotation
    #[inline]
    fn rotate_right(&mut self) -> isize {
        let d_node = replace(
            self,
            DynamicBitVectorNode::Leaf {
                bits: BlockType::zero(),
            },
        );

        let (b_node, e_node, b_total, b_ones, d_balance) = match d_node {
            DynamicBitVectorNode::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => (left, right, left_total, left_ones, balance),
            DynamicBitVectorNode::Leaf { .. } => panic!("rotate_right called on Leaf node"),
        };

        let (a_node, c_node, a_total, a_ones, b_balance) = match *b_node {
            DynamicBitVectorNode::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => (left, right, left_total, left_ones, balance),
            DynamicBitVectorNode::Leaf { .. } => panic!("rotate_right: left child is Leaf"),
        };

        let c_height = 2usize;
        let a_height = (c_height as isize - b_balance) as usize;
        let e_height = ((max(a_height, c_height) + 1) as isize + d_balance) as usize;
        let c_total = b_total - a_total;
        let c_ones = b_ones - a_ones;

        let new_d_node = Box::new(DynamicBitVectorNode::Internal {
            left: c_node,
            right: e_node,
            left_total: c_total,
            left_ones: c_ones,
            balance: e_height as isize - c_height as isize,
        });

        let new_b_node = Box::new(DynamicBitVectorNode::Internal {
            left: a_node,
            right: new_d_node,
            left_total: a_total,
            left_ones: a_ones,
            balance: (max(c_height, e_height) + 1) as isize - a_height as isize,
        });

        *self = *new_b_node;

        let old_height = max(max(a_height, c_height) + 1, e_height) + 1;
        let new_height = max(a_height, max(c_height, e_height) + 1) + 1;
        let height_diff = new_height as isize - old_height as isize;
        debug_assert!(
            (-1..=1).contains(&height_diff),
            "rotate_right: invalid height difference"
        );
        height_diff
    }

    ///   b(self)         d(self)
    ///  / \             / \
    /// a   d     ->    b   e
    ///    / \         / \
    ///   c   e       a   c
    ///
    /// returns difference in height after rotation
    #[inline]
    fn rotate_left(&mut self) -> isize {
        let b_node = replace(
            self,
            DynamicBitVectorNode::Leaf {
                bits: BlockType::zero(),
            },
        );

        let (a_node, d_node, a_total, a_ones, b_balance) = match b_node {
            DynamicBitVectorNode::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => (left, right, left_total, left_ones, balance),
            DynamicBitVectorNode::Leaf { .. } => panic!("rotate_left called on Leaf node"),
        };

        let (c_node, e_node, c_total, c_ones, d_balance) = match *d_node {
            DynamicBitVectorNode::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => (left, right, left_total, left_ones, balance),
            DynamicBitVectorNode::Leaf { .. } => panic!("rotate_left: right child is Leaf"),
        };

        let c_height = 2usize;
        let e_height = (c_height as isize + d_balance) as usize;
        let a_height = ((max(c_height, e_height) + 1) as isize - b_balance) as usize;

        let new_b_node = Box::new(DynamicBitVectorNode::Internal {
            left: a_node,
            right: c_node,
            left_total: a_total,
            left_ones: a_ones,
            balance: c_height as isize - a_height as isize,
        });

        let new_d_node = Box::new(DynamicBitVectorNode::Internal {
            left: new_b_node,
            right: e_node,
            left_total: a_total + c_total,
            left_ones: a_ones + c_ones,
            balance: e_height as isize - (max(a_height, c_height) + 1) as isize,
        });

        *self = *new_d_node;

        let old_height = max(a_height, max(c_height, e_height) + 1) + 1;
        let new_height = max(max(a_height, c_height) + 1, e_height) + 1;
        let height_diff = new_height as isize - old_height as isize;
        debug_assert!(
            (-1..=1).contains(&height_diff),
            "rotate_right: invalid height difference"
        );
        height_diff
    }

    /// Rebalances the node if unbalanced.
    /// returns the change in height after rebalancing.
    #[inline]
    fn rebalance(&mut self) -> isize {
        match self {
            DynamicBitVectorNode::Internal {
                left,
                right,
                balance,
                ..
            } => {
                debug_assert!(
                    -2 <= *balance && *balance <= 2,
                    "rebalance called on balanced node"
                );
                if *balance == -2 {
                    if let DynamicBitVectorNode::Internal {
                        balance: left_balance,
                        ..
                    } = **left
                        && left_balance == 1
                    {
                        left.rotate_left();
                    }
                    self.rotate_right()
                } else if *balance == 2 {
                    if let DynamicBitVectorNode::Internal {
                        balance: right_balance,
                        ..
                    } = **right
                        && right_balance == -1
                    {
                        right.rotate_right();
                    }
                    self.rotate_left()
                } else {
                    0isize
                }
            }
            DynamicBitVectorNode::Leaf { .. } => 0isize,
        }
    }

    #[inline]
    fn split_leaf(&mut self, len: usize) {
        match self {
            DynamicBitVectorNode::Leaf { bits } => {
                let left_total = len / 2;
                let left_bits = *bits & ((BlockType::one() << left_total) - BlockType::one());
                let right_bits = *bits >> left_total;
                *self = DynamicBitVectorNode::Internal {
                    left: Box::new(DynamicBitVectorNode::Leaf { bits: left_bits }),
                    right: Box::new(DynamicBitVectorNode::Leaf { bits: right_bits }),
                    left_total,
                    left_ones: left_bits.count_ones() as usize,
                    balance: 0isize,
                };
            }
            DynamicBitVectorNode::Internal { .. } => {
                panic!("split_leaf called on Internal node");
            }
        }
    }

    /// Inserts a bit at the given index in the subtree rooted at this node.
    /// returns the change in height of the subtree after insertion.
    fn insert(&mut self, index: usize, bit: bool, len: usize) -> isize {
        match self {
            DynamicBitVectorNode::Leaf { bits } => {
                let left_bits = *bits & ((BlockType::one() << index) - BlockType::one());
                let right_bits = if index + 1 < BlockType::BITS as usize {
                    (*bits >> index) << (index + 1)
                } else {
                    BlockType::zero()
                };
                let new_bit = if bit {
                    BlockType::one()
                } else {
                    BlockType::zero()
                };
                *bits = left_bits | (new_bit << index) | right_bits;
                if len + 1 >= 2 * BITS_SIZE_LIMIT {
                    self.split_leaf(len + 1);
                    1isize
                } else {
                    0isize
                }
            }
            DynamicBitVectorNode::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => {
                let mut height_diff = 0isize;
                if index < *left_total {
                    let left_height_diff = left.insert(index, bit, *left_total);
                    let old_balance = *balance;
                    if left_height_diff > 0 && old_balance <= 0 {
                        height_diff += left_height_diff;
                    }
                    *left_total += 1;
                    *left_ones += if bit { 1 } else { 0 };
                    *balance -= left_height_diff;
                } else {
                    let right_height_diff =
                        right.insert(index - *left_total, bit, len - *left_total);
                    let old_balance = *balance;
                    if right_height_diff > 0 && old_balance >= 0 {
                        height_diff += right_height_diff;
                    }
                    *balance += right_height_diff;
                }
                height_diff += self.rebalance();
                debug_assert!(
                    (-1..=1).contains(&height_diff),
                    "insert: invalid height difference after rebalancing"
                );
                debug_assert!(
                    {
                        let balance = match self {
                            DynamicBitVectorNode::Internal { balance, .. } => *balance,
                            DynamicBitVectorNode::Leaf { .. } => 0,
                        };
                        (-1..=1).contains(&balance)
                    },
                    "unbalanced tree after insertion",
                );
                height_diff
            }
        }
    }

    #[inline]
    fn insert_batch(
        &mut self,
        mut index: usize,
        bits: BlockType,
        mut len: usize,
        block_len: usize,
    ) {
        let mut node = self;
        while let DynamicBitVectorNode::Internal {
            left,
            right,
            left_total,
            left_ones,
            ..
        } = node
        {
            debug_assert!(
                *left_total < len,
                "insert_batch: left_total should be less than len"
            );
            if index < *left_total {
                len = *left_total;
                *left_total += block_len;
                *left_ones += bits.count_ones() as usize;
                node = left;
            } else {
                index -= *left_total;
                len -= *left_total;
                node = right;
            }
        }
        debug_assert!(
            block_len + len <= BITS_SIZE_LIMIT * 2,
            "insert_batch: block_len too large for leaf"
        );

        let node_bits = match node {
            DynamicBitVectorNode::Leaf { bits } => bits,
            _ => unreachable!("insert_batch: reached non-leaf node"),
        };
        let left_bits = *node_bits & ((BlockType::one() << index) - BlockType::one());
        let right_bits = (*node_bits >> index) << (index + block_len);
        *node_bits = left_bits | (bits << index) | right_bits;
    }

    #[inline]
    fn leftmost_leaf_size(&self, mut len: usize) -> usize {
        let mut node = self;
        while let DynamicBitVectorNode::Internal {
            left, left_total, ..
        } = node
        {
            node = left;
            len = *left_total;
        }

        len
    }

    #[inline]
    fn rightmost_leaf_size(&self, mut len: usize) -> usize {
        let mut node = self;
        while let DynamicBitVectorNode::Internal {
            right, left_total, ..
        } = node
        {
            node = right;
            len -= *left_total;
        }

        len
    }

    /// Removes a bit at the given index in the subtree rooted at this node.
    /// returns the removed bit and the change in height of the subtree after removal.
    fn remove(&mut self, index: usize, mut len: usize) -> (bool, isize) {
        match self {
            DynamicBitVectorNode::Leaf { bits } => {
                let bit = ((*bits >> index) & BlockType::one()).is_one();
                let left_bits = *bits & ((BlockType::one() << index) - BlockType::one());
                let right_bits = (*bits >> (index + 1)) << index;
                *bits = left_bits | right_bits;
                (bit, 0isize)
            }
            DynamicBitVectorNode::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => {
                let mut height_diff = 0isize;
                let removed_bit;
                if index < *left_total {
                    let (bit, left_height_diff) = left.remove(index, *left_total);
                    debug_assert!((-1..=0).contains(&left_height_diff));
                    removed_bit = bit;
                    let old_balance = *balance;
                    if left_height_diff < 0 && old_balance < 0 {
                        height_diff += left_height_diff;
                    }
                    len -= 1;
                    *left_total -= 1;
                    *left_ones -= if bit { 1 } else { 0 };
                    *balance -= left_height_diff;
                    if *left_total <= BITS_SIZE_LIMIT / 2 {
                        if right.leftmost_leaf_size(len - *left_total) <= BITS_SIZE_LIMIT / 2 + 1 {
                            // merge left and right
                            let left_bits = match **left {
                                DynamicBitVectorNode::Leaf { bits } => bits,
                                _ => unreachable!("left child is not Leaf during merge"),
                            };
                            right.insert_batch(0, left_bits, len - *left_total, *left_total);
                            *self = replace(
                                right,
                                DynamicBitVectorNode::Leaf {
                                    bits: BlockType::zero(),
                                },
                            );
                            height_diff = -1isize;
                        } else {
                            // borrow from right
                            let (right_leftmost_bit, _) = right.remove(0, len - *left_total);
                            height_diff +=
                                left.insert(*left_total, right_leftmost_bit, *left_total);
                            *left_total += 1;
                            *left_ones += if right_leftmost_bit { 1 } else { 0 }
                        }
                    }
                } else {
                    let (bit, right_height_diff) =
                        right.remove(index - *left_total, len - *left_total);
                    debug_assert!((-1..=0).contains(&right_height_diff));
                    removed_bit = bit;
                    let old_balance = *balance;
                    if right_height_diff < 0 && old_balance > 0 {
                        height_diff += right_height_diff;
                    }
                    len -= 1;
                    *balance += right_height_diff;
                    if len - *left_total <= BITS_SIZE_LIMIT / 2 {
                        if left.rightmost_leaf_size(*left_total) <= BITS_SIZE_LIMIT / 2 + 1 {
                            // merge left and right
                            let right_bits = match **right {
                                DynamicBitVectorNode::Leaf { bits } => bits,
                                _ => unreachable!("right child is not Leaf during merge"),
                            };
                            left.insert_batch(
                                *left_total,
                                right_bits,
                                *left_total,
                                len - *left_total,
                            );
                            *self = replace(
                                left,
                                DynamicBitVectorNode::Leaf {
                                    bits: BlockType::zero(),
                                },
                            );
                            height_diff = -1isize;
                        } else {
                            // borrow from left
                            let (left_rightmost_bit, _) = left.remove(*left_total - 1, *left_total);
                            height_diff += right.insert(0, left_rightmost_bit, len - *left_total);
                            *left_total -= 1;
                            *left_ones -= if left_rightmost_bit { 1 } else { 0 }
                        }
                    }
                }
                height_diff += self.rebalance();
                debug_assert!(
                    (-1..=1).contains(&height_diff),
                    "remove: invalid height difference after rebalancing"
                );
                debug_assert!(
                    {
                        let balance = match self {
                            DynamicBitVectorNode::Internal { balance, .. } => *balance,
                            DynamicBitVectorNode::Leaf { .. } => 0,
                        };
                        (-1..=1).contains(&balance)
                    },
                    "unbalanced tree after removal",
                );
                (removed_bit, height_diff)
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct DynamicBitVector {
    len: usize,
    ones: usize,
    root: Box<DynamicBitVectorNode>,
}

impl DynamicBitVector {
    pub(super) fn new(bits: &[bool]) -> Self {
        Self {
            len: bits.len(),
            ones: bits.iter().filter(|&&b| b).count(),
            root: Box::new(DynamicBitVectorNode::new(bits)),
        }
    }
}

impl BitVectorTrait for DynamicBitVector {
    #[inline]
    fn values(&self) -> PyResult<Vec<bool>> {
        Ok(self.root.values(self.len))
    }

    #[inline]
    fn access(&self, mut index: usize) -> PyResult<bool> {
        if index >= self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        let mut node = &*self.root;
        while let DynamicBitVectorNode::Internal {
            left,
            right,
            left_total,
            ..
        } = node
        {
            if index < *left_total {
                node = left;
            } else {
                index -= *left_total;
                node = right;
            }
        }

        let bits = match node {
            DynamicBitVectorNode::Leaf { bits } => bits,
            _ => unreachable!("access: reached non-leaf node"),
        };
        let bit = ((bits >> index) & BlockType::one()).is_one();
        Ok(bit)
    }

    #[inline]
    fn rank(&self, bit: bool, mut end: usize) -> PyResult<usize> {
        if end > self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if self.len.is_zero() {
            return Ok(0);
        }
        if !bit {
            return Ok(end - self.rank(true, end)?);
        }

        let mut node = &*self.root;
        let mut rank = 0usize;
        while let DynamicBitVectorNode::Internal {
            left,
            right,
            left_total,
            left_ones,
            ..
        } = node
        {
            if end <= *left_total {
                node = left;
            } else {
                node = right;
                end -= *left_total;
                rank += *left_ones;
            }
        }

        let bits = match node {
            DynamicBitVectorNode::Leaf { bits } => bits,
            _ => unreachable!("access: reached non-leaf node"),
        };
        rank += (*bits & ((1 << end) - BlockType::one())).count_ones() as usize;
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

        let mut node = &*self.root;
        let mut index = 0usize;
        while let DynamicBitVectorNode::Internal {
            left,
            right,
            left_total,
            left_ones,
            ..
        } = node
        {
            let left_count = if bit {
                *left_ones
            } else {
                *left_total - *left_ones
            };
            if kth <= left_count {
                node = left;
            } else {
                node = right;
                index += *left_total;
                kth -= left_count;
            }
        }

        let bits = match node {
            DynamicBitVectorNode::Leaf { bits } => bits,
            _ => unreachable!("select: reached non-leaf node"),
        };
        let offset = bits.bit_select(bit, kth).ok_or(PyRuntimeError::new_err(
            "select: k-th bit not found in leaf",
        ))?;
        Ok(Some(index + offset))
    }
}

impl DynamicBitVectorTrait for DynamicBitVector {
    #[inline]
    fn insert(&mut self, index: usize, bit: bool) -> PyResult<()> {
        if index > self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        self.root.insert(index, bit, self.len);
        self.len += 1;
        self.ones += if bit { 1 } else { 0 };
        Ok(())
    }

    fn remove(&mut self, index: usize) -> PyResult<bool> {
        if index >= self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        let (bit, _) = self.root.remove(index, self.len);
        self.len -= 1;
        self.ones -= if bit { 1 } else { 0 };
        Ok(bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    impl DynamicBitVectorNode {
        fn assert_avl(&self) -> isize {
            match self {
                DynamicBitVectorNode::Leaf { .. } => 0,
                DynamicBitVectorNode::Internal {
                    left,
                    right,
                    balance,
                    ..
                } => {
                    let left_height = left.assert_avl();
                    let right_height = right.assert_avl();
                    let real_balance = right_height - left_height;

                    assert!(
                        (-1..=1).contains(&real_balance),
                        "AVL violation: real balance = {real_balance}",
                    );
                    assert!(
                        *balance == real_balance,
                        "stored balance = {balance}, real balance = {real_balance}"
                    );

                    max(left_height, right_height) + 1
                }
            }
        }
    }

    fn create_dummy() -> DynamicBitVector {
        let bits = vec![true, false, true, true, false, true, false, false].repeat(999);
        DynamicBitVector::new(&bits)
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let mut bv = DynamicBitVector::new(&vec![]);
        bv.root.assert_avl();
        assert_eq!(bv.values().unwrap(), Vec::<bool>::new());
        assert_eq!(
            bv.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(bv.rank(true, 0).unwrap(), 0);
        assert_eq!(bv.rank(false, 0).unwrap(), 0);
        assert_eq!(bv.select(true, 1).unwrap(), None);
        assert_eq!(bv.select(false, 1).unwrap(), None);
        assert_eq!(bv.insert(0, true).unwrap(), ());
        assert_eq!(bv.access(0).unwrap(), true);
        assert_eq!(bv.remove(0).unwrap(), true);
        assert_eq!(
            bv.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
    }

    #[test]
    fn test_exact_block() {
        Python::initialize();

        let bits = vec![true; 1024];
        let bv = DynamicBitVector::new(&bits);
        bv.root.assert_avl();

        for i in 0..1024 {
            assert_eq!(bv.access(i).unwrap(), true);
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
        bv.root.assert_avl();
        assert_eq!(
            bv.values().unwrap(),
            vec![true, false, true, true, false, true, false, false].repeat(999)
        );
    }

    #[test]
    fn test_access() {
        Python::initialize();

        let bv = create_dummy();
        bv.root.assert_avl();

        assert_eq!(bv.access(0).unwrap(), true);
        assert_eq!(bv.access(1001).unwrap(), false);
        assert_eq!(bv.access(2002).unwrap(), true);
        assert_eq!(bv.access(3003).unwrap(), true);
        assert_eq!(bv.access(4004).unwrap(), false);
        assert_eq!(bv.access(5005).unwrap(), true);
        assert_eq!(bv.access(6006).unwrap(), false);
        assert_eq!(bv.access(7007).unwrap(), false);
        assert_eq!(
            bv.access(7992).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
    }

    #[test]
    fn test_rank() {
        Python::initialize();

        let bv = create_dummy();
        bv.root.assert_avl();

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
        bv.root.assert_avl();

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

    #[test]
    fn test_insert() {
        Python::initialize();

        let mut bv = create_dummy();
        bv.root.assert_avl();
        assert_eq!(bv.insert(0, true).unwrap(), ());
        assert_eq!(bv.access(0).unwrap(), true);
        assert_eq!(bv.rank(true, 1).unwrap(), 1);
        assert_eq!(bv.rank(false, 1).unwrap(), 0);
        assert_eq!(bv.select(true, 1).unwrap(), Some(0));
        assert_eq!(bv.select(false, 1).unwrap(), Some(2));
        bv.root.assert_avl();

        assert_eq!(bv.insert(5000, false).unwrap(), ());
        assert_eq!(bv.access(5000).unwrap(), false);
        assert_eq!(bv.rank(true, 5001).unwrap(), 2501);
        assert_eq!(bv.rank(false, 5001).unwrap(), 2500);
        assert_eq!(bv.select(true, 2501).unwrap(), Some(4998));
        assert_eq!(bv.select(false, 2500).unwrap(), Some(5000));
        bv.root.assert_avl();
    }

    #[test]
    fn test_remove() {
        Python::initialize();

        let mut bv = create_dummy();
        bv.root.assert_avl();
        assert_eq!(bv.remove(0).unwrap(), true);
        assert_eq!(bv.access(0).unwrap(), false);
        assert_eq!(bv.rank(true, 1).unwrap(), 0);
        assert_eq!(bv.rank(false, 1).unwrap(), 1);
        assert_eq!(bv.select(true, 1).unwrap(), Some(1));
        assert_eq!(bv.select(false, 1).unwrap(), Some(0));
        bv.root.assert_avl();

        assert_eq!(bv.remove(5000).unwrap(), false);
        assert_eq!(bv.access(5000).unwrap(), true);
        assert_eq!(bv.rank(true, 5001).unwrap(), 2501);
        assert_eq!(bv.rank(false, 5001).unwrap(), 2500);
        assert_eq!(bv.select(true, 2500).unwrap(), Some(4999));
        assert_eq!(bv.select(false, 2501).unwrap(), Some(5002));
        bv.root.assert_avl();
    }

    #[test]
    fn test_insert_remove_values() {
        Python::initialize();

        let mut bv = DynamicBitVector::new(&vec![]);
        bv.root.assert_avl();
        let bits = vec![true, false, true, true, false, true, false, false].repeat(999);

        for (index, &bit) in bits.iter().enumerate() {
            bv.insert(index, bit).unwrap();
            bv.root.assert_avl();
            assert_eq!(bv.access(index).unwrap(), bit);
        }
        assert_eq!(bv.values().unwrap(), bits);

        for &bit in &bits {
            assert_eq!(bv.remove(0).unwrap(), bit);
            bv.root.assert_avl();
        }
        assert_eq!(bv.len, 0);

        for &bit in bits.iter().rev() {
            bv.insert(0, bit).unwrap();
            bv.root.assert_avl();
            assert_eq!(bv.access(0).unwrap(), bit);
        }
        assert_eq!(bv.values().unwrap(), bits);

        for (index, &bit) in bits.iter().enumerate().rev() {
            assert_eq!(bv.remove(index).unwrap(), bit);
            bv.root.assert_avl();
        }
        assert_eq!(bv.len, 0);
    }
}
