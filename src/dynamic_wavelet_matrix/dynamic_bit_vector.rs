use std::{
    cmp::max,
    mem::replace,
};
use num_traits::{One, Zero};
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
    PyResult,
};

use crate::traits::{
    bit_select::BitSelect,
    bit_vector::BitVectorTrait,
    dynamic_bit_vector::DynamicBitVectorTrait,
};

type BlockType = u64;
const BITS_SIZE_LIMIT: usize = BlockType::BITS as usize / 2;

#[derive(Debug, Clone)]
enum Node {
    Internal {
        left: Box<Node>,
        right:Box<Node>,
        left_total: usize,
        left_ones: usize,
        balance: i8,
    },
    Leaf {
        bits: BlockType,
    },
}

impl Node {
    fn new_leaf(bits: BlockType) -> Self {
        Node::Leaf { bits }
    }

    fn new_internal(
        left: Box<Node>,
        right: Box<Node>,
        left_total: usize,
        left_ones: usize,
        balance: i8,
    ) -> Self {
        Node::Internal {
            left,
            right,
            left_total,
            left_ones,
            balance,
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
    fn rotate_right(&mut self) -> i8 {
        let d_node = replace(self, Node::Leaf { bits: BlockType::zero() });

        let (b_node, e_node, b_total, b_ones, d_balance) = match d_node {
            Node::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => (left, right, left_total, left_ones, balance),
            Node::Leaf { .. } => panic!("rotate_right called on Leaf node"),
        };

        let (a_node, c_node, a_total, a_ones, b_balance) = match *b_node {
            Node::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => (left, right, left_total, left_ones, balance),
            Node::Leaf { .. } => panic!("rotate_right: left child is Leaf"),
        };

        let c_height = 2usize;
        let a_height = (c_height as i8 - b_balance) as usize;
        let e_height = ((max(a_height, c_height) + 1) as i8 + d_balance) as usize;
        let c_total = b_total - a_total;
        let c_ones = b_ones - a_ones;

        let new_d_node = Box::new(Node::new_internal(
            c_node,
            e_node,
            c_total,
            c_ones,
            e_height as i8 - c_height as i8,
        ));

        let new_b_node = Box::new(Node::new_internal(
            a_node,
            new_d_node,
            a_total,
            a_ones,
            (max(c_height, e_height) + 1) as i8 - a_height as i8,
        ));

        *self = *new_b_node;

        let old_height = max(max(a_height, c_height) + 1, e_height) + 1;
        let new_height = max(a_height, max(c_height, e_height) + 1) + 1;
        let height_diff = new_height as i8 - old_height as i8;
        debug_assert!(-1 <= height_diff && height_diff <= 1, "rotate_right: invalid height difference");
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
    fn rotate_left(&mut self) -> i8 {
        let b_node = replace(self, Node::Leaf { bits: BlockType::zero() });

        let (a_node, d_node, a_total, a_ones, b_balance) = match b_node {
            Node::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => (left, right, left_total, left_ones, balance),
            Node::Leaf { .. } => panic!("rotate_left called on Leaf node"),
        };

        let (c_node, e_node, c_total, c_ones, d_balance) = match *d_node {
            Node::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => (left, right, left_total, left_ones, balance),
            Node::Leaf { .. } => panic!("rotate_left: right child is Leaf"),
        };

        let c_height = 2usize;
        let e_height = (c_height as i8 + d_balance) as usize;
        let a_height = ((max(c_height, e_height) + 1) as i8 - b_balance) as usize;

        let new_b_node = Box::new(Node::new_internal(
            a_node,
            c_node,
            a_total,
            a_ones,
            c_height as i8 - a_height as i8,
        ));

        let new_d_node = Box::new(Node::new_internal(
            new_b_node,
            e_node,
            a_total + c_total,
            a_ones + c_ones,
            e_height as i8 - (max(a_height, c_height) + 1) as i8,
        ));

        *self = *new_d_node;

        let old_height = max(a_height, max(c_height, e_height) + 1) + 1;
        let new_height = max(max(a_height, c_height) + 1, e_height) + 1;
        let height_diff = new_height as i8 - old_height as i8;
        debug_assert!(-1 <= height_diff && height_diff <= 1, "rotate_right: invalid height difference");
        height_diff
    }

    /// Rebalances the node if unbalanced.
    /// returns the change in height after rebalancing.
    #[inline]
    fn rebalance(&mut self) -> i8 {
        match self {
            Node::Internal { 
                left,
                right,
                balance,
                ..
            } => {
                debug_assert!(-2 <= *balance && *balance <= 2, "rebalance called on balanced node");
                if *balance == -2 {
                    if let Node::Internal { balance: left_balance, .. } = **left && left_balance ==1 {
                        left.rotate_left()
                    } else {
                        self.rotate_right()
                    }
                } else if *balance == 2 {
                    if let Node::Internal { balance: right_balance, .. } = **right && right_balance == -1 {
                        return right.rotate_right();
                    } else {
                        return self.rotate_left();
                    }
                } else {
                    0i8
                }
            },
            Node::Leaf { .. } => {
                0i8
            },
        }
    }

    #[inline]
    fn split_leaf(&mut self, len: usize) -> () {
        match self {
            Node::Leaf { bits } => {
                let left_total = len / 2;
                let left_bits = *bits & ((BlockType::one() << left_total) - BlockType::one());
                let right_bits = *bits >> left_total;
                *self = Node::Internal {
                    left: Box::new(Node::Leaf { bits: left_bits }),
                    right: Box::new(Node::Leaf { bits: right_bits }),
                    left_total,
                    left_ones: left_bits.count_ones() as usize,
                    balance: 0i8,
                };
            },
            Node::Internal { .. } => {
                panic!("split_leaf called on Internal node");
            },
        }
    }

    /// Inserts a bit at the given index in the subtree rooted at this node.
    /// returns the change in height of the subtree after insertion.
    fn insert(&mut self, index: usize, bit: bool, len: usize) -> i8 {
        match self {
            Node::Leaf { bits } => {
                let left_bits = *bits & ((BlockType::one() << index) - BlockType::one());
                let right_bits = if index + 1 < BlockType::BITS as usize {
                    (*bits >> index) << (index + 1)
                } else {
                    BlockType::zero()
                };
                let new_bit = if bit { BlockType::one() } else { BlockType::zero() };
                *bits = left_bits | (new_bit << index) | right_bits;
                if len + 1 >= 2 * BITS_SIZE_LIMIT {
                    self.split_leaf(len + 1);
                    1i8
                } else {
                    0i8
                }
            },
            Node::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => {
                let mut height_diff = 0i8;
                if index < *left_total {
                    let left_height_diff = left.insert(index, bit, *left_total);
                    let old_balance = *balance;
                    if left_height_diff > 0 && old_balance < 0 {
                        height_diff += left_height_diff;
                    }
                    *left_total += 1;
                    *left_ones += if bit { 1 } else { 0 };
                    *balance -= left_height_diff;
                } else {
                    let right_height_diff = right.insert(index - *left_total, bit, len - *left_total);
                    let old_balance = *balance;
                    if right_height_diff > 0 && old_balance > 0 {
                        height_diff += right_height_diff;
                    }
                    *balance += right_height_diff;
                }
                height_diff += self.rebalance();
                debug_assert!(-1 <= height_diff && height_diff <= 1, "insert: invalid height difference after rebalancing");
                debug_assert!(
                    {
                        let balance = match self {
                            Node::Internal { balance, .. } => *balance,
                            Node::Leaf { .. } => 0,
                        };
                        -1 <= balance && balance <= 1
                    },
                    "unbalanced tree after insertion",
                );
                height_diff
            },
        }
    }

    #[inline]
    fn insert_batch(&mut self, mut index: usize, bits: BlockType, mut len: usize, block_len: usize) -> () {
        let mut node = self;
        while let Node::Internal { left, right, left_total, left_ones, .. } = node {
            debug_assert!(*left_total < len, "insert_batch: left_total should be less than len");
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
        debug_assert!(block_len + len <= BITS_SIZE_LIMIT * 2, "insert_batch: block_len too large for leaf");

        let node_bits = match node {
            Node::Leaf { bits } => bits,
            _ => unreachable!("insert_batch: reached non-leaf node"),
        };
        let left_bits = *node_bits & ((BlockType::one() << index) - BlockType::one());
        let right_bits = (*node_bits >> index) << (index + block_len);
        *node_bits = left_bits | (bits << index) | right_bits;
    }

    #[inline]
    fn leftmost_leaf_size(&self, mut len: usize) -> usize {
        let mut node = self;
        while let Node::Internal { left, left_total, .. } = node {
            node = left;
            len = *left_total;
        }

        len
    }

    #[inline]
    fn rightmost_leaf_size(&self, mut len: usize) -> usize {
        let mut node = self;
        while let Node::Internal { right, left_total, .. } = node {
            node = right;
            len -= *left_total;
        }

        len
    }

    /// Removes a bit at the given index in the subtree rooted at this node.
    /// returns the removed bit and the change in height of the subtree after removal.
    fn remove(&mut self, index: usize, mut len: usize) -> (bool, i8) {
        match self {
            Node::Leaf { bits } => {
                let bit = ((*bits >> index) & BlockType::one()).is_one();
                let left_bits = *bits & ((BlockType::one() << index) - BlockType::one());
                let right_bits = (*bits >> (index + 1)) << index;
                *bits = left_bits | right_bits;
                (bit, 0i8)
            },
            Node::Internal {
                left,
                right,
                left_total,
                left_ones,
                balance,
            } => {
                let mut height_diff = 0i8;
                let removed_bit;
                if index < *left_total {
                    let (bit, left_height_diff) = left.remove(index, *left_total);
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
                                Node::Leaf { bits } => bits,
                                _ => unreachable!("left child is not Leaf during merge"),
                            };
                            right.insert_batch(
                                0,
                                left_bits,
                                len - *left_total,
                                *left_total,
                            );
                            *self = replace(right, Node::Leaf { bits: BlockType::zero() });
                            height_diff = -1i8;
                        } else {
                            // borrow from right
                            let (right_leftmost_bit, _) = right.remove(0, len - *left_total);
                            height_diff += left.insert(*left_total, right_leftmost_bit, *left_total);
                            *left_total += 1;
                            *left_ones += if right_leftmost_bit { 1 } else { 0 }
                        }
                    }
                } else {
                    let (bit, right_height_diff) = right.remove(index - *left_total, len - *left_total);
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
                                Node::Leaf { bits } => bits,
                                _ => unreachable!("right child is not Leaf during merge"),
                            };
                            left.insert_batch(
                                *left_total,
                                right_bits,
                                *left_total,
                                len - *left_total,
                            );
                            *self = replace(left, Node::Leaf { bits: BlockType::zero() });
                            height_diff = -1i8;
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
                debug_assert!(-1 <= height_diff && height_diff <= 1, "remove: invalid height difference after rebalancing");
                debug_assert!(
                    {
                        let balance = match self {
                            Node::Internal { balance, .. } => *balance,
                            Node::Leaf { .. } => 0,
                        };
                        -1 <= balance && balance <= 1
                    },
                    "unbalanced tree after removal",
                );
                (removed_bit, height_diff)
            },
        }
    }
}

pub(crate) struct DynamicBitVector {
    len: usize,
    ones: usize,
    root: Box<Node>,
}

impl DynamicBitVector {
    pub(super) fn new(bits: &Vec<bool>) -> Self {
        if bits.is_empty() {
            return Self {
                len: 0,
                ones: 0,
                root: Node::new_leaf(BlockType::zero()).into(),
            };
        }

        struct NodeBuildItem {
            node: Box<Node>,
            total: usize,
            ones: usize,
            height: u8,
        }

        let mut nodes = bits
            .chunks(BITS_SIZE_LIMIT)
            .map(|chunk| {
                NodeBuildItem {
                    node: Box::new(
                        Node::new_leaf(
                            chunk
                                .iter()
                                .enumerate()
                                .fold(BlockType::zero(), |acc, (i, &bit)| {
                                    if bit {
                                        acc | (BlockType::one() << i)
                                    } else {
                                        acc
                                    }
                                }),
                        )
                    ),
                    total: chunk.len(),
                    ones: chunk.iter().filter(|&&b| b).count(),
                    height: 0u8,
                }
            })
            .collect::<Vec<NodeBuildItem>>();

        fn merge_nodes(left: &NodeBuildItem, right: &NodeBuildItem) -> NodeBuildItem {
            let balance = right.height as i8 - left.height as i8;
            debug_assert!(-1 <= balance && balance <= 1, "unbalanced tree detected during build");
            let internal = Box::new(Node::new_internal(
                left.node.clone(),
                right.node.clone(),
                left.total,
                left.ones,
                balance,
            ));
            NodeBuildItem {
                node: internal,
                total: left.total + right.total,
                ones: left.ones + right.ones,
                height: max(left.height, right.height) + 1,
            }
        }

        while nodes.len() > 1 {
            let mut next_nodes = (0..nodes.len() - 1).step_by(2)
                .map(|i| merge_nodes(&nodes[i], &nodes[i + 1]))
                .collect::<Vec<NodeBuildItem>>();
            if nodes.len() % 2 == 1 {
                let left = next_nodes.pop().unwrap();
                let right = nodes.last().unwrap();
                next_nodes.push(merge_nodes(&left, right));
            }
            nodes = next_nodes;
        }

        Self {
            len: bits.len(),
            ones: nodes[0].ones,
            root: nodes[0].node.clone(),
        }
    }
}

impl BitVectorTrait for DynamicBitVector {
    #[inline]
    fn access(&self, mut index: usize) -> PyResult<bool> {
        if index >= self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        let mut node = &*self.root;
        while let Node::Internal { left, right, left_total, .. } = node {
            if index < *left_total {
                node = left;
            } else {
                index -= *left_total;
                node = right;
            }
        }

        let bits = match node {
            Node::Leaf { bits } => bits,
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
        while let Node::Internal { left, right, left_total, left_ones, .. } = node {
            if end <= *left_total {
                node = left;
            } else {
                node = right;
                end -= *left_total;
                rank += *left_ones;
            }
        }

        let bits = match node {
            Node::Leaf { bits } => bits,
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
        while let Node::Internal { left, right, left_total, left_ones, .. } = node {
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
            Node::Leaf { bits } => bits,
            _ => unreachable!("select: reached non-leaf node"),
        };
        let offset = bits
            .bit_select(bit, kth)
            .ok_or(PyRuntimeError::new_err("select: k-th bit not found in leaf"))?;
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

    fn create_dummy() -> DynamicBitVector {
        let bits = vec![true, false, true, true, false, true, false, false].repeat(999);
        DynamicBitVector::new(&bits)
    }

    #[test]
    fn test_empty() {
        let mut bv = DynamicBitVector::new(&vec![]);
        assert_eq!(bv.access(0).unwrap_err().to_string(), "IndexError: index out of bounds");
        assert_eq!(bv.rank(true, 0).unwrap(), 0);
        assert_eq!(bv.rank(false, 0).unwrap(), 0);
        assert_eq!(bv.select(true, 1).unwrap(), None);
        assert_eq!(bv.select(false, 1).unwrap(), None);
        assert_eq!(bv.insert(0, true).unwrap(), ());
        assert_eq!(bv.access(0).unwrap(), true);
        assert_eq!(bv.remove(0).unwrap(), true);
        assert_eq!(bv.access(0).unwrap_err().to_string(), "IndexError: index out of bounds");
    }

    #[test]
    fn test_exact_block() {
        let bits = vec![true; 1024];
        let bv = DynamicBitVector::new(&bits);

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
        assert_eq!(
            bv.access(7992).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
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

    #[test]
    fn test_insert() {
        let mut bv = create_dummy();
        assert_eq!(bv.insert(0, true).unwrap(), ());
        assert_eq!(bv.access(0).unwrap(), true);
        assert_eq!(bv.rank(true, 1).unwrap(), 1);
        assert_eq!(bv.rank(false, 1).unwrap(), 0);
        assert_eq!(bv.select(true, 1).unwrap(), Some(0));
        assert_eq!(bv.select(false, 1).unwrap(), Some(2));

        assert_eq!(bv.insert(5000, false).unwrap(), ());
        assert_eq!(bv.access(5000).unwrap(), false);
        assert_eq!(bv.rank(true, 5001).unwrap(), 2501);
        assert_eq!(bv.rank(false, 5001).unwrap(), 2500);
        assert_eq!(bv.select(true, 2501).unwrap(), Some(4998));
        assert_eq!(bv.select(false, 2500).unwrap(), Some(5000));
    }

    #[test]
    fn test_remove() {
        let mut bv = create_dummy();
        assert_eq!(bv.remove(0).unwrap(), true);
        assert_eq!(bv.access(0).unwrap(), false);
        assert_eq!(bv.rank(true, 1).unwrap(), 0);
        assert_eq!(bv.rank(false, 1).unwrap(), 1);
        assert_eq!(bv.select(true, 1).unwrap(), Some(1));
        assert_eq!(bv.select(false, 1).unwrap(), Some(0));

        assert_eq!(bv.remove(5000).unwrap(), false);
        assert_eq!(bv.access(5000).unwrap(), true);
        assert_eq!(bv.rank(true, 5001).unwrap(), 2501);
        assert_eq!(bv.rank(false, 5001).unwrap(), 2500);
        assert_eq!(bv.select(true, 2500).unwrap(), Some(4999));
        assert_eq!(bv.select(false, 2501).unwrap(), Some(5002));
    }

    #[test]
    fn test_insert_remove_values() {
        let mut bv = DynamicBitVector::new(&vec![]);
        let bits = vec![true, false, true, true, false, true, false, false].repeat(999);

        for (index, &bit) in bits.iter().enumerate() {
            bv.insert(index, bit).unwrap();
            assert_eq!(bv.access(index).unwrap(), bit);
        }
        assert_eq!(bv.len, bits.len());

        for &bit in &bits {
            assert_eq!(bv.remove(0).unwrap(), bit);
        }
        assert_eq!(bv.len, 0);

        for &bit in bits.iter().rev() {
            bv.insert(0, bit).unwrap();
            assert_eq!(bv.access(0).unwrap(), bit);
        }
        assert_eq!(bv.len, bits.len());

        for (index, &bit) in bits.iter().enumerate().rev() {
            assert_eq!(bv.remove(index).unwrap(), bit);
        }
        assert_eq!(bv.len, 0);
    }
}