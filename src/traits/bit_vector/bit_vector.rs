use num_traits::Zero;
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyValueError},
};

pub(crate) trait BitVectorTrait {
    /// Get all bit values as a vector.
    fn values(&self) -> PyResult<Vec<bool>>;

    /// Get the bit value at the specified position.
    fn access(&self, index: usize) -> PyResult<bool>;

    /// Count the number of bit in the range [0, end).
    fn rank(&self, bit: bool, end: usize) -> PyResult<usize>;

    /// Find the position of the k-th bit (1-indexed).
    fn select(&self, bit: bool, kth: usize) -> PyResult<Option<usize>>;
}

#[allow(dead_code)]
pub(in crate::traits) struct SampleBitVector(Vec<bool>);

#[allow(dead_code)]
impl SampleBitVector {
    pub(in crate::traits) fn new(data: Vec<bool>) -> Self {
        SampleBitVector(data)
    }
}

impl BitVectorTrait for SampleBitVector {
    fn values(&self) -> PyResult<Vec<bool>> {
        Ok(self.0.clone())
    }

    fn access(&self, index: usize) -> PyResult<bool> {
        if index >= self.0.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        Ok(self.0[index])
    }

    fn rank(&self, bit: bool, end: usize) -> PyResult<usize> {
        if end > self.0.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        let count = self.0[..end].iter().filter(|&&b| b == bit).count();
        Ok(count)
    }

    fn select(&self, bit: bool, kth: usize) -> PyResult<Option<usize>> {
        if kth.is_zero() {
            return Err(PyValueError::new_err("kth must be greater than 0"));
        }
        let index = self
            .0
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b == bit)
            .nth(kth - 1)
            .map(|(i, _)| i);
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn create_dummy() -> SampleBitVector {
        let bits = vec![true, false, true, true, false, true, false, false].repeat(999);
        SampleBitVector::new(bits)
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let bv = SampleBitVector::new(vec![]);

        assert_eq!(bv.values().unwrap(), Vec::<bool>::new());
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
    fn test_values() {
        Python::initialize();

        let bv = create_dummy();
        assert_eq!(
            bv.values().unwrap(),
            vec![true, false, true, true, false, true, false, false].repeat(999)
        );
    }

    #[test]
    fn test_access() {
        Python::initialize();

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
