use num_traits::Zero;

pub(crate) trait BitSelect {
    /// Select the position of the k-th set bit (1-based index).
    fn bit_select(&self, bit: bool, k: usize) -> Option<usize>;
}

impl BitSelect for u32 {
    fn bit_select(&self, bit: bool, mut k: usize) -> Option<usize> {
        if k.is_zero() {
            return None;
        }

        let value = if bit { *self } else { !*self };
        if value.count_ones() < k as u32 {
            return None;
        }

        let x1 = value - ((value & 0xAAAAAAAAu32) >> 1);
        let x2 = (x1 & 0x33333333u32) + ((x1 >> 2) & 0x33333333u32);
        let x3 = (x2 + (x2 >> 4)) & 0x0F0F0F0Fu32;

        let mut pos = 0;

        loop {
            let cnt = ((x3 >> pos) & 0xFFu32) as usize;
            if k <= cnt {
                break;
            }
            k -= cnt;
            pos += 8;
        }

        let cnt4 = ((x2 >> pos) & 0x0Fu32) as usize;
        if k > cnt4 {
            k -= cnt4;
            pos += 4;
        }

        let cnt2 = ((x1 >> pos) & 0x03u32) as usize;
        if k > cnt2 {
            k -= cnt2;
            pos += 2;
        }

        let bit0 = ((value >> pos) & 1u32) as usize;
        if bit0 < k {
            pos += 1;
        }

        Some(pos)
    }
}

impl BitSelect for u64 {
    fn bit_select(&self, bit: bool, mut k: usize) -> Option<usize> {
        if k.is_zero() {
            return None;
        }

        let value = if bit { *self } else { !*self };
        if value.count_ones() < k as u32 {
            return None;
        }

        let x1 = value - ((value & 0xAAAAAAAAAAAAAAAAu64) >> 1);
        let x2 = (x1 & 0x3333333333333333u64) + ((x1 >> 2) & 0x3333333333333333u64);
        let x3 = (x2 + (x2 >> 4)) & 0x0F0F0F0F0F0F0F0Fu64;

        let mut pos = 0;

        loop {
            let cnt = ((x3 >> pos) & 0xFFu64) as usize;
            if k <= cnt {
                break;
            }
            k -= cnt;
            pos += 8;
        }

        let cnt4 = ((x2 >> pos) & 0x0Fu64) as usize;
        if k > cnt4 {
            k -= cnt4;
            pos += 4;
        }

        let cnt2 = ((x1 >> pos) & 0x03u64) as usize;
        if k > cnt2 {
            k -= cnt2;
            pos += 2;
        }

        let bit0 = ((value >> pos) & 1u64) as usize;
        if bit0 < k {
            pos += 1;
        }

        Some(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_selec_u32() {
        let x = 0b10110010010110001011001001011000u32;
        assert_eq!(x.bit_select(true, 0), None);
        assert_eq!(x.bit_select(true, 6), Some(13));
        assert_eq!(x.bit_select(true, 15), None);
        assert_eq!(x.bit_select(false, 0), None);
        assert_eq!(x.bit_select(false, 15), Some(24));
        assert_eq!(x.bit_select(false, 19), None);
        let y = 0u32;
        assert_eq!(y.bit_select(true, 1), None);
        assert_eq!(y.bit_select(false, 1), Some(0));
    }

    #[test]
    fn test_bit_selec_u64() {
        let x = 0b1011001001011000101100100101100010110010010110001011001001011000u64;
        assert_eq!(x.bit_select(true, 0), None);
        assert_eq!(x.bit_select(true, 20), Some(45));
        assert_eq!(x.bit_select(true, 29), None);
        assert_eq!(x.bit_select(false, 0), None);
        assert_eq!(x.bit_select(false, 25), Some(42));
        assert_eq!(x.bit_select(false, 37), None);
        let y = 0u64;
        assert_eq!(y.bit_select(true, 1), None);
        assert_eq!(y.bit_select(false, 1), Some(0));
    }
}
