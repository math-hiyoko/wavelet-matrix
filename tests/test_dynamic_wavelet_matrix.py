import pytest

from wavelet_matrix import DynamicWaveletMatrix


class TestDynamicWaveletMatrix:
    """Basic WaveletMatrix tests"""

    wm_small = DynamicWaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
    wm_large = DynamicWaveletMatrix(
        [
            5 << 500,
            4 << 500,
            5 << 500,
            5 << 500,
            2 << 500,
            1 << 500,
            5 << 500,
            6 << 500,
            1 << 500,
            3 << 500,
            5 << 500,
            0 << 500,
        ]
    )

    def test_empty(self):
        """Test DynamicWaveletMatrix with empty data"""
        wv_empty = DynamicWaveletMatrix([])

        assert len(wv_empty) == 0
        assert wv_empty.values() == []
        with pytest.raises(IndexError):
            wv_empty.access(0)
        assert wv_empty.rank(1, 0) == 0
        with pytest.raises(ValueError):
            wv_empty.select(1, 0)
        with pytest.raises(ValueError):
            wv_empty.quantile(0, 0, 0)
        with pytest.raises(ValueError):
            wv_empty.topk(0, 0, 1)
        with pytest.raises(ValueError):
            wv_empty.range_sum(0, 0)
        with pytest.raises(ValueError):
            wv_empty.range_intersection(0, 0, 0, 0)
        with pytest.raises(ValueError):
            wv_empty.range_freq(0, 0)
        with pytest.raises(ValueError):
            wv_empty.range_list(0, 0)
        with pytest.raises(ValueError):
            wv_empty.range_maxk(0, 0)
        with pytest.raises(ValueError):
            wv_empty.range_mink(0, 0)
        with pytest.raises(ValueError):
            wv_empty.prev_value(0, 0)
        with pytest.raises(ValueError):
            wv_empty.next_value(0, 0)
        with pytest.raises(ValueError):
            wv_empty.insert(0, 10)
        with pytest.raises(IndexError):
            wv_empty.update(0, 20)
        with pytest.raises(IndexError):
            wv_empty.remove(0)

    def test_all_zero(self):
        """Test DynamicWaveletMatrix with all zero elements"""
        wv_all_zero = DynamicWaveletMatrix([0] * 128, max_bit=8)

        assert len(wv_all_zero) == 128
        assert wv_all_zero.values() == [0] * 128
        assert wv_all_zero.rank(0, 1) == 1
        assert wv_all_zero.select(0, 1) == 0
        assert wv_all_zero.quantile(0, 10, 1) == 0
        assert wv_all_zero.topk(0, 10, 1) == [{"value": 0, "count": 10}]
        assert wv_all_zero.range_sum(0, 10) == 0
        assert wv_all_zero.range_intersection(0, 10, 5, 15) == [
            {"value": 0, "count1": 10, "count2": 10}
        ]
        assert wv_all_zero.range_freq(0, 10) == 10
        assert wv_all_zero.range_list(0, 10) == [{"value": 0, "count": 10}]
        assert wv_all_zero.range_maxk(0, 10, 1) == [{"value": 0, "count": 10}]
        assert wv_all_zero.range_mink(0, 10, 1) == [{"value": 0, "count": 10}]
        assert wv_all_zero.prev_value(0, 10) == 0
        assert wv_all_zero.next_value(0, 10) == 0

        wv_all_zero.insert(64, 10)
        assert len(wv_all_zero) == 129
        assert wv_all_zero.access(64) == 10
        wv_all_zero.update(64, 20)
        assert wv_all_zero.access(64) == 20
        removed_value = wv_all_zero.remove(64)
        assert removed_value == 20
        assert len(wv_all_zero) == 128

    def test_values(self):
        """Test values method"""
        assert self.wm_small.values() == [5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        assert self.wm_large.values() == [
            5 << 500,
            4 << 500,
            5 << 500,
            5 << 500,
            2 << 500,
            1 << 500,
            5 << 500,
            6 << 500,
            1 << 500,
            3 << 500,
            5 << 500,
            0 << 500,
        ]

    def test_access(self):
        """Test access method"""
        assert self.wm_small.access(6) == 5
        with pytest.raises(IndexError):
            self.wm_small.access(12)

        assert self.wm_large.access(6) == 5 << 500
        with pytest.raises(IndexError):
            self.wm_large.access(12)

    def test_rank(self):
        """Test rank method"""
        assert self.wm_small.rank(5, 8) == 4
        print(self.wm_small)
        assert self.wm_small.rank(10, 8) == 0
        with pytest.raises(IndexError):
            self.wm_small.rank(5, 13)

        assert self.wm_large.rank(5 << 500, 8) == 4
        assert self.wm_large.rank(10 << 500, 8) == 0
        with pytest.raises(IndexError):
            self.wm_large.rank(5 << 500, 13)

    def test_select(self):
        """Test select method"""
        assert self.wm_small.select(5, 4) == 6
        assert self.wm_small.select(5, 6) is None

        assert self.wm_large.select(5 << 500, 4) == 6
        assert self.wm_large.select(5 << 500, 6) is None

    def test_quantile(self):
        """Test quantile method"""
        assert self.wm_small.quantile(2, 12, 8) == 5
        with pytest.raises(ValueError):
            self.wm_small.quantile(2, 12, 13)

        assert self.wm_large.quantile(2, 12, 8) == 5 << 500
        with pytest.raises(ValueError):
            self.wm_large.quantile(2, 12, 13)

    def test_topk(self):
        """Test topk method"""
        assert self.wm_small.topk(1, 10, 2) == [{"value": 5, "count": 3}, {"value": 1, "count": 2}]
        with pytest.raises(IndexError):
            self.wm_small.topk(1, 13, 20)

        assert self.wm_large.topk(1, 10, 2) == [
            {"value": 5 << 500, "count": 3},
            {"value": 1 << 500, "count": 2},
        ]
        with pytest.raises(IndexError):
            self.wm_large.topk(1, 13, 20)

    def test_range_sum(self):
        """Test range_sum method"""
        assert self.wm_small.range_sum(2, 8) == 24
        with pytest.raises(IndexError):
            self.wm_small.range_sum(1, 13)

        assert self.wm_large.range_sum(2, 8) == 24 << 500
        with pytest.raises(IndexError):
            self.wm_large.range_sum(1, 13)

    def test_range_intersection(self):
        """Test range_intersection method"""
        assert self.wm_small.range_intersection(0, 6, 6, 11) == [
            {"value": 1, "count1": 1, "count2": 1},
            {"value": 5, "count1": 3, "count2": 2},
        ]
        with pytest.raises(IndexError):
            self.wm_small.range_intersection(0, 6, 4, 13)

        assert self.wm_large.range_intersection(0, 6, 6, 11) == [
            {"value": 1 << 500, "count1": 1, "count2": 1},
            {"value": 5 << 500, "count1": 3, "count2": 2},
        ]
        with pytest.raises(IndexError):
            self.wm_large.range_intersection(0, 6, 4, 13)

    def test_range_freq(self):
        """Test range_freq method"""
        assert self.wm_small.range_freq(1, 9, 4, 6) == 4
        with pytest.raises(IndexError):
            self.wm_small.range_freq(0, 13, 2, 5)

        assert self.wm_large.range_freq(1, 9, 4 << 500, 6 << 500) == 4
        with pytest.raises(IndexError):
            self.wm_large.range_freq(0, 13, 2 << 500, 5 << 500)

    def test_range_list(self):
        """Test range_list method"""
        assert self.wm_small.range_list(1, 9, 4, 6) == [
            {"value": 4, "count": 1},
            {"value": 5, "count": 3},
        ]
        with pytest.raises(IndexError):
            self.wm_small.range_list(0, 13, 0, 5)

        assert self.wm_large.range_list(1, 9, 4 << 500, 6 << 500) == [
            {"value": 4 << 500, "count": 1},
            {"value": 5 << 500, "count": 3},
        ]
        with pytest.raises(IndexError):
            self.wm_large.range_list(0, 13, 0 << 500, 5 << 500)

    def test_range_maxk(self):
        """Test range_maxk method"""
        assert self.wm_small.range_maxk(1, 9, 2) == [
            {"value": 6, "count": 1},
            {"value": 5, "count": 3},
        ]
        with pytest.raises(IndexError):
            self.wm_small.range_maxk(0, 13, 20)

        assert self.wm_large.range_maxk(1, 9, 2) == [
            {"value": 6 << 500, "count": 1},
            {"value": 5 << 500, "count": 3},
        ]
        with pytest.raises(IndexError):
            self.wm_large.range_maxk(0, 13, 20)

    def test_range_mink(self):
        """Test range_mink method"""
        assert self.wm_small.range_mink(1, 9, 2) == [
            {"value": 1, "count": 2},
            {"value": 2, "count": 1},
        ]
        with pytest.raises(IndexError):
            self.wm_small.range_mink(0, 13, 20)

        assert self.wm_large.range_mink(1, 9, 2) == [
            {"value": 1 << 500, "count": 2},
            {"value": 2 << 500, "count": 1},
        ]
        with pytest.raises(IndexError):
            self.wm_large.range_mink(0, 13, 20)

    def test_prev_value(self):
        """Test prev_value method"""
        assert self.wm_small.prev_value(1, 9, 7) == 6
        assert self.wm_small.prev_value(1, 10, 1) is None
        with pytest.raises(IndexError):
            self.wm_small.prev_value(0, 13)

        assert self.wm_large.prev_value(1, 9, 7 << 500) == 6 << 500
        assert self.wm_large.prev_value(1, 10, 1 << 500) is None
        with pytest.raises(IndexError):
            self.wm_large.prev_value(0, 13)

    def test_next_value(self):
        """Test next_value method"""
        assert self.wm_small.next_value(1, 9, 3) == 4
        with pytest.raises(IndexError):
            self.wm_small.next_value(0, 13)

        assert self.wm_large.next_value(1, 9, 3 << 500) == 4 << 500
        with pytest.raises(IndexError):
            self.wm_large.next_value(0, 13)

    def test_max_bit(self):
        """Test DynamicWaveletMatrix with max_bit parameter"""
        assert self.wm_small.max_bit() == 3

        assert self.wm_large.max_bit() == 503

    def test_insert_update_remove(self):
        """Test insert, update, and remove methods"""
        data = list(range(10000))
        wv = DynamicWaveletMatrix(data, max_bit=14)

        data.insert(2, 10)
        wv.insert(2, 10)
        assert wv.values() == data

        old_value = wv.update(3, 20)
        assert old_value == data[3]
        data[3] = 20
        assert wv.values() == data

        removed_value = wv.remove(4)
        assert removed_value == data.pop(4)
        assert wv.values() == data
