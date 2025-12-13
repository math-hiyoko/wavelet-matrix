import random

import pytest

from wavelet_matrix import WaveletMatrix


@pytest.fixture
def random_data(size: int, max_bit: int):
    """Helper function to create random data"""
    random.seed(42)
    base = [random.randint(0, (1 << max_bit) - 1) for _ in range(size // 100)]
    return base * 100


@pytest.fixture
def random_wavelet_matrix(random_data: list[int]) -> WaveletMatrix:
    """Helper function to create a WaveletMatrix with random data"""
    return WaveletMatrix(random_data)


@pytest.mark.parametrize("size", [500, 10000, 200000])
@pytest.mark.parametrize("max_bit", [8, 32, 128])
class BenchWaveletMatrix:
    def bench_construction(self, benchmark, random_data):
        """Benchmark WaveletMatrix construction"""
        benchmark(WaveletMatrix, random_data)

    def bench_values(self, benchmark, random_wavelet_matrix):
        """Benchmark WaveletMatrix values retrieval"""
        benchmark(random_wavelet_matrix.values)

    def bench_access(self, benchmark, random_wavelet_matrix, size):
        """Benchmark WaveletMatrix access"""
        index = random.randint(0, size - 1)
        benchmark(random_wavelet_matrix.access, index)

    def bench_rank(self, benchmark, random_wavelet_matrix, size):
        """Benchmark WaveletMatrix rank"""
        value = random_wavelet_matrix[random.randint(0, size - 1)]
        end = random.randint(0, size)
        benchmark(random_wavelet_matrix.rank, value, end)

    def bench_select(self, benchmark, random_wavelet_matrix, size):
        """Benchmark WaveletMatrix select"""
        value = random_wavelet_matrix[random.randint(0, size - 1)]
        kth = random_wavelet_matrix.rank(value, size)
        benchmark(random_wavelet_matrix.select, value, kth)

    def bench_quantile(self, benchmark, random_wavelet_matrix, size):
        """Benchmark WaveletMatrix quantile"""
        start = size // 4
        end = size * 3 // 4
        kth = random.randint(1, end - start)
        benchmark(random_wavelet_matrix.quantile, start, end, kth)

    def bench_range_freq(self, benchmark, random_wavelet_matrix, size, max_bit):
        """Benchmark WaveletMatrix range_freq"""
        start = size // 4
        end = size * 3 // 4
        lower = (1 << max_bit) // 4
        upper = (1 << max_bit) * 3 // 4
        benchmark(random_wavelet_matrix.range_freq, start, end, lower, upper)

    def bench_range_maxk(self, benchmark, random_wavelet_matrix, size):
        """Benchmark WaveletMatrix range_maxk"""
        start = size // 4
        end = size * 3 // 4
        k = 10
        benchmark(random_wavelet_matrix.range_maxk, start, end, k)

    def bench_range_mink(self, benchmark, random_wavelet_matrix, size):
        """Benchmark WaveletMatrix range_mink"""
        start = size // 4
        end = size * 3 // 4
        k = 10
        benchmark(random_wavelet_matrix.range_mink, start, end, k)

    def bench_prev_value(self, benchmark, random_wavelet_matrix, size, max_bit):
        """Benchmark WaveletMatrix prev_value"""
        start = size // 4
        end = size * 3 // 4
        upper = 1 << (max_bit - 1)
        benchmark(random_wavelet_matrix.prev_value, start, end, upper)

    def bench_next_value(self, benchmark, random_wavelet_matrix, size, max_bit):
        """Benchmark WaveletMatrix next_value"""
        start = size // 4
        end = size * 3 // 4
        lower = 1 << (max_bit - 1)
        benchmark(random_wavelet_matrix.next_value, start, end, lower)
