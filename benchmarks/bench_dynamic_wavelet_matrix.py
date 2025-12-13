import copy
import random

import pytest

from wavelet_matrix import DynamicWaveletMatrix


@pytest.fixture
def random_data(size: int, max_bit: int):
    """Helper function to create random data"""
    random.seed(42)
    base = [random.randint(0, (1 << max_bit) - 1) for _ in range(size // 100)]
    return base * 100


@pytest.fixture
def random_dynamic_wavelet_matrix(random_data: list[int], max_bit: int) -> DynamicWaveletMatrix:
    """Helper function to create a DynamicWaveletMatrix with random data"""
    return DynamicWaveletMatrix(random_data, max_bit)


@pytest.mark.parametrize("size", [500, 10000, 200000])
@pytest.mark.parametrize("max_bit", [8, 32, 128])
class BenchDynamicWaveletMatrix:
    def bench_dynamic_construction(self, benchmark, random_data, max_bit):
        """Benchmark DynamicWaveletMatrix construction"""
        benchmark(DynamicWaveletMatrix, random_data, max_bit)

    def bench_dynamic_values(self, benchmark, random_dynamic_wavelet_matrix):
        """Benchmark DynamicWaveletMatrix values retrieval"""
        benchmark(random_dynamic_wavelet_matrix.values)

    def bench_dynamic_access(self, benchmark, random_dynamic_wavelet_matrix, size):
        """Benchmark DynamicWaveletMatrix access"""
        index = random.randint(0, size - 1)
        benchmark(random_dynamic_wavelet_matrix.access, index)

    def bench_dynamic_rank(self, benchmark, random_dynamic_wavelet_matrix, size):
        """Benchmark DynamicWaveletMatrix rank"""
        value = random_dynamic_wavelet_matrix[random.randint(0, size - 1)]
        end = random.randint(0, size)
        benchmark(random_dynamic_wavelet_matrix.rank, value, end)

    def bench_dynamic_select(self, benchmark, random_dynamic_wavelet_matrix, size):
        """Benchmark DynamicWaveletMatrix select"""
        value = random_dynamic_wavelet_matrix[random.randint(0, size - 1)]
        kth = random_dynamic_wavelet_matrix.rank(value, size)
        benchmark(random_dynamic_wavelet_matrix.select, value, kth)

    def bench_dynamic_quantile(self, benchmark, random_dynamic_wavelet_matrix, size):
        """Benchmark DynamicWaveletMatrix quantile"""
        start = size // 4
        end = size * 3 // 4
        kth = random.randint(1, end - start)
        benchmark(random_dynamic_wavelet_matrix.quantile, start, end, kth)

    def bench_dynamic_range_freq(self, benchmark, random_dynamic_wavelet_matrix, size, max_bit):
        """Benchmark DynamicWaveletMatrix range_freq"""
        start = size // 4
        end = size * 3 // 4
        lower = (1 << max_bit) // 4
        upper = (1 << max_bit) * 3 // 4
        benchmark(random_dynamic_wavelet_matrix.range_freq, start, end, lower, upper)

    def bench_dynamic_range_maxk(self, benchmark, random_dynamic_wavelet_matrix, size):
        """Benchmark DynamicWaveletMatrix range_maxk"""
        start = size // 4
        end = size * 3 // 4
        k = 10
        benchmark(random_dynamic_wavelet_matrix.range_maxk, start, end, k)

    def bench_dynamic_range_mink(self, benchmark, random_dynamic_wavelet_matrix, size):
        """Benchmark DynamicWaveletMatrix range_mink"""
        start = size // 4
        end = size * 3 // 4
        k = 10
        benchmark(random_dynamic_wavelet_matrix.range_mink, start, end, k)

    def bench_dynamic_prev_value(self, benchmark, random_dynamic_wavelet_matrix, size, max_bit):
        """Benchmark DynamicWaveletMatrix prev_value"""
        start = size // 4
        end = size * 3 // 4
        upper = 1 << (max_bit - 1)
        benchmark(random_dynamic_wavelet_matrix.prev_value, start, end, upper)

    def bench_dynamic_next_value(self, benchmark, random_dynamic_wavelet_matrix, size, max_bit):
        """Benchmark DynamicWaveletMatrix next_value"""
        start = size // 4
        end = size * 3 // 4
        lower = 1 << (max_bit - 1)
        benchmark(random_dynamic_wavelet_matrix.next_value, start, end, lower)

    def bench_dynamic_insert(self, benchmark, random_dynamic_wavelet_matrix, size, max_bit):
        """Benchmark DynamicWaveletMatrix insert"""
        index = size // 2
        value = random.randint(0, (1 << max_bit) - 1)

        def insert():
            wm = copy.copy(random_dynamic_wavelet_matrix)
            wm.insert(index, value)

        benchmark(insert)

    def bench_dynamic_remove(self, benchmark, random_dynamic_wavelet_matrix, size, max_bit):
        """Benchmark DynamicWaveletMatrix remove"""
        index = size // 2

        def remove():
            wm = copy.copy(random_dynamic_wavelet_matrix)
            wm.remove(index)

        benchmark(remove)

    def bench_dynamic_update(self, benchmark, random_dynamic_wavelet_matrix, size, max_bit):
        """Benchmark DynamicWaveletMatrix update"""
        index = size // 2
        value = random.randint(0, (1 << max_bit) - 1)

        def update():
            wm = copy.copy(random_dynamic_wavelet_matrix)
            wm.update(index, value)

        benchmark(update)
