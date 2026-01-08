# Wavelet Matrix

[![CI](https://github.com/math-hiyoko/wavelet-matrix/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/math-hiyoko/wavelet-matrix/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/math-hiyoko/wavelet-matrix/graph/badge.svg?token=TXBR7MF2CP)](https://codecov.io/gh/math-hiyoko/wavelet-matrix)
![PyPI - Version](https://img.shields.io/pypi/v/wavelet-matrix)
![PyPI - License](https://img.shields.io/pypi/l/wavelet-matrix)
![PyPI - PythonVersion](https://img.shields.io/pypi/pyversions/wavelet-matrix)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/wavelet-matrix)
![PyPI - Types](https://img.shields.io/pypi/types/wavelet-matrix)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/wavelet-matrix?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=PyPI%20downloads)](https://pepy.tech/projects/wavelet-matrix)
![PyPI - Format](https://img.shields.io/pypi/format/wavelet-matrix)
![Rust](https://img.shields.io/badge/powered%20by-Rust-orange)
![Unsafe](https://img.shields.io/badge/unsafe-0-success)


High-performance Wavelet Matrix implementation powered by Rust,  
supporting fast rank / select / range queries over indexed sequences  

- PyPI: https://pypi.org/project/wavelet-matrix
- Document: https://math-hiyoko.github.io/wavelet-matrix
- Repository: https://github.com/math-hiyoko/wavelet-matrix

## Features:
- Fast rank, select, quantile
- Rich range queries (freq / sum / top-k / min / max)
- Optional dynamic updates (insert / remove / update)
- Safe Rust (no unsafe)

## Installation
```bash
pip install wavelet-matrix
```

## WaveletMatrix
WaveletMatrix indexes a static sequence of integers,  
enabling fast queries where runtime depends on bit-width, not data size.
```python
from wavelet_matrix import WaveletMatrix

data = [5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
wm = WaveletMatrix(data)
```

### Frequency Queries
#### Count occurrences (rank)
```python
wm.rank(value=5, end=9)
# 4
```

#### Find position (select)
```python
wm.select(value=5, kth=4)
# 6
```

### Order Statistics
#### k-th smallest value (quantile)
```python
wm.quantile(start=2, end=12, kth=8)
# 5
```

### Range Aggregation
#### Sum values (range_sum)
```python
wm.range_sum(start=2, end=8)
# 24
```

#### Count values in [lower, upper) (range_freq)
```python
wm.range_freq(start=1, end=9, lower=4, upper=6)
# 4
```

#### List values with counts (range_list)
```python
wm.range_list(start=1, end=9, lower=4, upper=6)
# [{'value': 4, 'count': 1}, {'value': 5, 'count': 3}]
```

### Top-K Queries
#### Most frequent values (topk)
```python
wm.topk(start=1, end=10, k=2)
# [{'value': 5, 'count': 3}, {'value': 1, 'count': 2}]
```

#### Extreme values (range_maxk / range_mink)
```python
wm.range_maxk(start=1, end=9, k=2)
# [{'value': 6, 'count': 1}, {'value': 5, 'count': 3}]
wm.range_mink(start=1, end=9, k=2)
# [{'value': 1, 'count': 2}, {'value': 2, 'count': 1}]
```

### Boundary Queries
```python
wm.prev_value(start=1, end=9, upper=7)
# 6
wm.next_value(start=1, end=9, lower=4)
# 4
```

## DynamicWaveletMatrix
DynamicWaveletMatrix supports mutable sequences with insert/remove/update.

### Trade-off:
- Higher overhead
- Values must fit within max_bit

```python
from wavelet_matrix import DynamicWaveletMatrix

dwm = DynamicWaveletMatrix(data, max_bit=4)
```

### Insert
```python
dwm.insert(index=4, value=8)
```

#### Remove
```python
dwm.remove(index=4)
```

#### Update
```python
dwm.update(index=4, value=5)
# or dwm[4] = 5
```

## Safety
- Powered by safe Rust
- Memory-safe by design

## Development

### Running Tests

```bash
pip install -e ".[test]"
cargo test --all --release
pytest
```

### Formating Code
```bash
pip install -e ".[dev]"
cargo fmt
ruff format
```

### Generating Docs
```bash
pdoc wavelet_matrix \
      --output-directory docs \
      --no-search \
      --no-show-source \
      --docformat markdown \
      --footer-text "© 2026 Koki Watanabe"
```

## References

- Francisco Claude, Gonzalo Navarro, Alberto Ordóñez,  
  The wavelet matrix: An efficient wavelet tree for large alphabets,  
  Information Systems,  
  Volume 47,  
  2015,  
  Pages 15-32,  
  ISSN 0306-4379,  
  https://doi.org/10.1016/j.is.2014.06.002.  
