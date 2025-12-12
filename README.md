# Wavelet Matrix

High-performance indexed sequence structure powered by Rust, providing fast rank/select, top-k, quantile, and range queries with optional dynamic updates.

- Document: https://math-hiyoko.github.io/wavelet-matrix/
- Repository: https://github.com/math-hiyoko/wavelet-matrix


## Quick Start

### WaveletMatrix
```python
>>> from wavelet_matrix import WaveletMatrix
>>>
>>> # Create a WaveletMatrix
>>> data = [5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
>>> wm = WaveletMatrix(data)
>>> wm
WaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0])
```

#### Count occurrences (rank)
```python
>>> # Count of 5 in the range [0, 9)
>>> wm.rank(value=5, end=9)
4
```

#### Find position (select)
```python
>>> # Find the index of 4th occurrence of value 5
>>> wm.select(value=5, kth=4)
6
```

#### Find k-th smallest (quantile)
```python
>>> # Find 8th smallest value in the range [2, 12)
>>> wm.quantile(start=2, end=12, kth=8)
5
```

#### Count values in a range (range_freq)
```python
>>> # Count values c in the range [1, 9) such that 4 <= c < 6.
>>> wm.range_freq(start=1, end=9, lower=4, upper=6)
4
```

#### List top-k maximum values (range_maxk)
```python
>>> # List values in [1, 9) with the top-2 maximum values.
>>> wm.range_maxk(start=1, end=9, k=2)
[{'value': 6, 'count': 1}, {'value': 5, 'count': 3}]
```

### Dynamic Wavelet Matrix
```python
>>> from wavelet_matrix import DynamicWaveletMatrix
>>>
>>> # Create a DynamicWaveletMatrix
>>> data = [5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
>>> dwm = DynamicWaveletMatrix(data, max_bit=4)
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
```

#### Insert value (insert)
```python
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
>>> # Inserts 8 at index 4.
>>> dwm.insert(index=4, value=8)
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 8, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
```

#### Remove value (remove)
```python
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 8, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
>>> # Remove a value at index 4.
>>> dwm.remove(index=4)
8
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
```

#### Update value (update)
```python
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
>>> # Update a value at index 4 to 5
>>> dwm.update(index=4, value=5)
2
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 5, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run benchmarks
pytest benchmarks/
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
