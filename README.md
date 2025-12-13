# Wavelet Matrix

High-performance indexed sequence structure powered by Rust, providing fast rank/select, top-k, quantile, and range queries with optional dynamic updates.

- Document: https://math-hiyoko.github.io/wavelet-matrix
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

#### List top-k highest frequent values (topk)
```python
>>> # List values in [1, 10) with the top-2 highest frequencies.
>>> wm.topk(start=1, end=10, k=2)
[{'value': 5, 'count': 3}, {'value': 1, 'count': 2}]
```

#### Sum values in a range (range_sum)
```python
>>> # Sum of elements in the range [2, 8).
>>> wm.range_sum(start=2, end=8)
24
```

#### List intersection of two ranges (range_intersection)
```python
>>> # List the intersection of two ranges [0, 6) and [6, 11).
>>> wm.range_intersection(start1=0, end1=6, start2=6, end2=11)
[{'value': 1, 'count1': 1, 'count2': 1}, {'value': 5, 'count1': 3, 'count2': 2}]
```

#### Count values in a range (range_freq)
```python
>>> # Count values c in the range [1, 9) such that 4 <= c < 6.
>>> wm.range_freq(start=1, end=9, lower=4, upper=6)
4
```

#### List values in a range (range_list)
```python
>>> # List values c in the range [1, 9) such that 4 <= c < 6.
>>> wm.range_list(start=1, end=9, lower=4, upper=6)
[{'value': 4, 'count': 1}, {'value': 5, 'count': 3}]
```

#### List top-k maximum values (range_maxk)
```python
>>> # List values in [1, 9) with the top-2 maximum values.
>>> wm.range_maxk(start=1, end=9, k=2)
[{'value': 6, 'count': 1}, {'value': 5, 'count': 3}]
```

#### List top-k minimum values (range_mink)
```python
>>> # List values in [1, 9) with the top-2 minimum values.
>>> wm.range_mink(start=1, end=9, k=2)
[{'value': 1, 'count': 2}, {'value': 2, 'count': 1}]
```

#### Get the maximun value (prev_value)
```python
>>> # Get the maximum value c in the range [1, 9) such that c < 7.
>>> wm.prev_value(start=1, end=9, upper=7)
6
```

#### Get the minimun value (next_value)
```python
>>> # Get the minimum value c in the range [1, 9) such that 4 <= c.
>>> wm.next_value(start=1, end=9, lower=4)
4
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
>>> # The bit width of the new value must not exceed max_bit.
>>> dwm.insert(index=4, value=8)
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 8, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
```

#### Remove value (remove)
```python
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 8, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
>>> # Remove the value at index 4.
>>> dwm.remove(index=4)
8
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
```

#### Update value (update)
```python
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
>>> # Update the value at index 4 to 5
>>> # The bit width of the new value must not exceed max_bit.
>>> dwm.update(index=4, value=5)
2
>>> dwm
DynamicWaveletMatrix([5, 4, 5, 5, 5, 1, 5, 6, 1, 3, 5, 0], max_bit=4)
```

## Development

### Running Tests

```bash
$ pip install -e ".[dev]"

# Cargo test
$ cargo test --all --release

# Run tests
$ pytest --benchmark-skip

# Run benchmarks
$ pytest --benchmark-only
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
