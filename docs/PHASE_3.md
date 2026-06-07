# Phase 3: Tensor Engine

Phase 3 delivers a fully tested N-dimensional `Tensor` type and a shared RNG used by both Matrix and Tensor factories.

## Included

- `neuronix::Tensor` class in `include/neuronix/core/tensor.hpp`
- Private implementation in `src/core/tensor.cpp`
- Shared inline RNG in `src/core/rng_impl.hpp` (not installed)
- Comprehensive unit tests in `tests/tensor_tests.cpp`

## Design

| Property | Detail |
|---|---|
| Storage | Row-major flat `std::vector<double>` |
| Indexing | Row-major strides precomputed at construction |
| Rank | Any N >= 0 (rank 1 = vector, rank 2 = matrix-like, rank 3/4 = CNN tensors) |
| Shape | `std::vector<std::size_t>` aliased as `Tensor::Shape` |

### Stride computation

For shape `{d0, d1, d2}`, strides are `{d1*d2, d2, 1}`.  
Element `(i, j, k)` maps to flat index `i * d1*d2 + j * d2 + k`.

## Public API

### Construction

| Expression | Description |
|---|---|
| `Tensor(Shape)` | Zero-initialized |
| `Tensor(Shape, value)` | Fill with scalar |
| `Tensor(Shape, {v0, v1, ...})` | From initializer_list |
| `Tensor(Shape, std::vector<double>)` | From vector (moved) |

### Static Factories

| Factory | Description |
|---|---|
| `Tensor::zeros(shape)` | All zeros |
| `Tensor::ones(shape)` | All ones |
| `Tensor::random_uniform(shape, low, high)` | Uniform random in [low, high] |
| `Tensor::random_normal(shape, mean, stddev)` | Normal distribution |
| `Tensor::from_matrix(m)` | Rank-2 tensor from Matrix |

### Element Access

| Expression | Description |
|---|---|
| `t[i]` | Flat linear index (unchecked) |
| `t(i, j, k, ...)` | Multi-dim index (unchecked, variadic template) |
| `t.at({i, j, k})` | Multi-dim index (bounds-checked, throws `std::out_of_range`) |

### Arithmetic

- Element-wise: `+`, `-`, `+=`, `-=` (shape must match)
- Scalar: `*`, `/`, `*=`, `/=`, unary `-`
- Left scalar: `scalar * tensor`
- Hadamard (element-wise) product: `hadamard(other)`

### Shape Operations

| Method | Description |
|---|---|
| `reshape(new_shape)` | New view with same total elements |
| `flatten()` | Reshape to 1D |

### Reductions

- `sum()`, `mean()`, `min()`, `max()`, `frobenius_norm()`
- `mean()`, `min()`, `max()` throw `std::logic_error` on empty tensor

### Apply

- `apply(func)` — returns new tensor with element-wise function applied

### Matrix Interop

| Method | Description |
|---|---|
| `to_matrix()` | Convert rank-2 tensor to Matrix (throws if rank != 2) |
| `Tensor::from_matrix(m)` | Convert Matrix to rank-2 tensor |

## Shared RNG

`src/core/rng_impl.hpp` exposes `neuronix::detail::global_rng()` as an `inline` function, ensuring a single shared `std::mt19937_64` across all translation units. The public `seed_random(uint64_t)` declared in `matrix.hpp` seeds this shared RNG, so seeding once controls both Matrix and Tensor random factories.

## Error Handling

| Condition | Exception |
|---|---|
| Data size mismatch at construction | `std::invalid_argument` |
| Shape mismatch in arithmetic | `std::invalid_argument` |
| Wrong index count in `at()` | `std::invalid_argument` |
| Index out of range in `at()` | `std::out_of_range` |
| `dim()` axis out of range | `std::out_of_range` |
| `reshape` element count mismatch | `std::invalid_argument` |
| `to_matrix` on non-rank-2 tensor | `std::invalid_argument` |
| Reduction on empty tensor | `std::logic_error` |
| Division by zero | `std::invalid_argument` |

## Acceptance Criteria

- All tensor tests pass alongside all matrix and smoke tests.
- CI workflow (GCC on Ubuntu, MSVC on Windows) passes.
- Layers and models may now depend on Tensor.
