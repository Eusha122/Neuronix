# Phase 2: Matrix Engine

Phase 2 delivers a fully tested, production-grade `Matrix` type that all later systems will depend on.

## Included

- `neuronix::Matrix` class in `include/neuronix/core/matrix.hpp`
- Private implementation in `src/core/matrix.cpp`
- Row-major `std::vector<double>` storage
- Comprehensive unit tests in `tests/matrix_tests.cpp`

## Public API

### Construction

| Expression | Description |
|---|---|
| `Matrix(rows, cols)` | Zero-initialized |
| `Matrix(rows, cols, value)` | Fill with scalar |
| `Matrix(rows, cols, {v0, v1, ...})` | From initializer_list |
| `Matrix(rows, cols, std::vector<double>)` | From vector (moved) |

### Static Factories

| Factory | Description |
|---|---|
| `Matrix::zeros(r, c)` | All zeros |
| `Matrix::ones(r, c)` | All ones |
| `Matrix::identity(n)` | n×n identity |
| `Matrix::random_uniform(r, c, low, high)` | Uniform random in [low, high] |
| `Matrix::random_normal(r, c, mean, stddev)` | Normal distribution |
| `Matrix::xavier_uniform(r, c)` | Glorot uniform: limit = sqrt(6/(r+c)) |
| `Matrix::xavier_normal(r, c)` | Glorot normal: stddev = sqrt(2/(r+c)) |
| `Matrix::he_uniform(r, c)` | He uniform: limit = sqrt(6/r) |
| `Matrix::he_normal(r, c)` | He normal: stddev = sqrt(2/r) |

### Arithmetic

- Element-wise: `+`, `-`, `+=`, `-=` (dimension-checked)
- Scalar: `*`, `/`, `*=`, `/=`, unary `-`
- Left scalar: `scalar * matrix` (free function)
- Matrix multiplication: `A * B` (not element-wise; checks `A.cols == B.rows`)
- Hadamard product: `A.hadamard(B)` (element-wise matrix multiply)
- Transpose: `A.transpose()`

### Reductions

- `sum()`, `mean()`, `min()`, `max()`, `frobenius_norm()`
- `mean()`, `min()`, `max()` throw `std::logic_error` on empty matrix

### Apply

- `apply(func)` — returns new matrix with element-wise function applied

### Comparison

- `operator==`, `operator!=` — exact equality
- `approx_equal(other, tol=1e-9)` — floating-point safe comparison

### RNG Control

- `seed_random(uint64_t)` — seeds the global RNG for reproducible tests

## Error Handling

| Condition | Exception |
|---|---|
| Initializer size mismatch | `std::invalid_argument` |
| Element-wise dimension mismatch | `std::invalid_argument` |
| Matmul dimension mismatch | `std::invalid_argument` |
| Division by zero | `std::invalid_argument` |
| Out-of-range `at()` | `std::out_of_range` |
| Reduction on empty matrix | `std::logic_error` |

## Test Coverage

Tests verify:
- All construction forms and error cases
- All static factories (shape, range, reproducibility)
- Element access (operator() and at())
- All arithmetic operations and their dimension checks
- Matrix multiplication correctness and identity property
- Hadamard product
- Transpose and its double-inverse property
- All reductions including empty-matrix edge cases
- `apply` with identity, ReLU, and square functions
- Comparison operators and approx_equal tolerance
- Mathematical properties: A·I=A, (A+B)ᵀ=Aᵀ+Bᵀ, (AB)ᵀ=BᵀAᵀ, k(A+B)=kA+kB
- Seeded RNG reproducibility

## Excluded

- Sparse matrix formats
- SIMD or BLAS acceleration (planned for later)
- Complex number support
- Row/column slice views
- Broadcasting

## Acceptance Criteria

- All matrix tests pass through CTest.
- CI workflow (GCC on Ubuntu, MSVC on Windows) passes.
- Higher-level modules (layers, models) may now depend on Matrix.
