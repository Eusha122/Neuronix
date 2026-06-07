# Phase 4: Dense Networks and Early Model API

Phase 4 introduces the layer abstraction, the `Dense` layer, four activation functions, and the `Model` class that chains them into a forward-pass pipeline.

## Included

- `neuronix::Layer` — abstract base in `include/neuronix/layers/layer.hpp`
- `neuronix::Dense` — fully-connected layer in `include/neuronix/layers/dense.hpp` + `src/layers/dense.cpp`
- `neuronix::ReLU`, `Sigmoid`, `Tanh` — header-only activations in `include/neuronix/activations/`
- `neuronix::Softmax` — non-element-wise activation in `include/neuronix/activations/softmax.hpp` + `src/activations/softmax.cpp`
- `neuronix::Model` — sequential model in `include/neuronix/models/model.hpp` + `src/models/model.cpp`

## Design

### Layer

```cpp
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) const = 0;
    virtual std::string_view name() const noexcept = 0;
    virtual std::size_t param_count() const noexcept { return 0; }
};
```

All layers are `const`-forward for Phase 4. Backpropagation (Phase 5) will introduce a separate training path.

### Dense

| Property | Detail |
|---|---|
| Weights shape | `(out_features, in_features)` |
| Bias shape | `(out_features, 1)` |
| Weight init | He normal |
| Bias init | Zero |
| Supports batching | Yes — input `(in, batch)` → output `(out, batch)` |

Forward pass:

```
output = W * input + b   (bias broadcast across all batch columns)
```

### Activations

| Class | Formula | Header-only |
|---|---|---|
| `ReLU` | `max(0, x)` | Yes |
| `Sigmoid` | `1 / (1 + exp(-x))` | Yes |
| `Tanh` | `tanh(x)` | Yes |
| `Softmax` | `exp(x - max) / Σexp(x - max)` per column | No |

Softmax applies numerically-stable per-column softmax, supporting batched input.

### Model

```cpp
Model model;
model.add<Dense>(784, 128);
model.add<ReLU>();
model.add<Dense>(128, 10);
model.add<Softmax>();

Matrix output = model.predict(input);
model.summary();
```

`add<L>(args...)` constructs the layer in-place. `add(std::unique_ptr<Layer>)` also accepted for pre-built layers with custom weights.

## Public API

### Layer (abstract)

| Method | Description |
|---|---|
| `forward(input)` | Returns layer output (const) |
| `name()` | Layer type name |
| `param_count()` | Trainable parameter count (default 0) |

### Dense

| Method | Description |
|---|---|
| `Dense(in, out)` | He-normal weights, zero bias |
| `forward(input)` | Matrix multiply + bias broadcast |
| `set_weights(w)` | Replace weights (shape-checked) |
| `set_bias(b)` | Replace bias (shape-checked) |
| `weights() / bias()` | Read-only access |
| `in_features() / out_features()` | Dimensions |
| `param_count()` | `out * in + out` |

### Model

| Method | Description |
|---|---|
| `add<L>(args...)` | Construct and append layer |
| `add(unique_ptr<Layer>)` | Append pre-built layer |
| `predict(input)` | Run full forward pass |
| `depth()` | Number of layers |
| `total_params()` | Sum of all layer param counts |
| `summary()` | Print layer table to stdout |

## Error Handling

| Condition | Exception |
|---|---|
| `Dense::forward` input rows != in_features | `std::invalid_argument` |
| `Dense::set_weights` wrong shape | `std::invalid_argument` |
| `Dense::set_bias` wrong shape | `std::invalid_argument` |

## Acceptance Criteria

- All 181 tests pass (126 existing + 16 dense + 27 activation + 12 model).
- `Model::predict` chains any sequence of layers.
- Softmax output per column sums to 1.0 within 1e-9.
- CI passes on MSVC/Windows.
