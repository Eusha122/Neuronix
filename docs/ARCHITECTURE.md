# Neuronix Architecture

Neuronix is organized as a small framework core that grows outward into layers, models, training, inference, serialization, and production tools.

## Directory Layout

```text
include/neuronix/    Public headers installed or consumed by users
src/                 Private implementation files
tests/               Catch2 unit tests
examples/            Small executables that demonstrate public APIs
benchmarks/          Performance experiments and comparisons
docs/                Design notes, roadmap, and phase documentation
cmake/               Future reusable CMake helpers
```

Public API headers belong under `include/neuronix/...`. Internal implementation details belong under `src/...`.

## Dependency Direction

Future code should flow inward:

```text
applications/examples
      |
    models
      |
   training
      |
    layers
      |
 core math/tensor
```

Higher-level modules may depend on lower-level modules. Lower-level modules must not depend on higher-level modules.

## Module Responsibilities

- `core`: Matrix, Tensor, random number generation, numeric utilities, and low-level math.
- `layers`: Dense, convolution, pooling, flattening, and future trainable/non-trainable layers.
- `activations`: ReLU, Sigmoid, Softmax, and activation interfaces.
- `losses`: Mean squared error, cross entropy, and loss interfaces.
- `optimizers`: SGD, Adam, and parameter update strategies.
- `models`: User-facing model composition and training/inference orchestration.
- `serialization`: Future `.nx` model and weight persistence.
- `logging`: Future diagnostics for training, inference, benchmarking, and deployment.

## Phase 1 Boundary

Phase 1 creates the buildable foundation only. It does not define the long-term public ML APIs prematurely.
