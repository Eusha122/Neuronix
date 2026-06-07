# Neuronix

Neuronix is a modern C++20 neural network framework built from absolute scratch.

The long-term goal is to grow into a modular AI engine that can support dense neural networks, tensors, CNNs, image classification, face authentication, model serialization, GPU acceleration, and production inference pipelines.

Phase 1 is foundation only. It intentionally does not implement Matrix, Tensor, layers, optimizers, losses, serialization formats, or training behavior.

## Build

```powershell
cmake -S . -B build -DNEURONIX_BUILD_TESTS=ON
cmake --build build
```

## Test

```powershell
ctest --test-dir build --output-on-failure
```

## Optional Targets

```powershell
cmake -S . -B build -DNEURONIX_BUILD_EXAMPLES=ON
cmake -S . -B build -DNEURONIX_BUILD_BENCHMARKS=ON
```

## Current Status

- C++20 CMake project foundation
- Public/private source layout
- Catch2 smoke test
- GitHub Actions configure/build/test workflow
- Example executable
- Benchmark scaffold
- Architecture and roadmap documentation

## Non-Goals For Phase 1

- No TensorFlow, PyTorch, Keras, or external machine learning frameworks
- No external math libraries
- No Matrix or Tensor implementation yet
- No training, inference, serialization, logging, image, or GPU implementation yet
