# Neuronix Roadmap

## Phase 1: Foundation

Create the C++20/CMake project foundation, CI, tests, examples, docs, and scaffolding.

## Phase 2: Matrix Engine

Implement a fully tested dynamic `Matrix` type with arithmetic, multiplication, transpose, dot products, and random initialization.

Matrix quality is critical because dense networks, backpropagation, CNNs, and face authentication will depend on it.

## Phase 3: Tensor Engine

Introduce Tensor support earlier than originally planned so later CNN work has the right foundation.

## Phase 4: Dense Networks And Early Model API

Establish the early user-facing composition shape:

```cpp
Model model;
model.add(Dense(...));
model.add(ReLU());
model.add(Dense(...));
model.train(...);
```

## Phase 5: Backpropagation And Optimizers

Implement gradient flow, chain rule mechanics, and parameter updates.

## Phase 6: MNIST

Add dataset loading, batching, evaluation metrics, and model persistence hooks.

## Phase 7: CNNs

Implement convolution, pooling, padding, stride, and feature maps using the Tensor foundation.

## Later Phases

- CIFAR-10 image classification
- Face authentication
- Production serialization, logging, profiling, and benchmarking
- GPU acceleration with OpenCL or oneAPI research
