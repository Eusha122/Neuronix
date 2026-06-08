# Neuronix

A modern C++20 neural network framework built from absolute scratch — no external math or ML libraries (no Eigen, OpenBLAS, OpenCV, PyTorch, TensorFlow).

## Results

| Task | Model | Accuracy |
|------|-------|----------|
| Image classification | CIFAR-10 CNN | 75.91% |
| Face vs. no-face | LFW binary classifier | 99.72% |
| Digit recognition | MNIST CNN | 99.18% |
| Handwriting A–Z | EMNIST Letters CNN | 91.86% |

## Features

- **Layers** — Dense, Conv2D (im2col), MaxPool2D, BatchNorm, Flatten, Dropout
- **Activations** — ReLU, LeakyReLU, GELU, Sigmoid, Tanh, Softmax
- **Optimizers** — SGD, Adam, AdamW
- **LR Schedulers** — StepLR, CosineAnnealingLR
- **Losses** — CrossEntropyLoss, MSELoss
- **Data loaders** — MNIST, CIFAR-10, EMNIST Letters, LFW faces
- **Augmentation** — RandomHorizontalFlip, RandomCrop, per-channel normalize
- **Model I/O** — save/load to `.nnx` binary format
- **OpenMP** parallel matrix multiply (6× speedup on multi-core)

## Build

```powershell
cmake -S . -B build -DNEURONIX_BUILD_TESTS=ON -DNEURONIX_BUILD_EXAMPLES=ON
cmake --build build --config Release
```

## Examples

```powershell
# MNIST digit recognition (99.18%)
.\build\Release\neuronix_lenet.exe

# CIFAR-10 image classification (75.91%)
.\build\Release\neuronix_cifar10.exe

# Face vs. no-face classifier (99.72%)
.\build\Release\neuronix_face.exe

# Handwriting A-Z recognition (91.86%)
.\build\Release\neuronix_emnist.exe
```

## Dataset Setup

```powershell
# MNIST — place the 4 IDX files in data/mnist/
# CIFAR-10
python scripts/prepare_cifar10.py

# LFW faces
python scripts/prepare_lfw.py

# EMNIST Letters
python scripts/prepare_emnist.py
```

## Test

```powershell
ctest --test-dir build --output-on-failure
```

## Design Constraints

- C++20 throughout (MSVC / GCC / Clang)
- No external math or ML libraries
- Public API only in `include/neuronix/`
- Private implementation only in `src/`
- Catch2 v3 as test framework
