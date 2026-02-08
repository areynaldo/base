# Base

A minimal C base library for personal projects.

## Overview

Base provides essential building blocks for C projects: memory management, common types, and utilities. Designed to be simple, dependency-free, and easy to drop into any project.

## Modules

### Core (`base.h`)
- Arena-based memory allocation
- Common type definitions (float32_t, etc.)
- String utilities
- Error handling
- Byte manipulation (endian swap, etc.)

### Tensor (`tensor.h`)
- N-dimensional tensors (float32)
- Element-wise operations
- Activations (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (MSE, Cross-Entropy)
- Linear layers with backprop
- SGD optimizer

## Quick Start

```bash
build.bat
out\mnist.exe
```

## Examples

### MNIST
A 2-layer MLP trained on MNIST demonstrating the tensor module:

```
784 (input) -> 128 (ReLU) -> 10 (Softmax)
```

**Results:** ~98% accuracy after 5 epochs.

Download MNIST from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and extract to `data/`.

## Structure

```
base.h            - Core library
tensor.h          - Tensor/NN module
examples/
  mnist.c         - MNIST training example
```

## License

MIT
