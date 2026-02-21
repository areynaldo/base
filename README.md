# MNIST in C

A minimal MNIST neural network implementation in C.

## Overview

This project demonstrates a simple 2-layer MLP (Multi-Layer Perceptron) trained on the MNIST dataset, written in C with minimal dependencies.

## Quick Start

```bash
build.bat
out\mnist.exe
```

## Model Architecture

```
784 (input) -> 128 (ReLU) -> 10 (Softmax)
```

## Data

Place the MNIST data files in the `data/` directory:
- train-images.idx3-ubyte
- train-labels.idx1-ubyte
- t10k-images.idx3-ubyte
- t10k-labels.idx1-ubyte

## License

MIT

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
