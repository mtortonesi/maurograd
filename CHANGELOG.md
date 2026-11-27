# Changelog

All notable changes to Maurograd are documented here. The project follows
[Semantic Versioning](https://semver.org/); before 1.0, a minor version bump
(0.1 → 0.2) may include breaking changes.

The `maurograd` and `maurograd-mlx` gems are released together, at the same
version.

## [0.1.0] - 2026-11-27

First public release.

### Added

- Tensors with reverse-mode automatic differentiation over arbitrary
  computation graphs (gradients from multiple paths are summed).
- Ops: addition, subtraction and multiplication with broadcasting, power,
  sum, 2D matrix product, linear, 2D convolution (stride, padding), 2D max
  pooling, ReLU, 2D batch normalization, dropout, flatten, and axis
  permutation (`Tensor#transpose`, `Tensor#permute`).
- Layers: `Conv2D`, `Linear`, `MaxPool2D`, `ReLU`, `BatchNorm2D`,
  `Dropout`, `Flatten`, `Sequential`, with `train`/`eval` modes.
- Losses: cross-entropy with logits (class indices or soft targets), MSE,
  binary cross-entropy.
- Optimizers: SGD with momentum and weight decay; Adam.
- MNIST download and loading, and shuffled mini-batching.
- `Maurograd.seed!` for reproducible runs.
- Compute backends behind one interface: `:cpu` (Numo plus C kernels, with
  Apple's Accelerate on macOS and OpenBLAS elsewhere) and `:mlx` (the Apple
  Silicon GPU through MLX, in the separate `maurograd-mlx` gem). Select
  with `MAUROGRAD_BACKEND`, `Maurograd.backend =` or `model.to`.
- Examples: LeNet on MNIST, and LeNet with and without BatchNorm.

### Defaults worth knowing

- BLAS/OpenMP thread pools default to one thread unless
  `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` are set.
- The MLX backend computes float32 matrix products at full precision
  (`MLX_ENABLE_TF32=0`) unless the variable is set.

### Known limitations

- `MatMul` supports 2D matrices only (no batched products).
- No module base class, general reshape, `no_grad` mode or model
  saving/loading yet.
- No pure-Ruby fallback: the C extension must build.

[0.1.0]: https://github.com/mtortonesi/maurograd/releases/tag/v0.1.0
