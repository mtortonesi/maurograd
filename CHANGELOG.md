# Changelog

All notable changes to Maurograd are documented here. The project follows
[Semantic Versioning](https://semver.org/); before 1.0, a minor version bump
(0.1 → 0.2) may include breaking changes.

The `maurograd` and `maurograd-mlx` gems are released together, at the same
version.

## [Unreleased]

### Added

- `Tensor#reshape` (`Ops::Reshape`), differentiable, with one dimension
  inferable as `-1`, e.g. `x.reshape(n, -1)`.
- `Layers::Module`, a base class for layers and models: it finds
  parameters and sublayers in instance variables (also inside Arrays and
  Hashes), and provides `parameters`, `named_parameters`, `children`,
  `modules`, `train`/`eval`/`training?`, `to`, `call` (so `model.call(x)`
  works like `forward`) and a short `inspect`. Shared parameters are listed once.
- Differentiable elementwise functions `Tensor#exp`, `#log`, `#sigmoid`,
  `#tanh` and `#silu`, and the activation layers `Layers::Sigmoid`,
  `Layers::Tanh` and `Layers::SiLU`.
- `Tensor#sum` and the new `Tensor#mean` take `axis:` (an axis or a list of
  them, negative counting from the end) and `keepdims:`; e.g.
  `x.mean(axis: [2, 3])` is global average pooling.
- `Tensor#/` (by a number or a tensor) and unary minus.
- `Maurograd.no_grad { ... }`: ops inside the block build no graph (for
  evaluation and sampling); `Maurograd.grad_enabled?`. Per thread,
  nestable, and separate from `model.eval`.
- `Module#state_dict` and `#load_state_dict` (strict by default; keys and
  shapes are checked before anything changes), and buffers: non-trainable
  state declared with `buffers :name`, which `to` moves and `state_dict`
  saves, as BatchNorm2D's running statistics now are.
- `Maurograd::SafeTensors.save` and `.load`, for the safetensors file
  format: weights saved by Maurograd load in Python's safetensors package,
  and the other way round, and loading a file can't run code.
- `Tensor#slice(axis, range)` (`Ops::Slice`), with any Ruby Range
  (`0...4`, `4..`, `-2..`), and `Tensor#chunk(n, axis:)` to split a tensor
  into equal parts, e.g. a VAE encoder's output into mean and log-variance.
- `Maurograd.concat(tensors, axis:)` (`Ops::Concat`), e.g. for the skip
  connections of a UNet.
- `Tensor#softplus` and `#log_sigmoid`, computed so they never overflow or
  underflow (`log(1 + e^x)` and `log(sigmoid(x))` directly give Infinity
  and -Infinity for large inputs); `Tensor#leaky_relu(negative_slope:)`;
  `Tensor#clamp(min:, max:)`; and the layers `Layers::LeakyReLU` and
  `Layers::Softplus`.
- `Tensor#detach`: the same data as a new leaf that stops gradients.
- `Tensor#logsumexp(axis:, keepdims:)` (`Ops::LogSumExp`),
  `Tensor#log_softmax(axis:)` (`Ops::LogSoftmax`) and `Tensor#softmax`,
  computed with the maximum subtracted first, so they stay finite for
  logits in the thousands.
- `Losses::BCEWithLogits` (`reduction: :mean` or `:sum`): binary
  cross-entropy on logits, finite and with a full gradient even for
  confident wrong predictions.
- `Maurograd.randn`, `.rand`, `.randn_like` and `.rand_like`: random
  tensors drawn by their backend (on the GPU for MLX), seeded by
  `Maurograd.seed!`.
- Backend methods `slice`, `pad`, `concat`, `normal`, `uniform`, `log1p`,
  `abs`, `maximum`, `minimum`, `greater` and `greater_equal`, on the CPU
  and MLX backends.

### Changed

- Every built-in layer, and `Sequential`, is now a `Layers::Module`.
  Parameter order is unchanged.
- `BatchNorm2D` no longer overrides `to`; its running statistics are
  buffers.
- `Sequential.new` raises `TypeError` for a layer that isn't a
  `Layers::Module`, instead of silently skipping its parameters.

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

[Unreleased]: https://github.com/mtortonesi/maurograd/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mtortonesi/maurograd/releases/tag/v0.1.0
