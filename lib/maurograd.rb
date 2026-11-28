# Maurograd (Tensor/Numo autograd) - default entrypoint.
#
# This file intentionally requires an explicit, minimal set of files
# (rather than a directory-wide Dir[...] require) to keep the public API
# predictable.
#
# The original micrograd-style scalar engine this project started as now
# lives in its own repo: https://github.com/mtortonesi/maurograd_scalar

require_relative "maurograd/bootstrap"

Maurograd::Bootstrap.limit_blas_threads!

require "numo/narray"
require "numo/linalg"

module Maurograd
  # Version first (so clients can check Maurograd::VERSION early).
  require_relative "maurograd/version"

  # Maurograd.debug flag, off by default (used by Utils.assert_finite!).
  require_relative "maurograd/debug"

  # Maurograd.no_grad { ... }: switches gradient recording off.
  require_relative "maurograd/grad_mode"

  # Maurograd.seed! for reproducible runs (also re-randomizes Numo's global
  # RNG at load time, so unseeded weight init is genuinely random).
  require_relative "maurograd/seed"

  # Compute backends (where tensor data lives): the CPU one is always
  # loaded; optional ones (MLX) load on first use.
  require_relative "maurograd/backends"
  require_relative "maurograd/backends/cpu"

  # Core tensor type (autograd graph node).
  require_relative "maurograd/tensor"

  # Utilities used by multiple ops (broadcast reduction, im2col/col2im, etc.).
  require_relative "maurograd/utils/utils"

  # Ops (autograd operations).
  require_relative "maurograd/ops/add"
  require_relative "maurograd/ops/sub"
  require_relative "maurograd/ops/mul"
  require_relative "maurograd/ops/matmul"
  require_relative "maurograd/ops/power"
  require_relative "maurograd/ops/sum"
  require_relative "maurograd/ops/conv2d"
  require_relative "maurograd/ops/linear"
  require_relative "maurograd/ops/flatten"
  require_relative "maurograd/ops/relu"
  require_relative "maurograd/ops/maxpool2d"
  require_relative "maurograd/ops/permute"
  require_relative "maurograd/ops/reshape"
  require_relative "maurograd/ops/elementwise"
  require_relative "maurograd/ops/slice"
  require_relative "maurograd/ops/concat"
  require_relative "maurograd/ops/log_softmax"
  require_relative "maurograd/ops/batchnorm2d"
  require_relative "maurograd/ops/dropout"

  # Layers (high-level modules that own parameters).
  require_relative "maurograd/layers/module"
  require_relative "maurograd/layers/conv2d"
  require_relative "maurograd/layers/linear"
  require_relative "maurograd/layers/flatten"
  require_relative "maurograd/layers/maxpool2d"
  require_relative "maurograd/layers/relu"
  require_relative "maurograd/layers/activations"
  require_relative "maurograd/layers/batchnorm2d"
  require_relative "maurograd/layers/dropout"
  require_relative "maurograd/layers/sequential"

  # Loss functions
  require_relative "maurograd/losses/bce"
  require_relative "maurograd/losses/bce_with_logits"
  require_relative "maurograd/losses/mse"
  require_relative "maurograd/losses/cross_entropy_with_logits"

  # Optimizers
  require_relative "maurograd/optim/optimizer"
  require_relative "maurograd/optim/sgd"
  require_relative "maurograd/optim/adam"

  # Datasets
  # Saving and loading weights (safetensors format).
  require_relative "maurograd/safetensors"

  require_relative "maurograd/datasets/mnist"
  require_relative "maurograd/datasets/batching"
end
