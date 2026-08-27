# Maurograd (Tensor/Numo autograd) - default entrypoint.
#
# This file intentionally requires an explicit, minimal set of files.
# We avoid directory-wide autoload (Dir[...] require) to prevent accidentally
# loading legacy/scalar code and to keep the public API predictable.

require_relative 'maurograd/bootstrap'

Maurograd::Bootstrap.ensure_openblas!

require 'numo/narray'
require 'numo/linalg'

module Maurograd
  # Version first (so clients can check Maurograd::VERSION early).
  require_relative 'maurograd/version'

  # Maurograd.debug flag, off by default (used by Utils.assert_finite!).
  require_relative 'maurograd/debug'

  # Core tensor type (autograd graph node).
  require_relative 'maurograd/tensor'

  # Utilities used by multiple ops (broadcast reduction, im2col/col2im, etc.).
  require_relative 'maurograd/utils/utils'

  # Ops (autograd operations).
  require_relative 'maurograd/ops/add'
  require_relative 'maurograd/ops/sub'
  require_relative 'maurograd/ops/mul'
  require_relative 'maurograd/ops/matmul'
  require_relative 'maurograd/ops/power'
  require_relative 'maurograd/ops/sum'
  require_relative 'maurograd/ops/conv2d'
  require_relative 'maurograd/ops/linear'
  require_relative 'maurograd/ops/flatten'
  require_relative 'maurograd/ops/relu'
  require_relative 'maurograd/ops/maxpool2d'

  # Layers (high-level modules that own parameters).
  require_relative 'maurograd/layers/conv2d'
  require_relative 'maurograd/layers/linear'
  require_relative 'maurograd/layers/flatten'
  require_relative 'maurograd/layers/maxpool2d'
  require_relative 'maurograd/layers/relu'
  require_relative 'maurograd/layers/sequential'

  # Loss functions
  require_relative 'maurograd/losses/bce'
  require_relative 'maurograd/losses/mse'
  require_relative 'maurograd/losses/cross_entropy_with_logits'

  # Optimizers
  require_relative 'maurograd/optim/sgd'

  # Datasets
  require_relative 'maurograd/datasets/mnist'
  require_relative 'maurograd/datasets/batching'
end
