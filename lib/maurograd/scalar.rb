# Maurograd Scalar (legacy micrograd-style) entrypoint.
#
# This is intentionally separate from the default Tensor-based Maurograd.
# Users must opt-in explicitly:
#
#   require 'maurograd/scalar'
#
# To avoid name collisions with the Tensor API, scalar classes should live
# under Maurograd::Scalar (recommended). If your current code defines top-level
# constants like Value, Neuron, Layer, MLP, you can keep it temporarily, but
# migrating them under the namespace is strongly recommended.

require_relative 'scalar/backpropagate'
require_relative 'scalar/value'
require_relative 'scalar/neuron'
require_relative 'scalar/layer'
require_relative 'scalar/mlp'
