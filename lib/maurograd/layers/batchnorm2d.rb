require_relative "../tensor"
require_relative "../ops/batchnorm2d"

module Maurograd
  module Layers
    # BatchNorm2D Layer (high-level API), for NCHW tensors.
    #
    # Owns the trainable per-channel scale/shift (gamma/beta) and the
    # non-trainable running mean/variance used at eval time. Delegates the
    # actual normalization math to Ops::BatchNorm2D.
    #
    # running_mean/running_var are plain Numo::SFloat arrays, not Tensors:
    # they are statistics, not parameters trained by gradient descent, so
    # they must NOT appear in #parameters (Optim::SGD#step would otherwise
    # try to update them with a gradient they don't have).
    #
    class BatchNorm2D
      attr_accessor :gamma, :beta, :running_mean, :running_var, :momentum, :eps

      def initialize(num_features, momentum: 0.1, eps: 1e-5)
        @momentum = momentum
        @eps = eps
        @training = true

        @gamma = Maurograd::Tensor.new(Numo::SFloat.ones(num_features), requires_grad: true)
        @beta = Maurograd::Tensor.new(Numo::SFloat.zeros(num_features), requires_grad: true)

        @running_mean = Numo::SFloat.zeros(num_features)
        @running_var = Numo::SFloat.ones(num_features)
      end

      def forward(input)
        Maurograd::Ops::BatchNorm2D.apply(
          input, @gamma, @beta,
          running_mean: @running_mean,
          running_var: @running_var,
          training: @training,
          momentum: @momentum,
          eps: @eps
        )
      end

      def parameters
        [@gamma, @beta]
      end

      def train
        @training = true
        self
      end

      def eval
        @training = false
        self
      end
    end
  end
end
