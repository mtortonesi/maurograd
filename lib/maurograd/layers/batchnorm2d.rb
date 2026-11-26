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
    # running_mean/running_var are plain native arrays of the layer's
    # backend (Numo::SFloat on the CPU backend), not Tensors: they are
    # statistics, not parameters trained by gradient descent, so they must
    # NOT appear in #parameters (Optim::SGD#step would otherwise try to
    # update them with a gradient they don't have). In training mode,
    # forward replaces them with their updated values.
    #
    class BatchNorm2D
      attr_accessor :gamma, :beta, :running_mean, :running_var, :momentum, :eps

      def initialize(num_features, momentum: 0.1, eps: 1e-5)
        @momentum = momentum
        @eps = eps
        @training = true

        @gamma = Maurograd::Tensor.new(Numo::SFloat.ones(num_features), requires_grad: true)
        @beta = Maurograd::Tensor.new(Numo::SFloat.zeros(num_features), requires_grad: true)

        @running_mean = @gamma.backend.zeros([num_features])
        @running_var = @gamma.backend.ones([num_features])
      end

      def forward(input)
        Maurograd::Ops::BatchNorm2D.apply(
          input, @gamma, @beta,
          stats: self,
          training: @training,
          momentum: @momentum,
          eps: @eps
        )
      end

      def parameters
        [@gamma, @beta]
      end

      # Moves parameters and running statistics to another backend.
      def to(backend)
        source = @gamma.backend
        parameters.each { |p| p.to(backend) }
        target = @gamma.backend
        @running_mean = target.from(source.to_numo(@running_mean))
        @running_var = target.from(source.to_numo(@running_var))
        self
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
