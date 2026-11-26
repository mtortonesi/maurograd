require_relative "optimizer"

module Maurograd
  module Optim
    # Stochastic Gradient Descent (SGD), with optional momentum and weight decay.
    #
    # Plain SGD (the default, momentum: 0, weight_decay: 0) is the simplest
    # optimizer:
    #
    #   param = param - lr * grad
    #
    # Weight decay adds an L2 penalty (weight_decay/2 * ||param||^2) to the
    # loss. Its gradient is weight_decay * param, so we simply fold it into
    # the gradient before doing anything else:
    #
    #   g = grad + weight_decay * param
    #
    # Momentum keeps a "velocity" v per parameter: an exponentially decaying
    # sum of past gradients. Consistent gradient directions build up speed,
    # while directions that keep flipping sign (noise, or oscillation across a
    # narrow valley) largely cancel out:
    #
    #   v     = momentum * v + g
    #   param = param - lr * v
    #
    # This is the formulation PyTorch uses (velocity accumulates raw
    # gradients, lr is applied at update time), so results are directly
    # comparable. On the first step v is initialized to g itself, so the
    # first update is identical to plain SGD.
    #
    # We keep it intentionally small and readable (didactic goal).
    class SGD < Optimizer
      attr_reader :lr, :momentum, :weight_decay

      def initialize(params, lr: 1e-2, momentum: 0.0, weight_decay: 0.0)
        super(params)
        raise ArgumentError, "momentum must be >= 0, got #{momentum}" if momentum.negative?
        raise ArgumentError, "weight_decay must be >= 0, got #{weight_decay}" if weight_decay.negative?

        @lr = lr
        @momentum = momentum
        @weight_decay = weight_decay

        # Velocity buffers, aligned with @params (set on the first step that
        # touches each parameter). Arrays are never modified in place, so
        # the first velocity can simply be the gradient array itself.
        @velocities = Array.new(params.size)
      end

      # Apply one optimization step.
      #
      # Notes:
      # - If a parameter has no gradient yet (grad=nil), we skip it.
      # - We replace `.data` with the updated array (like typical optimizers,
      #   the parameter Tensor object itself stays the same).
      def step
        each_updatable_param do |p, grad, i|
          g = grad
          g += @weight_decay * p.data if @weight_decay > 0

          if @momentum > 0
            @velocities[i] = @velocities[i] ? (@momentum * @velocities[i]) + g : g
            g = @velocities[i]
          end

          p.data -= @lr * g
        end
      end
    end
  end
end
