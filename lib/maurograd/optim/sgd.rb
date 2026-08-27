module Maurograd
  module Optim
    # Stochastic Gradient Descent (SGD).
    #
    # This is the simplest optimizer:
    #   param = param - lr * grad
    #
    # We keep it intentionally small and readable (didactic goal).
    class SGD
      attr_reader :params, :lr

      def initialize(params, lr: 1e-2)
        @params = params
        @lr = lr
      end

      # Reset gradients of all parameters to zero.
      #
      # This mirrors PyTorch's optimizer.zero_grad().
      def zero_grad
        @params.each(&:zero_grad)
      end

      # Apply one optimization step.
      #
      # For each parameter tensor:
      #   data -= lr * grad
      #
      # Notes:
      # - If a parameter has no gradient yet (grad=nil), we skip it.
      # - We update `.data` in-place (like typical optimizers).
      def step
        @params.each do |p|
          next unless p.requires_grad
          next if p.grad.nil?

          p.data -= @lr * p.grad
        end
      end
    end
  end
end
