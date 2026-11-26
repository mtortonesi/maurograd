require_relative "../tensor"

module Maurograd
  module Ops
    # Dropout (Srivastava et al., 2014), "inverted" variant.
    #
    # Forward (training):
    #   mask = 1 / (1 - p)  with probability 1 - p   (element kept)
    #          0            with probability p       (element dropped)
    #   y    = x * mask
    #
    # Forward (eval):
    #   y = x
    #
    # Why "inverted": the original formulation dropped units at training time
    # and then multiplied by (1 - p) at test time, so that the expected input
    # of the next layer was the same in both phases. Scaling the survivors by
    # 1 / (1 - p) during training instead gives the same expectation
    # (E[y] = (1 - p) * x / (1 - p) = x) and leaves eval mode a plain
    # identity - nothing to remember at inference time. This is what PyTorch
    # does too.
    #
    # Backward (training): y is x times a constant mask, so
    #   dx = dout * mask
    # (dropped elements get no gradient, kept ones get it scaled by 1/(1-p)).
    #
    # The mask is drawn from the backend's random number generator, so
    # Maurograd.seed! makes it reproducible.
    class Dropout
      def self.apply(input, p: 0.5, training: true)
        new(input, p, training).forward
      end

      def self.validate_p!(p)
        return if p.is_a?(Numeric) && p >= 0.0 && p <= 1.0

        raise ArgumentError, "dropout probability must be in [0, 1], got #{p.inspect}"
      end

      attr_reader :inputs

      def initialize(input, p, training)
        self.class.validate_p!(p)
        @inputs = [input]
        @p = p
        @training = training
        @mask = nil
      end

      def forward
        input = @inputs.first

        # Identity: in eval mode, or when nothing would be dropped. Returning
        # the input itself keeps the graph unchanged (no extra node at all).
        return input if !@training || @p == 0.0

        bk = input.backend
        @mask = if @p >= 1.0 # i.e. p == 1, since validate_p! caps it at 1
          # Everything dropped; avoid the 1 / (1 - p) division by zero.
          bk.zeros(input.shape)
        else
          bk.bernoulli(input.shape, 1.0 - @p) * (1.0 / (1.0 - @p))
        end

        y = input.data * @mask

        requires_grad = input.requires_grad
        out = Tensor.new(y, requires_grad: requires_grad, backend: @inputs.first.backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        dout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        input.accumulate_grad(dout * @mask)
      end
    end
  end
end
