require_relative "../tensor"
require_relative "../utils/utils"

module Maurograd
  module Ops
    # ReLU activation:
    #   y = max(0, x)
    #
    # Backward:
    #   dy/dx = 1 if x > 0 else 0
    #
    # Both passes are backend kernels (on the CPU backend, C functions in
    # maurograd_ext). Backward only needs to know where x was positive,
    # which is exactly where y is positive, so the CPU kernel caches the
    # output y instead of a separate mask.
    class ReLU
      def self.apply(input)
        new(input).forward
      end

      attr_reader :inputs

      def initialize(input)
        @inputs = [input]
        @ctx = nil
      end

      def forward
        input = @inputs.first
        y, ctx = input.backend.relu_forward(input.data)

        requires_grad = Ops.requires_grad?(input)
        @ctx = ctx if requires_grad

        out = Tensor.new(y, requires_grad: requires_grad, backend: @inputs.first.backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        dout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        dx = input.backend.relu_backward(@ctx, dout)
        input.accumulate_grad(dx)
      end
    end
  end
end
