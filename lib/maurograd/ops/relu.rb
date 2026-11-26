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
    # Both passes run in the C extension. Backward only needs to know where
    # x was positive, and that is exactly where y is positive, so we cache
    # the output y instead of a separate mask.
    class ReLU
      def self.apply(input)
        new(input).forward
      end

      attr_reader :inputs

      def initialize(input)
        @inputs = [input]
        @y = nil
      end

      def forward
        input = @inputs.first
        y = Maurograd::Ext.relu_forward(Utils.sfloat_contiguous(input.data))

        requires_grad = input.requires_grad
        @y = y if requires_grad

        out = Tensor.new(y, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        dx = Maurograd::Ext.relu_backward(Utils.sfloat_contiguous(grad_output), @y)
        input.accumulate_grad(dx)
      end
    end
  end
end
