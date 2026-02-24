require_relative '../tensor'

module Maurograd
  module Ops
    # ReLU activation:
    #   y = max(0, x)
    #
    # Backward:
    #   dy/dx = 1 if x > 0 else 0
    class ReLU
      def self.apply(input)
        new(input).forward
      end

      attr_reader :inputs

      def initialize(input)
        @inputs = [input]
        @mask = nil
      end

      def forward
        x = @inputs.first.data

        # mask is 1 where x > 0, else 0 (float array, easy to multiply in backward)
        @mask = Numo::SFloat.cast(x > 0)
        y = x * @mask

        requires_grad = @inputs.first.requires_grad
        out = Tensor.new(y, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        dx = dy * @mask
        input.accumulate_grad(dx)
      end
    end
  end
end

