module Maurograd
  module Ops
    class Power
      def self.apply(tensor, exponent)
        new(tensor, exponent).forward
      end

      attr_reader :inputs

      def initialize(tensor, exponent)
        # Check that the exponent is a number (Integer, Float, etc.)
        unless exponent.is_a?(Numeric)
          raise ArgumentError, "The exponent must be a scalar numeric value, not a #{exponent.class}"
        end
        # Store the tensor in inputs for backtracking.
        @inputs = [tensor]
        # Store the exponent as a static parameter.
        @exponent = exponent
      end

      def forward
        tensor = @inputs.first
        result_data = tensor.data ** @exponent

        requires_grad = tensor.requires_grad
        result = Tensor.new(result_data, requires_grad: requires_grad)

        result.creator = self if requires_grad
        result
      end

      def backward(grad_output)
        tensor = @inputs.first
        return unless tensor.requires_grad

        # Local derivative: n * x^(n-1)
        local_derivative = @exponent * (tensor.data ** (@exponent - 1))

        # Accumulate the gradient (chain rule)
        grad_input = grad_output * local_derivative

        tensor.accumulate_grad(grad_input)
      end
    end
  end
end
