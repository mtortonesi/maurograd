module Maurograd
  module Ops
    class Sum
      def self.apply(tensor)
        new(tensor).forward
      end

      attr_reader :inputs

      def initialize(tensor)
        # Store the tensor in inputs for backtracking.
        @inputs = [tensor]
      end

      def forward
        tensor = @inputs.first

        # The result of the sum is a single scalar value containing
        # the sum of all elements.
        result_data = tensor.data.sum

        requires_grad = tensor.requires_grad
        result = Tensor.new(result_data, requires_grad: requires_grad)

        result.creator = self if requires_grad
        result
      end

      def backward(grad_output)
        tensor = @inputs.first
        return unless tensor.requires_grad

        # If y = sum(x), then dy/dx_i = 1 for every i.
        # The total gradient is grad_output * 1, broadcast to tensor's shape.
        #
        # grad_output is a scalar. We use Numo::SFloat.cast to make sure
        # grad_output is handled correctly, then multiply it by an array
        # of ones with the same shape as the input.
        ones = Numo::SFloat.ones(*tensor.data.shape)
        grad_input = grad_output * ones

        tensor.accumulate_grad(grad_input)
      end
    end
  end
end
