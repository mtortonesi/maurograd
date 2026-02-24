require_relative '../tensor'

module Maurograd
  module Ops
    # Flatten operation.
    #
    # Converts a tensor of shape:
    #   [N, d1, d2, ..., dk]
    # into:
    #   [N, d1*d2*...*dk]
    #
    # This is typically used to connect convolutional stacks (4D NCHW)
    # to fully connected layers (2D).
    #
    # Forward caches the original shape so backward can reshape gradients back.
    class Flatten
      def self.apply(input)
        new(input).forward
      end

      attr_reader :inputs

      def initialize(input)
        @inputs = [input]
        @orig_shape = nil
      end

      def forward
        x = @inputs.first.data
        @orig_shape = x.shape

        # Keep batch dimension N intact.
        n = @orig_shape[0]
        raise "Flatten expects at least 1 dimension" if n.nil?

        flat_dim = x.size / n
        y = x.reshape(n, flat_dim)

        requires_grad = @inputs.first.requires_grad
        out = Tensor.new(y, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        dx = dy.reshape(*@orig_shape)
        input.accumulate_grad(dx)
      end
    end
  end
end
