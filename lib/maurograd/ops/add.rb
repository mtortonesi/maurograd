require_relative '../utils/utils'

module Maurograd
  module Ops
    class Add
      def self.apply(a, b)
        new(a, b).forward
      end

      attr_reader :inputs

      def initialize(a, b)
        @inputs = [a, b]
      end

      def forward
        a, b = @inputs
        result_data = a.data + b.data

        requires_grad = a.requires_grad || b.requires_grad

        result = Tensor.new(result_data, requires_grad: requires_grad)
        result.creator = self if requires_grad
        result
      end

      def backward(grad_output)
        a, b = @inputs

        # Note: We explicitly address the broadcasting issue because in the Add
        # operation case broadcasting is "wild" and very common. You can add a
        # scalar to a matrix, or a row vector to a matrix. This is the typical
        # case of bias in CNNs: you have an image [64,32,32] (64 channels) and
        # you add a bias of [64,1,1]. Numo expands the bias across all pixels.
        # Backward has to add the gradients of all those pixels to get back to
        # the single bias value. Without unbroadcast, the bias would learn
        # nothing.

        if a.requires_grad
          # The derivative of a + b w.r.t. a is 1.
          # The gradient is simply grad_output, but we must handle broadcasting.
          grad_a = Utils.unbroadcast(grad_output, a.shape)
          a.accumulate_grad(grad_a)
        end

        if b.requires_grad
          # The derivative of a + b w.r.t. b is 1.
          grad_b = Utils.unbroadcast(grad_output, b.shape)
          b.accumulate_grad(grad_b)
        end
      end
    end
  end
end
