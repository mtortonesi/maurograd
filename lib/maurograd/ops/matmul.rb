# Note: In PyTorch, to optimize memory, the forward method doesn't save the entire input
# Tensor object unless needed, but only the strictly required data. Later we might want to
# implement a small Context object (as you mentioned in your conv2d_context.rb
# file structure) to "save for backward."
#
# # Example of "cache" to save memory
# def forward
#   # ... calcolo ...
#   @ctx_save_for_backward = [a.data, b.data]
#   # ...
# end
require 'numo/linalg'

module Maurograd
  module Ops
    class MatMul
      def self.apply(a, b)
        new(a, b).forward
      end

      attr_reader :inputs

      def initialize(a, b)
        @inputs = [a, b]
      end

      def forward
        a, b = @inputs

        # We multiply the underlying contents of the a and b tensors with
        # Numo::NArray using the .dot method
        result_data = a.data.dot(b.data)

        # The output requires the gradient if at least one of the inputs requires it.
        requires_grad = a.requires_grad || b.requires_grad
        result = Tensor.new(result_data, requires_grad: requires_grad)

        # We set the creator ONLY if it is needed for backward
        result.creator = self if requires_grad

        result
      end

      def backward(grad_output)
        a, b = @inputs

        # TODO: Handle broadcasting in gradient computation.
        # Currently, we assume grad_output shape matches the expected output shape
        # of the matmul operation without broadcasting across batch dimensions.
        # If broadcasting was applied during the forward pass, we must aggregate
        # (sum) the gradients across the broadcasted dimensions to match
        # the original input shapes.
        #
        # The broadcasting issue emerges in Batch Matrix Multiplication. If you
        # have a batch of matrices [B,N,M] and multiply them by a matrix [M,P],
        # you expect the operation to be "repeated" for each element of the
        # batch. In PyTorch, this is handled by torch.matmul. This is a more
        # complex operation because it's not a simple element-by-element
        # addition, but involves index contraction. For now, we've decided to
        # limit MatMul to the standard 2D case (or fixed batches) to stabilize
        # the Maurograd core before tackling multidimensional tensor calculus.

        # Explicit check for shape mismatch to avoid silent errors
        expected_shape = [a.shape[0], b.shape[1]]
        if grad_output.shape != expected_shape
          raise ArgumentError, "Broadcasting detected: grad_output shape #{grad_output.shape} " \
                               "does not match expected shape #{expected_shape}. " \
                               "Broadcasting support in backward pass is not yet implemented."
        end

        if a.requires_grad
          # grad_a = grad_output @ B.T (transposing the last 2 dimensions of B)
          # Note: transpose(-1, -2) and transpose(-2, -1) are, of course, equivalent
          grad_a = grad_output.dot(b.data.transpose(-1, -2))
          a.accumulate_grad(grad_a)
        end

        if b.requires_grad
          # grad_b = A.T (transposing the last 2 dimensions of A) @ grad_output
          grad_b = a.data.transpose(-1, -2).dot(grad_output)
          b.accumulate_grad(grad_b)
        end
      end
    end
  end
end
