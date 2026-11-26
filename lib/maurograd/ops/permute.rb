require_relative "../tensor"

module Maurograd
  module Ops
    # Permute (reorder) the axes of a tensor.
    #
    # Forward:
    #   y = x with its axes reordered: y.shape[i] == x.shape[axes[i]]
    #   (e.g. axes = [0, 2, 3, 1] turns NCHW into NHWC)
    #
    # Backward:
    #   A permutation only moves values around, so the gradient is dy with
    #   its axes moved back: apply the inverse permutation, the one that
    #   sends axis axes[i] back to position i.
    #
    #   Example: axes = [0, 2, 3, 1] (NCHW -> NHWC) has inverse
    #   [0, 3, 1, 2] (NHWC -> NCHW).
    #
    # Tensor#transpose (swap two axes) and Tensor#permute both go through
    # this op, so gradients flow through them like through any other op.
    class Permute
      def self.apply(input, axes)
        new(input, axes).forward
      end

      # Normalizes `axes` for a tensor with `ndim` dimensions (negative axes
      # count from the end) and checks that it is a permutation of
      # 0...ndim. Returns the normalized axes.
      def self.normalize_axes(axes, ndim)
        unless axes.length == ndim
          raise ArgumentError, "permute needs #{ndim} axes for a #{ndim}-dimensional tensor, got #{axes.inspect}"
        end

        normalized = axes.map do |a|
          raise ArgumentError, "axes must be Integers, got #{axes.inspect}" unless a.is_a?(Integer)

          a.negative? ? a + ndim : a
        end
        unless normalized.sort == (0...ndim).to_a
          raise ArgumentError, "axes #{axes.inspect} are not a permutation of 0...#{ndim}"
        end

        normalized
      end

      attr_reader :inputs, :axes

      def initialize(input, axes)
        @inputs = [input]
        @axes = self.class.normalize_axes(axes, input.ndim)
      end

      def forward
        input = @inputs.first
        y = input.backend.transpose(input.data, *@axes)

        requires_grad = input.requires_grad
        out = Tensor.new(y, requires_grad: requires_grad, backend: input.backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        inverse = Array.new(@axes.length)
        @axes.each_with_index { |a, i| inverse[a] = i }

        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        input.accumulate_grad(input.backend.transpose(dy, *inverse))
      end
    end
  end
end
