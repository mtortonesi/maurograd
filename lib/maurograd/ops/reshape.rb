require_relative "../tensor"

module Maurograd
  module Ops
    # Reshape a tensor: same elements, same order, different shape.
    #
    # Forward:
    #   y = x read in row-major order (last axis fastest) and laid out again
    #   in `shape`. The number of elements must not change, e.g.
    #   [2, 3, 4] -> [6, 4], [2, 12] or [24]. One dimension may be -1, which
    #   means "whatever makes the sizes match", as in NumPy and PyTorch:
    #   [2, 3, 4] -> [2, -1] gives [2, 12].
    #
    # Backward:
    #   Reshape doesn't change any value, it only regroups them, so element
    #   k (in row-major order) of y is element k of x. The gradient is
    #   therefore dy reshaped back to x's shape. Nothing is summed or
    #   reordered, which is what makes it different from Permute: permute
    #   moves elements to new positions in row-major order, reshape doesn't.
    #
    # Reshape vs Permute: [2, 3] -> [3, 2] by reshape regroups 0 1 2 | 3 4 5
    # as 0 1 | 2 3 | 4 5, while transpose gives 0 3 | 1 4 | 2 5. Using one
    # where the other was meant is a classic silent bug: the shapes match,
    # the model trains, and the channels are scrambled.
    #
    # Flatten is the special case [N, ...] -> [N, -1].
    class Reshape
      def self.apply(input, shape)
        new(input, shape).forward
      end

      # Resolves a requested `shape` for a tensor of `size` elements: checks
      # the dimensions and replaces a -1 with the size it must have. Returns
      # the concrete shape. The backends get only concrete shapes, since
      # Numo doesn't accept -1 (it spells it nil or true) and MLX does.
      def self.resolve_shape(shape, size)
        unless shape.all? { |d| d.is_a?(Integer) && (d >= 0 || d == -1) }
          raise ArgumentError, "reshape dimensions must be non-negative Integers or a single -1, got #{shape.inspect}"
        end

        inferred = shape.count(-1)
        raise ArgumentError, "reshape can infer only one dimension, got #{shape.inspect}" if inferred > 1

        known = shape.reject { |d| d == -1 }.reduce(1, :*)
        if inferred == 1
          if known.zero? || size % known != 0
            raise ArgumentError, "cannot reshape #{size} elements into #{shape.inspect}"
          end

          return shape.map { |d| (d == -1) ? size / known : d }
        end

        raise ArgumentError, "cannot reshape #{size} elements into #{shape.inspect}" unless known == size

        shape
      end

      attr_reader :inputs, :shape

      def initialize(input, shape)
        @inputs = [input]
        @shape = self.class.resolve_shape(shape, input.size)
      end

      def forward
        input = @inputs.first
        # Both backends read strided views (e.g. the output of permute) in
        # logical row-major order, so no explicit copy is needed here.
        y = input.backend.reshape(input.data, *@shape)

        requires_grad = input.requires_grad
        out = Tensor.new(y, requires_grad: requires_grad, backend: input.backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        input.accumulate_grad(input.backend.reshape(dy, *input.shape))
      end
    end
  end
end
