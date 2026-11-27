module Maurograd
  module Ops
    # Sum of a tensor's elements: over all of them, or over chosen axes.
    #
    # Forward:
    #   axis: nil       -> a scalar, the sum of every element
    #   axis: 1         -> sum over axis 1: [N, C, H, W] -> [N, H, W]
    #   axis: [2, 3]    -> sum over several axes: [N, C, H, W] -> [N, C]
    #   keepdims: true  -> the summed axes stay, with size 1: [N, C, 1, 1],
    #                      which broadcasts against the input
    #   Negative axes count from the end.
    #
    # Backward:
    #   Each input element contributes to exactly one output element, with
    #   derivative 1. So the gradient of an input element is the upstream
    #   gradient of the output element it was summed into: dy, copied
    #   ("broadcast back") along the summed axes. To do that, dy is first
    #   given the keepdims shape (size-1 axes where the sums were), then
    #   broadcast to the input's shape.
    #
    # Mean is this sum times 1/count (Tensor#mean).
    class Sum
      def self.apply(tensor, axis: nil, keepdims: false)
        new(tensor, axis: axis, keepdims: keepdims).forward
      end

      # Turns `axis` (nil, an Integer or an Array of them) into a sorted
      # Array of non-negative axes for a tensor with `ndim` dimensions, or
      # nil for "all axes". Raises ArgumentError for an axis out of range or
      # given twice.
      def self.normalize_axes(axis, ndim)
        return nil if axis.nil?

        axes = Array(axis).map do |a|
          raise ArgumentError, "axes must be Integers, got #{axis.inspect}" unless a.is_a?(Integer)

          n = a.negative? ? a + ndim : a
          raise ArgumentError, "axis #{a} is out of range for a #{ndim}-dimensional tensor" unless n.between?(0, ndim - 1)

          n
        end
        raise ArgumentError, "axis #{axis.inspect} repeats an axis" unless axes.uniq.length == axes.length

        axes.sort
      end

      attr_reader :inputs, :axes

      def initialize(tensor, axis: nil, keepdims: false)
        # Store the tensor in inputs for backtracking.
        @inputs = [tensor]
        @axes = self.class.normalize_axes(axis, tensor.ndim)
        @keepdims = keepdims
      end

      def forward
        tensor = @inputs.first
        bk = tensor.backend

        result_data = if @axes.nil?
          s = bk.sum(tensor.data)
          @keepdims ? bk.reshape(s, *Array.new(tensor.ndim, 1)) : s
        else
          bk.sum(tensor.data, axis: @axes, keepdims: @keepdims)
        end

        requires_grad = Ops.requires_grad?(tensor)
        result = Tensor.new(result_data, requires_grad: requires_grad, backend: bk)

        result.creator = self if requires_grad
        result
      end

      def backward(grad_output)
        tensor = @inputs.first
        return unless tensor.requires_grad

        bk = tensor.backend
        g = grad_output.is_a?(Tensor) ? grad_output.data : grad_output

        # Give the gradient the keepdims shape, so that it broadcasts along
        # the summed axes. (With axis: nil and no keepdims, it's a scalar,
        # which broadcasts as it is.)
        if @axes && !@keepdims
          kept = tensor.shape.each_with_index.map { |d, i| @axes.include?(i) ? 1 : d }
          g = bk.reshape(g, *kept)
        end

        tensor.accumulate_grad(g * bk.ones(tensor.shape))
      end
    end
  end
end
