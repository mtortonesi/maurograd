require_relative "../tensor"

module Maurograd
  module Ops
    # log(sum(exp(x))) over the given axes, computed without overflow.
    #
    # Forward:
    #   exp overflows float32 above about 88, so the sum is taken after
    #   shifting by the maximum m, which cancels out exactly:
    #
    #     logsumexp(x) = m + log(sum(exp(x - m)))
    #
    #   Every exp(x - m) is then at most 1, and at least one is exactly 1,
    #   so the log is finite too.
    #
    # Backward:
    #   d/dx_i logsumexp(x) = exp(x_i) / sum(exp(x)) = softmax(x)_i, which is
    #   exp(x_i - y), y being the output: each element's gradient is dy
    #   weighted by its share of the sum.
    #
    # It turns sums of probabilities into sums in log space: the
    # log-likelihood of a mixture of Gaussians, the VampPrior's log-density,
    # and an importance-sampled log-likelihood estimate are all logsumexps.
    class LogSumExp
      def self.apply(input, axis: nil, keepdims: false)
        new(input, axis: axis, keepdims: keepdims).forward
      end

      attr_reader :inputs, :axes

      def initialize(input, axis: nil, keepdims: false)
        @inputs = [input]
        @axes = Sum.normalize_axes(axis, input.ndim) || (0...input.ndim).to_a
        @keepdims = keepdims
      end

      def forward
        input = @inputs.first
        bk = input.backend
        x = input.data

        m = bk.max(x, axis: @axes, keepdims: true)
        @y_kept = bk.log(bk.sum(bk.exp(x - m), axis: @axes, keepdims: true)) + m
        y = @keepdims ? @y_kept : drop_axes(bk, @y_kept)

        requires_grad = Ops.requires_grad?(input)
        out = Tensor.new(y, requires_grad: requires_grad, backend: bk)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        bk = input.backend
        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        dy = bk.reshape(dy, *@y_kept.shape) unless @keepdims
        input.accumulate_grad(dy * bk.exp(input.data - @y_kept))
      end

      private

      # The keepdims result without its size-1 axes: a scalar when every
      # axis was reduced.
      def drop_axes(bk, y)
        shape = @inputs.first.shape.each_with_index.reject { |_, i| @axes.include?(i) }.map(&:first)
        shape.empty? ? bk.sum(y) : bk.reshape(y, *shape)
      end
    end

    # log(softmax(x)) along one axis: x - logsumexp(x), the log-probabilities
    # of a categorical distribution whose logits are x.
    #
    # Forward:
    #   y = x - logsumexp(x, axis, keepdims: true). Taking the log of a
    #   softmax computed first would give log(0) = -Infinity for classes
    #   whose probability underflows; this form doesn't.
    #
    # Backward:
    #   With p = softmax(x) = exp(y), dy_j/dx_i = [i == j] - p_i, so
    #
    #     dx = dy - p * sum(dy)        (the sum along the axis)
    class LogSoftmax
      def self.apply(input, axis: -1)
        new(input, axis: axis).forward
      end

      attr_reader :inputs, :axis

      def initialize(input, axis: -1)
        @inputs = [input]
        @axis = Sum.normalize_axes(axis, input.ndim).first
      end

      def forward
        input = @inputs.first
        bk = input.backend
        x = input.data

        m = bk.max(x, axis: @axis, keepdims: true)
        shifted = x - m
        @y = shifted - bk.log(bk.sum(bk.exp(shifted), axis: @axis, keepdims: true))

        requires_grad = Ops.requires_grad?(input)
        out = Tensor.new(@y, requires_grad: requires_grad, backend: bk)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        bk = input.backend
        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        input.accumulate_grad(dy - (bk.exp(@y) * bk.sum(dy, axis: @axis, keepdims: true)))
      end
    end
  end
end
