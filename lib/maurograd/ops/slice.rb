require_relative "../tensor"

module Maurograd
  module Ops
    # A contiguous range of a tensor along one axis.
    #
    # Forward:
    #   y = x[..., start...stop, ...]   (the range on `axis`, everything on
    #   the other axes), e.g. for x of shape [N, 8]:
    #     x.slice(1, 0...4)  -> [N, 4], the first four columns
    #     x.slice(1, 4..)    -> [N, 4], the rest
    #     x.slice(1, -2..)   -> [N, 2], the last two
    #
    # Backward:
    #   Each element of y is one element of x, copied, so its gradient goes
    #   back to the position it came from, and the elements of x that were
    #   left out get zero: dx is dy padded with zeros, `start` of them before
    #   and `size - stop` after, along `axis`.
    #
    # The usual use is splitting a network's output in parts: a VAE encoder
    # that outputs [mean, log-variance] side by side, or a coupling layer
    # that transforms half of its input (see Tensor#chunk).
    class Slice
      def self.apply(input, axis, range)
        new(input, axis, range).forward
      end

      # Turns `range` (any Ruby Range of Integers: 1...3, 2.., ..-1, -2..)
      # into [start, stop] with 0 <= start < stop <= size, stop excluded, as
      # Array#[] reads ranges. Raises ArgumentError for a range that selects
      # nothing.
      def self.resolve_range(range, size)
        raise ArgumentError, "slice needs a Range, got #{range.inspect}" unless range.is_a?(Range)

        first = range.begin || 0
        last = range.end
        integers = first.is_a?(Integer) && (last.nil? || last.is_a?(Integer))
        raise ArgumentError, "slice ranges must be of Integers, got #{range.inspect}" unless integers

        start = first.negative? ? first + size : first
        stop = if last.nil?
          size
        else
          (last.negative? ? last + size : last) + (range.exclude_end? ? 0 : 1)
        end
        stop = stop.clamp(0, size)
        if start.negative? || start >= stop
          raise ArgumentError, "range #{range.inspect} selects nothing from an axis of size #{size}"
        end

        [start, stop]
      end

      attr_reader :inputs, :axis, :start, :stop

      def initialize(input, axis, range)
        @inputs = [input]
        @axis = Sum.normalize_axes(axis, input.ndim).first
        @start, @stop = self.class.resolve_range(range, input.shape[@axis])
      end

      def forward
        input = @inputs.first
        y = input.backend.slice(input.data, @axis, @start, @stop)

        requires_grad = Ops.requires_grad?(input)
        out = Tensor.new(y, requires_grad: requires_grad, backend: input.backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        after = input.shape[@axis] - @stop
        input.accumulate_grad(input.backend.pad(dy, @axis, @start, after))
      end
    end
  end
end
