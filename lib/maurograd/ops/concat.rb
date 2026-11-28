require_relative "../tensor"

module Maurograd
  module Ops
    # Joins tensors along an existing axis.
    #
    # Forward:
    #   The inputs must have the same shape except along `axis`, and the
    #   output's size there is the sum of theirs:
    #     concat([a [N, 3], b [N, 5]], axis: 1)  -> [N, 8]
    #   The UNet's skip connections are the main use: decoder features
    #   [N, C1, H, W] joined with the encoder's [N, C2, H, W] give
    #   [N, C1 + C2, H, W].
    #
    # Backward:
    #   Concat is Slice in reverse: each input's gradient is its own range
    #   of dy along `axis`, cut out at the offset where it was placed.
    class Concat
      def self.apply(tensors, axis: 0)
        new(tensors, axis: axis).forward
      end

      attr_reader :inputs, :axis

      def initialize(tensors, axis: 0)
        unless tensors.is_a?(Array) && !tensors.empty? && tensors.all?(Tensor)
          raise ArgumentError, "concat needs a non-empty Array of Tensors"
        end

        @inputs = tensors
        first = tensors.first
        @axis = Sum.normalize_axes(axis, first.ndim).first
        tensors.each do |t|
          next if t.ndim == first.ndim && t.shape.each_index.all? { |i| i == @axis || t.shape[i] == first.shape[i] }

          raise ArgumentError, "concat along axis #{axis} needs the same shape on every other axis, " \
            "got #{tensors.map(&:shape).inspect}"
        end
      end

      def forward
        backend = Ops.backend_of(*@inputs)
        y = backend.concat(@inputs.map(&:data), @axis)

        requires_grad = Ops.requires_grad?(*@inputs)
        out = Tensor.new(y, requires_grad: requires_grad, backend: backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        offset = 0
        @inputs.each do |t|
          size = t.shape[@axis]
          t.accumulate_grad(t.backend.slice(dy, @axis, offset, offset + size)) if t.requires_grad
          offset += size
        end
      end
    end
  end

  # Joins tensors along an existing axis; see Ops::Concat.
  #
  #   h = Maurograd.concat([decoder_features, skip], axis: 1)
  def self.concat(tensors, axis: 0)
    Ops::Concat.apply(tensors, axis: axis)
  end
end
