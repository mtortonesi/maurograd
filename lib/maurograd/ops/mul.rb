require_relative "../utils/utils"

module Maurograd
  module Ops
    # Elementwise multiplication: a * b
    #
    # Forward: out = a * b (Numo broadcasting allowed)
    # Backward:
    #   dL/da = grad_output * b
    #   dL/db = grad_output * a
    #
    # Broadcasting notes:
    # If broadcasting happened in forward, we must "undo" it in backward by summing
    # gradients across broadcasted dimensions (Utils.unbroadcast).
    class Mul
      def self.apply(a, b)
        new(a, b).forward
      end

      attr_reader :inputs

      def initialize(a, b)
        @inputs = [a, b]
      end

      def forward
        a, b = @inputs
        @backend = Ops.backend_of(a, b)
        out_data = a.data * b.data
        requires_grad = Ops.requires_grad?(a, b)
        out = Tensor.new(out_data, requires_grad: requires_grad, backend: @inputs.first.backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        a, b = @inputs
        gout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output

        if a.requires_grad
          ga = gout * b.data
          a.accumulate_grad(Utils.unbroadcast(ga, a.shape, @backend))
        end

        if b.requires_grad
          gb = gout * a.data
          b.accumulate_grad(Utils.unbroadcast(gb, b.shape, @backend))
        end
      end
    end
  end
end
