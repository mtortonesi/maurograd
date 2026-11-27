require_relative "../tensor"

module Maurograd
  module Ops
    # Elementwise functions: y = f(x), applied to every element on its own.
    #
    # Backward is the chain rule, one element at a time:
    #
    #   dL/dx = dL/dy * f'(x)
    #
    # so each function only has to say how to compute f and f'. Often f' is
    # cheapest in terms of the output y, which forward already computed:
    # exp' = exp = y, sigmoid' = y * (1 - y), tanh' = 1 - y^2.
    #
    # All of them are built from backend methods that already exist (exp,
    # log and arithmetic), so they need no new kernels on any backend.
    class Elementwise
      def self.apply(input)
        new(input).forward
      end

      attr_reader :inputs

      def initialize(input)
        @inputs = [input]
      end

      def forward
        input = @inputs.first
        @y = compute(input.backend, input.data)

        requires_grad = Ops.requires_grad?(input)
        out = Tensor.new(@y, requires_grad: requires_grad, backend: input.backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        input = @inputs.first
        return unless input.requires_grad

        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        input.accumulate_grad(dy * derivative(input.backend, input.data, @y))
      end

      private

      # f(x).
      def compute(bk, x)
        raise NotImplementedError
      end

      # f'(x), given x and y = f(x).
      def derivative(bk, x, y)
        raise NotImplementedError
      end

      # 1 / (1 + e^-x). For large negative x, e^-x overflows to infinity
      # and the result is 0, which is the right limit (not NaN).
      def sigmoid(bk, x)
        1.0 / (bk.exp(-x) + 1.0)
      end
    end

    # e^x. Its own derivative.
    class Exp < Elementwise
      private

      def compute(bk, x) = bk.exp(x)

      def derivative(_bk, _x, y) = y
    end

    # Natural logarithm, d/dx log x = 1/x. Inputs must be positive: log of
    # 0 is -Infinity and of a negative number is NaN, as in NumPy.
    class Log < Elementwise
      private

      def compute(bk, x) = bk.log(x)

      def derivative(_bk, x, _y) = 1.0 / x
    end

    # Logistic sigmoid, 1 / (1 + e^-x): squashes to (0, 1).
    # d/dx = sigmoid(x) * (1 - sigmoid(x)) = y * (1 - y).
    class Sigmoid < Elementwise
      private

      def compute(bk, x) = sigmoid(bk, x)

      def derivative(_bk, _x, y) = y * (1.0 - y)
    end

    # Hyperbolic tangent: squashes to (-1, 1). Computed as
    # 2 * sigmoid(2x) - 1, which is the same function and reuses the
    # overflow-safe sigmoid. d/dx tanh x = 1 - tanh^2 x = 1 - y^2.
    class Tanh < Elementwise
      private

      def compute(bk, x) = (sigmoid(bk, x * 2.0) * 2.0) - 1.0

      def derivative(_bk, _x, y) = 1.0 - (y * y)
    end

    # SiLU (also called swish): x * sigmoid(x). Smooth, and close to ReLU
    # for large |x|; the usual activation in diffusion UNets.
    # d/dx = s + x * s * (1 - s) = s * (1 + x * (1 - s)), with s = sigmoid(x).
    class SiLU < Elementwise
      private

      def compute(bk, x) = x * sigmoid(bk, x)

      def derivative(bk, x, _y)
        s = sigmoid(bk, x)
        s * ((x * (1.0 - s)) + 1.0)
      end
    end
  end
end
