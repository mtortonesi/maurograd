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
    # All of them are built from backend methods (exp, log, comparisons and
    # arithmetic), so they need no kernels of their own on any backend.
    # Subclasses with a parameter (LeakyReLU's slope, Clamp's bounds) take
    # it as a keyword argument after the input.
    class Elementwise
      def self.apply(input, ...)
        new(input, ...).forward
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

      # log(1 + e^-|x|), which is between 0 and log 2 for every x and never
      # overflows: the part softplus and log_sigmoid share.
      def log1p_exp_neg_abs(bk, x)
        bk.log1p(bk.exp(-bk.abs(x)))
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

    # Softplus, log(1 + e^x): a smooth ReLU, always positive. Used to
    # make a network output positive (a standard deviation, say), and the
    # heart of binary cross-entropy with logits.
    #
    # Computed as max(x, 0) + log(1 + e^-|x|), which is the same function
    # (for x > 0, log(1 + e^x) = x + log(1 + e^-x)) but can't overflow: the
    # direct formula gives Infinity for x above about 88 in float32.
    # d/dx = e^x / (1 + e^x) = sigmoid(x).
    class Softplus < Elementwise
      private

      def compute(bk, x) = bk.maximum(x, 0.0) + log1p_exp_neg_abs(bk, x)

      def derivative(bk, x, _y) = sigmoid(bk, x)
    end

    # log(sigmoid(x)) = -softplus(-x): the log-probability of a Bernoulli
    # outcome 1 when x is its logit (and log_sigmoid(-x) that of 0), so a
    # Bernoulli likelihood is x*log_sigmoid(z) + (1-x)*log_sigmoid(-z).
    # Computing log(sigmoid(x)) directly gives log(0) = -Infinity once
    # sigmoid(x) underflows, around x = -88 in float32; this form is
    # min(x, 0) - log(1 + e^-|x|), finite everywhere.
    # d/dx = 1 - sigmoid(x) = sigmoid(-x).
    class LogSigmoid < Elementwise
      private

      def compute(bk, x) = bk.minimum(x, 0.0) - log1p_exp_neg_abs(bk, x)

      def derivative(bk, x, _y) = sigmoid(bk, -x)
    end

    # Leaky ReLU: x for x > 0, negative_slope * x otherwise. Unlike ReLU,
    # negative inputs keep a small gradient, so units can't "die"; the usual
    # activation in GAN discriminators and energy-based models.
    # d/dx = 1 for x > 0, negative_slope otherwise (at 0: negative_slope, as
    # ReLU takes 0 there).
    class LeakyReLU < Elementwise
      def initialize(input, negative_slope: 0.01)
        super(input)
        @slope = Float(negative_slope)
      end

      private

      def compute(bk, x) = x * slope_mask(bk, x)

      def derivative(bk, x, _y) = slope_mask(bk, x)

      # 1 where x > 0, the slope elsewhere.
      def slope_mask(bk, x)
        positive = bk.greater(x, 0.0)
        positive + ((1.0 - positive) * @slope)
      end
    end

    # Clamp (clip) to [min, max]; either bound may be nil. Used to keep
    # values in a valid range, e.g. a log-variance, or pixels in [-1, 1]
    # after sampling.
    # d/dx = 1 inside [min, max], bounds included (as in PyTorch), and 0
    # outside, where the output no longer depends on x.
    class Clamp < Elementwise
      def initialize(input, min: nil, max: nil)
        super(input)
        raise ArgumentError, "clamp needs a min:, a max:, or both" if min.nil? && max.nil?
        raise ArgumentError, "clamp needs min <= max, got #{min} and #{max}" if min && max && min > max

        @min = min&.to_f
        @max = max&.to_f
      end

      private

      def compute(bk, x)
        y = @min ? bk.maximum(x, @min) : x
        @max ? bk.minimum(y, @max) : y
      end

      def derivative(bk, x, _y)
        inside = @min ? bk.greater_equal(x, @min) : bk.ones(x.shape)
        @max ? inside * (1.0 - bk.greater(x, @max)) : inside
      end
    end
  end
end
