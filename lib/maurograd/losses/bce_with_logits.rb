require_relative "../tensor"

module Maurograd
  module Losses
    # Binary cross-entropy on logits: BCE applied to sigmoid(logits), in one
    # numerically stable step. The usual loss for a GAN discriminator, and
    # for any model with a yes/no (or 0..1) output.
    #
    # Per element, with z the logit and y the target in [0, 1]:
    #
    #   l = -[y * log(sigmoid(z)) + (1 - y) * log(1 - sigmoid(z))]
    #     = softplus(z) - y * z
    #     = max(z, 0) - y * z + log(1 + e^-|z|)
    #
    # The last form never overflows, and never takes the log of 0, which
    # Losses::BCE on probabilities has to prevent with an epsilon: there a
    # confident wrong prediction (sigmoid(z) rounding to 0 or 1) gives a
    # loss capped by the epsilon and a gradient that vanishes, while here
    # the loss keeps growing linearly and the gradient stays -1 or 1.
    #
    # Backward:
    #   dl/dz = sigmoid(z) - y        dl/dy = -z
    #
    # reduction: :mean (default) averages over every element; :sum adds
    # them up, e.g. for a Bernoulli log-likelihood summed over the pixels of
    # an image.
    class BCEWithLogits
      REDUCTIONS = %i[mean sum].freeze

      def self.apply(logits, target, reduction: :mean)
        new(logits, target, reduction: reduction).forward
      end

      attr_reader :inputs

      def initialize(logits, target, reduction: :mean)
        unless REDUCTIONS.include?(reduction)
          raise ArgumentError, "reduction must be one of #{REDUCTIONS.inspect}, got #{reduction.inspect}"
        end

        @inputs = [logits, target]
        @reduction = reduction
      end

      def forward
        logits, target = @inputs
        unless logits.shape == target.shape
          raise ArgumentError, "BCEWithLogits expects logits and target of the same shape, " \
            "got #{logits.shape.inspect} and #{target.shape.inspect}"
        end

        bk = @backend = Ops.backend_of(logits, target)
        z = logits.data
        y = target.data
        per_element = bk.maximum(z, 0.0) - (y * z) + bk.log1p(bk.exp(-bk.abs(z)))
        @scale = (@reduction == :mean) ? 1.0 / logits.size : 1.0
        loss = bk.sum(per_element) * @scale

        requires_grad = Ops.requires_grad?(logits, target)
        out = Tensor.new(loss, requires_grad: requires_grad, backend: bk)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        logits, target = @inputs
        bk = @backend
        g = (grad_output.is_a?(Tensor) ? grad_output.data : grad_output) * @scale

        if logits.requires_grad
          sigmoid = 1.0 / (bk.exp(-logits.data) + 1.0)
          logits.accumulate_grad(g * (sigmoid - target.data))
        end
        target.accumulate_grad(g * -logits.data) if target.requires_grad
      end
    end
  end
end
