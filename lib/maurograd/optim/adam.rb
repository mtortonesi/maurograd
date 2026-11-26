require_relative "optimizer"

module Maurograd
  module Optim
    # Adam (Kingma & Ba, 2014): "adaptive moment estimation".
    #
    # Adam keeps two running averages per parameter:
    #
    #   m = beta1 * m + (1 - beta1) * g        # 1st moment: mean of gradients
    #   v = beta2 * v + (1 - beta2) * g^2      # 2nd moment: mean of squared gradients
    #
    # m plays the role of momentum. v tracks how large the gradients of each
    # individual weight usually are, so dividing by sqrt(v) gives every
    # weight its own effective step size: weights with consistently large
    # gradients take smaller steps, weights with tiny gradients take larger
    # ones.
    #
    # Both averages start at zero, so in the first steps they are biased
    # towards zero (after one step, m is only (1 - beta1) * g, i.e. 10% of
    # g with the default beta1). Bias correction undoes that:
    #
    #   m_hat = m / (1 - beta1^t)
    #   v_hat = v / (1 - beta2^t)
    #
    # where t is the step count (starting at 1). Then:
    #
    #   param = param - lr * m_hat / (sqrt(v_hat) + eps)
    #
    # A useful consequence: on the very first step m_hat = g and
    # v_hat = g^2, so every weight moves by almost exactly lr (in the
    # direction opposite to its gradient), regardless of the gradient's
    # magnitude. lr in Adam is therefore roughly "how far each weight moves
    # per step", which is why its default (1e-3) is much smaller than
    # typical SGD learning rates.
    #
    # weight_decay here is the classic L2 penalty folded into the gradient
    # (g = grad + weight_decay * param), exactly as PyTorch's Adam does. Note
    # that this interacts with the adaptive scaling (the penalty gets divided
    # by sqrt(v_hat) too), which is what AdamW's "decoupled" weight decay
    # was later designed to avoid.
    #
    # Defaults match PyTorch's torch.optim.Adam.
    class Adam < Optimizer
      attr_reader :lr, :beta1, :beta2, :eps, :weight_decay, :t

      def initialize(params, lr: 1e-3, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0.0)
        super(params)
        @beta1, @beta2 = betas
        [@beta1, @beta2].each do |b|
          raise ArgumentError, "betas must be in [0, 1), got #{betas.inspect}" unless b >= 0.0 && b < 1.0
        end
        raise ArgumentError, "weight_decay must be >= 0, got #{weight_decay}" if weight_decay.negative?

        @lr = lr
        @eps = eps
        @weight_decay = weight_decay

        # Step counter, shared by all parameters (used for bias correction).
        @t = 0

        # Moment estimates, aligned with @params (allocated lazily).
        @m = Array.new(params.size)
        @v = Array.new(params.size)
      end

      def step
        @t += 1
        bias_correction1 = 1.0 - (@beta1**@t)
        bias_correction2 = 1.0 - (@beta2**@t)

        each_updatable_param do |p, grad, i|
          g = grad
          g += @weight_decay * p.data if @weight_decay > 0

          @m[i] ||= p.backend.zeros(p.shape)
          @v[i] ||= p.backend.zeros(p.shape)

          @m[i] = (@beta1 * @m[i]) + ((1.0 - @beta1) * g)
          @v[i] = (@beta2 * @v[i]) + ((1.0 - @beta2) * (g**2))

          m_hat = @m[i] / bias_correction1
          v_hat = @v[i] / bias_correction2

          p.data -= @lr * m_hat / (p.backend.sqrt(v_hat) + @eps)
        end
      end
    end
  end
end
