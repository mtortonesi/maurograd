require_relative "../tensor"

module Maurograd
  module Losses
    # Binary Cross-Entropy (BCE) loss for probabilities.
    #
    # We implement BCE as a dedicated Op for two reasons:
    # 1) Didactic clarity: BCE has a "canonical" formula and a simple closed-form gradient.
    # 2) We avoid requiring Log/Exp Ops in the core right now, keeping the early framework small.
    #
    # Formula (mean over all elements):
    #   L = - mean( y * log(p + eps) + (1 - y) * log(1 - p + eps) )
    #
    # Gradient w.r.t. p:
    #   dL/dp = (1/N) * ( (1 - y)/(1 - p + eps) - y/(p + eps) )
    # and then multiply by upstream grad_output (scalar).
    #
    # Parameters:
    # - pred:   Tensor of probabilities (0..1), any shape
    # - target: Tensor with the exact same shape as pred
    # - eps: small constant for numerical stability
    class BCE
      def self.apply(pred, target, eps: 1e-8)
        new(pred, target, eps: eps).forward
      end

      attr_reader :inputs

      def initialize(pred, target, eps: 1e-8)
        @inputs = [pred, target]
        @eps = eps
        @p = nil
        @y = nil
        @n_elems = nil
      end

      def forward
        pred, target = @inputs

        unless pred.shape == target.shape
          raise "BCE expects pred and target to have the same shape, got #{pred.shape.inspect} and #{target.shape.inspect}"
        end

        # Cache raw arrays for backward.
        @p = pred.data
        @y = target.data

        # Compute BCE in raw Numo space.
        term1 = @y * Numo::NMath.log(@p + @eps)
        term2 = (1.0 - @y) * Numo::NMath.log(1.0 - @p + @eps)

        @n_elems = term1.size
        loss_data = -(term1 + term2).sum / @n_elems.to_f

        requires_grad = pred.requires_grad || target.requires_grad
        out = Tensor.new(loss_data, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        pred, target = @inputs

        gout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        gout = Numo::SFloat.cast(gout)

        # dL/dp (mean reduction) times upstream gradient.
        #
        # dL/dp = (1/N) * [ (1-y)/(1-p+eps) - y/(p+eps) ]
        inv_n = 1.0 / @n_elems.to_f
        dldp = inv_n * ((1.0 - @y) / (1.0 - @p + @eps) - (@y / (@p + @eps)))
        gpred = gout * dldp

        if pred.requires_grad
          pred.accumulate_grad(gpred)
        end

        # Optional: gradient w.r.t. target (rarely used in practice),
        # but we implement it for completeness.
        #
        # dL/dy = -(1/N) * [ log(p+eps) - log(1-p+eps) ]
        if target.requires_grad
          dldy = -inv_n * (Numo::NMath.log(@p + @eps) - Numo::NMath.log(1.0 - @p + @eps))
          gtgt = gout * dldy
          target.accumulate_grad(gtgt)
        end
      end
    end
  end
end
