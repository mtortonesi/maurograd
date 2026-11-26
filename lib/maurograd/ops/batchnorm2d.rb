require_relative "../tensor"
require_relative "../utils/utils"

module Maurograd
  module Ops
    # 2D Batch Normalization (per-channel), for NCHW tensors.
    #
    # Forward (training):
    #   mu    = mean(x, over N,H,W)                       [C]
    #   var   = mean((x - mu)^2, over N,H,W)               [C]  (biased)
    #   x_hat = (x - mu) / sqrt(var + eps)
    #   y     = gamma * x_hat + beta
    #
    #   running_mean/running_var are updated in place (exponential moving
    #   average). Matches the common convention: the *unbiased* variance
    #   estimate (Bessel's correction, factor m/(m-1)) is used for the
    #   running estimate, while the *biased* variance is used to actually
    #   normalize the batch.
    #
    # Forward (eval):
    #   x_hat = (x - running_mean) / sqrt(running_var + eps)
    #   y     = gamma * x_hat + beta
    #   (no running-stat update; batch statistics are never computed)
    #
    # Backward (training) follows the standard derivation (Ioffe & Szegedy,
    # 2015), broken into the same steps as the forward pass rather than the
    # single compressed closed-form expression, so each intermediate value
    # is independently checkable:
    #
    #   dgamma = sum(dout * x_hat,      over N,H,W)
    #   dbeta  = sum(dout,              over N,H,W)
    #   dxhat  = dout * gamma
    #   dvar   = sum(dxhat * (x - mu),  over N,H,W) * -0.5 * (var+eps)^-1.5
    #   dmu    = sum(dxhat,             over N,H,W) * -1/sqrt(var+eps)
    #            + dvar * mean(-2*(x - mu), over N,H,W)
    #   dx     = dxhat / sqrt(var+eps) + dvar * 2*(x-mu)/m + dmu/m
    #
    # Backward (eval) is simpler: running_mean/running_var are fixed
    # constants there, not functions of x, so there's no dvar/dmu term:
    #
    #   dx = dxhat / sqrt(running_var + eps)
    #
    # Implementation: the per-element math runs in the C extension
    # (Ext.batchnorm2d_forward/backward), one channel at a time, with the
    # sums accumulated in double precision. Substituting dvar and dmu into dx
    # above and simplifying gives the compact per-channel form it uses:
    #
    #   dx = gamma * inv_std / m * (m * dout - sum(dout) - x_hat * sum(dout * x_hat))
    #
    # (using sum(x - mu) = 0, which makes dmu's second term vanish). Only
    # the running-statistics update, which works on [C]-sized arrays, stays
    # in Ruby.
    #
    class BatchNorm2D
      def self.apply(input, gamma, beta, running_mean:, running_var:, training: true, momentum: 0.1, eps: 1e-5)
        new(input, gamma, beta, running_mean, running_var, training, momentum, eps).forward
      end

      attr_reader :inputs

      def initialize(input, gamma, beta, running_mean, running_var, training, momentum, eps)
        @inputs = [input, gamma, beta].compact
        @input = input
        @gamma = gamma
        @beta = beta
        @running_mean = running_mean
        @running_var = running_var
        @training = training
        @momentum = momentum
        @eps = eps
      end

      def forward
        x = Utils.sfloat_contiguous(@input.data)
        n, _c, h, w = x.shape
        m = n * h * w

        gamma = @gamma && Utils.sfloat_contiguous(@gamma.data)
        beta = @beta && Utils.sfloat_contiguous(@beta.data)

        y, x_hat, mean, var, inv_std =
          if @training
            Maurograd::Ext.batchnorm2d_forward(x, nil, nil, gamma, beta, @eps)
          else
            Maurograd::Ext.batchnorm2d_forward(
              x, Utils.sfloat_contiguous(@running_mean), Utils.sfloat_contiguous(@running_var),
              gamma, beta, @eps
            )
          end

        if @training
          # Exponential moving average into the running buffers, in place
          # (`arr[] = ...` overwrites arr's contents without rebinding it,
          # which matters since the Layer holds the same array by reference).
          # Unbiased variance estimate (Bessel's correction) for the running
          # value, matching the common convention; `var` itself is biased.
          unbiased_var = var * (m.to_f / (m - 1))
          @running_mean[] = (@running_mean * (1.0 - @momentum)) + (mean.reshape(*@running_mean.shape) * @momentum)
          @running_var[] = (@running_var * (1.0 - @momentum)) + (unbiased_var.reshape(*@running_var.shape) * @momentum)
        end

        requires_grad = @inputs.any?(&:requires_grad)
        if requires_grad
          # Cached for backward. In training mode inv_std comes from the
          # *batch* statistics; in eval mode from the running ones (fixed
          # constants, hence no dvar/dmu term there).
          @x_hat = x_hat
          @inv_std = inv_std
          @gamma_data = gamma
        end

        out = Tensor.new(y, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        dout = Utils.sfloat_contiguous(grad_output)

        dx, dgamma, dbeta = Maurograd::Ext.batchnorm2d_backward(dout, @x_hat, @inv_std, @gamma_data, @training)

        @gamma.accumulate_grad(dgamma) if @gamma&.requires_grad
        @beta.accumulate_grad(dbeta) if @beta&.requires_grad
        @input.accumulate_grad(dx) if @input.requires_grad
      end
    end
  end
end
