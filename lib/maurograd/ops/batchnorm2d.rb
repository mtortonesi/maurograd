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
        x = @input.data
        n, _c, h, w = x.shape
        m = n * h * w

        if @training
          mean = x.mean(axis: [0, 2, 3], keepdims: true) # [1,C,1,1]
          diff = x - mean
          var = (diff**2).mean(axis: [0, 2, 3], keepdims: true) # biased, [1,C,1,1]

          # Exponential moving average into the running buffers, in place
          # (`arr[] = ...` overwrites arr's contents without rebinding it,
          # which matters since the Layer holds the same array by reference).
          # Unbiased variance estimate (Bessel's correction) for the running
          # value, matching the common convention.
          unbiased_var = var * (m.to_f / (m - 1))
          @running_mean[] = (@running_mean * (1.0 - @momentum)) + (mean.reshape(*@running_mean.shape) * @momentum)
          @running_var[] = (@running_var * (1.0 - @momentum)) + (unbiased_var.reshape(*@running_var.shape) * @momentum)
        else
          c = x.shape[1]
          mean = @running_mean.reshape(1, c, 1, 1)
          var = @running_var.reshape(1, c, 1, 1)
          diff = x - mean
        end

        inv_std = 1.0 / Numo::NMath.sqrt(var + @eps)
        x_hat = diff * inv_std

        y = x_hat
        if @gamma
          c = x.shape[1]
          y *= @gamma.data.reshape(1, c, 1, 1)
        end
        if @beta
          c = x.shape[1]
          y += @beta.data.reshape(1, c, 1, 1)
        end

        # Cached for backward (training mode only needs mean/var/diff/inv_std
        # to be the *batch* statistics; eval mode's backward doesn't touch
        # them, since they're fixed constants there).
        @x_hat = x_hat
        @inv_std = inv_std
        @diff = diff
        @m = m

        requires_grad = @inputs.any?(&:requires_grad)
        out = Tensor.new(y, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        dout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        c = @input.data.shape[1]

        if @gamma&.requires_grad
          dgamma = (dout * @x_hat).sum(axis: [0, 2, 3])
          @gamma.accumulate_grad(dgamma)
        end

        if @beta&.requires_grad
          dbeta = dout.sum(axis: [0, 2, 3])
          @beta.accumulate_grad(dbeta)
        end

        return unless @input.requires_grad

        dxhat = @gamma ? dout * @gamma.data.reshape(1, c, 1, 1) : dout

        if @training
          m = @m
          var_eps = (1.0 / @inv_std)**2 # var + eps, recovered from inv_std

          dvar = (dxhat * @diff).sum(axis: [0, 2, 3], keepdims: true) * -0.5 * var_eps**-1.5
          dmu = (dxhat * -@inv_std).sum(axis: [0, 2, 3], keepdims: true) +
            dvar * (@diff * -2.0).mean(axis: [0, 2, 3], keepdims: true)

          dx = (dxhat * @inv_std) + (dvar * @diff * (2.0 / m)) + (dmu * (1.0 / m))
        else
          # running_mean/running_var are fixed constants in eval mode, not
          # functions of x, so there's no dvar/dmu term at all.
          dx = dxhat * @inv_std
        end

        @input.accumulate_grad(dx)
      end
    end
  end
end
