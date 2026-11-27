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
    # Implementation: the per-element math is a backend kernel (on the CPU
    # backend, Ext.batchnorm2d_forward/backward in C, one channel at a time,
    # with the sums accumulated in double precision). Substituting dvar and dmu into dx
    # above and simplifying gives the compact per-channel form it uses:
    #
    #   dx = gamma * inv_std / m * (m * dout - sum(dout) - x_hat * sum(dout * x_hat))
    #
    # (using sum(x - mu) = 0, which makes dmu's second term vanish). Only
    # the running-statistics update, which works on [C]-sized arrays, stays
    # in Ruby.
    #
    class BatchNorm2D
      # `stats` holds the running statistics: any object with
      # running_mean/running_var readers and writers (Layers::BatchNorm2D
      # passes itself). In training mode they are replaced with their
      # updated values; arrays are never modified in place, since not every
      # backend's arrays are mutable.
      def self.apply(input, gamma, beta, stats:, training: true, momentum: 0.1, eps: 1e-5)
        new(input, gamma, beta, stats, training, momentum, eps).forward
      end

      attr_reader :inputs

      def initialize(input, gamma, beta, stats, training, momentum, eps)
        @inputs = [input, gamma, beta].compact
        @input = input
        @gamma = gamma
        @beta = beta
        @stats = stats
        @training = training
        @momentum = momentum
        @eps = eps
      end

      def forward
        @backend = bk = Ops.backend_of(*@inputs)
        n, _c, h, w = @input.shape
        m = n * h * w

        mean, var = @training ? [nil, nil] : [@stats.running_mean, @stats.running_var]
        y, @ctx, batch_mean, batch_var =
          bk.batchnorm2d_forward(@input.data, mean, var, @gamma&.data, @beta&.data, @eps)

        if @training
          # Exponential moving average of the batch statistics. Unbiased
          # variance estimate (Bessel's correction) for the running value,
          # matching the common convention; batch_var itself is biased.
          shape = @stats.running_mean.shape
          unbiased_var = batch_var * (m.to_f / (m - 1))
          @stats.running_mean = (@stats.running_mean * (1.0 - @momentum)) + (bk.reshape(batch_mean, *shape) * @momentum)
          @stats.running_var = (@stats.running_var * (1.0 - @momentum)) + (bk.reshape(unbiased_var, *shape) * @momentum)
          # Nothing reads the running statistics during training, so on a
          # lazy backend each update would chain onto the previous one for a
          # whole epoch: start computing them now instead.
          bk.eval_async([@stats.running_mean, @stats.running_var])
        end

        # ctx (cached for backward) holds x_hat and inv_std. In training
        # mode inv_std comes from the *batch* statistics; in eval mode from
        # the running ones (fixed constants, hence no dvar/dmu term there).
        requires_grad = Ops.requires_grad?(*@inputs)
        @ctx = nil unless requires_grad

        out = Tensor.new(y, requires_grad: requires_grad, backend: bk)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        dout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output

        dx, dgamma, dbeta = @backend.batchnorm2d_backward(@ctx, dout, training: @training)

        @gamma.accumulate_grad(dgamma) if @gamma&.requires_grad
        @beta.accumulate_grad(dbeta) if @beta&.requires_grad
        @input.accumulate_grad(dx) if @input.requires_grad
      end
    end
  end
end
