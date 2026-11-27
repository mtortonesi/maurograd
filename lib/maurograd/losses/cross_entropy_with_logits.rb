require_relative "../tensor"
require_relative "../utils/utils"

module Maurograd
  module Losses
    # CrossEntropyWithLogits loss (multi-class), numerically stable.
    #
    # We implement CE-with-logits as a dedicated loss to:
    # 1) keep the early core small (no need for Ops::Log/Ops::Exp yet),
    # 2) provide a stable implementation via the log-sum-exp trick.
    #
    # Inputs:
    # - logits: Tensor [N, C]
    # - target: Tensor [N] with class indices, or Tensor [N, C] one-hot / class distribution
    #
    # Forward (indices):
    #   log_probs = logits - logsumexp(logits)   (per-row)
    #   L = -mean( log_probs[n, y_n] )
    #
    # Backward (indices):
    #   dL/dlogits = (softmax - one_hot(y)) / N
    #
    # Upstream grad_output is a scalar and is multiplied in.
    class CrossEntropyWithLogits
      def self.apply(logits, target)
        new(logits, target).forward
      end

      attr_reader :inputs

      def initialize(logits, target)
        @inputs = [logits, target]

        # cached for backward
        @softmax = nil   # [N, C]
        @target = nil   # raw target.data
        @is_indices = nil
        @n = nil
        @c = nil
      end

      def forward
        logits, target = @inputs
        x = logits.data
        t = target.data

        Maurograd::Utils.assert_finite!(x, where: "CEWithLogits forward: logits")

        raise "CrossEntropyWithLogits expects logits [N,C], got #{x.shape.inspect}" unless x.shape.length == 2
        @n, @c = x.shape
        bk = @backend = Ops.backend_of(logits, target)

        # Determine target format
        if t.shape.length == 1
          raise "Target [N] must match logits N=#{@n}, got #{t.shape.inspect}" unless t.shape[0] == @n
          @is_indices = true
        elsif t.shape.length == 2
          raise "Target [N,C] must match logits [#{@n},#{@c}], got #{t.shape.inspect}" unless t.shape == [@n, @c]
          @is_indices = false
        else
          raise "Target must be [N] indices or [N,C] one-hot/probs, got #{t.shape.inspect}"
        end

        # log-sum-exp trick (row-wise)
        # shift = max per row (keepdims)
        shift = bk.max(x, axis: 1, keepdims: true)           # [N,1]
        Maurograd::Utils.assert_finite!(shift, where: "CEWithLogits forward: shift")

        exps = bk.exp(x - shift)                             # [N,C]
        Maurograd::Utils.assert_finite!(exps, where: "CEWithLogits forward: exps")

        sumexp = bk.sum(exps, axis: 1, keepdims: true)       # [N,1]
        Maurograd::Utils.assert_finite!(sumexp, where: "CEWithLogits forward: sumexp")

        # softmax cached for backward
        @softmax = exps / sumexp                             # [N,C]
        Maurograd::Utils.assert_finite!(@softmax, where: "CEWithLogits forward: softmax")

        # log_probs = x - shift - log(sumexp)
        logsumexp = shift + bk.log(sumexp)                   # [N,1]
        log_probs = x - logsumexp                            # [N,C]

        # Class indices are turned into one-hot rows, so both target formats
        # share the same vectorized formula: -mean(sum(target * log_probs, axis=1)).
        # (Selecting -log_probs[n, y_n] with a one-hot mask is the same
        # "gather", without a per-sample Ruby loop.) Soft targets can be any
        # probability distribution.
        @target = @is_indices ? bk.one_hot(t, @c) : t # [N,C]
        loss_data = -bk.sum(@target * log_probs) / @n.to_f

        requires_grad = Ops.requires_grad?(logits, target)
        out = Tensor.new(loss_data, requires_grad: requires_grad, backend: bk)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        Maurograd::Utils.assert_finite!(grad_output, where: "CEWithLogits backward: grad_output")

        logits, target = @inputs

        gout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output

        # dL/dlogits = (softmax - target_one_hot) / N
        if logits.requires_grad
          dx = (@softmax - @target) * (gout / @n.to_f)

          Maurograd::Utils.assert_finite!(dx, where: "CEWithLogits backward: dx")

          logits.accumulate_grad(dx)
        end

        # Optional target gradient (rarely needed). Implemented for completeness.
        #
        # If target is one-hot/probabilities, gradient is:
        #   dL/dtarget = -(log_probs)/N
        # If target is indices, gradient w.r.t. indices is not meaningful (skip).
        if target.requires_grad && !@is_indices
          # Recompute log_probs cheaply from softmax:
          # log_probs = log(softmax)  (still stable enough since softmax in (0,1))
          # but we used stable logsumexp forward; keeping it simple here:
          log_probs = @backend.log(@softmax)
          dt = -(log_probs * (gout / @n.to_f))
          target.accumulate_grad(dt)
        end
      end
    end
  end
end
