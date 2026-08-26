require_relative '../tensor'
require_relative '../utils/utils'
require 'numo/narray'

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
        @target  = nil   # raw target.data
        @is_indices = nil
        @n = nil
        @c = nil
      end

      def forward
        logits, target = @inputs
        x = logits.data
        t = target.data
        
        Maurograd::Utils.assert_finite!(x, where: "CEWithLogits forward: logits")


        raise "CrossEntropyWithLogits expects logits [N,C], got #{x.shape.inspect}" unless x.ndim == 2
        @n, @c = x.shape

        # Determine target format
        if t.ndim == 1
          raise "Target [N] must match logits N=#{@n}, got #{t.shape.inspect}" unless t.shape[0] == @n
          @is_indices = true
        elsif t.ndim == 2
          raise "Target [N,C] must match logits [#{@n},#{@c}], got #{t.shape.inspect}" unless t.shape == [@n, @c]
          @is_indices = false
        else
          raise "Target must be [N] indices or [N,C] one-hot/probs, got #{t.shape.inspect}"
        end

        # log-sum-exp trick (row-wise)
        # shift = max per row (keepdims)
        shift = x.max(axis: 1, keepdims: true)               # [N,1]
        Maurograd::Utils.assert_finite!(shift, where: "CEWithLogits forward: shift")

        exps  = Numo::NMath.exp(x - shift)                   # [N,C]
        Maurograd::Utils.assert_finite!(exps, where: "CEWithLogits forward: exps")

        sumexp = exps.sum(axis: 1, keepdims: true)           # [N,1]
        Maurograd::Utils.assert_finite!(sumexp, where: "CEWithLogits forward: sumexp")


        # softmax cached for backward
        @softmax = exps / sumexp                             # [N,C]
        Maurograd::Utils.assert_finite!(@softmax, where: "CEWithLogits forward: softmax")


        # log_probs = x - shift - log(sumexp)
        logsumexp = shift + Numo::NMath.log(sumexp)          # [N,1]
        log_probs = x - logsumexp                            # [N,C]

        loss_data =
          if @is_indices
            # gather -log_probs[n, y_n]
            ys = t.to_a.map(&:to_i)
            acc = 0.0
            ys.each_with_index do |yy, i|
              raise "Class index out of range: #{yy} for C=#{@c}" if yy < 0 || yy >= @c
              acc += -log_probs[i, yy]
            end
            acc / @n.to_f
          else
            # one-hot / distribution: -mean(sum(target * log_probs, axis=1))
            # Here t can be one-hot or probability distribution.
            per_row = (t * log_probs).sum(axis: 1)           # [N]
            -(per_row.sum / @n.to_f)
          end

        @target = t

        requires_grad = logits.requires_grad || target.requires_grad
        out = Tensor.new(loss_data, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        Maurograd::Utils.assert_finite!(grad_output, where: "CEWithLogits backward: grad_output")

        logits, target = @inputs

        gout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        gout = Numo::SFloat.cast(gout)

        # dL/dlogits = (softmax - target_one_hot) / N
        if logits.requires_grad
          dx = @softmax.dup

          if @is_indices
            ys = @target.to_a.map(&:to_i)
            ys.each_with_index do |yy, i|
              dx[i, yy] -= 1.0
            end
          else
            dx -= @target
          end

          dx *= (gout / @n.to_f)

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
          log_probs = Numo::NMath.log(@softmax)
          dt = -(log_probs * (gout / @n.to_f))
          target.accumulate_grad(dt)
        end
      end
    end
  end
end
