require_relative '../tensor'

module Maurograd
  module Losses
    # Mean Squared Error (MSE) loss.
    #
    # Given:
    #   pred   (Tensor)
    #   target (Tensor), with the exact same shape as pred
    #
    # Returns a scalar loss:
    #   mean((pred - target)^2)
    #
    # Backward:
    #   dL/dpred   = (2 / N) * (pred - target) * dL/dL
    #   dL/dtarget = -(2 / N) * (pred - target) * dL/dL   (only if target.requires_grad)
    #
    # where N is the number of elements.
    class MSE
      def self.apply(pred, target)
        new(pred, target).forward
      end

      attr_reader :inputs

      def initialize(pred, target)
        @inputs = [pred, target]
        @diff = nil
        @n_elems = nil
      end

      def forward
        pred, target = @inputs

        unless pred.shape == target.shape
          raise "MSE expects pred and target to have the same shape, got #{pred.shape.inspect} and #{target.shape.inspect}"
        end

        # Compute diff in raw Numo space (simple, reliable).
        @diff = pred.data - target.data

        # Mean squared error (scalar).
        @n_elems = @diff.size
        loss_data = (@diff ** 2).sum / @n_elems.to_f

        requires_grad = pred.requires_grad || target.requires_grad
        out = Tensor.new(loss_data, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        pred, target = @inputs

        # grad_output is dL/d(out). For a scalar loss it is usually 1.0.
        gout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        gout = Numo::SFloat.cast(gout)

        # Scale factor: (2/N) * grad_output
        scale = (2.0 / @n_elems.to_f) * gout

        if pred.requires_grad
          gpred = scale * @diff
          pred.accumulate_grad(gpred)
        end

        if target.requires_grad
          gtgt = -scale * @diff
          target.accumulate_grad(gtgt)
        end
      end
    end
  end
end

