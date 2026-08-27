require_relative "../tensor"
require_relative "../utils/utils"

module Maurograd
  module Ops
    # Linear (fully connected) operation.
    #
    # This is the tensor analogue of a "dense layer":
    #   Y = X @ W^T + b
    #
    # Shapes (we keep it minimal and explicit):
    # - X: [N, In]           (batch of vectors)
    # - W: [Out, In]         (one row per output unit)
    # - b: [Out] or [1, Out] (optional)
    # - Y: [N, Out]
    #
    # Why W is [Out, In] (and not [In, Out])?
    # ---------------------------------------
    # It's consistent with Conv2D weights being "out-first".
    # Each output unit has its own weight vector of length In.
    #
    # Forward:
    # - Compute X @ W^T using Numo dot:
    #     X: [N, In]
    #     W^T: [In, Out]
    #     -> [N, Out]
    # - Add bias (broadcast over N) if present.
    #
    # Backward:
    # - dX = dY @ W        where W is [Out, In]
    # - dW = dY^T @ X      -> [Out, In]
    # - db = sum(dY over batch axis) -> [Out] (then reshape to bias.shape)
    #
    class Linear
      def self.apply(input, weight, bias = nil)
        new(input, weight, bias).forward
      end

      attr_reader :inputs

      def initialize(input, weight, bias = nil)
        @inputs = [input, weight, bias].compact

        # Cached for backward (Numo arrays)
        @x = nil
        @w = nil
      end

      def forward
        input, weight, bias = @inputs

        x = input.data
        w = weight.data

        # Cache raw arrays for backward.
        @x = x
        @w = w

        # Shapes sanity (didactic; can be removed later if you want).
        # X: [N, In]
        # W: [Out, In]
        in_dim = x.shape[-1]
        raise "Linear: input last dim #{in_dim} must match weight In #{w.shape[1]}" if in_dim != w.shape[1]

        # Y = X @ W^T
        y = x.dot(w.transpose(1, 0)) # [N, Out]

        # Add bias (broadcast over batch).
        if bias
          # bias can be [Out] or [1, Out]; reshape to [1, Out] then broadcast over N.
          y.inplace + bias.data.reshape(1, w.shape[0])
        end

        requires_grad = @inputs.any?(&:requires_grad)
        out = Tensor.new(y, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        Maurograd::Utils.assert_finite!(grad_output, where: "Linear backward: grad_output")

        input, weight, bias = @inputs
        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        # dy: [N, Out]

        # A) Bias gradient: db = sum over batch axis -> [Out]
        if bias&.requires_grad
          db = dy.sum(axis: 0) # [Out]
          bias.accumulate_grad(db.copy.reshape(*bias.shape))
        end

        # B) Weight gradient: dw = dy^T @ X  -> [Out, In]
        if weight.requires_grad
          dw = dy.transpose(1, 0).dot(@x) # [Out, In]
          weight.accumulate_grad(dw)
        end

        # C) Input gradient: dx = dy @ W -> [N, In]
        if input.requires_grad
          dx = dy.dot(@w) # [N, In]
          input.accumulate_grad(dx)
        end
      end
    end
  end
end
