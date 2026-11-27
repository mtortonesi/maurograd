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
        @backend = bk = Ops.backend_of(*@inputs)

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
        y = bk.matmul(x, bk.transpose(w, 1, 0)) # [N, Out]

        # Add bias (broadcast over batch).
        if bias
          # bias can be [Out] or [1, Out]; reshape to [1, Out] then broadcast over N.
          y += bk.reshape(bias.data, 1, w.shape[0])
        end

        requires_grad = Ops.requires_grad?(*@inputs)
        out = Tensor.new(y, requires_grad: requires_grad, backend: @inputs.first.backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        Maurograd::Utils.assert_finite!(grad_output, where: "Linear backward: grad_output")

        input, weight, bias = @inputs
        dy = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        bk = @backend
        # dy: [N, Out]

        # A) Bias gradient: db = sum over batch axis -> [Out]
        if bias&.requires_grad
          db = bk.sum(dy, axis: 0) # [Out]
          bias.accumulate_grad(bk.reshape(db, *bias.shape))
        end

        # B) Weight gradient: dw = dy^T @ X  -> [Out, In]
        if weight.requires_grad
          dw = bk.matmul(bk.transpose(dy, 1, 0), @x) # [Out, In]
          weight.accumulate_grad(dw)
        end

        # C) Input gradient: dx = dy @ W -> [N, In]
        if input.requires_grad
          dx = bk.matmul(dy, @w) # [N, In]
          input.accumulate_grad(dx)
        end
      end
    end
  end
end
