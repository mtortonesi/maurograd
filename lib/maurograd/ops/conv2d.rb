require_relative '../tensor'
require_relative '../utils/utils'

module Maurograd
  module Ops
    # Conv2D operation (NCHW) implemented via im2col + GEMM.
    #
    # This class is an autograd "Op":
    # - forward() builds the output Tensor and stores any context needed for backward()
    # - backward(dout) propagates gradients to inputs (input, weight, bias)
    #
    # Shapes (PyTorch-style, channels-first):
    # - input  X: [N, C_in, H, W]
    # - weight W: [C_out, C_in, KH, KW]
    # - bias   b: [C_out] (optional)
    # - output Y: [N, C_out, OH, OW]
    #
    # Hyperparameters:
    # - stride:  integer step of the sliding kernel
    # - padding: integer zero-padding applied to H and W
    #
    # Output spatial sizes:
    #   OH = (H + 2*padding - KH) / stride + 1
    #   OW = (W + 2*padding - KW) / stride + 1
    #
    # Implementation overview (forward):
    # ---------------------------------
    # 1) im2col(X) produces a 2D matrix X_col of shape:
    #      X_col: [N*OH*OW, C_in*KH*KW]
    #
    #    Each row corresponds to one receptive field (one output spatial location).
    #
    # 2) Flatten weights into W_col with shape:
    #      W_col: [C_in*KH*KW, C_out]
    #
    # 3) Compute:
    #      Y_col = X_col dot W_col   -> [N*OH*OW, C_out]
    #    Then add bias (broadcasted over rows) if present.
    #
    # 4) Reshape Y_col back into NCHW:
    #      [N, OH, OW, C_out] -> transpose -> [N, C_out, OH, OW]
    #
    # Backward overview:
    # ------------------
    # Let dout be dL/dY with shape [N, C_out, OH, OW].
    #
    # We reshape it to:
    #   dout_col: [N*OH*OW, C_out]
    #
    # Then:
    # - db = sum(dout over N,OH,OW) -> [C_out]
    # - dW_col = X_col^T dot dout_col -> [C_in*KH*KW, C_out]
    # - dX_col = dout_col dot W_flat  -> [N*OH*OW, C_in*KH*KW]
    #   and finally col2im(dX_col) -> dX: [N, C_in, H, W]
    #
    class Conv2D

      def self.apply(input, weight, bias = nil, stride = 1, padding = 0)
        new(input, weight, bias, stride, padding).forward
      end

     
      attr_reader :inputs, :stride, :padding


      def initialize(input, weight, bias = nil, stride = 1, padding = 0)
        # Store only non-nil inputs so that graph traversal is easy.
        # Backward must mirror the same order.
        @inputs = [input, weight, bias].compact
        @stride = stride
        @padding = padding

        @workspace = {}
      end


      def forward
        input, weight, bias = @inputs

        x_data = if input.data.contiguous?
                   input.data
                 else
                   input.data.copy
                 end

        w_data = if weight.data.contiguous?
                   weight.data
                 else
                   weight.data.copy
                 end

        b_data = if bias.nil?
                   nil
                 elsif bias.data.contiguous?
                   bias.data
                 else
                   bias.data.copy
                 end

        out_data = Maurograd::Ext.conv2d_forward(x_data, w_data, b_data, @stride, @padding, @workspace)

        requires_grad = @inputs.any?(&:requires_grad)
        if requires_grad
          @x_shape = input.data.shape
          @w_shape = weight.data.shape
        end

        out = Tensor.new(out_data, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end


      def backward(grad_output)
        input, weight, bias = @inputs

        go = grad_output.is_a?(Maurograd::Tensor) ? grad_output.data : grad_output

        x_data = if input.data.contiguous?
                   input.data
                 else
                   input.data.copy
                 end

        w_data = if weight.data.contiguous?
                   weight.data
                 else
                   weight.data.copy
                 end



        unless go.contiguous?
          go = go.copy
        end

        want_dx = input.requires_grad ? 1 : 0
        want_dw = weight.requires_grad ? 1 : 0
        want_db = (bias && bias.requires_grad) ? 1 : 0

        dx, dw, db = Maurograd::Ext.conv2d_backward(
          x_data, w_data, go,
          @stride, @padding,
          want_dx, want_dw, want_db,
          @workspace
        )

        if bias && bias.grad && bias.grad.shape != db.shape
          raise "db shape mismatch: grad=#{bias.grad.shape} db=#{db.shape}"
        end

        if weight.grad && weight.grad.shape != dw.shape
          raise "dw shape mismatch: grad=#{weight.grad.shape} dw=#{dw.shape}"
        end

        if input.grad && input.grad.shape != dx.shape
          raise "dx shape mismatch: grad=#{input.grad.shape} dx=#{dx.shape}"
        end

        if want_db == 1
          bias.accumulate_grad(db)
        end

        if want_dw == 1
          weight.accumulate_grad(dw)
        end

        if want_dx == 1
          input.accumulate_grad(dx)
        end
      end

    end
  end
end

