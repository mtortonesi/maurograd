require_relative '../tensor'
require_relative '../utils/utils'

module Maurograd
  module Ops
    # MaxPool2D (NCHW) implemented with im2col + reduction.
    #
    # Input:  X [N, C, H, W]
    # Output: Y [N, C, OH, OW]
    #
    # Pooling is per-channel (no mixing across channels).
    #
    # Forward:
    # - col = im2col(X, KH, KW, stride, padding) -> [R, C*KH*KW], R = N*OH*OW
    # - reshape col -> [R, C, KH*KW]
    # - max over last axis -> [R, C]
    # - reshape back to [N, C, OH, OW]
    #
    # Backward:
    # - distribute dout only to the positions that were maximal in each window
    #   (scatter-add via col2im).
    class MaxPool2D
      def self.apply(input, kernel_h, kernel_w, stride = nil, padding = 0)
        stride ||= kernel_h
        new(input, kernel_h, kernel_w, stride, padding).forward
      end

      attr_reader :inputs, :kernel_h, :kernel_w, :stride, :padding

      def initialize(input, kernel_h, kernel_w, stride, padding)
        @inputs = [input]
        @kernel_h = kernel_h
        @kernel_w = kernel_w
        @stride = stride
        @padding = padding

        # Cached for backward:
        @input_shape = nil
        @out_h = nil
        @out_w = nil
        @mask = nil          # [R, C, KH*KW] float {0,1}
      end

      def forward
        # NOTE ON PADDING FOR MAXPOOL:
        # ----------------------------
        # Using padding in pooling is conceptually the same as convolution padding:
        # we pad the spatial dimensions with zeros BEFORE extracting windows.
        # Therefore windows near the borders may include padded zeros.
        #
        # This affects:
        # - output size OH/OW
        # - which elements can become the max in a window
        #
        # In our implementation, padding is delegated to Utils.im2col/col2im,
        # so forward/backward stay clean and consistent.
        input = @inputs.first
        x = input.data
        @input_shape = x.shape
        n, c, h, w = @input_shape

        @out_h = (h + 2 * @padding - @kernel_h) / @stride + 1
        @out_w = (w + 2 * @padding - @kernel_w) / @stride + 1

        # 1) Extract windows
        col = Utils.im2col(x, @kernel_h, @kernel_w, @stride, @padding) # [R, C*KH*KW]
        r = n * @out_h * @out_w

        # 2) Reshape so pooling is per-channel
        col3 = col.reshape(r, c, @kernel_h * @kernel_w) # [R, C, K]

        # 3) Max over K
        max_vals = col3.max(axis: 2)                    # [R, C]

        # 4) Build mask for backward (1 where equals max, else 0)
        max3 = max_vals.reshape(r, c, 1)                # [R, C, 1]
        @mask = Numo::SFloat.cast(col3.eq(max3))         # [R, C, K]

        # 5) Reshape to NCHW
        y =
          max_vals
            .reshape(n, @out_h, @out_w, c)
            .transpose(0, 3, 1, 2)
            .copy

        requires_grad = input.requires_grad
        out = Tensor.new(y, requires_grad: requires_grad)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        Maurograd::Utils.assert_finite!(grad_output, where: "MaxPool2D backward: grad_output")

        input = @inputs.first
        return unless input.requires_grad

        dout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output
        n, c, _h, _w = @input_shape
        r = n * @out_h * @out_w

        # dout: [N, C, OH, OW] -> [R, C]
        dout_col = dout.transpose(0, 2, 3, 1).copy.reshape(r, c)

        # Expand dout to [R, C, 1], multiply by mask -> [R, C, K]
        dcol3 = @mask * dout_col.reshape(r, c, 1)

        # Back to [R, C*K]
        dcol = dcol3.reshape(r, c * @kernel_h * @kernel_w)

        # Scatter back into input shape
        dx = Utils.col2im(dcol.copy, @input_shape, @kernel_h, @kernel_w, @stride, @padding)
        input.accumulate_grad(dx)
      end
    end
  end
end
