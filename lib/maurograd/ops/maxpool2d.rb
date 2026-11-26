require_relative "../tensor"
require_relative "../utils/utils"

module Maurograd
  module Ops
    # MaxPool2D (NCHW).
    #
    # Input:  X [N, C, H, W]
    # Output: Y [N, C, OH, OW]
    #
    # Pooling is per-channel (no mixing across channels).
    #
    # Both passes are backend kernels; on the CPU backend, C functions in
    # maurograd_ext (Ext.maxpool2d_forward/backward):
    #
    # Forward: for every output position, scan its KH x KW window and record
    # both the maximum (-> Y) and WHERE it was (-> @argmax, the flat index
    # into X of the winning element, or -1 if it was a padding cell).
    #
    # Backward: max(...) passes its gradient only to the element that won,
    # so dX is all zeros except dX[argmax] += dY for each output (+= because
    # overlapping windows, i.e. stride < kernel, can pick the same element).
    #
    # Ties go to the first maximum in the window (row-major order), so each
    # output's gradient lands on exactly one input - as in PyTorch.
    #
    # An earlier version built the full im2col window matrix [R, C*KH*KW]
    # plus a float mask of the same size with Numo, and scattered back with
    # col2im. Same result (bar ties, which it split across ALL tied
    # elements), but several times slower and far more memory-hungry.
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

        # Cached for backward: backend-specific (on the CPU backend, an
        # Int32 [N, C, OH, OW] array of flat input indices, -1 = padding).
        @ctx = nil
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
        # - which elements can become the max in a window (a padded zero
        #   beats an all-negative window; its gradient then goes nowhere)
        #
        # Note that PyTorch pads with -inf instead, so padding never wins
        # there. With non-negative inputs (e.g. after ReLU, the usual case)
        # the two conventions agree.
        input = @inputs.first

        y, ctx = input.backend.maxpool2d_forward(input.data, @kernel_h, @kernel_w, @stride, @padding)

        requires_grad = input.requires_grad
        @ctx = ctx if requires_grad

        out = Tensor.new(y, requires_grad: requires_grad, backend: @inputs.first.backend)
        out.creator = self if requires_grad
        out
      end

      def backward(grad_output)
        Maurograd::Utils.assert_finite!(grad_output, where: "MaxPool2D backward: grad_output")

        input = @inputs.first
        return unless input.requires_grad

        dout = grad_output.is_a?(Tensor) ? grad_output.data : grad_output

        dx = input.backend.maxpool2d_backward(@ctx, dout)
        input.accumulate_grad(dx)
      end
    end
  end
end
