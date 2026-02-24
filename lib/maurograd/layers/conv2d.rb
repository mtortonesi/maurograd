require_relative '../tensor'
require_relative '../ops/conv2d'

module Maurograd
  module Layers
    # Conv2D Layer (high-level API).
    #
    # This class represents a *neural network layer*, not a low-level operation.
    #
    # Responsibilities of a Layer:
    # -----------------------------
    # - Own trainable parameters (weights and bias)
    # - Initialize parameters with a suitable strategy
    # - Expose a forward(input) method that applies the corresponding Op
    #
    # What a Layer does NOT do:
    # -------------------------
    # - It does NOT implement the actual convolution math
    # - It does NOT implement backward logic
    #
    # Those responsibilities belong to Ops::Conv2D, which builds the computation
    # graph and handles autograd.
    #
    # This separation mirrors PyTorch's design:
    # - nn.Conv2d   -> Layer (parameters + forward interface)
    # - aten/ops    -> low-level operations used by autograd
    #
    class Conv2D
      attr_accessor :weights, :bias, :stride, :padding

      # Create a 2D convolutional layer.
      #
      # Parameters:
      # - in_channels:  number of input channels (C_in)
      # - out_channels: number of output channels / filters (C_out)
      # - kernel_size:  integer (e.g. 3) or [kh, kw]
      # - stride:       convolution stride (default: 1)
      # - padding:      zero-padding applied to H and W (default: 0)
      #
      # Weight tensor shape:
      #   [C_out, C_in, KH, KW]
      #
      # Bias shape:
      #   [C_out]
      #
      # Initialization strategy:
      # ------------------------
      # We use He (Kaiming) initialization, which is standard for layers
      # followed by ReLU-like activations.
      #
      # The idea is to keep the variance of activations roughly constant
      # across layers, preventing vanishing or exploding gradients.
      #
      # The standard deviation is:
      #
      #   std = sqrt(2 / fan_in)
      #
      # where:
      #   fan_in = C_in * KH * KW
      #
      # Note:
      # -----
      # fan_in depends only on the number of *inputs* to each neuron,
      # not on out_channels.
      #
      def initialize(in_channels, out_channels, kernel_size, stride: 1, padding: 0)
        @stride = stride
        @padding = padding

        # Normalize kernel_size to (KH, KW).
        #
        # If kernel_size is an integer (e.g. 3), we assume a square kernel (3x3).
        # Otherwise, we expect [kh, kw].
        kh, kw = kernel_size.is_a?(Integer) ? [kernel_size, kernel_size] : kernel_size

        # Compute fan_in for He initialization.
        #
        # Each output unit receives inputs from:
        #   C_in channels * KH * KW spatial positions
        fan_in = in_channels * kh * kw
        std = Math.sqrt(2.0 / fan_in)

        # Initialize weights with a normal distribution N(0, std).
        #
        # Shape: [C_out, C_in, KH, KW]
        weight_data =
          Numo::SFloat
            .new(out_channels, in_channels, kh, kw)
            .rand_norm(0, std)

        @weights = Maurograd::Tensor.new(weight_data, requires_grad: true)

        # Initialize bias to zero.
        #
        # One bias value per output channel.
        # Shape: [C_out]
        @bias = Maurograd::Tensor.new(
          Numo::SFloat.zeros(out_channels),
          requires_grad: true
        )
      end

      # Forward pass of the Conv2D layer.
      #
      # This method does NOT implement convolution itself.
      # Instead, it delegates the computation to Ops::Conv2D,
      # which:
      # - builds the computation graph
      # - caches intermediate values for backward
      # - handles gradient propagation
      #
      # Parameters:
      # - input: Tensor with shape [N, C_in, H, W]
      #
      # Returns:
      # - output Tensor with shape [N, C_out, OH, OW]
      #
      def forward(input)
        Maurograd::Ops::Conv2D.apply(input, @weights, @bias, @stride, @padding)
      end

      def parameters
        [@weights, @bias]
      end
    end
  end
end

