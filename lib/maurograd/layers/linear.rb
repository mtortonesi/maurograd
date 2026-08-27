require_relative '../tensor'
require_relative '../ops/linear'
require_relative '../utils/utils'

module Maurograd
  module Layers
    # Linear layer (high-level API).
    #
    # Owns parameters:
    # - weights: [Out, In]
    # - bias:    [Out]
    #
    # Delegates actual computation + autograd logic to Ops::Linear.
    class Linear
      attr_accessor :weights, :bias

      # - seed: optional seed for reproducible weight initialization.
      #         Without it, weight init draws from Numo's global RNG, which
      #         Maurograd.seed!(n) can seed for a whole run at once instead
      #         of passing seed: to every layer.
      def initialize(in_features, out_features, bias: true, seed: nil)
        # He initialization is usually associated with ReLU.
        # For a generic Linear layer, Xavier (Glorot) is also common.
        # We'll keep He for consistency with your Conv2D layer style,
        # and because many MLPs will use ReLU.
        fan_in = in_features
        std = Math.sqrt(2.0 / fan_in)

        w = Maurograd::Utils.rand_norm(out_features, in_features, std: std, seed: seed)
        @weights = Maurograd::Tensor.new(w, requires_grad: true)

        if bias
          @bias = Maurograd::Tensor.new(Numo::SFloat.zeros(out_features), requires_grad: true)
        else
          @bias = nil
        end
      end

      def forward(input)
        Maurograd::Ops::Linear.apply(input, @weights, @bias)
      end

      def parameters
        ps = [@weights]
        ps << @bias if @bias
        ps
      end
    end 
  end
end
