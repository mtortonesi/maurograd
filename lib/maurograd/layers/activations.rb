require_relative "../ops/elementwise"
require_relative "module"

module Maurograd
  module Layers
    # Activation layers without parameters, for use in Sequential; in a
    # custom Module's forward, calling x.sigmoid, x.tanh or x.silu directly
    # is the same thing. (ReLU has its own file, and a C kernel.)

    class Sigmoid < Module
      def forward(input) = input.sigmoid
    end

    class Tanh < Module
      def forward(input) = input.tanh
    end

    # SiLU / swish: x * sigmoid(x).
    class SiLU < Module
      def forward(input) = input.silu
    end

    # Leaky ReLU, with a fixed negative slope (default 0.01).
    class LeakyReLU < Module
      attr_reader :negative_slope

      def initialize(negative_slope: 0.01)
        @negative_slope = negative_slope
      end

      def forward(input) = input.leaky_relu(negative_slope: @negative_slope)
    end

    # Softplus, log(1 + e^x): a smooth, positive ReLU.
    class Softplus < Module
      def forward(input) = input.softplus
    end
  end
end
