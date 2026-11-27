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
  end
end
