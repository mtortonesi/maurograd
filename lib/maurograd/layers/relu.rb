require_relative "../ops/relu"
require_relative "module"

module Maurograd
  module Layers
    class ReLU < Module
      def forward(input)
        Maurograd::Ops::ReLU.apply(input)
      end
    end
  end
end
