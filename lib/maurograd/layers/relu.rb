require_relative '../ops/relu'

module Maurograd
  module Layers
    class ReLU
      def forward(input)
        Maurograd::Ops::ReLU.apply(input)
      end
    end
  end
end

