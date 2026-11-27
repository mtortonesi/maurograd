require_relative "../ops/flatten"
require_relative "module"

module Maurograd
  module Layers
    class Flatten < Module
      def forward(input)
        Maurograd::Ops::Flatten.apply(input)
      end
    end
  end
end
