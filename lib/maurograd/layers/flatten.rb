require_relative '../ops/flatten'

module Maurograd
  module Layers
    class Flatten
      def forward(input)
        Maurograd::Ops::Flatten.apply(input)
      end
    end
  end
end

