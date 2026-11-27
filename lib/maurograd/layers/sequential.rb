require_relative "module"
require_relative "../utils/utils"

module Maurograd
  module Layers
    # A chain of layers: the output of each is the input of the next.
    #
    # Everything else (parameters, train/eval, to) comes from Module, which
    # finds the layers in @layers. Their parameters are named after their
    # position: "layers.0.weights", "layers.3.bias".
    class Sequential < Module
      attr_reader :layers

      def initialize(*layers)
        @layers = layers.flatten
        @layers.each_with_index do |layer, i|
          next if layer.is_a?(Module)

          raise TypeError, "layer #{i} (#{layer.class}) is not a Maurograd::Layers::Module: " \
            "subclass it, so that its parameters and train/eval mode are found"
        end
      end

      def forward(x)
        @layers.each_with_index.reduce(x) do |acc, (layer, i)|
          a = layer.call(acc)
          Maurograd::Utils.assert_finite!(a.data, where: "after layer #{i + 1}")
          a
        end
      end
    end
  end
end
