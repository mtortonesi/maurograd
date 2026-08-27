module Maurograd
  module Layers
    class Sequential
      attr_reader :layers

      def initialize(*layers)
        @layers = layers.flatten
      end

      def forward(x)
        layer_num = 1
        @layers.reduce(x) do |acc, layer|
          a = layer.forward(acc)
          Maurograd::Utils.assert_finite!(a.data, where: "after layer #{layer_num}")
          layer_num += 1
          a
        end
      end

      # Collect parameters (tensors with grad) from all layers that expose them.
      def parameters
        @layers.flat_map do |layer|
          layer.respond_to?(:parameters) ? layer.parameters : []
        end
      end

      def train
        @layers.each { |l| l.train if l.respond_to?(:train) }
        self
      end

      def eval
        @layers.each { |l| l.eval if l.respond_to?(:eval) }
        self
      end
    end
  end
end
