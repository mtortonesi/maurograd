require_relative "../ops/maxpool2d"

module Maurograd
  module Layers
    class MaxPool2D
      def initialize(kernel_size, stride: nil, padding: 0)
        @kh, @kw = kernel_size.is_a?(Integer) ? [kernel_size, kernel_size] : kernel_size
        @stride = stride || @kh
        @padding = padding
      end

      def forward(input)
        Maurograd::Ops::MaxPool2D.apply(input, @kh, @kw, @stride, @padding)
      end
    end
  end
end
