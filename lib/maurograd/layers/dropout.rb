require_relative "../ops/dropout"
require_relative "module"

module Maurograd
  module Layers
    # Dropout Layer (high-level API).
    #
    # Has no parameters; it only remembers the drop probability and whether
    # it is in training mode (dropout active) or eval mode (identity).
    # Delegates the actual work to Ops::Dropout.
    #
    # Like PyTorch, a freshly constructed layer is in training mode: call
    # #eval (usually on the whole model) before evaluating it.
    class Dropout < Module
      attr_reader :p

      def initialize(p = 0.5)
        Maurograd::Ops::Dropout.validate_p!(p)
        @p = p
      end

      def forward(input)
        Maurograd::Ops::Dropout.apply(input, p: @p, training: training?)
      end
    end
  end
end
