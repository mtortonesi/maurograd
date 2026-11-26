require_relative "../ops/dropout"

module Maurograd
  module Layers
    # Dropout Layer (high-level API).
    #
    # Has no parameters; it only remembers the drop probability and whether
    # it is in training mode (dropout active) or eval mode (identity).
    # Delegates the actual work to Ops::Dropout.
    #
    # Like PyTorch, a freshly constructed layer is in training mode: call
    # #eval (usually via Sequential#eval) before evaluating a model.
    class Dropout
      attr_reader :p

      def initialize(p = 0.5)
        Maurograd::Ops::Dropout.validate_p!(p)
        @p = p
        @training = true
      end

      def forward(input)
        Maurograd::Ops::Dropout.apply(input, p: @p, training: @training)
      end

      def train
        @training = true
        self
      end

      def eval
        @training = false
        self
      end

      def training?
        @training
      end
    end
  end
end
