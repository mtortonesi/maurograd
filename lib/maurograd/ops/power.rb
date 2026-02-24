module Maurograd
  module Ops
    class Power
      def self.apply(tensor, exponent)
        new(tensor, exponent).forward
      end

      attr_reader :inputs

      def initialize(tensor, exponent)
        # Verifichiamo che l'esponente sia un numero (Integer, Float, etc.)
        unless exponent.is_a?(Numeric)
          raise ArgumentError, "L'esponente deve essere un valore numerico scalare, non un #{exponent.class}"
        end
        # Salviamo il tensore negli inputs per il backtracing
        @inputs = [tensor]
        # Salviamo l'esponente come parametro statico
        @exponent = exponent
      end

      def forward
        tensor = @inputs.first
        result_data = tensor.data ** @exponent

        requires_grad = tensor.requires_grad
        result = Tensor.new(result_data, requires_grad: requires_grad)

        result.creator = self if requires_grad
        result
      end

      def backward(grad_output)
        tensor = @inputs.first
        return unless tensor.requires_grad

        # Calcolo locale della derivata: n * x^(n-1)
        local_derivative = @exponent * (tensor.data ** (@exponent - 1))

        # Accumulo del gradiente (Chain Rule)
        grad_input = grad_output * local_derivative

        tensor.accumulate_grad(grad_input)
      end
    end
  end
end
