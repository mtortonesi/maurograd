module Maurograd
  module Ops
    class Sum
      def self.apply(tensor)
        new(tensor).forward
      end

      attr_reader :inputs

      def initialize(tensor)
        # Salviamo il tensore negli inputs per il backtracing
        @inputs = [tensor]
      end

      def forward
        tensor = @inputs.first

        # Il risultato della somma è un singolo valore scalare che conteniene
        # la somma di tutti gli elementi
        result_data = tensor.data.sum

        requires_grad = tensor.requires_grad
        result = Tensor.new(result_data, requires_grad: requires_grad)

        result.creator = self if requires_grad
        result
      end

      def backward(grad_output)
        tensor = @inputs.first
        return unless tensor.requires_grad


        # Se y = sum(x), allora dy/dx_i = 1 per ogni i.
        # Il gradiente totale è grad_output * 1, espanso sulla forma di tensor.
        #
        # grad_output è uno scalare. Usiamo Numo::SFloat.cast per assicurarci
        # che grad_output sia trattato correttamente e poi lo moltiplichiamo
        # per un array di 1 della stessa forma dell'input.
        ones = Numo::SFloat.ones(*tensor.data.shape)
        grad_input = grad_output * ones

        tensor.accumulate_grad(grad_input)
      end
    end
  end
end
