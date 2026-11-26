module Maurograd
  module Optim
    # Base class for optimizers.
    #
    # Holds the list of parameters to optimize and implements what every
    # optimizer shares (zero_grad). Subclasses implement #step.
    #
    # Optimizers that need per-parameter state (momentum buffers, Adam's moment
    # estimates, ...) keep it in plain arrays aligned with @params: state for
    # @params[i] lives at index i. That keeps Tensor itself free of any
    # optimizer-specific fields.
    class Optimizer
      attr_reader :params

      def initialize(params)
        @params = params
      end

      # Reset gradients of all parameters to zero.
      #
      # This mirrors PyTorch's optimizer.zero_grad().
      def zero_grad
        @params.each(&:zero_grad)
      end

      def step
        raise NotImplementedError, "#{self.class} must implement #step"
      end

      private

      # Lazy backends (MLX) only record the update; computing the new
      # parameter values right away keeps each step's graph from being
      # chained onto the next one. A no-op on the CPU backend.
      def eval_params!
        @params.group_by(&:backend).each { |backend, ps| backend.eval(ps.map(&:data)) }
      end

      # Yields [param, grad, index] for each parameter that should be updated:
      # it requires grad and already has one (grad=nil means it never took
      # part in a backward pass yet, so there's nothing to apply).
      def each_updatable_param
        @params.each_with_index do |p, i|
          next unless p.requires_grad
          next if p.grad.nil?

          yield p, p.grad, i
        end
      end
    end
  end
end
