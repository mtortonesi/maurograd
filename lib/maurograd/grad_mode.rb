module Maurograd
  # Gradient recording, on by default.
  #
  # Every op normally records how it computed its output (the output's
  # creator), so that backward can walk the graph later. That costs memory
  # (each op keeps its inputs and whatever it saved for backward) and some
  # time, and is wasted when no backward will follow: evaluating a model,
  # sampling from a generative model, updating weights by hand.
  #
  #   Maurograd.no_grad do
  #     logits = model.call(x_test)   # no graph is built
  #   end
  #
  # Inside the block, op outputs don't require grad and have no creator,
  # whatever their inputs. Tensors created directly keep the requires_grad
  # they were given, and parameters stay trainable: only recording stops.
  #
  # no_grad does NOT switch layers to evaluation mode (Dropout, BatchNorm2D
  # running statistics): that is model.eval, a separate setting, as in
  # PyTorch. Evaluation usually wants both.
  #
  # The setting is per thread, and blocks can be nested.
  GRAD_ENABLED_KEY = :maurograd_grad_enabled

  class << self
    # Runs the block with gradient recording off, and returns its value.
    def no_grad
      raise ArgumentError, "Maurograd.no_grad needs a block" unless block_given?

      previous = grad_enabled?
      Thread.current.thread_variable_set(GRAD_ENABLED_KEY, false)
      begin
        yield
      ensure
        Thread.current.thread_variable_set(GRAD_ENABLED_KEY, previous)
      end
    end

    # True unless inside a no_grad block (on this thread).
    def grad_enabled?
      Thread.current.thread_variable_get(GRAD_ENABLED_KEY) != false
    end
  end
end
