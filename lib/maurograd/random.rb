module Maurograd
  # Random tensors, drawn by the backend that will hold them.
  #
  #   eps = Maurograd.randn_like(mean)      # reparameterization: mean + std * eps
  #   noise = Maurograd.randn(64, 1, 28, 28) # a batch of diffusion noise
  #   t = Maurograd.rand(64)                 # uniform times in [0, 1)
  #
  # Drawing on the backend matters on the GPU: generative models draw new
  # noise at every training and sampling step, and noise drawn on the host
  # would have to be copied to the GPU each time.
  #
  # Maurograd.seed! seeds every loaded backend, so runs are reproducible on
  # each of them, but the numbers differ between backends (each has its own
  # generator). Weight initialization doesn't use these: layers draw their
  # initial weights on the host, so a seed gives the same model everywhere.
  class << self
    # Samples of the standard normal distribution N(0, 1), as a float32
    # tensor of the given shape: randn(2, 3) or randn([2, 3]).
    def randn(*shape, backend: nil, requires_grad: false)
      random_tensor(:normal, shape, backend, requires_grad)
    end

    # Samples of the uniform distribution on [0, 1).
    def rand(*shape, backend: nil, requires_grad: false)
      random_tensor(:uniform, shape, backend, requires_grad)
    end

    # randn with the shape and backend of `tensor`.
    def randn_like(tensor)
      randn(tensor.shape, backend: tensor.backend)
    end

    # rand with the shape and backend of `tensor`.
    def rand_like(tensor)
      rand(tensor.shape, backend: tensor.backend)
    end

    private

    def random_tensor(kind, shape, backend, requires_grad)
      shape = shape.first if shape.length == 1 && shape.first.is_a?(Array)
      unless shape.all? { |d| d.is_a?(Integer) && d >= 0 }
        raise ArgumentError, "dimensions must be non-negative Integers, got #{shape.inspect}"
      end

      backend = backend ? Backends.fetch(backend) : self.backend
      Tensor.new(backend.public_send(kind, shape), requires_grad: requires_grad, backend: backend)
    end
  end
end
