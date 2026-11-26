module Maurograd
  # Compute backends: where tensor data lives and which code runs the math.
  #
  # Maurograd's autograd (Tensor, Ops, the backward graph walk) is the same
  # for every backend. What a backend provides is the array type stored in
  # Tensor#data / Tensor#grad (its "native" arrays) and the operations the
  # Ops need on them. Backends::CPU is the reference implementation (Numo +
  # the C extension); see it for the full interface, documented method by
  # method.
  #
  # The contract has two parts:
  #
  # 1. Native arrays support Ruby arithmetic directly: + - * / ** and unary
  #    minus, between two arrays (with NumPy-style broadcasting) and between
  #    an array and a Numeric on either side. Op formulas use these
  #    operators as-is, so they read like the math.
  #
  # 2. Everything else (creation, reductions, reshaping, math functions,
  #    the fused NN kernels, conversion to/from host memory) is a method on
  #    the backend object: `tensor.backend.sum(x, axis: 1)`.
  #
  # Usage:
  #
  #   Maurograd.backend = :mlx          # default for tensors created from now on
  #   model.to(:mlx)                    # move an existing model's parameters
  #   t = Maurograd::Tensor.new(data, backend: :cpu)
  #
  module Backends
    @registry = {}
    @loaders = {}

    class << self
      # Registers a backend object under `name`.
      def register(name, backend)
        @registry[name.to_sym] = backend
      end

      # Registers a lazy loader for an optional backend: `block` runs the
      # first time the backend is requested, and must register it.
      def register_loader(name, &block)
        @loaders[name.to_sym] = block
      end

      # Looks a backend up by name (Symbol/String), or returns it unchanged
      # if it already is a backend object.
      def fetch(name_or_backend)
        return name_or_backend unless name_or_backend.is_a?(Symbol) || name_or_backend.is_a?(String)

        name = name_or_backend.to_sym
        unless @registry.key?(name)
          loader = @loaders[name]
          raise ArgumentError, "unknown backend #{name.inspect} (available: #{available.inspect})" unless loader

          loader.call
          raise LoadError, "loading backend #{name.inspect} did not register it" unless @registry.key?(name)
        end
        @registry[name]
      end

      # Backend objects loaded so far.
      def loaded
        @registry.values
      end

      # Names of the backends that are registered or can be loaded.
      def available
        (@registry.keys + @loaders.keys).uniq
      end
    end
  end

  module Ops
    # The backend shared by all the given tensors (nils ignored). Ops run on
    # their inputs' backend, so mixing backends in one op is an error rather
    # than a silent copy: move tensors explicitly with Tensor#to.
    def self.backend_of(*tensors)
      tensors = tensors.compact
      backend = tensors.first.backend
      tensors.each do |t|
        next if t.backend.equal?(backend)

        raise ArgumentError, "tensors on different backends (#{backend.name} and #{t.backend.name}) " \
          "in one operation; move them with Tensor#to first"
      end
      backend
    end
  end

  class << self
    # The backend used for tensors created without an explicit backend:.
    def backend
      @backend ||= Backends.fetch(:cpu)
    end

    def backend=(name_or_backend)
      @backend = Backends.fetch(name_or_backend)
    end
  end
end
