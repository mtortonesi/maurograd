require "numo/narray"

require_relative "backends"
require_relative "backends/cpu"

require_relative "ops/add"
require_relative "ops/sub"
require_relative "ops/mul"
require_relative "ops/matmul"
require_relative "ops/power"
require_relative "ops/sum"
require_relative "ops/conv2d"
require_relative "ops/linear"
require_relative "ops/flatten"
require_relative "ops/relu"
require_relative "ops/maxpool2d"
require_relative "ops/permute"
require_relative "ops/reshape"
require_relative "ops/elementwise"
require_relative "ops/slice"
require_relative "ops/concat"
require_relative "ops/log_softmax"

module Maurograd
  # Tensor is the core data structure of Maurograd.
  #
  # A Tensor wraps:
  # - `data`: the actual numeric values, as a native array of its backend
  #   (a Numo::NArray on the default CPU backend)
  # - `grad`: the accumulated gradient dL/d(data) (same array type, lazily allocated)
  # - `creator`: the Op (operation) that created this tensor in the computation graph
  # - `backend`: where `data` lives and which code runs the math (see
  #   Maurograd::Backends)
  #
  # This mirrors the design of modern autograd frameworks:
  # - Operations (Ops) build a directed acyclic graph (DAG) of tensors.
  # - Calling backward() on a final tensor (usually the loss) propagates gradients
  #   through the graph in reverse topological order.
  #
  class Tensor
    attr_accessor :data, :grad, :creator
    attr_reader :requires_grad, :backend

    # Create a new Tensor.
    #
    # Params:
    # - data: numeric data (Ruby array, scalar, Numo::NArray, or a native
    #   array of any backend). Converted to a native array of `backend`.
    # - requires_grad: if true, this tensor participates in autograd (gradients are tracked)
    # - backend: backend name (e.g. :cpu, :mlx) or object; defaults to
    #   Maurograd.backend
    #
    # Notes on dtype:
    # - We default to Numo::SFloat (float32). This is the common choice in neural networks:
    #   it is faster and uses less memory than float64, with sufficient precision in practice.
    #
    def initialize(data, requires_grad: false, backend: nil)
      @backend = backend ? Backends.fetch(backend) : Maurograd.backend

      # Ensure @data is a native array of our backend (float32 by default).
      @data = Tensor.to_native(data, @backend)

      # If false, backward() becomes a no-op and gradients are not stored.
      @requires_grad = requires_grad

      # Pointer to the Op that produced this tensor.
      # This is how we traverse the computation graph during backprop.
      # Note: leaf tensors (parameters) have creator=nil
      @creator = nil

      # Gradient buffer (native array), allocated lazily.
      # Many tensors in a network never need to store gradients (e.g., inference),
      # so allocating eagerly would waste memory.
      @grad = nil
    end

    # Converts `x` (host data, a Tensor, or a native array of any backend)
    # into a native array of `backend`, copying only when needed.
    def self.to_native(x, backend)
      x = x.data if x.is_a?(Tensor)
      return x if backend.native?(x)

      # A native array of another backend: go through host memory.
      x = x.to_numo if x.respond_to?(:to_numo) && !x.is_a?(Numo::NArray)
      backend.from(x)
    end

    # Name of the backend (kept for compatibility with the old `device`
    # placeholder).
    def device
      @backend.name
    end

    # Copy of the data in host memory, as a Numo array.
    def to_numo
      @backend.to_numo(@data)
    end

    # The value of a one-element tensor (e.g. a loss), as a Ruby Float.
    def item
      @backend.item(@data)
    end

    # Convenience accessors.
    def shape
      @data.shape
    end

    def ndim
      shape.length
    end

    def size
      shape.reduce(1, :*)
    end

    # Start backpropagation from this tensor.
    #
    # Autograd contract:
    # ------------------
    # backward() accumulates gradients into `.grad` and then propagates them to
    # upstream tensors using the `.creator` links (the computation graph).
    #
    # Why we need a topological traversal (not simple recursion):
    # ----------------------------------------------------------
    # Modern neural networks often have DAG-shaped graphs, not chains:
    # - skip connections (ResNet)
    # - multi-branch modules (Inception)
    # - U-Nets with concatenations
    #
    # In a DAG, a tensor may contribute to the output through multiple paths.
    # Its gradient must be the SUM of contributions from all downstream branches.
    #
    # If we do naive recursion, we risk:
    # - calling backward on a shared parent multiple times before all child gradients
    #   have been accumulated
    # - redundant work / exponential blow-ups
    # - deep recursion / stack overflows
    #
    # The robust solution is:
    # 1) Build a topological ordering of tensors reachable from `self`.
    # 2) Traverse that ordering in reverse to propagate gradients exactly once per node.
    #
    # Params:
    # - gradient: optional initial gradient dL/d(self).
    #   If omitted, this tensor must be a scalar (loss), and gradient defaults to 1.
    #
    def backward(gradient = nil)
      # If this tensor does not require gradients, we stop immediately.
      return unless @requires_grad

      # 1) Initialize the starting gradient.
      #
      # For a scalar loss L, dL/dL = 1, so backward() can be called without arguments.
      # For non-scalars, the caller must explicitly provide a gradient of matching shape.
      if gradient.nil?
        raise "Gradient can be omitted only for scalar tensors" if size > 1
        gradient = @backend.ones(shape)
      end

      # 2) Accumulate gradient into this tensor.
      accumulate_grad(gradient)

      # 3) Build a topological ordering of the computation graph.
      #
      # We collect tensors that:
      # - are Tensors
      # - have a creator Op (i.e., are not leaf tensors)
      #
      # `visited` prevents revisiting shared nodes in DAGs.
      topo = []
      visited = Set.new

      build_topo = lambda do |t|
        # Only traverse tensors that were created by an Op.
        return unless t.is_a?(Tensor) && t.creator
        return if visited.include?(t)

        visited.add(t)

        # Recursively visit the inputs of the creator Op.
        # This walks the graph upstream.
        t.creator.inputs.each { |inp| build_topo.call(inp) }

        # Post-order push: parents first, then the node.
        topo << t
      end

      build_topo.call(self)

      # 4) Traverse in reverse topological order to run backward passes.
      #
      # Each tensor calls backward on its creator Op exactly once,
      # using the gradient accumulated in tensor.grad.
      topo.reverse_each do |t|
        t.creator&.backward(t.grad)
      end
    end

    # Accumulates gradient into this tensor.
    # This is called by Ops during the backward pass.
    def accumulate_grad(gradient)
      return unless @requires_grad

      # Ensure gradients are native arrays of our backend (not Tensors, not
      # host arrays passed in by a caller).
      gradient = Tensor.to_native(gradient, @backend)

      @grad = @grad ? (@grad + gradient) : gradient
    end

    # Reset gradient buffer to zeros (common at the start of each training step).
    #
    # Note:
    # - We allocate a new zero array of the same shape as data.
    # - For memory/performance, some frameworks reuse buffers; here we keep it simple.
    def zero_grad
      @grad = @backend.zeros(shape) if @requires_grad
    end

    # --- Basic Ops (syntactic sugar) -----------------------------------------
    #
    # These methods create new tensors by calling Ops, which:
    # - compute the forward result
    # - record graph links (creator + inputs) if gradients are required
    #
    # In Ruby we must explicitly wrap scalars/arrays into Tensor objects.

    def +(other)
      other = Tensor.new(other, backend: @backend) unless other.is_a?(Tensor)
      Ops::Add.apply(self, other)
    end

    def matmul(other)
      other = Tensor.new(other, backend: @backend) unless other.is_a?(Tensor)
      Ops::MatMul.apply(self, other)
    end

    def conv2d(other)
      other = Tensor.new(other, backend: @backend) unless other.is_a?(Tensor)
      Ops::Conv2D.apply(self, other)
    end

    def **(other)
      Ops::Power.apply(self, other)
    end

    # Sum of all elements (a scalar), or over the given axes:
    # x.sum(axis: 1), x.sum(axis: [2, 3], keepdims: true). See Ops::Sum.
    def sum(axis: nil, keepdims: false)
      Ops::Sum.apply(self, axis: axis, keepdims: keepdims)
    end

    # Mean of all elements, or over the given axes (same arguments as
    # #sum): the sum times 1/count, so its gradient is the upstream
    # gradient spread evenly over the elements that were averaged.
    def mean(axis: nil, keepdims: false)
      axes = Ops::Sum.normalize_axes(axis, ndim)
      count = axes ? axes.map { |a| shape[a] }.reduce(1, :*) : size
      sum(axis: axis, keepdims: keepdims) * (1.0 / count)
    end

    def -(other)
      other = Tensor.new(other, backend: @backend) unless other.is_a?(Tensor)
      Ops::Sub.apply(self, other)
    end

    def *(other)
      other = Tensor.new(other, backend: @backend) unless other.is_a?(Tensor)
      Ops::Mul.apply(self, other)
    end

    # x / y: by a number it's x * (1/y); by a Tensor, x * y**-1, so the
    # gradient comes from Mul and Power with no op of its own.
    def /(other)
      return self * (1.0 / other) if other.is_a?(Numeric)

      other = Tensor.new(other, backend: @backend) unless other.is_a?(Tensor)
      self * (other**-1)
    end

    # -x, as x * -1.
    def -@
      self * -1.0
    end

    # Elementwise functions (Ops::Elementwise and its subclasses).
    def exp = Ops::Exp.apply(self)

    def log = Ops::Log.apply(self)

    def sigmoid = Ops::Sigmoid.apply(self)

    def tanh = Ops::Tanh.apply(self)

    def silu = Ops::SiLU.apply(self)

    # log(1 + e^x), computed without overflow.
    def softplus = Ops::Softplus.apply(self)

    # log(sigmoid(x)), computed without underflow.
    def log_sigmoid = Ops::LogSigmoid.apply(self)

    def leaky_relu(negative_slope: 0.01) = Ops::LeakyReLU.apply(self, negative_slope: negative_slope)

    # log(sum(exp(x))) over all elements or the given axes (same arguments
    # as #sum), computed without overflow; see Ops::LogSumExp.
    def logsumexp(axis: nil, keepdims: false)
      Ops::LogSumExp.apply(self, axis: axis, keepdims: keepdims)
    end

    # Log-probabilities from logits along `axis`: x - logsumexp(x).
    def log_softmax(axis: -1)
      Ops::LogSoftmax.apply(self, axis: axis)
    end

    # Probabilities from logits along `axis`, summing to 1 there:
    # exp(log_softmax(x)), so its gradient comes from those two ops.
    def softmax(axis: -1)
      log_softmax(axis: axis).exp
    end

    # Clips to [min, max] (either may be nil).
    def clamp(min: nil, max: nil) = Ops::Clamp.apply(self, min: min, max: max)

    # The same data, cut off from the graph: a new leaf tensor that doesn't
    # require grad, so nothing is backpropagated through it. Use it to treat
    # a value as a constant, e.g. a GAN generator's samples when training
    # the discriminator. The data is shared, not copied (ops never modify
    # their inputs, so that's safe).
    def detach
      Tensor.new(@data, backend: @backend)
    end

    def linear(weight, bias = nil)
      weight = Tensor.new(weight, backend: @backend) unless weight.is_a?(Tensor)
      bias = Tensor.new(bias, backend: @backend) if bias && !bias.is_a?(Tensor)
      Ops::Linear.apply(self, weight, bias)
    end

    def flatten
      Ops::Flatten.apply(self)
    end

    # Same elements in the same (row-major) order, in a new shape:
    # reshape(6, 4), reshape([6, 4]), or reshape(2, -1) to infer one
    # dimension. The element count must not change.
    #
    # An autograd op (Ops::Reshape). It regroups elements and never moves
    # them, so it is not a substitute for permute/transpose: [2, 3] ->
    # [3, 2] by reshape is not the transposed matrix.
    def reshape(*shape)
      shape = shape.first if shape.length == 1 && shape.first.is_a?(Array)
      Ops::Reshape.apply(self, shape)
    end

    # The elements in `range` along `axis` (Ops::Slice): x.slice(1, 0...4)
    # is x[:, 0:4] in NumPy. Any Ruby Range works, including negative and
    # endless ones: x.slice(-1, -2..) is the last two along the last axis.
    def slice(axis, range)
      Ops::Slice.apply(self, axis, range)
    end

    # Splits the tensor into `n` parts along `axis`, as consecutive slices
    # of equal size (the last one smaller if the size isn't a multiple of
    # n, so there may be fewer than n parts, as in PyTorch):
    #
    #   mean, log_var = encoder.call(x).chunk(2, axis: 1)
    def chunk(n, axis: 0)
      raise ArgumentError, "chunk needs a positive number of parts, got #{n.inspect}" unless n.is_a?(Integer) && n.positive?

      axis = Ops::Sum.normalize_axes(axis, ndim).first
      step = (shape[axis] + n - 1) / n
      (0...shape[axis]).step(step).map { |start| slice(axis, start...(start + step)) }
    end

    def relu
      Ops::ReLU.apply(self)
    end

    def maxpool2d(kernel_size, stride: nil, padding: 0)
      kh, kw = kernel_size.is_a?(Integer) ? [kernel_size, kernel_size] : kernel_size
      Ops::MaxPool2D.apply(self, kh, kw, stride, padding)
    end

    # Moves this tensor (data and gradient) to another backend, in place,
    # and returns self. In place, so that everything holding this Tensor
    # (layers, optimizers) keeps working with it. Meant for leaf tensors
    # such as parameters, not for tensors in the middle of a graph.
    def to(backend)
      target = Backends.fetch(backend)
      return self if target.equal?(@backend)

      @data = target.from(@backend.to_numo(@data))
      @grad = target.from(@backend.to_numo(@grad)) if @grad
      @backend = target
      self
    end

    # Swap two axes of a tensor.
    #
    # This is the typical "transpose" used in neural nets:
    # - For matrices (2D): swap rows and columns.
    # - For NCHW images (4D): swap H and W (or other chosen axes) without permuting everything.
    #
    # Why not call Numo's transpose() directly with no args?
    # -----------------------------------------------------
    # Numo::NArray#transpose with no arguments reverses ALL axes.
    # For a 4D tensor [N, C, H, W], that would become [W, H, C, N],
    # which is almost never what we want in CNNs.
    #
    # Therefore this method swaps only two axes (default: the last two).
    # It is an autograd op (Ops::Permute): gradients flow back through it.
    def transpose(ax0 = -2, ax1 = -1)
      axes = (0...ndim).to_a
      axes[ax0], axes[ax1] = axes[ax1], axes[ax0]
      Ops::Permute.apply(self, axes)
    end

    # Permute tensor axes arbitrarily.
    #
    # This is useful to change data layout conventions, e.g.:
    # - NCHW -> NHWC: permute(0, 2, 3, 1)
    # - NHWC -> NCHW: permute(0, 3, 1, 2)
    #
    # Notes:
    # - It is an autograd op (Ops::Permute): gradients flow back through it.
    # - Negative axes count from the end; the axes must be a permutation of
    #   0...ndim, or it raises ArgumentError.
    # - On the CPU backend the result may be a strided Numo view; the kernels
    #   that need contiguous memory copy it themselves.
    def permute(*axes)
      Ops::Permute.apply(self, axes)
    end
  end
end
