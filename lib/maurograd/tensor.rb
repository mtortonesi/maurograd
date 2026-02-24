require 'numo/narray'
require 'set'

require_relative 'ops/add'
require_relative 'ops/sub'
require_relative 'ops/mul'
require_relative 'ops/matmul'
require_relative 'ops/power'
require_relative 'ops/sum'
require_relative 'ops/conv2d'
require_relative 'ops/linear'
require_relative 'ops/flatten'
require_relative 'ops/relu'
require_relative 'ops/maxpool2d'


module Maurograd
  # Tensor is the core data structure of Maurograd.
  #
  # A Tensor wraps:
  # - `data`: the actual numeric values (Numo::NArray)
  # - `grad`: the accumulated gradient dL/d(data) (Numo::NArray, lazily allocated)
  # - `creator`: the Op (operation) that created this tensor in the computation graph
  #
  # This mirrors the design of modern autograd frameworks:
  # - Operations (Ops) build a directed acyclic graph (DAG) of tensors.
  # - Calling backward() on a final tensor (usually the loss) propagates gradients
  #   through the graph in reverse topological order.
  #
  class Tensor
    attr_accessor :data, :grad, :creator
    attr_reader :requires_grad, :device

    # Create a new Tensor.
    #
    # Params:
    # - data: numeric data (Ruby array, scalar, or Numo::NArray). Converted to Numo::SFloat.
    # - requires_grad: if true, this tensor participates in autograd (gradients are tracked)
    # - device: placeholder for future CPU/GPU support (currently only :cpu)
    #
    # Notes on dtype:
    # - We default to Numo::SFloat (float32). This is the common choice in neural networks:
    #   it is faster and uses less memory than float64, with sufficient precision in practice.
    #
    def initialize(data, requires_grad: false, device: :cpu)
      # Ensure @data is a Numo::NArray (float32).
      @data = data.is_a?(Numo::NArray) ? data : Numo::SFloat.cast(data)

      # If false, backward() becomes a no-op and gradients are not stored.
      @requires_grad = requires_grad

      # Pointer to the Op that produced this tensor.
      # This is how we traverse the computation graph during backprop.
      # Note: leaf tensors (parameters) have creator=nil
      @creator = nil

      # Gradient buffer (Numo::NArray), allocated lazily.
      # Many tensors in a network never need to store gradients (e.g., inference),
      # so allocating eagerly would waste memory.
      @grad = nil

      # Device support placeholder.
      # In the future this could dispatch to CUDA arrays or other backends.
      @device = device
    end

    # Convenience accessors.
    def shape; @data.shape; end
    def ndim;  @data.ndim;  end
    def size;  @data.size;  end

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
        raise "Gradient can be omitted only for scalar tensors" if @data.size > 1
        gradient = Numo::SFloat.ones(*@data.shape)
      end

      # Ensure gradients are raw Numo arrays (not Tensors).
      # (Your Ops generally pass Numo arrays already, but this keeps the API consistent.)
      gradient = gradient.data if gradient.is_a?(Tensor)

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
        t.creator.backward(t.grad) if t.creator
      end
    end

    # Accumulates gradient into this tensor.
    # This is called by Ops during the backward pass.
    def accumulate_grad(gradient)
      return unless @requires_grad

      # Ensure gradients are raw Numo arrays (not Tensors).
      gradient = gradient.data if gradient.is_a?(Tensor)

      @grad = @grad ? (@grad + gradient) : gradient
    end

    # Reset gradient buffer to zeros (common at the start of each training step).
    #
    # Note:
    # - We allocate a new zero array of the same shape as data.
    # - For memory/performance, some frameworks reuse buffers; here we keep it simple.
    def zero_grad
      @grad = Numo::SFloat.zeros(*@data.shape) if @requires_grad
    end

    # --- Basic Ops (syntactic sugar) -----------------------------------------
    #
    # These methods create new tensors by calling Ops, which:
    # - compute the forward result
    # - record graph links (creator + inputs) if gradients are required
    #
    # In Ruby we must explicitly wrap scalars/arrays into Tensor objects.

    def +(other)
      other = Tensor.new(other) unless other.is_a?(Tensor)
      Ops::Add.apply(self, other)
    end

    def matmul(other)
      other = Tensor.new(other) unless other.is_a?(Tensor)
      Ops::MatMul.apply(self, other)
    end

    def conv2d(other)
      other = Tensor.new(other) unless other.is_a?(Tensor)
      Ops::Conv2D.apply(self, other)
    end

    def **(exponent)
      Ops::Power.apply(self, exponent)
    end

    def sum
      Ops::Sum.apply(self)
    end

    def -(other)
      other = Tensor.new(other) unless other.is_a?(Tensor)
      Ops::Sub.apply(self, other)
    end

    def *(other)
      other = Tensor.new(other) unless other.is_a?(Tensor)
      Ops::Mul.apply(self, other)
    end

    def linear(weight, bias = nil)
      weight = Tensor.new(weight) unless weight.is_a?(Tensor)
      bias   = Tensor.new(bias)   if bias && !bias.is_a?(Tensor)
      Ops::Linear.apply(self, weight, bias)
    end

    def flatten
      Ops::Flatten.apply(self)
    end

    def relu
      Ops::ReLU.apply(self)
    end

    def maxpool2d(kernel_size, stride: nil, padding: 0)
      kh, kw = kernel_size.is_a?(Integer) ? [kernel_size, kernel_size] : kernel_size
      Ops::MaxPool2D.apply(self, kh, kw, stride, padding)
    end

    # Move tensor to a different device (future work).
    #
    # Today Maurograd runs only on CPU with Numo.
    # This method exists to keep the API forward-compatible with GPU support.
    def to(device)
      return self if @device == device
      @device = device
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
    def transpose(ax0 = -2, ax1 = -1)
      axes = (0...@data.ndim).to_a
      axes[ax0], axes[ax1] = axes[ax1], axes[ax0]
      Tensor.new(@data.transpose(*axes), requires_grad: @requires_grad)
    end

    # Permute tensor axes arbitrarily.
    #
    # This is useful to change data layout conventions, e.g.:
    # - NCHW -> NHWC: permute(0, 2, 3, 1)
    # - NHWC -> NCHW: permute(0, 3, 1, 2)
    #
    # Note:
    # - This returns a new Tensor that shares the underlying data view semantics of Numo.
    # - If you later need contiguous memory, you may need an explicit copy (future feature).
    def permute(*axes)
      Tensor.new(@data.transpose(*axes), requires_grad: @requires_grad)
    end
  end
end
