require "numo/narray"
require "numo/linalg"
require_relative "../backends"
require_relative "../ext"

module Maurograd
  module Backends
    # CPU backend: Numo::NArray storage, BLAS through Numo::Linalg and the
    # maurograd_ext C extension (convolution, pooling, ReLU, BatchNorm).
    #
    # This is the reference implementation of the backend interface: every
    # backend implements the same methods, with the same semantics, and the
    # cross-backend tests compare the others against this one.
    #
    # Native arrays are Numo arrays, normally Numo::SFloat (float32). Integer
    # arrays (e.g. class labels) are kept as they are.
    #
    # Note: inside Maurograd::Backends, `Numo` must be written `::Numo`, or
    # Ruby would look for Maurograd::Backends::Numo first.
    module CPU
      module_function

      def name
        :cpu
      end

      # --- Conversion ---------------------------------------------------------

      def native?(x)
        x.is_a?(::Numo::NArray)
      end

      # Converts host data (Numeric, Array, any Numo array) into a native
      # array. Numo arrays are kept as they are; everything else becomes
      # SFloat.
      def from(x)
        native?(x) ? x : ::Numo::SFloat.cast(x)
      end

      # Forces the computation of the given arrays. A no-op here: Numo
      # computes eagerly. (Lazy backends such as MLX need it.)
      def eval(_arrays)
        nil
      end

      # Starts computing the given arrays without waiting for the result.
      # Also a no-op here.
      def eval_async(_arrays)
        nil
      end

      # Copies a native array to host memory, as a Numo array.
      def to_numo(x)
        x
      end

      # The single value of a one-element array, as a Ruby Float.
      def item(x)
        raise ArgumentError, "item needs a one-element array, got shape #{x.shape.inspect}" unless x.size == 1

        x.to_a.flatten.first.to_f
      end

      # --- Creation and randomness --------------------------------------------

      def zeros(shape)
        ::Numo::SFloat.zeros(*shape)
      end

      def ones(shape)
        ::Numo::SFloat.ones(*shape)
      end

      # Float 0/1 array where each element is 1 with probability keep_prob.
      def bernoulli(shape, keep_prob)
        ::Numo::SFloat.cast(::Numo::SFloat.new(*shape).rand >= (1.0 - keep_prob))
      end

      # Seeds this backend's random number generator.
      def seed(n)
        ::Numo::NArray.srand(n)
      end

      # --- Shape --------------------------------------------------------------

      def reshape(x, *shape)
        x.reshape(*shape)
      end

      # Permutes axes (all of them must be given, like NumPy's transpose).
      def transpose(x, *axes)
        x.transpose(*axes)
      end

      # --- Reductions ---------------------------------------------------------
      #
      # axis: nil reduces over everything and returns a 0-dim array.

      def sum(x, axis: nil, keepdims: false)
        return ::Numo::SFloat.cast(x.sum) if axis.nil?

        x.sum(axis: axis, keepdims: keepdims)
      end

      def mean(x, axis: nil, keepdims: false)
        return ::Numo::SFloat.cast(x.mean) if axis.nil?

        x.mean(axis: axis, keepdims: keepdims)
      end

      def max(x, axis: nil, keepdims: false)
        return ::Numo::SFloat.cast(x.max) if axis.nil?

        x.max(axis: axis, keepdims: keepdims)
      end

      # --- Elementwise math ---------------------------------------------------

      def exp(x)
        ::Numo::NMath.exp(x)
      end

      def log(x)
        ::Numo::NMath.log(x)
      end

      def sqrt(x)
        ::Numo::NMath.sqrt(x)
      end

      # True iff every element is finite (used by Maurograd.debug checks).
      def all_finite?(x)
        flat = ::Numo::SFloat.cast(x).flatten
        !(flat.ne(flat).any? || flat.abs.gt(1e30).any?)
      end

      # --- Linear algebra -----------------------------------------------------

      # 2D matrix product [M, K] x [K, N] -> [M, N].
      def matmul(a, b)
        a.dot(b)
      end

      # --- Losses support -----------------------------------------------------

      # Float [N, C] one-hot encoding of integer class indices [N].
      def one_hot(indices, num_classes)
        idx = ::Numo::Int32.cast(indices)
        n = idx.size
        if n.positive? && (idx.min.negative? || idx.max >= num_classes)
          raise ArgumentError, "class index out of range for C=#{num_classes}"
        end

        out = ::Numo::SFloat.zeros(n, num_classes)
        out[(::Numo::Int32.new(n).seq * num_classes) + idx] = 1.0 unless n.zero?
        out
      end

      # --- Fused NN kernels ---------------------------------------------------
      #
      # Each *_forward returns the result plus a backend-specific `ctx` with
      # whatever its *_backward needs; Ops treat ctx as opaque. All image
      # tensors are NCHW.

      # x [N,C,H,W], w [Cout,C,KH,KW], b [Cout] or nil -> [y [N,Cout,OH,OW], ctx]
      def conv2d_forward(x, w, b, stride, padding)
        x = contiguous_sfloat(x)
        w = contiguous_sfloat(w)
        b &&= contiguous_sfloat(b)
        workspace = {}
        y = Maurograd::Ext.conv2d_forward(x, w, b, stride, padding, workspace)
        [y, {x: x, w: w, stride: stride, padding: padding, workspace: workspace}]
      end

      # -> [dx, dw, db] (nil for each gradient not requested)
      #
      # The want_* flags must be real booleans: the C side tests them with
      # Ruby truthiness (RTEST), where 0 counts as true.
      def conv2d_backward(ctx, dout, want_dx:, want_dw:, want_db:)
        Maurograd::Ext.conv2d_backward(
          ctx[:x], ctx[:w], contiguous_sfloat(dout),
          ctx[:stride], ctx[:padding],
          want_dx ? true : false, want_dw ? true : false, want_db ? true : false,
          ctx[:workspace]
        )
      end

      # Padding cells count as 0; ties go to the first maximum.
      # -> [y [N,C,OH,OW], ctx]
      def maxpool2d_forward(x, kh, kw, stride, padding)
        x = contiguous_sfloat(x)
        y, argmax = Maurograd::Ext.maxpool2d_forward(x, kh, kw, stride, padding)
        [y, {argmax: argmax, input_shape: x.shape}]
      end

      def maxpool2d_backward(ctx, dout)
        Maurograd::Ext.maxpool2d_backward(contiguous_sfloat(dout), ctx[:argmax], ctx[:input_shape])
      end

      # -> [y, ctx]
      def relu_forward(x)
        y = Maurograd::Ext.relu_forward(contiguous_sfloat(x))
        [y, y]
      end

      def relu_backward(ctx, dout)
        Maurograd::Ext.relu_backward(contiguous_sfloat(dout), ctx)
      end

      # Per-channel batch normalization. With mean/var nil (training) uses
      # the batch statistics, otherwise the given running ones.
      # -> [y, ctx, batch_mean [C], batch_var [C] (biased)]
      def batchnorm2d_forward(x, mean, var, gamma, beta, eps)
        x = contiguous_sfloat(x)
        mean &&= contiguous_sfloat(mean)
        var &&= contiguous_sfloat(var)
        gamma &&= contiguous_sfloat(gamma)
        beta &&= contiguous_sfloat(beta)
        y, x_hat, bmean, bvar, inv_std = Maurograd::Ext.batchnorm2d_forward(x, mean, var, gamma, beta, eps)
        [y, {x_hat: x_hat, inv_std: inv_std, gamma: gamma}, bmean, bvar]
      end

      # training: whether forward used batch statistics. -> [dx, dgamma, dbeta]
      def batchnorm2d_backward(ctx, dout, training:)
        Maurograd::Ext.batchnorm2d_backward(
          contiguous_sfloat(dout), ctx[:x_hat], ctx[:inv_std], ctx[:gamma], training
        )
      end

      # The C kernels read raw float32 memory, so they need contiguous
      # SFloat arrays; arrays reaching an op may be strided views (e.g. after
      # a transpose) or another dtype (e.g. a DFloat upstream gradient).
      def contiguous_sfloat(a)
        a = ::Numo::SFloat.cast(a) unless a.is_a?(::Numo::SFloat)
        a.contiguous? ? a : a.copy
      end
    end

    register(:cpu, CPU)
  end
end
