require "numo/narray"
require_relative "../backends"

# MLX runs float32 matrix products in TF32 on GPUs that support it (e.g.
# Apple M5): faster, but only ~10 mantissa bits, i.e. ~1e-3 relative error,
# where the CPU backend gives ~1e-7. Like PyTorch, we default to full float32
# precision; set MLX_ENABLE_TF32=1 in the environment to trade it for speed.
# (MLX reads the variable once, on first use.)
ENV["MLX_ENABLE_TF32"] ||= "0"

# Load the optional maurograd_mlx extension (built only where mlx-c is
# available), the same way ext.rb loads maurograd_ext.
begin
  require "maurograd_mlx"
rescue LoadError
  pattern = File.join(File.expand_path("../../..", __dir__), "ext", "maurograd_mlx", "maurograd_mlx.{bundle,so}")
  candidate = Dir.glob(pattern).first
  raise LoadError, <<~MSG unless candidate
    the MLX backend needs the maurograd_mlx extension, which was not found
    (looked for #{pattern}). It needs mlx-c (`brew install mlx-c`); build
    it with `./bin/build_ext`.
  MSG

  require candidate
end

module Maurograd
  module Backends
    # MLX backend: arrays live in MLX (Apple's array framework), and run on
    # the Apple Silicon GPU through Metal (or on MLX's CPU device, see
    # Maurograd::MLX.device=).
    #
    # Native arrays are Maurograd::MLX::Array objects. Same interface and
    # semantics as Backends::CPU (the reference implementation, see there for
    # what each method does), except where noted.
    #
    # How it differs from the CPU backend:
    #
    # - MLX is lazy: every operation records a graph node, and computation
    #   only happens when values are needed (item/to_numo) or explicitly
    #   evaluated (#eval, which the optimizers call after each step so the
    #   graph never grows past one training step).
    #
    # - Kernels are written here, in Ruby, on top of MLX primitives. The
    #   backward passes of conv2d and maxpool2d come from MLX's own
    #   reverse-mode differentiation (MLX.vjp) of the forward function: the
    #   MLX counterpart of the cuDNN backward routines a CUDA backend would
    #   call. Maurograd's autograd still drives the graph as a whole.
    #
    # - MLX convolutions are channels-last (NHWC), while Maurograd tensors
    #   are NCHW: conv2d transposes at its boundaries.
    #
    # - Random numbers come from MLX's generator, so Dropout masks differ
    #   from the CPU backend's even with the same seed (weight init does
    #   not: layers initialize weights on the host, with Numo).
    module MLX
      Lib = Maurograd::MLX

      module_function

      def name
        :mlx
      end

      # --- Conversion ---------------------------------------------------------

      def native?(x)
        x.is_a?(Lib::Array)
      end

      def from(x)
        return x if native?(x)

        x = ::Numo::SFloat.cast(x) unless x.is_a?(::Numo::NArray)
        x = integer_type?(x) ? ::Numo::Int32.cast(x) : ::Numo::SFloat.cast(x) unless x.is_a?(::Numo::SFloat) || x.is_a?(::Numo::Int32)
        Lib.from_numo(x.contiguous? ? x : x.copy)
      end

      def to_numo(x)
        Lib.to_numo(x)
      end

      def item(x)
        raise ArgumentError, "item needs a one-element array, got shape #{x.shape.inspect}" unless x.size == 1

        to_numo(x).to_a.flatten.first.to_f
      end

      # Forces the computation of the given arrays (nils are ignored).
      def eval(arrays)
        Lib.eval(arrays)
      end

      # Schedules the given arrays for computation, without waiting.
      def eval_async(arrays)
        Lib.async_eval(arrays)
      end

      # --- Creation and randomness --------------------------------------------

      def zeros(shape)
        Lib.zeros(shape)
      end

      def ones(shape)
        Lib.ones(shape)
      end

      def bernoulli(shape, keep_prob)
        Lib.bernoulli(shape, keep_prob)
      end

      def seed(n)
        Lib.seed(n)
      end

      # --- Shape --------------------------------------------------------------

      def reshape(x, *shape)
        Lib.reshape(x, shape)
      end

      def transpose(x, *axes)
        Lib.transpose(x, axes)
      end

      # --- Reductions ---------------------------------------------------------

      def sum(x, axis: nil, keepdims: false)
        Lib.sum(x, axis, keepdims)
      end

      def mean(x, axis: nil, keepdims: false)
        Lib.mean(x, axis, keepdims)
      end

      def max(x, axis: nil, keepdims: false)
        Lib.max(x, axis, keepdims)
      end

      # --- Elementwise math ---------------------------------------------------

      def exp(x)
        Lib.exp(x)
      end

      def log(x)
        Lib.log(x)
      end

      def sqrt(x)
        Lib.sqrt(x)
      end

      def all_finite?(x)
        CPU.all_finite?(to_numo(x))
      end

      # --- Linear algebra -----------------------------------------------------

      def matmul(a, b)
        Lib.matmul(a, b)
      end

      # --- Losses support -----------------------------------------------------

      def one_hot(indices, num_classes)
        # Range check on the host: MLX can't raise from inside a kernel.
        idx = to_numo(indices)
        if idx.size.positive? && (idx.min.negative? || idx.max >= num_classes)
          raise ArgumentError, "class index out of range for C=#{num_classes}"
        end

        Lib.one_hot(indices, num_classes)
      end

      # --- Fused NN kernels ---------------------------------------------------

      def conv2d_forward(x, w, b, stride, padding)
        y = conv2d_nchw(x, w, stride, padding)
        y += reshape(b, 1, b.shape[0], 1, 1) if b
        [y, {x: x, w: w, stride: stride, padding: padding}]
      end

      def conv2d_backward(ctx, dout, want_dx:, want_dw:, want_db:)
        dx = dw = db = nil
        if want_dx || want_dw
          dx, dw = Lib.vjp([ctx[:x], ctx[:w]], [dout]) do |x, w|
            [conv2d_nchw(x, w, ctx[:stride], ctx[:padding])]
          end
        end
        db = sum(dout, axis: [0, 2, 3]) if want_db
        [want_dx ? dx : nil, want_dw ? dw : nil, db]
      end

      # Same semantics as the CPU backend: zero padding, and on ties the
      # gradient goes to the first maximum of the window only. (MLX's own
      # vjp of max would split a tied window's gradient equally among the
      # tied elements, and MNIST's flat backgrounds produce plenty of
      # positive ties.) In both paths below, a first-maximum mask (see
      # first_max_mask) routes dout to the winning element.
      def maxpool2d_forward(x, kh, kw, stride, padding)
        n, c, h, w = x.shape
        if tiling_pool?(h, w, kh, kw, stride, padding)
          # Non-overlapping windows that tile the input exactly (the common
          # case, e.g. 2x2 with stride 2): the windows are just a reshape +
          # transpose of x, [N,C,OH,KH,OW,KW] -> [N,C,OH,OW,KH*KW].
          oh = h / kh
          ow = w / kw
          tiles = reshape(transpose(reshape(x, n, c, oh, kh, ow, kw), 0, 1, 2, 4, 3, 5), n, c, oh, ow, kh * kw)
          y = Lib.max(tiles, [4], true)
          return [reshape(y, n, c, oh, ow), {tiles: tiles, max: y, shape: [n, c, h, w], kh: kh, kw: kw}]
        end

        windows = pool_windows(x, kh, kw, stride, padding)
        y = Lib.max(windows, [0], true)
        [reshape(y, *y.shape[1..]), {x: x, windows: windows, max: y, kh: kh, kw: kw, stride: stride, padding: padding}]
      end

      def maxpool2d_backward(ctx, dout)
        k = ctx[:kh] * ctx[:kw]

        if ctx[:tiles]
          # Mask the tiles, then undo the forward reshape + transpose: no
          # scatter needed, since every input element is in exactly one tile.
          n, c, h, w = ctx[:shape]
          oh = h / ctx[:kh]
          ow = w / ctx[:kw]
          mask = first_max_mask(ctx[:tiles], ctx[:max], reshape(Lib.arange(k), k), 4)
          dtiles = mask * reshape(dout, n, c, oh, ow, 1)
          untiled = transpose(reshape(dtiles, n, c, oh, ow, ctx[:kh], ctx[:kw]), 0, 1, 2, 4, 3, 5)
          return reshape(untiled, n, c, h, w)
        end

        # General case (overlapping windows, padding): windows [K,N,C,OH,OW]
        # come from strided slices, so the vjp of that (linear) extraction
        # scatters the masked gradient back into the input. Correct, but MLX
        # builds a full-size update per window offset, so it is much slower.
        mask = first_max_mask(ctx[:windows], ctx[:max], reshape(Lib.arange(k), k, 1, 1, 1, 1), 0)
        dx, = Lib.vjp([ctx[:x]], [mask * dout]) do |x|
          [pool_windows(x, ctx[:kh], ctx[:kw], ctx[:stride], ctx[:padding])]
        end
        dx
      end

      # Float mask with a 1 at the FIRST maximum of each window, given the
      # windows, their max along `axis` (keepdims) and the window-offset
      # index k along that axis (shaped to broadcast). Every maximum gets weight K - k,
      # so the first one has the largest, and comparing the weights with
      # their max keeps exactly that one. This avoids MLX's argmax, which
      # was ~100x slower than max on these shapes (39 ms vs 0.4 ms for a
      # [512, 6, 14, 14, 4] tensor on an M5).
      def first_max_mask(windows, max, offsets, axis)
        k = offsets.size
        weights = Lib.equal(windows, max) * (k - offsets)
        Lib.equal(weights, Lib.max(weights, [axis], true))
      end

      def tiling_pool?(h, w, kh, kw, stride, padding)
        padding.zero? && stride == kh && stride == kw && (h % kh).zero? && (w % kw).zero?
      end

      def relu_forward(x)
        y = Lib.maximum(x, 0.0)
        [y, y]
      end

      def relu_backward(ctx, dout)
        dout * Lib.greater(ctx, 0.0)
      end

      def batchnorm2d_forward(x, mean, var, gamma, beta, eps)
        c = x.shape[1]
        axes = [0, 2, 3]
        if mean.nil?
          mu = Lib.mean(x, axes, true)
          var4 = Lib.mean((x - mu)**2, axes, true)
        else
          mu = reshape(mean, 1, c, 1, 1)
          var4 = reshape(var, 1, c, 1, 1)
        end

        inv_std = 1.0 / sqrt(var4 + eps)
        x_hat = (x - mu) * inv_std
        g = gamma ? reshape(gamma, 1, c, 1, 1) : nil
        y = g ? x_hat * g : x_hat
        y += reshape(beta, 1, c, 1, 1) if beta

        [y, {x_hat: x_hat, inv_std: inv_std, gamma: g}, reshape(mu, c), reshape(var4, c)]
      end

      # Same compact formulas as the CPU kernel (see Ops::BatchNorm2D).
      def batchnorm2d_backward(ctx, dout, training:)
        x_hat = ctx[:x_hat]
        axes = [0, 2, 3]
        dgamma = sum(dout * x_hat, axis: axes)
        dbeta = sum(dout, axis: axes)

        k = ctx[:gamma] ? ctx[:gamma] * ctx[:inv_std] : ctx[:inv_std]
        dx = if training
          n, _c, h, w = dout.shape
          m = (n * h * w).to_f
          (k / m) * ((m * dout) - sum(dout, axis: axes, keepdims: true) -
                     (x_hat * sum(dout * x_hat, axis: axes, keepdims: true)))
        else
          k * dout
        end

        [dx, dgamma, dbeta]
      end

      # --- Helpers ------------------------------------------------------------

      # NCHW convolution on top of MLX's channels-last conv2d.
      def conv2d_nchw(x, w, stride, padding)
        x_nhwc = transpose(x, 0, 2, 3, 1)
        w_ohwi = transpose(w, 0, 2, 3, 1)
        transpose(Lib.conv2d(x_nhwc, w_ohwi, stride, padding), 0, 3, 1, 2)
      end

      # The kh*kw strided slices of the (zero-padded) input, one per window
      # offset, stacked into [K, N, C, OH, OW]: slice k holds, for every output
      # position, the element at offset k of its window. Max pooling is their
      # elementwise max.
      def pool_windows(x, kh, kw, stride, padding)
        n, c, h, w = x.shape
        if padding.positive?
          x = Lib.pad(x, [2, 3], [padding, padding], [padding, padding], 0.0)
          h += 2 * padding
          w += 2 * padding
        end
        oh = (h - kh) / stride + 1
        ow = (w - kw) / stride + 1

        windows = []
        kh.times do |ky|
          kw.times do |kx|
            windows << Lib.slice(
              x, [0, 0, ky, kx],
              [n, c, ky + stride * (oh - 1) + 1, kx + stride * (ow - 1) + 1],
              [1, 1, stride, stride]
            )
          end
        end
        Lib.stack(windows)
      end

      def integer_type?(x)
        x.is_a?(::Numo::Int8) || x.is_a?(::Numo::Int16) || x.is_a?(::Numo::Int32) || x.is_a?(::Numo::Int64) ||
          x.is_a?(::Numo::UInt8) || x.is_a?(::Numo::UInt16) || x.is_a?(::Numo::UInt32) || x.is_a?(::Numo::UInt64)
      end
    end

    register(:mlx, MLX)
  end
end
