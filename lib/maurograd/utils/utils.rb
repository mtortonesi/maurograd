require_relative '../ext'
require_relative '../debug'
require_relative '../seed'

module Maurograd
  module Utils
    # Reduce ("undo") broadcasting for gradients.
    #
    # In the forward pass, Numo (like NumPy/PyTorch) may broadcast tensors automatically.
    # Example (typical in CNNs):
    #
    #   A: [N, C, H, W]
    #   B: [1, C, 1, 1]   # per-channel bias
    #   Y = A + B         # B is broadcast over N, H, W
    #
    # In the backward pass, the gradient coming from Y has shape [N, C, H, W],
    # but the gradient w.r.t. B must have B's *original* shape: [1, C, 1, 1].
    #
    # To get that, we must SUM the incoming gradient along every axis that was
    # broadcast (i.e., every axis where the original shape had size 1, and any
    # extra leading axes that were added).
    #
    # Params:
    # - grad:  Numo::NArray gradient (shape = broadcasted shape)
    # - shape: target/original shape we want to reduce back to
    #
    # Returns:
    # - A Numo::NArray with shape == `shape`
    def self.unbroadcast(grad, shape)
      # If shapes already match, nothing to do.
      return grad if grad.shape == shape

      res = grad

      # 1) Remove extra leading dimensions.
      #
      # Example:
      #   grad  = [N, C, H, W]
      #   shape = [C, H, W]
      #
      # This happens when an operand had fewer dimensions and was implicitly
      # left-padded with ones. We reduce along axis 0 until ranks match.
      while res.ndim > shape.length
        res = res.sum(axis: 0, keepdims: false)
      end

      # 2) For remaining axes, if the target shape has a 1, that axis was broadcast.
      # We must sum along that axis and keep the dimension (keepdims: true)
      # so the final shape matches exactly.
      res.shape.each_with_index do |dim, i|
        next if dim == shape[i]

        if shape[i] == 1
          res = res.sum(axis: i, keepdims: true)
        else
          # If shapes differ and target axis is not 1, it's a shape mismatch.
          raise ArgumentError, "Incompatible shapes for unbroadcast: grad #{res.shape} vs target #{shape}"
        end
      end

      res
    end


    # Zero-pad an NCHW tensor: [N, C, H, W] -> [N, C, H+2P, W+2P]
    #
    # Why do we need this helper?
    # ---------------------------
    # Some Numo versions do not provide `Numo::NArray#pad`, while padding is essential
    # for convolution. Implementing it explicitly keeps Maurograd compatible and
    # avoids relying on optional extensions.
    #
    # Padding is applied only on the spatial dimensions (H and W), leaving batch (N)
    # and channels (C) unchanged.
    #
    # Params:
    # - x:        Numo::NArray (expected float type), shape [N, C, H, W]
    # - padding:  integer P
    #
    # Returns:
    # - If padding == 0: returns x (no copy)
    # - Else: a new tensor with x copied into the centered region
    def self.pad_nchw(x, padding)
      return x if padding == 0

      n, c, h, w = x.shape

      # Allocate padded output filled with zeros.
      out = Numo::SFloat.zeros(n, c, h + 2 * padding, w + 2 * padding)

      # Copy x into the centered region.
      out[true, true, padding...(padding + h), padding...(padding + w)] = x
      out
    end


    def self.im2col(*args)
      Maurograd::Ext.im2col(*args)
    end


    def self.col2im(*args)
      Maurograd::Ext.col2im(*args)
    end


    # 2D transpose helper.
    # In Numo, transpose(*axes) is a general axis permutation.
    # For a true 2D matrix transpose you want transpose(1, 0), not transpose(0, 1).
    def self.t2d(x)
      x.transpose(1, 0)
    end

    def self.assert_finite!(na, where:)
      # Off by default (see Maurograd.debug): this walks the entire array,
      # which is pure overhead on every forward/backward call once a model
      # is known to be numerically stable.
      return unless Maurograd.debug

      # na is a Numo::NArray.
      # Numo does not provide isfinite? universally, so we use two checks:
      flat = na.flatten
      # NaN check: NaN != NaN is true in IEEE 754.
      if (flat.ne(flat).any?)
        raise "Non-finite detected (NaN) at #{where}"
      end
      # Inf check: treat values with abs > 1e30 as effectively infinite.
      if (flat.abs.gt(1e30).any?)
        raise "Non-finite detected (Inf/huge) at #{where}"
      end
    end

    # Draws a Numo::SFloat array of `shape` from a Gaussian(mean, std).
    #
    # Without a seed, this is exactly Numo::SFloat#rand_norm.
    #
    # With a seed, it deliberately avoids Numo::NArray.rand_norm: that method
    # draws from a single RNG that is global and process-wide (seeded to a
    # fixed constant at load time, unless Numo::NArray.srand is called), so
    # seeding it for one layer would silently perturb every other unrelated
    # Numo random call for the rest of the process. Instead we use a private
    # Random.new(seed) instance plus a Box-Muller transform, matching the
    # same seeding pattern Datasets::Batching already uses, so a seed here
    # affects only this call.
    def self.rand_norm(*shape, mean: 0.0, std: 1.0, seed: nil)
      return Numo::SFloat.new(*shape).rand_norm(mean, std) if seed.nil?

      rng = Random.new(seed)
      n = shape.reduce(1, :*)

      data = Array.new(n) do
        u1 = rng.rand
        u1 = Float::EPSILON if u1 <= 0.0 # avoid log(0)
        u2 = rng.rand
        r = Math.sqrt(-2.0 * Math.log(u1))
        mean + std * r * Math.cos(2.0 * Math::PI * u2)
      end

      Numo::SFloat.asarray(data).reshape(*shape)
    end

    def self.clip_grad_norm_(params, max_norm, eps: 1e-6)
      # params: array of Tensor (model parameters).
      # Assumes p.grad is a Numo::NArray float.
      total_sq = 0.0

      params.each do |p|
        next unless p.grad
        g = p.grad
        # Sum of squared gradient elements.
        total_sq += (g * g).sum.to_f
      end

      total_norm = Math.sqrt(total_sq)
      if total_norm > max_norm
        scale = max_norm / (total_norm + eps)
        params.each do |p|
          next unless p.grad
          p.grad.inplace * scale
        end
      end

      total_norm
    end

  end
end
