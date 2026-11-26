require "maurograd"

# Checks the C BatchNorm2D kernels element-by-element against a pure-Numo
# reference that follows the step-by-step derivation in ops/batchnorm2d.rb
# (dvar, dmu, then dx), while the kernel uses the compact closed form.
describe "BatchNorm2D C kernels" do
  let(:eps) { 1e-5 }

  def reference(x, dout, gamma, beta, mean: nil, var: nil)
    n, c, h, w = x.shape
    m = n * h * w
    training = mean.nil?
    axes = [0, 2, 3]

    if training
      mean = x.mean(axis: axes, keepdims: true)
      var = ((x - mean)**2).mean(axis: axes, keepdims: true)
    else
      mean = mean.reshape(1, c, 1, 1)
      var = var.reshape(1, c, 1, 1)
    end
    diff = x - mean
    inv_std = 1.0 / Numo::NMath.sqrt(var + eps)
    x_hat = diff * inv_std
    g = gamma.reshape(1, c, 1, 1)
    y = x_hat * g + beta.reshape(1, c, 1, 1)

    dgamma = (dout * x_hat).sum(axis: axes)
    dbeta = dout.sum(axis: axes)
    dxhat = dout * g
    dx = if training
      dvar = (dxhat * diff).sum(axis: axes, keepdims: true) * -0.5 * (var + eps)**-1.5
      dmu = (dxhat * -inv_std).sum(axis: axes, keepdims: true) +
        dvar * (diff * -2.0).mean(axis: axes, keepdims: true)
      (dxhat * inv_std) + (dvar * diff * (2.0 / m)) + (dmu * (1.0 / m))
    else
      dxhat * inv_std
    end

    [y, dx, dgamma, dbeta]
  end

  def run_layer(bn, x, dout)
    input = Maurograd::Tensor.new(x, requires_grad: true)
    y = bn.forward(input)
    y.backward(dout)
    [y.data, input.grad, bn.gamma.grad, bn.beta.grad]
  end

  def max_rel_err(a, b)
    ((a - b).abs.max / [b.abs.max, 1e-6].max).to_f
  end

  it "matches the step-by-step reference in training mode" do
    Maurograd.seed!(11)
    x = Numo::DFloat.new(4, 3, 5, 6).rand_norm * 2.0 + 0.5
    dout = Numo::DFloat.new(4, 3, 5, 6).rand_norm
    bn = Maurograd::Layers::BatchNorm2D.new(3, eps: eps)
    bn.gamma.data = Numo::SFloat[0.5, 1.5, -2.0]
    bn.beta.data = Numo::SFloat[0.1, -0.3, 0.0]

    got = run_layer(bn, Numo::SFloat.cast(x), Numo::SFloat.cast(dout))
    want = reference(x, dout, Numo::DFloat.cast(bn.gamma.data), Numo::DFloat.cast(bn.beta.data))

    got.zip(want).each { |g, w| expect(max_rel_err(Numo::DFloat.cast(g), w)).to be < 1e-4 }
  end

  it "matches the reference in eval mode, using the running statistics" do
    Maurograd.seed!(12)
    x = Numo::DFloat.new(2, 2, 4, 4).rand_norm
    dout = Numo::DFloat.new(2, 2, 4, 4).rand_norm
    bn = Maurograd::Layers::BatchNorm2D.new(2, eps: eps)
    bn.running_mean[] = Numo::SFloat[0.2, -0.4]
    bn.running_var[] = Numo::SFloat[1.5, 0.3]
    bn.gamma.data = Numo::SFloat[2.0, 0.7]
    bn.eval

    got = run_layer(bn, Numo::SFloat.cast(x), Numo::SFloat.cast(dout))
    want = reference(
      x, dout, Numo::DFloat.cast(bn.gamma.data), Numo::DFloat.cast(bn.beta.data),
      mean: Numo::DFloat.cast(bn.running_mean), var: Numo::DFloat.cast(bn.running_var)
    )

    got.zip(want).each { |g, w| expect(max_rel_err(Numo::DFloat.cast(g), w)).to be < 1e-4 }
  end

  it "accepts non-contiguous inputs and DFloat upstream gradients" do
    Maurograd.seed!(13)
    base = Numo::SFloat.new(2, 4, 4, 3).rand_norm
    x = base.transpose(0, 3, 1, 2) # NHWC -> NCHW view
    dout = Numo::DFloat.new(2, 3, 4, 4).rand_norm
    bn = Maurograd::Layers::BatchNorm2D.new(3, eps: eps)

    got = run_layer(bn, x, dout)
    want = reference(Numo::DFloat.cast(x), dout, Numo::DFloat.ones(3), Numo::DFloat.zeros(3))

    got.zip(want).each { |g, w| expect(max_rel_err(Numo::DFloat.cast(g), w)).to be < 1e-4 }
  end
end
