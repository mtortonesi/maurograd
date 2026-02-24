require 'maurograd/tensor'
require 'maurograd/layers/conv2d'

describe "Conv2D Bias Gradient Check (multi out_channels)" do
  it "matches numerical gradient for one bias element when out_channels > 1" do
    epsilon = 1e-4
    tolerance = 1e-5

    in_channels = 1
    out_channels = 3
    kh = 2
    kw = 2

    conv = Maurograd::Layers::Conv2D.new(in_channels, out_channels, [kh, kw], stride: 1, padding: 0)

    # Deterministic parameters:
    # - Use distinct weights per output channel so each bias affects a different output map,
    #   making channel-order bugs easier to catch.
    #   W[0] = 1s, W[1] = 2s, W[2] = 3s
    w = Numo::SFloat.zeros(out_channels, in_channels, kh, kw)
    w[0, 0, true, true] = Numo::SFloat.ones(kh, kw) * 1.0
    w[1, 0, true, true] = Numo::SFloat.ones(kh, kw) * 2.0
    w[2, 0, true, true] = Numo::SFloat.ones(kh, kw) * 3.0
    conv.weights.data = w

    # Start bias at zero
    conv.bias.data = Numo::SFloat.zeros(out_channels)

    # Random input (requires_grad not strictly needed for bias check, but harmless)
    x = Maurograd::Tensor.new(Numo::SFloat.new(1, 1, 3, 3).rand, requires_grad: true)

    # --- Analytic gradient ---
    out = conv.forward(x)
    loss = (out**2).sum
    loss.backward

    # Check a non-trivial bias index
    b_idx = 2
    grad_analytic = conv.bias.grad[b_idx]

    # --- Numerical gradient ---
    original = conv.bias.data[b_idx]

    # L(b + eps)
    conv.bias.data[b_idx] = original + epsilon
    out_plus = conv.forward(x)
    loss_plus = (out_plus**2).sum.data

    # L(b - eps)
    conv.bias.data[b_idx] = original - epsilon
    out_minus = conv.forward(x)
    loss_minus = (out_minus**2).sum.data

    # restore
    conv.bias.data[b_idx] = original

    grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)
    diff = (grad_analytic - grad_numeric).abs

    # puts "analytic=#{grad_analytic} numeric=#{grad_numeric.to_f} diff=#{diff.to_f}"
    expect(diff).to be < tolerance
  end
end

