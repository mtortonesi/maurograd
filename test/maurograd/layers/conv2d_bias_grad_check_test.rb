# test/maurograd/layers/conv2d_bias_grad_check.rb
require "maurograd/tensor"
require "maurograd/layers/conv2d"

describe "Conv2D Bias Gradient Check" do
  it "matches numerical gradient for one bias element" do
    epsilon = 1e-4
    tolerance = 1e-5

    conv = Maurograd::Layers::Conv2D.new(1, 1, 2, stride: 1, padding: 0)

    # deterministic weights, deterministic bias start
    conv.weights.data = Numo::SFloat.ones(1, 1, 2, 2)
    conv.bias.data = Numo::SFloat.zeros(1)

    x = Maurograd::Tensor.new(Numo::SFloat.new(1, 1, 3, 3).rand, requires_grad: true)

    # --- Analytic gradient ---
    out = conv.forward(x)
    loss = (out**2).sum
    loss.backward

    # Bias has shape [1] here, we check element 0
    grad_analytic = conv.bias.grad[0]

    # --- Numerical gradient ---
    original = conv.bias.data[0]

    # L(b + eps)
    conv.bias.data[0] = original + epsilon
    out_plus = conv.forward(x)
    loss_plus = (out_plus**2).sum.data

    # L(b - eps)
    conv.bias.data[0] = original - epsilon
    out_minus = conv.forward(x)
    loss_minus = (out_minus**2).sum.data

    # restore
    conv.bias.data[0] = original

    grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)

    diff = (grad_analytic - grad_numeric).abs

    # puts "analytic=#{grad_analytic} numeric=#{grad_numeric.to_f} diff=#{diff.to_f}"
    expect(diff).to be < tolerance
  end
end
