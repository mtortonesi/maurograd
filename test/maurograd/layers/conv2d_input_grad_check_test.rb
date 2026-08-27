require 'maurograd/tensor'
require 'maurograd/layers/conv2d'

describe "Conv2D Input Gradient Check" do
  it "matches numerical gradient for one input element" do
    epsilon = 1e-4
    tolerance = 1e-5

    conv = Maurograd::Layers::Conv2D.new(1, 1, 2, stride: 1, padding: 0)

    # Make everything deterministic and well-conditioned
    conv.weights.data = Numo::SFloat.ones(1, 1, 2, 2)
    conv.bias.data = Numo::SFloat.zeros(1)

    x_data = Numo::SFloat.new(1, 1, 3, 3).rand
    x = Maurograd::Tensor.new(x_data, requires_grad: true)

    # --- Analytic gradient ---
    out = conv.forward(x)
    loss = (out**2).sum
    loss.backward

    # Pick one input element to check (e.g. x[0,0,1,1])
    idx = [0, 0, 1, 1]
    grad_analytic = x.grad[*idx]

    # --- Numerical gradient ---
    original = x.data[*idx]

    # L(x + eps)
    x.data[*idx] = original + epsilon
    out_plus = conv.forward(x)
    loss_plus = (out_plus**2).sum.data

    # L(x - eps)
    x.data[*idx] = original - epsilon
    out_minus = conv.forward(x)
    loss_minus = (out_minus**2).sum.data

    # restore
    x.data[*idx] = original

    grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)

    diff = (grad_analytic - grad_numeric).abs

    # puts "analytic=#{grad_analytic} numeric=#{grad_numeric.to_f} diff=#{diff.to_f}"
    expect(diff).to be < tolerance
  end
end

