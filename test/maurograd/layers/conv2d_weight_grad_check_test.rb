require 'maurograd/tensor'
require 'maurograd/layers/conv2d'

describe "Conv2D Gradient Check" do
  it "has an analytic gradient that matches the numerical one" do
    epsilon = 1e-4
    tolerance = 1e-5

    # 1. Setup: small convolution for speed.
    conv = Maurograd::Layers::Conv2D.new(1, 1, 2, stride: 1, padding: 0)
    input = Maurograd::Tensor.new(Numo::SFloat.new(1, 1, 3, 3).rand, requires_grad: true)

    # 2. Analytic forward and backward.
    output = conv.forward(input)

    # Use the sum of squares as a scalar loss function.
    loss = (output**2).sum
    loss.backward

    # Save the analytic gradient computed by the framework.
    # We only check one element for brevity (e.g. weight [0,0,0,0]).
    grad_analytic = conv.weights.grad[0, 0, 0, 0]

    # 3. Numerical gradient computation.
    original_weight = conv.weights.data[0, 0, 0, 0]

    # Loss(W + epsilon)
    conv.weights.data[0, 0, 0, 0] = original_weight + epsilon
    out_plus = conv.forward(input)
    loss_plus = (out_plus**2).sum.data

    # Loss(W - epsilon)
    conv.weights.data[0, 0, 0, 0] = original_weight - epsilon
    out_minus = conv.forward(input)
    loss_minus = (out_minus**2).sum.data

    grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)

    # 4. Compare.
    diff = (grad_analytic - grad_numeric).abs

    expect(diff).to be < tolerance
  end
end
