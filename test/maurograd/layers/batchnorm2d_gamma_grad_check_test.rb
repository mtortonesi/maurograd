require "maurograd/tensor"
require "maurograd/layers/batchnorm2d"

describe "BatchNorm2D gamma gradient" do
  it "has an analytic gradient that matches the numerical one" do
    epsilon = 1e-3
    tolerance = 1e-3

    bn = Maurograd::Layers::BatchNorm2D.new(3)
    input = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4, 4).rand + 0.1, requires_grad: false)

    output = bn.forward(input)
    loss = (output**2).sum
    loss.backward

    grad_analytic = bn.gamma.grad[1]

    original = bn.gamma.data[1]

    bn.gamma.data[1] = original + epsilon
    loss_plus = (bn.forward(input)**2).sum.data

    bn.gamma.data[1] = original - epsilon
    loss_minus = (bn.forward(input)**2).sum.data

    bn.gamma.data[1] = original

    grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)

    diff = (grad_analytic - grad_numeric).abs
    expect(diff).to be < tolerance
  end
end
