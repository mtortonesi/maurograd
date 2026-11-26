require "maurograd/tensor"
require "maurograd/layers/batchnorm2d"

describe "BatchNorm2D eval mode" do
  it "normalizes using the running statistics, not the batch statistics" do
    bn = Maurograd::Layers::BatchNorm2D.new(2)
    bn.running_mean = Numo::SFloat[1.0, -2.0]
    bn.running_var = Numo::SFloat[4.0, 9.0]
    bn.eval

    input = Maurograd::Tensor.new(Numo::SFloat.new(1, 2, 2, 2).rand, requires_grad: false)
    output = bn.forward(input)

    c = 0
    expected = (input.data[0, c, true, true] - bn.running_mean[c]) / Math.sqrt(bn.running_var[c] + bn.eps)
    diff = (output.data[0, c, true, true] - expected).abs.max
    expect(diff).to be < 1e-5

    c = 1
    expected = (input.data[0, c, true, true] - bn.running_mean[c]) / Math.sqrt(bn.running_var[c] + bn.eps)
    diff = (output.data[0, c, true, true] - expected).abs.max
    expect(diff).to be < 1e-5
  end

  it "does not update the running statistics" do
    bn = Maurograd::Layers::BatchNorm2D.new(2)
    bn.running_mean = Numo::SFloat[1.0, -2.0]
    bn.running_var = Numo::SFloat[4.0, 9.0]
    bn.eval

    before_mean = bn.running_mean.dup
    before_var = bn.running_var.dup

    input = Maurograd::Tensor.new(Numo::SFloat.new(1, 2, 2, 2).rand, requires_grad: false)
    bn.forward(input)

    expect((bn.running_mean - before_mean).abs.max).to be == 0.0
    expect((bn.running_var - before_var).abs.max).to be == 0.0
  end

  it "has an analytic input gradient that matches the numerical one" do
    epsilon = 1e-3
    tolerance = 1e-3

    bn = Maurograd::Layers::BatchNorm2D.new(2)
    bn.running_mean = Numo::SFloat[1.0, -2.0]
    bn.running_var = Numo::SFloat[4.0, 9.0]
    bn.eval

    input = Maurograd::Tensor.new(Numo::SFloat.new(1, 2, 2, 2).rand, requires_grad: true)

    output = bn.forward(input)
    loss = (output**2).sum
    loss.backward

    idx = [0, 1, 0, 1]
    grad_analytic = input.grad[*idx]

    original = input.data[*idx]

    input.data[*idx] = original + epsilon
    loss_plus = (bn.forward(input)**2).sum.data

    input.data[*idx] = original - epsilon
    loss_minus = (bn.forward(input)**2).sum.data

    input.data[*idx] = original

    grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)

    diff = (grad_analytic - grad_numeric).abs
    expect(diff).to be < tolerance
  end
end
