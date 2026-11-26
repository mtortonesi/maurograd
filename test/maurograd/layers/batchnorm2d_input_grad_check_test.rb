require "maurograd/tensor"
require "maurograd/layers/batchnorm2d"

describe "BatchNorm2D input gradient (training mode)" do
  it "has an analytic gradient that matches the numerical one" do
    epsilon = 1e-3
    tolerance = 1e-3

    bn = Maurograd::Layers::BatchNorm2D.new(3)
    input = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4, 4).rand + 0.1, requires_grad: true)

    output = bn.forward(input)
    loss = (output**2).sum
    loss.backward

    # Check a handful of elements, not just one: batch statistics couple
    # every element in a channel together, so this exercises more of the
    # dvar/dmu cross terms than a single-element check would.
    [[0, 0, 0, 0], [1, 1, 2, 3], [0, 2, 3, 1]].each do |idx|
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
end
