require 'maurograd/tensor'
require 'maurograd/ops/flatten'
require 'maurograd/ops/sum'

describe Maurograd::Ops::Flatten do
  it "flattens [N, ...] to [N, D] and backprop reshapes gradients correctly" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4).seq, requires_grad: true)

    y = Maurograd::Ops::Flatten.apply(x)
    expect(y.shape).to be == [2, 12]

    loss = Maurograd::Ops::Sum.apply(y)
    loss.backward

    expect(x.grad.shape).to be == [2, 3, 4]
    expect(x.grad).to be == Numo::SFloat.ones(2, 3, 4)
  end
end

