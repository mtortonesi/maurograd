require "maurograd/tensor"
require "maurograd/ops/relu"
require "maurograd/ops/sum"

describe Maurograd::Ops::ReLU do
  it "zeros out negatives and backprop passes gradient only where x>0" do
    x = Maurograd::Tensor.new(Numo::SFloat[-1.0, 0.0, 2.0], requires_grad: true)
    y = Maurograd::Ops::ReLU.apply(x)

    expect(y.data).to be == Numo::SFloat[0.0, 0.0, 2.0]

    loss = Maurograd::Ops::Sum.apply(y)
    loss.backward

    # d/dx sum(relu(x)) is [0,0,1] for [-1,0,2]
    expect(x.grad).to be == Numo::SFloat[0.0, 0.0, 1.0]
  end
end
