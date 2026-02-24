require 'maurograd/tensor'
require 'maurograd/losses/cross_entropy_with_logits'

describe Maurograd::Losses::CrossEntropyWithLogits do
  it "computes a finite scalar loss and produces grad with correct shape (indices target)" do
    x = Maurograd::Tensor.new(
      Numo::SFloat[[2.0, 1.0, 0.1],
                   [0.5, 0.2, -1.0]],
      requires_grad: true
    )

    y = Maurograd::Tensor.new(Numo::SFloat[0, 2], requires_grad: false)

    loss = Maurograd::Losses::CrossEntropyWithLogits.apply(x, y)
    expect(finite_scalar?(loss.data)).to be_truthy

    loss.backward
    expect(x.grad.shape).to be == x.shape
  end

  it "supports one-hot targets [N,C]" do
    x = Maurograd::Tensor.new(
      Numo::SFloat[[1.0, 2.0],
                   [3.0, 0.0]],
      requires_grad: true
    )

    y = Maurograd::Tensor.new(
      Numo::SFloat[[0.0, 1.0],
                   [1.0, 0.0]],
      requires_grad: false
    )

    loss = Maurograd::Losses::CrossEntropyWithLogits.apply(x, y)
    expect(finite_scalar?(loss.data)).to be_truthy

    loss.backward
    expect(x.grad.shape).to be == [2, 2]
  end

  def finite_scalar?(x)
    x = x.to_f if x.respond_to?(:to_f)
  end
end
