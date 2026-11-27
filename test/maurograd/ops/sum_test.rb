require "maurograd"

describe Maurograd::Ops::Sum do
  def weighted(y, w)
    (y * Maurograd::Tensor.new(w)).sum
  end

  let(:xd) { Numo::SFloat.new(2, 3, 4, 5).seq }

  it "sums everything by default" do
    y = Maurograd::Tensor.new(xd).sum
    expect(y.data.to_f).to be == xd.sum
  end

  it "sums over one axis, several axes and negative axes" do
    x = Maurograd::Tensor.new(xd)
    expect(x.sum(axis: 1).data).to be == xd.sum(axis: 1)
    expect(x.sum(axis: [2, 3]).shape).to be == [2, 3]
    expect(x.sum(axis: [2, 3]).data).to be == xd.sum(axis: [2, 3])
    expect(x.sum(axis: -1).data).to be == xd.sum(axis: 3)
  end

  it "keeps the summed axes with keepdims" do
    x = Maurograd::Tensor.new(xd)
    expect(x.sum(axis: [2, 3], keepdims: true).shape).to be == [2, 3, 1, 1]
    expect(x.sum(keepdims: true).shape).to be == [1, 1, 1, 1]
  end

  it "copies the gradient back along the summed axes" do
    Maurograd.seed!(1)
    x = Maurograd::Tensor.new(xd, requires_grad: true)
    w = Numo::SFloat.new(2, 4).rand_norm # one weight per output element
    weighted(x.sum(axis: [1, 3]), w).backward

    # Every x[n, c, h, w'] was summed into output [n, h]: its gradient is w[n, h].
    expected = Numo::SFloat.zeros(2, 3, 4, 5) + w.reshape(2, 1, 4, 1)
    expect(x.grad).to be == expected
  end

  it "backpropagates with keepdims too" do
    Maurograd.seed!(2)
    x = Maurograd::Tensor.new(xd, requires_grad: true)
    w = Numo::SFloat.new(2, 3, 1, 1).rand_norm
    weighted(x.sum(axis: [2, 3], keepdims: true), w).backward

    expect(x.grad).to be == Numo::SFloat.zeros(2, 3, 4, 5) + w
  end

  it "rejects axes out of range, repeated or not Integers" do
    x = Maurograd::Tensor.new(xd)
    expect { x.sum(axis: 4) }.to raise_exception(ArgumentError)
    expect { x.sum(axis: -5) }.to raise_exception(ArgumentError)
    expect { x.sum(axis: [1, -3]) }.to raise_exception(ArgumentError)
    expect { x.sum(axis: 1.0) }.to raise_exception(ArgumentError)
  end
end

describe "Tensor#mean" do
  it "averages over everything or over chosen axes" do
    xd = Numo::SFloat.new(2, 3, 4).seq
    x = Maurograd::Tensor.new(xd)
    expect(x.mean.data.to_f).to be == xd.mean
    expect((x.mean(axis: [0, 2]).data - xd.mean(axis: [0, 2])).abs.max).to be < 1e-6
  end

  it "spreads the gradient evenly over the averaged elements" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4).seq, requires_grad: true)
    x.mean(axis: -1).sum.backward
    expect(x.grad).to be == Numo::SFloat.new(2, 3, 4).fill(0.25)
  end

  it "works as global average pooling: [N, C, H, W] -> [N, C]" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4, 4).rand, requires_grad: true)
    y = x.mean(axis: [2, 3])
    expect(y.shape).to be == [2, 3]
    y.sum.backward
    expect(x.grad).to be == Numo::SFloat.new(2, 3, 4, 4).fill(1.0 / 16)
  end
end
