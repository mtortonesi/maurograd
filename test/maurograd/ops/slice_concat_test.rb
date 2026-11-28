require "maurograd"

describe Maurograd::Ops::Slice do
  def tensor(*shape, requires_grad: true)
    Maurograd::Tensor.new(Numo::SFloat.new(*shape).seq, requires_grad: requires_grad)
  end

  it "selects a range along one axis, as Ruby ranges read" do
    x = tensor(2, 6)
    expect(x.slice(1, 1...3).data).to be == Numo::SFloat[[1, 2], [7, 8]]
    expect(x.slice(1, 1..3).shape).to be == [2, 3]
    expect(x.slice(1, 4..).data).to be == Numo::SFloat[[4, 5], [10, 11]]
    expect(x.slice(1, ..1).data).to be == Numo::SFloat[[0, 1], [6, 7]]
    expect(x.slice(-1, -2..).data).to be == Numo::SFloat[[4, 5], [10, 11]]
    expect(x.slice(0, 1..).data).to be == Numo::SFloat[[6, 7, 8, 9, 10, 11]]
    expect(x.slice(1, 3..100).shape).to be == [2, 3]
  end

  it "sends the gradient back to the selected positions, and zero elsewhere" do
    x = tensor(2, 3, 4)
    w = Numo::SFloat.new(2, 2, 4).rand_norm
    (x.slice(1, 1..2) * Maurograd::Tensor.new(w)).sum.backward

    expected = Numo::SFloat.zeros(2, 3, 4)
    expected[true, 1..2, true] = w
    expect(x.grad).to be == expected
  end

  it "rejects ranges that select nothing, and bad arguments" do
    x = tensor(2, 3)
    expect { x.slice(1, 3..) }.to raise_exception(ArgumentError, message: be =~ /selects nothing/)
    expect { x.slice(1, 2...2) }.to raise_exception(ArgumentError)
    expect { x.slice(1, -5..) }.to raise_exception(ArgumentError)
    expect { x.slice(2, 0..1) }.to raise_exception(ArgumentError, message: be =~ /out of range/)
    expect { x.slice(1, 1) }.to raise_exception(ArgumentError, message: be =~ /Range/)
  end

  it "splits into equal chunks, the last one smaller, as PyTorch does" do
    x = tensor(2, 8)
    expect(x.chunk(2, axis: 1).map(&:shape)).to be == [[2, 4], [2, 4]]
    expect(x.chunk(3, axis: -1).map(&:shape)).to be == [[2, 3], [2, 3], [2, 2]]
    expect(x.chunk(5, axis: 0).map(&:shape)).to be == [[1, 8], [1, 8]]
    expect { x.chunk(0) }.to raise_exception(ArgumentError)
  end

  it "accumulates the gradients of every chunk into one input" do
    x = tensor(3, 4)
    mean, log_var = x.chunk(2, axis: 1)
    (mean * 2.0 + log_var.exp).sum.backward

    expected = Numo::SFloat.zeros(3, 4)
    expected[true, 0..1] = 2.0
    expected[true, 2..3] = Numo::NMath.exp(x.data[true, 2..3])
    expect((x.grad - expected).abs.max).to be < 1e-6
  end
end

describe Maurograd::Ops::Concat do
  def randn(*shape, requires_grad: true)
    Maurograd::Tensor.new(Numo::SFloat.new(*shape).rand_norm, requires_grad: requires_grad)
  end

  it "joins tensors along an axis" do
    Maurograd.seed!(1)
    a = randn(2, 3, 4)
    b = randn(2, 1, 4)
    y = Maurograd.concat([a, b, a], axis: 1)
    expect(y.shape).to be == [2, 7, 4]
    expect(y.data[true, 0...3, true]).to be == a.data
    expect(y.data[true, 3...4, true]).to be == b.data
    expect(Maurograd.concat([a, a], axis: -1).shape).to be == [2, 3, 8]
  end

  it "gives each input its own part of the gradient, summed when it appears twice" do
    Maurograd.seed!(2)
    a = randn(2, 3)
    b = randn(2, 2)
    c = randn(2, 1, requires_grad: false)
    w = Numo::SFloat.new(2, 9).rand_norm
    (Maurograd.concat([a, b, c, a], axis: 1) * Maurograd::Tensor.new(w)).sum.backward

    expect(a.grad).to be == w[true, 0...3] + w[true, 6...9]
    expect(b.grad).to be == w[true, 3...5]
    expect(c.grad).to be_nil
  end

  it "undoes a chunk, and gradients flow through both" do
    Maurograd.seed!(3)
    x = randn(4, 6)
    y = Maurograd.concat(x.chunk(3, axis: 1).reverse, axis: 1)
    expect(y.data[true, 0...2]).to be == x.data[true, 4...6]
    (y * y).sum.backward
    expect((x.grad - x.data * 2).abs.max).to be < 1e-6
  end

  it "checks the shapes" do
    a = randn(2, 3)
    expect { Maurograd.concat([a, randn(3, 3)], axis: 1) }.to raise_exception(ArgumentError, message: be =~ /same shape/)
    expect { Maurograd.concat([a, randn(2, 3, 1)], axis: 1) }.to raise_exception(ArgumentError)
    expect { Maurograd.concat([], axis: 0) }.to raise_exception(ArgumentError)
    expect { Maurograd.concat([a, a.data], axis: 0) }.to raise_exception(ArgumentError)
  end

  it "doesn't record a graph without gradients" do
    a = randn(2, 2, requires_grad: false)
    expect(Maurograd.concat([a, a], axis: 0).creator).to be_nil
    expect(a.slice(0, 0..0).creator).to be_nil
  end
end
