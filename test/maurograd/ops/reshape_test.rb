require "maurograd"

describe Maurograd::Ops::Reshape do
  # loss = sum(y * w) for a fixed random w, so dL/dy = w. (With a plain sum,
  # dL/dy would be all ones and couldn't reveal a wrongly ordered gradient.)
  def weighted_loss(y, w)
    (y * Maurograd::Tensor.new(w)).sum
  end

  it "keeps the elements in row-major order" do
    xd = Numo::SFloat.new(2, 3, 4).seq
    y = Maurograd::Tensor.new(xd).reshape(6, 4)

    expect(y.shape).to be == [6, 4]
    expect(y.data.to_a.flatten).to be == xd.to_a.flatten
  end

  it "accepts the shape as an Array" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3).seq)
    expect(x.reshape([3, 2]).shape).to be == [3, 2]
  end

  it "regroups rather than transposes" do
    x = Maurograd::Tensor.new(Numo::SFloat[[0, 1, 2], [3, 4, 5]])
    expect(x.reshape(3, 2).data.to_a).to be == [[0, 1], [2, 3], [4, 5]]
    expect(x.transpose.data.to_a).to be == [[0, 3], [1, 4], [2, 5]]
  end

  it "infers one dimension given as -1" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4).seq)
    expect(x.reshape(2, -1).shape).to be == [2, 12]
    expect(x.reshape(-1).shape).to be == [24]
    expect(x.reshape(-1, 3, 2).shape).to be == [4, 3, 2]
  end

  it "backpropagates by reshaping the gradient back" do
    Maurograd.seed!(1)
    xd = Numo::SFloat.new(2, 3, 4).rand_norm
    x = Maurograd::Tensor.new(xd, requires_grad: true)
    y = x.reshape(4, 6)
    w = Numo::SFloat.new(4, 6).rand_norm

    weighted_loss(y, w).backward

    expect(y.creator).to be_a(Maurograd::Ops::Reshape)
    expect(x.grad.shape).to be == [2, 3, 4]
    expect(x.grad).to be == w.reshape(2, 3, 4)
  end

  it "reads a permuted (strided) input in logical order, forward and backward" do
    Maurograd.seed!(2)
    xd = Numo::SFloat.new(2, 3, 4).rand_norm
    x = Maurograd::Tensor.new(xd, requires_grad: true)
    y = x.permute(0, 2, 1).reshape(2, 12) # [2, 4, 3] -> [2, 12]
    w = Numo::SFloat.new(2, 12).rand_norm

    weighted_loss(y, w).backward

    expect(y.data).to be == xd.transpose(0, 2, 1).dup.reshape(2, 12)
    expect(x.grad).to be == w.reshape(2, 4, 3).transpose(0, 2, 1)
  end

  it "matches a numerical gradient between two Linear layers" do
    Maurograd.seed!(3)
    xd = Numo::DFloat.new(2, 6).rand_norm
    w1 = Maurograd::Tensor.new(Numo::SFloat.new(6, 6).rand_norm)
    w2 = Maurograd::Tensor.new(Numo::SFloat.new(2, 3).rand_norm)
    loss_of = lambda do |x_tensor|
      # [2, 6] -> Linear [2, 6] -> reshape [4, 3] -> Linear [4, 2], squared
      h = Maurograd::Ops::Linear.apply(x_tensor, w1).reshape(4, 3)
      out = Maurograd::Ops::Linear.apply(h, w2)
      (out * out).sum
    end

    x = Maurograd::Tensor.new(Numo::SFloat.cast(xd), requires_grad: true)
    loss_of.call(x).backward

    eps = 1e-2
    [[0, 0], [0, 5], [1, 2]].each do |i, j|
      plus = xd.dup
      plus[i, j] += eps
      minus = xd.dup
      minus[i, j] -= eps
      numeric = (loss_of.call(Maurograd::Tensor.new(Numo::SFloat.cast(plus))).item -
                 loss_of.call(Maurograd::Tensor.new(Numo::SFloat.cast(minus))).item) / (2 * eps)
      expect((x.grad[i, j] - numeric).abs).to be < 2e-2 * [numeric.abs, 1].max
    end
  end

  it "does not build a graph node when no gradient is needed" do
    y = Maurograd::Tensor.new(Numo::SFloat.new(2, 3).seq).reshape(6)
    expect(y.creator).to be_nil
    expect(y.requires_grad).to be == false
  end

  it "rejects shapes with a different element count, and bad dimensions" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4).seq)
    expect { x.reshape(5, 5) }.to raise_exception(ArgumentError)
    expect { x.reshape(5, -1) }.to raise_exception(ArgumentError)
    expect { x.reshape(-1, -1) }.to raise_exception(ArgumentError)
    expect { x.reshape(0, -1) }.to raise_exception(ArgumentError)
    expect { x.reshape(-2, -12) }.to raise_exception(ArgumentError)
    expect { x.reshape(2.0, 12) }.to raise_exception(ArgumentError)
  end
end
