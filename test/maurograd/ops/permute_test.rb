require "maurograd"

describe Maurograd::Ops::Permute do
  # loss = sum(y * w) for a fixed random w, so dL/dy = w and the expected
  # dL/dx is w moved back to x's layout. (With a plain sum, dL/dy would be
  # all ones and couldn't reveal a wrongly ordered gradient.)
  def weighted_loss(y, w)
    (y * Maurograd::Tensor.new(w)).sum
  end

  it "reorders axes in forward" do
    xd = Numo::SFloat.new(2, 3, 4, 5).seq
    y = Maurograd::Tensor.new(xd).permute(0, 2, 3, 1)

    expect(y.shape).to be == [2, 4, 5, 3]
    expect(y.data).to be == xd.transpose(0, 2, 3, 1)
  end

  it "backpropagates through permute with the inverse permutation" do
    Maurograd.seed!(1)
    xd = Numo::SFloat.new(2, 3, 4, 5).rand_norm
    x = Maurograd::Tensor.new(xd, requires_grad: true)
    y = x.permute(0, 2, 3, 1) # NCHW -> NHWC
    w = Numo::SFloat.new(*y.shape).rand_norm

    weighted_loss(y, w).backward

    expect(x.grad.shape).to be == [2, 3, 4, 5]
    expect(x.grad).to be == w.transpose(0, 3, 1, 2) # NHWC -> NCHW
  end

  it "backpropagates through transpose (last two axes by default)" do
    Maurograd.seed!(2)
    x = Maurograd::Tensor.new(Numo::SFloat.new(3, 4).rand_norm, requires_grad: true)
    y = x.transpose
    w = Numo::SFloat.new(4, 3).rand_norm

    weighted_loss(y, w).backward

    expect(y.creator).to be_a(Maurograd::Ops::Permute)
    expect(x.grad).to be == w.transpose
  end

  it "backpropagates through transpose of two chosen axes" do
    Maurograd.seed!(3)
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4).rand_norm, requires_grad: true)
    y = x.transpose(0, 2) # [4, 3, 2]
    w = Numo::SFloat.new(4, 3, 2).rand_norm

    weighted_loss(y, w).backward

    expect(x.grad).to be == w.transpose(2, 1, 0)
  end

  it "accepts negative axes" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4).seq)
    expect(x.permute(0, -1, -2).data).to be == x.data.transpose(0, 2, 1)
  end

  it "matches a numerical gradient when feeding a Linear layer" do
    Maurograd.seed!(4)
    xd = Numo::DFloat.new(3, 5).rand_norm
    w = Maurograd::Tensor.new(Numo::SFloat.new(2, 3).rand_norm)
    loss_of = lambda do |x_tensor|
      # x [3, 5] -> x^T [5, 3] -> Linear to [5, 2]
      Maurograd::Ops::Linear.apply(x_tensor.transpose, w).sum
    end

    x = Maurograd::Tensor.new(Numo::SFloat.cast(xd), requires_grad: true)
    loss_of.call(x).backward

    eps = 1e-2
    [[0, 0], [1, 3], [2, 4]].each do |i, j|
      plus = xd.dup
      plus[i, j] += eps
      minus = xd.dup
      minus[i, j] -= eps
      numeric = (loss_of.call(Maurograd::Tensor.new(Numo::SFloat.cast(plus))).item -
                 loss_of.call(Maurograd::Tensor.new(Numo::SFloat.cast(minus))).item) / (2 * eps)
      expect((x.grad[i, j] - numeric).abs).to be < 1e-2
    end
  end

  it "does not build a graph node when no gradient is needed" do
    y = Maurograd::Tensor.new(Numo::SFloat.new(2, 3).seq).transpose
    expect(y.creator).to be_nil
    expect(y.requires_grad).to be == false
  end

  it "rejects axes that are not a permutation" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3, 4).seq)
    expect { x.permute(0, 1) }.to raise_exception(ArgumentError)
    expect { x.permute(0, 1, 1) }.to raise_exception(ArgumentError)
    expect { x.permute(0, 1, 3) }.to raise_exception(ArgumentError)
  end
end
