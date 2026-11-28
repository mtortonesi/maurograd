require "maurograd"

describe "Tensor#logsumexp, #log_softmax and #softmax" do
  def reference_lse(x, axis)
    Numo::NMath.log(Numo::NMath.exp(x).sum(axis: axis))
  end

  # Finite-difference gradient of sum(f(x) * w), in double precision.
  def numeric_grad(xd, w, h = 1e-3)
    grad = Numo::DFloat.zeros(*xd.shape)
    xd.size.times do |i|
      plus = xd.dup
      minus = xd.dup
      plus[i] += h
      minus[i] -= h
      grad[i] = ((yield(plus) - yield(minus)) * w).sum / (2 * h)
    end
    grad
  end

  it "computes logsumexp over all elements, one axis or several" do
    Maurograd.seed!(1)
    xd = Numo::DFloat.new(2, 3, 4).rand_norm
    x = Maurograd::Tensor.new(Numo::SFloat.cast(xd))

    expect((x.logsumexp.item - Math.log(Numo::NMath.exp(xd).sum)).abs).to be < 1e-5
    expect((Numo::DFloat.cast(x.logsumexp(axis: 1).data) - reference_lse(xd, 1)).abs.max).to be < 1e-5
    expect((Numo::DFloat.cast(x.logsumexp(axis: [0, 2]).data) - reference_lse(xd, [0, 2])).abs.max).to be < 1e-5
    expect(x.logsumexp(axis: -1, keepdims: true).shape).to be == [2, 3, 1]
    expect(x.logsumexp(axis: 1).shape).to be == [2, 4]
  end

  it "backpropagates logsumexp (checked by finite differences)" do
    Maurograd.seed!(2)
    xd = Numo::DFloat.new(3, 4).rand_norm
    w = Numo::DFloat.new(3).rand_norm
    x = Maurograd::Tensor.new(Numo::SFloat.cast(xd), requires_grad: true)
    (x.logsumexp(axis: 1) * Maurograd::Tensor.new(Numo::SFloat.cast(w))).sum.backward

    numeric = numeric_grad(xd, w) { |v| reference_lse(v, 1) }
    expect((Numo::DFloat.cast(x.grad) - numeric).abs.max).to be < 1e-4
  end

  it "backpropagates log_softmax (checked by finite differences)" do
    Maurograd.seed!(3)
    xd = Numo::DFloat.new(3, 4).rand_norm
    w = Numo::DFloat.new(3, 4).rand_norm
    x = Maurograd::Tensor.new(Numo::SFloat.cast(xd), requires_grad: true)
    (x.log_softmax(axis: 1) * Maurograd::Tensor.new(Numo::SFloat.cast(w))).sum.backward

    numeric = numeric_grad(xd, w) { |v| v - reference_lse(v, 1).reshape(3, 1) }
    expect((Numo::DFloat.cast(x.grad) - numeric).abs.max).to be < 1e-4
  end

  it "gives probabilities that sum to one along the axis" do
    Maurograd.seed!(4)
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 5, 3).rand_norm * 3)
    p = x.softmax(axis: 1)
    expect((p.data.sum(axis: 1) - 1).abs.max).to be < 1e-6
    expect((p.data - Numo::NMath.exp(x.log_softmax(axis: 1).data)).abs.max).to be < 1e-7
  end

  it "stays finite where exp overflows or underflows" do
    x = Maurograd::Tensor.new(Numo::SFloat[[1000, 1000, -1000], [-1000, -1001, -1002]], requires_grad: true)
    lse = x.logsumexp(axis: 1)
    expect((lse.data - Numo::SFloat[1000 + Math.log(2), -1000 + Math.log(1 + Math.exp(-1) + Math.exp(-2))]).abs.max).to be < 1e-3

    lp = x.log_softmax
    expect(x.backend.all_finite?(lp.data)).to be == true
    expect((lp.data[0, 2] - (-2000 - Math.log(2))).abs).to be < 1e-2
    (lse.sum + lp.sum).backward
    expect(x.backend.all_finite?(x.grad)).to be == true
  end

  it "doesn't record a graph without gradients" do
    x = Maurograd::Tensor.new(Numo::SFloat[[1, 2]])
    expect(x.logsumexp.creator).to be_nil
    expect(x.log_softmax.creator).to be_nil
  end
end
