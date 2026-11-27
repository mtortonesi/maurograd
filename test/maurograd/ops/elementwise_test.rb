require "maurograd"

describe Maurograd::Ops::Elementwise do
  # name => [reference f on DFloat, input range]
  def functions
    nm = Numo::NMath
    {
      exp: [->(x) { nm.exp(x) }, -3..3],
      log: [->(x) { nm.log(x) }, 0.1..5],
      sigmoid: [->(x) { 1.0 / (nm.exp(-x) + 1.0) }, -6..6],
      tanh: [->(x) { nm.tanh(x) }, -3..3],
      silu: [->(x) { x / (nm.exp(-x) + 1.0) }, -6..6]
    }
  end

  def inputs_in(range, n = 24)
    lo = range.begin.to_f
    Numo::DFloat.new(n).rand * (range.end - lo) + lo
  end

  it "computes each function like the Numo reference" do
    Maurograd.seed!(1)
    functions.each do |name, (ref, range)|
      xd = inputs_in(range)
      y = Maurograd::Tensor.new(Numo::SFloat.cast(xd)).public_send(name)
      expect((Numo::DFloat.cast(y.data) - ref.call(xd)).abs.max).to be < 1e-5 * [ref.call(xd).abs.max, 1].max
    end
  end

  it "backpropagates each function's derivative (checked by finite differences)" do
    Maurograd.seed!(2)
    functions.each do |name, (ref, range)|
      xd = inputs_in(range)
      x = Maurograd::Tensor.new(Numo::SFloat.cast(xd), requires_grad: true)
      w = Numo::SFloat.new(xd.size).rand_norm
      (x.public_send(name) * Maurograd::Tensor.new(w)).sum.backward

      h = 1e-4
      numeric = (ref.call(xd + h) - ref.call(xd - h)) / (2 * h) * w
      expect((Numo::DFloat.cast(x.grad) - numeric).abs.max).to be < 1e-4 * [numeric.abs.max, 1].max
    end
  end

  it "stays finite for large inputs in sigmoid, tanh and silu" do
    x = Maurograd::Tensor.new(Numo::SFloat[-200, -50, 0, 50, 200], requires_grad: true)
    %i[sigmoid tanh silu].each do |name|
      x.zero_grad
      y = x.public_send(name)
      y.sum.backward
      expect(x.backend.all_finite?(y.data)).to be == true
      expect(x.backend.all_finite?(x.grad)).to be == true
    end
    expect(x.sigmoid.data.to_a).to be == [0.0, x.sigmoid.data[1], 0.5, x.sigmoid.data[3], 1.0]
    expect(x.tanh.data.to_a.values_at(0, 2, 4)).to be == [-1.0, 0.0, 1.0]
  end

  it "doesn't build a graph node when no gradient is needed" do
    y = Maurograd::Tensor.new(Numo::SFloat[1, 2]).exp
    expect(y.creator).to be_nil
  end

  it "provides Sigmoid, Tanh and SiLU layers" do
    x = Maurograd::Tensor.new(Numo::SFloat[-1, 0, 2])
    model = Maurograd::Layers::Sequential.new(
      Maurograd::Layers::SiLU.new, Maurograd::Layers::Tanh.new, Maurograd::Layers::Sigmoid.new
    )
    expect(model.call(x).data).to be == x.silu.tanh.sigmoid.data
  end
end

describe "Tensor division and negation" do
  it "divides by a number and by a tensor, with gradients" do
    a = Maurograd::Tensor.new(Numo::SFloat[2, 6], requires_grad: true)
    b = Maurograd::Tensor.new(Numo::SFloat[4, 3], requires_grad: true)

    expect((a / 2).data.to_a).to be == [1.0, 3.0]
    (a / b).sum.backward

    expect((a.grad - Numo::SFloat[1.0 / 4, 1.0 / 3]).abs.max).to be < 1e-6
    expect((b.grad - Numo::SFloat[-2.0 / 16, -6.0 / 9]).abs.max).to be < 1e-6
  end

  it "negates, with gradient -1" do
    a = Maurograd::Tensor.new(Numo::SFloat[1, -2], requires_grad: true)
    (-a).sum.backward
    expect((-a).data.to_a).to be == [-1.0, 2.0]
    expect(a.grad.to_a).to be == [-1.0, -1.0]
  end
end
