require "maurograd/tensor"
require "maurograd/ops/linear"
require "maurograd/ops/power"
require "maurograd/ops/sum"

describe "Linear grad-check (input element)" do
  it "matches numerical gradient for one input element" do
    h = 1e-4
    tol = 1e-5

    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 3).rand, requires_grad: true)
    w = Maurograd::Tensor.new(Numo::SFloat.new(4, 3).rand, requires_grad: true)
    b = Maurograd::Tensor.new(Numo::SFloat.new(4).rand, requires_grad: true)

    y = Maurograd::Ops::Linear.apply(x, w, b)
    loss = Maurograd::Ops::Sum.apply(Maurograd::Ops::Power.apply(y, 2)) # sum(y^2)
    loss.backward

    i = 0
    j = 1
    g_analytic = x.grad[i, j]

    orig = x.data[i, j]

    x.data[i, j] = orig + h
    y1 = Maurograd::Ops::Linear.apply(x, w, b)
    l1 = Maurograd::Ops::Sum.apply(Maurograd::Ops::Power.apply(y1, 2)).data

    x.data[i, j] = orig - h
    y2 = Maurograd::Ops::Linear.apply(x, w, b)
    l2 = Maurograd::Ops::Sum.apply(Maurograd::Ops::Power.apply(y2, 2)).data

    x.data[i, j] = orig

    g_num = (l1 - l2) / (2.0 * h)
    diff = (g_analytic - g_num).abs
    expect(diff).to be < tol
  end
end
