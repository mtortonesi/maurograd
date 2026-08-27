require "maurograd/tensor"
require "maurograd/ops/linear"
require "maurograd/ops/power"
require "maurograd/ops/sum"

describe "Linear bias grad-check (multi out_features)" do
  it "matches numerical gradient for one bias element and keeps correct shape" do
    h = 1e-4
    tol = 1e-5

    x = Maurograd::Tensor.new(Numo::SFloat.new(5, 3).rand, requires_grad: true)
    w = Maurograd::Tensor.new(Numo::SFloat.new(7, 3).rand, requires_grad: true)  # Out=7
    b = Maurograd::Tensor.new(Numo::SFloat.new(7).rand, requires_grad: true)

    y = Maurograd::Ops::Linear.apply(x, w, b)
    loss = Maurograd::Ops::Sum.apply(Maurograd::Ops::Power.apply(y, 2))
    loss.backward

    expect(b.grad.shape).to be == b.shape

    k = 4
    g_analytic = b.grad[k]

    orig = b.data[k]

    b.data[k] = orig + h
    y1 = Maurograd::Ops::Linear.apply(x, w, b)
    l1 = Maurograd::Ops::Sum.apply(Maurograd::Ops::Power.apply(y1, 2)).data

    b.data[k] = orig - h
    y2 = Maurograd::Ops::Linear.apply(x, w, b)
    l2 = Maurograd::Ops::Sum.apply(Maurograd::Ops::Power.apply(y2, 2)).data

    b.data[k] = orig

    g_num = (l1 - l2) / (2.0 * h)
    diff = (g_analytic - g_num).abs
    expect(diff).to be < tol
  end
end
