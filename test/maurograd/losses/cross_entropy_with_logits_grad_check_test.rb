require "maurograd/tensor"
require "maurograd/losses/cross_entropy_with_logits"

describe "CrossEntropyWithLogits gradient check" do
  it "matches numerical gradient for one logit element (indices target)" do
    h = 1e-4
    tol = 1e-5

    x_data = Numo::SFloat[[2.0, 1.0, 0.1],
      [0.5, 0.2, -1.0]]
    x = Maurograd::Tensor.new(x_data.copy, requires_grad: true)
    y = Maurograd::Tensor.new(Numo::SFloat[0, 2], requires_grad: false)

    loss = Maurograd::Losses::CrossEntropyWithLogits.apply(x, y)
    loss.backward

    i = 1
    j = 0
    g_analytic = x.grad[i, j]

    orig = x.data[i, j]

    x.data[i, j] = orig + h
    l1 = Maurograd::Losses::CrossEntropyWithLogits.apply(x, y).data

    x.data[i, j] = orig - h
    l2 = Maurograd::Losses::CrossEntropyWithLogits.apply(x, y).data

    x.data[i, j] = orig

    g_num = (l1 - l2) / (2.0 * h)
    diff = (g_analytic - g_num).abs

    expect(diff).to be < tol
  end
end
