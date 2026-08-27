require 'maurograd/tensor'
require 'maurograd/ops/conv2d'

describe "Conv2D backward without bias" do
  it "does not crash and matches numerical gradient on one input and one weight element" do
    epsilon = 1e-3
    tolerance = 5e-2

    x = Maurograd::Tensor.new(Numo::SFloat.new(1, 2, 5, 5).rand - 0.5, requires_grad: true)
    w = Maurograd::Tensor.new(Numo::SFloat.new(3, 2, 3, 3).rand - 0.5, requires_grad: true)

    recompute = -> { (Maurograd::Ops::Conv2D.apply(x, w) ** 2).sum }

    loss = recompute.call
    loss.backward

    expect(x.grad).not.to be == nil
    expect(w.grad).not.to be == nil

    xi = [0, 1, 2, 2]
    grad_analytic_x = x.grad[*xi]
    orig_x = x.data[*xi]
    x.data[*xi] = orig_x + epsilon
    loss_plus = recompute.call.data.to_f
    x.data[*xi] = orig_x - epsilon
    loss_minus = recompute.call.data.to_f
    x.data[*xi] = orig_x
    grad_numeric_x = (loss_plus - loss_minus) / (2 * epsilon)
    expect((grad_analytic_x - grad_numeric_x).abs).to be < tolerance

    wi = [1, 0, 1, 1]
    grad_analytic_w = w.grad[*wi]
    orig_w = w.data[*wi]
    w.data[*wi] = orig_w + epsilon
    loss_plus = recompute.call.data.to_f
    w.data[*wi] = orig_w - epsilon
    loss_minus = recompute.call.data.to_f
    w.data[*wi] = orig_w
    grad_numeric_w = (loss_plus - loss_minus) / (2 * epsilon)
    expect((grad_analytic_w - grad_numeric_w).abs).to be < tolerance
  end
end
