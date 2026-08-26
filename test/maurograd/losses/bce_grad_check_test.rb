require 'maurograd/tensor'
require 'maurograd/losses/bce'

describe "BCE gradient check" do
  it "matches numerical gradient on one pred element" do
    eps = 1e-8
    h = 1e-4
    tol = 1e-5

    # pred in (0,1)
    p_data = Numo::SFloat[[0.2, 0.7, 0.4],
                          [0.9, 0.1, 0.6]]
    y_data = Numo::SFloat[[0.0, 1.0, 1.0],
                          [1.0, 0.0, 0.0]]

    pred = Maurograd::Tensor.new(p_data.copy, requires_grad: true)
    y    = Maurograd::Tensor.new(y_data, requires_grad: false)

    loss = Maurograd::Losses::BCE.apply(pred, y, eps: eps)
    loss.backward

    # pick one element (i=0,j=1)
    i = 0
    j = 1
    grad_analytic = pred.grad[i, j]

    # numerical gradient: (L(p+h)-L(p-h))/(2h)
    orig = pred.data[i, j]

    pred.data[i, j] = orig + h
    loss_plus = Maurograd::Losses::BCE.apply(pred, y, eps: eps).data

    pred.data[i, j] = orig - h
    loss_minus = Maurograd::Losses::BCE.apply(pred, y, eps: eps).data

    pred.data[i, j] = orig # restore

    grad_numeric = (loss_plus - loss_minus) / (2.0 * h)

    diff = (grad_analytic - grad_numeric).abs
    expect(diff).to be < tol
  end

  it "raises when pred and target shapes differ" do
    # pred is [N, C] but target is [1, C]: BCE requires exact shape match,
    # it does not broadcast target over the batch.
    p = Maurograd::Tensor.new(Numo::SFloat[[0.2, 0.7],
                                          [0.9, 0.1]], requires_grad: true)
    y = Maurograd::Tensor.new(Numo::SFloat[[0.0, 1.0]], requires_grad: true)

    expect { Maurograd::Losses::BCE.apply(p, y) }.to raise_exception
  end
end

