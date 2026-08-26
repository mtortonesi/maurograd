# test/maurograd/losses/mse_test.rb
require 'maurograd/tensor'
require 'maurograd/losses/mse'

describe Maurograd::Losses::MSE do
  it "computes correct scalar MSE and correct gradient w.r.t. pred" do
    pred = Maurograd::Tensor.new(Numo::SFloat[1.0, 2.0, 3.0], requires_grad: true)
    target = Maurograd::Tensor.new(Numo::SFloat[2.0, 2.0, 2.0], requires_grad: false)

    loss = Maurograd::Losses::MSE.apply(pred, target)

    # diff = [-1, 0, 1]
    # mse  = mean(diff^2) = (1 + 0 + 1)/3 = 2/3
    expect(loss.data).to be == (2.0 / 3.0)

    loss.backward

    # d/dpred = (2/N) * diff = (2/3) * [-1, 0, 1] = [-2/3, 0, 2/3]
    expected_grad = Numo::SFloat[-2.0/3.0, 0.0, 2.0/3.0]
    expect(pred.grad).to be == expected_grad
  end

  it "matches numerical gradient on one pred element" do
    h = 1e-4
    tol = 1e-5

    p_data = Numo::SFloat[[0.2, 0.7, 0.4],
                          [0.9, 0.1, 0.6]]
    y_data = Numo::SFloat[[0.0, 1.0, 1.0],
                          [1.0, 0.0, 0.0]]

    pred = Maurograd::Tensor.new(p_data.copy, requires_grad: true)
    y    = Maurograd::Tensor.new(y_data, requires_grad: false)

    loss = Maurograd::Losses::MSE.apply(pred, y)
    loss.backward

    i = 0
    j = 1
    grad_analytic = pred.grad[i, j]

    orig = pred.data[i, j]

    pred.data[i, j] = orig + h
    loss_plus = Maurograd::Losses::MSE.apply(pred, y).data

    pred.data[i, j] = orig - h
    loss_minus = Maurograd::Losses::MSE.apply(pred, y).data

    pred.data[i, j] = orig # restore

    grad_numeric = (loss_plus - loss_minus) / (2.0 * h)

    diff = (grad_analytic - grad_numeric).abs
    expect(diff).to be < tol
  end

  it "raises when pred and target shapes differ" do
    pred = Maurograd::Tensor.new(Numo::SFloat[[0.2, 0.7],
                                              [0.9, 0.1]], requires_grad: true)
    target = Maurograd::Tensor.new(Numo::SFloat[[0.0, 1.0]], requires_grad: true)

    expect { Maurograd::Losses::MSE.apply(pred, target) }.to raise_exception
  end
end

