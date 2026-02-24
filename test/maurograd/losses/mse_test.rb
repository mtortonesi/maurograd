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
end

