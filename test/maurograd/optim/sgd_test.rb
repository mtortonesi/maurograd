# test/maurograd/optim/sgd_test.rb
require "maurograd/tensor"
require "maurograd/optim/sgd"

describe Maurograd::Optim::SGD do
  it "updates parameters using gradients and can zero them" do
    w = Maurograd::Tensor.new(Numo::SFloat[2.0], requires_grad: true)

    # loss = sum(w^2) => grad = 2w = 4
    loss = (w**2).sum
    loss.backward

    expect(w.grad).to be == Numo::SFloat[4.0]

    opt = Maurograd::Optim::SGD.new([w], lr: 0.1)
    opt.step

    # w_new = 2.0 - 0.1 * 4.0 = 1.6
    expect(w.data).to be == Numo::SFloat[1.6]

    opt.zero_grad
    expect(w.grad).to be == Numo::SFloat[0.0]
  end

  it "skips params with nil grad" do
    w = Maurograd::Tensor.new(Numo::SFloat[2.0], requires_grad: true)
    expect(w.grad).to be_nil

    opt = Maurograd::Optim::SGD.new([w], lr: 0.1)
    opt.step

    expect(w.data).to be == Numo::SFloat[2.0]
  end
end
