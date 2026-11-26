require "maurograd/tensor"
require "maurograd/optim/sgd"

describe "Maurograd::Optim::SGD with momentum and weight decay" do
  def param(*values)
    Maurograd::Tensor.new(Numo::SFloat[*values], requires_grad: true)
  end

  it "accumulates a velocity across steps" do
    w = param(2.0)
    opt = Maurograd::Optim::SGD.new([w], lr: 0.1, momentum: 0.9)

    # Step 1: v = g = 1.0 (identical to plain SGD) -> w = 2.0 - 0.1 * 1.0 = 1.9
    w.grad = Numo::SFloat[1.0]
    opt.step
    expect((w.data[0] - 1.9).abs).to be < 1e-6

    # Step 2: v = 0.9 * 1.0 + 1.0 = 1.9 -> w = 1.9 - 0.1 * 1.9 = 1.71
    w.grad = Numo::SFloat[1.0]
    opt.step
    expect((w.data[0] - 1.71).abs).to be < 1e-6
  end

  it "keeps a separate velocity per parameter" do
    a = param(0.0)
    b = param(0.0)
    opt = Maurograd::Optim::SGD.new([a, b], lr: 1.0, momentum: 0.5)

    a.grad = Numo::SFloat[1.0]
    b.grad = Numo::SFloat[-2.0]
    opt.step
    a.grad = Numo::SFloat[1.0]
    b.grad = Numo::SFloat[-2.0]
    opt.step

    # a: v1 = 1, v2 = 1.5 -> a = -2.5 ; b: v1 = -2, v2 = -3 -> b = 5
    expect((a.data[0] - -2.5).abs).to be < 1e-6
    expect((b.data[0] - 5.0).abs).to be < 1e-6
  end

  it "adds weight_decay * param to the gradient" do
    w = param(2.0)
    opt = Maurograd::Optim::SGD.new([w], lr: 0.1, weight_decay: 0.5)

    # g = 1.0 + 0.5 * 2.0 = 2.0 -> w = 2.0 - 0.1 * 2.0 = 1.8
    w.grad = Numo::SFloat[1.0]
    opt.step
    expect((w.data[0] - 1.8).abs).to be < 1e-6
  end

  it "does not modify the parameter's gradient" do
    w = param(2.0)
    opt = Maurograd::Optim::SGD.new([w], lr: 0.1, momentum: 0.9, weight_decay: 0.5)

    w.grad = Numo::SFloat[1.0]
    opt.step
    expect(w.grad).to be == Numo::SFloat[1.0]
  end

  it "skips params with nil grad without allocating state for them" do
    w = param(2.0)
    opt = Maurograd::Optim::SGD.new([w], lr: 0.1, momentum: 0.9)
    opt.step
    expect(w.data).to be == Numo::SFloat[2.0]

    # First real step must still behave like a first step (v = g).
    w.grad = Numo::SFloat[1.0]
    opt.step
    expect((w.data[0] - 1.9).abs).to be < 1e-6
  end

  it "rejects negative hyperparameters" do
    expect { Maurograd::Optim::SGD.new([], momentum: -0.1) }.to raise_exception(ArgumentError)
    expect { Maurograd::Optim::SGD.new([], weight_decay: -0.1) }.to raise_exception(ArgumentError)
  end

  it "minimizes a quadratic through autograd" do
    target = Maurograd::Tensor.new(Numo::SFloat[1.0, -2.0, 3.0])
    w = param(0.0, 0.0, 0.0)
    opt = Maurograd::Optim::SGD.new([w], lr: 0.05, momentum: 0.9)

    200.times do
      opt.zero_grad
      ((w - target)**2).sum.backward
      opt.step
    end

    expect((w.data - target.data).abs.max).to be < 1e-3
  end
end
