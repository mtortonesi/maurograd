require "maurograd/tensor"
require "maurograd/optim/adam"

describe Maurograd::Optim::Adam do
  def param(*values)
    Maurograd::Tensor.new(Numo::SFloat[*values], requires_grad: true)
  end

  it "moves every weight by ~lr on the first step, whatever the gradient's size" do
    # Bias correction makes m_hat = g and v_hat = g^2 on step 1, so the
    # update is lr * g / |g| = lr * sign(g).
    w = param(0.0, 0.0, 0.0)
    opt = Maurograd::Optim::Adam.new([w], lr: 0.01)

    w.grad = Numo::SFloat[100.0, 0.01, -3.0]
    opt.step

    expected = Numo::SFloat[-0.01, -0.01, 0.01]
    expect((w.data - expected).abs.max).to be < 1e-5
  end

  it "matches the bias-corrected update rule over several steps" do
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    grads = [1.0, 3.0, -0.5]

    w = param(0.5)
    opt = Maurograd::Optim::Adam.new([w], lr: lr, betas: [beta1, beta2], eps: eps)

    # Reference: the same recurrence on plain Ruby Floats.
    expected_w = 0.5
    m = 0.0
    v = 0.0
    grads.each_with_index do |g, idx|
      t = idx + 1
      m = beta1 * m + (1 - beta1) * g
      v = beta2 * v + (1 - beta2) * g * g
      m_hat = m / (1 - beta1**t)
      v_hat = v / (1 - beta2**t)
      expected_w -= lr * m_hat / (Math.sqrt(v_hat) + eps)

      w.grad = Numo::SFloat[g]
      opt.step
      expect((w.data[0] - expected_w).abs).to be < 1e-5
    end

    expect(opt.t).to be == 3
  end

  it "adds weight_decay * param to the gradient" do
    # With weight decay the effective gradient is 1.0 + 1.0 * 2.0 = 3.0; on
    # step 1 only its sign matters, so the result equals a plain step on 3.0.
    w = param(2.0)
    opt = Maurograd::Optim::Adam.new([w], lr: 0.1, weight_decay: 1.0)

    w.grad = Numo::SFloat[1.0]
    opt.step
    expect((w.data[0] - 1.9).abs).to be < 1e-5

    # Second step, gradient -1.0 -> effective -1.0 + 1.9 = 0.9; compare with
    # an Adam fed those effective gradients directly.
    ref = param(2.0)
    ref_opt = Maurograd::Optim::Adam.new([ref], lr: 0.1)
    ref.grad = Numo::SFloat[3.0]
    ref_opt.step
    ref.grad = Numo::SFloat[0.9]
    ref_opt.step

    w.grad = Numo::SFloat[-1.0]
    opt.step
    expect((w.data[0] - ref.data[0]).abs).to be < 1e-5
  end

  it "skips params with nil grad" do
    w = param(2.0)
    opt = Maurograd::Optim::Adam.new([w])
    opt.step
    expect(w.data).to be == Numo::SFloat[2.0]
  end

  it "can zero gradients" do
    w = param(2.0)
    w.grad = Numo::SFloat[5.0]
    Maurograd::Optim::Adam.new([w]).zero_grad
    expect(w.grad).to be == Numo::SFloat[0.0]
  end

  it "rejects invalid hyperparameters" do
    expect { Maurograd::Optim::Adam.new([], betas: [1.0, 0.999]) }.to raise_exception(ArgumentError)
    expect { Maurograd::Optim::Adam.new([], betas: [0.9, -0.1]) }.to raise_exception(ArgumentError)
    expect { Maurograd::Optim::Adam.new([], weight_decay: -1.0) }.to raise_exception(ArgumentError)
  end

  it "minimizes a quadratic through autograd" do
    target = Maurograd::Tensor.new(Numo::SFloat[1.0, -2.0, 3.0])
    w = param(0.0, 0.0, 0.0)
    opt = Maurograd::Optim::Adam.new([w], lr: 0.1)

    500.times do
      opt.zero_grad
      ((w - target)**2).sum.backward
      opt.step
    end

    expect((w.data - target.data).abs.max).to be < 1e-2
  end
end
