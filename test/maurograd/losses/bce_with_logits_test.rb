require "maurograd"

describe Maurograd::Losses::BCEWithLogits do
  def tensor(values, requires_grad: false)
    Maurograd::Tensor.new(Numo::SFloat.cast(values), requires_grad: requires_grad)
  end

  # The textbook formula, in double precision.
  def reference(z, y)
    p = 1.0 / (Numo::NMath.exp(-z) + 1.0)
    -(y * Numo::NMath.log(p) + (1.0 - y) * Numo::NMath.log(1.0 - p))
  end

  it "matches the binary cross-entropy of sigmoid(logits), with mean and sum" do
    Maurograd.seed!(1)
    z = Numo::DFloat.new(3, 4).rand_norm * 3
    y = Numo::DFloat.new(3, 4).rand

    mean = subject.apply(tensor(z), tensor(y))
    sum = subject.apply(tensor(z), tensor(y), reduction: :sum)
    expect((mean.item - reference(z, y).mean).abs).to be < 1e-5
    expect((sum.item - reference(z, y).sum).abs).to be < 1e-4
  end

  it "has gradients sigmoid(z) - y and -z, scaled by the reduction" do
    Maurograd.seed!(2)
    z = Numo::DFloat.new(2, 5).rand_norm
    y = Numo::DFloat.new(2, 5).rand
    logits = tensor(z, requires_grad: true)
    target = tensor(y, requires_grad: true)
    subject.apply(logits, target).backward

    sigmoid = 1.0 / (Numo::NMath.exp(-z) + 1.0)
    expect((Numo::DFloat.cast(logits.grad) - (sigmoid - y) / 10).abs.max).to be < 1e-6
    expect((Numo::DFloat.cast(target.grad) + z / 10).abs.max).to be < 1e-6

    logits.zero_grad
    subject.apply(logits, tensor(y), reduction: :sum).backward
    expect((Numo::DFloat.cast(logits.grad) - (sigmoid - y)).abs.max).to be < 1e-6
  end

  it "stays finite and keeps its gradient for confident wrong predictions" do
    logits = tensor([200, -200, 30], requires_grad: true)
    loss = subject.apply(logits, tensor([0, 1, 1]), reduction: :sum)
    expect((loss.item - 400.0).abs).to be < 1e-3
    loss.backward
    expect((logits.grad - Numo::SFloat[1, -1, 0]).abs.max).to be < 1e-6
  end

  it "trains a logistic regression" do
    Maurograd.seed!(3)
    x = Numo::SFloat.new(64, 2).rand_norm
    y = Numo::SFloat.cast((x[true, 0] + x[true, 1] * 2).gt(0)).reshape(64, 1)
    layer = Maurograd::Layers::Linear.new(2, 1)
    opt = Maurograd::Optim::Adam.new(layer.parameters, lr: 0.1)
    losses = 100.times.map do
      opt.zero_grad
      loss = subject.apply(layer.call(tensor(x)), tensor(y))
      loss.backward
      opt.step
      loss.item
    end
    expect(losses.last).to be < losses.first / 4
  end

  it "checks shapes and the reduction" do
    expect { subject.apply(tensor([[1, 2]]), tensor([1, 2])) }.to raise_exception(ArgumentError, message: be =~ /same shape/)
    expect { subject.apply(tensor([1]), tensor([1]), reduction: :none) }.to raise_exception(ArgumentError)
  end
end
