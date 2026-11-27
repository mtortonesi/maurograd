require "maurograd"

describe "Maurograd.no_grad" do
  def param(*values)
    Maurograd::Tensor.new(Numo::SFloat[*values], requires_grad: true)
  end

  it "builds no graph inside the block" do
    w = param(1, 2)
    y = Maurograd.no_grad { ((w * 3) + 1).exp.sum }

    expect(y.requires_grad).to be == false
    expect(y.creator).to be_nil
  end

  it "records again after the block, and returns the block's value" do
    w = param(1, 2)
    expect(Maurograd.no_grad { 42 }).to be == 42
    expect(Maurograd.grad_enabled?).to be == true

    (w * w).sum.backward
    expect(w.grad.to_a).to be == [2.0, 4.0]
  end

  it "restores recording when the block raises" do
    expect { Maurograd.no_grad { raise "boom" } }.to raise_exception(RuntimeError)
    expect(Maurograd.grad_enabled?).to be == true
  end

  it "nests" do
    Maurograd.no_grad do
      Maurograd.no_grad { nil }
      expect(Maurograd.grad_enabled?).to be == false
    end
    expect(Maurograd.grad_enabled?).to be == true
  end

  it "leaves parameters trainable and creatable" do
    w = param(1)
    fresh = Maurograd.no_grad { param(2) }
    expect(w.requires_grad).to be == true
    expect(fresh.requires_grad).to be == true
  end

  it "covers every kind of op: layers, conv, pooling, BatchNorm, losses" do
    Maurograd.seed!(1)
    model = Maurograd::Layers::Sequential.new(
      Maurograd::Layers::Conv2D.new(1, 2, 3, padding: 1), Maurograd::Layers::BatchNorm2D.new(2),
      Maurograd::Layers::ReLU.new, Maurograd::Layers::MaxPool2D.new(2), Maurograd::Layers::Flatten.new,
      Maurograd::Layers::Linear.new(8, 3), Maurograd::Layers::SiLU.new
    )
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 1, 4, 4).rand)
    target = Maurograd::Tensor.new(Numo::Int32[0, 2])

    loss = Maurograd.no_grad do
      Maurograd::Losses::CrossEntropyWithLogits.apply(model.call(x), target)
    end
    expect(loss.requires_grad).to be == false
    expect(loss.creator).to be_nil

    # The same computation with recording on, for comparison.
    expect(Maurograd::Losses::CrossEntropyWithLogits.apply(model.call(x), target).creator).not.to be_nil
  end

  it "is per thread" do
    Maurograd.no_grad do
      other = Thread.new { Maurograd.grad_enabled? }.value
      expect(other).to be == true
    end
  end

  it "applies inside fibers and enumerators started in the block" do
    Maurograd.no_grad do
      expect(Fiber.new { Maurograd.grad_enabled? }.resume).to be == false
    end
  end

  it "needs a block" do
    expect { Maurograd.no_grad }.to raise_exception(ArgumentError)
  end
end
